"""Metrics for yes/no outcomes, the metrics added on request, and recording your own metrics in a report."""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from timegym import (
    Forecast,
    ForecastRun,
    GaussianAR1Forecaster,
    LastValueForecaster,
    MeanForecaster,
    Pipeline,
    Scenario,
    cases_from_series,
)
from timegym import metrics
from timegym.diagnostics import _METRIC_NAMES, evaluate_forecast
from timegym.metrics import (
    UndefinedMetricError,
    brier,
    interval_score,
    log_loss,
    mse,
    pinball,
    rmse,
    rmsse,
    weighted_interval_score,
)


def rain_days(periods=200, seed=0):
    """Whether it rained each day: 1 or 0, wetter in the second half."""
    rng = np.random.default_rng(seed)
    chance = np.where(np.arange(periods) < periods // 2, 0.3, 0.6)
    days = pd.date_range("2024-01-01", periods=periods, freq="D")
    return pd.Series((rng.random(periods) < chance).astype(float), index=days, name="rain")


def within_10_percent(observed, predicted):
    return float(np.mean(np.abs(predicted - observed) <= 0.1 * np.abs(observed)))


# ---------------------------------------------------------------- yes/no scores

def test_brier_is_mse_on_yes_no_outcomes():
    observed = np.array([[1.0], [0.0], [1.0], [0.0]])
    predicted = np.array([[0.9], [0.2], [0.4], [0.0]])
    assert brier(observed, predicted) == pytest.approx(mse(observed, predicted))
    assert brier(observed, predicted) == pytest.approx((0.01 + 0.04 + 0.36 + 0.0) / 4)


def test_log_loss_is_minus_the_log_of_the_probability_given_to_what_happened():
    observed = np.array([1.0, 0.0, 1.0, 0.0])
    predicted = np.array([0.9, 0.2, 0.4, 0.0])  # an exact 0 that was right costs nothing
    expected = -(np.log(0.9) + np.log(0.8) + np.log(0.4) + np.log(1.0)) / 4
    assert log_loss(observed, predicted) == pytest.approx(expected)


def test_a_confident_mistake_needs_eps():
    observed, predicted = np.array([1.0, 0.0]), np.array([0.7, 1.0])
    with pytest.raises(UndefinedMetricError, match=r"exactly 1 at position \(1,\) turned out wrong"):
        log_loss(observed, predicted)
    clipped = log_loss(observed, predicted, eps=1e-6)
    assert clipped == pytest.approx(-(np.log(0.7) + np.log(1e-6)) / 2)
    for eps in (0, 0.5, -1e-3, True, "small", float("nan")):
        with pytest.raises(ValueError, match="eps must be a number between 0 and 0.5"):
            log_loss(observed, predicted, eps=eps)


@pytest.mark.parametrize("score", [brier, log_loss])
def test_outcomes_and_probabilities_are_checked(score):
    with pytest.raises(ValueError, match=r"observed must be 0 or 1; got 0.5 at position \(1, 0\)"):
        score(np.array([[1.0], [0.5]]), np.array([[0.5], [0.5]]))
    with pytest.raises(ValueError, match=r"predicted must be between 0 and 1; got 1.2 at position \(2,\)"):
        score(np.array([1.0, 0.0, 1.0]), np.array([0.5, 0.5, 1.2]))
    with pytest.raises(ValueError, match="identical shapes"):
        score(np.array([1.0, 0.0]), np.array([0.5]))


@pytest.mark.parametrize("score", [brier, log_loss])
def test_honest_probabilities_score_best(score):
    truth = 0.3  # rain on 30% of such days
    grid = np.linspace(0.05, 0.95, 19)
    expected = [truth * score(np.array([1.0]), np.array([p])) + (1 - truth) * score(np.array([0.0]), np.array([p]))
                for p in grid]
    assert grid[int(np.argmin(expected))] == pytest.approx(truth)


# ------------------------------------------------------------ point scores

def test_rmse_is_the_root_of_mse():
    observed, predicted = np.array([[1.0, 2.0], [3.0, 5.0]]), np.array([[2.0, 2.0], [3.0, 1.0]])
    assert rmse(observed, predicted) == pytest.approx(np.sqrt(mse(observed, predicted)))


def test_rmsse_scales_each_series_by_its_history():
    history = np.array([[0.0, 10.0], [1.0, 10.0], [3.0, 14.0], [2.0, 12.0]])
    observed = np.array([[2.0, 12.0], [4.0, 13.0]])
    predicted = np.array([[3.0, 12.0], [2.0, 10.0]])
    scale = np.array([(1 + 4 + 1) / 3, (0 + 16 + 4) / 3])  # mean squared one-step change
    per_series = np.sqrt(np.array([(1 + 4) / 2, (0 + 9) / 2]) / scale)
    assert rmsse(observed, predicted, history) == pytest.approx(per_series.mean())
    seasonal = np.array([[1.0], [2.0], [4.0], [3.0], [5.0]])
    assert rmsse(np.array([[6.0]]), np.array([[5.0]]), seasonal, seasonality=2) == pytest.approx(
        1 / np.sqrt((9 + 1 + 1) / 3))


def test_rmsse_is_undefined_without_change_in_the_history():
    with pytest.raises(UndefinedMetricError, match="zero naive-error"):
        rmsse(np.array([[1.0]]), np.array([[2.0]]), np.array([[3.0], [3.0], [3.0]]))
    with pytest.raises(UndefinedMetricError, match="more observations of history"):
        rmsse(np.array([[1.0]]), np.array([[2.0]]), np.array([[3.0], [4.0]]), seasonality=2)


# ---------------------------------------------------------- band scores

def explicit_interval_score(y, lower, upper, alpha):
    return (upper - lower) + 2 / alpha * (lower - y) * (y < lower) + 2 / alpha * (y - upper) * (y > upper)


def test_interval_score_is_width_plus_the_penalty_for_misses():
    observed = np.array([[0.0], [5.0], [-4.0]])
    levels = (0.1, 0.5, 0.9)
    quantiles = np.array([[[-1.0, 0.0, 1.0]], [[-1.0, 0.0, 1.0]], [[-2.0, 0.0, 2.0]]])
    expected = np.mean([explicit_interval_score(0.0, -1, 1, 0.2), explicit_interval_score(5.0, -1, 1, 0.2),
                        explicit_interval_score(-4.0, -2, 2, 0.2)])
    assert interval_score(observed, quantiles, levels) == pytest.approx(expected)
    assert expected == pytest.approx((2 + (2 + 40) + (4 + 20)) / 3)


def test_weighted_interval_score_follows_its_definition():
    rng = np.random.default_rng(4)
    levels = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
    observed = rng.normal(size=(5, 2))
    quantiles = np.sort(rng.normal(size=(5, 2, len(levels))), axis=-1)
    pairs = [(0, 6), (1, 5), (2, 4)]
    total = 0.5 * np.abs(observed - quantiles[..., 3])
    for low, high in pairs:
        alpha = 2 * levels[low]
        total = total + alpha / 2 * explicit_interval_score(observed, quantiles[..., low], quantiles[..., high], alpha)
    expected = np.mean(total / (len(pairs) + 0.5))
    assert weighted_interval_score(observed, quantiles, levels) == pytest.approx(expected)
    assert weighted_interval_score(observed, quantiles, levels) == pytest.approx(2 * pinball(observed, quantiles, levels))


def test_band_scores_refuse_levels_they_cannot_read():
    observed, quantiles = np.zeros((2, 1)), np.tile([-1.0, 0.0, 2.0], (2, 1, 1))
    with pytest.raises(ValueError, match="lowest and highest levels must be symmetric"):
        interval_score(observed, quantiles, (0.1, 0.5, 0.8))
    with pytest.raises(ValueError, match="lowest and highest levels must be symmetric"):
        interval_score(observed[..., :1], quantiles[..., :1], (0.5,))
    with pytest.raises(ValueError, match="0.5 plus pairs symmetric"):
        weighted_interval_score(observed, quantiles, (0.1, 0.4, 0.9))
    with pytest.raises(ValueError, match="0.5 plus pairs symmetric"):
        weighted_interval_score(observed, quantiles[..., [0, 2]], (0.1, 0.9))
    with pytest.raises(ValueError, match="must not cross"):
        interval_score(observed, quantiles[..., ::-1], (0.1, 0.5, 0.9))


def test_every_metric_function_can_be_asked_for_by_name():
    public = {name for name in metrics.__all__ if callable(getattr(metrics, name))
              and not isinstance(getattr(metrics, name), type)} - {"empirical_crps"}
    assert set(_METRIC_NAMES) == public


# ------------------------------------------------------- metrics= in a report

def yes_no_cases():
    return cases_from_series(rain_days(), origins=[100, 130, 160], horizon=7, name="rain")


def test_a_report_records_the_metrics_asked_for():
    cases = yes_no_cases()
    report = Pipeline(MeanForecaster()).test(
        cases=cases, metrics=["brier", "log_loss", within_10_percent])
    for case in report.cases:
        assert {"brier", "log_loss", "within_10_percent"} <= set(case.metrics)
        assert case.metrics["brier"] == pytest.approx(case.metrics["mse"])
    assert [record["name"] for record in report.metadata["metrics"]] == ["brier", "log_loss", "within_10_percent"]
    assert [record["replayable"] for record in report.metadata["metrics"]] == [True, True, False]
    assert report.metadata["metrics"][0]["function"] == "timegym.metrics.brier"
    assert report.metadata["metrics"][2]["function"].endswith("within_10_percent")
    report.assert_passed()


def test_no_request_leaves_reports_as_they_were():
    report = Pipeline(LastValueForecaster()).test([Scenario("ar1", history_length=40, horizon=4)], seeds=[1])
    assert "metrics" not in report.metadata and "scores" not in report.metadata
    assert set(report.cases[0].metrics) == {"mse", "mae", "mase", "avoidable_mse"}


def test_settings_go_through_a_dictionary():
    cases = yes_no_cases()
    report = Pipeline(LastValueForecaster()).test(
        cases=cases, metrics={"log_loss": "log_loss", "log_loss_clipped": partial(log_loss, eps=1e-6)})
    for case in report.cases:  # repeating the last day forecasts exactly 0 or 1
        assert "log_loss_clipped" in case.metrics
        if "log_loss" not in case.metrics:
            (problem,) = [d for d in case.diagnostics if d.name == "metric_log_loss"]
            assert problem.status == "INCONCLUSIVE" and "pass eps" in problem.message
    assert any("log_loss" not in case.metrics for case in report.cases)
    clipped = report.metadata["metrics"][1]
    assert clipped["settings"] == {"eps": 1e-6} and clipped["replayable"] is True


def test_a_metric_gets_what_its_argument_names_ask_for():
    seen = {}

    def spread(samples, observed):
        seen["shapes"] = samples.shape, observed.shape
        return float(np.std(samples))

    def history_length(history, predicted, quantiles=None):
        seen["quantiles"] = quantiles
        return float(len(history))

    case = Scenario("ar1", history_length=40, horizon=4).generate(2)
    forecast = Pipeline(GaussianAR1Forecaster(n_samples=32, seed=0)).predict(case.task.history, case.task.future_index)
    result = evaluate_forecast(case, forecast, metrics=[spread, history_length])
    assert seen["shapes"] == ((32, 4, 1), (4, 1))
    assert seen["quantiles"].shape == (4, 1, 3)
    assert result.metrics["history_length"] == 40.0
    point_only = Forecast(case.task.future_index, point=forecast.point)
    result = evaluate_forecast(case, point_only, metrics=[spread, history_length])
    assert seen["quantiles"] is None  # the forecast has no quantiles, so the default stands
    (missing,) = [d for d in result.diagnostics if d.name == "metric_spread"]
    assert (missing.status, missing.category) == ("UNSUPPORTED", "capability")
    assert "needs samples" in missing.message and missing.evidence == {"needs": ["samples", "observed"]}
    assert result.metrics["history_length"] == 40.0


def test_a_broken_metric_is_an_error_with_its_message():
    def broken(observed, predicted):
        raise RuntimeError("divided by the wrong thing")

    def several(observed, predicted):
        return np.abs(observed - predicted)

    report = Pipeline(LastValueForecaster()).test([Scenario("ar1", history_length=40, horizon=4)], seeds=[1],
                             metrics=[broken, several, "brier"])
    problems = {d.name: d for d in report.cases[0].diagnostics if d.name.startswith("metric_")}
    assert problems["metric_broken"].status == "ERROR"
    assert problems["metric_broken"].message == "Score calculation raised RuntimeError: divided by the wrong thing"
    assert "must return one number" in problems["metric_several"].message
    assert "observed must be 0 or 1" in problems["metric_brier"].message
    with pytest.raises(AssertionError, match="metric_broken"):
        report.assert_passed()


@pytest.mark.parametrize("asked, error, message", [
    ("brier", TypeError, "a list or a dictionary"),
    (["mape"], ValueError, "timegym.metrics has no metric 'mape'"),
    (["mse"], ValueError, "'mse' is recorded in every report already"),
    ({"mase": partial(metrics.mase, seasonality=12)}, ValueError, "recorded in every report already"),
    (["brier", "brier"], ValueError, "asked for twice"),
    ([lambda observed, predicted: 0.0], ValueError, "no name of its own"),
    ([partial(log_loss, eps=1e-6)], ValueError, "no name of its own"),
    ({"weighted": lambda observed, predicted, weights: 0.0}, TypeError, "needs 'weights', which TimeGym cannot give"),
    ([3], TypeError, "a name from timegym.metrics or a function"),
])
def test_requests_that_cannot_work_are_refused_before_anything_runs(asked, error, message):
    class Unreachable:
        def fit(self, history):
            raise AssertionError("no case should run")

    with pytest.raises(error, match=message):
        Pipeline(Unreachable()).run(cases=yes_no_cases(), metrics=asked)
    with pytest.raises(error, match=message):
        ForecastRun.from_forecasts(yes_no_cases(), [None] * 3, metrics=asked)


def test_forecasts_made_earlier_get_the_metrics_too():
    cases = yes_no_cases()
    recorded = ForecastRun.from_forecasts(cases, [np.full(7, 0.4)] * 3, metrics=["brier"])
    assert all(entry.result.metrics["brier"] > 0 for entry in recorded.cases)
    assert recorded.metadata["metrics"][0]["name"] == "brier"


def test_replay_computes_named_metrics_again_and_lists_the_others():
    pipeline = Pipeline(GaussianAR1Forecaster(n_samples=32, seed=3))
    report = pipeline.test([Scenario("ar1", history_length=40, horizon=4)], seeds=[5],
                           metrics={"rmse": "rmse", "wis": "weighted_interval_score", "close": within_10_percent})
    assert {"rmse", "wis", "close"} <= set(report.cases[0].metrics)
    replay = report.replay_case(0).metadata["replay"]
    assert replay["reproduced"] and replay["not_recomputed"] == ["close"]
    # The same holds after a round trip through JSON.
    from timegym import TestReport
    again = TestReport.from_json(report.to_json()).replay_case(0).metadata["replay"]
    assert again["reproduced"] and again["not_recomputed"] == ["close"]
    # A metric replay can compute is compared: a changed value is caught.
    report.cases[0].metrics["rmse"] += 1.0
    assert not report.replay_case(0).metadata["replay"]["metrics_match"]
