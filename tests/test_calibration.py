"""Calibration of yes/no forecasts: the table, the slope and its interval, the intercept and the plot."""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

from timegym import Forecast, NaiveForecaster, Pipeline, cases_from_series, diagnostics
from timegym.diagnostics import Calibration, calibration
from timegym.plotting import plot_calibration


def honest(n=4000, power=1.0, seed=0):
    """Outcomes drawn from true chances, and forecasts whose log-odds are ``power`` times the truth's."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.05, 0.95, n)
    outcomes = (rng.random(n) < truth).astype(float)
    odds = (truth / (1 - truth)) ** power
    return outcomes, odds / (1 + odds)


def log_odds(p):
    return np.log(p) - np.log1p(-p)


def test_table_counts_each_forecast_once_in_bins_of_equal_width():
    observed = np.array([0, 1, 1, 0, 1, 0, 1, 1.0])
    predicted = np.array([0.0, 0.05, 0.3, 0.35, 0.7, 0.75, 0.999, 1.0])
    cal = calibration(observed, predicted, bins=10)
    table = cal.table
    assert list(table.columns) == ["low", "high", "forecasts", "mean_forecast", "happened"]
    assert np.allclose(table["low"], np.arange(10) / 10) and np.allclose(table["high"], np.arange(1, 11) / 10)
    assert table["forecasts"].sum() == 8
    assert table.loc[0, "forecasts"] == 2 and table.loc[0, "happened"] == 0.5
    assert table.loc[3, "forecasts"] == 2  # 0.3 opens the bin from 0.3 to 0.4
    assert math.isclose(table.loc[3, "mean_forecast"], 0.325)
    assert table.loc[9, "forecasts"] == 2 and table.loc[9, "happened"] == 1  # 1.0 falls in the last bin
    assert table.loc[1, "forecasts"] == 0 and np.isnan(table.loc[1, "happened"])
    assert cal.certain == 2  # the forecasts of exactly 0 and 1


@pytest.mark.parametrize("power, expected", [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0)])
def test_slope_finds_honest_too_confident_and_too_timid_forecasts(power, expected):
    observed, predicted = honest(power=power)
    cal = calibration(observed, predicted)
    low, high = cal.slope_interval
    assert low < expected < high
    assert abs(cal.slope - expected) < 0.15 * expected
    assert abs(cal.intercept) < 0.1


def test_slope_and_intercept_are_the_maximum_likelihood_fits():
    sklearn = pytest.importorskip("sklearn.linear_model")
    observed, predicted = honest(n=1500, power=1.4, seed=3)
    predicted = np.clip(predicted + 0.04, 0.01, 0.99)  # too high on average as well
    x = log_odds(predicted)
    fit = sklearn.LogisticRegression(C=np.inf, tol=1e-10, max_iter=10_000).fit(x[:, None], observed)
    cal = calibration(observed, predicted)
    assert math.isclose(cal.slope, fit.coef_[0, 0], rel_tol=1e-5)

    def minus_loglik(a):
        eta = a + x
        return -np.sum(observed * eta - np.logaddexp(0, eta))

    intercept = minimize_scalar(minus_loglik, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10}).x
    assert math.isclose(cal.intercept, intercept, abs_tol=1e-6)
    assert cal.intercept < 0  # forecasts too high


def test_interval_depends_on_seed_and_level_only_as_it_should():
    observed, predicted = honest(n=800)
    first, again = calibration(observed, predicted, seed=4), calibration(observed, predicted, seed=4)
    assert first.slope_interval == again.slope_interval
    other = calibration(observed, predicted, seed=5)
    assert other.slope == first.slope and other.slope_interval != first.slope_interval
    narrow = calibration(observed, predicted, level=0.5, seed=4)
    assert first.slope_interval[0] < narrow.slope_interval[0] < narrow.slope_interval[1] < first.slope_interval[1]


def test_clusters_resample_together_and_widen_the_interval_when_they_share_luck():
    rng = np.random.default_rng(7)
    clusters = np.repeat(np.arange(30), 100)
    truth = rng.uniform(0.1, 0.9, 3000)
    steepness = rng.normal(1.0, 0.5, 30)[clusters]  # how far a cluster's outcomes follow the forecast
    chance = 1 / (1 + np.exp(-steepness * log_odds(truth)))
    observed = (rng.random(3000) < chance).astype(float)
    alone = calibration(observed, truth)
    together = calibration(observed, truth, cluster=clusters)
    assert together.clusters == 30 and alone.clusters is None
    assert together.slope == alone.slope
    width = lambda cal: cal.slope_interval[1] - cal.slope_interval[0]
    assert width(together) > 1.5 * width(alone)


# --- From a run ----------------------------------------------------------------------------

class CycleShare:
    """The share of rainy steps so far at the same point of the 12-step cycle, with one of each added."""

    def fit(self, context):
        values = context.to_numpy()[:, 0]
        self.length_ = len(values)
        self.share_ = np.array([(values[k::12].sum() + 1) / (len(values[k::12]) + 2) for k in range(12)])
        return self

    def predict(self, future_index):
        return self.share_[np.arange(self.length_, self.length_ + len(future_index)) % 12]


@pytest.fixture(scope="module")
def rain_cases():
    rng = np.random.default_rng(5)
    chance = 0.5 + 0.37 * np.sin(2 * np.pi * np.arange(240) / 12)
    cases = []
    for station in range(6):
        rain = pd.Series((rng.random(240) < chance).astype(float), name="rain")
        cases += cases_from_series(rain, origins=[120, 168, 216], horizon=24, name="rain",
                                   manifest={"station": station})
    return cases


def test_a_run_gives_the_same_numbers_as_its_arrays(rain_cases):
    run = Pipeline(CycleShare()).run(cases=rain_cases)
    from_run = calibration(run)
    observed = np.concatenate([entry.case.observed.to_numpy()[:, 0] for entry in run.cases])
    predicted = np.concatenate([np.asarray(entry.forecast.point)[:, 0] for entry in run.cases])
    labels = np.repeat(np.arange(len(run.cases)), 24)
    from_arrays = calibration(observed, predicted, cluster=labels)
    assert from_run.cases == 18 and from_run.forecasts == 18 * 24 and from_run.skipped == 0
    assert from_run.slope == from_arrays.slope and from_run.intercept == from_arrays.intercept
    assert from_run.slope_interval == from_arrays.slope_interval  # each case resampled whole
    pd.testing.assert_frame_equal(from_run.table, from_arrays.table)


def test_cluster_names_a_manifest_key_of_the_run(rain_cases):
    run = Pipeline(CycleShare()).run(cases=rain_cases)
    cal = calibration(run, cluster="station")
    assert cal.clusters == 6 and cal.cluster == "station"
    assert "in 6 clusters by station" in str(cal)
    with pytest.raises(ValueError, match="has no manifest key 'region'"):
        calibration(run, cluster="region")


def test_samples_stand_in_for_a_missing_point_forecast(rain_cases):
    class SharePaths(CycleShare):
        def predict(self, future_index):
            share = super().predict(future_index)
            paths = (np.random.default_rng(0).random((400, len(share))) < share).astype(float)
            return Forecast(future_index, samples=paths[:, :, None], target_names=["rain"])

    cal = calibration(Pipeline(SharePaths()).run(cases=rain_cases))
    assert cal.forecasts == 18 * 24 and cal.skipped == 0
    assert 0.5 < cal.slope < 1.5


def test_cases_without_a_usable_forecast_are_left_out_and_counted(rain_cases):
    class SometimesFails(CycleShare):
        def fit(self, context):
            if len(context) == 168:
                raise RuntimeError("no forecast for this origin")
            return super().fit(context)

    cal = calibration(Pipeline(SometimesFails()).run(cases=rain_cases))
    assert cal.skipped == 6 and cal.cases == 12 and cal.forecasts == 12 * 24
    assert "(6 cases left out: no usable forecast)" in str(cal)

    class AlwaysFails(CycleShare):
        def fit(self, context):
            raise RuntimeError("never")

    with pytest.raises(ValueError, match="none of the run's 18 cases has a forecast to read"):
        calibration(Pipeline(AlwaysFails()).run(cases=rain_cases))


def test_a_run_of_counts_is_refused_naming_the_case():
    sales = pd.Series(np.arange(60.0) % 5)
    run = Pipeline(NaiveForecaster("mean")).run(cases=cases_from_series(sales, origins=[40], horizon=5, name="sales"))
    with pytest.raises(ValueError, match=r"case 0 \(sales\): observed must be 0 or 1; got 2"):
        calibration(run)


# --- Refusals and numbers that do not exist ---------------------------------------------------

@pytest.mark.parametrize("arguments, error, message", [
    (([0, 1, 2], [0.1, 0.5, 0.9]), ValueError, "observed must be 0 or 1"),
    (([0, 1, 1], [0.1, 0.5, 1.2]), ValueError, "predicted must be between 0 and 1"),
    (([0, 1], [0.1, 0.5, 0.9]), ValueError, "observed and predicted must have identical shapes"),
    (("not a run",), TypeError, "calibration takes a ForecastRun"),
])
def test_refuses_what_is_not_yes_no_forecasts(arguments, error, message):
    with pytest.raises(error, match=message):
        calibration(*arguments)


@pytest.mark.parametrize("keywords, error, message", [
    ({"bins": 1}, ValueError, "bins must be at least 2"),
    ({"bins": 2.5}, (TypeError, ValueError), "bins"),
    ({"level": 1}, ValueError, "level must be a number between 0 and 1"),
    ({"level": True}, ValueError, "level must be a number between 0 and 1"),
    ({"seed": -1}, (TypeError, ValueError), "seed"),
    ({"cluster": "station"}, TypeError, "with arrays, cluster gives one label per value"),
    ({"cluster": [1, 2]}, ValueError, "cluster must give one label per value: 4 values, 2 labels"),
])
def test_refuses_settings_that_cannot_work(keywords, error, message):
    with pytest.raises(error, match=message):
        calibration([0, 1, 1, 0], [0.2, 0.6, 0.7, 0.4], **keywords)


def test_says_why_there_is_no_slope():
    all_dry = calibration([0, 0, 0, 0], [0.2, 0.6, 0.7, 0.4])
    assert np.isnan(all_dry.slope) and np.isnan(all_dry.intercept)
    assert all_dry.note == "every outcome is 0, so there is no slope or intercept"
    same = calibration([0, 1, 1, 0, 1], [0.6] * 5)
    assert np.isnan(same.slope) and math.isclose(same.intercept, log_odds(0.6) - log_odds(0.6), abs_tol=1e-9)
    assert same.note == "every forecast is the same, so there is no slope"
    separated = calibration([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    assert np.isnan(separated.slope) and np.isnan(separated.slope_interval[0])
    assert "separate the outcomes perfectly" in separated.note
    assert "Every outcome is 0" in str(all_dry) and "nan" not in str(all_dry)


def test_an_interval_needs_most_resamples_to_have_a_slope():
    # Twelve values: many resamples separate the outcomes, so the interval is not given.
    cal = calibration([0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1], [.1, .2, .3, .35, .4, .5, .55, .6, .7, .8, .2, .9])
    assert np.isfinite(cal.slope)
    assert np.isnan(cal.slope_interval[0]) and "resamples, so it has no interval" in cal.note


def test_a_resample_has_no_slope_exactly_when_the_forecasts_separate_it():
    # Decided by comparing values, so every machine counts the same resamples. A fit drifting toward an
    # infinite slope used to stop at 75 to 350 on some machines and count as a slope.
    observed = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1], float)
    x = log_odds(np.array([.1, .2, .3, .35, .4, .5, .55, .6, .7, .8, .2, .9]))
    start = diagnostics._logistic_fit(x, observed, np.ones(12), free_slope=True)
    slopes = diagnostics._bootstrap_slopes(x, observed, np.arange(12), 12, start, np.random.default_rng(0))
    rng = np.random.default_rng(0)
    draws = np.concatenate([rng.integers(0, 12, (100, 12)) for _ in range(20)])  # the same draws
    for draw, slope in zip(draws, slopes):
        used = np.bincount(draw, minlength=12) > 0
        ones, zeros = x[used & (observed == 1)], x[used & (observed == 0)]
        apart = ones.size == 0 or zeros.size == 0 or zeros.max() <= ones.min() or ones.max() <= zeros.min()
        assert np.isnan(slope) == apart
    assert np.isnan(slopes).sum() == 113 and np.nanmax(np.abs(slopes)) < 10
    # A tie at the edge still separates; a value with weight 0 does not count.
    y = np.array([0, 0, 1, 1.0])
    assert diagnostics._separated(np.array([1, 3, 3, 4.0]), y, np.ones(4))[0]
    assert not diagnostics._separated(np.array([1, 3, 2, 4.0]), y, np.ones(4))[0]
    assert diagnostics._separated(np.array([1, 3, 2, 4.0]), y, np.array([1, 0, 1, 1.0]))[0]


def test_sure_forecasts_are_counted_and_pull_the_slope_down():
    observed, predicted = honest(n=3000, seed=2)
    sure = predicted.copy()
    sure[:60] = np.round(sure[:60])  # sixty forecasts of exactly 0 or 1, some of them wrong
    before, after = calibration(observed, predicted), calibration(observed, sure)
    assert after.certain == 60 and before.certain == 0
    assert after.slope < before.slope - 0.2 and before.slope > 0.85
    assert "60 forecasts say 0 or 1 exactly" in str(after)
    assert "say 0 or 1 exactly" not in str(before)


def test_printed_summary_and_repr():
    observed, predicted = honest(n=1000, power=2.0)
    cal = calibration(observed, predicted)
    text = str(cal).splitlines()
    assert text[0] == "Calibration of 1,000 forecasts"
    assert text[1].startswith(f"  slope {cal.slope:.2f} (95% interval ")
    assert "below 1 too confident" in text[1]
    assert text[2].startswith(f"  intercept {cal.intercept:+.2f}: 0 is right on average")
    assert text[3].split() == ["forecast", "mean", "happened", "forecasts"]
    assert len(text) == 4 + int((cal.table["forecasts"] > 0).sum())
    assert repr(cal).startswith(f"Calibration(slope={cal.slope:.3g}, slope_interval=(")
    assert isinstance(cal, Calibration)


def test_plot_draws_the_reliability_diagram():
    cal = calibration(*honest(n=1000))
    figure = cal.plot()
    top, bottom = figure.axes
    titles = [top.get_title(loc="left"), top.get_title(loc="right")]
    assert titles[0] == "How often it happened, by forecast"
    assert titles[1].startswith(f"slope {cal.slope:.2f} (95%: ") and titles[1].endswith("1,000 forecasts")
    assert len(bottom.patches) == 10
    plt.close(figure)
    no_slope = calibration([0, 1, 1, 0, 1], [0.6] * 5).plot()
    assert no_slope.axes[0].get_title(loc="right").startswith("no slope")
    plt.close(no_slope)
    with pytest.raises(TypeError, match="plot_calibration takes the result of timegym.diagnostics.calibration"):
        plot_calibration(pd.DataFrame())
