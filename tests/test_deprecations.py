"""Names that leave in TimeGym 2.0 still work, warn once at the caller's line and name what to use instead."""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from timegym import (Case, DriftForecaster, LastValueForecaster, MeanForecaster, NaiveForecaster, Pipeline, Scenario,
                     SeasonalLastValueForecaster, SeasonalMeanForecaster, TestReport)
from timegym.metrics import crps, empirical_crps
from timegym.processes import PROCESS_KINDS, generate_process, process_label, resolve_process
from timegym.joint_processes import generate_joint_process
from timegym.scenarios import default_scenarios, scenario_label

GONE = "will be removed in TimeGym 2.0"


def deprecated_call(match, function, *args, **kwargs):
    """Call ``function`` expecting exactly one deprecation warning, raised at this file's call line."""
    with pytest.warns(DeprecationWarning) as caught:
        result = function(*args, **kwargs)
    messages = [str(warning.message) for warning in caught if warning.category is DeprecationWarning]
    assert len(messages) == 1, messages
    assert GONE in messages[0] and match in messages[0], messages[0]
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    return result


@pytest.fixture
def frame():
    return pd.DataFrame({"y": np.arange(12, dtype=float) ** 1.5},
                        index=pd.date_range("2024-01-01", periods=12, freq="D"))


@pytest.mark.parametrize("module, name, arguments, chapter", [
    ("smoothing", "simple_moving_average", lambda f: (f,), "source/03_smoothing.ipynb"),
    ("smoothing", "exponential_moving_average", lambda f: (f, 0.5), "source/03_smoothing.ipynb"),
    ("features", "add_lagged_target", lambda f: (f, 2), "source/04_feature_engineering.ipynb"),
    ("features", "cyclical_encoding", lambda f: (np.arange(7), False, 0, 6), "source/04_feature_engineering.ipynb"),
    ("features", "add_time_features", lambda f: (f,), "source/04_feature_engineering.ipynb"),
    ("target", "multioutput_target", lambda f: (f, 3), "source/05_target_engineering.ipynb"),
    ("target", "onestep_target", lambda f: (f,), "source/05_target_engineering.ipynb"),
    ("target", "step_ahead_target", lambda f: (f, 2), "source/05_target_engineering.ipynb"),
    ("target", "fractional_weights", lambda f: (0.4, 5), "source/05_target_engineering.ipynb"),
    ("target", "fractional_differencing", lambda f: (f, 0.4, 5), "source/05_target_engineering.ipynb"),
])
def test_data_preparation_helpers_warn_and_name_their_chapter(frame, module, name, arguments, chapter):
    import importlib

    function = getattr(importlib.import_module(f"timegym.{module}"), name)
    deprecated_call(f"{name} is deprecated", function, *arguments(frame))
    deprecated_call(chapter, function, *arguments(frame))


def test_helpers_that_call_other_deprecated_helpers_warn_once(frame):
    from timegym.features import add_time_features, cyclical_encoding
    from timegym.target import fractional_differencing, fractional_weights, multioutput_target, onestep_target

    labels = deprecated_call("onestep_target", onestep_target, frame)
    pd.testing.assert_frame_equal(labels, deprecated_call("multioutput_target", multioutput_target, frame, 1))
    calendar = deprecated_call("add_time_features", add_time_features, frame)
    cosine, _ = deprecated_call("cyclical_encoding", cyclical_encoding, frame.index.month, period=12, min_value=1)
    np.testing.assert_allclose(calendar["month_cos"], cosine)
    weights = deprecated_call("fractional_weights", fractional_weights, 0.4, 3)[:, 0]
    differences = deprecated_call("fractional_differencing", fractional_differencing, frame, 0.4, 3)
    np.testing.assert_allclose(differences["y"], sum(weights[k] * frame["y"].shift(k) for k in range(3)).iloc[2:])


def test_tsfel_features_warn_before_validating_their_input():
    from timegym.features import add_tsfel_features

    with pytest.warns(DeprecationWarning, match="add_tsfel_features is deprecated"), pytest.raises(TypeError):
        add_tsfel_features([1.0, 2.0])


def test_fractional_differencing_orders_warn_and_integer_orders_stay_silent(frame, recwarn):
    from timegym.target import differencing, fractional_differencing

    expected = deprecated_call("fractional_differencing", fractional_differencing, frame, 0.4, 10)
    result = deprecated_call("differencing with a noninteger order, and its fractional_coefficients argument",
                             differencing, frame, 0.4)
    pd.testing.assert_frame_equal(result, expected)
    recwarn.clear()
    pd.testing.assert_frame_equal(differencing(frame, 2, lag=3), frame.diff(3).diff(3).iloc[6:])
    assert not [warning for warning in recwarn if warning.category is DeprecationWarning]


def test_get_stock_warns_and_names_its_chapter():
    from timegym.data.real import get_stock

    with pytest.warns(DeprecationWarning, match=r'get_stock is deprecated .*"Real data" \(source/02_real_data\.ipynb\)'):
        with pytest.raises(ValueError, match="ticker"):
            get_stock("")


@pytest.mark.parametrize("kind", PROCESS_KINDS)
@pytest.mark.parametrize("n_targets", [1, 2])
def test_generate_process_matches_scenario_generate(kind, n_targets):
    expected = Scenario(kind, 24, 4, {}, n_targets, 7).generate()
    case = deprecated_call("use Scenario(kind, history_length, horizon, parameters, n_targets, seed).generate()",
                           generate_process, kind, {}, 24, 4, n_targets, 7)
    assert case.manifest == expected.manifest
    pd.testing.assert_frame_equal(case.observed, expected.observed)
    np.testing.assert_array_equal(case.reference.mean, expected.reference.mean)


@pytest.mark.parametrize("kind", PROCESS_KINDS)
def test_generate_joint_process_matches_scenario_generate(kind):
    parameters = {"correlation": .5}
    expected = Scenario(kind, 12, 2, parameters, 2, 9).generate()
    resolved = resolve_process(kind, parameters, 12, 2, 2, 9)
    case = deprecated_call("use Scenario(kind, history_length, horizon, parameters, n_targets, seed).generate()",
                           generate_joint_process, kind, resolved, 12, 2, 2, 9)
    assert case.manifest == expected.manifest
    np.testing.assert_array_equal(case.reference.sample(8, 2), expected.reference.sample(8, 2))


@pytest.mark.parametrize("kind, parameters", [
    ("composite", {"amplitude": 2., "residual": "random_walk"}),
    ("merton_jump_diffusion", None),
    ("garch", {"mu": [0., 0.]}),
    ("markov_switching", {"regime_jump_rate": [0., 2.]}),
])
def test_process_label_matches_scenario_label(kind, parameters):
    label = deprecated_call("use timegym.scenarios.scenario_label(kind, parameters)", process_label, kind, parameters)
    assert label == scenario_label(kind, parameters)


def test_empirical_crps_matches_crps():
    observed = np.array([[0.2], [1.0], [-0.5]])
    samples = np.random.default_rng(0).normal(size=(50, 3, 1))
    assert deprecated_call("use crps instead", empirical_crps, observed, samples) == crps(observed, samples)


def test_scorecard_function_matches_the_pipeline_method():
    from timegym.scorecard import scorecard

    pipeline = Pipeline(LastValueForecaster())
    card = deprecated_call("use Pipeline(model).run(...).scorecard", scorecard, pipeline, ["random_walk"], instances=2)
    expected = pipeline.run(["random_walk"], instances=2).scorecard
    pd.testing.assert_frame_equal(card.table, expected.table)
    assert card.title == expected.title


def test_replay_case_function_matches_the_report_method():
    from timegym.pipeline import replay_case

    report = Pipeline(LastValueForecaster()).run([Scenario("ar1", history_length=20, horizon=3)], seeds=[0, 1]).report
    replayed = deprecated_call("use report.replay_case(case_index)", replay_case, report, 1)
    expected = report.replay_case(1)
    assert replayed.metadata["replay"] == expected.metadata["replay"]
    assert [case.to_dict() for case in replayed.cases] == [case.to_dict() for case in expected.cases]


def _pixels(figure):
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return pixels


def test_pipeline_plot_tests_matches_run_plot_forecasts():
    scenarios = [Scenario("white_noise", history_length=10, horizon=3)]
    pipeline = Pipeline(LastValueForecaster())
    figure = deprecated_call(
        "use Pipeline.run(..., keep_data=True).plot_forecasts()", pipeline.plot_tests, scenarios, seeds=[4]
    )
    expected = pipeline.run(scenarios, seeds=[4], keep_data=True).plot_forecasts()
    np.testing.assert_array_equal(_pixels(figure), _pixels(expected))
    default = deprecated_call("Pipeline.plot_tests", pipeline.plot_tests, seeds=[0])
    suite = default_scenarios(history_length=240, horizon=80)
    np.testing.assert_array_equal(_pixels(default), _pixels(pipeline.run(suite, seeds=[0], keep_data=True).plot_forecasts()))


def test_pipeline_plot_errors_matches_run_plot_errors():
    scenarios = [Scenario("white_noise", history_length=10, horizon=3)]
    pipeline = Pipeline(LastValueForecaster())
    figure = deprecated_call("use Pipeline.run(...).plot_errors()", pipeline.plot_errors, scenarios, metric="mse")
    expected = pipeline.run(scenarios, instances=20).plot_errors(metric="mse")
    np.testing.assert_array_equal(_pixels(figure), _pixels(expected))


def test_old_default_scenarios_path_is_the_same_function_and_silent(recwarn):
    from timegym.data import synthetic

    assert synthetic.default_scenarios is default_scenarios
    assert synthetic.Case is Case and synthetic.Scenario is Scenario
    assert synthetic.default_scenarios()[0].kind == "white_noise"
    assert not [warning for warning in recwarn if warning.category is DeprecationWarning]


# The past a forecaster sees is its history: the old name, context, still works and warns.

LATER = "is deprecated and will be removed in a later release"
FIXTURES = Path(__file__).parent / "fixtures"


def renamed_call(match, function, *args, **kwargs):
    """Call ``function`` expecting one warning that an old name was used, raised at this file's call line."""
    with pytest.warns(DeprecationWarning) as caught:
        result = function(*args, **kwargs)
    messages = [str(warning.message) for warning in caught if warning.category is DeprecationWarning]
    assert len(messages) == 1, messages
    assert LATER in messages[0] and match in messages[0], messages[0]
    assert Path(caught[0].filename).resolve() == Path(__file__).resolve()
    return result


def test_scenario_takes_history_length_and_still_its_old_name():
    scenario = Scenario("ar1", history_length=30, horizon=4)
    old = renamed_call("use Scenario(history_length=...)", Scenario, "ar1", context_length=30, horizon=4)
    assert old == scenario and old.history_length == 30
    assert renamed_call("use Scenario.history_length", getattr, scenario, "context_length") == 30
    renamed_call("use Scenario.history_length", setattr, old, "context_length", 40)
    assert old.history_length == 40
    with pytest.raises(TypeError, match="give history_length only"):
        Scenario("ar1", history_length=30, context_length=30)
    case = scenario.generate(seed=3)
    assert case.manifest["history_length"] == 30 and "context_length" not in case.manifest


def test_default_scenarios_and_scenario_grid_take_the_old_keyword():
    from timegym.grid import scenario_grid

    assert renamed_call("use default_scenarios(history_length=...)", default_scenarios, context_length=30) == \
        default_scenarios(history_length=30)
    cells = renamed_call("use scenario_grid(history_length=...)", scenario_grid, ["trend"], context_length=60)
    assert [cell.scenario.history_length for cell in cells] == [60, 60]


def test_process_functions_take_the_old_keyword():
    expected = Scenario("composite", 24, 4, {}, 1, 7).generate()
    with pytest.warns(DeprecationWarning) as caught:
        case = generate_process("composite", {}, context_length=24, horizon=4, n_targets=1, seed=7)
        resolved = resolve_process("garch", {}, context_length=12, horizon=2, n_targets=2, seed=9)
        joint = generate_joint_process("garch", resolved, context_length=12, horizon=2, n_targets=2, seed=9)
    messages = [str(warning.message) for warning in caught]
    assert sum("use generate_process(history_length=...)" in message for message in messages) == 1
    assert sum("use resolve_process(history_length=...)" in message for message in messages) == 1
    assert sum("use generate_joint_process(history_length=...)" in message for message in messages) == 1
    assert all(Path(warning.filename).resolve() == Path(__file__).resolve() for warning in caught)
    assert case.manifest == expected.manifest
    assert joint.manifest == Scenario("garch", 12, 2, {}, 2, 9).generate().manifest


def test_forecast_task_takes_history_and_still_its_old_name():
    from timegym import ForecastTask

    history, future = pd.DataFrame({"y": [1.0, 2.0, 3.0]}), pd.RangeIndex(3, 5)
    task = ForecastTask(history, future)
    old = renamed_call("use ForecastTask(history=...)", ForecastTask, context=history, future_index=future)
    pd.testing.assert_frame_equal(old.history, task.history)
    pd.testing.assert_frame_equal(renamed_call("use ForecastTask.history", getattr, task, "context"), history)
    renamed_call("use ForecastTask.history", setattr, task, "context", history + 1)
    pd.testing.assert_frame_equal(task.history, history + 1)


def test_predict_metrics_and_the_shift_check_take_the_old_keyword():
    from timegym.checks import check_shift_equivariance
    from timegym.metrics import mase, rmsse

    case = Scenario("random_walk", history_length=30, horizon=4).generate(seed=2)
    pipeline = Pipeline(LastValueForecaster())
    history, future = case.task.history, case.task.future_index
    forecast = renamed_call("use Pipeline.predict(history=...)", pipeline.predict, context=history, future_index=future)
    np.testing.assert_array_equal(forecast.point, pipeline.predict(history, future).point)
    observed, past = case.observed.to_numpy(), history.to_numpy()
    for metric in (mase, rmsse):
        old = renamed_call(f"use {metric.__name__}(history=...)", metric, observed, forecast.point, context=past)
        assert old == metric(observed, forecast.point, history=past)
    old = renamed_call("use check_shift_equivariance(history=...)", check_shift_equivariance, pipeline,
                       context=history, future_index=future)
    assert old.status == check_shift_equivariance(pipeline, history, future).status


def test_scorecard_calls_take_the_old_keyword():
    from timegym.scorecard import scorecard

    pipeline = Pipeline(LastValueForecaster())
    expected = pipeline.run(["trend"], instances=2, history_length=40).scorecard
    with pytest.warns(DeprecationWarning) as caught:
        method = pipeline.scorecard(["trend"], instances=2, context_length=40)
        function = scorecard(pipeline, ["trend"], instances=2, context_length=40)
    messages = [str(warning.message) for warning in caught]
    assert sum("use Pipeline.scorecard(history_length=...)" in message for message in messages) == 1
    assert sum("use scorecard(history_length=...)" in message for message in messages) == 1
    for card in (method, function):
        pd.testing.assert_frame_equal(card.table, expected.table)


def test_scorecard_calls_take_attainable_by_its_old_name_and_run_takes_only_the_new_one():
    from timegym.scorecard import scorecard

    pipeline = Pipeline(LastValueForecaster())
    settings = dict(noise_levels=[1.0], instances=1, horizon=6)
    expected = pipeline.run(["trend"], **settings, simulate_attainable=True).scorecard
    assert np.isfinite(expected.table["attainable"].iloc[0])  # off the default grid, so simulated
    with pytest.warns(DeprecationWarning) as caught:
        method = pipeline.scorecard(["trend"], **settings, attainable=True)
        function = scorecard(pipeline, ["trend"], **settings, attainable=True)
    renamed = [warning for warning in caught if "(attainable=...)" in str(warning.message)]
    assert [str(warning.message) for warning in renamed] == [
        f"Pipeline.scorecard(attainable=...) {LATER}: use Pipeline.scorecard(simulate_attainable=...)",
        f"scorecard(attainable=...) {LATER}: use scorecard(simulate_attainable=...)"]
    assert all(Path(warning.filename).resolve() == Path(__file__).resolve() for warning in renamed)
    for card in (method, function):
        pd.testing.assert_frame_equal(card.table, expected.table)
    with pytest.raises(TypeError, match="unexpected keyword argument 'attainable'"):
        pipeline.run(["trend"], attainable=True)


def test_the_baselines_keep_their_history_and_the_old_attribute_warns():
    history = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    model = LastValueForecaster().fit(history)
    pd.testing.assert_frame_equal(model.history_, history)
    pd.testing.assert_frame_equal(renamed_call("use LastValueForecaster.history_", getattr, model, "context_"), history)
    assert not hasattr(LastValueForecaster(), "context_")
    with pytest.warns(DeprecationWarning, match="use LastValueForecaster()"):
        old = NaiveForecaster().fit(history)
    pd.testing.assert_frame_equal(renamed_call("use NaiveForecaster.history_", getattr, old, "context_"), history)


def test_a_metric_or_check_that_asks_for_context_gets_the_history_and_warns_once_per_run():
    def history_rows(history):
        return float(len(history))

    def context_rows(context):
        return float(len(context))

    def history_is_finite(history):
        """History is finite"""
        return bool(np.isfinite(history).all())

    def context_is_finite(context):
        """Context is finite"""
        return bool(np.isfinite(context).all())

    scenarios, pipeline = [Scenario("ar1", history_length=30, horizon=3)], Pipeline(LastValueForecaster())
    with pytest.warns(DeprecationWarning) as caught:
        run = pipeline.run(scenarios, seeds=[0, 1, 2], metrics=[history_rows, context_rows],
                           checks=[history_is_finite, context_is_finite])
    messages = [str(warning.message) for warning in caught]
    assert messages == [
        f"context as an input of metric 'context_rows' {LATER}: name the argument history, which gets the same values",
        f"context as an input of check 'context_is_finite' {LATER}: name the argument history, which gets the same "
        "values",
    ]
    assert all(Path(warning.filename).resolve() == Path(__file__).resolve() for warning in caught)
    for entry in run.cases:
        assert entry.metrics["context_rows"] == entry.metrics["history_rows"] == 30.0
    run.assert_passed()


def test_a_build_of_your_own_whose_scenario_says_context_length_still_works():
    from timegym import Pattern
    from timegym.grid import scenario_grid

    class OldScenario:
        """A scenario of a user's own, written for TimeGym 1.13.1."""

        kind = "old"

        def __init__(self, length, horizon):
            self.context_length, self.horizon = length, horizon
            self.scenario = Scenario("trend", history_length=length, horizon=horizon)

        def generate(self, seed):
            return self.scenario.generate(seed)

    pattern = Pattern("old", "Old scenario", "", "a trend", "noise", build=lambda noise, length, horizon:
                      OldScenario(length, horizon))
    with pytest.warns(DeprecationWarning, match=f"a scenario's context_length {LATER}: name it history_length"):
        cells = scenario_grid([pattern], [0.5], history_length=40, horizon=4)
    with pytest.warns(DeprecationWarning, match=f"a scenario's context_length {LATER}"):
        card = Pipeline(LastValueForecaster()).run([pattern], noise_levels=[0.5], instances=2, history_length=40,
                                               horizon=4).scorecard
    assert len(cells) == 1 and card.table["instances"].tolist() == [2]


def test_a_report_saved_by_timegym_1_13_loads_replays_and_compares():
    from timegym import TestReport

    old = TestReport.load(FIXTURES / "report-1.13.1.json")
    assert old.cases[0].manifest["context_length"] == 24
    assert list(old.metrics.columns[:2]) == ["kind", "seed"]  # saved before runs recorded their scenarios
    assert old.metadata["pipeline_config"]["model"]["class"] == "NaiveForecaster"
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # its model is rebuilt without NaiveForecaster's warning
        replayed = old.replay_case(0)
    assert replayed.metadata["replay"]["manifest_match"] is True
    assert replayed.cases[0].manifest["history_length"] == 24
    scenarios = [Scenario("ar1", history_length=24, horizon=3), Scenario("trend", history_length=16, horizon=2)]
    new = Pipeline(LastValueForecaster()).run(scenarios, seeds=[0, 1]).report
    comparison = new.compare(old, metric="mse")
    assert comparison.table["cases"].sum() > 0
    by_length = new.compare(old, metric="mse", by="history_length")
    assert set(by_length.table["group"]) == {24, 16}


def test_a_scorecard_baseline_saved_by_timegym_1_13_still_compares():
    card = Pipeline(LastValueForecaster()).run(["noise", "trend"], instances=20, seed=4).scorecard
    comparison = card.compare(FIXTURES / "scorecard-baseline-1.13.1.json")
    assert set(comparison.table["change"]) == {"unchanged"}


# One class per built-in forecaster: NaiveForecaster still works, warns once and names the class that does the same.

SAME_FORECASTS = [
    (("last",), LastValueForecaster(), "LastValueForecaster()"),
    (("last", 7), LastValueForecaster(), "LastValueForecaster()"),
    (("mean",), MeanForecaster(), "MeanForecaster()"),
    (("mean", 12), MeanForecaster(), "MeanForecaster()"),
    (("seasonal", 12), SeasonalLastValueForecaster(period=12), "SeasonalLastValueForecaster(period=12)"),
    (("seasonal", 1), SeasonalLastValueForecaster(period=1), "SeasonalLastValueForecaster(period=1)"),
    (("seasonal_mean", 4), SeasonalMeanForecaster(period=4), "SeasonalMeanForecaster(period=4)"),
    (("drift",), DriftForecaster(), "DriftForecaster()"),
]


def several_series():
    """Histories and the dates to forecast: one series and several, a cycle, a trend, a walk and calendar dates."""
    cases = [Scenario("seasonal", history_length=36, horizon=15).generate(seed=1),
             Scenario("trend", history_length=30, horizon=5, n_targets=3).generate(seed=2),
             Scenario("random_walk", history_length=24, horizon=30, n_targets=2).generate(seed=3)]
    days = pd.date_range("2024-01-01", periods=20, freq="D")
    rng = np.random.default_rng(4)
    calendar = pd.DataFrame({"north": rng.poisson(3, 20), "south": rng.poisson(5, 20)}, index=days).astype(float)
    return [(case.task.history, case.task.future_index) for case in cases] + [
        (calendar, pd.date_range("2024-01-21", periods=9, freq="D"))]


@pytest.mark.parametrize("arguments, model, instead", SAME_FORECASTS)
def test_naive_forecaster_warns_and_makes_the_forecasts_of_the_class_it_names(arguments, model, instead):
    old = renamed_call(f"NaiveForecaster {LATER}: use {instead}, which makes the same forecasts", NaiveForecaster,
                       *arguments)
    assert old.params() == {"strategy": arguments[0], "period": arguments[1] if len(arguments) > 1 else 1}
    for history, future in several_series():
        expected = model.clone_unfitted().fit(history).predict(future)
        forecast = old.fit(history).predict(future)
        np.testing.assert_array_equal(forecast.point, expected.point)
        assert forecast.target_names == expected.target_names and forecast.index.equals(expected.index)
    for rows in (1, 3):  # a history too short for one is too short for the other
        short = pd.DataFrame({"y": np.arange(float(rows))})
        refusals = []
        for candidate in (old, model.clone_unfitted()):
            try:
                candidate.fit(short)
                refusals.append(None)
            except ValueError as error:
                refusals.append(str(error))
        assert refusals[0] == refusals[1]


def test_naive_forecaster_refuses_an_unknown_strategy_before_warning(recwarn):
    with pytest.raises(ValueError, match="unknown.*last.*mean.*seasonal.*seasonal_mean.*drift"):
        NaiveForecaster("unknown")
    assert not [warning for warning in recwarn if warning.category is DeprecationWarning]


def test_a_pipeline_of_naive_forecaster_warns_once_and_its_report_still_names_and_replays_it():
    with pytest.warns(DeprecationWarning) as caught:
        pipeline = Pipeline(NaiveForecaster("seasonal", period=12))
        run = pipeline.run(["seasonal", "trend"], instances=3)
        also = Pipeline(NaiveForecaster).run(["trend"], instances=2)  # the class: one warning, not one per case
    messages = [str(warning.message) for warning in caught if warning.category is DeprecationWarning]
    assert len(messages) == 2 and "SeasonalLastValueForecaster(period=12)" in messages[0], messages
    assert "LastValueForecaster" in messages[1]
    assert all(Path(warning.filename).resolve() == Path(__file__).resolve() for warning in caught)
    same = Pipeline(SeasonalLastValueForecaster(period=12)).run(["seasonal", "trend"], instances=3)
    assert [case.metrics for case in run.report.cases] == [case.metrics for case in same.report.cases]
    assert run.metadata["model"] == "NaiveForecaster" and same.metadata["model"] == "SeasonalLastValueForecaster"
    assert run.scorecard.title == "NaiveForecaster(strategy='seasonal', period=12)"
    assert same.scorecard.title == "SeasonalLastValueForecaster(period=12)"
    assert also.metadata["pipeline_config"]["replayable"] is True
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        replayed = TestReport.from_json(run.report.to_json()).replay_case(4)
    assert replayed.metadata["replay"]["reproduced"] is True
    assert replayed.metadata["model"] == "NaiveForecaster"


# The model goes first in Pipeline: scikit-learn steps as a list, a function in the model's place, and always a model.

def _sklearn_series():
    """Histories long enough for every reduction: one series and several, steps and calendar dates."""
    cases = [Scenario("seasonal", history_length=48, horizon=6).generate(seed=1),
             Scenario("trend", history_length=40, horizon=5, n_targets=3).generate(seed=2)]
    days = pd.date_range("2024-01-01", periods=40, freq="D")
    rng = np.random.default_rng(4)
    calendar = pd.DataFrame({"north": rng.poisson(3, 40), "south": rng.poisson(5, 40)}, index=days).astype(float)
    return [(case.task.history, case.task.future_index) for case in cases] + [
        (calendar, pd.date_range("2024-02-10", periods=7, freq="D"))]


@pytest.mark.parametrize("reduction", ["recursive", "direct", "multioutput"])
def test_add_warns_once_per_pipeline_and_makes_the_forecasts_of_the_list_of_steps(reduction):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    chained = Pipeline(window_length=4, reduction=reduction)
    renamed_call(f"Pipeline.add() {LATER}: use a list of the steps, such as Pipeline([StandardScaler(), Ridge()], "
                 "window_length=12), which makes the same forecasts", chained.add, StandardScaler())
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        chained.add(Ridge, alpha=2.)  # the second step of the same pipeline: no second warning
        listed = Pipeline([StandardScaler(), Ridge(alpha=2.)], window_length=4, reduction=reduction)
        made = Pipeline(make_pipeline(StandardScaler(), Ridge(alpha=2.)), window_length=4, reduction=reduction)
        for history, future in _sklearn_series():
            expected = chained.predict(history, future)
            for other in (listed, made):
                forecast = other.predict(history, future)
                np.testing.assert_array_equal(forecast.point, expected.point)
                assert forecast.target_names == expected.target_names and forecast.index.equals(expected.index)
        old, new = (pipeline.run(["trend", "seasonal"], instances=2) for pipeline in (chained, listed))
    assert [case.metrics for case in old.report.cases] == [case.metrics for case in new.report.cases]
    assert (old.metadata["model"], new.metadata["model"]) == ("SklearnForecaster", "StandardScaler + Ridge")
    assert old.scorecard.title == f"StandardScaler → Ridge ({reduction}, window 4)"
    assert new.scorecard.title == "StandardScaler + Ridge"
    assert old.metadata["pipeline_config"]["model"]["kind"] == "sklearn_steps"


def test_an_add_chain_with_no_regressor_still_fails_before_any_case():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    pipeline = renamed_call("Pipeline.add()", Pipeline(window_length=4).add, StandardScaler())
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="Final step.*StandardScaler"):
            pipeline.run([Scenario("ar1")], seeds=[0])
        pipeline.add(Ridge())
        assert pipeline.run([Scenario("ar1")], seeds=[0]).report.cases[0].diagnostics[0].status == "PASS"
    with pytest.raises(ValueError, match="not both"):
        Pipeline(LastValueForecaster()).add(Ridge())


def test_model_factory_warns_and_is_the_function_in_the_models_place():
    made = []

    def make_model():
        made.append(LastValueForecaster())
        return made[-1]

    old = renamed_call(f"Pipeline(model_factory=...) {LATER}: use Pipeline(make_model)", Pipeline,
                       model_factory=make_model)
    new = Pipeline(make_model)
    scenarios = [Scenario("trend", history_length=20, horizon=3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        runs = [pipeline.run(scenarios, seeds=[0, 1]) for pipeline in (old, new)]
        cards = [pipeline.run(["trend"], instances=2).scorecard for pipeline in (old, new)]
    cases = sum(len(run.cases) for run in runs) + sum(len(card.run.cases) for card in cards)
    assert len(made) == len({id(model) for model in made}) == cases  # a fresh model for every case
    assert [case.metrics for case in runs[0].report.cases] == [case.metrics for case in runs[1].report.cases]
    assert runs[0].metadata["model"] == runs[1].metadata["model"] == make_model.__qualname__
    assert cards[0].title == cards[1].title == make_model.__qualname__
    assert old.to_config() == new.to_config() and new.to_config()["model"]["kind"] == "factory"
    with pytest.raises(ValueError, match="not both"):
        Pipeline(LastValueForecaster(), model_factory=make_model)
    with pytest.raises(TypeError, match="model_factory must be a callable"):
        Pipeline(model_factory=LastValueForecaster())


def test_a_pipeline_with_no_model_warns_once_when_used_and_still_tests_the_last_value():
    scenarios = [Scenario("random_walk", history_length=20, horizon=3)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        pipeline = Pipeline(window_length=6)  # nothing to warn about until it is used
    run = renamed_call(f"Pipeline() with no model {LATER}: use Pipeline(LastValueForecaster()), which makes the "
                       "same forecasts", pipeline.run, scenarios, seeds=[0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        forecast = pipeline.predict([1.0, 2.0, 3.0], range(3, 5))  # once for each pipeline
        config = pipeline.to_config()
        assert Pipeline.from_config(config).to_config() == config
        replayed = TestReport.from_json(run.report.to_json()).replay_case(1)
    same = Pipeline(LastValueForecaster()).run(scenarios, seeds=[0, 1])
    assert [case.metrics for case in run.report.cases] == [case.metrics for case in same.report.cases]
    assert forecast.point.ravel().tolist() == [3.0, 3.0]
    assert run.metadata["model"] == "NaiveForecaster"
    assert config["model"] == {"kind": "builtin", "class": "NaiveForecaster",
                               "parameters": {"strategy": "last", "period": 1}}
    assert replayed.metadata["replay"]["reproduced"] is True
    for use in ("to_config", "predict", "run"):
        fresh = Pipeline()
        arguments = {"to_config": (), "predict": ([1.0, 2.0], range(2, 3)), "run": (scenarios, [0])}[use]
        renamed_call("Pipeline() with no model", getattr(fresh, use), *arguments)


# The model is named once, with Pipeline(..., name=...) and from_forecasts(..., name=...).

def test_recorded_forecasts_named_in_metadata_still_work_and_warn():
    from timegym import ForecastRun, compare_reports

    cases = [Scenario("trend", history_length=20, horizon=3).generate(seed=seed) for seed in (0, 1)]
    forecasts = [np.full(3, case.task.history.iloc[-1, 0]) for case in cases]
    old = renamed_call(f"ForecastRun.from_forecasts(metadata={{'model': ...}}) {LATER}: use "
                       "ForecastRun.from_forecasts(cases, forecasts, name=...)", ForecastRun.from_forecasts,
                       cases, forecasts, metadata={"model": "weather service", "made_on": "2024-05-01"})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = ForecastRun.from_forecasts(cases, forecasts, name="weather service", metadata={"made_on": "2024-05-01"})
        base = Pipeline(LastValueForecaster()).run(cases=cases)
        labels = [compare_reports([run], base, metric="mse").table["candidate"].tolist() for run in (old, new)]
    assert old.metadata["model"] == "weather service" and "name" not in old.metadata
    assert new.metadata["name"] == "weather service" and new.metadata["model"] == "recorded forecasts"
    assert old.metadata["made_on"] == new.metadata["made_on"] == "2024-05-01"
    assert [case.metrics for case in old.report.cases] == [case.metrics for case in new.report.cases]
    assert labels == [["weather service"], ["weather service"]]  # both still label a comparison


def test_the_old_scorecard_calls_keep_their_title_which_heads_the_card_over_the_name():
    from timegym.scorecard import scorecard

    pipeline = Pipeline(LastValueForecaster(), name="Repeat")
    titled = renamed_call("use pipeline.run(...).scorecard", pipeline.scorecard, ["trend"], instances=2, title="Mine")
    named = renamed_call("use pipeline.run(...).scorecard", pipeline.scorecard, ["trend"], instances=2)
    old = deprecated_call("timegym.scorecard.scorecard", scorecard, pipeline, ["trend"], instances=2, title="Old")
    assert (titled.title, named.title, old.title) == ("Mine", "Repeat", "Old")
    assert titled.report.metadata["name"] == "Repeat"
    pd.testing.assert_frame_equal(titled.table, pipeline.run(["trend"], instances=2).scorecard.table)


# The numbers asked for in a run are metrics: metrics= on every call, and scores= still works and warns.

def test_scores_still_asks_for_metrics_and_warns_on_every_call():
    from timegym import ForecastRun
    from timegym.diagnostics import evaluate_forecast

    scenarios = [Scenario("ar1", history_length=30, horizon=3)]
    pipeline = Pipeline(LastValueForecaster())
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = pipeline.run(scenarios, seeds=[0, 1], metrics=["rmse"])
    old = renamed_call(f"Pipeline.run(scores=...) {LATER}: use Pipeline.run(metrics=...)", pipeline.run, scenarios,
                       seeds=[0, 1], scores=["rmse"])
    assert [case.metrics for case in old.report.cases] == [case.metrics for case in new.report.cases]
    assert old.metadata["metrics"] == new.metadata["metrics"] and "scores" not in new.metadata
    with pytest.warns(DeprecationWarning) as caught:
        tested = pipeline.test(scenarios, seeds=[0, 1], scores=["rmse"])
    messages = [str(warning.message) for warning in caught]
    assert len(messages) == 2 and "use Pipeline.test(metrics=...)" in messages[0], messages
    assert all(Path(warning.filename).resolve() == Path(__file__).resolve() for warning in caught)
    assert [case.metrics for case in tested.cases] == [case.metrics for case in new.report.cases]
    cases = [scenarios[0].generate(seed=seed) for seed in (0, 1)]
    forecasts = [np.full(3, case.task.history.iloc[-1, 0]) for case in cases]
    recorded = renamed_call("use ForecastRun.from_forecasts(metrics=...)", ForecastRun.from_forecasts, cases,
                            forecasts, scores=["rmse"])
    assert recorded.metadata["metrics"][0]["name"] == "rmse"
    one = renamed_call("use evaluate_forecast(metrics=...)", evaluate_forecast, cases[0],
                       pipeline.predict(cases[0].task.history, cases[0].task.future_index), scores=["rmse"])
    assert one.metrics == new.report.cases[0].metrics
    with pytest.raises(TypeError, match="give metrics only"):
        pipeline.run(scenarios, seeds=[0], metrics=["rmse"], scores=["rmse"])


def test_a_report_that_lists_its_metrics_under_scores_still_replays_them():
    def close(observed, predicted):
        return float(np.mean(np.abs(observed - predicted) < 0.5))

    report = Pipeline(LastValueForecaster()).run([Scenario("ar1", history_length=30, horizon=3)], seeds=[4],
                                                 metrics={"rmse": "rmse", "close": close}).report
    payload = report.to_dict()
    payload["metadata"]["scores"] = payload["metadata"].pop("metrics")  # as TimeGym 1.13.1 saved it
    old = TestReport.from_dict(payload)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        replay = old.replay_case(0).metadata["replay"]
    assert replay["reproduced"] is True and replay["not_recomputed"] == ["close"]


# The error against the answer key is avoidable_mse everywhere; mean_reference_mse still reads it and warns.

def test_every_case_with_an_answer_key_records_avoidable_mse_the_scorecard_number():
    result = Pipeline(LastValueForecaster()).run(["noise", "trend"], instances=3)
    for row in result.scorecard.table.itertuples():
        cases = [entry.metrics for entry in result.cases if entry.scenario_index == row.Index]
        assert all("avoidable_mse" in metrics and "mean_reference_mse" not in metrics.keys() for metrics in cases)
        assert np.mean([metrics["avoidable_mse"] for metrics in cases]) == pytest.approx(row.avoidable_mse, rel=1e-12)


def test_reading_mean_reference_mse_from_a_case_still_works_and_warns():
    import json
    import pickle

    metrics = Pipeline(LastValueForecaster()).run(["trend"], instances=1).cases[0].metrics
    value = metrics["avoidable_mse"]
    message = 'case.metrics["mean_reference_mse"] ' + LATER + ': use case.metrics["avoidable_mse"]'
    assert renamed_call(message, metrics.__getitem__, "mean_reference_mse") == value
    assert renamed_call(message, metrics.get, "mean_reference_mse") == value
    assert renamed_call(message, metrics.__contains__, "mean_reference_mse") is True
    # Otherwise it is a plain dictionary: it compares, saves and copies like one.
    plain = dict(metrics)
    assert metrics == plain and json.dumps(metrics) == json.dumps(plain) and pickle.loads(pickle.dumps(metrics)) == plain
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert metrics.get("crps") is None and "crps" not in metrics and list(metrics) == list(plain)


def test_metric_mean_reference_mse_still_draws_and_compares_and_warns():
    from timegym.diagnostics import compare_reports
    from timegym.plotting import plot_errors

    pipeline = Pipeline(LastValueForecaster())
    run = pipeline.run(["trend"], instances=6)
    other = Pipeline(MeanForecaster()).run(["trend"], instances=6)
    message = f'metric="mean_reference_mse" {LATER}: use metric="avoidable_mse"'
    for draw in (run.plot_errors, lambda **kwargs: plot_errors(run, **kwargs)):
        figure = renamed_call(message, draw, metric="mean_reference_mse")
        assert figure.axes[-1].get_xlabel() == draw().axes[-1].get_xlabel() == (
            "Error against the answer key (MSE), one value per case")
        plt.close("all")
    old = renamed_call(message, other.compare, run, metric="mean_reference_mse")
    pd.testing.assert_frame_equal(old.table, other.compare(run, metric="avoidable_mse").table)
    old = renamed_call(message, compare_reports, other, run, metric="mean_reference_mse")
    pd.testing.assert_frame_equal(old.table, compare_reports(other, run, metric="avoidable_mse").table)


def test_a_report_saved_by_timegym_1_13_loads_its_error_against_the_answer_key_as_avoidable_mse():
    import json

    path = FIXTURES / "report-1.13.1.json"
    saved = json.loads(path.read_text(encoding="utf-8"))["cases"]
    old = TestReport.load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for case, record in zip(old.cases, saved):
            assert case.metrics["avoidable_mse"] == record["metrics"]["mean_reference_mse"]
            assert "mean_reference_mse" not in case.metrics.keys()
            evidence = next(d.evidence for d in case.diagnostics if d.name == "point_accuracy")
            assert evidence["avoidable_mse"] == record["metrics"]["mean_reference_mse"]
        replayed = old.replay_case(0)
    # The file was saved on another machine, so the recomputed numbers may differ in their last digits:
    # compare them with a tolerance, and the findings by name and status.
    again, first = replayed.cases[0], old.cases[0]
    assert set(again.metrics.keys()) == set(first.metrics.keys()) and "avoidable_mse" in again.metrics.keys()
    for name, value in first.metrics.items():
        assert again.metrics[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name
    assert ([(d.name, d.status) for d in again.diagnostics] == [(d.name, d.status) for d in first.diagnostics])
    scenarios = [Scenario("ar1", history_length=24, horizon=3), Scenario("trend", history_length=16, horizon=2)]
    new = Pipeline(LastValueForecaster()).run(scenarios, seeds=[0, 1])
    assert new.compare(old, metric="avoidable_mse").table["cases"].sum() == 4


def test_a_saved_finding_on_the_old_name_loads_under_the_new_one():
    report = Pipeline(LastValueForecaster()).run(["trend"], instances=1).report
    payload = report.to_dict()
    case = payload["cases"][0]
    case["metrics"]["mean_reference_mse"] = case["metrics"].pop("avoidable_mse")
    case["diagnostics"].append({"name": "metric_mean_reference_mse", "status": "INCONCLUSIVE",
                                "category": "capability", "message": "undefined", "evidence": {}})
    payload["summary"]["diagnostics"] += 1
    payload["summary"]["status_counts"]["INCONCLUSIVE"] += 1
    loaded = TestReport.from_dict(payload)
    assert loaded.cases[0].metrics == report.cases[0].metrics
    assert loaded.cases[0].diagnostics[-1].name == "metric_avoidable_mse"


# The dependence check is asked for like the other checks: checks=[dependence()]. Its four old arguments warn.

def _dependence_scenarios():
    return [Scenario("ar1", history_length=30, horizon=3, n_targets=2)]


@pytest.mark.parametrize("settings", [{}, {"n_calibration": 19, "alpha": 0.1}])
def test_dependence_runs_the_check_of_check_dependence_with_the_same_seeds_and_results(settings):
    from timegym import GaussianAR1Forecaster
    from timegym.checks import dependence

    pipeline = Pipeline(GaussianAR1Forecaster(n_samples=40, seed=3))
    old = renamed_call("use Pipeline.run(checks=[dependence(", pipeline.run, _dependence_scenarios(), seeds=[1, 2, 3],
                       check_dependence=True, **settings)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        new = pipeline.run(_dependence_scenarios(), seeds=[1, 2, 3], checks=[dependence(**settings)])
        replay = new.report.replay_case(2).metadata["replay"]
    assert new.report.to_dict() == old.report.to_dict()  # every finding, seed and setting
    findings = [d for case in new.report.cases for d in case.diagnostics if d.name == "reference_covariance"]
    assert len(findings) == 3 and all(d.evidence["n_calibration"] == settings.get("n_calibration", 99)
                                      for d in findings)
    assert new.metadata["diagnostic_settings"] == {"check_dependence": True, "n_calibration": 19 if settings else 99,
                                                   "alpha": 0.1 if settings else 0.05,
                                                   "dependence_seed_offset": 100_000}
    assert [row for row in new.checks if row["check"] == "dependence"] == [
        {"check": "dependence", "title": "Joint spread matches the answer key", "passed": 3, "failed": 0}]
    assert "checks" not in new.metadata and replay["reproduced"] is True


def test_the_old_dependence_arguments_warn_on_every_call_and_give_the_same_findings():
    from timegym import ForecastRun, GaussianAR1Forecaster
    from timegym.checks import dependence
    from timegym.diagnostics import evaluate_forecast

    pipeline = Pipeline(GaussianAR1Forecaster(n_samples=40, seed=3))
    new = pipeline.run(_dependence_scenarios(), seeds=[1, 2], checks=[dependence(n_calibration=19)])
    with pytest.warns(DeprecationWarning) as caught:
        tested = pipeline.test(_dependence_scenarios(), seeds=[1, 2], check_dependence=True, n_calibration=19)
    messages = [str(warning.message) for warning in caught]
    assert len(messages) == 2 and "use Pipeline.test(checks=[dependence(n_calibration=...)])" in messages[1], messages
    assert all(Path(warning.filename).resolve() == Path(__file__).resolve() for warning in caught)
    assert tested.to_dict()["cases"] == new.report.to_dict()["cases"]
    cases = [scenario.generate(seed=1) for scenario in _dependence_scenarios()]
    forecasts = [pipeline.predict(case.task.history, case.task.future_index) for case in cases]
    recorded = renamed_call("use ForecastRun.from_forecasts(checks=[dependence(n_calibration=...)])",
                            ForecastRun.from_forecasts, cases, forecasts, check_dependence=True, n_calibration=19)
    assert recorded.report.to_dict()["cases"] == ForecastRun.from_forecasts(
        cases, forecasts, checks=[dependence(n_calibration=19)]).report.to_dict()["cases"]
    one = renamed_call("use evaluate_forecast(checks=[dependence(alpha=...)])", evaluate_forecast, cases[0],
                       forecasts[0], check_dependence=True, alpha=0.2, dependence_seed=5)
    assert one.to_dict() == evaluate_forecast(cases[0], forecasts[0], checks=[dependence(alpha=0.2)],
                                              dependence_seed=5).to_dict()
    offset = renamed_call("dependence_seed_offset has no new form", pipeline.run, _dependence_scenarios(),
                          seeds=[1], dependence_seed_offset=51)
    assert offset.metadata["diagnostic_settings"]["dependence_seed_offset"] == 51
    with pytest.raises(ValueError, match="boolean"):
        renamed_call("use evaluate_forecast(checks=[dependence()])", evaluate_forecast, None, None,
                     check_dependence="false")


def test_run_has_no_dependence_arguments_and_both_forms_are_refused():
    import inspect
    from timegym import ForecastRun
    from timegym.checks import dependence, within
    from timegym.diagnostics import evaluate_forecast

    old = {"check_dependence", "n_calibration", "alpha", "dependence_seed_offset"}
    assert not old & set(inspect.signature(Pipeline.run).parameters)
    pipeline, scenarios = Pipeline(LastValueForecaster()), _dependence_scenarios()
    with pytest.raises(TypeError, match=r"Pipeline.run\(\) got checks=\[dependence\(\)\] and alpha= of its old form; "
                                        r"give checks=\[dependence\(...\)\] only"):
        pipeline.run(scenarios, seeds=[0], checks=[dependence()], alpha=0.1)
    case = scenarios[0].generate(seed=0)
    with pytest.raises(TypeError, match="give checks=.dependence"):
        evaluate_forecast(case, pipeline.predict(case.task.history, case.task.future_index),
                          checks=[dependence()], check_dependence=True)
    with pytest.raises(TypeError, match="give checks=.dependence"):
        ForecastRun.from_forecasts([case], [None], checks={"dependence": dependence()}, n_calibration=19)
    with pytest.raises(TypeError, match=r"Pipeline.run\(\) got an unexpected keyword argument 'dependence_seed'"):
        pipeline.run(scenarios, seeds=[0], dependence_seed=3)
    with pytest.raises(ValueError, match="asked for twice"):
        pipeline.run(scenarios, seeds=[0], checks=[dependence(), dependence(alpha=0.1)])
    with pytest.raises(ValueError, match="named 'dependence' in every report"):
        pipeline.run(scenarios, seeds=[0], checks={"joint": dependence()})
    from timegym import GaussianAR1Forecaster

    named = Pipeline(GaussianAR1Forecaster(n_samples=40, seed=3)).run(
        scenarios, seeds=[0], checks={"dependence": dependence(), "range": within(-50, 50)})
    assert [row["check"] for row in named.checks][-2:] == ["dependence", "range"]
    assert named.metadata["diagnostic_settings"]["check_dependence"] is True
    assert repr(dependence(alpha=0.1)) == "dependence(n_calibration=99, alpha=0.1)"


def test_an_entrys_result_still_reads_and_sets_its_record_and_warns():
    from timegym import EvaluatedCase

    run = Pipeline(LastValueForecaster()).run(["trend"], instances=2)
    entry = run.cases[0]
    message = f"entry.result {LATER}: read the case's parts on the entry itself, such as entry.metrics"
    record = renamed_call(message, lambda: entry.result)
    assert record is run.report.cases[0]
    assert (record.metrics, record.diagnostics, record.manifest) == (entry.metrics, entry.diagnostics, entry.manifest)
    other = run.cases[1]
    renamed_call(message, setattr, entry, "result", renamed_call(message, lambda: other.result))
    assert entry.metrics == other.metrics
    # Making an entry by hand takes the record as before and stays silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        made = EvaluatedCase(None, None, record, 0, 0)
        assert made.metrics is record.metrics and EvaluatedCase(case=None, forecast=None, result=record,
                                                                scenario_index=0, seed_index=1).seed_index == 1


def test_to_json_with_a_path_and_read_still_write_and_load_a_report_and_warn(tmp_path):
    result = Pipeline(LastValueForecaster()).run(["trend"], instances=2)
    report = result.report
    message = f"report.to_json(path) {LATER}: use report.save(path)"
    text = renamed_call(message, report.to_json, tmp_path / "report.json")
    assert (tmp_path / "report.json").read_text(encoding="utf-8") == text + "\n"
    message = f"result.to_json(path) {LATER}: use result.save(path)"
    text = renamed_call(message, result.to_json, tmp_path / "run.json")
    assert (tmp_path / "run.json").read_text(encoding="utf-8") == text + "\n"
    message = f"TestReport.read(path) {LATER}: use TestReport.load(path)"
    loaded = renamed_call(message, TestReport.read, tmp_path / "run.json")
    assert loaded.to_dict() == TestReport.load(tmp_path / "report.json").to_dict() == report.to_dict()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert result.to_json() == report.to_json() == text  # text only: no warning
