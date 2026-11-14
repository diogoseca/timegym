"""Names that leave in TimeGym 2.0 still work, warn once at the caller's line and name what to use instead."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from timegym import Case, NaiveForecaster, Pipeline, Scenario
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
    case = deprecated_call("use Scenario(kind, context_length, horizon, parameters, n_targets, seed).generate()",
                           generate_process, kind, {}, 24, 4, n_targets, 7)
    assert case.manifest == expected.manifest
    pd.testing.assert_frame_equal(case.observed, expected.observed)
    np.testing.assert_array_equal(case.reference.mean, expected.reference.mean)


@pytest.mark.parametrize("kind", PROCESS_KINDS)
def test_generate_joint_process_matches_scenario_generate(kind):
    parameters = {"correlation": .5}
    expected = Scenario(kind, 12, 2, parameters, 2, 9).generate()
    resolved = resolve_process(kind, parameters, 12, 2, 2, 9)
    case = deprecated_call("use Scenario(kind, context_length, horizon, parameters, n_targets, seed).generate()",
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

    pipeline = Pipeline(NaiveForecaster())
    card = deprecated_call("use Pipeline(model).scorecard(...)", scorecard, pipeline, ["random_walk"], instances=2)
    pd.testing.assert_frame_equal(card.table, pipeline.scorecard(["random_walk"], instances=2).table)
    assert card.title == pipeline.scorecard(["random_walk"], instances=2).title


def test_replay_case_function_matches_the_report_method():
    from timegym.pipeline import replay_case

    report = Pipeline().test([Scenario("ar1", context_length=20, horizon=3)], seeds=[0, 1])
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
    scenarios = [Scenario("white_noise", context_length=10, horizon=3)]
    pipeline = Pipeline()
    figure = deprecated_call("use Pipeline.run(...).plot_forecasts()", pipeline.plot_tests, scenarios, seeds=[4])
    np.testing.assert_array_equal(_pixels(figure), _pixels(pipeline.run(scenarios, seeds=[4]).plot_forecasts()))
    default = deprecated_call("Pipeline.plot_tests", pipeline.plot_tests, seeds=[0])
    suite = default_scenarios(context_length=240, horizon=80)
    np.testing.assert_array_equal(_pixels(default), _pixels(pipeline.run(suite, seeds=[0]).plot_forecasts()))


def test_pipeline_plot_errors_matches_run_plot_errors():
    scenarios = [Scenario("white_noise", context_length=10, horizon=3)]
    pipeline = Pipeline()
    figure = deprecated_call("use Pipeline.run(...).plot_errors()", pipeline.plot_errors, scenarios, metric="mse")
    expected = pipeline.run(scenarios, instances=20).plot_errors(metric="mse")
    np.testing.assert_array_equal(_pixels(figure), _pixels(expected))


def test_old_default_scenarios_path_is_the_same_function_and_silent(recwarn):
    from timegym.data import synthetic

    assert synthetic.default_scenarios is default_scenarios
    assert synthetic.Case is Case and synthetic.Scenario is Scenario
    assert synthetic.default_scenarios()[0].kind == "white_noise"
    assert not [warning for warning in recwarn if warning.category is DeprecationWarning]
