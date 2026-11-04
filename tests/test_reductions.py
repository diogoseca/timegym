"""Recursive, direct and multioutput reductions of sklearn regressors."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from timegym import NaiveForecaster, Pipeline, Scenario, TestReport
from timegym.models import REDUCTIONS, SklearnForecaster, _lag_windows


def fitted_parameters(model):
    """Every fitted scaler and regressor array of a SklearnForecaster, in a fixed order."""
    pipelines = []
    for entry in model.models_:
        pipelines.extend(entry if isinstance(entry, list) else [entry])
    arrays = []
    for pipeline in pipelines:
        for step in pipeline.named_steps.values():
            for estimator in getattr(step, "estimators_", [step]):
                arrays.extend(np.atleast_1d(getattr(estimator, name)) for name in
                              ("mean_", "scale_", "coef_", "intercept_") if hasattr(estimator, name))
    return arrays


def test_lag_windows_label_only_observed_context_rows():
    values = np.arange(20.).reshape(10, 2)
    windows, labels = _lag_windows(values, 3, 4)
    assert windows.shape == (4, 6) and labels.shape == (4, 4, 2)
    np.testing.assert_array_equal(windows[0], values[:3].ravel())
    np.testing.assert_array_equal(windows[-1], values[3:6].ravel())
    np.testing.assert_array_equal(labels[-1], values[-4:])
    np.testing.assert_array_equal(labels[:, 0], values[3:7])


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_every_reduction_recovers_a_noiseless_linear_trend(reduction):
    case = Scenario("trend", context_length=60, horizon=9, parameters={"sigma": 0, "slope": 2}).generate()
    pipeline = Pipeline(window_length=4, reduction=reduction).add(StandardScaler()).add(LinearRegression())
    forecast = pipeline.predict(case.task.context, case.task.future_index)
    np.testing.assert_allclose(forecast.point, case.reference.mean, atol=1e-7)
    assert forecast.validate(case.task) == []


def test_direct_and_multioutput_fit_the_expected_regressors():
    case = Scenario("ar1", context_length=50, horizon=5, n_targets=2).generate(seed=1)
    direct = SklearnForecaster([StandardScaler(), Ridge()], window_length=3, reduction="direct", horizon=5)
    direct.fit(case.task.context)
    assert len(direct.models_) == 2 and all(len(models) == 5 for models in direct.models_)
    assert direct.models_[0][0] is not direct.models_[0][1]
    assert direct.models_[1][4].named_steps["step_0"].n_samples_seen_ == 50 - 3 - 5 + 1

    native = SklearnForecaster([StandardScaler(), Ridge()], window_length=3, reduction="multioutput", horizon=5)
    native.fit(case.task.context)
    assert len(native.models_) == 2 and type(native.models_[0].named_steps["step_1"]) is Ridge
    assert native.models_[0].named_steps["step_1"].coef_.shape == (5, 6)

    wrapped = SklearnForecaster([SVR()], window_length=3, reduction="multioutput", horizon=5)
    wrapped.fit(case.task.context)
    regressor = wrapped.models_[1].named_steps["step_0"]
    assert type(regressor) is MultiOutputRegressor and len(regressor.estimators_) == 5
    for model in (direct, native, wrapped):
        assert model.predict(case.task.future_index).validate(case.task) == []


def test_linear_least_squares_gives_identical_direct_and_multioutput_forecasts():
    case = Scenario("seasonal", context_length=120, horizon=12,
                    parameters={"amplitude": 2., "period": 12., "sigma": .5}).generate(seed=3)
    forecasts = {reduction: Pipeline(window_length=6, reduction=reduction).add(StandardScaler()).add(Ridge(alpha=2.))
                 .predict(case.task.context, case.task.future_index) for reduction in REDUCTIONS}
    np.testing.assert_allclose(forecasts["direct"].point, forecasts["multioutput"].point, rtol=1e-9, atol=1e-9)
    assert not np.allclose(forecasts["direct"].point, forecasts["recursive"].point)


def test_one_step_multioutput_needs_no_wrapper():
    context = pd.DataFrame({"y": np.sin(np.arange(40.))})
    model = SklearnForecaster([SVR()], window_length=5, reduction="multioutput", horizon=1).fit(context)
    assert type(model.models_[0].named_steps["step_0"]) is SVR
    assert model.predict(pd.RangeIndex(40, 41)).point.shape == (1, 1)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_training_never_reads_evaluation_futures(reduction, monkeypatch):
    fitted = []
    original_fit = SklearnForecaster.fit

    def recording_fit(self, context):
        fitted.append(self)
        return original_fit(self, context)

    monkeypatch.setattr(SklearnForecaster, "fit", recording_fit)
    scenario = Scenario("ar1", context_length=40, horizon=6, n_targets=2)
    pipeline = Pipeline(window_length=5, reduction=reduction).add(StandardScaler()).add(Ridge())
    clean = pipeline.run([scenario], seeds=[11])

    original_generate = Scenario.generate

    def mutated_future(self, seed=None):
        case = original_generate(self, seed)
        case.observed.iloc[:, :] = 1000. - 50. * case.observed.to_numpy()
        return case

    monkeypatch.setattr(Scenario, "generate", mutated_future)
    mutated = pipeline.run([scenario], seeds=[11])
    assert not np.allclose(clean.cases[0].case.observed, mutated.cases[0].case.observed)
    first, second = fitted
    assert first.reduction == second.reduction == reduction
    for before, after in zip(fitted_parameters(first), fitted_parameters(second), strict=True):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(clean.cases[0].forecast.point, mutated.cases[0].forecast.point)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_reduction_is_recorded_in_the_run_metadata_and_configuration(reduction):
    pipeline = Pipeline(window_length=4, reduction=reduction).add(Ridge())
    run = pipeline.run([Scenario("ar1", context_length=30, horizon=3)], seeds=[0])
    config = pipeline.to_config()
    assert run.metadata["reduction"] == config["reduction"] == reduction
    assert run.metadata["pipeline_config"]["reduction"] == reduction
    assert TestReport.from_json(run.report.to_json()).metadata["reduction"] == reduction
    assert all(d.status != "ERROR" for d in run.cases[0].result.diagnostics)


@pytest.mark.parametrize("reduction", REDUCTIONS)
def test_native_models_ignore_the_reduction_and_still_replay(reduction):
    case = Scenario("trend", context_length=20, horizon=4).generate(seed=5)
    baseline = Pipeline(NaiveForecaster("drift")).predict(case.task.context, case.task.future_index)
    pipeline = Pipeline(NaiveForecaster("drift"), reduction=reduction)
    np.testing.assert_array_equal(pipeline.predict(case.task.context, case.task.future_index).point, baseline.point)
    config = json.loads(json.dumps(pipeline.to_config()))
    assert config["replayable"] is True and Pipeline.from_config(config).reduction == reduction
    report = pipeline.test([Scenario("trend", context_length=20, horizon=4)], seeds=[5])
    assert TestReport.from_json(report.to_json()).replay_case().metadata["replay"]["reproduced"]


def test_reduction_configuration_errors_are_precise():
    with pytest.raises(ValueError, match="Unknown reduction 'iterated'; choose from"):
        SklearnForecaster([Ridge()], reduction="iterated")
    for reduction in ("direct", "multioutput"):
        with pytest.raises(ValueError, match=f"The {reduction} reduction requires a positive integer horizon"):
            SklearnForecaster([Ridge()], reduction=reduction)
    with pytest.raises(ValueError, match="horizon"):
        SklearnForecaster([Ridge()], reduction="direct", horizon=0)
    context = pd.DataFrame({"y": np.arange(10.)})
    with pytest.raises(ValueError, match="direct reduction needs at least window_length \\+ 4 = 10 context rows"):
        SklearnForecaster([Ridge()], window_length=6, reduction="direct", horizon=4).fit(context.iloc[:9])
    model = SklearnForecaster([Ridge()], window_length=6, reduction="multioutput", horizon=4).fit(context)
    assert model.predict(pd.RangeIndex(10, 12)).point.shape == (2, 1)
    with pytest.raises(ValueError, match="5 labels; this model forecasts at most 4"):
        model.predict(pd.RangeIndex(10, 15))
    with pytest.raises(ValueError, match="consecutive future labels"):
        model.predict(pd.Index([11, 12]))


def test_reductions_are_part_of_the_constructor_parameters_and_clones():
    model = SklearnForecaster([Ridge(alpha=3.)], window_length=4, reduction="direct", horizon=7)
    clone = model.clone_unfitted()
    assert clone.params() == model.params() and clone.params()["reduction"] == "direct"
    assert clone.params()["horizon"] == 7 and clone.steps is not model.steps
