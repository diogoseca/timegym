import numpy as np
import pandas as pd
import pytest

from timegym import Pipeline, Scenario, NaiveForecaster, GaussianAR1Forecaster, Forecast
from timegym.checks import check_shift_equivariance
from timegym.mutations import mutate_forecast


def test_default_end_to_end_report(tmp_path):
    report = Pipeline().test([Scenario("random_walk")], seeds=[4, 5])
    assert len(report.cases) == 2
    assert all(c.metrics for c in report.cases)
    report.assert_passed()
    report.to_json(tmp_path / "report.json")
    assert (tmp_path / "report.json").exists()


def test_sklearn_transformations_and_refit_isolation():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    scaler, regressor = StandardScaler(), LinearRegression()
    pipeline = Pipeline(window_length=4).add(scaler).add(regressor)
    case = Scenario("trend", parameters={"sigma": 0, "slope": 2}).generate()
    forecast = pipeline.predict(case.task.context, case.task.future_index)
    np.testing.assert_allclose(forecast.point, case.reference.mean, atol=1e-8)
    assert not hasattr(scaler, "mean_") and not hasattr(regressor, "coef_")
    again = pipeline.predict(case.task.context, case.task.future_index)
    np.testing.assert_array_equal(forecast.point, again.point)


def test_native_templates_not_mutated_and_have_no_future_targets():
    model = NaiveForecaster()
    case = Scenario("ar1", n_targets=2).generate()
    forecast = Pipeline(model).predict(case.task.context, case.task.future_index)
    assert not hasattr(model, "context_")
    assert forecast.validate(case.task) == []


def test_native_probabilistic_model_and_shift_relation():
    case = Scenario("ar1", n_targets=2, parameters={"correlation": .5}).generate()
    pipeline = Pipeline(GaussianAR1Forecaster(n_samples=64))
    forecast = pipeline.predict(case.task.context, case.task.future_index)
    assert forecast.validate(case.task) == []
    assert check_shift_equivariance(pipeline, case.task.context, case.task.future_index).status == "PASS"


def test_exceptions_are_reported_not_hidden():
    class Broken:
        def fit(self, context):
            raise RuntimeError("intentional failure")
    report = Pipeline(model_factory=Broken).test([Scenario("ar1")], seeds=[0])
    assert report.cases[0].diagnostics[0].status == "ERROR"
    assert "intentional failure" in report.cases[0].diagnostics[0].message
    assert "RuntimeError" in report.cases[0].diagnostics[0].evidence["traceback"]
    with pytest.raises(AssertionError):
        report.assert_passed()


def test_marginal_preserving_shuffle_and_bypass_repair():
    case = Scenario("ar1", n_targets=2, horizon=5).generate()
    original = case.reference.forecast(n_samples=128, seed=3)
    for strength in (0, .25, .5, 1):
        mutated = mutate_forecast(original, "dependence_shuffle", seed=4, strength=strength)
        np.testing.assert_array_equal(np.sort(original.samples, axis=0), np.sort(mutated.samples, axis=0))
    repaired = mutate_forecast(original, "dependence_shuffle", enabled=False)
    np.testing.assert_array_equal(repaired.samples, original.samples)


def test_future_index_boundary():
    context = pd.DataFrame({"y": [1., 2., 3.]})
    with pytest.raises(ValueError, match="after"):
        Pipeline().predict(context, pd.RangeIndex(2, 5))


def test_supplied_prefitted_native_template_is_reset_by_clone_protocol():
    class Accumulator:
        def __init__(self):
            self.total = 0

        def clone_unfitted(self):
            return type(self)()

        def fit(self, context):
            self.total += context.to_numpy().sum()
            self.names = tuple(context.columns)
            return self

        def predict(self, future_index):
            return Forecast(future_index, point=np.full((len(future_index), len(self.names)), self.total), target_names=self.names)

    model = Accumulator()
    model.total = 9000  # Deliberately already fitted/contaminated, without '_' attributes.
    context = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    pipeline = Pipeline(model)
    for _ in range(2):
        np.testing.assert_array_equal(pipeline.predict(context, pd.RangeIndex(3, 5)).point, [[6], [6]])
    assert model.total == 9000


def test_native_factory_is_called_per_fit_and_unsafe_instances_are_rejected():
    instances = []

    class Native:
        def fit(self, context):
            self.last = context.iloc[-1].to_numpy()
            return self

        def predict(self, future_index):
            return Forecast(future_index, point=np.tile(self.last, (len(future_index), 1)))

    def factory():
        model = Native()
        instances.append(model)
        return model

    context, future = pd.DataFrame({"y": [1.0, 2.0]}), pd.RangeIndex(2, 4)
    pipeline = Pipeline(model_factory=factory)
    pipeline.predict(context, future)
    pipeline.predict(context, future)
    assert len(instances) == 2 and instances[0] is not instances[1]
    with pytest.raises(TypeError, match="clone_unfitted"):
        Pipeline(Native()).predict(context, future)
    with pytest.raises(ValueError, match="not both"):
        Pipeline(Native(), model_factory=factory)


def test_clone_protocol_must_not_return_shared_instance():
    class BrokenClone:
        def clone_unfitted(self):
            return self
    with pytest.raises(ValueError, match="new model instance"):
        Pipeline(BrokenClone()).predict(pd.DataFrame({"y": [1.0, 2.0]}), pd.RangeIndex(2, 4))


@pytest.mark.parametrize("mode", ["instance", "class", "factory"])
def test_direct_sklearn_model_supports_current_estimator_tags_and_cloning(mode):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit([[0], [1]], [999, 999])  # Old fitted coefficients must be discarded.
    pipeline = (Pipeline(model, window_length=4) if mode == "instance" else
                Pipeline(LinearRegression, window_length=4) if mode == "class" else
                Pipeline(model_factory=LinearRegression, window_length=4))
    case = Scenario("trend", parameters={"sigma": 0, "slope": 2}).generate()
    np.testing.assert_allclose(pipeline.predict(case.task.context, case.task.future_index).point,
                               case.reference.mean, atol=1e-8)
    np.testing.assert_array_equal(model.predict([[0], [1]]), [999, 999])


@pytest.mark.parametrize("model", [NaiveForecaster("drift"), GaussianAR1Forecaster(n_samples=8)])
def test_step_based_models_reject_skipped_future_labels(model):
    context = pd.DataFrame({"y": np.arange(8.0)})
    with pytest.raises(ValueError, match="consecutive future labels"):
        Pipeline(model).predict(context, pd.Index([9, 10]))


def test_sklearn_reduction_rejects_time_gap():
    from sklearn.linear_model import LinearRegression
    context = pd.DataFrame({"y": np.arange(8.0)})
    with pytest.raises(ValueError, match="consecutive future labels"):
        Pipeline(LinearRegression(), window_length=3).predict(context, pd.Index([8, 10]))


@pytest.mark.parametrize("context_index,future_index", [
    (pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC"), pd.date_range("2026-01-06", periods=2, freq="D", tz="UTC")),
    (pd.timedelta_range("0 days", periods=5, freq="2h"), pd.timedelta_range("10h", periods=2, freq="2h")),
    (pd.period_range("2026-01", periods=5, freq="M"), pd.period_range("2026-06", periods=2, freq="M")),
    (pd.RangeIndex(0, 10, 2), pd.RangeIndex(10, 14, 2)),
    (pd.Index(np.arange(5) * 0.1), pd.Index(np.arange(5, 7) * 0.1)),
])
def test_step_based_models_support_regular_time_indexes(context_index, future_index):
    context = pd.DataFrame({"y": np.arange(5.0)}, index=context_index)
    forecast = Pipeline(NaiveForecaster("drift")).predict(context, future_index)
    np.testing.assert_allclose(forecast.point, [[5], [6]])
    assert forecast.index.equals(future_index)


def test_duplicate_target_names_are_rejected_at_pipeline_boundary():
    context = pd.DataFrame([[1, 2], [3, 4]], columns=["y", "y"])
    with pytest.raises(ValueError, match="unique"):
        Pipeline().predict(context, pd.RangeIndex(2, 4))


def test_series_integer_zero_name_is_preserved():
    forecast = Pipeline().predict(pd.Series([1.0, 2.0], name=0), pd.RangeIndex(2, 4))
    assert forecast.target_names == (0,)


def test_last_baseline_supports_one_observation():
    forecast = Pipeline().predict(pd.DataFrame({"y": [3.0]}), pd.RangeIndex(1, 3))
    np.testing.assert_array_equal(forecast.point, [[3], [3]])


def test_mutation_fixtures_reject_missing_or_ineffective_output_types():
    case = Scenario("white_noise", horizon=2).generate()
    point_only = Forecast(case.task.future_index, point=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="predictive samples"):
        mutate_forecast(point_only, "dependence_shuffle")
    with pytest.raises(ValueError, match="distinct ordered quantiles"):
        mutate_forecast(point_only, "quantile_crossing")
    with pytest.raises(ValueError, match="different from horizon"):
        mutate_forecast(case.reference.forecast(n_samples=2), "sample_axis_swap")
    integer_point = Forecast(case.task.future_index, point=[[1], [2]])
    assert np.isnan(mutate_forecast(integer_point, "nonfinite_point").point[0, 0])


def test_metamorphic_tolerances_must_be_declared_valid_values():
    case = Scenario("ar1").generate()
    with pytest.raises(ValueError, match="nonnegative"):
        check_shift_equivariance(Pipeline(), case.task.context, case.task.future_index, atol=-1)


def test_default_run_uses_the_public_core_scenario_suite():
    from timegym.data.synthetic import default_scenarios
    from timegym.scenarios import CORE_KINDS

    specifications = default_scenarios()
    assert tuple(scenario.kind for scenario in specifications) == CORE_KINDS
    run = Pipeline().run(seeds=[0])
    assert [entry.case.manifest for entry in run.cases] == [
        scenario.generate().manifest for scenario in specifications
    ]


@pytest.mark.parametrize("model", [None, NaiveForecaster, NaiveForecaster(), GaussianAR1Forecaster()])
def test_model_metadata_name_for_native_models(model):
    pipeline = Pipeline(model)
    expected = "GaussianAR1Forecaster" if isinstance(model, GaussianAR1Forecaster) else "NaiveForecaster"
    assert pipeline._model_name() == expected


def test_model_metadata_name_does_not_invoke_factories_and_identifies_steps():
    def factory():
        raise AssertionError("Metadata must not instantiate the model")

    assert Pipeline(model_factory=factory)._model_name() == factory.__qualname__
    pipeline = Pipeline().add(object())
    assert pipeline._model_name() == "SklearnForecaster"

    class Factory:
        def __call__(self):
            raise AssertionError("Metadata must not instantiate the model")

    assert Pipeline(model_factory=Factory())._model_name() == "model_factory"
