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


def test_native_factory_is_called_per_fit_and_plain_instances_are_copied_per_case():
    instances = []

    class Native:
        def __init__(self):
            self.fits = []

        def fit(self, context):
            self.fits.append(len(context))
            self.last = context.iloc[-1].to_numpy()
            return self

        def predict(self, future_index):
            # State from an earlier case would show up as more than one fit.
            assert len(self.fits) == 1, "a model copy was fitted twice"
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
    original = Native()
    copied = Pipeline(original)
    copied.predict(context, future)
    copied.predict(context, future)
    assert original.fits == [], "the caller's instance must never be fitted"
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


@pytest.mark.parametrize("strategy", ["last", "mean", "seasonal"])
@pytest.mark.parametrize("index,future", [(pd.RangeIndex(1), pd.RangeIndex(1, 3)),
                                         (pd.Index([5]), pd.Index([6, 9]))])
def test_last_baseline_supports_one_observation(strategy, index, future):
    forecast = Pipeline(NaiveForecaster(strategy)).predict(pd.DataFrame({"y": [3.0]}, index=index), future)
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
    from timegym.scenarios import CORE_KINDS, default_scenarios

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


@pytest.mark.parametrize('seed', [-1, 1.5, 'x', True, np.bool_(False)])
def test_ar1_rejects_invalid_seeds_at_construction(seed):
    with pytest.raises(ValueError, match='seed'):
        GaussianAR1Forecaster(seed=seed)


@pytest.mark.parametrize('seed', [None, 0, np.int64(19)])
def test_ar1_accepts_optional_nonnegative_integer_seeds(seed):
    assert GaussianAR1Forecaster(seed=seed).seed == seed


def test_sklearn_configuration_errors_fail_before_case_execution():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from timegym.models import SklearnForecaster

    for steps in ([], None, 'Ridge', [StandardScaler()]):
        with pytest.raises(ValueError, match='steps|Final step'):
            SklearnForecaster(steps)
    pipeline = Pipeline(window_length=4).add(StandardScaler())
    with pytest.raises(ValueError, match='Final step.*StandardScaler'):
        pipeline.run([Scenario('ar1')], seeds=[0])
    pipeline.add(Ridge())
    assert not any(d.status == 'ERROR' for d in pipeline.test(seeds=[0]).cases[0].diagnostics)


@pytest.mark.parametrize('levels', [(), (.5, .1), (.5, .5), (0, .5), (.5, 1), (np.nan,), (1j,), [[.1, .9]]])
def test_ar1_rejects_invalid_quantile_levels_at_construction(levels):
    with pytest.raises(ValueError, match='levels'):
        GaussianAR1Forecaster(levels=levels)


def test_ar1_custom_quantiles_use_the_returned_joint_samples():
    model = GaussianAR1Forecaster(n_samples=64, seed=7, levels=(.2, .8))
    case = Scenario('ar1', context_length=30, horizon=4, n_targets=2).generate()
    forecast = model.fit(case.task.context).predict(case.task.future_index)
    assert forecast.levels == (.2, .8)
    np.testing.assert_array_equal(forecast.quantiles,
                                  np.moveaxis(np.quantile(forecast.samples, model.levels, axis=0), 0, -1))
    assert forecast.validate(case.task) == []


# --- What a custom model's predict may return -------------------------------------------

class _LastRow:
    """The README's last-value forecaster: remember the last row, repeat it."""

    def fit(self, context):
        self.last = context.iloc[-1]
        return self

    def predict(self, future_index):
        return Forecast(future_index, [self.last] * len(future_index))


class _Returns:
    """Return whatever ``make(last, steps)`` builds from the last value of ``y``."""

    def __init__(self, make):
        self.make = make

    def fit(self, context):
        self.last = float(context.iloc[-1, 0])
        return self

    def predict(self, future_index):
        return self.make(self.last, len(future_index))


@pytest.mark.parametrize("make", [
    lambda last, steps: [last] * steps,
    lambda last, steps: tuple([last] * steps),
    lambda last, steps: np.full(steps, last),
    lambda last, steps: np.full((steps, 1), last),
    lambda last, steps: [[last]] * steps,
])
def test_lists_and_arrays_are_one_value_per_requested_step(make):
    case = Scenario("trend", context_length=20, horizon=3).generate(seed=0)
    forecast = Pipeline(_Returns(make)).predict(case.task.context, case.task.future_index)
    assert forecast.index.equals(case.task.future_index)
    assert forecast.target_names == ("y",)
    np.testing.assert_array_equal(forecast.point, np.full((3, 1), case.task.context.iloc[-1, 0]))
    assert forecast.validate(case.task) == []


def test_the_readme_forecaster_needs_no_numpy_or_target_names_for_several_series():
    scenarios = [Scenario("ar1", context_length=12, horizon=3, n_targets=2),
                 Scenario("trend", context_length=12, horizon=3)]
    report = Pipeline(_LastRow()).test(scenarios, instances=2)
    report.assert_passed()
    reference = Pipeline(NaiveForecaster()).test(scenarios, instances=2)
    assert ([case.metrics for case in report.cases]
            == [case.metrics for case in reference.cases])


def test_wrong_lengths_and_unusable_outputs_fail_with_their_reason():
    scenario = Scenario("trend", context_length=12, horizon=3)
    short = Pipeline(_Returns(lambda last, steps: [last] * (steps - 1))).test([scenario], instances=1)
    assert short.cases[0].diagnostics[0].name == "contract_point_shape"
    ragged = Pipeline(_Returns(lambda last, steps: [[last], [last, last], [last]])).test(
        [scenario], instances=1)
    assert ragged.cases[0].diagnostics[0].status == "FAIL"
    several = Pipeline(_Returns(lambda last, steps: [last] * steps)).test(
        [Scenario("ar1", context_length=12, horizon=3, n_targets=2)], instances=1)
    assert any(d.name == "contract_point_shape" for d in several.cases[0].diagnostics)
    scalar = Pipeline(_Returns(lambda last, steps: last)).test([scenario], instances=1)
    execution = scalar.cases[0].diagnostics[0]
    assert execution.name == "model_execution" and "one value per future step" in execution.message


def test_a_pandas_output_still_has_its_dates_checked():
    def shifted(last, steps):
        return pd.Series([last] * steps)  # labelled 0, 1, 2 instead of the requested steps
    report = Pipeline(_Returns(shifted)).test([Scenario("trend", context_length=12, horizon=3)],
                                              instances=1)
    assert any(d.name == "contract_index_mismatch" for d in report.cases[0].diagnostics)


def test_unnamed_forecast_series_are_the_tasks_and_named_ones_are_still_checked():
    case = Scenario("ar1", context_length=12, horizon=3, n_targets=2).generate(seed=0)
    point = np.zeros((3, 2))
    assert Forecast(case.task.future_index, point).validate(case.task) == []
    assert Forecast(case.task.future_index, np.zeros((3, 1))).validate() == []
    wrong = Forecast(case.task.future_index, point, target_names=("y1", "y0"))
    assert {item["code"] for item in wrong.validate(case.task)} == {"target_mismatch"}
    # Outside the pipeline a one-dimensional point is still a shape fault.
    one = Scenario("trend", context_length=12, horizon=3).generate(seed=0)
    flat = Forecast(one.task.future_index, np.zeros(3))
    assert "point_shape" in {item["code"] for item in flat.validate(one.task)}


def test_pipeline_predict_accepts_a_plain_list_or_array_as_the_past():
    model = Pipeline(_LastRow())
    forecast = model.predict([1.0, 2.0, 3.0], range(3, 6))
    assert list(forecast.index) == [3, 4, 5] and forecast.target_names == ("y",)
    np.testing.assert_array_equal(forecast.point, [[3.0], [3.0], [3.0]])
    forecast = model.predict(np.arange(6.0).reshape(3, 2), range(3, 5))
    assert forecast.target_names == (0, 1)
    np.testing.assert_array_equal(forecast.point, [[4.0, 5.0], [4.0, 5.0]])
    with pytest.raises(TypeError, match="context must be"):
        model.predict({"y": [1.0, 2.0]}, range(2, 4))


def test_last_value_forecaster_is_naive_last_with_nothing_to_set():
    import timegym
    from timegym import LastValueForecaster

    assert "LastValueForecaster" in timegym.__all__
    model = LastValueForecaster()
    assert model.params() == {} and type(model.clone_unfitted()) is LastValueForecaster
    case = Scenario("ar1", 40, 5, n_targets=2).generate(seed=6)
    named = model.fit(case.task.context).predict(case.task.future_index)
    naive = NaiveForecaster("last").fit(case.task.context).predict(case.task.future_index)
    np.testing.assert_array_equal(named.point, naive.point)
    assert named.target_names == naive.target_names and named.validate(case.task) == []
    with pytest.raises(TypeError):
        LastValueForecaster(strategy="mean")
