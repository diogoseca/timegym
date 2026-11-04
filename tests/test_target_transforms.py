"""Reversible target transforms and their Pipeline integration."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from timegym import Forecast, ForecastTask, GaussianAR1Forecaster, NaiveForecaster, Pipeline, Scenario, TestReport
from timegym.diagnostics import evaluate_forecast
from timegym.mutations import mutate_forecast
from timegym.target import (BoxCoxTransform, ComposedTransform, DifferenceTransform, LogTransform,
                            differencing, transform_config, transform_from_config)

HORIZON = 6


def positive_series(length=60, targets=1, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(length)[:, None]
    values = np.exp(1 + .02 * time + .3 * np.sin(2 * np.pi * time / 12 + np.arange(targets))
                    + rng.normal(0, .05, (length, targets)))
    return pd.DataFrame(values, columns=[f"y{d}" for d in range(targets)])


def split(full):
    return full.iloc[:-HORIZON], full.iloc[-HORIZON:]


def forward(transform, full):
    """Apply a transform fitted on the context to the whole series, as an independent oracle."""
    if isinstance(transform, ComposedTransform):
        for member in transform.transforms:
            full = forward(member, full)
        return full
    if isinstance(transform, LogTransform):
        return np.log(full)
    if isinstance(transform, BoxCoxTransform):
        return pd.DataFrame(np.column_stack([stats.boxcox(full.iloc[:, d], lmbda)
                                             for d, lmbda in enumerate(transform.lambdas_)]),
                            index=full.index, columns=full.columns)
    return differencing(full, transform.order, lag=transform.lag)


def exact_forecast(values, levels=(.1, .5, .9), samples=4):
    """A degenerate forecast whose every output equals the supplied future."""
    values = np.asarray(values, dtype=float)
    return Forecast(index=pd.RangeIndex(60 - HORIZON, 60), point=values.copy(),
                    quantiles=np.repeat(values[:, :, None], len(levels), axis=2), levels=levels,
                    samples=np.repeat(values[None], samples, axis=0),
                    target_names=tuple(f"y{d}" for d in range(values.shape[1])))


TRANSFORMS = [LogTransform(), BoxCoxTransform(), DifferenceTransform(), DifferenceTransform(lag=12),
              DifferenceTransform(order=2, lag=3),
              ComposedTransform([LogTransform(), DifferenceTransform(lag=12)]),
              ComposedTransform([BoxCoxTransform(), DifferenceTransform(order=2)])]


@pytest.mark.parametrize("transform", TRANSFORMS, ids=lambda t: json.dumps(transform_config(t)))
@pytest.mark.parametrize("targets", [1, 2])
def test_identity_round_trip_recovers_the_original_future(transform, targets):
    full = positive_series(targets=targets)
    context, future = split(full)
    transform = transform.clone_unfitted()
    transformed = transform.fit_transform(context)
    oracle = forward(transform, full)
    pd.testing.assert_frame_equal(transformed, oracle.iloc[:len(transformed)], check_exact=False, rtol=1e-12)
    restored = transform.inverse_transform(exact_forecast(oracle.iloc[-HORIZON:]))
    for output in (restored.point, restored.samples[0], restored.samples[-1], restored.quantiles[..., 1]):
        np.testing.assert_allclose(output, future.to_numpy(), rtol=1e-10)
    assert restored.levels == (.1, .5, .9) and restored.target_names == tuple(full.columns)


def test_differencing_inverts_known_series_exactly():
    time = np.arange(48.)
    quadratic = pd.DataFrame({"y": time ** 2})
    transform = DifferenceTransform(order=2)
    np.testing.assert_array_equal(transform.fit_transform(quadratic.iloc[:40]).to_numpy(), 2.)
    restored = transform.inverse_transform(Forecast(index=pd.RangeIndex(40, 48), point=np.full((8, 1), 2.)))
    np.testing.assert_array_equal(restored.point[:, 0], time[40:] ** 2)

    pattern = np.tile([3., -1., 4., 1., -5., 9.], 8)
    seasonal = pd.DataFrame({"y": pattern + .5 * time})
    transform = DifferenceTransform(lag=6)
    np.testing.assert_allclose(transform.fit_transform(seasonal.iloc[:36]).to_numpy(), 3.)
    restored = transform.inverse_transform(Forecast(index=pd.RangeIndex(36, 48), point=np.full((12, 1), 3.)))
    np.testing.assert_allclose(restored.point, seasonal.iloc[36:].to_numpy())


def test_two_composed_first_differences_equal_one_second_difference():
    full = positive_series()
    context, _ = split(full)
    composed = ComposedTransform([DifferenceTransform(), DifferenceTransform()])
    second = DifferenceTransform(order=2)
    pd.testing.assert_frame_equal(composed.fit_transform(context), second.fit_transform(context))
    increments = np.random.default_rng(3).normal(size=(50, HORIZON, 1))
    forecast = Forecast(index=pd.RangeIndex(54, 60), point=increments.mean(axis=0), samples=increments,
                        quantiles=np.quantile(increments, [.2, .8], axis=0).transpose(1, 2, 0), levels=(.2, .8),
                        target_names=("y0",))
    np.testing.assert_allclose(composed.inverse_transform(forecast).samples, second.inverse_transform(forecast).samples)


@pytest.mark.parametrize("transform", [LogTransform(), BoxCoxTransform()], ids=["log", "box_cox"])
def test_monotone_inverses_keep_quantiles_ordered_and_elementwise(transform):
    context, future = split(positive_series(targets=2))
    model = GaussianAR1Forecaster(n_samples=200, seed=4, levels=(.05, .25, .5, .75, .95))
    transformed = transform.fit_transform(context)
    forecast = model.fit(transformed).predict(future.index)
    restored = transform.inverse_transform(forecast)
    assert np.all(np.diff(restored.quantiles, axis=-1) > 0)
    reference = transform.clone_unfitted()
    reference.fit_transform(context)
    for level in range(5):
        single = Forecast(index=future.index, quantiles=forecast.quantiles[..., [level]], levels=(.5,),
                          target_names=forecast.target_names)
        np.testing.assert_allclose(restored.quantiles[..., level], reference.inverse_transform(single).quantiles[..., 0])
    assert restored.validate(ForecastTask(context, future.index)) == []


def test_differencing_recomputes_ordered_quantiles_from_integrated_samples():
    context, future = split(positive_series())
    transform = DifferenceTransform(lag=12)
    forecast = GaussianAR1Forecaster(n_samples=300, seed=1).fit(transform.fit_transform(context)).predict(future.index)
    restored = transform.inverse_transform(forecast)
    np.testing.assert_array_equal(restored.quantiles,
                                  np.moveaxis(np.quantile(restored.samples, forecast.levels, axis=0), 0, -1))
    assert np.all(np.diff(restored.quantiles, axis=-1) >= 0)
    np.testing.assert_allclose(restored.point, restored.samples.mean(axis=0), rtol=0, atol=.2)


def test_the_point_of_a_nonlinear_inverse_is_never_the_inverted_mean():
    context, future = split(positive_series())
    transform = LogTransform()
    transform.fit_transform(context)
    draws = np.random.default_rng(7).normal(4., .5, size=(4000, HORIZON, 1))
    quantiles = np.moveaxis(np.quantile(draws, [.1, .5, .9], axis=0), 0, -1)
    full = Forecast(index=future.index, point=draws.mean(axis=0), quantiles=quantiles, levels=(.1, .5, .9),
                    samples=draws, target_names=("y0",))
    restored = transform.inverse_transform(full)
    np.testing.assert_array_equal(restored.point, np.exp(draws).mean(axis=0))
    np.testing.assert_allclose(restored.point, np.exp(4 + .5 ** 2 / 2), rtol=.03)
    assert np.all(restored.point > 1.1 * np.exp(full.point))

    without_samples = Forecast(index=future.index, point=full.point, quantiles=quantiles, levels=(.1, .5, .9),
                               target_names=("y0",))
    np.testing.assert_array_equal(transform.inverse_transform(without_samples).point, np.exp(quantiles[..., 1]))
    point_only = Forecast(index=future.index, point=full.point, target_names=("y0",))
    np.testing.assert_array_equal(transform.inverse_transform(point_only).point, np.exp(full.point))
    assert transform.inverse_transform(Forecast(index=future.index, samples=draws, target_names=("y0",))).point is None


@pytest.mark.parametrize("transform,name", [(LogTransform(), "LogTransform"), (BoxCoxTransform(), "BoxCoxTransform")])
@pytest.mark.parametrize("bad", [0., -2.5])
def test_nonpositive_contexts_are_rejected_precisely(transform, name, bad):
    context = pd.DataFrame({"a": [1., 2., 3., 4., 5.], "b": [1., 2., 3., bad, bad]}, index=pd.RangeIndex(10, 15))
    with pytest.raises(ValueError, match=rf"{name} requires strictly positive context values; target 'b' "
                                         r"has 2 nonpositive value\(s\), the first at index 13"):
        transform.fit_transform(context)


def test_box_cox_lambda_is_fitted_on_the_context_only():
    full = positive_series(length=80, targets=2)
    context, _ = split(full)
    transform = BoxCoxTransform()
    transform.fit_transform(context)
    expected = [stats.boxcox_normmax(context.iloc[:, d].to_numpy(), method="mle") for d in range(2)]
    assert all(-1 < value < 2 for value in expected)
    np.testing.assert_allclose(transform.lambdas_, expected, rtol=0, atol=1e-4)
    shifted = transform.clone_unfitted()
    shifted.fit_transform(full.iloc[:-HORIZON - 1])
    assert not np.allclose(shifted.lambdas_, transform.lambdas_)
    assert not hasattr(transform.clone_unfitted(), "lambdas_")


def test_box_cox_lambda_stays_within_reciprocal_and_square():
    prices = Scenario("merton_jump_diffusion", context_length=50, horizon=4,
                      parameters={"output": "price"}).generate(seed=2).task.context
    assert stats.boxcox_normmax(prices.iloc[:, 0].to_numpy(), method="mle") < -5
    transform = BoxCoxTransform()
    transform.fit_transform(prices)
    np.testing.assert_allclose(transform.lambdas_, [-1.], atol=1e-4)
    fourth_roots = pd.DataFrame({"y": (10. + 3. * stats.norm.ppf(np.linspace(.01, .99, 200))) ** .25})
    assert stats.boxcox_normmax(fourth_roots["y"].to_numpy(), method="mle") > 3
    transform.fit_transform(fourth_roots)
    np.testing.assert_allclose(transform.lambdas_, [2.], atol=1e-4)


def test_box_cox_rejects_constant_contexts_and_values_without_a_preimage():
    with pytest.raises(ValueError, match="not identified for target 'y'.*constant"):
        BoxCoxTransform().fit_transform(pd.DataFrame({"y": np.full(10, 3.)}))
    context, future = split(positive_series())
    transform = BoxCoxTransform()
    transform.fit_transform(context)
    lam = transform.lambdas_[0]
    impossible = np.full((HORIZON, 1), -2. / lam)
    with pytest.raises(ValueError, match=r"undefined for 6 forecast value\(s\)"):
        transform.inverse_transform(Forecast(index=future.index, point=impossible, target_names=("y0",)))


def test_differencing_without_samples_keeps_points_and_omits_unknowable_quantiles():
    context, future = split(positive_series())
    transform = DifferenceTransform()
    transform.fit_transform(context)
    increments = np.full((HORIZON, 1), .5)
    quantiles = increments[..., None] + np.array([-1., 0., 1.])
    restored = transform.inverse_transform(Forecast(index=future.index, point=increments, quantiles=quantiles,
                                                    levels=(.1, .5, .9), target_names=("y0",)))
    np.testing.assert_allclose(restored.point[:, 0], context.iloc[-1, 0] + .5 * np.arange(1, HORIZON + 1))
    assert restored.quantiles is None and restored.levels == ()
    assert restored.validate(ForecastTask(context, future.index)) == []
    with pytest.raises(ValueError, match="Quantiles of differences do not determine quantiles of levels"):
        transform.inverse_transform(Forecast(index=future.index, quantiles=quantiles, levels=(.1, .5, .9),
                                             target_names=("y0",)))


def test_inversion_requires_a_fit_and_a_valid_matching_forecast():
    context, future = split(positive_series())
    forecast = Forecast(index=future.index, point=np.ones((HORIZON, 1)), target_names=("y0",))
    with pytest.raises(ValueError, match="fit_transform"):
        DifferenceTransform().inverse_transform(forecast)
    transform = LogTransform()
    transform.fit_transform(context)
    with pytest.raises(TypeError, match="Forecast"):
        transform.inverse_transform(forecast.point)
    crossing = Forecast(index=future.index, quantiles=np.ones((HORIZON, 1, 2)) * [[[1., 0.]]], levels=(.1, .9),
                        target_names=("y0",))
    with pytest.raises(ValueError, match="violates its contract: quantile_crossing"):
        transform.inverse_transform(crossing)
    with pytest.raises(ValueError, match="targets or their order"):
        transform.inverse_transform(Forecast(index=future.index, point=np.ones((HORIZON, 1)), target_names=("x",)))


def test_constructor_and_context_validation():
    for order, lag in [(0, 1), (1, 0), (1.5, 1), (True, 1)]:
        with pytest.raises(ValueError, match="order|lag"):
            DifferenceTransform(order=order, lag=lag)
    with pytest.raises(ValueError, match="needs more than 12 context rows"):
        DifferenceTransform(lag=12).fit_transform(pd.DataFrame({"y": np.arange(12.)}))
    with pytest.raises(TypeError, match="DataFrame"):
        LogTransform().fit_transform(pd.Series([1., 2.]))
    with pytest.raises(ValueError, match="finite"):
        DifferenceTransform().fit_transform(pd.DataFrame({"y": [1., np.nan, 3.]}))
    for members in ([], "log", None):
        with pytest.raises(ValueError, match="nonempty sequence"):
            ComposedTransform(members)
    for member in (ComposedTransform([LogTransform()]), NaiveForecaster()):
        with pytest.raises(TypeError, match="members must be"):
            ComposedTransform([LogTransform(), member])


def test_composed_members_are_copied_unfitted_and_never_aliased():
    shared = DifferenceTransform()
    composed = ComposedTransform([shared, shared])
    assert composed.transforms[0] is not composed.transforms[1] and shared not in composed.transforms
    context, _ = split(positive_series())
    composed.fit_transform(context)
    assert not hasattr(shared, "tails_")
    assert composed.transforms[0].tails_[0].tolist() != composed.transforms[1].tails_[0].tolist()
    clone = composed.clone_unfitted()
    assert not any(hasattr(member, "tails_") for member in clone.transforms)


@pytest.mark.parametrize("transform", TRANSFORMS, ids=lambda t: json.dumps(transform_config(t)))
def test_bundled_transform_configuration_round_trips(transform):
    config = json.loads(json.dumps(transform_config(transform)))
    assert config["kind"] == "builtin"
    assert transform_config(transform_from_config(config)) == config


def test_transform_descriptions_are_allowlisted_and_never_imported():
    class Custom(LogTransform):
        pass
    assert transform_config(Custom())["kind"] == "external"
    assert "Custom" in transform_config(Custom())["qualname"]
    valid = transform_config(ComposedTransform([LogTransform(), DifferenceTransform(lag=4)]))
    invalid = []
    for path, value in [(("kind",), "external"), (("class",), "os.system"), (("class",), ["LogTransform"]),
                        (("parameters",), {"transforms": []}), (("parameters",), {"transforms": "LogTransform"}),
                        (("parameters", "transforms", 1, "parameters", "lag"), 0),
                        (("parameters", "transforms", 1, "parameters", "callback"), "print"),
                        (("parameters", "transforms", 0), valid)]:
        changed = deepcopy(valid)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        invalid.append(changed)
    invalid.append({**valid, "module": "os"})
    for config in invalid:
        with pytest.raises(ValueError):
            transform_from_config(config)


def recorded_fit_contexts():
    contexts = []

    class Recorder(GaussianAR1Forecaster):
        def fit(self, context):
            contexts.append(context.copy())
            return super().fit(context)

    return contexts, Recorder


def test_pipeline_fits_the_model_on_the_transformed_context_and_inverts_its_forecast():
    case = Scenario("merton_jump_diffusion", context_length=80, horizon=8,
                    parameters={"output": "price", "jump_rate": 2.}).generate(seed=3)
    transform = ComposedTransform([LogTransform(), DifferenceTransform()])
    contexts, Recorder = recorded_fit_contexts()
    pipeline = Pipeline(model_factory=lambda: Recorder(n_samples=64, seed=9), target_transform=transform)
    forecast = pipeline.predict(case.task.context, case.task.future_index)

    manual = transform.clone_unfitted()
    transformed = manual.fit_transform(case.task.context)
    pd.testing.assert_frame_equal(contexts[0], transformed)
    expected = manual.inverse_transform(GaussianAR1Forecaster(n_samples=64, seed=9).fit(transformed)
                                        .predict(case.task.future_index))
    for name in ("point", "quantiles", "samples"):
        np.testing.assert_array_equal(getattr(forecast, name), getattr(expected, name))
    assert not hasattr(transform.transforms[0], "target_names_")
    assert forecast.validate(case.task) == []
    statuses = {d.name: d.status for d in evaluate_forecast(case, forecast).diagnostics}
    assert statuses["forecast_contract"] == "PASS"


def test_contract_violations_reach_the_evaluator_uninverted():
    class Crossing(GaussianAR1Forecaster):
        def predict(self, future_index):
            return mutate_forecast(super().predict(future_index), "quantile_crossing")

    scenario = Scenario("merton_jump_diffusion", context_length=60, horizon=5, parameters={"output": "price"})
    pipeline = Pipeline(model_factory=lambda: Crossing(n_samples=32, seed=2), target_transform=DifferenceTransform())
    case = scenario.generate(seed=1)
    raw = pipeline.predict(case.task.context, case.task.future_index)
    assert np.any(np.diff(raw.quantiles, axis=-1) < 0)
    report = pipeline.test([scenario], seeds=[1])
    statuses = {d.name: d.status for d in report.cases[0].diagnostics}
    assert statuses["contract_quantile_crossing"] == "FAIL"
    assert "model_execution" not in statuses


def test_target_transform_is_recorded_and_replayed_from_the_report(tmp_path):
    transform = ComposedTransform([BoxCoxTransform(), DifferenceTransform(lag=5)])
    pipeline = Pipeline(GaussianAR1Forecaster(n_samples=40, seed=6), target_transform=transform)
    config = json.loads(json.dumps(pipeline.to_config()))
    assert config["schema_version"] == 3 and config["replayable"] is True
    assert config["target_transform"] == transform_config(transform)
    restored = Pipeline.from_config(config)
    assert type(restored.target_transform) is ComposedTransform and restored.to_config() == config

    scenario = Scenario("merton_jump_diffusion", context_length=50, horizon=4, parameters={"output": "price"})
    run = pipeline.run([scenario], seeds=[2, 7])
    assert run.metadata["target_transform"] == config["target_transform"]
    assert run.metadata["pipeline_config"] == config
    assert all(d.status != "ERROR" for entry in run.cases for d in entry.result.diagnostics)
    path = tmp_path / "report.json"
    run.report.to_json(path)
    replayed = TestReport.read(path).replay_case(1)
    assert replayed.metadata["replay"]["reproduced"]
    assert replayed.cases[0].to_dict() == run.report.cases[1].to_dict()


def test_schema_two_configurations_still_restore_without_a_transform():
    config = Pipeline(NaiveForecaster("drift")).to_config()
    assert config["target_transform"] is None
    legacy = {key: value for key, value in config.items() if key != "target_transform"}
    legacy["schema_version"] = 2
    assert Pipeline.from_config(legacy).target_transform is None
    with pytest.raises(ValueError, match="configuration schema"):
        Pipeline.from_config({**legacy, "target_transform": None})
    with pytest.raises(ValueError, match="configuration schema"):
        Pipeline.from_config({key: value for key, value in config.items() if key != "target_transform"})


def test_external_target_transforms_are_described_but_not_replayable():
    class Shift:
        def clone_unfitted(self):
            return Shift()

        def fit_transform(self, context):
            return context - 1.

        def inverse_transform(self, forecast):
            return Forecast(index=forecast.index, point=forecast.point + 1., target_names=forecast.target_names)

    pipeline = Pipeline(NaiveForecaster(), target_transform=Shift())
    case = Scenario("trend").generate(seed=0)
    np.testing.assert_allclose(pipeline.predict(case.task.context, case.task.future_index).point,
                               Pipeline(NaiveForecaster()).predict(case.task.context, case.task.future_index).point)
    config = pipeline.to_config()
    assert config["replayable"] is False and config["target_transform"]["kind"] == "external"
    assert config["target_transform"]["qualname"].endswith("Shift")
    with pytest.raises(ValueError, match="not automatically replayable"):
        Pipeline.from_config(config)


def test_pipeline_rejects_target_transforms_without_the_protocol():
    for invalid in (LogTransform, object(), "log"):
        with pytest.raises(TypeError, match="target_transform must be an instance"):
            Pipeline(target_transform=invalid)

    class Shared(LogTransform):
        def clone_unfitted(self):
            return self

    with pytest.raises(ValueError, match="new target transform instance"):
        Pipeline(target_transform=Shared()).predict(pd.DataFrame({"y": [1., 2., 3.]}), pd.RangeIndex(3, 5))
