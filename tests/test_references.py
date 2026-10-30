import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from timegym.scenarios import Scenario


KINDS = ("white_noise", "trend", "seasonal", "ar1", "random_walk")


def assert_monte_carlo_moments(samples, mean, covariance):
    """Check simulation against six-standard-error Gaussian sampling bounds."""
    flat = samples.reshape(samples.shape[0], -1)
    n = len(flat)
    diagonal = np.diag(covariance)
    mean_error = np.abs(flat.mean(axis=0) - mean.reshape(-1))
    assert np.all(mean_error <= 6 * np.sqrt(diagonal / n) + 1e-10)
    covariance_error = np.abs(np.cov(flat, rowvar=False) - covariance)
    covariance_se = np.sqrt((np.outer(diagonal, diagonal) + covariance ** 2) / (n - 1))
    assert np.all(covariance_error <= 6 * covariance_se + 1e-10)


@pytest.mark.parametrize("kind", KINDS)
def test_reproducibility_manifest_round_trip_and_no_global_rng(kind):
    state_before = np.random.get_state()
    scenario = Scenario(kind, context_length=20, horizon=4, n_targets=2, parameters={"correlation": 0.4}, seed=7)
    case = scenario.generate()
    same = scenario.generate()
    pd.testing.assert_frame_equal(case.task.context, same.task.context)
    pd.testing.assert_frame_equal(case.observed, same.observed)
    assert case.reference.forecast(seed=13).validate(case.task) == []
    assert case.reference.kind == "analytic_conditional"
    assert case.reference.conditioning["future_observations_used"] is False
    manifest = json.loads(json.dumps(case.manifest))
    recreated = Scenario(**{key: manifest[key] for key in ("kind", "context_length", "horizon", "parameters", "n_targets", "seed")}).generate()
    pd.testing.assert_frame_equal(case.task.context, recreated.task.context)
    pd.testing.assert_frame_equal(case.observed, recreated.observed)
    altered = scenario.generate(seed=8)
    assert not altered.task.context.equals(case.task.context)
    assert scenario.seed == 7
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


@pytest.mark.parametrize("kind,coefficient", [("ar1", 0.65), ("ar1", -0.45), ("random_walk", 1.0)])
def test_conditional_mean_and_full_covariance_against_independent_simulation(kind, coefficient):
    parameters = {"sigma": [0.7, 1.4], "correlation": 0.35}
    if kind == "ar1":
        parameters.update(mean=[2.0, -1.0], phi=coefficient)
    else:
        parameters.update(intercept=[2.0, -1.0], drift=[0.3, -0.2])
    case = Scenario(kind, context_length=25, horizon=4, n_targets=2, parameters=parameters, seed=21).generate()
    last = case.task.context.iloc[-1].to_numpy()
    rng = np.random.default_rng(90210)
    innovation_covariance = np.array([[0.7 ** 2, 0.35 * 0.7 * 1.4], [0.35 * 0.7 * 1.4, 1.4 ** 2]])
    n = 60_000
    innovations = rng.multivariate_normal(np.zeros(2), innovation_covariance, size=(n, 4))
    simulations = np.empty((n, 4, 2))
    previous = np.broadcast_to(last, (n, 2)).copy()
    for h in range(4):
        if kind == "ar1":
            previous = np.array([2.0, -1.0]) + coefficient * (previous - [2.0, -1.0]) + innovations[:, h]
        else:
            previous = previous + [0.3, -0.2] + innovations[:, h]
        simulations[:, h] = previous
    assert_monte_carlo_moments(simulations, case.reference.mean, case.reference.covariance)
    if kind == "ar1":
        expected_first = [2.0, -1.0] + coefficient * (last - [2.0, -1.0])
    else:
        expected_first = last + [0.3, -0.2]
    np.testing.assert_allclose(case.reference.mean[0], expected_first)
    np.testing.assert_allclose(case.reference.covariance[:2, :2], innovation_covariance)
    np.testing.assert_array_equal(case.reference.conditioning["last_observed"], last)


@pytest.mark.parametrize("kind", KINDS)
def test_reference_joint_samples_have_correct_conditional_moments(kind):
    case = Scenario(kind, context_length=20, horizon=3, n_targets=2, parameters={"correlation": -0.4}, seed=15).generate()
    samples = case.reference.sample(40_000, seed=42)
    assert samples.shape == (40_000, 3, 2)
    assert_monte_carlo_moments(samples, case.reference.mean, case.reference.covariance)
    np.testing.assert_array_equal(case.reference.sample(8, seed=1), case.reference.sample(8, seed=1))
    assert not np.array_equal(case.reference.sample(8, seed=1), case.reference.sample(8, seed=2))


@pytest.mark.parametrize("kind", KINDS)
def test_history_and_first_reference_do_not_depend_on_held_out_horizon(kind):
    short = Scenario(kind, context_length=18, horizon=1, seed=19).generate()
    long = Scenario(kind, context_length=18, horizon=8, seed=19).generate()
    pd.testing.assert_frame_equal(short.task.context, long.task.context)
    np.testing.assert_array_equal(short.reference.mean, long.reference.mean[:1])
    np.testing.assert_array_equal(short.reference.variance, long.reference.variance[:1])
    assert short.task.future_index[0] == short.task.context.index[-1] + 1


def test_known_signal_scenarios_and_observation_noise():
    trend = Scenario("trend", context_length=5, horizon=3, parameters={"intercept": 2, "slope": 0.5, "sigma": 0.3}).generate()
    np.testing.assert_allclose(trend.reference.mean[:, 0], [4.5, 5.0, 5.5])
    np.testing.assert_allclose(trend.reference.covariance, 0.09 * np.eye(3))
    seasonal = Scenario("seasonal", context_length=5, horizon=3, parameters={"mean": 1, "amplitude": 2, "period": 4}).generate()
    np.testing.assert_allclose(seasonal.reference.mean[:, 0], [3.0, 1.0, -1.0], atol=1e-14)
    white = Scenario("white_noise", context_length=5, horizon=3, parameters={"mean": 3, "sigma": 2}).generate()
    np.testing.assert_array_equal(white.reference.mean, np.full((3, 1), 3.0))
    np.testing.assert_array_equal(white.reference.covariance, 4 * np.eye(3))


def test_gaussian_quantiles_and_zero_variance_reference():
    reference = Scenario("random_walk", horizon=3, parameters={"sigma": 2.0}).generate().reference
    levels = (0.05, 0.5, 0.95)
    quantiles = reference.quantile(levels)
    assert quantiles.shape == (3, 1, 3)
    np.testing.assert_array_equal(quantiles[:, :, 1], reference.mean)
    np.testing.assert_allclose(quantiles[:, :, 2], reference.mean + np.sqrt(reference.variance) * norm.ppf(0.95))
    deterministic = Scenario("ar1", parameters={"sigma": 0, "mean": 4}).generate().reference
    np.testing.assert_array_equal(deterministic.sample(5), np.broadcast_to(deterministic.mean, (5, 12, 1)))
    np.testing.assert_array_equal(deterministic.quantile(levels), np.repeat(deterministic.mean[:, :, None], 3, axis=-1))


def test_singular_cross_target_correlation_is_supported():
    case = Scenario("random_walk", context_length=10, horizon=3, n_targets=2, parameters={"correlation": 1.0}, seed=7).generate()
    np.testing.assert_allclose(case.task.context.iloc[:, 0], case.task.context.iloc[:, 1], atol=1e-14)
    samples = case.reference.sample(20)
    np.testing.assert_allclose(samples[:, :, 0], samples[:, :, 1], atol=1e-7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "unknown"},
        {"context_length": 0},
        {"horizon": -1},
        {"horizon": 1.5},
        {"n_targets": True},
        {"seed": -1},
        {"parameters": {"sigma": -1}},
        {"parameters": {"sigma": np.nan}},
        {"parameters": {"sigma": [1, 2]}},
        {"parameters": {"typo": 1}},
        {"n_targets": 3, "parameters": {"correlation": -0.6}},
        {"n_targets": 2, "parameters": {"correlation": [[1, 2], [2, 1]]}},
        {"n_targets": 2, "parameters": {"correlation": [[1, 0], [0.4, 1]]}},
        {"kind": "ar1", "parameters": {"phi": 1}},
        {"kind": "ar1", "parameters": {"phi": [0.2]}},
        {"kind": "seasonal", "parameters": {"period": 0}},
    ],
)
def test_invalid_scenario_parameters_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Scenario(**{"kind": "white_noise", **kwargs})


def test_invalid_reference_sampling_and_quantile_requests():
    reference = Scenario("white_noise").generate().reference
    for count in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            reference.sample(count)
    for levels in ((), (0, 0.5), (0.5, 1), (0.5, 0.5), (0.9, 0.1), (np.nan,)):
        with pytest.raises(ValueError):
            reference.quantile(levels)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "kind,parameter,value,dimensions",
    [
        ("white_noise", "mean", np.complex128(1 + 2j), 1),
        ("white_noise", "sigma", np.complex128(1 + 2j), 1),
        ("white_noise", "mean", np.array([np.complex128(1 + 2j)], dtype=object), 1),
        ("white_noise", "correlation", np.complex128(0.2 + 0.1j), 2),
        ("white_noise", "correlation", np.array([[1, 0.2j], [-0.2j, 1]], dtype=object), 2),
        ("trend", "slope", np.complex128(0.1 + 0.1j), 1),
        ("ar1", "phi", np.complex128(0.5 + 0.1j), 1),
        ("seasonal", "period", np.complex128(12 + 1j), 1),
        ("random_walk", "drift", np.array([np.complex128(1 + 2j), 0], dtype=object), 2),
    ],
)
def test_complex_scenario_parameters_are_rejected_before_conversion(kind, parameter, value, dimensions):
    with pytest.raises(ValueError, match="real"):
        Scenario(kind, n_targets=dimensions, parameters={parameter: value})
