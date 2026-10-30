"""Non-Gaussian references preserve their law and diagnostic capabilities."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from timegym.contracts import SimulationReference
from timegym.diagnostics import covariance_dependence_check, evaluate_forecast
from timegym.scenarios import Case, Scenario


def bimodal_reference(**overrides):
    def sampler(count, seed):
        return np.random.default_rng(seed).choice([-2.0, 2.0], size=(count, 2, 2))

    parameters = dict(mean=np.zeros((2, 2)), variance=np.full((2, 2), 4.0),
                      covariance=np.eye(4) * 4, index=pd.RangeIndex(12, 14),
                      target_names=("y0", "y1"), sampler=sampler,
                      quantile_samples=2048, quantile_seed=71)
    parameters.update(overrides)
    return SimulationReference(**parameters)


def test_bimodal_quantiles_preserve_the_law_instead_of_gaussian_moments():
    reference = bimodal_reference()
    levels = (0.1, 0.9)
    actual = reference.quantile(levels)
    np.testing.assert_array_equal(actual, np.broadcast_to([-2.0, 2.0], (2, 2, 2)))
    assert not np.allclose(actual, 2 * norm.ppf(levels))
    forecast = reference.forecast(n_samples=128, seed=12, levels=levels)
    assert forecast.validate() == []
    np.testing.assert_array_equal(forecast.point, reference.mean)
    np.testing.assert_array_equal(forecast.quantiles, actual)
    assert set(np.unique(forecast.samples)) == {-2.0, 2.0}
    assert reference.conditioning["quantiles_kind"] == "monte_carlo"
    assert reference.conditioning["moments_kind"] == "analytic"


def test_sampling_and_empirical_quantiles_are_replayable_without_global_rng():
    reference = bimodal_reference()
    state = deepcopy(np.random.get_state())
    first = reference.sample(100, seed=11)
    reference.sample(20, seed=17)
    np.testing.assert_array_equal(first, reference.sample(100, seed=11))
    assert not np.array_equal(first, reference.sample(100, seed=12))
    np.testing.assert_array_equal(reference.quantile(), reference.quantile())
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_conditioning_metadata_is_owned_without_mutating_caller_or_callable():
    supplied = {"observed_through": 11, "nested": {"source": "history"}}
    original = deepcopy(supplied)
    reference = bimodal_reference(conditioning=supplied)
    assert supplied == original
    reference.conditioning["nested"]["source"] = "changed"
    assert supplied == original
    capabilities = reference.capabilities
    capabilities["sampler_kind"] = "changed"
    assert reference.sampler_kind == "exact_conditional"
    assert reference.conditioning["quantile_samples"] == 2048
    assert reference.conditioning["quantile_seed"] == 71
    assert not hasattr(reference.sampler, "quantile_seed")


@pytest.mark.parametrize("sampler,match", [
    (lambda n, seed: np.zeros((n, 2)), "shape"),
    (lambda n, seed: np.zeros((n + 1, 2, 2)), "shape"),
    (lambda n, seed: np.full((n, 2, 2), np.nan), "finite"),
    (lambda n, seed: np.ones((n, 2, 2), dtype=complex), "real-valued"),
])
def test_invalid_simulator_outputs_are_rejected(sampler, match):
    reference = bimodal_reference(sampler=sampler)
    with pytest.raises(ValueError, match=match):
        reference.sample(8, seed=3)
    with pytest.raises(ValueError, match=match):
        reference.quantile()


@pytest.mark.parametrize("count,seed", [(0, 1), (True, 1), (1.5, 1), (8, -1), (8, True), (8, 1.5)])
def test_sample_rejects_invalid_count_or_seed_before_invoking_simulator(count, seed):
    def should_not_run(*args):
        raise AssertionError("invalid inputs reached the simulator")

    reference = bimodal_reference(sampler=should_not_run)
    with pytest.raises(ValueError, match="integer"):
        reference.sample(count, seed)


@pytest.mark.parametrize("overrides,exception,match", [
    ({"sampler": None}, TypeError, "callable"),
    ({"moments_kind": "monte_carlo"}, ValueError, "analytical moments"),
    ({"sampler_kind": "gaussian_assumption"}, ValueError, "sampler_kind"),
    ({"quantile_samples": 0}, ValueError, "quantile_samples"),
    ({"quantile_seed": -1}, ValueError, "quantile_seed"),
    ({"covariance": -np.eye(4)}, ValueError, "semidefinite"),
])
def test_capabilities_and_moments_are_validated(overrides, exception, match):
    with pytest.raises(exception, match=match):
        bimodal_reference(**overrides)


@pytest.mark.parametrize("levels", [(), (0.1, 0.1), (0.9, 0.1), (0, 1), (float("nan"),), (0.5 + 1j,)])
def test_empirical_quantiles_keep_the_probability_contract(levels):
    with pytest.raises(ValueError):
        bimodal_reference().quantile(levels)


def test_approximate_sampler_is_not_used_as_an_exact_diagnostic_null():
    def should_not_run(*args):
        raise AssertionError("approximate sampler used for null calibration")

    reference = bimodal_reference(sampler=should_not_run, sampler_kind="approximate_conditional")
    result = covariance_dependence_check(reference, np.zeros((64, 2, 2)))
    assert result.status == "UNSUPPORTED"
    assert "approximate" in result.message
    assert result.evidence["reference"]["sampler_kind"] == "approximate_conditional"
    assert "p_value" not in result.evidence


def test_exact_non_gaussian_sampler_calibrates_and_detects_collapsed_variance():
    reference = bimodal_reference()
    result = covariance_dependence_check(reference, np.zeros((256, 2, 2)),
                                         seed=34, n_calibration=19)
    assert result.status == "FAIL"
    assert result.evidence["p_value"] == 0.05
    assert result.evidence["reference"]["sampler_kind"] == "exact_conditional"
    assert result.to_dict() == covariance_dependence_check(
        reference, np.zeros((256, 2, 2)), seed=34, n_calibration=19).to_dict()


def test_non_gaussian_reference_evidence_is_explicit_without_changing_point_scores():
    baseline = Scenario("white_noise", context_length=12, horizon=2,
                        n_targets=2).generate(seed=6)
    reference = bimodal_reference()
    case = Case(baseline.task, baseline.observed, reference, baseline.manifest)
    result = evaluate_forecast(case, reference.forecast(n_samples=32, seed=5))
    point = next(d for d in result.diagnostics if d.name == "point_accuracy")
    assert point.status == "INCONCLUSIVE"
    assert point.evidence["reference"]["kind"] == "analytic_moments_simulation"
    assert point.evidence["reference"]["quantiles_kind"] == "monte_carlo"
    assert result.metrics["mean_reference_mse"] == 0
    assert not any(d.status == "ERROR" for d in result.diagnostics)
    old_result = evaluate_forecast(baseline, baseline.reference.forecast(n_samples=32, seed=5))
    old_point = next(d for d in old_result.diagnostics if d.name == "point_accuracy")
    assert "reference" not in old_point.evidence
