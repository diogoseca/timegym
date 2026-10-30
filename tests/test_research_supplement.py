"""Regression checks for selected research fixtures and inference accounting.

No assertion requires a random clean statistical sample to pass. These tests
exercise algebraic specifications, rank decisions, paired records, and retained
misses; they do not fit thresholds or select evaluation scenarios.
"""
from copy import deepcopy
import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.model_faults import (FAULTS, LagRidgeImplementation,
                                     normal_equation_forecast, point_agreement,
                                     target_shift_agreement)
from experiments.run_supplement import (PROTOCOL, conformal_pvalue,
                                       conformal_threshold, holm_rejections,
                                       run_covariance_study, run_model_study,
                                       validate_protocol)
from timegym import Pipeline, Scenario
from timegym.checks import check_shift_equivariance


def fixture_case():
    return Scenario("ar1", context_length=72, horizon=7, n_targets=2,
                    parameters={"phi": .65, "mean": [5., -3.],
                                "sigma": [1., .7], "correlation": .4}).generate(seed=931)


def test_independent_normal_equations_agree_with_two_separate_implementations():
    case = fixture_case()
    independent = normal_equation_forecast(case.task.context, case.task.future_index, window=4, alpha=.7)
    standardized_targets = Pipeline(LagRidgeImplementation(window=4, alpha=.7)).predict(
        case.task.context, case.task.future_index)
    public_adapter = Pipeline(window_length=4).add(StandardScaler()).add(Ridge(alpha=.7)).predict(
        case.task.context, case.task.future_index)
    np.testing.assert_allclose(independent.point, standardized_targets.point, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(independent.point, public_adapter.point, atol=1e-10, rtol=1e-10)
    assert independent.validate(case.task) == []


@pytest.mark.parametrize("fault", FAULTS)
def test_actual_prediction_branches_break_agreement_without_breaking_contract(fault):
    case = fixture_case()
    reference = normal_equation_forecast(case.task.context, case.task.future_index)
    broken = Pipeline(LagRidgeImplementation(fault)).predict(case.task.context, case.task.future_index)
    repaired = Pipeline(LagRidgeImplementation()).predict(case.task.context, case.task.future_index)
    assert broken.validate(case.task) == repaired.validate(case.task) == []
    assert point_agreement(broken.point, reference.point, atol=1e-8, rtol=1e-7)["alarm"]
    assert not point_agreement(repaired.point, reference.point, atol=1e-8, rtol=1e-7)["alarm"]
    if fault == "stale_recursive_state":
        np.testing.assert_allclose(broken.point, np.tile(broken.point[0], (len(broken.point), 1)))
        np.testing.assert_allclose(broken.point[0], repaired.point[0])


@pytest.mark.parametrize("fault,scalar_alarm,target_alarm", [
    ("clean", False, False),
    ("reverse_prediction_lags", False, False),
    ("stale_recursive_state", False, False),
    ("omit_inverse_target_scaling", True, True),
    ("swap_recursive_targets", False, True),
])
def test_declared_shift_checks_retain_expected_blind_spots(fault, scalar_alarm, target_alarm):
    case = fixture_case()
    pipeline = Pipeline(LagRidgeImplementation(fault))
    original = pipeline.predict(case.task.context, case.task.future_index)
    scalar = check_shift_equivariance(pipeline, case.task.context, case.task.future_index)
    target = target_shift_agreement(pipeline, case.task.context, case.task.future_index, original,
                                    shifts=[7., -11.], atol=1e-8, rtol=1e-7)
    assert (scalar.status == "FAIL") == scalar_alarm
    assert target["alarm"] == target_alarm


def test_conformal_order_statistic_has_no_interpolation_and_matches_rank_pvalue():
    calibration = np.arange(128, dtype=float)
    threshold, rank = conformal_threshold(calibration, .05)
    assert rank == 123 and threshold == 122
    for score in [0., 121.999, 122., 122.001, 128.]:
        assert (score > threshold) == (conformal_pvalue(calibration, score) <= .05)
    tied = np.ones(128)
    threshold, _ = conformal_threshold(tied, .05)
    assert threshold == 1
    assert conformal_pvalue(tied, 1) == 1
    assert conformal_pvalue(tied, 1.001) == 1/129
    with pytest.raises(ValueError, match="Too few"):
        conformal_threshold(np.arange(10.), .05)


def test_holm_stepdown_stops_at_first_failed_threshold_and_preserves_order():
    np.testing.assert_array_equal(holm_rejections([.04, .01, .20, .02], .05),
                                  [False, True, False, False])
    np.testing.assert_array_equal(holm_rejections([.05, .001, .01, .016], .05),
                                  [True, True, True, True])
    with pytest.raises(ValueError):
        holm_rejections([.01, np.nan], .05)


def small_protocol():
    protocol = json.loads(PROTOCOL.read_text())
    model = protocol["model_study"]
    model.update(calibration_cases=20, evaluation_cases=2,
                 calibration_seed_base=900000, evaluation_seed_base=910000)
    model["conditions"] = model["conditions"][:1]
    model["conditions"][0].update(context_length=24, horizon=3)
    covariance = protocol["covariance_study"]
    covariance.update(independent_suites=2, seed_base=920000, n_samples=20, n_calibration=79)
    for condition in covariance["conditions"]:
        condition.update(context_length=12, horizon=2)
    return protocol


def test_protocol_rejects_overlapping_evaluation_calibration_and_unresolvable_holm():
    protocol = small_protocol()
    validate_protocol(protocol)
    bad = deepcopy(protocol)
    bad["model_study"]["evaluation_seed_base"] = bad["model_study"]["calibration_seed_base"]
    with pytest.raises(ValueError, match="overlap"):
        validate_protocol(bad)
    bad = deepcopy(protocol)
    bad["covariance_study"]["n_calibration"] = 19
    with pytest.raises(ValueError, match="Holm"):
        validate_protocol(bad)


def test_small_study_preserves_paired_repairs_blind_spots_and_negative_controls():
    protocol = small_protocol()
    model = run_model_study(protocol["model_study"])
    assert len(model["calibration_records"]) == 20
    assert len(model["records"]) == 12
    assert len(model["condition_summaries"]) == 4*5
    for seed in {row["seed"] for row in model["records"]}:
        rows = {row["branch"]: row for row in model["records"] if row["seed"] == seed}
        assert rows["clean"]["forecast_sha256"] == rows["repaired"]["forecast_sha256"]
        assert rows["clean"]["alarms"] == rows["repaired"]["alarms"]
        assert all(not row["alarms"]["contract"] for row in rows.values())
    covariance = run_covariance_study(protocol["covariance_study"])
    assert len(covariance["records"]) == 2*4*3
    for row in covariance["records"]:
        assert not row["holm_alarm"] or row["raw_alarm"]
        if row["condition"] == "factorized_white_noise" and row["variant"] == "partial_shuffle":
            assert row["null_true"]
            assert abs(row["score_delta_from_clean"]["crps"]) < 1e-12
    for row in covariance["suite_records"]:
        assert row["null_hypotheses"] == {"clean": 4, "partial_shuffle": 1, "reduced_spread": 0}[row["variant"]]
        if row["variant"] == "reduced_spread":
            assert row["holm_any_false_alarm"] is None
    assert json.loads(json.dumps({"model": model, "covariance": covariance}, allow_nan=False))
    from experiments.render_supplement import branch_counts, render_tex
    results = {"model": model, "covariance": covariance}
    table_rows = branch_counts(results, protocol)
    assert [row["n"] for row in table_rows] == [2]*6
    assert table_rows[0]["counts"] == table_rows[-1]["counts"]
    tex = render_tex(results, {"protocol": protocol})
    assert "2 cases per row" in tex and "2 paired cases" in tex
    assert "384 cases" not in tex and "64 suites" not in tex
    assert "Reduced spread & --- & ---" in tex


def test_render_rejects_changed_numerical_records(tmp_path):
    import hashlib
    from experiments.render_supplement import validate_recorded_results
    results = {"protocol_sha256": "recorded-protocol", "protocol_version": "2.0"}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))
    manifest = {"results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "protocol_sha256": "recorded-protocol", "protocol": {"protocol_version": "2.0"}}
    validate_recorded_results(path, results, manifest)
    path.write_text(json.dumps({**results, "unrecorded": True}))
    with pytest.raises(ValueError, match="fingerprint"):
        validate_recorded_results(path, results, manifest)
