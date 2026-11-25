import json

import numpy as np
import pandas as pd
import pytest
import warnings

from timegym.contracts import Forecast
from timegym.diagnostics import RankAlarm, covariance_dependence_check, evaluate_forecast, rank_alarm
from timegym.reporting import CaseResult, DiagnosticResult, TestReport
from timegym.scenarios import Scenario


def get_diagnostic(result, name):
    return next(item for item in result.diagnostics if item.name == name)


def test_reference_scores_do_not_claim_automatic_capability_pass():
    case = Scenario("white_noise", history_length=12, horizon=4).generate(seed=3)
    result = evaluate_forecast(case, case.reference.forecast(n_samples=64, seed=5))
    assert get_diagnostic(result, "forecast_contract").status == "PASS"
    assert get_diagnostic(result, "point_accuracy").status == "INCONCLUSIVE"
    assert get_diagnostic(result, "interval_coverage").status == "INCONCLUSIVE"
    assert result.metrics["mean_reference_mse"] == 0
    assert {"mse", "mae", "mase", "pinball", "crps", "energy_score", "variogram_score"} <= result.metrics.keys()
    assert result.manifest == case.manifest


def test_interval_width_overflow_is_an_error_without_losing_valid_coverage():
    case = Scenario("white_noise", history_length=8, horizon=2).generate(seed=0)
    forecast = Forecast(case.task.future_index,
                        quantiles=np.broadcast_to([-1e308, 1e308], (2, 1, 2)).copy(),
                        levels=(0.1, 0.9))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = evaluate_forecast(case, forecast)
    assert get_diagnostic(result, "forecast_contract").status == "PASS"
    assert get_diagnostic(result, "metric_interval_width").status == "ERROR"
    assert get_diagnostic(result, "metric_interval_width").evidence["exception_type"] == "FloatingPointError"
    assert "interval_width" not in result.metrics
    assert result.metrics["interval_coverage"] == 1.0
    assert result.metrics["interval_nominal_coverage"] == pytest.approx(0.8)
    coverage = get_diagnostic(result, "interval_coverage")
    assert coverage.status == "INCONCLUSIVE"
    assert "mean_width" not in coverage.evidence
    assert result.metrics["pinball"] == pytest.approx(1e307)
    report = TestReport([result])
    assert TestReport.from_json(report.to_json()).to_dict() == report.to_dict()
    with pytest.raises(AssertionError, match="metric_interval_width"):
        report.assert_passed()


def test_representable_mean_width_survives_large_individual_intervals():
    case = Scenario("white_noise", history_length=8, horizon=2).generate(seed=0)
    forecast = Forecast(case.task.future_index,
                        quantiles=np.array([[[-1e308, 1e308]], [[0.0, 0.0]]]),
                        levels=(0.1, 0.9))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = evaluate_forecast(case, forecast)
    assert result.metrics["interval_width"] == pytest.approx(1e308)
    assert not any(item.name == "metric_interval_width" for item in result.diagnostics)


def test_missing_outputs_are_unsupported_and_not_reconstructed():
    case = Scenario("white_noise", history_length=12, horizon=4).generate()
    forecast = Forecast(index=case.task.future_index, samples=case.reference.sample(30, 3), target_names=case.task.target_names)
    result = evaluate_forecast(case, forecast)
    assert get_diagnostic(result, "point_accuracy").status == "UNSUPPORTED"
    assert get_diagnostic(result, "quantile_accuracy").status == "UNSUPPORTED"
    assert "mse" not in result.metrics
    assert "pinball" not in result.metrics
    assert "crps" in result.metrics


@pytest.mark.parametrize("fault,code", [
    ("index", "index_mismatch"),
    ("point_shape", "point_shape"),
    ("nonfinite", "nonfinite_point"),
    ("targets", "target_mismatch"),
    ("crossing", "quantile_crossing"),
    ("levels", "duplicate_levels"),
])
def test_explicit_contract_violations_fail_before_scoring(fault, code):
    case = Scenario("white_noise", history_length=12, horizon=4).generate()
    forecast = case.reference.forecast(n_samples=32)
    if fault == "index":
        forecast.index = forecast.index + 1
    elif fault == "point_shape":
        forecast.point = forecast.point[:, 0]
    elif fault == "nonfinite":
        forecast.point[0, 0] = np.nan
    elif fault == "targets":
        forecast.target_names = ("wrong",)
    elif fault == "crossing":
        forecast.quantiles = forecast.quantiles[..., ::-1]
    elif fault == "levels":
        forecast.levels = (0.1, 0.1, 0.9)
    result = evaluate_forecast(case, forecast, check_dependence=True)
    assert get_diagnostic(result, "contract_" + code).status == "FAIL"
    assert result.metrics == {}
    with pytest.raises(AssertionError, match=code):
        TestReport([result]).assert_passed()


def test_validation_errors_preserve_exception_details():
    class BrokenOutput:
        def validate(self, task):
            raise RuntimeError("original failure detail")
    case = Scenario("white_noise", history_length=12, horizon=2).generate()
    result = evaluate_forecast(case, BrokenOutput())
    diagnostic = get_diagnostic(result, "forecast_contract")
    assert diagnostic.status == "ERROR"
    assert diagnostic.evidence["exception_type"] == "RuntimeError"
    assert "original failure detail" in diagnostic.evidence["traceback"]


def test_invalid_evaluation_data_is_an_error_not_a_model_failure():
    case = Scenario("white_noise", history_length=12, horizon=2).generate()
    case.observed.index = pd.RangeIndex(100, 102)
    result = evaluate_forecast(case, case.reference.forecast(n_samples=20))
    assert get_diagnostic(result, "evaluation_data").status == "ERROR"
    assert result.metrics == {}


@pytest.mark.parametrize("fault", ["index", "target_order", "mean_shape", "covariance_shape"])
def test_misaligned_or_malformed_reference_is_an_evaluation_error(fault):
    case = Scenario("ar1", history_length=12, horizon=3, n_targets=2).generate()
    forecast = case.reference.forecast(n_samples=20)
    if fault == "index":
        case.reference.index = case.reference.index + 100
    elif fault == "target_order":
        case.reference.target_names = tuple(reversed(case.reference.target_names))
    elif fault == "mean_shape":
        case.reference.mean = case.reference.mean[:-1]
    elif fault == "covariance_shape":
        case.reference.covariance = case.reference.covariance[:-1, :-1]
    result = evaluate_forecast(case, forecast, check_dependence=True)
    assert get_diagnostic(result, "forecast_contract").status == "PASS"
    diagnostic = get_diagnostic(result, "evaluation_reference")
    assert diagnostic.status == "ERROR"
    assert diagnostic.evidence["exception_type"] == "ValueError"
    assert result.metrics == {}
    assert not any(item.name == "reference_covariance" for item in result.diagnostics)


@pytest.mark.parametrize("field", ["observed", "mean", "variance", "covariance"])
def test_complex_evaluation_inputs_are_errors_without_discarding_imaginary_values(field):
    case = Scenario("ar1", history_length=12, horizon=3, n_targets=2).generate()
    forecast = case.reference.forecast(n_samples=20)
    if field == "observed":
        case.observed = case.observed.astype(complex) + 1j
    else:
        setattr(case.reference, field, getattr(case.reference, field).astype(complex) + 1j)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = evaluate_forecast(case, forecast, check_dependence=True)
    name = "evaluation_data" if field == "observed" else "evaluation_reference"
    diagnostic = get_diagnostic(result, name)
    assert diagnostic.status == "ERROR"
    assert "real-valued" in diagnostic.evidence["exception_message"]
    assert result.metrics == {}


@pytest.mark.parametrize("field", ["samples", "mean", "covariance", "sampler"])
def test_covariance_check_rejects_complex_inputs_without_conversion_warnings(field):
    case = Scenario("ar1", history_length=12, horizon=3, n_targets=2).generate()
    samples = case.reference.sample(20)
    if field == "samples":
        samples = samples.astype(complex)
    elif field == "sampler":
        case.reference.sample = lambda count, seed: np.ones((count, 3, 2), dtype=complex)
    else:
        setattr(case.reference, field, getattr(case.reference, field).astype(complex))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real-valued"):
            covariance_dependence_check(case.reference, samples)


def test_constant_history_reports_undefined_mase_explicitly():
    case = Scenario("white_noise", history_length=12, horizon=2, parameters={"sigma": 0}).generate()
    result = evaluate_forecast(case, case.reference.forecast(n_samples=20))
    assert get_diagnostic(result, "metric_mase").status == "INCONCLUSIVE"
    assert "mase" not in result.metrics


def test_covariance_check_detects_marginal_preserving_permutation_and_repairs():
    case = Scenario("random_walk", history_length=16, horizon=6, n_targets=2,
                    parameters={"correlation": 0.8}).generate(seed=3)
    original = case.reference.sample(256, 17)
    shuffled = original.copy().reshape(256, -1)
    rng = np.random.default_rng(32)
    for column in range(shuffled.shape[1]):
        shuffled[:, column] = rng.permutation(shuffled[:, column])
    shuffled = shuffled.reshape(original.shape)
    np.testing.assert_array_equal(np.sort(original, axis=0), np.sort(shuffled, axis=0))
    bad = covariance_dependence_check(case.reference, shuffled, seed=101, n_calibration=99)
    repaired = covariance_dependence_check(case.reference, original, seed=101, n_calibration=99)
    assert bad.status == "FAIL"
    assert bad.evidence["p_value"] <= 0.05
    assert repaired.status == "PASS"
    assert repaired.evidence["p_value"] > 0.05
    assert bad.to_dict() == covariance_dependence_check(case.reference, shuffled, seed=101).to_dict()
    assert bad.evidence["p_value"] >= 1 / 100


def test_covariance_check_exposes_insufficient_calibration_and_support():
    case = Scenario("white_noise", history_length=12, horizon=2).generate()
    assert covariance_dependence_check(case.reference, None).status == "UNSUPPORTED"
    assert covariance_dependence_check(case.reference, case.reference.sample(10)).status == "INCONCLUSIVE"
    check = covariance_dependence_check(case.reference, case.reference.sample(20), n_calibration=19, alpha=0.01)
    assert check.status == "INCONCLUSIVE"
    assert check.evidence["minimum_p_value"] == 0.05
    assert check.evidence["critical_value"] is None


def _shuffled_dependence_case():
    case = Scenario("random_walk", history_length=16, horizon=6, n_targets=2,
                    parameters={"correlation": 0.8}).generate(seed=3)
    original = case.reference.sample(256, 17)
    shuffled = original.copy().reshape(256, -1)
    rng = np.random.default_rng(32)
    for column in range(shuffled.shape[1]):
        shuffled[:, column] = rng.permutation(shuffled[:, column])
    return case, original, shuffled.reshape(original.shape)


@pytest.mark.parametrize("which, n_calibration, alpha, status", [
    ("shuffled", 99, 0.05, "FAIL"),
    ("original", 99, 0.05, None),  # PASS or FAIL: one correct ensemble among 99 others
    ("shuffled", 19, 0.05, "FAIL"),
    ("original", 19, 0.01, "INCONCLUSIVE"),
])
def test_covariance_check_keeps_its_evidence_through_rank_alarm(monkeypatch, which, n_calibration, alpha,
                                                                 status):
    import math

    from timegym import diagnostics

    seen = {}

    def spy(observed, calibration, *, alpha):
        seen.update(observed=float(observed), calibration=np.asarray(calibration, dtype=float))
        return rank_alarm(observed, calibration, alpha=alpha)

    monkeypatch.setattr(diagnostics, "rank_alarm", spy)
    case, original, shuffled = _shuffled_dependence_case()
    samples = shuffled if which == "shuffled" else original
    check = covariance_dependence_check(case.reference, samples, seed=101, n_calibration=n_calibration,
                                        alpha=alpha)
    # The rule as the check wrote it out before it called rank_alarm, on the same statistics.
    observed, null = seen["observed"], seen["calibration"]
    assert len(null) == n_calibration
    p_value = float((1 + np.sum(null >= observed)) / (n_calibration + 1))
    rank = math.ceil((n_calibration + 1) * (1 - alpha))
    critical_value = float(np.sort(null)[rank - 1]) if rank <= n_calibration else None
    if 1 / (n_calibration + 1) > alpha:
        expected = "INCONCLUSIVE"
    else:
        expected = "FAIL" if p_value <= alpha else "PASS"
    messages = {
        "FAIL": "Ensemble covariance differs from the specified reference at the declared per-check level.",
        "PASS": "No covariance discrepancy detected at the declared level; this does not establish "
                "equality of distributions.",
        "INCONCLUSIVE": "Too few calibration ensembles to reject at the requested alpha.",
    }
    assert (check.status, check.message) == (expected, messages[expected])
    assert status in (None, expected)
    evidence = check.evidence
    assert evidence["statistic"] == observed
    assert evidence["p_value"] == p_value
    assert evidence["minimum_p_value"] == pytest.approx(1 / (n_calibration + 1), abs=1e-15)
    assert evidence["critical_value"] == critical_value
    assert (evidence["alpha"], evidence["n_calibration"]) == (alpha, n_calibration)


# --- The rank rule ---------------------------------------------------------------------------

def test_rank_alarm_p_value_counts_calibration_values_at_least_as_large():
    calibration = np.arange(1.0, 20.0)  # 1, 2, ..., 19
    result = rank_alarm(15.5, calibration, alpha=0.25)
    assert isinstance(result, RankAlarm)
    assert result.p_value == (1 + 4) / 20  # 16, 17, 18 and 19
    assert (result.n, result.alpha, result.observed) == (19, 0.25, 15.5)
    assert (result.rank, result.threshold) == (15, 15.0)  # ceil(20 * 0.75)
    assert result.status == "FAIL"  # p = 0.25 is at most alpha
    assert rank_alarm(14.5, calibration, alpha=0.25).status == "PASS"  # p = 6 / 20
    assert rank_alarm(100.0, calibration).p_value == 1 / 20
    assert rank_alarm(-100.0, calibration).p_value == 1.0
    with pytest.raises(Exception):
        result.p_value = 0.0  # frozen


def test_rank_alarm_counts_ties_against_the_alarm():
    tied = np.ones(128)
    assert rank_alarm(1.0, tied).p_value == 1.0
    assert rank_alarm(1.0, tied).status == "PASS"
    assert rank_alarm(1.001, tied).p_value == 1 / 129
    calibration = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    assert rank_alarm(2.0, calibration).p_value == (1 + 4) / 6


def test_rank_alarm_fails_exactly_when_the_observed_value_is_above_the_threshold():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        n = int(rng.integers(1, 300))
        alpha = float(rng.choice([0.01, 0.05, 0.1, 0.059, 0.172, float(rng.uniform(0.001, 0.999))]))
        # Few distinct values, so ties and observed values equal to the threshold are common.
        calibration = rng.integers(0, 12, size=n).astype(float)
        observed = float(rng.integers(-1, 14))
        result = rank_alarm(observed, calibration, alpha=alpha)
        assert (observed > result.threshold) == (result.p_value <= alpha)
        assert (result.status == "FAIL") == (result.p_value <= alpha and result.status != "INCONCLUSIVE")
        if result.rank <= n:
            assert result.threshold == np.sort(calibration)[result.rank - 1]
    # 1000 * (1 - 0.059) rounds to just above 941, so a plain ceil would give rank 942 and
    # threshold 941, and 940.5 would fail without being above it.
    result = rank_alarm(940.5, np.arange(999.0), alpha=0.059)
    assert result.rank == 941 and result.threshold == 940.0
    assert result.p_value == (1 + 58) / 1000 and result.status == "FAIL"


def test_rank_alarm_is_inconclusive_when_no_value_could_fail():
    result = rank_alarm(1e9, np.arange(19.0), alpha=0.01)  # 1 / 20 > 0.01
    assert result.status == "INCONCLUSIVE"
    assert result.p_value == 1 / 20
    assert result.rank == 20 and result.threshold == float("inf")
    edge = rank_alarm(1e9, np.arange(19.0), alpha=0.05)  # 1 / 20 = 0.05: one value can fail
    assert (edge.status, edge.rank, edge.threshold) == ("FAIL", 19, 18.0)
    assert rank_alarm(5.0, [1.0], alpha=0.3).status == "INCONCLUSIVE"


@pytest.mark.parametrize("observed, calibration, alpha, message", [
    (1.0, [], 0.05, "calibration"),
    (1.0, [[1.0, 2.0]], 0.05, "calibration"),
    (1.0, [1.0, np.nan], 0.05, "calibration"),
    (1.0, [1.0, np.inf], 0.05, "calibration"),
    (1.0, [1.0 + 1j], 0.05, "real"),
    (np.nan, [1.0], 0.05, "observed"),
    (np.inf, [1.0], 0.05, "observed"),
    ([1.0, 2.0], [1.0], 0.05, "observed"),
    (True, [1.0], 0.05, "observed"),
    (1.0, [1.0], 0.0, "alpha"),
    (1.0, [1.0], 1.0, "alpha"),
    (1.0, [1.0], np.nan, "alpha"),
    (1.0, [1.0], True, "alpha"),
    (1.0, [1.0], "0.05", "alpha"),
])
def test_rank_alarm_rejects_invalid_inputs(observed, calibration, alpha, message):
    with pytest.raises(ValueError, match=message):
        rank_alarm(observed, calibration, alpha=alpha)


def test_report_json_is_safe_and_preserves_detailed_failures(tmp_path):
    diagnostic = DiagnosticResult("example", "ERROR", "evaluation", "Model failed", {
        "exception": ValueError("original failure"), "numpy": np.array([1.0, 2.0]),
        "integer": np.int64(7), "date": pd.Timestamp("2026-01-01"),
    })
    report = TestReport([CaseResult("example", 2, [diagnostic])], {"number": np.float64(3.0)})
    path = tmp_path / "report.json"
    encoded = report.to_json(path)
    payload = json.loads(encoded)
    assert json.loads(path.read_text()) == payload
    assert payload["summary"]["status_counts"]["ERROR"] == 1
    assert payload["summary"]["failed_cases"] == 1
    evidence = payload["cases"][0]["diagnostics"][0]["evidence"]
    assert evidence["exception"] == {"exception_type": "ValueError", "exception_message": "original failure",
                                     "traceback": "ValueError: original failure\n"}
    assert evidence["numpy"] == [1.0, 2.0]
    assert payload["metadata"]["number"] == 3.0
    with pytest.raises(AssertionError, match="Model failed"):
        report.assert_passed()


def test_ci_gate_does_not_relabel_inconclusive_or_unsupported_checks():
    diagnostics = [DiagnosticResult("coverage", "INCONCLUSIVE", "uncertainty", "One path"),
                   DiagnosticResult("quantiles", "UNSUPPORTED", "capability", "Absent")]
    report = TestReport([CaseResult("example", 0, diagnostics)])
    assert report.assert_passed() is report
    assert report.summary["status_counts"]["PASS"] == 0
    with pytest.raises(ValueError):
        DiagnosticResult("bad", "SUCCESS", "correctness", "Unknown status")



def test_large_singular_reference_uses_the_same_invariants_during_evaluation():
    from timegym import Pipeline

    scenario = Scenario("random_walk", history_length=40, horizon=12, n_targets=3,
                        parameters={"correlation": 1.0, "sigma": 1e3})
    case = scenario.generate(seed=0)
    case.reference.validate()
    result = evaluate_forecast(case, case.reference.forecast(32))
    assert result.metrics["mean_reference_mse"] == 0
    assert not any(item.failed for item in result.diagnostics)
    report = Pipeline().test([scenario], seeds=(0,))
    assert report.summary["failed_cases"] == 0
    assert report.cases[0].metrics


@pytest.mark.parametrize("fault", ["asymmetric", "negative_variance", "diagonal", "nonfinite", "indefinite"])
def test_reference_validation_rechecks_mutated_moments(fault):
    case = Scenario("white_noise", horizon=2).generate()
    forecast = case.reference.forecast(24)
    if fault == "asymmetric":
        case.reference.covariance[0, 1] = 1
    elif fault == "negative_variance":
        case.reference.variance[0, 0] = -1
    elif fault == "diagonal":
        case.reference.variance[0, 0] += 1
    elif fault == "nonfinite":
        case.reference.mean[0, 0] = np.nan
    else:
        case.reference.covariance[:] = [[1, 2], [2, 1]]
    with pytest.raises(ValueError):
        case.reference.validate()
    assert get_diagnostic(evaluate_forecast(case, forecast), "evaluation_reference").status == "ERROR"


@pytest.mark.parametrize("kwargs, message", [
    ({"n_calibration": 0}, "n_calibration"),
    ({"n_calibration": True}, "n_calibration"),
    ({"alpha": 0}, "alpha"),
    ({"alpha": None}, "alpha"),
    ({"alpha": True}, "alpha"),
    ({"alpha": float("nan")}, "alpha"),
    ({"dependence_seed": None}, "dependence_seed"),
    ({"dependence_seed": -1}, "dependence_seed"),
    ({"dependence_seed": True}, "dependence_seed"),
    ({"dependence_seed": 1.5}, "dependence_seed"),
    ({"check_dependence": "false"}, "boolean"),
])
@pytest.mark.parametrize("check_dependence", [False, True])
def test_evaluate_settings_raise_before_case_or_forecast_processing(kwargs, message, check_dependence):
    settings = {"check_dependence": check_dependence, **kwargs}
    with pytest.raises(ValueError, match=message):
        evaluate_forecast(None, None, **settings)


@pytest.mark.parametrize("seed", [None, -1, True, np.bool_(False), 1.5, "2"])
def test_covariance_seed_must_be_replayable_even_without_samples(seed):
    with pytest.raises(ValueError, match="seed.*nonnegative integer"):
        covariance_dependence_check(None, None, seed=seed)


def test_covariance_normalizes_numpy_seed_and_distinguishes_unsupported_coordinates():
    case = Scenario("white_noise", horizon=1).generate()
    for count in (10, 30):
        check = covariance_dependence_check(case.reference, case.reference.sample(count))
        assert check.status == "UNSUPPORTED"
    case = Scenario("white_noise", horizon=2).generate()
    check = covariance_dependence_check(case.reference, case.reference.sample(24), seed=np.int64(3), n_calibration=1)
    assert type(check.evidence["seed"]) is int
    assert check.evidence["seed"] == 3


def test_exception_evidence_uses_its_argument_outside_and_inside_another_handler():
    from timegym.diagnostics import exception_evidence

    try:
        raise ValueError("deferred exception")
    except ValueError as exc:
        deferred = exc
    expected = exception_evidence(deferred)
    assert "deferred exception" in expected["traceback"]
    assert "test_exception_evidence_uses_its_argument" in expected["traceback"]
    try:
        raise RuntimeError("other exception")
    except RuntimeError:
        assert exception_evidence(deferred) == expected
        diagnostic = DiagnosticResult("custom", "ERROR", "evaluation", "", {"exception": deferred})
    assert diagnostic.to_dict()["evidence"]["exception"] == expected


def test_descriptive_scores_share_evidence_and_mase_scaling():
    case = Scenario("seasonal", history_length=24, horizon=3, parameters={"period": 6}).generate()
    result = evaluate_forecast(case, case.reference.forecast(24))
    for name, scores in (("point_accuracy", ("mse", "mae", "mase", "mean_reference_mse")),
                         ("quantile_accuracy", ("pinball",)),
                         ("sample_scores", ("crps", "energy_score", "variogram_score"))):
        diagnostic = get_diagnostic(result, name)
        assert diagnostic.category == "capability"
        for score in scores:
            assert diagnostic.evidence[score] == result.metrics[score]
    assert get_diagnostic(result, "point_accuracy").evidence["mase_seasonality"] == 1


def test_check_categories_survive_errors_and_undefined_metrics(monkeypatch):
    import timegym.diagnostics as diagnostics
    from timegym.checks import check_shift_equivariance
    from timegym import Pipeline

    case = Scenario("white_noise", history_length=12, horizon=2, parameters={"sigma": 0}).generate()
    forecast = case.reference.forecast(24)
    undefined = get_diagnostic(evaluate_forecast(case, forecast), "metric_mase")
    def broken(*args, **kwargs):
        raise RuntimeError("intentional error")
    monkeypatch.setattr(diagnostics.metrics, "mase", broken)
    monkeypatch.setattr(case.reference, "sample", broken)
    case.reference.covariance[:] = np.eye(2)
    case.reference.variance[:] = 1
    result = evaluate_forecast(case, forecast, check_dependence=True)
    failed = get_diagnostic(result, "metric_mase")
    assert (undefined.status, failed.status) == ("INCONCLUSIVE", "ERROR")
    assert undefined.category == failed.category == "capability"
    dependence = get_diagnostic(result, "reference_covariance")
    assert (dependence.status, dependence.category) == ("ERROR", "distribution")
    pipeline = Pipeline()
    monkeypatch.setattr(pipeline, "predict", broken)
    shift = check_shift_equivariance(pipeline, case.task.history, case.task.future_index)
    assert (shift.status, shift.category) == ("ERROR", "metamorphic")


def test_malformed_manifest_does_not_invent_a_case_identity():
    case = Scenario("white_noise", horizon=2).generate()
    forecast = case.reference.forecast(24)
    del case.manifest["kind"]
    with pytest.raises(KeyError, match="kind"):
        evaluate_forecast(case, forecast)
