import json

import numpy as np
import pandas as pd
import pytest
import warnings

from timegym.contracts import Forecast
from timegym.diagnostics import covariance_dependence_check, evaluate_forecast
from timegym.reporting import CaseResult, DiagnosticResult, TestReport
from timegym.scenarios import Scenario


def get_diagnostic(result, name):
    return next(item for item in result.diagnostics if item.name == name)


def test_reference_scores_do_not_claim_automatic_capability_pass():
    case = Scenario("white_noise", context_length=12, horizon=4).generate(seed=3)
    result = evaluate_forecast(case, case.reference.forecast(n_samples=64, seed=5))
    assert get_diagnostic(result, "forecast_contract").status == "PASS"
    assert get_diagnostic(result, "point_accuracy").status == "INCONCLUSIVE"
    assert get_diagnostic(result, "interval_coverage").status == "INCONCLUSIVE"
    assert result.metrics["mean_reference_mse"] == 0
    assert {"mse", "mae", "mase", "pinball", "crps", "energy_score", "variogram_score"} <= result.metrics.keys()
    assert result.manifest == case.manifest


def test_interval_width_overflow_is_an_error_without_losing_valid_coverage():
    case = Scenario("white_noise", context_length=8, horizon=2).generate(seed=0)
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
    case = Scenario("white_noise", context_length=8, horizon=2).generate(seed=0)
    forecast = Forecast(case.task.future_index,
                        quantiles=np.array([[[-1e308, 1e308]], [[0.0, 0.0]]]),
                        levels=(0.1, 0.9))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = evaluate_forecast(case, forecast)
    assert result.metrics["interval_width"] == pytest.approx(1e308)
    assert not any(item.name == "metric_interval_width" for item in result.diagnostics)


def test_missing_outputs_are_unsupported_and_not_reconstructed():
    case = Scenario("white_noise", context_length=12, horizon=4).generate()
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
    case = Scenario("white_noise", context_length=12, horizon=4).generate()
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
    case = Scenario("white_noise", context_length=12, horizon=2).generate()
    result = evaluate_forecast(case, BrokenOutput())
    diagnostic = get_diagnostic(result, "forecast_contract")
    assert diagnostic.status == "ERROR"
    assert diagnostic.evidence["exception_type"] == "RuntimeError"
    assert "original failure detail" in diagnostic.evidence["traceback"]


def test_invalid_evaluation_data_is_an_error_not_a_model_failure():
    case = Scenario("white_noise", context_length=12, horizon=2).generate()
    case.observed.index = pd.RangeIndex(100, 102)
    result = evaluate_forecast(case, case.reference.forecast(n_samples=20))
    assert get_diagnostic(result, "evaluation_data").status == "ERROR"
    assert result.metrics == {}


@pytest.mark.parametrize("fault", ["index", "target_order", "mean_shape", "covariance_shape"])
def test_misaligned_or_malformed_reference_is_an_evaluation_error(fault):
    case = Scenario("ar1", context_length=12, horizon=3, n_targets=2).generate()
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
    case = Scenario("ar1", context_length=12, horizon=3, n_targets=2).generate()
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
    case = Scenario("ar1", context_length=12, horizon=3, n_targets=2).generate()
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


def test_constant_context_reports_undefined_mase_explicitly():
    case = Scenario("white_noise", context_length=12, horizon=2, parameters={"sigma": 0}).generate()
    result = evaluate_forecast(case, case.reference.forecast(n_samples=20))
    assert get_diagnostic(result, "metric_mase").status == "INCONCLUSIVE"
    assert "mase" not in result.metrics


def test_covariance_check_detects_marginal_preserving_permutation_and_repairs():
    case = Scenario("random_walk", context_length=16, horizon=6, n_targets=2,
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
    case = Scenario("white_noise", context_length=12, horizon=2).generate()
    assert covariance_dependence_check(case.reference, None).status == "UNSUPPORTED"
    assert covariance_dependence_check(case.reference, case.reference.sample(10)).status == "INCONCLUSIVE"
    check = covariance_dependence_check(case.reference, case.reference.sample(20), n_calibration=19, alpha=0.01)
    assert check.status == "INCONCLUSIVE"
    assert check.evidence["minimum_p_value"] == 0.05
    assert check.evidence["critical_value"] is None


def test_report_json_is_safe_and_preserves_detailed_failures(tmp_path):
    diagnostic = DiagnosticResult("example", "ERROR", "evaluation", "Model failed", {
        "exception": ValueError("original failure"), "numpy": np.array([1.0, np.nan]),
        "integer": np.int64(7), "date": pd.Timestamp("2026-01-01"),
    })
    report = TestReport([CaseResult("example", 2, [diagnostic])], {"number": np.inf})
    path = tmp_path / "report.json"
    encoded = report.to_json(path)
    payload = json.loads(encoded)
    assert json.loads(path.read_text()) == payload
    assert payload["summary"]["status_counts"]["ERROR"] == 1
    assert payload["summary"]["failed_cases"] == 1
    evidence = payload["cases"][0]["diagnostics"][0]["evidence"]
    assert evidence["exception"] == {"exception_type": "ValueError", "exception_message": "original failure"}
    assert evidence["numpy"] == [1.0, None]
    assert payload["metadata"]["number"] is None
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

