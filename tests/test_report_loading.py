"""Saved evidence must round-trip without coercion or executing model code."""

from copy import deepcopy
import json

import pytest

from timegym import GaussianAR1Forecaster, Pipeline, Scenario, TestReport
from timegym.reporting import CaseResult, DiagnosticResult


@pytest.fixture(scope="module")
def saved_report():
    return Pipeline(GaussianAR1Forecaster(n_samples=24, seed=18)).test(
        [Scenario("ar1", context_length=16, horizon=3)], seeds=[6]).to_dict()


def test_report_roundtrip_file_and_replay_preserve_evidence(saved_report, tmp_path):
    report = TestReport.from_dict(saved_report)
    serialized = report.to_json()
    assert TestReport.from_json(serialized).to_dict() == saved_report
    assert TestReport.from_json(serialized.encode("utf-8")).to_dict() == saved_report
    assert isinstance(report.cases[0], CaseResult)
    assert all(isinstance(item, DiagnosticResult) for item in report.cases[0].diagnostics)
    destination = tmp_path / "forecast-report.json"
    report.to_json(destination)
    restored = TestReport.read(destination)
    assert restored.to_dict() == saved_report
    replayed = restored.replay_case()
    assert replayed.metadata["replay"]["metrics_match"]
    assert replayed.metadata["replay"]["diagnostics_match"]
    with pytest.raises(ValueError, match="Invalid report JSON"):
        TestReport.from_json(str(destination))


def test_loaded_report_owns_detached_nested_values(saved_report):
    payload = deepcopy(saved_report)
    report = TestReport.from_dict(payload)
    payload["metadata"]["source"]["digest"] = "changed outside the report"
    payload["cases"][0]["diagnostics"][0]["message"] = "changed outside the report"
    assert report.to_dict() == saved_report
    report.cases[0].metrics["mse"] = 999
    assert payload["cases"][0]["metrics"]["mse"] == saved_report["cases"][0]["metrics"]["mse"]


def test_loaded_failures_remain_failures_for_ci_and_summary_tampering_is_rejected():
    report = TestReport(cases=[CaseResult("fixture", 0, diagnostics=[
        DiagnosticResult("contract_index_mismatch", "FAIL", "correctness", "Wrong future labels"),
        DiagnosticResult("point_accuracy", "INCONCLUSIVE", "capability", "No tolerance declared"),
    ])])
    restored = TestReport.from_json(report.to_json())
    with pytest.raises(AssertionError, match="contract_index_mismatch"):
        restored.assert_passed()
    payload = report.to_dict()
    payload["summary"]["failed_cases"] = 0
    with pytest.raises(ValueError, match="summary disagrees"):
        TestReport.from_dict(payload)


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p.update(schema_version="9.0"), "schema_version"),
    (lambda p: p.update(extra="not part of this schema"), "exactly these fields"),
    (lambda p: p.pop("summary"), "exactly these fields"),
    (lambda p: p["cases"][0].update(seed=True), "seed"),
    (lambda p: p["cases"][0].update(diagnostics={}), "diagnostics must be a list"),
    (lambda p: p["cases"][0]["diagnostics"][0].update(status="PASSED"), "diagnostic status"),
    (lambda p: p["cases"][0]["diagnostics"][0].pop("evidence"), "exactly these fields"),
    (lambda p: p["cases"][0]["diagnostics"][0].update(evidence=[]), "evidence must be an object"),
    (lambda p: p["cases"][0]["metrics"].update(mse=True), "finite number"),
    (lambda p: p["cases"][0]["metrics"].update(mse="0.2"), "finite number"),
    (lambda p: p["cases"][0]["metrics"].update(mse=None), "finite number"),
    (lambda p: p["cases"][0]["metrics"].update(mse=float("nan")), "nonfinite number"),
    (lambda p: p["summary"].update(cases=True), "summary disagrees"),
])
def test_invalid_saved_records_are_not_coerced(saved_report, mutate, message):
    payload = deepcopy(saved_report)
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        TestReport.from_dict(payload)


@pytest.mark.parametrize("text, message", [
    ('{"schema_version":"2.0","schema_version":"2.0"}', "Duplicate JSON object key"),
    ('{"nested":{"status":"PASS","status":"FAIL"}}', "Duplicate JSON object key"),
    ('{"x":NaN}', "Nonfinite JSON constant"),
    ('{"x":Infinity}', "Nonfinite JSON constant"),
    ('{"x":-Infinity}', "Nonfinite JSON constant"),
    ('{"x":1e999}', "nonfinite number"),
])
def test_json_rejects_ambiguous_keys_and_nonfinite_numbers(text, message):
    with pytest.raises(ValueError, match=message):
        TestReport.from_json(text)


def test_saved_model_descriptors_remain_inert_data(saved_report):
    payload = deepcopy(saved_report)
    payload["metadata"]["pipeline_config"]["model"] = {
        "kind": "external", "module": "this_module_must_never_be_imported",
        "qualname": "run", "parameters": {"command": "never execute report content"},
    }
    payload["metadata"]["pipeline_config"]["replayable"] = False
    report = TestReport.from_json(json.dumps(payload))
    assert report.metadata["pipeline_config"]["model"]["module"] == "this_module_must_never_be_imported"
    with pytest.raises(ValueError, match="not automatically replayable"):
        report.replay_case()
