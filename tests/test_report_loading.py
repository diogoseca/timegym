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


@pytest.mark.parametrize("kwargs, message", [
    ({"name": ""}, "nonempty"),
    ({"name": None}, "nonempty"),
    ({"status": "PASSED"}, "diagnostic status"),
    ({"category": "custom"}, "diagnostic category"),
    ({"category": ""}, "diagnostic category"),
    ({"message": None}, "string"),
    ({"evidence": []}, "evidence must be an object"),
    ({"evidence": {"p": float("nan")}}, "nonfinite"),
    ({"evidence": {"p": float("inf")}}, "nonfinite"),
])
def test_diagnostic_construction_enforces_the_read_contract(kwargs, message):
    fields = dict(name="custom", status="PASS", category="correctness", message="")
    fields.update(kwargs)
    with pytest.raises(ValueError, match=message):
        DiagnosticResult(**fields)


@pytest.mark.parametrize("kwargs, message", [
    ({"scenario": ""}, "nonempty"),
    ({"seed": -1}, "seed"),
    ({"seed": True}, "seed"),
    ({"seed": 1.0}, "seed"),
    ({"metrics": {"mse": None}}, "finite number"),
    ({"metrics": {"mse": True}}, "finite number"),
    ({"metrics": {"mse": float("inf")}}, "finite number"),
    ({"metrics": {"": 1}}, "nonempty"),
    ({"manifest": []}, "objects"),
    ({"diagnostics": ["PASS"]}, "DiagnosticResult"),
])
def test_case_construction_enforces_the_read_contract(kwargs, message):
    fields = dict(scenario="custom", seed=None)
    fields.update(kwargs)
    with pytest.raises(ValueError, match=message):
        CaseResult(**fields)


def test_numpy_scalars_and_builtin_subclasses_roundtrip_as_exact_json_types():
    import numpy as np
    from timegym import DiagnosticCategory, DiagnosticStatus

    class Text(str):
        pass

    class Number(float):
        pass

    class Count(int):
        pass

    diagnostic = DiagnosticResult(Text("custom"), DiagnosticStatus.PASS,
                                  DiagnosticCategory.CORRECTNESS, Text(""),
                                  {"p": np.float64(0.1), "text": Text("value"),
                                   "count": Count(2), "number": Number(0.5)})
    report = TestReport([CaseResult(Text("custom"), np.int64(2), [diagnostic],
                                   {Text("mse"): Number(1.5)})])
    payload = report.to_dict()
    evidence = payload["cases"][0]["diagnostics"][0]["evidence"]
    assert [type(evidence[key]) for key in ("p", "text", "count", "number")] == [float, str, int, float]
    assert TestReport.from_dict(payload).to_dict() == payload
    assert TestReport.from_json(report.to_json()).to_dict() == payload


@pytest.mark.parametrize("mutate", [
    lambda r: r.cases[0].metrics.update(mse=None),
    lambda r: r.cases[0].diagnostics[0].evidence.update(p=float("nan")),
    lambda r: setattr(r.cases[0].diagnostics[0], "category", "invalid"),
    lambda r: r.metadata.update(p=float("inf")),
])
def test_writing_revalidates_mutable_records(saved_report, mutate):
    report = TestReport.from_dict(saved_report)
    mutate(report)
    with pytest.raises(ValueError):
        report.to_json()


def test_category_is_validated_when_loading(saved_report):
    payload = deepcopy(saved_report)
    payload["cases"][0]["diagnostics"][0]["category"] = "unknown"
    with pytest.raises(ValueError, match="diagnostic category"):
        TestReport.from_dict(payload)


@pytest.mark.parametrize("status", ["PASS", "FAIL", "ERROR", "INCONCLUSIVE", "UNSUPPORTED"])
def test_public_status_failure_semantics_match_report_summary_and_gate(status):
    from timegym import CaseResult, DiagnosticCategory, DiagnosticResult, DiagnosticStatus
    import timegym

    assert {"CaseResult", "DiagnosticCategory", "DiagnosticResult", "DiagnosticStatus"} <= set(timegym.__all__)
    diagnostic = DiagnosticResult("custom", DiagnosticStatus(status), DiagnosticCategory.CORRECTNESS, "check")
    failed = status in ("FAIL", "ERROR")
    assert diagnostic.failed is failed
    report = TestReport([CaseResult("custom", None, [diagnostic])])
    assert report.summary["failed_cases"] == int(failed)
    if failed:
        with pytest.raises(AssertionError):
            report.assert_passed()
    else:
        report.assert_passed()


@pytest.mark.parametrize("value", [10**400, pytest.param("numpy_extended", id="numpy-extended-float")])
def test_numeric_canonicalization_preserves_json_representable_metric_values(value):
    import numpy as np

    if value == "numpy_extended":
        value = np.longdouble(0.5)
    report = TestReport([CaseResult("custom", 0, metrics={"custom_metric": value})])
    payload = report.to_dict()
    assert type(payload["cases"][0]["metrics"]["custom_metric"]) in (int, float)
    assert TestReport.from_dict(payload).to_dict() == payload
