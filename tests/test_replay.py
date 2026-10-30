from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from timegym import GaussianAR1Forecaster, NaiveForecaster, Pipeline, Scenario
from timegym.provenance import replay_case


@pytest.mark.parametrize("model", [None, NaiveForecaster("seasonal", period=3),
                                   GaussianAR1Forecaster(n_samples=40, seed=37)])
def test_builtin_configuration_roundtrip_reproduces_exact_outputs(model):
    pipeline = Pipeline(model, window_length=7)
    config = json.loads(json.dumps(pipeline.to_config()))
    assert config["replayable"] is True
    restored = Pipeline.from_config(config)
    assert restored.to_config() == config
    report = pipeline.test([Scenario("ar1", context_length=24, horizon=3, n_targets=2)],
                           seeds=[4, 8], check_dependence=True, n_calibration=19,
                           alpha=0.1, dependence_seed_offset=51)
    replayed = replay_case(json.loads(report.to_json()), 1)
    assert replayed.cases[0].to_dict() == report.cases[1].to_dict()
    assert all(replayed.metadata["replay"].values())
    assert report.metadata["diagnostic_settings"]["dependence_seed_offset"] == 51
    assert report.metadata["source"]["algorithm"] == "sha256"
    assert len(report.metadata["source"]["digest"]) == 64
    assert {"numpy", "pandas", "scipy"} <= report.metadata["dependencies"].keys()


def test_factory_configuration_is_descriptive_without_invoking_factory():
    calls = []
    def factory():
        calls.append(True)
        return NaiveForecaster()
    config = Pipeline(model_factory=factory).to_config()
    assert calls == []
    assert config["replayable"] is False
    assert config["model"]["kind"] == "factory"
    assert "factory" in config["model"]["qualname"]
    with pytest.raises(ValueError, match="not automatically replayable"):
        Pipeline.from_config(config)
    report = Pipeline(model_factory=factory).test([Scenario("ar1")], seeds=[1])
    assert len(report.cases) == 1 and len(calls) == 1


def test_nondeterministic_model_seed_is_explicitly_not_replayable():
    config = Pipeline(GaussianAR1Forecaster(seed=None)).to_config()
    assert config["replayable"] is False
    assert "fixed nonnegative integer" in config["model"]["reason"]
    with pytest.raises(ValueError, match="not automatically replayable"):
        Pipeline.from_config(config)


def test_sklearn_configuration_retains_parameters_but_forbids_automatic_imports():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    config = Pipeline(window_length=4).add(StandardScaler(with_mean=False)).add(Ridge(alpha=2.5)).to_config()
    assert config["replayable"] is False
    assert config["model"]["steps"][0]["parameters"]["with_mean"] is False
    assert config["model"]["steps"][1]["parameters"]["alpha"] == 2.5
    with pytest.raises(ValueError, match="not automatically replayable"):
        Pipeline.from_config(config)


def test_untrusted_model_names_and_parameters_cannot_trigger_imports(tmp_path):
    malicious = Pipeline().to_config()
    malicious["model"]["class"] = "os.system"
    with pytest.raises(ValueError, match="allowlist"):
        Pipeline.from_config(malicious)
    config = Pipeline().to_config()
    config["model"]["module"] = "os"
    with pytest.raises(ValueError, match="allowlisted"):
        Pipeline.from_config(config)
    config = Pipeline().to_config()
    config["model"]["parameters"]["callback"] = "print('do not execute')"
    with pytest.raises(ValueError, match="constructor parameters"):
        Pipeline.from_config(config)


def test_native_subclasses_do_not_inherit_replay_permission():
    class CustomNaive(NaiveForecaster):
        pass
    assert Pipeline(CustomNaive()).to_config()["replayable"] is False


def test_replay_rejects_ambiguous_flags_and_case_identity():
    report = Pipeline().test([Scenario("ar1")], seeds=[2]).to_dict()
    changed = deepcopy(report)
    changed["metadata"]["diagnostic_settings"]["check_dependence"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        replay_case(changed)
    changed = deepcopy(report)
    changed["cases"][0]["seed"] = 3
    with pytest.raises(ValueError, match="disagree"):
        replay_case(changed)
    with pytest.raises(ValueError, match="outside"):
        replay_case(report, 1)


def test_replay_exposes_source_changes_instead_of_asserting_identical_environment():
    report = Pipeline().test([Scenario("random_walk")], seeds=[2]).to_dict()
    report["metadata"]["source"]["digest"] = "different source"
    replayed = replay_case(report)
    assert replayed.metadata["replay"]["source_matches"] is False
    assert replayed.metadata["replay"]["metrics_match"] is True


def test_report_replay_method_and_cli(tmp_path):
    report = Pipeline(NaiveForecaster("drift")).test([Scenario("trend")], seeds=[2])
    assert report.replay_case().metadata["replay"]["metrics_match"] is True
    saved, output = tmp_path / "report.json", tmp_path / "replayed.json"
    report.to_json(saved)
    script = Path(__file__).resolve().parents[1] / "examples" / "reproduce_report.py"
    completed = subprocess.run([sys.executable, str(script), str(saved), "--output", str(output)],
                               text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["diagnostics_match"] is True
    assert json.loads(output.read_text())["metadata"]["replay"]["metrics_match"] is True

