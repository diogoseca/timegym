from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from timegym import GaussianAR1Forecaster, LastValueForecaster, NaiveForecaster, Pipeline, Scenario, TestReport
from timegym.pipeline import replay_case


@pytest.mark.parametrize("model", [None, NaiveForecaster("seasonal", period=3), LastValueForecaster(),
                                   GaussianAR1Forecaster(n_samples=40, seed=37),
                                   GaussianAR1Forecaster(n_samples=40, seed=37, levels=(.2, .8))])
def test_builtin_configuration_roundtrip_reproduces_exact_outputs(model):
    pipeline = Pipeline(model, window_length=7)
    config = json.loads(json.dumps(pipeline.to_config()))
    assert config["replayable"] is True
    restored = Pipeline.from_config(config)
    assert restored.to_config() == config
    report = pipeline.test([Scenario("ar1", context_length=24, horizon=3, n_targets=2)],
                           seeds=[4, 8], check_dependence=True, n_calibration=19,
                           alpha=0.1, dependence_seed_offset=51)
    replayed = TestReport.from_json(report.to_json()).replay_case(1)
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
    class CustomLastValue(LastValueForecaster):
        pass
    assert Pipeline(CustomNaive()).to_config()["replayable"] is False
    assert Pipeline(CustomLastValue()).to_config()["replayable"] is False


def test_replay_rejects_ambiguous_flags_and_case_identity():
    report = Pipeline().test([Scenario("ar1")], seeds=[2]).to_dict()
    changed = deepcopy(report)
    changed["metadata"]["diagnostic_settings"]["check_dependence"] = "false"
    with pytest.raises(ValueError, match="boolean"):
        TestReport.from_dict(changed).replay_case()
    changed = deepcopy(report)
    changed["cases"][0]["seed"] = 3
    with pytest.raises(ValueError, match="disagree"):
        TestReport.from_dict(changed).replay_case()
    with pytest.raises(ValueError, match="outside"):
        TestReport.from_dict(report).replay_case(1)


def test_replay_exposes_source_changes_instead_of_asserting_identical_environment():
    report = Pipeline().test([Scenario("random_walk")], seeds=[2]).to_dict()
    report["metadata"]["source"]["digest"] = "different source"
    replayed = TestReport.from_dict(report).replay_case()
    assert replayed.metadata["replay"]["source_matches"] is False
    assert replayed.metadata["replay"]["metrics_match"] is True


def test_report_replay_method_and_example_script(tmp_path):
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
    moved = report.to_dict()
    moved["metadata"]["source"]["digest"] = "different source"
    TestReport.from_dict(moved).to_json(saved)
    for gate, status in [([], 0), (["--require-same-environment"], 1)]:
        completed = subprocess.run([sys.executable, str(script), str(saved), *gate],
                                   text=True, capture_output=True, check=False)
        assert completed.returncode == status, completed.stderr



def test_configuration_parser_does_not_construct_a_pipeline_and_preserves_subclasses():
    from timegym.provenance import pipeline_from_config

    class SpecializedPipeline(Pipeline):
        pass

    config = Pipeline(NaiveForecaster("drift")).to_config()
    arguments = pipeline_from_config(config)
    assert isinstance(arguments, dict)
    assert type(arguments["model"]) is NaiveForecaster
    restored = SpecializedPipeline.from_config(config)
    assert type(restored) is SpecializedPipeline
    assert restored.to_config() == config


@pytest.mark.parametrize("version", [True, 1, 1.0, 2.0, 4, None, "2"])
def test_pipeline_config_requires_an_exact_supported_schema_version(version):
    config = Pipeline().to_config()
    config["schema_version"] = version
    with pytest.raises(ValueError, match="configuration schema"):
        Pipeline.from_config(config)


def test_pipeline_config_rejects_ambiguous_reduction_fields():
    config = Pipeline().to_config()
    config["strategy"] = "recursive"
    with pytest.raises(ValueError, match="configuration schema"):
        Pipeline.from_config(config)


def test_pipeline_reduction_and_naive_strategy_errors_name_the_supported_options():
    with pytest.raises(ValueError, match="'iterated'.*recursive.*direct.*multioutput"):
        Pipeline(reduction="iterated")
    with pytest.raises(ValueError, match="unknown.*last.*mean.*seasonal.*drift"):
        NaiveForecaster(strategy="unknown")


@pytest.mark.parametrize('model', [NaiveForecaster('seasonal', period=3), LastValueForecaster(),
                                    GaussianAR1Forecaster(n_samples=16, seed=13, levels=(.2, .8))])
def test_builtin_params_cover_constructor_clone_and_replay(model):
    import inspect
    assert set(model.params()) == set(inspect.signature(type(model)).parameters)
    case = Scenario('ar1', context_length=24, horizon=3).generate()
    model.fit(case.task.context)
    clone = model.clone_unfitted()
    assert clone is not model and clone.params() == model.params()
    assert not any(name.endswith('_') for name in vars(clone))
    restored = Pipeline.from_config(json.loads(json.dumps(Pipeline(model).to_config()))).model
    assert restored.params() == model.params()


@pytest.mark.parametrize('seed', [-1, True, 1.5, 'x', None])
def test_replay_uses_model_seed_validation(seed):
    config = Pipeline(GaussianAR1Forecaster()).to_config()
    config['model']['parameters']['seed'] = seed
    with pytest.raises(ValueError, match='seed'):
        Pipeline.from_config(config)


def test_old_ar1_configs_restore_the_original_default_quantiles():
    from timegym.contracts import DEFAULT_LEVELS
    config = Pipeline(GaussianAR1Forecaster()).to_config()
    config['schema_version'] = 2
    del config['target_transform']
    del config['model']['parameters']['levels']
    restored = Pipeline.from_config(config)
    assert restored.model.levels == DEFAULT_LEVELS
    assert 'levels' not in config['model']['parameters']


def test_sklearn_clone_uses_constructor_parameters_without_fitted_state():
    import inspect
    from sklearn.linear_model import Ridge
    from timegym.models import SklearnForecaster
    model = SklearnForecaster([Ridge(alpha=3.)], window_length=4)
    assert set(model.params()) == set(inspect.signature(type(model)).parameters)
    case = Scenario('ar1', context_length=24, horizon=3).generate()
    model.fit(case.task.context)
    clone = model.clone_unfitted()
    assert clone.params() == model.params()
    assert clone.steps is not model.steps
    assert not hasattr(clone, 'models_')


def test_replay_rejects_raw_dicts_instead_of_bypassing_report_validation():
    payload = Pipeline().test([Scenario("trend", context_length=12, horizon=2)], seeds=[0]).to_dict()
    payload["cases"][0]["diagnostics"][0]["status"] = "PASSED"
    with pytest.raises(TypeError, match="requires a TestReport"):
        replay_case(payload)
    with pytest.raises(ValueError, match="diagnostic status"):
        TestReport.from_dict(payload)


def test_reproduced_error_ignores_traceback_paths_but_records_the_difference(tmp_path):
    report = Pipeline(NaiveForecaster("seasonal", period=50)).test(
        [Scenario("ar1", context_length=16, horizon=3)], seeds=[0])
    diagnostic = report.cases[0].diagnostics[0]
    assert diagnostic.status == "ERROR"
    diagnostic.evidence["traceback"] = diagnostic.evidence["traceback"].replace(
        str(Path(__file__).resolve().parents[1]), "/another/machine/timegym")
    replayed = report.replay_case()
    comparison = replayed.metadata["replay"]
    assert comparison["reproduced"]
    assert comparison["diagnostics_match"]
    assert not comparison["traceback_match"]
    assert comparison["same_environment"]
    path = tmp_path / "error.json"
    report.to_json(path)
    loaded = TestReport.read(path).replay_case().metadata["replay"]
    assert loaded["reproduced"] and loaded["same_environment"]
    diagnostic.evidence["exception_message"] = "a different failure"
    assert not report.replay_case().metadata["replay"]["reproduced"]


@pytest.mark.parametrize("field", ["metrics", "diagnostics", "manifest", "source", "dependencies", "python"])
def test_replay_verdicts_distinguish_results_from_environment(field):
    report = Pipeline().test([Scenario("ar1", context_length=12, horizon=2)], seeds=[0])
    if field == "metrics":
        report.cases[0].metrics["mse"] += 1
    elif field == "diagnostics":
        report.cases[0].diagnostics[0].message = "different message"
    elif field == "manifest":
        report.cases[0].manifest["extra"] = "different manifest"
    elif field == "source":
        report.metadata["source"]["digest"] = "changed"
    elif field == "dependencies":
        report.metadata["dependencies"]["numpy"] = "changed"
    else:
        report.metadata["python"] = "changed"
    comparison = report.replay_case().metadata["replay"]
    result_changed = field in {"metrics", "diagnostics", "manifest"}
    assert comparison["reproduced"] is (not result_changed)
    assert comparison["same_environment"] is result_changed


def test_schema_constants_describe_current_writes():
    from timegym.reporting import REPORT_SCHEMA_VERSION
    from timegym.provenance import PIPELINE_CONFIG_SCHEMA_VERSION
    from timegym.scenarios import MANIFEST_SCHEMA_VERSION

    report = Pipeline().test([Scenario("ar1", context_length=12, horizon=2)], seeds=[0])
    assert report.to_dict()["schema_version"] == REPORT_SCHEMA_VERSION == "2.0"
    assert report.metadata["pipeline_config"]["schema_version"] == PIPELINE_CONFIG_SCHEMA_VERSION == 3
    assert report.cases[0].manifest["schema_version"] == MANIFEST_SCHEMA_VERSION == 1
