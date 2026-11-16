"""The protocol 3.0 file and the pieces every Study 3 runner shares. No test here computes a study outcome."""
import copy
import json

import pytest

from experiments import study3_common as common
from experiments.common import sha256


@pytest.fixture(scope="module")
def protocol():
    return common.load_protocol()


def test_protocol_is_ascii_and_canonically_formatted():
    raw = common.PROTOCOL.read_text()
    assert raw.isascii()
    assert json.dumps(json.loads(raw), indent=2) + "\n" == raw


def test_frozen_inputs_match_their_digests(protocol):
    for item in protocol["frozen_inputs"].values():
        assert sha256(common.ROOT / item["path"]) == item["sha256"], item["path"]


def test_case_seed_blocks_match_the_registered_sizes(protocol):
    assert common.case_seeds(protocol, "model_calibration", block=5)[-1] == 7_050_238
    assert common.case_seeds(protocol, "model_evaluation", block=0) == list(range(7_100_000, 7_100_093))
    assert common.case_seed(protocol, "covariance_cases", index=3, block=92) == 8_000_923
    assert common.case_seed(protocol, "covariance_shuffle", index=3, block=92) == 11_000_923
    assert len(common.case_seeds(protocol, "predictability_series")) == 200
    part1, part3 = protocol["part_1_generated_mutants"], protocol["part_3_suite_questions"]
    blocks = protocol["seeds"]["case_seed_blocks"]
    assert blocks["model_calibration"][3] == part1["cases"]["calibration_cases"] == 239
    assert blocks["model_evaluation"][3] == part1["cases"]["evaluation_cases"] == 93
    assert blocks["covariance_cases"][:3] == [part3["settings"]["seed_base"], part3["settings"]["suite_seed_stride"],
                                              part3["settings"]["independent_suites"]]
    study2 = json.loads((common.ROOT / "experiments/protocol-v2.json").read_text())
    assert blocks["covariance_cases"][3] == len(study2["covariance_study"]["conditions"]) == 4
    assert blocks["model_calibration"][2] == len(study2["model_study"]["conditions"]) == 6
    for name, offset in (("covariance_reference_samples", "reference_sample_seed_offset"),
                         ("covariance_calibration", "calibration_seed_offset"),
                         ("covariance_shuffle", "mutation_seed_offset")):
        assert blocks[name][0] == blocks["covariance_cases"][0] + part3["settings"][offset]
        assert blocks[name][1:] == blocks["covariance_cases"][1:]
    with pytest.raises(IndexError):
        common.case_seed(protocol, "model_evaluation", index=93)
    with pytest.raises(KeyError):
        common.case_seed(protocol, "format", index=0)


def test_no_earlier_protocol_uses_a_study_3_seed():
    def integers(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from integers(item)
        elif isinstance(value, list):
            for item in value:
                yield from integers(item)
        elif isinstance(value, int) and not isinstance(value, bool):
            yield value

    for name in common.EARLIER_PROTOCOLS:
        earlier = json.loads((common.ROOT / "experiments" / name).read_text())
        assert max(integers(earlier), default=0) < common.SEED_FLOOR, name
    from timegym import scorecard

    assert scorecard._ATTAINABLE_SEED + scorecard._ATTAINABLE_SERIES < common.SEED_FLOOR


@pytest.mark.parametrize("change, message", [
    (lambda seeds: seeds["case_seed_blocks"].__setitem__("classification_cases", [7_100_050, 10_000, 6, 16]),
     "overlap"),
    (lambda seeds: seeds["case_seed_blocks"].__setitem__("zoo_series", [6_999_990, 1, 1, 30]), "must lie in"),
    (lambda seeds: seeds["case_seed_blocks"].__setitem__("model_evaluation", [7_100_000, 50, 6, 93]),
     "overlaps itself"),
    (lambda seeds: seeds["generator_tags"].__setitem__("sign_flip", 7_000_005), "falls inside"),
    (lambda seeds: seeds["generator_tags"].__setitem__("sign_flip", 7_500_000), "repeat"),
])
def test_a_broken_seed_table_is_refused(protocol, change, message):
    broken = copy.deepcopy(protocol)
    change(broken["seeds"])
    with pytest.raises(ValueError, match=message):
        common.check_seed_table(broken)


def test_a_draft_protocol_may_not_start(protocol):
    draft = copy.deepcopy(protocol)
    draft.update(status="draft")
    draft["registration"].update(merge_commit="", registered_on="")
    for part in common.PARTS:
        assert common.unmet_start_conditions(draft, part)
        with pytest.raises(RuntimeError, match="may not start"):
            common.require_start(draft, part)
    with pytest.raises(ValueError):
        common.unmet_start_conditions(draft, "part-7")


def test_start_conditions_name_what_is_missing(protocol):
    ready = copy.deepcopy(protocol)
    ready.update(status="registered")
    ready["registration"].update(merge_commit="0" * 40, registered_on="2026-12-01")
    assert common.unmet_start_conditions(ready, "part-3") == []
    ready["registration"]["registered_on"] = ""
    assert common.unmet_start_conditions(ready, "part-4") == [
        "registration.merge_commit and registration.registered_on are not filled in"]
    ready["registration"]["registered_on"] = "2026-12-01"
    assert common.unmet_start_conditions(ready, "generation") == common.unmet_start_conditions(ready, "zoo") == []
    assert common.unmet_start_conditions(ready, "part-1") == [
        "part_1_generated_mutants.sources.manifest.sha256 is not filled in"]
    ready["part_1_generated_mutants"]["subject"]["frozen_files_sha256"] = {"experiments/study3_subject.py": "0" * 64}
    reasons = common.unmet_start_conditions(ready, "generation")
    assert len(reasons) == 4 and all("study3_" in reason for reason in reasons)
    assert "experiments/study3_subject.py differs from its registered digest" in reasons
    assert "experiments/study3_learners.py has no digest in frozen_files_sha256" in reasons
    assert common.unmet_start_conditions(ready, "part-2") == reasons + [
        "part_1_generated_mutants.sources.manifest.sha256 is not filled in"]
    assert common.unmet_start_conditions(ready, "part-5") == []


def test_the_frozen_files_match_their_registered_digests(protocol):
    subject = protocol["part_1_generated_mutants"]["subject"]
    assert sorted(subject["frozen_files_sha256"]) == sorted(subject["frozen_files"])
    assert common.frozen_file_problems(protocol) == []


def test_the_mutant_manifest_must_match_its_registered_digest(protocol, tmp_path, monkeypatch):
    registered = protocol["part_1_generated_mutants"]["sources"]["manifest"]
    assert registered["path"] == "experiments/results-v3/mutants/manifest.json"
    monkeypatch.setattr(common, "ROOT", tmp_path)
    ready = copy.deepcopy(protocol)
    entry = ready["part_1_generated_mutants"]["sources"]["manifest"]
    entry["sha256"] = ""
    assert common.manifest_problems(ready) == ["part_1_generated_mutants.sources.manifest.sha256 is not filled in"]
    entry["sha256"] = "0" * 64
    assert common.manifest_problems(ready) == [f"{entry['path']} does not exist"]
    manifest = tmp_path / entry["path"]
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n")
    assert common.manifest_problems(ready) == [f"{entry['path']} differs from its registered digest"]
    entry["sha256"] = sha256(manifest)
    assert common.manifest_problems(ready) == []


def test_write_record_keeps_runtime_out_of_the_numbers(tmp_path):
    manifest = common.write_record(tmp_path / "out", {"b": 1.5, "a": [1, 2]}, runtime={"seconds": 2.0},
                                   sources=(common.ROOT / "experiments/study3_common.py",), extra={"note": "test"})
    results = (tmp_path / "out/results.json").read_text()
    assert json.loads(results) == {"a": [1, 2], "b": 1.5} and "seconds" not in results
    assert manifest["results_sha256"] == sha256(tmp_path / "out/results.json")
    assert manifest["protocol_sha256"] == sha256(common.PROTOCOL) and manifest["note"] == "test"
    assert list(manifest["source_files"]) == ["experiments/study3_common.py"]
    assert json.loads((tmp_path / "out/runtime.json").read_text()) == {"seconds": 2.0}
    with pytest.raises(FileExistsError):
        common.write_record(tmp_path / "out", {"a": 1})
    common.write_record(tmp_path / "out", {"a": 1}, force=True)
