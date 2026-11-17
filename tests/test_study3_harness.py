"""The Study 3 harness and the runner of Parts 1 and 2, on a toy subject, toy mutants, toy learners and toy
conditions.

Nothing here runs the study's subject, its mutants or its learners. Seeds are below 1,000,000.
"""
import copy
import json
from pathlib import Path

import pytest

from experiments import run_study3_mutants as runner
from experiments import study3_common as common
from experiments import study3_harness as harness

FIXTURES = Path(__file__).parent / "fixtures"
TOYS = str(FIXTURES / "study3_toys.py")
CLEAN = (FIXTURES / "study3_toy_subject.py").read_text()
LEVEL = "        self.level = values.mean(axis=0)\n"


def mutant(old, new):
    assert CLEAN.count(old) == 1
    return CLEAN.replace(old, new)


WRONG = mutant(LEVEL, "        self.level = values.mean(axis=0) + 0.5\n")
EQUIVALENT = mutant(LEVEL, "        self.level = np.mean(values, axis=0)\n")
RAISING = mutant("        point = np.tile(", "        raise RuntimeError('toy')\n        point = np.tile(")
SYNTAX = CLEAN.replace("class ToyForecaster:", "class ToyForecaster(:")
STATEFUL = mutant(LEVEL, "        self.level = first.setdefault('level', values.mean(axis=0))\n").replace(
    "from timegym import Forecast\n", "from timegym import Forecast\nfirst = {}\n")
# Loops forever from its fourth fit on, in the condition whose context has 24 rows only.
LOOPING = mutant(LEVEL, LEVEL + "        calls.append(1)\n        while len(calls) > 3 and len(values) == 24:\n"
                 "            pass\n")
LOOPING = LOOPING.replace("from timegym import Forecast\n", "from timegym import Forecast\ncalls = []\n")
SETTINGS = {"atol": 1e-8, "rtol": 1e-7, "scalar_shift": 10.0, "target_shifts": [7.0, -11.0], "refits": 39,
            "alpha": 0.05}


def toy_config(tmp_path, workers):
    import sys

    sys.path.insert(0, str(FIXTURES))
    import study3_toys

    return {"conditions": list(study3_toys.CONDITIONS),
            "seeds": {"screen": [[100, 101, 102], [200, 201, 202]],
                      "calibration": [list(range(1000, 1019)), list(range(2000, 2019))],
                      "evaluation": [list(range(3000, 3006)), list(range(4000, 4006))]},
            "screen_sizes": {"6": 1, "24": 2, "48": 3}, "settings": SETTINGS, "learner_module": TOYS,
            "learner_tag": 400, "reference": TOYS + ":reference_forecast", "clean_source": CLEAN,
            "time_limit": {"factor": 5.0, "minimum": 5.0}, "workers": workers, "scratch": str(tmp_path / "units"),
            "sign_flip_tag": 600, "sign_flips": 999, "part1_checks": runner.PART1_CHECKS}


def source(name, text, *kinds):
    return {"id": name, "source": text, "family": "Toy",
            "labels": [{"label": f"{name}-{i}", "kind": kind} for i, kind in enumerate(kinds)]}


def part1_sources():
    return [source("wrong", WRONG, "cosmic_ray"), source("stateful", STATEFUL, "catalogue", "study2"),
            source("equivalent", EQUIVALENT, "cosmic_ray", "cosmic_ray"), source("raising", RAISING, "study2"),
            source("syntax", SYNTAX, "cosmic_ray")]


@pytest.fixture(scope="module")
def part1(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("part1")
    config = toy_config(tmp_path, workers=4)
    sources = part1_sources()
    result = runner.run_part1(config, {"name": "mean", "index": 0}, sources, {item["id"] for item in sources})
    return config, result, {item["id"]: item for item in result["sources"]}


def test_screens_classify_every_toy_mutant(part1):
    _, _, sources = part1
    sizes = {name: {size: (entry["runnable"], entry["equivalent"], entry["reason"])
                    for size, entry in sources[name]["screen"]["sizes"].items()} for name in sources}
    assert sizes["wrong"] == {size: (True, False, None) for size in ("6", "24", "48")}
    assert sizes["equivalent"] == {size: (True, True, None) for size in ("6", "24", "48")}
    assert sizes["raising"] == {size: (False, False, "exception on every input") for size in ("6", "24", "48")}
    assert sizes["syntax"] == {size: (False, False, "syntax") for size in ("6", "24", "48")}
    # The stateful mutant keeps the warm-up context's level; the warm-up context is the first input.
    assert sizes["stateful"] == {"6": (True, True, None), "24": (True, False, None), "48": (True, False, None)}
    assert sources["equivalent"]["screen_misses"] == {"differing_cases": 0, "not_evaluated": 0, "miss": False}


def test_checks_and_decisions_on_toy_mutants(part1):
    _, result, sources = part1
    for clean in result["clean"].values():
        assert all(counts["count"] == 0 for counts in clean["alarms"].values())
        assert set(clean["alarms"]) == {"output", "scalar_shift", "target_shift", "second_implementation"}
    stateful = sources["stateful"]["conditions"]
    for row in stateful.values():
        assert row["decisions"]["scalar_shift"]["caught"] is True
        assert row["decisions"]["second_implementation"]["caught"] is True
        assert set(row["decisions"]["loss"]) == {"catalogue", "study2"}
    wrong = sources["wrong"]["conditions"]["toy_white_noise"]
    assert wrong["completed"] == 6 and wrong["not_evaluated"] == 0 and wrong["evaluated"] == 6
    assert wrong["decisions"]["scalar_shift"]["caught"] is False
    assert wrong["decisions"]["target_shift"]["caught"] is False
    assert wrong["decisions"]["second_implementation"]["caught"] is True
    answer_key = wrong["decisions"]["answer_key"]["cosmic_ray"]
    assert wrong["statistical"]["answer_key"]["alarms"] >= 5 and answer_key["holm"] and answer_key["unadjusted"]
    assert answer_key["family_size"] == 1
    paired = wrong["statistical"]["paired"]["p_value"] * 1000
    assert abs(paired - round(paired)) < 1e-9 and wrong["statistical"]["paired"]["mean_difference"] > 0
    funnel = result["funnel"]
    stages = {key: funnel["cosmic_ray"]["stages"][key] for key in ("generated", "duplicate", "distinct", "runnable",
                                                                   "equivalent", "not_equivalent", "caught_output",
                                                                   "caught_second_implementation")}
    assert stages == {"generated": 4, "duplicate": 1, "distinct": 3, "runnable": 2, "equivalent": 1,
                      "not_equivalent": 1, "caught_output": 0, "caught_second_implementation": 1}
    assert funnel["cosmic_ray"]["not_runnable_reasons"] == {"syntax": 1}
    assert funnel["catalogue"]["equivalence_sensitivity"] == {"6": {"runnable": 1, "equivalent": 1},
                                                              "24": {"runnable": 1, "equivalent": 0},
                                                              "48": {"runnable": 1, "equivalent": 0}}
    assert funnel["study2"]["stages"]["generated"] == 2 and funnel["study2"]["stages"]["not_equivalent"] == 1
    assert funnel["study2"]["not_runnable_reasons"] == {"exception on every input": 1}
    curve = result["curves"]["answer_key"]
    assert len(curve["false_alarm"]["pooled"]) == 20 and curve["false_alarm"]["pooled"][-1] == 1.0
    assert curve["detection"]["cosmic_ray"]["members"] == 1


def test_records_survive_a_time_limit_and_report_load_errors(tmp_path):
    base = {"learner": {"module": TOYS, "name": "mean", "index": 0, "tag": 400}, "settings": SETTINGS,
            "stages": [["screen", [200, 201, 202, 203]]], "checks": [], "warmup_seed": 200}
    import sys

    sys.path.insert(0, str(FIXTURES))
    import study3_toys

    specs = [{**base, "output": str(tmp_path / f"{name}.jsonl"), "source": text, "module_name": f"toy_{name}",
              "condition": study3_toys.CONDITIONS[1]}
             for name, text in (("looping", LOOPING), ("import", "import toy_module_that_does_not_exist\n"))]
    looping, failed = harness.run_units(specs, [3.0, 30.0], workers=2)
    assert looping["timed_out"] and not looping["finished"]
    assert [record["kind"] for record in looping["records"]] == ["load", "warmup", "case", "case"]
    assert failed["records"][0]["reason"] == "import" and not failed["timed_out"]


def test_units_do_not_depend_on_the_number_of_workers(tmp_path):
    import sys

    sys.path.insert(0, str(FIXTURES))
    import study3_toys

    def specs(folder):
        return [{"output": str(tmp_path / folder / f"{name}.jsonl"), "source": text, "module_name": f"toy_{name}",
                 "learner": {"module": TOYS, "name": "noisy", "index": 1, "tag": 400}, "settings": SETTINGS,
                 "condition": study3_toys.CONDITIONS[0], "warmup_seed": 100,
                 "stages": [["screen", [100, 101]], ["evaluation", [3000, 3001]]],
                 "checks": ["scores", "scalar_shift", "calibrated_shift"]}
                for name, text in (("clean", CLEAN), ("stateful", STATEFUL), ("wrong", WRONG))]

    def numbers(results):
        return [[{key: value for key, value in record.items() if "seconds" not in key}
                 for record in result["records"]] for result in results]

    one = harness.run_units(specs("one"), [60.0] * 3, workers=1)
    three = harness.run_units(specs("three"), [60.0] * 3, workers=3)
    assert numbers(one) == numbers(three)
    statuses = [record["calibrated_shift"]["status"] for record in one[0]["records"]
                if record.get("stage") == "evaluation"]
    assert statuses and set(statuses) <= {"PASS", "FAIL"}


def test_a_stopped_run_is_continued_from_the_units_it_ended(tmp_path):
    import os
    import sys

    sys.path.insert(0, str(FIXTURES))
    import study3_toys

    base = {"learner": {"module": TOYS, "name": "noisy", "index": 1, "tag": 400}, "settings": SETTINGS,
            "condition": study3_toys.CONDITIONS[0], "warmup_seed": 100,
            "stages": [["screen", [100, 101]], ["evaluation", [3000, 3001]]], "checks": ["scores"]}
    specs = [{**base, "output": str(tmp_path / f"{name}.jsonl"), "source": text, "module_name": f"toy_{name}"}
             for name, text in (("clean", CLEAN), ("wrong", WRONG), ("syntax", SYNTAX))]
    limits = [None, 60.0, 60.0]

    def written():
        return [os.stat(spec["output"]).st_mtime_ns for spec in specs]

    def numbers(results):
        return [[{key: value for key, value in record.items() if "seconds" not in key}
                 for record in result["records"]] for result in results]

    first, stamps = harness.run_units(specs, limits, workers=3), written()
    assert [result["finished"] for result in first] == [True, True, False]
    assert all(harness.outcome_path(spec).is_file() for spec in specs)
    # Nothing is run again: the stored results, wall times included, come back.
    assert harness.run_units(specs, limits, workers=3, resume=True) == first and written() == stamps
    # The machine stopped during the second unit: half of its lines are on disk and it has no outcome file.
    lines = Path(specs[1]["output"]).read_text().splitlines(keepends=True)
    Path(specs[1]["output"]).write_text("".join(lines[:3]) + lines[3][:10])
    harness.outcome_path(specs[1]).unlink()
    again, after = harness.run_units(specs, limits, workers=3, resume=True), written()
    assert numbers(again) == numbers(first) and again[0] == first[0] and again[2] == first[2]
    assert [new != old for new, old in zip(after, stamps)] == [False, True, False]
    # A unit is run again when its limit or its specification differs, when its lines changed after it ended,
    # and when it crashed; the others are kept.
    stamps = after
    harness.run_units(specs, [None, 61.0, 60.0], workers=3, resume=True)
    assert [new != old for new, old in zip(written(), stamps)] == [False, True, False]
    stamps = written()
    changed = [dict(specs[0], warmup_seed=101)] + specs[1:]
    harness.run_units(changed, [None, 61.0, 60.0], workers=3, resume=True)
    assert [new != old for new, old in zip(written(), stamps)] == [True, False, False]
    stamps = written()
    with open(specs[2]["output"], "a") as handle:
        handle.write("{}\n")
    outcome = json.loads(harness.outcome_path(specs[1]).read_text())
    harness.outcome_path(specs[1]).write_text(json.dumps({**outcome, "crashed": True}))
    stamps = written()
    harness.run_units(changed, [None, 61.0, 60.0], workers=3, resume=True)
    assert [new != old for new, old in zip(written(), stamps)] == [False, True, True]
    # Without resume every unit is run.
    stamps = written()
    fresh = harness.run_units(specs, limits, workers=3)
    assert numbers(fresh) == numbers(first) and all(new != old for new, old in zip(written(), stamps))


def test_part1_gives_the_same_result_when_it_is_continued(part1):
    config, result, _ = part1
    outcomes = sorted(Path(config["scratch"]).rglob("*.outcome.json"))
    lines = sorted(Path(config["scratch"]).rglob("*.jsonl"))
    assert len(outcomes) == len(lines) == 18
    # The run stopped late: the outcome files of the last evaluation units are missing.
    for path in [path for path in outcomes if path.name.startswith("evaluation-")][-3:]:
        path.unlink()
    learner, sources = {"name": "mean", "index": 0}, copy.deepcopy(part1_sources())
    continued = runner.run_part1({**config, "resume": True}, learner, sources, {item["id"] for item in sources})
    assert json.dumps(continued, sort_keys=True) == json.dumps(result, sort_keys=True)
    assert len(sorted(Path(config["scratch"]).rglob("*.outcome.json"))) == len(outcomes)


def test_sampling_and_generic_order():
    sizes = (("A", 6), ("B", 3), ("C", 1))
    pool = [{"id": f"{family}{i}", "family": family} for family, n in sizes for i in range(n)]
    assert runner.sample_generic(pool, 10, 7) == pool
    sample = runner.sample_generic(pool, 5, 7)
    counts = {family: sum(item["family"] == family for item in sample) for family in "ABC"}
    assert counts == {"A": 3, "B": 1, "C": 1} and sample == [item for item in pool if item in sample]
    assert runner.sample_generic(pool, 5, 7) == sample
    order = runner.generic_order(pool, 7)
    assert sorted(item["id"] for item in order) == sorted(item["id"] for item in pool)
    first = [item["family"] for item in order[:3]]
    assert sorted(first) == ["A", "B", "C"] and runner.generic_order(pool, 7) == order


def test_main_refuses_a_draft_and_plans_without_computing(capsys, tmp_path, monkeypatch):
    draft = common.load_protocol()
    draft.update(status="draft")
    draft["registration"].update(merge_commit="", registered_on="")
    draft["part_1_generated_mutants"]["sources"]["manifest"]["path"] = "experiments/results-v3/no-such-manifest.json"
    monkeypatch.setattr(common, "load_protocol", lambda: copy.deepcopy(draft))
    assert common.unmet_start_conditions(draft, "part-1")
    with pytest.raises(RuntimeError, match="may not start"):
        runner.main(["--part", "1", "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
    with pytest.raises(SystemExit):
        runner.main(["--part", "2", "--plan", "--mutants", str(tmp_path / "missing.json")])
    capsys.readouterr()
    runner.main(["--part", "2", "--plan"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["part"] == 2 and [row["learner"] for row in printed["rows"]] == ["hgb", "mlp_window_standardized",
                                                                                   "mlp_raw"]
    assert printed["note"].startswith("no manifest at") and all(row["sources"] == 29 for row in printed["rows"])
    assert not (tmp_path / "out").exists()


def test_bind_reads_the_registered_seeds():
    config = runner.bind(common.load_protocol(), Path("/nonexistent"), workers=2)
    assert [len(seeds) for seeds in config["seeds"]["calibration"]] == [239] * 6
    assert config["seeds"]["evaluation"][5][-1] == 7_150_092
    assert config["seeds"]["screen"][0][:4] == [7_200_000, 7_200_001, 7_200_002, 7_200_003]
    assert config["learner_tag"] == 7_400_000 and config["sign_flip_tag"] == 7_600_000
    assert copy.deepcopy(config["time_limit"]) == {"factor": 20.0, "minimum": 120.0}
    assert config["screen_sizes"] == {"6": 1, "24": 4, "48": 8} and config["sign_flips"] == 9999
    assert config["generic_per_learner"] == 12 and config["cap"] == 400
    assert [condition["id"] for condition in config["conditions"]][::5] == ["white_noise_locations",
                                                                           "random_walk_drifts"]
    protocol = common.load_protocol()
    protocol["part_1_generated_mutants"]["condition_ids"].reverse()
    with pytest.raises(ValueError, match="condition_ids"):
        runner.bind(protocol, Path("/nonexistent"), workers=1)
    assert runner.manifest_path(common.load_protocol()) == common.RESULTS / "mutants/manifest.json"


def test_part2_classifies_learners_and_selects_generic_mutants(tmp_path):
    config = toy_config(tmp_path, workers=4)
    config.update(conditions=config["conditions"][:1], generic_per_learner=1,
                  checks_by_class=common.load_protocol()["part_2_learners"]["checks_by_class"])
    config["seeds"] = {name: seeds[:1] for name, seeds in config["seeds"].items()}
    config["seeds"]["classification"] = [list(range(5000, 5012))]
    learners = [{"name": name, "index": index} for index, name in enumerate(("mean", "noisy", "level_noise"))]
    generic = [source("equivalent", EQUIVALENT, "cosmic_ray"), source("stateful", STATEFUL, "cosmic_ray"),
               source("third", WRONG.replace("0.5", "0.25"), "cosmic_ray")]
    result = runner.run_part2(config, learners[1:2], [source("wrong", WRONG, "catalogue")], generic, learners)
    classes = result["shift_classification"]
    assert classes["mean"]["class"] == "exact" and classes["mean"]["calibrated_fails"] is None
    assert classes["noisy"]["public_pass"] == 0 and classes["noisy"]["class"] == "in distribution"
    assert classes["level_noise"]["class"] == "not applicable" and classes["level_noise"]["calibrated_cases"] == 12
    noisy = result["learners"]["noisy"]
    assert noisy["shift_class"] == "in distribution" and noisy["checks"] == ["calibrated_shift", "scores"]
    # Generic sources are screened in batches of max(sources wanted, workers), here 4; selection keeps their order.
    assert noisy["generic_screened"] == ["equivalent", "stateful", "third"]
    assert noisy["generic_selected"] == ["stateful"]
    clean = noisy["clean"]["toy_white_noise"]
    assert clean["calibrated_shift"]["cases"] == 6 and set(clean["alarms"]) == {"output"}
    rows = {item["id"]: item for item in noisy["sources"]}
    decisions = rows["stateful"]["conditions"]["toy_white_noise"]["decisions"]
    assert set(decisions) == {"output", "loss", "answer_key", "paired", "calibrated_shift"}
    assert decisions["calibrated_shift"]["cosmic_ray"]["family_size"] == 1


def test_records_tables_and_the_manifest_check(part1, tmp_path):
    _, result, _ = part1
    text = json.dumps(runner._strip(result), allow_nan=False, sort_keys=True)
    assert '"ranks"' not in text and '"differences"' not in text
    rows = runner.tables(result)
    assert {row["stage"] for row in rows["funnel"]} >= {"generated", "runnable", "caught_scalar_shift"}
    stateful = [row for row in rows["decisions"] if row["source"] == "stateful" and row["check"] == "loss"]
    assert sorted(row["family"] for row in stateful) == ["catalogue", "catalogue", "study2", "study2"]
    assert len(rows["clean"]) == 8
    runner._write_csv(tmp_path / "decisions.csv", rows["decisions"])
    assert (tmp_path / "decisions.csv").read_text().startswith("learner,source,condition,check,family")
    subject = tmp_path / "subject.py"
    subject.write_text(CLEAN)
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources/a.py").write_text(WRONG)
    from experiments.common import sha256

    entry = {"file": "sources/a.py", "sha256": sha256(tmp_path / "sources/a.py"),
             "labels": [{"label": "x#0", "kind": "cosmic_ray", "family": "NumberReplacer"}]}
    manifest = {"subject": {"sha256": sha256(subject)}, "sources": [entry]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    _, sources = runner.manifest_sources(tmp_path / "manifest.json", subject)
    assert sources[0]["family"] == "NumberReplacer" and sources[0]["source"] == WRONG
    (tmp_path / "sources/a.py").write_text(WRONG + "\n")
    with pytest.raises(ValueError, match="differs"):
        runner.manifest_sources(tmp_path / "manifest.json", subject)
    subject.write_text(CLEAN + "\n")
    with pytest.raises(ValueError, match="frozen subject"):
        runner.manifest_sources(tmp_path / "manifest.json", subject)
