"""The stored records of Study 3 (experiments/results-v3).

Each record matches its manifest, the registered protocol and the lock file it was run with, and was made on one
numerical thread per process. The counts that experiments/README.md states are read back from the records here.
Nothing is computed again: no source, learner or model is run. The last test of the warning table builds toy unit
records and tallies them.
"""
import csv
import hashlib
import json

import pytest

from experiments import run_study3_mutants as runner
from experiments import study3_common as common
from experiments.common import sha256

RESULTS = common.RESULTS
PARTS = ("part-1", "part-2", "part-3", "part-4", "part-5")
LEARNERS = ("hgb", "mlp_window_standardized", "mlp_raw")
LOCAL_MODELS = ("last_value", "seasonal_naive", "ets", "auto_arima", "theta", "hgb_lags", "mlp_lags",
                "context_parroting")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def lock(name):
    pins = {}
    for line in (common.ROOT / name).read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            package, version = line.split("==")
            pins[package.strip().lower().replace("_", "-")] = version.strip()
    return pins


def protocol_digest_at_the_runs():
    """The digest of the protocol as the runs read it: registered, and before the Chronos revision was written."""
    protocol = load(common.PROTOCOL)
    for model in protocol["part_6_zoo"]["models"]:
        if model["id"] == "chronos_small":
            model["revision"] = None
    return hashlib.sha256((json.dumps(protocol, indent=2) + "\n").encode("utf-8")).hexdigest()


def one_thread(runtime):
    return runtime["thread_limit"] == 1 and bool(runtime["threadpools"]) and all(
        pool["num_threads"] == 1 for pool in runtime["threadpools"])


@pytest.mark.parametrize("part", PARTS)
def test_a_part_matches_its_manifest_the_protocol_and_the_study_lock(part):
    folder, protocol = RESULTS / part, common.load_protocol()
    manifest, results = load(folder / "manifest.json"), load(folder / "results.json")
    assert manifest["results_sha256"] == sha256(folder / "results.json")
    assert manifest["protocol_sha256"] == protocol_digest_at_the_runs()
    assert results["protocol_version"] == "3.0" and results["part"] == part
    assert results["study_id"] == protocol["study_id"]
    for item in protocol["frozen_inputs"].values():
        assert manifest["frozen_inputs_sha256"][item["path"]] == item["sha256"]
    pins = lock("requirements-study.txt")
    assert manifest["versions"] == {name: pins[name] for name in manifest["versions"]}
    assert manifest["python"].startswith("3.13.")
    assert one_thread(load(folder / "runtime.json"))


def rows(items):
    """Table rows in an order and a form that do not depend on how they were written."""
    return sorted(tuple(sorted((key, "" if value is None else str(value)) for key, value in row.items()))
                  for row in items)


@pytest.mark.parametrize("part", ("part-1", "part-2"))
def test_a_mutant_part_ran_the_committed_mutants_with_the_frozen_files(part):
    protocol, manifest = common.load_protocol(), load(RESULTS / part / "manifest.json")
    registered = protocol["part_1_generated_mutants"]
    assert manifest["mutant_manifest_sha256"] == registered["sources"]["manifest"]["sha256"]
    assert load(RESULTS / part / "results.json")["mutant_manifest_sha256"] == manifest["mutant_manifest_sha256"]
    for name, digest in registered["subject"]["frozen_files_sha256"].items():
        assert manifest["source_files"][name] == digest
    assert {"experiments/study3_harness.py", "experiments/study3_threads.py"} <= set(manifest["source_files"])
    assert load(RESULTS / part / "runtime.json")["resumed"] is False


def test_part_1_tables_are_those_of_its_results():
    folder = RESULTS / "part-1"
    results = load(folder / "results.json")
    assert results["generic_generated"] == results["generic_run"] == 119
    for name, built in runner.tables(results["ridge"]).items():
        assert rows(built) == rows(table(folder / f"{name}.csv")), name


def test_part_1_counts_stated_in_the_readme():
    funnel = {(row["kind"], row["stage"]): int(row["count"]) for row in table(RESULTS / "part-1/funnel.csv")}
    stages = ("generated", "runnable", "equivalent", "not_equivalent", "caught_output", "caught_scalar_shift",
              "caught_target_shift", "caught_second_implementation", "caught_loss", "caught_answer_key",
              "caught_paired")
    assert [funnel["cosmic_ray", stage] for stage in stages] == [119, 64, 20, 44, 3, 22, 22, 44, 37, 40, 43]
    assert [funnel["catalogue", stage] for stage in stages] == [16, 16, 1, 15, 0, 2, 4, 15, 13, 14, 15]
    assert [funnel["study2", stage] for stage in stages] == [4, 4, 0, 4, 0, 1, 2, 4, 4, 4, 4]
    ridge = load(RESULTS / "part-1/results.json")["ridge"]
    assert ridge["funnel"]["cosmic_ray"]["not_runnable_reasons"] == {"exception on every input": 55}
    assert [ridge["funnel"][kind]["screen_misses"] for kind in ("cosmic_ray", "catalogue", "study2")] == [0, 0, 0]
    assert all(int(row["count"]) == 0 and int(row["cases"]) == 332 for row in table(RESULTS / "part-1/clean.csv"))
    false_alarms = {alarm: ridge["curves"][alarm]["false_alarm"]["intervals"]["12"] for alarm in ("loss",
                                                                                                  "answer_key")}
    assert (false_alarms["loss"]["count"], false_alarms["answer_key"]["count"]) == (29, 33)
    assert false_alarms["loss"]["n"] == false_alarms["answer_key"]["n"] == 558
    assert not any(row["conditions"][name]["not_evaluated"] for row in ridge["sources"] if row.get("conditions")
                   for name in row["conditions"])


def test_part_2_tables_are_those_of_its_results():
    folder = RESULTS / "part-2"
    results = load(folder / "results.json")
    assert tuple(sorted(results["learners"])) == tuple(sorted(LEARNERS))
    built = {"funnel": [], "decisions": [], "clean": []}
    for learner in results["learners"].values():
        for name, items in runner.tables(learner).items():
            built[name] += items
    for name, items in built.items():
        assert rows(items) == rows(table(folder / f"{name}.csv")), name
    classes = [{"learner": name, **row} for name, row in results["shift_classification"].items()]
    assert rows(classes) == rows(table(folder / "shift_classification.csv"))


def test_part_2_classification_stated_in_the_readme():
    classes = {row["learner"]: row for row in table(RESULTS / "part-2/shift_classification.csv")}
    stated = {"ridge": ("96", "", "exact"), "hgb": ("96", "", "exact"),
              "mlp_window_standardized": ("0", "4", "in distribution"), "mlp_raw": ("0", "51", "not applicable")}
    assert {name: (row["public_pass"], row["calibrated_fails"], row["class"])
            for name, row in classes.items()} == stated
    assert all(row["public_cases"] == "96" and row["public_errors"] == "0" for row in classes.values())
    assert all(classes[name]["calibrated_cases"] == "96" and classes[name]["calibrated_errors"] == "0"
               for name in ("mlp_window_standardized", "mlp_raw"))
    assert round(float(classes["mlp_window_standardized"]["binomial_p_value"]), 2) == 0.71
    assert float(classes["mlp_raw"]["binomial_p_value"]) < 0.001
    expected = common.load_protocol()["part_2_learners"]["learners"]
    assert all(classes[name]["class"] == expected[name]["expected_class"] for name in LEARNERS)


def test_part_2_counts_stated_in_the_readme():
    from experiments.study3_stats import rate

    results = load(RESULTS / "part-2/results.json")["learners"]
    funnel = {(row["learner"], row["kind"], row["stage"]): int(row["count"])
              for row in table(RESULTS / "part-2/funnel.csv")}
    kinds = ("cosmic_ray", "catalogue", "study2")
    mutants = {entry["sha256"]: entry for entry in load(RESULTS / "mutants/manifest.json")["sources"]}
    defined_in = {source["id"]: mutants[source["sha256"]]["labels"][0].get("definition")
                  for source in load(RESULTS / "part-1/results.json")["ridge"]["sources"]}

    def caught(learner, check):
        return [funnel.get((learner, kind, f"caught_{check}")) for kind in kinds]

    for learner in LEARNERS:
        for stage, counts in (("generated", [12, 16, 4]), ("runnable", [12, 16, 4]), ("equivalent", [0, 4, 1]),
                              ("not_equivalent", [12, 12, 3])):
            assert [funnel[learner, kind, stage] for kind in kinds] == counts
        assert caught(learner, "output") == [3, 0, 0]
        screen = [entry["sizes"]["24"] for entry in results[learner]["generic_screen"].values()]
        assert len(screen) == 84 and sum(not entry["runnable"] for entry in screen) == 24
        assert sum(entry["runnable"] and entry["equivalent"] for entry in screen) == 48
        unchanged = [name for name, entry in results[learner]["generic_screen"].items()
                     if entry["sizes"]["24"]["runnable"] and entry["sizes"]["24"]["equivalent"]]
        assert sum(defined_in[name] in ("fit_scaler", "apply_scaler", "invert_scaler") for name in unchanged) == 34
        assert results[learner]["generic_selected"] == results["hgb"]["generic_selected"]
        assert len(results[learner]["generic_selected"]) == 12
        assert [results[learner]["funnel"][kind]["screen_misses"] for kind in kinds] == [0, 0, 0]
        conditions = [row for source in results[learner]["sources"]
                      for row in (source.get("conditions") or {}).values()]
        assert len(conditions) == 25 * 6
        assert not any(row["timed_out"] or row["crashed"] or row["not_evaluated"] for row in conditions)
    assert [caught("hgb", check) for check in ("scalar_shift", "target_shift", "loss", "answer_key", "paired")] == [
        [6, 0, 0], [6, 2, 1], [8, 11, 3], [9, 12, 3], [11, 12, 3]]
    assert [caught("mlp_window_standardized", check) for check in ("calibrated_shift", "loss", "answer_key",
                                                                     "paired")] == [
        [0, 0, 0], [5, 8, 3], [5, 7, 3], [6, 9, 3]]
    assert [caught("mlp_raw", check) for check in ("loss", "answer_key", "paired")] == [
        [8, 6, 3], [8, 7, 3], [11, 12, 3]]
    assert caught("mlp_raw", "scalar_shift") == caught("mlp_window_standardized", "target_shift") == [None] * 3
    assert caught("hgb", "calibrated_shift") == caught("mlp_raw", "calibrated_shift") == [None] * 3

    clean = table(RESULTS / "part-2/clean.csv")
    assert len(clean) == 30 and all(int(row["count"]) == 0 and int(row["cases"]) == 332 for row in clean)
    alarms = {learner: [results[learner]["curves"][alarm]["false_alarm"]["intervals"]["12"] for alarm in (
        "loss", "answer_key")] for learner in LEARNERS}
    assert {learner: [item["count"] for item in items] for learner, items in alarms.items()} == {
        "hgb": [30, 17], "mlp_window_standardized": [28, 32], "mlp_raw": [29, 22]}
    assert all(item["n"] == 558 for items in alarms.values() for item in items)

    shift = [row["calibrated_shift"] for row in results["mlp_window_standardized"]["clean"].values()]
    assert (sum(row["fails"] for row in shift), sum(row["cases"] for row in shift),
            sum(row["errors"] for row in shift)) == (27, 558, 0)
    assert [round(100 * bound, 1) for bound in rate(27, 558)["ci95"]] == [3.3, 6.9]
    decided = [row["statistical"]["calibrated_shift"] for source in results["mlp_window_standardized"]["sources"]
               for row in (source.get("conditions") or {}).values()]
    decided = [row for row in decided if row["cases"]]
    assert (len(decided), sum(row["alarms"] for row in decided), sum(row["cases"] for row in decided)) == (
        136, 608, 12648)
    assert sum(row["p_value"] < 0.05 for row in decided) == 5
    assert round(load(RESULTS / "part-2/runtime.json")["seconds"] / 3600, 1) == 5.5


def test_warning_counts_stated_in_the_readme():
    part_2 = {(row["learner"], row["subject"], row["phase"], row["stage"]): row
              for row in table(RESULTS / "part-2/warnings.csv")}
    assert set(next(iter(part_2.values()))) - {"learner", "subject", "phase", "stage", "records",
                                               "records_with_warnings"} == {"ConvergenceWarning"}
    window = part_2["mlp_window_standardized", "clean", "full", "evaluation"]
    assert (window["records"], window["records_with_warnings"], window["ConvergenceWarning"]) == (
        "558", "558", str(558 * 41))
    assert all(row["records"] == row["records_with_warnings"] for key, row in part_2.items()
               if key[:2] == ("mlp_window_standardized", "clean"))
    raw = [part_2["mlp_raw", "clean", "full", stage] for stage in ("calibration", "evaluation")]
    assert (sum(int(row["records_with_warnings"]) for row in raw), sum(int(row["records"]) for row in raw)) == (
        888, 1992)
    assert all(row["records_with_warnings"] == "0" for key, row in part_2.items() if key[0] in ("hgb", "ridge"))
    part_1 = {(row["subject"], row["phase"], row["stage"]): row for row in table(RESULTS / "part-1/warnings.csv")}
    assert all(row["records_with_warnings"] == "0" for key, row in part_1.items() if key[0] == "clean")
    screen = part_1["mutants", "screen", "screen"]
    assert (screen["records_with_warnings"], screen["records"]) == ("191", "6528")
    assert int(screen["RuntimeWarning"]) > 0


def test_the_warning_table_counts_records_and_categories(tmp_path):
    from experiments import study3_warnings

    clean, mutant = tmp_path / "units/mlp/clean", tmp_path / "units/mlp/0123abcd"
    clean.mkdir(parents=True)
    mutant.mkdir()
    records = [{"kind": "load"}, {"kind": "warmup", "warnings": {"ConvergenceWarning": 1}},
               {"kind": "case", "stage": "evaluation", "warnings": {"ConvergenceWarning": 41}},
               {"kind": "case", "stage": "evaluation", "warnings": {}}, {"kind": "done"}]
    (clean / "full-0.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (mutant / "screen-3.jsonl").write_text(json.dumps({"kind": "case", "stage": "screen",
                                                       "warnings": {"RuntimeWarning": 2}}) + "\n")
    expected = [
        {"learner": "mlp", "subject": "clean", "phase": "full", "stage": "evaluation", "records": 2,
         "records_with_warnings": 1, "ConvergenceWarning": 41, "RuntimeWarning": 0},
        {"learner": "mlp", "subject": "clean", "phase": "full", "stage": "warmup", "records": 1,
         "records_with_warnings": 1, "ConvergenceWarning": 1, "RuntimeWarning": 0},
        {"learner": "mlp", "subject": "mutants", "phase": "screen", "stage": "screen", "records": 1,
         "records_with_warnings": 1, "ConvergenceWarning": 0, "RuntimeWarning": 2}]
    assert study3_warnings.tally(tmp_path / "units") == expected
    study3_warnings.main([str(tmp_path)])
    assert table(tmp_path / "warnings.csv") == [{key: str(value) for key, value in row.items()}
                                                for row in expected]
    with pytest.raises(SystemExit):
        study3_warnings.main([str(tmp_path / "units")])


def test_part_3_counts_stated_in_the_readme():
    variants = {(row["variant"], row["method"]): row for row in table(RESULTS / "part-3/variants.csv")}
    flagged = {key: int(row["suites_flagged"]) for key, row in variants.items()}
    assert all(int(row["n"]) == 93 for row in variants.values())
    assert [flagged["clean", method] for method in ("raw", "holm", "fisher")] == [17, 2, 5]
    assert [flagged["partial_shuffle", method] for method in ("raw", "holm", "fisher")] == [93, 93, 93]
    assert [flagged["reduced_spread", method] for method in ("raw", "holm", "fisher")] == [93, 68, 93]
    assert [int(variants["partial_shuffle", method]["true_null_rejections"]) for method in ("raw", "holm")] == [4, 3]
    conditions = {(row["variant"], row["condition"]): row for row in table(RESULTS / "part-3/conditions.csv")}
    reduced = [int(row["holm"]) for (variant, _), row in conditions.items() if variant == "reduced_spread"]
    assert sorted(reduced) == [14, 32, 34, 45]


def test_part_4_counts_stated_in_the_readme():
    cells = table(RESULTS / "part-4/cells.csv")
    assert len(cells) == 240 and all(int(row["repetitions"]) == 381 for row in cells)
    on_margin = [row for row in cells if float(row["multiple"]) == 1.0]
    assert len(on_margin) == 32
    for verdict, low, high, above in (("FAIL", 0, 29, 1), ("WITHIN_TOLERANCE", 5, 111, 16)):
        counts = [int(row[f"{verdict}_count"]) for row in on_margin]
        assert (min(counts), max(counts)) == (low, high)
        assert sum(float(row[f"{verdict}_lower95"]) > 0.05 for row in on_margin) == above
    beyond = max(int(row["WITHIN_TOLERANCE_count"]) for row in cells if float(row["multiple"]) > 1.0)
    below = max(int(row["FAIL_count"]) for row in cells if float(row["multiple"]) < 1.0)
    assert (beyond, below) == (95, 18)
    at_zero = {(row["margin"], row["n_paths"]): 0 for row in cells if float(row["multiple"]) == 0.0}
    for row in cells:
        if float(row["multiple"]) == 0.0:
            at_zero[row["margin"], row["n_paths"]] += int(row["WITHIN_TOLERANCE_count"])
    assert at_zero == {("0.25", "64"): 670, ("0.25", "256"): 1482, ("0.4375", "64"): 1443, ("0.4375", "256"): 1524}


def test_part_5_numbers_stated_in_the_readme():
    rows = {(row["pattern"], row["noise_sd"]): row for row in table(RESULTS / "part-5/table.csv")}
    assert len(rows) == 20 and all(int(row["series"]) == 200 for row in rows.values())
    medians = {key: round(float(row["primary_median"]), 2) for key, row in rows.items()}
    assert medians["noise", "0.5"] == medians["level", "0.5"] == 0.86
    assert (medians["trend", "0.5"], medians["trend", "2.0"]) == (24.0, 2.35)
    assert (medians["seasonal", "0.5"], medians["seasonal", "2.0"]) == (1.12, 1.03)
    assert (medians["level_shift", "0.5"], medians["level_shift", "2.0"]) == (0.83, 0.83)
    assert round(float(rows["level_shift", "0.5"]["history_median"]), 1) == 81.1
    assert (medians["trend_seasonal", "0.5"], medians["trend_seasonal", "2.0"]) == (24.28, 2.47)
    assert (medians["random_walk", "0.5"], medians["random_walk", "2.0"]) == (0.85, 0.83)
    assert (medians["random_walk_drift", "0.5"], medians["random_walk_drift", "2.0"]) == (2.46, 1.52)
    stationary = [medians[pattern, sd] for pattern in ("noise", "level", "mean_reverting", "cycles")
                  for sd in ("0.5", "2.0")]
    assert 0.86 <= min(stationary) and max(stationary) <= 0.98
    correlations = load(RESULTS / "part-5/results.json")["correlations"]
    assert round(correlations["wpe_vs_floor"]["rho"], 2) == 0.26
    assert round(correlations["wpe_vs_scale_free_floor"]["rho"], 2) == 0.72
    assert all(int(row[name]) == 0 for row in rows.values()
               for name in ("primary_undefined", "history_undefined", "wpe_undefined"))


@pytest.mark.parametrize("model", LOCAL_MODELS)
def test_a_zoo_model_matches_its_manifest_and_ran_on_one_thread(model):
    folder = RESULTS / "zoo"
    manifest = load(folder / f"{model}.manifest.json")
    assert manifest["model_id"] == model
    assert manifest["baseline_sha256"] == sha256(folder / f"{model}.json")
    assert manifest["table_sha256"] == sha256(folder / f"{model}.table.csv")
    assert manifest["protocol_sha256"] == protocol_digest_at_the_runs()
    assert manifest["scorecard"] == {"seed": 14_000_000, "instances": 30, "other_arguments": "defaults"}
    pins = lock("experiments/zoo/requirements-zoo.txt" if manifest["versions"]["statsforecast"]
                else "requirements-study.txt")
    assert all(pins[name] == version for name, version in manifest["versions"].items() if version)
    assert one_thread(load(folder / f"{model}.runtime.json"))
    rows = table(folder / f"{model}.table.csv")
    assert len(rows) == 20 and all(row["failed_checks"] in ("", "0", "[]") for row in rows)
    assert all(int(row["instances"]) == 30 for row in rows)


def test_the_pretrained_model_ran_at_the_registered_revision_on_one_thread():
    folder = RESULTS / "zoo"
    manifest = load(folder / "chronos_small.manifest.json")
    entry = next(row for row in load(common.PROTOCOL)["part_6_zoo"]["models"] if row["id"] == "chronos_small")
    assert manifest["model_id"] == "chronos_small"
    assert manifest["baseline_sha256"] == sha256(folder / "chronos_small.json")
    assert manifest["table_sha256"] == sha256(folder / "chronos_small.table.csv")
    # This run came after the commit that wrote the revision, so it read the protocol as it stands.
    assert manifest["protocol_sha256"] == sha256(common.PROTOCOL) != protocol_digest_at_the_runs()
    assert manifest["scorecard"] == {"seed": 14_000_000, "instances": 30, "other_arguments": "defaults"}
    pretrained = manifest["pretrained_environment"]
    assert pretrained["model_id"] == "amazon/chronos-t5-small"
    assert pretrained["resolved_commit"] == entry["revision"] and len(entry["revision"]) == 40
    assert pretrained["torch_threads"] == 1 and one_thread(load(folder / "chronos_small.runtime.json"))
    assert pretrained["file_sha256"] and all(len(digest) == 64 for digest in pretrained["file_sha256"].values())
    pins = lock("experiments/zoo/requirements-chronos.txt")
    assert pins == {"chronos-forecasting": manifest["versions"]["chronos-forecasting"]}
    assert f"chronos-forecasting=={pins['chronos-forecasting']}" in pretrained["pip_freeze"]
    rows = table(folder / "chronos_small.table.csv")
    assert len(rows) == 20 and all(row["failed_checks"] in ("", "0", "[]") for row in rows)
    assert all(int(row["instances"]) == 30 for row in rows)


def test_zoo_scores_stated_in_the_readme():
    def scores(model, column="score"):
        return {(row["pattern"], row["noise_sd"]): round(float(row[column]))
                for row in table(RESULTS / f"zoo/{model}.table.csv")}

    marks = scores("ets", "attainable")
    assert scores("ets")["cycles", "0.5"] == 35 and marks["cycles", "0.5"] == 93
    assert scores("hgb_lags")["trend", "0.5"] == 5 and marks["trend", "0.5"] == 95
    assert scores("hgb_lags")["level_shift", "0.5"] == 93 and marks["level_shift", "0.5"] == 97
    assert scores("theta")["level_shift", "0.5"] == 22
    assert scores("mlp_lags")["random_walk", "0.5"] == 4 and marks["random_walk", "0.5"] == 100
    chronos = scores("chronos_small")
    resembling = [chronos[pattern, sd] for pattern in ("noise", "level", "trend", "seasonal", "trend_seasonal")
                  for sd in ("0.5", "2.0")]
    others = [chronos[pattern, sd] for pattern in ("level_shift", "mean_reverting", "random_walk", "random_walk_drift")
              for sd in ("0.5", "2.0")]
    assert (min(resembling), max(resembling)) == (23, 99) and (min(others), max(others)) == (59, 97)
    assert (chronos["cycles", "0.5"], chronos["cycles", "2.0"]) == (73, 78)
    assert min(chronos, key=chronos.get) == ("trend", "0.5")
    assert (chronos["mean_reverting", "0.5"], chronos["mean_reverting", "2.0"]) == (92, 97)
    assert (marks["mean_reverting", "0.5"], marks["mean_reverting", "2.0"]) == (93, 96)
