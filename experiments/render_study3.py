"""Read and check the Study 3 records (protocol 3.0) for the paper.

    python -m experiments.render_study3

Writes paper/generated-study3.tex, the Study 3 part of the Results section. Nothing is run again: every record
under experiments/results-v3 is checked against its manifest, the registered protocol and the frozen files, every
statement the manuscript makes from the records is checked, and the numbers that the hand-written prose quotes
become LaTeX macros, which experiments.render_supplement appends to paper/generated-macros.tex.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import stdev

from timegym.statistics import wilson_interval

from .common import require_conclusion, sha256

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results-v3"
PROTOCOL = ROOT / "experiments/protocol-v3.json"
PARTS = ("part-1", "part-2", "part-3", "part-4", "part-5")
KINDS = ("cosmic_ray", "catalogue", "study2")
LEARNERS = ("hgb", "mlp_window_standardized", "mlp_raw")
SCALING_FUNCTIONS = ("fit_scaler", "apply_scaler", "invert_scaler")
CHECK_LABELS = {
    "output": "Output checks", "scalar_shift": "Common shift check", "target_shift": "Per-series shift check",
    "calibrated_shift": "Calibrated shift check", "second_implementation": "Second implementation",
    "loss": "Calibrated loss alarm", "answer_key": "Answer-key alarm",
    "paired": "Paired check with the clean pipeline",
}
PART_1_CHECKS = ("output", "scalar_shift", "target_shift", "second_implementation", "loss", "answer_key", "paired")
PART_2_CHECKS = ("output", "scalar_shift", "target_shift", "calibrated_shift", "loss", "answer_key", "paired")
LEARNER_LABELS = {"ridge": "Ridge with global scaling (control)", "hgb": "Histogram gradient boosting",
                  "mlp_window_standardized": "Perceptron, window-standardized lags",
                  "mlp_raw": "Perceptron, raw lags"}
LEARNER_HEADS = {"hgb": "Boosting", "mlp_window_standardized": "Perceptron, standardized",
                 "mlp_raw": "Perceptron, raw"}
CLASS_LABELS = {"exact": "exact", "in distribution": "in distribution", "not applicable": "not applicable"}
SUITE_CONDITIONS = {"factorized_white_noise": "White noise: independent targets",
                    "correlated_white_noise": "White noise: correlated targets",
                    "negative_ar1": "Negative AR(1)",
                    "correlated_random_walk": "Correlated random walk with drift"}
VARIANT_LABELS = {"clean": "Exact-reference control", "partial_shuffle": "Partial dependence shuffle",
                  "reduced_spread": "Reduced spread"}
PATTERN_LABELS = {"noise": "Pure noise", "level": "Noise around a fixed level", "trend": "Straight-line trend",
                  "seasonal": "Repeating seasonal cycle", "trend_seasonal": "Trend plus seasonal cycle",
                  "level_shift": "Sudden level shift", "mean_reverting": "Mean reversion (AR(1))",
                  "cycles": "Irregular cycles (AR(2))", "random_walk": "Random walk",
                  "random_walk_drift": "Random walk with drift"}
STATIONARY = ("noise", "level", "mean_reverting", "cycles")
MODEL_HEADS = {"last_value": "Last", "seasonal_naive": "Seas.", "ets": "ETS", "auto_arima": "ARIMA",
               "theta": "Theta", "hgb_lags": "Boost.", "mlp_lags": "Perc.", "chronos_small": "Chronos",
               "context_parroting": "Parrot"}
PRETRAINED = "chronos_small"
# The protocol's part_6_zoo.declared_overlap, written before the run: how each kind of series relates to the
# synthetic data the pretrained model was trained on.
PRETRAINING_OVERLAP = {"resembling": ("noise", "level", "trend", "seasonal", "trend_seasonal"),
                       "partly": ("cycles",),
                       "not_contained": ("level_shift", "mean_reverting", "random_walk", "random_walk_drift")}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rate(count, n):
    """Count, size and Wilson 95% interval, as the records store a rate."""
    return wilson_interval([True] * count + [False] * (n - count))


def digest_at_the_runs(protocol):
    """The digest of the protocol as Parts 1 to 5 and the eight local models read it.

    Those runs came before the commit that wrote the Chronos revision, the one value the protocol still allowed
    to change; with that value set back to null, the file is the one they hashed.
    """
    earlier = json.loads(json.dumps(protocol))
    for model in earlier["part_6_zoo"]["models"]:
        if model["id"] == PRETRAINED:
            model["revision"] = None
    return hashlib.sha256((json.dumps(earlier, indent=2) + "\n").encode("utf-8")).hexdigest()


def validate_part(part, results, manifest, protocol, folder, root=ROOT):
    """Raise unless a part's record, its manifest, the protocol and the frozen inputs agree."""
    if sha256(folder / "results.json") != manifest["results_sha256"]:
        raise ValueError(f"{part}: the results do not match their manifest fingerprint")
    if manifest["protocol_sha256"] != digest_at_the_runs(protocol):
        raise ValueError(f"{part}: the record was not made from the registered protocol")
    if (results["protocol_version"], results["part"], results["study_id"]) != ("3.0", part, protocol["study_id"]):
        raise ValueError(f"{part}: not a record of this part of protocol 3.0")
    for item in protocol["frozen_inputs"].values():
        if manifest["frozen_inputs_sha256"][item["path"]] != item["sha256"]:
            raise ValueError(f"{part}: frozen input {item['path']} differs from the registered one")
    if part in ("part-1", "part-2"):
        registered = protocol["part_1_generated_mutants"]
        if not (results["mutant_manifest_sha256"] == manifest["mutant_manifest_sha256"]
                == registered["sources"]["manifest"]["sha256"]):
            raise ValueError(f"{part}: the mutants are not the committed set")
        for name, digest in registered["subject"]["frozen_files_sha256"].items():
            if manifest["source_files"][name] != digest or sha256(root / name) != digest:
                raise ValueError(f"{part}: frozen file {name} changed")


def validate_model(model, manifest, protocol, folder, protocol_path=PROTOCOL):
    """Raise unless a scorecard's files match its manifest and the protocol it was made from."""
    if manifest["model_id"] != model:
        raise ValueError(f"{model}: the manifest names another model")
    if manifest["baseline_sha256"] != sha256(folder / f"{model}.json"):
        raise ValueError(f"{model}: the saved scorecard does not match its manifest")
    if manifest["table_sha256"] != sha256(folder / f"{model}.table.csv"):
        raise ValueError(f"{model}: the scorecard table does not match its manifest")
    registered = protocol["part_6_zoo"]
    if manifest["scorecard"] != {"seed": 14_000_000, "instances": 30, "other_arguments": "defaults"}:
        raise ValueError(f"{model}: not the registered scorecard")
    if model == PRETRAINED:
        entry = next(row for row in registered["models"] if row["id"] == model)
        pretrained = manifest["pretrained_environment"]
        if manifest["protocol_sha256"] != sha256(protocol_path) or pretrained["resolved_commit"] != entry["revision"]:
            raise ValueError(f"{model}: the scorecard was not made from the registered model revision")
    elif manifest["protocol_sha256"] != digest_at_the_runs(protocol):
        raise ValueError(f"{model}: the record was not made from the registered protocol")


def load_study3(directory=RESULTS, root=ROOT):
    """Every Study 3 record, validated: parts, mutant manifest, warning tables and the scorecards."""
    directory = Path(directory)
    protocol_path = root / "experiments/protocol-v3.json"
    protocol = load(protocol_path)
    if protocol["protocol_version"] != "3.0" or protocol["status"] != "registered":
        raise ValueError("Expected the registered protocol 3.0")
    records = {"protocol": protocol, "parts": {}, "manifests": {}, "zoo": {}, "zoo_manifests": {}}
    for part in PARTS:
        results, manifest = load(directory / part / "results.json"), load(directory / part / "manifest.json")
        validate_part(part, results, manifest, protocol, directory / part, root=root)
        records["parts"][part], records["manifests"][part] = results, manifest
    mutants = directory / "mutants/manifest.json"
    if sha256(mutants) != protocol["part_1_generated_mutants"]["sources"]["manifest"]["sha256"]:
        raise ValueError("The mutant manifest is not the registered one")
    records["mutants"] = load(mutants)
    # The warning tables were tallied after the run from unit records that are not kept; no manifest lists them.
    records["warnings"] = {part: table(directory / part / "warnings.csv") for part in ("part-1", "part-2")}
    for row in protocol["part_6_zoo"]["models"]:
        model = row["id"]
        manifest = load(directory / "zoo" / f"{model}.manifest.json")
        validate_model(model, manifest, protocol, directory / "zoo", protocol_path)
        records["zoo"][model] = table(directory / "zoo" / f"{model}.table.csv")
        records["zoo_manifests"][model] = manifest
    records["subject_lines"] = len((root / "experiments/study3_subject.py").read_text().splitlines())
    return records


def caught_flags(source, check, kind):
    """Per condition, whether ``check`` caught ``source`` as a member of the Holm family of ``kind``."""
    flags = []
    for row in source["conditions"].values():
        decision = row["decisions"][check]
        flags.append(bool(decision["caught"]) if "caught" in decision else bool(decision[kind]["holm"]))
    return flags


def kinds_of(source):
    return {label["kind"] for label in source["labels"]}


def caught_summary(learner, checks):
    """Per kind and check: mutants caught in at least one condition, in every condition, and pairs caught."""
    out = {}
    for kind in KINDS:
        members = [source for source in learner["sources"] if source.get("conditions") and kind in kinds_of(source)]
        stages = learner["funnel"][kind]["stages"]
        out[kind] = {"mutants": len(members), "pairs": sum(len(source["conditions"]) for source in members),
                     "generated": stages["generated"], "runnable": stages["runnable"],
                     "equivalent": stages["equivalent"], "checks": {}}
        require_conclusion(len(members) == stages["not_equivalent"],
                           f"the evaluated {kind} sources are the non-equivalent ones of the funnel")
        for check in checks:
            if f"caught_{check}" not in stages:
                continue
            flags = [caught_flags(source, check, kind) for source in members]
            row = {"any": sum(any(item) for item in flags), "all": sum(all(item) for item in flags),
                   "pairs": sum(sum(item) for item in flags)}
            require_conclusion(row["any"] == stages[f"caught_{check}"],
                               f"the per-condition decisions of {check} reproduce the funnel count for {kind}")
            out[kind]["checks"][check] = row
    return out


def clean_alarms(learner):
    """Deterministic alarms on the clean pipeline (count, cases) and the two calibrated alarms at the 5% level."""
    deterministic = [(row["count"], row["cases"]) for condition in learner["clean"].values()
                     for row in condition["alarms"].values()]
    level = {alarm: learner["curves"][alarm]["false_alarm"]["intervals"]["12"] for alarm in ("loss", "answer_key")}
    return {"deterministic_alarms": sum(count for count, _ in deterministic),
            "deterministic_cases": sorted({cases for _, cases in deterministic}),
            "conditions": len(learner["clean"]), **level}


def part_1_summary(records):
    ridge = records["parts"]["part-1"]["ridge"]
    funnel = ridge["funnel"]["cosmic_ray"]
    summary = {"caught": caught_summary(ridge, PART_1_CHECKS), "clean": clean_alarms(ridge),
               "generated": funnel["stages"]["generated"],
               "raise_on_every_input": funnel["not_runnable_reasons"].get("exception on every input", 0),
               "not_runnable": funnel["stages"]["run"] - funnel["stages"]["runnable"],
               "screen_misses": sum(ridge["funnel"][kind]["screen_misses"] for kind in KINDS),
               "equivalent_at": {size: row["equivalent"] for size, row in funnel["equivalence_sensitivity"].items()},
               "subject_lines": records["subject_lines"]}
    members = [source for source in ridge["sources"] if source.get("conditions") and "cosmic_ray" in kinds_of(source)]
    groups = {"loop": ("predict",), "scaling": ("apply_scaler", "invert_scaler"),
              "lags": ("training_matrix", "prediction_row", "fit_scaler")}
    summary["by_function"] = {}
    for name, functions in groups.items():
        inside = [source for source in members
                  if next(label for label in source["labels"] if label["kind"] == "cosmic_ray")["definition"]
                  in functions]
        summary["by_function"][name] = {
            "mutants": len(inside),
            "scalar_shift": sum(any(caught_flags(source, "scalar_shift", "cosmic_ray")) for source in inside),
            "target_shift": sum(any(caught_flags(source, "target_shift", "cosmic_ray")) for source in inside)}
    require_conclusion(sum(row["mutants"] for row in summary["by_function"].values()) == len(members),
                       "the three groups of functions hold every evaluated Cosmic Ray mutant")
    catalogue = {}
    for source in ridge["sources"]:
        for label in source["labels"]:
            if label["kind"] == "catalogue":
                catalogue[label["label"]] = None if not source.get("conditions") else {
                    check: sum(caught_flags(source, check, "catalogue")) for check in PART_1_CHECKS}
    summary["catalogue"] = catalogue
    summary["unpaired"] = [source["id"] for source in members
                           if not any(row["statistical"]["paired"]["cases"] for row in source["conditions"].values())]
    # Pairs of a mutant and a condition where the mutant returned no forecast, so only the output checks could run.
    raised = [row for source in members for row in source["conditions"].values()
              if row["deterministic"]["second_implementation"]["cases"] == 0]
    summary["raised_pairs"] = len(raised)
    summary["raised_pairs_output"] = sum(bool(row["decisions"]["output"]["caught"]) for row in raised)
    summary["raised_mutants"] = len({source["id"] for source in members for row in source["conditions"].values()
                                     if row["deterministic"]["second_implementation"]["cases"] == 0})
    return summary


def part_2_summary(records):
    results = records["parts"]["part-2"]
    definitions = {source["id"]: next(label for label in source["labels"]
                                      if label["kind"] == "cosmic_ray").get("definition")
                   for source in records["parts"]["part-1"]["ridge"]["sources"] if "cosmic_ray" in kinds_of(source)}
    summary = {"classification": results["shift_classification"], "learners": {}}
    for name in LEARNERS:
        learner = results["learners"][name]
        screen = {key: entry["sizes"]["24"] for key, entry in learner["generic_screen"].items()}
        unchanged = [key for key, entry in screen.items() if entry["runnable"] and entry["equivalent"]]
        summary["learners"][name] = {
            "caught": caught_summary(learner, PART_2_CHECKS), "clean": clean_alarms(learner),
            "screened": len(screen), "not_runnable": sum(not entry["runnable"] for entry in screen.values()),
            "equivalent": len(unchanged),
            "equivalent_in_scaling": sum(definitions[key] in SCALING_FUNCTIONS for key in unchanged),
            "selected": list(learner["generic_selected"]),
            "shift_caught": sorted(
                source["id"] for source in learner["sources"]
                if source.get("conditions") and "cosmic_ray" in kinds_of(source)
                and any(any(caught_flags(source, check, "cosmic_ray"))
                        for check in ("scalar_shift", "target_shift", "calibrated_shift")
                        if check in next(iter(source["conditions"].values()))["decisions"])),
            "screen_misses": sum(learner["funnel"][kind]["screen_misses"] for kind in KINDS),
            "stopped": sum(bool(row["timed_out"] or row["crashed"] or row["not_evaluated"])
                           for source in learner["sources"] for row in (source.get("conditions") or {}).values())}
    window = results["learners"]["mlp_window_standardized"]
    shift = [row["calibrated_shift"] for row in window["clean"].values()]
    decided = [row["statistical"]["calibrated_shift"] for source in window["sources"]
               for row in (source.get("conditions") or {}).values()]
    decided = [row for row in decided if row["cases"]]
    summary["calibrated_shift"] = {
        "clean": rate(sum(row["fails"] for row in shift), sum(row["cases"] for row in shift)),
        "clean_errors": sum(row["errors"] for row in shift),
        "mutants": rate(sum(row["alarms"] for row in decided), sum(row["cases"] for row in decided)),
        "pairs": len(decided), "pairs_unadjusted": sum(row["p_value"] <= 0.05 for row in decided),
        "evaluated_sources": sum(bool(source.get("conditions")) for source in window["sources"])}
    rows = {(row["learner"], row["subject"], row["phase"], row["stage"]): row for row in records["warnings"]["part-2"]}
    categories = set(next(iter(rows.values()))) - {"learner", "subject", "phase", "stage", "records",
                                                    "records_with_warnings"}
    evaluation = rows["mlp_window_standardized", "clean", "full", "evaluation"]
    raw = [rows["mlp_raw", "clean", "full", stage] for stage in ("calibration", "evaluation")]
    summary["warnings"] = {
        "categories": sorted(categories),
        "window_cases": int(evaluation["records"]), "window_cases_warned": int(evaluation["records_with_warnings"]),
        "window_warnings": int(evaluation["ConvergenceWarning"]),
        "window_clean_always": all(row["records"] == row["records_with_warnings"] for key, row in rows.items()
                                   if key[:2] == ("mlp_window_standardized", "clean")),
        "raw_cases": sum(int(row["records"]) for row in raw),
        "raw_cases_warned": sum(int(row["records_with_warnings"]) for row in raw),
        "others_warned": sum(int(row["records_with_warnings"]) for key, row in rows.items()
                             if key[0] in ("hgb", "ridge"))}
    return summary


def part_3_summary(records):
    analysis = records["parts"]["part-3"]["analysis"]
    variants = {row["variant"]: row for row in analysis["variant_summaries"]}
    conditions = {(row["variant"], row["condition"]): row for row in analysis["condition_summaries"]}
    return {"suites": analysis["suites"], "alpha": analysis["alpha"], "variants": variants,
            "conditions": conditions, "condition_ids": list(analysis["conditions"])}


def part_4_summary(records):
    cells = records["parts"]["part-4"]["cells"]
    on_margin = [cell for cell in cells if cell["multiple"] == 1.0]
    summary = {"cells": len(cells), "repetitions": sorted({cell["repetitions"] for cell in cells}),
               "on_margin": on_margin, "all_cells": cells, "verdicts": {}}
    for verdict in ("FAIL", "WITHIN_TOLERANCE"):
        counts = [cell["verdicts"][verdict]["count"] for cell in on_margin]
        above = [cell for cell in on_margin if cell["verdicts"][verdict]["ci95"][0] > 0.05]
        summary["verdicts"][verdict] = {
            "low": min(counts), "high": max(counts), "above": len(above),
            "above_inflated": sum(cell["direction"] == "inflate" for cell in above),
            "above_dependent": sum(cell["condition"] in ("negative_ar1", "correlated_random_walk") for cell in above),
            "above_cells": above}
    summary["within_beyond"] = max(cell["verdicts"]["WITHIN_TOLERANCE"]["count"] for cell in cells
                                   if cell["multiple"] > 1.0)
    summary["within_beyond_multiple"] = min(cell["multiple"] for cell in cells if cell["multiple"] > 1.0)
    summary["fail_below"] = max(cell["verdicts"]["FAIL"]["count"] for cell in cells if cell["multiple"] < 1.0)
    at_zero = {}
    for cell in cells:
        if cell["multiple"] == 0.0:
            key = (cell["margin"], cell["n_paths"])
            found = at_zero.setdefault(key, {"count": 0, "n": 0})
            found["count"] += cell["verdicts"]["WITHIN_TOLERANCE"]["count"]
            found["n"] += cell["repetitions"]
    summary["at_zero"] = at_zero
    # The mean of the estimator against the squared distance, in standard errors of that mean, per cell.
    summary["bias_z"] = [(cell["mean_T"] - cell["d_squared"]) / (stdev(cell["T"]) / cell["repetitions"] ** 0.5)
                         for cell in cells]
    summary["reading"] = margin_reading(summary)
    return summary


def part_5_summary(records):
    results = records["parts"]["part-5"]
    rows = {(row["pattern"], row["noise_sd"]): row for row in results["table"]}
    medians = {key: row["primary_ratio"]["median"] for key, row in rows.items()}
    stationary = [medians[key] for key in rows if key[0] in STATIONARY]
    return {"rows": rows, "order": list(dict.fromkeys(row["pattern"] for row in results["table"])),
            "noise": sorted({row["noise_sd"] for row in results["table"]}),
            "series": sorted({row["series"] for row in results["table"]}),
            "medians": medians, "stationary_low": min(stationary), "stationary_high": max(stationary),
            "undefined": sum(row[name] for row in rows.values()
                             for name in ("primary_undefined", "history_undefined", "wpe_undefined")),
            "correlations": results["correlations"]}


def zoo_summary(records):
    models = list(records["zoo"])
    cells, marks, failed = {}, {}, 0
    for model, rows in records["zoo"].items():
        for row in rows:
            key = (row["pattern"], float(row["noise_sd"]))
            cells[model, key] = float(row["score"])
            mark = float(row["attainable"])
            if marks.setdefault(key, mark) != mark:
                raise ValueError("The scorecards disagree on a row's attainable mark")
            failed += row["failed_checks"] not in ("", "0", "[]")
            if int(row["instances"]) != 30:
                raise ValueError(f"{model}: not 30 instances per cell")
    order = list(dict.fromkeys(row["pattern"] for row in records["zoo"][models[0]]))
    noise = sorted({key[1] for key in marks})
    pretrained = records["zoo_manifests"][PRETRAINED]
    if sorted(pattern for group in PRETRAINING_OVERLAP.values() for pattern in group) != sorted(order):
        raise ValueError("The declared overlap does not place every kind of series once")
    groups = {name: [cells[PRETRAINED, (pattern, sd)] for pattern in patterns for sd in noise]
              for name, patterns in PRETRAINING_OVERLAP.items()}
    lowest = min((key for key in marks), key=lambda key: cells[PRETRAINED, key])
    return {"models": models, "cells": cells, "marks": marks, "failed": failed, "order": order, "noise": noise,
            "pretrained": {"revision": pretrained["pretrained_environment"]["resolved_commit"],
                           "torch": pretrained["pretrained_environment"]["torch_version"],
                           "threads": pretrained["pretrained_environment"]["torch_threads"],
                           "commit": pretrained["timegym_commit"], "groups": groups, "lowest": lowest}}


def summarize(records):
    """Every number the manuscript takes from the Study 3 records."""
    pythons = sorted({manifest["python"] for manifest in records["manifests"].values()})
    libraries = ("numpy", "pandas", "scipy", "scikit-learn")
    versions = {name: sorted({manifest["versions"][name] for manifest in records["manifests"].values()})
                for name in libraries}
    return {"part_1": part_1_summary(records), "part_2": part_2_summary(records),
            "part_3": part_3_summary(records), "part_4": part_4_summary(records),
            "part_5": part_5_summary(records), "zoo": zoo_summary(records),
            "records": {"mutant_sources": len(records["mutants"]["sources"]), "pythons": pythons,
                        "versions": versions}}


def margin_reading(summary):
    """How the jackknife standard error relates to the estimate (a reading made after the run)."""
    def correlation(first, second):
        mean_first, mean_second = sum(first) / len(first), sum(second) / len(second)
        cross = sum((a - mean_first) * (b - mean_second) for a, b in zip(first, second))
        return cross / (sum((a - mean_first) ** 2 for a in first) * sum((b - mean_second) ** 2 for b in second)) ** 0.5

    inflated = [cell for cell in summary["on_margin"] if cell["direction"] == "inflate"]
    shrunk = [cell for cell in summary["on_margin"] if cell["direction"] == "shrink"]
    ratios = sorted(sum(cell["standard_error"]) / len(cell["standard_error"]) / stdev(cell["T"])
                    for cell in summary["all_cells"])
    middle = len(ratios) // 2
    return {"ratio_median": (ratios[middle - 1] + ratios[middle]) / 2 if len(ratios) % 2 == 0 else ratios[middle],
            "inflated": [correlation(cell["T"], cell["standard_error"]) for cell in inflated],
            "shrunk": [correlation(cell["T"], cell["standard_error"]) for cell in shrunk]}


def check_study3_conclusions(summary):
    """Check every statement the manuscript makes from the Study 3 records."""
    first, second = summary["part_1"], summary["part_2"]
    ray, catalogue, bugs = (first["caught"][kind] for kind in KINDS)
    require_conclusion(first["not_runnable"] == first["raise_on_every_input"]
                       and first["generated"] == first["not_runnable"] + ray["runnable"]
                       and ray["runnable"] == ray["equivalent"] + ray["mutants"],
                       "every generated mutant raises on every input, is equivalent, or changes a forecast")
    require_conclusion(first["screen_misses"] == 0 and len(set(first["equivalent_at"].values())) == 1,
                       "no equivalent source differs on an evaluation case, and 6, 24 and 48 screening inputs "
                       "give the same number of equivalent mutants")
    require_conclusion(catalogue["equivalent"] == 1 and bugs["equivalent"] == 0
                       and first["catalogue"]["target_scaler_reused_across_contexts"] is None,
                       "the one equivalent catalogue mutant reuses the target scaler, and no Study 2 bug is "
                       "equivalent")
    for kind in KINDS:
        checks = first["caught"][kind]["checks"]
        require_conclusion(checks["second_implementation"]["any"] == first["caught"][kind]["mutants"],
                           f"the second implementation caught every non-equivalent {kind} source")
        require_conclusion(checks["answer_key"]["any"] >= checks["loss"]["any"]
                           and checks["answer_key"]["pairs"] > checks["loss"]["pairs"],
                           f"the answer-key alarm caught at least as many {kind} sources as the loss alarm, "
                           "in more pairs")
        require_conclusion(checks["output"]["any"] <= checks["scalar_shift"]["any"] <= checks["target_shift"]["any"]
                           < checks["second_implementation"]["any"],
                           f"for {kind}: output checks, common shift, per-series shift and second implementation "
                           "catch in that order")
    order = [ray["checks"][check]["any"]
             for check in ("output", "scalar_shift", "loss", "answer_key", "paired", "second_implementation")]
    require_conclusion(all(low < high for low, high in zip(order, order[1:])),
                       "on the generated mutants the output checks caught fewest, then the shift checks, the loss "
                       "alarm, the answer-key alarm, the paired check and the second implementation")
    require_conclusion(ray["checks"]["second_implementation"]["pairs"] + first["raised_pairs"] == ray["pairs"]
                       and first["raised_pairs_output"] == first["raised_pairs"],
                       "the pairs the second implementation did not flag returned no forecast and were reported "
                       "by the output checks")
    require_conclusion(2 * ray["checks"]["scalar_shift"]["any"] == ray["mutants"]
                       and ray["checks"]["target_shift"]["any"] == ray["checks"]["scalar_shift"]["any"],
                       "each shift check caught half of the generated mutants")
    require_conclusion(ray["checks"]["scalar_shift"]["any"] / ray["mutants"]
                       > catalogue["checks"]["target_shift"]["any"] / catalogue["mutants"],
                       "the shift checks caught a larger share of the generated mutants than of the catalogue")
    require_conclusion(all(ray["checks"][alarm]["pairs"] / ray["pairs"]
                           > catalogue["checks"][alarm]["pairs"] / catalogue["pairs"]
                           for alarm in ("loss", "answer_key")),
                       "the two alarms caught the generated mutants in a larger share of the conditions than the "
                       "catalogue mutants")
    require_conclusion(ray["mutants"] - ray["checks"]["paired"]["any"] == len(first["unpaired"]) == 1,
                       "the paired check missed only the generated mutant that has no evaluable case")
    groups = first["by_function"]
    require_conclusion(groups["loop"]["scalar_shift"] == groups["loop"]["target_shift"] == groups["loop"]["mutants"]
                       and groups["lags"]["scalar_shift"] == groups["lags"]["target_shift"] == 0
                       and groups["scaling"]["scalar_shift"] == groups["scaling"]["target_shift"],
                       "the shift checks caught every mutant of the forecasting loop and none of the lag "
                       "construction or the scaler fit")
    shifts = {name: (row["scalar_shift"], row["target_shift"]) for name, row in first["catalogue"].items() if row}
    both = {name for name, (scalar, _) in shifts.items() if scalar}
    target_only = {name for name, (scalar, target) in shifts.items() if target and not scalar}
    require_conclusion(both == {"inverse_transform_skipped", "inverse_transform_applied_twice"}
                       and target_only == {"target_columns_swapped", "series_columns_swapped_prediction"},
                       "the common shift check caught only the two inverse-transform mutants of the catalogue, and "
                       "the per-series check those and the two column swaps")
    for name, clean in [("ridge", first["clean"])] + [(name, second["learners"][name]["clean"]) for name in LEARNERS]:
        require_conclusion(clean["deterministic_alarms"] == 0 and clean["deterministic_cases"] == [332],
                           f"no deterministic check alarmed on the clean {name} pipeline, on 332 cases a condition")
    require_conclusion(first["clean"]["loss"]["ci95"][0] <= 0.05 <= first["clean"]["loss"]["ci95"][1],
                       "the interval of the clean loss-alarm rate holds the nominal level")

    classes = second["classification"]
    expected = {"ridge": "exact", "hgb": "exact", "mlp_window_standardized": "in distribution",
                "mlp_raw": "not applicable"}
    require_conclusion({name: row["class"] for name, row in classes.items()} == expected,
                       "every learner has the shift class the protocol expected")
    learners = second["learners"]
    require_conclusion(len({tuple(row["selected"]) for row in learners.values()}) == 1
                       and len({(row["screened"], row["not_runnable"], row["equivalent"],
                                 row["equivalent_in_scaling"]) for row in learners.values()}) == 1,
                       "the three learners screened and selected the same generated mutants")
    require_conclusion(all(row["caught"]["cosmic_ray"]["mutants"] == 12 and row["screen_misses"] == 0
                           and row["stopped"] == 0 for row in learners.values()),
                       "each learner evaluated 12 generated mutants, without a time-out, a crash or a screen miss")
    require_conclusion(all(row["caught"]["catalogue"]["equivalent"] == 4 and row["caught"]["study2"]["equivalent"] == 1
                           for row in learners.values()),
                       "four catalogue mutants and one Study 2 bug are equivalent for each learner without scaling")
    shift = second["calibrated_shift"]
    window = learners["mlp_window_standardized"]["caught"]
    require_conclusion(all(window[kind]["checks"]["calibrated_shift"]["any"] == 0 for kind in KINDS),
                       "the calibrated shift check caught no mutant")
    require_conclusion(shift["clean"]["ci95"][0] <= 0.05 <= shift["clean"]["ci95"][1] and shift["clean_errors"] == 0,
                       "the calibrated shift check kept its level on the clean pipeline")
    require_conclusion(shift["pairs_unadjusted"] <= 0.05 * shift["pairs"],
                       "unadjusted rejections of the calibrated shift check are no more frequent than its level")
    require_conclusion(shift["mutants"]["ci95"][0] <= 0.05 <= shift["mutants"]["ci95"][1],
                       "on the mutants' cases the calibrated shift check failed at its clean rate")
    require_conclusion(learners["hgb"]["shift_caught"] and not learners["mlp_window_standardized"]["shift_caught"]
                       and set(learners["hgb"]["shift_caught"]) <= set(learners["mlp_window_standardized"]["selected"]),
                       "the calibrated shift check caught none of the generated mutants that the exact shift "
                       "checks caught with gradient boosting, on the same mutants")
    others = [learners[name]["caught"] for name in ("hgb", "mlp_raw")]
    require_conclusion(all(window[kind]["checks"]["paired"]["any"] < other[kind]["checks"]["paired"]["any"]
                           for other in others for kind in ("cosmic_ray", "catalogue"))
                       and len({(other["cosmic_ray"]["checks"]["paired"]["any"],
                                 other["catalogue"]["checks"]["paired"]["any"]) for other in others}) == 1,
                       "the paired check caught fewer generated and catalogue mutants with the window-standardized "
                       "perceptron than with either other learner, which agree")
    warned = second["warnings"]
    require_conclusion(warned["categories"] == ["ConvergenceWarning"] and warned["window_clean_always"]
                       and warned["window_cases"] == warned["window_cases_warned"]
                       and warned["window_warnings"] == 41 * warned["window_cases"] and warned["others_warned"] == 0,
                       "every fit of the clean window-standardized perceptron stopped at its iteration limit, and "
                       "gradient boosting and Ridge raised no warning")
    boosting = learners["hgb"]["caught"]
    require_conclusion(2 * boosting["cosmic_ray"]["checks"]["scalar_shift"]["any"] == boosting["cosmic_ray"]["mutants"]
                       and boosting["catalogue"]["checks"]["scalar_shift"]["any"] == 0,
                       "on gradient boosting the common shift check caught half of the generated mutants and no "
                       "catalogue mutant")

    third = summary["part_3"]
    clean, shuffle, spread = (third["variants"][name] for name in ("clean", "partial_shuffle", "reduced_spread"))
    flagged = {name: {method: row["suites_flagged"][method]["count"] for method in ("raw", "holm", "fisher")}
               for name, row in third["variants"].items()}
    require_conclusion(flagged["clean"]["holm"] <= flagged["clean"]["fisher"] < flagged["clean"]["raw"],
                       "on clean suites Fisher's combination flags more suites than Holm and fewer than no correction")
    low, high = clean["suites_flagged"]["fisher"]["ci95"]
    require_conclusion(low <= third["alpha"] <= high,
                       "the interval of Fisher's clean false-alarm rate holds the nominal level")
    require_conclusion(clean["suites_flagged"]["raw"]["ci95"][0] > third["alpha"],
                       "without a correction the suite-level false-alarm rate is above the nominal level")
    require_conclusion(set(flagged["partial_shuffle"].values()) == {third["suites"]},
                       "every method flags every suite under the partial shuffle")
    require_conclusion(flagged["reduced_spread"]["fisher"] == flagged["reduced_spread"]["raw"] == third["suites"]
                       and flagged["reduced_spread"]["holm"] < third["suites"],
                       "under the reduced spread Fisher's combination flags every suite and Holm does not")
    require_conclusion(spread["suites_flagged"]["holm"]["ci95"][1] < spread["suites_flagged"]["fisher"]["ci95"][0],
                       "the intervals of Holm and Fisher under the reduced spread do not overlap")

    fourth = summary["part_4"]
    fail, within = fourth["verdicts"]["FAIL"], fourth["verdicts"]["WITHIN_TOLERANCE"]
    require_conclusion(len(fourth["on_margin"]) == 32 and fourth["repetitions"] == [381] and fourth["cells"] == 240,
                       "32 of the 240 cells have the distance on the margin, each with 381 repetitions")
    require_conclusion(fail["above"] <= 1 and within["above"] >= len(fourth["on_margin"]) // 2,
                       "FAIL keeps its level on the margin in all cells but at most one; WITHIN_TOLERANCE exceeds "
                       "it in at least half")
    require_conclusion(max(abs(value) for value in fourth["bias_z"]) < 3,
                       "the mean of the estimator is within three standard errors of the squared distance in "
                       "every cell")
    reading = fourth["reading"]
    require_conclusion(min(reading["inflated"]) > 0.8 and max(reading["shrunk"]) < min(reading["inflated"])
                       and 0.95 < reading["ratio_median"] < 1.15,
                       "with an inflated spread the jackknife standard error moves with the estimate, while its "
                       "typical size is close to the standard deviation of the estimator")

    fifth = summary["part_5"]
    medians = fifth["medians"]
    require_conclusion(0.5 <= fifth["stationary_low"] and fifth["stationary_high"] < 1.0 and fifth["undefined"] == 0,
                       "on the eight stationary rows the median ratio of the spectral bound to the exact floor is "
                       "below 1 and within the expected range")
    require_conclusion(all(medians[pattern, sd] > 1.5 for pattern in ("trend", "trend_seasonal")
                           for sd in fifth["noise"]),
                       "on the four rows with a trend the spectral bound is far above the exact floor")
    require_conclusion(all(0.5 < medians[pattern, sd] < 1.5 for pattern in ("seasonal", "level_shift")
                           for sd in fifth["noise"]),
                       "on the seasonal and level-shift rows the spectral bound is close to the exact floor, where "
                       "the protocol expected it to be off")
    rows = fifth["rows"]
    top = sorted(rows, key=lambda key: -rows[key]["median_wpe"])[:4]
    require_conclusion({pattern for pattern, _ in top} == {"noise", "level"},
                       "weighted permutation entropy is highest on the pure-noise and fixed-level rows")
    correlations = fifth["correlations"]
    require_conclusion(correlations["wpe_vs_scale_free_floor"]["rho"] > correlations["wpe_vs_floor"]["rho"] > 0
                       and correlations["wpe_vs_scale_free_floor"]["p_value"] < 0.05,
                       "the entropy follows the scale-free floor more closely than the floor itself")

    zoo = summary["zoo"]
    require_conclusion(zoo["failed"] == 0 and len(zoo["models"]) == 9 and len(zoo["marks"]) == 20,
                       "nine models are scored on the 20 cells and no check failed for any of them")
    groups = zoo["pretrained"]["groups"]
    require_conclusion(zoo["pretrained"]["lowest"][0] in PRETRAINING_OVERLAP["resembling"]
                       and min(groups["resembling"]) < min(groups["not_contained"])
                       and max(groups["resembling"]) > max(groups["not_contained"]),
                       "the pretrained model's lowest score is on a kind of series declared as resembling its "
                       "synthetic pretraining data, and its scores on those kinds span its scores on the kinds "
                       "that data do not contain")
    require_conclusion(len(summary["records"]["pythons"]) == 1
                       and all(len(found) == 1 for found in summary["records"]["versions"].values()),
                       "the five parts ran with one Python version and one version of each library")


def _percent(value, digits=1):
    return f"{100 * value:.{digits}f}"


def study3_macros(summary):
    """Macro name to value for every number the hand-written prose takes from the Study 3 records."""
    check_study3_conclusions(summary)
    first, second, third = summary["part_1"], summary["part_2"], summary["part_3"]
    fourth, fifth, zoo = summary["part_4"], summary["part_5"], summary["zoo"]
    ray, catalogue = first["caught"]["cosmic_ray"], first["caught"]["catalogue"]
    shift, learners = second["calibrated_shift"], second["learners"]
    window, boosting = learners["mlp_window_standardized"]["caught"], learners["hgb"]["caught"]
    variants = third["variants"]
    holm_by_condition = [third["conditions"]["reduced_spread", name]["holm"]["count"]
                         for name in third["condition_ids"]]
    within = fourth["verdicts"]["WITHIN_TOLERANCE"]

    return {
        "ThirdSubjectLines": first["subject_lines"], "ThirdMutantSources": summary["records"]["mutant_sources"],
        "ThirdPython": summary["records"]["pythons"][0],
        "ThirdGenerated": first["generated"], "ThirdCrashing": first["not_runnable"],
        "ThirdEquivalent": ray["equivalent"], "ThirdSilent": ray["mutants"],
        "ThirdCatalogueSilent": catalogue["mutants"],
        "ThirdOutputCaught": ray["checks"]["output"]["any"],
        "ThirdShiftCaught": ray["checks"]["scalar_shift"]["any"],
        "ThirdSecondCaught": ray["checks"]["second_implementation"]["any"],
        "ThirdLossCaught": ray["checks"]["loss"]["any"], "ThirdKeyCaught": ray["checks"]["answer_key"]["any"],
        "ThirdPairedCaught": ray["checks"]["paired"]["any"],
        "ThirdCatalogueShiftCaught": catalogue["checks"]["scalar_shift"]["any"],
        "ThirdCatalogueSeriesShiftCaught": catalogue["checks"]["target_shift"]["any"],
        "ThirdLoopMutants": first["by_function"]["loop"]["mutants"],
        "ThirdScalingMutants": first["by_function"]["scaling"]["mutants"],
        "ThirdScalingShiftCaught": first["by_function"]["scaling"]["scalar_shift"],
        "ThirdLagMutants": first["by_function"]["lags"]["mutants"],
        "ThirdLearnerMutants": boosting["cosmic_ray"]["mutants"],
        "ThirdBoostingShiftCaught": len(learners["hgb"]["shift_caught"]),
        "ThirdCalibratedCleanPercent": _percent(shift["clean"]["rate"]),
        "ThirdCalibratedMutantPercent": _percent(shift["mutants"]["rate"]),
        "ThirdWindowPairedCaught": window["cosmic_ray"]["checks"]["paired"]["any"],
        "ThirdWindowCataloguePairedCaught": window["catalogue"]["checks"]["paired"]["any"],
        "ThirdOtherPairedCaught": boosting["cosmic_ray"]["checks"]["paired"]["any"],
        "ThirdOtherCataloguePairedCaught": boosting["catalogue"]["checks"]["paired"]["any"],
        "ThirdSuites": third["suites"],
        "ThirdCleanRawSuites": variants["clean"]["suites_flagged"]["raw"]["count"],
        "ThirdCleanFisherSuites": variants["clean"]["suites_flagged"]["fisher"]["count"],
        "ThirdSpreadFisherSuites": variants["reduced_spread"]["suites_flagged"]["fisher"]["count"],
        "ThirdSpreadHolmSuites": variants["reduced_spread"]["suites_flagged"]["holm"]["count"],
        "ThirdSpreadHolmLowest": min(holm_by_condition), "ThirdSpreadHolmHighest": max(holm_by_condition),
        "ThirdMarginCells": len(fourth["on_margin"]), "ThirdWithinAbove": within["above"],
        "ThirdWithinHighest": within["high"], "ThirdRepetitions": fourth["repetitions"][0],
        "ThirdFailAbove": fourth["verdicts"]["FAIL"]["above"],
        "ThirdSpectralLowest": f"{fifth['stationary_low']:.2f}",
        "ThirdSpectralHighest": f"{fifth['stationary_high']:.2f}",
        "ThirdSpectralTrend": f"{fifth['medians']['trend', 0.5]:.0f}",
        "ThirdEntropyRho": f"{fifth['correlations']['wpe_vs_scale_free_floor']['rho']:.2f}",
        "ThirdModels": len(zoo["models"]),
    }


def _cell(stat):
    """A rate as count/n with its Wilson 95% interval in percent."""
    low, high = stat["ci95"]
    return f"{stat['count']}/{stat['n']} [{100 * low:.1f}, {100 * high:.1f}]"


def _table(caption, label, columns, head, body, size="small", separation=None):
    narrow = rf"\setlength{{\tabcolsep}}{{{separation}}}" if separation else ""
    lines = [rf"\begin{{table}}[htbp]\centering\{size}{narrow}", "", rf"\caption{{{caption}}}", "",
             rf"\label{{{label}}}", "", rf"\begin{{tabular}}{{{columns}}}\toprule", ""]
    lines += [line for row in head for line in (row, "")]
    lines += [line for row in body for line in (row, "")]
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    return "\n".join(lines)


def mutants_section(first):
    ray, catalogue, bugs = (first["caught"][kind] for kind in KINDS)
    groups, clean = first["by_function"], first["clean"]

    def entry(kind, check):
        row = first["caught"][kind]["checks"][check]
        return f"{row['any']} ({row['pairs']})"

    body = [f"{CHECK_LABELS[check]} & " + " & ".join(entry(kind, check) for kind in KINDS) + r" \\"
            for check in PART_1_CHECKS]
    text = [
        r"\subsection{Study 3: generated mutants}",
        r"\label{sec:results-mutants}",
        "",
        f"Cosmic Ray generated {first['generated']} mutants of the {first['subject_lines']}-line subject. "
        f"{first['not_runnable']} raise an exception on every input, so no check is needed to see them. Of the "
        f"{ray['runnable']} that run, {ray['equivalent']} return the clean forecast on all 24 screening inputs and "
        f"are counted as equivalent; {ray['mutants']} change a forecast. Of the {catalogue['generated']} catalogue "
        "mutants one is equivalent, a target scaler reused across contexts (the Ridge fit is linear in its "
        "targets, so the inverse transform undoes it), and none of the four Study 2 bugs is. The "
        "equivalent sources were also run on every evaluation case, and none differed from the clean pipeline "
        "there; 6 and 48 screening inputs give the same equivalent mutants as 24.",
        "",
        r"Table~\ref{tab:mutants} counts, for each check, the mutants caught in at least one of the six "
        "conditions and, in parentheses, the pairs of a mutant and a condition in which it was caught.",
        "",
        _table("Study 3, Ridge pipeline: mutants that change a forecast and are caught by each check in at least "
               "one of six conditions; in parentheses, the mutant--condition pairs caught. A deterministic check "
               "catches a mutant in a condition when it alarms on any of the 93 evaluation cases, a calibrated "
               r"check when its Holm-adjusted test rejects (Section~\ref{sec:third-study}). The counts describe "
               "these sets of mutants, not a population of real faults.",
               "tab:mutants", "@{}lrrr@{}",
               [r"Check & Cosmic Ray & Catalogue & Study 2 bugs \\",
                f" & ({ray['mutants']} mutants, {ray['pairs']} pairs) & ({catalogue['mutants']}, "
                f"{catalogue['pairs']}) & ({bugs['mutants']}, {bugs['pairs']}) \\\\ \\midrule"], body),
        "",
        "The second implementation flagged every mutant in every condition in which the mutant returned a "
        f"forecast. The {first['raised_pairs']} pairs it did not flag belong to {first['raised_mutants']} "
        "generated mutants that raise an exception on every case of four conditions, where the output checks "
        "report them. The paired check missed one generated mutant, which returns no valid forecast on any case "
        "and so leaves nothing to pair; the output checks caught it. The answer-key alarm caught at least as "
        f"many mutants as the loss alarm in every set: {ray['checks']['answer_key']['any']} against "
        f"{ray['checks']['loss']['any']} generated mutants ({ray['checks']['answer_key']['pairs']} against "
        f"{ray['checks']['loss']['pairs']} pairs), {catalogue['checks']['answer_key']['any']} against "
        f"{catalogue['checks']['loss']['any']} catalogue mutants ({catalogue['checks']['answer_key']['pairs']} "
        f"against {catalogue['checks']['loss']['pairs']}), and the Study 2 bugs in "
        f"{bugs['checks']['answer_key']['pairs']} against {bugs['checks']['loss']['pairs']} pairs.",
        "",
        f"Each shift check caught {ray['checks']['scalar_shift']['any']} of the {ray['mutants']} generated "
        "mutants. Grouped by the function they change, a breakdown made after the results were known, the shift "
        f"checks caught all {groups['loop']['mutants']} mutants of the forecasting loop, "
        f"{groups['scaling']['scalar_shift']} of the {groups['scaling']['mutants']} mutants of the two functions "
        f"that apply and invert the scaling, and none of the {groups['lags']['mutants']} mutants of the lag "
        "construction and of the scaler fit. The catalogue agrees: the common shift check caught only its two "
        "inverse-transform mutants, the per-series check those and the two column swaps, and the other "
        f"{catalogue['mutants'] - catalogue['checks']['target_shift']['any']} (lag window, lag order, recursion "
        "state, horizon index, seasonal period, dropped training rows and a reused feature scaler) passed both.",
        "",
        "No deterministic check alarmed on the clean pipeline, on "
        f"{clean['deterministic_cases'][0]} cases in each condition (239 calibration and 93 evaluation cases). "
        f"At the 5\\% level the loss alarm fired on {_cell(clean['loss'])}\\% of the clean evaluation cases and "
        f"the answer-key alarm on {_cell(clean['answer_key'])}\\%.",
    ]
    return "\n".join(text)


def learners_section(second):
    classes, learners, shift, warned = (second[name] for name in ("classification", "learners", "calibrated_shift",
                                                                   "warnings"))

    def class_row(name):
        row = classes[name]
        calibrated = "---" if row["calibrated_cases"] is None else f"{row['calibrated_fails']}/{row['calibrated_cases']}"
        return (f"{LEARNER_LABELS[name]} & {row['public_pass']}/{row['public_cases']} & {calibrated} & "
                f"{CLASS_LABELS[row['class']]} \\\\")

    def entry(name, check):
        caught = learners[name]["caught"]
        if check not in caught["cosmic_ray"]["checks"]:
            return "---"
        return " / ".join(str(caught[kind]["checks"][check]["any"]) for kind in KINDS)

    first = learners[LEARNERS[0]]
    sizes = " / ".join(str(first["caught"][kind]["mutants"]) for kind in KINDS)
    body = [f"{CHECK_LABELS[check]} & " + " & ".join(entry(name, check) for name in LEARNERS) + r" \\"
            for check in PART_2_CHECKS]
    window, boosting, raw = (learners[name] for name in ("mlp_window_standardized", "hgb", "mlp_raw"))

    def paired(learner, kind):
        return learner["caught"][kind]["checks"]["paired"]["any"]

    clean_cases = boosting["clean"]["deterministic_cases"][0] * boosting["clean"]["conditions"]
    text = [
        r"\subsection{Study 3: learners beyond Ridge}",
        r"\label{sec:results-learners}",
        "",
        r"Table~\ref{tab:shift-classes} classifies each learner on its clean pipeline, on 16 cases in each of the "
        "six conditions. Every learner received the class that the protocol expected.",
        "",
        _table("Study 3: how each learner's clean pipeline responds to a shift of 10 in its history. The public "
               "shift check asks for the same forecast plus 10; the calibrated check asks only that the shifted "
               "forecast look like one more refit with a new training seed. A learner is exact when the public "
               "check passes on every case, in distribution when the calibrated check fails no more often than "
               "its 5\\% level allows, and not applicable otherwise.",
               "tab:shift-classes", "@{}lrrl@{}",
               [r"Learner & Public: PASS & Calibrated: FAIL & Class \\ \midrule"],
               [class_row(name) for name in ("ridge", *LEARNERS)]),
        "",
        "The three learners run the subject without its scaling step, so a mutant of a scaling function changes "
        f"nothing. Of the first {first['screened']} Cosmic Ray mutants in the seeded order, "
        f"{first['not_runnable']} raise an exception on every input and {first['equivalent']} are equivalent, "
        f"{first['equivalent_in_scaling']} of them in the three scaling functions; the "
        f"{first['caught']['cosmic_ray']['mutants']} that remain are the same for the three learners. Of the "
        f"{first['caught']['catalogue']['generated']} catalogue mutants "
        f"{first['caught']['catalogue']['equivalent']} are equivalent, and one of the four Study 2 bugs, as the "
        r"protocol expected. Table~\ref{tab:learners} counts the mutants caught.",
        "",
        _table("Study 3: mutants caught in at least one condition with each learner, as Cosmic Ray / catalogue / "
               f"Study 2 bugs (of {sizes}). A dash marks a check that does not apply to the learner's shift class "
               r"(Table~\ref{tab:shift-classes}); no second implementation exists for these learners.",
               "tab:learners", "@{}lccc@{}",
               ["Check & " + " & ".join(LEARNER_HEADS[name] for name in LEARNERS) + r" \\ \midrule"], body),
        "",
        "The calibrated shift check kept its level and caught no mutant. On the clean window-standardized "
        f"pipeline it returned FAIL on {_cell(shift['clean'])}\\% of the evaluation cases. On the "
        f"{shift['evaluated_sources']} evaluated mutants it returned FAIL on {shift['mutants']['count']:,} of "
        f"{shift['mutants']['n']:,} cases ({_percent(shift['mutants']['rate'])}\\%); "
        f"{shift['pairs_unadjusted']} of the {shift['pairs']} pairs of a mutant and a condition have an unadjusted "
        f"$p$-value of at most 0.05, no more than its level gives by chance, and none is left after Holm's "
        "correction. With this learner the paired check also caught fewer mutants than with the other two: "
        f"{paired(window, 'cosmic_ray')} generated and {paired(window, 'catalogue')} catalogue mutants, against "
        f"{paired(boosting, 'cosmic_ray')} and {paired(boosting, 'catalogue')} with each of the others. Every fit "
        "of this learner stopped at its limit of 500 iterations "
        f"({warned['window_warnings']:,} convergence warnings on the {warned['window_cases']} clean evaluation "
        "cases, 41 fits a case); the perceptron on raw lags did so on "
        f"{warned['raw_cases_warned']} of its {warned['raw_cases']:,} clean calibration and evaluation cases, and "
        "gradient boosting raised no warning. With gradient boosting the exact shift checks behaved as with "
        f"Ridge: no alarm on its {clean_cases:,} clean cases, "
        f"{boosting['caught']['cosmic_ray']['checks']['scalar_shift']['any']} of the "
        f"{boosting['caught']['cosmic_ray']['mutants']} generated mutants caught, and at most "
        f"{boosting['caught']['catalogue']['checks']['target_shift']['any']} catalogue mutants.",
        "",
        "No deterministic check alarmed on a clean pipeline "
        f"({boosting['clean']['deterministic_cases'][0]} cases for each learner and condition). At the 5\\% level "
        "the loss alarm fired on "
        + ", ".join(str(learners[name]["clean"]["loss"]["count"]) for name in LEARNERS[:-1])
        + f" and {raw['clean']['loss']['count']} of the {raw['clean']['loss']['n']} clean evaluation cases of "
        "gradient boosting, the window-standardized perceptron and the raw-lag perceptron, and the answer-key "
        "alarm on "
        + ", ".join(str(learners[name]["clean"]["answer_key"]["count"]) for name in LEARNERS[:-1])
        + f" and {raw['clean']['answer_key']['count']}.",
    ]
    return "\n".join(text)


def suites_section(third):
    variants = third["variants"]

    def flagged(name, method):
        return variants[name]["suites_flagged"][method]

    clean_fisher = flagged("clean", "fisher")
    null_condition = third["condition_ids"][0]
    shuffled = third["conditions"]["partial_shuffle", null_condition]
    holm = [third["conditions"]["reduced_spread", name]["holm"]["count"] for name in third["condition_ids"]]
    body = [f"{VARIANT_LABELS[name]} & " + " & ".join(_cell(flagged(name, method))
                                                      for method in ("raw", "holm", "fisher")) + r" \\"
            for name in third["variants"]]
    text = [
        r"\subsection{Study 3: two questions about a suite}",
        r"\label{sec:results-suites}",
        "",
        r"Table~\ref{tab:suite-questions} repeats the covariance suites of Study 2 on "
        f"{third['suites']} new suites with 239 calibration draws. Of the clean suites, "
        f"{flagged('clean', 'raw')['count']} contain at least one unadjusted rejection, "
        f"{flagged('clean', 'holm')['count']} a Holm rejection, and Fisher's combination flags "
        f"{clean_fisher['count']}, a rate whose interval holds the nominal 5\\%. Under the partial shuffle all "
        "three flag every suite; on the condition with independent targets, where the shuffle changes nothing, "
        f"the unadjusted check rejected in {shuffled['raw']['count']} suites and Holm in "
        f"{shuffled['holm']['count']}. Under the reduced spread Fisher's combination flags all "
        f"{flagged('reduced_spread', 'fisher')['count']} suites, while Holm rejects at least one condition in "
        f"{flagged('reduced_spread', 'holm')['count']} and a given condition in {min(holm)} to {max(holm)}.",
        "",
        _table(f"Study 3: suites of four covariance checks flagged, of {third['suites']}, with Wilson 95\\% "
               "intervals (percent). Raw counts a suite with any unadjusted rejection, Holm one with any rejection "
               "after Holm's adjustment, and Fisher one whose combined test rejects. Every flag on the control is "
               "a false alarm; the other two rows measure detection.",
               "tab:suite-questions", "@{}lrrr@{}", [r"Variant & Raw & Holm & Fisher \\ \midrule"], body),
    ]
    return "\n".join(text)


def margin_section(fourth):
    fail, within, reading = fourth["verdicts"]["FAIL"], fourth["verdicts"]["WITHIN_TOLERANCE"], fourth["reading"]
    cells = {(cell["condition"], cell["direction"], cell["margin"], cell["n_paths"]): cell
             for cell in fourth["on_margin"]}
    margins = sorted({key[2] for key in cells})
    paths = sorted({key[3] for key in cells})
    conditions = list(dict.fromkeys(cell["condition"] for cell in fourth["on_margin"]))

    def pair(condition, direction, margin, n_paths):
        cell = cells[condition, direction, margin, n_paths]["verdicts"]
        return f"{cell['FAIL']['count']} / {cell['WITHIN_TOLERANCE']['count']}"

    body = []
    for condition in conditions:
        for direction, label in (("shrink", "narrower"), ("inflate", "wider")):
            name = SUITE_CONDITIONS[condition] if direction == "shrink" else ""
            body.append(f"{name} & {label} & " + " & ".join(pair(condition, direction, margin, n_paths)
                                                           for margin in margins for n_paths in paths) + r" \\")
        body.append(r"\addlinespace")
    zero = fourth["at_zero"]

    def at_zero(margin, n_paths):
        return f"{zero[margin, n_paths]['count']:,}"

    repetitions = fourth["repetitions"][0]
    head = [" & & " + " & ".join(rf"\multicolumn{{{len(paths)}}}{{c}}{{Margin {margin:g}}}" for margin in margins)
            + r" \\",
            "Condition & Spread & " + " & ".join(f"{n_paths} paths" for _ in margins for n_paths in paths)
            + r" \\ \midrule"]
    outside = sum(abs(value) > 2 for value in fourth["bias_z"])
    text = [
        r"\subsection{Study 3: a joint-spread decision with a margin}",
        r"\label{sec:results-margin}",
        "",
        r"Table~\ref{tab:margin} gives how often each one-sided verdict was returned when the true distance lay "
        "exactly on the margin, where each was designed to occur in at most about 5\\% of repetitions. FAIL meets "
        f"the design: between {fail['low']} and {fail['high']} of {repetitions} repetitions over the "
        f"{len(fourth['on_margin'])} cells, with a 95\\% interval above 5\\% in {fail['above']} of them. "
        f"WITHIN\\_TOLERANCE does not: between {within['low']} and {within['high']} of {repetitions}, with the "
        f"interval above 5\\% in {within['above']} of the {len(fourth['on_margin'])} cells, "
        f"{within['above_inflated']} of them with a spread that is too wide and {within['above_dependent']} in "
        "the two conditions with dependence over time. With the distance "
        f"{100 * (fourth['within_beyond_multiple'] - 1):.0f}\\% beyond the margin it was still returned in up to "
        f"{fourth['within_beyond']} of {repetitions} repetitions. The expectation that the protocol wrote down "
        "for this verdict was therefore wrong.",
        "",
        _table(f"Study 3: verdicts of the margin decision when the true distance equals the margin, as FAIL / "
               f"WITHIN\\_TOLERANCE counts of {repetitions} repetitions; the remainder is INCONCLUSIVE. Either "
               f"verdict is an error here, designed to occur in at most about 5\\% of repetitions "
               f"({0.05 * repetitions:.0f} of {repetitions}).",
               "tab:margin", "@{}ll" + "r" * (len(margins) * len(paths)) + "@{}", head,
               body[:-1], size="footnotesize"),
        "",
        "The estimator itself shows no bias: over the "
        f"{fourth['cells']} cells its mean differs from the squared distance by at most "
        f"{max(abs(value) for value in fourth['bias_z']):.1f} of its standard errors, and by more than two in "
        f"{outside}. A reading made after the run locates the problem in the interval. The jackknife standard "
        "error has the right size on average (its mean is "
        f"{reading['ratio_median']:.2f} times the standard deviation of the estimator in the median cell), but "
        "with a spread that is too wide it moves with the estimate (correlation between "
        f"{min(reading['inflated']):.2f} and {max(reading['inflated']):.2f} in the {len(reading['inflated'])} such "
        "cells of "
        r"Table~\ref{tab:margin}), so a low estimate arrives with a short interval and falls below the margin. "
        f"At a distance of zero, where it is the correct verdict, WITHIN\\_TOLERANCE was returned in "
        f"{at_zero(margins[0], paths[0])} of "
        f"{zero[margins[0], paths[0]]['n']:,} repetitions at {paths[0]} paths and {at_zero(margins[0], paths[1])} "
        f"at {paths[1]} with the margin {margins[0]:g}, and in {at_zero(margins[1], paths[0])} and "
        f"{at_zero(margins[1], paths[1])} with the margin {margins[1]:g}.",
    ]
    return "\n".join(text)


def predictability_section(fifth):
    rows, low, high = fifth["rows"], *fifth["noise"]

    def ratio(pattern, noise):
        found = rows[pattern, noise]["primary_ratio"]
        return f"{found['median']:.2f} [{found['p05']:.2f}, {found['p95']:.2f}]"

    def entropy(pattern, noise):
        return f"{rows[pattern, noise]['median_wpe']:.2f}"

    def median(pattern, noise):
        return f"{fifth['medians'][pattern, noise]:.2f}"

    body = [f"{PATTERN_LABELS[pattern]} & {ratio(pattern, low)} & {ratio(pattern, high)} & "
            f"{entropy(pattern, low)} & {entropy(pattern, high)} \\\\" for pattern in fifth["order"]]
    history = rows["level_shift", low]["history_ratio"]["median"]
    correlations = fifth["correlations"]
    text = [
        r"\subsection{Study 3: predictability estimates against the exact floor}",
        r"\label{sec:results-predictability}",
        "",
        r"Table~\ref{tab:predictability} divides the spectral bound by the exact unavoidable error on "
        f"{fifth['series'][0]} series of each scorecard cell. On the eight stationary rows (pure noise, a fixed "
        "level, mean reversion and irregular cycles) the median ratio lies between "
        f"{fifth['stationary_low']:.2f} and {fifth['stationary_high']:.2f}, as the protocol expected. On the four "
        f"rows with a trend it is far above 1 ({median('trend', low)} and {median('trend_seasonal', low)} at low "
        f"noise, {median('trend', high)} and {median('trend_seasonal', high)} at high noise), as expected. Two "
        "expectations were wrong: on the seasonal rows "
        f"({median('seasonal', low)} and {median('seasonal', high)}) and on the level-shift rows "
        f"({median('level_shift', low)} and {median('level_shift', high)}) the bound is close to the exact value, "
        "where the protocol expected it to be off. Those ratios use the realized future segment, as the bound is "
        "defined; computed from the history alone, the level-shift ratio is "
        f"{history:.1f} at low noise. The random-walk rows have no stationary spectrum, so the bound is not "
        "defined there in theory; its values are reported as computed. Weighted permutation entropy is highest on "
        "the pure-noise and fixed-level rows. Across the 20 rows its rank correlation with the exact floor is "
        f"{correlations['wpe_vs_floor']['rho']:.2f}, and {correlations['wpe_vs_scale_free_floor']['rho']:.2f} once "
        "the floor is divided by the variance of the history, which removes the scale that the entropy ignores.",
        "",
        _table("Study 3: the spectral bound divided by the exact unavoidable error (median and 5th to 95th "
               f"percentile over {fifth['series'][0]} series; 1 is exact), and the median weighted permutation "
               "entropy (0 to 1), at low and high noise. The two random-walk rows have no stationary spectrum.",
               "tab:predictability", "@{}lrrrr@{}",
               [r" & \multicolumn{2}{c}{Spectral bound / exact floor} & \multicolumn{2}{c}{Entropy} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                r"Kind of series & Low noise & High noise & Low & High \\ \midrule"], body),
    ]
    return "\n".join(text)


def zoo_section(zoo):
    cells, marks, low, high = zoo["cells"], zoo["marks"], *zoo["noise"]

    def score(model, pattern, noise=low):
        return f"{cells[model, (pattern, noise)]:.0f}"

    def mark(pattern, noise=low):
        return f"{marks[pattern, noise]:.0f}"

    body = []
    for pattern in zoo["order"]:
        for noise, label in ((low, "low"), (high, "high")):
            name = PATTERN_LABELS[pattern] if noise == low else ""
            body.append(f"{name} & {label} & {marks[pattern, noise]:.0f} & "
                        + " & ".join(f"{cells[model, (pattern, noise)]:.0f}" for model in zoo["models"]) + r" \\")
    pretrained = zoo["pretrained"]

    def spread(group):
        values = pretrained["groups"][group]
        return f"{min(values):.0f} to {max(values):.0f}"

    lowest_pattern, lowest_noise = pretrained["lowest"]
    cycles = " and ".join(f"{value:.0f}" for value in pretrained["groups"]["partly"])
    text = [
        r"\subsection{Study 3: scorecards of nine models}",
        r"\label{sec:results-zoo}",
        "",
        r"Table~\ref{tab:zoo} gives the scorecard of each model beside the row's attainable mark. No check "
        "failed for any model. The scores show where each model stops; they rank no model, because the rows are "
        "mechanisms chosen for their known answers and not a sample of real series. At low noise, ETS scores "
        f"{score('ets', 'cycles')} on irregular cycles (mark {mark('cycles')}); gradient boosting on lags scores "
        f"{score('hgb_lags', 'trend')} on a straight-line trend (mark {mark('trend')}), which a tree cannot "
        f"extrapolate, and {score('hgb_lags', 'level_shift')} on a sudden level shift (mark "
        f"{mark('level_shift')}), where Theta scores {score('theta', 'level_shift')}; the perceptron on lags "
        f"scores {score('mlp_lags', 'random_walk')} on a random walk (mark {mark('random_walk')}).",
        "",
        "Chronos is the one pretrained model. Before its run the protocol declared how each kind of series "
        "relates to the synthetic part of its pretraining data, which is drawn from Gaussian processes with "
        r"constant, white-noise, linear, RBF, rational-quadratic and periodic kernels~\cite{ansari2024chronos}: "
        "five kinds resemble it (pure noise, a fixed "
        "level, a straight-line trend, a seasonal cycle and a trend with a seasonal cycle), irregular cycles "
        "resemble it in part, and four are not in its kernel bank (a level shift, mean reversion and the two "
        f"random walks). Chronos scores {spread('resembling')} on the five kinds that resemble that data, "
        f"{cycles} on irregular cycles, and {spread('not_contained')} on the four that it does not contain. Its "
        f"lowest score, {score(PRETRAINED, lowest_pattern, lowest_noise)}, is on the "
        f"{'low' if lowest_noise == low else 'high'}-noise {PATTERN_LABELS[lowest_pattern].lower()} (mark "
        f"{mark(lowest_pattern, lowest_noise)}), and it scores {score(PRETRAINED, 'mean_reverting', low)} and "
        f"{score(PRETRAINED, 'mean_reverting', high)} on mean reversion (marks {mark('mean_reverting', low)} and "
        f"{mark('mean_reverting', high)}). The declared resemblance does not order its scores; what the model "
        "saw in its real pretraining data cannot be stated.",
        "",
        _table("Study 3: scores out of 100 of nine models on the default scorecard grid (30 series a cell) at low "
               "and high noise, beside the attainable mark. Last is the last value, Seas.\\ the seasonal naive "
               "forecast (period 12), Boost.\\ and Perc.\\ gradient boosting and a perceptron on lags, Chronos "
               "the pretrained small model, Parrot context parroting. The interval of every score is in the "
               "study's records. No ranking is claimed.",
               "tab:zoo", "@{}p{0.24\\linewidth}l" + "r" * (1 + len(zoo["models"])) + "@{}",
               ["Kind of series & Noise & Mark & " + " & ".join(MODEL_HEADS[model] for model in zoo["models"])
                + r" \\ \midrule"], body, size="footnotesize", separation="3.4pt"),
    ]
    return "\n".join(text)


def render_tex(summary):
    """The Study 3 part of the Results section."""
    check_study3_conclusions(summary)
    sections = [mutants_section(summary["part_1"]), learners_section(summary["part_2"]),
                suites_section(summary["part_3"]), margin_section(summary["part_4"]),
                predictability_section(summary["part_5"]), zoo_section(summary["zoo"])]
    return ("% Generated by experiments/render_study3.py from experiments/results-v3; do not edit by hand.\n\n"
            + "\n\n".join(sections) + "\n")


def macros_text(summary):
    """The macro definitions that experiments.render_supplement appends to generated-macros.tex."""
    return ("% Study 3 (protocol 3.0, experiments/results-v3), registered before its runs; do not edit.\n"
            + "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in study3_macros(summary).items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    summary = summarize(load_study3(args.results_dir))
    fragment = args.paper_dir / "generated-study3.tex"
    fragment.write_text(render_tex(summary), encoding="utf-8")
    print(json.dumps({"results": str(fragment), "macros": len(study3_macros(summary))}))


if __name__ == "__main__":
    main()
