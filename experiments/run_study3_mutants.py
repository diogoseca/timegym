"""Run Study 3, Parts 1 and 2 (protocol 3.0): generated mutants of the subject pipeline, and learners beyond Ridge.

    python -m experiments.run_study3_mutants --part 1 --output experiments/results-v3/part-1 --workers 4
    python -m experiments.run_study3_mutants --part 2 --output experiments/results-v3/part-2 --workers 4
    python -m experiments.run_study3_mutants --part 1 --plan

main() calls study3_common.require_start before any computation; no flag bypasses it. --plan prints the units,
cases and fits that would run and an estimate of CPU time, computes nothing and works on a draft protocol.

The runner reads the committed mutant manifest (experiments/study3_generate.py), whose digest the protocol holds,
and refuses a source whose digest differs. Part 1, for Ridge: the clean subject on the calibration and evaluation cases of every condition, the
equivalence screen of every source, then every source that is not equivalent at N = 24 with every check, and the
equivalent ones with forecasts only (screen misses). Part 2: the shift classification of the four learners on the
clean subject, then the same for the three learners with the 4 Study 2 bugs, the 16 catalogue mutants and the
first 12 generic mutants that are runnable and not equivalent for that learner. Every unit runs in its own process
(experiments/study3_harness.py); the runner assembles their records in a fixed order, so results do not depend on
the number of workers. The statistics are those of experiments/study3_stats.py.

Everything is passed to run_part1 and run_part2 as arguments (conditions, seeds, learner module, sources, sizes,
limits); only main() binds them to the protocol.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

from . import study3_common as common
from .common import sha256

KINDS = ("cosmic_ray", "catalogue", "study2")
PRIMARY = "24"
DETERMINISTIC = ("output", "scalar_shift", "target_shift", "second_implementation")
STATISTICAL = ("loss", "answer_key", "paired", "calibrated_shift")
CHECK_NAMES = {"output_checks": "output", "scalar_shift": "scalar_shift", "per_target_shift": "target_shift",
               "calibrated_shift_check": "calibrated_shift", "loss_alarm": "scores", "answer_key_alarm": "scores",
               "paired_check": "scores"}
PART1_CHECKS = ("scores", "scalar_shift", "target_shift", "second_implementation")
# Seconds per fit and forecast measured in the clean timing pilot (pilots_before_registration), and an allowance
# for starting one unit's process; used only by --plan.
PILOT_FIT_SECONDS = {"ridge": 0.003, "hgb": 0.06, "mlp_window_standardized": 0.11, "mlp_raw": 0.06}
UNIT_START_SECONDS = 1.5


# ---------------------------------------------------------------------------------------------------------------
# Units


def _unit_spec(config, learner, source, condition_index, stages, checks, name, phase, mode="cases"):
    condition = config["conditions"][condition_index]
    return {"output": str(Path(config["scratch"]) / learner["name"] / name / f"{phase}-{condition_index}.jsonl"),
            "source": source, "module_name": f"study3_source_{name.replace('-', '_')}",
            "learner": {"module": config["learner_module"], "name": learner["name"], "index": learner["index"],
                        "tag": config["learner_tag"]},
            "condition": condition, "warmup_seed": config["seeds"]["screen"][condition_index][0],
            "stages": stages, "checks": list(checks), "mode": mode,
            "stage_checks": {"calibration": [check for check in checks if check != "calibrated_shift"]},
            "reference": config.get("reference") if "second_implementation" in checks else None,
            "settings": config["settings"]}


def _case_records(result, stage):
    return {record["seed"]: record for record in result["records"]
            if record.get("kind") == "case" and record["stage"] == stage}


def _loaded(result):
    records = result["records"]
    return bool(records) and records[0].get("kind") == "load" and records[0]["ok"]


def _clean_seconds(clean, stages, forecast_only=False):
    """Wall time of the clean unit for the given stages, with its start-up and warm-up."""
    records = clean["records"]
    cases = [record for record in records if record.get("kind") == "case"]
    case_total = sum(record.get("seconds", record.get("forecast_seconds", 0.0)) for record in cases)
    overhead = max(clean["seconds"] - case_total, 0.0)
    same = sum(record.get("forecast_seconds" if forecast_only else "seconds", record.get("forecast_seconds", 0.0))
               for record in cases if record["stage"] in stages)
    return overhead + same


def _limit(config, clean, stages, forecast_only=False):
    rule = config["time_limit"]
    return max(rule["minimum"], rule["factor"] * _clean_seconds(clean, stages, forecast_only))


def _stages(config, condition_index, *names):
    seeds = config["seeds"]
    return [[name, list(seeds[name][condition_index])] for name in names]


def run_clean(config, learner, checks):
    """The clean subject in every condition: equivalence inputs, calibration and evaluation cases."""
    specs = [_unit_spec(config, learner, config["clean_source"], c,
                        _stages(config, c, "screen", "calibration", "evaluation"), checks, "clean", "full")
             for c in range(len(config["conditions"]))]
    from .study3_harness import run_units

    results = run_units(specs, [None] * len(specs), config["workers"])
    for condition, result in zip(config["conditions"], results):
        if not result["finished"]:
            raise RuntimeError(f"The clean subject did not finish in {condition['id']}")
        if any(record.get("raised") for record in result["records"] if record.get("kind") == "case"):
            raise RuntimeError(f"The clean subject raised in {condition['id']}; the study cannot proceed")
    return results


def run_screens(config, learner, sources, clean):
    """The equivalence screen of every source in every condition, in the order of ``sources``."""
    from .study3_harness import run_units

    n = len(config["conditions"])
    specs, limits = [], []
    for source in sources:
        for c in range(n):
            specs.append(_unit_spec(config, learner, source["source"], c, _stages(config, c, "screen"), (),
                                    source["id"], "screen"))
            limits.append(_limit(config, clean[c], ("screen",), forecast_only=True))
    results = run_units(specs, limits, config["workers"])
    return [results[i * n:(i + 1) * n] for i in range(len(sources))]


def run_evaluations(config, learner, sources, classes, clean, checks):
    """Evaluation units: every check for sources not equivalent at N = 24, forecasts only for equivalent ones."""
    from .study3_harness import run_units

    n = len(config["conditions"])
    specs, limits, owners = [], [], []
    for source in sources:
        status = classes[source["id"]]["sizes"][PRIMARY]
        if not status["runnable"]:
            continue
        forecast_only = status["equivalent"]
        for c in range(n):
            specs.append(_unit_spec(config, learner, source["source"], c, _stages(config, c, "screen", "evaluation"),
                                    () if forecast_only else checks, source["id"], "evaluation"))
            limits.append(_limit(config, clean[c], ("screen", "evaluation"), forecast_only))
            owners.append(source["id"])
    results = run_units(specs, limits, config["workers"])
    grouped: dict[str, list] = {}
    for owner, result in zip(owners, results):
        grouped.setdefault(owner, []).append(result)
    return grouped


# ---------------------------------------------------------------------------------------------------------------
# Equivalence


def same_forecast(mutant: dict, clean: dict, atol: float, rtol: float) -> bool:
    """funnel.equivalence.rule for one input: same index, names and shape, finite, and numpy.allclose."""
    import numpy as np

    if mutant.get("raised") or clean.get("raised"):
        return False
    left, right = mutant["forecast"], clean["forecast"]
    if left["index"] != right["index"] or left["target_names"] != right["target_names"]:
        return False
    if left["shape"] != right["shape"] or not left["finite"] or not right["finite"]:
        return False
    return bool(np.allclose(np.asarray(left["point"]), np.asarray(right["point"]), atol=atol, rtol=rtol))


def classify(config, clean, screens):
    """Runnable and equivalent at each screen size, with the reason when not runnable."""
    atol, rtol = config["settings"]["atol"], config["settings"]["rtol"]
    out = {"load": None, "sizes": {}}
    if not all(_loaded(result) for result in screens):
        first = screens[0]["records"][0] if screens[0]["records"] else {"reason": "crashed", "error": None}
        out["load"] = {"reason": first.get("reason"), "error": first.get("error")}
    for size, per_condition in config["screen_sizes"].items():
        forecasts, equal, timed_out, crashed = 0, True, False, False
        for c, result in enumerate(screens):
            seeds = config["seeds"]["screen"][c][:per_condition]
            got, reference = _case_records(result, "screen"), _case_records(clean[c], "screen")
            for seed in seeds:
                record = got.get(seed)
                if record is None:
                    timed_out |= result["timed_out"]
                    crashed |= result.get("crashed", False)
                    equal = False
                    continue
                forecasts += not record["raised"]
                if not reference[seed]["raised"] and not same_forecast(record, reference[seed], atol, rtol):
                    equal = False
        runnable = out["load"] is None and forecasts > 0
        reason = None
        if not runnable:
            reason = (out["load"]["reason"] if out["load"] else "timeout" if timed_out
                      else "crashed" if crashed else "exception on every input")
        out["sizes"][size] = {"runnable": runnable, "equivalent": runnable and equal, "reason": reason,
                              "inputs": per_condition * len(screens), "forecasts": forecasts}
    return out


def screen_misses(config, clean, evaluations):
    """Evaluation cases on which an equivalent source's forecast differs from the clean forecast."""
    atol, rtol = config["settings"]["atol"], config["settings"]["rtol"]
    differ, missing = 0, 0
    for c, result in enumerate(evaluations):
        reference, got = _case_records(clean[c], "evaluation"), _case_records(result, "evaluation")
        for seed in config["seeds"]["evaluation"][c]:
            if seed not in got:
                missing += 1
            elif not same_forecast(got[seed], reference[seed], atol, rtol):
                differ += 1
    return {"differing_cases": differ, "not_evaluated": missing, "miss": differ > 0}


# ---------------------------------------------------------------------------------------------------------------
# Per-condition summaries and decisions


def _alarm(record, check):
    if check == "output":
        return record["raised"] or record["output_alarm"]
    if record["raised"] or check not in record:
        return None
    if check == "scalar_shift":
        return record[check]["status"] in ("FAIL", "ERROR")
    return record[check]["alarm"]


def clean_summary(config, result, checks):
    """Thresholds, clean alarm counts and the evaluation scores of the clean subject in one condition."""
    from .study3_stats import rank_counts, rate

    alpha = config["settings"]["alpha"]
    calibration = list(_case_records(result, "calibration").values())
    evaluation = list(_case_records(result, "evaluation").values())
    out = {"calibration_cases": len(calibration), "evaluation_cases": len(evaluation), "alarms": {}}
    for check in DETERMINISTIC:
        flags = [_alarm(record, check) for record in calibration + evaluation]
        flags = [flag for flag in flags if flag is not None]
        if check != "output" and not flags:
            continue
        out["alarms"][check] = {"count": int(sum(flags)), "cases": len(flags)}
    if any(record.get("scalar_shift", {}).get("status") == "ERROR" for record in calibration + evaluation):
        out["scalar_shift_errors"] = sum(record.get("scalar_shift", {}).get("status") == "ERROR"
                                         for record in calibration + evaluation)
    if "scores" in checks:
        scores = {name: [record["scores"][name] for record in calibration] for name in ("loss", "answer_key")}
        if any(record["scores"] is None for record in calibration + evaluation):
            raise RuntimeError("The clean subject returned an invalid forecast on a calibration or evaluation case")
        out["calibration_scores"] = scores
        out["evaluation"] = {record["seed"]: record["scores"] for record in evaluation}
        out["evaluation_ranks"] = {name: rank_counts([record["scores"][name] for record in evaluation],
                                                     scores[name]) for name in scores}
        levels = len(calibration) + 1
        out["false_alarms"] = {name: rate(sum(j / levels <= alpha for j in ranks), len(ranks))
                               for name, ranks in out["evaluation_ranks"].items()}
    if "calibrated_shift" in checks:
        statuses = [record["calibrated_shift"]["status"] for record in evaluation]
        decided = [status for status in statuses if status in ("PASS", "FAIL")]
        out["calibrated_shift"] = {"fails": decided.count("FAIL"), "cases": len(decided),
                                   "errors": statuses.count("ERROR"),
                                   "false_alarm_rate": rate(decided.count("FAIL"), len(decided))}
    return out


def source_condition(config, clean, result, checks, expected):
    """Alarm counts and the inputs of the statistical decisions of one source in one condition.

    ``expected`` is the number of evaluation cases; those the unit did not finish are counted as not evaluated.
    """
    from .study3_stats import beta_binomial_pvalue, binomial_pvalue, rank_counts

    alpha = config["settings"]["alpha"]
    done = list(_case_records(result, "evaluation").values())
    row = {"cases": expected, "completed": len(done), "not_evaluated": expected - len(done),
           "raised": sum(record["raised"] for record in done), "timed_out": bool(result["timed_out"]),
           "crashed": bool(result.get("crashed")), "deterministic": {}, "statistical": {}}
    for check in DETERMINISTIC:
        flags = [flag for flag in (_alarm(record, check) for record in done) if flag is not None]
        if check == "output" or check in checks:
            row["deterministic"][check] = {"alarms": int(sum(flags)), "cases": len(flags)}
    if "scalar_shift" in checks:
        row["scalar_shift_errors"] = sum(record.get("scalar_shift", {}).get("status") == "ERROR" for record in done)
    if "target_shift" in checks:
        row["target_shift_errors"] = sum(bool(record.get("target_shift", {}).get("error")) for record in done)
    valid = [record for record in done if record.get("scores") is not None]
    row["evaluated"] = len(valid)
    row["unevaluable"] = len(done) - len(valid)
    if "scores" in checks:
        levels = len(clean["calibration_scores"]["loss"]) + 1
        for name in ("loss", "answer_key"):
            ranks = rank_counts([record["scores"][name] for record in valid], clean["calibration_scores"][name])
            alarms = sum(j / levels <= alpha for j in ranks)
            row["statistical"][name] = {"alarms": int(alarms), "cases": len(ranks), "ranks": ranks,
                                        "p_value": beta_binomial_pvalue(alarms, len(ranks), levels - 1, alpha)}
        differences = [record["scores"]["answer_key"] - clean["evaluation"][record["seed"]]["answer_key"]
                       for record in valid]
        row["statistical"]["paired"] = {"cases": len(differences),
                                        "mean_difference": (sum(differences) / len(differences)
                                                            if differences else None),
                                        "differences": differences}
    if "calibrated_shift" in checks:
        statuses = [record["calibrated_shift"]["status"] for record in done if not record["raised"]]
        decided = [status for status in statuses if status in ("PASS", "FAIL")]
        row["statistical"]["calibrated_shift"] = {"alarms": decided.count("FAIL"), "cases": len(decided),
                                                  "errors": statuses.count("ERROR"),
                                                  "p_value": binomial_pvalue(decided.count("FAIL"), len(decided))}
    return row


def decide(config, clean_by_condition, rows, kinds_of):
    """Holm within (kind, condition) families and the deterministic rule; adds decisions to ``rows`` in place.

    ``rows`` maps source id to the list of per-condition rows of sources evaluated with every check.
    """
    from .study3_stats import deterministic_decision, holm_decisions, paired_pvalue

    alpha = config["settings"]["alpha"]
    for row_list in rows.values():
        for row in row_list:
            paired = row["statistical"].get("paired")
            if paired is not None:
                paired["p_value"] = paired_pvalue(paired["differences"], config["sign_flip_tag"],
                                                  config["sign_flips"])
    for c, clean in enumerate(clean_by_condition):
        for source, row_list in rows.items():
            row = row_list[c]
            row["decisions"] = {}
            for check, counts in row["deterministic"].items():
                clean_count = clean["alarms"].get(check, {"count": 0})["count"]
                row["decisions"][check] = deterministic_decision(counts["alarms"], counts["cases"], clean_count)
        for kind in KINDS:
            members = [source for source in rows if kind in kinds_of[source]]
            for check in STATISTICAL:
                pvalues = {source: rows[source][c]["statistical"][check]["p_value"] for source in members
                           if check in rows[source][c]["statistical"]}
                if not pvalues:
                    continue
                for source, decision in holm_decisions(pvalues, alpha).items():
                    rows[source][c]["decisions"].setdefault(check, {})[kind] = decision


def caught_anywhere(rows, check, kind):
    """Whether the check caught the source in at least one condition (Holm decision for statistical checks)."""
    for row in rows:
        decision = row.get("decisions", {}).get(check)
        if decision is None:
            continue
        if check in STATISTICAL:
            decision = decision.get(kind)
            if decision and decision["holm"]:
                return True
        elif decision["caught"]:
            return True
    return False


# ---------------------------------------------------------------------------------------------------------------
# Curves and funnel


def curves(config, clean_by_condition, rows, kinds_of):
    """Detection and false-alarm curves of the two calibrated alarms over every attainable level k / (n + 1)."""
    from .study3_stats import curve_intervals, detection_curve

    levels = len(clean_by_condition[0]["calibration_scores"]["loss"]) + 1
    out = {"levels": levels}
    for name in ("loss", "answer_key"):
        clean_ranks = [clean["evaluation_ranks"][name] for clean in clean_by_condition]
        entry = {"false_alarm": {"pooled": detection_curve([sum(clean_ranks, [])], levels),
                                 "intervals": curve_intervals([sum(clean_ranks, [])]),
                                 "by_condition": {condition["id"]: detection_curve([ranks], levels)
                                                  for condition, ranks in zip(config["conditions"], clean_ranks)}},
                 "detection": {}}
        for kind in KINDS:
            members = [source for source in rows if kind in kinds_of[source]]
            if not members:
                continue
            per_condition = {condition["id"]: detection_curve([rows[source][c]["statistical"][name]["ranks"]
                                                               for source in members], levels)
                             for c, condition in enumerate(config["conditions"])}
            pooled = [sum((row["statistical"][name]["ranks"] for row in rows[source]), []) for source in members]
            entry["detection"][kind] = {"pooled": detection_curve(pooled, levels),
                                        "intervals": curve_intervals(pooled), "by_condition": per_condition,
                                        "members": len(members)}
        out[name] = entry
    return out


def funnel(sources, classes, rows, misses, checks, run_ids=None):
    """Funnel counts per kind of source with Wilson intervals over distinct sources."""
    from .study3_stats import rate

    out = {}
    for kind in KINDS:
        labels = sum(sum(label["kind"] == kind for label in source["labels"]) for source in sources)
        members = [source["id"] for source in sources if any(label["kind"] == kind for label in source["labels"])]
        if not labels:
            continue
        run = [source for source in members if run_ids is None or source in run_ids]
        runnable = [source for source in run if classes[source]["sizes"][PRIMARY]["runnable"]]
        equivalent = [source for source in runnable if classes[source]["sizes"][PRIMARY]["equivalent"]]
        evaluated = [source for source in runnable if source not in equivalent]
        stages = {"generated": labels, "duplicate": labels - len(members), "distinct": len(members),
                  "run": len(run), "runnable": len(runnable), "equivalent": len(equivalent),
                  "not_equivalent": len(evaluated)}
        rates = {"runnable": rate(len(runnable), len(run)), "equivalent": rate(len(equivalent), len(runnable)),
                 "not_equivalent": rate(len(evaluated), len(runnable))}
        reasons: dict[str, int] = {}
        for source in run:
            reason = classes[source]["sizes"][PRIMARY]["reason"]
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        for check in ("output",) + tuple(check for check in DETERMINISTIC[1:] + STATISTICAL if check in checks):
            count = sum(caught_anywhere(rows[source], check, kind) for source in evaluated)
            stages[f"caught_{check}"] = count
            rates[f"caught_{check}"] = rate(count, len(evaluated))
        sensitivity = {size: {"runnable": sum(classes[source]["sizes"][size]["runnable"] for source in run),
                              "equivalent": sum(classes[source]["sizes"][size]["equivalent"] for source in run)}
                       for size in (classes[run[0]]["sizes"] if run else ())}
        out[kind] = {"stages": stages, "rates": rates, "not_runnable_reasons": reasons,
                     "equivalence_sensitivity": sensitivity,
                     "screen_misses": sum(misses.get(source, {}).get("miss", False) for source in equivalent)}
    return out


# ---------------------------------------------------------------------------------------------------------------
# One learner


def statistical_checks(checks):
    out = []
    if "scores" in checks:
        out += ["loss", "answer_key", "paired"]
    if "calibrated_shift" in checks:
        out.append("calibrated_shift")
    return tuple(out)


def evaluate_learner(config, learner, sources, checks, clean=None, classes=None, all_sources=None, run_ids=None):
    """Clean runs, screens, evaluations, decisions, curves and funnel for one learner and a list of sources.

    ``all_sources`` (default ``sources``) and ``run_ids`` give the funnel its generated set and the part of it that
    was run, when a sample was drawn.
    """
    clean = clean or run_clean(config, learner, checks)
    clean_by_condition = [clean_summary(config, result, checks) for result in clean]
    if classes is None:
        screens = run_screens(config, learner, sources, clean)
        classes = {source["id"]: classify(config, clean, screen) for source, screen in zip(sources, screens)}
    evaluations = run_evaluations(config, learner, sources, classes, clean, checks)
    rows, misses = {}, {}
    for source in sources:
        status = classes[source["id"]]["sizes"][PRIMARY]
        if not status["runnable"]:
            continue
        results = evaluations[source["id"]]
        if status["equivalent"]:
            misses[source["id"]] = screen_misses(config, clean, results)
        else:
            rows[source["id"]] = [source_condition(config, clean_by_condition[c], result, checks,
                                                   len(config["seeds"]["evaluation"][c]))
                                  for c, result in enumerate(results)]
    kinds_of = {source["id"]: {label["kind"] for label in source["labels"]} for source in sources}
    decide(config, clean_by_condition, rows, kinds_of)
    decided = statistical_checks(checks) + tuple(check for check in DETERMINISTIC if check in checks)
    result = {
        "learner": learner["name"], "checks": list(checks),
        "clean": {condition["id"]: _public_clean(summary)
                  for condition, summary in zip(config["conditions"], clean_by_condition)},
        "sources": [_public_source(source, classes[source["id"]], rows.get(source["id"]), misses.get(source["id"]),
                                   config) for source in sources],
        "funnel": funnel(all_sources or sources, classes, rows, misses, decided, run_ids),
    }
    if "scores" in checks:
        result["curves"] = curves(config, clean_by_condition, rows, kinds_of)
    return result


def _public_clean(summary):
    keep = {key: value for key, value in summary.items() if key not in ("evaluation", "calibration_scores")}
    keep["calibration_scores"] = summary.get("calibration_scores")
    return keep


def _public_source(source, classes, rows, misses, config):
    out = {"id": source["id"], "sha256": source.get("sha256"), "labels": source["labels"],
           "screen": classes, "screen_misses": misses}
    if rows is not None:
        out["conditions"] = {condition["id"]: {key: value for key, value in row.items()}
                             for condition, row in zip(config["conditions"], rows)}
    return out


# ---------------------------------------------------------------------------------------------------------------
# Parts


def run_part1(config, learner, sources, run_ids):
    """Part 1 for one learner (Ridge in the protocol): every source in ``run_ids`` is screened and run."""
    chosen = [source for source in sources if source["id"] in run_ids]
    return evaluate_learner(config, learner, chosen, config["part1_checks"], all_sources=sources, run_ids=run_ids)


def generic_order(sources, seed):
    """part_2_learners.bugs.generic: a seeded order of the generic sources, round-robin over operator families."""
    import numpy as np

    families: dict[str, list] = {}
    for source in sources:
        families.setdefault(source["family"], []).append(source)
    generator = np.random.default_rng(seed)
    names = list(families)
    order = [names[i] for i in generator.permutation(len(names))]
    queues = {name: [families[name][i] for i in generator.permutation(len(families[name]))] for name in order}
    out = []
    while any(queues.values()):
        for name in order:
            if queues[name]:
                out.append(queues[name].pop(0))
    return out


def classify_learners(config, learners):
    """part_2_learners.shift_classification: the public check on every classification case, then the calibrated
    check when needed."""
    from .study3_harness import run_units
    from .study3_stats import shift_class

    n = len(config["conditions"])

    def classification_units(mode, chosen):
        specs = [_unit_spec(config, learner, config["clean_source"], c, _stages(config, c, "classification"), (),
                            "clean", mode, mode=mode) for learner in chosen for c in range(n)]
        results = run_units(specs, [None] * len(specs), config["workers"])
        return {learner["name"]: [record for result in results[i * n:(i + 1) * n] for record in result["records"]
                                  if record.get("kind") == "case"] for i, learner in enumerate(chosen)}

    public = classification_units("classification_public", learners)
    needs = [learner for learner in learners if any(record["status"] != "PASS" for record in public[learner["name"]])]
    calibrated = classification_units("classification_calibrated", needs) if needs else {}
    out = {}
    for learner in learners:
        name = learner["name"]
        rows = calibrated.get(name)
        fails = None if rows is None else sum(record["status"] == "FAIL" for record in rows)
        cases = None if rows is None else sum(record["status"] in ("PASS", "FAIL") for record in rows)
        out[name] = {**shift_class([record["status"] for record in public[name]], fails, cases),
                     "public_errors": sum(record["status"] == "ERROR" for record in public[name]),
                     "public_max_absolute_error": max((record["max_absolute_error"] or 0.0
                                                       for record in public[name]
                                                       if isinstance(record["max_absolute_error"], float)),
                                                      default=None),
                     "calibrated_errors": None if rows is None else sum(record["status"] == "ERROR"
                                                                        for record in rows)}
    return out


def run_part2(config, learners, fixed_sources, generic_sources, learners_to_classify):
    """Part 2: shift classification, then per learner the fixed sources and the first generic sources that pass the
    screens."""
    classification = classify_learners(config, learners_to_classify)
    out = {"shift_classification": classification, "learners": {}}
    for learner in learners:
        shift_class = classification[learner["name"]]["class"]
        checks = tuple(sorted({CHECK_NAMES[name] for name in config["checks_by_class"][shift_class]} - {"output"}))
        clean = run_clean(config, learner, checks)
        screens = run_screens(config, learner, fixed_sources, clean)
        classes = {source["id"]: classify(config, clean, screen) for source, screen in zip(fixed_sources, screens)}
        chosen, screened, position = [], [], 0
        batch = max(config["generic_per_learner"], config["workers"])
        while len(chosen) < config["generic_per_learner"] and position < len(generic_sources):
            group = generic_sources[position:position + batch]
            position += len(group)
            for source, screen in zip(group, run_screens(config, learner, group, clean)):
                classes[source["id"]] = classify(config, clean, screen)
                screened.append(source["id"])
                status = classes[source["id"]]["sizes"][PRIMARY]
                if status["runnable"] and not status["equivalent"] and len(chosen) < config["generic_per_learner"]:
                    chosen.append(source)
        sources = fixed_sources + chosen
        result = evaluate_learner(config, learner, sources, checks, clean=clean, classes=classes)
        result["shift_class"] = shift_class
        result["generic_screened"] = screened
        result["generic_selected"] = [source["id"] for source in chosen]
        result["generic_screen"] = {source_id_: classes[source_id_] for source_id_ in screened}
        out["learners"][learner["name"]] = result
    return out


# ---------------------------------------------------------------------------------------------------------------
# Protocol binding


def manifest_sources(manifest_path: Path, subject: Path) -> tuple[dict, list[dict]]:
    """The committed mutant manifest and its sources; refuses a missing file or a digest that differs."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["subject"]["sha256"] != sha256(subject):
        raise ValueError("The mutant manifest was not generated from the frozen subject")
    sources = []
    for entry in manifest["sources"]:
        path = Path(manifest_path).parent / entry["file"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"Mutant source {entry['file']} is missing or differs from its manifest digest")
        sources.append({"id": entry["sha256"][:16], "sha256": entry["sha256"], "source": path.read_text(),
                        "labels": entry["labels"], "family": _family(entry["labels"])})
    return manifest, sources


def _family(labels):
    for label in labels:
        if label["kind"] == "cosmic_ray":
            return label["family"]
    return None


def sample_generic(sources: list[dict], cap: int, seed: int) -> list[dict]:
    """sources.cosmic_ray.sampling_rule: all if at most ``cap``, else a stratified sample of ``cap``."""
    import numpy as np

    if len(sources) <= cap:
        return list(sources)
    families: dict[str, list] = {}
    for source in sources:
        families.setdefault(source["family"], []).append(source)
    quotas = {name: cap * len(members) / len(sources) for name, members in families.items()}
    places = {name: min(len(families[name]), max(1, math.floor(quota + 0.5))) for name, quota in quotas.items()}
    names = list(families)
    while sum(places.values()) != cap:
        if sum(places.values()) < cap:
            open_ = [name for name in names if places[name] < len(families[name])]
            name = max(open_, key=lambda item: (quotas[item] - places[item], -names.index(item)))
            places[name] += 1
        else:
            open_ = [name for name in names if places[name] > 1]
            name = min(open_, key=lambda item: (quotas[item] - places[item], names.index(item)))
            places[name] -= 1
    generator = np.random.default_rng(seed)
    chosen = set()
    for name in names:
        order = generator.permutation(len(families[name]))
        chosen.update(families[name][i]["id"] for i in order[:places[name]])
    return [source for source in sources if source["id"] in chosen]


def manifest_path(protocol: dict) -> Path:
    """Where the committed mutant manifest lives (part_1_generated_mutants.sources.manifest.path)."""
    return common.ROOT / protocol["part_1_generated_mutants"]["sources"]["manifest"]["path"]


def bind(protocol: dict, output: Path, workers: int) -> dict:
    """The run configuration of Parts 1 and 2 from the protocol."""
    study2 = json.loads((common.ROOT / protocol["frozen_inputs"]["study_2_protocol"]["path"]).read_text())
    conditions = study2["model_study"]["conditions"]
    shared, part1, part2 = protocol["shared"], protocol["part_1_generated_mutants"], protocol["part_2_learners"]
    if [condition["id"] for condition in conditions] != part1["condition_ids"]:
        raise ValueError("The Study 2 conditions differ from part_1_generated_mutants.condition_ids")
    equivalence, limit = part1["funnel"]["equivalence"], part1["funnel"]["time_limit"]
    sizes = {str(size): int(count) for size, count in equivalence["inputs_per_condition"].items()}
    if str(equivalence["N"]) != PRIMARY or any(int(size) != count * len(conditions) for size, count in sizes.items()):
        raise ValueError("The equivalence screen sizes do not match funnel.equivalence")
    n = len(conditions)
    seeds = {"screen": [common.case_seeds(protocol, "equivalence_inputs", c) for c in range(n)],
             "calibration": [common.case_seeds(protocol, "model_calibration", c) for c in range(n)],
             "evaluation": [common.case_seeds(protocol, "model_evaluation", c) for c in range(n)],
             "classification": [common.case_seeds(protocol, "classification_cases", c) for c in range(n)]}
    model = study2["model_study"]
    return {"conditions": conditions, "seeds": seeds, "screen_sizes": sizes,
            "settings": {"atol": shared["tolerances"]["atol"], "rtol": shared["tolerances"]["rtol"],
                         "scalar_shift": float(model["scalar_shift"]), "target_shifts": model["target_shifts"],
                         "refits": part2["calibrated_shift_check"]["refits"], "alpha": shared["alpha"]},
            "learner_module": "experiments.study3_learners",
            "learner_tag": common.generator_tag(protocol, "learner_training"),
            "reference": "experiments.study3_reference:reference_forecast",
            "clean_source": (common.ROOT / part1["subject"]["module"]).read_text(),
            "time_limit": {"factor": float(limit["factor"]), "minimum": float(limit["minimum_seconds"])},
            "workers": workers, "scratch": str(Path(output) / "units"),
            "sign_flip_tag": common.generator_tag(protocol, "sign_flip"),
            "sign_flips": int(shared["per_mutant_statistical_rule"]["sign_flips"]),
            "part1_checks": PART1_CHECKS, "checks_by_class": part2["checks_by_class"],
            "generic_per_learner": int(part2["bugs"]["generic_per_learner"]),
            "cap": part1["sources"]["cosmic_ray"]["cap"],
            "mutant_sampling": common.generator_tag(protocol, "mutant_sampling"),
            "generic_order": common.generator_tag(protocol, "generic_mutant_order")}


def learner_entry(name):
    from .study3_learners import LEARNERS

    return {"name": name, "index": LEARNERS.index(name)}


def plan(protocol: dict, part: int, manifest_path: Path) -> dict:
    """Units, cases, fits and CPU seconds that a part would use; computes nothing."""
    from .study3_learners import LEARNERS

    counts = protocol["seeds"]["case_seed_blocks"]
    n = counts["model_calibration"][2]
    screen, calibration, evaluation, classification = (counts["equivalence_inputs"][3], counts["model_calibration"][3],
                                                       counts["model_evaluation"][3],
                                                       counts["classification_cases"][3])
    if Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text())
        kinds = [{label["kind"] for label in entry["labels"]} for entry in manifest["sources"]]
        generic = sum(kind == {"cosmic_ray"} for kind in kinds)
        fixed = len(kinds) - generic
        note = f"manifest {manifest_path}: {fixed} catalogue or Study 2 bug sources, {generic} generic sources"
    else:
        generic, fixed = protocol["part_1_generated_mutants"]["sources"]["cosmic_ray"]["cap"], 17
        note = f"no manifest at {manifest_path}; assumed {fixed} catalogue or Study 2 bug sources and {generic} generic"
    fits_per_case = {"exact": 4, "in distribution": 41, "not applicable": 1}
    rows = []
    if part == 1:
        sources = fixed + min(generic, protocol["part_1_generated_mutants"]["sources"]["cosmic_ray"]["cap"])
        work = [("ridge", "exact", sources)]
    else:
        wanted = int(protocol["part_2_learners"]["bugs"]["generic_per_learner"])
        work = [(name, protocol["part_2_learners"]["learners"][name]["expected_class"], fixed + wanted)
                for name in LEARNERS[1:]]
    total = 0.0
    for name, kind, sources in work:
        per_fit = PILOT_FIT_SECONDS[name]
        clean_cases = n * (screen + calibration + evaluation)
        units = n + 2 * n * sources
        cases = clean_cases + n * sources * (screen + screen + evaluation)
        fits = (n * screen + n * calibration * min(fits_per_case[kind], 4) + n * evaluation * fits_per_case[kind]
                + n * sources * (2 * screen + evaluation * fits_per_case[kind]))
        if part == 2:
            units += 2 * n
            cases += 2 * n * classification
            fits += n * classification * (2 + (41 if kind != "exact" else 0))
        seconds = fits * per_fit + units * UNIT_START_SECONDS
        total += seconds
        rows.append({"learner": name, "expected_class": kind, "sources": sources, "units": units, "cases": cases,
                     "fits": fits, "cpu_seconds": round(seconds)})
    return {"part": part, "note": note, "rows": rows, "cpu_hours": round(total / 3600, 2),
            "upper_bound": "every source is assumed runnable and not equivalent"}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tables(result: dict) -> dict[str, list[dict]]:
    """CSV rows: funnel, per-source decisions and clean alarm counts of one learner's result."""
    learner = result["learner"]
    funnel_rows, decision_rows, clean_rows = [], [], []
    for kind, entry in result["funnel"].items():
        for stage, count in entry["stages"].items():
            interval = entry["rates"].get(stage)
            funnel_rows.append({"learner": learner, "kind": kind, "stage": stage, "count": count,
                                "n": interval["n"] if interval else "",
                                "lower95": interval["ci95"][0] if interval else "",
                                "upper95": interval["ci95"][1] if interval else ""})
    for source in result["sources"]:
        for condition, row in (source.get("conditions") or {}).items():
            for check, decision in row.get("decisions", {}).items():
                variants = decision.items() if check in STATISTICAL else [("", decision)]
                for kind, item in variants:
                    statistic = row["statistical"].get(check, {})
                    counts = row["deterministic"].get(check, statistic)
                    decision_rows.append({"learner": learner, "source": source["id"], "condition": condition,
                                          "check": check, "family": kind, "alarms": counts.get("alarms", ""),
                                          "cases": counts.get("cases", ""), "p_value": item.get("p_value", ""),
                                          "family_size": item.get("family_size", ""),
                                          "unadjusted": item.get("unadjusted", ""), "holm": item.get("holm", ""),
                                          "deterministic": item.get("deterministic", ""),
                                          "caught": item.get("caught", "")})
    for condition, clean in result["clean"].items():
        for check, counts in clean["alarms"].items():
            clean_rows.append({"learner": learner, "condition": condition, "check": check, **counts})
    return {"funnel": funnel_rows, "decisions": decision_rows, "clean": clean_rows}


def _strip(result):
    """Results without the per-case lists that only the tables and curves need."""
    if isinstance(result, dict):
        return {key: _strip(value) for key, value in result.items() if key not in ("ranks", "differences")}
    if isinstance(result, list):
        return [_strip(value) for value in result]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Replace existing study artifacts")
    parser.add_argument("--plan", action="store_true", help="Print what would run and compute nothing")
    args = parser.parse_args(argv)
    protocol = common.load_protocol()
    part = f"part-{args.part}"
    mutants = manifest_path(protocol)
    if args.plan:
        print(json.dumps(plan(protocol, args.part, mutants), indent=2))
        return
    common.require_start(protocol, part)
    output = args.output or common.RESULTS / part
    from .common import protect_output

    protect_output(output, force=args.force)
    subject = common.ROOT / protocol["part_1_generated_mutants"]["subject"]["module"]
    manifest, sources = manifest_sources(mutants, subject)
    config = bind(protocol, output, args.workers)
    generic = [source for source in sources if {label["kind"] for label in source["labels"]} == {"cosmic_ray"}]
    fixed = [source for source in sources if source not in generic]
    started = time.perf_counter()
    from threadpoolctl import threadpool_info, threadpool_limits

    with threadpool_limits(limits=1):
        if args.part == 1:
            sampled = sample_generic(generic, config["cap"], config["mutant_sampling"])
            run_ids = {source["id"] for source in fixed + sampled}
            result = {"ridge": run_part1(config, learner_entry("ridge"), sources, run_ids),
                      "generic_generated": len(generic), "generic_run": len(sampled)}
            learner_results = [result["ridge"]]
        else:
            from .study3_learners import LEARNERS

            ordered = generic_order(generic, config["generic_order"])
            result = run_part2(config, [learner_entry(name) for name in LEARNERS[1:]], fixed, ordered,
                               [learner_entry(name) for name in LEARNERS])
            learner_results = list(result["learners"].values())
        pools = threadpool_info()
    seconds = time.perf_counter() - started
    results = {"study_id": protocol["study_id"], "protocol_version": protocol["protocol_version"], "part": part,
               "mutant_manifest_sha256": sha256(mutants), **_strip(result)}
    paths = [Path(__file__)] + [common.ROOT / name for name in protocol["part_1_generated_mutants"]["subject"]
                                ["frozen_files"]]
    paths += [common.ROOT / f"experiments/{name}.py" for name in ("study3_harness", "study3_stats", "study3_shift",
                                                                   "study3_common")]
    common.write_record(output, results, sources=tuple(paths), force=args.force,
                        runtime={"seconds": seconds, "workers": args.workers, "threadpools": pools},
                        extra={"mutant_manifest_sha256": sha256(mutants),
                               "cosmic_ray_version": manifest.get("tool", {}).get("version")})
    rows = {"funnel": [], "decisions": [], "clean": []}
    for learner_result in learner_results:
        for name, table in tables(learner_result).items():
            rows[name] += table
    for name, table in rows.items():
        if table:
            _write_csv(output / f"{name}.csv", table)
    if args.part == 2:
        _write_csv(output / "shift_classification.csv", [{"learner": name, **row}
                                           for name, row in result["shift_classification"].items()])
    print(json.dumps({"output": str(output), "seconds": seconds}, indent=2))


if __name__ == "__main__":
    main()
