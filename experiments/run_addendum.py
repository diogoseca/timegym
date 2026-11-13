"""Run the second exploratory record for Study 2 (protocol 2.2) from stored values only.

    python -m experiments.run_addendum --output experiments/results-v2-addendum

The two analyses were chosen on 2026-11-13, after the protocol 2.0 results and
the protocol 2.1 follow-up were known (see experiments/protocol-v2-addendum.json).
Both read only values stored in experiments/results-v2/results.json and refit
nothing: suite-level multiplicity from the stored covariance p-values, and a
paired check of the distance from the answer key from the stored evaluation
records. The runner writes results.json and manifest.json; frozen records are
read, never written.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform

import numpy as np

import timegym
from timegym.statistics import wilson_interval

from .common import protect_output, sha256
from .run_supplement import holm_rejections

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/protocol-v2-addendum.json"
STUDY2_PROTOCOL = ROOT / "experiments/protocol-v2.json"
STUDY2_RESULTS = ROOT / "experiments/results-v2/results.json"
VERSION = "2.2"
PER_CHECK = ("raw", "holm", "hochberg")
GLOBAL = ("simes", "fisher")


def validate_protocol(protocol, root=ROOT):
    if protocol.get("protocol_version") != VERSION or protocol.get("status") != "exploratory":
        raise ValueError("Expected the exploratory addendum protocol 2.2")
    for item in protocol["frozen_inputs"].values():
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input {item['path']} differs from the digest the addendum declares")


def _pvalues(pvalues, alpha):
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not len(pvalues) or not np.isfinite(pvalues).all():
        raise ValueError("pvalues must be a finite nonempty vector")
    if np.any((pvalues <= 0) | (pvalues > 1)) or not 0 < alpha < 1:
        raise ValueError("Invalid p-values or alpha")
    return pvalues


def hochberg_rejections(pvalues, alpha):
    """Hochberg step-up rejection mask, retaining the input hypothesis order."""
    pvalues = _pvalues(pvalues, alpha)
    m = len(pvalues)
    order = np.argsort(pvalues, kind="stable")
    rejected = np.zeros(m, dtype=bool)
    for rank in range(m, 0, -1):
        if pvalues[order[rank - 1]] <= alpha / (m - rank + 1):
            rejected[order[:rank]] = True
            break
    return rejected


def simes_rejects(pvalues, alpha):
    """Simes' global test of the intersection of all nulls."""
    pvalues = np.sort(_pvalues(pvalues, alpha))
    m = len(pvalues)
    return bool(np.any(pvalues <= alpha * np.arange(1, m + 1) / m))


def chi_square_sf_even(x, df):
    """Chi-square survival function for even degrees of freedom, in closed form."""
    if df <= 0 or df % 2:
        raise ValueError("The closed form needs a positive even number of degrees of freedom")
    if x < 0:
        raise ValueError("A chi-square statistic is nonnegative")
    half = x / 2
    return float(math.exp(-half) * sum(half ** k / math.factorial(k) for k in range(df // 2)))


def fisher_combination(pvalues):
    """Fisher's statistic -2 sum log p and its chi-square probability with 2m degrees of freedom."""
    pvalues = _pvalues(pvalues, 0.5)
    statistic = float(-2 * np.log(pvalues).sum())
    return statistic, chi_square_sf_even(statistic, 2 * len(pvalues))


def suite_multiplicity(spec, conditions, records):
    """Raw, Holm, Hochberg, Simes and Fisher decisions for every stored covariance suite."""
    alpha = spec["alpha"]
    by_key = {(row["variant"], row["suite"], row["condition"]): row for row in records}
    if len(by_key) != len(records):
        raise ValueError("Covariance records repeat a suite, variant and condition")
    suites = sorted({row["suite"] for row in records})
    suite_rows, checked = [], 0
    for variant in spec["variants"]:
        for suite in suites:
            rows = [by_key[variant, suite, condition] for condition in conditions]
            p = [row["p_value"] for row in rows]
            raw = np.asarray(p) <= alpha
            holm = holm_rejections(p, alpha)
            for row, raw_alarm, holm_alarm in zip(rows, raw, holm):
                if bool(raw_alarm) != row["raw_alarm"] or bool(holm_alarm) != row["holm_alarm"]:
                    raise RuntimeError(f"Recomputed decision for {variant} suite {suite} {row['condition']} "
                                       "differs from the frozen record")
                checked += 1
            statistic, fisher_p = fisher_combination(p)
            suite_rows.append({"variant": variant, "suite": suite, "p_values": p,
                               "null_true": [row["null_true"] for row in rows],
                               "raw": raw.tolist(), "holm": holm.tolist(),
                               "hochberg": hochberg_rejections(p, alpha).tolist(),
                               "simes": simes_rejects(p, alpha), "fisher_statistic": statistic,
                               "fisher_p_value": fisher_p, "fisher": fisher_p <= alpha})
    return {"alpha": alpha, "conditions": list(conditions), "family_size": len(conditions),
            "suites": len(suites), "frozen_decisions_checked": checked, "suite_rows": suite_rows,
            "variants": multiplicity_summaries(suite_rows, spec["variants"], conditions)}


def multiplicity_summaries(suite_rows, variants, conditions):
    """Suite-level counts with Wilson intervals, and per-condition counts, for each variant."""
    out = []
    for variant in variants:
        rows = [row for row in suite_rows if row["variant"] == variant]
        nulls = rows[0]["null_true"]
        if any(row["null_true"] != nulls for row in rows):
            raise ValueError(f"True nulls differ between suites of {variant}")
        detected = {method: wilson_interval([any(row[method]) for row in rows]) for method in PER_CHECK}
        detected.update({method: wilson_interval([row[method] for row in rows]) for method in GLOBAL})
        if any(nulls):
            false = {method: wilson_interval([any(flag and null for flag, null in zip(row[method], nulls))
                                              for row in rows]) for method in PER_CHECK}
            false.update({method: wilson_interval([row[method] for row in rows]) if all(nulls) else None
                          for method in GLOBAL})
        else:
            false = {method: None for method in PER_CHECK + GLOBAL}
        per_condition = {condition: {method: int(sum(row[method][index] for row in rows)) for method in PER_CHECK}
                         for index, condition in enumerate(conditions)}
        out.append({"variant": variant, "n": len(rows),
                    "true_null_conditions": [c for c, null in zip(conditions, nulls) if null],
                    "suites_with_any_rejection": detected, "suites_rejecting_a_true_null": false,
                    "rejections_per_condition": per_condition})
    return out


def sign_flip_pvalue(differences, flips, generator):
    """One-sided sign-flip p-value for a positive mean paired difference."""
    differences = np.asarray(differences, dtype=float)
    simulated = (generator.choice([-1, 1], size=(flips, len(differences))) * differences).mean(axis=1)
    return float((1 + np.sum(simulated >= differences.mean())) / (flips + 1))


def paired_answer_key(spec, conditions, records):
    """Sign-flip tests of the paired distance from the answer key, Holm across pairs, at each seed count."""
    test = spec["test"]
    if test["generator"] != "numpy.random.default_rng":
        raise ValueError("The addendum declares numpy.random.default_rng")
    distance = {(row["condition"], row["branch"], row["seed"]): row["mean_reference_mse"] for row in records}
    if len(distance) != len(records):
        raise ValueError("Model records repeat a condition, branch and seed")
    seeds = {condition: sorted({row["seed"] for row in records if row["condition"] == condition})
             for condition in conditions}
    generator = np.random.default_rng(test["seed"])
    analyses = []
    for count in spec["evaluation_seed_counts"]:
        pairs = []
        for condition in conditions:
            chosen = seeds[condition][:count]
            if len(chosen) != count:
                raise ValueError(f"{condition} has fewer than {count} evaluation seeds")
            for fault in spec["faults"]:
                d = np.asarray([distance[condition, fault, seed] - distance[condition, "clean", seed]
                                for seed in chosen])
                clean = np.mean([distance[condition, "clean", seed] for seed in chosen])
                pairs.append({"condition": condition, "fault": fault, "n": count,
                              "clean_mean_reference_mse": float(clean),
                              "mean_difference": float(d.mean()), "worse": int((d > 0).sum()),
                              "p_value": sign_flip_pvalue(d, test["flips"], generator)})
        rejected = holm_rejections([pair["p_value"] for pair in pairs], spec["alpha"])
        for pair, flag in zip(pairs, rejected):
            pair["holm_worse"] = bool(flag)
        analyses.append({"evaluation_seeds": count, "pairs": pairs, "flagged": int(rejected.sum()),
                         "n_pairs": len(pairs),
                         "not_flagged": [[pair["condition"], pair["fault"]] for pair in pairs
                                         if not pair["holm_worse"]]})
    return {"alpha": spec["alpha"], "flips": test["flips"], "seed": test["seed"], "conditions": list(conditions),
            "faults": list(spec["faults"]), "analyses": analyses}


def source_fingerprints():
    paths = sorted((ROOT / "timegym").rglob("*.py"))
    paths += [ROOT / "experiments" / name for name in ("common.py", "run_supplement.py", "run_addendum.py")]
    paths += [PROTOCOL]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def run(protocol):
    study2_protocol = json.loads(STUDY2_PROTOCOL.read_text())
    study2 = json.loads(STUDY2_RESULTS.read_text())
    covariance = [condition["id"] for condition in study2_protocol["covariance_study"]["conditions"]]
    model = [condition["id"] for condition in study2_protocol["model_study"]["conditions"]]
    if protocol["paired_answer_key"]["faults"] != study2_protocol["model_study"]["faults"]:
        raise ValueError("The addendum faults must be the protocol 2.0 faults in their order")
    return {
        "study_id": protocol["study_id"], "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256(PROTOCOL), "status": protocol["status"],
        "suite_multiplicity": suite_multiplicity(protocol["suite_multiplicity"], covariance,
                                                 study2["covariance"]["records"]),
        "paired_answer_key": paired_answer_key(protocol["paired_answer_key"], model, study2["model"]["records"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results-v2-addendum")
    parser.add_argument("--force", action="store_true", help="Replace an existing addendum record")
    args = parser.parse_args()
    frozen = {(ROOT / "experiments" / name).resolve() for name in ("results", "results-v2", "results-v2-followup")}
    if args.output.resolve() in frozen:
        parser.error("The frozen study directories are protected; choose a separate output")
    protect_output(args.output, force=args.force)
    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    fingerprints = source_fingerprints()
    results = run(protocol)
    if fingerprints != source_fingerprints():
        raise RuntimeError("Source files changed during the addendum; rerun it")
    args.output.mkdir(parents=True, exist_ok=True)
    numerical = args.output / "results.json"
    numerical.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest = {"protocol": protocol, "protocol_sha256": sha256(PROTOCOL), "results_sha256": sha256(numerical),
                "frozen_inputs_sha256": {item["path"]: sha256(ROOT / item["path"])
                                         for item in protocol["frozen_inputs"].values()},
                "source_files": fingerprints, "python": platform.python_version(), "platform": platform.platform(),
                "package_version": timegym.__version__,
                "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy")},
                "source_note": "Hashes identify the working source, including generated package files. "
                               "The addendum was chosen after the protocol 2.0 results and the protocol 2.1 "
                               "follow-up were known, reads only stored values and is exploratory."}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    multiplicity = {row["variant"]: {method: stat["count"] for method, stat in row["suites_with_any_rejection"].items()}
                    for row in results["suite_multiplicity"]["variants"]}
    paired = {row["evaluation_seeds"]: row["flagged"] for row in results["paired_answer_key"]["analyses"]}
    print(json.dumps({"output": str(args.output), "results_sha256": manifest["results_sha256"],
                      "suites_with_any_rejection": multiplicity, "pairs_flagged": paired}, indent=2))


if __name__ == "__main__":
    main()
