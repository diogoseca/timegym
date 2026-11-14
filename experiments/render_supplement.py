"""Render recorded supplement results; never rerun or modify the protocol.

    python -m experiments.render_supplement

Writes paper/generated-supplement.tex and paper/generated-macros.tex. Numerical
results and their protocol are checked against the recorded manifest first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import close_statistics, interval_cell, rate_cell, require_conclusion, scientific_tex, sha256


ROOT = Path(__file__).resolve().parents[1]
CHECKS = ("contract", "scalar_shift", "target_shift", "conformance", "loss")
CONDITION_LABELS = {
    "white_noise_locations": "White noise",
    "ar1_positive": "Positive AR(1)",
    "ar1_negative": "Negative AR(1)",
    "opposing_trends": "Opposing trends",
    "seasonal_scales": "Seasonal scales",
    "random_walk_drifts": "Random walk with drift",
    "factorized_white_noise": "White noise: independent targets",
    "correlated_white_noise": "White noise: correlated targets",
    "negative_ar1": "Negative AR(1)",
    "correlated_random_walk": "Correlated random walk with drift",
}
VARIANT_LABELS = {"clean": "Exact-reference control", "partial_shuffle": "Partial dependence shuffle",
                  "reduced_spread": "Reduced spread"}
SHORT_FAULTS = {"reverse_prediction_lags": "Reversed lags", "stale_recursive_state": "Stale state",
                "omit_inverse_target_scaling": "No inverse scale", "swap_recursive_targets": "Target swap"}


def validate_recorded_results(results_path, results, manifest):
    digest = sha256(results_path)
    if manifest["results_sha256"] != digest:
        raise ValueError("Recorded results do not match their manifest fingerprint")
    if results["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("Recorded protocol fingerprints disagree")
    if results["protocol_version"] != "2.0" or manifest["protocol"]["protocol_version"] != "2.0":
        raise ValueError("Expected supplement protocol 2.0")


def branch_counts(results, protocol):
    branches = ["clean", *protocol["model_study"]["faults"]]
    rows = []
    for branch in branches:
        records = [row for row in results["model"]["records"] if row["branch"] == branch]
        rows.append({"branch": branch, "n": len(records),
                     "counts": [sum(row["alarms"][check] for row in records) for check in CHECKS]})
    return rows


def render_tex(results, manifest, followup=None):
    """Study 2 text and tables; ``followup`` adds the exploratory answer-key alarm (protocol 2.1)."""
    protocol = manifest["protocol"]
    model, covariance = protocol["model_study"], protocol["covariance_study"]
    losses = results["model"]["loss_summaries"]
    improved = sum(row["clean_minus_faulty_mse"]["mean"] < 0 for row in losses)
    intervals_below_zero = sum(row["clean_minus_faulty_mse"]["ci95"][1] < 0 for row in losses)
    loss_comparisons = [row for row in results["model"]["condition_summaries"] if row["check"] == "loss"]
    loss_gained = sum(row["paired_alarm_gain"] for row in loss_comparisons)
    loss_lost = sum(row["paired_alarm_loss"] for row in loss_comparisons)
    repair_counterexamples = []
    for row in losses:
        delta = row["clean_minus_faulty_mse"]
        if delta["ci95"][0] > 0:
            repair_counterexamples.append(
                f"For {CONDITION_LABELS[row['condition']].lower()} with {SHORT_FAULTS[row['fault']].lower()}, "
                f"clean-minus-faulty mean MSE was {interval_cell(delta)} (paired 95\\% t interval); "
                f"only {row['cases_improved_by_repair']}/{row['n']} realized forecasts improved."
            )
    max_crps = max(abs(row["score_delta_from_clean"]["crps"]) for row in results["covariance"]["records"]
                   if row["variant"] == "partial_shuffle")
    suites = {row["variant"]: row for row in results["covariance"]["suite_summaries"]}
    require_conclusion(all(not row["alarms"]["contract"] for row in results["model"]["records"]),
                       "all retained implementation branches preserve the output contract")
    require_conclusion(suites["clean"]["holm_any_false_alarm"]["rate"] < suites["clean"]["raw_any_false_alarm"]["rate"],
                       "Holm reduces clean suite false alarms")
    require_conclusion(suites["reduced_spread"]["holm_any_alarm"]["rate"] < suites["reduced_spread"]["raw_any_alarm"]["rate"],
                       "Holm reduces detection in reduced-spread suites")
    require_conclusion(max_crps < 1e-12, "partial permutations preserve marginal CRPS up to roundoff")
    require_conclusion(all(row["null_hypotheses"] == {"clean": 4, "partial_shuffle": 1, "reduced_spread": 0}[row["variant"]]
                           for row in suites.values()), "declared suite null counts match the table caption")

    lines = [
        "% Generated from experiments/results-v2; do not edit numerical statements manually.",
        r"\subsection{Study 2: implementation regressions}",
        "Figure~\\ref{fig:diagnostic-evidence} gives the pooled "
        "alarm counts, and Table~\\ref{tab:supplement-complete} gives them for each condition. "
        "The paired comparison is clean-minus-faulty, with no separate repaired arm. "
        "Any output-contract violation terminates the study.",
    ]
    lines.extend([
        f"Across the fixed condition--fault pairs, the calibrated loss alarm changed from no alarm to alarm "
        f"in {loss_gained} paired comparisons and from alarm to no alarm in {loss_lost}. "
        "The same clean case appears once per fault in these descriptive paired counts; they are not "
        "independent replicates of a pooled statistical test.",
        f"The clean implementation reduced mean realized MSE in {improved}/{len(losses)} condition--fault pairs; "
        f"{intervals_below_zero}/{len(losses)} paired 95\\% t intervals were wholly below zero. "
        "The full paired loss differences, intervals, and numbers of individually improved cases are retained "
        r"in \code{model-repair-losses.csv}.",
        *repair_counterexamples,
    ])
    key_cells = None
    if followup is not None:
        from .render_followup import answer_key_cells, check_followup_conclusions
        check_followup_conclusions(followup)
        key_cells = answer_key_cells(followup)
        expected = {(condition["id"], branch) for condition in model["conditions"] for branch in ["clean", *model["faults"]]}
        require_conclusion(set(key_cells) == expected, "the follow-up covers every Study 2 condition and branch")
        for (condition, branch), row in {(row["condition"], row["branch"]): row
                                         for row in followup["answer_key_alarm"]["rows"]}.items():
            frozen = sum(record["alarms"]["loss"] for record in results["model"]["records"]
                         if record["condition"] == condition and record["branch"] == branch)
            require_conclusion(row["loss_alarms"] == frozen, "the follow-up quotes the frozen loss alarms")
        lines.append(
            "As an exploratory addition, chosen after these results were known (protocol 2.1, "
            r"\code{experiments/results-v2-followup/}), an alarm on the distance from the answer key, calibrated "
            r"in the same way, caught reversed lags or stale state in \AnswerKeySubtleAlarms{} of \SubtleCases{} "
            r"cases outside the seasonal condition, against \LossSubtleAlarms{} for the realized-loss alarm, with "
            r"\AnswerKeyCleanAlarms{} rather than \LossCleanAlarms{} false alarms in \CleanNonSeasonalCases{} clean "
            "cases. It helps most on the autoregressive conditions, where future noise hides these faults, and "
            "little on trends and random walks, where the clean pipeline's own estimation error varies too much "
            "from case to case.")
    columns = "C & S & T & N & L" + (" & K" if key_cells else "")
    lines.extend([
        r"\begin{table}[p]\centering\scriptsize",
        r"\caption{Complete condition-specific implementation alarm counts. Each entry is out of "
        f"{model['evaluation_cases']} paired cases. C, S, T, N, and L denote contract, scalar shift, target "
        "shift, NumPy agreement, and calibrated loss respectively. "
        + ("K is the alarm on the distance from the answer key, an exploratory addition chosen after these "
           "results were known (protocol 2.1), calibrated like L on the same clean cases. " if key_cells else "")
        + "Clean rows are negative controls; there is no separate corrected-refit row.}",
        r"\label{tab:supplement-complete}",
        r"\begin{tabular}{@{}ll" + "r" * (len(CHECKS) + bool(key_cells)) + r"@{}}\toprule",
        r"Condition & Branch & " + columns + r" \\ \midrule",
    ])
    for condition in model["conditions"]:
        records = [row for row in results["model"]["records"] if row["condition"] == condition["id"]]
        for index, branch in enumerate(["clean", *model["faults"]]):
            values = [sum(row["alarms"][check] for row in records if row["branch"] == branch) for check in CHECKS]
            if key_cells:
                values.append(key_cells[condition["id"], branch])
            label = "Clean" if branch == "clean" else SHORT_FAULTS[branch]
            lines.append((CONDITION_LABELS[condition["id"]] if index == 0 else "") + " & " + label
                         + " & " + " & ".join(str(v) for v in values) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        r"\subsection{Study 2: covariance checks under multiplicity}",
        f"For clean suites, any false alarm occurred in {suites['clean']['raw_any_false_alarm']['count']}/"
        f"{suites['clean']['raw_any_false_alarm']['n']} raw suites versus "
        f"{suites['clean']['holm_any_false_alarm']['count']}/{suites['clean']['holm_any_false_alarm']['n']} "
        "after Holm correction. For reduced-spread suites, detection of at least one altered condition "
        f"fell from {suites['reduced_spread']['raw_any_alarm']['count']}/"
        f"{suites['reduced_spread']['raw_any_alarm']['n']} "
        r"([\SpreadRawAnyLow{}, \SpreadRawAnyHigh{}]\%) to "
        f"{suites['reduced_spread']['holm_any_alarm']['count']}/{suites['reduced_spread']['holm_any_alarm']['n']} "
        r"([\SpreadHolmAnyLow{}, \SpreadHolmAnyHigh{}]\%).",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Any false alarm per four-check suite, counts and Wilson 95\% intervals (percent). Clean suites have four true nulls and partial-shuffle suites one; reduced-spread suites have none and are omitted.}",
        r"\label{tab:supplement-familywise}",
        r"\begin{tabular}{@{}lrr@{}}\toprule",
        r"Variant & Raw & Holm \\ \midrule",
    ])
    for row in results["covariance"]["suite_summaries"]:
        if row["null_hypotheses"]:
            lines.append(VARIANT_LABELS[row["variant"]] + " & " + rate_cell(row["raw_any_false_alarm"])
                         + " & " + rate_cell(row["holm_any_false_alarm"]) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{All condition-specific covariance rejections, raw and Holm, out of "
        f"{covariance['independent_suites']} suites. Clean columns and the first shuffled row are negative "
        "controls; other entries measure power on these alternatives.}",
        r"\label{tab:supplement-covariance}",
        r"\begin{tabular}{@{}p{0.34\linewidth}rrrrrr@{}}\toprule",
        r" & \multicolumn{2}{c}{Clean} & \multicolumn{2}{c}{Partial shuffle} & \multicolumn{2}{c}{Reduced spread} \\",
        r"Condition & Raw & Holm & Raw & Holm & Raw & Holm \\ \midrule",
    ])
    for condition in covariance["conditions"]:
        counts = []
        for variant in covariance["variants"]:
            row = next(row for row in results["covariance"]["condition_summaries"]
                       if row["condition"] == condition["id"] and row["variant"] == variant)
            counts.extend([row["raw"]["count"], row["holm"]["count"]])
        lines.append(CONDITION_LABELS[condition["id"]] + " & " + " & ".join(str(v) for v in counts) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        f"Coordinate-wise sorted sample arrays were identical before and after every partial permutation. "
        f"The largest absolute marginal CRPS change was {scientific_tex(max_crps)}. Paired energy and variogram "
        "score differences are retained for every condition.",
    ])
    return "\n\n".join(lines) + "\n"


def render_macros(results, manifest, pilot_manifest, followup=None, addendum=None):
    """Caption and prose macros.

    ``followup`` adds those of the exploratory follow-up (protocol 2.1) and
    ``addendum`` those of the second exploratory record (protocol 2.2).
    """
    selected = next(row for row in results["model"]["loss_summaries"]
                    if row["condition"] == "random_walk_drifts" and row["fault"] == "stale_recursive_state")
    records = results["model"]["records"]
    clean = {(row["condition"], row["seed"]): row for row in records if row["branch"] == "clean"}
    faulty = [row for row in records if row["condition"] == selected["condition"] and row["branch"] == selected["fault"]]
    from timegym.statistics import paired_t_interval
    differences = [clean[(row["condition"], row["seed"])]["realized_mse"] - row["realized_mse"] for row in faulty]
    delta = paired_t_interval(differences)
    require_conclusion(close_statistics(delta, selected["clean_minus_faulty_mse"]), "caption interval agrees with paired raw losses")
    improved = sum(value < 0 for value in differences)
    require_conclusion(improved == selected["cases_improved_by_repair"], "caption improvement count agrees with raw losses")
    counts = {sum(row["condition"] == condition for row in clean.values())
              for condition in {row["condition"] for row in clean.values()}}
    require_conclusion(counts == {selected["n"]}, "caption has a common case count per condition")
    shared = ("numpy", "pandas", "scipy")
    require_conclusion(pilot_manifest["python"] == manifest["python"]
                       and all(pilot_manifest["versions"][name] == manifest["versions"][name] for name in shared),
                       "both studies ran with the same Python, NumPy, pandas and SciPy versions")
    costs = {**results["model"]["cost_counts"], **results["covariance"]["cost_counts"]}
    suites = {row["variant"]: row for row in results["covariance"]["suite_summaries"]}
    intervals = {"CleanRawAny": suites["clean"]["raw_any_false_alarm"],
                 "CleanHolmAny": suites["clean"]["holm_any_false_alarm"],
                 "SpreadRawAny": suites["reduced_spread"]["raw_any_alarm"],
                 "SpreadHolmAny": suites["reduced_spread"]["holm_any_alarm"]}
    values = {"RepairDeltaMean": f"{delta['mean']:.3f}", "RepairDeltaLow": f"{delta['ci95'][0]:.3f}",
              "RepairDeltaHigh": f"{delta['ci95'][1]:.3f}", "RepairImproved": improved,
              "CasesPerCondition": selected["n"], "CasesPerRow": len(clean),
              "StudyPython": manifest["python"], "StudyNumPy": manifest["versions"]["numpy"],
              "StudyPandas": manifest["versions"]["pandas"], "StudySciPy": manifest["versions"]["scipy"],
              "StudySklearn": manifest["versions"]["scikit-learn"],
              "StudySklearnFits": f"{costs['sklearn_fits']:,}",
              "StudyNumPySolves": f"{costs['independent_ridge_solves']:,}",
              "StudyCovarianceChecks": f"{costs['covariance_checks']:,}",
              "StudyCalibrationEnsembles": f"{costs['calibration_ensembles']:,}"}
    for name, stat in intervals.items():
        values[name + "Low"], values[name + "High"] = (f"{100 * bound:.0f}" for bound in stat["ci95"])
    text = "% Generated from frozen supplement records; do not edit.\n" + "".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items())
    if followup is None:
        return text
    text += followup_text(results, records, clean, followup)
    if addendum is None:
        return text
    return text + addendum_text(results, addendum)


def followup_text(results, records, clean, followup):
    from .render_followup import followup_macros
    repair = followup["repair_mechanism"]
    recorded = next(row for row in results["model"]["loss_summaries"]
                    if row["condition"] == repair["condition"] and row["fault"] == repair["fault"])
    require_conclusion(close_statistics(repair["frozen_clean_minus_faulty_reference_mse"],
                                        recorded["clean_minus_faulty_reference_mse"]),
                       "the follow-up quotes the frozen clean-minus-faulty distance from the answer key")
    closer = sum(row["mean_reference_mse"] < clean[(row["condition"], row["seed"])]["mean_reference_mse"]
                 for row in records if row["condition"] == repair["condition"] and row["branch"] == repair["fault"])
    require_conclusion(closer == repair["faulty_closer_to_answer_key"],
                       "the follow-up count of faulty cases closer to the answer key agrees with the frozen records")
    return "% Exploratory follow-up chosen after the Study 2 results were known (protocol 2.1,\n" \
        "% experiments/results-v2-followup); do not edit.\n" + "".join(
            f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in followup_macros(followup).items())


def addendum_text(results, addendum):
    from .render_addendum import addendum_macros, variant_summaries
    frozen = {row["variant"]: row for row in results["covariance"]["suite_summaries"]}
    def count(stat):
        return None if stat is None else (stat["count"], stat["n"])

    for variant, row in variant_summaries(addendum).items():
        detected, false = row["suites_with_any_rejection"], row["suites_rejecting_a_true_null"]
        require_conclusion(all(count(ours[method]) == count(frozen[variant][key])
                               for ours, kind in ((detected, "alarm"), (false, "false_alarm"))
                               for method in ("raw", "holm") for key in [f"{method}_any_{kind}"]),
                           f"the addendum's raw and Holm suite counts for {variant} equal the frozen ones")
    return "% Second exploratory record, chosen after the Study 2 results and the follow-up were known\n" \
        "% (protocol 2.2, experiments/results-v2-addendum); do not edit.\n" + "".join(
            f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in addendum_macros(addendum).items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments/results-v2")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    parser.add_argument("--followup-dir", type=Path, default=None,
                        help="exploratory follow-up record (default experiments/results-v2-followup)")
    parser.add_argument("--addendum-dir", type=Path, default=None,
                        help="second exploratory record (default experiments/results-v2-addendum)")
    parser.add_argument("--pilot-manifest", type=Path, default=ROOT / "experiments/results/manifest.json",
                        help="Study 1 manifest whose software versions must match (default the recorded one)")
    args = parser.parse_args()
    path = args.results_dir / "results.json"
    results = json.loads(path.read_text())
    manifest = json.loads((args.results_dir / "manifest.json").read_text())
    validate_recorded_results(path, results, manifest)
    from .render_followup import FOLLOWUP_DIR, load_followup
    followup, _ = load_followup(args.followup_dir if args.followup_dir else FOLLOWUP_DIR)
    from .render_addendum import ADDENDUM_DIR, load_addendum
    addendum, _ = load_addendum(args.addendum_dir if args.addendum_dir else ADDENDUM_DIR)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    fragment = args.paper_dir / "generated-supplement.tex"
    fragment.write_text(render_tex(results, manifest, followup))
    pilot_manifest = json.loads(args.pilot_manifest.read_text())
    (args.paper_dir / "generated-macros.tex").write_text(render_macros(results, manifest, pilot_manifest, followup,
                                                                             addendum))
    print(json.dumps({"tables": str(fragment), "macros": str(args.paper_dir / "generated-macros.tex")}))


if __name__ == "__main__":
    main()
