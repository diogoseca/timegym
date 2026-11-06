"""Render recorded supplement results; never rerun or modify the protocol.

    python -m experiments.render_supplement

Writes paper/generated-supplement.tex and two PNG/PDF figure pairs. Numerical
results and their protocol are checked against the recorded manifest first.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from textwrap import fill

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "timegym-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np

from .common import close_statistics, interval_cell, rate_cell, require_conclusion, scientific_tex, sha256


ROOT = Path(__file__).resolve().parents[1]
CHECKS = ("contract", "scalar_shift", "target_shift", "conformance", "loss")
CHECK_LABELS = ("Output\ncontract", "Scalar shift\n(public)", "Target shift\n(experimental)",
                "NumPy agreement\n(experimental)", "Calibrated MSE\n(experimental)")
FAULT_LABELS = {
    "clean": "Correct implementation",
    "reverse_prediction_lags": "Reversed prediction lags",
    "stale_recursive_state": "Stale recursive state",
    "omit_inverse_target_scaling": "Missing inverse target scaling",
    "swap_recursive_targets": "Swapped recursive targets",
}
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


def save_figure(fig, directory, name):
    for suffix in ("png", "pdf"):
        kwargs = {"dpi": 220} if suffix == "png" else {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(directory / f"{name}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def plot_ablation(results, protocol, directory):
    rows = branch_counts(results, protocol)
    values = np.asarray([[count/row["n"] for count in row["counts"]] for row in rows])
    cmap = LinearSegmentedColormap.from_list("supplement-alarm", ["#f3f7f9", "#aad0df", "#165877"])
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    heat = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    for i, row in enumerate(rows):
        for j, count in enumerate(row["counts"]):
            ax.text(j, i, f"{count}/{row['n']}", ha="center", va="center", fontsize=10.5,
                    color="white" if values[i, j] > .65 else "#172b3a")
    ax.set_xticks(range(len(CHECKS)), CHECK_LABELS)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [FAULT_LABELS[row["branch"]] for row in rows])
    ax.tick_params(length=0, pad=9)
    ax.set_xticks(np.arange(-.5, len(CHECKS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.axhline(.5, color="#536774", lw=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(heat, ax=ax, pad=.025, fraction=.025)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    colorbar.set_label("Alarm rate")
    fig.suptitle("Different specifications expose different implementation defects", x=.025, ha="left",
                 fontsize=16, fontweight="bold")
    fig.text(.025, .025, f"Counts over {len(protocol['model_study']['conditions'])} fixed conditions, "
             f"{protocol['model_study']['evaluation_cases']} independent cases each. Clean alarms are negative controls.\n"
             "All faults satisfy output contracts. Engineering agreement is not a claim of optimal forecasting accuracy.",
             fontsize=9.5, color="#425466")
    fig.subplots_adjust(left=.30, right=.96, top=.78, bottom=.16)
    save_figure(fig, directory, "supplement-implementation-ablation")


def plot_covariance(results, protocol, directory):
    spec = protocol["covariance_study"]
    conditions = [row["id"] for row in spec["conditions"]]
    fig, axes = plt.subplots(1, len(spec["variants"]), figsize=(12.3, 4.8), sharey=True)
    for ax, variant in zip(axes, spec["variants"]):
        rows = [next(row for row in results["covariance"]["condition_summaries"]
                     if row["condition"] == condition and row["variant"] == variant) for condition in conditions]
        for name, offset, marker, color in (("raw", -.12, "o", "#0072b2"), ("holm", .12, "s", "#d55e00")):
            rates = np.asarray([row[name]["rate"] for row in rows])
            lower = np.asarray([row[name]["ci95"][0] for row in rows])
            upper = np.asarray([row[name]["ci95"][1] for row in rows])
            ax.errorbar(rates, np.arange(len(rows))+offset,
                        xerr=np.maximum(0, np.asarray([rates-lower, upper-rates])), fmt=marker,
                        color=color, capsize=3, markersize=5, linestyle="none",
                        label="Raw" if name == "raw" else "Holm (4 checks)")
        ax.axvline(spec["alpha"], color="#b1b8bd", linestyle="--", linewidth=.9)
        ax.set_xlim(-.035, 1.04)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(VARIANT_LABELS[variant], loc="left", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Rejection rate (Wilson 95% interval)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6ebed", linewidth=.6)
    axes[0].set_yticks(range(len(conditions)), [fill(CONDITION_LABELS[name], width=24) for name in conditions])
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=2, frameon=False)
    fig.suptitle("Multiplicity correction trades rejection power for fewer false alarms", x=.025, ha="left",
                 fontsize=15.5, fontweight="bold")
    fig.text(.025, .025, f"{spec['independent_suites']} independent suites per variant; each suite contains the four displayed checks. Dashed line: nominal 5%.\n"
             "Every clean rejection is a false alarm. Shuffled factorized white noise is also a valid negative control.\n"
             "Other shuffled and reduced-spread cases are alternatives. These are exact conditional-reference sample checks.",
             fontsize=9.5, color="#425466")
    fig.subplots_adjust(left=.19, right=.985, top=.73, bottom=.25, wspace=.13)
    save_figure(fig, directory, "supplement-covariance-holm")


def render_tex(results, manifest):
    protocol = manifest["protocol"]
    model, covariance = protocol["model_study"], protocol["covariance_study"]
    rows = branch_counts(results, protocol)
    clean = rows[0]
    calibration_count = len(results["model"]["calibration_records"])
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
                f"only {row['cases_improved_by_repair']}/{row['n']} realized forecasts improved. "
                "Restoring the declared algorithm can therefore worsen forecast accuracy on a particular condition."
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
        r"\subsection{A separately versioned implementation supplement}",
        f"Protocol {protocol['protocol_version']} evaluates {len(model['faults'])} selected implementation defects "
        f"under {len(model['conditions'])} fixed conditions, with {model['calibration_cases']} independent clean calibration cases and "
        f"{model['evaluation_cases']} disjoint evaluation cases each ({calibration_count} calibration and "
        f"{clean['n']} evaluation cases in total). The ridge penalty is {model['ridge_alpha']:g} and "
        f"the lag window is {model['window']}. The comparison specifications and calibration assumptions "
        "are defined above; target-specific translation, NumPy agreement, and calibrated loss are "
        "experiment-only comparators. These selected fixtures are not an independent historical bug corpus.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Selected implementation ablation: alarm counts over the same fixed conditions, "
        f"{clean['n']} cases per row. These pooled counts are descriptive; full condition-specific counts "
        "follow. All three rightmost comparators are experiment-only.}",
        r"\label{tab:supplement-ablation}",
        r"\begin{tabular}{@{}lrrrrr@{}}\toprule",
        r"Implementation & Contract & Scalar & Target & NumPy & MSE \\ \midrule",
    ]
    for row in rows:
        lines.append(FAULT_LABELS[row["branch"]] + " & " + " & ".join(str(v) for v in row["counts"]) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        "The complete table retains the blind spots and false alarms of every comparison. These results "
        "concern these specifications and selected implementations, rather than the sensitivity of arbitrary "
        "metamorphic tests. In the supplement, corrected means the clean implementation refitted; "
        "the paired comparison is clean-minus-faulty, with no separate repaired arm. "
        "Any output-contract violation terminates the study.",
        f"Across the fixed condition--fault pairs, the calibrated loss alarm changed from no alarm to alarm "
        f"in {loss_gained} paired comparisons and from alarm to no alarm in {loss_lost}. "
        "The same clean case appears once per fault in these descriptive paired counts; they are not "
        "independent replicates of a pooled statistical test.",
        f"The clean implementation reduced mean realized MSE in {improved}/{len(losses)} condition--fault pairs; "
        f"{intervals_below_zero}/{len(losses)} paired 95\\% t intervals were wholly below zero. "
        "The full paired loss differences, intervals, and numbers of individually improved cases are retained "
        r"in \code{model-repair-losses.csv}. These exploratory intervals are not adjusted across comparisons. "
        "Restoring implementation agreement need not improve every noisy realized forecast.",
        *repair_counterexamples,
        r"\begin{table}[p]\centering\scriptsize",
        r"\caption{Complete condition-specific implementation alarm counts. Each entry is out of "
        f"{model['evaluation_cases']} paired cases. C, S, T, N, and L denote contract, scalar shift, target "
        "shift, NumPy agreement, and calibrated loss respectively. Clean rows are negative controls; "
        "there is no separate corrected-refit row.}",
        r"\label{tab:supplement-complete}",
        r"\begin{tabular}{@{}llrrrrr@{}}\toprule",
        r"Condition & Branch & C & S & T & N & L \\ \midrule",
    ])
    for condition in model["conditions"]:
        records = [row for row in results["model"]["records"] if row["condition"] == condition["id"]]
        for index, branch in enumerate(["clean", *model["faults"]]):
            values = [sum(row["alarms"][check] for row in records if row["branch"] == branch) for check in CHECKS]
            label = "Clean" if branch == "clean" else SHORT_FAULTS[branch]
            lines.append((CONDITION_LABELS[condition["id"]] if index == 0 else "") + " & " + label
                         + " & " + " & ".join(str(v) for v in values) + r" \\")
        lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        r"\subsection{Negative controls and multiple covariance checks}",
        f"The supplement's covariance study evaluates {covariance['independent_suites']} independent suites, each containing "
        f"four conditional-reference covariance checks with {covariance['n_samples']} predictive trajectories, "
        f"{covariance['n_calibration']} calibration ensembles, and nominal level {covariance['alpha']:g}. "
        "The fixed variants are exact-reference samples, partial dependence permutation "
        f"(strength {covariance['shuffle_strength']:g}), and spread multiplied by {covariance['spread_multiplier']:g}. "
        "Raw and Holm decisions are reported for the declared four-check families. Shuffled factorized "
        "white noise remains a negative control.",
        f"For clean suites, any false alarm occurred in {suites['clean']['raw_any_false_alarm']['count']}/"
        f"{suites['clean']['raw_any_false_alarm']['n']} raw suites versus "
        f"{suites['clean']['holm_any_false_alarm']['count']}/{suites['clean']['holm_any_false_alarm']['n']} "
        "after Holm correction. For reduced-spread suites, detection of at least one altered condition "
        f"fell from {suites['reduced_spread']['raw_any_alarm']['count']}/"
        f"{suites['reduced_spread']['raw_any_alarm']['n']} to "
        f"{suites['reduced_spread']['holm_any_alarm']['count']}/{suites['reduced_spread']['holm_any_alarm']['n']}. "
        "The correction thus reduces family-wise false alarms at a substantial power cost in this setting.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Any false alarm per four-check suite, counts and Wilson 95\% intervals (percent). Clean suites have four true nulls, partial-shuffle suites one, and reduced-spread suites none. --- denotes an undefined result, not a negative result.}",
        r"\label{tab:supplement-familywise}",
        r"\begin{tabular}{@{}lrr@{}}\toprule",
        r"Variant & Raw & Holm \\ \midrule",
    ])
    for row in results["covariance"]["suite_summaries"]:
        lines.append(VARIANT_LABELS[row["variant"]] + " & " + rate_cell(row["raw_any_false_alarm"])
                     + " & " + rate_cell(row["holm_any_false_alarm"]) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{All condition-specific covariance rejections, raw and Holm, out of "
        f"{covariance['independent_suites']} suites. Clean columns and the first shuffled row are negative "
        "controls; other entries measure power on the selected alternatives.}",
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
        "score differences are retained for every condition. Correction can reduce power; every miss is "
        "reported. The covariance null remains IID trajectory sampling from the specified exact conditional "
        "reference, not general multivariate calibration of an arbitrary fitted model.",
        r"\paragraph{Cost and reproducibility.}",
        f"The retained study design requires {results['model']['cost_counts']['sklearn_fits']:,} sklearn fits, "
        f"{results['model']['cost_counts']['independent_ridge_solves']:,} independent NumPy solves, and "
        f"{results['covariance']['cost_counts']['covariance_checks']:,} covariance checks containing "
        f"{results['covariance']['cost_counts']['calibration_ensembles']:,} calibration ensembles. "
        "Raw case records, protocol and source fingerprints, paired examples, and CSV tables are saved "
        r"under \code{experiments/results-v2/}; the original pilot artifacts remain separate.",
    ])
    return "\n\n".join(lines) + "\n"


def render_macros(results):
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
    values = {"RepairDeltaMean": f"{delta['mean']:.3f}", "RepairDeltaLow": f"{delta['ci95'][0]:.3f}",
              "RepairDeltaHigh": f"{delta['ci95'][1]:.3f}", "RepairImproved": improved,
              "CasesPerCondition": selected["n"], "CasesPerRow": len(clean)}
    return "% Generated from frozen supplement records; do not edit.\n" + "".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments/results-v2")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    path = args.results_dir / "results.json"
    results = json.loads(path.read_text())
    manifest = json.loads((args.results_dir / "manifest.json").read_text())
    validate_recorded_results(path, results, manifest)
    directory = args.paper_dir / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": False,
                         "axes.labelcolor": "#263746", "text.color": "#172b3a",
                         "axes.edgecolor": "#a9b5be", "pdf.fonttype": 42, "ps.fonttype": 42})
    plot_ablation(results, manifest["protocol"], directory)
    plot_covariance(results, manifest["protocol"], directory)
    fragment = args.paper_dir / "generated-supplement.tex"
    fragment.write_text(render_tex(results, manifest))
    (args.paper_dir / "generated-macros.tex").write_text(render_macros(results))
    print(json.dumps({"tables": str(fragment), "figures": str(directory)}))


if __name__ == "__main__":
    main()
