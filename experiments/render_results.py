"""Render the controlled pilot into portable manuscript tables and figures.

Run: python -m experiments.render_results
Reads existing results only; it does not rerun or modify the study protocol.
"""

from __future__ import annotations

import argparse
import csv
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
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
from timegym.statistics import paired_t_interval
from timegym import Scenario
from .common import interval_cell, rate_cell, require_conclusion, scientific_tex, sha256


ROOT = Path(__file__).resolve().parents[1]
FAULT_LABELS = {
    "clean": "Clean control",
    "index_shift": "Index shift",
    "target_swap": "Target-label swap",
    "horizon_truncation": "Horizon truncation",
    "nonfinite_point": "Nonfinite point",
    "quantile_crossing": "Quantile crossing",
    "sample_axis_swap": "Sample-axis transpose",
    "variance_shrink": "Variance shrinkage",
    "dependence_shuffle": "Dependence shuffle",
}
MODEL_LABELS = {
    "last_value": "Last value",
    "seasonal_12": "Seasonal naive (12)",
    "gaussian_ar1": "Fitted Gaussian AR(1)",
}
SCENARIO_LABELS = {kind: Scenario(kind).display_name
                   for kind in ("white_noise", "trend", "seasonal", "ar1", "random_walk")}
COLORS = ["#087e8b", "#c45b35", "#493b88"]


def save(fig, directory, name):
    for ext in ("png", "pdf"):
        options = {"dpi": 200} if ext == "png" else {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(directory / f"{name}.{ext}", bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)


def plot_fault_matrix(results, protocol, directory):
    names = ["clean", *protocol["mutations"]]
    columns = ["mse", "contract", "covariance", "combined"]
    values = np.array([[np.nan if results["fault_summary"][name][key] is None
                        else results["fault_summary"][name][key]["rate"] for key in columns]
                       for name in names])
    cmap = LinearSegmentedColormap.from_list("timegym", ["#f4f8fa", "#a4ced8", "#164e63"])
    cmap.set_bad("#eeeeef")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    heat = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i, name in enumerate(names):
        for j, key in enumerate(columns):
            stat = results["fault_summary"][name][key]
            if stat is None:
                ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor="none", edgecolor="#c5c5c7", hatch="///", linewidth=0))
                text = "not run"
            else:
                text = f"{stat['count']}/{stat['n']}  ·  {stat['rate']:.1%}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    color="white" if stat and stat["rate"] > .65 else "#172b3a")
    ax.set_xticks(range(4), ["Point-MSE\nalarm", "Output\ncontract", "Reference\ncovariance", "Contract or\ncovariance"])
    ax.xaxis.tick_top()
    labels = [FAULT_LABELS[name] for name in names]
    labels[0] += " (false alarms)"
    ax.set_yticks(range(len(names)), labels)
    ax.tick_params(length=0, pad=10)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.axhline(.5, color="#172b3a", linewidth=1.2)
    ax.axhline(6.5, color="#172b3a", linewidth=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(heat, ax=ax, pad=.035, fraction=.035)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    colorbar.set_label("Alarm rate")
    fig.suptitle("Selected faults require different checks", x=.04, ha="left", fontsize=17, fontweight="bold")
    fig.text(.04, .018, "Each row uses the same 64 independent cases. Hatched cells are unsupported after a contract failure.\n"
             "Point-MSE ignores labels and non-point outputs; this is not a comparison with a complete validation framework.",
             fontsize=9.5, color="#425466")
    fig.subplots_adjust(left=.30, right=.91, top=.83, bottom=.13)
    save(fig, directory, "fault-alarm-matrix")


def plot_dependence(results, protocol, directory):
    strengths = protocol["dependence_strengths"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    titles = {"crps": "Marginal CRPS", "energy": "Joint energy score", "variogram": "Joint variogram score"}
    for ax, key, color in zip(axes, titles, COLORS):
        stats = [results["dependence_summary"][str(float(x))][key] for x in strengths]
        means = np.array([x["mean"] for x in stats])
        low = np.array([x["ci95"][0] for x in stats])
        high = np.array([x["ci95"][1] for x in stats])
        ax.axhline(0, color="#77838e", linestyle="--", linewidth=1)
        ax.errorbar(strengths, means, yerr=np.array([means-low, high-means]), color=color,
                    marker="o", capsize=3, linewidth=1.8, markersize=5)
        ax.set_title(titles[key], loc="left", fontsize=12, fontweight="bold")
        ax.set_xlabel("Permutation strength")
        ax.set_xticks(strengths)
        ax.set_xlim(-.04, 1.04)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        if key == "crps":
            ax.set_ylim(-2e-16, 2e-16)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.text(.04, .93, "Numerical roundoff only", transform=ax.transAxes, fontsize=9,
                    va="top", color="#425466")
        ax.set_ylabel("Mutated minus clean score")
    fig.suptitle("Changing trajectory pairing leaves marginal scores unchanged", x=.04, ha="left",
                 fontsize=16, fontweight="bold")
    fig.text(.04, .015, "Means and paired 95% t intervals over 64 independent cases; exploratory, without multiplicity correction.\n"
             "Each panel uses its own score scale. Positive differences indicate worse scores after mutation.",
             fontsize=9.5, color="#425466")
    fig.tight_layout(rect=(.02, .12, 1, .9), w_pad=2)
    save(fig, directory, "dependence-score-deltas")


def capability_summary(results, protocol):
    summary = []
    for scenario in protocol["capability_scenarios"]:
        for model in protocol["capability_models"]:
            for sigma in protocol["capability_sigma"]:
                values = [r["mean_reference_mse"] for r in results["capability_records"]
                          if r["scenario"] == scenario and r["model"] == model and r["sigma"] == sigma]
                stat = paired_t_interval(values)
                mean = stat["mean"]
                summary.append({"scenario": scenario, "model": model, "sigma": sigma,
                                "mean_reference_mse": mean, "ci95_low": stat["ci95"][0],
                                "ci95_high": stat["ci95"][1], "n": len(values)})
    return summary


def plot_capability(summary, protocol, directory):
    fig, axes = plt.subplots(1, 5, figsize=(14, 4.4))
    markers = ["o", "s", "^"]
    styles = ["-", "--", "-."]
    for ax, scenario in zip(axes, protocol["capability_scenarios"]):
        for model, color, marker, style in zip(protocol["capability_models"], COLORS, markers, styles):
            rows = [r for r in summary if r["scenario"] == scenario and r["model"] == model]
            ax.plot([r["sigma"] for r in rows], [r["mean_reference_mse"] for r in rows],
                    color=color, marker=marker, linestyle=style, label=MODEL_LABELS[model], linewidth=1.8)
        ax.set_title(fill(SCENARIO_LABELS[scenario], width=21), loc="left", fontsize=10, fontweight="bold")
        ax.set_xticks(protocol["capability_sigma"])
        ax.set_xlabel(r"Innovation scale $\sigma$")
        ax.set_ylim(bottom=-.02 * ax.get_ylim()[1])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Mean-reference MSE")
    fig.suptitle("Capability depends on the mechanism and noise level", x=.04, ha="left",
                 fontsize=16, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .08), ncol=3, frameon=False)
    fig.text(.04, .015, "Means over 16 independent cases per setting; each panel has its own vertical scale. "
             "These descriptive scores do not classify software faults.", fontsize=9.5, color="#425466")
    fig.tight_layout(rect=(.02, .19, 1, .89), w_pad=2)
    save(fig, directory, "capability-curves")


def render_tex(results, manifest, summary):
    protocol = manifest["protocol"]
    fault = results["fault_summary"]
    clean = fault["clean"]
    full = results["dependence_summary"]["1.0"]
    inverse = results["inverse_repair"]
    max_crps = max(abs(r["crps_delta"]) for r in results["dependence_records"])
    contract_mutations = {"index_shift", "target_swap", "horizon_truncation", "nonfinite_point",
                          "quantile_crossing", "sample_axis_swap"}
    require_conclusion(contract_mutations <= set(protocol["mutations"]) and all(
        fault[name]["contract"]["count"] == fault[name]["contract"]["n"] > 0
        for name in contract_mutations), "all six contract mutations alarm on every case")
    require_conclusion(all(fault[name]["covariance"]["count"] == fault[name]["covariance"]["n"] > 0
                           for name in ("variance_shrink", "dependence_shuffle")),
                       "both ensemble mutations alarm on every case")
    require_conclusion(all(full[name]["ci95"][0] > 0 for name in ("energy", "variogram")),
                       "full permutation worsens energy and variogram with intervals above zero")
    alpha, rate = protocol["alpha_per_check"], clean["covariance"]
    require_conclusion(rate["rate"] > alpha and rate["ci95"][0] <= alpha <= rate["ci95"][1],
                       "clean covariance rate exceeds nominal alpha and its interval includes alpha")
    require_conclusion(all(row["faulty_status"] == "FAIL" and row["repaired_status"] == "PASS"
                           and row["repaired_mae"] < row["faulty_mae"] for row in inverse["real"])
                       and len(inverse["real"]) == 4, "inverse repair passes and improves MAE at all four real origins")
    repair = fault[protocol["mutations"][0]]["repair_combined"]
    require_conclusion(all(row["repair_combined"] == repair for row in fault.values())
                       and repair == clean["covariance"]
                       and all(not row["repair_contract_alarm"] for row in results["fault_records"]),
                       "repaired outputs retain the clean covariance alarms without contract violations")
    require_conclusion(max_crps < 1e-12, "marginal CRPS changes are numerical roundoff")

    lines = [
        "% Generated by experiments/render_results.py; do not edit numerical tables by hand.",
        r"\subsection{Selected fault detection and false alarms}",
        "Table~\\ref{tab:faults} reports alarms for each selected corruption. "
        "The six explicit output violations were detected in every evaluation case. "
        "Both well-formed ensemble mutations were rejected by the covariance check "
        "in every case. These counts describe deliberately selected fixtures; "
        "they are not an estimate of detection across naturally occurring bugs.",
        r"\begin{table}[htbp]\centering\scriptsize",
        r"\caption{Alarm counts and Wilson 95\% intervals (percent). Covariance is not run on malformed outputs; --- denotes an undefined result, not a negative result.}",
        r"\label{tab:faults}",
        r"\begin{tabular}{@{}lccc@{}}\toprule",
        r"Case & Point-MSE alarm & Output contract & Covariance \\ \midrule",
    ]
    for name in ["clean", *protocol["mutations"]]:
        lines.append(" & ".join([FAULT_LABELS[name], *(rate_cell(fault[name][key]) for key in ("mse", "contract", "covariance"))]) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        f"The clean covariance check rejected {clean['covariance']['count']}/{clean['covariance']['n']} cases "
        f"({100*clean['covariance']['rate']:.1f}\\%; Wilson interval "
        f"[{100*clean['covariance']['ci95'][0]:.1f}, {100*clean['covariance']['ci95'][1]:.1f}]\\%). "
        "The observed rate is above the nominal 5\\% per-check level, with an interval that includes 5\\%. "
        "No threshold was changed to remove these false alarms. The selected run does not "
        "establish precise calibration or a family-wise error rate.",
        f"The point-MSE cutoff was {results['mse_threshold']:.4f}, the empirical 95th percentile "
        f"from {protocol['development_seeds'][1]-protocol['development_seeds'][0]} development cases. "
        f"It alarmed on {clean['mse']['count']}/{clean['mse']['n']} clean cases. It detects invalid point "
        "arrays, but ignores index labels, target names, quantiles, and trajectory samples. "
        "Where a mutation leaves the point values unchanged, its alarms are identical to "
        "the clean alarms and supply no mutation-specific evidence.",
        "Disabling an output mutation restored the exact clean output, including its "
        f"sample array. Repaired outputs had zero contract violations and the same "
        f"{repair['count']}/{repair['n']} covariance alarms as the clean control. "
        "These paired rechecks confirm removal of the selected corruption; they do "
        "not imply that every repaired case receives a statistical pass or that "
        "the framework discovered the repair.",
        r"\subsection{Marginal invariance and joint-score response}",
        "Table~\\ref{tab:dependence} varies the proportion of ensemble rows selected "
        "for coordinate-wise permutation. Every empirical marginal is preserved "
        "at every strength, as checked directly in the runner. The maximum absolute "
        f"CRPS difference across all cases and strengths was {scientific_tex(max_crps)}. "
        "Energy and variogram scores both worsened on average at full permutation; "
        "their paired intervals exclude zero for this selected setting. Their raw "
        "effect magnitudes have different units and must not be used to rank their sensitivity.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Mutated minus clean score, mean and paired 95\% t interval over 64 independent cases. CRPS differences are floating-point roundoff. Intervals are exploratory and not multiplicity-adjusted.}",
        r"\label{tab:dependence}",
        r"\begin{tabular}{@{}rccc@{}}\toprule",
        r"Strength & CRPS mean & Energy & Variogram \\ \midrule",
    ])
    for strength in protocol["dependence_strengths"]:
        row = results["dependence_summary"][str(float(strength))]
        lines.append(f"{strength:.2f} & {scientific_tex(row['crps']['mean'])} & {interval_cell(row['energy'])} & {interval_cell(row['variogram'])}" + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        r"\subsection{Capability profiles}",
        "The univariate capability sweep uses 16 evaluation seeds, three innovation "
        "scales, five mechanisms, and three unmodified models. Table~\\ref{tab:capability} "
        "shows the middle scale; the full three-scale sweep is in \\code{figures/capability-curves.pdf} and \\code{capability-summary.csv} of the reproducibility artifact. "
        "The fitted AR(1) has small mean-reference error on white noise and AR(1), "
        "but is not the strongest of these models on every scenario. For the "
        "zero-drift random walk, last-value forecasting is the exact conditional "
        "mean. These results illustrate mechanism-specific capability, not "
        "implementation-fault decisions or a general model ranking.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Mean distance from the conditional mean (MSE), at innovation scale $\sigma=1$, over 16 cases. These scores have no pass/fail threshold.}",
        r"\label{tab:capability}",
        r"\begin{tabular}{@{}p{0.34\linewidth}rrr@{}}\toprule",
        r"Scenario & Last value & Seasonal naive (12) & Fitted AR(1) \\ \midrule",
    ])
    for scenario in protocol["capability_scenarios"]:
        values = [next(r["mean_reference_mse"] for r in summary if r["scenario"] == scenario and r["sigma"] == 1.0 and r["model"] == model)
                  for model in protocol["capability_models"]]
        lines.append(SCENARIO_LABELS[scenario] + " & " + " & ".join(f"{v:.4f}" for v in values) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        r"\subsection{An implemented inverse-scaling repair}",
        "A separate predictor standardizes its training context and forecasts the "
        "mean in standardized units. The fault omits the inverse transform; the "
        "repair applies the training mean and scale. The historical-mean predictor "
        "is shift equivariant, so this particular metamorphic relation is a valid "
        "declared contract. On white-noise cases with mean 5 and innovation scale 2, "
        f"the relation failed for {inverse['synthetic_fault_detection']['count']}/{inverse['synthetic_fault_detection']['n']} "
        f"faulty fits (Wilson 95\\% interval "
        f"[{100*inverse['synthetic_fault_detection']['ci95'][0]:.1f}, {100*inverse['synthetic_fault_detection']['ci95'][1]:.1f}]\\%) "
        f"and {inverse['synthetic_repair_failure']['count']}/{inverse['synthetic_repair_failure']['n']} repaired fits "
        f"([{100*inverse['synthetic_repair_failure']['ci95'][0]:.1f}, {100*inverse['synthetic_repair_failure']['ci95'][1]:.1f}]\\%). "
        "The paired repaired-minus-faulty MAE difference was "
        f"{interval_cell(inverse['synthetic_paired_mae_delta'])} (95\\% t interval).",
        "We then apply the same known omission and repair to AirPassengers, the "
        "monthly 1949--1960 international-airline passenger series in thousands~\\cite{rAirPassengers}. "
        "All scaling statistics are fitted on each training prefix. The four "
        "ordered origins each forecast the next 12 months; their results are "
        "reported separately because they are not independent series replicates.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{AirPassengers inverse-scaling illustration. MAE is in thousands of passengers. The declared shift relation fails before and passes after repair at all four origins.}",
        r"\label{tab:real}",
        r"\begin{tabular}{@{}lrrr@{}}\toprule",
        r"Last training month & Faulty MAE & Repaired MAE & Difference \\ \midrule",
    ])
    for row in inverse["real"]:
        lines.append(f"{row['origin']} & {row['faulty_mae']:.2f} & {row['repaired_mae']:.2f} & {row['repaired_mae']-row['faulty_mae']:+.2f}" + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
        "Repair improves MAE at every displayed origin, but the repaired historical "
        "mean remains a poor model of this growing seasonal series. This is one "
        "known preprocessing defect on one dataset. It does not establish that "
        "synthetic diagnosis generally transfers to real forecasting tasks.",
        r"\paragraph{Recorded environment.}",
        f"The run used Python {manifest['python']}, NumPy {manifest['versions']['numpy']}, "
        f"pandas {manifest['versions']['pandas']}, and SciPy {manifest['versions']['scipy']}. "
        "The protocol, raw case records, paired score records, and source/data "
        "SHA-256 fingerprints are retained with the experiment artifacts. "
        "All tables in this section are generated from the saved JSON results.",
    ])
    return "\n\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "experiments/results/results.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/results/manifest.json")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    manifest = json.loads(args.manifest.read_text())
    if sha256(args.results) != manifest["results_sha256"]:
        raise ValueError("Recorded pilot results do not match their manifest fingerprint")
    protocol = manifest["protocol"]
    directory = args.paper_dir / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "#263746", "text.color": "#172b3a",
                         "axes.edgecolor": "#a9b5be", "pdf.fonttype": 42, "ps.fonttype": 42})
    summary = capability_summary(results, protocol)
    plot_fault_matrix(results, protocol, directory)
    plot_dependence(results, protocol, directory)
    plot_capability(summary, protocol, directory)
    with (directory / "capability-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.paper_dir / "generated-results.tex").write_text(render_tex(results, manifest, summary))
    print(json.dumps({"tables": str(args.paper_dir / "generated-results.tex"), "figures": str(directory)}))


if __name__ == "__main__":
    main()
