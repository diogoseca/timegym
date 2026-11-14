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
import numpy as np
from timegym.statistics import paired_t_interval
from timegym import Scenario
from timegym.grid import NOISE_LOW
from timegym.scorecard import verdict
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
TABLE_MODEL_LABELS = {
    "last_value": "Last value",
    "seasonal_12": "Seasonal naive (12)",
    "gaussian_ar1": "Fitted AR(1)",
}
TABLE_SCENARIO_LABELS = {
    "white_noise": "Noise around a fixed level",
    "trend": "Straight-line trend",
    "seasonal": "Repeating seasonal cycle",
    "ar1": "Mean reversion (AR(1))",
    "random_walk": "Random walk",
}
BAND_LETTERS = {"Nails it": "N", "Good": "G", "Struggles": "S", "Fails": "F"}
# Scenarios whose scores do not depend on the innovation scale: with a zero mean
# and no drift, a larger sigma only rescales the series and both errors with it.
SCALE_FREE_SCENARIOS = ("white_noise", "ar1", "random_walk")
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def save(fig, directory, name):
    for ext in ("png", "pdf"):
        options = {"dpi": 200} if ext == "png" else {"metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(directory / f"{name}.{ext}", bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)


def capability_score(unavoidable, avoidable):
    """Score out of 100 as the package scorecard defines it: 100 matches the answer key."""
    total = unavoidable + avoidable
    return 100.0 if total == 0 else 100.0 * unavoidable / total


def band_letter(score):
    """First letter of the scorecard band; the band is read from the rounded score."""
    return BAND_LETTERS[verdict(score)]


def score_cell(score):
    return f"{score:.0f} {band_letter(score)}"


def capability_unavoidable(protocol):
    """Mean conditional variance of each capability scenario's reference, by (kind, sigma).

    This is the error the answer key itself makes, read from a generated case in the
    same way as ``unavoidable_mse`` in timegym/scorecard.py. No model is fitted.
    """
    task = protocol["capability_task"]
    first = protocol["capability_seeds"][0]
    unavoidable = {}
    for kind in protocol["capability_scenarios"]:
        for sigma in protocol["capability_sigma"]:
            scenario = Scenario(kind, parameters={"sigma": sigma},
                                **{key: task[key] for key in ("context_length", "horizon", "n_targets")})
            variances = [np.asarray(scenario.generate(seed=seed).reference.variance, dtype=float)
                         for seed in (first, first + 1)]
            # The table uses one value per scenario and scale, which is valid only
            # while the conditional variance does not depend on the observed values.
            if not np.allclose(variances[0], variances[1], rtol=1e-12, atol=0):
                raise ValueError(f"Conditional variance of {kind} at sigma={sigma} depends on the seed")
            unavoidable[kind, sigma] = float(np.mean(variances[0]))
    return unavoidable


def capability_summary(results, protocol, unavoidable=None):
    if unavoidable is None:
        unavoidable = capability_unavoidable(protocol)
    summary = []
    for scenario in protocol["capability_scenarios"]:
        for model in protocol["capability_models"]:
            for sigma in protocol["capability_sigma"]:
                values = [r["mean_reference_mse"] for r in results["capability_records"]
                          if r["scenario"] == scenario and r["model"] == model and r["sigma"] == sigma]
                stat = paired_t_interval(values)
                mean = stat["mean"]
                best = unavoidable[scenario, sigma]
                summary.append({"scenario": scenario, "model": model, "sigma": sigma,
                                "mean_reference_mse": mean, "ci95_low": stat["ci95"][0],
                                "ci95_high": stat["ci95"][1], "n": len(values),
                                "unavoidable_mse": best, "score": capability_score(best, mean)})
    return summary


def capability_section(summary, protocol, followup=None):
    """Subsection text and score table for the capability sweep, checked against the records.

    ``followup`` is the exploratory follow-up record (protocol 2.1); when given, the
    explanation of the fitted AR(1) on the trend is checked against it as well.
    """
    models, scenarios = protocol["capability_models"], protocol["capability_scenarios"]
    low, high = min(protocol["capability_sigma"]), max(protocol["capability_sigma"])
    row = {(r["scenario"], r["model"], r["sigma"]): r for r in summary}

    def score(scenario, model, sigma):
        return row[scenario, model, sigma]["score"]

    def band(scenario, model, sigma):
        return verdict(score(scenario, model, sigma))

    seeds = protocol["capability_seeds"][1] - protocol["capability_seeds"][0]
    cases = {r["n"] for r in summary}
    require_conclusion(cases == {seeds}, "every capability setting has one case per evaluation seed")
    require_conclusion(all(score_cell(score(kind, model, low)) == score_cell(score(kind, model, high))
                           and np.isclose(score(kind, model, low), score(kind, model, high), rtol=1e-9, atol=0)
                           for kind in SCALE_FREE_SCENARIOS for model in models),
                       "white noise, mean reversion and the random walk score the same at both levels")
    require_conclusion(all(band(kind, "gaussian_ar1", sigma) == "Nails it"
                           for kind in ("white_noise", "ar1") for sigma in (low, high)),
                       "the fitted AR(1) nails white noise and mean reversion at both scales")
    require_conclusion(all(band("trend", "gaussian_ar1", sigma) == "Fails" for sigma in (low, high)),
                       "the fitted AR(1) fails on the trend at both scales")
    require_conclusion(score("trend", "gaussian_ar1", high) < score("trend", "gaussian_ar1", low),
                       "the fitted AR(1) gets worse with noise on the trend")
    if followup is not None:
        from .render_followup import check_followup_conclusions
        check_followup_conclusions(followup)
        fitted = {row["sigma"] for row in followup["trend_ar1"]["rows"]}
        require_conclusion(fitted == {low, high}, "the follow-up fits the AR(1) at the table's two noise levels")
    require_conclusion(all(row["random_walk", "last_value", sigma]["mean_reference_mse"] < 1e-12
                           for sigma in protocol["capability_sigma"]),
                       "the last value is the answer key for the zero-drift random walk")
    seasonal_low, seasonal_high = score("seasonal", "gaussian_ar1", low), score("seasonal", "gaussian_ar1", high)
    require_conclusion(band("seasonal", "gaussian_ar1", low) == "Fails"
                       and band("seasonal", "gaussian_ar1", high) == "Nails it",
                       "on the seasonal cycle the fitted AR(1) fails at low noise and nails it at high noise")
    require_conclusion(all(any(score_cell(score(kind, model, low)) != score_cell(score(kind, model, high))
                               for model in models) for kind in ("trend", "seasonal")),
                       "noise changes the scores of the trend and the seasonal cycle")

    def words(count):
        return NUMBER_WORDS.get(count, str(count))

    lines = [
        r"\subsection{Study 1: capability profiles}",
        f"The univariate capability sweep uses {seeds} evaluation seeds, "
        f"{words(len(protocol['capability_sigma']))} innovation scales, {words(len(scenarios))} kinds of series, "
        f"and {words(len(models))} unmodified models. Table~\\ref{{tab:capability}} gives each model's "
        "score (Equation~\\eqref{eq:score}) at the lowest and highest scale. The full three-scale sweep is in "
        "\\code{figures/capability-curves.pdf} and \\code{capability-summary.csv} of the reproducibility artifact. "
        "The fitted AR(1) nails white noise and mean reversion and fails on the trend at both scales, "
        "and gets worse with noise there: at low noise its coefficient is close to 1 and it follows the line, "
        "while at high noise the estimate falls to about \\TrendARPhiHigh{} and the forecast reverts towards "
        "the mean of the history, below the trend. "
        "For the zero-drift random walk, the last value is the answer key. "
        "Noise matters where a fixed-size pattern competes with it: on the seasonal cycle the fitted AR(1) "
        f"scores {seasonal_low:.0f} at low noise and {seasonal_high:.0f} at high noise, where the cycle is "
        "mostly buried and ignoring it costs little.",
        r"\begin{table}[htbp]\centering\small",
        f"\\caption{{Score out of 100 (100 matches the answer key) at low ($\\sigma={low:g}$) and high "
        f"($\\sigma={high:g}$) innovation noise, over {seeds} cases each. N nails it, G good, S struggles, "
        "F fails. White noise, mean reversion and the random walk score the same at both levels: with a "
        "zero mean and no drift, a larger $\\sigma$ only rescales the series. The package's scorecard adds "
        "measurement noise to tell those levels apart. The frozen Study 1 used "
        f"$\\sigma={low:g}$; the package scorecard's low level is {NOISE_LOW:g}.}}",
        r"\label{tab:capability}",
        "\n".join([
            r"\begin{tabular}{@{}l" + "rr" * len(models) + r"@{}}\toprule",
            " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{TABLE_MODEL_LABELS[model]}}}" for model in models) + r" \\",
            "".join(f"\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(len(models))),
            "Kind of series & " + " & ".join(["Low & High"] * len(models)) + r" \\ \midrule",
        ]),
    ]
    for kind in scenarios:
        cells = [score_cell(score(kind, model, sigma)) for model in models for sigma in (low, high)]
        lines.append(TABLE_SCENARIO_LABELS[kind] + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    return lines


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


def render_tex(results, manifest, summary, followup=None):
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
        r"\subsection{Study 1: fault detection and false alarms}",
        "Table~\\ref{tab:faults} reports alarms for each corruption. "
        "The six explicit output violations were detected in every evaluation case. "
        "Both well-formed ensemble mutations were rejected by the covariance check "
        "in every case.",
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
        "No threshold was changed to remove these false alarms.",
        f"The point-MSE cutoff was {results['mse_threshold']:.4f}, the empirical 95th percentile "
        f"from {protocol['development_seeds'][1]-protocol['development_seeds'][0]} development cases. "
        f"It alarmed on {clean['mse']['count']}/{clean['mse']['n']} clean cases. It detects invalid point "
        "arrays, but ignores index labels, target names, quantiles, and trajectory samples. "
        "Where a mutation leaves the point values unchanged, its alarms are identical to "
        "the clean alarms and supply no mutation-specific evidence.",
        "Disabling an output mutation restored the exact clean output, including its "
        f"sample array. Repaired outputs had zero contract violations and the same "
        f"{repair['count']}/{repair['n']} covariance alarms as the clean control.",
        r"\subsection{Study 1: marginal invariance and joint-score response}",
        "Table~\\ref{tab:dependence} varies the proportion of ensemble rows selected "
        "for coordinate-wise permutation. Every empirical marginal is preserved "
        "at every strength, as checked directly in the runner. The maximum absolute "
        f"CRPS difference across all cases and strengths was {scientific_tex(max_crps)}. "
        "Energy and variogram scores both worsened on average at full permutation; "
        "their paired intervals exclude zero for this setting. Their raw "
        "effect magnitudes have different units and must not be used to rank their sensitivity.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Mutated minus clean score, mean and paired 95\% t interval over 64 independent cases. CRPS is unchanged at every strength up to floating-point roundoff and is not tabulated. Intervals are exploratory and not multiplicity-adjusted.}",
        r"\label{tab:dependence}",
        r"\begin{tabular}{@{}rcc@{}}\toprule",
        r"Strength & Energy & Variogram \\ \midrule",
    ])
    for strength in protocol["dependence_strengths"]:
        row = results["dependence_summary"][str(float(strength))]
        lines.append(f"{strength:.2f} & {interval_cell(row['energy'])} & {interval_cell(row['variogram'])}" + r" \\")
    lines.extend([
        r"\bottomrule\end{tabular}\end{table}",
    ])
    lines.extend(capability_section(summary, protocol, followup))
    lines.extend([
        r"\subsection{Study 1: an inverse-scaling repair}",
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
        "mean remains a poor model of this growing seasonal series.",
    ])
    return "\n\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "experiments/results/results.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/results/manifest.json")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    parser.add_argument("--followup-dir", type=Path, default=None,
                        help="exploratory follow-up record (default experiments/results-v2-followup)")
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
    from .render_followup import FOLLOWUP_DIR, load_followup
    followup, _ = load_followup(args.followup_dir if args.followup_dir else FOLLOWUP_DIR)
    plot_capability(summary, protocol, directory)
    with (directory / "capability-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.paper_dir / "generated-results.tex").write_text(render_tex(results, manifest, summary, followup))
    print(json.dumps({"tables": str(args.paper_dir / "generated-results.tex"), "figures": str(directory)}))


if __name__ == "__main__":
    main()
