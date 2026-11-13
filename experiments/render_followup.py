"""Read and check the exploratory Study 2 follow-up record (protocol 2.1) for the paper.

The follow-up was chosen after the protocol 2.0 results were known; the paper
marks every number from it as exploratory. This module never reruns anything:
it validates the committed record against its manifest and the frozen inputs,
and turns it into LaTeX macros and table cells. ``experiments.render_supplement``
and ``experiments.render_results`` call it.
"""
from __future__ import annotations

import json
from pathlib import Path

from timegym.statistics import wilson_interval

from .common import require_conclusion, sha256

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP_DIR = ROOT / "experiments/results-v2-followup"
PROTOCOL = ROOT / "experiments/protocol-v2-followup.json"
NOISE_ROWS = ("noise", "level")
RANDOM_WALK_ROWS = ("random_walk", "random_walk_drift")


def validate_followup(results_path, results, manifest, root=ROOT):
    """Raise unless the record, its manifest, its protocol and the frozen inputs agree."""
    if sha256(results_path) != manifest["results_sha256"]:
        raise ValueError("Follow-up results do not match their manifest fingerprint")
    if results["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("Follow-up protocol fingerprints disagree")
    if results["protocol_version"] != "2.1" or manifest["protocol"]["protocol_version"] != "2.1":
        raise ValueError("Expected the exploratory follow-up protocol 2.1")
    if results.get("status") != "exploratory" or manifest["protocol"].get("status") != "exploratory":
        raise ValueError("The follow-up record must be marked exploratory")
    protocol_path = root / "experiments/protocol-v2-followup.json"
    if sha256(protocol_path) != manifest["protocol_sha256"] or json.loads(protocol_path.read_text()) != manifest["protocol"]:
        raise ValueError("Follow-up protocol does not match the recorded manifest")
    for path, digest in manifest["frozen_inputs_sha256"].items():
        if sha256(root / path) != digest:
            raise ValueError(f"Frozen input {path} changed after the follow-up was recorded")


def load_followup(directory=FOLLOWUP_DIR, root=ROOT):
    directory = Path(directory)
    path = directory / "results.json"
    results = json.loads(path.read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    validate_followup(path, results, manifest, root=root)
    return results, manifest


def answer_key_cells(followup):
    """Answer-key alarm count for each (condition, branch) of Study 2."""
    return {(row["condition"], row["branch"]): row["answer_key_alarms"] for row in followup["answer_key_alarm"]["rows"]}


def check_followup_conclusions(followup):
    """Check every statement the manuscript makes from the follow-up record."""
    alarm = followup["answer_key_alarm"]
    totals = alarm["totals"]
    subtle, clean = totals["subtle_faults"], totals["clean"]
    rows = {(row["condition"], row["branch"]): row for row in alarm["rows"]}
    faults = subtle["faults"]
    require_conclusion(alarm["largest_calibration_difference"] < 1e-9,
                       "the re-run calibration fits reproduce the frozen realized MSEs")
    require_conclusion(subtle["answer_key_alarms"] > subtle["loss_alarms"],
                       "the answer-key alarm catches more subtle-fault cases than the loss alarm")
    require_conclusion(round(subtle["answer_key_alarms"] / subtle["loss_alarms"]) == 4,
                       "the answer-key alarm catches about four times as many subtle-fault cases")
    require_conclusion(clean["answer_key_alarms"] < clean["loss_alarms"],
                       "the answer-key alarm has fewer clean false alarms than the loss alarm")
    low, high = wilson_interval([True] * clean["loss_alarms"] + [False] * (clean["n"] - clean["loss_alarms"]))["ci95"]
    require_conclusion(low <= subtle["loss_alarms"] / subtle["n"] <= high,
                       "outside the seasonal condition the loss alarm on the subtle faults stays at its clean false-alarm rate")
    require_conclusion(all(rows["seasonal_scales", fault]["loss_alarms"] == rows["seasonal_scales", fault]["n"]
                           for fault in faults), "the loss alarm catches both subtle faults on the seasonal condition")
    require_conclusion(all(rows[condition, fault]["answer_key_alarms"] > rows[condition, fault]["loss_alarms"]
                           for condition in ("ar1_positive", "ar1_negative") for fault in faults),
                       "the answer-key alarm helps on the autoregressive conditions")
    require_conclusion(all(sum(rows[condition, fault]["answer_key_alarms"] for fault in faults)
                           <= sum(rows[condition, fault]["n"] for fault in faults) // 8
                           for condition in ("opposing_trends", "random_walk_drifts")),
                       "the answer-key alarm helps little on trends and random walks")
    repair = followup["repair_mechanism"]
    sums = repair["own_lag_coefficient_sum"]["mean"]
    require_conclusion(all(value < 1 for value in sums), "the fitted own-lag coefficients sum below one on average")
    steps = repair["per_step_squared_distance"]
    require_conclusion(steps["clean"][-1] > steps[repair["fault"]][-1],
                       "the correct recursion ends farther from the answer key than the stale branch")
    delta = repair["frozen_clean_minus_faulty_reference_mse"]
    require_conclusion(delta["ci95"][0] > 0 and repair["faulty_closer_to_answer_key"] > repair["n"] / 2,
                       "the faulty branch is closer to the answer key")
    precision = followup["scorecard_precision"]
    require_conclusion(all(abs(cell["score_first_10"] - cell["figure_score"]) < 1e-9 for cell in precision["cells"]),
                       "the first ten instances reproduce the Figure 1 scores")
    largest = max(precision["cells"], key=lambda cell: cell["bootstrap_sd"])
    require_conclusion(largest["pattern"] in RANDOM_WALK_ROWS, "the largest score spread is on a random-walk row")
    noise_sd = [cell["bootstrap_sd"] for cell in precision["cells"] if cell["pattern"] in NOISE_ROWS]
    require_conclusion(round(max(noise_sd)) == round(min(noise_sd)), "the noise rows share one rounded score spread")
    noise_scores = {round(cell["figure_score"]) for cell in precision["cells"] if cell["pattern"] in NOISE_ROWS}
    require_conclusion(len(noise_scores) == 1, "both noise rows score the same at both noise levels in Figure 1")
    require_conclusion(all(round(cell["figure_score"]) < 90 for cell in precision["cells"]),
                       "the Figure 1 pipeline nails no row")
    trend = {row["sigma"]: row for row in followup["trend_ar1"]["rows"]}
    low_sigma, high_sigma = min(trend), max(trend)
    require_conclusion(trend[low_sigma]["mean_phi"] > .99, "at low noise the fitted AR(1) coefficient is close to one")
    require_conclusion(trend[high_sigma]["mean_phi"] < trend[low_sigma]["mean_phi"]
                       and trend[high_sigma]["mean_forecast_minus_answer_key"] < 0,
                       "at high noise the fitted AR(1) reverts below the trend")


def followup_macros(followup):
    """Macro name to value for every number the prose takes from the follow-up record."""
    check_followup_conclusions(followup)
    totals = followup["answer_key_alarm"]["totals"]
    repair = followup["repair_mechanism"]
    delta = repair["frozen_clean_minus_faulty_reference_mse"]
    sums = repair["own_lag_coefficient_sum"]["mean"]
    precision = followup["scorecard_precision"]
    noise_sd = max(cell["bootstrap_sd"] for cell in precision["cells"] if cell["pattern"] in NOISE_ROWS)
    noise_score = next(cell["figure_score"] for cell in precision["cells"] if cell["pattern"] == NOISE_ROWS[0])
    trend = {row["sigma"]: row for row in followup["trend_ar1"]["rows"]}
    return {
        "AnswerKeySubtleAlarms": totals["subtle_faults"]["answer_key_alarms"],
        "LossSubtleAlarms": totals["subtle_faults"]["loss_alarms"],
        "SubtleCases": totals["subtle_faults"]["n"],
        "AnswerKeyCleanAlarms": totals["clean"]["answer_key_alarms"],
        "LossCleanAlarms": totals["clean"]["loss_alarms"],
        "CleanNonSeasonalCases": totals["clean"]["n"],
        "OwnLagSumFirst": f"{sums[0]:.2f}", "OwnLagSumSecond": f"{sums[1]:.2f}",
        "ReferenceDeltaMean": f"{delta['mean']:.3f}", "ReferenceDeltaLow": f"{delta['ci95'][0]:.3f}",
        "ReferenceDeltaHigh": f"{delta['ci95'][1]:.3f}",
        "FaultyCloserCases": repair["faulty_closer_to_answer_key"],
        "ScorecardNoiseSD": f"{noise_sd:.0f}", "ScorecardMaxSD": f"{precision['largest_sd']:.0f}",
        "ScorecardNoiseScore": f"{noise_score:.0f}", "ScorecardNoiseLoss": f"{100 - round(noise_score)}",
        "TrendARPhiHigh": f"{trend[max(trend)]['mean_phi']:.2f}",
    }
