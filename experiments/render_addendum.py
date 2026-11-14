"""Read and check the second exploratory Study 2 record (protocol 2.2) for the paper.

The addendum was chosen after the protocol 2.0 results and the protocol 2.1
follow-up were known; the paper marks every number from it as exploratory.
This module never reruns anything: it validates the committed record against
its manifest and the frozen inputs, checks the statements the manuscript makes
from it, and turns it into LaTeX macros. ``experiments.render_supplement``
calls it.
"""
from __future__ import annotations

import json
from pathlib import Path

from .common import require_conclusion, sha256

ROOT = Path(__file__).resolve().parents[1]
ADDENDUM_DIR = ROOT / "experiments/results-v2-addendum"
PROTOCOL = ROOT / "experiments/protocol-v2-addendum.json"
UNCHANGED_BY_REVERSED_LAGS = ("white_noise_locations", "opposing_trends")
CLOSER_AFTER_STALE_STATE = "random_walk_drifts"


def validate_addendum(results_path, results, manifest, root=ROOT):
    """Raise unless the record, its manifest, its protocol and the frozen inputs agree."""
    if sha256(results_path) != manifest["results_sha256"]:
        raise ValueError("Addendum results do not match their manifest fingerprint")
    if results["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("Addendum protocol fingerprints disagree")
    if results["protocol_version"] != "2.2" or manifest["protocol"]["protocol_version"] != "2.2":
        raise ValueError("Expected the exploratory addendum protocol 2.2")
    if results.get("status") != "exploratory" or manifest["protocol"].get("status") != "exploratory":
        raise ValueError("The addendum record must be marked exploratory")
    protocol_path = root / "experiments/protocol-v2-addendum.json"
    if sha256(protocol_path) != manifest["protocol_sha256"] or json.loads(protocol_path.read_text()) != manifest["protocol"]:
        raise ValueError("Addendum protocol does not match the recorded manifest")
    for path, digest in manifest["frozen_inputs_sha256"].items():
        if sha256(root / path) != digest:
            raise ValueError(f"Frozen input {path} changed after the addendum was recorded")


def load_addendum(directory=ADDENDUM_DIR, root=ROOT):
    directory = Path(directory)
    path = directory / "results.json"
    results = json.loads(path.read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    validate_addendum(path, results, manifest, root=root)
    return results, manifest


def variant_summaries(addendum):
    return {row["variant"]: row for row in addendum["suite_multiplicity"]["variants"]}


def full_paired_analysis(addendum):
    """The paired answer-key analysis on every evaluation seed (the one the paper cites)."""
    analyses = addendum["paired_answer_key"]["analyses"]
    return max(analyses, key=lambda analysis: analysis["evaluation_seeds"])


def check_addendum_conclusions(addendum):
    """Check every statement the manuscript makes from the addendum record."""
    multiplicity = addendum["suite_multiplicity"]
    variants = variant_summaries(addendum)
    require_conclusion(multiplicity["frozen_decisions_checked"]
                       == multiplicity["suites"] * multiplicity["family_size"] * len(variants),
                       "the recomputed raw and Holm decisions match every frozen covariance decision")
    clean, spread = variants["clean"], variants["reduced_spread"]
    require_conclusion(len(clean["true_null_conditions"]) == multiplicity["family_size"]
                       and not spread["true_null_conditions"],
                       "every clean check is a true null and no reduced-spread check is")
    detected = {method: stat["count"] for method, stat in spread["suites_with_any_rejection"].items()}
    false = {method: stat["count"] for method, stat in clean["suites_rejecting_a_true_null"].items()}
    require_conclusion(detected["fisher"] > detected["holm"],
                       "Fisher's combination detects the shrunken spread in more suites than Holm")
    require_conclusion(detected["holm"] <= detected["hochberg"] <= detected["simes"] <= detected["fisher"],
                       "in reduced-spread suites detected, Holm <= Hochberg <= Simes <= Fisher")
    require_conclusion(detected["fisher"] == detected["raw"],
                       "a single global test detects the shrunken spread as often as any raw rejection")
    require_conclusion(false["holm"] <= false["fisher"] < false["raw"],
                       "Fisher's combination has more clean false alarms than Holm and fewer than no correction")
    low, high = clean["suites_rejecting_a_true_null"]["fisher"]["ci95"]
    require_conclusion(low <= multiplicity["alpha"] <= high,
                       "the interval of Fisher's clean false-alarm rate contains the nominal level")
    paired = full_paired_analysis(addendum)
    require_conclusion(paired["evaluation_seeds"] == max(pair["n"] for pair in paired["pairs"]),
                       "the cited paired check uses every evaluation seed")
    unflagged = {tuple(pair) for pair in paired["not_flagged"]}
    expected = {(condition, "reverse_prediction_lags") for condition in UNCHANGED_BY_REVERSED_LAGS}
    expected.add((CLOSER_AFTER_STALE_STATE, "stale_recursive_state"))
    require_conclusion(unflagged == expected and len(expected) == 3,
                       "the three pairs not flagged are reversed lags on white noise and opposing trends "
                       "and the stale state on the random walk with drift")
    pairs = {(pair["condition"], pair["fault"]): pair for pair in paired["pairs"]}
    require_conclusion(all(abs(pairs[condition, "reverse_prediction_lags"]["mean_difference"])
                           < 0.2 * pairs[condition, "reverse_prediction_lags"]["clean_mean_reference_mse"]
                           for condition in UNCHANGED_BY_REVERSED_LAGS),
                       "reversing the lags leaves the distance from the answer key almost unchanged on white "
                       "noise and opposing trends")
    require_conclusion(pairs[CLOSER_AFTER_STALE_STATE, "stale_recursive_state"]["mean_difference"] < 0,
                       "the stale state on the random walk with drift is closer to the answer key")
    require_conclusion(all(pair["mean_difference"] > 0 for pair in paired["pairs"] if pair["holm_worse"]),
                       "every flagged pair is further from the answer key on average")


def percent(value):
    return f"{100 * value:.0f}"


def addendum_macros(addendum):
    """Macro name to value for every number the prose takes from the addendum record."""
    check_addendum_conclusions(addendum)
    variants = variant_summaries(addendum)
    clean = variants["clean"]["suites_rejecting_a_true_null"]
    spread = variants["reduced_spread"]["suites_with_any_rejection"]
    paired = full_paired_analysis(addendum)
    return {
        "FisherSpreadSuites": spread["fisher"]["count"],
        "HolmSpreadSuites": spread["holm"]["count"],
        "FisherCleanSuites": clean["fisher"]["count"],
        "HolmCleanSuites": clean["holm"]["count"],
        "HochbergSpreadSuites": spread["hochberg"]["count"],
        "SimesSpreadSuites": spread["simes"]["count"],
        "HochbergCleanSuites": clean["hochberg"]["count"],
        "SimesCleanSuites": clean["simes"]["count"],
        "SuitesPerVariant": addendum["suite_multiplicity"]["suites"],
        "FisherCleanLow": percent(clean["fisher"]["ci95"][0]),
        "FisherCleanHigh": percent(clean["fisher"]["ci95"][1]),
        "SuiteAlphaPercent": percent(addendum["suite_multiplicity"]["alpha"]),
        "PairedFlagged": paired["flagged"],
        "PairedPairs": paired["n_pairs"],
        "PairedFlips": f"{addendum['paired_answer_key']['flips']:,}",
    }
