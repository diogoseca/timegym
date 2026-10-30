"""Publication vector scenes built from the two frozen TimeGym studies.

Only the illustrative first case is regenerated; all aggregate points and
intervals come from the saved records, with internal consistency checks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from vector_canvas import Canvas


NAVY = "#18314b"
BLUE = "#0000ff"
ORANGE = "#c65c16"
GRAY = "#808080"
MUTED = "#526577"
GRID = "#e1e7ec"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(root, names):
    return {name: _digest(root / name) for name in names}


def _interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "ci95": [mean - half, mean + half], "n": len(values)}


def _verify_summary(actual, saved):
    if actual["n"] != saved["n"]:
        raise ValueError("Saved summary has a different case count")
    np.testing.assert_allclose([actual["mean"], *actual["ci95"]],
                               [saved["mean"], *saved["ci95"]], rtol=1e-12, atol=1e-30)


def _axis(canvas, box, xlim, ylim, xticks, yticks, *, xfmt=None, yfmt=None):
    x, y, w, h = box
    sx = lambda value: x + w * (value - xlim[0]) / (xlim[1] - xlim[0])
    sy = lambda value: y + h - h * (value - ylim[0]) / (ylim[1] - ylim[0])
    xfmt = xfmt or (lambda value: f"{value:g}")
    yfmt = yfmt or (lambda value: f"{value:g}")
    for value in yticks:
        canvas.line(x, sy(value), x + w, sy(value), color=GRID, width=.4)
        canvas.text(x - 5, sy(value), yfmt(value), size=7, color=MUTED, align="right")
    canvas.line(x, y, x, y + h, color=GRAY, width=.5)
    canvas.line(x, y + h, x + w, y + h, color=GRAY, width=.5)
    for value in xticks:
        canvas.line(sx(value), y + h, sx(value), y + h + 2, color=GRAY, width=.5)
        canvas.text(sx(value), y + h + 10, xfmt(value), size=7, color=MUTED, align="center")
    return sx, sy


def _errorbar(canvas, x, y, low, high, color=BLUE):
    canvas.line(x, low, x, high, color=color, width=.85)
    canvas.line(x - 2, low, x + 2, low, color=color, width=.85)
    canvas.line(x - 2, high, x + 2, high, color=color, width=.85)
    canvas.circle(x, y, r=1.8, color=color)


def build_dependence(root: Path):
    """Joint-sample mechanism and paired score response, protocol 1.0."""
    from timegym import Scenario
    from timegym.metrics import crps, energy_score, variogram_score
    from timegym.mutations import mutate_forecast

    root = Path(root)
    protocol_path = root / "experiments/protocol.json"
    results_path = root / "experiments/results/results.json"
    manifest = _json(root / "experiments/results/manifest.json")
    protocol, results = _json(protocol_path), _json(results_path)
    if _digest(protocol_path) != manifest["protocol_sha256"] or protocol != manifest["protocol"]:
        raise ValueError("Pilot protocol does not match its recorded manifest")
    # The supplement records the digest of the original pilot it extends.
    supplement_manifest = _json(root / "experiments/results-v2/manifest.json")
    if _digest(results_path) != supplement_manifest["original_pilot_results_sha256"]:
        raise ValueError("Pilot results no longer match the supplement's recorded digest")

    seed = protocol["evaluation_seeds"][0]
    sample_seed = seed + protocol["seed_offsets"]["reference_samples"]
    mutation_seed = seed + protocol["seed_offsets"]["mutation"]
    case = Scenario(**protocol["scenario"]).generate(seed=seed)
    clean = case.reference.forecast(n_samples=protocol["n_samples"], seed=sample_seed)
    shuffled = mutate_forecast(clean, "dependence_shuffle", seed=mutation_seed, strength=1.0)
    np.testing.assert_array_equal(np.sort(clean.samples, axis=0), np.sort(shuffled.samples, axis=0))
    observed = case.observed.to_numpy()
    score_functions = {"crps": crps, "energy": energy_score, "variogram": variogram_score}
    verified_case_scores = []
    for strength in protocol["dependence_strengths"]:
        forecast = mutate_forecast(clean, "dependence_shuffle", seed=mutation_seed, strength=strength)
        saved = next(row for row in results["dependence_records"]
                     if row["seed"] == seed and row["strength"] == strength)
        scores = {key: fn(observed, forecast.samples) for key, fn in score_functions.items()}
        for key, value in scores.items():
            np.testing.assert_allclose(value, saved[key], rtol=1e-12, atol=1e-15)
        verified_case_scores.append({"strength": strength, **scores})

    summaries = {}
    for strength in protocol["dependence_strengths"]:
        rows = [row for row in results["dependence_records"] if row["strength"] == strength]
        summaries[str(strength)] = {}
        for score in score_functions:
            summary = _interval([row[f"{score}_delta"] for row in rows])
            _verify_summary(summary, results["dependence_summary"][str(strength)][score])
            summaries[str(strength)][score] = summary

    canvas = Canvas(width=448, height=376)
    canvas.text(12, 12, "Dependence changes while every marginal stays identical", size=10, weight="bold")
    canvas.text(12, 28, "128 joint samples from one predefined AR(1) case; independent coordinate-wise permutations.",
                size=7.5, color=MUTED)

    pair = clean.samples[:, 0, :]
    mutated_pair = shuffled.samples[:, 0, :]
    correlations = [float(np.corrcoef(points.T)[0, 1]) for points in (pair, mutated_pair)]
    bound = float(np.ceil(np.max(np.abs(np.vstack((pair, mutated_pair))))))
    ticks = [-bound, 0, bound]
    scatter_boxes = [(32, 73, 105, 84), (179, 73, 105, 84)]
    for box, points, label, color, correlation in zip(
            scatter_boxes, (pair, mutated_pair), ("Reference samples", "Shuffled samples"),
            (BLUE, ORANGE), correlations):
        x, y, w, h = box
        canvas.text(x + w / 2, 46, label, size=8, weight="bold", align="center")
        canvas.text(x + w / 2, 60, f"First-horizon correlation: {correlation:+.2f}",
                    size=7, color=MUTED, align="center")
        sx, sy = _axis(canvas, box, (-bound, bound), (-bound, bound), ticks, ticks)
        for px, py in points:
            canvas.circle(sx(px), sy(py), r=.9, color=color)
        canvas.text(x + w / 2, 183, "Target 1 (x), target 2 (y)", size=7, color=MUTED, align="center")

    canvas.text(379, 46, "Marginals overlap", size=8, weight="bold", align="center")
    canvas.text(379, 60, "Target 1 empirical CDF", size=7, color=MUTED, align="center")
    sx, sy = _axis(canvas, (326, 73, 106, 84), (-bound, bound), (0, 1), ticks, [0, .5, 1])
    values = np.sort(pair[:, 0])
    steps = [(sx(-bound), sy(0))]
    for i, value in enumerate(values):
        steps.extend([(sx(value), sy(i / len(values))), (sx(value), sy((i + 1) / len(values)))])
    steps.append((sx(bound), sy(1)))
    canvas.polyline(steps, color=BLUE, width=1.15)
    canvas.polyline(steps, color=ORANGE, width=.75, dash=True)
    canvas.text(379, 183, "Identical sorted values", size=7, color=MUTED, align="center")

    canvas.line(93, 198, 108, 198, color=BLUE, width=1.2)
    canvas.text(113, 198, "Reference", size=7, color=MUTED)
    canvas.line(183, 198, 198, 198, color=ORANGE, width=.8, dash=True)
    canvas.text(203, 198, "Shuffled", size=7, color=MUTED)
    canvas.text(12, 214, "All 16 horizon–target marginal sample sets are exactly unchanged.",
                size=7.5, weight="bold")
    canvas.line(12, 225, 436, 225, color=GRID, width=.6)

    panels = [
        ("crps", "Marginal CRPS (×10⁻¹⁶)", 1e16, (-1, 1), [-1, 0, 1]),
        ("energy", "Joint energy score", 1., (-.008, .112), [0, .05, .10]),
        ("variogram", "Joint variogram score", 1., (-.004, .055), [0, .025, .05]),
    ]
    for x, (score, title, scale, limits, yticks) in zip((32, 179, 326), panels):
        canvas.text(x + 52.5, 240, title, size=7.7, weight="bold", align="center")
        sx, sy = _axis(canvas, (x, 256, 105, 70), (0, 1), limits, [0, .5, 1], yticks)
        canvas.line(x, sy(0), x + 105, sy(0), color=GRAY, width=.6, dash=True)
        for strength in protocol["dependence_strengths"]:
            summary = summaries[str(strength)][score]
            _errorbar(canvas, sx(strength), sy(summary["mean"] * scale),
                      sy(summary["ci95"][0] * scale), sy(summary["ci95"][1] * scale))
    canvas.text(224, 352, "Shuffle strength (0 = unchanged; 1 = fully shuffled)", size=7.5, align="center")
    canvas.text(224, 368, "Score change from clean: paired mean and 95% t interval over 64 cases; CRPS shows roundoff only.",
                size=7, color=MUTED, align="center")

    metadata = {
        "purpose": "Marginal invariance and joint-score response under a selected dependence mutation",
        "protocol_version": "1.0", "case_seed": seed, "sample_seed": sample_seed,
        "mutation_seed": mutation_seed, "n_samples": protocol["n_samples"],
        "joint_pair": "first horizon, targets 1 and 2, identical axis limits",
        "scatter_limits": [-bound, bound], "scatter_correlations": correlations,
        "marginal_coordinate_count": int(np.prod(clean.samples.shape[1:])),
        "all_sorted_marginals_exactly_equal": True,
        "regenerated_scores_verified_against_saved_case": verified_case_scores,
        "plotted_score_deltas": summaries,
        "crps_display_multiplier": 1e16,
        "interval": "Unadjusted paired t 95% interval over independent case-level score differences",
        "sources": _sources(root, ["experiments/protocol.json", "experiments/results/results.json",
                                    "experiments/results/manifest.json", "experiments/results-v2/manifest.json"]),
    }
    return canvas, metadata


def _blend(fraction):
    low = np.array([241, 245, 248])
    high = np.array([24, 49, 75])
    rgb = np.rint(low + fraction * (high - low)).astype(int)
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def build_ablation(root: Path):
    """Diagnostic counts and a repair/accuracy counterexample, protocol 2.0."""
    root = Path(root)
    results_path = root / "experiments/results-v2/results.json"
    manifest_path = root / "experiments/results-v2/manifest.json"
    protocol_path = root / "experiments/protocol-v2.json"
    results, manifest, protocol = _json(results_path), _json(manifest_path), _json(protocol_path)
    if _digest(results_path) != manifest["results_sha256"]:
        raise ValueError("Supplement results do not match the recorded manifest")
    if _digest(protocol_path) != manifest["protocol_sha256"] or protocol != manifest["protocol"]:
        raise ValueError("Supplement protocol does not match the recorded manifest")
    if results["protocol_sha256"] != manifest["protocol_sha256"] or results["protocol_version"] != "2.0":
        raise ValueError("Supplement result/protocol identity mismatch")

    checks = ["contract", "scalar_shift", "target_shift", "conformance", "loss"]
    headers = ["Output\ncontract", "Scalar\nshift", "Target\nshift*", "NumPy\nagreement*", "Calibrated\nMSE*"]
    labels = ["Correct implementation", "Reversed prediction lags", "Stale recursive state",
              "Missing inverse scaling", "Swapped recursive targets", "Fresh corrected refit"]
    branches = ["clean", *protocol["model_study"]["faults"], "repaired"]
    rows = []
    for branch, label in zip(branches, labels):
        records = [row for row in results["model"]["records"] if row["branch"] == branch]
        rows.append({"branch": branch, "label": label, "n": len(records),
                     "counts": {check: sum(row["alarms"][check] for row in records) for check in checks}})
    expected_n = len(protocol["model_study"]["conditions"]) * protocol["model_study"]["evaluation_cases"]
    if any(row["n"] != expected_n for row in rows):
        raise ValueError("Ablation counts do not cover the specified fixed condition corpus")

    selected = next(row for row in results["model"]["loss_summaries"]
                    if row["condition"] == "random_walk_drifts" and row["fault"] == "stale_recursive_state")
    selected_records = {branch: {row["seed"]: row for row in results["model"]["records"]
                                if row["condition"] == selected["condition"] and row["branch"] == branch}
                        for branch in (selected["fault"], "repaired")}
    if set(selected_records["repaired"]) != set(selected_records[selected["fault"]]):
        raise ValueError("Repair comparison has unpaired cases")
    differences = [selected_records["repaired"][seed]["realized_mse"] - row["realized_mse"]
                   for seed, row in selected_records[selected["fault"]].items()]
    counterexample = _interval(differences)
    _verify_summary(counterexample, selected["repair_minus_faulty_mse"])
    improved = int(sum(delta < 0 for delta in differences))
    if improved != selected["cases_improved_by_repair"]:
        raise ValueError("Saved number of repair improvements is inconsistent")

    canvas = Canvas(width=448, height=352)
    canvas.text(12, 12, "Different checks expose different implementation faults", size=10, weight="bold")
    canvas.text(12, 28, "Alarms / 384 cases per row: six fixed conditions, 64 independent cases per condition.",
                size=7.5, color=MUTED)
    x0, y0, cell_w, cell_h = 148, 65, 57.6, 24
    for j, header in enumerate(headers):
        canvas.text(x0 + (j + .5) * cell_w, 48, header, size=7.5, weight="bold", align="center")
    for i, row in enumerate(rows):
        y = y0 + i * cell_h
        canvas.text(x0 - 9, y + cell_h / 2, row["label"], size=7.4, align="right",
                    weight="bold" if i in (0, len(rows) - 1) else "normal")
        for j, check in enumerate(checks):
            count = row["counts"][check]
            fraction = count / row["n"]
            canvas.rect(x0 + j * cell_w + .7, y + .7, cell_w - 1.4, cell_h - 1.4, fill=_blend(fraction))
            canvas.text(x0 + (j + .5) * cell_w, y + cell_h / 2, f"{count}/{row['n']}",
                        size=8, color="#ffffff" if fraction > .55 else NAVY,
                        weight="bold" if count else "normal", align="center")
    for boundary in (1, 5):
        canvas.line(12, y0 + boundary * cell_h, 436, y0 + boundary * cell_h, color=GRAY, width=.65)
    canvas.text(12, 223, "* Experiment-only comparators. Counts are descriptive; dark cells mean more alarms.",
                size=7, color=MUTED)
    canvas.text(12, 237, "Clean and refit rows retain statistical false alarms; output contracts accept all four faults.",
                size=7, color=MUTED)
    canvas.line(12, 249, 436, 249, color=GRID, width=.6)

    canvas.text(12, 264, "Restoring the intended algorithm can still increase forecast error", size=8.5, weight="bold")
    canvas.text(12, 279, "Random walk with drift / stale recursive state", size=7.5, color=MUTED)
    canvas.text(436, 279, f"Only {improved}/{counterexample['n']} forecasts improve", size=7.5,
                color=ORANGE, align="right")
    left, right, axis_y = 40, 408, 307
    convert = lambda value: left + (value + 1) / 2 * (right - left)
    canvas.line(left, axis_y, right, axis_y, color=GRAY, width=.5)
    canvas.line(convert(0), 292, convert(0), 316, color=GRAY, width=.65, dash=True)
    for value in (-1, -.5, 0, .5, 1):
        canvas.line(convert(value), axis_y, convert(value), axis_y + 3, color=GRAY, width=.5)
        canvas.text(convert(value), 321, f"{value:+g}" if value else "0", size=7, color=MUTED, align="center")
    lower, upper = counterexample["ci95"]
    mean = counterexample["mean"]
    canvas.line(convert(lower), axis_y, convert(upper), axis_y, color=ORANGE, width=1.4)
    for value in (lower, upper):
        canvas.line(convert(value), axis_y - 3, convert(value), axis_y + 3, color=ORANGE, width=1.1)
    canvas.circle(convert(mean), axis_y, r=2.5, color=ORANGE)
    canvas.text(convert(mean), 294, f"+{mean:.3f} [{lower:.3f}, {upper:.3f}]", size=7.5,
                color=ORANGE, align="center")
    canvas.text(224, 341, "Realized MSE: repair − faulty (paired mean and unadjusted 95% t interval)",
                size=7.5, align="center")

    metadata = {
        "purpose": "Complementary diagnostics and a correctness-versus-accuracy counterexample",
        "protocol_version": "2.0", "check_order": checks,
        "experiment_only_checks": ["target_shift", "conformance", "loss"],
        "plotted_rows": rows,
        "aggregation": "Descriptive counts over six equally weighted fixed conditions; no pooled IID interval",
        "counterexample": {"condition": selected["condition"], "fault": selected["fault"],
                           "selection": "Descriptive counterexample among 24 exploratory fault-condition comparisons",
                           "repair_minus_faulty_realized_mse": counterexample,
                           "cases_improved": improved, "axis_limits": [-1, 1]},
        "sources": _sources(root, ["experiments/protocol-v2.json", "experiments/results-v2/results.json",
                                    "experiments/results-v2/manifest.json"]),
    }
    return canvas, metadata
