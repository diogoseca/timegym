"""Diagnostic counts and clean-minus-faulty loss from frozen study records."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from timegym.statistics import paired_t_interval as _interval
from vector_canvas import Canvas
from figure_common import INK, BLUE, ORANGE, GRAY, MUTED, GRID
from figure_common import _json, _digest, _sources, _verify_summary


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
              "Missing inverse scaling", "Swapped recursive targets"]
    branches = ["clean", *protocol["model_study"]["faults"]]
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
                        for branch in (selected["fault"], "clean")}
    if set(selected_records["clean"]) != set(selected_records[selected["fault"]]):
        raise ValueError("Repair comparison has unpaired cases")
    differences = [selected_records["clean"][seed]["realized_mse"] - row["realized_mse"]
                   for seed, row in selected_records[selected["fault"]].items()]
    counterexample = _interval(differences)
    _verify_summary(counterexample, selected["clean_minus_faulty_mse"])
    improved = int(sum(delta < 0 for delta in differences))
    if improved != selected["cases_improved_by_repair"]:
        raise ValueError("Saved number of repair improvements is inconsistent")

    canvas = Canvas(width=448, height=328)
    canvas.text(12, 12, "Different checks expose different implementation faults", size=10, weight="bold")
    canvas.text(12, 28, f"Alarms / {expected_n} cases per row: {len(protocol['model_study']['conditions'])} fixed conditions, "
                f"{protocol['model_study']['evaluation_cases']} independent cases per condition.",
                size=7.5, color=MUTED)
    x0, y0, cell_w, cell_h = 148, 65, 57.6, 24
    for j, header in enumerate(headers):
        canvas.text(x0 + (j + .5) * cell_w, 48, header, size=7.5, weight="bold", align="center")
    for i, row in enumerate(rows):
        y = y0 + i * cell_h
        canvas.text(x0 - 9, y + cell_h / 2, row["label"], size=7.4, align="right",
                    weight="bold" if i == 0 else "normal")
        for j, check in enumerate(checks):
            count = row["counts"][check]
            fraction = count / row["n"]
            canvas.rect(x0 + j * cell_w + .7, y + .7, cell_w - 1.4, cell_h - 1.4, fill=_blend(fraction))
            canvas.text(x0 + (j + .5) * cell_w, y + cell_h / 2, f"{count}/{row['n']}",
                        size=8, color="#ffffff" if fraction > .55 else INK,
                        weight="bold" if count else "normal", align="center")
    for boundary in (1,):
        canvas.line(12, y0 + boundary * cell_h, 436, y0 + boundary * cell_h, color=GRAY, width=.65)
    canvas.text(12, 199, "* Experiment-only comparators. Counts are descriptive; dark cells mean more alarms.",
                size=7, color=MUTED)
    canvas.text(12, 213, "The clean row retains statistical false alarms; output contracts accept all four faults.",
                size=7, color=MUTED)
    canvas.line(12, 225, 436, 225, color=GRID, width=.6)

    canvas.text(12, 240, "Restoring the intended algorithm can still increase forecast error", size=8.5, weight="bold")
    canvas.text(12, 255, "Random walk with drift / stale recursive state", size=7.5, color=MUTED)
    canvas.text(436, 255, f"Only {improved}/{counterexample['n']} forecasts improve", size=7.5,
                color=ORANGE, align="right")
    left, right, axis_y = 40, 408, 283
    convert = lambda value: left + (value + 1) / 2 * (right - left)
    canvas.line(left, axis_y, right, axis_y, color=GRAY, width=.5)
    canvas.line(convert(0), 268, convert(0), 292, color=GRAY, width=.65, dash=True)
    for value in (-1, -.5, 0, .5, 1):
        canvas.line(convert(value), axis_y, convert(value), axis_y + 3, color=GRAY, width=.5)
        canvas.text(convert(value), 297, f"{value:+g}" if value else "0", size=7, color=MUTED, align="center")
    lower, upper = counterexample["ci95"]
    mean = counterexample["mean"]
    canvas.line(convert(lower), axis_y, convert(upper), axis_y, color=ORANGE, width=1.4)
    for value in (lower, upper):
        canvas.line(convert(value), axis_y - 3, convert(value), axis_y + 3, color=ORANGE, width=1.1)
    canvas.circle(convert(mean), axis_y, r=2.5, color=ORANGE)
    canvas.text(convert(mean), 270, f"+{mean:.3f} [{lower:.3f}, {upper:.3f}]", size=7.5,
                color=ORANGE, align="center")
    canvas.text(224, 317, "Realized MSE: clean − faulty (paired mean and unadjusted 95% t interval)",
                size=7.5, align="center")

    metadata = {
        "purpose": "Complementary diagnostics and a correctness-versus-accuracy counterexample",
        "protocol_version": "2.0", "check_order": checks,
        "experiment_only_checks": ["target_shift", "conformance", "loss"],
        "plotted_rows": rows,
        "aggregation": "Descriptive counts over six equally weighted fixed conditions; no pooled IID interval",
        "counterexample": {"condition": selected["condition"], "fault": selected["fault"],
                           "selection": "Descriptive counterexample among 24 exploratory fault-condition comparisons",
                           "clean_minus_faulty_realized_mse": counterexample,
                           "cases_improved": improved, "axis_limits": [-1, 1]},
        "sources": _sources(root, ["experiments/protocol-v2.json", "experiments/results-v2/results.json",
                                    "experiments/results-v2/manifest.json"]),
    }
    return canvas, metadata
