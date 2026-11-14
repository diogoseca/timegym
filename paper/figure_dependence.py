"""Publication vector scene of the first Study 1 dependence case.

Only the illustrative first case is regenerated, and its scores are checked
against the saved records.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vector_canvas import Canvas


from figure_common import BLUE, ORANGE, GRAY, MUTED, GRID, _json, _digest, _sources


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


def build_dependence(root: Path):
    """Joint-sample mechanism of the first pilot-protocol case."""
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

    canvas = Canvas(width=448, height=222)
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

    metadata = {
        "purpose": "Marginal invariance and changed joint dependence under a selected dependence mutation",
        "protocol_version": protocol["protocol_version"], "case_seed": seed, "sample_seed": sample_seed,
        "mutation_seed": mutation_seed, "n_samples": protocol["n_samples"],
        "joint_pair": "first horizon, targets 1 and 2, identical axis limits",
        "scatter_limits": [-bound, bound], "scatter_correlations": correlations,
        "marginal_coordinate_count": int(np.prod(clean.samples.shape[1:])),
        "all_sorted_marginals_exactly_equal": True,
        "regenerated_scores_verified_against_saved_case": verified_case_scores,
        "sources": _sources(root, ["experiments/protocol.json", "experiments/results/results.json",
                                    "experiments/results/manifest.json", "experiments/results-v2/manifest.json"]),
    }
    return canvas, metadata


