"""Forecast matrix from literal quick-start notebook settings."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from timegym import Pipeline, Scenario
from vector_canvas import Canvas
from figure_common import AXIS, BLUE, GRAY, MUTED, REFERENCE, notebook_literal


def build_matrix(root: Path):
    """Render the quick-start notebook specifications and retain plotted values."""
    path = root / "notebooks/00_quick_start.ipynb"
    specs = notebook_literal(path, "quick-03", "matrix_specs", expect_len=6)
    scenarios = [Scenario(**spec) for spec in specs]
    seeds = notebook_literal(path, "quick-03", "matrix_seeds", expect_len=3)
    model = notebook_literal(path, "quick-03", "matrix_model")
    pipeline = Pipeline(window_length=model["window_length"]).add(StandardScaler()).add(Ridge(alpha=model["ridge_alpha"]))
    run = pipeline.run(scenarios, seeds=seeds)
    canvas = Canvas(448, 481)
    canvas.text(0, 10, "Five families. Six configurations. Three seeds each.", size=10.2, weight="bold")
    for x, label in ((22, "Observed values"), (150, "Point forecast"), (271, "Conditional mean")):
        if x == 22: canvas.circle(x-12, 29, r=1.1, color=GRAY)
        elif x == 150: canvas.cross(x-12, 29, r=2, color=BLUE)
        else: canvas.line(x-17, 29, x-5, 29, color=REFERENCE, width=.8)
        canvas.text(x, 29, label, size=7.4)
    notes = ["Independent over time", "Fresh shocks around a line", "Fresh shocks around a cycle",
             "Shocks decay", "Shocks accumulate; drift = 0", "Shocks accumulate; drift = 0.03 / step"]
    metadata = {"model": model, "seeds": seeds, "specification_cell": "quick-03",
                "specification_source": str(path.relative_to(root)), "matrix_specs": specs,
                "context_length": specs[0]["context_length"], "forecast_horizon": specs[0]["horizon"],
                "visible_context": 72, "visible_forecast": 48,
                "note": "Illustrative quick-start configurations; no change to the study evaluation protocol.",
                "random_walk_pairing": "Within each seed the two walks share innovations; paths differ by 0.03*t.",
                "cases": [], "run_metadata": run.report.metadata}
    for col in range(len(seeds)):
        walks = [run.cases[row*len(seeds)+col].case for row in (4, 5)]
        paths = [np.r_[case.task.context.to_numpy()[:, 0], case.observed.to_numpy()[:, 0]] for case in walks]
        np.testing.assert_allclose(paths[1]-paths[0], .03*np.arange(len(paths[0])), atol=1e-12)
    for row, scenario in enumerate(scenarios):
        top = 61 + row*67
        canvas.text(0, top-9, scenario.display_name, size=8, weight="bold")
        canvas.text(448, top-9, notes[row], size=6.8, align="right", color=MUTED)
        for col, seed in enumerate(seeds):
            retained = run.cases[row*3+col]
            case, forecast = retained.case, retained.forecast
            # Plot a disclosed common window rather than compressing 400 markers
            # into a 1.8-inch panel. Fits and forecasts use the full notebook task.
            actual = np.r_[case.task.context.to_numpy()[-72:, 0], case.observed.to_numpy()[:48, 0]]
            pred, reference = forecast.point[:48, 0], case.reference.mean[:48, 0]
            values = np.r_[actual, pred, reference]
            low, high = float(values.min()), float(values.max())
            pad = max((high-low)*.1, .1); low -= pad; high += pad
            left, width, height = 22+col*146, 125, 43
            xx = lambda v: left+(float(v)+72)/120*width
            yy = lambda v: top+height-(float(v)-low)/(high-low)*height
            canvas.line(left, top, left, top+height, color=AXIS, width=.45)
            canvas.line(left, top+height, left+width, top+height, color=AXIS, width=.45)
            for v in np.linspace(low+.1*(high-low), high-.1*(high-low), 2):
                label = f"{v:.0f}" if high-low>=3 else f"{v:.1f}"
                if float(label) == 0:
                    label = label.lstrip("-")
                canvas.text(left-3, yy(v), label, size=6.7, align="right", color=MUTED)
            canvas.line(xx(0), top, xx(0), top+height, color="#9da7b0", width=.6, dash=True)
            for t, value in zip(range(-72, 48), actual): canvas.circle(xx(t), yy(value), r=.52, color=GRAY)
            canvas.polyline([(xx(t), yy(v)) for t,v in enumerate(reference)], color=REFERENCE, width=.65)
            for t, value in enumerate(pred): canvas.cross(xx(t), yy(value), r=.9, color=BLUE, width=.38)
            if row == 0: canvas.text(left+width/2, 44, f"Seed {seed}", size=7.4, align="center", weight="bold")
            if row == len(scenarios)-1:
                for t, label in ((-72, "-72"), (0, "0"), (48, "+48")):
                    canvas.text(xx(t), top+height+9, label, size=6.8, align="center", color=MUTED)
            metadata["cases"].append({"manifest": case.manifest,
                                      "observed_visible": actual.tolist(), "forecast_visible": pred.tolist(),
                                      "conditional_mean_visible": reference.tolist(), "y_limits": [low,high]})
    canvas.text(224, 461, "Time relative to forecast origin  |  Each panel has its own value scale", size=7.3, align="center")
    canvas.text(224, 475, "Paired walk rows share the same shocks: only their drift changes.", size=7.4, align="center", weight="bold")
    return canvas, metadata


