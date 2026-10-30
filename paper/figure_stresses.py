"""A descriptive financial-process matrix sourced from the application notebook.

The notebook is parsed as data: only its literal ``financial_specs`` assignment
is read. No notebook cell is executed and no study result is changed.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from vector_canvas import Canvas


BLUE = "#0000ff"
GRAY = "#808080"
REFERENCE = "#575757"
MUTED = "#626b73"


def _financial_specs(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell.get("id") == "stress-finance"]
    if len(cells) != 1 or cells[0]["cell_type"] != "code":
        raise ValueError("Expected one stress-finance code cell in the application notebook")
    tree = ast.parse("".join(cells[0]["source"]))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "financial_specs"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Expected one literal financial_specs assignment")
    specs = ast.literal_eval(assignments[0].value)
    if not isinstance(specs, list) or len(specs) != 6 or not all(isinstance(spec, dict) for spec in specs):
        raise ValueError("The financial figure requires the notebook's six explicit configurations")
    return specs


def _tick_label(value, span):
    decimals = max(0, min(6, int(np.ceil(-np.log10(span))) + 1))
    label = f"{value:.{decimals}f}"
    if float(label) == 0:
        label = label.lstrip("-")
    return label


def build_stresses(root: Path):
    """Return the financial mechanism matrix and the complete plotted evidence."""
    from timegym import NaiveForecaster, Pipeline, Scenario

    root = Path(root)
    notebook_path = root / "notebooks/26_composed_and_financial_stresses.ipynb"
    specs = _financial_specs(notebook_path)
    seeds, context_length, horizon, visible_context = [11, 22, 33], 240, 24, 72
    scenarios = [Scenario(context_length=context_length, horizon=horizon, **spec) for spec in specs]
    run = Pipeline(NaiveForecaster(strategy="last")).run(scenarios, seeds=seeds)
    if len(run.cases) != len(specs) * len(seeds):
        raise ValueError("The financial matrix has missing scenario/seed combinations")

    canvas = Canvas(width=448, height=484)
    canvas.text(0, 10, "Six financial configurations. Three seeds each.", size=10.2, weight="bold")
    for x, label in ((22, "Observed values"), (150, "Last-value forecast"), (301, "Conditional mean")):
        if x == 22:
            canvas.circle(x - 12, 29, r=1.1, color=GRAY)
        elif x == 150:
            canvas.cross(x - 12, 29, r=2, color=BLUE)
        else:
            canvas.line(x - 17, 29, x - 5, 29, color=REFERENCE, width=.8)
        canvas.text(x, 29, label, size=7.4)
    notes = ["GBM limit; zero jumps", "Compound Poisson jumps", "Volatility depends on past shocks",
             "Hidden states inferred from history", "Approximate filtered reference", "Approximate filtered reference"]
    metadata = {
        "purpose": "Illustration of six selected known-parameter process configurations; not performance evidence",
        "model": "NaiveForecaster(strategy='last')",
        "seeds": seeds,
        "context_length": context_length,
        "forecast_horizon": horizon,
        "visible_context": visible_context,
        "visible_forecast": horizon,
        "specification_source": str(notebook_path.relative_to(root)),
        "specification_cell": "stress-finance",
        "specification_loading": "AST literal_eval of financial_specs; no notebook cell execution",
        "input_notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "financial_specs": specs,
        "scale_disclosure": "Each panel has its own vertical limits; units are stated in the row labels",
        "reference_scope": "Known generator parameters; filtered regime-jump means use an approximate state posterior",
        "interpretation": "These selected realizations do not establish comparative forecasting performance or financial validity",
        "cases": [],
        "run_metadata": run.report.metadata,
    }
    for row, scenario in enumerate(scenarios):
        top = 61 + row * 67
        canvas.text(0, top - 9, scenario.display_name, size=8, weight="bold")
        canvas.text(448, top - 9, notes[row], size=7, align="right", color=MUTED)
        for col, seed in enumerate(seeds):
            retained = run.cases[row * len(seeds) + col]
            case, forecast = retained.case, retained.forecast
            if forecast is None or forecast.point is None:
                raise ValueError("The descriptive matrix requires valid point forecasts for every case")
            if forecast.validate(case.task):
                raise ValueError("A financial matrix point forecast violates its output contract")
            context = case.task.context.to_numpy()[-visible_context:, 0]
            future = case.observed.to_numpy()[:, 0]
            observed = np.r_[context, future]
            point, reference = forecast.point[:, 0], case.reference.mean[:, 0]
            values = np.r_[observed, point, reference]
            if not np.isfinite(values).all():
                raise ValueError("Financial matrix values must all be finite")
            low, high = float(values.min()), float(values.max())
            span = high - low
            pad = max(span * .12, max(abs(low), abs(high)) * 1e-12, 1e-6)
            low, high = low - pad, high + pad
            # Return axes need room for signed decimal ticks, unlike prices.
            left, width, height = 28 + col * 146, 119, 43
            xx = lambda value: left + (float(value) + visible_context) / (visible_context + horizon) * width
            yy = lambda value: top + height - (float(value) - low) / (high - low) * height

            canvas.line(left, top, left, top + height, color="#c9cdd1", width=.45)
            canvas.line(left, top + height, left + width, top + height, color="#c9cdd1", width=.45)
            tick_values = np.linspace(low + .14 * (high - low), high - .14 * (high - low), 2)
            tick_labels = [_tick_label(value, high - low) for value in tick_values]
            for value, label in zip(tick_values, tick_labels):
                canvas.text(left - 3, yy(value), label, size=6.7, align="right", color=MUTED)
            canvas.line(xx(0), top, xx(0), top + height, color="#9da7b0", width=.6, dash=True)
            for time, value in zip(range(-visible_context, horizon), observed):
                canvas.circle(xx(time), yy(value), r=.52, color=GRAY)
            canvas.polyline([(xx(time), yy(value)) for time, value in enumerate(reference)],
                            color=REFERENCE, width=.65)
            for time, value in enumerate(point):
                # At this print width adjacent forecasts are 1.24 bp apart.
                # A 1 bp-wide cross keeps constant forecasts visibly separate.
                canvas.cross(xx(time), yy(value), r=.5, color=BLUE, width=.38)
            if row == 0:
                canvas.text(left + width / 2, 40, f"Seed {seed}", size=7.4, align="center", weight="bold")
            if row == len(scenarios) - 1:
                for time, label in ((-visible_context, "-72"), (0, "0"), (horizon, "+24")):
                    canvas.text(xx(time), top + height + 9, label, size=6.8, align="center", color=MUTED)

            capabilities = dict(getattr(case.reference, "capabilities", {}))
            approximate = capabilities.get("sampler_kind") == "approximate_conditional"
            if approximate != (row in (4, 5)):
                raise ValueError("Financial row approximation labels no longer match reference capabilities")
            metadata["cases"].append({
                "row": row, "seed": seed, "display_name": scenario.display_name,
                "row_note": notes[row], "manifest": case.manifest,
                "reference_kind": case.reference.kind,
                "reference_capabilities": capabilities,
                "reference_conditioning": case.reference.conditioning,
                "approximate_filtered_reference": approximate,
                "observed_relative_times": list(range(-visible_context, horizon)),
                "forecast_relative_times": list(range(horizon)),
                "context_visible": context.tolist(), "future_visible": future.tolist(),
                "observed_visible": observed.tolist(), "forecast_visible": point.tolist(),
                "conditional_mean_visible": reference.tolist(),
                "y_limits": [low, high], "y_tick_values": tick_values.tolist(), "y_tick_labels": tick_labels,
            })
    canvas.text(224, 461, "Time relative to forecast origin  |  Each panel has its own value scale", size=7.3, align="center")
    canvas.text(224, 477, "Known parameters; regime+jump reference means use approximate filtering.",
                size=7.4, align="center", weight="bold")
    return canvas, metadata
