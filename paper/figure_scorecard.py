"""Figure 1: the package scorecard of the quick start's Ridge pipeline, in print style.

Every score, verdict, mistake, title and sentence comes from the ``Scorecard``
object; this module only lays them out on the shared vector canvas. The
scorecard is an illustration of the package output, not a study result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from figure_common import AXIS, BLUE, GRAY, INK, MUTED
from vector_canvas import Canvas

# The quick start's contender, scored with the scorecard defaults.
SCORECARD_MODEL = {"window_length": 24, "ridge_alpha": 1.0}
ANSWER_KEY = "#000000"
WIDTH, HEIGHT = 448, 590
TEXT_SIZE = 6.2

# Column geometry in bp: kind of series, one panel per noise level, mistakes.
TEXT_LEFT, TEXT_WIDTH = 0, 162
PANEL_LEFT, PANEL_WIDTH, PANEL_GAP = 168, 80, 7
MISTAKES_WIDTH = 102
INDENT = 6
PANEL_MIN = 40

_RENDERER = RendererAgg(10, 10, 72)


def _width(text, size, weight="normal"):
    """Width in bp of one line set in DejaVu Sans, the wider of the two typefaces used."""
    prop = FontProperties(family="DejaVu Sans", size=size, weight=weight)
    return _RENDERER.get_text_width_height_descent(text, prop, ismath=False)[0]


def _wrap(text, width, size, weight="normal", sep=" "):
    """Break ``text`` at ``sep`` so that every line fits ``width`` bp; words are never split."""
    lines = []
    pieces = text.split(sep)
    for index, word in enumerate(pieces):
        glue = sep.rstrip() if index < len(pieces) - 1 else ""
        space = sep[len(sep.rstrip()):]
        if lines and _width(f"{lines[-1]}{space}{word}{glue}", size, weight) <= width:
            lines[-1] = f"{lines[-1]}{space}{word}{glue}"
        else:
            lines.append(f"{word}{glue}")
    if any(_width(line, size, weight) > width for line in lines):
        raise ValueError(f"A word is wider than {width} bp: {text!r}")
    return lines


def _mistake_lines(note):
    """Lay out one package mistake note, one mistake per line; later lines are indented.

    The text is the package's, unchanged: only line breaks are chosen here.
    """
    parts = note.split("; ")
    parts = [part + (";" if index < len(parts) - 1 else "") for index, part in enumerate(parts)]
    # The first line is not indented, so the first mistake may use the full width.
    pieces = ([parts[0]] if _width(parts[0], TEXT_SIZE) <= MISTAKES_WIDTH
              else _wrap(parts[0], MISTAKES_WIDTH - INDENT, TEXT_SIZE))
    pieces += [piece for part in parts[1:] for piece in _wrap(part, MISTAKES_WIDTH - INDENT, TEXT_SIZE)]
    return [(0 if index == 0 else INDENT, piece) for index, piece in enumerate(pieces)]


def _block(canvas, x, top, lines, size, **style):
    """Write lines from ``top`` downwards; return the y below the last line."""
    step = size * 1.2
    for index, line in enumerate(lines):
        canvas.text(x, top + step * (index + .5), line, size=size, **style)
    return top + step * len(lines)


def _example(entry, visible):
    """Instance 1 of a cell: recent history, what happened, answer key and forecast."""
    from timegym.reporting import _case_checks

    case = entry.case
    if entry.forecast is None or entry.forecast.point is None or not all(_case_checks(entry.result).values()):
        raise ValueError("The scorecard figure expects a valid point forecast in every example")
    context = case.task.context.iloc[-visible:, 0]
    return {
        "context_time": [int(t) for t in context.index],
        "context": context.to_numpy(dtype=float).tolist(),
        "future_time": [int(t) for t in case.task.future_index],
        "observed": case.observed.iloc[:, 0].to_numpy(dtype=float).tolist(),
        "answer_key": np.asarray(case.reference.mean, dtype=float)[:, 0].tolist(),
        "forecast": np.asarray(entry.forecast.point, dtype=float)[:, 0].tolist(),
        "forecast_origin": float(context.index[-1]) + .5,
    }


def build_scorecard(root: Path):
    """Return the scorecard canvas and the values behind every mark."""
    from timegym import Pipeline
    from timegym.grid import PATTERNS
    from timegym.plotting import _BAND_COLORS
    from timegym.scorecard import _LEGEND, _mistake_notes

    model = SCORECARD_MODEL
    pipeline = (Pipeline(window_length=model["window_length"]).add(StandardScaler())
                .add(Ridge(alpha=model["ridge_alpha"])))
    card = pipeline.scorecard()
    table = card.table
    keys = list(dict.fromkeys(table["pattern"]))
    levels = list(dict.fromkeys(table["noise"]))
    sds = list(dict.fromkeys(table["noise_sd"]))
    horizon = len(card.run.cases[0].case.task.future_index)
    visible = min(len(card.run.cases[0].case.task.context), 4 * horizon)  # as the package plot
    first = {}
    for entry in card.run.cases:
        first.setdefault(entry.scenario_index, entry)

    canvas = Canvas(WIDTH, HEIGHT)
    hex_color = lambda value: "#ffffff" if value == "white" else value

    # Header: model and overall score, design, verdict lines and output checks.
    canvas.text(0, 7, f"{card.title}: {card.score:.0f}/100", size=10.2, weight="bold")
    subtitle = (f"TimeGym scorecard · {len(keys)} kinds of series × {len(levels)} noise levels · "
                f"{card.instances} instances each · seed {card.seed}")
    canvas.text(0, 20, subtitle, size=6.8, color=MUTED)
    y = 27
    for label, names in (("Good at:", card.good_at), ("Fails at:", card.fails_at),
                         ("Breaks under heavy noise:", card.noise_sensitive)):
        if label.startswith("Breaks") and not names:
            continue
        y = _block(canvas, 0, y, _wrap(f"{label} {', '.join(names) or 'none'}.", WIDTH, 7, sep=", "), 7)
    checks = "Output checks:  " + "   ".join(
        f"{'✓' if not row['failed'] else '✗'} {row['title']}"
        + (f" ({row['failed']} failed)" if row["failed"] else "") for row in card.checks)
    y = _block(canvas, 0, y + 1.5, _wrap(checks, WIDTH, TEXT_SIZE, sep="   "), TEXT_SIZE, color=MUTED)

    # Column headers.
    header_y = y + 9
    panel_x = [PANEL_LEFT + i * (PANEL_WIDTH + PANEL_GAP) for i in range(len(levels))]
    mistakes_x = WIDTH - MISTAKES_WIDTH
    headers = [(TEXT_LEFT, "What can and can't be predicted")]
    headers += [(x, f"{level} (sd {sd:g})") for x, level, sd in zip(panel_x, levels, sds)]
    headers += [(mistakes_x, "Mistakes")]
    for x, header in headers:
        canvas.text(x, header_y, header, size=7, weight="bold")
    canvas.line(0, header_y + 6, WIDTH, header_y + 6, color=AXIS, width=.45)

    legend_lines = _wrap(_LEGEND + " Each panel has its own vertical scale.", WIDTH, TEXT_SIZE)
    footer_height = TEXT_SIZE * 1.2 * (len(legend_lines) + 1) + 4
    rows_top = header_y + 10

    # Rows take the height their text needs, at least PANEL_MIN; the rest is shared equally.
    def row_text(key):
        pattern = PATTERNS[key]
        title = _wrap(pattern.title, TEXT_WIDTH, 7.2, "bold")
        sentences = [_wrap(f"{mark} {sentence}", TEXT_WIDTH, TEXT_SIZE)
                     for mark, sentence in (("✓", pattern.predictable), ("✗", pattern.unpredictable))]
        notes = _mistake_notes(table.loc[table["pattern"] == key])
        mistakes = [line for note in notes for line in _mistake_lines(note)]
        need = max(2.5 + 7.2 * 1.2 * len(title) + sum(.4 + TEXT_SIZE * 1.2 * len(lines) for lines in sentences),
                   2.5 + TEXT_SIZE * 1.2 * max(1, len(mistakes)), PANEL_MIN) + 1.5
        return title, sentences, notes, mistakes, need

    texts = {key: row_text(key) for key in keys}
    spare = HEIGHT - footer_height - rows_top - sum(texts[key][-1] for key in keys)
    if spare < 0:
        raise ValueError(f"The scorecard needs {-spare:.1f} bp more height")
    heights = [texts[key][-1] + spare / len(keys) for key in keys]
    tops = rows_top + np.r_[0, np.cumsum(heights)]

    metadata = {
        "purpose": "Illustration of the package scorecard for the quick start's Ridge pipeline; not a study result",
        "model": {**model, "pipeline": "Pipeline(window_length=24).add(StandardScaler()).add(Ridge(alpha=1.0))",
                  "title": card.title},
        "specification_source": "paper/figure_scorecard.py",
        "scorecard_call": "pipeline.scorecard() with package defaults",
        "seed": int(card.seed), "instances": int(card.instances),
        "context_length": len(card.run.cases[0].case.task.context), "forecast_horizon": horizon,
        "visible_context": visible, "example_instance": 1,
        "overall_score": float(card.score), "overall_score_shown": f"{card.score:.0f}",
        "good_at": card.good_at, "fails_at": card.fails_at, "noise_sensitive": card.noise_sensitive,
        "checks": card.checks, "legend": _LEGEND,
        "scale_disclosure": "Each panel has its own vertical limits; panels have no value axis, as in the package plot",
        "cells": [],
    }

    rows_by_cell = {(row.pattern, row.noise): (index, row) for index, row in enumerate(table.itertuples())}
    for number, key in enumerate(keys):
        title, sentences, notes, mistakes, _ = texts[key]
        top, row_height = tops[number], heights[number]
        if number:
            canvas.line(0, top, WIDTH, top, color="#eceef0", width=.4)
        # Kind of series: title, then what can and cannot be predicted.
        y = _block(canvas, TEXT_LEFT, top + 2.5, title, 7.2, weight="bold")
        for lines in sentences:
            y = _block(canvas, TEXT_LEFT, y + .4, lines, TEXT_SIZE)
        # One panel per noise level: badge, then the example series.
        for left, level in zip(panel_x, levels):
            index, row = rows_by_cell[(key, level)]
            if row.failed_checks or not np.isfinite(row.score):
                raise ValueError("The print scorecard expects every cell to be scored")
            label = f"{row.score:.0f}  {row.verdict}"
            fill, ink = (hex_color(value) for value in _BAND_COLORS[row.verdict])
            badge_w = _width(label, 6.6, "bold") + 5
            canvas.rect(left, top + 2.5, badge_w, 8.6, fill=fill)
            canvas.text(left + badge_w / 2, top + 6.8, label, size=6.6, weight="bold", color=ink, align="center")
            panel = _example(first[index], visible)
            plot_top, plot_bottom = top + 13.5, top + row_height - 3
            times = panel["context_time"] + panel["future_time"]
            values = np.r_[panel["context"], panel["observed"], panel["answer_key"], panel["forecast"]]
            low, high = float(values.min()), float(values.max())
            pad = max((high - low) * .06, 1e-6)
            low, high = low - pad, high + pad
            t0, t1 = times[0] - .5, times[-1] + .5
            xx = lambda t: left + (float(t) - t0) / (t1 - t0) * PANEL_WIDTH
            yy = lambda v: plot_bottom - (float(v) - low) / (high - low) * (plot_bottom - plot_top)
            canvas.line(left, plot_bottom, left + PANEL_WIDTH, plot_bottom, color=AXIS, width=.45)
            canvas.line(xx(panel["forecast_origin"]), plot_top, xx(panel["forecast_origin"]), plot_bottom,
                        color="#9da7b0", width=.5, dash=True)
            for t, v in zip(times, panel["context"] + panel["observed"]):
                canvas.circle(xx(t), yy(v), r=.42, color=GRAY)
            canvas.polyline([(xx(t), yy(v)) for t, v in zip(panel["future_time"], panel["answer_key"])],
                            color=ANSWER_KEY, width=.7)
            for t, v in zip(panel["future_time"], panel["forecast"]):
                canvas.cross(xx(t), yy(v), r=.55, color=BLUE, width=.3)
            metadata["cells"].append({
                "pattern": key, "title": row.title, "noise": row.noise, "noise_sd": float(row.noise_sd),
                "score": float(row.score), "score_shown": f"{row.score:.0f}", "verdict": row.verdict,
                "mistakes": row.mistakes, "unavoidable_mse": float(row.unavoidable_mse),
                "avoidable_mse": float(row.avoidable_mse), "mse": float(row.mse),
                "instances": int(row.instances), "failed_checks": int(row.failed_checks),
                "badge_fill": fill, "example": {**panel, "y_limits": [low, high],
                                                "manifest": first[index].case.manifest},
            })
        # Mistakes named for the row, one note per noise level, as the package plot.
        for line_number, (indent, line) in enumerate(mistakes or [(0, "None found")]):
            canvas.text(mistakes_x + indent, top + 2.5 + TEXT_SIZE * 1.2 * (line_number + .5), line,
                        size=TEXT_SIZE, color=INK if notes else MUTED)

    # Footnote: what the score means and what the marks are.
    y = HEIGHT - footer_height + 3
    canvas.line(0, y - 1.5, WIDTH, y - 1.5, color=AXIS, width=.45)
    y = _block(canvas, 0, y, legend_lines, TEXT_SIZE, color=MUTED)
    marks_y = y + TEXT_SIZE * .6
    x = 0
    canvas.text(x, marks_y, "Panels show instance 1:", size=TEXT_SIZE, color=MUTED)
    x += _width("Panels show instance 1:", TEXT_SIZE) + 9
    for kind, label in (("dot", "observations"), ("line", "answer key"),
                        ("cross", "forecast"), ("origin", "forecast origin.")):
        if kind == "dot":
            for dx in (-5, -2.5, 0):
                canvas.circle(x + dx, marks_y, r=.6, color=GRAY)
        elif kind == "line":
            canvas.line(x - 7, marks_y, x, marks_y, color=ANSWER_KEY, width=.8)
        elif kind == "cross":
            for dx in (-5, -1):
                canvas.cross(x + dx, marks_y, r=1.1, color=BLUE, width=.45)
        else:
            canvas.line(x - 2, marks_y - 3.2, x - 2, marks_y + 3.2, color="#9da7b0", width=.6, dash=True)
        x += 4
        canvas.text(x, marks_y, label, size=TEXT_SIZE, color=MUTED)
        x += _width(label, TEXT_SIZE) + 12
    metadata["run_metadata"] = card.report.metadata
    return canvas, metadata
