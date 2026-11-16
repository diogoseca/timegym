"""Draw the figure of protocol 3.0, Part 6 (part_6_zoo.figure) from saved scorecards only.

    python -m experiments.zoo.figure --output experiments/results-v3/zoo/zoo-scorecards.pdf

Rows are the ten patterns and columns the nine models, in the order of part_6_zoo.models; no model is
ordered by score. Each cell shows the low-noise and the high-noise score with its interval (score_low
to score_high) beside the row's attainable mark (timegym.scorecard._ATTAINABLE). Scores come from the
baselines that Scorecard.save_baseline wrote (<model id>.json); the intervals, which a baseline does
not hold, come from the table written beside it (<model id>.table.csv), whose scores must equal the
baseline's.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from experiments import study3_common as common

from .models import MODEL_IDS

ZOO_RESULTS = common.ROOT / "experiments/results-v3/zoo"


def _number(text):
    return math.nan if text in ("", "nan", "None") else float(text)


def load(directory, model_ids=MODEL_IDS):
    """Per model, the cells {(pattern, noise_sd): (score, low, high)} and the grid of its baseline."""
    directory, found = Path(directory), {}
    for model_id in model_ids:
        baseline = directory / f"{model_id}.json"
        if not baseline.exists():
            continue
        record = json.loads(baseline.read_text(encoding="utf-8"))
        with (directory / f"{model_id}.table.csv").open(encoding="utf-8") as handle:
            table = {(row["pattern"], float(row["noise_sd"])): row for row in csv.DictReader(handle)}
        cells = {}
        for cell in record["cells"]:
            key = (cell["pattern"], float(cell["noise_sd"]))
            row = table[key]
            score, listed = cell["score"], _number(row["score"])
            if (score is None and not math.isnan(listed)) or (score is not None and listed != score):
                raise ValueError(f"{model_id} {key}: the table's score differs from the baseline's")
            low, high = _number(row["score_low"]), _number(row["score_high"])
            cells[key] = (score, low, high)
        found[model_id] = {"grid": record["grid"], "cells": cells, "title": record["title"]}
    if not found:
        raise FileNotFoundError(f"No zoo baselines in {directory}")
    grids = {json.dumps(item["grid"], sort_keys=True) for item in found.values()}
    if len(grids) != 1:
        raise ValueError("The baselines do not share one grid")
    return found


def draw(directory, path, model_ids=MODEL_IDS):
    """Write the figure to ``path`` and return it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from timegym.grid import PATTERNS
    from timegym.scorecard import _ATTAINABLE

    found = load(directory, model_ids)
    columns = [model_id for model_id in model_ids if model_id in found]
    grid = next(iter(found.values()))["grid"]
    patterns, noise = grid["patterns"], grid["noise_sds"]
    figure, axes = plt.subplots(len(patterns), len(columns), figsize=(1.6 * len(columns) + 1.5,
                                                                        0.55 * len(patterns) + 1.0),
                                sharex=True, squeeze=False)
    colours = ("#1f5a96", "#c0602a")
    for row, pattern in enumerate(patterns):
        for column, model_id in enumerate(columns):
            ax = axes[row][column]
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, len(noise) - 0.5)
            ax.set_yticks([])
            for level, sd in enumerate(noise):
                score, low, high = found[model_id]["cells"][(pattern, float(sd))]
                y = len(noise) - 1 - level
                mark = _ATTAINABLE.get(pattern, {}).get(float(sd))
                if mark is not None:
                    ax.plot([mark, mark], [y - 0.35, y + 0.35], color="0.35", linewidth=1.0)
                if score is None:
                    continue
                if math.isfinite(low) and math.isfinite(high):
                    ax.plot([low, high], [y, y], color=colours[level % 2], linewidth=1.2)
                ax.plot([score], [y], "o", color=colours[level % 2], markersize=3)
            if column == 0:
                ax.set_ylabel(PATTERNS[pattern].title, rotation=0, ha="right", va="center", fontsize=7)
            if row == 0:
                ax.set_title(model_id.replace("_", " "), fontsize=7)
            ax.tick_params(labelsize=6)
    figure.text(0.5, 0.01, f"Score (0-100); upper dot sd {noise[0]:g}, lower dot sd {noise[-1]:g}; "
                "grey tick: attainable mark", ha="center", fontsize=7)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(path)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baselines", type=Path, default=ZOO_RESULTS)
    parser.add_argument("--output", type=Path, default=ZOO_RESULTS / "zoo-scorecards.pdf")
    args = parser.parse_args()
    print(draw(args.baselines, args.output))


if __name__ == "__main__":
    main()
