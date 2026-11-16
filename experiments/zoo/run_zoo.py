"""Make the scorecards of protocol 3.0, Part 6 (part_6_zoo), one model at a time.

    python -m experiments.zoo.run_zoo --plan
    python -m experiments.zoo.run_zoo --models last_value seasonal_naive ets auto_arima theta hgb_lags mlp_lags \\
        context_parroting
    python -m experiments.zoo.run_zoo --models chronos_small

Each model gets Pipeline(model).scorecard(seed=14000000, instances=30) with every other argument at its
default, saved with Scorecard.save_baseline as experiments/results-v3/zoo/<model id>.json, beside
<model id>.table.csv (the scorecard's table, which holds the intervals the figure draws) and
<model id>.manifest.json. The Chronos scorecard may run alone on another machine; it is refused while
part_6_zoo.models holds no revision, and its manifest records what part_6_zoo.pretrained_environment
lists. The run refuses to start until study3_common.require_start allows the zoo; --plan prints what
would run and computes nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from threadpoolctl import threadpool_info, threadpool_limits

from timegym import Pipeline
from timegym.grid import scenario_grid

from experiments import study3_common as common
from experiments.common import sha256

from . import models

PART = "zoo"
SEED, INSTANCES = 14_000_000, 30
TABLE_COLUMNS = ("pattern", "noise_sd", "score", "score_low", "score_high", "attainable", "instances",
                 "failed_checks")
ZOO = common.ROOT / "experiments/zoo"
PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl", "statsforecast", "coreforecast",
            "statsmodels", "matplotlib", "torch", "transformers", "accelerate", "chronos-forecasting")


# The dependence seed offset that Pipeline.run uses by default, and so the scorecard.
DEPENDENCE_SEED_OFFSET = int(inspect.signature(Pipeline.run).parameters["dependence_seed_offset"].default)


def check_scorecard_seeds(protocol, seed=SEED, instances=INSTANCES):
    """The series and dependence seeds the scorecard derives must be the zoo_series and zoo_dependence blocks."""
    series = [seed + index for index in range(instances)]
    dependence = [value + DEPENDENCE_SEED_OFFSET for value in series]
    if series != common.case_seeds(protocol, "zoo_series"):
        raise ValueError("The scorecard's series seeds differ from the zoo_series block")
    if dependence != common.case_seeds(protocol, "zoo_dependence"):
        raise ValueError("The scorecard's dependence seeds differ from the zoo_dependence block")
    for cell in scenario_grid():
        if cell.scenario.n_targets != 1:
            raise ValueError(f"{cell.pattern}: the zoo models forecast one target")
    return series, dependence


def source_files():
    """The zoo's Python files and the modules it imports from experiments/, for the manifest."""
    shared = ("study3_common.py", "common.py", "study3_learners.py", "study3_subject.py")
    return sorted(ZOO.glob("*.py")) + [common.ROOT / "experiments" / name for name in shared
                                       if (common.ROOT / "experiments" / name).exists()]


def scorecard(model):
    """The registered scorecard of one model."""
    return Pipeline(model).scorecard(seed=SEED, instances=INSTANCES)


def _git_commit():
    try:
        found = subprocess.run(["git", "-C", str(common.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                               check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return found.stdout.strip()


def _pip_freeze():
    found = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return found.stdout.splitlines() if found.returncode == 0 else None


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def chronos_environment(revision, model_id=models.CHRONOS_ID):
    """What part_6_zoo.pretrained_environment lists for the Chronos manifest.

    The commit the model host resolves must equal the registered ``revision``.
    """
    import torch
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(model_id, revision=revision))
    resolved = snapshot.name
    if resolved != revision:
        raise RuntimeError(f"The model host resolved {model_id} to {resolved}, not the registered {revision}")
    files = {str(path.relative_to(snapshot)): hashlib.sha256(path.resolve().read_bytes()).hexdigest()
             for path in sorted(snapshot.rglob("*")) if path.is_file()}
    return {"cpu_model": _cpu_model(), "pip_freeze": _pip_freeze(), "torch_version": torch.__version__,
            "torch_config": torch.__config__.show(), "torch_threads": torch.get_num_threads(),
            "model_id": model_id, "resolved_commit": resolved, "file_sha256": files}


def table_rows(card):
    """The scorecard table's columns that the figure needs, one dictionary per cell."""
    rows = []
    for row in card.table.to_dict(orient="records"):
        rows.append({key: row[key] for key in TABLE_COLUMNS})
    return rows


def _write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TABLE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def outputs(output, model_id):
    return {"baseline": output / f"{model_id}.json", "table": output / f"{model_id}.table.csv",
            "manifest": output / f"{model_id}.manifest.json", "runtime": output / f"{model_id}.runtime.json"}


def run_models(model_ids, output, protocol, *, make=models.build, score=scorecard, force=False, extra=None):
    """Score each model with ``score(make(model_id, protocol))`` and write its four files; returns the manifests.

    ``extra(model_id)`` may return more manifest entries (the Chronos environment). Existing files of a
    model are refused unless ``force``.
    """
    output = Path(output)
    for model_id in model_ids:
        existing = [path for path in outputs(output, model_id).values() if path.exists()]
        if existing and not force:
            raise FileExistsError("Zoo artifacts already exist: " + ", ".join(map(str, existing))
                                  + "; use --force only for an intentional replacement")
    output.mkdir(parents=True, exist_ok=True)
    series, dependence = check_scorecard_seeds(protocol)
    entries = {row["id"]: row for row in protocol["part_6_zoo"]["models"]}
    manifests = {}
    for model_id in model_ids:
        paths = outputs(output, model_id)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            model = make(model_id, protocol)
            if model_id == "chronos_small":
                import torch

                torch.set_num_threads(1)
                torch.manual_seed(common.generator_tag(protocol, "chronos_sampling"))
            card = score(model)
            pools = threadpool_info()
        seconds = time.perf_counter() - started
        card.save_baseline(paths["baseline"])
        _write_table(paths["table"], table_rows(card))
        manifest = {"model_id": model_id, "model_index": models.model_index(model_id),
                    "implementation": entries[model_id]["implementation"],
                    "scorecard": {"seed": SEED, "instances": INSTANCES, "other_arguments": "defaults"},
                    "series_seeds": [series[0], series[-1]], "dependence_seeds": [dependence[0], dependence[-1]],
                    "protocol_sha256": sha256(common.PROTOCOL), "baseline_sha256": sha256(paths["baseline"]),
                    "table_sha256": sha256(paths["table"]), "timegym_commit": _git_commit(),
                    "source_files": {str(path.relative_to(common.ROOT)): sha256(path) for path in source_files()},
                    **common.environment(PACKAGES), **((extra(model_id) if extra else None) or {})}
        paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        paths["runtime"].write_text(json.dumps({"seconds": seconds, "thread_limit": 1, "threadpools": pools},
                                               indent=2, sort_keys=True) + "\n")
        manifests[model_id] = manifest
        print(f"Zoo scorecard written: {model_id}", flush=True)
    return manifests


def plan(protocol, model_ids):
    series, dependence = check_scorecard_seeds(protocol)
    return {"part": PART, "models": list(model_ids), "cells": len(scenario_grid()), "instances": INSTANCES,
            "fits_per_model": len(scenario_grid()) * INSTANCES, "series_seeds": [series[0], series[-1]],
            "dependence_seeds": [dependence[0], dependence[-1]],
            "chronos_revision": models.chronos_revision(protocol),
            "cpu_hours_estimate": protocol["runtime_estimate_cpu_hours"]["part_6"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[name for name in models.MODEL_IDS if name != "chronos_small"],
                        choices=models.MODEL_IDS, help="Model ids to score (default: the eight local models)")
    parser.add_argument("--output", type=Path, default=common.RESULTS / PART)
    parser.add_argument("--force", action="store_true", help="Replace existing artifacts of the chosen models")
    parser.add_argument("--plan", action="store_true", help="Print what would run and compute nothing")
    args = parser.parse_args()
    protocol = common.load_protocol()
    models.registered_ids(protocol)
    if args.plan:
        print(json.dumps(plan(protocol, args.models), indent=2))
        return
    common.require_start(protocol, PART)
    revision = models.chronos_revision(protocol)
    if "chronos_small" in args.models and revision is None:
        parser.error("part_6_zoo.models holds no Chronos revision; the Chronos scorecard may not run")

    # The weights are fetched and their commit checked before any scorecard is made.
    pretrained = chronos_environment(revision) if "chronos_small" in args.models else None

    def extra(model_id):
        return {"pretrained_environment": pretrained} if model_id == "chronos_small" else None

    run_models(args.models, args.output, protocol, force=args.force, extra=extra)


if __name__ == "__main__":
    main()
