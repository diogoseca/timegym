"""Artifact integrity and formatting shared by the two studies."""
from __future__ import annotations

import hashlib
from pathlib import Path

MISSING = "---"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protect_output(output: Path, *, force: bool = False) -> None:
    artifacts = sorted({path for pattern in ("results.json", "manifest.json", "runtime.json", "example-*.json", "*.csv")
                        for path in output.glob(pattern)})
    if artifacts and not force:
        raise FileExistsError("Study artifacts already exist: " + ", ".join(str(path) for path in artifacts)
                              + "; use --force only for an intentional replacement")


def interval_cell(stat: dict | None, precision: int = 4, signed: bool = True, math: bool = False) -> str:
    if stat is None:
        return MISSING
    spec = f"{'+' if signed else ''}.{precision}f"
    lo, hi = stat["ci95"]
    text = f"{stat['mean']:{spec}} [{lo:{spec}}, {hi:{spec}}]"
    return f"${text}$" if math else text


def require_conclusion(predicate: bool, statement: str) -> None:
    if not predicate:
        raise ValueError(f"Recorded evidence does not support: {statement}")
