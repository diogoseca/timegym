"""Artifact integrity and formatting shared by the two studies."""
from __future__ import annotations

import math

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


def rate_cell(stat: dict | None) -> str:
    if stat is None:
        return MISSING
    lo, hi = stat["ci95"]
    return f"{stat['count']}/{stat['n']} [{100*lo:.1f}, {100*hi:.1f}]"


def scientific_tex(value: float) -> str:
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.2e}".split("e")
    return "$" + mantissa + r"\times10^{" + str(int(exponent)) + "}$"


def require_conclusion(predicate: bool, statement: str) -> None:
    if not predicate:
        raise ValueError(f"Recorded evidence does not support: {statement}")


def close_statistics(computed: dict, recorded: dict, rel_tol: float = 1e-8) -> bool:
    """Compare interval summaries up to floating-point roundoff across library versions.

    The tolerance is relative to the largest magnitude in either summary, so a
    bound that happens to sit near zero is judged on the scale of its interval.
    """
    if computed.keys() != recorded.keys():
        return False

    def numbers(summary: dict) -> list[float]:
        out = []
        for value in summary.values():
            items = value if isinstance(value, (list, tuple)) else [value]
            out.extend(abs(item) for item in items if isinstance(item, (int, float)) and not isinstance(item, bool))
        return out

    abs_tol = max(1e-9, rel_tol * max(numbers(computed) + numbers(recorded), default=0.0))
    for key, value in computed.items():
        other = recorded[key]
        if isinstance(value, (list, tuple)):
            if len(value) != len(other) or not all(math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(value, other)):
                return False
        elif isinstance(value, float) or isinstance(other, float):
            if not math.isclose(value, other, rel_tol=rel_tol, abs_tol=abs_tol):
                return False
        elif value != other:
            return False
    return True
