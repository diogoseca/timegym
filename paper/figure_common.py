"""Notebook specifications and palette shared by the manuscript figures."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np

INK = "#18314b"
MUTED = "#626b73"
GRID = "#e1e4e7"
AXIS = "#c9cdd1"
GRAY = "#808080"
BLUE = "#0000ff"
REFERENCE = "#575757"
ORANGE = "#c65c16"


def notebook_literal(path: Path, cell_id: str, name: str, *, expect_len: int | None = None):
    """Read one literal assignment without executing notebook code."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = [cell for cell in notebook["cells"] if cell.get("id") == cell_id]
        if len(cells) != 1 or cells[0]["cell_type"] != "code":
            raise ValueError(f"Expected one {cell_id} code cell in {path}")
        tree = ast.parse("".join(cells[0]["source"]))
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
        if len(assignments) != 1:
            raise ValueError(f"Expected one literal {name} assignment in {path}:{cell_id}")
        value = ast.literal_eval(assignments[0].value)
    except (KeyError, TypeError, SyntaxError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid notebook specification {path}:{cell_id}:{name}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid literal {path}:{cell_id}:{name}: {exc}") from exc
    if expect_len is not None and (not isinstance(value, (list, tuple, dict)) or len(value) != expect_len):
        raise ValueError(f"Expected {expect_len} entries in {path}:{cell_id}:{name}")
    return value


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(root, names):
    return {name: _digest(root / name) for name in names}


def _verify_summary(actual, saved):
    if actual["n"] != saved["n"]:
        raise ValueError("Saved summary has a different case count")
    np.testing.assert_allclose([actual["mean"], *actual["ci95"]],
                               [saved["mean"], *saved["ci95"]], rtol=1e-12, atol=1e-30)
