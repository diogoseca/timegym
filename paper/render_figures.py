"""Render manuscript figures from notebook settings and saved study records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from figure_matrix import build_matrix

ROOT = Path(__file__).resolve().parents[1]


def main():
    from figure_dependence import build_dependence
    from figure_evidence import build_ablation
    from figure_stresses import build_stresses
    from figure_joint_stresses import build_joint_stresses
    directory = ROOT / "paper/figures"
    metadata = {}
    for name, builder in (("paper-forecast-matrix", lambda: build_matrix(ROOT)),
                          ("paper-financial-stresses", lambda: build_stresses(ROOT)),
                          ("paper-joint-stresses", lambda: build_joint_stresses(ROOT)),
                          ("paper-dependence", lambda: build_dependence(ROOT)),
                          ("paper-diagnostic-evidence", lambda: build_ablation(ROOT))):
        canvas, info = builder()
        canvas.render(directory, name)
        metadata[name] = info
        print(f"Rendered {name}: SVG, PNG, TikZ", flush=True)
    metadata["inputs_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / "experiments/results/results.json", ROOT / "experiments/results-v2/results.json",
                     ROOT / "notebooks/00_quick_start.ipynb", Path(__file__), ROOT / "paper/figure_evidence.py",
                     ROOT / "paper/vector_canvas.py", ROOT / "paper/figure_common.py",
                     ROOT / "paper/figure_matrix.py", ROOT / "paper/figure_dependence.py", ROOT / "paper/figure_stresses.py",
                     ROOT / "notebooks/26_composed_and_financial_stresses.ipynb",
                     ROOT / "paper/figure_joint_stresses.py", ROOT / "notebooks/27_joint_financial_uncertainty.ipynb"]}
    (directory / "paper-figure-provenance.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
