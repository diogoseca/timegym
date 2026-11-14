"""Render manuscript figures from notebook settings and saved study records.

With no arguments every figure is rendered and the provenance record is
rewritten. ``--only NAME [NAME ...]`` renders the named figures and replaces
only their entries in the existing provenance record; the record's top-level
``inputs_sha256`` and the other figures' entries stay as they are. A partial
render lets one figure be updated on a machine that cannot reproduce the
others byte for byte (Matplotlib PNG bytes, and the Study 1 dependence case,
differ between platforms).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = "paper"
SHARED_INPUTS = [f"{PAPER}/render_figures.py", f"{PAPER}/vector_canvas.py", f"{PAPER}/figure_common.py"]


def _builder(module, function):
    def build(root):
        return getattr(__import__(module), function)(root)
    return build


# Name, builder, and the inputs that figure reads besides SHARED_INPUTS.
FIGURES = {
    "paper-scorecard": (_builder("figure_scorecard", "build_scorecard"),
                        [f"{PAPER}/figure_scorecard.py"]),
    "paper-dependence": (_builder("figure_dependence", "build_dependence"),
                         [f"{PAPER}/figure_dependence.py", "experiments/protocol.json",
                          "experiments/results/results.json", "experiments/results/manifest.json",
                          "experiments/results-v2/manifest.json"]),
    "paper-diagnostic-evidence": (_builder("figure_evidence", "build_ablation"),
                                  [f"{PAPER}/figure_evidence.py", "experiments/protocol-v2.json",
                                   "experiments/results-v2/results.json",
                                   "experiments/results-v2/manifest.json",
                                   "experiments/protocol-v2-followup.json",
                                   "experiments/results-v2-followup/results.json",
                                   "experiments/results-v2-followup/manifest.json"]),
}

# Fingerprints recorded at the top level of a full render.
ALL_INPUTS = ["experiments/results/results.json", "experiments/results-v2/results.json",
              "experiments/results-v2-followup/results.json",
              f"{PAPER}/render_figures.py", f"{PAPER}/figure_evidence.py",
              f"{PAPER}/vector_canvas.py", f"{PAPER}/figure_common.py",
              f"{PAPER}/figure_scorecard.py", f"{PAPER}/figure_dependence.py"]


def digests(root: Path, names):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def figure_inputs(name):
    return FIGURES[name][1] + SHARED_INPUTS


def render(root: Path, figures=FIGURES, only=None):
    """Render ``only`` (all figures when None) and update the provenance record."""
    directory = root / PAPER / "figures"
    provenance = directory / "paper-figure-provenance.json"
    names = list(figures) if only is None else list(dict.fromkeys(only))
    unknown = [name for name in names if name not in figures]
    if unknown:
        raise SystemExit(f"Unknown figure(s): {', '.join(unknown)}; choose from {', '.join(figures)}")
    if only is None:
        metadata = {}
    else:
        if not provenance.exists():
            raise SystemExit(f"--only needs an existing {provenance.relative_to(root)}; run a full render first")
        metadata = json.loads(provenance.read_text(encoding="utf-8"))
        # Entries of figures that no longer exist are dropped.
        metadata = {key: value for key, value in metadata.items()
                    if key in figures or key == "inputs_sha256"}
    for name in names:
        builder, inputs = figures[name]
        canvas, info = builder(root)
        canvas.render(directory, name)
        metadata[name] = {**info, "inputs_sha256": digests(root, inputs + SHARED_INPUTS)}
        print(f"Rendered {name}: SVG, PNG, TikZ", flush=True)
    if only is None:
        metadata["inputs_sha256"] = digests(root, ALL_INPUTS)
    provenance.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", metavar="NAME", choices=list(FIGURES),
                        help="render only these figures and keep the other provenance entries")
    args = parser.parse_args(argv)
    render(ROOT, only=args.only)


if __name__ == "__main__":
    main()
