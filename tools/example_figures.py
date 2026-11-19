"""Copy each example's gallery picture out of its executed notebook.

An example notebook names one of its code cells in ``metadata.timegym.gallery_figure``.
This tool writes that cell's plot to ``docs/images/examples/<notebook name>.png``, the
picture on the example's card in ``docs/examples.md``. Run it after executing an example
with ``tools/check_notebooks.py --write``; ``--check`` changes nothing and fails when a
picture is missing, out of date or left over from a removed example.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PICTURES = ROOT / "docs" / "images" / "examples"


def gallery_picture(path: Path) -> bytes:
    """The PNG of the cell the notebook names as its gallery figure."""
    notebook = nbformat.read(path, as_version=4)
    wanted = notebook.metadata.get("timegym", {}).get("gallery_figure")
    if not wanted:
        raise ValueError(f"{path.name}: metadata.timegym.gallery_figure names no cell")
    for cell in notebook.cells:
        if cell.get("id") == wanted and cell.cell_type == "code":
            for output in cell.get("outputs", []):
                picture = output.get("data", {}).get("image/png")
                if picture:
                    return base64.b64decode(picture)
            raise ValueError(f"{path.name}: cell {wanted!r} has no plot; execute the notebook first")
    raise ValueError(f"{path.name}: no code cell with id {wanted!r}")


def expected() -> dict[Path, bytes]:
    return {PICTURES / f"{path.stem}.png": gallery_picture(path) for path in sorted(EXAMPLES.glob("*.ipynb"))}


def problems() -> list[str]:
    wanted = expected()
    found = [f"{path.relative_to(ROOT)} is missing or out of date" for path, picture in wanted.items()
             if not path.is_file() or path.read_bytes() != picture]
    found += [f"{path.relative_to(ROOT)} belongs to no example" for path in sorted(PICTURES.glob("*.png"))
              if path not in wanted]
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report stale pictures without writing")
    args = parser.parse_args()
    if not args.check:
        PICTURES.mkdir(parents=True, exist_ok=True)
        wanted = expected()
        for path, picture in wanted.items():
            path.write_bytes(picture)
        for path in PICTURES.glob("*.png"):
            if path not in wanted:
                path.unlink()
    found = problems()
    for problem in found:
        print(problem, file=sys.stderr)
    if not found:
        print(f"{len(expected())} gallery pictures match their notebooks.")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
