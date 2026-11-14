"""Execute the source notebooks and tutorials in fresh kernels using this Python runtime.

Run from a source checkout with ``python tools/check_notebooks.py``. Existing
outputs are cleared in memory before execution. Use --write to retain the new
outputs; CI checks execution without modifying the notebooks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"source": ROOT / "source", "tutorials": ROOT / "tutorials"}


def execute(path: Path, scratch: Path, *, write: bool, html_dir: Path | None) -> dict:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            # A tutorial must run all its cells, including intentionally caught
            # exceptions. Do not allow metadata to hide an unhandled error.
            if {"skip-execution", "raises-exception"} & set(cell.metadata.get("tags", [])):
                raise ValueError(f"{path.name}: execution-skipping/error tags are forbidden")

    kernel_root = scratch / "kernels"
    kernel = kernel_root / "timegym-tutorial"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "TimeGym tutorial validation",
        "language": "python",
    }), encoding="utf-8")
    env = dict(os.environ)
    for key, directory in {
        "MPLCONFIGDIR": "matplotlib", "XDG_CACHE_HOME": "cache",
        "JUPYTER_CONFIG_DIR": "jupyter-config", "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
        "IPYTHONDIR": "ipython",
    }.items():
        folder = scratch / directory
        folder.mkdir(parents=True, exist_ok=True)
        env[key] = str(folder)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"  # CI sets Agg for pytest; kernels must render inline
    # Build the temporary font cache before tutorial execution, so setup chatter
    # does not become part of a notebook's teaching outputs. Cache reuse carries
    # no Python variables or model state between the independent kernels.
    if not list((scratch / "matplotlib").glob("fontlist-*.json")):
        subprocess.run([sys.executable, "-c", "import matplotlib.font_manager"],
                       env=env, check=True, capture_output=True, text=True)
    manager = AsyncKernelManager(
        kernel_name="timegym-tutorial",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)], ensure_native_kernel=False),
    )
    client = NotebookClient(
        notebook, km=manager, timeout=180, allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
        record_timing=False,
    )
    start = perf_counter()
    client.execute(cleanup_kc=True, env=env)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code" and cell.source.strip()]
    if not code_cells or any(cell.execution_count is None for cell in code_cells):
        raise AssertionError(f"{path.name}: not all code cells executed")
    images = 0
    for cell in code_cells:
        cell.metadata.pop("execution", None)
        for output in cell.outputs:
            if output.output_type == "error":
                raise AssertionError(f"{path.name}: contains an error output")
            images += int("image/png" in output.get("data", {}))
    if images == 0 and notebook.metadata.get("timegym", {}).get("role") != "source":
        raise AssertionError(f"{path.name}: expected an inline plot")
    nbformat.validate(notebook)
    if html_dir is not None:
        from nbconvert import HTMLExporter

        html_dir.mkdir(parents=True, exist_ok=True)
        body, _ = HTMLExporter().from_notebook_node(notebook)
        (html_dir / f"{path.stem}.html").write_text(body, encoding="utf-8")
    if write:
        nbformat.write(notebook, path)
    return {"notebook": path.name, "code_cells": len(code_cells), "plots": images,
            "seconds": round(perf_counter() - start, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Selected notebook paths; default: every source notebook and tutorial")
    parser.add_argument("--write", action="store_true", help="Save successfully executed outputs in the notebooks")
    parser.add_argument("--group", choices=["all", "source", "tutorials"], default="all",
                        help="Execute all notebooks, or only the source notebooks or the tutorials")
    parser.add_argument("--html-dir", type=Path, help="Also export executed notebooks as HTML for inspection")
    args = parser.parse_args()
    group = args.group
    folders = list(FOLDERS.values()) if group == "all" else [FOLDERS[group]]
    if args.notebooks:
        paths = [p for p in args.notebooks if p.resolve().parent in {f.resolve() for f in folders}]
    else:
        paths = [p for folder in folders for p in sorted(folder.glob("[0-9][0-9]_*.ipynb"))]
    if not paths:
        parser.error("No source notebooks or tutorials found")
    with TemporaryDirectory(prefix="timegym-notebook-") as temporary:
        for path in paths:
            result = execute(path.resolve(), Path(temporary), write=args.write, html_dir=args.html_dir)
            print(json.dumps(result), flush=True)
    print(f"Validated {len(paths)} notebook(s) in independent kernels.", flush=True)


if __name__ == "__main__":
    main()
