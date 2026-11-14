# Python examples

This folder has two terminal examples. The Jupyter tutorials are in
[`tutorials/`](../tutorials/README.md): start with
[01 · Quick start](../tutorials/01_quick_start.ipynb), then
[02 · Write and test a custom model](../tutorials/02_custom_model.ipynb).

- [`quickstart.py`](quickstart.py): generate controlled cases and inspect a
  diagnostic report.
- [`reproduce_report.py`](reproduce_report.py): replay a saved report case for a
  supported built-in model.

From the project root:

```bash
python examples/quickstart.py
python examples/reproduce_report.py report.json --case-index 0
```

The README's figures are redrawn by
[`tools/readme_figures.py`](../tools/readme_figures.py). The original example
notebook is preserved as
[Quick Start (2021)](../source/archive/Quick%20Start%20%282021%29.ipynb).
