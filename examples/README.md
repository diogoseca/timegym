# Python examples

The Jupyter notebooks are in [`notebooks/`](../notebooks/README.md),
alongside the package's source notebooks. Start with
[00 · Quick start](../notebooks/00_quick_start.ipynb) for visual forecast matrices
and error distributions. See the [notebook guide](../notebooks/README.md) for
univariate and multivariate uncertainty, real data, diagnosis, and development.

This folder has two terminal examples:

- [`quickstart.py`](quickstart.py): generate controlled cases and inspect a
  diagnostic report.
- [`reproduce_report.py`](reproduce_report.py): replay a saved report case for a
  supported built-in model.

From the project root:

```bash
python examples/quickstart.py
python examples/reproduce_report.py report.json --case-index 0
```

The original example notebook is preserved as
[Quick Start (2021)](../notebooks/archive/Quick%20Start%20%282021%29.ipynb).
