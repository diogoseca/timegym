# Build the package from notebooks

The source notebooks listed in `notebooks/source-map.json` own their generated
Python modules. Edit their marked cells, save the notebooks, then run:

```bash
python tools/export_notebooks.py
python -m pytest
python notebooks/check_notebooks.py
```

The build command reads **saved files**, so unsaved edits in a running kernel
are not exported. Running a notebook does not write package modules. Notebook
execution, source export, and package tests are separate steps.

Use the check mode in CI or before committing:

```bash
python tools/export_notebooks.py --check
```

It generates the modules into a temporary directory and compares their complete
bytes with the package files. A mismatch exits with status 1 without changing
anything; invalid source or ownership exits with status 2. Commands work from
any directory when the script is addressed by its full path; `--root PATH`
selects another checkout explicitly.

## Source ownership

The manifest uses project-relative paths and lists every Python module under
`timegym/`, either as generated source or an explicit handwritten exception:

```json
{
  "schema_version": 1,
  "sources": [
    {
      "notebook": "notebooks/01_example.ipynb",
      "modules": ["timegym/example.py", "timegym/helpers/scaling.py"]
    }
  ],
  "handwritten": []
}
```

Each source notebook declares `{"timegym": {"role": "source"}}` in its
notebook metadata. Its cells have stable, unique IDs. The notebook has one
default module directive, for example `#| default_exp example`, and marks
library cells with `#| export`. A cell for another module can use
`#| export helpers.scaling`. Both modules must belong to that notebook in the
manifest. Use one consistent destination per module: do not mix bare exports
and explicit exports that refer to the same default module.

Ordinary demonstration cells are not exported. A notebook may own several
modules; each module has exactly one owner. Notebook imports can use absolute
`timegym` imports, which nbdev translates to package-relative imports when
appropriate. Generated source includes a relative link to its notebook and
cell IDs, independent of the current directory and temporary build location.

## Validation boundary

The exporter uses the installed nbdev exporter; it does not execute notebook
cells or import generated modules. Every exported path must match its declared
owner. All exports and Python syntax checks complete before any package file
is changed. Each replacement is atomic; the command does not claim a single
filesystem transaction across multiple files if an OS error interrupts the
replacement phase.

Syntax-valid code can still be wrong at runtime. Package tests check the
generated library, and fresh-kernel notebook execution checks the examples.
The dedicated build tests use disposable fixture projects, including failures,
source drift, multiple module ownership, and unexecuted demonstrations.

This uses [nbdev's modular exporter](https://nbdev.fast.ai/tutorials/modular_nbdev.html)
without replacing the project's package build, test runner, or release process.

## Check distributions and documentation

```bash
python -m pip install -e '.[docs,release]'
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

`docs_hooks.py` stages the README and linked repository documents in a temporary
directory, so the same relative links work on GitHub and in the generated site.
It fails on missing local links and does not execute notebooks.

After installing the wheel in a separate environment, run its interpreter with
`-I` to exclude checkout imports:

```bash
/path/to/wheel-environment/bin/python -I tools/smoke_installed.py
```

This checks point and joint-sample forecasts, retained runs, validated report
loading, replay, and the diagnostic demo while deliberately blocking optional
plotting, sklearn, and notebook imports. CI runs the installed-wheel check on
Linux, macOS, and Windows; these jobs must actually pass before claiming support
from those runs. The broader test matrix also includes an older compatible
scientific stack on Python 3.12. Building artifacts does not publish a release.
