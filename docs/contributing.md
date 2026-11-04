# Contributing

Install the development and notebook dependencies:

```bash
python -m pip install -e '.[dev,notebooks]'
```

## Change the library in its source notebook

All 23 modules under `timegym/` are generated from the 14 source notebooks
listed in [the source map](../notebooks/source-map.json). Edit the marked
implementation cells there; expand collapsed cells when needed. Keep explanatory
examples beside the implementation. Application notebooks 00 and 20–30 import
the package and demonstrate its use.

The original notebook-first approach used `%%runwrite` to execute and append
cells to modules. The current build uses nbdev 3.3.26 to export selected saved
cells into complete modules. Normal notebook execution does not write modules.

```bash
# Save notebook edits before exporting.
python tools/export_notebooks.py
python tools/export_notebooks.py --check
python -m pytest
python examples/quickstart.py
```

The explicit export command validates declared module ownership and syntax
before replacing files. Check mode reports drift without writing. Do not make
lasting edits directly in generated modules: the next export replaces them.
See [the build guide](../tools/README.md) for directives and ownership rules.

Within a chapter, restart and Run All after implementation edits so the development
bindings and nearby experiments use the new definitions. To use a change from
another chapter, save and export first,
then restart the dependent kernel so imports load the updated package. Export
reads saved source cells, not kernel state or stored outputs.

## Verify notebooks and plots

```bash
python notebooks/check_notebooks.py --group source
python notebooks/check_notebooks.py --group demo
python notebooks/check_notebooks.py --write --html-dir /tmp/timegym-notebooks
```

The checker executes active notebooks in fresh independent kernels with the
invoking Python interpreter. It clears stale outputs first, rejects uncaught
errors or skipped code, and excludes the archive. Successful execution does not
prove that labels, scales or explanations are useful, so inspect the rendered
plots and explanations. Seeded results are reproducible within an environment,
while numerical and rendering versions can affect output. Use discrete markers
for point predictions and observations, distinguish conditional references from
realized futures, and label marginal intervals separately from joint paths.
Keep seeds explicit, use local data, and preserve uncertainty caveats. Keep
implementation cells exportable and demonstration cells unmarked. Run
`python -m build` before packaging changes.

Source notebooks [14](../notebooks/14_controlled_processes.ipynb) and
[15](../notebooks/15_joint_processes.ipynb) explain and build the univariate and
joint stress generators that tutorials 26 and 27 use.

## Check the documentation site

```bash
python -m pip install -e '.[docs]'
python -m mkdocs build --strict
python tools/check_site_links.py site
```

`--strict` checks Markdown links but not the HTML rendered from notebooks; the
link checker covers both. Notebook links between tutorials are written relative
to `notebooks/`, and `tools/docs_hooks.py` points them at the built pages, or at
GitHub for files the site does not render. The tutorial groups are listed in
`mkdocs.yml`, the README and the [notebook guide](../notebooks/README.md);
`tests/test_docs_navigation.py` keeps the three in step.

## Preserve the scientific contract

New scenarios require explicit conditioning assumptions and independent
conditional-simulation checks. New metrics require hand-computable fixtures and
shape validation. New statistical diagnostics must declare their null, unit of
replication, calibration method and multiplicity policy. Keep false alarms and
negative results in the study artifacts.

The current study is specified in `experiments/protocol.json`. Change protocol
and artifact versions deliberately when an experiment changes; do not tune
thresholds or choose seeds after seeing evaluation results. Regenerate tables and
figures from results, then run `python paper/build_standalone.py`.
The standalone is generated from `main.tex`, `references.bib`, result fragments,
and vector scenes. Run `python paper/build_standalone.py --check` to detect
drift without rewriting it; CI runs this check beside the package export check.

## Report a problem

Start with the [quick start](../README.md) and the [tutorials](notebooks.md).
For a diagnostic failure, include the JSON report, exact package versions, model
configuration, scenario manifest and seed when opening an
[issue](https://github.com/diogoseca/timegym/issues). Avoid including private
training data or credentials.
