# Contributing

Install the development and notebook dependencies:

```bash
python -m pip install -e '.[dev,notebooks]'
```

## Change the library in its source notebook

All 25 modules under `timegym/` are generated from the 15 source notebooks
listed in [the source map](../source/source-map.json). Edit the marked
implementation cells there; expand collapsed cells when needed. Keep explanatory
examples beside the implementation. The 12 tutorials in `tutorials/` import the
package the way a user would and demonstrate its use.

```bash
# Save notebook edits before exporting.
python tools/export_notebooks.py
python tools/export_notebooks.py --check
python -m pytest
python examples/quickstart.py
```

Do not edit generated modules; the next export replaces them. The [build guide](../tools/README.md) gives the directives, ownership rules and what check mode compares.

Within a chapter, restart and Run All after implementation edits so the development
bindings and nearby experiments use the new definitions. To use a change from
another chapter, save and export first,
then restart the dependent kernel so imports load the updated package. Export
reads saved source cells, not kernel state or stored outputs.

## Verify notebooks and plots

Run every notebook in a fresh kernel with the checker the [source guide](../source/README.md#execute-and-review) describes. Successful execution does not
prove that labels, scales or explanations are useful, so inspect the rendered
plots and explanations. Seeded results are reproducible within an environment,
while numerical and rendering versions can affect output. Use discrete markers
for point predictions and observations, distinguish conditional references from
realized futures, and label marginal intervals separately from joint paths.
Keep seeds explicit, use local data, and preserve uncertainty caveats. Keep
implementation cells exportable and demonstration cells unmarked. Run
`python -m build` before packaging changes.

Source notebooks [14](../source/14_controlled_processes.ipynb) and
[15](../source/15_joint_processes.ipynb) explain and build the univariate and
joint stress generators that tutorials 11 and 12 use.

## Check the documentation site

```bash
python -m pip install -e '.[docs]'
python -m mkdocs build --strict
python tools/check_site_links.py site
```

`--strict` checks Markdown links but not the HTML rendered from notebooks; the
link checker covers both. Notebook links are written relative to the notebook's
own folder, and `tools/docs_hooks.py` points them at the built pages, or at
GitHub for files the site does not render. The tutorial groups are listed in
`mkdocs.yml`, the README, the [tutorials guide](../tutorials/README.md) and the
[tutorials page](tutorials.md); `tests/test_docs_navigation.py` keeps the four
in step.

Each API page except Market data ends with a Signatures section that
mkdocstrings generates from the docstrings; `default_scenarios` and `get_stock`
keep hand-written descriptions because griffe cannot load `timegym.data`. The
docstrings live in the source notebooks, so edit the owning
notebook and export it; never edit `timegym/*.py`. Members without a docstring
are left out of the page, so `tests/test_docs_signatures.py` fails when a public
class, method, property or function lacks one, or when a public name in the
[API overview](reference.md) has no `:::` directive on the page its row links
to. The [capability map](concepts/capability-map.md) is checked by
`tests/test_docs_catalogue.py` against the scenario families, the diagnostics
catalogue and the tutorials it links.

## Preserve the scientific contract

New scenarios require explicit conditioning assumptions and independent
conditional-simulation checks. New metrics require hand-computable fixtures and
shape validation. New statistical diagnostics must declare their null, unit of
replication, calibration method and multiplicity policy. Keep false alarms and
negative results in the study artifacts.

The two studies are specified in `experiments/protocol.json` and
`experiments/protocol-v2.json`. Change protocol
and artifact versions deliberately when an experiment changes; do not tune
thresholds or choose seeds after seeing evaluation results. Regenerate tables and
figures from results, then run `python paper/build_standalone.py`.
The standalone is generated from `main.tex`, `references.bib`, result fragments,
and vector scenes. Run `python paper/build_standalone.py --check` to detect
drift without rewriting it; CI runs this check beside the package export check.

## Report a problem

Start with the [quick start](../tutorials/01_quick_start.ipynb) and the [tutorials](tutorials.md).
For a diagnostic failure, include the JSON report, exact package versions, model
configuration, scenario manifest and seed when opening an
[issue](https://github.com/diogoseca/timegym/issues). Avoid including private
training data or credentials.
