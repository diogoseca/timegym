# TimeGym v2 manuscript

`main.tex` is the editable v2 manuscript. The generated result fragments contain tables and narrative derived from saved study records; do not edit their numbers by hand. `references.bib` is the single bibliography source, and `claims.md` links each substantive claim to its evidence and boundary. The historical manuscript and source are available from [arXiv:2105.01404v1](https://arxiv.org/abs/2105.01404v1).

From the repository root, install the recorded study environment with Python
3.13 (the lock is `requirements-study.txt`):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-study.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

For development on other supported Python versions, install `.[dev]` instead;
that environment is not the recorded study lock. The canonical reproduction
recipe below writes each new study to a fresh temporary directory and compares
its numerical record with the frozen result before rendering the committed
records. It does not replace the experiment manifests.

```sh
pilot_dir=$(mktemp -d /tmp/timegym-pilot.XXXXXX)
supplement_dir=$(mktemp -d /tmp/timegym-supplement.XXXXXX)
.venv/bin/python -m experiments.run_study --output "$pilot_dir"
.venv/bin/python -m experiments.run_supplement --output "$supplement_dir"
cmp experiments/results/results.json "$pilot_dir/results.json" &&
cmp experiments/results-v2/results.json "$supplement_dir/results.json" &&
.venv/bin/python -m experiments.render_results &&
.venv/bin/python -m experiments.render_supplement &&
.venv/bin/python paper/render_figures.py &&
.venv/bin/python paper/build_standalone.py
```

To render the frozen evidence without rerunning the studies, start with the
`experiments.render_results` command. Verify the generated single-file build
without rewriting it with `.venv/bin/python paper/build_standalone.py --check`.
The builder uses only the Python standard library. It reads the version and
date from the root `CITATION.cff` and stamps them into the standalone title
block; keep that file beside `paper/` when building an extracted source archive.
See the [release guide](../docs/release.md) for tag-driven PDF publication and
Zenodo setup. Its bibliography renderer
accepts the braced, one-field-per-line entries in `references.bib`, sorts by
author and title like `plain`, and links titles to their canonical identifiers.
Unsupported bibliography syntax or missing cited entries fails the build.

For new CPU runs in separate output directories and SHA256 comparisons, follow [the study guide](../experiments/README.md). Both runners limit numerical thread pools to one thread and refuse existing artifacts unless `--force` is explicit.

Both studies run locally on CPU. The original protocol and records remain in `experiments/protocol.json` and `experiments/results/`. The implementation-regression and covariance supplement has its own `experiments/protocol-v2.json` and `experiments/results-v2/`, including case records, summary CSVs, a source/environment manifest, and separate wall-time metadata. Rendering consumes saved records; it does not rerun experiments. The standalone output, `timegym-v2.tex`, includes the generated tables and bibliography, so the single-file standalone build needs no additional source files or BibTeX pass. Compile `main.tex` from this directory using LaTeX, BibTeX, and two further LaTeX passes (`plain` bibliography style).

The original pilot exports three scientific figures in both PNG and vector PDF under `figures/`:

- `fault-alarm-matrix`: observed alarms for each selected fixture, with unsupported covariance cells visibly masked. The clean row shows false alarms.
- `dependence-score-deltas`: paired score changes against permutation strength, with 95% t intervals across independent seeds. Panels have different scales; marginal CRPS differences are numerical roundoff.
- `capability-curves`: mean-reference MSE across three innovation scales, five scenario families, and three unmodified models. Curves are descriptive means over 16 cases; panel scales differ. `capability-summary.csv` includes the means and seed-level intervals.

The supplement adds `supplement-implementation-ablation` (complementary checks on the selected implementation regressions) and `supplement-covariance-holm` (raw versus Holm-adjusted suite rejection rates), also in PNG and PDF. The generated tables retain per-condition counts so the aggregate ablation cannot conceal differences among conditions.

The manuscript now embeds five primary vector figures, designed at print width:

1. [Forecast matrix (SVG)](figures/paper-forecast-matrix.svg): six configurations from five families by three fixed seeds, using gray observation dots, unconnected blue forecast crosses, and thin conditional references. It uses the quick-start model and settings, with paired random walks without drift and with drift (0.03 per step) sharing identical innovations. To remain readable, it shows the final 72 context values and first 48 predictions from each complete 300-context/100-horizon run.
2. [Marginal versus joint dependence (SVG)](figures/paper-dependence.svg): the first predefined pilot case, identical empirical marginals, different joint sample clouds, and saved paired score changes over all 64 evaluation cases.
3. [Diagnostic evidence (SVG)](figures/paper-diagnostic-evidence.svg): the implementation ablation with clean controls and the random-walk-with-drift counterexample where correcting an algorithm worsens forecast loss.

4. [Financial stress matrix (SVG)](figures/paper-financial-stresses.svg): six univariate configurations from application notebook 26, showing diffusion, jumps, GARCH, hidden regimes, and combined regimes/jumps. Filter approximation and output units are explicit. These are illustrative software cases; neither Gaussian fault-study protocol is expanded.
5. [Joint financial uncertainty (SVG)](figures/paper-joint-stresses.svg): notebook 27 preserves each target trajectory while breaking cross-series pairing, revealing changes in joint and aggregate uncertainty. This is a controlled reference demonstration.

`render_figures.py` generates editable SVGs, PNG previews, and self-contained
TikZ scenes from shared plotted coordinates; TikZ type is set by the document. It regenerates the illustrative
quick-start fits and first dependence case; aggregate results are read from
the existing studies and checked against their raw records. It does not change
either study protocol or numerical results. The
[figure provenance record](figures/paper-figure-provenance.json) records the
inputs, seeds, plotted values, settings, and file fingerprints.

`main.tex` inputs these scenes; `build_standalone.py` regenerates
`timegym-v2.tex` with their complete vector commands and the bibliography.
The single-file standalone build therefore needs no external images, raster
fallbacks, or SVG conversion tools. Edit the inputs and rerun the builder;
changes made directly to the generated file are overwritten.
The SVG files are separate reusable assets for websites, slides, and submission
workflows. Their text remains editable. The earlier five research figures remain
supplementary assets, including the full three-scale capability plot and Holm
comparison. The text-only workflow schematic is a table rather than a numbered
main figure.

The studies retain clean false alarms, missed defects, negative controls, and paired repair effects. The first pilot reverses selected output corruptions. The supplement changes implementation branches inside a fitted recursive model and compares complementary diagnostic rules. Repairs are specified by the experimenter; automatic cause identification and repair discovery are outside the claim. The covariance supplement reports a declared four-check Holm family; other exploratory comparisons are not covered by that guarantee. See [the claim ledger](claims.md) and [publication strategy](publication-strategy.md).

The [notebook collection](../notebooks/README.md) contains executable source
notebooks that export the package and application notebooks that explain its
methods, including univariate and joint multivariate
uncertainty. They are teaching examples with their own seeds and smaller studies,
not additional empirical results in this manuscript. Install `.[notebooks]` and
run `python tools/export_notebooks.py --check` and then
`python notebooks/check_notebooks.py` from the repository root to verify source
parity and execute every active notebook in a fresh kernel. Their source and saved
outputs accompany the source distribution. The original notebooks are
preserved separately in `notebooks/archive/` in Git and excluded from both
execution and the source distribution.
