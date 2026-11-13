# TimeGym manuscript

`main.tex` is the editable manuscript. The generated result fragments contain tables and narrative derived from saved study records; do not edit their numbers by hand. `references.bib` is the single bibliography source, and `claims.md` links each substantive claim to its evidence and boundary. The historical manuscript and source are available from [arXiv:2105.01404v1](https://arxiv.org/abs/2105.01404v1).

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
Zenodo setup. The builder's bibliography renderer
accepts the braced, one-field-per-line entries in `references.bib`, sorts by
author and title like `plain`, and links titles to their canonical identifiers.
Unsupported bibliography syntax or missing cited entries fails the build.

For new CPU runs in separate output directories and SHA256 comparisons, follow [the study guide](../experiments/README.md).

Both studies run locally on CPU. The Study 1 protocol and records are in `experiments/protocol.json` and `experiments/results/`. Study 2 has its own `experiments/protocol-v2.json` and `experiments/results-v2/`, including case records, summary CSVs, a source/environment manifest, and separate wall-time metadata. An exploratory follow-up to Study 2, chosen after its results were known, has its own `experiments/protocol-v2-followup.json` and `experiments/results-v2-followup/`; the renderers read it for the answer-key alarm and the other numbers it supplies, and `python -m experiments.run_followup` reproduces it. A second exploratory record (protocol 2.2), chosen after the follow-up was known, has `experiments/protocol-v2-addendum.json` and `experiments/results-v2-addendum/`; it reads only stored Study 2 values, supplies the suite-level multiplicity tests and the paired answer-key check, and `python -m experiments.run_addendum` reproduces it. Rendering consumes saved records; it does not rerun experiments. The standalone output, `timegym.tex`, includes the generated tables and bibliography, so the single-file standalone build needs no additional source files or BibTeX pass. Compile `main.tex` from this directory using LaTeX, BibTeX, and two further LaTeX passes (`plain` bibliography style).

To compile the standalone PDF, run `latexmk -pdf -outdir=build timegym.tex` from this directory. It needs latexmk, the standard LaTeX and TikZ packages, and the cm-super fonts; MacTeX includes them all, and on Debian or Ubuntu they are `apt-get install latexmk texlive-latex-recommended texlive-pictures texlive-fonts-recommended cm-super`. Without cm-super the T1-encoded fonts fall back to Metafont bitmaps, which can fail to build. The `paper` job in the checks workflow compiles the same file on every push and uploads `timegym.pdf` as the `paper-pdf` artifact.

Study 1 exports three scientific figures in both PNG and vector PDF under `figures/`:

- `fault-alarm-matrix`: observed alarms for each selected fixture, with unsupported covariance cells visibly masked. The clean row shows false alarms.
- `dependence-score-deltas`: paired score changes against permutation strength, with 95% t intervals across independent seeds. Panels have different scales; marginal CRPS differences are numerical roundoff.
- `capability-curves`: mean-reference MSE across three innovation scales, five scenario families, and three unmodified models. Curves are descriptive means over 16 cases; panel scales differ. `capability-summary.csv` includes the means and seed-level intervals, and for each row the unavoidable error and the 0 to 100 score that the manuscript's capability table prints at the lowest and highest scale.

Study 2 adds `supplement-implementation-ablation` (complementary checks on the selected implementation regressions) and `supplement-covariance-holm` (raw versus Holm-adjusted suite rejection rates), also in PNG and PDF. The generated tables retain per-condition counts so the aggregate ablation cannot conceal differences among conditions.

The manuscript embeds five vector figures, designed at print width; figures 4
and 5 are in Appendix A:

1. [Scorecard (SVG)](figures/paper-scorecard.svg): the package scorecard of the quick start's Ridge pipeline (`Pipeline(window_length=24)` with `StandardScaler` and `Ridge(alpha=1.0)`), redrawn in print style. It uses the scorecard defaults: ten kinds of series by two noise levels (sd 0.5 and 2), 10 instances per cell, seed 0. Each row gives the package's sentences on what can and cannot be predicted, one panel per noise level with the cell's score and verdict and the series of instance 1 (gray observation dots, black answer key, blue forecast crosses, dashed forecast origin), and the mistakes the package names. Every number and label is read from the `Scorecard` object.
2. [Marginal versus joint dependence (SVG)](figures/paper-dependence.svg): the first predefined Study 1 case, identical empirical marginals, different joint sample clouds, and saved paired score changes over all 64 evaluation cases.
3. [Diagnostic evidence (SVG)](figures/paper-diagnostic-evidence.svg): the implementation ablation with clean controls, including the exploratory answer-key alarm of the Study 2 follow-up (marked with a dagger), and the random-walk-with-drift counterexample where correcting an algorithm worsens forecast loss.

4. [Financial stress matrix (SVG)](figures/paper-financial-stresses.svg): six univariate configurations from tutorial 11 (composed and financial stresses), three series each (their fixed seeds are recorded in the provenance file), showing diffusion, jumps, GARCH, hidden regimes, and combined regimes/jumps. Filter approximation and output units are explicit. These are illustrative software cases; neither Gaussian fault-study protocol is expanded.
5. [Joint financial uncertainty (SVG)](figures/paper-joint-stresses.svg): tutorial 12 (joint financial uncertainty) preserves each target trajectory while breaking cross-series pairing, revealing changes in joint and aggregate uncertainty. This is a controlled reference demonstration.

`render_figures.py` generates editable SVGs, PNG previews, and self-contained
TikZ scenes from shared plotted coordinates; TikZ type is set by the document. It regenerates the
scorecard and the first dependence case; aggregate results are read from
the existing studies and checked against their raw records. It does not change
either study protocol or numerical results. The
[figure provenance record](figures/paper-figure-provenance.json) records the
inputs, seeds, plotted values, settings, and file fingerprints; each figure's
entry has an `inputs_sha256` record of its own inputs.

`render_figures.py --only NAME [NAME ...]` renders only the named figures (for
example `--only paper-scorecard`). It replaces those figures' entries in the
existing provenance record and leaves the other entries and the top-level
`inputs_sha256` unchanged. The partial render exists because not every figure
reproduces byte for byte on every platform: Matplotlib PNG bytes differ between
operating systems even when the SVG and TikZ are identical, and the dependence
figure re-creates a Study 1 case and compares it with the records made on the
recording machine. A partial render updates one figure without rewriting the
others.

`main.tex` inputs these scenes; `build_standalone.py` regenerates
`timegym.tex` with their complete vector commands and the bibliography.
The single-file standalone build therefore needs no external images, raster
fallbacks, or SVG conversion tools. Edit the inputs and rerun the builder;
changes made directly to the generated file are overwritten.
The SVG files are separate reusable assets for websites, slides, and submission
workflows. Their text remains editable. The five study figures listed above are
supplementary assets, including the full three-scale capability plot and Holm
comparison. The text-only workflow schematic is a table rather than a numbered
main figure.

The studies retain clean false alarms, missed defects, negative controls, and paired repair effects. Study 1 reverses selected output corruptions. Study 2 changes implementation branches inside a fitted recursive model and compares complementary diagnostic rules. Repairs are specified by the experimenter; automatic cause identification and repair discovery are outside the claim. For its covariance conditions, Study 2 reports a declared four-check Holm family; other exploratory comparisons are not covered by that guarantee. See [the claim ledger](claims.md) and [publication strategy](publication-strategy.md).

The [source notebooks](../source/README.md) export the package and the
[tutorials](../tutorials/README.md) explain its
methods, including univariate and joint multivariate
uncertainty. They are teaching examples with their own seeds and smaller studies,
not additional empirical results in this manuscript. Install `.[notebooks]` and
run `python tools/export_notebooks.py --check` and then
`python tools/check_notebooks.py` from the repository root to verify source
parity and execute every source notebook and tutorial in a fresh kernel. Their source and saved
outputs accompany the source distribution. The original notebooks are
preserved separately in `source/archive/` in Git and excluded from both
execution and the source distribution.
