# TimeGym manuscript

`main.tex` is the editable manuscript. The generated result fragments contain tables and narrative derived from saved study records; do not edit their numbers by hand. `references.bib` is the single bibliography source, and `claims.md` links each substantive claim to its evidence and boundary. The historical manuscript and source are available from [arXiv:2105.01404v1](https://arxiv.org/abs/2105.01404v1).

To rerun either study, follow [the study guide](../experiments/README.md): it creates the recorded Python 3.13 environment from `requirements-study.txt`, writes each new record to a fresh directory and compares it with the frozen one.

To render the frozen evidence without rerunning the studies, use the study
guide's render commands. Verify the generated single-file build
without rewriting it with `.venv/bin/python paper/build_standalone.py --check`.
The builder uses only the Python standard library. It reads the version and
date from the root `CITATION.cff` and stamps them into the standalone title
block; keep that file beside `paper/` when building an extracted source archive.
See the [release guide](../docs/release.md) for tag-driven PDF publication and
Zenodo setup. The builder's bibliography renderer
accepts the braced, one-field-per-line entries in `references.bib`, sorts by
author and title like `plain`, and links titles to their canonical identifiers.
Unsupported bibliography syntax or missing cited entries fails the build.

Rendering consumes saved records; it does not rerun experiments. The standalone output, `timegym.tex`, includes the generated tables and bibliography, so the single-file standalone build needs no additional source files or BibTeX pass. Compile `main.tex` from this directory using LaTeX, BibTeX, and two further LaTeX passes (`plain` bibliography style).

To compile the standalone PDF, run `latexmk -pdf -outdir=build timegym.tex` from this directory. It needs latexmk, the standard LaTeX and TikZ packages, and the cm-super fonts; MacTeX includes them all, and on Debian or Ubuntu they are `apt-get install latexmk texlive-latex-recommended texlive-pictures texlive-fonts-recommended cm-super`. Without cm-super the T1-encoded fonts fall back to Metafont bitmaps, which can fail to build. The `paper` job in the checks workflow compiles the same file when `paper/`, `CITATION.cff` or the workflow changed, and on every release, and uploads `timegym.pdf` as the `paper-pdf` artifact.

Study 1 exports one scientific figure in both PNG and vector PDF under `figures/`:

- `capability-curves`: mean-reference MSE across three innovation scales, five scenario families, and three unmodified models. Curves are descriptive means over 16 cases; panel scales differ. `capability-summary.csv` includes the means and seed-level intervals, and for each row the unavoidable error and the 0 to 100 score that the manuscript's capability table prints at the lowest and highest scale.

The manuscript embeds three vector figures, designed at print width:

1. [Scorecard (SVG)](figures/paper-scorecard.svg): the package scorecard of the quick start's Ridge pipeline (`Pipeline(window_length=24)` with `StandardScaler` and `Ridge(alpha=1.0)`), redrawn in print style. It uses the scorecard defaults: ten kinds of series by two noise levels (sd 0.5 and 2), 10 instances per cell, seed 0. Each row gives the package's sentences on what can and cannot be predicted, one panel per noise level with the cell's score and verdict and the series of instance 1 (gray observation dots, black answer key, blue forecast crosses, dashed forecast origin), and the mistakes the package names. Every number and label is read from the `Scorecard` object.
2. [Marginal versus joint dependence (SVG)](figures/paper-dependence.svg): the first predefined Study 1 case, identical empirical marginals, different joint sample clouds.
3. [Diagnostic evidence (SVG)](figures/paper-diagnostic-evidence.svg): the implementation ablation with clean controls, including the exploratory answer-key alarm of the Study 2 follow-up (marked with a dagger), and the random-walk-with-drift counterexample where correcting an algorithm worsens forecast loss.

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
The SVG files are separate, editable assets for websites, slides and submission
workflows.

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
preserved in Git at the [v1.4.0 tag](https://github.com/diogoseca/timegym/tree/v1.4.0/source/archive) and are not part of
execution or the source distribution.
