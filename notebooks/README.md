# TimeGym notebooks

Start with [00 · Quick start](00_quick_start.ipynb): specify a forecasting
pipeline, see its predictions across controlled scenarios, and inspect its error
distributions. The plots continue the 2021 Quick Start's visual language: gray
observation dots, blue prediction crosses, thin reference curves, and a clear
forecast origin.

Everything lives in this folder: **14 source notebooks generate the package,
12 application notebooks use it**, and the [11 original notebooks](archive/README.md)
retain their development history and saved figures in Git. The historical
archive is excluded from source distributions; the 26 active notebooks are included.

## Open and explore

From a clone or extracted source distribution:

```bash
python -m pip install -e '.[notebooks]'
python -m jupyterlab notebooks
```

Select the Python kernel from that environment and use **Restart Kernel and Run
All Cells**. The active notebooks run independently on CPU, using local data and
explicit seeds. Installing the wheel alone does not install these notebook files.

| Application | Explore |
|---|---|
| [00 · Quick start](00_quick_start.ipynb) | A pipeline, a scenario matrix, and error distributions |
| [20 · Univariate uncertainty](20_univariate_uncertainty.ipynb) | Marginal intervals, coverage, CRPS, and uncertainty that is too narrow |
| [21 · Multivariate forecasting](21_multivariate_forecasting.ipynb) | Several targets, per-target scores, lag features, and target order |
| [22 · Multivariate uncertainty](22_multivariate_uncertainty.ipynb) | Joint paths, dependence, covariance checks, and joint scores |
| [23 · Faults, repairs, and replay](23_faults_repairs_and_replay.ipynb) | Invalid outputs, inverse scaling, declared equivariance, and report replay |
| [24 · Real-data forecasts](24_real_data_forecasts.ipynb) | Training-only transforms and ordered AirPassengers forecast origins |
| [25 · Capability sweeps](25_capability_sweeps.ipynb) | Five scenario families, custom models, context, noise, and repeated cases |
| [26 · Composed and financial stresses](26_composed_and_financial_stresses.ipynb) | Controlled combinations, jumps, GARCH, hidden regimes, price means, and replay |
| [27 · Joint financial uncertainty](27_joint_financial_uncertainty.ipynb) | Multi-target mechanisms, joint paths, dependence faults, aggregate uncertainty, and repair |
| [28 · Write and test a custom model](28_custom_model.ipynb) | Nearest analogues, contract failures, real observations, and a supplied semi-synthetic truth function |
| [29 · Target transforms](29_target_transforms.ipynb) | Log, Box-Cox and difference transforms fitted per context and inverted on the forecast |
| [30 · Reductions](30_reductions.ipynb) | Recursive, direct and multioutput reductions of one regressor, and where their errors differ |

For multiple plots of the same fitted outputs, retain the run:

```python
run = pipeline.run(seeds=[0, 1, 2])
run.plot_forecasts()
run.plot_errors(metric="mean_reference_mse")
report = run.report
```

The plot methods reuse `run.cases`; they do not refit. Calling
`pipeline.plot_tests(...)` or `pipeline.plot_errors(...)` creates a separate run.

## Develop the package here

The source notebooks combine reasoning, editable implementation cells, examples,
and plots. Implementation cells are collapsed for reading; expand them to work
on the code. The [source map](source-map.json) gives each of the 23 package modules
one notebook owner.

| Source notebook | Responsibility |
|---|---|
| [01 · Synthetic data](01_synthetic_data.ipynb) | Scenario generation and conditional references |
| [02 · Real data](02_real_data.ipynb) | Real-data helper |
| [03 · Smoothing](03_smoothing.ipynb) | Moving averages |
| [04 · Feature engineering](04_feature_engineering.ipynb) | Lagged and calendar features |
| [05 · Target engineering](05_target_engineering.ipynb) | Integer and fractional differencing, reversible target transforms |
| [06 · Forecasting models](06_forecasting_models.ipynb) | Native baselines and recursive, direct and multioutput reductions |
| [07 · Pipeline](07_pipeline.ipynb) | Fresh fitting, evaluation, and retained runs |
| [08 · Forecast contracts](08_forecast_contracts.ipynb) | Tasks, forecasts, and reference shapes |
| [09 · Scores and diagnostics](09_scores_and_diagnostics.ipynb) | Scores, case-level intervals, checks, and fault fixtures |
| [10 · Reports and replay](10_reports_and_replay.ipynb) | Structured results and provenance |
| [11 · Plotting](11_plotting.ipynb) | Forecast matrices and case-score distributions |
| [12 · Package](12_package.ipynb) | Public exports and package structure |
| [14 · Controlled processes](14_controlled_processes.ipynb) | Compositions, parameter changes, Merton/GARCH/regime simulation and references |
| [15 · Joint controlled processes](15_joint_processes.ipynb) | Correlated innovations, shared jumps and regimes, CCC-GARCH, and joint reference laws |

The original `%%runwrite` workflow developed code and explanations together.
The current workflow keeps that notebook-first approach with nbdev 3.3.26's
modular exporter. Save notebook edits, then build explicitly:

```bash
python tools/export_notebooks.py
python tools/export_notebooks.py --check
python -m pytest
```

Export reads selected source cells; it does not execute them or include ordinary
demonstration cells. Normal notebook execution does not write package files.
Generated modules should be changed through their source notebooks. See
[contributing](../docs/contributing.md) and the [build guide](../tools/README.md).

After editing an implementation cell, restart and Run All to refresh the chapter's
development bindings and its nearby experiments.
For changes used by another chapter, save, export, and restart the dependent
kernel so it imports the updated package. Export uses saved source, not the
current kernel state or notebook outputs.

## Execute and review

```bash
python notebooks/check_notebooks.py
python notebooks/check_notebooks.py --group source
python notebooks/check_notebooks.py --group demo
python notebooks/check_notebooks.py --write --html-dir /tmp/timegym-notebooks
```

The checker uses fresh kernels with the invoking Python interpreter. It clears
old outputs before execution, rejects uncaught errors and skipped code, and
excludes the archive. `--write` saves the executed outputs; the HTML option makes
plot and narrative review convenient. A single notebook path can also be passed
to the checker while developing it.

## Interpret the evidence

- Point forecasts are predictive means at discrete horizons. Connected
  reference curves and joint sample paths serve different purposes.
- Marginal intervals are not simultaneous prediction regions. Good marginal
  coverage or CRPS can coexist with incorrect joint dependence.
- Known-parameter references have information that fitted models do not.
  The Gaussian AR(1) baseline omits parameter-estimation uncertainty.
- Independent synthetic cases are the replications; dependent horizons and
  targets within one forecast are not additional independent cases. The four
  ordered real-data origins are a descriptive example.
- Valid shapes do not establish forecasting skill. Statistical checks can
  reject a clean ensemble, and accuracy remains descriptive without a justified
  acceptance rule.

These examples complement the [paper's controlled study](../experiments/README.md);
they do not broaden its empirical claims. Historical themes that remain
unfinished are identified in the [archive guide](archive/README.md).
