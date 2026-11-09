# TimeGym notebooks

Start with the [Quick start](00_quick_start.ipynb): specify a forecasting
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

| Group | Application | Explore |
|---|---|---|
| Start | [Quick start](00_quick_start.ipynb) | A pipeline, a scenario matrix, and error distributions |
| Find what breaks | [Faults, repairs and replay](23_faults_repairs_and_replay.ipynb) | Four ways to break a correct forecast, a bug caught by a broken promise, and replay |
| | [Write and test a custom model](28_custom_model.ipynb) | The 10-line `LastValue` model, nearest analogues, and real data with and without an answer key |
| Shape the forecast | [Target transforms](29_target_transforms.ipynb) | Why logs help when the swing grows with the level, and why exp(mean log) is too low |
| | [Reductions](30_reductions.ipynb) | Recursive, direct and multioutput: when recursive errors compound |
| | [Multivariate forecasting](21_multivariate_forecasting.ipynb) | Right numbers under the wrong series name, and when extra history doesn't help |
| | [Real-data forecasts](24_real_data_forecasts.ipynb) | No answer key: four origins, three baselines, and a leakage check |
| Measure uncertainty | [Univariate uncertainty](20_univariate_uncertainty.ipynb) | Same predictions, overconfident spread: which scores notice |
| | [Multivariate uncertainty](22_multivariate_uncertainty.ipynb) | Scrambled pairings: every series right, joint risk gone |
| Stress-test | [Capability sweeps](25_capability_sweeps.ipynb) | Five scenario families, custom models, context, noise, and repeated cases |
| | [Composed and financial stresses](26_composed_and_financial_stresses.ipynb) | A ladder of mechanisms, bad luck vs a bad model, and a conversion bug |
| | [Joint financial uncertainty](27_joint_financial_uncertainty.ipynb) | Assets that crash together, and the check that notices a lost link |

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

The package is exported with nbdev 3.3.26's modular exporter. Save notebook
edits, then build explicitly:

```bash
python tools/export_notebooks.py
python tools/export_notebooks.py --check
python -m pytest
```

Export reads selected source cells; it does not execute them or include ordinary
demonstration cells. Normal notebook execution does not write package files.
Generated modules should be changed through their source notebooks. See
[contributing](../docs/contributing.md) and the [build guide](../tools/README.md).

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

These examples complement the [paper's controlled studies](../experiments/README.md);
they do not broaden their empirical claims. Historical themes that remain
unfinished are identified in the [archive guide](archive/README.md).
