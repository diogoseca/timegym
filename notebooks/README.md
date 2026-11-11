# TimeGym notebooks

Start with the [Quick start](00_quick_start.ipynb): one call scores a
forecaster on ten kinds of series at two noise levels, and a second model shows
that each is blind where the other sees. The forecast plots continue the 2021
Quick Start's visual language: gray observation dots, blue prediction crosses,
thin answer-key curves, and a clear forecast origin.

Everything lives in this folder: **15 source notebooks generate the package,
12 application notebooks use it**, and the [11 original notebooks](archive/README.md)
retain their development history and saved figures in Git. The historical
archive is excluded from source distributions; the 27 active notebooks are included.

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
| Start | [Quick start](00_quick_start.ipynb) | Two scorecards, last value against Ridge, and why TimeGym scores against the answer key |
| Find what breaks | [Faults, repairs and replay](23_faults_repairs_and_replay.ipynb) | Four ways to break a correct forecast, a bug caught by a broken promise, and replay |
| | [Write and test a custom model](28_custom_model.ipynb) | The six-line `LastValueForecaster`, nearest analogues, and real data with and without an answer key |
| Shape the forecast | [Target transforms](29_target_transforms.ipynb) | Why logs help when the swing grows with the level, and why exp(mean log) is too low |
| | [Reductions](30_reductions.ipynb) | Recursive, direct and multioutput: when recursive errors compound |
| | [Multivariate forecasting](21_multivariate_forecasting.ipynb) | Right numbers under the wrong series name, and when extra history doesn't help |
| | [Real-data forecasts](24_real_data_forecasts.ipynb) | No answer key: four origins, three baselines, and a leakage check |
| Measure uncertainty | [Univariate uncertainty](20_univariate_uncertainty.ipynb) | Same predictions, overconfident spread: which scores notice |
| | [Multivariate uncertainty](22_multivariate_uncertainty.ipynb) | Scrambled pairings: every series right, joint risk gone |
| Stress-test | [Capability sweeps](25_capability_sweeps.ipynb) | Three simple bets, a weak model versus a broken one, and history and noise sweeps |
| | [Composed and financial stresses](26_composed_and_financial_stresses.ipynb) | A ladder of mechanisms, bad luck vs a bad model, and a conversion bug |
| | [Joint financial uncertainty](27_joint_financial_uncertainty.ipynb) | Assets that crash together, and the check that notices a lost link |

## Develop the package here

The source notebooks combine reasoning, editable implementation cells, examples,
and plots. Implementation cells are collapsed for reading; expand them to work
on the code. The [source map](source-map.json) gives each of the 25 package modules
one notebook owner.

| Source notebook | Responsibility |
|---|---|
| [01 · Synthetic data](01_synthetic_data.ipynb) | Recipes for test series and their answer keys: six kinds of series, what the model sees, measurement noise |
| [02 · Real data](02_real_data.ipynb) | Freeze a stock download in a snapshot; test three simple rules on a hidden year of airline passengers |
| [03 · Smoothing](03_smoothing.ipynb) | Trailing averages: how long a window, why a centered one leaks the future, exponential weights |
| [04 · Feature engineering](04_feature_engineering.ipynb) | Inputs known at forecast time: past values as columns, cycles as circles, calendar columns |
| [05 · Target engineering](05_target_engineering.ipynb) | Differences, logs and Box-Cox: when forecasting the changes helps, and how a forecast is turned back without a bias |
| [06 · Forecasting models](06_forecasting_models.ipynb) | Built-in baselines, which one to beat, and three ways to turn a regressor into a forecaster |
| [07 · Pipeline](07_pipeline.ipynb) | `test`, `predict`, `scorecard` and `run`: a fresh model for every case, and runs kept for plots |
| [08 · Forecast contracts](08_forecast_contracts.ipynb) | What a model sees and returns, the five output checks, and what sample paths say that quantiles can't |
| [09 · Scores and diagnostics](09_scores_and_diagnostics.ipynb) | Which score to read, why the answer key settles comparisons sooner, the five verdicts, and what each check catches |
| [10 · Reports and replay](10_reports_and_replay.ipynb) | Read a report, gate CI on it, save and load it strictly, and replay a case months later |
| [11 · Plotting](11_plotting.ipynb) | Forecast panels and error histograms: read a forecast against the answer key, find the draws a model gets wrong |
| [12 · Package](12_package.ipynb) | Shared guards, which notebook owns each public name, and how an edit becomes package code |
| [13 · Scorecard](13_scorecard.ipynb) | How the scorecard is built and read: ten patterns at two noise levels, the score, named mistakes, seeds |
| [14 · Controlled processes](14_controlled_processes.ipynb) | Series built one ingredient at a time, plus jumps, calm and stormy periods and hidden regimes, each with its answer key |
| [15 · Joint controlled processes](15_joint_processes.ipynb) | Several series with declared links: shared jumps, one hidden regime, CCC-GARCH, and links zero correlation misses |

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
