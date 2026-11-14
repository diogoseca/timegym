# TimeGym source notebooks

These 15 notebooks are TimeGym's source code. Each one explains a part of the
package and holds the implementation cells that `tools/export_notebooks.py`
writes into `timegym/`. The [source map](source-map.json) gives each of the 25
package modules one owning notebook. To use TimeGym rather than change it, read
the [tutorials](../tutorials/README.md) instead.

The [11 original notebooks](archive/README.md) from 2021 keep the development
history and saved figures in Git. They are excluded from source distributions
and from the notebook checks; the 15 source notebooks and the 12 tutorials are
included.

## Open and edit

From a clone or extracted source distribution:

```bash
python -m pip install -e '.[notebooks]'
python -m jupyterlab source
```

The source notebooks combine reasoning, editable implementation cells, examples,
and plots. Implementation cells are collapsed for reading; expand them to work
on the code.

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

## Develop the package here

Save notebook edits, then follow [Contributing](../docs/contributing.md) and the [build guide](../tools/README.md): export, check, test.

## Execute and review

```bash
python tools/check_notebooks.py
python tools/check_notebooks.py --group source
python tools/check_notebooks.py --group tutorials
python tools/check_notebooks.py --write --html-dir /tmp/timegym-notebooks
```

The checker runs the source notebooks and the tutorials in fresh kernels with
the invoking Python interpreter. It clears old outputs before execution, rejects
uncaught errors and skipped code, and excludes the archive. `--write` saves the
executed outputs; the HTML option makes plot and narrative review convenient. A
single notebook path can also be passed to the checker while developing it.

Historical themes that remain unfinished are identified in the
[archive guide](archive/README.md).
