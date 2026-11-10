# Tutorials

TimeGym is for testing a forecasting pipeline before deployment and for
reproducing a failure after one. The twelve tutorials are notebooks that run on
CPU with local data and explicit seeds. Start with the quick start, then pick by
what you want to check.

## What do you want to check?

| I want to | Start with | Then look up |
| --- | --- | --- |
| See what a report says about my pipeline | [00 · Quick start](../notebooks/00_quick_start.ipynb) | [Pipeline and reports](api/pipeline.md) |
| Plug in my own model and check its outputs | [28 · Write and test a custom model](../notebooks/28_custom_model.ipynb) | [Tasks and forecasts](api/tasks-and-forecasts.md) |
| Tell a broken output from a weak model, and replay a failure | [23 · Faults, repairs and replay](../notebooks/23_faults_repairs_and_replay.ipynb) | [Scores and diagnostics](api/diagnostics.md), [Fault fixtures](api/fault-fixtures.md) |
| Choose a target transform or a reduction | [29 · Target transforms](../notebooks/29_target_transforms.ipynb), [30 · Reductions](../notebooks/30_reductions.ipynb) | [Data preparation](api/data-preparation.md), [Models and reductions](api/models.md) |
| Forecast several targets, or real data at several origins | [21 · Multivariate forecasting](../notebooks/21_multivariate_forecasting.ipynb), [24 · Real-data forecasts](../notebooks/24_real_data_forecasts.ipynb) | [Tasks and forecasts](api/tasks-and-forecasts.md) |
| Check intervals, quantiles or sample paths | [20 · Univariate uncertainty](../notebooks/20_univariate_uncertainty.ipynb), [22 · Multivariate uncertainty](../notebooks/22_multivariate_uncertainty.ipynb) | [Scores and diagnostics](api/diagnostics.md) |
| Stress a model with jumps, volatility or regime changes | [25 · Capability sweeps](../notebooks/25_capability_sweeps.ipynb), [26 · Composed and financial stresses](../notebooks/26_composed_and_financial_stresses.ipynb), [27 · Joint financial uncertainty](../notebooks/27_joint_financial_uncertainty.ipynb) | [Composed and joint processes](api/processes.md), [Synthetic scenarios](stress-cases.md) |

## What a clean report covers

A report with no `FAIL` and no `ERROR` does not say the forecasts are accurate;
[How TimeGym decides](concepts/how-timegym-decides.md) lists what it does
establish.

## The tutorials in order

The groups follow a path: a first run, then the failures TimeGym exists to
catch, then modeling choices, uncertainty and stress tests.

| Group | Tutorial | Question it answers |
| --- | --- | --- |
| Start | [00 · Quick start](../notebooks/00_quick_start.ipynb) | Does my pipeline behave sensibly across controlled series? |
| Find what breaks | [23 · Faults, repairs and replay](../notebooks/23_faults_repairs_and_replay.ipynb) | What does a broken output look like, and how do I replay it? |
| | [28 · Write and test a custom model](../notebooks/28_custom_model.ipynb) | How do I test a model I wrote myself? |
| Shape the forecast | [29 · Target transforms](../notebooks/29_target_transforms.ipynb) | Which transform of the target, and what does inverting it do? |
| | [30 · Reductions](../notebooks/30_reductions.ipynb) | Recursive, direct or multioutput: where do their errors differ? |
| | [21 · Multivariate forecasting](../notebooks/21_multivariate_forecasting.ipynb) | How do I forecast and score several targets? |
| | [24 · Real-data forecasts](../notebooks/24_real_data_forecasts.ipynb) | How do I evaluate on real data at ordered origins? |
| Measure uncertainty | [20 · Univariate uncertainty](../notebooks/20_univariate_uncertainty.ipynb) | Are my intervals and quantiles calibrated? |
| | [22 · Multivariate uncertainty](../notebooks/22_multivariate_uncertainty.ipynb) | Do my joint paths keep the dependence between targets? |
| Stress-test | [25 · Capability sweeps](../notebooks/25_capability_sweeps.ipynb) | Where does the model stop coping as context and noise change? |
| | [26 · Composed and financial stresses](../notebooks/26_composed_and_financial_stresses.ipynb) | How does it handle jumps, GARCH and hidden regimes? |
| | [27 · Joint financial uncertainty](../notebooks/27_joint_financial_uncertainty.ipynb) | Does joint uncertainty survive shared shocks and broken dependence? |

## Run the notebooks

From a clone or extracted source distribution:

```bash
python -m pip install -e '.[notebooks]'
python -m jupyterlab notebooks
```

Select the Python kernel belonging to that environment and use **Restart Kernel
and Run All Cells**. Each notebook runs in its own kernel. The real-data
notebook reads the included AirPassengers CSV; no notebook uses the network
([Market data](api/market-data.md) covers the optional fetch helper).

The 15 source notebooks that generate the package, and the checks that execute
every notebook, are described in [Contributing](contributing.md) and the
[notebook guide](../notebooks/README.md).

## Bring your own model

[28 · Write and test a custom model](../notebooks/28_custom_model.ipynb) builds a
k-nearest-analogues forecaster in three steps: write `fit(context)` and
`predict(future_index)`, test it on known scenarios with `Pipeline.test`, then
forecast real observations and a semi-synthetic series with supplied truth.
Whole historical continuations supply empirical joint paths; their average
supplies point predictions. An injected missing-row fault demonstrates a
contract `FAIL`, while valid but inaccurate forecasts retain descriptive scores.

![Nearest-analogue forecasts on Gaussian noise, linear trend, and seasonal scenarios: gray observations, blue prediction crosses, known-mean curves, and passing forecast contracts.](../notebooks/assets/custom-model-knn.png)

This revives the 2021 KNN figure's visual comparison using the plotting
helpers. It does not reproduce the original recursive implementation. Contract
success does not establish accuracy: raw historical continuations cannot
extrapolate a rising level beyond their observed range.

For raw real data, use `Case(reference=None, ...)`; unavailable reference checks
remain `UNSUPPORTED`. For a declared Gaussian semi-synthetic process,
`Reference.from_function(truth, index, covariance=..., target_names=...)`
evaluates a known mean function once and validates its explicit covariance.
The function returns `(horizon, targets)` values or an exactly aligned
DataFrame. The tutorial's fixed curve comes from AirPassengers and is truth
only for the constructed process, not an estimate promoted to the true law
of the historical data. Custom models and truth functions require the notebook
code for reproduction; automatic report replay does not reconstruct them.
