# Tutorials

TimeGym is for testing a forecasting pipeline before deployment and for
reproducing a failure after one. The twelve tutorials are notebooks that run on
CPU with local data and explicit seeds. Start with the quick start, then write
and test your own model, then pick from the table by the question you have.

## What a clean report covers

A report with no `FAIL` and no `ERROR` does not say the forecasts are accurate;
[How TimeGym decides](concepts/how-timegym-decides.md) lists what it does
establish.

## The tutorials in order

| Group | Tutorial | Question it answers |
| --- | --- | --- |
| Start | [01 · Quick start](../tutorials/01_quick_start.ipynb) | Which kinds of series does my forecaster get right, and which does it get wrong? |
| | [02 · Write and test a custom model](../tutorials/02_custom_model.ipynb) | How do I test a model I wrote myself? |
| Find what breaks | [03 · Faults, repairs and replay](../tutorials/03_faults_repairs_and_replay.ipynb) | What does a broken output look like, and how do I replay it? |
| Shape the forecast | [04 · Target transforms](../tutorials/04_target_transforms.ipynb) | Which transform of the target, and what does inverting it do? |
| | [05 · Reductions](../tutorials/05_reductions.ipynb) | Recursive, direct or multioutput: where do their errors differ? |
| | [06 · Multivariate forecasting](../tutorials/06_multivariate_forecasting.ipynb) | How do I forecast and score several targets? |
| | [07 · Real-data forecasts](../tutorials/07_real_data_forecasts.ipynb) | How do I evaluate on real data at ordered origins? |
| Measure uncertainty | [08 · Univariate uncertainty](../tutorials/08_univariate_uncertainty.ipynb) | Are my intervals and quantiles calibrated? |
| | [09 · Multivariate uncertainty](../tutorials/09_multivariate_uncertainty.ipynb) | Do my joint paths keep the dependence between targets? |
| Stress-test | [10 · Capability sweeps](../tutorials/10_capability_sweeps.ipynb) | Where does each model stop coping as the history shortens or the noise grows? |
| | [11 · Composed and financial stresses](../tutorials/11_composed_and_financial_stresses.ipynb) | How does it handle jumps, GARCH and hidden regimes? |
| | [12 · Joint financial uncertainty](../tutorials/12_joint_financial_uncertainty.ipynb) | Does joint uncertainty survive shared shocks and broken dependence? |

## Run the notebooks

From a clone or extracted source distribution:

```bash
python -m pip install -e '.[notebooks]'
python -m jupyterlab tutorials
```

Select the Python kernel belonging to that environment and use **Restart Kernel
and Run All Cells**. Each notebook runs in its own kernel. The real-data
notebook reads the included AirPassengers CSV; no notebook uses the network
([Market data](api/market-data.md) covers the optional fetch helper).

The tutorials live in `tutorials/` ([tutorials guide](../tutorials/README.md)).
The 15 source notebooks in `source/` are the package's code, not tutorials: they
and the checks that execute every notebook are described in
[Contributing](contributing.md) and the [source guide](../source/README.md).
