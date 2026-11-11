# TimeGym tutorials

These twelve notebooks show how to test a forecaster with TimeGym. Each one
imports the package the way your own code would. Read them on the
[documentation site](https://diogoseca.github.io/timegym/docs/tutorials/), or
run them yourself. Start with the [Quick start](01_quick_start.ipynb), then
[Write and test a custom model](02_custom_model.ipynb) to plug in your own.

The package itself is written in the [source notebooks](../source/README.md).
You don't need them to use TimeGym.

## Run them

From a clone or an extracted source distribution:

```bash
python -m pip install -e '.[notebooks]'
python -m jupyterlab tutorials
```

Select the Python kernel from that environment and use **Restart Kernel and Run
All Cells**. Each tutorial runs on its own, on CPU, with local data and explicit
seeds. Installing the wheel alone does not install these notebooks.

| Group | Tutorial | Explore |
|---|---|---|
| Start | [Quick start](01_quick_start.ipynb) | Two scorecards, last value against Ridge, and why TimeGym scores against the answer key |
| | [Write and test a custom model](02_custom_model.ipynb) | The six-line `LastValueForecaster`, nearest analogues, and real data with and without an answer key |
| Find what breaks | [Faults, repairs and replay](03_faults_repairs_and_replay.ipynb) | Four ways to break a correct forecast, a bug caught by a broken promise, and replay |
| Shape the forecast | [Target transforms](04_target_transforms.ipynb) | Why logs help when the swing grows with the level, and why exp(mean log) is too low |
| | [Reductions](05_reductions.ipynb) | Recursive, direct and multioutput: when recursive errors compound |
| | [Multivariate forecasting](06_multivariate_forecasting.ipynb) | Right numbers under the wrong series name, and when extra history doesn't help |
| | [Real-data forecasts](07_real_data_forecasts.ipynb) | No answer key: four origins, three baselines, and a leakage check |
| Measure uncertainty | [Univariate uncertainty](08_univariate_uncertainty.ipynb) | Same predictions, overconfident spread: which scores notice |
| | [Multivariate uncertainty](09_multivariate_uncertainty.ipynb) | Scrambled pairings: every series right, joint risk gone |
| Stress-test | [Capability sweeps](10_capability_sweeps.ipynb) | Three simple bets, a weak model versus a broken one, and history and noise sweeps |
| | [Composed and financial stresses](11_composed_and_financial_stresses.ipynb) | A ladder of mechanisms, bad luck vs a bad model, and a conversion bug |
| | [Joint financial uncertainty](12_joint_financial_uncertainty.ipynb) | Assets that crash together, and the check that notices a lost link |

The forecast plots continue the 2021 Quick Start's visual language: gray
observation dots, blue prediction crosses, thin answer-key curves, and a clear
forecast origin.

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
they do not broaden their empirical claims.
