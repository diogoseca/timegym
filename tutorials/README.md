# TimeGym tutorials

These twelve notebooks show how to test a forecaster with TimeGym. Each one
imports the package the way your own code would. Read them on the
[documentation site](https://diogoseca.github.io/timegym/docs/tutorials/), or
run them yourself. Start with the [Quick start](01_quick_start.ipynb), then
[Write and test a custom model](02_custom_model.ipynb) to plug in your own.

The package itself is written in the [source notebooks](../source/README.md).
You don't need them to use TimeGym.

## Run them

They run on CPU with local data and explicit seeds; [Run the notebooks](../docs/tutorials.md#run-the-notebooks) gives the two commands. Installing the wheel alone does not install these notebooks.

| Group | Tutorial | Explore |
|---|---|---|
| Start | [Quick start](01_quick_start.ipynb) | Two scorecards, last value against Ridge, and why TimeGym scores against the answer key |
| | [Write and test a custom model](02_custom_model.ipynb) | The six-line `RepeatLast` model, nearest analogues, and real data with and without an answer key |
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
