# TimeGym

**A gym for time series forecasting algorithms.**

[![Documentation](https://img.shields.io/badge/docs-diogoseca.github.io%2Ftimegym-black)](https://diogoseca.github.io/timegym/)
[![Checks](https://github.com/diogoseca/timegym/actions/workflows/checks.yml/badge.svg)](https://github.com/diogoseca/timegym/actions/workflows/checks.yml)
[![PyPI](https://img.shields.io/pypi/v/timegym)](https://pypi.org/project/timegym/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[OpenAI Gym](https://arxiv.org/abs/1606.01540) gave reinforcement-learning
agents a standard set of environments to train and be tested in. TimeGym does
the same for forecasters. It generates time series whose future it already
knows the odds of: pure noise, a trend, a season, a level pulled back to its
mean, a random walk. It hands your forecaster the past, asks for the future,
and scores the answer against the truth. One picture shows what your algorithm
nails and where it falls over.

[![Six kinds of series, three draws each: the data as gray dots, the forecast as blue crosses, the answer key as a thin line](notebooks/assets/quick-start-matrix.png)](#sixty-seconds)

**Why not just backtest?** A backtest compares your forecast with the one
future that happened. That future is noisy, so a lucky model looks good and a
sound one can look bad. TimeGym made the data, so it holds the answer key: the
expected value and the spread of every future step. It scores your forecast
against that key, which measures skill rather than luck.

**A score can also hide a bug.** Put the right numbers on the wrong dates and
any scorer that compares arrays gives them exactly the same error. TimeGym
checks the dates, the series and the shape first, and fails the forecast with
the reason before it scores anything.

[![A forecast with the right values on the wrong dates: the same error, a different diagnosis](notebooks/assets/alignment-demo.png)](https://diogoseca.github.io/timegym/notebooks/23_faults_repairs_and_replay/)

**[Documentation](https://diogoseca.github.io/timegym/)** ·
[Quick start](https://diogoseca.github.io/timegym/notebooks/00_quick_start/) ·
[Tutorials](https://diogoseca.github.io/timegym/docs/notebooks/) ·
[API](https://diogoseca.github.io/timegym/docs/reference/) ·
[Changelog](CHANGELOG.md)

## Install

```bash
pip install timegym
```

<!-- BEGIN PYTHON SUPPORT -->
Python 3.11–3.15 (`>=3.11,<3.16`), following pandas support.
<!-- END PYTHON SUPPORT -->

## Sixty seconds

Wrap your forecaster in a `Pipeline` and plot it against every kind of series.
The built-in last-value forecaster stands in for yours here:

```python
from timegym import NaiveForecaster, Pipeline

pipeline = Pipeline(NaiveForecaster())   # put your own forecaster here
fig = pipeline.plot_tests()
fig.savefig("forecasts.png")
```

You get a picture like the one at the top. Each row is a kind of series and
each column a fresh draw of it. Gray dots are the data, blue crosses the
forecast, and the thin line is the answer key: the best forecast anyone could
make from that past. Where the crosses sit on the line, your model nails it;
where they wander off, it doesn't.

### More things to do

Keep the forecasts, so several views share the same fits:

```python
run = pipeline.run(seeds=range(20))
run.plot_forecasts()
```

See how the error is spread across those 20 draws of each kind of series:

```python
run.plot_errors(metric="mse")
```

`mse` is the error against what actually happened. The default,
`mean_reference_mse`, is the error against the answer key: the part your model
could have avoided.

Get the scores as a table:

```python
import pandas as pd

scores = pd.DataFrame(
    {"scenario": case.scenario, "seed": case.seed, **case.metrics}
    for case in run.report.cases
)
scores.groupby("scenario")[["mse", "mean_reference_mse"]].mean()
```

Stop a CI build when a forecast is malformed or the model crashes:

```python
run.report.assert_passed()   # raises on FAIL or ERROR and lists each one
```

Save every case to analyse later, or to replay one on another machine:

```python
run.report.to_json("report.json")
```

The file records each case's scenario, seed and scores, every check's verdict
and the numbers behind it, the pipeline's settings, and the Python and library
versions, so a failure with a built-in model can be replayed from the file
alone.

Change the size of the test, or pick your own series. The same seed always
gives the same series; when tuning a model, 20 or 30 seeds per kind of series
keep the score from rewarding luck on a few draws:

```python
from timegym import Scenario

suite = [
    Scenario("seasonal", context_length=240, horizon=24),
    Scenario("random_walk", parameters={"drift": 0.1}),
]
report = pipeline.test(suite, seeds=range(30))
```

## Bring your own model

A forecaster is any Python object with two methods. TimeGym calls them once
per test series:

- `fit(context)` receives the past: a pandas DataFrame with one row per time
  step and one column per series (a single series is one column, `y`). Learn
  whatever you need from it.
- `predict(future_index)` receives the time labels to forecast, for example the
  12 steps after the context ends. Return a `timegym.Forecast` holding one row
  of values for each of those labels.

There is nothing to subclass; `Forecast` is only the container for the answer.
This model forecasts that every future value equals the last one it saw:

```python
import numpy as np
from timegym import Forecast, Pipeline

class LastValue:
    def fit(self, context):            # DataFrame: one row per past step, one column per target
        self.last_ = context.iloc[-1].to_numpy()
        self.targets_ = tuple(context.columns)
        return self

    def predict(self, future_index):   # the time labels to forecast
        point = np.tile(self.last_, (len(future_index), 1))   # (steps, targets)
        return Forecast(index=future_index, point=point, target_names=self.targets_)

report = Pipeline(LastValue).test()
```

Pass the class, not an instance: TimeGym builds a fresh model for every series,
so nothing learned on one leaks into the next. To set parameters, pass a
factory, as in `Pipeline(model_factory=lambda: MyModel(window=48))`.

On a random walk, this naive forecast is hard to beat: the best guess for
tomorrow is today's value. On a trend or a season it is plainly wrong, and the
scores say so.

The forecast must use exactly the `future_index` it was given, the same column
names in the same order, one row per step and finite numbers. Break any of these
and the case fails with the reason, for example `contract_index_mismatch` when
the dates are shifted. A `Forecast` can also carry quantiles or sample paths
for uncertainty, and TimeGym scores those too.
[Write and test a custom model](https://diogoseca.github.io/timegym/notebooks/28_custom_model/)
walks through it step by step, up to a nearest-analogues forecaster on real
data.

[![A nearest-analogues forecaster on white-noise, trend and seasonal scenarios, two seeds each: observations as dots, forecasts as crosses, known conditional means as lines](notebooks/assets/custom-model-knn.png)](https://diogoseca.github.io/timegym/notebooks/28_custom_model/)

scikit-learn regressors plug in directly. TimeGym feeds them the last
`window_length` values as features and turns their predictions into a
multi-step forecast, fitting any scaler on the training window only:

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from timegym import Pipeline

pipeline = Pipeline(window_length=12).add(StandardScaler()).add(Ridge(alpha=1.0))
report = pipeline.test(seeds=[0])
```

## What TimeGym checks

Every forecast goes through three questions, in this order.

**1. Is the forecast well-formed?** These output checks run on every forecast
(in the code, the *forecast contract*):

- Forecasts the requested dates
- Returns every series, in order
- Right number of steps and series
- No missing or infinite values
- Quantiles in order, when the forecast has quantiles

A forecast that fails a check fails the test with a named reason, such as
`contract_index_mismatch` when the dates are off, and is never scored, so a bug
cannot pass as a slightly worse error. A model that crashes gets `ERROR`, with
its traceback.

**2. How far is it from the answer key?** Because TimeGym generated the series,
it knows the expected value and the spread of every future step (in the code,
the *reference*). Your forecast is scored against both the key and what actually
happened:

| Score | What it measures |
| --- | --- |
| `mean_reference_mse` | Error against the answer key: how far your forecast is from the best possible one |
| `mse`, `mae`, `mase` | Error against what actually happened, which also contains noise nobody could predict |
| `crps`, `energy_score`, `pinball` | For quantiles and sample paths: how close your spread is to the true spread |

**3. Does it do what you declared?** Optional checks for specific properties: a
model that claims to shift its forecast when the data shift can be tested for
it, and sample paths can be tested for the right correlation between series.

Each check returns a verdict and the numbers that led to it (in the code, a
*diagnostic* with its *evidence*):

| Verdict | Meaning |
| --- | --- |
| `PASS` | The check was run and satisfied |
| `FAIL` | Something is wrong: a malformed forecast, or a broken declared property |
| `INCONCLUSIVE` | Scored, but no pass mark is set. TimeGym tells you how far off you are; what counts as good enough is your call |
| `UNSUPPORTED` | Not checked, because the forecast lacks what the check needs, such as quantiles |
| `ERROR` | The model crashed; the traceback is kept |

[How TimeGym decides](https://diogoseca.github.io/timegym/docs/concepts/how-timegym-decides/)
explains each verdict in more depth.

## Scenarios

Each scenario is a kind of series with a known answer key. The useful question
for each is what a forecast *can* get right, and what nobody can predict.

| Scenario | What the series does | What a good forecast gets right | What nobody can predict |
| --- | --- | --- | --- |
| Noise around a fixed level (`white_noise`) | Scatters randomly around a constant | The level | Each individual wiggle |
| Noise around a trend (`trend`) | A straight line plus scatter | The line, extended | The scatter |
| Noise around a season (`seasonal`) | A repeating wave plus scatter | The wave, repeated | The scatter |
| Mean reversion (`ar1`, also called AR(1)) | Wanders, but is pulled back to its mean | The next few steps, sliding back to the mean | Anything further out: there the best forecast is the mean |
| Random walk (`random_walk`) | Each step adds a fresh shock to the last value, and shocks never fade | That it stays near where it is now | Which way it goes; the uncertainty grows with every step |
| Random walk with drift (`random_walk` with `drift`) | A random walk with a steady push | The push | The wandering around it |

Mean reversion and the random walk are the same equation: each step keeps a
fraction of the last step's distance from the mean and adds a fresh shock. Keep
80%, the default, and the series is pulled back; keep 100% and the pull is
gone. The [scenario guide](docs/stress-cases.md) explains each kind in plain
terms, and the composed and financial ones: trends, seasons and noise with
scheduled changes, price jumps, volatility clusters (GARCH) and hidden regime
switches, alone or across several correlated series.

## Tutorials

| Group | Notebook | What it shows |
|---|---|---|
| Start | [Quick start](https://diogoseca.github.io/timegym/notebooks/00_quick_start/) | Two scorecards, last value against Ridge: each is blind where the other sees |
| Find what breaks | [Faults, repairs and replay](https://diogoseca.github.io/timegym/notebooks/23_faults_repairs_and_replay/) | Four ways to break a correct forecast, a bug caught by a broken promise, replay |
| | [Write and test a custom model](https://diogoseca.github.io/timegym/notebooks/28_custom_model/) | The 10-line LastValue model, nearest analogues, real data |
| Shape the forecast | [Target transforms](https://diogoseca.github.io/timegym/notebooks/29_target_transforms/) | Why logs help when the swing grows with the level, and why exp(mean log) is too low |
| | [Reductions](https://diogoseca.github.io/timegym/notebooks/30_reductions/) | Recursive, direct and multioutput: when recursive errors compound |
| | [Multivariate forecasting](https://diogoseca.github.io/timegym/notebooks/21_multivariate_forecasting/) | Right numbers under the wrong series name; when extra history doesn't help |
| | [Real-data forecasts](https://diogoseca.github.io/timegym/notebooks/24_real_data_forecasts/) | No answer key: four origins, three baselines, a leakage check |
| Measure uncertainty | [Univariate uncertainty](https://diogoseca.github.io/timegym/notebooks/20_univariate_uncertainty/) | Same predictions, overconfident spread: which scores notice |
| | [Multivariate uncertainty](https://diogoseca.github.io/timegym/notebooks/22_multivariate_uncertainty/) | Scrambled pairings: every series right, joint risk gone |
| Stress-test | [Capability sweeps](https://diogoseca.github.io/timegym/notebooks/25_capability_sweeps/) | Three simple bets, and where each breaks as history and noise change |
| | [Composed and financial stresses](https://diogoseca.github.io/timegym/notebooks/26_composed_and_financial_stresses/) | A ladder of mechanisms, bad luck vs a bad model, a conversion bug |
| | [Joint financial uncertainty](https://diogoseca.github.io/timegym/notebooks/27_joint_financial_uncertainty/) | Assets that crash together, and the check that notices a lost link |

The notebooks run on CPU with local data. Clone the repository, then
`pip install -e '.[notebooks]'` and `python -m jupyterlab notebooks`; the
[notebook guide](notebooks/README.md) describes the folder.

## Scope and development

TimeGym tests forecasters; it does not fix them. It does not find the root
cause of a bug, repair a model, detect data leakage in general, or work out an
answer key for data you bring yourself: real data is scored against what
happened, and the checks that need a key say `UNSUPPORTED`. Panels, hierarchies,
and missing or irregular observations are out of scope for now.

The package is written in notebooks: the source notebooks in `notebooks/`
generate every module in `timegym/`. [Contributing](docs/contributing.md)
describes that cycle, and [releasing](docs/release.md) the version numbers and
how a merged pull request becomes a PyPI release.

## Cite

Cite the software by version; GitHub's "Cite this repository" reads the same
metadata from [CITATION.cff](CITATION.cff). When a release is archived on
Zenodo, cite that release's DOI.

<!-- BEGIN GENERATED CITATION -->

```bibtex
@software{seca2026timegym,
  author = {Seca, Diogo},
  title = {{TimeGym: verification and diagnosis for forecasting pipelines}},
  year = {2026},
  version = {1.1.0},
  url = {https://github.com/diogoseca/timegym},
}
```

Seca, D. (2026). *TimeGym: verification and diagnosis for forecasting pipelines*
(Version 1.1.0) [Computer software]. https://github.com/diogoseca/timegym

The 2021 preprint (arXiv version 1, the original prototype):

```bibtex
@article{seca2021timegym,
  author = {Seca, Diogo},
  title = {{TimeGym: Debugging for Time Series Modeling in Python}},
  year = {2021},
  doi = {10.48550/arXiv.2105.01404},
  url = {https://arxiv.org/abs/2105.01404v1},
}
```

<!-- END GENERATED CITATION -->

MIT license. Datasets and cited works keep their own attribution; see
[data provenance](experiments/data/README.md).
