# TimeGym

**A gym for time series forecasting algorithms.**

[![Documentation](https://img.shields.io/badge/docs-diogoseca.github.io%2Ftimegym-black)](https://diogoseca.github.io/timegym/)
[![Checks](https://github.com/diogoseca/timegym/actions/workflows/checks.yml/badge.svg)](https://github.com/diogoseca/timegym/actions/workflows/checks.yml)
[![PyPI](https://img.shields.io/pypi/v/timegym)](https://pypi.org/project/timegym/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

[OpenAI Gym](https://arxiv.org/abs/1606.01540) gave reinforcement-learning
agents a standard set of environments to train and be tested in. TimeGym does
the same for forecasters. It generates time series whose future it already
knows the odds of: noise, trends, seasons, sudden shifts, cycles, levels pulled
back to their mean, random walks, each at low and at high noise. It hands your
forecaster the past, asks for the future, and scores the answer against the
truth. One picture shows what your algorithm nails, where it struggles and
where it fails.

**[Documentation](https://diogoseca.github.io/timegym/)** ·
[Quick start](https://diogoseca.github.io/timegym/notebooks/00_quick_start/) ·
[Tutorials](https://diogoseca.github.io/timegym/docs/notebooks/) ·
[API](https://diogoseca.github.io/timegym/docs/reference/) ·
[Changelog](CHANGELOG.md)

## Sixty seconds

```bash
pip install timegym
```

<!-- BEGIN PYTHON SUPPORT -->
Python 3.11–3.15 (`>=3.11,<3.16`), following pandas support.
<!-- END PYTHON SUPPORT -->

Wrap your forecaster in a `Pipeline` and ask for its scorecard. The built-in
last-value forecaster stands in for yours here:

```python
from timegym import NaiveForecaster, Pipeline

pipeline = Pipeline(NaiveForecaster())   # put your own forecaster here
card = pipeline.scorecard()
card.save("scorecard.png")
```

[![The scorecard of the last-value forecaster: a score per kind of series and noise level, with example forecasts](notebooks/assets/scorecard.png)](notebooks/assets/scorecard.png)

Each row is a kind of series and each column a noise level. The left column
says what can and cannot be predicted for that kind of series, and the right
column names the mistakes the model made there. A score of 100 means the
forecasts were as close as those of a forecaster that knows exactly how the
series was made; 50 means the model's avoidable error is as large as the noise
nobody could avoid. The bands read "Nails it" from 90, "Good" from 70,
"Struggles" from 40 and "Fails" below that. The last-value forecaster scores
51: it nails the random walk and fails trends and seasons. In a notebook, `card`
on its own shows the picture.

### Scorecard, test and report

A scorecard is a test on TimeGym's ten kinds of series, drawn as one picture.
"Test" is what TimeGym does; the report is the record it hands back. Three
calls give you three views of the same work:

| You call | You get | Use it to |
| --- | --- | --- |
| `pipeline.scorecard()` | A `Scorecard`: the picture and its table | See what the model is good and bad at |
| `pipeline.test(scenarios)` | A `TestReport`: every case's scores and output checks | Test series you choose, save them as JSON, fail a CI build |
| `pipeline.run(scenarios)` | A `ForecastRun`: the report plus every series and forecast | Plot the forecasts behind the scores |

A scorecard carries the other two: `card.report` and `card.run`.

### More things to do with a scorecard

Print it as Markdown, for a pull request or a log:

```python
print(card)
```

```text
## NaiveForecaster(strategy='last', period=1): 51/100

| Pattern | Low noise | High noise | Mistakes |
|---|---|---|---|
| Pure noise | 56 Struggles | 56 Struggles | Misreads the current level |
| Straight-line trend | 10 Fails | 43 Struggles | Low: Forecasts too low; Trend too flat · High: Misreads the current level |
| Random walk: shocks never fade | 99 Nails it | 87 Good | none found |
…
```

List what it is good at, where it fails, and where noise throws it off:

```python
card.good_at, card.fails_at, card.noise_sensitive
```

```text
(['Mean reversion (AR(1)): shocks fade', 'Random walk: shocks never fade'],
 ['Straight-line trend', 'Repeating seasonal cycle', 'Trend + seasonal cycle', 'Irregular cycles (AR(2))'],
 [])
```

Read the scores as a DataFrame, one row per kind of series and noise level:

```python
card.table[["title", "noise", "score", "verdict", "mse"]].round(2)
```

```text
                        title       noise  score    verdict   mse
0                  Pure noise   Low noise  56.37  Struggles  0.49
1                  Pure noise  High noise  56.37  Struggles  7.87
2  Noise around a fixed level   Low noise  56.37  Struggles  0.49
…
```

`card.table` also names the mistakes and splits each error into the part
nobody could avoid and the part the model could have.

Zoom in on one row:

```python
trend = pipeline.scorecard(["trend"], instances=3)   # one row, both noise levels
trend.save("trend.png")
```

[![The scorecard's straight-line trend row on its own](notebooks/assets/readme-trend-scorecard.png)](notebooks/assets/readme-trend-scorecard.png)

Plot every draw behind that row, with the answer key and the forecast:

```python
trend.run.plot_forecasts()
```

[![The three draws of the trend row at each noise level, with the answer key and the last-value forecast](notebooks/assets/readme-trend-forecasts.png)](notebooks/assets/readme-trend-forecasts.png)

See how the error spreads over 20 draws per noise level:

```python
pipeline.scorecard(["trend"], instances=20).run.plot_errors(metric="mse")
```

[![Histograms of the error against what happened, 20 draws at low and at high noise](notebooks/assets/readme-trend-errors.png)](notebooks/assets/readme-trend-errors.png)

`mse` is the error against what actually happened. The default,
`mean_reference_mse`, is the error against the answer key: the part your model
could have avoided.

Stop a CI build when a forecast is malformed or the model crashes. Each output
check counts the cases that passed it:

```python
card.checks
```

```text
[{'check': 'runs', 'title': 'Runs without errors', 'passed': 200, 'failed': 0},
 {'check': 'dates', 'title': 'Forecasts the requested dates', 'passed': 200, 'failed': 0},
 …]
```

```python
card.report.assert_passed()   # raises on any failure and says what went wrong
```

Save every case to analyse later, or to replay one on another machine:

```python
card.report.to_json("report.json")
```

```text
{"cases": [{"diagnostics": [{"name": "forecast_contract", "status": "PASS", "message": …}, …],
            "manifest": {"kind": "white_noise", "horizon": 24, "parameters": …, …},
            "metrics": {"mae": 0.571, "mase": 1.081, "mean_reference_mse": 0.339, "mse": 0.538},
            "scenario": "white_noise",
            "seed": 0},
           …],
 "metadata": {"instances": 10, "model": "NaiveForecaster", "seed": 0, "timegym_version": …, …},
 "summary": {…}}
```

The file holds each case's scenario, seed and scores, every check's verdict
and the numbers behind it, the pipeline's settings, and the Python and library
versions, so a failure with a built-in model can be replayed from the file
alone.

Change the size of the test. Each kind of series is drawn `instances` times at
each noise level; draw `i` uses seed `seed + i`, so the same seed always gives
the same series:

```python
card = pipeline.scorecard(instances=30)     # more draws: steadier scores
card = pipeline.scorecard(noise_levels=3)   # low, medium and high noise
card = pipeline.scorecard(seed=None)        # fresh draws every time
card.seed                                   # the seed that was drawn, to repeat the run
```

When tuning a model on the scorecard, 20 or 30 instances keep the score from
rewarding luck on a few draws; with `seed=None` every evaluation sees new
series, so the tuning cannot overfit to fixed ones.

Pick your own series. `test` returns the report; `run` keeps the forecasts too:

```python
from timegym import Scenario

suite = [
    Scenario("seasonal", context_length=240, horizon=24),
    Scenario("random_walk", parameters={"drift": 0.1}),
]
report = pipeline.test(suite, seed=0, instances=30)   # scores and checks
run = pipeline.run(suite, seed=0, instances=3)         # the same, plus every forecast
run.plot_forecasts()
```

[![Three draws each of a seasonal series and a random walk with drift, with the last-value forecast](notebooks/assets/readme-custom-scenarios.png)](notebooks/assets/readme-custom-scenarios.png)

## Why not just backtest?

A backtest compares your forecast with the one future that happened. That
future is noisy, so a lucky model looks good and a sound one can look bad.
TimeGym made the data, so it holds the answer key: the expected value and the
spread of every future step. It scores your forecast against that key, which
measures skill rather than luck.

**A score can also hide a bug.** Put the right numbers on the wrong dates and
any scorer that compares arrays gives them exactly the same error. TimeGym
checks the dates, the series and the shape first, and fails the forecast with
the reason before it scores anything.

[![A forecast with the right values on the wrong dates: the same error, a different diagnosis](notebooks/assets/alignment-demo.png)](https://diogoseca.github.io/timegym/notebooks/23_faults_repairs_and_replay/)

## Bring your own model

A forecaster is any object with two methods. `fit(context)` gets the series so
far. `predict(future_index)` gets the dates to forecast, the steps right after
the context, and returns one value per date: a `Forecast`, or simply a list
or NumPy array. This one repeats the last value it saw:

```python
from timegym import Forecast, Pipeline

class LastValueForecaster:
    def fit(self, context):            # the series so far, one column per series
        self.last = context.iloc[-1]   # the last row seen
        return self

    def predict(self, future_index):   # the dates to forecast
        return Forecast(future_index, [self.last] * len(future_index))

card = Pipeline(LastValueForecaster()).scorecard()
```

It scores 51, like the built-in one.
[Write and test a custom model](https://diogoseca.github.io/timegym/notebooks/28_custom_model/)
adds several series, quantiles and a nearest-analogues forecaster.

scikit-learn regressors plug in directly, fed the last `window_length` values:

```python
from sklearn.linear_model import Ridge

card = Pipeline(Ridge(), window_length=12).scorecard()
```

## What TimeGym checks

Every forecast goes through three questions, in this order.

**1. Is the forecast well-formed?** These output checks run on every forecast
(in the code, the *forecast contract*):

- Forecasts the requested dates; a failure says how many steps they are off
- Returns every series, in order
- Right number of steps and series
- No missing or infinite values
- Quantiles in order, when the forecast has quantiles
- Runs without errors

A forecast that fails a check fails the test with the reason and is never
scored, so a bug cannot pass as a slightly worse error.

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

The scorecard's rows are ten kinds of series, each drawn at low noise (standard
deviation 0.5) and high noise (2.0). For each one TimeGym knows what a forecast
*can* get right, and what nobody can predict:

| Kind of series | What a good forecast gets right | What nobody can predict |
| --- | --- | --- |
| Pure noise | Only the average, zero: the best forecast is a flat line at zero. | Every individual value. Any pattern a model finds here is invented. |
| Noise around a fixed level | The level, estimated from the whole history. | Each wiggle around the level. |
| Straight-line trend | The line, extended any number of steps ahead. | The noise around the line. |
| Repeating seasonal cycle | The repeating shape, any number of steps ahead. | The noise around the cycle. |
| Trend + seasonal cycle | Both the line and the repeating shape. | The noise around them. |
| Sudden level shift | The new level, from the history since the jump. | The noise. Older history sits at the old level and misleads. |
| Mean reversion (AR(1)): shocks fade | The pull back to the average: a gap halves in about three steps. | New shocks. After about ten steps only the average is left; measurement noise hides where the series is now. |
| Irregular cycles (AR(2)) | The rest of the current swing, fading over the next cycles. | When the next swings come and how big they are; noise hides the current phase. |
| Random walk: shocks never fade | Where it is now: the best forecast is flat at the current level, smoothed when noisy. | The direction of the next move. Uncertainty grows without limit with the horizon. |
| Random walk with drift | The current level plus the steady drift. | The wandering around the drift line, which grows with the horizon. |

Mean reversion and the random walk are the same equation: each step keeps a
fraction of the last step's distance from the mean and adds a fresh shock. Keep
80% and the series is pulled back; keep 100% and the pull is gone. The
[scenario guide](docs/stress-cases.md) explains each kind in plain terms, and
the composed and financial ones beyond the scorecard: price jumps, volatility
clusters (GARCH) and hidden regime switches, alone or across several correlated
series.

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
  version = {1.2.0},
  url = {https://github.com/diogoseca/timegym},
}
```

Seca, D. (2026). *TimeGym: verification and diagnosis for forecasting pipelines*
(Version 1.2.0) [Computer software]. https://github.com/diogoseca/timegym

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
