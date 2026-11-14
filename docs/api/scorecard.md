# Scorecards and the scenario grid

One call that tells you what a forecaster can and cannot do:
`Pipeline.scorecard()` and `Scorecard` from `timegym`, the grid itself
(`PATTERNS`, `GridCell`, `scenario_grid`) from `timegym`, and the lower-level
`verdict`, `VERDICTS`, `Pattern`, `noise_levels`, `NOISE_LOW` and
`NOISE_HIGH` from `timegym.scorecard` and `timegym.grid`.

```python
from timegym import LastValueForecaster, Pipeline

card = Pipeline(LastValueForecaster()).scorecard()
card                       # in a notebook: the image
print(card)                # Markdown summary and table
card.save("scorecard.png") # or .svg / .pdf
```

## The grid

`PATTERNS` is ten kinds of series, in plain words. Each `Pattern` carries a
`title`, an `aka` (the textbook name), and two sentences: `predictable`, what a
forecaster that knew the process could get right, and `unpredictable`, what no
forecaster can get right by design.

| Key | Title | What can be predicted | What cannot, by design |
| --- | --- | --- | --- |
| `noise` | Pure noise | Only the average, zero: the best forecast is a flat line at zero. | Every individual value. Any pattern a model finds here is invented. |
| `level` | Noise around a fixed level | The level, estimated from the whole history. | Each wiggle around the level. |
| `trend` | Straight-line trend | The line, extended any number of steps ahead. | The noise around the line. |
| `seasonal` | Repeating seasonal cycle | The repeating shape, any number of steps ahead. | The noise around the cycle. |
| `trend_seasonal` | Trend + seasonal cycle | Both the line and the repeating shape. | The noise around them. |
| `level_shift` | Sudden level shift | The new level, from the history since the jump. | The noise. Older history sits at the old level and misleads. |
| `mean_reverting` | Mean reversion (AR(1)): shocks fade | The pull back to the average: a gap halves in about three steps. | New shocks. After about ten steps only the average is left; measurement noise hides where the series is now. |
| `cycles` | Irregular cycles (AR(2)) | The rest of the current swing, fading over the next cycles. | When the next swings come and how big they are; noise hides the current phase. |
| `random_walk` | Random walk: shocks never fade | Where it is now: the best forecast is flat at the current level, smoothed when noisy. | The direction of the next move. Uncertainty grows without limit with the horizon. |
| `random_walk_drift` | Random walk with drift | The current level plus the steady drift. | The wandering around the drift line, which grows with the horizon. |

The last four rows are autoregressive; [Synthetic scenarios](../stress-cases.md#the-core-scenarios) explains how mean reversion, irregular cycles and the random walk differ.

`noise_levels(levels=2)` returns `(name, sd)` pairs from low to high: two
levels by default, `("Low noise", 0.5)` and `("High noise", 2.0)`.
An integer `n >= 2` spaces `n` levels geometrically between `NOISE_LOW = 0.5`
and `NOISE_HIGH = 2.0`, so every grid keeps both extremes; three levels are
named Low, Medium and High. A list of standard deviations is used as given.
The noise is the standard deviation of an error added to each observation. For
the AR and random-walk rows it is measurement noise on top of the hidden
process, so the answer key comes from an exact Kalman filter.

`scenario_grid(patterns=None, noise_levels=2, *, context_length=120,
horizon=24)` returns one `GridCell(pattern, title, noise, noise_sd, scenario)`
per pattern and noise level, rows first. Each `scenario` is an ordinary
`Scenario` that can also be passed to `Pipeline.run` or `Pipeline.test`.

## The score

Every forecast error splits into two parts. The **unavoidable** part is the
error a forecaster that knows the process exactly still makes, because the
future noise has not happened yet. The **avoidable** part is the distance from
the forecast to that forecaster's answer (the answer key). A cell's score is

    score = 100 * unavoidable / (unavoidable + avoidable)

pooled over its instances. 100 matches the answer key; 50 means the avoidable
error equals the unavoidable noise. Instances that crash or fail an output
check score zero, so a broken forecast cannot score well. The overall score is
the mean of the cell scores. `VERDICTS` names the bands on the rounded score:
90 or more "Nails it", 70 "Good", 40 "Struggles", below 40 "Fails";
`verdict(score)` returns the band.

## Mistakes and checks

Cells that struggle or fail (a shown score below 70) name up to two **mistakes**. Each instance's error
against the answer key splits exactly into a constant offset (*Forecasts too
high*, *Forecasts too low*, or *Misreads the current level* when it is too
high in some instances and too low in others), a straight-line drift (*Trend too steep*, *Trend
too flat*, *Misjudges the trend*, *Invents a trend*) and the remaining shape
(*Misses the cycle*, *Cycle out of step*, *Exaggerates the cycle*, *Too
wiggly*). A forecast that matches the answer key one step later or
earlier is called *Values one step late* or *Values one step early*. Failed output checks are
mistakes too, for example *Wrong dates: shifted 1 step early*.

`card.checks` (and `report.checks` on any `TestReport`) lists the output checks
in plain words with how many cases passed and failed:

| Check | Title |
| --- | --- |
| `runs` | Runs without errors |
| `dates` | Forecasts the requested dates |
| `series` | Returns every series, in order |
| `shape` | Right number of steps and series |
| `finite` | No missing or infinite values |
| `quantiles` | Quantiles in order |
| `dependence` | Joint spread matches the answer key (only with `check_dependence=True`) |

`report.assert_passed()` raises with one line per failed check, for example
`Forecasts the requested dates: failed in 4 of 4 cases`, followed by the
underlying diagnostics.

## Seeds and instances

`instances=10` independent series per cell use seeds `seed, seed + 1, ...`.
The same seeds are used in every cell and the hidden path is drawn before the
measurement noise, so the noise columns are a paired comparison. `seed=None`
draws a fresh seed and keeps it on `card.seed` and in the report metadata, so
an interesting run can be repeated. Raise `instances` (20 or 30) when comparing
close models.

## The Scorecard

`Scorecard` keeps `title`, `table` (one row per pattern and noise level with
`score`, `verdict`, `mistakes`, `unavoidable_mse`, `avoidable_mse`, `mse`,
`instances` and `failed_checks`), `run` (the `ForecastRun` with every series and
forecast), `cells`, `seed` and `instances`. `good_at` lists patterns scoring at
least 70 at every noise level, `fails_at` those below 40 at some level and
`noise_sensitive` those that lose 25 points or more from low to high noise.
`report` is the JSON-ready `TestReport`. `plot(compact=False)` returns the
Matplotlib figure; `compact=True` draws only the colored scores.

## Signatures

::: timegym.Scorecard

::: timegym.scenario_grid

::: timegym.GridCell

::: timegym.scorecard.verdict

::: timegym.grid.Pattern

::: timegym.grid.noise_levels
