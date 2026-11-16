# Scorecards and the scenario grid

One call that tells you what a forecaster can and cannot do:
`Pipeline.scorecard()` and `Scorecard` from `timegym`, the grid itself
(`PATTERNS`, `GridCell`, `scenario_grid`) from `timegym`, and the lower-level
`verdict`, `VERDICTS`, `ScorecardComparison`, `Pattern`, `noise_levels`,
`NOISE_LOW` and `NOISE_HIGH` from `timegym.scorecard` and `timegym.grid`.

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
forecaster can get right by design. `build(noise, context_length, horizon)`
returns the row's `Scenario`, and `fitted()` an estimator of the generator's own
model family (see [What a fitted model could score](#what-a-fitted-model-could-score)).

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

## How sure a score is

Ten series per cell are a small sample, so every score comes with an interval.
`card.table` has three columns for it:

- `score_low` and `score_high`: a 95% percentile bootstrap interval over the
  cell's series. The cell is scored again on 2,000 resamples of its series,
  drawn with replacement; an instance that crashed or failed an output check
  scores zero in every resample that draws it. The interval always contains
  the score. The same seed gives the same interval, whichever other patterns
  were scored. With fewer than two series there is no interval (NaN).
- `settled`: True when both ends of the interval fall in the band printed for
  the score.

```python
card.table[["title", "noise", "score", "score_low", "score_high", "verdict", "settled"]]
card.unsettled        # the rows whose interval crosses 90, 70 or 40
```

The printed card adds a question mark to a verdict that is not settled
(`56 Struggles?`) and ends with a line such as `12 of 20 verdicts are not
settled at 10 instances. Run scorecard(instances=30) to narrow them.` On the
image, a short line beside each score draws its interval on a 0 to 100 scale,
ticked at 40, 70 and 90; a verdict that is not settled gets a question mark
and a paler badge.

The interval only knows the series the card drew. With ten series it is rough
and tends to be too narrow; raise `instances` to tighten it.

## What a fitted model could score

The answer key knows the true parameters; a model fitted to the history has to
estimate them, so even one of exactly the right kind falls short of 100 on
average. `card.table` has a column
`attainable` for that: the score that the pattern's own model family, fitted to
each history, reaches on 200 fresh series of the row, with seeds 1,000,000 to
1,000,199, pooled as a cell is pooled:

    attainable = 100 * mean unavoidable / (mean unavoidable + mean avoidable)

over those 200 series. Every numeric parameter is estimated; everything
structural is given: the period, that there is exactly one break, the number
of lags, that there is measurement noise. The value belongs to the row
(pattern, noise level, history length and horizon), not to the series a card
happens to draw, so on ten series a model of the right kind can land a little
above it. The value is itself an estimate from 200 series, good to about a
point: across blocks of 200 series it moves with a standard deviation of 0.1
to 0.7 points on most rows and 1.1 on the random walk with drift at low noise.

| Pattern | Model family fitted to each history | Low noise | High noise |
| --- | --- | --- | --- |
| `noise` | sample mean | 99.1 | 99.1 |
| `level` | sample mean | 99.1 | 99.1 |
| `trend` | least-squares line | 95.3 | 95.3 |
| `seasonal` | constant + sine wave of period 12, least squares | 97.5 | 97.5 |
| `trend_seasonal` | line + sine wave of period 12, least squares | 93.9 | 93.9 |
| `level_shift` | mean after one break, dated by least squares | 96.7 | 96.7 |
| `mean_reverting` | AR(1) + measurement noise, maximum likelihood | 93.0 | 95.8 |
| `cycles` | AR(2) + measurement noise, maximum likelihood | 92.6 | 92.9 |
| `random_walk` | random walk + measurement noise, maximum likelihood | 99.8 | 98.9 |
| `random_walk_drift` | random walk with drift + measurement noise, maximum likelihood | 87.3 | 90.2 |

Read it beside the score. 100 minus the attainable score is what the row's own
model family loses to estimating its parameters from the history. The
attainable score minus the model's score is what this model loses beyond that,
and that is the part worth investigating.

```python
table = card.table
table.assign(lost_on_top=table["attainable"] - table["score"])[
    ["title", "noise", "score", "attainable", "lost_on_top"]]
```

Each `Pattern` carries the estimator as `fitted`, a factory with no arguments.
The estimator is an ordinary model with `fit(context)` and `predict(future_index)`,
so it can be scored like any other:

```python
from timegym import PATTERNS, Pipeline

Pipeline(PATTERNS["trend"].fitted()).scorecard(["trend"])
```

The four maximum-likelihood fits run a Kalman filter with the scale and the
mean (or the drift) solved for in closed form. A `Pattern` without `fitted`
(`None`) has no attainable score (NaN). On the image the attainable score is a
small triangle above each score's line.

The 20 values of the default grid are stored; `tools/attainable_table.py`
recomputes them. Any other noise level, history length or horizon gets NaN and
no triangle, unless you ask for it:

```python
card = pipeline.scorecard(noise_levels=3, attainable=True)
```

`attainable=True` simulates each missing value once and keeps it for the
session. Each autoregressive or random-walk cell takes about 3 to 5 seconds,
the others well under a second; nothing is simulated without it.

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

## Compare with a saved baseline

This is a regression test for forecast quality. The reference is the version
of your model that you last trusted. Checking that a new version gives exactly
the same forecasts is not a useful test, because every change you make on
purpose changes the forecasts too. The answer key makes the comparison mean
something: on the same series, is the new version further from the best
possible forecast?

```python
# once, on the version you trust
Pipeline(model).scorecard(instances=30).save_baseline("scorecard-baseline.json")

# in the test suite, on every change
card = Pipeline(model).scorecard(instances=30)
diff = card.compare("scorecard-baseline.json")   # same seed, so the same series
print(diff)
diff.assert_no_regression()
```

`save_baseline(path)` writes a small JSON file and returns `path`. It holds a
format name and version, the TimeGym version, the card's title, the grid
(pattern keys, noise standard deviations, history length and horizon), the
seed, the number of instances and, for each cell, its score and each series'
avoidable error in instance order, with `null` for a series that crashed or
failed an output check. Floats are written so that they read back exactly,
and keys are sorted, so the same card always writes the same file. For the
default grid at 30 instances that is 600 avoidable errors, about 17 kB, small
enough to commit beside the tests.

`compare(baseline, *, alpha=0.05)` takes such a file or another `Scorecard`
and returns a `ScorecardComparison`; both give the same result. The two cards
must have the same grid, seed and number of instances, so that they drew the
same series: otherwise `compare` raises `ValueError` and says what differs. A
file that is not a baseline, or one written in a newer format than this
TimeGym reads, raises `ValueError` too. Series come from the seed and from
TimeGym's generators, so if a TimeGym release draws other series for the same
seed, save a new baseline from the version you trust.

Each cell gets one of three outcomes in the `change` column:

- `"unchanged"`: every series came out the same as in the baseline. Each has
  the same status (scored, or crashed or failed an output check), and its
  avoidable error agrees with the baseline's up to roundoff: they differ by at
  most `1e-9`, or by `1e-8` times the largest avoidable error in the cell (in
  either version) if that is larger.
- `"changed and worse"`: on the same series, the new version is further from
  the answer key by more than chance would explain, or a series that was
  scored in the baseline now crashes or fails an output check. Only this
  outcome fails `assert_no_regression()`.
- `"changed, no worse"`: the forecasts differ, and the test did not find the
  new version worse.

Four things to keep in mind when you read the result:

- "Changed, no worse" does not mean better. It only means the test did not
  find the new version worse.
- A small worsening can pass. With few series only large ones are found, so
  `compare` warns below 20 instances. Use 30.
- A very small worsening can fail when it shows up on almost every series.
  Adding a ridge penalty of 1 to a least-squares regression on six lags raised
  one cell's mean avoidable error by 0.04% on the default grid at 30
  instances, and the comparison called that cell changed and worse. The table
  and the printed comparison show how far each cell moved, so you can judge
  whether a worse cell matters. When a change is intended, save a new
  baseline.
- The test counts each series as separate evidence. That is right when the
  model handles each series on its own. A model that draws its random numbers
  from one fixed seed adds the same noise to every series, so with another
  seed it is a different model, and the comparison can find it worse.

### How the test works

This part is for readers who want the statistics.

The test is paired: it takes the series scored in both cards and their
differences `d = new avoidable error - baseline avoidable error`. Its
statistic is the mean of `d`. It flips the sign of each difference at random
9,999 times and ranks the observed mean among the flipped means with
[`rank_alarm`](diagnostics.md#statistical-and-metamorphic-diagnostics), a
one-sided p-value for a mean above zero:

```
p = (1 + number of flipped means >= observed mean) / 10,000
```

The flips are seeded from the card's seed and the number of pairs, so a
cell's p-value repeats exactly and does not depend on which other patterns
were scored. Because several cells are tested at once, Holm's step-down
correction at `alpha` runs across the cells that changed, the cells with at
least one difference beyond roundoff. The sign-flip test conditions on the
sizes of the differences, and whether a cell changed depends only on those
sizes and on how large the errors are, which swapping the two values of a
pair does not change, so restricting the correction to the changed cells
keeps the test valid. A series that was broken in the baseline and is scored
now is left out of the test and counted as newly passing.

The test assumes that, when the new version is no worse, the paired
differences are not shifted upward: symmetric about zero, or below it. It
also assumes that the series' differences are independent.

`ScorecardComparison.table` has one row per cell: `pattern`, `title`,
`noise`, `noise_sd`, `change`, `baseline_score`, `score`,
`baseline_avoidable` and `avoidable` (each card's mean avoidable error over
the series it scored), `p_value` (before Holm's correction; NaN where no
series moved beyond roundoff), and `newly_failing` and `newly_passing`, the
numbers of series that crash or fail an output check now but did not in the
baseline, and the reverse. `worse` holds the rows that are changed and worse.
`assert_no_regression()` raises `AssertionError` naming each of them with its
numbers. `baseline_version` is the TimeGym version that wrote the baseline
and `card_version` this card's. `print(diff)` gives a plain summary, and names
the baseline's version when the two differ. This one, from
[source notebook 13](../../source/13_scorecard.ipynb), compares a lag
regression with an off-by-one bug against the baseline of the correct model,
with 15 of its "changed, no worse" rows left out:

```text
Compared with scorecard-baseline.json (30 series per cell)
  17 cells  changed, no worse
             Pure noise, low noise                             avoidable error 0.0175 -> 0.0177 (+1.1%)
             ...
             Random walk with drift, high noise                avoidable error 5.10 -> 5.02 (-1.6%)
   3 cells  changed and worse
             Repeating seasonal cycle, low noise               avoidable error 0.164 -> 1.21 (+635%), p = 0.0001
             Mean reversion (AR(1)): shocks fade, low noise    avoidable error 0.299 -> 0.358 (+20%), p = 0.0004
             Mean reversion (AR(1)): shocks fade, high noise   avoidable error 0.497 -> 0.546 (+9.9%), p = 0.0002
```

## The Scorecard

`Scorecard` keeps `title`, `table` (one row per pattern and noise level with
`score`, `score_low`, `score_high`, `verdict`, `settled`, `attainable`, `mistakes`,
`unavoidable_mse`, `avoidable_mse`, `mse`, `instances` and `failed_checks`),
`run` (the `ForecastRun` with every series and forecast), `cells`, `seed` and
`instances`. `good_at` lists patterns scoring at least 70 at every noise level,
`fails_at` those below 40 at some level and `noise_sensitive` those that lose
25 points or more from low to high noise. `unsettled` holds the table rows
whose verdict is not settled.
`report` is the JSON-ready `TestReport`. `plot(compact=False)` returns the
Matplotlib figure; `compact=True` draws only the colored scores.
`save_baseline(path)` writes the card's baseline file and `compare(baseline,
*, alpha=0.05)` compares the card with one
([Compare with a saved baseline](#compare-with-a-saved-baseline)).

## Signatures

::: timegym.Scorecard

::: timegym.scenario_grid

::: timegym.GridCell

::: timegym.scorecard.verdict

::: timegym.scorecard.ScorecardComparison

::: timegym.grid.Pattern

::: timegym.grid.noise_levels
