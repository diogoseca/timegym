# What TimeGym tests, and what it does not

TimeGym simulates the series it tests on, so it knows the best possible
forecast for every future step: the answer key. That lets it answer some
questions exactly. Others it cannot answer at all. It also scores forecasts on
data you bring, where nobody has an answer key. This page says what each job
can tell you.

## It checks your code, not your choice of model

TimeGym answers one question: does the pipeline do what it was built to do? It
does not answer a second one: is this the right pipeline for your data?
Engineers call the first verification and the second validation. TimeGym
checks forecasting code the way numerical software is checked, on a problem
built so that the exact answer is known
([Roache 2002](https://doi.org/10.1115/1.1436090)).

So a scorecard tells you whether a pipeline can learn each kind of behaviour
it shows: a trend, a seasonal cycle, a random walk, mean reversion. It does
not tell you whether your data behave that way, or which of those behaviours
matter there. A pipeline that scores well can still forecast your series
badly, and one that fails a row can be the right choice for data without that
pattern.

## On your own data

TimeGym does two jobs. The first, above, checks your code on series it
simulated. The second scores forecasts on data you bring. That data has no
answer key, so TimeGym scores each forecast against what happened. It still
runs every output check, so a forecast for the wrong dates or with two series
swapped still fails with the reason. That tells you how each forecaster did
on these cases. It does not tell you how it will do on cases you have not
seen, unless your cases are a fair sample of them.

To use it, cut your series into cases with
[`cases_from_series`](../api/tasks-and-forecasts.md#your-own-data), or build
`Case` objects yourself, and pass them to `pipeline.test(cases=...)` or
`pipeline.run(cases=...)`. Forecasts made earlier, by a model TimeGym cannot
run, are scored with
[`ForecastRun.from_forecasts`](../api/pipeline.md#signatures). The checks that
need an answer key say `UNSUPPORTED`, and these cases cannot be replayed from
a report, because they cannot be drawn again: keep your data with the report.
Several cases cut from one series are several looks at one history, not
independent tests.

## The model never sees the future

During a test the model gets two things: the history, and the dates it has to
forecast. It never gets the values at those dates. TimeGym scores the forecast
afterwards, against what happened and against the answer key. So inside a
TimeGym test there is no data leakage: a model cannot use future values, even
by accident.

In code: `Pipeline.predict(context, future_index)` fits the model on a copy of
the history and then asks for a forecast with a copy of the future dates. A
target transform is fitted on that history alone. `Pipeline.test`,
`Pipeline.run` and `Pipeline.scorecard` all forecast this way.

What TimeGym cannot see is leakage in how you prepared your own data before
the test. If a feature was computed with a window that reaches forward in
time, values were scaled using the whole series, or irregular readings were
put on a grid with a reading from after each step, the future has already
leaked in, and no test here will notice.
[Data preparation](../api/data-preparation.md#irregular-readings) shows how
to put readings on a regular grid without that last leak.

## The shift check is not for every model

`check_shift_equivariance` adds the same number to every value of the history,
fits the model again, and checks that the forecast moves by that same number.
If every temperature reading were 10 degrees higher, the forecast should be 10
degrees higher. A model that behaves this way is called shift-equivariant. The
two forecasts have to agree within `atol=1e-8` and `rtol=1e-7`.

Some models promise this and some do not, and a model that does not is not a
worse model. Run the check when your model should behave this way:

- **Should pass.** A last-value, moving-average or differenced model, and a
  linear model on past values whose intercept is not regularised, such as
  ordinary least squares. Adding a number to every value adds it to every
  input and every target, so the fitted coefficients stay the same and the
  intercept takes up the shift.
- **Can fail without a bug: tree models on past values.** On paper a tree
  passes. In practice, when two ways of splitting the data are equally good,
  tiny rounding differences decide between them, and after the shift the tree
  can pick the other one. Both trees fit the history equally well and still
  forecast differently. Look into a `FAIL` from a tree model before you call
  it a bug.
- **Need not pass: a neural network on raw past values.** It responds to how
  high the values are, not only to how they change, so a shifted history can
  give a forecast of a different shape.
- **Breaks it on purpose.** A model that clips at zero, uses the level itself
  as a feature, or pulls its forecasts towards a fixed level.

## How to read a score

A score compares two errors: the error nobody could avoid, because the future
has noise in it, and the extra error the model added. 100 means no extra
error, and 50 means the extra error is as large as the noise. The formula is
`100 * unavoidable / (unavoidable + avoidable)`, pooled over the cell's series
([The score](../api/scorecard.md#the-score)). Read a score in four ways:

- **Against 100.** 100 is the answer key itself: the forecast of someone who
  knows exactly how the series was made.
- **Against the attainable score.** The `attainable` column, the small
  triangle on the picture, is what the right kind of model scores when it has
  to estimate its parameters from the same history. It is an estimate from 200
  series, not a ceiling: a model can land above it on a given card
  ([What a fitted model could score](../api/scorecard.md#what-a-fitted-model-could-score)).
- **Within its interval.** With other series of the same kind the score would
  come out a little different. `score_low` and `score_high` show by how much:
  a 95% confidence interval, or another level if you pass `interval_level`. `settled`
  says whether the whole interval stays in the printed band
  ([How sure a score is](../api/scorecard.md#how-sure-a-score-is)). The
  interval only knows the series the card drew. It cannot see a model that
  fails on a kind of series the card did not happen to draw, and it says
  nothing about data that the patterns do not resemble.
- **Against a saved baseline.** On the same series, is a new version of the
  pipeline further from the answer key than the version you trusted?
  ([Compare with a saved baseline](../api/scorecard.md#compare-with-a-saved-baseline)).

## When there is no score

The score needs some noise. On a series with no noise at all, nobody is forced
to make any error, so the score is 100 for a forecast that matches the answer
key exactly and 0 for every other forecast, however close: an extra error of
`1e-30` scores 0. That number says nothing about how good a forecast is, so do
not read it. Compare the extra error itself instead (`avoidable_mse` in
`card.table`, `mean_reference_mse` in a report). The scorecard's own grid
always has noise, so this only comes up with series you define yourself.

Chaotic series are the usual example. They are deterministic: their future is
fully fixed by their present, so there is no noise. Yet the smallest error
grows fast, and every practical forecaster would score 0.
