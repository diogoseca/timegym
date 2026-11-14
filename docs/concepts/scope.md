# What TimeGym tests, and what it does not

TimeGym runs a forecaster on series it generated itself, so it knows the
answer key for every future step. That makes some questions answerable
exactly and leaves others outside its reach. This page says which is which.

## Verification, not validation

Verification asks whether code does what its authors meant it to do;
validation asks whether that is the right thing to do for the real system.
TimeGym does verification, in the way manufactured solutions verify numerical
code: a problem is built so that its exact solution is known, and the code is
checked against it
([Roache 2002](https://doi.org/10.1115/1.1436090)).

A scorecard therefore says whether a pipeline can learn each mechanism it
shows: a trend, a seasonal cycle, a random walk, mean reversion. It never says
whether those mechanisms are in your data, or which of them matter there. A
pipeline that scores well can still forecast your series badly, and one that
fails a row can be the right choice for data without that pattern.

## What can leak and what cannot

`Pipeline.predict(context, future_index)` fits the model on a copy of the
context, the past only, and then calls `predict` with a copy of the future
index: the labels of the steps to forecast, never their values. A target
transform is fitted on that context alone. `Pipeline.test`, `Pipeline.run` and
`Pipeline.scorecard` call `predict` the same way for every case and score the
forecast afterwards against the held-out values and the answer key.

So a model under test never receives the future values, and nothing TimeGym
passes to it can leak them: look-ahead inside a test is impossible through the
interface. Leakage while you prepare your own data, such as features computed
with a centered window or scaled with statistics of the whole series, happens
before TimeGym sees anything and is outside it.

## Which learners the shift check applies to

`check_shift_equivariance` adds a constant to the history, refits, and checks
that the point forecast moves by the same constant, `f(y + c) = f(y) + c`,
within `atol=1e-8` and `rtol=1e-7`. Its docstring says the property is not
universal, and the caller decides whether it applies. Whether it applies is a
property of the learner, not a sign of quality:

- **A linear model with an unpenalised intercept on lagged values.** Shifting
  every value by `c` shifts every input and every target by `c`. The fitted
  slopes are unchanged, the intercept absorbs the shift, and so each forecast,
  and each recursive step built on it, moves by exactly `c`. Only rounding
  differs, well inside the tolerances. Penalising the intercept breaks this.
- **Trees and tree ensembles on lagged values.** In exact arithmetic every
  split threshold moves by `c` and every leaf mean moves by `c`, so the
  forecast moves by `c`. In floating point the shifted values give slightly
  different sums when the learner scores candidate splits. Where two splits
  separate the training rows equally well, which happens in small nodes,
  rounding decides between them, and the refitted tree can pick the other
  one. Both trees fit the history equally well and still forecast a new input
  differently, so the forecast can move by more than rounding. A tree model
  can fail the check with no bug, so its `FAIL` needs a look before it is
  called a fault.
- **A neural network on raw lagged values.** Its activations respond to the
  level of the inputs, not only to their differences, so a shifted history
  can give a different forecast shape. It need not move by `c`, and that is
  no bug.

The check is for a model that promises the property, such as a last-value,
moving-average or differenced model, or a linear model as above. A model that
clips at zero, uses absolute levels as features or shrinks towards a fixed
level breaks it on purpose.

## How to read a score

A cell's score is `100 * unavoidable / (unavoidable + avoidable)`, pooled over
its series ([The score](../api/scorecard.md#the-score)). Read it in four ways:

- **Against 100.** 100 is the answer key itself: the forecast of someone who
  knows the process and its parameters.
- **Against the triangle.** The `attainable` column is what the row's own model
  family scores when it is fitted to each history. It is an estimate from 200
  series, not a ceiling: a model can land above it on a given card
  ([What a fitted model could score](../api/scorecard.md#what-a-fitted-model-could-score)).
- **Within its interval.** `score_low` and `score_high` are a 95% bootstrap
  interval over the card's series, and `settled` says whether both ends fall in
  the printed band ([How sure a score is](../api/scorecard.md#how-sure-a-score-is)).
- **Against a saved baseline.** On the same series, a paired test says whether
  a new version of the pipeline is further from the answer key than the version
  you trusted ([Compare with a saved baseline](../api/scorecard.md#compare-with-a-saved-baseline)).

## When there is no score

The score divides the unavoidable error by the total error. On a series with
no noise the unavoidable error is zero, so the score is 100 for a forecast
that matches the answer key exactly and 0 for every other forecast, however
close: by the formula, an avoidable error of `1e-30` scores 0. Such a score says nothing about
how good a forecast is and should not be read; compare the avoidable error
itself (`avoidable_mse` in `card.table`, `mean_reference_mse` in a report)
instead. The scorecard's grid accepts only positive noise levels.

Deterministic chaos is the standard case. Its future is fixed by its present,
so the unavoidable error is zero, yet any error in the state grows quickly,
and every practical forecaster would score 0.
