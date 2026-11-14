# Data preparation

Differences of a target series, and the reversible target transforms that a
`Pipeline` can fit and invert.

`differencing` is public through `timegym.target`; import it from that module. It
is a standalone pandas transformation, not an automatic Pipeline stage, and it
never mutates the supplied object. Row offsets are positional, so a lag of one
means the previous row even when timestamps are irregular. The reversible target
transforms below are the exception: a `Pipeline` fits and inverts them on each
case. The smoothing, feature, label and fractional-difference helpers of TimeGym
1.4.0 are deprecated, leave the package in TimeGym 2.0 and are no longer
documented here; the changelog names what to use instead.

## Differences

`differencing` takes a whole-number `order` (default 1) and a positive `lag`
(default 1). `order` counts repeated differences and is not the lag distance: order two is
`y[t]-2*y[t-lag]+y[t-2*lag]`. It trims `order*lag` rows. Its
`fractional_coefficients` argument serves only the deprecated fractional orders.

Differencing accepts a Series or DataFrame and preserves its type and labels on
the retained rows. Orders must be finite and nonnegative, and lag must be a
positive integer. Missing observations inside the series propagate; they are
not filled with zero. It applies no inverse; use
`DifferenceTransform` below for differences that a pipeline undoes.

## Reversible target transforms

`LogTransform()`, `BoxCoxTransform()` and `DifferenceTransform(order=1, lag=1)`
implement `fit_transform(context)`, `inverse_transform(forecast)` and
`clone_unfitted()`. `ComposedTransform(transforms)` applies bundled members in
order and inverts them in reverse order. Each transform is fitted on the context
only. Log and Box-Cox require strictly positive finite targets; Box-Cox fits one
lambda per target by maximum likelihood, bounded to `[-1, 2]`, and rejects a
constant target and transformed forecasts without a preimage. Inversion
requires a valid forecast with the fitted target names in the fitted order.

- Samples and quantiles invert elementwise. Monotone transforms preserve
  quantile order.
- After a nonlinear inverse, the point is the mean of the inverted samples,
  else the inverted 0.5 quantile, else the inverted point read as a median.
- `DifferenceTransform` integrates points and samples exactly. It recomputes
  quantiles from the integrated samples, or omits them when there are none,
  and assumes the forecast rows immediately follow the context.

A pipeline's saved configuration describes exact bundled classes as `builtin`
and any other object, including a subclass, as `external`; replay rebuilds
bundled transforms only.

## Signatures

::: timegym.target.differencing

::: timegym.target.LogTransform

::: timegym.target.BoxCoxTransform

::: timegym.target.DifferenceTransform

::: timegym.target.ComposedTransform
