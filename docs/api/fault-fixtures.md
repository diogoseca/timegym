# Fault fixtures

Output fault fixtures inject a known fault into an otherwise valid forecast,
perturbing outputs rather than code: `MUTATIONS` and `mutate_forecast` from
`timegym.mutations`.

`timegym.mutations.mutate_forecast(forecast, name, *, seed=0, enabled=True,
strength=1.0)` deep-copies outputs and coerces their arrays before checking a
fixture's preconditions. `timegym.mutations.MUTATIONS` lists these eight names.
The validator codes below assume an otherwise valid forecast for its task;
truncation can also produce output-shape violations.

| Name | Transformation | Precondition | Validator code |
| --- | --- | --- | --- |
| `index_shift` | Add one to index labels | Integer index | `index_mismatch` |
| `target_swap` | Reverse target names, leave values in place | At least two targets | `target_mismatch` |
| `horizon_truncation` | Drop the last horizon in every output and index | No additional condition | `index_mismatch` |
| `nonfinite_point` | Set the first point to NaN | Nonempty explicit points | `nonfinite_point` |
| `quantile_crossing` | Reverse the quantile axis, retain levels | At least two distinct ordered quantiles | `quantile_crossing` |
| `sample_axis_swap` | Transpose sample and horizon axes | Samples with unequal sample/horizon sizes | `sample_shape` |
| `variance_shrink` | Multiply deviations from points by `1 - 0.75 * strength` | Point, quantile and sample outputs | None; distribution checks may detect it |
| `dependence_shuffle` | Independently permute selected samples per coordinate | Sample outputs | None; marginals are preserved |

`strength` lies in `[0, 1]` and controls only the last two transformations;
`seed` controls only the shuffle and must be a nonnegative integer. The other
fixtures ignore these controls after validation. `enabled=False` returns the
prepared copy without applying the fault, but still validates the name, seed,
strength and preconditions. This bypass is a specified repair, not repair discovery.

## Signatures

::: timegym.mutations.mutate_forecast
