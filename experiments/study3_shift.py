"""The calibrated shift check of Study 3, Part 2 (protocol 3.0, part_2_learners.calibrated_shift_check).

For a learner with a free training seed, a shift of the context by c should move the law of the forecast by c,
not each forecast. On one case, from the learner's seed stream in this order: the forecast fit R on the context
(draw 1, the forecast every other check evaluates), 39 refits F_1..F_39 on the context (draws 2-40) and one fit G on
the context plus c (draw 41). The calibration values are dist(R, F_j), the observed value dist(R, G - c), with
dist(a, b) the mean over horizon steps and targets of (a - b)^2; timegym.diagnostics.rank_alarm at alpha 0.05 gives
the p-value, a multiple of 1/40, and FAIL when at most one calibration value is at least the observed one.

When G - c equals R within the tolerances of the deterministic checks (numpy.allclose), the check returns PASS
whatever the ranks. A forecast that does not depend on the training seed has 39 calibration distances of exactly
zero, so rounding in G - c alone would otherwise rank above them and FAIL. The rank p-value is still returned.

The stream is the one the pipeline's forecaster holds: every fit of the pipeline draws the next seed, so the draw
order is the order of the calls below. The per-target shift check is
experiments.model_faults.target_shift_agreement, unchanged.

This module is not part of the package.
"""
from __future__ import annotations

import numpy as np

from timegym.diagnostics import rank_alarm

REFITS = 39
SHIFT = 10.0
ALPHA = 0.05
ATOL, RTOL = 1e-8, 1e-7


def distance(a, b) -> float:
    """Mean over horizon steps and targets of the squared difference."""
    return float(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2))


def _points(forecast, shape):
    point = None if forecast.point is None else np.asarray(forecast.point, dtype=float)
    if point is None or point.shape != shape or not np.isfinite(point).all():
        raise ValueError("A refit returned no finite point forecast of the forecast's shape")
    return point


def calibrated_shift_check(pipeline, context, future_index, *, forecast=None, refits=REFITS, shift=SHIFT,
                           alpha=ALPHA, atol=ATOL, rtol=RTOL) -> dict:
    """Run the check on one case and return status (PASS, FAIL or ERROR), p_value, within_tolerance and the distances.

    ``forecast`` is R when the caller has already made it with the same pipeline (draw 1); otherwise R is fitted
    here first. An exception, or a refit without a finite point forecast of R's shape, gives ERROR with its message.
    """
    try:
        first = pipeline.predict(context, future_index) if forecast is None else forecast
        reference = np.asarray(first.point, dtype=float)
        if reference.ndim != 2 or not np.isfinite(reference).all():
            raise ValueError("The forecast R has no finite two-dimensional point forecast")
        calibration = [distance(reference, _points(pipeline.predict(context, future_index), reference.shape))
                       for _ in range(refits)]
        moved = _points(pipeline.predict(context + shift, future_index), reference.shape)
        observed = distance(reference, moved - shift)
        within = bool(np.allclose(moved - shift, reference, atol=atol, rtol=rtol))
        alarm = rank_alarm(observed, np.asarray(calibration), alpha=alpha)
    except Exception as exc:  # a mutant may raise anywhere; the check reports it
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "p_value": None,
                "within_tolerance": None, "observed": None, "calibration_max": None}
    return {"status": "PASS" if within else alarm.status, "error": None, "p_value": float(alarm.p_value),
            "within_tolerance": within, "observed": observed, "calibration_max": float(max(calibration))}
