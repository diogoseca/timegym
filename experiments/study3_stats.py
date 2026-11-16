"""The statistics of Study 3, Parts 1 and 2 (protocol 3.0): per-mutant decisions, funnel rates, curves, shift classes.

- Calibrated per-case alarms (loss and answer key): x alarms among m evaluated cases against the k-th order
  statistic of n clean calibration scores give p = P(X >= x), X ~ BetaBinomial(m, n + 1 - k, k)
  (shared.per_mutant_statistical_rule.split_calibration_alarms; k = 228 and n + 1 - k = 12 for n = 239).
- The calibrated shift check: x FAILs among m cases, exact binomial tail against 0.05.
- The paired check: experiments.run_addendum.sign_flip_pvalue with 9999 flips and
  numpy.random.default_rng([7600000, number of pairs]).
- Holm (experiments.run_supplement.holm_rejections) within a family, with the unadjusted decision p <= alpha
  beside it. Members without a p-value (no evaluated case) are left out of the family and reported as such.
- The deterministic rule: caught when the check alarms on at least one case, valid only when the clean pipeline
  never alarms in that condition and learner.
- Wilson intervals (timegym.statistics.wilson_interval), detection and false-alarm curves over k/240, and the
  classification rule of Part 2.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import betabinom, binom

from timegym.statistics import wilson_interval

from .run_addendum import sign_flip_pvalue
from .run_supplement import conformal_threshold, holm_rejections

ALPHA = 0.05
CALIBRATION_CASES = 239
SIGN_FLIPS = 9999
SIGN_FLIP_TAG = 7_600_000
CLASSIFICATION_RATE = 0.05
INTERVAL_LEVELS = (3, 6, 12, 24)


def calibration_rank(n_calibration: int = CALIBRATION_CASES, alpha: float = ALPHA) -> int:
    """The order statistic k of the split-calibration threshold (228 for 239 cases at 0.05)."""
    return conformal_threshold(np.zeros(n_calibration), alpha)[1]


def beta_binomial_pvalue(alarms: int, cases: int, n_calibration: int = CALIBRATION_CASES,
                         alpha: float = ALPHA) -> float | None:
    """P(X >= alarms) for X ~ BetaBinomial(cases, n + 1 - k, k); None without evaluated cases."""
    if cases == 0:
        return None
    if not 0 <= alarms <= cases:
        raise ValueError("alarms must lie between 0 and the number of cases")
    k = calibration_rank(n_calibration, alpha)
    return float(min(1.0, betabinom.sf(alarms - 1, cases, n_calibration + 1 - k, k)))


def binomial_pvalue(fails: int, cases: int, rate: float = CLASSIFICATION_RATE) -> float | None:
    """P(X >= fails) for X ~ Binomial(cases, rate); None without cases."""
    if cases == 0:
        return None
    if not 0 <= fails <= cases:
        raise ValueError("fails must lie between 0 and the number of cases")
    return float(min(1.0, binom.sf(fails - 1, cases, rate)))


def paired_pvalue(differences, tag: int = SIGN_FLIP_TAG, flips: int = SIGN_FLIPS) -> float | None:
    """One-sided sign-flip p-value of a positive mean of d = distance(mutant) - distance(clean)."""
    differences = np.asarray(differences, dtype=float)
    if differences.size == 0:
        return None
    if differences.ndim != 1 or not np.isfinite(differences).all():
        raise ValueError("Paired differences must be a finite vector")
    return float(sign_flip_pvalue(differences, flips, np.random.default_rng([tag, len(differences)])))


def alarms_needed(cases: int, family_size: int, alpha: float = ALPHA,
                  n_calibration: int = CALIBRATION_CASES) -> int | None:
    """Smallest alarm count whose beta-binomial p-value reaches the first Holm threshold alpha / family_size."""
    for alarms in range(cases + 1):
        if beta_binomial_pvalue(alarms, cases, n_calibration, alpha) <= alpha / family_size:
            return alarms
    return None


def holm_decisions(pvalues: dict, alpha: float = ALPHA) -> dict:
    """Holm and unadjusted decisions for one family; members whose p-value is None are not in the family."""
    members = [name for name, value in pvalues.items() if value is not None]
    rejected = holm_rejections([pvalues[name] for name in members], alpha) if members else []
    holm = dict(zip(members, (bool(flag) for flag in rejected)))
    return {name: {"p_value": value, "family_size": len(members),
                   "unadjusted": None if value is None else bool(value <= alpha),
                   "holm": holm.get(name)} for name, value in pvalues.items()}


def deterministic_decision(alarms: int, cases: int, clean_alarms: int) -> dict:
    """shared.per_mutant_deterministic_rule for one check, mutant and condition."""
    deterministic = clean_alarms == 0
    return {"alarms": alarms, "cases": cases, "clean_alarms": clean_alarms, "deterministic": deterministic,
            "caught": (alarms >= 1) if deterministic and cases else None}


def rate(count: int, n: int) -> dict | None:
    """Wilson 95% interval of count successes among n, or None when n is 0."""
    if n == 0:
        return None
    if not 0 <= count <= n:
        raise ValueError("count must lie between 0 and n")
    return wilson_interval([True] * count + [False] * (n - count))


def rank_counts(scores, calibration) -> list[int]:
    """For each score, j = 1 + the number of calibration scores >= it, so that its rank p-value is j / (n + 1)."""
    calibration = np.asarray(calibration, dtype=float)
    return [1 + int(np.count_nonzero(calibration >= score)) for score in scores]


def detection_curve(ranks_by_member: list[list[int]], levels: int = CALIBRATION_CASES + 1) -> list[float | None]:
    """At each level k/levels (k = 1..levels): the alarm rate of each member among its cases, averaged over
    members with at least one case, so that every member weighs the same."""
    members = [np.asarray(ranks) for ranks in ranks_by_member if len(ranks)]
    if not members:
        return [None] * levels
    return [float(np.mean([np.mean(ranks <= k) for ranks in members])) for k in range(1, levels + 1)]


def curve_intervals(ranks_by_member: list[list[int]], levels=INTERVAL_LEVELS) -> dict:
    """Wilson intervals over all (member, case) pairs at the levels k/240 of shared curves.interval."""
    pooled = np.concatenate([np.asarray(ranks, dtype=int) for ranks in ranks_by_member] or [np.zeros(0, int)])
    return {str(k): rate(int(np.count_nonzero(pooled <= k)), int(pooled.size)) for k in levels}


def shift_class(public_statuses: list[str], calibrated_fails: int | None = None,
                calibrated_cases: int | None = None, rate_under_null: float = CLASSIFICATION_RATE,
                alpha: float = ALPHA) -> dict:
    """part_2_learners.shift_classification.rule: exact, in distribution or not applicable."""
    passes = sum(status == "PASS" for status in public_statuses)
    row = {"public_pass": passes, "public_cases": len(public_statuses), "calibrated_fails": calibrated_fails,
           "calibrated_cases": calibrated_cases, "binomial_p_value": None}
    if public_statuses and passes == len(public_statuses):
        return {**row, "class": "exact"}
    if calibrated_fails is None:
        return {**row, "class": None}
    p_value = binomial_pvalue(calibrated_fails, calibrated_cases, rate_under_null)
    return {**row, "binomial_p_value": p_value, "class": "in distribution" if p_value > alpha else "not applicable"}
