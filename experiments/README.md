# Reproduce the controlled studies

From the repository root, with Python 3.13 (the version the lock records),
create the environment once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-study.txt
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

Render the frozen records without running either study:

```bash
.venv/bin/python -m experiments.render_results
.venv/bin/python -m experiments.render_supplement
.venv/bin/python paper/render_figures.py
.venv/bin/python paper/build_standalone.py
```

Both runners refuse existing `results.json`, `manifest.json`, `runtime.json`,
`example-*.json`, or CSV artifacts before doing any computation. `--force` is an
explicit replacement option for an approved re-freeze; routine reproductions
use a new output directory. Both runners enforce one numerical thread with
`threadpoolctl` and record the observed environment and thread pools in separate
`runtime.json` files.

## Study 1: controlled pilot

```bash
.venv/bin/python -m experiments.run_study --output /tmp/timegym-pilot-repeat
.venv/bin/python -m experiments.compare_results --results /tmp/timegym-pilot-repeat/results.json --manifest experiments/results/manifest.json
```

Packages and the public dataset are available locally after setup; the
experiment itself uses no network services or paid compute.

`protocol.json` version 1.1 makes the original capability task and inverse-transform
dimensions explicit, with unchanged values. The runner validates the protocol
and constructs only the models it declares. The original pilot design predates
inspection of its evaluation results.
Development seeds 0–15 calibrate a deliberately simple point-MSE alarm. Evaluation
seeds 10000–10063 are independent series. Every output mutation is paired with the
same clean and repaired ensemble. Six mutations violate the explicit output
contract; variance shrinkage and coordinate-wise sample shuffling preserve output
shapes. The shuffle preserves every empirical marginal distribution exactly.

The statistical check uses 99 independent reference ensembles and a per-case
0.05 rank-test threshold. Its null is independent sampling from the full known
reference law. It is not a test of arbitrary covariance equality, and it is not
a guarantee that a clean ensemble will pass. False alarms and Wilson confidence
intervals are retained in the paper. The MSE alarm ignores index labels and non-point
outputs.

The separate model-level fault omits inverse standardization in a historical-mean
forecast. Its repair applies the training location and scale exactly once. The
location-equivariance relation is explicitly declared for this model. The same
repair is illustrated at four ordered AirPassengers origins, without treating
them as independent replications.

Outputs:

- `results/results.json`: individual cases, scores, confidence intervals and summaries.
- `results/manifest.json`: protocol, dependency versions, base Git commit, current source
  fingerprint, results hash, and dataset hash. The source fingerprint identifies
  historical working files; the offline metadata migration preserves that evidence.
- `results/runtime.json`: thread environment for new executions; the frozen pilot
  predates this metadata, so its historical thread state is unknown.
- `results/example-*.json`: machine-readable counterexamples.
- `../paper/figures/*.png` and `../paper/figures/*.pdf`: figures rendered from the recorded results.
- `../paper/generated-results.tex`: manuscript tables derived from the same JSON.

There are no timestamps or runtime values in numerical results. A
platform-dependent tolerance may be needed on other BLAS/NumPy versions.
The CI `research-smoke` job reruns both studies and renders the paper's tables
from the fresh output, so it fails when a conclusion the paper states does not
hold there; it does not compare hashes. See [the validation record](../paper/validation.md).
Local comparisons fail on a mismatch
unless `--warn` is explicit.

Scope limits: no held-out generator families, representative real-bug corpus,
foundation models, sktime-conformance comparator, or broad real-data transfer study.

## Study 2: implementation and covariance supplement

`protocol-v2.json` keeps the supplement separate from the pilot. It declares the
four implementation faults, six model conditions, calibration/evaluation seeds,
and the permutation-invariant covariance null set. A contract violation aborts
the model study. Corrected means the clean implementation refitted; comparisons
are clean-minus-faulty, with no duplicate repaired arm. The paired loss fields
are `clean_minus_faulty_mse` and `clean_minus_faulty_reference_mse`; the
`model-repair-losses.csv` filename is retained for compatibility.

```bash
.venv/bin/python -m experiments.run_supplement --output /tmp/timegym-supplement-repeat
.venv/bin/python -m experiments.compare_results --results /tmp/timegym-supplement-repeat/results.json --manifest experiments/results-v2/manifest.json
.venv/bin/python -m experiments.render_supplement --results-dir /tmp/timegym-supplement-repeat --paper-dir /tmp/timegym-supplement-paper
```

The renderer checks the manuscript conclusions against the supplied records and
raises if they are unsupported. Small smoke runs may therefore be unsuitable for
rendering the frozen manuscript's empirical conclusions.

The `results-v2/` output contains:

- `results.json`: calibration, implementation and covariance records, paired
  loss summaries, gained/lost alarm counts, and deterministic operation counts.
- `manifest.json`: protocol, result, source, environment, and pilot-result hashes.
- `runtime.json`: observed wall time and thread state, excluded from manuscript
  text and numerical results.
- `model-ablation.csv`: clean and faulty alarm totals.
- `model-conditions.csv`: per-condition clean/faulty counts and gained/lost alarms.
- `model-repair-losses.csv`: clean-minus-faulty MSE, paired t intervals, and
  individually improved cases.
- `covariance-conditions.csv`: raw and Holm counts and Wilson intervals.

The supplement renderer also emits `paper/generated-macros.tex` for caption
numbers (`RepairDeltaMean`, `RepairDeltaLow`, `RepairDeltaHigh`, `RepairImproved`,
`CasesPerCondition`, and `CasesPerRow`). It checks these against paired raw losses.

## Exploratory follow-up to Study 2 (protocol 2.1)

`protocol-v2-followup.json` declares four analyses chosen on 13 November 2026,
after the protocol 2.0 results were known. They are exploratory, and the paper
labels every number from them as such. The runner imports the frozen Study 1
and Study 2 code, re-runs only the fits it needs, checks each re-run value
against the frozen records (within 1e-9), and never writes to `results/` or
`results-v2/`:

```bash
.venv/bin/python -m experiments.run_followup --output /tmp/timegym-followup-repeat
cmp experiments/results-v2-followup/results.json /tmp/timegym-followup-repeat/results.json
```

It takes well under a minute on one CPU thread. The record answers:

- **Answer-key alarm.** The 768 clean Study 2 calibration fits are re-run; the
  same conformal rule at 5% is applied to each case's mean squared distance
  from the answer key, and the threshold to the frozen evaluation records'
  `mean_reference_mse`. Counts appear beside the frozen loss alarm in
  column K of the complete Study 2 table and in the diagnostic-evidence figure.
- **Repair mechanism** on the random walk with drift: own-lag coefficient sums
  of the clean fit, the per-step distance from the answer key of the clean and
  stale-state branches, and the frozen paired distance from the answer key.
- **Scorecard precision:** the Figure 1 pipeline with 30 instances per cell and
  a bootstrap of ten-instance scores.
- **Fitted AR(1) on the trend** in the Study 1 capability setting: its mean
  coefficient and its mean offset from the answer key.

The `results-v2-followup/` output contains `results.json` (including the 768
stored calibration distances) and `manifest.json` (protocol, results and
frozen-input digests, source digests, Python, library and package versions).
`experiments/render_followup.py` validates the record and supplies its macros
and table column to `render_results` and `render_supplement`; both accept
`--followup-dir`. The tests recompute the thresholds and counts from the stored
distances and the frozen records without re-running any fit.

## Second exploratory record for Study 2 (protocol 2.2)

`protocol-v2-addendum.json` declares two analyses chosen on 13 November 2026,
after the protocol 2.0 results and the protocol 2.1 follow-up were known. They
are exploratory and read only values stored in `results-v2/results.json`;
nothing is refitted, and the runner never writes to `results/`, `results-v2/`
or `results-v2-followup/`:

```bash
.venv/bin/python -m experiments.run_addendum --output /tmp/timegym-addendum-repeat
cmp experiments/results-v2-addendum/results.json /tmp/timegym-addendum-repeat/results.json
```

It takes a few seconds. The record answers:

- **Suite-level multiplicity.** From the stored p-values of the four condition
  checks in each of the 64 suites, per variant: Holm at 0.05 (recomputed, and
  required to equal every stored `holm_alarm`), Hochberg's step-up procedure,
  Simes' global test and Fisher's combination (chi-square with 8 degrees of
  freedom, in closed form). It stores suites with any rejection and suites
  rejecting a true null for each method, with Wilson 95% intervals, and
  per-condition rejection counts for raw, Holm and Hochberg decisions.
- **Paired answer-key check.** For each of the 24 pairs of condition and fault,
  the paired differences of the stored `mean_reference_mse` against the clean
  branch on the same seeds, a one-sided sign-flip test with 9,999 flips (seed 0,
  declared in the protocol), and Holm across the 24 pairs. The same analysis on
  the first 30 and the first 10 evaluation seeds is a sensitivity check.

The `results-v2-addendum/` output contains `results.json` (including every
suite's p-values and decisions) and `manifest.json` (protocol, results and
frozen-input digests, source digests, Python, library and package versions).
`experiments/render_addendum.py` validates the record, checks the statements
the manuscript makes from it (`check_addendum_conclusions`) and supplies its
macros to `render_supplement`, which accepts `--addendum-dir`. The tests
recompute both analyses from the frozen records. The `research-smoke` job of
the checks workflow re-runs the follow-up and this record and checks their
conclusions on the fresh output.

### Frozen-record migration

The records were migrated twice without rerunning either study (30 October and 3 November 2026); no value changed. Each manifest keeps the original hashes under `record_migration` and `loss_key_migration`.

## Study 3 (protocol 3.0)

`protocol-v3.json` is the plan of the paper's third study. It is fixed before
anything is run (a pre-registration), so that the results cannot shape the
questions. Its field `status` stays `draft` until it is merged. The run may
not start until a later commit writes the merge commit and its date into the
file and sets the status to `registered`.

The study asks six questions:

1. A tool plants hundreds of small bugs in a correct forecasting pipeline.
   Which checks catch them?
2. Do the checks still work when the model inside the pipeline is not a
   linear regression?
3. When one statistical check runs on several settings at once, can it answer
   both "did anything fail?" and "which one failed?"
4. Can the joint-spread check also say "close enough", and not only
   "different"?
5. How close do two published estimates of predictability come to the exact
   unavoidable error, which the answer key gives?
6. What scorecards do forecasting models that people use get?

Terms used below:

- **Mutation testing** makes many small deliberate changes to a correct
  program and counts how many of them the checks notice. Each changed copy is
  a **mutant**, and a rule that makes one kind of change is an **operator**.
  An **equivalent mutant** behaves exactly like the original, so no check
  could catch it.
- The **subject** is the program under test, here one small forecasting
  pipeline.
- A **second implementation** is the same pipeline written again
  independently. Where the two disagree, one of them has a bug.
- **Calibration cases** are runs of the bug-free pipeline. They show how
  large a statistic gets by chance, so that a check can tell a bug from noise.
- The **catalogue** is this study's list of 16 hand-written time-series bugs.
- The **zoo** is the set of nine forecasting models scored for question 6.

Once the plan is registered, the run is these commands in this order. The
first uses the Cosmic Ray environment, the last three the zoo's environments,
and the others the study lock:

    python -m experiments.study3_generate --output experiments/results-v3/mutants
    python -m experiments.run_study3_mutants --part 1 --workers 4
    python -m experiments.run_study3_mutants --part 2 --workers 4
    python -m experiments.run_study3_suites
    python -m experiments.run_study3_margin --workers 4
    python -m experiments.run_study3_predictability
    python -m experiments.zoo.run_zoo
    python -m experiments.zoo.run_zoo --models chronos_small
    python -m experiments.zoo.figure

Every runner first asks `experiments/study3_common.py` whether its part may
start and stops with the list of what is missing; no flag bypasses that. With
`--plan` a runner prints what it would run and computes nothing, which also
works on the draft. Part 2 takes hours and writes its record at the end, so
`run_study3_mutants` has `--resume` for a run that a stop of the machine
interrupted: started again with the same `--output`, it keeps every unit of
work that the stopped run ended under the same specification and time limit
and runs the others. After the merge, three commits may each change one value
of the protocol and nothing else: the one that registers it (date, merge
commit and status), the one that adds the generated mutant sources and writes
the digest of their manifest, without which Parts 1 and 2 do not start, and
the one that writes the Chronos model revision. The four files that Part 1
freezes (the subject, its second implementation, the catalogue and the
learners) carry their digests in the protocol already, and a test fails when
one of them changes.

The study runs on one numerical thread per process, on CPU only. Every Monte
Carlo rank check uses 239 calibration draws, so that p-values are multiples of
1/240 and the four Holm thresholds of a family of four at 0.05 (0.0125,
0.0167, 0.025 and 0.05) are attained exactly. Calibrated per-case alarms use
239 clean calibration cases per condition. The number of evaluation cases per
condition, 93, and the number of covariance suites, 93, are the smallest
numbers for which every Wilson 95% interval has a half-width of at most 10
percentage points. Every case seed lies in a new namespace between 7,000,000
and 15,000,000; the protocol lists the seed ranges of the earlier protocols
and of the package, and loading the protocol fails if any of its own ranges
overlap or leave that namespace. Rates carry Wilson 95% intervals, and every
case that cannot be evaluated is counted and reported.

The six parts answer the six questions, in this order:

1. **Generated mutants.** A lag-regression pipeline written for this study
   (`experiments/study3_subject.py`: lags, a seasonal lag of 12, scaling of
   inputs and target, a pluggable regressor, recursive forecasting and the
   inverse transform) is committed and hashed before any mutant exists.
   Mutants come from Cosmic Ray 8.9.0 with its standard operators and from a
   catalogue of ten kinds of time-series bug, written as 16 operators and
   defined in the protocol in a sentence or two each; the four Study 2 bugs
   are re-expressed on the same pipeline. All mutant sources are generated and
   their digests committed before any is executed. Each mutant is kept only
   if it was generated, runs, and is not equivalent to the clean pipeline on
   24 fixed inputs; the counts at each step are reported. It is then run
   through the output checks, the common and
   per-series shift checks, a second implementation, the calibrated loss
   alarm, the answer-key alarm of protocol 2.1, and a paired sign-flip check
   against the clean pipeline, in the six Study 2 model conditions. A
   deterministic check catches a mutant in a condition when it alarms on any
   case, provided it never alarms on the clean pipeline there. A calibrated
   alarm catches it when an exact beta-binomial test of its alarm count,
   Holm-adjusted across the mutants of the condition, rejects; the unadjusted
   decision is reported beside it. Detection rates are also drawn against
   false-alarm rates over all 240 attainable levels.
2. **Learners beyond Ridge.** Histogram gradient boosting on raw lags, a
   perceptron on window-standardized lags with a free training seed, and a
   perceptron on raw lags are first classified on clean pipelines as exactly
   shift equivariant, equivariant in distribution (checked with 39 refits per
   case), or not applicable (not shift equivariant even in distribution, so the
   shift checks do not apply). Each then receives the four Study 2 bugs, the 16
   catalogue mutants and 12 seeded generic mutants, checked with the checks
   that apply to its shift class.
3. **Two questions about a suite.** The covariance study of protocol 2.0 is
   repeated with 239 calibration draws: Fisher's combination answers whether
   a suite contains any failure, and Holm answers which condition failed.
4. **A joint-spread check with a margin.** An unbiased estimate of the
   squared relative Frobenius distance between the ensemble covariance and
   the reference covariance is compared, through a leave-one-out (jackknife)
   interval, with a margin of 0.25 or 0.4375 (the exact distance of the 0.75
   spread multiplier of Study 2). A simulation with 381 repetitions per cell
   measures how often each verdict occurs below, at and above the margin, with
   64 and 256 paths.
5. **Predictability estimators against the exact floor.** On 200 series of
   each of the 20 default scorecard cells, the spectral bound of Feng et al.,
   computed as their paper describes and also from the history alone, and
   weighted permutation entropy are compared with the unavoidable error that
   the answer key gives. The expectations for each row are written in the
   protocol.
6. **Scorecards of models people use.** Nine models are scored on the default
   grid with 30 instances, from `experiments/zoo/`, which has its own lock
   files: last value, seasonal naive, ETS, automatic ARIMA and Theta from
   statsforecast 2.1.1, gradient boosting and a perceptron on lags, Chronos
   (small) and context parroting. The pretrained model may run on another
   machine; the protocol states what its manifest records, and its model
   revision is written into the protocol before that run. No ranking claim is
   made.

Before registration, only the computations listed under
`pilots_before_registration` were made: installing Cosmic Ray and listing its
operators; counting, without executing any of them, the mutants it generates
on `experiments/model_faults.py` and on a copy of its clean branch; timing
clean pipelines and running the public shift check on 12 clean cases of each
learner, and of seven regressors on raw lags; installing statsforecast and
timing it on two generic series; testing which package indexes, model hosts
and literature servers this machine can reach; reading the cited papers and
the published code of the spectral bound; and checking formulas on textbook
cases. When the code was written, the clean subject alone was run on the 12
pilot cases, to confirm that its forecasts are valid, that the second
implementation agrees with it and that the shift checks are silent on it; the
catalogue was generated as text and compared with the expected text, never
executed. The protocol also says what the tests of the Study 3 code execute:
toy pipelines, toy mutants and hand-made arrays, on seeds below 1,000,000. The
protocol also lists the earlier exploratory records that bear on
the study: protocols 2.1 and 2.2, and measurements made while the scorecard's
comparison with a saved baseline and the interval beside its scores were
developed. The interval was changed on those measurements, which included
scores of the last-value forecaster, before the plan was registered. No mutant
and no Study 2 bug was executed on a new learner, the margin simulation was
not run, neither predictability estimator was run on a scorecard cell, and no
scorecard of a Part 6 model was made other than those of the two built-in
baselines, at seeds outside this protocol.
