# Reproduce the controlled studies

From the repository root, create the development environment once:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install -r requirements-study.txt
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
`runtime.json` files. Environment-variable prefixes are unnecessary.

## Study 1: controlled pilot

```bash
.venv/bin/python -m experiments.run_study --output /tmp/timegym-pilot-repeat
.venv/bin/python -m experiments.compare_results --results /tmp/timegym-pilot-repeat/results.json --manifest experiments/results/manifest.json
```

The exact environment lock records the tested Python 3.13 environment; supported
Python versions can instead install `pip install -e '.[dev]'`. Packages and the
public dataset are available locally after setup; the experiment itself uses no
network services or paid compute.

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
outputs; this illustrates a restricted numerical comparator, not superiority to
properly validated evaluation frameworks.

The separate model-level fault omits inverse standardisation in a historical-mean
forecast. Its repair applies the training location and scale exactly once. The
location-equivariance relation is explicitly declared for this model. The same
repair is illustrated at four ordered AirPassengers origins, without treating
them as independent replications. It does not make this weak baseline a good
forecasting model.

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

For a numerical repeat, run `python -m experiments.run_study --output /tmp/timegym-repeat`
and compare `results.json`. There are no timestamps or runtime values in numerical
results. A platform-dependent tolerance may be needed on other BLAS/NumPy versions.
The CI `research-smoke` job compares both rerun hashes to their frozen manifests,
warns on platform differences, and uploads the observed Linux hashes as
`research-linux-hashes`; see [the validation record](../paper/validation.md).
Local comparisons fail on a mismatch
unless `--warn` is explicit.

To reproduce one selected controlled counterexample without running the full
study, use `python -m experiments.reproduce_fault --mutation dependence_shuffle --seed 10000`.
This reconstructs its saved scenario, ensemble and mutation seeds, then compares
the contract and covariance decisions with the original record. Its successful
exit means reproduction matched, even when the forecast correctly receives FAIL.

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

### Frozen-record migration

The 30 October 2026 update did not rerun either study. `migrate_frozen.py` removed
only verified duplicate repaired records and unused alarm-difference t intervals,
updated protocol metadata, and regenerated CSVs. Pilot numerical bytes and all
retained supplement measurements and paired loss summaries stayed unchanged.
Each manifest preserves its original manifest and input hash under
`record_migration`; historical source/environment evidence is not replaced by
current code hashes. The supplement operation counts now describe the retained
branches; the separate historical timing includes the removed duplicate work.
The migration can be replayed with `python -m experiments.migrate_frozen --root PATH`
on a checkout containing the pre-migration records and the updated protocols.
An already migrated record is rejected.

The 3 November 2026 update renamed the supplement's paired loss fields
`repair_minus_faulty_mse` and `repair_minus_faulty_reference_mse` to
`clean_minus_faulty_mse` and `clean_minus_faulty_reference_mse`, in `results.json`
and the `model-repair-losses.csv` header. No study was rerun and no value changed.
`python -m experiments.migrate_frozen --root PATH --step loss-keys` applies the
rename to a record matching its manifest, rewrites the CSVs and the manifest hash,
and logs the original hash under `loss_key_migration`. An already renamed record
is rejected.
