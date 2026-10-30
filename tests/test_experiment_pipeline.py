from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from experiments import run_study, run_supplement
from experiments.common import interval_cell, protect_output, sha256
from experiments.render_results import capability_summary, render_tex as pilot_tex
from experiments.render_supplement import render_macros, render_tex as supplement_tex

ROOT = Path(__file__).resolve().parents[1]


def records(directory):
    path = ROOT / 'experiments' / directory
    return json.loads((path / 'results.json').read_text()), json.loads((path / 'manifest.json').read_text())


@pytest.mark.parametrize('runner,folder', [(run_study, 'results'), (run_supplement, 'results-v2')])
def test_cli_refuses_frozen_output_before_running(runner, folder, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['study', '--output', str(ROOT / 'experiments' / folder)])
    with pytest.raises(FileExistsError, match='results.json'):
        runner.main()


@pytest.mark.parametrize('filename', ['results.json', 'manifest.json', 'example-test.json', 'runtime.json', 'model-ablation.csv'])
def test_overwrite_requires_explicit_force(tmp_path, filename):
    (tmp_path / filename).write_text('original')
    with pytest.raises(FileExistsError, match=filename):
        protect_output(tmp_path)
    protect_output(tmp_path, force=True)
    assert (tmp_path / filename).read_text() == 'original'


def test_pilot_rejects_unknown_models_overlap_and_incomplete_task():
    protocol = json.loads((ROOT / 'experiments/protocol.json').read_text())
    run_study.validate_protocol(protocol)
    bad = deepcopy(protocol)
    bad['capability_models'] = ['invented']
    with pytest.raises(ValueError, match='capability_models'):
        run_study.validate_protocol(bad)
    bad = deepcopy(protocol)
    bad['development_seeds'] = protocol['evaluation_seeds']
    with pytest.raises(ValueError, match='overlap'):
        run_study.validate_protocol(bad)
    bad = deepcopy(protocol)
    bad['capability_task']['horizon'] = 0
    with pytest.raises(ValueError, match='horizon'):
        run_study.validate_protocol(bad)


def test_capability_runner_uses_declared_models_and_dimensions(monkeypatch):
    protocol = json.loads((ROOT / 'experiments/protocol.json').read_text())
    protocol.update(capability_models=['last_value'], capability_scenarios=['white_noise'],
                    capability_sigma=[1.], capability_seeds=[7, 9])
    protocol['capability_task'].update(context_length=16, horizon=3, n_targets=2)
    observed = []
    original = run_study.Pipeline.predict
    def predict(self, context, future_index):
        observed.append((context.shape, len(future_index)))
        return original(self, context, future_index)
    monkeypatch.setattr(run_study.Pipeline, 'predict', predict)
    rows = run_study.capability_study(protocol)
    assert {row['model'] for row in rows} == {'last_value'}
    assert observed == [((16, 2), 3)] * 2


def test_pilot_source_drift_aborts_before_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['study', '--output', str(tmp_path)])
    fingerprints = iter([{'source': 'before'}, {'source': 'after'}])
    monkeypatch.setattr(run_study, 'source_fingerprints', lambda: next(fingerprints))
    monkeypatch.setattr(run_study, 'run', lambda protocol: {})
    monkeypatch.setattr(run_study, 'capability_study', lambda protocol: [])
    monkeypatch.setattr(run_study, 'inverse_repair_study', lambda protocol: {})
    monkeypatch.setattr(run_study, 'provenance', lambda *args: {})
    with pytest.raises(RuntimeError, match='Source files changed'):
        run_study.main()
    assert list(tmp_path.iterdir()) == []


def test_pilot_records_only_used_versions(monkeypatch):
    versions = []
    monkeypatch.setattr(run_study.importlib.metadata, 'version', lambda name: versions.append(name) or '1.0')
    run_study.provenance({})
    assert versions == ['numpy', 'pandas', 'scipy']


def test_permutation_null_validation_survives_rename_and_rejects_false_null():
    protocol = json.loads(run_supplement.PROTOCOL.read_text())
    spec = protocol['covariance_study']
    spec['conditions'][0]['id'] = 'renamed_null'
    spec['permutation_invariant_conditions'] = ['renamed_null']
    run_supplement.validate_protocol(protocol)
    for nulls in [['missing'], [spec['conditions'][1]['id']], ['renamed_null', 'renamed_null']]:
        spec['permutation_invariant_conditions'] = nulls
        with pytest.raises(ValueError, match='Permutation-invariant'):
            run_supplement.validate_protocol(protocol)


def test_frozen_fragments_and_macros_are_reproducible():
    pilot, manifest = records('results')
    text = pilot_tex(pilot, manifest, capability_summary(pilot, manifest['protocol']))
    assert text == (ROOT / 'paper/generated-results.tex').read_text()
    assert 'accompanying capability figure' not in text and 'scikit-learn' not in text
    supplement, manifest = records('results-v2')
    text = supplement_tex(supplement, manifest)
    assert text == (ROOT / 'paper/generated-supplement.tex').read_text()
    assert 'Fresh corrected refit' not in text and 'wall time' not in text
    assert 'Reduced spread & --- & ---' in text
    assert 'clean-minus-faulty mean MSE was +0.5132 [+0.2122, +0.8141]' in text
    macros = render_macros(supplement)
    assert macros == (ROOT / 'paper/generated-macros.tex').read_text()
    for macro, value in [('RepairDeltaMean', '0.513'), ('RepairDeltaLow', '0.212'), ('RepairDeltaHigh', '0.814'),
                         ('RepairImproved', '21'), ('CasesPerCondition', '64'), ('CasesPerRow', '384')]:
        assert rf'\newcommand{{\{macro}}}{{{value}}}' in macros


@pytest.mark.parametrize('change', ['contract', 'covariance', 'energy', 'alpha', 'repair', 'real'])
def test_pilot_conclusions_reject_contradictory_records(change):
    data, manifest = records('results')
    summary = capability_summary(data, manifest['protocol'])
    if change == 'contract': data['fault_summary']['index_shift']['contract']['count'] -= 1
    if change == 'covariance': data['fault_summary']['variance_shrink']['covariance']['count'] -= 1
    if change == 'energy': data['dependence_summary']['1.0']['energy']['ci95'][0] = -1
    if change == 'alpha': data['fault_summary']['clean']['covariance']['rate'] = 0
    if change == 'repair': data['fault_summary']['index_shift']['repair_combined']['count'] = 0
    if change == 'real': data['inverse_repair']['real'][0]['repaired_mae'] = 1e6
    with pytest.raises(ValueError, match='Recorded evidence'):
        pilot_tex(data, manifest, summary)


@pytest.mark.parametrize('variant,key', [('clean', 'holm_any_false_alarm'), ('reduced_spread', 'holm_any_alarm')])
def test_supplement_conclusions_reject_contradictory_records(variant, key):
    data, manifest = records('results-v2')
    next(row for row in data['covariance']['suite_summaries'] if row['variant'] == variant)[key]['rate'] = 1
    with pytest.raises(ValueError, match='Recorded evidence'):
        supplement_tex(data, manifest)


def test_shared_interval_format():
    stat = {'mean': .1, 'ci95': [-.2, .3]}
    assert interval_cell(stat) == '+0.1000 [-0.2000, +0.3000]'
    assert interval_cell(stat, precision=2, signed=False, math=True) == '$0.10 [-0.20, 0.30]$'
    assert interval_cell(None) == '---'


def test_hash_comparison_reports_mismatch_and_strict_cli_fails(tmp_path, monkeypatch):
    from experiments.compare_results import compare, main
    results = tmp_path / 'results.json'
    manifest = tmp_path / 'manifest.json'
    report = tmp_path / 'comparison.json'
    results.write_text('{}')
    manifest.write_text(json.dumps({'results_sha256': sha256(results)}))
    assert compare(results, manifest)['matches']
    results.write_text('{"changed":true}')
    assert not compare(results, manifest)['matches']
    args = ['compare', '--results', str(results), '--manifest', str(manifest), '--report', str(report)]
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit, match='differs'):
        main()
    assert json.loads(report.read_text())['actual_sha256'] == sha256(results)
    monkeypatch.setattr(sys, 'argv', args + ['--warn'])
    main()


def test_frozen_migration_retained_cases_and_loss_statistics_are_consistent():
    from timegym.statistics import paired_t_interval, wilson_interval
    data, manifest = records('results-v2')
    original = manifest['record_migration']['original_manifest']
    assert manifest['source_files'] == original['source_files']
    assert manifest['versions'] == original['versions']
    assert manifest['record_migration']['original_results_sha256'] == original['results_sha256']
    model = data['model']
    assert len(model['records']) == 1920
    assert {row['branch'] for row in model['records']} == {'clean', *run_supplement.FAULTS}
    index = {(row['condition'], row['seed'], row['branch']): row for row in model['records']}
    for summary in model['loss_summaries']:
        broken = [row for row in model['records'] if row['condition'] == summary['condition'] and row['branch'] == summary['fault']]
        clean = [index[(row['condition'], row['seed'], 'clean')] for row in broken]
        delta = [a['realized_mse'] - b['realized_mse'] for a, b in zip(clean, broken)]
        assert paired_t_interval(delta) == summary['repair_minus_faulty_mse']
        assert sum(value < 0 for value in delta) == summary['cases_improved_by_repair']
    for summary in model['condition_summaries']:
        for branch, key in [('clean', 'clean'), (summary['fault'], 'faulty')]:
            alarms = [row['alarms'][summary['check']] for row in model['records']
                      if row['condition'] == summary['condition'] and row['branch'] == branch]
            assert wilson_interval(alarms) == summary[key]
    pilot, manifest = records('results')
    assert sha256(ROOT / 'experiments/results/results.json') == manifest['record_migration']['original_results_sha256']


def test_pilot_main_enforces_one_thread_and_writes_result_hash(monkeypatch, tmp_path):
    from threadpoolctl import threadpool_info
    seen = []
    def run(protocol):
        seen.extend(pool['num_threads'] for pool in threadpool_info())
        return {'example_reports': {}, 'fault_summary': {}, 'dependence_summary': {'1.0': {}}}
    monkeypatch.setattr(sys, 'argv', ['study', '--output', str(tmp_path)])
    monkeypatch.setattr(run_study, 'run', run)
    monkeypatch.setattr(run_study, 'capability_study', lambda protocol: [])
    monkeypatch.setattr(run_study, 'inverse_repair_study', lambda protocol: {'real': []})
    run_study.main()
    assert seen and all(count == 1 for count in seen)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['results_sha256'] == sha256(tmp_path / 'results.json')
    assert json.loads((tmp_path / 'runtime.json').read_text())['thread_limit'] == 1


def test_reuse_rejects_unexpected_sample_mutation(monkeypatch):
    import numpy as np
    protocol = json.loads((ROOT / 'experiments/protocol.json').read_text())
    protocol.update(evaluation_seeds=[10000, 10001], development_seeds=[0, 2], mutations=['index_shift'])
    original = run_study.mutate_forecast
    def mutate(forecast, name, **kwargs):
        if name == 'index_shift' and kwargs.get('enabled', True):
            changed = deepcopy(forecast)
            changed.samples = changed.samples * 2
            return changed
        return original(forecast, name, **kwargs)
    monkeypatch.setattr(run_study, 'mutate_forecast', mutate)
    with pytest.raises(AssertionError, match='different samples'):
        run_study.run(protocol)
