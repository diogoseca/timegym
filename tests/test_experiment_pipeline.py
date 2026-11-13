from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import pytest

from experiments import run_followup, run_study, run_supplement
from experiments.common import close_statistics, interval_cell, protect_output, sha256
from experiments.render_results import capability_summary, render_tex as pilot_tex
from experiments.render_followup import check_followup_conclusions, load_followup, validate_followup
from experiments.run_followup import answer_key_counts, answer_key_totals
from experiments.render_supplement import render_macros, render_tex as supplement_tex

ROOT = Path(__file__).resolve().parents[1]


def records(directory):
    path = ROOT / 'experiments' / directory
    return json.loads((path / 'results.json').read_text()), json.loads((path / 'manifest.json').read_text())


@pytest.mark.parametrize('runner,folder', [(run_study, 'results'), (run_supplement, 'results-v2'),
                                           (run_followup, 'results-v2-followup')])
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
    followup, _ = load_followup()
    pilot, manifest = records('results')
    text = pilot_tex(pilot, manifest, capability_summary(pilot, manifest['protocol']), followup)
    assert text == (ROOT / 'paper/generated-results.tex').read_text()
    assert 'accompanying capability figure' not in text and 'scikit-learn' not in text
    assert r'the estimate falls to about \TrendARPhiHigh{}' in text
    supplement, manifest = records('results-v2')
    text = supplement_tex(supplement, manifest, followup)
    assert text == (ROOT / 'paper/generated-supplement.tex').read_text()
    assert r'Condition & Branch & C & S & T & N & L & K \\' in text
    assert r' & Stale state & 0 & 0 & 0 & 64 & 5 & 28 \\' in text
    assert 'Fresh corrected refit' not in text and 'wall time' not in text
    assert 'Reduced spread & --- & ---' in text
    assert 'clean-minus-faulty mean MSE was +0.5132 [+0.2122, +0.8141]' in text
    macros = render_macros(supplement, manifest, records('results')[1], followup)
    assert macros == (ROOT / 'paper/generated-macros.tex').read_text()
    for macro, value in [('RepairDeltaMean', '0.513'), ('RepairDeltaLow', '0.212'), ('RepairDeltaHigh', '0.814'),
                         ('RepairImproved', '21'), ('CasesPerCondition', '64'), ('CasesPerRow', '384'),
                         ('StudySklearnFits', '8,448'), ('StudyCalibrationEnsembles', '152,832'),
                         ('AnswerKeySubtleAlarms', '148'), ('LossSubtleAlarms', '36'), ('SubtleCases', '640'),
                         ('AnswerKeyCleanAlarms', '12'), ('LossCleanAlarms', '13'), ('CleanNonSeasonalCases', '320'),
                         ('OwnLagSumFirst', '0.95'), ('OwnLagSumSecond', '0.91'), ('ReferenceDeltaMean', '0.354'),
                         ('ReferenceDeltaLow', '0.242'), ('ReferenceDeltaHigh', '0.467'), ('FaultyCloserCases', '53'),
                         ('ScorecardNoiseSD', '3'), ('ScorecardMaxSD', '19'), ('ScorecardNoiseScore', '84'),
                         ('ScorecardNoiseLoss', '16'), ('TrendARPhiHigh', '0.77')]:
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


def test_capability_score_and_band_letters_follow_the_scorecard():
    from experiments.render_results import band_letter, capability_score, score_cell
    from timegym.scorecard import verdict
    assert capability_score(1.0, 0.0) == 100.0
    assert capability_score(1.0, 1.0) == 50.0
    assert capability_score(1.0, 3.0) == 25.0
    assert capability_score(0.0, 0.0) == 100.0
    for score, letter in [(100, 'N'), (89.6, 'N'), (89.4, 'G'), (69.5, 'G'), (69.4, 'S'), (39.6, 'S'),
                          (39.4, 'F'), (0, 'F')]:
        assert band_letter(score) == letter == verdict(score)[0]
    assert score_cell(50.56) == '51 S'
    assert score_cell(36.91) == '37 F'


def test_capability_summary_scores_a_synthetic_record():
    protocol = {'capability_scenarios': ['white_noise'], 'capability_models': ['last_value'],
                'capability_sigma': [0.5, 2.0]}
    records_ = {'capability_records': [
        {'scenario': 'white_noise', 'model': 'last_value', 'sigma': sigma, 'seed': seed, 'mean_reference_mse': value}
        for sigma, values in [(0.5, [0.2, 0.3]), (2.0, [4.0, 4.0])] for seed, value in enumerate(values)]}
    rows = capability_summary(records_, protocol, unavoidable={('white_noise', 0.5): 0.25, ('white_noise', 2.0): 4.0})
    assert [row['sigma'] for row in rows] == [0.5, 2.0]
    assert rows[0]['unavoidable_mse'] == 0.25 and math.isclose(rows[0]['score'], 50.0)
    assert rows[1]['unavoidable_mse'] == 4.0 and math.isclose(rows[1]['score'], 50.0)
    assert list(rows[0])[-2:] == ['unavoidable_mse', 'score']


def test_capability_unavoidable_reads_the_reference_variance():
    from experiments.render_results import capability_unavoidable
    protocol = json.loads((ROOT / 'experiments/protocol.json').read_text())
    protocol.update(capability_scenarios=['white_noise', 'random_walk'], capability_sigma=[0.5])
    unavoidable = capability_unavoidable(protocol)
    assert math.isclose(unavoidable['white_noise', 0.5], 0.25)
    horizon = protocol['capability_task']['horizon']
    assert math.isclose(unavoidable['random_walk', 0.5], 0.25 * (horizon + 1) / 2)


def test_capability_unavoidable_rejects_seed_dependent_variance(monkeypatch):
    import numpy as np
    from experiments import render_results
    protocol = json.loads((ROOT / 'experiments/protocol.json').read_text())
    protocol.update(capability_scenarios=['white_noise'], capability_sigma=[1.0])
    original = render_results.Scenario.generate
    def generate(self, seed=None, **kwargs):
        case = original(self, seed=seed, **kwargs)
        case.reference.variance = np.asarray(case.reference.variance) * (1 + seed % 2)
        return case
    monkeypatch.setattr(render_results.Scenario, 'generate', generate)
    with pytest.raises(ValueError, match='depends on the seed'):
        render_results.capability_unavoidable(protocol)


def test_frozen_capability_table_scores():
    pilot, manifest = records('results')
    text = pilot_tex(pilot, manifest, capability_summary(pilot, manifest['protocol']))
    for line in [r'Noise around a fixed level & 51 S & 51 S & 55 S & 55 S & 99 N & 99 N \\',
                 r'Straight-line trend & 17 F & 50 S & 4 F & 47 S & 39 F & 18 F \\',
                 r'Repeating seasonal cycle & 5 F & 46 S & 55 S & 55 S & 9 F & 91 N \\',
                 r'Mean reversion (AR(1)) & 68 S & 68 S & 40 S & 40 S & 95 N & 95 N \\',
                 r'Random walk & 100 N & 100 N & 37 F & 37 F & 87 G & 87 G \\']:
        assert line in text
    assert r'at low ($\sigma=0.25$) and high ($\sigma=2$) innovation noise, over 16 cases each' in text
    assert 'scores 9 at low noise and 91 at high noise' in text
    assert 'no pass/fail threshold' not in text and 'general model ranking' not in text


@pytest.mark.parametrize('change', ['equal_levels', 'ar1_trend', 'seasonal', 'random_walk'])
def test_capability_conclusions_reject_contradictory_records(change):
    data, manifest = records('results')
    summary = capability_summary(data, manifest['protocol'])
    row = {(r['scenario'], r['model'], r['sigma']): r for r in summary}
    if change == 'equal_levels': row['ar1', 'last_value', 2.0]['score'] += 1
    if change == 'ar1_trend': row['trend', 'gaussian_ar1', 0.25]['score'] = 75
    if change == 'seasonal': row['seasonal', 'gaussian_ar1', 2.0]['score'] = 60
    if change == 'random_walk': row['random_walk', 'last_value', 1.0]['mean_reference_mse'] = 0.1
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
        assert close_statistics(paired_t_interval(delta), summary['clean_minus_faulty_mse'])
        assert sum(value < 0 for value in delta) == summary['cases_improved_by_repair']
    for summary in model['condition_summaries']:
        for branch, key in [('clean', 'clean'), (summary['fault'], 'faulty')]:
            alarms = [row['alarms'][summary['check']] for row in model['records']
                      if row['condition'] == summary['condition'] and row['branch'] == branch]
            assert close_statistics(wilson_interval(alarms), summary[key])
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


def test_followup_record_matches_its_manifest_and_frozen_inputs():
    followup, manifest = load_followup()
    protocol = json.loads(run_followup.PROTOCOL.read_text())
    run_followup.validate_protocol(protocol)
    assert manifest['protocol'] == protocol and followup['status'] == 'exploratory'
    assert '2026-11-13' in protocol['selection'] and 'after the protocol 2.0 results were known' in protocol['selection']
    for item in protocol['frozen_inputs'].values():
        assert manifest['frozen_inputs_sha256'][item['path']] == item['sha256'] == sha256(ROOT / item['path'])
    assert manifest['package_version'] and set(manifest['versions']) == {'numpy', 'pandas', 'scipy', 'scikit-learn'}
    assert 'experiments/run_followup.py' in manifest['source_files']


def test_followup_answer_key_alarm_recomputes_from_stored_distances_and_frozen_records():
    followup, manifest = load_followup()
    spec = manifest['protocol']['answer_key_alarm']
    alarm = followup['answer_key_alarm']
    assert alarm['largest_calibration_difference'] < spec['reproduction_tolerance']
    assert all(len(values) == 128 for values in alarm['calibration_distances'].values())
    supplement, _ = records('results-v2')
    rows = answer_key_counts(alarm['calibration_distances'], supplement['model']['records'], spec['alpha'],
                             alarm['conditions'], spec['branches'])
    stored = [{key: row[key] for key in rows[0]} for row in alarm['rows']]
    assert rows == stored
    frozen_loss = {row['condition']: row['loss_calibration_threshold']
                   for row in supplement['model']['condition_summaries'] if row['check'] == 'loss'}
    assert all(row['loss_threshold'] == frozen_loss[row['condition']] for row in alarm['rows'])
    totals = answer_key_totals(rows, spec['summary_faults'], spec['summary_excluded_conditions'])
    assert totals == alarm['totals']
    assert (totals['subtle_faults']['answer_key_alarms'], totals['subtle_faults']['loss_alarms'],
            totals['subtle_faults']['n']) == (148, 36, 640)
    assert (totals['clean']['answer_key_alarms'], totals['clean']['loss_alarms'], totals['clean']['n']) == (12, 13, 320)
    cells = {(row['condition'], row['branch']): (row['answer_key_alarms'], row['loss_alarms']) for row in rows}
    order = ['white_noise_locations', 'ar1_positive', 'ar1_negative', 'opposing_trends', 'seasonal_scales',
             'random_walk_drifts']
    assert [cells[c, 'reverse_prediction_lags'] for c in order] == [(5, 2), (42, 5), (23, 6), (0, 2), (64, 64), (2, 5)]
    assert [cells[c, 'stale_recursive_state'] for c in order] == [(16, 2), (28, 5), (24, 2), (8, 6), (64, 64), (0, 1)]


def test_followup_repair_scorecard_and_trend_summaries_agree_with_frozen_records():
    followup, _ = load_followup()
    supplement, _ = records('results-v2')
    repair = followup['repair_mechanism']
    by = {(row['branch'], row['seed']): row['mean_reference_mse'] for row in supplement['model']['records']
          if row['condition'] == repair['condition']}
    seeds = {seed for _, seed in by}
    assert sum(by[repair['fault'], seed] < by['clean', seed] for seed in seeds) == repair['faulty_closer_to_answer_key'] == 53
    summary = next(row for row in supplement['model']['loss_summaries']
                   if row['condition'] == repair['condition'] and row['fault'] == repair['fault'])
    assert repair['frozen_clean_minus_faulty_reference_mse'] == summary['clean_minus_faulty_reference_mse']
    assert [round(value, 2) for value in repair['own_lag_coefficient_sum']['mean']] == [0.95, 0.91]
    assert [round(value, 2) for value in repair['own_lag_coefficient_sum']['min']] == [0.78, 0.66]
    steps = repair['per_step_squared_distance']
    assert steps['clean'][0] == steps[repair['fault']][0]
    assert (round(steps['clean'][-1], 2), round(steps[repair['fault']][-1], 2)) == (0.95, 0.17)
    precision = followup['scorecard_precision']
    figure = json.loads((ROOT / 'paper/figures/paper-figure-provenance.json').read_text())['paper-scorecard']
    shown = {(cell['pattern'], cell['noise']): cell['score'] for cell in figure['cells']}
    assert len(precision['cells']) == len(shown) == 20
    for cell in precision['cells']:
        assert math.isclose(cell['score_first_10'], shown[cell['pattern'], cell['noise']], rel_tol=0, abs_tol=1e-9)
        assert cell['bootstrap_p2_5'] <= cell['bootstrap_p97_5']
    sds = [cell['bootstrap_sd'] for cell in precision['cells']]
    assert (precision['smallest_sd'], precision['largest_sd']) == (min(sds), max(sds))
    assert (round(min(sds)), round(max(sds))) == (3, 19)
    pilot, _ = records('results')
    trend = {row['sigma']: row for row in followup['trend_ar1']['rows']}
    assert set(trend) == {0.25, 2.0} and all(row['n'] == 16 for row in trend.values())
    assert round(trend[0.25]['mean_phi'], 3) == 0.995 and round(trend[2.0]['mean_phi'], 2) == 0.77
    assert round(trend[2.0]['mean_forecast_minus_answer_key'], 1) == -3.9
    check_followup_conclusions(followup)


def test_followup_rejects_changed_records_and_contradictions(tmp_path):
    followup, manifest = load_followup()
    path = tmp_path / 'results.json'
    path.write_text(json.dumps({**followup, 'unrecorded': True}))
    with pytest.raises(ValueError, match='fingerprint'):
        validate_followup(path, followup, manifest)
    changed = deepcopy(followup)
    changed['answer_key_alarm']['totals']['clean']['answer_key_alarms'] = 20
    with pytest.raises(ValueError, match='Recorded evidence'):
        check_followup_conclusions(changed)
    changed = deepcopy(followup)
    changed['trend_ar1']['rows'][0]['mean_phi'] = .9
    with pytest.raises(ValueError, match='Recorded evidence'):
        check_followup_conclusions(changed)


def test_followup_cli_protects_the_frozen_study_directories(monkeypatch):
    for folder in ('results', 'results-v2'):
        monkeypatch.setattr(sys, 'argv', ['followup', '--output', str(ROOT / 'experiments' / folder)])
        with pytest.raises(SystemExit):
            run_followup.main()
