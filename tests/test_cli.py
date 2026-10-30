"""Exercise user-visible command outcomes and persisted diagnostic evidence."""
import json
import subprocess
import sys

import pytest

from timegym import TestReport
from timegym.cli import main


def test_demo_inspection_and_replay_form_a_complete_workflow(tmp_path, capsys):
    assert main(['demo', '--output', str(tmp_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['expected_behavior_verified']
    assert summary['label_blind_mse']['faulty'] == summary['label_blind_mse']['repaired']
    assert 'index_mismatch' in summary['fault_codes']
    faulty = TestReport.read(tmp_path / 'faulty.json')
    repaired = TestReport.read(tmp_path / 'repaired.json')
    assert faulty.summary['failed_cases'] == 1
    assert repaired.summary['failed_cases'] == 0
    assert main(['inspect', str(tmp_path / 'faulty.json')]) == 1
    assert main(['inspect', str(tmp_path / 'repaired.json')]) == 0
    assert main(['replay', str(tmp_path / 'repaired.json'), '--require-same-environment']) == 0
    capsys.readouterr()
    assert main(['replay', str(tmp_path / 'faulty.json')]) == 2
    assert 'not automatically replayable' in capsys.readouterr().err


def test_replay_environment_gate_is_distinct_from_numerical_reproduction(tmp_path, capsys):
    main(['demo', '--output', str(tmp_path)])
    report = TestReport.read(tmp_path / 'repaired.json')
    report.metadata['source']['digest'] = 'different-source'
    report.to_json(tmp_path / 'other-environment.json')
    assert main(['replay', str(tmp_path / 'other-environment.json')]) == 0
    assert main(['replay', str(tmp_path / 'other-environment.json'), '--require-same-environment']) == 1


@pytest.mark.parametrize('text', ['{}', '{"schema_version":"2.0","schema_version":"2.0"}', 'not json'])
def test_inspect_rejects_invalid_evidence(tmp_path, capsys, text):
    path = tmp_path / 'report.json'
    path.write_text(text)
    assert main(['inspect', str(path)]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err.startswith('timegym: ')


def test_demo_refuses_to_replace_artifacts(tmp_path, capsys):
    path = tmp_path / 'faulty.json'
    path.write_text('preserve this evidence')
    assert main(['demo', '--output', str(tmp_path)]) == 2
    assert path.read_text() == 'preserve this evidence'
    assert not (tmp_path / 'repaired.json').exists()


def test_replay_rejects_unrepresentable_parameters_without_a_traceback(tmp_path, capsys):
    main(['demo', '--output', str(tmp_path)])
    report = TestReport.read(tmp_path / 'repaired.json')
    report.cases[0].manifest['parameters']['sigma'] = 10**400
    report.to_json(tmp_path / 'overflow.json')
    assert main(['replay', str(tmp_path / 'overflow.json')]) == 2
    assert 'timegym:' in capsys.readouterr().err


def test_invalid_demo_arguments_do_not_create_outputs(tmp_path, capsys):
    path = tmp_path / 'absent'
    assert main(['demo', '--seed', '-1', '--output', str(path)]) == 2
    assert not path.exists()
    assert main(['demo', '--plot']) == 2


def test_demo_plot_has_a_real_png_and_closes_its_figure(tmp_path, capsys):
    import matplotlib.pyplot as plt
    original = plt.get_fignums()
    assert main(['demo', '--output', str(tmp_path), '--plot']) == 0
    assert (tmp_path / 'alignment.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert plt.get_fignums() == original


def test_module_entrypoint_runs_without_notebook_or_plotting_imports():
    script = '''
import importlib.abc, runpy, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'matplotlib', 'sklearn', 'nbdev', 'IPython'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
sys.argv = ['timegym', 'demo']
runpy.run_module('timegym', run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['expected_behavior_verified']
