"""Check the major.features.fixes counters and the release metadata they write."""
from datetime import date
import json
import shutil

import pytest

pytest.importorskip('yaml')
from tools.release_version import ROOT, apply, next_version, package_version, release_notes
from tools.sync_citation import read_metadata


def pr(number, *names, title='Change'):
    return {'number': number, 'title': title, 'labels': [{'name': name} for name in names]}


def test_counters_add_merged_features_and_fixes_without_resetting():
    assert next_version('1.2.7', [pr(1, 'feature')]) == '1.3.7'
    assert next_version('1.3.7', [pr(2, 'fix')]) == '1.3.8'
    merged = [pr(3, 'feature'), pr(4, 'fix'), pr(5, 'feature', 'documentation'), pr(6, 'fix', 'feature')]
    assert next_version('1.3.8', merged) == '1.6.10'


def test_unlabelled_merges_do_not_release_and_major_restarts_counters():
    assert next_version('1.3.8', [pr(1), pr(2, 'documentation')]) is None
    assert next_version('1.3.8', []) is None
    assert next_version('1.3.8', [pr(1, 'feature')], major=True) == '2.0.0'
    with pytest.raises(ValueError, match='three integer counters'):
        next_version('1.3', [pr(1, 'fix')])


def test_release_notes_prefer_unreleased_text_and_otherwise_list_titles():
    changelog = '# Changelog\n\n## Unreleased\n\n- Hand-written note.\n\n## 1.1.0 — 2026-11-05\n\n- Old.\n'
    rest, notes = release_notes(changelog, [pr(9, 'fix', title='Ignored title')])
    assert notes == '- Hand-written note.'
    assert '## Unreleased' not in rest and '## 1.1.0 — 2026-11-05' in rest
    rest, notes = release_notes('# Changelog\n\n## 1.1.0 — 2026-11-05\n', [
        pr(7, 'feature', title='Add seasonal references'), pr(8, title='Fix a docs typo'),
        pr(9, 'fix', title='Keep index labels in evidence')])
    assert notes == '- Add seasonal references (#7)\n- Keep index labels in evidence (#9)'


def test_apply_updates_notebook_citation_and_changelog_consistently(tmp_path):
    for name in ('notebooks/12_package.ipynb', 'CITATION.cff', 'CHANGELOG.md'):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, tmp_path / name)
    # Pending notes under "## Unreleased" would replace the PR title checked below.
    unreleased = tmp_path / 'CHANGELOG.md'
    unreleased.write_text(release_notes(unreleased.read_text(), [])[0])
    ids =[cell['id'] for cell in json.loads((tmp_path / 'notebooks/12_package.ipynb').read_text())['cells']]
    citation = tmp_path / 'CITATION.cff'
    citation.write_text(citation.read_text() + 'doi: 10.5281/zenodo.1234567\n')
    current = package_version(tmp_path)
    version = next_version(current, [pr(10, 'feature', title='Add a feature')])
    apply(tmp_path, version, date(2027, 1, 2), [pr(10, 'feature', title='Add a feature')])
    assert package_version(tmp_path) == version
    assert ids == [cell['id'] for cell in json.loads((tmp_path / 'notebooks/12_package.ipynb').read_text())['cells']]
    changelog = (tmp_path / 'CHANGELOG.md').read_text()
    assert changelog.startswith(f'# Changelog\n\n## {version} — 2027-01-02\n\n- Add a feature (#10)\n\n## {current} — ')
    assert '\ndoi:' not in citation.read_text()
    assert '    doi: "10.48550/arXiv.2105.01404"' in citation.read_text()
    (tmp_path / 'timegym').mkdir()
    (tmp_path / 'timegym/__init__.py').write_text(f'__version__ = "{version}"\n')
    metadata = read_metadata(tmp_path)
    assert (str(metadata['version']), str(metadata['date-released'])) == (version, '2027-01-02')
