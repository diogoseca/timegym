"""Keep package, citation and manuscript identities synchronized."""
from pathlib import Path
import shutil

import pytest

pytest.importorskip('yaml')
from timegym import __version__
from tools.sync_citation import ROOT, read_metadata, render, synchronize


def test_citation_changelog_and_generated_readme_match_package():
    assert read_metadata()['version'] == __version__
    assert synchronize(check=True, tag='v' + __version__)


def test_citation_check_detects_drift_without_rewriting(tmp_path):
    (tmp_path / 'timegym').mkdir()
    for name in ('README.md', 'CHANGELOG.md', 'CITATION.cff', 'timegym/__init__.py'):
        shutil.copyfile(ROOT / name, tmp_path / name)
    readme = tmp_path / 'README.md'
    original = readme.read_text().replace('[Computer software]', '[Stale software]')
    readme.write_text(original)
    assert not synchronize(tmp_path, check=True)
    assert readme.read_text() == original
    synchronize(tmp_path)
    assert synchronize(tmp_path, check=True)
    with pytest.raises(ValueError, match='Release tag'):
        synchronize(tmp_path, check=True, tag='v999.0.0')
    citation = tmp_path / 'CITATION.cff'
    citation.write_text(citation.read_text().replace(f'version: {__version__}', 'version: 999.0.0'))
    with pytest.raises(ValueError, match='version differs'):
        read_metadata(tmp_path)


def test_citation_uses_registered_doi_when_present():
    metadata = read_metadata()
    metadata['doi'] = '10.5281/zenodo.1234567'
    citation = render(metadata)
    assert 'doi = {10.5281/zenodo.1234567}' in citation
    assert '[Computer software]. https://doi.org/10.5281/zenodo.1234567' in citation
