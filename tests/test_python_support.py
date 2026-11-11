"""Exercise release discovery and a coordinated update in a disposable tree."""
from pathlib import Path
import shutil
import tomllib

from tools.update_python_support import ROOT, supported_versions, update


def pandas_release(*, minor=15, wheels=True):
    return {'info': {'requires_python': '>=3.11', 'classifiers': [
        f'Programming Language :: Python :: 3.{i}' for i in range(11, minor + 1)
    ]}, 'urls': [{'filename': f'pandas-3.1.0-cp3{minor}-cp3{minor}-manylinux.whl'}] if wheels else []}


def test_discovery_requires_final_python_and_pandas_wheel_support():
    html = '<a>Python 3.15.2</a><a>Python 3.16.0rc1</a><a>Python 3.9.25</a>'
    assert supported_versions(html, pandas_release()) == [f'3.{i}' for i in range(11, 16)]
    assert supported_versions(html, pandas_release(minor=14)) == []
    assert supported_versions(html, pandas_release(wheels=False)) == []


def test_update_rewrites_support_and_leaves_versioning_to_the_merge_workflow(tmp_path):
    files = ['pyproject.toml', 'Dockerfile', 'README.md', 'docs/release.md',
             'source/12_package.ipynb', 'CITATION.cff', 'CHANGELOG.md']
    files += [str(p.relative_to(ROOT)) for p in (ROOT / '.github/workflows').glob('*.yml')]
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    versioned = {name: (tmp_path / name).read_bytes()
                 for name in ('source/12_package.ipynb', 'CITATION.cff', 'CHANGELOG.md')}
    versions = [f'3.{i}' for i in range(12, 17)]
    assert update(tmp_path, versions)
    project = tomllib.loads((tmp_path / 'pyproject.toml').read_text())['project']
    assert project['requires-python'] == '>=3.12,<3.17'
    assert 'FROM python:3.16-slim' in (tmp_path / 'Dockerfile').read_text()
    assert "python-version: ['3.12', '3.13', '3.14', '3.15', '3.16']" in (tmp_path / '.github/workflows/checks.yml').read_text()
    assert "python-version: '3.16' # current-python" in (tmp_path / '.github/workflows/tag-release.yml').read_text()
    assert 'Python 3.12–3.16' in (tmp_path / 'docs/release.md').read_text()
    assert versioned == {name: (tmp_path / name).read_bytes() for name in versioned}
    snapshots = {name: (tmp_path / name).read_bytes() for name in files}
    assert not update(tmp_path, versions)
    assert snapshots == {name: (tmp_path / name).read_bytes() for name in files}
