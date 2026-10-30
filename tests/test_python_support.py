"""Exercise release discovery and a coordinated update in a disposable tree."""
from datetime import date
import json
import re
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


def test_update_keeps_notebook_ids_and_independent_version_counters(tmp_path):
    files = ['pyproject.toml', 'Dockerfile', 'README.md', 'docs/release.md',
             'notebooks/12_package.ipynb', 'CITATION.cff', 'CHANGELOG.md']
    files += [str(p.relative_to(ROOT)) for p in (ROOT / '.github/workflows').glob('*.yml')]
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    notebook = tmp_path / 'notebooks/12_package.ipynb'
    data = json.loads(notebook.read_text())
    ids = [c['id'] for c in data['cells']]
    for cell in data['cells']:
        cell['source'] = [re.sub(r'__version__ = "[0-9.]+"', '__version__ = "1.2.7"', line) for line in cell['source']]
    notebook.write_text(json.dumps(data))
    citation = tmp_path / 'CITATION.cff'
    citation.write_text(citation.read_text() + 'doi: 10.5281/zenodo.1234567\n')
    versions = [f'3.{i}' for i in range(12, 17)]
    assert update(tmp_path, versions, released=date(2027, 10, 10))
    data = json.loads(notebook.read_text())
    assert ids == [c['id'] for c in data['cells']]
    assert any('__version__ = "1.2.8"' in ''.join(c['source']) for c in data['cells'])
    project = tomllib.loads((tmp_path / 'pyproject.toml').read_text())['project']
    assert project['requires-python'] == '>=3.12,<3.17'
    assert 'FROM python:3.16-slim' in (tmp_path / 'Dockerfile').read_text()
    assert "python-version: ['3.12', '3.13', '3.14', '3.15', '3.16']" in (tmp_path / '.github/workflows/checks.yml').read_text()
    assert 'Python 3.12–3.16' in (tmp_path / 'docs/release.md').read_text()
    assert '\ndoi:' not in citation.read_text()
    assert '    doi: "10.48550/arXiv.2105.01404"' in citation.read_text()
    assert '## 1.2.8 — 2027-10-10' in (tmp_path / 'CHANGELOG.md').read_text()
    snapshots = {name: (tmp_path / name).read_bytes() for name in files}
    assert not update(tmp_path, versions, released=date(2027, 10, 11))
    assert snapshots == {name: (tmp_path / name).read_bytes() for name in files}
