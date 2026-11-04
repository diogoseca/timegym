"""Check real Markdown links without treating code examples as documents."""
from pathlib import Path

import pytest

pytest.importorskip('mkdocs')
from mkdocs.exceptions import PluginError
from tools.docs_hooks import local_link_targets, on_config


def test_links_exclude_code_and_separate_titles():
    source = '''
```python
[example](missing.py)
```
~~~markdown
[example](also-missing.md)
~~~
`[inline](missing.md)` and ``[inline ` code](missing.md)``
[actual](guide.md "A guide") and [spaces](<some file.md> 'A title')
![image](plot.png "Figure")
[reference][guide]

[guide]: other.md "Other guide"
'''
    assert local_link_targets(source) == ['guide.md', 'some file.md', 'plot.png', 'other.md']


class Config(dict):
    def __init__(self, root):
        super().__init__()
        self.config_file_path = root / 'mkdocs.yml'


def test_staging_preserves_hard_failure_and_accepts_titled_links(tmp_path):
    (tmp_path / 'README.md').write_text('[guide](guide.md "Guide")\n`[fake](missing.md)`')
    (tmp_path / 'guide.md').write_text('A real guide.')
    config = on_config(Config(tmp_path))
    assert (Path(config['docs_dir']) / 'guide.md').read_text() == 'A real guide.'
    (tmp_path / 'guide.md').unlink()
    with pytest.raises(PluginError, match='Missing linked documentation file'):
        on_config(Config(tmp_path))


def test_staging_rejects_links_outside_checkout(tmp_path):
    (tmp_path / 'README.md').write_text('[outside](../outside.md)')
    with pytest.raises(PluginError, match='leaves the source tree'):
        on_config(Config(tmp_path))


def test_notebook_links_point_at_built_pages_or_github():
    from types import SimpleNamespace
    from tools.docs_hooks import notebook_link

    def page(url):
        return SimpleNamespace(url=url, is_documentation_page=lambda: True)

    built = {
        'notebooks/20_univariate_uncertainty.ipynb': page('notebooks/20_univariate_uncertainty/'),
        'notebooks/README.md': page('notebooks/'),
        'docs/stress-cases.md': page('docs/stress-cases/'),
        'notebooks/07_pipeline.ipynb': SimpleNamespace(url='notebooks/07_pipeline.ipynb',
                                                       is_documentation_page=lambda: False),
    }
    files = SimpleNamespace(get_file_from_path=built.get)
    repo = 'https://github.com/diogoseca/timegym'

    def link(href):
        return notebook_link(href, 'notebooks/00_quick_start.ipynb', 'notebooks/00_quick_start/', files, repo)

    assert link('20_univariate_uncertainty.ipynb') == '../20_univariate_uncertainty/'
    assert link('20_univariate_uncertainty.ipynb#scores') == '../20_univariate_uncertainty/#scores'
    assert link('README.md') == '../'
    assert link('../docs/stress-cases.md') == '../../docs/stress-cases/'
    assert link('07_pipeline.ipynb') == f'{repo}/blob/main/notebooks/07_pipeline.ipynb'
    assert link('archive/3.%20Semi-Synthetic%20Data.ipynb') == (
        f'{repo}/blob/main/notebooks/archive/3.%20Semi-Synthetic%20Data.ipynb')
    for unchanged in ('https://doi.org/10.1/x', '#section', '/absolute/', '../../outside.md'):
        assert link(unchanged) == unchanged


def test_site_link_check_reports_missing_targets(tmp_path):
    from tools.check_site_links import broken_links

    (tmp_path / 'page').mkdir()
    (tmp_path / 'page' / 'index.html').write_text(
        '<a href="../other/">ok</a><a href="missing.ipynb">bad</a>'
        '<img src="../img.png"><a href="https://example.org/x">external</a><a href="#top">top</a>')
    (tmp_path / 'other').mkdir()
    (tmp_path / 'other' / 'index.html').write_text('<a href="../page/">back</a>')
    assert broken_links(tmp_path) == [('page/index.html', '../img.png'), ('page/index.html', 'missing.ipynb')]
    (tmp_path / 'img.png').write_bytes(b'')
    assert broken_links(tmp_path) == [('page/index.html', 'missing.ipynb')]
