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
        'tutorials/08_univariate_uncertainty.ipynb': page('tutorials/08_univariate_uncertainty/'),
        'tutorials/README.md': page('tutorials/'),
        'docs/stress-cases.md': page('docs/stress-cases/'),
        'source/07_pipeline.ipynb': SimpleNamespace(url='source/07_pipeline.ipynb',
                                                    is_documentation_page=lambda: False),
    }
    files = SimpleNamespace(get_file_from_path=built.get)
    repo = 'https://github.com/diogoseca/timegym'

    def link(href):
        return notebook_link(href, 'tutorials/01_quick_start.ipynb', 'tutorials/01_quick_start/', files, repo)

    assert link('08_univariate_uncertainty.ipynb') == '../08_univariate_uncertainty/'
    assert link('08_univariate_uncertainty.ipynb#scores') == '../08_univariate_uncertainty/#scores'
    assert link('README.md') == '../'
    assert link('../docs/stress-cases.md') == '../../docs/stress-cases/'
    assert link('../source/07_pipeline.ipynb') == f'{repo}/blob/main/source/07_pipeline.ipynb'
    assert link('../source/drafts/3.%20Semi-Synthetic%20Data.ipynb') == (
        f'{repo}/blob/main/source/drafts/3.%20Semi-Synthetic%20Data.ipynb')
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


def test_moved_pages_redirect_to_built_pages(tmp_path):
    from tools.docs_hooks import MOVED_PAGES, on_post_build

    for new in MOVED_PAGES.values():
        (tmp_path / new).mkdir(parents=True, exist_ok=True)
        (tmp_path / new / 'index.html').write_text('<html></html>', encoding='utf-8')
    on_post_build({'site_dir': str(tmp_path), 'site_url': 'https://diogoseca.github.io/timegym/'})
    page = (tmp_path / 'notebooks/00_quick_start/index.html').read_text(encoding='utf-8')
    assert 'url=../../tutorials/01_quick_start/' in page
    assert 'href="https://diogoseca.github.io/timegym/tutorials/01_quick_start/"' in page
    assert 'url=../tutorials/' in (tmp_path / 'docs/notebooks/index.html').read_text(encoding='utf-8')


def test_raw_github_images_are_served_from_the_site():
    from types import SimpleNamespace
    from tools.docs_hooks import on_page_markdown

    config = {'repo_url': 'https://github.com/diogoseca/timegym'}
    raw = 'https://raw.githubusercontent.com/diogoseca/timegym/main/docs/images/scorecard.png'
    home = SimpleNamespace(file=SimpleNamespace(src_uri='README.md'))
    deeper = SimpleNamespace(file=SimpleNamespace(src_uri='docs/tutorials.md'))
    assert on_page_markdown(f'[![card]({raw})]({raw})', home, config, None) == (
        '[![card](docs/images/scorecard.png)](docs/images/scorecard.png)')
    assert on_page_markdown(f'![card]({raw})', deeper, config, None) == '![card](images/scorecard.png)'
