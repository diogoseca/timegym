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
