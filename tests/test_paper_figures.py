import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'paper'))
from figure_common import notebook_literal
from vector_canvas import Canvas


def notebook(tmp_path, source, **cell):
    path = tmp_path / 'spec.ipynb'
    path.write_text(json.dumps({'cells': [{'id': 'spec', 'cell_type': 'code', 'source': source, **cell}]}))
    return path


def test_notebook_literals_are_data_and_never_execute(tmp_path):
    path = notebook(tmp_path, 'specs = [{"kind": "ar1"}]\nraise RuntimeError("must not execute")')
    assert notebook_literal(path, 'spec', 'specs', expect_len=1) == [{'kind': 'ar1'}]
    with pytest.raises(ValueError, match='Expected 2 entries'):
        notebook_literal(path, 'spec', 'specs', expect_len=2)


@pytest.mark.parametrize('source', ['specs = make_specs()', 'specs = [', 'specs = []\nspecs = []', 'other = []'])
def test_notebook_literal_failures_identify_source(tmp_path, source):
    path = notebook(tmp_path, source)
    with pytest.raises(ValueError, match='spec.ipynb:spec:specs'):
        notebook_literal(path, 'spec', 'specs')


def test_notebook_literal_rejects_missing_and_noncode_cells(tmp_path):
    path = notebook(tmp_path, 'specs = []', cell_type='markdown')
    with pytest.raises(ValueError, match='code cell'):
        notebook_literal(path, 'spec', 'specs')


def test_markers_group_only_consecutive_equal_styles_and_keep_coordinates(tmp_path):
    canvas = Canvas(100, 100)
    canvas.circle(10, 20, r=1)
    canvas.circle(30, 40, r=1)
    canvas.line(0, 50, 100, 50, width=.5, dash=True)
    canvas.circle(50, 60, r=1)
    canvas.circle(70, 80, r=2)
    assert [len(group) for group in canvas._groups()] == [2, 1, 1, 1]
    tikz = canvas.tikz()
    assert tikz.count('\\path[fill=') == 3
    assert '(10.000,20.000) circle[radius=1bp] (30.000,40.000) circle[radius=1bp]' in tikz
    assert 'dash pattern=on 2bp off 2bp' in tikz
    canvas.render(tmp_path, 'geometry')
    svg = (tmp_path / 'geometry.svg').read_text()
    assert 'stroke-dasharray: 2,2' in svg
    tree = ET.fromstring(svg)
    ns = {'s': 'http://www.w3.org/2000/svg'}
    assert len(tree.findall('.//s:use', ns)) == 2
    assert svg.count('PathCollection_') == 3


def test_partial_render_replaces_only_named_entries(tmp_path):
    import hashlib
    import render_figures

    for name in render_figures.SHARED_INPUTS + ['paper/figure_a.py', 'paper/figure_b.py']:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name)
    def builder(label):
        def build(root):
            canvas = Canvas(40, 20)
            canvas.text(2, 10, label, size=6)
            return canvas, {'label': label}
        return build
    figures = {'figure-a': (builder('a'), ['paper/figure_a.py']),
               'figure-b': (builder('b'), ['paper/figure_b.py'])}
    provenance = tmp_path / 'paper/figures/paper-figure-provenance.json'
    provenance.parent.mkdir(parents=True)
    old = {'inputs_sha256': {'kept': 'as is'}, 'figure-b': {'label': 'old b'},
           'figure-a': {'label': 'old a'}, 'retired-figure': {'label': 'gone'}}
    provenance.write_text(json.dumps(old))
    render_figures.render(tmp_path, figures, only=['figure-a'])
    saved = json.loads(provenance.read_text())
    assert saved['inputs_sha256'] == {'kept': 'as is'}
    assert saved['figure-b'] == {'label': 'old b'}
    assert 'retired-figure' not in saved
    assert saved['figure-a']['label'] == 'a'
    expected = {name: hashlib.sha256(name.encode()).hexdigest()
                for name in ['paper/figure_a.py', *render_figures.SHARED_INPUTS]}
    assert saved['figure-a']['inputs_sha256'] == expected
    assert (tmp_path / 'paper/figures/figure-a.tikz').exists()
    assert not (tmp_path / 'paper/figures/figure-b.tikz').exists()
    with pytest.raises(SystemExit):
        render_figures.render(tmp_path, figures, only=['missing'])


def test_full_render_writes_every_entry_and_its_inputs(tmp_path, monkeypatch):
    import render_figures

    for name in render_figures.SHARED_INPUTS + ['paper/figure_a.py', 'top.txt']:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(render_figures, 'ALL_INPUTS', ['top.txt'])
    def build(root):
        canvas = Canvas(40, 20)
        canvas.text(2, 10, 'a', size=6)
        return canvas, {}
    saved = render_figures.render(tmp_path, {'figure-a': (build, ['paper/figure_a.py'])})
    assert set(saved) == {'figure-a', 'inputs_sha256'}
    assert set(saved['inputs_sha256']) == {'top.txt'}
    assert 'paper/figure_a.py' in saved['figure-a']['inputs_sha256']
