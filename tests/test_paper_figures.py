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


@pytest.mark.parametrize('source', [r'\input{generated-macros.tex}',
                                   r'\IfFileExists{generated-macros.tex}{\input{generated-macros.tex}}{}',
                                   r'\IfFileExists{generated-macros.tex}{\input{generated-macros.tex}}{\textbf{Missing}}'])
def test_standalone_inlines_caption_macros_and_removes_external_guard(tmp_path, source):
    from build_standalone import inline_macros
    macros = r'\newcommand{\RepairImproved}{21}'
    (tmp_path / 'generated-macros.tex').write_text(macros)
    assert inline_macros(source, tmp_path) == macros
