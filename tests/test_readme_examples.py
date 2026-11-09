"""Run the README's Python examples in order, as a reader would paste them."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


def readme_blocks():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.M | re.S)


def test_readme_examples_run(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blocks = readme_blocks()
    assert blocks, "the README has no Python examples"
    monkeypatch.chdir(tmp_path)  # Examples that save files write here.
    namespace = {"__name__": "readme"}
    try:
        for number, block in enumerate(blocks, start=1):
            exec(compile(block, f"README.md Python block {number}", "exec"), namespace)
    finally:
        plt.close("all")
