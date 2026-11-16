"""Run the Python examples of the docs pages that show how to bring your own code and data."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Pages whose examples run in order in one namespace, as a reader pasting them would.
GROUPS = {
    "extend": ["docs/concepts/extend.md"],
    "irregular readings": ["docs/api/data-preparation.md"],
    "own data": ["docs/api/tasks-and-forecasts.md", "docs/api/pipeline.md"],
}


def python_blocks(page):
    text = (ROOT / page).read_text(encoding="utf-8")
    return re.findall(r"^```python\n(.*?)^```", text, flags=re.M | re.S)


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_docs_examples_run(group):
    pytest.importorskip("sklearn")
    import matplotlib

    matplotlib.use("Agg")
    namespace = {"__name__": "docs"}
    ran = 0
    for page in GROUPS[group]:
        for number, block in enumerate(python_blocks(page), start=1):
            exec(compile(block, f"{page} Python block {number}", "exec"), namespace)
            ran += 1
    assert ran, f"no Python examples found for {group}"


def test_extend_examples_show_what_they_say():
    import matplotlib

    matplotlib.use("Agg")
    namespace = {"__name__": "docs"}
    for block in python_blocks("docs/concepts/extend.md"):
        exec(block, namespace)
    report = namespace["report"]  # the last example: a kind of series with its answer key
    assert [(case.scenario, case.seed) for case in report.cases] == [("noisy_line", 0), ("noisy_line", 1), ("noisy_line", 2)]
    assert all("mean_reference_mse" in case.metrics for case in report.cases)
    assert namespace["run"].metadata["model"] == "weather service"


def test_irregular_readings_recipe_never_looks_ahead():
    namespace = {}
    (block,) = python_blocks("docs/api/data-preparation.md")
    exec(block, namespace)
    hourly, readings = namespace["hourly"], namespace["readings"]
    assert hourly.loc["2024-05-01 11:00", "level"] == 2.0  # the 10:50 reading, seen at 11:00
    assert hourly.loc["2024-05-01 10:00", "level"] == 1.0  # not the 10:50 reading
    assert hourly["level"].isna().sum() == 1  # 19:00: the last reading is too old
    resampled = readings.resample("1h").last()
    assert resampled.loc["2024-05-01 10:00"] == 2.0  # what the page warns about
