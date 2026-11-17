"""Run the Python examples of the docs pages that show how to bring your own code and data."""
from pathlib import Path
import re

import numpy as np
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


def test_own_scores_example_on_the_scores_page_runs():
    namespace = {}
    exec(python_blocks("docs/api/diagnostics.md")[0], namespace)  # the later block needs your own data
    close = namespace["within_10_percent"]
    assert close(observed=np.array([10.0, 20.0]), predicted=np.array([10.5, 25.0])) == 0.5
    assert set(namespace["scores"]) == {"log_loss", "close"}


def test_comparison_examples_on_the_scores_page_print_what_the_page_shows():
    import contextlib
    import io

    text = (ROOT / "docs/api/diagnostics.md").read_text(encoding="utf-8")
    section = text[text.index("## Compare two forecasters"):text.index("### How the test works")]
    code = re.findall(r"^```python\n(.*?)^```", section, flags=re.M | re.S)
    shown = re.findall(r"^```text\n(.*?)^```", section, flags=re.M | re.S)
    namespace = {"__name__": "docs"}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        for block in code:
            exec(block, namespace)
    assert printed.getvalue() == "".join(shown)


def test_own_scores_on_the_extend_page_land_in_the_report():
    import matplotlib

    matplotlib.use("Agg")
    namespace = {"__name__": "docs"}
    for block in python_blocks("docs/concepts/extend.md"):
        exec(block, namespace)
        if "rainy" in namespace and "scored" not in namespace:
            namespace["scored"] = namespace["report"]
    scored = namespace["scored"]
    assert all({"brier", "log_loss"} <= set(case.metrics) for case in scored.cases)
    assert scored.metadata["scores"][0]["name"] == "brier"


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
