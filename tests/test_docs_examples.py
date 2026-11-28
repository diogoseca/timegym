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
def test_docs_examples_run(group, tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    monkeypatch.chdir(tmp_path)  # examples that save a report write it here
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
    result = namespace["result"]  # the last example: a kind of series with its answer key
    report = result.report
    assert [(case.scenario, case.seed) for case in report.cases] == [("noisy_line", 0), ("noisy_line", 1), ("noisy_line", 2)]
    assert all("mean_reference_mse" in case.metrics for case in report.cases)
    assert namespace["run"].metadata["model"] == "weather service"
    assert result.scorecard.table["pattern"].tolist() == ["noisy_line"]


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
            namespace["scored"] = namespace["result"]
    scored = namespace["scored"]
    assert all({"brier", "log_loss"} <= set(case.metrics) for case in scored.report.cases)
    assert scored.metadata["scores"][0]["name"] == "brier"


def test_own_checks_examples_on_the_scores_page_run():
    import pandas as pd
    from timegym import LastValueForecaster, NaiveForecaster, Pipeline, cases_from_series

    text = (ROOT / "docs/api/diagnostics.md").read_text(encoding="utf-8")
    section = text[text.index("## Your own output checks"):text.index("## Statistical and metamorphic")]
    blocks = re.findall(r"^```python\n(.*?)^```", section, flags=re.M | re.S)
    days = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    rain = pd.Series((rng.random(120) < 0.3).astype(float), index=days, name="rain")
    namespace = {"pipeline": Pipeline(NaiveForecaster("mean")),
                 "cases": cases_from_series(rain, origins=[60, 80, 100], horizon=7, name="rain")}
    exec(blocks[0], namespace)  # a chance of rain stays between 0 and 1
    assert namespace["result"].checks[-1] == {"check": "within", "title": "Values between 0 and 1",
                                              "passed": 3, "failed": 0}
    exec(blocks[1], namespace)
    north, south = rng.poisson(3, 120), rng.poisson(2, 120)
    sales = pd.DataFrame({"north": north, "south": south, "total": north + south}, index=days).astype(float)
    namespace.update(pipeline=Pipeline(LastValueForecaster()),
                     cases=cases_from_series(sales, origins=[60, 80, 100], horizon=7, name="coats"))
    exec(blocks[2], namespace)
    assert namespace["result"].checks[-2:] == [
        {"check": "within", "title": "Values at least 0", "passed": 3, "failed": 0},
        {"check": "totals_add_up", "title": "Store totals add up", "passed": 3, "failed": 0},
    ]
    assert len(blocks) == 3


def test_own_checks_on_the_extend_page_fail_and_pass_as_the_comment_says():
    import matplotlib

    matplotlib.use("Agg")
    namespace = {"__name__": "docs"}
    for block in python_blocks("docs/concepts/extend.md"):
        exec(block, namespace)
    within_row, record_row = namespace["checked"].checks[-2:]
    assert within_row == {"check": "within", "title": "Values at least 0", "passed": 0, "failed": 3}
    assert record_row == {"check": "below_record", "title": "Point forecast no wetter than the wettest day so far",
                          "passed": 3, "failed": 0}
    messages = [d.message for case in namespace["checked"].report.cases for d in case.diagnostics if d.name == "check_within"]
    assert all("samples:" in message for message in messages)  # sample paths dip below 0 mm, as the comment says


def test_own_kind_of_series_on_the_scorecard_page_scores_as_the_table_says():
    from timegym import NaiveForecaster, Pipeline

    text = (ROOT / "docs/api/scorecard.md").read_text(encoding="utf-8")
    section = text[text.index("## Your own kind of series"):text.index("## Mistakes and checks")]
    (block,) = re.findall(r"^```python\n(.*?)^```", section, flags=re.M | re.S)
    namespace = {"__name__": "docs"}
    exec(block, namespace)
    forecasters = {
        "Seasonal mean": namespace["card"],
        "Historical mean": Pipeline(NaiveForecaster("mean")).scorecard(patterns=[namespace["rain"]]),
        "Last cycle repeated": Pipeline(NaiveForecaster("seasonal", period=12)).scorecard(patterns=[namespace["rain"]]),
    }
    for label, card in forecasters.items():
        (row,) = [line for line in section.splitlines() if line.startswith(f"| {label}")]
        cells = [cell.strip() for cell in row.strip("|").split("|")][1:]
        printed = [line for line in card.to_markdown().splitlines() if line.startswith("| Rain or no rain |")][0]
        assert [cell.strip() for cell in printed.strip("|").split("|")][1:3] == cells
    rain = namespace["card"].table[namespace["card"].table["pattern"] == "rain"]
    assert (rain["attainable"].round() == 91).all()  # "The attainable score is 91 in both columns"
    assert namespace["simulate_rain"].__doc__  # the function the page explains
    for noise, low, high in [(0.5, 0.20, 0.80), (2.0, 0.35, 0.65)]:  # "from 20% to 80% ... from 35% to 65%"
        parts = namespace["simulate_rain"](np.random.default_rng(0), 120, 24, noise)
        assert np.isclose(parts["mean"].min(), low) and np.isclose(parts["mean"].max(), high)


def test_calibration_example_on_the_scores_page_prints_what_the_page_shows():
    import contextlib
    import io

    from timegym import NaiveForecaster, Pipeline
    from timegym.diagnostics import calibration

    text = (ROOT / "docs/api/diagnostics.md").read_text(encoding="utf-8")
    section = text[text.index("## Are the chances honest?"):text.index("## Case-level uncertainty intervals")]
    (code,) = re.findall(r"^```python\n(.*?)^```", section, flags=re.M | re.S)
    (shown,) = re.findall(r"^```text\n(.*?)^```", section, flags=re.M | re.S)
    namespace = {"__name__": "docs"}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(code, namespace)
    assert printed.getvalue() == shown
    cases = namespace["cases"]

    class WithoutAddedSteps(namespace["ShareAtThisPointOfTheCycle"]):
        def fit(self, context):
            values = context.to_numpy()[:, 0]
            self.length_ = len(values)
            self.share_ = np.array([values[phase::12].mean() for phase in range(12)])
            return self

    plain = calibration(Pipeline(WithoutAddedSteps()).run(cases=cases, keep_data=True), cluster="station")
    assert plain.certain == 130 and f"{plain.slope:.2f}" == "0.46"  # "says 0 or 1 exactly 130 times"
    mean = calibration(Pipeline(NaiveForecaster("mean")).run(cases=cases, keep_data=True), cluster="station")
    used = mean.table[mean.table["forecasts"] > 0]
    assert used["low"].min() == 0.4 and used["high"].max() == 0.6  # "between 40% and 60%"
    assert f"{mean.slope:.2f}" == "-0.14"


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
