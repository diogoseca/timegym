"""The Study 3 part of the manuscript is generated from the stored records (experiments/render_study3.py).

Nothing is run again here: the tests read the records under experiments/results-v3, compare the generated text and
macros with the committed ones, and check that a changed record or an unsupported statement is refused.
"""
import copy
import json
import re

import pytest

from experiments import render_study3 as render
from experiments.common import sha256

ROOT = render.ROOT


@pytest.fixture(scope="module")
def records():
    return render.load_study3()


@pytest.fixture(scope="module")
def summary(records):
    return render.summarize(records)


def test_the_generated_section_is_the_committed_one(summary):
    text = render.render_tex(summary)
    assert text == (ROOT / "paper/generated-study3.tex").read_text(encoding="utf-8")
    for label in ("mutants", "learners", "suites", "margin", "predictability", "zoo"):
        assert text.count(rf"\label{{sec:results-{label}}}") == 1
    assert "nan" not in text.lower().replace("nonan", "")


def test_the_manuscript_uses_exactly_the_macros_that_are_generated(summary):
    macros = render.study3_macros(summary)
    manuscript = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    used = set(re.findall(r"\\(Third[A-Za-z]+)", manuscript))
    assert used == set(macros)
    committed = (ROOT / "paper/generated-macros.tex").read_text(encoding="utf-8")
    assert committed.endswith(render.macros_text(summary))
    for name, value in macros.items():
        assert committed.count(rf"\newcommand{{\{name}}}{{{value}}}") == 1


def test_the_numbers_the_abstract_quotes(summary):
    macros = render.study3_macros(summary)
    quoted = {"ThirdSilent": 44, "ThirdSecondCaught": 44, "ThirdPairedCaught": 43, "ThirdKeyCaught": 40,
              "ThirdLossCaught": 37, "ThirdShiftCaught": 22, "ThirdOutputCaught": 3, "ThirdGenerated": 119,
              "ThirdCrashing": 55, "ThirdEquivalent": 20, "ThirdCatalogueSilent": 15, "ThirdMutantSources": 136,
              "ThirdModels": 9, "ThirdMarginCells": 32, "ThirdWithinAbove": 16, "ThirdFailAbove": 1}
    assert {name: macros[name] for name in quoted} == quoted


def test_every_statement_is_checked_against_the_records(summary):
    render.check_study3_conclusions(summary)
    changes = [("part_1", lambda part: part["caught"]["cosmic_ray"]["checks"]["second_implementation"].update(any=43)),
               ("part_2", lambda part: part["calibrated_shift"]["mutants"].update(ci95=[0.19, 0.21])),
               ("part_4", lambda part: part["verdicts"]["WITHIN_TOLERANCE"].update(above=0)),
               ("zoo", lambda part: part.update(failed=1))]
    for name, change in changes:
        changed = copy.deepcopy(summary)
        change(changed[name])
        with pytest.raises(ValueError, match="Recorded evidence does not support"):
            render.check_study3_conclusions(changed)


def test_a_changed_record_is_refused(records, tmp_path):
    protocol = records["protocol"]
    folder = ROOT / "experiments/results-v3/part-3"
    results, manifest = render.load(folder / "results.json"), render.load(folder / "manifest.json")
    render.validate_part("part-3", results, manifest, protocol, folder)
    changed = tmp_path / "part-3"
    changed.mkdir()
    for path in folder.iterdir():
        (changed / path.name).write_bytes(path.read_bytes())
    (changed / "results.json").write_text(json.dumps(results) + " ", encoding="utf-8")
    with pytest.raises(ValueError):
        render.validate_part("part-3", results, manifest, protocol, changed)
    zoo = ROOT / "experiments/results-v3/zoo"
    for model in ("ets", render.PRETRAINED):
        stored = render.load(zoo / f"{model}.manifest.json")
        render.validate_model(model, stored, protocol, zoo)
        with pytest.raises(ValueError, match="registered"):
            render.validate_model(model, {**stored, "protocol_sha256": "0" * 64}, protocol, zoo)
    stored = render.load(zoo / f"{render.PRETRAINED}.manifest.json")
    other = {**stored, "pretrained_environment": {**stored["pretrained_environment"], "resolved_commit": "0" * 40}}
    with pytest.raises(ValueError, match="registered model revision"):
        render.validate_model(render.PRETRAINED, other, protocol, zoo)


def test_the_two_protocol_digests(records):
    protocol = records["protocol"]
    earlier = {records["manifests"][part]["protocol_sha256"] for part in render.PARTS}
    assert earlier == {render.digest_at_the_runs(protocol)}
    pretrained = records["zoo_manifests"][render.PRETRAINED]
    assert pretrained["protocol_sha256"] == sha256(render.PROTOCOL) != render.digest_at_the_runs(protocol)
