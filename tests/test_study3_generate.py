"""The Cosmic Ray wrapper of Study 3 on a toy module (tests/fixtures/study3_cr_toy.py). Skipped where cosmic_ray is
not importable; run it with the Cosmic Ray interpreter. Mutated sources are compiled, never executed."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("cosmic_ray")

from experiments import study3_generate as generate  # noqa: E402

TOY = Path(__file__).parent / "fixtures" / "study3_cr_toy.py"


def test_enumeration_matches_the_operators():
    mutants, skipped = generate.cosmic_ray_mutants(TOY)
    names = generate.operator_names()
    assert len(names) == 213 and len({generate.family(name) for name in names}) == 16
    assert skipped == [name for name in names if generate.family(name) in ("VariableInserter", "VariableReplacer")]
    by_operator = {}
    for mutant in mutants:
        by_operator.setdefault(mutant["operator"], []).append(mutant["occurrence"])
    add = [name for name in names if name.startswith("core/ReplaceBinaryOperator_Add_")]
    # Cosmic Ray does not replace a comparison with a number literal by "is" or "is not".
    greater = [name for name in names if name.startswith("core/ReplaceComparisonOperator_Gt_")
               and not name.endswith(("_Is", "_IsNot"))]
    expected = {**{name: [0] for name in add + greater}, "core/NumberReplacer": [0, 1]}
    assert by_operator == expected
    sources = {mutant["label"]: mutant["source"] for mutant in mutants}
    assert "return a - b" in sources["core/ReplaceBinaryOperator_Add_Sub#0"]
    assert "return x > 1" in sources["core/NumberReplacer#0"]
    assert "return x > -1" in sources["core/NumberReplacer#1"]
    assert {mutant["definition"] for mutant in mutants} == {"add", "is_positive"}
    for mutant in mutants:
        compile(mutant["source"], mutant["label"], "exec")


def test_generate_deduplicates_and_writes_the_manifest(tmp_path):
    toy = TOY.read_text()
    extra = [{"origin": "catalogue", "label": "hand_written_sub", "operator": "hand_written_sub", "family": None,
              "site": None, "source": toy.replace("return a + b", "return a  -  b")},
             {"origin": "study2", "label": "unchanged", "operator": "unchanged", "family": None,
              "source": toy.replace('"""A toy', '"""Another toy')}]
    manifest = generate.generate(TOY, tmp_path, extra)
    mutants, _ = generate.cosmic_ray_mutants(TOY)
    assert manifest["counts"]["labels"] == len(mutants) + 2
    assert manifest["counts"]["by_kind"] == {"cosmic_ray": len(mutants), "catalogue": 1, "study2": 1}
    assert manifest["counts"]["distinct_sources"] == len(mutants) + 1
    assert manifest["tool"]["version"] == "8.9.0" and len(manifest["tool"]["operators"]) == 213
    assert manifest["subject"]["sha256"] == hashlib.sha256(TOY.read_bytes()).hexdigest()
    entries = {tuple(label["label"] for label in entry["labels"]): entry for entry in manifest["sources"]}
    shared = entries[("core/ReplaceBinaryOperator_Add_Sub#0", "hand_written_sub")]
    assert [label["kind"] for label in shared["labels"]] == ["cosmic_ray", "catalogue"]
    assert entries[("unchanged",)]["identical_to_subject"] is False
    for entry in manifest["sources"]:
        data = (tmp_path / entry["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        ast.parse(data.decode())
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        generate.generate(TOY, tmp_path, extra)


def test_generation_refuses_a_draft_protocol(tmp_path, monkeypatch):
    from experiments import study3_common as common

    draft = common.load_protocol()
    draft.update(status="draft")
    monkeypatch.setattr(common, "load_protocol", lambda: draft)
    with pytest.raises(RuntimeError, match="generation may not start"):
        generate.main(["--output", str(tmp_path / "mutants")])
    assert not (tmp_path / "mutants").exists()
