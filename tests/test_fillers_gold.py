"""Synthetic gold intervals exercise coverage, preservation, and pooled scores."""

from __future__ import annotations

import json
import math
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from yap_studio.checks.fillers_gold import (
    Gold,
    GoldSpan,
    load_gold,
    load_removals,
    parse_gold,
    score_fillers_gold,
    score_gold_set,
)
from yap_studio.cli import main
from yap_studio.errors import InputError, ValidationError
from yap_studio.project.model import Removals, parse_document

_FIXTURES = Path(__file__).parent / "fixtures" / "filler-gold"


def _span(
    start: float, end: float, label: str = "must-cut", *, take: str = "t1", **extra: Any
) -> dict[str, Any]:
    return {"take": take, "start": start, "end": end, "label": label, **extra}


def _gold(*spans: dict[str, Any], takes: tuple[str, ...] = ("t1",)) -> dict[str, Any]:
    return {"schema": "yap-studio/filler-gold@1", "takes": list(takes), "spans": list(spans)}


def _removal(
    start: float,
    end: float,
    *,
    id: str = "r1",
    take: str = "t1",
    kind: str = "filler",
    status: str = "applied",
    by: str = "fillers",
) -> dict[str, Any]:
    return {"id": id, "take": take, "start": start, "end": end,
            "kind": kind, "status": status, "by": by}


def _removals(*rows: dict[str, Any]) -> dict[str, Any]:
    return {"schema": "yap-studio/removals@1", "removals": list(rows)}


@pytest.mark.parametrize(
    "case,recall,precision,passed,missed,keep_cut,wrong,counts",
    [
        ("clean", 1.0, 1.0, True, [], [], [], (2, 2, 1, 1, 0, 3, 3, 1)),
        ("keep-cut", 1.0, 0.0, False, [], ["g2"], ["r1"], (1, 1, 0, 1, 1, 1, 0, 0)),
        ("partial", 2 / 3, 2 / 3, True, ["g2"], [], ["r4"], (3, 2, 1, 1, 0, 3, 2, 0)),
    ],
)
def test_fixture_scores(
    case: str,
    recall: float,
    precision: float,
    passed: bool,
    missed: list[str],
    keep_cut: list[str],
    wrong: list[str],
    counts: tuple[int, ...],
) -> None:
    folder = _FIXTURES / case
    gold = load_gold(folder / "gold.json")
    removals = load_removals(folder)
    score = score_fillers_gold(gold, removals)

    assert score["schema"] == "yap-studio/filler-gold-score@1"
    assert score["recall"] == recall
    assert score["precision"] == precision
    assert score["f1"] == (2 * recall * precision / (recall + precision))
    assert score["passed"] is passed
    assert [row["id"] for row in score["missed"]] == missed
    assert [row["id"] for row in score["keep_cut"]] == keep_cut
    assert [row["id"] for row in score["wrong"]] == wrong
    assert score["counts"] == dict(zip(
        ("must_cut", "must_cut_recalled", "optional_cut", "must_keep", "must_keep_cut",
         "scored_removals", "correct_removals", "ignored_removals"), counts, strict=True,
    ))
    assert score["settings"] == {
        "statuses": ["applied", "proposed"], "kinds": ["filler", "stutter", "restart"],
        "min_coverage": 0.8, "min_marked": 0.5, "tolerance_s": 1 / 30,
    }
    assert json.loads(json.dumps(score, allow_nan=False)) == score
    assert score_fillers_gold(gold, removals) == score
    assert score_fillers_gold(
        json.loads((folder / "gold.json").read_text()),
        json.loads((folder / "removals.json").read_text()),
    ) == score


def test_fixture_report_details() -> None:
    keep = score_fillers_gold(
        load_gold(_FIXTURES / "keep-cut" / "gold.json"), load_removals(_FIXTURES / "keep-cut")
    )
    assert keep["keep_cut"] == [{
        "id": "g2", "take": "t1", "start": 2.0, "end": 3.0,
        "cut_s": 0.25, "removal_ids": ["r1"],
    }]
    assert keep["wrong"] == [{"id": "r1", "reason": "cuts must-keep"}]
    partial = score_fillers_gold(
        load_gold(_FIXTURES / "partial" / "gold.json"), load_removals(_FIXTURES / "partial")
    )
    assert partial["missed"] == [{
        "id": "g2", "take": "t1", "start": 3.0, "end": 4.0,
        "coverage": pytest.approx(0.4),
    }]
    assert partial["wrong"] == [{"id": "r4", "reason": "unmarked"}]


def test_gold_set_pools_counts_instead_of_averaging() -> None:
    score = score_gold_set(_FIXTURES)
    assert score["schema"] == "yap-studio/filler-gold-set@1"
    assert score["passed"] is False
    assert list(score["cases"]) == ["clean", "keep-cut", "partial"]
    counts = score["totals"]["counts"]
    assert counts == {
        "must_cut": 6, "must_cut_recalled": 5, "optional_cut": 2,
        "must_keep": 3, "must_keep_cut": 1, "scored_removals": 7,
        "correct_removals": 5, "ignored_removals": 1,
    }
    for key, total in counts.items():
        assert total == sum(case["counts"][key] for case in score["cases"].values())
    assert score["totals"]["recall"] == 5 / 6
    assert score["totals"]["precision"] == 5 / 7
    assert score["totals"]["f1"] == pytest.approx(10 / 13)
    assert score["totals"]["recall"] != sum(
        case["recall"] for case in score["cases"].values()
    ) / 3
    assert json.loads(json.dumps(score, allow_nan=False)) == score


@pytest.mark.parametrize("extra_s,passed", [(0.0, True), (0.5e-9, True), (2e-9, False)])
def test_must_keep_tolerance_edge(extra_s: float, passed: bool) -> None:
    tolerance = 1 / 30
    score = score_fillers_gold(
        _gold(_span(0, 1), _span(1, 2, "must-keep")),
        _removals(_removal(0, 1 + tolerance + extra_s)), tolerance_s=tolerance,
    )
    assert score["passed"] is passed
    assert score["recall"] == 1.0
    assert score["precision"] == float(passed)
    assert score["counts"]["must_keep_cut"] == int(not passed)
    assert score["wrong"] == ([] if passed else [{"id": "r1", "reason": "cuts must-keep"}])


@pytest.mark.parametrize("coverage,recalled", [(0.8 - 2e-9, False), (0.8, True)])
def test_minimum_coverage_edge(coverage: float, recalled: bool) -> None:
    score = score_fillers_gold(_gold(_span(0, 1)), _removals(_removal(0, coverage)))
    assert score["recall"] == float(recalled)
    assert score["precision"] == 1.0
    assert score["passed"] is True
    assert score["counts"]["must_cut_recalled"] == int(recalled)
    assert score["missed"] == ([] if recalled else [{
        "id": "g1", "take": "t1", "start": 0.0, "end": 1.0, "coverage": coverage,
    }])


def test_removal_union_recall_avoids_double_counting() -> None:
    gold = _gold(_span(0, 1), _span(2, 3))
    score = score_fillers_gold(gold, _removals(
        _removal(0, 0.5), _removal(0.1, 0.6, id="r2"),
        _removal(2, 2.4, id="r3"), _removal(2.6, 3, id="r4", kind="silence"),
    ))
    assert score["recall"] == 0.5
    assert score["precision"] == 1.0
    assert score["missed"] == [{
        "id": "g1", "take": "t1", "start": 0.0, "end": 1.0, "coverage": 0.6,
    }]


def test_must_keep_union_counts_overlaps_once_and_sums_separate_cuts() -> None:
    gold = _gold(_span(0, 1, "must-keep"))
    rows = _removals(_removal(0, 0.02), _removal(0.01, 0.03, id="r2", kind="silence"))
    score = score_fillers_gold(gold, rows)
    assert score["passed"] is True
    rows["removals"].extend([
        _removal(0.1, 0.12, id="r3", kind="silence"),
        _removal(1, 2, id="r4"),
    ])
    score = score_fillers_gold(gold, rows)
    assert score["passed"] is False
    assert score["keep_cut"] == [{
        "id": "g1", "take": "t1", "start": 0.0, "end": 1.0,
        "cut_s": pytest.approx(0.05), "removal_ids": ["r1", "r2", "r3"],
    }]


def test_precision_unions_marks_but_checks_must_keep_per_span() -> None:
    marks = _gold(_span(0, 0.02), _span(0.01, 0.03, "optional-cut"))
    score = score_fillers_gold(marks, _removals(_removal(0, 0.5)))
    assert score["precision"] == 0.0
    assert score["wrong"] == [{"id": "r1", "reason": "mostly unmarked"}]
    keeps = _gold(
        _span(0, 0.5), _span(0.5, 0.52, "must-keep"), _span(0.51, 0.53, "must-keep")
    )
    score = score_fillers_gold(keeps, _removals(_removal(0, 0.53)))
    assert score["passed"] is False
    assert score["precision"] == 0.0
    assert score["counts"]["must_keep_cut"] == 2
    assert score["wrong"] == [{"id": "r1", "reason": "cuts must-keep"}]
    separate = _gold(
        _span(0, 0.5), _span(0.5, 0.52, "must-keep"), _span(0.6, 0.62, "must-keep")
    )
    score = score_fillers_gold(separate, _removals(_removal(0, 0.62)))
    assert score["passed"] is False
    assert score["precision"] == 0.0
    assert score["counts"]["must_keep_cut"] == 2
    assert score["wrong"] == [{"id": "r1", "reason": "cuts must-keep"}]


@pytest.mark.parametrize("length,precision", [(0.5e-9, 0.0), (1e-9, 0.0), (2e-9, 1.0), (0.01, 1.0)])
def test_marked_overlap_uses_epsilon_instead_of_tolerance(length: float, precision: float) -> None:
    score = score_fillers_gold(
        _gold(_span(0, length, "optional-cut")), _removals(_removal(0, length))
    )
    assert score["precision"] == precision
    assert score["recall"] == 1.0
    assert score["counts"]["correct_removals"] == int(precision)
    assert score["wrong"] == (
        [] if precision else [{"id": "r1", "reason": "mostly unmarked"}]
    )


def test_statuses_and_manual_authorship() -> None:
    gold = _gold(_span(0, 1), _span(1, 2, "must-keep"))
    rows = _removals(
        _removal(0, 1, status="proposed"),
        _removal(1, 2, id="r2", status="rejected"),
        _removal(1, 2, id="r3", by="manual"),
    )
    score = score_fillers_gold(gold, rows)
    assert score["passed"] is True
    assert score["recall"] == score["precision"] == 1.0
    assert score["counts"]["scored_removals"] == 1
    applied = score_fillers_gold(gold, rows, statuses=("applied",))
    assert applied["passed"] is True
    assert applied["recall"] == 0.0
    assert applied["precision"] == 1.0
    assert applied["counts"]["scored_removals"] == 0
    assert applied["settings"]["statuses"] == ["applied"]
    proposed_keep = _removals(_removal(1, 2, status="proposed"))
    assert score_fillers_gold(gold, proposed_keep)["passed"] is False
    assert score_fillers_gold(gold, proposed_keep, statuses=("applied",))["passed"] is True


def test_silence_counts_for_recall_and_preservation_but_precision_is_opt_in() -> None:
    gold = _gold(_span(0, 1), _span(2, 3, "must-keep"))
    rows = _removals(_removal(0, 1, kind="silence"), _removal(2, 3, id="r2", kind="silence"))
    score = score_fillers_gold(gold, rows)
    assert score["recall"] == score["precision"] == 1.0
    assert score["passed"] is False
    assert score["counts"]["scored_removals"] == 0
    assert score["keep_cut"][0]["removal_ids"] == ["r2"]
    opted_in = score_fillers_gold(gold, rows, kinds=("silence",))
    assert opted_in["precision"] == 0.5
    assert opted_in["wrong"] == [{"id": "r2", "reason": "cuts must-keep"}]


def test_takes_do_not_share_intervals_and_uncovered_removals_are_reported() -> None:
    gold = _gold(_span(0, 1), _span(0, 1, "must-keep", take="t2"), takes=("t1", "t2"))
    rows = _removals(_removal(0, 1), _removal(0, 1, id="r2", take="t3"))
    score = score_fillers_gold(gold, rows)
    assert score["passed"] is True
    assert score["recall"] == score["precision"] == 1.0
    assert score["counts"]["ignored_removals"] == 1
    assert score["counts"]["scored_removals"] == 1


@pytest.mark.parametrize(
    "gold,rows,recall,precision,f1",
    [
        (_gold(), _removals(), 1.0, 1.0, 1.0),
        (_gold(_span(0, 1)), _removals(), 0.0, 1.0, 0.0),
        (_gold(), _removals(_removal(0, 1)), 1.0, 0.0, 0.0),
    ],
)
def test_empty_denominators(
    gold: dict, rows: dict, recall: float, precision: float, f1: float
) -> None:
    score = score_fillers_gold(gold, rows)
    assert (score["recall"], score["precision"], score["f1"]) == (recall, precision, f1)
    assert score["passed"] is True


@pytest.mark.parametrize(
    "span",
    [
        _span(0, 1, "bad"), _span(1, 1), _span(2, 1), _span(-1, 1),
        _span(math.nan, 1), _span(0, math.inf), _span(-math.inf, 1),
        _span(0, 1, take="t2"), _span(0, 1, unexpected=True),
        _span("0", 1), _span(True, 1), _span(0, 1, note=123),
        _span(0, 1, note=None), _span(0, 1, id=None), _span(0, 10**400),
    ],
)
def test_invalid_gold_spans_name_the_span(span: dict) -> None:
    with pytest.raises(ValidationError, match="span") as caught:
        parse_gold(_gold(span))
    assert caught.value.code == "invalid_parameter"
    assert caught.value.to_dict()["type"] == "validation_error"


@pytest.mark.parametrize("label", ["must-cut", "optional-cut"])
def test_must_keep_cannot_overlap_cut_marks(label: str) -> None:
    with pytest.raises(ValidationError, match="g2.*overlaps.*g1"):
        parse_gold(_gold(_span(0, 1, label), _span(1 - 1e-12, 2, "must-keep")))
    assert len(parse_gold(_gold(_span(0, 1, label), _span(1, 2, "must-keep"))).spans) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "bad"}, {"unknown": 1}, {"takes": []}, {"takes": ["t1", "t1"]},
        {"takes": ["t0"]}, {"takes": ["t01"]}, {"takes": ["t1\n"]},
        {"takes": [123]}, {"takes": "t1"}, {"spans": None},
    ],
)
def test_invalid_gold_document(changes: dict) -> None:
    with pytest.raises(ValidationError):
        parse_gold({**_gold(), **changes})


@pytest.mark.parametrize("field", ["schema", "takes", "spans"])
def test_gold_requires_all_document_fields(field: str) -> None:
    data = _gold()
    del data[field]
    with pytest.raises(ValidationError, match=field):
        parse_gold(data)


@pytest.mark.parametrize(
    "settings",
    [
        {"min_coverage": 0}, {"min_coverage": -0.1}, {"min_coverage": 1.01},
        {"min_coverage": math.nan}, {"min_coverage": math.inf}, {"min_coverage": True},
        {"min_marked": 0}, {"min_marked": -0.1}, {"min_marked": 1.01},
        {"min_marked": math.nan}, {"min_marked": math.inf}, {"min_marked": -math.inf},
        {"min_marked": True}, {"min_marked": "0.5"}, {"min_marked": None},
        {"tolerance_s": -0.1}, {"tolerance_s": math.inf}, {"tolerance_s": math.nan},
        {"tolerance_s": "0"}, {"statuses": ("bad",)}, {"kinds": ("bad",)},
        {"statuses": "applied"}, {"kinds": (None,)},
    ],
)
def test_invalid_settings(settings: dict) -> None:
    with pytest.raises(ValidationError) as caught:
        score_fillers_gold(_gold(), _removals(), **settings)
    assert caught.value.parameter == next(iter(settings))


def test_valid_setting_limits_and_empty_selections() -> None:
    score = score_fillers_gold(
        _gold(_span(0, 1)), _removals(_removal(0, 1)), min_coverage=1, min_marked=1, tolerance_s=0,
    )
    assert score["recall"] == score["precision"] == 1.0
    empty = score_fillers_gold(_gold(_span(0, 1)), _removals(_removal(0, 1)), statuses=())
    assert empty["recall"] == 0.0 and empty["precision"] == 1.0
    no_precision = score_fillers_gold(_gold(), _removals(_removal(0, 1)), kinds=())
    assert no_precision["recall"] == no_precision["precision"] == 1.0


def test_load_removals_accepts_project_folder_and_file() -> None:
    folder = _FIXTURES / "partial"
    model = load_removals(folder)
    assert isinstance(model, Removals)
    assert load_removals(folder / "removals.json") == model
    assert model == parse_document("removals", json.loads((folder / "removals.json").read_text()))
    assert len(model.removals) == 8


def test_loaded_removals_use_existing_schema_validation(tmp_path: Path) -> None:
    source = tmp_path / "removals.json"
    rows = _removals(_removal(0, 1, kind="unknown"))
    source.write_text(json.dumps(rows))
    with pytest.raises(ValidationError, match="kind"):
        load_removals(source)
    with pytest.raises(ValidationError, match="kind"):
        score_fillers_gold(_gold(), rows)


@pytest.mark.parametrize(
    "duplicate_id,second_id",
    [("duplicate", {"id": "duplicate"}), ("g2", {})],
    ids=["explicit-ids", "explicit-and-generated-id"],
)
def test_duplicate_gold_span_ids_are_rejected(duplicate_id: str, second_id: dict) -> None:
    with pytest.raises(ValidationError, match="duplicate id") as caught:
        parse_gold(_gold(_span(0, 1, id=duplicate_id), _span(2, 3, **second_id)))
    assert caught.value.parameter == f"span {duplicate_id!r}"


def test_direct_gold_construction_rejects_duplicate_span_ids_across_takes() -> None:
    with pytest.raises(ValidationError, match="duplicate id") as caught:
        Gold(("t1", "t2"), (
            GoldSpan("duplicate", "t1", 0, 1, "must-cut"),
            GoldSpan("duplicate", "t2", 0, 1, "must-cut"),
        ))
    assert caught.value.parameter == "span 'duplicate'"


def test_gold_models_are_frozen_and_keep_explicit_ids_and_notes() -> None:
    gold = load_gold(_FIXTURES / "clean" / "gold.json")
    assert gold.takes == ("t1",)
    assert [span.id for span in gold.spans] == ["g1", "cut-restart", "keep", "g4"]
    assert gold.spans[0].note == "um"
    with pytest.raises(FrozenInstanceError):
        gold.spans[0].start = 0
    with pytest.raises(FrozenInstanceError):
        gold.takes = ("t2",)


def test_missing_removals_in_a_set_names_the_case(tmp_path: Path) -> None:
    case = tmp_path / "missing-case"
    case.mkdir()
    (case / "gold.json").write_text(json.dumps(_gold()))
    with pytest.raises(InputError, match="missing-case/removals.json"):
        score_gold_set(tmp_path)


def test_empty_or_missing_gold_set_is_an_input_error(tmp_path: Path) -> None:
    (tmp_path / "unrelated").mkdir()
    with pytest.raises(InputError, match="no cases"):
        score_gold_set(tmp_path)
    with pytest.raises(InputError):
        score_gold_set(tmp_path / "missing")


def test_gold_set_accepts_a_private_folder_and_forwards_options(tmp_path: Path) -> None:
    case = tmp_path / "case"
    case.mkdir()
    (case / "gold.json").write_text(json.dumps(_gold(_span(0, 1))))
    (case / "removals.json").write_text(json.dumps(_removals(
        _removal(0, 0.6), _removal(0.6, 1, id="r2", status="proposed"),
    )))
    nested = case / "nested"
    nested.mkdir()
    (nested / "gold.json").write_text(json.dumps(_gold()))
    (tmp_path / "gold.json").write_text(json.dumps(_gold()))
    score = score_gold_set(
        tmp_path, statuses=("applied",), kinds=("filler",), min_coverage=0.6,
        min_marked=0.9, tolerance_s=0,
    )
    assert list(score["cases"]) == ["case"]
    assert score["passed"] is True
    assert score["totals"]["recall"] == score["totals"]["precision"] == 1.0
    assert score["cases"]["case"]["settings"] == {
        "statuses": ["applied"], "kinds": ["filler"], "min_coverage": 0.6,
        "min_marked": 0.9, "tolerance_s": 0.0,
    }


def test_unreadable_and_malformed_json_are_input_errors(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        load_gold(tmp_path / "missing.json")
    with pytest.raises(InputError):
        load_removals(tmp_path)
    source = tmp_path / "gold.json"
    source.write_text("{")
    with pytest.raises(InputError, match="cannot read JSON"):
        load_gold(source)


def test_dev_private_gold_acceptance() -> None:
    directory = os.environ.get("YAP_STUDIO_FILLER_GOLD")
    if not directory:
        pytest.skip("Set YAP_STUDIO_FILLER_GOLD to a private gold set folder.")
    score = score_gold_set(Path(directory))
    assert score["passed"], score


def test_fillers_score_clean_file_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fillers-score", str(_FIXTURES / "clean" / "gold.json")]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "recall 100% (2 of 2 must-cut)\n"
        "precision 100% (3 of 3 removals)\n"
        "passed\n"
    )


def test_fillers_score_keep_cut_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fillers-score", str(_FIXTURES / "keep-cut" / "gold.json")]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "MUST-KEEP CUT g2 t1 2.000 to 3.000 by r1 (0.250 s)\n" in captured.out
    assert "wrong r1: cuts must-keep\n" in captured.out
    assert captured.out.endswith("FAILED: 1 must-keep spans cut\n")


def test_fillers_score_partial_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fillers-score", str(_FIXTURES / "partial" / "gold.json")]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "recall 67% (2 of 3 must-cut)\n"
        "precision 67% (2 of 3 removals)\n"
        "missed g2 t1 3.000 to 4.000 coverage 40%\n"
        "wrong r4: unmarked\n"
        "passed\n"
    )


def test_fillers_score_partial_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fillers-score", str(_FIXTURES / "partial" / "gold.json"), "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    score = json.loads(captured.out)
    assert score["schema"] == "yap-studio/filler-gold-score@1"
    assert score["recall"] == score["precision"] == 2 / 3
    assert score["passed"] is True


@pytest.mark.parametrize("json_output", [False, True])
def test_fillers_score_gold_set(
    capsys: pytest.CaptureFixture[str], json_output: bool,
) -> None:
    args = ["fillers-score", str(_FIXTURES)] + (["--json"] if json_output else [])
    assert main(args) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        score = json.loads(captured.out)
        assert score["schema"] == "yap-studio/filler-gold-set@1"
        assert score["passed"] is False
        assert score["totals"]["recall"] == 5 / 6
        assert score["totals"]["precision"] == 5 / 7
        assert list(score["cases"]) == ["clean", "keep-cut", "partial"]
    else:
        assert all(f"{name}:\n" in captured.out for name in ("clean", "keep-cut", "partial"))
        assert "MUST-KEEP CUT g2 t1 2.000 to 3.000 by r1 (0.250 s)" in captured.out
        assert captured.out.endswith(
            "pooled totals: recall 83% (5 of 6 must-cut), precision 71% (5 of 7 removals)\n"
        )


@pytest.mark.parametrize("project_folder", [False, True])
def test_fillers_score_explicit_removals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], project_folder: bool,
) -> None:
    gold = tmp_path / "gold.json"
    gold.write_bytes((_FIXTURES / "partial" / "gold.json").read_bytes())
    project = tmp_path / "project"
    project.mkdir()
    removals = project / "removals.json"
    removals.write_bytes((_FIXTURES / "partial" / "removals.json").read_bytes())
    source = project if project_folder else removals
    assert main(["fillers-score", str(gold), "--removals", str(source), "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    score = json.loads(captured.out)
    assert score["recall"] == score["precision"] == 2 / 3


def test_fillers_score_applied_status_replaces_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([
        "fillers-score", str(_FIXTURES / "partial" / "gold.json"),
        "--status", "applied", "--json",
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    score = json.loads(captured.out)
    assert score["settings"]["statuses"] == ["applied"]
    assert score["counts"]["scored_removals"] == 2
    assert score["counts"]["correct_removals"] == 1
    assert score["recall"] == 2 / 3
    assert score["precision"] == 1 / 2


def test_fillers_score_repeated_selections_and_numeric_settings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([
        "fillers-score", str(_FIXTURES / "partial" / "gold.json"),
        "--status", "applied", "--status", "rejected",
        "--kind", "filler", "--kind", "silence",
        "--min-coverage", "0.5", "--tolerance-s", "0", "--json",
    ]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    score = json.loads(captured.out)
    assert score["settings"] == {
        "statuses": ["applied", "rejected"], "kinds": ["filler", "silence"],
        "min_coverage": 0.5, "min_marked": 0.5, "tolerance_s": 0.0,
    }
    assert score["counts"]["scored_removals"] == 4
    assert score["counts"]["correct_removals"] == 3
    assert score["recall"] == 1.0
    assert score["precision"] == 3 / 4
    assert score["keep_cut"][0]["removal_ids"] == ["r8"]


def test_fillers_score_forwards_set_settings(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([
        "fillers-score", str(_FIXTURES), "--status", "rejected", "--kind", "silence",
        "--min-coverage", "0.5", "--tolerance-s", "0", "--json",
    ]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    score = json.loads(captured.out)
    assert all(case["settings"] == {
        "statuses": ["rejected"], "kinds": ["silence"],
        "min_coverage": 0.5, "min_marked": 0.5, "tolerance_s": 0.0,
    } for case in score["cases"].values())


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("problem", ["set-removals", "min-coverage", "min-marked", "missing-gold"])
def test_fillers_score_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], problem: str, json_output: bool,
) -> None:
    if problem == "set-removals":
        args = ["fillers-score", str(_FIXTURES), "--removals", str(tmp_path)]
        error = InputError(str(_FIXTURES), "--removals cannot be used with a gold set folder")
    elif problem == "min-coverage":
        args = ["fillers-score", str(_FIXTURES / "clean" / "gold.json"), "--min-coverage", "0"]
        error = ValidationError("min_coverage", "require 0 < min_coverage <= 1")
    elif problem == "min-marked":
        args = ["fillers-score", str(_FIXTURES / "clean" / "gold.json"), "--min-marked", "0"]
        error = ValidationError("min_marked", "require 0 < min_marked <= 1")
    else:
        args = ["fillers-score", str(tmp_path / "missing.json")]
        error = None
    if json_output:
        args.append("--json")
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        if error is not None:
            assert json.loads(captured.err) == {"error": error.to_dict()}
        else:
            reported = json.loads(captured.err)["error"]
            assert reported["type"] == "input_error"
            assert "missing.json" in reported["message"]
    elif error is not None:
        assert captured.err == f"error: {error.message}\n"
    else:
        assert captured.err.startswith("error: ")
        assert "missing.json" in captured.err


def test_overlong_removal_is_mostly_unmarked() -> None:
    score = score_fillers_gold(_gold(_span(50, 51)), _removals(_removal(0, 100)))
    assert score["recall"] == 1.0
    assert score["precision"] == score["f1"] == 0.0
    assert score["counts"]["scored_removals"] == 1
    assert score["counts"]["correct_removals"] == 0
    assert score["wrong"] == [{"id": "r1", "reason": "mostly unmarked"}]
    assert score["passed"] is True


def test_removal_with_silence_padding_is_correct() -> None:
    score = score_fillers_gold(_gold(_span(0.1, 0.4)), _removals(_removal(0, 0.5)))
    assert score["recall"] == score["precision"] == score["f1"] == 1.0
    assert score["counts"]["correct_removals"] == 1
    assert score["wrong"] == []


@pytest.mark.parametrize("min_marked", [0.5, 0.8, 1.0])
@pytest.mark.parametrize("just_under", [False, True])
def test_min_marked_share_boundary(min_marked: float, just_under: bool) -> None:
    marked = math.nextafter(min_marked, 0.0) if just_under else min_marked
    score = score_fillers_gold(
        _gold(_span(0, marked)), _removals(_removal(0, 1)), min_marked=min_marked,
    )
    assert score["recall"] == 1.0
    assert score["precision"] == float(not just_under)
    assert score["counts"]["correct_removals"] == int(not just_under)
    assert score["settings"]["min_marked"] == min_marked
    assert score["wrong"] == (
        [{"id": "r1", "reason": "mostly unmarked"}] if just_under else []
    )


def test_short_must_cut_match_has_full_precision() -> None:
    score = score_fillers_gold(_gold(_span(0, 0.01)), _removals(_removal(0, 0.01)))
    assert score["recall"] == score["precision"] == score["f1"] == 1.0
    assert score["counts"]["must_cut_recalled"] == 1
    assert score["counts"]["correct_removals"] == 1
    assert score["wrong"] == []


def test_zero_length_removal_raises_the_shared_parser_error() -> None:
    rows = _removals(_removal(1.5, 1.5))
    with pytest.raises(ValidationError, match="must exceed start") as expected:
        parse_document("removals", rows)
    with pytest.raises(type(expected.value)) as caught:
        score_fillers_gold(_gold(_span(1, 2)), rows)
    assert caught.value.to_dict() == expected.value.to_dict()


@pytest.mark.parametrize(
    "cut_s,passed", [(0.02, False), (0.009, True), (0.01, True), (0.01 + 0.5e-9, False)],
)
def test_short_must_keep_uses_half_its_length(cut_s: float, passed: bool) -> None:
    gold = _gold(_span(0, 1), _span(1, 1.02, "must-keep"))
    score = score_fillers_gold(gold, _removals(_removal(0, 1 + cut_s)))
    assert score["passed"] is passed
    assert score["recall"] == 1.0
    assert score["precision"] == float(passed)
    assert score["counts"]["must_keep_cut"] == int(not passed)
    assert score["wrong"] == ([] if passed else [{"id": "r1", "reason": "cuts must-keep"}])
    if passed:
        assert score["keep_cut"] == []
    else:
        assert score["keep_cut"] == [{
            "id": "g2", "take": "t1", "start": 1.0, "end": 1.02,
            "cut_s": pytest.approx(cut_s), "removal_ids": ["r1"],
        }]


def test_full_must_keep_cut_fails_even_below_comparison_epsilon() -> None:
    score = score_fillers_gold(
        _gold(_span(0, 0.5e-9, "must-keep")), _removals(_removal(0, 0.5e-9)),
    )
    assert score["passed"] is False
    assert score["precision"] == 0.0
    assert score["counts"]["must_keep_cut"] == 1
    assert score["keep_cut"][0]["cut_s"] == 0.5e-9
    assert score["wrong"] == [{"id": "r1", "reason": "cuts must-keep"}]


def test_precision_does_not_sum_grazes_of_separate_keep_spans() -> None:
    gold = _gold(_span(0, 1, "must-keep"), _span(1, 2), _span(2, 3, "must-keep"))
    score = score_fillers_gold(gold, _removals(_removal(0.98, 2.02)))
    assert score["recall"] == score["precision"] == 1.0
    assert score["counts"]["must_keep_cut"] == 0
    assert score["counts"]["correct_removals"] == 1
    assert score["keep_cut"] == score["wrong"] == []
    assert score["passed"] is True


@pytest.mark.parametrize("set_folder", [False, True])
def test_fillers_score_min_marked_changes_precision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], set_folder: bool,
) -> None:
    case = tmp_path / "case"
    case.mkdir()
    gold = case / "gold.json"
    gold.write_text(json.dumps(_gold(_span(0.1, 0.4))))
    (case / "removals.json").write_text(json.dumps(_removals(_removal(0, 0.5))))
    args = ["fillers-score", str(tmp_path if set_folder else gold), "--json"]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    result = json.loads(captured.out)
    score = result["cases"]["case"] if set_folder else result
    assert score["precision"] == 1.0
    assert score["settings"]["min_marked"] == 0.5
    assert main([*args, "--min-marked", "0.7"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    result = json.loads(captured.out)
    score = result["cases"]["case"] if set_folder else result
    assert score["recall"] == 1.0
    assert score["precision"] == 0.0
    assert score["counts"]["correct_removals"] == 0
    assert score["settings"]["min_marked"] == 0.7
    assert score["wrong"] == [{"id": "r1", "reason": "mostly unmarked"}]
    if set_folder:
        assert result["totals"]["precision"] == 0.0
        assert result["totals"]["counts"]["correct_removals"] == 0
