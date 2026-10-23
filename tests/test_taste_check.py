"""Applied rule comparisons and measurements of finished pictures."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import synth
import taste_finals

from yap_studio import cli, ffmpeg
from yap_studio.errors import InputError
from yap_studio.style import SHIPPED_PRESETS, resolve
from yap_studio.taste import (
    Taste,
    add_proposals,
    approve,
    check_taste,
    learn,
    load_taste,
    measure_style,
    read_taste_record,
    render_style,
    revoke,
    validate_taste,
    write_taste_record,
)
from yap_studio.taste.learn import UNMEASURED, tolerance
from yap_studio.taste.measure_words import LIMIT_FIELDS, NEEDS_WORDS
from yap_studio.transcript import Transcript, Word
from yap_studio.verify import verify


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_setup_rules_are_enforced_and_named(tmp_path, monkeypatch, capsys, failed, legacy):
    from yap_studio.checks.model import Finding
    from yap_studio.style.answers import ANSWERS_SCHEMA, taste_rules
    from yap_studio.taste.store import approve_proposal

    rules = taste_rules({"schema": ANSWERS_SCHEMA, "never": {"cover_face": True, "emoji": True}})
    for rule in rules:
        approve_proposal(rule)
    application = render_style()
    assert len(application.pin()["enforced"]) == 3
    assert [item["reason"] for item in application.skipped] == ["followed_by_agent"]
    media = tmp_path / "final.mp4"
    media.write_bytes(b"synthetic render")
    write_taste_record(media, application)
    if legacy:
        target = media.with_suffix(".taste.json")
        old = json.loads(target.read_text())
        old["skipped"].extend({"id": item["id"], "reason": "not_a_style_rule"}
                              for item in old.pop("enforced"))
        old["skipped"][0]["reason"] = "not_a_style_rule"
        target.write_text(json.dumps(old))
    record = read_taste_record(media)
    finding = Finding("faces.clearance", "fail", "Caption covers the face.",
                      evidence={"target_type": "caption", "frame": 12})
    clearance = ([finding] if failed else [], {"ran": True, "reason": None})
    module = importlib.import_module("yap_studio.taste.check")
    monkeypatch.setattr(module, "_face_clearance", lambda _media: clearance)
    checked = check_taste(media, record)
    assert checked.coverage["ran"]
    placement = [row for row in checked.rules if row.get("check") == "faces.clearance"]
    assert len(placement) == 3
    assert {row["field"]: row["status"] for row in placement} == {
        "captions clear of face": "fail" if failed else "pass",
        "title clear of face": "pass", "cards clear of face": "pass",
    }
    assert all(not row["evidence"] for row in placement if row["field"] != "captions clear of face")
    failures = [item for item in checked.findings if item.severity == "fail"]
    assert len(failures) == int(failed)
    if failed:
        assert failures[0].evidence["findings"] == [finding.evidence]
    assert cli.main(["taste", "check", str(media)]) == int(failed)
    text = capsys.readouterr().out
    for layer in ("captions", "title", "cards"):
        status = "fail" if failed and layer == "captions" else "pass"
        assert f"{layer} clear of face: {status} (faces.clearance)" in text
    assert "No emoji on screen.: followed by agent" in text
    assert "no approved taste rule" not in text


@pytest.mark.parametrize("failed", [False, True])
def test_verify_runs_placement_taste_rules(tmp_path, failed):
    from test_face_clearance_check import FACE, _faces

    from yap_studio.checks.caption_layout import write_layout_record
    from yap_studio.faces.detect import write_faces
    from yap_studio.style.answers import ANSWERS_SCHEMA, taste_rules
    from yap_studio.taste.store import approve_proposal

    for rule in taste_rules({"schema": ANSWERS_SCHEMA, "never": {"cover_face": True}}):
        approve_proposal(rule)
    media = synth.calib_source(tmp_path / "final.mov", seconds=1, size="320x180")
    write_taste_record(media, render_style())
    layout = synth.calibration_layout()
    layout.update(captions=[{
        **layout["captions"][0], "text": "Hello", "start_frame": 1, "end_frame": 29,
        "words": [{"text": "Hello", "start_s": 0.1, "end_s": 0.8}],
        "boxes": [{"x": 120, "y": 70, "w": 80, "h": 60}],
    }], elements=[])
    write_layout_record(media, layout)
    write_faces(media.with_suffix(".faces.json"),
                _faces(media, frames=30, box=FACE if failed else None))
    report = verify(str(media), asr="none")
    assert report["checks"]["taste.rules"]["ran"]
    assert report["checks"]["taste.rules"]["status"] == ("fail" if failed else "pass")


def test_unmeasured_captions_do_not_pass_placement_rule(tmp_path):
    from test_face_clearance_check import _faces

    from yap_studio.checks.caption_layout import write_layout_record
    from yap_studio.faces.detect import write_faces
    from yap_studio.style.answers import ANSWERS_SCHEMA, taste_rules
    from yap_studio.taste.store import approve_proposal

    for rule in taste_rules({"schema": ANSWERS_SCHEMA, "never": {"cover_face": True}}):
        approve_proposal(rule)
    media = synth.calib_source(tmp_path / "final.mov", seconds=1, size="320x180")
    write_taste_record(media, render_style())
    layout = synth.calibration_layout()
    layout.update(measured=False, reason="caption geometry unavailable", captions=[], elements=[])
    write_layout_record(media, layout)
    write_faces(media.with_suffix(".faces.json"), _faces(media, frames=30, box=None))
    checked = check_taste(media, read_taste_record(media))
    placement = {row["field"]: row for row in checked.rules}
    captions = placement["captions clear of face"]
    assert captions["status"] == "not_measured"
    assert captions["reason"] == "captions were not measured: caption geometry unavailable"
    assert placement["title clear of face"]["status"] == "pass"
    assert placement["cards clear of face"]["status"] == "pass"


def test_no_approved_rules_says_so(tmp_path):
    result = check_taste(tmp_path / "unused.mp4", render_style().pin())
    assert result.rules == []
    assert result.coverage == {
        "ran": False, "reason": "no approved taste rule applied to this render",
    }


def test_setup_rules_with_missing_history_still_have_readable_status(tmp_path, monkeypatch, capsys):
    from yap_studio.style.answers import ANSWERS_SCHEMA, taste_rules
    from yap_studio.taste.store import approve_proposal

    for rule in taste_rules({"schema": ANSWERS_SCHEMA,
                            "never": {"cover_face": True, "emoji": True}}):
        approve_proposal(rule)
    media = tmp_path / "final.mp4"
    media.write_bytes(b"synthetic render")
    write_taste_record(media, render_style())
    monkeypatch.setattr(importlib.import_module("yap_studio.taste.check"), "load_taste", Taste)
    assert cli.main(["taste", "check", str(media)]) == 0
    text = capsys.readouterr().out
    assert text.count("not measured") == 4
    assert "does not match the pinned revision 4" in text


@pytest.fixture(autouse=True)
def isolated_taste(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


def _approve(field: str, value: object) -> str:
    body = {
        "scope": "captions" if field.startswith("captions.") else "cards",
        "sentence": "A private rule sentence that must never appear in a check.",
        "machine": {"kind": "style_field", "field": field, "value": value},
        "sources": [{
            "type": "comment", "media_sha256": "a" * 64, "comment_id": "c1", "label": "fix",
            "start_frame": None, "end_frame": None, "quote": None,
        }],
    }
    identity = add_proposals(body)[0].id
    approve(identity)
    return identity


@pytest.mark.parametrize("field,wanted,measured,expected", [
    ("captions.text_color", "#FFFFFF", "#FDFDFD", "pass"),
    ("captions.text_color", "#FFF4DD", "#101010", "fail"),
    ("captions.size_px", 80, 84, "pass"),
    ("captions.size_px", 80, 85, "fail"),
    ("hook.duration_s", 2.0, 2.04, "pass"),
    ("hook.duration_s", 2.0, 2.06, "fail"),
    ("captions.position", "bottom", "bottom", "pass"),
    ("captions.position", "bottom", "middle", "fail"),
    ("captions.italic_stress", True, False, "fail"),
])
def test_rule_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    field: str, wanted: object, measured: object, expected: str,
) -> None:
    identity = _approve(field, wanted)
    media = tmp_path / "final.mp4"
    measure = Mock(return_value={field: {"value": measured, "frames": [15, 39]}})
    timing = Mock(return_value={15: 0.57})
    monkeypatch.setattr(importlib.import_module("yap_studio.taste.check"), "_first_times", timing)
    result = check_taste(media, render_style().pin(), measure=measure)
    measure.assert_called_once_with(media, fields=(field,))
    assert result.coverage == {"ran": True, "reason": None}
    assert result.rules == [{
        "id": identity, "field": field, "wanted": wanted, "measured": measured,
        "status": expected, "reason": None, "frames": [15, 39],
    }]
    if expected == "pass":
        assert result.findings == []
    else:
        finding, = result.findings
        assert finding.check == "taste.rules" and finding.severity == "fail"
        assert finding.time_s == 0.57
        words = field.replace(".", " ").replace("_", " ").replace("color", "colour")
        for text in (identity, str(wanted), str(measured), words):
            assert text.lower() in finding.message.lower()
        assert "private rule sentence" not in finding.message
        assert finding.evidence == {
            "rule_id": identity, "field": field, "wanted": wanted, "measured": measured,
            "frames": [15, 39], "tolerance": tolerance(field),
        }
        assert finding.suggested_action == (
            "Render again with the approved style, or revoke the rule if it no longer holds."
        )
    assert json.loads(json.dumps(result.to_dict()))["rules"] == result.rules


@pytest.mark.parametrize("partial", [False, True])
def test_unmeasured_rules_are_named(tmp_path: Path, partial: bool) -> None:
    identity = _approve("captions.font", "Inter")
    measurements = {"captions.font": {"value": None, "reason": "Needs text recognition."}}
    if partial:
        _approve("captions.text_color", "#FFFFFF")
        measurements["captions.text_color"] = {"value": "#FFFFFF", "frames": [2]}
    measure = Mock(return_value=measurements)
    result = check_taste(tmp_path / "final.mp4", render_style().pin(), measure=measure)
    assert result.coverage == (
        {"ran": True, "reason": None, "note": "1 of 2 rules not measured"} if partial else
        {"ran": False, "reason": (
            "no applied rule could be measured from the picture: "
            "captions.font (Needs text recognition.)"
        )}
    )
    assert len(result.rules) == (2 if partial else 1)
    row = next(row for row in result.rules if row["id"] == identity)
    assert row == {
        "id": identity, "field": "captions.font", "wanted": "Inter", "measured": None,
        "status": "not_measured", "reason": "Needs text recognition.", "frames": [],
    }
    finding, = result.findings
    assert finding.severity == "info"
    assert identity in finding.message and "captions font" in finding.message
    assert "Needs text recognition." in finding.message
    assert "private rule sentence" not in finding.message


def test_unmeasured_coverage_joins_each_rule_reason(tmp_path: Path) -> None:
    _approve("captions.font", "Inter")
    _approve("captions.max_words_per_phrase", 3)
    result = check_taste(tmp_path / "final.mp4", render_style().pin(), measure=Mock(return_value={
        "captions.font": {"value": None, "reason": "Fonts need text recognition."},
        "captions.max_words_per_phrase": {
            "value": None, "reason": "Words per caption need text recognition.",
        },
    }))
    assert result.coverage == {
        "ran": False,
        "reason": "no applied rule could be measured from the picture: " + "; ".join(
            f"{row['field']} ({row['reason']})" for row in result.rules
        ),
    }


def test_unmeasured_coverage_counts_omitted_rule_reasons(tmp_path: Path) -> None:
    fields = {
        "captions.font": "Inter", "captions.max_words_per_phrase": 3,
        "captions.width_scale": 1, "captions.entry_ms": 300,
        "captions.shadow_opacity": 0.5,
    }
    for field, value in fields.items():
        _approve(field, value)
    reason = "This field needs text recognition before it can be measured."
    result = check_taste(tmp_path / "final.mp4", render_style().pin(), measure=Mock(return_value={
        field: {"value": None, "reason": reason} for field in fields
    }))
    summary = result.coverage["reason"]
    assert result.coverage["ran"] is False
    assert len(summary) <= 300
    shown = [row for row in result.rules if f"{row['field']} ({reason})" in summary]
    assert 0 < len(shown) < len(fields)
    assert summary.endswith(f"; and {len(fields) - len(shown)} more")
    assert len(result.rules) == len(result.findings) == len(fields)
    assert all(row["reason"] == reason for row in result.rules)


@pytest.mark.parametrize("extra_rule", [False, True])
def test_unmeasured_coverage_bounds_a_long_first_reason(tmp_path: Path, extra_rule: bool) -> None:
    _approve("captions.font", "Inter")
    if extra_rule:
        _approve("captions.max_words_per_phrase", 3)
    record = render_style().pin()
    reason = "Needs text recognition. " * 40
    result = check_taste(tmp_path / "final.mp4", record, measure=Mock(return_value={
        row["field"]: {"value": None, "reason": reason} for row in record["applied"]
    }))
    summary = result.coverage["reason"]
    assert len(summary) <= 300
    assert f"{result.rules[0]['field']} (Needs text recognition." in summary
    assert summary.endswith("...; and 1 more" if extra_rule else "...")
    assert all(row["reason"] == reason for row in result.rules)


def test_no_applied_rule_does_not_measure(tmp_path: Path) -> None:
    _approve("captions.text_color", "#FFFFFF")
    measure = Mock(side_effect=AssertionError("must not measure"))
    record = render_style(explicit={"captions.text_color": "#123456"}).pin()
    result = check_taste(tmp_path / "final.mp4", record, measure=measure)
    assert result.coverage == {
        "ran": False, "reason": "no applied rule could be measured from the picture: "
        "captions.text_color (project_override)",
    }
    assert result.rules == [{"id": record["skipped"][0]["id"], "field": "captions.text_color",
                             "status": "skipped", "reason": "project_override"}]
    assert result.findings == []
    measure.assert_not_called()


@pytest.mark.parametrize("mismatch", ["newer", "hash", "migrated"])
def test_history_must_match_before_measuring(tmp_path: Path, mismatch: str) -> None:
    identity = _approve("captions.text_color", "#FFFFFF")
    record = render_style().pin()
    private = load_taste()
    if mismatch == "newer":
        private = Taste()
    elif mismatch == "hash":
        record["sha256"] = "0" * 64
    else:
        revoke(identity)
        _approve("captions.text_color", "#FFFFFF")
        legacy = load_taste().to_json()
        del legacy["history"]
        private = validate_taste(legacy)
    measure = Mock(side_effect=AssertionError("must not measure"))
    result = check_taste(tmp_path / "final.mp4", record, taste=private, measure=measure)
    reason = "this computer's taste history does not match the pinned revision 1"
    assert result.coverage == {"ran": False, "reason": reason}
    finding, = result.findings
    assert finding.severity == "info" and finding.message == reason
    assert result.rules == [{
        "id": identity, "field": "captions.text_color", "wanted": None, "measured": None,
        "status": "not_measured", "reason": reason, "frames": [],
    }]
    measure.assert_not_called()


def test_pinned_rules_survive_later_decisions(tmp_path: Path) -> None:
    identity = _approve("captions.text_color", "#FFF4DD")
    record = render_style().pin()
    _approve("captions.text_color", "#123456")
    result = check_taste(
        tmp_path / "final.mp4", record,
        measure=Mock(return_value={"captions.text_color": {"value": "#FFF4DD", "frames": [1]}}),
    )
    assert result.findings == []
    assert result.rules[0]["id"] == identity
    assert result.rules[0]["wanted"] == "#FFF4DD"


@pytest.mark.parametrize("receipt_style", ["same", "different", "absent"])
def test_receipt_style_matches_record(tmp_path: Path, receipt_style: str) -> None:
    _approve("captions.text_color", "#FFFFFF")
    media = tmp_path / "final.mp4"
    record = render_style().pin()
    data = {} if receipt_style == "absent" else {
        "style": {"sha256": record["style_sha256"] if receipt_style == "same" else "0" * 64},
    }
    media.with_suffix(".receipt.json").write_text(json.dumps(data), encoding="utf-8")
    result = check_taste(media, record, measure=Mock(return_value={
        "captions.text_color": {"value": "#FFFFFF", "frames": [1]},
    }))
    assert result.rules[0]["status"] == "pass"
    if receipt_style == "different":
        finding, = result.findings
        assert finding.severity == "fail"
        assert finding.message == (
            "The render's receipt names a different style than its taste record."
        )
    else:
        assert result.findings == []


def test_missing_measurement_and_unknown_applied_rule_are_explained(tmp_path: Path) -> None:
    _approve("captions.text_color", "#FFFFFF")
    record = render_style().pin()
    record["applied"].append({"id": "r-000000000000", "field": "captions.position"})
    result = check_taste(tmp_path / "final.mp4", record, measure=Mock(return_value={}))
    assert [row["status"] for row in result.rules] == ["not_measured", "not_measured"]
    assert result.rules[0]["reason"] == "Not measured from pixels."
    assert "not in the pinned taste revision" in result.rules[1]["reason"]
    assert len(result.findings) == 2


def test_measurement_error_explains_unchecked_rules(tmp_path: Path) -> None:
    identity = _approve("captions.text_color", "#FFFFFF")
    measure = Mock(side_effect=InputError("final.mp4", "no usable video stream"))
    result = check_taste(tmp_path / "final.mp4", render_style().pin(), measure=measure)
    assert result.coverage == {
        "ran": False, "reason": (
            "no applied rule could be measured from the picture: "
            "captions.text_color (no usable video stream)"
        ),
    }
    assert result.rules[0]["reason"] == "no usable video stream"
    assert identity in result.findings[0].message
    assert result.findings[0].severity == "info"


@pytest.fixture
def vertical_clip(tmp_path: Path) -> Path:
    return taste_finals.build_source(
        tmp_path / "clip.mov", style=resolve("preset:electric"), seconds=2, size=(180, 320),
    )


def test_measure_style_is_read_only_and_uses_only_picture(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("yap_studio.taste.learn")
    refusal = Mock(side_effect=AssertionError("must only measure picture"))
    monkeypatch.setattr(module, "resolve_backend", refusal)
    monkeypatch.setattr(module.measure_pacing, "measure", refusal)
    monkeypatch.setattr(module.paths, "atomic_write_text", refusal)
    before = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    fields = [
        "captions.font", *LIMIT_FIELDS, "captions.width_scale",
        "captions.shadow_opacity", "captions.entry_ms", "graphics.paper_color",
    ]
    result = measure_style(vertical_clip, fields)
    assert set(result) == set(fields)
    reasons = {row["field"]: row["reason"] for row in UNMEASURED}
    reasons.update(dict.fromkeys(LIMIT_FIELDS, NEEDS_WORDS))
    reasons["captions.font"] = (
        "No measured caption band and text colour to measure font class. "
        "Naming the exact typeface needs text recognition (OCR)."
    )
    for field in fields[:-1]:
        assert result[field] == {
            "value": None, "reason": reasons.get(field, "Not measured from pixels."),
        }
    assert result["graphics.paper_color"]["reason"] == "No card was found."
    assert {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()} == before
    refusal.assert_not_called()


def test_measure_style_matches_learning_on_source(vertical_clip: Path, tmp_path: Path) -> None:
    learned = learn([vertical_clip], tmp_path / "learned", asr="none").measurements
    measured = measure_style(vertical_clip)
    for field, row in learned["fields"].items():
        assert measured[field]["value"] == row["value"]
        assert measured[field]["frames"] == row["per_video"][0]["frames"]
    for row in learned["not_measured"]:
        if row["field"] in measured and measured[row["field"]]["value"] is None:
            assert measured[row["field"]]["reason"] == row["reason"]


def test_short_caption_highlight_reason_matches_learning_and_check(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import numpy as np

    from yap_studio.taste.measure_text import TextSample

    module = importlib.import_module("yap_studio.taste.learn")
    info = replace(ffmpeg.probe(vertical_clip), video_frames=50, duration_s=50 / 30)
    monkeypatch.setattr(module.ffmpeg, "probe", Mock(return_value=info))
    monkeypatch.setattr(module, "iter_rgb", lambda path, info, indices, **kwargs: (
        (index, np.full((2, 2, 3), index, dtype=np.uint8)) for index in indices
    ))
    monkeypatch.setattr(module.CaptionDiscovery, "add", lambda *args: None)
    monkeypatch.setattr(module.CaptionDiscovery, "band", lambda *args: (400, 550))
    monkeypatch.setattr(module, "caption_band", lambda image, *args: TextSample(
        color="#FFF4DD", highlight="underline" if image[0, 0, 0] == 15 else "none",
        highlight_color="#D2FF2D", highlight_text_color=None,
        top=508, bottom=524, pitch=None, line_top=508, glyph_height=16,
    ))
    monkeypatch.setattr(module, "find_panels", lambda *args, **kwargs: [])
    monkeypatch.setattr(module.measure_shapes, "measure", Mock(return_value=({}, [])))
    monkeypatch.setattr(module.measure_pacing, "measure", Mock(return_value={}))
    reason = "Too few caption samples to separate a highlight from unstressed phrases."
    field = "captions.highlight_style"
    learned = learn([vertical_clip], tmp_path / "learned", asr="none").measurements
    assert field not in learned["fields"]
    assert any(row["field"] == field and row["reason"] == reason
               for row in learned["not_measured"])
    assert learned["proposed_style"]["sources"][field]["source"] == "base"
    assert learned["proposed_style"]["sources"][field]["reason"] == reason
    measured = measure_style(vertical_clip, fields=(field,))
    assert measured[field] == {"value": None, "reason": reason}
    identity = _approve(field, "underline")
    write_taste_record(vertical_clip, render_style())
    checked = check_taste(vertical_clip, render_style().pin())
    assert checked.rules == [{
        "id": identity, "field": field, "wanted": "underline", "measured": None,
        "status": "not_measured", "reason": reason, "frames": [],
    }]
    assert cli.main(["taste", "check", str(vertical_clip), "--json"]) == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported["rules"] == checked.rules


@pytest.mark.parametrize("fields", [None, ("captions.font",)])
@pytest.mark.parametrize("font_class,expected_reason", [
    ({"family": "serif", "weight": "regular"}, "Measured font class: serif, regular weight."),
    (None, "Too few caption glyph pixels."),
])
def test_measure_style_font_reason_matches_learning(
    vertical_clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    fields: tuple[str, ...] | None, font_class: dict[str, str] | None, expected_reason: str,
) -> None:
    module = importlib.import_module("yap_studio.taste.learn")
    monkeypatch.setattr(module, "_visual", lambda path, info: (
        {"captions.text_color": ("#FFFFFF", [2], {})}, {}, [],
    ))
    monkeypatch.setattr(module.measure_pacing, "measure", Mock(return_value={}))
    observations = {"graphics.corner_radius_px": (24, [3], {})}
    missing = []
    if font_class is None:
        missing.append({"field": "captions.font_class", "reason": expected_reason})
    else:
        observations["captions.font_class"] = font_class, [2, 5], {}
    shapes = Mock(return_value=(observations, missing))
    monkeypatch.setattr(module.measure_shapes, "measure", shapes)
    learned = learn([vertical_clip], tmp_path / "learned", asr="none").measurements
    measured = measure_style(vertical_clip, fields)
    reason = expected_reason + " Naming the exact typeface needs text recognition (OCR)."
    assert measured["captions.font"] == {"value": None, "reason": reason}
    assert next(
        row["reason"] for row in learned["not_measured"] if row["field"] == "captions.font"
    ) == reason
    assert shapes.call_count == 2
    assert "captions.font_class" not in measured
    assert "graphics.corner_radius_px" not in measured
    if fields is not None:
        assert set(measured) == set(fields)


@pytest.mark.parametrize("fields", [(), ("captions.text_color",), ("hook.font",)])
def test_measure_style_skips_shapes_without_caption_font(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch, fields: tuple[str, ...],
) -> None:
    module = importlib.import_module("yap_studio.taste.learn")
    shapes = Mock(side_effect=AssertionError("caption font was not requested"))
    monkeypatch.setattr(module.measure_shapes, "measure", shapes)
    assert set(measure_style(vertical_clip, fields)) == set(fields)
    shapes.assert_not_called()


def _assert_same_measurements(final: Path, out: Path) -> None:
    learned = learn([final], out, asr="none").measurements
    fields = list(learned["fields"])
    assert "captions.text_color" in fields
    measured = measure_style(final, fields)
    assert set(measured) == set(fields)
    for field, row in learned["fields"].items():
        assert measured[field] == {"value": row["value"], "frames": row["per_video"][0]["frames"]}


@pytest.mark.parametrize("preset", SHIPPED_PRESETS)
def test_measure_style_matches_learning_on_dev_finals(preset: str, tmp_path: Path) -> None:
    directory = os.environ.get("YAP_STUDIO_TASTE_FINALS")
    if not directory:
        pytest.skip("Set YAP_STUDIO_TASTE_FINALS to pre-rendered finished videos.")
    _assert_same_measurements(Path(directory) / preset / "final.mp4", tmp_path / "learned")


@pytest.mark.render
def test_measure_style_matches_learning_on_rendered_final(tmp_path: Path) -> None:
    truth = taste_finals.build_final("electric", tmp_path / "render")
    _assert_same_measurements(truth.final, tmp_path / "learned")


def test_failure_time_uses_original_presentation_timestamp(
    vertical_clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("yap_studio.taste.check")
    from yap_studio.taste.frames import presentation_timing

    timing = presentation_timing(vertical_clip, ffmpeg.probe(vertical_clip))
    starts = list(timing.starts)
    starts[15] = 0.6
    monkeypatch.setattr(module, "presentation_timing", lambda *args: replace(
        timing, starts=tuple(starts), variable_rate=True,
    ))
    _approve("captions.text_color", "#FFFFFF")
    result = check_taste(vertical_clip, render_style().pin(), measure=Mock(return_value={
        "captions.text_color": {"value": "#111111", "frames": [15]},
    }))
    assert result.findings[0].time_s == 0.6


@pytest.mark.render
def test_verify_fails_named_rule_on_the_same_finished_render(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    first = _approve("captions.text_color", "#FFF4DD")
    source = synth.calib_source(tmp_path / "source.mov", seconds=4, size="180x320")
    words = tmp_path / "words.json"
    Transcript([
        Word("First", 0.1, 0.6), Word("words.", 0.6, 1.8),
        Word("Next", 2.0, 2.6), Word("phrase.", 2.6, 3.9),
    ]).save(words)
    style = tmp_path / "style.json"
    style.write_text(json.dumps({
        "schema_version": 1, "captions": {"highlight_style": "none", "shadow_opacity": 0},
        "hook": {"style": "none"},
    }), encoding="utf-8")
    rendered = tmp_path / "rendered.mp4"
    assert cli.main([
        "captions", str(source), "--words", str(words), "--out", str(rendered),
        "--style", str(style), "--json",
    ]) == 0
    capsys.readouterr()
    assert read_taste_record(rendered)["applied"] == [{"id": first, "field": "captions.text_color"}]
    original = rendered.read_bytes()
    # Keep the publisher's receipt and taste intact while checking the same finished picture.
    final = tmp_path / "final.mp4"
    final.write_bytes(original)
    write_taste_record(final, render_style(style))
    passed = verify(str(final), asr="none")
    assert passed["checks"]["taste.rules"] == {"ran": True, "status": "pass", "reason": None}
    assert not [row for row in passed["findings"] if row["check"] == "taste.rules"]
    second = _approve("captions.text_color", "#FF0000")
    write_taste_record(final, render_style(style))
    failed = verify(str(final), asr="none")
    assert failed["checks"]["taste.rules"] == {"ran": True, "status": "fail", "reason": None}
    broken = next(row for row in failed["findings"] if row["evidence"].get("rule_id") == second)
    assert broken["severity"] == "fail"
    assert second in broken["message"] and "Captions text colour" in broken["message"]
    assert broken["evidence"]["wanted"] == "#FF0000"
    assert broken["evidence"]["measured"] is not None
    assert final.read_bytes() == original
    assert rendered.read_bytes() == original
    assert read_taste_record(rendered)["applied"] == [{"id": first, "field": "captions.text_color"}]
    final.with_suffix(".taste.json").unlink()
    absent = verify(str(final), asr="none")
    assert absent["checks"]["taste.rules"] == {
        "ran": False, "status": "skipped", "reason": "no taste record next to this render",
    }
