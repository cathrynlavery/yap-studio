"""Caption record validation, timing, glyph overlap, and verify coverage."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import synth

from yap_studio import ffmpeg
from yap_studio.captions import CaptionWord, build_phrases
from yap_studio.checks.caption_layout import (
    LAYOUT_SUFFIX,
    caption_coverage,
    check_caption_cards,
    check_caption_timing,
    read_layout_record,
    write_layout_record,
)
from yap_studio.checks.model import Thresholds
from yap_studio.errors import InputError
from yap_studio.faces import detect
from yap_studio.receipt import sha256_file
from yap_studio.stages.captions import caption_frame_window
from yap_studio.verify import verify


@pytest.fixture(autouse=True)
def portable_face_detector(monkeypatch):
    """Keep layout verification independent of the system detector."""
    monkeypatch.setattr(detect, "unavailable_reason",
                        lambda: "System detector unavailable in tests")

    def unexpected_detector():
        pytest.fail("Layout tests must not invoke the system face detector")

    monkeypatch.setattr(detect, "default_detector", unexpected_detector)


@pytest.fixture
def media(tmp_path: Path) -> Path:
    output = tmp_path / "final.mp4"
    output.write_bytes(b"render bytes for layout record validation")
    return output


@pytest.fixture
def document() -> dict:
    return synth.calibration_layout()


def _read(media: Path, document: dict):
    write_layout_record(media, document)
    record = read_layout_record(media, width=320, height=180)
    assert record is not None
    return record


def _timing(record, *, duration_s: float = 19.2):
    return check_caption_timing(record, fps=30, duration_s=duration_s, thresholds=Thresholds())


@pytest.mark.parametrize("start,end,expected", [(0, 30, []), (30, 60, [(30, 45)])])
def test_cards_use_caption_box_windows(media, start, end, expected) -> None:
    document = {"schema": "yap-studio/clearance-layout@1", "width": 320, "height": 180,
                "measured": True, "caption_gap_px": 10, "elements": [
                    {"id": "card", "kind": "card", "x": 10, "y": 20, "w": 80, "h": 40,
                     "start_frame": start, "end_frame": end}], "captions": [
                    {"id": "caption", "text": "Words", "start_frame": 0, "end_frame": 60,
                     "words": [{"text": "Words", "start_s": .1, "end_s": 1.8}], "boxes": [
                         {"x": 10, "y": 20, "w": 80, "h": 40,
                          "start_frame": 30, "end_frame": 45}]}]}
    findings = check_caption_cards(_read(media, document), fps=30)
    assert [(f.evidence["start_frame"], f.evidence["end_frame"]) for f in findings] == expected


def test_write_layout_record_round_trip(media, document, tmp_path):
    document["media_sha256"] = "stale"
    document["captions"][0]["text"] = "Caf\u00e9 time"
    custom = tmp_path / "staged.layout.json"
    assert write_layout_record(media, document, custom) == custom
    assert json.loads(custom.read_text())["media_sha256"] == sha256_file(media)
    assert document["media_sha256"] == "stale"
    record = _read(media, document)
    assert media.with_suffix(LAYOUT_SUFFIX).name == "final.layout.json"
    assert record.measured is True and record.reason is None
    assert record.layout.captions[0].text == "Caf\u00e9 time"
    word = record.words["caption-1"][0]
    assert (word.text, word.start_s, word.end_s) == ("Make", 1.0, 1.4)
    assert isinstance(record.words["caption-1"], tuple)
    assert record.face_windows == {}
    assert record.element_ids == {"card-1": "card-1"}
    assert record.layout.caption_gap_px == 100


@pytest.mark.parametrize("element_id", ["intro~alt", "intro#alt", "card~2", "card#2"])
def test_read_element_id_preserves_original_name(media, document, element_id):
    document["elements"][0].update(id="drawn-piece", element_id=element_id)
    assert _read(media, document).element_ids == {"drawn-piece": element_id}


def test_overlay_label_is_preserved_in_caption_finding(media):
    document = synth.calibration_layout(card_over_caption=True)
    document["elements"][0].update(kind="overlay", id="piece", element_id="broll-photo",
                                    label="photo.png", x=100, w=30)
    record = _read(media, document)
    assert record.labels == {"piece": "photo.png"}
    (finding,) = check_caption_cards(record, fps=30)
    assert finding.message == (
        'Overlay "broll-photo" (photo.png) covers the caption '
        '"Keep moving forward" at 0:07.2.'
    )
    assert finding.evidence["label"] == "photo.png"
    assert finding.evidence["element_id"] == "broll-photo"
    assert finding.evidence["element_piece"] == "piece"


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("label", [None, "", True, 2, [], {}])
def test_read_invalid_label_is_corrupt(media, document, measured, label):
    document["measured"] = measured
    document["elements"][0]["label"] = label
    write_layout_record(media, document)
    with pytest.raises(InputError, match="corrupt layout record: .*label"):
        read_layout_record(media, width=320, height=180)


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("element_id", [None, "", True, 2, [], {}])
def test_read_invalid_element_id_is_corrupt(media, document, measured, element_id):
    document["measured"] = measured
    document["elements"][0]["element_id"] = element_id
    write_layout_record(media, document)
    with pytest.raises(InputError, match="corrupt layout record: .*element_id"):
        read_layout_record(media, width=320, height=180)


@pytest.mark.parametrize("windows", [[], [[216, 270]], [[216, 217], [269, 270]]])
def test_read_face_windows_preserves_full_element_window(media, document, windows):
    document["elements"][0]["face_windows"] = windows
    record = _read(media, document)
    assert record.face_windows == {"card-1": tuple(tuple(window) for window in windows)}
    element = record.layout.elements[0]
    assert (element.start_frame, element.end_frame) == (216, 270)


@pytest.mark.parametrize("measured", [False, True])
@pytest.mark.parametrize("windows", [
    None, {}, "216,270", [216, 270], [None], [[]], [[216]], [[216, 217, 218]],
    [[True, 217]], [[216, False]], [[216.0, 217]], [[216, 217.0]],
    [["216", 217]], [[216, None]], [[217, 217]], [[218, 217]],
    [[215, 217]], [[269, 271]], [[216, 217], [269, 271]],
])
def test_read_invalid_face_windows(media, document, measured, windows):
    document["measured"] = measured
    document["elements"][0]["face_windows"] = windows
    write_layout_record(media, document)
    with pytest.raises(InputError, match="corrupt layout record: .*face_windows"):
        read_layout_record(media, width=320, height=180)


def test_read_absent(media):
    assert read_layout_record(media, width=320, height=180) is None


def test_read_dangling_symlink(media):
    media.with_suffix(LAYOUT_SUFFIX).symlink_to(media.parent / "missing.json")
    with pytest.raises(InputError, match="layout record points to a missing file"):
        read_layout_record(media, width=320, height=180)


def test_read_unreadable(media, monkeypatch):
    def unreadable(*args, **kwargs):
        raise PermissionError("unreadable")

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(InputError, match="cannot read layout record"):
        read_layout_record(media, width=320, height=180)


@pytest.mark.parametrize("content", ["{", "[]", '{"schema": "wrong"}',
                                    '{"schema": "a", "schema": "b"}'])
def test_read_corrupt_json_or_schema(media, content):
    media.with_suffix(LAYOUT_SUFFIX).write_text(content)
    with pytest.raises(InputError, match="corrupt layout record"):
        read_layout_record(media, width=320, height=180)


@pytest.mark.parametrize("fault", ["invalid_layout", "missing_measured", "bad_measured",
                                   "bad_reason"])
def test_read_invalid_layout_or_measurement(media, document, fault):
    if fault == "invalid_layout":
        document["captions"][0]["end_frame"] = document["captions"][0]["start_frame"]
    elif fault == "missing_measured":
        del document["measured"]
    elif fault == "bad_measured":
        document["measured"] = 1
    else:
        document["reason"] = []
    write_layout_record(media, document)
    with pytest.raises(InputError, match="corrupt layout record"):
        read_layout_record(media, width=320, height=180)


@pytest.mark.parametrize("fault", ["changed_media", "missing_hash", "bad_hash"])
def test_read_hash_mismatch(media, document, fault):
    target = write_layout_record(media, document)
    if fault == "changed_media":
        media.write_bytes(b"different render")
    else:
        stored = json.loads(target.read_text())
        if fault == "missing_hash":
            del stored["media_sha256"]
        else:
            stored["media_sha256"] = 123
        target.write_text(json.dumps(stored))
    with pytest.raises(InputError, match="layout record media hash does not match"):
        read_layout_record(media, width=320, height=180)


@pytest.mark.parametrize("width,height", [(180, 320), (321, 180), (320, 181)])
def test_read_size_mismatch(media, document, width, height):
    write_layout_record(media, document)
    with pytest.raises(InputError, match="layout size does not match the render's display size"):
        read_layout_record(media, width=width, height=height)


def test_read_measured_false_without_words(media, document):
    document.update(measured=False, reason="Glyphs were unavailable.")
    for caption in document["captions"]:
        del caption["words"]
    record = _read(media, document)
    assert record.measured is False
    assert record.reason == "Glyphs were unavailable."
    assert record.words == {}


@pytest.mark.parametrize("words", [None, [], {}, "words"])
def test_read_missing_words(media, document, words):
    if words is None:
        del document["captions"][0]["words"]
    else:
        document["captions"][0]["words"] = words
    write_layout_record(media, document)
    with pytest.raises(InputError, match="needs a non-empty words list"):
        read_layout_record(media, width=320, height=180)


@pytest.mark.parametrize("word", [
    None, {"text": 1, "start_s": 1, "end_s": 2}, {"text": "word", "start_s": 1},
    {"text": "word", "start_s": True, "end_s": 2},
    {"text": "word", "start_s": "1", "end_s": 2},
    {"text": "word", "start_s": math.nan, "end_s": 2},
    {"text": "word", "start_s": 1, "end_s": math.inf},
    {"text": "word", "start_s": 2, "end_s": 1},
])
def test_read_invalid_word(media, document, word):
    document["captions"][0]["words"] = [word]
    # Write directly so malformed non-finite input reaches the reader.
    document["media_sha256"] = sha256_file(media)
    media.with_suffix(LAYOUT_SUFFIX).write_text(json.dumps(document))
    with pytest.raises(InputError, match="corrupt layout record"):
        read_layout_record(media, width=320, height=180)


@pytest.mark.parametrize("shift,direction,offset", [(9, "after", 333.3), (-9, "before", -266.7)])
def test_timing_late_or_early(media, shift, direction, offset):
    record = _read(media, synth.calibration_layout(late_phrase="caption-2", late_frames=shift))
    findings = _timing(record)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.check == "captions.timing" and finding.severity == "fail"
    assert finding.time_s == 7.2
    assert f'{direction} its first word at 0:07.2' in finding.message
    assert '"Keep moving forward"' in finding.message
    assert finding.evidence == {"caption_id": "caption-2", "offset_ms": offset,
                                "start_frame": 217 + shift, "end_frame": 252 + shift}
    assert "current cut words" in finding.suggested_action
    assert "caption layer is placed" in finding.suggested_action


def test_timing_leaves_before_word(media, document):
    document["captions"][1]["end_frame"] = 230
    findings = _timing(_read(media, document))
    assert len(findings) == 1
    assert 'leaves before the word "forward" is said at 0:08.0' in findings[0].message
    assert findings[0].time_s == 7.2
    assert findings[0].evidence["end_frame"] == 230


def test_timing_leaves_mid_word(media, document):
    document["captions"][1]["end_frame"] = 246
    findings = _timing(_read(media, document))
    assert len(findings) == 1
    assert findings[0].message == (
        'Caption "Keep moving forward" leaves before the word "forward" ends at 0:08.4.'
    )
    assert findings[0].time_s == 7.2
    assert findings[0].evidence["end_frame"] == 246


def test_timing_sentence_final_word_with_padded_end_passes(media, document):
    caption = document["captions"][1]
    caption["text"] = "Keep moving forward."
    caption["words"][-1].update(text="forward.", end_s=10.5)
    caption["end_frame"] = 282  # 1.4 s after the last word starts at 8.0 s.
    (finding,) = _timing(_read(media, document))
    assert finding.check == "captions.timing" and finding.severity == "warn"
    assert finding.message == (
        'Caption "Keep moving forward." leaves at 0:09.4; '
        'the transcript has "forward." running to 0:10.5.'
    )
    assert finding.evidence["transcript_end_s"] == 10.5


def test_timing_long_word_held_to_its_end_has_no_finding(media, document):
    caption = document["captions"][1]
    caption["words"][-1]["end_s"] = 9.4
    caption["end_frame"] = 282
    assert _timing(_read(media, document)) == []


def test_timing_leaves_half_second_into_one_second_word(media, document):
    caption = document["captions"][1]
    caption["words"][-1]["end_s"] = 9.0
    caption["end_frame"] = 255  # Leaves at 8.5 s, halfway through "forward".
    (finding,) = _timing(_read(media, document))
    assert finding.message == (
        'Caption "Keep moving forward" leaves before the word "forward" ends at 0:09.0.'
    )
    assert finding.check == "captions.timing" and finding.severity == "fail"
    assert finding.time_s == 7.2
    assert finding.evidence["end_frame"] == 255


@pytest.mark.parametrize("end_frame", [261, 259])
@pytest.mark.parametrize("reverse", [False, True])
def test_timing_next_phrase_can_end_previous_word_hold(media, document, end_frame, reverse):
    caption = document["captions"][1]
    caption["words"][-1]["end_s"] = 10.5
    caption["end_frame"] = end_frame  # The next phrase starts at 8.7 s.
    following = document["captions"][2]
    following.update(start_frame=262, end_frame=285)
    for word, start in zip(following["words"], (8.7, 9.1), strict=True):
        word.update(start_s=start, end_s=start + 0.4)
    if reverse:
        document["captions"].reverse()
    assert all(finding.severity == "warn" for finding in _timing(_read(media, document)))


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_timing_builder_phrases_with_padded_ends_pass(media, document, fps):
    for padding_s in (1.6, 2.5):
        for spacing_s in (0.25, 0.4, 0.8, 2.0):
            for max_words in (1, 2, 4):
                words = [
                    CaptionWord(str(index), text, 1.0 + index * spacing_s,
                                1.0 + index * spacing_s + padding_s)
                    for index, text in enumerate(("Keep", "moving", "forward.",
                                                  "Stay", "curious."))
                ]
                phrases = build_phrases(words, {
                    "max_words_per_phrase": max_words, "max_characters_per_phrase": 80,
                })
                assert phrases
                document["captions"] = []
                for phrase in phrases:
                    start_frame, end_frame = caption_frame_window(
                        phrase.start, phrase.end, fps=fps, layer_start=0,
                    )
                    document["captions"].append({
                        "id": f"caption-{phrase.index}", "text": phrase.text,
                        "start_frame": start_frame, "end_frame": end_frame,
                        "boxes": [{"x": 30, "y": 145, "w": 60, "h": 20}],
                        "words": [{"text": word.text, "start_s": word.start, "end_s": word.end}
                                  for word in phrase.words],
                    })
                findings = check_caption_timing(
                    _read(media, document), fps=fps, duration_s=19.2, thresholds=Thresholds(),
                )
                failures = [finding for finding in findings if finding.severity == "fail"]
                assert failures == [], (padding_s, spacing_s, max_words, failures)


def test_timing_leaves_mid_overlapping_word(media, document):
    document["captions"][1]["words"][1]["end_s"] = 8.8
    findings = _timing(_read(media, document))
    assert len(findings) == 1
    assert findings[0].check == "captions.timing" and findings[0].severity == "fail"
    assert findings[0].message == (
        'Caption "Keep moving forward" leaves before the word "moving" ends at 0:08.8.'
    )


@pytest.mark.parametrize(("word_end", "fails"), [(8.433, False), (8.434, True)])
def test_timing_word_end_limit(media, document, word_end, fails):
    document["captions"][1]["end_frame"] = 250
    document["captions"][1]["words"][-1]["end_s"] = word_end
    findings = _timing(_read(media, document))
    assert len(findings) == int(fails)
    if fails:
        assert 'word "forward" ends at 0:08.4' in findings[0].message


@pytest.mark.parametrize(("duration_s", "fails"), [(8.35, False), (8.5, True)])
def test_timing_last_word_end_is_clipped_to_video_end(media, document, duration_s, fails):
    document["captions"] = document["captions"][:2]
    document["captions"][1]["end_frame"] = 249
    document["captions"][1]["words"][-1]["end_s"] = 20.0
    findings = _timing(_read(media, document), duration_s=duration_s)
    assert len(findings) == int(fails)
    if fails:
        assert 'word "forward" ends at 0:08.5' in findings[0].message


def test_timing_calibration_windows_cover_last_word_ends(media, document):
    for caption in document["captions"]:
        end = caption["words"][-1]["end_s"]
        assert caption["end_frame"] == math.ceil(round(end * 30, 6))
    assert _timing(_read(media, document)) == []


@pytest.mark.parametrize("frame,word_start", [(33, 1.001), (27, 0.999)])
def test_timing_just_inside_limit(media, document, frame, word_start):
    document["captions"][0]["start_frame"] = frame
    document["captions"][0]["words"][0]["start_s"] = word_start
    assert _timing(_read(media, document)) == []


@pytest.mark.parametrize("word_start", [19.2, 20.0])
def test_timing_words_at_or_after_video_end_ignored(media, document, word_start):
    document["captions"][0]["words"].append({
        "text": "later", "start_s": word_start, "end_s": word_start + 0.4,
    })
    assert _timing(_read(media, document)) == []


def test_timing_one_finding_per_phrase(media):
    document = synth.calibration_layout(late_phrase="caption-2", late_frames=9)
    document["captions"][1]["end_frame"] = 246
    findings = _timing(_read(media, document))
    assert len(findings) == 1
    assert "after its first word" in findings[0].message


@pytest.mark.parametrize("kind", ["card", "title", "overlay"])
@pytest.mark.parametrize("face_windows", [None, [], [[260, 270]]])
def test_cards_overlap_and_kind_wording(media, kind, face_windows):
    document = synth.calibration_layout(card_over_caption=True)
    document["elements"][0].update(kind=kind, x=100, w=30)
    if face_windows is not None:
        document["elements"][0]["face_windows"] = face_windows
    record = _read(media, document)
    findings = check_caption_cards(record, fps=30)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.check == "captions.cards" and finding.severity == "fail"
    assert finding.time_s == 217 / 30
    assert finding.message == (
        f'{kind.capitalize()} "card-1" covers the caption "Keep moving forward" at 0:07.2.'
    )
    assert finding.evidence == {
        "element_id": "card-1", "element_piece": "card-1", "element_kind": kind,
        "caption_id": "caption-2",
        "start_frame": 217, "end_frame": 252,
        "element_box": {"x": 100, "y": 140, "w": 30, "h": 30},
        "glyph_box": {"x": 95, "y": 145, "w": 60, "h": 20},
        "gap_px": -25, "required_px": 100,
    }


@pytest.mark.parametrize("explicit_id", [False, True])
@pytest.mark.parametrize("element_id", ["intro~alt", "intro#alt", "card~2", "card#2"])
def test_cards_preserve_element_name_and_piece_evidence(media, explicit_id, element_id):
    document = synth.calibration_layout(card_over_caption=True)
    piece_id = "drawn-piece" if explicit_id else element_id
    document["elements"][0]["id"] = piece_id
    if explicit_id:
        document["elements"][0]["element_id"] = element_id
    (finding,) = check_caption_cards(_read(media, document), fps=30)
    assert f'Card "{element_id}" covers the caption' in finding.message
    assert finding.evidence["element_id"] == element_id
    assert finding.evidence["element_piece"] == piece_id


def _near_glyph_card(document, *, gap: int) -> None:
    document["elements"][0].update(x=155, y=145 - gap - 20, w=30, h=20)


def test_issue_15_card_20px_above_glyphs_fails_naming_phrase_and_card(media, document):
    _near_glyph_card(document, gap=20)
    (finding,) = check_caption_cards(_read(media, document), fps=30)
    assert finding.check == "captions.cards" and finding.severity == "fail"
    assert finding.time_s == 217 / 30
    assert finding.message == (
        'Card "card-1" is 20 px from the caption "Keep moving forward" '
        'at 0:07.2; keep at least 100 px.'
    )
    assert finding.evidence["gap_px"] == 20
    assert finding.evidence["required_px"] == 100
    assert finding.evidence["glyph_box"] == {"x": 95, "y": 145, "w": 60, "h": 20}


def test_issue_15_card_120px_clear_passes(media, document):
    _near_glyph_card(document, gap=120)
    assert check_caption_cards(_read(media, document), fps=30) == []


@pytest.mark.parametrize("required,expected", [(10, 0), (100, 1)])
def test_issue_15_gap_comes_from_layout_record(media, document, required, expected):
    _near_glyph_card(document, gap=20)
    document["caption_gap_px"] = required
    assert len(check_caption_cards(_read(media, document), fps=30)) == expected


def test_issue_15_background_within_gap_is_ignored(media, document):
    _near_glyph_card(document, gap=20)
    document["elements"][0]["kind"] = "background"
    assert check_caption_cards(_read(media, document), fps=30) == []


@pytest.mark.parametrize("change", [
    {"x": 40, "w": 0}, {"y": 150, "h": 0},
    {"end_frame": 217}, {"start_frame": 252},
])
def test_cards_empty_or_no_shared_frames_pass(media, change):
    document = synth.calibration_layout(card_over_caption=True)
    document["elements"][0].update(change)
    assert check_caption_cards(_read(media, document), fps=30) == []


@pytest.mark.parametrize("required,expected", [(0, 0), (100, 1)])
def test_cards_touching_uses_layout_gap(media, document, required, expected):
    document["elements"][0].update(x=100, y=125, w=30, h=20)
    document["caption_gap_px"] = required
    assert len(check_caption_cards(_read(media, document), fps=30)) == expected


@pytest.mark.parametrize("state,timing_reason,cards_reason", [
    ("measured", None, None),
    ("unmeasured", "caption layout was not measured: Glyphs unavailable.",
     "caption layout was not measured: Glyphs unavailable."),
    ("no_captions", "no captions in the layout", "no captions in the layout"),
    ("no_elements", None, "no cards in the layout"),
    ("background_only", None, "no cards in the layout"),
])
def test_coverage_reasons(media, document, state, timing_reason, cards_reason):
    if state == "unmeasured":
        document.update(measured=False, reason="Glyphs unavailable.")
    elif state == "no_captions":
        document["captions"] = []
    elif state == "no_elements":
        document["elements"] = []
    elif state == "background_only":
        # Captions draw above backgrounds, so the card check has nothing to inspect.
        document["elements"] = [{**document["elements"][0], "kind": "background"}]
    coverage = caption_coverage(_read(media, document))
    assert coverage == {
        "captions.timing": {"ran": timing_reason is None, "reason": timing_reason},
        "captions.cards": {"ran": cards_reason is None, "reason": cards_reason},
    }


@pytest.mark.parametrize("with_edl", [False, True])
@pytest.mark.parametrize("state", ["absent", "corrupt", "measured", "unmeasured"])
def test_verify_layout_record_coverage(synth_media, tmp_path, with_edl, state):
    media = tmp_path / "final.mp4"
    shutil.copyfile(synth_media.path("clean"), media)
    if state == "corrupt":
        media.with_suffix(LAYOUT_SUFFIX).write_text("{")
    elif state in {"measured", "unmeasured"}:
        document = synth.calibration_layout()
        if state == "unmeasured":
            document.update(measured=False, reason="Glyphs unavailable.")
        write_layout_record(media, document)
    options = {}
    if with_edl:
        edl = tmp_path / "ranges.json"
        edl.write_text(json.dumps({
            "sources": {"main": str(synth_media.path("calib"))},
            "ranges": [{"source": "main", "start": start, "end": end}
                       for start, end in synth_media.spans],
        }))
        options["edl"] = str(edl)
    report = verify(str(media), asr="none", loudness="note", **options)
    layout_path = media.with_suffix(LAYOUT_SUFFIX)
    inputs = {row["path"]: row for row in report["receipt"]["inputs"]}
    if state == "absent":
        assert str(layout_path) not in inputs
    else:
        assert inputs[str(layout_path)]["sha256"] == sha256_file(layout_path)
        assert inputs[str(layout_path)]["bytes"] == layout_path.stat().st_size
    captions = {"captions.timing", "captions.cards"}
    findings = [row for row in report["findings"] if row["check"] in captions]
    if state == "corrupt":
        assert {row["check"] for row in findings} == captions
        assert len(findings) == 2
        assert all(row["severity"] == "warn" for row in findings)
    else:
        assert findings == []
    for check in captions:
        coverage = report["checks"][check]
        if state == "absent":
            assert coverage == {
                "ran": False, "status": "skipped",
                "reason": "no layout record next to this render",
            }
        elif state == "corrupt":
            assert coverage == {"ran": True, "status": "warn", "reason": None}
            assert next(row for row in findings if row["check"] == check)["message"].startswith(
                "corrupt layout record:"
            )
        elif state == "unmeasured":
            assert coverage == {
                "ran": False, "status": "skipped",
                "reason": "caption layout was not measured: Glyphs unavailable.",
            }
        else:
            assert coverage == {"ran": True, "status": "pass", "reason": None}


@pytest.mark.parametrize("gap,expected_status", [(20, "fail"), (120, "pass")])
def test_issue_15_verify_checks_caption_gap(synth_media, tmp_path, gap, expected_status):
    media = tmp_path / "final.mp4"
    shutil.copyfile(synth_media.path("clean"), media)
    document = synth.calibration_layout()
    _near_glyph_card(document, gap=gap)
    write_layout_record(media, document)
    report = verify(str(media), asr="none", loudness="note")
    assert report["checks"]["captions.cards"] == {
        "ran": True, "status": expected_status, "reason": None,
    }
    findings = [row for row in report["findings"] if row["check"] == "captions.cards"]
    if gap == 20:
        assert len(findings) == 1
        assert 'Card "card-1"' in findings[0]["message"]
        assert '"Keep moving forward"' in findings[0]["message"]
        assert findings[0]["evidence"]["gap_px"] == 20
    else:
        assert findings == []


@pytest.mark.parametrize("state", ["measured", "unmeasured", "no_captions"])
def test_verify_vfr_skips_both_caption_checks(synth_media, tmp_path, monkeypatch, state):
    import importlib

    verify_module = importlib.import_module("yap_studio.verify")
    media = tmp_path / "final.mp4"
    shutil.copyfile(synth_media.path("clean"), media)
    document = synth.calibration_layout(late_phrase="caption-2", late_frames=9,
                                        card_over_caption=True)
    if state == "unmeasured":
        document.update(measured=False, reason="Glyphs unavailable.")
    elif state == "no_captions":
        document["captions"] = []
    layout_path = write_layout_record(media, document)
    probe = ffmpeg.probe

    def vfr_probe(path, *args, **kwargs):
        info = probe(path, *args, **kwargs)
        return replace(info, nominal_fps=info.fps + 1) if Path(path) == media else info

    def unexpected_caption_check(*args, **kwargs):
        pytest.fail("Caption frame checks must not run on variable frame rate media")

    monkeypatch.setattr(ffmpeg, "probe", vfr_probe)
    monkeypatch.setattr(verify_module, "check_caption_timing", unexpected_caption_check)
    monkeypatch.setattr(verify_module, "check_caption_cards", unexpected_caption_check)
    report = verify(str(media), asr="none", loudness="note")
    captions = {"captions.timing", "captions.cards"}
    assert not any(row["check"] in captions for row in report["findings"])
    for check in captions:
        assert report["checks"][check] == {
            "ran": False, "status": "skipped",
            "reason": "variable frame rate render; caption frames have no fixed clock",
        }
    assert next(row for row in report["receipt"]["inputs"]
                if row["path"] == str(layout_path))["sha256"] == sha256_file(layout_path)
