"""Hash-bound faces, clearance findings, measured coverage, and verify integration."""

from __future__ import annotations

import importlib
import json
import shutil
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import synth

from yap_studio import ffmpeg
from yap_studio.checks.caption_layout import LAYOUT_SUFFIX, LayoutRecord, write_layout_record
from yap_studio.checks.face_clearance import FACES_SUFFIX, check_face_clearance, final_faces
from yap_studio.errors import ValidationError
from yap_studio.faces import detect
from yap_studio.faces.clearance import (
    DEFAULT_HAIR,
    NO_HAIR,
    CaptionPhrase,
    Element,
    HairAllowance,
    Layout,
    check_layout,
    head_box,
)
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.faces.project import FACE_EVERY
from yap_studio.receipt import sha256_file
from yap_studio.report import _CHECK_LABELS
from yap_studio.verify import verify

FACE = Box(140, 90, 20, 15)
TARGET = Box(120, 70, 80, 60)


@pytest.fixture(autouse=True)
def portable_face_detector(monkeypatch):
    """Keep the system detector disabled, including when loaded as a suite plugin."""
    monkeypatch.setattr(detect, "unavailable_reason",
                        lambda: "System detector unavailable in tests")

    def unexpected_detector():
        pytest.fail("Tests must supply measured records or a fake face detector")

    monkeypatch.setattr(detect, "default_detector", unexpected_detector)


@pytest.fixture
def clip(tmp_path):
    return synth.picture_source(tmp_path / "final.mp4", "testsrc2=s=320x180:r=30", 90)


def _faces(media=None, *, frames=300, box=FACE):
    per_frame = tuple((FaceBox(box, 0.9, "detected"),) if box else () for _ in range(frames))
    return FacesData(
        "measured", None,
        VideoIdentity(media.name if media else "final.mp4", sha256_file(media) if media else "hash",
                      frames, Fraction(30), 320, 180),
        {"name": "fake"}, {}, per_frame, tuple(detect.find_gaps(per_frame)),
    )


def _record(*, caption=False, kind="card", box=TARGET):
    target = CaptionPhrase("phrase-1", "First words", (box,), 30, 75) if caption else \
        Element("card-1", kind, box, 30, 75)
    return LayoutRecord(
        Layout(320, 180, () if caption else (target,), (target,) if caption else (),
               clearance_px=24, hair=DEFAULT_HAIR),
        True, None, {},
    )


class FakeDetector:
    def __init__(self):
        self.calls = []

    def detect(self, video, *, every):
        self.calls.append((video, every))
        info = ffmpeg.probe(video)
        samples = {frame: (FaceBox(FACE, 0.9, "detected"),)
                   for frame in detect.sample_frames(info.video_frames, every)}
        return detect.DetectorRun(info.width, info.height, info.video_frames, samples,
                                  {"name": "fake"})


@pytest.mark.parametrize("measured", [True, False])
def test_matching_faces_record_is_used(clip, measured):
    faces = _faces(clip, frames=90)
    if not measured:
        faces = replace(faces, status="unmeasured", reason="Record was not measured.",
                        frames=(), gaps=())
    detect.write_faces(clip.with_suffix(FACES_SUFFIX), faces)
    detector = FakeDetector()
    actual, source = final_faces(clip, detector=detector)
    assert actual == faces and source == "record" and detector.calls == []


@pytest.mark.parametrize("state,reason", [
    ("absent", "no file at this path"),
    ("another_video", "video hash does not match"),
    ("corrupt", "invalid faces JSON"),
    ("bad_schema", "faces.schema"),
    ("bad_utf8", "invalid faces JSON"),
    ("missing_link", "no file at this path"),
])
def test_bad_or_absent_faces_record_falls_back_to_fake_detector(clip, state, reason):
    target = clip.with_suffix(FACES_SUFFIX)
    if state == "another_video":
        faces = _faces(clip, frames=90)
        detect.write_faces(target, replace(faces, video=replace(faces.video, sha256="another")))
    elif state == "corrupt":
        target.write_text("{")
    elif state == "bad_schema":
        target.write_text("{}")
    elif state == "bad_utf8":
        target.write_bytes(b"\xff")
    elif state == "missing_link":
        target.symlink_to(clip.parent / "missing.json")
    detector = FakeDetector()
    faces, source = final_faces(clip, detector=detector)
    assert faces.status == "measured" and faces.video.sha256 == sha256_file(clip)
    assert detector.calls == [(clip, FACE_EVERY)]
    assert "detected on the render" in source and reason in source


def test_unmeasured_fallback_reports_detector_reason(clip, monkeypatch):
    reason = "face measurement is unavailable on this system"
    monkeypatch.setattr(detect, "unavailable_reason", lambda: reason)
    faces, source = final_faces(clip)
    assert faces.status == "unmeasured" and faces.reason == reason
    assert "detected on the render" in source
    findings, coverage = check_face_clearance(_record(), faces, faces_source=source)
    assert findings == []
    assert coverage == {"ran": False, "reason": f"faces not measured: {reason}"}


@pytest.mark.parametrize("caption,kind", [
    (True, "caption"), (False, "card"), (False, "title"), (False, "overlay"),
])
def test_face_violation_names_target_time_and_geometry(caption, kind):
    record = _record(caption=caption, kind=kind)
    findings, coverage = check_face_clearance(record, _faces(), faces_source="record")
    assert coverage == {"ran": True, "reason": None}
    (finding,) = findings
    assert finding.check == "faces.clearance" and finding.severity == "fail"
    name = "First words" if caption else "card-1"
    assert finding.message == (
        f'{kind.capitalize()} "{name}" is over the speaker\'s face from 0:01.0 to 0:02.5.'
    )
    assert finding.time_s == 1.0
    assert finding.suggested_action == (
        "Run captions again so the phrase is placed clear of the face, "
        "or move the face out of the caption band." if caption else
        "Move the overlay in the timeline (change its placement or box) "
        "so it sits clear of the face." if kind == "overlay" else
        "Plan the card again (run graphics) so it sits clear of the face."
    )
    assert finding.evidence == {
        "target_type": "caption" if caption else "element",
        "target_id": "phrase-1" if caption else "card-1",
        **({} if caption else {"element_id": "card-1"}),
        "target_kind": kind, "start_frame": 30, "end_frame": 75, "frame": 30,
        "gap_px": -35, "required_px": 24, "target_box": TARGET.to_dict(),
        "face": FACE.to_dict(), "head": head_box(FACE).to_dict(), "faces_source": "record",
    }


def test_card_fully_covering_face_fails():
    record = _record()
    head = head_box(FACE)
    assert TARGET.x < head.x < head.x1 < TARGET.x1
    assert TARGET.y < head.y < head.y1 < TARGET.y1
    findings, _ = check_face_clearance(record, _faces(), faces_source="record")
    assert len(findings) == 1 and findings[0].evidence["gap_px"] < 0


def test_overlay_label_names_original_element_and_time():
    record = _record(kind="overlay")
    record = replace(record, element_ids={"card-1": "broll-photo"},
                     labels={"card-1": "photo.png"})
    (finding,), coverage = check_face_clearance(record, _faces(), faces_source="record")
    assert coverage["ran"]
    assert finding.message == (
        'Overlay "broll-photo" (photo.png) is over the speaker\'s face '
        'from 0:01.0 to 0:02.5.'
    )
    assert finding.evidence["label"] == "photo.png"
    assert finding.evidence["element_id"] == "broll-photo"
    assert finding.evidence["target_id"] == "card-1"


@pytest.mark.parametrize("piece_id", [
    "card-1#2", "card-1~2", "card-1#2~3", "card-1~2#3",
])
def test_card_piece_names_original_card_and_keeps_piece_evidence(piece_id):
    record = _record()
    piece = replace(record.layout.elements[0], id=piece_id)
    record = replace(record, layout=replace(record.layout, elements=(piece,)),
                     element_ids={piece_id: "card-1"})
    findings, _ = check_face_clearance(record, _faces(), faces_source="record")
    (finding,) = findings
    assert 'Card "card-1"' in finding.message and "#" not in finding.message
    assert "~" not in finding.message
    assert finding.evidence["target_id"] == piece_id
    assert finding.evidence["element_id"] == "card-1"


@pytest.mark.parametrize("piece_id", ["intro~alt", "intro#alt", "card~2", "card#2"])
@pytest.mark.parametrize("restricted", [False, True])
def test_literal_card_name_is_preserved_without_element_id(piece_id, restricted):
    record = _record()
    piece = replace(record.layout.elements[0], id=piece_id)
    record = replace(record, layout=replace(record.layout, elements=(piece,)),
                     face_windows={piece_id: ((30, 40), (55, 65))} if restricted else {})
    findings, coverage = check_face_clearance(record, _faces(), faces_source="record")
    assert coverage == {"ran": True, "reason": None}
    assert len(findings) == (2 if restricted else 1)
    for finding in findings:
        assert f'Card "{piece_id}" is over the speaker\'s face' in finding.message
        assert finding.evidence["target_id"] == finding.evidence["element_id"] == piece_id


@pytest.mark.parametrize("windows", [(), ((30, 75),), ((35, 40), (55, 65), (70, 75))])
def test_face_windows_restrict_only_the_marked_element(windows):
    record = _record()
    card = replace(record.layout.elements[0], id="card-1#2")
    elements = (card, replace(card, id="front-card"))
    caption = CaptionPhrase("phrase-1", "First words", (TARGET,), 30, 75)
    record = replace(record, layout=replace(record.layout, elements=elements, captions=(caption,)),
                     face_windows={card.id: windows}, element_ids={card.id: "card-1"})
    findings, coverage = check_face_clearance(record, _faces(), faces_source="record")
    assert coverage == {"ran": True, "reason": None}
    expected = [("front-card", 30, 75), ("phrase-1", 30, 75)]
    expected.extend((card.id, start, end) for start, end in windows)
    assert sorted((row.evidence["target_id"], row.evidence["start_frame"],
                   row.evidence["end_frame"]) for row in findings) == sorted(expected)
    for row in findings:
        if row.evidence["target_id"] == card.id:
            assert 'Card "card-1"' in row.message
            assert "#" not in row.message and "~" not in row.message
            assert row.evidence["element_id"] == "card-1"
    assert record.layout.elements == elements


def test_face_window_ids_avoid_existing_caption_and_element_ids():
    record = _record()
    card = record.layout.elements[0]
    caption = CaptionPhrase("face-window-0", "First words", (TARGET,), 30, 75)
    record = replace(record, layout=replace(
        record.layout, elements=(card, replace(card, id="face-window-1")), captions=(caption,),
    ), face_windows={card.id: ((35, 40), (55, 65))})
    findings, coverage = check_face_clearance(record, _faces(), faces_source="record")
    assert coverage == {"ran": True, "reason": None}
    assert sorted((row.evidence["target_id"], row.evidence["start_frame"],
                   row.evidence["end_frame"]) for row in findings) == [
        ("card-1", 35, 40), ("card-1", 55, 65),
        ("face-window-0", 30, 75), ("face-window-1", 30, 75),
    ]


@pytest.mark.parametrize("measured", [False, True])
def test_restricted_layout_validation_error_is_not_run(monkeypatch, measured):
    checker = importlib.import_module("yap_studio.checks.face_clearance")
    record = replace(_record(), measured=measured, reason=None if measured else "No glyphs.")
    error = ValidationError("layout.elements", "Invalid restricted layout")

    def invalid_layout(obj, **changes):
        if "elements" in changes:
            raise error
        return replace(obj, **changes)

    monkeypatch.setattr(checker, "replace", invalid_layout)
    findings, coverage = check_face_clearance(record, _faces(), faces_source="record")
    assert findings == []
    assert coverage == {"ran": False, "reason": str(error),
                        **({} if measured else {"note": "captions were not measured: No glyphs."})}


@pytest.mark.parametrize("caption", [False, True])
def test_moving_target_clear_passes(caption):
    record = _record(caption=caption, box=Box(120, 5, 80, 20))
    assert check_face_clearance(record, _faces(), faces_source="record") == (
        [], {"ran": True, "reason": None},
    )


@pytest.mark.parametrize("caption", [False, True])
@pytest.mark.parametrize("margin,gap,fails", [(24, 23, True), (24, 24, False), (36, 35, True)])
def test_layout_margin_and_hair_are_used(caption, margin, gap, fails):
    hair = HairAllowance(0.8, 0.25)
    head = head_box(FACE, hair)
    record = _record(caption=caption, box=Box(head.x, head.y - gap - 10, head.w, 10))
    record = replace(record, layout=replace(record.layout, clearance_px=margin, hair=hair))
    findings, coverage = check_face_clearance(record, _faces(), faces_source="record")
    assert coverage["ran"] and len(findings) == int(fails)
    if fails:
        assert findings[0].evidence["gap_px"] == gap
        assert findings[0].evidence["required_px"] == margin
        assert findings[0].evidence["head"] == head.to_dict()


def test_glyph_boxes_instead_of_caption_band_are_checked():
    record = _record(caption=True, box=Box(20, 90, 20, 15))
    phrase = record.layout.captions[0]
    record = replace(record, layout=replace(record.layout, captions=(
        replace(phrase, boxes=(*phrase.boxes, Box(270, 90, 20, 15))),
    )))
    assert check_face_clearance(record, _faces(), faces_source="record")[0] == []


def test_layout_beyond_face_frames_is_not_run():
    findings, coverage = check_face_clearance(_record(), _faces(frames=60), faces_source="record")
    assert findings == [] and coverage["ran"] is False
    assert coverage["reason"] == (
        "Bad value for layout.elements[0].end_frame: "
        "must be at or before video frame count 60"
    )


@pytest.mark.parametrize("source", ["insert", "transition"])
def test_replacement_and_estimated_frame_sources_are_noted(source):
    faces = _faces(frames=90)
    frames = faces.frames[:40] + ((),) * 10 + faces.frames[50:]
    faces = replace(faces, frames=frames, gaps=((40, 50),), sampling={"frame_sources": [
        {"source": "cut", "start_frame": 0, "end_frame": 40},
        {"source": source, "start_frame": 40, "end_frame": 50},
        {"source": "zoom_estimate", "start_frame": 50, "end_frame": 80},
        {"source": "hold", "start_frame": 80, "end_frame": 90},
    ]})
    findings, coverage = check_face_clearance(_record(), faces, faces_source="record")
    assert [(row.evidence["start_frame"], row.evidence["end_frame"]) for row in findings] == [
        (30, 40), (50, 75),
    ]
    assert coverage["ran"] is True
    assert coverage["note"] == (
        "10 frames have no face boxes; frame sources: cut: 40, hold: 10, "
        f"{source}: 10, zoom_estimate: 30"
    )


def test_estimates_with_face_boxes_are_noted():
    faces = replace(_faces(), sampling={"frame_sources": [
        {"source": "zoom_estimate", "start_frame": 0, "end_frame": 300},
    ]})
    _, coverage = check_face_clearance(_record(), faces, faces_source="record")
    assert coverage["note"] == "frame sources: zoom_estimate: 300"


def test_caption_gap_violations_are_ignored():
    record = _record(caption=True)
    record = replace(record, layout=replace(record.layout, elements=(
        Element("card-1", "card", TARGET, 30, 75),
    ), hair=NO_HAIR))
    faces = _faces(box=Box(270, 140, 20, 15))
    assert {row.kind for row in check_layout(faces, record.layout).violations} == {"caption_gap"}
    assert check_face_clearance(record, faces, faces_source="record") == (
        [], {"ran": True, "reason": None},
    )


def test_presentation_times_are_used_for_findings():
    faces = _faces()
    faces = replace(faces, times=tuple(frame / 24 for frame in range(300)), end_time=12.5)
    findings, _ = check_face_clearance(_record(), faces, faces_source="record")
    assert findings[0].time_s == 1.25
    assert "from 0:01.2 to 0:03.1." in findings[0].message


@pytest.mark.parametrize("state", [
    "absent", "corrupt", "pass", "caption", "card", "unmeasured_layout", "unmeasured_faces",
])
@pytest.mark.parametrize("with_edl", [False, True])
def test_verify_face_clearance_and_receipt(synth_media, tmp_path, monkeypatch, state, with_edl):
    module = importlib.import_module("yap_studio.verify")
    media = tmp_path / "final.mp4"
    shutil.copyfile(synth_media.path("clean"), media)
    if state in {"absent", "corrupt"}:
        detect.write_faces(media.with_suffix(FACES_SUFFIX), synth.calibration_faces(media))
    if state == "corrupt":
        media.with_suffix(LAYOUT_SUFFIX).write_text("{")
    elif state != "absent":
        document = synth.calibration_layout(card_over_face=state in {"card", "unmeasured_layout"},
                                            caption_over_face=state == "caption")
        if state == "unmeasured_layout":
            document.update(measured=False, reason="Glyphs unavailable.", captions=[])
        write_layout_record(media, document)
        faces = synth.calibration_faces(media, over_face=state != "pass")
        if state == "unmeasured_faces":
            faces = replace(faces, status="unmeasured", reason="Faces unavailable.",
                            frames=(), gaps=())
        detect.write_faces(media.with_suffix(FACES_SUFFIX), faces)

    def unexpected_detection(*args, **kwargs):
        pytest.fail("Verification must reuse records and skip detection without a layout")

    monkeypatch.setattr(detect, "detect_faces", unexpected_detection)
    checker = importlib.import_module("yap_studio.checks.face_clearance")
    monkeypatch.setattr(checker, "detect_faces", unexpected_detection)
    if state in {"absent", "corrupt"}:
        monkeypatch.setattr(module, "final_faces", unexpected_detection)
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
    coverage = report["checks"]["faces.clearance"]
    findings = [row for row in report["findings"] if row["check"] == "faces.clearance"]
    if state == "absent":
        assert coverage == {"ran": False, "status": "skipped",
                            "reason": "no layout record next to this render"}
        assert findings == []
    elif state == "corrupt":
        for check in ("captions.timing", "captions.cards", "faces.clearance"):
            assert report["checks"][check] == {"ran": True, "status": "warn", "reason": None}
            warnings = [row for row in report["findings"] if row["check"] == check]
            assert len(warnings) == 1 and warnings[0]["severity"] == "warn"
            assert warnings[0]["message"].startswith("corrupt layout record:")
    elif state == "unmeasured_faces":
        assert coverage == {"ran": False, "status": "skipped",
                            "reason": "faces not measured: Faces unavailable."}
        assert findings == []
    else:
        expected = "pass" if state == "pass" else "fail"
        assert report["status"] == expected, report["findings"]
        assert coverage["ran"] is True and coverage["status"] == expected
        assert len(findings) == int(expected == "fail")
        if state == "unmeasured_layout":
            assert "captions were not measured: Glyphs unavailable." in coverage["note"]
            assert 'Card "card-1"' in findings[0]["message"]
        if findings:
            assert findings[0]["evidence"]["faces_source"] == "record"
            assert "0:07.2" in findings[0]["message"]
    target = media.with_suffix(FACES_SUFFIX)
    inputs = {row["path"]: row for row in report["receipt"]["inputs"]}
    assert inputs[str(target)]["sha256"] == sha256_file(target)
    assert inputs[str(target)]["bytes"] == target.stat().st_size
    assert _CHECK_LABELS["faces.clearance"] == "Captions and cards clear of the face"


def test_verify_variable_frame_rate_still_checks_face_clearance(synth_media, tmp_path, monkeypatch):
    media = tmp_path / "final.mp4"
    shutil.copyfile(synth_media.path("clean"), media)
    write_layout_record(media, synth.calibration_layout(caption_over_face=True))
    detect.write_faces(media.with_suffix(FACES_SUFFIX),
                       synth.calibration_faces(media, over_face=True))
    probe = ffmpeg.probe

    def vfr_probe(path, *args, **kwargs):
        info = probe(path, *args, **kwargs)
        return replace(info, nominal_fps=info.fps + 1) if Path(path) == media else info

    monkeypatch.setattr(ffmpeg, "probe", vfr_probe)
    report = verify(str(media), asr="none", loudness="note")
    assert report["checks"]["faces.clearance"] == {"ran": True, "status": "fail", "reason": None,
                                                  "note": "522 frames have no face boxes"}
    assert report["checks"]["captions.timing"]["ran"] is False
    assert report["checks"]["captions.cards"]["ran"] is False
