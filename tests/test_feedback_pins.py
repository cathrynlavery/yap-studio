"""Pinned frame decoding and geometry use synthetic media and sidecars."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_feedback_batch import _cut, _export, _json, _layers
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_locate import LAYERS, _comment
from test_feedback_versions import _receipt_inputs, _render_media

from yap_studio.errors import YapError
from yap_studio.faces.clearance import LAYOUT_SCHEMA
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.faces.project import FACE_EVERY, _cut_faces_path
from yap_studio.feedback import pins
from yap_studio.feedback.batch import build_batch
from yap_studio.feedback.versions import VersionManifest, save_version
from yap_studio.receipt import sha256_file


@pytest.fixture(params=["rawvideo", "libx264rgb", "libx264"])
def pin_video(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Make sixteen lossless RGB frames whose red channel identifies the frame."""
    media = tmp_path / ("colours.mp4" if request.param == "libx264" else "colours.nut")
    pixels = np.zeros((16, 96, 128, 3), dtype=np.uint8)
    for frame in range(16):
        pixels[frame, :, :, 0] = 8 * frame
    codec = ["-c:v", request.param]
    if request.param == "libx264rgb":
        codec += ["-crf", "0", "-g", "30"]
    elif request.param == "libx264":
        codec += ["-bf", "2", "-g", "8", "-qp", "0"]
    pixel_format = "yuv444p" if request.param == "libx264" else "rgb24"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "128x96", "-r", "30", "-i", "pipe:0", *codec,
                    "-pix_fmt", pixel_format, str(media)], input=pixels.tobytes(),
                   capture_output=True, timeout=30, check=True)
    return media


@pytest.mark.parametrize("frame", [0, 7, 9, 15])
def test_still_is_exact_frame_and_pin_is_deterministic(pin_video: Path, frame: int) -> None:
    pixels = pins.decode_frame(pin_video, frame=frame, time_s=frame / 30,
                               fps=Fraction(30), width=128, height=96, video_index=0)
    assert pixels.shape == (96, 128, 3)
    tolerance = 2 if pin_video.suffix == ".mp4" else 0
    assert np.all(np.abs(pixels[:, :, 0].astype(int) - frame * 8) <= tolerance)
    assert np.all(pixels[:, :, 1:] <= tolerance)
    original = pixels.copy()
    drawn = pins.draw_pin(pixels, 64, 48)
    assert tuple(drawn[48, 64]) == (255, 94, 77)
    assert tuple(drawn[48, 75]) == (255, 255, 255)
    assert tuple(drawn[48, 78]) == (16, 18, 21)
    assert np.array_equal(drawn, pins.draw_pin(pixels, 64, 48))
    assert np.array_equal(pixels, original)
    for x, y in ((0, 0), (127, 95)):
        assert tuple(pins.draw_pin(pixels, x, y)[y, x]) == (255, 94, 77)


@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize("time_s,frame,vfr_frame", [
    (0, 0, 0), (0.05, 0, 0), (0.1, 1, 1), (0.15, 1, 1), (0.35, 3, 1), (0.95, 9, 9),
])
def test_time_only_still_uses_containing_frame(tmp_path: Path, time_s: float, frame: int,
                                              vfr_frame: int, variable: bool) -> None:
    media = tmp_path / "timed.nut"
    select = ["-vf", "select=eq(n\\,0)+eq(n\\,1)+eq(n\\,4)+eq(n\\,9)"] if variable else []
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "nullsrc=s=128x96:r=10:d=1,geq=r='N*20':g=0:b=0,format=rgb24",
                    *select, "-fps_mode", "passthrough", "-c:v", "libx264rgb", "-crf", "0",
                    "-g", "3", str(media)],
                   stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True)
    pixels = pins.decode_frame(media, frame=None, time_s=time_s, fps=None,
                               width=128, height=96, video_index=0)
    assert np.all(pixels[:, :, 0] == (vfr_frame if variable else frame) * 20)
    assert np.all(pixels[:, :, 1:] == 0)


def _saved_final(root: Path, *, end_hold_frames: int = 0,
                 layers: Path | None = None) -> VersionManifest:
    edit = json.loads((root / "edit.json").read_text())
    edit["end_hold_s"] = end_hold_frames / 30
    _json(root / "edit.json", edit)
    receipt_path = root / "cut/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"] = _receipt_inputs(root)
    _json(receipt_path, receipt)
    media = root / "final/picture.mp4"
    media.parent.mkdir()
    _render_media(media, 120 + end_hold_frames, label="assembled picture")
    _json(root / "final/receipt.json", {
        "schema": "yap-studio/stage-receipt@1", "stage": "assemble", "status": "verified",
        "outputs": [{"path": "final/picture.mp4", "sha256": sha256_file(media)}],
    })
    return save_version(root, media=media, layers=layers)


def _layout(digest: str, width: int = 128, height: int = 96) -> dict:
    return {"schema": LAYOUT_SCHEMA, "media_sha256": digest, "width": width, "height": height,
            "captions": [{"id": "caption-12", "text": "Here is why", "start_frame": 0,
                          "end_frame": 16, "boxes": [{"x": 50, "y": 50, "w": 10, "h": 10},
                                                     {"x": 70, "y": 50, "w": 10, "h": 10}]}],
            "elements": [{"id": f"{kind}-piece", "element_id": kind, "kind": kind,
                          "label": f"Example {kind}", "start_frame": 0, "end_frame": 16,
                          "x": 0, "y": 0, "w": 100, "h": 96}
                         for kind in ("background", "overlay", "card", "title")]}


def _faces(digest: str, width: int = 128, height: int = 96) -> FacesData:
    face = FaceBox(Box(50, 40, 20, 20), 1, "detected")
    return FacesData("measured", None, VideoIdentity("colours.nut", digest, 16,
                                                   Fraction(30), width, height), {},
                     {"frame_sources": [{"start_frame": 7, "end_frame": 8, "source": "insert"}]},
                     ((face,),) * 16, ())


def test_caption_hits_use_only_active_box_windows(pin_video: Path) -> None:
    layout = _layout(sha256_file(pin_video))
    layout["captions"][0]["boxes"][0].update(start_frame=0, end_frame=8)
    layout["captions"][0]["boxes"][1].update(start_frame=8, end_frame=16)
    hit, = [h for h in pins.layout_hits(layout, 7, 55, 50) if h["kind"] == "caption"]
    assert hit["box"] == {"x": 50, "y": 50, "w": 10, "h": 10}
    assert not any(h["kind"] == "caption" for h in pins.layout_hits(layout, 8, 35, 50))


def test_geometry_padding_order_and_active_windows(pin_video: Path) -> None:
    layout = _layout(sha256_file(pin_video))
    hits = pins.layout_hits(layout, 7, 35, 50)
    assert [hit["kind"] for hit in hits] == ["caption", "title", "card", "overlay", "background"]
    assert hits[0]["box"] == {"x": 50, "y": 50, "w": 30, "h": 10}
    assert hits[0]["text"] == "Here is why"
    assert hits[2]["id"] == "card" and hits[2]["text"] == "Example card"
    assert not any(hit["kind"] == "caption" for hit in pins.layout_hits(layout, 7, 25, 50))
    assert pins.layout_hits(layout, 16, 50, 50) == []
    faces = _faces(sha256_file(pin_video))
    hits = pins.face_hits(faces, 7, 47, 50)
    assert [hit["kind"] for hit in hits] == ["speaker", "insert"]
    assert hits[0]["box"] == {"x": 50, "y": 40, "w": 20, "h": 20}
    assert [hit["kind"] for hit in pins.face_hits(faces, 7, 45, 50)] == ["insert"]
    assert pins.face_hits(faces, 16, 50, 50) == []


def test_saved_pin_batch_and_missing_sources(feedback_project: Path) -> None:
    root = feedback_project
    version = save_version(root)
    comment = replace(_comment("c1", target="element", start_frame=7, end_frame=None),
                      pin={"x": 0.4213, "y": 0.3105, "frame": 7, "time_s": 0.2333})
    batch = build_batch(root, _export(root, [comment, _comment("c2")], version=version))
    item = next(item for item in batch["items"] if item["comment"]["id"] == "c1")
    assert item["comment"]["pin"] == comment.pin
    pin = item["pin"]
    assert pin["under"] == [] and pin["known"] == dict.fromkeys(
        ("layout", "faces", "layers", "timeline"), False)
    assert pin["still_error"] is None and pin["still"] == "feedback/b1/v1-c1.png"
    png = (root / pin["still"]).read_bytes()
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(root / pin["still"]),
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True)
    canvas = pin["canvas"]
    assert len(decoded.stdout) == canvas["width"] * canvas["height"] * 3
    assert tuple(np.frombuffer(decoded.stdout, np.uint8).reshape(
        canvas["height"], canvas["width"], 3)[canvas["y"], canvas["x"]]) == (255, 94, 77)
    unpinned = next(item for item in batch["items"] if item["comment"]["id"] == "c2")
    assert "pin" not in unpinned and "pin" not in unpinned["comment"]
    second = build_batch(root, _export(root, [comment], version=version))
    assert (root / second["items"][0]["pin"]["still"]).read_bytes() == png


@pytest.mark.parametrize("stale", [None, "hash", "size", "malformed"])
def test_sidecar_identity_and_combined_order(pin_video: Path, tmp_path: Path, stale) -> None:
    source = tmp_path / "final" / "picture.nut"
    source.parent.mkdir()
    digest = sha256_file(pin_video)
    version = SimpleNamespace(id="v1", media_path=pin_video,
                              media={"source": "final/picture.nut", "sha256": digest,
                                     "fps": "30/1"}, snapshots={"layers": None})
    layout = _layout("b" * 64 if stale == "hash" else digest,
                     width=256 if stale == "size" else 128)
    _json(source.with_suffix(".layout.json"), {} if stale == "malformed" else layout)
    faces = _faces("b" * 64 if stale == "hash" else digest,
                   width=256 if stale == "size" else 128)
    _json(source.with_suffix(".faces.json"), {} if stale == "malformed" else faces.to_dict())
    comment = replace(_comment("c1", start_frame=7, end_frame=None),
                      pin={"x": 0.4, "y": 0.5, "frame": 7, "time_s": 7 / 30})
    context = pins.PinContext(tmp_path, (version,), None)
    result = pins.pin_spot(comment, batch_id="b1", version_id="v1", context=context)
    assert result["known"]["layout"] == (stale is None)
    assert result["known"]["faces"] == (stale is None)
    assert [hit["kind"] for hit in result["under"]] == (
        ["caption", "title", "card", "overlay", "speaker", "insert", "background"]
        if stale is None else [])


@pytest.mark.parametrize("measured", [False, True])
def test_pin_layout_measurement_status_preserves_card_and_reason(feedback_project: Path,
                                                                measured: bool) -> None:
    root = feedback_project
    version = save_version(root)
    reason = None if measured else "Runtime checks were skipped."
    _json(root / "cut/cut.layout.json", {
        "schema": LAYOUT_SCHEMA, "media_sha256": version.media["sha256"],
        "width": 32, "height": 32, "measured": measured, "reason": reason, "captions": [],
        "elements": [{"id": "card-piece", "element_id": "card", "kind": "card",
                      "label": "Example card", "start_frame": 0, "end_frame": 120,
                      "x": 0, "y": 0, "w": 32, "h": 32}],
    })
    comment = replace(_comment("c1", start_frame=7, end_frame=None),
                      pin={"x": 0.4, "y": 0.5, "frame": 7, "time_s": 7 / 30})
    spot = build_batch(root, _export(root, [comment], version=version))["items"][0]["pin"]
    assert spot["under"] == [{"kind": "card", "id": "card", "text": "Example card",
                              "box": {"x": 0, "y": 0, "w": 32, "h": 32}, "source": "layout"}]
    assert spot["known"]["layout"] is measured
    assert spot.get("layout_reason") == reason


def test_unsaved_series_pin_has_reason_and_unknown_geometry(feedback_project: Path) -> None:
    root = feedback_project
    version = save_version(root)
    export = _export(root, [], version=version)
    pin = {"x": 0.5, "y": 0.5, "frame": 7, "time_s": 7 / 30}
    raw = _export(root, [replace(_comment("c1", start_frame=7, end_frame=None), pin=pin)])[
        "comments"][0]
    export["series"] = {"id": "example", "version": 2, "versions": [
        {"number": 1, "sha256": "a" * 64, "version_id": None},
        {"number": 2, "sha256": version.media["sha256"], "version_id": version.id}],
        "threads": [{**raw, "version": 1, "reply": None, "state": "open"}]}
    item = build_batch(root, export)["items"][0]
    assert item["comment"]["pin"] == pin and item["pin"]["still"] is None
    assert item["pin"]["still_error"] == "The comment's version is not saved in this project."
    assert not any(item["pin"]["known"].values()) and item["pin"]["canvas"] is None


def test_decode_failure_is_recorded(feedback_project: Path, monkeypatch) -> None:
    root = feedback_project
    version = save_version(root, layers=_layers(root))
    def failed(*args, **kwargs):
        raise YapError("Could not decode the pinned frame.")
    monkeypatch.setattr(pins, "decode_frame", failed)
    comment = replace(_comment("c1", start_frame=7, end_frame=None),
                      pin={"x": 1, "y": 1, "frame": 7, "time_s": 7 / 30})
    result = build_batch(root, _export(root, [comment], version=version))["items"][0]["pin"]
    assert result["still"] is None and result["still_error"] == "Could not decode the pinned frame."
    assert result["canvas"]["x"] == result["canvas"]["width"] - 1
    assert result["known"]["layers"] is True


@pytest.mark.parametrize("end_hold_frames", [0, 30])
def test_timeline_current_snapshot_guard_cache_hits_and_layers(feedback_project: Path,
                                                            monkeypatch,
                                                            end_hold_frames: int) -> None:
    from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

    root = feedback_project
    version = _saved_final(root, end_hold_frames=end_hold_frames, layers=_layers(root))
    elements = tuple(ResolvedElement(id, kind, "agent", 0, 60, "picture", payload)
                     for id, kind, payload in (
                         ("insert-full", "insert", {"media": {"fit": "cover"}}),
                         ("insert-default", "insert", {"media": {}}),
                         ("insert-split", "insert", {"media": {"fit": "split"}}),
                         ("moment-one", "moment", {"style": "color_pop"}),
                         ("zoom-one", "zoom", {}),
                     ))
    calls = []
    monkeypatch.setattr(pins, "resolve_timeline", lambda project: (
        calls.append(project) or ResolvedTimeline(elements, (), (), 120, end_hold_frames)))
    context = pins.PinContext(root, (version,), version)
    comments = [replace(_comment(f"c{number}", start_frame=7, end_frame=None),
                        pin={"x": 0.4, "y": 0.5, "frame": 7, "time_s": 7 / 30})
                for number in (1, 2)]
    for comment in comments:
        spot = pins.pin_spot(comment, batch_id="b1", version_id=version.id, context=context)
        assert spot["known"]["timeline"] is True
        assert spot["under"] == [{"kind": "insert", "id": id, "text": None,
                                  "box": {"x": 0, "y": 0, "w": spot["canvas"]["width"],
                                          "h": spot["canvas"]["height"]}, "source": "timeline"}
                                 for id in ("insert-full", "insert-default")]
        assert [(item["kind"], item["id"], item["text"]) for item in spot["on_screen"]
                if item["source"] == "timeline"] == [
                    ("insert", "insert-split", "split"), ("moment", "moment-one", "color_pop"),
                    ("zoom", "zoom-one", None)]
    assert calls == [root]
    for name in ("edit.json", "removals.json", "timeline/agent.json", "timeline/new.json"):
        file = root / name
        original = file.read_bytes() if file.exists() else None
        file.write_bytes(b"{}")
        stale = pins.PinContext(root, (version,), version)
        assert stale.timeline_for(version) is None
        if original is None:
            file.unlink()
        else:
            file.write_bytes(original)
    assert calls == [root]
    assert pins.PinContext(root, (version,), None).timeline_for(version) is None


@pytest.mark.parametrize("evidence", ["cut", "missing", "hash", "failed", "stage", "sidecar"])
def test_pin_timeline_requires_assembled_media(feedback_project: Path, monkeypatch,
                                             evidence: str) -> None:
    from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

    root = feedback_project
    version = save_version(root) if evidence == "cut" else _saved_final(root)
    receipt_path = root / "final/receipt.json"
    if evidence == "missing":
        receipt_path.unlink()
    elif evidence != "cut":
        receipt = json.loads(receipt_path.read_text())
        if evidence == "hash":
            receipt["outputs"][0]["sha256"] = "a" * 64
        elif evidence == "failed":
            receipt["status"] = "failed"
        elif evidence == "stage":
            receipt["stage"] = "cut"
        else:
            receipt["outputs"][0]["path"] = "final/picture.layout.json"
        _json(receipt_path, receipt)
    timeline = ResolvedTimeline(tuple(
        ResolvedElement(kind, kind, "agent", 0, 60, "picture", payload)
        for kind, payload in (("insert", {"media": {}}),
                              ("moment", {"style": "color_pop"}), ("zoom", {}))
    ), (), (), 120, 0)
    monkeypatch.setattr(pins, "resolve_timeline", lambda project: timeline)
    comment = replace(_comment("c1", start_frame=7, end_frame=None),
                      pin={"x": 0.4, "y": 0.5, "frame": 7, "time_s": 7 / 30})
    context = pins.PinContext(root, (version,), version)
    spot = pins.pin_spot(comment, batch_id="b1", version_id=version.id, context=context)
    assert spot["known"]["timeline"] is False
    assert spot["under"] == [] and spot["on_screen"] == []


def test_pin_current_cut_uses_only_cached_faces(feedback_project: Path, monkeypatch) -> None:
    root = feedback_project
    calls = []
    monkeypatch.setattr(pins, "cached_cut_faces", lambda project, media, **kwargs:
                        calls.append((project, media)) or None)
    comment = replace(_comment("c1", start_frame=7, end_frame=None),
                      pin={"x": 0.4, "y": 0.5, "frame": 7, "time_s": 7 / 30})
    item = build_batch(root, _export(root, [comment]))["items"][0]
    assert item["pin"]["still"] == "feedback/b1/cut-c1.png"
    assert calls == [(root, root / "cut/cut.mp4")]


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("matching", [False, True])
def test_pin_saved_cut_keeps_hash_checked_face_cache(feedback_project: Path,
                                                    saved: bool, matching: bool) -> None:
    root = feedback_project
    version = save_version(root) if saved else None
    digest = sha256_file(root / "cut/cut.mp4")
    face = FaceBox(Box(8, 8, 16, 16), 1, "detected")
    faces = FacesData("measured", None,
                      VideoIdentity("cut.mp4", digest if matching else "a" * 64,
                                    120, Fraction(30), 32, 32), {},
                      {"every": FACE_EVERY}, ((face,),) * 120, ())
    _json(_cut_faces_path(root, digest), faces.to_dict())
    comment = replace(_comment("c1", start_frame=7, end_frame=None),
                      pin={"x": 0.4, "y": 0.5, "frame": 7, "time_s": 7 / 30})
    spot = build_batch(root, _export(root, [comment], version=version))["items"][0]["pin"]
    assert spot["known"]["faces"] is matching
    assert spot["under"] == ([{"kind": "speaker", "id": None, "text": None,
                               "box": face.box.to_dict(), "source": "faces"}] if matching else [])


@pytest.mark.parametrize("changed", ["cut/cut-words.json", "cut/spans.json", "frame_count"])
@pytest.mark.parametrize("end_hold_frames", [0, 30])
def test_pin_timeline_rejects_changed_cut(feedback_project: Path, monkeypatch,
                                        changed: str, end_hold_frames: int) -> None:
    from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

    root = feedback_project
    version = _saved_final(root, end_hold_frames=end_hold_frames)
    calls = []
    total_frames = version.media["frame_count"]
    if changed == "frame_count":
        total_frames += 1
    else:
        path = root / changed
        data = json.loads(path.read_text())
        if changed == "cut/cut-words.json":
            data["words"][0]["start"] += 1 / 30
        else:
            data["fps"] = "60/1"
        _json(path, data)
    assert sha256_file(root / "timeline/agent.json") == (
        version.snapshots["timeline"]["lanes"]["agent"]["sha256"])
    timeline = ResolvedTimeline((
        ResolvedElement("insert-one", "insert", "agent", 0, 60, "picture", {"media": {}}),
    ), (), (), total_frames - end_hold_frames, end_hold_frames)
    monkeypatch.setattr(pins, "resolve_timeline", lambda project:
                        calls.append(project) or timeline)
    comment = replace(_comment("c1", start_frame=7, end_frame=None),
                      pin={"x": 0.4, "y": 0.5, "frame": 7, "time_s": 7 / 30})
    context = pins.PinContext(root, (version,), version)
    spot = pins.pin_spot(comment, batch_id="b1", version_id=version.id, context=context)
    assert spot["known"]["timeline"] is False
    assert spot["under"] == [] and spot["on_screen"] == []
    assert calls == ([root] if changed == "frame_count" else [])


@pytest.mark.parametrize("saved", [False, True])
def test_pin_base_uses_explicit_layer_windows(feedback_project: Path, saved: bool) -> None:
    root = feedback_project
    version = save_version(root) if saved else None
    comment = replace(_comment("c1", target="element"),
                      pin={"x": 0.4, "y": 0.5, "frame": 69, "time_s": 2.3})
    item = build_batch(root, _export(root, [comment], version=version),
                       layers=_layers(root))["items"][0]
    assert item["comment"]["version"] == (version.id if version else None)
    assert item["location"]["layers_known"] is True
    assert [layer["id"] for layer in item["location"]["layers"]] == ["card", "caption"]
    assert item["pin"]["known"]["layers"] is True
    assert item["pin"]["on_screen"] == [
        {"kind": "graphic", "id": "card", "text": None, "source": "layers"},
        {"kind": "caption", "id": "caption", "text": "Start End", "source": "layers"},
    ]


@pytest.mark.parametrize("current_cut", [False, True])
def test_pin_earlier_version_keeps_snapshot_layers(feedback_project: Path,
                                                  current_cut: bool) -> None:
    root = feedback_project
    first = save_version(root, layers=_layers(root))
    _cut(root, label="next cut")
    second = save_version(root)
    context = pins.PinContext(root, (first, second), second, current_cut=current_cut,
                              cut_layers=(replace(LAYERS[0], id="new-card"),))
    assert context.source_for(first)["layers"] == LAYERS
    if current_cut:
        assert context.source_for(second)["layers"] is None


@pytest.mark.parametrize("windows,frame,behind", [
    (None, 7, False), ([], 7, True), ([[7, 9]], 6, True),
    ([[7, 9]], 7, False), ([[7, 9]], 8, False), ([[7, 9]], 9, True),
])
def test_pin_card_order_respects_face_windows(tmp_path: Path, monkeypatch,
                                            windows, frame: int, behind: bool) -> None:
    version = SimpleNamespace(id="v1", media={"fps": "30/1"})
    layout = _layout("a" * 64)
    card = next(element for element in layout["elements"] if element["kind"] == "card")
    if windows is not None:
        card["face_windows"] = windows
    source = {"media": tmp_path / "picture.mp4", "layout": layout, "faces": _faces("a" * 64),
              "info": SimpleNamespace(width=128, height=96, video_index=0), "layers": None}
    context = pins.PinContext(tmp_path, (version,), None)
    monkeypatch.setattr(context, "source_for", lambda _: source)
    monkeypatch.setattr(pins, "decode_frame", lambda *args, **kwargs:
                        np.zeros((96, 128, 3), dtype=np.uint8))
    comment = replace(_comment("c1", start_frame=frame, end_frame=None),
                      pin={"x": 0.4, "y": 0.5, "frame": frame, "time_s": frame / 30})
    spot = pins.pin_spot(comment, batch_id="b1", version_id="v1", context=context)
    expected = (["caption", "title", "overlay", "speaker", "card"] if behind else
                ["caption", "title", "card", "overlay", "speaker"])
    if frame == 7:
        expected.append("insert")
    assert [hit["kind"] for hit in spot["under"]] == [*expected, "background"]


@pytest.mark.parametrize("saved", [True, False])
def test_answered_pin_round_trips_into_new_batch(feedback_project: Path, saved: bool) -> None:
    from yap_studio.feedback.clarify import answer_clarification, ask_clarification

    root = feedback_project
    version = save_version(root) if saved else None
    comment = replace(_comment("c1", target="element", start_frame=7, end_frame=None),
                      pin={"x": 0.4, "y": 0.5, "frame": 7, "time_s": 7 / 30})
    build_batch(root, _export(root, [comment], version=version))
    ref = version.id if version is not None else None
    ask_clarification(root, "b1", ref, "c1", "Which card?", options=[
        {"value": "first", "label": "First card"}, {"value": "second", "label": "Second card"}])
    answer_clarification(root, "b1", ref, "c1", option="first")
    if version is None:
        version = save_version(root)
    carried = build_batch(root, _export(root, [], version=version))["items"][0]
    assert carried["carried"] is True and carried["comment"]["pin"] == comment.pin
    if saved:
        assert carried["pin"]["still"] == "feedback/b2/v1-c1.png"
    else:
        assert carried["pin"]["still"] is None
        assert carried["pin"]["still_error"] == (
            "The comment's version is not saved in this project.")


@pytest.mark.parametrize("fps,frame,end", [(Fraction(30), 580, 581), (Fraction(15), 290, 582)])
def test_pin_windows_use_exact_frame_middle(tmp_path: Path, monkeypatch, fps, frame, end) -> None:
    from yap_studio.feedback.versions import LayerWindow
    from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

    version = SimpleNamespace(id="v1", media={"fps": str(fps)})
    context = pins.PinContext(tmp_path, (version,), version)
    source = {"media": tmp_path / "picture.mp4", "layout": None, "faces": None,
              "info": SimpleNamespace(width=128, height=96, video_index=0),
              "layers": (LayerWindow("edge", "overlay", 19.333333, 19.4, None),
                         LayerWindow("ended", "title", 19, 19.35, None))}
    monkeypatch.setattr(context, "source_for", lambda _: source)
    timeline = ResolvedTimeline((
        ResolvedElement("moment", "moment", "agent", 580, end, "picture", {"style": "color_pop"}),
    ), (), (), 600, 0)
    monkeypatch.setattr(context, "timeline_for", lambda _: timeline)
    monkeypatch.setattr(pins, "decode_frame", lambda *args, **kwargs:
                        np.zeros((96, 128, 3), dtype=np.uint8))
    comment = replace(_comment("c1", start_frame=frame, start_s=19.3333),
                      pin={"x": 0.4, "y": 0.5, "frame": frame, "time_s": 19.3333})
    spot = pins.pin_spot(comment, batch_id="b1", version_id="v1", context=context)
    assert [(item["id"], item["source"]) for item in spot["on_screen"]] == [
        ("edge", "layers"), ("moment", "timeline")]


@pytest.mark.parametrize("location", [
    "relative", "absolute", "parent", "internal_parent", "symlink", "loop", "dot",
])
def test_sidecars_require_a_source_inside_project(pin_video: Path, tmp_path: Path,
                                                 monkeypatch, location: str) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if location == "relative":
        value = "final/picture.nut"
        source = root / value
        source.parent.mkdir()
    elif location == "dot":
        value, source = ".", root
    else:
        source = outside / "picture.nut"
        value = {"absolute": str(source), "parent": "../outside/picture.nut",
                 "internal_parent": "sub/../../outside/picture.nut",
                 "symlink": "link/picture.nut", "loop": "loop/picture.nut"}[location]
        (root / "sub").mkdir()
        (root / "link").symlink_to(outside, target_is_directory=True)
        if location == "loop":
            (root / "loop").symlink_to(root / "loop", target_is_directory=True)
    digest = sha256_file(pin_video)
    _json(source.with_suffix(".layout.json"), _layout(digest))
    _json(source.with_suffix(".faces.json"), _faces(digest).to_dict())
    version = SimpleNamespace(id="v1", media_path=pin_video,
                              media={"source": value, "sha256": digest},
                              snapshots={"layers": None})
    reads = []
    read_text = Path.read_text
    def track_read(file, *args, **kwargs):
        reads.append(file)
        return read_text(file, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", track_read)
    result = pins.PinContext(root, (version,), None).source_for(version)
    assert bool(reads) == (location == "relative")
    assert (result["layout"] is not None) == (location == "relative")
    assert (result["faces"] is not None) == (location == "relative")
