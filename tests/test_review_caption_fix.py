"""Caption taps use saved geometry and change synthetic transcript display fixes."""

import json
import os
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_camera3d import _silhouette, _words
from test_feedback_batch import _cut, _json
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_removed import _snapshot, _source_words
from test_review_server import _in_process_request

from yap_studio.feedback.versions import load_version, save_version
from yap_studio.receipt import sha256_file
from yap_studio.review import server as server_module
from yap_studio.review.captions import caption_phrases
from yap_studio.review.launch import version_video
from yap_studio.review.server import ReviewServer
from yap_studio.review.store import ReviewStore
from yap_studio.style import resolve


@pytest.fixture
def caption_project(feedback_project, request):
    root = feedback_project
    merged = getattr(request, "param", None) == "merged"
    text = "BLOGGING WELL NYC" if merged else "BLOGGING WELL"
    _source_words(root, ("blogging", "well", "blogging.", "new" if merged else "side",
                         "york" if merged else "extra"))
    for number in (1, 2):
        _cut(root, label=f"caption {number}")
        receipt_file = root / "cut/receipt.json"
        record = json.loads(receipt_file.read_bytes())
        record["inputs"]["words:t1"] = sha256_file(root / "ingest/t1/words.json")
        _json(receipt_file, record)
        media = root / "final.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            f"color=c={'blue' if number == 1 else 'teal'}:s=270x480:r=30:d=4", "-vf",
            # A plain box stands in for the caption; the runner's ffmpeg has no drawtext.
            f"drawbox=x=50:y=300:w={10 * len(text)}:h=24:color=white:t=fill",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(media),
        ], stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True)
        _json(root / "captions/phrases.json", {
            "schema": "yap-studio/phrases@1", "time_base": "output", "fps": 30,
            "phrases": [{"ids": ["t1:0", "t1:1"] + (["t1:3"] if merged else []),
                         "text": text,
                         "start": 0, "end": 1.5, "stress": None}],
        })
        layout = {
            "schema": "yap-studio/clearance-layout@1", "width": 1080, "height": 1920,
            "measured": True, "media_sha256": sha256_file(media), "elements": [],
            "caption_phrases": json.loads((root / "captions/phrases.json").read_bytes()),
            "captions": [{"id": "caption-0", "text": text,
                          "start_frame": 0, "end_frame": 45,
                          "boxes": [{"x": 100 * number, "y": 1200, "w": 700, "h": 100}],
                          "words": [{"id": "t1:0", "source_ids": ["t1:0"], "text": "BLOGGING",
                                     "start_s": 0, "end_s": 0.5},
                                    {"id": "t1:1", "source_ids": ["t1:1"], "text": "WELL",
                                     "start_s": 0.6, "end_s": 1}]}],
        }
        if merged:
            layout["captions"][0]["words"].append({
                "id": "t1:3", "source_ids": ["t1:3", "t1:4"], "text": "NYC",
                "start_s": .8, "end_s": 1})
        _json(media.with_suffix(".layout.json"), layout)
        save_version(root, media=media)
    return root


@pytest.fixture
def caption_server(caption_project, tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "store")
    video = version_video([str(caption_project)], store, lambda _: [])
    return ReviewServer(video.media, video.sha256, None, store, token="test-token",
                        series=video.series, send_runner=lambda work: work())


def call(server, method, route, body=None, **kwargs):
    status, _, payload = _in_process_request(server, method, route, body, **kwargs)
    return status, json.loads(payload) if payload else None


def test_caption_endpoint_keeps_each_versions_ids_text_and_boxes(caption_server):
    root = caption_server.series.project_dir
    _json(root / "captions/phrases.json", {})
    for version in (1, 2):
        status, data = call(caption_server, "GET", f"/api/captions?version={version}")
        assert status == 200
        assert data["available"] and data["version"] == version
        phrase, = data["phrases"]
        assert (phrase["start"], phrase["end"]) == (0, 1.5)
        assert phrase["boxes"] == [{"x": 100 * version, "y": 1200, "w": 700, "h": 100}]
        assert [word["id"] for word in phrase["words"]] == ["t1:0", "t1:1"]
        assert [word["text"] for word in phrase["words"]] == ["BLOGGING", "WELL"]
        assert phrase["words"][0]["heard"] == "blogging"
        assert phrase["words"][0]["repeats"] == 2


@pytest.mark.parametrize("change", ["text", "timing", "missing_hash"])
def test_caption_transcript_identity_refuses_stale_targets(caption_server, change):
    root = caption_server.series.project_dir
    assert call(caption_server, "GET", "/api/captions?version=1")[1]["available"]
    if change == "missing_hash":
        manifest = load_version(root, "v1")
        pieces = json.loads((manifest.path / "cut-pieces.json").read_bytes())
        del pieces["inputs"]["words:t1"]
        _snapshot(manifest, "cut_pieces", pieces)
        _json(manifest.path / "manifest.json", manifest.to_dict())
        assert call(caption_server, "GET", "/api/captions?version=2")[1]["available"]
    else:
        file = root / "ingest/t1/words.json"
        data = json.loads(file.read_bytes())
        data["words"][0]["text" if change == "text" else "start"] = (
            "different" if change == "text" else 0.8)
        _json(file, data)
    file = root / "project.json"
    before = file.read_bytes()
    assert call(caption_server, "GET", "/api/captions?version=1")[1] == {
        "available": False, "version": 1, "phrases": []}
    assert call(caption_server, "POST", "/api/caption-fixes?version=1",
                {"word": "t1:0", "start": 0, "to": "logging"})[0] == 409
    assert file.read_bytes() == before
    assert not caption_server.store.comments(caption_server.series.versions[0].sha256)


def test_caption_unused_take_keeps_fixes_available(caption_server):
    root = caption_server.series.project_dir
    file = root / "project.json"
    data = json.loads(file.read_bytes())
    data["takes"].append({**data["takes"][0], "id": "t2"})
    _json(file, data)
    assert call(caption_server, "GET", "/api/captions?version=1")[1]["available"]
    assert call(caption_server, "POST", "/api/caption-fixes?version=1",
                {"word": "t1:0", "start": 0, "to": "logging"})[0] == 201


@pytest.mark.parametrize("camera", [False, True])
@pytest.mark.parametrize("every_time", [False, True])
def test_caption_merged_word_is_disabled_and_refuses_save(caption_server, camera, every_time):
    from yap_studio.captions import CaptionWord, build_phrases
    from yap_studio.render.camera3d import plan_camera3d
    from yap_studio.render.camera3d_look import display_words, plan_layout
    from yap_studio.render.layer_checks import CAPTION_SHRINK_PREFIX
    from yap_studio.stages.captions import caption_layer_plan, caption_layout

    root = caption_server.series.project_dir
    source = root / "ingest/t1/words.json"
    data = json.loads(source.read_bytes())
    data["words"][0]["text"], data["words"][1]["text"] = "new", "york"
    _json(source, data)
    style = resolve("preset:clean_minimal")
    fixes = {"new york": "NYC"}
    if camera:
        displayed = display_words(data["words"], style.data["captions"], word_fixes=fixes,
                                  vocabulary=[], gaps=[])
        plan = plan_camera3d({"time_base": "output", "words": data["words"]},
                             6, _silhouette()[1], display_words=displayed, onset_lead=.1)
        layout = plan_layout(plan)
    else:
        phrases = build_phrases([CaptionWord(w["id"], w["text"], w["start"], w["end"])
                                 for w in data["words"]], style.data["captions"], word_fixes=fixes)
        plan = caption_layer_plan(phrases, style, frames=180)
        summary = {"layout": [{"slot": i, "boxes": [{"x": 100, "y": 1200, "w": 700, "h": 100}]}
                              for i in range(len(json.loads(plan.variables["phrases"])))]}
        layout = caption_layout({"passes": {"runtime": {"findings": [
            {"message": CAPTION_SHRINK_PREFIX + json.dumps(summary)}]}}}, plan)
    manifest = load_version(root, "v2")
    pieces = json.loads((manifest.path / "cut-pieces.json").read_bytes())
    pieces["inputs"]["words:t1"] = sha256_file(source)
    _snapshot(manifest, "cut_pieces", pieces)
    _snapshot(manifest, "caption_layout", layout)
    _json(manifest.path / "manifest.json", manifest.to_dict())
    project = root / "project.json"
    before = project.read_bytes()
    status, targets = call(caption_server, "GET", "/api/captions?version=2")
    assert status == 200 and targets["available"]
    words = [word for phrase in targets["phrases"] for word in phrase["words"]]
    joined = next(word for word in words if word["text"] == "NYC")
    assert joined["editable"] is False
    assert joined["reason"] == "This caption joins several words. Fix it with a note."
    assert next(word for word in words if word["id"] == "t1:2")["editable"] is True
    status, response = call(caption_server, "POST", "/api/caption-fixes?version=2",
                            {"word": joined["id"], "start": joined["start"], "to": "Manhattan",
                             "every_time": every_time})
    assert status == 409
    assert response["error"]["message"] == joined["reason"]
    assert project.read_bytes() == before
    assert not caption_server.store.comments(caption_server.sha256)


def test_camera_caption_layout_keeps_group_and_hero_source_ids(caption_project):
    from yap_studio.render.camera3d import plan_camera3d
    from yap_studio.render.camera3d_look import plan_layout
    from yap_studio.stages.captions_3d import plan_phrases

    words = _words()
    for index, word in enumerate(words["words"]):
        word["id"] = f"t1:{index}"
    source = caption_project / "ingest/t1/words.json"
    data = json.loads(source.read_bytes())
    data["words"] = [{key: word[key] for key in ("id", "text", "start", "end")}
                     for word in words["words"]]
    _json(source, data)
    plan = plan_camera3d(words, 5.3, _silhouette()[1], onset_lead=.1,
                         max_words=2, max_characters=18)
    layout = plan_layout(plan)
    assert any(item["id"].startswith("hero-") for item in layout["captions"])
    layout["caption_phrases"] = {"phrases": [phrase.to_dict() for phrase in plan_phrases(
        words["words"], resolve("preset:clean_minimal").data["captions"], frames=159)]}
    manifest = load_version(caption_project, "v2")
    pieces = json.loads((manifest.path / "cut-pieces.json").read_bytes())
    pieces["inputs"]["words:t1"] = sha256_file(source)
    _snapshot(manifest, "cut_pieces", pieces)
    _snapshot(manifest, "caption_layout", layout)
    response = caption_phrases(caption_project, manifest)
    shown = [word for phrase in response["phrases"] for word in phrase["words"]]
    assert {word["text"]: word["id"] for word in shown} == {
        word["text"]: word["id"] for word in words["words"]}
    assert len(response["phrases"]) == len(layout["captions"])


@pytest.mark.parametrize("field", ["id", "source_ids"])
def test_caption_layout_without_word_ids_is_unavailable(caption_project, field):
    manifest = load_version(caption_project, "v2")
    layout = json.loads((manifest.path / "caption_layout.json").read_bytes())
    del layout["captions"][0]["words"][0][field]
    _snapshot(manifest, "caption_layout", layout)
    assert caption_phrases(caption_project, manifest) == {"available": False, "phrases": []}


@pytest.mark.parametrize("versions", [(1, 2), (2, 1)])
@pytest.mark.parametrize("every_time", [False, True])
@pytest.mark.parametrize("same_second", [False, True])
def test_caption_undo_checks_later_fixes_across_versions(caption_server, monkeypatch,
                                                        versions, every_time, same_second):
    notes = []
    for second, version in enumerate(versions):
        monkeypatch.setattr(caption_server.store, "_now",
                            lambda second=second: datetime(
                                2026, 10, 22, 12, 0, 0 if same_second else second,
                                second if same_second else 0, tzinfo=UTC))
        status, note = call(caption_server, "POST", f"/api/caption-fixes?version={version}",
                            {"word": "t1:0", "start": 0, "to": "logging", "every_time": every_time})
        assert status == 201
        notes.append(note)
    file = caption_server.series.project_dir / "project.json"
    before = file.read_bytes()
    status, response = call(caption_server, "POST",
                            f"/api/caption-fixes?version={versions[0]}", {"undo": notes[0]["id"]})
    assert status == 409
    assert response["error"]["message"] == "This word has another fix. Undo it first."
    assert file.read_bytes() == before
    for version, note in reversed(list(zip(versions, notes, strict=True))):
        assert call(caption_server, "GET", f"/api/review?version={version}")[1]["comments"]
        assert call(caption_server, "POST", f"/api/caption-fixes?version={version}",
                    {"undo": note["id"]})[0] == 200


@pytest.mark.parametrize("identifier, blocked", [("t1:2", True), ("t2:0", True), ("t1:1", False)])
def test_caption_every_time_checks_other_matching_source_words(caption_server, identifier, blocked):
    root = caption_server.series.project_dir
    file = root / "project.json"
    data = json.loads(file.read_bytes())
    data["word_text_fixes"] = {identifier: "one-off"}
    if identifier.startswith("t2:"):
        data["takes"].append({**data["takes"][0], "id": "t2"})
        words = json.loads((root / "ingest/t1/words.json").read_bytes())
        words["take"] = "t2"
        words["words"] = [{"id": "t2:0", "text": "BLOGGING!", "start": .9, "end": 1.1}]
        _json(root / "ingest/t2/words.json", words)
    _json(file, data)
    before = file.read_bytes()
    status, response = call(caption_server, "POST", "/api/caption-fixes",
                            {"word": "t1:0", "start": 0, "to": "writing", "every_time": True})
    assert status == (409 if blocked else 201)
    if blocked:
        assert response["error"]["message"] == "This word has a one-off fix. Undo it first."
        assert file.read_bytes() == before
        assert not caption_server.store.comments(caption_server.sha256)
    else:
        project = json.loads(file.read_bytes())
        assert project["word_fixes"] == {"blogging": "writing"}
        assert project["word_text_fixes"] == {identifier: "one-off"}


@pytest.mark.parametrize("first_start, first_merged, requested_start", [
    (0, False, 2), (0, True, 2), (0, False, 2 - 1 / 60), (0, False, 2 + 1 / 60),
    (1.99, False, 1.998), (1.99, True, 1.998),
])
def test_caption_repeated_excerpt_uses_tapped_occurrence(caption_server, first_start,
                                                        first_merged, requested_start):
    root = caption_server.series.project_dir
    manifest = load_version(root, "v2")
    layout = json.loads((manifest.path / "caption_layout.json").read_bytes())
    first = layout["captions"][0]
    later = {**first, "id": "caption-1", "start_frame": 60, "end_frame": 105,
             "words": [{**first["words"][0], "start_s": 2, "end_s": 2.5}]}
    first["words"] = [{**first["words"][0], "start_s": first_start, "end_s": first_start + .4}]
    first["start_frame"] = int(first_start * 30)
    first["end_frame"] = int((first_start + .4) * 30)
    if first_merged:
        first["words"][0]["source_ids"] = ["t1:0", "t1:1"]
    layout["captions"].append(later)
    _snapshot(manifest, "caption_layout", layout)
    _json(manifest.path / "manifest.json", manifest.to_dict())
    status, note = call(caption_server, "POST", "/api/caption-fixes?version=2",
                        {"word": "t1:0", "start": requested_start, "to": "logging"})
    assert status == 201
    assert note["start_frame"] == 60 and note["start_s"] == 2
    assert "at 0:02" in note["text"]
    review = call(caption_server, "GET", "/api/review?version=2")[1]
    assert note["text"] in review["text"]
    assert json.loads((root / "project.json").read_bytes())["word_text_fixes"] == {
        "t1:0": "logging"}


def test_caption_every_time_refuses_a_one_off_fix_without_writing(caption_server):
    assert call(caption_server, "POST", "/api/caption-fixes",
                {"word": "t1:0", "start": 0, "to": "logging"})[0] == 201
    file = caption_server.series.project_dir / "project.json"
    before = file.read_bytes()
    notes = caption_server.store.comments(caption_server.sha256)
    status, response = call(caption_server, "POST", "/api/caption-fixes",
                            {"word": "t1:0", "start": 0, "to": "writing", "every_time": True})
    assert status == 409
    assert response["error"]["message"] == "This word has a one-off fix. Undo it first."
    assert file.read_bytes() == before
    assert caption_server.store.comments(caption_server.sha256) == notes


@pytest.mark.parametrize("every_time", [False, True])
def test_caption_save_note_export_and_undo(caption_server, every_time):
    root = caption_server.series.project_dir
    source = (root / "ingest/t1/words.json").read_bytes()
    status, note = call(caption_server, "POST", "/api/caption-fixes?version=2",
                        {"word": "t1:0", "start": 0, "to": "logging", "every_time": every_time})
    assert status == 201
    project = json.loads((root / "project.json").read_bytes())
    field, key = ("word_fixes", "blogging") if every_time else ("word_text_fixes", "t1:0")
    assert project[field] == {key: "logging"}
    assert note["target"] == "caption" and note["request"]["type"] == "caption_fix"
    assert "Caption: BLOGGING -> logging at 0:00" in note["text"]
    assert (root / "ingest/t1/words.json").read_bytes() == source
    assert note["text"] in call(caption_server, "GET", "/api/review?version=2")[1]["text"]
    assert call(caption_server, "POST", "/api/caption-fixes?version=2",
                {"undo": note["id"]})[0] == 200
    assert json.loads((root / "project.json").read_bytes())[field] == {}
    assert not call(caption_server, "GET", "/api/review?version=2")[1]["comments"]


@pytest.mark.parametrize("body", [
    {"word": "t1:99", "start": 0, "to": "logging"}, {"word": "t1:0", "start": 0, "to": " "},
    {"word": "t1:0", "start": 0, "to": "line\nbreak"},
    {"word": "t1:1", "start": 0.6, "to": "good", "every_time": True},
    {"word": "t1:0", "start": 0, "to": "logging", "every_time": "yes"},
    {"word": "t1:0", "to": "logging"},
    {"word": "t1:0", "start": False, "to": "logging"},
    {"word": "t1:0", "start": "0", "to": "logging"},
    {"word": "t1:0", "start": -1, "to": "logging"},
    {"word": "t1:0", "start": float("nan"), "to": "logging"},
    {"word": "t1:0", "start": 0.1, "to": "logging"},
])
def test_caption_bad_save_does_not_write(caption_server, body):
    file = caption_server.series.project_dir / "project.json"
    before = file.read_bytes()
    assert call(caption_server, "POST", "/api/caption-fixes", body)[0] == 400
    assert file.read_bytes() == before
    assert not caption_server.store.comments(caption_server.sha256)


def test_caption_guards(caption_server):
    body = {"word": "t1:0", "start": 0, "to": "logging"}
    assert call(caption_server, "GET", "/api/captions", token=None)[0] == 403
    assert call(caption_server, "POST", "/api/caption-fixes?t=test-token", body,
                token=None)[0] == 403
    assert call(caption_server, "GET", "/api/caption-fixes")[0] == 405
    assert call(caption_server, "GET", "/api/captions?version=99")[0] == 400


@pytest.mark.parametrize("new_text", ["writing", "logging"])
def test_caption_undo_restores_previous_and_refuses_a_newer_fix(caption_server, new_text):
    file = caption_server.series.project_dir / "project.json"
    data = json.loads(file.read_bytes())
    data["word_text_fixes"] = {"t1:0": "earlier"}
    _json(file, data)
    _, first = call(caption_server, "POST", "/api/caption-fixes",
                    {"word": "t1:0", "start": 0, "to": "logging"})
    _, second = call(caption_server, "POST", "/api/caption-fixes",
                     {"word": "t1:0", "start": 0, "to": new_text})
    assert call(caption_server, "POST", "/api/caption-fixes", {"undo": first["id"]})[0] == 409
    assert json.loads(file.read_bytes())["word_text_fixes"] == {"t1:0": new_text}
    for note in (second, first):
        assert call(caption_server, "POST", "/api/caption-fixes", {"undo": note["id"]})[0] == 200
    assert json.loads(file.read_bytes())["word_text_fixes"] == {"t1:0": "earlier"}


def test_caption_notes_reach_the_agent_without_a_heard_fix(caption_server):
    _, note = call(caption_server, "POST", "/api/caption-fixes",
                   {"word": "t1:0", "start": 0, "to": "logging"})
    assert call(caption_server, "POST", "/api/rounds", {})[0] == 202
    root = caption_server.series.project_dir
    batch = json.loads((root / "feedback/b1.json").read_bytes())
    assert batch["items"][0]["comment"]["request"] == note["request"]
    assert not json.loads((root / "project.json").read_bytes()).get("word_fixes")
    review = call(caption_server, "GET", "/api/review")[1]
    assert review["series"]["rounds"][0]["status"] == "sent"


def test_old_caption_versions_are_unavailable(caption_server):
    file = caption_server.series.project_dir / "versions/v1/manifest.json"
    data = json.loads(file.read_bytes())
    del data["snapshots"]["caption_layout"]
    _json(file, data)
    status, response = call(caption_server, "GET", "/api/captions?version=1")
    assert status == 200 and response == {"available": False, "version": 1, "phrases": []}


def test_caption_failed_note_rolls_back_fix(caption_server, monkeypatch):
    file = caption_server.series.project_dir / "project.json"
    before = file.read_bytes()
    monkeypatch.setattr(caption_server.store, "add_comment",
                        lambda *_args: (_ for _ in ()).throw(OSError("synthetic failure")))
    assert call(caption_server, "POST", "/api/caption-fixes",
                {"word": "t1:0", "start": 0, "to": "logging"})[0] == 500
    assert file.read_bytes() == before


@pytest.mark.skipif(
    not os.environ.get("YAP_STUDIO_PUPPETEER") or not os.environ.get("YAP_STUDIO_CHROME"),
    reason="set YAP_STUDIO_PUPPETEER and YAP_STUDIO_CHROME for the real browser flow",
)
@pytest.mark.parametrize("caption_project", ["merged"], indirect=True)
def test_caption_browser_phone_and_desktop(caption_project, tmp_path):
    store = ReviewStore(tmp_path / "store")
    video = version_video([str(caption_project)], store, lambda _: [])
    server = ReviewServer(video.media, video.sha256, None, store, series=video.series)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    result_file = tmp_path / "caption-browser.json"
    shots = Path(os.environ.get("YAP_STUDIO_CAPTION_SHOTS", str(tmp_path / "shots")))
    try:
        result = subprocess.run([
            "node", str(Path(__file__).parent / "js/review-caption-browser.mjs"),
            server.url, str(result_file), str(shots),
        ], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result_file.read_text()) == [390, 1440]
        project = json.loads((caption_project / "project.json").read_bytes())
        assert project["word_text_fixes"] == {} and project["word_fixes"] == {}
        assert not store.comments(video.sha256)
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()
