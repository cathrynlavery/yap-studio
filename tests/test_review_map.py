"""The video map reads only the shown version's saved records."""

import json
from types import SimpleNamespace

import pytest
from test_feedback_batch import _json
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_versions import _render_media
from test_review_server import _in_process_request

from yap_studio.feedback.versions import LAYERS_SCHEMA, save_version
from yap_studio.receipt import sha256_file
from yap_studio.review import server as server_module
from yap_studio.review.launch import version_video
from yap_studio.review.server import ReviewServer
from yap_studio.review.store import ReviewStore

NAMES = ["Story", "Captions", "Titles and Text hook", "Cards", "B-roll and backgrounds",
         "Moments", "Music and sounds"]


@pytest.fixture
def mapped_server(feedback_project, tmp_path, monkeypatch):
    root = feedback_project
    _json(root / "timeline/agent.json", {"schema": "yap-studio/timeline@1", "words_sha256": {},
                                       "elements": [
        {"id": "agent-title", "kind": "title", "lane": "agent", "start": {"time": 0},
         "end": {"time": 2}, "lines": ["A saved title"]},
        {"id": "agent-moment", "kind": "moment", "lane": "agent",
         "start": {"word": "t1:2", "segment": "s2"},
         "hold_s": 0.5, "style": "sticker_freeze"},
        {"id": "agent-sound", "kind": "sound", "lane": "agent",
         "start": {"time": 1}, "sound": "chime"},
    ]})
    layer_path = root / "layers.json"
    _json(layer_path, {"schema": LAYERS_SCHEMA, "time_base": "output", "layers": [
        {"id": name, "kind": kind, "start": start, "end": end, "text": label}
        for name, kind, start, end, label in [
            ("phrase", "caption", 0.2, 1.5, "A phrase"),
            ("agent-title", "title", 0, 2, None),
            ("text-hook", "title", 0, 4, "A hook"),
            ("card", "graphic", 1, 2, "A card"),
            ("insert", "insert", 2, 3, "A picture"),
            ("back", "background", 0, 4, "A background"),
            ("music", "music_level", 0, 4, "Quiet music"),
        ]
    ]})
    final = root / "final.mp4"
    _render_media(final, 120, label="synthetic final")
    save_version(root, media=final, layers=layer_path)
    save_version(root, layers=layer_path)  # Story cut, even with final records saved beside it.
    # Live edits cannot change either version's map.
    _json(root / "timeline/agent.json", {"schema": "yap-studio/timeline@1", "elements": []})
    monkeypatch.setattr(server_module, "_QuietThreadingHTTPServer",
                        lambda *_args: SimpleNamespace(server_port=0, server_close=lambda: None))
    store = ReviewStore(tmp_path / "store")
    video = version_video([str(root)], store, lambda _: [])
    return ReviewServer(video.media, video.sha256, None, store, token="test-token",
                        series=video.series)


def lanes(server, version):
    code, _, body = _in_process_request(server, "GET", f"/api/video-map?version={version}")
    assert code == 200, body.decode()
    data = json.loads(body)
    assert data["version"] == version
    return data["lanes"]


def test_map_includes_each_kind_from_the_selected_snapshot(mapped_server):
    rows = lanes(mapped_server, 1)
    assert [row["name"] for row in rows] == NAMES
    assert all(row["items"] for row in rows)
    assert rows[0]["items"] == [
        {"start": 0, "end": 2, "label": "Start really", "target": "story"},
        {"start": 2, "end": 2, "label": "Join", "target": "story"},
        {"start": 2, "end": 4, "label": "Start End", "target": "story"},
    ]
    assert rows[1]["items"] == [
        {"start": 0.2, "end": 1.5, "label": "A phrase", "target": "caption"},
    ]
    assert rows[2]["items"][0]["label"] == "A saved title"
    assert rows[2]["items"][1]["label"] == "A hook"
    assert rows[3]["items"][0]["target"] == "element"
    assert len(rows[4]["items"]) == 2
    assert rows[5]["items"] == [
        {"start": 2.8, "end": 3.3, "label": "Sticker freeze", "target": "video"},
    ]
    assert {item["label"] for item in rows[6]["items"]} == {"Quiet music", "Chime"}
    assert all(item["target"] == "video" for item in rows[6]["items"])


def test_story_cut_has_only_story_and_captions(mapped_server):
    assert [row["name"] for row in lanes(mapped_server, 2)] == NAMES[:2]


def test_missing_records_leave_empty_lanes(mapped_server):
    file = mapped_server.series.project_dir / "versions/v1/manifest.json"
    manifest = json.loads(file.read_text())
    for key in ("layers", "timeline"):
        manifest["snapshots"][key] = None
    pieces_file = file.parent / "cut-pieces.json"
    pieces = json.loads(pieces_file.read_text())
    pieces["pieces"] = []
    _json(pieces_file, pieces)
    manifest["snapshots"]["cut_pieces"]["sha256"] = sha256_file(pieces_file)
    _json(file, manifest)
    rows = lanes(mapped_server, 1)
    assert [row["name"] for row in rows] == NAMES
    assert all(row["items"] == [] for row in rows)


def test_map_is_authenticated_read_only_and_checks_versions(mapped_server):
    assert _in_process_request(mapped_server, "GET", "/api/video-map", token=None)[0] == 403
    assert _in_process_request(mapped_server, "POST", "/api/video-map", {})[0] == 405
    assert _in_process_request(mapped_server, "GET", "/api/video-map?version=3")[0] == 400


def test_map_refuses_changed_snapshot(mapped_server):
    file = mapped_server.series.project_dir / "versions/v1/layers.json"
    file.write_text("{}")
    code, _, body = _in_process_request(mapped_server, "GET", "/api/video-map?version=1")
    assert code in {400, 409, 500}
    assert b"snapshot changed after it was saved" in body


def test_map_uses_saved_style_for_an_implicit_hold(mapped_server):
    folder = mapped_server.series.project_dir / "versions/v1"
    timeline_file = folder / "timeline/agent.json"
    document = json.loads(timeline_file.read_text())
    document["elements"].append({"id": "agent-default-title", "kind": "title", "lane": "agent",
                                 "start": {"time": 2.8}, "lines": ["Saved style title"]})
    _json(timeline_file, document)
    style_file = folder / "style.json"
    _json(style_file, {"schema_version": 1, "name": "Synthetic", "hook": {"duration_s": 1}})
    file = folder / "manifest.json"
    manifest = json.loads(file.read_text())
    manifest["snapshots"]["timeline"]["lanes"]["agent"]["sha256"] = sha256_file(timeline_file)
    manifest["snapshots"]["style"] = {
        "ref": "style.json", "file": "style.json", "sha256": sha256_file(style_file),
    }
    _json(file, manifest)
    assert lanes(mapped_server, 1)[2]["items"][-1] == {
        "start": 2.8, "end": 3.8, "label": "Saved style title", "target": "element",
    }


@pytest.mark.parametrize("explicit_end", [False, True])
def test_text_hook_fallback_spans_the_saved_video(mapped_server, explicit_end):
    folder = mapped_server.series.project_dir / "versions/v1"
    timeline_file = folder / "timeline/agent.json"
    document = json.loads(timeline_file.read_text())
    document["elements"][0]["start"] = {"time": 1}
    if not explicit_end:
        del document["elements"][0]["end"]
    _json(timeline_file, document)
    style_file = folder / "style.json"
    _json(style_file, {"schema_version": 1, "name": "Synthetic",
                       "hook": {"style": "text_hook", "duration_s": 1}})
    file = folder / "manifest.json"
    manifest = json.loads(file.read_text())
    manifest["snapshots"]["layers"] = None
    manifest["snapshots"]["timeline"]["lanes"]["agent"]["sha256"] = sha256_file(timeline_file)
    manifest["snapshots"]["style"] = {
        "ref": "style.json", "file": "style.json", "sha256": sha256_file(style_file),
    }
    _json(file, manifest)
    assert lanes(mapped_server, 1)[2]["items"] == [
        {"start": 0, "end": 4, "label": "A saved title", "target": "element"},
    ]


def test_unavailable_word_anchor_leaves_an_empty_lane(mapped_server):
    folder = mapped_server.series.project_dir / "versions/v1"
    timeline_file = folder / "timeline/agent.json"
    document = json.loads(timeline_file.read_text())
    document["elements"][1]["start"]["word"] = "t1:999"
    _json(timeline_file, document)
    file = folder / "manifest.json"
    manifest = json.loads(file.read_text())
    manifest["snapshots"]["timeline"]["lanes"]["agent"]["sha256"] = sha256_file(timeline_file)
    _json(file, manifest)
    assert lanes(mapped_server, 1)[5]["items"] == []
