"""Uncertain joins follow retained source edges into the reviewed version."""

import json
from fractions import Fraction

import pytest
from test_feedback_batch import _cut, _json
from test_feedback_batch import feedback_project as feedback_project

from yap_studio.feedback.versions import save_version
from yap_studio.project.model import CutWord, CutWords
from yap_studio.receipt import sha256_file
from yap_studio.review.findings import findings_for_media


@pytest.fixture
def join_project(feedback_project):
    root = feedback_project
    words = CutWords("output", {}, tuple(
        CutWord(f"t1:{index}", f"s{index // 2 + 1}", "speech", start, start + 0.2)
        for index, start in enumerate((0.2, 1.0, 1.5, 2.3, 2.9, 3.6))
    ))
    _json(root / "cut/cut-words.json", words.to_dict())
    path = root / "cut/receipt.json"
    record = json.loads(path.read_text())
    record["plan"]["pieces"] = [
        {"segment_id": f"s{index + 1}", "take": "t1", "out_f0": index * 40,
         "n_frames": 40, "src_t0": "0", "src_t1": "4/3"}
        for index in range(3)
    ]
    record["plan"]["listen_joins"] = [
        {"frame": frame, "time_s": frame / 30, "left": left, "right": right}
        for frame, left, right in ((40, "s1", "s2"), (80, "s2", "s3"))
    ]
    for output in record["outputs"]:
        output["sha256"] = sha256_file(root / output["path"])
    _json(path, record)
    return root, record, words


def listens(root, media):
    return findings_for_media(media, sha256_file(media), project_dir=root)["items"]


def save_legacy_version(root):
    version = save_version(root)
    path = version.path / "cut-pieces.json"
    pieces = json.loads(path.read_text())
    pieces.pop("listen_joins", None)
    _json(path, pieces)
    manifest = version.to_dict()
    manifest["snapshots"]["cut_pieces"]["sha256"] = sha256_file(path)
    _json(version.path / "manifest.json", manifest)
    return version


def test_two_joins_offer_two_listens_with_speech_on_both_sides(join_project):
    root, record, words = join_project
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    items = listens(root, root / "cut/cut.mp4")
    assert len(items) == 2
    for item, join in zip(items, record["plan"]["listen_joins"], strict=True):
        at = join["time_s"]
        assert item["reason"] == "Listen to this join"
        assert item["target"] == "story"
        assert item["start_s"] == pytest.approx(max(0, at - 1.5))
        assert item["end_s"] == pytest.approx(min(4, at + 1.5))
        assert any(item["start_s"] <= word.start < word.end <= at for word in words.words)
        assert any(at <= word.start < word.end <= item["end_s"] for word in words.words)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_legacy_saved_timeline_maps_shifted_joins_and_omits_a_changed_edge(join_project):
    root, record, _ = join_project
    version = save_legacy_version(root)
    for piece in record["plan"]["pieces"]:
        piece["out_f0"] += 30
    record["plan"]["pieces"].insert(0, {
        "segment_id": "intro", "take": "t1", "out_f0": 0, "n_frames": 30,
        "src_t0": "10", "src_t1": "11",
    })
    record["plan"]["total_frames"] += 30
    for join in record["plan"]["listen_joins"]:
        join["frame"] += 30
        join["time_s"] += 1
    _json(root / "cut/receipt.json", record)
    assert [(item["start_frame"], item["end_frame"]) for item in
            listens(root, version.media_path)] == [(0, 85), (35, 120)]
    record["plan"]["pieces"][2]["src_t0"] = "1/30"
    _json(root / "cut/receipt.json", record)
    items = listens(root, version.media_path)
    assert len(items) == 1
    assert items[0]["start_frame"] == 35


@pytest.mark.parametrize("receipt_state", ["revised", "missing", "failed", "take-changed"])
@pytest.mark.parametrize("explicit_version", [False, True])
def test_saved_version_keeps_listens_after_cut_revision(
    join_project, receipt_state, explicit_version,
):
    root, record, _ = join_project
    version = save_version(root)
    record["plan"]["listen_joins"] = []
    record["plan"]["pieces"] = record["plan"]["pieces"][:1]
    record["status"] = "failed" if receipt_state == "failed" else "verified"
    if receipt_state == "take-changed":
        record["inputs"]["take:t1"] = "a" * 64
    receipt = root / "cut/receipt.json"
    if receipt_state == "revised":
        _cut(root, label="revised cut", segments=("s1",))
        newer = save_version(root)
        assert newer.media["sha256"] != version.media["sha256"]
        assert listens(root, newer.media_path) == []
    elif receipt_state == "missing":
        receipt.unlink()
    else:
        _json(receipt, record)
    items = findings_for_media(
        version.media_path, version.media["sha256"], project_dir=root,
        version_id=version.id if explicit_version else None,
    )["items"]
    assert [(item["start_frame"], item["end_frame"]) for item in items] == [(0, 85), (35, 120)]


def test_saved_version_without_listens_does_not_gain_current_joins(join_project):
    root, record, _ = join_project
    record["plan"]["listen_joins"] = []
    _json(root / "cut/receipt.json", record)
    version = save_version(root)
    record["plan"]["listen_joins"] = [{"frame": 40, "left": "s1", "right": "s2"}]
    _json(root / "cut/receipt.json", record)
    assert listens(root, version.media_path) == []


@pytest.mark.parametrize("damage", ["hash", "shape"])
def test_saved_join_snapshot_must_remain_valid(join_project, damage):
    root, _, _ = join_project
    version = save_version(root)
    path = version.path / "cut-pieces.json"
    pieces = json.loads(path.read_text())
    pieces["listen_joins"] = "invalid" if damage == "shape" else []
    _json(path, pieces)
    if damage == "shape":
        manifest = version.to_dict()
        manifest["snapshots"]["cut_pieces"]["sha256"] = sha256_file(path)
        _json(version.path / "manifest.json", manifest)
    assert listens(root, version.media_path) == []


@pytest.mark.parametrize("damage", ["none", "failed", "foreign", "malformed"])
def test_absent_or_untrusted_joins_show_no_listens(join_project, damage):
    root, record, _ = join_project
    if damage == "none":
        record["plan"]["listen_joins"] = []
    elif damage == "failed":
        record["status"] = "failed"
    elif damage == "foreign":
        record["outputs"][0]["sha256"] = "a" * 64
    else:
        record["plan"]["listen_joins"] = [{"frame": True}, {"frame": 999}]
    _json(root / "cut/receipt.json", record)
    assert listens(root, root / "cut/cut.mp4") == []


@pytest.mark.parametrize("damage", ["take", "snapshot", "ambiguous"])
def test_untrusted_or_ambiguous_saved_mapping_is_omitted(join_project, damage):
    root, record, _ = join_project
    version = save_legacy_version(root)
    if damage == "take":
        record["inputs"]["take:t1"] = "a" * 64
    elif damage == "snapshot":
        _json(version.path / "cut-pieces.json", {})
    else:
        pieces = json.loads((version.path / "cut-pieces.json").read_text())
        # The same source edges occur twice, so neither destination can be assumed.
        pieces["pieces"].append({**pieces["pieces"][0], "out_f0": 120})
        pieces["pieces"].append({**pieces["pieces"][1], "out_f0": 160})
        pieces["total_frames"] = 200
        _json(version.path / "cut-pieces.json", pieces)
        manifest = version.to_dict()
        manifest["snapshots"]["cut_pieces"]["sha256"] = sha256_file(
            version.path / "cut-pieces.json",
        )
        _json(version.path / "manifest.json", manifest)
    _json(root / "cut/receipt.json", record)
    items = listens(root, version.media_path)
    assert len(items) == (1 if damage == "ambiguous" else 0)


@pytest.mark.parametrize("duration", [4, 5])
def test_unsaved_final_uses_its_verified_cut_clock(join_project, duration):
    root, record, _ = join_project
    _json(root / "final/receipt.json", {
        "schema": "yap-studio/stage-receipt@1", "status": "verified",
        "inputs": {"media:cut/cut.mp4": record["outputs"][0]["sha256"]},
        "outputs": [{"path": "final/final.mp4", "sha256": "b" * 64}],
        "timeline": {"duration_s": duration},
    })
    items = findings_for_media(root / "final/final.mp4", "b" * 64, project_dir=root)["items"]
    assert len(items) == 2
    assert Fraction(items[1]["end_frame"], 30) == min(duration, Fraction(25, 6))
