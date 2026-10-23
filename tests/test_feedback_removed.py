"""Removed ranges use saved pieces, removal decisions, and matching source words."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from yap_studio.errors import InputError
from yap_studio.feedback.removed import removed_ranges
from yap_studio.feedback.versions import VersionManifest
from yap_studio.receipt import sha256_file


def _source_words(root: Path, texts: tuple[str, ...] = ("um", "well", "kept", "side", "extra")):
    path = root / "ingest" / "t1" / "words.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": "yap-studio/words@1", "take": "t1", "take_sha256": "b" * 64,
        "time_base": "source", "audio_offset_s": 0, "engine": "none", "model": None,
        "transcript_sha256": "c" * 64,
        "words": [{"id": f"t1:{index}", "text": text, "start": start, "end": end}
                  for index, (text, (start, end)) in enumerate(zip(
                      texts, ((0.9, 1.1), (1.4, 1.6), (1.9, 2.1), (3.9, 4.1), (5.5, 5.7)),
                      strict=True,
                  ))],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _snapshot(manifest: VersionManifest, key: str, data: dict) -> None:
    path = manifest.path / manifest.snapshots[key]["file"]
    path.write_text(json.dumps(data), encoding="utf-8")
    manifest.snapshots[key]["sha256"] = sha256_file(path)


def _synthetic_version(root: Path) -> VersionManifest:
    directory = root / "versions" / "v1"
    directory.mkdir(parents=True)
    words_path = _source_words(root)
    manifest = VersionManifest(directory, "sample-video", "v1", None,
                               "2026-10-06T12:00:00Z", None, {}, None, {
                                   "cut_pieces": {"file": "cut-pieces.json"},
                                   "removals": {"file": "removals.json"},
                               })
    pieces = [
        ("s1", "t1", "0", "1"), ("s1", "t1", "2", "3"),
        ("s2", "t1", "5", "6"), ("s2", "t1", "4", "9/2"),
        ("s2", "t1", "451/100", "551/100"), ("s2", "t1", "6", "7"),
        ("s3", "t2", "9", "10"),
    ]
    _snapshot(manifest, "cut_pieces", {
        "schema": "yap-studio/version-pieces@1", "fps": "30000/1001", "total_frames": 210,
        "inputs": {"words:t1": sha256_file(words_path)},
        "pieces": [{"segment": segment, "take": take, "out_f0": index * 30,
                    "n_frames": 30, "src_t0": start, "src_t1": end}
                   for index, (segment, take, start, end) in enumerate(pieces)],
    })
    _snapshot(manifest, "removals", {"schema": "yap-studio/removals@1", "removals": [
        {"id": "r1", "take": "t1", "start": 1, "end": 2,
         "kind": "filler", "status": "applied"},
        {"id": "r2", "take": "t1", "start": 1.5, "end": 1.75,
         "kind": "manual", "status": "applied"},
        {"id": "r3", "take": "t1", "start": 3, "end": 5,
         "kind": "silence", "status": "proposed"},
        {"id": "r4", "take": "t2", "start": 3, "end": 5,
         "kind": "restart", "status": "applied"},
        {"id": "r5", "take": "t1", "start": 2, "end": 3,
         "kind": "stutter", "status": "applied"},
        {"id": "r6", "take": "t1", "start": 3, "end": 5,
         "kind": "silence", "status": "rejected"},
    ]})
    return manifest


def test_forward_gaps_kinds_joins_and_word_midpoints(tmp_path: Path) -> None:
    manifest = _synthetic_version(tmp_path)
    ranges = removed_ranges(tmp_path, manifest)
    assert [item["id"] for item in ranges] == ["s1.0", "s1.1", "s2.4"]
    filler, story, trim = ranges
    assert filler == {
        "id": "s1.0", "take": "t1", "start": 1.0, "end": 2.0,
        "at_s": float(Fraction(30) / Fraction("30000/1001")), "at_frame": 30,
        "kind": "filler", "removal_ids": ["r1", "r2"], "words_known": True,
        "words": [{"id": "t1:0", "text": "um"}, {"id": "t1:1", "text": "well"}],
        "text": "um well",
    }
    assert story["kind"] == "story" and story["removal_ids"] == []
    assert story["start"] == 3.0 and story["end"] == 5.0 and story["text"] == "side"
    assert trim["kind"] == "trim" and trim["start"] == float(Fraction("551/100"))
    assert trim["text"] == "extra"
    with_short = removed_ranges(tmp_path, manifest, min_s=0)
    assert [item["id"] for item in with_short] == ["s1.0", "s1.1", "s2.3", "s2.4"]
    assert with_short[-1] == trim


@pytest.mark.parametrize("kind", ["filler", "silence", "stutter", "restart", "manual"])
def test_applied_kind_wins_even_between_segments(tmp_path: Path, kind: str) -> None:
    manifest = _synthetic_version(tmp_path)
    _snapshot(manifest, "removals", {"schema": "yap-studio/removals@1", "removals": [
        {"id": "cut", "take": "t1", "start": 3, "end": 5,
         "kind": kind, "status": "applied"},
    ]})
    story = removed_ranges(tmp_path, manifest)[1]
    assert story["kind"] == kind and story["removal_ids"] == ["cut"]


@pytest.mark.parametrize("state", ["changed", "missing", "unrecorded"])
def test_words_are_unknown_without_matching_hash(tmp_path: Path, state: str) -> None:
    manifest = _synthetic_version(tmp_path)
    path = tmp_path / "ingest" / "t1" / "words.json"
    if state == "changed":
        path.write_text("invalid changed words", encoding="utf-8")
    elif state == "missing":
        path.unlink()
    else:
        data = json.loads((manifest.path / "cut-pieces.json").read_text())
        del data["inputs"]
        _snapshot(manifest, "cut_pieces", data)
    for item in removed_ranges(tmp_path, manifest):
        assert item["words"] == [] and item["text"] is None and not item["words_known"]


def test_missing_removals_and_known_empty_words(tmp_path: Path) -> None:
    manifest = _synthetic_version(tmp_path)
    manifest.snapshots["removals"] = None
    data = json.loads((manifest.path / "cut-pieces.json").read_text())
    data["pieces"][1]["src_t0"] = "11/10"
    _snapshot(manifest, "cut_pieces", data)
    item = removed_ranges(tmp_path, manifest)[0]
    assert item["kind"] == "trim" and item["words_known"]
    # The midpoint at 1 belongs, even though that word starts before the gap.
    assert item["text"] == "um"
    path = tmp_path / "ingest" / "t1" / "words.json"
    words = json.loads(path.read_text())
    words["words"] = []
    path.write_text(json.dumps(words), encoding="utf-8")
    data["inputs"]["words:t1"] = sha256_file(path)
    _snapshot(manifest, "cut_pieces", data)
    item = removed_ranges(tmp_path, manifest)[0]
    assert item["words_known"] and item["text"] is None and item["words"] == []


@pytest.mark.parametrize("count", [0, 1])
def test_fewer_than_two_pieces_has_no_ranges(tmp_path: Path, count: int) -> None:
    manifest = _synthetic_version(tmp_path)
    data = json.loads((manifest.path / "cut-pieces.json").read_text())
    data["pieces"] = data["pieces"][:count]
    _snapshot(manifest, "cut_pieces", data)
    assert removed_ranges(tmp_path, manifest) == []


def test_changed_snapshot_is_refused(tmp_path: Path) -> None:
    manifest = _synthetic_version(tmp_path)
    (manifest.path / "removals.json").write_text("{}", encoding="utf-8")
    with pytest.raises(InputError, match="snapshot changed"):
        removed_ranges(tmp_path, manifest)
