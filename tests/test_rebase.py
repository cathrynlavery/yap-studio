"""Re-anchor authored project files after source word ids change."""

from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import synth
from test_cut import make_project as make_cut_project
from test_cut import write_edit
from test_ingest import FakeBackend, make_project
from test_ingest import backend as backend
from test_timeline import insert, write
from test_timeline import project as project

from yap_studio import cli, receipt
from yap_studio.errors import InputError, YapError
from yap_studio.plan import snap_frame
from yap_studio.project.lock import ConflictError
from yap_studio.project.model import Word, Words, load_cut_words
from yap_studio.project.receipts import StageRun
from yap_studio.project.status import ProjectContext, cut_inputs, ingest_inputs
from yap_studio.stages import rebase as stage
from yap_studio.stages.cut import cut, resolve_cut
from yap_studio.stages.fillers import fillers
from yap_studio.stages.ingest import ingest
from yap_studio.stages.timeline import resolve_timeline
from yap_studio.transcript import Word as SpeechWord


@pytest.fixture
def changed(project: Path) -> Path:
    root = project
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"] = [
        {"id": "s1", "take": "t1", "from": {"word": "t1:0", "text": "ALPHA"},
         "to": {"word": "t1:5", "offset_s": 0.01}, "text": "alpha beta gamma delta epsilon"},
        {"id": "s2", "take": "t1", "from": {"word": "t1:6"},
         "to": {"word": "t1:10"}, "note": "second"},
    ]
    write(root / "edit.json", edit)
    write(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 1.2, "end": 1.6,
             "kind": "filler", "status": "applied", "words": ["t1:1"],
             "evidence": {"refined_source_words": [
                 {"id": "t1:1", "text": "um", "start": 1.2, "end": 1.6}]}},
            {"id": "r2", "take": "t1", "start": 5.1, "end": 5.2,
             "kind": "manual", "status": "applied", "by": "manual", "words": ["t1:5"]},
        ],
    })
    write(root / "timeline/broll.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": edit["words_sha256"],
        "elements": [insert(root)],
    })
    before = (root / "ingest/t1/words.json").read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    snapshot = root / f"ingest/t1/previous/{digest}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(before)
    words = json.loads(before)
    words["words"].insert(0, {"id": "t1:0", "text": "new", "start": 0.01, "end": 0.08})
    for number, word in enumerate(words["words"][1:], 1):
        word["id"] = f"t1:{number}"
    write(root / "ingest/t1/words.json", words)
    return root


def test_rebase_remaps_edit_removals_and_broll(changed: Path) -> None:
    report = stage.rebase_edit(changed)
    assert report["status"] == "rebased"
    edit = json.loads((changed / "edit.json").read_bytes())
    assert edit["segments"][0]["from"] == {"word": "t1:1", "text": "ALPHA"}
    assert edit["segments"][0]["to"] == {"word": "t1:6", "offset_s": 0.01}
    assert edit["segments"][1]["note"] == "second"
    assert edit["words_sha256"]["t1"] == hashlib.sha256(
        (changed / "ingest/t1/words.json").read_bytes()).hexdigest()
    removals = json.loads((changed / "removals.json").read_bytes())["removals"]
    assert removals[0]["words"] == ["t1:2"]
    assert removals[0]["evidence"]["refined_source_words"][0]["id"] == "t1:2"
    assert removals[0]["start"] == 1.2
    assert removals[1]["words"] == ["t1:6"]
    lane = json.loads((changed / "timeline/broll.json").read_bytes())
    assert lane["elements"][0]["start"]["word"] == "t1:3"
    assert lane["elements"][0]["end"]["word"] == "t1:4"
    assert stage.rebase_edit(changed)["status"] == "unchanged"


def test_repeated_old_words_cannot_share_one_new_word(project: Path) -> None:
    root = project
    words_path = root / "ingest/t1/words.json"
    old = json.loads(words_path.read_bytes())
    old["words"] = [
        {"id": "t1:0", "text": "the", "start": 1.0, "end": 1.05},
        {"id": "t1:1", "text": "the", "start": 1.15, "end": 1.2},
    ]
    original = write(words_path, old).read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    write(root / "edit.json", {
        "schema": "yap-studio/edit@1", "words_sha256": {"t1": digest},
        "segments": [{"id": "s1", "take": "t1", "from": {"word": "t1:0"},
                      "to": {"word": "t1:1"}}],
    })
    snapshot = root / f"ingest/t1/previous/{digest}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(original)
    old["words"] = [{"id": "t1:0", "text": "the", "start": 1.06, "end": 1.11}]
    write(words_path, old)
    report = stage.rebase_edit(root)
    assert report["status"] == "unmatched"
    mapping = report["files"]["edit.json"]["mapping"]
    targets = [item["new"] for item in mapping if item.get("new")]
    assert len(targets) == len(set(targets)) == 1
    assert {item["anchor"] for item in report["unmatched"]} == {"to"}


def _word_sequence(rows: list[tuple[str, float]]) -> Words:
    return Words(take="t1", take_sha256="0" * 64, time_base="source",
                 audio_offset_s=0, engine="none", model=None,
                 transcript_sha256="1" * 64,
                 words=tuple(Word(f"t1:{index}", token, start, start + 0.05)
                             for index, (token, start) in enumerate(rows)))


def test_alignment_prefers_nearer_repeated_word() -> None:
    old = _word_sequence([("the", 1.0), ("the", 1.25)])
    new = _word_sequence([("the", 1.26)])
    assert {source: target.id for source, target in stage._alignment(old, new).items()} == {
        "t1:1": "t1:0"}


def test_alignment_leaves_equal_optimal_choice_unmatched() -> None:
    old = _word_sequence([("the", 1.0)])
    new = _word_sequence([("the", 0.9), ("the", 1.1)])
    assert stage._alignment(old, new) == {}


def test_alignment_handles_long_inserted_transcript() -> None:
    old = _word_sequence([(f"token{index}", index * 0.4) for index in range(2000)])
    new = _word_sequence([("found", 0.1), *[(f"token{index}", index * 0.4)
                                             for index in range(2000)]])
    mapping = stage._alignment(old, new)
    assert len(mapping) == 2000
    assert all(mapping[f"t1:{index}"].id == f"t1:{index + 1}"
               for index in range(2000))


@pytest.mark.parametrize("snapshot_present", [True, False])
def test_updated_edit_hash_recovers_or_reports_stale_removal(
    changed: Path, snapshot_present: bool,
) -> None:
    root = changed
    current_hash = receipt.sha256_file(root / "ingest/t1/words.json")
    edit = json.loads((root / "edit.json").read_bytes())
    edit["words_sha256"]["t1"] = current_hash
    for segment in edit["segments"]:
        segment["from"], segment["to"] = {"time": 0.2}, {"time": 10.5}
    write(root / "edit.json", edit)
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    lane["words_sha256"]["t1"] = current_hash
    lane["elements"] = []
    write(root / "timeline/broll.json", lane)
    removals = json.loads((root / "removals.json").read_bytes())
    removals["removals"][1]["words"] = ["t1:6"]
    write(root / "removals.json", removals)
    if not snapshot_present:
        snapshot, = (root / "ingest/t1/previous").glob("*.json")
        snapshot.rename(snapshot.with_suffix(".missing"))
    report = stage.rebase_edit(root)
    if snapshot_present:
        assert report["status"] == "rebased"
        updated = json.loads((root / "removals.json").read_bytes())["removals"]
        assert updated[0]["words"] == ["t1:2"]
        assert updated[0]["evidence"]["refined_source_words"][0]["id"] == "t1:2"
        assert updated[1] == removals["removals"][1]
    else:
        assert report["status"] == "unmatched"
        assert any(item.get("removal") == "r1" for item in report["unmatched"])
        assert json.loads((root / "removals.json").read_bytes()) == removals


def test_stale_removal_without_snapshot_drops_ids_when_pinned(changed: Path) -> None:
    root = changed
    edit = json.loads((root / "edit.json").read_bytes())
    edit["words_sha256"]["t1"] = receipt.sha256_file(root / "ingest/t1/words.json")
    edit["segments"] = [{"id": "s1", "take": "t1", "from": {"time": 0.2},
                         "to": {"time": 9.5}}]
    write(root / "edit.json", edit)
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    lane["words_sha256"] = edit["words_sha256"]
    lane["elements"] = []
    write(root / "timeline/broll.json", lane)
    snapshot, = (root / "ingest/t1/previous").glob("*.json")
    snapshot.rename(snapshot.with_suffix(".missing"))
    report = stage.rebase_edit(root, pin_unmatched=True)
    assert report["status"] == "rebased"
    assert any(item["removal"] == "r1" for item in report["pinned"])
    removal = json.loads((root / "removals.json").read_bytes())["removals"][0]
    assert removal["words"] == []
    assert removal["evidence"]["refined_source_words"] == []


def test_stale_removal_prefers_snapshot_matching_evidence_text(changed: Path) -> None:
    root = changed
    current_hash = receipt.sha256_file(root / "ingest/t1/words.json")
    edit = json.loads((root / "edit.json").read_bytes())
    edit["words_sha256"]["t1"] = current_hash
    edit["segments"] = [{"id": "s1", "take": "t1", "from": {"time": 0.2},
                         "to": {"time": 9.5}}]
    write(root / "edit.json", edit)
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    lane["words_sha256"] = edit["words_sha256"]
    lane["elements"] = []
    write(root / "timeline/broll.json", lane)
    original, = (root / "ingest/t1/previous").glob("*.json")
    competing = json.loads(original.read_bytes())
    competing["words"][1]["text"] = "hmm"
    payload = (json.dumps(competing) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    (root / f"ingest/t1/previous/{digest}.json").write_bytes(payload)
    report = stage.rebase_edit(root)
    assert report["status"] == "rebased"
    assert json.loads((root / "removals.json").read_bytes())["removals"][0]["words"] == [
        "t1:2"]


@pytest.mark.parametrize("pin_unmatched", [False, True])
def test_stale_removal_refuses_equally_scored_snapshots_with_different_targets(
    changed: Path, pin_unmatched: bool, capsys: pytest.CaptureFixture[str],
) -> None:
    root = changed
    words_path = root / "ingest/t1/words.json"
    current = json.loads(words_path.read_bytes())
    current["words"].insert(3, {"id": "t1:3", "text": "um", "start": 1.5,
                                "end": 1.6})
    for index, word in enumerate(current["words"]):
        word["id"] = f"t1:{index}"
    write(words_path, current)

    current_hash = receipt.sha256_file(words_path)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["words_sha256"]["t1"] = current_hash
    edit["segments"] = [{"id": "s1", "take": "t1", "from": {"time": 0.2},
                         "to": {"time": 9.5}}]
    write(root / "edit.json", edit)
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    lane["words_sha256"] = edit["words_sha256"]
    lane["elements"] = []
    write(root / "timeline/broll.json", lane)

    original, = (root / "ingest/t1/previous").glob("*.json")
    competing = json.loads(original.read_bytes())
    competing["words"][1]["start"] = 1.5
    competing["words"][1]["end"] = 1.6
    payload = (json.dumps(competing) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    (original.parent / f"{digest}.json").write_bytes(payload)

    now = stage.parse_document("words", current, filename=str(words_path))
    candidates = [stage.parse_document("words", json.loads(path.read_bytes()),
                                       filename=str(path))
                  for path in (original, original.parent / f"{digest}.json")]
    assert [stage._alignment(old, now)["t1:1"].id for old in candidates] == [
        "t1:2", "t1:3"]

    authored = [root / name for name in ("edit.json", "removals.json",
                                        "timeline/broll.json")]
    before = {path: path.read_bytes() for path in authored}
    report = stage.rebase_edit(root, pin_unmatched=pin_unmatched)
    assert report["status"] == "unmatched"
    assert {path: path.read_bytes() for path in authored} == before
    assert any(issue.get("removal") == "r1" and "ambigu" in str(issue).lower()
               for issue in report["unmatched"])

    assert cli.main(["edit", "rebase", str(root)]) == 1
    output = capsys.readouterr().out
    assert "r1" in output
    assert "ambigu" in output.lower()


def test_stale_removal_accepts_equally_scored_snapshots_with_same_target(
    changed: Path,
) -> None:
    root = changed
    current_hash = receipt.sha256_file(root / "ingest/t1/words.json")
    edit = json.loads((root / "edit.json").read_bytes())
    edit["words_sha256"]["t1"] = current_hash
    edit["segments"] = [{"id": "s1", "take": "t1", "from": {"time": 0.2},
                         "to": {"time": 9.5}}]
    write(root / "edit.json", edit)
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    lane["words_sha256"] = edit["words_sha256"]
    lane["elements"] = []
    write(root / "timeline/broll.json", lane)

    original, = (root / "ingest/t1/previous").glob("*.json")
    competing = json.loads(original.read_bytes())
    competing["words"][0]["text"] = "different"
    payload = (json.dumps(competing) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    second = original.parent / f"{digest}.json"
    second.write_bytes(payload)

    now = stage.parse_document("words", json.loads(
        (root / "ingest/t1/words.json").read_bytes()), filename="words.json")
    candidates = [stage.parse_document("words", json.loads(path.read_bytes()),
                                       filename=str(path)) for path in (original, second)]
    assert [stage._alignment(old, now)["t1:1"].id for old in candidates] == [
        "t1:2", "t1:2"]

    report = stage.rebase_edit(root)
    assert report["status"] == "rebased"
    removal = json.loads((root / "removals.json").read_bytes())["removals"][0]
    assert removal["words"] == ["t1:2"]
    assert removal["evidence"]["refined_source_words"][0]["id"] == "t1:2"


def test_unreferenced_hash_needs_no_snapshot(changed: Path) -> None:
    root = changed
    old = json.loads((root / "ingest/t1/words.json").read_bytes())
    old["take"] = "t2"
    for word in old["words"]:
        word["id"] = word["id"].replace("t1:", "t2:")
    prior = write(root / "ingest/t2/words.json", old)
    old_hash = hashlib.sha256(prior.read_bytes()).hexdigest()
    old["words"][0]["text"] = "changed"
    write(prior, old)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["words_sha256"]["t2"] = old_hash
    write(root / "edit.json", edit)
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    lane["words_sha256"]["t2"] = old_hash
    write(root / "timeline/broll.json", lane)
    assert not (root / "ingest/t2/previous").exists()
    report = stage.rebase_edit(root)
    assert report["status"] == "rebased" and not report["missing_snapshots"]
    updated = json.loads((root / "edit.json").read_bytes())
    assert updated["words_sha256"]["t2"] == hashlib.sha256(prior.read_bytes()).hexdigest()
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    assert lane["words_sha256"]["t2"] == updated["words_sha256"]["t2"]


def test_removal_without_edit_hash_is_left_untouched(changed: Path) -> None:
    root = changed
    document = json.loads((root / "removals.json").read_bytes())
    extra = {"id": "r3", "take": "t2", "start": 0.2, "end": 0.4,
             "kind": "manual", "status": "applied", "words": ["t2:7"]}
    document["removals"].append(extra)
    write(root / "removals.json", document)
    report = stage.rebase_edit(root)
    assert report["status"] == "rebased" and not report["missing_snapshots"]
    assert json.loads((root / "removals.json").read_bytes())["removals"][-1] == extra


@pytest.mark.parametrize("busy", ["running", "journaled"])
def test_fillers_writer_refuses_before_planning(
    changed: Path, busy: str, capsys: pytest.CaptureFixture[str],
) -> None:
    if busy == "running":
        write(changed / "fillers/progress.json", {"pid": os.getpid()})
        expected = "stage_running"
    else:
        write(changed / "fillers/receipt.pending.json", {})
        expected = "unfinished stage journal"
    before = (changed / "edit.json").read_bytes()
    with pytest.raises((YapError, InputError), match=(
            expected if busy == "journaled" else "fillers is already running")):
        stage.rebase_edit(changed)
    assert cli.main(["edit", "rebase", str(changed), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    if busy == "running":
        assert error["code"] == expected
    else:
        assert expected in error["message"]
    assert (changed / "edit.json").read_bytes() == before


def test_new_neighbour_reports_nominal_edge_shift(
    changed: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    report = stage.rebase_edit(changed, dry_run=True)
    assert report["edges_compared"] == "dry run"
    assert any(item["segment"] == "s1" and item["anchor"] == "from"
               and item["new_s"] - item["old_s"] >= 1 / 30
               and item["estimated"]
               for item in report["shifted"])
    assert cli.main(["edit", "rebase", str(changed), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "estimated shift: s1 from" in output
    assert "Edges are checked on a real run" in output


def test_distant_new_word_does_not_shift_edges(changed: Path) -> None:
    words = json.loads((changed / "ingest/t1/words.json").read_bytes())
    words["words"][0].update(start=12.5, end=12.6)
    write(changed / "ingest/t1/words.json", words)
    assert stage.rebase_edit(changed)["shifted"] == []


def _resolved_change(root: Path, *, synthetic_receipt: bool = True,
                     guarded: bool = False) -> tuple[Path, dict]:
    """Keep a true old cut plan and insert a non-anchor word beside one edge."""
    make_cut_project(root, [synth.calib_source(root / "source.mov", seconds=7)],
                     word_rows=[[('before', 0.2, 0.4), ('alpha', 0.8, 1.0),
                                 ('beta', 1.4, 1.6), ('gamma', 4.0, 4.2),
                                 ('delta', 4.5, 4.7)]])
    first_edge = {"word": "t1:1"}
    if guarded:
        first_edge["text"] = "alpha"
    write_edit(root, [
        {"id": "s1", "take": "t1", "from": first_edge,
         "to": {"word": "t1:2", "offset_s": 0.01}},
        {"id": "s2", "take": "t1", "from": {"word": "t1:3", "offset_s": 0.01},
         "to": {"word": "t1:4", "offset_s": 0.01}},
    ])
    old_plan = resolve_cut(root).to_dict()
    if synthetic_receipt:
        write(root / "cut/receipt.json", {"inputs": cut_inputs(ProjectContext(root)),
                                          "plan": old_plan})
    before = (root / "ingest/t1/words.json").read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    snapshot = root / f"ingest/t1/previous/{digest}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(before)
    words = json.loads(before)
    words["words"].insert(1, {"id": "t1:1", "text": "found", "start": 0.58,
                              "end": 0.7})
    for number, word in enumerate(words["words"]):
        word["id"] = f"t1:{number}"
    write(root / "ingest/t1/words.json", words)
    return root, old_plan


def _publish_words(root: Path, payload: bytes) -> None:
    """Publish fixture words through a matching ingest receipt."""
    staged = root / "staged-rebase-words.json"
    staged.write_bytes(payload)
    inputs = ingest_inputs(ProjectContext(root))
    with StageRun(root, "ingest") as run:
        for name, digest in inputs.items():
            run.record_inputs(name, digest)
        run.replace_output(root / "ingest/t1/words.json", staged)
        run.finish()


def test_guarded_renumbered_word_keeps_old_cut_edge(tmp_path: Path) -> None:
    root, _ = _resolved_change(tmp_path, guarded=True)
    report = stage.rebase_edit(root)
    assert report["status"] == "rebased"
    assert report["edges_compared"] == "resolved"
    assert report["kept"] == [{"segment": "s1", "edge": "from", "time": 2 / 3,
                               "frames": 3}]


def test_resolved_cut_keeps_non_anchor_edge_and_untouched_segment(tmp_path: Path) -> None:
    root, old = _resolved_change(tmp_path)
    report = stage.rebase_edit(root)
    new = resolve_cut(root).to_dict()
    assert report["status"] == "rebased"
    assert report["edges_compared"] == "resolved"
    assert report["shifted"] == []
    assert report["kept"] == [{"segment": "s1", "edge": "from", "time": 2 / 3,
                               "frames": 3}]
    old_piece = next(item for item in old["pieces"] if item["segment_id"] == "s1")
    new_piece = next(item for item in new["pieces"] if item["segment_id"] == "s1")
    assert (new_piece["src_t0"], new_piece["src_t1"], new_piece["n_frames"]) == (
        old_piece["src_t0"], old_piece["src_t1"], old_piece["n_frames"])
    assert report["cut_change"] == {"frames_old": old["total_frames"],
                                    "frames_new": new["total_frames"]}


def test_pinned_edge_keeps_cut_removal_and_broll_frames(tmp_path: Path) -> None:
    root, _ = _resolved_change(tmp_path, synthetic_receipt=False)
    words_path = root / "ingest/t1/words.json"
    newer = words_path.read_bytes()
    old_snapshot, = (root / "ingest/t1/previous").glob("*.json")
    words_path.write_bytes(old_snapshot.read_bytes())
    digest = receipt.sha256_file(words_path)
    write(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 5.5, "end": 5.7,
             "kind": "manual", "status": "applied", "by": "manual", "words": []},
        ],
    })
    write(root / "timeline/broll.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": {"t1": digest},
        "elements": [{"id": "broll-one", "kind": "insert", "lane": "broll",
                      "start": {"word": "t1:1"}, "end": {"word": "t1:2"},
                      "media": {"path": "takes/t1.mov", "sha256": receipt.sha256_file(
                          root / "takes/t1.mov"), "in_s": 0, "speed": 1}}],
    })
    old = resolve_cut(root)
    cut(root)
    old_broll = resolve_timeline(root).elements[0]
    old_removal = json.loads((root / "removals.json").read_bytes())["removals"][0]
    _publish_words(root, newer)
    report = stage.rebase_edit(root)
    assert report["status"] == "rebased"
    assert report["kept"] and report["shifted"] == []
    recut = resolve_cut(root)
    assert [(piece.segment_id, piece.src_f0, piece.src_f1, piece.out_f0, piece.n_frames)
            for piece in recut.pieces] == [
                (piece.segment_id, piece.src_f0, piece.src_f1, piece.out_f0, piece.n_frames)
                for piece in old.pieces]
    assert recut.total_frames == old.total_frames
    removal = json.loads((root / "removals.json").read_bytes())["removals"][0]
    assert (removal["start"], removal["end"], removal["status"]) == (
        old_removal["start"], old_removal["end"], old_removal["status"])
    assert removal["words"] == []
    cut(root)
    new_broll = resolve_timeline(root).elements[0]
    assert (new_broll.start_frame, new_broll.end_frame) == (
        old_broll.start_frame, old_broll.end_frame)


def test_timeline_pin_resolves_on_reordered_cut_output_clock(tmp_path: Path) -> None:
    root, _ = _resolved_change(tmp_path, synthetic_receipt=False)
    words_path = root / "ingest/t1/words.json"
    newer = json.loads(words_path.read_bytes())
    newer["words"][3]["text"] = "altered"
    old_snapshot, = (root / "ingest/t1/previous").glob("*.json")
    words_path.write_bytes(old_snapshot.read_bytes())
    digest = receipt.sha256_file(words_path)
    write(root / "timeline/broll.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": {"t1": digest},
        "elements": [{"id": "broll-one", "kind": "insert", "lane": "broll",
                      "start": {"word": "t1:1"},
                      "end": {"word": "t1:2", "text": "beta"},
                      "media": {"path": "takes/t1.mov", "sha256": receipt.sha256_file(
                          root / "takes/t1.mov"), "in_s": 0, "speed": 1}}],
    })
    cut(root)
    old_end = resolve_timeline(root).elements[0].end_frame
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"].reverse()
    write(root / "edit.json", edit)
    _publish_words(root, (json.dumps(newer) + "\n").encode())
    report = stage.rebase_edit(root, pin_unmatched=True)
    assert report["status"] == "rebased"
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    assert lane["elements"][0]["end"] == {"word": "t1:3"}
    assert any(item.get("new_word") == "t1:3" and item.get("new_text") == "altered"
               for item in report["pinned"])
    cut(root)
    updated = resolve_timeline(root).elements[0]
    word = next(item for item in load_cut_words(root / "cut/cut-words.json").words
                if item.id == "t1:3")
    assert updated.end_frame == snap_frame(Fraction(str(word.end)), Fraction(30))
    assert updated.end_frame != old_end


def test_resolved_cut_reports_found_word_inside_short_piece(tmp_path: Path) -> None:
    root = tmp_path
    make_cut_project(root, [synth.calib_source(root / "source.mov", seconds=7)],
                     word_rows=[[('alpha', 0.4, 0.6), ('later', 2.1, 2.3)]])
    write_edit(root, [
        {"id": "s1", "take": "t1", "from": {"time": 0.2}, "to": {"time": 1.0}},
        {"id": "s2", "take": "t1", "from": {"time": 1.0}, "to": {"time": 1.2}},
        {"id": "s3", "take": "t1", "from": {"time": 2.0}, "to": {"time": 2.7}},
    ])
    old = resolve_cut(root).to_dict()
    write(root / "cut/receipt.json", {"inputs": cut_inputs(ProjectContext(root)),
                                      "plan": old})
    words_path = root / "ingest/t1/words.json"
    words = json.loads(words_path.read_bytes())
    words["words"].insert(1, {"id": "t1:1", "text": "found", "start": 1.04,
                              "end": 1.10})
    words["words"][2]["id"] = "t1:2"
    write(words_path, words)
    report = stage.rebase_edit(root)
    assert report["status"] == "moved"
    assert report["edges_compared"] == "resolved"
    assert [item["segment"] for item in report["shifted"]] == ["s2"]
    second, = report["shifted"]
    assert second["pieces_old"] == 0 and second["pieces_new"] == 1
    assert report["kept"][0]["edge"] == "to"
    assert report["cut_change"]["frames_new"] != 51


def test_internal_source_piece_move_is_listed_with_unchanged_totals() -> None:
    old = {"pieces": [
        {"segment_id": "s1", "src_t0": str(start / 30), "src_t1": str(end / 30),
         "src_f0": start, "src_f1": end, "n_frames": 30}
        for start, end in ((0, 30), (60, 90), (120, 150))
    ]}
    new = json.loads(json.dumps(old))
    new["pieces"][1].update(src_t0="31/15", src_t1="46/15",
                             src_f0=62, src_f1=92)
    moved, = stage._cut_changes(old, new)
    assert moved["segment"] == "s1"
    assert moved["start_frames"] == moved["end_frames"] == 0
    assert moved["frames_old"] == moved["frames_new"] == 90
    assert moved["pieces_old"] == moved["pieces_new"] == 3
    assert moved["source_ranges_old"] == [[0, 30], [60, 90], [120, 150]]
    assert moved["source_ranges_new"] == [[0, 30], [62, 92], [120, 150]]


def test_unrestored_cut_reports_moved_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    root, old = _resolved_change(tmp_path)
    still_moved = json.loads(json.dumps(old))
    still_moved["pieces"][0]["src_f0"] += 2
    still_moved["pieces"][0]["src_t0"] = "4/5"
    still_moved["pieces"][0]["n_frames"] -= 2
    still_moved["total_frames"] -= 2
    monkeypatch.setattr(stage, "resolve_cut", lambda _root: SimpleNamespace(
        to_dict=lambda: still_moved))
    assert cli.main(["edit", "rebase", str(root), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "moved"
    assert report["shifted"][0]["segment"] == "s1"
    assert report["shifted"][0]["start_frames"] == 2
    assert report["cut_change"]["frames_old"] != report["cut_change"]["frames_new"]
    assert json.loads((root / "edit.json").read_bytes())["segments"][0]["from"]["time"] == (
        float(Fraction(old["pieces"][0]["src_t0"])))


def test_conflict_while_pin_writes_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager

    root, _ = _resolved_change(tmp_path)
    original = stage.project_lock
    locks = 0

    @contextmanager
    def changing_lock(project_root: Path):
        nonlocal locks
        with original(project_root):
            locks += 1
            if locks == 2:
                with (project_root / "edit.json").open("ab") as stream:
                    stream.write(b" ")
            yield

    monkeypatch.setattr(stage, "project_lock", changing_lock)
    with pytest.raises(ConflictError):
        stage.rebase_edit(root)


def test_edit_writer_between_rebase_and_pin_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _resolved_change(tmp_path)
    original_resolve = stage.resolve_cut
    external: bytes | None = None

    def concurrent_edit(project_root: Path):
        nonlocal external
        if external is None:
            document = json.loads((project_root / "edit.json").read_bytes())
            document["segments"][0]["note"] = "concurrent writer"
            external = write(project_root / "edit.json", document).read_bytes()
        return original_resolve(project_root)

    monkeypatch.setattr(stage, "resolve_cut", concurrent_edit)
    with pytest.raises(ConflictError):
        stage.rebase_edit(root)
    assert external is not None
    assert (root / "edit.json").read_bytes() == external


def test_old_cut_receipt_with_changed_removal_is_not_used_for_pinning(
    tmp_path: Path,
) -> None:
    root, _ = _resolved_change(tmp_path)
    removal = {"id": "r1", "take": "t1", "start": 4.0, "end": 4.2,
               "kind": "manual", "status": "applied", "by": "manual", "words": []}
    write(root / "removals.json", {"schema": "yap-studio/removals@1",
                                   "removals": [removal]})
    receipt_path = root / "cut/receipt.json"
    record = json.loads(receipt_path.read_bytes())
    record["inputs"]["removals:applied"] = cut_inputs(
        ProjectContext(root))["removals:applied"]
    write(receipt_path, record)
    removal["start"] = 4.1
    write(root / "removals.json", {"schema": "yap-studio/removals@1",
                                   "removals": [removal]})
    report = stage.rebase_edit(root)
    assert report["status"] == "rebased"
    assert report["edges_compared"] == (
        "cut is older than other changes; run cut before rebase")
    assert report["kept"] == []
    assert all(item["estimated"] for item in report["shifted"])
    assert json.loads((root / "edit.json").read_bytes())["segments"][0]["from"] == {
        "word": "t1:2"}


def test_resolved_cli_reports_kept_edge_and_cut(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    text_root = tmp_path / "text"
    text_root.mkdir()
    _resolved_change(text_root)
    assert cli.main(["edit", "rebase", str(text_root)]) == 0
    output = capsys.readouterr().out
    assert "kept: s1 start" in output
    assert "cut: " in output
    json_root = tmp_path / "json"
    json_root.mkdir()
    _resolved_change(json_root)
    assert cli.main(["edit", "rebase", str(json_root), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["edges_compared"] == "resolved"
    assert report["shifted"] == []
    assert report["kept"][0]["segment"] == "s1"


def test_wrong_old_receipt_uses_estimate(changed: Path) -> None:
    report = stage.rebase_edit(changed)
    assert report["edges_compared"] == "no cut for the old edit"
    assert report["kept"] == []
    assert report["shifted"]
    assert all(item["estimated"] for item in report["shifted"])


def _all_rebase_outputs(root: Path) -> dict[str, bytes]:
    """Add authored references and a receipt that rebase must restore together."""
    digest = json.loads((root / "edit.json").read_bytes())["words_sha256"]["t1"]
    write(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 4.0, "end": 4.2,
             "kind": "filler", "status": "rejected", "words": ["t1:3"]},
        ],
    })
    write(root / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": {"t1": digest},
        "elements": [{"id": "title-one", "kind": "title", "lane": "agent",
                      "start": {"word": "t1:3"}, "end": {"word": "t1:4"},
                      "lines": ["Title"]}],
    })
    write(root / "timeline/broll.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": {"t1": digest},
        "elements": [{"id": "broll-one", "kind": "insert", "lane": "broll",
                      "start": {"word": "t1:1"}, "end": {"word": "t1:2"},
                      "media": {"path": "takes/t1.mov", "sha256": receipt.sha256_file(
                          root / "takes/t1.mov"), "in_s": 0, "speed": 1}}],
    })
    write(root / "fillers/receipt.json", {
        "schema": "yap-studio/stage-receipt@1", "stage": "fillers",
        "status": "verified", "started": "fixture", "inputs": {},
        "outputs": [], "asserts": [],
    })
    names = ("edit.json", "removals.json", "timeline/agent.json",
             "timeline/broll.json", "fillers/receipt.json")
    return {name: (root / name).read_bytes() for name in names}


@pytest.mark.parametrize("after_pin", [False, True])
def test_resolve_failure_restores_all_written_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_pin: bool,
) -> None:
    root, _ = _resolved_change(tmp_path)
    before = _all_rebase_outputs(root)
    original_resolve = stage.resolve_cut
    calls = 0

    def failure(project_root: Path):
        nonlocal calls
        calls += 1
        if not after_pin or calls == 2:
            raise InputError("edit.json", "no speech engine")
        return original_resolve(project_root)

    monkeypatch.setattr(stage, "resolve_cut", failure)
    report = stage.rebase_edit(root)
    assert report["status"] == "cut_failed"
    assert "no speech engine" in report["resolve_error"]
    assert set(report["restored"]) == set(before)
    assert {name: (root / name).read_bytes() for name in before} == before
    assert calls == (2 if after_pin else 1)


def test_cut_failure_cli_exits_one_and_reports_restoration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _resolved_change(tmp_path)
    before = (root / "edit.json").read_bytes()

    def failure(_root: Path):
        raise InputError("edit.json", "no speech engine")

    monkeypatch.setattr(stage, "resolve_cut", failure)
    assert cli.main(["edit", "rebase", str(root)]) == 1
    assert "error: the cut does not resolve after rebase: no speech engine; " \
           "restored edit.json" in capsys.readouterr().out
    assert (root / "edit.json").read_bytes() == before


def test_word_moving_before_time_start_restores_edit_after_resolve_error(
    tmp_path: Path,
) -> None:
    root = tmp_path
    make_cut_project(root, [synth.calib_source(root / "source.mov", seconds=3)],
                     word_rows=[[('alpha', 1.1, 1.25), ('beta', 2.0, 2.2)]])
    write_edit(root, [{"id": "s1", "take": "t1", "from": {"time": 1.0},
                       "to": {"word": "t1:0", "offset_s": 0.01}}])
    old_plan = resolve_cut(root).to_dict()
    write(root / "cut/receipt.json", {"inputs": cut_inputs(ProjectContext(root)),
                                      "plan": old_plan})
    words_path = root / "ingest/t1/words.json"
    before_words = words_path.read_bytes()
    digest = hashlib.sha256(before_words).hexdigest()
    snapshot = root / f"ingest/t1/previous/{digest}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(before_words)
    words = json.loads(before_words)
    words["words"][0].update(start=0.85, end=0.95)
    write(words_path, words)
    before_edit = (root / "edit.json").read_bytes()
    report = stage.rebase_edit(root)
    assert report["status"] == "cut_failed"
    assert report["restored"] == ["edit.json"]
    assert (root / "edit.json").read_bytes() == before_edit


def test_cut_failure_removes_new_fillers_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _resolved_change(tmp_path)
    write(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 4.0, "end": 4.2,
             "kind": "filler", "status": "rejected", "words": ["t1:3"]},
        ],
    })
    before = (root / "removals.json").read_bytes()
    assert not (root / "fillers/receipt.json").exists()

    def failure(_root: Path):
        raise InputError("edit.json", "no speech engine")

    monkeypatch.setattr(stage, "resolve_cut", failure)
    report = stage.rebase_edit(root)
    assert report["status"] == "cut_failed"
    assert "fillers/receipt.json" in report["restored"]
    assert (root / "removals.json").read_bytes() == before
    assert not (root / "fillers/receipt.json").exists()


def test_cut_failure_rollback_conflict_restores_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _resolved_change(tmp_path)
    before = _all_rebase_outputs(root)
    external: bytes | None = None

    def failure(project_root: Path):
        nonlocal external
        document = json.loads((project_root / "edit.json").read_bytes())
        document["segments"][0]["note"] = "concurrent writer"
        external = write(project_root / "edit.json", document).read_bytes()
        raise InputError("edit.json", "no speech engine")

    monkeypatch.setattr(stage, "resolve_cut", failure)
    with pytest.raises(ConflictError):
        stage.rebase_edit(root)
    assert external is not None
    assert (root / "edit.json").read_bytes() == external
    assert all((root / name).read_bytes() != original
               for name, original in before.items() if name != "edit.json")


def test_dry_run_writes_nothing(changed: Path) -> None:
    files = [changed / name for name in ("edit.json", "removals.json", "timeline/broll.json")]
    before = [path.read_bytes() for path in files]
    report = stage.rebase_edit(changed, dry_run=True)
    assert report["status"] == "dry_run"
    assert report["files"]["edit.json"]["reanchored"] == 4
    assert [path.read_bytes() for path in files] == before


def test_unmatched_refuses_then_pins(changed: Path) -> None:
    words = json.loads((changed / "ingest/t1/words.json").read_bytes())
    words["words"][3]["text"] = "different"
    write(changed / "ingest/t1/words.json", words)
    before = (changed / "edit.json").read_bytes()
    report = stage.rebase_edit(changed)
    assert report["status"] == "unmatched"
    assert (changed / "edit.json").read_bytes() == before
    assert any(item["anchor"] == "start" and item["element"] == "broll-one"
               and item["nearest"] for item in report["unmatched"])
    pinned = stage.rebase_edit(changed, pin_unmatched=True)
    assert pinned["status"] == "rebased"
    assert pinned["pinned"]
    lane = json.loads((changed / "timeline/broll.json").read_bytes())
    assert lane["elements"][0]["start"] == {"word": "t1:3"}
    assert any(item.get("new_word") == "t1:3" and item["text"] == "beta"
               and item["new_text"] == "different" for item in pinned["pinned"])


def test_transition_with_unmatched_word_cannot_be_pinned(changed: Path) -> None:
    root = changed
    edit = json.loads((root / "edit.json").read_bytes())
    write(root / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": edit["words_sha256"],
        "elements": [{"id": "join", "kind": "transition", "lane": "agent",
                      "start": {"word": "t1:2", "text": "beta"},
                      "look": "whip_pan"}],
    })
    words = json.loads((root / "ingest/t1/words.json").read_bytes())
    words["words"][3]["text"] = "changed"
    write(root / "ingest/t1/words.json", words)
    before = (root / "timeline/agent.json").read_bytes()
    report = stage.rebase_edit(root, pin_unmatched=True)
    assert report["status"] == "unmatched"
    assert (root / "timeline/agent.json").read_bytes() == before
    assert any(item["element"] == "join" and item["note"] ==
               "transition needs a word at a join; place it again"
               for item in report["unmatched"])


def test_timeline_pin_chooses_earlier_word_on_equal_distance(changed: Path) -> None:
    root = changed
    words = json.loads((root / "ingest/t1/words.json").read_bytes())
    words["words"][3].update(text="later", start=2.3, end=2.4)
    words["words"].insert(3, {"id": "t1:3", "text": "earlier",
                              "start": 2.1, "end": 2.15})
    for index, word in enumerate(words["words"]):
        word["id"] = f"t1:{index}"
    write(root / "ingest/t1/words.json", words)
    report = stage.rebase_edit(root, pin_unmatched=True)
    assert report["status"] == "rebased"
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    assert lane["elements"][0]["start"] == {"word": "t1:3"}
    assert any(item.get("new_word") == "t1:3" and item.get("new_text") == "earlier"
               for item in report["pinned"])


def test_guard_disagreement_never_moves(changed: Path) -> None:
    edit = json.loads((changed / "edit.json").read_bytes())
    edit["segments"][0]["from"]["text"] = "unrelated"
    write(changed / "edit.json", edit)
    report = stage.rebase_edit(changed)
    assert report["status"] == "unmatched"
    assert any(item["file"] == "edit.json" and item["anchor"] == "from"
               for item in report["unmatched"])


def test_lane_uses_its_own_recorded_hash(changed: Path) -> None:
    root = changed
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    lane["words_sha256"]["t1"] = hashlib.sha256(
        (root / "ingest/t1/words.json").read_bytes()).hexdigest()
    lane["elements"][0]["start"] = {"word": "t1:1", "text": "alpha"}
    lane["elements"][0]["end"] = {"word": "t1:3"}
    write(root / "timeline/broll.json", lane)
    before = (root / "ingest/t1/words.json").read_bytes()
    write(root / f"ingest/t1/previous/{hashlib.sha256(before).hexdigest()}.json",
          json.loads(before))
    words = json.loads(before)
    words["words"].insert(1, {"id": "t1:1", "text": "second", "start": 0.1, "end": 0.15})
    for index, word in enumerate(words["words"]):
        word["id"] = f"t1:{index}"
    write(root / "ingest/t1/words.json", words)
    assert stage.rebase_edit(root)["status"] == "rebased"
    lane = json.loads((root / "timeline/broll.json").read_bytes())
    assert lane["elements"][0]["start"]["word"] == "t1:2"
    assert lane["elements"][0]["end"]["word"] == "t1:4"


@pytest.mark.parametrize("slot", ["steps", "cues"])
def test_nested_graphic_step_anchor_moves(changed: Path, slot: str) -> None:
    root = changed
    edit = json.loads((root / "edit.json").read_bytes())
    slots = {slot: [{"at": {"word": "t1:6", "text": "zeta"}}]}
    if slot == "cues":
        slots[slot].insert(0, 0.2)
        slots["options"] = ["One", "Two"]
    write(root / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": edit["words_sha256"],
        "elements": [{"id": "agent-card", "kind": "graphic", "lane": "agent",
                      "start": {"time": 4.0}, "hold_s": 4,
                      "template": "choices" if slot == "cues" else "text_tag",
                      "slots": slots}],
    })
    assert stage.rebase_edit(root)["status"] == "rebased"
    lane = json.loads((root / "timeline/agent.json").read_bytes())
    expected = [{"at": {"word": "t1:7", "text": "zeta"}}]
    if slot == "cues":
        expected.insert(0, 0.2)
    assert lane["elements"][0]["slots"][slot] == expected


def test_pin_drops_unmatched_removal_ids(changed: Path) -> None:
    root = changed
    words = json.loads((root / "ingest/t1/words.json").read_bytes())
    words["words"][2]["text"] = "changed"
    write(root / "ingest/t1/words.json", words)
    assert stage.rebase_edit(root)["status"] == "unmatched"
    report = stage.rebase_edit(root, pin_unmatched=True)
    assert report["status"] == "rebased"
    assert any(item["anchor"] == "words[0]" for item in report["pinned"])
    item = json.loads((root / "removals.json").read_bytes())["removals"][0]
    assert item["words"] == []
    assert item["evidence"]["refined_source_words"] == []
    assert (item["start"], item["end"]) == (1.2, 1.6)


def test_missing_snapshot_refuses_all_writes(changed: Path) -> None:
    snapshot, = (changed / "ingest/t1/previous").glob("*.json")
    snapshot.rename(snapshot.with_suffix(".missing"))
    before = (changed / "edit.json").read_bytes()
    report = stage.rebase_edit(changed)
    assert report["status"] == "unmatched"
    assert any(item["file"] == "edit.json" and item["take"] == "t1"
               for item in report["missing_snapshots"])
    assert (changed / "edit.json").read_bytes() == before


def test_conflict_before_writes_changes_no_other_file(
    changed: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = stage.project_lock

    def concurrent_change(root: Path):
        from contextlib import contextmanager

        @contextmanager
        def changed_lock():
            with original(root):
                with (root / "timeline/broll.json").open("ab") as stream:
                    stream.write(b" ")
                yield

        return changed_lock()

    monkeypatch.setattr(stage, "project_lock", concurrent_change)
    before_edit = (changed / "edit.json").read_bytes()
    before_removals = (changed / "removals.json").read_bytes()
    with pytest.raises(ConflictError):
        stage.rebase_edit(changed)
    assert (changed / "edit.json").read_bytes() == before_edit
    assert (changed / "removals.json").read_bytes() == before_removals


def test_cli_text_json_and_exit_codes(
    changed: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["edit", "rebase", str(changed), "--dry-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry_run"
    assert cli.main(["edit", "rebase", str(changed)]) == 0
    assert "yap-studio cut" in capsys.readouterr().out
    assert cli.main(["edit", "rebase", str(changed)]) == 0
    assert "already matches" in capsys.readouterr().out


def test_cli_unmatched_and_input_error_codes(
    changed: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    edit = json.loads((changed / "edit.json").read_bytes())
    edit["segments"][0]["from"]["text"] = "wrong"
    write(changed / "edit.json", edit)
    assert cli.main(["edit", "rebase", str(changed), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "unmatched"
    assert cli.main(["edit", "rebase", str(changed / "missing"), "--json"]) == 2
    assert "error" in json.loads(capsys.readouterr().err)


def test_cli_conflict_exits_three(
    changed: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from contextlib import contextmanager

    @contextmanager
    def conflict(root: Path):
        raise ConflictError(root / "edit.json")
        yield

    monkeypatch.setattr(stage, "project_lock", conflict)
    assert cli.main(["edit", "rebase", str(changed), "--json"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "base_changed"


def test_reingest_rebase_keeps_cut_removals_and_broll(
    tmp_path: Path, backend: FakeBackend,
) -> None:
    root = make_project(tmp_path, [synth.calib_source(tmp_path / "take.mov", seconds=7)])
    project = json.loads((root / "project.json").read_bytes())
    project["asr"] = {"engine": "none"}
    project["style"] = "preset:clean_minimal"
    write(root / "project.json", project)
    ingest(root, asr="parakeet-mlx")
    digest = receipt.sha256_file(root / "ingest/t1/words.json")
    write(root / "edit.json", {
        "schema": "yap-studio/edit@1", "words_sha256": {"t1": digest},
        "segments": [
            {"id": "s1", "take": "t1", "from": {"word": "t1:0", "text": "here's",
                                                    "offset_s": 0.01},
             "to": {"word": "t1:2", "offset_s": 0.01}},
            {"id": "s2", "take": "t1", "from": {"word": "t1:3", "offset_s": 0.01},
             "to": {"word": "t1:5", "offset_s": 0.01}},
        ],
    })
    write(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 0.53, "end": 0.72,
             "kind": "filler", "status": "applied", "words": ["t1:1"],
             "evidence": {"refined_source_words": [
                 {"id": "t1:1", "text": "the", "start": 0.55, "end": 0.7}]}},
            {"id": "r2", "take": "t1", "start": 1.47, "end": 1.72,
             "kind": "manual", "status": "applied", "by": "manual", "words": ["t1:4"]},
            {"id": "r3", "take": "t1", "start": 0.76, "end": 0.8,
             "kind": "filler", "status": "rejected", "words": ["t1:2"]},
        ],
    })
    write(root / "timeline/broll.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": {"t1": digest},
        "elements": [{"id": "broll-one", "kind": "insert", "lane": "broll",
                      "start": {"word": "t1:3"}, "end": {"word": "t1:5"},
                      "media": {"path": "take.mov", "sha256": receipt.sha256_file(
                          root / "take.mov"), "in_s": 0, "speed": 1}}],
    })
    first = resolve_cut(root)
    cut(root)
    broll = resolve_timeline(root).elements[0]
    before = ([(piece.src_t0, piece.src_t1, piece.out_f0, piece.n_frames)
               for piece in first.pieces],
              [(word.id, word.text, word.start, word.end)
               for word in load_cut_words(root / "cut/cut-words.json").words],
              [(item["start"], item["end"]) for item in json.loads(
                  (root / "removals.json").read_bytes())["removals"]],
              (broll.start_frame, broll.end_frame))
    backend.words = [*[
        SpeechWord(word.text, word.start + (0.003 if index in {1, 4} else 0),
                   word.end + (0.003 if index in {1, 4} else 0), word.conf)
        for index, word in enumerate(backend.words)
    ]]
    backend.words.insert(2, SpeechWord("extra", 0.61, 0.63))
    backend.model = "fixture-v2"
    ingest(root, force=True, asr="parakeet-mlx")
    with pytest.raises(Exception, match="word snapshot changed"):
        resolve_cut(root)
    rebased = stage.rebase_edit(root)
    assert rebased["status"] == "rebased"
    assert rebased["shifted"] == []
    assert rebased["kept"] == []
    assert rebased["edges_compared"] == "resolved"
    second = resolve_cut(root)
    cut(root)
    broll = resolve_timeline(root).elements[0]
    after = ([(piece.src_t0, piece.src_t1, piece.out_f0, piece.n_frames)
              for piece in second.pieces],
             [(word.id, word.text, word.start, word.end)
              for word in load_cut_words(root / "cut/cut-words.json").words],
             [(item["start"], item["end"]) for item in json.loads(
                 (root / "removals.json").read_bytes())["removals"]],
             (broll.start_frame, broll.end_frame))
    assert after[0] == before[0]
    assert [(item[1], item[2], item[3]) for item in after[1]] == [
        (item[1], item[2], item[3]) for item in before[1]]
    assert after[2:] == before[2:]
    decisions = [(item["id"], item["kind"], item["status"], item["start"], item["end"],
                  tuple(item["words"]))
                 for item in json.loads((root / "removals.json").read_bytes())["removals"]]
    assert sorted((item["id"], item["kind"], item["status"], item["start"], item["end"],
                   tuple(item["words"]))
                  for item in fillers(root)["entries"]) == sorted(decisions)



def test_rebase_remaps_zoom_key_anchors(changed: Path) -> None:
    edit = json.loads((changed / "edit.json").read_bytes())
    write(changed / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": edit["words_sha256"],
        "elements": [{"id": "agent-camera", "kind": "zoom", "start": {"time": 0},
                      "end": {"time": 6}, "move": "keys", "keys": [
                          {"at": {"word": "t1:2", "text": "beta"}, "scale": 1.2}]}],
    })
    assert stage.rebase_edit(changed)["status"] == "rebased"
    zoom = json.loads((changed / "timeline/agent.json").read_bytes())["elements"][0]
    assert zoom["keys"][0]["at"] == {"word": "t1:3", "text": "beta"}
