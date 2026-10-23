"""Batched join trials, durable decisions and source-clock filler publishing."""

from __future__ import annotations

import json
import wave
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth

from yap_studio import ffmpeg, receipt
from yap_studio.errors import InputError, YapError
from yap_studio.project.model import Removal
from yap_studio.project.model import Word as SourceWord
from yap_studio.project.receipts import StageRun
from yap_studio.project.status import ProjectContext, fillers_inputs, ingest_inputs, status
from yap_studio.stages import cut as cut_stage
from yap_studio.stages import edges
from yap_studio.stages import fillers as stage
from yap_studio.transcript import ScriptedBackend, Transcript, Word, resolve_backend

ROWS = [("bright", 0.25, 0.55), ("morning", 0.6, 0.9), ("um", 1.1, 1.3),
        ("garden", 1.5, 1.8), ("flowers", 1.85, 2.15), ("grow", 2.2, 2.5)]


def _requested_cut(**fields) -> dict:
    return {"segment": "s1", "take": "t1", "edge": None, "start": 1.0, "end": 1.4,
            "words": ["t1:2"], "texts": ["um"], **fields}


def test_requested_cut_clean_middle_is_read_only(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    batches = install_engine(monkeypatch)
    before = {p.relative_to(root): p.read_bytes() if p.is_file() else None
              for p in root.rglob("*")}
    result, = stage.test_cuts(root, [_requested_cut()])
    assert result["pass"] and result["nudge"] == [0, 0] and result["words"] == "pass"
    assert result["removed_words"] == ["um"] and len(batches) == 1
    assert {p.relative_to(root): p.read_bytes() if p.is_file() else None
            for p in root.rglob("*")} == before


@pytest.mark.parametrize("recovers", [False, True])
def test_requested_cut_searches_nudges_or_reports_clipped_word(tmp_path, monkeypatch,
                                                               recovers) -> None:
    root = make_project(tmp_path / "project")

    def script(target):
        tokens = [row[0] for row in ROWS]
        if target.stem.endswith("cut"):
            tokens.remove("um")
            if not recovers or "round-0" in target.stem:
                tokens.remove("garden")
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    result, = stage.test_cuts(root, [_requested_cut()])
    assert result["pass"] == recovers
    if recovers:
        assert sum(abs(value) for value in result["nudge"]) == 1
    else:
        assert result["words"] == "fail" and "garden" in str(result["diff"])


def test_requested_edge_cut_uses_retained_side_and_only_inner_nudges(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    trials = []
    original = stage._trial

    def record(segment, candidates, nudge=(0, 0), *, edge_aware=False):
        trial = original(segment, candidates, nudge, edge_aware=edge_aware)
        if trial is not None:
            trials.append(trial)
        return trial

    monkeypatch.setattr(stage, "_trial", record)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(
        lambda target: transcript(["um", "garden", "flowers"]
                                  if "round-0" not in target.stem else ["garden", "flowers"])))
    result, = stage.test_cuts(root, [_requested_cut(edge="start", start=0.2, end=1.0,
                                                  words=["t1:0", "t1:1"])])
    assert result["pass"] and result["nudge"][0] == 0
    assert trials and all(trial.edge == "start" and trial.nudge[0] == 0 for trial in trials)
    assert all(trial.start == trial.candidates[0].end for trial in trials)


@pytest.mark.parametrize("max_pause_s,passes", [(0.8, True), (0.4, False)])
def test_requested_cut_without_backend_skips_words_and_requires_listen(
    tmp_path, monkeypatch, max_pause_s, passes,
) -> None:
    rows = [("bright", 0.25, 0.55), ("um", 1.0, 1.05), ("garden", 1.5, 1.8)]
    source = synth.voiced_source(tmp_path / "source.mov", [(0.25, 0.55), (1.5, 1.8)],
                                 seconds=2.3)
    root = make_project(tmp_path / "project", rows, source)
    project = json.loads((root / "project.json").read_bytes())
    project["asr"] = {"engine": "none", "model": None}
    project["rules"] = {"max_pause_s": max_pause_s}
    write_json(root / "project.json", project)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: pytest.fail("engine is none"))
    result, = stage.test_cuts(root, [_requested_cut(start=0.8, end=1.1, words=["t1:1"])])
    assert result["words"] == "skipped" and result["listen"] is True
    assert result["pass"] == passes
    if passes:
        assert result["nudge"] == [0, 0]


def test_requested_cut_missing_segment_fails_plainly(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    result, = stage.test_cuts(root, [_requested_cut(segment="s3")])
    assert not result["pass"] and result["diff"] == ["segment s3 is not in the saved cut"]


@pytest.mark.parametrize("other_kind", ["cut", "remove"])
def test_requested_cuts_touching_planned_removals_hear_combined_join(
    tmp_path, monkeypatch, other_kind,
) -> None:
    rows = [("alpha", 0.1, 0.2), ("bravo", 0.4, 0.5),
            ("charlie", 0.6, 0.7), ("delta", 0.8, 0.9)]
    root = make_project(tmp_path / "project", rows)
    edit = json.loads((root / "edit.json").read_bytes())
    spans = [_requested_cut(start=0.3, end=0.55, words=["t1:1"], texts=["bravo"], action=0)]
    if other_kind == "cut":
        spans.append(_requested_cut(start=0.55, end=0.75, words=["t1:2"],
                                    texts=["charlie"], action=1))
    removals = [{"take": "t1", "start": 0.3, "end": 0.55},
                {"take": "t1", "start": 0.55, "end": 0.75}]
    windows, trials, planned = [], [], []

    class RecordingBackend(ScriptedBackend):
        def transcribe_many(self, paths, *, language=None):
            for target in paths:
                if target.stem.endswith("cut"):
                    with wave.open(str(target)) as stream:
                        assert stream.getnframes() == round((2 + 1.4 - 0.45) * stage._RATE)
                        windows.append(stream.readframes(stream.getnframes()))
            return super().transcribe_many(paths, language=language)

    def script(target):
        if target.stem.endswith("cut"):
            return transcript(["alpha", "delta"])
        return transcript(["alpha", "bravo" if target.stem.endswith("0-control")
                           else "charlie", "delta"])

    original_round, original_planned = stage._round, stage._planned

    def record_round(batch, *args):
        trials.extend(batch)
        return original_round(batch, *args)

    def record_planned(context, decisions=()):
        planned.append(context.json("edit.json"))
        return original_planned(context, decisions)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: RecordingBackend(script))
    monkeypatch.setattr(stage, "_round", record_round)
    monkeypatch.setattr(stage, "_planned", record_planned)
    results = stage.test_cuts(root, spans, edit=edit, removals=removals)
    assert all(result["pass"] and result["nudge"] == [0, 0] for result in results)
    assert len(trials) == len(spans) and len(windows) == len(spans)
    assert len(set(windows)) == 1 and planned == [edit]
    assert all(len(trial.joins) == 1 and trial.start == 0 and trial.end == 1.4 for trial in trials)
    assert all([word.text for word in trial.stored if word.id not in trial.removed]
               == ["alpha", "delta"] for trial in trials)
    assert all(trial.joins == (stage._RATE + round(0.3 * stage._RATE),) for trial in trials)


def test_requested_prefix_cut_after_trim_uses_restored_anchor(tmp_path, monkeypatch) -> None:
    rows = [("alpha", 0.1, 0.2), ("bravo", 0.4, 0.5),
            ("charlie", 0.6, 0.7), ("delta", 0.8, 0.9)]
    root = make_project(tmp_path / "project", rows)
    saved = (root / "edit.json").read_bytes()
    edit = json.loads(saved)
    edit["segments"][0]["from"] = {"word": "t1:2"}
    seen = []
    original = stage._planned

    def planned(context, decisions=()):
        assert context.json("edit.json")["segments"][0]["from"] == {"word": "t1:1"}
        result = original(context, decisions)
        assert "alpha" not in [word.text for word in result["s1"].words]
        seen.append(result)
        return result

    monkeypatch.setattr(stage, "_planned", planned)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(
        lambda _: transcript(["charlie", "delta"])))
    span = _requested_cut(edge="start", start=0.28, end=0.55,
                          words=["t1:1"], texts=["bravo"],
                          restore={"edge": "start", "anchor": {"word": "t1:1"}})
    results = stage.test_cuts(root, [span, span], edit=edit)
    assert len(results) == 2
    assert all(result["pass"] and result["words"] == "pass" and result["diff"] == []
               for result in results)
    assert len(seen) == 1 and edit["segments"][0]["from"] == {"word": "t1:2"}
    assert (root / "edit.json").read_bytes() == saved


@pytest.mark.parametrize("diff", [
    "removal clips kept words", "outside segment", "overlapping cuts",
])
def test_requested_cut_pre_evaluation_failure_keeps_bounds_and_diff(tmp_path, monkeypatch, diff):
    root = make_project(tmp_path / "project")
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(
        lambda _: pytest.fail("failed before audio evaluation")))
    monkeypatch.setattr(stage, "_source_round", lambda requests, *args: [
        {"pass": False, "words": "fail", "diff": [diff]} for request in requests])
    span = _requested_cut()
    result, = stage.test_cuts(root, [span])
    assert not result["pass"] and result["diff"] == [diff]
    assert (result["start"], result["end"]) == (span["start"], span["end"])


def test_requested_cut_weak_control_passes_with_listen(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")

    def script(target):
        tokens = [row[0] for row in ROWS]
        tokens.remove("um" if target.stem.endswith("cut") else "bright")
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    result, = stage.test_cuts(root, [_requested_cut()])
    assert result["pass"] and result["words"] == "unverifiable" and result["listen"]


def test_requested_cut_uses_applied_removals_as_baseline(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    saved = stage.add_manual(root, "t1", 0.56, 0.98)
    original = stage._planned

    def planned(context, decisions=()):
        assert [item.id for item in decisions] == [saved["id"]]
        return original(context, decisions)

    def script(target):
        tokens = [row[0] for row in ROWS if row[0] != "morning"]
        if target.stem.endswith("cut"):
            tokens.remove("um")
        return transcript(tokens)

    monkeypatch.setattr(stage, "_planned", planned)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    before = (root / "removals.json").read_bytes()
    result, = stage.test_cuts(root, [_requested_cut()])
    assert result["pass"] and result["words"] == "pass"
    assert (root / "removals.json").read_bytes() == before


def _save_cut_baseline(root, monkeypatch, candidate):
    with monkeypatch.context() as patch:
        patch.setattr(stage, "_proposals", lambda *_, **__: [candidate])
        patch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(
            lambda target: transcript([text for index, (text, _, _) in enumerate(ROWS)
                                       if target.stem.endswith("control")
                                       or f"t1:{index}" not in candidate.word_ids])))
        saved, = stage.fillers(root, apply=True)["entries"]
    assert saved["status"] == "applied"
    return saved


@pytest.mark.parametrize("restored", [False, True])
def test_requested_cut_includes_rejected_saved_filler_in_both_trials(
    tmp_path, monkeypatch, restored,
) -> None:
    root = make_project(tmp_path / "project")
    saved = _save_cut_baseline(root, monkeypatch, stage.Candidate(
        "t1", "filler", ("t1:2",), 1.0, 1.4, (), "s1", False, {}))
    trials = []
    original = stage._round

    def checked(batch, *args):
        trials.extend(batch)
        return original(batch, *args)

    def script(target):
        tokens = [text for text, _, _ in ROWS if restored or text != "um"]
        if target.stem.endswith("cut"):
            tokens.remove("garden")
        return transcript(tokens)

    monkeypatch.setattr(stage, "_round", checked)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    rejects = (saved["id"],) if restored else ()
    result, = stage.test_cuts(root, [_requested_cut(start=1.4, end=1.825,
                                                   words=["t1:3"], texts=["garden"])],
                              rejects=rejects)
    assert result["pass"] and result["nudge"] == [0, 0]
    trial, = trials
    assert ("t1:2" in {word.id for word in trial.stored}) == restored
    retained = {word.id for word in trial.stored if word.id not in trial.removed}
    assert ("t1:2" in retained) == restored


@pytest.mark.parametrize("baseline", ["rejected", "saved", "planned"])
def test_requested_edge_cut_counts_newly_audible_words_as_removed(
    tmp_path, monkeypatch, baseline,
) -> None:
    root = make_project(tmp_path / "project")
    saved = _save_cut_baseline(root, monkeypatch, stage.Candidate(
        "t1", "filler", ("t1:1",), 0.56, 1.0, (), "s1", False, {}))
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["from"] = {"word": "t1:3"}
    trials = []
    original = stage._round

    def checked(batch, *args):
        trials.extend(batch)
        return original(batch, *args)

    monkeypatch.setattr(stage, "_round", checked)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(
        lambda _: transcript(["garden", "flowers", "grow"])))
    span = _requested_cut(edge="start", start=0, end=1.4,
                          words=["t1:0", "t1:2"], texts=["bright", "um"],
                          restore={"edge": "start", "anchor": {"time": 0}})
    result, = stage.test_cuts(root, [span], edit=edit,
                              rejects=() if baseline == "saved" else (saved["id"],),
                              removals=[{"take": "t1", "start": 0.56, "end": 1.0}]
                              if baseline == "planned" else ())
    assert result["pass"] and result["words"] == "pass" and result["nudge"] == [0, 0]
    assert ("morning" in result["removed_words"]) == (baseline == "rejected")
    trial, = trials
    assert ("t1:1" in trial.removed) == (baseline == "rejected")
    assert [word.text for word in trial.stored] == ["garden", "flowers", "grow"]


def test_requested_middle_cut_still_refuses_extra_kept_word(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    saved = _save_cut_baseline(root, monkeypatch, stage.Candidate(
        "t1", "filler", ("t1:1",), 0.56, 1.0, (), "s1", False, {}))
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(
        lambda _: pytest.fail("the extra kept word must be refused before transcription")))
    span = _requested_cut(start=0.575, end=1.4)
    result, = stage.test_cuts(root, [span], rejects=(saved["id"],))
    assert not result["pass"] and any("removal clips kept words" in diff
                                      and "morning" in diff for diff in result["diff"])


def test_requested_cut_saved_silence_straddling_start_uses_union(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    saved = _save_cut_baseline(root, monkeypatch, stage.Candidate(
        "t1", "silence", (), 0.95, 1.05, (), "s1", False, {}))
    assert (saved["start"], saved["end"]) == (0.95, 1.05)
    install_engine(monkeypatch)
    trials = []
    original = stage._trial

    def trial(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is not None:
            trials.append(result)
        return result

    monkeypatch.setattr(stage, "_trial", trial)
    result, = stage.test_cuts(root, [_requested_cut()])
    assert result["pass"] and result["words"] == "pass" and result["nudge"] == [0, 0]
    tested, = trials
    assert tested.joins == (stage._RATE + round((0.95 - tested.start) * stage._RATE),)
    assert len(tested.control) - len(tested.cut) == round((1.4 - 1.05) * stage._RATE)


def test_requested_cut_surrounding_saved_filler_excludes_its_word(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    _save_cut_baseline(root, monkeypatch, stage.Candidate(
        "t1", "filler", ("t1:2",), 1.0, 1.4, (), "s1", False, {}))

    def script(target):
        tokens = ["bright", "morning", "garden", "flowers", "grow"]
        if target.stem.endswith("cut"):
            tokens.remove("morning")
            tokens.remove("garden")
        return transcript(tokens)

    original = stage._round

    def checked(trials, *args, **kwargs):
        assert all("t1:2" not in {word.id for word in trial.stored} for trial in trials)
        return original(trials, *args, **kwargs)

    monkeypatch.setattr(stage, "_round", checked)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    result, = stage.test_cuts(root, [_requested_cut(
        start=0.575, end=1.825, words=["t1:1", "t1:3"], texts=["morning", "garden"])])
    assert result["pass"] and result["words"] == "pass" and result["nudge"] == [0, 0]
    assert result["diff"] == [] and result["removed_words"] == ["morning", "garden"]


def test_filler_proposal_overlapping_saved_removal_is_still_refused(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    saved = stage.add_manual(root, "t1", 0.95, 1.05)
    segments = stage._planned(ProjectContext(root), [Removal(
        saved["id"], "t1", 0.95, 1.05, "manual", "applied")])
    candidate = stage.Candidate("t1", "filler", ("t1:2",), 1.0, 1.4, (), "s1", False, {})
    assert stage._trial(segments["s1"], (candidate,)) is None
    install_engine(monkeypatch)
    result, = stage._source_round([stage._JoinRequest((candidate,))], segments,
                                  stage.resolve_backend("auto"), tmp_path, "en", 0)
    assert result["pass"] is False and result["diff"] == ["overlapping cuts"]


@pytest.mark.parametrize("force", [False, True])
def test_overlapping_manual_and_silence_survive_fillers_and_cut(
    tmp_path, monkeypatch, force,
) -> None:
    root = make_project(tmp_path / "project")
    saved = _save_cut_baseline(root, monkeypatch, stage.Candidate(
        "t1", "silence", (), 0.9, 1.1, (), "s1", False, {}))
    manual = stage.add_manual(root, "t1", 1.0, 1.4)
    before = json.loads((root / "removals.json").read_bytes())["removals"]
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: [])
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(
        lambda _: pytest.fail("saved overlapping decisions must stay untested")))
    result = stage.fillers(root, apply=True, force=force)
    assert {item["id"]: item for item in result["entries"]} == {
        item["id"]: item for item in before}
    assert {item["id"] for item in result["entries"]} == {saved["id"], manual["id"]}
    assert all(item["status"] == "applied" for item in result["entries"])
    plan = cut_stage.resolve_cut(root)
    assert plan.pieces and any(float(piece.src_t0) >= 1.4 for piece in plan.pieces)
    assert all(float(piece.src_t1) <= 0.9 or float(piece.src_t0) >= 1.4 for piece in plan.pieces)
    assert "t1:2" not in {word["id"] for word in cut_stage._cut_words(plan)["words"]}


def test_applied_silence_does_not_own_stretched_final_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = synth.voiced_source(tmp_path / "source.mov", [(0.4, 0.7), (1.4, 1.8)],
                                 seconds=2.9)
    root = make_project(tmp_path / "project", [("hello", 0.4, 0.7),
                                               ("house.", 1.4, 2.4)], source)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 0.3}
    write_json(root / "project.json", project)
    context = ProjectContext(root)
    decisions = [Removal("r1", "t1", 1.85, 2.45, "silence", "applied"),
                 Removal("r2", "t1", 2.65, 2.75, "silence", "applied")]
    with monkeypatch.context() as patch:
        patch.setattr(edges, "voiced_words", lambda _take, words: tuple(words))
        segment = stage._planned(context, decisions)["s1"]
        speech = replace(decisions[0], kind="filler")
        assert stage._planned(context, [speech])["s1"].existing[0].word_ids == ("t1:1",)
    assert all(not item.word_ids for item in segment.existing)
    assert any(item.kind == "silence" and item.start >= 2.45
               for item in stage._proposals(context, {"s1": segment}))
    assert "t1:1" in {word.id for word in segment.words}
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": item.id, "take": item.take, "start": item.start, "end": item.end,
             "kind": item.kind, "status": item.status, "words": []}
            for item in decisions
        ],
    })
    plan = cut_stage.resolve_cut(root)
    assert "t1:1" in {word["id"] for word in cut_stage._cut_words(plan)["words"]}


def test_applied_silence_listing_a_stretched_word_keeps_it_in_cut_words(tmp_path: Path) -> None:
    source = synth.voiced_source(tmp_path / "source.mov",
                                 [(0.4, 0.7), (1.4, 1.8), (2.70, 2.72)], seconds=3.5)
    root = make_project(tmp_path / "project", [("hello", 0.4, 0.7),
                                               ("house.", 1.4, 3.0)], source)
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 1.9, "end": 2.6,
             "kind": "silence", "status": "applied", "words": ["t1:1"]},
        ],
    })

    plan = cut_stage.resolve_cut(root)
    words = {word["id"]: word for word in cut_stage._cut_words(plan)["words"]}
    assert "t1:1" in words
    first = plan.pieces[0]
    assert words["t1:1"]["end"] <= (first.out_f0 + first.n_frames) / float(cut_stage.FPS)


def test_silence_proposal_in_a_stretched_word_owns_no_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = synth.voiced_source(tmp_path / "source.mov",
                                 [(0.4, 0.7), (1.4, 1.8), (2.70, 2.72)], seconds=3.5)
    root = make_project(tmp_path / "project", [("hello", 0.4, 0.7),
                                               ("house.", 1.4, 3.0)], source)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 0.3}
    write_json(root / "project.json", project)
    context = ProjectContext(root)
    clip = stage._RefinedSpan(1.0, 3.4, ())
    monkeypatch.setattr(edges, "voiced_words", lambda _take, words: tuple(words))

    candidates = stage._proposals(context, stage._planned(context), refinements={"s1": (clip,)})
    silences = [item for item in candidates if item.kind == "silence"]
    assert silences
    assert all(item.word_ids == () for item in silences)


def write_json(target: Path, data: dict[str, Any]) -> None:
    """Write portable fixture documents."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def suppress_tail_pause(root: Path) -> None:
    """Keep tests for other decisions from acquiring a segment-final pause."""
    target = root / "project.json"
    project = json.loads(target.read_bytes())
    project.setdefault("rules", {})["max_pause_s"] = 0.6
    write_json(target, project)


def make_project(root: Path, rows: list[tuple[str, float, float]] | None = None,
                 source: Path | None = None) -> Path:
    """Publish source words and matching ingest ownership around synthetic PCM."""
    rows = ROWS if rows is None else rows
    root.mkdir(parents=True, exist_ok=True)
    take = root / "takes/source.mov"
    take.parent.mkdir()
    duration = max(last for _, _, last in rows) + 0.5
    if source is None:
        wav = root / "source.wav"
        sound = np.zeros(round(duration * 48_000), dtype=np.float32)
        for _, first, last in rows:
            start, end = round(first * 48_000), round(last * 48_000)
            sound[start:end] = 0.1 * np.sin(np.arange(end - start) * 2 * np.pi * 220 / 48_000)
        synth._write_wav(wav, sound, 48_000)
        synth._run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=s=160x90:r=30:d={duration}", "-i", str(wav),
                    "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                    "-c:v", "libx264", "-c:a", "pcm_s16le", str(take)])
        wav.unlink()
    else:
        import shutil

        shutil.copyfile(source, take)
        duration = float(ffmpeg.probe(take).video_duration_s)
    write_json(root / "project.json", {
        "schema": "yap-studio/project@1", "name": "join_story",
        "takes": [{"id": "t1", "path": "takes/source.mov"}],
        "asr": {"engine": "auto", "model": None},
    })
    inputs = ingest_inputs(ProjectContext(root))
    with StageRun(root, "ingest") as run:
        for key, digest in inputs.items():
            run.record_inputs(key, digest)
        staged = root / "staged.json"
        write_json(staged, {
            "schema": "yap-studio/words@1", "take": "t1", "take_sha256": inputs["take:t1"],
            "time_base": "source", "audio_offset_s": 0, "engine": "none", "model": None,
            "transcript_sha256": "1" * 64,
            "words": [{"id": f"t1:{index}", "text": text, "start": first, "end": last}
                      for index, (text, first, last) in enumerate(rows)],
        })
        run.replace_output(root / "ingest/t1/words.json", staged)
        run.finish()
    write_json(root / "edit.json", {
        "schema": "yap-studio/edit@1",
        "words_sha256": {"t1": receipt.sha256_file(root / "ingest/t1/words.json")},
        "segments": [{"id": "s1", "take": "t1", "from": {"time": 0},
                      "to": {"time": duration}}],
    })
    return root


def transcript(tokens: Sequence[str]) -> Transcript:
    """Place scripted words inside the audible window, beyond the digital lead."""
    return Transcript([Word(text, 1.05 + index * 0.12, 1.14 + index * 0.12)
                       for index, text in enumerate(tokens)])


def install_engine(monkeypatch: pytest.MonkeyPatch, *, clipped: bool = False,
                   leftover: bool = False) -> list[list[Path]]:
    """Install a recording ScriptedBackend with a known cut/control difference."""
    batches: list[list[Path]] = []

    def script(target: Path) -> Transcript:
        """Return the planted filler only for controls or a deliberately bad cut."""
        tokens = [row[0] for row in ROWS]
        if target.stem.endswith("cut"):
            if not leftover:
                tokens.remove("um")
            if clipped:
                tokens.remove("garden")
        return transcript(tokens)

    class RecordingBackend(ScriptedBackend):
        """Retain batch boundaries and verify that both leads are silent."""

        def transcribe_many(self, paths: Sequence[Path], *, language: str | None = None,
                            ) -> dict[Path, Transcript]:
            batches.append(list(paths))
            for target in paths:
                with wave.open(str(target)) as stream:
                    assert stream.getframerate() == 48_000
                    sound = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
                assert not sound[:48_000].any()
                assert not sound[-48_000:].any()
            return super().transcribe_many(paths, language=language)

    engine = RecordingBackend(script)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: engine)
    return batches


def test_clean_filler_and_final_pause_use_expected_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project")
    batches = install_engine(monkeypatch)
    result = stage.fillers(root, apply=True)
    entry = result["entries"][0]
    assert entry["id"] == "r1"
    assert entry["status"] == "applied"
    assert entry["evidence"]["words"] == "pass"
    assert entry["evidence"]["click_db"] <= 0
    assert [len(batch) for batch in batches] == [4, 8]
    assert any(item["kind"] == "silence" for item in result["entries"])
    fillers_state = next(item for item in status(root)["stages"] if item["name"] == "fillers")
    assert fillers_state["state"] == "done"
    before = (root / "removals.json").read_bytes()
    assert stage.fillers(root, apply=True)["noop"]
    assert (root / "removals.json").read_bytes() == before


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("speech", [False, True])
def test_fillers_records_each_takes_pause_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apply: bool, speech: bool,
) -> None:
    root = make_project(tmp_path / "project")
    second = root / "takes/second.mov"
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-i", str(root / "takes/source.mov"),
        "-af", "aeval=val(0)+0.0001", "-c:v", "copy", "-c:a", "pcm_s16le", str(second),
    ], timeout_s=60)
    project = json.loads((root / "project.json").read_bytes())
    project["takes"].append({"id": "t2", "path": "takes/second.mov"})
    write_json(root / "project.json", project)
    words = json.loads((root / "ingest/t1/words.json").read_bytes())
    words["take"] = "t2"
    words["take_sha256"] = receipt.sha256_file(second)
    for word in words["words"]:
        word["id"] = word["id"].replace("t1:", "t2:")
    write_json(root / "ingest/t2/words.json", words)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["words_sha256"]["t2"] = receipt.sha256_file(root / "ingest/t2/words.json")
    edit["segments"].append({**edit["segments"][0], "id": "s2", "take": "t2"})
    write_json(root / "edit.json", edit)
    if speech:
        install_engine(monkeypatch)
    else:
        monkeypatch.setattr(stage, "resolve_backend", lambda *_: None)
    expected = {segment.take: segment.pause
                for segment in stage._planned(ProjectContext(root)).values()}

    result = stage.fillers(root, apply=apply)
    report = json.loads((root / "fillers/report.json").read_bytes())
    owned = json.loads((root / "fillers/receipt.json").read_bytes())
    assert result["pause_thresholds"] == report["pause_thresholds"] == expected
    assert owned["pause_thresholds"] == expected
    assert set(expected) == {"t1", "t2"}
    for pause in expected.values():
        assert pause is not None
        assert pause["rule"] == "floor"
    assert expected["t1"]["threshold_db"] == -165.0
    assert -70 < expected["t2"]["threshold_db"] < -60
    assert stage.fillers(root, apply=apply)["pause_thresholds"] == expected


@pytest.mark.parametrize("refined_text", ["um", "uh"])
@pytest.mark.parametrize("decision", ["applied", "rejected"])
def test_refined_filler_projects_source_ids_and_clocks_into_cut_words(
    tmp_path: Path, refined_text: str, decision: str,
) -> None:
    """A removed occurrence keeps its identity even when its stored midpoint survives."""
    stored = [*ROWS]
    stored[3] = ("garden", 1.3, 1.6)
    audible = [*stored]
    audible[2] = (refined_text, 1.0, 1.2)
    audible[3] = ("garden", 1.2, 1.6)
    source = synth.voiced_source(
        tmp_path / "tones.mov", [(first, last) for _, first, last in audible], seconds=3,
    )
    root = make_project(tmp_path / "project", stored, source)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["text"] = " ".join(text for text, _, _ in stored)
    write_json(root / "edit.json", edit)
    words_path = root / "ingest/t1/words.json"
    before = words_path.read_bytes()
    context = ProjectContext(root)
    clip = stage._RefinedSpan(0.6, 1.6, tuple(
        SourceWord("", text, first, last) for text, first, last in audible[1:4]
    ))
    candidate, = [item for item in stage._proposals(
        context, stage._planned(context), refinements={"s1": (clip,)},
    ) if item.kind == "filler"]
    assert (candidate.start, candidate.end) == pytest.approx((1.0, 1.2))
    assert candidate.word_ids == ("t1:2",)
    if refined_text == "uh":
        assert candidate.listen
        assert candidate.plan["refined_provenance"] == "inferred"
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [{
            "id": "r1", "take": "t1", "start": candidate.start, "end": candidate.end,
            "kind": "filler", "status": decision, "words": list(candidate.word_ids),
            "listen": candidate.listen, "evidence": candidate.plan,
        }],
    })

    plan = cut_stage.resolve_cut(root)
    assert plan.text_mismatches == ()
    document = cut_stage._cut_words(plan)
    retained = {word["id"]: word for word in document["words"]}
    assert ("t1:2" not in retained) == (decision == "applied")
    assert retained["t1:3"]["start"] == pytest.approx(1.0 if decision == "applied" else 1.3)
    assert words_path.read_bytes() == before


def test_refined_filler_keeps_unmapped_word_outside_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An overlapping source word survives when its midpoint is outside the removal."""
    root = make_project(tmp_path / "project", [("um", 0.30, 0.60), ("garden", 0.35, 0.80)])
    context = ProjectContext(root)
    clip = stage._RefinedSpan(0.30, 0.80, (
        SourceWord("", "uh", 0.30, 0.40), SourceWord("", "flower", 0.50, 0.80),
    ))
    # Supply the reviewed removal span independently of detector boundary planning.
    detected = stage.Candidate("t1", "filler", (), 0.30, 0.50, ("filler uh",), "", False, {})
    monkeypatch.setattr(stage, "detect", lambda *_, **__: stage.DetectResult((detected,), True, ()))
    candidate, = [item for item in stage._proposals(
        context, stage._planned(context), refinements={"s1": (clip,)},
    ) if item.kind == "filler"]
    assert (candidate.start, candidate.end) == pytest.approx((0.30, 0.50))
    assert candidate.word_ids == ("t1:0",)
    assert candidate.listen
    assert candidate.plan["refined_provenance"] == "inferred"
    assert candidate.plan["refined_source_words"] == []
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [{
            "id": "r1", "take": "t1", "start": candidate.start, "end": candidate.end,
            "kind": "filler", "status": "applied", "words": list(candidate.word_ids),
            "listen": candidate.listen, "evidence": candidate.plan,
        }],
    })
    document = cut_stage._cut_words(cut_stage.resolve_cut(root))
    assert [word["id"] for word in document["words"]] == ["t1:1"]
    assert document["words"][0]["text"] == "garden"


def test_refined_restart_keeps_unmapped_word_after_actual_removal(tmp_path: Path) -> None:
    """A late retained-copy clock cannot remove source speech after the audio cut."""
    audible = [("intro", 1.9, 2.1), ("build", 2.4, 2.9), ("a", 3.0, 3.15),
               ("gard", 3.3, 3.5), ("build", 4.2, 4.55), ("a", 4.65, 4.8),
               ("garden", 4.9, 5.15), ("plan", 5.25, 5.5)]
    stored = [*audible[:3], ("garden", 3.3, 3.5), ("moss", 4.23, 4.27),
              ("build", 4.3, 4.55), *audible[5:]]
    source = synth.voiced_source(
        tmp_path / "tones.mov", [(first, last) for _, first, last in audible], seconds=6,
    )
    root = make_project(tmp_path / "project", stored, source)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 10}
    write_json(root / "project.json", project)
    refined = [*audible[1:]]
    refined[3] = ("build", 4.3, 4.55)
    clip = stage._RefinedSpan(2.4, 5.5, tuple(
        SourceWord("", text, first, last) for text, first, last in refined
    ))
    context = ProjectContext(root)
    candidate, = stage._proposals(context, stage._planned(context), refinements={"s1": (clip,)})
    assert candidate.kind == "restart"
    assert candidate.end == pytest.approx(4.2)
    assert candidate.plan["restart_audio"]["clip_end_s"] == pytest.approx(4.3)
    assert candidate.plan["refined_removal"] == {"start": 2.4, "end": 4.3}
    assert candidate.word_ids == ("t1:1", "t1:2", "t1:3")
    assert candidate.listen
    assert candidate.plan["refined_provenance"] == "inferred"
    kept_copy, = (word for word in candidate.plan["refined_source_words"] if word["id"] == "t1:5")
    assert kept_copy["start"] == pytest.approx(4.2)
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [{
            "id": "r1", "take": "t1", "start": candidate.start, "end": candidate.end,
            "kind": "restart", "status": "applied", "words": list(candidate.word_ids),
            "listen": candidate.listen, "evidence": candidate.plan,
        }],
    })
    document = cut_stage._cut_words(cut_stage.resolve_cut(root))
    retained = {word["id"]: word for word in document["words"]}
    assert set(retained) == {"t1:0", "t1:4", "t1:5", "t1:6", "t1:7", "t1:8"}
    assert retained["t1:4"]["text"] == "moss"


@pytest.mark.parametrize("shape", ["long-word", "voiced-gap"])
@pytest.mark.parametrize("join_passes", [True, False], ids=["safe-join", "clipped-word"])
def test_hidden_restart_is_refined_then_passes_the_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str, join_passes: bool,
) -> None:
    if shape == "long-word":
        rows = [("build", 2.4, 2.7), ("a", 2.7, 2.9), ("garden", 2.9, 4.7),
                ("plan", 4.7, 5.0)]
        revealed = [("build", 2.4, 2.7), ("a", 2.7, 2.9), ("gard", 2.9, 3.1),
                    ("build", 3.6, 3.9), ("a", 3.9, 4.1), ("garden", 4.1, 4.7),
                    ("plan", 4.7, 5.0)]
        removed, end, rule = 3, 3.6, "hidden repeated 2-gram"
    else:
        rows = [("but", 2.4, 2.6), ("we'll", 2.9, 3.1), ("see", 3.1, 3.4)]
        revealed = [("but", 2.4, 2.5), ("we'll", 2.5, 2.6),
                    ("but", 2.6, 2.9), ("we'll", 2.9, 3.1), ("see", 3.1, 3.4)]
        removed, end, rule = 2, 2.6, "repeated 2-gram"
    source = synth.voiced_source(tmp_path / "tones.mov",
                                [(first, last) for _, first, last in revealed], seconds=5.5)
    root = make_project(tmp_path / "project", rows, source)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["from"] = {"time": 2.0}
    write_json(root / "edit.json", edit)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 10}
    write_json(root / "project.json", project)
    words_path = root / "ingest/t1/words.json"
    words_before = words_path.read_bytes()
    cut_words = root / "cut/cut-words.json"
    write_json(cut_words, {"synthetic": True})
    cut_before = cut_words.read_bytes()
    batches: list[list[Path]] = []

    def script(target: Path) -> Transcript:
        """Reveal the first attempt locally and hear it only in the control."""
        if target.stem.startswith("refine-"):
            rate, sound = synth.read_wav(target)
            assert len(sound) / rate >= 1.2
            clip_start = 2.4 if shape == "long-word" else 2.3
            return Transcript([Word(text, first - clip_start, last - clip_start)
                               for text, first, last in revealed])
        tokens = [text for text, _, _ in revealed]
        if target.stem.endswith("cut"):
            tokens = tokens[removed:]
            if not join_passes:
                tokens.remove("garden" if shape == "long-word" else "see")
        return transcript(tokens)

    class RecordingBackend(ScriptedBackend):
        """Separate the single refinement batch from the existing join batches."""

        def transcribe_many(self, paths: Sequence[Path], *, language: str | None = None,
                            ) -> dict[Path, Transcript]:
            batches.append(list(paths))
            return super().transcribe_many(paths, language=language)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: RecordingBackend(script))
    result = stage.fillers(root, apply=True)
    entry, = result["entries"]
    assert entry["kind"] == "restart"
    assert entry["status"] == ("applied" if join_passes else "rejected")
    assert entry["start"] == pytest.approx(2.4)
    assert entry["end"] == pytest.approx(end)
    assert entry["evidence"]["words"] == ("pass" if join_passes else "fail")
    assert entry["evidence"]["pass"] is join_passes
    assert rule in entry["evidence"]["reasons"]
    assert entry["evidence"]["refined_words"] == [
        {"text": text, "start": first, "end": last} for text, first, last in revealed
    ]
    assert entry["words"] == (["t1:0", "t1:1"] if shape == "long-word" else ["t1:0"])
    assert result["refinement"]["spots"] == 1
    assert result["refinement"]["clips"] == 1
    assert result["counts"]["restart_rules"][rule] == 1
    assert len(batches) == (2 if join_passes else 4)
    assert len(batches[0]) == 1 and batches[0][0].stem.startswith("refine-")
    assert len(batches[1]) == 2
    assert words_path.read_bytes() == words_before
    assert cut_words.read_bytes() == cut_before


def test_no_engine_skips_hidden_restart_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [("bring", 0.4, 0.7), ("a", 0.7, 0.9), ("blanket", 0.9, 2.7),
            ("outside", 2.7, 3.0)]
    source = synth.voiced_source(tmp_path / "tones.mov", [(0.4, 1.2), (1.7, 3.0)], seconds=3.5)
    root = make_project(tmp_path / "project", rows, source)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: None)
    monkeypatch.setattr(stage, "_write_wav", lambda *_: pytest.fail("refinement must be skipped"))
    result = stage.fillers(root)
    assert result["refinement"]["spots"] == result["refinement"]["clips"] == 0
    assert result["refinement"]["status"] == "skipped"
    assert "restart refinement skipped: no speech engine" in result["notes"]
    assert not any(item["kind"] == "restart" for item in result["entries"])


@pytest.mark.parametrize("fragment", [False, True], ids=["hidden-repeat", "fragment"])
@pytest.mark.parametrize("second_offset", [0.0, -0.1], ids=["second-on-time", "second-early"])
@pytest.mark.parametrize("first_in_quiet", [False, True], ids=["first-in-voice", "first-in-quiet"])
def test_refined_restart_uses_voice_onsets_and_passes_the_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fragment: bool, second_offset: float,
    first_in_quiet: bool,
) -> None:
    """Hear a leftover first copy even when its clip start falls in quiet after it."""
    audible = [("before", 1.9, 2.1), ("build", 2.4, 2.6 if first_in_quiet else 2.9),
               ("a", 2.66 if first_in_quiet else 3.0, 3.15),
               ("gard" if fragment else "basket", 3.3, 3.5),
               ("build", 4.2, 4.55), ("a", 4.65, 4.8), ("garden", 4.9, 5.15),
               ("plan", 5.25, 5.5)]
    rows = [*audible[:3], ("garden", 3.3, 5.15), audible[-1]]
    source = synth.voiced_source(tmp_path / "tones.mov",
                                [(first, last) for _, first, last in audible], seconds=6,
                                frequencies=[220, 440, 220, 220, 220, 220, 220, 220])
    root = make_project(tmp_path / "project", rows, source)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["from"] = {"time": 1.7}
    write_json(root / "edit.json", edit)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 10}
    write_json(root / "project.json", project)
    refined = [(text, first + (0.25 if index == 1 else second_offset if index == 4 else 0),
                2.66 if index == 1 and first_in_quiet else
                4.25 if index == 4 and second_offset else last)
               for index, (text, first, last) in enumerate(audible[1:], start=1)]

    def script(target: Path) -> Transcript:
        """Hear the first copy's distinct tone even if it touches the second copy."""
        if target.stem.startswith("refine-"):
            return Transcript([Word(text, first - 2.4, last - 2.4)
                               for text, first, last in refined])
        tokens = [text for text, _, _ in audible[:-1]]
        if target.stem.endswith("cut"):
            tokens = [audible[0][0], *(text for text, _, _ in audible[4:-1])]
            rate, sound = synth.read_wav(target)
            marker = np.exp(-2j * np.pi * 440 * np.arange(len(sound)) / rate)
            if abs(np.dot(sound, marker)) / rate > 0.001:
                tokens.insert(1, "build")
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    result = stage.fillers(root, apply=True)
    entry, = result["entries"]
    assert entry["status"] == "applied", entry["evidence"]
    assert entry["evidence"]["words"] == "pass"
    assert entry["evidence"]["attempts"][0]["pass"], entry["evidence"]["attempts"]
    assert 2.1 <= entry["start"] < 2.4
    assert entry["end"] == pytest.approx(4.2)
    snapped = entry["evidence"]["restart_audio"]
    assert snapped["clip_start_s"] == pytest.approx(2.65)
    assert snapped["clip_end_s"] == pytest.approx(4.2 + second_offset)
    assert snapped["start_s"] == pytest.approx(2.4)
    assert snapped["end_s"] == pytest.approx(4.2)


def test_refined_restart_removes_voice_under_stretched_left_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent gap separates the left word from a first copy timed under it."""
    audible = [("before", 1.9, 2.2), ("make", 2.35, 2.9), ("a", 3.0, 3.15),
               ("comp", 3.3, 3.5), ("make", 4.2, 4.55), ("a", 4.65, 4.8),
               ("brand", 4.9, 5.15), ("plan", 5.25, 5.5)]
    rows = [("before", 1.9, 2.5), ("make", 2.5, 2.9), audible[2],
            ("brand", 3.3, 5.15), audible[-1]]
    source = synth.voiced_source(tmp_path / "tones.mov",
                                [(first, last) for _, first, last in audible], seconds=6,
                                frequencies=[220, 440, 220, 220, 220, 220, 220, 220])
    root = make_project(tmp_path / "project", rows, source)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["from"] = {"time": 1.7}
    write_json(root / "edit.json", edit)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 10}
    write_json(root / "project.json", project)
    context = ProjectContext(root)
    segment = stage._planned(context)["s1"]
    ranges = stage._refinement_ranges(segment, stage._suspicious_spots(segment))
    assert len(ranges) == 1
    refine_start = ranges[0][0]

    def script(target: Path) -> Transcript:
        """Hear the first copy's tone if any of it remains after the cut."""
        if target.stem.startswith("refine-"):
            heard = ([Word("before", 1.9 - refine_start, 2.5 - refine_start)]
                     if refine_start <= 1.9 else [])
            heard.extend(Word(text, (2.65 if index == 1 else first) - refine_start,
                              last - refine_start)
                         for index, (text, first, last) in enumerate(audible[1:], 1))
            return Transcript(heard)
        tokens = [text for text, _, _ in audible[:-1]]
        if target.stem.endswith("cut"):
            tokens = ["before", *(text for text, _, _ in audible[4:-1])]
            rate, sound = synth.read_wav(target)
            marker = np.exp(-2j * np.pi * 440 * np.arange(len(sound)) / rate)
            if abs(np.dot(sound, marker)) / rate > 0.001:
                tokens.insert(1, "make")
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    result = stage.fillers(root, apply=True)
    entry, = result["entries"]
    assert entry["status"] == "applied", entry["evidence"]
    assert entry["evidence"]["words"] == "pass"
    assert entry["evidence"]["attempts"][0]["pass"]
    assert 2.2 < entry["start"] < 2.35
    assert entry["evidence"]["restart_audio"]["start_s"] == pytest.approx(2.35)


def test_refined_restart_keeps_stretched_left_word_at_join(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the measured left end when a cut contains the stored midpoint."""
    audible = [("before", 2.02, 2.09), ("make", 2.25, 2.9), ("a", 3.0, 3.15),
               ("comp", 3.3, 3.5), ("make", 4.2, 4.55), ("a", 4.65, 4.8),
               ("brand", 4.9, 5.15), ("plan", 5.25, 5.5)]
    rows = [("before", 2.02, 2.45), ("make", 2.45, 2.9), audible[2],
            ("brand", 3.3, 5.15), audible[-1]]
    source = synth.voiced_source(tmp_path / "tones.mov",
                                [(first, last) for _, first, last in audible], seconds=6,
                                frequencies=[220, 440, 220, 220, 220, 220, 220, 220])
    root = make_project(tmp_path / "project", rows, source)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["from"] = {"time": 1.7}
    write_json(root / "edit.json", edit)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 10}
    write_json(root / "project.json", project)
    segment = stage._planned(ProjectContext(root))["s1"]
    ranges = stage._refinement_ranges(segment, stage._suspicious_spots(segment))
    assert len(ranges) == 1
    refine_start = ranges[0][0]

    def script(target: Path) -> Transcript:
        """Report the first copy if its tone remains after the cut."""
        if target.stem.startswith("refine-"):
            heard = ([Word("before", 2.02 - refine_start, 2.45 - refine_start)]
                     if refine_start <= 2.02 else [])
            heard.extend(Word(text, (2.45 if index == 1 else first) - refine_start,
                              last - refine_start)
                         for index, (text, first, last) in enumerate(audible[1:], 1))
            return Transcript(heard)
        tokens = [text for text, _, _ in audible[:-1]]
        if target.stem.endswith("cut"):
            tokens = ["before", *(text for text, _, _ in audible[4:-1])]
            rate, sound = synth.read_wav(target)
            marker = np.exp(-2j * np.pi * 440 * np.arange(len(sound)) / rate)
            if abs(np.dot(sound, marker)) / rate > 0.001:
                tokens.insert(1, "make")
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    result = stage.fillers(root, apply=True)
    entry, = result["entries"]
    assert entry["start"] < (2.02 + 2.45) / 2, entry["evidence"]["attempts"]
    assert entry["status"] == "applied", entry["evidence"]
    assert entry["evidence"]["words"] == "pass"
    assert entry["evidence"]["attempts"][0]["pass"]
    assert "t1:0" not in entry["words"]
    assert entry["evidence"]["restart_audio"]["left_end_s"] == pytest.approx(2.09,
                                                                               abs=0.02)
    cut_words = cut_stage._cut_words(cut_stage.resolve_cut(root))["words"]
    retained_ids = {word["id"] for word in cut_words}
    assert "t1:0" in retained_ids
    assert not retained_ids.intersection(entry["words"])
    left, = (word for word in entry["evidence"]["refined_source_words"]
             if word["id"] == "t1:0")
    assert left == {"id": "t1:0", "text": "before", "start": 2.02,
                    "end": pytest.approx(entry["evidence"]["restart_audio"]["left_end_s"])}


def test_refined_restart_at_first_word_respects_segment_edge(tmp_path: Path) -> None:
    """Backtracking must stop at the decoded edge even when voice began earlier."""
    rows = [("build", 2.4, 2.9), ("a", 3.0, 3.15), ("garden", 3.3, 5.15),
            ("plan", 5.25, 5.5)]
    source = synth.voiced_source(tmp_path / "tones.mov", [(2.3, 2.9), (3, 3.15),
                                (3.3, 3.5), (4.2, 4.55), (4.65, 4.8), (4.9, 5.15),
                                (5.25, 5.5)], seconds=6)
    root = make_project(tmp_path / "project", rows, source)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["from"] = {"time": 2.4}
    write_json(root / "edit.json", edit)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 10}
    write_json(root / "project.json", project)
    words = tuple(SourceWord("", text, first, last) for text, first, last in [
        ("build", 2.65, 2.9), ("a", 3, 3.15), ("basket", 3.3, 3.5),
        ("build", 4.2, 4.55), ("a", 4.65, 4.8), ("garden", 4.9, 5.15),
        ("plan", 5.25, 5.5),
    ])
    context = ProjectContext(root)
    segments = stage._planned(context)
    clip = stage._RefinedSpan(2.4, 5.5, words)
    candidate, = stage._proposals(context, segments, refinements={"s1": (clip,)})
    assert candidate.start == segments["s1"].start == pytest.approx(2.4)
    assert candidate.end == pytest.approx(4.2)
    assert candidate.listen


def test_refined_restart_keeps_the_word_at_the_measured_voice_onset(tmp_path: Path) -> None:
    """An early refined midpoint cannot delete the retained copy from cut words."""
    audible = [("intro", 1.9, 2.1), ("build", 2.4, 2.9), ("a", 3.0, 3.15),
               ("gard", 3.3, 3.5), ("build", 4.2, 4.55), ("a", 4.65, 4.8),
               ("garden", 4.9, 5.15), ("plan", 5.25, 5.5)]
    stored = [*audible]
    stored[3] = ("garden", 3.3, 3.5)
    stored[4] = ("build", 4.0, 4.15)
    source = synth.voiced_source(
        tmp_path / "tones.mov", [(first, last) for _, first, last in audible], seconds=6,
    )
    root = make_project(tmp_path / "project", stored, source)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["from"] = {"time": 1.7}
    write_json(root / "edit.json", edit)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 10}
    write_json(root / "project.json", project)
    refined = [*audible[1:]]
    refined[0] = ("build", 2.65, 2.9)
    refined[3] = ("build", 4.05, 4.25)
    clip = stage._RefinedSpan(2.4, 5.5, tuple(
        SourceWord("", text, first, last) for text, first, last in refined
    ))
    context = ProjectContext(root)
    candidate, = stage._proposals(context, stage._planned(context), refinements={"s1": (clip,)})
    assert candidate.end == pytest.approx(4.2)
    assert candidate.word_ids == ("t1:1", "t1:2", "t1:3")
    write_json(root / "removals.json", {
        "schema": "yap-studio/removals@1", "removals": [{
            "id": "r1", "take": "t1", "start": candidate.start, "end": candidate.end,
            "kind": "restart", "status": "applied", "words": list(candidate.word_ids),
            "listen": candidate.listen, "evidence": candidate.plan,
        }],
    })

    plan = cut_stage.resolve_cut(root)
    retained = {word["id"]: word for word in cut_stage._cut_words(plan)["words"]}
    assert set(retained) == {"t1:0", "t1:4", "t1:5", "t1:6", "t1:7"}
    assert retained["t1:4"]["start"] == pytest.approx(plan.pieces[1].out_f0 / 30)


@pytest.mark.parametrize(("quiet_frames", "expected"), [(14, False), (15, True)])
def test_suspicion_requires_quiet_inside_a_voiced_word(quiet_frames: int, expected: bool) -> None:
    sound = np.full(48_000, 0.05, dtype=np.float32)
    sound[20 * 480:(20 + quiet_frames) * 480] = 0
    words = (SourceWord("t1:0", "garden", 2, 3),)
    segment = stage._PlannedAudio("t1", 2, 3, sound, words)
    assert bool(stage._suspicious_spots(segment)) is expected
    sound[:(20 + quiet_frames) * 480] = 0
    assert not stage._suspicious_spots(segment)
    sound[:] = 0.05
    sound[20 * 480:] = 0
    assert not stage._suspicious_spots(segment)


@pytest.mark.parametrize(("gap", "voiced_frames", "expected"), [
    (0.19, 8, False), (0.2, 7, False), (0.2, 8, True), (0.3, 8, True),
])
def test_suspicion_requires_speech_in_a_long_enough_gap(
    gap: float, voiced_frames: int, expected: bool,
) -> None:
    sound = np.zeros(48_000, dtype=np.float32)
    sound[30 * 480:(30 + voiced_frames) * 480] = 0.05
    words = (SourceWord("t1:0", "but", 2, 2.3),
             SourceWord("t1:1", "we'll", 2.3 + gap, 2.9))
    segment = stage._PlannedAudio("t1", 2, 3, sound, words)
    assert bool(stage._suspicious_spots(segment)) is expected


def test_refinement_merges_context_and_batches_all_segments_once(tmp_path: Path) -> None:
    words = (SourceWord("t1:0", "bright", 2.4, 2.7),
             SourceWord("t1:1", "morning", 3.0, 3.3),
             SourceWord("t1:2", "flowers", 3.6, 3.9))
    sound = np.full(2 * 48_000, 0.05, dtype=np.float32)
    first = stage._PlannedAudio("t1", 2, 4, sound, words)
    second = replace(first, take="t2", words=tuple(replace(word, id=word.id.replace("t1", "t2"))
                                                 for word in words))
    batches: list[list[Path]] = []

    class RecordingBackend(ScriptedBackend):
        """Require both authored segments' merged clips in one speech batch."""

        def transcribe_many(self, paths: Sequence[Path], *, language: str | None = None,
                            ) -> dict[Path, Transcript]:
            batches.append(list(paths))
            return super().transcribe_many(paths, language=language)

    backend = RecordingBackend(lambda _: Transcript([Word("bright", 0, 0.3),
                                                     Word("morning", 0.6, 0.9),
                                                     Word("flowers", 1.2, 1.5)]))
    segments = {"s1": first, "s2": second, "s3": first}
    refined, summary = stage._refine(segments, backend, tmp_path, "en")
    assert summary["spots"] == 6 and summary["clips"] == 2
    assert len(batches) == 1 and len(batches[0]) == 2
    assert set(refined) == set(segments)
    assert all(len(clips) == 1 for clips in refined.values())
    assert all(clips[0].start == 2.4 and clips[0].end == 3.9 for clips in refined.values())
    assert all(word.id == "" for clips in refined.values() for word in clips[0].words)
    assert [word.text for word in first.words] == ["bright", "morning", "flowers"]


def test_missing_refinement_result_refuses_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [("bright", 0.4, 0.7), ("morning", 1.0, 1.3), ("outside", 1.3, 1.6)]
    source = synth.voiced_source(tmp_path / "tones.mov", [(0.4, 1.6)], seconds=2)
    root = make_project(tmp_path / "project", rows, source)

    class MissingBackend(ScriptedBackend):
        """Omit the result for a planted voiced gap."""

        def transcribe_many(self, paths: Sequence[Path], *, language: str | None = None,
                            ) -> dict[Path, Transcript]:
            return {}

    monkeypatch.setattr(stage, "resolve_backend",
                        lambda *_: MissingBackend(lambda _: Transcript([])))
    with pytest.raises(InputError, match="missing refinement transcript"):
        stage.fillers(root)
    assert not (root / "removals.json").exists()


@pytest.mark.parametrize(
    ("all_pass", "expected_batches"),
    [(True, [12, 4, 4]), (False, [12, 16, 32, 4, 4])],
    ids=["all-pass", "one-fails"],
)
def test_repeated_source_removal_tests_every_authored_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    all_pass: bool, expected_batches: list[int],
) -> None:
    root = make_project(tmp_path / "project")
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"].append({"id": "s2", "take": "t1", "from": {"time": 0.56},
                             "to": {"time": 3}})
    write_json(root / "edit.json", edit)
    batches = []

    def script(target: Path) -> Transcript:
        """The longer first occurrence can fail while the shorter repeat passes."""
        with wave.open(str(target)) as stream:
            duration = stream.getnframes() / stream.getframerate()
        control = target.stem.endswith("control")
        first_occurrence = duration > (4.2 if control else 3.8)
        tokens = [row[0] for row in (ROWS if first_occurrence else ROWS[1:])]
        if not control:
            tokens.remove("um")
            if not all_pass and first_occurrence:
                tokens.remove("garden")
        return transcript(tokens)

    class RecordingBackend(ScriptedBackend):
        """Record one batch containing both cut/control pairs per boundary option."""

        def transcribe_many(self, paths: Sequence[Path], *, language: str | None = None,
                            ) -> dict[Path, Transcript]:
            batches.append(list(paths))
            return super().transcribe_many(paths, language=language)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: RecordingBackend(script))
    proposals = stage._proposals(ProjectContext(root), stage._planned(ProjectContext(root)))
    assert len([item for item in proposals if item.kind == "filler"]) == 1
    assert any(item.kind == "silence" for item in proposals)
    entry = stage.fillers(root, apply=True)["entries"][0]
    assert entry["status"] == ("applied" if all_pass else "rejected")
    assert {item["segment"] for item in entry["evidence"]["segments"]} == {"s1", "s2"}
    assert all({item["segment"] for item in attempt["segments"]} == {"s1", "s2"}
               for attempt in entry["evidence"]["attempts"])
    assert [len(batch) for batch in batches] == expected_batches
    if not all_pass:
        failed = next(item for item in entry["evidence"]["segments"] if item["segment"] == "s1")
        assert not failed["pass"] and "garden" not in failed["got"]


def test_repeated_occurrences_require_one_common_passing_nudge(tmp_path: Path) -> None:
    words = (SourceWord("t1:0", "bright", 0.2, 0.5), SourceWord("t1:1", "um", 0.6, 0.8),
             SourceWord("t1:2", "garden", 0.9, 1.1))
    segments = {identifier: stage._PlannedAudio("t1", 0, 2, np.zeros(96_000), words)
                for identifier in ("s1", "s2")}
    candidate = stage.Candidate("t1", "filler", ("t1:1",), 0.6, 0.8, (), "s1", False, {})

    def script(target: Path) -> Transcript:
        """Each occurrence passes a different one-frame nudge, never the same one."""
        if target.stem.endswith("control"):
            return transcript(["bright", "um", "garden"])
        index = int(target.stem.split("-")[2])
        passed = "round-1" in target.stem and index in {0, 3}
        return transcript(["bright", "garden"] if passed else ["bright"])

    result = stage._test([candidate], segments, ScriptedBackend(script), tmp_path, "en")[0]
    assert not result.passed
    assert len(result.evidence["attempts"]) == 13
    for nudge in ([-1, 0], [1, 0]):
        attempt = next(item for item in result.evidence["attempts"] if item["nudge"] == nudge)
        assert sorted(item["pass"] for item in attempt["segments"]) == [False, True]


def test_source_removal_tests_segment_without_its_own_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project")
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"].append({"id": "s2", "take": "t1", "from": {"time": 1.21},
                             "to": {"time": 3}})
    write_json(root / "edit.json", edit)
    planned = stage._planned(ProjectContext(root))
    proposals = stage._proposals(ProjectContext(root), planned)
    filler, = [item for item in proposals if item.kind == "filler"]
    assert filler.segment == "s1"
    # The source time is an exact cut inside "um", so s2 overlaps the removal it did not propose.
    assert filler.start < planned["s2"].start < filler.end

    def script(target: Path) -> Transcript:
        """Fail the padded second occurrence even though it proposed no removal."""
        second = int(target.stem.split("-")[2]) % 2 == 1
        tokens = [row[0] for row in (ROWS[2:] if second else ROWS)]
        if target.stem.endswith("cut"):
            tokens.remove("um")
            if second:
                tokens.remove("garden")
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    entry = stage.fillers(root, apply=True)["entries"][0]
    assert entry["status"] == "rejected"
    assert {item["segment"] for item in entry["evidence"]["segments"]} == {"s1", "s2"}
    assert "garden" in entry["reason"]


def test_source_removal_includes_partially_overlapping_segment(tmp_path: Path) -> None:
    words = (SourceWord("t1:0", "bright", 0.2, 0.5), SourceWord("t1:1", "um", 0.6, 0.8),
             SourceWord("t1:2", "garden", 0.9, 1.1))
    segments = {
        "s1": stage._PlannedAudio("t1", 0, 2, np.zeros(96_000), words),
        "s2": stage._PlannedAudio("t1", 0, 0.75, np.zeros(36_000), words[:2]),
    }
    candidate = stage.Candidate("t1", "filler", ("t1:1",), 0.6, 0.8, (), "s1", False, {})

    def script(target: Path) -> Transcript:
        """The shortened occurrence loses its retained word; the full one passes."""
        second = int(target.stem.split("-")[2]) % 2 == 1
        if target.stem.endswith("control"):
            return transcript(["bright", "um"] if second else ["bright", "um", "garden"])
        return transcript(["wrong"] if second else ["bright", "garden"])

    result = stage._test([candidate], segments, ScriptedBackend(script), tmp_path, "en")[0]
    assert not result.passed
    occurrences = result.evidence["segments"]
    assert {item["segment"] for item in occurrences} == {"s1", "s2"}
    shortened = next(item for item in occurrences if item["segment"] == "s2")
    assert shortened["end"] == 0.75 and not shortened["pass"]


@pytest.mark.parametrize("speech", [False, True], ids=["silence-only", "speech"])
def test_planned_audio_ends_at_settled_stretched_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, speech: bool,
) -> None:
    from yap_studio.stages import cut

    root = make_project(tmp_path / "project", [
        ("bright", 0.25, 0.55), ("morning", 0.6, 0.9), ("garden", 6, 6.3),
    ])
    if not speech:
        project = json.loads((root / "project.json").read_bytes())
        project["asr"]["engine"] = "none"
        write_json(root / "project.json", project)
    target = root / "ingest/t1/words.json"
    words = json.loads(target.read_bytes())
    words["words"][1]["end"] = 5.5
    write_json(target, words)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["words_sha256"]["t1"] = receipt.sha256_file(target)
    edit["segments"][0]["to"] = {"word": "t1:1", "text": "morning"}
    write_json(root / "edit.json", edit)
    backend = ScriptedBackend(lambda _: Transcript([
        Word("bright", 0.25, 0.55), Word("morning", 0.6, 0.9),
    ])) if speech else None
    monkeypatch.setattr(cut, "_edge_backend", lambda _: backend)

    piece, = cut.resolve_cut(root).pieces
    # Without an engine silence alone never moves an edge, so the word-time end stands.
    assert float(piece.src_t1) == pytest.approx(1.1 if speech else 5.7, abs=1 / 30)
    context = ProjectContext(root)
    planned = stage._planned(context)
    segment = planned["s1"]
    assert (segment.start, segment.end) == (float(piece.src_t0), float(piece.src_t1))
    assert len(segment.sound) == round((segment.end - segment.start) * 48_000)
    proposals = stage._proposals(context, planned)
    assert len(proposals) == (1 if not speech else 0)
    if not speech:
        assert proposals[0].kind == "silence"
    assert len(stage.candidates_for_project(root)) == (1 if not speech else 0)
    if not speech:
        assert any(item["kind"] == "silence" for item in stage.fillers(root)["entries"])


def test_planned_audio_snaps_exact_source_time_anchors(tmp_path: Path) -> None:
    from yap_studio.stages.cut import resolve_cut

    root = make_project(tmp_path / "project")
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0].update({"from": {"time": 0.21}, "to": {"time": 2.81}})
    write_json(root / "edit.json", edit)
    piece, = resolve_cut(root).pieces
    segment = stage._planned(ProjectContext(root))["s1"]
    assert (segment.start, segment.end) == (0.2, 2.8)
    assert (segment.start, segment.end) == (float(piece.src_t0), float(piece.src_t1))
    assert len(segment.sound) == round((segment.end - segment.start) * 48_000)


def test_planned_audio_shares_continuous_segment_boundary(tmp_path: Path) -> None:
    from yap_studio.stages.cut import resolve_cut

    root = make_project(tmp_path / "project", [
        ("bright", 0.25, 0.55), ("morning", 0.6, 0.9),
        ("garden", 1.5, 1.8), ("flowers", 1.85, 2.15),
    ])
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"] = [
        {"id": "s1", "take": "t1", "from": {"time": 0}, "to": {"word": "t1:1"}},
        {"id": "s2", "take": "t1", "from": {"word": "t1:2"}, "to": {"time": 2.5}},
    ]
    write_json(root / "edit.json", edit)
    left, right = resolve_cut(root).pieces
    planned = stage._planned(ProjectContext(root))
    assert left.src_t1 == right.src_t0
    assert planned["s1"].end == planned["s2"].start == float(left.src_t1)
    assert planned["s1"].start == float(left.src_t0)
    assert planned["s2"].end == float(right.src_t1)


@pytest.mark.parametrize("cached", [True, False])
def test_planned_audio_reuses_pause_threshold_or_loads_once_per_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool,
) -> None:
    root = make_project(tmp_path / "project")
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"].append({**edit["segments"][0], "id": "s2"})
    write_json(root / "edit.json", edit)
    pause = edges.PauseThreshold(-50.04, "floor", -65.04, None, len(ROWS))
    monkeypatch.setattr(edges, "choose_pause_threshold", lambda level, midpoints_s: pause)
    context = ProjectContext(root)
    plan = cut_stage._resolve(context)
    assert plan.pauses["t1"] is pause
    expected = plan.pause_thresholds["t1"]
    assert expected["threshold_db"] == -50.0
    if not cached:
        plan = replace(plan, pauses={}, pause_thresholds={})
    monkeypatch.setattr(cut_stage, "_resolve", lambda _: plan)
    calls: list[tuple[Path, int | None, Sequence[tuple[float, float]]]] = []
    load = edges.load_take

    def record_load(source: Path, stream: int | None,
                    words: Sequence[tuple[float, float]] = ()) -> edges._Take:
        calls.append((source, stream, words))
        return load(source, stream, words)

    monkeypatch.setattr(edges, "load_take", record_load)
    planned = stage._planned(context)
    assert set(planned) == {"s1", "s2"}
    assert planned["s1"].pause is planned["s2"].pause
    for segment in planned.values():
        assert segment.pause == expected
        assert segment.silence_db == -50.04
    assert calls == ([] if cached else [
        (context.take_path("t1"), cut_stage._selected_stream(context, "t1"),
         [(word.start, word.end) for word in plan.source_words["t1"].words]),
    ])


def test_proposals_trim_noisy_bed_but_keep_padded_speech(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [(f"word{index}", 2.2 + index * 0.6, 2.4 + index * 0.6) for index in range(12)]
    rate, duration = stage._RATE, 9.5
    times = np.arange(round(duration * rate)) / rate
    sound = 10 ** (-38 / 20) * np.sqrt(2) * np.sin(2 * np.pi * 400 * times)
    for _, first, last in rows:
        start, end = round((first - 0.13) * rate), round((last + 0.13) * rate)
        sound[start:end] += 0.1 * np.sin(2 * np.pi * 200 * times[start:end])
    wav, source = tmp_path / "noisy.wav", tmp_path / "noisy.mov"
    synth._write_wav(wav, sound, rate)
    synth._run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                f"testsrc2=s=160x90:r=30:d={duration}", "-i", str(wav),
                "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                "-c:v", "libx264", "-c:a", "pcm_s16le", str(source)])
    root = make_project(tmp_path / "project", rows, source)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 0.2}
    write_json(root / "project.json", project)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["from"] = {"time": 2.0}
    write_json(root / "edit.json", edit)
    context = ProjectContext(root)
    planned = stage._planned(context)
    segment = planned["s1"]
    assert segment.start == 2.0
    assert segment.pause is not None and segment.pause["rule"] == "words"
    assert segment.pause["floor_db"] == pytest.approx(-38, abs=0.1)
    assert segment.silence_db > -38
    assert len(stage.detect(segment.words, max_pause_s=0.2).candidates) == len(rows) - 1

    candidates = stage._proposals(context, planned)
    assert any(candidate.kind == "silence" for candidate in candidates)
    assert all(candidate.kind == "silence" and candidate.word_ids == ()
               for candidate in candidates)
    assert all(candidate.end <= first - 0.13 or candidate.start >= last + 0.13
               for candidate in candidates for _, first, last in rows)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: None)
    result = stage.fillers(root)
    assert result["pause_thresholds"] == {"t1": segment.pause}
    owned = json.loads((root / "fillers/receipt.json").read_bytes())
    assert owned["pause_thresholds"] == result["pause_thresholds"]


def test_measured_pause_keeps_quiet_speech_between_loud_words(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project", [
        ("bright", 0.2, 0.5), ("garden", 1.5, 1.8),
    ])
    rate = stage._RATE
    sound = np.zeros(2 * rate, dtype=np.float32)
    for first, last, db in [(0.2, 0.5, -20), (0.9, 1.1, -48), (1.5, 1.8, -20)]:
        low, high = round(first * rate), round(last * rate)
        sound[low:high] = 10 ** (db / 20) * np.sqrt(2) * np.sin(
            2 * np.pi * 200 * np.arange(high - low) / rate
        )
    context = ProjectContext(root)
    original = stage._planned(context)["s1"]
    segment = replace(original, end=2.0, sound=sound, silence_db=-35,
                      pause={"floor_db": -65.0})
    runs = stage._measured_silent_runs(segment)(0.5, 1.5)
    assert len(runs) == 2
    assert runs[0] == pytest.approx((0.5, 0.9))
    assert runs[1] == pytest.approx((1.1, 1.5))
    pauses = [item for item in stage._proposals(context, {"s1": segment})
              if item.kind == "silence" and item.end <= 1.5]
    assert len(pauses) == 2
    assert all(item.end <= 0.9 or item.start >= 1.1 for item in pauses)


@pytest.mark.parametrize(("silence_db", "quiet"), [(-40.0, True), (-55.0, False)])
def test_trial_uses_segment_threshold_for_quiet_and_listen(silence_db: float, quiet: bool) -> None:
    segment = stage._PlannedAudio(
        "t1", 2, 4, np.full(2 * stage._RATE, 10 ** (-50 / 20)), (), silence_db=silence_db,
    )
    candidate = stage.Candidate("t1", "silence", (), 2.5, 3.0, (), "s1", False, {})
    trial = stage._trial(segment, (candidate,))
    assert trial is not None
    assert bool(trial.quiet) is quiet
    assert bool(trial.listen) is not quiet


@pytest.mark.parametrize(("silence_db", "suspicious"), [(-40.0, False), (-55.0, True)])
def test_suspicious_spots_uses_segment_threshold(silence_db: float, suspicious: bool) -> None:
    words = (SourceWord("t1:0", "bright", 2.2, 2.4),
             SourceWord("t1:1", "garden", 2.8, 3.0))
    segment = stage._PlannedAudio(
        "t1", 2, 4, np.full(2 * stage._RATE, 10 ** (-50 / 20)), words, silence_db=silence_db,
    )
    assert bool(stage._suspicious_spots(segment)) is suspicious


def test_proposals_use_planned_words_and_bounds(tmp_path: Path) -> None:
    root = make_project(tmp_path / "project", [
        ("bright", 0.3, 0.6), ("um", 0.65, 0.85),
        ("uh", 0.95, 1.15), ("garden", 1.5, 1.8),
    ])
    context = ProjectContext(root)
    planned = stage._planned(context)
    segment = planned["s1"]
    # The settled range keeps "um"; authored anchors would merge it with the dropped "uh".
    planned["s1"] = replace(segment, end=0.9, words=segment.words[:2],
                            sound=segment.sound[:43_200])
    candidate, = stage._proposals(context, planned)
    assert candidate.word_ids == ("t1:1",)
    assert planned["s1"].start <= candidate.start < candidate.end <= planned["s1"].end


def test_project_candidates_use_settled_ranges_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project")
    context = ProjectContext(root)
    planned = stage._planned(context)
    segment = planned["s1"]
    planned["s1"] = replace(segment, end=1.4, words=segment.words[:3],
                            sound=segment.sound[:67_200])
    monkeypatch.setattr(stage, "_planned", lambda _: planned)
    before = {target.relative_to(root): target.read_bytes()
              for target in root.rglob("*") if target.is_file()}
    candidate, = stage.candidates_for_project(root)
    assert candidate.word_ids == ("t1:2",)
    assert candidate.end <= planned["s1"].end
    assert stage.candidates_for_project(root, protect=((1.1, 1.3),)) == []
    assert before == {target.relative_to(root): target.read_bytes()
                      for target in root.rglob("*") if target.is_file()}


def test_project_shortens_silence_after_stretched_word(tmp_path: Path) -> None:
    source = synth.voiced_source(
        tmp_path / "source.mov", [(0.5, 0.8), (2.0, 2.3)], seconds=2.9,
    )
    root = make_project(tmp_path / "project", [
        ("tank.", 0.5, 1.7), ("next", 2.0, 2.3),
    ], source=source)
    segment = stage._planned(ProjectContext(root))["s1"]

    assert segment.words[0].end == pytest.approx(0.87)
    candidates = stage._proposals(ProjectContext(root), {"s1": segment})
    silence = [item for item in candidates if item.kind == "silence" and item.end <= 2.0]
    assert silence
    assert all(item.word_ids == () and 0.8 <= item.start < item.end <= 2.0
               for item in silence)
    assert sum(item.end - item.start for item in silence) == pytest.approx(
        2.0 - segment.words[0].end - 0.3, abs=2 / 30,
    )


def test_project_shortens_silence_after_stretched_final_word(tmp_path: Path) -> None:
    source = synth.voiced_source(tmp_path / "source.mov", [(0.5, 0.8)], seconds=2.5)
    root = make_project(tmp_path / "project", [("tank.", 0.5, 1.7)], source=source)
    segment = stage._planned(ProjectContext(root))["s1"]
    assert segment.words[0].end == pytest.approx(0.87)
    candidates = stage._proposals(ProjectContext(root), {"s1": segment})
    silence = [item for item in candidates if item.kind == "silence"]
    assert silence
    assert all(item.word_ids == () and segment.words[0].end <= item.start
               < item.end <= segment.end for item in silence)
    assert sum(item.end - item.start for item in silence) == pytest.approx(
        segment.end - segment.words[0].end - 0.3, abs=2 / 30,
    )
    assert not [item for item in stage._proposals(
        ProjectContext(root), {"s1": segment}, protect=((segment.words[0].end, segment.end),)
    ) if item.kind == "silence"]
    project = json.loads((root / "project.json").read_text())
    project["rules"] = {"max_pause_s": segment.end - segment.words[0].end}
    write_json(root / "project.json", project)
    assert not [item for item in stage._proposals(
        ProjectContext(root), {"s1": segment}
    ) if item.kind == "silence"]


def test_planned_audio_keeps_authored_segment_merged_by_cut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages.cut import resolve_cut

    root = make_project(tmp_path / "project")
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"] = [
        {"id": "intro", "take": "t1", "from": {"time": 0}, "to": {"time": 0.2}},
        {"id": "speech", "take": "t1", "from": {"time": 0.2}, "to": {"time": 3}},
    ]
    write_json(root / "edit.json", edit)
    assert {piece.segment_id for piece in resolve_cut(root).pieces} == {"speech"}
    planned = stage._planned(ProjectContext(root))
    assert set(planned) == {"intro", "speech"}
    assert planned["intro"].words == () and len(planned["intro"].sound) == 9_600
    assert planned["speech"].start == 0.2
    install_engine(monkeypatch)
    result = stage.fillers(root, apply=True)
    assert result["entries"][0]["status"] == "applied"
    assert result["entries"][0]["evidence"]["segments"][0]["segment"] == "speech"


def test_clipped_next_word_retries_nudges_and_rejects_with_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project")
    batches = install_engine(monkeypatch, clipped=True)
    result = stage.fillers(root, apply=True)
    entry = result["entries"][0]
    assert entry["status"] == "rejected"
    attempts = entry["evidence"]["attempts"]
    assert [item["nudge"] for item in attempts] == [list(nudge) for level in stage._NUDGES
                                                  for nudge in level]
    assert [len(batch) for batch in batches] == [4, 16, 32]
    assert "garden" in entry["reason"]
    assert "garden" in entry["evidence"]["expected"]
    assert "garden" not in entry["evidence"]["got"]


@pytest.mark.parametrize("changed_inputs", [False, True], ids=["noop", "changed-inputs"])
def test_rejected_removal_survives_same_identity_rerun(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch,
                                                      changed_inputs: bool) -> None:
    root = make_project(tmp_path / "project")
    suppress_tail_pause(root)
    install_engine(monkeypatch, clipped=True)
    rejected = stage.fillers(root, apply=True)["entries"][0]
    before = (root / "removals.json").read_bytes()
    if changed_inputs:
        project = json.loads((root / "project.json").read_bytes())
        project.setdefault("rules", {})["max_pause_s"] = 0.61
        write_json(root / "project.json", project)
    batches = install_engine(monkeypatch)
    rerun = stage.fillers(root, apply=True)
    assert json.loads(json.dumps(rerun["entries"])) == [rejected]
    assert (root / "removals.json").read_bytes() == before
    assert rerun["noop"] == (not changed_inputs)
    assert not batches


def test_machine_rejection_is_retested_and_can_pass_on_force_with_same_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project")
    suppress_tail_pause(root)
    install_engine(monkeypatch, clipped=True)
    rejected = stage.fillers(root, apply=True)["entries"][0]
    assert rejected["status"] == "rejected" and rejected["by"] == "fillers"
    assert "decision" not in rejected["evidence"]
    batches = install_engine(monkeypatch)
    rerun = stage.fillers(root, apply=True, force=True)
    assert not rerun["noop"]
    assert len(rerun["entries"]) == 1
    applied = rerun["entries"][0]
    assert applied["id"] == rejected["id"]
    assert applied["status"] == "applied" and applied.get("reason") is None
    assert applied["evidence"]["words"] == "pass"
    assert len(applied["evidence"]["attempts"]) == 1
    assert [len(batch) for batch in batches] == [2]


def test_creator_rejection_is_not_retested_on_force(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path / "project")
    suppress_tail_pause(root)
    install_engine(monkeypatch, clipped=True)
    rejected = stage.fillers(root, apply=True)["entries"][0]
    # Confirming an existing machine rejection must record the creator's decision.
    creator = stage.set_removal(root, rejected["id"], "rejected")["removals"][0]
    assert creator["evidence"]["decision"] is True
    batches = install_engine(monkeypatch)
    assert stage.fillers(root, apply=True, force=True)["entries"] == [creator]
    assert not batches


def test_manual_rejection_without_decision_is_not_retested_on_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project", [("bright", 0.3, 0.6), ("morning", 0.65, 0.9)])
    suppress_tail_pause(root)
    stage.add_manual(root, "t1", 0.65, 0.9)
    def reject_manual(data: dict[str, Any]) -> str:
        """Create a saved manual rejection whose provenance is its manual author."""
        data["removals"][0]["status"] = "rejected"
        data["removals"][0]["reason"] = "keep morning"
        return str(data["removals"][0]["id"])

    rejected = stage._change(root, reject_manual)["removals"][0]
    assert rejected["status"] == "rejected" and rejected["by"] == "manual"
    assert "decision" not in rejected["evidence"]
    batches = install_engine(monkeypatch)
    assert stage.fillers(root, apply=True, force=True)["entries"] == [rejected]
    assert not batches


def test_manual_removal_survives_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    stage.fillers(root, apply=True)
    added = stage.add_manual(root, "t1", 2.7, 2.8, reason="chosen trim")
    manual = added["removals"][-1]
    assert manual["kind"] == "manual" and manual["by"] == "manual"
    assert manual in stage.fillers(root, apply=True)["entries"]


def test_zero_pause_voiced_join_is_flagged_for_listening(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [("bright", 0.3, 1), ("um", 1, 1.3), ("garden", 1.3, 2), ("flowers", 2, 2.4)]
    root = make_project(tmp_path / "project", rows)
    engine = ScriptedBackend(lambda target: transcript(
        ["bright", "um", "garden", "flowers"] if target.stem.endswith("control")
        else ["bright", "garden", "flowers"],
    ))
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: engine)
    entry = stage.fillers(root, apply=True)["entries"][0]
    assert entry["status"] == "applied" and entry["listen"]


def test_no_engine_applies_only_pauses_and_reports_skipped(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [*ROWS, ("later", 3.5, 3.8)]
    root = make_project(tmp_path / "project", rows)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: None)
    result = stage.fillers(root)
    assert result["skipped"] == 1
    assert result["notes"] == ["fillers skipped: no speech engine",
                               "restart refinement skipped: no speech engine"]
    assert [(item["kind"], item["status"]) for item in result["entries"]] == [
        ("silence", "applied"), ("silence", "rejected"),
    ]


def test_no_engine_keeps_measured_final_pause_without_speech_proposals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = synth.voiced_source(tmp_path / "source.mov", [(0.4, 0.7), (1.0, 1.2)],
                                 seconds=2.5)
    root = make_project(tmp_path / "project", [("bright", 0.4, 0.7),
                                                ("um", 1.0, 1.2)], source)
    context = ProjectContext(root)
    segment = stage._planned(context)["s1"]
    measured = [item for item in stage._proposals(context, {"s1": segment})
                if item.kind == "silence"
                and item.plan["right_word_start_s"] == segment.end]
    assert len(measured) == 1
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: None)

    result = stage.fillers(root)

    assert result["skipped"] >= 1
    assert result["entries"]
    assert all(item["kind"] == "silence" and item["words"] == []
               for item in result["entries"])
    assert any(item["start"] == pytest.approx(measured[0].start)
               and item["end"] == pytest.approx(measured[0].end)
               for item in result["entries"]), [
                   (item["start"], item["end"], item["status"]) for item in result["entries"]]


def test_no_engine_keeps_measured_pause_touching_filler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [("bright", 0.25, 0.55), ("um", 1.4, 1.6), ("garden", 1.7, 2.0)]
    root = make_project(tmp_path / "project", rows)
    project = json.loads((root / "project.json").read_bytes())
    project["rules"] = {"max_pause_s": 0}
    write_json(root / "project.json", project)
    context = ProjectContext(root)
    segment = stage._planned(context)["s1"]
    full = stage._proposals(context, {"s1": segment})
    measured_runs = stage._measured_silent_runs(segment)
    pause_only = stage.detect(segment.words, language="und", take=segment.take,
                              max_pause_s=0, silent_runs=measured_runs)
    measured = [item for item in pause_only.candidates if item.kind == "silence"
                and item.plan["right_word_start_s"] == 1.4]
    assert len(measured) == 1
    assert measured[0].word_ids == ()
    assert any(item.kind == "filler" and item.start <= measured[0].start
               and item.end >= measured[0].end for item in full), full
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: None)

    result = stage.fillers(root)

    assert result["skipped"] >= 1
    assert any(item["kind"] == "silence" and item["words"] == []
               and item["start"] == pytest.approx(measured[0].start)
               and item["end"] == pytest.approx(measured[0].end)
               for item in result["entries"])
    assert all(item["kind"] == "silence" for item in result["entries"])


def test_removed_optional_filler_must_be_absent(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch, leftover=True)
    entry = stage.fillers(root, apply=True)["entries"][0]
    assert entry["status"] == "rejected"
    assert "removed words remain" in entry["reason"]


def test_unverifiable_control_is_last_resort_with_listen(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path / "project")
    engine = ScriptedBackend(lambda _: transcript(["bright", "morning", "flowers", "grow"]))
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: engine)
    result = stage.fillers(root, apply=True)
    entry = result["entries"][0]
    assert entry["status"] == "proposed" and entry["listen"]
    assert entry["evidence"]["words"] == "unverifiable"
    assert len(entry["evidence"]["attempts"]) == 13


def test_applied_removal_keeps_reason_after_unverifiable_retest(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    candidate = stage.Candidate("t1", "filler", ("t1:2",), 1.1, 1.3, (), "s1", False, {})
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: [candidate])
    install_engine(monkeypatch)
    saved, = stage.fillers(root, apply=True)["entries"]
    assert saved["status"] == "applied" and saved.get("reason") is None
    engine = ScriptedBackend(lambda _: transcript(["bright", "morning", "flowers", "grow"]))
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: engine)

    entry, = stage.fillers(root, apply=True, force=True)["entries"]

    assert entry["id"] == saved["id"] and entry["status"] == "applied"
    assert entry["evidence"]["words"] == "unverifiable" and entry["listen"]
    assert entry.get("reason") is None


def test_unverifiable_group_stays_proposed_with_apply(tmp_path, monkeypatch) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    candidates = [
        stage.Candidate("t1", "filler", ("t1:2",), 1.1, 1.3, (), "s1", False, {}),
        stage.Candidate("t1", "silence", (), 1.31, 1.44, (), "s1", False, {}),
    ]
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: candidates)
    original = stage._source_round
    group_rounds = []

    def source_round(*args):
        results = original(*args)
        if args[-1] >= 100:
            group_rounds.append(args[-1])
            return [{**item, "pass": False, "unverifiable": True, "words": "unverifiable"}
                    for item in results]
        return results

    monkeypatch.setattr(stage, "_source_round", source_round)
    entries = stage.fillers(root, apply=True)["entries"]
    assert group_rounds
    assert entries and all(item["status"] == "proposed" for item in entries)
    assert all(item["listen"] and "could not be verified" in item["reason"] for item in entries)


@pytest.mark.parametrize("apply", [False, True])
def test_propose_only_reuses_receipt_whatever_apply_flag(tmp_path, monkeypatch, apply):
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    saved = stage.fillers(root, apply=apply)
    before = (root / "removals.json").read_bytes()
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: pytest.fail("receipt is current"))
    result = stage.fillers(root, apply=not apply, propose_only=True)
    assert result["noop"]
    assert json.loads(json.dumps(result["entries"])) == saved["entries"]
    assert (root / "removals.json").read_bytes() == before


def test_propose_only_preserves_saved_applied_boundaries(tmp_path, monkeypatch):
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    saved = stage.fillers(root, apply=True)["entries"]
    result = stage.fillers(root, apply=True, force=True, propose_only=True)
    applied = [item for item in result["entries"] if item["status"] == "applied"]
    assert [(item["take"], item["start"], item["end"]) for item in applied] == [
        (item["take"], item["start"], item["end"]) for item in saved]


def test_smallest_passing_nudge_keeps_identity_on_rerun(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path / "project")

    def script(target: Path) -> Transcript:
        """Fail the original pair and pass all one-frame alternatives."""
        tokens = [row[0] for row in ROWS]
        if target.stem.endswith("cut"):
            tokens.remove("um")
            if "round-0" in target.stem:
                tokens.remove("garden")
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    first = stage.fillers(root, apply=True)["entries"][0]
    assert sum(abs(value) for value in first["evidence"]["nudge"]) == 1
    assert stage.fillers(root, apply=True, force=True)["entries"][0]["id"] == first["id"]


@pytest.mark.parametrize("apply", [False, True], ids=["inspect", "apply"])
def test_saved_applied_nudge_changes_interval_only_with_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apply: bool,
) -> None:
    from yap_studio.stages.cut import resolve_cut

    root = make_project(tmp_path / "project")
    candidate = stage.Candidate("t1", "filler", ("t1:2",), 1.1, 1.3, (), "s1", False, {})
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: [candidate])
    install_engine(monkeypatch)
    saved = stage.fillers(root, apply=True)["entries"][0]

    def removed_intervals() -> bytes:
        """Serialize the removed frame ranges consumed by the following cut."""
        pieces = resolve_cut(root).pieces
        return json.dumps([(left.src_f1, right.src_f0)
                           for left, right in zip(pieces, pieces[1:], strict=False)]).encode()

    before = removed_intervals()
    assert before == b"[[33, 39]]"

    def script(target: Path) -> Transcript:
        """Fail the saved join and pass only the one-frame expansion at its start."""
        tokens = [row[0] for row in ROWS]
        if target.stem.endswith("cut"):
            tokens.remove("um")
            if not target.stem.startswith("round-1-0-"):
                tokens.remove("garden")
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    entries = stage.fillers(root, apply=apply, force=True)["entries"]
    assert len({item["id"] for item in entries}) == len(entries)
    if apply:
        assert len(entries) == 1
        shifted = entries[0]
        assert shifted["id"] == saved["id"] and shifted["status"] == "applied"
        assert removed_intervals() == b"[[32, 39]]"
    else:
        assert next(item for item in entries if item["id"] == saved["id"]) == saved
        shifted = next(item for item in entries if item["status"] == "proposed")
        assert shifted["evidence"]["applied_parent"] == saved["id"]
        assert len(entries) == 2
        assert removed_intervals() == before
    assert (shifted["start"], shifted["end"]) == (1.1 - 1 / 30, 1.3)


def test_passing_saved_boundary_inspection_leaves_removals_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project")
    suppress_tail_pause(root)
    install_engine(monkeypatch)
    saved = stage.fillers(root, apply=True)["entries"]
    before = (root / "removals.json").read_bytes()
    result = stage.fillers(root, force=True)
    assert not result["noop"]
    assert result["entries"] == saved
    assert (root / "removals.json").read_bytes() == before


@pytest.mark.parametrize("different_boundary", [False, True], ids=["same", "different"])
def test_creator_rejected_inspection_boundary_is_not_proposed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    different_boundary: bool,
) -> None:
    root = make_project(tmp_path / "project")
    candidate = stage.Candidate("t1", "filler", ("t1:2",), 1.1, 1.3, (), "s1", False, {})
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: [candidate])
    install_engine(monkeypatch)
    saved = stage.fillers(root, apply=True)["entries"][0]
    passing_round = 1

    def script(target: Path) -> Transcript:
        """Pass only the selected expansion at the saved removal's start."""
        tokens = [row[0] for row in ROWS]
        if target.stem.endswith("cut"):
            tokens.remove("um")
            if not target.stem.startswith(f"round-{passing_round}-0-"):
                tokens.remove("garden")
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    proposal = next(item for item in stage.fillers(root, force=True)["entries"]
                    if item["status"] == "proposed")
    assert proposal["evidence"]["applied_parent"] == saved["id"]
    rejected = next(item for item in stage.set_removal(
        root, proposal["id"], "rejected", reason="keep the saved boundary",
    )["removals"] if item["id"] == proposal["id"])
    assert rejected["evidence"]["decision"] is True
    if different_boundary:
        passing_round = 2
    entries = stage.fillers(root, force=True)["entries"]
    assert next(item for item in entries if item["id"] == saved["id"]) == saved
    assert next(item for item in entries if item["id"] == rejected["id"]) == rejected
    assert len({item["id"] for item in entries}) == len(entries)
    proposed = [item for item in entries if item["status"] == "proposed"]
    if different_boundary:
        new, = proposed
        assert new["evidence"]["applied_parent"] == saved["id"]
        assert (new["start"], new["end"]) == (1.1 - 2 / 30, 1.3)
        assert len(entries) == 3
    else:
        assert not proposed
        assert len(entries) == 2


@pytest.mark.parametrize("edge", ["start", "end"])
def test_segment_edge_filler_absorbs_silent_sliver_without_short_piece(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge: str,
) -> None:
    from yap_studio.stages.cut import resolve_cut

    rows = ([("like", 2 / 30, 8 / 30), ("bright", 0.7, 1)] if edge == "start"
            else [("bright", 0.3, 0.6), ("like", 1, 1.2)])
    root = make_project(tmp_path / "project", rows)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["to"]["time"] = 1.2 if edge == "start" else 38 / 30
    write_json(root / "edit.json", edit)
    batches = []

    class RecordingBackend(ScriptedBackend):
        """Ensure quiet edge absorption needs no second speech batch."""

        def transcribe_many(self, paths: Sequence[Path], *, language: str | None = None,
                            ) -> dict[Path, Transcript]:
            batches.append(list(paths))
            return super().transcribe_many(paths, language=language)

    engine = RecordingBackend(lambda target: transcript(
        [row[0] for row in rows] if target.stem.endswith("control") else ["bright"],
    ))
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: engine)
    entry = stage.fillers(root, apply=True)["entries"][0]
    assert entry["status"] == "applied"
    if edge == "start":
        assert entry["start"] == 0
        assert entry["evidence"]["absorbed_s"] == [[0, 2 / 30]]
    else:
        assert entry["end"] == pytest.approx(38 / 30)
        assert entry["evidence"]["absorbed_s"] == [[1.2, 38 / 30]]
    assert len(batches) == 1
    plan = resolve_cut(root)
    assert len(plan.pieces) == 1
    assert all(float(piece.src_t1 - piece.src_t0) >= 0.25 for piece in plan.pieces)
    assert not any("short piece kept" in warning for warning in plan.warnings)


@pytest.mark.parametrize("passes", [True, False], ids=["passes", "fails"])
def test_voiced_edge_sliver_requires_extended_join_test(tmp_path: Path, passes: bool) -> None:
    words = (SourceWord("t1:0", "bright", 0.3, 0.6), SourceWord("t1:1", "like", 1, 1.2))
    segment = stage._PlannedAudio("t1", 0, 38 / 30, np.full(60_800, 0.1), words)
    candidate = stage.Candidate("t1", "filler", ("t1:1",), 1, 1.2, (), "s1", False, {})
    tested = stage._Tested(candidate, True, False, {
        "identity": stage.candidate_id(candidate), "attempts": [], "words": "pass",
    })
    engine = ScriptedBackend(lambda target: transcript(
        ["bright", "like"] if target.stem.endswith("control")
        else ["bright"] if passes else ["wrong"],
    ))
    result = stage._slivers([tested], {"s1": segment}, engine, tmp_path, "en")[0]
    assert result.passed == passes
    assert result.candidate.end == segment.end
    assert len(result.evidence["attempts"]) == 1
    assert result.evidence["attempts"][0]["absorbed_s"] == [[1.2, 38 / 30]]


@pytest.mark.parametrize("edge", ["start", "end"])
def test_removed_word_voiced_head_or_tail_is_absorbed_without_retest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge: str,
) -> None:
    from yap_studio.stages.cut import resolve_cut

    rows = ([("like", 0, 7 / 30), ("bright", 0.5, 0.7), ("time", 0.8, 1)]
            if edge == "start" else
            [("bright", 0.4, 0.6), ("time", 0.8, 1.05), ("like", 1.1, 41 / 30)])
    root = make_project(tmp_path / "project", rows)
    if edge == "end":
        edit = json.loads((root / "edit.json").read_bytes())
        # Resolve the same frame without triggering the authored word-end padding.
        edit["segments"][0]["to"]["time"] = 41 / 30 + 0.000001
        write_json(root / "edit.json", edit)
    first, last, word_id = (2 / 30, 7 / 30, "t1:0") if edge == "start" else (
        1.1, 1.3, "t1:2",
    )
    candidate = stage.Candidate("t1", "filler", (word_id,), first, last, (), "s1", False, {})
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: [candidate])
    batches = []

    class RecordingBackend(ScriptedBackend):
        """A second speech round would reproduce the reported false rejection."""

        def transcribe_many(self, paths: Sequence[Path], *, language: str | None = None,
                            ) -> dict[Path, Transcript]:
            batches.append(list(paths))
            assert len(batches) == 1
            return super().transcribe_many(paths, language=language)

    engine = RecordingBackend(lambda target: transcript(
        [row[0] for row in rows] if target.stem.endswith("control") else ["bright", "time"],
    ))
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: engine)
    planned = stage._planned(ProjectContext(root))["s1"]
    absorbed = (0, first) if edge == "start" else (last, planned.end)
    assert stage.audio.rms_db(planned.sound, sum(absorbed) / 2, 48_000,
                              (absorbed[1] - absorbed[0]) / 2) > -45
    entry = stage.fillers(root, apply=True)["entries"][0]
    assert entry["status"] == "applied"
    assert entry["start"] == (0 if edge == "start" else first)
    assert entry["end"] == (last if edge == "start" else 41 / 30)
    assert entry["evidence"]["absorbed_s"] == [list(absorbed)]
    assert len(entry["evidence"]["attempts"]) == 1
    plan = resolve_cut(root)
    assert len(plan.pieces) == 1
    assert not any("short piece kept" in warning for warning in plan.warnings)


@pytest.mark.parametrize("edge", ["start", "end"])
@pytest.mark.parametrize("outcome", [
    "pass", "outer-partial", "missing-kept", "weak-control", "leftover",
])
def test_unrelated_voiced_edge_retests_only_retained_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edge: str, outcome: str,
) -> None:
    tokens = ["don't", "get", "that", "time"]
    if edge == "end":
        # The first word straddles the outer window; the last touches the removal.
        words = tuple(SourceWord(f"t1:{index}", token, first, last)
                      for index, (token, first, last) in enumerate([
                          ("don't", 0.15, 0.3), ("get", 0.4, 0.55),
                          ("that", 0.7, 0.85), ("time", 1.0, 1.2), ("like", 1.2, 1.4),
                      ]))
        candidate = stage.Candidate("t1", "filler", ("t1:4",), 1.2, 1.4, (), "s1", False, {})
        segment_end, expected_window = 44 / 30, (0.2, 1.2)
    else:
        # The last word straddles the outer window; the first touches the removal.
        words = tuple(SourceWord(f"t1:{index}", token, first, last)
                      for index, (token, first, last) in enumerate([
                          ("like", 2 / 30, 0.3), ("don't", 0.3, 0.45),
                          ("get", 0.6, 0.75), ("that", 0.9, 1.05), ("time", 1.25, 1.34),
                      ]))
        candidate = stage.Candidate("t1", "filler", ("t1:0",), 2 / 30, 0.3, (), "s1", False, {})
        segment_end, expected_window = 1.8, (0.3, 1.3)
    segment = stage._PlannedAudio("t1", 0, segment_end,
                                  np.full(round(segment_end * 48_000), 0.1), words)
    tested = stage._Tested(candidate, True, False, {
        "identity": stage.candidate_id(candidate), "attempts": [], "words": "pass",
    })
    recorded = []
    original_trial = stage._trial

    def record_trial(*args: Any, **kwargs: Any) -> Any:
        """Inspect the actual retained-side PCM supplied to the speech batch."""
        trial = original_trial(*args, **kwargs)
        recorded.append(trial)
        return trial

    def script(target: Path) -> Transcript:
        """Only an outer partial word may be absent from the cut transcript."""
        got = list(tokens)
        if outcome == "weak-control":
            got.pop(-1 if edge == "end" else 0)
        elif target.stem.endswith("cut"):
            if outcome == "outer-partial":
                got.pop(0 if edge == "end" else -1)
            elif outcome == "missing-kept":
                got.pop(-1 if edge == "end" else 0)
            elif outcome == "leftover":
                got.append("like")
        return transcript(got)

    monkeypatch.setattr(stage, "_trial", record_trial)
    result = stage._slivers([tested], {"s1": segment}, ScriptedBackend(script), tmp_path, "en")[0]
    assert len(recorded) == 1
    trial = recorded[0]
    assert trial.edge == edge
    assert (trial.start, trial.end) == pytest.approx(expected_window)
    assert len(trial.cut) == len(trial.control) == 3 * 48_000
    assert trial.joins == ()
    assert [word.text for word in trial.stored] == tokens
    assert result.passed == (outcome in {"pass", "outer-partial"})
    assert len(result.evidence["attempts"]) == 1
    assert result.evidence["clicks_db"] == []
    assert result.evidence["removed_words"] == ["like"]
    if outcome in {"missing-kept", "weak-control"}:
        assert "missing join words" in str(result.evidence["diff"])
    elif outcome == "leftover":
        assert "removed words remain" in str(result.evidence["diff"])


def test_edge_sliver_with_word_midpoint_is_retained(tmp_path: Path) -> None:
    words = (SourceWord("t1:0", "um", 0.5, 0.7), SourceWord("t1:1", "bright", 0.7, 0.8))
    segment = stage._PlannedAudio("t1", 0, 0.9, np.zeros(43_200), words)
    candidate = stage.Candidate("t1", "filler", ("t1:0",), 0.5, 0.7, (), "s1", False, {})
    tested = stage._Tested(candidate, True, False, {"attempts": []})
    assert stage._slivers([tested], {"s1": segment}, None, tmp_path, "en") == [tested]


@pytest.mark.parametrize("merged_passes", [True, False], ids=["merged", "merged-fails"])
def test_wordless_gap_merges_fillers_or_rejects_later_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, merged_passes: bool,
) -> None:
    rows = [("bright", 0.3, 0.6), ("um", 0.9, 1.1), ("uh", 1.2, 1.4),
            ("garden", 1.7, 2)]
    root = make_project(tmp_path / "project", rows)
    candidates = [stage.Candidate("t1", "filler", (f"t1:{index}",), rows[index][1],
                                  rows[index][2], (f"rule {index}",), "s1", index == 2, {})
                  for index in (1, 2)]
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: candidates)
    batches = []

    def script(target: Path) -> Transcript:
        """Preserve the other filler in individual trials and test both in the merge."""
        if target.stem.endswith("control"):
            return transcript([row[0] for row in rows])
        if "round-0" in target.stem and len(batches[-1]) == 4:
            return transcript(["bright", "uh", "garden"] if "-0-cut" in target.stem
                              else ["bright", "um", "garden"])
        return transcript(["bright", "garden"] if merged_passes else ["bright"])

    class RecordingBackend(ScriptedBackend):
        """Record the one additional cut/control batch for the merged span."""

        def transcribe_many(self, paths: Sequence[Path], *, language: str | None = None,
                            ) -> dict[Path, Transcript]:
            batches.append(list(paths))
            return super().transcribe_many(paths, language=language)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: RecordingBackend(script))
    result = stage.fillers(root, apply=True)
    assert [len(batch) for batch in batches] == [4, 2]
    if merged_passes:
        assert len(result["entries"]) == 1
        entry = result["entries"][0]
        assert entry["status"] == "applied" and entry["listen"]
        assert (entry["start"], entry["end"], entry["kind"]) == (0.9, 1.4, "filler")
        assert entry["words"] == ["t1:1", "t1:2"]
        assert entry["evidence"]["reasons"] == ["rule 1", "rule 2"]
        assert entry["evidence"]["absorbed_s"] == [[1.1, 1.2]]
        inspected = stage.fillers(root, apply=True, force=True)["entries"]
        assert len(inspected) == 1 and inspected[0]["id"] == entry["id"]
        assert inspected[0]["status"] == "applied"
        assert inspected[0]["evidence"]["merged_identities"] == entry["evidence"][
            "merged_identities"
        ]
        assert inspected[0]["evidence"]["absorbed_s"] == entry["evidence"]["absorbed_s"]
        stage.set_removal(root, entry["id"], "rejected", reason="keep both")
        rerun = stage.fillers(root, apply=True)
        assert len(rerun["entries"]) == 1
        assert rerun["entries"][0]["status"] == "rejected"
    else:
        earlier, later = result["entries"]
        assert earlier["status"] == "applied" and earlier["end"] == 1.1
        assert later["status"] == "rejected"
        assert later["reason"] == "leaves a sliver under 0.25 s"
        assert not later["evidence"]["sliver_merge"]["pass"]
        assert "garden" in str(later["evidence"]["sliver_merge"]["diff"])


def test_failing_merged_span_splits_into_rule_components(tmp_path: Path) -> None:
    words = (SourceWord("t1:0", "bright", 0.2, 0.5), SourceWord("t1:1", "um", 0.6, 0.8),
             SourceWord("t1:2", "uh", 0.9, 1.1), SourceWord("t1:3", "garden", 1.2, 1.5))
    segment = stage._PlannedAudio("t1", 0, 2, np.zeros(96_000), words)
    merged = replace(stage.detect(words).candidates[0], segment="s1")
    parts = stage._components(merged, segment, {"max_pause_s": 0.3, "beat_breath_s": 0.4})
    assert len(parts) == 2
    engine = ScriptedBackend(lambda target: transcript(
        ["bright", "um", "uh", "garden"] if target.stem.endswith("control")
        else ["bright", "garden"] if "round-0" in target.stem else ["bright", "garden"],
    ))
    results = stage._test(parts, {"s1": segment}, engine, tmp_path, "en")
    assert len(results) == 2 and all(item.passed for item in results)


@pytest.mark.parametrize("silence_db", [-40.0, -55.0])
def test_components_use_word_times_independent_of_segment_threshold(silence_db: float) -> None:
    words = (SourceWord("t1:0", "bright", 2.2, 2.5), SourceWord("t1:1", "um", 2.6, 2.8),
             SourceWord("t1:2", "uh", 2.9, 3.1), SourceWord("t1:3", "garden", 3.2, 3.5))
    segment = stage._PlannedAudio(
        "t1", 2, 4, np.full(2 * stage._RATE, 10 ** (-50 / 20)), words, silence_db=silence_db,
    )
    merged = replace(stage.detect(words).candidates[0], segment="s1")
    parts = stage._components(merged, segment, {"max_pause_s": 0.3, "beat_breath_s": 0.4})
    assert len(parts) == 2
    assert all(item.plan["silence_db"] == -45 and not item.listen for item in parts)


@pytest.mark.parametrize("silence_db", [-40.0, -55.0])
def test_components_restore_pause_trims_independent_of_segment_threshold(
    silence_db: float,
) -> None:
    words = (SourceWord("t1:0", "bright", 2.2, 2.5), SourceWord("t1:1", "garden", 3.2, 3.5),
             SourceWord("t1:2", "flowers", 4.2, 4.5))
    segment = stage._PlannedAudio(
        "t1", 2, 5, np.zeros(3 * stage._RATE), words, silence_db=silence_db,
    )
    merged = stage.Candidate("t1", "silence", (), 2.6, 4.1, (), "s1", False,
                             {"components": [{}, {}]})
    parts = stage._components(merged, segment, {"max_pause_s": 0.3, "beat_breath_s": 0.4})
    assert len(parts) == 2
    assert all(item.kind == "silence" and item.plan["silence_db"] == -45
               and not item.listen for item in parts)


def test_components_do_not_restore_pause_across_voiced_island() -> None:
    words = (SourceWord("t1:0", "bright", 2.2, 2.5),
             SourceWord("t1:1", "garden", 3.2, 3.5),
             SourceWord("t1:2", "flowers", 4.2, 4.5))
    rate = stage._RATE
    sound = np.zeros(3 * rate, dtype=np.float32)
    low, high = round(0.8 * rate), round(1.0 * rate)
    sound[low:high] = 10 ** (-48 / 20) * np.sqrt(2) * np.sin(
        2 * np.pi * 200 * np.arange(high - low) / rate
    )
    segment = stage._PlannedAudio("t1", 2, 5, sound, words, silence_db=-35,
                                  pause={"floor_db": -65.0})
    merged = stage.Candidate("t1", "silence", (), 2.6, 4.1, (), "s1", False,
                             {"components": [{}, {}]})
    parts = stage._components(merged, segment,
                              {"max_pause_s": 0.3, "beat_breath_s": 0.4})
    assert len(parts) >= 2
    assert all(item.kind == "silence" and item.word_ids == () for item in parts)
    assert all(item.end <= 2.8 or item.start >= 3.0 for item in parts)


def test_failed_merged_join_tests_components_and_keeps_safe_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [("bright", 0.2, 0.5), ("basically", 0.6, 0.8), ("um", 0.9, 1.1),
            ("garden", 1.2, 1.5)]
    root = make_project(tmp_path / "project", rows)
    suppress_tail_pause(root)

    def script(target: Path) -> Transcript:
        """A long removal loses garden; separate rule components preserve it."""
        if target.stem.endswith("control"):
            return transcript(["bright", "basically", "um", "garden"])
        with wave.open(str(target)) as stream:
            duration = stream.getnframes() / stream.getframerate()
        if duration < 3.5:
            return transcript(["bright"])
        tokens = ["bright", "um", "garden"] if "-0-cut" in target.stem else [
            "bright", "basically", "garden",
        ]
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    result = stage.fillers(root, apply=True)
    assert len(result["entries"]) == 2
    assert [item["status"] for item in result["entries"]] == ["applied", "rejected"], result
    assert all(len(item["evidence"]["merged_attempts"]) == 13 for item in result["entries"])


def test_failing_group_rejects_last_member_and_retries(tmp_path: Path) -> None:
    words = tuple(SourceWord(f"t1:{i}", token, 0.2 + i * 0.2, 0.35 + i * 0.2)
                  for i, token in enumerate(["bright", "um", "garden", "uh", "flowers"]))
    segment = stage._PlannedAudio("t1", 0, 2, np.zeros(96_000), words)
    candidates = [stage.Candidate("t1", "filler", (f"t1:{index}",), words[index].start,
                                  words[index].end, (), "s1", False, {}) for index in (1, 3)]
    tested = [stage._Tested(item, True, False, {"attempts": [], "words": "pass"})
              for item in candidates]
    engine = ScriptedBackend(lambda target: transcript(
        ["bright", "um", "garden", "uh", "flowers"] if target.stem.endswith("control")
        else ["bright", "flowers"],
    ))
    result = stage._groups(tested, {"s1": segment}, engine, tmp_path, "en")
    assert result[0].passed and not result[1].passed
    assert "garden" in str(result[1].evidence["diff"])


def test_close_cut_group_checks_every_repeated_occurrence(tmp_path: Path) -> None:
    words = tuple(SourceWord(f"t1:{i}", token, 0.2 + i * 0.2, 0.35 + i * 0.2)
                  for i, token in enumerate(["bright", "um", "garden", "uh", "flowers"]))
    segments = {identifier: stage._PlannedAudio("t1", 0, 2, np.zeros(96_000), words)
                for identifier in ("s1", "s2")}
    candidates = [stage.Candidate("t1", "filler", (f"t1:{index}",), words[index].start,
                                  words[index].end, (), "s1", False, {}) for index in (1, 3)]
    tested = [stage._Tested(item, True, False, {"attempts": []}) for item in candidates]
    engine = ScriptedBackend(lambda target: transcript(
        ["bright", "um", "garden", "uh", "flowers"] if target.stem.endswith("control")
        else ["bright", "garden", "flowers"] if "-0-cut" in target.stem
        else ["bright", "flowers"],
    ))
    result = stage._groups(tested, segments, engine, tmp_path, "en")
    assert result[0].passed and not result[1].passed
    group = result[1].evidence["groups"][0]
    assert {item["segment"] for item in group["segments"]} == {"s1", "s2"}
    assert "garden" in str(group["diff"])


def test_sliver_merge_checks_every_repeated_occurrence(tmp_path: Path) -> None:
    words = (SourceWord("t1:0", "bright", 0.2, 0.5), SourceWord("t1:1", "um", 0.6, 0.8),
             SourceWord("t1:2", "uh", 0.9, 1.1), SourceWord("t1:3", "garden", 1.2, 1.5))
    segments = {identifier: stage._PlannedAudio("t1", 0, 2, np.zeros(96_000), words)
                for identifier in ("s1", "s2")}
    candidates = [stage.Candidate("t1", "filler", (f"t1:{index}",), words[index].start,
                                  words[index].end, (), "s1", False, {}) for index in (1, 2)]
    tested = [stage._Tested(item, True, False, {
        "attempts": [], "identity": stage.candidate_id(item),
    }) for item in candidates]
    engine = ScriptedBackend(lambda target: transcript(
        ["bright", "um", "uh", "garden"] if target.stem.endswith("control")
        else ["bright", "garden"] if "-0-cut" in target.stem else ["bright"],
    ))
    result = stage._slivers(tested, segments, engine, tmp_path, "en")
    assert result[0].passed and not result[1].passed
    assert result[1].evidence["diff"] == ["leaves a sliver under 0.25 s"]
    assert {item["segment"] for item in result[1].evidence["sliver_merge"]["segments"]} == {
        "s1", "s2",
    }


def test_edge_extension_checks_other_story_occurrences(tmp_path: Path) -> None:
    words = (SourceWord("t1:0", "bright", 0.3, 0.6), SourceWord("t1:1", "like", 1, 1.2),
             SourceWord("t1:2", "garden", 1.4, 1.7))
    segments = {
        "short": stage._PlannedAudio("t1", 0, 38 / 30, np.zeros(60_800), words[:2]),
        "long": stage._PlannedAudio("t1", 0, 2, np.zeros(96_000), words),
    }
    candidate = stage.Candidate("t1", "filler", ("t1:1",), 1, 1.2, (), "short", False, {})
    tested = stage._Tested(candidate, True, False, {
        "attempts": [], "identity": stage.candidate_id(candidate),
    })
    engine = ScriptedBackend(lambda target: transcript(
        ["bright", "like", "garden"] if "-1-control" in target.stem else ["bright"],
    ))
    result = stage._slivers([tested], segments, engine, tmp_path, "en")[0]
    assert not result.passed
    assert result.candidate.end == 38 / 30
    assert {item["segment"] for item in result.evidence["segments"]} == {"short", "long"}
    assert "garden" in str(result.evidence["diff"])


def test_failing_group_of_three_retries_remaining_pair(tmp_path: Path) -> None:
    tokens = ["bright", "um", "garden", "uh", "flowers", "erm", "today"]
    words = tuple(SourceWord(f"t1:{index}", token, 0.2 + index * 0.2, 0.35 + index * 0.2)
                  for index, token in enumerate(tokens))
    segment = stage._PlannedAudio("t1", 0, 2, np.zeros(96_000), words)
    candidates = [stage.Candidate("t1", "filler", (f"t1:{index}",), words[index].start,
                                  words[index].end, (), "s1", False, {}) for index in (1, 3, 5)]
    tested = [stage._Tested(item, True, False, {"attempts": []}) for item in candidates]

    def script(target: Path) -> Transcript:
        """Only the pair remaining after the last rejection preserves flowers."""
        if target.stem.endswith("control"):
            return transcript(tokens)
        return transcript(["bright", "garden", "today"] if "round-100" in target.stem
                          else ["bright", "garden", "flowers", "erm", "today"])

    result = stage._groups(tested, {"s1": segment}, ScriptedBackend(script), tmp_path, "en")
    assert [item.passed for item in result] == [True, True, False]
    assert [len(item.evidence["groups"]) for item in result] == [2, 2, 1]


def test_saved_manual_trim_is_present_in_both_planned_controls(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch,
                                                              ) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    stage.fillers(root, apply=True)
    stage.add_manual(root, "t1", 2.2, 2.5, reason="chosen ending")
    context = ProjectContext(root)
    manual = next(item for item in stage._removals(context).removals if item.by == "manual")
    segments = stage._planned(context, [manual])
    candidate = stage._proposals(context, segments)[0]
    trial = stage._trial(segments[candidate.segment], (candidate,))
    assert trial is not None
    assert "grow" not in [word.text for word in trial.stored]
    assert len(trial.control) == round((trial.end - trial.start - 0.3 + 2) * 48_000)


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("saved_status", ["applied", "proposed"])
def test_manual_is_retested_only_on_force_and_keeps_manual_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apply: bool, saved_status: str,
) -> None:
    root = make_project(tmp_path / "project")
    suppress_tail_pause(root)
    install_engine(monkeypatch)
    stage.fillers(root, apply=True)
    added = stage.add_manual(root, "t1", 2.7, 2.8, reason="chosen trim")
    if saved_status != "applied":
        stage.set_removal(root, added["id"], saved_status, reason="chosen trim")
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: [])
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(
        lambda _: transcript(["flowers", "grow"]),
    ))
    result = stage.fillers(root, apply=apply, force=True)
    manual = next(item for item in result["entries"] if item["by"] == "manual")
    assert manual["kind"] == "manual" and manual["status"] == saved_status
    assert manual["reason"] == "chosen trim"
    assert len(manual["evidence"]["attempts"]) == 1


def test_failed_manual_inspection_keeps_saved_applied_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project", [("bright", 0.3, 0.6), ("morning", 0.65, 0.9)])
    added = stage.add_manual(root, "t1", 0.65, 0.9, reason="chosen word")
    engine = ScriptedBackend(lambda target: transcript(
        ["bright", "morning"] if target.stem.endswith("control") else ["wrong"],
    ))
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: engine)
    entry = stage.fillers(root, force=True)["entries"][0]
    assert entry["id"] == added["id"] and entry["status"] == "applied"
    assert entry["by"] == "manual" and entry["listen"]
    assert not entry["evidence"]["pass"]
    assert (entry["start"], entry["end"]) == (0.65, 0.9)
    assert len(entry["evidence"]["attempts"]) == 1


def test_forced_automatic_tests_include_applied_manual_audio_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages.cut import resolve_cut

    root = make_project(tmp_path / "project")
    suppress_tail_pause(root)
    install_engine(monkeypatch)
    automatic_id = stage.fillers(root, apply=True)["entries"][0]["id"]
    manual_id = stage.add_manual(root, "t1", 2.2, 2.5, reason="chosen ending")["id"]

    def script(target: Path) -> Transcript:
        """Automatic trials include the saved trim; its own inspection restores the control."""
        manual = int(target.stem.split("-")[2]) == 1
        tokens = ["um", "garden", "flowers", "grow"] if manual else [
            "bright", "morning", "um", "garden", "flowers",
        ]
        if target.stem.endswith("cut"):
            tokens.remove("grow" if manual else "um")
        return transcript(tokens)

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(script))
    entries = stage.fillers(root, apply=True, force=True)["entries"]
    assert {entry["id"] for entry in entries} == {automatic_id, manual_id}
    assert all(entry["status"] == "applied" and entry["evidence"]["words"] == "pass"
               and len(entry["evidence"]["attempts"]) == 1 for entry in entries)
    automatic = next(entry for entry in entries if entry["id"] == automatic_id)
    assert "grow" not in automatic["evidence"]["control"]
    retained = {word_id for piece in resolve_cut(root).pieces for word_id in piece.words}
    assert "t1:2" not in retained and "t1:5" not in retained


def test_forced_overlapping_manual_keeps_applied_automatic_and_cut_union(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages.cut import resolve_cut

    rows = [("bright", 0.25, 0.55), ("morning", 0.6, 0.9), ("um", 1.3, 1.6),
            ("garden", 1.9, 2.2)]
    root = make_project(tmp_path / "project", rows)
    candidate = stage.Candidate("t1", "filler", ("t1:2",), 1.3, 1.8, (), "s1", False, {})
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: [candidate])
    engine = ScriptedBackend(lambda target: transcript(
        [row[0] for row in rows if target.stem.endswith("control") or row[0] != "um"],
    ))
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: engine)
    automatic = stage.fillers(root, apply=True)["entries"][0]
    assert automatic["status"] == "applied"
    manual_id = stage.add_manual(root, "t1", 1.0, 1.5, reason="chosen trim")["id"]
    before = json.loads((root / "removals.json").read_bytes())["removals"]

    def untested(_: Path) -> Transcript:
        """Neither overlapping decision has an independent restored control."""
        pytest.fail("overlapping saved decisions must stay untested")

    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(untested))
    result = stage.fillers(root, apply=True, force=True)
    assert {item["id"]: item for item in result["entries"]} == {
        item["id"]: item for item in before
    }
    assert {item["id"] for item in result["entries"]} == {automatic["id"], manual_id}
    assert all(item["status"] == "applied" for item in result["entries"])
    pieces = resolve_cut(root).pieces
    assert pieces and any(float(piece.src_t0) >= 1.8 for piece in pieces)
    assert all(float(piece.src_t1) <= 1.0 or float(piece.src_t0) >= 1.8 for piece in pieces)


def test_forced_saved_edge_removal_tests_retained_side_and_stays_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [("bright", 0.4, 0.6), ("time", 0.8, 1.05), ("like", 1.1, 41 / 30)]
    root = make_project(tmp_path / "project", rows)
    edit = json.loads((root / "edit.json").read_bytes())
    edit["segments"][0]["to"]["time"] = 41 / 30 + 0.000001
    write_json(root / "edit.json", edit)
    candidate = stage.Candidate("t1", "filler", ("t1:2",), 1.1, 1.3, (), "s1", False, {})
    monkeypatch.setattr(stage, "_proposals", lambda *_, **__: [candidate])
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(
        lambda target: transcript([row[0] for row in rows] if target.stem.endswith("control")
                                  else ["bright", "time"]),
    ))
    saved = stage.fillers(root, apply=True)["entries"][0]
    assert saved["status"] == "applied" and saved["end"] == 41 / 30
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: ScriptedBackend(
        lambda _: transcript(["bright", "time"]),
    ))
    inspected = stage.fillers(root, force=True)["entries"]
    assert len(inspected) == 1 and inspected[0]["id"] == saved["id"]
    assert inspected[0]["status"] == "applied"
    assert (inspected[0]["start"], inspected[0]["end"]) == (saved["start"], saved["end"])
    assert inspected[0]["evidence"]["words"] == "pass"
    assert inspected[0]["evidence"]["edge"] == "end"
    assert inspected[0]["evidence"]["clicks_db"] == []
    assert len(inspected[0]["evidence"]["attempts"]) == 1
    assert inspected[0]["evidence"]["absorbed_s"] == saved["evidence"]["absorbed_s"]


@pytest.mark.parametrize("apply", [False, True], ids=["inspect", "apply"])
@pytest.mark.parametrize("force", [False, True], ids=["plain", "forced"])
def test_rerun_applied_source_union_never_shrinks_without_join_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apply: bool, force: bool,
) -> None:
    from yap_studio.stages.cut import resolve_cut

    rows = [("bright", 0.25, 0.55), ("morning", 0.6, 0.9), ("um", 1.1, 1.3),
            ("garden", 1.9, 2.1)]
    source = make_project(tmp_path / "source", rows) / "takes/source.mov"
    random = np.random.default_rng(11)
    proposal_modes = ("same", "shifted", "missing")
    outcomes = ("pass", "reject", "repair")
    decision_spans = (None, (0.9, 1.4), (1.4, 1.9), (1.2, 1.4), (0.9, 1.9))
    # Each transition changes the stage inputs but leaves the speech engine fixed.
    # Vary ownership, overlap geometry, detector output and actual join outcomes.
    cases = [(None, mode, outcome) for mode in proposal_modes for outcome in outcomes]
    cases.extend((span, mode, outcome) for span in decision_spans[1:]
                 for mode, outcome in zip(proposal_modes, outcomes, strict=True))
    active_outcome = "pass"

    def script(target: Path) -> Transcript:
        """Expose real failures, including a repair that removes less source time."""
        tokens = [row[0] for row in rows]
        if target.stem.endswith("cut"):
            tokens.remove("um")
            repaired = target.stem.startswith("round-1-1-")
            if active_outcome == "reject" or active_outcome == "repair" and not repaired:
                tokens.remove("garden")
        return transcript(tokens)

    engine = ScriptedBackend(script)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: engine)
    for index, (span, mode, active_outcome) in enumerate(cases):
        root = make_project(tmp_path / f"project-{index}", rows, source=source)
        start = int(random.choice([30, 31, 33])) / 30
        end = int(random.choice([45, 48, 54])) / 30
        candidate = stage.Candidate("t1", "filler", ("t1:2",), start, end,
                                    (), "s1", False, {})
        old = {"id": "r1", "take": "t1", "start": start, "end": end,
               "kind": "filler", "status": "applied", "words": ["t1:2"], "by": "fillers",
               "listen": False, "evidence": {"words": "pass", "pass": True,
                                             "identity": stage.candidate_id(candidate)}}
        before = [old]
        if span is not None:
            creator = index % 2 == 0
            decision = {"id": "r2", "take": "t1", "start": span[0], "end": span[1],
                        "kind": "filler" if creator else "manual", "status": "applied",
                        "words": ["t1:2"] if span[0] <= 1.2 < span[1] else [],
                        "by": "fillers" if creator else "manual", "listen": False,
                        "evidence": {"decision": {"status": "applied"}} if creator else {}}
            before.append(decision)

        def seed(data: dict[str, Any], saved: list[dict[str, Any]] = before) -> str:
            """Install an owned saved decision snapshot for this generated transition."""
            data["removals"] = saved
            return "r1"

        stage._change(root, seed)
        proposed = ([] if mode == "missing" else
                    [replace(candidate, end=end - 1 / 30)] if mode == "shifted" else [candidate])
        monkeypatch.setattr(stage, "_proposals",
                            lambda *_, saved_proposals=proposed, **__: saved_proposals)
        project = json.loads((root / "project.json").read_bytes())
        project["rules"] = {"max_pause_s": 0.3 + index / 1000}
        write_json(root / "project.json", project)
        result = stage.fillers(root, apply=apply, force=force)
        assert not result["noop"]
        after = result["entries"]
        assert len({item["id"] for item in after}) == len(after)
        points = sorted({item[edge] for item in [*before, *after] for edge in ("start", "end")})
        restored = []
        for left, right in zip(points, points[1:], strict=False):
            midpoint = (left + right) / 2
            was_removed = [item for item in before if item["start"] <= midpoint < item["end"]]
            now_removed = any(item["status"] == "applied"
                              and item["start"] <= midpoint < item["end"] for item in after)
            if not was_removed or now_removed:
                continue
            restored.append((left, right))
            for saved in was_removed:
                rejected = next((item for item in after if item["id"] == saved["id"]), None)
                assert saved["by"] == "fillers" and "decision" not in saved["evidence"]
                assert rejected is not None and rejected["status"] == "rejected", (index, after)
                assert rejected["evidence"]["pass"] is False
                assert rejected["evidence"]["attempts"]
        if span is not None or active_outcome == "pass":
            assert not restored, (index, before, after)
        else:
            assert restored, (index, before, after)
        pieces = resolve_cut(root).pieces
        for entry in after:
            if entry["status"] == "applied":
                assert all(float(piece.src_t1) <= entry["start"] + 1e-9
                           or float(piece.src_t0) >= entry["end"] - 1e-9 for piece in pieces)


@pytest.mark.parametrize("kind", ["filler", "stutter", "restart"])
@pytest.mark.parametrize("configured_none", [False, True], ids=["unavailable", "configured-none"])
@pytest.mark.parametrize("force", [False, True], ids=["plain", "forced"])
def test_no_engine_keeps_applied_speech_untested_without_unmeasured_pauses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, configured_none: bool, force: bool,
) -> None:
    from yap_studio.stages.cut import resolve_cut

    rows = [*ROWS, ("later", 3.5, 3.8)]
    indexes = (2,)
    end = 1.5
    if kind == "stutter":
        rows[2] = ("garden", 1.1, 1.3)
    elif kind == "restart":
        rows = [*ROWS[:2], ("blue", 1.1, 1.3), ("garden", 1.35, 1.5),
                ("blue", 1.7, 1.9), ("garden", 1.95, 2.1), ("flowers", 2.2, 2.5),
                ("later", 3.5, 3.8)]
        indexes, end = (2, 3), 1.7
    root = make_project(tmp_path / "project", rows)
    candidate = stage.Candidate("t1", kind, tuple(f"t1:{index}" for index in indexes),
                                1.1, end, (), "s1", False, {})
    engine = ScriptedBackend(lambda target: transcript(
        [row[0] for row in rows[:-1]] if target.stem.endswith("control")
        else [row[0] for index, row in enumerate(rows[:-1]) if index not in indexes],
    ))
    with monkeypatch.context() as patch:
        patch.setattr(stage, "_proposals", lambda *_, **__: [candidate])
        patch.setattr(stage, "resolve_backend", lambda *_: engine)
        saved = stage.fillers(root, apply=True)["entries"][0]
    assert saved["status"] == "applied" and saved["kind"] == kind
    project = json.loads((root / "project.json").read_bytes())
    project.setdefault("rules", {})["max_pause_s"] = 2.0
    if configured_none:
        project["asr"] = {"engine": "none", "model": None}
    write_json(root / "project.json", project)
    # The source's measured gaps are shorter than the configured pause limit.
    context = ProjectContext(root)
    measured = stage._proposals(context, stage._planned(context))
    assert not [item for item in measured if item.kind == "silence"]
    if configured_none:
        monkeypatch.setattr(stage, "resolve_backend", lambda *_: pytest.fail("speech disabled"))
    else:
        monkeypatch.setattr(stage, "resolve_backend", lambda *_: None)
    result = stage.fillers(root, force=force)
    assert not result["noop"] and result["engine"] == "none"
    kept = next(item for item in result["entries"] if item["id"] == saved["id"])
    assert kept == saved
    assert result["kept_untested"] == [saved["id"]]
    assert "speech removals kept untested: no speech engine (1)" in result["notes"]
    fresh = [item for item in result["entries"] if item["id"] != saved["id"]]
    assert fresh == []
    report = json.loads((root / "fillers/report.json").read_bytes())
    assert report["kept_untested"] == [saved["id"]]
    retained = {word_id for piece in resolve_cut(root).pieces for word_id in piece.words}
    assert not retained.intersection(saved["words"])


def test_no_engine_excludes_measured_silence_overlapping_applied_speech(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [*ROWS, ("later", 3.5, 3.8)]
    root = make_project(tmp_path / "project", rows)
    speech = stage.Candidate("t1", "filler", ("t1:2",), 1.1, 1.5, (), "s1", False, {})
    engine = ScriptedBackend(lambda target: transcript(
        [row[0] for row in rows[:-1]] if target.stem.endswith("control")
        else [row[0] for row in rows[:-1] if row[0] != "um"],
    ))
    with monkeypatch.context() as patch:
        patch.setattr(stage, "_proposals", lambda *_, **__: [speech])
        patch.setattr(stage, "resolve_backend", lambda *_: engine)
        saved = stage.fillers(root, apply=True)["entries"][0]
    assert saved["status"] == "applied" and saved["kind"] == "filler"

    # The source is quiet between "um" and "garden", inside the saved removal.
    silence = stage.Candidate("t1", "silence", (), 1.35, 1.45, (), "s1", False, {})
    segment = stage._planned(ProjectContext(root))["s1"]
    assert any(start <= silence.start and silence.end <= end
               for start, end in stage._measured_silent_runs(segment)(1.3, 1.5))
    proposed = []

    def propose(*_args: Any, **_kwargs: Any) -> list[stage.Candidate]:
        proposed.append(silence)
        return [silence]

    monkeypatch.setattr(stage, "_proposals", propose)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: None)
    result = stage.fillers(root, force=True)

    assert proposed == [silence]
    assert result["kept_untested"] == [saved["id"]]
    assert result["entries"] == [saved]


def test_project_none_engine_never_resolves_speech(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path / "project", [*ROWS, ("later", 3.5, 3.8)])
    project = json.loads((root / "project.json").read_bytes())
    project["asr"] = {"engine": "none", "model": None}
    write_json(root / "project.json", project)
    monkeypatch.setattr(stage, "resolve_backend", lambda *_: pytest.fail("speech was disabled"))
    result = stage.fillers(root)
    assert result["engine"] == "none" and result["skipped"] == 1
    assert result["entries"][0]["status"] == "applied"


@pytest.mark.parametrize("selection", ["requested", "recorded"])
def test_planned_audio_honors_selected_stream_and_container_offset(
    tmp_path: Path, selection: str,
) -> None:
    root = make_project(tmp_path / "project")
    original = root / "takes/source.mov"
    staged = root / "takes/selected.mov"
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-i", str(original), "-itsoffset", "0.2", "-f", "lavfi",
        "-i", "sine=frequency=440:sample_rate=48000:duration=2.8", "-map", "0:v:0",
        "-map", "0:a:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", str(staged),
    ], timeout_s=60)
    staged.replace(original)
    if selection == "requested":
        project = json.loads((root / "project.json").read_bytes())
        project["takes"][0]["audio_stream"] = 1
        write_json(root / "project.json", project)
    else:
        write_json(root / "ingest/ingest.json", {"takes": [
            {"id": "t1", "requested_audio_stream": None, "audio_stream": 1},
        ]})
    context = ProjectContext(root)
    quiet = stage._decode_segment(context, "t1", 0, 0.1, 0)
    assert not quiet.any()
    sound = stage._decode_segment(context, "t1", 0.4, 0.9, 0)
    frequencies = np.fft.rfftfreq(len(sound), 1 / 48_000)
    peak = frequencies[np.argmax(np.abs(np.fft.rfft(sound)))]
    assert peak == pytest.approx(440, abs=2)


def test_applied_fillers_recut_and_verify_clean(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from yap_studio.stages.cut import cut

    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    assert stage.fillers(root, apply=True)["entries"][0]["status"] == "applied"
    result = cut(root)
    assert result["status"] == "verified", result
    report = json.loads((root / "cut/verify.json").read_bytes())
    assert report["summary"]["fail"] == 0, report["findings"]
    cut_words = json.loads((root / "cut/cut-words.json").read_bytes())
    assert "t1:2" not in {word["id"] for word in cut_words["words"]}


def test_hand_edited_removals_are_not_overwritten(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    stage.fillers(root)
    target = root / "removals.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(YapError, match="removals.json was changed outside yap-studio") as exc:
        stage.fillers(root, force=True)
    assert exc.value.code == "unsafe_path"


def test_pending_removals_journal_uses_existing_recovery_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    stage.fillers(root)
    target = root / "removals.json"
    old_sha = receipt.sha256_file(target)
    target.write_bytes(target.read_bytes() + b" ")
    pending = root / "fillers" / "receipt.pending.json"
    pending.write_text(json.dumps({
        "schema": "yap-studio/stage-pending@1", "stage": "fillers",
        "started": "interrupted-run", "inputs": {},
        "outputs": [{"path": "removals.json", "old_sha256": old_sha,
                     "new_sha256": receipt.sha256_file(target)}],
    }))

    with pytest.raises(InputError, match="receipt and journal conflict"):
        stage.fillers(root, force=True)


def test_missing_batch_result_refuses_publication(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path / "project")

    class MissingBackend(ScriptedBackend):
        """Deliberately omit one clip result from a speech round."""

        def transcribe_many(self, paths: Sequence[Path], *, language: str | None = None,
                            ) -> dict[Path, Transcript]:
            return {}

    monkeypatch.setattr(stage, "resolve_backend",
                        lambda *_: MissingBackend(lambda _: Transcript([])))
    with pytest.raises(InputError, match="missing cut or control"):
        stage.fillers(root)
    assert not (root / "removals.json").exists()


def test_fillers_input_projection_ignores_generated_evidence(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    stage.fillers(root)
    before = fillers_inputs(ProjectContext(root))
    data = json.loads((root / "removals.json").read_bytes())
    data["removals"][0]["evidence"]["click_db"] = -99
    write_json(root / "removals.json", data)
    assert fillers_inputs(ProjectContext(root)) == before


def test_interrupted_manual_write_recovers_on_fillers_rerun(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_project(tmp_path / "project")
    install_engine(monkeypatch)
    stage.fillers(root, apply=True)
    original_writer = stage._write_json

    def interrupted_writer(target: Path, data: dict[str, Any]) -> None:
        """Interrupt after the removal bytes change and before receipt publication."""
        if target.name == "receipt.json":
            raise RuntimeError("interrupted decision write")
        original_writer(target, data)

    monkeypatch.setattr(stage, "_write_json", interrupted_writer)
    with pytest.raises(RuntimeError, match="interrupted decision write"):
        stage.add_manual(root, "t1", 2.7, 2.8, reason="chosen trim")
    monkeypatch.setattr(stage, "_write_json", original_writer)
    rerun = stage.fillers(root, apply=True)
    assert any(item["by"] == "manual" and item["reason"] == "chosen trim"
               for item in rerun["entries"])
    assert not (root / "fillers/receipt.pending.json").exists()


@pytest.mark.asr
def test_asr_planted_um_join_passes(tmp_path: Path) -> None:
    source, spans = synth.speech_source(tmp_path / "speech.mov", [
        "Bright morning", "um", "Garden flowers grow today",
    ])
    root = make_project(tmp_path / "project", [(word.text, word.start, word.end)
                                              for word in spans], source)
    context = ProjectContext(root)
    segments = stage._planned(context)
    filler = next(item for item in stage._proposals(context, segments) if item.kind == "filler")
    backend = resolve_backend("parakeet-onnx")
    tested = stage._test([filler], segments, backend, tmp_path, "en")[0]
    assert tested.passed, tested.evidence
    assert tested.evidence["words"] == "pass", tested.evidence


@pytest.mark.asr
def test_asr_cut_through_word_fails_and_names_word(tmp_path: Path) -> None:
    source, spans = synth.speech_source(tmp_path / "speech.mov", [
        "Bright morning", "um", "Garden flowers grow today",
    ])
    root = make_project(tmp_path / "project", [(word.text, word.start, word.end)
                                              for word in spans], source)
    context = ProjectContext(root)
    segments = stage._planned(context)
    filler = next(item for item in stage._proposals(context, segments) if item.kind == "filler")
    garden = next(word for word in spans if word.text.lower() == "garden")
    # Remove almost the whole next word; two frame retries cannot repair it.
    planted = replace(filler, end=stage._snap_time(garden.end - 0.02))
    tested = stage._test([planted], segments, resolve_backend("parakeet-onnx"), tmp_path, "en")[0]
    assert not tested.passed, tested.evidence
    assert "garden" in str(tested.evidence["diff"]).lower(), tested.evidence


@pytest.mark.asr
def test_asr_repeated_phrase_restart_join_passes(tmp_path: Path) -> None:
    source, spans = synth.speech_source(tmp_path / "speech.mov", [
        "Bright morning", "and I'll", "and I'll water the garden today",
    ])
    root = make_project(tmp_path / "project", [(word.text, word.start, word.end)
                                              for word in spans], source)
    context = ProjectContext(root)
    segments = stage._planned(context)
    restart, = [item for item in stage._proposals(context, segments) if item.kind == "restart"]
    assert "repeated 2-gram" in restart.reasons
    assert restart.word_ids == ("t1:2", "t1:3")
    assert [word.text for word in segments["s1"].words if word.id in restart.word_ids] == [
        "and", "I'll",
    ]
    tested = stage._test([restart], segments, resolve_backend("parakeet-onnx"), tmp_path, "en")[0]
    assert tested.passed, tested.evidence
    assert tested.evidence["words"] == "pass", tested.evidence


@pytest.mark.asr
def test_asr_fragment_restart_join_passes(tmp_path: Path) -> None:
    source, spans = synth.speech_source(tmp_path / "speech.mov", [
        "Yes", "red car,", "red carpet today",
    ])
    root = make_project(tmp_path / "project", [(word.text, word.start, word.end)
                                              for word in spans], source)
    context = ProjectContext(root)
    segments = stage._planned(context)
    restart, = [item for item in stage._proposals(context, segments) if item.kind == "restart"]
    assert "fragment restart" in restart.reasons
    assert restart.word_ids == ("t1:1", "t1:2")
    assert [word.text for word in segments["s1"].words if word.id in restart.word_ids] == [
        "red", "car,",
    ]
    tested = stage._test([restart], segments, resolve_backend("parakeet-onnx"), tmp_path, "en")[0]
    assert tested.passed, tested.evidence
    assert tested.evidence["words"] == "pass", tested.evidence


@pytest.mark.asr
def test_asr_emphasis_repeat_is_kept(tmp_path: Path) -> None:
    source, spans = synth.speech_source(tmp_path / "speech.mov", [
        "It was very, very good today",
    ])
    root = make_project(tmp_path / "project", [(word.text, word.start, word.end)
                                              for word in spans], source)
    context = ProjectContext(root)
    segments = stage._planned(context)
    assert not [item for item in stage._proposals(context, segments)
                if item.kind in {"restart", "stutter"}]
