"""Per-take pause calibration and cut joins on one noisy synthetic speech set."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from itertools import groupby
from pathlib import Path

import numpy as np
import pytest
import synth
from test_fillers import make_project, write_json

from yap_studio.project.model import Word
from yap_studio.project.status import ProjectContext
from yap_studio.stages import edges
from yap_studio.stages import fillers as stage
from yap_studio.stages.cut import resolve_cut
from yap_studio.transcript import resolve_backend

QUIET_DBFS = -65
NOISY_DBFS = -38
SENTENCES = [
    "So the plan um is simple and clear",
    "We start uh with the garden today",
    "Then we um rest for a while",
    "That light is uh soft in the morning",
    "The small birds um sing near our home",
    "We walk uh down the road today",
    "Our dog um runs near the door",
    "We rest um here for a while",
]
FILLERS = {"um", "uh"}


@pytest.fixture(scope="module")
def synthetic_pair(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[int, tuple[Path, list[synth.WordSpan]]]:
    """Render the same speech and seeded room noise once at each floor."""
    directory = tmp_path_factory.mktemp("pause-threshold")
    pair = {}
    for floor in (QUIET_DBFS, NOISY_DBFS):
        source, spans = synth.speech_source(
            directory / f"speech-{floor}.mov", SENTENCES,
            size="160x90", noise_dbfs=floor, seed=0,
        )
        root = make_project(
            directory / f"project-{floor}",
            [(word.text, word.start, word.end) for word in spans], source,
        )
        pair[floor] = root, spans
    return pair


def test_noise_floors_choose_distinct_thresholds_and_keep_words(
    synthetic_pair: dict[int, tuple[Path, list[synth.WordSpan]]],
) -> None:
    assert synthetic_pair[QUIET_DBFS][1] == synthetic_pair[NOISY_DBFS][1]
    thresholds = []
    for floor, (root, spans) in synthetic_pair.items():
        assert len(spans) >= 14
        context = ProjectContext(root)
        take = edges.load_take(
            context.take_path("t1"), 0, words=[(word.start, word.end) for word in spans],
        )
        pause = take.pause
        assert pause is not None and pause.rule == "words"
        assert pause.words == len(spans)
        speech_db = float(np.median([
            take.level[round((word.start + word.end) / 2 / edges.HOP_S)] for word in spans
        ]))
        assert pause.floor_db < floor < take.threshold < speech_db
        assert all(not edges.in_silence(take, Fraction(str((word.start + word.end) / 2)))
                   for word in spans)
        assert edges._runs(take, 0, len(take.level))
        thresholds.append(take.threshold)
        if floor == NOISY_DBFS:
            fixed = replace(take, threshold=-45.0)
            assert not edges._runs(fixed, 0, len(fixed.level))

        segments = stage._planned(context)
        segment = segments["s1"]
        assert len(segment.words) == len(spans)
        assert segment.pause == pause.to_dict()
        assert segment.silence_db == pause.threshold_db
        filler_ids = {word.id for word in segment.words if word.text.lower() in FILLERS}
        proposals = stage._proposals(context, segments)
        assert {word_id for candidate in proposals for word_id in candidate.word_ids} == filler_ids
        assert any(candidate.kind == "silence" for candidate in proposals)
        for candidate in proposals:
            assert set(candidate.word_ids) <= filler_ids
            for word in segment.words:
                if candidate.kind == "silence" or word.id not in filler_ids:
                    assert not candidate.start <= (word.start + word.end) / 2 < candidate.end
    assert abs(thresholds[0] - thresholds[1]) >= 10


def _loose_rows(
    words: Sequence[Word], *, start: float, end: float,
) -> list[tuple[str, float, float]]:
    """Jitter boundaries by signed 50–120 ms offsets, then keep ordered 30 ms words."""
    rng = np.random.default_rng(36)
    rows = []
    previous_end = start
    for index, word in enumerate(words):
        offsets = rng.choice([-1.0, 1.0], size=2) * rng.uniform(0.05, 0.12, size=2)
        # Reserve enough room for this word and the words still to come inside the take.
        latest_end = end - (len(words) - index - 1) * 0.03
        first = min(max(previous_end, word.start + float(offsets[0])), latest_end - 0.03)
        last = min(max(first + 0.03, word.end + float(offsets[1])), latest_end)
        rows.append((word.text, first, last))
        previous_end = last
    return rows


@pytest.mark.asr
def test_cut_edges_pass_more_join_tests(
    tmp_path: Path, synthetic_pair: dict[int, tuple[Path, list[synth.WordSpan]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Undo conftest's silence-only cut engine so word anchors get real ASR confirmation.
    monkeypatch.undo()
    backend = resolve_backend("parakeet-onnx")
    real_threshold = edges.choose_pause_threshold
    variants = ("word_times", "fixed_rule", "per_take")
    counts = {"floors": {}, "total": {
        variant: {"planned": 0, "planned_unverifiable": 0, "searched": 0, "tested": 0}
        for variant in variants
    }}

    def word_time_edges(requests: Sequence[edges.EdgeRequest], *args, **kwargs
                        ) -> tuple[list[edges.EdgeResult], list[str]]:
        return [edges.EdgeResult(
            request.segment, request.start, request.nominal, request.nominal, "unconfirmed", True,
        ) for request in requests], []

    def join_counts(results: dict) -> str:
        return " ".join(
            f"{metric} " + " ".join(
                f"{variant}={results[variant][metric]}/{results[variant]['tested']}"
                for variant in variants
            ) for metric in ("planned", "planned_unverifiable", "searched")
        )

    for floor, (original, _) in synthetic_pair.items():
        original_context = ProjectContext(original)
        original_segment = stage._planned(original_context)["s1"]
        rows = _loose_rows(
            original_segment.words, start=original_segment.start, end=original_segment.end,
        )
        assert all(0 <= first and last - first >= 0.03 - 1e-9 for _, first, last in rows)
        assert all(left[2] <= right[1] for left, right in zip(rows, rows[1:], strict=False))
        assert original_segment.start <= rows[0][1] < rows[-1][2] <= original_segment.end
        # The join test judges each variant against the synthetic word truth.
        join_project = original
        cut_project = make_project(
            tmp_path / f"cut-project-{floor}", rows, original_context.take_path("t1"),
        )
        segments = stage._planned(ProjectContext(join_project))
        segment = segments["s1"]
        kept_runs = [list(run) for filler, run in groupby(
            segment.words, key=lambda word: word.text.lower() in FILLERS,
        ) if not filler]
        edit = json.loads((cut_project / "edit.json").read_bytes())
        edit["segments"] = [
            {"id": f"c{index}", "take": "t1", "from": {"word": run[0].id},
             "to": {"word": run[-1].id}}
            for index, run in enumerate(kept_runs, 1)
        ]
        write_json(cut_project / "edit.json", edit)

        floor_counts = {}
        for variant in variants:
            with monkeypatch.context() as patch:
                if variant == "word_times":
                    patch.setattr(edges, "place", word_time_edges)
                elif variant == "fixed_rule":
                    patch.setattr(edges, "choose_pause_threshold",
                                  lambda level, midpoints: real_threshold(level, ()))
                plan = resolve_cut(cut_project)
            candidates = []
            for left, right in zip(plan.pieces, plan.pieces[1:], strict=False):
                if left.take != right.take:
                    continue
                first, last = float(left.src_t1), float(right.src_t0)
                if last <= first:
                    continue
                candidates.append(stage.Candidate(
                    take="t1", kind="filler",
                    word_ids=tuple(word.id for word in segment.words
                                   if first <= (word.start + word.end) / 2 < last),
                    start=first, end=last, reasons=("cut join",), segment="s1", listen=False,
                    plan={"word_indexes": (), "joined_pause_s": 0.0, "want_pause_s": 0.3,
                          "left_word_end_s": None, "right_word_start_s": None},
                ))

            directory = tmp_path / f"joins-{floor}-{variant}"
            directory.mkdir()
            tested = stage._test(candidates, segments, backend, directory, "en")
            assert len(tested) == len(candidates)
            planned = 0
            planned_unverifiable = 0
            for item in tested:
                attempts = [attempt for attempt in item.evidence["attempts"]
                            if attempt["nudge"] == [0, 0]]
                assert len(attempts) == 1, item.evidence
                planned += bool(attempts[0]["pass"])
                planned_unverifiable += attempts[0]["words"] == "unverifiable"
            floor_counts[variant] = {
                "planned": planned, "planned_unverifiable": planned_unverifiable,
                "searched": sum(item.passed for item in tested),
                "tested": len(tested),
                "edge_how": dict(Counter(edge["how"] for edge in plan.edges)),
                "edges": list(plan.edges), "pause_thresholds": plan.pause_thresholds,
            }
            for metric in ("planned", "planned_unverifiable", "searched", "tested"):
                counts["total"][variant][metric] += floor_counts[variant][metric]
        counts["floors"][str(floor)] = floor_counts
        fixed = floor_counts["fixed_rule"]["pause_thresholds"]["t1"]["threshold_db"]
        per_take = floor_counts["per_take"]["pause_thresholds"]["t1"]["threshold_db"]
        print(f"JOIN-COUNTS floor={floor} {join_counts(floor_counts)} "
              f"thresholds fixed={fixed:.1f} per_take={per_take:.1f}", flush=True)

    print(f"JOIN-COUNTS total {join_counts(counts['total'])}", flush=True)
    output = tmp_path / "join_counts.json"
    write_json(output, counts)
    print(f"JOIN-COUNTS path={output}", flush=True)
    assert (
        counts["total"]["per_take"]["searched"] >= counts["total"]["word_times"]["searched"]
    ), counts
