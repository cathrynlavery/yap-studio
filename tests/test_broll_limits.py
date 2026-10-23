"""B-roll pacing and named-word timing warnings on synthetic timelines."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
from test_planner_broll import catalog_item, requests
from test_timeline import add, insert
from test_timeline import project as project

from yap_studio.planner.broll import plan_broll
from yap_studio.project.model import CutWord, CutWords, Take
from yap_studio.stages.broll_limits import broll_limits
from yap_studio.stages.timeline import ResolvedElement


def shot(number: int, start: int, end: int, *, kind="insert", fit="cover", **payload):
    return ResolvedElement(f"broll-{number}", kind, "broll", start, end, "front",
                           {"media": {"fit": fit}, **payload})


def findings(elements, *, frames=1800, words=(), voiceover=False, existing=()):
    result = []

    def finding(rule, message, *items, **kwargs):
        assert kwargs["severity"] == "warning"
        result.append({"rule": rule, "message": message, "ids": [item.id for item in items],
                       **kwargs})

    broll_limits(elements, frames, CutWords("output", {}, tuple(words)),
                 [Take("t1", "take.mov", voiceover=voiceover)], finding,
                 insert_length_ids=set(existing))
    return {entry["rule"]: entry for entry in result}


@pytest.mark.parametrize("frames", [900, 1800, 3600])
def test_rate_threshold_densest_window_and_split(frames):
    inserts = [shot(i, i * 45, i * 45 + 40, fit="split") for i in range(4)]
    assert "broll_rate" not in findings(inserts[:3], frames=frames)
    result = findings(inserts, frames=frames)
    assert result["broll_rate"]["ids"] == [f"broll-{i}" for i in range(4)]
    assert "3" in result["broll_rate"]["suggested_action"]
    assert "broll_full_screen" not in result and "face_on_screen" not in result


def test_rate_half_open_minute_and_montage_groups():
    inserts = [shot(i, i * 600, i * 600 + 40) for i in range(4)]
    assert "broll_rate" not in findings(inserts, frames=3600)
    montage = [shot(i, i * 30, i * 30 + 30, group="montage") for i in range(4)]
    assert "broll_rate" not in findings(montage)
    assert "insert_short" not in findings(montage)
    later = [shot(i + 4, 1900 + i * 40, 1936 + i * 40) for i in range(5)]
    assert findings(inserts[:1] + later, frames=4000)["broll_rate"]["ids"] == [
        f"broll-{i + 4}" for i in range(5)
    ]


def test_rate_counts_repeated_montage_group_as_separate_runs():
    inserts = [shot(0, 0, 30, group="A"), shot(1, 2000, 2030, group="A"),
               *(shot(i + 2, start, start + 36) for i, start in enumerate((2050, 2100, 2150)))]
    assert findings(inserts, frames=3600)["broll_rate"]["ids"] == [
        "broll-1", "broll-2", "broll-3", "broll-4"]


@pytest.mark.parametrize("gap", [0, 1])
def test_rate_keeps_touching_and_overlapping_montage_parts_together(gap):
    inserts = [shot(0, 0, 300, group="A"), shot(1, 20, 30, group="A"),
               shot(2, 300 + gap, 330 + gap, group="A"),
               shot(3, 400, 436), shot(4, 500, 536)]
    assert "broll_rate" not in findings(inserts)


@pytest.mark.parametrize(("length", "warn"), [(449, False), (450, False), (451, True)])
def test_full_screen_threshold(length, warn):
    assert ("broll_full_screen" in findings([shot(0, 0, length)])) is warn


@pytest.mark.parametrize("payload", [{}, {"media": {}}])
def test_missing_media_or_fit_counts_as_full_screen(payload):
    item = replace(shot(0, 0, 900), payload=payload)
    result = findings([item])
    assert "broll_full_screen" in result and "face_on_screen" in result


@pytest.mark.parametrize(("length", "warn"), [(629, False), (630, False), (631, True)])
def test_face_on_screen_threshold_counts_transitions(length, warn):
    result = findings([shot(0, 0, 400), shot(1, 400, length, kind="transition")])
    assert ("face_on_screen" in result) is warn


def test_screen_coverage_is_union_and_ignores_split_overlay_background():
    inserts = [shot(0, 0, 300), shot(1, 100, 400), shot(2, 0, 1800, fit="split"),
               shot(3, 0, 1800, kind="overlay"), shot(4, 0, 1800, kind="background")]
    result = findings(inserts)
    assert "broll_full_screen" not in result and "face_on_screen" not in result


def test_voiceover_only_skips_screen_rules():
    inserts = [shot(i, i * 400, (i + 1) * 400) for i in range(4)]
    assert {"broll_rate", "broll_full_screen", "face_on_screen"} <= findings(inserts).keys()
    assert not ({"broll_rate", "broll_full_screen", "face_on_screen"}
                & findings(inserts, voiceover=True).keys())


@pytest.mark.parametrize(("frames", "warn"), [(35, True), (36, False), (37, False)])
def test_short_insert_threshold_and_existing_warning(frames, warn):
    assert ("insert_short" in findings([shot(0, 0, frames)])) is warn
    assert "insert_short" not in findings([shot(0, 0, frames)], existing=["broll-0"])


@pytest.mark.parametrize("kind", ["insert", "overlay"])
@pytest.mark.parametrize(("start", "warn"), [(1.1, False), (1.15, False), (1.16, True)])
def test_named_word_late_threshold(kind, start, warn):
    word = CutWord("t1:0", "s1", "Dogs", round(2 - start, 2), 1.3)
    result = findings([shot(0, 30, 60, kind=kind, matches="the dog")], words=[word])
    assert ("broll_late" in result) is warn
    if warn:
        assert "Dogs" in result["broll_late"]["message"]
        assert "0.16" in result["broll_late"]["message"]
        assert "-0.1" in result["broll_late"]["suggested_action"]


def test_late_uses_first_matching_word_in_lookback_only():
    words = [CutWord("t1:0", "s1", "dog", 0, 0.2),
             CutWord("t1:1", "s1", "park", 1.1, 1.4),
             CutWord("t1:2", "s1", "dogs", 2.2, 2.4)]
    assert "broll_late" not in findings([shot(0, 60, 66, matches="dog")], words=words)
    assert "broll_late" not in findings([shot(0, 30, 60)], words=words)
    assert "park" in findings([shot(0, 60, 90, matches="park dogs")], words=words)[
        "broll_late"]["message"]


def test_resolver_calls_limits_and_deduplicates_insert_length(project):
    short = insert(project, start={"time": 0}, end={"time": 0.9})
    result = add(project, short).require_valid()
    assert [entry.rule for entry in result.findings] == ["insert_length"]
    short["end"]["time"] = 1.1
    result = add(project, short).require_valid()
    assert [entry.rule for entry in result.findings] == ["insert_short"]


@pytest.mark.parametrize(("anchor", "matches", "start", "offset"), [
    ({"word": "t1:0", "text": "Dog"}, "dogs", 0.5, -0.1),
    ({"word": "t1:0", "text": "Dog", "edge": "start"}, "dogs", 0.5, -0.1),
    ({"word": "t1:0", "text": "Dog", "edge": "end"}, "dogs", 0.5, None),
    ({"word": "t1:0", "text": "Dog", "offset_s": 0}, "dogs", 0.5, 0),
    ({"word": "t1:0", "text": "Dog"}, "park", 0.5, None),
    ({"word": "t1:0", "text": "Dog"}, None, 0.5, None),
    ({"word": "t1:0", "text": "Dog"}, "dogs", 0.05, None),
    ({"time": 0.5}, "dogs", 0.5, None),
])
def test_planner_leads_only_named_word_start(tmp_path: Path, anchor, matches, start, offset):
    item = catalog_item(tmp_path)
    words = CutWords("output", {}, (CutWord("t1:0", "s1", "Dog", start, 1.0),
                                    CutWord("t1:1", "s1", "park.", 1.1, 2.0)))
    request = requests(start=anchor)
    if matches is not None:
        request["broll"][0]["matches"] = matches
    before = copy.deepcopy(request)
    plan = plan_broll(request, words, [item], {}, project_dir=tmp_path, frames=90)
    assert plan.broll[0]["element"]["start"].get("offset_s") == offset
    assert request == before
