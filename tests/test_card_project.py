"""Project facts use retained JSON and the output frame clock."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements
from test_stage_graphics import graphic, title
from test_timeline import write

from yap_studio.card_project import project_facts, sample_project
from yap_studio.errors import ValidationError
from yap_studio.project.status import ProjectContext
from yap_studio.stages.timeline import ResolvedElement


def spans(root: Path, *, end: float = 3) -> None:
    write(root / "cut/spans.json", {
        "source": "../source.mov",
        "segments": [{"runs": [[0, 0.5], [1, 2]]}, {"start": 0, "end": end - 1.5}],
    })


def test_project_facts_track_joins_and_snapshot_json(layer_project: Path) -> None:
    spans(layer_project)
    set_elements(layer_project, title(start={"time": 1.5}, hold_s=0.5), graphic())
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    element = next(item for item in timeline.elements if item.kind == "graphic")
    facts = project_facts(context, timeline, element)
    assert (facts.fps, facts.start_frame, facts.total_frames) == (30, 6, 90)
    assert [(shot.start_frame, shot.end_frame) for shot in facts.shots] == [(0, 90)]
    assert [(track.name, track.clips) for track in facts.tracks] == [
        ("Video", ((0, 15), (15, 45), (45, 90))),
        ("Cards", tuple((item.start_frame, item.end_frame) for item in timeline.elements
                        if item.kind in {"title", "graphic"})),
    ]
    spans(layer_project, end=2)
    assert project_facts(context, timeline, element) == facts
    with pytest.raises(FrozenInstanceError):
        facts.start_frame = 0
    with pytest.raises(ValidationError, match="rerun cut"):
        project_facts(ProjectContext(layer_project), timeline, element)


@pytest.mark.parametrize("kind", ["insert", "overlay", "background"])
def test_project_facts_optional_broll(layer_project: Path, kind: str) -> None:
    spans(layer_project)
    set_elements(layer_project, graphic())
    context = ProjectContext(layer_project)
    timeline = context.resolved_timeline()
    picture = ResolvedElement("agent-picture", kind, "agent", 15, 40, "front", {})
    timeline = replace(timeline, elements=(*timeline.elements, picture))
    facts = project_facts(context, timeline, timeline.elements[0])
    assert [track.name for track in facts.tracks] == ["Video", "B-roll", "Cards"]
    assert facts.tracks[1].clips == ((15, 40),)


def test_sample_project_uses_preview_clock() -> None:
    facts = sample_project("edit_timeline", 75, 30)
    assert (facts.fps, facts.start_frame, facts.total_frames) == (30, 720, 1800)
    assert [track.name for track in facts.tracks] == ["Video", "B-roll", "Cards"]
    assert (720, 795) in facts.tracks[-1].clips
    scaled = sample_project("edit_timeline", 60, 24)
    assert (scaled.fps, scaled.start_frame, scaled.total_frames) == (24, 576, 1440)
