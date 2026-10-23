"""Retained project facts for cards that draw the output timeline."""

from __future__ import annotations

import json
import os
from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from yap_studio import paths
from yap_studio.edl import yapjson
from yap_studio.errors import ValidationError
from yap_studio.plan import Plan, snap_frame

if TYPE_CHECKING:
    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

PROJECT_CARD_TEMPLATES = ("edit_timeline", "storyboard")


@dataclass(frozen=True)
class CardTrack:
    """Clip windows on one output track."""

    name: str
    clips: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CardShot:
    """One output shot with a project-relative source and exact source time."""

    start_frame: int
    end_frame: int
    label: str
    source: str | None = None
    source_sha256: str | None = None
    source_frame: int | None = None
    source_time: str | None = None


@dataclass(frozen=True)
class CardProject:
    """The output clock and picture facts retained for a card."""

    fps: int
    start_frame: int
    total_frames: int
    tracks: tuple[CardTrack, ...]
    shots: tuple[CardShot, ...] = ()


def project_facts(context: ProjectContext, timeline: ResolvedTimeline,
                  element: ResolvedElement) -> CardProject:
    """Plan span joins from snapshot bytes without reading source media."""
    source = context.project_dir / "cut/spans.json"
    # Use the span reader behind load_edit_list with the context's retained bytes.
    spans = yapjson.read(source, context.read("cut/spans.json"))
    plan = Plan.build(spans, fps=Fraction(30), sample_rate=48_000)
    if plan.total_frames != timeline.total_frames:
        raise ValidationError("cut/spans.json", "rerun cut so spans match the timeline frames")
    tracks = [CardTrack("Video", tuple((part.out_f0, part.out_f0 + part.n_frames)
                                      for part in plan.segments))]
    for name, kinds in (("B-roll", {"insert", "overlay", "background"}),
                        ("Cards", {"title", "graphic"})):
        clips = tuple((item.start_frame, item.end_frame) for item in timeline.elements
                      if item.kind in kinds)
        if clips or name == "Cards":
            tracks.append(CardTrack(name, clips))
    return CardProject(30, element.start_frame, timeline.total_frames, tuple(tracks),
                       _shots(context, timeline, plan))


def _shots(context: ProjectContext, timeline: ResolvedTimeline, plan: Plan) -> tuple[CardShot, ...]:
    words = []
    for word in context.json("cut/cut-words.json")["words"]:
        segment = word.get("segment")
        if not segment:
            raise ValidationError("cut/cut-words.json", "give each cut word a named segment")
        words.append((snap_frame(Fraction(str(word["start"])), plan.fps), segment))
    word_frames = [frame for frame, _ in words]
    clips = []
    for part in plan.segments:
        index = bisect_left(word_frames, part.out_f0)
        segment = (words[index][1] if index < len(words)
                   and word_frames[index] < part.out_f0 + part.n_frames else None)
        clips.append((part.out_f0, segment))
    groups = []
    # Leading wordless clips belong to the first named shot.
    previous = next((segment for _, segment in clips if segment is not None), None)
    for first, segment in clips:
        segment = segment if segment is not None else previous
        if not groups or segment != previous:
            groups.append(first)
        previous = segment
    if groups:
        groups[0] = 0
    shots = []
    for index, first in enumerate(groups):
        last = groups[index + 1] if index + 1 < len(groups) else timeline.total_frames
        if not 0 <= first < last <= timeline.total_frames:
            raise ValidationError("cut/cut-words.json", "separate shot starts on the output clock")
        frame = (first + last) // 2
        source = context.project_dir / "cut/cut.mp4"
        source_frame, source_time = frame, None
        pictures = [item for item in timeline.elements
                    if item.kind == "insert"
                    and item.start_frame <= frame < item.end_frame]
        if pictures:
            picture = pictures[0]
            media = picture.payload["media"]
            source = paths.validate_input(context.project_dir / media["path"]).resolve()
            source_frame = None
            if source.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".avif", ".tif", ".tiff",
            }:
                source_time = str(Fraction(str(media.get("in_s", 0)))
                                  + Fraction(frame - picture.start_frame, 30)
                                  * Fraction(str(media.get("speed", 1))))
        shots.append(CardShot(first, last, f"{index + 1:02}",
                              os.path.relpath(source, context.project_dir),
                              context.identity.sha256(source), source_frame, source_time))
    return tuple(shots)


def sample_project(template: str, frames: int, fps: int) -> CardProject:
    """Scale a template's synthetic output clock for standalone previews."""
    from yap_studio.cards import template_dir

    data = json.loads((template_dir(template) / "preview.json").read_text())["project"]
    scale = Fraction(fps, data["fps"])
    start = round(data["start_frame"] * scale)
    tracks = tuple(CardTrack(track["name"], tuple(
        (round(first * scale), start + frames if first == data["start_frame"]
         and track["name"] == "Cards" else round(last * scale))
        for first, last in track["clips"]
    )) for track in data["tracks"])
    shots = tuple(CardShot(round(shot["start_frame"] * scale),
                           round(shot["end_frame"] * scale), shot["label"])
                  for shot in data.get("shots", []))
    return CardProject(fps, start, max(round(data["total_frames"] * scale), start + frames),
                       tracks, shots)
