"""Split insert geometry and picture branches with a fixed seam."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any

from yap_studio.edl import READERS
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import head_box
from yap_studio.faces.model import Box
from yap_studio.faces.placement import Band
from yap_studio.plan import Plan
from yap_studio.project.model import load_project

if TYPE_CHECKING:
    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.assemble_picture import _Graph, _Source
    from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

WIDTH = 1080
HEIGHT = 1920
# Moving the seam also moves the person's picture, so it cannot add headroom.
DEFAULT_SEAM_Y = 960
HEAD_GAP = 24


@dataclass(frozen=True)
class SplitLayout:
    """The seam and cut crop used by both rendering and face placement."""

    seam_y: int
    crop: tuple[int, int, int, int]
    scale: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return portable layout evidence."""
        return {"seam_y": self.seam_y, "crop": list(self.crop), "scale": self.scale,
                "warnings": list(self.warnings)}

    def map_box(self, box: Box) -> Box:
        """Map a cut-space rectangle into the creator's lower box."""
        x, y, _, _ = self.crop
        return Box((box.x - x) * self.scale, self.seam_y + (box.y - y) * self.scale,
                   box.w * self.scale, box.h * self.scale)


def calculate_layout(element_id: str, faces: Sequence[Box]) -> SplitLayout:
    """Keep every sampled face inside one crop with headroom at a fixed seam."""
    seam_y = DEFAULT_SEAM_Y
    boxes = [box for box in faces if box.w > 0 and box.h > 0]
    if not boxes:
        return SplitLayout(seam_y, (0, 0, WIDTH, HEIGHT - seam_y), 1.0,
                           (f"{element_id}: no measured face in the split window; "
                            "using the centred crop at the top of the cut.",))
    face = Box(*(median(getattr(box, coordinate) for box in boxes)
                 for coordinate in ("x", "y", "w", "h")))
    head_top = min(head_box(box).y for box in boxes)

    scale = max(1.0, min(1.25, 0.34 * (HEIGHT - seam_y) / face.h))
    w, h = (math.floor(size / scale / 2) * 2 for size in (WIDTH, HEIGHT - seam_y))
    y = max(0, min(HEIGHT - h, math.floor(face.y + face.h / 2 - 0.40 * h + 0.5)))
    if (head_top - y) * scale < HEAD_GAP:
        y = math.floor(head_top - HEAD_GAP / scale)
        if not 0 <= y <= HEIGHT - h or (head_top - y) * scale < HEAD_GAP:
            from yap_studio.stages.assemble_picture import PictureError

            raise PictureError(f"{element_id}: split cannot keep the head 24 px below the seam; "
                               "use fit cover for this shot, or start it where the person sits "
                               "lower in the frame")
    left, right = min(box.x for box in boxes), max(box.x + box.w for box in boxes)
    min_x, max_x = max(0, math.ceil(right - w)), min(WIDTH - w, math.floor(left))
    min_y = max(0, math.ceil(max(box.y + box.h for box in boxes) - h))
    max_y = min(HEIGHT - h, math.floor(head_top - HEAD_GAP / scale))
    if min_x > max_x or min_y > max_y:
        from yap_studio.stages.assemble_picture import PictureError

        raise PictureError(f"{element_id}: the person moves too much for one crop; "
                           "use fit cover for this shot or shorten it")
    x = max(min_x, min(max_x, math.floor((left + right - w) / 2 + 0.5)))
    y = max(min_y, min(max_y, y))
    return SplitLayout(seam_y, (x, y, w, h), scale)


def check_split_voiceover(
    context: ProjectContext | Path, elements: Sequence[ResolvedElement],
    finding: Callable[..., None] | None = None,
) -> None:
    """Refuse letters and voiceover splits, opening a root only when needed."""
    inserts = [item for item in elements
               if item.kind == "insert" and item.payload.get("media", {}).get("fit") == "split"]
    if not inserts:
        return
    for item in inserts:
        if "letters" not in item.payload:
            continue
        message = "Video in letters fills the frame, so it cannot be a split"
        action = "use fit cover for this shot"
        if finding is None:
            from yap_studio.stages.assemble_picture import PictureError

            raise PictureError(f"{item.id}: {message}; {action}")
        finding("letters_split", message, item, severity="violation", suggested_action=action)
    if isinstance(context, Path):
        if not any(take.voiceover for take in load_project(context / "project.json").takes):
            return
        from yap_studio.project.status import ProjectContext

        context = ProjectContext(context)
    voiceovers = {take.id for take in context.project.takes if take.voiceover}
    if not voiceovers:
        return
    source = context.project_dir / "cut/spans.json"
    data = context.read("cut/spans.json")
    reader = next(reader for reader in READERS if reader.sniff(source, data))
    plan = Plan.build(reader.read(source, data), fps=Fraction(30), sample_rate=48_000)
    for item in inserts:
        if not any(piece.source_id in voiceovers
                   and item.start_frame < piece.out_f0 + piece.n_frames
                   and piece.out_f0 < item.end_frame for piece in plan.segments):
            continue
        message = "split needs the take under it, and this part is voiceover"
        action = "set fit cover for this shot"
        if finding is None:
            from yap_studio.stages.assemble_picture import PictureError

            raise PictureError(f"{item.id}: {message}; {action}")
        finding("split_voiceover", message, item, severity="violation", suggested_action=action)


def split_layouts(root: Path, timeline: ResolvedTimeline) -> dict[str, SplitLayout]:
    """Load measured cut faces once, and only when an insert uses split."""
    inserts = [item for item in timeline.elements
               if item.kind == "insert" and item.payload.get("media", {}).get("fit") == "split"]
    if not inserts:
        return {}
    from yap_studio.faces.project import cut_faces

    faces = cut_faces(root, root / "cut/cut.mp4")
    layouts = {}
    for item in inserts:
        boxes = [max(frame, key=lambda face: face.box.w * face.box.h).box
                 for frame in faces.frames[item.start_frame:item.end_frame] if frame]
        layouts[item.id] = calculate_layout(item.id, boxes)
    return layouts


def split_windows(timeline: ResolvedTimeline, layouts: Mapping[str, SplitLayout]
                  ) -> tuple[tuple[int, int, SplitLayout], ...]:
    """Pair each split layout with its exclusive frame window."""
    return tuple((item.start_frame, item.end_frame, layouts[item.id])
                 for item in timeline.elements if item.id in layouts)


def split_frame(
    graph: _Graph, source: _Source, item: ResolvedElement, layout: SplitLayout, *,
    insert: Callable[..., str], cut: _Source, top_filter: str | None = None,
) -> str:
    """Stack a covered insert above a separate, cropped cut decode."""
    if "crop" in item.payload["media"]:
        from yap_studio.stages.assemble_picture import PictureError

        raise PictureError(f"{item.id}: split does not accept crop; use fit cover.")
    media = {**item.payload["media"], "fit": "cover"}
    top = insert(graph, source, replace(item, start_frame=0,
                                       end_frame=item.end_frame - item.start_frame,
                                       payload={**item.payload, "media": media}),
                 size=(WIDTH, layout.seam_y))
    if top_filter is not None:
        top = graph.filter(top, top_filter)
    x, y, w, h = layout.crop
    bottom = graph.filter(
        graph.input(cut, element=item),
        f"trim=start_frame={item.start_frame}:end_frame={item.end_frame},"
        f"setpts=PTS-STARTPTS,crop={w}:{h}:{x}:{y}:exact=1,"
        f"scale={WIDTH}:{HEIGHT - layout.seam_y}:flags=lanczos,setsar=1,format=yuv444p",
    )
    return graph.filter(
        f"{top}][{bottom}",
        f"vstack=inputs=2,drawbox=x=0:y={layout.seam_y - 1}:w={WIDTH}:h=2:"
        f"color=white@0.6:t=fill,setsar=1,settb=1/30,setpts=N+{item.start_frame}",
    )


def check_insert_bounds(item: ResolvedElement, size: tuple[int, int]) -> None:
    """Keep privacy boxes inside the fitted insert canvas."""
    for x, y, w, h in item.payload["media"].get("blur", []):
        if min(x, y) < 0 or min(w, h) < 8 or x + w > size[0] or y + h > size[1]:
            from yap_studio.stages.assemble_picture import PictureError

            raise PictureError(f"{item.id}: blur boxes must be at least 8 px and inside "
                               f"the {size[0]}x{size[1]} insert canvas.")


def caption_split_band(
    phrase: Mapping[str, Any], height: int, top: int,
    splits: Sequence[tuple[int, int, int]], safe_bottom: int, *, fps: int = 30,
) -> tuple[Band, int] | None:
    """Constrain a whole overlapping phrase below every split seam it crosses."""
    seams = [seam for start, end, seam in splits
             if phrase["start"] * fps < end and phrase["end"] * fps > start]
    if not seams:
        return None
    band = Band(max(seams) + HEAD_GAP, safe_bottom)
    if height > band.bottom - band.top:
        raise ValidationError("captions", "caption band does not fit below the split seam")
    return band, max(band.top, min(top, math.floor(band.bottom - height)))
