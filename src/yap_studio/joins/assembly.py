"""Small assemble hooks for retained join inputs and verification windows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from yap_studio.checks.model import TransitionWindow
from yap_studio.joins import render
from yap_studio.joins.render import render_joins as render_joins
from yap_studio.project.receipts import projection_sha256

if TYPE_CHECKING:
    from yap_studio.project.status import ProjectContext


def snapshot_inputs(context: ProjectContext, inputs: dict[str, str], small: dict[str, bytes]
                    ) -> None:
    """Retain join contracts and picture identity only for smooth cuts."""
    moves = context.render_style().data["moves"]
    if (context.render_style().data.get("captions", {}).get("camera_3d", False)
            or moves.get("jump_cuts", "hard") != "smooth"):
        return
    for relative in ("cut/receipt.json", "cut/cut-words.json"):
        small[relative] = context.read(relative)
    inputs["picture:jump_cuts"] = projection_sha256(render.identity(moves))


def verification_windows(placements: list[dict]) -> list[TransitionWindow]:
    """Check frame counts and the cut frames on both sides of each smooth window."""
    return [TransitionWindow(item["join_frame"], item["start_frame"],
                             item["start_frame"] + item["frames"], item["id"])
            for item in placements]


def skip_grade(edl: dict, placements: list[dict]) -> None:
    """Exclude generated picture from comparisons to graded cut frames."""
    if "grade" in edl and placements:
        edl["grade"]["skip_frames"].extend(
            [item["start_frame"], item["start_frame"] + item["frames"]] for item in placements)
