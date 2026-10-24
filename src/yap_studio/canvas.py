"""Project delivery geometry and the deliberately small horizontal feature set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from yap_studio.errors import ValidationError, YapError

if TYPE_CHECKING:
    from yap_studio.project.model import TimelineLane
    from yap_studio.project.status import ProjectContext


@dataclass(frozen=True)
class Canvas:
    """Delivery pixels, with a readable caption column inside side safe areas."""

    width: int
    height: int
    caption_inset: int

    @property
    def caption_width(self) -> int:
        return self.width - 2 * self.caption_inset

    @property
    def caption_top(self) -> int:
        return self.height - 216

    @property
    def caption_safe_bottom(self) -> int:
        return self.height - 72


def canvas(shape: str = "vertical") -> Canvas:
    """Resolve the only supported project shapes; vertical preserves the old canvas."""
    if shape == "vertical":
        return Canvas(1080, 1920, 70)
    if shape == "horizontal":
        return Canvas(1920, 1080, 160)
    raise ValidationError("shape", "choose vertical or horizontal")


def unsupported(feature: str, fix: str) -> None:
    """Give an actionable refusal before a horizontal render can use portrait geometry."""
    raise YapError(
        f"Horizontal phase 1 does not support {feature}.",
        error_type="validation", code="horizontal_unsupported",
        suggested_action=f"{fix}, or set project.json shape to vertical.",
    )


def require_supported(context: ProjectContext,
                      lanes: dict[str, TimelineLane] | None = None) -> None:
    """Check authored visual features even before a cut or timeline receipt exists."""
    if getattr(getattr(context, "project", None), "shape", "vertical") != "horizontal":
        return
    if context.project.video_type not in (None, "talking_head") or any(
        take.voiceover for take in context.project.takes
    ):
        unsupported("this video type", "Use talking_head with horizontal video takes")
    if context.project.platform not in ("none", "youtube"):
        unsupported(f"platform {context.project.platform}", "Set platform to youtube or none")
    style = context.render_style().data
    if style["captions"].get("camera_3d", False):
        unsupported("3D camera captions", "Set captions.camera_3d to false in the style")
    if style["moves"]["jump_cuts"] != "hard" or style["moves"]["opening_push"]:
        unsupported("zoom moves or smooth joins",
                    "Set moves.jump_cuts to hard and moves.opening_push to false in the style")
    if style["hook"]["style"] == "text_hook":
        unsupported("text hooks", "Choose a style without text_hook")
    grade = context.project.grade if context.project.grade is not None else style.get("grade", {})
    if grade.get("look", "none") != "none":
        unsupported("grading", "Set grade.look to none")
    if context.taste_application().applied:
        unsupported("applied taste preferences", "Disable the applied taste preferences")
    if lanes is None:
        from yap_studio.stages.timeline import _lanes

        lanes = _lanes(context)
    for lane in lanes.values():
        for item in lane.elements:
            if item.kind not in {"caption_gap", "sound"}:
                unsupported(f"{item.kind} element {item.id}",
                            f"Remove {item.id} from the {lane.lane} timeline")
