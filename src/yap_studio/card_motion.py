"""Plan horizontal card moves and prepare their offline band hosts."""

from __future__ import annotations

import html
import json
import math
import re
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from yap_studio import paths
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import face_clearance_px, gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.render import hyperframes
from yap_studio.style import ResolvedStyle

if TYPE_CHECKING:
    from yap_studio.cards import CardPlan

SUPPORTED = frozenset((
    "quote", "chart", "callout_circle", "checklist", "comment_cta", "comment_drop",
    "icon_strike", "image_card", "image_compare", "name_tag", "progress_stat", "prompt_bar",
    "steps", "success_check", "terminal", "transfer", "flow", "stat", "specs_list", "text_tag",
))
IN_S = 0.4
OUT_S = 0.35
_SCRIPTS = Path(__file__).with_name("templates")


def plan_motion(
    plan: CardPlan, style: ResolvedStyle, faces: Sequence[Sequence[Box]] | None,
    evidence: dict,
) -> tuple[CardPlan, dict]:
    """Keep the rest box and prove each move's swept corridor against measured heads."""
    if style.data["graphics"].get("card_motion", "none") == "none":
        return replace(plan, motion=None), evidence
    registry = plan.template.startswith("hf:")
    reason = None
    if plan.template == "big_number":
        reason = "number cards use the behind-the-head path"
    elif plan.template in {"playing_grid", "reel_feed"}:
        reason = "full-frame cards cannot fly as one panel"
    elif plan.template == "video_card":
        reason = "video windows are composited separately and cannot follow the fly"
    elif not registry and plan.template not in SUPPORTED:
        reason = "this template has no GSAP panel timeline"
    elif plan.y + plan.variables.get("glow_pad_px", 0) < 280:
        reason = "the card row starts above the 280 px line"
    elif plan.frames / plan.fps < IN_S + OUT_S:
        reason = "the card window is too short for the fly and a hold"
    elif evidence.get("status") in {"blocked", "behind_head"}:
        reason = "the card has no clear front-layer path"
    if reason:
        return replace(plan, motion=None), {**evidence, "card_motion": {
            "status": "none", "reason": reason}}
    boxes = [face for frame in faces or () for face in frame]
    side = "right"
    if boxes:
        centre = sum(face.x + face.w / 2 for face in boxes) / len(boxes)
        side = "right" if centre < 540 else "left"
        heads = [head_box(face) for face in boxes]
        above = all(plan.y + plan.height + face_clearance_px(style.data) <= h.y for h in heads)
        if above and abs(plan.x + plan.width / 2 - centre) <= max(40, plan.width / 4):
            side = "left" if plan.x <= 1080 - plan.x - plan.width else "right"
    move_frames = set(range(max(0, math.floor(plan.frames - OUT_S * plan.fps)), plan.frames))
    if not registry:
        move_frames.update(range(min(plan.frames, math.ceil(IN_S * plan.fps) + 1)))
    margin = face_clearance_px(style.data)
    for chosen in (side, "left" if side == "right" else "right"):
        sweep = (Box(0, plan.y, plan.x + plan.width, plan.height) if chosen == "left"
                 else Box(plan.x, plan.y, 1080 - plan.x, plan.height))
        gap = min((gap_px(sweep, head_box(face)) for index in move_frames
                   for face in (faces[index] if faces is not None and index < len(faces) else ())),
                  default=None)
        if gap is None or gap >= margin:
            motion = {"side": chosen, "enter": not registry}
            return replace(plan, motion=motion), {**evidence, "card_motion": {
                "status": "streak", **motion, "gap_px": gap,
                "rest_box": {"x": plan.x, "y": plan.y,
                             "width": plan.width, "height": plan.height},
                "reason": "horizontal swept box clears the head during both moves"}}
    return replace(plan, motion=None), {**evidence, "card_motion": {
        "status": "none", "reason": "neither swept box keeps the required head clearance"}}


def preview_motion(plan: CardPlan, style: ResolvedStyle) -> CardPlan:
    """A preview without faces uses the right edge."""
    return plan_motion(plan, style, None, {})[0]


def drawn_template(source: str, template: str) -> str:
    """Finish entrance reveals before the fly, preserving later card animation."""
    code = (_SCRIPTS / "card-motion-drawn.js").read_text(encoding="utf-8")

    def tag(pattern: str, replacement: str) -> str:
        matches = source.count(pattern)
        if matches == 0:
            raise ValidationError(f"template.{template}",
                                  f"missing motion cue pattern: {pattern!r}")
        return source.replace(pattern, replacement, matches)

    # Tag authored actions only in motion pages so ordinary pages and cache keys stay unchanged.
    targets = {
        "checklist": ("row", "tick", "path", "fill"),
        "comment_cta": ("send",),
        "comment_drop": ("bubble",),
        "flow": ("tag", "arrow", "path", "label"),
        "icon_strike": ("slash", "content"),
        "image_card": ("el", "arc"),
        "image_compare": ("divider", "afterChip"),
        "prompt_bar": ("arrow", "check", "send"),
        "specs_list": ("clock",),
        "stat": ("row", "clock", "strip"),
        "terminal": ("line",),
        "transfer": ("traveler", "travel", "outline", "tint", "tick"),
    }
    for target in targets.get(template, ()):
        source = tag(f"tl.fromTo({target},", f"cardMotionCue(tl).fromTo({target},")
    if template in {"comment_cta", "flow", "prompt_bar", "stat", "steps", "terminal"}:
        source = tag("tl.set(", "cardMotionCue(tl).set(")
    if template == "specs_list":
        source = tag("tl.fromTo(row,", "cardMotionCue(tl, onCue).fromTo(row,")
    for prefix in {
        "stat": ('stage, { scale: 1 }', 'stage, { scale: 1.07 }'),
        "image_card": ('panel, { y: 0 }', 'panel, { y: -2 }'),
        "image_compare": ('panel, { "--card-split-position": 0 }',),
    }.get(template, ()):
        source = tag(f"tl.fromTo({prefix}", f"cardMotionCue(tl).fromTo({prefix}")
    source = source.replace("Number(root.dataset.duration)",
                            "Number(root.dataset.cardDuration || root.dataset.duration)")
    source = source.replace("root.insertBefore(frame, panel)",
                            "panel.parentNode.insertBefore(frame, panel)")
    source = source.replace('"use strict";', f'"use strict";\n{code}', 1)
    return source.replace("tl.seek(0);", "cardMotionDrawn(root, tl, duration);\n    tl.seek(0);")


def _scripts(plan: CardPlan, composition: Path, inner: str,
             renderer: hyperframes.RendererInstall) -> tuple[str, str]:
    from yap_studio.hf_registry import localize

    helper = localize.localize("echo-smear", composition / "motion", install=renderer)
    shutil.copyfile(hyperframes.gsap_file(renderer), composition / hyperframes.GSAP_SCRIPT)
    duration = hyperframes.layer_duration_s(plan.frames, plan.fps)
    offset = (1080 - plan.x + 40 if plan.motion["side"] == "right"
              else -plan.x - plan.width - 40)
    config = json.dumps({"duration": duration, "offset": offset,
                         "enter": plan.motion["enter"], "inner": inner,
                         "settle": max(0, duration - min(0.3, duration * 0.15) - 1 / plan.fps)})
    script = (_SCRIPTS / "card-motion-host.js").read_text(encoding="utf-8")
    paths.atomic_write_text(composition / "card-motion-host.js", script)
    return (f'<script src="motion/{helper.main}"></script>',
            f'window.__cardMotion={config};window.__cardMotionTimeline=tl;\n' +
            '</script><script src="card-motion-host.js"></script><script>')


def prepare_builtin(plan: CardPlan, inner: Path, *,
                    install: hyperframes.RendererInstall | None = None) -> Path:
    """Mount the ordinary card box inside a full-width transparent band."""
    from yap_studio.render import fonts

    composition = paths.validate_output_dir(inner.with_name(inner.name + "-motion"))
    shutil.copytree(inner, composition / "card", dirs_exist_ok=True)
    fonts.copy_fonts(plan.fonts, composition / "fonts")
    for filename, _ in plan.assets:
        shutil.copyfile(inner / filename, composition / filename)
    renderer = install or hyperframes.ensure_installed()
    helper, motion = _scripts(plan, composition, plan.template, renderer)
    duration = hyperframes.layer_duration_s(plan.frames, plan.fps)
    variables = html.escape(json.dumps(plan.variables, ensure_ascii=False).replace("/", r"\/"),
                            quote=False).replace("'", "&#39;")
    source = (
        '<!doctype html><html><head><meta charset="utf-8"><script src="gsap.min.js"></script>'
        f'<style>html,body{{margin:0;width:1080px;height:{plan.height}px;'
        'background:transparent;overflow:hidden}'
        f'#band{{position:relative;width:1080px;height:{plan.height}px;overflow:hidden}}'
        '</style></head><body><div id="band" data-composition-id="band" data-width="1080" '
        f'data-height="{plan.height}" data-start="0" data-duration="{duration}">'
        f'<div id="card-motion-in" style="position:absolute;left:{plan.x}px;top:0;'
        f'width:{plan.width}px;height:{plan.height}px">'
        f'<div id="hf-mount" data-card-motion="streak" data-card-duration="{duration}" '
        'data-hf-authored-id="root" '
        f'data-composition-id="{plan.template}" data-composition-src="card/index.html" '
        f'data-start="0" data-duration="{duration}" data-width="{plan.width}" '
        f'data-height="{plan.height}" data-track-index="0" data-variable-values=\'{variables}\' '
        f'style="position:absolute;left:0;top:0;width:{plan.width}px;'
        f'height:{plan.height}px"></div></div></div>{helper}'
        '<script>const tl=gsap.timeline({paused:true});' + motion + '</script></body></html>'
    )
    paths.atomic_write_text(composition / "index.html", source)
    return composition


def prepare_registry(plan: CardPlan, composition: Path, *,
                     install: hyperframes.RendererInstall | None = None) -> Path:
    """Retain the item's entrance and shared fade, adding only a horizontal exit."""
    source = (composition / "index.html").read_text(encoding="utf-8")
    original = composition
    composition = paths.validate_output_dir(original.with_name(original.name + "-motion"))
    shutil.copytree(original, composition, dirs_exist_ok=True,
                    ignore=lambda folder, names: ["index.html"] if Path(folder) == original else [])
    helper, motion = _scripts(plan, composition, plan.template.split(":", 1)[1],
                              install or hyperframes.ensure_installed())
    source = source.replace(f"width:{plan.width}px;height:{plan.height}px;background:transparent",
                            f"width:1080px;height:{plan.height}px;background:transparent", 1)
    source = re.sub(r"(#hf(?:-card)?-host\{position:relative;width:)\d+px",
                    r"\g<1>1080px", source, count=1)
    source = source.replace(f'data-width="{plan.width}"', 'data-width="1080"', 1)
    source = source.replace('style="position:absolute;left:',
                            f'style="position:absolute;margin-left:{plan.x}px;left:', 1)
    source = source.replace('<script>const D=', helper + '<script>const D=', 1)
    source = source.replace('window.__timelines=window.__timelines||{};'
                            'window.__timelines["hf-host"]=tl;', motion, 1)
    paths.atomic_write_text(composition / "index.html", source)
    return composition
