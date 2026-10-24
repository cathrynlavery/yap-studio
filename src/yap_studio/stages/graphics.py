"""Render resolved titles, chapters and cards as placed project layers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from yap_studio import cards, grade, paths, receipt
from yap_studio.brand_words import list_words
from yap_studio.captions import (
    CaptionWord,
    kinetic_display_times,
    kinetic_word_times,
)
from yap_studio.card_motion import plan_motion
from yap_studio.card_project import PROJECT_CARD_TEMPLATES, CardShot, project_facts
from yap_studio.card_stills import cached_frame, cached_still
from yap_studio.cards_video import caption_ceiling
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import (
    DEFAULT_HAIR,
    face_boxes_per_frame,
    face_clearance_px,
    gap_px,
    head_box,
)
from yap_studio.faces.mapping import behind_face_windows
from yap_studio.faces.model import Box, FacesData
from yap_studio.faces.placement import Band, suggest_offset
from yap_studio.faces.project import cut_faces_identity, picture_faces
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.hf_registry import catalog
from yap_studio.matte.project import ensure_cut_matte
from yap_studio.planner.cards import (
    card_caption_top,
    fitted_card,
    place_card,
    place_video_card,
    safe_card_order,
)
from yap_studio.planner.number_behind import resize_number
from yap_studio.project.model import CutWords, parse_document
from yap_studio.project.receipts import StageRun, projection_sha256, read_receipt
from yap_studio.project.status import ProjectContext
from yap_studio.render import fonts, hyperframes
from yap_studio.sfx import bleep
from yap_studio.stages.captions import (
    LayerAssets,
    LayerPlan,
    chapter_layer_plan,
    chapter_text_box,
    layer_assets,
    render_layer_plan,
    scale_kinetic_title,
    title_layer_plan,
)
from yap_studio.stages.number_graphics import (
    number_warnings,
    place_number_graphic,
    publish_number_layer,
)
from yap_studio.style import ResolvedStyle
from yap_studio.text_hook import TEXT_HOOK_ID, require_top_text
from yap_studio.text_rules import apply_title_rules


@dataclass(frozen=True)
class TitleSnapshot:
    """One title or chapter's render, placement and input digest."""

    id: str
    layer: str
    plan: LayerPlan
    digest: str
    clearance: dict
    assets: LayerAssets | None = None
    display_text: tuple[str, ...] | None = None

    @property
    def kind(self) -> str:
        """Keep timeline identity separate from the rendering template."""
        if self.plan.template == "wide_titles":
            kind = self.plan.variables["kind"]
            return "graphic" if kind == "name_tag" else kind
        return "chapter" if self.plan.template == "chapter" else "title"

    def placement(self) -> dict[str, Any]:
        """Return this title's assembly contract without composition-local paths."""
        return {"id": self.id, "kind": self.kind, "layer": self.layer,
                **self.plan.placement(f"graphics/{self.id}.mov"), "input_sha256": self.digest,
                "clearance": self.clearance,
                **({"letters": True} if self.plan.template == "letters" else {}),
                **({"display_text": list(self.display_text)}
                   if self.display_text is not None else {})}


@dataclass(frozen=True)
class GraphicSnapshot:
    """One card's fitted plan, retained image bytes, placement and input digest."""

    id: str
    layer: str
    start_frame: int
    plan: cards.CardPlan
    assets: LayerAssets
    images: tuple[tuple[str, bytes], ...]
    digest: str
    clearance: dict
    shot_sources: tuple[tuple[str, CardShot], ...] = ()
    project_dir: Path | None = None
    poster_grade: grade.GradeSpec | None = None
    display_text: tuple[str, ...] | None = None

    def placement(self) -> dict[str, Any]:
        """Use the same box and frame contract as independently rendered titles."""
        result = {"id": self.id, "kind": "graphic", "layer": self.layer,
                "path": f"graphics/{self.id}.mov", "start_frame": self.start_frame,
                "frames": self.plan.frames, **self.plan.layer_box(),
                "input_sha256": self.digest, "clearance": self.clearance}
        if self.display_text is not None:
            result["display_text"] = list(self.display_text)
        if self.plan.windows:
            result["windows"] = [
                {"slot": window["slot"], "index": window["index"],
                 "x": self.plan.x + window["x"], "y": self.plan.y + window["y"],
                 "width": window["width"], "height": window["height"],
                 "radius": window["radius"], "start_frame": self.start_frame + window["start"],
                 "frames": window["frames"], "fade_in_frames": window["fade_in"],
                 "fade_out_frames": window["fade_out"], "source": copy.deepcopy(window["source"]),
                 **({"motion": {"start_frame": self.start_frame + window["motion"]["start"],
                                "frames": window["motion"]["frames"],
                                "to": [self.plan.x + window["motion"]["to"][0],
                                       self.plan.y + window["motion"]["to"][1],
                                       *window["motion"]["to"][2:]],
                                "ease": window["motion"]["ease"]}}
                    if "motion" in window else {})}
                for window in self.plan.windows
            ]
        if self.plan.template == "playing_grid":
            result["parts"] = json.loads(self.plan.variables["tiles"])
        if self.plan.template == "reel_feed":
            result["full_frame"] = True
            result["backdrop"] = self.plan.variables["backdrop"]
        return result


@dataclass(frozen=True)
class GraphicsSnapshot:
    """Current title and card plans with assets and per-element projections."""

    titles: tuple[TitleSnapshot, ...]
    graphics: tuple[GraphicSnapshot, ...]
    assets: LayerAssets
    inputs: dict[str, str]


def _card_assets(plan: cards.CardPlan, context: ProjectContext) -> LayerAssets:
    """Identify the card implementation and both proportional and mono font faces."""
    template = cards.template_dir(plan.template)
    faces = tuple((name, context.read(str(paths.validate_input(source))))
                  for name, source in fonts.font_files(plan.fonts))
    if registry_cards.is_kind(plan.template):
        entry = catalog.item(registry_cards.item_name(plan.template))
        files = {name: hashlib.sha256(context.read(str(template / name))).hexdigest()
                 for name in entry["files"]}
        files["adapter"] = context.read(str(Path(registry_cards.__file__))).decode("utf-8")
        metadata = {"version": 1, "catalog": catalog.card_catalog(),
                    "catalog_sha256": hashlib.sha256(
                        catalog.card_catalog_source(
                            context.read(str(catalog.CATALOG_PATH)))).hexdigest()}
        return LayerAssets(json.dumps(files, sort_keys=True), json.dumps(metadata).encode("utf-8"),
                           fonts.font_face_css(plan.fonts), faces)
    return LayerAssets(cards.template_html(plan.template, read=lambda p: context.read(str(p))),
                       context.read(str(template / "template.json")),
                       fonts.font_face_css(plan.fonts), faces)


def _validate_card_box(box: Any) -> None:
    """Require integer pixel bounds inside the delivery frame before refitting."""
    keys = {"x", "y", "width", "height"}
    if (not isinstance(box, dict) or set(box) != keys
            or any(isinstance(value, bool) or not isinstance(value, int) for value in box.values())
            or box["x"] < 0 or box["y"] < 0 or box["width"] <= 0 or box["height"] <= 0
            or box["x"] + box["width"] > 1080 or box["y"] + box["height"] > 1920):
        raise ValidationError("graphic.box",
                              "use x, y, width and height in pixels inside 1080x1920")


def _card_box(plan: cards.CardPlan, box: Any) -> cards.CardPlan:
    """Require the entire fitted panel, including labels and padding, to fit its box."""
    _validate_card_box(box)
    if box["width"] < plan.width or box["height"] < plan.height:
        raise ValidationError("graphic.box", f"give the whole card panel a width of at least "
                              f"{plan.width} px and height of at least {plan.height} px, "
                              "including its label and padding")
    return replace(plan, **box)


def _video_paths(slots: dict, rules: dict, context: ProjectContext) -> None:
    """Resolve project clips and bind their declared byte identity before planning."""
    for name, rule in rules.items():
        if rule["type"] not in {"video", "video_list"} or name not in slots:
            continue
        values = slots[name] if rule["type"] == "video_list" else [slots[name]]
        if not isinstance(values, list):
            continue
        resolved = []
        for value in values:
            if isinstance(value, dict) and isinstance(value.get("path"), str):
                source = paths.validate_input(context.project_dir / value["path"]).resolve()
                if not source.is_relative_to(context.project_dir):
                    raise ValidationError(f"slots.{name}", "use a file inside the project folder")
                if context.identity.sha256(source) != value.get("sha256"):
                    raise ValidationError(f"slots.{name}", "the clip changed; update its sha256")
                value = {**value, "path": str(source)}
            resolved.append(value)
        slots[name] = resolved if rule["type"] == "video_list" else resolved[0]


def _render_card(snapshot: GraphicSnapshot, work: Path, out: Path) -> hyperframes.LayerResult:
    """Render retained images and refuse template or font changes during rendering."""
    def check_assets() -> None:
        if registry_cards.is_kind(snapshot.plan.template):
            current = _card_assets(snapshot.plan, ProjectContext(snapshot.project_dir))
            if current != snapshot.assets:
                raise ValidationError("graphic.inputs",
                                      "template or font changed; run graphics again")
            return
        template = cards.template_dir(snapshot.plan.template)
        try:
            current_html = cards.template_html(snapshot.plan.template)
        except OSError:
            current_html = None
        files = [(template / "template.json", snapshot.assets.metadata),
                 *((fonts.font_source(name), data) for name, data in snapshot.assets.font_files)]
        # The card API reads these shared files; a changed implementation cannot claim this digest.
        current_faces = [name for name, _ in fonts.font_files(snapshot.plan.fonts)]
        saved_faces = [name for name, _ in snapshot.assets.font_files]
        if (current_html != snapshot.assets.html or
                current_faces != saved_faces or
                fonts.font_face_css(snapshot.plan.fonts) != snapshot.assets.font_css or
                any(not target.is_file() or target.read_bytes() != data for target, data in files)):
            raise ValidationError("graphic.inputs", "template or font changed; run graphics again")

    check_assets()
    assets = []
    for filename, data in snapshot.images:
        target = work / "assets" / filename
        with paths.atomic_output(target) as temporary:
            temporary.write_bytes(data)
        assets.append((filename, target))
    verified_sources = {}
    for filename, shot in snapshot.shot_sources:
        source = cached_still(snapshot.project_dir, shot,
                              width=snapshot.plan.variables["thumb_width_px"] * 2,
                              height=snapshot.plan.variables["thumb_height_px"] * 2,
                              verified_sources=verified_sources)
        assets.append((filename, source))
    plan = snapshot.plan
    if plan.poster is not None and snapshot.poster_grade is not None:
        spec = snapshot.poster_grade
        cube = grade.write_cube(spec, work / "poster-grade")
        poster = plan.poster
        source = cached_frame(snapshot.project_dir, poster.source, poster.sha256,
                               source_time=poster.time_s,
                               width=plan.variables["poster_width_px"] * 2,
                               height=plan.variables["poster_height_px"] * 2,
                               source_kind="clip", cube=cube,
                               grade_identity={**spec.to_dict(), "version": grade.GRADE_VERSION})
        assets.append(("poster.png", source))
        plan = replace(plan, poster=None, variables={**plan.variables, "poster_src": "poster.png"})
    if plan.template == "big_number":
        rendered = publish_number_layer(plan, snapshot.project_dir, out,
                                       assets=snapshot.assets)
    else:
        rendered = cards.render_card(replace(plan, assets=tuple(assets)), work, out)
    check_assets()
    return rendered


def _place_kinetic_title(
    plan: LayerPlan, style: ResolvedStyle, faces: FacesData,
) -> tuple[LayerPlan, dict]:
    """Centre above the highest head, preserving a readable floor when space runs out."""
    margin = face_clearance_px(style.data)
    evidence = {"status": "unchecked", "reason": faces.reason,
                "margin_px": margin, "hair": DEFAULT_HAIR.to_dict(),
                "original_y": plan.y, "original_scale": 1, "scale": 1}
    if faces.status != "measured":
        return plan, evidence
    window = face_boxes_per_frame(faces)[plan.start_frame:plan.start_frame + plan.frames]
    heads = [head_box(face) for frame in window for face in frame]
    if not heads:
        evidence.update(status="clear", reason="no face in the title window", y=plan.y)
        return plan, evidence
    bottom = max(0, min(1920, math.floor(min(head.y for head in heads) - margin)))
    sizes = [line["size_px"] for line in json.loads(plan.variables["kinetic_lines"])]
    floor_scale = 110 / max(sizes)
    scales = [step / 40 for step in range(40, 0, -1) if step / 40 > floor_scale]
    for scale in [*scales, floor_scale]:
        candidate = scale_kinetic_title(plan, scale)
        if candidate.height <= bottom:
            candidate = replace(candidate, y=round((bottom - candidate.height) / 2))
            evidence.update(status="clear", reason="kinetic title centred clear above the head",
                            y=candidate.y, scale=scale)
            return candidate, evidence
    candidate = replace(candidate, y=0)
    evidence.update(status="overlap", reason="kinetic title at the 110 px floor overlaps the head; "
                    "there is not enough clear space above it", y=0, scale=floor_scale)
    return candidate, evidence


def place_title(plan: LayerPlan, style: ResolvedStyle, faces: FacesData) -> tuple[LayerPlan, dict]:
    """Keep a clear default, otherwise move up and then shrink only this layer."""
    if plan.variables["look"] == "kinetic":
        return _place_kinetic_title(plan, style, faces)
    evidence = {"status": "unchecked", "reason": faces.reason,
                "margin_px": face_clearance_px(style.data), "hair": DEFAULT_HAIR.to_dict(),
                "original_y": plan.y, "original_size_px": plan.variables["size_px"]}
    if faces.status != "measured":
        return plan, evidence
    window = face_boxes_per_frame(faces)[plan.start_frame:plan.start_frame + plan.frames]
    if plan.variables["look"] == "text_hook":
        hint = suggest_offset(window, x=plan.x, width=plan.width, height=plan.height,
                              band=Band(120, plan.y + plan.height),
                              margin_px=evidence["margin_px"], preferred_y=plan.y)
        if hint.y is None:
            raise ValidationError("text_hook", "Text hook would cover the face. Please use "
                                  "shorter text or a take with more room above the face.")
        evidence.update(status="clear", reason=hint.reason, y=hint.y,
                        size_px=plan.variables["size_px"])
        return replace(plan, y=hint.y), evidence
    default = suggest_offset(window, x=plan.x, width=plan.width, height=plan.height,
                             band=Band(0, 1920), margin_px=evidence["margin_px"],
                             preferred_y=plan.y)
    if default.y == plan.y:
        evidence.update(status="clear", reason=default.reason, y=plan.y,
                        size_px=plan.variables["size_px"])
        return plan, evidence
    original_size = plan.variables["size_px"]
    for size in range(original_size, 39, -1):
        candidate = plan
        if size != original_size:
            data = copy.deepcopy(style.data)
            data["hook"].update(position="top", size_px=size)
            candidate = title_layer_plan(
                plan.variables["plain"], plan.variables["key"], replace(style, data=data),
                lines=json.loads(plan.variables["lines"]) or None,
                frames=plan.frames, start_frame=plan.start_frame,
                word_times=json.loads(plan.variables["word_times"]),
            )
        hint = suggest_offset(
            window, x=candidate.x, width=candidate.width, height=candidate.height,
            band=Band(0, style.data["internal"]["key_text_bottom_px"]),
            margin_px=evidence["margin_px"], preferred_y=min(plan.y, candidate.y),
        )
        if hint.y is None:
            continue
        candidate = replace(candidate, y=hint.y)
        evidence.update(status="clear", y=hint.y, size_px=size,
                        reason="default title collides with the head; " + hint.reason
                               + (f"; shrank from {original_size} to {size} px"
                                  if size != original_size else ""))
        return candidate, evidence
    raise ValidationError("title.clearance", "no readable title size fits above the head; "
                          "shorten the title or choose a different window")


def _caption_top(context: ProjectContext, style: ResolvedStyle, start: int, end: int) -> int:
    try:
        layout = context.json("captions/layout.json")
    except FileNotFoundError:
        layout = {}
    return card_caption_top(style, start, end, layout)


def graphics_snapshot(
    context: ProjectContext, *, placements: list[dict] | None = None,
) -> GraphicsSnapshot:
    """Plan new layers, or reconstruct receipted layers without measuring media."""
    style = context.render_style()
    bleep_words = bleep.listed(style.data["music"]) if bleep.enabled(style.data) else None
    bleep_input = ({"bleep": projection_sha256(bleep.projection(style.data["music"]))}
                   if bleep_words is not None else {})
    timeline = context.resolved_timeline()
    assets = layer_assets("letters" if style.data["hook"]["style"] == "video_letters" else "title",
                          "Inter" if style.data["hook"]["style"] == "text_hook"
                          else style.data["hook"]["font"], context)
    vocabulary = (tuple(list_words()) if any(item.kind == "title" for item in timeline.elements)
                  else ())
    if placements is not None:
        required = {"id", "kind", "layer", "x", "y", "width", "height",
                    "start_frame", "frames", "clearance"}
        if (not isinstance(placements, list)
                or any(not isinstance(item, dict) or not required <= item.keys()
                       or not isinstance(item["id"], str)
                       or not isinstance(item["clearance"], dict) for item in placements)):
            raise ValidationError("graphics.receipt", "invalid layer placements; rerun graphics")
    saved = {placement["id"]: placement for placement in placements or []}
    faces = None
    face_frames = None
    number_matte = None
    titles: list[TitleSnapshot] = []
    graphics: list[GraphicSnapshot] = []
    words = None
    chapters = sorted((item for item in timeline.elements if item.kind == "chapter"),
                      key=lambda item: (item.start_frame, item.id))
    numbers = {item.id: index for index, item in enumerate(chapters, 1)}
    chapter_assets = (layer_assets("chapter", style.data["hook"]["font"], context)
                      if chapters else None)
    for element in timeline.elements:
        if element.kind not in {"title", "chapter", "graphic"}:
            continue
        if re.fullmatch(r"agent-[A-Za-z0-9][A-Za-z0-9_.-]*", element.id) is None:
            raise ValidationError(f"{element.kind}.id",
                                  "use agent- followed by a portable file name")
        placement_record = saved.get(element.id)
        if placements is not None and placement_record is None:
            raise ValidationError("graphics.receipt",
                                  f"no placement for {element.id}; rerun graphics")
        if placement_record is not None and (
            placement_record["kind"] != element.kind
            or placement_record["start_frame"] != element.start_frame
            or placement_record["frames"] != element.end_frame - element.start_frame
            or placement_record["layer"] not in {"front", "behind"}
        ):
            raise ValidationError("graphics.receipt", f"placement changed for {element.id}")
        if context.project.shape == "horizontal" and (
            element.kind in {"title", "chapter"}
            or element.kind == "graphic" and element.payload["template"] == "name_tag"
        ):
            from yap_studio.wide_titles import wide_layer_plan

            kind = "name_tag" if element.kind == "graphic" else element.kind
            if kind == "name_tag" and cards.card_kind_off(style, kind):
                raise ValidationError(f"graphic.{element.id}.template",
                                      cards.card_kind_off_detail(style, kind))
            if kind == "title":
                lines, _ = apply_title_rules(element.payload["lines"], "", vocabulary)
                text = {"title": " ".join(lines)}
            else:
                text = element.payload["slots"] if kind == "name_tag" else element.payload
            if bleep_words is not None:
                text = {key: bleep.mask_text(value, bleep_words) if isinstance(value, str)
                        else value for key, value in text.items()}
            if faces is None and placement_record is None:
                faces = picture_faces(context.project_dir, timeline, context)
                face_frames = (face_boxes_per_frame(faces) if faces.status == "measured" else None)
            plan, clearance = wide_layer_plan(
                kind, text, style, frames=element.end_frame - element.start_frame,
                start_frame=element.start_frame,
                faces=face_frames[element.start_frame:element.end_frame]
                if face_frames is not None else None,
                reason=faces.reason if faces is not None else None,
                number=numbers.get(element.id, 1), total=len(chapters),
                saved_box=Box(**placement_record["clearance"]["box"])
                if placement_record is not None else None,
                caption_top=None if any(
                    item.kind == "caption_gap" and item.start_frame == 0
                    and item.end_frame >= timeline.total_frames for item in timeline.elements
                ) else _caption_top(context, style, element.start_frame, element.end_frame),
            )
            if placement_record is not None:
                clearance = placement_record["clearance"]
            wide_assets = layer_assets("wide_titles", plan.variables["font_name"], context)
            digest = projection_sha256({
                "element": element.to_dict(), "variables": plan.variables, "clearance": clearance,
                "fonts": wide_assets.font_identity(),
                "template": wide_assets.template_identity(workers=hyperframes.RENDER_WORKERS),
                **bleep_input,
            })
            titles.append(TitleSnapshot(element.id, "front", plan, digest, clearance, wide_assets))
            continue
        if element.kind == "chapter":
            chapter_title = (bleep.mask_text(element.payload["title"], bleep_words)
                             if bleep_words is not None else element.payload["title"])
            plan = chapter_layer_plan(chapter_title, numbers[element.id], style,
                                      start_frame=element.start_frame,
                                      frames=element.end_frame - element.start_frame,
                                      element_id=element.id)
            clearance = {"status": "full_frame",
                         "reason": "chapter text sits over the live picture by design"}
            if placement_record is not None and any(
                placement_record[key] != value for key, value in {
                    "x": 0, "y": 0, "width": 1080, "height": 1920,
                    "layer": "front", "clearance": clearance,
                }.items()
            ):
                raise ValidationError("graphics.receipt", "chapter layer must span the full frame")
            assert chapter_assets is not None
            digest = projection_sha256({
                "element": element.to_dict(), "variables": plan.variables,
                "chapters": style.data["chapters"],
                "colors": {"paper": style.data["graphics"]["paper_color"],
                           "ink": style.data["graphics"]["ink_color"],
                           "accent": style.data["graphics"]["accent_color"],
                           "caption": style.data["captions"]["text_color"]},
                "fonts": chapter_assets.font_identity(),
                "template": chapter_assets.template_identity(workers=hyperframes.RENDER_WORKERS),
                **bleep_input,
            })
            titles.append(TitleSnapshot(element.id, "front", plan, digest, clearance,
                                        chapter_assets,
                                        (chapter_title,) if bleep_words is not None else None))
            continue
        if element.kind == "graphic":
            template = element.payload["template"]
            if placement_record is not None:
                motion_record = placement_record["clearance"].get("card_motion", {})
                requested_motion = style.data["graphics"].get("card_motion", "none")
                # A fallback note still records a request for streak.
                if (requested_motion == "streak"
                        and motion_record.get("status") not in {"none", "streak"}
                        or requested_motion == "none" and motion_record):
                    placement_record = None
            if cards.card_kind_off(style, template):
                raise ValidationError(
                    f"graphic.{element.id}.template",
                    cards.card_kind_off_detail(style, template),
                )
            if registry_cards.is_kind(template):
                context.read(str(catalog.CATALOG_PATH))
                metadata = registry_cards.template_metadata(template)
            else:
                metadata = json.loads(context.read(
                    str(cards.template_dir(template) / "template.json")))
            def relative_times(value: Any, *, first: int = element.start_frame) -> Any:
                if isinstance(value, list):
                    return [relative_times(child) for child in value]
                if isinstance(value, dict):
                    return {
                        key: (child["frame"] - first) / 30
                        if key == "at" and isinstance(child, dict)
                        and set(child) == {"frame", "time_s"} else relative_times(child)
                        for key, child in value.items()
                    }
                return value

            slots = relative_times(element.payload["slots"])
            display_text = None
            if bleep_words is not None:
                slots, display_text = bleep.mask_slots(slots, metadata["slots"], bleep_words)
            _video_paths(slots, metadata["slots"], context)
            for name, rule in metadata["slots"].items():
                if rule["type"] == "image_list" and isinstance(slots.get(name), list):
                    resolved = []
                    for index, value in enumerate(slots[name]):
                        if isinstance(value, str):
                            source = paths.validate_input(context.project_dir / value).resolve()
                            if not source.is_relative_to(context.project_dir):
                                raise ValidationError(f"graphic.slots.{name}[{index}]",
                                                      "use a file inside the project folder")
                            value = source
                        resolved.append(value)
                    slots[name] = resolved
                if ((rule["type"] == "image"
                     or (rule["type"] == "video" and "sources" not in rule)) and name in slots
                        and isinstance(slots[name], str)):
                    if slots[name] == "" and not rule["required"]:
                        continue
                    source = paths.validate_input(context.project_dir / slots[name]).resolve()
                    if not source.is_relative_to(context.project_dir):
                        raise ValidationError(f"graphic.slots.{name}",
                                              "use a file inside the project folder")
                    slots[name] = source
            project = (project_facts(context, timeline, element)
                       if template in PROJECT_CARD_TEMPLATES else None)
            scene = template in {"playing_grid", "reel_feed"}
            safe_order = safe_card_order(template, style, slots)
            if scene:
                if "box" in element.payload:
                    raise ValidationError("graphic.box", "full-frame cards arrange their own "
                                          "windows; remove box")
                ceiling = caption_ceiling(style, _caption_top(
                    context, style, element.start_frame, element.end_frame))
                placement = "behind_head" if element.layer == "behind" else "top"
                if template == "reel_feed":
                    card = cards.card_layer_plan(template, slots, style,
                                                 frames=element.end_frame - element.start_frame,
                                                 ceiling=ceiling)
                    placement = "top"
                    clearance = {"status": "full_frame", "reason": "the feed fills the frame "
                                 "and shows the video inside the phone"}
                elif placement_record is not None:
                    if "parts" not in placement_record:
                        raise ValidationError("graphics.receipt",
                                              "grid parts changed; rerun graphics")
                    card = cards.card_layer_plan(template, slots, style,
                                                 frames=element.end_frame - element.start_frame,
                                                 tiles=placement_record["parts"], ceiling=ceiling)
                else:
                    if faces is None:
                        faces = picture_faces(context.project_dir, timeline, context)
                        face_frames = (face_boxes_per_frame(faces)
                                       if faces.status == "measured" else None)
                    card, clearance = cards.grid_card_plan(
                        slots, style,
                        face_frames[element.start_frame:element.end_frame]
                        if face_frames is not None else None,
                        frames=element.end_frame - element.start_frame, ceiling=ceiling,
                        layer=element.layer, reason=faces.reason, id=element.id,
                        face_windows=behind_face_windows(timeline, element.start_frame,
                                                         element.end_frame),
                        start_frame=element.start_frame,
                    )
            else:
                card = cards.card_layer_plan(template, slots, style,
                                             frames=element.end_frame - element.start_frame,
                                             project=project)
            if "box" in element.payload and template != "big_number":
                box = element.payload["box"]
                _validate_card_box(box)
                card = fitted_card(template, slots, style,
                                   frames=element.end_frame - element.start_frame,
                                   width=box["width"], project=project,
                                   height=box["height"])
                card = _card_box(card, element.payload["box"])
            if placement_record is not None:
                box = {key: placement_record[key] for key in ("x", "y", "width", "height")}
                motion_record = placement_record["clearance"].get("card_motion", {})
                if motion_record.get("status") == "streak":
                    box = motion_record["rest_box"]
                if scene and box != {"x": 0, "y": 0, "width": 1080, "height": 1920}:
                    raise ValidationError("graphics.receipt",
                                          "full-frame box changed; rerun graphics")
                if template == "big_number":
                    if placement_record["clearance"].get("ink_status") not in {
                        "verified", "front", "unavailable",
                    }:
                        raise ValidationError("graphics.receipt", "number ink was not checked; "
                                              "run graphics again")
                    card = resize_number(card, placement_record["clearance"]["value_px"])
                elif (card.width != box["width"] or card.height != box["height"]
                      or safe_order is not None):
                    card = fitted_card(template, slots, style,
                                       frames=element.end_frame - element.start_frame,
                                       width=box["width"], project=project,
                                       height=box["height"])
                card = _card_box(card, box)
                card = replace(card, motion={key: motion_record[key] for key in ("side", "enter")}
                               if motion_record.get("status") == "streak" else None)
                placement = "behind_head" if placement_record["layer"] == "behind" else "top"
                clearance = placement_record["clearance"]
            elif not scene:
                if faces is None:
                    faces = picture_faces(context.project_dir, timeline, context)
                    face_frames = (face_boxes_per_frame(faces)
                                   if faces.status == "measured" else None)
                captions_top = _caption_top(context, style, element.start_frame, element.end_frame)
                if template == "big_number":
                    if number_matte is None:
                        number_matte = ensure_cut_matte(context.project_dir,
                                                       context.project_dir / "cut/cut.mp4")
                    card, placement, clearance = place_number_graphic(
                        card, element, style,
                        face_frames[element.start_frame:element.end_frame]
                        if face_frames is not None else None, number_matte, int(captions_top),
                        face_windows=behind_face_windows(timeline, element.start_frame,
                                                         element.end_frame),
                        project_dir=context.project_dir,
                    )
                else:
                    placer = (place_video_card if template == "video_card"
                              and "box" not in element.payload else place_card)
                    card, placement, clearance = placer(
                        card, slots, style,
                        face_frames[element.start_frame:element.end_frame]
                        if face_frames is not None else None,
                        reason=faces.reason,
                        face_windows=behind_face_windows(timeline, element.start_frame,
                                                         element.end_frame),
                        caption_top=int(captions_top),
                        preserve_box="box" in element.payload,
                        requested_layer=element.layer,
                        project=project,
                        **({"safe_order": safe_order} if safe_order is not None else {}),
                    )
                if clearance["status"] == "blocked":
                    raise ValidationError(f"graphic.{element.id}.clearance", clearance["reason"])
            if placement_record is None and "card_motion" not in clearance:
                card, clearance = plan_motion(
                    card, style, face_frames[element.start_frame:element.end_frame]
                    if face_frames is not None else None, clearance)
            if bleep_words is not None and template == "file_drop":
                display_text = [card.variables[key] for key in ("name", "badge", "duration_text")]
            elif bleep_words is not None and template in {"comment_drop", "reel_feed"}:
                display_text.append(card.variables["initials"])
                if template == "reel_feed":
                    display_text.append(card.variables["views_text"])
            card_assets = _card_assets(card, context)
            poster_grade = None
            if card.poster is not None:
                card = replace(card, poster=replace(
                    card.poster, sha256=context.identity.sha256(card.poster.source),
                    cache_root=context.project_dir,
                ))
                if template == "file_drop" and card.windows:
                    spec = grade.resolve_grade(style.data, context.project.grade,
                                                cut=context.project_dir / "cut/cut.mp4")
                    if spec.active:
                        poster_grade = spec
            images = tuple((filename, context.read(str(source)))
                           for filename, source in card.assets)
            card = replace(card, windows=tuple(
                {**window, "source": {**window["source"],
                    "sha256": window["source"].get("sha256") or context.identity.sha256(
                        Path(window["source"]["path"])), "path": str(
                    Path(window["source"]["path"]).relative_to(context.project_dir))}}
                if window["source"]["kind"] == "media" else window
                for window in card.windows
            ))
            projection = {
                "element": element.to_dict(),
                **bleep_input,
                **({"display_text": display_text} if display_text is not None else {}),
                **({"number": card.variables} if template == "big_number" else {}),
                "placement": {"x": card.x, "y": card.y,
                              "width": card.width, "height": card.height},
                "clearance": clearance,
                "graphics": {key: value for key, value in style.data["graphics"].items()
                             if key != "caption_gap_px"
                             and (key != "flow_look" or template == "flow")
                             and (key != "specs_rows" or template in {"specs_list", "stat"})
                             and (key != "card_motion" or value != "none")},
                "internal": style.data["internal"], "fonts": card_assets.font_identity(),
                "template": card_assets.template_identity(
                    workers=hyperframes.RENDER_WORKERS, browser_gpu=hyperframes.CARD_BROWSER_GPU),
                "images": {name: hashlib.sha256(data).hexdigest() for name, data in images},
                **({"captions": {key: style.data["captions"][key] for key in (
                    "text_color", "size_px", "shadow_opacity", "shadow_blur_px",
                    "shadow_x_px", "shadow_y_px",
                )}, "font_cycle": cards.gravity_drop_fonts(style)}
                   if template == "gravity_drop" else {}),
                **({"project": asdict(project)} if project is not None else {}),
                **({"poster": {"sha256": card.poster.sha256, "time_s": card.poster.time_s}}
                   if card.poster is not None else {}),
            }
            if template == "file_drop" and not card.windows:
                projection["element"]["slots"].pop("playback", None)
            if poster_grade is not None:
                projection["poster"]["grade"] = {**poster_grade.to_dict(),
                                                 "version": grade.GRADE_VERSION}
            if card.windows:
                projection["windows"] = card.windows
            if card.motion:
                from yap_studio import card_motion

                projection["motion"] = {**card.motion, "helper": catalog.item("echo-smear"),
                    "source": hashlib.sha256(context.read(str(Path(card_motion.__file__))))
                    .hexdigest(), "scripts": {name: hashlib.sha256(context.read(
                        str(card_motion._SCRIPTS / name))).hexdigest() for name in (
                            "card-motion-host.js", "card-motion-drawn.js")}}
            if template == "playing_grid":
                projection["parts"] = json.loads(card.variables["tiles"])
            if template == "reel_feed":
                projection["full_frame"] = True
                projection["backdrop"] = card.variables["backdrop"]
            digest = projection_sha256(projection)
            layer = "behind" if placement == "behind_head" else "front"
            shot_sources = (tuple((thumbnail["src"], project.shots[thumbnail["index"]])
                                  for thumbnail in json.loads(card.variables["thumbnails"])
                                  if thumbnail["src"])
                            if template == "storyboard" else ())
            snapshot = GraphicSnapshot(element.id, layer, element.start_frame,
                                       card, card_assets, images, digest, clearance,
                                       shot_sources, context.project_dir, poster_grade,
                                       tuple(display_text) if display_text is not None else None)
            if (placement_record is not None and scene
                    and placement_record.get("full_frame", False) != (template == "reel_feed")):
                raise ValidationError("graphics.receipt",
                                      "full-frame marker changed; rerun graphics")
            if (card.windows and placement_record is not None
                    and placement_record.get("input_sha256") == digest
                    and placement_record.get("windows") != snapshot.placement()["windows"]):
                raise ValidationError("graphics.receipt", "video windows changed; rerun graphics")
            graphics.append(snapshot)
            continue
        if style.data["hook"]["style"] == "text_hook":
            lines, accent = element.payload["lines"], ""
        else:
            lines, accent = apply_title_rules(
                element.payload["lines"], element.payload.get("accent") or "", vocabulary)
        word_fix_inputs = {}
        word_times = None
        if style.data["hook"]["style"] == "kinetic":
            if words is None:
                cut_words = cast(CutWords, parse_document("cut-words",
                                                         context.json("cut/cut-words.json")))
                words = [CaptionWord(word.id, word.text, word.start, word.end)
                         for word in cut_words.words]
            # Match what the agent wrote against speech, then follow brand and number rules.
            word_times = kinetic_display_times(element.payload["lines"], kinetic_word_times(
                element.payload["lines"], words, element.start_frame / 30, element.end_frame / 30,
                context.project.word_fixes, element_id=element.id, vocabulary=vocabulary,
                word_text_fixes=context.project.word_text_fixes,
            ), vocabulary)
            word_fix_inputs = {
                "word_fixes": projection_sha256(context.project.word_fixes),
                **({"word_text_fixes": projection_sha256(context.project.word_text_fixes)}
                   if context.project.word_text_fixes else {}),
            }
        if bleep_words is not None:
            lines = bleep.mask_text("\n".join(lines), bleep_words).split("\n")
            accent = bleep.mask_text(accent, bleep_words)
        plan = title_layer_plan(
            "", accent, style,
            lines=lines, frames=element.end_frame - element.start_frame,
            start_frame=element.start_frame,
            word_times=word_times,
        )
        if placement_record is not None:
            clearance = placement_record["clearance"]
            if style.data["hook"]["style"] == "kinetic":
                if "original_scale" in clearance and "scale" in clearance:
                    plan = scale_kinetic_title(plan, clearance["scale"])
                elif "original_size_px" not in clearance:
                    raise ValidationError("graphics.receipt", "kinetic title has no scale evidence")
            elif "original_size_px" not in clearance:
                raise ValidationError("graphics.receipt", "title has no sizing evidence")
            else:
                size = clearance.get("size_px", clearance["original_size_px"])
                if size != clearance["original_size_px"]:
                    data = copy.deepcopy(style.data)
                    data["hook"].update(position="top", size_px=size)
                    plan = title_layer_plan(
                        "", accent, replace(style, data=data),
                        lines=lines, frames=plan.frames,
                        start_frame=plan.start_frame,
                        word_times=word_times,
                    )
            plan = replace(plan, **{key: placement_record[key]
                                    for key in ("x", "y", "width", "height")})
        elif plan.template == "letters":
            clearance = {"original_size_px": plan.variables["size_px"],
                         "size_px": plan.variables["size_px"]}
        else:
            if faces is None:
                faces = picture_faces(context.project_dir, timeline, context)
                face_frames = (face_boxes_per_frame(faces)
                               if faces.status == "measured" else None)
            plan, clearance = place_title(plan, style, faces)
        digest = projection_sha256({
            "element": element.to_dict(), "hook": style.data["hook"],
            **word_fix_inputs,
            **bleep_input,
            "brand:words": projection_sha256(vocabulary),
            "placement": plan.placement(""), "clearance": clearance,
            **({"word_times": plan.variables["word_times"]}
               if "word_times" in plan.variables else {}),
            **({"ink": {name: plan.variables[name] for name in (
                "ink_color", "shadow_opacity", "shadow_blur_px", "shadow_x_px", "shadow_y_px",
            )}, "kinetic_lines": plan.variables["kinetic_lines"], "layer": "front"}
               if style.data["hook"]["style"] == "kinetic" else {}),
            "fonts": assets.font_identity(),
            "template": assets.template_identity(workers=hyperframes.RENDER_WORKERS),
        })
        layer = ("front" if style.data["hook"]["style"] in {"kinetic", "text_hook"}
                 else element.layer)
        display_text = ((json.loads(plan.variables["lines"]) if plan.template == "letters"
                         else [*lines, accent]) if bleep_words is not None else None)
        titles.append(TitleSnapshot(element.id, layer, plan, digest, clearance,
                                    display_text=tuple(display_text) if display_text is not None
                                    else None))
    hook = next((title for title in titles if title.id == TEXT_HOOK_ID
                 and title.plan.variables.get("look") == "text_hook"), None)
    if hook is not None:
        reserved = Box(hook.plan.x, hook.plan.y, hook.plan.width, hook.plan.height)
        margin = face_clearance_px(style.data)
        for item in (*titles, *graphics):
            if item is hook:
                continue
            box = Box(item.plan.x, item.plan.y, item.plan.width, item.plan.height)
            boxes = [box]
            if item.plan.template == "chapter":
                boxes = [chapter_text_box(item.plan)]
            elif item.plan.template == "playing_grid":
                boxes = [Box(tile["x"], tile["y"], tile["width"], tile["height"])
                         for tile in json.loads(item.plan.variables["tiles"])]
            if isinstance(item, GraphicSnapshot) and item.plan.motion:
                box = (Box(0, box.y, box.x1, box.h) if item.plan.motion["side"] == "left"
                       else Box(box.x, box.y, 1080 - box.x, box.h))
                boxes = [box]
            if any(gap_px(reserved, box) < margin for box in boxes):
                raise ValidationError(f"graphic.{item.id}.clearance",
                                      f"{item.id} is too close to the Text hook. "
                                      "Move its box clear of the Text hook.")
        for item in timeline.elements:
            if item.kind == "overlay" and item.payload.get("placement") != "tall_behind":
                if gap_px(reserved, Box(*item.payload["box_px"])) < margin:
                    raise ValidationError(f"overlay.{item.id}.clearance",
                                          f"{item.id} is too close to the Text hook. "
                                          "Move its box clear of the Text hook.")
    inputs = {f"{title.kind}:{title.id}": title.digest for title in titles}
    inputs.update({f"graphic:{graphic.id}": graphic.digest for graphic in graphics})
    camera_plan = context.project_dir / "captions/camera3d.json"
    if style.data["captions"].get("camera_3d", False) and camera_plan.is_file():
        inputs["captions/camera3d.json"] = hashlib.sha256(
            context.read("captions/camera3d.json")).hexdigest()
    if (any(title.kind == "title" for title in titles) or graphics
            or context.project.shape == "horizontal" and titles):
        # Receipt inspection can validate placement dependencies by identity alone.
        # It must not run the detector or probe the cut to reproduce a placement.
        face_inputs = {
            "cut": cut_faces_identity(context.identity.sha256(context.project_dir / "cut/cut.mp4")),
            "frames": timeline.total_frames, "hold": timeline.end_hold_frames,
            "picture": [element.to_dict() for element in timeline.elements
                        if element.kind in {"zoom", "insert", "chapter", "transition"}] + [
                {"id": element.id, "start_frame": element.start_frame,
                 "end_frame": element.end_frame, "tall_framing": element.payload["tall_framing"]}
                for element in timeline.elements if element.kind == "overlay"
                and element.payload.get("placement") == "tall_behind"
            ],
        }
        if graphics:
            try:
                face_inputs["captions"] = context.json("captions/layout.json")
            except FileNotFoundError:
                face_inputs["captions"] = None
        inputs["faces"] = projection_sha256(face_inputs)
    inputs.update(context.taste_inputs(*(["hook"] if titles else []),
                                       *(["graphics"] if graphics or chapters else []),
                                       *(["chapters"] if chapters else [])))
    return GraphicsSnapshot(tuple(titles), tuple(graphics), assets, inputs)


def graphics(project_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Publish title and card layers with box/frame placements through the stage journal."""
    with paths.operation():
        context = ProjectContext(project_dir)
        require_top_text(context)
        root = context.project_dir
        run = StageRun(root, "graphics")
        previous = read_receipt(root / "graphics") or {}
        if not force and previous.get("status") == "verified":
            try:
                saved = graphics_snapshot(context, placements=previous["elements"])
                if run.up_to_date(saved.inputs):
                    return {"status": "verified", "elements": previous["elements"],
                            "receipt": "graphics/receipt.json", "noop": True,
                            "warnings": previous.get("warnings", number_warnings(
                                previous["elements"]))}
            except (ValidationError, KeyError):
                pass
        snapshot = graphics_snapshot(context)
        layers = (*snapshot.titles, *snapshot.graphics)
        elements = [layer.placement() for layer in layers]
        warnings = number_warnings(elements)
        result = {"status": "verified", "elements": elements, "receipt": "graphics/receipt.json",
                  "warnings": warnings}
        if not force and run.up_to_date(snapshot.inputs):
            return {**result, "warnings": previous.get("warnings", warnings), "noop": True}
        owned = {item["path"]: item for item in previous.get("outputs", [])}
        layer_checks = {}
        work_root = paths.validate_output_dir(root / ".yap/work")
        with tempfile.TemporaryDirectory(prefix="graphics-", dir=work_root) as folder, run:
            work = Path(folder)
            for name, digest in snapshot.inputs.items():
                run.record_inputs(name, digest)
            for index, layer in enumerate(layers):
                kind = layer.kind if isinstance(layer, TitleSnapshot) else "graphic"
                relative = f"graphics/{layer.id}.mov"
                target = root / relative
                entry = owned.get(relative)
                staged = work / f"{layer.id}.mov"
                if (not force and previous.get("status") == "verified"
                        and previous.get("inputs", {}).get(f"{kind}:{layer.id}") == layer.digest
                        and entry and target.is_file()
                        and receipt.sha256_file(target) == entry["sha256"]):
                    run.outputs[relative] = entry
                    if layer.id in previous.get("layer_checks", {}):
                        layer_checks[layer.id] = previous["layer_checks"][layer.id]
                    continue
                run.progress(f"rendering {kind} {index + 1} of {len(layers)}", index / len(layers))
                if isinstance(layer, TitleSnapshot):
                    rendered = render_layer_plan(layer.plan, work / layer.id, staged,
                                                 assets=layer.assets or snapshot.assets)
                else:
                    rendered = _render_card(layer, work / layer.id, staged)
                layer_checks[layer.id] = rendered.checks
                run.replace_output(target, rendered.path)
            # Retain ownership of retired layers so an id can be restored without an unsafe write.
            active = {item["path"] for item in elements}
            for relative, entry in owned.items():
                target = root / relative
                if (relative not in active and target.is_file()
                        and receipt.sha256_file(target) == entry["sha256"]):
                    run.outputs[relative] = entry
            run.finish(extra={"elements": elements, "layer_checks": layer_checks,
                              "warnings": warnings,
                              "taste": context.taste_application().pin()})
        return {**result, "noop": False}
