"""Strict card requests, placement plans, and owned standalone artifacts."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from yap_studio import paths
from yap_studio.captions import caption_layout_settings
from yap_studio.card_motion import plan_motion
from yap_studio.card_project import PROJECT_CARD_TEMPLATES, CardProject
from yap_studio.cards import (
    CARD_TEMPLATES,
    PAD_X,
    CardPlan,
    card_kind_off,
    card_kind_off_detail,
    card_layer_plan,
    load_template,
    render_card,
    rows_on_cue,
    template_dir,
    template_html,
    video_slots,
)
from yap_studio.cards_video import caption_ceiling
from yap_studio.errors import InputError, OutputPathError, ValidationError
from yap_studio.faces.clearance import (
    DEFAULT_HAIR,
    Element,
    Layout,
    face_boxes_per_frame,
    face_clearance_px,
    gap_px,
    head_box,
)
from yap_studio.faces.mapping import behind_face_windows
from yap_studio.faces.model import Box, FacesData
from yap_studio.faces.placement import Band, suggest_offset, top_band
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.hf_registry import catalog
from yap_studio.matte.composite import CARDS_SCHEMA, CardOverlay
from yap_studio.plan import snap_frame
from yap_studio.planner.anchors import resolve_anchor_time
from yap_studio.planner.number_behind import provisional_number
from yap_studio.project.model import (
    RULE_DEFAULTS,
    Anchor,
    CutWords,
    ProjectFileError,
    parse_anchor,
    parse_document,
)
from yap_studio.receipt import sha256_file
from yap_studio.render import fonts, hyperframes
from yap_studio.render.hyperframes import LayerResult, RendererInstall
from yap_studio.style import ResolvedStyle

if TYPE_CHECKING:
    from yap_studio.stages.timeline import ResolvedTimeline

REQUESTS_SCHEMA = "yap-studio/card-requests@1"
PLAN_SCHEMA = "yap-studio/graphics-plan@1"
PENDING_SCHEMA = "yap-studio/stage-pending@1"
_DOCUMENTS = ("agent-lane.json", "layout.json", "cards.json")
Placement = Literal["top", "behind_head"]


def _object(data: Any, location: str) -> dict[str, Any]:
    if not isinstance(data, dict) or any(not isinstance(key, str) for key in data):
        raise ValidationError(location, "expected an object with string keys")
    return data


def _unknown(data: dict[str, Any], keys: set[str], location: str) -> None:
    for key in data:
        if key not in keys:
            raise ValidationError(f"{location}.{key}", "unknown field")


def _card_id(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(location, "expected a non-empty string")
    if (
        value in {".", ".."}
        or any(c in value for c in "/\\")
        or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)
    ):
        raise ValidationError(location, "card id must be a safe layer filename")
    return value


@dataclass(frozen=True)
class CardRequest:
    """A validated request retaining the creator's word anchors."""

    id: str
    template: str
    slots: dict[str, Any]
    start: Anchor
    end: Anchor | None = None
    hold_s: float | None = None
    placement: Placement = "top"

    def to_dict(self) -> dict[str, Any]:
        """Return the request shape for validation at the planning boundary."""
        data = {
            "id": self.id,
            "template": self.template,
            "slots": copy.deepcopy(self.slots),
            "start": self.start.to_dict(),
            "placement": self.placement,
        }
        if self.end is not None:
            data["end"] = self.end.to_dict()
        if self.hold_s is not None:
            data["hold_s"] = self.hold_s
        return data


def parse_requests(data: Any) -> tuple[CardRequest, ...]:
    """Parse requests strictly, using the timeline contract for anchor validation."""
    obj = _object(data, "requests")
    for key in obj:
        if key not in {"schema", "cards"}:
            raise ValidationError(f"requests.{key}", "unknown field; put cards in a cards list")
    if obj.get("schema") != REQUESTS_SCHEMA:
        raise ValidationError("requests.schema", f"expected {REQUESTS_SCHEMA}")
    if not isinstance(obj.get("cards"), list):
        raise ValidationError("cards", "expected a list")
    result = []
    seen: set[str] = set()
    for index, raw in enumerate(obj["cards"]):
        location = f"cards[{index}]"
        card = _object(raw, location)
        _unknown(card, {"id", "template", "slots", "start", "end", "hold_s", "placement"}, location)
        card_id = _card_id(card.get("id"), f"{location}.id")
        if card_id in seen:
            raise ValidationError(f"{location}.id", f"duplicate id {card_id!r}")
        seen.add(card_id)
        template = card.get("template")
        allowed = (*CARD_TEMPLATES, *registry_cards.kinds())
        if not isinstance(template, str) or template not in allowed:
            raise ValidationError(f"{location}.template", "choose " + ", ".join(allowed))
        slots = _object(card.get("slots"), f"{location}.slots")
        placement = card.get("placement", "top")
        if placement not in ("top", "behind_head"):
            raise ValidationError(f"{location}.placement", "expected top or behind_head")
        if "end" in card and "hold_s" in card:
            raise ValidationError(location, "use end or hold_s, not both")
        element = {
            "id": card_id,
            "kind": "graphic",
            "template": template,
            "slots": slots,
            "start": card.get("start"),
            **({"end": card["end"]} if "end" in card else {"hold_s": card.get("hold_s", 1)}),
        }
        try:
            lane = parse_document(
                "timeline",
                {
                    "schema": "yap-studio/timeline@1",
                    "words_sha256": {},
                    "elements": [element],
                },
                lane="agent",
            )
        except ProjectFileError as exc:
            field = exc.location.replace("elements[0]", location, 1)
            raise ValidationError(field, exc.detail) from exc
        parsed = lane.elements[0]
        result.append(
            CardRequest(
                card_id,
                template,
                copy.deepcopy(slots),
                parsed.start,
                parsed.end,
                parsed.hold_s if "hold_s" in card else None,
                placement,
            )
        )
    return tuple(result)


def load_requests(path: str | Path) -> tuple[CardRequest, ...]:
    """Load a request file and reserve it against output writes."""
    source = paths.validate_input(path)
    try:
        data = json.loads(source.read_bytes())
    except OSError as exc:
        raise InputError(str(path), f"cannot read requests JSON: {exc}") from exc
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError(str(path), f"invalid requests JSON: {exc}") from exc
    return parse_requests(data)


@dataclass(frozen=True)
class PlannedCard:
    """One timed layer box with explicit face-clearance evidence."""

    id: str
    template: str
    slots: dict
    placement: Placement
    start: Anchor
    end: Anchor | None
    hold_s: float | None
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    x: int
    y: int
    width: int
    height: int
    clearance: dict
    render_key: str
    render_slots: dict
    cue_anchors: tuple[tuple[int, Anchor], ...] = ()
    drop_anchors: tuple[tuple[int, Anchor], ...] = ()

    def layer(self) -> Literal["front", "behind"]:
        """Select the timeline layer corresponding to the placement."""
        return "front" if self.placement == "top" else "behind"

    def lane_slots(self) -> dict:
        """Export absolute landing anchors for the timeline's output clock."""
        slots = copy.deepcopy(self.slots)
        for index, anchor in self.drop_anchors:
            slots["drops"][index]["at"] = anchor.to_dict()
        for index, anchor in self.cue_anchors:
            slots["cues"][index] = {"at": anchor.to_dict()}
        return slots

    def to_dict(self) -> dict[str, Any]:
        """Return all placement fields with portable anchors."""
        data = asdict(self)
        data.pop("render_key")
        data.pop("render_slots")
        data.pop("cue_anchors")
        data.pop("drop_anchors")
        return {
            **data,
            "start": self.start.to_dict(),
            "end": self.end.to_dict() if self.end is not None else None,
            "layer": {
                "path": f"layers/{self.id}.mov",
                "render_key": self.render_key,
                "sha256": None,
                "status": "missing",
            },
        }


@dataclass(frozen=True)
class GraphicsPlan:
    """A standalone graphics lane and the geometry needed by later commands."""

    status: Literal["ok", "needs_review"]
    cards: tuple[PlannedCard, ...]
    warnings: tuple[str, ...]
    fps: int
    frames: int
    words_sha256: dict[str, str]
    style: str
    project_dir: Path | None = None

    def lane(self) -> dict:
        """Return only the timeline lane contract, without invoking its stage."""
        return {
            "schema": "yap-studio/timeline@1",
            "words_sha256": dict(self.words_sha256),
            "elements": [
                {
                    "id": card.id,
                    "kind": "graphic",
                    "lane": "agent",
                    "template": card.template,
                    "slots": card.lane_slots(),
                    "start": card.start.to_dict(),
                    "layer": card.layer(),
                    **({} if card.template in {"playing_grid", "reel_feed"} else {
                        "box": {"x": card.x, "y": card.y,
                                "width": card.width, "height": card.height}}),
                    **(
                        {"end": card.end.to_dict()}
                        if card.end is not None
                        else {"hold_s": card.hold_s}
                    ),
                }
                for card in self.cards
            ],
        }

    def to_dict(self) -> dict:
        """Return the graphics plan with resolved timing and style identity."""
        return {
            "schema": PLAN_SCHEMA,
            "status": self.status,
            "fps": self.fps,
            "frames": self.frames,
            "style": self.style,
            "words_sha256": dict(self.words_sha256),
            "warnings": list(self.warnings),
            "cards": [card.to_dict() for card in self.cards],
        }

    def layout(self) -> Layout:
        """Retain all geometry, including assembly's behind-card face windows."""
        elements = []
        used = {card.id for card in self.cards}
        for card in self.cards:
            boxes = ([Box.from_dict(tile["box"]) for tile in card.clearance["tiles"]]
                     if card.template == "playing_grid" else
                     [Box(card.x, card.y, card.width, card.height)])
            for index, box in enumerate(boxes):
                name = card.id
                if index:
                    number = 2
                    while f"{card.id}#{number}" in used:
                        number += 1
                    name = f"{card.id}#{number}"
                    used.add(name)
                elements.append(Element(
                    name, "background" if card.template == "reel_feed" else "card", box,
                    card.start_frame, card.end_frame,
                    () if card.placement == "behind_head" else None,
                ))
        return Layout(
            1080,
            1920,
            tuple(elements),
            (),
            clearance_px=self.cards[0].clearance["margin_px"] if self.cards else None,
            hair=DEFAULT_HAIR,
        )

    def overlays(self, layers_dir: str = "layers") -> tuple[CardOverlay, ...]:
        """Return relative alpha layer paths for ffmpeg compositing."""
        return tuple(
            CardOverlay(
                card.id,
                Path(layers_dir) / f"{card.id}.mov",
                0 if card.clearance.get("card_motion", {}).get("status") == "streak" else card.x,
                card.y,
                1080 if card.clearance.get("card_motion", {}).get("status") == "streak"
                else card.width,
                card.height,
                card.start_frame,
                card.end_frame,
                "front" if card.placement == "top" else "behind_head",
            )
            for card in self.cards
        )


def _render_key(layer: CardPlan) -> str:
    """Hash pixel inputs, using asset contents rather than their source paths."""
    data = asdict(layer)
    if layer.motion is None:
        data.pop("motion", None)
    else:
        from yap_studio import card_motion

        data["motion_source"] = sha256_file(Path(card_motion.__file__))
        data["motion_scripts"] = {name: sha256_file(card_motion._SCRIPTS / name) for name in (
            "card-motion-host.js", "card-motion-drawn.js")}
        data["motion_helper"] = catalog.item("echo-smear")
    if layer.registry is None:
        data.pop("registry", None)
    else:
        data["catalog"] = catalog.card_catalog()
        data["adapter_sha256"] = sha256_file(Path(registry_cards.__file__))
    data["browser_gpu"] = hyperframes.CARD_BROWSER_GPU
    data.pop("poster", None)
    if layer.poster is not None:
        data["poster"] = {"sha256": sha256_file(layer.poster.source),
                          "time_s": layer.poster.time_s}
    if not data.get("windows"):
        data.pop("windows", None)
    data["assets"] = [
        {"name": name, "sha256": sha256_file(asset)} for name, asset in layer.assets
    ]
    data["bundled_fonts"] = [
        {"file": filename, "sha256": sha256_file(source)}
        for filename, source in fonts.font_files(layer.fonts)
    ]
    root = template_dir(layer.template)
    template_hash = hashlib.sha256()
    for source in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if source.is_file():
            name = source.relative_to(root).as_posix().encode("utf-8")
            content = paths.validate_input(source).read_bytes()
            template_hash.update(len(name).to_bytes(8, "big"))
            template_hash.update(name)
            template_hash.update(len(content).to_bytes(8, "big"))
            template_hash.update(content)
    data["template_sha256"] = template_hash.hexdigest()
    if layer.template in {"choices", "specs_list", "stat"}:
        data["expanded_template_sha256"] = hashlib.sha256(
            template_html(layer.template).encode("utf-8")).hexdigest()
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def safe_card_order(
    template: str, style: ResolvedStyle, slots: Mapping[str, object],
) -> tuple[str, ...] | None:
    """Select readable placement for the card looks that opt in."""
    if template == "choices":
        return ("side", "headroom", "lower")
    if template == "flow" and style.data["graphics"].get("flow_look", "boxes") == "tags":
        return ("headroom", "side", "lower")
    if rows_on_cue(style, template, slots):
        return ("headroom", "side", "lower")
    return None


def fitted_card(
    template: str, slots: dict, style: ResolvedStyle, *, frames: int, fps: int = 30,
    width: int | None = None, height: int | None = None,
    project: CardProject | None = None,
) -> CardPlan:
    """Refit labels within placement limits without changing the creator's style."""
    if safe_card_order(template, style, slots) is not None:
        return card_layer_plan(template, slots, style, frames=frames, fps=fps, project=project,
                               max_width_px=width, max_height_px=height)
    if width is not None and template not in {"playing_grid", "reel_feed"}:
        data = copy.deepcopy(style.data)
        data["graphics"]["card_width_px"] = width - 2 * PAD_X
        style = replace(style, data=data)
    return card_layer_plan(template, slots, style, frames=frames, fps=fps, project=project,
                           max_height_px=height)


def _place_safe_card(
    layer: CardPlan, slots: dict, style: ResolvedStyle, heads: set[Box], *,
    order: tuple[str, ...], ceiling: int, margin: int, evidence: dict,
    preserve_box: bool, requested_layer: str, project: CardProject | None,
) -> tuple[CardPlan, Placement, dict]:
    """Fit each position at a readable size before moving to the next position."""
    safe_top = style.data["internal"]["key_text_top_px"]

    def gap(candidate: CardPlan) -> float | None:
        box = Box(candidate.x, candidate.y, candidate.width, candidate.height)
        return min((gap_px(box, head) for head in heads), default=None)

    def clear(candidate: CardPlan, position: str) -> tuple[CardPlan, Placement, dict]:
        evidence.update(status="clear" if heads else "unchecked", position=position,
                        gap_px=gap(candidate), reason=f"selected the safe {position} position")
        return candidate, "top", evidence

    if preserve_box:
        if (requested_layer == "front"
                and layer.y + layer.variables.get("glow_pad_px", 0) >= safe_top
                and layer.y + layer.height <= ceiling
                and (not heads or gap(layer) >= margin)):
            return clear(layer, "supplied")
    else:
        positions = order if heads else ("headroom",)
        for position in positions:
            if position == "side":
                spans = [(0, math.floor(min(h.x for h in heads) - margin)),
                         (math.ceil(max(h.x1 for h in heads) + margin), 1080)]
                spans.sort(key=lambda span: -(span[1] - span[0]))
            else:
                spans = [(0, 1080)]
            for left, right in spans:
                left = math.ceil(max(0, left) / 2) * 2
                width = math.floor((min(1080, right) - left) / 4) * 4
                if width <= 0:
                    continue
                bottom = min(ceiling, math.floor(min(h.y for h in heads) - margin)
                             if position == "headroom" and heads else ceiling)
                minimum_top = (math.ceil(max(h.y1 for h in heads) + margin)
                               if position == "lower" else 0)
                candidate = layer
                while True:
                    top = math.ceil(max(minimum_top, safe_top
                                         - candidate.variables.get("glow_pad_px", 0)) / 2) * 2
                    height = bottom - top
                    if height <= 0:
                        candidate = None
                        break
                    try:
                        fitted = fitted_card(layer.template, slots, style, frames=layer.frames,
                                             fps=layer.fps, width=width, height=height,
                                             project=project)
                    except ValidationError:
                        candidate = None
                        break
                    if (fitted.variables.get("glow_pad_px", 0)
                            == candidate.variables.get("glow_pad_px", 0)):
                        candidate = fitted
                        break
                    candidate = fitted
                if candidate is None:
                    continue
                if position == "side":
                    x = left
                    preferred = min(h.y for h in heads)
                else:
                    centre = ((min(h.x for h in heads) + max(h.x1 for h in heads)) / 2
                              if heads else 540)
                    x = round(max(0, min(1080 - candidate.width,
                                        centre - candidate.width / 2)) / 2) * 2
                    preferred = (top + bottom - candidate.height) / 2 if heads else top
                y = round(max(top, min(bottom - candidate.height, preferred)) / 2) * 2
                y = min(y, math.floor((bottom - candidate.height) / 2) * 2)
                candidate = replace(candidate, x=x, y=y)
                if y >= top and (not heads or gap(candidate) >= margin):
                    return clear(candidate, position)
    if layer.template == "flow":
        noun, fix = "tags", "use shorter tags"
    elif rows_on_cue(style, layer.template, slots):
        noun, fix = "rows", "use fewer or shorter rows"
    else:
        noun, fix = "options", "use fewer or shorter options"
    evidence.update(status="blocked", gap_px=gap(layer), reason=(
        f"no room above or beside the head for the {noun} at a readable size; "
        f"{fix} or move the card to another moment with more room above the head"))
    return layer, "top", evidence


def _resolved_slots(template: str, slots: dict, project_dir: Path | None) -> dict:
    """Resolve project video sources for validation without changing the exported slots."""
    slots = copy.deepcopy(slots)
    if project_dir is None:
        return slots
    for name, rule in video_slots(template).items():
        if name not in slots:
            continue
        values = slots[name] if rule["type"] == "video_list" else [slots[name]]
        if not isinstance(values, list):
            continue
        resolved = [
            {**value, "path": str(project_dir / value["path"])}
            if isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"]
            else value for value in values
        ]
        slots[name] = resolved if rule["type"] == "video_list" else resolved[0]
    return slots


def card_caption_top(style: ResolvedStyle, start_frame: int, end_frame: int,
                     layout: dict | None = None) -> int:
    """Use timed camera glyphs or the flat caption band as the card ceiling."""
    from yap_studio.stages.captions import caption_band_geometry

    top = 1920 if style.data["captions"].get("camera_3d", False) else caption_band_geometry(
        caption_layout_settings(style.data["captions"], style.data["brand"]),
        safe_bottom_px=style.data["internal"]["caption_safe_bottom_px"],
    ).y
    if layout and layout.get("measured"):
        top = min([top, *[
            box["y"] for phrase in layout["captions"]
            if phrase["start_frame"] < end_frame and phrase["end_frame"] > start_frame
            for box in phrase["boxes"]
            if box.get("start_frame", phrase["start_frame"]) < end_frame
            and box.get("end_frame", phrase["end_frame"]) > start_frame
        ]])
    return int(top)


def _place_card(
    layer: CardPlan, slots: dict, style: ResolvedStyle,
    faces: Sequence[Sequence[Box]] | None, *,
    reason: str = "faces not supplied",
    face_windows: tuple[tuple[int, int], ...] = (),
    caption_top: int | None = None,
    preserve_box: bool = False,
    requested_layer: Literal["front", "behind"] = "front",
    id: str = "card",
    start_frame: int = 0,
    project: CardProject | None = None,
    refit_sides: bool = True,
    safe_order: tuple[str, ...] | None = None,
) -> tuple[CardPlan, Placement, dict]:
    """Prefer a clear top, side or lower box; exempt only a readable behind card.

    Reserve the entire animated layer as the glyph area for the fallback. This
    conservative bound proves readability without a renderer or a subject mask.
    A supplied box is retained on its requested layer or blocked; every candidate
    respects the caption ceiling. An opt-in order fits readable text within safe
    headroom, side corridors and the lower band.
    """
    margin = face_clearance_px(style.data)
    evidence = {"status": "unchecked", "reason": reason, "margin_px": margin,
                "gap_px": None, "hair": DEFAULT_HAIR.to_dict()}
    ceiling = caption_ceiling(style, caption_top)
    if layer.template == "playing_grid":
        from yap_studio.cards import grid_card_plan

        plan, evidence = grid_card_plan(slots, style, faces, frames=layer.frames, fps=layer.fps,
                                        ceiling=ceiling, layer=requested_layer, reason=reason,
                                        id=id, face_windows=face_windows, start_frame=start_frame)
        return plan, "behind_head" if requested_layer == "behind" else "top", evidence
    if layer.template == "reel_feed":
        plan = card_layer_plan("reel_feed", slots, style, frames=layer.frames, fps=layer.fps,
                               ceiling=ceiling)
        evidence.update(status="full_frame",
                        reason="the feed fills the frame and shows the video inside the phone",
                        ceiling_px=ceiling)
        return plan, "top", evidence
    heads = {head_box(face) for frame in faces or () for face in frame}
    if safe_order is not None:
        return _place_safe_card(layer, slots, style, heads, order=safe_order, ceiling=ceiling,
                                margin=margin, evidence=evidence, preserve_box=preserve_box,
                                requested_layer=requested_layer, project=project)
    top = top_band(style.data)

    def gap(candidate: CardPlan) -> float | None:
        return min((gap_px(Box(candidate.x, candidate.y, candidate.width, candidate.height), head)
                    for head in heads), default=None)

    def behind_head_clear(candidate: CardPlan) -> bool:
        # No glyph may be hidden, even during entry animations. Assembly skips
        # restoring the keyed subject during zooms and picture replacements.
        return bool(heads) and not face_windows and (
            top.top <= candidate.y
            and candidate.y + candidate.height <= min(top.bottom, ceiling)
            and all(candidate.y + candidate.height <= head.y for head in heads)
        )

    if preserve_box:
        distance = gap(layer)
        if requested_layer == "behind" and behind_head_clear(layer):
            evidence.update(status="behind_head", gap_px=distance,
                            glyph_box=Box(layer.x, layer.y, layer.width, layer.height).to_dict(),
                            reason="keep the supplied box behind the head with the entire "
                                   "animated glyph area outside the subject head box")
            return layer, "behind_head", evidence
        if (requested_layer == "behind" or layer.y + layer.height > ceiling
                or (distance is not None and distance < margin)):
            evidence.update(status="blocked", gap_px=distance,
                            reason="supplied box violates head or caption clearance; "
                                   "move its box or plan the card again")
        else:
            evidence.update(status="clear" if faces is not None else "unchecked",
                            gap_px=distance, reason="keep the supplied box; " + (
                                "clear of heads and captions" if faces is not None else reason))
        return layer, "top", evidence
    if not heads and layer.y + layer.height <= ceiling:
        if faces is not None:
            evidence.update(status="clear",
                            reason="no face in the window; keep the preferred position")
        return layer, "top", evidence

    def candidate_in_band(candidate: CardPlan, band: Band, preferred: int) -> CardPlan | None:
        if layer.motion:
            minimum_top = 280 - candidate.variables.get("glow_pad_px", 0)
            if band.bottom <= minimum_top:
                return None
            band = Band(max(minimum_top, band.top), band.bottom)
            preferred = max(minimum_top, preferred)
        hint = suggest_offset(faces or (), x=candidate.x, width=candidate.width,
                              height=candidate.height, band=band, margin_px=margin,
                              preferred_y=preferred)
        if hint.y is None:
            return None
        even = [y for y in {hint.y // 2 * 2, (hint.y + 1) // 2 * 2}
                if band.top <= y <= band.bottom - candidate.height
                and (not heads or gap(replace(candidate, y=y)) >= margin)]
        if not even:
            return None
        y = min(even, key=lambda y: (abs(y - hint.y), y))
        evidence["reason"] = hint.reason
        if y != hint.y:
            evidence["reason"] += (
                f"; adjusted y {hint.y} to {y} for even composite coordinates"
            )
        return replace(candidate, y=y)

    def side_candidate() -> CardPlan | None:
        if ceiling <= 0:
            return None
        corridors = [(0, math.floor(min(head.x for head in heads) - margin)),
                     (math.ceil(max(head.x1 for head in heads) + margin), 1080)]
        spans = sorted(corridors, key=lambda span: span[0])
        for left, right in spans:
            x = math.ceil(max(0, left) / 2) * 2
            width = min(layer.width, math.floor((min(1080, right) - x) / 4) * 4)
            if width < 200 + 2 * PAD_X:
                continue
            try:
                side = fitted_card(layer.template, slots, style, frames=layer.frames,
                                   fps=layer.fps, width=width, project=project)
            except ValidationError:
                continue
            side = replace(side, x=x)
            selected = candidate_in_band(side, Band(0, ceiling), layer.y)
            if selected is not None:
                return selected
        return None

    selected = None
    position = "top"
    top_bottom = min(top.bottom, ceiling)
    if top_bottom > top.top:
        selected = candidate_in_band(layer, Band(top.top, top_bottom), top.top)
    if selected is None and heads and layer.template in {"steps", "text_tag"}:
        height = min(top_bottom, math.floor(min(head.y for head in heads) - margin)) - top.top
        if height > 0:
            try:
                compact = fitted_card(layer.template, slots, style, frames=layer.frames,
                                      fps=layer.fps, height=height, project=project)
            except ValidationError:
                pass
            else:
                selected = candidate_in_band(compact, Band(top.top, top_bottom), top.top)
    if selected is None and heads and ceiling >= layer.height and refit_sides:
        selected = side_candidate()
        if selected is not None:
            position = "side"
    if selected is None and ceiling >= layer.height:
        selected = candidate_in_band(layer, Band(0, ceiling), top.bottom)
        position = "lower"
    if selected is not None:
        evidence.update(status="clear" if faces is not None else "unchecked",
                        gap_px=gap(selected), position=position)
        evidence["reason"] += f"; selected the {position} position"
        if faces is None:
            evidence["reason"] += f"; faces unmeasured: {reason}"
        return selected, "top", evidence

    for y in range(top.top, min(top.bottom, ceiling) - layer.height + 1, 2):
        candidate = replace(layer, y=y)
        if behind_head_clear(candidate):
            evidence.update(status="behind_head", gap_px=gap(candidate),
                            glyph_box=Box(candidate.x, y, candidate.width,
                                          candidate.height).to_dict(),
                            reason="no clear position; behind the head with the entire "
                                   "animated glyph area outside the subject head box")
            return candidate, "behind_head", evidence
    hint = suggest_offset(faces or (), x=layer.x, width=layer.width, height=layer.height,
                          band=top, margin_px=margin, preferred_y=top.top)
    y = (hint.best_y or 0) // 2 * 2
    blocked = replace(layer, y=y)
    evidence.update(status="blocked", gap_px=gap(blocked),
                    reason="no clear top, readable side or lower position within caption limits; "
                           "behind_head would hide glyphs or lose its face exemption")
    return blocked, "top", evidence


def place_card(
    layer: CardPlan, slots: dict, style: ResolvedStyle,
    faces: Sequence[Sequence[Box]] | None, *, ordinary_layer: CardPlan | None = None,
    **options: Any,
) -> tuple[CardPlan, Placement, dict]:
    """Use ordinary placement if the requested horizontal motion cannot fly."""
    original = layer
    layer, placement, evidence = _place_card(layer, slots, style, faces, **options)
    blocked = evidence["status"] == "blocked"
    layer, evidence = plan_motion(layer, style, faces, evidence)
    if original.motion and layer.motion is None and "card_motion" in evidence:
        motion = evidence["card_motion"]
        if blocked:
            motion = {"status": "none", "reason":
                "no position below the 280 px line keeps the head clear for the fly"}
        fallback = replace(ordinary_layer or original, motion=None)
        layer, placement, evidence = _place_card(fallback, slots, style, faces, **options)
        layer = replace(layer, motion=None)
        evidence["card_motion"] = motion
    return layer, placement, evidence


def place_video_card(
    plan: CardPlan, slots: dict, style: ResolvedStyle,
    faces: Sequence[Sequence[Box]] | None, **options: Any,
) -> tuple[CardPlan, Placement, dict]:
    """Try progressively narrower centred video panels until one clears the head."""
    options.setdefault("caption_top", card_caption_top(style, 0, plan.frames))
    options["refit_sides"] = False
    widths = list(range(style.data["graphics"]["card_width_px"], 399, -80))
    if not widths:
        widths = [style.data["graphics"]["card_width_px"]]
    elif widths[-1] != 400:
        widths.append(400)
    for width in widths:
        candidate = fitted_card(plan.template, slots, style, frames=plan.frames, fps=plan.fps,
                                width=width + 2 * PAD_X)
        candidate = replace(candidate, x=(1080 - candidate.width) // 2, y=plan.y)
        result, placement, clearance = place_card(candidate, slots, style, faces, **options)
        if clearance["status"] != "blocked":
            return result, placement, clearance
    clearance = {**clearance, "reason": "move the card to a moment where the face sits lower, "
                 "give it a box, or use layer: behind"}
    return result, placement, clearance


def plan_cards(
    requests: dict | Sequence[CardRequest | dict],
    cut_words: CutWords,
    style: ResolvedStyle,
    *,
    frames: int,
    fps: int = 30,
    faces: FacesData | None = None,
    timeline: ResolvedTimeline | None = None,
    project_dir: Path | None = None,
    caption_layout: dict | None = None,
) -> GraphicsPlan:
    """Resolve card requests on the cut clock, size them, and clear the head."""
    for key, value in (("frames", frames), ("fps", fps)):
        if type(value) is not int or value < 1:
            raise ValidationError(key, "expected a positive integer")
    if cut_words.time_base != "output":
        raise ValidationError("cut_words.time_base", "expected output time")
    if project_dir is not None:
        project_dir = Path(project_dir).resolve()
    if not isinstance(requests, dict):
        requests = {
            "schema": REQUESTS_SCHEMA,
            "cards": [r.to_dict() if isinstance(r, CardRequest) else r for r in requests],
        }
    requests = parse_requests(requests)
    if faces is not None:
        face_count = frames + (timeline.end_hold_frames if timeline else 0)
        for key, expected in (("frames", face_count), ("fps", fps),
                              ("width", 1080), ("height", 1920)):
            if getattr(faces.video, key) != expected:
                raise ValidationError(
                    f"faces.video.{key}", f"must describe the same cut: {expected}"
                )
    face_frames = face_boxes_per_frame(faces) if faces and faces.status == "measured" else None
    planned = []
    warnings = []
    for index, request in enumerate(requests):
        location = f"cards[{index}]"
        if card_kind_off(style, request.template):
            raise ValidationError(
                f"{location}.template",
                card_kind_off_detail(style, request.template),
            )
        start_time = resolve_anchor_time(request.start, cut_words, start=True,
                                         location=f"{location}.start")
        start_frame = snap_frame(start_time, Fraction(fps))
        hold_s = request.hold_s
        if request.end is None:
            hold_s = hold_s if hold_s is not None else style.data["graphics"]["hold_s"]
            hold_time = Fraction(str(hold_s))
            end_time = start_time + hold_time
            end_frame = start_frame + snap_frame(hold_time, Fraction(fps))
        else:
            end_time = resolve_anchor_time(request.end, cut_words, start=False,
                                           location=f"{location}.end")
            end_frame = snap_frame(end_time, Fraction(fps))
        start_s, end_s = float(start_time), float(end_time)
        if not all(math.isfinite(t) for t in (start_s, end_s)):
            raise ValidationError(location, "card times must be finite")
        if not 0 <= start_frame < end_frame <= frames:
            raise ValidationError(
                location,
                f"window [{start_frame}, {end_frame}) must be at least one frame "
                f"and inside [0, {frames})",
            )
        for previous in planned:
            first, last = max(start_frame, previous.start_frame), min(end_frame, previous.end_frame)
            if first < last:
                raise ValidationError(
                    location,
                    f"cards {previous.id!r} and {request.id!r} overlap "
                    f"in frames [{first}, {last}); only one graphic at a time",
                )
        planned_slots = copy.deepcopy(request.slots)
        cue_anchors = []
        cue_rule = load_template(request.template)["slots"].get("cues", {})
        row_cues_off = (request.template in {"specs_list", "stat"}
                        and style.data["graphics"].get("specs_rows", "together") == "together")
        if (not row_cues_off and cue_rule.get("type") == "number_list"
                and isinstance(planned_slots.get("cues"), list)):
            for j, value in enumerate(planned_slots["cues"]):
                if not isinstance(value, dict):
                    continue
                cue_location = f"{location}.slots.cues[{j}]"
                try:
                    anchor = parse_anchor(value["at"] if set(value) == {"at"} else value)
                except ProjectFileError as exc:
                    field = exc.location.replace("anchor", cue_location, 1)
                    raise ValidationError(field, exc.detail) from exc
                cue_time = resolve_anchor_time(anchor, cut_words, start=True, location=cue_location)
                cue_frame = snap_frame(cue_time, Fraction(fps))
                if not (start_time <= cue_time < end_time and start_frame <= cue_frame < end_frame):
                    raise ValidationError(cue_location, "choose a word inside the card window "
                                          "or move the card start or end")
                planned_slots["cues"][j] = round((cue_frame - start_frame) / fps, 6)
                cue_anchors.append((j, anchor))

        drop_anchors = []
        if request.template == "gravity_drop" and isinstance(planned_slots.get("drops"), list):
            for j, drop in enumerate(planned_slots["drops"]):
                if not isinstance(drop, dict):
                    continue
                at = drop.get("at")
                if type(at) not in (int, float) or not math.isfinite(at):
                    continue
                drop_location = f"{location}.slots.drops[{j}].at"
                drop_time = start_time + Fraction(str(at))
                drop_frame = snap_frame(drop_time, Fraction(fps))
                if not (start_time <= drop_time < end_time
                        and start_frame <= drop_frame < end_frame):
                    raise ValidationError(drop_location, "choose a landing inside the card window "
                                          "or move the card start or end")
                drop["at"] = (drop_frame - start_frame) / fps
                drop_anchors.append((j, Anchor(time=float(drop_time))))

        def relative_times(value: Any, field: str, *, first: int = start_frame) -> Any:
            if isinstance(value, list):
                return [relative_times(child, f"{field}[{i}]") for i, child in enumerate(value)]
            if isinstance(value, dict):
                result = {}
                for key, child in value.items():
                    here = f"{field}.{key}"
                    if key == "at" and isinstance(child, dict):
                        anchor = parse_anchor(child)
                        time = resolve_anchor_time(anchor, cut_words, start=True, location=here)
                        result[key] = (snap_frame(time, Fraction(fps)) - first) / fps
                    else:
                        result[key] = relative_times(child, here)
                return result
            return value

        render_slots = relative_times(planned_slots, f"{location}.slots")
        scene = request.template in {"playing_grid", "reel_feed"}
        caption_top = (card_caption_top(style, start_frame, end_frame, caption_layout)
                       if scene or request.template == "video_card"
                       or safe_card_order(request.template, style, planned_slots) is not None
                       else None)
        ceiling = caption_ceiling(style, caption_top) if scene else None
        try:
            slots = _resolved_slots(request.template, render_slots, project_dir)
            if request.template == "playing_grid":
                from yap_studio.cards import grid_card_plan

                layer, _ = grid_card_plan(
                    slots, style,
                    face_frames[start_frame:end_frame] if face_frames is not None else None,
                    frames=end_frame - start_frame, fps=fps, id=request.id,
                    reason=faces.reason if faces else "faces not supplied",
                    layer="behind" if request.placement == "behind_head" else "front",
                    face_windows=behind_face_windows(timeline, start_frame, end_frame)
                    if timeline else (), start_frame=start_frame,
                    ceiling=ceiling,
                )
            else:
                layer = card_layer_plan(
                    request.template, slots, style, frames=end_frame - start_frame, fps=fps,
                    ceiling=ceiling,
                )
        except ValidationError as exc:
            raise ValidationError(f"{location}.{exc.parameter} ({request.id})", exc.detail) from exc
        duration = (end_frame - start_frame) / fps
        if duration > RULE_DEFAULTS["graphic_hold_violation_s"]:
            raise ValidationError(
                location,
                f"{request.id}: hold {duration:g} s exceeds "
                f"{RULE_DEFAULTS['graphic_hold_violation_s']:g} s",
            )
        if duration < RULE_DEFAULTS["graphic_hold_min_s"]:
            warnings.append(
                f"{request.id}: hold {duration:g} s is shorter than "
                f"{RULE_DEFAULTS['graphic_hold_min_s']:g} s"
            )
        elif (duration > RULE_DEFAULTS["graphic_hold_max_s"]
              and not rows_on_cue(style, request.template, slots)):
            warnings.append(
                f"{request.id}: hold {duration:g} s is longer than "
                f"{RULE_DEFAULTS['graphic_hold_max_s']:g} s"
            )
        windows = behind_face_windows(timeline, start_frame, end_frame) if timeline else ()
        if request.template == "big_number":
            layer, placement, clearance = provisional_number(
                layer, margin_px=face_clearance_px(style.data),
            )
        else:
            placer = place_video_card if request.template == "video_card" else place_card
            safe_order = safe_card_order(request.template, style, slots)
            ordinary = (layer if placer is place_video_card or safe_order is not None
                        else replace(layer, y=0))
            layer, placement, clearance = placer(
                layer if layer.motion else ordinary, slots, style,
                face_frames[start_frame:end_frame] if face_frames is not None else None,
                reason=faces.reason if faces else "faces not supplied",
                face_windows=windows,
                caption_top=caption_top,
                **({"ordinary_layer": ordinary} if placer is place_card and layer.motion else {}),
                **({"id": request.id, "start_frame": start_frame, "requested_layer": (
                    "behind" if request.placement == "behind_head" else "front")}
                   if request.template == "playing_grid" else {}),
                **({"safe_order": safe_order} if safe_order is not None else {}),
            )
        if "card_motion" not in clearance:
            layer, clearance = plan_motion(
                layer, style,
                face_frames[start_frame:end_frame] if face_frames is not None else None,
                clearance)
        planned.append(
            PlannedCard(
                request.id,
                request.template,
                planned_slots,
                placement,
                request.start,
                request.end,
                hold_s,
                start_s,
                end_s,
                start_frame,
                end_frame,
                layer.x,
                layer.y,
                layer.width,
                layer.height,
                clearance,
                _render_key(layer),
                render_slots,
                tuple(cue_anchors),
                tuple(drop_anchors),
            )
        )
    return GraphicsPlan(
        "needs_review" if any(c.clearance["status"] == "blocked" for c in planned) else "ok",
        tuple(planned),
        tuple(warnings),
        fps,
        frames,
        dict(cut_words.words_sha256),
        style.sha256(),
        project_dir,
    )


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _plan_documents(plan: GraphicsPlan) -> dict[str, dict]:
    """Use the same document shapes when publishing and checking a plan."""
    return {
        "agent-lane.json": plan.lane(),
        "layout.json": plan.layout().to_dict(),
        "cards.json": {"schema": CARDS_SCHEMA, "cards": [c.to_dict() for c in plan.overlays()]},
    }


@contextlib.contextmanager
def _ownership(out: Path) -> Iterator[tuple[Path, ...]]:
    """Snapshot a validated manifest so its own replacement has a raw hash receipt.

    The durable manifest hashes artifacts, excluding itself: a file cannot
    contain its own SHA256. The temporary receipt hashes the old manifest too.
    """
    manifest = out / "graphics-plan.json"
    pending = out / "receipt.pending.json"
    paths.validate_output(pending)
    if pending.exists():
        try:
            journal = json.loads(pending.read_bytes())
            if not isinstance(journal, dict) or journal.get("schema") != PENDING_SCHEMA:
                raise ValueError("not a publication journal")
            if not isinstance(journal.get("outputs"), list):
                raise ValueError("missing journal outputs")
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            raise OutputPathError("Invalid pending plan receipt.", code="unsafe_path") from exc
    # Run the safety checks even before reading an existing receipt.
    if manifest.is_symlink():
        raise OutputPathError("Plan receipt must not be a symlink.", code="unsafe_path")
    if not manifest.exists():
        yield (manifest, pending)
        return
    try:
        data = json.loads(manifest.read_bytes())
        if not isinstance(data, dict) or data.get("schema") != PLAN_SCHEMA:
            raise ValueError("not a graphics plan")
        records = data["outputs"]
        if (
            not isinstance(records, list)
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("path"), str)
                and isinstance(item.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                for item in records
            )
            or not set(_DOCUMENTS) <= {item["path"] for item in records}
        ):
            raise ValueError("missing output hashes")
    except (OSError, ValueError, KeyError, UnicodeError, RecursionError) as exc:
        raise OutputPathError(
            "Existing plan has no valid ownership receipt; choose a new directory.",
            code="unsafe_path",
        ) from exc
    # Only planner artifacts can be authorized by this receipt.
    records = [
        item
        for item in records
        if item["path"] in _DOCUMENTS or re.fullmatch(r"layers/[^/\\]+\.mov", item["path"])
    ]
    descriptor, name = tempfile.mkstemp(prefix=".yap-owner-", suffix=".json", dir=out)
    os.close(descriptor)
    snapshot = Path(name)
    try:
        snapshot.write_text(
            _json(
                {
                    "schema": PLAN_SCHEMA,
                    "outputs": [
                        *records,
                        {"path": manifest.name, "sha256": sha256_file(manifest)},
                    ],
                }
            ),
            encoding="utf-8",
        )
        yield (snapshot, pending)
    finally:
        snapshot.unlink(missing_ok=True)


def _manifest(plan: GraphicsPlan, documents: dict[str, str], layers: dict[str, str]) -> str:
    """Build the final receipt from the hashes being published or retained."""
    data = plan.to_dict()
    outputs = [{"path": name, "sha256": documents[name]} for name in _DOCUMENTS]
    for card in data["cards"]:
        layer = card["layer"]
        if layer["path"] in layers:
            layer.update(sha256=layers[layer["path"]], status="current")
            outputs.append({"path": layer["path"], "sha256": layer["sha256"]})
    return _json({**data, "outputs": outputs})


def _recorded_layers(out: Path, owners: tuple[Path, ...]) -> set[str]:
    """Include layers left by an interrupted publication, even without a plan."""
    names = set()
    for owner in owners:
        if not owner.exists():
            continue
        for item in json.loads(owner.read_bytes())["outputs"]:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            candidate = Path(item["path"])
            if candidate.is_absolute():
                try:
                    candidate = candidate.relative_to(out.resolve())
                except ValueError:
                    continue
            name = candidate.as_posix()
            if re.fullmatch(r"layers/[^/\\]+\.mov", name):
                names.add(name)
    return names


def _publish(out: Path, outputs: dict[Path, str | Path | None], owners: tuple[Path, ...]) -> None:
    """Journal every replacement and deletion, then publish the plan last."""
    entries = []
    for target, source in outputs.items():
        paths.validate_output(target, owner_receipts=owners)
        new_digest = None
        if isinstance(source, Path):
            new_digest = sha256_file(source)
        elif source is not None:
            new_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        entries.append(
            {
                "path": str(target.resolve()),
                "old_sha256": sha256_file(target) if target.exists() else None,
                "new_sha256": new_digest,
            }
        )
    pending = out / "receipt.pending.json"
    with paths.atomic_output(pending) as temporary:
        temporary.write_text(
            _json({"schema": PENDING_SCHEMA, "outputs": entries}), encoding="utf-8"
        )
    for target, source in outputs.items():
        if source is None:
            paths.validate_output(target, owner_receipts=owners)
            target.unlink(missing_ok=True)
        else:
            with paths.atomic_output(target, owner_receipts=owners) as temporary:
                if isinstance(source, Path):
                    shutil.copyfile(source, temporary)
                else:
                    temporary.write_text(source, encoding="utf-8")
    pending.unlink()


def write_plan(plan: GraphicsPlan, out_dir: str | Path) -> dict[str, Path]:
    """Keep current layers, retire owned stale layers, and journal all publication."""
    out = paths.validate_output_dir(out_dir)
    documents = _plan_documents(plan)
    parse_document("timeline", documents["agent-lane.json"], lane="agent")
    with _ownership(out) as owners:
        manifest = out / "graphics-plan.json"
        previous = json.loads(manifest.read_bytes()) if manifest.exists() else {}
        old_layers = {card["id"]: card.get("layer", {}) for card in previous.get("cards", [])}
        current = {}
        deletions = _recorded_layers(out, owners)
        for card in plan.cards:
            name = f"layers/{card.id}.mov"
            target = paths.validate_output(out / name, owner_receipts=owners)
            old = old_layers.get(card.id, {})
            if target.exists():
                digest = sha256_file(target)
                if (
                    old.get("path") == name
                    and old.get("render_key") == card.render_key
                    and old.get("sha256") == digest
                    and old.get("status") == "current"
                ):
                    current[name] = digest
                    deletions.discard(name)
                else:
                    deletions.add(name)
        contents = {out / name: _json(data) for name, data in documents.items()}
        document_hashes = {
            target.name: hashlib.sha256(content.encode("utf-8")).hexdigest()
            for target, content in contents.items()
        }
        publication = {
            **contents,
            **{
                out / name: None for name in sorted(deletions)
                if (out / name).exists() or (out / name).is_symlink()
            },
            manifest: _manifest(plan, document_hashes, current),
        }
        _publish(out, publication, owners)
    return {name: out / name for name in (*_DOCUMENTS, "graphics-plan.json")}


def render_layers(
    plan: GraphicsPlan,
    style: ResolvedStyle,
    out_dir: str | Path,
    *,
    install: RendererInstall | None = None,
) -> list[LayerResult]:
    """Stage alpha cards, then journal their publication and updated render identities."""
    if style.sha256() != plan.style:
        raise ValidationError("style", "must match the planned resolved style")
    for card in plan.cards:
        if card.template in PROJECT_CARD_TEMPLATES:
            raise ValidationError("template",
                                  f"render {card.template} cards through the project run")
    out = paths.validate_output_dir(out_dir)
    if not (out / "graphics-plan.json").exists():
        write_plan(plan, out)
    results = []
    with _ownership(out) as owners:
        stored = json.loads((out / "graphics-plan.json").read_bytes())
        identity = copy.deepcopy(stored)
        identity.pop("outputs")
        for card in identity["cards"]:
            if isinstance(card.get("layer"), dict):
                card["layer"].update(sha256=None, status="missing")
        if identity != plan.to_dict():
            raise ValidationError("plan", "write this plan before rendering its layers")
        for name in _DOCUMENTS:
            paths.validate_output(out / name, owner_receipts=owners)
        for name, data in _plan_documents(plan).items():
            target = out / name
            if not target.is_file() or target.read_bytes() != _json(data).encode("utf-8"):
                raise ValidationError(
                    "plan", "plan documents do not match; rerun the plan to finish recovery"
                )
        layers = paths.validate_output_dir(out / "layers")
        for card in plan.cards:
            paths.validate_output(layers / f"{card.id}.mov", owner_receipts=owners)
        work = paths.validate_output_dir(out / ".work")
        try:
            with tempfile.TemporaryDirectory(prefix="cards-", dir=work) as folder:
                staged = []
                for card in plan.cards:
                    slots = _resolved_slots(card.template, card.render_slots, plan.project_dir)
                    if card.template == "playing_grid":
                        tiles = [{"x": int(tile["box"]["x"]), "y": int(tile["box"]["y"]),
                                  "width": int(tile["box"]["w"]),
                                  "height": int(tile["box"]["h"])}
                                 for tile in card.clearance["tiles"]]
                        layer = card_layer_plan(card.template, slots, style,
                                                frames=card.end_frame - card.start_frame,
                                                fps=plan.fps, tiles=tiles)
                    elif card.template == "reel_feed":
                        layer = card_layer_plan(card.template, slots, style,
                                                frames=card.end_frame - card.start_frame,
                                                fps=plan.fps,
                                                ceiling=card.clearance.get("ceiling_px"))
                    else:
                        layer = fitted_card(
                            card.template,
                            slots,
                            style,
                            frames=card.end_frame - card.start_frame,
                            fps=plan.fps,
                            width=card.width,
                            height=card.height,
                        )
                    layer = replace(layer, x=card.x, y=card.y)
                    motion = card.clearance.get("card_motion", {})
                    layer = replace(layer, motion={key: motion[key] for key in ("side", "enter")}
                                    if motion.get("status") == "streak" else None)
                    if _render_key(layer) != card.render_key:
                        raise ValidationError("plan", "pixel inputs changed; write a new plan")
                    root = paths.validate_output_dir(Path(folder) / card.id)
                    result = render_card(
                        layer, root / "composition", root / "layer.mov", install=install
                    )
                    staged.append((result, layers / f"{card.id}.mov"))
                # A render failure leaves existing alpha layers and hashes intact.
                publication = {target: result.path for result, target in staged}
                document_hashes = {name: sha256_file(out / name) for name in _DOCUMENTS}
                layer_hashes = {
                    target.relative_to(out).as_posix(): sha256_file(result.path)
                    for result, target in staged
                }
                publication[out / "graphics-plan.json"] = _manifest(
                    plan, document_hashes, layer_hashes
                )
                _publish(out, publication, owners)
                for result, target in staged:
                    results.append(replace(result, path=target))
        finally:
            try:
                work.rmdir()
            except OSError:
                pass
    return results
