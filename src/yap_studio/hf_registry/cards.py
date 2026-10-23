"""Fit installed registry items into the engine's ordinary card panel."""

from __future__ import annotations

import copy
import html
import json
import math
from collections.abc import Mapping
from html.parser import HTMLParser
from pathlib import Path

from yap_studio import cards, paths
from yap_studio.errors import ValidationError
from yap_studio.hf_registry import catalog, install
from yap_studio.render import hyperframes
from yap_studio.style import ResolvedStyle

KIND_PREFIX = "hf:"


def is_kind(kind: str) -> bool:
    """Recognize registry kind ids, including names that need a validation error."""
    return kind.startswith(KIND_PREFIX)


def item_name(kind: str) -> str:
    """Resolve an enabled catalog kind, refusing install-only pilots."""
    name = kind[len(KIND_PREFIX):] if is_kind(kind) else ""
    if name not in catalog.items() or catalog.item(name)["status"] != "on":
        raise ValidationError("template", "choose " + ", ".join(kinds()))
    return name


def kinds(status: str = "on") -> tuple[str, ...]:
    """Return the catalog's kinds in their shipped order."""
    return tuple(KIND_PREFIX + name for name, item in catalog.items().items()
                 if item["status"] == status)


def preview_slots(kind: str) -> dict[str, object]:
    """Return an independent copy of the catalog's validated sample slots."""
    return copy.deepcopy(catalog.item(item_name(kind))["preview"])


def minimum_duration_s(kind: str) -> float:
    """Read the catalog's minimum window for a registry card."""
    return catalog.item(item_name(kind))["min_s"]


def template_metadata(kind: str) -> dict:
    """Expose the same slot and provenance contract as built-in card metadata."""
    name = item_name(kind)
    entry = catalog.item(name)
    return {
        "id": kind, "version": 1, "reads": entry["kind"]["role"],
        "label": entry["kind"]["label"], "use": entry["kind"]["use"],
        "derived_from": [{"item": name, "path": entry["upstream_path"]}],
        "slots": copy.deepcopy(entry["slots"]),
        "box": {"width": "graphics.card_width_px + 80",
                "height": "internal.key_text_bottom_px - internal.key_text_top_px",
                "y": "internal.key_text_top_px"},
    }


class _ItemMetadata(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__()
        self.root: str | None = None
        self.root_element_id: str | None = None
        self.variables: dict[str, str] = {}
        self.feed(source)
        self.close()
        if not self.root:
            raise ValidationError("registry item", "composition root is missing")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if self.root is None and attributes.get("data-composition-id"):
            self.root = attributes["data-composition-id"]
        if self.root and tag != "html" and attributes.get("data-composition-id") == self.root:
            self.root_element_id = attributes.get("id")
        declaration = attributes.get("data-composition-variables")
        if declaration:
            try:
                variables = json.loads(declaration)
                if not isinstance(variables, list):
                    raise ValueError("expected a variable list")
                self.variables.update({rule["id"]: rule["type"] for rule in variables})
            except (ValueError, KeyError, TypeError) as exc:
                raise ValidationError("registry item", "invalid variable declarations") from exc


def card_plan(
    kind: str, slots: Mapping[str, object], style: ResolvedStyle, *,
    frames: int | None = None, fps: int = 30,
) -> cards.CardPlan:
    """Validate content, timing and the minimum panel before placement."""
    name = item_name(kind)
    entry = catalog.item(name)
    graphics, internal = style.data["graphics"], style.data["internal"]
    if type(fps) is not int or fps < 1:
        raise ValidationError("fps", "use a positive integer")
    if frames is None:
        frames = max(1, round(graphics["hold_s"] * fps))
    if type(frames) is not int or frames < 1:
        raise ValidationError("frames", "use a positive integer")
    minimum = math.ceil(entry["min_s"] * fps)
    if frames < minimum:
        raise ValidationError(
            "frames", f"a {kind} card needs at least {entry['min_s']:g} s "
            f"({minimum} frames at {fps} fps); raise graphics.hold_s or pass more frames",
        )
    values, _ = cards._slots(slots, entry["slots"])
    width = graphics["card_width_px"] + 2 * cards.PAD_X
    height = internal["key_text_bottom_px"] - internal["key_text_top_px"]
    panel_w, panel_h = width - 2 * cards.PAD_X, height - cards.PAD_TOP - cards.PAD_BOTTOM
    min_w, min_h = entry["min_panel"]
    if panel_w < min_w or panel_h < min_h:
        raise ValidationError("graphic.box", f"a {kind} card needs a panel of at least "
                              f"{min_w} x {min_h} px; raise graphics.card_width_px or "
                              "the key text band height, or choose another card")
    root = install.item_files(name)
    source = (root / entry["main"]).read_text(encoding="utf-8")
    catalog.validate_color_tokens(name, entry, source)
    metadata = _ItemMetadata(source)
    variables: dict[str, object] = {}
    for key, pattern in entry["variables"].items():
        if key not in metadata.variables:
            raise ValidationError("registry item", f"variable {key!r} is not declared by {name}")
        value = pattern.format_map(values) if isinstance(pattern, str) else pattern
        if metadata.variables[key] == "number":
            try:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("nonfinite number")
            except (ValueError, TypeError) as exc:
                raise ValidationError("registry item", f"variable {key!r} needs a number") from exc
        else:
            value = str(value)
        variables[key] = value
    fonts_by_role = {"cards": graphics["font"], "titles": style.data["hook"]["font"]}
    colors = {"accent": style.data.get("brand", {}).get("accent_color") or
              graphics["accent_color"], "ink": graphics["ink_color"],
              "paper": graphics["paper_color"]}
    colors["accent_tint"] = cards.glow_colors(colors["accent"])[0]
    return cards.CardPlan(
        kind, (1080 - width) // 2, internal["key_text_top_px"], width, height, frames, fps,
        variables, (fonts_by_role[entry["kind"]["role"]],),
        registry={"entry": copy.deepcopy(entry), "fonts_by_role": fonts_by_role,
                  "tokens": {**catalog.HOST_TOKENS,
                             **{token: colors[color] for token, color in entry["tokens"].items()}}},
    )


def prepare(
    plan: cards.CardPlan, work_dir: Path, *, install: hyperframes.RendererInstall | None = None,
) -> Path:
    """Localize a nested composition and give its host the shared card fade-out."""
    from yap_studio.hf_registry import localize

    name = item_name(plan.template)
    if plan.registry is None:
        raise ValidationError("registry plan", "plan the card with card_layer_plan first")
    entry = plan.registry["entry"]
    composition = work_dir / name
    with paths.operation():
        paths.validate_output_dir(composition)
        duration = hyperframes.layer_duration_s(plan.frames, plan.fps)
        localized = localize.localize(
            name, composition, fonts_by_role=plan.registry["fonts_by_role"],
            role=entry["kind"]["role"], install=install or hyperframes.ensure_installed(),
            duration_s=duration,
        )
        metadata = _ItemMetadata((composition / localized.main).read_text(encoding="utf-8"))
        host_id = "hf-card-host" if metadata.root_element_id == "hf-host" else "hf-host"
        panel_w = plan.width - 2 * cards.PAD_X
        panel_h = plan.height - cards.PAD_TOP - cards.PAD_BOTTOM
        w, h = panel_w, panel_h
        position = f"left:{cards.PAD_X}px;top:{cards.PAD_TOP}px;"
        transform = ""
        if entry["mount"] == "scale":
            w, h = entry["stage"]
            x, y, cw, ch = entry["content"]
            scale = min(entry["scale_max"], panel_w / cw, panel_h / ch)
            tx = cards.PAD_X + (panel_w - cw * scale) / 2 - x * scale
            ty = cards.PAD_TOP + (panel_h - ch * scale) / 2 - y * scale
            position = "left:0;top:0;"
            transform = (f"transform-origin:0 0;transform:translate({tx:g}px,{ty:g}px) "
                         f"scale({scale:g});")
        tokens = ";".join(f"{token}:{value}" for token, value in plan.registry["tokens"].items())
        readability = ("filter:drop-shadow(0 0 1px #000) drop-shadow(0 0 1px #000) "
                       "drop-shadow(0 0 3px rgba(0,0,0,0.6));"
                       if entry["readability"] == "halo" else "")
        mount_style = (f"position:absolute;{position}width:{w}px;height:{h}px;"
                       f"{transform}{readability}{tokens}")
        # JSON escapes keep text out of the asset URL scan; JSON.parse restores every slash.
        variables = json.dumps(plan.variables, ensure_ascii=False, allow_nan=False)
        variables = html.escape(variables.replace("/", r"\/"), quote=False).replace("'", "&#39;")
        # The pinned compiler strips timing from the inner root. Route authored root
        # lookups to the mount so scripts read its card duration instead of a fallback.
        authored_root = (
            f'data-hf-authored-id="{html.escape(metadata.root_element_id, quote=True)}" '
            if metadata.root_element_id else ""
        )
        source = (
            '<!doctype html><html><head><meta charset="utf-8">' + localize.host_head(localized) +
            f'<style>html,body{{margin:0;width:{plan.width}px;height:{plan.height}px;'
            'background:transparent;overflow:hidden}'
            f'#{host_id}{{position:relative;width:{plan.width}px;height:{plan.height}px;'
            'overflow:hidden}'
            f'</style></head><body><div id="{host_id}" data-composition-id="hf-host" '
            f'data-width="{plan.width}" data-height="{plan.height}" data-start="0" '
            f'data-duration="{duration}"><div id="hf-mount" '
            + authored_root +
            f'data-composition-id="{html.escape(metadata.root, quote=True)}" '
            f'data-composition-src="{html.escape(localized.main, quote=True)}" '
            f'data-start="0" data-duration="{duration}" data-width="{w}" data-height="{h}" '
            f'data-track-index="0" data-variable-values=\'{variables}\' '
            f'style="{html.escape(mount_style, quote=True)}"></div></div>'
            '<script>const D=' + str(duration) + ';const OUT=Math.min(0.3,D*0.15);'
            'const tl=gsap.timeline({paused:true});'
            'tl.to("#hf-mount",{opacity:0,duration:OUT,ease:"power2.in",immediateRender:false},D-OUT);'
            'window.__timelines=window.__timelines||{};window.__timelines["hf-host"]=tl;'
            '</script></body></html>'
        )
        paths.atomic_write_text(composition / "index.html", source)
    return composition
