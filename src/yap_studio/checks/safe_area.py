"""Placed text and pictures against the 9:16 social safe areas HyperFrames checks."""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction

from yap_studio.checks.caption_layout import LayoutRecord
from yap_studio.checks.model import Finding, format_time
from yap_studio.faces.model import Box

# Mirrors SOCIAL_SAFE_AREAS (core/src/safeAreas.ts) and safeZoneEdgesCrossed
# (cli/src/utils/safeZones.ts) in HyperFrames 0.9.68, bundled in dist/chunk-QFO7O6CB.js,
# so `hyperframes check` and verify agree. Bands are (top, bottom, left, right) on the
# zone's own grid, top to bottom.
SAFE_AREAS: dict[str, tuple[str, int, int, tuple[tuple[int, int, int, int], ...]]] = {
    "tiktok": ("TikTok", 720, 1280, ((160, 560, 80, 640), (560, 840, 80, 520))),
    "reels": ("Instagram Reels", 100, 100, ((14, 65, 6, 94),)),
    "shorts": ("YouTube Shorts", 1080, 1920, ((288, 1248, 48, 888),)),
}
# HyperFrames checks all three by default. Yap Studio checks TikTok and Reels, where the
# creator posts, and YouTube Shorts only when the render targets Shorts.
DEFAULT_ZONES = ("tiktok", "reels")
_EDGES = ("left", "top", "right", "bottom")
# Yap Studio measures a card's drawn panel; HyperFrames measures the text inside it. A card
# fails only when the crossing reaches this far into the panel, where padded text would sit
# (the NTTD title card pads its text 26 px); a shallower crossing is a warning.
PANEL_INSET_PX = 24  # on a 1080 px wide frame; scaled with the frame width


def is_vertical(width: int, height: int) -> bool:
    """Match HyperFrames' 9:16 test, which applies the social safe areas."""
    return height > 0 and abs(width / height - 9 / 16) < 0.01


def edges_crossed(box: Box, width: int, height: int, zone: str) -> list[str]:
    """Return the safe-area edges a box crosses, in HyperFrames' order."""
    _, zone_width, zone_height, bands = SAFE_AREAS[zone]

    def x(value: float) -> float:
        return value / zone_width * width

    def y(value: float) -> float:
        return value / zone_height * height

    edges = set()
    if box.y < y(bands[0][0]):
        edges.add("top")
    if box.y1 > y(bands[-1][1]):
        edges.add("bottom")
    for top, bottom, left, right in bands:
        if box.y1 <= y(top) or box.y >= y(bottom):
            continue
        if box.x < x(left):
            edges.add("left")
        if box.x1 > x(right):
            edges.add("right")
    return [edge for edge in _EDGES if edge in edges]


def _crossed(boxes: Iterable[Box], width: int, height: int,
             zones: tuple[str, ...]) -> tuple[Box, dict[str, list[str]]] | None:
    """Return the first box outside any zone, with every zone's crossed edges."""
    for box in boxes:
        if box.w <= 0 or box.h <= 0:
            continue
        crossed = {zone: edges for zone in zones
                   if (edges := edges_crossed(box, width, height, zone))}
        if crossed:
            return box, crossed
    return None


def _zones_text(crossed: dict[str, list[str]]) -> str:
    return ", ".join(
        f"{SAFE_AREAS[zone][0]} ({' and '.join(edges)} edge{'s' if len(edges) > 1 else ''})"
        for zone, edges in crossed.items()
    )


def check_safe_area(
    record: LayoutRecord, *, fps: float | Fraction, platform: str,
) -> tuple[list[Finding], dict]:
    """Fail cards and titles deep outside the safe areas; warn for edges, captions, pictures."""
    layout = record.layout
    if not is_vertical(layout.width, layout.height):
        return [], {"ran": False, "reason": "not a 9:16 render"}
    zones = (platform,) if platform in SAFE_AREAS else DEFAULT_ZONES
    findings = []
    for element in layout.elements:
        if element.kind == "background":
            continue
        hit = _crossed([element.box], layout.width, layout.height, zones)
        if hit is None:
            continue
        box, crossed = hit
        element_id = record.element_ids.get(element.id, element.id)
        label = record.labels.get(element.id)
        picture = element.kind == "overlay"
        inset = PANEL_INSET_PX * layout.width / 1080
        inner = Box(box.x + inset, box.y + inset, max(0, box.w - 2 * inset),
                    max(0, box.h - 2 * inset))
        deep = not picture and _crossed([inner], layout.width, layout.height, zones) is not None
        name = f'{"Picture" if picture else element.kind.capitalize()} "{element_id}"' + (
            f" ({label})" if label else "")
        time_s = element.start_frame / float(fps)
        edge_note = "" if picture or deep else (
            f" Only the panel's outer {inset:g} px crosses, so its text is likely inside.")
        findings.append(Finding(
            "platform.safe_area", "fail" if deep else "warn",
            f"{name} is outside the safe area at {format_time(time_s)}: "
            f"{_zones_text(crossed)}.{edge_note}",
            time_s=time_s,
            suggested_action=(
                "Move the picture inside the safe area, or accept that app buttons may cover it."
                if picture else
                "Plan the card again (run graphics) so it sits inside the safe area."
            ),
            evidence={"target_type": "element", "target_id": element.id,
                      "element_id": element_id, "target_kind": element.kind,
                      **({"label": label} if label else {}),
                      "target_box": box.to_dict(), "zones": crossed,
                      **({} if picture else {"panel_inset_px": inset, "text_area_crosses": deep}),
                      "start_frame": element.start_frame, "end_frame": element.end_frame},
        ))
    captions = layout.captions if record.measured else ()
    for caption in captions:
        hit = _crossed(caption.boxes, layout.width, layout.height, zones)
        if hit is None:
            continue
        box, crossed = hit
        frame = box.start_frame if box.start_frame is not None else caption.start_frame
        time_s = frame / float(fps)
        # The caption band is the creator's chosen style, so it is reported, not failed.
        findings.append(Finding(
            "platform.safe_area", "warn",
            f'Caption "{caption.text}" is outside the safe area at {format_time(time_s)}: '
            f"{_zones_text(crossed)}.",
            time_s=time_s,
            suggested_action="App buttons and descriptions may cover it; raise the caption "
                             "band in the style if that matters for this video.",
            evidence={"target_type": "caption", "target_id": caption.id,
                      "target_box": box.to_dict(), "zones": crossed,
                      "start_frame": caption.start_frame, "end_frame": caption.end_frame},
        ))
    coverage = {"ran": True, "reason": None, "zones": list(zones)}
    if not record.measured:
        coverage["note"] = f"captions were not measured: {record.reason}"
    return findings, coverage
