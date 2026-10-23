"""Fixed video tile geometry and the shared caption ceiling."""

from __future__ import annotations

import math
from collections.abc import Sequence

from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import DEFAULT_HAIR, gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.style import ResolvedStyle

GRID_RING = 6
GRID_PAD_X = 24
GRID_PAD_TOP = 16
GRID_PAD_BOTTOM = 40
GRID_CHIP_HEIGHT = 48


def window_rect(window: dict, frame: int) -> tuple[int, int, int, int]:
    """Return even geometry on the window's layer or receipt frame clock."""
    initial = tuple(window[key] for key in ("x", "y", "width", "height"))
    motion = window.get("motion")
    if motion is None:
        return initial
    start = motion.get("start_frame", motion.get("start"))
    if frame <= start:
        return initial
    if frame >= start + motion["frames"] - 1:
        return tuple(motion["to"])
    progress = (frame - start) / (motion["frames"] - 1)
    eased = 4 * progress ** 3 if progress < 0.5 else 1 - (-2 * progress + 2) ** 3 / 2
    return tuple(int(math.floor((first + (last - first) * eased) / 2 + 0.5)) * 2
                 for first, last in zip(initial, motion["to"], strict=True))


def caption_ceiling(style: ResolvedStyle, caption_top: int | None = None) -> int:
    """Reserve captions, entry motion, shadows and the creator's gap."""
    captions = style.data["captions"]
    if caption_top is not None:
        ceiling = caption_top
    elif captions["highlight_style"] == "giant":
        from yap_studio.captions import caption_layout_settings
        from yap_studio.stages.captions import caption_band_geometry

        ceiling = caption_band_geometry(
            caption_layout_settings(captions, style.data["brand"]),
            safe_bottom_px=style.data["internal"]["caption_safe_bottom_px"],
        ).y
    else:
        ceiling = math.floor(min(
            captions["top_px"], style.data["internal"]["caption_safe_bottom_px"],
        ) - 2.24 * captions["size_px"] - captions["entry_rise_px"]
            - 3 * captions["shadow_blur_px"] - abs(captions["shadow_y_px"]) - 8)
    return max(0, ceiling - style.data["graphics"]["caption_gap_px"])


def tile_window(tile: dict, aspect: float) -> dict:
    """Keep the paper ring and shadow inside the reserved tile box."""
    width = tile["width"] - 2 * (GRID_PAD_X + GRID_RING)
    return {"x": tile["x"] + GRID_PAD_X + GRID_RING,
            "y": tile["y"] + GRID_PAD_TOP + GRID_RING,
            "width": width, "height": int(width / aspect) // 2 * 2}


def grid_layout(
    aspects: Sequence[float], faces: Sequence[Sequence[Box]] | None, *,
    margin: float, ceiling: int, top_bottom: int, id: str,
    labels: Sequence[str] = (), layer: str = "front", reason: str = "faces not measured",
    face_windows: tuple[tuple[int, int], ...] = (), start_frame: int = 0,
) -> tuple[tuple[dict, ...], dict]:
    """Fit fixed tiles, with face windows on the output clock at start_frame."""
    aspects = [min(16 / 9, max(3 / 4, aspect)) for aspect in aspects]
    heads = {head_box(face) for frame in faces or () for face in frame}
    restoration_gaps = layer == "behind" and bool(face_windows)
    unrestored_heads = {head_box(face) for index, frame in enumerate(faces or ())
                        if any(first <= start_frame + index < last
                               for first, last in face_windows)
                        for face in frame} if restoration_gaps else set()
    ceiling = min(1920, ceiling)

    def height(width: int, index: int) -> int:
        picture = int((width - 2 * (GRID_PAD_X + GRID_RING)) / aspects[index]) // 2 * 2
        return (picture + 2 * GRID_RING + GRID_PAD_TOP + GRID_PAD_BOTTOM
                + (GRID_CHIP_HEIGHT if index < len(labels) and labels[index] else 0))

    def pack(width: int) -> list[dict]:
        if layer == "behind":
            columns = 3 if len(aspects) >= 5 else 2
            if columns * width + (columns - 1) * 16 > 1032:
                return []
            rows = [max(height(width, index) for index in range(first, min(
                first + columns, len(aspects)))) for first in range(0, len(aspects), columns)]
            total = sum(rows) + (len(rows) - 1) * 16
            if total > ceiling - 96:
                return []
            y = (96 + (ceiling - 96 - total) / 2) // 2 * 2
            tiles = []
            for row, row_height in enumerate(rows):
                count = min(columns, len(aspects) - row * columns)
                x = (1080 - count * width - (count - 1) * 16) // 2 // 2 * 2
                for column in range(count):
                    index = row * columns + column
                    tiles.append({"x": int(x + column * (width + 16)), "y": int(y),
                                  "width": width, "height": height(width, index)})
                y += row_height + 16
            return tiles
        if faces is None:
            regions = [Box(24, 96, 1032, max(0, min(top_bottom, ceiling) - 96))]
        elif not heads:
            regions = [Box(24, 96, 1032, max(0, ceiling - 96))]
        else:
            left, right = min(head.x for head in heads), max(head.x1 for head in heads)
            top, bottom = min(head.y for head in heads), max(head.y1 for head in heads)
            middle = max(96, math.ceil(top / 2) * 2)
            right_start = max(24, math.ceil((right + margin) / 2) * 2)
            lower = max(96, math.ceil((bottom + margin) / 2) * 2)
            regions = [
                Box(24, 96, 1032, max(0, min(ceiling, top - margin) - 96)),
                Box(24, middle, max(0, left - margin - 24), max(0, ceiling - middle)),
                Box(right_start, middle, max(0, 1056 - right_start), max(0, ceiling - middle)),
                Box(24, lower, 1032, max(0, ceiling - lower)),
            ]
        tiles = []
        for region in regions:
            y = math.ceil(region.y / 2) * 2
            while len(tiles) < len(aspects):
                x = math.ceil(region.x / 2) * 2
                row_height = 0
                while x + width <= region.x1 and len(tiles) < len(aspects):
                    size = height(width, len(tiles))
                    candidate = Box(x, y, width, size)
                    row_height = max(row_height, size)
                    if candidate.y1 <= region.y1 and all(
                        gap_px(candidate, Box(tile["x"], tile["y"], tile["width"], tile["height"]))
                        >= 16 for tile in tiles
                    ):
                        tiles.append({"x": x, "y": y, "width": width, "height": size})
                    x += width + 16
                if not row_height or y + row_height > region.y1:
                    break
                y += row_height + 16
        return tiles

    for width in range(480, 199, -40):
        tiles = pack(width)
        if len(tiles) == len(aspects):
            if restoration_gaps and (faces is None or any(
                gap_px(Box(tile["x"], tile["y"], tile["width"], tile["height"]), head) < margin
                for tile in tiles for head in unrestored_heads
            )):
                continue
            evidence = [{"box": Box(tile["x"], tile["y"], tile["width"], tile["height"]).to_dict(),
                         "gap_px": min((gap_px(Box(tile["x"], tile["y"], tile["width"],
                                                    tile["height"]), head) for head in heads),
                                       default=None)} for tile in tiles]
            return tuple(tiles), {
                "status": "behind_head" if layer == "behind" else (
                    "unchecked" if faces is None else "clear"),
                "reason": ("fixed grid clears heads while person restoration is skipped"
                           if restoration_gaps else
                           "fixed grid behind the speaker" if layer == "behind" else
                           reason if faces is None else "tiles clear the heads and captions"),
                "margin_px": margin, "hair": DEFAULT_HAIR.to_dict(),
                "gap_px": min((tile["gap_px"] for tile in evidence
                               if tile["gap_px"] is not None), default=None), "tiles": evidence,
            }
    if restoration_gaps:
        raise ValidationError(f"graphic.{id}.clearance", "behind tiles cannot clear the speaker "
                              "without person restoration; move the grid off the zoom, "
                              "or use the front layer")
    raise ValidationError(f"graphic.{id}.clearance", f"no room around the face for {len(aspects)} "
                          "tiles; use fewer clips, layer behind, or a moment where the "
                          "face sits lower")


def feed_window(ceiling: int) -> dict:
    """Centre a generic screen above the caption ceiling with room for its body."""
    width = int(min(648, (ceiling - 96 - 104) * 9 / 16)) // 18 * 18
    if width < 324:
        raise ValidationError("graphic.clearance", "give the feed more room above the captions")
    height = width * 16 // 9
    return {"x": (1080 - width) // 2 // 2 * 2,
            "y": (96 + ceiling - height) // 2 // 2 * 2,
            "width": width, "height": height, "radius": 28}
