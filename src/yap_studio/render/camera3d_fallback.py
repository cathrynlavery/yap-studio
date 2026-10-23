"""Keep every display word visible when the camera look runs out of space."""

import math

from yap_studio.render import camera3d as camera
from yap_studio.render.camera3d_geometry import painted_box


def flat_size(text: str) -> int:
    style = camera._ACTIVE_STYLE.get() or camera.resolve_style("preset:clean_minimal").data
    requested = min(camera.CAPTION_MAX_SIZE, style["captions"]["size_px"])
    fitted = math.floor((camera.SAFE[2] - camera.SAFE[0])
                        / max(.01, camera._advance(text, "caption")))
    return max(camera.CAPTION_MIN_SIZE, min(requested, fitted))


def lowest_box(text: str) -> list[float]:
    """Start flat captions on the creator's caption line."""
    from yap_studio.stages.captions import caption_band_geometry

    style = camera._ACTIVE_STYLE.get() or camera.resolve_style("preset:clean_minimal").data
    bottom = style["internal"]["caption_safe_bottom_px"]
    size = flat_size(text)
    captions = {**style["captions"], "size_px": size}
    top = captions["top_px"]
    if captions["position"] != "bottom":
        geometry = caption_band_geometry(captions, safe_bottom_px=bottom)
        top = geometry.y + geometry.padding
    top = max(camera._SAFE_TOP.get(), min(top, bottom - size))
    width = min(camera.SAFE[2] - camera.SAFE[0], camera._advance(text, "caption") * size)
    return [(camera.WIDTH - width) / 2, top, (camera.WIDTH + width) / 2, top + size]


def band_positions(box: list[float], blocked: list) -> list[float]:
    """Search down from the caption line, then back up within the safe frame."""
    style = camera._ACTIVE_STYLE.get() or camera.resolve_style("preset:clean_minimal").data
    height = box[3] - box[1]
    first, top = box[1], camera._SAFE_TOP.get()
    bottom = style["internal"]["caption_safe_bottom_px"] - height
    positions = {first, top, bottom, *range(math.ceil(first), math.floor(bottom) + 1, 12),
                 *range(math.floor(first), math.ceil(top) - 1, -12)}
    positions.update(y for other in blocked for y in (other[3] + 24, other[1] - height - 24))
    return (sorted(y for y in positions if first <= y <= bottom)
            + sorted((y for y in positions if top <= y < first), reverse=True))


def _conflicts(first: dict, second: dict, samples: list[dict]) -> bool:
    start = max(first["on_frame"], second["on_frame"])
    end = min(first["off_frame"], second["off_frame"])
    return any(camera._overlap(painted_box(first, samples, f),
                               painted_box(second, samples, f), 24)
               for f in range(start, end))


def _anchors(y: float, width: float, lowest: list[float], blocked: list) -> list[float]:
    left, right = camera.SAFE[0], camera.SAFE[2] - width
    anchors = [lowest[0], left, right]
    for box in blocked:
        if y < box[3] + 24 and y + lowest[3] - lowest[1] > box[1] - 24:
            anchors.extend((max(left, box[0] - width - 24), min(right, box[2] + 24)))
    return list(dict.fromkeys(x for x in anchors if left <= x <= right))


def _band_box(group: dict, groups: list[dict], sections: list[dict], plate: dict,
              samples: list[dict], heroes: list, obstacles: list[dict],
              heads: list | None = None) -> list[float]:
    from yap_studio.render.camera3d_look import Camera3DRefusal

    lowest = lowest_box(group["text"])
    width = lowest[2] - lowest[0]
    height = lowest[3] - lowest[1]
    first, last = group["on_frame"], group["off_frame"]
    blocked = [heads[f] if heads is not None else camera._head_at(plate, samples[f],
                              camera._section_at(sections, f)["head_box"])
               for f in range(first, last)]
    blocked += camera._hero_obstacles(heroes, first, last)
    blocked += [item["box"] for item in obstacles
                if item["start_frame"] < last and item["end_frame"] > first]
    occupied = [other for other in groups if other is not group and not other.get("unplaced")
                and other["on_frame"] < last and first < other["off_frame"]]
    freed = []
    for y in band_positions(lowest, blocked):
        for x in _anchors(y, width, lowest, blocked):
            box = [x, y, x + width, y + height]
            if any(camera._overlap(box, other, 24) for other in blocked):
                continue
            conflicts = [other for other in occupied if any(
                camera._overlap(box, painted_box(other, samples, frame), 24)
                for frame in range(max(first, other["on_frame"]),
                                   min(last, other["off_frame"])))]
            if not conflicts:
                return box
            if all(other["on_frame"] < first for other in conflicts):
                freed.append((box, conflicts))
    if freed:
        box, conflicts = freed[0]
        camera._end_replaced_groups(conflicts, first)
        return box
    minutes, seconds = divmod(first // camera.FPS, 60)
    raise Camera3DRefusal(
        f"3D camera captions need room around you, but there is none at {minutes}:{seconds:02}.",
        "Frame the shot wider, or pick another caption look.")


def recover_groups(groups: list[dict], sections: list[dict], plate: dict,
                   samples: list[dict], fallbacks: list[dict], heroes: list,
                   obstacles: list[dict], heads: list | None = None) -> None:
    """Replace rejected poses with a flat bottom band and retire conflicting fades."""
    reasons = {(item["sentence"], tuple(item["word_ids"])): item["reason"]
               for item in fallbacks if item["action"] == "unplaced"}
    fallbacks[:] = [item for item in fallbacks if item["action"] != "unplaced"]
    base = camera._metrics()["base"]["caption"]
    for index, group in enumerate(groups):
        if group.get("unplaced"):
            camera._end_replaced_groups(camera._replaced_groups(groups[:index],
                                                               group["on_frame"]),
                                        group["on_frame"])
            box = _band_box(group, groups, sections, plate, samples, heroes, obstacles, heads)
            size = box[3] - box[1]
            group.update(flat=True, bottom_band=True, box=box, size=size, ref_size=size,
                         depth=0, ref_x=box[0], ref_y=box[1] + base * size,
                         tracking=round(min(0.0, (box[2] - box[0]
                             - camera._advance(group["text"], "caption") * size)
                             / max(1, len(group["text"]) - 1)), 5))
            group["poses"] = [{"frame": f, "box": box, "ref_x": box[0],
                               "ref_y": box[1] + base * size, "scale": 1}
                              for f in range(group["on_frame"], group["off_frame"])]
            group.pop("unplaced")
            reason = reasons.get((group["sentence"], tuple(group["word_ids"])),
                                 "no conflict-free torso box")
            fallbacks.append({"sentence": group["sentence"], "word_ids": group["word_ids"],
                              "reason": reason, "action": "bottom band"})
        for prior in groups[:index]:
            if (not prior.get("unplaced") and prior["on_frame"] < group["on_frame"]
                    and _conflicts(prior, group, samples)):
                camera._end_replaced_groups([prior], group["on_frame"])
