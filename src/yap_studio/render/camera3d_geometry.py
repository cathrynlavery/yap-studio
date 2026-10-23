"""Match the painter's posterized text poses and shutter bounds."""

import math

from yap_studio.render import camera3d as camera


def _lerp(a, b, u):
    return a + (b - a) * u


def _sample(samples: list[dict], frame: float) -> dict:
    frame = max(0, min(len(samples) - 1, frame))
    index = min(len(samples) - 2, math.floor(frame))
    return {key: _lerp(samples[index][key], samples[index + 1][key], frame - index)
            for key in ("px", "py", "tx", "ty", "cz")}


def _pose(item: dict, samples: list[dict], frame: float, hero: bool) -> list[float]:
    x, y, scale = item["ref_x"], item["ref_y"], 1
    if poses := item.get("poses"):
        position = max(0, min(len(poses) - 1, frame - item["on_frame"]))
        index = max(0, min(len(poses) - 2, math.floor(position)))
        a, b = poses[index], poses[min(index + 1, len(poses) - 1)]
        x, y, scale = [_lerp(a[key], b[key], position - index)
                       for key in ("ref_x", "ref_y", "scale")]
    step = max(0, min(len(camera._HERO_ENTRY) - 1, (frame - item["on_frame"]) / 2))
    index = min(len(camera._HERO_ENTRY) - 2, math.floor(step))
    entry = _lerp(camera._HERO_ENTRY[index], camera._HERO_ENTRY[index + 1], step - index)
    if not item.get("flat"):
        y += (140 if hero else 30) * entry
        x, y, k = camera._project(_sample(samples, frame), x, y, item["depth"])
        scale *= k
    return [x, y, scale]


def shutter_poses(item: dict, samples: list[dict], frame: int, *, hero=False) -> list:
    first = frame // 2 * 2
    return [_pose(item, samples, first + offset, hero) for offset in (-.5, .5)]


def motion(item: dict, samples: list[dict], frame: int, *, hero=False) -> float:
    """Return the same anchor travel used by the painter's sharp rest test."""
    a, b = shutter_poses(item, samples, frame, hero=hero)
    return math.hypot(a[0] - b[0], a[1] - b[1])


def painted_box(item: dict, samples: list[dict], frame: int, *, hero=False,
                metrics: dict | None = None) -> list[float]:
    """Include entry rise, held frames and every visible shutter ghost."""
    poses = shutter_poses(item, samples, frame, hero=hero)
    if not hero and math.hypot(poses[0][0] - poses[1][0], poses[0][1] - poses[1][1]) < .6:
        poses = [[(a + b) / 2 for a, b in zip(*poses, strict=True)]]
    kind = "hero" if hero else "caption"
    base = (metrics or camera._metrics())["base"][kind]
    size = item["ref_size"]
    width = (sum(item["advances"]) * size if hero else
             item["box"][2] - item["box"][0] if item.get("flat") else
             (sum(metrics["adv"][kind].get(char, .6) for char in item["text"])
              if metrics else camera._advance(item["text"], kind)) * size)
    boxes = [[x, y - base * size * scale, x + width * scale,
              y + (1 - base) * size * scale] for x, y, scale in poses]
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def sharp_frame(item: dict, samples: list[dict], *, hero=False) -> int:
    """Prefer opaque rest frames, or the smallest shutter travel during a push."""
    first, last = item["on_frame"], item["off_frame"]
    frames = list(range(first, min(last, item["fade_start_frame"]), 2)) or [first]
    middle = (first + last - 1) / 2

    def rank(frame):
        travel = motion(item, samples, frame, hero=hero)
        return (0 if not hero and travel < .6 else travel, abs(frame - middle), frame)

    return min(frames, key=rank)
