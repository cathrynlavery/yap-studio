"""Streaming checks of cards composited behind the plate's person."""

from __future__ import annotations

import json
from collections.abc import Callable, Generator, Sequence
from contextlib import ExitStack, closing
from dataclasses import asdict, dataclass, fields, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np

from yap_studio import ffmpeg, paths
from yap_studio.errors import InputError, ValidationError
from yap_studio.matte.composite import CardOverlay, _matte_path, _validate_cards
from yap_studio.matte.ink import PERSON_ALPHA_THRESHOLD as PERSON_ALPHA_THRESHOLD
from yap_studio.matte.ink import covered_ink_pixels
from yap_studio.matte.model import _finite, _object, load_matte

CUTOUT_REPORT_SCHEMA = "yap-studio/cutout-report@1"
MAX_CARDS_AT_ONCE = 4
HIDDEN_GLYPH_SHARE = 0.5
HIDDEN_TIME_SHARE = 0.25
GLYPH_ALPHA_THRESHOLD = 128
CheckStatus = Literal["pass", "fail", "review", "skipped"]
_CHECK_IDS = (
    "matte.frames", "matte.doubled_edges", "matte.flicker", "matte.hand_dropout", "matte.occlusion"
)


@dataclass(frozen=True)
class CutoutThresholds:
    """Named defaults measured on real takes, in native 8-bit luma and alpha."""

    edge_diff: int = 24
    edge_share_max: float = 0.0005
    flicker_px: int = 32
    flicker_p95_max: float = 1500
    flicker_frame_max: float = 10000
    dropout_iou_dip: float = 0.03
    dropout_motion: float = 2.0
    ink_diff: int = 40
    occlusion_max: float = 0.02

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            location = f"thresholds.{field.name}"
            if field.name in ("edge_diff", "flicker_px", "ink_diff") and type(value) is not int:
                raise ValidationError(location, "expected a positive integer")
            if _finite(value, location) <= 0:
                raise ValidationError(location, "must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return all thresholds without coercing pixel levels."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CutoutThresholds:
        """Parse threshold overrides, rejecting unknown names and invalid values."""
        obj = _object(data, "thresholds")
        names = {field.name for field in fields(cls)}
        for key in obj:
            if key not in names:
                raise ValidationError(f"thresholds.{key}", "unknown field")
        return cls(**obj)


DEFAULT_THRESHOLDS = CutoutThresholds()


@dataclass(frozen=True)
class CheckResult:
    """One check's explicit outcome and JSON-compatible measurements."""

    id: str
    status: CheckStatus
    detail: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the check record."""
        return asdict(self)


@dataclass(frozen=True)
class CutoutReport:
    """Cutout results, including skipped measurements and exact rational fps."""

    status: CheckStatus
    reason: str | None
    checks: tuple[CheckResult, ...]
    cards: int
    behind: int
    fps: Fraction
    frames: int
    thresholds: CutoutThresholds

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned cutout report document."""
        return {
            "schema": CUTOUT_REPORT_SCHEMA,
            "status": self.status,
            "reason": self.reason,
            "checks": [check.to_dict() for check in self.checks],
            "cards": self.cards,
            "behind": self.behind,
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "frames": self.frames,
            "thresholds": self.thresholds.to_dict(),
        }


def _decoded_count(source: Path) -> int:
    """Count decoded frames rather than trusting packet or container counts."""
    result = ffmpeg.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(source)],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    try:
        return int(json.loads(result.stdout)["streams"][0]["nb_read_frames"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValidationError(str(source), "video needs a decoded frame count") from exc


def _validate_plate_cards(cards: Sequence[CardOverlay], info: ffmpeg.MediaInfo) -> None:
    """Match the compositor's box, window, and alpha-layer validation."""
    assert info.width is not None and info.height is not None and info.video_frames is not None
    if not cards:
        return
    formats = ffmpeg.run(
        ["ffprobe", "-v", "error", "-show_pixel_formats", "-show_entries",
         "pixel_format=name:pixel_format_flags=alpha", "-of", "json"],
        timeout_s=ffmpeg.PROBE_TIMEOUT_S,
    )
    alpha_formats = {
        item["name"] for item in json.loads(formats.stdout)["pixel_formats"]
        if item.get("flags", {}).get("alpha")
    }
    for index, card in enumerate(cards):
        location = f"cards[{index}]"
        for coordinate, size, limit in (
            ("x", card.width, info.width), ("y", card.height, info.height)
        ):
            if getattr(card, coordinate) + size > limit:
                raise ValidationError(
                    f"{location}.{coordinate}", "card box extends outside the frame"
                )
        if card.start_frame >= info.video_frames:
            raise ValidationError(f"{location}.start_frame", "must be before the plate's end")
        if card.end_frame > info.video_frames:
            raise ValidationError(f"{location}.end_frame", "window extends past the plate's end")
        try:
            layer = ffmpeg.probe(paths.validate_input(card.layer), count_frames=True)
        except InputError as exc:
            raise ValidationError(f"{location}.layer", exc.reason) from exc
        if layer.pix_fmt not in alpha_formats:
            raise ValidationError(f"{location}.layer", "layer must have an alpha channel")
        if (layer.width, layer.height) != (card.width, card.height):
            raise ValidationError(f"{location}.layer", "layer dimensions must equal the card box")
        if layer.video_frames is None or layer.video_frames < card.end_frame - card.start_frame:
            raise ValidationError(f"{location}.layer", "layer is shorter than its frame window")


def _window_frames(
    source: Path, info: ffmpeg.MediaInfo, start: int, end: int, *,
    matte: bool = False, rgba: bool = False, layer_alpha: bool = False,
) -> Generator[np.ndarray, None, None]:
    """Yield a half-open frame window as native luma, matte alpha, or layer RGBA."""
    assert info.width is not None and info.height is not None
    conversion = (
        "format=rgba" if rgba else "alphaextract" if layer_alpha
        else "format=gray" if matte else "extractplanes=y"
    )
    shape = (info.height, info.width, 4) if rgba else (info.height, info.width)
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-map", "0:v:0", "-an",
        "-vf", f"trim=start_frame={start}:end_frame={end},{conversion}",
        "-fps_mode", "passthrough", "-pix_fmt", "rgba" if rgba else "gray",
        "-f", "rawvideo", "pipe:1",
    ]
    frame = np.empty(shape, dtype=np.uint8)
    target = memoryview(frame).cast("B")
    used = count = 0
    with closing(ffmpeg.run_stream(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)) as chunks:
        for chunk in chunks:
            raw = memoryview(chunk)
            while raw:
                size = min(len(raw), len(target) - used)
                target[used:used + size] = raw[:size]
                used += size
                raw = raw[size:]
                if used == len(target):
                    yield frame
                    count += 1
                    frame = np.empty(shape, dtype=np.uint8)
                    target = memoryview(frame).cast("B")
                    used = 0
    if used or count != end - start:
        raise ValidationError(str(source), f"decoded {count} frames in window [{start}, {end})")


def _difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(first.astype(np.int16) - second.astype(np.int16))


def measure_behind_layers(
    root: Path, matte: Path, info: ffmpeg.MediaInfo, layers: Sequence[dict[str, Any]], *,
    skipped_windows: Sequence[tuple[int, int]] = (),
    hidden_glyph_share: float = HIDDEN_GLYPH_SHARE,
    hidden_time_share: float = HIDDEN_TIME_SHARE,
    ink_alpha_threshold: int = GLYPH_ALPHA_THRESHOLD,
    require_ink: bool = False,
    report_worst: bool = False,
    covered_pixels: Callable[[np.ndarray, np.ndarray, int, int, int], int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Count hidden ink frames from layer RGBA and person alpha, streaming both.

    Panels use the cutout check's median-luma ink detection. Text on transparency
    keeps its alpha mask at ink_alpha_threshold, defaulting to 128.
    Off-canvas glyph pixels cannot be covered by the person. Zoom and insert
    windows restore no person, so they count as visible but never hidden.
    """
    assert info.width is not None and info.height is not None
    records = []
    for layer in layers:
        start, frames = layer["start_frame"], layer["frames"]
        x, y, width, height = (layer[key] for key in ("x", "y", "width", "height"))
        left, top = max(0, x), max(0, y)
        right, bottom = min(info.width, x + width), min(info.height, y + height)
        hidden = 0
        ink_frames = 0
        worst_visible = 1.0
        with ExitStack() as stack:
            person = stack.enter_context(closing(_window_frames(
                matte, info, start, start + frames, matte=True,
            )))
            glyphs = stack.enter_context(closing(_window_frames(
                root / layer["path"], replace(info, width=width, height=height),
                layer.get("layer_start_frame", 0),
                layer.get("layer_start_frame", 0) + frames, rgba=True,
            )))
            for frame, (alpha, pixels) in enumerate(zip(person, glyphs, strict=True), start=start):
                ink = _behind_ink(pixels, alpha_threshold=ink_alpha_threshold)
                total = int(np.count_nonzero(ink))
                if total:
                    ink_frames += 1
                if (not total or right <= left or bottom <= top
                        or any(first <= frame < last for first, last in skipped_windows)):
                    continue
                if covered_pixels is None:
                    count = covered_ink_pixels(ink, alpha, x, y)
                else:
                    count = covered_pixels(ink, alpha, frame, x, y)
                share = float(count / total)
                worst_visible = min(worst_visible, 1 - share)
                hidden += share > hidden_glyph_share
        if require_ink and not ink_frames:
            worst_visible = 0.0
            hidden = frames
        share = hidden / frames
        record = {"id": layer["id"], "start_frame": start,
                  "frames": frames, "hidden_frames": hidden,
                  "hidden_share": share, "flag": share > hidden_time_share}
        if report_worst:
            record["worst_visible_share"] = round(worst_visible, 4)
        records.append(record)
    return tuple(records)


def _ink(layer: np.ndarray, ink_diff: int, *, alpha_threshold: int = 250) -> np.ndarray:
    """Visible text, ticks, and rules differing from the panel's median luma."""
    opaque = layer[..., 3] >= alpha_threshold
    if not np.any(opaque):
        return opaque
    luma = np.dot(layer[..., :3], [299, 587, 114]) / 1000
    return opaque & (np.abs(luma - np.median(luma[opaque])) > ink_diff)


def _behind_ink(
    layer: np.ndarray, *, alpha_threshold: int = GLYPH_ALPHA_THRESHOLD,
) -> np.ndarray:
    """Exclude opaque panel backing while retaining glyphs on transparency.

    A full opaque layer is a panel. With transparent padding or rounded corners,
    a panel supplies continuous opaque spans enclosing its contrasting ink in
    both axes. Separate glyphs have gaps in those spans and keep their alpha.
    """
    visible = layer[..., 3] >= alpha_threshold
    ink = _ink(layer, DEFAULT_THRESHOLDS.ink_diff, alpha_threshold=alpha_threshold)
    if np.all(visible):
        return ink
    opaque = visible
    if not np.any(ink & opaque):
        return visible
    # Check only spans containing ink, so transparent padding and corners do not
    # turn an opaque panel into text on transparency.
    for backing, glyphs in ((opaque, ink), (opaque.T, ink.T)):
        for row in np.flatnonzero(np.any(glyphs, axis=1)):
            columns = np.flatnonzero(backing[row])
            if (not columns.size or not np.all(backing[row, columns[0]:columns[-1] + 1])
                    or glyphs[row, columns[0]] or glyphs[row, columns[-1]]):
                return visible
    return ink


def _solid(matte: np.ndarray) -> np.ndarray:
    """255 after a separable 5x5 minimum filter, with replicated frame borders."""
    height, width = matte.shape
    padded = np.pad(matte == 255, ((2, 2), (0, 0)), mode="edge")
    vertical = np.logical_and.reduce([padded[i:i + height] for i in range(5)])
    padded = np.pad(vertical, ((0, 0), (2, 2)), mode="edge")
    return np.logical_and.reduce([padded[:, i:i + width] for i in range(5)])


def _static(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Static 16x16 plate blocks, including partial blocks at the frame border."""
    difference = _difference(first, second)
    height, width = difference.shape
    rows, columns = np.arange(0, height, 16), np.arange(0, width, 16)
    totals = np.add.reduceat(np.add.reduceat(difference, rows, axis=0), columns, axis=1)
    maxima = np.maximum.reduceat(
        np.maximum.reduceat(difference, rows, axis=0), columns, axis=1
    )
    sizes = np.outer(np.minimum(16, height - rows), np.minimum(16, width - columns))
    blocks = (totals / sizes < 1) & (maxima < 8)
    return blocks.repeat(16, axis=0).repeat(16, axis=1)[:height, :width]


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else 1.0


def _clock(seconds: float) -> str:
    minutes, hundredths = divmod(round(seconds * 100), 6000)
    return f"{minutes}:{hundredths / 100:05.2f}"


def _measure_card(
    plate: Path, composite: Path, matte: Path, info: ffmpeg.MediaInfo,
    cards: Sequence[CardOverlay], card: CardOverlay, thresholds: CutoutThresholds,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Retain two preceding frames and scalar measurements for a card window."""
    assert info.fps is not None and info.width is not None and info.height is not None
    region = np.s_[card.y:card.y + card.height, card.x:card.x + card.width]
    edges: dict[str, Any] = {"card": card.id, "outside": None, "solid": None}
    occlusion: dict[str, Any] = {
        "card": card.id, "frame": card.start_frame, "time": float(card.start_frame / info.fps),
        "share": 0.0, "pixels": 0,
    }
    counts: list[int] = []
    dropouts: list[dict[str, Any]] = []
    previous_plate = previous_matte = older_plate = older_matte = None
    with ExitStack() as stack:
        streams = [stack.enter_context(closing(_window_frames(
            source, info, card.start_frame, card.end_frame, matte=is_matte
        ))) for source, is_matte in ((plate, False), (composite, False), (matte, True))]
        streams.append(stack.enter_context(closing(_window_frames(
            card.layer, replace(info, width=card.width, height=card.height),
            0, card.end_frame - card.start_frame, rgba=True,
        ))))
        layers = []
        for active in cards:
            if active.id == card.id:
                continue  # Reuse this card's RGBA alpha for its edge mask.
            start = max(card.start_frame, active.start_frame)
            end = min(card.end_frame, active.end_frame)
            if start < end:
                layer_frames = stack.enter_context(closing(_window_frames(
                    active.layer, replace(info, width=active.width, height=active.height),
                    start - active.start_frame, end - active.start_frame, layer_alpha=True,
                )))
                layers.append((active, layer_frames))
        for frame, (pixels, rendered, alpha, layer) in enumerate(
            zip(*streams, strict=True), start=card.start_frame
        ):
            ink = _ink(layer, thresholds.ink_diff)
            total = int(np.count_nonzero(ink))
            share = float(np.count_nonzero(ink & (alpha[region] >= 128)) / total) if total else 0.0
            if frame == card.start_frame or share > occlusion["share"]:
                occlusion.update(
                    frame=frame, time=float(frame / info.fps), share=share, pixels=total,
                )
            changed = _difference(pixels, rendered) > thresholds.edge_diff
            outside = np.ones(pixels.shape, dtype=bool)
            outside[region] &= layer[..., 3] == 0
            solid = _solid(alpha)
            for active, layer_frames in layers:
                if active.start_frame <= frame < active.end_frame:
                    box = np.s_[active.y:active.y + active.height,
                                active.x:active.x + active.width]
                    uncovered = next(layer_frames) == 0
                    outside[box] &= uncovered
                    if active.placement == "front":
                        solid[box] &= uncovered
                    if frame + 1 == min(card.end_frame, active.end_frame):
                        # Finish the stream's count/error checks and reap its decoder.
                        next(layer_frames, None)
            for part, difference, mask in (
                ("outside", changed, outside), ("solid", changed[region], solid[region])
            ):
                total = int(np.count_nonzero(mask))
                share = float(np.count_nonzero(difference & mask) / total) if total else 0.0
                if edges[part] is None or share > edges[part]["share"]:
                    edges[part] = {
                        "frame": frame, "time": float(frame / info.fps),
                        "share": share, "pixels": total,
                    }
            if previous_plate is not None and previous_matte is not None:
                static = _static(previous_plate, pixels)[region]
                toggled = (
                    _difference(previous_matte[region], alpha[region]) >= thresholds.flicker_px
                )
                counts.append(int(np.count_nonzero(static & toggled)))
            if older_plate is not None and older_matte is not None:
                assert previous_matte is not None
                before, middle, after = (
                    mask[region] >= 128 for mask in (older_matte, previous_matte, alpha)
                )
                dip = _iou(before, after) - min(_iou(before, middle), _iou(middle, after))
                motion = float(_difference(older_plate[region], pixels[region]).mean())
                if dip > thresholds.dropout_iou_dip and motion > thresholds.dropout_motion:
                    dropouts.append({"card": card.id, "frame": frame - 1,
                                     "time": float((frame - 1) / info.fps), "dip": dip,
                                     "motion": motion})
            older_plate, previous_plate = previous_plate, pixels
            older_matte, previous_matte = previous_matte, alpha
    worst = int(np.argmax(counts)) if counts else None
    pair = card.start_frame + worst if worst is not None else None
    flicker = {
        "card": card.id, "pairs": len(counts),
        "median": float(np.median(counts)) if counts else None,
        "p95": float(np.percentile(counts, 95)) if counts else None,
        "max": max(counts) if counts else None,
        "worst_pair": {"frames": [pair, pair + 1], "time": float((pair + 1) / info.fps)}
        if pair is not None else None,
    }
    return edges, flicker, dropouts, occlusion


def _pixel_checks(
    plate: Path, composite: Path, matte: Path, info: ffmpeg.MediaInfo,
    cards: Sequence[CardOverlay], thresholds: CutoutThresholds,
) -> tuple[CheckResult, ...]:
    edge_records, flicker_records, flags, occlusion_records = [], [], [], []
    for card in cards:
        if card.placement == "behind_head":
            edges, flicker, dropouts, occlusion = _measure_card(
                plate, composite, matte, info, cards, card, thresholds
            )
            edge_records.append(edges)
            flicker_records.append(flicker)
            flags.extend(dropouts)
            occlusion_records.append(occlusion)
    edges_fail = any(record[part]["share"] > thresholds.edge_share_max
                     for record in edge_records for part in ("outside", "solid"))
    edges_detail = "; ".join(
        f"{part} share {record[part]['share']:.6g} at frame {record[part]['frame']} "
        f"({_clock(record[part]['time'])}) in card {record['card']}"
        for record in edge_records for part in ("outside", "solid")
    ) + f" (limit {thresholds.edge_share_max:g}, luma difference > {thresholds.edge_diff})"
    measured = [record for record in flicker_records if record["pairs"]]
    flicker_fail = any(record["p95"] > thresholds.flicker_p95_max
                       or record["max"] > thresholds.flicker_frame_max for record in measured)
    flicker_detail = "; ".join(
        f"median {record['median']:g}, p95 {record['p95']:g}, max {record['max']} "
        f"static pixels per frame pair in card {record['card']}"
        if record["pairs"] else f"no consecutive frame pair in card {record['card']}"
        for record in flicker_records
    ) + f" (limits {thresholds.flicker_p95_max:g} and {thresholds.flicker_frame_max:g})"
    flags.sort(key=lambda flag: (flag["frame"], flag["card"]))
    dropout_detail = "; ".join(
        f"card {flag['card']} frame {flag['frame']} at {_clock(flag['time'])}, "
        f"dip {flag['dip']:.4g}"
        for flag in flags
    ) if flags else "no moving hand dropout flagged"
    triples = any(card.end_frame - card.start_frame >= 3 for card in cards
                  if card.placement == "behind_head")
    if not triples:
        dropout_detail = "no three-frame window for hand dropout review"
    occlusion_review = any(record["share"] > thresholds.occlusion_max
                           for record in occlusion_records)
    occlusion_detail = "; ".join(
        f"card {record['card']} hides {record['share']:.1%} of its text "
        f"at frame {record['frame']} ({_clock(record['time'])})"
        for record in occlusion_records
    ) + f" (limit {thresholds.occlusion_max * 100:g}%)"
    return (
        CheckResult("matte.doubled_edges", "fail" if edges_fail else "pass", edges_detail,
                    {"cards": edge_records}),
        CheckResult("matte.flicker", "fail" if flicker_fail else "pass" if measured else "skipped",
                    flicker_detail, {"cards": flicker_records}),
        CheckResult("matte.hand_dropout", "review" if flags else "pass" if triples else "skipped",
                    dropout_detail, {"flags": flags}),
        CheckResult("matte.occlusion", "review" if occlusion_review else "pass", occlusion_detail,
                    {"cards": occlusion_records}),
    )


def check_cutout(
    plate: Path, matte_dir: Path | None, composite: Path, cards: Sequence[CardOverlay],
    thresholds: CutoutThresholds = DEFAULT_THRESHOLDS,
) -> CutoutReport:
    """Validate media and evaluate behind-card cutouts without retaining videos."""
    if not isinstance(thresholds, CutoutThresholds):
        raise ValidationError("thresholds", "expected CutoutThresholds")
    cards = tuple(cards)
    _validate_cards(cards)
    active = peak = peak_frame = 0
    for frame, change in sorted(
        (frame, change) for card in cards
        for frame, change in ((card.start_frame, 1), (card.end_frame, -1))
    ):
        active += change
        if active > peak:
            peak, peak_frame = active, frame
    if peak > MAX_CARDS_AT_ONCE:
        raise ValidationError(
            "cards", f"{peak} cards active at frame {peak_frame} exceeds the limit of "
            f"{MAX_CARDS_AT_ONCE}; the timeline shows one graphic at a time; "
            "check fewer overlapping cards",
        )
    with paths.operation():
        source, rendered = paths.validate_input(plate), paths.validate_input(composite)
        info = ffmpeg.probe(source, count_frames=False)
        if info.has_video:
            info = replace(info, video_frames=_decoded_count(source))
        if (not info.has_video or not info.video_frames or not info.width or not info.height
                or info.fps is None or info.fps <= 0):
            raise InputError(
                str(plate), "cutout checking needs video with a known frame count and fps"
            )
        output = ffmpeg.probe(rendered, count_frames=False)
        if output.has_video:
            output = replace(output, video_frames=_decoded_count(rendered))
        for field in ("video_frames", "width", "height"):
            if getattr(output, field) != getattr(info, field):
                raise ValidationError(f"composite.{field}", "must equal the plate's")
        _validate_plate_cards(cards, info)
        behind = sum(card.placement == "behind_head" for card in cards)
        reason = None
        matte = None
        if not behind:
            reason = "no card sits behind the person"
        elif matte_dir is None or not (matte_dir / "matte.json").exists():
            reason = "person matte is missing"
        else:
            data = load_matte(paths.validate_input(matte_dir / "matte.json"))
            if data.status == "unavailable":
                reason = data.reason
            else:
                assert data.matte is not None
                if not (matte_dir / data.matte["file"]).exists():
                    reason = "person matte file is missing"
                else:
                    matte = _matte_path(source, info, matte_dir)
                    assert matte is not None
                    if _decoded_count(matte) != info.video_frames:
                        raise ValidationError(
                            "matte.matte.frames", "decoded count must equal the plate's frame count"
                        )
        if reason is not None:
            checks = tuple(CheckResult(id, "skipped", reason, {}) for id in _CHECK_IDS)
            return CutoutReport("skipped", reason, checks, len(cards), behind,
                                info.fps, info.video_frames, thresholds)
        assert matte is not None
        checks = (CheckResult("matte.frames", "pass",
                              f"{info.video_frames} frames, {info.width}x{info.height}, "
                              "plate and matte hashes match",
                              {"frames": info.video_frames, "width": info.width,
                               "height": info.height}),
                  *_pixel_checks(source, rendered, matte, info, cards, thresholds))
        status: CheckStatus = (
            "fail" if any(check.status == "fail" for check in checks)
            else "review" if any(check.status == "review" for check in checks) else "pass"
        )
        return CutoutReport(status, None, checks, len(cards), behind,
                            info.fps, info.video_frames, thresholds)
