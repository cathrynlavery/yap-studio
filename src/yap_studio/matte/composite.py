"""Card overlays and person keys made only from the plate's own pixels."""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any, Literal

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import InputError, ProcessingError, ValidationError
from yap_studio.faces.model import Box
from yap_studio.matte import chroma
from yap_studio.matte.model import _field, _object, load_matte
from yap_studio.matte.timing import require_constant_rate

Placement = Literal["front", "behind_head"]
PLACEMENTS: tuple[str, ...] = ("front", "behind_head")
DEFAULT_PLACEMENT: Placement = "front"
CARDS_SCHEMA = "yap-studio/card-overlays@1"


@dataclass(frozen=True)
class CardOverlay:
    """An alpha layer at a pixel box in a half-open plate frame window."""

    id: str
    layer: Path
    x: int
    y: int
    width: int
    height: int
    start_frame: int
    end_frame: int
    placement: Placement = DEFAULT_PLACEMENT

    def box(self) -> Box:
        """Return the card's display rectangle."""
        return Box(self.x, self.y, self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        """Return a portable card record with its layer path as text."""
        return {
            "id": self.id,
            "layer": str(self.layer),
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "placement": self.placement,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, index: int = 0) -> CardOverlay:
        """Parse strict field types, reporting errors at cards[index].field."""
        location = f"cards[{index}]"
        obj = _object(data, location)
        keys = {
            "id", "layer", "x", "y", "width", "height",
            "start_frame", "end_frame", "placement",
        }
        for key in obj:
            if key not in keys:
                raise ValidationError(f"{location}.{key}", "unknown field")
        layer = _field(obj, "layer", str, location)
        if not layer.strip() or "\x00" in layer:
            raise ValidationError(f"{location}.layer", "expected a non-empty file path")
        card = cls(
            _field(obj, "id", str, location),
            Path(layer),
            *(_field(obj, key, int, location) for key in (
                "x", "y", "width", "height", "start_frame", "end_frame"
            )),
            _field(obj, "placement", str, location) if "placement" in obj else DEFAULT_PLACEMENT,
        )
        _validate_card(card, location)
        return card


def _validate_card(card: CardOverlay, location: str) -> None:
    if not isinstance(card, CardOverlay):
        raise ValidationError(location, "expected a CardOverlay")
    if type(card.id) is not str or not card.id.strip():
        raise ValidationError(f"{location}.id", "expected a non-empty string")
    if not isinstance(card.layer, Path) or "\x00" in str(card.layer):
        raise ValidationError(f"{location}.layer", "expected a file path")
    for key in ("x", "y", "width", "height"):
        value = getattr(card, key)
        minimum = 0 if key in ("x", "y") else 2
        if type(value) is not int or value < minimum or value % 2:
            raise ValidationError(f"{location}.{key}", f"expected an even integer >= {minimum}")
    if type(card.start_frame) is not int or card.start_frame < 0:
        raise ValidationError(f"{location}.start_frame", "expected an integer >= 0")
    if type(card.end_frame) is not int or card.end_frame <= card.start_frame:
        raise ValidationError(f"{location}.end_frame", "expected an integer > start_frame")
    if type(card.placement) is not str or card.placement not in PLACEMENTS:
        raise ValidationError(f"{location}.placement", "expected front or behind_head")


def _validate_cards(cards: Sequence[CardOverlay]) -> None:
    seen: set[str] = set()
    for index, card in enumerate(cards):
        location = f"cards[{index}]"
        _validate_card(card, location)
        if card.id in seen:
            raise ValidationError(f"{location}.id", "card ids must be unique")
        seen.add(card.id)


def load_cards(path: Path) -> tuple[CardOverlay, ...]:
    """Read a card document, resolving layer paths relative to its folder."""
    source = paths.validate_input(path)
    try:
        obj = _object(json.loads(source.read_text(encoding="utf-8")), "cards")
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValidationError(str(path), f"invalid cards JSON: {exc}") from exc
    except OSError as exc:
        raise InputError(str(path), f"cannot read cards JSON: {exc}") from exc
    if _field(obj, "schema", str, "cards") != CARDS_SCHEMA:
        raise ValidationError("cards.schema", f"expected {CARDS_SCHEMA}")
    cards = []
    for index, record in enumerate(_field(obj, "cards", list, "cards")):
        card = CardOverlay.from_dict(record, index=index)
        if not card.layer.is_absolute():
            card = replace(card, layer=path.absolute().parent / card.layer)
        cards.append(card)
    _validate_cards(cards)
    return tuple(cards)


def composite_graph(
    cards: Sequence[CardOverlay],
    *,
    fps: Fraction,
    matte_input: int | None,
    first_layer_input: int,
    start_s: Fraction = Fraction(0),
    cutout: dict[str, Any] | None = None,
) -> tuple[str, str, tuple[str, ...]]:
    """Return a deterministic filter graph, bare output label, and warnings.

    Input 0 is the plate. Card k uses first_layer_input + k. Person crops
    always branch from input 0, before any card has been placed on it.
    """
    _validate_cards(cards)
    if not isinstance(fps, Fraction) or fps <= 0:
        raise ValidationError("fps", "expected a positive Fraction")
    if not isinstance(start_s, Fraction):
        raise ValidationError("start_s", "expected a Fraction")
    if type(first_layer_input) is not int or first_layer_input < 1:
        raise ValidationError("first_layer_input", "expected an input index >= 1")
    if matte_input is not None and (
        type(matte_input) is not int
        or matte_input < 1
        or first_layer_input <= matte_input < first_layer_input + len(cards)
    ):
        raise ValidationError("matte_input", "expected a separate input index >= 1")
    if not cards:
        return "[0:v]null[out]", "out", ()
    behind = [i for i, card in enumerate(cards) if card.placement == "behind_head"]
    front = [i for i, card in enumerate(cards) if card.placement == "front"]
    keyed = behind if matte_input is not None else []
    filters: list[str] = []
    warnings: list[str] = []
    rate = f"{fps.numerator}/{fps.denominator}"
    # Recover the exact frame clock from millisecond-rounded container timestamps.
    # Keep the origin exactly representable even when it falls between frame ticks.
    clock = Fraction(
        gcd(fps.denominator * start_s.denominator, start_s.numerator * fps.numerator),
        fps.numerator * start_s.denominator,
    )
    time_base = f"{clock.numerator}/{clock.denominator}"
    start = f"({start_s.numerator}/{start_s.denominator})"
    if keyed:
        filters.append(
            f"[0:v]split={len(keyed) + 1}[plate]"
            + "".join(f"[person{i}]" for i in keyed)
        )
        filters.append(
            f"[{matte_input}:v]settb=AVTB,setpts=N/({rate})/TB+{start}/TB,"
            f"settb=expr={time_base},split={len(keyed)}"
            + "".join(f"[matte{i}]" for i in keyed)
        )
        current = "plate"
    else:
        current = "0:v"
    for index in (*behind, *front):
        card = cards[index]
        filters.append(
            f"[{first_layer_input + index}:v]"
            "settb=AVTB,"
            f"setpts=(N+{card.start_frame})/({rate})/TB+{start}/TB,"
            f"settb=expr={time_base}[layer{index}]"
        )
        # Containers may round the plate PTS to milliseconds around this frame clock.
        overlay = (
            f"overlay=x={card.x}:y={card.y}:eof_action=pass:"
            "ts_sync_mode=nearest:"
            f"enable='between(n,{card.start_frame},{card.end_frame - 1})'"
        )
        filters.append(f"[{current}][layer{index}]{overlay}[card{index}]")
        current = f"card{index}"
        if index in keyed:
            crop = f"crop={card.width}:{card.height}:{card.x}:{card.y}"
            person, mask = chroma.append_despill(
                filters, f"person{index}", f"matte{index}", cutout,
            )
            filters.extend([
                f"[{person}]{crop}[pixels{index}]",
                f"[{mask}]{crop},format=gray[alpha{index}]",
                f"[pixels{index}][alpha{index}]alphamerge=ts_sync_mode=nearest[key{index}]",
                f"[{current}][key{index}]{overlay}[keyed{index}]",
            ])
            current = f"keyed{index}"
        elif card.placement == "behind_head":
            warnings.append(
                f"person cutout unavailable; card {card.id} renders in front of the person"
            )
    filters.append(f"[{current}]null[out]")
    return ";".join(filters), "out", tuple(warnings)


@dataclass(frozen=True)
class CompositeResult:
    """A verified composite and the cards and fallback warnings used to make it."""

    path: Path
    frames: int
    cards: tuple[CardOverlay, ...]
    warnings: tuple[str, ...]
    matte_used: bool

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible result fields."""
        return {
            "path": str(self.path),
            "frames": self.frames,
            "cards": [card.to_dict() for card in self.cards],
            "warnings": list(self.warnings),
            "matte_used": self.matte_used,
        }


def _matte_path(plate: Path, info: ffmpeg.MediaInfo, folder: Path) -> Path | None:
    document = paths.validate_input(folder / "matte.json")
    data = load_matte(document)
    if data.status == "unavailable":
        return None
    if data.video.sha256 != receipt.sha256_file(plate):
        error = ValidationError("matte.video.sha256", "matte was made from a different video")
        error.suggested_action = (
            f"yap-studio matte make {shlex.quote(str(plate))} --out {shlex.quote(str(folder))}"
        )
        raise error
    if data.video.frames != info.video_frames:
        raise ValidationError("matte.video.frames", "must equal the plate's frame count")
    assert data.matte is not None
    source = paths.validate_input(folder / data.matte["file"])
    if receipt.sha256_file(source) != data.matte["sha256"]:
        raise ValidationError("matte.matte.sha256", "matte file hash differs from the document")
    matte_info = ffmpeg.probe(source, count_frames=True)
    if matte_info.video_frames != info.video_frames:
        raise ValidationError("matte.matte.frames", "must equal the plate's frame count")
    if (matte_info.width, matte_info.height) != (info.width, info.height):
        raise ValidationError("matte.matte", "matte dimensions must equal the plate's")
    if (matte_info.video_codec, matte_info.pix_fmt) != ("ffv1", "gray"):
        raise ValidationError("matte.matte", "expected an FFV1 gray video")
    return source


def _audio_options(
    plate: Path, info: ffmpeg.MediaInfo, out: Path
) -> tuple[list[str], tuple[str, ...]]:
    """Copy supported container codecs, using AAC for other audio."""
    ext = out.suffix.lower().lstrip(".")
    copy = ext == "mkv" or (
        ext in ("mp4", "mov") and info.audio_codec in {"aac", "alac", "mp3", "ac3", "eac3"}
    )
    if ext == "mp4" and info.audio_codec == "opus":
        # Query this muxer's actual support by remuxing one existing audio packet.
        try:
            ffmpeg.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(plate), "-map", "0:a:0",
                 "-c:a", "copy", "-frames:a", "1", "-f", "mp4", "-movflags", "empty_moov",
                 "pipe:1"], timeout_s=ffmpeg.PROBE_TIMEOUT_S, binary=True,
            )
            copy = True
        except ProcessingError:
            copy = False
    if not info.has_audio or copy:
        return ["-c:a", "copy"], ()
    return ["-c:a", "aac", "-b:a", "256k"], (f"audio re-encoded to AAC for {ext}",)


def composite_video(
    plate: Path,
    cards: Sequence[CardOverlay],
    out: Path,
    *,
    matte_dir: Path | None,
    lossless: bool = False,
) -> CompositeResult:
    """Validate inputs, render once with ffmpeg, and atomically publish the result.

    Plates must have BT.709 or unknown colour tags and limited or unknown range.
    Convert other plates to BT.709 limited range first; the cut already is.
    """
    cards = tuple(cards)
    _validate_cards(cards)
    with paths.operation():
        source = paths.validate_input(plate)
        info = ffmpeg.probe(source, count_frames=True)
        if (
            not info.has_video or not info.video_frames or not info.width or not info.height
            or info.fps is None or info.fps <= 0
        ):
            raise InputError(str(plate), "compositing needs video with a known frame count and fps")
        try:
            require_constant_rate(source, info)
        except InputError:
            raise ValidationError(
                str(plate), "composite needs a constant-rate plate; run it on the cut"
            ) from None
        for tag in ("color_primaries", "color_transfer", "color_space", "color_range"):
            value = getattr(info, tag)
            expected = "tv" if tag == "color_range" else "bt709"
            if value not in (None, "unknown", "unspecified", expected):
                raise ValidationError(
                    f"plate.{tag}",
                    f"{value}: convert to BT.709 limited range first; the cut already is",
                )
        layers = []
        alpha_formats = set()
        if cards:
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
            if card.x + card.width > info.width:
                raise ValidationError(f"{location}.x", "card box extends outside the frame")
            if card.y + card.height > info.height:
                raise ValidationError(f"{location}.y", "card box extends outside the frame")
            if card.start_frame >= info.video_frames:
                raise ValidationError(f"{location}.start_frame", "must be before the plate's end")
            if card.end_frame > info.video_frames:
                raise ValidationError(
                    f"{location}.end_frame", "window extends past the plate's end"
                )
            try:
                layer = paths.validate_input(card.layer)
                layer_info = ffmpeg.probe(layer, count_frames=True)
            except InputError as exc:
                raise ValidationError(f"{location}.layer", exc.reason) from exc
            if layer_info.pix_fmt not in alpha_formats:
                raise ValidationError(f"{location}.layer", "layer must have an alpha channel")
            if (layer_info.width, layer_info.height) != (card.width, card.height):
                raise ValidationError(
                    f"{location}.layer", "layer dimensions must equal the card box"
                )
            if (
                layer_info.video_frames is None
                or layer_info.video_frames < card.end_frame - card.start_frame
            ):
                raise ValidationError(f"{location}.layer", "layer is shorter than its frame window")
            layers.append(layer)
        matte = None
        if matte_dir is not None and any(card.placement == "behind_head" for card in cards):
            matte = _matte_path(source, info, matte_dir)
        graph, label, warnings = composite_graph(
            cards, fps=info.fps, matte_input=1 if matte is not None else None,
            first_layer_input=2 if matte is not None else 1,
            start_s=Fraction(str(info.video_start_s or 0)),
            cutout=chroma.cutout_metadata(matte),
        )
        # Relabel colour metadata without transforming the plate's pixel values.
        # Otherwise ffmpeg can insert a matrix conversion to satisfy output tags.
        graph += (
            f";[{label}]setparams=colorspace=bt709:color_primaries=bt709:"
            "color_trc=bt709:range=limited[encoded]"
        )
        audio, audio_warnings = _audio_options(source, info, out)
        warnings += audio_warnings
        command = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-copyts", "-i", str(source)]
        for path in ([matte] if matte is not None else []) + layers:
            command.extend(["-i", str(path)])
        command.extend([
            "-filter_complex", graph, "-map", "[encoded]", "-map", "0:a?", *audio,
            "-c:v", "libx264",
        ])
        command.extend(["-qp", "0"] if lossless else ["-preset", "medium", "-crf", "12"])
        command.extend([
            "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv", "-fps_mode", "passthrough",
            "-enc_time_base:v", "filter",
        ])
        with paths.atomic_output(out, allowed_existing_suffixes=frozenset()) as temporary:
            ffmpeg.run([*command, str(temporary)], timeout_s=120 + 0.5 * info.video_frames)
            result_info = ffmpeg.probe(temporary, count_frames=True)
            if result_info.video_frames != info.video_frames:
                raise ProcessingError(
                    "ffprobe", 0, "composite frame count differs from the plate's",
                    code="composite_frame_count",
                )
        return CompositeResult(Path(out), info.video_frames, cards, warnings, matte is not None)
