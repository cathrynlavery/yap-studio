"""Strict, portable person matte documents and cleanup settings."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from yap_studio.errors import InputError, ValidationError
from yap_studio.faces.model import VideoIdentity

MATTE_SCHEMA = "yap-studio/matte@1"


def _object(value: Any, location: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValidationError(location, "expected an object with string keys")
    return value


def _field(obj: dict[str, Any], key: str, kind: type, location: str) -> Any:
    if key not in obj:
        raise ValidationError(f"{location}.{key}", "missing required key")
    value = obj[key]
    if type(value) is not kind:
        raise ValidationError(f"{location}.{key}", f"expected {kind.__name__}")
    return value


def _finite(value: Any, location: str) -> float:
    try:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError
        return float(value)
    except (ValueError, OverflowError):
        raise ValidationError(location, "expected a finite number") from None


def _hash(value: Any, location: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValidationError(location, "expected a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class Cleanup:
    """Temporal cleanup; feather_sigma blurs only unrefined Vision or HyperFrames alpha."""

    median: bool = True
    low: int = 8
    high: int = 247
    feather_sigma: float = 1.0
    refine: bool = True

    def __post_init__(self) -> None:
        if type(self.median) is not bool:
            raise ValidationError("cleanup.median", "expected bool")
        if type(self.refine) is not bool:
            raise ValidationError("cleanup.refine", "expected bool")
        for key in ("low", "high"):
            value = getattr(self, key)
            if type(value) is not int or not 0 <= value <= 255:
                raise ValidationError(f"cleanup.{key}", "expected an integer between 0 and 255")
        if self.low >= self.high:
            raise ValidationError("cleanup.high", "must be greater than low")
        sigma = _finite(self.feather_sigma, "cleanup.feather_sigma")
        if not 0 <= sigma <= 8:
            raise ValidationError("cleanup.feather_sigma", "must be between 0 and 8")

    def filter_chain(self, temporal: bool = True, *, levels: bool = True) -> str:
        """Return the count-preserving ffmpeg cleanup filter chain."""
        chain = []
        if self.median and temporal:
            chain.extend(["tpad=start=1:stop=1:start_mode=clone:stop_mode=clone",
                          "tmedian=radius=1"])
        if levels:
            chain.append(f"lut=c0='clip((val-{self.low})*255/({self.high}-{self.low}),0,255)'")
        if self.feather_sigma > 0:
            sigma = format(Decimal(str(self.feather_sigma)), "f")
            if "." in sigma:
                sigma = sigma.rstrip("0").rstrip(".")
            chain.append(f"gblur=sigma={sigma}")
        return ",".join(chain) or "null"

    def to_dict(self) -> dict[str, Any]:
        """Return all cleanup settings as JSON values."""
        return {
            "median": self.median,
            "low": self.low,
            "high": self.high,
            "feather_sigma": self.feather_sigma,
            "refine": self.refine,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cleanup:
        """Parse required settings without number or boolean coercion."""
        obj = _object(data, "cleanup")
        if "feather_sigma" not in obj:
            raise ValidationError("cleanup.feather_sigma", "missing required key")
        return cls(
            _field(obj, "median", bool, "cleanup"),
            _field(obj, "low", int, "cleanup"),
            _field(obj, "high", int, "cleanup"),
            _finite(obj["feather_sigma"], "cleanup.feather_sigma"),
            _field(obj, "refine", bool, "cleanup") if "refine" in obj else False,
        )


@dataclass(frozen=True)
class MatteData:
    """A measured gray matte with raw timing and coverage, or an unavailable result."""

    status: Literal["measured", "unavailable"]
    reason: str | None
    video: VideoIdentity
    backend: dict[str, Any]
    cleanup: Cleanup
    matte: dict[str, Any] | None
    times: tuple[float, ...] | None
    end_time: float | None
    coverage: tuple[float, ...] | None
    warnings: tuple[str, ...] = ()
    empty_frames: tuple[int, ...] = ()
    joins: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("measured", "unavailable"):
            raise ValidationError("matte.status", "expected measured or unavailable")
        if not isinstance(self.video, VideoIdentity):
            raise ValidationError("matte.video", "expected a VideoIdentity")
        if not isinstance(self.cleanup, Cleanup):
            raise ValidationError("matte.cleanup", "expected Cleanup")
        _object(self.backend, "matte.backend")
        if (
            type(self.joins) is not tuple
            or any(type(j) is not int or not 0 < j < self.video.frames for j in self.joins)
            or any(a >= b for a, b in zip(self.joins, self.joins[1:], strict=False))
        ):
            raise ValidationError("matte.joins", "expected sorted, unique interior frame indices")
        if type(self.warnings) is not tuple or any(type(w) is not str for w in self.warnings):
            raise ValidationError("matte.warnings", "expected a tuple of strings")
        if (
            type(self.empty_frames) is not tuple
            or any(type(i) is not int or not 0 <= i < self.video.frames for i in self.empty_frames)
            or any(a >= b for a, b in zip(self.empty_frames, self.empty_frames[1:], strict=False))
        ):
            raise ValidationError(
                "matte.empty_frames", "expected sorted, unique video frame indices"
            )
        if self.status == "unavailable":
            if type(self.reason) is not str or not self.reason.strip():
                raise ValidationError("matte.reason", "unavailable matte needs a non-empty reason")
            if self.backend:
                raise ValidationError("matte.backend", "unavailable matte needs an empty backend")
            if self.empty_frames:
                raise ValidationError("matte.empty_frames", "must be empty when unavailable")
            if self.joins:
                raise ValidationError("matte.joins", "must be empty when unavailable")
            for key in ("matte", "times", "end_time", "coverage"):
                if getattr(self, key) is not None:
                    raise ValidationError(f"matte.{key}", "must be null when unavailable")
            return
        if self.reason is not None:
            raise ValidationError("matte.reason", "must be null when measured")
        name = _field(self.backend, "name", str, "matte.backend")
        if not name.strip():
            raise ValidationError("matte.backend.name", "must not be empty")
        # Other backends may supply their own metadata without changing the schema.
        if name == "vision-person-segmentation":
            quality = _field(self.backend, "quality", str, "matte.backend")
            if quality not in ("accurate", "balanced"):
                raise ValidationError("matte.backend.quality", "expected accurate or balanced")
            for key in ("revision", "mask_width", "mask_height"):
                if _field(self.backend, key, int, "matte.backend") < 1:
                    raise ValidationError(f"matte.backend.{key}", "must be positive")
            if not _field(self.backend, "os", str, "matte.backend").strip():
                raise ValidationError("matte.backend.os", "must not be empty")
            _hash(self.backend.get("helper_sha256"), "matte.backend.helper_sha256")
        elif name == "hyperframes-remove-background":
            if not _field(self.backend, "version", str, "matte.backend").strip():
                raise ValidationError("matte.backend.version", "must not be empty")
            share = _finite(self.backend.get("alignment"), "matte.backend.alignment")
            if not 0 <= share <= 1:
                raise ValidationError("matte.backend.alignment", "must be between 0 and 1")
        artifact = _object(self.matte, "matte.matte")
        filename = _field(artifact, "file", str, "matte.matte")
        if not filename or filename in (".", "..") or "/" in filename or "\\" in filename:
            raise ValidationError("matte.matte.file", "expected a bare file name")
        _hash(artifact.get("sha256"), "matte.matte.sha256")
        for key, expected in (("codec", "ffv1"), ("pix_fmt", "gray")):
            if _field(artifact, key, str, "matte.matte") != expected:
                raise ValidationError(f"matte.matte.{key}", f"expected {expected}")
        for key in ("times", "coverage"):
            values = getattr(self, key)
            if type(values) is not tuple or len(values) != self.video.frames:
                raise ValidationError(f"matte.{key}", "expected one value per video frame")
            for index, value in enumerate(values):
                number = _finite(value, f"matte.{key}[{index}]")
                if key == "coverage" and not 0 <= number <= 1:
                    raise ValidationError(f"matte.coverage[{index}]", "must be between 0 and 1")
        assert self.times is not None
        if any(a > b for a, b in zip(self.times, self.times[1:], strict=False)):
            raise ValidationError("matte.times", "must be non-decreasing")
        if _finite(self.end_time, "matte.end_time") < self.times[-1]:
            raise ValidationError("matte.end_time", "must be at or after the last frame time")

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned matte.json layout."""
        return {
            "schema": MATTE_SCHEMA,
            "status": self.status,
            "reason": self.reason,
            "video": self.video.to_dict(),
            "backend": self.backend,
            "cleanup": self.cleanup.to_dict(),
            "matte": self.matte,
            "times": list(self.times) if self.times is not None else None,
            "end_time": self.end_time,
            "coverage": list(self.coverage) if self.coverage is not None else None,
            "warnings": list(self.warnings),
            "empty_frames": list(self.empty_frames),
            "joins": list(self.joins),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatteData:
        """Check the schema, required fields, metadata, and per-frame arrays."""
        obj = _object(data, "matte")
        if _field(obj, "schema", str, "matte") != MATTE_SCHEMA:
            raise ValidationError("matte.schema", f"expected {MATTE_SCHEMA}")
        for key in ("reason", "matte", "times", "end_time", "coverage"):
            if key not in obj:
                raise ValidationError(f"matte.{key}", "missing required key")
        if obj["reason"] is not None and type(obj["reason"]) is not str:
            raise ValidationError("matte.reason", "expected a string or null")
        arrays = {}
        for key in ("times", "coverage"):
            arrays[key] = tuple(_field(obj, key, list, "matte")) if obj[key] is not None else None
        warnings = _field(obj, "warnings", list, "matte")
        if any(type(w) is not str for w in warnings):
            raise ValidationError("matte.warnings", "expected strings")
        # Task 1 documents predate the guard and have no empty_frames key.
        empty_frames = obj.get("empty_frames", [])
        if type(empty_frames) is not list:
            raise ValidationError("matte.empty_frames", "expected a list of frame indices")
        joins = obj.get("joins", [])
        if type(joins) is not list:
            raise ValidationError("matte.joins", "expected a list of frame indices")
        return cls(
            _field(obj, "status", str, "matte"),
            obj["reason"],
            VideoIdentity.from_dict(_field(obj, "video", dict, "matte")),
            _field(obj, "backend", dict, "matte"),
            Cleanup.from_dict(_field(obj, "cleanup", dict, "matte")),
            obj["matte"],
            arrays["times"],
            obj["end_time"],
            arrays["coverage"],
            tuple(warnings),
            tuple(empty_frames),
            tuple(joins),
        )


def load_matte(path: Path) -> MatteData:
    """Read a matte document, identifying unreadable or invalid JSON by path."""
    try:
        content = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(str(path), f"cannot read matte JSON: {exc}") from exc
    except UnicodeError as exc:
        raise ValidationError(str(path), f"invalid matte JSON: {exc}") from exc
    try:
        return MatteData.from_dict(json.loads(content))
    except (ValueError, RecursionError) as exc:
        raise ValidationError(str(path), f"invalid matte JSON: {exc}") from exc
