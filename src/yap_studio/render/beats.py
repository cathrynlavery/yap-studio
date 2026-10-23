"""Detect source music beats in an isolated, disposable renderer project."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from html import escape
from pathlib import Path
from typing import Any

from yap_studio import paths, process
from yap_studio.errors import YapError
from yap_studio.render import hyperframes


class BeatsError(YapError):
    """Report failed beat analysis without preventing a music bed publication."""

    def __init__(self, message: str) -> None:
        super().__init__(
            f"Beat detection failed: {message}", error_type="processing_error",
            code="beats_failed",
            suggested_action="Check the source audio and pinned renderer with yap-studio doctor, "
                             "then run music again with --force.",
        )


def _number(value: Any, name: str) -> float:
    """Require a finite JSON number, excluding booleans and numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def detect_beats(
    audio: Path, *, duration_s: float, install: hyperframes.RendererInstall | None = None,
    timeout_s: float = 120,
) -> dict[str, Any]:
    """Return validated source-time beats using the pinned offline renderer.

    The supplied duration is the source audio length, before bed trimming. The
    renderer sees only a copy in a temporary directory and all its files are
    removed on success, invalid output, process failure or timeout.
    """
    try:
        duration = _number(duration_s, "audio duration")
        if duration <= 0:
            raise ValueError("audio duration must be positive")
        with paths.operation(), tempfile.TemporaryDirectory(prefix="yap-beats-") as folder:
            source = paths.validate_input(audio)
            root = Path(folder).resolve()
            local = root / f"music{source.suffix}"
            shutil.copyfile(source, local)
            paths.atomic_write_text(
                root / "index.html",
                '<!doctype html><html><body><div data-composition-id="root" '
                'data-width="1080" data-height="1920" '
                f'data-duration="{duration}" data-no-timeline>'
                '<audio id="music" data-timeline-role="music" '
                f'src="{escape(local.name, quote=True)}" data-start="0" '
                f'data-duration="{duration}"></audio></div></body></html>',
            )
            renderer = install if install is not None else hyperframes.ensure_installed()
            result = process.spawn(
                [str(renderer.executable), "beats", "--json", str(root)],
                timeout_s=timeout_s, tool="hyperframes", env=hyperframes.renderer_env(),
            )
            if result.returncode != 0:
                raise BeatsError(f"renderer exited with code {result.returncode}")
            summary = json.loads(result.stdout)
            relative = f"beats/{local.name}.json"
            if (not isinstance(summary, dict) or summary.get("ok") is not True
                    or summary.get("file") != relative):
                raise ValueError("renderer did not report the expected beat file")
            target = root / relative
            if not target.resolve().is_relative_to(root):
                raise ValueError("beat file is outside the temporary project")
            data = json.loads(target.read_text(encoding="utf-8"))
            if (not isinstance(data, dict) or type(data.get("version")) is not int
                    or data["version"] != 1 or data.get("audio") != local.name
                    or not isinstance(data.get("beats"), list)):
                raise ValueError("invalid beat file contract")
            beats = []
            previous = -1.0
            for item in data["beats"]:
                if not isinstance(item, dict):
                    raise ValueError("each beat must be an object")
                time = _number(item["time"], "beat time")
                strength = _number(item["strength"], "beat strength")
                if not 0 <= time <= duration or time <= previous:
                    raise ValueError("beat times must increase within the audio length")
                if not 0 <= strength <= 1:
                    raise ValueError("beat strength must be between 0 and 1")
                beats.append({"time_s": time, "strength": strength})
                previous = time
            if type(summary.get("count")) is not int or summary["count"] != len(beats):
                raise ValueError("renderer beat count does not match its file")
            bpm = summary.get("bpm")
            if bpm is not None:
                bpm = _number(bpm, "BPM")
                if bpm <= 0:
                    raise ValueError("BPM must be positive")
            return {"bpm": bpm, "beats": beats}
    except BeatsError:
        raise
    except (YapError, OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise BeatsError(str(exc)) from exc
