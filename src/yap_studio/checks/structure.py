"""Whole-file counts, loudness, and black or frozen picture checks."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import audio, ffmpeg
from yap_studio.checks.model import Finding, Thresholds, format_time
from yap_studio.edl.model import Still
from yap_studio.plan import Plan

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|inf|nan)"
_DETECTED_PARTS_MIN_S = 3.0


@dataclass(frozen=True)
class Interval:
    """A detected picture interval; a missing end means it continues to EOF."""

    start_s: float
    end_s: float | None
    duration_s: float | None


@dataclass(frozen=True)
class StructureResult:
    """Measured file structure and its findings."""

    integrated_lufs: float | None
    lra_lu: float | None
    true_peak_dbtp: float | None
    black: list[Interval]
    freeze: list[Interval]
    frames: int | None
    samples: int | None
    sample_rate: int | None
    video_duration_s: float | None
    audio_duration_s: float | None
    parts_lufs: list[float | None]
    findings: list[Finding]
    audio_timeline: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _intervals(stderr: str, prefix: str) -> list[Interval]:
    intervals: list[Interval] = []
    current: dict[str, float] = {}

    def finish() -> None:
        if "start" in current:
            start, end = current["start"], current.get("end")
            duration = current.get("duration") if end is not None else None
            if end is not None and duration is None:
                duration = end - start
            intervals.append(Interval(start, end, duration))

    for match in re.finditer(rf"\b{prefix}_(start|end|duration)[:=]\s*({_NUMBER})", stderr):
        key, value = match.group(1), float(match.group(2))
        if not math.isfinite(value):
            continue
        if key == "start":
            finish()
            current = {}
        current[key] = value
    finish()
    return intervals


def _loudness(stderr: str) -> tuple[float | None, float | None, float | None]:
    # Filter initialization can print an empty summary before the final one.
    summary = stderr.rsplit("Summary:", 1)[-1] if "Summary:" in stderr else ""

    def number(pattern: str) -> float | None:
        match = re.search(pattern, summary)
        value = float(match.group(1)) if match else None
        return value if value is not None and math.isfinite(value) else None

    return (
        number(rf"\bI:\s*({_NUMBER})\s+LUFS"),
        number(rf"\bLRA:\s*({_NUMBER})\s+LU"),
        number(rf"True peak:\s*Peak:\s*({_NUMBER})"),
    )


def _momentary(stderr: str) -> list[tuple[float, float]]:
    return [
        (float(match.group(1)), float(match.group(2)))
        for match in re.finditer(rf"\bt:\s*({_NUMBER})[^\n]*?\bM:\s*({_NUMBER})", stderr)
        if all(math.isfinite(float(value)) for value in match.groups())
    ]


def _part_loudness(
    blocks: Sequence[tuple[float, float]],
    parts: Sequence[tuple[float, float]],
    sample_rate: int | None,
    thresholds: Thresholds,
) -> list[float | None]:
    values: list[float | None] = []
    # ebur128 timestamps the last sample of each 400 ms block. Only use blocks
    # wholly inside a part so a loud neighbour cannot contaminate its result.
    epsilon = 2 / sample_rate if sample_rate else 0.0001
    for start, end in parts:
        energies = [
            10 ** (level / 10)
            for t, level in blocks
            if t - 0.4 >= start - epsilon and t <= end + epsilon and level > -70
        ]
        if end - start < thresholds.parts_min_s or not energies:
            values.append(None)
            continue
        relative_gate = sum(energies) / len(energies) / 10
        gated = [energy for energy in energies if energy >= relative_gate]
        values.append(10 * math.log10(sum(gated) / len(gated)))
    return values


def _decode_pass(render: Path, info: ffmpeg.MediaInfo) -> str:
    # Verbose includes the info-level summaries and the requested verbose
    # momentary lines. At loglevel info, framelog=verbose would hide the blocks.
    command = ["ffmpeg", "-loglevel", "verbose", "-nostats", "-i", str(render)]
    if info.video_index is not None:
        command.extend(["-map", f"0:{info.video_index}"])
    if info.audio_index is not None:
        command.extend(["-map", f"0:{info.audio_index}"])
    if info.has_video:
        minimum = 1 / float(info.fps) if info.fps else 0.0
        command.extend(
            [
                "-vf",
                f"blackdetect=d={minimum:.9f}:pix_th=0.10,freezedetect=n=-60dB:d=0.5",
            ]
        )
    if info.has_audio:
        command.extend(["-af", "ebur128=peak=true:framelog=verbose"])
    command.extend(["-f", "null", "-"])
    return ffmpeg.run(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S).stderr


def _packet_count(render: Path, stream_index: int) -> int | None:
    result = ffmpeg.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            str(stream_index),
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "json",
            str(render),
        ],
        timeout_s=ffmpeg.PROBE_TIMEOUT_S,
    )
    streams = json.loads(result.stdout).get("streams", [])
    value = streams[0].get("nb_read_packets") if streams else None
    return int(value) if value not in {None, "N/A", ""} else None


_DISCONTINUITY_EVIDENCE_LIMIT = 200


def _timestamp_findings(clock: audio.Timeline) -> list[Finding]:
    """Report every break in the audio timestamps as one whole-stream warning."""
    breaks = clock.discontinuities
    if not breaks:
        return []
    first = breaks[0]
    action = "Encode the audio once for the whole video and mux it again."
    if len(breaks) == 1:
        size = abs(first.samples)
        if first.samples > 0:
            detail = (
                f"the audio timestamps jump {size:,} samples ({abs(first.ms):.1f} ms) ahead. "
                "A player that follows them inserts silence here, which delays the rest "
                "of the sound."
            )
        else:
            detail = (
                f"the audio timestamps step back {size:,} samples ({abs(first.ms):.1f} ms). "
                "The decoded sound here overlaps the sound before it."
            )
        message = f"At {format_time(first.time_s)}, {detail} {action}"
    else:
        gaps = [item for item in breaks if item.samples > 0]
        overlaps = [item for item in breaks if item.samples < 0]
        parts = []
        if gaps:
            parts.append(
                f"{len(gaps)} {'gap' if len(gaps) == 1 else 'gaps'} "
                f"({clock.gap_samples * 1000 / clock.sample_rate:.1f} ms in all)"
            )
        if overlaps:
            parts.append(
                f"{len(overlaps)} {'overlap' if len(overlaps) == 1 else 'overlaps'} "
                f"({clock.overlap_samples * 1000 / clock.sample_rate:.1f} ms in all)"
            )
        message = (
            f"From {format_time(first.time_s)}, the audio timestamps break at "
            f"{len(breaks)} points: {' and '.join(parts)}. {action}"
        )
    rows = [
        {
            "time_s": item.time_s,
            "samples": item.samples,
            "ms": item.ms,
            "kind": "gap" if item.samples > 0 else "overlap",
            "frame": item.frame,
            "decoded_position": item.position,
        }
        for item in breaks[:_DISCONTINUITY_EVIDENCE_LIMIT]
    ]
    return [
        Finding(
            "structure.audio_timestamps",
            "warn",
            message,
            time_s=first.time_s,
            suggested_action=action,
            evidence={
                "discontinuities": rows,
                "count": len(breaks),
                "listed": len(rows),
                "sample_rate": clock.sample_rate,
                "tolerance_samples": clock.tolerance_samples,
                "decoded_samples": clock.decoded_samples,
                "timestamp_span_samples": clock.span_samples,
                "gap_samples": clock.gap_samples,
                "overlap_samples": clock.overlap_samples,
            },
        )
    ]


def _clock_evidence(clock: audio.Timeline | None) -> dict[str, Any]:
    """Separate decoded content, the timestamp span and breaks for length findings."""
    if clock is None:
        return {}
    evidence: dict[str, Any] = {
        "decoded_samples": clock.decoded_samples,
        "timestamp_span_samples": clock.span_samples,
        "gap_samples": clock.gap_samples,
        "overlap_samples": clock.overlap_samples,
        "filled_samples": clock.fill_samples,
        "timestamp_breaks": len(clock.discontinuities),
    }
    if clock.discontinuities:
        evidence["note"] = (
            "Decoded samples count the audio content and set the sample count. The "
            f"timestamp span differs by {clock.span_samples - clock.decoded_samples:+,} "
            "samples because the audio timestamps break; see structure.audio_timestamps. "
            f"The sound end is on the container clock, with {clock.fill_samples:,} samples "
            "of silence filling timestamp gaps."
        )
    return evidence


@dataclass(frozen=True)
class _HoldWindow:
    """A declared hold on the render's container clock."""

    index: int
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    sample_start: int
    sample_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_s": self.start_s,
            "end_s": self.end_s,
        }


# freezedetect's d=0.5 cannot report a still picture shorter than this.
_FREEZE_MINIMUM_S = 0.5
_HOLD_SILENCE_DBFS = -90.0
# Codec frames overlap by about 21 ms at 48 kHz; skip the hold's edges.
_HOLD_EDGE_S = 0.05
# A keyframe inside a hold can redraw the still picture with small rounding changes
# across the whole frame. Accept that one step only when it is small overall
# (freezedetect's measure) and no 32x32 luma or 16x16 chroma tile changes much,
# as it would for a caption or a cut.
_HOLD_STEP_LIMIT_DB = -45.0
_HOLD_STEP_TILE = 32
_HOLD_STEP_TILE_LIMIT = 12.0


@dataclass(frozen=True)
class _KeyframeStep:
    """The picture change at one keyframe."""

    time_s: float
    mafd_db: float
    tile_change: float
    accepted: bool

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence, including an identical pair."""
        return {**asdict(self), "mafd_db": self.mafd_db if math.isfinite(self.mafd_db) else None}


def _hold_segments(
    render: Path, info: ffmpeg.MediaInfo, hold: _HoldWindow, picture_end: float | None,
) -> list[Interval]:
    """Measure short still segments around a hold on the main clock."""
    if not info.fps or info.video_index is None:
        return []
    fps = float(info.fps)
    start = max(0.0, hold.start_s - 2 / fps)
    length = hold.end_s - hold.start_s + 4 / fps
    result = ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "info", "-nostats", "-ss", f"{start:.6f}",
            "-t", f"{length:.6f}", "-i", str(render), "-map", f"0:{info.video_index}",
            "-vf", f"freezedetect=n=-60dB:d={1.5 / fps:.9f}", "-f", "null", "-",
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    window_end = min(start + length, picture_end) if picture_end is not None else start + length
    segments = []
    for interval in _intervals(result.stderr, "freeze"):
        first = interval.start_s + start
        end = interval.end_s + start if interval.end_s is not None else window_end
        duration = interval.duration_s if interval.end_s is not None else end - first
        segments.append(Interval(first, end, duration))
    return segments


def _keyframe_times(render: Path, info: ffmpeg.MediaInfo) -> list[float]:
    """Read keyframe packet times on the format clock without decoding."""
    result = ffmpeg.run(
        [
            "ffprobe", "-v", "error", "-select_streams", str(info.video_index),
            "-show_entries", "packet=pts_time,flags", "-of", "json", str(render),
        ],
        timeout_s=ffmpeg.PROBE_TIMEOUT_S,
    )
    return [
        float(packet["pts_time"]) - info.format_start_s
        for packet in json.loads(result.stdout).get("packets", [])
        if "K" in packet.get("flags", "") and packet.get("pts_time") not in {None, "N/A", ""}
    ]


def _keyframe_step(render: Path, info: ffmpeg.MediaInfo, time_s: float) -> _KeyframeStep | None:
    """Find a keyframe by time and compare it with the preceding picture."""
    if not info.fps or not info.width or not info.height or info.video_index is None:
        return None
    fps = float(info.fps)
    start = max(0.0, time_s - max(4 / fps, 0.25))
    length = time_s - start + 0.5 / fps
    result = ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "info", "-nostats", "-ss", f"{start:.6f}",
            "-t", f"{length:.6f}", "-i", str(render), "-map", f"0:{info.video_index}",
            "-vf", "showinfo",
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
        binary=True,
    )
    width, height = info.width, info.height
    chroma_width, chroma_height = (width + 1) // 2, (height + 1) // 2
    samples = width * height + 2 * chroma_width * chroma_height
    stderr = (result.stderr.decode("utf-8", errors="replace")
              if isinstance(result.stderr, bytes) else result.stderr)
    times = []
    for line in stderr.splitlines():
        if "showinfo" not in line:
            continue
        match = re.search(rf"\bpts_time:\s*({_NUMBER})", line)
        if match and math.isfinite(pts := float(match.group(1))):
            times.append(start + pts)
    if len(result.stdout) != len(times) * samples:
        return None
    keyframe = min(range(len(times)), key=lambda frame: abs(times[frame] - time_s), default=0)
    if keyframe == 0 or abs(times[keyframe] - time_s) > 0.25 / fps:
        return None
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(len(times), samples)
    frames = frames[keyframe - 1:keyframe + 1].astype(np.int16)
    change = np.abs(frames[1] - frames[0])
    mean = float(np.mean(change)) / 256
    mafd_db = 20 * math.log10(mean) if mean else -math.inf
    tile_change, offset = 0.0, 0
    for plane_width, plane_height, tile in (
        (width, height, _HOLD_STEP_TILE),
        (chroma_width, chroma_height, _HOLD_STEP_TILE // 2),
        (chroma_width, chroma_height, _HOLD_STEP_TILE // 2),
    ):
        end = offset + plane_width * plane_height
        plane = change[offset:end].reshape(plane_height, plane_width)
        tile_change = max(tile_change, max(
            float(np.mean(plane[y:y + tile, x:x + tile]))
            for y in range(0, plane_height, tile)
            for x in range(0, plane_width, tile)
        ))
        offset = end
    return _KeyframeStep(
        time_s, mafd_db, tile_change,
        mafd_db <= _HOLD_STEP_LIMIT_DB and tile_change <= _HOLD_STEP_TILE_LIMIT,
    )


def _still_across_keyframes(
    hold: _HoldWindow,
    segments: Sequence[Interval],
    picture_end: float | None,
    tolerance: float,
    fps: float,
    keyframes: Callable[[], Sequence[float]],
    measure_step: Callable[[float], _KeyframeStep | None],
) -> tuple[bool, list[_KeyframeStep]]:
    """Chain still segments only across small changes at keyframes."""
    steps: list[_KeyframeStep] = []
    end = None
    times = None
    half_frame = 0.5 / fps + 1e-6
    for segment in segments:
        segment_end = segment.end_s if segment.end_s is not None else picture_end or 0.0
        if segment_end <= hold.start_s:
            continue
        if end is None:
            if segment.start_s > hold.start_s + tolerance + 1e-6:
                return False, steps
        else:
            if abs(segment.start_s - end) > half_frame:
                return False, steps
            if times is None:
                times = keyframes()
            if not any(abs(time - end) <= half_frame for time in times):
                return False, steps
            step = measure_step(end)
            if step is None:
                return False, steps
            steps.append(step)
            if not step.accepted:
                return False, steps
        end = segment_end
        if end >= hold.end_s - tolerance - 1e-6:
            return True, steps
    return False, steps


def _hold_windows(plan: Plan | None, info: ffmpeg.MediaInfo, sample_rate: int | None) -> list:
    if plan is None:
        return []
    rate = sample_rate or plan.sample_rate
    offset = round(info.audio_offset_s * rate)
    return [
        _HoldWindow(
            index=hold.index,
            start_frame=hold.out_f0,
            end_frame=hold.out_f1,
            start_s=info.video_offset_s + float(hold.out_time),
            end_s=info.video_offset_s + float(hold.end_time),
            sample_start=offset + math.floor(hold.sample_start * rate / plan.sample_rate + 0.5),
            sample_end=offset + math.floor(hold.sample_end * rate / plan.sample_rate + 0.5),
        )
        for hold in plan.holds
        if hold.audio == "silence"
    ]


def _inside_hold(
    start_s: float, end_s: float, holds: Sequence[_HoldWindow], tolerance: float
) -> _HoldWindow | None:
    return next(
        (
            hold
            for hold in holds
            if start_s >= hold.start_s - tolerance - 1e-6 and end_s <= hold.end_s + tolerance + 1e-6
        ),
        None,
    )


def _hold_findings(
    holds: Sequence[_HoldWindow],
    freeze: Sequence[Interval],
    picture_end: float | None,
    decoded: np.ndarray | None,
    sample_rate: int | None,
    fps: float | None,
    thresholds: Thresholds,
    segments: Callable[[_HoldWindow], Sequence[Interval]] | None = None,
    keyframes: Callable[[], Sequence[float]] | None = None,
    measure_step: Callable[[float], _KeyframeStep | None] | None = None,
) -> list[Finding]:
    """Check that each declared hold shows a still picture over silence."""
    findings: list[Finding] = []
    tolerance = thresholds.touch_cut_frames / fps if fps else 0.0
    for hold in holds:
        length = hold.end_s - hold.start_s
        label = f"At {format_time(hold.start_s)}"
        still = None
        if freeze is not None and length >= _FREEZE_MINIMUM_S:
            still = any(
                interval.start_s <= hold.start_s + tolerance + 1e-6
                and (interval.end_s if interval.end_s is not None else picture_end or 0.0)
                >= hold.end_s - tolerance - 1e-6
                for interval in freeze
            )
        hold_segments = []
        steps = []
        if still is False and fps and segments and keyframes and measure_step:
            hold_segments = segments(hold)
            still, steps = _still_across_keyframes(
                hold, hold_segments, picture_end, tolerance, fps, keyframes, measure_step,
            )
        peak_dbfs = None
        audible = False
        if decoded is not None and sample_rate:
            edge = round(_HOLD_EDGE_S * sample_rate)
            first, last = hold.sample_start + edge, hold.sample_end - edge
            window = decoded[max(0, first) : max(0, last)]
            if last > first and len(window):
                peak = float(np.max(np.abs(window)))
                peak_dbfs = 20 * math.log10(peak) if peak > 0 else None
                audible = peak_dbfs is not None and peak_dbfs > _HOLD_SILENCE_DBFS
        evidence = {
            "hold": hold.to_dict(),
            "still_picture": still,
            "freeze_intervals": [asdict(interval) for interval in freeze or ()],
            "hold_segments": [asdict(interval) for interval in hold_segments],
            "keyframe_steps": [step.to_dict() for step in steps],
            "peak_dbfs": peak_dbfs,
            "silence_limit_dbfs": _HOLD_SILENCE_DBFS,
            "edge_excluded_s": _HOLD_EDGE_S,
        }
        if still is False:
            action = "Render the hold as a still picture, or remove it from the edit list."
            findings.append(
                Finding(
                    "plan.holds",
                    "warn",
                    f"{label}, the edit list declares a {length:.2f} s hold (hold {hold.index}), "
                    f"but the picture keeps moving. {action}",
                    time_s=hold.start_s,
                    suggested_action=action,
                    evidence=evidence,
                )
            )
        elif still is True and steps:
            times = [format_time(step.time_s) for step in steps]
            label_times = (times[0] if len(times) == 1
                           else ", ".join(times[:-1]) + " and " + times[-1])
            noun = "keyframe" if len(times) == 1 else "keyframes"
            findings.append(
                Finding(
                    "plan.holds",
                    "info",
                    f"{label}, hold {hold.index} is still. The encoder redraws the picture "
                    f"at the {noun} at {label_times}, a small step spread across the frame.",
                    time_s=hold.start_s,
                    evidence=evidence,
                )
            )
        elif still is None and freeze is not None:
            findings.append(
                Finding(
                    "plan.holds",
                    "info",
                    f"{label}, hold {hold.index} is shorter than {_FREEZE_MINIMUM_S:g} s, "
                    "too short to confirm a still picture.",
                    time_s=hold.start_s,
                    evidence=evidence,
                )
            )
        if audible:
            action = "Render silence under the hold, or correct the edit list."
            findings.append(
                Finding(
                    "plan.holds",
                    "warn",
                    f"{label}, the edit list declares silence for hold {hold.index}, but the "
                    f"sound peaks at {peak_dbfs:.1f} dBFS. {action}",
                    time_s=hold.start_s,
                    suggested_action=action,
                    evidence=evidence,
                )
            )
    return findings


@dataclass
class _DeclaredPictureWindow:
    start_s: float
    end_s: float
    stills: list[Still]
    holds: list[_HoldWindow]


def _declared_picture_windows(
    stills: Sequence[Still], holds: Sequence[_HoldWindow], fps: float | None,
    picture_offset_s: float,
) -> list[_DeclaredPictureWindow]:
    windows = [
        _DeclaredPictureWindow(picture_offset_s + still.start_frame / fps,
                               picture_offset_s + still.end_frame / fps, [still], [])
        for still in stills
    ] if fps else []
    windows.extend(_DeclaredPictureWindow(hold.start_s, hold.end_s, [], [hold])
                   for hold in holds)
    windows.sort(key=lambda window: window.start_s)
    merged: list[_DeclaredPictureWindow] = []
    for window in windows:
        if merged and window.start_s <= merged[-1].end_s + 1e-6:
            merged[-1].end_s = max(merged[-1].end_s, window.end_s)
            merged[-1].stills.extend(window.stills)
            merged[-1].holds.extend(window.holds)
        else:
            merged.append(window)
    return merged


def _picture_findings(
    check: str,
    intervals: Sequence[Interval],
    cuts: Sequence[tuple[int, float]],
    duration: float | None,
    fps: float | None,
    thresholds: Thresholds,
    holds: Sequence[_HoldWindow] = (),
    *,
    stills: Sequence[Still] = (),
    picture_offset_s: float = 0.0,
) -> list[Finding]:
    findings: list[Finding] = []
    label = "black" if check == "picture.black" else "frozen"
    minimum = thresholds.black_warn_s if label == "black" else thresholds.freeze_warn_s
    tolerance = thresholds.touch_cut_frames / fps if fps else 0.0
    declared_windows = (_declared_picture_windows(stills, holds, fps, picture_offset_s)
                        if check == "picture.freeze" else [])
    for interval in intervals:
        end = interval.end_s if interval.end_s is not None else duration
        if end is None:
            continue
        length = interval.duration_s if interval.duration_s is not None else end - interval.start_s
        declared = next((window for window in declared_windows
                         if interval.start_s >= window.start_s - tolerance - 1e-6
                         and end <= window.end_s + tolerance + 1e-6), None)
        if declared is not None:
            declarations = []
            evidence: dict[str, Any] = {
                "start_s": interval.start_s, "end_s": interval.end_s,
                "duration_s": length, "runs_to_end": interval.end_s is None,
                "declared_span": {"start_s": declared.start_s, "end_s": declared.end_s},
            }
            if declared.stills:
                names = ", ".join(still.id for still in declared.stills)
                noun = "insert" if len(declared.stills) == 1 else "inserts"
                declarations.append(f"the timeline declares ({noun} {names})")
                key = "declared_still" if len(declared.stills) == 1 else "declared_stills"
                evidence[key] = (declared.stills[0].to_dict() if len(declared.stills) == 1
                                 else [still.to_dict() for still in declared.stills])
            if declared.holds:
                names = ", ".join(str(hold.index) for hold in declared.holds)
                noun = "hold" if len(declared.holds) == 1 else "holds"
                declarations.append(f"the edit list declares ({noun} {names})")
                key = "declared_hold" if len(declared.holds) == 1 else "declared_holds"
                evidence[key] = (declared.holds[0].to_dict() if len(declared.holds) == 1
                                 else [hold.to_dict() for hold in declared.holds])
            findings.append(Finding(
                check, "info",
                f"At {format_time(interval.start_s)}: the picture holds for {length:.2f} s, "
                f"as {' and '.join(declarations)}.",
                time_s=interval.start_s,
                evidence=evidence,
            ))
            continue
        touching = [
            (index, time)
            for index, time in cuts
            if interval.start_s - tolerance - 1e-6 <= time <= end + tolerance + 1e-6
        ]
        if not touching and length < minimum:
            continue
        cut = touching[0][0] if touching else None
        action = f"Replace the {label} picture or adjust the edit."
        location = f"Cut {cut} at " if cut is not None else "At "
        time = touching[0][1] if touching else interval.start_s
        findings.append(
            Finding(
                check,
                "fail" if touching else "warn",
                f"{location}{format_time(time)}: the picture is {label} for {length:.2f} s. "
                f"{action}",
                cut=cut,
                time_s=time,
                suggested_action=action,
                evidence={
                    "start_s": interval.start_s,
                    "end_s": interval.end_s,
                    "duration_s": length,
                    "runs_to_end": interval.end_s is None,
                },
            )
        )
    return findings


def check_structure(
    render: Path,
    *,
    plan: Plan | None = None,
    cuts_s: Sequence[float] | None = None,
    parts_s: Sequence[tuple[float, float]] | None = None,
    thresholds: Thresholds = Thresholds(),  # noqa: B008 - frozen value
) -> StructureResult:
    """Measure structure in one filter pass plus native audio-clock decoding."""
    info = ffmpeg.probe(render, count_frames=False)
    stderr = _decode_pass(render, info)
    integrated, lra, peak = _loudness(stderr) if info.has_audio else (None, None, None)
    black = _intervals(stderr, "black") if info.has_video else []
    freeze = _intervals(stderr, "freeze") if info.has_video else []
    frames = _packet_count(render, info.video_index) if info.video_index is not None else None
    sample_rate, samples, clock, decoded = None, None, None, None
    if info.has_audio:
        sample_rate, decoded = audio.decode(render)
        samples = len(decoded)
        clock = audio.timeline(render)
    holds = _hold_windows(plan, info, sample_rate)
    fps = float(info.fps) if info.fps else None
    video_duration = frames / fps if frames is not None and fps else None
    audio_duration = samples / sample_rate if samples is not None and sample_rate else None
    picture_end = (info.video_offset_s + video_duration) if video_duration is not None else None

    @cache
    def segments(hold: _HoldWindow) -> list[Interval]:
        return _hold_segments(render, info, hold, picture_end)

    @cache
    def keyframes() -> list[float]:
        return _keyframe_times(render, info)

    @cache
    def measure_step(time_s: float) -> _KeyframeStep | None:
        return _keyframe_step(render, info, time_s)

    sound_end = audio_duration
    duration = max(picture_end or 0.0, sound_end or 0.0)
    if plan is not None:
        output_cuts = [(cut.index, float(cut.out_time)) for cut in plan.cuts]
        picture_cuts = [
            (index, time + info.video_offset_s) for index, time in output_cuts
        ]
        loudness_cuts = [
            (index, time + info.audio_offset_s) for index, time in output_cuts
        ]
        parts = [
            (float(s.out_time), float(s.out_time + s.n_frames / plan.fps)) for s in plan.segments
        ]
        # ebur128 preserves the selected audio stream's timestamps on the
        # container clock, so plan windows use the audio offset.
        loudness_parts = [
            (start + info.audio_offset_s, end + info.audio_offset_s) for start, end in parts
        ]
    else:
        picture_cuts = list(enumerate(cuts_s if cuts_s is not None else (), 1))
        if parts_s is not None:
            parts = list(parts_s)
            if cuts_s is None:
                picture_cuts = [
                    (index, part[0]) for index, part in enumerate(parts[1:], 1)
                ]
        else:
            boundaries = [0.0, *(time for _, time in picture_cuts), duration]
            parts = list(zip(boundaries, boundaries[1:], strict=False))
        loudness_cuts = picture_cuts
        loudness_parts = parts
    parts_lufs = _part_loudness(_momentary(stderr), loudness_parts, sample_rate, thresholds)
    findings: list[Finding] = []
    if not info.has_audio or not info.has_video:
        missing = "audio" if not info.has_audio else "video"
        findings.append(
            Finding(
                "structure.av_length",
                "info",
                f"At 0:00.0, this file has no {missing}; checks requiring it were skipped.",
                time_s=0.0,
                evidence={
                    "has_audio": info.has_audio,
                    "has_video": info.has_video,
                    "skipped": (
                        [
                            "plan.samples",
                            "structure.audio_timestamps",
                            "loudness.integrated",
                            "loudness.true_peak",
                            "loudness.parts",
                            "structure.av_length",
                        ]
                        if not info.has_audio
                        else [
                            "plan.frames",
                            "picture.black",
                            "picture.freeze",
                            "structure.av_length",
                        ]
                    ),
                },
            )
        )
    if plan is not None:
        if frames is not None and abs(frames - plan.total_frames) > thresholds.frames_vs_plan_warn:
            difference = frames - plan.total_frames
            action = "Re-render to match the edit list's picture length."
            findings.append(
                Finding(
                    "plan.frames",
                    "warn",
                    f"At 0:00.0, the video has {abs(difference):,} "
                    f"{'more' if difference > 0 else 'fewer'} frames than the edit list plans. "
                    f"{action}",
                    time_s=0.0,
                    suggested_action=action,
                    evidence={
                        "expected": plan.total_frames,
                        "actual": frames,
                        "difference": difference,
                    },
                )
            )
        if samples is not None:
            # Compare on the decoded clock even if the plan uses another sample rate.
            expected = math.floor(plan.total_samples * sample_rate / plan.sample_rate + 0.5)
            # Count decoded content: silence the container clock inserts at a
            # timestamp gap is not audio the render holds.
            fill = clock.fill_samples if clock is not None else 0
            actual = samples - round(info.audio_offset_s * sample_rate) - fill
            difference = actual - expected
            if abs(difference) > sample_rate / float(plan.fps):
                offset = round(info.audio_offset_s * sample_rate)
                # A shortfall entirely inside a declared silent hold loses no sound.
                covering = (
                    next(
                        (
                            hold
                            for hold in holds
                            if hold.sample_start - offset <= actual
                            and expected <= hold.sample_end - offset
                        ),
                        None,
                    )
                    if difference < 0
                    else None
                )
                summary = (
                    f"At 0:00.0, the video has {abs(difference):,} "
                    f"{'more' if difference > 0 else 'fewer'} audio samples "
                    f"({abs(difference) / sample_rate:.3f} s) than the edit list plans."
                )
                evidence = {
                    "expected": expected,
                    "actual": actual,
                    "difference": difference,
                    "sample_rate": sample_rate,
                    "plan_sample_rate": plan.sample_rate,
                    **_clock_evidence(clock),
                }
                if covering is not None:
                    evidence["declared_hold"] = covering.to_dict()
                    findings.append(
                        Finding(
                            "plan.samples",
                            "info",
                            f"{summary} The missing samples fall inside the declared silent "
                            f"hold at {format_time(covering.start_s)}, so no sound is lost.",
                            time_s=0.0,
                            evidence=evidence,
                        )
                    )
                else:
                    action = "Re-render the audio once to match the edit list's length."
                    findings.append(
                        Finding(
                            "plan.samples",
                            "warn",
                            f"{summary} {action}",
                            time_s=0.0,
                            suggested_action=action,
                            evidence=evidence,
                        )
                    )
    if picture_end is not None and sound_end is not None and fps:
        difference_s = sound_end - picture_end
        if abs(difference_s) > 1 / fps:
            summary = (
                f"At 0:00.0, the sound ends {abs(difference_s):.3f} s "
                f"{'later' if difference_s > 0 else 'earlier'} than the picture."
            )
            evidence = {
                "video_duration_s": video_duration,
                "audio_duration_s": audio_duration,
                "picture_end_s": picture_end,
                "sound_end_s": sound_end,
                "difference_s": difference_s,
                **_clock_evidence(clock),
            }
            covering = (
                _inside_hold(sound_end, picture_end, holds, 1 / fps) if difference_s < 0 else None
            )
            if covering is not None:
                evidence["declared_hold"] = covering.to_dict()
                findings.append(
                    Finding(
                        "structure.av_length",
                        "info",
                        f"{summary} The picture runs on inside the declared silent hold at "
                        f"{format_time(covering.start_s)}, so no sound is lost.",
                        time_s=0.0,
                        evidence=evidence,
                    )
                )
            else:
                action = "Re-render picture and sound to the same length."
                findings.append(
                    Finding(
                        "structure.av_length",
                        "warn",
                        f"{summary} {action}",
                        time_s=0.0,
                        suggested_action=action,
                        evidence=evidence,
                    )
                )
    if clock is not None:
        findings.extend(_timestamp_findings(clock))
    if integrated is not None and abs(integrated - thresholds.loudness_target_lufs) > (
        thresholds.loudness_tolerance_lu
    ):
        action = f"Normalize the whole video to {thresholds.loudness_target_lufs:g} LUFS."
        findings.append(
            Finding(
                "loudness.integrated",
                "warn",
                f"At 0:00.0, overall loudness is {integrated:.1f} LUFS. {action}",
                time_s=0.0,
                suggested_action=action,
                evidence={
                    "integrated_lufs": integrated,
                    "target_lufs": thresholds.loudness_target_lufs,
                },
            )
        )
    if peak is not None and peak > thresholds.true_peak_max_dbtp:
        action = f"Limit the true peak to {thresholds.true_peak_max_dbtp:g} dBTP."
        findings.append(
            Finding(
                "loudness.true_peak",
                "warn",
                f"At 0:00.0, sound peaks at {peak:.1f} dBTP. {action}",
                time_s=0.0,
                suggested_action=action,
                evidence={"true_peak_dbtp": peak, "maximum_dbtp": thresholds.true_peak_max_dbtp},
            )
        )
    for index, (left, right) in enumerate(zip(parts_lufs, parts_lufs[1:], strict=False), 1):
        if plan is not None:
            left_source = plan.segments[index - 1].source_id
            right_source = plan.segments[index].source_id
            if left_source is None or right_source is None or left_source == right_source:
                continue
            reason = "different sources"
        else:
            minimum = max(_DETECTED_PARTS_MIN_S, thresholds.parts_min_s)
            left_part, right_part = parts[index - 1], parts[index]
            if left_part[1] - left_part[0] < minimum or right_part[1] - right_part[0] < minimum:
                continue
            reason = "parts at least 3 s"
        if left is None or right is None or abs(right - left) <= thresholds.parts_loudness_warn_lu:
            continue
        time = loudness_parts[index][0]
        cut = next(
            (number for number, t in loudness_cuts if abs(t - time) < 1e-6), index
        )
        action = "Normalize the loudness of this part to match the rest."
        findings.append(
            Finding(
                "loudness.parts",
                "warn",
                f"Cut {cut} at {format_time(time)}: the next part is {abs(right - left):.1f} LU "
                f"{'louder' if right > left else 'quieter'} than the one before it. {action}",
                cut=cut,
                time_s=time,
                suggested_action=action,
                evidence={
                    "reason": reason,
                    "left_lufs": left,
                    "right_lufs": right,
                    "difference_lu": right - left,
                    **({"left_source": left_source, "right_source": right_source}
                       if plan is not None else {}),
                },
            )
        )
    # Open-ended picture intervals end at the last picture, not at a longer audio tail.
    findings.extend(
        _picture_findings("picture.black", black, picture_cuts, picture_end, fps, thresholds)
    )
    findings.extend(
        _picture_findings(
            "picture.freeze", freeze, picture_cuts, picture_end, fps, thresholds, holds,
            stills=plan.stills if plan else (), picture_offset_s=info.video_offset_s,
        )
    )
    findings.extend(
        _hold_findings(
            holds,
            freeze if info.has_video else None,
            picture_end,
            decoded,
            sample_rate,
            fps,
            thresholds,
            segments,
            keyframes,
            measure_step,
        )
    )
    return StructureResult(
        integrated,
        lra,
        peak,
        black,
        freeze,
        frames,
        samples,
        sample_rate,
        video_duration,
        audio_duration,
        parts_lufs,
        findings,
        clock.to_dict() if clock is not None else None,
    )
