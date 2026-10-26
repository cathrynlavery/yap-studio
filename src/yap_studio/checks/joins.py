"""Verify audio, words, and boundary frames at cuts in a finished render."""

from __future__ import annotations

import logging
import math
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np

from yap_studio import audio, ffmpeg, video
from yap_studio.checks.flash import flash_at, flash_finding, reference_window
from yap_studio.checks.model import Finding, Thresholds, TransitionWindow, format_time
from yap_studio.checks.relisten import RelistenResult, relisten_at, relisten_findings
from yap_studio.checks.sync import _MIN_COMPARED, SegmentSync, SyncResult
from yap_studio.edl.model import PictureRegion
from yap_studio.errors import InputError, ValidationError, YapError
from yap_studio.grade import GradeSpec
from yap_studio.paths import atomic_output, validate_output_dir
from yap_studio.plan import Plan, PlanCut, PlanSegment
from yap_studio.transcript import Backend, Transcript, compare_words, norm_words, words_between

_LOGGER = logging.getLogger("yap_studio")

_WORD_FRAME_S = 0.01
_WORD_SEARCH_S = 0.6
_WORD_EXCLUDED_S = 0.8  # Hear the source speech on the excluded side of an edge.
_WORD_CLOCK_S = 0.4
_WORD_HEARD_EDGE_S = 0.08  # A heard word edge can be one token frame off either way.
_WORD_OVERLAP_S = 0.02
_WORD_JOIN_GAP_S = 0.03  # Bound the distance from a crossing to its shared syllable valley.
_WORD_PROMINENCE_DB = 4.0
_WORD_PEAK_DROP_DB = 20.0
_WORD_VALLEY_FRACTION = 0.2
IN_SOURCE_RATIO = 0.25  # A source match must stay small next to both flash changes.


@dataclass(frozen=True)
class WordsResult:
    """Audio agreement with the plan, with transcripts retained for word names.

    Transcript differences are evidence only. Source energy decides whether
    an identified planned boundary word was clipped by the edit.
    """

    status: Literal["pass", "fail", "skipped", "unverifiable"]
    reason: str | None
    expected: list[str]
    heard: list[str]
    diff: list[str]
    artifacts: list[str]


@dataclass(frozen=True)
class FramesResult:
    """Source matches and adjacency of the two rendered boundary frames."""

    status: Literal["pass", "fail", "skipped"]
    reason: str | None
    left_expected: int | None
    left_best: int | None
    right_expected: int | None
    right_best: int | None
    render_left: int | None
    render_right: int | None
    planned_gap_frames: int
    distances: dict


@dataclass(frozen=True)
class CutResult:
    """Measurements for one planned cut on the render's container clock."""

    index: int
    out_time_s: float
    j_end_s: float
    j_start_s: float
    landing: Literal["measured", "planned"]
    gap_ms: float | None
    click_db: float | None
    click_peak_dbfs: float | None
    click_floor_dbfs: float
    click_db_control: float | None
    pause_s: float | None
    level_left_db: float | None
    level_right_db: float | None
    words: WordsResult
    frames: FramesResult
    flash: dict | None = None
    relisten: RelistenResult | None = None
    click_source: Literal["voice_stem", "mix"] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class JoinsResult:
    """Per-cut measurements and actionable findings."""

    cuts: list[CutResult]
    findings: list[Finding]
    transitions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _sample(t: float, sr: int) -> int:
    return math.floor(t * sr + 0.5)


def _click_window(
    render_audio: np.ndarray, sr: int, t: float, search_s: float
) -> tuple[np.ndarray, int, int]:
    """Slice the shared click search and reference, retaining local candidate indices."""
    if sr <= 0 or not math.isfinite(search_s) or search_s < 0:
        raise ValidationError("click", "rate must be positive and search non-negative")
    if not math.isfinite(t) or render_audio.ndim != 1 or not len(render_audio):
        raise ValidationError("click", "requires a finite time and non-empty mono audio")
    j = _sample(t, sr)
    if not 0 <= j < len(render_audio):
        raise ValidationError("t", "must index the render audio")
    radius = _sample(search_s, sr)
    first, last = max(0, j - radius), min(len(render_audio) - 1, j + radius)
    # click_db only uses 100 ms on each side. Slice once to avoid differencing
    # an entire feature-length render for every candidate sample.
    low = max(0, first - int(0.1 * sr))
    high = min(len(render_audio), last + int(0.1 * sr) + 2)
    local = render_audio[low:high]
    return local, first - low, last - low


def click_at(render_audio: np.ndarray, sr: int, t: float, *, search_s: float = 0.002) -> float:
    """Measure the largest click within a sample-rounded radius of container time t."""
    local, first, last = _click_window(render_audio, sr, t, search_s)
    return max(audio.click_db(local, index, sr) for index in range(first, last + 1))


def click_peak_dbfs_at(
    render_audio: np.ndarray, sr: int, t: float, *, search_s: float = 0.002
) -> float:
    """Measure peak sample-step dBFS over the click search, flooring silence at -240 dBFS."""
    local, first, last = _click_window(render_audio, sr, t, search_s)
    difference = np.abs(np.diff(local.astype(np.float64)))
    # Each candidate in click_at includes its own 2 ms sharpness window.
    radius = int(0.002 * sr)
    near = difference[max(0, first - radius) : min(len(difference), last + radius)]
    peak = float(np.max(near, initial=0.0))
    return 20 * math.log10(max(peak, 1e-12))


def pause_len_at(render_audio: np.ndarray, sr: int, t: float) -> float:
    """Measure the quiet run touching container time t."""
    return audio.pause_at(render_audio, _sample(t, sr), sr)


def _sound_ends_before_cut(
    render_audio: np.ndarray, sr: int, *landings: float, click_radius_s: float = 0.002
) -> bool:
    """Return whether a landing or its click radius reaches past the sound."""
    radius = _sample(click_radius_s, sr)
    return any(_sample(landing, sr) + radius >= len(render_audio) for landing in landings)


# A frame counts as repeated when its step to a neighbour is essentially zero, or
# is under a quarter of the median step between neighbouring frames on the same
# side of the cut, and under the old fixed 0.02 limit. Talking-head footage moves
# about 0.013 to 0.017 per frame at 64 px, so a fixed limit alone reads ordinary
# motion as a repeat; a repeat re-encoded by a codec measures about 0.0005 to 0.005.
REPEAT_IDENTICAL = 0.002
REPEAT_RATIO = 0.25
REPEAT_MAX = 0.02


def _median_step(frames: np.ndarray, first: int, last: int) -> float | None:
    """Median distance of the neighbouring pairs ending at frames first..last."""
    steps = [
        video.frame_distance(frames[index - 1], frames[index])
        for index in range(max(1, first), min(len(frames), last + 1))
    ]
    return float(np.median(steps)) if len(steps) >= 3 else None


def frame_step_at(render_frames: np.ndarray, frame: int, *, radius: int = 15) -> dict:
    """Report repeated neighbours and a black frame at a rendered cut index.

    ``frame`` is the first frame of the new shot. The step into it is judged
    against the outgoing shot's motion over ``radius`` frames, and the step out
    of it against the incoming shot's. At the first or last frame, only
    available neighbours are compared.
    """
    if render_frames.ndim != 3 or not 0 <= frame < len(render_frames):
        raise ValidationError("frame", "must index a grey frame array")
    pairs = {"left": (frame - 1, frame), "right": (frame, frame + 1)}
    distances = {
        name: video.frame_distance(render_frames[a], render_frames[b])
        for name, (a, b) in pairs.items()
        if 0 <= a and b < len(render_frames)
    }
    baselines = {
        "left": _median_step(render_frames, frame - radius, frame - 1),
        "right": _median_step(render_frames, frame + 2, frame + radius + 1),
    }

    def repeated(name: str) -> bool:
        value, baseline = distances[name], baselines[name]
        return value <= REPEAT_IDENTICAL or (
            baseline is not None and value < REPEAT_MAX and value < REPEAT_RATIO * baseline
        )

    return {
        "duplicate": any(repeated(name) for name in distances),
        "black": bool(float(np.asarray(render_frames[frame], dtype=np.float32).mean()) < 16),
        "distances": distances,
        "baselines": {name: baselines[name] for name in distances},
        "rule": {
            "identical_below": REPEAT_IDENTICAL,
            "ratio_to_local_motion": REPEAT_RATIO,
            "maximum": REPEAT_MAX,
        },
    }


@dataclass(frozen=True)
class _Media:
    """Keep a decoded picture and its frame grid together with the sound."""

    info: ffmpeg.MediaInfo
    sound: np.ndarray | None
    frames: np.ndarray | None
    rate: Fraction | None = None


def _landing(
    segment: SegmentSync, neighbour: SegmentSync, planned: float, edge: Literal["head", "tail"]
) -> tuple[float, dict]:
    value = segment.head_s if edge == "head" else segment.tail_s
    corr = segment.head_corr if edge == "head" else segment.tail_corr
    offset = segment.audio_vs_plan_ms
    if offset is not None and math.isfinite(offset):
        offset_source = "own"
    elif neighbour.audio_vs_plan_ms is not None and math.isfinite(
        neighbour.audio_vs_plan_ms
    ):
        offset = neighbour.audio_vs_plan_ms
        offset_source = "neighbour"
    else:
        offset = None
        offset_source = "none"
    expected = planned + (offset or 0.0) / 1000
    reason = None
    if value is None or not math.isfinite(value):
        reason = "landing missing"
    elif corr is None or not math.isfinite(corr) or corr < 0.8:
        reason = "landing correlation below 0.8"
    elif abs(value - expected) > 0.1:
        reason = "landing more than 0.1 s from the measured audio offset"
    return (expected if reason else value), {
        "reason": reason,
        "measured_s": value if value is not None and math.isfinite(value) else None,
        "corr": corr if corr is not None and math.isfinite(corr) else None,
        "expected_s": expected,
        "offset_source": offset_source,
        "offset_ms": offset,
    }


def _render_frame(
    segment: PlanSegment,
    measured: SegmentSync,
    source_frame: int,
    source_fps: float,
    plan_fps: float,
    render_fps: float,
) -> int:
    # The picture landing was measured at its probe frame, normally the middle.
    probe = (
        measured.probe_frame
        if measured.probe_frame is not None
        else segment.src_f0 + segment.n_frames // 2
    )
    reference = math.floor(probe / plan_fps * source_fps + 0.5)
    if measured.video_frame is not None:
        position = measured.video_frame + (source_frame - reference) / source_fps * render_fps
    else:
        position = (
            float(segment.out_time) + source_frame / source_fps - segment.src_f0 / plan_fps
        ) * render_fps
    return math.floor(position + 0.5)


def _planned_gap_s(cut: PlanCut, fps: float) -> float:
    frames = cut.right.out_f0 - (cut.left.out_f0 + cut.left.n_frames)
    return max(0, frames) / fps


def _frame_match(
    rendered: _Media,
    source: _Media | None,
    row: SegmentSync,
    wanted: int,
    index: int,
    excluded: np.ndarray | None = None,
) -> dict:
    """Compare a render frame with the planned source and its nearby candidates."""
    result = {
        "checked": False,
        "reason": None,
        "planned_frame": wanted,
        "best_frame": None,
        "matched": None,
        "distances": {},
    }
    if source is None:
        return {**result, "reason": "source not found"}
    if source.frames is None:
        return {**result, "reason": "no source picture"}
    if rendered.frames is None:
        return {**result, "reason": "no render picture"}
    if not 0 <= wanted < len(source.frames):
        return {**result, "reason": "planned source frame not found"}
    if not 0 <= index < len(rendered.frames):
        return {**result, "reason": "render frame not found"}
    mapping = row.mapping
    if mapping is None:
        if source.frames.shape[1:] != rendered.frames.shape[1:]:
            return {**result, "reason": "picture mapping not measured"}
        height, width = source.frames.shape[1:]
        mapping = video.Mapping("full", 1.0, 0.0, 0.0, width, height, 1.0)
    size = (max(1, round(mapping.height)), max(1, round(mapping.width)))
    indices = range(max(0, wanted - 3), min(len(source.frames), wanted + 4))
    candidates = video.apply_mapping(
        source.frames[list(indices)], mapping, role="source", size=size
    )
    picture = video.apply_mapping(rendered.frames[index], mapping, role="render", size=size)
    compared = None if excluded is None else video.valid_in_comparison(excluded, mapping, size)
    if compared is not None and float(np.mean(compared)) < _MIN_COMPARED:
        return {**result, "reason": "picture masked"}
    errors = {
        f: video.frame_distance(picture, candidate, compared)
        for f, candidate in zip(indices, candidates, strict=True)
    }
    best = min(errors, key=errors.get)
    return {
        **result,
        "checked": True,
        "best_frame": best,
        "matched": best == wanted or abs(errors[best] - errors[wanted]) < 0.02,
        "distances": errors,
    }


def _frames(
    cut: PlanCut,
    left_sync: SegmentSync,
    right_sync: SegmentSync,
    rendered: _Media,
    left: _Media | None,
    right: _Media | None,
    fps: float,
    masks: Sequence[tuple[Fraction, Fraction]],
    regions: Sequence[PictureRegion] = (),
) -> FramesResult:
    rate = float(rendered.info.fps or fps)
    source_rates = [
        float(media.rate or media.info.fps or fps) if media else fps for media in (left, right)
    ]
    expected = tuple(
        math.floor(frame / fps * source_rate + 0.5)
        for frame, source_rate in zip(
            (cut.left.src_f1 - 1, cut.right.src_f0), source_rates, strict=True
        )
    )
    positions = (
        _render_frame(cut.left, left_sync, expected[0], source_rates[0], fps, rate),
        _render_frame(cut.right, right_sync, expected[1], source_rates[1], fps, rate),
    )
    planned_gap_frames = round(_planned_gap_s(cut, fps) * rate)
    result = FramesResult(
        "skipped",
        None,
        expected[0],
        None,
        expected[1],
        None,
        *positions,
        planned_gap_frames,
        {},
    )
    start = rendered.info.video_offset_s + min(positions) / rate
    end = rendered.info.video_offset_s + (max(positions) + 1) / rate
    if any(float(a) < end and float(b) > start for a, b in masks):
        return replace(result, reason="picture masked")
    if left is None or right is None or left.frames is None or right.frames is None:
        return replace(result, reason="source not found")
    if rendered.frames is None:
        return replace(result, reason="no render picture")
    if any(not 0 <= index < len(rendered.frames) for index in positions):
        return replace(result, status="fail", reason="a frame is missing at the join")
    rects = [
        region.rect
        for region in regions
        if float(region.start) < end and float(region.end) > start
    ]
    excluded = (
        video.exclusion_mask(
            rendered.frames.shape[1:],
            rects,
            rendered.info.width or rendered.frames.shape[2],
            rendered.info.height or rendered.frames.shape[1],
        )
        if rects
        else None
    )
    bests, matched, distances = [], [], {}
    for role, source, row, wanted, index in zip(
        ("left", "right"),
        (left, right),
        (left_sync, right_sync),
        expected,
        positions,
        strict=True,
    ):
        comparison = _frame_match(rendered, source, row, wanted, index, excluded)
        if comparison["reason"]:
            return replace(result, reason=comparison["reason"])
        bests.append(comparison["best_frame"])
        matched.append(comparison["matched"])
        distances[role] = comparison["distances"]
    step = positions[1] - positions[0]
    expected_step = 1 + planned_gap_frames
    reason = None
    if step > expected_step:
        reason = "a frame is repeated at the join"
    elif step < expected_step:
        reason = "a frame is missing at the join"
    elif not all(matched):
        reason = "the wrong frame shows at the join"
    return replace(
        result,
        status="fail" if reason else "pass",
        reason=reason,
        left_best=bests[0],
        right_best=bests[1],
        distances=distances,
    )


def _transition(
    window: TransitionWindow,
    plan: Plan,
    rendered: _Media,
    source: Callable[[str | None], _Media | None],
    rows: Mapping[int, SegmentSync],
    masks: Sequence[tuple[Fraction, Fraction]],
    regions: Sequence[PictureRegion],
) -> dict:
    """Check exact picture length and the planned frames outside a transition."""
    actual = len(rendered.frames) if rendered.frames is not None else None
    result = {**asdict(window), "expected_frames": plan.total_frames, "actual_frames": actual,
              "boundaries": {}, "status": "pass", "reason": None}
    problems, unavailable = [], []
    if not any(cut.out_frame == window.join_frame for cut in plan.cuts):
        problems.append(f"declared join frame {window.join_frame} is not a planned join")
    if actual is None:
        unavailable.append("no render picture")
    elif actual != plan.total_frames:
        problems.append(f"expected {plan.total_frames} frames, found {actual}")
    fps = float(plan.fps)
    for side, index in (("before", window.start_frame - 1), ("after", window.end_frame)):
        segment = next((item for item in plan.segments
                        if item.out_f0 <= index < item.out_f0 + item.n_frames), None)
        if segment is None:
            reason = f"{side} frame {index} is outside the planned content"
            unavailable.append(reason)
            result["boundaries"][side] = {"render_frame": index, "checked": False,
                                          "reason": reason, "matched": None}
            continue
        media = source(segment.source_id)
        source_rate = float(media.rate or media.info.fps or fps) if media else fps
        wanted = math.floor((segment.src_f0 + index - segment.out_f0) / fps * source_rate + 0.5)
        row = rows.get(segment.index, SegmentSync(segment.index, segment.source_id))
        start = rendered.info.video_offset_s + index / fps
        end = start + 1 / fps
        excluded = None
        rects = [region.rect for region in regions
                 if float(region.start) < end and float(region.end) > start]
        if rects and rendered.frames is not None:
            excluded = video.exclusion_mask(
                rendered.frames.shape[1:], rects,
                rendered.info.width or rendered.frames.shape[2],
                rendered.info.height or rendered.frames.shape[1],
            )
        comparison = _frame_match(rendered, media, row, wanted, index, excluded)
        if any(float(a) < end and float(b) > start for a, b in masks):
            comparison.update(checked=False, reason="picture masked", matched=None)
        result["boundaries"][side] = {"render_frame": index, **comparison}
        if comparison["reason"] == "render frame not found":
            problems.append(f"{side} frame {index} is missing")
        elif comparison["reason"]:
            unavailable.append(f"{side}: {comparison['reason']}")
        elif not comparison["matched"]:
            problems.append(f"{side} frame {index} does not match the cut")
    if problems or unavailable:
        result.update(status="fail" if problems else "skipped",
                      reason="; ".join([*problems, *unavailable]))
    return result


def _flash(
    rendered: _Media,
    frame: int,
    fps: float,
    touch_frames: int,
    masks: Sequence[tuple[Fraction, Fraction]],
    regions: Sequence[PictureRegion],
    transitions: Sequence[TransitionWindow] = (),
) -> dict:
    """Check the shared picture decode, excluding masks over the whole window."""
    if rendered.frames is None:
        return {"flash": False, "reason": "no render picture"}
    first, stop = reference_window(
        len(rendered.frames), frame, fps=fps, touch_frames=touch_frames
    )
    start = rendered.info.video_offset_s + first / fps
    end = rendered.info.video_offset_s + stop / fps
    if any(float(a) < end and float(b) > start for a, b in masks):
        return {"flash": False, "reason": "picture masked"}
    rects = [
        region.rect
        for region in regions
        if float(region.start) < end and float(region.end) > start
    ]
    valid = None
    if rects:
        valid = ~video.exclusion_mask(
            rendered.frames.shape[1:],
            rects,
            rendered.info.width or rendered.frames.shape[2],
            rendered.info.height or rendered.frames.shape[1],
        )
        if float(np.mean(valid)) < _MIN_COMPARED:
            return {"flash": False, "reason": "picture masked"}
    return flash_at(rendered.frames, frame, fps=fps, touch_frames=touch_frames, valid=valid,
                    ignored_windows=[(item.start_frame, item.end_frame) for item in transitions])


def _flash_in_source(
    flash: dict,
    cut: PlanCut,
    frame: int,
    rendered: _Media,
    left: _Media | None,
    right: _Media | None,
    left_sync: SegmentSync,
    right_sync: SegmentSync,
    fps: float,
    regions: Sequence[PictureRegion],
) -> dict:
    """Confirm whether every flash frame is the picture planned at that point."""
    rate = float(rendered.info.fps or fps)
    start = rendered.info.video_offset_s + flash["window"]["first_frame"] / rate
    end = rendered.info.video_offset_s + (flash["window"]["last_frame"] + 1) / rate
    rects = [
        region.rect
        for region in regions
        if float(region.start) < end and float(region.end) > start
    ]
    excluded = (
        video.exclusion_mask(
            rendered.frames.shape[1:],
            rects,
            rendered.info.width or rendered.frames.shape[2],
            rendered.info.height or rendered.frames.shape[1],
        )
        if rects
        else None
    )
    limit = IN_SOURCE_RATIO * min(change["distance"] for change in flash["changes"])
    matches = []
    for index in range(flash["run"]["first_frame"], flash["run"]["last_frame"] + 1):
        if index < frame:
            role, source, row = "left", left, left_sync
            planned = cut.left.src_f1 - 1 - (frame - 1 - index)
        else:
            role, source, row = "right", right, right_sync
            planned = cut.right.src_f0 + (index - frame)
        source_rate = float(source.rate or source.info.fps or fps) if source else fps
        wanted = math.floor(planned / fps * source_rate + 0.5)
        comparison = _frame_match(rendered, source, row, wanted, index, excluded)
        if comparison["checked"]:
            comparison["matched"] = (
                comparison["matched"] and comparison["distances"][wanted] <= limit
            )
        matches.append(
            {
                "render_frame": index,
                "side": role,
                "in_source_ratio": IN_SOURCE_RATIO,
                "in_source_limit": limit,
                **comparison,
            }
        )
    return {
        **flash,
        "source_checked": all(match["checked"] for match in matches),
        "in_source": all(match["matched"] is True for match in matches),
        "source_matches": matches,
    }


def _slice(sound: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Keep the requested clock, padding beyond the recording with silence."""
    first, last = _sample(start, sr), _sample(end, sr)
    out = np.zeros(max(0, last - first), dtype=np.float32)
    low, high = max(0, first), min(len(sound), last)
    if high > low:
        out[low - first : high - first] = sound[low:high]
    return out


def _quiet_edge(sound: np.ndarray, sr: int, nominal: float) -> float:
    """Find the quietest centered 20 ms window within 300 ms of an edge."""
    half = max(1, _sample(0.01, sr))
    first = max(half, _sample(nominal - 0.3, sr))
    last = min(len(sound) - half, _sample(nominal + 0.3, sr))
    if first > last:
        return min(max(nominal, 0.0), len(sound) / sr)
    region = sound[first - half : last + half].astype(np.float64)
    energy = np.concatenate(([0.0], np.cumsum(region * region)))
    levels = energy[2 * half :] - energy[: -2 * half]
    # Prefer the nominal edge when a silent interval gives identical minima.
    ties = np.flatnonzero(levels <= levels.min() + 1e-15) + first
    chosen = ties[np.argmin(np.abs(ties - _sample(nominal, sr)))]
    return int(chosen) / sr


def _write_wav(path: Path, sound: np.ndarray, sr: int) -> None:
    """Write an analysis WAV with a one-second lead and half-second silent tail."""
    padded = np.concatenate((np.zeros(sr), sound, np.zeros(_sample(0.5, sr))))
    pcm = (np.clip(padded, -1, 1) * 32767).round().astype("<i2")
    with atomic_output(path, allowed_existing_suffixes=frozenset({".wav"})) as temporary:
        with wave.open(str(temporary), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(pcm.tobytes())


@dataclass(frozen=True)
class _WordTrial:
    """Keep render fidelity windows and unpadded source controls on their own clocks."""

    paths: list[Path]
    artifacts: list[str]
    a: float
    b: float
    quiet: bool
    audio: dict
    controls: list[np.ndarray]
    control_levels: list[dict]
    clock_start_s: float
    gaps: list[tuple[float, float]]


def _mono_16k(sound: np.ndarray, sr: int) -> np.ndarray:
    """Downmix and resample a short analysis window, filtering before decimation."""
    mono = np.asarray(sound, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if sr == 16000 or not len(mono):
        return mono
    if sr > 16000:
        positions = np.arange(-48, 49)
        cutoff = 0.45 * 16000 / sr
        kernel = 2 * cutoff * np.sinc(2 * cutoff * positions) * np.hanning(len(positions))
        mono = np.convolve(np.pad(mono, (48, 48)), kernel / kernel.sum(), mode="valid")
    return np.interp(
        np.arange(_sample(len(mono) / sr, 16000)) * sr / 16000, np.arange(len(mono)), mono
    )


def _band_frames(sound: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return 40 mel-band powers and RMS dB for 25 ms frames at a 10 ms hop."""
    frames = np.lib.stride_tricks.sliding_window_view(sound, 400)[::160]
    power = abs(np.fft.rfft(frames * np.hanning(400), n=512)) ** 2 / 512**2
    frequencies = np.fft.rfftfreq(512, 1 / 16000)[:, None]
    edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 42) / 2595) - 1)
    rising = (frequencies - edges[:-2]) / (edges[1:-1] - edges[:-2])
    falling = (edges[2:] - frequencies) / (edges[2:] - edges[1:-1])
    bands = np.maximum(0, np.minimum(rising, falling))
    bands /= bands.sum(axis=0)
    energy = 10 * np.log10(np.maximum(np.mean(frames**2, axis=1), 1e-12))
    return power @ bands, energy


def _audio_mismatches(
    plan: np.ndarray, render: np.ndarray, sr: int, *, join_s: float | None = None
) -> dict:
    """Find sustained spectral changes, tolerating codec noise and brief gaps."""
    rate, hop, frame = 16000, 160, 400
    # Clean AAC stays below 2 dB; allow a 6 dB level change while detecting
    # the muted-speech fixture's sustained distance above 10 dB.
    threshold, minimum = 8.0, 0.06
    planned, rendered = _mono_16k(plan, sr), _mono_16k(render, sr)
    lag_s, corr = (
        audio.lag(planned, rendered, rate, max_lag_s=0.02)
        if len(planned) and len(rendered)
        else (0.0, 0.0)
    )
    if not np.any(planned) or not np.any(rendered):
        lag_s = 0.0
    shift = _sample(lag_s, rate)
    first, other = max(0, -shift), max(0, shift)
    count = min(len(planned) - first, len(rendered) - other)
    result = {
        "lag_s": lag_s,
        "alignment_corr": corr,
        "constant_gain_db": 0.0,
        "threshold_db": threshold,
        "minimum_run_s": minimum,
        "speech_gate_db": -60.0,
        "peak_distance_db": 0.0,
        "sustained_distance_db": 0.0,
        "longest_run_s": 0.0,
        "join_peak_distance_db": 0.0,
        "join_sustained_distance_db": 0.0,
        "join_longest_run_s": 0.0,
        "mismatches": [],
    }
    if count < frame:
        return result
    planned_window = planned[first : first + count]
    rendered_window = rendered[other : other + count]
    expected, plan_energy = _band_frames(planned_window)
    heard, render_energy = _band_frames(rendered_window)
    # Final delivery may apply one project-wide loudness gain. Fit and remove one
    # least-squares scalar for this comparison window only: missing audio, a different
    # spectrum, and a gain change inside the window still leave a sustained mismatch.
    # A whole-window level error is loudness.parts job, not this checks.
    denominator = float(np.dot(planned_window, planned_window))
    gain = float(np.dot(planned_window, rendered_window) / denominator) if denominator else 1.0
    gain = gain if math.isfinite(gain) and gain > 0 else 1.0
    constant_gain_db = 20 * math.log10(gain)
    heard /= gain**2
    render_energy -= constant_gain_db
    result["constant_gain_db"] = constant_gain_db
    floor = max(1e-12, float(np.percentile(expected.mean(axis=1), 95)) * 1e-4)
    distance = np.mean(
        abs(10 * np.log10(np.maximum(expected, floor)) - 10 * np.log10(np.maximum(heard, floor))),
        axis=1,
    )
    gate = max(-60.0, float(np.percentile(plan_energy, 95)) - 30)
    active = (plan_energy > gate) | (render_energy > gate)
    changed = active & (distance > threshold)
    edges = np.diff(np.r_[False, changed, False].astype(int))
    result.update(
        speech_gate_db=gate,
        peak_distance_db=float(distance[active].max()) if active.any() else 0.0,
    )
    run_frames = round(minimum * rate / hop)
    times = (first + np.arange(len(distance)) * hop) / rate
    near = (
        np.ones(len(times), dtype=bool)
        if join_s is None
        else ((times < join_s + 0.4) & (times + frame / rate > join_s - 0.4))
    )
    result["join_peak_distance_db"] = (
        float(distance[active & near].max()) if (active & near).any() else 0.0
    )
    if len(distance) >= run_frames:
        sustained = np.lib.stride_tricks.sliding_window_view(
            np.where(active, distance, 0), run_frames
        ).min(axis=1)
        result["sustained_distance_db"] = float(sustained.max())
        touching = (
            np.ones(len(sustained), dtype=bool)
            if join_s is None
            else (
                (times[: len(sustained)] < join_s + 0.4)
                & (times[: len(sustained)] + ((run_frames - 1) * hop + frame) / rate > join_s - 0.4)
            )
        )
        result["join_sustained_distance_db"] = (
            float(sustained[touching].max()) if touching.any() else 0.0
        )
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        duration = (int(end) - int(start)) * hop / rate
        result["longest_run_s"] = max(result["longest_run_s"], duration)
        if near[start:end].any():
            result["join_longest_run_s"] = max(result["join_longest_run_s"], duration)
        if duration >= minimum:
            result["mismatches"].append(
                {
                    "start_s": (first + int(start) * hop) / rate,
                    "end_s": (first + (int(end) - 1) * hop + frame) / rate,
                    "peak_distance_db": float(distance[start:end].max()),
                    "plan_energy_db": float(plan_energy[start:end].max()),
                    "render_energy_db": float(render_energy[start:end].max()),
                }
            )
    return result


def _control_levels(sound: np.ndarray, sr: int, point: float) -> dict:
    """Reject word timestamps that extend into silence at a planned cut."""
    hop = max(1, _sample(0.01, sr))
    frames = sound[: len(sound) // hop * hop].reshape(-1, hop).astype(np.float64)
    energy = 10 * np.log10(np.maximum(np.mean(frames**2, axis=1), 1e-12))
    return {
        "before_db": audio.rms_db(sound, point - 0.015, sr, 0.015),
        "after_db": audio.rms_db(sound, point + 0.015, sr, 0.015),
        "gate_db": max(-55.0, float(np.percentile(energy, 95)) - 30),
    }


def _short_gaps(plan: Plan, rows: Mapping[int, SegmentSync]) -> list[tuple[float, float]]:
    """Remove only independently measured insertions shorter than the words limit."""
    gaps = []
    for cut in plan.cuts:
        left, right = rows.get(cut.left.index), rows.get(cut.right.index)
        if left is None or right is None:
            continue
        planned_gap = _planned_gap_s(cut, float(plan.fps))
        end_planned = float(cut.out_time) - planned_gap
        end, end_evidence = _landing(left, right, end_planned, "tail")
        start, start_evidence = _landing(right, left, float(cut.out_time), "head")
        extra_gap = start - end - planned_gap
        if (
            not end_evidence["reason"]
            and not start_evidence["reason"]
            and 0.001 < extra_gap < 0.06
        ):
            gaps.append((end + planned_gap, start))
    return sorted(gaps)


def _restore_gaps(time: float, gaps: Sequence[tuple[float, float]]) -> float:
    """Map a time with short insertions removed back to the render clock."""
    for start, end in gaps:
        if time < start:
            break
        time += end - start
    return time


def _comparison_window(
    sound: np.ndarray,
    sr: int,
    join: float,
    a: float,
    count: int,
    gaps: Sequence[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    """Compare content on a clock excluding measured sub-60 ms insertions.

    Removing a 32 ms insertion also removes its displacement of later words.
    The original audio, WAV artifacts, landings and gap finding stay intact.
    """
    intervals = [(_sample(start, sr), _sample(end, sr)) for start, end in gaps]
    at = _sample(join, sr)
    at -= sum(max(0, min(at, end) - start) for start, end in intervals)
    first = at - _sample(a, sr)
    positions = np.arange(first, first + count)
    for start, end in intervals:
        positions += np.where(positions >= start, end - start, 0)
    window = np.zeros(count, dtype=np.float32)
    valid = (positions >= 0) & (positions < len(sound))
    window[valid] = sound[positions[valid]]
    return window, first / sr


def _plan_window(
    plan: Plan,
    source: Callable[[str | None], _Media | None],
    sr: int,
    start: float,
    end: float,
) -> tuple[np.ndarray | None, str | None]:
    """Splice every planned piece on the requested output sample clock."""
    first, last = _sample(start, sr), _sample(end, sr)
    window = np.zeros(max(0, last - first), dtype=np.float32)
    boundaries = {0, len(window)}
    for segment in plan.segments:
        segment_start = _sample(float(segment.out_time), sr)
        segment_end = _sample(float(segment.out_time + segment.n_frames / plan.fps), sr)
        low, high = max(first, segment_start), min(last, segment_end)
        if high <= low:
            continue
        boundaries.update((low - first, high - first))
        if segment.source_id is None:
            continue
        media = source(segment.source_id)
        if media is None:
            return None, "source not found"
        if media.sound is None:
            return None, "no audio in the source"
        source_start = _sample(media.info.video_offset_s + segment.src_f0 / float(plan.fps), sr)
        source_start += low - segment_start
        window[low - first : high - first] = _slice(
            media.sound, sr, source_start / sr, (source_start + high - low) / sr
        )
    edges = sorted(boundaries)
    pieces = list(zip(edges, edges[1:], strict=False))
    return audio.splice(window, pieces, fade=int(0.005 * sr))[0], None


def _word_trial(
    cut: PlanCut,
    plan: Plan,
    sound: np.ndarray,
    sr: int,
    join: float,
    source: Callable[[str | None], _Media | None],
    work_dir: Path,
    artifacts_dir: Path | None,
    gaps: Sequence[tuple[float, float]] = (),
) -> tuple[_WordTrial | None, str | None]:
    window_start = min(join, _quiet_edge(sound, sr, join - 2.0))
    window_end = max(join, _quiet_edge(sound, sr, join + 2.0))
    a, b = join - window_start, window_end - join
    planned, reason = _plan_window(
        plan, source, sr, float(cut.out_time) - a, float(cut.out_time) + b
    )
    if planned is None:
        return None, reason
    controls = []
    for segment, edge, before, after in (
        (cut.left, cut.left.src_f1, 1.5, 1.0),
        (cut.right, cut.right.src_f0, 1.0, 1.5),
    ):
        media = source(segment.source_id)
        if segment.source_id is None:
            controls.append(np.zeros(_sample(before + after, sr), dtype=np.float32))
            continue
        if media is None:
            return None, "source not found"
        if media.sound is None:
            return None, "no audio in the source"
        point = media.info.video_offset_s + edge / float(plan.fps)
        controls.append(_slice(media.sound, sr, point - before, point + after))
    windows = (
        _slice(sound, sr, window_start, window_end),
        planned,
        *controls,
    )
    paths, artifacts = [], []
    for role, window in zip(("render", "plan", "left", "right"), windows, strict=True):
        name = f"cut-{cut.index:02d}-{role}.wav"
        path = work_dir / name
        _write_wav(path, window, sr)
        paths.append(path)
        if artifacts_dir is not None:
            artifact = artifacts_dir / name
            if artifact.resolve() != path.resolve():
                _write_wav(artifact, window, sr)
            artifacts.append(str(artifact))
    half = len(planned) / sr / 2
    quiet = not len(planned) or audio.rms_db(planned, half, sr, half) < -45
    compared, clock_start = _comparison_window(sound, sr, join, a, len(planned), gaps)
    comparison = _audio_mismatches(planned, compared, sr, join_s=a)
    comparison["ignored_gaps"] = [{"start_s": start, "end_s": end} for start, end in gaps]
    return _WordTrial(
        paths,
        artifacts,
        a,
        b,
        quiet,
        comparison,
        controls,
        [
            _control_levels(window, sr, point)
            for window, point in zip(controls, (1.5, 1.0), strict=True)
        ],
        clock_start,
        list(gaps),
    ), None


def _words(
    trial: _WordTrial, transcripts: Mapping[Path, Transcript] | None
) -> tuple[WordsResult, list[dict], dict]:
    shifted = [
        replace(
            transcripts[path] if transcripts is not None else Transcript([]),
            words=[
                replace(word, start=word.start - 1, end=word.end - 1)
                for word in (transcripts[path].words if transcripts is not None else [])
            ],
        )
        for path in trial.paths
    ]
    render, plan, left, right = shifted
    expected = norm_words(
        " ".join(word.text for word in words_between(plan, 0.0, trial.a + trial.b))
    )
    heard = norm_words(
        " ".join(word.text for word in words_between(render, 0.0, trial.a + trial.b))
    )
    comparison = compare_words(expected, heard, fillers_optional=True)
    mismatches = [
        dict(run)
        for run in trial.audio["mismatches"]
        if run["start_s"] < trial.a + 0.4 and run["end_s"] > trial.a - 0.4
    ]
    for run in mismatches:
        overlapping = words_between(plan, run["start_s"], run["end_s"], rule="overlap")
        word = max(
            overlapping,
            key=lambda word: min(word.end, run["end_s"]) - max(word.start, run["start_s"]),
            default=None,
        )
        run["word"] = " ".join(norm_words(word.text)) if word is not None else None
    status, reason = ("fail" if mismatches else "pass"), None
    if not mismatches and trial.quiet:
        status, reason = "skipped", "no speech near the cut"
    elif not mismatches and transcripts is not None and not expected:
        status, reason = "skipped", "no speech near the cut"
    elif not mismatches and transcripts is not None and len(expected) == 1:
        status, reason = "unverifiable", "plan window contains fewer than 2 words"
    plan_words_near_join = norm_words(
        " ".join(word.text for word in words_between(plan, trial.a - 0.6, trial.a + 0.6))
    )
    listen = [
        {
            "side": side,
            "word": " ".join(norm_words(word.text)),
            "word_start_s": word.start,
            "word_end_s": word.end,
            "cut_point_s": point,
            "plan_words_near_join": plan_words_near_join,
            **levels,
        }
        for side, control, point, levels in zip(
            ("left", "right"), (left, right), (1.5, 1.0), trial.control_levels, strict=True
        )
        for word in control.words
        if point - word.start >= 0.10
        and word.end - point >= 0.10
        and min(levels["before_db"], levels["after_db"]) > levels["gate_db"]
        and not any(
            plan_words_near_join[index : index + len(normalized)] == normalized
            for normalized in [norm_words(word.text)]
            if normalized
            for index in range(len(plan_words_near_join) - len(normalized) + 1)
        )
    ]
    evidence = {**trial.audio, "mismatches": mismatches}
    if transcripts is None:
        evidence["reason"] = "no speech engine, word not named"
    return (
        WordsResult(status, reason, expected, heard, comparison["diff"], trial.artifacts),
        listen,
        evidence,
    )


@dataclass(frozen=True)
class _WordBoundary:
    """A relative envelope crossing between a word nucleus and its neighbour."""

    time_s: float
    valley_s: float
    valley_db: float
    level_db: float


@dataclass(frozen=True)
class _WordShape:
    """An energy nucleus and its two independently measurable boundaries."""

    nucleus_s: float
    peak_db: float
    onset: _WordBoundary | None
    offset: _WordBoundary | None


def _word_envelope(sound: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Smooth 10 ms log RMS frames without diluting a short valley with a loud peak."""
    frame = max(1, _sample(_WORD_FRAME_S, sr))
    count = math.ceil(len(sound) / frame)
    if not count:
        return np.empty(0), frame / sr
    padded = np.pad(sound.astype(np.float64), (0, count * frame - len(sound)))
    sizes = np.minimum(frame, len(sound) - np.arange(count) * frame)
    power = np.sum(padded.reshape(-1, frame) ** 2, axis=1) / sizes
    levels = 10 * np.log10(np.maximum(power, 1e-12))
    return np.convolve(np.pad(levels, (1, 1), mode="edge"), [0.25, 0.5, 0.25],
                       mode="valid"), frame / sr


def _word_nuclei(levels: np.ndarray, hop_s: float) -> list[int]:
    """Find prominent peaks, treating a sustained flat vowel as one nucleus."""
    if len(levels) < 3:
        return []
    edges = [0, *(np.flatnonzero(np.abs(np.diff(levels)) > 0.1) + 1).tolist(), len(levels)]
    radius = max(1, round(0.15 / hop_s))
    gate = max(-55.0, float(np.percentile(levels, 95)) - _WORD_PEAK_DROP_DB)
    candidates = []
    for start, end in zip(edges, edges[1:], strict=False):
        peak = float(np.max(levels[start:end]))
        if start == 0 or end == len(levels) or peak <= gate:
            continue
        if peak <= max(levels[start - 1], levels[end]):
            continue
        floor = max(float(np.min(levels[max(0, start - radius):end])),
                    float(np.min(levels[start:min(len(levels), end + radius)])))
        if peak - floor >= _WORD_PROMINENCE_DB:
            candidates.append((start + end - 1) // 2)
    chosen: list[int] = []
    for index in sorted(candidates, key=lambda index: -levels[index]):
        if all(abs(index - other) * hop_s >= 0.08 for other in chosen):
            chosen.append(index)
    return sorted(chosen)


def _word_boundary(
    levels: np.ndarray, hop_s: float, nucleus: int, neighbour: int,
) -> _WordBoundary | None:
    """Locate a crossing relative to the valley, keeping low room noise out of the word."""
    first, last = sorted((nucleus, neighbour))
    valley = first + int(np.argmin(levels[first:last + 1]))
    contrast = float(levels[nucleus] - levels[valley])
    if contrast < _WORD_PROMINENCE_DB:
        return None
    level = max(float(levels[valley]) + _WORD_VALLEY_FRACTION * contrast,
                float(levels[nucleus]) - _WORD_PEAK_DROP_DB)
    direction = 1 if valley < nucleus else -1
    crossing = next(index for index in range(valley, nucleus + direction, direction)
                    if levels[index] >= level)
    previous = crossing - direction
    fraction = (level - levels[previous]) / (levels[crossing] - levels[previous])
    time = (previous + 0.5 + fraction * direction) * hop_s
    return _WordBoundary(round(time / hop_s) * hop_s, (valley + 0.5) * hop_s,
                         float(levels[valley]), level)


def _word_shapes(sound: np.ndarray, sr: int) -> list[_WordShape]:
    """Measure each nucleus against its adjacent valleys on the source control."""
    levels, hop_s = _word_envelope(sound, sr)
    nuclei = _word_nuclei(levels, hop_s)
    radius = max(1, round(_WORD_SEARCH_S / hop_s))
    return [
        _WordShape(
            (nucleus + 0.5) * hop_s, float(levels[nucleus]),
            _word_boundary(levels, hop_s, nucleus,
                           nuclei[index - 1] if index else max(0, nucleus - radius)),
            _word_boundary(levels, hop_s, nucleus,
                           nuclei[index + 1] if index + 1 < len(nuclei)
                           else min(len(levels) - 1, nucleus + radius)),
        )
        for index, nucleus in enumerate(nuclei)
    ]


def _word_clock_shift(
    transcript: Transcript, shapes: Sequence[_WordShape], duration_s: float,
) -> float:
    """Correct a shared clock only when several complete word shapes agree on it."""
    intervals = [(word.start - 1.0, word.end - 1.0) for word in transcript.words
                 if 0.06 <= word.end - word.start <= 1.2]
    # A word can have several syllables; its clock must match both outer boundaries.
    measured = [(first.onset.time_s, last.offset.time_s, first_index, last_index)
                for first_index, first in enumerate(shapes) if first.onset is not None
                for last_index, last in enumerate(shapes) if last.offset is not None
                and first_index <= last_index
                and 0.06 <= last.offset.time_s - first.onset.time_s <= 1.2]
    scores = []
    for shift in np.arange(-_WORD_CLOCK_S, _WORD_CLOCK_S + _WORD_FRAME_S / 2, _WORD_FRAME_S):
        eligible = [(start + shift, end + shift) for start, end in intervals
                    if 0 <= start + shift < end + shift <= duration_s]
        pairs = sorted(
            (abs(start - onset) + abs(end - offset), word_index, shape_index)
            for word_index, (start, end) in enumerate(eligible)
            for shape_index, (onset, offset, _, _) in enumerate(measured)
            if max(abs(start - onset), abs(end - offset)) <= 0.03
        )
        words_used: set[int] = set()
        shapes_used: set[int] = set()
        error = 0.0
        for difference, word_index, shape_index in pairs:
            covered = set(range(measured[shape_index][2], measured[shape_index][3] + 1))
            if word_index not in words_used and not covered & shapes_used:
                words_used.add(word_index)
                shapes_used.update(covered)
                error += difference
        count = len(words_used)
        if count >= 2 and count >= 0.75 * len(eligible) and error / count <= 0.02:
            scores.append((count, -error, float(shift)))
    if not scores:
        return 0.0
    best = max(scores, key=lambda score: (score[0], score[1], -abs(score[2])))
    if any(score[0] == best[0] and abs(score[2] - best[2]) >= 0.1
           and score[1] >= best[1] - best[0] * 0.015 for score in scores):
        return 0.0
    return round(best[2], 2)


def _locate_word_shape(
    shapes: Sequence[_WordShape], start_s: float, end_s: float,
    side: Literal["first", "last"], point_s: float, neighbour_s: float | None,
) -> _WordShape | None:
    """Associate the word with an edge-facing nucleus within the clock uncertainty."""
    candidates = [shape for shape in shapes
                  if start_s - _WORD_CLOCK_S <= shape.nucleus_s <= end_s + _WORD_CLOCK_S]
    if not candidates:
        return None
    # Match both ends to retain a clipped short word or its final syllable.
    matches = [(abs(first.onset.time_s - start_s) + abs(last.offset.time_s - end_s),
                first, last)
               for first in candidates if first.onset is not None
               for last in candidates if last.offset is not None
               and first.nucleus_s <= last.nucleus_s
               and abs(first.onset.time_s - start_s) <= 0.03
               and abs(last.offset.time_s - end_s) <= 0.03]
    if matches:
        _, first, last = min(matches, key=lambda match: match[0])
        return first if side == "first" else last
    def counts(shape: _WordShape) -> bool:
        if start_s <= shape.nucleus_s <= end_s:
            return True
        if (shape.nucleus_s < start_s if side == "last" else shape.nucleus_s > end_s):
            return False
        inner = shape.onset if side == "last" else shape.offset
        outer = shape.offset if side == "last" else shape.onset
        inner_s = inner.time_s if inner else shape.nucleus_s
        outer_s = outer.time_s if outer else shape.nucleus_s
        edge_s = end_s if side == "last" else start_s
        within_edge = (inner_s <= edge_s + _WORD_HEARD_EDGE_S if side == "last"
                       else inner_s >= edge_s - _WORD_HEARD_EDGE_S)
        if not within_edge:
            return False
        if neighbour_s is None:
            return True
        return (outer_s < neighbour_s - _WORD_HEARD_EDGE_S if side == "last"
                else outer_s > neighbour_s + _WORD_HEARD_EDGE_S)

    # Kept-side shapes cannot stand in for the word's own syllables.
    # Outer shapes must stay near its heard edge and clear the next heard word.
    if any(start_s <= shape.nucleus_s <= end_s for shape in candidates):
        candidates = [shape for shape in candidates if counts(shape)]
    center = start_s + min(end_s - start_s, _WORD_SEARCH_S) / 2
    if side == "first":
        facing = [shape for shape in candidates if shape.nucleus_s >= point_s - _WORD_OVERLAP_S]
        if facing:
            return min(facing, key=lambda shape: shape.nucleus_s)
    else:
        facing = [shape for shape in candidates if shape.nucleus_s <= point_s + _WORD_OVERLAP_S]
        if facing and start_s <= point_s + _WORD_OVERLAP_S:
            return max(facing, key=lambda shape: shape.nucleus_s)
    return min(candidates, key=lambda shape: abs(shape.nucleus_s - center))


def _extend_word_boundary(
    shapes: Sequence[_WordShape], shape: _WordShape,
    side: Literal["first", "last"], neighbour_s: float | None,
) -> tuple[_WordBoundary | None, int]:
    """Walk to the word's outer boundary and count the additional shapes joined."""
    index = shapes.index(shape)
    following = shapes[index + 1:] if side == "last" else reversed(shapes[:index])
    boundary = shape.offset if side == "last" else shape.onset
    joined = 0
    for other in following:
        outer = other.offset if side == "last" else other.onset
        inner = other.onset if side == "last" else other.offset
        if boundary is None or outer is None or inner is not None:
            break
        # A missing inner crossing shares the valley with the adjacent shape.
        shared_s = boundary.valley_s
        if abs(shared_s - boundary.time_s) > _WORD_JOIN_GAP_S + 1e-9:
            break
        # Reject a shape that reaches the heard neighbour at either boundary.
        if neighbour_s is not None and (
            shared_s >= neighbour_s - 1e-9 if side == "last" else shared_s <= neighbour_s + 1e-9
        ):
            break
        if neighbour_s is not None and (
            outer.time_s >= neighbour_s - 1e-9 if side == "last"
            else outer.time_s <= neighbour_s + 1e-9
        ):
            break
        boundary = outer
        joined += 1
    return boundary, joined


def _locate_excluded_word_shape(
    shapes: Sequence[_WordShape], side: Literal["first", "last"], point_s: float,
) -> _WordShape | None:
    """Locate the nearest excluded voice without using the speech clock."""
    candidates = [shape for shape in shapes
                  if ((shape.onset.time_s if shape.onset else shape.nucleus_s)
                      <= point_s + _WORD_SEARCH_S
                      and shape.offset is not None
                      and shape.offset.time_s > point_s + _WORD_OVERLAP_S if side == "last"
                      else (shape.offset.time_s if shape.offset else shape.nucleus_s)
                      >= point_s - _WORD_SEARCH_S
                      and shape.onset is not None
                      and shape.onset.time_s < point_s - _WORD_OVERLAP_S)]
    return (min(candidates, key=lambda shape: shape.nucleus_s) if side == "last"
            else max(candidates, key=lambda shape: shape.nucleus_s)) if candidates else None


def _excluded_word_attributed(
    tokens: list[str], expected: list[str], index: int,
    side: Literal["first", "last"], excluded: list[str] | None,
) -> bool:
    """Attribute excluded edge speech by matching the surrounding token order."""
    if excluded is None or not expected:
        return False
    edge = excluded[:len(expected)] if side == "last" else excluded[-len(expected):]
    if edge != expected:
        return False
    context = excluded[len(expected):] if side == "last" else excluded[:-len(expected)]
    scores = {}
    for other in range(len(tokens) - len(expected) + 1):
        if other < index if side == "last" else other > index:
            continue
        if tokens[other:other + len(expected)] != expected:
            continue
        continuation = (tokens[other + len(expected):other + len(expected) + len(context)]
                        if side == "last" else tokens[max(0, other - len(context)):other])
        pairs = (zip(context, continuation, strict=False) if side == "last"
                 else zip(reversed(context), reversed(continuation), strict=False))
        score = 0
        for heard, control in pairs:
            if heard != control:
                break
            score += 1
        scores[other] = score
    return all(scores[index] > score for other, score in scores.items() if other != index)


def _planned_word_finding(
    transcript: Transcript,
    word: str,
    side: Literal["first", "last"],
    point_s: float,
    label: str,
    time_s: float,
    *,
    sound: np.ndarray,
    sr: int,
    cut: int | None = None,
    artifact: str | None = None,
    excluded: Transcript | None = None,
) -> Finding | None:
    """Locate intent in uncut source speech, then measure excluded voice by energy."""
    expected = norm_words(word)
    tokens = [(token, heard) for heard in transcript.words for token in norm_words(heard.text)]
    excluded_tokens = norm_words(excluded.text) if excluded is not None else None
    candidates = []
    for index in range(len(tokens) - len(expected) + 1):
        if not expected or [token for token, _ in tokens[index:index + len(expected)]] != expected:
            continue
        # WAVs have a silent lead; energy is measured on the unpadded source control.
        start = tokens[index][1].start - 1.0
        end = tokens[index + len(expected) - 1][1].end - 1.0
        if start - _WORD_SEARCH_S <= point_s <= end + _WORD_SEARCH_S:
            candidates.append((start, end, index))
    evidence = {
        "planned_word": word, "side": side, "cut_point_s": point_s,
        "window_s": [point_s - _WORD_SEARCH_S, point_s + _WORD_SEARCH_S],
        "frame_s": _WORD_FRAME_S, "clock_search_s": _WORD_CLOCK_S,
        "peak_drop_db": _WORD_PEAK_DROP_DB, "valley_fraction": _WORD_VALLEY_FRACTION,
        "allowed_overlap_ms": _WORD_OVERLAP_S * 1000,
        "artifact": artifact,
        "excluded_heard": (None if excluded_tokens is None else
                           excluded_tokens[:3] if side == "last" else excluded_tokens[-3:]),
        "located_by": "energy",
    }
    if not candidates:
        return Finding(
            "join.listen", "info",
            f'{label}: the planned {side} word "{word}" could not be heard in the source '
            "near this edge; listen.",
            cut=cut, time_s=time_s,
            suggested_action=f'Listen near this source edge and confirm that "{word}" is kept.',
            evidence=evidence,
        )
    start, end, index = min(candidates, key=lambda span:
                           abs(span[0] + min(span[1] - span[0], _WORD_SEARCH_S) / 2 - point_s))
    shapes = _word_shapes(sound, sr)
    shift = _word_clock_shift(transcript, shapes, len(sound) / sr)
    excluded_applies = _excluded_word_attributed(
        [token for token, _ in tokens], expected, index, side, excluded_tokens,
    )
    shape = None
    if excluded_applies:
        shape = _locate_excluded_word_shape(shapes, side, point_s)
        if shape is not None:
            evidence["located_by"] = "excluded speech"
    edge_word = tokens[index if side == "first" else index + len(expected) - 1][1]
    neighbours = reversed(tokens[:index]) if side == "first" else tokens[index + len(expected):]
    neighbour = next((heard for _, heard in neighbours if heard is not edge_word), None)
    neighbour_s = None
    locate_neighbour_s = None
    if neighbour is not None:
        neighbour_s = (neighbour.end if side == "first" else neighbour.start) - 1.0 + shift
        locate_neighbour_s = neighbour_s
        if excluded_tokens:
            neighbour_tokens = norm_words(neighbour.text)
            # Excluded audio adjusts only which shape counts as the word's own.
            if side == "last" and excluded_tokens[0] == neighbour_tokens[0]:
                locate_neighbour_s = min(neighbour_s, point_s)
            elif side == "first" and excluded_tokens[-1] == neighbour_tokens[-1]:
                locate_neighbour_s = max(neighbour_s, point_s)
    if shape is None:
        shape = _locate_word_shape(
            shapes, start + shift, end + shift, side, point_s, locate_neighbour_s,
        )
    boundary, joined = ((None, 0) if shape is None else
                        _extend_word_boundary(shapes, shape, side, neighbour_s))
    evidence.update(word_start_s=start, word_end_s=end, clock_shift_s=shift,
                    joined_shapes=joined)
    overlap = (None if boundary is None else
               _sample(point_s, sr) - _sample(boundary.time_s, sr) if side == "first"
               else _sample(boundary.time_s, sr) - _sample(point_s, sr))
    if overlap is None or (excluded_applies and overlap <= _sample(_WORD_OVERLAP_S, sr)):
        if excluded_applies:
            evidence["located_by"] = "excluded speech"
        reason = ("was heard in the excluded audio but its energy could not be measured"
                  if excluded_applies else "has no distinct energy boundary in the source "
                  "near this edge")
        return Finding(
            "join.listen", "info",
            f'{label}: the planned {side} word "{word}" {reason}; listen.',
            cut=cut, time_s=time_s,
            suggested_action=f'Listen near this source edge and confirm that "{word}" is kept.',
            evidence=evidence,
        )
    if overlap <= _sample(_WORD_OVERLAP_S, sr):
        return None
    overlap_ms = overlap / sr * 1000
    evidence.update(boundary_s=boundary.time_s, nucleus_s=shape.nucleus_s,
                    peak_dbfs=shape.peak_db, valley_s=boundary.valley_s,
                    valley_dbfs=boundary.valley_db, boundary_level_dbfs=boundary.level_db,
                    overlap_ms=overlap_ms)
    direction = "earlier" if side == "first" else "later"
    portion = "start" if side == "first" else "end"
    return Finding(
        "join.words", "fail",
        f'{label}: the source voice for the planned {side} word "{word}" overlaps this '
        f'edge by {overlap_ms:.0f} ms; the cut drops its {portion} or the whole word.',
        cut=cut, time_s=time_s,
        suggested_action=f'Move this segment\'s {side} edge {direction} to keep "{word}".',
        evidence=evidence,
    )


@dataclass(frozen=True)
class _SourceWordTrial:
    """An outer segment's uncut source control on its unpadded PCM clock."""

    path: Path
    sound: np.ndarray
    word: str
    side: Literal["first", "last"]
    point_s: float
    time_s: float
    artifact: str | None


def _excluded_word_clip(
    sound: np.ndarray, sr: int, point_s: float, side: Literal["first", "last"],
    name: str, work_dir: Path, artifacts_dir: Path | None,
) -> Path:
    """Write excluded source speech with the usual silent lead."""
    point, count = _sample(point_s, sr), _sample(_WORD_EXCLUDED_S, sr)
    window = sound[point:point + count] if side == "last" else sound[max(0, point - count):point]
    path = work_dir / name
    _write_wav(path, window, sr)
    if artifacts_dir is not None:
        target = artifacts_dir / name
        if target.resolve() != path.resolve():
            _write_wav(target, window, sr)
    return path


def _outer_word_trials(
    plan: Plan, source: Callable[[str | None], _Media | None], sr: int,
    work_dir: Path, artifacts_dir: Path | None,
) -> list[_SourceWordTrial]:
    """Add source controls for authored outer edges to the shared speech batch."""
    trials = []
    for side, segment, word, frame, before, after in (
        ("first", plan.segments[0], plan.segments[0].first_word,
         plan.segments[0].src_f0, 1.0, 1.5),
        ("last", plan.segments[-1], plan.segments[-1].last_word,
         plan.segments[-1].src_f1, 1.5, 1.0),
    ):
        if word is None:
            continue
        media = source(segment.source_id)
        if media is None or media.sound is None:
            continue
        point = media.info.video_offset_s + frame / float(plan.fps)
        window = _slice(media.sound, sr, point - before, point + after)
        name = f"edge-{side}-source.wav"
        path = work_dir / name
        _write_wav(path, window, sr)
        artifact = None
        if artifacts_dir is not None:
            target = artifacts_dir / name
            if target.resolve() != path.resolve():
                _write_wav(target, window, sr)
            artifact = str(target)
        time = float(segment.out_time + (segment.n_frames / plan.fps if side == "last" else 0))
        trials.append(_SourceWordTrial(path, window, word, side, before, time, artifact))
    return trials


def _gap_findings(cuts: Sequence[CutResult], threshold: float) -> list[Finding]:
    groups: list[list[CutResult]] = []
    for cut in cuts:
        if cut.gap_ms is None or abs(cut.gap_ms) < threshold or cut.gap_ms == 0:
            continue
        if (
            groups
            and cut.index == groups[-1][-1].index + 1
            and (cut.gap_ms > 0) == (groups[-1][-1].gap_ms > 0)
        ):
            groups[-1].append(cut)
        else:
            groups.append([cut])
    findings = []
    for group in groups:
        inserted = group[0].gap_ms > 0
        label = (
            f"Cut {group[0].index}"
            if len(group) == 1
            else f"Cuts {group[0].index} to {group[-1].index}"
        )
        amount = sum(abs(cut.gap_ms) for cut in group) / len(group)
        verb = "inserted" if inserted else "lost"
        where = "the join" if len(group) == 1 else "each join"
        findings.append(
            Finding(
                "join.gap",
                "warn",
                f"{label}: about {amount:.0f} ms of audio was {verb} at {where}.",
                cut=group[0].index,
                time_s=group[0].out_time_s,
                suggested_action=(
                    "Audio was encoded per clip and joined without re-encoding. "
                    "Encode the audio once for the whole video."
                    if inserted
                    else "Audio is missing at the join. "
                    "Re-render so each clip keeps its full audio."
                ),
                evidence={"gap_ms": {str(cut.index): cut.gap_ms for cut in group}},
            )
        )
    return findings


def check_joins(
    render: Path,
    plan: Plan,
    sources: Mapping[str, Path | None],
    sync: SyncResult,
    *,
    source_streams: Mapping[str, int | None] | None = None,
    backend: Backend | None,
    second_backend: Backend | None = None,
    masks: Sequence[tuple[Fraction, Fraction]] = (),
    regions: Sequence[PictureRegion] = (),
    transitions: Sequence[TransitionWindow] = (),
    click_audio: Path | None = None,
    words_audio: Path | None = None,
    thresholds: Thresholds = Thresholds(),  # noqa: B008 - frozen value
    work_dir: Path,
    artifacts_dir: Path | None = None,
    grade: GradeSpec | None = None,
) -> JoinsResult:
    """Check every join, batching all speech windows in one transcription call."""
    outer_intent = bool(plan.segments and (
        plan.segments[0].first_word is not None or plan.segments[-1].last_word is not None
    ))
    if not plan.cuts and not transitions and not (backend is not None and outer_intent):
        return JoinsResult([], [])
    if artifacts_dir is not None:
        artifacts_dir = validate_output_dir(artifacts_dir)
    info = ffmpeg.probe(render, count_frames=False)
    sr, sound = audio.decode(render) if info.has_audio else (plan.sample_rate, None)
    click_sound = audio.decode(click_audio, sr=sr)[1] if click_audio is not None else None
    words_sound = audio.decode(words_audio, sr=sr)[1] if words_audio is not None else sound
    rendered = _Media(info, sound, video.grey_frames(render, w=64) if info.has_video else None)
    # A VFR render's native decode cannot stand in for a source on the plan grid.
    cache: dict[tuple[Path, int | None], _Media | None] = (
        {} if info.variable_frame_rate else {(Path(render).resolve(), None): rendered}
    )

    def source(source_id: str | None) -> _Media | None:
        path = sources.get(source_id)
        if path is None or not Path(path).is_file():
            return None
        path = Path(path).resolve()
        stream = (source_streams or {}).get(source_id)
        key = (path, stream)
        if key not in cache:
            metadata = ffmpeg.probe(path, count_frames=False)
            rate = video.source_rate(metadata, plan.fps)
            try:
                source_sound = (
                    audio.decode(
                        path, sr=sr, **({"stream": stream} if stream is not None else {})
                    )[1]
                    if sound is not None and metadata.has_audio else None
                )
            except InputError:
                cache[key] = None
                return None
            cache[key] = _Media(
                metadata,
                source_sound,
                video.grey_frames(
                    path, w=64, **({"rate": rate} if rate != metadata.fps else {}),
                    **({"grade": grade} if grade is not None and grade.active else {}),
                ) if metadata.has_video else None,
                rate,
            )
        return cache[key]

    def word_source(source_id: str | None) -> _Media | None:
        media = source(source_id)
        path = sources.get(source_id)
        # Source controls normally use the original take. A render used as its own source
        # must hear the same no-effects audio as the render trial.
        if (words_audio is not None and media is not None and path is not None
                and Path(path).resolve() == Path(render).resolve()
                and (source_streams or {}).get(source_id) in (None, 0)):
            return replace(media, sound=words_sound)
        return media

    rows = {row.index: row for row in sync.segments}
    gaps = _short_gaps(plan, rows)
    cuts, findings, trials = [], [], {}
    fps = float(plan.fps)
    transition_results = [
        _transition(window, plan, rendered, source, rows, masks, regions)
        for window in transitions
    ]
    for result in transition_results:
        cut = next((item for item in plan.cuts if item.out_frame == result["join_frame"]), None)
        reason = result["reason"] or "frame count and boundary frames match the cut"
        findings.append(Finding(
            "picture.transition", "fail" if result["status"] == "fail" else "info",
            f"Transition {result['id']} [{result['start_frame']}, {result['end_frame']}): "
            f"{reason}.",
            cut=cut.index if cut else None, time_s=result["join_frame"] / fps,
            suggested_action="Re-render the transition at its declared window."
            if result["status"] == "fail" else None,
            evidence=result,
        ))
    for cut in plan.cuts:
        planned = float(cut.out_time)
        planned_gap = _planned_gap_s(cut, fps)
        label = f"Cut {cut.index} at {format_time(planned)}"
        left_sync = rows.get(cut.left.index, SegmentSync(cut.left.index, cut.left.source_id))
        right_sync = rows.get(cut.right.index, SegmentSync(cut.right.index, cut.right.source_id))
        j_end, end_evidence = _landing(left_sync, right_sync, planned - planned_gap, "tail")
        j_start, start_evidence = _landing(right_sync, left_sync, planned, "head")
        fallback = bool(end_evidence["reason"] or start_evidence["reason"])
        landing = "planned" if fallback else "measured"
        end_measured = not end_evidence["reason"] or (
            end_evidence["offset_source"] == "own" and left_sync.confidence == "high"
        )
        start_measured = not start_evidence["reason"] or (
            start_evidence["offset_source"] == "own" and right_sync.confidence == "high"
        )
        sound_ended = sound is not None and _sound_ends_before_cut(sound, sr, j_end, j_start)
        if sound is None:
            findings.append(
                Finding(
                    "join.listen",
                    "info",
                    f"{label}: audio checks skipped: no audio in the render.",
                    cut=cut.index,
                    time_s=planned,
                    evidence={
                        "reason": "no audio in the render",
                        "skipped": ["click", "gap", "pause", "levels", "words"],
                    },
                )
            )
        elif sound_ended:
            findings.append(
                Finding(
                    "join.listen",
                    "info",
                    f"{label}: audio checks skipped: the sound ends before this cut.",
                    cut=cut.index,
                    time_s=planned,
                    evidence={
                        "reason": "the sound ends before this cut",
                        "skipped": ["click", "gap", "pause", "levels", "words"],
                    },
                )
            )
        elif fallback:
            findings.append(
                Finding(
                    "join.listen",
                    "info",
                    f"{label}: audio landing is uncertain; using the planned time "
                    "plus the measured audio offset for "
                    + (
                        "both sides."
                        if end_evidence["reason"] and start_evidence["reason"]
                        else "the left side."
                        if end_evidence["reason"]
                        else "the right side."
                    ),
                    cut=cut.index,
                    time_s=planned,
                    evidence={"landing": landing, "end": end_evidence, "start": start_evidence},
                )
            )
        left, right = source(cut.left.source_id), source(cut.right.source_id)
        end = cut.left.src_f1 / fps + (left.info.video_offset_s if left else 0)
        start = cut.right.src_f0 / fps + (right.info.video_offset_s if right else 0)
        level_left = (
            audio.rms_db(left.sound, end, sr, 0.004)
            if sound is not None
            and not sound_ended
            and left is not None
            and left.sound is not None
            else None
        )
        level_right = (
            audio.rms_db(right.sound, start, sr, 0.004)
            if sound is not None
            and not sound_ended
            and right is not None
            and right.sound is not None
            else None
        )
        control = None
        words = WordsResult("skipped", None, [], [], [], [])
        if (
            sound is not None
            and not sound_ended
            and left is not None
            and right is not None
            and left.sound is not None
            and right.sound is not None
        ):
            before = _slice(left.sound, sr, end - 0.3, end)
            after = _slice(right.sound, sr, start, start + 0.3)
            ideal, joins = audio.splice(
                np.concatenate((before, after)),
                [(0, len(before)), (len(before), len(before) + len(after))],
                fade=int(0.005 * sr),
            )
            control = audio.click_db(ideal, joins[0], sr)
        if sound is None:
            words = replace(words, reason="no audio in the render")
        elif sound_ended:
            words = replace(words, reason="the sound ends before this cut")
        else:
            trial, reason = _word_trial(
                cut, plan, words_sound, sr, j_start, word_source, work_dir, artifacts_dir, gaps
            )
            if trial is not None:
                trials[cut.index] = trial
            else:
                words = replace(words, reason=reason)
        transition = next((item for item in transitions
                           if item.start_frame <= cut.out_frame < item.end_frame), None)
        if transition is not None:
            frames = FramesResult("skipped", "covered by transition", None, None, None, None,
                                  None, None, round(planned_gap * fps),
                                  {"transition": asdict(transition)})
        else:
            frames = _frames(cut, left_sync, right_sync, rendered, left, right, fps, masks, regions)
        render_fps = float(rendered.info.fps or fps)
        incoming_frame = (
            frames.render_right
            if frames.render_right is not None
            else math.floor(planned * render_fps + 0.5)
        )
        flash = {"flash": False, "reason": "covered by transition",
                 "transition": asdict(transition)} if transition is not None else _flash(
            rendered,
            incoming_frame,
            render_fps,
            thresholds.touch_cut_frames,
            masks,
            regions,
            transitions,
        )
        if flash["flash"]:
            flash = _flash_in_source(
                flash,
                cut,
                incoming_frame,
                rendered,
                left,
                right,
                left_sync,
                right_sync,
                fps,
                regions,
            )
        click_db: float | None = None
        click_peak_dbfs: float | None = None
        click_source = "voice_stem" if click_sound is not None else "mix"
        if sound is not None and not sound_ended:
            click_signal = click_sound
            if click_signal is None or any(
                not 0 <= round(t * sr) < len(click_signal) for t in (j_end, j_start)
            ):
                click_signal = sound
                click_source = "mix"
            # Keep sharpness and peak paired so separate landings cannot combine into a click.
            clicks = [
                (click_at(click_signal, sr, t), click_peak_dbfs_at(click_signal, sr, t))
                for t in (j_end, j_start)
            ]
            qualifying = [
                click
                for click in clicks
                if click[0] > thresholds.click_fail_db and click[1] >= thresholds.click_floor_dbfs
            ]
            click_db, click_peak_dbfs = max(qualifying or clicks, key=lambda click: click[0])
        row = CutResult(
            cut.index,
            planned,
            j_end,
            j_start,
            landing,
            (j_start - j_end - planned_gap) * 1000
            if sound is not None and not sound_ended and end_measured and start_measured
            else None,
            click_db,
            click_peak_dbfs,
            thresholds.click_floor_dbfs,
            control,
            pause_len_at(sound, sr, j_start)
            if sound is not None and not sound_ended
            else None,
            level_left,
            level_right,
            words,
            frames,
            flash,
            click_source=click_source if click_db is not None else None,
        )
        cuts.append(row)
        if flash["flash"] and not flash["in_source"]:
            findings.append(
                flash_finding(
                    flash, cut.index, planned, fps=render_fps,
                    video_offset_s=rendered.info.video_offset_s,
                )
            )
        if (
            row.click_db is not None
            and row.click_db > thresholds.click_fail_db
            and row.click_peak_dbfs is not None
            and row.click_peak_dbfs >= thresholds.click_floor_dbfs
        ):
            findings.append(
                Finding(
                    "join.click",
                    "fail",
                    f"{label}: click at the join ({row.click_db:+.1f} dB).",
                    cut=cut.index,
                    time_s=planned,
                    suggested_action="Re-render this cut with a short audio fade (5 to 30 ms).",
                    evidence={
                        "click_db": row.click_db,
                        "click_peak_dbfs": row.click_peak_dbfs,
                        "click_floor_dbfs": row.click_floor_dbfs,
                        "click_db_control": control,
                        "click_source": click_source,
                    },
                )
            )
        if (
            row.pause_s is not None
            and row.pause_s < 0.03
            and level_left is not None
            and level_right is not None
            and min(level_left, level_right) > -35
        ):
            findings.append(
                Finding(
                    "join.listen",
                    "info",
                    f"{label}: no pause and voice on both sides; listen to this join.",
                    cut=cut.index,
                    time_s=planned,
                    evidence={
                        "pause_s": row.pause_s,
                        "level_left_db": level_left,
                        "level_right_db": level_right,
                    },
                )
            )
        if frames.status == "fail":
            findings.append(
                Finding(
                    "join.frames",
                    "fail",
                    f"{label}: {frames.reason}.",
                    cut=cut.index,
                    time_s=planned,
                    suggested_action=(
                        "Re-render this cut; each segment should end and start "
                        "on its planned frame."
                    ),
                    evidence=asdict(frames),
                )
            )
    outer_trials = (
        _outer_word_trials(plan, word_source, sr, work_dir, artifacts_dir)
        if backend is not None and sound is not None and len(sound) and outer_intent else []
    )
    if trials or outer_trials:
        trial_paths = [path for trial in trials.values() for path in trial.paths]
        planned_cuts = {cut.index: cut for cut in plan.cuts}
        excluded_paths = {}
        if backend is not None:
            for index, trial in trials.items():
                planned_cut = planned_cuts[index]
                for side, word, point, control in (
                    ("last", planned_cut.left.last_word, 1.5, 0),
                    ("first", planned_cut.right.first_word, 1.0, 1),
                ):
                    if word is not None:
                        excluded_paths[index, side] = _excluded_word_clip(
                            trial.controls[control], sr, point, side,
                            f"excluded-cut-{index:02d}-{side}.wav", work_dir, artifacts_dir,
                        )
            for outer in outer_trials:
                name = f"excluded-{'start' if outer.side == 'first' else 'end'}.wav"
                excluded_paths[None, outer.side] = _excluded_word_clip(
                    outer.sound, sr, outer.point_s, outer.side, name, work_dir, artifacts_dir,
                )
        transcripts = None
        if backend is not None:
            batch = [
                *trial_paths,
                *[trial.path for trial in outer_trials],
                *excluded_paths.values(),
            ]
            _LOGGER.info("Transcribing %d clips", len(batch))
            transcripts = backend.transcribe_many(batch)
        second_transcripts = None
        second_unmeasured_reason = None
        if trials and backend is not None and second_backend is not None:
            try:
                second_transcripts = second_backend.transcribe_many(trial_paths)
            except YapError as exc:
                if exc.code == "second_engine_unmeasured":
                    second_unmeasured_reason = exc.reason
                else:
                    second_unmeasured_reason = (
                        f"unmeasured: the second speech engine {second_backend.name} "
                        f"failed: {getattr(exc, 'reason', exc.message)}"
                    )
        for index, cut in enumerate(cuts):
            if cut.index not in trials:
                continue
            trial = trials[cut.index]
            words, listen, evidence = _words(trial, transcripts)
            relisten = None
            if transcripts is not None:
                planned_cut = planned_cuts[cut.index]
                relisten = relisten_at(
                    [transcripts[path] for path in trial.paths],
                    engine=backend.name,
                    left_length_s=planned_cut.left.n_frames / fps,
                    right_length_s=planned_cut.right.n_frames / fps,
                    j_end_s=cut.j_end_s,
                    j_start_s=cut.j_start_s,
                    window_start_s=cut.j_start_s - trial.a,
                    second_transcripts=(
                        [second_transcripts[path] for path in trial.paths]
                        if second_transcripts is not None else None
                    ),
                    second_engine=second_backend.name if second_backend is not None else None,
                    second_unmeasured_reason=second_unmeasured_reason,
                    words_failed=words.status == "fail",
                )
                findings.extend(relisten_findings(relisten, cut.index, cut.out_time_s))
            cuts[index] = replace(cut, words=words, relisten=relisten)
            if listen:
                names = ", ".join(
                    f'"{word}"' for word in dict.fromkeys(row["word"] for row in listen)
                )
                findings.append(
                    Finding(
                        "join.listen",
                        "info",
                        f"Cut {cut.index} at {format_time(cut.out_time_s)}: "
                        f"the edit cuts through {names}; listen.",
                        cut=cut.index,
                        time_s=cut.out_time_s,
                        evidence={
                            "controls": listen,
                            "plan_words_near_join": listen[0]["plan_words_near_join"],
                        },
                    )
                )
            if words.status == "fail":
                for run in evidence["mismatches"]:
                    for edge in ("start_s", "end_s"):
                        run[edge] = _restore_gaps(
                            trial.clock_start_s + run[edge] + evidence["lag_s"], trial.gaps
                        )
                mismatch = evidence["mismatches"][0]
                detail = f' around "{mismatch["word"]}"' if mismatch["word"] else ""
                interval = f"{format_time(mismatch['start_s'])} to {format_time(mismatch['end_s'])}"
                findings.append(
                    Finding(
                        "join.words",
                        "fail",
                        f"Cut {cut.index} at {format_time(cut.out_time_s)}: "
                        f"the render's audio differs from the edit{detail} "
                        f"({interval} in the render).",
                        cut=cut.index,
                        time_s=cut.out_time_s,
                        suggested_action=(
                            "Re-render this cut; its audio does not match the edit list."
                        ),
                        evidence={**asdict(words), **evidence},
                    )
                )
            if transcripts is not None:
                planned_cut = plan.cuts[index]
                for side, word, point, role in (
                    ("last", planned_cut.left.last_word, 1.5, 2),
                    ("first", planned_cut.right.first_word, 1.0, 3),
                ):
                    if word is None:
                        continue
                    finding = _planned_word_finding(
                        transcripts[trial.paths[role]], word, side, point,
                        f"Cut {cut.index} at {format_time(cut.out_time_s)}", cut.out_time_s,
                        sound=trial.controls[role - 2], sr=sr, cut=cut.index,
                        artifact=trial.artifacts[role] if trial.artifacts else None,
                        excluded=transcripts[excluded_paths[cut.index, side]],
                    )
                    if finding is not None:
                        findings.append(finding)
                        if finding.severity == "fail":
                            cuts[index] = replace(cuts[index], words=replace(
                                cuts[index].words, status="fail",
                                reason="planned boundary word clipped",
                            ))
        for outer in outer_trials:
            label = (
                f"Render {'start' if outer.side == 'first' else 'end'} at "
                f"{format_time(outer.time_s)}"
            )
            finding = _planned_word_finding(
                transcripts[outer.path], outer.word, outer.side, outer.point_s, label,
                outer.time_s, sound=outer.sound, sr=sr, artifact=outer.artifact,
                excluded=transcripts[excluded_paths[None, outer.side]],
            )
            if finding is not None:
                findings.append(finding)
    findings.extend(_gap_findings(cuts, thresholds.gap_warn_ms))
    return JoinsResult(cuts, findings, transition_results)
