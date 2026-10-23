"""Find picture cuts and check a finished render without its source or edit list.

Audio joins are unknown without an edit list. Search asymmetrically from 0.10 s
before to 0.35 s after each picture cut to allow for accumulating audio delay.
Search-selected clicks must reach +6 dB and stand 6 dB above nearby sound:
maximizing over ordinary sound can otherwise select speech onsets. Below either
limit, retain the best candidate measurement but keep the picture time as the
unmeasured audio join.
"""

from __future__ import annotations

import math
import wave
from collections.abc import Iterator
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import audio, ffmpeg, video
from yap_studio.checks import joins, structure
from yap_studio.checks.flash import flash_at, flash_finding
from yap_studio.checks.model import CHECK_IDS, Finding, Thresholds, format_time
from yap_studio.checks.structure import StructureResult
from yap_studio.errors import InputError, ValidationError
from yap_studio.paths import atomic_output, validate_output_dir
from yap_studio.transcript import Backend, Transcript

_WIDTH = 160
_SCORE = 5.0
_SEARCH_BEFORE_S = 0.10
_SEARCH_AFTER_S = 0.35
_SEARCH_CLICK_DB = 6.0
_CLICK_OUTLIER_DB = 6.0
_REFERENCE_RADIUS_S = 2.0
_SOURCE_CHECKS = (
    "sync.av_offset",
    "sync.drift",
    "plan.picture",
    "plan.frames",
    "plan.samples",
    "join.words",
    "join.relisten",
    "join.second_model",
    "join.gap",
)


@dataclass(frozen=True)
class DetectedCut:
    """The first frame of a new shot, on the container clock."""

    index: int
    frame: int
    time_s: float
    confidence: float
    evidence: dict

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class DetectedCutResult:
    """Source-independent measurements at a detected picture cut."""

    cut: DetectedCut
    join_s: float
    click_db: float | None
    pause_s: float | None
    duplicate: bool
    black: bool
    listen: bool | None
    flash: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class DetectResult:
    """Detected joins, whole-render checks, and explicit coverage."""

    cuts: list[DetectedCut]
    results: list[DetectedCutResult]
    structure: StructureResult
    findings: list[Finding]
    checks: dict[str, dict]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def _local_reference(values: np.ndarray, radius: int, floor: float) -> tuple[np.ndarray, ...]:
    medians = np.zeros(len(values))
    scales = np.full(len(values), floor)
    for i in range(1, len(values)):
        neighbours = np.concatenate(
            (values[max(1, i - radius) : i], values[i + 1 : i + radius + 1])
        )
        if len(neighbours) < 3:
            continue
        median = float(np.median(neighbours))
        mad = float(np.median(np.abs(neighbours - median)))
        medians[i] = median
        scales[i] = max(1.4826 * mad, 0.05 * median, floor)
    return medians, scales


def _reference_radius(fps: float) -> int:
    return max(2, round(fps))


def _picture_evidence(frames: np.ndarray, fps: float) -> list[dict]:
    """Score every confirmable boundary, retaining rejected scores for calibration.

    Compare both cross-boundary distances to the local *one-frame* motion
    distribution. Their job is to rule out a return to the old picture, rather
    than demand a second motion jump. Ratio gates also reject codec catch-up
    frames and flashes when the local MAD alone makes them look significant.
    """
    if frames.ndim != 3 or min(frames.shape[1:], default=0) < 1:
        raise ValidationError("frames", "must have shape [n, h, w] with non-empty pictures")
    if any(not np.isfinite(frame).all() for frame in frames):
        raise ValidationError("frames", "must contain finite grey values")
    if len(frames) < 4:
        return []
    distance = np.zeros(len(frames))
    across = np.zeros(len(frames))
    for i in range(1, len(frames)):
        distance[i] = video.frame_distance(frames[i - 1], frames[i])
        if i >= 2:
            across[i] = video.frame_distance(frames[i - 2], frames[i])
    histograms = np.asarray(
        [
            np.bincount((np.clip(f, 0, 255).astype(np.uint8) // 8).ravel(), minlength=32)
            for f in frames
        ],
        dtype=np.float64,
    )
    histograms -= histograms.mean(axis=1, keepdims=True)
    histograms /= np.maximum(np.linalg.norm(histograms, axis=1, keepdims=True), 1e-12)
    drop = np.zeros(len(frames))
    drop[1:] = np.clip(1 - (histograms[:-1] * histograms[1:]).sum(axis=1), 0, 2)
    radius = _reference_radius(fps)
    median, scale = _local_reference(distance, radius, 0.002)
    hist_median, hist_scale = _local_reference(drop, radius, 0.005)
    evidence = []
    for i in range(2, len(frames) - 1):
        before, after = float(across[i]), float(across[i + 1])
        change = float(distance[i])
        z = float((change - median[i]) / scale[i])
        hist_z = float((drop[i] - hist_median[i]) / hist_scale[i])
        before_z = float((before - median[i]) / scale[i])
        after_z = float((after - median[i]) / scale[i])
        motion_ratio = min(change, before, after) / max(float(median[i]), 0.002)
        persistence = min(before, after) / max(change, 0.002)
        score = min(
            max(z, hist_z), before_z, after_z, _SCORE * motion_ratio / 3, _SCORE * persistence / 0.5
        )
        evidence.append(
            {
                "frame": i,
                "distance": change,
                "histogram_drop": float(drop[i]),
                "local_motion": float(median[i]),
                "motion_scale": float(scale[i]),
                "distance_z": z,
                "histogram_z": hist_z,
                "before_distance": before,
                "after_distance": after,
                "before_z": before_z,
                "after_z": after_z,
                "motion_ratio": motion_ratio,
                "persistence": persistence,
                "score": score,
                "threshold": _SCORE,
                "analysis_width": frames.shape[2],
            }
        )
    return evidence


def _stream_picture_evidence(render: Path, fps: float) -> Iterator[dict]:
    """Score blocks with exactly the neighbours used by ``_picture_evidence``."""
    radius = _reference_radius(fps)
    # Reference distances/drops at i +/- radius need frames [i-radius-1, i+radius].
    # The two cross-boundary comparisons also need [i-2, i+1].
    before, after = max(2, radius + 1), max(1, radius)
    pending = None
    offset, next_frame = 0, 2
    with closing(video.iter_grey_frames(render, w=_WIDTH)) as blocks:
        for first, frames in blocks:
            pending = frames if pending is None else np.concatenate((pending, frames))
            stop = first + len(frames) - after
            if stop <= next_frame:
                continue
            for row in _picture_evidence(pending, fps):
                row["frame"] += offset
                if next_frame <= row["frame"] < stop:
                    yield row
            next_frame = stop
            keep = max(0, next_frame - before)
            pending = pending[keep - offset :].copy()
            offset = keep
    if pending is not None:
        # At EOF use the same truncated reference windows as the full-array path.
        for row in _picture_evidence(pending, fps):
            row["frame"] += offset
            if row["frame"] >= next_frame:
                yield row


def detect_cuts(render: Path, *, frames: np.ndarray | None = None) -> list[DetectedCut]:
    """Detect hard cuts using 160 px grey pictures, independently of audio.

    Use a +/-1 s reference, excluding the candidate, and robust scales floored
    at 5% of the median and 0.002 distance / 0.005 histogram drop (32 bins).
    Require a score of 5, both cross-boundary distances to be outliers, a jump
    at least 3 times local motion, and persistence at least half the jump.
    Merge candidates within two frames. Confidence is 1-exp(-score/5), a
    ranking measure rather than a calibrated probability. The first two and
    final frame cannot supply both cross-boundary comparisons.
    """
    info = ffmpeg.probe(render, count_frames=False)
    if not info.has_video or info.fps is None:
        raise InputError(str(render), "no video stream")
    fps = float(info.fps)
    evidence = (
        _stream_picture_evidence(render, fps)
        if frames is None
        else _picture_evidence(frames, fps)
    )
    candidates = [row for row in evidence if row["score"] >= _SCORE]
    groups: list[list[dict]] = []
    for row in candidates:
        if groups and row["frame"] - groups[-1][-1]["frame"] <= 2:
            groups[-1].append(row)
        else:
            groups.append([row])
    strongest = [max(group, key=lambda row: row["score"]) for group in groups]
    return [
        DetectedCut(
            index,
            row["frame"],
            info.video_offset_s + row["frame"] / fps,
            1 - math.exp(-row["score"] / _SCORE),
            row,
        )
        for index, row in enumerate(strongest, 1)
    ]


def _sample(time_s: float, sr: int) -> int:
    return math.floor(time_s * sr + 0.5)


def _click_ratio(sound: np.ndarray, sr: int, index: int) -> float:
    # click_db differences its entire argument, so give it only its reference.
    low = max(0, index - int(0.1 * sr))
    high = min(len(sound), index + int(0.1 * sr) + 2)
    return audio.click_db(sound[low:high], index - low, sr)


def _search_bounds(count: int, sr: int, time_s: float) -> tuple[int, int]:
    return (
        max(0, math.ceil((time_s - _SEARCH_BEFORE_S) * sr)),
        min(count, math.floor((time_s + _SEARCH_AFTER_S) * sr) + 1),
    )


def _peak_candidates(
    difference: np.ndarray, sr: int, first: int, last: int
) -> list[int]:
    peaks = [
        first + index
        for index in _spaced_peaks(difference[first:last], max(1, math.ceil(0.005 * sr)))
    ]
    return sorted(peaks, key=lambda index: difference[index], reverse=True)[:20]


def _click_reference(
    sound: np.ndarray,
    sr: int,
    difference: np.ndarray,
    time_s: float,
    search_first: int,
    search_last: int,
) -> tuple[float | None, int]:
    first = max(0, math.ceil((time_s - _REFERENCE_RADIUS_S) * sr))
    last = min(len(sound), math.floor((time_s + _REFERENCE_RADIUS_S) * sr) + 1)
    local = difference[first:last].copy()
    excluded_first = max(first, search_first) - first
    excluded_last = min(last, search_last) - first
    if excluded_last > excluded_first:
        local[excluded_first:excluded_last] = 0
    candidates = _peak_candidates(local, sr, 0, len(local))
    levels = [_click_ratio(sound, sr, first + index) for index in candidates]
    if not levels:
        return None, 0
    reference = max(levels) if len(levels) < 10 else float(np.percentile(levels, 95))
    return reference, len(levels)


def _place_join(
    sound: np.ndarray, sr: int, difference: np.ndarray, time_s: float
) -> tuple[float, dict]:
    first, last = _search_bounds(len(sound), sr, time_s)
    if first >= last:
        raise ValidationError("cut", "picture cut is outside the render audio")
    candidates = _peak_candidates(difference, sr, first, last)
    measured = [(_click_ratio(sound, sr, index), index) for index in candidates]
    # No positive local maximum means no candidate, represented by the quiet floor.
    level, index = max(measured, key=lambda pair: pair[0], default=(-120.0, None))
    reference, reference_candidates = _click_reference(
        sound, sr, difference, time_s, first, last
    )
    found = level >= _SEARCH_CLICK_DB and (
        reference is None or level >= reference + _CLICK_OUTLIER_DB
    )
    join_s = index / sr if found else time_s
    return join_s, {
        "click_db": level,
        "reference_db": reference,
        "reference_candidates": reference_candidates,
        "join_s": join_s,
        "candidate_s": index / sr if index is not None else None,
        "found_by_search": found,
        "threshold_db": _SEARCH_CLICK_DB,
        "outlier_db": _CLICK_OUTLIER_DB,
        "window_s": [time_s - _SEARCH_BEFORE_S, time_s + _SEARCH_AFTER_S],
        "candidates_evaluated": len(candidates),
    }


def _skipped_search(time_s: float) -> dict:
    """Describe an audio search whose entire window follows the sound."""
    return {
        "click_db": None,
        "reference_db": None,
        "reference_candidates": 0,
        "join_s": time_s,
        "candidate_s": None,
        "found_by_search": False,
        "threshold_db": _SEARCH_CLICK_DB,
        "outlier_db": _CLICK_OUTLIER_DB,
        "window_s": [time_s - _SEARCH_BEFORE_S, time_s + _SEARCH_AFTER_S],
        "candidates_evaluated": 0,
        "reason": "the sound ends before this cut",
    }


def _spaced_peaks(values: np.ndarray, spacing: int) -> list[int]:
    padded = np.pad(values, (1, 1), constant_values=-np.inf)
    maxima = np.flatnonzero((values > padded[:-2]) & (values >= padded[2:]) & (values > 0))
    blocked = np.zeros(len(values), dtype=bool)
    chosen = []
    for index in maxima[np.argsort(-values[maxima], kind="stable")]:
        if blocked[index]:
            continue
        chosen.append(int(index))
        blocked[max(0, index - spacing + 1) : index + spacing] = True
    return sorted(chosen)


def _listen_wav(path: Path, sound: np.ndarray, sr: int, join_s: float) -> None:
    # Keep a fixed clock even at the file edges: the join is always at 3 s.
    first = _sample(join_s - 2, sr)
    last = first + 4 * sr
    padded = np.zeros(_sample(5.5, sr), dtype=np.float32)
    low, high = max(0, first), min(len(sound), last)
    if high > low:
        padded[sr + low - first : sr + high - first] = sound[low:high]
    pcm = (np.clip(padded, -1, 1) * 32767).round().astype("<i2")
    with atomic_output(path, allowed_existing_suffixes=frozenset({".wav"})) as temporary:
        with wave.open(str(temporary), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(pcm.tobytes())


def _listen(transcript: Transcript) -> tuple[bool | None, dict]:
    words = [word for word in transcript.words if word.end > 1 and word.start < 5]
    if len(words) < 3 or any(word.conf is None or not math.isfinite(word.conf) for word in words):
        return None, {"reason": "fewer than three words with speech confidence"}
    at_cut = [word.conf for word in words if word.end >= 2.7 and word.start <= 3.3]
    others = [word.conf for word in words if word.end < 2.7 or word.start > 3.3]
    if not at_cut or not others:
        return None, {"reason": "no speech confidence at the cut or in its context"}
    lowest, median = min(at_cut), float(np.median(others))
    return lowest < 0.5 and median - lowest >= 0.3, {
        "lowest_confidence": lowest,
        "context_median_confidence": median,
    }


def check_without_edl(
    render: Path,
    *,
    backend: Backend | None,
    thresholds: Thresholds = Thresholds(),  # noqa: B008 - frozen value
    work_dir: Path,
) -> DetectResult:
    """Check detected joins and structure, explicitly skipping source comparisons.

    Join measurements share one native-rate decode. Structure performs its own
    native decode through its public API. Platform checks belong to T11 and are
    not included in this task's coverage map.
    """
    cuts = detect_cuts(render)
    boundary_frames = video.grey_frames(render, w=64) if cuts else None
    info = ffmpeg.probe(render, count_frames=False)
    if info.has_audio:
        sr, sound = audio.decode(render)
        if not len(sound):
            raise ValidationError("audio", "render audio must not be empty")
        difference = np.concatenate(([0.0], np.abs(np.diff(sound.astype(np.float64)))))
    else:
        sr, sound, difference = None, None, None
    results, findings, searches, flashes = [], [], [], []
    for cut in cuts:
        if sound is not None and sr is not None and difference is not None:
            search_first, _ = _search_bounds(len(sound), sr, cut.time_s)
            if search_first >= len(sound):
                join_s, search = cut.time_s, _skipped_search(cut.time_s)
            else:
                join_s, search = _place_join(sound, sr, difference, cut.time_s)
            searches.append({"cut": cut.index, **search})
            click_db = search["click_db"]
            pause_s = (
                None
                if search.get("reason")
                else joins.pause_len_at(sound, sr, join_s)
            )
        else:
            join_s, search = cut.time_s, None
            click_db = pause_s = None
        step = joins.frame_step_at(
            boundary_frames, cut.frame, radius=_reference_radius(float(info.fps or 30))
        )
        flash = flash_at(
            boundary_frames, cut.frame, fps=float(info.fps or 30),
            touch_frames=thresholds.touch_cut_frames,
        )
        flashes.append(flash)
        row = DetectedCutResult(
            cut,
            join_s,
            click_db,
            pause_s,
            step["duplicate"],
            step["black"],
            None,
            None if flash["reason"] else flash["flash"],
        )
        results.append(row)
        if row.flash:
            findings.append(
                flash_finding(
                    flash, cut.index, cut.time_s, fps=float(info.fps or 30),
                    video_offset_s=info.video_offset_s,
                )
            )
        label = f"Cut {cut.index} at {format_time(cut.time_s)}"
        if search is not None and search["found_by_search"]:
            reference = search["reference_db"]
            context = (
                f", {row.click_db - reference:.1f} dB above the sound around it"
                if reference is not None
                else ""
            )
            findings.append(
                Finding(
                    "join.click",
                    "fail",
                    f"{label}: click at the join ({row.click_db:+.1f} dB{context}).",
                    cut=cut.index,
                    time_s=join_s,
                    suggested_action="Re-render this cut with a short audio fade (5 to 30 ms).",
                    evidence=search,
                )
            )
        for present, message in (
            (row.duplicate, "a frame is repeated at the cut"),
            (row.black, "a black frame shows at the cut"),
        ):
            if present:
                findings.append(
                    Finding(
                        "join.frames",
                        "fail",
                        f"{label}: {message}.",
                        cut=cut.index,
                        time_s=cut.time_s,
                        suggested_action="Check the boundary frames and re-render this cut.",
                        evidence=step,
                    )
                )
    if backend is not None and results and sound is not None and sr is not None:
        directory = validate_output_dir(work_dir)
        paths = [directory / f"detected-cut-{row.cut.index:02d}.wav" for row in results]
        for path, row in zip(paths, results, strict=True):
            _listen_wav(path, sound, sr, row.join_s)
        transcripts = backend.transcribe_many(paths)
        for index, (path, row) in enumerate(zip(paths, results, strict=True)):
            listen, evidence = _listen(transcripts[path])
            results[index] = replace(row, listen=listen)
            if listen:
                findings.append(
                    Finding(
                        "join.listen",
                        "info",
                        f"Cut {row.cut.index} at {format_time(row.cut.time_s)}: "
                        "the speech engine is unsure across this cut; listen to it.",
                        cut=row.cut.index,
                        time_s=row.join_s,
                        evidence=evidence,
                    )
                )
    whole = structure.check_structure(
        render, cuts_s=[row.join_s for row in results], thresholds=thresholds
    )
    findings.extend(whole.findings)
    findings.append(
        Finding(
            "detect.cuts",
            "info",
            f"{len(cuts)} clear {'cut' if len(cuts) == 1 else 'cuts'} found in the picture. "
            "Jump cuts inside one take may be missed; pass the edit list for a full check.",
            evidence={
                "cuts": [cut.to_dict() for cut in cuts],
                "audio_searches": searches,
                "jump_cuts_may_be_missed": True,
            },
        )
    )
    checks = {
        check: {"ran": True, "reason": None}
        for check in CHECK_IDS
        if not check.startswith("platform.")
    }
    for check in _SOURCE_CHECKS:
        checks[check] = {"ran": False, "reason": "needs the source and edit list"}
    checks["plan.holds"] = {"ran": False, "reason": "needs the edit list"}
    checks["edit_list.evidence"] = {"ran": False, "reason": "needs the edit list"}
    checks["detect.cuts"]["note"] = (
        "Jump cuts inside one take may be missed; pass the edit list for a full check."
    )
    if sound is None:
        checks["join.click"] = {"ran": False, "reason": "no audio in the render"}
        checks["join.listen"] = {"ran": False, "reason": "no audio in the render"}
    elif backend is None:
        checks["join.listen"] = {"ran": False, "reason": "no speech engine"}
    if not cuts:
        checks["picture.flash"] = {"ran": False, "reason": "no cuts found in the picture"}
    elif all(flash["reason"] for flash in flashes):
        reasons = {flash["reason"] for flash in flashes}
        checks["picture.flash"] = {
            "ran": False,
            "reason": next(iter(reasons)) if len(reasons) == 1 else "flash windows unavailable",
        }
    return DetectResult(cuts, results, whole, findings, checks)
