"""Locate source picture and sound independently on the rendered timeline."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

import numpy as np

from yap_studio import audio, ffmpeg, video
from yap_studio.checks.model import Finding, Thresholds, format_time
from yap_studio.edl.model import PictureRegion
from yap_studio.errors import InputError, ValidationError
from yap_studio.grade import GradeSpec
from yap_studio.plan import Plan, PlanSegment
from yap_studio.video import Mapping as PictureMapping


@dataclass(frozen=True)
class SegmentSync:
    """Independent picture and sound landings for one planned segment.

    Correlations use mean-centered plain audio. ``whitened_corr`` and
    ``ambiguity`` retain the locator's peak measurements for the middle needle.
    """

    index: int
    source_id: str | None
    measured: bool = False
    reason: str | None = None
    mapping: PictureMapping | None = None
    video_frame: int | None = None
    video_landing_s: float | None = None
    video_vs_plan_frames: int | None = None
    frame_distance: float | None = None
    frame_second_best: float | None = None
    audio_landing_s: float | None = None
    audio_vs_plan_ms: float | None = None
    corr: float | None = None
    whitened_corr: float | None = None
    ambiguity: float | None = None
    av_offset_ms: float | None = None
    confidence: Literal["high", "low"] | None = None
    head_s: float | None = None
    head_corr: float | None = None
    tail_s: float | None = None
    tail_corr: float | None = None
    probe_frame: int | None = None
    masked_region: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


# Picture landings search this many frames either side of the expected position.
_SEARCH_FRAMES = 25
# A region-masked window must keep at least this share of the picture to compare.
_MIN_COMPARED = 0.2


@dataclass(frozen=True)
class SyncResult:
    """Segment measurements and grouped findings."""

    segments: list[SegmentSync]
    drift_ms_per_cut: float | None
    findings: list[Finding]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True)
class _Media:
    """Keep a decoded picture and its frame grid together with the sound."""

    info: ffmpeg.MediaInfo
    frames: np.ndarray | None
    sound: np.ndarray | None
    rate: Fraction | None = None


@dataclass(frozen=True)
class _AudioMatch:
    """A whitened peak location with plain correlation for confidence."""

    time_s: float
    corr: float
    whitened_corr: float
    ambiguity: float


def _decode(
    path: Path, sample_rate: int, stream: int | None = None, *, plan_fps: Fraction | None = None,
    grade: GradeSpec | None = None,
) -> _Media:
    info = ffmpeg.probe(path, count_frames=False)
    rate = video.source_rate(info, plan_fps) if plan_fps is not None else info.fps
    return _Media(
        info,
        video.grey_frames(
            path, w=64, **({"rate": rate} if rate != info.fps else {}),
            **({"grade": grade} if grade is not None and grade.active else {}),
        ) if info.has_video else None,
        audio.decode(path, sr=sample_rate, **({"stream": stream} if stream is not None else {}))[1]
        if info.has_audio else None,
        rate,
    )


def _plain_corr(needle: np.ndarray, hay: np.ndarray, sr: int, time_s: float) -> float:
    first = math.floor(time_s * sr + 0.5)
    last = first + len(needle)
    if not len(needle) or first < 0 or last > len(hay):
        return 0.0
    query = np.asarray(needle, dtype=np.float64)
    matched = np.asarray(hay[first:last], dtype=np.float64)
    query = query - query.mean()
    matched = matched - matched.mean()
    energy = float(np.linalg.norm(query) * np.linalg.norm(matched))
    if energy == 0:
        return 0.0
    return float(np.clip(np.dot(query, matched) / energy, -1.0, 1.0))


def _locate_audio(
    source: np.ndarray | None,
    render: np.ndarray | None,
    start: float,
    end: float,
    expect: float,
    sr: int,
) -> _AudioMatch | None:
    if source is None or render is None:
        return None
    first, last = (math.floor(t * sr + 0.5) for t in (start, end))
    if first < 0 or last > len(source) or last - first < 2:
        return None
    needle = source[first:last]
    try:
        located = audio.locate(needle, render, sr, expect_s=expect, search_s=0.35)
    except ValidationError:
        # A truncated render can end before this needle's search window.
        return None
    # Whitening chooses the peak and scores ambiguity. Plain, mean-centered
    # correlation measures fidelity without its one-sample-wide peak penalty.
    return _AudioMatch(
        time_s=located.time_s,
        corr=_plain_corr(needle, render, sr, located.time_s),
        whitened_corr=located.corr,
        ambiguity=located.ambiguity,
    )


def _probe_state(
    segment: PlanSegment,
    probe: int,
    drift_frames: int,
    rendered: _Media,
    render_fps: float,
    masks: Sequence[tuple[Fraction, Fraction]],
    regions: Sequence[PictureRegion],
) -> tuple[bool, np.ndarray | None]:
    """Return whether a probe's picture window can be compared, and its exclusion."""
    q = segment.out_f0 + probe - segment.src_f0 + drift_frames
    first = rendered.info.video_offset_s + (q - 4) / render_fps
    last = rendered.info.video_offset_s + (q + 5) / render_fps
    if any(float(a) < last and float(b) > first for a, b in masks):
        return False, None
    if rendered.frames is None:
        return True, None
    # Regions cover the whole search span, not only the 9-frame window.
    span = _SEARCH_FRAMES / render_fps
    rects = [
        region.rect
        for region in regions
        if float(region.start) < last + span and float(region.end) > first - span
    ]
    if not rects:
        return True, None
    excluded = video.exclusion_mask(
        rendered.frames.shape[1:],
        rects,
        rendered.info.width or rendered.frames.shape[2],
        rendered.info.height or rendered.frames.shape[1],
    )
    return float(np.mean(~excluded)) >= _MIN_COMPARED, excluded


def _probe(
    segment: PlanSegment,
    drift_frames: int,
    rendered: _Media,
    render_fps: float,
    masks: Sequence[tuple[Fraction, Fraction]],
    regions: Sequence[PictureRegion],
) -> tuple[int | None, np.ndarray | None]:
    """Choose the middle frame, or the nearest frame whose window is not masked."""
    midpoint = segment.src_f0 + segment.n_frames // 2
    usable, excluded = _probe_state(
        segment, midpoint, drift_frames, rendered, render_fps, masks, regions
    )
    if usable:
        return midpoint, excluded
    low, high = segment.src_f0 + 4, segment.src_f1 - 5
    for step in range(1, segment.n_frames):
        candidates = [probe for probe in (midpoint + step, midpoint - step) if low <= probe <= high]
        if not candidates:
            break
        for probe in candidates:
            usable, excluded = _probe_state(
                segment, probe, drift_frames, rendered, render_fps, masks, regions
            )
            if usable:
                return probe, excluded
    return None, None


def measure_segments(
    render: Path,
    plan: Plan,
    sources: Mapping[str, Path | None],
    *,
    source_streams: Mapping[str, int | None] | None = None,
    masks: Sequence[tuple[Fraction, Fraction]] = (),
    regions: Sequence[PictureRegion] = (),
    sample_rate: int = 16000,
    grade: GradeSpec | None = None,
) -> list[SegmentSync]:
    """Decode each distinct file once, then measure source landings per segment.

    The picture landing is measured around the segment's middle frame. Time-only
    ``masks`` hide the whole picture: when one covers the middle window, the
    nearest unmasked window inside the segment is used instead, recorded as
    ``probe_frame``. ``regions`` leave their rectangles out of the comparison.
    """
    if sample_rate <= 0:
        raise ValidationError("sample_rate", "must be positive")
    rendered = _decode(render, sample_rate)
    # A VFR render's native decode cannot stand in for a source on the plan grid.
    cache: dict[tuple[Path, int | None], _Media | None] = (
        {} if rendered.info.variable_frame_rate else {(Path(render).resolve(), None): rendered}
    )
    rows: list[SegmentSync] = []
    drift_frames = 0
    previous_offset_s = 0.0
    render_fps = float(rendered.info.fps or plan.fps)
    plan_fps = float(plan.fps)
    for segment in plan.segments:
        row = SegmentSync(segment.index, segment.source_id)
        if segment.source_id is None:
            rows.append(replace(row, reason="gap"))
            continue
        path = sources.get(segment.source_id)
        if path is None or not Path(path).is_file():
            rows.append(replace(row, reason="source not found"))
            continue
        path = Path(path).resolve()
        stream = (source_streams or {}).get(segment.source_id)
        key = (path, stream)
        if key not in cache:
            try:
                cache[key] = (
                    _decode(path, sample_rate, plan_fps=plan.fps,
                            **({"grade": grade} if grade is not None and grade.active else {}))
                    if stream is None else
                    _decode(path, sample_rate, stream, plan_fps=plan.fps,
                            **({"grade": grade} if grade is not None and grade.active else {}))
                )
            except InputError:
                cache[key] = None
        source = cache[key]
        if source is None:
            rows.append(replace(row, reason="no match"))
            continue
        no_audio_reason = (
            "no audio in the render"
            if rendered.sound is None
            else "no audio in the source"
            if source.sound is None
            else None
        )

        start = source.info.video_offset_s + segment.src_f0 / plan_fps
        end = source.info.video_offset_s + segment.src_f1 / plan_fps
        output_start = float(segment.out_time)
        # Head and tail refer to the source range's edges, outside the fades.
        tracked_start = (
            output_start
            + rendered.info.video_offset_s
            + drift_frames / render_fps
            + previous_offset_s
        )
        head = _locate_audio(
            source.sound,
            rendered.sound,
            start + 0.03,
            start + 0.33,
            tracked_start + 0.03,
            sample_rate,
        )
        tail = _locate_audio(
            source.sound,
            rendered.sound,
            end - 0.33,
            end - 0.03,
            tracked_start + end - start - 0.33,
            sample_rate,
        )
        row = replace(
            row,
            head_s=head.time_s - 0.03 if head else None,
            head_corr=head.corr if head else None,
            tail_s=tail.time_s + 0.33 if tail else None,
            tail_corr=tail.corr if tail else None,
        )
        if segment.n_frames / plan_fps < 0.4 and no_audio_reason is None:
            rows.append(replace(row, reason="shorter than 0.4 s"))
            continue

        p, excluded = _probe(segment, drift_frames, rendered, render_fps, masks, regions)
        masked = p is None
        if p is None:
            p = segment.src_f0 + segment.n_frames // 2
        q_plan = segment.out_f0 + p - segment.src_f0
        q_expect = q_plan + drift_frames
        source_fps = float(source.rate or source.info.fps or plan.fps)
        p_src = math.floor(p / plan_fps * source_fps + 0.5)
        reference = source.info.video_offset_s + p_src / source_fps
        planned_time = output_start + reference - start

        if not masked and source.frames is not None and rendered.frames is not None:
            indices = [
                math.floor((p_src / source_fps + i / render_fps) * source_fps + 0.5)
                for i in range(-4, 5)
            ]
            # Keep cached decodes as uint8; promote only these small fitting windows.
            mapping_frames = np.asarray(
                rendered.frames[max(0, q_expect - 3) : q_expect + 4], dtype=np.float32
            )
            if min(indices) >= 0 and max(indices) < len(source.frames) and len(mapping_frames):
                needle = np.asarray(source.frames[indices], dtype=np.float32)
                # A uniform frame matches every crop equally and cannot establish
                # geometry. Keep it in the timing needle, but not in the mapping fit.
                mapping_source = needle[2:7]
                mapping_source = mapping_source[np.ptp(mapping_source, axis=(1, 2)) > 0]
                mapping_frames = mapping_frames[np.ptp(mapping_frames, axis=(1, 2)) > 0]
            else:
                mapping_source = None
            if mapping_source is not None and len(mapping_source) and len(mapping_frames):
                mapping = video.estimate_mapping(
                    mapping_source,
                    mapping_frames,
                    render_valid=None if excluded is None else ~excluded,
                )
                first = max(0, q_expect - 4 - _SEARCH_FRAMES)
                last = min(len(rendered.frames), q_expect + 5 + _SEARCH_FRAMES)
                # Only resample the search region, not the entire render per segment.
                # The crop/fit rectangle is the shared picture's native analysis
                # resolution. Stretching it to the outer canvas adds mismatch.
                size = (max(1, round(mapping.height)), max(1, round(mapping.width)))
                mapped_source = video.apply_mapping(needle, mapping, role="source", size=size)
                mapped_render = video.apply_mapping(
                    rendered.frames[first:last],
                    mapping,
                    role="render",
                    size=size,
                )
                compared = (
                    None
                    if excluded is None
                    else video.valid_in_comparison(excluded, mapping, size)
                )
                if (
                    compared is None or float(np.mean(compared)) >= _MIN_COMPARED
                ) and len(mapped_render) >= len(mapped_source):
                    q, distance, second = video.locate_sequence(
                        mapped_source,
                        mapped_render,
                        expect=q_expect - 4 - first,
                        search=_SEARCH_FRAMES,
                        valid=compared,
                    )
                    q += first + 4
                    row = replace(
                        row,
                        mapping=mapping,
                        video_frame=q,
                        video_landing_s=rendered.info.video_offset_s + q / render_fps,
                        video_vs_plan_frames=q - q_plan,
                        frame_distance=distance,
                        frame_second_best=second if math.isfinite(second) else None,
                        probe_frame=p,
                        masked_region=excluded is not None,
                    )

        if no_audio_reason is not None:
            measured = row.video_landing_s is not None
            if measured:
                drift_frames = row.video_vs_plan_frames
            rows.append(replace(row, measured=measured, reason=no_audio_reason))
            continue

        expected_audio = (
            (row.video_landing_s if row.video_landing_s is not None else planned_time)
            - 0.25
            + previous_offset_s
        )
        located = _locate_audio(
            source.sound,
            rendered.sound,
            reference - 0.25,
            reference + 0.25,
            expected_audio,
            sample_rate,
        )
        if located is not None:
            row = replace(
                row,
                corr=located.corr,
                whitened_corr=located.whitened_corr,
                ambiguity=located.ambiguity,
            )
        if located is not None and located.whitened_corr > 0:
            landing = located.time_s + 0.25
            row = replace(
                row,
                audio_landing_s=landing,
                audio_vs_plan_ms=(landing - planned_time) * 1000,
            )
        if row.video_landing_s is not None and row.audio_landing_s is not None:
            offset = (row.audio_landing_s - row.video_landing_s) * 1000
            high = (
                row.corr >= 0.8
                and row.ambiguity < 0.8
                and row.frame_second_best is not None
                and row.frame_distance < 0.8 * row.frame_second_best
            )
            row = replace(
                row, measured=True, av_offset_ms=offset, confidence="high" if high else "low"
            )
            drift_frames = row.video_vs_plan_frames
            previous_offset_s = offset / 1000
        else:
            row = replace(row, reason="picture masked" if masked else "no match")
        rows.append(row)
    return rows


def _runs(
    rows: Sequence[SegmentSync],
    predicate: Callable[[SegmentSync], bool],
) -> list[list[SegmentSync]]:
    runs: list[list[SegmentSync]] = []
    current: list[SegmentSync] = []
    for row in rows:
        if predicate(row):
            if current and row.index != current[-1].index + 1:
                runs.append(current)
                current = []
            current.append(row)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def check_sync(
    render: Path,
    plan: Plan,
    sources: Mapping[str, Path | None],
    *,
    source_streams: Mapping[str, int | None] | None = None,
    masks: Sequence[tuple[Fraction, Fraction]] = (),
    regions: Sequence[PictureRegion] = (),
    thresholds: Thresholds = Thresholds(),  # noqa: B008 - frozen value
    grade: GradeSpec | None = None,
) -> SyncResult:
    """Check offsets and drift using only confident A/V measurements."""
    rows = measure_segments(
        render, plan, sources, masks=masks, regions=regions, source_streams=source_streams,
        **({"grade": grade} if grade is not None and grade.active else {}),
    )
    good = [
        row
        for row in rows
        if row.av_offset_ms is not None and row.confidence == "high"
    ]
    drift = None
    if len(good) >= 2:
        drift = float(
            np.polyfit([row.index for row in good], [row.av_offset_ms for row in good], 1)[0]
        )
    drifting = drift is not None and abs(drift) >= thresholds.drift_warn_ms_per_cut
    planned = {segment.index: segment for segment in plan.segments}
    findings: list[Finding] = []
    warn_ms = thresholds.av_warn_frames / float(plan.fps) * 1000

    def fails(row: SegmentSync) -> bool:
        return (
            row.av_offset_ms < -thresholds.av_early_fail_ms
            or row.av_offset_ms > thresholds.av_late_fail_ms
        )

    for run in _runs(
        rows,
        lambda row: (
            row.av_offset_ms is not None
            and row.confidence == "high"
            and (fails(row) or abs(row.av_offset_ms) > warn_ms)
        ),
    ):
        offsets = [row.av_offset_ms for row in run]
        first, last = planned[run[0].index], planned[run[-1].index]
        start, end = float(first.out_time), float(last.out_time + last.n_frames / plan.fps)
        average = float(np.mean(offsets))
        direction = "late" if average >= 0 else "early"
        if drifting:
            action = (
                "Audio was encoded per clip and joined without re-encoding. "
                "Encode the audio once for the whole video."
            )
        else:
            action = (
                f"Shift the audio by {-average:.0f} ms to match the picture, "
                "or re-render with the audio aligned."
            )
        if max(offsets) - min(offsets) <= 2:
            message = (
                f"Sound is {abs(average):.0f} ms {direction} from "
                f"{format_time(start)} to {format_time(end)}."
            )
        elif min(offsets) > 0 or max(offsets) < 0:
            message = (
                f"From {format_time(start)} to {format_time(end)}, sound runs {direction}, "
                f"from {abs(offsets[0]):.0f} ms at the start to {abs(offsets[-1]):.0f} ms "
                "by the end."
            )
            if drift is not None:
                message += f" It changes by {drift:+.1f} ms per cut."
        else:
            message = (
                f"From {format_time(start)} to {format_time(end)}, sound alignment "
                f"changes from {offsets[0]:+.0f} to {offsets[-1]:+.0f} ms "
                "(positive means late)."
            )
            if drift is not None:
                message += f" It changes by {drift:+.1f} ms per cut."
        findings.append(
            Finding(
                "sync.av_offset",
                "fail" if any(fails(row) for row in run) else "warn",
                f"{message} {action}",
                segment=first.index,
                time_s=start,
                suggested_action=action,
                evidence={"segments": [row.index for row in run], "offsets_ms": offsets},
            )
        )
    if drifting and len(good) >= 3:
        start = float(planned[good[0].index].out_time)
        action = "Encode the audio once for the whole video."
        findings.append(
            Finding(
                "sync.drift",
                "warn",
                f"From {format_time(start)}, sound drifts {drift:+.1f} ms per cut. {action}",
                segment=good[0].index,
                time_s=start,
                suggested_action=action,
                evidence={"drift_ms_per_cut": drift, "segments": [row.index for row in good]},
            )
        )
    for run in _runs(
        rows,
        lambda row: (
            row.video_vs_plan_frames is not None
            and abs(row.video_vs_plan_frames) > thresholds.frames_vs_plan_warn
        ),
    ):
        start = float(planned[run[0].index].out_time)
        action = "Re-render the picture from the edit list."
        findings.append(
            Finding(
                "plan.picture",
                "warn",
                f"From {format_time(start)}, the picture is off the planned frame positions. "
                f"{action}",
                segment=run[0].index,
                time_s=start,
                suggested_action=action,
                evidence={
                    "segments": [row.index for row in run],
                    "differences_frames": [row.video_vs_plan_frames for row in run],
                },
            )
        )
    audio_reasons = {"no audio in the render", "no audio in the source"}
    unmeasured = [row for row in rows if not row.measured or row.reason in audio_reasons]
    if unmeasured:
        problems = [
            row.index
            for row in unmeasured
            if row.reason not in {"gap", "picture masked", *audio_reasons}
        ]
        audio_problems = {
            reason: [row.index for row in unmeasured if row.reason == reason]
            for reason in audio_reasons
            if any(row.reason == reason for row in unmeasured)
        }
        start = float(planned[unmeasured[0].index].out_time)
        action = "Check the sources and review these parts manually." if problems else None
        messages: list[str] = []
        if problems:
            messages.append(
                f"sync could not be measured for segments {', '.join(map(str, problems))}. "
                f"{action}"
            )
        elif any(row.reason in {"gap", "picture masked"} for row in unmeasured):
            messages.append("gaps and masked picture were skipped for sync.")
        messages.extend(
            f"Audio sync was not measured: {reason} "
            f"(segments {', '.join(map(str, indices))})."
            for reason, indices in sorted(audio_problems.items())
        )
        message = f"From {format_time(start)}, {' '.join(messages)}"
        findings.append(
            Finding(
                "sync.unmeasured",
                "info",
                message,
                segment=unmeasured[0].index,
                time_s=start,
                suggested_action=action,
                evidence={
                    "segments": [{"index": row.index, "reason": row.reason} for row in unmeasured]
                },
            )
        )
    return SyncResult(rows, drift, findings)
