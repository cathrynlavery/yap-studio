"""Finished-picture cuts, centre punch-ins, speech gaps and whole-file loudness."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.checks.structure import _loudness
from yap_studio.taste.frames import FrameTiming, iter_rgb, presentation_timing, runs
from yap_studio.transcript.model import Transcript

MIN_PAUSE_S = 0.15
CUT_JUMP_RATIO = 1.7
MIN_CUT_DISTANCE = 0.04


def scaled_picture(image: np.ndarray, scale: float) -> np.ndarray:
    """Bilinearly sample a centre-scaled continuation on a small grey grid."""
    h, w = image.shape
    y = (np.arange(h) - (h - 1) / 2) / scale + (h - 1) / 2
    x = (np.arange(w) - (w - 1) / 2) / scale + (w - 1) / 2
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (y - y0)[:, None], (x - x0)[None, :]
    return (1 - fy) * ((1 - fx) * image[y0[:, None], x0] + fx * image[y0[:, None], x1]) + fy * (
        (1 - fx) * image[y1[:, None], x0] + fx * image[y1[:, None], x1]
    )


def punch_in(
    before: np.ndarray, after: np.ndarray, valid: np.ndarray | None = None
) -> dict[str, float] | None:
    """Search 1.05 through 1.6 for a better continuation outside both frames' overlays."""
    valid = np.ones(before.shape, dtype=bool) if valid is None else valid
    if valid.sum() < 30:
        return None
    source_valid = valid.astype(float)
    scores = []
    for scale in np.arange(1.05, 1.601, 0.01):
        # Scaling can move excluded source captions into valid destination rows.
        # Require every contributing bilinear sample to be outside an overlay.
        shared = valid & (scaled_picture(source_valid, float(scale)) >= 1 - 1e-12)
        if shared.sum() < 30:
            continue
        predicted = scaled_picture(before, float(scale))
        baseline = float(np.mean(np.abs(before[shared] - after[shared])))
        error = float(np.mean(np.abs(predicted[shared] - after[shared])))
        if baseline < 2 or error > 6 or error > baseline * 0.4:
            continue
        scores.append((error, float(scale), baseline))
    if not scores:
        return None
    error, scale, baseline = min(scores)
    return {"scale": round(scale, 3), "error": error, "unscaled_error": baseline}


def pause_envelope(rms: np.ndarray, step_s: float = 0.01) -> list[dict[str, float]]:
    """Find internal low-energy runs with speech on both sides."""
    if not len(rms) or rms.max(initial=0) < 1e-5:
        return []
    threshold = max(10 ** (-48 / 20), float(np.percentile(rms, 80)) * 0.055)
    silent = rms < threshold
    return [
        {"start_s": round(a * step_s, 3), "length_s": round((b - a) * step_s, 3)}
        for a, b in runs(silent)
        if a > 0 and b < len(rms) and (b - a) * step_s >= MIN_PAUSE_S - 1e-9
    ]


def audio_envelope(path: Path) -> np.ndarray:
    """Stream audio to 10 ms RMS windows, retaining only the short envelope."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    pending = bytearray()
    values = []
    with closing(ffmpeg.run_stream(command, timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)) as stream:
        for chunk in stream:
            pending.extend(chunk)
            count = len(pending) // (160 * 4)
            if count:
                samples = np.frombuffer(pending, dtype="<f4", count=count * 160).reshape(-1, 160)
                values.extend(np.sqrt(np.mean(samples.astype(float) ** 2, axis=1)).tolist())
                del samples
                del pending[: count * 160 * 4]
    return np.asarray(values)


def loudness(path: Path) -> dict[str, float]:
    """Parse one independent ebur128 true-peak pass, omitting nonfinite results."""
    result = ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "info",
            "-nostats",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    integrated, lra, peak = _loudness(result.stderr)
    return {
        key: value
        for key, value in (
            ("loudness.integrated_lufs", integrated),
            ("loudness.true_peak_dbtp", peak),
            ("loudness.lra_lu", lra),
        )
        if value is not None
    }


def transcript_pacing(transcript: Transcript, duration_s: float) -> dict[str, Any]:
    """Return speech rate and positive timestamp gaps from recognised words."""
    words = transcript.words
    gaps = [
        {"start_s": round(previous.end, 3), "length_s": round(following.start - previous.end, 3)}
        for previous, following in zip(words, words[1:], strict=False)
        if following.start - previous.end >= MIN_PAUSE_S - 1e-9
    ]
    return {"pacing.words_per_minute": len(words) * 60 / duration_s, "pacing.word_gap_pauses": gaps}


def cut_evidence(
    frames: np.ndarray, valid: np.ndarray, mean_rgb: np.ndarray | None = None
) -> list[dict[str, Any]]:
    """Score discontinuities against motion immediately before and after them.

    A gesture accelerates across adjacent frames. A hard cut has a much larger
    one-frame change, followed by a new continuing picture. Normalisation uses
    only picture pixels outside measured overlay boxes. The decision uses the
    ratio of this change to the larger immediately adjacent motion distance.
    """
    values = frames[:, valid].astype(np.float32)
    means = values.mean(axis=1)
    deviations = values.std(axis=1)
    values -= means[:, None]
    values /= np.maximum(deviations[:, None], 1)
    distances = np.r_[0, np.mean(abs(np.diff(values, axis=0)), axis=1)]
    # Normalisation erases solid fills. Use raw colour distance only when
    # both frames are flat; textured-picture decisions remain unchanged.
    colors = means[:, None] if mean_rgb is None else mean_rgb
    flat = (deviations[:-1] < 1) & (deviations[1:] < 1)
    raw = np.mean(abs(np.diff(colors, axis=0)), axis=1) / 255
    distances[1:] = np.where(flat, np.maximum(distances[1:], raw), distances[1:])
    output = []
    for n in range(1, len(frames) - 1):
        change = float(distances[n])
        adjacent = max(float(distances[n - 1]), float(distances[n + 1]), 0.01)
        ratio = change / adjacent
        if change < MIN_CUT_DISTANCE or change < distances[n - 1] or change < distances[n + 1]:
            continue
        output.append(
            {
                "frame": n,
                "distance": change,
                "score": ratio,
                "accepted": ratio >= CUT_JUMP_RATIO,
            }
        )
    return output


def picture_cuts(
    path: Path, info: ffmpeg.MediaInfo, exclusions: Sequence[dict[str, int]]
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    """Stream small pictures, retaining at most one block and its adjacent context."""
    count = info.video_frames or round((info.duration_s or 0) * float(info.fps or 30))
    pending = []
    colors = []
    first = 0
    evidence = []
    punches = []
    valid = None
    for index, picture in iter_rgb(path, info, list(range(count)), width=128):
        grey = picture.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        if valid is None:
            valid = np.ones(grey.shape, dtype=bool)
            for box in exclusions:
                x, y, w, h = (round(box[k] * 128 / 1080) for k in ("x", "y", "width", "height"))
                valid[max(0, y - 3) : y + h + 3, max(0, x - 3) : x + w + 3] = False
            if not valid.any():
                valid[:] = True
        pending.append(grey)
        colors.append(picture[valid].mean(axis=0))
        if len(pending) < 130 and index < count - 1:
            continue
        block = np.stack(pending)
        for row in cut_evidence(block, valid, np.asarray(colors)):
            # The first context frame belongs to the previous block.
            if first and row["frame"] < 2:
                continue
            local = row["frame"]
            row["frame"] += first
            evidence.append(row)
            if row["accepted"]:
                result = punch_in(block[local - 1], block[local], valid)
                if result:
                    punches.append({"frame": row["frame"], **result})
        first += len(pending) - 3
        pending = pending[-3:]
        colors = colors[-3:]
    cuts = sorted({r["frame"] for r in evidence if r["accepted"]})
    return cuts, evidence, punches


def measure(
    path: Path,
    info: ffmpeg.MediaInfo,
    exclusions: Sequence[dict[str, int]],
    *,
    timing: FrameTiming | None = None,
) -> dict[str, tuple[Any, list[int], dict[str, Any]]]:
    """Measure pacing with event evidence and bounded picture decoding."""
    fps = float(info.fps or 30)
    count = info.video_frames or round((info.duration_s or 0) * fps)
    timing = timing or presentation_timing(path, info)
    duration = timing.end_s
    cuts, candidates, punches = picture_cuts(path, info, exclusions)
    edges = [0, *cuts, count]
    shots = [
        timing.boundary(b) - timing.boundary(a) for a, b in zip(edges, edges[1:], strict=False)
    ]
    endpoints = [0, count - 1]
    shot_evidence = cuts or endpoints
    output = {
        "pacing.cut_frames": (
            cuts,
            cuts,
            {
                "detector": "adjacent-motion jump ratio outside overlay boxes",
                "candidates": [
                    row
                    for row in candidates
                    if row["accepted"] or row["score"] >= CUT_JUMP_RATIO / 2
                ],
                "weakest_accepted_score": min(
                    (row["score"] for row in candidates if row["accepted"]), default=None
                ),
                "strongest_rejected_score": max(
                    (row["score"] for row in candidates if not row["accepted"]), default=None
                ),
                "jump_ratio_threshold": CUT_JUMP_RATIO,
                "minimum_distance": MIN_CUT_DISTANCE,
                "confirmation": "abrupt picture discontinuity outside overlay boxes",
            },
        ),
        "pacing.shot_lengths_s": ([round(s, 3) for s in shots], shot_evidence, {}),
        "pacing.median_shot_length_s": (round(float(np.median(shots)), 3), shot_evidence, {}),
        "pacing.zoom_punch_ins": (
            {
                "count": len(punches),
                "frames": [p["frame"] for p in punches],
                "punches": punches,
                "per_minute": len(punches) * 60 / duration,
            },
            sorted({p["frame"] for p in punches}),
            {},
        ),
    }
    if info.has_audio:
        pauses = pause_envelope(audio_envelope(path))
        output["pacing.pause_s"] = (
            {
                "pauses": pauses,
                "count": len(pauses),
                "median_s": round(float(np.median([p["length_s"] for p in pauses])), 3)
                if pauses
                else None,
            },
            sorted({timing.frame_at(p["start_s"]) for p in pauses}),
            {"window_s": 0.01, "minimum_pause_s": MIN_PAUSE_S, "includes_music": True},
        )
        for key, value in loudness(path).items():
            output[key] = (value, endpoints, {"frame_range": endpoints})
    return output
