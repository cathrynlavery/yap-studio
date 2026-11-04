"""Fit replacement speech and blend it against the cut's own PCM."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from yap_studio import audio, ffmpeg, paths
from yap_studio.checks.joins import click_at, click_peak_dbfs_at
from yap_studio.checks.model import Thresholds
from yap_studio.stages import edges, take_levels
from yap_studio.voice.model import VoiceError

RATE = 48000
FADE = 720


def write_wav(target: Path, sound: np.ndarray) -> None:
    """Atomically retain exact float PCM without quantising patch edges."""
    frames = np.asarray(sound, dtype="<f4")
    channels = frames.shape[1]
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + frames.nbytes,
        b"WAVE",
        b"fmt ",
        16,
        3,
        channels,
        RATE,
        RATE * channels * 4,
        channels * 4,
        32,
        b"data",
        frames.nbytes,
    )
    paths.validate_output_dir(target.parent)
    with paths.atomic_output(target) as staged:
        staged.write_bytes(header + frames.tobytes())


def decode(path: Path, channels: int) -> np.ndarray:
    result = ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(RATE),
            "-f",
            "f32le",
            "-",
        ],
        timeout_s=180,
        binary=True,
    )
    sound = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, channels).copy()
    if not len(sound) or not np.isfinite(sound).all():
        raise VoiceError("The generated speech has no usable audio.", code="voice_audio_invalid")
    return sound


def envelope(meta: dict) -> np.ndarray:
    w = np.ones(meta["samples"], dtype=np.float32)
    first, last = meta["fade_in_samples"], meta["fade_out_samples"]
    if first:
        w[:first] = audio.raised_cosine(first)
    if last:
        w[-last:] = audio.raised_cosine(last)[::-1]
    return w[:, None]


def blend(
    original: np.ndarray, patch: np.ndarray, meta: dict, *, source_start_sample: int
) -> np.ndarray:
    """Blend signal, never a pre-faded copy of a different source decode."""
    result = original.copy()
    first = meta["start_sample"] - source_start_sample
    low, high = max(0, first), min(len(result), first + len(patch))
    if high <= low:
        return result
    signal = patch
    if signal.shape[1] != result.shape[1]:
        signal = (
            np.repeat(signal, 2, axis=1)
            if signal.shape[1] == 1
            else signal.mean(axis=1, keepdims=True)
        )
    a, b = low - first, high - first
    weight = envelope(meta)[a:b]
    result[low:high] = original[low:high] * (1 - weight) + signal[a:b] * weight
    return result


def level(sound: np.ndarray) -> float:
    return audio.rms_db(sound, len(sound) / RATE / 2, RATE, len(sound) / RATE / 2)


def _speech(sound: np.ndarray) -> np.ndarray:
    """Keep speech frames using the edge stage's local energy floor."""
    levels = edges._levels(sound.mean(axis=1)[::3])
    if not len(levels):
        return sound[:0]
    gate = min(float(np.max(levels)) - 6, float(np.percentile(levels, 10)) + 12)
    active = np.repeat(levels > gate, 480)[: len(sound)]
    return sound[: len(active)][active]


def _snap(sound: np.ndarray, nominal: int) -> int:
    """Find the quietest centred 10 ms within 40 ms, preferring a near tie."""
    mono = sound.mean(axis=1).astype(np.float64)
    power = np.r_[0.0, np.cumsum(mono * mono)]
    candidates = np.arange(max(240, nominal - 1920), min(len(sound) - 240, nominal + 1920) + 1, 48)
    if not len(candidates):
        return min(len(sound), max(0, nominal))
    energy = (power[candidates + 240] - power[candidates - 240]) / 480
    best = np.flatnonzero(energy <= float(energy.min()) + 1e-12)
    return int(candidates[min(best, key=lambda i: abs(candidates[i] - nominal))])


def edge_check(sound: np.ndarray, first: int, last: int) -> dict:
    mono = sound.mean(axis=1)
    thresholds = Thresholds()
    measurements = []
    for point in (first, last):
        point = min(len(mono) - 1, max(0, point))
        sharpness = click_at(mono, RATE, point / RATE)
        peak = click_peak_dbfs_at(mono, RATE, point / RATE)
        measurements.append({"sample": point, "db": sharpness, "peak_dbfs": peak})
    failed = any(
        row["db"] > thresholds.click_fail_db and row["peak_dbfs"] > thresholds.click_floor_dbfs
        for row in measurements
    )
    return {"id": "splice.click", "status": "fail" if failed else "pass", "edges": measurements}


def _tone(
    sound: np.ndarray, first: int, last: int, word_spans: list[tuple[float, float]], count: int
) -> tuple[np.ndarray, float]:
    """Join nearby quiet gaps against an independently measured source floor."""
    low, high = max(0, first - 10 * RATE), min(len(sound), last + 10 * RATE)
    levels = [
        edges._levels(part.mean(axis=1)[::3])
        for part in (sound[low:first], sound[last:high])
        if len(part)
    ]
    if not levels:
        raise VoiceError(
            "There is no quiet gap nearby for your room tone.", code="voice_room_tone_missing"
        )
    floor = float(np.percentile(np.concatenate(levels), 10))
    quiet = np.ones(high - low, dtype=bool)
    quiet[max(0, first - low) : min(high - low, last - low)] = False
    for a, b in word_spans:
        a, b = max(low, round(a * RATE)), min(high, round(b * RATE))
        if b > a:
            quiet[a - low : b - low] = False
    bounds = np.flatnonzero(np.diff(np.r_[False, quiet, False].astype(int)))
    chunks = [
        sound[low + a : low + b]
        for a, b in zip(bounds[::2], bounds[1::2], strict=True)
        if b - a >= round(0.1 * RATE)
    ]
    if not chunks:
        raise VoiceError(
            "There is no quiet gap nearby for your room tone.",
            code="voice_room_tone_missing",
            suggested_action=(
                "Choose a moment with a quiet gap of at least 100 ms within 10 seconds."
            ),
        )

    def distance(chunk: np.ndarray) -> float:
        return abs(level(chunk.mean(axis=1)) - floor)

    chunks.sort(key=distance)
    closest = distance(chunks[0])
    chunks = [chunk for chunk in chunks if distance(chunk) <= closest + 3]
    # Use several gaps before repeating any one of them. Each source is at least 100 ms.
    portion = max(round(0.1 * RATE), int(np.ceil(count / min(4, len(chunks)))))
    chunks = [chunk[:portion] for chunk in chunks]
    result = chunks[0].copy()
    index = 1
    while len(result) < count:
        source = chunks[index % len(chunks)]
        fade = min(480, len(source) // 2, len(result) // 2)
        w = audio.raised_cosine(fade)[:, None]
        result = np.concatenate(
            (result[:-fade], result[-fade:] * (1 - w) + source[:fade] * w, source[fade:])
        )
        index += 1
    result = result[:count].copy()
    result *= 10 ** ((floor - level(result.mean(axis=1))) / 20)
    return result, floor


def _floor(sound: np.ndarray, first: int, last: int) -> float:
    levels = [edges._levels(part.mean(axis=1)[::3])
              for part in (sound[max(0, first - 10 * RATE):first],
                           sound[last:min(len(sound), last + 10 * RATE)]) if len(part)]
    usable = [part for part in levels if len(part)]
    return float(np.percentile(np.concatenate(usable), 10)) if usable else -80.0


def room_tone_check(patch: np.ndarray, floor: float) -> dict:
    """Compare the patch's own quiet frames with her independent source floor."""
    levels = edges._levels(patch.mean(axis=1)[::3])
    gate = min(float(np.max(levels)) - 6, float(np.percentile(levels, 10)) + 12)
    quiet = levels <= gate
    # Edge fades are not room-tone gaps. Exclude their 20 ms analysis windows too.
    margin = int(np.ceil((FADE / RATE + edges.WINDOW_S / 2) / edges.HOP_S))
    quiet[:margin] = False
    quiet[-margin:] = False
    measured = float(np.median(levels[quiet])) if quiet.any() else float(np.percentile(levels, 10))
    difference = abs(measured - floor) if quiet.any() else 0.0
    return {
        "id": "splice.room_tone",
        "status": ("pass" if difference <= 3 else "warn") if quiet.any() else "skipped",
        "difference_db": difference,
        "source_floor_db": floor,
        "patch_floor_db": measured,
        "measured": bool(quiet.any()),
        **({"reason": "Speech fills the slot; no quiet frames remain."} if not quiet.any() else {}),
    }


def build_patch(
    original: np.ndarray,
    generated: Path,
    start_s: float,
    end_s: float,
    work: Path,
    *,
    word_spans: list[tuple[float, float]],
    word_range: tuple[int, int],
    pace: str = "fit",
    recorded: bool = False,
) -> tuple[np.ndarray, dict, dict]:
    """Fit the slot without moving later words, then measure the proposed splice."""
    channels = original.shape[1]
    sound = decode(generated, channels)
    levels = edges._levels(sound.mean(axis=1)[::3])
    gate = min(float(np.max(levels)) - 6, float(np.percentile(levels, 10)) + 12)
    voiced = np.flatnonzero(levels > gate)
    if not len(voiced):
        raise VoiceError("No speech was found in the generated audio.", code="voice_audio_invalid")
    if not recorded:
        sound = sound[max(0, int(voiced[0]) * 480 - 240):
                      min(len(sound), int(voiced[-1]) * 480 + 720)]
    first, last = _snap(original, round(start_s * RATE)), _snap(original, round(end_s * RATE))
    if last <= first:
        raise VoiceError("The replacement slot is empty.", code="voice_slot_invalid")
    before = word_spans[: word_range[0]]
    after = word_spans[word_range[1] + 1 :]
    if before:
        first = max(first, math.ceil(max(b for _, b in before) * RATE))
    if after:
        last = min(last, math.floor(min(a for a, _ in after) * RATE))
    if last <= first:
        raise VoiceError(
            "The new speech cannot fit without covering another word.", code="voice_speech_too_long"
        )
    duration = len(sound) / RATE
    needed = round(len(sound) * (0.8 if pace == "fit" else 1))
    if needed > last - first:
        next_word = min(
            (a for a, _ in after), default=min(len(original) / RATE, end_s + 0.25)
        )
        available = max(last, min(len(original), math.floor(next_word * RATE) - 480))
        if needed > available - first:
            raise VoiceError(
                f"The new speech is {duration:.3f} s; the slot is "
                f"{(available - first) / RATE:.3f} s, even using the next quiet gap.",
                code="voice_speech_too_long",
                suggested_action="Use fewer words or record the line again.",
            )
        last = first + needed
    count = last - first
    stretch = max(0.8, min(1.25, count / len(sound))) if pace == "fit" else 1.0
    raw, fitted = work / "speech.wav", work / "fitted.wav"
    if pace == "fit":
        write_wav(raw, sound)
        ffmpeg.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(raw),
             "-af", f"atempo={1 / stretch:.12g}", "-c:a", "pcm_f32le", str(fitted)],
            timeout_s=180,
        )
        sound = decode(fitted, channels)[:count]
    # atempo may leave a few milliseconds unfilled; keep the whole patch's clock fixed.
    gap = (count - len(sound)) / RATE
    surrounding = []
    for a, b in word_spans:
        a, b = max(a, start_s - 1.5), min(b, end_s + 1.5)
        if b > a:
            surrounding.append(
                original[max(0, round(a * RATE)) : min(len(original), round(b * RATE))]
            )
    reference = (
        _speech(np.concatenate(surrounding)) if surrounding else _speech(original[first:last])
    )
    speech = _speech(sound)
    if not len(reference) or not len(speech) or level(speech) < -80:
        raise VoiceError(
            "The speech is too quiet to match its neighbours.", code="voice_audio_invalid"
        )
    gain = level(reference) - level(speech)
    sound *= 10 ** (gain / 20)
    if recorded:
        floor = _floor(original, first, last)
        patch = np.zeros((count, channels), dtype=np.float32)
        if len(sound) < count:
            tone, floor = _tone(original, first, last, word_spans, count - len(sound))
            patch[len(sound):] = tone
    else:
        patch, floor = _tone(original, first, last, word_spans, count)
    speech_fade = min(FADE, len(sound) // 2)
    sound[:speech_fade] *= audio.raised_cosine(speech_fade)[:, None]
    sound[-speech_fade:] *= audio.raised_cosine(speech_fade)[::-1, None]
    patch[: len(sound)] += sound
    if np.max(np.abs(patch), initial=0) >= 1:
        raise VoiceError(
            "Matching the voice would clip the audio. Record the line again.",
            code="voice_audio_clips",
        )
    meta = {
        "start_sample": first,
        "samples": count,
        "channels": channels,
        "fade_in_samples": min(FADE, count // 2),
        "fade_out_samples": min(FADE, count // 2),
        "stretch": stretch,
        "gain_db": gain,
    }
    after = blend(original, patch, meta, source_start_sample=0)
    ref_lu, _ = take_levels.measure(reference, work, "voice-reference")
    new_lu, _ = take_levels.measure(_speech(sound), work, "voice-new")
    difference = abs(
        (new_lu - ref_lu)
        if new_lu is not None and ref_lu is not None
        else level(_speech(sound)) - level(reference)
    )
    items = [
        edge_check(after, first, last),
        {
            "id": "splice.level",
            "status": "pass" if difference <= 1.5 else "warn",
            "difference_lu": difference,
            "reference_lufs": ref_lu,
            "reference_rms_db": level(reference),
            "rms_difference_db": abs(level(_speech(sound)) - level(reference)),
        },
        room_tone_check(patch, floor),
        {
            "id": "splice.pace",
            "status": "warn" if gap > 0.25 else "pass",
            "stretch": stretch,
            "gap_s": gap,
            "message": "A short pause remains." if gap > 0.25 else "The speech fits.",
        },
    ]
    return patch, meta, {"status": checks_status(items), "items": items}


def checks_status(items: list[dict]) -> str:
    states = {row["status"] for row in items}
    return "fail" if "fail" in states else "warn" if "warn" in states else "pass"
