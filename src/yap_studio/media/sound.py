"""Locate loud events and existing transcript phrases in catalog footage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from yap_studio import ffmpeg
from yap_studio.errors import InputError, ValidationError
from yap_studio.media import catalog
from yap_studio.media.sheet import _source
from yap_studio.transcript import Transcript

SAMPLE_RATE = 16_000
WINDOW_S = 0.1
ABOVE_FLOOR_DB = 12.0
MIN_LEVEL_DBFS = -40.0
MERGE_GAP_S = 0.3
INSERT_LEAD_S = 1.0
PHRASE_GAP_S = 0.6


def _levels(source: Path, *, duration_hint_s: float | None = None) -> tuple[np.ndarray, float]:
    command = [
        "ffmpeg", "-v", "error", "-i", str(source), "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
    ]
    window_samples = round(SAMPLE_RATE * WINDOW_S)
    window_bytes = window_samples * 4
    pending = bytearray()
    rms: list[float] = []
    sample_count = 0
    timeout_s = max(ffmpeg.DEFAULT_TIMEOUT_S, 120 + 1.5 * (duration_hint_s or 0))
    for chunk in ffmpeg.run_stream(command, timeout_s=timeout_s):
        pending.extend(chunk)
        full_bytes = len(pending) // window_bytes * window_bytes
        if full_bytes:
            windows = np.frombuffer(bytes(pending[:full_bytes]), dtype="<f4").reshape(
                -1, window_samples)
            rms.extend(np.sqrt(np.mean(np.square(windows), axis=1, dtype=np.float64)))
            sample_count += full_bytes // 4
            del pending[:full_bytes]
    if pending:
        tail = np.frombuffer(pending, dtype="<f4")
        rms.append(float(np.sqrt(np.mean(np.square(tail), dtype=np.float64))))
        sample_count += len(tail)
    levels = 20 * np.log10(np.maximum(np.asarray(rms), 1e-12))
    return levels, sample_count / SAMPLE_RATE


def _moments(levels: np.ndarray, duration_s: float, top: int) -> list[dict]:
    if not len(levels):
        return []
    floor = float(np.percentile(levels, 20))
    active = np.flatnonzero((levels >= floor + ABOVE_FLOOR_DB) & (levels > MIN_LEVEL_DBFS))
    runs: list[list[int]] = []
    merge_windows = round(MERGE_GAP_S / WINDOW_S)
    for index in active:
        point = int(index)
        if runs and point - runs[-1][1] < merge_windows:
            runs[-1][1] = point + 1
        else:
            runs.append([point, point + 1])
    ranked = []
    for start, end in runs:
        peak = start + int(np.argmax(levels[start:end]))
        peak_dbfs = float(levels[peak])
        peak_s = round((peak * WINDOW_S + min((peak + 1) * WINDOW_S, duration_s)) / 2, 2)
        ranked.append((peak_dbfs, {
            "start_s": round(start * WINDOW_S, 2),
            "peak_s": peak_s,
            "end_s": round(min(end * WINDOW_S, duration_s), 2),
            "peak_dbfs": round(peak_dbfs, 1),
            "above_floor_db": round(peak_dbfs - floor, 1),
            "in_s": round(max(0.0, peak_s - INSERT_LEAD_S), 2),
        }))
    selected = sorted(ranked, key=lambda pair: (-pair[0], pair[1]["start_s"]))[:top]
    return sorted((moment for _, moment in selected), key=lambda moment: moment["start_s"])


def _phrases(path: str) -> list[dict]:
    words = Transcript.load(path).words
    phrases: list[dict] = []
    for word in words:
        if not phrases or word.start - phrases[-1]["end_s"] > PHRASE_GAP_S:
            phrases.append({"start_s": word.start, "end_s": word.end, "text": word.text})
        else:
            phrases[-1]["end_s"] = word.end
            phrases[-1]["text"] += " " + word.text
    return phrases


def find_sound(item_id: str, *, top: int = 5) -> dict:
    """Return loud moments and saved speech phrases for one catalog clip."""
    if type(top) is not int or not 1 <= top <= 20:
        raise ValidationError("top", "must be an integer from 1 to 20")
    document = catalog.load_catalog()
    item = next((entry for entry in document["items"] if entry["id"] == item_id), None)
    if item is None:
        raise catalog.UnknownMediaError(item_id)
    source = _source(item, document)
    if item["kind"] == "image":
        raise InputError(str(source), "no sound: still image")
    if not item["has_audio"]:
        raise InputError(str(source), "no sound: clip has no audio track")
    levels, duration_s = _levels(source, duration_hint_s=item["duration_s"])
    result = {"id": item_id, "moments": _moments(levels, duration_s, top), "speech": None}
    speech = item["speech"]
    if speech is not None and speech["status"] == "words" and speech["path"]:
        try:
            result["speech"] = _phrases(speech["path"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise InputError(speech["path"], "cannot read saved speech transcript") from exc
    else:
        result["speech_note"] = f"Run yap-studio media sheet {item_id} first for speech."
    return result
