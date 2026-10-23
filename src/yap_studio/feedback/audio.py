"""Measure render audio levels on output time for speech keep delivery checks."""

from __future__ import annotations

import math
import wave
from contextlib import closing
from pathlib import Path

import numpy as np

from yap_studio import ffmpeg
from yap_studio.errors import InputError, ProcessingError, ValidationError

ENVELOPE_SCHEMA = "yap-studio/voice-envelope@1"
ENVELOPE_RATE = 100


def _rms_db(rms: float) -> float:
    return round(max(-80.0, 20 * math.log10(rms)) if rms else -80.0, 1)


def _rms_envelope(mono: np.ndarray, rate: int) -> dict:
    hop = rate // ENVELOPE_RATE
    levels = []
    for start in range(0, len(mono), hop):
        rms = float(np.sqrt(np.mean(mono[start:start + hop] ** 2)))
        levels.append(_rms_db(rms))
    return {"schema": ENVELOPE_SCHEMA, "rate": ENVELOPE_RATE, "db": levels}


class _PCMEnvelope:
    """Accumulate mono 48 kHz PCM with less than one window of retained bytes."""

    def __init__(self) -> None:
        self.levels: list[float] = []
        self._remainder = b""

    def update(self, chunk: bytes) -> None:
        data = self._remainder + chunk
        complete = len(data) // 960 * 960
        self._remainder = data[complete:]
        if complete:
            mono = np.frombuffer(data, dtype="<i2", count=complete // 2).astype(np.float64)
            mono = (mono / 32768).reshape(-1, 480)
            rms = np.sqrt(np.mean(mono ** 2, axis=1))
            self.levels.extend(_rms_db(float(value)) for value in rms)

    def finish(self) -> dict:
        if self._remainder:
            mono = np.frombuffer(self._remainder, dtype="<i2").astype(np.float64) / 32768
            self.levels.extend(_rms_envelope(mono, 48000)["db"])
            self._remainder = b""
        return {"schema": ENVELOPE_SCHEMA, "rate": ENVELOPE_RATE, "db": self.levels}


def voice_envelope(wav_path: Path) -> dict:
    """Return rounded RMS dB levels of the mono mix in 10 ms windows."""
    try:
        with wave.open(str(wav_path), "rb") as wav:
            if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise InputError(str(wav_path), "voice must be 16-bit PCM WAV")
            rate, channels = wav.getframerate(), wav.getnchannels()
            if rate % ENVELOPE_RATE or channels not in (1, 2):
                raise InputError(str(wav_path), "unsupported voice sample rate or channels")
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        mono = samples.astype(np.float64).reshape(-1, channels).mean(axis=1) / 32768
        return _rms_envelope(mono, rate)
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        raise InputError(str(wav_path), "cannot read 16-bit PCM WAV") from exc


def media_envelope(media_path: Path, *, chunk_size: int = 64 * 1024) -> dict | None:
    """Decode the render's first audio stream to a mono 48 kHz delivery envelope."""
    try:
        info = ffmpeg.probe(media_path, count_frames=False)
        if not info.has_audio:
            return None
        offset = info.audio_offset_s - info.video_offset_s if info.has_video else 0.0
        accumulator = _PCMEnvelope()
        accumulator.levels.extend([-80.0] * max(0, round(offset * ENVELOPE_RATE)))
        skip_bytes = max(0, round(-offset * 48000)) * 2
        # Allow five seconds per MiB for larger renders, with a one minute minimum.
        timeout_s = max(60.0, media_path.stat().st_size / (1024 * 1024) * 5)
        # Reset only the first timestamp; preserve later gaps for the resampler to fill.
        with closing(ffmpeg.run_stream([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(media_path), "-map", "0:a:0",
            "-af", "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
            "-ac", "1", "-ar", "48000", "-f", "s16le", "-",
        ], timeout_s=timeout_s, chunk_size=chunk_size)) as chunks:
            for chunk in chunks:
                skipped = min(skip_bytes, len(chunk))
                skip_bytes -= skipped
                accumulator.update(chunk[skipped:])
            return accumulator.finish()
    except ProcessingError as exc:
        stderr = exc.full_stderr.lower()
        if "stream map '" in stderr and "matches no streams" in stderr:
            return None
        raise InputError(str(media_path), "cannot decode render audio") from exc
    except (OSError, ValueError) as exc:
        raise InputError(str(media_path), "cannot decode render audio") from exc


def parse_envelope(data: object) -> dict:
    """Validate the exact schema, fixed rate, and finite full-scale dB values."""
    if (not isinstance(data, dict) or set(data) != {"schema", "rate", "db"}
            or data["schema"] != ENVELOPE_SCHEMA or type(data["rate"]) is not int
            or data["rate"] != ENVELOPE_RATE or not isinstance(data["db"], list)):
        raise ValidationError("voice envelope", "expected a 100 Hz voice-envelope@1 object")
    for value in data["db"]:
        if type(value) not in (int, float) or not -80 <= value <= 0:
            raise ValidationError("voice envelope.db", "levels must be finite and from -80 to 0 dB")
    return {"schema": ENVELOPE_SCHEMA, "rate": ENVELOPE_RATE,
            "db": [float(value) for value in data["db"]]}


def envelope_slice(envelope: dict, start_s: float, end_s: float) -> list[float]:
    """Return windows overlapping the half-open output span, clipped to available audio."""
    start = max(0, math.floor(round(start_s * ENVELOPE_RATE, 9)))
    end = max(start, math.ceil(round(end_s * ENVELOPE_RATE, 9)))
    return envelope["db"][start:end]


def compare_delivery(kept: list[float], new: list[float]) -> tuple[str, str]:
    """Compare speech length, gain and contour, ignoring jointly quiet windows."""
    difference = len(new) - len(kept)
    if abs(difference) > 2:
        longer = new if difference > 0 else kept
        tail = longer[min(len(kept), len(new)):]
        if sum(level > -50 for level in tail) > 2:
            direction = "longer" if difference > 0 else "shorter"
            return ("broken", f"the kept speech is {abs(difference) / ENVELOPE_RATE:.2f} s "
                    f"{direction} in the new version")
    pairs = [(a, b) for a, b in zip(kept, new, strict=False) if max(a, b) > -50]
    if len(pairs) < 5:
        return "unmeasured", "too little speech to measure delivery"
    differences = np.array([b - a for a, b in pairs])
    gain = float(np.median(differences))
    residual = float(np.mean(np.abs(differences - gain)))
    if abs(gain) > 3.0:
        return "broken", f"speech is {abs(gain):.1f} dB {'louder' if gain > 0 else 'quieter'}"
    if residual > 2.0:
        return "broken", f"delivery changed: level contour differs by {residual:.1f} dB on average"
    return "kept", "words, pauses and delivery kept"
