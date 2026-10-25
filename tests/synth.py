"""Synthetic media fixtures and independent calibration measurements."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

RATE = 48_000
_SEED = 20_260_928


@dataclass(frozen=True)
class WordSpan:
    """Approximate position of a spoken word."""

    text: str
    start: float
    end: float
    sentence: int


@dataclass(frozen=True)
class OracleEvent:
    """A measured flash and its nearest beep."""

    video_t: float
    audio_t: float
    offset_ms: float


@dataclass(frozen=True)
class Oracle:
    """Independent A/V measurements for a synthetic render."""

    events: list[OracleEvent]
    slope_ms_per_event: float
    segment_of_event: list[int]
    slope_ms_per_cut: float | None
    segment_offsets_ms: dict[int, float]
    cut_jumps_ms: dict[int, float]


def _run(cmd: list[str], timeout: int = 300) -> bytes:
    """Run ffmpeg with the fixture safety contract."""
    if not cmd or Path(cmd[0]).name != "ffmpeg":
        raise ValueError("_run only accepts ffmpeg commands")
    full = [cmd[0], "-nostdin", *cmd[1:]]
    try:
        result = subprocess.run(
            full,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg timed out after {timeout} seconds") from exc
    if result.returncode:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed ({result.returncode}):\n{stderr}")
    return result.stdout


def _process(cmd: list[str], timeout: int = 60) -> bytes:
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout} seconds: {cmd[0]}") from exc
    if result.returncode:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"{cmd[0]} failed ({result.returncode}):\n{stderr}")
    return result.stdout


def _probe(path: Path) -> dict[str, Any]:
    raw = _process(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(raw)


def _stream(path: Path, kind: str) -> dict[str, Any]:
    for stream in _probe(path)["streams"]:
        if stream.get("codec_type") == kind:
            return stream
    raise AssertionError(f"{path} has no {kind} stream")


def picture_source(path: Path, lavfi: str, frames: int, *, alpha: bool = False,
                   transfer: str = "bt709") -> Path:
    """Create lossless lavfi picture, including small synthetic alpha layer boxes."""
    _run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", lavfi,
          "-map", "0:v:0", "-an", "-frames:v", str(frames), "-threads", "2",
          *([] if alpha else ["-vf", "scale=in_color_matrix=bt601:out_color_matrix=bt709,"
                             f"setparams=color_primaries=bt709:color_trc={transfer}:"
                             "colorspace=bt709"]),
          "-c:v", "qtrle" if alpha else "libx264",
          *([] if alpha else ["-crf", "0", "-preset", "ultrafast"]),
          "-pix_fmt", "argb" if alpha else "yuv444p",
          "-color_primaries", "bt709", "-color_trc", transfer, "-colorspace", "bt709",
          str(path)])
    return path


def counter_source(path: Path, frames: int = 36) -> Path:
    """Stamp frame identity as binary boxes without fonts or optional text filters."""
    boxes = ["drawbox=x=40:y=40:w=128:h=80:color=black:t=fill"]
    for bit in range(frames.bit_length()):
        boxes.append(f"drawbox=x={40 + bit * 16}:y=40:w=12:h=80:color=white:t=fill:"
                     f"enable='bitand(n,{1 << bit})'")
    return picture_source(path, "testsrc2=s=1080x1920:r=30," + ",".join(boxes), frames)


def picture_pixels(path: Path, frame: int, *, box: tuple[int, int, int, int] | None = None
                   ) -> np.ndarray:
    """Decode one RGB frame or region for independent picture measurements."""
    stream = _stream(path, "video")
    width, height = int(stream["width"]), int(stream["height"])
    filters = [f"trim=start_frame={frame}:end_frame={frame + 1}"]
    if box:
        x, y, width, height = box
        filters.append(f"crop={width}:{height}:{x}:{y}:exact=1")
    data = _run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
                 "-vf", ",".join(filters), "-frames:v", "1", "-pix_fmt", "rgb24",
                 "-f", "rawvideo", "-"])
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def picture_hashes(path: Path, first: int = 0) -> list[str]:
    """Hash decoded picture frames, independently of packet timestamps and encoding."""
    data = _run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
                 "-vf", f"trim=start_frame={first}", "-f", "framehash", "-"])
    return [line.rsplit(",", 1)[1].strip() for line in data.decode().splitlines()
            if line and not line.startswith("#")]


def _fraction(value: str | float | int | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def _snap(value: str | float | int | Fraction, rate: Fraction) -> int:
    exact = _fraction(value) * rate
    return math.floor(exact + Fraction(1, 2))


def _even(value: float) -> int:
    return max(2, int(math.floor(value / 2.0 + 0.5)) * 2)


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    pcm = np.clip(samples, -0.999, 0.999)
    data = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sr)
        output.writeframes(data.tobytes())


def voiced_source(path: Path, runs: list[tuple[float, float]], *, seconds: float,
                  frequencies: list[float] | None = None) -> Path:
    """Mux lavfi picture with numpy tones marking speech and quiet runs."""
    sound = np.zeros(round(seconds * RATE), dtype=np.float32)
    tones = [220.0] * len(runs) if frequencies is None else frequencies
    for (first, last), frequency in zip(runs, tones, strict=True):
        start, end = round(first * RATE), round(last * RATE)
        sound[start:end] = 0.1 * np.sin(np.arange(end - start) * 2 * np.pi * frequency / RATE)
    with tempfile.TemporaryDirectory(prefix="yap-voiced-") as temporary:
        wav = Path(temporary) / "sound.wav"
        _write_wav(wav, sound, RATE)
        _run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
              f"testsrc2=s=160x90:r=30:d={seconds}", "-i", str(wav),
              "-map", "0:v:0", "-map", "1:a:0", "-shortest",
              "-c:v", "libx264", "-c:a", "pcm_s16le", str(path)])
    return path


def _pink_noise(count: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(count)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(count)
    scale = np.zeros_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * scale, n=count)
    pink -= pink.mean()
    return pink * (0.03 / np.sqrt(np.mean(pink * pink)))


def _calib_audio(seconds: float, fps: Fraction, sr: int, *, beeps: bool) -> np.ndarray:
    count = _snap(seconds, Fraction(sr, 1))
    rng = np.random.default_rng(_SEED)
    audio = _pink_noise(count, rng)
    time = np.arange(count, dtype=np.float64) / sr
    audio += 0.125 * np.sin(2.0 * np.pi * 220.0 * time)
    if beeps:
        beep_count = _snap(Fraction(40, 1000), Fraction(sr, 1))
        beep = 0.7 * np.sin(2.0 * np.pi * 1000.0 * np.arange(beep_count) / sr)
        for flash in flash_times(seconds, fps):
            start = _snap(flash, Fraction(sr, 1))
            stop = min(start + beep_count, count)
            audio[start:stop] += beep[: stop - start]
    return np.clip(audio, -0.999, 0.999)


def flash_times(seconds: float, fps: str | Fraction = "30/1") -> list[Fraction]:
    """Return source times of the single-frame white flashes."""
    rate = _fraction(fps)
    cycle = max(1, _snap(Fraction(2, 1), rate))
    phase = max(0, _snap(Fraction(1, 1), rate))
    frame_count = math.floor(_fraction(seconds) * rate)
    return [Fraction(frame, 1) / rate for frame in range(phase, frame_count, cycle)]


def calib_source(
    path: Path,
    seconds: float = 30,
    fps: str = "30/1",
    sr: int = RATE,
    size: str = "320x180",
) -> Path:
    """Create a unique-frame flash and beep calibration source."""
    path = Path(path)
    rate = _fraction(fps)
    cycle = max(1, _snap(Fraction(2, 1), rate))
    phase = max(0, _snap(Fraction(1, 1), rate))
    with tempfile.TemporaryDirectory(prefix="yap-synth-") as raw_dir:
        wav = Path(raw_dir) / "audio.wav"
        _write_wav(wav, _calib_audio(seconds, rate, sr, beeps=True), sr)
        source = f"testsrc2=size={size}:rate={fps}:duration={seconds}"
        vf = f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='eq(mod(n,{cycle}),{phase})'"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                source,
                "-i",
                str(wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-vf",
                vf,
                "-t",
                str(seconds),
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "pcm_s16le",
                str(path),
            ]
        )
    return path


def full_range_source(src: Path, out: Path, *, matrix: str = "bt470bg") -> Path:
    """Keep calibration audio and add dark, light and red full-range patches."""
    filters = ["format=rgb24"]
    for y, colour in ((0.1, "0x282828"), (0.4, "0xc8c8c8"), (0.7, "0xd22850")):
        filters.append(f"drawbox=x=0:y=ih*{y}:w=iw:h=ih*0.2:color={colour}:t=fill")
    filters.extend([f"scale=out_color_matrix={matrix}:out_range=pc", "format=yuvj420p"])
    _run(["ffmpeg", "-v", "error", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0",
          "-vf", ",".join(filters), "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
          "-pix_fmt", "yuvj420p", "-color_range", "pc", "-colorspace", matrix,
          "-color_primaries", "bt709", "-color_trc", "bt709", "-c:a", "copy", str(out)])
    return out


def patch_colours(path: Path) -> np.ndarray:
    """Measure the fixture's patch centres using the decoded range and matrix."""
    info = _stream(path, "video")
    width, height = info["width"], info["height"]
    data = _run([
        "ffmpeg", "-v", "error", "-i", str(path), "-an", "-vf",
        f"scale=in_color_matrix={info['color_space']}:in_range={info['color_range']},"
        "format=rgb24", "-frames:v", "1", "-f", "rawvideo", "-",
    ])
    rgb = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    return np.array([rgb[round(height * y), width // 2] for y in (0.2, 0.5, 0.8)])


def tone_speech_source(
    path: Path,
    words: list[str],
    pitches: dict[str, float],
    *,
    word_s: float = 0.3,
    pause_s: float = 0.1,
    lead_s: float = 0.6,
    tail_s: float = 0.6,
) -> tuple[Path, list[WordSpan]]:
    """Build harmonic word bursts over a low noise bed with a testsrc2 picture."""
    spans = [
        WordSpan(word, lead_s + i * (word_s + pause_s),
                 lead_s + i * (word_s + pause_s) + word_s, 0)
        for i, word in enumerate(words)
    ]
    seconds = math.ceil((spans[-1].end + tail_s) * 30) / 30
    rng = np.random.default_rng(_SEED)
    samples = rng.normal(0, 0.002, round(seconds * RATE))
    for span in spans:
        first, last = round(span.start * RATE), round(span.end * RATE)
        time = np.arange(last - first) / RATE
        phase = 2 * np.pi * pitches[span.text] * time
        tone = 0.18 * np.sin(phase) + 0.06 * np.sin(2 * phase) + 0.025 * np.sin(3 * phase)
        fade = round(0.005 * RATE)
        envelope = np.ones(len(tone))
        envelope[:fade] = np.linspace(0, 1, fade)
        envelope[-fade:] = np.linspace(1, 0, fade)
        samples[first:last] += tone * envelope
    with tempfile.TemporaryDirectory(prefix="yap-tone-speech-") as raw_dir:
        wav = Path(raw_dir) / "audio.wav"
        _write_wav(wav, samples, RATE)
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate=30:duration={seconds}",
            "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", str(path),
        ])
    return Path(path), spans


def vfr_source(path: Path, seconds: float = 30, size: str = "320x180") -> Path:
    """Create a nominal 30 fps take with drops, jitter, and a 60 ms audio offset."""
    path = Path(path)
    rate = Fraction(30)
    with tempfile.TemporaryDirectory(prefix="yap-synth-") as raw_dir:
        wav = Path(raw_dir) / "audio.wav"
        _write_wav(wav, _calib_audio(seconds, rate, RATE, beeps=True), RATE)
        # Early drops make the average-rate grid drift at late joins. The jitter
        # stays below half a frame and preserves the original timestamp order.
        vf = (
            "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='eq(mod(n,60),30)',"
            "select='not(eq(n,45)+eq(n,75)+eq(n,105)+eq(n,135))',"
            "settb=1/3000,setpts='PTS+9*sin(2*PI*T)'"
        )
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}",
            "-itsoffset", "0.06", "-i", str(wav),
            "-map", "0:v:0", "-map", "1:a:0", "-vf", vf,
            "-fps_mode", "passthrough", "-enc_time_base", "1:3000",
            "-video_track_timescale", "3000", "-t", str(seconds),
            "-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", str(path),
        ])
    return path


def slowpan_source(
    path: Path,
    seconds: float = 30,
    fps: str = "30/1",
    sr: int = RATE,
    size: str = "320x180",
    speed_px_s: float = 3.0,
) -> Path:
    """Create a slowly drifting, lightly noisy synthetic scene."""
    path = Path(path)
    width, height = (int(part) for part in size.split("x"))
    rate = _fraction(fps)
    frames = math.floor(_fraction(seconds) * rate)
    margin = math.ceil(seconds * abs(speed_px_s)) + 4
    rng = np.random.default_rng(_SEED + 1)
    base = rng.normal(127.0, 45.0, (height, width + margin))
    spectrum = np.fft.rfft2(base)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width + margin)[None, :]
    spectrum *= np.exp(-120.0 * (fx * fx + fy * fy))
    texture = np.fft.irfft2(spectrum, s=base.shape)
    texture = 45.0 * (texture - texture.mean()) / max(texture.std(), 1e-9) + 127.0
    with tempfile.TemporaryDirectory(prefix="yap-synth-") as raw_dir:
        raw = Path(raw_dir) / "video.gray"
        wav = Path(raw_dir) / "audio.wav"
        with raw.open("wb") as output:
            for frame in range(frames):
                offset = speed_px_s * frame / float(rate)
                left = max(0, min(int(math.floor(offset)), margin - 2))
                frac = offset - math.floor(offset)
                image = (1.0 - frac) * texture[:, left : left + width]
                image += frac * texture[:, left + 1 : left + width + 1]
                image += rng.normal(0.0, 2.0, image.shape)
                output.write(np.clip(image, 0, 255).astype(np.uint8).tobytes())
        _write_wav(wav, _calib_audio(seconds, rate, sr, beeps=False), sr)
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "gray",
                "-video_size",
                size,
                "-framerate",
                fps,
                "-i",
                str(raw),
                "-i",
                str(wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "pcm_s16le",
                "-t",
                str(seconds),
                str(path),
            ]
        )
    return path


def _read_wav_resampled(path: Path, sr: int) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        source_rate = source.getframerate()
        width = source.getsampwidth()
        raw = source.readframes(source.getnframes())
    if width != 2:
        raise AssertionError(f"unexpected espeak sample width: {width}")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, channels).mean(axis=1)
    audio /= 32768.0
    if source_rate == sr:
        return audio
    count = _snap(Fraction(len(audio), source_rate), Fraction(sr, 1))
    old_x = np.arange(len(audio), dtype=np.float64) / source_rate
    new_x = np.arange(count, dtype=np.float64) / sr
    return np.interp(new_x, old_x, audio)


def _voiced_bounds(audio: np.ndarray, sr: int) -> tuple[int, int]:
    window = max(1, int(0.01 * sr))
    usable = len(audio) - len(audio) % window
    if usable <= 0:
        return 0, len(audio)
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, window) ** 2, axis=1))
    voiced = np.flatnonzero(rms > 0.01)
    if not len(voiced):
        return 0, len(audio)
    return int(voiced[0] * window), min(len(audio), int((voiced[-1] + 1) * window))


def _voiced_runs(audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
    window = max(1, int(0.01 * sr))
    usable = len(audio) - len(audio) % window
    if usable <= 0:
        return []
    rms = np.sqrt(np.mean(audio[:usable].reshape(-1, window) ** 2, axis=1))
    voiced = rms > 0.01
    gap_frames = max(1, math.ceil(0.05 * sr / window))
    splits: list[tuple[int, int]] = []
    start = next((index for index, active in enumerate(voiced) if active), None)
    if start is None:
        return []
    last = start
    quiet = 0
    for index in range(start + 1, len(voiced)):
        if voiced[index]:
            if quiet >= gap_frames:
                splits.append((start * window, (last + 1) * window))
                start = index
            last = index
            quiet = 0
        else:
            quiet += 1
    splits.append((start * window, min(len(audio), (last + 1) * window)))
    return splits


def speech_source(
    path: Path,
    sentences: list[str],
    fps: str = "30/1",
    sr: int = RATE,
    *,
    picture: str = "testsrc2",
    size: str = "320x180",
    noise_dbfs: float | None = None,
    seed: int = 0,
) -> tuple[Path, list[WordSpan]]:
    """Create speech and word spans, optionally over a seeded pink-noise RMS bed."""
    if shutil.which("espeak-ng") is None:
        import pytest

        pytest.skip("espeak-ng not installed")
    path = Path(path)
    pieces = [np.zeros(_snap(Fraction(1, 2), Fraction(sr, 1)), dtype=np.float64)]
    cursor = len(pieces[0])
    spans: list[WordSpan] = []
    with tempfile.TemporaryDirectory(prefix="yap-speech-") as raw_dir:
        directory = Path(raw_dir)
        for sentence_index, sentence in enumerate(sentences):
            wav = directory / f"sentence-{sentence_index}.wav"
            _process(
                [
                    "espeak-ng",
                    "-v",
                    "en-us",
                    "-s",
                    "150",
                    "-g",
                    "10",
                    "-w",
                    str(wav),
                    sentence,
                ]
            )
            raw_audio = _read_wav_resampled(wav, sr)
            first, last = _voiced_bounds(raw_audio, sr)
            audio = raw_audio[first:last]
            words = sentence.split()
            runs = _voiced_runs(audio, sr)
            if len(runs) == len(words):
                locations = runs
            else:
                weights = np.array([max(1, len(word.strip(".,!?;:"))) for word in words])
                edges = np.concatenate(([0.0], np.cumsum(weights) / weights.sum()))
                locations = [
                    (int(edges[index] * len(audio)), int(edges[index + 1] * len(audio)))
                    for index in range(len(words))
                ]
            for word, (start, end) in zip(words, locations, strict=True):
                spans.append(
                    WordSpan(
                        text=word,
                        start=(cursor + start) / sr,
                        end=(cursor + end) / sr,
                        sentence=sentence_index,
                    )
                )
            pieces.append(audio)
            cursor += len(audio)
            if sentence_index != len(sentences) - 1:
                pause = np.zeros(_snap(Fraction(3, 5), Fraction(sr, 1)), dtype=np.float64)
                pieces.append(pause)
                cursor += len(pause)
        pieces.append(np.zeros(_snap(Fraction(1, 2), Fraction(sr, 1)), dtype=np.float64))
        joined = np.concatenate(pieces)
        if noise_dbfs is not None:
            noise = _pink_noise(len(joined), np.random.default_rng(seed))
            joined += noise * (10 ** (noise_dbfs / 20) / np.sqrt(np.mean(noise * noise)))
        combined = directory / "speech.wav"
        _write_wav(combined, joined, sr)
        duration = len(joined) / sr
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"{picture}=size={size}:rate={fps}:duration={duration:.9f}",
                "-i",
                str(combined),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-g",
                "30",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "pcm_s16le",
                "-t",
                f"{duration:.9f}",
                str(path),
            ]
        )
    return path, spans


def crop_window(src_w: int, src_h: int, x_frac: float) -> tuple[int, int, int, int]:
    """Return the even 9:16 crop window in source pixels."""
    if not 0.0 <= x_frac <= 1.0:
        raise ValueError("x_frac must be between 0 and 1")
    width = min(src_w, _even(src_h * 9.0 / 16.0))
    x = int(math.floor((src_w - width) * x_frac + 0.5))
    x = min(src_w - width, max(0, x - x % 2))
    return x, 0, width, src_h


def reframe(src: Path, out: Path, kind: str, x_frac: float = 0.5) -> Path:
    """Reframe media. A 320x180 crop becomes 180x320; a 180x320 fit becomes 568x320."""
    src = Path(src)
    out = Path(out)
    video = _stream(src, "video")
    width, height = int(video["width"]), int(video["height"])
    if kind == "crop":
        x, y, crop_w, crop_h = crop_window(width, height, x_frac)
        out_h = _even(max(320, crop_h))
        out_w = _even(out_h * 9.0 / 16.0)
        vf = f"crop={crop_w}:{crop_h}:{x}:{y},scale={out_w}:{out_h},setsar=1"
    elif kind == "fit":
        out_h = _even(max(320, height))
        out_w = _even(out_h * 16.0 / 9.0)
        vf = (
            "split=2[bg][fg];"
            f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},boxblur=20:2[blur];"
            f"[fg]scale={out_w}:{out_h}:force_original_aspect_ratio=decrease[front];"
            "[blur][front]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )
    else:
        raise ValueError(f"unknown reframe kind: {kind}")
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "passthrough",
            "-c:a",
            "copy",
            str(out),
        ]
    )
    return out


def display_rotation_supported() -> bool:
    """Return whether this ffmpeg supports the rotation metadata option."""
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-h", "full"]
    return b"display_rotation" in _process(command)


def rotate_metadata(src: Path, out: Path, degrees: int = 90) -> Path:
    """Store an upright vertical source as rotated landscape video."""
    if degrees not in {-90, 90}:
        raise ValueError("only 90 degree rotations are supported")
    if not display_rotation_supported():
        raise RuntimeError("ffmpeg does not support -display_rotation")
    transpose = "clock" if degrees == 90 else "cclock"
    with tempfile.TemporaryDirectory(prefix="yap-rotate-") as raw_dir:
        physical = Path(raw_dir) / "landscape.mov"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-vf",
                f"transpose={transpose}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                str(physical),
            ]
        )
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-display_rotation",
                str(degrees),
                "-i",
                str(physical),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c",
                "copy",
                str(out),
            ]
        )
    return Path(out)


def default_spans() -> list[tuple[float, float]]:
    """Return eight 30 fps source spans with safe edges."""
    frames = [
        (12, 84),
        (114, 186),
        (216, 288),
        (315, 387),
        (420, 492),
        (522, 594),
        (624, 696),
        (726, 798),
    ]
    return [(start / 30.0, end / 30.0) for start, end in frames]


def calibration_layout(
    *, late_phrase: str | None = None, late_frames: int = 0, card_over_caption: bool = False,
    card_over_face: bool = False, caption_over_face: bool = False,
) -> dict[str, Any]:
    """Describe glyphs and one card on the 320x180, 30 fps clean output grid."""
    phrases = [
        ("Make time count", [1.0, 1.5, 2.0]),
        ("Keep moving forward", [7.2, 7.6, 8.0]),
        ("Stay curious", [11.2, 11.6]),
        ("One small step today", [15.0, 15.4, 15.8, 16.2]),
    ]
    captions = []
    for slot, (text, starts) in enumerate(phrases, 1):
        identity = f"caption-{slot}"
        words = [
            {"text": word, "start_s": start, "end_s": round(start + 0.4, 6)}
            for word, start in zip(text.split(), starts, strict=True)
        ]
        shift = late_frames if identity == late_phrase else 0
        captions.append({
            "id": identity, "text": text,
            "start_frame": math.floor(round(words[0]["start_s"] * 30, 6)) + 1 + shift,
            "end_frame": math.ceil(round(words[-1]["end_s"] * 30, 6)) + shift,
            "boxes": [{"x": 30 + index * 65,
                       "y": 90 if caption_over_face and slot == 2 else 145, "w": 60, "h": 20}
                      for index in range(len(words))],
            "words": words,
        })
    return {
        "schema": "yap-studio/clearance-layout@1", "width": 320, "height": 180,
        "measured": True, "reason": None, "captions": captions,
        "elements": [{"id": "card-1", "kind": "card", "x": 25,
                      "y": 140 if card_over_caption else 75 if card_over_face else 10,
                      "w": 260, "h": 40 if card_over_face else 30,
                      "start_frame": 216, "end_frame": 270}],
    }


def calibration_faces(media: Path, *, over_face: bool = False):
    """Bind measured synthetic faces to a final, with an optional card-window face."""
    from yap_studio import ffmpeg
    from yap_studio.faces.detect import find_gaps
    from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
    from yap_studio.receipt import sha256_file

    info = ffmpeg.probe(media)
    frames = tuple(
        (FaceBox(Box(140, 90, 20, 15), 0.9, "detected"),)
        if over_face and 216 <= frame < 270 else ()
        for frame in range(info.video_frames)
    )
    return FacesData(
        "measured", None,
        VideoIdentity(media.name, sha256_file(media), info.video_frames, info.fps,
                      info.width, info.height),
        {"name": "fake"}, {}, frames, tuple(find_gaps(frames)),
    )


def _decode_audio(path: Path, sr: int = RATE, *, sync_timestamps: bool = False) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
    ]
    if sync_timestamps:
        # Make timestamps continuous from the stream's first PTS, preserving its start time.
        # Mid-stream packet overlaps are played instead of hard-trimmed, matching a player
        # that runs on the audio clock.
        audio_filter = "asetpts=N/SR/TB+STARTPTS,aresample=async=1:first_pts=0"
        cmd.extend(["-af", audio_filter])
    cmd.extend(["-ac", "1", "-ar", str(sr), "-f", "f32le", "-"])
    return np.frombuffer(_run(cmd), dtype="<f4").copy()


def _click_adjusted_spans(
    src: Path, spans: list[tuple[float, float]], cut: int, sr: int = RATE
) -> tuple[list[tuple[float, float]], float]:
    if not 1 <= cut < len(spans):
        raise ValueError("no_fade_cut must identify an existing 1-based cut")
    audio = _decode_audio(src, sr)
    left_end = _snap(spans[cut - 1][1], Fraction(sr, 1))
    right_start = _snap(spans[cut][0], Fraction(sr, 1))
    radius = _snap(Fraction(5, 1000), Fraction(sr, 1))
    left_indices = np.arange(left_end - radius - 1, left_end + radius)
    right_indices = np.arange(right_start - radius, right_start + radius + 1)
    left_indices = left_indices[(left_indices >= 0) & (left_indices < len(audio))]
    right_indices = right_indices[(right_indices >= 0) & (right_indices < len(audio))]
    high_left = float(np.max(audio[left_indices]))
    low_left = float(np.min(audio[left_indices]))
    high_right = float(np.max(audio[right_indices]))
    low_right = float(np.min(audio[right_indices]))
    candidates = [(abs(high_left - low_right), 0), (abs(low_left - high_right), 1)]
    _, direction = max(candidates)
    if direction == 0:
        left_index = int(left_indices[np.argmax(audio[left_indices])])
        right_index = int(right_indices[np.argmin(audio[right_indices])])
    else:
        left_index = int(left_indices[np.argmin(audio[left_indices])])
        right_index = int(right_indices[np.argmax(audio[right_indices])])
    step = abs(float(audio[left_index] - audio[right_index]))
    if step < 0.2:
        raise AssertionError(f"could not plant a 0.2 full-scale step: {step:.4f}")
    adjusted = list(spans)
    adjusted[cut - 1] = (adjusted[cut - 1][0], (left_index + 1) / sr)
    adjusted[cut] = (right_index / sr, adjusted[cut][1])
    return adjusted, step


def render_perclip_aac(
    src: Path,
    spans: list[tuple[float, float]],
    out: Path,
    fps: str = "30/1",
    no_fade_cut: int | None = None,
) -> Path:
    """Render the per-clip AAC pattern: AAC encoded per span, then joined with
    the concat demuxer and -c copy.

    Cut numbering is 1-based. Cut k joins spans k and k+1, so no_fade_cut=3 removes
    the fade-out of spans[2] and the fade-in of spans[3] in Python indexing.
    """
    actual = list(spans)
    if no_fade_cut is not None:
        actual, step = _click_adjusted_spans(Path(src), actual, no_fade_cut)
        if step < 0.2:
            raise AssertionError("planted click is below 0.2 full scale")
    with tempfile.TemporaryDirectory(prefix="yap-perclip-") as raw_dir:
        directory = Path(raw_dir)
        parts: list[Path] = []
        for index, (start, end) in enumerate(actual):
            duration = end - start
            fades: list[str] = []
            segment_number = index + 1
            if no_fade_cut != segment_number - 1:
                fades.append("afade=t=in:st=0:d=0.03")
            if no_fade_cut != segment_number:
                fades.append(f"afade=t=out:st={duration - 0.03:.9f}:d=0.03")
            part = directory / f"part-{index:02d}.mp4"
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.9f}",
                "-i",
                str(src),
                "-t",
                f"{duration:.9f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
            ]
            if fades:
                cmd.extend(["-af", ",".join(fades)])
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-r",
                    fps,
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-ar",
                    str(RATE),
                    str(part),
                ]
            )
            _run(cmd)
            parts.append(part)
        concat = directory / "parts.txt"
        concat.write_text("".join(f"file '{part}'\n" for part in parts))
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c",
                "copy",
                str(out),
            ]
        )
    return Path(out)


def _packet_count(path: Path) -> int:
    raw = _process(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
    )
    return int(raw.strip())


def _duration_samples(path: Path) -> int:
    raw = _process(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration_ts",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ]
    )
    return int(raw.strip())


def render_clean(
    src: Path,
    spans: list[tuple[float, float]],
    out: Path,
    fps: str = "30/1",
    fade_s: float = 0.03,
) -> Path:
    """Render frame-snapped spans and encode concatenated audio once."""
    rate = _fraction(fps)
    with tempfile.TemporaryDirectory(prefix="yap-clean-") as raw_dir:
        directory = Path(raw_dir)
        videos: list[Path] = []
        audios: list[Path] = []
        for index, (start, end) in enumerate(spans):
            first = _snap(start, rate)
            last = _snap(end, rate)
            frames = last - first
            samples = _snap(Fraction(frames, 1) * RATE / rate, Fraction(1, 1))
            if frames <= 0 or samples <= 0:
                raise AssertionError(f"empty clean segment {index + 1}")
            duration = samples / RATE
            source_time = Fraction(first, 1) / rate
            margin = min(Fraction(1, 1), source_time)
            seek = source_time - margin
            video = directory / f"video-{index:02d}.mp4"
            audio = directory / f"audio-{index:02d}.wav"
            trim_start = max(0.0, float(margin) - 0.5 / float(rate))
            vf = (
                f"trim=start={trim_start:.9f},"
                f"fps=fps={fps}:start_time={float(margin):.9f},setpts=PTS-STARTPTS"
            )
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{float(seek):.9f}",
                    "-i",
                    str(src),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    vf,
                    "-frames:v",
                    str(frames),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-bf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ]
            )
            af = (
                f"atrim=start={float(margin):.9f},asetpts=PTS-STARTPTS,"
                f"aresample={RATE},atrim=end_sample={samples}"
            )
            if fade_s > 0:
                fade = min(fade_s, duration / 2.0)
                af += (
                    f",afade=t=in:st=0:d={fade:.9f},"
                    f"afade=t=out:st={duration - fade:.9f}:d={fade:.9f}"
                )
            _run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{float(seek):.9f}",
                    "-i",
                    str(src),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-af",
                    af,
                    "-c:a",
                    "pcm_s16le",
                    str(audio),
                ]
            )
            got_frames = _packet_count(video)
            got_samples = _duration_samples(audio)
            if got_frames != frames or got_samples != samples:
                raise AssertionError(
                    f"segment {index + 1}: {got_frames}/{frames} frames, "
                    f"{got_samples}/{samples} samples"
                )
            videos.append(video)
            audios.append(audio)
        video_list = directory / "videos.txt"
        audio_list = directory / "audios.txt"
        video_list.write_text("".join(f"file '{path}'\n" for path in videos))
        audio_list.write_text("".join(f"file '{path}'\n" for path in audios))
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(video_list),
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(audio_list),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                str(RATE),
                "-movflags",
                "+faststart",
                str(out),
            ]
        )
    return Path(out)


def shift_audio(src: Path, out: Path, ms: int, method: str) -> Path:
    """Shift audio either in samples or by changing its stream start time."""
    if ms < 0:
        raise ValueError("negative shifts are not supported")
    if method == "samples":
        samples = len(_decode_audio(Path(src), RATE))
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "copy",
                "-af",
                f"adelay={ms}:all=1,atrim=end_sample={samples}",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                str(RATE),
                str(out),
            ]
        )
    elif method == "start_time":
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-itsoffset",
                f"{ms / 1000.0:.9f}",
                "-i",
                str(src),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                str(out),
            ]
        )
    else:
        raise ValueError(f"unknown shift method: {method}")
    return Path(out)


def audio_gap_render(
    path: Path,
    reference: Path,
    *,
    seconds: float = 4.0,
    gap_samples: int = 1600,
    at_sample: int = 96_000,
    sr: int = RATE,
    video_seconds: float | None = None,
) -> Path:
    """Encode pink noise as AAC in MP4 with one forward jump in its audio timestamps.

    Audio frames that start at or after ``at_sample`` get ``gap_samples`` added to
    their timestamps, as a muxer that writes one wrong packet duration does. The
    decoded content stays continuous. ``reference`` receives the same noise as
    16-bit PCM without the jump. ``gap_samples=0`` gives a clean AAC encode with
    its usual 1024-sample priming. The picture lasts ``video_seconds``, by default
    as long as the sound.
    """
    picture_seconds = seconds if video_seconds is None else video_seconds
    noise = f"anoisesrc=color=pink:seed=7:sample_rate={sr}:amplitude=0.25:duration={seconds}"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            noise,
            "-c:a",
            "pcm_s16le",
            str(reference),
        ]
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=160x90:rate=30:duration={picture_seconds}",
        "-f",
        "lavfi",
        "-i",
        noise,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    if gap_samples:
        command.extend(["-af", f"asetpts=PTS+{gap_samples}*gte(N\\,{at_sample})"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(path),
        ]
    )
    _run(command)
    return Path(path)


def hold_render(
    path: Path,
    *,
    moving_s: float = 2.0,
    hold_s: float = 1.5,
    picture: str = "held",
    sound: str = "silent",
    sr: int = RATE,
    keyframe: int | None = None,
    x264: Sequence[str] = (),
    change_frame: int | None = None,
    output_ts_offset: float | None = None,
) -> Path:
    """Render moving picture and pink noise, then a held end.

    ``picture="held"`` clones the last moving frame for ``hold_s``; ``"moving"``
    keeps the picture moving to the end. ``sound="silent"`` pads silence under the
    hold, ``"noise"`` keeps the noise going, and ``"short"`` stops the sound 0.2 s
    into the hold, after silence, so the file holds less audio than picture.
    ``"box"`` adds a white box, ``"tint"`` changes only its colour, ``"cut"`` switches
    stills, and ``"resumes"`` moves again at ``change_frame``.
    """
    total = moving_s + hold_s
    video_seconds = moving_s if picture in {"held", "box", "tint"} else total
    noise_seconds = total if sound == "noise" else moving_s
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=160x90:rate=30:duration={video_seconds}",
        "-f",
        "lavfi",
        "-i",
        f"anoisesrc=color=pink:seed=11:sample_rate={sr}:amplitude=0.25:duration={noise_seconds}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    if picture == "held":
        command.extend(["-vf", f"tpad=stop_mode=clone:stop_duration={hold_s}"])
    elif picture == "box":
        assert change_frame is not None
        command.extend([
            "-vf", f"tpad=stop_mode=clone:stop_duration={hold_s},"
            f"drawbox=x=iw*0.3:y=ih*0.4:w=iw*0.4:h=ih*0.2:color=white:t=fill:"
            f"enable='gte(n,{change_frame})'",
        ])
    elif picture == "tint":
        assert change_frame is not None
        command.extend([
            "-vf", f"tpad=stop_mode=clone:stop_duration={hold_s},split[base][patch];"
            "[patch]crop=w=iw*0.4:h=ih*0.2:x=iw*0.3:y=ih*0.4,"
            "lutyuv=u='clip(val+60,0,255)':v='clip(val-60,0,255)'[tint];"
            "[base][tint]overlay=x=main_w*0.3:y=main_h*0.4:"
            f"enable='gte(t,{change_frame / 30})'",
        ])
    elif picture in {"cut", "resumes"}:
        assert change_frame is not None
        moving_frames = round(moving_s * 30)
        tail = f"trim=start_frame={change_frame}"
        if picture == "cut":
            tail += f":end_frame={change_frame + 1}"
        tail += ",setpts=PTS-STARTPTS"
        if picture == "cut":
            tail += f",tpad=stop_mode=clone:stop_duration={(total * 30 - change_frame - 1) / 30}"
        command.extend([
            "-vf", "split=3[m][h][t];"
            f"[m]trim=end_frame={moving_frames},setpts=PTS-STARTPTS[moving];"
            f"[h]trim=start_frame={moving_frames - 1}:end_frame={moving_frames},"
            "setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop_duration={(change_frame - moving_frames - 1) / 30}[held];"
            f"[t]{tail}[tail];[moving][held][tail]concat=n=3:v=1:a=0",
        ])
    if sound == "silent":
        command.extend(["-af", f"apad=whole_dur={total}"])
    elif sound == "short":
        command.extend(["-af", f"apad=whole_dur={moving_s + 0.2}"])
    if keyframe is not None:
        command.extend(["-force_key_frames", f"expr:eq(n,{keyframe})"])
    if output_ts_offset is not None:
        command.extend(["-output_ts_offset", str(output_ts_offset)])
    command.extend(
        [
            "-c:v",
            "libx264",
            *x264,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(path),
        ]
    )
    _run(command)
    return Path(path)


def watch_render(path: Path) -> list[float]:
    """Build a vertical moving, frozen, and louder moving render; return its cuts.

    The frozen piece spans 3 to 7.5 s, with silence from 4 to 6 s. At 7.5 s,
    a different moving picture starts and the continuing pink noise rises 15 dB.
    """
    _run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=90x160:rate=30:duration=3",
            "-f", "lavfi", "-i", "testsrc2=size=90x160:rate=30:duration=0.1",
            "-f", "lavfi", "-i", "testsrc2=size=90x160:rate=30:duration=3",
            "-f", "lavfi", "-i",
            "anoisesrc=color=pink:seed=11:sample_rate=48000:amplitude=0.1:duration=10.5",
            "-filter_complex",
            "[1:v]trim=end_frame=1,setpts=PTS-STARTPTS,hue=h=90,"
            "tpad=stop_mode=clone:stop_duration=4.466666667[held];"
            "[2:v]hflip,negate[last];[0:v][held][last]concat=n=3:v=1:a=0[v];"
            "[3:a]volume=5.6234132519:enable='gte(t,7.5)',"
            "volume=0:enable='gte(t,4)*lt(t,6)'[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", str(path),
        ],
        timeout=60,
    )
    return [3.0, 7.5]


def watch_spans(path: Path, source: Path, cuts: list[float], end_s: float = 10.5) -> Path:
    """Write yap spans with source gaps and cuts at the requested output times."""
    edges = [0.0, *cuts, end_s]
    data = {"sources": {"a": str(source)}, "fps": "30/1", "segments": [
        {"source": "a", "start": start + index, "end": end + index}
        for index, (start, end) in enumerate(zip(edges, edges[1:], strict=False))
    ]}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def watch_words(path: Path, words: list[tuple[str, float, float]]) -> Path:
    """Write a canonical transcript with chosen video-clock word times."""
    data = {"schema": "yap-studio/transcript@1", "backend": "prepared", "words": [
        {"text": text, "start": start, "end": end, "conf": 1.0} for text, start, end in words
    ]}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def watch_audio_offset(path: Path, source: Path, offset_s: float = 1.0) -> Path:
    """Remux a render with its audio stream starting later than its picture."""
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-itsoffset", str(offset_s), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0",
        "-c", "copy", str(path),
    ], timeout=60)
    return path


def watch_four_second_freeze(path: Path) -> Path:
    """Render 2 s of motion, exactly 4 s of one picture, then 2 s of motion."""
    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=90x160:rate=30:duration=4.1",
            "-f", "lavfi", "-i",
            "anoisesrc=color=pink:seed=11:sample_rate=48000:amplitude=0.1:duration=8",
            "-filter_complex",
            "[0:v]split=3[head][picture][tail];"
            "[head]trim=end_frame=60,setpts=PTS-STARTPTS[first];"
            "[picture]trim=start_frame=60:end_frame=61,setpts=PTS-STARTPTS,"
            "tpad=stop_mode=clone:stop_duration=3.966666667[held];"
            "[tail]trim=start_frame=61:end_frame=121,setpts=PTS-STARTPTS[last];"
            "[first][held][last]concat=n=3:v=1:a=0[v]",
            "-map", "[v]", "-map", "1:a:0", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(path),
        ],
        timeout=60,
    )
    return path


def watch_ranges(
    path: Path,
    source: Path,
    cuts: list[float],
    *,
    end_s: float = 10.5,
    overlays: list[dict] | None = None,
    holds: list[tuple[int, int]] | None = None,
) -> Path:
    """Write source-gapped ranges with optional output-clock graphics and holds."""
    edges = [0.0, *cuts, end_s]
    data = {
        "sources": {"a": str(source)},
        "ranges": [
            {"source": "a", "start": start + index, "end": end + index}
            for index, (start, end) in enumerate(zip(edges, edges[1:], strict=False))
        ],
        "overlays": overlays or [],
        "holds": [
            {"start_frame": start, "end_frame": end, "audio": "silence"}
            for start, end in holds or []
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    """Read a mono or multichannel 16-bit WAV as mono float samples."""
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        rate = source.getframerate()
        raw = source.readframes(source.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, channels)
    return rate, samples.mean(axis=1) / 32768.0


def _least_squares(values: list[float], positions: list[float]) -> float:
    if len(values) < 2 or len(set(positions)) < 2:
        return 0.0
    return float(np.polyfit(np.asarray(positions), np.asarray(values), 1)[0])


def oracle(
    path: Path,
    spans: list[tuple[float, float]] | None = None,
    fps: str | Fraction | None = None,
) -> Oracle:
    """Measure flash/beep alignment using each decoded video's frame timestamp."""
    if fps is None:
        rate = _fraction(_stream(Path(path), "video")["avg_frame_rate"])
    else:
        rate = _fraction(fps)
    video_raw = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-vf",
            "scale=16:12,format=gray",
            "-f",
            "rawvideo",
            "-",
        ]
    )
    means = np.frombuffer(video_raw, dtype=np.uint8).reshape(-1, 16 * 12).mean(axis=1)
    frame_probe = json.loads(
        _process(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    frame_pts = [
        float(frame["best_effort_timestamp_time"]) for frame in frame_probe["frames"]
    ]
    if len(means) != len(frame_pts):
        raise AssertionError(
            f"decoded video has {len(means)} frames but ffprobe returned "
            f"{len(frame_pts)} frame timestamps"
        )
    white = np.flatnonzero(means > 200)
    flash_frames = white[np.insert(np.diff(white) > 1, 0, True)] if len(white) else white
    flashes = np.asarray([frame_pts[index] for index in flash_frames], dtype=np.float64)
    audio = np.abs(_decode_audio(Path(path), RATE, sync_timestamps=True))
    hot = np.flatnonzero(audio > 0.45)
    onsets = hot[np.insert(np.diff(hot) > int(0.1 * RATE), 0, True)] if len(hot) else hot
    beeps = np.asarray(onsets, dtype=np.float64) / RATE
    events: list[OracleEvent] = []
    for video_t in flashes:
        if not len(beeps):
            break
        audio_t = float(beeps[np.argmin(np.abs(beeps - video_t))])
        events.append(
            OracleEvent(
                video_t=float(video_t),
                audio_t=audio_t,
                offset_ms=(audio_t - float(video_t)) * 1000.0,
            )
        )
    offsets = [event.offset_ms for event in events]
    slope_event = _least_squares(offsets, list(range(len(events))))
    segment_of_event: list[int] = []
    segment_offsets: dict[int, float] = {}
    cut_jumps: dict[int, float] = {}
    slope_cut: float | None = None
    if spans is not None:
        boundaries: list[float] = []
        elapsed_frames = 0
        for start, end in spans:
            elapsed_frames += _snap(end, rate) - _snap(start, rate)
            boundaries.append(elapsed_frames / float(rate))
        for event in events:
            relative_video_t = event.video_t - frame_pts[0]
            segment = next(
                (
                    index + 1
                    for index, boundary in enumerate(boundaries)
                    if relative_video_t < boundary
                ),
                len(boundaries),
            )
            segment_of_event.append(segment)
        for segment in sorted(set(segment_of_event)):
            selected = [
                event.offset_ms
                for event, event_segment in zip(events, segment_of_event, strict=True)
                if event_segment == segment
            ]
            segment_offsets[segment] = float(np.mean(selected))
        for cut in range(1, len(spans)):
            if cut in segment_offsets and cut + 1 in segment_offsets:
                cut_jumps[cut] = segment_offsets[cut + 1] - segment_offsets[cut]
        slope_cut = _least_squares(offsets, [float(value) for value in segment_of_event])
    return Oracle(
        events=events,
        slope_ms_per_event=slope_event,
        segment_of_event=segment_of_event,
        slope_ms_per_cut=slope_cut,
        segment_offsets_ms=segment_offsets,
        cut_jumps_ms=cut_jumps,
    )


def float_audio(path: Path, samples: np.ndarray) -> Path:
    """Write exact numpy float samples as a 48 kHz PCM fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = samples.shape[1] if samples.ndim > 1 else 1
    with tempfile.TemporaryDirectory(prefix="yap-float-") as folder:
        raw = Path(folder) / "audio.f32le"
        raw.write_bytes(samples.astype("<f4").tobytes())
        _run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(RATE),
              "-ac", str(channels), "-i", str(raw), "-map", "0:a:0",
              "-c:a", "pcm_f32le", str(path)])
    return path


def loudness_master(path: Path, *, frames: int = 120, lufs: float = -16,
                    picture_filter: str | None = None) -> Path:
    """Make a measured numpy master, optionally muxed with synthetic lavfi picture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(frames * 1600) / RATE
    sound = (0.1 * np.sin(2 * np.pi * 220 * time) +
             0.02 * np.sin(2 * np.pi * 1031 * time)) * (0.7 + 0.3 * np.sin(2 * np.pi * time))
    with tempfile.TemporaryDirectory(prefix="yap-master-") as folder:
        voice = float_audio(Path(folder) / "source.wav", sound)
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "info", "-i", str(voice), "-map", "0:a:0",
             "-af", "loudnorm=print_format=json", "-f", "null", "-"],
            stdin=subprocess.DEVNULL, capture_output=True, timeout=60, check=True,
        )
        stats, _ = json.JSONDecoder().raw_decode(
            result.stderr.decode()[result.stderr.decode().rfind("{"):])
        normalised = float_audio(Path(folder) / "master.wav",
                                 sound * 10 ** ((lufs - float(stats["input_i"])) / 20))
        if path.suffix == ".wav":
            shutil.copyfile(normalised, path)
        else:
            picture = f"testsrc2=s=108x192:r=30:d={frames / 30}"
            if picture_filter:
                picture += f",{picture_filter}"
            _run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                  picture, "-i", str(normalised),
                  "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
                  "-preset", "ultrafast", "-c:a", "pcm_f32le", str(path)])
    return path


def replace_frame(src: Path, out: Path, frame: int, donor_frame: int) -> Path:
    """Replace one picture with another decoded picture, keeping sound and frame count."""
    if min(frame, donor_frame) < 0:
        raise ValueError("frame indices must be non-negative")
    graph = (
        "[0:v:0]split=3[before][donor][after];"
        f"[before]trim=end_frame={frame},setpts=PTS-STARTPTS[p];"
        f"[donor]trim=start_frame={donor_frame}:end_frame={donor_frame + 1},"
        "setpts=PTS-STARTPTS[x];"
        f"[after]trim=start_frame={frame + 1},setpts=PTS-STARTPTS[n];"
        "[p][x][n]concat=n=3:v=1:a=0[v]"
    )
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-filter_complex", graph, "-map", "[v]", "-map", "0:a:0",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", str(out),
    ])
    return Path(out)


def scene_change_source(src: Path, out: Path, change_s: float) -> Path:
    """Switch a synthetic source to a different moving pattern and keep its sound."""
    info = _stream(src, "video")
    seconds = float(info["duration"])
    if not 0 < change_s < seconds:
        raise ValueError("scene change must be inside the source")
    pattern = (
        f"testsrc=size={info['width']}x{info['height']}:"
        f"rate={info['r_frame_rate']}:duration={seconds},format=yuv420p"
    )
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-f", "lavfi", "-i", pattern, "-filter_complex",
        f"[0:v:0][1:v:0]overlay=shortest=1:enable='gte(t,{change_s})'[v]",
        "-map", "[v]", "-map", "0:a:0", "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy", str(out),
    ])
    return Path(out)


def drop_frame(src: Path, out: Path, frame: int) -> Path:
    """Remove one picture, close its timestamp gap, and copy the sound."""
    count = _packet_count(src)
    if not 0 <= frame < count:
        raise ValueError("frame must identify an existing picture")
    fps = _fraction(_stream(src, "video")["avg_frame_rate"])
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0", "-vf",
        f"select='not(eq(n,{frame}))',setpts=N/({fps}*TB)",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", str(out),
    ])
    assert _packet_count(out) == count - 1, "the missing picture was not removed"
    return Path(out)


def repeat_frame(src: Path, out: Path, frame: int) -> Path:
    """Insert a clone immediately after one picture and copy the sound."""
    count = _packet_count(src)
    if not 0 <= frame < count - 1:
        raise ValueError("frame must identify a picture before the last one")
    graph = (
        "[0:v:0]split=2[a][b];"
        f"[a]trim=end_frame={frame + 1},setpts=PTS-STARTPTS,"
        "tpad=stop=1:stop_mode=clone[left];"
        f"[b]trim=start_frame={frame + 1},setpts=PTS-STARTPTS[right];"
        "[left][right]concat=n=2:v=1:a=0[v]"
    )
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-filter_complex", graph, "-map", "[v]", "-map", "0:a:0",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy", str(out),
    ])
    assert _packet_count(out) == count + 1, "the repeated picture was not inserted"
    return Path(out)


def _calibration_pictures(src: Path, frames: list[int]) -> np.ndarray:
    """Decode selected small pictures for independent fixture assertions."""
    selected = "+".join(f"eq(n,{frame})" for frame in sorted(set(frames)))
    raw = _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
        "-map", "0:v:0", "-vf", f"select='{selected}',scale=64:36,format=gray",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-",
    ])
    pictures = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 36, 64).astype(np.float32)
    assert len(pictures) == len(set(frames)), "a selected picture was not decoded"
    return pictures


def plant_flash(src: Path, out: Path, frame: int, donor_frame: int) -> Path:
    """Replace a picture with a foreign picture and verify the encoded replacement."""
    assert frame != donor_frame, "a flash needs a different donor picture"
    replace_frame(src, out, frame, donor_frame)
    assert _packet_count(out) == _packet_count(src), "a flash must keep the frame count"
    originals = _calibration_pictures(src, [frame, donor_frame])
    original, donor = originals if frame < donor_frame else originals[::-1]
    planted = _calibration_pictures(out, [frame])[0]
    error = float(np.mean(abs(planted - donor)))
    distance = float(np.mean(abs(original - donor)))
    assert distance > 10, "the donor picture is not visibly foreign"
    assert error < 2 and error < distance / 5, "the foreign picture was not planted"
    return Path(out)


def _calibration_audio(src: Path) -> np.ndarray:
    """Read nominal 48 kHz content, excluding any decoded encoder tail."""
    assert int(_stream(src, "audio")["sample_rate"]) == RATE
    count = _duration_samples(src)
    samples = _decode_audio(src)
    assert len(samples) >= count, "the sound is shorter than its declared duration"
    return samples[:count]


def _mux_calibration_audio(src: Path, out: Path, samples: np.ndarray) -> Path:
    """Encode changed sound once at 192k and copy the picture stream."""
    assert len(samples) == _duration_samples(src), "the sound length must stay unchanged"
    with tempfile.TemporaryDirectory(prefix="yap-calibration-") as raw_dir:
        wav = Path(raw_dir) / "audio.wav"
        _write_wav(wav, samples, RATE)
        _run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
            "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(RATE), str(out),
        ])
    assert _duration_samples(out) == len(samples), "the encoded sound length changed"
    return Path(out)


def mute_audio(src: Path, out: Path, start_s: float, duration_s: float) -> Path:
    """Mute a sound window with 10 ms ramps so its edges do not click."""
    samples = _calibration_audio(src)
    first = _snap(start_s, Fraction(RATE))
    stop = first + _snap(duration_s, Fraction(RATE))
    fade = _snap(0.01, Fraction(RATE))
    if not 0 <= first < first + 2 * fade < stop <= len(samples):
        raise ValueError("mute window must fit the sound and both ramps")
    assert np.sqrt(np.mean(samples[first:stop] ** 2)) > 0.02, "the window is already quiet"
    samples[first:first + fade] *= np.linspace(1, 0, fade)
    samples[first + fade:stop - fade] = 0
    samples[stop - fade:stop] *= np.linspace(0, 1, fade)
    _mux_calibration_audio(src, out, samples)
    decoded = _decode_audio(out)
    assert np.max(abs(decoded[first + 2 * fade:stop - 2 * fade])) < 1e-4, (
        "the encoded mute window is not silent"
    )
    return Path(out)


def plant_click(src: Path, out: Path, at_s: float) -> Path:
    """Add a sharp 0.6 full-scale step at the join with a 1 ms decay."""
    samples = _calibration_audio(src)
    at = _snap(at_s, Fraction(RATE))
    decay = _snap(0.001, Fraction(RATE))
    if not 1 <= at < at + decay <= len(samples):
        raise ValueError("click and decay must fit inside the sound")
    samples[at:at + decay] += 0.6 * np.linspace(1, 0, decay)
    assert abs(float(samples[at] - samples[at - 1])) > 0.5, "the step was not planted"
    assert np.max(abs(samples)) < 0.999, "the click must not clip"
    _mux_calibration_audio(src, out, samples)
    decoded = _decode_audio(out)
    radius = _snap(0.002, Fraction(RATE))
    assert np.max(abs(np.diff(decoded[at - radius:at + radius]))) > 0.1, (
        "the encoded step is too small"
    )
    return Path(out)


def shift_audio_early(src: Path, out: Path, samples_early: int) -> Path:
    """Advance the whole sound by an exact sample count and pad the tail with silence."""
    original = _calibration_audio(src)
    if not 0 < samples_early < len(original):
        raise ValueError("the advance must be shorter than the sound")
    shifted = np.zeros_like(original)
    shifted[:-samples_early] = original[samples_early:]
    _mux_calibration_audio(src, out, shifted)
    decoded = _decode_audio(out)[:len(original)]
    # Compare a steady interior window against the requested sample advance.
    first, stop = RATE, 2 * RATE
    expected = original[first + samples_early:stop + samples_early]
    assert np.corrcoef(decoded[first:stop], expected)[0, 1] > 0.98, (
        "the encoded sound does not match the requested advance"
    )
    assert np.max(abs(decoded[-samples_early // 2:])) < 1e-4, (
        "the advanced sound's padded tail is not silent"
    )
    return Path(out)


def watch_short_audio(path: Path) -> Path:
    """Render 10 s of moving picture with a pink noise stream ending at 7 s."""
    _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=90x160:rate=30:duration=10",
            "-f", "lavfi", "-i",
            "anoisesrc=color=pink:seed=11:sample_rate=48000:amplitude=0.1:duration=7",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(path),
        ],
        timeout=60,
    )
    return path


def cover_assets(folder: Path):
    """Make a patterned portrait and synthetic person matte for cover renders."""
    from yap_studio import ffmpeg
    from yap_studio.covers.render import CoverAssets

    frame = folder / "frame.png"
    matte = folder / "matte.png"
    cutout = folder / "cutout.png"
    blur = folder / "blur.png"
    dark = folder / "dark.png"
    ffmpeg.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                "testsrc2=size=1080x1920:rate=30", "-frames:v", "1", str(frame)],
               timeout_s=120)
    ffmpeg.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                "color=c=black:s=1080x1920:r=30", "-vf",
                "drawbox=x=450:y=350:w=600:h=1400:color=white:t=fill,format=gray",
                "-frames:v", "1", str(matte)], timeout_s=120)
    ffmpeg.run(["ffmpeg", "-y", "-i", str(frame), "-i", str(matte),
                "-filter_complex", "[0:v]format=rgb24[p];[1:v]format=gray[a];"
                "[p][a]alphamerge,format=rgba[out]", "-map", "[out]",
                "-frames:v", "1", str(cutout)], timeout_s=120)
    ffmpeg.run(["ffmpeg", "-y", "-i", str(frame), "-vf", "gblur=sigma=30",
                "-frames:v", "1", str(blur)], timeout_s=120)
    ffmpeg.run(["ffmpeg", "-y", "-i", str(frame), "-vf",
                "format=rgb24,colorchannelmixer=rr=0.75:gg=0.75:bb=0.75",
                "-frames:v", "1", str(dark)], timeout_s=120)
    return CoverAssets(frame, cutout, blur, dark, matte, face_box=(450, 350, 500, 700))
