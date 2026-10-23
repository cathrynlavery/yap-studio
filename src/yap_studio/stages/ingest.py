"""Verified source analysis, container-clock words, and recording preflight."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg, paths, receipt, transcript
from yap_studio.checks.structure import _loudness
from yap_studio.errors import DependencyError, InputError, ProcessingError, YapError
from yap_studio.project.lock import ConflictError, project_lock
from yap_studio.project.model import Word, Words, load_cut_words, load_words, parse_document
from yap_studio.project.receipts import StageRun, audio_stream_sha256, read_receipt
from yap_studio.project.status import ProjectContext, ingest_inputs
from yap_studio.stages import coverage, edges
from yap_studio.transcript import shared

SAMPLE_RATE = 16_000
MUSIC_THRESHOLDS = {
    "sample_rate": SAMPLE_RATE,
    "frame_samples": 512,
    "hop_s": 0.01,
    "window_s": 5.0,
    "band_hz": [100, 5000],
    "gap_db": 20,
    "gap20_max": 0.05,
    "quiet_flatness_max": 0.03,
    "floor10_min_dbfs": -45,
    "speech_words_per_s": 0.5,
}
_REFERENCE_FILES = ("edit.json", "timeline/agent.json", "timeline/broll.json")


class IngestRefusal(YapError):
    """Refuse replacement of words that an edit or timeline references."""

    def __init__(self, files: list[str]) -> None:
        super().__init__(
            "Replacement words are referenced by " + ", ".join(files) + ".",
            error_type="ingest_refusal",
            code="words_referenced",
            suggested_action="Run ingest with --force, then yap-studio edit rebase DIR.",
        )


@dataclass(frozen=True)
class _Decoded:
    """One selected stream and its shared analysis and speech input."""

    source: Path
    wav: Path
    info: ffmpeg.MediaInfo
    stream: int
    offset_s: float
    avg_frame_rate: str | None
    r_frame_rate: str | None
    loudness: dict[str, float | None]


def _rounded(value: float) -> float:
    """Round source timings consistently with the other pipeline stages."""
    return math.floor(value * 1000 + 0.5) / 1000


def _timeout(duration_s: float | None) -> float:
    """Allow longer media proportionally more processing time."""
    return max(ffmpeg.DEFAULT_TIMEOUT_S, 120 + 1.5 * (duration_s or 0))


def _decode(source: Path, wav: Path, audio_stream: int | None = None) -> _Decoded:
    """Choose a decodable stream and decode it once for both consumers."""
    source = paths.validate_input(source)
    info = ffmpeg.probe(source, count_frames=False)
    if not info.has_audio:
        raise InputError(str(source), "the take has no audio; record with sound")
    raw = ffmpeg.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(source)],
        timeout_s=ffmpeg.PROBE_TIMEOUT_S,
    )
    streams = json.loads(raw.stdout)["streams"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    candidates = list(range(len(audio_streams))) if audio_stream is None else [audio_stream]
    chosen = None
    for index in candidates:
        if index >= len(audio_streams):
            raise InputError(str(source), f"audio stream {index} is missing; choose another stream")
        try:
            ffmpeg.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(source),
                    "-map",
                    f"0:a:{index}",
                    "-t",
                    "1",
                    "-f",
                    "null",
                    "-",
                ],
                timeout_s=_timeout(1),
            )
        except (ProcessingError, InputError):
            if audio_stream is not None:
                raise InputError(
                    str(source), f"audio stream {index} cannot decode; choose another stream"
                ) from None
            continue
        chosen = index
        break
    if chosen is None:
        raise InputError(str(source), "no audio stream can decode; re-export the sound and retry")
    selected = audio_streams[chosen]
    offset = (
        info.audio_offset_s
        if chosen == 0
        else round(
            float(selected.get("start_time") or 0) - info.format_start_s,
            6,
        )
    )
    duration = max(
        info.duration_s or 0, info.audio_duration_s or 0, float(selected.get("duration") or 0)
    )
    with paths.atomic_output(wav) as temporary:
        ffmpeg.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                f"0:a:{chosen}",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_f32le",
                "-f",
                "wav",
                str(temporary),
            ],
            timeout_s=_timeout(duration),
        )
    loud = ffmpeg.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(source),
            "-map",
            f"0:a:{chosen}",
            "-af",
            "ebur128=peak=true:framelog=verbose",
            "-f",
            "null",
            "-",
        ],
        timeout_s=_timeout(duration),
    )
    integrated, _, peak = _loudness(loud.stderr)
    video = next((stream for stream in streams if stream.get("index") == info.video_index), {})
    return _Decoded(
        source,
        wav,
        info,
        chosen,
        offset,
        video.get("avg_frame_rate"),
        video.get("r_frame_rate"),
        {"integrated_lufs": integrated, "true_peak_dbtp": peak},
    )


def _float_wav(wav: Path) -> np.memmap:
    """Map the float WAV's data chunk without loading a long take into memory."""
    with wav.open("rb") as source:
        header = source.read(12)
        if header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise InputError(wav.name, "expected a float WAV; decode the take again")
        valid_format = False
        while chunk := source.read(8):
            if len(chunk) != 8:
                break
            kind, size = struct.unpack("<4sI", chunk)
            if kind == b"fmt ":
                fmt = source.read(size)
                encoding, channels, rate = struct.unpack_from("<HHI", fmt)
                bits = struct.unpack_from("<H", fmt, 14)[0]
                if encoding == 65534 and len(fmt) >= 40:
                    encoding = struct.unpack_from("<H", fmt, 24)[0]
                valid_format = (
                    encoding == 3 and channels == 1 and rate == SAMPLE_RATE and bits == 32
                )
                source.seek(size % 2, 1)
            elif kind == b"data":
                if valid_format and size >= 4:
                    return np.memmap(
                        wav, dtype="<f4", mode="r", offset=source.tell(), shape=size // 4
                    )
                break
            else:
                source.seek(size + size % 2, 1)
    raise InputError(wav.name, "empty or invalid analysis WAV; record with sound and retry")


def _speech_clip(samples: np.ndarray, window: tuple[float, float], target: Path) -> None:
    """Write an exact mono float WAV slice for the speech engine."""
    first, last = (round(time * SAMPLE_RATE) for time in window)
    sound = np.asarray(samples[first:last], dtype="<f4")
    size = sound.nbytes
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + size, b"WAVE", b"fmt ", 16,
        3, 1, SAMPLE_RATE, SAMPLE_RATE * 4, 4, 32, b"data", size,
    )
    with target.open("wb") as output:
        output.write(header)
        output.write(sound.tobytes())


def _music(wav: Path, words: tuple[Word, ...], offset_s: float) -> dict[str, Any]:
    """Measure continuous tonal sound in bounded, non-overlapping windows."""
    samples = _float_wav(wav)
    frame_size = MUSIC_THRESHOLDS["frame_samples"]
    hop = round(MUSIC_THRESHOLDS["hop_s"] * SAMPLE_RATE)
    window_size = round(MUSIC_THRESHOLDS["window_s"] * SAMPLE_RATE)
    hann = np.hanning(frame_size)
    frequencies = np.fft.rfftfreq(frame_size, 1 / SAMPLE_RATE)
    band = (frequencies >= 100) & (frequencies <= 5000)
    windows: list[dict[str, Any]] = []
    ranges: list[dict[str, Any]] = []
    for start in range(0, len(samples), window_size):
        end = min(start + window_size, len(samples))
        # A frame belongs to the window containing its start; keep its full tail.
        block = np.asarray(samples[start : min(end + frame_size - hop, len(samples))])
        if len(block) < frame_size:
            continue
        frames = np.lib.stride_tricks.sliding_window_view(block, frame_size)[::hop]
        power = np.abs(np.fft.rfft(frames * hann, axis=1))[:, band] ** 2
        power = np.maximum(power, 1e-24)
        flatness = np.exp(np.mean(np.log(power), axis=1)) / np.mean(power, axis=1)
        levels = 10 * np.log10(np.maximum(np.mean(frames.astype(np.float64) ** 2, axis=1), 1e-24))
        p95 = float(np.percentile(levels, 95))
        gap20 = float(np.mean(levels < p95 - MUSIC_THRESHOLDS["gap_db"]))
        floor10 = float(np.percentile(levels, 10))
        quiet = np.argsort(levels)[: max(1, math.ceil(len(levels) * 0.2))]
        quiet_flatness = float(np.median(flatness[quiet]))
        tonal = (
            gap20 < MUSIC_THRESHOLDS["gap20_max"]
            and quiet_flatness < MUSIC_THRESHOLDS["quiet_flatness_max"]
            and floor10 > MUSIC_THRESHOLDS["floor10_min_dbfs"]
        )
        t0, t1 = start / SAMPLE_RATE + offset_s, end / SAMPLE_RATE + offset_s
        windows.append(
            {
                "start": _rounded(t0),
                "end": _rounded(t1),
                "p95_dbfs": p95,
                "gap20": gap20,
                "floor10_dbfs": floor10,
                "quiet_flatness": quiet_flatness,
                "music": tonal,
            }
        )
        if not tonal:
            continue
        density = sum(t0 <= word.start < t1 for word in words) / (t1 - t0)
        kind = (
            "music_under_speech"
            if density >= MUSIC_THRESHOLDS["speech_words_per_s"]
            else "music_only"
        )
        if ranges and ranges[-1]["kind"] == kind and ranges[-1]["end"] == _rounded(t0):
            ranges[-1]["end"] = _rounded(t1)
        else:
            ranges.append({"start": _rounded(t0), "end": _rounded(t1), "kind": kind})
    return {"ranges": ranges, "windows": windows, "thresholds": dict(MUSIC_THRESHOLDS)}


def _source_words(
    take: str, result: transcript.Transcript, offset_s: float, duration_s: float | None,
) -> tuple[Word, ...]:
    """Give speech words stable ids on the container clock, bounded by the media end."""
    limit = duration_s if duration_s is not None else math.inf
    return tuple(
        Word(
            f"{take}:{index}",
            word.text,
            _rounded(min(word.start + offset_s, limit)),
            _rounded(min(word.end + offset_s, limit)),
            word.conf,
        )
        for index, word in enumerate(result.words)
    )


def _rate(value: str | None) -> Fraction | None:
    """Parse a usable stream rate, ignoring unavailable values."""
    try:
        rate = Fraction(value or "0")
        return rate if rate > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def _facts(decoded: _Decoded, words: tuple[Word, ...]) -> dict[str, Any]:
    """Collect colour, frame-rate, level, and music facts with plain warnings."""
    info = decoded.info
    hdr = {"arib-std-b67": "hlg", "smpte2084": "pq"}.get(info.color_transfer or "")
    avg, nominal = _rate(decoded.avg_frame_rate), _rate(decoded.r_frame_rate)
    vfr = ffmpeg.variable_frame_rate(avg, nominal)
    music = _music(decoded.wav, words, decoded.offset_s)
    warnings = []
    level = decoded.loudness["integrated_lufs"]
    if level is None or not -28 <= level <= -16:
        warnings.append("voice loudness is outside -28 to -16 LUFS; adjust the mic gain")
    kinds = {item["kind"] for item in music["ranges"]}
    if "music_under_speech" in kinds:
        warnings.append("music under the voice; record without the tonal sound")
    if "music_only" in kinds:
        warnings.append("music only; record speech without the tonal sound")
    if vfr:
        warnings.append("VFR source; frame rate will convert for the edit")
    if hdr:
        warnings.append(f"HDR source ({hdr.upper()}); colour needs conversion for the edit")
    if decoded.stream != 0:
        warnings.append(f"non-default audio stream {decoded.stream} selected; check the sound")
    if info.fps in {Fraction(24), Fraction(25)}:
        warnings.append(f"{info.fps} fps source; frame rate will convert for the edit")
    probe = info.to_dict()
    probe.pop("path")
    probe.update(
        audio_offset_s=decoded.offset_s,
        avg_frame_rate=decoded.avg_frame_rate,
        r_frame_rate=decoded.r_frame_rate,
    )
    return {
        "probe": probe,
        "hdr": hdr,
        "vfr": vfr,
        "audio_stream": decoded.stream,
        "audio_sha256": audio_stream_sha256(decoded.source, decoded.stream),
        "audio_offset_s": decoded.offset_s,
        "loudness": decoded.loudness,
        "music_ranges": music["ranges"],
        "music_windows": music["windows"],
        "music_thresholds": music["thresholds"],
        "warnings": warnings,
    }


def _references(context: ProjectContext) -> dict[str, dict[str, Any]]:
    """Snapshot and validate every edit or lane that can reference source words."""
    result = {}
    for relative in _REFERENCE_FILES:
        try:
            data = context.json(relative)
        except FileNotFoundError:
            continue
        parse_document("edit" if relative == "edit.json" else "timeline", data, filename=relative)
        result[relative] = data
    return result


def _replacement_guard(
    context: ProjectContext,
    documents: dict[str, dict[str, Any]],
    replacements: dict[str, Words],
    *,
    force: bool,
) -> list[dict[str, Any]]:
    """Refuse changed referenced words, or enumerate every damaged word anchor."""
    changed: dict[str, tuple[dict[str, Word], dict[str, Word]]] = {}
    for take, new in replacements.items():
        old_path = context.project_dir / f"ingest/{take}/words.json"
        old = load_words(old_path) if old_path.exists() else None
        if old is None or old.to_dict() != new.to_dict():
            changed[take] = (
                {word.id: word for word in old.words} if old else {},
                {word.id: word for word in new.words},
            )
    files = set()
    orphans = []
    for relative, data in documents.items():
        is_edit = relative == "edit.json"
        for item in data.get("segments" if is_edit else "elements", []):
            if is_edit and item["take"] in changed:
                files.add(relative)
            for edge in ("from", "to") if is_edit else ("start", "end"):
                anchor = item.get(edge) or {}
                word_id = anchor.get("word")
                if not word_id or word_id.split(":", 1)[0] not in changed:
                    continue
                files.add(relative)
                old_words, new_words = changed[word_id.split(":", 1)[0]]
                old_word, new_word = old_words.get(word_id), new_words.get(word_id)
                old_text = old_word.text if old_word else anchor.get("text", "")
                if new_word is None or transcript.norm_words(old_text) != transcript.norm_words(
                    new_word.text
                ):
                    orphans.append(
                        {
                            "file": relative,
                            "segment" if is_edit else "element": item["id"],
                            "anchor": edge,
                            "word": word_id,
                            "old_word": old_text,
                        }
                    )
    if files and not force:
        raise IngestRefusal(sorted(files))
    return orphans


def _json_output(path: Path, data: dict[str, Any]) -> Path:
    """Stage deterministic JSON through the guarded atomic writer."""
    return paths.atomic_write_text(
        path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def ingest(
    project_dir: Path,
    *,
    force: bool = False,
    copy: bool = False,
    asr: str | None = None,
) -> dict:
    """Analyze all takes and publish verified source words under one stage receipt."""
    with paths.operation():
        context = ProjectContext(project_dir)
        root = context.project_dir
        inputs = ingest_inputs(context)
        run = StageRun(root, "ingest")
        try:
            previous = context.json("ingest/ingest.json")
        except FileNotFoundError:
            previous = {}
        prior_receipt = read_receipt(root / "ingest") or {}
        requested = asr or context.project.asr["engine"]
        previous_takes = {item["id"]: item for item in previous.get("takes", [])}
        copies_ready = all(
            (root / take.path).resolve().is_relative_to(root)
            or previous_takes.get(take.id, {}).get("copy")
            for take in context.project.takes
        )
        streams_match = all(
            previous_takes.get(take.id, {}).get("requested_audio_stream") == take.audio_stream
            for take in context.project.takes
        )
        if (
            not force
            and run.up_to_date(inputs)
            and previous
            and all("speech_check" in item for item in previous_takes.values())
            and requested == previous.get("requested_asr")
            and streams_match
            and (not copy or copies_ready)
        ):
            return {**previous, "noop": True, "orphans": []}
        inputs.update({f"take:{take.id}": context.take_sha256(take.id)
                       for take in context.project.takes})
        source_entries = {
            take.id: context.identity.known_entry(context.take_path(take.id))
            for take in context.project.takes
        }
        documents = _references(context)
        backend = transcript.resolve_backend(requested, context.project.asr.get("model"))
        if backend is None:
            raise DependencyError.missing_engine("speech engine")
        work_root = paths.validate_output_dir(root / ".yap/work")
        with (
            run,
            tempfile.TemporaryDirectory(prefix=f"ingest-{os.getpid()}-", dir=work_root) as raw,
        ):
            work = Path(raw)
            for name, digest in inputs.items():
                run.record_inputs(name, digest)
            decoded: dict[str, _Decoded] = {}
            for index, take in enumerate(context.project.takes):
                run.progress(f"decoding {take.id}", index / len(context.project.takes) * 0.4)
                decoded[take.id] = _decode(
                    context.take_path(take.id), work / f"{take.id}.wav", take.audio_stream
                )
            run.progress("transcribing", 0.4)
            reusable = {}
            owned = {item["path"]: item["sha256"] for item in prior_receipt.get("outputs", [])}
            if (not force and prior_receipt.get("status") == "verified" and
                    requested == previous.get("requested_asr") and
                    context.project.language == previous.get("language") and
                    backend.name == previous.get("engine") and
                    backend.model == previous.get("model")):
                for take in context.project.takes:
                    old = previous_takes.get(take.id, {})
                    transcript_path = root / f"ingest/{take.id}/transcript.json"
                    if (old.get("sha256") == inputs[f"take:{take.id}"] and
                            old.get("requested_audio_stream") == take.audio_stream and
                            owned.get(f"ingest/{take.id}/transcript.json") and
                            transcript_path.is_file() and
                            receipt.sha256_file(transcript_path) ==
                            owned[f"ingest/{take.id}/transcript.json"]):
                        reusable[decoded[take.id].wav] = transcript.Transcript.from_dict(
                            json.loads(transcript_path.read_bytes()))
            raw = shared.transcribe(backend, [
                shared.Request(context.take_sha256(take.id), decoded[take.id].stream,
                               decoded[take.id].wav, source=decoded[take.id].source,
                               source_entry=source_entries[take.id])
                for take in context.project.takes if decoded[take.id].wav not in reusable
            ], language=context.project.language, refresh=force)
            results = {**reusable, **{path: result for path, (result, _hit) in raw.items()}}
            run.progress("checking speech coverage", 0.8)
            checks: dict[str, dict[str, Any]] = {}
            requests: list[tuple[str, tuple[float, float], Path]] = []
            for take in context.project.takes:
                item = decoded[take.id]
                if item.wav not in results:
                    raise ProcessingError(
                        "speech engine", 1, f"missing result for {take.id}",
                        code="speech_engine_failed",
                    )
                samples = _float_wav(item.wav)
                level = edges._levels(samples)
                clusters = coverage.uncovered_speech(level, results[item.wav].words)
                ranges = coverage.windows(
                    clusters, results[item.wav].words, len(samples) / SAMPLE_RATE
                )
                checks[take.id] = {
                    "level": level, "windows": [], "words_added": 0,
                }
                if item.wav in reusable:
                    continue
                for index, window in enumerate(ranges):
                    first, last = (round(time * SAMPLE_RATE) for time in window)
                    exact = (first / SAMPLE_RATE, last / SAMPLE_RATE)
                    target = work / f"{take.id}-speech-{index}.wav"
                    _speech_clip(samples, exact, target)
                    requests.append((take.id, exact, target))
                    checks[take.id]["windows"].append(exact)
            if requests:
                recovered = backend.transcribe_many(
                    [target for _, _, target in requests], language=context.project.language
                )
                for take_id, window, target in requests:
                    if target not in recovered:
                        raise ProcessingError(
                            "speech engine", 1, f"missing result for {take_id}",
                            code="speech_engine_failed",
                        )
                    result = results[decoded[take_id].wav]
                    result.words, added = coverage.merge(
                        result.words, window, recovered[target].words
                    )
                    checks[take_id]["words_added"] += added
            for take in context.project.takes:
                item = decoded[take.id]
                check = checks[take.id]
                remaining = coverage.uncovered_speech(check["level"], results[item.wav].words)
                check["untranscribed"] = [
                    {
                        "start": _rounded(start + item.offset_s),
                        "end": _rounded(end + item.offset_s),
                        "seconds": _rounded(end - start),
                    }
                    for start, end in remaining
                ]
            replacements = {}
            outputs: list[tuple[Path, Path]] = []
            take_facts = []
            for take in context.project.takes:
                item = decoded[take.id]
                result = results[item.wav]
                staged_transcript = _json_output(
                    work / f"{take.id}-transcript.json", result.to_dict()
                )
                words = Words(
                    take.id,
                    inputs[f"take:{take.id}"],
                    "source",
                    item.offset_s,
                    backend.name,
                    backend.model,
                    receipt.sha256_file(staged_transcript),
                    _source_words(take.id, result, item.offset_s,
                                  item.info.video_duration_s or item.info.duration_s),
                )
                replacements[take.id] = words
                staged_words = _json_output(work / f"{take.id}-words.json", words.to_dict())
                old_path = root / f"ingest/{take.id}/words.json"
                if old_path.is_file():
                    old_bytes = old_path.read_bytes()
                    if hashlib.sha256(old_bytes).hexdigest() != receipt.sha256_file(staged_words):
                        old_hash = hashlib.sha256(old_bytes).hexdigest()
                        snapshot = root / f"ingest/{take.id}/previous/{old_hash}.json"
                        if snapshot.exists():
                            if receipt.sha256_file(snapshot) != old_hash:
                                raise ConflictError(snapshot)
                        else:
                            staged_old = work / f"{take.id}-previous.json"
                            with paths.atomic_output(staged_old) as temporary:
                                temporary.write_bytes(old_bytes)
                            outputs.append((snapshot, staged_old))
                outputs.extend(
                    [
                        (root / f"ingest/{take.id}/transcript.json", staged_transcript),
                        (root / f"ingest/{take.id}/words.json", staged_words),
                    ]
                )
                facts = {
                    "id": take.id,
                    "path": take.path,
                    "sha256": words.take_sha256,
                    "words_sha256": receipt.sha256_file(staged_words),
                    "requested_audio_stream": take.audio_stream,
                    **_facts(item, words.words),
                }
                check = checks[take.id]
                facts["speech_check"] = {
                    "windows": [
                        [_rounded(start + item.offset_s), _rounded(end + item.offset_s)]
                        for start, end in check["windows"]
                    ],
                    "words_added": check["words_added"],
                    "untranscribed": check["untranscribed"],
                    "thresholds": dict(coverage.THRESHOLDS),
                }
                facts["warnings"].extend(
                    coverage.speech_warning(**span) for span in check["untranscribed"]
                )
                recorded_copy = previous_takes.get(take.id, {}).get("copy")
                if recorded_copy and item.source == (root / recorded_copy).resolve():
                    facts["copy"] = recorded_copy
                elif copy and not item.source.is_relative_to(root):
                    copied = work / f"{take.id}-copy{item.source.suffix}"
                    digest = hashlib.sha256()
                    with paths.atomic_output(copied) as temporary:
                        with item.source.open("rb") as source, temporary.open("wb") as target:
                            while chunk := source.read(1024 * 1024):
                                digest.update(chunk)
                                target.write(chunk)
                    if (
                        digest.hexdigest() != words.take_sha256
                        or receipt.sha256_file(copied) != words.take_sha256
                    ):
                        raise InputError(
                            take.path, "copy hash mismatch; re-read the source and retry"
                        )
                    relative = f"ingest/takes/{take.id}{item.source.suffix}"
                    facts["copy"] = relative
                    outputs.append((root / relative, copied))
                take_facts.append(facts)
            for target, _staged in outputs:
                paths.validate_output(target, owner_receipts=(run.receipt_path, run.pending_path))
            orphans = _replacement_guard(context, documents, replacements, force=force)
            manifest = {
                "schema": "yap-studio/ingest@1",
                "engine": backend.name,
                "model": backend.model,
                "requested_asr": requested,
                "language": context.project.language,
                "takes": take_facts,
                "warnings": [
                    f"{item['id']}: {warning}"
                    for item in take_facts
                    for warning in item["warnings"]
                ],
                "orphans": orphans,
                **({"next": "yap-studio edit rebase DIR"} if orphans else {}),
            }
            outputs.append(
                (root / "ingest/ingest.json", _json_output(work / "ingest.json", manifest))
            )
            run.progress("publishing", 0.9)
            for take in context.project.takes:
                if context.identity.sha256(decoded[take.id].source) != inputs[f"take:{take.id}"]:
                    raise ConflictError(decoded[take.id].source)
            with project_lock(root):
                for relative in ("project.json", *_REFERENCE_FILES):
                    try:
                        before = context.read(relative)
                    except FileNotFoundError:
                        before = None
                    target = root / relative
                    current = target.read_bytes() if target.exists() else None
                    if current != before:
                        raise ConflictError(target)
            for target, staged in outputs:
                run.replace_output(target, staged)
            run.finish(
                asserts=[
                    {"id": "take_hashes", "pass": True},
                    {"id": "source_clock", "pass": True},
                    {"id": "speech_covered", "pass": not any(
                        item["speech_check"]["untranscribed"] for item in take_facts
                    )},
                ],
                extra={
                    "orphans": orphans,
                    "untranscribed": [
                        {"take": item["id"], **span}
                        for item in take_facts
                        for span in item["speech_check"]["untranscribed"]
                    ],
                },
            )
            for take in context.project.takes:
                source = decoded[take.id].source
                try:
                    entry = context.identity.known_entry(source)
                    if entry is not None and entry["sha256"] == inputs[f"take:{take.id}"]:
                        shared.remember_digest(source, entry["sha256"], entry)
                except OSError:
                    pass  # A path memo can always be made again.
            return {**manifest, "noop": False}


def preflight(clip: Path, *, asr: str = "auto") -> dict:
    """Return recording readiness without creating or writing a project."""
    with paths.operation(), tempfile.TemporaryDirectory(prefix="yap-preflight-") as raw:
        decoded = _decode(clip, Path(raw) / "audio.wav")
        backend = transcript.resolve_backend(asr)
        if backend is None:
            raise DependencyError.missing_engine("speech engine")
        results = backend.transcribe_many([decoded.wav])
        if decoded.wav not in results:
            raise ProcessingError(
                "speech engine", 1, "missing preflight transcript", code="speech_engine_failed"
            )
        words = _source_words("t1", results[decoded.wav], decoded.offset_s,
                              decoded.info.video_duration_s or decoded.info.duration_s)
        music = _music(decoded.wav, words, decoded.offset_s)
        level, peak = decoded.loudness["integrated_lufs"], decoded.loudness["true_peak_dbtp"]
        checks = {
            "at_least_five_words": len(words) >= 5,
            "no_music": not music["ranges"],
            "loudness": level is not None and -28 <= level <= -16,
            "true_peak": peak is not None and peak <= -0.3,
        }
        notes = (
            ["hot peak: back the mic gain off a little"]
            if peak is not None and -1.0 <= peak <= -0.3
            else []
        )
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "notes": notes,
            "words_count": len(words),
            "integrated_lufs": level,
            "true_peak_dbtp": peak,
            "music_ranges": music["ranges"],
        }


def words_listing(project_dir: Path, *, cut: bool = False) -> dict:
    """List word ids and compact sentences on the source or output clock."""
    root = Path(project_dir).resolve()
    words: list[dict[str, Any]] = []
    if cut:
        target = root / "cut/cut-words.json"
        if not target.is_file():
            raise InputError("cut/cut-words.json", "missing cut words; run yap-studio cut first")
        words.extend(load_cut_words(paths.validate_input(target)).to_dict()["words"])
    else:
        context = ProjectContext(root)
        for take in context.project.takes:
            target = root / f"ingest/{take.id}/words.json"
            if not target.is_file():
                raise InputError(
                    f"ingest/{take.id}/words.json",
                    "missing source words; run yap-studio ingest first",
                )
            words.extend(load_words(paths.validate_input(target)).to_dict()["words"])
    lines = []
    sentence: list[dict[str, Any]] = []

    def finish() -> None:
        if sentence:
            first, last = sentence[0], sentence[-1]
            lines.append(
                f"{first['id']}-{last['id']}  {first['start']:.3f}-{last['end']:.3f}  "
                + " ".join(word["text"] for word in sentence)
            )
            sentence.clear()

    for word in words:
        if sentence and (
            word["start"] - sentence[-1]["end"] > 1
            or word["id"].split(":")[0] != sentence[-1]["id"].split(":")[0]
            or word.get("segment") != sentence[-1].get("segment")
        ):
            finish()
        sentence.append(word)
        if word["text"].rstrip().endswith((".", "!", "?")):
            finish()
    finish()
    return {"time_base": "output" if cut else "source", "words": words, "lines": lines}
