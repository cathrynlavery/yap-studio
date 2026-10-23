"""Watch pack orchestration, receipt safety, and the deterministic reading guide."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import __version__, audio, ffmpeg, paths, transcript
from yap_studio import video as video_module
from yap_studio.checks.detect import detect_cuts
from yap_studio.edl import load_edit_list
from yap_studio.errors import (
    DependencyError,
    InputError,
    OutputPathError,
    ValidationError,
    YapError,
)
from yap_studio.plan import Plan
from yap_studio.receipt import sha256_file
from yap_studio.transcript import Backend, Transcript, shared

from . import PACK_SCHEMA, load_pack
from . import images as im
from . import measure as m
from .command import ASR_CHOICES

PENDING_SCHEMA = "yap-studio/watch-pack-pending@1"
_BOOKKEEPING_NAMES = frozenset({"pack.json", "pack.pending.json"})
_OUTPUT_NAMES = frozenset(
    {"pack.md", "motion.json", "transcript.json", "transcript.txt", "verify.json"}
)
_STRIP_NAME = re.compile(r"strips/strip-[0-9]{3,}\.png")
_TEMP_PREFIX = ".yap_watch_tmp_"
_TEMP_NAME = re.compile(r"\.yap_watch_tmp_[a-z0-9_]{8}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise InputError(str(path), "unreadable JSON") from exc
    if not isinstance(data, dict):
        raise ValidationError(str(path), "expected a JSON object")
    return data


def _refusal(message: str) -> OutputPathError:
    error = OutputPathError(message, code="unsafe_path")
    error.suggested_action = "Choose a new --out folder, or restore the previous pack's files."
    return error


def _pack_name(name: Any, *, bookkeeping: bool = False, temporary: bool = False) -> str:
    """Require a canonical name from the pack's fixed ownership namespace."""
    if not isinstance(name, str) or not (
        name in _OUTPUT_NAMES
        or _STRIP_NAME.fullmatch(name)
        or bookkeeping and name in _BOOKKEEPING_NAMES
        or temporary and _TEMP_NAME.fullmatch(name)
    ):
        raise _refusal("Watch pack lists an unsafe or unknown output path.")
    return name


def pack_output_name(name: Any) -> str:
    """Require a canonical output name, excluding bookkeeping and temporary files."""
    return _pack_name(name)


def _folder_root(directory: Path, inputs: list[Path] | None = None) -> Path:
    """Check the folder itself without following a link, including dangling links."""
    if ".." in directory.parts or directory.is_symlink():
        raise _refusal("Output folder must not be a symlink or contain parent components.")
    resolved = directory.resolve()
    if inputs is not None and any(item == resolved or resolved in item.parents for item in inputs):
        raise _refusal("Output folder refers to or contains an input file.")
    try:
        mode = directory.lstat().st_mode
    except FileNotFoundError:
        return resolved
    if not stat.S_ISDIR(mode):
        raise _refusal("Output folder is not a directory.")
    return resolved


def _pack_file(directory: Path, name: str) -> Path:
    """Check every component before resolving or reading a known pack file."""
    _pack_name(name, bookkeeping=True, temporary=True)
    root = _folder_root(directory)
    target = directory / name
    if target.parent != directory:
        try:
            mode = target.parent.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISDIR(mode):
                raise _refusal("Pack strips must be a real directory, not a symlink.")
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        pass
    else:
        if not stat.S_ISREG(mode):
            raise _refusal("Output folder contains a symlink or a non-regular file.")
    if not target.resolve().is_relative_to(root):
        raise _refusal("Pack file resolves outside the output folder.")
    return target


def _current_hash(directory: Path, name: str) -> str | None:
    target = _pack_file(directory, name)
    return sha256_file(target) if target.exists() else None


def _require_hash(directory: Path, name: str, digest: str | None) -> Path:
    """Recheck containment, regular-file status, and the journaled current hash."""
    if _current_hash(directory, name) != digest:
        raise _refusal("Pack file changed during publication.")
    return directory / name


def _receipt_hashes(data: dict[str, Any], *, pending: bool = False) -> dict[str, set[str]]:
    """Validate receipt names and hashes before letting them prove ownership."""
    outputs = data.get("outputs")
    if not isinstance(outputs, list):
        raise _refusal("Watch pack bookkeeping has no output receipt.")
    expected: dict[str, set[str]] = {}
    for item in outputs:
        if not isinstance(item, dict):
            raise _refusal("Watch pack has an invalid output receipt.")
        name = _pack_name(item.get("path"), bookkeeping=pending, temporary=pending)
        if name == "pack.pending.json" or name in expected:
            raise _refusal("Watch pack lists a bookkeeping or duplicate output path.")
        if pending:
            if not {"sha256", "old_sha256", "new_sha256"} <= item.keys():
                raise _refusal("Watch pack journal is missing current or intended hashes.")
            old, new = item["old_sha256"], item["new_sha256"]
            if item["sha256"] != old or old is None and new is None:
                raise _refusal("Watch pack journal has inconsistent output hashes.")
            if _TEMP_NAME.fullmatch(name) and new is not None:
                raise _refusal("Temporary pack files may only be journaled for deletion.")
            digests = [value for value in (old, new) if value is not None]
        else:
            digests = [item.get("sha256")]
        if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in digests):
            raise _refusal("Watch pack has an invalid output hash.")
        expected[name] = set(digests)
    return expected


def _check_folder(directory: Path, inputs: list[Path]) -> list[Path]:
    """Accept clean packs or journal-owned interruptions, refusing unsafe folders."""
    _folder_root(directory, inputs)
    if not directory.exists():
        return []
    files: dict[str, Path] = {}
    for item in directory.iterdir():
        mode = item.lstat().st_mode
        if stat.S_ISDIR(mode):
            if item.name != "strips":
                raise _refusal("The only directory allowed in a watch pack is strips.")
            for child in item.iterdir():
                name = child.relative_to(directory).as_posix()
                files[name] = _pack_file(directory, name)
        else:
            files[item.name] = _pack_file(directory, item.name)
    leftovers = {name: item for name, item in files.items() if _TEMP_NAME.fullmatch(name)}
    files = {name: item for name, item in files.items() if name not in leftovers}
    if not files:
        return sorted(leftovers.values())
    expected: dict[str, set[str]] = {}
    pending = "pack.pending.json" in files
    actual = {name: item for name, item in files.items() if name not in _BOOKKEEPING_NAMES}
    try:
        if pending:
            # Folder writers can also rewrite the journal, so this guards interrupted
            # runs and accidental edits, not local attackers. Known names, containment
            # and symlink checks still keep deletions inside the pack.
            journal = _read_json(files["pack.pending.json"])
            if journal.get("schema") != PENDING_SCHEMA:
                raise _refusal(f"Watch pack journal must use schema {PENDING_SCHEMA}.")
            expected = _receipt_hashes(journal, pending=True)
        elif "pack.json" in files:
            expected = _receipt_hashes(load_pack(directory))
        else:
            raise _refusal("Nonempty output folder must contain a previous watch pack.")
        checked = {name: item for name, item in files.items() if name != "pack.pending.json"}
        if not pending:
            checked = actual
        if not pending and actual.keys() != expected.keys() or any(
            _current_hash(directory, name) not in expected.get(name, set()) for name in checked
        ):
            raise _refusal("Output folder contains foreign, missing, or modified pack files.")
        if pending and "pack.json" in files:
            old_hashes = {item["path"]: item["old_sha256"] for item in journal["outputs"]}
            if _current_hash(directory, "pack.json") == old_hashes["pack.json"]:
                indexed = {
                    name: next(iter(digests))
                    for name, digests in _receipt_hashes(load_pack(directory)).items()
                }
                if any(
                    digest != indexed.get(name)
                    for name, digest in old_hashes.items()
                    if name not in _BOOKKEEPING_NAMES and not _TEMP_NAME.fullmatch(name)
                ):
                    raise _refusal("Watch pack journal old hashes do not match the old index.")
    except (YapError, RecursionError) as exc:
        raise _refusal("Output folder has invalid watch pack bookkeeping.") from exc
    return sorted([*actual.values(), *leftovers.values()])


@contextmanager
def _offline() -> Iterator[None]:
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def _validate_transcript(result: Transcript) -> Transcript:
    for word in result.words:
        if (
            not math.isfinite(word.start)
            or not math.isfinite(word.end)
            or word.start < 0
            or word.end < word.start
            or word.conf is not None
            and not math.isfinite(word.conf)
        ):
            raise ValidationError("words", "word times must be finite, non-negative, and ordered")
    result.words.sort(key=lambda word: (word.start, word.end, word.text))
    return result


def _get_transcript(
    source: Path,
    words: Path | None,
    asr: str,
    backend: Backend | None,
    info: ffmpeg.MediaInfo,
    digest: str,
) -> tuple[dict[str, Any], Transcript | None]:
    record = {
        "status": "ok",
        "reason": None,
        "source": None,
        "engine": None,
        "model": None,
        "words": None,
        "file": None,
        "text_file": None,
    }
    if words is not None:
        data = _read_json(words)
        try:
            if data.get("schema") == "yap-studio/cut-words@1":
                if data.get("time_base") != "output":
                    raise ValidationError("words.time_base", "cut words must use output time")
                result = Transcript.from_dict({**data, "schema": transcript.SCHEMA})
            else:
                result = transcript.load_any(words)
            _validate_transcript(result)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, YapError):
                raise
            raise ValidationError("words", "invalid transcript words or times") from exc
        record["source"] = "file"
    elif asr == "none":
        record.update(status="skipped", reason="speech transcription disabled by --asr none")
        return record, None
    elif not info.has_audio:
        record.update(status="skipped", reason="video has no audio stream")
        return record, None
    else:
        try:
            with _offline():
                engine = backend if backend is not None else transcript.resolve_backend(asr)
        except DependencyError as exc:
            record.update(status="unavailable", reason=exc.message)
            return record, None
        try:
            with _offline():
                if engine is None:
                    record.update(
                        status="unavailable",
                        reason="no local speech engine installed; "
                        "pip install 'yap-studio[asr]' or pass --words",
                    )
                    return record, None
                record.update(source="asr", engine=engine.name, model=engine.model)
                request = shared.Request(digest, 0, source)
                result = shared.lookup(engine, request)
                if result is None and engine.download_notice() is not None:
                    record.update(
                        status="skipped",
                        reason="speech model is not downloaded yet; "
                        "run yap-studio verify or yap-studio ingest once while online, "
                        "or pass --words",
                    )
                    return record, None
                if result is None:
                    result, _hit = shared.transcribe(engine, [request])[source]
                result = _validate_transcript(result)
                result = replace(
                    result,
                    words=m.words_on_picture_clock(
                        result.words,
                        info.audio_offset_s,
                        info.video_offset_s,
                    ),
                )
        except YapError as exc:
            record.update(status="failed", reason=exc.message)
            return record, None
    record.update(
        engine=record["engine"] or result.backend,
        model=record["model"] or result.model,
        words=len(result.words),
        file="transcript.json",
        text_file="transcript.txt",
    )
    return record, result


def _verify(source: Path | None, digest: str) -> tuple[dict[str, Any], bytes | None]:
    record = {
        "status": "missing",
        "reason": "no verify report found",
        "file": None,
        "result": None,
        "summary_text": None,
        "findings": [],
    }
    if source is None:
        return record, None
    data = _read_json(source)
    schema = data.get("schema")
    if not isinstance(schema, str) or not schema.startswith("yap-studio/verify-report@"):
        raise ValidationError("verify.schema", "expected a yap-studio verify report")
    receipt = data.get("receipt", {})
    inputs = receipt.get("inputs", []) if isinstance(receipt, dict) else []
    if not isinstance(inputs, list) or not any(
        isinstance(item, dict) and item.get("sha256") == digest for item in inputs
    ):
        record.update(status="stale", reason="verify report does not match the video's sha256")
        return record, None
    summary = data.get("summary", {})
    if not isinstance(summary, dict) or not isinstance(data.get("findings", []), list):
        raise ValidationError("verify", "invalid summary or findings")
    findings = [
        {key: finding.get(key) for key in ("check", "severity", "time_s", "message")}
        for finding in data.get("findings", [])
        if isinstance(finding, dict)
    ]
    for finding in findings:
        if isinstance(finding["time_s"], (int, float)):
            finding["time_s"] = round(finding["time_s"], 3)
    record.update(
        status="found",
        reason=None,
        file="verify.json",
        result=data.get("status"),
        summary_text=summary.get("text"),
        findings=findings,
    )
    return record, source.read_bytes()


def _discover_verify(video: Path, digest: str) -> tuple[Path | None, dict[str, Any], bytes | None]:
    """Try stem then folder reports for a matching input sha256, skipping invalid JSON.

    If none matches, mark the first existing candidate stale; if none exists, missing.
    Explicit --verify files use _verify directly and retain its validation errors.
    """
    first: Path | None = None
    for candidate in (video.with_suffix(".verify.json"), video.parent / "verify.json"):
        if not candidate.exists():
            continue
        if first is None:
            first = candidate.resolve()
        try:
            source = paths.validate_input(candidate)
            record, content = _verify(source, digest)
        except (InputError, ValidationError, OSError):
            continue
        if record["status"] == "found":
            return source, record, content
    record, content = _verify(None, digest)
    if first is not None:
        record.update(status="stale", reason="verify report does not match the video's sha256")
    return first, record, content


def _structure(
    source: Path,
    edl: Path | None,
    info: ffmpeg.MediaInfo,
    frames: int,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict | None, list[str]
]:
    rate = float(info.fps)
    graphics: list[dict[str, Any]] = []
    warnings: list[str] = []
    edl_record = None
    if edl is not None:
        edits = load_edit_list(edl)
        plan = Plan.build(edits, fps=info.fps, sample_rate=info.sample_rate or 48000)
        warnings = plan.warnings
        cut_frames = [(cut.index, cut.out_frame) for cut in plan.cuts]
        bounds = [
            (segment.index, segment.out_f0, segment.out_f0 + segment.n_frames)
            for segment in plan.segments
        ]
        edl_record = {
            "path": str(edl),
            "sha256": sha256_file(edl),
            "segments": len(plan.segments),
            "cuts": len(plan.cuts),
        }
        for start, end in edits.picture_masks:
            graphics.append(
                {
                    "kind": "mask",
                    "start_s": round(float(start), 3),
                    "end_s": round(float(end), 3),
                    "rect": None,
                }
            )
        for region in edits.picture_regions:
            graphics.append(
                {
                    "kind": "region",
                    "start_s": round(float(region.start), 3),
                    "end_s": round(float(region.end), 3),
                    "rect": [round(value, 3) for value in region.rect],
                }
            )
        for hold in plan.holds:
            graphics.append(
                {
                    "kind": "hold",
                    "start_s": round(hold.out_f0 / rate, 3),
                    "end_s": round(hold.out_f1 / rate, 3),
                }
            )
        if plan.total_frames != frames:
            warnings.append(
                f"Edit list plans {plan.total_frames} frames; video has {frames} frames."
            )
    else:
        cut_frames = [(cut.index, cut.frame) for cut in detect_cuts(source)]
        edges = [0, *(frame for _, frame in cut_frames if 0 < frame < frames), frames]
        bounds = [
            (index, start, end)
            for index, (start, end) in enumerate(zip(edges, edges[1:], strict=False), 1)
        ]
    cuts = []
    for index, frame in cut_frames:
        if not 0 < frame < frames:
            warnings.append(f"Cut {index} at frame {frame} is outside the video and was omitted.")
            continue
        cuts.append(
            {
                "index": index,
                "frame": frame,
                "time_s": round(frame / rate, 3),
                "source": "edl" if edl else "detected",
                "before": {"frame": frame - 1, "time_s": round((frame - 1) / rate, 3)},
                "after": {"frame": frame, "time_s": round(frame / rate, 3)},
            }
        )
    cuts.sort(key=lambda cut: (cut["frame"], cut["index"]))
    segments = [
        {
            "index": index,
            "start_frame": max(0, start),
            "end_frame": min(frames, end),
            "start_s": round(max(0, start) / rate, 3),
            "end_s": round(min(frames, end) / rate, 3),
            "shot_s": round((min(frames, end) - max(0, start)) / rate, 3),
        }
        for index, start, end in bounds
        if min(frames, end) > max(0, start)
    ]
    return cuts, segments, graphics, edl_record, warnings


def _sound(source: Path, info: ffmpeg.MediaInfo, duration: float) -> tuple[dict, list]:
    if not info.has_audio:
        return {"status": "no_audio", "integrated_lufs": None, "per_second": [], "silences": []}, []
    # Place block ends on the picture's zero-based clock, matching the grey decoder.
    offset = info.audio_offset_s - info.video_offset_s
    result = ffmpeg.run(
        [
            "ffmpeg",
            "-loglevel",
            "verbose",
            "-nostats",
            "-i",
            str(source),
            "-map",
            f"0:{info.audio_index}",
            "-vn",
            "-af",
            f"asetpts=PTS-STARTPTS+({offset:.9f})/TB,ebur128=framelog=verbose",
            "-f",
            "null",
            "-",
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    blocks, integrated = m.parse_ebur128(result.stderr)
    sr, samples = audio.decode(source, sr=m.SILENCE_SAMPLE_RATE, mono=True)
    start = round(info.video_offset_s * sr)
    if start < 0:
        samples = np.pad(samples, (-start, 0))
        start = 0
    count = round(duration * sr)
    samples = samples[start : start + count]
    if len(samples) < count:
        samples = np.pad(samples, (0, count - len(samples)))
    epsilon = 2 / (info.sample_rate or 48000)
    return {
        "status": "ok",
        "integrated_lufs": None if integrated is None else round(integrated, 1),
        "per_second": m.loudness_per_second(blocks, duration, epsilon=epsilon),
        "silences": m.silences(samples, sr),
    }, blocks


def _transcript_text(
    result: Transcript, segments: list[dict], rate: float, engine: str | None
) -> str:
    lines = [f"Transcript, engine: {engine or 'file'}", ""]
    for segment in segments:
        words = m.segment_words(
            result.words, segment["start_frame"] / rate, segment["end_frame"] / rate
        )
        for offset in range(0, len(words), 12):
            group = words[offset : offset + 12]
            lines.append(f"[{im.format_time(group[0].start)}] " + " ".join(w.text for w in group))
    return "\n".join(lines) + "\n"


def _markdown(pack: dict) -> str:
    from shlex import quote

    from .findings import KINDS, SEVERITIES

    facts = pack["video"]
    lines = [
        f"# Watch pack: {facts['file']}",
        "",
        f"Video: {facts['width']} x {facts['height']}, {facts['fps']} fps, "
        f"{facts['frames']} frames, {facts['duration_s']:.3f} seconds.",
        "All times use video seconds from frame zero; frame ends are exclusive.",
        "",
        "Read this guide, then look at the strips in time order. Labels show M:SS.ss.",
        "C3< is the frame before cut 3, with an orange border. "
        "C3> is the frame after it, with a cyan border.",
        "Columns start at 1. Cut pairs stay together; a connected cut group can use "
        "more columns than requested.",
        "",
    ]
    for warning in pack["warnings"]:
        lines.append(f"Plan warning: {warning}")
    tr = pack["transcript"]
    lines.extend(
        [
            f"Transcript: {tr['status']}, {tr['reason'] or str(tr['words']) + ' words'}.",
            "Read transcript.txt for segment lines and transcript.json for word times."
            if tr["file"]
            else "Word pacing is unavailable.",
            "",
            "## Segments",
            "",
            "| Segment | Time | Shot s | Words | Words/s | Longest pause s, at | Silence s |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    def number(value: Any) -> str:
        return "n/a" if value is None else str(value)

    for segment in pack["segments"]:
        at = segment["longest_pause_at_s"]
        pause = (
            number(segment["longest_pause_s"])
            + ", "
            + ("n/a" if at is None else im.format_time(at))
        )
        lines.append(
            f"| {segment['index']} | {im.format_time(segment['start_s'])} to "
            f"{im.format_time(segment['end_s'])} | {segment['shot_s']} | "
            f"{number(segment['words'])} | {number(segment['words_per_s'])} | "
            f"{pause} | {number(segment['longest_silence_s'])} |"
        )
    locations = {}
    for strip in pack["timeline"]["strips"]:
        for tile in strip["tiles"]:
            for role in tile["cut_roles"]:
                locations[(role["cut"], role["kind"])] = (
                    f"{strip['image']}, column {tile['column']}"
                )
    lines.extend(
        [
            "",
            "## Cuts",
            "",
            "| Cut | Time | Before tile | After tile | Before LUFS | After LUFS | Jump LU |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for cut in pack["cuts"]:
        lines.append(
            f"| {cut['index']} | {im.format_time(cut['time_s'])} | "
            f"{locations.get((cut['index'], 'cut_before'), 'n/a')} | "
            f"{locations.get((cut['index'], 'cut_after'), 'n/a')} | "
            f"{number(cut['loudness_before_lufs'])} | {number(cut['loudness_after_lufs'])} | "
            f"{number(cut['loudness_jump_lu'])} |"
        )
    for title, spans in (
        ("Still spans", pack["motion"]["still_spans"]),
        ("Motion spans", pack["motion"]["motion_spans"]),
        (
            "Silences longer than 1 s",
            [span for span in pack["loudness"]["silences"] if span["duration_s"] > 1],
        ),
    ):
        lines.extend(["", f"## {title}", ""])
        lines.extend(
            f"- {im.format_time(span['start_s'])} to {im.format_time(span['end_s'])}, "
            f"{span['duration_s']} seconds"
            for span in spans
        )
        if not spans:
            lines.append("None.")
    vr = pack["verify"]
    lines.extend(["", "## Verify", "", f"{vr['status']}: {vr['result'] or vr['reason']}."])
    if vr["summary_text"]:
        lines.append(vr["summary_text"])
    for finding in vr["findings"]:
        if finding["severity"] in {"fail", "warn"}:
            at = "" if finding["time_s"] is None else im.format_time(finding["time_s"]) + ": "
            lines.append(f"- {finding['severity']}, {finding['check']}: {at}{finding['message']}")
    lines.extend(["", "## Strips", ""])
    for strip in pack["timeline"]["strips"]:
        lines.append(
            f"- [{strip['image']}]({strip['image']}), "
            f"{im.format_time(strip['start_s'])} to {im.format_time(strip['end_s'])}"
        )
    video = quote(facts["path"])
    lines.extend(
        [
            "",
            "## Reporting findings",
            "",
            "From this pack folder, run `yap-studio watch scan .` for machine hints, "
            f"then `yap-studio watch list {video}` to read them.",
            "Look at the strips and transcript for slow stretches, jarring cuts, "
            "covered faces, caption problems and audio problems.",
            f"Kinds: {', '.join(KINDS)}. Severities: {', '.join(SEVERITIES)}.",
            "Write FINDINGS.json with your name and point or range findings:",
            "",
            "```json",
            '{"reviewer": "<your name>", "findings": [',
            '  {"at": 12.4, "kind": "audio", "severity": "medium", '
            '"reason": "Speech level jumps.", "fix": "Match the speech levels."},',
            '  {"start": 30.0, "end": 36.5, "kind": "slow", "severity": "medium", '
            '"reason": "The picture stays still.", "fix": "Shorten this stretch."}',
            "]}",
            "```",
            "",
            "Use times within this video. Record your findings with "
            f"`yap-studio watch note {video} --file FINDINGS.json`.",
        ]
    )
    return ("\n".join(lines) + "\n").replace(chr(0x2013), " to ").replace(chr(0x2014), ", ")


def build_pack(
    video: str | Path,
    *,
    edl: str | Path | None = None,
    out: str | Path | None = None,
    every_s: float = 2.0,
    columns: int = 6,
    asr: str = "auto",
    words: str | Path | None = None,
    verify: str | Path | None = None,
    backend: Backend | None = None,
) -> dict[str, Any]:
    """Measure and publish a deterministic offline pack, writing its index last."""
    if not math.isfinite(every_s) or every_s <= 0:
        raise ValidationError("every", "must be finite and greater than zero")
    if isinstance(columns, bool) or not isinstance(columns, int) or not 1 <= columns <= 12:
        raise ValidationError("columns", "must be an integer from 1 to 12")
    if asr not in ASR_CHOICES:
        raise ValidationError("asr", "unknown speech engine")
    with paths.operation():
        source = paths.validate_input(video)
        edl_path = paths.validate_input(edl) if edl is not None else None
        words_path = paths.validate_input(words) if words is not None else None
        digest = shared.take_digest(source)
        if verify is None:
            verify_path, vr_record, vr_bytes = _discover_verify(source, digest)
        else:
            verify_path = paths.validate_input(verify)
            vr_record, vr_bytes = _verify(verify_path, digest)
        verify_digest = None
        if verify_path is not None:
            try:
                verify_digest = sha256_file(verify_path)
            except OSError:
                if verify is not None:
                    raise
        inputs = [
            source,
            *(item for item in (edl_path, words_path, verify_path) if item is not None),
        ]
        directory = Path(out) if out is not None else source.with_name(f"{source.stem}-watch")
        _check_folder(directory, inputs)
        info = ffmpeg.probe(source, count_frames=True)
        if not info.has_video or not info.fps or not info.width or not info.height:
            raise InputError(str(source), "no usable video stream")
        values = []
        previous = None
        with closing(video_module.iter_grey_frames(source, w=64, rate=info.fps)) as blocks:
            for _, block in blocks:
                values.extend(m.motion_values(block, previous))
                previous = block[-1].copy()
        frames = len(values)
        if not frames:
            raise InputError(str(source), "video contains no decoded frames")
        rate = float(info.fps)
        duration = frames / rate
        cuts, segments, graphics, edl_record, warnings = _structure(source, edl_path, info, frames)
        rows = im.layout_strips(im.timeline_tiles(frames, info.fps, every_s, cuts), columns)
        sound, loudness_blocks = _sound(source, info, duration)
        for cut, levels in zip(
            cuts,
            m.cut_loudness(
                loudness_blocks,
                [cut["frame"] / rate for cut in cuts],
                duration,
                epsilon=2 / (info.sample_rate or 48000),
            ),
            strict=True,
        ):
            cut.update(levels)
        tr_record, tr = _get_transcript(source, words_path, asr, backend, info, digest)
        for segment in segments:
            segment.update(
                m.pacing(
                    segment["start_frame"] / rate,
                    segment["end_frame"] / rate,
                    tr.words if tr is not None else None,
                    sound["silences"] if info.has_audio else None,
                )
            )
        width, height = im.tile_dimensions(info.width, info.height)
        strips = [
            {
                "image": f"strips/strip-{index:03d}.png",
                "start_s": row[0]["time_s"],
                "end_s": row[-1]["time_s"],
                "tiles": row,
            }
            for index, row in enumerate(rows, 1)
        ]
        pack = {
            "schema": PACK_SCHEMA,
            "tool": {"name": "yap-studio", "version": __version__},
            "video": {
                "path": str(source),
                "file": source.name,
                "sha256": digest,
                "bytes": source.stat().st_size,
                "width": info.width,
                "height": info.height,
                "fps": f"{info.fps.numerator}/{info.fps.denominator}",
                "frames": frames,
                "duration_s": round(duration, 3),
                "has_audio": info.has_audio,
                "video_offset_s": round(info.video_offset_s, 3),
            },
            "edl": edl_record,
            "settings": {
                "every_s": every_s,
                "columns": columns,
                "asr": asr,
                "asr_word_offset_s": round(info.audio_offset_s - info.video_offset_s, 3),
                "words": None
                if words_path is None
                else {"path": str(words_path), "sha256": sha256_file(words_path)},
                "verify": None
                if verify_path is None
                else {"path": str(verify_path), "sha256": verify_digest},
                "still_threshold": m.STILL_THRESHOLD,
                "still_min_s": m.STILL_MIN_S,
                "motion_threshold": m.MOTION_THRESHOLD,
                "motion_min_s": m.MOTION_MIN_S,
                "motion_width": 64,
                "momentary_s": m.MOMENTARY_S,
                "loudness_floor_lufs": m.LOUDNESS_FLOOR_LUFS,
                "speech_gate_lufs": m.SPEECH_GATE_LUFS,
                "relative_gate_lu": m.RELATIVE_GATE_LU,
                "cut_side_s": m.CUT_SIDE_S,
                "silence_dbfs": m.SILENCE_DBFS,
                "silence_min_s": m.SILENCE_MIN_S,
                "silence_window_s": m.SILENCE_WINDOW_S,
                "silence_sample_rate": m.SILENCE_SAMPLE_RATE,
                "loudness_epsilon_s": round(2 / (info.sample_rate or 48000), 9),
                "thumbnail_long_side": im.LONG_SIDE,
                "label_height": im.LABEL_HEIGHT,
                "gap": im.TILE_GAP,
                "max_tiles": im.MAX_TILES,
                "font_scale": im.FONT_SCALE,
                "border": im.BORDER,
                "background": im.BACKGROUND,
                "foreground": im.FOREGROUND,
                "cut_before_colour": im.BEFORE_COLOUR,
                "cut_after_colour": im.AFTER_COLOUR,
                "png_compression": im.PNG_COMPRESSION,
                "cut_pair_min_columns": 2,
                "time_base": "output",
            },
            "cuts": cuts,
            "segments": segments,
            "graphics": graphics,
            "warnings": warnings,
            "timeline": {
                "every_s": every_s,
                "columns": columns,
                "tile": {
                    "width": width,
                    "height": height,
                    "label_height": im.LABEL_HEIGHT,
                    "gap": im.TILE_GAP,
                },
                "strips": strips,
            },
            "motion": {
                "file": "motion.json",
                "still_threshold": m.STILL_THRESHOLD,
                "motion_threshold": m.MOTION_THRESHOLD,
                "per_second": m.motion_per_second(values, rate),
                "still_spans": m.motion_spans(
                    values, rate, threshold=m.STILL_THRESHOLD, minimum_s=m.STILL_MIN_S, still=True
                ),
                "motion_spans": m.motion_spans(
                    values,
                    rate,
                    threshold=m.MOTION_THRESHOLD,
                    minimum_s=m.MOTION_MIN_S,
                    still=False,
                ),
            },
            "loudness": sound,
            "transcript": tr_record,
            "verify": vr_record,
        }
        with tempfile.TemporaryDirectory(prefix="yap-watch-") as temporary:
            stage = Path(temporary)
            (stage / "strips").mkdir()
            thumbnails: dict[int, np.ndarray] = {}
            strip_index = 0
            with closing(
                im.decode_thumbnails(
                    source,
                    [tile["frame"] for row in rows for tile in row],
                    info=info,
                    width=width,
                    height=height,
                )
            ) as decoded:
                for frame, thumbnail in decoded:
                    thumbnails[frame] = thumbnail
                    del thumbnail
                    row = rows[strip_index]
                    if len(thumbnails) == len(row):
                        (stage / strips[strip_index]["image"]).write_bytes(
                            im.png_bytes(im.compose_strip(row, thumbnails))
                        )
                        thumbnails.clear()
                        strip_index += 1
            (stage / "motion.json").write_text(
                _json(
                    {
                        "schema": "yap-studio/watch-motion@1",
                        "fps": pack["video"]["fps"],
                        "frames": frames,
                        "values": [round(value, 3) for value in values],
                    }
                ),
                encoding="utf-8",
            )
            if tr is not None:
                tr_data = tr.to_dict()
                tr_data["duration_s"] = round(duration, 3)
                for word in tr_data["words"]:
                    for key in ("start", "end", "conf"):
                        if word[key] is not None:
                            word[key] = round(word[key], 3)
                (stage / "transcript.json").write_text(_json(tr_data), encoding="utf-8")
                (stage / "transcript.txt").write_text(
                    _transcript_text(tr, segments, rate, tr_record["engine"]),
                    encoding="utf-8",
                )
            if vr_bytes is not None:
                (stage / "verify.json").write_bytes(vr_bytes)
            (stage / "pack.md").write_text(_markdown(pack), encoding="utf-8")
            outputs = sorted(item for item in stage.rglob("*") if item.is_file())
            pack["outputs"] = [
                {"path": item.relative_to(stage).as_posix(), "sha256": sha256_file(item)}
                for item in outputs
            ]
            (stage / "pack.json").write_text(_json(pack), encoding="utf-8")
            _publish(stage, directory, inputs, outputs)
        # Return precisely the JSON shape, including arrays rather than colour tuples.
        return json.loads(_json(pack))


@contextmanager
def _atomic_pack_output(
    directory: Path,
    name: str,
    *,
    old_sha256: str | None,
    new_sha256: str,
    owner_receipts: tuple[Path, ...] = (),
) -> Iterator[Path]:
    """Replace a guarded file with journal ownership and a final hash recheck."""
    target = _require_hash(directory, name, old_sha256)
    paths.validate_output(target, owner_receipts=owner_receipts)
    # Root temporaries stay on the pack's filesystem and have an exact cleanup name rule.
    descriptor, temporary_name = tempfile.mkstemp(prefix=_TEMP_PREFIX, dir=directory)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        if _current_hash(directory, temporary.name) != new_sha256:
            raise _refusal("Staged pack file changed during publication.")
        _require_hash(directory, name, old_sha256)
        paths.validate_output(target, owner_receipts=owner_receipts)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, target)
    finally:
        _pack_file(directory, temporary.name).unlink(missing_ok=True)


def _delete_pack_file(directory: Path, name: str, digest: str | None, journal: Path) -> None:
    """Delete only a regular file whose current hash is owned by the journal."""
    target = _require_hash(directory, name, digest)
    paths.validate_output(target, owner_receipts=(journal,))
    _require_hash(directory, name, digest).unlink(missing_ok=True)


def _publish(stage: Path, directory: Path, inputs: list[Path], outputs: list[Path]) -> None:
    """Journal all replacements and deletions, then publish the index last."""
    old = _check_folder(directory, inputs)
    sources = {_pack_name(item.relative_to(stage).as_posix()): item for item in outputs}
    sources["pack.json"] = stage / "pack.json"
    names = sorted({*sources, *(item.relative_to(directory).as_posix() for item in old)})
    entries = [
        {
            "path": name,
            "sha256": _current_hash(directory, name),
            "new_sha256": sha256_file(sources[name]) if name in sources else None,
        }
        for name in names
    ]
    for entry in entries:
        entry["old_sha256"] = entry["sha256"]
    journal = directory / "pack.pending.json"
    journal_old = _current_hash(directory, journal.name)
    paths.validate_output(journal)
    paths.validate_output_dir(directory)
    text = _json({"schema": PENDING_SCHEMA, "outputs": entries})
    journal_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    with _atomic_pack_output(
        directory, journal.name, old_sha256=journal_old, new_sha256=journal_digest
    ) as temporary:
        temporary.write_text(text, encoding="utf-8")
    owners = (journal,)
    by_name = {entry["path"]: entry for entry in entries}
    for entry in entries:
        if _TEMP_NAME.fullmatch(entry["path"]):
            _delete_pack_file(directory, entry["path"], entry["old_sha256"], journal)
    index_entry = by_name["pack.json"]
    _delete_pack_file(directory, "pack.json", index_entry["old_sha256"], journal)
    for name, source in sources.items():
        if name == "pack.json":
            continue
        entry = by_name[name]
        with _atomic_pack_output(
            directory, name, old_sha256=entry["old_sha256"],
            new_sha256=entry["new_sha256"], owner_receipts=owners,
        ) as temporary:
            shutil.copyfile(source, temporary)
    for entry in entries:
        if entry["new_sha256"] is None and not _TEMP_NAME.fullmatch(entry["path"]):
            _delete_pack_file(directory, entry["path"], entry["old_sha256"], journal)
    for name in sources:
        if name != "pack.json":
            _require_hash(directory, name, by_name[name]["new_sha256"])
    with _atomic_pack_output(
        directory, "pack.json", old_sha256=None,
        new_sha256=index_entry["new_sha256"], owner_receipts=owners,
    ) as temporary:
        shutil.copyfile(stage / "pack.json", temporary)
    _require_hash(directory, journal.name, journal_digest).unlink()
