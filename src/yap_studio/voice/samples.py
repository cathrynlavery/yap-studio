"""Select clean retained speech for an own-voice clone."""

from __future__ import annotations

import secrets
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yap_studio import audio, ffmpeg, paths, receipt
from yap_studio.project.lock import project_lock
from yap_studio.stages.cut import (
    FADE_SAMPLES,
    SAMPLE_RATE,
    CutPlan,
    Piece,
    _audio_piece,
    resolve_cut,
)
from yap_studio.voice.model import VoiceError, local_path, read, write
from yap_studio.voice.plan import list_fixes


@dataclass(frozen=True)
class _Candidate:
    confidence: float
    take: str
    ranges: list[list[float]]
    sound: np.ndarray

    @property
    def duration_s(self) -> float:
        return len(self.sound) / SAMPLE_RATE


def _runs(plan: CutPlan, fixes: list[dict]) -> list[list[Piece]]:
    """Keep consecutive pieces together, breaking at excluded speech or a new take."""
    runs: list[list[Piece]] = []
    current: list[Piece] = []
    for piece in plan.pieces:
        excluded = any(
            fix["take"] == piece.take and fix["status"] != "rejected"
            and fix["source"]["start_s"] - 0.1 < float(piece.src_t1)
            and fix["source"]["end_s"] + 0.1 > float(piece.src_t0)
            for fix in fixes
        )
        if current and (excluded or piece.take != current[-1].take
                        or piece.src_t0 < current[-1].src_t1):
            runs.append(current)
            current = []
        if not excluded:
            current.append(piece)
    if current:
        runs.append(current)
    return runs


def _joined(pieces: list[Piece], plan: CutPlan) -> np.ndarray:
    sounds = [_audio_piece(piece, plan) for piece in pieces]
    offsets = np.cumsum([0, *[len(sound) for sound in sounds]])
    sound, _ = audio.splice(
        np.concatenate(sounds), list(zip(offsets[:-1], offsets[1:], strict=True)), FADE_SAMPLES,
    )
    return sound


def _windows(
    pieces: list[Piece], sound: np.ndarray, plan: CutPlan, *, short: bool = False,
) -> list[_Candidate]:
    """Split joined speech evenly and retain the source ranges behind each window."""
    if len(sound) < 15 * SAMPLE_RATE:
        return []
    count = (len(sound) + 30 * SAMPLE_RATE - 1) // (30 * SAMPLE_RATE)
    if short:
        count = len(sound) // (15 * SAMPLE_RATE)
    words = {word.id: word for word in plan.source_words[pieces[0].take].words}
    candidates = []
    for index in range(count):
        first, last = index * len(sound) // count, (index + 1) * len(sound) // count
        clip = sound[first:last]
        if np.max(np.abs(clip), initial=0) >= 0.999 or not np.any(clip):
            continue
        ranges, kept = [], []
        cursor = 0
        for piece in pieces:
            end = cursor + piece.sample_end - piece.sample_start
            left, right = max(first, cursor), min(last, end)
            if left < right:
                a = float(piece.src_t0) + (left - cursor) / SAMPLE_RATE
                b = float(piece.src_t0) + (right - cursor) / SAMPLE_RATE
                ranges.append([a, b])
                kept.extend(words[word] for word in piece.words
                            if a <= words[word].start and words[word].end <= b)
            cursor = end
        if kept:
            confidence = sum(word.conf or 0 for word in kept) / len(kept)
            candidates.append(_Candidate(confidence, pieces[0].take, ranges, clip))
    return candidates


def _choose(candidates: list[_Candidate]) -> list[_Candidate]:
    chosen: list[_Candidate] = []
    total = 0
    for candidate in sorted(candidates, key=lambda row: (-row.confidence, row.take, row.ranges)):
        if any(
            row.take == candidate.take
            and any(a < d and b > c for a, b in row.ranges for c, d in candidate.ranges)
            for row in chosen
        ):
            continue
        if total + len(candidate.sound) > 150 * SAMPLE_RATE:
            continue
        chosen.append(candidate)
        total += len(candidate.sound)
        if len(chosen) >= 4 and total >= 120 * SAMPLE_RATE or len(chosen) == 6:
            break
    return chosen


def _samples_in_use(consent: dict | None) -> bool:
    return bool(consent and consent["status"] == "given"
                and (consent["samples"] or consent["voice"].get("id_sha256")))


def samples(project_dir: Path) -> dict:
    """Export 4 to 6 clean windows of 15 to 30 seconds from kept words."""
    root = Path(project_dir)
    with paths.operation():
        consent = read(root, "consent", optional=True)
        if _samples_in_use(consent):
            raise VoiceError(
                "Withdraw consent before changing the voice samples.", code="voice_samples_in_use"
            )
        plan = resolve_cut(root)
        fixes = list_fixes(root)["fixes"]
        runs = [(pieces, _joined(pieces, plan)) for pieces in _runs(plan, fixes)]
        chosen = _choose([row for pieces, sound in runs for row in _windows(pieces, sound, plan)])
        if len(chosen) < 4:
            chosen = _choose([
                row for pieces, sound in runs for row in _windows(pieces, sound, plan, short=True)
            ])
        if len(chosen) < 4 or sum(row.duration_s for row in chosen) < 45:
            raise VoiceError(
                "There is not enough clean kept speech for 4 voice samples.",
                code="voice_samples_too_short",
                suggested_action="Keep at least 60 seconds of clean speech and try again.",
            )
        records = []
        work_root = paths.validate_output_dir(root / ".yap/work")
        with tempfile.TemporaryDirectory(prefix="voice-samples-", dir=work_root) as folder:
            work = Path(folder)
            for index, candidate in enumerate(chosen, 1):
                sound = candidate.sound
                raw, target = work / "input.f32le", work / f"s{index:02}.wav"
                raw.write_bytes(sound.astype("<f4").tobytes())
                ffmpeg.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-f",
                        "f32le",
                        "-ar",
                        "48000",
                        "-ac",
                        str(sound.shape[1]),
                        "-i",
                        str(raw),
                        "-ac",
                        "1",
                        "-ar",
                        "44100",
                        "-c:a",
                        "pcm_s16le",
                        str(target),
                    ],
                    timeout_s=120,
                )
                records.append(
                    {
                        "path": f"voice/samples/{target.name}",
                        "take": candidate.take,
                        "start_s": candidate.ranges[0][0],
                        "end_s": candidate.ranges[-1][1],
                        "duration_s": candidate.duration_s,
                        "ranges": candidate.ranges,
                        "sha256": receipt.sha256_file(target),
                        "size_bytes": target.stat().st_size,
                    }
                )
            data = {"schema": "yap-studio/voice-samples@1", "samples": records}
            with project_lock(root):
                current = read(root, "consent", optional=True)
                if _samples_in_use(current):
                    raise VoiceError("Voice consent changed while choosing samples.",
                                     code="voice_samples_in_use")
                previous = read(root, "samples", optional=True) or {"samples": []}
                owners = [root / "voice/samples.json"]
                old = {Path(row["path"]).name: row for row in previous["samples"]}
                for row in records:
                    destination = old.get(Path(row["path"]).name, row)
                    paths.validate_output(local_path(root, destination["path"]),
                                          owner_receipts=owners)
                paths.validate_output(root / "voice/samples.json")
                relative = f"voice/samples/g{secrets.token_hex(8)}"
                records = [{**row, "path": f"{relative}/{Path(row['path']).name}"}
                           for row in records]
                data = {**data, "samples": records}
                folder = local_path(root, relative)
                paths.validate_output_dir(folder.parent)
                folder.mkdir()
                published = False
                written = []
                try:
                    for row in records:
                        paths.validate_output(local_path(root, row["path"]), owner_receipts=owners)
                    for row in records:
                        target = local_path(root, row["path"])
                        with paths.atomic_output(target, owner_receipts=owners) as staged:
                            shutil.copyfile(work / target.name, staged)
                        written.append(row)
                    write(root, "samples", data)
                    published = True
                finally:
                    if not published:
                        for row in written:
                            target = local_path(root, row["path"])
                            if target.is_file() and receipt.sha256_file(target) == row["sha256"]:
                                target.unlink()
                        try:
                            folder.rmdir()
                        except OSError:
                            pass
                notices = []
                retained = {row["path"] for row in records}
                for row in previous["samples"]:
                    if row["path"] in retained:
                        continue
                    target = local_path(root, row["path"])
                    if not target.exists():
                        continue
                    if target.is_file() and receipt.sha256_file(target) == row["sha256"]:
                        target.unlink()
                    else:
                        notices.append(f"Left {row['path']} because it changed.")
                if notices:
                    data = {**data, "notices": notices}
        return data
