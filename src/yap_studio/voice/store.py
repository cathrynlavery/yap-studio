"""Review decisions and the exact projection consumed by cuts."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from yap_studio import audio, paths
from yap_studio.checks.model import Finding
from yap_studio.project.lock import project_lock, timestamp
from yap_studio.stages import take_levels
from yap_studio.voice.consent import require_consent
from yap_studio.voice.model import (
    VoiceError,
    checked_file,
    read,
    require_generation,
    require_idle_stages,
    write,
)
from yap_studio.voice.plan import (
    list_fixes,
    occurrences,
    patch_range,
    require_picture,
    source_sample,
    validate_words,
)
from yap_studio.voice.splice import (
    _speech,
    blend,
    checks_status,
    decode,
    edge_check,
    level,
    room_tone_check,
    write_wav,
)


def approved_fixes(project_dir: Path) -> list[dict]:
    """Return applicable fixes, refusing every missing or changed approved patch."""
    root = Path(project_dir)
    data = list_fixes(root)
    consent = read(root, "consent", optional=True)
    if consent is None and any(fix["status"] == "approved"
                               and fix.get("from", {}).get("kind") != "own"
                               for fix in data["fixes"]):
        raise VoiceError(
            "The consent record for approved voice fixes is missing.", code="voice_consent_required"
        )
    result = [fix for fix in data["fixes"] if fix["status"] == "approved"
              and (fix.get("from", {}).get("kind") == "own"
                   or consent and consent["status"] == "given")]
    for fix in result:
        checked_file(root, fix["patch"])
    return result


def projection(project_dir: Path, *, fixes: list[dict] | None = None) -> dict:
    root = Path(project_dir)
    consent = read(root, "consent", optional=True)
    return {
        "consent": consent["status"] if consent else None,
        "approved": [
            {
                key: fix[key]
                for key in ("id", "take", "words", "say", "said", "source", "patch", "checks")
            } | {"kind": fix.get("from", {}).get("kind", "clone")}
            for fix in (approved_fixes(root) if fixes is None else fixes)
        ],
    }


def _require_no_overlap(fix: dict, fixes: list[dict]) -> None:
    patch = fix["patch"]
    for other in fixes:
        if (
            other["id"] != fix["id"]
            and other["status"] == "approved"
            and other["take"] == fix["take"]
            and other["patch"]["start_sample"] < patch["start_sample"] + patch["samples"]
            and other["patch"]["start_sample"] + other["patch"]["samples"] > patch["start_sample"]
        ):
            raise VoiceError(
                f"This patch overlaps voice fix {other['id']}. Reject {other['id']} first.",
                code="voice_fix_overlap",
            )


def decide(
    project_dir: Path, fix_id: str, action: str, *, via: str, generation: str | None = None,
) -> dict:
    """Record a review or chat decision without generating audio."""
    root = Path(project_dir)
    if action not in ("approve", "reject") or via not in ("review", "chat"):
        raise VoiceError(
            "Use approve or reject, via review or chat.", code="voice_decision_invalid"
        )
    with paths.operation(), project_lock(root):
        require_idle_stages(root)
        data = list_fixes(root)
        fix = next((fix for fix in data["fixes"] if fix["id"] == fix_id), None)
        if fix is None:
            raise VoiceError(f"Voice fix {fix_id} was not found.", code="voice_fix_missing")
        if generation is not None:
            require_generation(fix, generation)
        if action == "approve":
            if fix.get("from", {}).get("kind") != "own":
                require_consent(root)
            if fix["status"] not in ("ready", "approved", "rejected") or "patch" not in fix:
                raise VoiceError(
                    "Make the fix and listen before approving it.", code="voice_fix_not_ready"
                )
            if fix["checks"]["status"] == "fail":
                raise VoiceError(
                    "This fix failed its splice checks. Make it again before approving.",
                    code="voice_checks_failed",
                )
            checked_file(root, fix["patch"])
            for clip in fix["clips"].values():
                checked_file(root, clip)
            _require_no_overlap(fix, data["fixes"])
        fix["status"] = "approved" if action == "approve" else "rejected"
        fix["decision"] = {"action": action, "at": timestamp(), "via": via}
        write(root, "fixes", data)
        return fix


def patches_for_plan(plan, *, fixes: list[dict] | None = None) -> list[tuple[dict, np.ndarray]]:
    """Validate full retained slots before a cut writes any output."""
    result = []
    if fixes is None:
        fixes = approved_fixes(plan.context.project_dir)
    for fix in fixes:
        if fix["take"] not in plan.media:
            continue
        first, last = patch_range(plan, fix)
        if not any(piece.take == fix["take"] and float(piece.src_t0) < last
                   and float(piece.src_t1) > first for piece in plan.pieces):
            continue
        _require_no_overlap(fix, fixes)
        validate_words(plan, fix)
        meta = fix["patch"]
        occurrences(plan, fix, start_s=first, end_s=last)
        sound = decode(checked_file(plan.context.project_dir, meta), meta["channels"])
        if len(sound) != meta["samples"]:
            raise VoiceError(f"Voice fix {fix['id']} has a wrong patch length.")
        result.append((fix, sound))
    return result


def apply_piece(
    sound: np.ndarray, piece, patches: list[tuple[dict, np.ndarray]], plan
) -> np.ndarray:
    for fix, patch in patches:
        if fix["take"] == piece.take:
            sound = blend(
                sound,
                patch,
                fix["patch"],
                source_start_sample=source_sample(plan, piece.take, piece.src_t0),
            )
    return sound


def require_piece_headroom(
    sound: np.ndarray, piece, patches: list[tuple[dict, np.ndarray]], plan, gain_db: float,
) -> None:
    """Refuse patched samples that exceed the ceiling under the take's fixed gain."""
    source_start = source_sample(plan, piece.take, piece.src_t0)
    ceiling = 10 ** ((take_levels.PEAK_CEILING_DBFS - gain_db) / 20)
    for fix, _ in patches:
        if fix["take"] != piece.take:
            continue
        meta = fix["patch"]
        first = meta["start_sample"] - source_start
        low, high = max(0, first), min(len(sound), first + meta["samples"])
        if high > low and float(np.max(np.abs(sound[low:high]))) > ceiling:
            raise VoiceError(
                f"Voice fix {fix['id']} would clip after this take's level match. "
                "Make it again with --new or keep the original.",
                code="voice_fix_peak",
            )


def applied_records(plan, patches: list[tuple[dict, np.ndarray]]) -> list[dict]:
    rows = []
    for fix, _ in patches:
        meta = fix["patch"]
        first, last = patch_range(plan, fix)
        occurrences(plan, fix, start_s=first, end_s=last)
        for piece in plan.pieces:
            if (
                piece.take == fix["take"]
                and float(piece.src_t0) <= first + 1e-8
                and float(piece.src_t1) >= last - 1e-8
            ):
                start = (
                    piece.sample_start
                    + meta["start_sample"]
                    - source_sample(plan, piece.take, piece.src_t0)
                )
                rows.append(
                    {"id": fix["id"], "kind": fix.get("from", {}).get("kind", "clone"),
                     "start_sample": start, "end_sample": start + meta["samples"]}
                )
    return rows


def replacement_words(fix: dict) -> dict[str, str]:
    """Map source ids to new text, leaving merged ids as empty aliases."""
    parts = fix["say"].split()
    if len(parts) == len(fix["words"]):
        return dict(zip(fix["words"], parts, strict=True))
    return {word: fix["say"] if index == 0 else "" for index, word in enumerate(fix["words"])}


def replaced_text(rows: list[dict], fixes: list[dict]) -> list[dict]:
    """Project changed phrases onto word occurrences, retaining every anchor id."""
    result = copy.deepcopy(rows)
    for fix in fixes:
        replacements = replacement_words(fix)
        for segment in {row["segment"] for row in result if row["id"] in fix["words"]}:
            selected = [
                row for row in result if row["segment"] == segment and row["id"] in fix["words"]
            ]
            if len(fix["say"].split()) != len(selected):
                selected[0]["end"] = selected[-1]["end"]
            for row in selected:
                row["text"] = replacements[row["id"]]
    return result


def source_transform(plan, patches, *, video_offsets=None):
    """Patch the verifier's container-clock decode at its requested sample rate."""
    cache = {}

    def transform(take, sound, rate):
        result = sound[:, None].copy()
        for fix, _ in patches:
            if fix["take"] != take:
                continue
            meta = fix["patch"]
            count = round(meta["samples"] * rate / 48000)
            key = (fix["id"], rate)
            if key not in cache:
                _, mono = audio.decode(
                    checked_file(plan.context.project_dir, meta),
                    sr=rate, dur_s=count / rate,
                )
                signal = np.zeros((count, 1), dtype=np.float32)
                available = min(count, len(mono))
                signal[:available, 0] = mono[:available]
                cache[key] = signal
            signal = cache[key]
            offset = (video_offsets[take] - plan.media[take].video_offset_s
                      if video_offsets is not None and take in video_offsets else 0)
            # blend uses only sample counts, so it works on this decode's rate too.
            scaled = {
                **meta,
                "samples": count,
                "start_sample": round((meta["start_sample"] / 48000 + offset) * rate),
                "fade_in_samples": round(meta["fade_in_samples"] * rate / 48000),
                "fade_out_samples": round(meta["fade_out_samples"] * rate / 48000),
            }
            end = scaled["start_sample"] + count
            if end > len(result):
                result = np.pad(result, ((0, end - len(result)), (0, 0)))
            result = blend(result, signal, scaled, source_start_sample=0)
        return result[:, 0]

    return transform


def require_assembly_picture(
    plan, elements: list[dict], graphic_coverage: dict[str, set[int]], *,
    fixes: list[dict] | None = None,
) -> None:
    if fixes is None:
        fixes = [fix for fix, _ in patches_for_plan(plan)]
    for fix in fixes:
        try:
            first, last = patch_range(plan, fix)
            require_picture(
                plan,
                fix,
                elements=elements,
                graphic_coverage=graphic_coverage,
                start_s=first,
                end_s=last,
            )
        except VoiceError as exc:
            raise VoiceError(f"Voice fix {fix['id']}: {exc}", code=exc.code) from None


def add_cut_checks(
    report: dict,
    sound: np.ndarray,
    applied: list[dict],
    fixes: list[dict],
    *,
    plan,
    work: Path,
    levels: dict | None,
) -> dict:
    """Run splice checks again on the cut's delivered PCM."""
    from yap_studio.voice.make import _heard

    for row in applied:
        fix = next(fix for fix in fixes if fix["id"] == row["id"])
        items = copy.deepcopy(fix["checks"]["items"])
        first, last = row["start_sample"], row["end_sample"]
        signal = sound[first:last]
        speech = _speech(signal)
        if speech.shape[1] == 2 and fix.get("patch", {}).get("channels") == 1:
            speech = speech.mean(axis=1, keepdims=True)
        gain = levels["takes"][fix["take"]]["gain_db"] if levels else 0.0
        for item in items:
            if item["id"] == "splice.level":
                lufs, _ = take_levels.measure(speech, work, f"cut-{fix['id']}-{first}")
                reference = item["reference_lufs"]
                difference = (
                    abs(lufs - reference - gain)
                    if lufs is not None and reference is not None
                    else abs(level(speech) - item["reference_rms_db"] - gain)
                )
                item.update(
                    status="pass" if difference <= 1.5 else "warn", difference_lu=difference
                )
            elif item["id"] == "splice.room_tone":
                floor = item["source_floor_db"] + gain
                speech_spans = item.get("speech_spans")
                item.clear()
                item.update(room_tone_check(signal, floor, speech_spans=speech_spans))
            elif item["id"] == "splice.pace":
                item["status"] = "warn" if item["gap_s"] > 0.25 else "pass"
        after = work / f"voice-check-{fix['id']}-{first}.wav"
        write_wav(after, signal)
        click = next(item for item in items if item["id"] == "splice.click")
        internal = [first + edge["sample"] - click["edges"][0]["sample"]
                    for edge in click["edges"][2:]]
        items = [
            edge_check(sound, first, last, *internal),
            *[item for item in items if item["id"] not in ("splice.click", "splice.heard")],
            _heard(plan, after, fix["say"], respelled=bool(
                fix.get("from", {}).get("pronunciations")
                and not fix.get("generation", {}).get("plain", False))),
        ]
        report.setdefault("voice_fixes", []).append(
            {**row, "checks": {"status": checks_status(items), "items": items}}
        )
        for item in items:
            previous = report.setdefault("checks", {}).get(item["id"], {})
            rank = {"skipped": -1, "pass": 0, "warn": 1, "fail": 2}
            if rank[item["status"]] >= rank[previous.get("status", "skipped")]:
                report["checks"][item["id"]] = {
                    "ran": item["status"] != "skipped",
                    "status": item["status"],
                    "reason": item.get("reason"),
                }
            if item["status"] in ("warn", "fail"):
                report["findings"].append(
                    Finding(
                        item["id"],
                        item["status"],
                        f"Voice fix {row['id']}: {item['id']} needs a listen.",
                        time_s=first / 48000,
                        evidence=item,
                    ).to_dict()
                )
    statuses = {row["severity"] for row in report["findings"]}
    report["status"] = (
        "fail"
        if "fail" in statuses or report["status"] == "fail"
        else "warn"
        if "warn" in statuses or report["status"] == "warn"
        else "pass"
    )
    if "summary" in report:
        report["summary"]["status"] = report["status"]
        for severity in ("fail", "warn", "info"):
            if severity in report["summary"]:
                report["summary"][severity] = sum(
                    row["severity"] == severity for row in report["findings"]
                )
        if "text" in report["summary"] and "media" in report:
            from yap_studio.report import summarize

            report["summary"]["text"] = summarize(report)
    return report
