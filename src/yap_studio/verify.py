"""End-to-end verification orchestration."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from yap_studio import ffmpeg, paths, receipt
from yap_studio.checks.caption_contrast import (
    CONTRAST_SUFFIX,
    check_caption_contrast,
    read_contrast_record,
)
from yap_studio.checks.caption_layout import (
    LAYOUT_SUFFIX,
    caption_coverage,
    check_caption_cards,
    check_caption_timing,
    read_layout_record,
)
from yap_studio.checks.caption_words import check_caption_words, heard_transcript
from yap_studio.checks.declared import check_declared_joins
from yap_studio.checks.detect import check_without_edl
from yap_studio.checks.face_clearance import FACES_SUFFIX, check_face_clearance, final_faces
from yap_studio.checks.grade import check_grade
from yap_studio.checks.joins import check_joins
from yap_studio.checks.model import CHECK_IDS, Finding, Thresholds, TransitionWindow
from yap_studio.checks.platform import PLATFORMS, check_platform, load_presets
from yap_studio.checks.structure import StructureResult, check_structure
from yap_studio.checks.sync import check_sync
from yap_studio.edl import _apply_overrides, load_edit_list
from yap_studio.errors import InputError, ValidationError
from yap_studio.grade import GradeSpec
from yap_studio.plan import Plan
from yap_studio.report import build_report
from yap_studio.transcript import (
    resolve_backend,
    resolve_second_backend,
    second_engine_unmeasured_reason,
)

_LOGGER = logging.getLogger("yap_studio")
_NO_ENGINE_REASON = "no speech engine installed; pip install 'yap-studio[asr]'"
_NO_AUDIO_REASON = "no audio in the render"
_WHOLE_FILE_LOUDNESS_CHECKS = {"loudness.integrated", "loudness.true_peak"}
_AUDIO_CHECKS = (
    "join.click",
    "join.gap",
    "join.words",
    "join.listen",
    "sync.av_offset",
    "sync.drift",
    "plan.samples",
    "structure.av_length",
    "structure.audio_timestamps",
    "loudness.integrated",
    "loudness.true_peak",
    "loudness.parts",
)


def _absolute_overrides(overrides: list[str] | None) -> list[str]:
    result = []
    for override in overrides or []:
        if "=" in override:
            source_id, raw_path = override.split("=", 1)
            prefix = f"{source_id}="
        else:
            raw_path = override
            prefix = ""
        if not raw_path:
            result.append(prefix)
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        result.append(prefix + str(path.resolve()))
    return result


def _edit_list_input(edl: str) -> tuple[Path, Path]:
    requested = Path(edl)
    if requested.suffix == ".fcpxmld" and requested.is_dir():
        info = paths.validate_input(requested / "Info.fcpxml")
        return info.parent, info
    source = paths.validate_input(requested)
    return source, source


def _edit_coverage(
    info, sync, joins, backend, plan=None, *, second_backend=None, click_source="mix",
) -> dict[str, dict[str, object]]:
    coverage = {check: {"ran": True, "reason": None} for check in CHECK_IDS}
    click_rows = [cut for cut in joins.cuts if getattr(cut, "click_db", None) is not None]
    if click_rows:
        sources = {cut.click_source or click_source for cut in click_rows}
        coverage["join.click"]["click_source"] = sources.pop() if len(sources) == 1 else "mixed"
        coverage["join.click"]["evidence"] = [
            {"cut": cut.index, "click_source": cut.click_source or click_source}
            for cut in click_rows
        ]
    else:
        coverage["join.click"]["click_source"] = click_source
    coverage["detect.cuts"] = {"ran": False, "reason": "edit list supplied"}
    if plan is None or not plan.holds:
        coverage["plan.holds"] = {"ran": False, "reason": "no holds in the edit list"}
    for check in CHECK_IDS:
        if check.startswith("platform."):
            coverage[check] = {"ran": False, "reason": "no platform chosen"}
    word_rows = [cut.words for cut in joins.cuts]
    if any(row.status in {"pass", "fail"} for row in word_rows):
        if backend is None:
            coverage["join.words"]["note"] = f"Words not named: {_NO_ENGINE_REASON}"
    else:
        reasons = {row.reason for row in word_rows if row.reason}
        reason = next(iter(reasons)) if len(reasons) == 1 else "no speech near the cuts"
        coverage["join.words"] = {
            "ran": False,
            "reason": reason or "no speech near the cuts",
        }
    for check, status_field, reason_field in (
        ("join.relisten", "status", "reason"),
        ("join.second_model", "second_status", "second_reason"),
    ):
        if backend is None:
            coverage[check] = {"ran": False, "reason": _NO_ENGINE_REASON}
        elif check == "join.second_model" and second_backend is None:
            coverage[check] = {
                "ran": False, "reason": second_engine_unmeasured_reason(backend.name),
            }
        elif not any(
            cut.relisten is not None and getattr(cut.relisten, status_field) in {"pass", "warn"}
            for cut in joins.cuts
        ):
            reasons = {
                getattr(cut.relisten, reason_field)
                if cut.relisten is not None else cut.words.reason
                for cut in joins.cuts
            } - {None}
            coverage[check] = {
                "ran": False,
                "reason": next(iter(reasons)) if len(reasons) == 1 else "no speech near the cuts",
            }
    frame_rows = [cut.frames for cut in joins.cuts]
    if frame_rows and not any(row.status in {"pass", "fail"} for row in frame_rows):
        reasons = {row.reason for row in frame_rows if row.reason}
        coverage["join.frames"] = {
            "ran": False,
            "reason": next(iter(reasons)) if len(reasons) == 1 else "source picture unavailable",
        }
    flashes = [cut.flash for cut in joins.cuts]
    if flashes and not any(flash is not None and not flash.get("reason") for flash in flashes):
        reasons = {flash["reason"] for flash in flashes if flash and flash.get("reason")}
        coverage["picture.flash"] = {
            "ran": False,
            "reason": next(iter(reasons)) if len(reasons) == 1 else "flash windows unavailable",
        }
    for check in ("join.frames", "picture.flash"):
        covered = [
            {"cut": cut.index, "status": "covered by transition",
             **cut.frames.distances["transition"]}
            for cut in joins.cuts if cut.frames.reason == "covered by transition"
        ]
        if covered:
            coverage[check]["evidence"] = covered
            coverage[check]["note"] = "Covered by transition: " + ", ".join(
                f"cut {row['cut']}, {row['id']} [{row['start_frame']}, {row['end_frame']})"
                for row in covered
            )
    transitions = getattr(joins, "transitions", [])
    skipped_transitions = [row for row in transitions if row["status"] == "skipped"]
    measured_transitions = [row for row in transitions if row["status"] in {"pass", "fail"}]
    transition_reason = None
    if skipped_transitions:
        prefix = "partial transition coverage" if measured_transitions else "transitions skipped"
        transition_reason = prefix + ": " + "; ".join(
            f"{row['id']}: {row['reason']}" for row in skipped_transitions
        )
    transition_ran = bool(measured_transitions) and (
        not skipped_transitions or any(row["status"] == "fail" for row in measured_transitions)
    )
    coverage["picture.transition"] = {
        "ran": transition_ran,
        "reason": None if transition_ran else transition_reason or "no declared transitions",
        **({"note": transition_reason} if measured_transitions and skipped_transitions else {}),
        **({"evidence": transitions} if transitions else {}),
    }
    audio_measured = [row for row in sync.segments if row.av_offset_ms is not None]
    audio_eligible = [row for row in audio_measured if row.confidence == "high"]
    picture_measured = [row for row in sync.segments if row.video_frame is not None]
    if not audio_measured:
        coverage["sync.av_offset"] = {"ran": False, "reason": "sync could not be measured"}
        coverage["sync.drift"] = {"ran": False, "reason": "sync could not be measured"}
    elif not audio_eligible:
        reason = "sync measurements were not confident enough"
        coverage["sync.av_offset"] = {"ran": False, "reason": reason}
        coverage["sync.drift"] = {"ran": False, "reason": reason}
    elif len(audio_eligible) < 2:
        coverage["sync.drift"] = {
            "ran": False,
            "reason": "needs two confident sync measurements",
        }
    if not picture_measured:
        coverage["plan.picture"] = {"ran": False, "reason": "source picture unavailable"}
    if joins.cuts and not any(cut.gap_ms is not None for cut in joins.cuts):
        coverage["join.gap"] = {
            "ran": False,
            "reason": "audio landings could not be measured",
        }
    if not info.has_audio:
        for check in _AUDIO_CHECKS:
            coverage[check] = {"ran": False, "reason": _NO_AUDIO_REASON}
    if not joins.cuts:
        for check in _JOIN_CHECKS:
            coverage[check] = {"ran": False, "reason": _NO_CUTS_REASON}
    if plan is not None and info.has_audio:
        intent = any(segment.first_word is not None or segment.last_word is not None
                     for segment in plan.segments)
        outer_intent = bool(plan.segments and (
            plan.segments[0].first_word is not None or plan.segments[-1].last_word is not None
        ))
        words = coverage["join.words"]
        if backend is None and intent:
            note = f"planned boundary words not checked: {_NO_ENGINE_REASON}"
            if words["ran"]:
                words["note"] = f"{words['note']}; {note}" if words.get("note") else note
            else:
                coverage["join.words"] = {"ran": False, "reason": note}
        elif backend is not None and outer_intent:
            words.update(ran=True, reason=None)
            if any(finding.check == "join.listen" and finding.cut is None
                   for finding in joins.findings):
                coverage["join.listen"] = {"ran": True, "reason": None}
    second_run = getattr(second_backend, "last_run", None)
    if second_run and second_run["used_result"] and second_run["returncode"]:
        returncode = second_run["returncode"]
        ending = f"exit status {returncode}"
        if returncode < 0:
            ending = f"signal {-returncode}"
            if second_run["signal"]:
                ending += f" ({second_run['signal']})"
        coverage["join.second_model"]["note"] = (
            f"{second_backend.name} exited with {ending} after writing a complete result; "
            "the result was used"
        )
    return coverage


def _declared_coverage(coverage, edit_list, declared, backend, info) -> None:
    """Record declared joins under join.words, which they check by transcript."""
    if not edit_list.joins or not info.has_audio:
        # Without audio, join.words is already skipped with the other sound checks.
        return
    count = len(edit_list.joins)
    noun = "join" if count == 1 else "joins"
    words = coverage["join.words"]
    if declared is not None:
        note = (
            f"{count} declared {noun} checked by {'its' if count == 1 else 'their'} words "
            "(source: declared)."
        )
        if words.get("ran"):
            words["note"] = f"{words['note']} {note}" if words.get("note") else note
        else:
            coverage["join.words"] = {"ran": True, "reason": None, "note": note}
        return
    if words.get("ran"):
        note = f"Declared {noun} not checked: {_NO_ENGINE_REASON}"
        words["note"] = f"{words['note']} {note}" if words.get("note") else note
    else:
        coverage["join.words"] = {
            "ran": False,
            "reason": f"declared {noun} not checked: {_NO_ENGINE_REASON}",
        }


_NO_CUTS_REASON = "no cuts in the edit list"
_JOIN_CHECKS = (
    "join.click", "join.gap", "join.words", "join.frames", "join.listen",
    "join.relisten", "join.second_model", "picture.flash",
)
_COVERAGE_LABELS = {
    "checked": "checked",
    "partial": "partial",
    "unavailable": "unavailable",
    "none": "none",
    "not_exposed": "not exposed",
    "declared": "declared",
    "attested": "attested",
}
ATTESTED_LABEL = "checked outside Yap Studio"


def _attest(item) -> dict[str, object]:
    """Hash one attested evidence file; a match is recorded, never counted as a pass."""
    found = item.path.is_file()
    actual = receipt.sha256_file(paths.validate_input(item.path)) if found else None
    return {
        "path": str(item.path),
        "sha256": item.sha256,
        "actual_sha256": actual,
        "found": found,
        "match": (actual == item.sha256) if found else None,
        "label": ATTESTED_LABEL,
    }


def _is_finished_render(plan, info, source_paths) -> bool:
    """Return whether one range copies a finished render of the same length."""
    if len(plan.segments) != 1 or info.video_frames is None:
        return False
    segment = plan.segments[0]
    covered = segment.n_frames + sum(hold.n_frames for hold in plan.holds)
    if segment.out_f0 != 0 or segment.src_f0 != 0 or abs(covered - info.video_frames) > 1:
        return False
    source = source_paths.get(segment.source_id)
    if source is None or not Path(source).is_file():
        return False
    try:
        source_info = ffmpeg.probe(source)
    except InputError:
        return False
    return (
        source_info.video_frames is not None
        and abs(source_info.video_frames - info.video_frames) <= 1
    )


def _compared(cut) -> bool:
    """Return whether a planned cut was actually compared with its source."""
    return cut.frames.status in {"pass", "fail"} or cut.words.status in {"pass", "fail"}


# A declared join covers a planned cut when it sits within this many frames of it.
_DECLARED_SLACK_FRAMES = 1


def _declared_reason(declared) -> str:
    failed = sum(row.status == "fail" for row in declared.joins)
    count = len(declared.joins)
    return (
        f"{count} {'join' if count == 1 else 'joins'} declared in the edit list "
        f"{'was' if count == 1 else 'were'} checked by "
        f"{'its' if count == 1 else 'their'} words"
        + (f"; {failed} failed" if failed else "")
    )


def _cut_coverage(edit_list, plan, info, source_paths, declared, joins) -> dict[str, object]:
    """Say how the render's cuts were covered, by the weakest coverage of any planned cut.

    A planned cut counts as checked only when its frames or its words were compared
    with the source, so cuts whose sources are missing never count. A declared join
    covers a planned cut only when it sits at that cut's output frame.
    """
    attested = [_attest(item) for item in edit_list.evidence]
    planned = len(plan.cuts)
    results = {cut.index: cut for cut in joins.cuts} if joins is not None else {}
    declared_frames = [row.out_frame for row in declared.joins] if declared is not None else []
    compared = declared_cuts = 0
    for cut in plan.cuts:
        row = results.get(cut.index)
        if row is not None and _compared(row):
            compared += 1
        elif any(abs(frame - cut.out_frame) <= _DECLARED_SLACK_FRAMES for frame in declared_frames):
            declared_cuts += 1
    covered = compared + declared_cuts
    if planned and compared == planned:
        state = "checked"
        reason = "the edit list's cuts were checked against their sources"
    elif planned and covered == planned:
        state = "declared"
        reason = (
            f"{compared} of {planned} cuts were checked against their sources and the other "
            f"{declared_cuts} by declared joins at their frames"
            if compared
            else f"none of the {planned} cuts could be compared with a source; declared joins "
            "at their frames checked them by their words"
        )
    elif covered:
        state = "partial"
        how = (
            f" ({compared} against their sources, {declared_cuts} by declared joins)"
            if declared_cuts
            else " against their sources"
        )
        reason = f"{covered} of {planned} cuts were checked{how}; the others could not be compared"
    elif attested:
        state = "attested"
        matched = sum(row["match"] is True for row in attested)
        reason = (
            f"cuts were {ATTESTED_LABEL}; this is not counted as a pass. "
            f"{matched} of {len(attested)} evidence "
            f"{'file matches' if len(attested) == 1 else 'files match'} "
            "the recorded hash"
        )
    elif planned:
        state = "unavailable"
        reason = (
            f"the edit list has {planned} {'cut' if planned == 1 else 'cuts'}, but none "
            "could be compared with a source"
            + ("; no declared join sits at a planned cut" if declared_frames else "")
        )
    elif declared is not None:
        state = "declared"
        reason = _declared_reason(declared)
    elif edit_list.composite or _is_finished_render(plan, info, source_paths):
        state = "not_exposed"
        reason = (
            "the edit list says the render is a composite"
            if edit_list.composite
            else "the edit list's one range is a finished render of the same length"
        ) + ", so its cuts cannot be checked from it"
    else:
        state = "none"
        reason = "the edit list is one continuous range, so there are no cuts to check"
    return {
        "state": state,
        "label": _COVERAGE_LABELS[state],
        "reason": reason,
        "attested": attested,
        "declared_joins": len(edit_list.joins),
        "composite": edit_list.composite,
        "planned_cuts": planned,
        "checked_cuts": compared,
        "declared_cuts": declared_cuts,
    }


_NO_EVIDENCE_REASON = "no evidence in the edit list"


def _evidence_findings(attested) -> list[Finding]:
    """Warn for each attested evidence file that is missing or no longer matches its hash."""
    findings = []
    for row in attested:
        name = Path(row["path"]).name
        if not row["found"]:
            problem = "is missing"
        elif row["match"] is False:
            problem = (
                f"does not match its recorded hash (recorded {row['sha256'][:12]}, "
                f"found {row['actual_sha256'][:12]})"
            )
        else:
            continue
        findings.append(
            Finding(
                "edit_list.evidence",
                "warn",
                f"Evidence file {name} {problem}, so the cuts it attests are unconfirmed.",
                suggested_action=(
                    "Point the edit list at the evidence file that was checked, or review "
                    "the file again and record its current sha256."
                ),
                evidence=dict(row),
            )
        )
    return findings


def _evidence_coverage(attested, findings) -> dict[str, object]:
    """A matching hash is never a pass: report it as checked outside Yap Studio."""
    if not attested:
        return {"ran": False, "reason": _NO_EVIDENCE_REASON}
    if findings:
        return {"ran": True, "reason": None}
    return {
        "ran": False,
        "reason": f"{ATTESTED_LABEL}; the recorded hash matches, not counted as a pass",
    }


def _apply_cut_coverage(coverage, cut_coverage) -> None:
    """Replace the generic no-cuts reason with the cut coverage state."""
    for check in _JOIN_CHECKS:
        if coverage[check].get("reason") == _NO_CUTS_REASON:
            coverage[check] = {
                "ran": False,
                "reason": f"cut coverage: {cut_coverage['label']}",
            }


def _loudness_notes(structure: StructureResult, thresholds: Thresholds) -> list[Finding]:
    """Report measured whole-file levels before assembly sets the final loudness."""
    notes = []
    if structure.integrated_lufs is not None:
        notes.append(
            Finding(
                "loudness.integrated",
                "info",
                f"At 0:00.0, overall loudness is {structure.integrated_lufs:.1f} LUFS. "
                "Loudness is set when the video is assembled.",
                time_s=0.0,
                evidence={
                    "integrated_lufs": structure.integrated_lufs,
                    "target_lufs": thresholds.loudness_target_lufs,
                },
            )
        )
    if structure.true_peak_dbtp is not None:
        notes.append(
            Finding(
                "loudness.true_peak",
                "info",
                f"At 0:00.0, sound peaks at {structure.true_peak_dbtp:.1f} dBTP. "
                "Loudness is set when the video is assembled.",
                time_s=0.0,
                evidence={
                    "true_peak_dbtp": structure.true_peak_dbtp,
                    "maximum_dbtp": thresholds.true_peak_max_dbtp,
                },
            )
        )
    return notes


def verify(
    media: str,
    edl: str | None = None,
    source: list[str] | None = None,
    asr: Literal["auto", "parakeet-mlx", "parakeet-onnx", "faster-whisper", "none"] = "auto",
    asr_model: str | None = None,
    loudness_target: float = -14.0,
    strict: bool = False,
    report: str | None = None,
    artifacts: str | None = None,
    detail: Literal["summary", "full"] = "summary",
    platform: Literal["none", "youtube", "shorts", "tiktok", "reels"] = "none",
    loudness: Literal["check", "note"] = "check",
) -> dict:
    """Verify a finished render and return a pinned JSON-friendly report.

    Args:
        media: Path to the finished video render to verify.
        edl: Optional edit-list path used to compare the render with its source edits.
        source: Optional source path overrides, either PATH or repeatable ID=PATH values.
        asr: Speech engine name, auto selection, or none to skip speech checks.
        asr_model: Optional model name passed to the selected speech engine.
        loudness_target: Integrated loudness target in LUFS.
        strict: Treat warnings as a failing command result when the CLI chooses its exit code.
        report: Optional JSON report output path. No report file is written when omitted.
        artifacts: Optional directory for retained join audio artifacts.
        detail: Report detail level; full includes segment rows and complete cut evidence.
        platform: Published platform preset to check, or none to skip platform checks.
        loudness: check warns when integrated loudness or true peak miss their limits;
            note reports them as information for an intermediate render whose loudness
            is set when the video is assembled. Part-to-part loudness checks are unchanged.
    """
    return _verify(
        media, edl=edl, source=source, asr=asr, asr_model=asr_model,
        loudness_target=loudness_target, strict=strict, report=report, artifacts=artifacts,
        detail=detail, platform=platform, loudness=loudness,
    )


def _verify(
    media: str,
    edl: str | None = None,
    source: list[str] | None = None,
    asr: Literal["auto", "parakeet-mlx", "parakeet-onnx", "faster-whisper", "none"] = "auto",
    asr_model: str | None = None,
    loudness_target: float = -14.0,
    strict: bool = False,
    report: str | None = None,
    artifacts: str | None = None,
    detail: Literal["summary", "full"] = "summary",
    platform: Literal["none", "youtube", "shorts", "tiktok", "reels"] = "none",
    loudness: Literal["check", "note"] = "check",
    *,
    transitions: Sequence[TransitionWindow | Mapping[str, Any]] | None = None,
    click_audio: str | Path | None = None,
    words_audio: str | Path | None = None,
) -> dict:
    """Verify internal transition windows while retaining every audio join check."""
    del strict  # Exit policy belongs to report.exit_code; keep this public parameter.
    if detail not in {"summary", "full"}:
        raise ValidationError("detail", "must be summary or full")
    if platform not in {"none", *PLATFORMS}:
        raise ValidationError("platform", f"must be none or one of {', '.join(PLATFORMS)}")
    if loudness not in {"check", "note"}:
        raise ValidationError("loudness", "must be check or note")
    windows = tuple(item if isinstance(item, TransitionWindow) else TransitionWindow(**item)
                    for item in transitions or ())
    if windows and edl is None:
        raise ValidationError("transitions", "requires an edit list")
    preset = load_presets().get(platform) if platform != "none" else None
    platform_report = None
    effective_loudness_target = loudness_target
    if preset is not None:
        platform_report = {
            "name": platform,
            "label": preset["label"],
            "checked": preset["checked"],
            "rules": preset,
        }
        preset_loudness = preset.get("loudness")
        if preset_loudness is not None:
            effective_loudness_target = preset_loudness["target_lufs"]
    started_at = datetime.now(UTC)
    with paths.operation():
        media_path = paths.validate_input(media)
        click_path = paths.validate_input(click_audio) if click_audio is not None else None
        words_path = paths.validate_input(words_audio) if words_audio is not None else None
        edit_list = None
        edl_receipt_path = None
        if edl is not None:
            _LOGGER.info("Reading the edit list")
            load_path, edl_receipt_path = _edit_list_input(edl)
            edit_list = load_edit_list(load_path)
            if source:
                edit_list = _apply_overrides(edit_list, _absolute_overrides(source))
        elif source:
            raise ValidationError("source", "requires an edit list")

        input_paths = [media_path]
        if edl_receipt_path is not None:
            input_paths.append(edl_receipt_path)
        if click_path is not None:
            input_paths.append(click_path)
        if words_path is not None and words_path not in input_paths:
            input_paths.append(words_path)
        source_rows = []
        source_paths: dict[str, Path | None] = {}
        if edit_list is not None:
            for source_id, item in edit_list.sources.items():
                source_path = item.path
                found = source_path is not None and source_path.is_file()
                if found:
                    source_path = paths.validate_input(source_path)
                    input_paths.append(source_path)
                source_paths[source_id] = source_path
                source_rows.append(
                    {
                        "id": source_id,
                        "path": str(source_path) if source_path is not None else None,
                        "found": found,
                        **({"audio_stream": item.audio_stream}
                           if item.audio_stream is not None else {}),
                    }
                )

        report_path = (
            paths.validate_output(
                report,
                frozenset({".json"}),
                existing_json="verify-report",
            )
            if report is not None
            else None
        )
        artifacts_path = paths.validate_output_dir(artifacts) if artifacts is not None else None

        _LOGGER.info("Probing %s", media_path.name)
        info = ffmpeg.probe(media_path)
        if not info.has_video:
            raise InputError(str(media_path), "no video stream")
        backend = resolve_backend(asr, asr_model)
        second_backend = resolve_second_backend(backend)
        for engine in (backend, second_backend):
            if engine is not None:
                notice = engine.download_notice()
                if notice:
                    _LOGGER.info("%s", notice)
        thresholds = Thresholds().with_target(effective_loudness_target)
        work_dir = Path(tempfile.mkdtemp(prefix="yap-verify-"))
        try:
            sync = joins = detected = None
            if edit_list is not None:
                grade = (GradeSpec.from_dict({key: edit_list.grade[key]
                                             for key in ("look", "strength", "version", "color")
                                             if key in edit_list.grade})
                         if edit_list.grade is not None else None)
                grade_args = {"grade": grade} if grade is not None and grade.active else {}
                plan = Plan.build(
                    edit_list,
                    fps=info.fps,
                    sample_rate=info.sample_rate or 48000,
                )
                _LOGGER.info("Checking sync")
                sync = check_sync(
                    media_path,
                    plan,
                    source_paths,
                    source_streams={key: item.audio_stream
                                    for key, item in edit_list.sources.items()},
                    masks=edit_list.picture_masks,
                    regions=edit_list.picture_regions,
                    thresholds=thresholds,
                    **grade_args,
                )
                _LOGGER.info("Checking joins")
                joins = check_joins(
                    media_path,
                    plan,
                    source_paths,
                    sync,
                    source_streams={key: item.audio_stream
                                    for key, item in edit_list.sources.items()},
                    backend=backend,
                    second_backend=second_backend,
                    masks=edit_list.picture_masks,
                    regions=edit_list.picture_regions,
                    transitions=windows,
                    click_audio=click_path,
                    **({"words_audio": words_path} if words_path is not None else {}),
                    thresholds=thresholds,
                    work_dir=work_dir,
                    artifacts_dir=artifacts_path,
                    **grade_args,
                )
                declared = None
                if edit_list.joins and backend is not None and info.has_audio:
                    _LOGGER.info("Transcribing %d declared joins", len(edit_list.joins))
                    declared = check_declared_joins(
                        words_path or media_path,
                        edit_list.joins,
                        backend=backend,
                        fps=info.fps,
                        work_dir=work_dir,
                        video_offset_s=info.video_offset_s,
                    )
                _LOGGER.info("Checking loudness and picture")
                structure = check_structure(media_path, plan=plan, thresholds=thresholds)
                findings = [
                    *sync.findings,
                    *joins.findings,
                    *(declared.findings if declared is not None else []),
                    *structure.findings,
                ]
                coverage = _edit_coverage(
                    info, sync, joins, backend, plan, second_backend=second_backend,
                    click_source="voice_stem" if click_path is not None else "mix",
                )
                grade_findings, grade_coverage = check_grade(media_path, edit_list, plan, info)
                findings.extend(grade_findings)
                coverage["picture.grade"] = grade_coverage
                _declared_coverage(coverage, edit_list, declared, backend, info)
                cut_coverage = _cut_coverage(edit_list, plan, info, source_paths, declared, joins)
                _apply_cut_coverage(coverage, cut_coverage)
                evidence_findings = _evidence_findings(cut_coverage["attested"])
                findings.extend(evidence_findings)
                coverage["edit_list.evidence"] = _evidence_coverage(
                    cut_coverage["attested"], evidence_findings
                )
            else:
                cut_coverage = None
                _LOGGER.info("Finding cuts")
                if backend is not None:
                    _LOGGER.info("Transcribing detected cut clips")
                _LOGGER.info("Checking loudness and picture")
                detected = check_without_edl(
                    media_path,
                    backend=backend,
                    thresholds=thresholds,
                    work_dir=work_dir,
                )
                structure = detected.structure
                findings = list(detected.findings)
                coverage = dict(detected.checks)
                coverage["picture.transition"] = {"ran": False, "reason": "no declared transitions"}
                coverage["picture.grade"] = {"ran": False, "reason": "no grade in the edit list"}
                for check in CHECK_IDS:
                    if check.startswith("platform."):
                        coverage[check] = {"ran": False, "reason": "no platform chosen"}
                if backend is None and info.has_audio:
                    coverage["join.listen"] = {"ran": False, "reason": _NO_ENGINE_REASON}
                if not info.has_audio:
                    for check in _AUDIO_CHECKS:
                        coverage[check] = {"ran": False, "reason": _NO_AUDIO_REASON}
                if not detected.cuts:
                    for check in ("join.click", "join.frames", "join.listen", "picture.flash"):
                        if check in coverage:
                            coverage[check] = {
                                "ran": False,
                                "reason": "no cuts found in the picture",
                            }

            if loudness == "note":
                notes = _loudness_notes(structure, thresholds)
                findings = [
                    finding for finding in findings
                    if finding.check not in _WHOLE_FILE_LOUDNESS_CHECKS
                ] + notes
                structure = replace(
                    structure,
                    findings=[
                        finding for finding in structure.findings
                        if finding.check not in _WHOLE_FILE_LOUDNESS_CHECKS
                    ] + notes,
                )

            if preset is not None:
                findings.extend(check_platform(info, preset, name=platform))
                for check in CHECK_IDS:
                    if check.startswith("platform."):
                        coverage[check] = {"ran": True, "reason": None}

            caption_checks = ("captions.timing", "captions.cards")
            layout_checks = (*caption_checks, "captions.words", "faces.clearance")
            layout_path = media_path.with_suffix(LAYOUT_SUFFIX)
            if layout_path.is_file():
                try:
                    input_paths.append(paths.validate_input(layout_path))
                except InputError:
                    pass  # read_layout_record reports the unreadable record as a warning.
            faces_path = media_path.with_suffix(FACES_SUFFIX)
            if faces_path.is_file():
                try:
                    input_paths.append(paths.validate_input(faces_path))
                except InputError:
                    pass
            try:
                layout_record = read_layout_record(media_path, width=info.width, height=info.height)
            except InputError as exc:
                for check in layout_checks:
                    coverage[check] = {"ran": False, "reason": exc.reason}
                    if check != "captions.words":
                        findings.append(Finding(check, "warn", exc.reason))
            else:
                if layout_record is None:
                    for check in layout_checks:
                        coverage[check] = {
                            "ran": False, "reason": "no layout record next to this render",
                        }
                elif info.variable_frame_rate:
                    for check in caption_checks:
                        coverage[check] = {
                            "ran": False,
                            "reason": (
                                "variable frame rate render; caption frames have no fixed clock"
                            ),
                        }
                else:
                    coverage.update(caption_coverage(layout_record))
                    if coverage["captions.timing"]["ran"]:
                        findings.extend(check_caption_timing(
                            layout_record, fps=info.fps,
                            duration_s=info.video_duration_s or info.duration_s or 0.0,
                            thresholds=thresholds,
                        ))
                    if coverage["captions.cards"]["ran"]:
                        findings.extend(check_caption_cards(layout_record, fps=info.fps))

                if layout_record is not None:
                    words_reason = None
                    if not layout_record.measured:
                        words_reason = f"caption layout was not measured: {layout_record.reason}"
                    elif not any(layout_record.words.values()):
                        words_reason = "no captions in this render"
                    elif backend is None:
                        words_reason = _NO_ENGINE_REASON
                    elif not info.has_audio:
                        words_reason = _NO_AUDIO_REASON
                    coverage["captions.words"] = {"ran": words_reason is None,
                                                  "reason": words_reason}
                    if words_reason is None:
                        findings.extend(check_caption_words(
                            layout_record.words,
                            heard_transcript(click_path or media_path, backend),
                        ))
                    faces, faces_source = final_faces(media_path)
                    face_findings, face_coverage = check_face_clearance(
                        layout_record, faces, faces_source=faces_source,
                    )
                    findings.extend(face_findings)
                    coverage["faces.clearance"] = face_coverage

            contrast_path = media_path.with_suffix(CONTRAST_SUFFIX)
            if contrast_path.is_file():
                try:
                    input_paths.append(paths.validate_input(contrast_path))
                except InputError:
                    pass
            try:
                contrast_record = read_contrast_record(media_path)
            except InputError as exc:
                coverage["captions.contrast"] = {"ran": False, "reason": exc.reason}
                findings.append(Finding("captions.contrast", "warn", exc.reason))
            else:
                if contrast_record is None:
                    coverage["captions.contrast"] = {
                        "ran": False, "reason": "no contrast record next to this render",
                    }
                else:
                    contrast_findings, contrast_coverage = check_caption_contrast(
                        contrast_record, fps=info.fps, thresholds=thresholds,
                    )
                    findings.extend(contrast_findings)
                    coverage["captions.contrast"] = contrast_coverage

            from yap_studio.taste.render import read_taste_record

            try:
                taste_record = read_taste_record(media_path)
            except InputError as exc:
                coverage["taste.rules"] = {"ran": False, "reason": exc.reason}
                findings.append(Finding("taste.rules", "warn", exc.reason))
            else:
                if taste_record is None:
                    coverage["taste.rules"] = {
                        "ran": False, "reason": "no taste record next to this render",
                    }
                else:
                    from yap_studio.taste.check import check_taste

                    taste_check = check_taste(media_path, taste_record)
                    findings.extend(taste_check.findings)
                    coverage["taste.rules"] = taste_check.coverage

            youtube_text = None
            coverage["platform.chapters"] = {"ran": False, "reason": "no chapters in this video"}
            if edl is not None and load_path.suffix == ".json":
                from yap_studio.youtube_chapters import chapter_list

                edit_data = json.loads(load_path.read_bytes())
                chapter_data = edit_data.get("youtube_chapters")
                if chapter_data is not None and not (
                    isinstance(chapter_data, dict) and not chapter_data.get("chapters")
                ):
                    if isinstance(chapter_data, dict):
                        try:
                            edit_fps = Fraction(str(edit_data.get(
                                "fps", edit_list.fps or info.fps)))
                            chapter_fps = Fraction(str(chapter_data.get("fps", edit_fps)))
                            if chapter_fps != info.fps or chapter_fps != edit_fps:
                                raise ValueError
                        except (ValueError, ZeroDivisionError):
                            fix = ("Use chapter evidence with the same frame rate "
                                   "as the video and edit list.")
                        else:
                            youtube_text, fix = chapter_list(
                                chapter_data.get("chapters", []), info.video_frames,
                                fps=chapter_fps, opening=chapter_data.get("opening", "Opening"))
                    else:
                        fix = "Give every chapter a name and a valid start time inside the video."
                    coverage["platform.chapters"] = {"ran": True, "valid": fix is None}
                    if fix:
                        findings.append(Finding("platform.chapters", "warn",
                                                "YouTube chapters need a change.",
                                                suggested_action=fix))
            ended_at = datetime.now(UTC)
            artifact_outputs = list(
                dict.fromkeys(
                    artifact
                    for cut in (joins.cuts if joins is not None else [])
                    for artifact in cut.words.artifacts
                )
            )
            run_receipt = receipt.build_receipt(
                inputs=input_paths,
                outputs=artifact_outputs,
                started_at=started_at,
                ended_at=ended_at,
                extra={
                    "speech": (
                        {
                            "engine": backend.name,
                            "model": backend.model,
                            "version": backend.version(),
                        }
                        if backend is not None
                        else None
                    ),
                    "speech_second": (
                        {
                            "engine": second_backend.name,
                            "model": second_backend.model,
                            "version": second_backend.version(),
                            **({"run": second_backend.last_run}
                               if getattr(second_backend, "last_run", None) is not None else {}),
                        }
                        if second_backend is not None
                        else None
                    ),
                    "thresholds": asdict(thresholds),
                },
            )
            if report_path is not None:
                run_receipt["outputs"].append(
                    {
                        "path": str(report_path),
                        "sha256": None,
                        "bytes": None,
                        "role": "this report",
                    }
                )
            if words_path is not None and edit_list is not None:
                coverage["join.words"]["words_source"] = "no_effects_stem"
            result = build_report(
                media_info=info,
                edit_list=edit_list,
                sources=source_rows,
                findings=findings,
                structure=structure,
                checks=coverage,
                receipt=run_receipt,
                target_lufs=effective_loudness_target,
                platform=platform_report,
                detail=detail,
                joins=joins,
                sync=sync,
                detected=detected,
                cut_coverage=cut_coverage,
            )
            if youtube_text is not None:
                result["youtube_chapters"] = youtube_text
            if report_path is not None:
                paths.atomic_write_text(
                    report_path,
                    json.dumps(result, indent=2, allow_nan=False) + "\n",
                    allowed_existing_suffixes=frozenset({".json"}),
                    existing_json="verify-report",
                )
            return result
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
