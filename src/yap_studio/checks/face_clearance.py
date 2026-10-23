"""Verify timed caption and element geometry against hash-bound render faces."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from yap_studio import paths
from yap_studio.checks.caption_layout import LayoutRecord
from yap_studio.checks.model import Finding, format_time
from yap_studio.errors import InputError, ValidationError
from yap_studio.faces.clearance import check_layout
from yap_studio.faces.detect import FaceDetector, detect_faces
from yap_studio.faces.model import FacesData, load_faces
from yap_studio.faces.project import FACE_EVERY
from yap_studio.receipt import sha256_file

FACES_SUFFIX = ".faces.json"


def final_faces(media: Path, *, detector: FaceDetector | None = None) -> tuple[FacesData, str]:
    """Reuse faces belonging to this media, otherwise measure the render itself."""
    source = paths.validate_input(media)
    digest = sha256_file(source)
    target = media.with_suffix(FACES_SUFFIX)
    try:
        faces = load_faces(paths.validate_input(target))
    except InputError as exc:
        reason = exc.reason
    except (ValueError, OSError) as exc:
        reason = str(exc)
    else:
        if faces.video.sha256 == digest:
            return faces, "record"
        reason = "faces record video hash does not match the media"
    return (
        detect_faces(media, every=FACE_EVERY, detector=detector),
        f"detected on the render; faces record not used: {reason}",
    )


def check_face_clearance(
    record: LayoutRecord, faces: FacesData, *, faces_source: str,
) -> tuple[list[Finding], dict]:
    """Name face clearance violations and disclose gaps in measured coverage."""
    notes = []
    layout = record.layout
    if not record.measured:
        notes.append(f"captions were not measured: {record.reason}")
    coverage = {"ran": True, "reason": None}
    if faces.status == "unmeasured":
        coverage = {"ran": False, "reason": f"faces not measured: {faces.reason}"}
        if notes:
            coverage["note"] = "; ".join(notes)
        return [], coverage
    try:
        if not record.measured:
            layout = replace(layout, captions=())
        layout = replace(layout, elements=tuple(
            replace(element, face_windows=record.face_windows[element.id])
            if element.id in record.face_windows else element for element in layout.elements
        ))
        report = check_layout(faces, layout)
    except ValidationError as exc:
        coverage = {"ran": False, "reason": str(exc)}
        if notes:
            coverage["note"] = "; ".join(notes)
        return [], coverage

    missing = sum(not frame for frame in faces.frames)
    if missing:
        notes.append(f"{missing} frames have no face boxes")
    sources = Counter()
    runs = faces.sampling.get("frame_sources", [])
    for run in runs if isinstance(runs, list) else ():
        if (isinstance(run, dict) and isinstance(run.get("source"), str)
                and type(run.get("start_frame")) is int and type(run.get("end_frame")) is int):
            sources[run["source"]] += max(
                0, min(run["end_frame"], faces.video.frames) - max(run["start_frame"], 0),
            )
    if sources and (missing or sources["zoom_estimate"]):
        notes.append("frame sources: " + ", ".join(
            f"{source}: {count}" for source, count in sorted(sources.items()) if count
        ))
    if notes:
        coverage["note"] = "; ".join(notes)

    findings = []
    for violation in report.violations:
        if violation.kind != "face":
            continue
        caption = violation.target_type == "caption"
        label = "Caption" if caption else violation.target_kind.capitalize()
        piece_id = violation.target_id
        element_id = record.element_ids.get(piece_id, piece_id)
        name = violation.text if caption else element_id
        media_label = record.labels.get(piece_id) if not caption else None
        target_name = f'{label} "{name}"' + (f" ({media_label})" if media_label else "")
        start = report.time_at(violation.start_frame)
        end = report.time_at(violation.end_frame)
        findings.append(Finding(
            "faces.clearance", "fail",
            f'{target_name} is over the speaker\'s face '
            f"from {format_time(start)} to {format_time(end)}.",
            time_s=start,
            suggested_action=(
                "Run captions again so the phrase is placed clear of the face, "
                "or move the face out of the caption band." if caption else
                "Move the overlay in the timeline (change its placement or box) "
                "so it sits clear of the face." if violation.target_kind == "overlay" else
                "Plan the card again (run graphics) so it sits clear of the face."
            ),
            evidence={
                "target_type": violation.target_type, "target_id": piece_id,
                **({"element_id": element_id} if not caption else {}),
                **({"label": media_label} if media_label else {}),
                "target_kind": violation.target_kind,
                "start_frame": violation.start_frame, "end_frame": violation.end_frame,
                "frame": violation.frame, "gap_px": violation.gap_px,
                "required_px": violation.required_px,
                "target_box": violation.target_box.to_dict(),
                "face": violation.face.to_dict() if violation.face is not None else None,
                "head": violation.head.to_dict() if violation.head is not None else None,
                "faces_source": faces_source,
            },
        ))
    return findings, coverage
