"""Compare a finished picture with the approved rules pinned by its render."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from yap_studio import ffmpeg
from yap_studio.checks.model import Finding
from yap_studio.errors import YapError
from yap_studio.taste.frames import delta_e, presentation_timing
from yap_studio.taste.learn import measure_style, tolerance
from yap_studio.taste.model import Taste, snapshot_at
from yap_studio.taste.store import load_taste

_ACTION = "Render again with the approved style, or revoke the rule if it no longer holds."


@dataclass(frozen=True)
class TasteCheck:
    """Findings, check coverage and a result for each approved rule."""

    findings: list[Finding]
    coverage: dict
    rules: list[dict]

    def to_dict(self) -> dict:
        """Return rule results and findings as plain JSON data."""
        return asdict(self)


def _field_words(field: str) -> str:
    return field.replace(".", " ").replace("_", " ").replace("color", "colour")


def _unmeasured_reason(rows: list[dict]) -> str:
    """Keep rule reasons in unchecked coverage, with a bounded summary."""
    prefix = "no applied rule could be measured from the picture: "
    shown = []
    for index, row in enumerate(rows):
        detail = f"{row['field']} ({row['reason']})"
        remaining = len(rows) - index - 1
        suffix = f"; and {remaining} more" if remaining else ""
        candidate = prefix + "; ".join([*shown, detail]) + suffix
        if len(candidate) > 300:
            if not shown:
                budget = 300 - len(prefix) - len(suffix) - 3
                return prefix + detail[:budget].rstrip() + "..." + suffix
            return prefix + "; ".join(shown) + f"; and {len(rows) - len(shown)} more"
        shown.append(detail)
    return prefix + "; ".join(shown)


def _receipt_findings(media: Path, record: dict) -> list[Finding]:
    target = media.with_suffix(".receipt.json")
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, UnicodeError, ValueError, RecursionError):
        return [Finding("taste.rules", "warn", "The render's receipt could not be read.")]
    style = data.get("style") if isinstance(data, dict) else None
    if isinstance(style, dict) and "sha256" in style and style["sha256"] != record["style_sha256"]:
        return [Finding(
            "taste.rules", "fail",
            "The render's receipt names a different style than its taste record.",
            suggested_action=_ACTION,
            evidence={"wanted": record["style_sha256"], "measured": style["sha256"]},
        )]
    return []


def _first_times(media: Path, frames: list[int]) -> dict[int, float]:
    """Use original presentation times when the evidence frames can be located."""
    if not frames:
        return {}
    try:
        timing = presentation_timing(media, ffmpeg.probe(media))
    except (YapError, OSError, ValueError):
        return {}
    return {frame: timing.starts[frame] for frame in frames if 0 <= frame < len(timing.starts)}


def _face_clearance(media: Path) -> tuple[list[Finding], dict]:
    """Use the same hash-bound geometry and faces as final verification."""
    from yap_studio.checks.caption_layout import read_layout_record
    from yap_studio.checks.face_clearance import check_face_clearance, final_faces

    if not media.with_suffix(".faces.json").is_file():
        return [], {"ran": False, "reason": "no faces record next to this render"}
    try:
        info = ffmpeg.probe(media)
        layout = read_layout_record(media, width=info.width, height=info.height)
        if layout is None:
            return [], {"ran": False, "reason": "no layout record next to this render"}
        faces, source = final_faces(media)
        findings, coverage = check_face_clearance(layout, faces, faces_source=source)
        return findings, {**coverage, "captions_measured": layout.measured,
                          "captions_reason": layout.reason}
    except (YapError, OSError, ValueError) as exc:
        return [], {"ran": False, "reason": str(exc)}


def check_taste(
    media: Path, record: dict, *, taste: Taste | None = None, measure=measure_style,
) -> TasteCheck:
    """Report approved rules from the render's matching private taste revision."""
    findings = _receipt_findings(media, record)
    applied = record["applied"]
    decisions = [*applied, *record.get("enforced", []), *record["skipped"]]
    if not decisions:
        return TasteCheck(findings, {
            "ran": False, "reason": "no approved taste rule applied to this render",
        }, [])
    private = taste if taste is not None else load_taste()
    revision = record["revision"]
    snapshot = snapshot_at(private, revision) if revision <= private.revision else None
    if snapshot is None or snapshot.sha256 != record["sha256"]:
        reason = f"this computer's taste history does not match the pinned revision {revision}"
        findings.append(Finding("taste.rules", "info", reason))
        return TasteCheck(findings, {"ran": False, "reason": reason}, [
            {**item, "field": item.get("field", "approved rule"),
             "wanted": None, "measured": None, "status": "not_measured",
             "reason": reason, "frames": []}
            for item in decisions
        ])
    pinned = {rule["id"]: rule["machine"] for rule in snapshot.rules}
    fields = tuple(dict.fromkeys(item["field"] for item in applied))
    try:
        measured = measure(media, fields=fields) if fields else {}
    except YapError as exc:
        reason = getattr(exc, "reason", getattr(exc, "detail", exc.message))
        measured = {field: {"value": None, "reason": reason} for field in fields}
    rows = []
    for item in applied:
        identity, field = item["id"], item["field"]
        machine = pinned.get(identity)
        valid = (
            machine is not None and machine["kind"] == "style_field" and machine["field"] == field
        )
        wanted = machine["value"] if valid else None
        observation = measured.get(field, {})
        value = observation.get("value") if valid else None
        frames = observation.get("frames", []) if value is not None else []
        reason = None
        limit = tolerance(field)
        if not valid:
            reason = "The applied rule and field are not in the pinned taste revision."
        elif value is None:
            reason = observation.get("reason") or "Not measured from pixels."
        if reason:
            status = "not_measured"
            findings.append(Finding(
                "taste.rules", "info", f"Rule {identity}, {_field_words(field)}: {reason}",
                evidence={"rule_id": identity, "field": field, "wanted": wanted,
                          "measured": None, "frames": [], "reason": reason},
            ))
        else:
            if field.endswith("color"):
                agrees = delta_e(value, wanted) <= limit
            elif (
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and isinstance(wanted, (int, float)) and not isinstance(wanted, bool)
            ):
                agrees = abs(value - wanted) <= limit
            else:
                agrees = value == wanted
            status = "pass" if agrees else "fail"
            if not agrees:
                findings.append(Finding(
                    "taste.rules", "fail",
                    f"{_field_words(field).capitalize()} is {value}; "
                    f"approved rule {identity} wants {wanted}.",
                    suggested_action=_ACTION,
                    evidence={"rule_id": identity, "field": field, "wanted": wanted,
                              "measured": value, "frames": frames, "tolerance": limit},
                ))
        rows.append({**item, "wanted": wanted, "measured": value, "status": status,
                     "reason": reason, "frames": frames})
    applied_ids = {item["id"] for item in applied}
    skipped = {item["id"]: item["reason"] for item in record["skipped"]}
    clearance = None
    for rule in snapshot.rules:
        identity, machine = rule["id"], rule["machine"]
        if identity in applied_ids:
            continue
        reason = skipped.get(identity)
        if machine is not None and machine["kind"] == "placement" and reason in (
            None, "not_a_style_rule",
        ):
            if clearance is None:
                clearance = _face_clearance(media)
            face_findings, face_coverage = clearance
            layer = machine["layer"]
            face_findings = [item for item in face_findings if (
                layer == "captions" and item.evidence.get("target_type") == "caption"
                or layer == "title" and item.evidence.get("target_kind") == "title"
                or layer == "cards" and item.evidence.get("target_kind") == "card"
            )]
            status = ("fail" if any(item.severity == "fail" for item in face_findings) else
                      "warn" if any(item.severity == "warn" for item in face_findings) else "pass")
            reason = face_coverage.get("reason")
            if not face_coverage["ran"]:
                status = "not_measured"
            if layer == "captions" and face_coverage.get("captions_measured") is False:
                status = "not_measured"
                reason = f"captions were not measured: {face_coverage['captions_reason']}"
            evidence = [item.evidence for item in face_findings]
            rows.append({"id": identity, "field": f"{machine['layer']} clear of face",
                         "check": "faces.clearance", "status": status,
                         "reason": reason, "evidence": evidence})
            if status in {"fail", "warn"}:
                findings.append(Finding(
                    "taste.rules", status,
                    f"Rule {identity}: faces.clearance failed." if status == "fail" else
                    f"Rule {identity}: the style allows captions over the face.",
                    evidence={"rule_id": identity, "check": "faces.clearance",
                              "findings": evidence},
                ))
        elif machine is None:
            rows.append({"id": identity, "field": rule["sentence"],
                         "status": "followed_by_agent", "reason": "followed_by_agent"})
        else:
            rows.append({"id": identity,
                         "field": machine.get("field", machine["kind"].replace("_", " ")),
                         "status": "skipped", "reason": reason or "not_in_render_record"})
    failures = [f for f in findings if f.severity == "fail" and f.evidence.get("frames")]
    times = _first_times(media, [f.evidence["frames"][0] for f in failures])
    if times:
        findings = [
            replace(f, time_s=times.get(f.evidence["frames"][0])) if f in failures else f
            for f in findings
        ]
    missing = sum(row["status"] in ("not_measured", "skipped") for row in rows)
    coverage = (
        {"ran": False, "reason": _unmeasured_reason(rows)}
        if missing == len(rows) else {"ran": True, "reason": None}
    )
    if 0 < missing < len(rows):
        coverage["note"] = f"{missing} of {len(rows)} rules not measured"
    return TasteCheck(findings, coverage, rows)
