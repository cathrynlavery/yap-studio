"""Serve reviewer findings only for the selected render's known hash."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
from math import isfinite
from pathlib import Path

from yap_studio.errors import InputError
from yap_studio.feedback.versions import (
    _receipt_pieces,
    find_version_by_media,
    load_version,
    version_pieces,
)
from yap_studio.project.receipts import read_receipt
from yap_studio.review.media import stat_identity
from yap_studio.watch.findings import default_findings_path, read_findings


def note_target(kind: str, *, layer_kind: str | None = None) -> str | None:
    """Suggest the creator's comment target for a reviewer finding."""
    if kind == "layer_check":
        return "caption" if layer_kind == "caption" else "element"
    return {
        "slow": "story",
        "jarring_cut": "story",
        "join_listen": "story",
        "face_covered": "element",
        "behind_hidden": "element",
        "caption_issue": "caption",
        "caption_contrast": "caption",
        "audio": "speech",
    }.get(kind)


def _short_name(path: Path, project_dir: Path | None) -> str:
    if project_dir is not None:
        try:
            return path.resolve().relative_to(project_dir.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


@lru_cache(maxsize=128)
def _read_cached(
    path: Path, identity: tuple[int, int, int, int, int],
) -> tuple[dict | None, str | None]:
    try:
        return read_findings(path), None
    except InputError:
        return None, "malformed findings file"
    except OSError:
        return None, "cannot read findings file"


def _item_order(item: dict) -> tuple[int, str, int]:
    """Sort reviewer and receipt finding ids numerically within their prefix."""
    return item["start_frame"], item["id"][0], int(item["id"][1:])


def _behind_measurements(value: object) -> dict[str, dict]:
    """Keep valid flagged measurements, including their original output placement."""
    measurements = {}
    if not isinstance(value, list):
        return measurements
    for entry in value:
        if not isinstance(entry, dict):
            continue
        identifier = entry.get("id")
        start, frames, share = (entry.get(key) for key in ("start_frame", "frames", "hidden_share"))
        if (not isinstance(identifier, str) or not identifier
                or type(start) is not int or start < 0
                or type(frames) is not int or frames <= 0
                or not isinstance(share, (int, float)) or isinstance(share, bool)
                or not 0 <= share <= 1 or not isfinite(share)
                or entry.get("flag") is not True):
            continue
        if "hidden_frames" in entry:
            hidden = entry["hidden_frames"]
            if type(hidden) is not int or not 0 <= hidden <= frames:
                continue
        measurements[identifier] = entry
    return measurements


def _layer_findings(project_dir: Path, sha256: str) -> tuple[list, list, list]:
    """Join checks to the final's retained layer hashes without probing media."""
    items, summary, stale = [], [], []
    try:
        assembled = read_receipt(project_dir / "final")
    except InputError:
        return items, summary, stale
    if (not assembled or assembled.get("status") != "verified"
            or not any(output.get("sha256") == sha256
                       for output in assembled.get("outputs", []))):
        return items, summary, stale
    fps = Fraction(str(assembled.get("timeline", {}).get("fps", "30/1")))
    verification = assembled.get("verify", {})
    contrast_ran = verification.get("checks", {}).get("captions.contrast", {}).get("ran") is True
    weak = {caption["id"]: caption
            for caption in assembled.get("caption_contrast", {}).get("weak", [])}
    for finding in verification.get("findings", []):
        if (finding.get("check") != "captions.contrast"
                or finding.get("severity") not in {"fail", "warn"}):
            continue
        evidence = finding.get("evidence", {})
        point = evidence
        if "frame" not in point:
            point = next(iter(evidence.get("worst", [])), {})
        caption = weak.get(point.get("caption_id"), {})
        frame = point.get("frame")
        if frame is None:
            frame = round(point.get("time_s", finding.get("time_s") or 0) * fps)
        first = evidence.get("start_frame", caption.get("start_frame", max(0, frame)))
        end = evidence.get("end_frame", caption.get("end_frame", first + 1))
        start = min(end - 1, max(first, frame))
        items.append({
            "kind": "caption_contrast", "source": "machine", "reviewer": None,
            "start_frame": start, "end_frame": start + 1,
            "start_s": float(start / fps), "end_s": float((start + 1) / fps),
            "reason": finding["message"], "fix": finding.get("suggested_action"),
            "severity": "error" if finding["severity"] == "fail" else "warning",
            "evidence": deepcopy(evidence), "target": note_target("caption_contrast"),
        })
    behind = _behind_measurements(assembled.get("behind_layers"))
    for stage in ("captions", "graphics"):
        name = f"{stage}/receipt.json"
        try:
            record = read_receipt(project_dir / stage)
        except InputError:
            stale.append(name)
            continue
        if not record:
            continue
        if record.get("status") != "verified":
            stale.append(name)
            continue
        outputs = {output["path"]: output["sha256"] for output in record.get("outputs", [])}
        layers = [record.get("layer")] if stage == "captions" else record.get("elements", [])
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            path = layer["path"]
            digest = assembled.get("inputs", {}).get(f"media:{path}")
            if digest is None or outputs.get(path) != digest:
                if name not in stale:
                    stale.append(name)
                continue
            measurement = behind.get(layer["id"]) if stage == "graphics" else None
            if measurement is not None:
                start = measurement["start_frame"]
                end = start + measurement["frames"]
                percent = round(measurement["hidden_share"] * 100)
                items.append({
                    "kind": "behind_hidden", "source": "machine", "reviewer": None,
                    "start_frame": start, "end_frame": end,
                    "start_s": float(start / fps), "end_s": float(end / fps),
                    "reason": f"Card {layer['id']} is mostly hidden behind the speaker "
                              f"for {percent} percent of its time",
                    "fix": "Move the card or set it in front of the speaker",
                    "severity": "warning", "evidence": deepcopy(measurement),
                    "target": note_target("behind_hidden"),
                })
            checks = (record.get("layer_checks") if stage == "captions" else
                      record.get("layer_checks", {}).get(layer["id"]))
            if not isinstance(checks, dict) or checks.get("schema") != "yap-studio/layer-checks@1":
                continue
            skipped = []
            for pass_name, check in checks.get("passes", {}).items():
                if pass_name == "lint":
                    continue
                if check.get("status") == "skipped":
                    if not (stage == "captions" and pass_name == "contrast" and contrast_ran):
                        skipped.append({"pass": pass_name, "reason": check.get("reason")})
                    continue
                findings = list(check.get("findings", []))
                unlisted = {
                    severity: max(0, check.get(count, 0) - sum(
                        finding.get("severity") == severity for finding in findings
                    ))
                    for severity, count in (("error", "errors"), ("warning", "warnings"))
                }
                if any(unlisted.values()):
                    details = " and ".join(
                        f"{count} more {pass_name} {severity}{'s' if count != 1 else ''}"
                        for severity, count in unlisted.items() if count
                    )
                    findings.append({
                        "severity": "error" if unlisted["error"] else "warning",
                        "message": f"{details} not listed", "time": None,
                    })
                for finding in findings:
                    if finding.get("severity") not in {"error", "warning"}:
                        continue
                    start = layer["start_frame"]
                    end = start + layer["frames"]
                    time = finding.get("time")
                    if (isinstance(time, (int, float)) and not isinstance(time, bool)
                            and isfinite(time)):
                        start = min(end - 1, max(start, start + round(time * fps)))
                        end = start + 1
                    items.append({
                        "kind": "layer_check", "source": "machine", "reviewer": None,
                        "start_frame": start, "end_frame": end,
                        "start_s": float(start / fps), "end_s": float(end / fps),
                        "reason": finding["message"], "fix": "Review the layer.",
                        "pass": pass_name, "check": pass_name, "severity": finding["severity"],
                        "evidence": {}, "target": note_target(
                            "layer_check",
                            layer_kind="caption" if stage == "captions" else "element",
                        ),
                    })
            summary.append({"path": path, "status": checks["status"], "skipped": skipped})
    items.sort(key=lambda item: item["start_frame"])
    for index, item in enumerate(items, 1):
        item["id"] = f"l{index}"
    return items, summary, stale


def _join_findings(project_dir: Path, sha256: str, version_id: str | None) -> list[dict]:
    """Keep uncertain source edges only where they still meet on the review timeline."""
    try:
        record = read_receipt(project_dir / "cut")
        if not record or record.get("status") != "verified":
            return []
        joins = record.get("plan", {}).get("listen_joins", [])
        if not isinstance(joins, list) or not joins:
            return []
        source = _receipt_pieces(record, project_dir / "cut/receipt.json")
        manifest = (load_version(project_dir, version_id) if version_id is not None else
                    find_version_by_media(project_dir, sha256))
        if manifest is not None:
            if manifest.media["sha256"] != sha256 or not manifest.snapshots.get("cut_pieces"):
                return []
            target = version_pieces(manifest)
            duration = Fraction(str(manifest.media["duration_s"]))
        else:
            digest = next((item["sha256"] for item in record.get("outputs", [])
                           if item.get("path") == "cut/cut.mp4"), None)
            duration = Fraction(source["total_frames"], 1) / Fraction(source["fps"])
            if digest != sha256:
                final = read_receipt(project_dir / "final")
                if (not final or final.get("status") != "verified" or digest is None
                        or final.get("inputs", {}).get("media:cut/cut.mp4") != digest
                        or not any(item.get("sha256") == sha256
                                   for item in final.get("outputs", []))):
                    return []
                duration = Fraction(str(final.get("timeline", {}).get("duration_s", duration)))
            target = source
    except (InputError, OSError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return []

    def boundary(left, right):
        return (left["segment"], left["take"], Fraction(left["src_t1"]),
                right["segment"], right["take"], Fraction(right["src_t0"]))

    source_edges = {right["out_f0"]: (left, right) for left, right in
                    zip(source["pieces"], source["pieces"][1:], strict=False)}
    target_edges = {}
    for left, right in zip(target["pieces"], target["pieces"][1:], strict=False):
        same_takes = all(source.get("inputs", {}).get(f"take:{piece['take']}") ==
                         target.get("inputs", {}).get(f"take:{piece['take']}")
                         for piece in (left, right))
        if same_takes and left["out_f0"] + left["n_frames"] == right["out_f0"]:
            target_edges.setdefault(boundary(left, right), []).append(right["out_f0"])
    fps = Fraction(target["fps"])
    last = int(duration * fps)
    padding = round(Fraction(3, 2) * fps)
    items, seen = [], set()
    for index, join in enumerate(joins, 1):
        if not isinstance(join, dict) or type(join.get("frame")) is not int:
            continue
        pair = source_edges.get(join["frame"])
        if pair is None or (join.get("left"), join.get("right")) != (
                pair[0]["segment"], pair[1]["segment"]):
            continue
        matches = target_edges.get(boundary(*pair), [])
        if len(matches) != 1 or matches[0] in seen:
            continue
        frame = matches[0]
        if not 0 < frame < last:
            continue
        seen.add(frame)
        start, end = max(0, frame - padding), min(last, frame + padding)
        items.append({
            "id": f"j{index}", "kind": "join_listen", "source": "machine", "reviewer": None,
            "start_frame": start, "end_frame": end,
            "start_s": float(start / fps), "end_s": float(end / fps),
            "reason": "Listen to this join", "fix": "Check that the speech flows across the join.",
            "severity": "warning", "evidence": {}, "target": note_target("join_listen"),
        })
    return items


def findings_for_media(
    media_path: Path, sha256: str, *, project_dir: Path | None = None,
    version_id: str | None = None,
) -> dict:
    """Find the first matching findings file, keeping stale notes off the review."""
    result = {"available": False, "file": None, "stale": [], "items": []}
    first_error = None
    candidates = [default_findings_path(media_path)]
    if project_dir is not None:
        candidates.append(project_dir / "cut" / "cut.findings.json")
    for path in dict.fromkeys(candidates):
        name = _short_name(path, project_dir)
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        except OSError:
            first_error = first_error or f"{name}: cannot read findings file"
            continue
        data, error = _read_cached(path, stat_identity(info))
        if error is not None:
            first_error = first_error or f"{name}: {error}"
            continue
        if data is None:
            continue
        if data["video"]["sha256"] != sha256:
            result["stale"].append(name)
            continue
        if result["available"]:
            continue
        items = sorted(deepcopy(data["findings"]), key=_item_order)
        result.update(available=True, file=name, items=[
            {**item, "target": note_target(item["kind"])} for item in items
        ])
    if project_dir is not None:
        items, summary, stale = _layer_findings(project_dir, sha256)
        items.extend(_join_findings(project_dir, sha256, version_id))
        result["items"] = sorted([*result["items"], *items], key=_item_order)
        if summary:
            result["layer_checks"] = summary
        if stale:
            result["layer_stale"] = stale
        result["available"] = result["available"] or bool(items) or bool(summary)
    if not result["available"] and first_error is not None:
        result["error"] = first_error
    return result
