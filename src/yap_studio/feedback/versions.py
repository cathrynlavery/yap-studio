"""Save immutable render and input snapshots with strict version manifests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from yap_studio.errors import InputError, ValidationError, YapError
from yap_studio.feedback.audio import media_envelope, parse_envelope
from yap_studio.plan import snap_frame
from yap_studio.project import load_project
from yap_studio.project.lock import project_lock
from yap_studio.project.model import CutWords, Edit, parse_document
from yap_studio.project.receipts import projection_sha256, read_receipt
from yap_studio.project.status import ProjectContext
from yap_studio.receipt import sha256_file
from yap_studio.review.media import probe_media

VERSION_SCHEMA = "yap-studio/version@1"
LAYERS_SCHEMA = "yap-studio/layer-windows@1"
_PIECES_SCHEMA = "yap-studio/version-pieces@1"
_VERSION_ID = re.compile(r"v([1-9][0-9]*)\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RULE_ID = re.compile(r"r-[0-9a-f]{12}\Z")
_LAYER_KINDS = ("caption", "title", "chapter", "graphic", "insert", "overlay", "background",
                "zoom", "caption_gap", "music_level")


class VersionError(YapError):
    """Refuse a version whose render or cut prerequisites cannot be trusted."""

    def __init__(self, reason: str, *, suggested_action: str = "Run yap-studio cut first.") -> None:
        super().__init__(reason, error_type="version_error", code="version_refused",
                         suggested_action=suggested_action)


class PilotVersionError(VersionError):
    """A pilot media refusal uses the refusal exit code at the CLI."""


@dataclass(frozen=True)
class LayerWindow:
    """A unique output-time layer window preserved with the reviewed render."""

    id: str
    kind: str
    start: float
    end: float
    text: str | None


@dataclass(frozen=True)
class VersionManifest:
    """A saved version folder and the identities of its immutable snapshots."""

    path: Path
    project: str
    id: str
    parent: str | None
    created_at: str
    note: str | None
    media: dict
    qa: dict | None
    snapshots: dict

    @property
    def number(self) -> int:
        """Return the numeric version sequence position."""
        return int(self.id[1:])

    @property
    def media_path(self) -> Path:
        """Return the preserved render's absolute path."""
        return self.path / self.media["file"]

    def to_dict(self) -> dict:
        """Return the manifest JSON without the runtime folder path."""
        return copy.deepcopy({
            "schema": VERSION_SCHEMA,
            "project": self.project,
            "id": self.id,
            "parent": self.parent,
            "created_at": self.created_at,
            "note": self.note,
            "media": self.media,
            "qa": self.qa,
            "snapshots": self.snapshots,
        })


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise InputError(str(path), "cannot read valid JSON") from exc


def _fields(data: object, keys: set[str]) -> dict:
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError(f"expected exactly these fields: {', '.join(sorted(keys))}")
    return data


def _string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _cut_inputs(data: object) -> dict:
    if not isinstance(data, dict) or any(
        not isinstance(key, str) or not _digest(value) for key, value in data.items()
    ):
        raise ValueError("inputs must be an object of strings to SHA256 digests")
    return data


def applied_removals_sha256(project_dir: Path) -> str:
    """Hash exactly the applied removals projection used by cut_inputs, in file order."""
    root = Path(project_dir)
    edit = cast(Edit, parse_document("edit", cast(dict, _read_json(root / "edit.json"))))
    referenced = {segment.take for segment in edit.segments}
    path = root / "removals.json"
    data = _read_json(path) if path.exists() else {"schema": "yap-studio/removals@1",
                                                "removals": []}
    parse_document("removals", cast(dict, data))
    return projection_sha256([
        {key: item[key] for key in ("id", "take", "start", "end")}
        for item in cast(dict, data)["removals"]
        if item["status"] == "applied" and item["take"] in referenced
    ])


def verify_cut(project_dir: Path, receipt: dict | None) -> dict[str, str]:
    """Verify feedback's cut inputs and outputs before using or copying them."""
    root = Path(project_dir)
    action = "Run yap-studio cut again, then save the version."
    if receipt is None or receipt.get("status") != "verified":
        raise VersionError("A verified cut/receipt.json is required.")
    changed, verified = [], {}
    inputs = receipt.get("inputs")
    try:
        inputs = _cut_inputs(inputs)
    except ValueError:
        changed.append("cut/receipt.json has missing or invalid inputs")
        inputs = {}
    edit_path = root / "edit.json"
    if not edit_path.is_file() or sha256_file(edit_path) != inputs.get("edit.json"):
        changed.append("edit.json changed after the verified cut")
    else:
        verified["edit.json"] = inputs["edit.json"]
    try:
        removals_hash = applied_removals_sha256(root)
    except (YapError, KeyError, TypeError, ValueError):
        removals_hash = None
    if removals_hash is None or removals_hash != inputs.get("removals:applied"):
        changed.append("removals.json changed after the verified cut")
    if (root / "removals.json").is_file():
        verified["removals.json"] = sha256_file(root / "removals.json")
    outputs = receipt.get("outputs")
    filenames = ["cut/cut.mp4", "cut/spans.json", "cut/cut-words.json"]
    recorded_voice = isinstance(outputs, list) and any(
        isinstance(item, dict) and item.get("path") == "cut/voice.wav"
        for item in outputs
    )
    if (root / "cut" / "voice.wav").exists() or recorded_voice:
        filenames.append("cut/voice.wav")
    if isinstance(outputs, list) and any(
        isinstance(item, dict) and item.get("path") == "cut/verify.json" for item in outputs
    ):
        filenames.append("cut/verify.json")
    for filename in filenames:
        hashes = [item.get("sha256") for item in outputs
                  if isinstance(item, dict) and item.get("path") == filename
                  ] if isinstance(outputs, list) else []
        path = root / filename
        digest = sha256_file(path) if path.is_file() else None
        if digest is None or hashes != [digest]:
            changed.append(f"{filename} changed after the verified cut")
        else:
            verified[filename] = digest
    if changed:
        raise VersionError("; ".join(changed) + ".", suggested_action=action)
    return verified


def _seconds(value: object) -> bool:
    if type(value) not in (int, float) or value < 0:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _relative(value: object) -> bool:
    return _string(value) and not Path(value).is_absolute() and ".." not in Path(value).parts


def parse_layers(data: object, *, source: str = "layers") -> tuple[LayerWindow, ...]:
    """Validate unique layer ids, known kinds, and finite output-time windows."""
    try:
        data = _fields(data, {"schema", "time_base", "layers"})
        if data["schema"] != LAYERS_SCHEMA or data["time_base"] != "output":
            raise ValueError("expected output-time layer windows")
        if not isinstance(data["layers"], list):
            raise ValueError("layers must be an array")
        result, ids = [], set()
        for index, raw in enumerate(data["layers"]):
            item = _fields(raw, {"id", "kind", "start", "end", "text"})
            if not _string(item["id"]) or item["id"] in ids:
                raise ValueError(f"layers[{index}].id must be unique and nonempty")
            ids.add(item["id"])
            if item["kind"] not in _LAYER_KINDS:
                raise ValueError(f"layers[{index}].kind is not a layer kind")
            if not _seconds(item["start"]) or not _seconds(item["end"]):
                raise ValueError(f"layers[{index}] times must be finite and non-negative")
            if item["end"] <= item["start"]:
                raise ValueError(f"layers[{index}].end must be greater than start")
            if item["text"] is not None and not isinstance(item["text"], str):
                raise ValueError(f"layers[{index}].text must be a string or null")
            result.append(LayerWindow(item["id"], item["kind"], float(item["start"]),
                                      float(item["end"]), item["text"]))
        return tuple(result)
    except (TypeError, ValueError) as exc:
        raise ValidationError(source, str(exc)) from exc


def _snapshot(data: object, filename: str) -> dict:
    data = _fields(data, {"file", "sha256"})
    if data["file"] != filename or not _digest(data["sha256"]):
        raise ValueError(f"invalid snapshot for {filename}")
    return data


def _taste_snapshot(data: object) -> dict:
    data = _fields(data, {"revision", "sha256", "applied"})
    if type(data["revision"]) is not int or data["revision"] < 0:
        raise ValueError("taste revision must be an integer at least 0")
    if not _digest(data["sha256"]):
        raise ValueError("taste sha256 must be a SHA256 digest")
    applied = data["applied"]
    if not isinstance(applied, list) or any(
        not isinstance(identity, str) or _RULE_ID.fullmatch(identity) is None
        for identity in applied
    ):
        raise ValueError("taste applied must be an array of rule ids")
    if len(set(applied)) != len(applied):
        raise ValueError("taste applied rule ids must be unique")
    return copy.deepcopy(data)


def _render_taste_pin(taste: Mapping[str, Any], source_hash: str) -> dict:
    try:
        if not isinstance(taste, Mapping):
            raise ValueError("taste pin must be an object")
        if "media_sha256" in taste and taste["media_sha256"] != source_hash:
            raise ValueError("taste record media hash does not match the saved media")
        if not isinstance(taste["applied"], list):
            raise ValueError("taste applied must be an array")
        applied = []
        for item in taste["applied"]:
            if isinstance(item, Mapping):
                item = _fields(dict(item), {"id", "field"})
                if not _string(item["field"]):
                    raise ValueError("taste applied field must be a nonempty string")
                item = item["id"]
            applied.append(item)
        return _taste_snapshot({
            "revision": taste["revision"], "sha256": taste["sha256"], "applied": applied,
        })
    except (KeyError, TypeError, ValueError) as exc:
        raise VersionError(
            f"Invalid render taste pin: {exc}.",
            suggested_action="Render again to write a matching taste record, "
                             "then save the version.",
        ) from exc


def _manifest(path: Path) -> VersionManifest:
    filename = path / "manifest.json"
    data = _read_json(filename)
    try:
        data = _fields(data, {
            "schema", "project", "id", "parent", "created_at", "note", "media", "qa", "snapshots"
        })
        if data["schema"] != VERSION_SCHEMA or not _string(data["project"]):
            raise ValueError("invalid schema or project name")
        if data["id"] != path.name or _VERSION_ID.fullmatch(path.name) is None:
            raise ValueError("id must match the version folder")
        number = int(path.name[1:])
        expected_parent = f"v{number - 1}" if number > 1 else None
        if data["parent"] != expected_parent:
            raise ValueError(f"parent must be {expected_parent}")
        stamp = data["created_at"]
        if (not isinstance(stamp, str)
                or re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", stamp) is None):
            raise ValueError("created_at must be a UTC timestamp")
        datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
        if data["note"] is not None and not isinstance(data["note"], str):
            raise ValueError("note must be a string or null")
        media = _fields(data["media"], {
            "file", "source", "sha256", "fps", "frame_count", "duration_s"
        })
        if (not _relative(media["file"])
                or re.fullmatch(r"media(?:\.[^/\\]+)?", media["file"]) is None):
            raise ValueError("media.file must name the preserved media")
        if not _relative(media["source"]):
            raise ValueError("media.source must be relative to the project without '..' parts")
        if not _digest(media["sha256"]) or not isinstance(media["fps"], str):
            raise ValueError("invalid media hash or fps")
        if Fraction(media["fps"]) <= 0:
            raise ValueError("media fps must be positive")
        if type(media["frame_count"]) is not int or media["frame_count"] < 1:
            raise ValueError("media frame_count must be a positive integer")
        if not _seconds(media["duration_s"]) or media["duration_s"] <= 0:
            raise ValueError("media duration must be positive and finite")
        snapshot_keys = {
            "edit", "removals", "spans", "cut_words", "cut_pieces", "timeline", "layers",
            "style", "taste"
        }
        # Earlier versions have no delivery evidence or saved check report.
        for optional in ("voice_envelope", "verify", "caption_layout"):
            if isinstance(data["snapshots"], dict) and optional in data["snapshots"]:
                snapshot_keys.add(optional)
        snapshots = _fields(data["snapshots"], snapshot_keys)
        if snapshots.get("voice_envelope") is not None:
            _snapshot(snapshots["voice_envelope"], "voice-envelope.json")
        if snapshots.get("verify") is not None:
            _snapshot(snapshots["verify"], "verify.json")
        if snapshots.get("caption_layout") is not None:
            _snapshot(snapshots["caption_layout"], "caption_layout.json")
        for key, name in (("edit", "edit.json"), ("spans", "spans.json"),
                          ("cut_words", "cut-words.json"), ("cut_pieces", "cut-pieces.json")):
            _snapshot(snapshots[key], name)
        for key in ("removals", "layers"):
            if snapshots[key] is not None:
                _snapshot(snapshots[key], f"{key}.json")
        if snapshots["taste"] is not None:
            _taste_snapshot(snapshots["taste"])
        if snapshots["timeline"] is not None:
            timeline = _fields(snapshots["timeline"], {"lanes"})
            if not isinstance(timeline["lanes"], dict) or not timeline["lanes"]:
                raise ValueError("timeline lanes must be a nonempty object")
            for lane, item in timeline["lanes"].items():
                if not isinstance(lane, str) or re.fullmatch(r"[a-zA-Z0-9_-]+", lane) is None:
                    raise ValueError("invalid timeline lane")
                _snapshot(item, f"timeline/{lane}.json")
        if snapshots["style"] is not None:
            style = _fields(snapshots["style"], {"ref", "file", "sha256"})
            if not _string(style["ref"]):
                raise ValueError("style ref must be nonempty")
            if style["ref"].startswith("preset:"):
                if not style["ref"][7:] or style["file"] is not None or style["sha256"] is not None:
                    raise ValueError("preset style must have null file and sha256")
            else:
                if not _relative(style["ref"]):
                    raise ValueError("style ref must be relative to the project")
                _snapshot({key: style[key] for key in ("file", "sha256")}, "style.json")
        if data["qa"] is not None:
            qa = _fields(data["qa"], {"file", "sha256", "status"})
            _snapshot({key: qa[key] for key in ("file", "sha256")}, "qa-report.json")
        return VersionManifest(path.resolve(), data["project"], data["id"], data["parent"],
                               stamp, data["note"], media, data["qa"], snapshots)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise InputError(str(filename), str(exc)) from exc


def load_versions(project_dir: Path) -> tuple[VersionManifest, ...]:
    """Read every numbered manifest, validating a contiguous parent chain."""
    root = Path(project_dir).resolve() / "versions"
    if not root.exists():
        return ()
    try:
        folders = [item for item in root.iterdir() if _VERSION_ID.fullmatch(item.name)]
    except OSError as exc:
        raise InputError(str(root), "cannot read version folders") from exc
    folders.sort(key=lambda item: int(item.name[1:]))
    versions = []
    for number, folder in enumerate(folders, 1):
        manifest = _manifest(folder)
        if manifest.number != number:
            raise InputError(str(folder / "manifest.json"), f"missing version v{number}")
        versions.append(manifest)
    return tuple(versions)


def load_version(project_dir: Path, version_id: str) -> VersionManifest:
    """Load a version only from a valid immutable version chain."""
    if not isinstance(version_id, str) or _VERSION_ID.fullmatch(version_id) is None:
        raise ValidationError("version_id", "must be v followed by a positive integer")
    for manifest in load_versions(project_dir):
        if manifest.id == version_id:
            return manifest
    raise InputError(str(Path(project_dir) / "versions" / version_id / "manifest.json"))


def current_version(project_dir: Path) -> VersionManifest | None:
    """Read the rollback pointer, falling back to the newest saved version."""
    path = Path(project_dir) / "versions" / "current.json"
    if not path.exists():
        versions = load_versions(project_dir)
        return versions[-1] if versions else None
    try:
        data = _fields(_read_json(path), {"schema", "current", "set_at", "reason"})
        if data["schema"] != "yap-studio/version-pointer@1" or data["reason"] != "rollback":
            raise ValueError("invalid version pointer")
        datetime.strptime(data["set_at"], "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as exc:
        raise InputError(str(path), str(exc)) from exc
    return load_version(project_dir, data["current"])


def _parse_pieces(data: object) -> dict:
    keys = {"schema", "fps", "total_frames", "pieces"}
    if isinstance(data, dict) and "inputs" in data:
        keys.add("inputs")
    if isinstance(data, dict) and "listen_joins" in data:
        keys.add("listen_joins")
    data = _fields(data, keys)
    if "inputs" in data:
        _cut_inputs(data["inputs"])
    if "listen_joins" in data and not isinstance(data["listen_joins"], list):
        raise ValueError("listen_joins must be an array")
    if data["schema"] != _PIECES_SCHEMA or not isinstance(data["fps"], str):
        raise ValueError("invalid pieces schema or fps")
    if Fraction(data["fps"]) <= 0:
        raise ValueError("pieces fps must be positive")
    if type(data["total_frames"]) is not int or data["total_frames"] < 1:
        raise ValueError("total_frames must be a positive integer")
    if not isinstance(data["pieces"], list):
        raise ValueError("pieces must be an array")
    previous_end = 0
    for raw in data["pieces"]:
        item = _fields(raw, {"segment", "take", "out_f0", "n_frames", "src_t0", "src_t1"})
        if not _string(item["segment"]) or not _string(item["take"]):
            raise ValueError("piece segment and take must be nonempty")
        if type(item["out_f0"]) is not int or item["out_f0"] < previous_end:
            raise ValueError("pieces must be in output order without overlaps")
        if type(item["n_frames"]) is not int or item["n_frames"] < 1:
            raise ValueError("piece n_frames must be positive")
        previous_end = item["out_f0"] + item["n_frames"]
        if previous_end > data["total_frames"]:
            raise ValueError("piece exceeds total_frames")
        if not isinstance(item["src_t0"], str) or not isinstance(item["src_t1"], str):
            raise ValueError("source times must be fraction strings")
        if not 0 <= Fraction(item["src_t0"]) < Fraction(item["src_t1"]):
            raise ValueError("invalid source time range")
    return data


def _receipt_pieces(receipt: dict, path: Path) -> dict:
    try:
        plan = receipt["plan"]
        pieces = {
            "schema": _PIECES_SCHEMA,
            "fps": plan["fps"],
            "total_frames": plan["total_frames"],
            "pieces": [{"segment": item["segment_id"], **{
                key: item[key] for key in ("take", "out_f0", "n_frames", "src_t0", "src_t1")
            }} for item in plan["pieces"]],
            "inputs": receipt["inputs"],
            "listen_joins": plan.get("listen_joins", []),
        }
        return _parse_pieces(pieces)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise InputError(str(path), f"invalid cut plan: {exc}") from exc


def _refuse_pilot_media(root: Path, media: Path) -> None:
    """Keep pilot media out of versions even after it is renamed or copied."""
    from yap_studio.stages.assemble import pilot_record

    source = Path(media).resolve()
    if not source.is_file():
        return
    pilot_receipt = read_receipt(root / "pilot") or {}
    marked = pilot_record(source)
    outputs = pilot_receipt.get("outputs", [])
    digest = sha256_file(source) if outputs and marked is None else None
    owned = digest is not None and any(output["sha256"] == digest for output in outputs)
    inside = source.is_relative_to(root / "pilot")
    if marked or owned or inside:
        window = (marked or pilot_receipt.get("pilot") or {}).get("frames")
        label = f"frames {window[0]}-{window[1]}" if window else "unknown frames"
        reference = str(source.relative_to(root)) if source.is_relative_to(root) else source.name
        raise PilotVersionError(
            f"{reference} is a pilot render ({label}); pilots are never saved as versions.",
            suggested_action="Render the full final with yap-studio assemble, then save it.",
        )


def save_version(
    project_dir: Path,
    *,
    media: Path | None = None,
    note: str | None = None,
    layers: Path | None = None,
    now: Callable[[], datetime] | None = None,
    taste: Mapping[str, Any] | None = None,
) -> VersionManifest:
    """Publish immutable snapshots, pinning supplied taste or the explicit media's record.

    The render pipeline should pass the full record to preserve it with the media.
    A short pin preserves only manifest identities, without a taste record file.
    """
    root = Path(project_dir).resolve()
    if note is not None and not isinstance(note, str):
        raise ValidationError("note", "must be a string or null")
    with project_lock(root):
        from yap_studio.feedback.release import _complete_pending_rollback

        _complete_pending_rollback(root)
        if media is not None:
            _refuse_pilot_media(root, media)
        project = load_project(root / "project.json")
        receipt_path = root / "cut" / "receipt.json"
        cut_receipt = read_receipt(root / "cut")
        if cut_receipt is None or cut_receipt.get("status") != "verified":
            raise VersionError("A verified cut/receipt.json is required to save a version.")
        verified = verify_cut(root, cut_receipt)
        pieces = _receipt_pieces(cut_receipt, receipt_path)
        fps = Fraction(pieces["fps"])
        source = Path(media).resolve() if media is not None else root / "cut" / "cut.mp4"
        source_ref = str(source.relative_to(root)) if source.is_relative_to(root) else source.name
        if not source.is_file():
            raise VersionError(f"{os.path.relpath(source, root)} is missing.")
        is_cut = source == root / "cut" / "cut.mp4"
        source_hash = verified["cut/cut.mp4"] if is_cut else sha256_file(source)
        # Import only when saving a version.
        from yap_studio.taste.render import read_taste_record

        if taste is None and media is not None:
            try:
                taste = read_taste_record(Path(media))
            except InputError as exc:
                raise VersionError(
                    f"Cannot save render taste: {exc.reason}.",
                    suggested_action="Render again to write a matching taste record, "
                                     "then save the version.",
                ) from exc
        taste_pin = _render_taste_pin(taste, source_hash) if taste is not None else None
        expected_frames = pieces["total_frames"]
        if media is not None:
            if not is_cut:
                edit_data = cast(dict, _read_json(root / "edit.json"))
                edit = cast(Edit, parse_document("edit", edit_data))
                hold_s = (edit.end_hold_s if edit.end_hold_s is not None else
                          ProjectContext(root).render_style().data["internal"]["end_hold_s"])
                expected_frames += snap_frame(Fraction(str(hold_s)), fps)
            facts = probe_media(source)
            if (not facts.cfr or facts.fps != fps or facts.duration_s is None
                    or round(facts.duration_s * fps) != expected_frames):
                raise VersionError(
                    f"{os.path.relpath(source, root)}: the render does not match the current cut.",
                    suggested_action="Render the current cut at its plan fps and final "
                                     "frame count.",
                )
        layer_path = Path(layers).resolve() if layers is not None else None
        if layer_path is not None:
            parse_layers(_read_json(layer_path), source=str(layer_path))
        parse_document("cut-words", cast(dict, _read_json(root / "cut" / "cut-words.json")))
        envelope = media_envelope(source)
        previous = load_versions(root)
        if previous and previous[-1].media["sha256"] == source_hash:
            raise VersionError(f"Render unchanged since {previous[-1].id}.",
                               suggested_action="Make and render changes before saving a version.")
        version_id = f"v{previous[-1].number + 1}" if previous else "v1"
        versions_dir = root / "versions"
        versions_dir.mkdir(exist_ok=True)
        temporary = versions_dir / f".{version_id}.tmp-{uuid.uuid4().hex}"
        destination = versions_dir / version_id
        temporary.mkdir()
        try:
            def snapshot(source_path: Path, filename: str) -> dict:
                relative = source_path.relative_to(root).as_posix() if (
                    source_path.is_relative_to(root)
                ) else source_path.name
                expected = verified.get(relative)
                if source_path == source:
                    expected = source_hash
                if expected is None:
                    expected = sha256_file(source_path)
                target = temporary / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target)
                digest = sha256_file(target)
                if digest != expected:
                    raise VersionError(f"{relative} changed while it was being copied.",
                                       suggested_action="Run yap-studio cut again, "
                                       "then save the version.")
                return {"file": filename, "sha256": digest}

            media_copy = snapshot(source, f"media{source.suffix}")
            if taste is not None and "schema" in taste:
                preserved_media = temporary / media_copy["file"]
                try:
                    preserved_media.with_suffix(".taste.json").write_text(
                        json.dumps(dict(taste), indent=2, ensure_ascii=False, allow_nan=False)
                        + "\n", encoding="utf-8",
                    )
                    read_taste_record(preserved_media)
                except (InputError, TypeError, ValueError) as exc:
                    reason = exc.reason if isinstance(exc, InputError) else str(exc)
                    raise VersionError(
                        f"Cannot save render taste: {reason}.",
                        suggested_action="Render again to write a matching taste record, "
                                         "then save the version.",
                    ) from exc
            snapshots = {
                "edit": snapshot(root / "edit.json", "edit.json"),
                "removals": None,
                "spans": snapshot(root / "cut" / "spans.json", "spans.json"),
                "cut_words": snapshot(root / "cut" / "cut-words.json", "cut-words.json"),
                "cut_pieces": None,
                "timeline": None,
                "layers": None,
                "style": None,
                "taste": taste_pin,
                "voice_envelope": None,
                "verify": None,
            }
            if (root / "cut" / "verify.json").is_file():
                snapshots["verify"] = snapshot(root / "cut" / "verify.json", "verify.json")
            layout_path = source.with_suffix(".layout.json")
            if not is_cut and layout_path.is_file():
                layout = _read_json(layout_path)
                if (isinstance(layout, dict) and layout.get("media_sha256") == source_hash
                        and isinstance(layout.get("caption_phrases"), dict)):
                    snapshots["caption_layout"] = snapshot(layout_path, "caption_layout.json")
            if envelope is not None:
                envelope_path = temporary / "voice-envelope.json"
                envelope_path.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
                snapshots["voice_envelope"] = {
                    "file": envelope_path.name, "sha256": sha256_file(envelope_path)
                }
            pieces_path = temporary / "cut-pieces.json"
            pieces_path.write_text(json.dumps(pieces, indent=2) + "\n", encoding="utf-8")
            snapshots["cut_pieces"] = {
                "file": "cut-pieces.json", "sha256": sha256_file(pieces_path)
            }
            if "removals.json" in verified:
                snapshots["removals"] = snapshot(root / "removals.json", "removals.json")
            lanes = {}
            for lane_path in sorted((root / "timeline").glob("*.json")):
                if re.fullmatch(r"[a-zA-Z0-9_-]+", lane_path.stem) is None:
                    raise InputError(str(lane_path), "invalid timeline lane name")
                lanes[lane_path.stem] = snapshot(lane_path, f"timeline/{lane_path.name}")
            if lanes:
                snapshots["timeline"] = {"lanes": lanes}
            if layer_path is not None:
                snapshots["layers"] = snapshot(layer_path, "layers.json")
                parse_layers(_read_json(temporary / "layers.json"), source=str(layer_path))
            if project.style is not None:
                if project.style.startswith("preset:"):
                    snapshots["style"] = {"ref": project.style, "file": None, "sha256": None}
                else:
                    if not _relative(project.style):
                        raise InputError(
                            "project.json", "style must be a path relative to the project"
                        )
                    snapshots["style"] = {"ref": project.style,
                                          **snapshot(root / project.style, "style.json")}
            qa = None
            if (root / "qa" / "report.json").is_file():
                qa = snapshot(root / "qa" / "report.json", "qa-report.json")
                report = _read_json(temporary / "qa-report.json")
                if not isinstance(report, dict):
                    raise InputError("qa/report.json", "expected a JSON object")
                qa["status"] = report.get("status")
            value = now() if now is not None else datetime.now(UTC)
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            manifest = VersionManifest(
                destination, project.name, version_id, previous[-1].id if previous else None,
                value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), note,
                {**media_copy, "source": source_ref, "fps": pieces["fps"],
                 "frame_count": expected_frames,
                 "duration_s": round(float(expected_frames / fps), 4)},
                qa, snapshots,
            )
            (temporary / "manifest.json").write_text(
                json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            if destination.exists():
                raise VersionError(f"{version_id} appeared while saving the version.",
                                   suggested_action="Reload saved versions and retry.")
            try:
                os.rename(temporary, destination)
            except OSError as exc:
                if destination.exists():
                    raise VersionError(
                        f"{version_id} appeared while saving the version.",
                        suggested_action="Reload saved versions and retry.",
                    ) from exc
                raise
            (versions_dir / "current.json").unlink(missing_ok=True)
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def _snapshot_json(manifest: VersionManifest, key: str) -> object:
    entry = manifest.qa if key == "qa" else manifest.snapshots[key]
    path = manifest.path / entry["file"]
    try:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise InputError(str(path), "snapshot changed after it was saved")
        return json.loads(data)
    except (OSError, UnicodeError, ValueError) as exc:
        raise InputError(str(path), "cannot read valid JSON snapshot") from exc


def version_report(manifest: VersionManifest) -> dict | None:
    """Read hash-checked QA and cut reports with a normalized top-level status."""
    qa = _snapshot_json(manifest, "qa") if manifest.qa is not None else None
    verify = _snapshot_json(manifest, "verify") if manifest.snapshots.get("verify") else None
    if qa is None and verify is None:
        return None
    for name, report in (("qa", qa), ("verify", verify)):
        if report is not None and not isinstance(report, dict):
            raise InputError(str(manifest.path), f"{name} report must be a JSON object")
    statuses = {}
    for name, report in (("qa", qa), ("verify", verify)):
        if report is not None:
            status = report.get("status")
            normalized = status.lower() if isinstance(status, str) else None
            statuses[name] = normalized if normalized in {"pass", "warn", "fail"} else None
    valid = [value for value in statuses.values() if value is not None]
    worst = max(valid, key={"pass": 0, "warn": 1, "fail": 2}.__getitem__) if valid else None
    return {"status": worst if all(statuses.values()) else None, "statuses": statuses,
            "subject": "render" if manifest.media["source"] == "cut/cut.mp4" else "cut",
            "qa": qa, "verify": verify}


def complete_pending_rollback(project_dir: Path) -> dict | None:
    """Recover the pending input transaction before using a project version."""
    from yap_studio.feedback.release import complete_pending_rollback as complete

    return complete(project_dir)


def version_cut_words(manifest: VersionManifest) -> CutWords:
    """Read the preserved word occurrences for a saved version."""
    path = manifest.path / manifest.snapshots["cut_words"]["file"]
    return cast(CutWords, parse_document(
        "cut-words", cast(dict, _snapshot_json(manifest, "cut_words")), filename=str(path)
    ))


def version_pieces(manifest: VersionManifest) -> dict:
    """Read and validate the saved cut receipt's output piece map."""
    path = manifest.path / manifest.snapshots["cut_pieces"]["file"]
    try:
        return _parse_pieces(_snapshot_json(manifest, "cut_pieces"))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise InputError(str(path), str(exc)) from exc


def version_layers(manifest: VersionManifest) -> tuple[LayerWindow, ...] | None:
    """Read preserved layer windows, or return None when none were supplied."""
    entry = manifest.snapshots["layers"]
    if entry is None:
        return None
    path = manifest.path / entry["file"]
    return parse_layers(_snapshot_json(manifest, "layers"), source=str(path))


def version_envelope(manifest: VersionManifest) -> dict | None:
    """Read verified voice delivery evidence, or None for an unmeasured version."""
    if manifest.snapshots.get("voice_envelope") is None:
        return None
    return parse_envelope(_snapshot_json(manifest, "voice_envelope"))


def find_version_by_media(project_dir: Path, sha256: str) -> VersionManifest | None:
    """Find a saved version by its recorded media hash without hashing the copy."""
    return next(
        (item for item in load_versions(project_dir) if item.media["sha256"] == sha256), None
    )
