"""Extensible stage projections and the project's agent-facing next action."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from yap_studio import paths
from yap_studio.canvas import canvas
from yap_studio.errors import InputError, YapError
from yap_studio.project.anchors import edit_needs_edge_engine
from yap_studio.project.lock import pid_alive, project_lock
from yap_studio.project.model import Edit, Project, Removals, Words, parse_document
from yap_studio.project.receipts import (
    AudioStreamMissingError,
    IdentityCache,
    _read_json,
    failed_assert_lines,
    projection_sha256,
    read_receipt,
    receipt_output_paths,
)
from yap_studio.transcript import resolve_backend, shared

if TYPE_CHECKING:
    from yap_studio.stages.timeline import ResolvedTimeline
    from yap_studio.style import ResolvedStyle
    from yap_studio.taste.resolve import TasteApplication


class TakeChangedError(YapError):
    """Refuse a changed take whose recorded audio cannot validate a stand-in."""

    def __init__(self, take_id: str, path: str, *, since: str, detail: str) -> None:
        super().__init__(
            f"Take {take_id} ({path}) changed since {since}: {detail}.",
            error_type="take_changed",
            code="take_changed",
            suggested_action=(
                f"Put the original file back at {path}, or run yap-studio ingest DIR --force "
                "to transcribe the new file, update edit.json, then run yap-studio cut DIR."
            ),
        )


class ProjectContext:
    """One project snapshot, with memoized bytes for every small input file."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir).resolve()
        self._snapshots: dict[str, bytes | None] = {}
        self._take_hashes: dict[str, str] = {}
        self._take_audio_hashes: dict[tuple[str, int], str] = {}
        self.project = cast(Project, parse_document("project", self.json("project.json")))
        self.identity = IdentityCache(self.project_dir)
        self._render_style: ResolvedStyle | None = None
        self._taste_application: TasteApplication | None = None
        self._style_path: Path | None = None
        self._timeline: ResolvedTimeline | None = None

    def read(self, relative: str) -> bytes:
        """Read each input exactly once for this context."""
        if relative not in self._snapshots:
            try:
                self._snapshots[relative] = (self.project_dir / relative).read_bytes()
            except FileNotFoundError:
                self._snapshots[relative] = None
        snapshot = self._snapshots[relative]
        if snapshot is None:
            raise FileNotFoundError(errno.ENOENT, "missing snapshot input",
                                    str(self.project_dir / relative))
        return snapshot

    def json(self, relative: str) -> dict[str, Any]:
        """Decode bytes retained in this snapshot."""
        return json.loads(self.read(relative))

    def take_path(self, take_id: str) -> Path:
        """Resolve a take id to its read-only source."""
        take = next((take for take in self.project.takes if take.id == take_id), None)
        if take is None:
            from yap_studio.project.model import ProjectFileError

            raise ProjectFileError("edit.json", "segments.take", f"unknown take {take_id!r}")
        try:
            manifest = self.json("ingest/ingest.json")
        except FileNotFoundError:
            manifest = {}
        for item in manifest.get("takes", []):
            if (item.get("id") != take_id or item.get("path") != take.path
                    or not item.get("copy")):
                continue
            copied = (self.project_dir / item["copy"]).resolve()
            if not copied.is_relative_to(self.project_dir / "ingest/takes"):
                raise InputError(take_id, "recorded copy is outside ingest/takes; ingest again")
            if self.identity.sha256(copied) != item.get("sha256"):
                raise InputError(take_id,
                                 "recorded copy hash mismatch; restore it and ingest again")
            return copied
        return self.project_dir / take.path

    def take_sha256(self, take_id: str) -> str:
        """Retain a take's identity hash from this input snapshot."""
        if take_id not in self._take_hashes:
            self._take_hashes[take_id] = self.identity.sha256(self.take_path(take_id))
        return self._take_hashes[take_id]

    def selected_audio_stream(self, take_id: str) -> int:
        """Select the same audio ordinal for rendering and source verification."""
        take = next(take for take in self.project.takes if take.id == take_id)
        selected = take.audio_stream if take.audio_stream is not None else 0
        try:
            manifest = self.json("ingest/ingest.json")
        except FileNotFoundError:
            manifest = {}
        for item in manifest.get("takes", []):
            if (item.get("id") == take_id
                    and item.get("requested_audio_stream") == take.audio_stream):
                selected = item.get("audio_stream", selected)
        return selected

    def ingested_take(self, take_id: str) -> dict[str, Any] | None:
        """Return a take's recorded ingest facts from this snapshot."""
        try:
            manifest = self.json("ingest/ingest.json")
        except FileNotFoundError:
            return None
        return next((item for item in manifest.get("takes", []) if item.get("id") == take_id),
                    None)

    def take_audio_sha256(self, take_id: str, ordinal: int) -> str:
        """Retain a selected audio stream's hash from this input snapshot."""
        key = (take_id, ordinal)
        if key not in self._take_audio_hashes:
            self._take_audio_hashes[key] = self.identity.audio_sha256(self.take_path(take_id),
                                                                    ordinal)
        return self._take_audio_hashes[key]

    def check_take(self, take_id: str, recorded: Mapping[str, Any], *, since: str) -> bool:
        """Accept a changed picture source only when its encoded audio is identical."""
        if self.take_sha256(take_id) == recorded.get("sha256"):
            return False
        take = next(take for take in self.project.takes if take.id == take_id)
        audio_hash = recorded.get("audio_sha256")
        if isinstance(audio_hash, str):
            try:
                current_audio_hash = self.take_audio_sha256(take_id, recorded["audio_stream"])
            except AudioStreamMissingError as exc:
                raise TakeChangedError(
                    take_id, take.path, since=since, detail="its audio stream differs",
                ) from exc
            if audio_hash == current_audio_hash:
                return True
            detail = "its audio stream differs"
        else:
            detail = "no audio-stream hash was recorded, so a stand-in cannot be accepted"
        raise TakeChangedError(take_id, take.path, since=since, detail=detail)

    def take_identity(self, take_id: str) -> str:
        """Bind audio-only consumers to ingest when a stand-in preserves its audio."""
        recorded = self.ingested_take(take_id)
        if recorded is not None:
            try:
                self.check_take(take_id, recorded, since="ingest")
            except TakeChangedError:
                pass
            else:
                return recorded["sha256"]
        return self.take_sha256(take_id)

    def take_fingerprint(self, take_id: str) -> dict[str, Any]:
        """Record the current file and selected encoded audio for a picture consumer."""
        digest = self.take_sha256(take_id)
        ordinal = self.selected_audio_stream(take_id)
        recorded = self.ingested_take(take_id)
        audio_hash = None
        if (recorded is not None and recorded.get("sha256") == digest
                and recorded.get("audio_stream") == ordinal):
            audio_hash = recorded.get("audio_sha256")
        if not isinstance(audio_hash, str):
            audio_hash = self.take_audio_sha256(take_id, ordinal)
        take = next(take for take in self.project.takes if take.id == take_id)
        return {"path": take.path, "sha256": digest, "audio_stream": ordinal,
                "audio_sha256": audio_hash}

    def render_style(self) -> ResolvedStyle:
        """Snapshot the base style, approved taste and enforced schema limits."""
        from yap_studio.style import (
            PRESETS_DIR,
            SHIPPED_PRESETS,
            ResolvedStyle,
            StyleError,
            library_dir,
            resolve_style_data,
        )
        from yap_studio.taste.render import render_style

        if self._render_style is None:
            spec = self.project.style or "preset:clean_minimal"
            local = self.project_dir / spec
            if local.is_file():
                target = local
            elif spec.startswith("preset:"):
                preset = spec.removeprefix("preset:")
                if preset not in SHIPPED_PRESETS:
                    raise StyleError("style", "choose a shipped preset or saved style")
                target = PRESETS_DIR / f"{preset}.json"
            elif (re.fullmatch(r"[a-z0-9_-]{1,64}", spec)
                  and (library_dir() / f"{spec}.json").is_file()):
                target = library_dir() / f"{spec}.json"
            else:
                target = Path(spec).expanduser()
                if not target.is_absolute():
                    target = self.project_dir / target
            self._style_path = paths.validate_input(target)
            data = resolve_style_data(self.json(str(self._style_path)))
            # project.json has no overrides for taste-supported style fields.
            self._taste_application = render_style(ResolvedStyle(data["name"], spec, data),
                                                   explicit=None)
            self._render_style = self._taste_application.style
            if self.project.shape == "horizontal":
                data = copy.deepcopy(self._render_style.data)
                frame = canvas(self.project.shape)
                data["captions"].update(position="bottom", placement="fixed",
                                        top_px=frame.caption_top)
                data["internal"]["caption_safe_bottom_px"] = frame.caption_safe_bottom
                self._render_style = replace(self._render_style, data=data)
        return self._render_style

    def taste_application(self) -> TasteApplication:
        """Return the same taste snapshot used by this context's render style."""
        self.render_style()
        assert self._taste_application is not None
        return self._taste_application

    def taste_inputs(self, *fields: str) -> dict[str, str]:
        """Identify rules for consumed fields, preserving inputs when none apply."""
        applied = [item for item in self.taste_application().pin()["applied"]
                   if any(item["field"] == field or item["field"].startswith(f"{field}.")
                          for field in fields)]
        return {"taste": projection_sha256(applied)} if applied else {}

    def resolved_timeline(self, *, allow_violations: bool = False) -> ResolvedTimeline:
        """Resolve and validate lanes using the same retained words and style bytes."""
        from yap_studio.stages.timeline import _lanes, _resolve

        if self._timeline is None:
            self._timeline = _resolve(self, _lanes(self))
        return self._timeline if allow_violations else self._timeline.require_valid()


@dataclass(frozen=True)
class StageSpec:
    """A stage's folder, prerequisites, and precise input projection."""

    name: str
    folder: str
    after: tuple[str, ...]
    inputs: Callable[[ProjectContext], dict[str, str]]
    human: bool = False


STAGES: list[StageSpec] = []


def register(spec: StageSpec) -> None:
    """Append a stage, or replace its registration while preserving stage order."""
    for index, previous in enumerate(STAGES):
        if previous.name == spec.name:
            STAGES[index] = spec
            return
    STAGES.append(spec)


def ingest_inputs(context: ProjectContext) -> dict[str, str]:
    """Hash takes plus only the ASR and language settings ingest consumes."""
    return {
        **{f"take:{take.id}": context.take_identity(take.id)
           for take in context.project.takes},
        "asr": projection_sha256(context.project.asr),
        "language": projection_sha256(context.project.language),
        "audio_stream": projection_sha256({take.id: take.audio_stream
                                            for take in context.project.takes}),
    }


def _edge_engine_identity(context: ProjectContext) -> dict[str, str | None]:
    """Identify the usable edge engine and model without loading speech weights."""
    settings = context.project.asr or {}
    try:
        backend = resolve_backend(settings.get("engine") or "auto", settings.get("model"))
    except YapError as exc:
        if exc.code != "speech_engine_not_found":
            raise
        backend = None
    return {"engine": backend.name if backend else "none",
            "model": backend.model if backend else None}


def cut_inputs(context: ProjectContext) -> dict[str, str]:
    """Hash the edit snapshot, applied trims, source facts, and edge speech settings."""
    from yap_studio.stages import edges, take_levels

    edit_bytes = context.read("edit.json")
    edit = cast(Edit, parse_document("edit", json.loads(edit_bytes)))
    referenced = sorted({segment.take for segment in edit.segments})
    try:
        removal_data = context.json("removals.json")
    except FileNotFoundError:
        removal_data = {"schema": "yap-studio/removals@1", "removals": []}
    cast(Removals, parse_document("removals", removal_data))
    applied = [
        {key: item[key] for key in ("id", "take", "start", "end")}
        for item in removal_data["removals"]
        if item["status"] == "applied" and item["take"] in referenced
    ]
    try:
        manifest = context.json("ingest/ingest.json")
    except FileNotFoundError:
        manifest = {}
    selected_streams = {item["id"]: item.get("audio_stream") for item in manifest.get("takes", [])}
    source_words = {
        take: cast(Words, parse_document("words", context.json(f"ingest/{take}/words.json"),
                                         filename=f"ingest/{take}/words.json"))
        for take in referenced
    }
    needs_engine = edit_needs_edge_engine(edit.segments, source_words) or any(
        edges.might_need_silent_word_realign(snapshot.words)
        for snapshot in source_words.values()
    )
    engine_identity = _edge_engine_identity(context) if needs_engine else {"engine": "not-needed"}
    return {
        "edit.json": hashlib.sha256(edit_bytes).hexdigest(),
        **({"project:shape": projection_sha256(context.project.shape)}
           if context.project.shape != "vertical" else {}),
        "removals:applied": projection_sha256(applied),
        "asr": projection_sha256(context.project.asr),
        "asr:resolved": projection_sha256(engine_identity),
        "language": projection_sha256(context.project.language),
        **({"project:platform": projection_sha256(context.project.platform)}
           if context.project.platform != "none" else {}),
        "edges:pause_threshold": projection_sha256(edges.pause_rule_identity()),
        **({"audio:take_levels": projection_sha256({
            "rule": take_levels.rule_identity(),
            "match_takes": take_levels.match_enabled(context.project),
        })} if len(referenced) > 1 else {}),
        "crop_x": projection_sha256({take.id: take.crop_x for take in context.project.takes
                                     if take.id in referenced}),
        "audio_stream": projection_sha256({
            take.id: {"requested": take.audio_stream, "selected": selected_streams.get(take.id)}
            for take in context.project.takes if take.id in referenced
        }),
        **{f"words:{take}": hashlib.sha256(context.read(f"ingest/{take}/words.json")).hexdigest()
           for take in referenced},
        **{f"take:{take}": context.take_sha256(take) for take in referenced},
    }


def removal_decisions(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Project persistent decisions without generated join evidence."""
    return [
        {key: item.get(key) for key in
         ("id", "take", "start", "end", "kind", "words", "status", "by", "reason")}
        for item in data["removals"]
        if (item["status"] == "rejected" or item.get("by") == "manual"
            or item.get("evidence", {}).get("decision"))
    ]


def fillers_inputs(context: ProjectContext) -> dict[str, str]:
    """Hash segment ranges, speech settings, rules and creator decisions."""
    from yap_studio.stages import edges

    edit = context.json("edit.json")
    parse_document("edit", edit)
    takes = sorted({segment["take"] for segment in edit["segments"]})
    try:
        removals = context.json("removals.json")
    except FileNotFoundError:
        removals = {"schema": "yap-studio/removals@1", "removals": []}
    parse_document("removals", removals)
    try:
        manifest = context.json("ingest/ingest.json")
    except FileNotFoundError:
        manifest = {}
    try:
        recorded = context.json("fillers/receipt.json").get("inputs", {}).get("edit:segments")
    except (OSError, ValueError, TypeError, AttributeError):
        recorded = None
    segments_hash = projection_sha256([
        {key: segment[key] for key in ("id", "take", "from", "to") if key in segment}
        for segment in edit["segments"]
    ])
    # Keep an unchanged receipt written with the old full-segment hash.
    if recorded == projection_sha256(edit["segments"]):
        segments_hash = recorded
    return {
        "edit:segments": segments_hash,
        "language": projection_sha256(context.project.language),
        "asr": projection_sha256(context.project.asr),
        "edges:pause_threshold": projection_sha256(edges.pause_rule_identity()),
        "rules": projection_sha256({key: context.project.rules[key]
                                    for key in ("max_pause_s", "beat_breath_s")}),
        "audio_stream": projection_sha256({
            "requested": {take.id: take.audio_stream for take in context.project.takes
                          if take.id in takes},
            "selected": {item["id"]: item.get("audio_stream")
                         for item in manifest.get("takes", []) if item["id"] in takes},
        }),
        "removals:decisions": projection_sha256(removal_decisions(removals)),
        **{f"words:{take}": hashlib.sha256(context.read(f"ingest/{take}/words.json")).hexdigest()
           for take in takes},
        **{f"take:{take}": context.take_identity(take) for take in takes},
    }


def captions_inputs(context: ProjectContext) -> dict[str, str]:
    """Hash only the inputs consumed by the project caption layer."""
    from yap_studio.stages.captions import caption_snapshot

    return caption_snapshot(_layer_projection_context(context)).inputs


def music_inputs(context: ProjectContext) -> dict[str, str]:
    """Hash the track, content and hold lengths, and internal stem settings."""
    from yap_studio.stages.music import music_snapshot

    return music_snapshot(context).inputs


def graphics_inputs(context: ProjectContext) -> dict[str, str]:
    """Hash titles, chapters and cards independently of captions."""
    from yap_studio.stages.graphics import graphics_snapshot

    view = _layer_projection_context(context)
    record = read_receipt(context.project_dir / "graphics")
    placements = (record.get("elements", [])
                  if record and record.get("status") == "verified" else None)
    return graphics_snapshot(view, placements=placements).inputs


def _layer_projection_context(context: ProjectContext) -> ProjectContext:
    """Honor an assembly override only for the exact timeline it verified."""
    try:
        final = context.json("final/receipt.json")
    except FileNotFoundError:
        return context
    if not final.get("allow_violations") or final.get("status") != "verified":
        return context
    timeline = context.resolved_timeline(allow_violations=True)
    if final.get("inputs", {}).get("timeline") != projection_sha256(timeline.to_dict()):
        return context
    view = copy.copy(context)
    view._timeline = replace(timeline, findings=())
    return view


def assemble_inputs(context: ProjectContext) -> dict[str, str]:
    """Hash the exact picture, mix, policy and verification inputs assembly consumes."""
    from yap_studio.stages.assemble import assembly_snapshot

    return assembly_snapshot(context, allow_violations=True).inputs


register(StageSpec("ingest", "ingest", (), ingest_inputs))
register(StageSpec("fillers", "fillers", ("ingest",), fillers_inputs))
register(StageSpec("cut", "cut", ("ingest",), cut_inputs))
register(StageSpec("music", "music", ("cut",), music_inputs))
register(StageSpec("captions", "captions", ("cut",), captions_inputs))
register(StageSpec("graphics", "graphics", ("cut",), graphics_inputs))
register(StageSpec("assemble", "final", ("captions", "graphics", "music"), assemble_inputs))


def human_items(context: ProjectContext) -> list[str]:
    """Approval units can supply human decisions through this hook."""
    from yap_studio import sfx
    from yap_studio.stages.music import music_human_items

    items = music_human_items(context)
    kit = context.render_style().data["music"]["sound_effects"]
    account = sfx.saved_account()
    if kit not in {"none", "made"} and account is not None:
        try:
            listing = sfx.list_kit(kit)
        except sfx.SfxError:
            listing = {"sounds": []}
        if not listing["sounds"] or any(sound["state"] != "fresh"
                                         for sound in listing["sounds"]):
            command = "yap-studio sfx kit" + (f" --kit {kit}" if kit != "soft" else "")
            if account in {"paid", "unknown"}:
                items.append("sound effects: generate your kit once "
                             f"(spends ElevenLabs credits): {command} --yes")
            else:
                items.append("sound effects: set up a paid ElevenLabs account, then "
                             f"generate your kit: {command} --yes")
    return items


def _stage_state(
    context: ProjectContext, spec: StageSpec, previous: dict[str, str],
) -> dict[str, Any]:
    folder = context.project_dir / spec.folder
    result: dict[str, Any] = {
        "name": spec.name, "state": "missing", "reason": "no receipt",
        "receipt": None, "progress": None,
    }
    # Bookkeeping must be read as one transaction; input hashing stays outside it.
    with project_lock(context.project_dir):
        progress = _read_json(folder / "progress.json")
        data = read_receipt(folder)
        pending = (folder / "receipt.pending.json").exists()
    if data is not None:
        result["receipt"] = f"{spec.folder}/receipt.json"
        try:
            targets = receipt_output_paths(context.project_dir, data)
        except InputError:
            result.update(state="failed", reason="receipt names a file outside the project")
            return result
    if progress and pid_alive(progress.get("pid")):
        result.update(state="running", reason=progress.get("step", "running"), progress=progress)
        return result
    for upstream in spec.after:
        if previous.get(upstream) != "done":
            result["reason"] = f"waits for {upstream}"
            return result
    if data is None:
        if pending or progress or (folder.is_dir() and any(folder.iterdir())):
            result.update(state="failed", reason="interrupted")
        return result
    if data.get("status") == "failed":
        failed = failed_assert_lines(data.get("asserts", []))
        result.update(state="failed", reason=("failed: " + "; ".join(failed)
                                              if failed else "stage reported failure"))
        return result
    if pending and data.get("started") != (_read_json(folder / "receipt.pending.json") or {}).get(
        "started"
    ):
        result.update(state="failed", reason="interrupted")
        return result
    if data.get("status") != "verified":
        result.update(state="failed", reason="receipt is not verified")
        return result
    try:
        current = spec.inputs(context)
    except FileNotFoundError as exc:
        result.update(state="missing", reason=f"missing input {Path(exc.filename or '').name}")
        return result
    except YapError as exc:
        result.update(state="stale", reason=exc.message)
        return result
    except (ValueError, OSError) as exc:
        result.update(state="failed", reason=f"invalid input: {exc}")
        return result
    recorded = data.get("inputs", {})
    changes = sorted(key for key in current.keys() | recorded.keys()
                     if current.get(key) != recorded.get(key))
    if changes:
        result.update(state="stale", reason="changed input: " + ", ".join(changes))
        return result
    for output, target in zip(data.get("outputs", []), targets, strict=True):
        if not target.is_file():
            result.update(state="stale", reason=f"missing output: {output['path']}")
            return result
        if context.identity.sha256(target) != output["sha256"]:
            result.update(state="stale", reason=f"changed output: {output['path']}")
            return result
    result.update(state="done", reason="inputs match verified receipt")
    return result


def _voiceover_picture_need(context: ProjectContext) -> str | None:
    """Ask for picture over a completed voiceover cut before rendering layers."""
    from yap_studio.stages.voiceover import frame_ranges_text, uncovered_frames

    if not all(take.voiceover for take in context.project.takes):
        return None
    try:
        if any((context.project_dir / f"timeline/{lane}.json").is_file()
               for lane in ("agent", "broll")):
            timeline = context.resolved_timeline(allow_violations=True)
            gaps = uncovered_frames(timeline.total_frames,
                                    [item.to_dict() for item in timeline.elements])
        else:
            total = context.json("cut/receipt.json")["plan"]["total_frames"]
            gaps = uncovered_frames(total, [])
    except (YapError, ValueError, OSError, KeyError, TypeError):
        return None
    if not gaps:
        return None
    return (f"cover the voiceover with picture: frames {frame_ranges_text(gaps)} have no "
            "B-roll yet; plan inserts, then yap-studio timeline add FRAGMENT DIR --lane broll")


def _time_anchor_needs(context: ProjectContext) -> list[str]:
    receipt_path = context.project_dir / "cut" / "receipt.json"
    if not receipt_path.is_file():
        return []
    needs = []
    for lane in ("agent", "broll"):
        relative = f"timeline/{lane}.json"
        lane_path = context.project_dir / relative
        if (not lane_path.is_file()
                or lane_path.stat().st_mtime_ns >= receipt_path.stat().st_mtime_ns):
            continue
        data = context.json(relative)
        def has_time(value: Any) -> bool:
            if isinstance(value, list):
                return any(has_time(child) for child in value)
            if isinstance(value, dict):
                return any((key in {"start", "end", "at"} and isinstance(child, dict)
                            and "time" in child) or has_time(child)
                           for key, child in value.items())
            return False

        if has_time(data.get("elements", [])):
            needs.append(f"review output-time anchors in {relative} after re-cut")
    return needs


def _cut_review(project_dir: Path) -> str:
    """Ask to save the cut only when the current saved version holds another cut."""
    from yap_studio.feedback.versions import current_version, version_pieces

    held = False
    try:
        saved = current_version(project_dir)
        if saved is not None:
            spans = hashlib.sha256((project_dir / "cut/spans.json").read_bytes()).hexdigest()
            cut = read_receipt(project_dir / "cut") or {}
            # Spans omit picture inputs such as crop_x, so the cut inputs must match too.
            held = ((saved.snapshots.get("spans") or {}).get("sha256") == spans
                    and saved.snapshots.get("cut_pieces") is not None
                    and "inputs" in cut
                    and version_pieces(saved).get("inputs") == cut["inputs"])
    except (YapError, OSError, ValueError, TypeError, KeyError):
        held = False
    # Saving again would put the cut ahead of a final made from it.
    if held:
        return "review the current version: yap-studio review --versions DIR"
    return ("review the cut: yap-studio versions save DIR, then yap-studio review --versions DIR "
            "(reviews of loose files cannot take replies)")


def _review_note_needs(project_dir: Path) -> tuple[str | None, str | None]:
    """Find batched notes needing replies and earlier notes needing a batch."""
    from yap_studio.feedback.batch import batch_ids, load_batch
    from yap_studio.feedback.versions import current_version, load_versions
    from yap_studio.review.store import ReviewStore, data_dir

    try:
        versions = load_versions(project_dir)
        if len(versions) < 2:
            return None, None
        current = current_version(project_dir)
        if current is None:
            return None, None
        positions = {saved.id: index for index, saved in enumerate(versions)}
        current_position = positions[current.id]
        if current_position == 0:
            return None, None
        ids = batch_ids(project_dir)
        latest_batch = {}
        for batch_id in ids:
            for item in load_batch(project_dir, batch_id)["items"]:
                comment = item["comment"]
                if positions.get(comment["version"], current_position) < current_position:
                    latest_batch[(comment["version"], comment["id"])] = batch_id
        store = ReviewStore(data_dir())
        try:
            series = store.series(versions[0].media["sha256"])
        except InputError as exc:
            if exc.reason != "series does not exist":
                raise
            series = None
        answered = set()
        if series is not None:
            numbers = {item.sha256: item.number for item in series.versions}
            saved_by_number = {numbers[saved.media["sha256"]]: saved for saved in versions
                               if saved.media["sha256"] in numbers}
            answered = {
                (saved_by_number[reply.comment_version].id, reply.comment_id)
                for reply in series.replies
                if reply.comment_version in saved_by_number
                and reply.version in saved_by_number
                and positions[saved_by_number[reply.version].id] <= current_position
            }
        unanswered = {ref: batch_id for ref, batch_id in latest_batch.items()
                      if ref not in answered}
        batched = set(latest_batch)
        unsent_count = sum(
            (saved.id, comment.id) not in batched
            and (saved.id, comment.id) not in answered
            for saved in versions[:current_position]
            for comment in store.comments(saved.media["sha256"])
        )
    except (YapError, OSError, ValueError, TypeError, KeyError):
        return None, None
    reply_need = None
    if unanswered:
        needed_batches = set(unanswered.values())
        batches = [batch_id for batch_id in ids if batch_id in needed_batches]
        count = len(unanswered)
        reply_need = (f"reply to {count} {'review note' if count == 1 else 'review notes'} in "
                      f"{', '.join(batches)} before the next review: "
                      "yap-studio feedback replies DIR --batch bN, then "
                      "yap-studio feedback reply DIR --batch bN --comment vK:cN --version vM "
                      '--status changed|kept|declined --text "..."')
    unsent_need = None
    if unsent_count:
        unsent_need = (f"send {unsent_count} "
                       f"{'review note' if unsent_count == 1 else 'review notes'} from earlier "
                       "versions that are in no feedback batch: open the current version with "
                       "yap-studio review --versions DIR, save Export JSON, then "
                       "yap-studio feedback build DIR EXPORT "
                       "(open earlier notes are carried into the new batch)")
    return reply_need, unsent_need


def ingest_speech_warnings(project_dir: Path) -> list[str]:
    """Read source-clock speech gaps from a completed ingest manifest."""
    from yap_studio.stages.coverage import speech_warning

    try:
        manifest = json.loads((project_dir / "ingest/ingest.json").read_text())
    except FileNotFoundError:
        return []
    return [
        f"{take['id']}: {speech_warning(**span)}"
        for take in manifest.get("takes", [])
        for span in take.get("speech_check", {}).get("untranscribed", [])
    ]


def _cached_transcripts(context: ProjectContext) -> list[dict]:
    """Report reusable speech results using only known file identities."""
    try:
        backend = resolve_backend(context.project.asr.get("engine") or "auto",
                                  context.project.asr.get("model"))
        if backend is None:
            return []
        cached = []
        for take in context.project.takes:
            path = context.project_dir / take.path
            recorded = context.ingested_take(take.id)
            if (recorded is not None and recorded.get("path") == take.path
                    and recorded.get("copy")):
                path = (context.project_dir / recorded["copy"]).resolve()
                if not path.is_relative_to(context.project_dir / "ingest/takes"):
                    return []
            digest = shared.known_digest(path) or context.identity.known_sha256(path)
            if digest is not None and shared.lookup(
                backend, shared.Request(digest, context.selected_audio_stream(take.id), path),
                language=context.project.language,
            ) is not None:
                cached.append({"take": take.id, "engine": backend.name, "model": backend.model})
        return cached
    except (YapError, OSError, ValueError, TypeError, KeyError, AttributeError):
        return []


def status(project_dir: Path) -> dict:
    """Report stage states, agent writes, human decisions, and the next action."""
    with paths.operation():
        try:
            context = ProjectContext(project_dir)
        except (OSError, ValueError, UnicodeError) as exc:
            from yap_studio.project.model import ProjectFileError

            if isinstance(exc, ProjectFileError):
                raise
            raise ProjectFileError("project.json", "$", f"cannot read project: {exc}") from exc
        stages = []
        previous: dict[str, str] = {}
        for spec in STAGES:
            state = _stage_state(context, spec, previous)
            stages.append(state)
            previous[spec.name] = state["state"]
        needs = []
        if previous.get("ingest") == "done" and not (context.project_dir / "edit.json").is_file():
            needs.append("write edit.json segments (read `yap-studio words DIR`)")
        needs.extend(_time_anchor_needs(context))
        reply_need, unsent_need = _review_note_needs(context.project_dir)
        for need in (reply_need, unsent_need):
            if need is not None:
                needs.append(need)
        needs_human = human_items(context)
        warnings = (
            ingest_speech_warnings(context.project_dir)
            if previous.get("ingest") == "done" else []
        )
        timeline_fix = None
        cut_review = None
        picture_need = None
        if previous.get("cut") == "done":
            picture_need = _voiceover_picture_need(context)
            if picture_need is not None:
                needs.append(picture_need)
            cut_review = _cut_review(context.project_dir)
            needs_human.append(cut_review)
            if ((context.project_dir / "cut/cut-words.json").is_file()
                    and any((context.project_dir / f"timeline/{lane}.json").is_file()
                            for lane in ("agent", "broll"))):
                from yap_studio.stages.timeline import resolve_timeline

                try:
                    timeline = resolve_timeline(context.project_dir)
                    warnings.extend(item.message for item in timeline.findings
                                    if item.rule == "voiceover_graphic_only")
                    violations = [item for item in timeline.findings
                                  if item.severity == "violation"]
                    needs_human.extend(f"{item.message}; {item.suggested_action}"
                                       for item in violations)
                    if violations:
                        timeline_fix = violations[0].suggested_action
                except (YapError, ValueError, OSError) as exc:
                    needs_human.append(f"check timeline: {exc}")
        next_action = None
        for spec, state in zip(STAGES, stages, strict=True):
            if state["state"] in {"done", "running"}:
                continue
            # A verified cut has already consumed the applied filler removals.
            if spec.name == "fillers" and cut_review is not None:
                continue
            if any(previous.get(upstream) != "done" for upstream in spec.after):
                continue
            if spec.name in {"fillers", "cut"} and needs and not (
                context.project_dir / "edit.json"
            ).exists():
                next_action = needs[0]
            elif spec.human:
                needs_human.append(f"review {spec.name}")
            else:
                import shlex

                next_action = f"yap-studio {spec.name} {shlex.quote(str(context.project_dir))}"
            break
        if next_action is None and needs:
            next_action = needs[0]
        if picture_need is not None and next_action is not None and next_action.startswith(
            tuple(f"yap-studio {stage} " for stage in ("music", "captions", "graphics", "assemble"))
        ):
            next_action = picture_need
        if cut_review is not None:
            cut_index = next(i for i, stage in enumerate(stages) if stage["name"] == "cut")
            if (not any((context.project_dir / f"timeline/{lane}.json").exists()
                        for lane in ("agent", "broll"))
                    and all(stage["state"] == "missing" and stage["receipt"] is None
                            for stage in stages[cut_index + 1:])):
                next_action = cut_review
        review_need = reply_need or unsent_need
        if review_need is not None and (next_action is None or next_action == cut_review):
            next_action = review_need
        if timeline_fix is not None:
            # Rendering cannot proceed until the responsible lane's violation is repaired.
            next_action = timeline_fix
        from yap_studio.stages import run_detach

        run = run_detach.status(context.project_dir, stages)
        if run is not None and run["state"] == "running":
            next_action = ("the background run is still working; check again with "
                           + run_detach.commands(context.project_dir)[0])
        from yap_studio.matte.chroma import cached_status

        transcripts = _cached_transcripts(context)
        return {
            "schema": "yap-studio/status@1", "project": context.project.name,
            "video": {"type": context.project.video_type,
                      "broll_sources": (list(context.project.broll_sources)
                                        if context.project.broll_sources is not None else None)},
            "stages": stages, "needs": needs, "needs_human": needs_human,
            "warnings": warnings, "next": next_action, "run": run,
            **({"transcripts": transcripts} if transcripts else {}),
            **cached_status(context),
        }
