"""Publish the composited picture, policy-driven mix, verification EDL and receipt."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np

from yap_studio import elevenlabs, ffmpeg, letters, paths, receipt
from yap_studio import sfx as sound_effects
from yap_studio.canvas import canvas, require_supported
from yap_studio.captions import phrase_keys
from yap_studio.checks.caption_contrast import (
    CONTRAST_SCHEMA,
    CONTRAST_SUFFIX,
    HELD_SAMPLES,
    OPAQUE_ALPHA,
    SAMPLE_EVERY,
    SURROUND_PERCENTILE,
    measure_caption_contrast,
    write_contrast_record,
)
from yap_studio.checks.caption_layout import LAYOUT_SUFFIX, write_layout_record
from yap_studio.checks.model import Thresholds, TransitionWindow, format_time
from yap_studio.errors import InputError, ProcessingError, YapError
from yap_studio.faces.clearance import Layout, caption_gap_px, check_layout, face_clearance_px
from yap_studio.faces.detect import write_faces
from yap_studio.faces.final import drawn_boxes, final_layout
from yap_studio.faces.mapping import (
    behind_face_windows,
    map_cut_faces,
    map_picture_faces,
    picture_zooms,
)
from yap_studio.faces.model import Box, VideoIdentity
from yap_studio.faces.project import cut_faces
from yap_studio.grade import GRADE_VERSION, GradeSpec, describe_compare, resolve_grade
from yap_studio.joins import assembly as jump_cuts
from yap_studio.matte.check import HIDDEN_GLYPH_SHARE, HIDDEN_TIME_SHARE
from yap_studio.matte.frames import decode_frames
from yap_studio.matte.project import CutMatteError, ensure_cut_matte
from yap_studio.project.lock import pid_alive, project_lock
from yap_studio.project.model import CutWord, CutWords, parse_document
from yap_studio.project.receipts import (
    StageRun,
    _read_json,
    failed_assert_lines,
    projection_sha256,
    read_pending,
    read_receipt,
    retained_outputs,
)
from yap_studio.project.status import STAGES, ProjectContext
from yap_studio.render.moment_clones import CLONE_RENDER_VERSION
from yap_studio.render.moments import MOMENT_RENDER_VERSION
from yap_studio.sfx import bleep
from yap_studio.stages import tall_box
from yap_studio.stages.assemble_audio import (
    decode,
    delivery_asserts,
    render_audio,
    validate_voice_chain,
)
from yap_studio.stages.assemble_picture import (
    _source,
    _zoom_inputs,
    overlay_alpha_runs,
    render_picture,
)
from yap_studio.stages.broll_split import split_layouts, split_windows
from yap_studio.stages.captions import (
    add_caption_halos,
    caption_frame_window,
    caption_halos,
    caption_receipt_needs_recheck,
    caption_snapshot,
    chapter_number,
)
from yap_studio.stages.number_visibility import add_number_visibility
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline, TimelineRefusal
from yap_studio.stages.transitions import render_transitions, template_file
from yap_studio.taste.render import write_taste_record
from yap_studio.text_hook import TEXT_HOOK_ID, require_top_text
from yap_studio.verify import _verify as verify

_COMMANDS: ContextVar[list[list[str]] | None] = ContextVar("assembly_commands", default=None)
FACES_CLEARANCE_CHECK = "faces.clearance"
CAPTIONS_CARDS_CHECK = "captions.cards"


def require_voiceover_picture(total_frames: int, elements: list[dict[str, Any]],
                              *, frames: tuple[int, int] | None = None,
                              graphic_coverage: dict[str, set[int]] | None = None) -> None:
    """Require each output frame to have authored picture over the stand-in."""
    covered = bytearray(total_frames)
    for item in elements:
        if (item["kind"] not in {"insert", "graphic"}
                or item.get("media", {}).get("fit") == "split"):
            continue
        if item["kind"] == "graphic":
            for frame in (graphic_coverage or {}).get(item["id"], set()):
                if 0 <= frame < total_frames:
                    covered[frame] = 1
            continue
        first = max(0, item["start_frame"])
        last = min(total_frames, item["end_frame"])
        covered[first:last] = b"\x01" * max(0, last - first)
    gaps = []
    start = None
    first_required, last_required = frames if frames is not None else (0, total_frames - 1)
    for frame in range(first_required, last_required + 1):
        # The end hold freezes the last content frame, so it shares that frame's picture.
        present = covered[min(frame, total_frames - 1)] if total_frames else 0
        if not present and start is None:
            start = frame
        elif present and start is not None:
            gaps.append(f"{start}-{frame - 1}")
            start = None
    if start is not None:
        gaps.append(f"{start}-{last_required}")
    if gaps:
        raise AssemblyError(
            "Voiceover picture uncovered at frames " + ", ".join(gaps),
            suggested_action=(
                "Plan B-roll inserts over these frames and add them with yap-studio timeline "
                "add FRAGMENT DIR --lane broll; yap-studio timeline check DIR lists frames "
                "still uncovered."
            ),
        )


def _graphic_picture_frames(root: Path, snapshot: AssemblySnapshot,
                            frames: tuple[int, int] | None) -> dict[str, set[int]]:
    """Count a graphic only where every canvas pixel is opaque."""
    result: dict[str, set[int]] = {}
    required = frames or (0, snapshot.timeline.total_frames - 1)
    for layer in snapshot.title_layers:
        if (layer["kind"] != "graphic" or layer["x"] != 0 or layer["y"] != 0 or
                layer["width"] != 1080 or layer["height"] != 1920):
            continue
        start = layer["start_frame"]
        if start > required[1] or start + layer["frames"] <= required[0]:
            continue
        stream = ffmpeg.run_stream([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(root / layer["path"]),
            "-vf", "alphaextract", "-frames:v", str(layer["frames"]),
            "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
        ], timeout_s=ffmpeg.DEFAULT_TIMEOUT_S, chunk_size=64 * 1024)
        frame_size = layer["width"] * layer["height"]
        used = index = 0
        opaque = True
        try:
            for chunk in stream:
                while chunk:
                    part, chunk = chunk[:frame_size - used], chunk[frame_size - used:]
                    opaque &= part.count(255) == len(part)
                    used += len(part)
                    if used == frame_size:
                        if opaque:
                            result.setdefault(layer["id"], set()).add(start + index)
                        index += 1
                        used = 0
                        opaque = True
            if used or index != layer["frames"]:
                raise AssemblyError(f"{layer['id']}: graphic alpha decode is incomplete.")
        finally:
            stream.close()
    return result


def _audit_commands(event: str, args: tuple[Any, ...]) -> None:
    """Observe media/renderer subprocesses, including probes inside verification."""
    commands = _COMMANDS.get()
    if event == "subprocess.Popen" and commands is not None:
        commands.append([str(value) for value in args[1]])


sys.addaudithook(_audit_commands)


@contextmanager
def _record_commands(commands: list[list[str]]) -> Iterator[None]:
    """Scope command recording to this run, including nested verifier calls."""
    token = _COMMANDS.set(commands)
    try:
        yield
    finally:
        _COMMANDS.reset(token)


class AssemblyError(YapError):
    """Explain a failed final delivery or stale prerequisite with its next action."""

    def __init__(self, message: str, *, failed_checks: bool = False,
                 suggested_action: str | None = None) -> None:
        super().__init__(message, error_type="assembly_error",
                         code=("assembly_checks_failed" if failed_checks
                               else "assembly_inputs_invalid"),
                         suggested_action=(suggested_action or
                                           "Correct inputs, re-run stale stages and assemble."))


class StaleStageError(AssemblyError):
    """Name a prerequisite whose receipt needs refreshing."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"{stage} has no current verified receipt; run {stage} first.")
        self.stage = stage


class _HaloRedo(Exception):
    """Restart the final once after adding automatic caption halos."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys


class PilotRangeError(YapError):
    """Refuse a pilot window outside the final or ending inside a placed phrase."""

    def __init__(self, message: str, *, suggested_end: int | None = None,
                 suggested_action: str | None = None) -> None:
        super().__init__(message, error_type="pilot_range_error", code="pilot_range_refused",
                         suggested_action=suggested_action or "Choose frames inside the final.")
        self.suggested_end = suggested_end

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        if self.suggested_end is not None:
            result["suggested_end"] = self.suggested_end
        return result


def pilot_record(path: Path) -> dict[str, Any] | None:
    """Read the pilot marker from a media file, including a renamed copy."""
    try:
        result = ffmpeg.run(["ffprobe", "-v", "error", "-show_entries",
                             "format_tags=comment", "-of", "json", str(path)],
                            timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)
        tag = json.loads(result.stdout).get("format", {}).get("tags", {}).get("comment", "")
    except (YapError, ValueError):
        return None
    match = re.fullmatch(r"yap-studio pilot frames (\d+)-(\d+) of (\d+); not a final", tag)
    return {"frames": [int(match[1]), int(match[2])], "of_frames": int(match[3])} \
        if match else None


def _pilot_phrases(context: ProjectContext) -> tuple[bytes, str]:
    """Read caption phrases against their verified output hash for a pilot only."""
    name = "captions/phrases.json"
    record = read_receipt(context.project_dir / "captions") or {}
    output = next((item for item in record.get("outputs", []) if item.get("path") == name), None)
    if record.get("status") != "verified" or output is None:
        raise AssemblyError(f"{name} is missing from the verified captions receipt.")
    try:
        data = context.read(name)
    except (YapError, OSError) as exc:
        raise AssemblyError(f"{name} is missing or unreadable.") from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest != output.get("sha256"):
        raise AssemblyError(f"{name} differs from its verified receipt.")
    return data, digest


def _pilot_window(snapshot: AssemblySnapshot, frames: tuple[int, int],
                  phrases_bytes: bytes) -> None:
    """Check final bounds and every caption and graphics window before rendering."""
    first, last = frames
    total = snapshot.timeline.total_frames + snapshot.timeline.end_hold_frames
    if first < 0 or first > last or last >= total:
        raise PilotRangeError(f"Frames {first}-{last} fall outside final frames 0-{total - 1}.")
    layout = json.loads(snapshot.small_files["captions/layout.json"])
    if layout.get("measured") is False:
        try:
            phrases = json.loads(phrases_bytes)
            if (phrases["schema"] != "yap-studio/phrases@1" or
                    phrases["time_base"] != "output" or phrases["fps"] != 30 or
                    phrases["layer"] != snapshot.caption_layer):
                raise ValueError("phrase clock or layer differs from the verified caption layer")
            elements = []
            for item in phrases["phrases"]:
                start, end = caption_frame_window(item["start"], item["end"], fps=30,
                                                  layer_start=snapshot.caption_layer["start_frame"])
                if end > start:
                    elements.append(("caption", item["text"], start, end))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise AssemblyError(f"captions/phrases.json has invalid pilot phrase windows: "
                                f"{exc}") from exc
    else:
        elements = [("caption", item.get("text", item["id"]), item["start_frame"],
                     item["end_frame"]) for item in layout["captions"]]
    elements.extend((item["kind"], item["id"], item["start_frame"],
                     item["start_frame"] + item["frames"]) for item in snapshot.title_layers)
    crossing = next(((kind, label, start, end) for kind, label, start, end in elements
                     if start <= last and end > last + 1), None)
    if crossing is None:
        return
    candidates = {end - 1 for _, _, _, end in elements} | {total - 1}
    complete = [candidate for candidate in candidates if candidate >= first and
                not any(start <= candidate and end > candidate + 1
                        for _, _, start, end in elements)]
    suggested = min(complete, key=lambda candidate: (abs(candidate - last), -candidate))
    kind, label, start, end = crossing
    name = f'"{label}"' if kind == "caption" else str(label)
    command = f"--frames {first}-{suggested}"
    raise PilotRangeError(
        f"Frames {first}-{last} end inside the {kind} {name} "
        f"(frames {start}-{end - 1}). Nearest complete phrase end: {suggested}; "
        f"try {command}.", suggested_end=suggested, suggested_action=f"Try {command}.",
    )


@dataclass(frozen=True)
class AssemblySnapshot:
    """All small file bytes and media identities used by a single final render."""

    timeline: ResolvedTimeline
    caption_layer: dict[str, Any] | None
    title_layers: list[dict[str, Any]]
    spans: dict[str, Any]
    bed: str | None
    small_files: dict[str, bytes]
    media: dict[str, Path]
    sources: dict[str, str]
    audio_streams: dict[str, int]
    inputs: dict[str, str]
    sfx: dict[str, Any]
    grade: GradeSpec | None = None
    grade_sample: dict[str, float] | None = None
    grade_style: dict[str, Any] | None = None
    caption_slots: tuple[dict[str, Any], ...] = ()
    caption_windows: dict[str, str] = field(default_factory=dict)
    caption_style: dict[str, str] | None = None
    picture: str | None = None


def _visual_media_key(item: ResolvedElement) -> str:
    """Preserve HEIF suffixes so retained photos keep their tile-grid decoding."""
    suffix = Path(item.payload["media"]["path"]).suffix
    if suffix.lower() in {".heic", ".heif"}:
        return f"heif-{item.kind}s/{item.id}{suffix}"
    return f"{item.kind}s/{item.id}"


def _resolve_tall_framing(context: ProjectContext, timeline: ResolvedTimeline,
                         style: dict, inputs: dict, captions_layout: dict) -> ResolvedTimeline:
    """Bind automatic framing to the cut matte and already rendered caption geometry."""
    elements = []
    matte = None
    faces = None
    for item in timeline.elements:
        if item.kind != "overlay" or item.payload.get("placement") != "tall_behind":
            elements.append(item)
            continue
        payload = copy.deepcopy(item.payload)
        requested = payload["tall_framing"]
        if requested == "auto":
            if matte is None:
                try:
                    matte = ensure_cut_matte(context.project_dir,
                        context.project_dir / "cut/cut.mp4", cutout=style["broll"]["cutout"])
                except CutMatteError as exc:
                    raise CutMatteError("The tall box behind your head needs a usable person "
                                        "cutout: " + str(exc)) from exc
                inputs["picture:tall_matte"] = context.identity.sha256(matte)
            count = item.end_frame - item.start_frame
            decision = tall_box.auto_framing(decode_frames(matte, 1080, 1920,
                time.monotonic() + 120 + count, first=item.start_frame, count=count),
                face_clearance_px(style))
            decision["captions_clear"] = None
            if decision["chosen"] == "shrink":
                if faces is None:
                    faces = cut_faces(context.project_dir, context.project_dir / "cut/cut.mp4")
                mapped, _ = map_cut_faces(
                    faces, total_frames=timeline.total_frames, end_hold_frames=0,
                    zooms=(), inserts=(), video=faces.video,
                    layouts=[(item.start_frame, item.end_frame, tall_box.SHRINK_BOX_PX, 0)],
                )
                if mapped.status == "measured" and captions_layout["measured"]:
                    layout = Layout.from_dict(captions_layout)
                    # Check only the box window, including phrases crossing either cut.
                    layout = replace(layout, elements=(), captions=tuple(
                        replace(phrase, start_frame=max(phrase.start_frame, item.start_frame),
                                end_frame=min(phrase.end_frame, item.end_frame))
                        for phrase in layout.captions
                        if phrase.start_frame < item.end_frame
                        and phrase.end_frame > item.start_frame
                    ))
                    decision["captions_clear"] = not check_layout(
                        mapped, layout, margin_px=face_clearance_px(style)).violations
                    if not decision["captions_clear"]:
                        decision.update(chosen="keep", reason="captions")
            payload["framing"] = {**decision, "from": "auto"}
        else:
            payload["framing"] = {"chosen": requested, "from": "style",
                                  "visible_min": None, "above_min_px": None}
        if tall_box.chosen_framing(payload) == "shrink":
            src_w, src_h = payload["page_source_px"]
            payload["box_px"] = list(tall_box.shrink_page_box(src_w))
            _, _, width, height = payload["box_px"]
            if payload.get("motion") == "scroll":
                page_h = tall_box.page_height(src_w, src_h, width)
                payload["scroll"] = {"page_px": [width, page_h], "flicks": [list(row) for row in
                    tall_box.scroll_plan(item.end_frame - item.start_frame, page_h, height)]}
        elements.append(replace(item, payload=payload))
    return replace(timeline, elements=tuple(elements))


def assembly_snapshot(context: ProjectContext, *, allow_violations: bool = False
                      ) -> AssemblySnapshot:
    """Validate upstream receipts and retain every input before any render starts."""
    require_supported(context)
    timeline = context.resolved_timeline(allow_violations=allow_violations)
    # An override permits editorial rule violations, never broken media or clock bounds.
    unsafe = {"duration", "inside_video", "end_hold", "media_hash", "insert_bounds",
              "insert_media", "words_hash", "anchor", "insert_speed", "zoom_peak",
              "insert_motion", "zoom_keys",
              "insert_hdr", "overlay_bounds", "overlay_media", "overlay_speed", "overlay_hdr",
              "overlay_box", "background_bounds", "background_media", "background_speed",
              "background_hdr", "background_overlap", "step_bounds", "lane_owner",
              "transition_join", "transition_sentence", "transition_overlap",
              "transition_duplicate", "grade_compare_grade", "grade_compare_overlap",
              "grade_compare_length", "moment_overlap", "moment_off", "moment_length",
              "clone_keyed",
              "background_motion", "background_enter", "tall_box_overlap", "tall_box_media",
              "overlay_scroll", "camera_3d"}
    unsafe.update({"card_video_media", "card_video_speed", "card_video_bounds", "card_video_audio"})
    unsafe.add("card_video_handoff")
    if any(item.rule in unsafe and item.severity == "violation" for item in timeline.findings):
        raise TimelineRefusal("; ".join(item.message for item in timeline.findings
                                        if item.rule in unsafe and item.severity == "violation"))
    projection_context = copy.copy(context)
    projection_context._timeline = replace(timeline, findings=())
    inputs = {"project:name": projection_sha256(context.project.name),
              "project:audio": projection_sha256(context.project.audio),
              "project:verification": projection_sha256({
                  "asr": context.project.asr,
                  **({"platform": context.project.platform}
                     if context.project.platform != "none" else {}),
              }),
              "style:music": projection_sha256({
                  key: value for key, value in context.render_style().data["music"].items()
                  if key not in {"sound_effects_fallback", "sound_extras", "bleep_swears",
                                 "bleep_words_add", "bleep_words_remove"}}),
              "style:voice": projection_sha256(context.render_style().data["voice"]),
              "style:face_clearance": projection_sha256(
                  face_clearance_px(context.render_style().data)),
              "style:caption_gap": projection_sha256(
                  caption_gap_px(context.render_style().data)),
              "timeline": projection_sha256(timeline.to_dict()),
              **context.taste_inputs("music")}
    style_data = context.render_style().data
    if bleep.enabled(style_data):
        inputs["style:bleep"] = projection_sha256({
            **bleep.projection(style_data["music"]), "tone_hz": bleep.TONE_HZ, "pad_s": bleep.PAD_S,
            "extension": {"max_s": bleep.EXTEND_MAX_S, "limit": "neighbours"},
        })
    if any(item.kind == "overlay" and (item.payload.get("placement") == "tall_behind"
                                      or item.payload.get("motion") == "scroll")
           for item in timeline.elements):
        from yap_studio.stages.tall_box import rule_constants

        inputs["picture:tall_box"] = projection_sha256(rule_constants())
    grade_sample = copy.deepcopy((context.project.grade or {}).get("sample"))
    grade = resolve_grade(style_data, context.project.grade,
                          cut=context.project_dir / "cut/cut.mp4")
    if grade.active:
        inputs["grade"] = projection_sha256({**grade.to_dict(), "version": GRADE_VERSION,
                                             **({"sample": grade_sample} if grade_sample else {})})
    grade_style = None
    if any(item.kind == "grade_compare" for item in timeline.elements):
        grade_style = {"captions": {"text_color": style_data["captions"]["text_color"]},
                       "graphics": {"ink_color": style_data["graphics"]["ink_color"]}}
        inputs["grade:compare"] = projection_sha256(describe_compare(grade_style))
    if style_data["broll"]["cutout"] != "auto":
        inputs["picture:cutout"] = projection_sha256(style_data["broll"]["cutout"])

    if any(item.kind == "moment" for item in timeline.elements):
        inputs["picture:moments"] = projection_sha256({
            "version": MOMENT_RENDER_VERSION, "moments": style_data["moments"],
            "accent_color": style_data["graphics"]["accent_color"],
            "elements": [item.to_dict() for item in timeline.elements if item.kind == "moment"],
            **({"clone_version": CLONE_RENDER_VERSION} if any(
                item.kind == "moment" and item.payload["style"] == "clone_flash"
                for item in timeline.elements) else {}),
        })
    kit_name = style_data["music"]["sound_effects"]
    sfx_info: dict[str, Any] = {"kit": kit_name, "sounds": [], "ready": False}
    fallback = style_data["music"].get("sound_effects_fallback", "made")
    if kit_name != "none":
        try:
            sfx_info = sound_effects.list_kit(kit_name)
            sfx_info["ready"] = bool(sfx_info["sounds"]) and all(
                item["state"] == "fresh" for item in sfx_info["sounds"])
        except sound_effects.SfxError:
            pass
        if not sfx_info["ready"] and fallback == "made":
            sfx_info = {**sound_effects.list_kit("made"), "ready": True,
                        "fallback_from": kit_name}
        for item in sfx_info["sounds"]:
            inputs[f"sfx:recipe:{item['name']}"] = item["recipe_sha256"]
            if sfx_info["ready"]:
                inputs[f"sfx:audio:{item['name']}"] = item["audio_sha256"]
    extras = style_data["music"].get("sound_extras", [])
    inputs["style:sfx"] = projection_sha256({"kit": kit_name,
        **({"fallback": "made"} if "fallback_from" in sfx_info else {}),
        **({"sound_extras": extras} if extras else {}),
        "peak_below_voice_db": style_data["internal"]["sfx_peak_below_voice_db"],
        "min_gap_s": style_data["internal"]["sfx_min_gap_s"]})
    requested = {item.payload["sound"] for item in timeline.elements if item.kind == "sound"}
    if extras or requested:
        from yap_studio.sfx.place import candidates

        extra_kit = sound_effects.list_kit("extras")
        enabled = extras if kit_name not in {"none", "made"} else []
        # Find required moments even when their cached audio is not ready yet.
        required = {cue.sound for cue in candidates(timeline, [
            {**sound, "state": "fresh"} for sound in extra_kit["sounds"]],
            sound_extras=enabled)}
        additional = [sound for sound in extra_kit["sounds"] if sound["name"] in required]
        soft_requests = requested - {sound["name"] for sound in extra_kit["sounds"]}
        request_sounds = [sound for sound in extra_kit["sounds"] if sound["name"] in requested]
        if soft_requests:
            request_sounds.extend(sound for sound in sound_effects.list_kit("soft")["sounds"]
                                  if sound["name"] in soft_requests)
        if requested:
            sfx_info["requested_sounds"] = request_sounds
        sfx_info["sound_extras"] = enabled
        sfx_info["additional_sounds"] = additional
        sfx_info["missing_extras"] = [sound["name"] for sound in additional
                                      if sound["state"] != "fresh"
                                      and sound["name"] in required]
        sfx_info["missing_requests"] = [sound["name"] for sound in request_sounds
                                        if sound["state"] != "fresh"
                                        and sound["name"] in soft_requests]
        for sound in additional:
            inputs[f"sfx:recipe:{sound['name']}"] = sound["recipe_sha256"]
            if sound["state"] == "fresh":
                inputs[f"sfx:audio:{sound['name']}"] = sound["audio_sha256"]
        for sound in request_sounds:
            inputs[f"sfx:request:recipe:{sound['name']}"] = sound["recipe_sha256"]
            if sound["state"] == "fresh":
                inputs[f"sfx:request:audio:{sound['name']}"] = sound["audio_sha256"]
    small = {"project.json": context.read("project.json"),
             "cut/spans.json": context.read("cut/spans.json")}
    if style_data["music"]["duck_mode"] == "words" or bleep.enabled(style_data):
        small["cut/cut-words.json"] = context.read("cut/cut-words.json")
    media: dict[str, Path] = {}

    def retain(key: str, source: Path) -> str:
        """Bind media to a safe staging key and a retained identity."""
        if Path(key).is_absolute() or ".." in Path(key).parts:
            raise AssemblyError("An assembly receipt names an unsafe media path.")
        media[key] = paths.validate_input(source)
        inputs[f"media:{key}"] = context.identity.sha256(media[key])
        return key

    stage_receipts = {}
    for name, folder in (("captions", "captions"), ("graphics", "graphics"), ("music", "music")):
        try:
            record = context.json(f"{folder}/receipt.json")
        except FileNotFoundError:
            record = {}
        spec = next(spec for spec in STAGES if spec.name == name)
        current = spec.inputs(projection_context)
        if (record.get("schema") != "yap-studio/stage-receipt@1" or
                record.get("status") != "verified" or record.get("inputs") != current or
                (name == "captions" and caption_receipt_needs_recheck(
                    record, projection_context.creator_caption_placement)) or
                (context.project_dir / folder / "receipt.pending.json").exists()):
            raise StaleStageError(name)
        inputs[f"stage:{name}"] = projection_sha256(current)
        stage_receipts[name] = record
        for output in record.get("outputs", []):
            target = (context.project_dir / output["path"]).resolve()
            if not target.is_relative_to(context.project_dir):
                raise AssemblyError(f"{name} receipt names an output outside the project.")
            try:
                if output["path"] in {"captions/layout.json", "captions/phrases.json",
                                       "captions/camera3d.json"}:
                    small[output["path"]] = context.read(output["path"])
                    digest = hashlib.sha256(small[output["path"]]).hexdigest()
                else:
                    digest = context.identity.sha256(target)
            except (FileNotFoundError, InputError) as exc:
                if not target.is_file():
                    raise StaleStageError(name) from exc
                raise AssemblyError(f"{output['path']} is missing from its verified receipt.") \
                    from exc
            if digest != output["sha256"]:
                raise AssemblyError(f"{output['path']} differs from its verified receipt.")
    if "captions/layout.json" not in small:
        raise AssemblyError("captions has no verified layout; run captions first.")
    if bleep.enabled(style_data) and "captions/phrases.json" not in small:
        raise AssemblyError("captions has no verified words; run captions first.")
    caption = copy.deepcopy(stage_receipts["captions"]["layer"])
    if caption is None and (stage_receipts["captions"].get("phrases") != 0 or
                            json.loads(small["captions/layout.json"]).get("captions") != [] or
                            not json.loads(small["captions/layout.json"]).get("measured")):
        raise AssemblyError("captions has no layer or verified empty layout; run captions first.")
    picture = stage_receipts["captions"].get("picture")
    if picture is not None:
        if picture != "captions/camera3d.mp4" or "captions/camera3d.json" not in small:
            raise AssemblyError("captions has no verified camera picture and plan.")
        if not any(item["path"] == picture for item in stage_receipts["captions"]["outputs"]):
            raise AssemblyError("camera picture is missing from its verified receipt.")
        retain(picture, context.project_dir / picture)
    titles = copy.deepcopy(stage_receipts["graphics"]["elements"])
    behind = {layer["id"] for layer in titles
              if layer.get("kind") == "graphic" and layer["layer"] == "behind"}
    graphic_layers = {layer["id"]: layer["layer"] for layer in titles
                      if layer.get("kind") == "graphic"}
    timeline = replace(timeline, elements=tuple(
        replace(item, layer=graphic_layers[item.id]) if item.id in graphic_layers else item
        for item in timeline.elements
    ))
    for item in timeline.elements:
        if item.kind != "overlay" or item.payload.get("placement") != "tall_behind":
            continue
        for other in timeline.elements:
            if (other.kind in {"graphic", "title"}
                    and not (style_data["hook"]["style"] == "text_hook"
                             and other.kind == "title" and other.id == TEXT_HOOK_ID)
                    and max(item.start_frame, other.start_frame)
                    < min(item.end_frame, other.end_frame)):
                raise TimelineRefusal(f"{item.id}: tall box cannot overlap {other.id} after "
                                      "card or title placement.", code="tall_box_overlap",
                                      suggested_action="Move the card or title outside the tall "
                                                       "box window.")
    if behind or any(item.kind == "background" for item in timeline.elements):
        inputs["picture:behind"] = projection_sha256({
            "hidden_glyph_share": HIDDEN_GLYPH_SHARE, "hidden_time_share": HIDDEN_TIME_SHARE,
            "overlap": "skip_zoom_and_insert", "cut_matte": 2, "measurement": 2,
        })
    from yap_studio.stages.green_screen import rule_constants, uses_look
    if any(item.kind == "background" and uses_look(item.payload) for item in timeline.elements):
        inputs["picture:green_screen"] = projection_sha256(rule_constants())
    for placement in [caption, *titles]:
        if placement is None:
            continue
        retain(placement["path"], context.project_dir / placement["path"])
        for window in placement.get("windows", []):
            source = window["source"]
            if source["kind"] != "media":
                continue
            original = (context.project_dir / source["path"]).resolve()
            if not original.is_relative_to(context.project_dir):
                raise AssemblyError(f"{placement['id']}: use a clip inside the project folder.")
            key = (f"card-clips/{placement['id']}/"
                   f"{window['slot']}-{window['index']}{original.suffix}")
            retain(key, original)
            if inputs[f"media:{key}"] != source["sha256"]:
                raise AssemblyError(f"{placement['id']}: the clip changed; update its sha256.")
            source["source_name"] = original.name
            source["path"] = key
    retain("cut/cut.mp4", context.project_dir / "cut/cut.mp4")
    retain("cut/voice.wav", context.project_dir / "cut/voice.wav")
    music_record = stage_receipts["music"]
    bed = music_record.get("bed")
    if (music_record.get("mode") not in {"file", "elevenlabs", "none"} or
            bool(bed) != (music_record["mode"] in {"file", "elevenlabs"})):
        raise AssemblyError("The music receipt has no usable bed contract.")
    if bed:
        retain(bed, context.project_dir / bed)
    spans = context.json("cut/spans.json")
    sources = {}
    audio_streams = {}
    for number, (name, source) in enumerate(spans["sources"].items()):
        source_name = source["path"] if isinstance(source, dict) else source
        ordinal = source.get("audio_stream", 0) if isinstance(source, dict) else 0
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise AssemblyError(f"{name}: cut spans need a non-negative audio stream ordinal.")
        audio_streams[name] = ordinal
        source_path = paths.validate_input(context.project_dir / "cut" / source_name)
        sources[name] = retain(f"sources/{number}{source_path.suffix}", source_path)
    for item in timeline.elements:
        if item.kind in {"insert", "overlay", "background"}:
            retain(_visual_media_key(item), context.project_dir / item.payload["media"]["path"])
            if "letters" in item.payload:
                inputs.update(letters.assembly_inputs(item, style_data))
        elif item.kind == "title" and style_data["hook"]["style"] == "video_letters":
            inputs["template:letters"] = receipt.sha256_file(letters.TEMPLATE)
        elif item.kind == "transition":
            look = item.payload["look"]
            inputs[f"template:transition_{look}"] = receipt.sha256_file(template_file(look))
    jump_cuts.snapshot_inputs(context, inputs, small)
    inputs.update({key: hashlib.sha256(value).hexdigest() for key, value in small.items()
                   if key != "project.json"})
    # Crop settings are read by native-source zooms without coupling audio policy to the cut.
    inputs["project:crops"] = projection_sha256({take.id: take.crop_x
                                                 for take in context.project.takes})
    captions = caption_snapshot(projection_context)
    caption_slots = () if picture else tuple({"backing": slot.get("backing"), "auto": automatic}
                          for slot, automatic in zip(
                              json.loads(captions.plan.variables["phrases"]), captions.auto,
                              strict=True))
    if any(slot["backing"] for slot in caption_slots):
        inputs["caption:backing"] = projection_sha256(caption_slots)
    windows = {} if picture else dict(zip(
        phrase_keys([phrase.word_ids for phrase in captions.phrases]),
        captions.windows, strict=True))
    if any(item.kind == "overlay" and item.payload.get("placement") == "tall_behind"
           for item in timeline.elements):
        timeline = _resolve_tall_framing(context, timeline, style_data, inputs,
                                       json.loads(small["captions/layout.json"]))
        inputs["picture:tall_framing"] = projection_sha256([
            {"id": item.id, **{key: item.payload.get(key)
                               for key in ("framing", "box_px", "scroll")}}
            for item in timeline.elements
            if item.kind == "overlay" and item.payload.get("placement") == "tall_behind"
        ])
    return AssemblySnapshot(timeline, caption, titles, spans, bed, small, media,
                            sources, audio_streams, inputs, sfx_info, grade=grade,
                            grade_sample=grade_sample, grade_style=grade_style,
                            caption_slots=caption_slots, caption_windows=windows,
                            picture=picture,
                            caption_style=None if picture else {
                                key: captions.plan.variables[key] for key in (
                                "text_color", "highlight_color", "highlight_text_color",
                                "highlight_style",
                            )})


def _audio_clips(snapshot: AssemblySnapshot) -> list[dict[str, Any]]:
    """Select only audible media windows from the retained placement contract."""
    return [{"id": f"{layer['id']}:{window['slot']}:{window['index']}",
             "path": source["path"], "sha256": source["sha256"],
             "in_s": source["in_s"], "speed": source["speed"],
             "start_frame": window["start_frame"], "frames": window["frames"],
             "gain_db": source["audio"]["gain_db"]}
            for layer in snapshot.title_layers for window in layer.get("windows", [])
            if (source := window["source"])["kind"] == "media"
            and source.get("audio", "mute") != "mute"]


def _retained(snapshot: AssemblySnapshot, root: Path) -> ResolvedTimeline:
    """Copy media and contracts, checking identity before consuming the snapshot."""
    for relative, data in snapshot.small_files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for relative, source in snapshot.media.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if receipt.sha256_file(target) != snapshot.inputs[f"media:{relative}"]:
            raise AssemblyError(f"{relative} changed while the assembly snapshot was copied.")
    spans = copy.deepcopy(snapshot.spans)
    spans["sources"] = {
        name: {"path": str(root / relative), "audio_stream": snapshot.audio_streams[name]}
        if snapshot.audio_streams[name] else str(root / relative)
        for name, relative in snapshot.sources.items()
    }
    (root / "cut/spans.json").write_text(json.dumps(spans), encoding="utf-8")
    elements = []
    for item in snapshot.timeline.elements:
        if item.kind in {"insert", "overlay", "background"}:
            payload = copy.deepcopy(item.payload)
            payload["media"]["source_name"] = Path(payload["media"]["path"]).name
            payload["media"]["path"] = str(root / _visual_media_key(item))
            item = replace(item, payload=payload)
        elements.append(item)
    return replace(snapshot.timeline, elements=tuple(elements))


def verification_edl(snapshot: AssemblySnapshot, *, base: Path,
                     references: dict[str, str] | None = None) -> dict[str, Any]:
    """Describe source spans, changed picture regions and the silent composited hold."""
    frame = canvas(json.loads(getattr(snapshot, "small_files", {}).get("project.json", b"{}")).get(
        "shape", "vertical"))
    sources = {name: str(source) for name, source in
               ((name, snapshot.media[relative]) for name, relative in snapshot.sources.items())}
    picture = getattr(snapshot, "picture", None)
    overlays = []
    for placement in [*snapshot.title_layers, snapshot.caption_layer]:
        if placement is None:
            continue
        if picture and placement is snapshot.caption_layer:
            layout = json.loads(snapshot.small_files["captions/layout.json"])
            for caption in layout["captions"]:
                for box in caption["boxes"]:
                    first = max(caption["start_frame"],
                                box.get("start_frame", caption["start_frame"]))
                    last = min(caption["end_frame"], box.get("end_frame", caption["end_frame"]))
                    if first >= last:
                        continue
                    x, y = max(0, box["x"]), max(0, box["y"])
                    right = min(frame.width, box["x"] + box["w"])
                    bottom = min(frame.height, box["y"] + box["h"])
                    if x >= right or y >= bottom:
                        continue
                    overlays.append({"start_in_output": first / 30,
                                     "duration": (last - first) / 30,
                                     "region": [x, y, right - x, bottom - y]})
            continue
        x = max(0, placement["x"])
        y = max(0, placement["y"])
        right = min(frame.width, placement["x"] + placement["width"])
        bottom = min(frame.height, placement["y"] + placement["height"])
        # Invalid or empty regions become whole-frame masks in the ranges reader.
        if right <= x or bottom <= y:
            continue
        if frame.width > frame.height and placement.get("clearance", {}).get("box"):
            box = placement["clearance"]["box"]
            x, y, right, bottom = box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]
            if placement["kind"] == "title":
                x, y, right, bottom = 0, frame.height // 2, frame.width, frame.height
            if placement["kind"] == "chapter":
                overlays.append({"start_in_output": placement["start_frame"] / 30,
                                 "duration": placement["frames"] / 30,
                                 "region": [120, frame.height - 68, frame.width - 240, 24]})
        elif placement.get("kind") == "chapter":
            # Sync normalises brightness and contrast, so the uniform 45% wash
            # still matches. Exclude the text and varying band, retaining the
            # source above 240 and below 1560 (see the chapter template).
            x, y, right, bottom = 0, 240, 1080, 1560
        overlays.append({"start_in_output": placement["start_frame"] / 30,
                         "duration": placement["frames"] / 30,
                         "region": [x, y, right - x, bottom - y]})
    for item in snapshot.timeline.elements:
        if item.kind in {"insert", "zoom", "background", "grade_compare", "moment"}:
            overlays.append({"start_in_output": item.start_s,
                             "duration": item.end_s - item.start_s,
                             "region": [0, 0, 1080, 1920]})
        elif item.kind == "overlay":
            overlays.append({"start_in_output": item.start_s,
                             "duration": item.end_s - item.start_s,
                             "region": ([0, 0, 1080, 1920]
                                        if (item.payload.get("placement") == "tall_behind"
                                            and tall_box.chosen_framing(item.payload) == "shrink")
                                        else list(item.payload["box_px"]))})
    # Relative source paths make the published EDL portable with the project folder.
    sources = {name: os.path.relpath(source, base) for name, source in sources.items()}
    sources.update(references or {})
    timeline = snapshot.timeline
    if picture:
        overlays.append({"start_in_output": 0, "duration": timeline.total_frames / 30,
                         "region": [0, 0, 1080, 1920], "camera_3d": True})
    reference = picture or "cut/cut.mp4"
    grade = snapshot.grade
    grade_record = ({"grade": {"look": grade.look, "strength": grade.strength,
                                **({"color": grade.color} if grade.color is not None else {}),
                                "version": GRADE_VERSION,
                                "reference": os.path.relpath(snapshot.media[reference], base),
                                "reference_sha256": snapshot.inputs[f"media:{reference}"],
                                "skip_frames": [[item.start_frame, item.end_frame]
                                                for item in timeline.elements
                                                if item.kind == "transition"]}}
                    if grade is not None and grade.active else {})
    from yap_studio.youtube_chapters import timeline_chapters

    youtube = (timeline_chapters(timeline, snapshot.title_layers)
               if frame.width > frame.height else None)
    return {"sources": sources, "fps": "30/1", "composite": True, **grade_record,
            **({"youtube_chapters": youtube} if youtube else {}),
            "ranges": copy.deepcopy(snapshot.spans.get("ranges", snapshot.spans.get("segments"))),
            "overlays": overlays,
            "stills": [{"id": item.id, "start_frame": item.start_frame,
                        "end_frame": item.end_frame}
                       for item in timeline.elements
                       if (item.kind == "insert" and item.payload.get("motion") == "still")
                       or (item.kind == "moment" and item.payload["style"] == "sticker_freeze")],
            "holds": [{"start_frame": timeline.total_frames,
                       "end_frame": timeline.total_frames + timeline.end_hold_frames,
                       "audio": "silence"}] if timeline.end_hold_frames else []}


def _audio_references(snapshot: AssemblySnapshot, retained: Path, work: Path
                      ) -> dict[str, tuple[str, Path]]:
    """Give ranges JSON a portable one-audio source when the cut selected another stream."""
    result = {}
    for index, (name, ordinal) in enumerate(snapshot.audio_streams.items()):
        if not ordinal:
            continue
        reference = work / f"source-{index}.mov"
        ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-i",
                    str(retained / snapshot.sources[name]), "-map", "0:v:0", "-map",
                    f"0:a:{ordinal}", "-c", "copy", "-avoid_negative_ts", "disabled",
                    str(reference)], timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)
        result[name] = (f"source-audio/source-{index}.mov", reference)
    return result


def _camera_transforms(snapshot: AssemblySnapshot) -> list[tuple[float, float, float]]:
    from yap_studio.render.camera3d import face_transforms

    return face_transforms(json.loads(snapshot.small_files["captions/camera3d.json"]))


def _final_records(context: ProjectContext, snapshot: AssemblySnapshot, retained: Path,
                   timeline: ResolvedTimeline, final: Path) -> tuple[Path, Path]:
    """Bind retained caption/card geometry and mapped cut faces to the final bytes."""
    frame = canvas(context.project.shape)
    behind_ids = {item.id for item in timeline.elements
                  if item.kind == "graphic" and item.layer == "behind"}
    captions_layout = json.loads((retained / "captions/layout.json").read_bytes())
    used_ids = {layer["id"] for layer in snapshot.title_layers}
    used_ids.update(caption["id"] for caption in captions_layout["captions"])
    used_ids.update(item.id for item in timeline.elements
                    if item.kind in {"overlay", "background"})
    elements = []
    chapter_numbers = {item.id: chapter_number(index) for index, item in enumerate(
        sorted((item for item in timeline.elements if item.kind == "chapter"),
               key=lambda item: (item.start_frame, item.id)), 1)}
    for layer in snapshot.title_layers:
        start = layer["start_frame"]
        end = start + layer["frames"]
        if frame.width > frame.height and layer.get("clearance", {}).get("box"):
            elements.append({"id": layer["id"], "element_id": layer["id"],
                             "kind": "card", "label": layer["id"],
                             **layer["clearance"]["box"], "start_frame": start, "end_frame": end})
            continue
        if layer["kind"] == "chapter":
            elements.append({"id": layer["id"], "element_id": layer["id"], "kind": "background",
                             "label": (f"chapter {chapter_numbers[layer['id']]}: "
                                       "text sits over the live picture by design"),
                             "x": 0, "y": 0, "w": 1080, "h": 1920,
                             "start_frame": start, "end_frame": end})
            continue
        if layer.get("full_frame"):
            elements.append({"id": layer["id"], "element_id": layer["id"],
                             "kind": "background", "label": layer["id"],
                             "x": 0, "y": 0, "w": 1080, "h": 1920,
                             "start_frame": start, "end_frame": end})
            continue
        pieces = 0
        regions = [Box(part["x"], part["y"], part["width"], part["height"])
                   for part in layer["parts"]] if "parts" in layer else [None]
        drawn = []
        for region in regions:
            runs = drawn_boxes(
                retained / layer["path"], x=layer["x"], y=layer["y"],
                width=layer["width"], height=layer["height"],
                **({"region": region} if region is not None else {}),
            )
            if region is not None and runs:
                left, top = min(box.x for _, _, box in runs), min(box.y for _, _, box in runs)
                right = max(box.x1 for _, _, box in runs)
                bottom = max(box.y1 for _, _, box in runs)
                drawn.append((runs[0][0], runs[-1][1], Box(left, top, right - left, bottom - top)))
            else:
                drawn.extend(runs)
        for first, last, box in drawn:
            piece_start = max(start + first, start)
            piece_end = min(start + last, end)
            if piece_start >= piece_end:
                continue
            pieces += 1
            piece_id = layer["id"]
            if pieces > 1:
                number = 2
                piece_id = f"{layer['id']}#{number}"
                while piece_id in used_ids:
                    number += 1
                    piece_id = f"{layer['id']}#{number}"
                used_ids.add(piece_id)
            element = {"id": piece_id, "element_id": layer["id"],
                       "kind": "background" if layer.get("letters") else
                       "title" if layer["kind"] == "title" else "card",
                       **box.to_dict(), "start_frame": piece_start, "end_frame": piece_end}
            if layer["id"] in behind_ids:
                element["face_windows"] = [list(window) for window in
                    behind_face_windows(timeline, piece_start, piece_end)]
            elements.append(element)
    drawn_ids = {element["id"] for element in elements}
    drawn_ids.update(caption["id"] for caption in captions_layout["captions"])
    for item in timeline.elements:
        if item.kind != "background":
            continue
        original = next(element for element in snapshot.timeline.elements if element.id == item.id)
        elements.append({"id": item.id, "element_id": item.id, "kind": "background",
                         "label": Path(original.payload["media"]["path"]).name,
                         "x": 0, "y": 0, "w": 1080, "h": 1920,
                         "start_frame": item.start_frame, "end_frame": item.end_frame})
    for item in timeline.elements:
        if item.kind != "overlay":
            continue
        original = next(element for element in snapshot.timeline.elements if element.id == item.id)
        if (item.payload.get("placement") == "tall_behind"
                and tall_box.chosen_framing(item.payload) == "shrink"):
            elements.append({"id": item.id, "element_id": item.id, "kind": "background",
                "label": Path(original.payload["media"]["path"]).name,
                "x": 0, "y": 0, "w": 1080, "h": 1920,
                "start_frame": item.start_frame, "end_frame": item.end_frame})
            continue
        for first, last, box in overlay_alpha_runs(retained / item.payload["media"]["path"], item):
            piece_id = item.id
            if piece_id in drawn_ids:
                number = 2
                piece_id = f"{item.id}#{number}"
                while piece_id in used_ids:
                    number += 1
                    piece_id = f"{item.id}#{number}"
            used_ids.add(piece_id)
            drawn_ids.add(piece_id)
            elements.append({"id": piece_id, "element_id": item.id, "kind": "overlay",
                             "label": Path(original.payload["media"]["path"]).name,
                             **box.to_dict(), "start_frame": first, "end_frame": last,
                             **({"face_windows": []}
                                if item.payload.get("placement") == "tall_behind" else {})})
    layout_document = final_layout(
        captions_layout, elements, width=frame.width, height=frame.height,
        clearance_px=face_clearance_px(context.render_style().data),
        caption_gap_px=caption_gap_px(context.render_style().data),
    )
    phrases_path = retained / "captions/phrases.json"
    if phrases_path.is_file():
        layout_document["caption_phrases"] = json.loads(phrases_path.read_bytes())
    layout = write_layout_record(final, layout_document)
    cut = cut_faces(context.project_dir, retained / "cut/cut.mp4")
    cut_source = _source(retained / "cut/cut.mp4", [], [], owner="cut")
    zooms = picture_zooms(_zoom_inputs(retained, timeline, cut_source, [], [], []), cut_source)
    faces, _ = map_picture_faces(
        cut, timeline, zooms=zooms,
        **({"camera": _camera_transforms(snapshot)} if getattr(snapshot, "picture", None) else {}),
        splits=split_windows(timeline, split_layouts(context.project_dir, timeline)),
        video=VideoIdentity(final.name, receipt.sha256_file(final),
                            timeline.total_frames + timeline.end_hold_frames,
                            Fraction(30, 1), frame.width, frame.height),
    )
    return layout, write_faces(final.with_suffix(".faces.json"), faces)


def _measure_contrast(snapshot: AssemblySnapshot, retained: Path, picture: Path, *,
                      halo_ground: bool = True,
                      ) -> tuple[dict, dict]:
    """Measure the silent picture and annotate the caption slots used to render it."""
    try:
        layout = json.loads((retained / "captions/layout.json").read_bytes())
        if getattr(snapshot, "picture", None):
            from yap_studio.render.camera3d_look import measure_contrast

            document = measure_contrast(picture, json.loads(
                snapshot.small_files["captions/camera3d.json"]), layout, grade=snapshot.grade)
        elif snapshot.caption_layer is None:
            # A verified empty layout returns before contrast reads a layer or placement.
            document = measure_caption_contrast(picture, picture, {}, layout)
        else:
            document = measure_caption_contrast(
                picture, retained / snapshot.caption_layer["path"], snapshot.caption_layer, layout,
                caption_style=getattr(snapshot, "caption_style", None),
                **({"halo_ground": False} if not halo_ground else {}),
            )
    except (YapError, OSError, ValueError) as exc:
        document = {
            "schema": CONTRAST_SCHEMA, "measured": False, "reason": str(exc),
            "method": {"sample_every": SAMPLE_EVERY, "held_samples": HELD_SAMPLES,
                       "surround_percentile": SURROUND_PERCENTILE, "opaque_alpha": OPAQUE_ALPHA},
            "captions": [],
        }
    try:
        phrases = json.loads((retained / "captions/phrases.json").read_bytes())["phrases"]
    except (OSError, ValueError):
        phrases = []
    keys = phrase_keys([phrase.get("ids") or [] for phrase in phrases])
    weak = []
    warn_below = Thresholds().caption_contrast_warn
    for caption in document["captions"]:
        caption.update(backing=None, auto=False)
        slot = re.fullmatch(r"caption-([0-9]+)", caption["id"])
        index = int(slot[1]) if slot is not None else -1
        slots = getattr(snapshot, "caption_slots", ())
        if 0 <= index < len(slots):
            caption.update(slots[index])
        if caption["ratio"] is None or caption["ratio"] >= warn_below:
            continue
        item = {key: caption[key] for key in (
            "id", "text", "ratio", "word", "frame", "start_frame", "end_frame", "backing", "auto",
        )}
        if 0 <= index < len(phrases) and phrases[index]["text"] == caption["text"]:
            item["ids"] = phrases[index]["ids"]
            item["key"] = keys[index]
        weak.append(item)
    return document, {"measured": document["measured"], "reason": document["reason"],
                      "captions": len(document["captions"]), "weak": weak}


def _final_contrast(snapshot: AssemblySnapshot, retained: Path, final: Path, *,
                    measurement: tuple[dict, dict] | None = None) -> tuple[Path, dict]:
    """Bind the silent picture's measurement to the muxed final."""
    document, summary = measurement or _measure_contrast(snapshot, retained, final)
    return write_contrast_record(final, document), summary


def _remove_old_work(work_root: Path, prefix: str, current: Path) -> None:
    """Remove abandoned render folders after StageRun excludes a live peer."""
    for folder in work_root.iterdir():
        if (folder != current and folder.name.startswith(prefix) and
                folder.is_dir() and not folder.is_symlink()):
            shutil.rmtree(folder, ignore_errors=True)


def _clear_stale(root: Path) -> None:
    """Remove only stale files still owned by the stale manifest."""
    folder = root / "final"
    manifest = folder / "stale.json"
    if not manifest.exists():
        return
    data = json.loads(manifest.read_bytes())
    if data.get("schema") != "yap-studio/stale-final@1":
        raise AssemblyError("final/stale.json is not a stale-final manifest.")
    targets = []
    for output in data.get("outputs", []):
        target = (folder / output["path"]).resolve()
        if not target.is_relative_to(folder) or target == manifest:
            raise AssemblyError("final/stale.json names a file outside final/.")
        if target.exists():
            paths.validate_output(target, owner_receipts=(manifest,))
            targets.append(target)
    for target in targets:
        target.unlink()
    paths.validate_output(manifest)
    manifest.unlink()


def _mark_stale(root: Path, name: str, reason: str) -> str | None:
    """Move an owned final and its owned sidecars behind a stale manifest."""
    from yap_studio.project.lock import timestamp

    folder = root / "final"
    final = folder / f"{name}.mp4"
    receipt_path = folder / "receipt.json"
    pending_path = folder / "receipt.pending.json"
    with project_lock(root):
        progress = _read_json(folder / "progress.json")
        if progress and pid_alive(progress.get("pid")):
            return None
        owners = (receipt_path, pending_path)
        manifest = folder / "stale.json"
        stale_path = f"final/{name}.stale.mp4"
        sources = [final, *(final.with_suffix(suffix) for suffix in
                            (LAYOUT_SUFFIX, CONTRAST_SUFFIX, ".faces.json", ".taste.json",
                             ".chapters.txt"))]
        if manifest.exists():
            data = json.loads(manifest.read_bytes())
            expected = {f"{name}.stale{source.name[len(name):]}": source
                        for source in sources}
            outputs = data.get("outputs") if isinstance(data, dict) else None
            if (not isinstance(data, dict) or
                    data.get("schema") != "yap-studio/stale-final@1" or
                    not isinstance(outputs, list) or
                    any(not isinstance(item, dict) or item.get("path") not in expected or
                        not isinstance(item.get("sha256"), str)
                        for item in outputs) or
                    len({item["path"] for item in outputs}) != len(outputs)):
                raise AssemblyError("final/stale.json is not a valid stale-final manifest.")
            previous = read_receipt(folder) or {}
            pending_final = (final.is_file() and
                             not (folder / f"{name}.stale.mp4").exists() and
                             any(item["path"] == f"{name}.stale.mp4" and
                                 item["sha256"] == receipt.sha256_file(final)
                                 for item in outputs))
            verified_current = final.is_file() and previous.get("status") == "verified" and any(
                item.get("path") == f"final/{name}.mp4" and
                item.get("sha256") == receipt.sha256_file(final)
                for item in previous.get("outputs", [])) and not pending_final
            if not verified_current:
                for item in outputs:
                    source = expected[item["path"]]
                    stale = folder / item["path"]
                    if (source.is_file() and not stale.exists() and
                            receipt.sha256_file(source) == item["sha256"] and
                            paths._is_owned_output(str(source), owners)):
                        paths.validate_output(source, allowed_existing_suffixes=frozenset(),
                                              owner_receipts=owners)
                        paths.validate_output(stale, allowed_existing_suffixes=frozenset(),
                                              owner_receipts=(manifest,))
                        os.replace(source, stale)
                if not final.is_file():
                    return stale_path if (folder / f"{name}.stale.mp4").is_file() else None
                for source in sources:
                    if source.is_file() and paths._is_owned_output(str(source), owners):
                        paths.validate_output(source, owner_receipts=owners)
                        source.unlink()
                return stale_path if (folder / f"{name}.stale.mp4").is_file() else None
        if not final.is_file() or not paths._is_owned_output(str(final), owners):
            return None
        _clear_stale(root)
        moves = []
        for source in sources:
            if source.is_file() and paths._is_owned_output(str(source), owners):
                stale = folder / f"{name}.stale{source.name[len(name):]}"
                moves.append((source, stale))
        outputs = [{"path": stale.name, "sha256": receipt.sha256_file(source),
                    "bytes": source.stat().st_size} for source, stale in moves]
        paths.atomic_write_text(manifest, json.dumps({
            "schema": "yap-studio/stale-final@1", "outputs": outputs,
            "reason": reason or "stopped", "at": timestamp(),
        }, indent=2) + "\n")
        for source, stale in moves:
            paths.validate_output(source, allowed_existing_suffixes=frozenset(),
                                  owner_receipts=owners)
            paths.validate_output(stale, allowed_existing_suffixes=frozenset(),
                                  owner_receipts=(manifest,))
            os.replace(source, stale)
    return stale_path


def _bleep_words(context: ProjectContext, snapshot: AssemblySnapshot
                 ) -> list[bleep.Occurrence] | None:
    """Detect corrected retained words only when the switch is on."""
    if not bleep.enabled(context.render_style().data):
        return None
    return bleep.find_display(_retained_words(snapshot),
                              bleep.listed(context.render_style().data["music"]),
                              word_fixes=json.loads(snapshot.small_files["project.json"]).get(
                                  "word_fixes", {}),
                              word_text_fixes=json.loads(snapshot.small_files["project.json"]).get(
                                  "word_text_fixes", {}))


def _retained_words(snapshot: AssemblySnapshot) -> tuple[CutWord, ...]:
    return cast(CutWords, parse_document(
        "cut-words", json.loads(snapshot.small_files["cut/cut-words.json"]))).words


def _bleep_check(context: ProjectContext, snapshot: AssemblySnapshot, sound: Any,
                 retained: Path, work: Path, *, letters_layers: list[dict] | None = None
                 ) -> list[dict]:
    """Check retained text and the exact PCM voice input used by the mix."""
    style = context.render_style().data
    if not bleep.enabled(style):
        return []
    original = decode(retained / "cut/voice.wav", [], channels=ffmpeg.probe(
        retained / "cut/voice.wav").channels or 1)
    voice = (np.fromfile(sound.bleep_voice, dtype="<f4").reshape(original.shape)
             if sound.bleep_voice is not None else original)
    tone_path = work / "bleep.f32le"
    tone = (np.fromfile(tone_path, dtype="<f4").reshape(-1, 2) if tone_path.is_file() else None)
    from yap_studio import cards

    rules = {item.payload["template"]: cards.load_template(item.payload["template"])["slots"]
        for item in snapshot.timeline.elements if item.kind == "graphic"}
    return [bleep.check(_retained_words(snapshot), music=style["music"], voice=voice,
                         original_voice=original, tone_stem=tone,
                         phrases=json.loads(snapshot.small_files["captions/phrases.json"]),
                         title_layers=[*snapshot.title_layers, *(letters_layers or [])],
                         timeline=snapshot.timeline,
                         slot_rules=rules,
                         word_fixes=json.loads(snapshot.small_files["project.json"]).get(
                             "word_fixes", {}),
                         word_text_fixes=json.loads(snapshot.small_files["project.json"]).get(
                             "word_text_fixes", {}))]


def _pilot_assemble(context: ProjectContext, snapshot: AssemblySnapshot,
                    frames: tuple[int, int], commands: list[list[str]],
                    phrases_bytes: bytes, phrases_sha256: str, *, force: bool
                    ) -> dict[str, Any]:
    """Publish a sliced render with its own receipt and owned media outputs."""
    _pilot_window(snapshot, frames, phrases_bytes)
    root = context.project_dir
    first, last = frames
    count = last - first + 1
    total = snapshot.timeline.total_frames + snapshot.timeline.end_hold_frames
    out = f"pilot/{context.project.name}.pilot-{first}-{last}.mp4"
    result = {"status": "verified", "pilot": True, "frames": [first, last],
              "out": out, "receipt": "pilot/receipt.json",
              "notices": snapshot.sfx.get("notices", [])}
    run = StageRun(root, "pilot")
    inputs = {**snapshot.inputs,
              "pilot:frames": projection_sha256({"frames": [first, last]}),
              "pilot:phrases": phrases_sha256}
    inputs_sha256 = projection_sha256(inputs)
    range_key = f"{first}-{last}"
    previous = read_receipt(run.stage_dir) or {}
    entries = previous.get("ranges", {})
    entry = entries.get(range_key) if isinstance(entries, dict) else None
    owned = isinstance(entry, dict) and any(
        item.get("path") == out and item.get("sha256") == entry.get("sha256")
        for item in previous.get("outputs", []))
    if (not force and previous.get("status") == "verified" and
            not run.pending_path.exists() and not run.progress_path.exists() and owned and
            entry.get("inputs_sha256") == inputs_sha256 and entry.get("out") == out and
            (root / out).is_file() and
            receipt.sha256_file(root / out) == entry.get("sha256")):
        return {**result, "noop": True}
    paths.validate_output(root / out, allowed_existing_suffixes=frozenset(),
                          owner_receipts=(run.receipt_path, run.pending_path))
    retained_before = retained_outputs(root, previous, read_pending(run.stage_dir) or {})
    work_root = paths.validate_output_dir(root / ".yap/work")
    with tempfile.TemporaryDirectory(prefix="pilot-", dir=work_root) as folder, run:
        work = Path(folder)
        _remove_old_work(work_root, "pilot-", work)
        for name, value in inputs.items():
            run.record_inputs(name, value)
        run.outputs.update(retained_before)
        ranges = {key: item for key, item in entries.items()
                  if isinstance(item, dict) and item.get("out") in retained_before and
                  item.get("sha256") == retained_before[item["out"]]["sha256"]} \
            if isinstance(entries, dict) else {}
        retained = work / "snapshot"
        timeline = _retained(snapshot, retained)
        run.progress("assembling picture", 0.1)
        picture_path = work / "picture.mp4"
        base_picture = getattr(snapshot, "picture", None)
        source_picture = retained / (base_picture or "cut/cut.mp4")
        transitions = render_transitions(root, timeline, source_picture, work) \
            if any(item.kind == "transition" for item in timeline.elements) else []
        smooth, joins = jump_cuts.render_joins(root, timeline, retained / "cut/cut.mp4",
                                              context.render_style().data["moves"],
                                              placements=snapshot.title_layers)
        plates = letters.render_plates(root, timeline, context.render_style(), work) \
            if any("letters" in item.payload for item in timeline.elements) else []
        picture = render_picture(retained, picture_path, work_dir=work, timeline=timeline,
                                 **({"frame": canvas(context.project.shape)}
                                    if context.project.shape == "horizontal" else {}),
                                 title_layers=snapshot.title_layers,
                                 caption_layer=snapshot.caption_layer, frames=frames,
                                 grade=snapshot.grade, grade_style=snapshot.grade_style,
                                 moment_style=context.render_style().data,
                                 **({"base_picture": source_picture}
                                    if base_picture or timeline.camera_3d else {}),
                                 **({"transition_layers": transitions} if transitions else {}),
                                 **({"smooth_layers": smooth} if smooth else {}),
                                 **({"letters_layers": plates} if plates else {}),
                                 matte_project_dir=root,
                                 cutout=context.render_style().data["broll"]["cutout"])
        run.progress("mixing audio", 0.5)
        audio_path = work / "audio.m4a"
        bleeps = _bleep_words(context, snapshot)
        sound = render_audio(retained, audio_path, work_dir=work, timeline=timeline,
                             clips=_audio_clips(snapshot),
                             policy=context.project.audio,
                             music=context.render_style().data["music"],
                             voice=context.render_style().data["voice"], frames=frames,
                             **({"bleeps": bleeps} if bleeps is not None else {}),
                             cut_words=json.loads(snapshot.small_files["cut/cut-words.json"])
                             if "cut/cut-words.json" in snapshot.small_files else None,
                             bed=retained / snapshot.bed if snapshot.bed else None,
                             sfx=snapshot.sfx,
                             flow_look=context.render_style().data["graphics"]["flow_look"],
                             specs_rows=context.render_style().data["graphics"]["specs_rows"],
                             sfx_settings=context.render_style().data["internal"])
        pilot = work / f"{context.project.name}.mp4"
        ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(picture_path),
                    "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
                    "-metadata", f"comment=yap-studio pilot frames {first}-{last} "
                    f"of {total}; not a final", "-movie_timescale", "48000",
                    "-movflags", "+faststart", str(pilot)],
                   timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)
        run.progress("checking pilot audio", 0.8)
        asserts, measured = delivery_asserts(pilot, frames=count,
                                             policy=context.project.audio, commands=[])
        asserts = [item for item in asserts if item["id"] != "audio.integrated"]
        asserts.extend([
            {"id": "picture.frames", "pass": ffmpeg.probe(pilot).video_frames == count,
             "expected": count},
            {"id": "audio.pcm_samples", "pass": sound.samples == count * 1600,
             "expected": count * 1600},
        ])
        asserts.extend(_bleep_check(context, snapshot, sound, retained, work,
                                   letters_layers=plates))
        extra = {"pilot": {"frames": [first, last], "frame_count": count,
                           "samples": count * 1600, "of_frames": total, "final": False},
                 "audio": {**sound.to_dict(), "output": measured.to_dict()},
                 "picture": picture.to_dict(), "timeline": snapshot.timeline.to_dict(),
                 "tools": ffmpeg.tool_versions(), "commands": commands,
                 "ranges": ranges,
                 "verify": {"status": "skipped", "reason": "pilot"},
                 "style": {"name": context.render_style().name,
                           "sha256": context.render_style().sha256()}}
        if joins is not None:
            extra["jump_cuts"] = joins.to_dict()
        if not all(item["pass"] for item in asserts):
            run.finish("failed", asserts=asserts, extra=extra)
            raise AssemblyError("Pilot delivery failed: " + ", ".join(
                failed_assert_lines(asserts)), failed_checks=True,
                suggested_action=next((item["fix"] for item in asserts
                                       if not item["pass"] and "fix" in item), None))
        run.replace_output(root / out, pilot)
        ranges[range_key] = {"inputs_sha256": inputs_sha256, "out": out,
                             "sha256": run.outputs[out]["sha256"]}
        run.finish(asserts=asserts, extra=extra)
    return {**result, "noop": False}


def _assemble_impl(project_dir: Path, *, force: bool = False,
                   allow_violations: bool = False, frames: tuple[int, int] | None = None,
                   halo_added: list[str] | None = None,
                   ) -> dict[str, Any]:
    """Render and verify a final, publishing its owned outputs and receipt last."""
    with paths.operation():
        context = ProjectContext(project_dir)
        validate_voice_chain(context.project.audio, context.render_style().data["voice"])
        commands: list[list[str]] = []
        with _record_commands(commands):
            refreshed: list[str] = []
            while True:
                try:
                    require_top_text(context)
                    snapshot = assembly_snapshot(context, allow_violations=allow_violations)
                    break
                except StaleStageError as exc:
                    if (exc.stage in refreshed or
                            (exc.stage == "music" and context.project.music["mode"]
                             not in {"none", "file"})):
                        raise
                    from yap_studio.stages import captions as captions_stage
                    from yap_studio.stages import graphics as graphics_stage
                    from yap_studio.stages import music as music_stage

                    {"captions": captions_stage.captions, "graphics": graphics_stage.graphics,
                     "music": music_stage.music}[exc.stage](
                         context.project_dir,
                         **({"allow_violations": allow_violations}
                            if exc.stage == "captions" else {}))
                    refreshed.append(exc.stage)
                    context = ProjectContext(context.project_dir)
            phrases = _pilot_phrases(context) if frames is not None else None
            if snapshot.sfx["kit"] != "none" and not snapshot.sfx["ready"]:
                account = elevenlabs.account_state()
                sound_effects.remember_account(account)
                snapshot.sfx["account"] = account
                if account in {"missing_cli", "signed_out", "free"}:
                    snapshot.sfx["notices"] = elevenlabs.show_once(
                        context.project_dir, "elevenlabs_sound_effects",
                        sound_effects.signup(account))
            if "fallback_from" in snapshot.sfx:
                kit = snapshot.sfx["fallback_from"]
                snapshot.sfx["notices"] = elevenlabs.show_once(
                    context.project_dir, "elevenlabs_sound_effects_made_fallback",
                    ["Your ElevenLabs kit isn't ready, so the automatic sound effects "
                     "in this video come from Made by Yap Studio. "
                     f"To make the ElevenLabs kit, run yap-studio sfx kit --kit {kit} --yes."])
            for key, kit in (("missing_extras", "extras"), ("missing_requests", "soft")):
                if snapshot.sfx.get(key):
                    snapshot.sfx.setdefault("notices", []).extend(elevenlabs.show_once(
                        context.project_dir, f"elevenlabs_sound_effects_{key}",
                        [f"Sound effects: some {kit} sounds are missing or stale. "
                         "This video skips those sounds.",
                         f"Run yap-studio sfx kit --kit {kit} --yes to make them."]))
            if frames is not None:
                assert phrases is not None
                _pilot_window(snapshot, frames, phrases[0])
            if all(take.voiceover for take in context.project.takes):
                require_voiceover_picture(snapshot.timeline.total_frames,
                                          [item.to_dict() for item in snapshot.timeline.elements],
                                          frames=frames,
                                          graphic_coverage=_graphic_picture_frames(
                                              context.project_dir, snapshot, frames))
            if frames is not None:
                assert phrases is not None
                return {**_pilot_assemble(context, snapshot, frames, commands, *phrases,
                                          force=force), "refreshed": refreshed}
            root = context.project_dir
            # The public stage is assemble; the delivery artifacts live under final.
            run = StageRun(root, "final")
            run.stage = "assemble"
            run.owner_receipts = (root / "final/stale.json",)
            out = f"final/{context.project.name}.mp4"
            result = {"status": "verified", "out": out, "receipt": "final/receipt.json",
                      "edl": "final/verify-edl.json", "refreshed": refreshed,
                      "notices": snapshot.sfx.get("notices", []),
                      "caption_halo": {"added": len(halo_added or [])}}
            inputs = snapshot.inputs
            if not force and run.up_to_date(inputs):
                previous = read_receipt(run.stage_dir) or {}
                halos = caption_halos(context)
                needs_halo = halo_added is None and any(
                    phrase.get("key") is not None
                    and phrase["key"] not in context.project.caption_backing
                    and phrase["key"] in snapshot.caption_windows
                    and halos.get(phrase["key"], {}).get("window") !=
                    snapshot.caption_windows[phrase["key"]]
                    for phrase in previous.get("caption_contrast", {}).get("weak", [])
                )
                if not needs_halo and (
                    allow_violations or not previous.get("allowed_layout_violations")
                ):
                    with project_lock(root):
                        _clear_stale(root)
                    return {**result, "noop": True}
            targets = [out, str(Path(out).with_suffix(LAYOUT_SUFFIX)),
                       str(Path(out).with_suffix(CONTRAST_SUFFIX)),
                       str(Path(out).with_suffix(".faces.json")), "final/verify-edl.json", *[
                f"final/source-audio/source-{index}.mov"
                for index, ordinal in enumerate(snapshot.audio_streams.values()) if ordinal
            ]]
            for relative in targets:
                paths.validate_output(root / relative, allowed_existing_suffixes=frozenset(),
                                      owner_receipts=(run.receipt_path, run.pending_path,
                                                      *run.owner_receipts))
            previous = read_receipt(run.stage_dir) or {}
            retained_before = retained_outputs(root, previous,
                                               read_pending(run.stage_dir) or {})
            work_root = paths.validate_output_dir(root / ".yap/work")
            with tempfile.TemporaryDirectory(prefix="assemble-", dir=work_root) as folder, run:
                work = Path(folder)
                _remove_old_work(work_root, "assemble-", work)
                for name, digest in inputs.items():
                    run.record_inputs(name, digest)
                # Keep ownership if a name or selected-stream reference is later restored.
                run.outputs.update(retained_before)
                retained = work / "snapshot"
                timeline = _retained(snapshot, retained)
                run.progress("assembling picture", 0.1)
                picture_path = work / "picture.mp4"
                base_picture = getattr(snapshot, "picture", None)
                source_picture = retained / (base_picture or "cut/cut.mp4")
                transitions = render_transitions(root, timeline, source_picture, work) \
                    if any(item.kind == "transition" for item in timeline.elements) else []
                smooth, joins = jump_cuts.render_joins(root, timeline, retained / "cut/cut.mp4",
                                                      context.render_style().data["moves"],
                                                      placements=snapshot.title_layers)
                plates = letters.render_plates(root, timeline, context.render_style(), work) \
                    if any("letters" in item.payload for item in timeline.elements) else []
                picture = render_picture(retained, picture_path, work_dir=work,
                                         **({"frame": canvas(context.project.shape)}
                                            if context.project.shape == "horizontal" else {}),
                                         timeline=timeline, title_layers=snapshot.title_layers,
                                         caption_layer=snapshot.caption_layer,
                                         grade=snapshot.grade, grade_style=snapshot.grade_style,
                                         moment_style=context.render_style().data,
                                         **({"base_picture": source_picture}
                                            if base_picture or timeline.camera_3d else {}),
                                         **({"transition_layers": transitions} if transitions
                                            else {}),
                                         **({"smooth_layers": smooth} if smooth else {}),
                                         **({"letters_layers": plates} if plates else {}),
                                         matte_project_dir=root,
                                         cutout=context.render_style().data["broll"]["cutout"])
                run.progress("checking caption contrast", 0.4)
                measurement = _measure_contrast(snapshot, retained, picture_path,
                                                halo_ground=halo_added is not None)
                if halo_added is None:
                    added = add_caption_halos(root, measurement[1]["weak"],
                                              snapshot.caption_windows)
                    if added:
                        run.progress("adding a halo to weak captions", 0.45)
                        run.cancel()
                        raise _HaloRedo(added)
                    if any(slot.get("backing") == "shadow" for slot in snapshot.caption_slots):
                        measurement = _measure_contrast(snapshot, retained, picture_path)
                measurement[1]["halo_added"] = halo_added or []
                run.progress("mixing audio", 0.5)
                audio_path = work / "audio.m4a"
                bleeps = _bleep_words(context, snapshot)
                sound = render_audio(retained, audio_path, work_dir=work, timeline=timeline,
                                     clips=_audio_clips(snapshot),
                                     policy=context.project.audio,
                                     music=context.render_style().data["music"],
                                     voice=context.render_style().data["voice"],
                                     **({"bleeps": bleeps} if bleeps is not None else {}),
                                     cut_words=json.loads(snapshot.small_files["cut/cut-words.json"])
                                     if "cut/cut-words.json" in snapshot.small_files else None,
                                     bed=retained / snapshot.bed if snapshot.bed else None,
                                     sfx=snapshot.sfx,
                                     flow_look=context.render_style().data["graphics"]["flow_look"],
                                     specs_rows=context.render_style().data["graphics"]["specs_rows"],
                                     sfx_settings=context.render_style().data["internal"])
                final = work / f"{context.project.name}.mp4"
                ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(picture_path),
                            "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
                            "-movie_timescale", "48000", "-movflags", "+faststart", str(final)],
                           timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)
                frames = timeline.total_frames + timeline.end_hold_frames
                layout_path, faces_path = _final_records(context, snapshot, retained, timeline,
                                                        final)
                contrast_path, contrast_summary = _final_contrast(
                    snapshot, retained, final, measurement=measurement,
                )
                run.progress("checking final audio", 0.8)
                asserts, measured = delivery_asserts(final, frames=frames,
                                                     policy=context.project.audio, commands=[])
                asserts.extend([
                    {"id": "picture.frames", "pass": ffmpeg.probe(final).video_frames == frames,
                     "expected": frames},
                    {"id": "audio.pcm_samples", "pass": sound.samples == frames * 1600},
                ])
                asserts.extend(_bleep_check(context, snapshot, sound, retained, work,
                                           letters_layers=plates))
                run.progress("verifying final", 0.9)
                references = _audio_references(snapshot, retained, work)
                edl = verification_edl(snapshot, base=root / "final",
                                       references={name: relative for name, (relative, _)
                                                   in references.items()})
                verify_path = work / "verify-edl.json"
                jump_cuts.skip_grade(edl, smooth)
                # Verification reads retained sources; publication uses portable source paths.
                checking_edl = {**edl, "sources": {
                    name: str(references[name][1] if name in references else retained / relative)
                    for name, relative in snapshot.sources.items()
                }}
                if "grade" in checking_edl:
                    checking_edl["grade"] = {**edl["grade"],
                                             "reference": str(source_picture)}
                verify_path.write_text(json.dumps(checking_edl), encoding="utf-8")
                policy = context.project.audio
                application = context.taste_application()
                if application.applied or application.enforced or application.skipped:
                    write_taste_record(final, application)
                windows = [TransitionWindow(item.payload["join_frame"], item.start_frame,
                                            item.end_frame, item.id)
                           for item in timeline.elements if item.kind == "transition"]
                windows.extend(jump_cuts.verification_windows(smooth))
                words_stem = getattr(sound, "words_stem", None)
                click_stem = sound.click_stem or words_stem
                report = verify(str(final), edl=str(verify_path),
                                platform=context.project.platform,
                                asr=context.project.asr["engine"],
                                asr_model=context.project.asr.get("model"),
                                loudness="check" if policy["mode"] == "loudnorm" else "note",
                                loudness_target=policy.get("integrated_lufs", -14), detail="full",
                                **({"click_audio": str(click_stem)}
                                   if click_stem is not None else {}),
                                **({"words_audio": str(words_stem)}
                                   if words_stem is not None else {}),
                                **({"transitions": windows} if windows else {}))
                add_number_visibility(report, asserts, getattr(picture, "behind_layers", ()))
                failures = [item for item in report.get("findings", [])
                            if item["severity"] == "fail"]
                layout_failures = [item for item in failures if item["check"] in
                                   {FACES_CLEARANCE_CHECK, CAPTIONS_CARDS_CHECK}
                                   or (item["check"] == "taste.rules" and
                                       item.get("evidence", {}).get("check") ==
                                       FACES_CLEARANCE_CHECK)]
                allowed = allow_violations and bool(failures) and failures == layout_failures
                asserts.append({"id": "verify.status",
                                "pass": report["status"] != "fail" or allowed,
                                "expected": "pass",
                                "actual": report["status"]})
                audio_details = {**sound.to_dict(), "output": measured.to_dict()}
                extra = {"audio": audio_details, "picture": picture.to_dict(),
                         "timeline": snapshot.timeline.to_dict(),
                         "allow_violations": allow_violations,
                         "tools": ffmpeg.tool_versions(),
                         "commands": commands, "verify": report,
                         "caption_contrast": contrast_summary,
                         "taste": context.taste_application().pin(),
                         "style": {"name": context.render_style().name,
                                   "sha256": context.render_style().sha256()}}
                if joins is not None:
                    extra["jump_cuts"] = joins.to_dict()
                if picture.grade is not None:
                    extra["grade"] = {**picture.grade,
                                      **({"sample": snapshot.grade_sample}
                                         if snapshot.grade_sample is not None else {})}
                if allowed:
                    extra["allowed_layout_violations"] = [
                        {key: item.get(key) for key in ("check", "time_s", "message")}
                        for item in layout_failures
                    ]
                if transitions:
                    extra["transitions"] = [
                        {"id": item.id, "look": item.payload["look"],
                         "frames": item.payload["frames"], "start_frame": item.start_frame,
                         "join_frame": item.payload["join_frame"]}
                        for item in timeline.elements if item.kind == "transition"
                    ]
                if getattr(picture, "behind_layers", ()):
                    extra["behind_layers"] = list(picture.behind_layers)
                if getattr(picture, "cutout", None) is not None:
                    extra["cutout"] = picture.cutout
                if getattr(picture, "green_screen", ()):
                    extra["green_screen"] = list(picture.green_screen)
                if not all(item["pass"] for item in asserts):
                    run.finish("failed", asserts=asserts, extra=extra)
                    action = next((item["fix"] for item in asserts
                                   if not item["pass"] and "fix" in item), None)
                    message = "Final delivery failed: " + ", ".join(
                        failed_assert_lines(asserts))
                    if layout_failures:
                        details = []
                        for item in layout_failures:
                            when = format_time(item["time_s"]) if item.get("time_s") is not None \
                                else "unknown time"
                            evidence = item.get("evidence", {})
                            element = evidence.get("element_id", evidence.get("target_id", ""))
                            details.append(f"{when} {element}: {item['message']}")
                        action = ("Captions can be placed again (re-run captions) or the card "
                                  "re-planned (re-run graphics).")
                        if failures == layout_failures:
                            action += " --allow-violations publishes anyway."
                        message += "; " + "; ".join(details) + ". " + action
                    raise AssemblyError(message, failed_checks=True, suggested_action=action)
                verify_path.write_text(json.dumps(edl, indent=2) + "\n", encoding="utf-8")
                run.replace_output(root / out, final)
                if report.get("youtube_chapters"):
                    chapter_path = work / "chapters.txt"
                    chapter_path.write_text(report["youtube_chapters"], encoding="utf-8")
                    run.replace_output((root / out).with_suffix(".chapters.txt"), chapter_path)
                else:
                    chapter_path = (root / out).with_suffix(".chapters.txt")
                    if chapter_path.exists():
                        with project_lock(root):
                            paths.validate_output(chapter_path, owner_receipts=(run.pending_path,))
                            old = receipt.sha256_file(chapter_path)
                            run._journal_output({"path": str(chapter_path.relative_to(root)),
                                                 "old_sha256": old, "new_sha256": None})
                            chapter_path.unlink()
                            run.outputs.pop(str(chapter_path.relative_to(root)), None)
                run.replace_output((root / out).with_suffix(LAYOUT_SUFFIX), layout_path)
                run.replace_output((root / out).with_suffix(CONTRAST_SUFFIX), contrast_path)
                run.replace_output((root / out).with_suffix(".faces.json"), faces_path)
                with project_lock(root):
                    taste_record = write_taste_record(
                        root / out, context.taste_application(), out=work / "render.taste.json",
                    )
                run.replace_output((root / out).with_suffix(".taste.json"), taste_record)
                run.replace_output(root / "final/verify-edl.json", verify_path)
                for relative, reference in references.values():
                    run.replace_output(root / "final" / relative, reference)
                run.finish(asserts=asserts, extra=extra)
            with project_lock(root):
                _clear_stale(root)
            return {**result, "noop": False}


def assemble(project_dir: Path, *, force: bool = False, allow_violations: bool = False,
             frames: tuple[int, int] | None = None
             ) -> dict[str, Any]:
    """Render a final and mark any owned predecessor stale on failure."""
    with paths.operation():
        try:
            try:
                return _assemble_impl(project_dir, force=force,
                                      allow_violations=allow_violations, frames=frames)
            except _HaloRedo as redo:
                return _assemble_impl(project_dir, force=force, allow_violations=allow_violations,
                                      halo_added=redo.keys)
        except BaseException as exc:
            if isinstance(exc, ProcessingError) and not exc.suggested_action:
                rerun = f"yap-studio run {shlex.quote(str(project_dir))}"
                if frames is not None:
                    rerun += f" --frames {frames[0]}-{frames[1]}"
                exc.suggested_action = ("Run yap-studio doctor to check ffmpeg, then run "
                                        f"{rerun} again.")
            if frames is None and getattr(exc, "code", None) not in {
                "stage_running", "project_locked",
            }:
                try:
                    context = ProjectContext(project_dir)
                    moved = _mark_stale(context.project_dir, context.project.name, str(exc))
                except (YapError, OSError, ValueError):
                    moved = None
                if moved:
                    sentence = f"The previous final was moved to {moved}."
                    if isinstance(exc, YapError):
                        exc.args = (f"{exc.message} {sentence}",)
                    else:
                        exc.add_note(sentence)
            raise
