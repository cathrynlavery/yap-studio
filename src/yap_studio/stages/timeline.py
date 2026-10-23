"""Resolve word-anchored lanes and check rules before atomic lane transactions."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shlex
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from yap_studio import cards, ffmpeg, letters, paths, receipt
from yap_studio.captions import caption_layout_settings
from yap_studio.edl import READERS
from yap_studio.errors import YapError
from yap_studio.faces.clearance import caption_gap_px, face_clearance_px
from yap_studio.faces.model import Box, FacesData
from yap_studio.joins.steps import plan_steps
from yap_studio.matte import chroma
from yap_studio.media.captured import capture_time, fingerprint
from yap_studio.opening_push import OPENING_SCALE, end_scale, opening_end
from yap_studio.plan import Plan, snap_frame
from yap_studio.planner.number_spoken import spoken_number_fix
from yap_studio.project.anchors import resolve_anchor
from yap_studio.project.lock import ConflictError, project_lock, write_json_cas
from yap_studio.project.model import (
    Anchor,
    CutWord,
    CutWords,
    Edit,
    Element,
    ProjectFileError,
    TimelineLane,
    Words,
    load_timeline_fragment,
    parse_anchor,
    parse_document,
)
from yap_studio.project.status import STAGES, ProjectContext, TakeChangedError, _stage_state
from yap_studio.render.moments import check_moments
from yap_studio.stages import green_screen, tall_box
from yap_studio.stages.broll_limits import broll_limits
from yap_studio.stages.broll_split import check_split_voiceover
from yap_studio.stages.voiceover import frame_ranges_text, uncovered_frames
from yap_studio.taste.store import load_taste, taste_path
from yap_studio.text_hook import TEXT_HOOK_ID
from yap_studio.text_rules import apply_text_rules
from yap_studio.transcript.words import title_words
from yap_studio.zoom_curve import (
    CAPTION_SLACK_PX,
    FACE_MARGIN_PX,
    SCALE_LIMIT,
    SNAP_IN_FRAMES,
    SNAP_OUT_FRAMES,
    ZoomCurve,
    ZoomKey,
    declared_moves,
    face_center,
    face_leaves_frame,
    punch_keys,
    push_keys,
    scale_motion,
)

FPS = Fraction(30)
TIMELAPSE_SPEED_MAX = 20
CHRONOLOGY_SLACK = timedelta(hours=24)
NEAR_DUPLICATE_MAX = 24
LANES = ("agent", "broll")
VISUAL_KINDS = frozenset({"title", "chapter", "graphic", "insert", "overlay", "background", "zoom"})
CANVAS_WIDTH_PX = 1080
CANVAS_HEIGHT_PX = 1920
# Side boxes share the canvas edge margin and sit below the headroom band.
OVERLAY_MARGIN_PX = 60
OVERLAY_SIDE_GAP_PX = 40
OVERLAY_SIDE_SIZE_PX = (440, 560)


def _is_image(info: ffmpeg.MediaInfo, source: Path) -> bool:
    return source.suffix.lower() in {".heic", ".heif"} or any(
        name in {"image2", "image2pipe"} or name.endswith("_pipe")
        for name in (info.format_name or "").split(",")
    )


def _caption_band_top(style: Mapping) -> int:
    from yap_studio.stages.captions import caption_band_geometry

    settings = caption_layout_settings(style["captions"], style["brand"])
    return min(caption_band_geometry(
        settings, safe_bottom_px=style["internal"]["caption_safe_bottom_px"],
        backing=backing).y for backing in ((), ("box",), ("shadow",)))


def overlay_box(payload: Mapping, style_data: Mapping) -> tuple[int, int, int, int]:
    """Resolve an overlay destination in pixels on the vertical canvas."""
    placement = payload["placement"]
    if placement == "tall_behind":
        return tall_box.TALL_BOX_PX
    if placement == "box":
        x, y, width, height = payload["box"]
        return round(x), round(y), round(width), round(height)
    internal = style_data["internal"]
    top, bottom = internal["key_text_top_px"], internal["key_text_bottom_px"]
    if placement == "headroom":
        return OVERLAY_MARGIN_PX, top, CANVAS_WIDTH_PX - 2 * OVERLAY_MARGIN_PX, bottom - top
    width, height = OVERLAY_SIDE_SIZE_PX
    if placement == "side_left":
        return OVERLAY_MARGIN_PX, bottom + OVERLAY_SIDE_GAP_PX, width, height
    if placement == "side_right":
        return (CANVAS_WIDTH_PX - OVERLAY_MARGIN_PX - width,
                bottom + OVERLAY_SIDE_GAP_PX, width, height)
    raise ValueError(f"Unknown overlay placement {placement!r}")


class TimelineRefusal(YapError):
    """Refuse an unsafe lane write or resolution against an obsolete cut."""

    def __init__(self, message: str, *, code: str = "timeline_refused",
                 suggested_action: str | None = None) -> None:
        super().__init__(message, error_type="timeline_refusal", code=code,
                         suggested_action=(
                             "Correct the named timeline inputs and run check again."
                             if suggested_action is None else suggested_action
                         ))


@dataclass(frozen=True)
class TimelineFinding:
    """A rule result identifying all affected elements and their writers."""

    rule: str
    severity: str
    message: str
    elements: tuple[str, ...]
    lanes: tuple[str, ...]
    suggested_action: str
    bare_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable finding and its concrete correction."""
        result = asdict(self)
        result.pop("bare_message")
        result["elements"] = list(self.elements)
        result["lanes"] = list(self.lanes)
        return result


@dataclass(frozen=True)
class ResolvedElement:
    """An element on the 30 fps output clock, with its resolved kind payload."""

    id: str
    kind: str
    lane: str
    start_frame: int
    end_frame: int
    layer: str
    payload: dict[str, Any]

    @property
    def start_s(self) -> float:
        """Return output seconds from the exact start frame."""
        return float(self.start_frame / FPS)

    @property
    def end_s(self) -> float:
        """Return output seconds from the exact end frame."""
        return float(self.end_frame / FPS)

    def to_dict(self) -> dict[str, Any]:
        """Flatten the kind payload into the contract consumed by later stages."""
        return {**copy.deepcopy(self.payload), "id": self.id, "kind": self.kind,
                "lane": self.lane, "start_frame": self.start_frame, "end_frame": self.end_frame,
                "start_s": self.start_s, "end_s": self.end_s, "layer": self.layer}


@dataclass(frozen=True)
class ResolvedTimeline:
    """Merged lanes, exact output bounds, findings, and separate word and beat snaps."""

    elements: tuple[ResolvedElement, ...]
    findings: tuple[TimelineFinding, ...]
    snapped_anchors: tuple[dict[str, Any], ...]
    total_frames: int
    end_hold_frames: int
    beat_snaps: tuple[dict[str, Any], ...] = ()
    steps: dict[str, Any] | None = None
    camera_3d: bool = False

    @property
    def valid(self) -> bool:
        """Whether every timeline rule passed; warnings remain acceptable."""
        return not any(item.severity == "violation" for item in self.findings)

    def require_valid(self) -> ResolvedTimeline:
        """Refuse invalid timelines before a downstream stage consumes them."""
        if not self.valid:
            raise TimelineRefusal("; ".join(item.message for item in self.findings
                                            if item.severity == "violation"))
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the public resolved timeline and rule-check contract."""
        return {"schema": "yap-studio/resolved-timeline@1", "fps": "30/1",
                "total_frames": self.total_frames, "duration_s": float(self.total_frames / FPS),
                "end_hold_frames": self.end_hold_frames, "valid": self.valid,
                **({"steps": copy.deepcopy(self.steps)} if self.steps is not None else {}),
                **({"camera_3d": True} if self.camera_3d else {}),
                "elements": [item.to_dict() for item in self.elements],
                "findings": [item.to_dict() for item in self.findings],
                "snapped_anchors": copy.deepcopy(list(self.snapped_anchors)),
                "beat_snaps": copy.deepcopy(list(self.beat_snaps))}


def _document(context: ProjectContext, name: str, relative: str) -> Any:
    """Validate a document from the retained input snapshot."""
    return parse_document(name, context.json(relative), filename=relative)


def _context(project_dir: Path) -> ProjectContext:
    """Read project inputs with the persistent identity cache for unchanged media."""
    return ProjectContext(project_dir)


def _fresh_cut(context: ProjectContext) -> dict[str, Any]:
    """Require verified source and cut receipts, including unchanged output hashes."""
    for name in ("ingest", "cut"):
        try:
            context.read(f"{name}/receipt.json")
        except FileNotFoundError as exc:
            raise TimelineRefusal(f"The {name} is missing; run yap-studio {name} DIR.",
                                  code="stale_cut") from exc
    takes = context.json("cut/receipt.json").get("takes")
    if isinstance(takes, dict):
        project_takes = {take.id for take in context.project.takes}
        for take_id, recorded in takes.items():
            if take_id not in project_takes:
                continue
            if not isinstance(recorded, dict) or not isinstance(recorded.get("sha256"), str):
                raise TimelineRefusal(
                    f"The cut receipt has an unreadable record for take {take_id}; "
                    "run yap-studio cut DIR.", code="stale_cut",
                )
            try:
                context.check_take(take_id, recorded, since="the cut")
            except TakeChangedError as exc:
                raise TimelineRefusal(exc.message, code="stale_cut",
                                      suggested_action=exc.suggested_action) from exc
    previous: dict[str, str] = {}
    for name in ("ingest", "cut"):
        spec = next(item for item in STAGES if item.name == name)
        state = _stage_state(context, spec, previous)
        previous[name] = state["state"]
        if state["state"] != "done":
            raise TimelineRefusal(f"The {name} is {state['state']}: {state['reason']}; "
                                  f"run yap-studio {name} DIR.", code="stale_cut")
    data = context.json("cut/receipt.json")
    plan = data.get("plan")
    frames = plan.get("total_frames") if isinstance(plan, dict) else None
    if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
        raise TimelineRefusal("The cut receipt has no frame plan; run yap-studio cut DIR.",
                              code="stale_cut")
    owned = {item["path"] for item in data.get("outputs", [])}
    if "cut/cut-words.json" not in owned:
        raise TimelineRefusal("The cut receipt does not own cut/cut-words.json; re-run cut.",
                              code="stale_cut")
    return data


def _lane(context: ProjectContext, lane: str) -> TimelineLane:
    """Snapshot one lane, remembering an absent file for compare-and-swap."""
    relative = f"timeline/{lane}.json"
    try:
        data = context.json(relative)
    except FileNotFoundError:
        return TimelineLane((), {}, lane)
    return cast(TimelineLane, parse_document("timeline", data, filename=relative, lane=lane))


def _lanes(context: ProjectContext) -> dict[str, TimelineLane]:
    """Read both writers for a complete merge."""
    return {lane: _lane(context, lane) for lane in LANES}


def _style(context: ProjectContext) -> dict[str, Any]:
    """Use the shared effective style for every timeline command and render."""
    return context.render_style().data


def _fix(root: Path, elements: tuple[Element | ResolvedElement, ...]) -> str:
    """Give cross-lane findings a correction in the agent's own lane."""
    if any(item.lane == "engine" and item.id.startswith("steps-") for item in elements):
        return "Set moves.jump_cuts to hard."
    selected = next((item for item in elements if item.lane == "agent"), elements[0])
    return (f"Move {selected.id!r} or set an explicit hold_s in its fragment and run "
            f"yap-studio timeline add FRAGMENT --lane {selected.lane} {shlex.quote(str(root))}; "
            f"or run yap-studio timeline remove --lane {selected.lane} "
            f"--id {shlex.quote(selected.id)} {shlex.quote(str(root))}")


def _referenced_takes(element: Element) -> set[str]:
    """Find source takes used by element edges and nested step anchors."""
    takes = {anchor.word.split(":")[0] for anchor in (element.start, element.end)
             if anchor is not None and anchor.word is not None}

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key == "at" and isinstance(child, dict) and isinstance(child.get("word"), str):
                    takes.add(child["word"].split(":")[0])
                visit(child)

    visit(element.fields.get("slots", {}))
    visit(element.fields.get("keys", []))
    return takes


def _beat_frames(context: ProjectContext, style: dict[str, Any]) -> tuple[int, ...]:
    """Read optional beat hints only when their owning music bed is current."""
    if not style["music"]["beat_snap"]:
        return ()
    try:
        hints = context.json("music/beats.json")
        record = context.json("music/receipt.json")
        if not isinstance(hints, dict) or not isinstance(record, dict):
            return ()
        source = record.get("source")
        if (hints.get("schema") != "yap-studio/beats@1" or not isinstance(source, dict)
                or hints.get("music_sha256") != source.get("sha256")
                or record.get("mode") not in {"file", "elevenlabs"}
                or record.get("bed") != "music/bed.wav"
                or not isinstance(hints.get("beats"), list)):
            return ()
        owned = {item["path"] for item in record.get("outputs", [])}
        if not {"music/bed.wav", "music/beats.json"} <= owned:
            return ()
        spec = next(item for item in STAGES if item.name == "music")
        if _stage_state(context, spec, {"cut": "done"})["state"] != "done":
            return ()
        frames = []
        previous = -1.0
        for beat in hints["beats"]:
            time, frame, strength = beat["time_s"], beat["frame"], beat["strength"]
            if (isinstance(time, bool) or not isinstance(time, (int, float))
                    or not math.isfinite(time) or not previous < time
                    or not 0 <= time < record["cut_frames"] / float(FPS)
                    or type(frame) is not int or frame != round(time * float(FPS))
                    or isinstance(strength, bool) or not isinstance(strength, (int, float))
                    or not math.isfinite(strength) or not 0 <= strength <= 1):
                return ()
            frames.append(frame)
            previous = time
        return tuple(frames)
    except (YapError, OSError, ValueError, TypeError, KeyError, OverflowError):
        # Hints are optional. Missing, malformed or obsolete analysis cannot
        # make the speech-anchored timeline fail to resolve.
        return ()


def _zoom_faces(context: ProjectContext, *, measure: bool) -> FacesData | None:
    """Load cut faces once, measuring only when an automatic center needs them."""
    from yap_studio.faces.project import cached_cut_faces, cut_faces

    root = context.project_dir
    try:
        cut = root / "cut/cut.mp4"
        faces = (cut_faces if measure else cached_cut_faces)(
            root, cut, digest=context.identity.sha256(cut),
        )
        return faces if faces is not None and faces.status == "measured" else None
    except (YapError, OSError):
        return None


def _frame_time(frame: int) -> str:
    """Format an output frame as M:SS.s."""
    minutes, remainder = divmod(round(frame / 3), 600)
    return f"{minutes}:{remainder / 10:04.1f}"


def _zoom_needs_center(payload: Mapping[str, Any]) -> bool:
    """Check whether any framing needs an automatic center."""
    return "center" not in payload and (
        payload["move"] != "keys" or any("center" not in key for key in payload["keys"]))


def _card_videos(context: ProjectContext, item: ResolvedElement, finding: Callable) -> None:
    """Check every declared clip over its planned playback window."""
    template = item.payload["template"]
    if template not in cards.CARD_TEMPLATES:
        return
    for name, rule in cards.video_slots(template).items():
        value = item.payload["slots"].get(name, rule.get("default"))
        values = value if rule["type"] == "video_list" and isinstance(value, list) else [value]
        for index, media in enumerate(values):
            location = f"slots.{name}" + (f"[{index}]" if rule["type"] == "video_list" else "")
            if media == "self":
                if "self" not in rule.get("sources", []):
                    finding("card_video_media", f"{location}: self is not allowed; use media", item)
                continue
            if not isinstance(media, dict) or "media" not in rule.get("sources", []):
                finding("card_video_media", f"{location}: use a clip object with path and sha256",
                        item)
                continue
            speed = media.get("speed", 1)
            valid_speed = (not isinstance(speed, bool) and isinstance(speed, (int, float))
                           and math.isfinite(speed) and 0.5 <= speed <= 2)
            if not valid_speed:
                finding("card_video_speed", f"{location}: speed must be 0.5 to 2", item)
            try:
                source = paths.validate_input(context.project_dir / media.get("path", "")).resolve()
                if not source.is_relative_to(context.project_dir):
                    finding("card_video_media", f"{location}: use a file inside the project folder",
                            item)
                    continue
                if context.identity.sha256(source) != media.get("sha256"):
                    finding("media_hash", f"{location}: the clip changed; update its sha256", item)
                    continue
                info = cards.video_info(source)
                available = info.video_duration_s or info.duration_s
                in_s = media.get("in_s", 0)
                window_frames = item.end_frame - item.start_frame
                if template == "playing_grid":
                    _, window_frames = cards.grid_clip_window(index, window_frames,
                                                              item.payload["slots"])
                if (isinstance(in_s, bool) or not isinstance(in_s, (int, float))
                        or not math.isfinite(in_s) or in_s < 0
                        or (valid_speed and Fraction(str(in_s))
                            + Fraction(window_frames, 30) * Fraction(str(speed))
                            > Fraction(str(available)))):
                    finding("card_video_bounds", f"{location}: in_s + window * speed exceeds "
                            "the media; use an earlier in_s or shorter hold", item)
                if media.get("audio", "mute") != "mute" and not info.has_audio:
                    finding("card_video_audio", f"{location}: audio gain needs an audio stream; "
                            "use audio: mute", item)
                cards.video_value(name, {**media, "path": str(source), "speed": 1}, rule)
            except (YapError, OSError, TypeError, ValueError) as exc:
                finding("card_video_media", f"{location}: {exc}", item)


def _title_words_match(title: str, spoken: str) -> bool:
    """Match a contiguous spoken run with the same question and exclamation marks."""
    wanted: list[str] = []
    marks = {}
    for word in title.split():
        wanted.extend(title_words(word))
        ending = word.rstrip("\"'”’)]}")[-1:]
        if ending in {"?", "!"}:
            marks[len(wanted)] = ending
    if not wanted:
        return False
    tokens: list[str] = []
    boundaries = {0: ""}
    for word in spoken.split():
        tokens.extend(title_words(word))
        boundaries[len(tokens)] = word.rstrip("\"'”’)]}")[-1:]
    for start in boundaries:
        end = start + len(wanted)
        if (end in boundaries and tokens[start:end] == wanted
                and all(boundaries.get(start + index) == mark for index, mark in marks.items())):
            return True
    return False


def camera_findings(elements: list[ResolvedElement], moves: dict) -> list[TimelineFinding]:
    """Report camera look conflicts and disabled jump cut moves."""
    from yap_studio.render.camera3d_look import ELEMENT_FIX, element_message, incompatible

    findings = [TimelineFinding("camera_3d", "violation", element_message([item]),
                                 (item.id,), (item.lane,), ELEMENT_FIX, element_message([item]))
                for item in elements if incompatible(item)]
    if moves.get("jump_cuts", "hard") != "hard":
        message = "3D camera captions move the camera, so jump cut moves are off."
        findings.append(TimelineFinding("camera_3d", "warning", message, (), (),
                                         "Keep hard cuts with this caption look.", message))
    return findings


def _resolve(context: ProjectContext, lanes: dict[str, TimelineLane]) -> ResolvedTimeline:
    """Check a complete proposed merge from a single retained project snapshot."""
    from yap_studio.canvas import require_supported

    require_supported(context, lanes)
    data = _fresh_cut(context)
    cut_words = cast(CutWords, _document(context, "cut-words", "cut/cut-words.json"))
    edit = cast(Edit, _document(context, "edit", "edit.json"))
    sources = {take: cast(Words, _document(context, "words", f"ingest/{take}/words.json"))
               for take in cut_words.words_sha256}
    hashes = {take: hashlib.sha256(context.read(f"ingest/{take}/words.json")).hexdigest()
              for take in sources}
    if hashes != cut_words.words_sha256:
        raise TimelineRefusal("Cut words refer to changed source words; re-run cut.",
                              code="stale_cut")
    style = _style(context)
    total_frames = data["plan"]["total_frames"]
    hold_s = edit.end_hold_s if edit.end_hold_s is not None else style["internal"]["end_hold_s"]
    hold_frames = snap_frame(Fraction(str(hold_s)), FPS)
    text_hook = style["hook"]["style"] == "text_hook"
    if text_hook:
        from yap_studio.project.model import TextHook

        lanes = {lane: replace(document, elements=tuple(
            item for item in document.elements if item.kind != "title"))
            for lane, document in lanes.items()}
        try:
            choice = cast(TextHook, parse_document("text-hook",
                                                  context.json("plans/text-hook.json")))
        except FileNotFoundError:
            choice = None
        if choice is not None and choice.chosen is not None:
            agent = lanes.get("agent", TimelineLane(()))
            hook_element = Element(TEXT_HOOK_ID, "title", Anchor(time=0),
                                   end=Anchor(time=total_frames / 30),
                                   fields={"lines": [choice.chosen]})
            lanes["agent"] = replace(agent, elements=(*agent.elements, hook_element))
    beats = _beat_frames(context, style)
    beat_window = Fraction(str(style["music"]["beat_snap_window_ms"])) * FPS / 1000
    findings: list[TimelineFinding] = []
    snaps: list[dict[str, Any]] = []
    beat_snaps: list[dict[str, Any]] = []
    resolved: list[ResolvedElement] = []
    broll_media: list[tuple[ResolvedElement, Path, bool, float]] = []
    letters_images: dict[str, bool] = {}
    rules = context.project.rules
    vocabulary: tuple[str, ...] = ()
    spoken = ""
    display_spoken = ""
    if any(item.kind == "title" for lane in lanes.values() for item in lane.elements):
        from yap_studio.brand_words import list_words

        vocabulary = tuple(list_words())
        spoken = " ".join(word.text for word in cut_words.words)
        display_spoken = apply_text_rules(spoken, vocabulary)
    move_zooms = [replace(item, lane=lane) for lane, document in lanes.items()
                  for item in document.elements
                  if item.kind == "zoom" and "move" in item.fields]
    camera_3d = style["captions"].get("camera_3d", False)
    step_in = style["moves"].get("jump_cuts") == "step_in" and not camera_3d
    bottom = CANVAS_HEIGHT_PX
    bottom_margin = FACE_MARGIN_PX
    if (move_zooms or step_in) and style["captions"]["position"] == "bottom":
        bottom = _caption_band_top(style)
        bottom_margin = face_clearance_px(style) + CAPTION_SLACK_PX
    faces = _zoom_faces(
        context, measure=step_in or any(_zoom_needs_center(item.fields) for item in move_zooms),
    ) if move_zooms or step_in else None
    boxes = {frame: tuple(face.box for face in per_frame)
             for frame, per_frame in enumerate(faces.frames)} if faces is not None else {}
    joins = _transition_joins(context, cut_words, total_frames) if any(
        item.kind == "transition" for lane in lanes.values() for item in lane.elements
    ) else {}
    picture_cuts = _picture_cuts(context, total_frames) if any(
        item.kind == "overlay" and item.fields["placement"] == "tall_behind"
        for lane in lanes.values() for item in lane.elements) else ()
    openings = [item for item in move_zooms if item.fields["move"] == "opening"]
    opening_frames = ([cut.out_frame for cut in _cut_plan(context, total_frames).cuts
                       if cut.left.source_id != cut.right.source_id
                       or cut.left.src_f1 != cut.right.src_f0]
                      if openings else [])

    def finding(rule: str, message: str, *items: Element | ResolvedElement,
                severity: str = "violation", suggested_action: str | None = None) -> None:
        names = ", ".join(f"{item.id} (lane {item.lane})" for item in items)
        findings.append(TimelineFinding(rule, severity, f"{names}: {message}",
                                        tuple(item.id for item in items),
                                        tuple(cast(str, item.lane) for item in items),
                                        suggested_action or _fix(context.project_dir, items),
                                        message))

    if len(openings) > 1:
        finding("opening_count", "at most one opening push is allowed per video", *openings[1:],
                suggested_action="Remove the extra opening pushes; keep one on the video's "
                "first word.")

    for lane, document in lanes.items():
        for authored in document.elements:
            element = replace(authored, lane=lane)
            if not element.id.startswith(f"{lane}-"):
                finding("lane_owner", f"id must start with {lane}-", element)
            takes = _referenced_takes(element)
            if any((document.words_sha256 or {}).get(take) != hashes.get(take) for take in takes):
                finding("words_hash", "source words changed; re-anchor the element", element)
                continue

            def anchor_frame(anchor: Anchor, location: str, *,
                             owner: Element = element) -> int:
                point = resolve_anchor(anchor, cut_words, sources, edit)
                if point.snapped_from:
                    snaps.append({"element": owner.id, "lane": owner.lane, "anchor": location,
                                  "from_word": point.snapped_from, "word": point.word,
                                  "segment": point.segment, "frame": point.frame})
                return point.frame

            try:
                start = anchor_frame(element.start, "start")
                payload = copy.deepcopy(element.fields)
                if element.kind == "zoom" and payload.get("move") == "opening":
                    first = cut_words.words[0] if cut_words.words else None
                    if first is not None and start > snap_frame(Fraction(str(first.start)), FPS):
                        finding("opening_start", "an opening push starts on the video's first word",
                                element, suggested_action=f"Set start to word {first.id}.")
                    endpoint = opening_end(cut_words.words, opening_frames)
                    if endpoint is None:
                        finding("opening_skipped", "the first line has no sentence end, clause "
                                "end or cut between 0.8 and 4 s, so the opening push is skipped",
                                element, severity="warning", suggested_action=
                                "Leave the opening push out, or keep it off for this video.")
                        continue
                    start, end = 0, endpoint.frame
                    payload["opening"] = {"reason": endpoint.reason, "word": endpoint.word,
                                          "end_frame": end,
                                          "end_scale": end_scale(end, payload.get(
                                              "scale", OPENING_SCALE))}
                elif element.kind == "sound":
                    end = start + 1
                elif element.end is not None:
                    end = anchor_frame(element.end, "end")
                else:
                    duration = element.hold_s
                    if duration is None:
                        duration = (style["hook"]["duration_s"] if element.kind == "title"
                                    else style["chapters"]["hold_s"] if element.kind == "chapter"
                                    else style["graphics"]["hold_s"])
                        if element.kind == "title" and style["hook"]["style"] == "video_letters":
                            duration = letters.phase_frames() / FPS
                    end = start + snap_frame(Fraction(str(duration)), FPS)
                if element.kind == "overlay":
                    payload["box_px"] = list(overlay_box(payload, style))
                    if payload["placement"] == "tall_behind":
                        edges = []
                        for edge, frame in (("start", start), ("end", end)):
                            nearest = sorted(picture_cuts, key=lambda cut: (abs(cut - frame), cut))
                            distance = abs(nearest[0] - frame)
                            if distance > tall_box.CUT_SNAP_FRAMES:
                                choices = []
                                for cut in sorted(nearest[:2]):
                                    word = next((word.text for word in cut_words.words
                                                 if snap_frame(Fraction(str(word.start)), FPS)
                                                 >= cut),
                                                "end of picture")
                                    choices.append(f"{cut / 30:g} s ({word})")
                                finding("tall_box_cut", f"the {edge} is {distance} frames "
                                        f"({distance / 30:.2f} s) from the nearest picture cut",
                                        element, suggested_action=
                                        f"Nearest cuts: {', '.join(choices)}. Start on the first "
                                        "word after a cut and end on the last word before one.")
                                edges.append(frame)
                            else:
                                edges.append(nearest[0])
                        start, end = edges
                        payload["cut_frames"] = edges
                        payload["tall_framing"] = style["broll"]["tall_framing"]
                        payload["fill_color"] = style["graphics"]["paper_color"]
                if element.kind == "background":
                    payload["box_px"] = [0, 0, CANVAS_WIDTH_PX, CANVAS_HEIGHT_PX]
                    green_screen.resolve_payload(payload, style)
                if element.kind == "transition":
                    point = resolve_anchor(element.start, cut_words, sources, edit)
                    joined = joins.get((point.word, point.segment))
                    if (joined is None or element.start.word is None
                            or element.start.edge != "start" or element.start.offset_s):
                        finding("transition_join", "anchor must name the first word after a "
                                "story join, at its start with no offset", element)
                        continue
                    join, previous = joined
                    payload["frames"] = element.fields.get("frames", 12)
                    payload["join_frame"] = join
                    start = join - payload["frames"] // 2
                    end = start + payload["frames"]
                    if not previous.text.rstrip().endswith((".", "!", "?")):
                        finding("transition_sentence", "the last word before the join must "
                                "end a sentence", element)
                    allowed = style["moves"]["transitions"]
                    if payload["look"] not in allowed:
                        message = ("the style has transitions off" if not allowed else
                                   f"the style does not allow {payload['look']}; "
                                   f"it allows {', '.join(allowed)}")
                        finding("transition_style", message, element, severity="warning",
                                suggested_action="Use a transition look the style allows, or keep "
                                "this one only when a review comment asked for it.")
                step_frames: list[tuple[str, int]] = []

                def slots(value: Any, location: str, *,
                          steps: list[tuple[str, int]] = step_frames) -> Any:
                    if isinstance(value, list):
                        return [slots(child, f"{location}[{index}]")
                                for index, child in enumerate(value)]
                    if isinstance(value, dict):
                        result = {}
                        for key, child in value.items():
                            here = f"{location}.{key}"
                            if key == "at":
                                frame = anchor_frame(parse_anchor(child), here)
                                result[key] = {"frame": frame, "time_s": float(frame / FPS)}
                                steps.append((here, frame))
                            else:
                                result[key] = slots(child, here)
                        return result
                    return value

                key_frames = []
                if element.kind == "zoom":
                    key_frames = [anchor_frame(parse_anchor(key["at"]), f"keys[{index}].at")
                                  for index, key in enumerate(payload.get("keys", []))]
                if "slots" in payload:
                    payload["slots"] = slots(payload["slots"], "slots")
                if (beats and element.kind in {"graphic", "insert"}
                        and not (element.kind == "graphic" and step_frames)
                        and 0 <= start < end <= total_frames):
                    candidates = sorted(
                        (frame for frame in beats if abs(frame - start) <= beat_window),
                        key=lambda frame: (abs(frame - start), frame),
                    )
                    for candidate in candidates:
                        moved_end = end + candidate - start
                        if (not 0 <= candidate < moved_end <= total_frames
                                or any(not candidate <= frame < moved_end
                                       for _, frame in step_frames)):
                            continue
                        if candidate != start:
                            beat_snaps.append({"element": element.id, "lane": lane,
                                               "from_frame": start, "to_frame": candidate,
                                               "from_end_frame": end, "to_end_frame": moved_end})
                        start, end = candidate, moved_end
                        break
                for location, frame in step_frames:
                    if not start <= frame < end or not 0 <= frame < total_frames:
                        finding("step_bounds", f"{location} is outside its element or cut", element)
                if element.kind == "graphic" and isinstance(payload["slots"].get("cues"), list):
                    payload["slots"]["cues"] = [
                        round(float((cue["at"]["frame"] - start) / FPS), 6)
                        if isinstance(cue, dict) and set(cue) == {"at"} else cue
                        for cue in payload["slots"]["cues"]
                    ]
            except (ProjectFileError, TypeError) as exc:
                finding("anchor", str(exc), element)
                continue
            layer = payload.pop("layer", "behind" if element.kind == "title" else "front")
            item = ResolvedElement(element.id, element.kind, lane, start, end, layer, payload)
            if text_hook and item.kind == "title":
                end = total_frames + hold_frames
                item = replace(item, start_frame=0, end_frame=end, layer="front")
            resolved.append(item)
            seconds = Fraction(end - start, 30)
            if item.kind == "title":
                title = " ".join(payload["lines"])
                if text_hook:
                    from yap_studio.text_hook import unspoken_numbers

                    if unspoken_numbers(title, spoken):
                        finding("number_not_spoken", "Text hook uses a number or amount "
                                "the creator did not say. Use a number or amount from the take.",
                                item)
                elif not (_title_words_match(title, spoken) or _title_words_match(
                    apply_text_rules(title, vocabulary), display_spoken,
                )):
                    finding("title_words", "title is not a contiguous run of the cut's words "
                            "with its spoken punctuation", item, severity="warning",
                            suggested_action="Use the creator's words, shortened only at word "
                            "boundaries. This wording can stay only when the creator asked for it.")
            if end <= start:
                finding("duration", "end must be after start on the frame grid", item)
            if start < 0 or end > total_frames + (hold_frames if text_hook
                                                 and item.kind == "title" else 0):
                finding("inside_video", "element is outside the cut or enters the end hold", item)
            if item.kind == "transition" and (start <= 0 or end >= total_frames):
                finding("transition_bounds", "transition needs at least one content frame "
                        "before its start and after its end", item)
            if (hold_frames and start < total_frames + hold_frames and end > total_frames
                    and not (text_hook and item.kind == "title")):
                finding("end_hold", "elements cannot enter the frozen end hold", item)
            if item.kind == "zoom":
                curve = None
                mode = style["moves"]["zoom_mode"]
                if mode == "none" and style["moves"].get("jump_cuts") in {"punch", "smooth"}:
                    mode = "punches"
                move = payload.get("move")
                message = ("the style has zooms off" if mode == "none" else
                           "the style allows punches only" if mode == "punches" and move != "punch"
                           else "the style allows punches and a slow push only"
                           if mode == "punches_and_push" and move not in {"punch", "push"}
                           else None)
                if move == "opening":
                    message = (None if style["moves"].get("opening_push", False)
                               else "the style has the opening push off")
                if message is not None:
                    finding("zoom_style", message, item, severity="warning", suggested_action=
                            "Use zooms the style allows, or keep this one only when a review "
                            "comment asked for it.")
                if "move" not in payload:
                    if not rules["zoom_min_s"] <= seconds <= rules["zoom_max_s"]:
                        finding("zoom_length", "zoom length is outside the allowed range", item)
                    limit = min(rules["zoom_peak_max"], SCALE_LIMIT)
                    if not 1 < payload["peak_scale"] <= limit:
                        finding("zoom_peak", "zoom peak scale must be greater than 1 "
                                f"and at most {limit:g}", item)
                    if end > start:
                        curve = ZoomCurve.from_payload(start, end, payload)
                else:
                    move = payload["move"]
                    if move == "punch":
                        payload.setdefault("scale", style["moves"]["punch_scale"])
                    elif move == "opening":
                        payload.setdefault("scale", OPENING_SCALE)
                    maximum = max(key["scale"] for key in payload["keys"]) if move == "keys" else (
                        payload["scale"])
                    center = payload.get("center")
                    payload["center_from"] = "set"
                    if _zoom_needs_center(payload):
                        aimed = face_center(boxes, range(start, end), maximum, bottom=bottom,
                                            bottom_margin=bottom_margin)
                        if aimed is None:
                            center = (.5, .42)
                            payload["center_from"] = "default"
                            finding("zoom_center", "faces were not measured and the move aims "
                                    "at the default point; set center to aim it", item,
                                    severity="warning")
                        else:
                            center, _ = aimed
                            payload["center_from"] = "face"
                    elif center is None:
                        center = payload["keys"][0]["center"]
                    center = tuple(center)
                    payload["center"] = list(center)
                    keys = ()
                    if move == "keys":
                        if (any(not start <= frame < end for frame in key_frames)
                                or any(a >= b for a, b in zip(
                                    key_frames, key_frames[1:], strict=False))):
                            finding("zoom_keys", "key frames must be inside the zoom window "
                                    "and strictly increasing", item)
                        else:
                            authored_keys = tuple(ZoomKey(
                                frame, key["scale"], tuple(key.get("center", center)),
                                key.get("ease", "snap"))
                                for frame, key in zip(key_frames, payload["keys"], strict=True))
                            keys = (ZoomKey(start, 1, authored_keys[0].center), *authored_keys)
                    elif move == "opening":
                        keys = push_keys(0, end, payload["opening"]["end_scale"], center, "linear")
                    elif move == "push":
                        ramp = (snap_frame(Fraction(str(payload["push_s"])), FPS)
                                if "push_s" in payload else end - start)
                        if (ramp <= 0 or ramp > end - start
                                or payload.get("push_s", 0) > float(seconds)):
                            finding("zoom_keys", "push_s must fit inside the zoom window", item)
                        else:
                            keys = push_keys(start, end, maximum, center,
                                             payload.get("ease", "linear"), ramp)
                    else:
                        ease = payload.get("ease", "snap")
                        move_frames = 0 if ease == "cut" else SNAP_IN_FRAMES + SNAP_OUT_FRAMES
                        if end - start < move_frames + rules["zoom_hold_min_s"] * 30 - 1e-9:
                            finding("zoom_length", "punch needs at least "
                                    f"{rules['zoom_hold_min_s']:g} s of hold"
                                    + (" plus both snap moves" if move_frames else ""), item)
                        keys = punch_keys(start, end, maximum, center, ease)
                    curve = ZoomCurve(start, end, keys)
                    payload["curve"] = [asdict(key) for key in curve.keys]
                    for key in payload["curve"]:
                        key["center"] = list(key["center"])
                    limit = min(rules["zoom_scale_max"], SCALE_LIMIT)
                    if not 1 < maximum <= limit:
                        finding("zoom_peak", "zoom scale must be greater than 1 "
                                f"and at most {limit:g}", item)
                    if faces is not None:
                        outside = face_leaves_frame(curve, boxes, bottom=bottom)
                        if outside is not None:
                            frame, reason = outside
                            fix = ("lower the scale or set center" if move == "opening" else
                                   "use a 1.5x punch here or set center")
                            message = (f"the chin reaches the captions at {_frame_time(frame)}; "
                                       f"{fix}" if reason == "captions"
                                       else f"the face leaves the frame at {_frame_time(frame)}; "
                                       "lower the scale or set center")
                            finding("zoom_face", message, item)
                if curve is not None:
                    flaws = scale_motion(curve.scales(start - 1, end + 1), start - 1,
                                         declared_moves(curve))
                    if flaws:
                        flaw = flaws[0]
                        message = (f"the zoom jumps {abs(math.exp(flaw.change) - 1):.0%} at "
                                   f"{_frame_time(flaw.frame)} where no cut or snap is declared"
                                   if flaw.kind == "jump" else
                                   f"the zoom stutters at {_frame_time(flaw.frame)} inside a move")
                        finding("zoom_motion", message, item, suggested_action=
                                "Build the move from punch, push or keys with cut or snap eases.")
                    for motion in curve.moves():
                        change = abs(motion.to_scale - motion.from_scale)
                        duration = (motion.end_frame - motion.start_frame) / 30
                        message = None
                        if change + 1e-9 < rules["zoom_step_min"]:
                            message = (f"changes the framing by {change:.0%}; under "
                                       f"{rules['zoom_step_min']:.0%} reads as camera drift")
                        elif duration and change / duration + 1e-9 < rules["zoom_rate_min"]:
                            message = (f"moves {change:.0%} over {duration:.1f} s; slower than "
                                       f"{rules['zoom_rate_min']:g} per second reads as drift")
                        if message is not None:
                            finding("zoom_subtle", f"the zoom at "
                                    f"{_frame_time(motion.start_frame)} {message}", item,
                                    severity="warning", suggested_action=
                                    "Use a punch or a push, or remove the zoom.")
                            break
            if item.kind == "chapter" and not 1 <= seconds <= 4:
                finding("chapter_hold", "chapter hold must be between 1.0 and 4.0 seconds", item)
            if item.kind == "graphic":
                _card_videos(context, item, finding)
                if payload["template"] == "big_number" and (fix := spoken_number_fix(
                    payload["slots"], cut_words.words, start, end,
                )):
                    finding("number_not_spoken", fix, item, suggested_action=fix)
                if seconds > rules["graphic_hold_violation_s"]:
                    finding("graphic_hold", "graphic exceeds the maximum hold", item)
                elif (seconds < rules["graphic_hold_min_s"]
                      or seconds > rules["graphic_hold_max_s"]
                      and not cards.rows_on_cue(style, payload["template"], payload["slots"])):
                    finding("graphic_hold", "graphic hold is outside the preferred range", item,
                            severity="warning")
            if item.kind == "overlay":
                x, y, width, height = payload["box_px"]
                if (x < 0 or y < 0 or width <= 0 or height <= 0
                        or x + width > CANVAS_WIDTH_PX or y + height > CANVAS_HEIGHT_PX):
                    finding("overlay_box", "overlay box must lie inside the 1080x1920 canvas", item)
                if payload["placement"] == "tall_behind":
                    if not tall_box.TALL_MIN_S <= seconds <= tall_box.TALL_MAX_S:
                        finding("tall_box_length", "tall box length must be 2.5 to 8 s", item,
                                suggested_action="Keep the tall box on screen for 2.5 to 8 s.")
                    if (payload["tall_framing"] != "shrink"
                            and _caption_band_top(style) < y + height + caption_gap_px(style)):
                        finding("tall_box_captions", "captions would sit on the tall box", item,
                                suggested_action="Use Bottom captions, or move captions lower.")
            if item.kind in {"insert", "overlay", "background"}:
                group = item.kind == "insert" and bool(payload.get("group"))
                prefix = ("montage_part" if group else "overlay"
                          if item.kind == "background" else item.kind)
                if (payload.get("placement") != "tall_behind"
                        and not rules[f"{prefix}_min_s"] <= seconds <= rules[f"{prefix}_max_s"]):
                    finding(f"{item.kind}_length",
                            f"{item.kind} length is outside the preferred range", item,
                            severity="warning")
                media = payload["media"]
                try:
                    source = paths.validate_input(context.project_dir / media["path"])
                    if context.identity.sha256(source) != media["sha256"]:
                        finding("media_hash", f"{item.kind} media hash changed", item)
                        continue
                    if not source.is_relative_to(context.project_dir):
                        finding("external_media",
                                f"{item.kind} media is outside the project folder",
                                item, severity="warning")
                    info = ffmpeg.probe(source, count_frames=False)
                    available = info.video_duration_s
                    image = _is_image(info, source)
                    letters_images[item.id] = image
                    if item.kind == "overlay":
                        _, _, width, height = payload["box_px"]
                        if payload["placement"] == "tall_behind":
                            payload["page_source_px"] = [info.width, info.height]
                        page_h = (tall_box.page_height(info.width, info.height, width)
                                  if info.width and info.height else 0)
                        if payload["placement"] == "tall_behind" and (
                                not image or page_h < height):
                            finding("tall_box_media", "tall box needs a long still image", item,
                                    suggested_action="Use a long page, or Above your head "
                                    "for this shot.")
                        if payload.get("motion") == "scroll":
                            flicks = tall_box.scroll_plan(end - start, page_h, height)
                            if not image or page_h <= height or not flicks:
                                finding("overlay_scroll", "scroll needs a still page taller "
                                        "than its box and at least 63 frames", item,
                                        suggested_action="Use a long page for at least 2.1 s, "
                                        "or set motion to none.")
                            else:
                                payload["scroll"] = {"page_px": [width, page_h],
                                                     "flicks": [list(row) for row in flicks]}
                    if item.kind == "background":
                        for rule, message, fix in green_screen.findings(payload, end - start,
                                                                        image=image):
                            finding(rule, message, item, suggested_action=fix)
                    if not info.has_video or (available is None and not image):
                        finding(f"{item.kind}_media",
                                f"{item.kind} needs picture with a known duration for clips", item)
                        continue
                    if (item.kind in {"insert", "overlay"} and "color_match" not in payload
                            and style["broll"]["color_match"] == "on"):
                        payload["color_match"] = True
                    if item.kind == "insert":
                        motion = payload.get("motion")
                        direction = style["broll"]["still_push"]
                        if motion is None and image:
                            motion = "still" if direction == "none" else "push"
                        if motion is not None:
                            payload["motion"] = motion
                        if motion == "push":
                            payload["push_scale"] = payload.get(
                                "push_scale", style["broll"]["still_push_scale"])
                            payload["push_direction"] = "out" if direction == "out" else "in"
                        elif "push_scale" in payload:
                            finding("insert_motion", "push_scale needs motion: push", item)
                    in_s = media.get("in_s", 0)
                    speed = media.get("speed", 1)
                    if speed == "fit":
                        speed = (1 if image else
                                 (Fraction(str(available)) - Fraction(str(in_s))) / seconds
                                 if seconds > 0 else 0)
                    exact_speed = Fraction(str(speed))
                    media["speed"] = float(exact_speed)
                    if item.kind == "insert" and 2 < speed <= TIMELAPSE_SPEED_MAX:
                        if payload.get("audio", "mute") != "mute":
                            finding("insert_speed", "mute insert audio for a timelapse above 2.0x",
                                    item)
                    elif not 0.5 <= speed <= 2:
                        limit = TIMELAPSE_SPEED_MAX if item.kind == "insert" else 2
                        finding(f"{item.kind}_speed",
                                f"{item.kind} speed must be 0.5 to {limit:.1f}", item)
                    consumed = Fraction(str(in_s)) + seconds * exact_speed
                    if not image and (consumed > Fraction(str(available)) or in_s >= available):
                        finding(f"{item.kind}_bounds",
                                "in_s + window * speed exceeds the media", item)
                    if media.get("hdr", "auto") == "off" and info.color_transfer in {
                        "arib-std-b67", "smpte2084",
                    }:
                        finding(f"{item.kind}_hdr", "HDR picture requires media.hdr: auto", item)
                    if lane == "broll":
                        middle = float(Fraction(str(in_s)) + seconds * exact_speed / 2)
                        if available is not None:
                            middle = min(max(0.0, middle), available)
                        broll_media.append((item, source, image, middle))
                except (YapError, OSError) as exc:
                    finding(f"{item.kind}_media", str(exc), item)
    steps = None
    framing = []
    if step_in:
        pieces = data["plan"].get("pieces")
        if not pieces:
            message = "The cut receipt has no pieces for Step in."
            findings.append(TimelineFinding(
                "step_in_cut", "violation", message, (), (),
                "Run yap-studio cut DIR again.", message,
            ))
        else:
            covers = [(item.start_frame, item.end_frame, item.id, item.kind)
                      for item in resolved if item.kind in {
                          "zoom", "insert", "overlay", "background", "transition", "moment",
                          "grade_compare",
                      } or (item.kind == "graphic" and (
                          item.layer == "behind" or item.payload.get("template") == "big_number"))]
            front_cards = [(item.start_frame, item.end_frame, item.id,
                            Box(*(item.payload["box"][key]
                                  for key in ("x", "y", "width", "height"))))
                           for item in resolved if item.kind == "graphic" and item.layer == "front"
                           and "box" in item.payload]
            planned = plan_steps(pieces, cut_words.words, total_frames, covers,
                                 boxes if faces is not None else None,
                                 front_cards, bottom=bottom, bottom_margin=bottom_margin,
                                 card_margin=face_clearance_px(style),
                                 wide_from=[(item.end_frame,
                                             "The opening push cuts back wide here.")
                                            for item in resolved if item.kind == "zoom"
                                            and item.payload.get("move") == "opening"])
            steps = planned.to_dict()
            for shot in planned.shots:
                if not shot.level:
                    continue
                key = {"frame": shot.start, "scale": shot.scale,
                       "center": list(shot.center), "ease": "cut"}
                if framing and framing[-1].end_frame == shot.start:
                    run = framing[-1]
                    run.payload["curve"].append(key)
                    framing[-1] = replace(run, end_frame=shot.end)
                else:
                    framing.append(ResolvedElement(
                        f"steps-{len(framing) + 1}", "zoom", "engine", shot.start, shot.end,
                        "front", {"move": "steps", "curve": [key], "center_from": "face"}))
    # Framing is planned before overlap checks. Only authored elements take part
    # in their rules and caps; covers already keep engine runs clear of moves.
    _broll_facts(broll_media, resolved, finding)
    for item in resolved:
        if item.kind != "graphic" or item.payload.get("template") != "reel_feed":
            continue
        slots = item.payload.get("slots", {})
        if slots.get("ending", "handoff") != "handoff":
            continue
        if slots.get("post", "self") != "self":
            finding("card_video_handoff", "handoff continues this video; "
                    "use post self or ending fade", item)
        for other in [*resolved, *framing]:
            changes_picture = (other.kind in {"zoom", "insert", "transition", "background",
                                             "overlay", "moment", "grade_compare"}
                               or (other.kind == "title"
                                   and style["hook"]["style"] == "video_letters"))
            if (changes_picture
                    and other.start_frame <= item.end_frame - 1 < other.end_frame):
                change = "letters title" if other.kind == "title" else other.kind.replace("_", " ")
                finding("card_video_handoff", "handoff needs the cut picture on its last frame; "
                        f"move the feed off the {change}, or use ending fade",
                        item, other,
                        suggested_action=f"Move the feed off the {change}, or use ending fade.")
    for item in resolved:
        if item.kind != "overlay" or item.payload.get("placement") != "tall_behind":
            continue
        for other in resolved:
            if (other.id != item.id
                    and not (text_hook and other.kind == "title" and other.id == TEXT_HOOK_ID)
                    and max(item.start_frame, other.start_frame)
                    < min(item.end_frame, other.end_frame)
                    and other.kind in {"zoom", "insert", "background", "transition", "moment",
                                       "grade_compare", "overlay", "graphic", "title"}):
                finding("tall_box_overlap", "tall box cannot overlap a reframed picture, "
                        "card or title, or another overlay", item, other,
                        suggested_action="Move the other picture, card or title outside the "
                                         "tall box window.")
    check_split_voiceover(context, resolved, finding)
    broll_limits(resolved, total_frames, cut_words, context.project.takes, finding,
                 insert_length_ids={item for entry in findings if entry.rule == "insert_length"
                                    for item in entry.elements})
    findings.extend(letters.check_timeline(resolved, style, cut_words, letters_images))
    zooms = tuple(item for item in resolved if item.kind == "zoom")
    compares = [item for item in resolved if item.kind == "grade_compare"]
    if compares:
        from yap_studio.grade import EYEDROPPER_DEFAULT_STRENGTH

        selection = context.project.grade if context.project.grade is not None else style["grade"]
        default_strength = (EYEDROPPER_DEFAULT_STRENGTH
                            if selection.get("look") == "eyedropper" else 1)
        active = (selection.get("look", "none") != "none"
                  and selection.get("strength", default_strength) > 0)
        for index, item in enumerate(compares):
            if not 60 <= item.end_frame - item.start_frame <= 240:
                finding("grade_compare_length", "grade compare length must be 2 to 8 seconds", item)
            if not active:
                finding("grade_compare_grade", "grade compare needs an active grade", item)
            for other in resolved:
                if (other.kind in {"overlay", "background", "transition"}
                        or other in compares[:index]):
                    if item.start_frame < other.end_frame and other.start_frame < item.end_frame:
                        finding("grade_compare_overlap", "grade compare cannot overlap overlays, "
                                "backgrounds, transitions or another grade compare", other, item)
        if len(compares) > 2:
            finding("grade_compare_count", "at most two grade compares are allowed per video",
                    *compares)
    if len(zooms) > rules["zoom_max_per_video"]:
        finding("zoom_count", "too many zooms in the video", *zooms)
    for index, item in enumerate(zooms):
        for other in zooms[:index]:
            if ("opening" in {item.payload.get("move"), other.payload.get("move")}
                    and item.start_frame < other.end_frame and other.start_frame < item.end_frame):
                finding("opening_overlap", "an opening push cannot overlap another zoom",
                        other, item)
    transitions = [item for item in resolved if item.kind == "transition"]
    for index, item in enumerate(transitions):
        for other in transitions[:index]:
            if item.payload["join_frame"] == other.payload["join_frame"]:
                finding("transition_duplicate", "only one transition is allowed per join",
                        other, item)
        for other in resolved:
            if (other.id == item.id
                    or (text_hook and other.kind == "title" and other.id == TEXT_HOOK_ID)
                    or other.kind not in {"title", "chapter", "insert", "overlay", "background",
                                          "zoom", "transition"}
                    or (other.kind == "transition" and other not in transitions[:index])):
                continue
            if item.start_frame < other.end_frame and other.start_frame < item.end_frame:
                finding("transition_overlap", "transition windows cannot overlap titles, chapters, "
                        "inserts, overlays, backgrounds, zooms or other transitions", other, item)
    visuals = sorted((item for item in resolved if item.kind in VISUAL_KINDS
                      and not (text_hook and item.kind == "title" and item.id == TEXT_HOOK_ID)),
                     key=lambda item: (item.start_frame, item.id))
    for index, item in enumerate(visuals):
        active = [other for other in visuals[:index] if other.end_frame > item.start_frame]
        chapters = [other for other in active if other.kind == "chapter"]
        if item.kind == "chapter" and chapters:
            finding("chapter_overlap", "chapter cards cannot overlap", *chapters, item)

        def visual_unit(element: ResolvedElement) -> tuple[str, str]:
            group = element.payload.get("group") if element.kind == "insert" else None
            return element.lane, f"group:{group}" if group else f"id:{element.id}"

        def opening_push(element: ResolvedElement) -> bool:
            return element.kind == "zoom" and element.payload.get("move") == "opening"

        overlapping = [*active, item]
        has_insert = any(other.kind == "insert" for other in overlapping)
        units = {visual_unit(other) for other in overlapping
                 if not opening_push(other) and not (has_insert and other.kind == "chapter")}
        exclusive = any(
            visual_unit(other) != visual_unit(item)
            and {item.kind, other.kind} != {"chapter", "insert"}
            and not ((opening_push(item) and other.kind == "title")
                     or (opening_push(other) and item.kind == "title"))
            and ("insert" in {item.kind, other.kind}
                 or opening_push(item) or opening_push(other)
                 or ("zoom" in {item.kind, other.kind} and item.kind != other.kind))
            for other in active
        )
        if len(units) > rules["elements_at_once_max"] or exclusive:
            finding("visual_overlap", "too many visual elements overlap", *active, item)
    # The person matte no longer lines up on zoom and insert frames, so a background
    # there would cover the speaker. Assembly refuses this even with an override.
    for item in resolved:
        if item.kind != "background":
            continue
        for other in resolved:
            if (other.kind in {"zoom", "insert"} and item.start_frame < other.end_frame
                    and other.start_frame < item.end_frame):
                finding("background_overlap",
                        "backgrounds cannot overlap zooms or inserts", other, item)
    if all(take.voiceover for take in context.project.takes):
        elements = [item.to_dict() for item in resolved]
        gaps = uncovered_frames(total_frames, elements)
        if gaps:
            message = "Voiceover picture uncovered at frames " + frame_ranges_text(gaps)
            findings.append(TimelineFinding(
                "voiceover_uncovered", "warning", message, (), (),
                "Plan B-roll inserts over these frames and add them with yap-studio timeline "
                "add FRAGMENT DIR --lane broll; the last insert ends at the cut end.", message,
            ))
        graphic_only = []
        # Remove empty frames from gaps in inserts to isolate graphic coverage.
        for first, last in uncovered_frames(total_frames, elements,
                                            kinds=frozenset({"insert"})):
            cursor = first
            for gap_first, gap_last in gaps:
                if gap_last < cursor:
                    continue
                if gap_first > last:
                    break
                if gap_first > cursor:
                    graphic_only.append((cursor, gap_first - 1))
                cursor = gap_last + 1
            if cursor <= last:
                graphic_only.append((cursor, last))
        if graphic_only:
            message = ("Voiceover picture at frames " + frame_ranges_text(graphic_only)
                       + " comes only from graphics; assembly counts a frame only where a "
                       "graphic is opaque and fills the frame.")
            findings.append(TimelineFinding(
                "voiceover_graphic_only", "warning", message, (), (),
                "Cover these frames with B-roll inserts, or make sure the graphic fills the "
                "whole frame.", message,
            ))
    keyed_screen = (chroma.keyed_screen(context, style) if any(
        item.kind == "moment" and item.payload["style"] == "clone_flash" for item in resolved
    ) else None)
    check_moments(resolved, style, finding, keyed_screen=keyed_screen)
    if camera_3d:
        findings.extend(camera_findings(resolved, style["moves"]))
    return ResolvedTimeline(tuple(sorted([*resolved, *framing],
                                        key=lambda item: (item.start_frame, item.id))),
                            tuple(findings), tuple(snaps), total_frames, hold_frames,
                            beat_snaps=tuple(beat_snaps), steps=steps, camera_3d=camera_3d)


def _broll_facts(
    media: list[tuple[ResolvedElement, Path, bool, float]],
    resolved: list[ResolvedElement],
    finding: Callable[..., None],
) -> None:
    """Warn about backwards capture dates, repeated sources, and close pictures."""
    ordered = sorted(media, key=lambda entry: (entry[0].start_frame, entry[0].id))
    latest = None
    latest_item = None
    for item, source, _, _ in ordered:
        if item.payload.get("chronology"):
            continue
        captured = capture_time(source, sha256=item.payload["media"]["sha256"])
        if captured is None:
            continue
        if latest is not None and captured < latest - CHRONOLOGY_SLACK:
            finding("broll_order", f"captured {captured:%Y-%m-%d %H:%M}, before "
                    f"{latest_item.id} captured {latest:%Y-%m-%d %H:%M}; "
                    "B-roll goes back in time", item, severity="warning",
                    suggested_action=("Reorder the shots, or mark this element with "
                                      '"chronology": "hook" or "before_after" if deliberate.'))
        if latest is None or captured > latest:
            latest, latest_item = captured, item

    by_hash: dict[str, list[ResolvedElement]] = {}
    for item in sorted((item for item in resolved if item.lane == "broll" and
                        item.kind in {"insert", "overlay", "background"}),
                       key=lambda entry: (entry.start_frame, entry.id)):
        by_hash.setdefault(item.payload["media"]["sha256"], []).append(item)
    for group in by_hash.values():
        if len(group) > 2:
            finding("broll_reuse", f"the same source is used {len(group)} times and reads "
                    "as the same moment repeating", *group, severity="warning",
                    suggested_action="Use this source at most twice.")

    candidates = ordered
    if len({item.payload["media"]["sha256"] for item, _, _, _ in candidates}) < 2:
        return
    thumbs = []
    for item, source, still, middle in candidates:
        media = item.payload["media"]
        options = {"crop": tuple(media["crop"])} if (item.kind == "insert" and
                  media.get("fit") == "crop") else {}
        thumbs.append(fingerprint(source, sha256=media["sha256"],
                                  time_s=0 if still else middle, **options))
    parent = list(range(len(candidates)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, (item, _, _, _) in enumerate(candidates):
        left = thumbs[index]
        if left is None:
            continue
        for other in range(index):
            right = thumbs[other]
            if (right is None or item.payload["media"]["sha256"] ==
                    candidates[other][0].payload["media"]["sha256"]):
                continue
            difference = sum(abs(a - b) for a, b in zip(left, right, strict=True)) / 192
            if difference <= NEAR_DUPLICATE_MAX:
                parent[root(index)] = root(other)
    groups: dict[int, list[ResolvedElement]] = {}
    for index, (item, _, _, _) in enumerate(candidates):
        groups.setdefault(root(index), []).append(item)
    for group in groups.values():
        if len(group) > 1:
            finding("broll_near_duplicate", "these shots look nearly the same", *group,
                    severity="warning",
                    suggested_action="Keep one, or swap the others for different shots.")


def _picture_cuts(context: ProjectContext, total_frames: int) -> tuple[int, ...]:
    """Read picture jumps and hold boundaries from receipt-owned cut spans."""
    plan = _cut_plan(context, total_frames)
    cuts = {right.out_f0 for left, right in zip(plan.segments, plan.segments[1:], strict=False)
            if left.source_id != right.source_id or left.src_f1 != right.src_f0}
    cuts.update(frame for hold in plan.holds for frame in (hold.out_f0, hold.out_f1))
    return tuple(sorted({0, total_frames, *cuts}))


def _cut_plan(context: ProjectContext, total_frames: int) -> Plan:
    """Read exact output cut boundaries from receipt-owned spans."""
    relative = "cut/spans.json"
    source = context.project_dir / relative
    record = context.json("cut/receipt.json")
    owned = {item["path"] for item in record.get("outputs", [])}
    if relative not in owned:
        raise TimelineRefusal("The cut receipt does not own cut/spans.json; re-run cut.",
                              code="stale_cut")
    try:
        data = context.read(relative)
        reader = next(item for item in READERS if item.sniff(source, data))
        plan = Plan.build(reader.read(source, data), fps=FPS, sample_rate=48_000)
    except (OSError, YapError, StopIteration) as exc:
        raise TimelineRefusal("Cut joins need readable verified cut spans; re-run cut.",
                              code="stale_cut") from exc
    if plan.total_frames != total_frames:
        raise TimelineRefusal("Cut spans disagree with the cut frame count; re-run cut.",
                              code="stale_cut")
    return plan


def _transition_joins(context: ProjectContext, words: CutWords, total_frames: int
                      ) -> dict[tuple[str | None, str | None], tuple[int, CutWord]]:
    """Bind incoming word occurrences to exact story boundaries in the verified spans."""
    plan = _cut_plan(context, total_frames)
    record = context.json("cut/receipt.json")
    story_words: dict[str, list[CutWord]] = {}
    for word in words.words:
        story_words.setdefault(word.segment, []).append(word)
    # Receipt pieces follow span order, including retained wordless pauses.
    piece_ids = {index: piece["segment_id"]
                 for index, piece in enumerate(record["plan"].get("pieces", []), 1)
                 if piece.get("segment_id")}
    # Older receipts without piece ids retain the midpoint-based fallback.
    placed = [(word, (Fraction(str(word.start)) + Fraction(str(word.end))) / 2)
              for word in words.words]
    segment_ids = {}
    for segment in plan.segments:
        if segment.edl_index in piece_ids:
            segment_ids[segment.index] = piece_ids[segment.edl_index]
            continue
        end = Fraction(segment.out_f0 + segment.n_frames, 30)
        occurrence = next((word for word, midpoint in placed
                           if segment.out_time <= midpoint < end), None)
        if occurrence is not None:
            segment_ids[segment.index] = occurrence.segment
    result = {}
    for cut in plan.cuts:
        outgoing_id = segment_ids.get(cut.left.index)
        incoming_id = segment_ids.get(cut.right.index)
        if (outgoing_id is not None and incoming_id is not None and outgoing_id != incoming_id
                and outgoing_id in story_words and incoming_id in story_words):
            # Boundary words come from recorded story order, not their timestamps.
            previous = story_words[outgoing_id][-1]
            incoming = story_words[incoming_id][0]
            result[(incoming.id, incoming.segment)] = (cut.out_frame, previous)
    return result


def resolve_timeline(project_dir: Path) -> ResolvedTimeline:
    """Merge current lanes against a verified fresh cut without rendering or writing lanes.

    Consumers must call require_valid() before using elements. Nested graphic step
    anchors become {frame, time_s}; insert speed: fit becomes a numeric speed.
    """
    with paths.operation():
        context = _context(project_dir)
        return _resolve(context, _lanes(context))


def _incoming(path: Path, *, lane: str, words: CutWords) -> TimelineLane:
    """Read a fragment, or a planned lane whose source words still match the cut.

    `cards plan` writes a lane file with the words hash map it was planned against.
    """
    if not path.is_file():
        return load_timeline_fragment(path, lane=lane)
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeError) as exc:
        raise ProjectFileError(path.name, "$", f"cannot read JSON: {exc}") from exc
    if not isinstance(data, dict) or "words_sha256" not in data:
        return cast(TimelineLane, parse_document("timeline-fragment", data,
                                                 filename=path.name, lane=lane))
    planned = cast(TimelineLane, parse_document("timeline", data, filename=path.name, lane=lane))
    stale = sorted({take for element in planned.elements for take in _referenced_takes(element)
                    if (planned.words_sha256 or {}).get(take) != words.words_sha256.get(take)})
    if stale:
        raise TimelineRefusal(
            f"{path.name} was planned against older words for {', '.join(stale)}.",
            suggested_action="Plan it again from the current cut/cut-words.json, then add it.",
        )
    fragment = {key: value for key, value in data.items() if key != "words_sha256"}
    return cast(TimelineLane, parse_document("timeline-fragment", fragment,
                                             filename=path.name, lane=lane))


def add_timeline(project_dir: Path, fragment: Path, *, lane: str,
                 replace_lane: bool = False) -> ResolvedTimeline:
    """Check a proposed lane merge or replacement before writing it."""
    if lane not in LANES:
        raise TimelineRefusal(f"Unknown lane {lane!r}.")
    with paths.operation():
        context = _context(project_dir)
        words = cast(CutWords, _document(context, "cut-words", "cut/cut-words.json"))
        incoming = _incoming(paths.validate_input(fragment), lane=lane, words=words)
        from yap_studio.stock.guard import check_timeline

        style = _style(context)
        check_timeline(context.project_dir, incoming.elements, style)
        from yap_studio.pictures import guard as pictures_guard

        pictures_guard.check_timeline(context.project_dir, incoming.elements, style)
        lanes = _lanes(context)
        previous = lanes[lane]
        replacements = {element.id: replace(element, lane=lane) for element in incoming.elements}
        elements = () if replace_lane else tuple(
            replacements.pop(element.id, element) for element in previous.elements
        )
        for old in (() if replace_lane else previous.elements):
            if old.id not in {item.id for item in incoming.elements} and any(
                (previous.words_sha256 or {}).get(take) != words.words_sha256.get(take)
                for take in _referenced_takes(old)
            ):
                raise TimelineRefusal(f"{old.id}: source words changed; replace or remove this "
                                      "element before adding to its lane.")
        lanes[lane] = TimelineLane(elements + tuple(replacements.values()),
                                   words.words_sha256, lane)
        result = _resolve(context, lanes)
        if not result.valid:
            return result
        taste_pin = context.taste_application().pin()
        relative = f"timeline/{lane}.json"
        base = context._snapshots[relative]
        with project_lock(context.project_dir):
            # Checking every small input prevents a concurrent lane or cut edit from
            # invalidating the merge while this writer was resolving media and rules.
            for name, snapshot in context._snapshots.items():
                target = context.project_dir / name
                current = target.read_bytes() if target.exists() else None
                if snapshot != current:
                    raise ConflictError(target)
            # Hash exactly under the write lock so a cache hit cannot weaken the final check.
            for item in result.elements:
                if item.kind in {"insert", "overlay", "background"}:
                    media = item.payload["media"]
                    target = context.project_dir / media["path"]
                    if receipt.sha256_file(target) != media["sha256"]:
                        raise ConflictError(target)
            # Approved taste is private and sits outside the project file snapshots.
            current_taste = load_taste().snapshot()
            if (current_taste.revision, current_taste.sha256) != (
                taste_pin["revision"], taste_pin["sha256"],
            ):
                raise ConflictError(taste_path())
            pictures_guard.check_timeline(context.project_dir, incoming.elements, style)
            write_json_cas(context.project_dir / relative, lanes[lane].to_dict(),
                           base_sha256=hashlib.sha256(base).hexdigest() if base else None)
        return result


def remove_timeline(project_dir: Path, *, lane: str, ids: tuple[str, ...] = ()) -> dict[str, Any]:
    """Remove requested ids, or clear one lane, even while the cut needs rebuilding."""
    if lane not in LANES or any(not item.startswith(f"{lane}-") for item in ids):
        raise TimelineRefusal("A lane can remove only ids with its own prefix.")
    with paths.operation():
        context = _context(project_dir)
        document = _lane(context, lane)
        kept = tuple(item for item in document.elements if ids and item.id not in ids)
        removed = [item.id for item in document.elements if item not in kept]
        if removed:
            relative = f"timeline/{lane}.json"
            base = context.read(relative)
            with project_lock(context.project_dir):
                write_json_cas(context.project_dir / relative,
                               replace(document, elements=kept).to_dict(),
                               base_sha256=hashlib.sha256(base).hexdigest())
        return {"lane": lane, "removed": removed}
