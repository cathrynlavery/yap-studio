"""Catalog suggestions and explicit word-anchored B-roll request plans."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from yap_studio import letters, paths, receipt
from yap_studio.errors import InputError, OutputPathError, ValidationError
from yap_studio.media.select import (
    INSERT_MAX_S,
    INSERT_MIN_S,
    REQUEST_SCHEMA,
    avoided_tags,
    parse_request,
    rank_candidates,
)
from yap_studio.planner.anchors import frame_at, resolve_anchor
from yap_studio.planner.broll_placement import requested_placement, split_media
from yap_studio.planner.broll_timing import lead_named_start
from yap_studio.planner.cards import PENDING_SCHEMA, _json, _publish
from yap_studio.project.model import (
    CutWord,
    CutWords,
    ProjectFileError,
    parse_anchor,
    parse_document,
)
from yap_studio.stages.green_screen import WIPE_FRAMES
from yap_studio.stages.tall_box import (
    TALL_MAX_S,
    TALL_MIN_S,
    is_long_page,
    page_height,
    scroll_plan,
)
from yap_studio.style.loader import _schema

SUGGESTIONS_SCHEMA = "yap-studio/broll-suggestions@1"
REQUESTS_SCHEMA = "yap-studio/broll-requests@1"
PLAN_SCHEMA = "yap-studio/broll-plan@1"
_REQUEST_FIELDS = {
    "id",
    "item",
    "kind",
    "start",
    "end",
    "hold_s",
    "in_s",
    "speed",
    "fit",
    "crop",
    "placement",
    "box",
    "motion",
    "audio",
    "matches",
    "chronology",
    "layout",
    "letters",
    "color_match",
}
_NAMED_BOXES = ("headroom", "side_left", "side_right", "tall_behind")
_INTERNAL_DEFAULTS = {key: rule["default"] for key, rule in
                      _schema()["properties"]["internal"]["properties"].items()}


def _suggested_box(item: Mapping, style: Mapping, duration: Fraction) -> str:
    box = style.get("overlay_box", "headroom")
    if box == "tall_behind" and (not is_long_page(item)
                                 or not TALL_MIN_S <= duration <= TALL_MAX_S):
        return "headroom"
    return box


def _suggested_motion(item: Mapping, payload: Mapping, style_data: Mapping, frames: int) -> str:
    motion = style_data.get("broll", {}).get("overlay_motion", "none")
    if motion == "scroll":
        from yap_studio.stages.timeline import overlay_box

        geometry_style = {"internal": {**_INTERNAL_DEFAULTS,
                                       **style_data.get("internal", {})}}
        _, _, width, height = overlay_box(payload, geometry_style)
        return "scroll" if (item.get("kind") == "image" and item.get("width")
                            and item.get("height") and scroll_plan(
                                frames, page_height(item["width"], item["height"], width),
                                height)) else "none"
    return motion


def _object(value: Any, location: str) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValidationError(location, "expected an object with string keys")
    return value


def _unknown(data: dict, fields: set[str], location: str) -> None:
    for key in data:
        if key not in fields:
            raise ValidationError(f"{location}.{key}", "unknown field")


def _positive_int(value: int, field: str) -> None:
    if type(value) is not int or value < 1:
        raise ValidationError(field, "expected a positive integer")


def _output_words(words: CutWords) -> None:
    if words.time_base != "output":
        raise ValidationError("cut_words.time_base", "expected output time")


def _anchor(word: CutWord, edge: str) -> dict:
    return {"word": word.id, "text": word.text, "segment": word.segment, "edge": edge}


def suggest_broll(
    cut_words: CutWords,
    catalog_items: Sequence[Mapping[str, Any]],
    style_data: Mapping[str, Any],
    *,
    phrase_words: int = 8,
    limit: int = 3,
) -> dict:
    """Rank supporting media for phrases without selecting or writing a lane."""
    _positive_int(phrase_words, "phrase_words")
    _positive_int(limit, "limit")
    if limit > 50:
        raise ValidationError("limit", "expected an int from 1 to 50")
    _output_words(cut_words)
    style = style_data.get("broll", {})
    phrases, phrase = [], []
    for word in cut_words.words:
        phrase.append(word)
        if len(phrase) == phrase_words or re.search(r"[.!?][\"')\]]*$", word.text):
            phrases.append(phrase)
            phrase = []
    if phrase:
        phrases.append(phrase)
    suggestions = []
    excluded: list[dict] = []
    catalog = {item["id"]: item for item in catalog_items}
    locations: dict[str, tuple[Path | None, str]] = {}
    for phrase_index, phrase in enumerate(phrases):
        text = " ".join(word.text for word in phrase)
        length = min(INSERT_MAX_S, max(INSERT_MIN_S, phrase[-1].end - phrase[0].start))
        request = parse_request(
            {
                "schema": REQUEST_SCHEMA,
                "words": text,
                "length_s": length,
                "limit": limit,
            }
        )
        dropped: list[dict] = []
        window = Fraction(frame_at(phrase[-1].end, 30) - frame_at(phrase[0].start, 30), 30)
        if window <= 0:
            continue
        request = replace(request, length_s=float(window), exact_length_s=window)
        ranked = rank_candidates(catalog_items, request, style_broll=style, dropped=dropped)
        for entry in dropped:
            if entry not in excluded:
                excluded.append(entry)
        candidates, skipped = [], []
        for candidate in ranked:
            if len(candidates) == limit:
                break
            item_id = candidate["item"]
            if item_id not in locations:
                locations[item_id] = _matching_location(catalog[item_id])
            source, reason = locations[item_id]
            if source is None:
                skipped.append({"item": candidate["item"], "reasons": [reason]})
                continue
            overlay = candidate["placement"] == "overlay"
            result = {
                "item": candidate["item"],
                "media_type": candidate["media_type"],
                "kind": "overlay" if overlay else "insert",
                "placement": candidate["placement"],
                "box": _suggested_box(catalog[item_id], style, window) if overlay else None,
                "fit": "cover" if overlay else candidate["fit"],
                "motion": None,
                "score": candidate["score"],
                "matched": candidate["matched"],
                "reasons": candidate["reasons"],
            }
            if overlay:
                result["motion"] = _suggested_motion(
                    catalog[item_id], {"placement": result["box"]}, style_data, int(window * 30))
            if candidate["placement"] == "green_screen":
                result.update(kind="background", layout=style.get("green_screen_layout", "full"))
            if not candidate["still"]:
                result.update(in_s=candidate["in_s"], out_s=candidate["out_s"])
            choice = {
                "id": f"broll-phrase-{phrase_index + 1}",
                "item": result["item"],
                "kind": result["kind"],
                "start": _anchor(phrase[0], "start"),
                "end": _anchor(phrase[-1], "end"),
                "placement": result["placement"],
                "fit": result["fit"],
            }
            if overlay:
                choice.update(box=result["box"], motion=result["motion"])
            if result["kind"] == "background":
                choice["layout"] = result["layout"]
            if not candidate["still"]:
                choice.update(in_s=candidate["in_s"], speed=1)
            result["request"] = choice
            candidates.append(result)
        if candidates or skipped:
            suggestions.append(
                {
                    "text": text,
                    "word_ids": [word.id for word in phrase],
                    "start": _anchor(phrase[0], "start"),
                    "end": _anchor(phrase[-1], "end"),
                    "candidates": candidates,
                    "skipped": skipped,
                }
            )
    return {"schema": SUGGESTIONS_SCHEMA, "phrases": suggestions, "excluded": excluded}


@dataclass(frozen=True)
class BrollPlan:
    """An agent's requested elements and their resolved output frames."""

    status: str
    broll: tuple[dict, ...]
    warnings: tuple[str, ...]
    frames: int
    fps: int
    words_sha256: dict[str, str]

    def lane(self) -> dict:
        """Return a fragment accepted by timeline add with lane broll."""
        return {
            "schema": "yap-studio/timeline@1",
            "elements": [copy.deepcopy(entry["element"]) for entry in self.broll],
        }

    def to_dict(self) -> dict:
        """Return the review summary, retaining timing and catalog evidence."""
        return {
            "schema": PLAN_SCHEMA,
            "status": self.status,
            "frames": self.frames,
            "fps": self.fps,
            "words_sha256": dict(self.words_sha256),
            "warnings": list(self.warnings),
            "broll": copy.deepcopy(list(self.broll)),
        }


def load_requests(path: str | Path) -> dict:
    """Read and reserve the agent's requests against output writes."""
    source = paths.validate_input(path)
    try:
        return json.loads(source.read_bytes())
    except OSError as exc:
        raise InputError(str(path), f"cannot read requests JSON: {exc}") from exc
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValidationError(str(path), f"invalid requests JSON: {exc}") from exc


def _matching_location(item: Mapping[str, Any]) -> tuple[Path | None, str]:
    changed = False
    for entry in item["locations"]:
        candidate = Path(entry["path"]).expanduser()
        try:
            if candidate.is_file():
                if receipt.sha256_file(candidate) == item["sha256"]:
                    return candidate.resolve(), ""
                changed = True
        except OSError:
            continue
    return None, "media file changed" if changed else "media file missing or unreadable"


def _media_path(item: Mapping[str, Any], location: str, project_dir: Path) -> str:
    source, reason = _matching_location(item)
    if source is None:
        raise ValidationError(
            f"{location}.item",
            f"{item['id']}: {reason}; run yap-studio media add again to refresh the catalog",
        )
    source = paths.validate_input(source).resolve()
    return (source.relative_to(project_dir).as_posix()
            if source.is_relative_to(project_dir) else str(source))


def _element(request: dict, item: Mapping[str, Any], style_data: Mapping, location: str,
             project_dir: Path) -> dict:
    style = style_data.get("broll", {})
    placement = requested_placement(request, item, style)
    if not isinstance(placement, str):
        raise ValidationError(f"{location}.placement", "expected a placement name")
    kind = request.get("kind", {"full_frame": "insert", "split": "insert",
                                "green_screen": "background"}.get(placement, "overlay"))
    if not isinstance(kind, str) or kind not in {"insert", "overlay", "background"}:
        raise ValidationError(f"{location}.kind", "expected insert, overlay or background")
    if "layout" in request and kind != "background":
        raise ValidationError(f"{location}.layout", "only backgrounds accept layout")
    if "letters" in request and kind != "insert":
        raise ValidationError(f"{location}.letters", f"Use kind: insert for letters; remove it "
                              f"from this {kind}.")
    media = {
        "path": _media_path(item, location, project_dir),
        "sha256": item["sha256"],
        "in_s": request.get("in_s", 0),
        "speed": request.get("speed", 1),
    }
    element = {
        "id": request["id"],
        "kind": kind,
        "lane": "broll",
        "start": copy.deepcopy(request.get("start")),
        "media": media,
    }
    for field in ("end", "hold_s", "matches", "chronology"):
        if field in request:
            element[field] = copy.deepcopy(request[field])
    if "letters" in request:
        element["letters"] = copy.deepcopy(request["letters"])
        if item["kind"] == "image" and "end" not in request and "hold_s" not in request:
            element["hold_s"] = letters.phase_frames(request["letters"]) / letters.FPS
    if "color_match" in request:
        if not isinstance(request["color_match"], bool):
            raise ValidationError(f"{location}.color_match", "expected a boolean")
        element["color_match"] = request["color_match"]
    elif item["media_type"] in {"screenshot", "screen_recording"}:
        element["color_match"] = False
    provenance = item["provenance"]
    if provenance.get("kind") is not None:
        element["provenance"] = copy.deepcopy(provenance)
    if kind == "background":
        for field in ("crop", "audio", "box", "color_match"):
            if field in request:
                raise ValidationError(f"{location}.{field}", "backgrounds do not accept this field")
        # Colour match shifts inserts and overlays; a background blends the creator in instead.
        element.pop("color_match", None)
        if request.get("placement", "green_screen") != "green_screen":
            raise ValidationError(f"{location}.placement", "backgrounds use green_screen")
        if request.get("fit", "cover") != "cover":
            raise ValidationError(f"{location}.fit", "planned backgrounds use cover")
        motion = request.get("motion", "push"
                             if item["media_type"] in {"photo", "screenshot"}
                             and style.get("still_push", "in") != "none" else "none")
        if motion not in ("none", "push"):
            raise ValidationError(f"{location}.motion", "expected none or push")
        if motion == "push" and item["kind"] != "image":
            raise ValidationError(f"{location}.motion", "push needs a photo or screenshot")
        element.update(fit="cover", layout=request.get("layout", style.get("green_screen_layout",
                                                                         "full")),
                       blend="wrap", enter="wipe", motion=motion)
        return element
    if kind == "insert":
        for field in ("box", "motion"):
            if field in request:
                raise ValidationError(f"{location}.{field}", "only overlays accept this field")
        if placement not in ("full_frame", "split"):
            raise ValidationError(f"{location}.placement", "inserts use full_frame or split")
        crop = request.get("crop")
        if split_media(request, placement, media, location):
            pass
        elif "crop" in request:
            if "fit" in request and request["fit"] != "crop":
                raise ValidationError(f"{location}.fit", "fit must be crop when crop is set")
            width, height = item["width"], item["height"]
            if (not isinstance(crop, list) or len(crop) != 4
                    or any(type(value) is not int for value in crop)
                    or min(crop[:2]) < 0 or min(crop[2:]) <= 0
                    or crop[0] + crop[2] > width or crop[1] + crop[3] > height):
                raise ValidationError(f"{location}.crop",
                                      f"expected a rectangle inside the {width}x{height} item")
            media["fit"] = "crop"
            media["crop"] = crop.copy()
        elif request.get("fit") == "crop":
            raise ValidationError(f"{location}.crop", "fit crop requires a crop rectangle")
        else:
            media["fit"] = request.get(
                "fit", "cover" if "letters" in request or item["orientation"] == "vertical"
                else "contain"
            )
        element["audio"] = copy.deepcopy(request.get("audio", "mute"))
    else:
        if "crop" in request:
            raise ValidationError(f"{location}.crop", "only inserts accept this field")
        if "audio" in request:
            raise ValidationError(f"{location}.audio", "only inserts accept audio")
        box = request.get("box")
        placement = request.get("placement", "overlay")
        if isinstance(box, str) and box in _NAMED_BOXES:
            if placement != "overlay" and placement != box:
                raise ValidationError(f"{location}.box", "conflicts with placement")
            placement = box
        elif "box" in request:
            if placement == "overlay":
                placement = "box"
            element["box"] = copy.deepcopy(box)
        if placement == "overlay":
            placement = style.get("overlay_box", "headroom")
        element.update(
            placement=placement,
            fit=request.get("fit", "cover"),
        )
        element["motion"] = request.get("motion", "none")
    return element


def _source_end(item: Mapping[str, Any], element: dict, length: Fraction, location: str) -> float:
    media = element["media"]
    lower = Fraction(str(media["in_s"]))
    if item["kind"] == "image":
        return float(lower)
    duration = Fraction(str(item["duration_s"]))
    upper = (
        duration
        if media["speed"] == "fit"
        else lower + length * Fraction(str(media["speed"]))
    )
    if not lower < upper <= duration:
        raise ValidationError(f"{location}.in_s", "source window must fit the clip duration")
    return float(upper)


def plan_broll(
    requests: dict,
    cut_words: CutWords,
    catalog_items: Sequence[Mapping[str, Any]],
    style_data: Mapping[str, Any],
    *,
    project_dir: Path,
    frames: int,
    fps: int = 30,
) -> BrollPlan:
    """Validate explicit catalog choices and resolve them onto the cut clock."""
    _positive_int(frames, "frames")
    _positive_int(fps, "fps")
    _output_words(cut_words)
    project_dir = Path(project_dir).expanduser().resolve()
    obj = _object(requests, "requests")
    _unknown(obj, {"schema", "broll"}, "requests")
    if obj.get("schema") != REQUESTS_SCHEMA:
        raise ValidationError("requests.schema", f"expected {REQUESTS_SCHEMA}")
    if not isinstance(obj.get("broll"), list):
        raise ValidationError("broll", "expected a list")
    catalog = {item["id"]: item for item in catalog_items}
    style = style_data.get("broll", {})
    planned: list[dict] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(obj["broll"]):
        location = f"broll[{index}]"
        request = _object(raw, location)
        _unknown(request, _REQUEST_FIELDS, location)
        identity = request.get("id")
        if (
            not isinstance(identity, str)
            or not identity.startswith("broll-")
            or not identity[6:].strip()
        ):
            raise ValidationError(f"{location}.id", "expected an id with the broll- prefix")
        if identity in seen:
            raise ValidationError(f"{location}.id", f"duplicate id {identity!r}")
        seen.add(identity)
        item_id = request.get("item")
        if not isinstance(item_id, str) or item_id not in catalog:
            raise ValidationError(f"{location}.item", f"unknown catalog item {item_id!r}")
        item = catalog[item_id]
        if "letters" in request:
            first = frame_at(resolve_anchor(parse_anchor(request.get("start")), cut_words,
                             start=True, location=f"{location}.start"), fps)
            if not letters.allowed(style_data, first):
                raise ValidationError(f"{location}.letters", letters.permission_message(first))
        element = _element(request, item, style_data, location, project_dir)
        try:
            lane = parse_document(
                "timeline",
                {
                    "schema": "yap-studio/timeline@1",
                    "words_sha256": {},
                    "elements": [element],
                },
                lane="broll",
            )
        except ProjectFileError as exc:
            field = exc.location.replace("elements[0]", location, 1)
            raise ValidationError(field, exc.detail) from exc
        parsed = lane.elements[0]
        start_s = resolve_anchor(parsed.start, cut_words, start=True, location=f"{location}.start")
        end_s = (
            resolve_anchor(parsed.end, cut_words, start=False, location=f"{location}.end")
            if parsed.end is not None
            else start_s + parsed.hold_s
        )
        if not all(math.isfinite(value) for value in (start_s, end_s)):
            raise ValidationError(location, "B-roll times must be finite")
        media = element["media"]
        max_frames = None
        if item["kind"] != "image" and media["speed"] != "fit":
            available = Fraction(str(item["duration_s"])) - Fraction(str(media["in_s"]))
            max_frames = math.floor(available * fps / Fraction(str(media["speed"])))
        lead_named_start(element, cut_words, planned, end_s=end_s, max_frames=max_frames, fps=fps)
        start_s = resolve_anchor(parse_anchor(element["start"]), cut_words,
                                 start=True, location=f"{location}.start")
        if parsed.end is None:
            end_s = start_s + parsed.hold_s
        first, last = frame_at(start_s, fps), frame_at(end_s, fps)
        if not 0 <= first < last <= frames or end_s <= start_s:
            raise ValidationError(
                location,
                f"window [{first}, {last}) must be inside [0, {frames})",
            )
        if element["kind"] == "overlay":
            if request.get("placement", "overlay") == "overlay" and "box" not in request:
                element["placement"] = _suggested_box(item, style, Fraction(last - first, fps))
            if "motion" not in request:
                element["motion"] = _suggested_motion(item, element, style_data, last - first)
        if element["kind"] == "background" and last - first < 2 * WIPE_FRAMES:
            element["enter"] = "cut"
        for previous in planned:
            if max(first, previous["start_frame"]) < min(last, previous["end_frame"]):
                raise ValidationError(location, f"{identity!r} and {previous['id']!r} overlap")
        out_s = _source_end(item, element, Fraction(last - first, fps), location)
        for decision in item["decisions"]:
            if decision["verdict"] != "rejected" or decision["revoked_at"] is not None:
                continue
            if decision["start_s"] is None or (
                element["media"]["in_s"] < decision["end_s"] and decision["start_s"] < out_s
            ):
                raise ValidationError(
                    f"{location}.item",
                    f"{item_id}: source overlaps rejected decision {decision['id']}",
                )
        avoided = avoided_tags(item, style.get("avoid_tags", []))
        if avoided:
            warnings.append(f"{identity}: {item_id} carries avoided tags: " + ", ".join(avoided))
        planned.append(
            {
                "id": identity,
                "item": item_id,
                "media_type": item["media_type"],
                "sha256": item["sha256"],
                "provenance": copy.deepcopy(item["provenance"]),
                "element": element,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
                "start_frame": first,
                "end_frame": last,
                "in_s": element["media"]["in_s"],
                "out_s": out_s,
            }
        )
    return BrollPlan(
        "needs_review" if warnings else "ok",
        tuple(planned),
        tuple(warnings),
        frames,
        fps,
        dict(cut_words.words_sha256),
    )


@contextlib.contextmanager
def _ownership(out: Path) -> Iterator[tuple[Path, ...]]:
    manifest, pending = out / "broll-plan.json", out / "receipt.pending.json"
    paths.validate_output(pending)
    if pending.exists():
        try:
            journal = json.loads(pending.read_bytes())
            if (
                not isinstance(journal, dict)
                or journal.get("schema") != PENDING_SCHEMA
                or not isinstance(journal.get("outputs"), list)
            ):
                raise ValueError("not a publication journal")
        except (OSError, ValueError, UnicodeError, RecursionError) as exc:
            raise OutputPathError("Invalid pending plan receipt.", code="unsafe_path") from exc
    if manifest.is_symlink():
        raise OutputPathError("Plan receipt must not be a symlink.", code="unsafe_path")
    if not manifest.exists():
        yield (manifest, pending)
        return
    try:
        data = json.loads(manifest.read_bytes())
        if not isinstance(data, dict) or data.get("schema") != PLAN_SCHEMA:
            raise ValueError("not a B-roll plan")
        records = data["outputs"]
        if (
            not isinstance(records, list)
            or not all(
                isinstance(entry, dict)
                and isinstance(entry.get("path"), str)
                and isinstance(entry.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                for entry in records
            )
            or "broll-lane.json" not in {entry["path"] for entry in records}
        ):
            raise ValueError("missing output hashes")
    except (OSError, ValueError, KeyError, UnicodeError, RecursionError) as exc:
        raise OutputPathError(
            "Existing plan has no valid ownership receipt; choose a new directory.",
            code="unsafe_path",
        ) from exc
    descriptor, name = tempfile.mkstemp(prefix=".yap-owner-", suffix=".json", dir=out)
    os.close(descriptor)
    snapshot = Path(name)
    try:
        snapshot.write_text(
            _json(
                {
                    "schema": PLAN_SCHEMA,
                    "outputs": [
                        *[entry for entry in records if entry["path"] == "broll-lane.json"],
                        {"path": manifest.name, "sha256": receipt.sha256_file(manifest)},
                    ],
                }
            ),
            encoding="utf-8",
        )
        yield (snapshot, pending)
    finally:
        snapshot.unlink(missing_ok=True)


def write_broll_plan(plan: BrollPlan, out_dir: str | Path) -> dict[str, Path]:
    """Publish the fragment and summary with hashed ownership and a recovery journal."""
    fragment = plan.lane()
    parse_document("timeline-fragment", fragment, lane="broll")
    content = _json(fragment)
    data = {
        **plan.to_dict(),
        "outputs": [
            {
                "path": "broll-lane.json",
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        ],
    }
    try:
        out = paths.validate_output_dir(out_dir)
        with _ownership(out) as owners:
            _publish(
                out,
                {out / "broll-lane.json": content, out / "broll-plan.json": _json(data)},
                owners,
            )
    except OSError as exc:
        raise OutputPathError(f"Cannot write B-roll plan: {exc}") from exc
    return {name: out / name for name in ("broll-lane.json", "broll-plan.json")}
