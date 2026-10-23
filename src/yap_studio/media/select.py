"""Pure, deterministic narrowing of catalog media to candidate source ranges."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from yap_studio.errors import ValidationError

REQUEST_SCHEMA = "yap-studio/media-request@1"
# B-roll insert and montage part length bounds the timeline accepts.
INSERT_MIN_S = 1.0
INSERT_MAX_S = 4.0
MONTAGE_PART_MIN_S = 0.7
MONTAGE_PART_MAX_S = 1.3

_FIELDS = {
    "schema", "words", "length_s", "placement", "orientation", "montage", "kinds",
    "match", "exclude", "limit", "anchor", "avoid_tags",
}
_REQUIRED = {"schema", "words", "length_s"}
_ITEM_ID = re.compile(r"m[1-9][0-9]*\Z")
_WORDS = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
_STOPWORDS = frozenset(
    "a an the and or but of to in on at for with from by is are was were be it its "
    "this that these those i you we they my your our me so just really very then".split()
)
_MILLISECOND = Fraction(1, 1000)
_Interval = tuple[Fraction, Fraction]


@dataclass(frozen=True)
class SelectionRequest:
    """A validated word request; anchors are opaque JSON passed through to candidates."""

    schema: str
    words: str
    length_s: float
    placement: str | None = None
    orientation: str = "any"
    montage: bool = False
    kinds: tuple[str, ...] = ("video", "image")
    match: str = "words"
    exclude: tuple[str, ...] = ()
    limit: int = 5
    anchor: dict[str, Any] | None = None
    avoid_tags: tuple[str, ...] = ()
    # Internal callers can preserve frame timing without changing JSON request fields.
    exact_length_s: Fraction | None = None


def _choice(value: object, field: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValidationError(field, "choose " + ", ".join(choices))
    return value


def _json_value(value: object, field: str) -> None:
    """Validate JSON values without serializing or interpreting an anchor."""
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for index, entry in enumerate(value):
            _json_value(entry, f"{field}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, entry in value.items():
            _json_value(entry, f"{field}.{key}")
        return
    raise ValidationError(field, "must contain only JSON values")


def parse_request(data: object) -> SelectionRequest:
    """Validate a request strictly, naming the field responsible for any error."""
    if not isinstance(data, dict):
        raise ValidationError("request", "must be an object")
    for key in data:
        if key not in _FIELDS:
            raise ValidationError(str(key), "unknown field")
    for key in sorted(_REQUIRED - data.keys()):
        raise ValidationError(key, "required field is missing")
    if data["schema"] != REQUEST_SCHEMA:
        raise ValidationError("schema", f"must be {REQUEST_SCHEMA!r}")
    words = data["words"]
    if not isinstance(words, str) or not 1 <= len(words.strip()) <= 500:
        raise ValidationError("words", "must contain 1 to 500 characters after trimming")
    length = data["length_s"]
    try:
        finite = type(length) in {int, float} and math.isfinite(length)
    except OverflowError:
        finite = False
    if not finite:
        raise ValidationError("length_s", "must be a finite number")
    montage = data.get("montage", False)
    if type(montage) is not bool:
        raise ValidationError("montage", "must be a bool")
    low, high = (MONTAGE_PART_MIN_S, MONTAGE_PART_MAX_S) if montage else (
        INSERT_MIN_S, INSERT_MAX_S
    )
    if not low <= length <= high:
        raise ValidationError(
            "length_s", "an insert lasts 1.0 to 4.0 s; montage parts 0.7 to 1.3 s; "
            "choose anchor words that fit",
        )
    placement = (
        _choice(data["placement"], "placement", ("full_frame", "overlay", "split", "green_screen"))
        if "placement" in data else None
    )
    orientation = _choice(
        data.get("orientation", "any"), "orientation", ("vertical", "horizontal", "square", "any")
    )
    match = _choice(data.get("match", "words"), "match", ("words", "any"))
    kinds = data.get("kinds", ["video", "image"])
    if not isinstance(kinds, list) or not kinds:
        raise ValidationError("kinds", "must be a non-empty list of video or image")
    kinds = tuple(dict.fromkeys(
        _choice(kind, f"kinds[{i}]", ("video", "image")) for i, kind in enumerate(kinds)
    ))
    exclude = data.get("exclude", [])
    if not isinstance(exclude, list):
        raise ValidationError("exclude", "must be a list of item ids")
    for i, item_id in enumerate(exclude):
        if not isinstance(item_id, str) or _ITEM_ID.fullmatch(item_id) is None:
            raise ValidationError(f"exclude[{i}]", "must be an item id such as m3")
    limit = data.get("limit", 5)
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValidationError("limit", "must be an int from 1 to 50")
    anchor = data.get("anchor")
    avoid = data.get("avoid_tags", [])
    if not isinstance(avoid, list):
        raise ValidationError("avoid_tags", "must be a list of tags")
    avoid_tags = _tags(avoid, "avoid_tags")
    if "anchor" in data:
        if not isinstance(anchor, dict):
            raise ValidationError("anchor", "must be a JSON object")
        try:
            _json_value(anchor, "anchor")
        except RecursionError as exc:
            raise ValidationError("anchor", "must be a finite JSON object") from exc
    return SelectionRequest(
        schema=REQUEST_SCHEMA, words=words.strip(), length_s=float(length), placement=placement,
        orientation=orientation, montage=montage, kinds=kinds, match=match,
        exclude=tuple(exclude), limit=limit, anchor=deepcopy(anchor), avoid_tags=avoid_tags,
    )


def _tags(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValidationError(field, "must be a list of tags")
    for index, tag in enumerate(value):
        if not isinstance(tag, str) or not tag.strip():
            raise ValidationError(f"{field}[{index}]", "must be a non-empty tag")
    return tuple(value)


def avoided_tags(item: Mapping[str, Any], tags: Sequence[str]) -> list[str]:
    """Return the item's avoided tags in catalog order, ignoring letter case."""
    avoided = {tag.strip().casefold() for tag in _tags(tags, "avoid_tags")}
    return [
        tag for tag in (item.get("description") or {}).get("tags", [])
        if tag.strip().casefold() in avoided
    ]


def item_placement(
    item: Mapping[str, Any], style_broll: Mapping[str, Any] | None = None,
) -> str:
    """Choose placement by media type using the style's optional B-roll section."""
    media_type = item["media_type"]
    default = "overlay" if media_type in {"photo", "screenshot"} else "full_frame"
    return _choice(
        (style_broll or {}).get(f"{media_type}_placement", default),
        f"broll.{media_type}_placement", ("full_frame", "overlay", "split", "green_screen"),
    )


def _terms(text: str) -> list[str]:
    terms = []
    for token in _WORDS.findall(text.lower()):
        if token in _STOPWORDS:
            continue
        terms.append(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return list(dict.fromkeys(terms))


def content_terms(text: str) -> list[str]:
    """Return the content terms shared by selection and word timing."""
    return _terms(text)


def _match(item: Mapping[str, Any], terms: list[str]) -> tuple[int, list[str], list[str]]:
    description = item["description"] or {}
    speech = item["speech"] or {}
    sources = (
        ("tags", set(_terms(" ".join(description.get("tags", [])))), 2),
        ("description", set(_terms(description.get("text") or "")), 1),
        ("speech", set(_terms(speech["text"] if speech.get("status") == "words" else "")), 1),
    )
    score, found, reasons = 0, set(), []
    for name, source_terms, weight in sources:
        matches = [term for term in terms if term in source_terms]
        score += weight * len(matches)
        found.update(matches)
        if matches:
            reasons.append(f"matches {name}: " + ", ".join(matches))
    return score, [term for term in terms if term in found], reasons


def _time(value: int | float) -> Fraction:
    # Decimal strings preserve catalog boundaries without binary addition drift or epsilon.
    return Fraction(str(value))


def _free_intervals(duration: Fraction, rejected: list[Mapping[str, Any]]) -> list[_Interval]:
    blocked = sorted(
        (max(Fraction(0), _time(entry["start_s"])), min(duration, _time(entry["end_s"])))
        for entry in rejected
    )
    cursor, free = Fraction(0), []
    for start, end in blocked:
        if start >= end:
            continue
        if start > cursor:
            free.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        free.append((cursor, duration))
    return free


def _window(intervals: list[_Interval], length: Fraction) -> _Interval | None:
    """Choose a millisecond start and preserve the exact length inside the clear interval."""
    for start, end in sorted(intervals):
        if start + length > end:
            continue
        lower = round(start, 3)
        if lower < start:
            lower += _MILLISECOND
        upper = lower + length
        if start <= lower < upper <= end:
            return lower, upper
    return None


def _contains(decision: Mapping[str, Any], window: _Interval) -> bool:
    return decision["start_s"] is None or (
        _time(decision["start_s"]) <= window[0]
        and window[1] <= _time(decision["end_s"])
    )


def _video_window(
    item: Mapping[str, Any], approvals: list[Mapping[str, Any]],
    rejected: list[Mapping[str, Any]], length: Fraction,
) -> tuple[_Interval, bool, list[Mapping[str, Any]]] | None:
    free = _free_intervals(_time(item["duration_s"]), rejected)
    approved = []
    for decision in approvals:
        for start, end in free:
            if decision["start_s"] is not None:
                start = max(start, _time(decision["start_s"]))
                end = min(end, _time(decision["end_s"]))
            if start < end:
                approved.append((start, end))
    window = _window(approved, length)
    if window is not None:
        return window, True, [entry for entry in approvals if _contains(entry, window)]
    window = _window(free, length)
    return (window, False, []) if window is not None else None


def rank_candidates(
    items: Sequence[Mapping[str, Any]], request: SelectionRequest,
    *, style_broll: Mapping[str, Any] | None = None, avoid_tags: Sequence[str] = (),
    dropped: list[dict] | None = None,
) -> list[dict]:
    """Rank clear windows; optionally collect reasons for avoided-tag exclusions."""
    candidates = []
    terms = _terms(request.words)
    length = (
        request.exact_length_s if request.exact_length_s is not None else _time(request.length_s)
    )
    avoid = (*request.avoid_tags, *_tags(avoid_tags, "avoid_tags"))
    if style_broll is not None:
        avoid = (*avoid, *_tags(style_broll.get("avoid_tags", []), "broll.avoid_tags"))
    for item in items:
        if item["kind"] not in request.kinds or item["id"] in request.exclude:
            continue
        if request.orientation != "any" and item["orientation"] != request.orientation:
            continue
        avoided = avoided_tags(item, avoid)
        if avoided:
            if dropped is not None:
                dropped.append({"item": item["id"], "reasons": [
                    "avoided tags: " + ", ".join(avoided),
                ]})
            continue
        placement = request.placement or item_placement(item, style_broll)
        active = [entry for entry in item["decisions"] if entry["revoked_at"] is None]
        rejected = [entry for entry in active if entry["verdict"] == "rejected"]
        if any(entry["start_s"] is None for entry in rejected):
            continue
        score, matched, match_reasons = _match(item, terms)
        if request.match == "words" and not matched:
            continue
        approvals = [entry for entry in active if entry["verdict"] == "approved"]
        still = item["kind"] == "image"
        if still:
            in_s, out_s = 0.0, request.length_s
            approving = [entry for entry in approvals if entry["start_s"] is None]
            approved = bool(approving)
        else:
            chosen = _video_window(item, approvals, rejected, length)
            if chosen is None:
                continue
            window, approved, approving = chosen
            in_s, out_s = map(float, window)
        reasons = []
        for decision in approving:
            if decision["start_s"] is None:
                reasons.append(f"whole item approved by {decision['id']}")
            else:
                reasons.append(
                    f"inside approved range {decision['id']} "
                    f"({decision['start_s']:.3f} to {decision['end_s']:.3f} s)"
                )
        if not approved:
            reasons.append("no whole-item approval" if still else (
                "no approved range fits; earliest clear stretch"
            ))
        reasons.extend(match_reasons)
        source = "style" if request.placement is None else "request"
        reasons.append(f"placement from {source}: {placement}")
        if rejected:
            reasons.append("skips rejected ranges " + ", ".join(entry["id"] for entry in rejected))
        locations = sorted(
            enumerate(item["locations"]),
            key=lambda pair: (pair[1]["seen_at"], pair[0]), reverse=True,
        )
        candidates.append({
            "item": item["id"], "sha256": item["sha256"],
            "kind": item["kind"],
            "media_type": item["media_type"], "orientation": item["orientation"],
            "width": item["width"], "height": item["height"], "placement": placement,
            "in_s": in_s, "out_s": out_s, "length_s": request.length_s, "still": still,
            "fit": "split" if placement == "split" else None if placement == "overlay" else (
                "cover" if placement == "green_screen" or item["orientation"] == "vertical"
                else "contain"
            ),
            "score": score, "matched": matched, "approved": approved,
            "approved_by": [entry["id"] for entry in approving], "reasons": reasons,
            "locations": [entry["path"] for _, entry in locations],
            "anchor": deepcopy(request.anchor),
        })
    candidates.sort(key=lambda entry: (
        not entry["approved"], -entry["score"], int(entry["item"][1:]),
    ))
    return candidates


def select_candidates(
    items: Sequence[Mapping[str, Any]], request: SelectionRequest,
    *, style_broll: Mapping[str, Any] | None = None, avoid_tags: Sequence[str] = (),
    dropped: list[dict] | None = None,
) -> list[dict]:
    """Return the highest ranked candidates up to the request's limit without I/O."""
    return rank_candidates(
        items, request, style_broll=style_broll, avoid_tags=avoid_tags, dropped=dropped,
    )[:request.limit]
