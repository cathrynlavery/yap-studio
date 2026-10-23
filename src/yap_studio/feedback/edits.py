"""Pure planning of story edits and fillers work from review operations."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from yap_studio.errors import ValidationError
from yap_studio.feedback.batch import FeedbackError
from yap_studio.project.anchors import nominal_edge
from yap_studio.project.model import CutWords, Edit, Word, Words, parse_document

OPERATIONS_SCHEMA = "yap-studio/feedback-operations@1"
# Documented seam for the timeline stage, which will own caption and card positions.
TIMELINE_KINDS = frozenset({
    "caption", "element", "title", "card", "graphic", "insert", "overlay", "zoom", "timeline",
    "move",
})
_REMOVAL_KINDS = {"filler", "silence", "stutter", "restart", "manual"}
_COMMENT = re.compile(r"v[1-9][0-9]*:[^:\s]+\Z")
_WORD = re.compile(r"t[1-9][0-9]*:(0|[1-9][0-9]*)\Z")
_ASK = "Ask the creator with yap-studio feedback ask, or recompute the operations."


class NeedsTimelineError(FeedbackError):
    """Name an operation whose positions need the future timeline stage."""

    def __init__(self, operation: str, comment: str) -> None:
        super().__init__(
            f"Operation {operation} ({comment}): caption and card positions need the "
            "timeline stage, which this build does not have yet.",
            suggested_action="Reply needs_clarification or leave the note open.",
        )
        self.code = "needs_timeline"


@dataclass(frozen=True)
class RemovalAction:
    """Work the fillers stage performs: add a manual removal, or reject an applied one."""

    action: Literal["add_manual", "reject"]
    operation: str
    comment: str
    reason: str
    take: str | None = None
    start: float | None = None
    end: float | None = None
    removal_id: str | None = None


@dataclass(frozen=True)
class EditPlan:
    """The next edit document, deferred removal work, and one change per operation."""

    edit: dict
    removal_actions: tuple[RemovalAction, ...]
    changes: tuple[dict, ...]
    broll_operations: tuple[dict, ...] = ()
    cut_spans: tuple[dict, ...] = ()


def _invalid(operation: dict, comment: str, detail: str) -> ValidationError:
    return ValidationError(f"operation {operation.get('id', '<missing>')} ({comment})", detail)


def _refused(operation: dict, comment: str, detail: str) -> FeedbackError:
    return FeedbackError(f"Operation {operation['id']} ({comment}): {detail}",
                         suggested_action=_ASK)


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _keys(value: object, expected: set[str], parameter: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValidationError(parameter,
                              f"expected exactly these keys: {', '.join(sorted(expected))}")


def validate_operations(document: object) -> dict:
    """Validate the operations contract strictly and return an independent copy."""
    _keys(document, {"schema", "batch", "items"}, "operations")
    if document["schema"] != OPERATIONS_SCHEMA:
        raise ValidationError("schema", f"must be {OPERATIONS_SCHEMA}")
    if (not isinstance(document["batch"], str)
            or re.fullmatch(r"b[1-9][0-9]*", document["batch"]) is None):
        raise ValidationError("batch", "must be b followed by a positive integer")
    if not isinstance(document["items"], list):
        raise ValidationError("items", "must be an array")
    comments: set[str] = set()
    ids: set[str] = set()
    for item in document["items"]:
        _keys(item, {"comment", "operations"}, "item")
        comment = item["comment"]
        if not isinstance(comment, str) or _COMMENT.fullmatch(comment) is None:
            raise ValidationError("comment", "must be VERSION:ID, for example v1:c1")
        if comment in comments:
            raise ValidationError("comment", f"duplicate comment {comment}")
        comments.add(comment)
        if not isinstance(item["operations"], list) or not item["operations"]:
            raise ValidationError(f"operations ({comment})", "must contain at least one operation")
        for operation in item["operations"]:
            if not isinstance(operation, dict):
                raise ValidationError(f"operation ({comment})", "must be an object")
            parameter = f"operation {operation.get('id', '<missing>')} ({comment})"
            if not _nonempty(operation.get("id")):
                raise _invalid(operation, comment, "id must be a nonempty string")
            if operation["id"] in ids:
                raise _invalid(operation, comment, "duplicate operation id")
            ids.add(operation["id"])
            kind = operation.get("kind")
            if not isinstance(kind, str):
                raise _invalid(operation, comment, "kind must be a string")
            if kind in TIMELINE_KINDS:
                raise NeedsTimelineError(operation["id"], comment)
            if kind.startswith("broll_"):
                from yap_studio.feedback.broll import validate_operation

                validate_operation(operation, comment)
                continue
            expected = {"id", "kind"}
            if kind == "remove":
                expected |= {"segment"} if "segment" in operation else {"from", "to"}
            elif kind == "trim":
                expected |= {"segment", "edge", "word"}
            elif kind == "restore":
                if "range" in operation:
                    expected.add("range")
            elif kind not in {"keep", "cut"}:
                raise _invalid(operation, comment, f"unknown kind {kind!r}")
            _keys(operation, expected, parameter)
            for key in ("segment", "range"):
                if key in operation and not _nonempty(operation[key]):
                    raise _invalid(operation, comment, f"{key} must be a nonempty string")
            if kind == "trim":
                if operation["edge"] not in ("start", "end"):
                    raise _invalid(operation, comment, "edge must be start or end")
                if (not isinstance(operation["word"], str)
                        or _WORD.fullmatch(operation["word"]) is None):
                    raise _invalid(operation, comment, "word must be a source word id like t1:12")
            if kind == "remove" and "from" in operation:
                for key in ("from", "to"):
                    anchor = operation[key]
                    _keys(anchor, {"word", "segment"}, f"{parameter}.{key}")
                    if (not _nonempty(anchor["segment"]) or not isinstance(anchor["word"], str)
                            or _WORD.fullmatch(anchor["word"]) is None):
                        raise _invalid(operation, comment, f"{key} needs a word id and segment")
                if operation["from"]["segment"] != operation["to"]["segment"]:
                    raise _invalid(operation, comment, "from and to must be in the same segment")
    return copy.deepcopy(document)


def _quote(words: Sequence[dict]) -> str:
    text = " ".join(word["text"] for word in words)
    return f'"{text[:117] + "..." if len(text) > 120 else text}"'


class _Planner:
    """Own the evolving document and occurrences without retaining mutable inputs."""

    def __init__(self, edit: dict, cut_words: CutWords,
                 source_words: Mapping[str, Words | None]) -> None:
        self.edit = copy.deepcopy(edit)
        self.kept = {segment["id"]: [
            {"id": word.id, "text": word.text} for word in cut_words.words
            if word.segment == segment["id"]
        ] for segment in self.edit["segments"]}
        self.source_words = {take: words for take, words in source_words.items()
                             if words is not None}
        self.sources = {take: sorted(words.words, key=lambda word: (word.start, word.end))
                        for take, words in self.source_words.items()}
        self.actions: list[RemovalAction] = []
        self.changes: list[dict] = []
        self.cut_spans: list[dict] = []
        self.edge_words: dict[tuple[str, str], list[dict]] = {}
        self.edge_moves: dict[tuple[str, str], list[str]] = {}
        self.edge_restores: dict[tuple[str, str], set[str]] = {}

    def segment(self, segment_id: str, operation: dict, comment: str) -> dict:
        segment = next((segment for segment in self.edit["segments"]
                        if segment["id"] == segment_id), None)
        if segment is None:
            raise _invalid(operation, comment, f"segment {segment_id} is not in the evolving edit")
        return segment

    def occurrence(self, segment_id: str, word_id: str, operation: dict, comment: str) -> int:
        index = next((index for index, word in enumerate(self.kept[segment_id])
                      if word["id"] == word_id), None)
        if index is None:
            raise _invalid(operation, comment,
                           f"word {word_id} is not a kept occurrence of {segment_id}; "
                           "a trim must keep at least one word")
        return index

    def source(self, take: str, operation: dict, comment: str) -> list[Word]:
        if take not in self.sources:
            raise _refused(operation, comment,
                           f"take {take}: its words file does not match the saved version.")
        return self.sources[take]

    def source_index(self, take: str, word_id: str, operation: dict, comment: str) -> int:
        index = next((index for index, word in enumerate(self.source(take, operation, comment))
                      if word.id == word_id), None)
        if index is None:
            raise _invalid(operation, comment, f"word {word_id} is missing from take {take}")
        return index

    def source_span(self, segment: dict, operation: dict, comment: str) -> tuple[float, float]:
        self.source(segment["take"], operation, comment)
        words = self.source_words[segment["take"]]
        try:
            edit = cast(Edit, parse_document("edit", self.edit))
            resolved = next(item for item in edit.segments if item.id == segment["id"])
            return (float(nominal_edge(resolved, resolved.from_, words, None, start=True)),
                    float(nominal_edge(resolved, resolved.to, words, None, start=False)))
        except ValidationError as exc:
            raise _invalid(operation, comment, str(exc)) from exc

    def refresh_text(self, segment_id: str) -> None:
        segment = next((segment for segment in self.edit["segments"]
                        if segment["id"] == segment_id), None)
        if segment is not None and "text" in segment:
            segment["text"] = " ".join(word["text"] for word in self.kept[segment_id])

    def drop(self, segment: dict, operation: dict, comment: str) -> tuple[str, list[dict]]:
        if len(self.edit["segments"]) == 1:
            raise _refused(operation, comment, "dropping the segment would leave no segments.")
        words = self.kept.pop(segment["id"])
        self.edit["segments"].remove(segment)
        return f"Dropped segment {segment['id']} ({len(words)} words: {_quote(words)}).", words

    def removal_span(self, segment: dict, removed: list[dict], operation: dict,
                     comment: str) -> tuple[float, float]:
        take = segment["take"]
        source = self.source(take, operation, comment)
        first = self.source_index(take, removed[0]["id"], operation, comment)
        last = self.source_index(take, removed[-1]["id"], operation, comment)
        start = ((source[first - 1].end + source[first].start) / 2
                 if first else source[first].start)
        end = ((source[last].end + source[last + 1].start) / 2
               if last + 1 < len(source) else source[last].end)
        return start, end

    def cut_span(self, segment: dict, first: int, last: int, operation: dict,
                 comment: str) -> dict:
        kept = self.kept[segment["id"]]
        removed = kept[first:last + 1]
        start, end = self.removal_span(segment, removed, operation, comment)
        edge = "start" if first == 0 else "end" if last == len(kept) - 1 else None
        if edge is not None:
            outer_start, outer_end = self.source_span(segment, operation, comment)
            if edge == "start":
                start = outer_start
            else:
                end = outer_end
        return {"operation": operation["id"], "comment": comment, "segment": segment["id"],
                "take": segment["take"], "edge": edge, "start": start, "end": end,
                "words": [word["id"] for word in removed],
                "texts": [word["text"] for word in removed], "action": None,
                "restore": {"edge": edge, "anchor": copy.deepcopy(
                    segment["from" if edge == "start" else "to"])} if edge else None}

    def cut(self, operation: dict, comment: str, item: dict) -> tuple[list[str], str, list[dict]]:
        request = item.get("request")
        if not isinstance(request, dict) or request.get("type") != "cut":
            raise _invalid(operation, comment, "cut needs the item's cut request")
        if item["comment"]["kind"] != "fix":
            raise _invalid(operation, comment, "cut requires a fix comment")
        groups: dict[str, list[dict]] = {}
        for word in request["words"]:
            groups.setdefault(word["segment"], []).append(word)
        planned = []
        for identifier, words in groups.items():
            kept = self.kept.get(identifier, [])
            indexes = []
            for word in words:
                index = next((at for at, present in enumerate(kept)
                              if present["id"] == word["id"]), None)
                if index is None:
                    raise _refused(operation, comment,
                                   f'the words to cut are no longer in the cut: "{word["text"]}"')
                if kept[index]["text"] != word["text"]:
                    raise _refused(operation, comment,
                                   f'the words changed since the note: "{word["text"]}"')
                indexes.append(index)
            first, last = indexes[0], indexes[-1]
            if indexes != list(range(first, last + 1)):
                raise _refused(operation, comment,
                               "the words to cut are not next to each other "
                               f"in segment {identifier}")
            segment = self.segment(identifier, operation, comment)
            span = None
            if first != 0 or last != len(kept) - 1:
                span = self.cut_span(segment, first, last, operation, comment)
                options = []
                alternatives = []
                if first:
                    alternatives.append((first - 1, last, f'also cut "{kept[first - 1]["text"]}"'))
                if last + 1 < len(kept):
                    alternatives.append((first, last + 1, f'also cut "{kept[last + 1]["text"]}"'))
                if first != last:
                    alternatives.extend([
                        (first + 1, last, f'keep "{kept[first]["text"]}"'),
                        (first, last - 1, f'keep "{kept[last]["text"]}"'),
                    ])
                for low, high, label in alternatives:
                    if low == 0 and high == len(kept) - 1:
                        continue
                    option = self.cut_span(segment, low, high, operation, comment)
                    if option["edge"] is not None:
                        at = high + 1 if option["edge"] == "start" else low - 1
                        start, end = self.removal_span(segment, [kept[at]], operation, comment)
                        option["end" if option["edge"] == "start" else "start"] = (
                            start if option["edge"] == "start" else end)
                    options.append({**option, "label": label})
                span["options"] = options
            planned.append((identifier, words, span))
        summaries, removed = [], []
        for identifier, words, span in planned:
            internal = {"id": operation["id"], "kind": "remove",
                        "from": {"word": words[0]["id"], "segment": identifier},
                        "to": {"word": words[-1]["id"], "segment": identifier}}
            action = len(self.actions)
            if span is not None and span["edge"] is not None:
                before = copy.deepcopy(self.kept[identifier])
                self.edge_words[operation["id"], identifier] = before
                span["kept_before"] = [word["id"] for word in before]
            _, summary, cut_words = self.remove(internal, comment)
            summaries.append(summary)
            removed.extend(cut_words)
            if span is not None:
                if span["edge"] is not None:
                    segment = self.segment(identifier, operation, comment)
                    anchor = "from" if span["edge"] == "start" else "to"
                    span["anchor_after"] = copy.deepcopy(segment[anchor])
                if len(self.actions) > action:
                    span["action"] = action
                for option in span["options"]:
                    option.update(action=span["action"], restore=copy.deepcopy(span["restore"]))
                self.cut_spans.append(span)
        summary = f'Cut {_quote(request["words"])} ({len(removed)} words): ' + " ".join(summaries)
        return list(groups), summary, removed

    def finalize_cuts(self) -> None:
        segments = {segment["id"]: segment for segment in self.edit["segments"]}
        spans = []
        for span in self.cut_spans:
            segment = segments.get(span["segment"])
            if segment is None:
                continue
            operation, comment = {"id": span["operation"]}, span["comment"]
            kept = self.kept[span["segment"]]
            present = {word["id"] for word in kept}
            if present.intersection(span["words"]):
                raise _refused(operation, comment,
                               "a later operation brings back words cut by this note")
            edge = span["edge"]
            if edge is not None:
                anchor = "from" if edge == "start" else "to"
                source = self.source(span["take"], operation, comment)
                order = {word.id: index for index, word in enumerate(source)}
                boundary_word = segment[anchor].get(
                    "word", kept[0 if edge == "start" else -1]["id"])
                boundary = self.source_index(span["take"], boundary_word, operation, comment)
                before = self.edge_words[span["operation"], span["segment"]]
                words = sorted((word for word in before if (
                    order[word["id"]] < boundary if edge == "start" else
                    order[word["id"]] > boundary)), key=lambda word: order[word["id"]])
                start, end = self.removal_span(
                    segment, [{"id": boundary_word}], operation, comment)
                inner = "end" if edge == "start" else "start"
                value = start if edge == "start" else end
                restored = self.edge_restores.get((span["operation"], span["segment"]), set())
                if (segment[anchor] != span["anchor_after"]
                        or any(word["id"] in restored for word in words)):
                    moves = self.edge_moves[span["segment"], anchor]
                    span.update(options=[],
                                combined_with=moves[moves.index(span["operation"]) + 1:])
                span.update(words=[word["id"] for word in words],
                            texts=[word["text"] for word in words])
                span[inner] = value
            spans.append(span)
        self.cut_spans = spans

    def remove(self, operation: dict, comment: str) -> tuple[str, str, list[dict]]:
        if "segment" in operation:
            segment = self.segment(operation["segment"], operation, comment)
            summary, removed = self.drop(segment, operation, comment)
            return segment["id"], summary, removed
        segment_id = operation["from"]["segment"]
        segment = self.segment(segment_id, operation, comment)
        first = self.occurrence(segment_id, operation["from"]["word"], operation, comment)
        last = self.occurrence(segment_id, operation["to"]["word"], operation, comment)
        if first > last:
            raise _invalid(operation, comment, "from must not be after to")
        kept = self.kept[segment_id]
        removed = kept[first:last + 1]
        if first == 0 and last == len(kept) - 1:
            summary, removed = self.drop(segment, operation, comment)
            return segment_id, summary, removed
        if first == 0:
            segment["from"] = {"word": kept[last + 1]["id"]}
            summary = f"Removed {_quote(removed)} from the start of {segment_id}."
        elif last == len(kept) - 1:
            segment["to"] = {"word": kept[first - 1]["id"]}
            summary = f"Removed {_quote(removed)} from the end of {segment_id}."
        else:
            take = segment["take"]
            start, end = self.removal_span(segment, removed, operation, comment)
            if end <= start:
                raise _refused(operation, comment,
                               "the manual removal has no positive source span.")
            for other in self.edit["segments"]:
                if other["id"] == segment_id or other["take"] != take:
                    continue
                other_start, other_end = self.source_span(other, operation, comment)
                if max(start, other_start) < min(end, other_end):
                    raise _refused(
                        operation, comment,
                        f"the manual removal would also cut segment {other['id']}, "
                        f"which overlaps its source span on take {take}.",
                    )
            self.actions.append(RemovalAction(
                "add_manual", operation["id"], comment,
                f"Remove {_quote(removed)} for {comment}.", take=take, start=start, end=end,
            ))
            summary = f"Removed {_quote(removed)} inside {segment_id} (fillers stage)."
        del kept[first:last + 1]
        return segment_id, summary, removed

    def trim(self, operation: dict, comment: str) -> tuple[str, str, list[dict]]:
        segment_id = operation["segment"]
        segment = self.segment(segment_id, operation, comment)
        index = self.occurrence(segment_id, operation["word"], operation, comment)
        kept = self.kept[segment_id]
        word = kept[index]
        if operation["edge"] == "start":
            removed = kept[:index]
            del kept[:index]
            segment["from"] = {"word": word["id"]}
            summary = f'Trimmed {segment_id} to start on "{word["text"]}".'
        else:
            removed = kept[index + 1:]
            del kept[index + 1:]
            segment["to"] = {"word": word["id"]}
            summary = f'Trimmed {segment_id} to end on "{word["text"]}".'
        return segment_id, summary, removed

    def restore(self, operation: dict, comment: str, item: dict,
                ranges: Mapping[str, Sequence[dict]],
                pieces: Mapping[str, dict]) -> tuple[str, str, list[dict]]:
        request = item.get("request")
        if not isinstance(request, dict) or request.get("type") != "restore":
            request = None
        range_id = operation.get("range") or (request or {}).get("range")
        if not range_id:
            raise _invalid(operation, comment,
                           "restore needs a range or the item's restore request")
        version = comment.split(":", 1)[0]
        removed = next((record for record in ranges.get(version, ())
                        if record["id"] == range_id), None)
        if removed is None:
            raise _refused(operation, comment,
                           f"range {range_id} is missing from version {version}.")
        if request is not None and any(request.get(key) != removed[key]
                                       for key in ("take", "start", "end")):
            raise _refused(operation, comment, "the removed part changed from the restore request.")
        if removed["kind"] == "trim":
            raise _refused(operation, comment,
                           "there is no removal or story cut to undo at this trim range.")
        if removed["kind"] == "story":
            try:
                index = int(range_id.rsplit(".", 1)[1])
                if index < 0:
                    raise ValueError
                segment_id = pieces[version]["pieces"][index]["segment"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise _refused(
                    operation, comment,
                    f"range {range_id} has no previous piece in version {version}.",
                ) from exc
        else:
            segment_id = range_id.rsplit(".", 1)[0]
        segment = next((segment for segment in self.edit["segments"]
                        if segment["id"] == segment_id), None)
        if segment is None or segment["take"] != removed["take"]:
            raise _refused(operation, comment,
                           f"the removed part no longer follows {segment_id} in this version.")
        words = ([{"id": word["id"], "text": word["text"]} for word in removed["words"]]
                 if removed["words_known"] else [])
        added = {word["id"] for word in words} - {word["id"] for word in self.kept[segment_id]}
        take = segment["take"]
        for word in words:
            if word["id"].split(":", 1)[0] != take:
                raise _invalid(operation, comment,
                               f"restored word {word['id']} belongs to another take")
            if take in self.sources:
                self.source_index(take, word["id"], operation, comment)
        if removed["kind"] == "story":
            anchor = segment["to"]
            if "word" in anchor:
                source = self.source(take, operation, comment)
                at = self.source_index(take, anchor["word"], operation, comment)
                if words:
                    meets_boundary = (at + 1 < len(source)
                                      and source[at + 1].id == words[0]["id"])
                else:
                    meets_boundary = removed["start"] - 0.5 <= source[at].end <= removed["start"]
            else:
                meets_boundary = (removed["start"] - 0.001 <= anchor["time"]
                                  <= removed["start"] + 0.001)
            if not meets_boundary:
                raise _refused(
                    operation, comment,
                    f"segment {segment_id} no longer ends where the removed part began, "
                    "for example after an earlier trim; "
                    "restore before trimming or ask the creator.",
                )
            segment["to"] = {"word": words[-1]["id"]} if words else {"time": removed["end"]}
            present = {word["id"] for word in self.kept[segment_id]}
            self.kept[segment_id].extend(word for word in words if word["id"] not in present)
            summary = (f"Brought back {_quote(words)} after {segment_id}." if words else
                       f"Brought back removed audio after {segment_id}.")
        elif removed["kind"] in _REMOVAL_KINDS:
            for removal_id in removed["removal_ids"]:
                self.actions.append(RemovalAction(
                    "reject", operation["id"], comment,
                    f"Bring back the removed {removed['kind']} for {comment}.",
                    removal_id=removal_id,
                ))
            # Source order decides the insertion point, never the cut's output seconds.
            order = ({word.id: index for index, word in enumerate(self.sources[take])}
                     if take in self.sources else None)

            def rank(word: dict) -> int:
                if order is None:
                    return int(word["id"].split(":")[1])
                if word["id"] not in order:
                    raise _invalid(operation, comment,
                                   f"word {word['id']} is missing from take {take}")
                return order[word["id"]]

            kept = self.kept[segment_id]
            for word in words:
                if any(existing["id"] == word["id"] for existing in kept):
                    continue
                at = next((index for index, existing in enumerate(kept)
                           if rank(existing) > rank(word)), len(kept))
                kept.insert(at, word)
            summary = f"Brought back a removed {removed['kind']} (fillers stage)."
        else:
            raise _invalid(operation, comment, f"unknown removed range kind {removed['kind']!r}")
        for key, before in self.edge_words.items():
            _, identifier = key
            if identifier == segment_id:
                if added:
                    self.edge_restores.setdefault(key, set()).update(added)
                present = {word["id"] for word in before}
                before.extend(copy.deepcopy(word) for word in words if word["id"] not in present)
        return segment_id, summary, words

    def check_edit(self, operation: dict | None = None, comment: str = "") -> None:
        try:
            parse_document("edit", self.edit)
        except ValidationError as exc:
            if operation is None:
                raise
            raise _invalid(operation, comment, str(exc)) from exc
        for segment in self.edit["segments"]:
            if segment["take"] not in self.sources:
                continue
            for edge in ("from", "to"):
                word_id = segment[edge].get("word")
                if word_id is None:
                    continue
                if operation is not None:
                    self.source_index(segment["take"], word_id, operation, comment)
                elif not any(word.id == word_id for word in self.sources[segment["take"]]):
                    raise ValidationError(f"edit segment {segment['id']}.{edge}",
                                          f"word {word_id} is missing from take {segment['take']}")


def plan_edit(
    edit: dict, cut_words: CutWords, source_words: Mapping[str, Words | None],
    items: Sequence[dict], operations: dict, *,
    ranges: Mapping[str, Sequence[dict]], pieces: Mapping[str, dict],
) -> EditPlan:
    """Plan ordered operations against a base edit and comment-version restore evidence.

    No project files, batch records, or removal decisions are written. The caller
    applies the returned document and fillers actions under the project lock.
    """
    operations = validate_operations(operations)
    by_comment = {f"{item['comment']['version']}:{item['comment']['id']}": item for item in items}
    planner = _Planner(edit, cut_words, source_words)
    broll_operations = []
    for entry in operations["items"]:
        comment = entry["comment"]
        if comment not in by_comment:
            raise _invalid(entry["operations"][0], comment,
                           "comment reference is not in this batch")
        item = by_comment[comment]
        for operation in entry["operations"]:
            kind = operation["kind"]
            if kind.startswith("broll_"):
                if item["comment"]["kind"] != "fix":
                    raise _invalid(operation, comment, "B-roll edits require a fix comment")
                broll_operations.append({**copy.deepcopy(operation), "comment": comment})
                continue
            before = {segment["id"]: {anchor: copy.deepcopy(segment[anchor])
                                      for anchor in ("from", "to")}
                      for segment in planner.edit["segments"]}
            if kind == "keep":
                if item["comment"]["kind"] != "keep":
                    raise _invalid(operation, comment, "keep requires a keep comment")
                touched, summary, words = [], "Kept, no edit", []
            elif kind == "cut":
                touched, summary, words = planner.cut(operation, comment, item)
                for segment in touched:
                    planner.refresh_text(segment)
            else:
                if kind == "remove":
                    segment, summary, words = planner.remove(operation, comment)
                elif kind == "trim":
                    segment, summary, words = planner.trim(operation, comment)
                else:
                    segment, summary, words = planner.restore(
                        operation, comment, item, ranges, pieces,
                    )
                planner.refresh_text(segment)
                touched = [segment]
            planner.check_edit(operation, comment)
            for segment in planner.edit["segments"]:
                for anchor in ("from", "to"):
                    if segment[anchor] != before[segment["id"]][anchor]:
                        planner.edge_moves.setdefault((segment["id"], anchor), []).append(
                            operation["id"])
            planner.changes.append({"operation": operation["id"], "comment": comment,
                                    "kind": kind, "summary": summary, "segments": touched,
                                    "words": copy.deepcopy(words[:200])})
    planner.finalize_cuts()
    if not planner.changes:
        planner.check_edit()
    return EditPlan(planner.edit, tuple(planner.actions), tuple(planner.changes),
                    tuple(broll_operations), tuple(planner.cut_spans))
