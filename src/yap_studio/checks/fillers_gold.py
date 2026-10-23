"""Score filler removals against labeled source-time intervals.

A gold.json document has schema "yap-studio/filler-gold@1", a required nonempty
list of unique take ids (t1, t2, and so on), and a spans list. Each span has take,
start and end in seconds, and label: "must-cut", "optional-cut", or "must-keep".
An optional note is a string. An optional string id identifies the span in
reports; otherwise its position supplies g1, g2, and so on. All span ids must be
unique, including generated ids. Intervals must have positive length.
Must-keep intervals cannot overlap cut intervals on one take.
Unknown fields are rejected. For example, a span can be
{"take": "t1", "start": 1.2, "end": 1.45, "label": "must-cut", "note": "um"}.

A gold set folder contains one immediate subfolder per case, each containing
gold.json and the fillers stage's removals.json. Synthetic sets can live in the
test fixtures. The same layout can live anywhere outside the repository for
private takes; set YAP_STUDIO_FILLER_GOLD to that folder to enable the private
acceptance test. The scorer only reads these JSON files and needs no video.

Case reports have counts and settings objects, missed spans with coverage,
keep_cut spans with cut_s and removal_ids, and wrong removals with reasons.
Gold spans and removals retain document order in reports; set cases are sorted
by folder name. ignored_removals counts all removals on uncovered takes.
Precision requires overlap with the union of must-cut and optional-cut marks
to exceed 1e-9 seconds and cover at least min_marked of the removal's length.
The default min_marked is 0.5 because removals include silence padding around
a filler. Positive overlap below this share is reported as mostly unmarked;
no marked overlap is unmarked.
Each must-keep span is cut when its cut seconds exceed the smaller of
tolerance_s (with a 1e-9 second comparison allowance) and half its length.
Precision checks this same limit separately for each must-keep span. Only
cutting a must-keep span beyond its limit makes a case fail. Set totals pool
counts and recompute recall, precision and f1 from those counts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from yap_studio.checks.spans import Span
from yap_studio.errors import InputError, ValidationError
from yap_studio.project.model import Removals, json_schema, parse_document

_LABELS = ("must-cut", "optional-cut", "must-keep")
_EPSILON = 1e-9
_REMOVAL_FIELDS = json_schema("removals")["properties"]["removals"]["items"]["properties"]


def _number(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(field, "must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValidationError(field, "must be a finite number") from exc
    if not math.isfinite(number):
        raise ValidationError(field, "must be a finite number")
    return number


@dataclass(frozen=True)
class GoldSpan:
    """One named, validated interval on a take's source clock."""

    id: str
    take: str
    start: float
    end: float
    label: str
    note: str | None = None

    def __post_init__(self) -> None:
        field = f"span {self.id!r}"
        if not isinstance(self.id, str):
            raise ValidationError(field, "id must be a string")
        if not isinstance(self.take, str) or re.fullmatch(r"t[1-9][0-9]*", self.take) is None:
            raise ValidationError(field, "take must be a take id like t1")
        if self.label not in _LABELS:
            raise ValidationError(field, f"label must be one of {_LABELS}")
        if self.note is not None and not isinstance(self.note, str):
            raise ValidationError(field, "note must be a string")
        start = _number(self.start, f"{field}.start")
        end = _number(self.end, f"{field}.end")
        try:
            Span(self.take, self.label, start, end)
        except ValidationError as exc:
            raise ValidationError(field, exc.detail) from exc
        if start >= end:
            raise ValidationError(field, "require start < end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True)
class Gold:
    """The covered takes and labeled spans of a validated gold document."""

    takes: tuple[str, ...]
    spans: tuple[GoldSpan, ...]

    def __post_init__(self) -> None:
        if not self.takes or any(
            not isinstance(take, str) or re.fullmatch(r"t[1-9][0-9]*", take) is None
            for take in self.takes
        ):
            raise ValidationError("takes", "must contain take ids like t1")
        if len(set(self.takes)) != len(self.takes):
            raise ValidationError("takes", "take ids must be unique")
        seen_ids: set[str] = set()
        for span in self.spans:
            if span.id in seen_ids:
                raise ValidationError(f"span {span.id!r}", "duplicate id")
            seen_ids.add(span.id)
            if span.take not in self.takes:
                raise ValidationError(f"span {span.id!r}", "take is not in takes")
            if span.label != "must-keep":
                continue
            for other in self.spans:
                if (
                    other.take == span.take
                    and other.label != "must-keep"
                    and min(span.end, other.end) > max(span.start, other.start)
                ):
                    raise ValidationError(
                        f"span {span.id!r}", f"must-keep overlaps span {other.id!r}"
                    )


def _object(
    data: Any, field: str, required: tuple[str, ...], optional: tuple[str, ...] = ()
) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValidationError(field, "must be an object")
    unknown = data.keys() - set(required + optional)
    if unknown:
        raise ValidationError(field, "unknown keys: " + ", ".join(sorted(map(str, unknown))))
    missing = set(required) - data.keys()
    if missing:
        raise ValidationError(field, "missing keys: " + ", ".join(sorted(missing)))
    return data


def parse_gold(data: Mapping[str, Any]) -> Gold:
    """Validate a gold document and return frozen interval models."""
    data = _object(data, "gold", ("schema", "takes", "spans"))
    if data["schema"] != "yap-studio/filler-gold@1":
        raise ValidationError("schema", "expected yap-studio/filler-gold@1")
    if not isinstance(data["takes"], list):
        raise ValidationError("takes", "must be a nonempty list of unique take ids")
    if not isinstance(data["spans"], list):
        raise ValidationError("spans", "must be a list")
    spans = []
    for index, item in enumerate(data["spans"], 1):
        identity = item.get("id", f"g{index}") if isinstance(item, Mapping) else f"g{index}"
        item = _object(
            item, f"span {identity!r}", ("take", "start", "end", "label"), ("id", "note")
        )
        for key in ("id", "note"):
            if key in item and not isinstance(item[key], str):
                raise ValidationError(f"span {identity!r}", f"{key} must be a string")
        spans.append(GoldSpan(id=identity, **{key: item[key] for key in item if key != "id"}))
    return Gold(tuple(data["takes"]), tuple(spans))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeError) as exc:
        raise InputError(str(path), f"cannot read JSON: {exc}") from exc


def load_gold(path: str | Path) -> Gold:
    """Read and validate one gold.json file."""
    return parse_gold(_read_json(Path(path)))


def load_removals(path: str | Path) -> Removals:
    """Read validated removals from a file or a project folder."""
    source = Path(path)
    if source.is_dir():
        source = source / "removals.json"
    return cast(Removals, parse_document("removals", _read_json(source), filename=str(source)))


def _selection(values: Sequence[str], field: str) -> tuple[str, ...]:
    allowed = _REMOVAL_FIELDS["status" if field == "statuses" else "kind"]["enum"]
    if isinstance(values, str) or not isinstance(values, Sequence) or any(
        not isinstance(value, str) or value not in allowed for value in values
    ):
        raise ValidationError(field, f"must be a sequence drawn from {allowed}")
    return tuple(dict.fromkeys(values))


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap(start: float, end: float, intervals: Iterable[tuple[float, float]]) -> float:
    return math.fsum(max(0.0, min(end, right) - max(start, left)) for left, right in intervals)


def _cuts_keep(cut_s: float, span: GoldSpan, tolerance_s: float) -> bool:
    """Compare a cut against this must-keep span's absolute and relative limits."""
    return cut_s > min(tolerance_s + _EPSILON, (span.end - span.start) / 2)


def _metrics(counts: Mapping[str, int]) -> dict[str, float]:
    recall = counts["must_cut_recalled"] / counts["must_cut"] if counts["must_cut"] else 1.0
    precision = (
        counts["correct_removals"] / counts["scored_removals"]
        if counts["scored_removals"] else 1.0
    )
    return {
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / (recall + precision) if recall + precision else 0.0,
    }


def score_fillers_gold(
    gold: Gold | Mapping[str, Any],
    removals: Removals | dict[str, Any],
    *,
    statuses: Sequence[str] = ("applied", "proposed"),
    kinds: Sequence[str] = ("filler", "stutter", "restart"),
    min_coverage: float = 0.8,
    min_marked: float = 0.5,
    tolerance_s: float = 1 / 30,
) -> dict[str, Any]:
    """Measure span recall and removal precision, failing on must-keep cuts."""
    statuses = _selection(statuses, "statuses")
    kinds = _selection(kinds, "kinds")
    min_coverage = _number(min_coverage, "min_coverage")
    min_marked = _number(min_marked, "min_marked")
    tolerance_s = _number(tolerance_s, "tolerance_s")
    if not 0 < min_coverage <= 1:
        raise ValidationError("min_coverage", "require 0 < min_coverage <= 1")
    if not 0 < min_marked <= 1:
        raise ValidationError("min_marked", "require 0 < min_marked <= 1")
    if tolerance_s < 0:
        raise ValidationError("tolerance_s", "must be non-negative")
    gold = gold if isinstance(gold, Gold) else parse_gold(gold)
    if not isinstance(removals, Removals):
        removals = cast(Removals, parse_document("removals", removals))
    counted = [
        row for row in removals.removals
        if row.status in statuses and row.by == "fillers" and row.take in gold.takes
    ]
    removed = {
        take: _union((row.start, row.end) for row in counted if row.take == take)
        for take in gold.takes
    }
    cut = {
        take: _union((span.start, span.end) for span in gold.spans
                     if span.take == take and span.label != "must-keep")
        for take in gold.takes
    }
    keep = {
        take: tuple(span for span in gold.spans
                    if span.take == take and span.label == "must-keep")
        for take in gold.takes
    }
    missed = []
    keep_cut = []
    for span in gold.spans:
        cut_s = _overlap(span.start, span.end, removed[span.take])
        identity = {"id": span.id, "take": span.take, "start": span.start, "end": span.end}
        if span.label == "must-cut":
            coverage = min(1.0, cut_s / (span.end - span.start))
            if coverage + _EPSILON < min_coverage:
                missed.append({**identity, "coverage": coverage})
        elif span.label == "must-keep" and _cuts_keep(cut_s, span, tolerance_s):
            keep_cut.append({
                **identity, "cut_s": cut_s,
                "removal_ids": [row.id for row in counted if row.take == span.take
                                and min(row.end, span.end) > max(row.start, span.start)],
            })
    scored = [row for row in counted if row.kind in kinds]
    wrong = []
    for row in scored:
        if any(
            _cuts_keep(max(0.0, min(row.end, span.end) - max(row.start, span.start)),
                       span, tolerance_s)
            for span in keep[row.take]
        ):
            wrong.append({"id": row.id, "reason": "cuts must-keep"})
        else:
            marked_s = _overlap(row.start, row.end, cut[row.take])
            if marked_s == 0:
                wrong.append({"id": row.id, "reason": "unmarked"})
            elif marked_s <= _EPSILON or marked_s < min_marked * (row.end - row.start):
                wrong.append({"id": row.id, "reason": "mostly unmarked"})
    must_cut = sum(span.label == "must-cut" for span in gold.spans)
    counts = {
        "must_cut": must_cut,
        "must_cut_recalled": must_cut - len(missed),
        "optional_cut": sum(span.label == "optional-cut" for span in gold.spans),
        "must_keep": sum(span.label == "must-keep" for span in gold.spans),
        "must_keep_cut": len(keep_cut),
        "scored_removals": len(scored),
        "correct_removals": len(scored) - len(wrong),
        "ignored_removals": sum(row.take not in gold.takes for row in removals.removals),
    }
    return {
        "schema": "yap-studio/filler-gold-score@1", "passed": not keep_cut,
        **_metrics(counts), "counts": counts,
        "missed": missed, "keep_cut": keep_cut, "wrong": wrong,
        "settings": {"statuses": list(statuses), "kinds": list(kinds),
                     "min_coverage": min_coverage, "min_marked": min_marked,
                     "tolerance_s": tolerance_s},
    }


def score_gold_set(
    folder: str | Path,
    *,
    statuses: Sequence[str] = ("applied", "proposed"),
    kinds: Sequence[str] = ("filler", "stutter", "restart"),
    min_coverage: float = 0.8,
    min_marked: float = 0.5,
    tolerance_s: float = 1 / 30,
) -> dict[str, Any]:
    """Score immediate gold case folders and pool their counts and metrics."""
    root = Path(folder)
    try:
        folders = sorted(
            (case for case in root.iterdir() if case.is_dir() and (case / "gold.json").is_file()),
            key=lambda case: case.name,
        )
    except OSError as exc:
        raise InputError(str(root), str(exc)) from exc
    if not folders:
        raise InputError(str(root), "no cases containing gold.json")
    cases = {
        case.name: score_fillers_gold(
            load_gold(case / "gold.json"), load_removals(case), statuses=statuses,
            kinds=kinds, min_coverage=min_coverage, min_marked=min_marked, tolerance_s=tolerance_s,
        )
        for case in folders
    }
    counts = {
        key: sum(case["counts"][key] for case in cases.values())
        for key in next(iter(cases.values()))["counts"]
    }
    return {
        "schema": "yap-studio/filler-gold-set@1",
        "passed": all(case["passed"] for case in cases.values()), "cases": cases,
        "totals": {**_metrics(counts), "counts": counts},
    }


def _metric_lines(score: Mapping[str, Any]) -> tuple[str, str]:
    """Format recall and precision with their pooled or case counts."""
    counts = score["counts"]
    return (
        f"recall {score['recall']:.0%} "
        f"({counts['must_cut_recalled']} of {counts['must_cut']} must-cut)",
        f"precision {score['precision']:.0%} "
        f"({counts['correct_removals']} of {counts['scored_removals']} removals)",
    )


def _print_score(score: Mapping[str, Any]) -> None:
    """Print one case's metrics, findings, and preservation result."""
    for line in _metric_lines(score):
        print(line)
    for span in score["missed"]:
        print(
            f"missed {span['id']} {span['take']} {span['start']:.3f} to {span['end']:.3f} "
            f"coverage {span['coverage']:.0%}"
        )
    for span in score["keep_cut"]:
        print(
            f"MUST-KEEP CUT {span['id']} {span['take']} "
            f"{span['start']:.3f} to {span['end']:.3f} "
            f"by {', '.join(span['removal_ids'])} ({span['cut_s']:.3f} s)"
        )
    for removal in score["wrong"]:
        print(f"wrong {removal['id']}: {removal['reason']}")
    print("passed" if score["passed"] else
          f"FAILED: {score['counts']['must_keep_cut']} must-keep spans cut")


def _run(args: argparse.Namespace) -> int:
    """Score a gold file or set with the standard text or JSON error format."""
    source = Path(args.gold)
    settings: dict[str, Any] = {
        "min_coverage": args.min_coverage, "min_marked": args.min_marked,
        "tolerance_s": args.tolerance_s,
    }
    if args.status is not None:
        settings["statuses"] = args.status
    if args.kind is not None:
        settings["kinds"] = args.kind
    try:
        if source.is_dir():
            if args.removals is not None:
                raise InputError(str(source), "--removals cannot be used with a gold set folder")
            result = score_gold_set(source, **settings)
        else:
            removals = (
                args.removals if args.removals is not None else source.parent / "removals.json"
            )
            result = score_fillers_gold(load_gold(source), load_removals(removals), **settings)
    except (InputError, ValidationError) as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    elif "cases" in result:
        for name, score in result["cases"].items():
            print(f"{name}:")
            _print_score(score)
            print()
        print("pooled totals: " + ", ".join(_metric_lines(result["totals"])))
    else:
        _print_score(result)
    return 0 if result["passed"] else 1


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the command for scoring filler removals against gold labels."""
    parser = subparsers.add_parser(
        "fillers-score", help="score filler removals against gold labels"
    )
    parser.add_argument("gold", metavar="GOLD")
    parser.add_argument("--removals", metavar="PATH", help="removals file or project folder")
    parser.add_argument("--status", action="append", metavar="STATUS")
    parser.add_argument("--kind", action="append", metavar="KIND")
    parser.add_argument("--min-coverage", type=float, default=0.8, metavar="X")
    parser.add_argument("--min-marked", type=float, default=0.5, metavar="X")
    parser.add_argument("--tolerance-s", type=float, default=1 / 30, metavar="X")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=_run)
