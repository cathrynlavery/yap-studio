"""Scoped keep constraints and checks against the next cut's evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from yap_studio.errors import ValidationError, YapError
from yap_studio.feedback.audio import compare_delivery, envelope_slice, media_envelope
from yap_studio.feedback.locate import Location
from yap_studio.feedback.moments import Occurrence, occurrences
from yap_studio.feedback.versions import (
    LayerWindow,
    VersionError,
    _receipt_pieces,
    load_version,
    verify_cut,
    version_cut_words,
    version_envelope,
    version_layers,
    version_pieces,
)
from yap_studio.project.model import CutWords, load_cut_words
from yap_studio.project.receipts import read_receipt
from yap_studio.review.model import Comment


def keep_constraint(
    comment: Comment, location: Location, *, version: str | None,
    envelope: dict | None = None, inputs: dict | None = None,
) -> dict:
    """Capture only the material protected by the keep's target."""
    target = comment.target or "story"
    words = location.words if target in ("story", "speech", "element") else ()
    pause = location.status == "silence" and target in ("story", "speech")
    if pause:
        words = tuple(word for word in (location.before, location.after) if word is not None)
    gaps = [b.start - a.end for a, b in zip(words, words[1:], strict=False)] if (
        target == "speech" or pause
    ) else []
    layers = tuple(layer for layer in location.layers
                   if (target == "caption" and layer.kind == "caption")
                   or (target == "element" and layer.kind != "caption"))
    constraint = {
        "comment": {"version": version, "id": comment.id, "revision": comment.updated_at},
        "text": comment.text,
        "target": target,
        "pause": pause,
        "words": [{"id": word.id, "segment": word.segment, "text": word.text} for word in words],
        "gaps": gaps,
        "layers": [{"id": layer.id, "kind": layer.kind, "text": layer.text} for layer in layers],
        "layers_known": location.layers_known,
    }
    if target == "speech":
        constraint["delivery"] = None
        takes = {word.id.split(":", 1)[0] for word in words}
        if envelope is not None and inputs is not None and words and all(
            f"take:{take}" in inputs and f"words:{take}" in inputs for take in takes
        ):
            start, end = words[0].start, words[-1].end
            constraint["delivery"] = {
                "start_s": start, "end_s": end, "envelope": envelope_slice(envelope, start, end),
                "sources": {take: {"take": inputs[f"take:{take}"],
                                   "words": inputs[f"words:{take}"]} for take in sorted(takes)},
            }
    return constraint


def element_keep(constraints: Sequence[dict], element: str) -> dict | None:
    """Find the keep that protects this element identity."""
    return next((constraint for constraint in constraints
                 if constraint.get("target") == "element"
                 and any(layer["id"] == element for layer in constraint["layers"])), None)


@dataclass(frozen=True)
class KeepResult:
    """The outcome and reason for checking one review constraint."""

    comment: dict
    text: str
    target: str | None
    status: Literal["kept", "broken", "unmeasured"]
    detail: str


@dataclass(frozen=True)
class KeepReport:
    """All results, including constraints that still need a person."""

    results: tuple[KeepResult, ...]

    @property
    def ok(self) -> bool:
        """Return whether no checked keep was broken."""
        return all(result.status != "broken" for result in self.results)

    @property
    def unmeasured(self) -> tuple[KeepResult, ...]:
        """Return keeps whose protected material could not be measured."""
        return tuple(result for result in self.results if result.status == "unmeasured")

    def to_dict(self) -> dict:
        """Return a JSON-ready report."""
        return {"ok": self.ok, "results": [asdict(result) for result in self.results]}


class KeepBrokenError(YapError):
    """Stop a next-version run and name every broken review constraint."""

    def __init__(self, report: KeepReport) -> None:
        self.report = report
        message = "\n".join(
            f"Keep comment {result.comment['id']} on {result.comment['version'] or 'cut'} "
            f'("{result.text[:60]}") is broken: {result.detail}'
            for result in report.results if result.status == "broken"
        )
        super().__init__(message, error_type="feedback_error", code="keep_broken",
                         suggested_action="Restore the kept material or ask the creator.")


def _runs(kept: Sequence[dict], items: tuple[Occurrence, ...]) -> list[tuple[Occurrence, ...]]:
    ids = [(word["id"], word["segment"]) for word in kept]
    return [items[start:start + len(ids)] for start in range(len(items) - len(ids) + 1)
            if [(word.id, word.segment) for word in items[start:start + len(ids)]] == ids]


def _word_problem(kept: Sequence[dict], items: tuple[Occurrence, ...]) -> str:
    for word in kept:
        if not any((item.id, item.segment) == (word["id"], word["segment"]) for item in items):
            detail = (f'word {word["id"]} "{word["text"]}" in {word["segment"]} '
                      "is missing from the new cut")
            other = list(dict.fromkeys(item.segment for item in items if item.id == word["id"]))
            if other:
                detail += f" (the same word is in {', '.join(other)})"
            return detail
    return "the order changed or other words were inserted between the kept words"


def check_keeps(
    constraints: Sequence[dict], words: CutWords, layers: tuple[LayerWindow, ...] | None,
    *, fps: Fraction, envelope: dict | None = None, sources: dict | None = None,
) -> KeepReport:
    """Check kept words, speech gaps, and scoped layer identities in a new cut."""
    if fps <= 0:
        raise ValidationError("fps", "must be positive")
    items = occurrences(words)
    results = []
    for constraint in constraints:
        target = constraint.get("target") or "story"
        kept = constraint["words"]
        state, detail = "kept", "kept material is present"
        if target == "video":
            state, detail = "unmeasured", "needs a person to check"
        elif target in ("story", "speech"):
            runs = _runs(kept, items) if kept else []
            if not kept:
                state, detail = "unmeasured", "no words to check"
            elif not runs:
                state, detail = "broken", _word_problem(kept, items)
            elif constraint["gaps"]:
                problems = []
                for run in runs:
                    problem = None
                    for a, b, gap in zip(run, run[1:], constraint["gaps"], strict=False):
                        new_gap = b.start - a.end
                        if abs(new_gap - gap) > float(1 / fps) + 1e-9:
                            problem = (f'gap between {a.id} "{a.text}" and {b.id} "{b.text}" '
                                       f"changed from {gap:.4f}s to {new_gap:.4f}s")
                            break
                    if problem is None:
                        break
                    problems.append(problem)
                else:
                    state, detail = "broken", problems[0]
            if target == "speech" and state == "kept":
                delivery = constraint.get("delivery")
                reason = None
                if delivery is None:
                    reason = "no delivery evidence for the reviewed version"
                elif envelope is None:
                    reason = "no voice envelope for the new version"
                elif sources is None:
                    reason = "no source inputs for the new version"
                if reason is not None:
                    state = "unmeasured"
                    detail = f"words and pauses kept; delivery not measured: {reason}"
                else:
                    changed = next((take for take, hashes in delivery["sources"].items()
                                    if any(sources.get(f"{kind}:{take}") != digest
                                           for kind, digest in hashes.items())), None)
                    if changed is not None:
                        state, detail = "broken", f"take {changed} changed"
                    else:
                        # Only runs whose pauses passed can supply delivery evidence.
                        eligible = [run for run in runs if all(
                            abs(b.start - a.end - gap) <= float(1 / fps) + 1e-9
                            for a, b, gap in zip(run, run[1:], constraint["gaps"], strict=False)
                        )]
                        outcomes = [compare_delivery(delivery["envelope"], envelope_slice(
                            envelope, run[0].start, run[-1].end
                        )) for run in eligible]
                        state, detail = next((outcome for outcome in outcomes
                                              if outcome[0] == "kept"), outcomes[0])
                        if state == "unmeasured":
                            detail = f"words and pauses kept; delivery not measured: {detail}"
        elif target in ("caption", "element"):
            if not constraint.get("layers_known", False):
                state, detail = "unmeasured", "no layer data for the reviewed version"
            elif not constraint["layers"]:
                state, detail = "unmeasured", f"no {target} in the kept range"
            elif layers is None:
                state, detail = "unmeasured", "no layer data for the new version"
            else:
                by_id = {layer.id: layer for layer in layers}
                kept_ids = {(word["id"], word["segment"]) for word in kept}
                anchors = tuple(item for item in items if (item.id, item.segment) in kept_ids)
                for layer in constraint["layers"]:
                    found = by_id.get(layer["id"])
                    if found is None:
                        state, detail = "broken", f"layer {layer['id']} is missing"
                    elif target == "caption" and (
                        found.kind != "caption" or found.text != layer["text"]
                    ):
                        state, detail = "broken", f"caption {layer['id']} text or kind changed"
                    elif target == "element" and found.kind != layer["kind"]:
                        state, detail = "broken", f"layer {layer['id']} kind changed"
                    elif target == "element" and kept and not any(
                        found.start < word.end and found.end > word.start for word in anchors
                    ):
                        state = "broken"
                        detail = f"layer {layer['id']} no longer overlaps kept words"
                    if state == "broken":
                        break
        else:
            raise ValidationError("constraint.target", "unknown keep target")
        results.append(KeepResult(dict(constraint["comment"]), constraint["text"],
                                  constraint.get("target"), state, detail))
    return KeepReport(tuple(results))


def require_keeps(
    constraints: Sequence[dict], words: CutWords, layers: tuple[LayerWindow, ...] | None,
    *, fps: Fraction, envelope: dict | None = None, sources: dict | None = None,
) -> KeepReport:
    """Raise a named error when any scoped keep is broken."""
    report = check_keeps(constraints, words, layers, fps=fps, envelope=envelope, sources=sources)
    if not report.ok:
        raise KeepBrokenError(report)
    return report


def check_version(
    project_dir: Path, batch_id: str | None = None, version_id: str | None = None,
) -> KeepReport:
    """Check batch constraints against a saved version or the current cut."""
    from yap_studio.feedback.batch import FeedbackError, latest_batch, load_batch

    root = Path(project_dir)
    batch = load_batch(root, batch_id) if batch_id is not None else latest_batch(root)
    if batch is None:
        raise FeedbackError("No feedback batch exists.",
                            suggested_action="Run yap-studio feedback build first.")
    needs_delivery = any(constraint.get("delivery") is not None
                         for constraint in batch["constraints"])
    if version_id is not None:
        version = load_version(root, version_id)
        words, layers = version_cut_words(version), version_layers(version)
        envelope = version_envelope(version) if needs_delivery else None
        sources = version_pieces(version).get("inputs")
        fps = Fraction(version.media["fps"])
    else:
        receipt = read_receipt(root / "cut")
        try:
            verify_cut(root, receipt)
        except VersionError as exc:
            raise FeedbackError(str(exc), suggested_action=exc.suggested_action) from exc
        words, layers = load_cut_words(root / "cut" / "cut-words.json"), None
        envelope = media_envelope(root / "cut" / "cut.mp4") if needs_delivery else None
        sources = receipt["inputs"]
        fps = Fraction(_receipt_pieces(receipt, root / "cut" / "receipt.json")["fps"])
    return check_keeps(batch["constraints"], words, layers, fps=fps,
                       envelope=envelope, sources=sources)
