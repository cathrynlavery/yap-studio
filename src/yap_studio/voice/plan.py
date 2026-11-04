"""Free word-range proposals and picture coverage checks."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

from yap_studio import elevenlabs, paths
from yap_studio.checks.model import format_time
from yap_studio.plan import snap_frame
from yap_studio.project.lock import project_lock
from yap_studio.project.receipts import StageRun, read_receipt
from yap_studio.stages.cut import CutPlan, resolve_cut
from yap_studio.voice.model import VoiceError, read, write


def source_sample(plan: CutPlan, take: str, time: Fraction = Fraction(0)) -> int:
    """Use the cut's container clock, rounding after adding the video offset."""
    return snap_frame(Fraction(str(plan.media[take].video_offset_s)) + time, Fraction(48000))


def patch_range(plan: CutPlan, fix: dict) -> tuple[float, float]:
    first = fix["patch"]["start_sample"] - source_sample(plan, fix["take"])
    return first / 48000, (first + fix["patch"]["samples"]) / 48000


def list_fixes(project_dir: Path) -> dict:
    """Read validated fixes for the CLI and review page."""
    return read(Path(project_dir), "fixes", optional=True) or {
        "schema": "yap-studio/voice-fixes@1",
        "fixes": [],
    }


def occurrences(
    plan: CutPlan, fix: dict, *, start_s: float | None = None, end_s: float | None = None
) -> list[tuple[float, float]]:
    """Project a whole source slot onto every retained occurrence."""
    start = fix["source"]["start_s"] if start_s is None else start_s
    end = fix["source"]["end_s"] if end_s is None else end_s
    spans = [
        (
            piece.out_f0 / 30 + start - float(piece.src_t0),
            piece.out_f0 / 30 + end - float(piece.src_t0),
        )
        for piece in plan.pieces
        if piece.take == fix["take"]
        and float(piece.src_t0) <= start + 1e-8
        and float(piece.src_t1) >= end - 1e-8
    ]
    if not spans:
        raise VoiceError(
            "All replaced words and their slot must be kept together in the cut.",
            code="voice_words_dropped",
        )
    # A second retained partial occurrence must never get a partial word replacement.
    for piece in plan.pieces:
        if (
            piece.take == fix["take"]
            and float(piece.src_t0) < end
            and float(piece.src_t1) > start
            and not (float(piece.src_t0) <= start + 1e-8 and float(piece.src_t1) >= end - 1e-8)
        ):
            raise VoiceError("A cut drops part of the replaced words.", code="voice_words_dropped")
    return spans


def _rendered_graphic_frames(plan: CutPlan) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Use only intact graphic layers rendered for the current timeline."""
    from yap_studio.errors import YapError
    from yap_studio.stages.assemble import opaque_graphic_frames
    from yap_studio.stages.graphics import graphics_snapshot

    root = plan.context.project_dir
    record = read_receipt(root / "graphics")
    try:
        if record and record.get("status") == "verified":
            snapshot = graphics_snapshot(plan.context, placements=record["elements"])
            if StageRun(root, "graphics").up_to_date(snapshot.inputs):
                owned = {row["path"] for row in record["outputs"]}
                layers = [row for row in record["elements"]
                          if row["kind"] == "graphic" and row["layer"] == "front"
                          and row["path"] in owned]
                return opaque_graphic_frames(root, layers), {}
    except (YapError, OSError, KeyError, TypeError, ValueError):
        pass
    try:
        planned = graphics_snapshot(plan.context)
        layers = [layer.placement() for layer in planned.graphics]
        pending = {row["id"]: set(range(row["start_frame"], row["start_frame"] + row["frames"]))
                   for row in layers if row["layer"] == "front"
                   and row["x"] == row["y"] == 0 and row["width"] == 1080 and row["height"] == 1920}
        return {}, pending
    except (YapError, OSError, KeyError, TypeError, ValueError):
        return {}, {}


def require_picture(
    plan: CutPlan,
    fix: dict,
    *,
    elements: list[dict] | None = None,
    graphic_coverage: dict[str, set[int]] | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
) -> dict:
    """Require full-frame cover over the slot and a tenth of a second either side."""
    ranges = occurrences(plan, fix, start_s=start_s, end_s=end_s)
    take = next(take for take in plan.context.project.takes if take.id == fix["take"])
    if take.voiceover:
        return {"kind": "voiceover"}
    if elements is None:
        root = plan.context.project_dir
        if (root / "timeline").is_dir() and any((root / "timeline").glob("*.json")):
            elements = [item.to_dict() for item in plan.context.resolved_timeline().elements]
        else:
            elements = []
    pending = {}
    if graphic_coverage is None and any(
        item["kind"] == "graphic" and item.get("layer", "front") == "front" for item in elements
    ):
        graphic_coverage, pending = _rendered_graphic_frames(plan)
    covered, unrendered = set(), set()
    for item in elements:
        if (item["kind"] not in ("insert", "graphic")
                or item.get("layer", "front") != "front"
                or item.get("media", {}).get("fit") == "split"):
            continue
        if item["kind"] == "graphic":
            covered.update((graphic_coverage or {}).get(item["id"], set()))
            unrendered.update(pending.get(item["id"], set()))
        else:
            covered.update(range(item["start_frame"], item["end_frame"]))
    for first, last in ranges:
        first, last = max(0, first - 0.1), min(plan.total_frames / 30, last + 0.1)
        required = set(range(math.floor(first * 30 + 1e-7), math.ceil(last * 30 - 1e-7)))
        if not required <= covered:
            hint = ", or render the graphics first" if required <= covered | unrendered else ""
            raise VoiceError(
                f"Your face is on screen at {format_time(first + 1e-8)} to "
                f"{format_time(last + 1e-8)}, so a re-said word would not match your lips. "
                f"Put B-roll over this moment first, or record the line again{hint}.",
                code="voice_mouth_on_screen",
            )
    return {"kind": "covered"}


def validate_words(plan: CutPlan, fix: dict) -> None:
    snapshot = plan.source_words.get(fix["take"])
    words = list(snapshot.words) if snapshot else []
    selected = [word for word in words if word.id in fix["words"]]
    if (
        [word.id for word in selected] != fix["words"]
        or not selected
        or " ".join(word.text for word in selected) != fix["said"]
    ):
        raise VoiceError(
            "The replaced words changed. Propose this fix again.", code="voice_words_changed"
        )
    indices = [words.index(word) for word in selected]
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise VoiceError(
            "Choose contiguous words from one take.", code="voice_words_not_contiguous"
        )
    if (
        abs(selected[0].start - fix["source"]["start_s"]) > 0.001 + 1e-8
        or abs(selected[-1].end - fix["source"]["end_s"]) > 0.001 + 1e-8
    ):
        raise VoiceError(
            "The words moved since this fix was made. Propose it again.",
            code="voice_words_changed",
        )
    occurrences(plan, fix)


def estimate(say: str, budget: dict, *, sent: str | None = None) -> dict:
    unavailable = {
        name: (reason.get("permission") or reason.get("code", "could_not_read"))
        if isinstance(reason, dict) else reason
        for name, reason in budget.get("unavailable", {}).items()
    }
    return {
        "characters": len(say if sent is None else sent),
        "credits": len(say if sent is None else sent) * budget["multiplier"],
        "model": budget.get("model", elevenlabs.VOICE_MODEL),
        "multiplier": budget["multiplier"],
        "source": budget["source"],
        **({"unavailable": unavailable} if unavailable else {}),
    }


def propose(project_dir: Path, first: str, last: str, say: str, *, why: str = "",
            from_take_words: tuple[str, str] | None = None, clone: bool = False) -> dict:
    """Propose one short correction without making paid requests."""
    root = Path(project_dir)
    if clone and from_take_words:
        raise VoiceError("Choose recorded words or --clone, not both.")
    say = say.strip()
    if not say or len(say) > 60 or not 1 <= len(say.split()) <= 8:
        raise VoiceError("Use 1 to 8 new words, at most 60 characters.", code="voice_text_too_long")
    with paths.operation():
        plan = resolve_cut(root)
        take = first.split(":")[0]
        snapshot = plan.source_words.get(take)
        words = list(snapshot.words) if snapshot else []
        ids = [word.id for word in words]
        if first not in ids or last not in ids or ids.index(last) < ids.index(first):
            raise VoiceError(
                "Choose contiguous words from one take.", code="voice_words_not_contiguous"
            )
        selected = words[ids.index(first) : ids.index(last) + 1]
        if len(selected) > 6:
            raise VoiceError("Replace at most 6 words at once.", code="voice_too_many_words")
        fix = {
            "id": "f1",
            "take": take,
            "words": [word.id for word in selected],
            "said": " ".join(word.text for word in selected),
            "say": say,
            "why": why,
            "status": "proposed",
            "source": {"start_s": selected[0].start, "end_s": selected[-1].end},
        }
        validate_words(plan, fix)
        fix["picture"] = require_picture(plan, fix)
        with project_lock(root):
            data = list_fixes(root)
            for previous in data["fixes"]:
                if (
                    previous["status"] != "rejected"
                    and previous["take"] == take
                    and previous["source"]["start_s"] < fix["source"]["end_s"]
                    and previous["source"]["end_s"] > fix["source"]["start_s"]
                ):
                    raise VoiceError(
                        "This range overlaps another voice fix.", code="voice_fix_overlap"
                    )
            fix["id"] = f"f{max((int(row['id'][1:]) for row in data['fixes']), default=0) + 1}"
            from yap_studio.voice.own import find_own
            from yap_studio.voice.recipe import load_recipe

            origin = None if clone else find_own(plan, fix, data["fixes"], chosen=from_take_words)
            if origin is None:
                fix["from"] = {"kind": "clone", **load_recipe()}
                from yap_studio.voice.make import _request

                request, _ = _request(plan, fix, {"voice": {"id_sha256": ""}}, new=False)
                budget = {**elevenlabs.voice_budget(request["model"]), "model": request["model"]}
                fix["estimate"] = estimate(say, budget, sent=request["text"])
            else:
                fix["from"] = origin
                fix["estimate"] = estimate(say, {
                    "multiplier": 0, "source": "own", "model": "own",
                })
            data["fixes"].append(fix)
            write(root, "fixes", data)
        return fix
