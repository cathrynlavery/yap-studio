"""Run every deterministic stage in order, stop at the first failed check, resume on rerun."""

from __future__ import annotations

import json
import shlex
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from yap_studio import ffmpeg, paths, receipt
from yap_studio.captions import phrase_keys
from yap_studio.checks.kept_words import check_kept_words, words_summary
from yap_studio.checks.platform import check_platform, load_presets
from yap_studio.errors import InputError, YapError
from yap_studio.project.receipts import failed_assert_lines, read_receipt
from yap_studio.project.status import STAGES, ProjectContext, _stage_state, ingest_speech_warnings
from yap_studio.stages import assemble as assemble_stage
from yap_studio.stages import captions as captions_stage
from yap_studio.stages import cut as cut_stage
from yap_studio.stages import fillers as fillers_stage
from yap_studio.stages import graphics as graphics_stage
from yap_studio.stages import music as music_stage

STEPS = ("ingest", "fillers", "cut", "music", "captions", "graphics", "assemble", "verify",
         "words", "version")

# A failed check exits 1, a refusal to overwrite or a busy project exits 3, anything else 2.
_CHECK_CODES = frozenset({"assembly_checks_failed", "cut_failed", "platform_failed",
                          "final_unverified", "music_contract_failed", "cut_frame_mismatch",
                          "cut_sample_mismatch", "words_dropped"})
_REFUSED_CODES = frozenset({"unsafe_path", "timeline_refused", "camera3d_refused", "stale_cut",
                            "stage_running", "base_changed", "project_locked", "words_referenced",
                            "caption_output_exists", "caption_output_locked",
                            "version_refused", "pilot_range_refused"})


class RunStop(YapError):
    """A stage check failed; the message says why and the suggested action says how to fix it."""

    def __init__(self, message: str, *, code: str, suggested_action: str) -> None:
        super().__init__(message, error_type="run_stopped", code=code,
                         suggested_action=suggested_action)


def exit_status(error: YapError) -> int:
    """Map a stop to the command's exit code."""
    if error.code in _CHECK_CODES:
        return 1
    if error.code in _REFUSED_CODES or error.error_type in {"ingest_refusal",
                                                           "timeline_refusal"}:
        return 3
    return 2


def _ingest(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
    """Require verified source words and a written story; run never transcribes or writes it."""
    quoted = shlex.quote(str(root))
    spec = next(spec for spec in STAGES if spec.name == "ingest")
    ingest = _stage_state(ProjectContext(root), spec, {})
    if ingest["state"] != "done":
        raise RunStop(
            f"Ingest is {ingest['state']}: {ingest['reason']}.", code="ingest_needed",
            suggested_action=(f"Run yap-studio ingest {quoted}, check edit.json still matches "
                              f"the words, then run yap-studio run {quoted} again."),
        )
    state["warnings"].extend(ingest_speech_warnings(root))
    if not (root / "edit.json").is_file():
        raise RunStop(
            "edit.json is missing: the story order is not written yet.", code="edit_missing",
            suggested_action=(f"Read yap-studio words {quoted}, write the edit.json segments, "
                              f"then run yap-studio run {quoted} again."),
        )
    return True, "source words verified"


def _fillers(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
    spec = next(spec for spec in STAGES if spec.name == "cut")
    kept = _stage_state(ProjectContext(root), spec, {"ingest": "done"})["state"] == "done"
    if kept:
        try:
            entries = json.loads((root / "removals.json").read_bytes())["removals"]
        except FileNotFoundError:
            entries = []
        counts = Counter(item["status"] for item in entries)
        try:
            result = json.loads((root / "fillers/report.json").read_bytes())
        except (OSError, ValueError):
            result = {}
    else:
        result = fillers_stage.fillers(root, propose_only=True)
        counts = result["counts"]["status"]
    # Same notes as the fillers command, so a skipped speech check is never silent.
    state["warnings"].extend(
        f"{note} ({result['skipped']})" if note == "fillers skipped: no speech engine" else note
        for note in result.get("notes", [])
    )
    if counts.get("proposed", 0):
        state["warnings"].append(
            f"fillers: {counts['proposed']} removals proposed and not applied; "
            "the final keeps the reviewed cut's words. Apply one only after the creator agrees: "
            f"yap-studio fillers set {shlex.quote(str(root))} ID applied"
        )
    if kept:
        return True, f"kept the cut's removals: {counts.get('applied', 0)} applied"
    return result["noop"], (f"{counts.get('applied', 0)} applied, "
                            f"{counts.get('proposed', 0)} proposed, "
                            f"{counts.get('rejected', 0)} rejected")


def _cut(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
    result = cut_stage.cut(root)
    if result.get("status") == "failed":
        failed = failed_assert_lines(result.get("asserts", []))
        finding = None
        try:
            report = json.loads((root / "cut/verify.json").read_bytes())
            finding = next((item for item in report.get("findings", [])
                            if item.get("severity") == "fail"), None)
        except (OSError, ValueError):
            pass
        message = "Cut failed its checks: " + ", ".join(failed)
        if finding is not None:
            message += f". {finding.get('message', '')}".rstrip()
        raise RunStop(
            message, code="cut_failed",
            suggested_action=(finding or {}).get("suggested_action")
            or "Read cut/verify.json, correct edit.json, then run again.",
        )
    state["warnings"].extend(result.get("warnings", []))
    if result["noop"]:
        # A reused cut returns its plan warnings; its verify warnings live in cut/verify.json.
        try:
            report = json.loads((root / "cut/verify.json").read_bytes())
        except (OSError, ValueError):
            report = {}
        state["warnings"].extend(item["message"] for item in report.get("findings", [])
                                 if item.get("severity") == "warn")
    return result["noop"], result.get("out", "cut/cut.mp4")


def _music(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
    result = music_stage.music(root)
    state["notices"].extend(result.get("notices", []))
    return result["noop"], "music bed" if result.get("bed") else "voice only"


def _captions(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
    result = captions_stage.captions(root)
    return result["noop"], f"{result.get('phrases', 0)} phrases"


def _graphics(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
    result = graphics_stage.graphics(root)
    state["warnings"].extend(result.get("warnings", []))
    return result["noop"], f"{len(result.get('elements', []))} layers"


def _assemble(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
    result = assemble_stage.assemble(root, allow_violations=state["allow_violations"],
                                     frames=state["frames"])
    state["notices"].extend(result.get("notices", []))
    state["pilot" if state["frames"] is not None else "final"] = result["out"]
    return result["noop"], result["out"]


def _verify(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
    """Confirm the published final and check it against the project's platform rules."""
    final = read_receipt(root / "final") or {}
    if final.get("status") != "verified" or not (root / state["final"]).is_file():
        raise RunStop(f"{state['final']} has no verified receipt.", code="final_unverified",
                      suggested_action="Run yap-studio run again to assemble the final.")
    state["warnings"].extend(item["message"]
                             for item in final.get("verify", {}).get("findings", [])
                             if item.get("severity") == "warn")
    platform = ProjectContext(root).project.platform
    if platform == "none":
        return True, "final verified"
    findings = check_platform(ffmpeg.probe(root / state["final"]), load_presets()[platform],
                              name=platform)
    state["warnings"].extend(item.message for item in findings if item.severity == "warn")
    failed = [item for item in findings if item.severity == "fail"]
    if failed:
        raise RunStop(" ".join(item.message for item in failed), code="platform_failed",
                      suggested_action=failed[0].suggested_action
                      or f"Change the edit to meet the {platform} rules, then run again.")
    return True, f"final verified, {platform} rules pass"


def _words(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
    result = check_kept_words(root, root / state["final"])
    state["words"] = result
    if result["status"] == "skipped" and result["baseline"] is not None:
        state["warnings"].append(f"words: {result['reason']}")
    if result["status"] == "fail":
        dropped = [item for item in result["dropped"]
                   if item["why"] in {"unapproved", "unexplained"}]
        names = []
        for item in dropped:
            minutes, ticks = divmod(round(item["time"] * 10), 600)
            clock = f"{minutes}:{ticks / 10:04.1f}"
            removal = f" ({item['removal']})" if item["removal"] else ""
            names.append(f'{json.dumps(item["text"], ensure_ascii=False)} at {clock}{removal}')
        quoted = shlex.quote(str(root))
        commands = list(dict.fromkeys(
            f"yap-studio fillers set {quoted} {item['removal']} rejected"
            for item in dropped if item["removal"]
        ))
        fix = ("Put each word back with " + "; ".join(commands) + ", or restore the story. "
               if commands else "Restore the story to put each word back. ")
        noun = "word" if len(dropped) == 1 else "words"
        raise RunStop(
            f"The final drops {len(dropped)} {noun} from {result['baseline']} that nobody "
            f"approved: {', '.join(names)}.", code="words_dropped",
            suggested_action=(fix + f"Then run yap-studio run {quoted} again. "
                              "Apply a removal only after the creator agrees."),
        )
    return True, words_summary(result)


def layer_windows(root: Path) -> dict[str, Any]:
    """Describe the final's caption phrases and timeline elements on the output clock."""
    from yap_studio.feedback.versions import _LAYER_KINDS, LAYERS_SCHEMA
    from yap_studio.stages.timeline import resolve_timeline

    layers = []
    try:
        phrases = json.loads((root / "captions/phrases.json").read_bytes())["phrases"]
    except FileNotFoundError:
        phrases = []
    # Empty windows do not claim a word occurrence.
    keys = phrase_keys([
        phrase.get("ids") or [] if phrase["end"] > phrase["start"] else []
        for phrase in phrases
    ])
    for key, phrase in zip(keys, phrases, strict=True):
        if phrase["end"] <= phrase["start"]:
            continue
        layers.append({"id": key, "kind": "caption",
                       "text": phrase["text"], "start": phrase["start"], "end": phrase["end"]})
    for item in resolve_timeline(root).elements:
        if item.kind in _LAYER_KINDS and item.end_frame > item.start_frame:
            text = item.payload.get("text")
            layers.append({"id": item.id, "kind": item.kind,
                           "text": text if isinstance(text, str) else None,
                           "start": item.start_s, "end": item.end_s})
    return {"schema": LAYERS_SCHEMA, "time_base": "output", "layers": layers}


def _version(root: Path, state: dict[str, Any]) -> tuple[bool, str]:
    """Save the verified final and its layer windows unless it is already the newest version."""
    from yap_studio.feedback.versions import load_versions, save_version

    final = root / state["final"]
    previous = load_versions(root)
    if previous and previous[-1].media["sha256"] == receipt.sha256_file(final):
        state["version"] = previous[-1].id
        return True, previous[-1].id
    work_root = paths.validate_output_dir(root / ".yap/work")
    with tempfile.TemporaryDirectory(prefix="run-", dir=work_root) as folder:
        layers = Path(folder) / "layers.json"
        layers.write_text(json.dumps(layer_windows(root), indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n", encoding="utf-8")
        manifest = save_version(root, media=final, note=state["note"], layers=layers)
    state["version"] = manifest.id
    return False, manifest.id


_HANDLERS: dict[str, Callable[[Path, dict[str, Any]], tuple[bool, str]]] = {
    "ingest": _ingest, "fillers": _fillers, "cut": _cut, "music": _music,
    "captions": _captions, "graphics": _graphics, "assemble": _assemble,
    "verify": _verify, "words": _words, "version": _version,
}


def run(
    project_dir: Path,
    *,
    allow_violations: bool = False,
    note: str | None = None,
    frames: tuple[int, int] | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run each stage in order; finished stages are no-ops, so a rerun resumes at the stop."""
    root = Path(project_dir).resolve()
    if not (root / "project.json").is_file():
        raise InputError(str(project_dir), "no project.json here; run yap-studio init first")
    state: dict[str, Any] = {"allow_violations": allow_violations, "note": note,
                             "frames": frames, "pilot": None,
                             "final": None, "version": None, "words": None,
                             "warnings": [], "notices": []}
    steps: list[dict[str, Any]] = []
    stopped = None
    for name in STEPS:
        if frames is not None and name in {"verify", "words", "version"}:
            step = {"name": name, "state": "skipped", "detail": "skipped: pilot"}
            steps.append(step)
            if on_step is not None:
                on_step(step)
            continue
        try:
            with paths.operation():
                unchanged, detail = _HANDLERS[name](root, state)
        except (YapError, OSError, ValueError) as exc:
            if not isinstance(exc, YapError):
                exc = InputError(str(project_dir), f"{name} cannot read or write: {exc}")
            state["notices"].extend(getattr(exc, "notices", []))
            stopped = {"step": name, **exc.to_dict(), "exit": exit_status(exc)}
            step = {"name": name, "state": "failed", "detail": exc.message}
            steps.append(step)
            if on_step is not None:
                on_step(step)
            break
        step = {"name": name, "state": "unchanged" if unchanged else "done", "detail": detail}
        steps.append(step)
        if on_step is not None:
            on_step(step)
    quoted = shlex.quote(str(project_dir))
    return {
        "schema": "yap-studio/run@1",
        "status": "stopped" if stopped else "finished",
        "steps": steps,
        "stopped": stopped,
        "final": state["final"] if stopped is None else None,
        "pilot": state["pilot"] if stopped is None else None,
        "version": state["version"],
        "words": state["words"],
        "warnings": state["warnings"],
        "notices": state["notices"],
        "next": (f"yap-studio run {quoted}" if stopped or frames is not None else
                 f"yap-studio review --versions {quoted}"),
    }
