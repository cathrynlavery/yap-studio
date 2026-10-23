"""Export verified project layers for inspection without importing Studio edits."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import tempfile
from collections.abc import Callable
from fractions import Fraction
from html import escape
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from yap_studio import cards, paths, process
from yap_studio.errors import InputError, OutputPathError, YapError
from yap_studio.plan import snap_frame
from yap_studio.project.model import CutWords, Edit, parse_document
from yap_studio.project.receipts import _identity, projection_sha256, receipt_output_paths
from yap_studio.project.status import ProjectContext, cut_inputs, ingest_inputs
from yap_studio.render import hyperframes
from yap_studio.stages.captions import caption_snapshot, prepare_layer_plan
from yap_studio.stages.graphics import graphics_snapshot
from yap_studio.stages.timeline import FPS

_PURPOSE = (
    "HyperFrames Studio is for agent and debugging inspection, not editing. "
    "The creator comments in the review page and the agent edits the project. "
    "Nothing is read back from Studio into the project."
)


class StudioError(YapError):
    """Report an unusable export or a failed Studio command."""

    def __init__(self, message: str, *, code: str = "studio_export_failed") -> None:
        super().__init__(message, error_type="studio_error", code=code,
                         suggested_action="Inspect the named stage or Studio command and retry.")


def _run(args: list[str], install: hyperframes.RendererInstall | None = None) -> Any:
    renderer = install or hyperframes.ensure_installed()
    return process.spawn([str(renderer.executable), *args], timeout_s=120,
                         tool="hyperframes", env=hyperframes.renderer_env())


def _json_result(result: Any, command: str) -> dict[str, Any]:
    if result.returncode:
        raise StudioError(
            f"HyperFrames {command} failed ({result.returncode}): "
            f"{result.stdout.strip()} {result.stderr.strip()}", code=f"studio_{command}_failed",
        )
    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("expected an object")
        return payload
    except ValueError as exc:
        raise StudioError(f"HyperFrames {command} did not return a JSON object.") from exc


def _verified(
    context: ProjectContext, stage: str, inputs: Callable[[], dict[str, str]],
) -> tuple[dict[str, Any] | None, str]:
    """Check current receipts, allowing the shared identity cache's bookkeeping."""
    try:
        data = context.json(f"{stage}/receipt.json")
    except FileNotFoundError:
        return None, "no receipt"
    except (OSError, ValueError) as exc:
        return None, f"cannot read receipt: {exc}"
    if (not isinstance(data, dict) or data.get("schema") != "yap-studio/stage-receipt@1"
            or data.get("status") != "verified"):
        return None, "receipt is not verified"
    if any((context.project_dir / stage / name).exists()
           for name in ("receipt.pending.json", "progress.json")):
        return None, "stage has an unfinished run"
    try:
        if data.get("inputs") != inputs():
            return None, "inputs changed since the verified receipt"
        targets = receipt_output_paths(context.project_dir, data)
        for output, target in zip(data["outputs"], targets, strict=True):
            if not target.is_file() or context.identity.sha256(target) != output["sha256"]:
                return None, f"missing or changed output: {output['path']}"
    except (OSError, ValueError, KeyError, TypeError, YapError) as exc:
        return None, f"cannot verify current inputs or outputs: {exc}"
    return data, "inputs and outputs match the verified receipt"


def _segments(context: ProjectContext, record: dict[str, Any]) -> list[dict[str, Any]]:
    """Group retained cut spans by story segment, retaining removal-created ranges."""
    spans = context.json("cut/spans.json")
    if Fraction(spans["fps"]) != FPS:
        raise StudioError("The verified cut must use 30 fps.")
    edit = cast(Edit, parse_document("edit", context.json("edit.json")))
    words = cast(CutWords, parse_document("cut-words", context.json("cut/cut-words.json")))
    pieces = record["plan"].get("pieces", [])
    rows = spans["segments"]
    if pieces and len(pieces) != len(rows):
        raise StudioError("The cut frame plan and spans disagree; run cut again.")
    result: list[dict[str, Any]] = []
    cursor = 0
    for index, span in enumerate(rows):
        first, last = Fraction(span["start"]), Fraction(span["end"])
        frames = snap_frame(last, FPS) - snap_frame(first, FPS)
        piece = pieces[index] if pieces else None
        if piece:
            identifier = piece["segment_id"]
            if (piece["out_f0"] != cursor or piece["n_frames"] != frames
                    or piece["take"] != span["source"]
                    or Fraction(piece["src_t0"]) != first or Fraction(piece["src_t1"]) != last):
                raise StudioError("The cut frame plan and spans disagree; run cut again.")
        elif len(rows) == len(edit.segments):
            identifier = edit.segments[index].id
        else:
            ids = {word.segment for word in words.words
                   if cursor / 30 <= word.start < (cursor + frames) / 30}
            if len(ids) != 1:
                raise StudioError("The cut needs its piece frame plan to map wordless spans.")
            identifier = ids.pop()
        authored = next((item for item in edit.segments if item.id == identifier), None)
        if authored is None or authored.take != span["source"] or frames <= 0:
            raise StudioError("The cut contains an invalid story span; run cut again.")
        source_range = {"in": str(first), "out": str(last)}
        if result and result[-1]["id"] == identifier:
            result[-1]["frames"] += frames
            result[-1]["source_out"] = str(last)
            result[-1]["source_spans"].append(source_range)
        else:
            result.append({"index": edit.segments.index(authored), "id": identifier,
                           "take": authored.take, "source_in": str(first),
                           "source_out": str(last), "start_frame": cursor, "frames": frames,
                           "source_spans": [source_range]})
        cursor += frames
    if not result or cursor != record["plan"]["total_frames"]:
        raise StudioError("The cut spans do not cover the verified frame plan; run cut again.")
    return result


def _attrs(values: dict[str, Any]) -> str:
    return " ".join(f'{key}="{escape(str(value), quote=True)}"' for key, value in values.items())


def _provenance(segment: dict[str, Any]) -> dict[str, Any]:
    return {"data-yap-segment-index": segment["index"], "data-yap-segment-id": segment["id"],
            "data-yap-take-id": segment["take"], "data-yap-source-in": segment["source_in"],
            "data-yap-source-out": segment["source_out"],
            "data-yap-source-spans": json.dumps(segment["source_spans"], allow_nan=False)}


def _host(
    identifier: str, kind: str, placement: dict[str, Any], variables: dict[str, Any],
    track: int, segments: list[dict[str, Any]],
) -> str:
    covered = [item for item in segments if item["start_frame"] <
               placement["start_frame"] + placement["frames"] and
               item["start_frame"] + item["frames"] > placement["start_frame"]]
    if not covered:
        raise StudioError(f"{identifier} has no placement within the verified cut.")
    values = {"id": identifier, "class": "clip", "data-composition-id": identifier,
              "data-composition-src": f"layers/{identifier}/index.html",
              "data-start": placement["start_frame"] / 30,
              "data-duration": placement["frames"] / 30,
              "data-width": placement["width"], "data-height": placement["height"],
              "data-track-index": track,
              "data-track-kind": "captions" if kind == "captions" else "graphics",
              "data-yap-kind": kind,
              "data-yap-start-frame": placement["start_frame"],
              "data-yap-frames": placement["frames"],
              "data-variable-values": json.dumps(variables, ensure_ascii=False, allow_nan=False),
              **_provenance(covered[0]), "data-yap-segments": json.dumps(covered, allow_nan=False),
              "style": f"position:absolute;left:{placement['x']}px;top:{placement['y']}px;"
                       f"width:{placement['width']}px;height:{placement['height']}px;z-index:{track}"}
    return f"<div {_attrs(values)}></div>"


def _publish(work: Path, target: Path) -> None:
    """Replace the derived folder, restoring its predecessor if publication fails."""
    paths._guard_write(target)
    if not target.exists():
        os.replace(work, target)
        return
    if not target.is_dir():
        raise OutputPathError("Studio export path must be a directory.", code="unsafe_path")
    previous = work.with_name(work.name + ".previous")
    os.rename(target, previous)
    try:
        os.rename(work, target)
    except OSError:
        os.rename(previous, target)
        raise
    shutil.rmtree(previous)


def export_project(project_dir: Path) -> Path:
    """Rebuild a derived inspection export, leaving project outputs and receipts untouched."""
    with paths.operation():
        try:
            return _export_project(Path(project_dir))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StudioError(f"Cannot export the project: {exc}") from exc


def _export_project(project_dir: Path) -> Path:
    context = ProjectContext(project_dir)
    if context.project.shape == "horizontal":
        from yap_studio.canvas import unsupported

        unsupported("studio export", "Review final video playback on the review page")
    identities = {}
    for stage, input_fn in (("ingest", ingest_inputs), ("cut", cut_inputs)):
        record, reason = _verified(context, stage, lambda fn=input_fn: fn(context))
        if record is None:
            raise StudioError(f"The {stage} has no current verified receipt: {reason}.")
        identities.update({path: _identity(path)
                           for path in receipt_output_paths(context.project_dir, record)})
    identities.update({context.take_path(take.id): _identity(context.take_path(take.id))
                       for take in context.project.takes})
    assert record is not None
    owned = {item["path"] for item in record["outputs"]}
    if not {"cut/cut.mp4", "cut/spans.json", "cut/cut-words.json"} <= owned:
        raise StudioError("The cut receipt must own cut.mp4, spans.json and cut-words.json.")
    segments = _segments(context, record)
    frames = record["plan"]["total_frames"]
    edit = cast(Edit, parse_document("edit", context.json("edit.json")))
    hold_s = edit.end_hold_s
    if hold_s is None:
        hold_s = context.render_style().data["internal"]["end_hold_s"]
    hold = snap_frame(Fraction(str(hold_s)), FPS)
    stages = {"cut": {"included": True, "reason": "current verified receipt"}}
    install = hyperframes.ensure_installed()
    parent = paths.validate_output_dir(context.project_dir / ".yap")
    target = parent / "studio"
    paths._guard_write(target)
    with tempfile.TemporaryDirectory(prefix=".studio-", dir=parent) as folder:
        work = Path(folder)
        paths.validate_output_dir(work / "media")
        (work / "media/cut.mp4").symlink_to(context.project_dir / "cut/cut.mp4")
        clips = []
        for index, segment in enumerate(segments):
            clip_frames = segment["frames"] + (hold if index == len(segments) - 1 else 0)
            values = {"id": f"segment-{segment['index']}", "class": "clip", "src": "media/cut.mp4",
                      "data-start": segment["start_frame"] / 30,
                      "data-duration": float(Fraction(clip_frames, 30) - Fraction(1, 1000000)),
                      "data-media-start": segment["start_frame"] / 30, "data-track-index": 0,
                      "data-yap-end-hold-frames": hold if index == len(segments) - 1 else 0,
                      "data-has-audio": "true", "playsinline": "", "preload": "auto",
                      **_provenance(segment),
                      "style": "position:absolute;inset:0;width:1080px;height:1920px;"
                               "object-fit:cover;z-index:0"}
            clips.append(f"<video {_attrs(values)}></video>")
        track = 1
        for stage, snapshot_fn in (("captions", caption_snapshot), ("graphics", graphics_snapshot)):
            if (stage == "captions"
                    and context.render_style().data["captions"].get("camera_3d", False)):
                stages[stage] = {
                    "included": False,
                    "reason": ("3D camera captions are rendered video, "
                               "so Studio shows the cut without them"),
                }
                continue
            snapshot = None

            def inputs(fn: Any = snapshot_fn, name: str = stage) -> dict[str, str]:
                nonlocal snapshot
                snapshot = (fn(context, placements=context.json(f"{name}/receipt.json")["elements"])
                            if name == "graphics" else fn(context))
                return snapshot.inputs

            layer_record, reason = _verified(context, stage, inputs)
            stages[stage] = {"included": layer_record is not None, "reason": reason}
            if layer_record is None:
                continue
            identities.update({path: _identity(path)
                               for path in receipt_output_paths(context.project_dir, layer_record)})
            assert snapshot is not None
            if stage == "captions":
                plans = [("captions", "captions", snapshot.plan, snapshot.assets,
                          layer_record["layer"])]
            else:
                placements = {item["id"]: item for item in layer_record["elements"]}
                plans = [(item.id, "title", item.plan, snapshot.assets, placements[item.id])
                         for item in snapshot.titles]
                plans.extend((item.id, "graphic", item.plan, item.assets, placements[item.id])
                             for item in snapshot.graphics)
            for identifier, kind, plan, assets, placement in plans:
                layer = work / "layers" / identifier
                if kind == "graphic":
                    prepared = cards.prepare_card(plan, layer)
                    if (prepared / "index.html").read_text(encoding="utf-8").find(
                        hyperframes.GSAP_SCRIPT
                    ) >= 0:
                        with paths.atomic_output(prepared / hyperframes.GSAP_SCRIPT) as temporary:
                            shutil.copyfile(hyperframes.gsap_file(install), temporary)
                else:
                    prepared = prepare_layer_plan(plan, layer, assets=assets)
                # Preparation uses a template subfolder; mounts use a stable per-element folder.
                for child in prepared.iterdir():
                    os.replace(child, layer / child.name)
                prepared.rmdir()
                clips.append(_host(identifier, kind, placement, plan.variables, track, segments))
                track += 1
        root_attrs = {"id": "root", "data-composition-id": "root", "data-no-timeline": "",
                      "data-width": 1080, "data-height": 1920, "data-fps": 30,
                      "data-duration": (frames + hold) / 30,
                      "data-yap-segments": json.dumps(segments, allow_nan=False),
                      "style": "position:relative;width:1080px;height:1920px;overflow:hidden"}
        html = ('<!doctype html><html data-composition-variables="[]"><head>'
                '<meta charset="utf-8"><title>' + escape(context.project.name) + '</title>'
                '<style>html,body{margin:0;background:#000}</style></head><body>\n'
                f'<!-- {_PURPOSE} -->\n<div {_attrs(root_attrs)}>\n' + "\n".join(clips)
                + "\n</div></body></html>\n")
        paths.atomic_write_text(work / "index.html", html)
        lint_process = _run(["lint", "--json", str(work)], install)
        try:
            lint = json.loads(lint_process.stdout)
        except ValueError as exc:
            raise StudioError(
                f"HyperFrames lint returned invalid JSON: {lint_process.stdout}"
            ) from exc
        if (not isinstance(lint, dict) or lint_process.returncode or lint.get("ok") is not True
                or lint.get("errorCount", 0)
                or any(item.get("severity") == "error" for item in lint.get("findings", []))):
            raise StudioError(f"HyperFrames lint failed ({lint_process.returncode}): "
                              f"{lint_process.stdout.strip()} {lint_process.stderr.strip()}",
                              code="studio_lint_failed")
        lanes = {}
        for lane in ("agent", "broll"):
            try:
                lanes[lane] = hashlib.sha256(context.read(f"timeline/{lane}.json")).hexdigest()
            except FileNotFoundError:
                lanes[lane] = None
        summary = {"schema": "yap-studio/studio-export@1", "project_name": context.project.name,
                   "timeline_hash": projection_sha256({"segments": segments, "lanes": lanes,
                                                       "frames": frames, "end_hold_frames": hold}),
                   "fps": 30, "content_frames": frames, "end_hold_frames": hold,
                   "stages": stages, "lint": lint}
        paths.atomic_write_text(work / "yap-export.json",
                                json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
                                + "\n")
        for relative, data in context._snapshots.items():
            candidate = context.project_dir / relative
            current = candidate.read_bytes() if candidate.exists() else None
            if current != data:
                raise StudioError(f"Project input changed during export: {relative}; retry.")
        for candidate, before in identities.items():
            if not candidate.is_file() or _identity(candidate) != before:
                raise StudioError(f"Project media changed during export: {candidate.name}; retry.")
        _publish(work, target)
    return target


def _existing_export(project_dir: Path) -> Path:
    target = Path(project_dir).resolve() / ".yap/studio"
    if not (target / "index.html").is_file() or not (target / "yap-export.json").is_file():
        raise InputError(str(target), "no Studio export yet; run yap-studio studio export DIR")
    return target


def open_project(
    project_dir: Path, *, port: int | None = None, open_browser: bool = True,
) -> dict[str, Any]:
    """Export and start a background Studio inspection session."""
    if port is not None and (isinstance(port, bool) or not isinstance(port, int)
                             or not 1 <= port <= 65535):
        raise StudioError("Studio port must be an integer between 1 and 65535.",
                          code="studio_invalid_port")
    if port is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
    target = export_project(project_dir)
    args = ["preview", "--background", "--port", str(port)]
    if not open_browser:
        args.append("--no-open")
    result = _run([*args, str(target)])
    if result.returncode:
        raise StudioError(f"HyperFrames preview failed ({result.returncode}): "
                          f"{result.stdout.strip()} {result.stderr.strip()}")
    stdout = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.stdout)
    urls = re.findall(r"https?://[^\s\x1b<>\"']+", stdout)
    url = next((value for value in urls if urlsplit(value).fragment.startswith("project/")
                and len(urlsplit(value).fragment) > len("project/")), None)
    if url is None:
        raise StudioError("HyperFrames preview did not print a Studio URL with #project/<name>.",
                          code="studio_url_missing")
    return {"url": url, "port": urlsplit(url).port or port, "export": str(target)}


def stop_project(project_dir: Path) -> dict[str, Any]:
    """Stop the background preview for this derived export."""
    target = _existing_export(project_dir)
    result = _run(["preview", "--stop", str(target)])
    if result.returncode:
        raise StudioError(f"HyperFrames preview stop failed ({result.returncode}): "
                          f"{result.stdout.strip()} {result.stderr.strip()}")
    return {"stopped": True, "export": str(target)}


def catch_up(project_dir: Path) -> dict[str, Any]:
    """Read Studio session JSON without applying any session changes to the project."""
    target = _existing_export(project_dir)
    return _json_result(_run(["catch-up", "--json", str(target)]), "catch-up")
