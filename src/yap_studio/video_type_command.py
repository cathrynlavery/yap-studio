"""Show or record the video recipe and its allowed B-roll sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio import paths, receipt, video_type
from yap_studio.errors import OutputPathError, ValidationError, YapError
from yap_studio.project.lock import project_lock, write_json_cas
from yap_studio.project.model import Project, ProjectFileError, load_project, parse_document


def _choices(project: Project | dict | None) -> dict | None:
    if project is None:
        return None
    data = project.to_dict() if isinstance(project, Project) else project
    return {"video_type": data.get("video_type"), "broll_sources": data.get("broll_sources")}


def _check_stock_sources(root: Path) -> None:
    from yap_studio.stock.guard import stock_element_ids

    ids = []
    for lane in ("agent", "broll"):
        target = root / f"timeline/{lane}.json"
        try:
            data = json.loads(target.read_bytes())
        except FileNotFoundError:
            continue
        timeline = parse_document("timeline", data, filename=str(target), lane=lane)
        ids.extend(stock_element_ids(root, timeline.elements))
    if ids:
        raise YapError(
            "This video's timeline uses stock clips: " + ", ".join(ids) + ".",
            error_type="validation_error", code="stock_in_timeline",
            suggested_action="Remove them with yap-studio timeline remove DIR --lane LANE --id ID, "
                             "then set the sources again.",
        )


def _set(args: argparse.Namespace) -> dict:
    sources = video_type.parse_sources(args.broll) if args.broll is not None else None
    if args.directory is None:
        if args.type is None:
            raise ValidationError("type", "choose --type TYPE to save a default")
        if sources is None and args.type in video_type.USES_BROLL:
            raise ValidationError("broll", "choose --broll SOURCE to save a default")
        video_type.save_default(args.type, sources or [])
        return {"project": None, "default": _choices(video_type.load_default()),
                "default_saved": True}
    root = Path(args.directory).expanduser().resolve()
    with project_lock(root), video_type.defaults_lock():
        default = video_type.load_default()
        target = root / "project.json"
        digest = receipt.sha256_file(target)
        data = json.loads(target.read_bytes())
        project = parse_document("project", data)
        kind = args.type if args.type is not None else project.video_type
        if kind is None:
            raise ValidationError("type", "choose --type TYPE for this video")
        memo = all(take.voiceover for take in project.takes)
        if sources is None:
            sources = list(project.broll_sources) if project.broll_sources is not None else None
        _, sources = video_type.resolve(voiceover=memo, video_type=kind,
                                         broll_sources=sources, default=default)
        if args.default and sources is None:
            raise ValidationError("broll", "choose --broll SOURCE before saving the default")
        data["video_type"] = kind
        if sources is not None:
            data["broll_sources"] = sources
        parse_document("project", data)
        if sources is not None and "stock" not in sources:
            _check_stock_sources(root)
        default_path = video_type.defaults_path()
        old_default = default_path.read_bytes() if default_path.exists() else None
        save = False
        if args.default or (default is None and sources is not None):
            save = video_type._save_default_unlocked(kind, sources,
                                                      only_if_missing=not args.default)
            default = ({"video_type": kind, "broll_sources": sources} if save
                       else video_type.load_default())
        try:
            write_json_cas(target, data, base_sha256=digest)
        except BaseException:
            if save:
                if old_default is None:
                    default_path.unlink()
                else:
                    with paths.atomic_output(default_path) as temporary:
                        temporary.write_bytes(old_default)
            raise
    return {"project": _choices(data), "default": _choices(default), "default_saved": save}


def _run(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            if args.video_type_command == "set":
                result = _set(args)
            else:
                project = (load_project(Path(args.directory).expanduser() / "project.json")
                           if args.directory is not None else None)
                result = {"schema": video_type.QUESTIONS_SCHEMA,
                          "default": _choices(video_type.load_default()),
                          "project": _choices(project), "questions": video_type.QUESTIONS}
    except (OSError, ValueError, YapError) as error:
        exc = error if isinstance(error, YapError) else ProjectFileError(
            "project.json", "$", "cannot read project; restore valid JSON and retry")
        if not exc.suggested_action:
            exc.suggested_action = "Use a project folder with a writable project.json and retry."
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}\nfix: {exc.suggested_action}", file=sys.stderr)
        return 3 if isinstance(exc, OutputPathError) else 2
    if args.json:
        print(json.dumps(result, allow_nan=False))
    else:
        default = result["default"]
        print("default: " + (video_type.describe(**default) if default else
                              'none saved; the agent asks "What are you making?"'))
        if result["project"] is not None:
            print("this video: " + video_type.describe(**result["project"]))
        if args.video_type_command == "set":
            print("Default saved." if result["default_saved"] else "Default unchanged.")
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register video-type show and set."""
    parser = subparsers.add_parser("video-type", help="show or choose the video type and B-roll")
    commands = parser.add_subparsers(dest="video_type_command", required=True)
    for name in ("show", "set"):
        command = commands.add_parser(name)
        command.add_argument("directory", nargs="?", metavar="DIR")
        command.add_argument("--json", action="store_true")
        if name == "set":
            command.add_argument("--type", choices=video_type.VIDEO_TYPES)
            command.add_argument("--broll", action="append",
                                 choices=(*video_type.BROLL_SOURCES, "none"))
            command.add_argument("--default", action="store_true")
        command.set_defaults(func=_run)
