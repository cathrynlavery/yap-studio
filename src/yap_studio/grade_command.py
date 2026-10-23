"""Show or atomically change a project's effective picture grade."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio import paths, receipt
from yap_studio.errors import OutputPathError, ValidationError, YapError
from yap_studio.grade import (
    EYEDROPPER_DEFAULT_STRENGTH,
    GradeError,
    color_info,
    resolve_grade,
    sample_color,
)
from yap_studio.project.lock import project_lock, write_json_cas
from yap_studio.project.model import ProjectFileError, load_project, parse_document
from yap_studio.project.status import ProjectContext


def _selection(args: argparse.Namespace) -> dict:
    point = {"time_s": args.time, "x": args.x, "y": args.y}
    has_point = any(value is not None for value in point.values())
    if args.clear:
        if args.strength is not None or args.color is not None or has_point:
            raise ValidationError("grade", "use --clear without a colour, point or strength")
        return {}
    if args.look != "eyedropper" and (args.color is not None or has_point):
        raise ValidationError("grade", "choose --look eyedropper to use a colour or point")
    if args.look == "eyedropper" and (
        (args.color is not None) == has_point
        or (has_point and any(value is None for value in point.values()))
    ):
        raise GradeError("Choose --color HEX or all of --time T --x X --y Y.",
                         "Use exactly one hex colour or a complete sample point.")
    strength = args.strength
    if strength is None:
        strength = EYEDROPPER_DEFAULT_STRENGTH if args.look == "eyedropper" else 1
    return {"look": args.look, "strength": strength,
            **({"color": args.color} if args.color is not None else {}),
            **({"sample": point} if has_point else {})}


def _run(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            root = Path(args.directory).expanduser().resolve()
            load_project(root / "project.json")
            cut = root / "cut/cut.mp4"
            if args.grade_command == "sample":
                info = color_info(sample_color(cut, time_s=args.time, x=args.x, y=args.y))
                if args.json:
                    print(json.dumps(info, allow_nan=False))
                else:
                    enough = "yes" if info["colorful"] else "no; pick a more colourful point"
                    print(f"Color: {info['color']}, Oklab chroma {info['chroma']:.4f}, "
                          f"colourful enough: {enough}.")
                return 0
            if args.grade_command == "set":
                selected = _selection(args)
                with project_lock(root):
                    target = root / "project.json"
                    digest = receipt.sha256_file(target)
                    data = json.loads(target.read_bytes())
                    if args.clear:
                        data.pop("grade", None)
                    else:
                        spec = resolve_grade({}, selected, cut=cut)
                        data["grade"] = {**selected,
                                         **({"color": spec.color} if "color" in selected else {})}
                    parse_document("project", data)
                    write_json_cas(target, data, base_sha256=digest)
            context = ProjectContext(root)
            spec = resolve_grade(context.render_style().data, context.project.grade, cut=cut)
            sample = (context.project.grade or {}).get("sample")
            result = {**spec.to_dict(), **({"sample": sample} if sample is not None else {})}
    except (OSError, ValueError) as error:
        exc = error if isinstance(error, YapError) else ProjectFileError(
            "project.json", "$", "cannot read project or style; restore valid JSON and retry")
        return _error(args, exc)
    except YapError as exc:
        return _error(args, exc)
    if args.json:
        print(json.dumps(result, allow_nan=False))
    else:
        print(f"Grade: {spec.look}, strength {spec.strength:g}, from {spec.source}.")
        if spec.color is not None:
            print(f"Color: {spec.color}.")
        if sample is not None:
            print(f"Sample: time {sample['time_s']:g}s, x {sample['x']:g}, y {sample['y']:g}.")
    return 0


def _error(args: argparse.Namespace, exc: YapError) -> int:
    if not exc.suggested_action:
        exc.suggested_action = "Use a project folder with a writable project.json and retry."
    if args.json:
        print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
    else:
        print(f"error: {exc.message}\nfix: {exc.suggested_action}", file=sys.stderr)
    return 3 if isinstance(exc, OutputPathError) else 2


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register project grade show, set and clear commands."""
    parser = subparsers.add_parser("grade", help="show or change the picture grade")
    commands = parser.add_subparsers(dest="grade_command", required=True)
    for name in ("show", "set", "sample"):
        command = commands.add_parser(name)
        command.add_argument("directory")
        command.add_argument("--json", action="store_true")
        if name == "set":
            selection = command.add_mutually_exclusive_group(required=True)
            selection.add_argument("--look", choices=("none", "film", "eyedropper"))
            selection.add_argument("--clear", action="store_true")
            command.add_argument("--strength", type=float)
            command.add_argument("--color")
        if name in {"set", "sample"}:
            for flag in ("time", "x", "y"):
                command.add_argument(f"--{flag}", type=float, required=name == "sample")
        command.set_defaults(func=_run)
