"""CLI for standalone word-anchored graphics and B-roll planning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio import paths
from yap_studio.canvas import project_canvas
from yap_studio.errors import OutputPathError, ValidationError, YapError
from yap_studio.faces.model import load_faces
from yap_studio.faces.project import picture_faces
from yap_studio.media.catalog import catalog_path, load_catalog
from yap_studio.pictures import guard as pictures_guard
from yap_studio.planner.broll import load_requests as load_broll_requests
from yap_studio.planner.broll import plan_broll, suggest_broll, write_broll_plan
from yap_studio.planner.cards import load_requests, plan_cards, render_layers, write_plan
from yap_studio.project.model import load_cut_words
from yap_studio.project.status import ProjectContext
from yap_studio.receipt import sha256_file
from yap_studio.stock.guard import filter_approved, validate_requests
from yap_studio.style import resolve
from yap_studio.taste import apply_taste


def _time(seconds: float) -> str:
    hundredths = round(seconds * 100)
    minutes, remainder = divmod(hundredths, 6000)
    return f"{minutes}:{remainder / 100:05.2f}"


def _plan(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            requests = load_requests(Path(args.requests).expanduser())
            words_path = paths.validate_input(Path(args.cut_words).expanduser()).resolve()
            words = load_cut_words(words_path)
            style = resolve(args.style)
            faces = (
                load_faces(paths.validate_input(Path(args.faces).expanduser()))
                if args.faces
                else None
            )
            project = Path(args.project).expanduser().resolve() if args.project else (
                words_path.parent.parent
                if words_path.parent.name == "cut" and words_path.name == "cut-words.json" else None
            )
            timeline = None
            caption_layout = None
            if (project is not None and (project / "project.json").is_file()
                    and (args.project or (project / "cut/cut.mp4").is_file())):
                context = ProjectContext(project)
                timeline = context.resolved_timeline()
                project_style = context.render_style()
                if args.style is not None and style.sha256() != project_style.sha256():
                    raise ValidationError(
                        "style", "must match the project's resolved render style; "
                        "omit --style to use that style",
                    )
                style = project_style
                if timeline.total_frames != args.frames or args.fps != 30:
                    raise ValidationError("frames", "must match the project picture clock")
                if faces is not None and faces.video.sha256 != sha256_file(project / "cut/cut.mp4"):
                    raise ValidationError("faces", "must describe this project's cut/cut.mp4")
                # The assembly cache fixes the detector and sampling clock too.
                # An independently sampled --faces file can have smaller boxes
                # than assembly's interpolation envelope between samples.
                faces = picture_faces(project, timeline, context) if requests else None
                try:
                    caption_layout = context.json("captions/layout.json")
                except FileNotFoundError:
                    pass
            elif args.project:
                raise ValidationError("project", "project.json not found")
            plan = plan_cards(requests, words, style, frames=args.frames, fps=args.fps,
                              faces=faces, timeline=timeline, project_dir=project,
                              caption_layout=caption_layout)
            out = Path(args.out).expanduser()
            outputs = write_plan(plan, out)
            if args.render:
                render_layers(plan, style, out)
            data = json.loads(outputs["graphics-plan.json"].read_bytes())
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if isinstance(exc, OutputPathError) else 2
    if args.json:
        print(json.dumps(data, allow_nan=False))
    else:
        for card in plan.cards:
            clearance = card.clearance
            status = clearance["status"]
            if status == "clear":
                detail = "clear of the head"
                if clearance["gap_px"] is not None:
                    detail += f" (gap {clearance['gap_px']:.0f} px)"
            elif status == "behind_head":
                detail = "behind the head; person keyed back on top"
            else:
                detail = f"{status}: {clearance['reason']}"
            print(
                f"{card.id} {card.template} {card.placement} {_time(card.start_s)} "
                f"to {_time(card.end_s)} (frames {card.start_frame} to {card.end_frame}) "
                f"at y {card.y}: {detail}"
            )
        for warning in plan.warnings:
            print(f"warning: {warning}")
        blocked = sum(card.clearance["status"] == "blocked" for card in plan.cards)
        if blocked:
            noun = "card" if blocked == 1 else "cards"
            print(f"Plan: needs review ({blocked} {noun} could not clear the head)")
        else:
            print("Plan: ok")
        missing = sum(card["layer"]["status"] == "missing" for card in data["cards"])
        if missing:
            print(f"Layers missing for {missing} card(s): run again with --render")
        for output in outputs.values():
            print(f"Wrote {output}")
    return int(plan.status == "needs_review")


def add_cards_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the cards plan command, stopping at the timeline contract seam."""
    parser = subparsers.add_parser("cards", help="plan cards at the words they name")
    commands = parser.add_subparsers(dest="cards_command", required=True)
    planning = commands.add_parser("plan", help="write a graphics lane and card placement plan")
    planning.add_argument("requests", metavar="REQUESTS.json")
    planning.add_argument("--cut-words", required=True, metavar="CUT-WORDS.json")
    planning.add_argument("--frames", required=True, type=int, metavar="N")
    planning.add_argument("--out", required=True, metavar="DIR")
    planning.add_argument("--style", metavar="SPEC")
    planning.add_argument("--faces", metavar="FACES.json",
                          help="standalone faces; project plans use assembly's cut face cache")
    planning.add_argument("--project", metavar="DIR",
                          help="map cut faces through project zooms and picture replacements")
    planning.add_argument("--fps", type=int, default=30)
    planning.add_argument("--render", action="store_true")
    planning.add_argument("--json", action="store_true")
    planning.set_defaults(func=_plan)


def _broll(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            words_path = paths.validate_input(Path(args.cut_words).expanduser()).resolve()
            words = load_cut_words(words_path)
            style = apply_taste(resolve(args.style)).style.data
            paths.protect_input(catalog_path())
            items = load_catalog()["items"]
            project = (words_path.parent.parent if words_path.name == "cut-words.json"
                       and words_path.parent.name == "cut" else None)
            if args.broll_command == "suggest":
                items, excluded = filter_approved(items, style, project)
                items, pictures_excluded = pictures_guard.filter_approved(items, style, project)
                excluded.extend(pictures_excluded)
                frame = project_canvas(project) if project is not None else None
                shape = ("horizontal" if frame is not None and frame.width > frame.height
                         else "vertical")
                data = suggest_broll(words, items, style, limit=args.limit, shape=shape)
                data["excluded"].extend(excluded)
                outputs = {}
                plan = None
            else:
                if args.project:
                    project_dir = Path(args.project).expanduser().resolve()
                elif words_path.name == "cut-words.json" and words_path.parent.name == "cut":
                    project_dir = words_path.parent.parent
                else:
                    raise ValidationError(
                        "project", "use --project DIR when cut words are outside <project>/cut"
                    )
                requests = load_broll_requests(Path(args.requests).expanduser())
                validate_requests(requests, items, style, project_dir)
                pictures_guard.validate_requests(requests, items, style, project_dir)
                plan = plan_broll(requests, words, items, style, project_dir=project_dir,
                                  frames=args.frames, fps=args.fps)
                outputs = write_broll_plan(plan, Path(args.out).expanduser())
                data = json.loads(outputs["broll-plan.json"].read_bytes())
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if isinstance(exc, OutputPathError) else 2
    if args.json:
        print(json.dumps(data, allow_nan=False))
    elif plan is None:
        for phrase in data["phrases"]:
            print(f"{phrase['start']['word']} to {phrase['end']['word']}: {phrase['text']}")
            for candidate in phrase["candidates"]:
                print(f"  {candidate['item']} {candidate['kind']} {candidate['placement']} "
                      f"score {candidate['score']}: " + "; ".join(candidate["reasons"]))
        for entry in data["excluded"]:
            print(f"Skipped {entry['item']}: " + "; ".join(entry["reasons"]))
        for phrase in data["phrases"]:
            for entry in phrase["skipped"]:
                print(f"Skipped {entry['item']}: " + "; ".join(entry["reasons"]))
    else:
        for entry in plan.broll:
            print(f"{entry['id']} {entry['item']} {entry['element']['kind']} "
                  f"{_time(entry['start_s'])} to {_time(entry['end_s'])} "
                  f"(frames {entry['start_frame']} to {entry['end_frame']})")
        for warning in plan.warnings:
            print(f"warning: {warning}")
        print("Plan: needs review" if plan.status == "needs_review" else "Plan: ok")
        for output in outputs.values():
            print(f"Wrote {output}")
    return int(plan is not None and plan.status == "needs_review")


def add_broll_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register read-only suggestions and request-only lane planning."""
    parser = subparsers.add_parser("broll", help="suggest and plan supporting catalog media")
    commands = parser.add_subparsers(dest="broll_command", required=True)
    for name in ("suggest", "plan"):
        command = commands.add_parser(name)
        command.add_argument("--cut-words", required=True, metavar="CUT-WORDS.json")
        command.add_argument("--style", metavar="SPEC")
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_broll)
        if name == "suggest":
            command.add_argument("--limit", type=int, default=3, metavar="N")
        else:
            command.add_argument("requests", metavar="REQUESTS.json")
            command.add_argument("--project", metavar="DIR")
            command.add_argument("--frames", required=True, type=int, metavar="N")
            command.add_argument("--out", required=True, metavar="DIR")
            command.add_argument("--fps", type=int, default=30)
