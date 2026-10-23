"""CLI for the picture cost sheet and approved generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio import paths
from yap_studio.errors import OutputPathError, ValidationError, YapError
from yap_studio.media.catalog import load_catalog
from yap_studio.pictures import guard
from yap_studio.pictures.make import make_pictures
from yap_studio.pictures.sheet import ask_sheet, money
from yap_studio.project.lock import project_lock
from yap_studio.project.model import load_cut_words, parse_document
from yap_studio.project.status import ProjectContext
from yap_studio.stock.metadata import timestamp
from yap_studio.style import resolve
from yap_studio.taste import apply_taste


def _question(args):
    root = Path(args.project).expanduser().resolve()
    if not (root / "project.json").is_file():
        raise ValidationError("project", "use a project folder with project.json")
    return {"schema": "yap-studio/imagery-question@1", "project": str(root),
            "answer": guard.imagery_answer(root), "question": guard.IMAGERY_QUESTION}


def _check_generated_timeline(root):
    ids = []
    for lane in ("agent", "broll"):
        target = root / f"timeline/{lane}.json"
        try:
            data = json.loads(target.read_bytes())
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            raise ValidationError(f"timeline/{lane}.json", "cannot read timeline") from exc
        timeline = parse_document("timeline", data, filename=str(target), lane=lane)
        ids.extend(guard.generated_element_ids(root, timeline.elements))
    if ids:
        raise YapError(
            "This video's timeline uses generated pictures: " + ", ".join(ids) + ".",
            error_type="validation_error", code="generated_in_timeline",
            suggested_action="Remove them with yap-studio timeline remove DIR --lane LANE --id ID "
                             "first, then record the answer again.",
        )


def _answer(args):
    root = Path(args.project).expanduser().resolve()
    if not (root / "project.json").is_file():
        raise ValidationError("project", "use a project folder with project.json")
    with project_lock(root):
        if args.answer == "no":
            _check_generated_timeline(root)
        data = {"schema": guard.IMAGERY_SCHEMA, "question": guard.IMAGERY_QUESTION["prompt"],
                "answer": args.answer, "answered_at": timestamp()}
        paths.atomic_write_text(root / "pictures/imagery.json", json.dumps(data) + "\n")
    return data


def _ask(args):
    source = paths.validate_input(Path(args.requests).expanduser())
    try:
        requests = json.loads(source.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValidationError("requests", "cannot read picture requests JSON") from exc
    words_path = paths.validate_input(Path(args.cut_words).expanduser()).resolve()
    if args.project:
        project_dir = Path(args.project).expanduser().resolve()
    elif words_path.name == "cut-words.json" and words_path.parent.name == "cut":
        project_dir = words_path.parent.parent
    else:
        raise ValidationError(
            "project", "use --project DIR when cut words are outside <project>/cut"
        )
    style = (
        ProjectContext(project_dir).render_style().data
        if args.style is None and (project_dir / "project.json").is_file()
        else apply_taste(resolve(args.style)).style.data
    )
    data = ask_sheet(
        requests,
        load_cut_words(words_path),
        load_catalog()["items"],
        style,
        project_dir=project_dir,
        cut_words_path=words_path,
    )
    if data["status"] == "ok":
        folder = project_dir / "pictures/sheets" / data["sheet"]
        data = {
            **data,
            "html": str((folder / "ask-sheet.html").resolve()),
            "json": str((folder / "ask-sheet.json").resolve()),
        }
    return data


def _print(data, command):
    if command == "question":
        print("not asked yet: " + data["question"]["prompt"] if data["answer"] is None
              else "this video: " + data["answer"])
    elif command == "answer":
        print(f"Recorded: {data['answer']} for this video.")
    elif data["status"] == "unavailable":
        print(data["message"])
        print("\n".join(data["setup"]))
    elif command == "ask":
        print(data["html"])
        for line in data["lines"]:
            print(f"{line['number']}. {line['text']}")
            print("  " + line["prompt"])
            print("  about " + money(line["estimate_usd"]))
        print("Total: about " + money(data["total_usd"]))
    elif data["status"] == "dry_run":
        for preview in data["previews"]:
            print(f"{preview['number']}: {preview['request']}")
    else:
        for key in ("made", "recovered"):
            for entry in data[key]:
                cost = (
                    money(entry["cost_usd"]) + " charged"
                    if entry["cost_source"] == "charged"
                    else "about " + money(entry["cost_usd"])
                )
                print(f"{entry['number']}: {key} as {entry['item']} ({cost})")
        for entry in data["cached"]:
            print(f"{entry['number']}: already made as {entry['item']}")
        for entry in data["failed"]:
            print(f"{entry['number']}: {entry['reason']}")
        print(f"Spent: {money(data['spent_usd'])} of {money(data['ceiling_usd'])}")


def _pictures(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            if args.pictures_command == "question":
                data = _question(args)
            elif args.pictures_command == "answer":
                data = _answer(args)
            elif args.pictures_command == "ask":
                data = _ask(args)
            else:
                data = make_pictures(
                    Path(args.project).expanduser(),
                    args.pick,
                    yes=args.yes,
                    max_cost=args.max_cost,
                    sheet=args.sheet,
                    dry_run=args.dry_run,
                    new=args.new,
                )
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
        _print(data, args.pictures_command)
    return int(bool(data.get("failed")))


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("pictures", help="ask before making paid generated pictures")
    commands = parser.add_subparsers(dest="pictures_command", required=True)
    question = commands.add_parser("question", help="show this video's imagery question")
    question.add_argument("project", metavar="DIR")
    answer = commands.add_parser("answer", help="record this video's imagery answer")
    answer.add_argument("project", metavar="DIR")
    answer.add_argument("answer", choices=("yes", "no"))
    ask = commands.add_parser("ask", help="write a prompt and cost sheet")
    ask.add_argument("requests", metavar="REQUESTS.json")
    ask.add_argument("--cut-words", required=True, metavar="FILE")
    ask.add_argument("--style", metavar="SPEC")
    ask.add_argument("--project", metavar="DIR")
    make = commands.add_parser("make", help="make only pictures the creator approved")
    make.add_argument("project", metavar="DIR")
    make.add_argument("--pick", action="append", required=True, metavar="N")
    make.add_argument("--sheet", metavar="sN")
    make.add_argument("--max-cost", type=float, metavar="USD")
    make.add_argument("--yes", action="store_true")
    make.add_argument("--dry-run", action="store_true")
    make.add_argument("--new", action="store_true", help="approve a fresh paid attempt")
    for command in (question, answer, ask, make):
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_pictures)
