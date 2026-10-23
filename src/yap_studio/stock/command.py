"""CLI for asking about stock, collecting approved picks, and printing credits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio import paths
from yap_studio.errors import OutputPathError, ValidationError, YapError
from yap_studio.media.catalog import load_catalog
from yap_studio.project.model import load_cut_words
from yap_studio.project.status import ProjectContext
from yap_studio.stock.credits import credits
from yap_studio.stock.fetch import fetch
from yap_studio.stock.get import get_picks
from yap_studio.stock.pexels import PexelsProvider
from yap_studio.stock.sheet import option_label, search_sheet
from yap_studio.style import resolve
from yap_studio.taste import apply_taste


def _stock(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            if args.stock_command == "search":
                source = paths.validate_input(Path(args.requests).expanduser())
                try:
                    requests = json.loads(source.read_bytes())
                except (OSError, ValueError) as exc:
                    raise ValidationError("requests", "cannot read stock requests JSON") from exc
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
                data = search_sheet(
                    requests,
                    load_cut_words(words_path),
                    load_catalog()["items"],
                    style,
                    project_dir=project_dir,
                    provider=PexelsProvider(),
                    fetch=fetch,
                    per_phrase=args.per_phrase,
                )
            elif args.stock_command == "get":
                data = get_picks(
                    Path(args.project).expanduser(),
                    args.pick,
                    yes=args.yes,
                    sheet=args.sheet,
                    provider=PexelsProvider(),
                    fetch=fetch,
                )
            else:
                data = credits(Path(args.project).expanduser(), all_clips=args.all)
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if isinstance(exc, OutputPathError) else 2
    if args.json:
        if args.stock_command == "search" and data["status"] == "ok":
            sheet_dir = project_dir / "stock/sheets" / data["sheet"]
            data = {
                **data,
                "html": str((sheet_dir / "pick-sheet.html").resolve()),
                "json": str((sheet_dir / "pick-sheet.json").resolve()),
            }
        print(json.dumps(data, allow_nan=False))
    elif args.stock_command == "credits":
        print(data["text"])
    elif args.stock_command == "search":
        if data["status"] == "unavailable":
            print(data["message"])
            print("\n".join(data["setup"]))
        else:
            print(project_dir / "stock/sheets" / data["sheet"] / "pick-sheet.html")
            for phrase in data["phrases"]:
                print(f"{phrase['number']}. {phrase['text']}")
                for candidate in phrase["candidates"]:
                    print("  " + option_label(candidate))
                if phrase["note"]:
                    print("  " + phrase["note"])
            for entry in data["skipped"]:
                print(f"Skipped {entry['id']}: {entry['reason']}")
    else:
        for key in ("downloaded", "cached"):
            for entry in data[key]:
                print(f"{entry['code']}: {key} as {entry['item']}")
        for entry in data["failed"]:
            print(f"{entry['code']}: {entry['reason']}")
    return int(args.stock_command == "get" and bool(data["failed"]))


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("stock", help="ask about free stock clips before downloading")
    commands = parser.add_subparsers(dest="stock_command", required=True)
    search = commands.add_parser("search", help="write a stock pick sheet for the creator")
    search.add_argument("requests", metavar="REQUESTS.json")
    search.add_argument("--cut-words", required=True, metavar="FILE")
    search.add_argument("--style", metavar="SPEC")
    search.add_argument("--project", metavar="DIR")
    search.add_argument(
        "--per-phrase",
        type=int,
        default=3,
        metavar="N",
        help="clips per phrase, 1 to 3 (default: 3)",
    )
    get = commands.add_parser("get", help="download only approved codes")
    get.add_argument("project", metavar="DIR")
    get.add_argument("--pick", action="append", required=True, metavar="CODE")
    get.add_argument("--sheet", metavar="sN")
    get.add_argument("--yes", action="store_true")
    listing = commands.add_parser("credits", help="print the video's stock credits")
    listing.add_argument("project", metavar="DIR")
    listing.add_argument("--all", action="store_true")
    for command in (search, get, listing):
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_stock)
