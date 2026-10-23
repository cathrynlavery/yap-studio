"""Commands for inspecting derived projects and reading Studio session records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio.errors import YapError
from yap_studio.studio.export import catch_up, export_project, open_project, stop_project

_HELP = (
    "HyperFrames Studio is for agent and debugging inspection, not editing. "
    "The creator comments in the review page and the agent edits the project. "
    "Nothing is read back from Studio into the project."
)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the Studio inspection commands on the main CLI."""
    parser = subparsers.add_parser(
        "studio", help="agent and debugging inspection, not editing", description=_HELP,
    )
    commands = parser.add_subparsers(dest="studio_command", required=True)
    for name in ("export", "open", "stop", "catch-up"):
        command = commands.add_parser(name, description=_HELP)
        command.add_argument("directory", metavar="DIR", nargs="?", default=".")
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_studio)
        if name == "open":
            command.add_argument("--port", type=int, metavar="N")
            command.add_argument("--no-open", action="store_true")


def _studio(args: argparse.Namespace) -> int:
    try:
        root = Path(args.directory)
        if args.studio_command == "export":
            target = export_project(root)
            result = {"export": str(target),
                      **json.loads((target / "yap-export.json").read_bytes())}
        elif args.studio_command == "open":
            result = open_project(root, port=args.port, open_browser=not args.no_open)
        elif args.studio_command == "stop":
            result = stop_project(root)
        else:
            result = catch_up(root)
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}, ensure_ascii=False), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    elif args.studio_command == "export":
        print(result["export"])
        for stage, state in result["stages"].items():
            if not state["included"]:
                print(f"{stage} left out: {state['reason']}")
    elif args.studio_command == "open":
        print(result["url"])
    elif args.studio_command == "stop":
        print("Studio preview stopped.")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0
