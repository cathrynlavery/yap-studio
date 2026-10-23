"""List every picture jump in a fresh verified cut."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio import paths
from yap_studio.errors import YapError
from yap_studio.joins.helper import JoinError
from yap_studio.joins.plan import build_plan
from yap_studio.project.status import ProjectContext
from yap_studio.stages.timeline import _fresh_cut


def _joins(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            context = ProjectContext(Path(args.directory).expanduser())
            record = _fresh_cut(context)
            timeline = context.resolved_timeline()
            pieces = record["plan"].get("pieces")
            if pieces is None:
                raise JoinError("The cut receipt has no pieces for jump cuts.",
                                fix="Run yap-studio cut DIR, then list joins again.")
            result = build_plan(context.project_dir, timeline, pieces,
                                context.json("cut/cut-words.json")["words"],
                                context.render_style().data["moves"])
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), allow_nan=False))
    else:
        for join in result.joins:
            minutes, seconds = divmod(join.time_s, 60)
            before = join.before_word["text"] if join.before_word else ""
            after = join.after_word["text"] if join.after_word else ""
            print(f"{int(minutes)}:{seconds:04.1f}  {join.kind:<7}  {join.handling:<7}  "
                  f"{join.reason}  {before!r} / {after!r}")
        print(", ".join(f"{count} {kind}" for kind, count in result.to_dict()["counts"].items()))
    return 0


def add_joins_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the joins listing; it may save measured faces to the project cache."""
    parser = subparsers.add_parser("joins", help="list jump cuts and their picture handling")
    parser.add_argument("directory", metavar="DIR")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=_joins)
