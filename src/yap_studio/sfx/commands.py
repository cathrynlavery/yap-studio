"""Command line access to creator-owned sound kits."""

from __future__ import annotations

import argparse
import json
import sys

from yap_studio.sfx import RULES, SfxError, add, generate, list_kit


def _run(args: argparse.Namespace) -> int:
    try:
        if args.sfx_command == "list":
            result = list_kit(args.kit)
        elif args.sfx_command == "kit":
            result = generate(args.kit, yes=args.yes, dry_run=args.dry_run,
                              confirm_retry=args.confirm_retry, only=args.only)
        else:
            result = add(args.name, args.prompt, args.duration, args.rule,
                         kit=args.kit, gain_db=args.gain_db, priority=args.priority,
                         yes=args.yes, dry_run=args.dry_run)
    except (SfxError, OSError, ValueError) as exc:
        if not isinstance(exc, SfxError):
            exc = SfxError(str(exc))
        if args.json:
            print(json.dumps({"error": exc.to_dict(), "notices": exc.notices}),
                  file=sys.stderr)
        else:
            for line in exc.notices:
                print(line, file=sys.stderr)
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for line in result.get("notices", []):
            print(line, file=sys.stderr)
    if not args.json and args.sfx_command == "list":
        for sound in result["sounds"]:
            if sound.get("made"):
                print(f"{sound['name']}: {sound['rule']}, peak {sound['peak_s']:g} s")
            else:
                print(f"{sound['name']}: {sound['state']} - {sound['cache_path']}")
    elif not args.json and result["status"] == "dry_run":
        print(json.dumps(result.get("requests", result.get("request")), indent=2))
    elif not args.json:
        print("Sound kit ready." if result["status"] == "fresh" else
              "Generated: " + ", ".join(result["generated"]))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register list, kit and add without requiring a project."""
    parser = subparsers.add_parser("sfx", help="manage sound kits")
    commands = parser.add_subparsers(dest="sfx_command", required=True)
    listing = commands.add_parser("list", help="show kit sounds and cache states")
    kit = commands.add_parser("kit", help="generate missing or changed sounds")
    custom = commands.add_parser("add", help="save and generate a custom sound")
    custom.add_argument("name", metavar="NAME")
    custom.add_argument("--prompt", required=True, metavar="TEXT")
    custom.add_argument("--duration", required=True, type=float, metavar="S")
    custom.add_argument("--rule", required=True,
                        choices=sorted(RULES))
    custom.add_argument("--gain-db", type=float, default=0, metavar="G")
    custom.add_argument("--priority", type=int, default=1, metavar="N")
    for command in (listing, kit, custom):
        command.add_argument("--kit", default="soft", metavar="NAME")
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_run)
    for command in (kit, custom):
        command.add_argument("--yes", action="store_true")
        command.add_argument("--dry-run", action="store_true")
    kit.add_argument("--confirm-retry", action="store_true")
    kit.add_argument("--only", metavar="NAME", help="generate only this sound")
