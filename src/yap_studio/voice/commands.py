"""Command line access to own-voice samples, consent and fixes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio import voice
from yap_studio.errors import YapError
from yap_studio.voice.model import VoiceError


def _run(args: argparse.Namespace) -> int:
    root = Path(args.project).expanduser().absolute()

    def show_cost(cost):
        if not args.json:
            for line in cost["lines"]:
                print(line, flush=True)

    try:
        command = args.voice_command
        if command == "samples":
            result = voice.samples(root)
        elif command == "consent":
            result = voice.consent(
                root, yes=args.yes, by=args.by, voice_name=args.voice_name, withdraw=args.withdraw
            )
        elif command == "create":
            result = voice.create(root, yes=args.yes, dry_run=args.dry_run, samples=args.samples)
        elif command == "link":
            result = voice.link(root, name=args.name, yes=args.yes, samples=args.samples)
        elif command == "propose":
            result = voice.propose(root, args.first, args.last, args.say, why=args.why,
                                   from_take_words=args.from_take_words, clone=args.clone)
        elif command == "make":
            result = voice.make(
                root,
                args.fix,
                yes=args.yes,
                max_credits=args.max_credits,
                dry_run=args.dry_run,
                new=args.new,
                confirm_retry=args.confirm_retry,
                plain=args.plain,
                show_cost=show_cost,
            )
        elif command in ("approve", "reject"):
            result = voice.decide(root, args.fix, command, via="chat")
        elif command == "delete-voice":
            result = voice.delete_voice(root, yes=args.yes)
        else:
            result = voice.list_fixes(root)
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = VoiceError("The voice command could not read or write its files.")
        cost = getattr(exc, "cost", None)
        notices = getattr(exc, "notices", [])
        if args.json:
            print(
                json.dumps(
                    {
                        "error": exc.to_dict(),
                        **({"cost": cost} if cost else {}),
                        **({"notices": notices} if notices else {}),
                    }
                ),
                file=sys.stderr,
            )
        else:
            for line in notices:
                print(line, file=sys.stderr)
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    elif command == "list":
        for fix in result["fixes"]:
            kind = fix.get("from", {}).get("kind", "clone")
            print(f"{fix['id']}: {fix['said']} to {fix['say']} ({kind}, {fix['status']})")
    elif command == "propose":
        origin = result["from"]
        if origin["kind"] == "own":
            seconds = int(origin["start_s"])
            print(f"{result['id']}: from your own recording at {seconds // 60}:{seconds % 60:02}")
        else:
            print(f"{result['id']}: clone fallback")
    else:
        for line in result.get("notices", []):
            print(line, file=sys.stderr)
        print(result.get("status", "Voice samples ready."))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the complete voice workflow without importing a speech engine."""
    parser = subparsers.add_parser("voice", help="fix misspoken words in your own voice")
    commands = parser.add_subparsers(dest="voice_command", required=True)
    descriptions = {
        "samples": "export clean speech samples",
        "consent": "record or withdraw voice consent",
        "create": "create your voice",
        "link": "link your own voice",
        "propose": "propose a word fix",
        "make": "generate proposed fixes",
        "approve": "approve a ready fix",
        "reject": "keep the original words",
        "list": "list voice fixes",
        "delete-voice": "delete your linked voice",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(
            name, help=description, description=description.capitalize() + "."
        )
        command.add_argument("project", metavar="DIR")
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_run)
        if name in ("consent", "create", "link", "make", "delete-voice"):
            command.add_argument("--yes", action="store_true")
        if name in ("create", "make"):
            command.add_argument("--dry-run", action="store_true")
        if name in ("create", "link"):
            command.add_argument("--samples", type=Path, metavar="PATH",
                                 help="use WAV files from a samples manifest")
        if name == "consent":
            command.add_argument("--by")
            command.add_argument("--voice-name")
            command.add_argument("--withdraw", action="store_true")
        if name == "link":
            command.add_argument("--name", help="voice name, or the creator recipe default")
        if name == "propose":
            command.add_argument("--from", dest="first", required=True)
            command.add_argument("--to", dest="last", required=True)
            command.add_argument("--say", required=True)
            command.add_argument("--why", default="")
            source = command.add_mutually_exclusive_group()
            source.add_argument("--from-take-words", nargs=2, metavar=("FIRST", "LAST"),
                                help="use these recorded words")
            source.add_argument("--clone", action="store_true", help="use the clone fallback")
        if name in ("make", "approve", "reject"):
            command.add_argument(
                "--fix", action="append" if name == "make" else "store", required=True
            )
        if name == "make":
            command.add_argument("--max-credits", type=float)
            command.add_argument("--new", action="store_true")
            command.add_argument("--confirm-retry", action="store_true")
            command.add_argument("--plain", action="store_true",
                                 help="send plain words without tags or pronunciations")
