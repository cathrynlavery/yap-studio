"""Command line entry point for the gallery still and opening sentence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio.errors import InputError, ValidationError, YapError
from yap_studio.gallery import still
from yap_studio.transcript import resolve_backend


def _still(args: argparse.Namespace) -> int:
    try:
        if args.take is None and args.inbox is None:
            raise ValidationError("TAKE", "provide TAKE or --inbox DIR")
        if args.take is not None and args.inbox is not None:
            raise ValidationError("TAKE", "provide TAKE or --inbox DIR, not both")
        if args.take is not None:
            take = Path(args.take)
            if not take.is_file():
                raise InputError(str(take), "no file at this path")
            if take.suffix.lower() not in still.TAKE_SUFFIXES:
                raise InputError(str(take), "expected a .mov, .mp4 or .m4v take")
            backend = resolve_backend(args.asr) if args.asr != "none" else None
            result = still.prepare_take(take, Path(args.out), backend=backend,
                                        transcribe=args.asr != "none")
        else:
            inbox = Path(args.inbox)
            result = _prepare(inbox, args)
    except YapError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["still"])
        if result["source"] == "take":
            print(f"{result['time_s']:.2f} s")
        print(result.get("sentence") or "no speech engine; the gallery uses sample words")
    return 0


def _prepare(inbox: Path, args: argparse.Namespace) -> dict:
    if args.inbox is not None and not inbox.is_dir():
        raise ValidationError("--inbox", "directory not found")
    if still.newest_take(inbox) is None:
        return still.prepare_preview(inbox, Path(args.out), transcribe=False)
    backend = resolve_backend(args.asr) if args.asr != "none" else None
    return still.prepare_preview(inbox, Path(args.out), backend=backend,
                                 transcribe=args.asr != "none")


def add_still_parser(style_commands: argparse._SubParsersAction, port_type=None) -> None:
    """Add the still command under style."""
    del port_type
    parser = style_commands.add_parser("still", help="pick a face still and opening words")
    parser.add_argument("take", nargs="?", metavar="TAKE")
    parser.add_argument("--inbox", metavar="DIR")
    parser.add_argument("--out", required=True, metavar="DIR")
    parser.add_argument("--asr", choices=(
        "auto", "none", "parakeet-mlx", "parakeet-onnx", "faster-whisper"
    ), default="auto")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=_still)
