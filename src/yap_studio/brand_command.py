"""Command line management of creator brand assets."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from yap_studio.errors import YapError


def _fonts(args: argparse.Namespace) -> int:
    """Add or list private creator font faces."""
    from yap_studio.render import user_fonts

    try:
        if args.fonts_command == "add":
            faces = user_fonts.add_font_files(
                [Path(file) for file in args.files], family=args.family, replace=args.replace)
        else:
            faces = user_fonts.list_fonts()
    except YapError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([asdict(face) for face in faces], indent=2, ensure_ascii=False))
    elif faces:
        for face in faces:
            print(f"{face.family}: {face.style} ({face.file})")
    else:
        print("No creator fonts.")
    return 0


def _words(args: argparse.Namespace) -> int:
    """Manage the creator's caption spelling list."""
    from yap_studio import brand_words

    try:
        if args.words_command == "add":
            if not args.words and not args.file:
                print("error: provide words or --file", file=sys.stderr)
                return 2
            entries = brand_words.read_words_file(Path(args.file)) if args.file else []
            words = brand_words.add_words([*args.words, *entries])
        elif args.words_command == "remove":
            words = brand_words.remove_words(args.words)
        elif args.words_command == "set":
            words = brand_words.set_words_file(Path(args.file))
        else:
            words = brand_words.list_words()
    except YapError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(words, indent=2, ensure_ascii=False))
    elif words:
        for word in words:
            print(word)
    else:
        print("No creator words.")
    return 0


def add_brand_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register brand commands without importing the font implementation."""
    brand = subparsers.add_parser("brand", help="manage creator brand assets")
    commands = brand.add_subparsers(dest="brand_command", required=True)
    fonts = commands.add_parser("fonts", help="manage creator font files")
    font_commands = fonts.add_subparsers(dest="fonts_command", required=True)
    add = font_commands.add_parser("add", help="add private font files")
    add.add_argument("files", nargs="+", metavar="FILE")
    add.add_argument("--family", metavar="NAME")
    add.add_argument("--replace", action="store_true")
    add.add_argument("--json", action="store_true")
    add.set_defaults(func=_fonts)
    listing = font_commands.add_parser("list", help="list private font files")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=_fonts)
    words = commands.add_parser("words", help="manage creator spellings")
    word_commands = words.add_subparsers(dest="words_command", required=True)
    for command in ("add", "remove"):
        change = word_commands.add_parser(command, help=f"{command} creator spellings")
        change.add_argument("words", nargs="*" if command == "add" else "+", metavar="WORD")
        if command == "add":
            change.add_argument("--file", metavar="FILE")
        change.set_defaults(func=_words)
    word_list = word_commands.add_parser("list", help="list creator spellings")
    word_list.add_argument("--json", action="store_true")
    word_list.set_defaults(func=_words)
    word_set = word_commands.add_parser("set", help="replace spellings from a text file")
    word_set.add_argument("file", metavar="FILE")
    word_set.set_defaults(func=_words)
