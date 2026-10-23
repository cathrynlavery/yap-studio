"""Command registration and output for person mattes and card compositing."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from yap_studio import paths
from yap_studio.errors import OutputPathError, YapError
from yap_studio.matte import check, composite, make
from yap_studio.matte.model import Cleanup


def _make(args: argparse.Namespace) -> int:
    started = time.monotonic()
    out = Path(args.out).expanduser()
    try:
        data = make.make_matte(
            Path(args.video).expanduser(),
            out,
            backend=args.backend,
            quality=args.quality,
            cleanup=Cleanup(median=not args.no_median),
            notice=None if args.json else make._model_notice,
        )
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if exc.code == "unsafe_path" else 2
    seconds = time.monotonic() - started
    if args.json:
        print(
            json.dumps(
                {
                    "out": str(out),
                    "status": data.status,
                    "reason": data.reason,
                    "frames": data.video.frames,
                    "backend": data.backend,
                    "seconds": seconds,
                },
                allow_nan=False,
            )
        )
    else:
        if data.status == "unavailable":
            print(f"Matte unavailable: {data.reason}")
        else:
            median = "on" if data.cleanup.median else "off"
            refine = "on" if data.cleanup.refine else "off"
            quality = f" ({data.backend['quality']})" if "quality" in data.backend else ""
            print(
                f"Matte: {data.video.frames} frames, {data.backend['name']}{quality}, "
                f"median {median}, refine {refine}, {seconds:.2f} s."
            )
            if data.empty_frames:
                print(
                    f"Guard: {len(data.empty_frames)} frame(s) had no person and were left empty."
                )
            print(f"Wrote {out / 'matte.mkv'}")
        print(f"Wrote {out / 'matte.json'}")
    return 0


def _composite(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser()
    try:
        with paths.operation():
            cards = composite.load_cards(Path(args.cards).expanduser())
            # Reserve the cards document as well as the media inputs.
            paths.validate_output(out, allowed_existing_suffixes=frozenset())
            result = composite.composite_video(
                Path(args.plate).expanduser(),
                cards,
                out,
                matte_dir=Path(args.matte).expanduser() if args.matte else None,
                lossless=args.lossless,
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
        print(json.dumps(result.to_dict(), allow_nan=False))
    else:
        behind = sum(card.placement == "behind_head" for card in result.cards)
        matte = "matte used" if result.matte_used else "no matte"
        print(
            f"Composite: {result.frames} frames, {len(result.cards)} card(s) "
            f"({behind} behind the head), {matte}."
        )
        for warning in result.warnings:
            print(f"warning: {warning}")
        print(f"Wrote {result.path}")
    return 0


def _check(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            cards = composite.load_cards(Path(args.cards).expanduser())
            report = check.check_cutout(
                Path(args.plate).expanduser(),
                Path(args.matte).expanduser() if args.matte else None,
                Path(args.composite).expanduser(),
                cards,
            )
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if isinstance(exc, OutputPathError) or exc.code == "unsafe_path" else 2
    if args.json:
        print(json.dumps(report.to_dict(), allow_nan=False))
    else:
        if report.status == "skipped":
            print(f"SKIPPED person cutout: {report.reason}")
        else:
            noun = "card" if report.behind == 1 else "cards"
            print(f"{report.status.upper()} person cutout: "
                  f"{report.behind} {noun} behind the person, {report.frames} frames")
        for result in report.checks:
            print(f"  {result.id}: {result.status}, {result.detail}")
    return int(report.status == "fail" or (args.strict and report.status in ("review", "skipped")))


def add_matte_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register matte creation, card compositing, and cutout checking subcommands."""
    parser = subparsers.add_parser("matte", help="make a person matte for an upright video")
    commands = parser.add_subparsers(dest="matte_command", required=True)
    creation = commands.add_parser("make", help="write a gray person matte and its document")
    creation.add_argument("video", metavar="VIDEO")
    creation.add_argument("--out", required=True, metavar="DIR")
    creation.add_argument("--backend", choices=("auto", "vision", "hyperframes", "chroma"),
                          default="auto")
    creation.add_argument("--quality", choices=("accurate", "balanced"), default="accurate")
    creation.add_argument("--no-median", action="store_true")
    creation.add_argument("--json", action="store_true")
    creation.set_defaults(func=_make)
    compositing = commands.add_parser("composite", help="composite cards with optional person keys")
    compositing.add_argument("plate", metavar="PLATE")
    compositing.add_argument("--cards", required=True, metavar="CARDS.json")
    compositing.add_argument("--out", required=True, metavar="OUT.mp4")
    compositing.add_argument("--matte", metavar="DIR")
    compositing.add_argument("--lossless", action="store_true")
    compositing.add_argument("--json", action="store_true")
    compositing.set_defaults(func=_composite)
    checking = commands.add_parser("check", help="check cards composited behind the person")
    checking.add_argument("plate", metavar="PLATE")
    checking.add_argument("--cards", required=True, metavar="CARDS.json")
    checking.add_argument("--composite", required=True, metavar="VIDEO")
    checking.add_argument("--matte", metavar="DIR")
    checking.add_argument("--json", action="store_true")
    checking.add_argument("--strict", action="store_true")
    checking.set_defaults(func=_check)
