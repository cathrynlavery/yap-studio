"""Cheap parser registration for watch commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio.errors import OutputPathError, YapError

ASR_CHOICES = ("auto", "none", "parakeet-mlx", "parakeet-onnx", "faster-whisper")


def add_watch_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register watch and its extensible set of required subcommands."""
    watch = subparsers.add_parser("watch", help="prepare offline render review evidence")
    commands = watch.add_subparsers(dest="watch_command", required=True)
    pack = commands.add_parser("pack", help="write a watch pack for a video")
    pack.add_argument("video", metavar="VIDEO")
    for option in ("edl", "out", "words", "verify"):
        pack.add_argument(f"--{option}")
    pack.add_argument("--every", type=float, default=2.0, metavar="SECONDS")
    pack.add_argument("--columns", type=int, default=6, metavar="N")
    pack.add_argument("--asr", choices=ASR_CHOICES, default="auto")
    pack.add_argument("--json", action="store_true")
    pack.set_defaults(func=_pack)
    note = commands.add_parser("note", help="validate and record offline reviewer findings")
    note.add_argument("video", metavar="VIDEO")
    note.add_argument("--file", required=True, metavar="FINDINGS.json")
    note.add_argument("--findings", metavar="PATH")
    note.add_argument("--replace", action="store_true")
    note.add_argument("--json", action="store_true")
    note.set_defaults(func=_note)
    listing = commands.add_parser("list", help="print reviewer findings for a video")
    listing.add_argument("video", metavar="VIDEO")
    listing.add_argument("--findings", metavar="PATH")
    listing.add_argument("--source", choices=("agent", "machine", "all"), default="all")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=_list)
    scan = commands.add_parser("scan", help="record deterministic hints from a watch pack")
    scan.add_argument("pack", metavar="PACK")
    scan.add_argument("--findings", metavar="PATH")
    scan.add_argument("--still-s", type=float, default=4.0, metavar="S")
    scan.add_argument("--pause-s", type=float, default=1.0, metavar="S")
    scan.add_argument("--loudness-jump-lu", type=float, default=8.0, metavar="LU")
    scan.add_argument("--dry-run", action="store_true")
    scan.add_argument("--json", action="store_true")
    scan.set_defaults(func=_scan)


def _findings_error(args: argparse.Namespace, exc: YapError) -> int:
    if args.json:
        error = exc.to_dict()
        # A batch can contain many problems; report every one in JSON as well.
        error["message"] = exc.message
        print(json.dumps({"error": error}, ensure_ascii=False), file=sys.stderr)
    else:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)
    return 3 if isinstance(exc, OutputPathError) else 2


def _note(args: argparse.Namespace) -> int:
    from .findings import note_from_file

    try:
        result = note_from_file(
            args.video, args.file, findings_path=args.findings, replace=args.replace
        )
    except YapError as exc:
        return _findings_error(args, exc)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(
            f"Recorded {result['added']} findings for {Path(args.video).name} "
            f"({result['duplicates']} duplicates skipped, {result['removed']} replaced). "
            f"Total {result['total']} in {result['path']}."
        )
    return 0


def _list(args: argparse.Namespace) -> int:
    from .findings import format_findings, load_findings

    try:
        result = load_findings(args.video, findings_path=args.findings)
    except YapError as exc:
        return _findings_error(args, exc)
    if args.source != "all":
        result["findings"] = [
            finding for finding in result["findings"] if finding["source"] == args.source
        ]
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        if result["stale"]:
            print("warning: These findings were recorded for a different render.")
        video = Path(args.video).name
        findings = result["findings"]
        if not result["exists"]:
            print(f"No findings for {video}.")
        else:
            agents = sum(finding["source"] == "agent" for finding in findings)
            print(
                f"{len(findings)} findings for {video}: {agents} from agents, "
                f"{len(findings) - agents} machine"
            )
            for line in format_findings(findings):
                print(line)
    return 1 if result["stale"] else 0


def _pack(args: argparse.Namespace) -> int:
    from .pack import build_pack

    try:
        result = build_pack(
            args.video,
            edl=args.edl,
            out=args.out,
            every_s=args.every,
            columns=args.columns,
            asr=args.asr,
            words=args.words,
            verify=args.verify,
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
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        video = Path(args.video).resolve()
        out = Path(args.out) if args.out is not None else video.with_name(f"{video.stem}-watch")
        tiles = sum(len(strip["tiles"]) for strip in result["timeline"]["strips"])
        print(f"Pack: {out}, {tiles} frames, {len(result['timeline']['strips'])} strips written")
        source = "edit list" if result["edl"] else "detected"
        print(f"Cuts: {len(result['cuts'])}, source: {source}")
        for warning in result["warnings"]:
            print(f"warning: {warning}")
        for key in ("transcript", "verify"):
            status = result[key]
            if key == "transcript" and status["status"] == "ok":
                detail = f", {status['engine'] or status['source']}, {status['words']} words"
            else:
                detail = f", {status['reason']}" if status["reason"] else ""
            print(f"{key}: {status['status']}{detail}")
        print(f"next: yap-studio watch scan {out}, then read {out / 'pack.md'} and the strips")
    return 0


def _scan(args: argparse.Namespace) -> int:
    from .findings import format_findings
    from .scan import run_scan

    try:
        result = run_scan(
            args.pack,
            findings_path=args.findings,
            dry_run=args.dry_run,
            still_s=args.still_s,
            pause_s=args.pause_s,
            loudness_jump_lu=args.loudness_jump_lu,
        )
    except YapError as exc:
        return _findings_error(args, exc)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        for check, rule in result["rules"].items():
            if rule["status"] == "skipped":
                print(f"{check}: skipped, {rule['reason']}")
                continue
            count = sum(finding["check"] == check for finding in result["findings"])
            basis = " on silence, no transcript" if rule["basis"] == "silence" else ""
            print(f"{check}: ran{basis}, {count} finding{'s' if count != 1 else ''}")
        entries = [
            {**entry, "id": f"f{index}", "source": "machine", "reviewer": None}
            for index, entry in enumerate(result["findings"], 1)
        ]
        for line in format_findings(entries):
            print(line)
        if result["recorded"]:
            print(
                f"Recorded {result['added']} machine findings in {result['path']} "
                f"(replacing {result['removed']})."
            )
        else:
            print("Dry run: nothing written.")
    return 0
