"""Command registration and output for face detection and clearance checks."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from pathlib import Path

from yap_studio import paths, receipt
from yap_studio.errors import ValidationError, YapError
from yap_studio.faces import detect
from yap_studio.faces.clearance import (
    DEFAULT_FACE_CLEARANCE_PX,
    DEFAULT_HAIR,
    ClearanceReport,
    HairAllowance,
    Layout,
    check_layout,
    face_clearance_px,
    load_layout,
)
from yap_studio.faces.model import load_faces
from yap_studio.style import resolve


def _every(value: str) -> int:
    try:
        every = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("every must be an integer >= 1") from exc
    if every < 1:
        raise argparse.ArgumentTypeError("every must be an integer >= 1")
    return every


def _clock(seconds: float) -> str:
    hundredths = round(seconds * 100)
    minutes, remainder = divmod(hundredths, 6000)
    return f"{minutes}:{remainder / 100:05.2f}"


def _pixels(value: str) -> float:
    try:
        pixels = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pixels must be a finite number >= 0") from exc
    if not math.isfinite(pixels) or pixels < 0:
        raise argparse.ArgumentTypeError("pixels must be a finite number >= 0")
    return pixels


def _ratio(value: str) -> float:
    try:
        ratio = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ratio must be a finite number from 0 to 2") from exc
    if not math.isfinite(ratio) or not 0 <= ratio <= 2:
        raise argparse.ArgumentTypeError("ratio must be a finite number from 0 to 2")
    return ratio


def _print_error(exc: YapError, json_output: bool) -> None:
    if json_output:
        print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
    else:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)


def _detect(args: argparse.Namespace) -> int:
    started = time.monotonic()
    try:
        with paths.operation():
            video = Path(args.video).expanduser()
            paths.protect_input(video)
            data = detect.detect_faces(video, every=args.every)
            out = paths.validate_output(Path(args.out).expanduser())
            detect.write_faces(out, data)
    except YapError as exc:
        _print_error(exc, args.json)
        return 3 if exc.code == "unsafe_path" else 2
    seconds = time.monotonic() - started
    with_faces = sum(bool(faces) for faces in data.frames)
    if args.json:
        print(
            json.dumps(
                {
                    "out": str(out),
                    "status": data.status,
                    "reason": data.reason,
                    "frames": data.video.frames,
                    "frames_with_faces": with_faces,
                    "gaps": [{"start_frame": start, "end_frame": end} for start, end in data.gaps],
                    "seconds": seconds,
                },
                allow_nan=False,
            )
        )
    else:
        if data.status == "unmeasured":
            print(f"Faces unmeasured: {data.reason}")
        else:
            print(
                f"Faces: {with_faces} of {data.video.frames} frames have a face, "
                f"{len(data.gaps)} gaps, every {args.every} frame(s), {seconds:.2f} s."
            )
            for start, end in data.gaps[:10]:
                print(
                    f"gap: frames {start} to {end} "
                    f"({_clock(data.time_at(start))} to {_clock(data.time_at(end))})"
                )
        print(f"Wrote {out}")
    return 0


def _print_report(report: ClearanceReport, layout: Layout) -> None:
    if report.status == "unmeasured":
        print(f"UNMEASURED face clearance: {report.reason}")
        return
    clearance = (
        f"margin {report.margin_px:.0f} px ({report.margin_source}), "
        f"hair {report.hair.above:g} above and {report.hair.sides:g} each side"
    )
    if report.status == "pass":
        print(
            f"PASS face clearance: no overlap with detected faces; "
            f"{report.elements} elements and {report.captions} phrases in {report.frames} frames, "
            f"faces found in {report.frames_with_faces}, none in {report.frames_without_faces}, "
            f"{clearance}"
        )
        return
    coverage = (
        f"; faces found in {report.frames_with_faces} of {report.frames} frames"
        if report.frames_with_faces is not None
        else ""
    )
    print(
        f"FAIL face clearance: {len(report.violations)} violation(s), "
        f"{clearance}, caption gap {report.caption_gap_px:.0f} px{coverage}"
    )
    if report.reason is not None:
        print(f"  faces unmeasured: {report.reason}; the face check did not run")
    kinds = {element.id: element.kind for element in layout.elements}
    for violation in report.violations:
        target = f"{violation.target_kind} {violation.target_id}"
        if violation.target_type == "caption":
            target += f" {json.dumps(violation.text, ensure_ascii=False)}"
        window = (
            f"from {_clock(report.time_at(violation.start_frame))} "
            f"to {_clock(report.time_at(violation.end_frame))} "
            f"(frames {violation.start_frame} to {violation.end_frame})"
        )
        if violation.kind == "caption_gap":
            other = f"{kinds[violation.other_id]} {violation.other_id}"
            distance = (
                f"overlaps {other} by {abs(violation.gap_px):.0f} px"
                if violation.gap_px < 0
                else f"is {violation.gap_px:.0f} px from {other}"
            )
            print(f"  {target} {distance} {window}, needs {violation.required_px:.0f} px")
        else:
            distance = (
                f"overlaps by {abs(violation.gap_px):.0f} px"
                if violation.gap_px < 0
                else f"gap {violation.gap_px:.0f} px"
            )
            face = violation.face
            head = violation.head
            assert face is not None and head is not None
            print(
                f"  {target} covers the head {window}: {distance}, "
                f"needs {violation.required_px:.0f} px; face x {face.x:.0f} y {face.y:.0f} "
                f"w {face.w:.0f} h {face.h:.0f}, head top y {head.y:.0f} "
                f"at {_clock(report.time_at(violation.frame))}"
            )


def _check(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            video = paths.validate_input(Path(args.video).expanduser())
            layout = load_layout(paths.validate_input(Path(args.layout).expanduser()))
            style = resolve(args.style).data if args.style is not None else None
            if args.margin is not None:
                margin, margin_source = args.margin, "option"
            elif layout.clearance_px is not None:
                margin, margin_source = layout.clearance_px, "layout"
            elif style is not None:
                margin, margin_source = face_clearance_px(style), "style"
            else:
                margin, margin_source = DEFAULT_FACE_CLEARANCE_PX, "default"
            layout_hair = layout.hair if layout.hair is not None else DEFAULT_HAIR
            hair = HairAllowance(
                args.hair_above if args.hair_above is not None else layout_hair.above,
                args.hair_sides if args.hair_sides is not None else layout_hair.sides,
            )
            if args.faces is not None:
                data = load_faces(paths.validate_input(Path(args.faces).expanduser()))
                if data.video.sha256 != receipt.sha256_file(video):
                    error = ValidationError(
                        "faces.video.sha256", "faces file was made from a different video"
                    )
                    error.suggested_action = (
                        f"run yap-studio faces detect {shlex.quote(args.video)} --out faces.json"
                    )
                    raise error
            else:
                data = detect.detect_faces(video, every=args.every)
            report = check_layout(
                data,
                layout,
                margin_px=margin,
                caption_gap_px=args.caption_gap,
                hair=hair,
                margin_source=margin_source,
            )
    except YapError as exc:
        _print_error(exc, args.json)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), allow_nan=False))
    else:
        _print_report(report, layout)
    return int(report.status == "fail" or (report.status == "unmeasured" and args.strict))


def add_faces_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the faces command and its required detect or check subcommand."""
    parser = subparsers.add_parser("faces", help="measure faces and check video clearance")
    commands = parser.add_subparsers(dest="faces_command", required=True)
    detection = commands.add_parser("detect", help="write face boxes for every video frame")
    detection.add_argument("video", metavar="VIDEO")
    detection.add_argument("--out", required=True, metavar="FACES.json")
    detection.add_argument("--every", type=_every, default=1, metavar="N")
    detection.add_argument("--json", action="store_true")
    detection.set_defaults(func=_detect)
    check = commands.add_parser("check", help="check a finished video's layout clearance")
    check.add_argument("video", metavar="VIDEO")
    check.add_argument("--layout", required=True, metavar="LAYOUT.json")
    check.add_argument("--faces", metavar="FACES.json")
    check.add_argument("--every", type=_every, default=1, metavar="N")
    check.add_argument("--margin", type=_pixels, metavar="PX")
    check.add_argument("--style", metavar="SPEC")
    check.add_argument("--hair-above", type=_ratio, metavar="RATIO")
    check.add_argument("--hair-sides", type=_ratio, metavar="RATIO")
    check.add_argument("--caption-gap", type=_pixels, metavar="PX")
    check.add_argument("--json", action="store_true")
    check.add_argument("--strict", action="store_true")
    check.set_defaults(func=_check)
