"""Parsers and display handlers for the creator's media catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio.errors import InputError, YapError
from yap_studio.media.album import DEFAULT_WAIT_S, export_album
from yap_studio.media.album import summary as album_summary
from yap_studio.media.catalog import add_paths, catalog_path, get_item, list_items, load_catalog
from yap_studio.media.describe import describe_item, read_description_file
from yap_studio.media.inventory import search_inventory, sync_inventory
from yap_studio.media.select import parse_request, rank_candidates
from yap_studio.media.sheet import make_sheet
from yap_studio.media.sound import find_sound


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register media commands on the main CLI's subparser collection."""
    parser = subparsers.add_parser("media", help="catalog your own clips and images")
    commands = parser.add_subparsers(dest="media_command", required=True)
    album = commands.add_parser("album", help="export a Photos album on a Mac")
    album.add_argument("album", metavar="ALBUM")
    album.add_argument("--out", type=Path, required=True, metavar="FOLDER")
    album.add_argument("--library", type=Path, metavar="PATH")
    album.add_argument("--dry-run", action="store_true")
    album.add_argument("--no-download", dest="download", action="store_false",
                       help="skip items not downloaded to this Mac")
    album.add_argument("--wait", type=float, default=DEFAULT_WAIT_S, metavar="SECONDS",
                       help="stop after this many seconds without progress")
    album.add_argument("--json", action="store_true")
    album.set_defaults(func=_media)
    for name in ("add", "list", "show", "sheet", "sound", "describe", "pick"):
        command = commands.add_parser(name)
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_media)
        if name == "add":
            command.add_argument("paths", nargs="+", metavar="PATH")
        elif name == "list":
            command.add_argument(
                "--type", choices=("clip", "photo", "screenshot", "screen_recording")
            )
        elif name == "pick":
            command.add_argument("--file", required=True, metavar="REQUEST.json")
        else:
            command.add_argument("id", metavar="ID")
            if name == "sheet":
                command.add_argument("--frames", type=int, default=6)
                command.add_argument(
                    "--asr", default="auto",
                    choices=("auto", "parakeet-mlx", "parakeet-onnx", "faster-whisper", "none"),
                )
                command.add_argument("--force", action="store_true")
            elif name == "sound":
                command.add_argument("--top", type=int, default=5, metavar="N")
            elif name == "describe":
                command.add_argument("--file", required=True, metavar="JSON")
    inventory = commands.add_parser("inventory", help="mirror media for HyperFrames tools")
    inventory_commands = inventory.add_subparsers(dest="inventory_command", required=True)
    for name in ("sync", "search"):
        command = inventory_commands.add_parser(name)
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_media)
        if name == "search":
            command.add_argument("--type", required=True, choices=("video", "image"))
            command.add_argument("intent", metavar="INTENT")


def _list_line(item: dict) -> str:
    duration = "still" if item["duration_s"] is None else f"{item['duration_s']:.1f} s"
    described = "described" if item["description"] is not None else "undescribed"
    locations = item["locations"]
    # On timestamp ties, prefer the last inserted location.
    latest = max(enumerate(locations), key=lambda pair: (pair[1]["seen_at"], pair[0]))[1] if (
        locations
    ) else None
    name = Path(latest["path"]).name if latest else "(no locations)"
    return (
        f"{item['id']} {item['media_type']} {item['orientation']} "
        f"{item['width']}x{item['height']} {duration} {described} {name}"
    )


def _speech_line(speech: dict | None, *, item_id: str, still: bool = False) -> str:
    if speech is None:
        return "Speech: not applicable for a still" if still else (
            f"Speech: not checked yet; run yap-studio media sheet {item_id}"
        )
    line = f"Speech: {speech['status']}"
    if speech["message"]:
        line += f": {speech['message']}"
    if speech["status"] == "words" and speech["text"]:
        line += "; text: " + speech["text"][:200]
    return line


def _show_text(item: dict) -> str:
    by_agent = " (set by agent)" if item["media_type_by"] == "agent" else ""
    timing = "still" if item["duration_s"] is None else (
        f"{item['duration_s']:.3f} s; fps: {item['fps']}"
    )
    lines = [
        f"ID: {item['id']}",
        f"Kind: {item['kind']}; media type: {item['media_type']}{by_agent}",
        f"Size: {item['width']}x{item['height']}; orientation: {item['orientation']}",
        f"Duration: {timing}",
        f"HDR: {item['hdr'] or 'none'}",
        f"Audio: {'yes' if item['has_audio'] else 'no'}",
    ]
    if stock := item.get("stock"):
        lines.append(f"Stock: Pexels clip {stock['clip_id']} by {stock['author']}, "
                     f"{stock['license']}, {stock['page_url']}")
    camera = item["camera"]
    if camera is not None:
        name = " ".join(value for value in (camera["make"], camera["model"]) if value)
        if name:
            lines.append(f"Camera: {name}")
        if camera["captured_at"] is not None:
            lines.append(f"Captured: {camera['captured_at']}")
    lines.append(f"Provenance: {item['provenance']['kind'] or 'unknown'}")
    description = item["description"]
    if description is None:
        lines.append("No description yet.")
    else:
        lines.append(
            f"Description: {description['text']}" if description["text"] else "No description yet."
        )
        lines.append("Tags: " + (", ".join(description["tags"]) or "none"))
    for decision in item["decisions"]:
        span = "whole item" if decision["start_s"] is None else (
            f"{decision['start_s']:.3f} to {decision['end_s']:.3f} s"
        )
        revoked = " (revoked)" if decision["revoked_at"] is not None else ""
        lines.append(
            f"{decision['id']} {decision['verdict']} {span}: {decision['reason']} "
            f"({decision['by']}, {decision['at']}){revoked}"
        )
    lines.append(
        f"Sheet: {item['sheet']['path']}" if item["sheet"] is not None else
        f"No sheet yet. Run yap-studio media sheet {item['id']}."
    )
    speech = item["speech"]
    lines.append(_speech_line(speech, item_id=item["id"], still=item["kind"] == "image"))
    for location in item["locations"]:
        state = "exists" if location["exists"] else "missing"
        lines.append(f"Location ({state}): {location['path']}")
    return "\n".join(lines)


def _pick(file: str) -> dict:
    """Read a request and resolve candidate locations with stat calls only."""
    target = Path(file).expanduser()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(str(target), "request file not found") from exc
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InputError(str(target), "request file is not valid JSON") from exc
    except OSError as exc:
        raise InputError(str(target), f"cannot read request file: {exc}") from exc
    request = parse_request(data)
    candidates, missing = [], []
    for candidate in rank_candidates(load_catalog()["items"], request):
        for location in candidate["locations"]:
            try:
                if Path(location).is_file():
                    candidate["path"] = location
                    candidates.append(candidate)
                    break
            except OSError:
                continue
        else:
            missing.append(candidate["item"])
        if len(candidates) == request.limit:
            break
    return {"request": data, "candidates": candidates, "missing": missing}


def _pick_line(candidate: dict) -> str:
    state = "approved" if candidate["approved"] else "not approved"
    matched = ", ".join(candidate["matched"]) or "none"
    return (
        f"{candidate['item']} {candidate['in_s']:.3f} to {candidate['out_s']:.3f} s "
        f"{state}; score {candidate['score']}; matched: {matched}; {candidate['path']}"
    )


def _media(args: argparse.Namespace) -> int:
    exit_code = 0
    try:
        if args.media_command == "album":
            result = export_album(args.album, args.out, dry_run=args.dry_run,
                                  download=args.download, wait=args.wait, library=args.library,
                                  notify=lambda text: print(text, file=sys.stderr, flush=True))
            text = album_summary(result, args.out)
            if result["counts"]["not_downloaded"] and not args.dry_run:
                exit_code = 1
        elif args.media_command == "add":
            result = add_paths(args.paths)
            summary = (
                f"Added {len(result['added'])}, located {len(result['located'])}, "
                f"unchanged {len(result['unchanged'])}, skipped {len(result['skipped'])}."
            )
            lines = [summary]
            lines.extend(
                f"skipped: {entry['path']}: {entry['reason']}" for entry in result["skipped"]
            )
            lines.append(result["catalog"])
            text = "\n".join(lines)
            if result["unreadable"]:
                print(
                    f"warning: could not list {len(result['unreadable'])} folder(s); "
                    "media inside was not cataloged",
                    file=sys.stderr,
                )
                exit_code = 1
        elif args.media_command == "inventory":
            if args.inventory_command == "sync":
                result = sync_inventory()
                text = f"Mirrored {len(result['items'])}, skipped {len(result['skipped'])}."
            else:
                result = search_inventory(args.intent, kind=args.type)
                text = "\n".join(
                    f"{entry['id']} {entry['score']} {entry['description']}" for entry in result
                ) or "No candidates."
        elif args.media_command == "list":
            result = list_items(media_type=args.type)
            text = "\n".join(_list_line(item) for item in result) or "No cataloged media."
        elif args.media_command == "pick":
            result = _pick(args.file)
            text = "\n".join(_pick_line(entry) for entry in result["candidates"]) or (
                "No candidates."
            )
        elif args.media_command == "sheet":
            result = make_sheet(args.id, frames=args.frames, asr=args.asr, force=args.force)
            times = [
                "still" if frame["time_s"] is None else f"{frame['time_s']:.3f} s"
                for frame in result["sheet"]["frames"]
            ]
            text = "\n".join([
                result["sheet"]["path"], "Frames: " + ", ".join(times),
                _speech_line(result["speech"], item_id=args.id, still=result["speech"] is None),
            ])
            if result["speech"] is not None and result["speech"]["status"] == "failed":
                exit_code = 1
        elif args.media_command == "sound":
            result = find_sound(args.id, top=args.top)
            lines = [
                f"{moment['start_s']:.2f} s to {moment['end_s']:.2f} s, "
                f"peak {moment['peak_s']:.2f} s, {moment['peak_dbfs']:.1f} dBFS, "
                f"{moment['above_floor_db']:g} dB above the floor; "
                f"insert in_s {moment['in_s']:.2f}"
                for moment in result["moments"]
            ]
            if result["speech"] is None:
                lines.append(result["speech_note"])
            else:
                lines.extend(
                    f"{phrase['start_s']:.2f} s to {phrase['end_s']:.2f} s: "
                    f'"{phrase["text"]}"'
                    for phrase in result["speech"]
                )
            text = "\n".join(lines)
        elif args.media_command == "describe":
            result = describe_item(args.id, read_description_file(args.file))
            lines = [f"Updated {args.id}."]
            item = result["item"]
            if result["description"] is not None:
                lines.append(f"Description revision {result['description']['revision']}.")
            if result["decisions_added"]:
                lines.append("Added decisions: " + ", ".join(result["decisions_added"]))
            if result["revoked"]:
                lines.append("Revoked decisions: " + ", ".join(result["revoked"]))
            lines.append(
                f"Media type: {item['media_type']}; provenance: {item['provenance']['kind']}"
            )
            text = "\n".join(lines)
        else:
            result = get_item(args.id)
            result["locations"] = [
                {**location, "exists": Path(location["path"]).is_file()}
                for location in result["locations"]
            ]
            text = _show_text(result)
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            if args.media_command == "add":
                argument = args.paths[0]
            elif args.media_command in {"show", "sheet", "sound", "describe"}:
                argument = args.id
            elif args.media_command == "pick":
                argument = args.file
            elif args.media_command == "album":
                argument = str(args.out)
            else:
                argument = str(catalog_path())
            exc = InputError(argument, f"cannot read or write media: {exc}")
        if args.json:
            print(
                json.dumps({"error": exc.to_dict()}, indent=2, ensure_ascii=False, allow_nan=False),
                file=sys.stderr,
            )
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) if args.json else text)
    return exit_code
