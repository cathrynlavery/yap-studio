"""Commands for inspecting and deciding creator taste proposals."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from yap_studio.errors import InputError, YapError
from yap_studio.taste.check import check_taste
from yap_studio.taste.inbox import propose
from yap_studio.taste.learn import learn, measure_style
from yap_studio.taste.model import TasteRefusal
from yap_studio.taste.render import read_taste_record
from yap_studio.taste.store import (
    add_proposals,
    approve,
    load_proposals,
    load_taste,
    proposals_path,
    reject,
    revoke,
    taste_path,
)


def _line(rule: dict[str, Any], *, show_status: bool = False) -> str:
    text = f"{rule['id']} [{rule['scope']}] {rule['sentence']}"
    machine = rule["machine"]
    if machine is not None and machine["kind"] == "style_field":
        text += f" ({machine['field']} = {json.dumps(machine['value'], ensure_ascii=False)})"
    elif machine is not None and machine["kind"] == "filler_word":
        text += f" (filler word {json.dumps(machine['word'])}, {machine['position']})"
    elif machine is not None and machine["kind"] == "placement":
        text += f" (placement: {machine['layer']} {machine['rule']})"
    if any(source["type"] == "answer" for source in rule["sources"]):
        text += " (From your setup answers)"
    if show_status:
        text += f" [{rule['status']}]"
    return text


def _run(args: argparse.Namespace) -> int:
    """Run one taste command with the standard text or JSON error format."""
    exit_status = 0
    try:
        command = args.taste_command
        if command == "check":
            media = Path(args.video).expanduser()
            record = read_taste_record(media)
            if record is None:
                raise InputError(str(media), "no taste record next to this render")
            checked = check_taste(media, record, measure=measure_style)
            result = checked.to_dict()
            lines = [
                f"{row['id']} {row['field']}: {row['status'].replace('_', ' ')}"
                + (f" ({row['check']})" if row.get("check") else
                   f" ({row['reason']})" if row["reason"]
                   and row["reason"] != row["status"] else "")
                for row in checked.rules
            ]
            lines.extend(
                finding.message for finding in checked.findings
                if "rule_id" not in finding.evidence
            )
            text = "\n".join(lines) or checked.coverage["reason"] or "Taste rules passed."
            exit_status = int(any(finding.severity == "fail" for finding in checked.findings))
        elif command == "learn":
            learned = learn(
                [Path(p) for p in args.videos],
                Path(args.out),
                asr=args.asr,
                force=args.force,
                base=args.base,
                transcripts=[None if p == "-" else Path(p) for p in args.words]
                if args.words else None,
            )
            result = learned.to_dict()
            lines = [
                f"Style base: {result['style_base']}",
                "Measured style fields: " + (", ".join(result["measured_fields"]) or "None"),
                "Fields from base: " + (", ".join(result["base_fields"]) or "None"),
            ]
            for field, data in result["fields"].items():
                if data["status"] == "proposed":
                    frames = sum(len(row["frames"]) for row in data["per_video"])
                    count = len(data["per_video"])
                    total = len(result["videos"])
                    coverage = (
                        f"{count} of {total} videos"
                        if count < total
                        else f"{count} {'video' if count == 1 else 'videos'}"
                    )
                    lines.append(f"{field} = {data['value']} (from {frames} frames in {coverage})")
                else:
                    reason = data.get("reason", data.get("detail", {}).get("reason"))
                    lines.append(f"{field}: {data['status']}" + (f": {reason}" if reason else ""))
            lines.extend(f"Proposed rule: {sentence}" for sentence in result["proposed_rules"])
            for row in result["not_measured"]:
                label = (
                    f"Not measured in video {row['video'] + 1} ({row['name']})"
                    if "video" in row
                    else "Not measured"
                )
                lines.append(f"{label}: {row['field']}: {row['reason']}")
            lines.extend(result["paths"].values())
            if args.save:
                from yap_studio.taste.learned import save_learned

                saved = save_learned(learned)
                result["saved"] = saved
                style = saved["style"]
                lines.append(f"Saved style: {style['name']} ({style['id']}) from "
                             f"{style['videos']} finished "
                             f"{'video' if style['videos'] == 1 else 'videos'}. "
                             + ("It is your default." if style["default"] else
                                "It is not your default until you pick it."))
                lines.extend(f"Rule {item['id']}: {item['status'].replace('_', ' ')}"
                             for item in saved["rules"])
                if saved["rules"]:
                    lines.append("Approve rules with yap-studio taste approve ID.")
            text = "\n".join(lines)
        elif command == "show":
            taste, proposals = load_taste(), load_proposals()
            snapshot = taste.snapshot()
            rules = [rule for rule in taste.rules if args.all or rule["status"] == "approved"]
            pending = [
                item for item in proposals.proposals if args.all or item["status"] == "proposed"
            ]
            result = {
                "revision": taste.revision,
                "sha256": snapshot.sha256,
                "rules": rules,
                "proposals": pending,
                "history": taste.to_json()["history"],
                "paths": {"taste": str(taste_path()), "proposals": str(proposals_path())},
            }
            summary = (
                f"Taste revision {taste.revision} ({snapshot.sha256[:12]}), "
                f"{len(snapshot.rules)} approved, {len(proposals.pending)} proposed"
            )
            lines = [summary, *(_line(item, show_status=True) for item in (*rules, *pending))]
            if args.all:
                for entry in taste.history:
                    line = (
                        f"History revision {entry['revision']} at {entry['at']}: "
                        f"{entry['action']} {entry['rule_id']}"
                    )
                    if entry["by"]:
                        line += f" by {entry['by']}"
                    lines.append(line)
            text = "\n".join(lines)
        elif command == "add-proposal":
            before = {item["id"]: item for item in load_proposals().proposals}
            outcomes = add_proposals(Path(args.file))
            after = {item["id"]: item for item in load_proposals().proposals}
            result = [
                {
                    "id": item.id,
                    "status": "already_present"
                    if item.status == "merged" and before.get(item.id) == after.get(item.id)
                    else item.status,
                }
                for item in outcomes
            ]
            text = (
                "\n".join(f"{item['id']}: {item['status'].replace('_', ' ')}" for item in result)
                or "No proposals."
            )
        elif command == "propose":
            result = propose(Path(args.project), Path(args.file))
            text = "\n".join(f"{item['id']}: {item['status']}" for item in result)
        elif command == "approve":
            approval = approve(args.id)
            result = asdict(approval)
            text = f"Approved {args.id} at revision {approval.revision}."
            if approval.superseded_ids:
                text += f" Superseded {', '.join(approval.superseded_ids)}."
        elif command == "reject":
            result = reject(args.id)
            text = f"Rejected {args.id}."
        else:
            result = revoke(args.id)
            text = f"Revoked {args.id} at revision {result['ended_revision']}."
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        if args.taste_command == "check":
            return 2
        return 3 if isinstance(exc, TasteRefusal) or exc.code == "unsafe_path" else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(text)
    return exit_status


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the taste command and its decision subcommands."""
    parser = subparsers.add_parser("taste", help="inspect and decide creator taste preferences")
    commands = parser.add_subparsers(dest="taste_command", required=True)
    child = commands.add_parser("check", help="check the approved rules applied to a render")
    child.add_argument("video", metavar="VIDEO")
    child.add_argument("--json", action="store_true")
    child.set_defaults(func=_run)
    child = commands.add_parser(
        "learn", help="measure finished videos into proposed style and rules"
    )
    child.add_argument("videos", nargs="+", metavar="VIDEO")
    child.add_argument(
        "--words", action="append", metavar="FILE",
        help=("timed words file for one video, in video order, or - to use the speech engine "
              "for that video; repeat once per video"),
    )
    child.add_argument("--out", required=True, metavar="DIR")
    child.add_argument("--base", metavar="preset:ID|FILE", help="base for unmeasured style choices")
    child.add_argument(
        "--asr",
        choices=("auto", "none", "parakeet-mlx", "parakeet-onnx", "faster-whisper"),
        default="auto",
    )
    child.add_argument("--force", action="store_true")
    child.add_argument(
        "--save", action="store_true",
        help='save the proposed style as "Learned from your videos" (not the default) and '
        "queue its rules for approval",
    )
    child.add_argument("--json", action="store_true")
    child.set_defaults(func=_run)
    for command in ("show", "approve", "reject", "revoke", "add-proposal", "propose"):
        child = commands.add_parser(command)
        child.add_argument("--json", action="store_true")
        child.set_defaults(func=_run)
        if command == "show":
            child.add_argument("--all", action="store_true")
        elif command == "propose":
            child.add_argument("project", metavar="PROJECT")
            child.add_argument("file", metavar="FILE")
        elif command == "add-proposal":
            child.add_argument("file", metavar="FILE")
        else:
            child.add_argument("id", metavar="ID")
