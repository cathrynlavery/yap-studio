"""Project version and feedback batch commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from yap_studio.errors import InputError, ValidationError, YapError


def _versions(args: argparse.Namespace) -> int:
    """Manage immutable project versions using the command error convention."""
    try:
        from yap_studio.feedback.versions import load_versions, save_version

        if args.version_command == "save":
            manifest = save_version(
                Path(args.project), media=Path(args.media) if args.media else None,
                note=args.note, layers=Path(args.layers) if args.layers else None,
            )
            if args.json:
                print(json.dumps(manifest.to_dict(), ensure_ascii=False))
            else:
                taste = manifest.snapshots["taste"]
                suffix = f" with taste revision {taste['revision']}" if taste is not None else ""
                print(f"Saved {manifest.id} from {manifest.media['source']} "
                      f"({manifest.media['frame_count']} frames) in versions/{manifest.id}{suffix}")
        elif args.version_command in {"rollback", "approve"}:
            from yap_studio.feedback.release import approve, rollback

            action = rollback if args.version_command == "rollback" else approve
            result = action(Path(args.project), args.version)
            if args.json:
                print(json.dumps(result, ensure_ascii=False))
            elif args.version_command == "rollback":
                print(f"Current version: {result['version']}. Run cut again before saving.")
                if result["not_restored"]:
                    print(f"Not restored: {', '.join(result['not_restored'])}")
            else:
                print(f"Approved {result['version']} ({result['report_status'] or 'no status'})")
        elif args.version_command == "current":
            from yap_studio.feedback.release import approved_version
            from yap_studio.feedback.versions import current_version

            current = current_version(Path(args.project))
            approved = approved_version(Path(args.project))
            result = {"current": current.id if current else None,
                      "approved": approved.id if approved else None}
            print(json.dumps(result) if args.json else
                  f"Current: {result['current'] or 'none'}\n"
                  f"Approved: {result['approved'] or 'none'}")
        else:
            versions = load_versions(Path(args.project))
            if args.json:
                print(json.dumps([item.to_dict() for item in versions], ensure_ascii=False))
            else:
                for item in versions:
                    print(f"{item.id}  {item.created_at}  {item.media['sha256'][:12]}  "
                          f"{item.note or ''}")
        return 0
    except YapError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)
        from yap_studio.feedback.versions import PilotVersionError

        return 3 if isinstance(exc, PilotVersionError) else 2


def _feedback(args: argparse.Namespace) -> int:
    """Build feedback or report keep outcomes with stable command exit codes."""
    try:
        if args.feedback_command == "apply":
            from yap_studio.feedback.apply import apply_operations

            try:
                operations = json.loads(Path(args.operations).read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise InputError(args.operations, "cannot read valid JSON") from exc
            try:
                result = apply_operations(Path(args.project), operations, dry_run=args.dry_run)
            except YapError:
                raise
            except Exception as exc:
                from yap_studio.feedback.batch import FeedbackError

                raise FeedbackError(
                    f"Could not apply feedback operations: {exc}",
                    suggested_action="Fix the reported error and rerun the same operations file.",
                ) from exc
            if args.json:
                print(json.dumps(result, ensure_ascii=False))
            else:
                for change in result["changes"]:
                    print(change["summary"])
                count = len(result["changes"])
                notes = len({change["comment"] for change in result["changes"]})
                verb = "Would apply" if args.dry_run else "Applied"
                next_step = "" if args.dry_run else f" Next: yap-studio cut {args.project}"
                print(f"{verb} {count} operations for {notes} notes in {result['batch']}."
                      f"{next_step}")
            return 0
        if args.feedback_command in ("reply", "replies"):
            from yap_studio.feedback.replies import reply_status, write_reply
            from yap_studio.review.series import reply_to_dict
            from yap_studio.review.store import ReviewStore, data_dir

            store = ReviewStore(data_dir())
            if args.feedback_command == "reply":
                ref = args.comment.rsplit(":", 1)
                if len(ref) != 2 or not all(ref):
                    raise ValidationError("comment", "must be VERSION:COMMENT, for example v1:c3")
                reply = write_reply(Path(args.project), store, args.batch, *ref,
                                    args.status, args.text, version_id=args.version)
                print(json.dumps(reply_to_dict(reply), ensure_ascii=False) if args.json else
                      f"Replied to {args.comment} in {reply.evidence['version']}: {reply.status}")
                return 0
            result = reply_status(Path(args.project), store, args.batch)
            if args.json:
                print(json.dumps(result, ensure_ascii=False))
            else:
                for item in result["items"]:
                    status = item["reply"]["status"] if item["reply"] is not None else "missing"
                    comment = item["comment"]
                    first_line = item["text"].splitlines()[0]
                    print(f"{status} {comment['version']}:{comment['id']} "
                          f"{item['kind']}: {first_line}")
                    if item["reply"] is None and item["suggested"] is not None:
                        print(f"  suggested: {item['suggested']}")
            return 0 if result["complete"] else 3
        if args.feedback_command in ("ask", "answer"):
            from yap_studio.feedback.clarify import answer_clarification, ask_clarification

            ref = args.comment.split(":")
            if len(ref) != 2 or not all(ref):
                raise ValidationError("comment", "must be VERSION:COMMENT, for example v2:c3")
            if args.feedback_command == "ask":
                options = []
                for raw in args.option:
                    value, separator, label = raw.partition("=")
                    if not separator:
                        raise ValidationError("option", "must be VALUE=LABEL")
                    options.append({"value": value, "label": label})
                batch = ask_clarification(Path(args.project), args.batch, *ref, args.question,
                                          options=options)
            else:
                batch = answer_clarification(Path(args.project), args.batch, *ref,
                                             answer=args.answer, option=args.option)
            print(json.dumps(batch, ensure_ascii=False) if args.json else
                  f"{'Asked' if args.feedback_command == 'ask' else 'Answered'} "
                  f"{args.comment} in {batch['id']}")
            return 0
        if args.feedback_command == "build":
            from yap_studio.feedback.batch import build_batch

            try:
                export = json.loads(Path(args.export).read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise InputError(args.export, "cannot read valid JSON") from exc
            batch = build_batch(Path(args.project), export,
                                layers=Path(args.layers) if args.layers else None)
            if args.json:
                print(json.dumps(batch, ensure_ascii=False))
            else:
                items = batch["items"]
                fixes = sum(item["comment"]["kind"] == "fix" for item in items)
                carried = sum(item["carried"] for item in items)
                print(f"Wrote feedback/{batch['id']}.json: {len(items)} comments "
                      f"({fixes} fix, {len(items) - fixes} keep), "
                      f"{carried} carried, {len(batch['constraints'])} constraints")
            return 0
        from yap_studio.feedback.keeps import check_version

        report = check_version(Path(args.project), batch_id=args.batch, version_id=args.version)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False))
        else:
            for result in report.results:
                print(f"{result.status} {result.comment['version'] or 'cut'} "
                      f"{result.comment['id']}: {result.detail}")
        return 0 if report.ok else 3
    except YapError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2


def _outline(args: argparse.Namespace) -> int:
    """Build stories, send creator notes, and report the pre-render gate."""
    try:
        if args.outline_command == "review":
            return _outline_review(args)
        from yap_studio.feedback.outline import build_outline, outline_status, send_notes

        root = Path(args.project)
        if args.outline_command == "build":
            result = build_outline(root, summary=args.summary)
            if args.json:
                print(json.dumps(result, ensure_ascii=False))
            else:
                for beat in result["beats"]:
                    print(f"{beat['role']} {beat['id']}: {beat['text'][:80]}")
                print(f"Left out: {len(result['left_out'])}")
            return 0
        if args.outline_command == "send":
            result = send_notes(root)
            print(json.dumps(result, ensure_ascii=False) if args.json else
                  f"Wrote feedback/{result['id']}.json: {len(result['items'])} outline notes")
            return 0
        result = outline_status(root)
        print(json.dumps(result, ensure_ascii=False) if args.json else result["state"])
        return 0 if result["state"] in ("approved", "timed_out") else 3
    except YapError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2


def _outline_review(args: argparse.Namespace) -> int:
    """Open an existing story outline without building or rendering anything."""
    root = Path(args.project)
    if not (root / "outline" / "outline.json").is_file():
        raise YapError("No story outline exists.", error_type="input_error",
                       code="outline_missing",
                       suggested_action="Run yap-studio outline build PROJECT first.")
    from yap_studio.review.outline_server import OutlineServer

    try:
        server = OutlineServer(root, port=args.port)
    except OSError as exc:
        raise YapError("Cannot start the story outline server.", error_type="input_error",
                       code="outline_server_unavailable",
                       suggested_action="Check the port and run outside the sandbox.") from exc
    try:
        if args.json:
            print(json.dumps({"url": server.url}), flush=True)
        else:
            print(f"Story outline for {server.project} at {server.url}", flush=True)
            print("Press Ctrl-C to stop.", flush=True)
        if not args.no_open:
            import webbrowser

            webbrowser.open(server.url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        server.server_close()
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register version and feedback commands without importing the main CLI."""
    outline = subparsers.add_parser("outline", help="review the story before rendering")
    commands = outline.add_subparsers(dest="outline_command", required=True)
    for name in ("build", "status", "send", "review"):
        command = commands.add_parser(name, help=f"{name} the story outline")
        command.add_argument("project", metavar="PROJECT")
        if name == "build":
            command.add_argument("--summary", metavar="TEXT")
        if name == "review":
            command.add_argument("--port", type=int, default=0, metavar="N")
            command.add_argument("--no-open", action="store_true")
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_outline)
    versions = subparsers.add_parser("versions", help="save and list immutable project versions")
    commands = versions.add_subparsers(dest="version_command", required=True)
    save = commands.add_parser("save", help="save the current verified cut for review")
    save.add_argument("project", metavar="PROJECT")
    save.add_argument("--media", metavar="FILE")
    save.add_argument("--note", metavar="TEXT")
    save.add_argument("--layers", metavar="FILE")
    save.add_argument("--json", action="store_true")
    save.set_defaults(func=_versions)
    listing = commands.add_parser("list", help="list saved project versions")
    listing.add_argument("project", metavar="PROJECT")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=_versions)
    for name in ("rollback", "approve", "current"):
        command = commands.add_parser(name, help=f"{name} a saved project version")
        command.add_argument("project", metavar="PROJECT")
        if name != "current":
            command.add_argument("version", metavar="vK")
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_versions)
    feedback = subparsers.add_parser("feedback", help="build feedback batches and check keeps")
    commands = feedback.add_subparsers(dest="feedback_command", required=True)
    build = commands.add_parser("build", help="locate review comments in their render snapshots")
    build.add_argument("project", metavar="PROJECT")
    build.add_argument("export", metavar="EXPORT")
    build.add_argument("--layers", metavar="FILE")
    build.add_argument("--json", action="store_true")
    build.set_defaults(func=_feedback)
    apply = commands.add_parser("apply", help="apply feedback operations once to their saved base")
    apply.add_argument("project", metavar="PROJECT")
    apply.add_argument("operations", metavar="OPERATIONS.json")
    apply.add_argument("--dry-run", action="store_true")
    apply.add_argument("--json", action="store_true")
    apply.set_defaults(func=_feedback)
    check = commands.add_parser("check", help="check scoped keeps against a new cut")
    check.add_argument("project", metavar="PROJECT")
    check.add_argument("--batch", metavar="ID")
    check.add_argument("--version", metavar="ID")
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=_feedback)
    reply = commands.add_parser("reply", help="reply with evidence from the next saved version")
    reply.add_argument("project", metavar="PROJECT")
    reply.add_argument("--batch", required=True, metavar="ID")
    reply.add_argument("--comment", required=True, metavar="VERSION:COMMENT")
    reply.add_argument("--status", required=True,
                       choices=("changed", "kept", "declined", "needs_clarification"))
    reply.add_argument("--text", required=True, metavar="TEXT")
    reply.add_argument("--version", metavar="vN")
    reply.add_argument("--json", action="store_true")
    reply.set_defaults(func=_feedback)
    replies = commands.add_parser("replies", help="list replies and missing batch responses")
    replies.add_argument("project", metavar="PROJECT")
    replies.add_argument("--batch", metavar="ID")
    replies.add_argument("--json", action="store_true")
    replies.set_defaults(func=_feedback)
    for name in ("ask", "answer"):
        command = commands.add_parser(name, help=f"{name} a clarification question")
        command.add_argument("project", metavar="PROJECT")
        command.add_argument("--batch", required=True, metavar="ID")
        command.add_argument("--comment", required=True, metavar="VERSION:COMMENT")
        command.add_argument("--json", action="store_true")
        if name == "ask":
            command.add_argument("--question", required=True, metavar="TEXT")
            command.add_argument("--option", action="append", default=[], metavar="VALUE=LABEL")
        else:
            choice = command.add_mutually_exclusive_group(required=True)
            choice.add_argument("--answer", metavar="TEXT")
            choice.add_argument("--option", metavar="VALUE")
        command.set_defaults(func=_feedback)
