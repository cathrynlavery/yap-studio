"""Readiness, reuse and explicit stop for detached home pages."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from yap_studio import background
from yap_studio.errors import YapError
from yap_studio.homepage.bridge import private_dir, writer
from yap_studio.review.phone import link_label


def show(
    document: dict,
    *,
    json_output: bool,
    no_open: bool,
    pid: int | None = None,
    reused: bool = False,
) -> None:
    """Print the local and optional phone links, then optionally open the local page."""
    result = {**document}
    if pid is not None:
        result["detached"] = {"pid": pid, "stop": "yap-studio home --stop"}
        if reused:
            result["detached"]["reused"] = True
            result["reused"] = True
    if json_output:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        print(document["url"], flush=True)
        if document.get("phone"):
            phone = document["phone"]
            print(f"Open on your phone: {link_label(phone['via'])}: {phone['url']}", flush=True)
        print(
            "Running in the background. Stop it with: yap-studio home --stop"
            if pid is not None
            else "Press Ctrl-C to stop.",
            flush=True,
        )
    if not no_open:
        webbrowser.open(document["url"])


def stop(*, workspace: Path | None, json_output: bool) -> int:
    """Stop all detached home pages, or those serving one workspace."""
    stopped, failed = [], False
    for record in background.records("home"):
        saved = background.read_record(record)
        if saved is None:
            background.remove_record(record)
            continue
        if workspace is not None and saved.get("workspace") != str(workspace):
            continue
        try:
            signaled, done = background.stop_record(record, saved)
            if done is not True:
                raise OSError("The home page did not stop. Try again.")
            if signaled:
                stopped.append({"pid": saved["pid"], "url": saved.get("document", {}).get("url")})
        except OSError:
            background.error(
                "The home page could not stop. Try again.",
                "home_stop_failed",
                json_output=json_output,
            )
            failed = True
    if json_output:
        print(json.dumps({"stopped": stopped}))
    else:
        print(
            f"Stopped {len(stopped)} home page(s)."
            if stopped
            else "No background home page is running."
        )
    return 1 if failed else 0


def start(args: argparse.Namespace, workspace: Path) -> int:
    """Serialize launches and reuse a ready home with the same workspace and options."""
    folder = background.directory()
    folder.mkdir(parents=True, mode=0o700, exist_ok=True)
    private_dir(folder)
    # A launch lock serializes callers from different working directories without holding the
    # bridge lock the starting server needs.
    with writer(workspace, launch=True):
        command = [sys.executable, "-m", "yap_studio", "home", "--workspace", str(workspace)]
        for name in ("host", "port"):
            if (value := getattr(args, name)) is not None:
                command.extend((f"--{name}", str(value)))
        if args.phone:
            command.append("--phone")
        command.extend(("--json", "--no-open"))
        previous = []
        for record in background.records("home"):
            saved = background.read_record(record)
            if saved is None or saved.get("workspace") != str(workspace):
                continue
            matches = background.command_matches(saved.get("pid"), record)
            if matches is None:
                raise YapError(
                    "The running home page could not be checked. Try again.",
                    code="home_check_failed",
                )
            if not matches:
                background.remove_record(record)
            elif saved.get("command") == command:
                show(
                    saved["document"],
                    json_output=args.json,
                    no_open=args.no_open,
                    pid=saved["pid"],
                    reused=True,
                )
                return 0
            else:
                previous.append((record, saved))
        # A fixed port must be released before starting its replacement.
        displaced = []
        for record, saved in previous:
            old_command = saved.get("command", [])
            old_port = (
                old_command[old_command.index("--port") + 1] if "--port" in old_command else None
            )
            if args.port and old_port == str(args.port):
                _, done = background.stop_record(record, saved)
                if done is not True:
                    raise YapError(
                        "The earlier home page could not stop. Try again.", code="home_stop_failed"
                    )
                displaced.append(saved)
        child, record, log = background.spawn_record(command, kind="home", cwd=str(workspace))
        saved = background.ready(
            child, record, log, kind="home", label="home page", json_output=args.json
        )
        if saved is None:
            for earlier in displaced:
                restored, ready_record, ready_log = background.spawn_record(
                    earlier["command"], kind="home", cwd=earlier.get("cwd")
                )
                background.ready(
                    restored,
                    ready_record,
                    ready_log,
                    kind="home",
                    label="home page",
                    json_output=args.json,
                )
            return 2
        for earlier_record, earlier in previous:
            if earlier in displaced:
                continue
            _, done = background.stop_record(earlier_record, earlier)
            if done is not True:
                background.stop_record(record, saved)
                raise YapError(
                    "The earlier home page could not stop. Try again.", code="home_stop_failed"
                )
        show(saved["document"], json_output=args.json, no_open=args.no_open, pid=saved["pid"])
    return 0
