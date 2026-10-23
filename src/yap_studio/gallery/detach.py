"""Keep one style page serving after its caller returns."""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path

from yap_studio import background
from yap_studio.review.phone import link_label


def _read_record(path: Path) -> dict | None:
    saved = background.read_record(path)
    pid = saved.get("pid") if saved is not None else None
    if type(pid) is not int or pid <= 0:
        background.remove_record(path)
        return None
    return saved


def _url(saved: dict) -> str | None:
    document = saved.get("document")
    return document.get("url") if isinstance(document, dict) else None


def show(document: dict, pid: int, *, json_output: bool, no_open: bool,
         reused: bool = False) -> None:
    detached = {"pid": pid, "stop": "yap-studio style pick --stop"}
    if reused:
        detached["reused"] = True
    if json_output:
        print(json.dumps({**document, "detached": detached}, ensure_ascii=False))
    else:
        phone = document.get("phone")
        if phone:
            print(f"Open on your phone: {link_label(phone['via'])}: {document['url']}")
        else:
            print(document["url"])
        print("Running in the background. Stop it with: yap-studio style pick --stop")
    if not no_open:
        webbrowser.open(document["url"])


def stop(*, json_output: bool) -> int:
    stopped = []
    failed = False
    for path in background.records("style-pick"):
        saved = _read_record(path)
        if saved is None:
            continue
        try:
            signaled, done = background.stop_record(path, saved)
        except OSError as exc:
            background.error(f"could not stop style page process {saved['pid']}: {exc}",
                             "style_pick_stop_failed", json_output=json_output)
            failed = True
            continue
        if done is None:
            background.error(f"could not check style page process {saved['pid']}; try again",
                             "style_pick_check_failed", json_output=json_output)
            failed = True
        elif signaled and done:
            stopped.append({"pid": saved["pid"], "url": _url(saved)})
        elif signaled:
            background.error(f"style page process {saved['pid']} did not stop; try again",
                             "style_pick_stop_failed", json_output=json_output)
            failed = True
    if json_output:
        print(json.dumps({"stopped": stopped}))
    elif stopped:
        for item in stopped:
            print(f"Stopped the style page at {item['url']}")
    elif not failed:
        print("No background style page is running.")
    return 1 if failed else 0


def start(args: argparse.Namespace) -> int:
    import fcntl

    try:
        folder = background.directory()
        folder.mkdir(parents=True, mode=0o700, exist_ok=True)
        folder.chmod(0o700)
        fd = os.open(folder / "style-pick.lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as lock:
            os.fchmod(lock.fileno(), 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            return _start(args)
    except OSError as exc:
        background.error(f"could not start style page: {exc}", "style_pick_start_failed",
                         json_output=args.json)
        return 2


def _stop_page(path: Path, saved: dict, *, json_output: bool) -> bool | None:
    """Return whether a page was stopped, or None on failure."""
    try:
        signaled, done = background.stop_record(path, saved)
    except OSError as exc:
        background.error(f"could not stop style page process {saved['pid']}: {exc}",
                         "style_pick_stop_failed", json_output=json_output)
        return None
    if done is None or not done:
        code = "style_pick_check_failed" if done is None else "style_pick_stop_failed"
        background.error(f"could not stop style page process {saved['pid']}; try again",
                         code, json_output=json_output)
        return None
    return signaled


def _launch(command: list[str], *, cwd: str | None, json_output: bool,
            restoring: bool = False) -> tuple[dict, Path, Path] | None:
    """Start a page and wait for its readiness record."""
    try:
        child, record, log = background.spawn_record(command, kind="style-pick", cwd=cwd)
    except OSError as exc:
        action = "restore" if restoring else "start"
        background.error(f"could not {action} style page: {exc}", f"style_pick_{action}_failed",
                         json_output=json_output)
        return None
    saved = background.ready(child, record, log, kind="style_pick", label="style page",
                             json_output=json_output)
    return (saved, record, log) if saved is not None else None


def _restore(stopped: list[dict], *, json_output: bool) -> None:
    """Relaunch pages displaced by a fixed-port replacement."""
    for previous in stopped:
        command = previous.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            background.error(f"could not restore style page process {previous['pid']}: "
                             "command missing",
                             "style_pick_restore_failed", json_output=json_output)
            continue
        launched = _launch(command, cwd=previous.get("cwd"), json_output=json_output,
                           restoring=True)
        if launched is not None:
            saved, _record, _log = launched
            if json_output:
                print(json.dumps({"kept": {"pid": saved["pid"], "url": _url(saved)}},
                                 ensure_ascii=False), file=sys.stderr)
            else:
                print(f"kept the earlier style page at {_url(saved)}", file=sys.stderr)


def _port(command: list[str] | None) -> int | None:
    if not isinstance(command, list):
        return None
    try:
        return int(command[command.index("--port") + 1]) or None
    except (IndexError, TypeError, ValueError):
        return None


def _start(args: argparse.Namespace) -> int:
    command = [sys.executable, "-m", "yap_studio", "style", "pick"]
    for name in ("inbox", "still", "workspace"):
        if value := getattr(args, name):
            command.extend((f"--{name}", str(Path(value).expanduser().resolve())))
    for name in ("style", "host", "port"):
        if (value := getattr(args, name)) is not None:
            command.extend((f"--{name}", str(value)))
    if args.phone:
        command.append("--phone")
    command.extend(("--json", "--no-open"))
    reuse = None
    previous = []
    for path in background.records("style-pick"):
        saved = _read_record(path)
        if saved is None:
            continue
        matches = background.command_matches(saved.get("pid"), path)
        if matches is None:
            background.error(f"could not check style page process {saved['pid']}; try again",
                             "style_pick_check_failed", json_output=args.json)
            return 2
        if not matches:
            background.remove_record(path)
        elif (saved.get("command") == command and saved.get("cwd") == os.getcwd()
              and isinstance(_url(saved), str) and _url(saved)):
            if reuse is None:
                reuse = saved
            else:
                previous.append((path, saved))
        else:
            previous.append((path, saved))
    if reuse is not None:
        for path, saved in previous:
            if _stop_page(path, saved, json_output=args.json) is None:
                return 2
        show(reuse["document"], reuse["pid"], json_output=args.json, no_open=args.no_open,
             reused=True)
        return 0
    stopped = []
    deferred = []
    for path, saved in previous:
        if args.port and _port(saved.get("command")) == args.port:
            signaled = _stop_page(path, saved, json_output=args.json)
            if signaled is None:
                _restore(stopped, json_output=args.json)
                return 2
            if signaled:
                stopped.append(saved)
        else:
            deferred.append((path, saved))
    launched = _launch(command, cwd=os.getcwd(), json_output=args.json)
    if launched is None:
        _restore(stopped, json_output=args.json)
        return 2
    saved, record, log = launched
    for path, earlier in deferred:
        signaled = _stop_page(path, earlier, json_output=args.json)
        if signaled is None:
            if _stop_page(record, saved, json_output=args.json) is not None:
                _restore(stopped, json_output=args.json)
            return 2
        if signaled:
            stopped.append(earlier)
    for line in background.log_lines(log):
        try:
            printed = json.loads(line)
        except ValueError:
            printed = None
        if line and printed != saved["document"]:
            background.relay_child_line(line, json_output=args.json)
    show(saved["document"], saved["pid"], json_output=args.json, no_open=args.no_open)
    return 0
