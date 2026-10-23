"""Start and stop review servers that outlive their caller."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from yap_studio import background
from yap_studio.review.phone import link_label


def _url(document: dict) -> str:
    links = document.get("document") or {}
    phone = links.get("phone") or {}
    videos = links.get("videos") or []
    return phone.get("url") or links.get("list") or (videos[0]["url"] if videos else "")


def stop(media: list[str] | None, *, json_output: bool) -> int:
    """Stop matching detached reviews without touching foreground servers."""
    selected = [str(Path(item).resolve()) for item in media] if media is not None else None
    stopped: list[dict] = []
    failed = False
    for path in background.records("review"):
        document = background.read_record(path)
        if document is None:
            background.remove_record(path)
            continue
        if selected is not None and document.get("media") != selected:
            continue
        signaled, done = background.stop_record(path, document)
        if done is None:
            background.error(f"could not check review process {document['pid']}; try again",
                             "review_check_failed", json_output=json_output)
            failed = True
        elif signaled and done:
            stopped.append({"pid": document["pid"], "url": _url(document)})
        elif signaled:
            background.error(f"review process {document['pid']} did not stop; try again",
                             "review_stop_failed", json_output=json_output)
            failed = True
    if json_output:
        print(json.dumps({"stopped": stopped}, ensure_ascii=False))
    elif stopped:
        for item in stopped:
            print(f"Stopped the review at {item['url']}")
    elif not failed:
        print("No background review is running.")
    return 1 if failed else 0


def _restore(stopped: list[dict], *, json_output: bool) -> None:
    for previous in stopped:
        command = previous.get("command")
        if not isinstance(command, list) or not all(isinstance(part, str) for part in command):
            background.error(f"could not restore review process {previous['pid']}: command missing",
                             "review_restore_failed", json_output=json_output)
            continue
        try:
            child, record, log = background.spawn_record(command, kind="review")
        except OSError as exc:
            background.error(f"could not restore review process {previous['pid']}: {exc}",
                             "review_restore_failed", json_output=json_output)
            continue
        saved = background.ready(child, record, log, kind="review", label="review",
                                 json_output=json_output)
        if saved is not None:
            if json_output:
                print(json.dumps({"kept": {"pid": saved["pid"], "url": _url(saved)}},
                                 ensure_ascii=False), file=sys.stderr)
            else:
                print(f"kept the earlier review at {_url(saved)}", file=sys.stderr)


def start(args: argparse.Namespace) -> int:
    """Spawn a review and return when its links are ready."""
    media = [str(Path(value).resolve()) for value in args.media]
    stopped: list[dict] = []
    for path in background.records("review"):
        previous = background.read_record(path)
        if previous is None:
            background.remove_record(path)
        elif previous.get("media") == media and previous.get("versions") == args.versions:
            signaled, done = background.stop_record(path, previous)
            if done is None:
                background.error(f"could not check review process {previous['pid']}; try again",
                                 "review_check_failed", json_output=args.json)
                _restore(stopped, json_output=args.json)
                return 2
            if signaled and not done:
                background.error(f"review process {previous['pid']} did not stop; try again",
                                 "review_stop_failed", json_output=args.json)
                _restore(stopped, json_output=args.json)
                return 2
            if signaled:
                stopped.append(previous)

    command = [sys.executable, "-m", "yap_studio", "review", *media]
    if args.versions:
        command.append("--versions")
    if args.port is not None:
        command.extend(("--port", str(args.port)))
    if args.reviewer is not None:
        command.extend(("--reviewer", args.reviewer))
    for host in args.allow_host:
        command.extend(("--allow-host", host))
    if args.public_base is not None:
        command.extend(("--public-base", args.public_base))
    if args.phone:
        command.append("--phone")
    command.extend(("--json", "--no-open"))
    try:
        child, record, log = background.spawn_record(command, kind="review")
    except OSError as exc:
        background.error(f"could not start review: {exc}", "review_start_failed",
                         json_output=args.json)
        _restore(stopped, json_output=args.json)
        return 2
    saved = background.ready(child, record, log, kind="review", label="review",
                             json_output=args.json)
    if saved is None:
        _restore(stopped, json_output=args.json)
        return 2
    document = saved["document"]
    for line in background.log_lines(log):
        try:
            printed_document = json.loads(line)
        except ValueError:
            printed_document = None
        if line and printed_document != document:
            background.relay_child_line(line, json_output=args.json)
    if args.json:
        print(json.dumps({**document, "detached": {
            "pid": saved["pid"], "stop": "yap-studio review --stop",
        }}, ensure_ascii=False))
    else:
        phone = document.get("phone")
        url = _url(saved)
        if phone:
            print(f"Open on your phone: {link_label(phone['via'])}: {url}")
        elif document["list"] is None:
            print(f"Reviewing {document['videos'][0]['name']} at {url}")
        else:
            print(f"Review list at {url}")
            for item in document["videos"]:
                print(f"{item['name']}  {item['url']}")
        print("Running in the background. Stop it with: yap-studio review --stop")
    if not args.no_open:
        webbrowser.open(_url(saved))
    return 0


def posix_supported() -> bool:
    return background.posix_supported()


def write_record(path: Path, media: list[str], versions: bool, document: dict) -> None:
    """Publish the ready link atomically for the waiting parent."""
    background.write_record(path, document, media=media, versions=versions)
