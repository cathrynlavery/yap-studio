"""Local product feedback and explicit, scrubbed upstream reports."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from yap_studio import __version__, paths
from yap_studio.errors import ValidationError, YapError
from yap_studio.review.store import data_dir
from yap_studio.style.library import _config, _save_lock, config_root
from yap_studio.style.loader import SCHEMA_PATH

MODES = ("ask", "always", "never")
KINDS = ("bug", "confusing", "slow", "missing")
REPO = "cathrynlavery/yap-studio"
# Exact style setting names, such as music.level, stay readable in reports.
STYLE_SETTINGS = frozenset(
    f"{section}.{key}"
    for section, spec in json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["properties"].items()
    for key in spec.get("properties", {})
)


def scrub(text: str) -> str:
    """Remove recognizable private values before truncating, including on re-send."""
    text = "".join(char for char in unicodedata.normalize("NFKC", text)
                   if char.isprintable() or char in "\n\r\t")
    text = text.replace(str(Path.home()), "[path]")

    def domain(match: re.Match) -> str:
        try:
            host = urlsplit(match[0] if "://" in match[0] else "https://" + match[0]).hostname
        except ValueError:
            host = None
        return host if host in {"github.com", "yapstudio.app"} else "[url]"

    text = re.sub(r"[a-z][a-z0-9+.-]*://[^\s<>\"']+", domain, text, flags=re.I)
    text = re.sub(r"\b(?:www\.[\w.-]+|[\w.-]+\.[a-z]{2,}[/?#])[^\s<>\"']*",
                  domain, text, flags=re.I)
    # Consume spaces too: an unquoted Mac path can include several private names.
    text = re.sub(r"(?:\[path\]|/(?:Users|Volumes|private)(?:/|\b)|~/)[^\n\r\"'<>;,]*",
                  "[path]", text, flags=re.I)
    text = re.sub(r"\b[\w.+%-]+@[\w.-]+\.[a-z]{2,}\b", "[email]", text, flags=re.I)
    text = re.sub(r"\b(?:[\w-]+\.)+(?:ts\.net|local)\b", "[host]", text, flags=re.I)

    def ipv6(match: re.Match) -> str:
        try:
            address = match[0].rstrip(".")
            ipaddress.IPv6Address(address)
            return "[ip]" + match[0][len(address):]
        except ValueError:
            return match[0]

    text = re.sub(r"(?<![\w:])(?:[0-9a-f]{0,4}:){2,}[0-9a-f:.]*(?:%\w+)?",
                  ipv6, text, flags=re.I)
    text = re.sub(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", "[ip]", text)
    text = re.sub(r"\b(?:bearer\s+\S+|(?:api[ _-]?key|token|secret|password)"
                  r"[\"']?\s*[:=]\s*[\"']?[^\s,;\"'}]+)",
                  "[secret]", text, flags=re.I)
    text = re.sub(r"\b(?:sk[-_]|gh[pousr]_|github_pat_|xox[baprs]-|AKIA)[\w-]+",
                  "[secret]", text, flags=re.I)
    text = re.sub(r"(?<![\w/+.-])[a-z0-9_/+.-]{32,}={0,2}(?![\w/+.-])",
                  lambda m: m[0] if m[0].casefold() in STYLE_SETTINGS else "[secret]",
                  text, flags=re.I)
    public_names = {
        "github.com", "yapstudio.app", "e.g", "i.e",
        "project.json", "edit.json", "removals.json", "cut-words.json", "cut.mp4", "final.mp4",
        "text-hook.json", "cards.json", "faces.json", "layout.json", "progress.json",
        "answers.json", "new_project.py", "setup.py", "style.json", "words.json",
        "transcript.json", "phrases.json", "spans.json", "receipt.json", "report.json",
        "verify.json",
    } | STYLE_SETTINGS
    text = re.sub(r"(?<!\w)[\w@~-]+(?:[.-][\w@~-]+)*\.[a-z][a-z0-9]{0,15}\b",
                  lambda m: m[0] if m[0].casefold() in public_names else "[file]",
                  text, flags=re.I)
    text = re.sub(r"(?<!\w)\.[a-z][\w.-]*", "[file]", text, flags=re.I)
    return " ".join(text.split())[:500]


def fingerprint(kind: str, step: str | None, code: str | None, text: str,
                origin: str | None = None, checks: Sequence[str] = ()) -> str:
    """Group equivalent scrubbed notes independently of time and machine."""
    values = [kind, step or "", code or "", " ".join(scrub(text).casefold().split())]
    if origin:
        values.append(re.sub(r":[0-9]+(?= |$)", "", origin))
    values.extend(checks)
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def mode() -> str:
    """Read consent, with the explicit owner environment override first."""
    if os.environ.get("YAP_STUDIO_FEEDBACK_AUTO_SEND", "").strip().lower() in {"1", "true", "yes"}:
        return "always"
    value = _config().get("feedback_mode", "ask")
    if value not in MODES:
        raise ValidationError("feedback_mode", "choose ask, always or never")
    return value


def setting(value: str) -> None:
    """Save consent alongside the existing config without replacing other choices."""
    with _save_lock(), paths.operation():
        config = _config()
        config["feedback_mode"] = value
        paths.atomic_write_text(config_root() / "config.json", json.dumps(config) + "\n")


def _gh(*arguments: str) -> str:
    result = subprocess.run(["gh", *arguments], capture_output=True, text=True,
                            timeout=15, check=True, env={**os.environ, "GH_HOST": "github.com"})
    return result.stdout


def availability() -> dict:
    """Report transport readiness without exposing the endpoint or account."""
    gh = shutil.which("gh") is not None
    logged_in = False
    if gh:
        try:
            _gh("auth", "status", "--hostname", "github.com")
            logged_in = True
        except (OSError, subprocess.SubprocessError):
            pass
    return {"mode": mode(), "endpoint": bool(os.environ.get("YAP_STUDIO_FEEDBACK_ENDPOINT")),
            "gh": gh, "gh_logged_in": logged_in}


@contextmanager
def _lock():
    folder = data_dir()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(folder / ".feedback.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _safe(entry: dict) -> dict:
    # Only these fields may leave the machine, even if the ledger was edited.
    if not isinstance(entry, dict):
        raise ValueError("invalid feedback entry")
    result = {key: scrub(str(entry[key])) if entry.get(key) is not None else None
              for key in ("time", "version", "macos", "arch", "step", "code",
                          "kind", "text", "agent")}
    origin = entry.get("origin")
    result["origin"] = origin if isinstance(origin, str) and re.fullmatch(
        r"[a-z0-9_]+(/[a-z0-9_]+)*:[A-Za-z0-9_<>]+:[0-9]+( [A-Za-z][A-Za-z0-9_]*)?",
        origin) else None
    checks = entry.get("checks")
    result["checks"] = [check for check in checks if isinstance(check, str) and re.fullmatch(
        r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*", check
    )][:20] if isinstance(checks, list) else []
    result["fingerprint"] = fingerprint(result["kind"], result["step"],
                                        result["code"], result["text"], result["origin"],
                                        result["checks"])
    result["sent"] = entry.get("sent") is True
    return result


def entries(*, pending: bool = False) -> list[dict]:
    """Read the entire ledger, refusing corruption rather than losing notes."""
    try:
        lines = (data_dir() / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    result = [_safe(json.loads(line)) for line in lines if line.strip()]
    return [entry for entry in result if not pending or not entry["sent"]]


def record(text: str, *, kind: str = "confusing", step: str | None = None,
           code: str | None = None, origin: str | None = None, checks: Sequence[str] = ()) -> dict:
    """Append one scrubbed entry with owner-only permissions before any send."""
    entry = _safe({"time": datetime.now(UTC).isoformat(), "version": __version__,
                   "macos": platform.mac_ver()[0], "arch": platform.machine(),
                   "step": step, "code": code, "kind": kind, "text": text, "origin": origin,
                   "checks": list(checks), "agent": os.environ.get("AGENT_ID"), "sent": False})
    with _lock():
        fd = os.open(data_dir() / "feedback.jsonl",
                     os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return entry


def _save(entries: list[dict]) -> None:
    folder = data_dir()
    fd, temporary = tempfile.mkstemp(prefix=".feedback-", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for entry in entries:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, folder / "feedback.jsonl")
    finally:
        Path(temporary).unlink(missing_ok=True)


def _groups(notes: list[dict]) -> list[dict]:
    groups = {}
    for entry in notes:
        key = entry["fingerprint"]
        if key not in groups:
            groups[key] = {"fingerprint": key, "count": 0, "entries": []}
        groups[key]["count"] += 1
        groups[key]["entries"].append(entry)
    return list(groups.values())


def _issue(group: dict) -> dict:
    entry = group["entries"][-1]
    title = f"{entry['kind']}: {entry['step'] or 'feedback'} {entry['code'] or ''}".strip()[:100]
    body = "\n".join([
        "## Product feedback", "", entry["text"], "", "## Environment", "",
        f"Time (UTC): {entry['time']}", f"Yap Studio version: {entry['version']}",
        f"macOS version: {entry['macos']}", f"CPU arch: {entry['arch']}",
        f"Step: {entry['step'] or ''}", f"Code: {entry['code'] or ''}",
        f"Where: {entry['origin'] or ''}",
        f"Checks: {', '.join(entry['checks'])}",
        f"Kind: {entry['kind']}", f"Agent: {entry['agent'] or ''}",
        f"Count: {group['count']}", "", f"Fingerprint: {group['fingerprint']}",
    ])
    return {"title": title, "body": body, "label": "feedback"}


def _link(issue: dict) -> str:
    base = f"https://github.com/{REPO}/issues/new?"
    body = issue["body"]
    while True:
        link = base + urlencode({"template": "feedback.md", "title": issue["title"], "body": body})
        if len(link) < 8000:
            return link
        # Keep the fingerprint when shortening unusually expensive URL encoding.
        body = body[:len(body) // 2] + "\n\n" + issue["body"].splitlines()[-1]


def _plan(group: dict, state: dict) -> dict:
    if state["endpoint"]:
        return {"transport": "endpoint", "method": "POST", "json": group}
    issue = _issue(group)
    if state["gh_logged_in"]:
        entry = group["entries"][-1]
        fingerprints = {group["fingerprint"]}
        if entry["origin"]:
            fingerprints.add(fingerprint(entry["kind"], entry["step"],
                                         entry["code"], entry["text"], checks=entry["checks"]))
        # Walk every page; a full page is never evidence that the search is complete.
        issues = json.loads(_gh("api", "--paginate", "--slurp", "-X", "GET",
                                f"repos/{REPO}/issues?state=all&per_page=100"))
        for item in sorted((item for page in issues for item in page),
                           key=lambda item: item.get("state") != "open"):
            if ("pull_request" not in item
                    and not item.get("locked")
                    and any(value in (item.get("body") or "") for value in fingerprints)):
                body = f"seen again in {entry['version']}, {group['count']} times"
                if entry["origin"]:
                    body += f"\nWhere: {entry['origin']}"
                if entry["checks"]:
                    body += f"\nChecks: {', '.join(entry['checks'])}"
                return {"transport": "gh", "action": "comment", "issue": item["number"],
                        "body": body}
        return {"transport": "gh", "action": "create", **issue}
    return {"transport": "link", "url": _link(issue), **issue}


def _post(plan: dict) -> bool:
    if plan["transport"] == "link":
        return False
    if plan["transport"] == "endpoint":
        request = Request(os.environ["YAP_STUDIO_FEEDBACK_ENDPOINT"],
                          data=json.dumps(plan["json"]).encode(), method="POST",
                          headers={"Content-Type": "application/json",
                                   "User-Agent": "yap-studio/feedback"})
        with urlopen(request, timeout=15) as response:
            if not 200 <= response.status < 300:
                raise OSError("feedback endpoint refused the report")
        return True
    with tempfile.TemporaryDirectory(prefix="yap-feedback-") as directory:
        body = Path(directory) / "body.txt"
        body.write_text(plan["body"], encoding="utf-8")
        body.chmod(0o600)
        if plan["action"] == "comment":
            _gh("issue", "comment", str(plan["issue"]), "--repo", REPO, "--body-file", str(body))
        else:
            _gh("issue", "create", "--repo", REPO, "--title", plan["title"],
                "--label", "feedback", "--body-file", str(body))
    return True


def send(*, dry_run: bool = False, yes: bool = False) -> list[dict]:
    """Send pending groups once, saving each successful acknowledgement."""
    consent = mode()
    if not dry_run and consent == "never":
        raise ValidationError("feedback", "sharing is never; entries stay local")
    if not dry_run and consent == "ask" and not yes:
        raise ValidationError("feedback", "show the person feedback send --dry-run first; "
                              "send with --yes only after they agree")
    state = availability()
    if dry_run:
        return [_plan(group, state) for group in _groups(entries(pending=True))]
    results = []
    with _lock():
        notes = entries()
        for group in _groups([entry for entry in notes if not entry["sent"]]):
            try:
                plan = _plan(group, state)
                sent = _post(plan)
            except (OSError, ValueError, subprocess.SubprocessError):
                results.append({"fingerprint": group["fingerprint"], "sent": False,
                                "error": "feedback send failed; local entries remain pending"})
                continue
            results.append({**plan, "fingerprint": group["fingerprint"], "sent": sent})
            if sent:
                for entry in group["entries"]:
                    entry["sent"] = True
                _save(notes)
    return results


def auto_send(*, requested: bool = False, yes: bool = False, quiet: bool = False) -> list[dict]:
    """Best effort sharing never discards the local entry or leaks transport errors."""
    try:
        result = send(yes=yes) if requested or mode() == "always" else []
    except Exception:
        result = [{"sent": False, "error": "feedback send failed; local entries remain pending"}]
    if not quiet and any("error" in item for item in result):
        print("warning: feedback send failed; local entries remain pending", file=sys.stderr)
    return result


def automatic(command: str, error: YapError) -> None:
    """Capture the command, code and source location, preserving failing-command output."""
    try:
        record(f"{command} failed with {error.code}", kind="bug", step=command,
               code=error.code, origin=error.origin, checks=getattr(error, "checks", ()))
        # Automatic capture must leave stdout and stderr exactly as they were.
        auto_send(quiet=True)
    except Exception:
        pass


def command(args: argparse.Namespace) -> int:
    """Use the CLI's structured error and exit conventions for product feedback."""
    try:
        action = args.feedback_command
        if action == "setting":
            setting(args.mode)
            result = {"mode": mode()}
        elif action == "list":
            result = entries(pending=args.pending)
        elif action == "send":
            result = send(dry_run=args.dry_run, yes=args.yes)
        elif action == "remove":
            with _lock():
                notes = entries()
                kept = [entry for entry in notes
                        if entry["sent"] or entry["fingerprint"] != args.fingerprint]
                removed = len(notes) - len(kept)
                if not removed:
                    raise ValidationError("fingerprint", "no pending note has this fingerprint; "
                                          "see yap-studio feedback list --pending")
                _save(kept)
            result = {"removed": removed, "fingerprint": args.fingerprint}
        else:
            text = sys.stdin.read() if args.stdin else args.text
            if not text or not text.strip():
                raise ValidationError("feedback", "pass TEXT or --stdin with a nonempty note")
            entry = record(text, kind=args.kind, step=args.step, code=args.code)
            upstream = auto_send(requested=args.send, yes=args.yes)
            entry["sent"] = any(item["sent"] and item["fingerprint"] == entry["fingerprint"]
                                for item in upstream)
            result = {"entry": entry, "send": upstream}
            if not args.json:
                print("feedback recorded locally" + ("; sent" if entry["sent"] else "; pending"))
                for item in upstream:
                    if item.get("transport") == "link":
                        print(item["url"])
                return 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if action == "send" and any("error" in item for item in result):
            print("warning: feedback send failed; local entries remain pending", file=sys.stderr)
            return 2
        return 0
    except (OSError, ValueError, YapError, subprocess.SubprocessError) as exc:
        error = exc if isinstance(exc, ValidationError) else YapError(
            "Could not complete feedback; local entries remain pending.", code="feedback_failed")
        print(json.dumps({"error": error.to_dict()}) if args.json else f"error: {error.message}",
              file=sys.stderr)
        return 2


def _note_text(text: str) -> str:
    """Refuse a lone unknown subcommand before recording feedback."""
    word = text.strip()
    if word and not any(char.isspace() for char in word):
        raise argparse.ArgumentTypeError(
            f"unknown feedback subcommand '{word}'. To record a note, quote a sentence, "
            'as in yap-studio feedback "The render needed a retry", or use --stdin.'
        )
    return text


def record_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yap-studio feedback", description=
                                     "Record product feedback locally before optional sharing.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("text", nargs="?", metavar="TEXT", type=_note_text)
    source.add_argument("--stdin", action="store_true")
    parser.add_argument("--step")
    parser.add_argument("--code")
    parser.add_argument("--kind", choices=KINDS, default="confusing")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.set_defaults(func=command, command="feedback", feedback_command="record")
    return parser


def register(commands: argparse._SubParsersAction) -> None:
    for name in ("list", "send", "setting", "remove"):
        parser = commands.add_parser(name, help=f"{name} product feedback")
        parser.set_defaults(func=command)
        parser.add_argument("--json", action="store_true")
        if name == "list":
            parser.add_argument("--pending", action="store_true")
        elif name == "send":
            parser.add_argument("--dry-run", action="store_true")
            parser.add_argument("--yes", action="store_true")
        elif name == "remove":
            parser.add_argument("fingerprint")
        else:
            parser.add_argument("mode", choices=MODES)
