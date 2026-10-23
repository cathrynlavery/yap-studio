"""Product feedback uses private stores, a loopback endpoint and a fake gh only."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from yap_studio import __version__, cli
from yap_studio import product_feedback as feedback
from yap_studio.errors import InputError, YapError


@pytest.mark.parametrize("value,private,marker", [
    ("open /Users/Somebody/Private Folder/take.mov", "Somebody", "[path]"),
    ("open /Volumes/Secret Disk/Private Stuff/take.mov", "Secret", "[path]"),
    ("open /private/var/Personal Stuff/take", "Personal", "[path]"),
    ("open ~/Personal/take.mov", "Personal", "[path]"),
    ('open "/Users/Other/Secret Folder/take.mov" then retry', "Secret", "[path]"),
    ("portrait.MOV failed", "portrait", "[file]"),
    ("draft.final.v2.json failed", "draft", "[file]"),
    ("read .env now", ".env", "[file]"),
    ("meet person+private@example.invalid now", "person", "[email]"),
    ("visit https://private.example/video?token=hidden", "hidden", "[url]"),
    ("visit https://github.com/owner/private?secret=hidden", "hidden", "github.com"),
    ("visit https://yapstudio.app/private#hidden", "hidden", "yapstudio.app"),
    ("visit https://github.com.evil.invalid/private", "evil", "[url]"),
    ("visit https://github.com@evil.invalid/private", "evil", "[url]"),
    ("visit https://[invalid/private", "invalid", "[url]"),
    ("visit www.private.example/SecretName", "SecretName", "[url]"),
    ("visit github.com/PrivateOwner/Secret", "Secret", "github.com"),
    ("host 192.168.42.1:9090", "192", "[ip]"),
    ("host 2001:db8:1234::42", "1234", "[ip]"),
    ("host 2001:db8:1234::42.", "1234", "[ip]"),
    ("host ::1", "::1", "[ip]"),
    ("host fe80::1234%en0", "1234", "[ip]"),
    ("host ::ffff:192.168.42.1", "192", "[ip]"),
    ("host desk.secret.ts.net", "desk", "[host]"),
    ("host my-mac.local", "my-mac", "[host]"),
    ("Bearer " + "sensitive-value", "sensitive", "[secret]"),
    ("api_key=" + "sensitive-value", "sensitive", "[secret]"),
    ('"api_key": "' + 'sensitive-value"', "sensitive", "[secret]"),
    ("token: " + "sensitive-value", "sensitive", "[secret]"),
    ("sk-" + "example-sensitive-value", "sensitive", "[secret]"),
    ("ghp_" + "example-sensitive-value", "sensitive", "[secret]"),
    ("github_pat_" + "example-sensitive-value", "sensitive", "[secret]"),
    ("xoxb-" + "example-sensitive-value", "sensitive", "[secret]"),
    ("AKIA" + "EXAMPLESENSITIVEVALUE", "SENSITIVE", "[secret]"),
    ("ab12" * 16, "ab12", "[secret]"),
    ("YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXo=" * 2, "YWJ", "[secret]"),
    ("／Users／Someone／Private/take.mov", "Someone", "[path]"),
    ("/Us\u200bers/Someone/Private/take.mov", "Someone", "[path]"),
])
def test_scrubber_adversarial(value, private, marker):
    cleaned = feedback.scrub(value)
    assert private not in cleaned
    assert marker in cleaned
    assert feedback.scrub(cleaned) == cleaned


def test_home_scrub_limits_and_safe_domains(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "unique-home"))
    assert "unique-home" not in feedback.scrub(f"{Path.home()}/Secret Folder/file")
    assert feedback.scrub("go to https://github.com/private now") == "go to github.com now"
    assert feedback.scrub("2026-10-23T12:15:36.123456+00:00") == "2026-10-23T12:15:36.123456+00:00"
    assert len(feedback.scrub("note " * 500)) == 500
    assert "secret" not in feedback.scrub("x " * 240 + "/Users/Someone/secret.mov")
    assert feedback.scrub("retry\x1b[31m\nnow").isascii()


def test_entry_shape_permissions_and_stdin(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_ID", "test-agent")
    before = datetime.now(UTC)
    assert cli.main(["feedback", "The check needed a retry", "--step", "verify",
                     "--code", "invalid_input", "--kind", "bug", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    entry = result["entry"]
    assert set(entry) == {"time", "version", "macos", "arch", "step", "code", "kind",
                          "text", "agent", "fingerprint", "sent"}
    assert before <= datetime.fromisoformat(entry["time"]) <= datetime.now(UTC)
    assert entry["version"] == __version__
    assert entry["step"] == "verify" and entry["code"] == "invalid_input"
    assert entry["kind"] == "bug" and entry["agent"] == "test-agent"
    assert entry["sent"] is False and result["send"] == []
    ledger = feedback.data_dir() / "feedback.jsonl"
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600
    assert json.loads(ledger.read_text()) == entry
    from io import StringIO

    monkeypatch.setattr(sys, "stdin", StringIO("a second note"))
    ledger.chmod(0o644)
    assert cli.main(["feedback", "--stdin"]) == 0
    assert capsys.readouterr().out == "feedback recorded locally; pending\n"
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600
    assert len(feedback.entries()) == 2
    assert feedback.entries()[1]["agent"] == "test-agent"


def test_fingerprint_stability_and_distinctions():
    first = feedback.fingerprint("bug", "run", "failed", "  TRY   Again ")
    assert first == feedback.fingerprint("bug", "run", "failed", "try again")
    assert first != feedback.fingerprint("slow", "run", "failed", "try again")
    assert first != feedback.fingerprint("bug", "cut", "failed", "try again")
    assert first != feedback.fingerprint("bug", "run", "other", "try again")
    assert feedback.fingerprint("bug", None, None, "/Users/A/take.mov") == (
        feedback.fingerprint("bug", None, None, "/Users/B/other.mov"))


@pytest.mark.parametrize("arguments", [[], [""], ["--kind", "invalid", "note"],
                                       ["note", "--stdin"]])
def test_invalid_note_records_only_automatic_yap_errors(arguments, capsys):
    assert cli.main(["feedback", *arguments]) == 2
    capsys.readouterr()
    notes = feedback.entries()
    if arguments == [""]:
        assert len(notes) == 1 and notes[0]["kind"] == "bug"
        assert notes[0]["code"] == "invalid_parameter"
    else:
        assert notes == []


@contextmanager
def endpoint(monkeypatch, *, status=204):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(status)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("YAP_STUDIO_FEEDBACK_ENDPOINT", f"http://127.0.0.1:{server.server_port}")
    try:
        yield received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_ask_dry_run_endpoint_grouping_list_and_no_resend(monkeypatch, capsys):
    feedback.record("Retry the check", step="verify")
    feedback.record("retry the check", step="verify")
    with endpoint(monkeypatch) as received:
        assert cli.main(["feedback", "send"]) == 2
        assert "--dry-run" in capsys.readouterr().err
        # The refusal itself has an automatic bug note, which is also local.
        pending = feedback.entries(pending=True)
        ledger = feedback.data_dir() / "feedback.jsonl"
        before = ledger.read_bytes()
        assert cli.main(["feedback", "send", "--dry-run"]) == 0
        preview = json.loads(capsys.readouterr().out)
        assert len(preview) == 2
        assert preview[0]["json"]["count"] == 2
        assert preview[0]["json"]["entries"] == pending[:2]
        assert ledger.read_bytes() == before and received == []
        assert cli.main(["feedback", "list", "--pending", "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == pending
        assert cli.main(["feedback", "send", "--yes", "--json"]) == 0
        capsys.readouterr()
        assert received == [item["json"] for item in preview]
        assert feedback.entries(pending=True) == []
        assert all(entry["sent"] for entry in feedback.entries())
        assert feedback.send(yes=True) == []
        assert len(received) == 2


@pytest.mark.parametrize("consent", ["ask", "never"])
def test_local_modes_refuse_without_consent(monkeypatch, capsys, consent):
    assert cli.main(["feedback", "setting", consent]) == 0
    capsys.readouterr()
    with endpoint(monkeypatch) as received:
        assert cli.main(["feedback", "A tool problem"]) == 0
        assert "pending" in capsys.readouterr().out
        assert cli.main(["feedback", "send", *(["--yes"] if consent == "never" else [])]) == 2
        assert ("never" if consent == "never" else "--dry-run") in capsys.readouterr().err
        assert received == []


def test_always_sends_after_local_write(monkeypatch, capsys):
    feedback.setting("always")
    original = feedback._post

    def check_local(plan):
        assert len(feedback.entries(pending=True)) == 1
        return original(plan)

    monkeypatch.setattr(feedback, "_post", check_local)
    with endpoint(monkeypatch) as received:
        assert cli.main(["feedback", "The docs needed a retry", "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["entry"]["sent"] is True
        assert len(received) == 1
        assert feedback.entries()[0]["sent"] is True


@pytest.mark.parametrize("override", ["1", "true", "yes", " TRUE "])
def test_env_forces_always_over_never(monkeypatch, override):
    feedback.setting("never")
    monkeypatch.setenv("YAP_STUDIO_FEEDBACK_AUTO_SEND", override)
    assert feedback.mode() == "always"


@pytest.mark.parametrize("override", ["0", "false", "no", ""])
def test_other_env_values_keep_saved_mode(monkeypatch, override):
    feedback.setting("never")
    monkeypatch.setenv("YAP_STUDIO_FEEDBACK_AUTO_SEND", override)
    assert feedback.mode() == "never"


def test_send_on_record_uses_consent(monkeypatch, capsys):
    with endpoint(monkeypatch) as received:
        assert cli.main(["feedback", "Tool note", "--send", "--yes"]) == 0
        assert "sent" in capsys.readouterr().out
        assert len(received) == 1


def test_send_failure_warns_and_leaves_pending(monkeypatch, capsys):
    feedback.setting("always")
    with endpoint(monkeypatch, status=503) as received:
        assert cli.main(["feedback", "Tool note"]) == 0
        output = capsys.readouterr()
        assert "recorded locally" in output.out and "warning:" in output.err
        assert len(received) == 1 and len(feedback.entries(pending=True)) == 1


def test_send_failure_is_json_and_later_groups_still_send(monkeypatch, capsys):
    failed = feedback.record("First note")
    feedback.record("Second note")
    with endpoint(monkeypatch) as received:
        post = feedback._post

        def fail_first(plan):
            if plan["json"]["fingerprint"] == failed["fingerprint"]:
                raise OSError("synthetic private transport details")
            return post(plan)

        monkeypatch.setattr(feedback, "_post", fail_first)
        assert cli.main(["feedback", "send", "--yes", "--json"]) == 2
        output = capsys.readouterr()
        result = json.loads(output.out)
        assert result[0]["sent"] is False and "error" in result[0]
        assert result[1]["sent"] is True and len(received) == 1
        assert "private transport" not in output.err + output.out
        assert feedback.entries()[0]["sent"] is False
        assert feedback.entries()[1]["sent"] is True


@pytest.mark.parametrize("consent,status", [("ask", 204), ("always", 204), ("always", 503)])
def test_automatic_failure_preserves_output_and_exit(
    monkeypatch, tmp_path, capsys, consent, status,
):
    def fail(**_kwargs):
        raise InputError(str(tmp_path / "Private Name.mov"))

    monkeypatch.setattr(cli, "verify", fail)
    argv = ["verify", "anything.mov", "--json"]
    with monkeypatch.context() as baseline:
        baseline.setattr(feedback, "automatic", lambda *_args: None)
        before_exit = cli.main(argv)
        before_output = capsys.readouterr()
    feedback.setting(consent)
    with endpoint(monkeypatch, status=status) as received:
        assert cli.main(argv) == before_exit == 2
        assert capsys.readouterr() == before_output
        entry = feedback.entries()[0]
        assert entry["text"] == "verify failed with invalid_input"
        assert entry["step"] == "verify" and entry["kind"] == "bug"
        assert entry["code"] == "invalid_input"
        assert "Private" not in json.dumps(entry)
        assert len(received) == (consent == "always")
        assert entry["sent"] == (consent == "always" and status == 204)


def test_feedback_write_failure_never_changes_command(monkeypatch, capsys):
    def fail(**_kwargs):
        raise YapError("original error", code="original")

    monkeypatch.setattr(cli, "verify", fail)
    monkeypatch.setattr(feedback, "record", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        OSError("disk full")))
    assert cli.main(["verify", "test.mov"]) == 2
    assert capsys.readouterr().err == "error: original error\n"


def test_success_with_handled_yap_error_does_not_record(monkeypatch, capsys):
    def succeed(**_kwargs):
        str(YapError("handled warning"))
        return {"status": "pass", "checks": []}

    monkeypatch.setattr(cli, "verify", succeed)
    monkeypatch.setattr(cli, "exit_code", lambda *_args: 0)
    assert cli.main(["verify", "anything.mov", "--json"]) == 0
    capsys.readouterr()
    assert feedback.entries() == []


def fake_gh(tmp_path, monkeypatch, *, logged_in=True, pages=None):
    binary = tmp_path / "gh"
    log = tmp_path / "gh-calls.jsonl"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        f"log = Path({str(log)!r})\n"
        "args = sys.argv[1:]\n"
        "body = Path(args[args.index('--body-file')+1]).read_text() "
        "if '--body-file' in args else None\n"
        "with log.open('a') as out: out.write(json.dumps({'args': args, 'body': body})+'\\n')\n"
        f"if args[0] == 'auth': sys.exit({0 if logged_in else 1})\n"
        f"if args[0] == 'api': print({json.dumps(pages or [[]])!r})\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    return log


def test_gh_existing_fingerprint_comments_and_new_creates(tmp_path, monkeypatch, capsys):
    first = feedback.record("Check needed retry", kind="bug", step="verify")
    feedback.record("check needed retry", kind="bug", step="verify")
    feedback.record("Missing docs", kind="missing")
    # The match is on the second page, after a full first page.
    pages = [[{"number": n, "body": "unrelated"} for n in range(100)],
             [{"number": 101, "body": "**Fingerprint:** " + first["fingerprint"]}]]
    log = fake_gh(tmp_path, monkeypatch, pages=pages)
    preview = feedback.send(dry_run=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert all(call["args"][0] in {"auth", "api"} for call in calls)
    assert preview[0]["body"] == f"seen again in {__version__}, 2 times"
    assert preview[1]["action"] == "create"
    assert cli.main(["feedback", "send", "--yes"]) == 0
    capsys.readouterr()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    writes = [call for call in calls if call["args"][0] == "issue"]
    assert [call["args"][1] for call in writes] == ["comment", "create"]
    assert writes[0]["args"][2] == "101"
    assert writes[0]["body"] == preview[0]["body"]
    assert writes[1]["body"] == preview[1]["body"]
    assert "Fingerprint: " in writes[1]["body"]
    assert all("--repo" in call["args"] and feedback.REPO in call["args"] for call in writes)
    assert feedback.entries(pending=True) == []
    assert feedback.send(yes=True) == []
    assert len([line for line in log.read_text().splitlines() if '"issue"' in line]) == 2


@pytest.mark.parametrize("installed", [False, True])
def test_link_fallback_missing_or_logged_out(tmp_path, monkeypatch, installed):
    if installed:
        fake_gh(tmp_path, monkeypatch, logged_in=False)
    else:
        monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
    feedback.record("界" * 500, step="run")
    plans = feedback.send(yes=True)
    assert plans[0]["transport"] == "link" and plans[0]["sent"] is False
    link = plans[0]["url"]
    assert len(link) < 8000
    query = parse_qs(urlsplit(link).query)
    assert query["template"] == ["feedback.md"]
    assert "Fingerprint: " in query["body"][0]
    assert len(feedback.entries(pending=True)) == 1


def test_rescrub_edited_ledger_and_drop_unknown_fields(monkeypatch):
    feedback.record("A note")
    ledger = feedback.data_dir() / "feedback.jsonl"
    entry = json.loads(ledger.read_text())
    entry.update(text="/Users/Private/secret.mov", agent="agent@example.invalid",
                 step="/Volumes/Private/test", extra="unshared", fingerprint="fake")
    ledger.write_text(json.dumps(entry) + "\n")
    with endpoint(monkeypatch) as received:
        preview = feedback.send(dry_run=True)[0]["json"]
        assert "Private" not in json.dumps(preview)
        assert "unshared" not in json.dumps(preview)
        assert feedback.entries() == preview["entries"]
        feedback.send(yes=True)
        assert received == [preview]


def test_setting_preserves_other_config_and_env_is_effective(monkeypatch):
    config = feedback.config_root() / "config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"schema": "yap-studio/config@1",
                                  "default_style": "preset:clean"}))
    feedback.setting("never")
    assert json.loads(config.read_text())["default_style"] == "preset:clean"
    monkeypatch.setenv("YAP_STUDIO_FEEDBACK_AUTO_SEND", "yes")
    assert feedback.availability()["mode"] == "always"
