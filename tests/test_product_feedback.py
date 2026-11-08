"""Product feedback uses private stores, a loopback endpoint and a fake gh only."""

from __future__ import annotations

import json
import os
import re
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
    ("open /" + "Users/Somebody/Private Folder/take.mov", "Somebody", "[path]"),
    ("open /" + "Volumes/Secret Disk/Private Stuff/take.mov", "Secret", "[path]"),
    ("open /private/var/Personal Stuff/take", "Personal", "[path]"),
    ("open ~/Personal/take.mov", "Personal", "[path]"),
    ('open "/' + 'Users/Other/Secret Folder/take.mov" then retry', "Secret", "[path]"),
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


@pytest.mark.parametrize("value,expected", [
    ("e.g. lower band", "e.g. lower band"),
    ("i.e. the title", "i.e. the title"),
    ("fillers needs edit.json", "fillers needs edit.json"),
    ("new_project.py --name", "new_project.py --name"),
    ("plans/text-hook.json", "plans/text-hook.json"),
    ("cut/cut.mp4", "cut/cut.mp4"),
    ("see style.json top_px", "see style.json top_px"),
    ("project.json removals.json cut-words.json", "project.json removals.json cut-words.json"),
    ("final.mp4 cards.json faces.json layout.json", "final.mp4 cards.json faces.json layout.json"),
    ("progress.json answers.json setup.py", "progress.json answers.json setup.py"),
    ("words.json transcript.json phrases.json", "words.json transcript.json phrases.json"),
    ("spans.json receipt.json report.json verify.json",
     "spans.json receipt.json report.json verify.json"),
    ("E.G. lower band; I.E. the title", "E.G. lower band; I.E. the title"),
    ("EDIT.JSON PLANS/TEXT-HOOK.JSON", "EDIT.JSON PLANS/TEXT-HOOK.JSON"),
    ("my-holiday-clip.mov", "[file]"),
    ("Smith-contract.pdf", "[file]"),
    ("my-edit.json edit.json.backup edit-private.json", "[file] [file] [file]"),
    ("private.e.g private.i.e", "[file] [file]"),
    ("music.level captions.placement music.duck_threshold",
     "music.level captions.placement music.duck_threshold"),
    ("from.offset_s verify.status join.words", "from.offset_s verify.status join.words"),
    ("song.mp3 Music.Level.mov client-music.level", "[file] [file] [file]"),
])
def test_scrubber_keeps_product_file_names_and_abbreviations(value, expected):
    assert feedback.scrub(value) == expected
    assert feedback.scrub(expected) == expected


def test_home_scrub_limits_and_safe_domains(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "unique-home"))
    assert "unique-home" not in feedback.scrub(f"{Path.home()}/Secret Folder/file")
    assert feedback.scrub("go to https://github.com/private now") == "go to github.com now"
    assert feedback.scrub("2026-10-23T12:15:36.123456+00:00") == "2026-10-23T12:15:36.123456+00:00"
    assert len(feedback.scrub("note " * 500)) == 500
    assert "secret" not in feedback.scrub("x " * 240 + "/" + "Users/Someone/secret.mov")
    assert feedback.scrub("retry\x1b[31m\nnow").isascii()


def test_entry_shape_permissions_and_stdin(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_ID", "test-agent")
    before = datetime.now(UTC)
    assert cli.main(["feedback", "The check needed a retry", "--step", "verify",
                     "--code", "invalid_input", "--kind", "bug", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    entry = result["entry"]
    assert set(entry) == {"time", "version", "macos", "arch", "step", "code", "kind",
                          "text", "agent", "origin", "fingerprint", "sent"}
    assert entry["origin"] is None
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
    assert feedback.fingerprint("bug", None, None, "/" + "Users/A/take.mov") == (
        feedback.fingerprint("bug", None, None, "/" + "Users/B/other.mov"))


def test_unknown_subcommand_does_not_record_a_note(capsys, monkeypatch):
    from io import StringIO

    assert cli.main(["feedback", "schema"]) == 2
    output = capsys.readouterr()
    assert "usage:" in output.err
    assert "unknown feedback subcommand 'schema'" in output.err
    assert "--stdin" in output.err
    assert (
        "unknown feedback subcommand 'schema'. To record a note, quote a sentence, "
        'as in yap-studio feedback "The render needed a retry", or use --stdin.'
    ) in output.err
    assert "\\" not in output.err
    assert feedback.entries() == []
    assert cli.main(["feedback", "A quoted sentence"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(sys, "stdin", StringIO("schema"))
    assert cli.main(["feedback", "--stdin"]) == 0
    assert [entry["text"] for entry in feedback.entries()] == ["A quoted sentence", "schema"]


def test_remove_only_matching_pending_notes(capsys):
    sent = feedback.record("Repeated note")
    sent["sent"] = True
    feedback._save([sent])
    first = feedback.record("Repeated note")
    feedback.record("Repeated note")
    other = feedback.record("Another note")
    assert cli.main(["feedback", "remove", first["fingerprint"], "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "removed": 2, "fingerprint": first["fingerprint"],
    }
    assert feedback.entries() == [sent, other]
    for fingerprint in (first["fingerprint"], "unknown"):
        assert cli.main(["feedback", "remove", fingerprint]) == 2
        assert "no pending note has this fingerprint" in capsys.readouterr().err


def test_automatic_assembly_failure_names_checks(monkeypatch, capsys):
    from yap_studio.stages import assemble as assembly
    from yap_studio.stages.assemble import AssemblyError

    def fail(*_args, **_kwargs):
        raise AssemblyError("x", failed_checks=True, checks=("verify.status", "join.words"))

    monkeypatch.setattr(assembly, "assemble", fail)
    assert cli.main(["assemble", "synthetic", "--json"]) == 1
    capsys.readouterr()
    entry, = feedback.entries()
    assert entry["text"] == "assemble failed with assembly_checks_failed: verify.status, join.words"


@pytest.mark.parametrize("cause", ["", " FileNotFoundError"])
def test_fingerprint_groups_origins_without_line_numbers(cause):
    first = feedback.record("Failure", kind="bug", code="invalid_input",
                            origin=f"project/status:read:78{cause}")
    repeat = feedback.record("Failure", kind="bug", code="invalid_input",
                             origin=f"project/status:read:109{cause}")
    assert first["fingerprint"] == repeat["fingerprint"]
    for origin in ("project/status:json:78" + cause, "project/model:read:78" + cause,
                   "project/status:read:78 RuntimeError", None):
        other = feedback.record("Failure", kind="bug", code="invalid_input", origin=origin)
        assert first["fingerprint"] != other["fingerprint"]


@pytest.mark.parametrize("origin", [
    "project/status:read:78 FileNotFoundError",
    "stages/assemble_picture:prepare_background_video:372",
    "product_feedback:<lambda>:123",
])
def test_origin_is_kept_without_scrubbing(origin):
    entry = feedback.record("Failure", origin=origin)
    assert entry["origin"] == origin
    assert feedback.entries()[0]["origin"] == origin


@pytest.mark.parametrize("origin", [
    "/" + "Users" + "/Private/project/status:read:78",
    "project/status:read:78 Private Name",
    "project/status.py:read:78",
    "project/../status:read:78",
    "project/status:read:78\n",
    "project/status:read:78 FileNotFoundError\n",
    123, {"private": "value"},
])
def test_safe_drops_tampered_origins(origin):
    entry = feedback.record("Failure")
    ledger = feedback.data_dir() / "feedback.jsonl"
    entry["origin"] = origin
    ledger.write_text(json.dumps(entry) + "\n")
    assert feedback.entries()[0]["origin"] is None


@pytest.mark.parametrize("raised", [False, True])
def test_cli_failure_records_origin_in_ledger_and_reports(monkeypatch, capsys, raised):
    monkeypatch.setattr(feedback, "_config", lambda: {"feedback_mode": "invalid"})

    def fail(_arguments):
        if raised:
            feedback.mode()
        try:
            feedback.mode()
        except YapError as exc:
            print(exc, file=sys.stderr)
        return 2

    monkeypatch.setattr(cli, "_main", fail)
    if raised:
        with pytest.raises(YapError):
            cli.main(["verify", "Private input.mov"])
    else:
        assert cli.main(["verify", "Private input.mov"]) == 2
    capsys.readouterr()
    entry, = feedback.entries()
    assert re.fullmatch(r"product_feedback:mode:[0-9]+", entry["origin"])
    ledger = feedback.data_dir() / "feedback.jsonl"
    assert json.loads(ledger.read_text())["origin"] == entry["origin"]
    assert "Private" not in json.dumps(entry)
    monkeypatch.setattr(feedback, "_config", lambda: {})
    monkeypatch.setattr(feedback, "availability", lambda: {
        "endpoint": False, "gh_logged_in": False})
    plan, = feedback.send(dry_run=True)
    assert f"Code: invalid_parameter\nWhere: {entry['origin']}\n" in plan["body"]
    monkeypatch.setattr(feedback, "availability", lambda: {"endpoint": True})
    plan, = feedback.send(dry_run=True)
    assert plan["json"]["entries"][0]["origin"] == entry["origin"]


def test_cli_wrapped_failure_records_original_exception_site(tmp_path, capsys):
    # edit.json exists, so fillers reaches the project reads and wraps their OSError.
    (tmp_path / "edit.json").write_text("{}")
    assert cli.main(["fillers", str(tmp_path), "--json"]) == 2
    capsys.readouterr()
    entry, = feedback.entries()
    assert re.fullmatch(r"project/status:read:[0-9]+ FileNotFoundError", entry["origin"])
    assert str(tmp_path) not in json.dumps(entry)


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


@pytest.mark.parametrize("arguments,step", [
    (["versions", "save", "Private Project"], "versions save"),
    (["feedback", "reply", "Private Project", "--batch", "synthetic", "--comment", "v1:c1",
      "--status", "changed", "--text", "Private reply"], "feedback reply"),
])
def test_automatic_grouped_failure_records_subcommand(monkeypatch, capsys, arguments, step):
    from yap_studio.feedback import cli as feedback_cli

    def fail(_args):
        print(str(InputError("Private failure")), file=sys.stderr)
        return 2

    monkeypatch.setattr(feedback_cli, "_versions", fail)
    monkeypatch.setattr(feedback_cli, "_feedback", fail)
    assert cli.main(arguments) == 2
    capsys.readouterr()
    entry, = feedback.entries()
    assert entry["step"] == step
    assert entry["text"] == f"{step} failed with invalid_input"
    assert "Private" not in json.dumps(entry)
    assert feedback.scrub(step) == step
    assert entry["fingerprint"] == feedback.fingerprint(
        "bug", step, "invalid_input", entry["text"], entry["origin"])
    assert entry["fingerprint"] != feedback.fingerprint(
        "bug", arguments[0], "invalid_input", entry["text"], entry["origin"])
    preview, = feedback.send(dry_run=True)
    assert preview["title"] == f"bug: {step} invalid_input"
    assert f"Step: {step}\n" in preview["body"]


@pytest.mark.parametrize("arguments", [
    ["verify", "/synthetic/Private Project/take.mov"],
    ["verify", "Private text"],
    ["verify", "save"],
    ["versions", "/synthetic/Private Project"],
    ["versions", "sav"],
    ["feedback", "Private text"],
    [],
])
def test_automatic_failure_excludes_unregistered_arguments(monkeypatch, arguments):
    def fail(_arguments):
        str(InputError("Private failure"))
        return 2

    monkeypatch.setattr(cli, "_main", fail)
    assert cli.main(arguments) == 2
    entry, = feedback.entries()
    step = arguments[0] if arguments else "unknown"
    assert entry["step"] == step
    assert entry["text"] == f"{step} failed with invalid_input"
    assert "Private" not in json.dumps(entry)
    if len(arguments) > 1:
        assert arguments[1] not in json.dumps(entry)


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
        f"pages = {pages or [[]]!r}\n"
        "if 'state=open' in args[-1]: pages = [[item for item in page "
        "if item.get('state') != 'closed'] for page in pages]\n"
        "if args[0] == 'api': print('[[]]' if 'labels=feedback' in args[-1] "
        "else json.dumps(pages))\n"
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


def test_gh_unlabelled_fingerprint_comments_instead_of_creating(tmp_path, monkeypatch, capsys):
    entry = feedback.record("Check needed retry", kind="bug", step="verify")
    body = "Fingerprint: " + entry["fingerprint"]
    pages = [[{"number": 20, "body": body, "pull_request": {}, "labels": []},
              {"number": 21, "body": body, "labels": []}]]
    log = fake_gh(tmp_path, monkeypatch, pages=pages)
    assert cli.main(["feedback", "send", "--yes"]) == 0
    capsys.readouterr()
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    writes = [call for call in calls if call["args"][0] == "issue"]
    assert [call["args"][:3] for call in writes] == [["issue", "comment", "21"]]
    assert writes[0]["body"] == f"seen again in {__version__}, 1 times"
    assert next(call["args"] for call in calls if call["args"][0] == "api") == [
        "api", "--paginate", "--slurp", "-X", "GET",
        f"repos/{feedback.REPO}/issues?state=all&per_page=100",
    ]
    assert feedback.entries(pending=True) == []


def test_gh_closed_fingerprint_comments_instead_of_creating(tmp_path, monkeypatch):
    entry = feedback.record("Check needed retry", kind="bug", step="verify")
    feedback.record("check needed retry", kind="bug", step="verify")
    pages = [[{"number": 12, "body": "Fingerprint: " + entry["fingerprint"],
               "state": "closed"}]]
    fake_gh(tmp_path, monkeypatch, pages=pages)
    plan, = feedback.send(dry_run=True)
    assert plan["action"] == "comment"
    assert plan["issue"] == 12
    assert plan["body"] == f"seen again in {__version__}, 2 times"


@pytest.mark.parametrize("state", ["open", "closed"])
def test_gh_legacy_fingerprint_comments_with_origin(tmp_path, monkeypatch, state):
    entry = feedback.record("verify failed with invalid_input", kind="bug", step="verify",
                            code="invalid_input", origin="project/status:read:78")
    legacy = feedback.fingerprint(entry["kind"], entry["step"], entry["code"], entry["text"])
    assert legacy != entry["fingerprint"]
    pages = [[{"number": 12, "body": "Fingerprint: " + legacy, "state": state}]]
    log = fake_gh(tmp_path, monkeypatch, pages=pages)
    plan, = feedback.send(dry_run=True)
    assert plan["action"] == "comment"
    assert plan["issue"] == 12
    result, = feedback.send(yes=True)
    assert result["fingerprint"] == entry["fingerprint"]
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    writes = [call for call in calls if call["args"][0] == "issue"]
    assert [call["args"][:3] for call in writes] == [["issue", "comment", "12"]]
    assert writes[0]["body"] == (f"seen again in {__version__}, 1 times\n"
                                 "Where: project/status:read:78")
    assert feedback.entries(pending=True) == []


@pytest.mark.parametrize("locked", [True, False])
def test_gh_closed_fingerprint_creates_only_when_locked(tmp_path, monkeypatch, locked):
    entry = feedback.record("Check needed retry", kind="bug", step="verify")
    pages = [[{"number": 12, "body": "Fingerprint: " + entry["fingerprint"],
               "state": "closed", "locked": locked}]]
    fake_gh(tmp_path, monkeypatch, pages=pages)
    plan, = feedback.send(dry_run=True)
    assert plan["action"] == ("create" if locked else "comment")
    if locked:
        assert "Fingerprint: " + entry["fingerprint"] in plan["body"]
    else:
        assert plan["issue"] == 12


def test_gh_fingerprint_skips_locked_open_issue_for_unlocked_closed(tmp_path, monkeypatch):
    entry = feedback.record("Check needed retry", kind="bug", step="verify")
    body = "Fingerprint: " + entry["fingerprint"]
    pages = [[{"number": 12, "body": body, "state": "open", "locked": True}],
             [{"number": 21, "body": body, "state": "closed", "locked": False}]]
    fake_gh(tmp_path, monkeypatch, pages=pages)
    plan, = feedback.send(dry_run=True)
    assert plan["action"] == "comment"
    assert plan["issue"] == 21


@pytest.mark.parametrize("closed_first", [True, False])
def test_gh_fingerprint_prefers_open_issue(tmp_path, monkeypatch, closed_first):
    entry = feedback.record("Check needed retry", kind="bug", step="verify")
    body = "Fingerprint: " + entry["fingerprint"]
    pages = [[{"number": 12, "body": body, "state": "closed"}],
             [{"number": 21, "body": body, "state": "open"}]]
    if not closed_first:
        pages.reverse()
    pages[0].insert(0, {"number": 20, "body": body, "state": "open", "pull_request": {}})
    fake_gh(tmp_path, monkeypatch, pages=pages)
    plan, = feedback.send(dry_run=True)
    assert plan["action"] == "comment"
    assert plan["issue"] == 21
    assert plan["body"] == f"seen again in {__version__}, 1 times"


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
    entry.update(text="/" + "Users/Private/secret.mov", agent="agent@example.invalid",
                 step="/" + "Volumes/Private/test", extra="unshared", fingerprint="fake")
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
