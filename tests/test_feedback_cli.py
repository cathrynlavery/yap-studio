"""Version commands and launching immutable project snapshots for review."""

from __future__ import annotations

import http.client
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from test_feedback_apply import _document, _tree, _trim
from test_feedback_apply import apply_project as apply_project
from test_feedback_batch import _cut, _export, _json
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_locate import WORDS, _comment
from test_feedback_outline import outline_project as outline_project
from test_feedback_versions import _update_cut
from test_feedback_versions import project as project

from yap_studio.cli import _review_media_paths, main
from yap_studio.errors import InputError, ValidationError
from yap_studio.feedback.versions import save_version
from yap_studio.project.model import CutWords
from yap_studio.review.launch import version_video
from yap_studio.review.store import ReviewStore


def test_parser_does_not_import_feedback_implementation() -> None:
    result = subprocess.run([
        sys.executable, "-c",
        "import sys\n"
        "from yap_studio.cli import _parser\n"
        "parser = _parser()\n"
        "assert parser.parse_args(['feedback', 'check', 'project']).feedback_command == 'check'\n"
        "assert parser.parse_args(['outline', 'build', 'project']).outline_command == 'build'\n"
        "assert parser.parse_args(['outline', 'review', 'project']).outline_command == 'review'\n"
        "args = parser.parse_args(['feedback', 'apply', 'project', 'ops.json', '--dry-run'])\n"
        "assert args.feedback_command == 'apply' and args.dry_run\n"
        "assert 'yap_studio.review.outline_server' not in sys.modules\n"
        "assert 'yap_studio.review.server' not in sys.modules\n"
        "for name in ('apply', 'edits', 'batch', 'keeps', 'versions', 'removed', 'outline'):\n"
        "    assert 'yap_studio.feedback.' + name not in sys.modules, name\n",
    ], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_outline_review_missing_outline_exits_two(outline_project: Path, capsys) -> None:
    assert main(["outline", "review", str(outline_project), "--no-open"]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "error: No story outline exists." in captured.err
    assert "fix: Run yap-studio outline build PROJECT first." in captured.err
    assert not (outline_project / "outline").exists()


@pytest.mark.parametrize("as_json", [False, True])
def test_outline_review_launch_prints_url_and_closes(outline_project, monkeypatch, capsys, as_json):
    from yap_studio.feedback.outline import build_outline
    from yap_studio.review import outline_server

    build_outline(outline_project)
    seen = []

    class Server:
        project = "sample-story"
        url = "http://127.0.0.1:12345/?t=test-token"

        def __init__(self, root, *, port):
            assert root == outline_project and port == 23456

        def serve_forever(self):
            seen.append("served")
            raise KeyboardInterrupt

        def server_close(self):
            seen.append("closed")

    monkeypatch.setattr(outline_server, "OutlineServer", Server)
    args = ["outline", "review", str(outline_project), "--no-open", "--port", "23456"]
    assert main(args + (["--json"] if as_json else [])) == 0
    captured = capsys.readouterr()
    assert not captured.err and seen == ["served", "closed"]
    if as_json:
        assert json.loads(captured.out) == {"url": Server.url}
        assert len(captured.out.splitlines()) == 1
    else:
        assert captured.out == (
            f"Story outline for sample-story at {Server.url}\nPress Ctrl-C to stop.\n"
        )


@pytest.mark.e2e
@pytest.mark.parametrize("as_json", [False, True])
def test_outline_review_subprocess(outline_project: Path, as_json: bool) -> None:
    from yap_studio.feedback.outline import build_outline

    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("sandbox blocked loopback socket binding")
    build_outline(outline_project)
    root = Path(__file__).parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-u", "-m", "yap_studio", "outline", "review",
               str(outline_project), "--no-open", "--port", "0"]
    if as_json:
        command.append("--json")
    process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None and process.stderr is not None
    lines = queue.Queue()

    def read_lines():
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    reader = threading.Thread(target=read_lines, daemon=True)
    reader.start()
    try:
        line = lines.get(timeout=30)
        assert line is not None, process.stderr.read()
        if as_json:
            url = urlsplit(json.loads(line)["url"])
        else:
            assert line.startswith("Story outline for sample-story at http://")
            url = urlsplit(line.strip().split(" at ", 1)[1])
            assert lines.get(timeout=5) == "Press Ctrl-C to stop.\n"
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        try:
            connection.request("GET", "/api/outline?" + url.query)
            response = connection.getresponse()
            assert response.status == 200
            view = json.loads(response.read())
            assert view["project"] == "sample-story"
            assert view["outline"]["beats"][0]["role"] == "opening"
            assert view["status"]["state"] == "waiting"
            assert view["notes"] == []
        finally:
            connection.close()
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=10) == 0, process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=1)
        process.stdout.close()
        process.stderr.close()


def test_outline_cli_build_text_and_json(outline_project: Path, capsys) -> None:
    root = outline_project
    assert main(["outline", "build", str(root)]) == 0
    assert capsys.readouterr().out == (
        "opening s1: word2 word3 word4\nbeat s2: word8 word9 word10\n"
        "ending s3: word14 word15 word16\nLeft out: 4\n"
    )
    assert main(["outline", "build", str(root), "--summary", "A plain story", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "yap-studio/story-outline@1"
    assert result["summary"] == "A plain story"


def test_outline_cli_status_gate_exit_codes(outline_project: Path, capsys, monkeypatch) -> None:
    from datetime import timedelta

    from test_feedback_outline import NOW

    from yap_studio.feedback import outline as outline_module
    from yap_studio.feedback.outline import approve_outline, build_outline, set_wait

    root = outline_project
    assert main(["outline", "status", str(root)]) == 3
    assert capsys.readouterr().out == "missing\n"
    build_outline(root)
    assert main(["outline", "status", str(root), "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["state"] == "waiting"
    set_wait(root, 1)
    monkeypatch.setattr(outline_module, "_now", lambda: NOW + timedelta(minutes=1))
    assert main(["outline", "status", str(root)]) == 0
    assert capsys.readouterr().out == "timed_out\n"
    approve_outline(root)
    assert main(["outline", "status", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "approved"
    (root / "edit.json").write_bytes((root / "edit.json").read_bytes() + b"\n")
    assert main(["outline", "status", str(root)]) == 3
    assert capsys.readouterr().out == "stale\n"


def test_outline_cli_send_text_and_json(outline_project: Path, capsys) -> None:
    from yap_studio.feedback.outline import add_note, build_outline

    root = outline_project
    build_outline(root)
    add_note(root, beat="s1", kind="keep", text="Keep it.")
    assert main(["outline", "send", str(root)]) == 0
    assert capsys.readouterr().out == "Wrote feedback/b1.json: 1 outline notes\n"
    add_note(root, beat=None, kind="fix", text="Change it.")
    assert main(["outline", "send", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["id"] == "b2" and result["base"] == "outline"
    assert main(["outline", "status", str(root)]) == 3
    assert capsys.readouterr().out == "notes_sent\n"


@pytest.mark.parametrize("command", ["build", "status", "send"])
def test_outline_cli_errors_exit_two(outline_project: Path, capsys, command: str) -> None:
    from yap_studio.feedback.outline import build_outline

    root = outline_project
    build_outline(root)
    if command == "build":
        word_path = root / "ingest/t1/words.json"
        word_path.write_bytes(word_path.read_bytes() + b"\n")
    elif command == "status":
        (root / "outline/state.json").write_text("{invalid")
    assert main(["outline", command, str(root), "--json"]) == 2
    result = capsys.readouterr()
    assert result.out == "" and "error:" in result.err and "fix:" in result.err
@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_feedback_apply_text_json_and_dry_run(
    apply_project: Path, capsys, dry_run: bool, json_output: bool,
) -> None:
    operations_path = apply_project / "operations.json"
    _json(operations_path, _document(_trim()))
    before = _tree(apply_project)
    arguments = ["feedback", "apply", str(apply_project), str(operations_path)]
    if dry_run:
        arguments.append("--dry-run")
    if json_output:
        arguments.append("--json")
    assert main(arguments) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        result = json.loads(captured.out)
        assert result["batch"] == "b1" and result["dry_run"] is dry_run
        assert result["status"] == ("planned" if dry_run else "applied")
        assert result["changes"] == [{
            "operation": "trim", "comment": "v1:c1", "kind": "trim",
            "summary": 'Trimmed s1 to end on "so".', "segments": ["s1"],
            "words": [{"id": "t1:3", "text": "about."}],
        }]
        assert result["edit"][0]["to"] == {"word": "t1:2"}
    else:
        verb = "Would apply" if dry_run else "Applied"
        next_step = "" if dry_run else f" Next: yap-studio cut {apply_project}"
        assert captured.out == ('Trimmed s1 to end on "so".\n'
                                f"{verb} 1 operations for 1 notes in b1.{next_step}\n")
    if dry_run:
        assert _tree(apply_project) == before


def test_feedback_apply_stale_base_exits_two_with_error_and_fix(
    apply_project: Path, capsys,
) -> None:
    operations_path = apply_project / "operations.json"
    _json(operations_path, _document(_trim()))
    _cut(apply_project, label="second version")
    save_version(apply_project)
    assert main(["feedback", "apply", str(apply_project), str(operations_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "error: Batch base v1 is not the current version (v2).\n"
        "fix: Review the current saved version and recompute the operations.\n"
    )


def test_feedback_apply_invalid_json_exits_two(apply_project: Path, capsys) -> None:
    operations_path = apply_project / "operations.json"
    operations_path.write_text("{broken", encoding="utf-8")
    assert main(["feedback", "apply", str(apply_project), str(operations_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err and "cannot read valid JSON" in captured.err


def test_feedback_apply_writer_error_exits_two_and_preserves_pending(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    from test_feedback_apply import FakeRemovalWriter, _project, _removal_document

    from yap_studio.feedback import apply as apply_module
    from yap_studio.feedback.batch import load_batch

    root = _project(tmp_path / "project", filler=True)
    operations_path = root / "operations.json"
    _json(operations_path, _removal_document())
    original = apply_module.apply_operations
    writer = FakeRemovalWriter(crash=True)
    monkeypatch.setattr(apply_module, "apply_operations", lambda project, operations, **kwargs:
                        original(project, operations, removals=writer, **kwargs))
    assert main(["feedback", "apply", str(root), str(operations_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "error: Could not apply feedback operations: crash after first writer call\n"
        "fix: Fix the reported error and rerun the same operations file.\n"
    )
    assert load_batch(root, "b1")["application"]["status"] == "pending"


def test_feedback_ask_and_answer(feedback_project: Path, capsys) -> None:
    from yap_studio.feedback.batch import build_batch, load_batch

    root = feedback_project
    save_version(root)
    build_batch(root, _export(root, [_comment(), _comment("c2")]))
    command = ["feedback", "ask", str(root), "--batch", "b1", "--comment", "v1:c1",
               "--question", "Which caption?", "--option", "caption=The caption",
               "--option", "story=The words"]
    assert main(command) == 0
    assert capsys.readouterr().out == "Asked v1:c1 in b1\n"
    assert main(["feedback", "answer", str(root), "--batch", "b1", "--comment", "v1:c1",
                 "--option", "caption", "--json"]) == 0
    batch = json.loads(capsys.readouterr().out)
    assert batch["items"][0]["clarification"]["option"] == "caption"
    assert main(["feedback", "ask", str(root), "--batch", "b1", "--comment", "v1:c2",
                 "--question", "Which part?"]) == 0
    capsys.readouterr()
    assert main(["feedback", "answer", str(root), "--batch", "b1", "--comment", "v1:c2",
                 "--answer", "  The final line  "]) == 0
    assert capsys.readouterr().out == "Answered v1:c2 in b1\n"
    assert load_batch(root, "b1")["items"][1]["clarification"]["answer"] == "The final line"


@pytest.mark.parametrize("arguments", [
    ["ask", "--comment", "invalid", "--question", "Which part?"],
    ["ask", "--comment", "v1:c1", "--question", " "],
    ["ask", "--comment", "v1:c1", "--question", "Which part?", "--option", "invalid"],
    ["answer", "--comment", "v1:c1", "--answer", " "],
    ["answer", "--comment", "v1:c1", "--option", "bad"],
])
def test_feedback_clarification_errors_exit_two(feedback_project: Path, capsys, arguments) -> None:
    from yap_studio.feedback.batch import build_batch
    from yap_studio.feedback.clarify import ask_clarification

    root = feedback_project
    save_version(root)
    build_batch(root, _export(root, [_comment()]))
    ask_clarification(root, "b1", "v1", "c1", "Which part?")
    assert main(["feedback", arguments[0], str(root), "--batch", "b1", *arguments[1:]]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "error:" in captured.err


def test_versions_save_and_list_text_and_json(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["versions", "save", str(project), "--note", "First review"]) == 0
    assert capsys.readouterr().out == "Saved v1 from cut/cut.mp4 (60 frames) in versions/v1\n"
    _update_cut(project)
    assert main(["versions", "save", str(project), "--json"]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["id"] == "v2" and second["parent"] == "v1"
    assert second["media"]["source"] == "cut/cut.mp4"
    assert main(["versions", "list", str(project), "--json"]) == 0
    versions = json.loads(capsys.readouterr().out)
    assert [item["id"] for item in versions] == ["v1", "v2"]
    assert versions[0]["note"] == "First review"
    assert main(["versions", "list", str(project)]) == 0
    assert capsys.readouterr().out == "".join(
        f"{item['id']}  {item['created_at']}  {item['media']['sha256'][:12]}  "
        f"{item['note'] or ''}\n" for item in versions
    )


def test_versions_rollback_approve_and_current(project: Path, capsys) -> None:
    assert main(["versions", "current", str(project), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"current": None, "approved": None}
    _json(project / "cut" / "verify.json", {"status": "PASS"})
    save_version(project)
    _update_cut(project)
    save_version(project)
    assert main(["versions", "rollback", str(project), "v1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == "v1"
    assert main(["versions", "approve", str(project), "v1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["report_status"] == "pass"
    assert main(["versions", "current", str(project)]) == 0
    assert capsys.readouterr().out == "Current: v1\nApproved: v1\n"
    assert main(["versions", "approve", str(project), "v2"]) == 0
    assert capsys.readouterr().out == "Approved v2 (pass)\n"
    assert main(["versions", "current", str(project), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"current": "v1", "approved": "v2"}
    assert main(["versions", "rollback", str(project), "v2"]) == 0
    assert "Current version: v2" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["rollback", "approve"])
def test_versions_release_errors_exit_two(project: Path, capsys, command: str) -> None:
    assert main(["versions", command, str(project), "v0"]) == 2
    output = capsys.readouterr()
    assert output.out == "" and "error:" in output.err
    save_version(project)
    assert main(["versions", command, str(project), "v9"]) == 2
    assert "error:" in capsys.readouterr().err
    if command == "approve":
        assert main(["versions", command, str(project), "v1"]) == 2
        assert "This version has no check report" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="needs ffmpeg and ffprobe")
@pytest.mark.parametrize("inside", [True, False])
def test_versions_save_explicit_media_source_and_printed_path(
    project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], inside: bool,
) -> None:
    folder = project / "renders" if inside else tmp_path / "project-extra"
    folder.mkdir()
    media = folder / "out.mp4"
    _clip(media, "blue")
    source = "renders/out.mp4" if inside else "out.mp4"
    assert main(["versions", "save", str(project), "--media", str(media)]) == 0
    captured = capsys.readouterr()
    assert captured.out == f"Saved v1 from {source} (60 frames) in versions/v1\n"
    assert captured.err == ""
    assert main(["versions", "list", str(project), "--json"]) == 0
    manifests = json.loads(capsys.readouterr().out)
    assert manifests[0]["media"]["source"] == source


def test_versions_errors_have_fix_and_exit_two(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "cut" / "receipt.json").unlink()
    assert main(["versions", "save", str(project)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err and "fix: Run yap-studio cut first." in captured.err


def test_versions_list_empty_and_corrupt(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["versions", "list", str(project), "--json"]) == 0
    assert capsys.readouterr().out == "[]\n"
    manifest = save_version(project)
    (manifest.path / "manifest.json").write_text("{broken", encoding="utf-8")
    assert main(["versions", "list", str(project)]) == 2
    assert "error:" in capsys.readouterr().err


def test_project_launch_needs_saved_versions_and_refuses_mixing(
    project: Path, tmp_path: Path
) -> None:
    store = ReviewStore(tmp_path / "data")
    with pytest.raises(InputError) as error:
        version_video([str(project)], store, _review_media_paths)
    assert error.value.suggested_action == "Run yap-studio versions save first."
    with pytest.raises(ValidationError, match="cannot mix"):
        version_video([str(project), "other.mp4"], store, _review_media_paths)


def _clip(path: Path, color: str) -> None:
    subprocess.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
        f"color=c={color}:size=32x32:rate=30", "-frames:v", "60", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(path),
    ], stdin=subprocess.DEVNULL, capture_output=True, timeout=30, check=True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="needs ffmpeg and ffprobe")
@pytest.mark.parametrize("suffix", [".mp4", ".mov"])
def test_project_launch_saved_word_maps_and_changed_copy(
    project: Path, tmp_path: Path, suffix: str,
) -> None:
    cut = project / "cut" / "cut.mp4"
    _clip(cut, "red")
    _update_cut(project, cut.read_bytes())
    first = save_version(project)
    final = project / f"final{suffix}"
    _clip(final, "blue")
    _update_cut(project, final.read_bytes())
    second = save_version(project, media=final)
    store = ReviewStore(tmp_path / "data")
    video = version_video([str(project)], store, _review_media_paths)
    assert video.media.path == second.media_path and video.sha256 == second.media["sha256"]
    assert video.previous_sha is None
    assert video.series.id == first.media["sha256"]
    assert [item.version_id for item in video.series.versions] == ["v1", "v2"]
    names = ["sample-video-v1.mp4", f"sample-video-v2{suffix}"]
    assert [item.media.name for item in video.series.versions] == names
    assert video.media.name == names[-1]
    assert all(item.words.words[0].id == "t1:0" for item in video.series.versions)
    stored = store.series(video.series.id)
    assert stored.versions[0].project == "sample-video"
    assert [item.name for item in stored.versions] == names
    first.media_path.write_bytes(b"modified saved render")
    with pytest.raises(InputError, match="changed after it was saved") as error:
        version_video([str(project)], store, _review_media_paths)
    assert error.value.path == str(first.media_path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="needs ffmpeg and ffprobe")
def test_file_launch_one_duplicate_and_ordered_files(tmp_path: Path) -> None:
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    _clip(first, "red")
    _clip(second, "blue")
    store = ReviewStore(tmp_path / "data")
    with pytest.raises(ValidationError, match="--versions needs at least two files"):
        version_video([str(first)], store, _review_media_paths)
    identical = tmp_path / "identical.mp4"
    shutil.copyfile(first, identical)
    with pytest.raises(ValidationError, match="identical contents"):
        version_video([str(first), str(identical)], store, _review_media_paths)
    video = version_video([str(first), str(second)], store, _review_media_paths)
    assert [item.media.path for item in video.series.versions] == [first, second]
    assert all(item.words is None and item.version_id is None for item in video.series.versions)
    assert video.media.path == second
    with pytest.raises(ValidationError, match="expected order"):
        version_video([str(second), str(first)], store, _review_media_paths)


def test_file_launch_fills_saved_earlier_versions(tmp_path: Path) -> None:
    first, second, third = (tmp_path / name for name in ("v1.mp4", "v2.mp4", "v3.mp4"))
    for path, color in ((first, "red"), (second, "blue"), (third, "green")):
        _clip(path, color)
    store = ReviewStore(tmp_path / "data")
    version_video([str(first), str(second)], store, _review_media_paths)
    extended = version_video([str(second), str(third)], store, _review_media_paths)
    assert [(item.number, item.media.path) for item in extended.series.versions] == [
        (1, first), (2, second), (3, third),
    ]
    assert [item.number for item in version_video(
        [str(third)], store, _review_media_paths,
    ).series.versions] == [1, 2, 3]
    first.rename(tmp_path / "moved.mp4")
    with pytest.raises(ValidationError, match="missing version 1: v1.mp4"):
        version_video([str(third)], store, _review_media_paths)


def test_feedback_build_check_and_json_exit_codes(
    feedback_project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root = feedback_project
    save_version(root)
    export_path = root / "export.json"
    _json(export_path, _export(root, [
        _comment("c1", target="element"), _comment("c2", kind="keep", start_frame=66, end_frame=93),
    ]))
    assert main(["feedback", "build", str(root), str(export_path)]) == 0
    assert capsys.readouterr().out == (
        "Wrote feedback/b1.json: 2 comments (1 fix, 1 keep), 0 carried, 1 constraints\n"
    )
    assert main(["feedback", "check", str(root)]) == 0
    assert capsys.readouterr().out.startswith("kept v1 c2:")
    assert main(["feedback", "build", str(root), str(export_path), "--json"]) == 0
    batch = json.loads(capsys.readouterr().out)
    assert batch["id"] == "b2" and batch["base_version"] == "v1"
    assert main(["feedback", "check", str(root), "--batch", "b1", "--version", "v1",
                 "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["results"][0]["comment"] == {
        "version": "v1", "id": "c2", "revision": _comment().updated_at,
    }
    _cut(root, words=CutWords("output", {}, WORDS.words[:3]), label="dropped word")
    save_version(root)
    assert main(["feedback", "check", str(root)]) == 3
    captured = capsys.readouterr()
    assert captured.out.startswith("broken v1 c2:") and "t1:2" in captured.out
    assert captured.err == ""
    assert main(["feedback", "check", str(root), "--version", "v2", "--json"]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and report["results"][0]["status"] == "broken"


def test_feedback_cli_layers_unmeasured_and_errors(
    feedback_project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from test_feedback_batch import _layers

    root = feedback_project
    export_path = root / "export.json"
    _json(export_path, _export(root, [_comment("c1", kind="keep", target="caption")]))
    assert main(["feedback", "build", str(root), str(export_path), "--layers",
                 str(_layers(root))]) == 0
    capsys.readouterr()
    assert main(["feedback", "check", str(root)]) == 0
    assert capsys.readouterr().out == "unmeasured cut c1: no layer data for the new version\n"
    export_path.write_text("{invalid", encoding="utf-8")
    assert main(["feedback", "build", str(root), str(export_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "error:" in captured.err and "fix:" in captured.err
    _json(export_path, {"format": "invalid", "version": 2})
    assert main(["feedback", "build", str(root), str(export_path)]) == 2
    assert "error:" in capsys.readouterr().err
    assert main(["feedback", "check", str(root), "--batch", "b99"]) == 2
    assert "error:" in capsys.readouterr().err
