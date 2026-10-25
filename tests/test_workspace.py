"""Creator workspace contents, setup, and inbox isolation."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import tomllib
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from urllib.error import URLError

import pytest
from hygiene_patterns import VIDEO_TOOL_PATTERNS, _has_machine_details

from yap_studio import cli, install_info, setup_checks
from yap_studio.review import phone

pytestmark = pytest.mark.usefixtures("_no_real_node")

ROOT = Path(__file__).parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
WHEEL = f"yap_studio-{VERSION}-py3-none-any.whl"
FAKE_HOME = {"url": "http://127.0.0.1:9000/?t=synthetic", "phone": None}
EXPECTED = {
    "Start here.html",
    ".yap-studio-version",
    ".claude/commands/edit.md",
    ".claude/settings.json",
    ".codex/config.toml",
    ".codex/rules/yap-studio.rules",
    ".cursor/cli.json",
    ".cursor/permissions.json",
    ".claude/skills/edit-video/SKILL.md",
    ".claude/skills/voiceover-video/SKILL.md",
    "AGENTS.md",
    "CLAUDE.md",
    ".agents/skills/edit-video/SKILL.md",
    ".agents/skills/voiceover-video/SKILL.md",
    ".agents/skills/edit/SKILL.md",
    "LICENSE",
    "NOTICE",
    "brand/fonts/README.md",
    "inbox/README.md",
    "new_project.py",
    "setup.py",
    f"wheels/{WHEEL}",
}


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec.loader.exec_module(module)
    if path.name == "setup.py":
        module.NODE_INSTALL_PLACES = (str(Path(os.environ["TMPDIR"]) / "no-node-install"),)
        module.HOMEBREW_BIN_DIRS = (str(Path(os.environ["TMPDIR"]) / "no-homebrew"),)
    return module


def _synthetic_wheel(root: Path) -> Path:
    """Write a wheel with RECORD hashes for two synthetic package files."""
    wheel = root / "wheels" / WHEEL
    wheel.parent.mkdir(parents=True)
    files = {
        "yap_studio/__init__.py": f'__version__ = "{VERSION}"\n'.encode(),
        "yap_studio/cli.py": b"def main():\n    return 0\n",
    }
    rows = []
    for name, content in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        rows.append((name, f"sha256={digest}", str(len(content))))
    record_name = f"yap_studio-{VERSION}.dist-info/RECORD"
    rows.append((record_name, "", ""))
    record = io.StringIO()
    csv.writer(record, lineterminator="\n").writerows(rows)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr(record_name, record.getvalue())
    return wheel


def test_start_page_says_to_sign_in_to_claude_code_and_names_homebrew() -> None:
    page = (ROOT / "workspace/Start here.html").read_text(encoding="utf-8")
    assert "Pick your app" in page
    assert "Sign in once with <code>/login</code>" in page
    for phrase in ("choose Local", "Trust the folder once", "Agent mode on this Mac",
                   "Privacy Mode is suggested", "Chat apps (Claude Chat, Cowork, ChatGPT chat)"):
        assert phrase in page
    assert "install Homebrew first, for ffmpeg and Node" in page


def test_workspace_zip_uses_staged_content(tmp_path: Path) -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True,
    ).stdout.split(b"\0")
    repository = tmp_path / "repository"
    repository.mkdir()
    for raw_path in tracked:
        if not raw_path:
            continue
        relative = os.fsdecode(raw_path)
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for args in (["git", "init"], ["git", "add", "-A"]):
        subprocess.run(args, cwd=repository, capture_output=True, check=True)
    guidance = repository / "workspace/AGENTS.md"
    staged = guidance.read_bytes()
    guidance.write_bytes(b"Unstaged synthetic workspace guidance.\n")
    license_file = repository / "LICENSES/Apache-2.0.txt"
    staged_license = license_file.read_bytes()
    license_file.write_bytes(b"Unstaged synthetic license change.\n")
    package = repository / "src/yap_studio/__init__.py"
    staged_package = package.read_bytes()
    package.write_bytes(staged_package + b"# Unstaged synthetic package change.\n")
    metadata = repository / "pyproject.toml"
    metadata.write_text(metadata.read_text().replace(f'version = "{VERSION}"',
                                                    'version = "99.99.99"', 1))
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, "scripts/build_workspace.py", "--out", str(output)],
        cwd=repository, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True,
        timeout=120,
    )
    bundle = Path(result.stdout.strip().splitlines()[-1])
    assert bundle.name == f"yap-studio-workspace-{VERSION}.zip"
    with zipfile.ZipFile(bundle) as archive:
        assert archive.read("AGENTS.md") == staged
        assert archive.read("CLAUDE.md") == b"@AGENTS.md\n"
        assert archive.read("LICENSES/Apache-2.0.txt") == staged_license
        assert archive.read(".yap-studio-version") == f"{VERSION}\n".encode()
        with zipfile.ZipFile(archive.open(f"wheels/{WHEEL}")) as wheel:
            assert wheel.read("yap_studio/__init__.py") == staged_package


def test_every_tracked_workspace_file_is_bundled() -> None:
    builder = _module(ROOT / "scripts/build_workspace.py", "workspace_builder")
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "workspace/"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout.split(b"\0")
    assert {os.fsdecode(path) for path in tracked if path} <= builder.WORKSPACE_FILES.keys()


def _assert_clean_member(archive: zipfile.ZipFile, member: str, *, wheel: bool = False) -> None:
    try:
        content = archive.read(member).decode("utf-8")
    except UnicodeDecodeError:
        return
    relative = f"src/{member}" if wheel else f"workspace/{member}"
    assert not _has_machine_details(content, relative), member
    assert not any(pattern.search(content) for pattern in VIDEO_TOOL_PATTERNS), member
    license_text = (".dist-info/licenses/" in member if wheel
                    else member in {"LICENSE", "NOTICE"} or member.startswith("LICENSES/"))
    if not license_text:
        assert re.search(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", content, re.I) is None, member
    if not wheel:
        assert "node_modules" not in content, member


def test_workspace_zip_exact_files_and_clean_contents(tmp_path: Path) -> None:
    tracked_licenses = subprocess.run(
        ["git", "ls-files", "-z", "--", "LICENSES/"],
        cwd=ROOT, capture_output=True, check=True,
    ).stdout.split(b"\0")
    license_members = {os.fsdecode(path) for path in tracked_licenses if path}
    assert "LICENSES/Apache-2.0.txt" in license_members
    expected = EXPECTED | license_members
    result = subprocess.run(
        [sys.executable, "scripts/build_workspace.py", "--out", str(tmp_path)],
        cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True,
        timeout=120,
    )
    bundle = Path(result.stdout.strip().splitlines()[-1])
    assert bundle == tmp_path / f"yap-studio-workspace-{VERSION}.zip"
    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == expected
        for skill in ("edit-video", "voiceover-video"):
            member = f".agents/skills/{skill}/SKILL.md"
            assert archive.read(member) == archive.read(f".claude/skills/{skill}/SKILL.md")
            assert archive.getinfo(member).external_attr >> 16 & 0o170000 != 0o120000
        assert (archive.read(".agents/skills/edit/SKILL.md")
                == archive.read(".claude/commands/edit.md"))
        assert archive.read(".yap-studio-version") == f"{VERSION}\n".encode()
        wheel_file = tmp_path / WHEEL
        wheel_file.write_bytes(archive.read(f"wheels/{WHEEL}"))
        setup = _module(ROOT / "workspace/setup.py", "workspace_setup_bundled_wheel")
        assert re.fullmatch(r"[0-9a-f]{64}", setup._wheel_build(wheel_file) or "")
        for member in archive.namelist():
            _assert_clean_member(archive, member)
        with zipfile.ZipFile(archive.open(f"wheels/{WHEEL}")) as wheel:
            members = set(wheel.namelist())
            assert any(path.startswith("yap_studio/") for path in members)
            assert all(path.startswith(("yap_studio/", f"yap_studio-{VERSION}.dist-info/"))
                       for path in members)
            assert not any("node_modules/" in path or "site-packages/" in path
                           for path in members)
            for member in members:
                _assert_clean_member(wheel, member, wheel=True)


def _fake_project_runner(module: ModuleType, monkeypatch: pytest.MonkeyPatch,
                         tmp_path: Path) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> None:
        calls.append(args)
        project = Path(args[2])
        previous = json.loads((project / "project.json").read_text())["takes"] if (
            project / "project.json"
        ).exists() else []
        flag = "--voiceover" if "--voiceover" in args else "--take"
        paths = [args[i + 1] for i, arg in enumerate(args) if arg == flag]
        (project / "project.json").write_text(json.dumps({
            "schema": "yap-studio/project@1", "name": project.name,
            "takes": previous + [{"id": f"t{len(previous) + i + 1}",
                                  "path": (f".yap/voiceover/t{len(previous) + i + 1}.mp4"
                                           if flag == "--voiceover" else name),
                                  "voiceover": flag == "--voiceover"}
                                 for i, name in enumerate(paths)],
            "other": "preserved",
        }))

    monkeypatch.setattr(module, "_probe", lambda _path: None)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "yap-studio")
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tmp_path / "bin"))
    return calls


@pytest.mark.parametrize("named", [True, False])
def test_new_project_memos_keep_take_order_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    named: bool,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_memos")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for index, name in enumerate(("one.m4a", "two.wav")):
        (inbox / name).write_bytes(b"synthetic memo")
        os.utime(inbox / name, ns=(index + 1, index + 1))
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    names = ["two.wav", "one.m4a"] if named else ["one.m4a", "two.wav"]
    assert module.main([*(names if named else ["--newest", "2"]), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["voiceover"] is True
    assert result["takes"] == names
    assert calls[0][3:] == [value for name in names for value in (
        "--voiceover", str(Path(result["project"]) / "takes" / name))]


@pytest.mark.parametrize("named", [True, False])
def test_new_project_refuses_mixed_kinds_without_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: bool,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_mixed")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in ("memo.m4a", "take.mov"):
        (inbox / name).write_bytes(b"synthetic")
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="audio memos|Audio memos") as error:
        module.start(tmp_path, takes=["memo.m4a", "take.mov"] if named else None, newest=2)
    if not named:
        assert "memo.m4a" in str(error.value) and "take.mov" in str(error.value)
    assert not calls
    assert not (inbox / ".used.json").exists()
    assert not [p for p in (tmp_path / "projects").iterdir() if p.is_dir()]


def test_new_project_pickups_copy_collisions_and_are_not_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_pickups")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "memo.m4a").write_bytes(b"synthetic memo")
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    project = module.start(tmp_path, ["memo.m4a"])
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert module.main(["--add", str(project), "memo.m4a", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["voiceover"] is True and result["takes"] == ["memo.m4a", "memo.m4a"]
    assert calls[-1][3:] == ["--voiceover", str(project / "takes/memo-2.m4a")]
    assert (project / "takes/memo-2.m4a").read_bytes() == b"synthetic memo"
    (inbox / "pickup.wav").write_bytes(b"pickup")
    (inbox / "video.mov").write_bytes(b"video")
    module.add_pickups(tmp_path, project, None, 1)
    assert calls[-1][3:] == ["--voiceover", str(project / "takes/pickup.wav")]
    ledger = module._used(inbox)
    assert [item["file"] for item in ledger["takes"]] == [
        "memo.m4a", "memo.m4a", "pickup.wav"]
    assert all(item["project"] == f"projects/{project.name}" for item in ledger["takes"])
    with pytest.raises(ValueError, match="too few unused"):
        module.add_pickups(tmp_path, project, None, 1)
    assert module.main(["--add", str(project), "--name", str(project), "topic"]) == 1
    assert "Do not combine --add and --name." in capsys.readouterr().err


def test_new_project_pickups_refusals_and_failure_keep_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_pickup_failure")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in ("memo.m4a", "video.mov"):
        (inbox / name).write_bytes(b"synthetic")
    _fake_project_runner(module, monkeypatch, tmp_path)
    video = module.start(tmp_path, ["video.mov"])
    with pytest.raises(ValueError, match="is not a voiceover project"):
        module.add_pickups(tmp_path, video, ["memo.m4a"], 1)
    project = module.start(tmp_path, ["memo.m4a"])
    with pytest.raises(ValueError, match="video.mov is a video take"):
        module.add_pickups(tmp_path, project, ["video.mov"], 1)
    before = {p.relative_to(project): p.read_bytes() for p in project.rglob("*") if p.is_file()}
    ledger = (inbox / ".used.json").read_bytes()

    def fail(args: list[str], **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.add_pickups(tmp_path, project, ["memo.m4a"], 1)
    assert {p.relative_to(project): p.read_bytes() for p in project.rglob("*")
            if p.is_file()} == before
    assert (inbox / ".used.json").read_bytes() == ledger


def test_new_project_pickups_ledger_failure_restores_manifest_and_standins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_pickup_rollback")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "memo.m4a").write_bytes(b"synthetic memo")
    _fake_project_runner(module, monkeypatch, tmp_path)
    project = module.start(tmp_path, ["memo.m4a"])
    manifest = project / "project.json"
    manifest.write_text(json.dumps(json.loads(manifest.read_text()), indent=3) + "\n\n")
    standins = project / ".yap/voiceover"
    standins.mkdir(parents=True)
    (standins / "t1.mp4").write_bytes(b"original stand-in")
    before = manifest.read_bytes()
    original_standins = {path.name: path.read_bytes() for path in standins.iterdir()}
    ledger = (inbox / ".used.json").read_bytes()
    original_run = module.subprocess.run
    original_write = module._write_json

    def append(args: list[str], **kwargs: object) -> None:
        original_run(args, **kwargs)
        take = json.loads(manifest.read_text())["takes"][-1]
        (standins / f"{take['id']}.mp4").write_bytes(b"pickup stand-in")

    def fail_ledger(target: Path, data: dict) -> None:
        if target.name == ".used.json":
            raise OSError("synthetic ledger failure")
        original_write(target, data)

    monkeypatch.setattr(module.subprocess, "run", append)
    monkeypatch.setattr(module, "_write_json", fail_ledger)
    with pytest.raises(OSError, match="synthetic ledger failure"):
        module.add_pickups(tmp_path, project, ["memo.m4a"], 1)
    assert manifest.read_bytes() == before
    assert {path.name: path.read_bytes() for path in standins.iterdir()} == original_standins
    assert (inbox / ".used.json").read_bytes() == ledger
    assert not (project / "takes/memo-2.m4a").exists()
    monkeypatch.setattr(module, "_write_json", original_write)
    module.add_pickups(tmp_path, project, ["memo.m4a"], 1)
    assert [take["id"] for take in json.loads(manifest.read_text())["takes"]] == ["t1", "t2"]
    assert {path.name for path in standins.iterdir()} == {"t1.mp4", "t2.mp4"}
    assert len(module._used(inbox)["takes"]) == 2


@pytest.mark.parametrize(("suffix", "video", "cover", "expected"), [
    (".m4a", False, False, "memo"), (".mp4", False, False, "memo"),
    (".m4a", True, False, "take"), (".mp3", False, True, "memo"),
])
def test_new_project_kind_probes_streams(
    tmp_path: Path, suffix: str, video: bool, cover: bool, expected: str,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is missing")
    module = _module(ROOT / "workspace/new_project.py", "workspace_probe")
    target = tmp_path / f"memo{suffix}"
    args = [ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=duration=0.5"]
    if video or cover:
        args += ["-f", "lavfi", "-i", "color=s=16x16:r=2:d=0.5", "-map", "0:a", "-map", "1:v"]
    if video:
        args += ["-c:v", "mpeg4", "-f", "mp4"]
    if cover:
        args += ["-c:v", "mjpeg", "-frames:v", "1", "-disposition:v", "attached_pic"]
    subprocess.run([*args, str(target)], check=True, capture_output=True)
    assert module._kind(target) == expected


def test_voiceover_skill_commands_parse_with_the_engine() -> None:
    skill = (ROOT / "workspace/.claude/skills/voiceover-video/SKILL.md").read_text()
    assert 'yap-studio run DIR --note "Final" --json --detach' in skill
    assert "yap-studio status DIR --json --wait 540" in skill
    assert "30 minutes" in skill
    assert "before choosing B-roll" in skill and "children, friends" in skill
    assert "may appear on screen" in skill
    assert "Never reuse a shot for the closing frames." in skill
    assert "graphic or diagram has an obvious home in the script" in skill
    assert "put it there and say so" in skill
    for command in re.findall(r"`(yap-studio [^`]+)`", " ".join(skill.split())):
        try:
            arguments = ["300" if token == "N" else token for token in shlex.split(command)[1:]]
            parser, arguments = cli._argument_parser(arguments)
            parser.parse_args(arguments)
        except SystemExit:
            pytest.fail(f"Skill command does not parse: {command}")


def test_edit_skill_card_example_parses_with_the_engine() -> None:
    from yap_studio import cards
    from yap_studio.planner.cards import parse_requests

    skill = (ROOT / "workspace/.claude/skills/edit-video/SKILL.md").read_text()
    examples = [json.loads(text) for text in re.findall(r"```json\n(.*?)\n\s*```", skill, re.S)]
    example = next(data for data in examples if data.get("schema") == "yap-studio/card-requests@1")
    card, = parse_requests(example)
    assert card.id.startswith("agent-")
    assert card.start.word == "t1:12"
    assert card.template == "text_tag"
    slots, _ = cards._slots(card.slots, cards.load_template(card.template)["slots"])
    assert slots["text"] == "Technical terms"
    assert slots["variant"] == "sign"


def test_edit_skill_title_example_parses_with_the_engine() -> None:
    from yap_studio.project.model import ProjectFileError, parse_document

    skill = (ROOT / "workspace/.claude/skills/edit-video/SKILL.md").read_text()
    examples = [json.loads(text) for text in re.findall(r"```json\n(.*?)\n\s*```", skill, re.S)]
    example = next(data for data in examples if data.get("schema") == "yap-studio/timeline@1")
    title, = parse_document("timeline-fragment", example, lane="agent").elements
    assert title.kind == "title"
    assert title.lane == "agent"
    assert title.id.startswith("agent-")
    assert title.start.word is not None
    assert 1 <= len(title.fields["lines"]) <= 3
    assert all(isinstance(line, str) for line in title.fields["lines"])
    del example["elements"][0]["lane"]
    with pytest.raises(ProjectFileError, match="lane.*required"):
        parse_document("timeline-fragment", example, lane="agent")


def test_new_project_copies_newest_unused_take_and_never_reuses_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_new_project")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    old = inbox / "old.mov"
    new = inbox / "new.mp4"
    old.write_bytes(b"older synthetic take")
    new.write_bytes(b"newer synthetic take")
    old.touch()
    new.touch()
    old_time = old.stat().st_mtime_ns - 1_000_000_000
    os.utime(old, ns=(old_time, old_time))
    originals = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (old, new)}
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    first = module.start(tmp_path)
    second = module.start(tmp_path)
    assert first != second
    assert (first / "takes/new.mp4").read_bytes() == originals["new.mp4"][0]
    assert (second / "takes/old.mov").read_bytes() == originals["old.mov"][0]
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (old, new)} == originals
    assert {p.name for p in inbox.iterdir()} == {"old.mov", "new.mp4", ".used.json"}
    used = json.loads((inbox / ".used.json").read_text())
    assert used["schema"] == "yap-studio/inbox-used@1"
    assert [item["file"] for item in used["takes"]] == ["new.mp4", "old.mov"]
    assert len(calls) == 2
    assert all(call[:2] == ["yap-studio", "init"] for call in calls)
    with pytest.raises(ValueError, match="already in a project.*Name"):
        module.start(tmp_path)


def test_new_project_passes_answers_only_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_answers")
    (tmp_path / "inbox").mkdir()
    (tmp_path / "inbox/take.mov").write_bytes(b"synthetic take")
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    module.start(tmp_path)
    assert "--answers" not in calls[-1]
    (tmp_path / "style").mkdir()
    (tmp_path / "style/answers.json").write_text('{"schema":"yap-studio/style-answers@1"}')
    module.start(tmp_path, takes=["take.mov"])
    assert calls[-1][-2:] == ["--answers", "style/answers.json"]


def test_new_project_video_flags_and_json(tmp_path, monkeypatch, capsys):
    module = _module(ROOT / "workspace/new_project.py", "workspace_video_type")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "inbox").mkdir()
    (tmp_path / "inbox/take.mov").write_bytes(b"synthetic take")
    (tmp_path / "style").mkdir()
    (tmp_path / "style/answers.json").write_text("{}")
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    original_run = module.subprocess.run

    def run(args, **kwargs):
        original_run(args, **kwargs)
        manifest = Path(args[2]) / "project.json"
        data = json.loads(manifest.read_text())
        data.update(video_type="green_screen", broll_sources=["folder", "stock"])
        manifest.write_text(json.dumps(data))

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main(["--type", "green_screen", "--broll", "folder",
                        "--broll", "stock", "--json"]) == 0
    assert calls[-1][-8:] == ["--video-type", "green_screen", "--broll", "folder",
                              "--broll", "stock", "--answers", "style/answers.json"]
    result = json.loads(capsys.readouterr().out)
    assert result["video_type"] == "green_screen" and result["broll_sources"] == ["folder", "stock"]


@pytest.mark.parametrize("sources", [None, ["folder"], ["album", "stock"], ["none"]])
@pytest.mark.parametrize("branded", [False, True])
def test_new_project_horizontal_records_takes_and_style(tmp_path, monkeypatch, sources, branded):
    from yap_studio.homepage.state import inbox_files
    from yap_studio.style import resolve_style_data

    module = _module(ROOT / "workspace/new_project.py", "workspace_horizontal")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    names = ["first.mov", "second.mp4"]
    for name in names:
        (inbox / name).write_bytes(name.encode())
    (tmp_path / "style").mkdir()
    (tmp_path / "style/answers.json").write_text("{}")
    if branded:
        (tmp_path / "style/my-style.json").write_text(json.dumps({
            "schema_version": 1, "name": "Brand", "hook": {"font": "Gelasio"},
            "graphics": {"font": "Inter", "accent_color": "#AB1256"},
        }))
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    flags = [value for source in sources or [] for value in ("--broll", source)]
    assert module.main([*names, "--type", "talking_head", "--shape", "horizontal", *flags]) == 0
    project = Path(calls[0][2])
    assert calls[0][3:] == [
        *[value for name in names for value in ("--take", str(project / "takes" / name))],
        "--shape", "horizontal", "--video-type", "talking_head",
        "--style", "preset:clean_minimal",
        *[value for source in sources or ["none"] for value in ("--broll", source)],
    ]
    used = json.loads((inbox / ".used.json").read_text())["takes"]
    assert [item["file"] for item in used] == names
    for item in used:
        source = inbox / item["file"]
        assert (item["bytes"], item["mtime_ns"], item["project"]) == (
            source.stat().st_size, source.stat().st_mtime_ns, f"projects/{project.name}")
        assert (project / "takes" / source.name).read_bytes() == source.read_bytes()
    manifest = json.loads((project / "project.json").read_text())
    assert manifest["platform"] == "youtube"
    style_path = project / manifest["style"]
    assert style_path.is_relative_to(project)
    style_data = json.loads(style_path.read_text())
    assert style_data["captions"] == {"camera_3d": False}
    assert "taste" not in style_data
    style = resolve_style_data(style_data)
    if branded:
        assert style["hook"]["font"] == "Gelasio"
        assert style["graphics"]["accent_color"] == "#AB1256"
    assert style["hook"]["style"] == style["chapters"]["style"] == "none"
    assert style["grade"]["look"] == style["moves"]["zoom_mode"] == "none"
    assert style["moves"]["jump_cuts"] == "hard"
    assert style["moves"]["opening_push"] is False and style["moves"]["transitions"] == []
    assert inbox_files(tmp_path) == ([], 2)
    with pytest.raises(ValueError, match="already in a project"):
        module.start(tmp_path, shape="horizontal")


@pytest.mark.parametrize("takes,flags,fix", [
    (["memo.wav"], [], "Use video takes"),
    (["take.mov", "memo.wav"], [], "Use video takes"),
    (["take.mov"], ["--type", "green_screen"], "Use --type talking_head"),
])
def test_new_project_horizontal_refuses_unsupported(
    tmp_path, monkeypatch, capsys, takes, flags, fix,
):
    module = _module(ROOT / "workspace/new_project.py", "workspace_horizontal_refusal")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "inbox").mkdir()
    for name in takes:
        (tmp_path / "inbox" / name).write_bytes(b"take")
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    assert module.main([*takes, "--shape", "horizontal", *flags]) == 1
    assert fix in capsys.readouterr().err
    assert calls == [] and not (tmp_path / "inbox/.used.json").exists()
    assert not any(p.is_dir() for p in (tmp_path / "projects").iterdir())


@pytest.mark.parametrize("failed_file", ["init", "horizontal.json", "project.json", ".used.json"])
def test_new_project_horizontal_failure_rolls_back(tmp_path, monkeypatch, failed_file):
    module = _module(ROOT / "workspace/new_project.py", "workspace_horizontal_failure")
    (tmp_path / "inbox").mkdir()
    source = tmp_path / "inbox/take.mov"
    source.write_bytes(b"take")
    _fake_project_runner(module, monkeypatch, tmp_path)
    write = module._write_json
    run = module.subprocess.run

    def fail_write(target, data):
        assert not (tmp_path / "inbox/.used.json").exists()
        if target.name == failed_file:
            raise OSError("setup failed")
        write(target, data)

    def fail_init(args, **kwargs):
        if failed_file == "init":
            raise OSError("setup failed")
        return run(args, **kwargs)

    monkeypatch.setattr(module, "_write_json", fail_write)
    monkeypatch.setattr(module.subprocess, "run", fail_init)
    with pytest.raises(OSError, match="setup failed"):
        module.start(tmp_path, shape="horizontal")
    assert not any(p.is_dir() for p in (tmp_path / "projects").iterdir())
    assert not (tmp_path / "inbox/.used.json").exists()
    assert source.read_bytes() == b"take"


@pytest.mark.parametrize("shape", [None, "vertical"])
def test_new_project_vertical_init_unchanged(tmp_path, monkeypatch, capsys, shape):
    module = _module(ROOT / "workspace/new_project.py", "workspace_vertical_unchanged")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "inbox").mkdir()
    (tmp_path / "inbox/take.mov").write_bytes(b"take")
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    assert module.main([*(["--shape", shape] if shape else [])]) == 0
    project = Path(calls[0][2])
    assert calls == [["yap-studio", "init", str(project), "--take",
                      str(project / "takes/take.mov")]]
    assert capsys.readouterr().out == f"{project}\n"


@pytest.mark.parametrize("mode", [["--name", "DIR", "TOPIC"], ["--add", "DIR"]])
@pytest.mark.parametrize("flags", [["--type", "talking_head"], ["--broll", "none"],
                                   ["--shape", "horizontal"]])
def test_new_project_refuses_video_flags_with_name_or_add(mode, flags, capsys):
    module = _module(ROOT / "workspace/new_project.py", "workspace_video_flags")
    assert module.main([*mode, *flags]) == 1
    assert "Do not combine" in capsys.readouterr().err


def test_workspace_video_routing_and_commands_parse():
    guide = (ROOT / "workspace/AGENTS.md").read_text()
    for phrase in ("1280x720 YouTube thumbnail", "headline as plain text in chat and wait",
                   "creator to choose the text before rendering", "168x94 preview",
                   "bottom-right duration badge", "--look before-after"):
        assert phrase in guide
    assert "## What are you making?" in guide
    section = guide.split("## What are you making?", 1)[1].split("## Make a voiceover video", 1)[0]
    for phrase in ("yap-studio video-type show --json", "your app's question tool",
                   "not in this version",
                   "talking_head", "voiceover", "green_screen", "screen_recording",
                   "Behind the speaker", "cover every frame", "using its display dimensions"):
        assert phrase in section
    assert "python3 new_project.py TAKE ... --type talking_head --shape horizontal" in guide
    assert "yap-studio init DIR --take FILE --shape horizontal" not in guide
    assert "payload.captions` is not true" in guide
    assert "[Caption gaps](#caption-gaps)" in guide
    assert "yap-studio timeline add FRAGMENT DIR --lane agent" in guide
    skill = (ROOT / "workspace/.claude/skills/edit-video/SKILL.md").read_text()
    assert "broll_sources" in skill and "`stock` is in this video's B-roll sources" in skill
    files = ["workspace/AGENTS.md", "workspace/.claude/skills/edit-video/SKILL.md",
             "workspace/.claude/skills/voiceover-video/SKILL.md",
             "workspace/.claude/commands/edit.md"]
    substitutions = {"DIR": "project", "TYPE": "talking_head", "SOURCE": "folder",
                     "N": "300", "T": "1", "X": "0.5", "Y": "0.5", "TOTAL": "0.081"}
    # Bare command names in prose need representative required arguments.
    examples = {
        "yap-studio brand fonts add": "yap-studio brand fonts add FONT.ttf",
        "yap-studio media album": 'yap-studio media album "ALBUM" --out FOLDER',
        "yap-studio init": "yap-studio init DIR --take TAKE.mov",
        "yap-studio feedback ask": (
            'yap-studio feedback ask DIR --batch b1 --comment v1:c1 --question "Which part?"'),
        "yap-studio timeline add": "yap-studio timeline add FRAGMENT DIR --lane broll",
    }
    pictures_commands = set()
    for filename in files:
        text = " ".join((ROOT / filename).read_text().split())
        for command in re.findall(r"`(yap-studio [^`]+)`", text):
            arguments = [substitutions.get(token, token)
                         for token in shlex.split(examples.get(command, command))[1:]]
            if command.startswith(("yap-studio video-type", "yap-studio stock get",
                                   "yap-studio pictures make")):
                arguments = [token for token in arguments if token != "..."]
            parser, arguments = cli._argument_parser(arguments)
            try:
                parser.parse_args(arguments)
            except SystemExit:
                pytest.fail(f"Guide command does not parse: {command}")
            if arguments[:1] == ["pictures"]:
                pictures_commands.add(arguments[1])
    assert pictures_commands == {"question", "answer", "ask", "make"}


def test_workspace_asks_for_unchosen_broll_after_type():
    guide = " ".join((ROOT / "workspace/AGENTS.md").read_text().split())
    assert ("If `yap-studio status DIR` says B-roll is not chosen yet, ask only `broll_sources` "
            "for this video and record it with "
            "`yap-studio video-type set DIR --broll SOURCE ...`.") in guide
    edit = " ".join((ROOT / "workspace/.claude/skills/edit-video/SKILL.md").read_text().split())
    assert "type or B-roll is not chosen yet" in edit
    assert "ask only `broll_sources`" in edit
    assert "yap-studio video-type set DIR --broll SOURCE ..." in edit
    voiceover = " ".join(
        (ROOT / "workspace/.claude/skills/voiceover-video/SKILL.md").read_text().split())
    assert ("If B-roll is not chosen yet, ask only the B-roll question as in "
            "What are you making? in `AGENTS.md`.") in voiceover


def test_setup_questions_are_one_at_a_time_and_saved() -> None:
    guide = (ROOT / "workspace/AGENTS.md").read_text()
    for phrase in ("yap-studio style questions", "question tool", "one question at a time",
                   "style/answers.json", "yap-studio style from-answers", "Update my style"):
        assert phrase in guide
    assert "`never.other` is a list" in guide
    assert "use `[]` when there are none" in guide
    assert "Setup starts it" in guide
    assert "creator is not using the page" in guide
    assert "yap-studio home wait --workspace . --json" in guide
    assert "Never wait for its result before replying in chat" in guide
    assert '"You\'re set up" with the home link' in guide
    skill = (ROOT / "workspace/.claude/skills/edit-video/SKILL.md").read_text()
    assert 'reply "You\'re set up" with the home link in the same turn' in skill
    assert "Start the home wait as a background task" in " ".join(skill.split())


def test_new_project_replaced_file_is_unused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_replaced")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    take = inbox / "same.mp4"
    take.write_bytes(b"first")
    _fake_project_runner(module, monkeypatch, tmp_path)
    first = module.start(tmp_path)
    take.write_bytes(b"second version")
    second = module.start(tmp_path)
    assert first != second
    assert (second / "takes/same.mp4").read_bytes() == b"second version"
    assert len(json.loads((inbox / ".used.json").read_text())["takes"]) == 2


def test_new_project_newest_multiple_are_oldest_first_and_json_lists_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_multiple")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for index, name in enumerate(("old.mp4", "middle.mov", "new.mkv")):
        take = inbox / name
        take.write_bytes(name.encode())
        os.utime(take, ns=(1_000_000_000 + index, 1_000_000_000 + index))
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert module.main(["--newest", "2", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["takes"] == ["middle.mov", "new.mkv"]
    assert Path(result["project"]).name.endswith("-middle")
    assert result["next"].startswith("yap-studio ingest ")
    assert [Path(value).name for value in calls[0][4::2]] == result["takes"]


def test_new_project_named_takes_keep_order_and_can_be_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_named")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in ("one.mp4", "two.mov"):
        (inbox / name).write_bytes(name.encode())
    calls = _fake_project_runner(module, monkeypatch, tmp_path)
    first = module.start(tmp_path, takes=["two.mov", "one.mp4"])
    assert calls[0][3:] == ["--take", str(first / "takes/two.mov"),
                            "--take", str(first / "takes/one.mp4")]
    second = module.start(tmp_path, takes=["two.mov"])
    assert second != first
    assert (second / "takes/two.mov").read_bytes() == b"two.mov"
    assert [item["file"] for item in json.loads((inbox / ".used.json").read_text())["takes"]] == [
        "two.mov", "one.mp4", "two.mov",
    ]


def test_new_project_unknown_take_lists_inbox_without_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_unknown")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "there.mp4").write_bytes(b"take")
    _fake_project_runner(module, monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="there.mp4"):
        module.start(tmp_path, takes=["missing.mp4"])
    assert not any(path.is_dir() for path in (tmp_path / "projects").iterdir())
    assert not (inbox / ".used.json").exists()


def test_new_project_init_failure_and_corrupt_ledger_do_not_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_failure")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "take.mp4").write_bytes(b"take")
    original = {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in inbox.iterdir()}

    def fail(args: list[str], **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(2, args)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.start(tmp_path)
    assert not any(path.is_dir() for path in (tmp_path / "projects").iterdir())
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in inbox.iterdir()} == original
    assert not (inbox / ".used.json").exists()
    ledger = inbox / ".used.json"
    ledger.write_text("not json")
    with pytest.raises(ValueError, match=".used.json"):
        module.start(tmp_path)
    assert ledger.read_text() == "not json"


def test_new_project_copy_failure_removes_partial_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_copy_failure")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    take = inbox / "take.mp4"
    take.write_bytes(b"take")
    original = take.read_bytes(), take.stat().st_mtime_ns

    def fail(_source: Path, destination: Path) -> None:
        destination.write_bytes(b"partial")
        raise OSError("copy failed")

    monkeypatch.setattr(module.shutil, "copy2", fail)
    with pytest.raises(OSError, match="copy failed"):
        module.start(tmp_path)
    assert not any(path.is_dir() for path in (tmp_path / "projects").iterdir())
    assert (take.read_bytes(), take.stat().st_mtime_ns) == original
    assert not (inbox / ".used.json").exists()


@pytest.mark.parametrize("shape", ["vertical", "horizontal"])
def test_new_project_waits_for_project_lock_and_reads_ledger_inside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str,
) -> None:
    fcntl = pytest.importorskip("fcntl")
    module = _module(ROOT / "workspace/new_project.py", "workspace_locked")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in ("one.mp4", "two.mp4"):
        (inbox / name).write_bytes(name.encode())
    _fake_project_runner(module, monkeypatch, tmp_path)
    projects = tmp_path / "projects"
    projects.mkdir()
    done = threading.Event()
    result: list[Path] = []
    errors: list[BaseException] = []

    def run_start() -> None:
        try:
            result.append(module.start(tmp_path, shape=shape))
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    with (projects / ".new_project.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        worker = threading.Thread(target=run_start)
        worker.start()
        try:
            assert not done.wait(0.2)
            assert not (inbox / ".used.json").exists()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors == []
    assert len(result) == 1
    second = module.start(tmp_path, shape=shape)
    assert second != result[0]
    used = json.loads((inbox / ".used.json").read_text())["takes"]
    assert len(used) == 2
    assert {item["file"] for item in used} == {"one.mp4", "two.mp4"}


def test_new_project_changed_during_copy_removes_project_and_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_changing")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    take = inbox / "take.mp4"
    take.write_bytes(b"first")
    original_copy = module.shutil.copy2
    _fake_project_runner(module, monkeypatch, tmp_path)

    def change_source(source: Path, destination: Path) -> None:
        original_copy(source, destination)
        source.write_bytes(b"second version")
        later = source.stat().st_mtime_ns + 1_000_000_000
        os.utime(source, ns=(later, later))

    monkeypatch.setattr(module.shutil, "copy2", change_source)
    with pytest.raises(ValueError, match="take.mp4 changed while it was being copied"):
        module.start(tmp_path)
    assert not any(path.is_dir() for path in (tmp_path / "projects").iterdir())
    assert not (inbox / ".used.json").exists()


def test_new_project_ledger_write_failure_removes_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_ledger_failure")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "take.mp4").write_bytes(b"take")
    _fake_project_runner(module, monkeypatch, tmp_path)

    def fail(_target: Path, _data: dict) -> None:
        raise OSError("ledger write failed")

    monkeypatch.setattr(module, "_write_json", fail)
    with pytest.raises(OSError, match="ledger write failed"):
        module.start(tmp_path)
    assert not any(path.is_dir() for path in (tmp_path / "projects").iterdir())
    assert not (inbox / ".used.json").exists()


@pytest.mark.parametrize("failed_file", [".used.json", "project.json"])
def test_new_project_rename_write_failure_restores_folder_manifest_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_file: str,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_rename_failure")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "take.mp4").write_bytes(b"take")
    _fake_project_runner(module, monkeypatch, tmp_path)
    project = module.start(tmp_path)
    old_manifest = (project / "project.json").read_text()
    old_ledger = (inbox / ".used.json").read_text()
    original_write = module._write_json

    def fail_once(target: Path, data: dict) -> None:
        if target.name == failed_file:
            monkeypatch.setattr(module, "_write_json", original_write)
            raise OSError(f"{failed_file} write failed")
        original_write(target, data)

    monkeypatch.setattr(module, "_write_json", fail_once)
    with pytest.raises(OSError, match="write failed"):
        module.rename_project(tmp_path, project, "New Topic")
    assert project.is_dir()
    assert (project / "project.json").read_text() == old_manifest
    assert (inbox / ".used.json").read_text() == old_ledger
    assert not (project.parent / (project.name[:10] + "-new-topic")).exists()
    renamed = module.rename_project(tmp_path, project, "New Topic")
    assert renamed != project
    assert json.loads((inbox / ".used.json").read_text())["takes"][0]["project"] == (
        "projects/" + renamed.name)


def test_new_project_name_updates_manifest_and_refuses_after_story(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_rename")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "first.mp4").write_bytes(b"first")
    _fake_project_runner(module, monkeypatch, tmp_path)
    first = module.start(tmp_path)
    renamed = module.rename_project(tmp_path, first, "Inference Basics")
    assert renamed.name.endswith("-inference-basics")
    assert json.loads((renamed / "project.json").read_text())["name"] == renamed.name
    assert json.loads((renamed / "project.json").read_text())["other"] == "preserved"
    assert json.loads((inbox / ".used.json").read_text())["takes"][0]["project"] == (
        "projects/" + renamed.name)
    assert module.rename_project(tmp_path, renamed, "Inference Basics") == renamed
    monkeypatch.setattr(module, "ROOT", tmp_path)
    assert module.main(["--name", str(renamed), "Inference Basics", "--json"]) == 0
    second = module.start(tmp_path, takes=["first.mp4"])
    collision = module.rename_project(tmp_path, second, "Inference Basics")
    assert collision.name == renamed.name + "-2"
    (collision / "edit.json").write_text("{}")
    with pytest.raises(ValueError, match="story"):
        module.rename_project(tmp_path, collision, "Other")
    (collision / "edit.json").unlink()
    (collision / "cut").mkdir()
    with pytest.raises(ValueError, match="story"):
        module.rename_project(tmp_path, collision, "Other")
    assert module.main(["--name", str(collision), "Other"]) == 1
    assert "Could not name the project:" in capsys.readouterr().err


def test_ingest_survives_project_rename_and_words_remain_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from test_ingest import FakeBackend, make_media

    from yap_studio import transcript
    from yap_studio.project import init_project

    module = _module(ROOT / "workspace/new_project.py", "workspace_real_rename")
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_args: FakeBackend())
    (tmp_path / "inbox").mkdir()
    project = tmp_path / "projects" / "2026-10-12-first"
    source = make_media(project / "takes" / "first.mkv", seconds=3)
    init_project(project, takes=[source])
    assert cli.main(["ingest", str(project)]) == 0
    capsys.readouterr()
    renamed = module.rename_project(tmp_path, project, "Inference")
    assert cli.main(["status", str(renamed), "--json"]) == 0
    stages = json.loads(capsys.readouterr().out)["stages"]
    assert next(stage for stage in stages if stage["name"] == "ingest")["state"] == "done"
    assert cli.main(["words", str(renamed)]) == 0
    assert "Here's the thing nobody tells you." in capsys.readouterr().out


def test_new_project_imports_workspace_fonts_before_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_new_project_fonts")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "take.mp4").write_bytes(b"synthetic take")
    folder = tmp_path / "brand" / "fonts"
    folder.mkdir(parents=True)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(module, "_probe", lambda _path: None)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "yap-studio")
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tmp_path / "bin"))
    module.start(tmp_path)
    assert calls[0] == (["yap-studio", "brand", "fonts", "add", str(folder), "--replace"],
                        {"cwd": tmp_path, "check": True})
    assert calls[1][0][1] == "init"
    calls.clear()
    folder.rmdir()
    module.start(tmp_path, takes=["take.mp4"])
    assert len(calls) == 1
    assert calls[0][0][1] == "init"


def test_new_project_imports_workspace_words_before_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_new_project_words")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "take.mp4").write_bytes(b"synthetic take")
    words = tmp_path / "brand" / "words.txt"
    words.parent.mkdir()
    words.write_text("SparrowLight\n")
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> None:
        calls.append(args)

    monkeypatch.setattr(module, "_probe", lambda _path: None)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "yap-studio")
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tmp_path / "bin"))
    module.start(tmp_path)
    assert calls[0] == ["yap-studio", "brand", "words", "add", "--file", str(words)]
    assert calls[1][1] == "init"


def test_new_project_bad_font_leaves_no_project_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_new_project_bad_fonts")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "take.mp4").write_bytes(b"synthetic take")
    fonts = tmp_path / "brand" / "fonts"
    fonts.mkdir(parents=True)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> None:
        calls.append(args)
        assert kwargs == {"cwd": tmp_path, "check": True}
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(module, "_probe", lambda _path: None)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "yap-studio")
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tmp_path / "bin"))
    with pytest.raises(subprocess.CalledProcessError):
        module.start(tmp_path)
    assert calls == [["yap-studio", "brand", "fonts", "add", str(fonts), "--replace"]]
    projects = tmp_path / "projects"
    assert not projects.exists() or not any(path.is_dir() for path in projects.iterdir())


def test_setup_reports_one_command_for_each_missing_piece(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup")
    monkeypatch.setattr(module, "_open_home", lambda engine, phone: FAKE_HOME)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_synced_service", lambda path, home=None: None)
    monkeypatch.setattr(module, "_free_disk_gb", lambda path: 20.0)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / WHEEL).write_bytes(b"synthetic wheel")
    (tmp_path / ".yap-studio-version").write_text(VERSION + "\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "yap-studio").touch()
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(bin_dir))
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/opt/homebrew/bin/uv")
    report = {
        "install": {"version": VERSION}, "warnings": [], "problems": ["ffmpeg missing"],
        "ffmpeg": {"available": False}, "ffprobe": {"available": False},
        "filters": {"ebur128": False}, "node": {"available": False},
        "renderer": {"available": False, "browser": False},
        "speech": {"available": False},
        "edit_missing": ["node", "renderer", "browser", "speech"], "ready_to_edit": False,
    }
    calls: list[list[str]] = []
    installed_version = [VERSION]

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {installed_version[0]}\n", "")
        if args[1:4] == ["tool", "install", "--force"]:
            installed_version[0] = VERSION
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 2, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main() == 1
    assert calls[0] == [str(bin_dir / "yap-studio"), "--version"]
    assert calls[1] == [str(bin_dir / "yap-studio"), "doctor", "--edit", "--json"]
    assert len(calls) == 2
    output = capsys.readouterr().out
    for label in ("Install ffmpeg, which reads and writes video files.",
                  "Install Node 22 or newer to run the renderer.",
                  "Install the renderer and browser for video graphics.",
                  "Install the speech engine and download its model to transcribe takes."):
        assert output.count(label) == 1
    assert [line.split(":", 1)[0] for line in output.splitlines()
            if line.startswith("Step ")] == [f"Step {i} of 4" for i in range(1, 5)]
    assert "brew install node@22" in output
    assert "brew link" not in output
    assert "Ready to edit" not in output
    assert module.main(["--json"]) == 1
    steps = json.loads(capsys.readouterr().out)["steps"]
    speech_step = steps[-1]
    assert speech_step["id"] == "speech"
    assert shlex.split(speech_step["command"]) == [
        "/opt/homebrew/bin/uv", "tool", "install", "--force",
        f"yap-studio[asr] @ {(wheels / WHEEL).as_uri()}", "&&",
        str(bin_dir / "yap-studio"), "speech", "download",
    ]
    calls.clear()
    assert module.main() == 1
    assert calls == [
        [str(bin_dir / "yap-studio"), "--version"],
        [str(bin_dir / "yap-studio"), "doctor", "--edit", "--json"],
    ]
    capsys.readouterr()
    report.update({
        "problems": [], "ffmpeg": {"available": True}, "ffprobe": {"available": True},
        "filters": {"ebur128": True}, "node": {"available": True},
        "renderer": {"available": True, "browser": True}, "speech": {"available": True},
        "edit_missing": [], "ready_to_edit": True,
    })
    # A failing doctor that reports ready is unexpected; pass its exit code through.
    assert module.main() == 2
    assert "Ready to edit" not in capsys.readouterr().out

    def ready_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = fake_run(args, **kwargs)
        return result if args[1] != "doctor" else subprocess.CompletedProcess(
            args, 0, result.stdout, "")

    monkeypatch.setattr(module.subprocess, "run", ready_run)
    assert module.main() == 0
    assert "Ready to edit" in capsys.readouterr().out
    assert module.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "ready": True, "refused": None, "steps": [], "inbox_takes": 0,
        "phone": {"via": None, "tailscale": "missing"}, "home": FAKE_HOME,
        "engine": str(bin_dir / "yap-studio"),
    }


@pytest.mark.parametrize("installed_version", [None, "0.0.0"])
@pytest.mark.parametrize("folder", ["workspace", "Yap Studio"])
def test_setup_json_offers_engine_install_without_running_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    installed_version: str | None, folder: str,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_engine_step")
    monkeypatch.setattr(module, "_node_available", lambda: True)
    tmp_path = tmp_path / folder
    tmp_path.mkdir()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_synced_service", lambda path, home=None: None)
    monkeypatch.setattr(module, "_free_disk_gb", lambda path: 20.0)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / WHEEL
    wheel.touch()
    (tmp_path / ".yap-studio-version").write_text(VERSION)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    engine = bin_dir / "yap-studio"
    if installed_version is not None:
        engine.touch()
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(bin_dir))
    uv = str(tmp_path / "uv tool")
    monkeypatch.setattr(module.shutil, "which", lambda _name: uv)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, f"yap-studio {installed_version}\n", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(["--json"]) == 1
    assert calls == ([] if installed_version is None else [[str(engine), "--version"]])
    result = json.loads(capsys.readouterr().out)
    requirement = f"yap-studio[asr] @ {wheel.as_uri()}"
    assert result == {
        "ready": False, "refused": None, "steps": [{
            "id": "engine", "does": "Install Yap Studio and its speech engine, then download "
            "the speech model (about 2.5 GB on a Mac with Apple silicon; about 670 MB on an "
            "Intel Mac), then download and install "
            "the renderer and its browser for video graphics.",
            "command": f"{shlex.quote(uv)} tool install --force {shlex.quote(requirement)}"
            f" && {shlex.quote(str(engine))} speech download"
            f" && {shlex.quote(str(engine))} renderer install",
            "page": None, "optional": False,
        }],
    }
    assert shlex.split(result["steps"][0]["command"]) == [
        uv, "tool", "install", "--force", requirement, "&&", str(engine), "speech", "download",
        "&&", str(engine), "renderer", "install",
    ]
    if folder == "Yap Studio":
        assert "Yap%20Studio" in requirement


def test_setup_refuses_synced_folder_before_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_sync")
    home = tmp_path / "home"
    service = module.SYNC_LABELS["db"]
    workspace = home / module.SYNC_FOLDERS["db"] / "Yap Workspace"
    workspace.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", workspace)
    monkeypatch.setattr(module.Path, "home", lambda: home)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail(
        "setup installed before checking the folder"))
    assert module.main() == 1
    output = capsys.readouterr().out
    assert service in output and "move video files" in output
    assert "mv " in output and "Yap Workspace'" in output
    assert "Yap Studio" in output
    assert module.main(["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False and report["steps"] == []
    assert report["refused"]["service"] == service
    assert report["refused"]["move"].startswith("mv ")
    (home / "Yap Studio").mkdir()
    assert module.main(["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["refused"]["move"] is None


@pytest.mark.parametrize(("folder_kind", "service_key"), [
    ("mobile", "ic"),
    ("cloud_od", "od"),
    ("cloud_gd", "gd"),
    ("cloud_box", "box"),
    ("gd", "gd"),
])
def test_setup_synced_service_matches_engine_rules(
    tmp_path: Path, folder_kind: str, service_key: str,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_synced_rules")
    folders = module.SYNC_FOLDERS
    relatives = {
        "mobile": f"{folders['mobile']}/{folders['cloud_docs']}/Project",
        "cloud_od": f"{folders['cloud']}/{folders['od']}-Work/Project",
        "cloud_gd": f"{folders['cloud']}/{folders['gd_cloud']}-Work/Project",
        "cloud_box": f"{folders['cloud']}/Box-Work/Project",
        "gd": f"{folders['gd']}/Project",
    }
    home = tmp_path / "home"
    folder = home / relatives[folder_kind]
    folder.mkdir(parents=True)
    service = "Box" if service_key == "box" else module.SYNC_LABELS[service_key]
    assert module._synced_service(folder, home) == service


def test_setup_sync_marker_skips_home(tmp_path: Path) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_sync_marker")
    home = tmp_path / "home"
    local = home / "Yap Studio"
    local.mkdir(parents=True)
    marker = module.SYNC_FOLDERS["marker"]
    (home / marker).mkdir()
    assert module._synced_service(local, home) is None
    assert module._synced_service(home, home) is None
    (home / marker).rmdir()
    (home / marker).touch()
    assert module._synced_service(local, home) is None
    folder = home / "custom sync" / "Project"
    folder.mkdir(parents=True)
    (folder.parent / marker).touch()
    assert module._synced_service(folder, home) == module.SYNC_LABELS["db"]


@pytest.mark.parametrize("json_output", [False, True])
def test_setup_install_wheel_streams_output_by_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], json_output: bool,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_install_output")
    monkeypatch.setattr(module, "_node_available", lambda: True)
    wheel = tmp_path / "engine.whl"
    engine = tmp_path / "bin" / "yap-studio"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> None:
        calls.append(args)
        assert kwargs["check"] is True
        if args[0] == "uv":
            engine.parent.mkdir(parents=True, exist_ok=True)
            engine.touch()
        message = ("installing" if args[0] == "uv" else
                   "downloading" if args[1] == "speech" else "rendering")
        print(message,
              file=kwargs["stdout"] or sys.stdout)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._install_wheel("uv", wheel, engine, json_output=json_output) is None
    assert calls == [
        ["uv", "tool", "install", "--force", f"yap-studio[asr] @ {wheel.as_uri()}"],
        [str(engine), "speech", "download"],
        [str(engine), "renderer", "install"],
    ]
    captured = capsys.readouterr()
    assert captured.out == ("" if json_output else "installing\ndownloading\nrendering\n")
    assert captured.err == ("installing\ndownloading\nrendering\n" if json_output else "")


@pytest.mark.parametrize(("failure", "call_count"), [
    ("tool", 1), ("speech", 2), ("renderer", 3),
])
def test_setup_install_wheel_stops_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, call_count: int,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_install_failure")
    monkeypatch.setattr(module, "_node_available", lambda: True)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> None:
        calls.append(args)
        assert kwargs["check"] is True
        if failure in args:
            raise subprocess.CalledProcessError(2, args, stderr="failed " + failure)
        (tmp_path / "yap-studio").touch()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._install_wheel("uv", tmp_path / WHEEL, tmp_path / "yap-studio") == (
        "failed " + failure
    )
    assert len(calls) == call_count


@pytest.mark.parametrize("on_path", [True, False])
def test_setup_install_wheel_downloads_through_path_when_uv_bin_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, on_path: bool,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_install_path")
    monkeypatch.setattr(module, "_node_available", lambda: True)
    expected = tmp_path / "bin" / "yap-studio"
    actual = tmp_path / "other-bin" / "yap-studio"
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> None:
        calls.append(args)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.shutil, "which",
                        lambda name: str(actual) if on_path and name == "yap-studio" else None)
    result = module._install_wheel("uv", tmp_path / WHEEL, expected)
    if on_path:
        assert result is None
        assert calls[1:] == [
            [str(actual), "speech", "download"], [str(actual), "renderer", "install"],
        ]
    else:
        assert result is not None and "not in" in result
        assert len(calls) == 1


def test_setup_tool_bin_dir_asks_uv_without_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_tool_bin")
    monkeypatch.delenv("UV_TOOL_BIN_DIR", raising=False)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args == ["uv", "tool", "dir", "--bin"]
        return subprocess.CompletedProcess(args, 0, stdout=f"{tmp_path / 'xdg-bin'}\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._tool_bin_dir("uv") == tmp_path / "xdg-bin"
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tmp_path / "override"))
    assert module._tool_bin_dir("uv") == tmp_path / "override"


@pytest.mark.parametrize(("install_exit", "speech_exit", "renderer_exit"), [
    (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2),
])
def test_setup_engine_command_runs_with_spaces_and_stops_if_install_fails(
    tmp_path: Path, install_exit: int, speech_exit: int, renderer_exit: int,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_command_spaces")
    workspace = tmp_path / "Creator's Yap Studio"
    workspace.mkdir()
    wheel = workspace / "wheels" / WHEEL
    uv = workspace / "uv tool"
    engine = workspace / "yap-studio"
    install_args = workspace / "install args"
    download_args = workspace / "download args"
    renderer_args = workspace / "renderer args"
    # Stubs record the actual shell arguments without installing or downloading anything.
    uv.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$@\" > {shlex.quote(str(install_args))}\n"
        f"exit {install_exit}\n"
    )
    engine.write_text(
        f"#!/bin/sh\nif [ \"$1\" = speech ]; then\n"
        f"  printf '%s\\n' \"$@\" > {shlex.quote(str(download_args))}\n"
        f"  exit {speech_exit}\nfi\n"
        f"printf '%s\\n' \"$@\" > {shlex.quote(str(renderer_args))}\nexit {renderer_exit}\n"
    )
    uv.chmod(0o755)
    engine.chmod(0o755)
    command = module._engine_step(str(uv), wheel, engine)["command"]
    result = subprocess.run(["/bin/sh", "-c", command], check=False)
    assert result.returncode == (install_exit or speech_exit or renderer_exit)
    assert install_args.read_text().splitlines() == [
        "tool", "install", "--force", f"yap-studio[asr] @ {wheel.as_uri()}",
    ]
    if install_exit == 0:
        assert download_args.read_text().splitlines() == ["speech", "download"]
    else:
        assert not download_args.exists()
    if install_exit == 0 and speech_exit == 0:
        assert renderer_args.read_text().splitlines() == ["renderer", "install"]
    else:
        assert not renderer_args.exists()


def test_setup_refuses_low_disk_before_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_disk")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_synced_service", lambda path, home=None: None)
    monkeypatch.setattr(module, "_free_disk_gb", lambda path: 9.5)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail(
        "setup installed before checking disk"))
    assert module.main() == 1
    assert "9.5 GB free" in capsys.readouterr().out
    assert module.main(["--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["refused"]["reason"] == "disk"
    assert report["steps"] == []


def test_setup_json_steps_include_model_and_optional_cutout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_json")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_synced_service", lambda path, home=None: None)
    monkeypatch.setattr(module, "_free_disk_gb", lambda path: 20.0)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / WHEEL).touch()
    (tmp_path / ".yap-studio-version").write_text(VERSION)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "yap-studio").touch()
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(bin_dir))
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/uv")
    report = {
        "install": {"version": VERSION}, "warnings": [], "problems": [],
        "ffmpeg": {"available": False}, "ffprobe": {"available": False},
        "filters": {"ebur128": False}, "node": {"available": False},
        "renderer": {"available": False, "browser": False},
        "speech": {"available": True, "model": False},
        "cutout": {"available": False, "reason": "Xcode tools missing"},
        "edit_missing": ["node", "renderer", "browser"], "ready_to_edit": False,
    }

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        return subprocess.CompletedProcess(args, 2, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(["--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert set(output) == {"ready", "refused", "steps", "engine"}
    assert output["ready"] is False and output["refused"] is None
    assert [step["id"] for step in output["steps"]] == [
        "ffmpeg", "node", "renderer", "speech_model", "cutout",
    ]
    assert all(set(step) == {"id", "does", "command", "page", "optional"}
               for step in output["steps"])
    assert all(step["page"] is None for step in output["steps"])
    assert output["steps"][-1]["optional"] is True
    assert output["steps"][-1]["command"] == "xcode-select --install"
    assert output["steps"][-2]["command"].endswith("speech download")


@pytest.mark.parametrize(("phone_report", "does"), [
    ({"tailscale": "missing", "name": None, "via": "wifi"},
     "Optional: set up Tailscale so review links open on your phone anywhere, "
     "not only on this Wi-Fi."),
    ({"tailscale": "installed", "name": None, "via": "wifi"},
     "Optional: open Tailscale and sign in so review links open on your phone anywhere, "
     "not only on this Wi-Fi."),
    ({"tailscale": "connected", "name": "box.tailnet.example.invalid", "via": "tailscale"}, None),
    ({"tailscale": "missing", "name": None, "via": None},
     "Optional: set up Tailscale so review links open on your phone anywhere, "
     "not only on this Wi-Fi."),
    (None, None),
])
def test_setup_offers_optional_tailscale_after_other_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    phone_report: dict | None, does: str | None,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_phone")
    monkeypatch.setattr(module, "_open_home", lambda engine, phone: FAKE_HOME)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_synced_service", lambda path, home=None: None)
    monkeypatch.setattr(module, "_free_disk_gb", lambda path: 20.0)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / WHEEL).touch()
    (tmp_path / ".yap-studio-version").write_text(VERSION)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "yap-studio").touch()
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(bin_dir))
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/uv")
    report = {
        "install": {"version": VERSION}, "warnings": [], "problems": [],
        "ffmpeg": {"available": True}, "ffprobe": {"available": True},
        "filters": {"ebur128": True}, "node": {"available": True},
        "renderer": {"available": True, "browser": True},
        "speech": {"available": True}, "cutout": {"available": False},
        "edit_missing": [], "ready_to_edit": True,
    }
    if phone_report is not None:
        report["phone"] = phone_report

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        return subprocess.CompletedProcess(args, 0, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(["--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is True
    assert result["phone"] == {
        "via": phone_report["via"] if phone_report else None,
        "tailscale": phone_report["tailscale"] if phone_report else "missing",
    }
    steps = result["steps"]
    assert [step["id"] for step in steps] == (["cutout", "tailscale"] if does else ["cutout"])
    if does:
        assert steps[-1] == {
            "id": "tailscale", "does": does, "command": None,
            "page": "https://tailscale.com/download", "optional": True,
        }
    assert module.main() == 0
    output = capsys.readouterr().out
    if does:
        assert f"Step 2 of 2 (optional): {does}" in output
        assert "  Open https://tailscale.com/download" in output
    if phone_report is None or phone_report["via"] is None:
        assert "Your phone needs the same Wi-Fi or Tailscale to open review links." in output
        assert "Review links reach your phone on the same Wi-Fi." not in output
    elif phone_report["tailscale"] == "connected":
        assert "Review links reach your phone anywhere through Tailscale." in output
    else:
        assert "Review links reach your phone on the same Wi-Fi." in output
    assert "Ready to edit" in output


def test_doctor_json_reports_workspace_version_mismatch_only_inside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: None)
    monkeypatch.setattr(phone, "tailscale_name", lambda: None)
    monkeypatch.setattr(phone, "lan_address", lambda: "192.168.1.10")
    monkeypatch.setattr(setup_checks, "synced_service", lambda path: None)
    monkeypatch.setattr(setup_checks, "free_disk_gb", lambda path: 20.0)
    monkeypatch.setattr(setup_checks, "cutout_reason", lambda: None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".yap-studio-version").write_text("9.9.9\n")
    skill = tmp_path / ".claude/skills/edit-video/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("edit\n")
    cli.main(["doctor", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert "build" in report["install"]
    assert report["workspace"] == {"version": "9.9.9", "matches": False}
    assert any("differs" in warning for warning in report["warnings"])
    (tmp_path / ".yap-studio-version").write_text(VERSION + "\n")
    cli.main(["doctor", "--json"])
    matched = json.loads(capsys.readouterr().out)
    assert matched["workspace"] == {"version": VERSION, "matches": True}
    monkeypatch.chdir(tmp_path.parent)
    cli.main(["doctor", "--json"])
    outside = json.loads(capsys.readouterr().out)
    assert outside["workspace"] is None
    assert not any("workspace version" in warning for warning in outside["warnings"])


def _skill_commands() -> list[str]:
    skill = (ROOT / "workspace/.claude/skills/edit-video/SKILL.md").read_text()
    return re.findall(r"`(yap-studio [^`]+)`", " ".join(skill.split()))


def test_skill_rechecks_the_whole_story_after_notes() -> None:
    skill = " ".join((ROOT / "workspace/.claude/skills/edit-video/SKILL.md").read_text().split())
    revision = skill.split("4. Review the cut.", 1)[1].split("5. Build the picture.", 1)[0]
    assert revision.index("yap-studio feedback apply") < revision.index(
        "yap-studio edit check DIR") < revision.index("yap-studio cut DIR")
    for instruction in ("reread the whole kept script", "`Repeated`", "`Same point twice`",
                        "repeat kept on purpose", "version note", '"remove the repeats"',
                        "whole video"):
        assert instruction in revision


def test_workspace_setup_asks_before_each_json_step() -> None:
    guide = (ROOT / "workspace/AGENTS.md").read_text()
    assert "python3 setup.py --agent AGENT --json" in guide
    assert "For each step" in guide
    assert "Ask before running its `command`" in guide
    assert "a `page` and no `command`" in guide
    assert "wait until they say it is done" in guide
    assert "Tailscale is optional" in guide
    assert "Wi-Fi link works without it on the same network" in guide


def test_skill_transitions_follow_zooms_before_timeline_check() -> None:
    skill = " ".join((ROOT / "workspace/.claude/skills/edit-video/SKILL.md").read_text().split())
    start = skill.index("- Transitions.")
    assert skill.index("- Zooms.") < start < skill.index("yap-studio timeline check DIR")
    bullet = skill[start:].split(" - ", 1)[0]
    assert "`moves.transitions`" in bullet
    assert "`yap-studio/timeline@1`" in bullet
    assert "`kind` `transition`" in bullet
    assert "`yap-studio timeline add FRAGMENT DIR --lane agent`" in bullet


def test_skill_saves_a_version_before_every_versions_review() -> None:
    commands = _skill_commands()
    saved = False
    reviews = []
    for command in commands:
        if command.startswith("yap-studio cut "):
            saved = False
        if command.startswith(("yap-studio versions save DIR", "yap-studio run DIR")):
            saved = True
        if command.startswith("yap-studio review --versions"):
            reviews.append(command)
            assert saved, f"{command} runs before versions save"
    # Every review goes through saved versions so comments carry across them.
    assert set(reviews) == {"yap-studio review --versions DIR --phone --json --detach"}


def test_skill_saves_the_final_as_a_version_before_the_final_review() -> None:
    commands = _skill_commands()
    final = commands.index('yap-studio run DIR --note "Final" --json --detach')
    final_review = commands.index("yap-studio review --versions DIR --phone --json --detach", final)
    assert final < final_review
    assert not any(command.startswith("yap-studio review DIR/final") for command in commands)
    # Run saves the final with layer windows for card and caption keeps.
    assert any(command.startswith("yap-studio feedback check DIR --version vM")
               for command in commands[final:])


def test_skill_sends_a_phone_review_link() -> None:
    skill = (ROOT / "workspace/.claude/skills/edit-video/SKILL.md").read_text()
    assert "yap-studio review --versions DIR --phone --json --detach" in _skill_commands()
    assert "phone.url" in skill
    assert "yap-studio review --stop DIR" in skill
    assert "run the command in the background" not in skill
    assert "yap-studio status DIR --json --wait 540" in skill
    assert "30 minutes" in skill
    assert 'yap-studio run DIR --note "Notes from bN" --json --detach' in skill


def test_skill_replies_name_the_saved_version_and_music_comes_before_graphics() -> None:
    commands = _skill_commands()
    replies = [command for command in commands if command.startswith("yap-studio feedback reply")]
    # A batch has no resulting version until a reply names one.
    assert replies and all("--version vM" in command for command in replies)
    # Music is prepared before the run that renders graphics.
    assert commands.index("yap-studio music DIR") < commands.index(
        'yap-studio run DIR --note "Final" --json --detach')
    assert commands.index("yap-studio music DIR") < commands.index("yap-studio timeline check DIR")


def test_edit_command_names_project_after_transcript_and_runs_remaining_stages() -> None:
    command = (ROOT / "workspace/.claude/commands/edit.md").read_text()
    skill = (ROOT / "workspace/.claude/skills/edit-video/SKILL.md").read_text()
    assert "python3 new_project.py" in command
    assert "$ARGUMENTS" in command and "take names" in command
    assert "python3 new_project.py --name DIR TOPIC" in skill
    assert skill.index("yap-studio words DIR") < skill.index(
        "python3 new_project.py --name DIR TOPIC")
    assert 'yap-studio run DIR --note "Notes from bN" --json' in skill
    assert "--allow-violations" in skill
    assert "DIR/plans/layers.json" not in skill
    assert "yap-studio versions save DIR --media" not in skill


@pytest.mark.parametrize(("platform", "apt", "expected", "absent"), [
    ("darwin", False, ("brew install uv", "brew install ffmpeg", "brew install node@22"),
     "apt"),
    ("linux", True, ("astral.sh/uv/install.sh", "sudo apt install ffmpeg",
                     "deb.nodesource.com/setup_22.x | sudo -E bash - "
                     "&& sudo apt-get install -y nodejs"), "brew"),
    ("linux", False, ("astral.sh/uv/install.sh", "Open https://ffmpeg.org/",
                      "Open https://nodejs.org/"), "apt"),
    ("win32", False, ("Open https://docs.astral.sh/uv/",
                      "Open https://ffmpeg.org/", "Open https://nodejs.org/"), "brew"),
])
def test_setup_install_commands_match_the_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    platform: str, apt: bool, expected: tuple[str, str, str], absent: str,
) -> None:
    module = _module(ROOT / "workspace/setup.py", f"workspace_setup_{platform}_{apt}")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_synced_service", lambda path, home=None: None)
    monkeypatch.setattr(module, "_free_disk_gb", lambda path: 20.0)
    monkeypatch.setattr(module.sys, "platform", platform)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / WHEEL).write_bytes(b"synthetic wheel")
    (tmp_path / ".yap-studio-version").write_text(VERSION + "\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "yap-studio").touch()
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(bin_dir))
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    found = {"apt-get": "/usr/bin/apt-get"} if apt else {}
    if platform == "darwin":
        found["brew"] = "/opt/homebrew/bin/brew"
    monkeypatch.setattr(module.shutil, "which", found.get)
    assert module.main() == 1
    uv_output = capsys.readouterr().err
    assert expected[0] in uv_output and absent not in uv_output
    if platform == "win32":
        assert module.main(["--json"]) == 1
        uv_step = json.loads(capsys.readouterr().out)["steps"][0]
        assert uv_step["command"] is None
        assert uv_step["page"] == module.OTHER_HINTS["uv"]
    found["uv"] = "/usr/bin/uv"
    found["yap-studio"] = str(bin_dir / "yap-studio")
    report = {
        "install": {"version": VERSION}, "warnings": [], "problems": ["ffmpeg missing"],
        "ffmpeg": {"available": False, "path": None}, "ffprobe": {"available": False},
        "filters": {"ebur128": False}, "node": {"available": False},
        "renderer": {"available": True, "browser": True}, "speech": {"available": True},
        "edit_missing": ["node"], "ready_to_edit": False,
    }

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        return subprocess.CompletedProcess(args, 2, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main() == 1
    output = capsys.readouterr().out
    assert expected[1] in output and expected[2] in output
    assert absent not in output
    assert module.main(["--json"]) == 1
    steps = json.loads(capsys.readouterr().out)["steps"]
    if platform == "win32" or (platform == "linux" and not apt):
        assert [(step["command"], step["page"]) for step in steps] == [
            (None, module.OTHER_HINTS["ffmpeg"]),
            (None, module.OTHER_HINTS["node"]),
        ]
    else:
        assert all(step["command"] and step["page"] is None for step in steps)
    # Every hint is a runnable command or names its install page.
    for line in output.splitlines():
        if line.startswith("  "):
            assert not line.strip().startswith("see ")
    # An installed tool that fails or lacks filters is too old for the package command.
    for ffmpeg, ffprobe in (
        ({"available": False, "path": "/usr/bin/ffmpeg"}, {"available": False}),
        ({"available": True, "path": "/usr/bin/ffmpeg"},
         {"available": False, "path": "/usr/bin/ffprobe"}),
        ({"available": True, "path": "/usr/bin/ffmpeg"},
         {"available": True, "path": "/usr/bin/ffprobe"}),
    ):
        report.update(ffmpeg=ffmpeg, ffprobe=ffprobe)
        assert module.main() == 1
        output = capsys.readouterr().out
        assert "  Open https://ffmpeg.org/download.html" in output
        assert "apt install ffmpeg" not in output and "brew install ffmpeg" not in output
        assert module.main(["--json"]) == 1
        ffmpeg_step = json.loads(capsys.readouterr().out)["steps"][0]
        assert ffmpeg_step["id"] == "ffmpeg"
        assert ffmpeg_step["command"] is None
        assert ffmpeg_step["page"] == module.FFMPEG_PAGE


def test_setup_reinstalls_a_same_version_engine_without_doctor_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_stale")
    monkeypatch.setattr(module, "_open_home", lambda engine, phone: FAKE_HOME)
    monkeypatch.setattr(module, "_node_available", lambda: True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_synced_service", lambda path, home=None: None)
    monkeypatch.setattr(module, "_free_disk_gb", lambda path: 20.0)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / WHEEL).write_bytes(b"synthetic wheel")
    (tmp_path / ".yap-studio-version").write_text(VERSION + "\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    engine = str(bin_dir / "yap-studio")
    (bin_dir / "yap-studio").touch()
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(bin_dir))
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/uv")
    report = {
        "install": {"version": VERSION}, "warnings": [], "problems": [],
        "ffmpeg": {"available": True}, "ffprobe": {"available": True},
        "filters": {"ebur128": True}, "node": {"available": True},
        "renderer": {"available": True, "browser": True}, "speech": {"available": True},
        "edit_missing": [], "ready_to_edit": True,
    }
    calls: list[list[str]] = []
    fresh = [False]

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:4] == ["tool", "install", "--force"]:
            fresh[0] = True
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[1:] == ["speech", "download"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if not fresh[0]:
            # An older build with the same version has no doctor --edit.
            return subprocess.CompletedProcess(
                args, 2, "", "yap-studio: error: unrecognized arguments: --edit\n")
        return subprocess.CompletedProcess(args, 0, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(["--json"]) == 1
    assert calls == [[engine, "--version"], [engine, "doctor", "--edit", "--json"]]
    stale = json.loads(capsys.readouterr().out)
    assert stale == {
        "ready": False, "refused": None, "steps": [{
            "id": "engine", "does": "Install Yap Studio and its speech engine, then download "
            "the speech model (about 2.5 GB on a Mac with Apple silicon; about 670 MB on an "
            "Intel Mac), then download and install "
            "the renderer and its browser for video graphics.",
            "command": "/usr/bin/uv tool install --force "
            + shlex.quote(f"yap-studio[asr] @ {(wheels / WHEEL).as_uri()}")
            + f" && {shlex.quote(engine)} speech download"
            + f" && {shlex.quote(engine)} renderer install",
            "page": None, "optional": False,
        }],
    }
    calls.clear()
    assert module.main() == 0
    assert calls == [
        [engine, "--version"],
        [engine, "doctor", "--edit", "--json"],
        ["/usr/bin/uv", "tool", "install", "--force",
         f"yap-studio[asr] @ {(wheels / WHEEL).as_uri()}"],
        [engine, "speech", "download"],
        [engine, "renderer", "install"],
        [engine, "doctor", "--edit", "--json"],
    ]
    assert "Ready to edit" in capsys.readouterr().out
    # A fresh install that still cannot report stops with the doctor output.
    fresh[0] = False
    calls.clear()
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: (
        calls.append(args) or (
            subprocess.CompletedProcess(args, 0, "yap-studio 0.0.0\n", "")
            if args[1:] == ["--version"] else subprocess.CompletedProcess(args, 2, "", "broken\n")
        )))
    assert module.main() == 1
    assert "broken" in capsys.readouterr().err
    assert sum(call[1:2] == ["doctor"] for call in calls) == 1


def _build_setup_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mock_home: bool = True,
) -> tuple[ModuleType, Path, str, dict]:
    monkeypatch.setenv("PATH", os.environ.get("PATH", os.defpath))
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_build")
    monkeypatch.setattr(module, "_node_available", lambda: True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_synced_service", lambda path, home=None: None)
    monkeypatch.setattr(module, "_free_disk_gb", lambda path: 20.0)
    if mock_home:
        monkeypatch.setattr(module, "_open_home", lambda engine, phone: FAKE_HOME)
    wheel = _synthetic_wheel(tmp_path)
    (tmp_path / ".yap-studio-version").write_text(VERSION + "\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "yap-studio").touch()
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(bin_dir))
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/uv")
    report = {
        "install": {"version": VERSION}, "warnings": [], "problems": [],
        "ffmpeg": {"available": True}, "ffprobe": {"available": True},
        "filters": {"ebur128": True}, "node": {"available": True},
        "renderer": {"available": True, "browser": True}, "speech": {"available": True},
        "edit_missing": [], "ready_to_edit": True,
    }
    return module, wheel, str(bin_dir / "yap-studio"), report


def test_setup_confirms_home_state_before_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch, mock_home=False)
    report["install"]["build"] = module._wheel_build(wheel)
    report["phone"] = {"via": "wifi", "tailscale": "missing"}
    saved = {
        "url": "http://127.0.0.1:8765/?t=synthetic-private-token",
        "phone": {"via": "wifi", "url": "http://192.0.2.1:8765/?t=synthetic-private-token"},
        "detached": {"pid": 123, "stop": "yap-studio home --stop"},
    }
    calls = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:] == ["doctor", "--edit", "--json"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(report), "")
        assert args == [engine, "home", "--workspace", str(tmp_path), "--phone",
                        "--no-open", "--detach", "--json"]
        calls.append("launch")
        return subprocess.CompletedProcess(args, 0, json.dumps(saved), "")

    class Response(io.BytesIO):
        status = 200

    def fake_get(url: str, *, timeout: int) -> Response:
        assert url == "http://127.0.0.1:8765/api/state?t=synthetic-private-token"
        assert timeout == 5
        calls.append("state GET")
        return Response(json.dumps({"schema": "yap-studio/home@1",
                                    "workspace": tmp_path.name}).encode())

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "urlopen", fake_get, raising=False)
    assert module.main() == 0
    output = capsys.readouterr().out
    assert calls == ["launch", "state GET"]
    assert "Setup complete" in output
    assert f"Yap Studio home: {saved['url']}" in output
    assert f"Open on your phone: {saved['phone']['url']}" in output
    assert module.main(["--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is True
    assert result["home"] == {"url": saved["url"], "phone": saved["phone"]}


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("error_output", [
    json.dumps({"error": {"message": "No network your phone can reach."}}),
    "No network your phone can reach.",
    "",
])
@pytest.mark.parametrize("fallback_ready", [False, True])
def test_setup_home_retries_without_phone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    json_output: bool, error_output: str, fallback_ready: bool,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch, mock_home=False)
    report["install"]["build"] = module._wheel_build(wheel)
    report["phone"] = {"via": "tailscale", "tailscale": "connected"}
    calls = []
    reason = "No network your phone can reach." if error_output else "Home exited with status 2."

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:] == ["doctor", "--edit", "--json"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(report), "")
        calls.append(args)
        if "--phone" in args:
            return subprocess.CompletedProcess(args, 2, "", error_output)
        return subprocess.CompletedProcess(args, 0 if fallback_ready else 2,
                                           json.dumps(FAKE_HOME), "")

    class Response(io.BytesIO):
        status = 200

    def fake_get(url: str, *, timeout: int) -> Response:
        assert fallback_ready
        assert url == "http://127.0.0.1:9000/api/state?t=synthetic"
        assert timeout == 5
        calls.append("state GET")
        return Response(json.dumps({"schema": "yap-studio/home@1",
                                    "workspace": tmp_path.name}).encode())

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "urlopen", fake_get, raising=False)
    assert module.main(["--json"] if json_output else []) == (0 if fallback_ready else 1)
    assert calls == [
        [engine, "home", "--workspace", str(tmp_path), "--phone",
         "--no-open", "--detach", "--json"],
        [engine, "home", "--workspace", str(tmp_path), "--no-open", "--detach", "--json"],
        *(["state GET"] if fallback_ready else []),
    ]
    output = capsys.readouterr()
    if json_output:
        result = json.loads(output.out)
        assert result["ready"] is fallback_ready
        if fallback_ready:
            assert result["home"] == {**FAKE_HOME, "phone_error": reason}
        else:
            assert result["refused"]["reason"] == "home"
    elif fallback_ready:
        assert "Setup complete" in output.out
        assert f"Yap Studio home: {FAKE_HOME['url']}" in output.out
        assert f"Phone link unavailable: {reason}" in output.out
        assert "Review links reach your phone" not in output.out
    else:
        assert "Setup complete" not in output.out
        assert "Yap Studio home did not answer" in output.err


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("launch", ["local", "phone", "fallback"])
def test_setup_home_timeout_stops_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    json_output: bool, launch: str,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch, mock_home=False)
    report["install"]["build"] = module._wheel_build(wheel)
    if launch != "local":
        report["phone"] = {"via": "wifi", "tailscale": "missing"}
    calls = []
    launch_timeouts = []
    running = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:] == ["doctor", "--edit", "--json"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(report), "")
        calls.append(args)
        if "--stop" in args:
            assert args == [engine, "home", "--stop", "--workspace", str(tmp_path), "--json"]
            running.clear()
            return subprocess.CompletedProcess(args, 0, '{"stopped": []}', "")
        launch_timeouts.append(kwargs["timeout"])
        if launch == "fallback" and "--phone" in args:
            return subprocess.CompletedProcess(args, 2, "", "No local network.")
        running.append("synthetic detached home")
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    def unexpected_get(*args: object, **kwargs: object) -> None:
        pytest.fail("A timed out launch must not check home state.")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "urlopen", unexpected_get, raising=False)
    assert module.main(["--json"] if json_output else []) == 1
    assert running == []
    assert "--stop" in calls[-1]
    assert len(calls) == (3 if launch == "fallback" else 2)
    assert all(timeout >= 120 for timeout in launch_timeouts)
    output = capsys.readouterr()
    if json_output:
        result = json.loads(output.out)
        assert result["ready"] is False
        assert result["refused"]["reason"] == "home"
    else:
        assert "Setup complete" not in output.out
        assert "Ready to edit" not in output.out
        assert "Yap Studio home did not answer" in output.err


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("answers", [False, True])
@pytest.mark.parametrize("failure", ["connection", "open timeout", "read timeout"])
def test_setup_home_waits_for_first_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    json_output: bool, answers: bool, failure: str,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch, mock_home=False)
    report["install"]["build"] = module._wheel_build(wheel)
    elapsed = [0.0]
    timeouts = []

    def advance(seconds: float) -> None:
        elapsed[0] += seconds

    monkeypatch.setattr(module, "HOME_STATE_TIMEOUT", 3, raising=False)
    monkeypatch.setattr(module, "time", SimpleNamespace(
        monotonic=lambda: elapsed[0], sleep=advance,
    ), raising=False)

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:] == ["doctor", "--edit", "--json"]:
            return subprocess.CompletedProcess(args, 0, json.dumps(report), "")
        assert args == [engine, "home", "--workspace", str(tmp_path),
                        "--no-open", "--detach", "--json"]
        return subprocess.CompletedProcess(
            args, 0, json.dumps({"url": "http://127.0.0.1:8765/?t=synthetic"}), "",
        )

    class Response(io.BytesIO):
        status = 200

    class SlowResponse(Response):
        def read(self, *args: object) -> bytes:
            advance(1)
            raise TimeoutError("snapshot still reading the inbox")

    def first_snapshot(url: str, *, timeout: float) -> Response:
        assert url == "http://127.0.0.1:8765/api/state?t=synthetic"
        assert 0 < timeout <= 3 - elapsed[0]
        timeouts.append(timeout)
        if answers and len(timeouts) == 2:
            advance(0.5)
            return Response(json.dumps({"schema": "yap-studio/home@1",
                                        "workspace": tmp_path.name}).encode())
        if failure == "read timeout":
            return SlowResponse()
        advance(1)
        if failure == "connection":
            raise URLError(ConnectionRefusedError("connection refused"))
        raise TimeoutError("snapshot still reading the inbox")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module, "urlopen", first_snapshot, raising=False)
    assert module.main(["--json"] if json_output else []) == (0 if answers else 1)
    assert timeouts == [3, 1]
    assert elapsed[0] == (2.5 if answers else 3)
    output = capsys.readouterr()
    assert output.err.count("Waiting for Yap Studio home to read your inbox...") == 1
    if json_output:
        result = json.loads(output.out)
        assert result["ready"] is answers
        if answers:
            assert result["home"]["url"] == "http://127.0.0.1:8765/?t=synthetic"
        else:
            assert result["refused"]["reason"] == "home"
            assert "run setup again" in result["refused"]["fix"]
    elif answers:
        assert "Setup complete" in output.out
        assert "Yap Studio home: http://127.0.0.1:8765/?t=synthetic" in output.out
    else:
        assert "Setup complete" not in output.out
        assert "Ready to edit" not in output.out
        assert "Yap Studio home did not answer" in output.err
        assert "run setup again" in output.err


@pytest.mark.parametrize(("platform", "on_path", "missing"), [
    ("darwin", False, False), ("darwin", False, True),
    ("darwin", True, False), ("linux", False, False),
])
def test_setup_homebrew_path_finds_installed_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    platform: str, on_path: bool, missing: bool,
) -> None:
    which = shutil.which
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    folder = tmp_path / "homebrew/bin"
    folder.mkdir(parents=True)
    for name in ("brew", "uv", "ffmpeg", "ffprobe", "yap-studio"):
        if missing and name == "ffmpeg":
            continue
        binary = folder / name
        binary.touch()
        binary.chmod(0o755)
    original = str(tmp_path / "preferred")
    if on_path:
        original += os.pathsep + str(folder)
    monkeypatch.setenv("PATH", original)
    monkeypatch.setattr(module.sys, "platform", platform)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(module.shutil, "which", which)
    module.HOMEBREW_BIN_DIRS = (str(folder),)
    report["install"]["build"] = module._wheel_build(wheel)

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        assert args[1:] == ["doctor", "--edit", "--json"]
        assert module._uv_binary() == str(folder / "uv")
        for name in ("ffmpeg", "ffprobe"):
            report[name]["available"] = which(name) is not None
        report["ready_to_edit"] = not missing
        return subprocess.CompletedProcess(args, int(missing), json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", run)
    expected_ready = platform == "darwin" and not missing
    assert module.main(["--json"]) == (0 if expected_ready else 1)
    steps = json.loads(capsys.readouterr().out)["steps"]
    expected_path = original
    if platform == "darwin" and not on_path:
        expected_path += os.pathsep + str(folder)
    assert os.environ["PATH"] == expected_path
    if expected_ready:
        assert steps == []
    elif missing:
        assert steps[0]["id"] == "ffmpeg"
        assert steps[0]["command"] == f"{folder}/brew install ffmpeg"


@pytest.mark.parametrize(("brew", "machine"), [
    ("PATH", "arm64"), ("/opt/homebrew/bin/brew", "arm64"),
    ("/usr/local/bin/brew", "x86_64"), (None, "arm64"), (None, "x86_64"),
])
def test_setup_mac_finds_homebrew_or_offers_its_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    brew: str | None, machine: str,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr("platform.machine", lambda: machine)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    folder = tmp_path / "homebrew/bin"
    module.HOMEBREW_BIN_DIRS = (str(folder),)
    if brew not in (None, "PATH"):
        folder.mkdir(parents=True)
        binary = folder / "brew"
        binary.touch()
        binary.chmod(0o755)
        brew = str(binary)
    found = {"uv": "/usr/bin/uv", "yap-studio": engine}
    if brew == "PATH":
        found["brew"] = "/opt/homebrew/bin/brew"
    monkeypatch.setattr(module.shutil, "which", found.get)
    report["install"]["build"] = module._wheel_build(wheel)
    report.update(ffmpeg={"available": False}, ffprobe={"available": False},
                  node={"available": False}, ready_to_edit=False)
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: (
        subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:] == ["--version"] else
        subprocess.CompletedProcess(args, 1, json.dumps(report), "")))
    assert module.main(["--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    steps = result["steps"]
    if brew is None:
        assert [step["id"] for step in steps] == ["homebrew", "ffmpeg", "node"]
        assert steps[0] == {
            "id": "homebrew", "does": "Homebrew installs ffmpeg and Node. Open the Terminal "
            "app, paste the command from brew.sh, type your Mac password when it asks, "
            "then say done.", "command": None, "page": "https://brew.sh", "optional": False,
        }
        steps = steps[1:]
        prefix = "/opt/homebrew/bin/brew" if machine == "arm64" else "/usr/local/bin/brew"
    else:
        prefix = "brew" if brew == "PATH" else brew
    assert [step["command"] for step in steps] == [
        f"{prefix} install ffmpeg",
        f"{prefix} install node@22",
    ]
    found.pop("uv")
    assert module.main(["--json"]) == 1
    steps = json.loads(capsys.readouterr().out)["steps"]
    assert steps[-1]["id"] == "uv"
    assert steps[-1]["command"] == (module.UV_INSTALLER if brew is None else
                                    f"{'brew' if brew == 'PATH' else brew} install uv")
    if brew is None:
        assert steps[0]["id"] == "homebrew"


@pytest.mark.parametrize("missing", [None, "node", "ffmpeg", "ffprobe"])
def test_setup_missing_uv_only_offers_homebrew_for_missing_brew_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    missing: str | None,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    found = {name: f"/usr/bin/{name}" for name in ("ffmpeg", "ffprobe", "node", "npm")
             if name != missing}
    monkeypatch.setattr(module.shutil, "which", found.get)
    monkeypatch.setattr(module, "_node_available", lambda: missing != "node")
    is_file = Path.is_file
    monkeypatch.setattr(module.Path, "is_file", lambda path: (
        False if str(path) in ("/opt/homebrew/bin/brew", "/usr/local/bin/brew")
        else is_file(path)))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail(
        "setup must not install anything before uv is available"))
    assert module.main(["--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert [step["id"] for step in result["steps"]] == (
        ["homebrew", "uv"] if missing else ["uv"])
    assert result["steps"][-1]["command"] == module.UV_INSTALLER


@pytest.mark.parametrize("location", [".local/bin", ".cargo/bin", "xdg-bin", "xdg-data/../bin"])
def test_setup_finds_uv_outside_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    location: str,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setenv("XDG_BIN_HOME", str(tmp_path / "xdg-bin"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    (tmp_path / "xdg-data").mkdir()
    uv = tmp_path / location / "uv"
    uv.parent.mkdir(parents=True, exist_ok=True)
    uv.touch(mode=0o755)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    report["install"]["build"] = module._wheel_build(wheel)
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs: (
        subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:] == ["--version"] else
        subprocess.CompletedProcess(args, 0, json.dumps(report), "")))
    assert module.main(["--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is False
    assert result["engine"] == engine
    assert result["steps"] == [{
        "id": "engine_path",
        "does": "Lets your agent find the yap-studio command in new sessions.",
        "command": shlex.join([str(uv.absolute()), "tool", "update-shell"]),
        "page": None, "optional": False,
    }]


@pytest.mark.parametrize("take_count", [0, 1, 2, 5])
@pytest.mark.parametrize("inbox_exists", [False, True])
def test_setup_ready_reports_inbox_takes_in_text_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    take_count: int, inbox_exists: bool,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    report["install"]["build"] = module._wheel_build(wheel)
    inbox = tmp_path / "inbox"
    if inbox_exists:
        inbox.mkdir()
        suffixes = (".MOV", ".m4a", ".m4v", ".mkv", ".webm")
        for index, suffix in enumerate(suffixes[:take_count]):
            (inbox / f"take-{index}{suffix}").write_bytes(b"take")
        (inbox / "README.md").write_text("Put takes here.")
        (inbox / ".hidden.mov").write_bytes(b"hidden")
        (inbox / "subfolder.mp4").mkdir()
        (inbox / "subfolder.mp4" / "nested.mov").write_bytes(b"nested")
        (inbox / "link.mov").symlink_to(inbox / ".hidden.mov")

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        assert args == [engine, "doctor", "--edit", "--json"]
        return subprocess.CompletedProcess(args, 0, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    expected = take_count if inbox_exists else 0
    assert module.main() == 0
    output = capsys.readouterr().out
    noun = "take" if expected == 1 else "takes"
    assert f"Ready to edit. {expected} {noun} waiting in inbox/." in output
    if expected:
        assert "Say 'edit this video' to start." in output
        assert "Put a take in inbox/" not in output
    else:
        assert "Put a take in inbox/" in output
    assert module.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "ready": True, "refused": None, "steps": [], "inbox_takes": expected,
        "phone": {"via": None, "tailscale": "missing"}, "home": FAKE_HOME, "engine": engine,
    }


@pytest.mark.parametrize(("available", "browser", "ready"), [
    (False, False, False), (True, False, False), (True, True, True),
])
def test_setup_renderer_step_only_repairs_an_installed_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    available: bool, browser: bool, ready: bool,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    report["install"]["build"] = module._wheel_build(wheel)
    report["renderer"] = {"available": available, "browser": browser}
    report["ready_to_edit"] = ready

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        assert args == [engine, "doctor", "--edit", "--json"]
        return subprocess.CompletedProcess(args, 0 if ready else 2, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(["--json"]) == (0 if ready else 1)
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is ready
    assert [step["id"] for step in result["steps"]] == ([] if ready else ["renderer"])
    if not ready:
        assert shlex.split(result["steps"][0]["command"]) == [engine, "renderer", "install"]


@pytest.mark.parametrize(("speech", "step_id"), [
    ({"available": False, "model": False}, "speech"),
    ({"available": True, "model": False}, "speech_model"),
    ({"available": True, "model": True}, None),
])
def test_setup_speech_steps_only_repair_an_installed_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    speech: dict[str, bool], step_id: str | None,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    report["install"]["build"] = module._wheel_build(wheel)
    report["speech"] = speech
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        assert args[1:] == ["doctor", "--edit", "--json"]
        return subprocess.CompletedProcess(args, 0, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(["--json"]) == (0 if step_id is None else 1)
    result = json.loads(capsys.readouterr().out)
    assert result["refused"] is None
    assert [step["id"] for step in result["steps"]] == ([] if step_id is None else [step_id])
    assert calls == [[engine, "--version"], [engine, "doctor", "--edit", "--json"]]
    if step_id == "speech":
        assert shlex.split(result["steps"][0]["command"]) == [
            "/usr/bin/uv", "tool", "install", "--force", f"yap-studio[asr] @ {wheel.as_uri()}",
            "&&", engine, "speech", "download",
        ]
    elif step_id == "speech_model":
        assert shlex.split(result["steps"][0]["command"]) == [engine, "speech", "download"]


def test_setup_fresh_non_json_install_is_ready_to_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    Path(engine).unlink()
    report["install"]["build"] = module._wheel_build(wheel)
    report["speech"] = {"available": False, "model": False}
    report["renderer"] = {"available": False, "browser": False}
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1:4] == ["tool", "install", "--force"]:
            Path(engine).touch()
            report["speech"]["available"] = True
        elif args[1:] == ["speech", "download"]:
            assert report["speech"]["available"] is True
            report["speech"]["model"] = True
        elif args[1:] == ["renderer", "install"]:
            assert report["speech"]["model"] is True
            report["renderer"] = {"available": True, "browser": True}
        else:
            assert args[1:] == ["doctor", "--edit", "--json"]
            assert report["renderer"] == {"available": True, "browser": True}
            return subprocess.CompletedProcess(args, 0, json.dumps(report), "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main() == 0
    assert calls == [
        ["/usr/bin/uv", "tool", "install", "--force", f"yap-studio[asr] @ {wheel.as_uri()}"],
        [engine, "speech", "download"],
        [engine, "renderer", "install"],
        [engine, "doctor", "--edit", "--json"],
    ]
    output = capsys.readouterr().out
    assert "Ready to edit" in output
    assert "Step " not in output


def test_setup_keeps_a_same_version_matching_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    report["install"]["build"] = module._wheel_build(wheel)
    assert report["install"]["build"] is not None
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        assert args[1:] == ["doctor", "--edit", "--json"]
        return subprocess.CompletedProcess(args, 0, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    for argv in ([], ["--json"]):
        calls.clear()
        assert module.main(argv) == 0
        assert calls == [[engine, "--version"], [engine, "doctor", "--edit", "--json"]]
        capsys.readouterr()


@pytest.mark.parametrize("old_build", ["different", None])
@pytest.mark.parametrize("fresh_matches", [True, False])
def test_setup_reinstalls_a_same_version_different_build_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    old_build: str | None, fresh_matches: bool,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    wheel_build = module._wheel_build(wheel)
    assert wheel_build is not None
    calls: list[list[str]] = []
    installed = [False]

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:4] == ["tool", "install", "--force"]:
            installed[0] = True
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[1:] in (["speech", "download"], ["renderer", "install"]):
            return subprocess.CompletedProcess(args, 0, "", "")
        assert args[1:] == ["doctor", "--edit", "--json"]
        current = {**report, "install": {**report["install"]}}
        if installed[0]:
            current["install"]["build"] = wheel_build if fresh_matches else "still-different"
        elif old_build is not None:
            current["install"]["build"] = old_build
        return subprocess.CompletedProcess(args, 0, json.dumps(current), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(["--json"]) == 1
    assert calls == [[engine, "--version"], [engine, "doctor", "--edit", "--json"]]
    assert json.loads(capsys.readouterr().out) == {
        "ready": False, "refused": None,
        "steps": [module._engine_step("/usr/bin/uv", wheel, Path(engine))],
    }
    calls.clear()
    assert module.main() == (0 if fresh_matches else 1)
    assert calls == [
        [engine, "--version"],
        [engine, "doctor", "--edit", "--json"],
        ["/usr/bin/uv", "tool", "install", "--force", f"yap-studio[asr] @ {wheel.as_uri()}"],
        [engine, "speech", "download"],
        [engine, "renderer", "install"],
        [engine, "doctor", "--edit", "--json"],
    ]
    captured = capsys.readouterr()
    if fresh_matches:
        assert "Ready to edit" in captured.out
    else:
        assert "Ready to edit" not in captured.out
        assert captured.err == (
            f"Engine at {engine} is not the build bundled with this workspace; "
            "another install may come first on PATH.\n"
        )


def test_setup_refuses_a_wrong_build_after_version_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    assert module._wheel_build(wheel) is not None
    report["install"]["build"] = "still-different"

    def refused_result(ready: bool, refused: dict | None, steps: list[dict],
                       json_output: bool) -> int:
        assert ready is False
        assert refused == {"reason": "engine build", "service": None, "move": None}
        assert steps == [] and json_output is False
        return 1

    monkeypatch.setattr(module, "_result", refused_result)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, "yap-studio 0.0.0\n", "")
        if args[1:4] == ["tool", "install", "--force"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[1:] in (["speech", "download"], ["renderer", "install"]):
            return subprocess.CompletedProcess(args, 0, "", "")
        assert args[1:] == ["doctor", "--edit", "--json"]
        return subprocess.CompletedProcess(args, 0, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main() == 1
    assert calls == [
        [engine, "--version"],
        ["/usr/bin/uv", "tool", "install", "--force", f"yap-studio[asr] @ {wheel.as_uri()}"],
        [engine, "speech", "download"],
        [engine, "renderer", "install"],
        [engine, "doctor", "--edit", "--json"],
    ]
    captured = capsys.readouterr()
    assert "Ready to edit" not in captured.out
    assert captured.err == (
        f"Engine at {engine} is not the build bundled with this workspace; "
        "another install may come first on PATH.\n"
    )


@pytest.mark.parametrize("speech_engine", ["parakeet-onnx", "parakeet-mlx"])
def test_setup_preserves_speech_engine_on_stale_build_reinstall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    speech_engine: str,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    wheel_build = module._wheel_build(wheel)
    assert wheel_build is not None
    report["speech"]["engines"] = {speech_engine: True}
    calls: list[list[str]] = []
    installed = [False]

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:4] == ["tool", "install", "--force"]:
            installed[0] = True
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[1:] in (["speech", "download"], ["renderer", "install"]):
            return subprocess.CompletedProcess(args, 0, "", "")
        assert args[1:] == ["doctor", "--edit", "--json"]
        current = {**report, "install": {**report["install"]}}
        current["install"]["build"] = wheel_build if installed[0] else "different"
        return subprocess.CompletedProcess(args, 0, json.dumps(current), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(["--json"]) == 1
    assert calls == [[engine, "--version"], [engine, "doctor", "--edit", "--json"]]
    result = json.loads(capsys.readouterr().out)
    assert result["steps"][0]["command"] == (
        "/usr/bin/uv tool install --force " + shlex.quote(f"yap-studio[asr] @ {wheel.as_uri()}")
        + f" && {shlex.quote(engine)} speech download"
        + f" && {shlex.quote(engine)} renderer install"
    )
    calls.clear()
    assert module.main() == 0
    assert calls == [
        [engine, "--version"],
        [engine, "doctor", "--edit", "--json"],
        ["/usr/bin/uv", "tool", "install", "--force", f"yap-studio[asr] @ {wheel.as_uri()}"],
        [engine, "speech", "download"],
        [engine, "renderer", "install"],
        [engine, "doctor", "--edit", "--json"],
    ]
    assert "Ready to edit" in capsys.readouterr().out


def test_wheel_and_installed_build_ids_agree(tmp_path: Path) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_record")
    wheel = _synthetic_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        record = archive.read(f"yap_studio-{VERSION}.dist-info/RECORD").decode()
        package = tmp_path / "yap_studio"
        package.mkdir()
        for name in ("__init__.py", "cli.py"):
            (package / name).write_bytes(archive.read(f"yap_studio/{name}"))
    dist_info = tmp_path / f"yap_studio-{VERSION}.dist-info"
    dist_info.mkdir()
    installed_record = dist_info / "RECORD"
    extra = (
        f"yap_studio-{VERSION}.dist-info/INSTALLER,,\n"
        f"yap_studio-{VERSION}.dist-info/REQUESTED,,\n"
        f"yap_studio-{VERSION}.dist-info/direct_url.json,,\n"
        "../../../bin/yap-studio,,\n"
        "yap_studio/__pycache__/cli.cpython-312.pyc,sha256=ignored,1\n"
    )
    installed_record.write_text(record + extra)
    wheel_build = module._wheel_build(wheel)
    assert wheel_build is not None
    assert install_info.build_id(importlib.metadata.PathDistribution(dist_info)) == wheel_build
    installed_record.write_text((record + extra).replace("sha256=", "sha256=changed", 1))
    assert install_info.build_id(importlib.metadata.PathDistribution(dist_info)) != wheel_build
    installed_record.write_text(extra)
    assert install_info.build_id(importlib.metadata.PathDistribution(dist_info)) is None


@pytest.mark.parametrize("contents", ["invalid", "no_record", "no_package_hashes"])
def test_wheel_build_is_unknown_without_package_hashes(tmp_path: Path, contents: str) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_unknown_build")
    wheel = tmp_path / WHEEL
    if contents == "invalid":
        wheel.write_bytes(b"synthetic wheel")
    else:
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("yap_studio/cli.py", b"pass\n")
            if contents == "no_package_hashes":
                archive.writestr(f"yap_studio-{VERSION}.dist-info/RECORD",
                                 "yap_studio/cli.py,,\n")
    assert module._wheel_build(wheel) is None


def test_new_project_finds_the_engine_where_uv_installs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module(ROOT / "workspace/new_project.py", "workspace_engine_bin")
    xdg_bin = tmp_path / "xdg-bin"
    xdg_bin.mkdir()
    (xdg_bin / "yap-studio").touch()
    monkeypatch.delenv("UV_TOOL_BIN_DIR", raising=False)
    monkeypatch.setattr(module.shutil, "which",
                        lambda name: "/usr/bin/uv" if name == "uv" else None)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert args == ["/usr/bin/uv", "tool", "dir", "--bin"]
        return subprocess.CompletedProcess(args, 0, stdout=f"{xdg_bin}\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    # The folder is not on PATH; the engine is still found where uv put it.
    assert module._engine() == str(xdg_bin / "yap-studio")
    monkeypatch.setenv("UV_TOOL_BIN_DIR", str(tmp_path / "override"))
    assert module._tool_bin_dir() == tmp_path / "override"


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_setup_engine_step_is_posix_shell_text_on_every_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_posix")
    monkeypatch.setattr(module.sys, "platform", platform)
    uv = "C:/Tools&Co/uv.exe"
    wheel = tmp_path / "Yap Studio" / "wheels" / WHEEL
    engine = tmp_path / "R&D bin" / "yap-studio"
    command = module._engine_step(uv, wheel, engine)["command"]
    requirement = f"yap-studio[asr] @ {wheel.resolve().as_uri()}"
    # The agent runs step commands in Bash on every platform, so they must split back exactly.
    assert shlex.split(command) == [
        uv, "tool", "install", "--force", requirement, "&&", str(engine), "speech", "download",
        "&&", str(engine), "renderer", "install",
    ]


def test_guide_opens_the_style_page_in_the_background() -> None:
    guide = (ROOT / "workspace/AGENTS.md").read_text()
    assert "yap-studio style pick --workspace . --inbox inbox --phone --no-open --detach --json" \
        in guide
    assert "yap-studio style pick --stop" in guide
    assert "phone.url" in guide
    assert "same link" in guide
    assert "style/my-style.json" in guide
    assert "Make visual choices in the gallery" not in guide


@pytest.mark.parametrize(("version", "npm", "returncode", "available"), [
    (None, True, 0, False), ("v20.19.0", True, 0, False),
    ("v22.0.0", False, 0, False), ("invalid", True, 0, False),
    ("v22.0.0", True, 1, False), ("v22.0.0", True, 0, True),
    ("24.1.0", True, 0, True), ("v22.1.0-rc.1", True, 0, True),
])
def test_setup_requires_supported_node_and_npm(
    monkeypatch: pytest.MonkeyPatch, version: str | None, npm: bool,
    returncode: int, available: bool,
    node_bins: Callable[..., Path], tmp_path: Path,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_node")
    node = node_bins("node-case", version or "invalid", npm=npm, returncode=returncode)
    monkeypatch.setenv("PATH", str(node.parent if version else tmp_path / "empty"))
    assert module._node_available() is available


@pytest.mark.parametrize("failure", [OSError("cannot run node"),
                                     subprocess.TimeoutExpired("node", 5)])
def test_setup_treats_failed_node_probe_as_unavailable(
    monkeypatch: pytest.MonkeyPatch, failure: Exception,
    node_bins: Callable[..., Path],
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_node_failure")
    node = node_bins("failed-probe")
    monkeypatch.setenv("PATH", str(node.parent))

    def fail(args: list[str], **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(module.subprocess, "run", fail)
    assert module._node_available() is False


@pytest.mark.parametrize("off_path", [False, True])
def test_setup_finds_working_node(
    monkeypatch: pytest.MonkeyPatch, node_bins: Callable[..., Path], off_path: bool,
) -> None:
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_node_lookup")
    broken = node_bins("broken", returncode=1)
    working = node_bins("keg")
    monkeypatch.setenv("PATH", str(broken.parent) if off_path else
                       os.pathsep.join([str(broken.parent), str(working.parent)]))
    module.NODE_INSTALL_PLACES = (str(working.parent),) if off_path else ()
    assert module._node_available() is True


def test_setup_and_engine_node_install_places_match() -> None:
    def constant(file: str, name: str) -> object:
        tree = ast.parse((ROOT / file).read_text())
        return next(ast.literal_eval(item.value) for item in tree.body
                    if isinstance(item, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name
                            for target in item.targets))

    for name in ("NODE_INSTALL_PLACES", "MIN_NODE_MAJOR", "HOMEBREW_BIN_DIRS"):
        assert constant("workspace/setup.py", name) == constant("src/yap_studio/tools.py", name)


@pytest.mark.parametrize("stale", [False, True])
def test_setup_missing_node_installs_speech_and_offers_renderer_after_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    stale: bool,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    if not stale:
        Path(engine).unlink()
    node_ready = False
    installed = False
    report["renderer"] = {"available": False, "browser": False}
    calls: list[list[str]] = []
    monkeypatch.setattr(module, "_node_available", lambda: node_ready, raising=False)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal installed
        calls.append(args)
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:4] == ["tool", "install", "--force"]:
            installed = True
            Path(engine).touch()
        elif args[1:] == ["renderer", "install"]:
            if not node_ready:
                raise subprocess.CalledProcessError(2, args, stderr="Node missing")
            report["renderer"] = {"available": True, "browser": True}
        elif args[1:] == ["doctor", "--edit", "--json"]:
            report["install"]["build"] = module._wheel_build(wheel) if installed else "old"
            report["node"]["available"] = node_ready
            report["ready_to_edit"] = node_ready and report["renderer"]["browser"]
            return subprocess.CompletedProcess(
                args, 0 if report["ready_to_edit"] else 2, json.dumps(report), "")
        else:
            assert args[1:] == ["speech", "download"]
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main() == 1
    output = capsys.readouterr()
    assert output.err == ""
    assert "Doctor ran" in output.out
    assert output.out.index("Install Node 22") < output.out.index("Install the renderer")
    assert [engine, "speech", "download"] in calls
    assert [engine, "renderer", "install"] not in calls
    node_ready = True
    calls.clear()
    assert module.main(["--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["refused"] is None
    assert [step["id"] for step in result["steps"]] == ["renderer"]
    fake_run(shlex.split(result["steps"][0]["command"]), check=True)
    assert module.main() == 0
    assert "Ready to edit" in capsys.readouterr().out
    assert not any(call[1:2] == ["tool"] for call in calls)


@pytest.mark.parametrize("stale", [False, True])
def test_setup_json_missing_node_keeps_engine_step_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    stale: bool,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    if not stale:
        Path(engine).unlink()
    report["install"]["build"] = "old"
    monkeypatch.setattr(module, "_node_available", lambda: False, raising=False)
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        if args[1:] == ["doctor", "--edit", "--json"]:
            return subprocess.CompletedProcess(args, 2, json.dumps(report), "")
        if args[1:4] == ["tool", "install", "--force"]:
            Path(engine).touch()
        else:
            assert args == [engine, "speech", "download"]
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(["--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["refused"] is None and result["ready"] is False
    assert [step["id"] for step in result["steps"]] == ["node", "engine", "renderer"]
    assert shlex.split(result["steps"][1]["command"]) == [
        "/usr/bin/uv", "tool", "install", "--force", f"yap-studio[asr] @ {wheel.as_uri()}",
        "&&", engine, "speech", "download",
    ]
    assert "renderer" not in result["steps"][1]["does"]
    assert shlex.split(result["steps"][2]["command"]) == [engine, "renderer", "install"]
    assert calls == ([[engine, "--version"], [engine, "doctor", "--edit", "--json"]]
                     if stale else [])
    calls.clear()
    assert module._install_wheel("/usr/bin/uv", wheel, Path(engine)) is None
    assert calls == [
        ["/usr/bin/uv", "tool", "install", "--force", f"yap-studio[asr] @ {wheel.as_uri()}"],
        [engine, "speech", "download"],
    ]


@pytest.mark.parametrize("state", ["all_used", "unused", "size", "mtime", "renamed"])
def test_setup_counts_unused_identities_like_project_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    state: str,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    report["install"]["build"] = module._wheel_build(wheel)
    starter = _module(ROOT / "workspace/new_project.py", "workspace_unused_count")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    used_take = inbox / "used.mov"
    used_take.write_bytes(b"used")
    (inbox / "new.mov").write_bytes(b"new")
    with monkeypatch.context() as project_patch:
        _fake_project_runner(starter, project_patch, tmp_path)
        starter.start(tmp_path, takes=["used.mov"])
        if state == "all_used":
            starter.start(tmp_path)
    recorded = used_take.stat()
    if state == "size":
        used_take.write_bytes(b"replacement")
        os.utime(used_take, ns=(recorded.st_atime_ns, recorded.st_mtime_ns))
    elif state == "mtime":
        os.utime(used_take, ns=(recorded.st_atime_ns, recorded.st_mtime_ns + 1_000_000_000))
    elif state == "renamed":
        used_take.rename(inbox / "renamed.mov")
    expected = 0 if state == "all_used" else 1 if state == "unused" else 2

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        assert args == [engine, "doctor", "--edit", "--json"]
        return subprocess.CompletedProcess(args, 0, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main() == 0
    output = capsys.readouterr().out
    noun = "take" if expected == 1 else "takes"
    assert f"{expected} {noun} waiting in inbox/" in output
    assert ("Say 'edit this video' to start." in output) is bool(expected)
    assert module.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["inbox_takes"] == expected
    with monkeypatch.context() as project_patch:
        _fake_project_runner(starter, project_patch, tmp_path)
        for _ in range(expected):
            starter.start(tmp_path)
        with pytest.raises(ValueError, match="already in a project"):
            starter.start(tmp_path)
    assert module._inbox_take_count() == 0


@pytest.mark.parametrize("contents", [
    b"not json", b"\xff", b'{"schema":"wrong","takes":[]}',
    b'{"schema":"yap-studio/inbox-used@1","takes":[{"file":"take.mov"}]}', None,
])
def test_setup_refuses_invalid_inbox_ledger_like_project_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    contents: bytes | None,
) -> None:
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    report["install"]["build"] = module._wheel_build(wheel)
    starter = _module(ROOT / "workspace/new_project.py", "workspace_invalid_ledger")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "take.mov").write_bytes(b"take")
    ledger = inbox / ".used.json"
    if contents is None:
        ledger.mkdir()
    else:
        ledger.write_bytes(contents)
    with pytest.raises(ValueError, match="fix it before starting a project") as error:
        starter.start(tmp_path)

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if args[1:] == ["--version"]:
            return subprocess.CompletedProcess(args, 0, f"yap-studio {VERSION}\n", "")
        assert args == [engine, "doctor", "--edit", "--json"]
        return subprocess.CompletedProcess(args, 0, json.dumps(report), "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main() == 1
    output = capsys.readouterr()
    assert str(error.value) in output.err
    assert "Ready to edit" not in output.out
    assert module.main(["--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ready"] is False and "inbox_takes" not in result
    assert result["refused"]["reason"] == "inbox"
    assert result["refused"]["error"] == str(error.value)
    if contents is not None:
        assert ledger.read_bytes() == contents


def test_shared_workspace_instructions_resolve_to_one_source():
    workspace = ROOT / "workspace"
    imported = (workspace / "CLAUDE.md").read_text().strip()
    assert imported == "@AGENTS.md"
    assert (workspace / imported[1:]).read_text() == (workspace / "AGENTS.md").read_text()


@pytest.mark.parametrize("agent", ["claude-code", "codex", "cursor"])
def test_setup_records_app_and_preserves_it_without_option(tmp_path, monkeypatch, capsys, agent):
    from yap_studio.homepage.state import HomeState

    module = _module(ROOT / "workspace/setup.py", "workspace_setup_agent")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "_synced_service", lambda _: None)
    monkeypatch.setattr(module, "_free_disk_gb", lambda _: 100)
    folder = tmp_path / ".yap-studio"
    folder.mkdir()
    saved = folder / "setup.json"
    saved.write_text(json.dumps({"other": "preserved"}))
    # No wheel: recording the app must work before the first installation.
    assert module.main(["--agent", agent, "--json"]) == 1
    assert json.loads(saved.read_text()) == {"other": "preserved", "agent": agent}
    assert module.main(["--json"]) == 1
    (tmp_path / "inbox").mkdir()
    assert HomeState(tmp_path).snapshot()["agent"] == agent
    assert json.loads(saved.read_text())["agent"] == agent
    capsys.readouterr()


def test_setup_rejects_unknown_app_without_writing(tmp_path, monkeypatch):
    module = _module(ROOT / "workspace/setup.py", "workspace_setup_bad_agent")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        module.main(["--agent", "unknown"])
    assert not (tmp_path / ".yap-studio").exists()


@pytest.mark.parametrize("stored", [
    None, "{}", '{"agent":"unknown"}', '{"agent":[]}', '{"agent":{}}', "{broken", "[]",
])
def test_old_workspace_app_defaults_to_claude_code(tmp_path, stored):
    from yap_studio.homepage.state import HomeState

    if stored is not None:
        folder = tmp_path / ".yap-studio"
        folder.mkdir()
        (folder / "setup.json").write_text(stored)
    (tmp_path / "inbox").mkdir()
    assert HomeState(tmp_path).snapshot()["agent"] == "claude-code"


def test_workspace_permission_presets():
    workspace = ROOT / "workspace"
    config = tomllib.loads((workspace / ".codex/config.toml").read_text())
    assert config["sandbox_mode"] == "workspace-write"
    assert config["approval_policy"] == "on-request"
    sandbox = config["sandbox_workspace_write"]
    assert sandbox["network_access"] is True
    assert sandbox["writable_roots"] == [
        "~/Library/Application Support/yap-studio", "~/Library/Caches/yap-studio",
        "~/.local/share/yap-studio", "~/.cache/yap-studio",
    ]
    rules = []
    exec((workspace / ".codex/rules/yap-studio.rules").read_text(),
         {"prefix_rule": lambda **rule: rules.append(rule), "__builtins__": {}})
    assert [(rule["pattern"], rule["decision"]) for rule in rules] == [
        (["yap-studio"], "allow"), (["open"], "allow"),
    ]
    cursor = json.loads((workspace / ".cursor/permissions.json").read_text())
    assert cursor["terminalAllowlist"] == ["yap-studio", "open http://127.0.0.1:"]
    assert json.loads((workspace / ".cursor/cli.json").read_text()) == {
        "permissions": {"allow": ["Shell(yap-studio)", "Shell(open:http://127.0.0.1*)"],
                        "deny": []},
    }
    assert json.loads((workspace / ".claude/settings.json").read_text()) == {
        "permissions": {"allow": ["Bash(python3 setup.py *)", "Bash(python3 new_project.py *)",
                                  "Bash(yap-studio *)", "Bash(open http://*)"]},
    }


@pytest.mark.parametrize("on_path", [False, True])
def test_setup_engine_path_step(tmp_path, monkeypatch, capsys, on_path):
    module, wheel, engine, report = _build_setup_fixture(tmp_path, monkeypatch)
    report["install"]["build"] = module._wheel_build(wheel)
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "tools"))
    monkeypatch.setattr(module.shutil, "which", lambda name:
                        (engine if on_path else None) if name == "yap-studio"
                        else "/scratch/uv")
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0,
                            f"yap-studio {VERSION}\n" if args[1:] == ["--version"]
                            else json.dumps(report), ""))
    assert module.main(["--json"]) == (0 if on_path else 1)
    result = json.loads(capsys.readouterr().out)
    assert result["engine"] == engine
    assert result["ready"] is on_path
    assert [step["id"] for step in result["steps"]] == ([] if on_path else ["engine_path"])
    if not on_path:
        assert result["steps"][0]["command"] == "/scratch/uv tool update-shell"
        assert result["steps"][0]["optional"] is False


def test_question_and_wait_instructions_are_app_neutral():
    for name in ["AGENTS.md", ".claude/skills/edit-video/SKILL.md",
                 ".claude/skills/voiceover-video/SKILL.md"]:
        guide = (ROOT / "workspace" / name).read_text()
        assert "AskUserQuestion" not in guide
        for phrase in ("one question at a time", "recommended choice first", "otherwise in chat",
                       "--timeout 1", "check again each turn", "pressed a page button"):
            assert phrase in guide
