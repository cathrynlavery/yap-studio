"""Command line scaffold tests."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_install_info import build_checkout

from yap_studio import __version__, cli, install_info
from yap_studio.affiliate import ELEVENLABS_AFFILIATE_URL
from yap_studio.cli import main
from yap_studio.review import phone
from yap_studio.style import PRESETS_DIR, RESOLVED_SCHEMA, SHIPPED_PRESETS, load_style, resolve

pytestmark = pytest.mark.usefixtures("_no_real_node")

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(("photos_platform", "photos_installed"), [
    ("darwin", True), ("darwin", False), ("linux", False),
])
@pytest.mark.parametrize(("binary", "name", "lan", "state", "via", "text"), [
    (None, None, "192.168.1.10", "missing", "wifi",
     "phone    Wi-Fi link (Tailscale not set up)"),
    (None, None, None, "missing", None,
     "phone    no network a phone can reach (connect to Wi-Fi or set up Tailscale)"),
    ("/usr/bin/tailscale", None, "192.168.1.10", "installed", "wifi",
     "phone    Wi-Fi link (Tailscale not connected)"),
    ("/usr/bin/tailscale", None, None, "installed", None,
     "phone    no network a phone can reach (connect to Wi-Fi or set up Tailscale)"),
    ("/usr/bin/tailscale", "box.tailnet.example.invalid", None, "connected", "tailscale",
     "phone    Tailscale (box.tailnet.example.invalid)"),
])
def test_doctor_setup_checks_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    binary: str | None, name: str | None, lan: str | None, state: str,
    via: str | None, text: str,
    photos_platform: str, photos_installed: bool,
    node_bins: Callable[..., Path],
) -> None:
    from yap_studio import setup_checks
    from yap_studio.render import hyperframes

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "resolve_tool", lambda name: name)
    monkeypatch.setattr(cli, "check_version", lambda path: None)
    monkeypatch.setattr(cli, "tool_versions", lambda: {"ffmpeg": "6.1", "ffprobe": "6.1"})
    monkeypatch.setattr(cli, "has_filters", lambda names: dict.fromkeys(names, True))
    monkeypatch.setattr(cli, "available_engines", lambda: {"stub": True})
    monkeypatch.setattr(cli, "resolve_backend", lambda: SimpleNamespace(
        name="stub", download_notice=lambda: "needs download"))
    monkeypatch.setattr(sys, "platform", photos_platform)
    broken = node_bins("broken", returncode=1)
    working = node_bins("working")
    monkeypatch.setenv("PATH", os.pathsep.join([str(broken.parent), str(working.parent)]))
    find_tool, run_tool = cli.tools.find_tool, cli.tools.run_tool
    monkeypatch.setattr(cli.tools, "find_tool", lambda name: (
        find_tool(name) if name in {"node", "npm"} else
        None if name == "osxphotos" and not photos_installed else name))
    monkeypatch.setattr(cli.tools, "run_tool", lambda name, *args, **kwargs: (
        run_tool(name, *args, **kwargs) if name == "node" else SimpleNamespace(stdout="")))
    monkeypatch.setattr(hyperframes, "_npm_installed", lambda: object())
    monkeypatch.setattr(hyperframes, "installed", lambda: object())
    service = setup_checks.SYNC_LABELS["db"]
    monkeypatch.setattr(setup_checks, "synced_service", lambda path: service)
    monkeypatch.setattr(setup_checks, "free_disk_gb", lambda path: 8.5)
    monkeypatch.setattr(setup_checks, "cutout_reason", lambda: "Xcode tools missing")
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: binary)
    monkeypatch.setattr(phone, "tailscale_name", lambda: name)
    monkeypatch.setattr(phone, "lan_address", lambda: lan)

    from yap_studio.product_feedback import setting

    setting("never")
    assert main(["doctor", "--edit", "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["feedback"]["mode"] == "never"
    assert report["feedback"]["endpoint"] is False
    assert report["node"] == {"available": True, "version": "v26.1.0", "path": str(working)}
    assert report["folder"] == {"path": str(tmp_path), "synced": True, "service": service}
    assert report["disk"] == {"free_gb": 8.5, "min_gb": 10, "enough": False}
    assert report["speech"]["model"] is False
    assert report["cutout"] == {"available": False, "reason": "Xcode tools missing"}
    assert report["phone"] == {"tailscale": state, "name": name, "via": via}
    assert report["photos"] == (
        {"osxphotos": photos_installed, "optional": True} if photos_platform == "darwin" else None)
    assert not any("osxphotos" in warning for warning in report["warnings"])
    assert report["stock"] == {"provider": "pexels", "cli": True,
                               "signed_in": True, "optional": True}
    assert report["edit_missing"] == ["folder", "disk"]
    assert not report["ready_to_edit"]
    assert any("speech model" in warning for warning in report["warnings"])
    assert any("cutout" in warning for warning in report["warnings"])
    assert main(["doctor", "--json"]) == 0
    capsys.readouterr()
    assert main(["doctor", "--edit"]) == 2
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith("feedback never;") for line in lines)
    assert any(line.startswith("folder   ") for line in lines)
    assert any(line.startswith("disk     ") for line in lines)
    assert any(line.startswith("model    ") for line in lines)
    assert any(line.startswith("cutout   ") for line in lines)
    assert text in lines
    if photos_platform == "darwin":
        assert ("photos   osxphotos ready" if photos_installed else
                "photos   osxphotos not installed (optional, for Photos albums: "
                "uv tool install osxphotos)") in lines
    else:
        assert not any(line.startswith("photos ") for line in lines)
    monkeypatch.setattr(setup_checks, "synced_service", lambda path: None)
    monkeypatch.setattr(setup_checks, "free_disk_gb", lambda path: 10.0)
    assert main(["doctor", "--edit", "--json"]) == 0
    ready = json.loads(capsys.readouterr().out)
    assert ready["edit_missing"] == []
    assert ready["ready_to_edit"] is True


def test_doctor_signed_out_elevenlabs_is_an_optional_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    _no_real_elevenlabs: Callable[[Path], None],
) -> None:
    from yap_studio import setup_checks
    from yap_studio.render import hyperframes

    fake = ROOT / "tests/fakes/elevenlabs-pp-cli"
    _no_real_elevenlabs(fake)
    monkeypatch.setenv("PATH", str(fake.parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FAKE_ELEVENLABS_AUTH_EXIT", "4")
    log = tmp_path / "elevenlabs.log"
    monkeypatch.setenv("FAKE_ELEVENLABS_LOG", str(log))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "resolve_tool", lambda name: name)
    monkeypatch.setattr(cli, "check_version", lambda path: None)
    monkeypatch.setattr(cli, "tool_versions", lambda: {"ffmpeg": "6.1", "ffprobe": "6.1"})
    monkeypatch.setattr(cli, "has_filters", lambda names: dict.fromkeys(names, True))
    monkeypatch.setattr(cli, "available_engines", lambda: {})
    monkeypatch.setattr(cli, "resolve_backend", lambda: None)
    monkeypatch.setattr(setup_checks, "cutout_reason", lambda: None)
    monkeypatch.setattr(setup_checks, "free_disk_gb", lambda path: 20.0)
    monkeypatch.setattr(setup_checks, "synced_service", lambda path: None)
    monkeypatch.setattr(hyperframes, "_npm_installed", lambda: None)
    monkeypatch.setattr(hyperframes, "installed", lambda: None)
    monkeypatch.setattr(phone, "_tailscale_binary", lambda: None)
    monkeypatch.setattr(phone, "lan_address", lambda: None)
    original_find = cli.tools.find_tool
    monkeypatch.setattr(cli.tools, "find_tool", lambda name: (
        None if name in {"node", "npm"} else original_find(name)))

    assert main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["elevenlabs"]["cli"] is True
    assert report["node"] == {"available": False, "version": None, "path": None}
    assert report["elevenlabs"]["signed_in"] is False
    assert any(ELEVENLABS_AFFILIATE_URL in item for item in report["elevenlabs"]["setup"])
    assert "elevenlabs" not in report["edit_missing"]
    assert not any("ElevenLabs" in item for item in report["warnings"])
    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert "audio    ElevenLabs CLI signed out" in output
    assert any(line.startswith("note: ") and ELEVENLABS_AFFILIATE_URL in line
               for line in output.splitlines())
    assert all(json.loads(line)["argv"] == ["auth", "status", "--json"]
               for line in log.read_text().splitlines())
    monkeypatch.setattr(cli.tools, "find_tool", lambda name: (
        None if name in {"node", "npm", "elevenlabs-pp-cli"} else original_find(name)))
    assert main(["doctor", "--json"]) == 0
    missing = json.loads(capsys.readouterr().out)
    assert missing["elevenlabs"]["cli"] is False
    assert any("install elevenlabs --cli-only" in item
               for item in missing["elevenlabs"]["setup"])

    from yap_studio.pictures import wavespeed

    baseline = {key: missing[key] for key in ("ready_to_edit", "edit_missing")}
    for state in ("missing_cli", "signed_out", "ready"):
        monkeypatch.setattr(wavespeed, "status", lambda state=state: state)
        setup = ([] if state == "ready" else
                 ([wavespeed.INSTALL_LINE] if state == "missing_cli" else [])
                 + [wavespeed.SIGNUP_LINE, wavespeed.KEY_LINE])
        assert main(["doctor", "--json"]) == 0
        pictures = json.loads(capsys.readouterr().out)
        assert pictures["pictures"] == {
            "provider": "wavespeed", "cli": state != "missing_cli",
            "signed_in": state == "ready", "optional": True, "setup": setup,
        }
        assert {key: pictures[key] for key in baseline} == baseline
        assert main(["doctor"]) == 0
        output = capsys.readouterr().out
        label = {"missing_cli": "not installed", "signed_out": "signed out",
                 "ready": "ready"}[state]
        line = f"pictures WaveSpeed CLI {label} (optional, for generated pictures)"
        if state != "ready":
            line += ". " + wavespeed.SIGNUP_LINE
        assert line in output.splitlines()
        assert (wavespeed.SIGNUP_LINE in output) == (state != "ready")


def test_speech_download_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    backend = SimpleNamespace(
        name="stub", model="stub/model", download_notice=lambda: None,
        prefetch=lambda: calls.append("download"),
    )
    monkeypatch.setattr(cli, "resolve_backend", lambda: backend)
    assert main(["speech", "download"]) == 0
    assert capsys.readouterr().out == "Speech model ready.\n"
    assert calls == []
    backend.download_notice = lambda: "needs download"
    assert main(["speech", "download"]) == 0
    assert calls == ["download"]
    assert "stub/model" in capsys.readouterr().out
    monkeypatch.setattr(cli, "resolve_backend", lambda: None)
    assert main(["speech", "download"]) == 2
    assert "fix:" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["verify", "synthetic.mp4"], ["mcp"], ["style", "presets"]])
def test_parser_does_not_inspect_install(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    def unexpected() -> None:
        pytest.fail("ordinary command parsing inspected the install")

    monkeypatch.setattr(install_info, "version_text", unexpected)
    monkeypatch.setattr(install_info, "source_checkout", unexpected)
    parser = cli._parser()
    assert parser.parse_args(argv).command == argv[0]


@pytest.mark.parametrize(("arguments", "expected"), [
    (["fillers", "set", "DIR", "ID", "rejected"],
     {"directory": "DIR", "identifier": "ID", "removal_status": "rejected"}),
    (["fillers", "add", "DIR", "--manual", "TAKE", "START", "END"],
     {"directory": "DIR", "manual": ["TAKE", "START", "END"]}),
    (["status", "DIR"], {"directory": "DIR"}),
])
def test_argument_parser_selects_the_command_parser(arguments: list[str], expected: dict) -> None:
    parser, remaining = cli._argument_parser(arguments)
    args = parser.parse_args(remaining)
    assert all(getattr(args, name) == value for name, value in expected.items())


def test_ordinary_command_does_not_inspect_install(
    style_config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unexpected() -> None:
        pytest.fail("ordinary command execution inspected the install")

    monkeypatch.setattr(install_info, "version_text", unexpected)
    monkeypatch.setattr(install_info, "source_checkout", unexpected)
    assert main(["style", "presets", "--json"]) == 0
    assert [preset["id"] for preset in json.loads(capsys.readouterr().out)] == list(SHIPPED_PRESETS)


def test_main_version(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("yap_studio.install_info.source_checkout", lambda: None)
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"yap-studio {__version__}\n"


def test_main_version_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, commit = build_checkout(tmp_path, version=__version__)
    monkeypatch.setattr("yap_studio.install_info.source_checkout", lambda: tmp_path)
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"yap-studio {__version__} (source {commit})\n"


def test_module_version() -> None:
    from yap_studio.install_info import version_text

    env = dict(os.environ)
    source = str(ROOT / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "yap_studio", "--version"],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout == version_text() + "\n"


def test_installed_console_scripts() -> None:
    command = "yap-studio"
    executable = shutil.which(command)
    if executable is None:
        pytest.skip(f"{command} is not installed")
    result = subprocess.run(
        [executable, "--version"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert re.fullmatch(
        rf"yap-studio {re.escape(__version__)}"
        r"(?: \(source (?:[0-9a-f]{12}|unknown)(?:, modified)?\))?\n",
        result.stdout,
    )


def test_project_console_scripts() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert project["scripts"] == {"yap-studio": "yap_studio.cli:main"}


@pytest.fixture
def style_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "config"
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(config))
    return config


def test_style_presets_json(style_config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["style", "presets", "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    presets = json.loads(captured.out)
    assert [preset["id"] for preset in presets] == list(SHIPPED_PRESETS)
    assert {preset["name"] for preset in presets} == {
        "Clean Minimal", "Electric", "Warm Editorial", "Editorial Serif", "One Word",
        "Cobalt", "Mono", "Headline", "Candy", "Marker", "Bold", "Editorial", "Neutral",
    }
    assert all(set(preset) == {"id", "name"} for preset in presets)
    assert not style_config.exists()


def test_style_new_lists_presets_and_creates_named_file(
    style_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["style", "new", "--list", "--json"]) == 0
    assert [item["id"] for item in json.loads(capsys.readouterr().out)] == list(SHIPPED_PRESETS)
    assert main(["style", "new", "My Look", "--preset", "electric", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    target = tmp_path / "my_look.json"
    assert result == {"path": "my_look.json", "name": "My Look", "preset": "electric"}
    assert load_style(target)["name"] == "My Look"
    assert load_style(target)["captions"] == load_style(
        PRESETS_DIR / "electric.json"
    )["captions"]
    before = target.read_bytes()
    assert main(["style", "new", "My Look"]) == 2
    assert "already exists" in capsys.readouterr().err
    assert target.read_bytes() == before
    assert not style_config.exists()


def test_style_new_default_and_explicit_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "custom.json"
    assert main(["style", "new", "Neutral", "--out", str(target)]) == 0
    assert capsys.readouterr().out == f"wrote {target}\n"
    assert load_style(target)["name"] == "Neutral"
    assert load_style(target)["graphics"] == resolve("preset:clean_minimal").data["graphics"]
    assert main(["style", "new", "", "--out", str(tmp_path / "empty.json")]) == 2
    assert not (tmp_path / "empty.json").exists()


@pytest.mark.parametrize("json_output", [False, True])
def test_style_validate_valid_and_invalid_exit_codes(
    style_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], json_output: bool
) -> None:
    source = tmp_path / "style.json"
    source.write_text('{"schema_version": 1}', encoding="utf-8")
    args = ["style", "validate", str(source)] + (["--json"] if json_output else [])
    assert main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        assert json.loads(captured.out) == {"status": "valid"}
    else:
        assert captured.out == "valid\n"
    source.write_text('{"schema_version": 1, "captions": {"size_px": 1}}', encoding="utf-8")
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        error = json.loads(captured.err)["error"]
        assert error["code"] == "invalid_parameter"
        assert "style.captions.size_px" in error["message"]
        assert "style.captions.size_px" in error["suggested_action"]
    else:
        assert "error: " in captured.err
        assert "style.captions.size_px" in captured.err
        assert "fix: " in captured.err
    assert not style_config.exists()


@pytest.mark.parametrize("json_output", [False, True])
def test_style_add_refusal_exits_three_and_replace_succeeds(
    style_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], json_output: bool
) -> None:
    source = tmp_path / "mine.json"
    source.write_text('{"schema_version": 1, "name": "First"}', encoding="utf-8")
    args = ["style", "add", str(source)] + (["--json"] if json_output else [])
    assert main(args) == 0
    capsys.readouterr()
    target = style_config / "styles" / "mine.json"
    before = target.read_bytes()
    source.write_text('{"schema_version": 1, "name": "Second"}', encoding="utf-8")
    assert main(args) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        error = json.loads(captured.err)["error"]
        assert error["code"] == "style_exists"
        assert "--replace" in error["suggested_action"]
    else:
        assert "error: " in captured.err
        assert "fix: " in captured.err
        assert "--replace" in captured.err
    assert target.read_bytes() == before
    assert main([*args, "--replace"]) == 0
    capsys.readouterr()
    assert json.loads(target.read_text(encoding="utf-8"))["name"] == "Second"


def test_style_cli_default_show_list_and_resolve(
    style_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["style", "default", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"default_style": "preset:clean_minimal"}
    assert main(["style", "show", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "Clean Minimal"
    assert main(["style", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    source = tmp_path / "mine.json"
    source.write_text('{"schema_version": 1, "name": "Mine"}', encoding="utf-8")
    assert main(["style", "add", str(source), "--default", "--json"]) == 0
    entry = json.loads(capsys.readouterr().out)
    assert entry["id"] == "mine"
    assert entry["is_default"]
    assert entry["path"] == str(style_config / "styles" / "mine.json")
    assert main(["style", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [entry]
    assert main(["style", "show", "--json"]) == 0
    style = json.loads(capsys.readouterr().out)
    assert style["schema"] == RESOLVED_SCHEMA
    assert style["name"] == "Mine"
    output = tmp_path / "resolved-style.json"
    for _ in range(2):
        assert main(["style", "resolve", "--out", str(output), "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["path"] == str(output)
        assert len(result["sha256"]) == 64
        assert json.loads(output.read_text(encoding="utf-8")) == style
    assert main(["style", "default", "preset:electric", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"default_style": "preset:electric"}
    assert main(["style", "resolve", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "Electric"


def test_style_add_invalid_leaves_config_absent(
    style_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "invalid.json"
    source.write_text('{"schema_version": 1, "unknown": true}', encoding="utf-8")
    assert main(["style", "add", str(source), "--default", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert "style.unknown" in error["message"]
    assert not style_config.exists()
