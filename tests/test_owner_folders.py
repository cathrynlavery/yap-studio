"""Creator folders stay isolated and unexpected home writes fail the test."""

from __future__ import annotations

import json
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import conftest
import pytest
from conftest import _link_renderer_install, _renderer_install_failure
from test_render import stub_install_tools

from yap_studio import background, cache, paths, sfx, tools
from yap_studio.faces import detect
from yap_studio.media import captured, inventory
from yap_studio.render import hyperframes
from yap_studio.review.store import data_dir
from yap_studio.style.library import config_root

pytest_plugins = ["pytester"]


def test_owner_roots_are_temporary(
    tmp_path: Path, _owner_folders: Path, _shared_helpers: Path,
    _shared_cache_root: Path,
) -> None:
    owner = _owner_folders
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    facts, _ = captured._read_cache("guard")
    roots = (
        data_dir(), config_root(), cache.cache_root(), inventory.inventory_root(), facts.parent,
    )
    assert roots == (owner / "data", owner / "config", owner / "cache",
                     owner / "cache" / "media-inventory", owner / "cache" / "media-facts")
    for root in roots:
        assert root.is_relative_to(owner)
        assert not root.resolve().is_relative_to(real_home)
        assert not root.is_relative_to(tmp_path)
    assert background.directory() == owner / "data" / "detached"
    renderer = owner / "cache" / "renderer"
    assert renderer.is_dir() and not renderer.is_symlink()
    prefix = hyperframes.install_prefix()
    assert prefix.parent == renderer
    shared = _shared_cache_root / "renderer" / prefix.name
    if prefix.exists():
        assert prefix.is_dir() and not prefix.is_symlink()
        assert not (prefix / "install.json").is_symlink()
        assert (prefix / "install.json").read_bytes() == (
            shared / "install.json"
        ).read_bytes()
        assert (prefix / "node_modules").is_symlink()
        assert (prefix / "node_modules").resolve() == (
            shared / "node_modules"
        ).resolve()
    else:
        assert _renderer_install_failure(shared) is not None
    helpers = owner / "cache" / "helpers"
    assert helpers.is_symlink()
    assert helpers.resolve() == _shared_helpers.resolve()
    assert sfx.sfx_cache_root() == tmp_path / "sfx-cache"


def test_home_is_empty_and_temporary(_owner_folders: Path) -> None:
    assert Path.home() == _owner_folders / "home"
    assert list(Path.home().iterdir()) == []


def test_missing_helper_build_stays_in_worker_cache(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory,
    _shared_cache_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "synthetic.swift"
    source.write_text(f"// unique guard source {uuid4().hex}\n")
    monkeypatch.setattr(detect, "helper_source", lambda: source)
    basetemp = tmp_path_factory.getbasetemp().resolve()
    real_helpers = _shared_cache_root / "helpers"
    target = detect.helper_path()
    # Fail before the old guard could create a build temporary in the real cache.
    assert target.resolve().is_relative_to(basetemp)
    compiles: list[Path] = []

    def compile_helper(
        name: str, args: list[str], **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        assert name == "xcrun" and args[:3] == ["swiftc", "-O", "-o"]
        output = Path(args[3])
        assert output.resolve().is_relative_to(basetemp)
        output.write_text("#!/bin/sh\nexit 0\n")
        output.chmod(0o700)
        compiles.append(output)
        return subprocess.CompletedProcess([name, *args], 0, "", "")

    monkeypatch.setattr(tools, "run_tool", compile_helper)
    result = detect.helper_executable()
    assert result.resolve().is_relative_to(basetemp)
    assert not result.resolve().is_relative_to(real_helpers.resolve())
    assert result.is_file() and os.access(result, os.X_OK)
    assert not (real_helpers / result.name).exists()
    assert detect.helper_executable() == result
    assert len(compiles) == 1


def test_renderer_metadata_and_removal_leave_shared_install_unchanged(
    tmp_path: Path, _shared_cache_root: Path,
) -> None:
    prefix = hyperframes.install_prefix()
    real_prefix = _shared_cache_root / "renderer" / prefix.name
    metadata = hyperframes._install_metadata(real_prefix)
    executable = real_prefix / "node_modules" / ".bin" / "hyperframes"
    if (metadata is None or not executable.is_file() or not os.access(executable, os.X_OK)
            or metadata["browser_path"] is None or not Path(metadata["browser_path"]).exists()):
        pytest.skip("the real pinned renderer install is not complete")
    # These assertions must precede any write or removal so the old guard fails safely.
    assert prefix.is_dir() and not prefix.is_symlink()
    private_metadata = prefix / "install.json"
    assert private_metadata.is_file() and not private_metadata.is_symlink()
    assert (prefix / "node_modules").is_symlink()
    assert (prefix / "node_modules").resolve() == (real_prefix / "node_modules").resolve()
    original = (real_prefix / "install.json").read_bytes()
    metadata["browser_path"] = str(tmp_path / "missing-browser")
    paths.atomic_write_text(private_metadata, json.dumps(metadata) + "\n")
    assert private_metadata.read_bytes() != original
    shutil.rmtree(prefix)
    assert not prefix.exists()
    assert (real_prefix / "install.json").read_bytes() == original
    assert (real_prefix / "node_modules").is_dir()


def test_undo_keeps_owner_guards(
    tmp_path: Path, _owner_folders: Path, monkeypatch: pytest.MonkeyPatch,
    _shared_cache_root: Path,
) -> None:
    monkeypatch.setenv("YAP_TEST_LOCAL_PATCH", "temporary")
    monkeypatch.undo()
    assert data_dir() == _owner_folders / "data"
    assert config_root() == _owner_folders / "config"
    assert cache.cache_root() == _owner_folders / "cache"
    assert inventory.inventory_root() == _owner_folders / "cache" / "media-inventory"
    assert captured._read_cache("guard")[0].parent == _owner_folders / "cache" / "media-facts"
    assert sfx.sfx_cache_root() == tmp_path / "sfx-cache"
    assert Path.home() == _owner_folders / "home"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert tools.find_tool("elevenlabs-pp-cli") is None
    assert tools.find_tool("osxphotos") is None
    _assert_renderer_tool_caches_are_private(_owner_folders, _shared_cache_root)


def _assert_fake_renderer_install_stays_in_owner_cache(
    _owner_folders: Path, _shared_cache_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = f"0.0.0-guard-{uuid4().hex}"
    monkeypatch.setattr(hyperframes, "HYPERFRAMES_VERSION", fake)
    calls = stub_install_tools(_owner_folders / "cache", monkeypatch)
    real_prefix = _shared_cache_root / "renderer" / f"hyperframes-{fake}"
    assert not real_prefix.exists()

    result = hyperframes.ensure_installed(timeout_s=17)

    prefix = _owner_folders / "cache" / "renderer" / f"hyperframes-{fake}"
    assert result.prefix == prefix == hyperframes.install_prefix()
    assert prefix.is_dir()
    assert not prefix.is_symlink()
    assert (prefix.parent / ".install.lock").is_file()
    assert json.loads((prefix / "install.json").read_text())["version"] == fake
    assert result == hyperframes.installed()
    assert [name for name, _, _ in calls] == ["npm", "hyperframes"]
    assert not real_prefix.exists()
    assert not real_prefix.is_symlink()
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    # Search the complete cache trees used by the engine, npm and browser tools.
    # Other applications' caches can contain hundreds of thousands of folders.
    for root in (
        _shared_cache_root, real_home / "Library/Caches/yap-studio",
        real_home / "Library/Caches/hyperframes", real_home / ".cache/yap-studio",
        real_home / ".cache/hyperframes", real_home / ".cache/puppeteer", real_home / ".npm",
    ):
        for directory, folders, files in os.walk(root):
            assert prefix.name not in folders + files, f"fake renderer leaked under {directory}"


def test_fake_renderer_install_stays_in_owner_cache(
    _owner_folders: Path, _shared_cache_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_fake_renderer_install_stays_in_owner_cache(
        _owner_folders, _shared_cache_root, monkeypatch,
    )


@pytest.mark.render
def test_fake_renderer_install_with_real_home_stays_in_owner_cache(
    _owner_folders: Path, _shared_cache_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Path.home().resolve() == Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    _assert_fake_renderer_install_stays_in_owner_cache(
        _owner_folders, _shared_cache_root, monkeypatch,
    )


@pytest.mark.render
def test_render_uses_owner_folders_and_shared_install(
    _owner_folders: Path, _shared_renderer_install: Path | None, _shared_cache_root: Path,
    _npm_cache: Path,
) -> None:
    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    assert Path.home().resolve() == real_home
    assert (data_dir(), config_root(), cache.cache_root()) == tuple(
        _owner_folders / name for name in ("data", "config", "cache")
    )
    for root in (data_dir(), config_root(), cache.cache_root()):
        assert not root.resolve().is_relative_to(real_home)
    prefix = hyperframes.install_prefix()
    shared = _shared_cache_root / "renderer" / prefix.name
    session = "no shared install at session start; " if _shared_renderer_install is None else ""
    assert hyperframes.installed() is not None, session + (
        _renderer_install_failure(shared) or _renderer_install_failure(prefix)
        or "browser disappeared after validation"
    )
    _assert_renderer_tool_caches_are_private(_owner_folders, _shared_cache_root)
    assert Path(os.environ["npm_config_cache"]) == _npm_cache


def _assert_renderer_tool_caches_are_private(owner: Path, shared_cache: Path) -> None:
    for name in (
        "npm_config_userconfig", "npm_config_logs_dir", "PUPPETEER_CACHE_DIR",
        "NODE_COMPILE_CACHE", "NODE_REPL_HISTORY", "COREPACK_HOME", "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME", "XDG_STATE_HOME", "HYPERFRAMES_FONT_CACHE_DIR",
        "HYPERFRAMES_HISTORY_ROOT", "HYPERFRAMES_EXTRACT_CACHE_DIR",
    ):
        assert Path(os.environ[name]).is_relative_to(owner), name
    assert Path(os.environ["npm_config_userconfig"]).read_bytes() == b""
    assert os.environ["npm_config_update_notifier"] == "false"
    shared = shared_cache / "renderer" / hyperframes.install_prefix().name
    metadata = hyperframes._install_metadata(shared)
    if metadata and metadata["browser_path"] and Path(metadata["browser_path"]).exists():
        assert os.environ["HYPERFRAMES_BROWSER_PATH"] == metadata["browser_path"]


def test_renderer_tool_caches_are_private(
    _owner_folders: Path, _shared_cache_root: Path, _npm_cache: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _assert_renderer_tool_caches_are_private(_owner_folders, _shared_cache_root)
    assert Path(os.environ["npm_config_cache"]) == _npm_cache
    assert _npm_cache.is_relative_to(tmp_path_factory.getbasetemp())
    assert not _npm_cache.is_relative_to(_owner_folders)


@pytest.mark.parametrize("reuse_browser", [False, True])
def test_renderer_install_subprocess_caches_stay_out_of_home(
    _owner_folders: Path, _npm_cache: Path, monkeypatch: pytest.MonkeyPatch,
    reuse_browser: bool,
) -> None:
    owner = _owner_folders
    assert Path.home() == owner / "home"
    fake = f"0.0.0-subprocess-{uuid4().hex}"
    monkeypatch.setattr(hyperframes, "HYPERFRAMES_VERSION", fake)
    browser_script = owner / "fake-browser"
    browser_script.write_text(f'''#!{sys.executable}
import os
import sys
from pathlib import Path
assert sys.argv[1:] == ["browser", "ensure"]
assert os.environ["HYPERFRAMES_NO_TELEMETRY"] == "1"
assert os.environ.get("CI") == "1"
existing = os.environ.get("HYPERFRAMES_BROWSER_PATH")
if existing:
    assert Path(existing).is_file()
    browser = Path(existing)
else:
    cache = Path(os.environ.get("PUPPETEER_CACHE_DIR", str(Path.home() / ".cache/puppeteer")))
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "browser-download").touch()
    # The pinned CLI also hard-codes its own cache and locks under HOME.
    managed = Path.home() / ".cache/hyperframes"
    managed.mkdir(parents=True, exist_ok=True)
    (managed / ".chrome.install.lock").touch()
    browser = managed / "chrome"
    browser.touch()
print(f"Path: {{browser}}")
''')
    binaries = owner / "bin"
    binaries.mkdir()
    npm = binaries / "npm"
    npm.write_text(f'''#!{sys.executable}
import os
import shutil
import sys
from pathlib import Path
assert sys.argv[1:] == ["ci", "--no-audit", "--no-fund", "--ignore-scripts"]
assert os.environ["HYPERFRAMES_NO_TELEMETRY"] == "1"
cache = Path(os.environ.get("npm_config_cache", str(Path.home() / ".npm")))
cache.mkdir(parents=True, exist_ok=True)
(cache / {fake!r}).touch()
assert Path(os.environ["npm_config_userconfig"]).read_bytes() == b""
assert os.environ["npm_config_update_notifier"] == "false"
logs = Path(os.environ["npm_config_logs_dir"])
logs.mkdir(parents=True)
(logs / "install.log").touch()
program = Path.cwd() / "node_modules/.bin/hyperframes"
program.parent.mkdir(parents=True)
shutil.copyfile({str(browser_script)!r}, program)
program.chmod(0o700)
''')
    node = binaries / "node"
    node.write_text(f'''#!{sys.executable}
import os
import sys
from pathlib import Path
assert sys.argv[1:] == ["--version"]
if os.environ.get("HYPERFRAMES_NO_TELEMETRY") is not None:
    assert os.environ["HYPERFRAMES_NO_TELEMETRY"] == "1"
    cache = Path(os.environ["NODE_COMPILE_CACHE"])
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "compiled").touch()
print("v22.23.1")
''')
    for program in (npm, node):
        program.chmod(0o700)
    monkeypatch.setenv("PATH", str(binaries) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(tools, "NODE_INSTALL_PLACES", ())
    monkeypatch.delenv("HYPERFRAMES_NO_TELEMETRY", raising=False)
    tools._find_node.cache_clear()
    run = subprocess.run
    node_calls = []

    def record_probe(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if args == [str(node), "--version"]:
            node_calls.append((kwargs.get("env") or {}).get("HYPERFRAMES_NO_TELEMETRY"))
        return run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", record_probe)
    monkeypatch.delenv("HYPERFRAMES_BROWSER_PATH", raising=False)
    if reuse_browser:
        browser = owner / "existing-browser"
        browser.touch()
        monkeypatch.setenv("HYPERFRAMES_BROWSER_PATH", str(browser))

    result = hyperframes.ensure_installed(timeout_s=17)

    assert result == hyperframes.installed()
    assert result.node_version == "v22.23.1"
    assert tools.find_tool("node") == str(node)
    assert node_calls == [None]
    assert result.prefix == owner / "cache/renderer" / f"hyperframes-{fake}"
    assert (_npm_cache / fake).is_file()
    assert (owner / "npm-logs/install.log").is_file()
    assert not (owner / "cache/node").exists()
    if reuse_browser:
        assert result.browser_path == str(owner / "existing-browser")
        assert list((owner / "browser-home").iterdir()) == []
    else:
        assert (owner / "cache/puppeteer/browser-download").is_file()
        assert result.browser_path == str(owner / "browser-home/.cache/hyperframes/chrome")
        assert (owner / "browser-home/.cache/hyperframes/.chrome.install.lock").is_file()
    assert list(Path.home().iterdir()) == []


def test_linked_renderer_is_published_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = tmp_path / "shared/hyperframes-test"
    (shared / "node_modules").mkdir(parents=True)
    (shared / "install.json").write_text('{"private": true}\n')
    renderer = tmp_path / "renderer"
    renderer.mkdir()
    prefix = renderer / shared.name
    copy = shutil.copyfile
    replace = os.replace
    published = []

    def copy_metadata(source: Path, target: Path) -> None:
        assert not prefix.exists()
        copy(source, target)
        assert not prefix.exists(), "metadata copy exposed an incomplete prefix"

    def publish(source: Path, target: Path) -> None:
        assert target == prefix and not prefix.exists()
        assert source.parent == renderer
        assert (source / "install.json").read_bytes() == (shared / "install.json").read_bytes()
        assert not (source / "install.json").is_symlink()
        assert (source / "node_modules").is_symlink()
        replace(source, target)
        published.append(target)

    monkeypatch.setattr(shutil, "copyfile", copy_metadata)
    monkeypatch.setattr(os, "replace", publish)
    _link_renderer_install(shared, renderer)
    assert published == [prefix]
    assert list(renderer.iterdir()) == [prefix]


def test_subprocess_cache_writes_stay_in_owner_folder(
    _owner_folders: Path, _shared_cache_root: Path,
) -> None:
    name = f"guard-{uuid4().hex}.txt"
    directories = ("media-facts", "media-inventory", "sfx")
    for directory in directories:
        assert not (_shared_cache_root / directory / name).exists()
    subprocess.run(
        [sys.executable, "-c", """
import sys
from yap_studio.cache import cache_dir
for directory in ('media-facts', 'media-inventory', 'sfx'):
    (cache_dir(directory) / sys.argv[1]).write_text('synthetic cache data')
""", name],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=True, timeout=30,
    )
    for directory in directories:
        assert (_owner_folders / "cache" / directory / name).read_text() == "synthetic cache data"
        assert not (_shared_cache_root / directory / name).exists()


@pytest.fixture
def custom_owner_folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Override the autouse defaults from a normal test fixture."""
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "custom-data"))
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "custom-config"))
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path / "custom-cache"))
    return tmp_path


def test_test_fixtures_can_override_owner_folders(custom_owner_folders: Path) -> None:
    assert data_dir() == custom_owner_folders / "custom-data"
    assert config_root() == custom_owner_folders / "custom-config"
    assert cache.cache_root() == custom_owner_folders / "custom-cache"


def test_home_leak_fails_with_test_and_path(pytester: pytest.Pytester) -> None:
    tests = Path(__file__).parent
    pytester.makeconftest((tests / "conftest.py").read_text(encoding="utf-8"))
    pytester.makepyfile(synth=(tests / "synth.py").read_text(encoding="utf-8"))
    pytester.makepyfile(test_home_leak="""
        from pathlib import Path

        def test_writes_home():
            folder = Path.home() / "unexpected"
            folder.mkdir()
            (folder / "leaked.txt").write_text("synthetic leak")
    """)
    result = pytester.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={pytester.path / 'run'}",
    )
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines([
        "*test_home_leak.py::test_writes_home wrote to its fake HOME:*",
        "*unexpected/leaked.txt*",
    ])


def _copy_owner_conftest(pytester: pytest.Pytester, home: Path) -> None:
    tests = Path(__file__).parent
    text = (tests / "conftest.py").read_text(encoding="utf-8")
    pytester.makeconftest(text + f"\n\ndef _account_home():\n    return Path({str(home)!r})\n")
    pytester.makepyfile(synth=(tests / "synth.py").read_text(encoding="utf-8"))


def _owner_session_summary(output: str) -> str:
    title = "owner's real Yap Studio folders changed during this run"
    summary = re.search(
        rf"^=+ {re.escape(title)} =+\n(.*?)(?=^=+ |\Z)", output, re.MULTILINE | re.DOTALL,
    )
    return summary[1] if summary else ""


@pytest.mark.parametrize("following_summary", [False, True])
@pytest.mark.parametrize("reported_target", [False, True])
def test_owner_session_summary_excludes_error_recap(
    tmp_path: Path, following_summary: bool, reported_target: bool,
) -> None:
    target = tmp_path / "account/Library/Caches/yap-studio/scratch"
    nodeid = "test_relative_open.py::test_write"
    entries = [("changed", str(target.parent), 123, nodeid)]
    if reported_target:
        entries.append(("added", str(target), 123, nodeid))
    summary = "\n".join(conftest._owner_folder_summary(entries)) + "\n"
    output = "=== owner's real Yap Studio folders changed during this run ===\n" + summary
    if following_summary:
        message = conftest._owner_write_message(nodeid, {("open", str(target))})
        output += ("=== short test summary info ===\n"
                   f"ERROR {nodeid} - Failed: {message}\n")
    assert _owner_session_summary(output) == summary
    assert (str(target) in _owner_session_summary(output)) is reported_target


def test_owner_session_summary_absent() -> None:
    assert _owner_session_summary("=== short test summary info ===\n1 passed\n") == ""


@pytest.mark.parametrize(("scope", "when", "marker"), [
    ("function", "body", ""),
    ("session", "setup", ""),
    ("module", "setup", ""),
    ("class", "setup", ""),
    ("module", "teardown", ""),
    ("session", "teardown", ""),
    ("function", "body", "render"),
    ("function", "body", "asr"),
    ("function", "both", ""),
])
def test_real_folder_leak_fails_with_test_and_path(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch,
    scope: str, when: str, marker: str,
) -> None:
    _copy_owner_conftest(pytester, pytester.path / "account")
    if when == "both":
        with (pytester.path / "conftest.py").open("a", encoding="utf-8") as output:
            output.write('''
def _check_detached_processes(nodeid, roots, data):
    pytest.fail(f"{nodeid} left detached processes running: synthetic leftover")
''')
    monkeypatch.setenv("YAP_STUDIO_RENDER_TESTS", "1")
    monkeypatch.setenv("YAP_STUDIO_ASR_TESTS", "1")
    pytester.makepyfile(test_owner_leak=f'''
        from pathlib import Path
        import pytest
        from conftest import _owner_roots

        def leak():
            folder = _owner_roots()[0]
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / "leaked.txt"
            target.write_text("synthetic leak")
            Path({str(pytester.path / 'observed')!r}).write_text(str(target))

        @pytest.fixture(scope={scope!r}, autouse=True)
        def writer():
            if {when!r} == "setup":
                leak()
            yield
            if {when!r} == "teardown":
                leak()

        {f"@pytest.mark.{marker}" if marker else ""}
        def test_writes_owner():
            if {when!r} in ("body", "both"):
                leak()
    ''')
    result = pytester.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={pytester.path / 'run'}",
    )
    result.assert_outcomes(passed=1, errors=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    output = result.stdout.str()
    assert ("test_owner_leak.py::test_writes_owner wrote to the owner's real "
            "Yap Studio folders:") in output
    target = (pytester.path / "observed").read_text()
    assert f"open {target}" in output
    assert "owner's real Yap Studio folders changed during this run" not in output
    if when == "both":
        assert "test_owner_leak.py::test_writes_owner left detached processes running:" in output


def test_module_fixtures_keep_owner_folders_isolated(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = pytester.path / "account"
    _copy_owner_conftest(pytester, home)
    for name in ("YAP_STUDIO_CACHE", "YAP_STUDIO_CONFIG", "YAP_STUDIO_DATA_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))
    pytester.makepyfile(test_module_owner=f'''
        import os
        from pathlib import Path
        import pytest
        from yap_studio import cache, sfx
        from yap_studio.review.store import data_dir
        from yap_studio.style.library import config_root

        @pytest.fixture(scope="module")
        def stores():
            sfx.remember_account("unknown")
            cache.cache_dir("media-facts")
            return (cache.cache_root(), config_root(), data_dir()), {{
                name: os.environ.get(name) for name in (
                    "ELEVENLABS_CONFIG", "ELEVENLABS_API_KEY", "ELEVENLABS_BASE_URL",
                )
            }}

        def test_isolated(stores):
            roots, env = stores
            for root in roots:
                assert not root.is_relative_to(Path({str(home)!r})), root
            assert (roots[0] / "sfx/account.json").is_file()
            assert (roots[0] / "media-facts").is_dir()
            from conftest import _owner_roots
            assert all(not root.exists() for root in _owner_roots())
            owner = roots[0].parent
            assert roots == (owner / "cache", owner / "config", owner / "data")
            assert (owner / "cache/helpers").is_symlink()
            assert (owner / "cache/renderer").is_dir()
            assert Path(env["ELEVENLABS_CONFIG"]).is_relative_to(owner)
            assert env["ELEVENLABS_API_KEY"] is None
            assert env["ELEVENLABS_BASE_URL"] == "http://127.0.0.1:9"
    ''')
    result = pytester.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={pytester.path / 'run'}",
    )
    result.assert_outcomes(passed=1)
    assert result.ret == pytest.ExitCode.OK


def test_real_folder_guard_leaves_existing_file_unchanged(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = pytester.path / "account"
    _copy_owner_conftest(pytester, home)
    with monkeypatch.context() as patch:
        patch.setattr(conftest, "_account_home", lambda: home)
        roots = conftest._owner_roots()
    target = roots[0] / "existing.txt"
    target.parent.mkdir(parents=True)
    target.write_text("synthetic existing file")
    mtime = 1_700_000_000_000_000_000
    os.utime(target, ns=(mtime, mtime))
    pytester.makepyfile(test_clean="def test_clean(): pass")
    result = pytester.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={pytester.path / 'run'}",
    )
    result.assert_outcomes(passed=1)
    assert result.ret == pytest.ExitCode.OK
    assert target.stat().st_mtime_ns == mtime
    assert target.read_text() == "synthetic existing file"
    assert all(not root.exists() for root in roots[1:])
    assert "owner's real Yap Studio folders changed during this run" not in result.stdout.str()


@pytest.mark.parametrize("case", ["outside", "added", "changed", "node_modules"])
def test_relative_os_open_requires_state_confirmation(
    pytester: pytest.Pytester, case: str,
) -> None:
    home = pytester.path / "account"
    _copy_owner_conftest(pytester, home)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(conftest, "_account_home", lambda: home)
        root = conftest._owner_roots()[0]
    root.mkdir(parents=True)
    relative = (
        Path("renderer/node_modules/existing.js") if case == "node_modules" else Path("scratch")
    )
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if case in ("changed", "node_modules"):
        target.write_text("synthetic existing data")
    if case == "added":
        # Ensure creating scratch changes its parent's metadata on every filesystem.
        os.utime(root, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    before = conftest._snapshot_owner_folders((root,))
    pytester.makepyfile(test_relative_open=f'''
        import os

        def test_write(monkeypatch):
            monkeypatch.chdir({str(root)!r})
            directory = (
                os.open({str(pytester.path)!r}, os.O_RDONLY) if {case == 'outside'} else None
            )
            try:
                options = {{"dir_fd": directory}} if directory is not None else {{}}
                descriptor = os.open(
                    {str(relative)!r}, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, **options,
                )
                try:
                    os.write(descriptor, b"after")
                finally:
                    os.close(descriptor)
            finally:
                if directory is not None:
                    os.close(directory)
    ''')
    result = pytester.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={pytester.path / 'run'}",
    )
    output = result.stdout.str()
    title = "owner's real Yap Studio folders changed during this run"
    if case == "outside":
        result.assert_outcomes(passed=1)
        assert result.ret == pytest.ExitCode.OK
        assert conftest._snapshot_owner_folders((root,)) == before
        assert not target.exists()
        assert (pytester.path / "scratch").read_bytes() == b"after"
        assert title not in output
    else:
        result.assert_outcomes(passed=1, errors=1)
        assert result.ret == pytest.ExitCode.TESTS_FAILED
        assert ("test_relative_open.py::test_write wrote to the owner's real "
                "Yap Studio folders:") in output
        assert f"open {target}" in output
        assert target.read_bytes() == b"after"
        if case == "node_modules":
            assert conftest._snapshot_owner_folders((root,)) == before
        summary = _owner_session_summary(output)
        assert str(target) not in summary
        if case == "added":
            assert f"changed {root} (mtime " in summary


@pytest.mark.parametrize("workers", [0, 2])
def test_child_folder_write_warns_without_failing(pytester: pytest.Pytester, workers: int) -> None:
    home = pytester.path / "account"
    _copy_owner_conftest(pytester, home)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(conftest, "_account_home", lambda: home)
        target = conftest._owner_roots()[0] / "child.txt"
    pytester.makepyfile(test_child=f'''
        import subprocess
        import sys

        def test_child_write():
            code = ("from pathlib import Path; import sys; target = Path(sys.argv[1]); "
                    "target.parent.mkdir(parents=True); target.write_text('synthetic child')")
            subprocess.run([sys.executable, "-c", code, {str(target)!r}], check=True)
    ''')
    result = pytester.runpytest_subprocess(
        "-q", "-n", str(workers), "-p", "no:cacheprovider",
        f"--basetemp={pytester.path / 'run'}",
    )
    result.assert_outcomes(passed=1)
    assert result.ret == pytest.ExitCode.OK
    output = result.stdout.str()
    assert output.count("owner's real Yap Studio folders changed during this run") == 1
    assert re.search(
        rf"added {re.escape(str(target))} \(mtime \d{{2}}:\d{{2}}:\d{{2}}\) "
        r"while running: test_child.py::test_child_write", output,
    )
    assert "Not a test failure; another process" in output


def test_owner_folder_guard_reuses_snapshot_after_changes(pytester: pytest.Pytester) -> None:
    _copy_owner_conftest(pytester, pytester.path / "account")
    observed = pytester.path / "snapshot-count"
    with (pytester.path / "conftest.py").open("a", encoding="utf-8") as output:
        output.write(f'''
_original_snapshot = _snapshot_owner_folders
_snapshot_count = 0

def _snapshot_owner_folders(roots):
    global _snapshot_count
    _snapshot_count += 1
    Path({str(observed)!r}).write_text(str(_snapshot_count))
    return _original_snapshot(roots)
''')
    pytester.makepyfile(test_snapshots='''
        import subprocess
        import sys
        from conftest import _owner_roots

        def test_writes_owner():
            root = _owner_roots()[0]
            code = ("from pathlib import Path; import sys; root = Path(sys.argv[1]); "
                    "root.mkdir(parents=True); (root / 'leaked.txt').write_text('synthetic leak')")
            subprocess.run([sys.executable, "-c", code, str(root)], check=True)

        def test_clean():
            pass
    ''')
    result = pytester.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={pytester.path / 'run'}",
    )
    result.assert_outcomes(passed=2)
    assert result.ret == pytest.ExitCode.OK
    assert observed.read_text() == "3"
    output = result.stdout.str()
    summary = output.split("owner's real Yap Studio folders changed during this run", 1)[1]
    assert "while running: test_snapshots.py::test_writes_owner" in summary
    assert "test_snapshots.py::test_clean" not in summary


@pytest.mark.parametrize("change", ["added", "removed", "changed"])
def test_owner_folder_snapshot_changes(tmp_path: Path, change: str) -> None:
    root = tmp_path / "owner"
    root.mkdir()
    target = root / "entry"
    if change != "added":
        target.write_text("before")
    before = conftest._snapshot_owner_folders((root,))
    if change == "removed":
        target.unlink()
    else:
        target.write_text("after")
        os.utime(target, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    root_mtime = 1_700_000_001_000_000_000
    os.utime(root, ns=(root_mtime, root_mtime))
    after = conftest._snapshot_owner_folders((root,))
    expected_mtime = (before if change == "removed" else after)[root][str(target)][1]
    assert conftest._owner_folder_changes(before, after) == [
        ("changed", root, root_mtime), (change, target, expected_mtime),
    ]
    snapshot = (before if change == "removed" else after)[root]
    assert snapshot[str(target)][0] == stat.S_IFREG
    assert snapshot[str(target)][2] == len("before" if change == "removed" else "after")


def test_owner_folder_snapshot_ignores_node_modules_and_symlink_targets(tmp_path: Path) -> None:
    root = tmp_path / "owner"
    modules = root / "node_modules"
    modules.mkdir(parents=True)
    package = modules / "package.js"
    package.write_text("before")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    before = conftest._snapshot_owner_folders((root,))
    assert set(before[root]) == {str(root), str(modules), str(root / "linked")}
    assert before[root][str(root / "linked")][0] == stat.S_IFLNK
    package.write_text("after")
    (outside / "ignored").touch()
    after = conftest._snapshot_owner_folders((root,))
    assert conftest._owner_folder_changes(before, after) == []
    os.utime(modules, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    after = conftest._snapshot_owner_folders((root,))
    assert conftest._owner_folder_changes(before, after) == [
        ("changed", modules, 1_700_000_000_000_000_000),
    ]


def test_owner_folder_snapshot_detects_missing_root_created(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    before = conftest._snapshot_owner_folders((root,))
    assert before == {root: None}
    root.mkdir()
    after = conftest._snapshot_owner_folders((root,))
    assert conftest._owner_folder_changes(before, after) == [
        ("added", root, root.stat().st_mtime_ns),
    ]


@pytest.mark.parametrize("entry", [(stat.S_IFREG, 123, 9), (stat.S_IFLNK, 123, 6)])
def test_owner_folder_diff_detects_size_and_kind_changes(
    tmp_path: Path, entry: tuple[int, int, int],
) -> None:
    root = tmp_path / "owner"
    target = root / "entry"
    before = {root: {str(target): (stat.S_IFREG, 123, 6)}}
    after = {root: {str(target): entry}}
    assert conftest._owner_folder_changes(before, after) == [("changed", target, 123)]


@pytest.mark.parametrize("operation", ["lstat", "scandir"])
def test_owner_folder_snapshot_skips_unreadable_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    root = tmp_path / "owner"
    root.mkdir()
    original = getattr(os, operation)

    def unreadable(path, *args, **kwargs):
        if os.fspath(path) == str(root):
            raise PermissionError("synthetic unreadable folder")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, operation, unreadable)
        snapshot = conftest._snapshot_owner_folders((root,))
    if operation == "lstat":
        assert snapshot == {root: None}
    else:
        assert set(snapshot[root]) == {str(root)}


@pytest.mark.parametrize("operation", ["stat", "is_dir", "iteration"])
def test_owner_folder_snapshot_skips_unreadable_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    root = tmp_path / "owner"
    root.mkdir()
    readable = root / "readable"
    readable.touch()
    with os.scandir(root) as children:
        readable_entry = next(children)

    class UnreadableEntry:
        name = "unreadable"
        path = str(root / name)

        def stat(self, *, follow_symlinks):
            assert follow_symlinks is False
            if operation == "stat":
                raise PermissionError("synthetic unreadable entry")
            return os.lstat(readable)

        def is_dir(self, *, follow_symlinks):
            assert follow_symlinks is False
            raise PermissionError("synthetic unreadable entry")

    def entries():
        if operation != "iteration":
            yield UnreadableEntry()
        yield readable_entry
        if operation == "iteration":
            raise PermissionError("synthetic unreadable directory")

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", lambda path: nullcontext(entries()))
        snapshot = conftest._snapshot_owner_folders((root,))
    assert set(snapshot[root]) == {str(root), str(readable)}


def test_owner_folder_summary_groups_tests_and_limits_lines(tmp_path: Path) -> None:
    mtime = 1_700_000_000_000_000_000
    target = str(tmp_path / "shared")
    entries = [("changed", target, mtime, f"tests/a.py::t{index}") for index in range(7)]
    entries.append(entries[0])
    lines = conftest._owner_folder_summary(entries)
    assert len(lines) == 2
    assert re.match(rf"changed {re.escape(target)} \(mtime \d{{2}}:\d{{2}}:\d{{2}}\)", lines[0])
    assert lines[0].endswith(
        "while running: tests/a.py::t0, tests/a.py::t1, tests/a.py::t2, "
        "tests/a.py::t3, tests/a.py::t4, +2 more"
    )
    entries.extend(("changed", str(tmp_path / str(index)), mtime, "tests/b.py::t")
                   for index in range(32))
    lines = conftest._owner_folder_summary(entries)
    assert len(lines) == 32
    assert lines[-2] == "... and 3 more"
    assert "Not a test failure; another process" in lines[-1]


def test_owner_folder_reports_collect_json_only_from_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = []
    monkeypatch.setattr(conftest, "_OWNER_FOLDER_CHANGES", entries)
    target = str(tmp_path / "changed")
    report = SimpleNamespace(
        when="call", nodeid="tests/a.py::t", user_properties=[
            ("unrelated", "ignored"),
            ("yap_owner_folder_changes", json.dumps([
                {"kind": "changed", "path": target, "mtime_ns": 123},
            ])),
        ],
    )
    conftest.pytest_runtest_logreport(report)
    assert entries == []
    report.when = "teardown"
    conftest.pytest_runtest_logreport(report)
    assert entries == [("changed", target, 123, "tests/a.py::t")]


def test_owner_write_message_limits_events(tmp_path: Path) -> None:
    writes = {("open", str(tmp_path / str(index))) for index in range(23)}
    lines = conftest._owner_write_message("tests/a.py::t", writes).splitlines()
    assert lines[0] == "tests/a.py::t wrote to the owner's real Yap Studio folders:"
    assert len(lines) == 23
    assert all(line.startswith("open ") for line in lines[1:21])
    assert lines[-2] == "... and 3 more"
    assert lines[-1] == "Tests must use the folders conftest sets."


@pytest.fixture
def owner_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "guarded"
    root.mkdir()
    writes = {}
    with monkeypatch.context() as patch:
        patch.setattr(conftest, "_OWNER_AUDIT_ROOTS", (str(root.resolve()),))
        patch.setattr(conftest, "_OWNER_CURRENT_NODEID", "test_audit.py::test_write")
        patch.setattr(conftest, "_OWNER_AUDIT_WRITES", writes)
        patch.setattr(conftest, "_OWNER_AUDIT_CANDIDATES", {})
        yield root, writes


@pytest.mark.parametrize("case", [
    "open_mode", "open_flags", "open_bytes", "rename", "replace", "mkdir", "remove",
    "rmdir", "symlink", "link", "truncate", "utime", "chmod", "rmtree", "move",
])
def test_owner_audit_records_write_events(owner_audit, tmp_path: Path, case: str) -> None:
    root, writes = owner_audit
    target = str(root / "written")
    outside = str(tmp_path / "outside")
    event, args = {
        "open_mode": ("open", (target, "w", 0)),
        "open_flags": ("open", (target, None, os.O_CREAT | os.O_WRONLY)),
        "open_bytes": ("open", (os.fsencode(target), "a", 0)),
        "rename": ("os.rename", (target, outside, -1, -1)),
        "replace": ("os.replace", (outside, target, -1, -1)),
        "mkdir": ("os.mkdir", (target, 0o700, -1)),
        "remove": ("os.remove", (target, -1)),
        "rmdir": ("os.rmdir", (target, -1)),
        "symlink": ("os.symlink", (outside, target, -1)),
        "link": ("os.link", (outside, target, -1, -1)),
        "truncate": ("os.truncate", (target, 0)),
        "utime": ("os.utime", (target, None, None, -1)),
        "chmod": ("os.chmod", (target, 0o700, -1)),
        "rmtree": ("shutil.rmtree", (target, None)),
        "move": ("shutil.move", (outside, target)),
    }[case]
    conftest._audit_owner_write(event, args)
    assert writes == {"test_audit.py::test_write": {(event, target)}}


@pytest.mark.parametrize("case", [
    "read", "read_flags", "outside", "prefix", "fd", "dir_fd", "relative_open", "event",
    "link_source", "move_source",
])
def test_owner_audit_ignores_other_events(owner_audit, tmp_path: Path, case: str) -> None:
    root, writes = owner_audit
    target = str(root / "ignored")
    event, args = {
        "read": ("open", (target, "r", 0)),
        "read_flags": ("open", (target, None, os.O_RDONLY)),
        "outside": ("open", (str(tmp_path / "outside"), "w", 0)),
        "prefix": ("open", (str(root) + "-outside", "w", 0)),
        "fd": ("open", (3, "w", 0)),
        "dir_fd": ("os.mkdir", ("relative", 0o700, 42)),
        "relative_open": ("open", ("relative", None, os.O_CREAT | os.O_WRONLY)),
        "event": ("os.listdir", (str(root),)),
        "link_source": ("os.link", (target, str(tmp_path / "outside"), -1, -1)),
        "move_source": ("shutil.move", (target, str(tmp_path / "outside"))),
    }[case]
    conftest._audit_owner_write(event, args)
    assert writes == {}


@pytest.mark.parametrize("leaf", [False, True])
def test_owner_audit_resolves_symlinks(owner_audit, tmp_path: Path, leaf: bool) -> None:
    root, writes = owner_audit
    link = tmp_path / "link"
    if leaf:
        link.symlink_to(root / "written")
    else:
        link.symlink_to(root, target_is_directory=True)
    target = link if leaf else link / "written"
    conftest._audit_owner_write("open", (str(target), "w", 0))
    assert writes == {"test_audit.py::test_write": {("open", str(root / "written"))}}


@pytest.mark.parametrize("event", ["os.remove", "shutil.rmtree", "os.replace"])
def test_owner_audit_ignores_changes_to_outside_symlinks(
    owner_audit, tmp_path: Path, event: str,
) -> None:
    root, writes = owner_audit
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    args = {
        "os.remove": (str(link), -1),
        "shutil.rmtree": (str(link), None),
        "os.replace": (str(tmp_path / "replacement"), str(link), -1, -1),
    }[event]
    conftest._audit_owner_write(event, args)
    assert writes == {}


@pytest.mark.parametrize("kind", ["file", "link"])
def test_owner_audit_records_removing_guarded_entries(
    owner_audit, tmp_path: Path, kind: str,
) -> None:
    root, writes = owner_audit
    target = root / "written"
    if kind == "file":
        target.touch()
    else:
        target.symlink_to(tmp_path / "outside")
    writes.clear()
    conftest._audit_owner_write("os.remove", (str(target), -1))
    assert writes == {"test_audit.py::test_write": {("os.remove", str(target))}}


def test_owner_audit_resolves_relative_open_paths(owner_audit, monkeypatch) -> None:
    root, writes = owner_audit
    with monkeypatch.context() as patch:
        patch.chdir(root)
        conftest._audit_owner_write("open", ("written", "w", 0))
    assert writes == {"test_audit.py::test_write": {("open", str(root / "written"))}}


@pytest.mark.parametrize("case", ["guarded_write", "outside_write", "guarded_read"])
def test_owner_audit_resolves_relative_os_open(
    owner_audit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    root, writes = owner_audit
    folder = tmp_path if case == "outside_write" else root
    target = folder / "written"
    target.touch()
    writes.clear()
    info = os.lstat(target)
    state = (stat.S_IFMT(info.st_mode), info.st_mtime_ns, info.st_size)
    flags = os.O_RDONLY if case == "guarded_read" else os.O_WRONLY | os.O_TRUNC
    with monkeypatch.context() as patch:
        patch.chdir(folder)
        descriptor = os.open(target.name, flags)
        try:
            expected = ({"test_audit.py::test_write": {("open", str(target)): state}}
                        if case == "guarded_write" else {})
            assert writes == {}
            assert conftest._OWNER_AUDIT_CANDIDATES == expected
        finally:
            os.close(descriptor)


@pytest.mark.parametrize("exists", [False, True])
def test_owner_audit_drops_relative_open_candidate_without_change(
    owner_audit, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exists: bool,
) -> None:
    root, writes = owner_audit
    target = root / "scratch"
    state = None
    if exists:
        target.write_text("synthetic existing data")
        info = os.lstat(target)
        state = (stat.S_IFMT(info.st_mode), info.st_mtime_ns, info.st_size)
    writes.clear()
    before = conftest._snapshot_owner_folders((root,))
    with monkeypatch.context() as patch:
        patch.chdir(root)
        directory = os.open(tmp_path, os.O_RDONLY)
        try:
            descriptor = os.open("scratch", os.O_CREAT | os.O_WRONLY, dir_fd=directory)
            os.close(descriptor)
        finally:
            os.close(directory)
    assert writes == {}
    assert conftest._OWNER_AUDIT_CANDIDATES == {
        "test_audit.py::test_write": {("open", str(target)): state},
    }
    after = conftest._snapshot_owner_folders((root,))
    assert after == before
    assert conftest._promote_owner_candidates("test_audit.py::test_write") == set()
    assert writes == {}
    assert conftest._OWNER_AUDIT_CANDIDATES == {}


@pytest.mark.parametrize(("before", "after", "confirmed"), [
    (None, (stat.S_IFREG, 123, 6), True),
    ((stat.S_IFREG, 123, 6), None, True),
    ((stat.S_IFREG, 123, 6), (stat.S_IFLNK, 123, 6), True),
    ((stat.S_IFREG, 123, 6), (stat.S_IFREG, 124, 6), True),
    ((stat.S_IFREG, 123, 6), (stat.S_IFREG, 123, 9), True),
    ((stat.S_IFREG, 123, 6), (stat.S_IFREG, 123, 6), False),
    (None, None, False),
])
def test_owner_audit_confirms_candidate_state(
    owner_audit, monkeypatch: pytest.MonkeyPatch, before, after, confirmed: bool,
) -> None:
    root, writes = owner_audit
    target = str(root / "written")
    monkeypatch.setattr(conftest, "_owner_path_state", lambda path: after)
    conftest._OWNER_AUDIT_CANDIDATES["test_audit.py::test_write"] = {("open", target): before}
    promoted = conftest._promote_owner_candidates("test_audit.py::test_write")
    assert promoted == ({("open", target)} if confirmed else set())
    assert conftest._OWNER_AUDIT_CANDIDATES == {}
    assert writes == {}


@pytest.mark.parametrize("case", ["created", "node_modules"])
def test_owner_audit_promotes_relative_open_state(
    owner_audit, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    root, writes = owner_audit
    relative = (
        Path("renderer/node_modules/existing.js") if case == "node_modules" else Path("scratch")
    )
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    state = None
    if case == "node_modules":
        target.write_text("synthetic existing data")
        info = os.lstat(target)
        state = (stat.S_IFMT(info.st_mode), info.st_mtime_ns, info.st_size)
    writes.clear()
    before = conftest._snapshot_owner_folders((root,))
    with monkeypatch.context() as patch:
        patch.chdir(root)
        for _ in range(2):
            descriptor = os.open(str(relative), os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
            try:
                assert conftest._OWNER_AUDIT_CANDIDATES == {
                    "test_audit.py::test_write": {("open", str(target)): state},
                }
            finally:
                os.close(descriptor)
    if case == "node_modules":
        assert conftest._snapshot_owner_folders((root,)) == before
    promoted = conftest._promote_owner_candidates("test_audit.py::test_write")
    assert promoted == {("open", str(target))}
    assert conftest._OWNER_AUDIT_CANDIDATES == {}
    assert writes == {}


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError, OSError])
def test_owner_audit_candidate_read_errors_use_missing_state(
    owner_audit, monkeypatch: pytest.MonkeyPatch, error,
) -> None:
    root, writes = owner_audit
    target = str(root / "written")
    original = os.lstat

    def lstat(path, *args, **kwargs):
        if os.fspath(path) == target:
            raise error("synthetic stat error")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.chdir(root)
        patch.setattr(os, "lstat", lstat)
        conftest._audit_owner_write("open", ("written", None, os.O_WRONLY))
    assert conftest._OWNER_AUDIT_CANDIDATES == {
        "test_audit.py::test_write": {("open", target): None},
    }
    assert writes == {}


@pytest.mark.parametrize("event", ["os.remove", "os.rename"])
@pytest.mark.parametrize("guarded", [False, True])
def test_owner_audit_resolves_descriptor_relative_paths(
    owner_audit, tmp_path: Path, event: str, guarded: bool,
) -> None:
    root, writes = owner_audit
    folder = root if guarded else tmp_path
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        args = (("written", descriptor) if event == "os.remove" else
                (str(tmp_path / "source"), "written", -1, descriptor))
        conftest._audit_owner_write(event, args)
        expected = ({"test_audit.py::test_write": {(event, str(folder / "written"))}}
                    if guarded else {})
        assert writes == expected
    finally:
        os.close(descriptor)


def test_owner_audit_resolves_descriptors_through_proc(owner_audit, monkeypatch) -> None:
    root, writes = owner_audit
    descriptor = os.open(root, os.O_RDONLY)
    try:
        proc_path = f"/proc/self/fd/{descriptor}"
        original = os.readlink
        seen = []

        def readlink(path, *args, **kwargs):
            if path == proc_path:
                seen.append(path)
                return str(root)
            return original(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.delattr(conftest.fcntl, "F_GETPATH", raising=False)
            patch.setattr(os, "readlink", readlink)
            conftest._audit_owner_write("os.remove", ("written", descriptor))
        assert seen == [proc_path]
        assert writes == {"test_audit.py::test_write": {("os.remove", str(root / "written"))}}
    finally:
        os.close(descriptor)


def test_owner_audit_ignores_temporary_tree_cleanup(owner_audit, tmp_path: Path) -> None:
    root, writes = owner_audit
    tree = tmp_path / "tree"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested/file").write_text("synthetic data")
    (tree / "shared").symlink_to(root, target_is_directory=True)
    writes.clear()
    shutil.rmtree(tree)
    assert writes == {}
    assert not tree.exists()
    assert root.is_dir()


def test_owner_audit_ignores_writes_without_current_test(owner_audit, monkeypatch) -> None:
    root, writes = owner_audit
    with monkeypatch.context() as patch:
        patch.setattr(conftest, "_OWNER_CURRENT_NODEID", None)
        conftest._audit_owner_write("open", (str(root / "ignored"), "w", 0))
    assert writes == {}


def test_shared_renderer_recovers_after_invalid_session_snapshot(pytester: pytest.Pytester) -> None:
    tests = Path(__file__).parent
    pytester.makeconftest((tests / "conftest.py").read_text())
    pytester.makepyfile(synth=(tests / "synth.py").read_text())
    pytester.makepyfile(test_recovery='''
        import json
        import pytest
        from yap_studio.render import hyperframes

        @pytest.fixture(scope="session")
        def _shared_cache_root(tmp_path_factory):
            root = tmp_path_factory.mktemp("shared")
            prefix = root / "renderer" / f"hyperframes-{hyperframes.HYPERFRAMES_VERSION}"
            executable = prefix / "node_modules/.bin/hyperframes"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\\nexit 0\\n")
            executable.chmod(0o700)
            (prefix / "install.json").write_text(json.dumps({
                "version": hyperframes.HYPERFRAMES_VERSION,
                "node_version": "v22.23.1",
                "lock_sha256": hyperframes._lock_sha256(),
                "browser_path": str(root / "browser"),
            }))
            return root

        @pytest.fixture(scope="session")
        def _shared_renderer_install(_shared_renderer_install, _shared_cache_root):
            # Re-extraction finishes just after the worker took its session snapshot.
            assert _shared_renderer_install is None
            (_shared_cache_root / "browser").touch()
            return None

        @pytest.mark.parametrize("index", range(3))
        def test_recovered(index, _shared_renderer_install, _owner_folders):
            assert _shared_renderer_install is None
            install = hyperframes.installed()
            assert install is not None, "session snapshot stayed invalid after browser repair"
            assert install.prefix.is_relative_to(_owner_folders)
    ''')
    result = pytester.runpytest_subprocess(
        "-q", "-p", "no:cacheprovider", f"--basetemp={pytester.path / 'run'}",
    )
    result.assert_outcomes(passed=3)


@pytest.mark.render
def test_shared_renderer_links_under_parallel_load(
    pytester: pytest.Pytester, _shared_cache_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _shared_cache_root / "renderer" / f"hyperframes-{hyperframes.HYPERFRAMES_VERSION}"
    failure = _renderer_install_failure(shared)
    if failure is not None:
        pytest.skip(f"the real pinned renderer install is not complete: {failure}")
    tests = Path(__file__).parent
    pytester.makeconftest((tests / "conftest.py").read_text())
    pytester.makepyfile(synth=(tests / "synth.py").read_text())
    observed = pytester.path / "observed"
    observed.mkdir()
    pytester.makepyfile(test_links=f'''
        from pathlib import Path
        import pytest
        from yap_studio.render import hyperframes
        from conftest import _renderer_install_failure

        @pytest.mark.render
        @pytest.mark.parametrize("index", range(40))
        def test_link(index, worker_id, _owner_folders, _shared_renderer_install):
            prefix = hyperframes.install_prefix()
            session = ("no shared install at session start; "
                       if _shared_renderer_install is None else "")
            install = hyperframes.installed()
            assert install is not None, session + str(_renderer_install_failure(prefix))
            assert prefix.is_dir() and not prefix.is_symlink()
            metadata = prefix / "install.json"
            assert metadata.is_file() and not metadata.is_symlink()
            assert not metadata.samefile(Path({str(shared)!r}) / "install.json")
            assert prefix.is_relative_to(_owner_folders)
            (Path({str(observed)!r}) / f"{{worker_id}}-{{index}}").write_text(str(prefix))
    ''')
    monkeypatch.setenv("YAP_STUDIO_RENDER_TESTS", "1")
    # Nested workers must snapshot the actual shared cache before isolating their
    # own stores, rather than inheriting the outer test's private cache prefix.
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(_shared_cache_root))
    result = pytester.runpytest_subprocess(
        "-n", "4", "--dist=load", "-q", "-p", "no:cacheprovider", "-m", "render",
        f"--basetemp={pytester.path / 'run'}", timeout=120,
    )
    result.assert_outcomes(passed=40)
    observations = list(observed.iterdir())
    assert len(observations) == 40
    assert {path.name.split("-")[0] for path in observations} == {f"gw{i}" for i in range(4)}
    assert len({path.read_text() for path in observations}) == 40
