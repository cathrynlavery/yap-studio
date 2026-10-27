"""In-place workspace updates use synthetic releases and leave creator files alone."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_workspace import ROOT, _module

VERSION = "9.9.9"
WHEEL = f"wheels/yap_studio-{VERSION}-py3-none-any.whl"
MANIFEST = ".yap-studio-managed.json"


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def release(tmp_path, monkeypatch):
    module = _module(ROOT / "workspace/setup.py", "workspace_update")
    root = tmp_path / "workspace"
    root.mkdir()
    before = {
        ".yap-studio-version": b"1.1.1\n",
        "setup.py": b"old setup\n",
        "AGENTS.md": b"old instructions\n",
        "inbox/README.md": b"old inbox instructions\n",
        "projects/synthetic/edit.json": b'{"synthetic": true}\n',
        "inbox/take.mp4": b"synthetic take",
        "brand/logo.png": b"synthetic logo",
        "style/answers.json": b"{}\n",
        "creator.txt": b"keep this\n",
        "wheels/yap_studio-1.1.1-py3-none-any.whl": b"old wheel",
    }
    for name, content in before.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"yap_studio-{VERSION}.dist-info/RECORD",
                         "yap_studio/__init__.py,sha256=synthetic,12\n")
    managed = {
        ".yap-studio-version": f"{VERSION}\n".encode(),
        "setup.py": b"new setup\n",
        "AGENTS.md": b"new instructions\n",
        "inbox/README.md": b"new inbox instructions\n",
        WHEEL: wheel.getvalue(),
    }
    managed[MANIFEST] = json.dumps(sorted([*managed, MANIFEST])).encode()
    data = SimpleNamespace(module=module, root=root, before=before, managed=managed,
                           extra={}, installs=[], doctors=[], requests=[])

    def http(url, limit):
        data.requests.append(url)
        if url.endswith("/releases/latest"):
            return json.dumps(data.api).encode()
        assert url == data.asset["browser_download_url"]
        assert len(data.zip) <= limit
        return data.zip

    def install(uv, wheel, engine, **kwargs):
        assert uv == "/synthetic/uv"
        assert wheel == root / WHEEL
        assert wheel.read_bytes() == managed[WHEEL]
        data.installs.append(wheel)
        return None

    def doctor(engine):
        data.doctors.append(engine)
        report = {"install": {"version": VERSION, "build": module._wheel_build(root / WHEEL)},
                  "ready_to_edit": True}
        return subprocess.CompletedProcess([], 0, json.dumps(report), ""), report

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "_update_http", http, raising=False)
    monkeypatch.setattr(module, "_uv_binary", lambda: "/synthetic/uv")
    monkeypatch.setattr(module, "_tool_bin_dir", lambda uv: tmp_path / "bin")
    monkeypatch.setattr(module, "_install_wheel", install)
    monkeypatch.setattr(module, "_doctor", doctor)
    _package(data)
    return data


def _package(release):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        for name, value in (release.managed | release.extra).items():
            archive.writestr(name, value)
    release.zip = content.getvalue()
    release.asset = {
        "name": f"yap-studio-workspace-{VERSION}.zip",
        "browser_download_url": f"https://example.invalid/workspace-{VERSION}.zip",
        "digest": f"sha256:{hashlib.sha256(release.zip).hexdigest()}",
    }
    release.api = {"tag_name": f"v{VERSION}", "assets": [release.asset]}


@pytest.mark.parametrize("installed", [VERSION, "10.0.0"])
def test_update_already_current(release, capsys, installed):
    (release.root / ".yap-studio-version").write_text(installed + "\n")
    before = _snapshot(release.root)
    assert release.module.main(["--update"]) == 0
    assert "already up to date" in capsys.readouterr().out.lower()
    assert _snapshot(release.root) == before
    assert len(release.requests) == 1
    assert release.installs == release.doctors == []


def test_update_replaces_only_managed_files_and_installs_new_wheel(release, capsys):
    release.extra["creator.txt"] = b"not managed"
    release.extra["projects/synthetic/edit.json"] = b"not managed either"
    _package(release)
    assert release.module.main(["--update"]) == 0
    backups = list((release.root / ".yap-studio/backups").iterdir())
    assert len(backups) == 1
    assert backups[0].name.startswith("20")
    replaced = {name: content for name, content in release.before.items()
                if name in release.managed}
    assert _snapshot(backups[0]) == replaced
    for name, content in release.managed.items():
        assert (release.root / name).read_bytes() == content
    for name, content in release.before.items():
        if name not in release.managed:
            assert (release.root / name).read_bytes() == content
    assert release.installs == [release.root / WHEEL]
    assert len(release.doctors) == 1
    output = capsys.readouterr().out
    assert str(backups[0]) in output
    assert "AGENTS.md" in output
    assert VERSION in output


def test_update_check_changes_nothing(release, capsys):
    before = _snapshot(release.root)
    assert release.module.main(["--update", "--check"]) == 0
    assert _snapshot(release.root) == before
    assert release.installs == release.doctors == []
    output = capsys.readouterr().out
    assert "1.1.1" in output and VERSION in output
    assert "AGENTS.md" in output


@pytest.mark.parametrize("digest", [None, "sha256:" + "0" * 64])
def test_update_refuses_missing_or_wrong_digest_without_changes(release, capsys, digest):
    release.asset["digest"] = digest
    before = _snapshot(release.root)
    assert release.module.main(["--update"]) == 1
    assert "sha-256" in capsys.readouterr().err.lower()
    assert _snapshot(release.root) == before
    assert release.installs == release.doctors == []


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "safe/../../escape.txt",
                                 "C:/escape.txt", "safe\\..\\escape.txt", "link"])
def test_update_refuses_unsafe_zip_entries_without_changes(release, capsys, name):
    if name == "link":
        entry = zipfile.ZipInfo(name)
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        release.extra[entry] = b"../escape.txt"
    else:
        release.extra[name] = b"synthetic unsafe entry"
    _package(release)
    before = _snapshot(release.root)
    assert release.module.main(["--update"]) == 1
    assert "unsafe" in capsys.readouterr().err.lower()
    assert _snapshot(release.root) == before
    assert release.installs == []


@pytest.mark.parametrize("link", ["symlink", "hard link"])
def test_update_refuses_linked_workspace_files_before_replacing_anything(release, capsys, link):
    target = release.root / "AGENTS.md"
    target.unlink()
    if link == "symlink":
        target.symlink_to(release.root / "creator.txt")
    else:
        target.hardlink_to(release.root / "creator.txt")
    before = _snapshot(release.root)
    assert release.module.main(["--update"]) == 1
    assert link in capsys.readouterr().err.lower()
    assert _snapshot(release.root) == before
    assert target.is_symlink() == (link == "symlink")
    assert release.installs == []


def test_update_refuses_a_malformed_release_version_without_changes(release, capsys):
    release.api["tag_name"] = None
    before = _snapshot(release.root)
    assert release.module.main(["--update"]) == 1
    assert "version" in capsys.readouterr().err.lower()
    assert _snapshot(release.root) == before
    assert release.installs == []


@pytest.mark.parametrize("failure", ["install", "doctor", "build"])
def test_failed_update_keeps_old_version_and_reports_backup(release, monkeypatch, capsys, failure):
    if failure == "install":
        monkeypatch.setattr(release.module, "_install_wheel", lambda *a, **kw: "synthetic failure")
    else:
        report = {"ready_to_edit": failure == "build",
                  "install": {"version": VERSION, "build": "wrong"}}
        monkeypatch.setattr(release.module, "_doctor", lambda engine: (
            subprocess.CompletedProcess([], 1 if failure == "doctor" else 0, "", ""), report))
    assert release.module.main(["--update"]) == 1
    assert (release.root / ".yap-studio-version").read_bytes() == b"1.1.1\n"
    backups = list((release.root / ".yap-studio/backups").iterdir())
    assert (backups[0] / "AGENTS.md").read_bytes() == release.before["AGENTS.md"]
    output = capsys.readouterr()
    assert str(backups[0]) in output.out
    assert output.err
