"""Plugin manifests, skill metadata, and Desktop bundle tests."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
DESCRIPTION = (
    "Checks every cut in a finished video: clicks, sync, frames, loudness and platform rules."
)
JSON_MANIFESTS = (
    ".claude-plugin/marketplace.json",
    ".agents/plugins/marketplace.json",
    "plugins/yap-studio/.claude-plugin/plugin.json",
    "plugins/yap-studio/.codex-plugin/plugin.json",
    "plugins/yap-studio/.mcp.json",
    "mcpb/manifest.json",
)


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def _project() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _frontmatter(path: Path) -> dict[str, str]:
    lines = path.read_text().splitlines()
    assert lines[0] == "---"
    end = lines.index("---", 1)
    return dict(line.split(":", 1) for line in lines[1:end])


def test_every_json_manifest_parses() -> None:
    for relative in JSON_MANIFESTS:
        assert _json(relative)


def test_plugin_names_versions_descriptions_and_licenses_match() -> None:
    project = _project()
    version = project["version"]
    license_id = project["license"]
    claude_marketplace = _json(".claude-plugin/marketplace.json")
    codex_marketplace = _json(".agents/plugins/marketplace.json")
    claude = _json("plugins/yap-studio/.claude-plugin/plugin.json")
    codex = _json("plugins/yap-studio/.codex-plugin/plugin.json")
    desktop = _json("mcpb/manifest.json")

    assert claude_marketplace["metadata"]["version"] == version
    assert claude_marketplace["plugins"][0]["version"] == version
    assert claude["version"] == codex["version"] == desktop["version"] == version
    assert {
        claude_marketplace["name"],
        claude_marketplace["plugins"][0]["name"],
        codex_marketplace["name"],
        codex_marketplace["plugins"][0]["name"],
        claude["name"],
        codex["name"],
        desktop["name"],
    } == {project["name"]}
    assert claude["description"] == codex["description"] == DESCRIPTION
    assert claude["license"] == codex["license"] == license_id
    assert desktop["license"] == license_id


def test_project_is_apache_licensed() -> None:
    project = _project()
    assert project["license"] == "Apache-2.0"
    assert (ROOT / "LICENSE").read_bytes() == (ROOT / "LICENSES/Apache-2.0.txt").read_bytes()


def test_mcp_command_is_a_project_console_script() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    server = _json("plugins/yap-studio/.mcp.json")["mcpServers"]["yap-studio"]
    assert server["command"] in project["scripts"]
    assert server["args"] == ["mcp"]


def test_codex_paths_are_relative_and_exist() -> None:
    plugin_root = ROOT / "plugins" / "yap-studio"
    manifest = _json("plugins/yap-studio/.codex-plugin/plugin.json")
    for field in ("skills", "mcpServers"):
        relative = manifest[field]
        assert relative.startswith("./")
        assert (plugin_root / relative).exists()


def test_skill_has_required_frontmatter() -> None:
    skill = ROOT / "plugins" / "yap-studio" / "skills" / "verify-video" / "SKILL.md"
    metadata = _frontmatter(skill)
    assert metadata["name"].strip() == "verify-video"
    assert metadata["description"].strip()


def test_mcpb_manifest_has_required_uv_fields() -> None:
    manifest = _json("mcpb/manifest.json")
    assert manifest["manifest_version"] == "0.4"
    assert manifest["server"]["type"] == "uv"
    assert (ROOT / manifest["server"]["entry_point"]).is_file()
    assert manifest["compatibility"]["platforms"] == ["darwin", "linux", "win32"]


def test_build_mcpb_contains_exactly_allowed_files(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_mcpb.py", "--out", str(tmp_path)],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    bundle = Path(result.stdout.strip())
    assert bundle == tmp_path / f"yap-studio-{_project()['version']}.mcpb"
    assert bundle.is_file()

    expected = {"manifest.json", "pyproject.toml", "uv.lock", "LICENSE", "NOTICE", "README.md"}
    for directory_name in ("LICENSES", "src"):
        directory = ROOT / directory_name
        if directory.exists():
            expected.update(
                path.relative_to(ROOT).as_posix()
                for path in directory.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
            )

    with zipfile.ZipFile(bundle) as archive:
        members = set(archive.namelist())
    assert members == expected
    assert {
        "manifest.json",
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
        "NOTICE",
        "src/yap_studio/mcp_server.py",
    } <= members
    assert not any(
        forbidden in member
        for member in members
        for forbidden in (".venv", "tests/", "docs/", "__pycache__", ".git")
    )
