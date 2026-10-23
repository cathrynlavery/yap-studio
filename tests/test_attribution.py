"""Attribution, provenance coverage, and optional pinned registry verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
TEMPLATES = ROOT / "src" / "yap_studio" / "templates"
PROVENANCE = TEMPLATES / "PROVENANCE.json"
COMMIT = "0d5e395cd31d0f491e9b26d0da110da6e231c512"
COPYRIGHT = "Copyright 2026 HeyGen, Inc."
DERIVED = "Derived from HyperFrames registry item"


@pytest.fixture
def provenance() -> dict:
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


def _tracked_paths() -> set[Path] | None:
    if shutil.which("git") is None:
        return None
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if inside.returncode or inside.stdout.strip() != "true":
        return None
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout.split(b"\0")
    return {Path(raw.decode(errors="surrogateescape")) for raw in tracked if raw}


def test_provenance_schema_and_upstream_license(provenance: dict) -> None:
    assert provenance["schema"] == "yap-studio/provenance@1"
    upstream = provenance["upstream"]
    assert upstream == {
        "name": "HyperFrames registry",
        "repository": "github.com/heygen-com/hyperframes",
        "commit": COMMIT,
        "license": "Apache-2.0",
        "license_file": "LICENSES/Apache-2.0.txt",
        "license_sha256": "4259155fb06f127687ee7b0a8a3682d45132db0f2da26cbc0b7a2d1e796436b8",
        "copyright": COPYRIGHT,
        "notice_file": None,
    }
    assert re.fullmatch(r"[0-9a-f]{40}", upstream["commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", upstream["license_sha256"])
    assert (ROOT / upstream["license_file"]).is_file()
    assert isinstance(provenance["files"], list) and provenance["files"]


def test_provenance_entries_have_prominent_notices(provenance: dict) -> None:
    for entry in provenance["files"]:
        item = entry["item"]
        assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", item)
        assert re.fullmatch(r"[0-9a-f]{64}", entry["upstream_sha256"])
        assert re.fullmatch(
            rf"registry/(?:blocks|components)/{re.escape(item)}/{re.escape(item)}\.html",
            entry["upstream_path"],
        )
        assert isinstance(entry["changes"], str) and entry["changes"].strip()
        relative = Path(entry["yap_path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        file = ROOT / relative
        assert file.is_relative_to(TEMPLATES) and file.is_file(), entry["yap_path"]
        source = file.read_text(encoding="utf-8")
        header = re.match(r"<!doctype html>\s*<!--(.*?)-->", source, re.DOTALL)
        assert header is not None, entry["yap_path"]
        lines = [line.strip() for line in header.group(1).splitlines()]
        attribution = f'{DERIVED} "{item}"'
        assert attribution in lines, entry["yap_path"]
        commit = entry.get("upstream_commit", provenance["upstream"]["commit"])
        assert re.fullmatch(r"[0-9a-f]{40}", commit)
        origin = f'({entry["upstream_path"]} at commit {commit})'
        assert lines[lines.index(attribution) + 1] == origin, entry["yap_path"]
        license_file = provenance["upstream"]["license_file"]
        assert f"{COPYRIGHT} Apache License 2.0 ({license_file})." in lines
        assert any(line.startswith("Modified for Yap Studio, ") for line in lines)


def test_provenance_matches_template_metadata(provenance: dict) -> None:
    recorded = {(entry["item"], entry["yap_path"]) for entry in provenance["files"]}
    assert len(recorded) == len(provenance["files"]), "duplicate provenance entries"
    declared: dict[tuple[str, str], str] = {}
    for metadata_file in TEMPLATES.glob("*/template.json"):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        yap_path = (metadata_file.parent / "template.html").relative_to(ROOT).as_posix()
        for upstream in metadata.get("derived_from", []):
            declared[(upstream["item"], yap_path)] = upstream["path"]
    assert recorded == set(declared)
    for entry in provenance["files"]:
        assert entry["upstream_path"] == declared[(entry["item"], entry["yap_path"])]


def test_all_attributed_template_files_are_listed(provenance: dict) -> None:
    files = {file for file in TEMPLATES.rglob("*") if file.is_file()}
    files.update(
        ROOT / relative for relative in (_tracked_paths() or set())
        if (ROOT / relative).is_relative_to(TEMPLATES) and (ROOT / relative).is_file()
    )
    attributed_files: set[str] = set()
    attributed_items: set[tuple[str, str]] = set()
    for file in files:
        source = file.read_text(encoding="utf-8", errors="replace")
        if DERIVED not in source:
            continue
        relative = file.relative_to(ROOT).as_posix()
        attributed_files.add(relative)
        attributed_items.update(
            (item, relative) for item in re.findall(rf'{re.escape(DERIVED)} "([^"]+)"', source)
        )
    assert attributed_files == {entry["yap_path"] for entry in provenance["files"]}
    assert attributed_items == {(entry["item"], entry["yap_path"]) for entry in provenance["files"]}


def test_upstream_hashes_match_optional_registry_checkout(provenance: dict) -> None:
    checkout = os.environ.get("YAP_STUDIO_HF_REGISTRY")
    if not checkout:
        pytest.skip("set YAP_STUDIO_HF_REGISTRY to a checkout containing registry/ and LICENSE")
    root = Path(checkout).expanduser()
    for entry in provenance["files"]:
        upstream_file = root / entry["upstream_path"]
        assert upstream_file.is_file(), entry["upstream_path"]
        assert hashlib.sha256(upstream_file.read_bytes()).hexdigest() == entry["upstream_sha256"]
    license_file = root / "LICENSE"
    assert license_file.is_file(), "upstream LICENSE is missing"
    assert hashlib.sha256(license_file.read_bytes()).hexdigest() == (
        provenance["upstream"]["license_sha256"]
    )
    assert not (root / "NOTICE").exists(), "upstream notice_file must match the checkout"


def test_notice_describes_adopted_code_and_unbundled_runtime() -> None:
    notice = " ".join((ROOT / "NOTICE").read_text(encoding="utf-8").split())
    for required in (
        "HyperFrames registry",
        COMMIT,
        COPYRIGHT,
        "Apache License 2.0",
        "LICENSES/Apache-2.0.txt",
        "src/yap_studio/templates/PROVENANCE.json",
        "GSAP 3.14.2 is not included.",
        "GSAP Standard License",
        "not an open source license",
    ):
        assert required in notice, required
    assert "contains no code copied from other projects" not in notice


def test_notice_license_paths_exist() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    licenses = set(re.findall(r"\bLICENSES/[A-Za-z0-9_./-]+", notice))
    assert licenses
    for relative in licenses:
        assert (ROOT / relative).is_file(), relative


def test_tracked_files_exclude_blocked_upstream_assets() -> None:
    tracked = _tracked_paths()
    if tracked is None:
        pytest.skip("git is not installed or directory is not a git work tree")
    offenders = sorted(
        relative.as_posix() for relative in tracked
        if "hyperframes-logo" in relative.as_posix().casefold()
        or re.fullmatch(r"gsap.*\.js", relative.name, re.IGNORECASE)
    )
    assert not offenders, "blocked upstream assets tracked:\n" + "\n".join(offenders)


def test_provenance_is_in_the_wheel_package() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packages = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/yap_studio" in packages
    assert PROVENANCE.is_file() and PROVENANCE.is_relative_to(ROOT / "src" / "yap_studio")
    catalog = ROOT / "src/yap_studio/hf_registry/catalog.json"
    assert catalog.is_file() and catalog.is_relative_to(ROOT / "src/yap_studio")


def test_registry_catalog_attribution_and_notice() -> None:
    from yap_studio.hf_registry.catalog import load_catalog

    catalog = load_catalog()
    upstream = catalog["registry"]
    assert upstream["repository"] == "github.com/heygen-com/hyperframes"
    assert upstream["license"] == "Apache-2.0"
    assert upstream["license_file"] == "LICENSES/Apache-2.0.txt"
    assert upstream["copyright"] == COPYRIGHT
    assert (ROOT / upstream["license_file"]).is_file()
    for name, entry in catalog["items"].items():
        assert entry["upstream_path"] == f"registry/{entry['type']}s/{name}/{name}.html"
        assert entry["main"] in entry["files"]
    notice = (ROOT / "NOTICE").read_text()
    assert "src/yap_studio/hf_registry/catalog.json" in notice
    assert upstream["commit"] in notice
    caption_notice = "caption-ports.css reuse those modified caption ports under the same license."
    assert (
        notice.index(caption_notice)
        < notice.index("Setup downloads the HyperFrames registry items")
        < notice.index("HyperFrames theme packs")
    )
