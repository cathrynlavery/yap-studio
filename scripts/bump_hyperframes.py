"""Move the HyperFrames renderer pin, npm lock and registry catalog to a release."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/bump_hyperframes.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/bump_hyperframes.py"
    )

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from bump_version import ReleaseError, version_parts

from yap_studio.errors import ValidationError, YapError
from yap_studio.hf_registry.install import stage_item

ROOT = Path(__file__).resolve().parents[1]
CONSTANT = "src/yap_studio/render/hyperframes.py"
LOCK_DIR = "src/yap_studio/render/renderer-lock"
LOCK_ENTRY = "node_modules/hyperframes"
CATALOG = "src/yap_studio/hf_registry/catalog.json"
NOTICE = "NOTICE"
NOTICE_COMMIT = (
    r"(?m)(^Setup downloads the HyperFrames registry items[^\r\n]*\r?\nat commit )"
    r"[0-9a-fA-F]{40}\b"
)
# Each pattern must match exactly once; {v} is the escaped pinned version.
PINS = {
    CONSTANT: r'HYPERFRAMES_VERSION = "{v}"',
    f"{LOCK_DIR}/package.json": r'"hyperframes": "{v}"',
    "src/yap_studio/templates/transition_cinematic_zoom/template.html": r"HyperFrames {v} awaits",
    "src/yap_studio/templates/transition_light_leak/template.html": r"HyperFrames {v} awaits",
    "src/yap_studio/templates/transition_whip_pan/template.html": r"HyperFrames {v} awaits",
    "tests/test_fonts.py": r"HyperFrames {v} FONT_ALIAS_MAP",
    "tests/test_studio_export.py": r'\.version == "{v}"',
}
CHANGELOG_PIN = r"(?m)^- Pin the renderer to HyperFrames {v}\b"
# npm 10 drops the libc fields npm 11 writes, which churns the committed lock.
NPM_MAJOR = 11


def run_npm(args: list[str], cwd: Path) -> str:
    """Run npm without lifecycle scripts and return its standard output."""
    result = subprocess.run(
        ["npm", *args], cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=300, check=True,
    )
    return result.stdout


def run_git(args: list[str], cwd: Path) -> str:
    """Read upstream refs without changing the checkout."""
    result = subprocess.run(
        ["git", *args], cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True,
        text=True, timeout=60, check=True,
    )
    return result.stdout


def _registry_commit(root: Path, version: str) -> str:
    """Resolve the release tag, preferring an annotated tag's peeled commit."""
    tag = f"refs/tags/v{version}"
    output = run_git(
        ["ls-remote", "https://github.com/heygen-com/hyperframes", tag, f"{tag}^{{}}"], root,
    )
    refs = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2 and re.fullmatch(r"[0-9a-f]{40}", fields[0]):
            refs[fields[1]] = fields[0]
    commit = refs.get(f"{tag}^{{}}", refs.get(tag))
    if commit is None:
        raise ReleaseError(
            f"HyperFrames tag v{version} is missing.",
            f"Wait for upstream to publish tag v{version}, or choose a release with a tag.",
        )
    return commit


def _catalog_edit(text: str, keys: tuple[str, ...], value: object) -> str:
    """Replace one JSON value without reformatting its neighbours."""
    decoder = json.JSONDecoder()
    position = text.index("{") + 1
    while True:
        position += re.match(r"[\s,]*", text[position:]).end()
        if text[position] == "}":
            raise ValidationError(CATALOG, f"missing catalog field {keys[0]}")
        key, position = decoder.raw_decode(text, position)
        position += re.match(r"\s*:\s*", text[position:]).end()
        old, end = decoder.raw_decode(text, position)
        if key == keys[0]:
            if len(keys) > 1:
                replacement = _catalog_edit(text[position:end], keys[1:], value)
            elif old == value:
                return text
            else:
                indent = re.match(r"[ \t]*", text[text.rfind("\n", 0, position) + 1:])[0]
                replacement = json.dumps(value, indent=2).replace("\n", "\n" + indent)
            return text[:position] + replacement + text[end:]
        position = end


def _refresh_registry(root: Path, version: str, commit: str) -> dict:
    """Re-add cards and helpers with a temporary CLI and hash their written files."""
    source = root / CATALOG
    original = source.read_text(encoding="utf-8")
    data = json.loads(original)
    url = f"https://raw.githubusercontent.com/heygen-com/hyperframes/{commit}/registry"
    updated = _catalog_edit(original, ("hyperframes_version",), version)
    for key, value in {"tag": f"v{version}", "commit": commit, "url": url}.items():
        updated = _catalog_edit(updated, ("registry", key), value)
    result = {"commit": commit, "changed": [], "unchanged": []}
    with tempfile.TemporaryDirectory(prefix="hyperframes-registry-") as temporary:
        directory = Path(temporary).resolve()
        renderer = directory / "renderer"
        renderer.mkdir()
        for filename in ("package.json", "package-lock.json"):
            shutil.copyfile(root / LOCK_DIR / filename, renderer / filename)
        run_npm(["ci", "--ignore-scripts", "--no-audit", "--no-fund"], renderer)
        home = directory / "home"
        home.mkdir()
        for name, entry in data["items"].items():
            with tempfile.TemporaryDirectory(dir=directory, prefix="item-") as item:
                staging = Path(item)
                with tempfile.TemporaryDirectory(dir=directory, prefix="history-") as history:
                    try:
                        response = stage_item(
                            name, renderer / "node_modules/.bin/hyperframes", staging, url,
                            home=home, history=Path(history), timeout_s=300,
                        )
                    except YapError as exc:
                        raise ReleaseError(
                            str(exc), f"Check {name} at tag v{version}, then retry the refresh.",
                        ) from exc
                written = response.get("written")
                if not isinstance(written, list) or not written:
                    raise ValidationError(CATALOG, f"{name}: add returned no written file list")
                files = {}
                for value in written:
                    if not isinstance(value, str):
                        raise ValidationError(CATALOG, f"{name}: add returned an invalid path")
                    file = Path(value)
                    if not file.is_absolute() or not file.resolve().is_relative_to(staging):
                        raise ValidationError(
                            CATALOG, f"{name}: add wrote outside staging: {value}",
                        )
                    relative = file.relative_to(staging).as_posix()
                    files[relative] = hashlib.sha256(file.read_bytes()).hexdigest()
                if entry["main"] not in files:
                    raise ValidationError(
                        CATALOG, f"{name}: add did not write main {entry['main']}",
                    )
                state = "unchanged" if files == entry["files"] else "changed"
                result[state].append(name)
                keys = ("items", name, "files")
                if files.keys() == entry["files"].keys():
                    for relative, digest in files.items():
                        updated = _catalog_edit(updated, (*keys, relative), digest)
                else:
                    updated = _catalog_edit(updated, keys, dict(sorted(files.items())))
    if updated != original:
        source.write_text(updated, encoding="utf-8")
    return result


def pinned_version(root: Path) -> str:
    """Read the renderer pin from its Python constant."""
    text = (root / CONSTANT).read_text(encoding="utf-8")
    found = re.search(r'(?m)^HYPERFRAMES_VERSION = "([^"]+)"', text)
    if found is None:
        raise ValidationError(CONSTANT, "missing HYPERFRAMES_VERSION")
    version_parts(found[1])
    return found[1]


def latest_version(root: Path) -> str:
    """Ask npm for the newest published HyperFrames release."""
    value = json.loads(run_npm(["view", "hyperframes", "version", "--json"], root))
    if not isinstance(value, str):
        raise ValidationError("npm", "hyperframes version must be a string")
    version_parts(value)
    return value


def _replace_pin(text: str, relative: str, pattern: str, current: str, new: str) -> str:
    """Swap the version inside one pattern, refusing when the pattern moved."""
    old = re.compile(pattern.format(v=re.escape(current)))
    if len(old.findall(text)) != 1:
        raise ValidationError(relative, f"expected one pin matching {old.pattern}")
    return old.sub(lambda match: match[0].replace(current, new), text)


def _notice_update(text: str, commit: str) -> str:
    """Update only the downloaded registry commit, refusing when the sentence moved."""
    pattern = re.compile(NOTICE_COMMIT)
    if len(pattern.findall(text)) != 1:
        raise ValidationError(NOTICE, "expected one registry commit sentence")
    return pattern.sub(lambda match: match[1] + commit, text)


def _changelog_update(text: str, current: str, new: str) -> str:
    """Update the Unreleased pin bullet, or add one when the last pin already shipped."""
    heading = re.search(r"(?m)^## Unreleased[ \t]*\n", text)
    if heading is None:
        raise ValidationError("CHANGELOG.md", "missing ## Unreleased")
    following = re.search(r"(?m)^## ", text[heading.end():])
    end = len(text) if following is None else heading.end() + following.start()
    body = text[heading.end():end].strip("\n")
    pin = re.compile(CHANGELOG_PIN.format(v=re.escape(current)))
    if len(pin.findall(body)) == 1:
        body = pin.sub(f"- Pin the renderer to HyperFrames {new}", body)
    else:
        body = f"- Pin the renderer to HyperFrames {new}." + (f"\n{body}" if body else "")
    tail = "\n" if following is None else "\n\n"
    return text[:heading.end()] + "\n" + body + tail + text[end:]


def _lock_versions(lock: dict) -> dict[str, str]:
    """Map each locked package path to its version."""
    return {path: entry.get("version", "") for path, entry in lock["packages"].items() if path}


def bump(root: Path, new: str, *, registry_only: bool = False) -> dict:
    """Refresh the registry and optionally pins and lock, restoring files on failure."""
    current = pinned_version(root)
    if registry_only and new != current:
        raise ValidationError("version", f"--registry uses the pinned {current}")
    if version_parts(new) < version_parts(current):
        raise ValidationError("version", f"{new} is older than the pinned {current}")
    if new == current and not registry_only:
        return {"current": current, "new": new, "changed": False, "lock_changes": []}
    if not registry_only:
        npm = run_npm(["--version"], root).strip()
        if not npm.split(".")[0].isdigit() or int(npm.split(".")[0]) < NPM_MAJOR:
            raise ReleaseError(
                f"npm {npm} would rewrite the lock's libc fields.",
                f"Install npm {NPM_MAJOR} or newer (npm install -g npm@{NPM_MAJOR}) and retry.",
            )
    lock_path = root / LOCK_DIR / "package-lock.json"
    tracked = [*PINS, "CHANGELOG.md", f"{LOCK_DIR}/package-lock.json", CATALOG, NOTICE]
    original = {relative: (root / relative).read_bytes() for relative in tracked}
    before = _lock_versions(json.loads(original[f"{LOCK_DIR}/package-lock.json"]))
    try:
        commit = _registry_commit(root, new)
        notice = _notice_update(original[NOTICE].decode("utf-8"), commit)
        (root / NOTICE).write_bytes(notice.encode("utf-8"))
        if not registry_only:
            for relative, pattern in PINS.items():
                updated = _replace_pin(
                    original[relative].decode("utf-8"), relative, pattern, current, new,
                )
                (root / relative).write_text(updated, encoding="utf-8")
            (root / "CHANGELOG.md").write_text(
                _changelog_update(original["CHANGELOG.md"].decode("utf-8"), current, new),
                encoding="utf-8")
            run_npm(["install", "--package-lock-only", "--ignore-scripts", "--no-audit",
                     "--no-fund"], root / LOCK_DIR)
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            entry = lock["packages"].get(LOCK_ENTRY, {})
            expected = json.loads(run_npm(
                ["view", f"hyperframes@{new}", "dist.integrity", "--json"], root))
            if entry.get("version") != new or entry.get("integrity") != expected:
                raise ValidationError(
                    "package-lock.json", f"locked hyperframes does not match npm's {new}")
        registry = _refresh_registry(root, new, commit)
        after = _lock_versions(json.loads(lock_path.read_text(encoding="utf-8")))
        changed = any((root / name).read_bytes() != content for name, content in original.items())
    except (
        YapError, OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError,
    ) as exc:
        for relative, content in original.items():
            (root / relative).write_bytes(content)
        action = exc.suggested_action if isinstance(exc, YapError) else None
        raise ReleaseError(
            f"HyperFrames bump failed ({exc}); pinned files were restored.",
            action or "Check that git and npm can reach upstream and each pin in PINS "
            "still matches once, then retry.",
        ) from exc
    changes = sorted(
        f"{path.removeprefix('node_modules/')} {before.get(path, '(new)')} -> "
        f"{after.get(path, '(removed)')}"
        for path in before.keys() | after.keys() if before.get(path) != after.get(path)
    )
    return {"current": current, "new": new, "changed": changed,
            "lock_changes": changes, "registry": registry}


def main() -> int:
    """Report or apply a pin bump and print a plain actionable error on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", metavar="VERSION",
                        help="release to pin; defaults to the newest on npm")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="report the pinned and newest versions without writing")
    mode.add_argument("--registry", action="store_true",
                      help="refresh the registry catalog at the pinned version only")
    args = parser.parse_args()
    if args.registry and args.version:
        parser.error("--registry uses the pinned release; omit VERSION")
    try:
        new = pinned_version(ROOT) if args.registry else args.version or latest_version(ROOT)
        version_parts(new)
        if args.check:
            current = pinned_version(ROOT)
            result = {"current": current, "latest": new,
                      "outdated": version_parts(new) > version_parts(current)}
        else:
            result = bump(ROOT, new, registry_only=args.registry)
    except (YapError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        action = exc.suggested_action if isinstance(exc, YapError) else "Check npm and the pins."
        print(action, file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
