"""Install and verify registry items inside the versioned engine cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from yap_studio import paths, tools
from yap_studio.cache import cache_root
from yap_studio.errors import DependencyError, ProcessingError
from yap_studio.hf_registry import catalog
from yap_studio.render import hyperframes
from yap_studio.render.hyperframes import HYPERFRAMES_VERSION, RendererInstall

SCHEMA = "yap-studio/hf-registry-install@1"


def items_root() -> Path:
    """Return the cache folder for this renderer and registry commit."""
    commit = catalog.load_catalog()["registry"]["commit"]
    return cache_root() / "registry" / f"hyperframes-{HYPERFRAMES_VERSION}-{commit[:12]}"


def item_dir(name: str) -> Path:
    """Return a catalog item's cache folder."""
    catalog.item(name)
    return items_root() / name


def registry_home() -> Path:
    """Keep CLI manifest caches out of the user's home."""
    return cache_root() / "registry" / "home"


def _metadata(name: str) -> dict:
    return {
        "schema": SCHEMA,
        "name": name,
        "commit": catalog.load_catalog()["registry"]["commit"],
        "files": catalog.item(name)["files"],
    }


def _verify_files(root: Path, files: dict[str, str]) -> str | None:
    """Return the first missing, changed, or escaping file."""
    for relative, digest in files.items():
        file = root / relative
        try:
            if (
                not file.resolve().is_relative_to(root.resolve())
                or not file.is_file()
                or hashlib.sha256(file.read_bytes()).hexdigest().lower() != digest.lower()
            ):
                return relative
        except (OSError, RuntimeError, ValueError):
            return relative
    return None


def installed(name: str) -> Path | None:
    """Return the item folder only while its metadata and pinned files are intact."""
    root = item_dir(name)
    try:
        if not root.resolve().is_relative_to(items_root().resolve()) or not root.is_dir():
            return None
        record = root / "install.json"
        if not record.resolve().is_relative_to(root.resolve()):
            return None
        if json.loads(record.read_text(encoding="utf-8")) != _metadata(name):
            return None
        if _verify_files(root, catalog.item(name)["files"]) is None:
            return root
    except (OSError, UnicodeError, ValueError, RuntimeError):
        pass
    return None


def _changed(name: str, file: str) -> DependencyError:
    return DependencyError(
        f'HyperFrames registry item "{name}" differs from the pinned catalog: {file}.',
        error_type="dependency_error",
        code="registry_item_changed",
        suggested_action=(
            f"Delete the item folder {item_dir(name)} and run yap-studio renderer install again."
        ),
    )


def item_files(name: str) -> Path:
    """Return verified files or explain how to install or repair the item."""
    root = installed(name)
    if root is not None:
        return root
    directory = item_dir(name)
    if directory.exists() or directory.is_symlink():
        raise _changed(
            name, _verify_files(directory, catalog.item(name)["files"]) or "install.json"
        )
    raise DependencyError(
        f'HyperFrames registry item "{name}" is missing.',
        error_type="dependency_error",
        code="registry_item_missing",
        suggested_action="Run yap-studio renderer install.",
    )


def _cli_error(name: str, detail: str) -> DependencyError:
    return DependencyError(
        f'Could not install HyperFrames registry item "{name}": {detail}',
        error_type="dependency_error",
        code="registry_item_install",
        suggested_action="Run yap-studio renderer install again.",
    )


def _response(name: str, stdout: str) -> dict:
    try:
        result = json.loads(stdout.strip())
    except (ValueError, TypeError) as exc:
        raise _cli_error(name, "HyperFrames add did not return a JSON line.") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        detail = (
            result.get("error", "HyperFrames add failed.") if isinstance(result, dict) else result
        )
        raise _cli_error(name, str(detail))
    return result


def stage_item(
    name: str, executable: Path, staging: Path, registry_url: str, *,
    home: Path, history: Path, timeout_s: float,
) -> dict:
    """Add an item to staging with explicit registry, home and history folders."""
    (staging / "hyperframes.json").write_text(
        json.dumps(
            {
                "registry": registry_url,
                "paths": {
                    "blocks": "compositions",
                    "components": "compositions/components",
                    "assets": "assets",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    env = {
        **hyperframes.renderer_env(),
        "HOME": str(home),
        "HYPERFRAMES_HISTORY_ROOT": str(history),
    }
    try:
        process = tools.run_tool(
            "hyperframes",
            ["add", name, "--dir", str(staging), "--json", "--no-clipboard"],
            executable=executable,
            env=env,
            timeout_s=timeout_s,
        )
    except ProcessingError as exc:
        if exc.stdout:
            # Preserve add's structured error even when it exits unsuccessfully.
            _response(name, exc.stdout)
        raise _cli_error(name, exc.full_stderr or str(exc)) from exc
    return _response(name, process.stdout or "")


def _install_item(name: str, install: RendererInstall, timeout_s: float) -> None:
    staging = Path(tempfile.mkdtemp(dir=items_root(), prefix=".staging-"))
    try:
        home = paths.validate_output_dir(registry_home())
        with tempfile.TemporaryDirectory(dir=items_root(), prefix=".history-") as history:
            result = stage_item(
                name, install.executable, staging, catalog.registry_url(),
                home=home, history=Path(history), timeout_s=timeout_s,
            )
        written = result.get("written")
        if not isinstance(written, list):
            raise _changed(name, "written file list")
        relative_files: set[str] = set()
        for value in written:
            if not isinstance(value, str):
                raise _changed(name, "invalid written path")
            file = Path(value)
            try:
                if not file.is_absolute() or not file.resolve().is_relative_to(staging.resolve()):
                    raise _changed(name, value)
                relative_files.add(file.relative_to(staging).as_posix())
            except (OSError, RuntimeError, ValueError) as exc:
                raise _changed(name, value) from exc
        expected = catalog.item(name)["files"]
        differences = relative_files.symmetric_difference(expected)
        if differences:
            raise _changed(name, sorted(differences)[0])
        difference = _verify_files(staging, expected)
        if difference is not None:
            raise _changed(name, difference)
        (staging / "install.json").write_text(json.dumps(_metadata(name)) + "\n", encoding="utf-8")
        target = item_dir(name)
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def install_items(
    names: Sequence[str] | None = None,
    *,
    install: RendererInstall | None = None,
    notice: Callable[[str], None] | None = None,
    timeout_s: float = 600,
) -> list[dict]:
    """Install missing or altered catalog items with the pinned CLI."""
    selected = list(catalog.items() if names is None else dict.fromkeys(names))
    for name in selected:
        catalog.item(name)
    renderer = install if install is not None else hyperframes.ensure_installed(notice=notice)
    results: list[dict] = []
    with paths.operation(), hyperframes._install_lock(timeout_s, directory=items_root()):
        for name in selected:
            state = "already"
            if installed(name) is None:
                if notice:
                    notice(f"Installing HyperFrames registry item {name}.")
                _install_item(name, renderer, timeout_s)
                state = "installed"
            results.append(
                {"name": name, "status": state, "files": len(catalog.item(name)["files"])}
            )
    return results


def status() -> dict[str, list[str]]:
    """Report cache state without installing or making network requests."""
    result: dict[str, list[str]] = {"ready": [], "missing": [], "changed": []}
    for name in catalog.items():
        root = item_dir(name)
        state = (
            "ready"
            if installed(name) is not None
            else "changed"
            if root.exists() or root.is_symlink()
            else "missing"
        )
        result[state].append(name)
    return result
