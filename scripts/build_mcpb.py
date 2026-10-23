"""Build the Claude Desktop bundle from the repository source."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/build_mcpb.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/build_mcpb.py"
    )

import argparse
import os
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _manifest_version() -> str:
    import json

    return json.loads((ROOT / "mcpb" / "manifest.json").read_text())["version"]


def _members() -> list[tuple[Path, str]]:
    members = [
        (ROOT / "mcpb" / "manifest.json", "manifest.json"),
        *(
            (ROOT / name, name)
            for name in ("pyproject.toml", "uv.lock", "LICENSE", "NOTICE", "README.md")
        ),
    ]
    for directory in (ROOT / "LICENSES", ROOT / "src"):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            members.append((path, path.relative_to(ROOT).as_posix()))
    return members


def build(output_dir: Path) -> Path:
    """Write the bundle atomically and return its path."""
    version = _version()
    manifest_version = _manifest_version()
    if manifest_version != version:
        raise ValueError(
            f"manifest version {manifest_version!r} does not match project version {version!r}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"yap-studio-{version}.mcpb"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_dir,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for source, member in _members():
                archive.write(source, member)
        os.replace(temporary, output)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist", help="output directory")
    args = parser.parse_args()
    print(build(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
