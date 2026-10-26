"""Read release bullets from published and pending notes."""

from pathlib import Path


def release_bullets() -> list[str]:
    """Return changelog and change-file bullets in file order."""
    root = Path(__file__).parents[1]
    sources = [root / "CHANGELOG.md", *sorted((root / "changes").glob("*.md"))]
    return [
        line for source in sources if source.name != "README.md"
        for line in source.read_text(encoding="utf-8").splitlines() if line.startswith("- ")
    ]
