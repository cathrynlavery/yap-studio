"""Credits for stock actually placed in a video, or every approved pick."""

from __future__ import annotations

import json
from pathlib import Path

from yap_studio import paths, receipt
from yap_studio.errors import ValidationError
from yap_studio.project.model import load_timeline
from yap_studio.stock.guard import _timeline_clips, approved_clips


def _media_hashes(value) -> set[str]:
    if isinstance(value, dict):
        result = (
            {value["media"]["sha256"]}
            if (
                isinstance(value.get("media"), dict)
                and isinstance(value["media"].get("sha256"), str)
            )
            else set()
        )
        for child in value.values():
            result.update(_media_hashes(child))
        return result
    if isinstance(value, list):
        return set().union(*(_media_hashes(child) for child in value))
    return set()


def credits(project_dir: Path, *, all_clips: bool = False) -> dict:
    """Read placement evidence without touching the project or catalog."""
    project_dir = Path(project_dir)
    source = project_dir / "timeline/broll.json"
    placed = set()
    if source.exists():
        try:
            placed = _media_hashes(json.loads(paths.validate_input(source).read_bytes()))
        except (ValueError, OSError) as exc:
            raise ValidationError("timeline.broll", "cannot read B-roll placements") from exc
    source = project_dir / "timeline/agent.json"
    if source.exists():
        try:
            lane = load_timeline(paths.validate_input(source), lane="agent")
            elements = [element for element in lane.elements if element.kind == "graphic"]
            for _, clip in _timeline_clips(elements):
                if isinstance(clip, dict):
                    placed.add(clip["sha256"])
                else:
                    path = project_dir / clip
                    if path.is_file():
                        placed.add(receipt.sha256_file(path))
        except (ValueError, OSError) as exc:
            raise ValidationError("timeline.agent", "cannot read agent placements") from exc
    clips = [
        {**entry, "placed": entry["sha256"] in placed}
        for entry in approved_clips(project_dir)
        if all_clips or entry["sha256"] in placed
    ]
    lines = list(dict.fromkeys(f"Video by {c['author']} on Pexels: {c['page_url']}" for c in clips))
    text = (
        "\n".join([*lines, "Videos provided by Pexels: https://www.pexels.com"])
        if clips
        else "No stock clips in this video."
    )
    return {"schema": "yap-studio/stock-credits@1", "clips": clips, "text": text}
