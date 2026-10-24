"""Discover a running home page for another local creator page."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from yap_studio import background, install_info
from yap_studio.homepage.bridge import ACTIVE, events, listening, requests


def _home_for_path(path: str | Path | None) -> tuple[Path, dict] | None:
    if path is None:
        return None
    target = Path(path).expanduser().resolve()
    matches = []
    for record in background.records("home"):
        saved = background.read_record(record)
        if saved is None or not isinstance(saved.get("workspace"), str):
            continue
        workspace = Path(saved["workspace"]).resolve()
        if not target.is_relative_to(workspace):
            continue
        if background.command_matches(saved.get("pid"), record) is not True:
            continue
        document = saved.get("document", {})
        if not isinstance(document.get("url"), str):
            continue
        matches.append((workspace, document))
    return max(matches, key=lambda item: len(item[0].parts)) if matches else None


def workspace_for_path(path: str | Path) -> Path | None:
    """Find the closest workspace with a live detached home containing this path."""
    home = _home_for_path(path)
    return home[0] if home else None


def for_path(path: str | Path | None) -> dict | None:
    """Find a live home link and the delivered notes and final request for a project."""
    home = _home_for_path(path)
    if home is None:
        return None
    workspace, document = home
    target = Path(path).expanduser().resolve()
    relative = target.relative_to(workspace)
    project = (
        Path(*relative.parts[:2]).as_posix()
        if (len(relative.parts) >= 2 and relative.parts[0] == "projects")
        else None
    )
    # Note batches the agent has already received, so a page can say so truthfully.
    delivered = (
        [
            row["detail"]["batch"]
            for row in events(workspace)
            if row.get("kind") == "notes"
            and row.get("project") == project
            and row.get("delivered_at")
            and isinstance(row.get("detail"), dict)
            and isinstance(row["detail"].get("batch"), str)
        ]
        if project
        else []
    )
    final_requested = (
        any(
            row.get("kind") == "final"
            and row.get("state") in ACTIVE
            and row.get("payload", {}).get("project") == project
            for row in requests(workspace)
        )
        if project
        else False
    )
    phone_url = (document.get("phone") or {}).get("url")
    return {
        "url": urlsplit(document["url"])._replace(path="/", query="", fragment="").geturl(),
        "phone_url": urlsplit(phone_url)._replace(path="/", query="", fragment="").geturl()
        if phone_url
        else None,
        "phone_via": (document.get("phone") or {}).get("via"),
        "agent": install_info.workspace_agent(workspace),
        "listening": listening(workspace),
        "delivered": delivered,
        "final_requested": final_requested,
    }
