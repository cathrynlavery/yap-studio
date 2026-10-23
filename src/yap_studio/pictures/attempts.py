"""Durable evidence that a sheet line may already have been charged."""

from __future__ import annotations

import json
import re
from pathlib import Path

from yap_studio import paths
from yap_studio.errors import ValidationError
from yap_studio.media.catalog import _timestamp
from yap_studio.pictures import wavespeed
from yap_studio.project.lock import project_lock

SCHEMA = "yap-studio/picture-attempts@1"
FIELDS = {"sheet", "number", "prediction_id", "quote_usd", "started_at"}


def load_attempts(project_dir: Path) -> list[dict]:
    """Refuse unreadable or malformed attempt evidence."""
    source = project_dir / "pictures/attempts.json"
    try:
        data = json.loads(source.read_bytes())
        if (
            not isinstance(data, dict)
            or set(data) != {"schema", "attempts"}
            or data["schema"] != SCHEMA
            or not isinstance(data["attempts"], list)
        ):
            raise ValueError("invalid attempts document")
        seen = set()
        for entry in data["attempts"]:
            if not isinstance(entry, dict) or set(entry) != FIELDS:
                raise ValueError("invalid attempt fields")
            if (
                not isinstance(entry["sheet"], str)
                or re.fullmatch(r"s[1-9][0-9]*", entry["sheet"]) is None
                or type(entry["number"]) is not int
                or not 1 <= entry["number"] <= 8
            ):
                raise ValueError("invalid sheet line")
            identifier = entry["prediction_id"]
            if identifier is not None and wavespeed.prediction_id(identifier) is None:
                raise ValueError("invalid prediction id")
            wavespeed.number(entry["quote_usd"])
            _timestamp(entry["started_at"])
            identity = (entry["sheet"], entry["number"])
            if identity in seen:
                raise ValueError("duplicate attempt")
            seen.add(identity)
        return data["attempts"]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("pictures.attempts", "cannot read the picture attempts list") from exc


def save_attempt(project_dir: Path, sheet: str, number: int, entry: dict | None) -> None:
    """Replace or remove one attempt under the project write lock."""
    with project_lock(project_dir):
        entries = [
            old
            for old in load_attempts(project_dir)
            if (old["sheet"], old["number"]) != (sheet, number)
        ]
        if entry is not None:
            entries.append(entry)
        paths.atomic_write_text(
            project_dir / "pictures/attempts.json",
            json.dumps({"schema": SCHEMA, "attempts": entries}, indent=2, allow_nan=False) + "\n",
        )
