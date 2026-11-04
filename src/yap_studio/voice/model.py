"""Validated voice records and short creator-facing errors."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import YapError

STATEMENT = (
    "This is my own voice. I agree to Yap Studio making an ElevenLabs voice from my "
    "recordings and using it to re-say words I misspoke in my videos."
)


class VoiceError(YapError):
    """Report a voice fix that needs the creator's attention."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "voice_invalid",
        suggested_action: str = "Check the voice inputs and try again.",
        cost: dict | None = None,
        notices: list[str] | None = None,
        permission: str | None = None,
    ):
        super().__init__(
            message, error_type="input_error", code=code, suggested_action=suggested_action
        )
        self.cost = cost
        self.notices = notices or []
        self.permission = permission

    def to_dict(self) -> dict[str, str]:
        """Include the missing permission when the account needs an edit."""
        return {**super().to_dict(), **({"permission": self.permission} if self.permission else {})}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def generation_fingerprint(fix: dict) -> str:
    """Bind a listen and decision to both the speech and its fitted patch."""
    output = fix.get("generation", {}).get("output_sha256", "")
    patch = fix.get("patch", {}).get("sha256", "")
    return digest(f"{output}:{patch}")


def require_generation(fix: dict, generation: str) -> None:
    if generation != generation_fingerprint(fix):
        raise VoiceError(
            "This fix was made again. Listen to the new version first.", code="voice_fix_changed"
        )


def local_path(root: Path, relative: str) -> Path:
    """Reject paths outside the project's voice directories, including symlinks."""
    if not isinstance(relative, str):
        raise VoiceError("Voice file path must stay inside this project.")
    target = root / relative
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not relative.startswith(("voice/", ".yap/"))
        or not target.resolve().is_relative_to(root.resolve())
        or target.is_symlink()
    ):
        raise VoiceError("Voice file path must stay inside this project.")
    return target


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require(condition: Any) -> None:
    if not condition:
        raise ValueError("invalid voice record")


def _sample(root: Path, row: dict) -> None:
    _require(isinstance(row, dict) and _text(row["take"]) and _hash(row["sha256"]))
    local_path(root, row["path"])
    _require(re.fullmatch(r"voice/samples/(?:g[0-9a-f]{16}/)?s[0-9]{2}\.wav", row["path"]))
    _require(_number(row["start_s"]) and _number(row["end_s"]))
    _require(0 <= row["start_s"] < row["end_s"])
    if "size_bytes" in row:
        _require(type(row["size_bytes"]) is int and row["size_bytes"] > 0)


def _sample_duration(row: dict) -> float:
    return row.get("duration_s", row["end_s"] - row["start_s"])


def _sample_ranges(row: dict, take_duration: float | None = None) -> None:
    """Check joined source ranges against the recorded audio duration."""
    ranges = row["ranges"]
    _require(isinstance(ranges, list) and ranges)
    end = 0
    total = 0.0
    for span in ranges:
        _require(isinstance(span, list) and len(span) == 2)
        first, last = span
        _require(_number(first) and _number(last))
        _require(end <= first < last and (take_duration is None or last <= take_duration))
        total += last - first
        end = last
    _require(abs(row["start_s"] - ranges[0][0]) <= 1e-8)
    _require(abs(row["end_s"] - ranges[-1][1]) <= 1e-8)
    _require(abs(total - _sample_duration(row)) <= 0.05)


def validate_samples_manifest(data: Any, *, max_files: int = 25) -> dict:
    """Check external sample rows without this project's export or take limits."""
    json.dumps(data, allow_nan=False)
    _require(isinstance(data, dict) and data["schema"] == "yap-studio/voice-samples@1")
    rows = data["samples"]
    _require(isinstance(rows, list) and 1 <= len(rows) <= max_files)
    stems = set()
    for row in rows:
        _require(isinstance(row, dict) and _text(row["path"]) and _text(row["take"]))
        relative = Path(row["path"])
        _require(not relative.is_absolute() and relative.suffix.lower() == ".wav")
        _require(relative.stem not in stems)
        stems.add(relative.stem)
        _require(_hash(row["sha256"]) and type(row["size_bytes"]) is int and row["size_bytes"] > 0)
        _require(_number(row["start_s"]) and _number(row["end_s"]))
        _require(0 <= row["start_s"] < row["end_s"])
        _require(_number(row["duration_s"]) and row["duration_s"] > 0)
        if "ranges" in row:
            _sample_ranges(row)
        else:
            _require(abs(row["end_s"] - row["start_s"] - row["duration_s"]) <= 0.05)
    return data


def validate(root: Path, kind: str, data: Any) -> dict:
    """Validate persisted fields before planning, decisions or rendering."""
    try:
        json.dumps(data, allow_nan=False)
        _require(isinstance(data, dict) and data["schema"] == f"yap-studio/voice-{kind}@1")
        if kind == "consent":
            _require(isinstance(data["samples"], list))
        if kind == "samples" or kind == "consent" and data["samples"]:
            _require(isinstance(data["samples"], list) and data["samples"])
            _require(4 <= len(data["samples"]) <= 6)
            durations = {}
            context = None
            for row in data["samples"]:
                _sample(root, row)
                _require("duration_s" not in row or "ranges" in row)
                duration = _sample_duration(row)
                _require(_number(duration) and 15 - 1e-8 <= duration <= 30 + 1e-8)
                if "ranges" in row:
                    if context is None:
                        from yap_studio.project.status import ProjectContext

                        context = ProjectContext(root)
                    _require(any(take.id == row["take"] for take in context.project.takes))
                    if row["take"] not in durations:
                        info = ffmpeg.probe(context.take_path(row["take"]))
                        durations[row["take"]] = info.video_duration_s or info.duration_s
                    _sample_ranges(row, durations[row["take"]])
            _require(45 <= sum(_sample_duration(row) for row in data["samples"]) <= 150)
            _require(len({row["path"] for row in data["samples"]}) == len(data["samples"]))
        if kind == "consent":
            _require(data["status"] in ("given", "withdrawn") and data["statement"] == STATEMENT)
            _require(_text(data["given_at"]) and _text(data["given_by"]))
            voice = data["voice"]
            _require(isinstance(voice, dict) and _text(voice["name"]))
            _require("id" not in voice and "voice_id" not in voice)
            manifest = voice.get("samples_manifest")
            if manifest is not None:
                _require(isinstance(manifest, dict) and set(manifest) == {"sha256", "files"})
                _require(_hash(manifest["sha256"]))
                _require(type(manifest["files"]) is int and 1 <= manifest["files"] <= 25)
            if voice.get("id_sha256") is not None:
                _require(_hash(voice["id_sha256"]) and voice["made"] in ("linked", "yap-studio"))
                _require(_text(voice["at"]))
            if data["status"] == "withdrawn":
                _require(_text(data["withdrawn_at"]))
        if kind == "fixes":
            _require(isinstance(data["fixes"], list))
            seen = set()
            for fix in data["fixes"]:
                _require(re.fullmatch(r"f[1-9][0-9]*", fix["id"]) and fix["id"] not in seen)
                seen.add(fix["id"])
                _require(_text(fix["take"]) and isinstance(fix["words"], list))
                _require(
                    1 <= len(fix["words"]) <= 6 and len(set(fix["words"])) == len(fix["words"])
                )
                _require(
                    all(_text(word) and word.startswith(fix["take"] + ":") for word in fix["words"])
                )
                _require(_text(fix["said"]) and _text(fix["say"]) and isinstance(fix["why"], str))
                _require(len(fix["say"]) <= 60 and 1 <= len(fix["say"].split()) <= 8)
                if "from" in fix:
                    origin = fix["from"]
                    _require(isinstance(origin, dict) and origin["kind"] in ("own", "clone"))
                    if origin["kind"] == "own":
                        from yap_studio.stages.edges import heard_words

                        _require(_text(origin["take"]) and isinstance(origin["words"], list))
                        _require(origin["words"] and len(set(origin["words"]))
                                 == len(origin["words"]))
                        _require(all(_text(word) and word.startswith(origin["take"] + ":")
                                     for word in origin["words"]))
                        _require(_number(origin["start_s"]) and _number(origin["end_s"]))
                        _require(0 <= origin["start_s"] < origin["end_s"])
                        _require(_text(origin["text"]))
                        tokens = heard_words(origin["text"])
                        _require(tokens and "?" not in tokens and tokens == heard_words(fix["say"]))
                        if "take_sha256" in origin:
                            _require(_hash(origin["take_sha256"]))
                    else:
                        from yap_studio.voice.recipe import SCHEMA, validate_recipe

                        validate_recipe({"schema": SCHEMA,
                                         **{k: v for k, v in origin.items() if k != "kind"}})
                _require(fix["status"] in ("proposed", "ready", "approved", "rejected", "failed"))
                source = fix["source"]
                _require(_number(source["start_s"]) and _number(source["end_s"]))
                _require(0 <= source["start_s"] < source["end_s"])
                _require(fix["picture"]["kind"] in ("voiceover", "covered"))
                estimate = fix["estimate"]
                _require(type(estimate["characters"]) is int and estimate["characters"] > 0)
                _require(_number(estimate["credits"]) and estimate["credits"] >= 0)
                _require(_text(estimate["model"]) and _number(estimate["multiplier"]))
                if "patch" in fix:
                    patch = fix["patch"]
                    local_path(root, patch["path"])
                    folder = str(Path(patch["path"]).parent)
                    base = f"voice/fixes/{fix['id']}"
                    _require(folder == base or re.fullmatch(base + r"/g[0-9a-f]{16}", folder))
                    _require(patch["path"] == f"{folder}/patch.wav")
                    _require(_hash(patch["sha256"]))
                    for key in (
                        "start_sample",
                        "samples",
                        "channels",
                        "fade_in_samples",
                        "fade_out_samples",
                    ):
                        _require(type(patch[key]) is int and patch[key] >= 0)
                    _require(patch["samples"] > 0 and patch["channels"] in (1, 2))
                    _require(
                        patch["fade_in_samples"] + patch["fade_out_samples"] <= patch["samples"]
                    )
                    _require(_number(patch["stretch"]) and 0.8 <= patch["stretch"] <= 1.25)
                    _require(_number(patch["gain_db"]))
                    for name in ("before", "after"):
                        clip = fix["clips"][name]
                        local_path(root, clip["path"])
                        _require(clip["path"] == f"{folder}/{name}.wav")
                        _require(_hash(clip["sha256"]))
                    _require(fix["checks"]["status"] in ("pass", "warn", "fail"))
                    _require(isinstance(fix["checks"]["items"], list))
                    _require(
                        {item["id"] for item in fix["checks"]["items"]}
                        == {
                            "splice.click",
                            "splice.level",
                            "splice.room_tone",
                            "splice.pace",
                            "splice.heard",
                        }
                    )
                    for item in fix["checks"]["items"]:
                        _require(
                            item["id"]
                            in (
                                "splice.click",
                                "splice.level",
                                "splice.room_tone",
                                "splice.pace",
                                "splice.heard",
                            )
                        )
                        _require(item["status"] in ("pass", "warn", "fail", "skipped"))
                        if item["id"] == "splice.level":
                            _require(
                                item["reference_lufs"] is None or _number(item["reference_lufs"])
                            )
                            _require(_number(item["reference_rms_db"]))
                        if item["id"] == "splice.room_tone":
                            _require(_number(item["source_floor_db"]))
                        if item["id"] == "splice.pace":
                            _require(_number(item["gap_s"]) and _number(item["stretch"]))
                if "generation" in fix:
                    generation = fix["generation"]
                    _require(isinstance(generation, dict))
                    _require(_hash(generation["request_sha256"]))
                    _require(_hash(generation["output_sha256"]))
                    _require(type(generation["seed"]) is int and _text(generation["model"]))
                    _require(_text(generation["at"]) and _number(generation["cost"]["credits"]))
                    _require(generation["cost"]["source"] in ("history", "estimate", "own"))
                if fix["status"] in ("ready", "approved"):
                    _require("patch" in fix and "generation" in fix)
                if "decision" in fix:
                    _require(fix["decision"]["action"] in ("approve", "reject"))
                    _require(fix["decision"]["via"] in ("review", "chat"))
                    _require(_text(fix["decision"]["at"]))
        if kind == "pending":
            _require(data["state"] in ("pending", "unreconciled"))
            _require(_hash(data["request_sha256"]) and type(data["pid"]) is int)
            _require(_text(data["host"]))
    except (AssertionError, KeyError, TypeError, ValueError, AttributeError):
        raise VoiceError(
            f"voice/{kind}.json is not a valid voice record.",
            code="voice_record_invalid",
            suggested_action="Restore this file from a saved copy and try again.",
        ) from None
    return data


def read(root: Path, kind: str, *, optional: bool = False) -> dict | None:
    relative = ".yap/voice/pending.json" if kind == "pending" else f"voice/{kind}.json"
    target = local_path(root, relative)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if optional:
            return None
        raise VoiceError(f"voice/{kind}.json is missing.", code="voice_record_missing") from None
    except (OSError, ValueError, UnicodeError):
        raise VoiceError(
            f"voice/{kind}.json cannot be read.", code="voice_record_invalid"
        ) from None
    return validate(root, kind, data)


def write(root: Path, kind: str, data: dict) -> None:
    """Publish under the caller's project lock."""
    validate(root, kind, data)
    relative = ".yap/voice/pending.json" if kind == "pending" else f"voice/{kind}.json"
    target = local_path(root, relative)
    paths.validate_output_dir(target.parent)
    paths.atomic_write_text(
        target, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def checked_file(root: Path, row: dict) -> Path:
    target = local_path(root, row["path"])
    if not target.is_file() or receipt.sha256_file(target) != row["sha256"]:
        raise VoiceError(
            f"{row['path']} is missing or changed.",
            code="voice_patch_changed",
            suggested_action="Restore the file or make this fix again before cutting.",
        )
    return target
