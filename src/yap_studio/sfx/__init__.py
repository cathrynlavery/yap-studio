"""User-generated sound kits and their cache identities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from yap_studio import cache, elevenlabs, ffmpeg, receipt
from yap_studio.errors import YapError
from yap_studio.project.lock import pid_alive, project_lock

SCHEMA = "yap-studio/sfx-kit@1"
RULES = {"transition", "item", "count_start", "settle", "card_in", "card_out",
         "pop", "land", "chapter", "typing", "message", "photo", "request"}
AUTO_EXTRAS = ("typing", "message-ping", "camera-shutter")
_NAME = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")


def sfx_cache_root() -> Path:
    """Return the creator's sound cache without creating it for file-only reads."""
    return cache.cache_root() / "sfx"


class SfxError(YapError):
    """Explain a sound kit or paid generation that needs attention."""

    def __init__(self, message: str, *, code: str = "sfx_invalid",
                 suggested_action: str = "Correct the sound kit and try again.",
                 notices: list[str] | None = None) -> None:
        super().__init__(message, error_type="sfx_error", code=code,
                         suggested_action=suggested_action)
        self.notices = notices or []


def _check_sound(sound: Any, location: str) -> dict[str, Any]:
    fields = {"name", "prompt", "duration_s", "prompt_influence", "gain_db", "priority", "rule"}
    if not isinstance(sound, dict) or set(sound) != fields:
        raise SfxError(f"{location}: expected fields {', '.join(sorted(fields))}.")
    if not isinstance(sound["name"], str) or not _NAME.fullmatch(sound["name"]):
        raise SfxError(f"{location}.name: use lowercase letters, digits and hyphens.")
    if not isinstance(sound["prompt"], str) or not sound["prompt"].strip():
        raise SfxError(f"{location}.prompt: enter a sound description.")
    for key, low, high in (("duration_s", 0.5, 30), ("prompt_influence", 0, 1),
                           ("gain_db", -40, 12)):
        value = sound[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not low <= value <= high):
            raise SfxError(f"{location}.{key}: use a number from {low} to {high}.")
    if type(sound["priority"]) is not int or not 1 <= sound["priority"] <= 100:
        raise SfxError(f"{location}.priority: use an integer from 1 to 100.")
    if not isinstance(sound["rule"], str) or sound["rule"] not in RULES:
        raise SfxError(f"{location}.rule: choose {', '.join(sorted(RULES))}.")
    return sound


def _read_kit(path: Path, *, overlay: bool = False) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SfxError(f"Cannot read sound kit {path.name}: {exc}.") from exc
    required = {"schema", "name", "description", "sounds"}
    if not isinstance(data, dict) or set(data) != required or data.get("schema") != SCHEMA:
        raise SfxError(f"{path.name}: expected {SCHEMA} with name, description and sounds.")
    if not isinstance(data["name"], str) or not _NAME.fullmatch(data["name"]):
        raise SfxError(f"{path.name}.name: invalid kit name.")
    if not isinstance(data["description"], str) or not isinstance(data["sounds"], list):
        raise SfxError(f"{path.name}: description must be text and sounds must be a list.")
    names: set[str] = set()
    for index, sound in enumerate(data["sounds"]):
        _check_sound(sound, f"{path.name}.sounds[{index}]")
        if sound["name"] in names:
            raise SfxError(f"{path.name}: duplicate sound {sound['name']}.")
        names.add(sound["name"])
    if overlay and data["name"] != path.stem:
        raise SfxError(f"{path.name}: kit name differs from file name.")
    return data


def load_kit(name: str = "soft") -> dict[str, Any]:
    """Merge creator overrides over a shipped recipe, validating both."""
    if name == "made":
        raise SfxError("Made by Yap Studio is built in, not an ElevenLabs kit.",
                       code="sfx_builtin_kit", suggested_action="Use sfx list --kit made.")
    if not _NAME.fullmatch(name):
        raise SfxError("Kit name must use lowercase letters, digits and hyphens.")
    shipped = Path(__file__).parent / "kits" / f"{name}.json"
    overlay = sfx_cache_root() / "kits" / f"{name}.json"
    base = _read_kit(shipped) if shipped.is_file() else None
    custom = _read_kit(overlay, overlay=True) if overlay.is_file() else None
    if base is None and custom is None:
        raise SfxError(f"Unknown sound kit: {name}.", suggested_action="Use sfx list --kit soft.")
    sounds = {sound["name"]: sound for sound in (base or {}).get("sounds", [])}
    sounds.update({sound["name"]: sound for sound in (custom or {}).get("sounds", [])})
    return {"schema": SCHEMA, "name": name,
            "description": (custom or base)["description"], "sounds": list(sounds.values())}


def request(sound: dict[str, Any]) -> dict[str, Any]:
    """Return exactly the paid generation parameters from a recipe."""
    return {"text": sound["prompt"], "duration_seconds": sound["duration_s"],
            "prompt_influence": sound["prompt_influence"]}


def recipe_sha256(sound: dict[str, Any]) -> str:
    """Hash all recipe choices, including placement and gain."""
    data = json.dumps(sound, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _index() -> dict[str, str]:
    path = sfx_cache_root() / "generated.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def list_kit(name: str = "soft") -> dict[str, Any]:
    """Report only current request hashes as usable audio."""
    if name == "made":
        from yap_studio.sfx import made

        return made.listing()
    kit = load_kit(name)
    previous = _index()
    sounds = []
    for sound in kit["sounds"]:
        digest = elevenlabs.sound_request_sha256(request(sound))
        path = sfx_cache_root() / "audio" / f"{digest}.mp3"
        key = f"{name}/{sound['name']}"
        recipe_hash = recipe_sha256(sound)
        fresh = (path.is_file() and path.stat().st_size > 0 and
                 previous.get(key) == recipe_hash)
        peak = {}
        if fresh and name == "extras":
            try:
                peak = {"peak_s": cached_peak(path)}
            except (OSError, ValueError, YapError):
                fresh = False
        state = ("fresh" if fresh else "stale"
                 if key in previous and previous[key] != recipe_hash else "missing")
        sounds.append({**sound, **peak, "state": state, "cache_path": str(path),
                       "recipe_sha256": recipe_hash,
                       "request_sha256": digest,
                       "audio_sha256": receipt.sha256_file(path) if fresh else None})
    return {"kit": name, "description": kit["description"], "sounds": sounds}


def cached_peak(path: Path) -> float:
    """Measure the loudest decoded stereo sample on the mixer's 48 kHz clock."""
    import numpy as np

    result = ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                         "-map", "0:a:0", "-vn", "-ar", "48000", "-ac", "2",
                         "-f", "f32le", "-"], timeout_s=ffmpeg.DEFAULT_TIMEOUT_S, binary=True)
    audio = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Cached sound has no usable samples.")
    return int(np.argmax(np.max(np.abs(audio), axis=1))) / 48000


def signup(state: elevenlabs.AccountState) -> list[str]:
    """Show sound-specific setup and licensing terms."""
    first = (("Sound effects: your ElevenLabs account is on the free plan, "
              "so this video has no automatic sound effects.") if state == "free" else
             "Sound effects: ElevenLabs isn't set up, "
             "so this video has no automatic sound effects.",
             "ElevenLabs can make a small sound kit for your videos. "
             "Sound effects need a paid ElevenLabs plan.")
    lines = elevenlabs.signup_lines("sound effects", state, first_lines=first)[:-1]
    if state == "free":
        lines.insert(2, "Free-plan output is for non-commercial use only.")
    return lines


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as output:
            temporary = Path(output.name)
            output.write(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def remember_account(state: elevenlabs.AccountState) -> None:
    """Retain the last observed tier for file-only status checks."""
    root = sfx_cache_root()
    with project_lock(root):
        _write_json(root / "account.json", {"state": state})


def saved_account() -> str | None:
    """Read the last observed tier without invoking the CLI."""
    try:
        data = json.loads((sfx_cache_root() / "account.json").read_text())
    except (OSError, ValueError):
        return None
    return data.get("state") if isinstance(data, dict) else None


def _pending_error() -> SfxError:
    return SfxError("A sound generation may have been billed.", code="sfx_unreconciled",
                    suggested_action="Check ElevenLabs history, then pass --confirm-retry.")


def _pending_owner_alive(record: dict[str, Any]) -> bool:
    return (record.get("state") == "pending" and
            record.get("host") == socket.gethostname() and pid_alive(record.get("pid")))


def generate(name: str = "soft", *, yes: bool = False, dry_run: bool = False,
             confirm_retry: bool = False, only: str | None = None) -> dict[str, Any]:
    """Generate missing sounds after approval, preserving uncertain paid attempts."""
    listing = list_kit(name)
    if only is not None and only not in {sound["name"] for sound in listing["sounds"]}:
        raise SfxError(f"Unknown sound in {name}: {only}.")
    if name == "made":
        return {**listing, "generated": [], "status": "fresh",
                "notices": ["Made by Yap Studio needs no account."]}
    wanted = [sound for sound in listing["sounds"] if sound["state"] != "fresh"
              and (only is None or sound["name"] == only)]
    if not wanted:
        return {**listing, "generated": [], "status": "fresh"}
    if not yes and not dry_run:
        raise SfxError(f"Generating {len(wanted)} sound(s) spends ElevenLabs credits; pass --yes.",
                       code="sfx_approval_required", suggested_action="Run sfx kit --yes.")
    state = elevenlabs.account_state()
    if not dry_run:
        remember_account(state)
    if state == "missing_cli" or (not dry_run and state in {"signed_out", "free"}):
        lines = signup(state)
        raise SfxError(lines[0], code="sfx_account_required",
                       suggested_action="Set up a paid ElevenLabs plan.", notices=lines)
    if dry_run:
        previews = []
        for sound in wanted:
            with tempfile.TemporaryDirectory(prefix="yap-sfx-preview-") as folder:
                previews.append(elevenlabs.generate_sound(request(sound),
                                Path(folder) / "preview.mp3", dry_run=True))
        notices = signup(state) if state in {"signed_out", "free"} else []
        return {**listing, "status": "dry_run", "requests": previews, "generated": [],
                "notices": notices}
    root = sfx_cache_root()
    pending = root / "pending.json"
    generated = []
    for original in wanted:
        with project_lock(root):
            sound = next(item for item in list_kit(name)["sounds"]
                         if item["name"] == original["name"])
            if sound["state"] == "fresh":
                continue
            if pending.is_file():
                previous = json.loads(pending.read_text(encoding="utf-8"))
                if not confirm_retry or _pending_owner_alive(previous):
                    raise _pending_error()
                pending.unlink()
            body = request(sound)
            record = {"schema": "yap-studio/sfx-pending@1", "state": "pending",
                      "kit": name, "sound": sound["name"],
                      "request_sha256": sound["request_sha256"], "request": body,
                      "pid": os.getpid(), "host": socket.gethostname(),
                      "started": datetime.now(UTC).isoformat()}
            target = Path(sound["cache_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size > 0:
                try:
                    if name == "extras":
                        cached_peak(target)
                        valid_cached_audio = True
                    else:
                        valid_cached_audio = ffmpeg.probe(target, count_frames=False).has_audio
                except (OSError, ValueError, YapError):
                    valid_cached_audio = False
                if valid_cached_audio:
                    index = _index()
                    index[f"{name}/{sound['name']}"] = sound["recipe_sha256"]
                    _write_json(root / "generated.json", index)
                    continue
            _write_json(pending, record)
        with tempfile.TemporaryDirectory(prefix="yap-sfx-generate-",
                                         dir=root) as folder:
            staged = Path(folder) / "generated.mp3"
            try:
                elevenlabs.generate_sound(body, staged, dry_run=False)
                if not staged.is_file() or staged.stat().st_size == 0:
                    raise ValueError("empty sound output")
                if not ffmpeg.probe(staged, count_frames=False).has_audio:
                    raise ValueError("sound output has no audio stream")
                if name == "extras":
                    cached_peak(staged)
                with project_lock(root):
                    shutil.copyfile(staged, target)
                    index = _index()
                    index[f"{name}/{sound['name']}"] = sound["recipe_sha256"]
                    _write_json(root / "generated.json", index)
                    pending.unlink()
            except elevenlabs.GenerationError as exc:
                if exc.refused:
                    with project_lock(root):
                        pending.unlink(missing_ok=True)
                    lines = signup("free")
                    raise SfxError(
                        lines[0], code="sfx_paid_plan_required",
                        suggested_action="Upgrade to a paid plan.", notices=lines,
                    ) from exc
                with project_lock(root):
                    _write_json(pending, {**record, "state": "unreconciled",
                                          "error": str(exc)[-500:]})
                raise SfxError(
                    "Sound generation did not finish and may have been billed.",
                    code="sfx_unreconciled",
                    suggested_action="Check ElevenLabs history, then pass --confirm-retry.",
                ) from exc
            except (OSError, ValueError, YapError) as exc:
                with project_lock(root):
                    _write_json(pending, {**record, "state": "unreconciled",
                                          "error": str(exc)[-500:]})
                raise SfxError(
                    "Sound generation did not finish and may have been billed.",
                    code="sfx_unreconciled",
                    suggested_action="Check ElevenLabs history, then pass --confirm-retry.",
                ) from exc
        generated.append(sound["name"])
    return {**list_kit(name), "generated": generated,
            "status": "verified" if generated else "fresh"}


def add(name: str, prompt: str, duration_s: float, rule: str, *, kit: str = "soft",
        gain_db: float = 0, priority: int = 1, yes: bool = False,
        dry_run: bool = False) -> dict[str, Any]:
    """Save an authored recipe in the user cache and generate only that sound."""
    if kit == "made":
        raise SfxError("Made by Yap Studio is a built-in kit.", code="sfx_builtin_kit",
                       suggested_action="Choose an ElevenLabs kit with --kit soft.")
    sound = _check_sound({"name": name, "prompt": prompt, "duration_s": duration_s,
                          "prompt_influence": 0.5, "gain_db": gain_db,
                          "priority": priority, "rule": rule}, "sound")
    load_kit(kit)
    if not yes and not dry_run:
        raise SfxError("Generating this sound spends ElevenLabs credits; pass --yes.",
                       code="sfx_approval_required", suggested_action="Run sfx add with --yes.")
    if dry_run:
        state = elevenlabs.account_state()
        if state == "missing_cli":
            lines = signup(state)
            raise SfxError(lines[0], code="sfx_account_required",
                           suggested_action="Set up a paid ElevenLabs plan.", notices=lines)
        with tempfile.TemporaryDirectory(prefix="yap-sfx-preview-") as folder:
            preview = elevenlabs.generate_sound(request(sound),
                                                Path(folder) / "preview.mp3", dry_run=True)
        return {"status": "dry_run", "sound": sound, "request": preview,
                "notices": signup(state) if state in {"signed_out", "free"} else []}
    root = sfx_cache_root()
    overlay_path = root / "kits" / f"{kit}.json"
    with project_lock(root):
        overlay = (_read_kit(overlay_path, overlay=True) if overlay_path.is_file() else
                   {"schema": SCHEMA, "name": kit, "description": "Creator sounds", "sounds": []})
        overlay["sounds"] = [item for item in overlay["sounds"] if item["name"] != name] + [sound]
        _write_json(overlay_path, overlay)
    return generate(kit, yes=yes, only=name)
