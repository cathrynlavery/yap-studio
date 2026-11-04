"""Small ElevenLabs CLI adapter shared by generated audio stages."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from yap_studio import paths, tools
from yap_studio.affiliate import ELEVENLABS_AFFILIATE_URL
from yap_studio.errors import ProcessingError, YapError
from yap_studio.project.lock import project_lock

AccountState = Literal["missing_cli", "signed_out", "free", "paid", "unknown"]
MUSIC_MODEL = "music_v2"
SOUND_MODEL = "eleven_text_to_sound_v2"
MUSIC_FORMAT = "mp3_44100_128"
GENERATION_TIMEOUT_S = 4 * 600 + 300
SOUND_GENERATION_TIMEOUT_S = 4 * 120 + 60
VOICE_GENERATION_TIMEOUT_S = 4 * 120 + 60
INSTALL_LINE = ("Install the ElevenLabs CLI: "
                "npx -y @mvanhorn/printing-press-library install elevenlabs --cli-only")
TOKEN_LINE = ("Then create your own ElevenLabs API key with Music, Sound Effects, Text to Speech, "
              "Voices, History, Models and User read access, and save it: "
              "elevenlabs-pp-cli auth set-token <your-key>")
SIGNUP_LINE = ("Sign up here (affiliate link: Yap Studio's maker earns a commission "
               "if you subscribe):")


class GenerationError(Exception):
    """Distinguish a refused paid request from an uncertain attempt."""

    def __init__(self, message: str, *, refused: bool = False) -> None:
        super().__init__(message)
        self.refused = refused


def account_state() -> AccountState:
    """Read auth and plan tier without handling the credential itself."""
    if tools.find_tool("elevenlabs-pp-cli") is None:
        return "missing_cli"
    try:
        tools.run_tool("elevenlabs-pp-cli", ["auth", "status", "--json"], timeout_s=30)
    except ProcessingError as exc:
        return "signed_out" if exc.returncode == 4 else "unknown"
    try:
        result = tools.run_tool("elevenlabs-pp-cli",
                                ["user", "get-subscription-info", "--json"], timeout_s=60)
        data = json.loads(result.stdout)
    except (YapError, ValueError):
        return "unknown"
    if not isinstance(data, dict):
        return "unknown"
    subscription = data.get("subscription")
    tier = data.get("tier") or (subscription.get("tier")
                                if isinstance(subscription, dict) else None)
    return "free" if str(tier).lower() == "free" else "paid"


def signup_lines(
    product: str, state: AccountState, *, first_lines: tuple[str, str] | None = None,
) -> list[str]:
    """Build a disclosed signup message for an audio product."""
    free = state == "free"
    if first_lines is None:
        first_lines = (
            f"{product.capitalize()}: your ElevenLabs account is on the free plan, "
            "so this video has no generated music." if free else
            f"{product.capitalize()}: ElevenLabs isn't set up, "
            "so this video has no generated music.",
            "ElevenLabs can compose music to fit your video. "
            "Music needs a paid ElevenLabs plan.",
        )
    lines = [*first_lines]
    if state == "missing_cli":
        lines.append(INSTALL_LINE)
    lines.extend([(SIGNUP_LINE.replace("Sign up", "Upgrade") if free else SIGNUP_LINE),
                  f"  {ELEVENLABS_AFFILIATE_URL}"])
    if not free:
        lines.append(TOKEN_LINE)
    lines.append("Or use your own track: yap-studio music --file song.mp3")
    return lines


def doctor_setup_lines(*, cli_available: bool) -> list[str]:
    """Return optional account setup notes for doctor."""
    return ([INSTALL_LINE] if not cli_available else []) + [
        SIGNUP_LINE, f"  {ELEVENLABS_AFFILIATE_URL}", TOKEN_LINE,
    ]


def show_once(project_dir: Path, key: str, lines: list[str]) -> list[str]:
    """Return and record the first notice for a project key."""
    with paths.operation(), project_lock(project_dir):
        target = project_dir / ".yap/notices.json"
        try:
            notices = json.loads(target.read_text(encoding="utf-8"))
        except FileNotFoundError:
            notices = {"schema": "yap-studio/notices@1"}
        if notices.get(key):
            return []
        notices[key] = True
        paths.atomic_write_text(target, json.dumps(notices, indent=2) + "\n")
    return lines


def _payload(body: dict[str, Any]) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def redact_cli_text(text: str | None) -> str:
    """Remove credential header lines from ElevenLabs CLI stderr."""
    return "\n".join(line for line in (text or "").splitlines()
                     if "api-key" not in line.lower())


def compose_plan(body: dict[str, Any], *, dry_run: bool) -> dict[str, Any] | str:
    """Request a free composition plan or its CLI preview."""
    args = ["music", "compose-plan", "--stdin", "--json", "--timeout", "2m"]
    if dry_run:
        args.append("--dry-run")
    try:
        result = tools.run_tool("elevenlabs-pp-cli", args, timeout_s=180,
                                input_bytes=_payload(body))
    except ProcessingError as exc:
        raise ProcessingError(exc.command, exc.returncode,
                              redact_cli_text(exc.full_stderr), code=exc.code) from None
    if dry_run:
        return redact_cli_text(result.stderr)
    data = json.loads(result.stdout)
    return data.get("data", data)


def generate_music(body: dict[str, Any], target: Path, *, dry_run: bool) -> str | None:
    """Deliver paid audio without capturing its binary stdout."""
    args = ["music", "generate", "--stdin", "--timeout", "10m", "--output-format",
            MUSIC_FORMAT, "--deliver", f"file:{target}"]
    if dry_run:
        args.append("--dry-run")
    try:
        result = tools.run_tool("elevenlabs-pp-cli", args,
                                timeout_s=GENERATION_TIMEOUT_S, input_bytes=_payload(body),
                                stdout_path=(None if dry_run else
                                             target.with_name(target.name + ".stdout")))
    except ProcessingError as exc:
        tail = redact_cli_text(exc.full_stderr)[-500:]
        refused = "403" in tail and "subscription required" in tail.lower()
        raise GenerationError(tail or "ElevenLabs CLI failed.", refused=refused) from None
    return redact_cli_text(result.stderr) if dry_run else None


def request_sha256(body: dict[str, Any]) -> str:
    """Identify the exact paid request and output encoding."""
    return hashlib.sha256(_payload({"body": body, "output_format": MUSIC_FORMAT})).hexdigest()


def sound_request_sha256(request: dict[str, Any]) -> str:
    """Identify the exact sound request, model and output encoding."""
    return hashlib.sha256(_payload({"request": request, "model": SOUND_MODEL,
                                    "output_format": MUSIC_FORMAT})).hexdigest()


def generate_sound(request: dict[str, Any], target: Path, *, dry_run: bool) -> str | None:
    """Deliver a paid sound while diverting duplicate binary stdout to a file."""
    args = ["sound-generation", "--model-id", SOUND_MODEL,
            "--text", request["text"], "--duration-seconds",
            str(request["duration_seconds"]), "--prompt-influence",
            str(request["prompt_influence"]), "--output-format", MUSIC_FORMAT,
            "--timeout", "2m", "--deliver", f"file:{target}"]
    if dry_run:
        args.append("--dry-run")
    try:
        result = tools.run_tool(
            "elevenlabs-pp-cli", args, timeout_s=SOUND_GENERATION_TIMEOUT_S,
            stdout_path=None if dry_run else target.with_name(target.name + ".stdout"),
        )
    except ProcessingError as exc:
        tail = redact_cli_text(exc.full_stderr)[-500:]
        refused = "403" in tail and "subscription required" in tail.lower()
        raise GenerationError(tail or "ElevenLabs CLI failed.", refused=refused) from None
    return redact_cli_text(result.stderr) if dry_run else None


VOICE_MODEL = "eleven_multilingual_v2"


def _voice_permission(stderr: str) -> None:
    """Report the needed scope without retaining any command or identifier."""
    from yap_studio.voice.model import VoiceError

    match = re.search(r"missing the permission\s+['\"]?([\w.:-]+)", stderr, re.IGNORECASE)
    if match is None:
        return
    permission = match[1]
    plain = {"user_read": "User read", "models_read": "Models read",
             "voices_read": "Voices read", "voices_write": "Voices write",
             "text_to_speech": "Text to Speech", "speech_history_read": "History read"}
    action = ("In ElevenLabs, open Developers, then API keys, edit your key, turn on "
              f"{plain.get(permission, permission)}, then try again.")
    raise VoiceError(f"Your ElevenLabs key is missing the {permission} permission. {action}",
                     code="elevenlabs_permission_missing", suggested_action=action,
                     permission=permission) from None


def _voice_call(args: list[str], *, voice_id: str | None = None,
                stdout_path: Path | None = None, timeout_s: float = GENERATION_TIMEOUT_S) -> Any:
    """Keep voice identifiers and command lines out of all raised errors."""
    try:
        result = tools.run_tool("elevenlabs-pp-cli", args, timeout_s=timeout_s,
                                stdout_path=stdout_path)
    except YapError as exc:
        stderr = redact_cli_text(getattr(exc, "full_stderr", ""))
        _voice_permission(stderr)
        statuses = re.findall(r"\bHTTP\s+(\d{3})\b", stderr, re.IGNORECASE)
        status = int(statuses[-1]) if statuses else None
        refused = (isinstance(exc, ProcessingError) and exc.returncode == 4
                   or status is not None and 400 <= status < 500)
        if (args[:2] == ["voices", "add"] and status in (400, 422)
                and "files" in stderr.lower()):
            raise YapError("The ElevenLabs CLI cannot upload voice samples yet.",
                           error_type="input_error", code="voice_cli_cannot_upload") from None
        tail = stderr[-500:]
        if voice_id:
            tail = tail.replace(voice_id, "<voice>")
        # Free reads may echo identifiers that have not been parsed yet.
        if voice_id is None and args[:1] in (["voices"], ["history"]):
            tail = ""
        raise GenerationError(tail or "ElevenLabs voice request failed.", refused=refused) from None
    return result


def _voice_json(args: list[str], *, voice_id: str | None = None) -> Any:
    result = _voice_call(args, voice_id=voice_id)
    try:
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = data.get("data", data)
            if isinstance(data, dict) and "results" in data:
                data = data["results"]
        return data
    except (TypeError, ValueError):
        raise GenerationError("ElevenLabs returned unreadable voice data.", refused=True) from None


def personal_voices(name: str) -> list[dict[str, Any]]:
    """Find exact personal names without retaining the response."""
    data = _voice_json(["voices", "get-user-v2", "--search", name,
                        "--voice-type", "personal", "--all", "--json"])
    rows = data.get("voices", []) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise GenerationError("ElevenLabs returned unreadable voices.", refused=True)
    return [row for row in rows if isinstance(row, dict) and row.get("name") == name]


def voice_details(voice_id: str) -> dict[str, Any]:
    """Read sample metadata for one in-memory identifier."""
    data = _voice_json(["voices", "get-by-id", voice_id, "--json"], voice_id=voice_id)
    if not isinstance(data, dict):
        raise GenerationError("ElevenLabs returned unreadable voice details.", refused=True)
    return data


def add_voice(name: str, files: list[Path]) -> None:
    """Use the documented JSON file list; report the current upload limitation."""
    args = ["voices", "add", "--name", name, "--files",
            json.dumps([str(item.absolute()) for item in files]), "--description",
            "Own recordings for this project's voice fixes.", "--labels",
            json.dumps({"source": "yap-studio"})]
    _voice_call(args)


def remove_voice(voice_id: str) -> None:
    """Delete one resolved voice without echoing its identifier."""
    _voice_call(["voices", "delete", voice_id], voice_id=voice_id)


def voice_budget(model_id: str = VOICE_MODEL) -> dict[str, Any]:
    """Read the model multiplier and monthly balance without paid calls."""
    from yap_studio.voice.model import VoiceError

    multiplier, source, remaining = 1.0, "estimate", None
    unavailable = {}
    try:
        data = _voice_json(["models", "--json"])
        rows = data.get("models", []) if isinstance(data, dict) else data
        for model in rows:
            if model.get("model_id") == model_id:
                value = float(model["model_rates"]["character_cost_multiplier"])
                if 0 < value < float("inf"):
                    multiplier, source = value, "models"
    except VoiceError as exc:
        if exc.code != "elevenlabs_permission_missing":
            raise
        unavailable["models"] = exc.to_dict()
    except (GenerationError, TypeError, ValueError, KeyError, AttributeError):
        unavailable["models"] = {"code": "elevenlabs_read_failed",
                                  "message": "ElevenLabs models could not be read."}
    try:
        data = _voice_json(["user", "get-subscription-info", "--json"])
        data = data.get("subscription", data)
        remaining = max(0, int(data["character_limit"]) - int(data["character_count"]))
    except VoiceError as exc:
        if exc.code != "elevenlabs_permission_missing":
            raise
        unavailable["plan"] = exc.to_dict()
    except (GenerationError, TypeError, ValueError, KeyError, AttributeError):
        unavailable["plan"] = {"code": "elevenlabs_read_failed",
                                "message": "Your ElevenLabs plan could not be read."}
    return {"multiplier": multiplier, "source": source, "remaining": remaining,
            **({"unavailable": unavailable} if unavailable else {})}


def generate_voice(voice_id: str, request: dict[str, Any], target: Path) -> None:
    """Deliver speech with binary stdout diverted to a temporary file."""
    args = ["text-to-speech", voice_id, "--text", request["text"],
            "--model-id", request.get("model", VOICE_MODEL)]
    for key in ("previous_text", "next_text"):
        if key in request:
            args.extend(["--" + key.replace("_", "-"), request[key]])
    args.extend(["--seed", str(request["seed"]), "--output-format", MUSIC_FORMAT,
                 "--timeout", "2m", "--deliver", f"file:{target}"])
    if "voice_settings" in request:
        args.extend(["--voice-settings", json.dumps(request["voice_settings"])])
    _voice_call(args,
                voice_id=voice_id, stdout_path=target.with_suffix(".stdout"),
                timeout_s=VOICE_GENERATION_TIMEOUT_S)


def voice_charge(voice_id: str, text: str, started: float, estimate: float) -> dict[str, Any]:
    """Use the newest matching history item, falling back to the disclosed estimate."""
    try:
        data = _voice_json(["history", "get-speech", "--voice-id", voice_id,
                            "--page-size", "20", "--json"], voice_id=voice_id)
        rows = data.get("history", data.get("items", [])) if isinstance(data, dict) else data
        matches = [row for row in rows if row.get("text") == text
                   and float(row["date_unix"]) >= started]
        for row in sorted(matches, key=lambda row: float(row["date_unix"]), reverse=True):
            credits = row["character_count_change_to"] - row["character_count_change_from"]
            if credits >= 0:
                return {"credits": credits, "source": "history"}
    except (YapError, GenerationError, OSError, TypeError, ValueError, KeyError, AttributeError):
        pass
    return {"credits": estimate, "source": "estimate"}
