"""Small ElevenLabs CLI adapter shared by generated audio stages."""

from __future__ import annotations

import hashlib
import json
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
INSTALL_LINE = ("Install the ElevenLabs CLI: "
                "npx -y @mvanhorn/printing-press-library install elevenlabs --cli-only")
TOKEN_LINE = ("Then create your own ElevenLabs API key with Music, Sound Effects and "
              "User read access, and save it: "
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
