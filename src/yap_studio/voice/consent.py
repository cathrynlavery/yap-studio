"""Own-voice consent and exact-name resolution."""

from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

from yap_studio import elevenlabs, paths, receipt
from yap_studio.errors import YapError
from yap_studio.project.lock import project_lock, timestamp
from yap_studio.voice.model import (
    STATEMENT,
    VoiceError,
    checked_file,
    digest,
    read,
    validate_samples_manifest,
    write,
)
from yap_studio.voice.recipe import load_recipe


def require_consent(root: Path, *, linked: bool = True) -> dict:
    data = read(root, "consent", optional=True)
    if data is None or data["status"] != "given":
        raise VoiceError(
            "Your own-voice consent is needed first.",
            code="voice_consent_required",
            suggested_action="Read the statement, then run voice consent DIR --yes.",
        )
    if linked and not data["voice"].get("id_sha256"):
        raise VoiceError("Link or create your voice first.", code="voice_not_linked")
    return data


def require_account() -> None:
    state = elevenlabs.account_state()
    if state in {"missing_cli", "signed_out", "free"}:
        lines = elevenlabs.signup_lines(
            "voice fixes",
            state,
            first_lines=(
                "Voice fixes need a paid ElevenLabs plan."
                if state == "free"
                else "ElevenLabs is not set up for voice fixes.",
                "ElevenLabs can re-say a misspoken word using your own recordings, "
                "with your consent.",
            ),
        )
        lines[-1] = "Or record the line again in your own voice."
        raise VoiceError(
            lines[0],
            code="voice_account_required",
            notices=lines,
            suggested_action="Set up your paid ElevenLabs account for voice fixes.",
        )


def consent(
    project_dir: Path,
    *,
    yes: bool = False,
    by: str | None = None,
    voice_name: str | None = None,
    withdraw: bool = False,
) -> dict:
    """Record the exact statement or withdraw permission for future cuts."""
    root = Path(project_dir)
    with paths.operation(), project_lock(root):
        previous = read(root, "consent", optional=True)
        if withdraw:
            if previous is None:
                raise VoiceError("There is no voice consent to withdraw.")
            data = {**previous, "status": "withdrawn", "withdrawn_at": timestamp()}
        else:
            if not yes:
                raise VoiceError(
                    STATEMENT,
                    code="voice_consent_required",
                    suggested_action="If you agree, run voice consent DIR --yes.",
                )
            samples = read(root, "samples", optional=True) or {"samples": []}
            for row in samples["samples"]:
                checked_file(root, row)
            name = (
                voice_name
                or load_recipe()["voice_name"]
                or (previous or {}).get("voice", {}).get("name")
                or "My voice (Yap Studio)"
            ).strip()
            if not name or not (by or "Creator").strip():
                raise VoiceError("Give your name and a voice name.")
            voice = {"name": name, "id_sha256": None, "made": None, "at": None}
            if (
                previous
                and previous["voice"]["name"] == name
                and previous["samples"] == samples["samples"]
            ):
                voice = previous["voice"]
            data = {
                "schema": "yap-studio/voice-consent@1",
                "status": "given",
                "statement": STATEMENT,
                "given_by": (by or "Creator").strip(),
                "given_at": timestamp(),
                "voice": voice,
                "samples": samples["samples"],
                "withdrawn_at": None,
            }
        write(root, "consent", data)
    return data


def _find(name: str) -> dict:
    try:
        matches = elevenlabs.personal_voices(name)
        if len(matches) != 1:
            raise VoiceError(
                "Exactly one personal voice must have this name.", code="voice_name_ambiguous"
            )
        voice = matches[0]
        identifier = voice.get("voice_id")
        if not isinstance(identifier, str) or not identifier:
            raise VoiceError("ElevenLabs returned an unreadable voice.")
        if voice.get("category") != "cloned":
            raise VoiceError(
                "This voice was not cloned from your recordings.", code="voice_not_cloned"
            )
        return voice
    except elevenlabs.GenerationError:
        raise VoiceError(
            "Your personal voices could not be read.", code="voice_lookup_failed"
        ) from None


def resolve_voice(root: Path, *, allow_withdrawn: bool = False) -> str:
    """Return an identifier only in memory, refusing a replaced voice."""
    record = read(root, "consent") if allow_withdrawn else require_consent(root)
    return _resolve_voice(record)


def _resolve_voice(record: dict) -> str:
    voice = _find(record["voice"]["name"])
    identifier = voice["voice_id"]
    if digest(identifier) != record["voice"].get("id_sha256"):
        raise VoiceError(
            "The voice with this name changed. Link your own voice again.", code="voice_changed"
        )
    return identifier


def _sample_files(root: Path, base: dict, manifest: Path | None, *,
                  max_files: int = 25) -> tuple[list[Path], dict | None]:
    """Check either the consent's project samples or a manifest's own WAV files."""
    if manifest is None:
        rows = base["samples"] or (read(root, "samples", optional=True) or {}).get("samples", [])
        if not rows:
            raise VoiceError(
                "Export this project's samples first with yap-studio voice samples DIR, "
                "or pass --samples.",
                code="voice_samples_missing",
            )
        files = [checked_file(root, row) for row in rows]
        if any("size_bytes" in row and sample.stat().st_size != row["size_bytes"]
               for sample, row in zip(files, rows, strict=True)):
            raise VoiceError("Voice samples changed. Read them again before linking.")
        return files, None
    target = Path(manifest).expanduser().absolute()
    try:
        paths.validate_input(target)
        content = target.read_bytes()
        data = validate_samples_manifest(json.loads(content), max_files=max_files)
        folder = target.parent.resolve()
        files = []
        for row in data["samples"]:
            sample = folder / row["path"]
            if (not sample.is_file() and folder.name == "voice"
                    and row["path"].startswith("voice/samples/")):
                sample = folder.parent / row["path"]
            if not sample.resolve().is_relative_to(folder) or not sample.is_file():
                raise ValueError("sample path must stay inside the manifest folder")
            paths.validate_input(sample)
            if (sample.stat().st_size != row["size_bytes"]
                    or receipt.sha256_file(sample) != row["sha256"]):
                raise ValueError("sample is missing or changed")
            with sample.open("rb") as stream:
                header = stream.read(12)
            if header[:4] not in (b"RIFF", b"RF64") or header[8:] != b"WAVE":
                raise ValueError("sample must be a WAV file")
            files.append(sample)
        return files, {"sha256": hashlib.sha256(content).hexdigest(), "files": len(files)}
    except (YapError, OSError, ValueError, TypeError, KeyError, AttributeError):
        raise VoiceError(
            f"Samples manifest {target} is not valid.", code="voice_samples_invalid",
            suggested_action="Check its WAV paths, hashes, sizes and sample ranges.",
        ) from None


def _attach(root: Path, name: str, *, made: str, base: dict, samples: Path | None = None,
            selection: tuple[list[Path], dict | None]) -> dict:
    voice = _find(name)
    identifier = voice["voice_id"]
    try:
        details = elevenlabs.voice_details(identifier)
    except elevenlabs.GenerationError:
        raise VoiceError("Your voice samples could not be checked.") from None
    if details.get("category") != "cloned":
        raise VoiceError("This voice was not cloned from your recordings.", code="voice_not_cloned")
    max_files = 25
    files, manifest = _sample_files(root, base, samples, max_files=max_files)
    if (files, manifest) != selection:
        raise VoiceError("Voice samples changed. Read them again before linking.")
    expected = sorted((sample.stem, sample.stat().st_size) for sample in files)
    try:
        actual = sorted((Path(row["file_name"]).stem, row["size_bytes"])
                        for row in details["samples"])
    except (KeyError, TypeError):
        actual = []
    if (expected != actual or len({stem for stem, _ in expected}) != len(expected)
            or len({stem for stem, _ in actual}) != len(actual)):
        raise VoiceError(
            (f"The voice's sample stems or sizes differ from {samples}." if samples is not None
             else "The voice's sample stems or sizes differ from this project's samples."),
            code="voice_samples_differ",
            suggested_action=(f"Upload exactly the files listed in {samples}."
                              if samples is not None else
                              "Upload exactly these files: "
                              + ", ".join(str(row) for row in files) + "."),
        )
    with project_lock(root):
        if require_consent(root, linked=False) != base:
            raise VoiceError("Voice consent changed. Read it again before linking.")
        if _sample_files(root, base, samples, max_files=max_files) != selection:
            raise VoiceError("Voice samples changed. Read them again before linking.")
        if samples is None and digest(identifier) == base["voice"]["id_sha256"]:
            manifest = base["voice"].get("samples_manifest")
        data = {
            **base,
            "samples": (base["samples"] or read(root, "samples")["samples"]
                        if samples is None else base["samples"]),
            "voice": {
                "name": name,
                "id_sha256": digest(identifier),
                "made": made,
                "at": timestamp(),
                **({"samples_manifest": manifest} if manifest is not None else {}),
            },
        }
        write(root, "consent", data)
    return data


def link(project_dir: Path, *, name: str | None = None, yes: bool = False,
         samples: Path | None = None) -> dict:
    """Link only a personal clone made from the exact selected sample files."""
    root = Path(project_dir)
    if not yes:
        raise VoiceError(
            "Linking your voice needs your approval; pass --yes.", code="voice_approval_required"
        )
    with paths.operation():
        base = require_consent(root, linked=False)
        name = name or load_recipe()["voice_name"] or base["voice"]["name"]
        selection = _sample_files(root, base, samples)
        require_account()
        return _attach(root, name, made="linked", base=base, samples=samples, selection=selection)


def create(project_dir: Path, *, yes: bool = False, dry_run: bool = False,
           samples: Path | None = None) -> dict:
    """Create a personal clone with consent and an explicit voice-slot approval."""
    root = Path(project_dir)
    with paths.operation():
        base = require_consent(root, linked=False)
        if not yes and not dry_run:
            raise VoiceError(
                "Creating your voice uses one ElevenLabs voice slot; pass --yes.",
                code="voice_approval_required",
            )
        selection = _sample_files(root, base, samples, max_files=25)
        files, _ = selection
        require_account()
        if dry_run:
            return {
                "status": "dry_run",
                "name": base["voice"]["name"],
                "voice_slots": 1,
                "samples": [str(row) if samples is not None else str(row.relative_to(root))
                            for row in files],
            }
        name = base["voice"]["name"]
        try:
            if elevenlabs.personal_voices(name):
                raise VoiceError(
                    "A voice with this name already exists. Link it or choose a new name.",
                    code="voice_name_ambiguous",
                )
            with project_lock(root):
                if require_consent(root, linked=False) != base:
                    raise VoiceError("Voice consent changed. Read it again before uploading.")
                try:
                    current = _sample_files(root, base, samples, max_files=25)
                except (YapError, OSError):
                    raise VoiceError(
                        "Voice samples changed. Read them again before uploading."
                    ) from None
                if current != selection:
                    raise VoiceError("Voice samples changed. Read them again before uploading.")
                elevenlabs.add_voice(name, files)
        except YapError as exc:
            if exc.code != "voice_cli_cannot_upload":
                raise
            upload = (f"the files listed in {samples}: " + ", ".join(str(row) for row in files)
                      if samples is not None else
                      "these files: " + ", ".join(str(row) for row in files))
            link_args = ["yap-studio", "voice", "link", str(root), "--name", name, "--yes"]
            if samples is not None:
                link_args.extend(["--samples", str(samples)])
            link_command = shlex.join(link_args)
            raise VoiceError(
                str(exc),
                code=exc.code,
                suggested_action=(
                    "Open ElevenLabs, then Voices, then Instant Voice Clone. "
                    f"Upload {upload}. Name it {name}. Then run {link_command}."
                ),
            ) from None
        except elevenlabs.GenerationError:
            raise VoiceError(
                "ElevenLabs could not create your voice.", code="voice_create_failed"
            ) from None
        return _attach(root, name, made="yap-studio", base=base,
                       samples=samples, selection=selection)


def delete_voice(project_dir: Path, *, yes: bool = False) -> dict:
    """Delete the linked personal clone and withdraw its project consent."""
    root = Path(project_dir)
    with paths.operation(), project_lock(root):
        record = read(root, "consent", optional=not yes)
        if record is not None and record["voice"].get("samples_manifest") is not None:
            raise VoiceError(
                "This voice was made from recordings across your videos, so Yap Studio will not "
                "delete it. Delete it on the ElevenLabs site if you mean to.",
                code="voice_shared",
            )
        if not yes:
            raise VoiceError(
                "Deleting your ElevenLabs voice needs --yes.", code="voice_approval_required"
            )
        identifier = _resolve_voice(record)
        if read(root, "consent") != record:
            raise VoiceError("Voice consent changed. Read it again before deleting.")
        try:
            elevenlabs.remove_voice(identifier)
        except elevenlabs.GenerationError:
            raise VoiceError(
                "ElevenLabs could not delete your voice.", code="voice_delete_failed"
            ) from None
        write(root, "consent", {**record, "status": "withdrawn", "withdrawn_at": timestamp()})
    return {"status": "deleted"}
