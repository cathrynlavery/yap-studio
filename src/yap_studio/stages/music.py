"""Prepare a normalised, sample-exact music stem with a silent end hold."""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np

from yap_studio import elevenlabs, ffmpeg, paths, receipt
from yap_studio.errors import YapError
from yap_studio.plan import snap_frame
from yap_studio.project.lock import ConflictError, pid_alive, project_lock
from yap_studio.project.model import Edit, parse_document
from yap_studio.project.receipts import StageRun, projection_sha256, read_receipt
from yap_studio.project.status import ProjectContext
from yap_studio.render import hyperframes
from yap_studio.render.beats import BeatsError, detect_beats
from yap_studio.stages.timeline import FPS, _fresh_cut

RATE = 48_000
SAMPLES_PER_FRAME = 1600


class MusicError(YapError):
    """Report unusable music inputs with a concrete repair."""

    def __init__(self, message: str, *, code: str, suggested_action: str,
                 notices: list[str] | None = None) -> None:
        super().__init__(message, error_type="input_error", code=code,
                         suggested_action=suggested_action)
        self.notices = notices or []


@dataclass(frozen=True)
class MusicSnapshot:
    """The source identity, output clock and loudness used by one stage run."""

    mode: str
    source: Path | None
    source_sha256: str | None
    cut_frames: int
    end_hold_frames: int
    target_lufs: float
    target_source: str
    dry_open_s: float
    beat_snap: bool
    inputs: dict[str, str]

    @property
    def cut_samples(self) -> int:
        """Return the exact content length at the delivery sample rate."""
        return self.cut_frames * SAMPLES_PER_FRAME

    @property
    def final_samples(self) -> int:
        """Return the content plus silent hold length."""
        return (self.cut_frames + self.end_hold_frames) * SAMPLES_PER_FRAME


def music_snapshot(context: ProjectContext, *, require_fresh_cut: bool = True) -> MusicSnapshot:
    """Project music inputs without assembly level, ducking or timeline windows."""
    settings = context.project.music
    requested = settings["mode"]
    mode = ("elevenlabs" if requested == "elevenlabs" else "file"
            if requested == "file" or (requested == "auto" and settings["file"]) else "none")
    source = None
    digest = None
    if mode == "file":
        if not settings["file"]:
            raise MusicError(
                "File mode needs music.file.", code="music_file_missing",
                suggested_action="Set music.file to a track path, or set music.mode to none.",
            )
        source = paths.validate_input(context.project_dir / settings["file"])
        digest = context.identity.sha256(source)
    style = context.render_style().data
    internal = {key: value for key, value in style.get("internal", {}).items()
                if key.startswith("music_")}
    target = float(internal.get("music_lufs", -16))
    target_source = "style.internal.music_lufs" if "music_lufs" in internal else "default"
    dry_open_s = float(style["music"]["dry_open_s"])
    beat_snap = style["music"]["beat_snap"]
    cut = (_fresh_cut(context) if require_fresh_cut else context.json("cut/receipt.json"))
    edit = cast(Edit, parse_document("edit", context.json("edit.json")))
    cut_frames = cut["plan"]["total_frames"]
    hold_s = edit.end_hold_s if edit.end_hold_s is not None else style["internal"]["end_hold_s"]
    hold_frames = snap_frame(Fraction(str(hold_s)), FPS)
    inputs = {
        "project:music": projection_sha256(settings),
        "cut:frames": projection_sha256(cut_frames),
        "end_hold:frames": projection_sha256(hold_frames),
        "style:internal:music": projection_sha256(internal),
        "style:music:beat_snap": projection_sha256(beat_snap),
        **context.taste_inputs("music.beat_snap"),
    }
    if mode == "elevenlabs":
        plan_path = context.project_dir / "music/plan.json"
        inputs["music:plan"] = (receipt.sha256_file(plan_path) if plan_path.is_file()
                                else "absent")
        if plan_path.is_file():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            body = _generation_body(plan)
            request_hash = elevenlabs.request_sha256(body)
            inputs["music:request"] = request_hash
            cache = context.project_dir / ".yap/cache/music" / f"{request_hash}.mp3"
            if cache.is_file():
                source = cache
                digest = receipt.sha256_file(cache)
                inputs["music:source"] = digest
    if digest is not None:
        inputs["music:source"] = digest
        inputs["style:music:dry_open"] = projection_sha256(dry_open_s)
        inputs.update(context.taste_inputs("music.beat_snap", "music.dry_open_s"))
        if beat_snap:
            # Beat hints come from the pinned renderer's tracker; a new pin can move them.
            inputs["renderer:beats"] = projection_sha256(hyperframes.HYPERFRAMES_VERSION)
    return MusicSnapshot(mode, source, digest, cut_frames, hold_frames,
                         target, target_source, dry_open_s, beat_snap, inputs)


def _prompt_request(settings: dict[str, Any], cut_frames: int,
                    hold_frames: int) -> dict[str, Any]:
    """Describe a music bed with enough headroom for the current cut."""
    if (cut_frames + hold_frames) * 1000 / 30 > 600_000:
        raise MusicError(
            "The cut plus end hold is too long for ElevenLabs music.", code="music_too_long",
            suggested_action="Shorten the cut or end hold to 600 seconds or less.",
        )
    description = settings.get("description") or (
        "Calm instrumental bed under a talking-head video, no vocals"
    )
    tags = settings.get("tags") or []
    prompt = ", ".join([description, *tags])
    duration = min(600_000, max(3000, round(cut_frames * 1000 / 30) + 15_000))
    return {"prompt": prompt, "music_length_ms": duration,
            "model_id": elevenlabs.MUSIC_MODEL}


def _plan_covers(plan: dict[str, Any], requested: dict[str, Any],
                 cut_frames: int, hold_frames: int) -> bool:
    """Keep an authored plan while its music covers the cut and hold."""
    saved = plan.get("prompt_request")
    if not isinstance(saved, dict):
        return False
    length = saved.get("music_length_ms")
    return (saved.get("prompt") == requested["prompt"]
            and saved.get("model_id") == requested["model_id"]
            and isinstance(length, int) and not isinstance(length, bool)
            and length * 30 >= (cut_frames + hold_frames) * 1000)


def _generation_body(plan: dict[str, Any]) -> dict[str, Any]:
    """Build the paid request from the creator-editable composition plan."""
    composition = plan.get("composition_plan")
    if not isinstance(composition, dict):
        raise MusicError("music/plan.json needs a composition_plan object.",
                         code="music_plan_invalid",
                         suggested_action="Edit music/plan.json or run music DIR --new.")
    return {"composition_plan": composition, "model_id": elevenlabs.MUSIC_MODEL}


def _write_json(root: Path, relative: str, data: dict[str, Any]) -> None:
    """Publish non-receipt state under the project lock."""
    with project_lock(root):
        target = root / relative
        paths.validate_output_dir(target.parent)
        paths.atomic_write_text(target, json.dumps(data, indent=2) + "\n")


def _pending_error(root: Path) -> MusicError:
    return MusicError(
        "A paid music generation did not finish and may have been billed.",
        code="music_generation_unreconciled",
        suggested_action=("Check your ElevenLabs history, then run yap-studio music "
                          f"{root} --confirm-retry."),
    )


def _remote_pending_error(host: str) -> MusicError:
    return MusicError(
        f"A paid music generation started on another computer ({host}) and may still be running.",
        code="music_generation_unreconciled",
        suggested_action=(f"Only after checking that {host} is stopped and checking your "
                          "ElevenLabs history, run yap-studio music DIR --confirm-retry "
                          "--other-computer-stopped."),
    )


def _require_paid_account() -> elevenlabs.AccountState:
    """Check account readiness only at the point a remote request is needed."""
    account = elevenlabs.account_state()
    if account in {"missing_cli", "signed_out", "free"}:
        lines = elevenlabs.signup_lines("music", account)
        raise MusicError(lines[0], code=("elevenlabs_paid_plan_required"
                         if account == "free" else "elevenlabs_not_set_up"),
                         suggested_action="Set up a paid ElevenLabs plan or use --file.",
                         notices=lines)
    return account


def _pending_owner_alive(pending: dict[str, Any]) -> bool:
    return (pending.get("state") == "pending"
            and (pending.get("host") != socket.gethostname()
                 or pid_alive(pending.get("pid"))))


def _generated_source(
    context: ProjectContext, snapshot: MusicSnapshot, *, new: bool,
    confirm_retry: bool, other_computer_stopped: bool,
) -> tuple[MusicSnapshot, dict[str, Any]]:
    """Compose, reconcile, and cache one approved paid request."""
    root = context.project_dir
    pending_path = root / ".yap/music/pending.json"
    with project_lock(root):
        if pending_path.is_file():
            pending_record = json.loads(pending_path.read_text(encoding="utf-8"))
            remote_pending = (pending_record.get("state") == "pending"
                              and pending_record.get("host") != socket.gethostname())
            if (not confirm_retry or (_pending_owner_alive(pending_record)
                                     and not (remote_pending and other_computer_stopped))):
                if remote_pending:
                    raise _remote_pending_error(str(pending_record.get("host")))
                raise _pending_error(root)
            pending_path.unlink()
    requested = _prompt_request(context.project.music, snapshot.cut_frames,
                                snapshot.end_hold_frames)
    plan_path = root / "music/plan.json"
    previous = read_receipt(root / "music") or {}
    account: elevenlabs.AccountState | None = None
    plan_digest = receipt.sha256_file(plan_path) if plan_path.is_file() else None
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
    if plan is None or not _plan_covers(
        plan, requested, snapshot.cut_frames, snapshot.end_hold_frames
    ):
        if previous.get("elevenlabs", {}).get("request_sha256") and not new:
            raise _changed_request(root)
        account = _require_paid_account()
        composed = elevenlabs.compose_plan(requested, dry_run=False)
        plan = {"schema": "yap-studio/music-plan@1", "prompt_request": requested,
                "composition_plan": composed}
        with project_lock(root):
            current_digest = receipt.sha256_file(plan_path) if plan_path.is_file() else None
            if current_digest != plan_digest:
                current_plan = json.loads(plan_path.read_text(encoding="utf-8"))
                if not _plan_covers(current_plan, requested, snapshot.cut_frames,
                                    snapshot.end_hold_frames):
                    raise ConflictError(plan_path)
                plan = current_plan
            else:
                paths.validate_output_dir(plan_path.parent)
                paths.atomic_write_text(plan_path, json.dumps(plan, indent=2) + "\n")
    body = _generation_body(plan)
    request_hash = elevenlabs.request_sha256(body)
    cache = root / ".yap/cache/music" / f"{request_hash}.mp3"
    reused = cache.is_file() and not new
    old_hash = previous.get("elevenlabs", {}).get("request_sha256")
    if old_hash and old_hash != request_hash and not reused and not new:
        raise _changed_request(root)
    if not reused:
        if account is None:
            account = _require_paid_account()
        paths.validate_output_dir(cache.parent)
        work_root = paths.validate_output_dir(root / ".yap/work")
        with tempfile.TemporaryDirectory(prefix="music-generate-", dir=work_root) as folder:
            target = Path(folder) / "generated.mp3"
            pending = {"schema": "yap-studio/music-pending@1", "state": "pending",
                       "request": body, "request_sha256": request_hash,
                       "started": datetime.now(UTC).isoformat(),
                       "pid": os.getpid(), "host": socket.gethostname()}
            with project_lock(root):
                if pending_path.is_file():
                    raise _pending_error(root)
                if cache.is_file() and not new:
                    reused = True
                else:
                    paths.validate_output_dir(pending_path.parent)
                    paths.atomic_write_text(pending_path, json.dumps(pending, indent=2) + "\n")
            if reused:
                snapshot = music_snapshot(ProjectContext(root))
                return snapshot, {"account": previous.get("elevenlabs", {}).get(
                                      "account", "paid"), "model": elevenlabs.MUSIC_MODEL,
                                  "request_sha256": request_hash, "request": body,
                                  "output_sha256": receipt.sha256_file(cache),
                                  "plan_sha256": receipt.sha256_file(plan_path), "reused": True}
            try:
                elevenlabs.generate_music(body, target, dry_run=False)
            except elevenlabs.GenerationError as exc:
                if exc.refused:
                    with project_lock(root):
                        pending_path.unlink(missing_ok=True)
                    lines = elevenlabs.signup_lines(
                        "music", "free", first_lines=(
                            "Music: ElevenLabs refused the request because music needs a paid "
                            "plan, so this video has no generated music.",
                            "ElevenLabs can compose music to fit your video. "
                            "Music needs a paid ElevenLabs plan.",
                        ))
                    raise MusicError(lines[0], code="elevenlabs_paid_plan_required",
                                     suggested_action="Upgrade your ElevenLabs plan or use --file.",
                                     notices=lines) from exc
                _write_json(root, ".yap/music/pending.json",
                            {**pending, "state": "unreconciled",
                             "error": elevenlabs.redact_cli_text(str(exc))[-500:]})
                raise MusicError("ElevenLabs music generation did not finish.",
                                 code="music_generation_failed",
                                 suggested_action=(
                                     _pending_error(root).suggested_action or "")) from exc
            except (YapError, OSError, ValueError) as exc:
                _write_json(root, ".yap/music/pending.json",
                            {**pending, "state": "unreconciled",
                             "error": elevenlabs.redact_cli_text(str(exc))[-500:]})
                raise MusicError("ElevenLabs music generation did not finish.",
                                 code="music_generation_failed",
                                 suggested_action=(
                                     _pending_error(root).suggested_action or "")) from exc
            try:
                if not target.is_file() or target.stat().st_size == 0:
                    raise ValueError("empty audio output")
                decoded = _decode(target)
                if len(decoded) == 0 or not np.isfinite(decoded).all():
                    raise ValueError("unreadable audio output")
                if cache.is_file():
                    with project_lock(root):
                        cache.unlink()
                with paths.atomic_output(cache) as staged:
                    shutil.copyfile(target, staged)
            except (YapError, OSError, ValueError) as exc:
                _write_json(root, ".yap/music/pending.json",
                            {**pending, "state": "unreconciled", "error": str(exc)[-500:]})
                raise MusicError("ElevenLabs music output could not be decoded.",
                                 code="music_generation_failed",
                                 suggested_action=(
                                     _pending_error(root).suggested_action or "")) from exc
            with project_lock(root):
                pending_path.unlink(missing_ok=True)
    snapshot = music_snapshot(ProjectContext(root))
    return snapshot, {"account": account or previous.get("elevenlabs", {}).get(
                          "account", "paid"), "model": elevenlabs.MUSIC_MODEL,
                      "request_sha256": request_hash, "request": body,
                      "output_sha256": receipt.sha256_file(cache),
                      "plan_sha256": receipt.sha256_file(plan_path), "reused": reused}


def _changed_request(root: Path) -> MusicError:
    return MusicError("The ElevenLabs music request changed.", code="music_request_changed",
                      suggested_action=(f"Run yap-studio music {root} --new to compose again; "
                                        "it spends ElevenLabs credits."))


def _decode(source: Path) -> np.ndarray:
    """Decode a retained source as 48 kHz stereo floating-point samples."""
    result = ffmpeg.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:a:0", "-vn",
         "-ac", "2", "-ar", str(RATE), "-c:a", "pcm_f32le", "-f", "f32le", "-"],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S, binary=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2).copy()


def _arrange(snapshot: MusicSnapshot, work: Path) -> tuple[Path, dict[str, Any]]:
    """Trim, fade, normalise and pad a source snapshot without clipping its peaks."""
    retained = work / "source"
    shutil.copyfile(snapshot.source, retained)
    if receipt.sha256_file(retained) != snapshot.source_sha256:
        raise MusicError(
            "The music file changed while being read.", code="music_source_changed",
            suggested_action="Wait for the track to finish changing, then run music again.",
        )
    samples = _decode(retained)
    source_samples = len(samples)
    required_samples = (snapshot.cut_samples if snapshot.mode == "elevenlabs"
                        else snapshot.final_samples)
    if source_samples < required_samples:
        fix = ("Run yap-studio music DIR --new to compose a longer track; "
               "it spends ElevenLabs credits." if snapshot.mode == "elevenlabs" else
               "Choose a longer track or loop it yourself, then run music again.")
        raise MusicError(
            f"The track is {source_samples / RATE:.3f} s; the music needs "
            f"{required_samples / RATE:.3f} s.", code="music_too_short",
            suggested_action=fix,
        )
    samples = samples[:snapshot.cut_samples].copy()
    # A quarter of a short cut leaves room for music before the fade.
    fade_samples = min(RATE, snapshot.cut_samples // 4)
    envelope = (1 + np.cos(np.linspace(0, np.pi, fade_samples))) / 2
    samples[-fade_samples:] *= envelope[:, None]
    raw = work / "arranged.f32le"
    raw.write_bytes(samples.astype("<f4").tobytes())
    input_args = ["-f", "f32le", "-ar", str(RATE), "-ac", "2", "-i", str(raw),
                  "-map", "0:a:0"]
    measured = ffmpeg.run(
        ["ffmpeg", "-v", "info", *input_args, "-af",
         f"loudnorm=I={snapshot.target_lufs}:print_format=json", "-f", "null", "-"],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    try:
        # Some media-tool releases print their mux summary after the loudness JSON.
        stats, _ = json.JSONDecoder().raw_decode(measured.stderr[measured.stderr.rfind("{"):])
        input_lufs = float(stats["input_i"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MusicError(
            "Could not measure the track's integrated loudness.", code="music_loudness_failed",
            suggested_action="Check ffmpeg with yap-studio doctor and try the track again.",
        ) from exc
    if not math.isfinite(input_lufs):
        raise MusicError(
            "The trimmed music has no measurable integrated loudness.", code="music_silent",
            suggested_action="Choose an audible track with enough music inside the cut.",
        )
    gain_db = snapshot.target_lufs - input_lufs
    bed = work / "bed.wav"
    # Float PCM preserves stem peaks for assembly's later level and final limiter.
    ffmpeg.run(
        ["ffmpeg", "-v", "error", *input_args, "-af",
         f"volume={gain_db:.9f}dB,apad=whole_len={snapshot.final_samples},"
         f"atrim=end_sample={snapshot.final_samples}", "-c:a", "pcm_f32le", str(bed)],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )
    decoded = _decode(bed)
    if (len(decoded) != snapshot.final_samples or not np.isfinite(decoded).all()
            or np.any(decoded[snapshot.cut_samples - 1:])):
        raise MusicError(
            "The music stem failed its length or silence check.", code="music_contract_failed",
            suggested_action="Check the source audio and run music again.",
        )
    return bed, {
        "source": {"sha256": snapshot.source_sha256, "samples": source_samples},
        "trim": {"start_sample": 0, "end_sample": snapshot.cut_samples,
                 "discarded_samples": source_samples - snapshot.cut_samples},
        "fade": {"curve": "raised_cosine", "start_sample": snapshot.cut_samples - fade_samples,
                 "end_sample": snapshot.cut_samples, "duration_s": fade_samples / RATE},
        "normalisation": {"method": "integrated_linear_gain", "target_lufs": snapshot.target_lufs,
                          "target_source": snapshot.target_source, "input_lufs": input_lufs,
                          "gain_db": gain_db},
    }


def _remove_output(run: StageRun, name: str) -> None:
    """Retire a music output only when its receipt journal proves ownership."""
    target = run.stage_dir / name
    with project_lock(run.project_dir):
        if not target.exists() and not target.is_symlink():
            return
        old = receipt.sha256_file(target)
        paths.validate_output(target, owner_receipts=(run.receipt_path, run.pending_path))
        # Journal deletion so a crash preserves proof for either side of the unlink.
        run._journal_output({"path": f"music/{name}", "old_sha256": old, "new_sha256": None})
        if receipt.sha256_file(target) != old:
            raise ConflictError(target)
        target.unlink()


def _remove_bed(run: StageRun) -> None:
    """Remove a retired bed and its beat hints with journalled ownership checks."""
    for name in ("bed.wav", "beats.json"):
        _remove_output(run, name)


def _beats(snapshot: MusicSnapshot, details: dict[str, Any]) -> dict[str, Any]:
    """Map source beats through the bed trim onto its audible output clock."""
    source = cast(Path, snapshot.source)
    if receipt.sha256_file(source) != snapshot.source_sha256:
        raise BeatsError("the source music changed while preparing the bed")
    detected = detect_beats(source, duration_s=details["source"]["samples"] / RATE)
    if receipt.sha256_file(source) != snapshot.source_sha256:
        raise BeatsError("the source music changed during beat detection")
    first = details["trim"]["start_sample"] / RATE
    last = details["trim"]["end_sample"] / RATE
    mapped = []
    for beat in detected["beats"]:
        time = beat["time_s"] - first
        # The bed starts at output zero. A dry opening mutes samples rather
        # than delaying the track, and the cut end starts the silent hold.
        if first <= beat["time_s"] < last and snapshot.dry_open_s <= time:
            mapped.append({"time_s": time, "frame": round(time * float(FPS)),
                           "strength": beat["strength"]})
    return {"schema": "yap-studio/beats@1", "bpm": detected["bpm"],
            "music_sha256": snapshot.source_sha256, "beats": mapped}


def music(project_dir: Path, *, force: bool = False, new: bool = False,
          confirm_retry: bool = False, other_computer_stopped: bool = False,
          dry_run: bool = False,
          preview_mode: str | None = None,
          preview_description: str | None = None,
          preview_tags: list[str] | None = None) -> dict[str, Any]:
    """Publish a supplied or approved generated bed, or a voice-only receipt."""
    if other_computer_stopped and not confirm_retry:
        raise MusicError("--other-computer-stopped requires --confirm-retry.",
                         code="music_retry_confirmation_required",
                         suggested_action="Add --confirm-retry and try again.")
    with paths.operation():
        context = ProjectContext(project_dir)
        snapshot = music_snapshot(context, require_fresh_cut=not dry_run)
        root = context.project_dir
        notices: list[str] = []
        account: elevenlabs.AccountState | None = None
        extra_elevenlabs: dict[str, Any] | None = None
        mode = (preview_mode if dry_run and preview_mode is not None
                else context.project.music["mode"])
        if mode in {"auto", "elevenlabs"}:
            if dry_run:
                account = elevenlabs.account_state()
                if account == "missing_cli":
                    lines = elevenlabs.signup_lines("music", account)
                    raise MusicError(lines[0], code="elevenlabs_not_set_up",
                                     suggested_action=(
                                         "Set up a paid ElevenLabs plan or use --file."),
                                     notices=lines)
                notices = (elevenlabs.signup_lines("music", account)
                           if account in {"signed_out", "free"} else [])
                settings = dict(context.project.music)
                if preview_description is not None:
                    settings["description"] = preview_description
                if preview_tags is not None:
                    settings["tags"] = preview_tags
                requested = _prompt_request(settings, snapshot.cut_frames,
                                            snapshot.end_hold_frames)
                plan_path = root / "music/plan.json"
                if not plan_path.is_file():
                    preview = elevenlabs.compose_plan(requested, dry_run=True)
                    return {"status": "dry_run", "requests": [preview], "notices": notices}
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                saved_request = plan.get("prompt_request")
                if isinstance(saved_request, dict) and saved_request != requested:
                    preview = elevenlabs.compose_plan(requested, dry_run=True)
                    return {"status": "dry_run", "requests": [preview], "notices": notices}
                with tempfile.TemporaryDirectory(prefix="yap-music-preview-") as folder:
                    preview = elevenlabs.generate_music(_generation_body(plan),
                                                         Path(folder) / "preview.mp3",
                                                         dry_run=True)
                return {"status": "dry_run", "requests": [preview], "notices": notices}
            if mode == "elevenlabs":
                snapshot, extra_elevenlabs = _generated_source(
                    context, snapshot, new=new, confirm_retry=confirm_retry,
                    other_computer_stopped=other_computer_stopped)
            else:
                account = elevenlabs.account_state()
                extra_elevenlabs = {"account": account}
                if account in {"missing_cli", "signed_out", "free"}:
                    notices = elevenlabs.show_once(
                        root, "elevenlabs_music", elevenlabs.signup_lines("music", account))
        elif dry_run:
            return {"status": "dry_run", "requests": [], "notices": []}
        run = StageRun(root, "music")
        contract = {
            "mode": snapshot.mode,
            "bed": "music/bed.wav" if snapshot.mode in {"file", "elevenlabs"} else None,
            "sample_rate": RATE, "channels": 2, "sample_format": "float32",
            "fps": 30, "samples_per_frame": SAMPLES_PER_FRAME,
            "cut_frames": snapshot.cut_frames, "end_hold_frames": snapshot.end_hold_frames,
            "cut_samples": snapshot.cut_samples, "final_samples": snapshot.final_samples,
        }
        result = {"status": "verified", **contract, "receipt": "music/receipt.json",
                  "notices": notices}
        if not force and not new and run.up_to_date(snapshot.inputs):
            return {**result, "noop": True}
        work_root = paths.validate_output_dir(root / ".yap/work")
        with tempfile.TemporaryDirectory(prefix="music-", dir=work_root) as folder, run:
            for name, digest in snapshot.inputs.items():
                run.record_inputs(name, digest)
            details: dict[str, Any] = {"source": None, "trim": None, "fade": None,
                                       "normalisation": None}
            if snapshot.source is not None:
                paths.protect_input(snapshot.source)
                run.progress("preparing music", 0.25)
                bed, details = _arrange(snapshot, Path(folder))
                details["source"]["path"] = (context.project.music["file"]
                                              if snapshot.mode == "file" else
                                              str(snapshot.source.relative_to(root)))
                run.replace_output(root / "music/bed.wav", bed)
                if not snapshot.beat_snap:
                    _remove_output(run, "beats.json")
                    details["beats"] = {"status": "skipped",
                                        "reason": "music.beat_snap is disabled"}
                else:
                    try:
                        hints = _beats(snapshot, details)
                    except (YapError, OSError) as exc:
                        _remove_output(run, "beats.json")
                        details["beats"] = {"status": "skipped", "reason": str(exc)}
                    else:
                        staged = paths.atomic_write_text(
                            Path(folder) / "beats.json", json.dumps(hints, indent=2) + "\n",
                        )
                        run.replace_output(root / "music/beats.json", staged)
                        details["beats"] = {"status": "verified", "path": "music/beats.json",
                                            "bpm": hints["bpm"], "count": len(hints["beats"])}
            else:
                _remove_bed(run)
            run.finish(extra={**contract, **details,
                              **({"elevenlabs": extra_elevenlabs}
                                 if extra_elevenlabs is not None else {}),
                              "taste": context.taste_application().pin()},
                       asserts=[
                {"id": "music.samples", "pass": True},
                {"id": "music.silent_hold", "pass": True},
            ])
        return {**result, "noop": False}


def music_human_items(context: ProjectContext) -> list[str]:
    """Read saved music decisions without invoking the ElevenLabs CLI."""
    import shlex

    root = context.project_dir
    command = f"yap-studio music {shlex.quote(str(root))}"
    pending = root / ".yap/music/pending.json"
    if pending.is_file():
        return [f"music: a paid generation may have been billed; check ElevenLabs history, "
                f"then run {command} --confirm-retry"]
    record = read_receipt(root / "music") or {}
    if context.project.music["mode"] == "auto":
        account = record.get("elevenlabs", {}).get("account")
        if account in {"paid", "unknown"}:
            return ["music: ElevenLabs can compose music for this video; "
                    f"to spend credits on it run {command} --yes"]
    if context.project.music["mode"] == "elevenlabs":
        old = record.get("elevenlabs", {}).get("request_sha256")
        plan_path = root / "music/plan.json"
        if old:
            try:
                if not plan_path.is_file():
                    raise FileNotFoundError(plan_path)
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                snapshot = music_snapshot(context)
                requested = _prompt_request(context.project.music, snapshot.cut_frames,
                                            snapshot.end_hold_frames)
                changed = (old != elevenlabs.request_sha256(_generation_body(plan))
                           or not _plan_covers(plan, requested, snapshot.cut_frames,
                                               snapshot.end_hold_frames))
            except YapError as exc:
                if not isinstance(exc, MusicError):
                    return []
                changed = True
            except (OSError, ValueError, KeyError, TypeError):
                changed = True
            if changed:
                return [f"music: request changed; run {command} --new to spend credits again"]
    return []
