"""Approved paid speech requests, durable retry guards and local patch checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
import socket
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from yap_studio import elevenlabs, paths, receipt
from yap_studio.errors import YapError
from yap_studio.project.lock import pid_alive, project_lock, timestamp
from yap_studio.stages import cut as cut_stage
from yap_studio.stages import edges
from yap_studio.stages.cut import _audio_piece, _cut_words, resolve_cut
from yap_studio.voice.consent import require_account, require_consent, resolve_voice
from yap_studio.voice.model import VoiceError, digest, local_path, read, write
from yap_studio.voice.plan import (
    estimate,
    list_fixes,
    require_picture,
    source_sample,
    validate_words,
)
from yap_studio.voice.recipe import load_recipe, speech_text
from yap_studio.voice.splice import _snap, blend, build_patch, checks_status, decode, write_wav


def take_audio(plan, take: str) -> np.ndarray:
    """Decode a complete take on the cut's source sample clock."""
    piece = next(piece for piece in plan.pieces if piece.take == take)
    duration = Fraction(str(plan.media[take].video_duration_s))
    return _audio_piece(
        replace(
            piece,
            src_t0=Fraction(0),
            src_t1=duration,
            sample_start=0,
            sample_end=round(duration * 48000),
        ),
        plan,
    )


def _pending_error() -> VoiceError:
    return VoiceError(
        "A paid voice generation did not finish and may have been billed.",
        code="voice_generation_unreconciled",
        suggested_action=(
            "Check ElevenLabs history and that the earlier call has stopped, "
            "then run voice make with --confirm-retry."
        ),
    )


def _check_pending(root: Path, confirm_retry: bool) -> None:
    pending = read(root, "pending", optional=True)
    if pending is None:
        return
    alive = (
        pending["state"] == "pending"
        and pending["host"] == socket.gethostname()
        and pid_alive(pending["pid"])
    )
    if not confirm_retry or alive:
        raise _pending_error()
    local_path(root, ".yap/voice/pending.json").unlink()


def _cost(root: Path, fixes: list[dict], budget: dict, requests: dict) -> dict:
    lines = []
    total = 0.0
    characters = 0
    suffix = "Your monthly credits could not be read."
    if budget["remaining"] is not None:
        suffix = f"You have {budget['remaining']:,} credits left this month."
    elif (
        budget.get("unavailable", {}).get("plan", {}).get("code") == "elevenlabs_permission_missing"
    ):
        suffix = (
            "Your monthly credits could not be read because your ElevenLabs key "
            "cannot read your plan."
        )
    for fix in fixes:
        if fix.get("from", {}).get("kind") == "own":
            lines.append(f"Using your own recording for '{fix['say']}' costs 0 credits.")
            continue
        request, request_hash = requests[fix["id"]]
        row = estimate(fix["say"], budget, sent=request["text"])
        if _cached(root, request_hash, row["credits"], fix.get("generation", {})) is not None:
            lines.append(f"Reusing cached audio for '{fix['say']}' costs 0 credits.")
            continue
        total += row["credits"]
        characters += row["characters"]
        lines.append(
            f"Re-saying '{fix['say']}' costs about {row['credits']:g} ElevenLabs credits "
            f"({row['characters']} characters). {suffix}"
        )
    if len(fixes) > 1:
        lines.append(f"Total: about {total:g} ElevenLabs credits.")
    return {
        "credits": total,
        "characters": characters,
        "model": budget.get("model", elevenlabs.VOICE_MODEL),
        "multiplier": budget["multiplier"],
        "source": budget["source"],
        "remaining": budget["remaining"],
        "lines": lines,
        **({"unavailable": budget["unavailable"]} if budget.get("unavailable") else {}),
    }


def _request(plan, fix: dict, consent: dict, *, new: bool, plain: bool = False) -> tuple[dict, str]:
    words = _cut_words(plan)["words"]
    first = next(i for i, word in enumerate(words) if word["id"] == fix["words"][0])
    segment = words[first]["segment"]
    last = next(
        i
        for i, word in enumerate(words)
        if i >= first and word["id"] == fix["words"][-1] and word["segment"] == segment
    )
    previous_seed = fix.get("generation", {}).get("seed")
    seed = (
        previous_seed
        if previous_seed is not None
        else int(digest(json.dumps([fix["take"], fix["words"], fix["say"]]))[:8], 16)
    )
    recipe = load_recipe()
    previous = " ".join(w["text"] for w in words[:first] if w["text"].strip())[-300:]
    following = " ".join(w["text"] for w in words[last + 1:] if w["text"].strip())[:300]
    request = {
        "text": speech_text(fix["say"], previous if recipe["context"] else "", recipe, plain=plain),
        "seed": seed,
        "model": recipe["model_id"],
        "voice_sha256": consent["voice"]["id_sha256"],
        "format": elevenlabs.MUSIC_FORMAT,
        "recipe": recipe,
        "recipe_sha256": digest(json.dumps(recipe, sort_keys=True)),
        "plain": plain,
        "pace": recipe["pace"],
    }
    if recipe["context"]:
        request.update(previous_text=previous, next_text=following)
    if recipe["voice_settings"] is not None:
        request["voice_settings"] = recipe["voice_settings"]
    nonce = secrets.token_hex(8) if new else fix.get("generation", {}).get("nonce")
    if nonce:
        request["nonce"] = nonce
    return request, digest(json.dumps(request, sort_keys=True, separators=(",", ":")))


def _cached(root: Path, request_hash: str, estimate_credits: float,
            previous: dict) -> tuple[Path, dict] | None:
    cache = local_path(root, f".yap/cache/voice/{request_hash}.mp3")
    cache_record = local_path(root, f".yap/cache/voice/{request_hash}.json")
    if not cache.is_file():
        return None
    if not cache.stat().st_size:
        raise VoiceError(
            "The cached voice audio is empty. Use --new.", code="voice_cache_invalid"
        )
    try:
        entry = json.loads(cache_record.read_text())
    except (OSError, ValueError, UnicodeError):
        entry = None
    matching = (isinstance(entry, dict)
                and entry.get("schema") == "yap-studio/voice-cache@1"
                and entry.get("request_sha256") == request_hash)
    expected = (previous.get("output_sha256")
                if previous.get("request_sha256") == request_hash else None)
    if matching:
        expected = entry.get("output_sha256") or expected
    if expected and receipt.sha256_file(cache) != expected:
        raise VoiceError(
            "The cached voice audio changed. Use --new.", code="voice_cache_invalid"
        )
    charge = entry.get("cost") if matching else None
    if (not isinstance(charge, dict) or type(charge.get("credits")) not in (int, float)
            or not math.isfinite(charge["credits"]) or charge["credits"] < 0
            or charge.get("source") not in {"history", "estimate"}):
        charge = {"credits": estimate_credits, "source": "estimate"}
    return cache, charge


def _generated(
    root: Path,
    identifier: str,
    request: dict,
    request_hash: str,
    estimate_credits: float,
    previous: dict,
    *,
    consent: dict,
    max_credits: float,
    confirm_retry: bool,
) -> tuple[Path, dict, bool]:
    cache = local_path(root, f".yap/cache/voice/{request_hash}.mp3")
    cache_record = local_path(root, f".yap/cache/voice/{request_hash}.json")
    pending_path = local_path(root, ".yap/voice/pending.json")

    with project_lock(root):
        _check_pending(root, confirm_retry)
        result = _cached(root, request_hash, estimate_credits, previous)
        if result is not None:
            return *result, False
    require_account()
    work_root = paths.validate_output_dir(root / ".yap/work")
    with tempfile.TemporaryDirectory(prefix="voice-generate-", dir=work_root) as folder:
        target = Path(folder) / "generated.mp3"
        pending = {
            "schema": "yap-studio/voice-pending@1",
            "state": "pending",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started": timestamp(),
            "request_sha256": request_hash,
        }
        with project_lock(root):
            _check_pending(root, False)
            result = _cached(root, request_hash, estimate_credits, previous)
            if result is not None:
                return *result, False
            if require_consent(root) != consent:
                raise VoiceError("Voice consent changed. Read it again before generating.")
            if estimate_credits > max_credits:
                raise VoiceError(
                    "This voice fix costs more than your remaining credit ceiling.",
                    code="voice_over_ceiling",
                )
            write(root, "pending", pending)
        started = math.floor(time.time())
        try:
            elevenlabs.generate_voice(identifier, request, target)
            decode(target, 1)
            charge = elevenlabs.voice_charge(identifier, request["text"], started, estimate_credits)
            with project_lock(root):
                paths.validate_output_dir(cache.parent)
                with paths.atomic_output(cache) as staged:
                    shutil.copyfile(target, staged)
                paths.atomic_write_text(cache_record, json.dumps({
                    "schema": "yap-studio/voice-cache@1", "request_sha256": request_hash,
                    "output_sha256": receipt.sha256_file(cache), "cost": charge,
                }, allow_nan=False) + "\n")
                pending_path.unlink()
        except elevenlabs.GenerationError as exc:
            with project_lock(root):
                if exc.refused:
                    pending_path.unlink(missing_ok=True)
                else:
                    write(root, "pending", {**pending, "state": "unreconciled"})
            raise VoiceError(
                "ElevenLabs refused the voice request."
                if exc.refused
                else "Voice generation did not finish and may have been billed.",
                code="voice_generation_refused" if exc.refused else "voice_generation_unreconciled",
                suggested_action=(
                    "Check your ElevenLabs account and try again."
                    if exc.refused
                    else _pending_error().suggested_action
                ),
            ) from None
        except VoiceError as exc:
            with project_lock(root):
                if exc.code == "elevenlabs_permission_missing":
                    pending_path.unlink(missing_ok=True)
                else:
                    write(root, "pending", {**pending, "state": "unreconciled"})
            if exc.code == "elevenlabs_permission_missing":
                raise
            raise _pending_error() from None
        except (YapError, OSError, ValueError):
            with project_lock(root):
                write(root, "pending", {**pending, "state": "unreconciled"})
            raise _pending_error() from None
        return cache, charge, True


def _heard(plan, patch: Path, say: str, *, respelled: bool = False) -> dict:
    """Check only the replacement audio, without adjacent speech."""
    backend = cut_stage._edge_backend(plan.context)
    if backend is None:
        return {
            "id": "splice.heard",
            "status": "skipped",
            "reason": "No local speech engine is available.",
        }
    try:
        result = backend.transcribe_many([patch], language=plan.context.project.language)[patch]
        heard = edges.heard_words(result.text)
        expected = edges.heard_words(say)
        matches = any(
            edges._matches(heard[i : i + len(expected)], expected, True)
            for i in range(len(heard) - len(expected) + 1)
        )
        return {
            "id": "splice.heard",
            "status": "pass" if matches else "warn",
            "heard": result.text,
            "expected": say,
            **({"message": "Try voice make --plain to use the plain words."}
               if respelled and not matches else {}),
        }
    except YapError:
        return {
            "id": "splice.heard",
            "status": "warn",
            "reason": "The local speech check did not finish.",
        }


def _check_publication(root: Path, fix: dict) -> None:
    """Check all previous outputs before publishing a replacement generation."""
    for row in [fix.get("patch"), *fix.get("clips", {}).values()]:
        if row:
            paths.validate_output(
                local_path(root, row["path"]), owner_receipts=[root / "voice/fixes.json"],
            )


def make(
    project_dir: Path,
    fix_ids: list[str],
    *,
    yes: bool = False,
    max_credits: float | None = None,
    dry_run: bool = False,
    new: bool = False,
    confirm_retry: bool = False,
    plain: bool = False,
    show_cost: Callable[[dict], None] | None = None,
) -> dict:
    """Show a cost, reserve each paid call, then build reviewable patches."""
    root = Path(project_dir)
    if not fix_ids or len(fix_ids) > 3 or len(set(fix_ids)) != len(fix_ids):
        raise VoiceError("Choose 1 to 3 different fixes per run.", code="voice_too_many_fixes")
    with paths.operation():
        data = list_fixes(root)
        fixes = []
        for identifier in fix_ids:
            fix = next((fix for fix in data["fixes"] if fix["id"] == identifier), None)
            if fix is None:
                raise VoiceError(f"Voice fix {identifier} was not found.", code="voice_fix_missing")
            if fix["status"] in ("approved", "rejected"):
                raise VoiceError(
                    "Only proposed, failed or ready fixes can be made.", code="voice_fix_decided"
                )
            fixes.append(fix)
        clones = [fix for fix in fixes if fix.get("from", {}).get("kind") != "own"]
        consent = require_consent(root) if clones else None
        plan = resolve_cut(root)
        for fix in fixes:
            validate_words(plan, fix)
            require_picture(plan, fix)
        requests = {fix["id"]: _request(plan, fix, consent, new=new, plain=plain)
                    for fix in clones}
        model_id = next(iter(requests.values()))[0]["model"] if clones else "own"
        budget = {**elevenlabs.voice_budget(model_id), "model": model_id} if clones else {
            "multiplier": 0, "source": "own", "remaining": None, "model": "own",
        }
        cost = _cost(root, fixes, budget, requests)
        if show_cost:
            show_cost(cost)
        if clones and not yes and not dry_run:
            raise VoiceError(
                "Approve this cost first.",
                code="voice_approval_required",
                cost=cost,
                suggested_action="Approve this cost with --yes and --max-credits N.",
            )
        if max_credits is not None and (not math.isfinite(max_credits) or max_credits < 0):
            raise VoiceError(
                "Use a finite, non-negative credit ceiling.", code="voice_over_ceiling", cost=cost
            )
        if max_credits is not None and cost["credits"] > max_credits:
            raise VoiceError(
                "The voice fixes cost more than your credit ceiling.",
                code="voice_over_ceiling",
                cost=cost,
            )
        if dry_run:
            return {"status": "dry_run", "fixes": fixes, "cost": cost}
        if clones and max_credits is None:
            raise VoiceError(
                "Set --max-credits before making voice fixes.",
                code="voice_ceiling_required",
                cost=cost,
            )
        if clones:
            with project_lock(root):
                _check_pending(root, confirm_retry)
        results = []
        spent = 0
        work_root = paths.validate_output_dir(root / ".yap/work")
        for fix in fixes:
            own = fix.get("from", {}).get("kind") == "own"
            # Re-check every slot immediately before its paid request.
            plan = resolve_cut(root)
            validate_words(plan, fix)
            require_picture(plan, fix)
            if not own and require_consent(root) != consent:
                raise VoiceError("Voice consent changed. Read it again before generating.")
            with project_lock(root):
                current = next(row for row in list_fixes(root)["fixes"] if row["id"] == fix["id"])
                if current != fix:
                    raise VoiceError("The voice fix changed. Read it again before generating.")
                _check_publication(root, fix)
            request, request_hash = ({}, "") if own else _request(
                plan, fix, consent, new=new, plain=plain,
            )
            if not own:
                approved = requests[fix["id"]][0]
                if request["recipe_sha256"] != approved["recipe_sha256"]:
                    raise VoiceError("The voice recipe changed. Read the cost again.")
                if request["text"] != approved["text"]:
                    raise VoiceError(
                        "The speech text changed since the cost was shown.",
                        code="voice_request_changed",
                        suggested_action="Run voice make again without --yes to see the new cost.",
                    )
            old_hash = fix.get("generation", {}).get("request_sha256")
            previous = fix.get("generation", {})
            recipe_changed = (previous.get("recipe_sha256") != request.get("recipe_sha256")
                              or previous.get("plain", False) != plain)
            if not own and old_hash and old_hash != request_hash and not new and not recipe_changed:
                raise VoiceError(
                    "The voice request changed. Use --new to generate it again.",
                    code="voice_request_changed",
                )
            identifier = None if own else resolve_voice(root)
            original = take_audio(plan, fix["take"])
            require_picture(
                plan,
                fix,
                start_s=_snap(original, round(fix["source"]["start_s"] * 48000)) / 48000,
                end_s=_snap(original, round(fix["source"]["end_s"] * 48000)) / 48000,
            )
            try:
                if own:
                    from yap_studio.voice.own import recorded_audio

                    donor, request_hash = recorded_audio(plan, fix, data["fixes"])
                    cache, charge = None, {"credits": 0, "source": "own"}
                else:
                    cache, charge, fresh = _generated(
                        root, identifier, request, request_hash,
                        estimate(fix["say"], cost, sent=request["text"])["credits"],
                        fix.get("generation", {}), consent=consent,
                        max_credits=max_credits - spent, confirm_retry=False,
                    )
                    if fresh:
                        spent += charge["credits"]
            except VoiceError as exc:
                if exc.code == "voice_over_ceiling":
                    remaining = ", ".join(row["id"] for row in fixes[len(results):])
                    raise VoiceError(
                        f"Voice fixes {remaining} were not made because the remaining "
                        "credit ceiling is too low.",
                        code=exc.code, cost=cost,
                        suggested_action="Read the cost again before approving the fixes left.",
                    ) from None
                with project_lock(root):
                    saved = list_fixes(root)
                    for row in saved["fixes"]:
                        if row == fix and "patch" not in row:
                            row["status"] = "failed"
                            write(root, "fixes", saved)
                raise
            generation = {
                "request_sha256": request_hash,
                "seed": 0 if own else request["seed"],
                "model": "own" if own else request["model"],
                "output_sha256": (hashlib.sha256(donor.tobytes()).hexdigest() if own
                                  else receipt.sha256_file(cache)),
                "cost": charge,
                "at": timestamp(),
                **({} if own else {"recipe_sha256": request["recipe_sha256"], "plain": plain,
                                   "nonce": request.get("nonce")}),
            }
            try:
                words = plan.source_words[fix["take"]].words
                word_spans = [(word.start, word.end) for word in words]
                word_range = (
                    next(i for i, word in enumerate(words) if word.id == fix["words"][0]),
                    next(i for i, word in enumerate(words) if word.id == fix["words"][-1]),
                )
                with tempfile.TemporaryDirectory(prefix="voice-splice-", dir=work_root) as folder:
                    work = Path(folder)
                    if own:
                        cache = work / "recorded.wav"
                        write_wav(cache, donor)
                    patch, meta, checks = build_patch(
                        original,
                        cache,
                        fix["source"]["start_s"],
                        fix["source"]["end_s"],
                        work,
                        word_spans=word_spans,
                        word_range=word_range,
                        recorded=own,
                        pace="fit" if own else request["pace"],
                    )
                    picture = require_picture(
                        plan,
                        fix,
                        start_s=meta["start_sample"] / 48000,
                        end_s=(meta["start_sample"] + meta["samples"]) / 48000,
                    )
                    after = blend(original, patch, meta, source_start_sample=0)
                    first = max(0, meta["start_sample"] - 2 * 48000)
                    last = min(len(original), meta["start_sample"] + meta["samples"] + 2 * 48000)
                    write_wav(work / "patch.wav", patch)
                    write_wav(work / "before.wav", original[first:last])
                    write_wav(work / "after.wav", after[first:last])
                    heard = _heard(
                        plan, work / "patch.wav", fix["say"],
                        respelled=bool(not own and not plain
                                       and request["recipe"]["pronunciations"]),
                    )
                    checks["items"].append(heard)
                    checks["status"] = checks_status(checks["items"])
                    relative = f"voice/fixes/{fix['id']}/g{secrets.token_hex(8)}"
                    clips = {
                        name: {
                            "path": f"{relative}/{name}.wav",
                            "sha256": receipt.sha256_file(work / f"{name}.wav"),
                        }
                        for name in ("before", "after")
                    }
                    ready = {
                        **fix,
                        "status": "ready",
                        "picture": picture,
                        "generation": generation,
                        "estimate": (fix["estimate"] if own else
                                     estimate(fix["say"], cost, sent=request["text"])),
                        "from": fix["from"] if own else {"kind": "clone", **request["recipe"]},
                        "clips": clips,
                        "checks": checks,
                        "patch": {
                            **meta,
                            "start_sample": meta["start_sample"] + source_sample(plan, fix["take"]),
                            "path": f"{relative}/patch.wav",
                            "sha256": receipt.sha256_file(work / "patch.wav"),
                        },
                    }
                    with project_lock(root):
                        if not own and require_consent(root) != consent:
                            raise VoiceError("Voice consent changed during generation.")
                        fresh = resolve_cut(root)
                        validate_words(fresh, fix)
                        require_picture(
                            fresh,
                            fix,
                            start_s=meta["start_sample"] / 48000,
                            end_s=(meta["start_sample"] + meta["samples"]) / 48000,
                        )
                        saved = list_fixes(root)
                        index = next(
                            i for i, row in enumerate(saved["fixes"]) if row["id"] == fix["id"]
                        )
                        if saved["fixes"][index] != fix:
                            raise VoiceError("The voice fix changed during generation.")
                        new_patch = ready["patch"]
                        for other in saved["fixes"]:
                            old = other.get("patch")
                            if (
                                other["id"] != fix["id"]
                                and other["status"] != "rejected"
                                and other["take"] == fix["take"]
                                and old
                                and old["start_sample"]
                                < new_patch["start_sample"] + new_patch["samples"]
                                and old["start_sample"] + old["samples"] > new_patch["start_sample"]
                            ):
                                raise VoiceError(
                                    "This patch overlaps another voice fix.",
                                    code="voice_fix_overlap",
                                )
                        _check_publication(root, fix)
                        targets = {name: local_path(root, f"{relative}/{name}.wav")
                                   for name in ("patch", "before", "after")}
                        folder = local_path(root, relative)
                        paths.validate_output_dir(folder.parent)
                        folder.mkdir()
                        published = False
                        written = []
                        try:
                            for target in targets.values():
                                paths.validate_output(
                                    target, owner_receipts=[root / "voice/fixes.json"],
                                )
                            for name, target in targets.items():
                                with paths.atomic_output(
                                    target, owner_receipts=[root / "voice/fixes.json"],
                                ) as staged:
                                    shutil.copyfile(work / f"{name}.wav", staged)
                                written.append((name, target))
                            saved["fixes"][index] = ready
                            write(root, "fixes", saved)
                            published = True
                        finally:
                            if not published:
                                for name, target in written:
                                    if (target.is_file() and receipt.sha256_file(target)
                                            == receipt.sha256_file(work / f"{name}.wav")):
                                        target.unlink()
                                try:
                                    folder.rmdir()
                                except OSError:
                                    pass
                        for row in [fix.get("patch"), *fix.get("clips", {}).values()]:
                            if row:
                                target = local_path(root, row["path"])
                                if (target.is_file()
                                        and receipt.sha256_file(target) == row["sha256"]):
                                    target.unlink()
                    results.append(ready)
            except (VoiceError, OSError, ValueError):
                with project_lock(root):
                    saved = list_fixes(root)
                    for row in saved["fixes"]:
                        if row == fix and "patch" not in row:
                            row.update(status="failed", generation=generation)
                            write(root, "fixes", saved)
                raise
        return {"status": "ready", "fixes": results, "cost": cost}
