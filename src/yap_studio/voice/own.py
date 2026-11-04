"""Find and retain clean recorded words on their original source clock."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from yap_studio import audio, ffmpeg, paths
from yap_studio.project.model import load_words
from yap_studio.stages.cut import _selected_stream
from yap_studio.stages.edges import heard_words
from yap_studio.voice.model import VoiceError
from yap_studio.voice.plan import occurrences
from yap_studio.voice.splice import RATE, _snap, fit_slot


def take_words(context, take):
    """Refuse donor words that no longer describe the recorded take."""
    target = context.project_dir / f"ingest/{take}/words.json"
    if not target.exists():
        raise FileNotFoundError(target)
    words = load_words(target)
    if words.take != take or words.take_sha256 != context.take_identity(take):
        raise VoiceError("The recorded words changed. Ingest this take again.",
                         code="voice_own_changed")
    return words


def take_sound(context, take):
    source = paths.validate_input(context.take_path(take))
    info = ffmpeg.probe(source)
    _, sound = audio.decode(source, stream=_selected_stream(context, take), sr=RATE, mono=False)
    if sound.ndim == 1:
        sound = sound[:, None]
    return sound, info.video_offset_s


def _span(sound, offset, origin, words):
    first = _snap(sound, round((origin["start_s"] + offset) * RATE))
    last = _snap(sound, round((origin["end_s"] + offset) * RATE))
    ids = [word.id for word in words]
    before = words[:ids.index(origin["words"][0])]
    after = words[ids.index(origin["words"][-1]) + 1:]
    if before:
        first = max(first, math.ceil((max(word.end for word in before) + offset) * RATE))
    if after:
        last = min(last, math.floor((min(word.start for word in after) + offset) * RATE))
    if any(first >= (word.end + offset) * RATE or last <= (word.start + offset) * RATE
           for word in words if word.id in origin["words"]):
        last = first
    return first, last


def find_own(plan, fix, fixes, *, chosen=None):
    """Rank exact contiguous phrases before decoding their candidate spans."""
    expected = heard_words(fix["say"])
    if not expected or "?" in expected:
        if chosen:
            raise VoiceError("Those recorded words cannot be matched clearly.",
                             code="voice_own_words_differ")
        return None
    blocked = set(fix["words"])
    for other in fixes:
        if other["id"] != fix.get("id") and other["status"] != "rejected":
            blocked.update(other["words"])
    candidates = []
    for take in plan.context.project.takes:
        if chosen and any(not word.startswith(take.id + ":") for word in chosen):
            continue
        try:
            snapshot = take_words(plan.context, take.id)
        except FileNotFoundError:
            continue
        words = snapshot.words
        for i, word in enumerate(words):
            if chosen and word.id != chosen[0]:
                continue
            tokens = []
            for j in range(i, len(words)):
                tokens.extend(heard_words(words[j].text))
                if not chosen and tokens != expected[:len(tokens)]:
                    break
                if chosen and words[j].id != chosen[1]:
                    continue
                if len(tokens) >= len(expected) or chosen:
                    selected = words[i:j + 1]
                    if tokens != expected:
                        if chosen:
                            raise VoiceError("Those recorded words do not say the replacement.",
                                             code="voice_own_words_differ")
                        break
                    if (any(item.id in blocked for item in selected)
                            or any(item.conf is None or item.conf < 0.7 for item in selected)):
                        break
                    origin = {"kind": "own", "take": take.id,
                              "words": [item.id for item in selected],
                              "start_s": word.start, "end_s": selected[-1].end,
                              "text": " ".join(item.text for item in selected),
                              "take_sha256": snapshot.take_sha256}
                    rank = (take.id != fix["take"], -min(item.conf for item in selected),
                            abs(word.start - fix["source"]["start_s"]), take.id, word.id)
                    candidates.append((rank, origin, words))
                    break
    decoded = {}
    target = None
    for _, origin, donor_words in sorted(candidates, key=lambda item: item[0]):
        take = origin["take"]
        if take not in decoded:
            decoded[take] = take_sound(plan.context, take)
        sound, offset = decoded[take]
        first, last = _span(sound, offset, origin, donor_words)
        span = sound[first:last]
        if (len(span) and np.isfinite(span).all() and np.any(span)
                and np.max(np.abs(span)) < 0.999):
            if not chosen:
                if target is None:
                    from yap_studio.voice.make import take_audio

                    target = take_audio(plan, fix["take"])
                    words = plan.source_words[fix["take"]].words
                    ids = [word.id for word in words]
                    word_range = (ids.index(fix["words"][0]), ids.index(fix["words"][-1]))
                    word_spans = [(word.start, word.end) for word in words]
                try:
                    first, last = fit_slot(
                        target, fix["source"]["start_s"], fix["source"]["end_s"], len(span),
                        word_spans=word_spans, word_range=word_range,
                    )
                    occurrences(plan, fix, start_s=first / RATE, end_s=last / RATE)
                except VoiceError as exc:
                    if exc.code not in {
                        "voice_speech_too_long", "voice_slot_invalid", "voice_words_dropped",
                    }:
                        raise
                    continue
            return origin
    if chosen:
        raise VoiceError(
            "Choose clean recorded words with confidence at least 0.7, "
            "without clipping or another voice fix.",
            code="voice_own_unusable",
        )
    return None


def recorded_audio(plan, fix, fixes):
    """Recheck the chosen words and take before using their audio."""
    origin = fix["from"]
    current = find_own(plan, fix, fixes, chosen=(origin["words"][0], origin["words"][-1]))
    if current != origin:
        raise VoiceError("The recorded source changed. Propose this fix again.",
                         code="voice_own_changed")
    sound, offset = take_sound(plan.context, origin["take"])
    words = take_words(plan.context, origin["take"]).words
    first, last = _span(sound, offset, origin, words)
    speech_spans = [(word.start + offset - first / RATE, word.end + offset - first / RATE)
                    for word in words if word.id in origin["words"]]
    identity = hashlib.sha256(json.dumps(origin, sort_keys=True).encode()).hexdigest()
    return sound[first:last], identity, speech_spans
