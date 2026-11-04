"""Find and retain clean recorded words on their original source clock."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from yap_studio import audio, ffmpeg, paths
from yap_studio.project.model import load_words
from yap_studio.stages.cut import _selected_stream
from yap_studio.stages.edges import heard_words
from yap_studio.voice.model import VoiceError
from yap_studio.voice.splice import RATE, _snap


def take_words(context, take):
    """Refuse donor words that no longer describe the recorded take."""
    target = context.project_dir / f"ingest/{take}/words.json"
    words = load_words(target)
    if words.take != take or words.take_sha256 != context.take_sha256(take):
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


def _span(sound, offset, first, last):
    a, b = (_snap(sound, round((time + offset) * RATE)) for time in (first, last))
    return sound[a:b]


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
                    candidates.append((rank, origin))
                    break
    decoded = {}
    for _, origin in sorted(candidates, key=lambda item: item[0]):
        take = origin["take"]
        if take not in decoded:
            decoded[take] = take_sound(plan.context, take)
        sound, offset = decoded[take]
        span = _span(sound, offset, origin["start_s"], origin["end_s"])
        if (len(span) and np.isfinite(span).all() and np.any(span)
                and np.max(np.abs(span)) < 0.999):
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
    sound = _span(sound, offset, origin["start_s"], origin["end_s"])
    identity = hashlib.sha256(json.dumps(origin, sort_keys=True).encode()).hexdigest()
    return sound, identity
