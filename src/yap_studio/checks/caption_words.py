"""Compare the final's speech with its caption words."""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import tempfile
import unicodedata
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from yap_studio import audio as audio_tools
from yap_studio import cache, paths
from yap_studio.checks.caption_layout import CaptionWord
from yap_studio.checks.model import Finding, format_time
from yap_studio.errors import YapError
from yap_studio.sfx import bleep
from yap_studio.transcript.backends import Backend
from yap_studio.transcript.model import Transcript
from yap_studio.transcript.words import OPTIONAL, norm_words, unicode_words

CHECK = "captions.words"


def heard_transcript(audio: Path, backend: Backend) -> Transcript:
    """Transcribe decoded voice once per audio and speech engine."""
    rate, sound = audio_tools.decode(audio, sr=16_000, mono=True)
    with tempfile.TemporaryDirectory(prefix="yap-heard-") as temporary:
        wav = Path(temporary) / "heard.wav"
        with wave.open(str(wav), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes((np.clip(sound, -1, 1) * 32767).astype("<i2").tobytes())
        digest = hashlib.sha256(wav.read_bytes())
        digest.update(json.dumps([backend.name, backend.model, backend.version()]).encode())
        target = None
        try:
            target = cache.cache_dir("speech", "heard") / f"{digest.hexdigest()}.json"
            data = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("words"), list):
                raise ValueError("invalid cached transcript")
            cached = Transcript.from_dict(data)
            if any(not math.isfinite(word.start) or not math.isfinite(word.end)
                   or word.end < word.start for word in cached.words):
                raise ValueError("invalid cached word times")
            return cached
        except (OSError, ValueError, TypeError, KeyError, AttributeError,
                RecursionError, OverflowError):
            pass
        heard = backend.transcribe_many([wav])[wav]
        if target is not None:
            try:
                paths.atomic_write_text(
                    target, json.dumps(heard.to_dict(), ensure_ascii=False) + "\n",
                )
            except (OSError, ValueError, TypeError, YapError):
                pass
        return heard


def _tokens(words: Sequence[CaptionWord]) -> list[tuple[str, CaptionWord]]:
    return [(token, word) for word in words
            for normalized in (norm_words(word.text) if word.text.isascii()
                               else unicode_words(word.text, drop_apostrophes=True))
            if (token := normalized.replace("'", "")) not in OPTIONAL]


def _caption_tokens(word: CaptionWord) -> list[str | None]:
    parts = bleep.tokens(unicodedata.normalize("NFKC", word.text))
    if not any(bleep.masked(part) for part in parts):
        return [token for token, _ in _tokens([word])]
    return [token for part in parts for token in (
        [None] if bleep.masked(part) else
        [text for text, _ in _tokens([CaptionWord(part, word.start_s, word.end_s)])]
    )]


def _masked_speech(words: Sequence[CaptionWord], speech: list[tuple[str, CaptionWord]],
                   ) -> set[int]:
    """Use matching neighbours to narrow a mask inside a merged display word."""
    skipped = set()
    for word in words:
        expected = _caption_tokens(word)
        if None not in expected:
            continue
        local = [(index, text) for index, (text, heard) in enumerate(speech)
                 if word.start_s <= (heard.start_s + heard.end_s) / 2 <= word.end_s]
        matches = difflib.SequenceMatcher(None, expected, [text for _, text in local],
                                          autojunk=False).get_matching_blocks()
        anchors = {match.a + offset: match.b + offset
                   for match in matches for offset in range(match.size)}
        for index, token in enumerate(expected):
            if token is not None:
                continue
            first = max((heard + 1 for caption, heard in anchors.items() if caption < index),
                        default=0)
            last = min((heard for caption, heard in anchors.items() if caption > index),
                       default=len(local))
            skipped.update(heard for heard, _ in local[first:last])
    return skipped


def check_caption_words(
    words: Mapping[str, Sequence[CaptionWord]], heard: Transcript,
) -> list[Finding]:
    """Name words heard without captions and captions without spoken words."""
    all_words = sorted((word for phrase in words.values() for word in phrase),
                       key=lambda word: word.start_s)
    captions = [(token, word) for word in all_words for token in _caption_tokens(word)
                if token is not None]
    speech = _tokens([CaptionWord(word.text, word.start, word.end)
                      for word in sorted(heard.words, key=lambda word: word.start)])
    skipped = _masked_speech(all_words, speech)
    speech = [item for index, item in enumerate(speech) if index not in skipped]
    captioned = [token for token, _ in captions]
    heard_tokens = [token for token, _ in speech]
    matcher = difflib.SequenceMatcher(None, captioned, heard_tokens, autojunk=False)
    findings = []
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        expected, actual = captioned[i1:i2], heard_tokens[j1:j2]
        if operation == "equal" or "".join(expected) == "".join(actual):
            continue
        timed = speech[j1:j2] if operation == "insert" else captions[i1:i2]
        start = timed[0][1].start_s
        end = max(word.end_s for _, word in timed)
        when = format_time(start)
        if operation == "insert":
            message = f'Heard "{" ".join(actual)}" at {when} with no caption.'
        elif operation == "delete":
            message = f'Caption "{" ".join(expected)}" at {when} is not heard in the audio.'
        else:
            message = (f'Heard "{" ".join(actual)}" at {when} '
                       f'where the caption shows "{" ".join(expected)}".')
        action = (
            "Listen there. If the word is not spoken, fix the cut or the caption text, "
            "then run again."
            if operation == "delete" else
            "Listen there. If the word is spoken, keep it in the cut's words "
            "(undo the removal that dropped it), then run again."
        )
        findings.append(Finding(
            CHECK, "info" if operation == "replace" and len(expected) == len(actual) else "warn",
            message, time_s=start, suggested_action=action,
            evidence={"heard": actual, "captioned": expected, "start_s": start, "end_s": end},
        ))
    return findings
