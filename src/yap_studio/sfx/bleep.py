"""Find listed words, mask display text and cover speech with a local tone."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from yap_studio.sfx import SfxError

if TYPE_CHECKING:
    from yap_studio.captions import PhraseWord

RATE = 48000
TONE_HZ = 1000
FADE_S = .005
PAD_S = .06
MERGE_GAP_S = .1
EXTEND_MAX_S = .12
BLEEP_VERSION = 1
_TOKEN = re.compile(
    r"(?:[^\W_]\*+|[^\W_]+)(?:['‘’\-](?:[^\W_]\*+|[^\W_]+|\*+))*['‘’]?", re.UNICODE,
)
_ENTRY = re.compile(r"[a-z][a-z'-]{0,39}")
_NON_TEXT = {"image", "video", "image_list", "video_list", "enum"}


class Word(Protocol):
    """The retained output word fields used by detection."""

    id: str
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Occurrence:
    """Matched tokens with output times and safe display evidence."""

    indices: tuple[int, ...]
    ids: tuple[str, ...]
    start_s: float
    end_s: float
    masked: tuple[str, ...]
    previous_end_s: float | None = None
    next_start_s: float | None = None


@dataclass(frozen=True)
class Span:
    """A half-open sample range covering one or more occurrences."""

    start: int
    end: int
    occurrences: tuple[Occurrence, ...]

    @property
    def masked(self) -> tuple[str, ...]:
        return tuple(text for item in self.occurrences for text in item.masked)


@lru_cache(maxsize=1)
def builtin_words() -> tuple[str, ...]:
    """Read the sorted, explicit English list."""
    data = json.loads(Path(__file__).with_name("swear-words.json").read_text(encoding="utf-8"))
    words = data.get("words")
    if (data.get("schema") != "yap-studio/swear-words@1" or data.get("language") != "en"
            or not isinstance(words, list) or not words
            or any(not isinstance(word, str) or not _ENTRY.fullmatch(word) for word in words)
            or words != sorted(set(words))):
        raise SfxError("Invalid built-in bleep word list.")
    return tuple(words)


def enabled(style_data: Mapping[str, Any]) -> bool:
    return style_data.get("music", {}).get("bleep_swears", False) is True


def listed(music: Mapping[str, Any]) -> frozenset[str]:
    return frozenset((set(builtin_words()) | set(music.get("bleep_words_add", [])))
                     - set(music.get("bleep_words_remove", [])))


def projection(music: Mapping[str, Any]) -> dict[str, Any]:
    """Identify display masking without coupling it to the rest of Sound."""
    return {"version": BLEEP_VERSION, "words": sorted(listed(music)),
            "tokens": "hyphen_pairs_nfkc"}


def normal(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("ı", "i")
    text = text.replace("’", "'").replace("‘", "'")
    first = next((i for i, char in enumerate(text) if char.isalnum()), len(text))
    last = next((i + 1 for i in range(len(text) - 1, first - 1, -1)
                 if text[i].isalnum()), first)
    return text[first:last]


def _whole_listed(token: str, words: frozenset[str]) -> bool:
    if masked(token):
        return False
    token = normal(token)
    return token in words or token.replace("-", "") in words


def _listed_parts(token: str, words: frozenset[str]) -> set[int]:
    parts = token.split("-")
    indices = {i for i, part in enumerate(parts) if _whole_listed(part, words)}
    for i, (first, second) in enumerate(zip(parts, parts[1:], strict=False)):
        if (normal(first) and normal(second) and not masked(first) and not masked(second)
                and _whole_listed(normal(first) + normal(second), words)):
            indices.update((i, i + 1))
    return indices


def is_listed(token: str, words: frozenset[str]) -> bool:
    return _whole_listed(token, words) or bool(_listed_parts(token, words))


def mask_word(text: str) -> str:
    first = True
    result = []
    for char in text:
        if char.isalnum():
            result.append(char if first else "*")
            first = False
        else:
            result.append(char)
    return "".join(result)


def _mask_token(text: str, words: frozenset[str]) -> str:
    if _whole_listed(text, words):
        return mask_word(text)
    indices = _listed_parts(text, words)
    return "-".join(mask_word(part) if i in indices else part
                    for i, part in enumerate(text.split("-")))


def find(words: Sequence[Word], listed: frozenset[str]) -> list[Occurrence]:
    """Prefer a two-token compound and use each token at most once."""
    result = []
    index = 0
    while index < len(words):
        count = 0
        if (index + 1 < len(words) and not masked(words[index].text)
                and not masked(words[index + 1].text) and normal(words[index].text)
                and normal(words[index + 1].text) and _whole_listed(
            normal(words[index].text) + normal(words[index + 1].text), listed,
        )):
            count = 2
        elif is_listed(words[index].text, listed):
            count = 1
        if not count:
            index += 1
            continue
        tokens = words[index:index + count]
        result.append(Occurrence(tuple(range(index, index + count)),
                                 tuple(word.id for word in tokens),
                                 min(word.start for word in tokens),
                                 max(word.end for word in tokens),
                                 tuple(mask_word(word.text) if count == 2 else
                                       _mask_token(word.text, listed) for word in tokens),
                                 words[index - 1].end if index else None,
                                 (words[index + count].start
                                  if index + count < len(words) else None)))
        index += count
    return result


@dataclass(frozen=True)
class _TextWord:
    id: str
    text: str
    start: float = 0
    end: float = 0


def display_words(words: Sequence[Word], word_fixes: Mapping[str, str] | None = None, *,
                  word_text_fixes: Mapping[str, str] | None = None,
                  ) -> list[PhraseWord]:
    """Apply caption fixes and retain the ids and times of merged source words."""
    from yap_studio.captions import CaptionWord, _display_words

    return _display_words([CaptionWord(word.id, word.text, word.start, word.end)
                           for word in words], word_fixes or {}, word_text_fixes=word_text_fixes)


def _display_matches(words: Sequence[PhraseWord], vocabulary: frozenset[str]
                     ) -> tuple[list[tuple[int, re.Match[str]]], list[Occurrence]]:
    tokens = [(index, token) for index, word in enumerate(words)
              for token in _TOKEN.finditer(word.text)]
    matches = find([_TextWord(str(index), token.group(), words[index].start, words[index].end)
                    for index, token in tokens], vocabulary)
    return tokens, matches


def find_display(words: Sequence[Word], vocabulary: frozenset[str], *,
                 word_fixes: Mapping[str, str] | None = None,
                 word_text_fixes: Mapping[str, str] | None = None) -> list[Occurrence]:
    """Detect corrected text and map each match back to retained source words."""
    word_text_fixes = word_text_fixes or {}
    display = display_words([_TextWord(str(index), word.text, word.start, word.end)
                             for index, word in enumerate(words)], word_fixes,
                            word_text_fixes={str(index): word_text_fixes[word.id]
                                             for index, word in enumerate(words)
                                             if word.id in word_text_fixes})
    tokens, matches = _display_matches(display, vocabulary)
    result = []
    for item in matches:
        indices = tuple(sorted({int(source) for index in item.indices
                                for source in display[tokens[index][0]].source_ids}))
        result.append(replace(item, indices=indices, ids=tuple(words[i].id for i in indices),
                              previous_end_s=words[indices[0] - 1].end if indices[0] else None,
                              next_start_s=(words[indices[-1] + 1].start
                                            if indices[-1] + 1 < len(words) else None)))
    return result


def mask_display_words(words: Sequence[PhraseWord], vocabulary: frozenset[str]
                       ) -> list[PhraseWord]:
    """Mask matched tokens before caption grouping, including split compounds."""
    tokens, matches = _display_matches(words, vocabulary)
    texts = [word.text for word in words]
    masks = {index: text for item in matches
             for index, text in zip(item.indices, item.masked, strict=True)}
    for index in sorted(masks, reverse=True):
        owner, token = tokens[index]
        texts[owner] = (texts[owner][:token.start()] + masks[index]
                        + texts[owner][token.end():])
    return [replace(word, text=text) for word, text in zip(words, texts, strict=True)]


def mask_text(text: str, listed: frozenset[str]) -> str:
    tokens = list(_TOKEN.finditer(text))
    matches = find([_TextWord(str(i), token.group()) for i, token in enumerate(tokens)], listed)
    masks = {index: text for item in matches
             for index, text in zip(item.indices, item.masked, strict=True)}
    for index in sorted(masks, reverse=True):
        token = tokens[index]
        text = text[:token.start()] + masks[index] + text[token.end():]
    return text


def tokens(text: str) -> list[str]:
    """Read word tokens, keeping mask initials and their asterisks together."""
    return [token.group() for token in _TOKEN.finditer(text)]


def masked(text: str) -> bool:
    first = next((i for i, char in enumerate(text) if char.isalnum()), len(text))
    end = len(text)
    while end > first and not (text[end - 1].isalnum() or text[end - 1] in "*'‘’-"):
        end -= 1
    core = text[first:end]
    pattern = r"[^\W\d_][*'‘’\-]*\*[*'‘’\-]*"
    parts = core.split("-")
    return (re.fullmatch(pattern, core) is not None or
            any(re.fullmatch(pattern, part) for part in parts) and
            all(not part or re.fullmatch(pattern, part) or _TOKEN.fullmatch(part)
                for part in parts))


def text_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [text for child in value for text in text_leaves(child)]
    if isinstance(value, dict):
        return [text for child in value.values() for text in text_leaves(child)]
    return []


def mask_slots(slots: dict, rules: dict, words: frozenset[str]) -> tuple[dict, list[str]]:
    """Mask string leaves while retaining media paths and enum values."""
    display = []

    def walk(value: Any, rule: dict) -> Any:
        if rule.get("type") in _NON_TEXT or "enum" in rule:
            return value
        if isinstance(value, str):
            text = mask_text(value, words)
            display.append(text)
            return text
        if isinstance(value, list):
            item_rules = rule.get("items", {})
            child_rule = (item_rules if "type" in item_rules else {"properties": item_rules})
            return [walk(child, child_rule) for child in value]
        if isinstance(value, dict):
            return {key: walk(child, rule.get("properties", {}).get(key, {}))
                    for key, child in value.items()}
        return value

    result = {key: walk(value, rules.get(key, {})) for key, value in slots.items()}
    for key, rule in rules.items():
        if key not in slots and "default" in rule:
            masked_default = walk(rule["default"], rule)
            if masked_default != rule["default"]:
                result[key] = masked_default
    return result, display


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))) if signal.size else 0.0


def spans(occurrences: Sequence[Occurrence], voice: np.ndarray, cut_s: float) -> list[Span]:
    limit = min(len(voice), round(cut_s * RATE))
    window, extension = round(FADE_S * RATE), round(EXTEND_MAX_S * RATE)
    result: list[Span] = []
    for item in occurrences:
        first, last = max(0, round(item.start_s * RATE)), min(limit, round(item.end_s * RATE))
        if first > last:
            continue
        reference = voice[first:last]
        if not reference.size:
            reference = voice[max(0, first - window // 2):min(limit, last + window // 2)]
        peak = float(np.max(np.abs(reference))) if reference.size else 0
        threshold = peak * 10 ** (-35 / 20)
        start, end = max(0, first - round(PAD_S * RATE)), min(limit, last + round(PAD_S * RATE))
        if start >= end:
            continue
        low, high = max(0, start - extension), min(limit, end + extension)
        if item.previous_end_s is not None:
            low = max(low, min(start, round(item.previous_end_s * RATE)))
        if item.next_start_s is not None:
            high = min(high, max(end, round(item.next_start_s * RATE)))
        while start > low and _rms(voice[max(low, start - window):start]) > threshold:
            start = max(low, start - window)
        while end < high and _rms(voice[end:min(high, end + window)]) > threshold:
            end = min(high, end + window)
        if result and start - result[-1].end < round(MERGE_GAP_S * RATE):
            previous = result[-1]
            result[-1] = Span(min(previous.start, start), max(previous.end, end),
                              (*previous.occurrences, item))
        else:
            result.append(Span(start, end, (item,)))
    return result


def tone(samples: int) -> np.ndarray:
    signal = np.sin(2 * np.pi * TONE_HZ * np.arange(samples) / RATE)
    fade = min(round(FADE_S * RATE), samples // 2)
    if fade:
        ramp = (1 - np.cos(np.linspace(0, np.pi, fade))) / 2
        signal[:fade] *= ramp
        signal[-fade:] *= ramp[::-1]
    return signal.astype(np.float32)


def render(occurrences: Sequence[Occurrence], voice: np.ndarray, cut_s: float,
           samples: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mute treated speech and make a stereo tone stem on the output clock."""
    muted = voice.copy()
    envelope = np.ones(len(voice), dtype=np.float32)
    stem = np.zeros((samples, 2), dtype=np.float32)
    records = []
    fade = round(FADE_S * RATE)
    for span in spans(occurrences, voice, cut_s):
        first, last = span.start, span.end
        envelope[first:last] = 0
        low, high = max(0, first - fade), min(len(voice), last + fade)
        envelope[low:first] = np.minimum(envelope[low:first],
                                        (first - np.arange(low, first)) / fade)
        envelope[last:high] = np.minimum(envelope[last:high],
                                        (np.arange(last, high) - last) / fade)
        # A union counts overlapping word times once and leaves gaps out of the level.
        covered = np.zeros(last - first, dtype=bool)
        for item in span.occurrences:
            start = max(first, round(item.start_s * RATE))
            end = min(last, round(item.end_s * RATE))
            covered[start - first:end - first] = True
        rms = _rms(voice[first:last][covered])
        if rms < 10 ** (-50 / 20):
            rms = _rms(voice[max(0, first - RATE // 2):min(len(voice), last + RATE // 2)])
        near = voice[max(0, first - RATE // 4):min(len(voice), last + RATE // 4)]
        peak = float(np.max(np.abs(near))) if near.size else 0
        signal = tone(last - first)
        source_rms = _rms(signal)
        source_peak = float(np.max(np.abs(signal))) if signal.size else 0
        gain = min(rms / source_rms, peak / source_peak) if source_rms and source_peak else 0
        signal *= gain
        stem[first:last] = signal[:, None]
        level = _rms(signal)
        records.append({"start_s": first / RATE, "end_s": last / RATE,
                        "masked": list(span.masked),
                        "tone_rms_dbfs": 20 * math.log10(level) if level else None})
    muted *= envelope[:, None]
    return muted, stem, {"version": BLEEP_VERSION, "tone_hz": TONE_HZ, "pad_s": PAD_S,
                         "count": len(occurrences), "spans": records}


def check(words: Sequence[Word], *, music: Mapping[str, Any], voice: np.ndarray,
          original_voice: np.ndarray, tone_stem: np.ndarray | None, phrases: dict,
          title_layers: Sequence[dict], timeline: Any,
          slot_rules: Mapping[str, dict] | None = None,
          word_fixes: Mapping[str, str] | None = None,
          word_text_fixes: Mapping[str, str] | None = None) -> dict:
    """Re-detect retained words and check the delivered voice and displayed text."""
    vocabulary = listed(music)
    occurrences = find_display(words, vocabulary, word_fixes=word_fixes,
                               word_text_fixes=word_text_fixes)
    audible, unmasked = [], []
    for item in occurrences:
        first = max(0, round(item.start_s * RATE) - round(PAD_S * RATE))
        last = min(len(original_voice), round(item.end_s * RATE) + round(PAD_S * RATE))
        missing_tone = False
        for index in item.indices:
            start, end = round(words[index].start * RATE), round(words[index].end * RATE)
            if start == end:
                start, end = first, last
            if np.any(original_voice[start:end]) and (
                tone_stem is None or not np.any(tone_stem[start:end])
            ):
                missing_tone = True
        if len(voice) < last or np.any(voice[first:last]) or missing_tone:
            audible.append({"time_s": item.start_s, "where": "audio",
                            "masked": list(item.masked)})
    caption_words = [_TextWord(str(index), token.group(), phrase["start"], phrase["end"])
                     for index, phrase in enumerate(phrases.get("phrases", []))
                     for token in _TOKEN.finditer(phrase["text"])]
    for item in find(caption_words, vocabulary):
        unmasked.append({"time_s": item.start_s, "where": f"caption {item.ids[0]}",
                         "masked": list(item.masked)})
    for layer in title_layers:
        kind = "card" if layer["kind"] == "graphic" else layer["kind"]
        for text in layer.get("display_text", []):
            if mask_text(text, vocabulary) != text:
                unmasked.append({"time_s": layer.get("start_frame", 0) / 30,
                                 "where": f"{kind} {layer['id']}",
                                 "masked": mask_text(text, vocabulary)})
    recorded = {layer["id"] for layer in title_layers if "display_text" in layer}
    for element in timeline.elements:
        if element.id in recorded:
            continue
        payload = element.payload
        kind = "card" if element.kind == "graphic" else element.kind
        if "letters" in payload:
            texts = text_leaves(payload["letters"].get("text", ""))
        elif element.kind == "graphic":
            slots = payload.get("slots", {})
            _, texts = mask_slots(slots, (slot_rules or {}).get(
                payload.get("template", ""), {}), frozenset())
            if (payload.get("template") == "file_drop" and not slots.get("name")
                    and slots.get("clip")):
                texts.append(Path(slots["clip"]).name)
        elif element.kind in {"title", "chapter"}:
            texts = text_leaves(payload.get("lines", [payload.get("title", "")]))
            texts += [payload.get("accent", "")]
        else:
            continue
        for text in texts:
            if mask_text(text, vocabulary) != text:
                unmasked.append({"time_s": element.start_frame / 30,
                                 "where": f"{kind} {element.id}",
                                 "masked": mask_text(text, vocabulary)})
    return {"id": "bleep.words", "pass": not audible and not unmasked,
            "words": sum(len(item.indices) for item in occurrences),
            "audible": audible, "unmasked": unmasked,
            "fix": "Run captions, graphics and assemble again with Bleep swear words on."}
