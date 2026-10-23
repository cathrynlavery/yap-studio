"""Build deterministic caption phrases from timed words and resolved caption limits."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from yap_studio.errors import ValidationError
from yap_studio.plan import snap_frame
from yap_studio.style.loader import _contrast, caption_legibility
from yap_studio.text_rules import (
    CLAUSE_WORDS,
    FORWARD_WORDS,
    STOP_WORDS,
    _mapped_text_rules,
    number_merges,
    starts_sentence,
    vocabulary_fixes,
)
from yap_studio.transcript.words import title_words

CAPTION_LAST_WORD_HOLD_S = 1.4
CAPTION_MIN_DURATION_S = 0.35
CAPTION_PHRASE_GAP_S = 0.06

_PUNCTUATION = ".,!?;:\"'()[]{}<>…“”‘’«»—–-"
_CLOSERS = "\"')]}>'”’»"
_LOWERCASE_LOOKS = frozenset({"pill_karaoke", "weight_shift"})
_UPPERCASE_LOOKS = frozenset({
    "clip_wipe", "camera_follow", "glitch_rgb", "highlight", "neon_glow", "particle_burst",
})


def caption_layout_settings(
    captions: Mapping[str, Any], brand: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive caption layout values without changing the style."""
    settings = dict(captions)
    if settings["highlight_style"] != "giant":
        return settings
    size = min(180, round(settings["size_px"] * 1.5))
    characters = settings["max_characters_per_phrase"]
    settings.update(
        size_px=size, background_mode="box", background_opacity=1,
        background_color=brand.get("accent_color") or settings["background_color"],
        background_padding_px=max(settings["background_padding_px"], round(size * 0.18)),
        max_words_per_phrase=min(settings["max_words_per_phrase"], 3),
        max_characters_per_phrase=max(8, min(characters, math.floor(characters / 1.5))),
    )
    settings = caption_legibility(settings)
    if _contrast(settings["highlight_color"], settings["background_color"]) < 3:
        settings["highlight_color"] = settings["text_color"]
    return settings


def phrase_key(ids: Sequence[str], index: int) -> str:
    """Name a phrase by its words, falling back to its slot when ids are absent."""
    return (f"caption:{ids[0]}" if len(ids) == 1 else
            f"caption:{ids[0]}-{ids[-1]}" if ids else f"caption:{index}")


def phrase_keys(phrases: Sequence[Sequence[str]]) -> list[str]:
    """Name each occurrence, keeping the first key and suffixing later ones."""
    keys, seen = [], {}
    for index, ids in enumerate(phrases):
        key = phrase_key(ids, index)
        seen[key] = seen.get(key, 0) + 1
        keys.append(key if seen[key] == 1 else f"{key}#{seen[key]}")
    return keys


def caption_word_case(text: str, look: str, *, preserve_case: bool = False) -> str:
    """Apply a look's case without changing acronyms, I words, dotfiles or fixed names."""
    if look == "slam":
        return text.upper()
    token = text.strip(_PUNCTUATION)
    protected = (preserve_case or any(char.isupper() for char in token[1:])
                 or token == "I" or token.startswith(("I'", "I’")) or text.startswith("."))
    if protected:
        return text
    if look in _LOWERCASE_LOOKS:
        return text.lower()
    if look in _UPPERCASE_LOOKS:
        return text.upper()
    return text


@dataclass(frozen=True)
class CaptionWord:
    """A source word with caption-only visibility and optional name/case marking."""

    id: str
    text: str
    start: float
    end: float
    hidden: bool = False
    preserve_case: bool = False


@dataclass(frozen=True)
class PhraseWord:
    """Display text with the original timing and ids preserved through fixes."""

    id: str
    text: str
    start: float
    end: float
    source_ids: tuple[str, ...]
    preserve_case: bool = False


@dataclass(frozen=True)
class Phrase:
    """A timed caption with one optional stressed display word."""

    index: int
    words: tuple[PhraseWord, ...]
    start: float
    end: float
    stress_id: str | None

    @property
    def text(self) -> str:
        """Return display words separated by single spaces."""
        return " ".join(word.text for word in self.words)

    @property
    def word_ids(self) -> tuple[str, ...]:
        """Return display ids, including the first id of each merged fix."""
        return tuple(word.id for word in self.words)

    def to_dict(self) -> dict[str, Any]:
        """Return the portable representation shared with the style preview."""
        return {
            "ids": list(self.word_ids),
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "stress": self.stress_id,
        }


def _stripped(text: str) -> str:
    return text.strip(_PUNCTUATION).lower()


def _ending(text: str) -> str:
    return text.rstrip(_CLOSERS)


def _display_words(
    words: Sequence[CaptionWord], word_fixes: Mapping[str, str],
    vocabulary: Sequence[str] = (), fixed: set[int] | None = None,
    *, word_text_fixes: Mapping[str, str] | None = None,
) -> list[PhraseWord]:
    visible = sorted((word for word in words if not word.hidden), key=lambda word: word.start)
    word_text_fixes = word_text_fixes or {}
    project_keys = {key.casefold() for key in word_fixes}
    fixes = [
        (tuple(_stripped(part) for part in key.split()), replacement)
        for key, replacement in (list(word_fixes.items()) + [
            (key, value) for key, value in vocabulary_fixes(vocabulary).items()
            if key not in project_keys
        ])
        if key.split()
    ]
    fixes.sort(key=lambda fix: -len(fix[0]))
    normalized = [_stripped(word.text) for word in visible]
    display = []
    i = 0
    while i < len(visible):
        count, text = 1, visible[i].text
        preserve_case = visible[i].preserve_case
        matched_fix = False
        occurrence_fix = visible[i].id in word_text_fixes
        candidates = ([((normalized[i],), word_text_fixes[visible[i].id])]
                      if occurrence_fix else fixes)
        for parts, replacement in candidates:
            if not occurrence_fix and any(
                word.id in word_text_fixes for word in visible[i : i + len(parts)]
            ):
                continue
            if tuple(normalized[i : i + len(parts)]) == parts:
                matched_fix = True
                count = len(parts)
                last_text = visible[i + count - 1].text
                leading = visible[i].text[:len(visible[i].text) -
                                          len(visible[i].text.lstrip(_PUNCTUATION))]
                trailing = last_text[len(last_text.rstrip(_PUNCTUATION)) :]
                if replacement and replacement[-1] in _PUNCTUATION:
                    trailing = ""
                text = ("" if replacement.startswith(leading) else leading) + replacement + trailing
                preserve_case = (any(word.preserve_case for word in visible[i : i + count])
                                 or any(char.isupper() for char in replacement))
                break
        if not matched_fix and vocabulary:
            for entry in vocabulary:
                for ending in ("'s", "’s"):
                    if normalized[i] == entry.casefold() + ending:
                        last_text = visible[i].text
                        leading = last_text[:len(last_text) - len(last_text.lstrip(_PUNCTUATION))]
                        trailing = last_text[len(last_text.rstrip(_PUNCTUATION)):]
                        text = leading + entry + ending + trailing
                        preserve_case = True
                        if fixed is not None:
                            fixed.add(len(display))
                        break
        if fixed is not None and (matched_fix or text != visible[i].text):
            fixed.add(len(display))
        matched = visible[i : i + count]
        display.append(
            PhraseWord(
                id=matched[0].id,
                text=text,
                start=matched[0].start,
                end=matched[-1].end,
                source_ids=tuple(word.id for word in matched),
                preserve_case=preserve_case,
            )
        )
        i += count
    return display


def kinetic_word_times(
    lines: Sequence[str], words: Sequence[CaptionWord], start_s: float, end_s: float,
    word_fixes: Mapping[str, str], *, element_id: str = "hook", vocabulary: Sequence[str] = (),
    word_text_fixes: Mapping[str, str] | None = None,
) -> list[float]:
    """Match title words to fixed speech inside the title window on the output clock."""
    def match_times(
        targets: Sequence[str], spoken: Sequence[tuple[str, float]],
    ) -> tuple[list[float], str | None]:
        tokens = [token for token, _ in spoken]
        cursor = 0
        times = []
        leading_symbols = 0
        for word in targets:
            target = title_words(word)
            if not target:
                if times:
                    times.append(times[-1])
                else:
                    leading_symbols += 1
                continue
            for index in range(cursor, len(spoken) - len(target) + 1):
                if tokens[index:index + len(target)] == target:
                    time = max(0, round((spoken[index][1] - start_s) * 30)) / 30
                    if not times:
                        times.extend([time] * leading_symbols)
                    times.append(time)
                    cursor = index + len(target)
                    break
            else:
                return [], word
        return times, None

    written = [word for line in lines for word in line.split()]
    fps = Fraction(30)
    start_frame = snap_frame(Fraction(str(start_s)), fps)
    end_frame = snap_frame(Fraction(str(end_s)), fps)
    speech = [(text, word.start) for word in _display_words(
        words, word_fixes, word_text_fixes=word_text_fixes)
              if start_frame <= snap_frame(Fraction(str(word.start)), fps) < end_frame
              for text in word.text.split()]
    times, missing = match_times(written, [(token, start) for text, start in speech
                                          for token in title_words(text)])
    parameter = f"title.{element_id}.kinetic"
    if missing is not None:
        mapped_speech = _mapped_text_rules(" ".join(text for text, _ in speech), vocabulary)
        mapped_title = _mapped_text_rules(" ".join(written), vocabulary)
        matched, rules_missing = match_times(
            [text for text, _ in mapped_title],
            [(token, speech[indices[0]][1]) for text, indices in mapped_speech
             for token in title_words(text)],
        )
        if rules_missing is not None or not matched:
            raise ValidationError(parameter, f'"{missing}" is not spoken between {start_s:g} s '
                                  f'and {end_s:g} s; anchor the title where the phrase is said '
                                  'or use the spoken words')
        times = [0.0] * len(written)
        for time, (_, indices) in zip(matched, mapped_title, strict=True):
            for index in indices:
                times[index] = time
    if not times:
        raise ValidationError(parameter, "use spoken words in the kinetic title")
    duration = end_s - start_s
    exit_s = min(0.25, duration / 2)
    if duration - exit_s - times[-1] < 0.35 - 1e-9:
        raise ValidationError(parameter, "the last word needs 0.35 s before the exit fade; "
                              "raise hold_s or end the title later")
    return times


def kinetic_display_times(
    lines: Sequence[str], times: Sequence[float], vocabulary: Sequence[str],
) -> list[float]:
    """Carry spoken title word times onto the lines shown after brand and number rules."""
    shown: list[float] = []
    at = 0
    for line in lines:
        for text, indices in _mapped_text_rules(line, vocabulary):
            source_times = [times[at + index] for index in indices]
            count = len(text.split())
            shown.extend(source_times if count == len(source_times) else [source_times[0]] * count)
        at += len(line.split())
    return shown


def _segment_boundary(
    previous: PhraseWord, word: PhraseWord, forced_breaks: Collection[str],
    *, text_rules: bool = False,
) -> bool:
    return (
        _ending(previous.text).endswith((".", "!", "?"))
        or word.start - previous.end > 1.0
        or any(source_id in forced_breaks for source_id in word.source_ids)
        or (text_rules and starts_sentence(previous.text, word.text))
    )


def _segments(
    words: Sequence[PhraseWord], forced_breaks: Collection[str], *, text_rules: bool = False,
) -> list[list[PhraseWord]]:
    segments: list[list[PhraseWord]] = []
    for word in words:
        if not segments or _segment_boundary(segments[-1][-1], word, forced_breaks,
                                              text_rules=text_rules):
            segments.append([])
        segments[-1].append(word)
    return segments


def _cost(
    words: Sequence[PhraseWord], start: int, end: int, max_words: int, *, text_rules: bool = False,
) -> int:
    count = end - start
    cost = 10 + 2 * max(0, min(max_words, 3) - count)
    if count == 1 and len(words) != 1:
        cost += 30
    last = words[end - 1]
    if end != len(words) and (
        _stripped(last.text) in FORWARD_WORDS
        or (text_rules and any(char.isdigit() for char in last.text))
    ):
        cost += 40
    if _ending(last.text).endswith((",", ";", ":", "-", "–", "—")):
        cost -= 15
    for i in range(start, end - 1):
        if _ending(words[i].text).endswith((",", ";", ":")):
            cost += 25
        if words[i + 1].start - words[i].end > 0.3:
            cost += 20
    if start != 0 and _stripped(words[start].text) in CLAUSE_WORDS:
        cost -= 8
    return cost


def _split(
    words: Sequence[PhraseWord], max_words: int, max_chars: int, *, text_rules: bool = False,
) -> list[tuple[PhraseWord, ...]]:
    n = len(words)
    costs = [0] * (n + 1)
    breaks = [n] * n
    for start in range(n - 1, -1, -1):
        best_cost: int | None = None
        characters = 0
        for end in range(start + 1, min(n, start + max_words) + 1):
            characters += len(words[end - 1].text) + (1 if end > start + 1 else 0)
            if characters > max_chars and end > start + 1:
                break
            cost = _cost(words, start, end, max_words, text_rules=text_rules) + costs[end]
            # Each suffix already wins its own ties; a longer first phrase wins this tie.
            if best_cost is None or cost < best_cost or (cost == best_cost and end > breaks[start]):
                best_cost = cost
                breaks[start] = end
        assert best_cost is not None
        costs[start] = best_cost
    phrases = []
    start = 0
    while start < n:
        end = breaks[start]
        phrases.append(tuple(words[start:end]))
        start = end
    return phrases


def _stress(words: Sequence[PhraseWord], stress_ids: Collection[str]) -> str | None:
    for word in words:
        if any(source_id in stress_ids for source_id in word.source_ids):
            return word.id
    best_id, best_score = None, -math.inf
    for word in words:
        letters = len(re.findall(r"[A-Za-z0-9]", word.text))
        if letters <= 2 or _stripped(word.text) in STOP_WORDS:
            continue
        score = (word.end - word.start) / letters
        if score > best_score:
            best_id, best_score = word.id, score
    return best_id


def _milliseconds(seconds: float) -> float:
    return math.floor(seconds * 1000 + 0.5) / 1000


def caption_phrase_end(
    phrase_start_s: float, last_word_start_s: float, *, next_start_s: float | None = None,
) -> float:
    """Bound the last-word hold by the next phrase, preserving the minimum duration.

    Transcript ends may include trailing silence. The builder uses this end;
    verification requires words only through the earlier of their end and this
    bound. A following phrase can override both the minimum duration and gap.
    """
    end = max(last_word_start_s + CAPTION_LAST_WORD_HOLD_S,
              phrase_start_s + CAPTION_MIN_DURATION_S)
    if next_start_s is not None:
        end = max(min(next_start_s - CAPTION_PHRASE_GAP_S,
                      last_word_start_s + CAPTION_LAST_WORD_HOLD_S),
                  phrase_start_s + CAPTION_MIN_DURATION_S)
        end = min(end, next_start_s)
    return end


def build_phrases(
    words: Sequence[CaptionWord],
    captions: Mapping[str, Any],
    *,
    word_fixes: Mapping[str, str] | None = None,
    word_text_fixes: Mapping[str, str] | None = None,
    stress_ids: Collection[str] = (),
    forced_breaks: Collection[str] = (),
    text_rules: bool = False,
    vocabulary: Sequence[str] = (),
) -> list[Phrase]:
    """Apply display fixes, split by integer costs, choose stress and time each phrase.

    Hidden words are dropped before fixes. Breaks and manual stress can refer to
    any source id in a merged word. Equal starts retain input order; equal split
    costs favor phrase word counts lexicographically, longest first.
    """
    max_words = captions["max_words_per_phrase"]
    max_chars = captions["max_characters_per_phrase"]
    for limit in (max_words, max_chars):
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("caption phrase limits must be positive integers")
    fixed: set[int] = set()
    display = _display_words(words, word_fixes or {}, vocabulary if text_rules else (), fixed,
                             word_text_fixes=word_text_fixes)
    if text_rules:
        merges = []
        start = 0
        for boundary in range(len(display) + 1):
            if (boundary == len(display) or boundary in fixed
                    or boundary > 0 and _segment_boundary(
                        display[boundary - 1], display[boundary], forced_breaks,
                        text_rules=True)):
                merges.extend((start + first, start + end, replacement)
                              for first, end, replacement in number_merges(
                                  [word.text for word in display[start:boundary]]))
                start = boundary + 1 if boundary in fixed else boundary
        for first, end, replacement in reversed(merges):
            sources = display[first:end]
            display[first:end] = [PhraseWord(
                sources[0].id, replacement, sources[0].start, sources[-1].end,
                tuple(source_id for word in sources for source_id in word.source_ids),
                any(word.preserve_case for word in sources),
            )]
    groups = [
        phrase
        for segment in _segments(display, forced_breaks, text_rules=text_rules)
        for phrase in _split(segment, max_words, max_chars, text_rules=text_rules)
    ]
    phrases = []
    for index, group in enumerate(groups):
        start = group[0].start
        next_start = groups[index + 1][0].start if index + 1 < len(groups) else None
        end = caption_phrase_end(start, group[-1].start, next_start_s=next_start)
        phrases.append(
            Phrase(
                index, group, _milliseconds(start), _milliseconds(end), _stress(group, stress_ids)
            )
        )
    return phrases
