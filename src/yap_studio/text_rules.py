"""Pure display text rules shared by captions and opening titles."""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import combinations

FORWARD_WORDS = frozenset(
    "a an the to of in on at for with from by and or but so my your our their his her its "
    "this that these those is are was were be i we you they it's i'm i've i'll we're "
    "you're they're".split()
)
CLAUSE_WORDS = frozenset(
    "and but so because which that when while if then or who where although though unless "
    "until since".split()
)
STOP_WORDS = FORWARD_WORDS | CLAUSE_WORDS | frozenset(
    "just really very also not no yes have has had do does did can will would could should "
    "there here what about into out up all some any been being than them me us him".split()
)
SENTENCE_STARTERS = STOP_WORDS | frozenset(
    "why how alright okay ok well now yeah hey oh look honestly anyway right sure let's "
    "don't that's there's here's what's he she".split()
)
_SMALL = dict(zip("zero one two three four five six seven eight nine ten eleven twelve "
                  "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
                  range(20), strict=True))
_TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(),
                 range(20, 100, 10), strict=True))
_SCALES = {"billion": 1_000_000_000, "million": 1_000_000, "thousand": 1000}
_DECADES = dict(zip(
    "tens twenties thirties forties fifties sixties seventies eighties nineties".split(),
    range(10, 100, 10), strict=True,
))
_UNITS = frozenset(
    "second seconds minute minutes hour hours week weeks month months year years "
    "percent dollar dollars cent cents pound pounds kilo kilos kilogram kilograms mile miles "
    "kilometer kilometers kilometre kilometres foot feet inch inches meter meters metre metres "
    "rep reps set sets".split()
)
_PUNCTUATION = ".,!?;:\"'()[]{}<>…“”‘’«»—–"
_BOUNDARY = ".!?,;:"


def _parts(text: str) -> tuple[str, str, str]:
    first = len(text) - len(text.lstrip(_PUNCTUATION))
    last = len(text.rstrip(_PUNCTUATION))
    return text[:first], text[first:last].casefold(), text[last:]


def _under_hundred(parts: Sequence[str]) -> int | None:
    if len(parts) == 1:
        word = parts[0]
        if word in _SMALL:
            return _SMALL[word]
        if word in _TENS:
            return _TENS[word]
        pair = word.split("-")
        if len(pair) == 2 and pair[0] in _TENS and pair[1] in _SMALL and 0 < _SMALL[pair[1]] < 10:
            return _TENS[pair[0]] + _SMALL[pair[1]]
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _SMALL:
        value = _SMALL[parts[1]]
        if 0 < value < 10:
            return _TENS[parts[0]] + value
    return None


def _under_thousand(parts: Sequence[str]) -> int | None:
    if parts == ["hundred"]:
        return 100
    value = _under_hundred(parts)
    if value is not None:
        return value
    if len(parts) < 2 or parts[1] != "hundred":
        return None
    count = 1 if parts[0] == "a" else _SMALL.get(parts[0])
    if count is None or not 1 <= count <= 9:
        return None
    tail = parts[2:]
    if tail and tail[0] == "and":
        if len(tail) == 1:
            return None
        tail = tail[1:]
    remainder = _under_hundred(tail) if tail else 0
    return count * 100 + remainder if remainder is not None else None


def _cardinal(parts: Sequence[str]) -> int | None:
    if not parts:
        return None
    value = 0
    remainder = list(parts)
    for scale, multiplier in _SCALES.items():
        if scale not in remainder:
            continue
        position = remainder.index(scale)
        count = _under_thousand(remainder[:position])
        if count is None and remainder[:position] in (["a"], []):
            count = 1
        if count is None or count == 0:
            return None
        value += count * multiplier
        remainder = remainder[position + 1:]
        if remainder and remainder[0] == "and":
            if len(remainder) == 1:
                return None
            remainder = remainder[1:]
    tail = _under_thousand(remainder) if remainder else 0
    return value + tail if tail is not None else None


def _year(parts: Sequence[str]) -> int | None:
    if len(parts) < 2 or parts[0] not in {"nineteen", "twenty"}:
        return None
    if len(parts) == 2 and parts[1].startswith("oh-"):
        second = _SMALL.get(parts[1][3:])
    elif len(parts) == 2 and parts[1] == "oh":
        second = None
    elif len(parts) == 3 and parts[1] == "oh":
        second = _SMALL.get(parts[2])
    else:
        second = _under_hundred(parts[1:])
    if second is None or not (10 <= second <= 99 or parts[1] == "oh"):
        return None
    return (1900 if parts[0] == "nineteen" else 2000) + second


def _two_thousand_year(parts: Sequence[str]) -> int | None:
    if len(parts) < 3 or list(parts[:2]) != ["two", "thousand"]:
        return None
    remainder = parts[2:]
    if remainder and remainder[0] == "and":
        remainder = remainder[1:]
    year_part = _under_hundred(remainder)
    return 2000 + year_part if year_part is not None and 1 <= year_part <= 99 else None


def _decade(parts: Sequence[str]) -> str | None:
    if len(parts) != 2 or parts[0] not in {"nineteen", "twenty"}:
        return None
    decade = _DECADES.get(parts[1])
    if decade is None:
        return None
    return f"{(1900 if parts[0] == 'nineteen' else 2000) + decade}s"


def _decimal(parts: Sequence[str]) -> str | None:
    if "point" not in parts:
        return None
    point = parts.index("point")
    whole = _cardinal(parts[:point])
    fraction = parts[point + 1:]
    if whole is None or not fraction or any(_SMALL.get(word, 10) > 9 for word in fraction):
        return None
    return f"{whole:,}." + "".join(str(_SMALL[word]) for word in fraction)


def number_merges(tokens: Sequence[str]) -> list[tuple[int, int, str]]:
    """Return conservative number and unit replacements over display tokens."""
    parts = [_parts(token) for token in tokens]
    cores = [part[1] for part in parts]
    results: list[tuple[int, int, str]] = []
    counting: set[int] = set()
    run = 0
    for index, core in enumerate(cores + [""]):
        cardinal = index < len(cores) and _under_hundred([core]) is not None
        if cardinal and (index == 0 or parts[index - 1][2] in {"", ","}):
            continue
        year_run = (not any(parts[j][2] == "," for j in range(run, index - 1))
                    and _year(cores[run:index]) is not None)
        if index - run >= 3 and not year_run:
            counting.update(range(run, index))
        run = index if cardinal else index + 1
    i = 0
    while i < len(tokens):
        if i in counting:
            i += 1
            continue
        core = cores[i]
        if not core or (core.isdigit() and i + 1 >= len(tokens)):
            i += 1
            continue
        if core.isdigit():
            count, display_text = 1, core
        else:
            count, display_text = 0, ""
            for end in range(i + 1, min(len(tokens), i + 9) + 1):
                if end > i + 1 and parts[end - 2][2].endswith(tuple(_BOUNDARY)):
                    break
                candidate = cores[i:end]
                decade = _decade(candidate)
                decimal = _decimal(candidate)
                year = _year(candidate)
                if year is None:
                    year = _two_thousand_year(candidate)
                cardinal = _cardinal(candidate)
                if (decade is not None or decimal is not None or year is not None
                        or cardinal is not None):
                    count = end - i
                    if decade is not None:
                        display_text = decade
                    elif decimal is not None:
                        display_text = decimal
                    elif year is not None:
                        display_text = str(year)
                    else:
                        assert cardinal is not None
                        display_text = f"{cardinal:,}"
            if not count:
                i += 1
                continue
        if count == 1 and i + 1 < len(tokens) and not parts[i][2]:
            next_value = _under_hundred([cores[i + 1]])
            value = _under_hundred([core])
            if (value is not None and next_value is not None
                    and _under_hundred([core, cores[i + 1]]) is None):
                if (next_value == value + 1 or
                        value % 5 == next_value % 5 == 0 and next_value > value):
                    text = f"{value}-{next_value}"
                    results.append((i, i + 2, parts[i][0] + text + parts[i + 1][2]))
                i += 2
                continue
        unit_end = i + count
        unit = cores[unit_end] if unit_end < len(tokens) and not parts[unit_end - 1][2] else ""
        if (unit == "per" and unit_end + 1 < len(tokens) and cores[unit_end + 1] == "cent"
                and not parts[unit_end][2].endswith(tuple(_BOUNDARY))):
            unit_end += 2
            suffix = "%"
        elif unit == "percent":
            unit_end += 1
            suffix = "%"
        elif unit in {"dollar", "dollars"}:
            unit_end += 1
            suffix = "$"
        else:
            suffix = ""
        if core == "one" and count == 1 and not suffix and unit not in _UNITS:
            # Convert one beside a converting number on the other side of a range.
            if not (i + 2 < len(tokens) and cores[i + 1] in {"or", "to"}
                    and _under_hundred([cores[i + 2]]) is not None):
                i += 1
                continue
        if core.isdigit() and not suffix:
            i += 1
            continue
        replacement = ("$" + display_text if suffix == "$" else display_text + suffix)
        end = unit_end if suffix else i + count
        replacement = parts[i][0] + replacement + parts[end - 1][2]
        results.append((i, end, replacement))
        i = end
    return results


def balance_lines(
    widths: Sequence[float], space: float, max_width: float, *, max_lines: int = 2,
) -> tuple[int, ...]:
    """Choose breaks that minimize the widest row without an orphan last word."""
    n = len(widths)
    if n <= 2 or sum(widths) + max(0, n - 1) * space <= max_width:
        return ()
    candidates = []
    for count in range(1, min(max_lines, n - 1)):
        for breaks in combinations(range(1, n), count):
            edges = (0, *breaks, n)
            if n - breaks[-1] < 2:
                continue
            rows = [sum(widths[a:b]) + (b - a - 1) * space
                    for a, b in zip(edges, edges[1:], strict=False)]
            candidates.append((max(rows), tuple(-edge for edge in breaks), breaks))
    return min(candidates)[2] if candidates else ()


def starts_sentence(previous: str, word: str) -> bool:
    """Recognize a sentence break from punctuation or a closed starter list."""
    prior = previous.rstrip("\"')]}’”»")
    if prior.endswith((".", "!", "?")):
        return True
    if prior.endswith((",", ";", ":", "-", "–", "—")):
        return False
    core = _parts(word)[1]
    visible = word.lstrip(_PUNCTUATION)
    if core in {"i", "i'm", "i've", "i'll", "i'd"} or core not in SENTENCE_STARTERS:
        return False
    return bool(visible and visible[0].isupper() and
                (len(visible) == 1 or not visible.strip(_PUNCTUATION).isupper()))


def vocabulary_fixes(words: Sequence[str]) -> dict[str, str]:
    """Map vocabulary spellings to case-insensitive fix keys."""
    return {word.casefold(): word for word in words}


def _distance(first: str, second: str) -> int:
    rows = [[0] * (len(second) + 1) for _ in range(len(first) + 1)]
    for i in range(len(first) + 1):
        rows[i][0] = i
    for j in range(len(second) + 1):
        rows[0][j] = j
    for i in range(1, len(first) + 1):
        for j in range(1, len(second) + 1):
            rows[i][j] = min(rows[i - 1][j] + 1, rows[i][j - 1] + 1,
                             rows[i - 1][j - 1] + (first[i - 1] != second[j - 1]))
            if (i > 1 and j > 1 and first[i - 1] == second[j - 2]
                    and first[i - 2] == second[j - 1]):
                rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)
    return rows[-1][-1]


def near_vocabulary(texts: Sequence[str], vocabulary: Sequence[str]) -> list[tuple[int, int, str]]:
    """Find close caption spellings and split brand names."""
    findings = []
    cores = [_parts(text)[1] for text in texts]
    for index, core in enumerate(cores):
        for entry in vocabulary:
            target = entry.casefold()
            base = core[:-2] if core.endswith(("'s", "’s")) else core
            literal = texts[index].strip(_PUNCTUATION)
            if base == target:
                suffix = "’s" if literal.endswith("’s") else "'s" if literal.endswith("'s") else ""
                if literal != entry + suffix:
                    findings.append((index, 1, entry))
            elif len(target) >= 5 and min(
                _distance(base, target),
                _distance(base.replace("gh", ""), target.replace("gh", ""))
            ) <= (1 if len(target) <= 7 else 2):
                findings.append((index, 1, entry))
            if (index + 1 < len(cores) and core + cores[index + 1] == target
                    and " " not in entry):
                findings.append((index, 2, entry))
    return findings


def _mapped_text_rules(
    text: str, vocabulary: Sequence[str],
) -> list[tuple[str, tuple[int, ...]]]:
    tokens = text.split()
    fixes = vocabulary_fixes(vocabulary)
    ordered_fixes = sorted(fixes.items(), key=lambda fix: -len(fix[0].split()))
    normalized = [_parts(token)[1] for token in tokens]
    display: list[tuple[str, tuple[int, ...]]] = []
    fixed: set[int] = set()
    i = 0
    while i < len(tokens):
        match = next(((len(parts), replacement) for key, replacement in ordered_fixes
                      if (parts := key.split()) == normalized[i:i + len(parts)]), None)
        if match:
            length, replacement = match
            fixed.add(len(display))
            display.append((_parts(tokens[i])[0] + replacement +
                            _parts(tokens[i + length - 1])[2], tuple(range(i, i + length))))
            i += length
        else:
            core = normalized[i]
            possessive = next((entry for key, entry in ordered_fixes
                               if core in {key + "'s", key + "’s"}), None)
            if possessive:
                fixed.add(len(display))
                ending = "’s" if core.endswith("’s") else "'s"
                display.append((_parts(tokens[i])[0] + possessive + ending +
                                _parts(tokens[i])[2], (i,)))
            else:
                display.append((tokens[i], (i,)))
            i += 1
    merges = []
    start = 0
    for boundary in sorted((*fixed, len(display))):
        merges.extend((start + first, start + end, replacement)
                      for first, end, replacement in number_merges(
                          [item[0] for item in display[start:boundary]]))
        start = boundary + 1
    for start, end, replacement in reversed(merges):
        sources = tuple(index for _, indices in display[start:end] for index in indices)
        display[start:end] = [(replacement, sources)]
    return display


def apply_text_rules(text: str, vocabulary: Sequence[str]) -> str:
    """Apply brand spelling and numbers to one title line."""
    return " ".join(word for word, _ in _mapped_text_rules(text, vocabulary))


def apply_title_rules(
    lines: Sequence[str], accent: str, vocabulary: Sequence[str],
) -> tuple[list[str], str]:
    """Convert title lines and keep the accent on their source words."""
    mapped = [_mapped_text_rules(line, vocabulary) for line in lines]
    converted = [" ".join(word for word, _ in display) for display in mapped]
    if accent:
        for line, display, rendered in zip(lines, mapped, converted, strict=True):
            position = line.find(accent)
            if position < 0:
                continue
            spans = list(re.finditer(r"\S+", line))
            touched = {index for index, match in enumerate(spans)
                       if match.start() < position + len(accent) and match.end() > position}
            if not touched:
                break
            if accent in rendered and all(
                any(indices == (index,) and word == spans[index].group()
                    for word, indices in display) for index in touched
            ):
                return converted, accent
            return converted, " ".join(word for word, indices in display
                                         if touched.intersection(indices))
    return converted, apply_text_rules(accent, vocabulary)
