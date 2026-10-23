"""Pure opening push endpoints on the output frame clock."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from yap_studio.plan import snap_frame
from yap_studio.project.model import CutWord

OPENING_SCALE = 1.3
OPENING_MIN_FRAMES = 24
OPENING_MAX_FRAMES = 120
OPENING_REFERENCE_FRAMES = 31


@dataclass(frozen=True)
class OpeningEnd:
    """The word and rule that end an opening push."""

    frame: int
    word: str
    text: str
    reason: str


def end_scale(end_frame: int, target: float = OPENING_SCALE) -> float:
    """Keep the reference speed on short lines and reach the target on longer ones."""
    return min(target, target ** (end_frame / OPENING_REFERENCE_FRAMES))


def opening_end(words: Sequence[CutWord], cuts: Sequence[int]) -> OpeningEnd | None:
    """Choose a fitting sentence or cut, otherwise the first fitting clause end."""
    if not words:
        return None
    starts = [snap_frame(Fraction(str(word.start)), Fraction(30)) for word in words]
    minimum = starts[0] + OPENING_MIN_FRAMES
    maximum = starts[0] + OPENING_MAX_FRAMES
    line = clause = None
    for index, word in enumerate(words):
        end = snap_frame(Fraction(str(word.end)), Fraction(30))
        next_start = starts[index + 1] if index + 1 < len(words) else maximum
        end = next((cut for cut in cuts if minimum <= cut <= maximum
                    and starts[index] < cut <= next_start), end)
        if minimum <= end <= maximum:
            text = word.text.rstrip().rstrip('"\'”’)]}')
            if line is None and text.endswith((".", "!", "?")):
                line = OpeningEnd(end, word.id, word.text, "line")
            if clause is None and text.endswith((",", ";", ":")):
                clause = OpeningEnd(end, word.id, word.text, "clause")
    cut_end = None
    for cut in cuts:
        if minimum <= cut <= maximum:
            word = next((word for word, start in reversed(list(zip(words, starts, strict=True)))
                         if start < cut), None)
            if word is not None:
                cut_end = OpeningEnd(cut, word.id, word.text, "cut")
                break
    if line is not None and (cut_end is None or line.frame <= cut_end.frame):
        return line
    return cut_end or clause
