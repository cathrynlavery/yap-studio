"""Fixed Inter fitting and spoken-number checks for a creator's Text hook."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from yap_studio.brand_words import list_words
from yap_studio.errors import ValidationError
from yap_studio.render.text_metrics import text_width_em
from yap_studio.sfx import bleep
from yap_studio.text_rules import apply_text_rules, apply_title_rules

TEXT_HOOK_ID = "agent-text-hook"
MAX_WIDTH = 1080 * .84
PADDING_X = 26
PADDING_Y = 8
TRACKING_EM = -.01
LINE_HEIGHT = 1.3
_NUMBER = re.compile(r"(?<!\w)(\$?)(-?[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)(%?)")


def require_top_text(context):
    """Refuse a final until the creator chooses its title or whole-video label."""
    from yap_studio.project.model import parse_document

    style = context.render_style().data
    hook = style["hook"]
    if hook["style"] == "none":
        return
    # Existing projects keep their title layers until the creator answers once.
    if hook["style"] != "text_hook" and not hook.get("answered", False):
        return
    kind = "text_hook" if hook["style"] == "text_hook" else "opening"
    try:
        choice = parse_document("text-hook", context.json("plans/text-hook.json"))
    except FileNotFoundError:
        choice = None
    if choice is None or choice.chosen is None or choice.kind != kind:
        raise ValidationError("hook", "Choose the text at the top before rendering. "
                              "Show three ideas and wait for a pick or the creator's own text. "
                              "Save it as chosen in plans/text-hook.json.")
    if kind == "opening":
        titles = [item for item in context.resolved_timeline().elements if item.kind == "title"]
        vocabulary = tuple(list_words())
        displayed = [" ".join(apply_title_rules(item.payload["lines"], "", vocabulary)[0])
                     for item in titles]
        if bleep.enabled(style):
            masked = bleep.listed(style["music"])
            displayed = [bleep.mask_text(text, masked) for text in displayed]
        if not displayed or any(" ".join(text.split()) != " ".join(choice.chosen.split())
                                for text in displayed):
            raise ValidationError("hook", "Use the chosen opening text in the title's lines, "
                                  "including after number, spelling and masking rules, then render "
                                  "again. Do not invent or replace the chosen text.")


@dataclass(frozen=True)
class TextHookFit:
    """The exact lines, font size and measured widths used by the title layer."""

    lines: tuple[str, ...]
    size_px: int
    widths: tuple[float, ...]


def fit_text_hook(text: str) -> TextHookFit:
    """Use the largest readable size, balancing two lines without lone words."""
    words = text.split() if isinstance(text, str) else []
    if words and len(text) <= 1000:
        candidates = [(" ".join(words),)]
        candidates.extend((" ".join(words[:at]), " ".join(words[at:]))
                          for at in range(2, len(words) - 1))
        measured = [(lines, tuple(text_width_em(line, "Inter", 600, tracking_em=TRACKING_EM)
                                  for line in lines)) for lines in candidates]
        for size in range(72, 59, -1):
            fits = [(lines, tuple(width * size for width in widths))
                    for lines, widths in measured
                    if max(widths) * size + 2 * PADDING_X <= MAX_WIDTH
                    and min(widths) >= .65 * max(widths)]
            if fits:
                lines, widths = min(fits, key=lambda item: (
                    len(item[0]), max(item[1]) - min(item[1]), max(item[1])))
                return TextHookFit(lines, size, widths)
    raise ValidationError("text_hook", "Please use shorter text that fits on one line, or two "
                          "balanced lines with at least two words each.")


def unspoken_numbers(text: str, spoken: str) -> bool:
    """Keep numeric, money and percentage claims grounded in the take."""
    def claims(value: str) -> set[tuple[Decimal, str, str]]:
        normalized = apply_text_rules(value, ())
        # Bare one remains a word in the display rules.
        normalized = re.sub(r"\bone\b", "1", normalized, flags=re.IGNORECASE)
        return {(Decimal(number.replace(",", "")), money, percent)
                for money, number, percent in _NUMBER.findall(normalized)}

    heard = claims(spoken)
    return any(not any(value == number and (not money or money == currency)
                       and (not percent or percent == unit)
                       for number, currency, unit in heard)
               for value, money, percent in claims(text))
