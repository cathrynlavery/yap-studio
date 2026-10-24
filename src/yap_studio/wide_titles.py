"""Approved wide layouts, placed against every face in the layer's window."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

from yap_studio.canvas import canvas
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import face_clearance_px, gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.render import fonts
from yap_studio.render.text_metrics import text_width_em
from yap_studio.stages.captions import LayerPlan
from yap_studio.style.loader import ResolvedStyle


def _lines(text: str, font: str, width: int, maximum: int,
           min_words: int = 2) -> tuple[list[str], int]:
    words = text.split()
    if not words:
        raise ValidationError("title", "Supply the chosen text before rendering.")
    for size in range(maximum, 39, -1):
        candidates = [[" ".join(words)]]
        candidates.extend([" ".join(words[:i]), " ".join(words[i:])]
                          for i in range(min_words, len(words) - min_words + 1))
        fits = [(lines, [text_width_em(line, font, 700) * size for line in lines])
                for lines in candidates]
        fits = [(lines, widths) for lines, widths in fits if max(widths) <= width]
        if fits:
            lines, _ = min(fits, key=lambda item: (len(item[0]), max(item[1]) - min(item[1])))
            return lines, size
    raise ValidationError("title", "Please shorten the text to fit two balanced lines.")


def wide_layer_plan(
    kind: str, text: dict, style: ResolvedStyle, *, frames: int, start_frame: int = 0,
    faces: Sequence[Sequence[Box]] | None = None, reason: str | None = None,
    number: int = 1, total: int = 1,
    saved_box: Box | None = None,
) -> tuple[LayerPlan, dict]:
    """Keep content title-safe; translucent fades and navigation are decorative."""
    frame = canvas("horizontal")
    graphics = style.data["graphics"]
    font = graphics["font"] if kind == "name_tag" else style.data["hook"]["font"]
    margin = face_clearance_px(style.data)
    heads = [head_box(face) for row in faces or [] for face in row]
    variables = {"kind": kind, "font_name": font, "font_family": fonts.font_stack(font),
                 "paper_color": graphics["paper_color"], "ink_color": graphics["ink_color"],
                 "accent_color": graphics["accent_color"], "fps": 30,
                 "number": f"{number:02d}", "states": json.dumps([
                     "done" if i < number else "current" if i == number else "later"
                     for i in range(1, total + 1)]),
                 "name": text.get("name", ""), "role": text.get("role", ""),
                 "lines": "[]", "size_px": 52, "role_px": 30}
    if kind == "title":
        lines, size = _lines(text["title"], font, 1300, 84)
        width = math.ceil(max(text_width_em(line, font, 700) * size for line in lines)) + 4
        height = math.ceil(len(lines) * size * 1.05) + 8
        variables.update(lines=json.dumps(lines), size_px=size)
        candidates = [Box(120, frame.height - 120 - height, width, height)]
        candidates += [Box(frame.width - 120 - width, frame.height - 120 - height, width, height)]
    elif kind == "chapter":
        if style.data["chapters"]["style"] == "none":
            raise ValidationError("chapter", "Choose Title card under chapters on the style page.")
        right_first = (not heads or sum(head.x + head.w / 2 for head in heads) / len(heads)
                       < frame.width / 2)
        candidates = []
        for width in (640, 600, 560):
            lines, size = _lines(text["title"], font, width - 104, 64, min_words=1)
            height = 44 + 30 + 150 + 14 + math.ceil(len(lines) * size * 1.08) + 52
            positions = [frame.width - 150 - width, 120]
            if not right_first:
                positions.reverse()
            candidates.extend(Box(x, 300, width, height) for x in positions)
            candidates.extend(Box(x, 300, width, height) for x in (96, frame.width - 96 - width))
    elif kind == "name_tag":
        if not variables["name"].strip() or not variables["role"].strip():
            raise ValidationError("name_tag", "Supply a name and one short line.")
        width = math.ceil(max(text_width_em(variables["name"], font, 700) * 52,
                              text_width_em(variables["role"], font, 400) * 30)) + 86
        if width > 850:
            raise ValidationError("name_tag", "Please shorten the name or its short line.")
        height = 26 + 58 + 6 + 36 + 28
        candidates = [Box(x, frame.height - 150 - height, width, height)
                      for x in (96, frame.width - 96 - width)]
    else:
        raise ValidationError("wide title", "Choose title, chapter or name_tag.")
    if saved_box is not None:
        candidates = [saved_box]
    box = next((box for box in candidates if 96 <= box.x and box.x1 <= frame.width - 96
                and 108 <= box.y and box.y1 <= frame.height - 108
                and all(gap_px(box, head) >= margin for head in heads)), None)
    if box is None:
        raise ValidationError(kind, "Text would cover the face. Shorten it or choose a shot "
                              "with more room.")
    if kind == "chapter":
        lines, size = _lines(text["title"], font, int(box.w) - 104, 64, min_words=1)
        variables.update(lines=json.dumps(lines), size_px=size)
    variables.update(left=box.x, top=box.y, box_width=box.w, box_height=box.h)
    evidence = {"status": "clear" if faces is not None else "unchecked",
                "reason": "placed clear of faces" if faces is not None else reason,
                "margin_px": margin, "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h}}
    return LayerPlan("wide_titles", 0, 0, frame.width, frame.height, start_frame, frames,
                     variables, shape="horizontal"), evidence
