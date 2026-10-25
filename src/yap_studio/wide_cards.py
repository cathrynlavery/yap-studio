"""Wide floating panels, fitted locally and placed against the whole face window."""

from __future__ import annotations

import copy
import html
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from yap_studio import cards, paths
from yap_studio.canvas import canvas, unsupported
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import face_clearance_px, gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.render import hyperframes
from yap_studio.render.text_metrics import text_width_em
from yap_studio.sfx import bleep
from yap_studio.style import ResolvedStyle
from yap_studio.wide_titles import _lines

# Paper width and initial layout budget; the final paper hugs the fitted content.
LAYOUTS = {
    "terminal": (660, 480), "stat": (660, 420), "checklist": (660, 520),
    "text_tag": (660, 360), "image_card": (660, 520), "transfer": (660, 440),
    "image_compare": (660, 520), "steps": (660, 360), "flow": (660, 520),
    "chart": (660, 520), "quote": (660, 460), "progress_stat": (660, 420),
    "specs_list": (660, 520), "callout_circle": (660, 520), "name_tag": (660, 340),
    "success_check": (660, 360), "edit_timeline": (660, 520), "storyboard": (660, 520),
    "jiggle_delete": (660, 520), "file_drop": (660, 460), "video_card": (660, 540),
    "prompt_bar": (660, 360), "comment_cta": (660, 420), "comment_drop": (660, 460),
    "icon_strike": (660, 420), "choices": (660, 520),
    "hf:char-slam-explode": (660, 420), "hf:conic-progress-ring": (660, 460),
}
REFUSED = {
    "playing_grid": "Use one video_card; playing grids need a full-frame layout",
    "reel_feed": "Use one video_card; reel feeds need a phone frame",
    "gravity_drop": "Use a checklist; falling words need a full-frame layout",
    "big_number": "Use a stat card; behind-speaker numbers need a wide person layout",
}


def require_template(template: str) -> None:
    if template not in LAYOUTS:
        unsupported(f"floating card {template}", REFUSED.get(
            template, "Choose an available floating card from docs/limits.md"))


def card_plan(template: str, slots: Mapping[str, object], style: ResolvedStyle, *,
              frames: int | None = None, fps: int = 30,
              project: cards.CardProject | None = None) -> cards.CardPlan:
    require_template(template)
    width, height = LAYOUTS[template]
    chosen = copy.deepcopy(style)
    chosen.data["graphics"].update(card_width_px=width - 104, card_motion="none")
    if template in {"transfer", "storyboard", "specs_list"}:
        chosen.data["graphics"]["card_width_px"] = width
    chosen.data["internal"].update(key_text_top_px=0, key_text_bottom_px=height)
    native_slots = slots
    if template == "steps":
        if bleep.enabled(style.data):
            slots, _ = bleep.mask_slots(dict(slots), cards.load_template(template)["slots"],
                                        bleep.listed(style.data["music"]))
        # Validate the real labels, but use the portrait adapter only for its cue clock.
        values, _ = cards._slots(slots, cards.load_template(template)["slots"])
        native_slots = {**slots, "steps": ["x"] * len(values["steps"])}
    if template.startswith("hf:"):
        from yap_studio.hf_registry import catalog

        min_w, min_h = catalog.item(template[3:])["min_panel"]
        if min_w > width or min_h > height - cards.PAD_TOP - cards.PAD_BOTTOM:
            unsupported(f"floating registry card {template}",
                        "Choose another registry card that fits the floating panel")
        chosen.data["graphics"]["card_width_px"] = max(width - 104, min_w)
    plan = cards.card_layer_plan(template, native_slots, chosen, frames=frames, fps=fps,
                                 project=project, max_width_px=chosen.data["graphics"][
                                     "card_width_px"] + 80,
                                 max_height_px=460 if template == "choices" else 544)
    plan, crop = _content_plan(plan, slots, style)
    variables = dict(plan.variables)
    cx, cy, cw, ch = crop
    ch = math.ceil(ch / 2) * 2
    paper_width, paper_height = max(660, math.ceil(cw + 104)), ch + 100
    variables.update(wide_native_width_px=plan.width, wide_native_height_px=plan.height,
                     wide_content_x_px=cx, wide_content_y_px=cy,
                     wide_content_width_px=cw, wide_content_height_px=math.ceil(ch),
                     wide_paper_width_px=paper_width, wide_paper_height_px=paper_height)
    if not plan.registry:
        variables.update(card_radius_px=24, card_shadow=(
            "0 12px 30px rgb(0 0 0 / 0.18)" if plan.windows else "none"))
    registry = copy.deepcopy(plan.registry)
    if registry:
        registry["tokens"]["--fg"] = style.data["graphics"]["ink_color"]
        registry["entry"]["readability"] = "none"
        registry["wide_paper"] = style.data["graphics"]["paper_color"]
    windows = tuple({**window, "x": int(window["x"] + 92 - cx),
                     "y": int(window["y"] + 72 - cy)} for window in plan.windows)
    return replace(plan, x=0, y=0, width=paper_width + 80, height=paper_height + 88,
                   variables=variables, registry=registry, windows=windows,
                   shape="horizontal", motion=None)


def _content_plan(plan: cards.CardPlan, slots: Mapping[str, object],
                  style: ResolvedStyle) -> tuple[cards.CardPlan, tuple]:
    """Fit the existing content at wide type sizes; keep its timing and media windows."""
    v, kind = dict(plan.variables), plan.template
    font = style.data["graphics"]["font"]
    room = 556
    css = ""

    def fit(key, text, width=room, maximum=42, *, weight=600):
        size = cards._fit_measured(key, text, width, font, weight, maximum)
        v[key + "_px"] = size
        return size

    def lines(text, width=room, maximum=42):
        try:
            rows, size = _lines(text, font, width, maximum)
        except ValidationError:
            size = cards._fit_measured("text", text, width, font, 600, maximum)
            rows = [text]
        return "\n".join(rows), size, len(rows)

    if kind == "steps":
        labels = list(slots["steps"])
        fitted = [lines(label, room - 60) for label in labels]
        size = min(row[1] for row in fitted)
        fitted = [lines(label, room - 60, size) for label in labels]
        v.update(steps=json.dumps([row[0] for row in fitted]), text_px=size,
                 wide_steps_layout="vertical")
        padding, gap = (16, 26) if len(labels) >= 4 else (26, 44)
        content = (sum(row[2] * size * 1.2 + 2 * padding for row in fitted)
                   + (len(labels) - 1) * gap + 56.8)
        css = """.wide-kicker{font-family:var(--card-mono);font-size:24px;line-height:1.2;
          letter-spacing:.12em;text-transform:uppercase;color:var(--card-accent);margin-bottom:28px}
        .rail{flex-direction:column;align-items:stretch;gap:0}
        .chip{padding:26px 30px;border:0;border-radius:14px;background:var(--card-titlebar);
          color:var(--card-ink);
          font-family:var(--card-font);font-weight:550;letter-spacing:0;text-transform:none;
          white-space:pre-line;flex-shrink:0}
        .label{width:auto!important;white-space:pre-line}
        .chip+.chip{margin-top:44px}
        .chip+.chip::after{content:'↓';position:absolute;left:0;right:0;top:-44px;height:44px;
          display:grid;place-items:center;color:var(--card-accent);font-size:34px}
        .chip[data-state=done]{color:var(--card-ink)}
        .chip[data-state=done]::before{display:none}"""
        if len(labels) >= 4:
            css += (f".chip{{padding:{padding}px 30px}}.chip+.chip{{margin-top:{gap}px}}"
                    f".chip+.chip::after{{top:-{gap}px;height:{gap}px;font-size:24px}}")
    elif kind == "checklist":
        labels = json.loads(v["items"])
        fitted = [lines(label, room - 71) for label in labels]
        size = min(row[1] for row in fitted)
        fitted = [lines(label, room - 71, size) for label in labels]
        v.update(items=json.dumps([row[0] for row in fitted]), text_px=size)
        content = sum(row[2] * size * 1.3 + 7 for row in fitted) + (len(labels) - 1) * 24
        css = ".rows{gap:24px}.label{white-space:pre-line}"
    elif kind == "quote":
        v["quote"], v["quote_px"], count = lines(v["quote"], room, 56)
        fit("name", v["name"], weight=700)
        fit("role", v["role"], weight=400)
        content = (count * v["quote_px"] * 1.13 + 28 + v["name_px"] * 1.06
                   + (v["role_px"] * 1.06 if v["role"] else 0))
        css = """.mark{display:none}.quote-panel{min-height:0;justify-content:flex-start}
        blockquote{margin:0 0 28px;max-width:100%;white-space:pre-line}
        .attribution{margin:0}.name,.role{white-space:pre-line}"""
    elif kind == "stat":
        v["kicker_px"] = 24
        content = v["value_px"] + (24 * 1.2 + 28 if v["kicker"] else 0)
        rows = json.loads(v.get("rows", "[]"))
        if rows:
            fit("row", max(rows, key=len))
            count = len(rows)
            content += 12 + count * v["row_px"] * 1.2 + (count - 1) * 4
        css = """.kicker{font-family:var(--card-mono);margin-bottom:28px;color:var(--card-accent)}
        .rows{line-height:1.2!important;gap:4px!important}"""
    elif kind == "chart":
        bars = json.loads(v["bars"])
        fit("label", max((bar["label"] for bar in bars), key=len), room * .4)
        fit("value", max((str(bar["value"]) + v["unit"] for bar in bars), key=len),
            room * .25, weight=700)
        label_width = math.ceil(max(text_width_em(b["label"], font, 600)
                                    for b in bars) * v["label_px"]) + 2
        value_width = math.ceil(max(text_width_em(str(b["value"]) + v["unit"], font, 700)
                                    for b in bars) * v["value_px"]) + 2
        track_width = room - label_width - value_width - 32
        if track_width < 100:
            raise ValidationError("slots.bars", "Shorten labels to leave room for the bars.")
        peak = max(b["value"] for b in bars)
        v.update(label_width_px=label_width, value_width_px=value_width,
                 track_width_px=track_width, row_height_px=math.ceil(max(
                     v["label_px"], v["value_px"]) * 1.1),
                 bars=json.dumps([{**b, "length_px": round(track_width * b["value"] / peak)
                                  if peak else 0} for b in bars]))
        fit("title", v["title"], weight=700)
        content = v["title_px"] * 1.15 + 28 + len(bars) * v["row_height_px"] + (len(bars)-1)*24
        css = ".heading{margin-bottom:28px}.bars{gap:24px}"
    elif kind == "specs_list":
        rows = json.loads(v["rows"])
        fitted = [lines(r[k], (room-20)/2) for r in rows for k in ("name", "value")]
        v["row_px"] = min(row[1] for row in fitted)
        fitted = [{key: lines(row[key], (room-20)/2, v["row_px"])[0]
                   for key in ("name", "value")} for row in rows]
        v["rows"] = json.dumps(fitted)
        fit("title", v["title"], weight=700)
        row_height = sum(max(len(text.split("\n")) for text in row.values())
                         * v["row_px"] + 2 for row in fitted)
        content = v["title_px"] * 1.1 + 28 + row_height + (len(rows)-1)*24
        css = """.heading{margin-bottom:28px}.rows{gap:24px!important}
        .row{line-height:1!important;border-top:2px solid
          color-mix(in srgb,var(--card-ink) 20%,transparent)!important}
        .row-name,.row-value{white-space:pre-line}"""
    elif kind == "name_tag":
        fit("name", v["name"], room, 56, weight=700)
        fit("role", v["role"], weight=400)
        content = v["name_px"] * 1.1 + (4 + v["role_px"] * 1.2 if v["role"] else 0)
        css = (".tag-panel{min-height:0}.tab{display:none}.identity{padding:0}"
               ".role:empty{display:none}")
    elif kind == "progress_stat":
        fit("label", v["label"])
        content = v["label_px"] * 1.2 + v["value_px"] * 1.05 + 23
        css = ".progress-panel{min-height:0}.progress-label{line-height:1.2}"
    elif kind == "success_check":
        v["label"], v["label_px"], count = lines(v["label"], room - 110, 52)
        content = max(88, count * v["label_px"] * 1.2)
        css = ".success-panel{min-height:0}.success-label{white-space:pre-line;line-height:1.2}"
    elif kind == "icon_strike":
        fit("label", v["label"], room - 188, 56, weight=700)
        content = 160
        css = ".panel{width:100%!important}"
    elif kind == "terminal":
        longest = max([v["command"], *json.loads(v["output"])], key=len)
        v["text_px"] = cards._fit_measured("command", longest, room-90, cards.MONO_FONT, 400, 42)
        v["bar_px"] = 24
        content = 60 + 40 + (1 + len(json.loads(v["output"]))) * v["text_px"] * 1.35
        css = ".terminal-body{padding:20px 0}.bar{padding-left:0;padding-right:0}"
    elif kind == "text_tag":
        if v["look"] in {"block", "rule", "underline"}:
            v["text"] = v["text"].upper()
        v["text"], v["text_px"], count = lines(v["text"], room, 64)
        sub_font = (cards.MONO_FONT if v["look"] in {"block", "underline"}
                    and not style.data.get("brand", {}).get("cards_font") else font)
        v["subtext_px"] = cards._fit_measured("subtext", v["subtext"], room, sub_font, 600, 42)
        content = count * v["text_px"] * 1.06 + (28 + v["subtext_px"] * 1.2 if v["subtext"] else 0)
        css = """.tag-wrapper,.tag-body{width:100%!important;max-width:100%!important}
        .tag-body{padding:0!important;gap:28px!important}.tab,.rule{display:none!important}
        .panel{padding:0!important;gap:0!important}.tag-text{white-space:pre-line;line-height:1.06!important}
        .subtext{white-space:pre-line;letter-spacing:0!important;text-transform:none!important}
        [data-look=underline] .panel,[data-look=underline] .subtext{
          color:var(--card-ink)!important;text-shadow:none!important}"""
        if v["look"] == "block":
            v["paper_color"] = v["accent_color"]
    elif kind in {"prompt_bar", "comment_cta"}:
        content = 76 if kind == "prompt_bar" else 96
        if kind == "comment_cta" and v["label"]:
            fit("label", v["label"])
            content += 16 + v["label_px"] * 1.2
        css = ".panel{height:auto;border-radius:0}.line{line-height:1.2}"
    elif kind == "comment_drop":
        text, v["text_px"], count = lines(v["text"], room - 92)
        v["lines"] = json.dumps(text.split("\n"))
        name_px = cards._fit_measured("name", v["name"], room-92, font, 700, 42)
        content = max(72, count * v["text_px"] * 1.2 + (name_px*1.2 if v["name"] else 0))
        css = """.bubble{width:100%;padding:0;background:transparent;box-shadow:none}
        .bubble::before{display:none}.avatar{font-size:42px}.panel{padding:0}"""
        css += f".name{{font-size:{name_px}px}}"
    elif kind == "flow":
        # The boxes' paths and their reveal clock stay intact in a wide vertical flow.
        labels = json.loads(v["nodes"])
        size = min(cards._fit_measured("nodes", label, room-36, font, 600, 42) for label in labels)
        v.update(text_px=size, box_width_px=room, box_height_px=88, arrow_gap_px=44)
        if "tag_height_px" in v:
            v.update(tag_height_px=math.ceil(size * 1.46), tag_max_width_px=room)
        content = len(labels) * v.get("tag_height_px", 88) + (len(labels)-1)*44
        css = """.row{flex-direction:column}.node{flex:0 0 88px;width:100%}
        .connector{transform:rotate(90deg);flex:0 0 44px;width:44px}.label{white-space:pre-line}"""
    elif kind == "choices":
        v["text_color"] = style.data["graphics"]["ink_color"]
        labels = [(f"{chr(65+i)}) " if v["letters"] else "") + label
                  for i, label in enumerate(json.loads(v["options"]))]
        fitted = [lines(label, room/v["scale"]) for label in labels]
        size = min(row[1] for row in fitted)
        fitted = [lines(label, room/v["scale"], size) for label in labels]
        heights = [row[2] * size * cards.CHOICES_LINE_HEIGHT for row in fitted]
        overshoot = (v["scale"] - 1) / 2
        gaps = [math.ceil(overshoot * max(a, b) + cards.CHOICES_ROW_GAP_PX)
                for a, b in zip(heights, heights[1:], strict=False)]
        top = math.ceil(v["glow_pad_px"] + overshoot * heights[0])
        content = math.ceil((sum(heights) + sum(gaps))/2)*2
        height = math.ceil((top + content + overshoot*heights[-1] + v["glow_pad_px"])/2)*2
        v.update(text_px=size, lines=json.dumps([row[0].split("\n") for row in fitted]),
                 row_heights=json.dumps(heights), row_gaps=json.dumps(gaps), stack_top_px=top)
        return replace(plan, width=room+80, height=height, variables=v), (40, top, room, content)
    else:
        # Media and authored diagrams already expose their measured panel dimensions.
        if plan.registry:
            content = 280 if kind.endswith("progress-ring") else 200
            plan = replace(plan, height=content+88, variables=v)
            if kind.endswith("char-slam-explode"):
                return plan, (82, 92, room, 64)
            return plan, (40, 24, plan.width-80, content)
        if kind == "file_drop":
            from yap_studio.file_cards import _motion

            old_height = v["panel_height_px"]
            text_room = room - v["text_left_px"] - 24
            fit("name", v["name"], text_room)
            v.update(duration_px=42, badge_px=42, text_width_px=text_room,
                     file_panel_width_px=room)
            text_height = v["name_px"] * 1.2 + 12 + 50.4 + (72.4 if v["badge"] else 0)
            v["panel_height_px"] = math.ceil((max(v["poster_height_px"], text_height) + 48)/2)*2
            motion = _motion(v, drop=v["drop_at"], frames=plan.frames, fps=plan.fps)
            for frame in motion:
                frame["duration_line_px"] *= 1.4
                frame["badge_line_px"] *= 1.4
            v["motion"] = json.dumps(motion)
            play = json.loads(v.get("play_window", "{}"))
            if play:
                delta = (v["panel_height_px"] - old_height) / 2
                play["y"] += delta
                play["panel_top_px"] -= delta
                v["play_window"] = json.dumps(play)
            pw, ph = v["file_panel_width_px"], v["panel_height_px"]
            v["wide_css"] = "#root{align-items:center!important}"
            return replace(plan, variables=v), ((plan.width-pw)/2,
                24 + (v["panel_max_height_px"]-ph)/2, pw, ph)
        if kind == "jiggle_delete":
            content = v["panel_height_px"]
        elif kind == "storyboard":
            content = v["thumb_height_px"] + 8 + 50.4
            v.update(number_px=42, thumb_top_px=0)
            css = f".number{{line-height:50.4px}}.panel{{height:{content}px!important}}"
        elif kind == "edit_timeline":
            label_width = math.ceil(v["label_width_px"] * 42 / 30)
            lane_width = room - 48 - label_width - 20
            ratio = lane_width / v["lane_width_px"]
            tracks = json.loads(v["tracks"])
            for track in tracks:
                for clip in track["clips"]:
                    clip["x_px"] *= ratio
                    clip["width_px"] *= ratio
            v.update(text_px=42, timecode_px=42, label_width_px=label_width,
                     timecode_width_px=math.ceil(v["timecode_width_px"] * 42 / 40),
                     lane_left_px=24+label_width+20, lane_width_px=lane_width,
                     tracks=json.dumps(tracks),
                     ticks=json.dumps([{**tick, "x_px": tick["x_px"]*ratio}
                                       for tick in json.loads(v["ticks"])]),
                     playhead=json.dumps({key: value*ratio for key, value in
                                           json.loads(v["playhead"]).items()}))
            fit("title", v["title"], room - 72 - v["timecode_width_px"])
            content = v["panel_height_px"]
        elif kind == "video_card":
            fit("label", v["label"])
            content = v["panel_max_height_px"]
        elif kind == "callout_circle":
            fit("label", v["label"], v["image_width_px"] - 50, weight=700)
            content = v["image_height_px"]
        elif kind == "image_card":
            info = cards.ffmpeg.probe(plan.assets[0][1])
            scale = min(v["image_max_width_px"] / info.width,
                        v["image_max_height_px"] / info.height, 1)
            content = info.height * scale + 60
        elif kind == "image_compare":
            info = cards.ffmpeg.probe(plan.assets[0][1])
            scale = min(v["image_max_width_px"] / info.width,
                        v["image_max_height_px"] / info.height, 1)
            frame_width = info.width * scale
            chrome = 2 * (cards.COMPARE_CHIP_PAD + cards.COMPARE_CHIP_BORDER)
            reserve = (
                cards.COMPARE_CHIP_INSET + cards.COMPARE_DIVIDER_KNOB / 2
                + cards.COMPARE_CHIP_GAP + chrome)
            v["chip_px"] = min((cards._fit_measured("label", v[side + "_label"],
                frame_width*share/100 - reserve, font, 600, 42)
                for side, share in (("before", 100-v["split"]), ("after", v["split"]))
                if v[side + "_label"]), default=42)
            for side in ("before", "after"):
                v[side + "_chip_max_width_px"] = (math.ceil(text_width_em(
                    v[side + "_label"], font, 600) * v["chip_px"]) + chrome
                    if v[side + "_label"] else 0)
            if v["after_label"]:
                share = ((cards.COMPARE_CHIP_INSET + v["after_chip_max_width_px"])
                         / (frame_width * v["split"]/100))
                v["after_chip_reveal_progress"] = math.acos(1 - 2*share) / math.pi
            content = info.height * scale + 16
        else:  # transfer: the route and devices occupy the authored diagram box.
            text_room = (v["chip_max_width_px"] - v["chip_icon_px"] - v["chip_gap_px"]
                         - 2 * v["chip_pad_x_px"])
            try:
                chip_size = cards._fit_measured("payload", v["payload"], text_room, font, 600, 42)
            except ValidationError as error:
                error.suggested_action = "Shorten the transfer payload."
                raise
            v.update(label_px=min(cards._fit_measured("label", v[key],
                     v["device_column_width_px"], font, 600, 42)
                     for key in ("left_label", "right_label")),
                     chip_px=chip_size)
            content = v["panel_max_height_px"]
        v["wide_css"] = css
        return replace(plan, variables=v), (40, 24, plan.width-80, content)
    content = math.ceil(round(content, 6) / 2) * 2
    v.update(panel_width_px=room, panel_max_height_px=content,
             pad_x_px=0, pad_top_px=0, pad_bottom_px=0, wide_css=css,
             wide_natural=True)
    return replace(plan, width=room, height=content, variables=v), (0, 0, room, content)


def place_card(plan: cards.CardPlan, style: ResolvedStyle,
               faces: Sequence[Sequence[Box]] | None, *, caption_top: int | None = None,
               saved_box: Box | None = None) -> tuple[cards.CardPlan, dict]:
    frame = canvas("horizontal")
    ceiling = min(frame.height - 108, min(caption_top if caption_top is not None
                                         else frame.caption_top, frame.caption_top)
                  - style.data["graphics"]["caption_gap_px"])
    if plan.height > ceiling - 108 or plan.width > frame.width - 226:
        fix = "Shorten the card text or use fewer rows."
        if plan.template == "steps":
            labels = [label.replace("\n", " ") for label in json.loads(plan.variables["steps"])]
            limit = 0
            for count in range(len(labels) - 1, 1, -1):
                shorter = card_plan("steps", {"steps": labels[:count]}, style,
                                    frames=plan.frames, fps=plan.fps)
                if shorter.height <= ceiling - 108:
                    limit = count
                    break
            fix = (f"Use {limit} steps or fewer." if limit >= 2
                   else "Shorten the step labels or reduce the caption size.")
        error = ValidationError("graphic.clearance", fix)
        error.suggested_action = fix
        raise error
    heads = [head_box(face) for row in faces or () for face in row]
    margin = face_clearance_px(style.data)
    right_first = not heads or sum(h.x + h.w / 2 for h in heads) / len(heads) < frame.width / 2
    xs = [frame.width - 130 - plan.width, 130]
    if not right_first:
        xs.reverse()
    y = min(230 - cards.PAD_TOP, ceiling - plan.height)
    candidates = [Box(x, y, plan.width, plan.height) for x in xs]
    if saved_box is not None:
        candidates = [saved_box]
    box = next((box for box in candidates if 96 <= box.x and box.x1 <= frame.width - 96
                and 108 <= box.y and box.y1 <= ceiling
                and box.w == plan.width and box.h == plan.height
                and all(gap_px(box, head) >= margin for head in heads)), None)
    if faces is None or box is None:
        error = ValidationError("graphic.clearance",
                                "Please choose a moment with room beside the speaker.")
        error.suggested_action = "Choose a moment with room beside the speaker."
        raise error
    side = "right" if box.x + box.w / 2 > frame.width / 2 else "left"
    evidence = {"status": "clear", "layout": "floating",
                "reason": "floating card clears every measured face",
                "margin_px": margin,
                "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h}}
    motion = None
    if style.data["graphics"].get("card_motion", "none") != "none":
        sweep = (Box(box.x, box.y, frame.width - box.x, box.h) if side == "right"
                 else Box(0, box.y, box.x1, box.h))
        if not plan.windows and all(gap_px(sweep, head) >= margin for head in heads):
            motion = {"side": side, "enter": True}
            evidence["card_motion"] = {"status": "streak", **motion,
                "rest_box": {"x": box.x, "y": box.y, "width": box.w, "height": box.h},
                "sweep": {"x": sweep.x, "y": sweep.y, "w": sweep.w, "h": sweep.h}}
        else:
            evidence["card_motion"] = {"status": "none", "reason":
                "video windows stay fixed" if plan.windows else "the motion crosses a face"}
    return replace(plan, x=box.x, y=box.y, motion=motion), evidence


def prepare(plan: cards.CardPlan, work_dir: Path, *,
            install: hyperframes.RendererInstall | None = None) -> Path:
    """Mount a locally fitted card on a transparent delivery canvas; only the card moves."""
    renderer = install or hyperframes.ensure_installed()
    v = plan.variables
    native = replace(plan, shape="vertical", motion=None,
                     width=v["wide_native_width_px"], height=v["wide_native_height_px"])
    inner = cards.prepare_card(native,
                               work_dir / "inner", install=renderer)
    card_source = (inner / "index.html").read_text()
    if not plan.registry:
        css = """#root{align-items:flex-start!important}
          .panel,.shadow-backing{background:transparent!important;box-shadow:none!important;
          max-height:none!important;border-radius:0!important}
          .glow .row,.glow .row-name,.glow .row-value,.glow .kicker{
            color:var(--card-ink)!important;text-shadow:none!important}
          .glow .value-stage,.glow .affix,.glow .heading{color:var(--card-accent)!important}
          .kicker{font-family:var(--card-mono)}"""
        if v.get("wide_natural"):
            css += """#root{padding:0!important;align-items:flex-start!important}
              .panel{padding:0!important;width:100%!important;min-height:0!important;
              flex-shrink:0;overflow:visible!important}"""
        css += v.get("wide_css", "")
        card_source = card_source.replace("</head>", f"<style>{css}</style></head>")
        card_source = card_source.replace("Number(root.dataset.duration)",
            "Number(root.dataset.duration || root.dataset.hfAuthoredDuration)")
        card_source = card_source.replace("root.insertBefore(frame, panel);",
                                           "panel.parentNode.insertBefore(frame, panel);")
        card_source = card_source.replace('panel.style.height = complete ? "" : `${height}px`;',
            'panel.style.height = complete ? "" : `${height}px`;'
            'globalThis.document.getElementById("wide-paper").style.height = `${height+100}px`;')
        if plan.template == "flow":
            card_source = card_source.replace("panel.remove();",
                "const panelParent = panel.parentNode; panel.remove();").replace(
                "root.appendChild(stack);", "panelParent.appendChild(stack);")
        if plan.template == "steps":
            card_source = card_source.replace('<div class="rail"',
                '<div class="wide-kicker">How it works</div><div class="rail"')
            card_source = card_source.replace("label.textContent = step.toUpperCase();",
                                               "label.textContent = step;")
    elif plan.template.endswith("progress-ring"):
        card_source = card_source.replace("</head>", '<style>#hf-host .ring-stage{'
            'width:min(100cqw,100cqh)}#hf-host .ring-label{font-size:84px}</style></head>')
    composition = paths.validate_output_dir(work_dir / "wide-card")
    shutil.copytree(inner, composition, dirs_exist_ok=True,
                    ignore=lambda folder, names: ["index.html"] if Path(folder) == inner else [])
    paths.atomic_write_text(composition / "card.html", card_source)
    no_timeline = " data-no-timeline" if " data-no-timeline" in card_source else ""
    shutil.copyfile(hyperframes.gsap_file(renderer), composition / "gsap.min.js")
    duration = hyperframes.layer_duration_s(plan.frames, plan.fps)
    values = html.escape(json.dumps(plan.variables, ensure_ascii=False).replace("/", r"\/"),
                         quote=True)
    frame = canvas("horizontal")
    paper = plan.variables.get("paper_color")
    if plan.registry:
        paper = plan.registry["wide_paper"]
    shell = f"background:{paper};border-radius:24px;box-shadow:0 24px 60px rgb(0 0 0 / .28);"
    offset = (frame.width - plan.x + 40 if plan.motion and plan.motion["side"] == "right"
              else -plan.x - plan.width - 40)
    motion = (f'tl.fromTo("#floating",{{x:{offset}}},{{x:0,duration:.4,ease:"expo.out"}},0);'
              f'tl.to("#floating",{{x:{offset},duration:.35,ease:"expo.in"}},D-.35);'
              if plan.motion else "")
    inner_id = "hf-host" if plan.registry else plan.template
    paper_panel = (f'<div id="wide-paper" style="position:absolute;left:40px;top:24px;'
                   f'width:{v["wide_paper_width_px"]}px;height:{v["wide_paper_height_px"]}px;'
                   f'{shell}"></div>')
    if plan.windows:
        holes = "".join(f'<rect x="{w["x"]-40}" y="{w["y"]-24}" width="{w["width"]}" '
                        f'height="{w["height"]}" rx="{w["radius"]}" fill="black"/>'
                        for w in plan.windows)
        paper_panel = (f'<svg id="wide-paper" width="{plan.width-80}" height="{plan.height-88}" '
                       'style="position:absolute;left:40px;top:24px;overflow:visible">'
                       '<defs><mask id="paper-mask" maskUnits="userSpaceOnUse" x="-40" '
                       f'y="-24" width="{plan.width}" height="{plan.height}">'
                       f'<rect x="-40" y="-24" width="{plan.width}" height="{plan.height}" '
                       f'fill="white"/>{holes}</mask></defs><g mask="url(#paper-mask)">'
                       f'<rect width="100%" height="100%" rx="24" fill="{paper}" '
                       'style="filter:drop-shadow(0 24px 30px rgb(0 0 0 / .28))"/></g></svg>')
    left, top = 92 - v["wide_content_x_px"], 72 - v["wide_content_y_px"]
    paper_motion = ""
    if plan.template == "jiggle_delete":
        paper_motion = (f'const paperFrames={v["panel_keyframes"]};'
                        'paperFrames.forEach(frame=>tl.set("#wide-paper",'
                        '{height:frame.height_px+100},frame.at_s));')
    head = ""
    mounted = (
        f'<div data-composition-id="{html.escape(inner_id, quote=True)}"{no_timeline} '
        'data-composition-src="card.html" data-hf-authored-id="root" data-start="0" '
        f'data-duration="{duration}" data-width="{native.width}" data-height="{native.height}" '
        f'data-track-index="0" data-variable-values="{values}"></div>'
    )
    if plan.registry:
        # Mount the registry item directly so the compiler preserves its variable bindings.
        head = card_source.partition("<head>")[2].partition("</head>")[0]
        mounted = card_source.partition("<body>")[2].rpartition("</body>")[0]
    source = (
        '<!doctype html><html><head><meta charset="utf-8"><script src="gsap.min.js"></script>'
        + head +
        f'<style>html,body{{margin:0;width:{frame.width}px;height:{frame.height}px;'
        'background:transparent;overflow:hidden}</style></head><body>'
        f'<div id="wide" data-composition-id="wide-card" data-width="{frame.width}" '
        f'data-height="{frame.height}" data-start="0" data-duration="{duration}" '
        f'style="clip-path:inset({plan.y}px 96px '
        f'{frame.height - plan.y - plan.height}px 96px)">'
        f'<div id="floating" style="position:absolute;left:{plan.x}px;top:{plan.y}px;'
        f'width:{plan.width}px;height:{plan.height}px;overflow:hidden">'
        f'{paper_panel}'
        f'<div id="wide-content" style="position:absolute;left:{left}px;top:{top}px">'
        f'{mounted}</div></div></div>'
        f'<script>(()=>{{const D={duration};const tl=gsap.timeline({{paused:true}});'
        'tl.fromTo("#floating",{opacity:0},{opacity:1,duration:.25},0);'
        'tl.to("#floating",{opacity:0,duration:.3},D-.3);' + motion + paper_motion +
        'window.__timelines=window.__timelines||{};window.__timelines["wide-card"]=tl;'
        '})();</script></body></html>'
    )
    paths.atomic_write_text(composition / "index.html", source)
    return composition
