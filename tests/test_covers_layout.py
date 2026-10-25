"""Cover text and pixel checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yap_studio.covers import pick, render
from yap_studio.covers.layout import (
    balance_lines,
    check_contrast,
    check_geometry,
    parse_accent,
    plan_blocks,
    readable_accent,
    validate_text,
)
from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import head_box
from yap_studio.faces.model import Box
from yap_studio.style import resolve


@pytest.mark.parametrize("backdrop", ["#101010", "#000000", "#FFFFFF"])
def test_readable_accent_tints_first_candidate(backdrop: str) -> None:
    from yap_studio.cards import _contrast

    style = resolve().data
    original = "#DDDDDD" if backdrop == "#FFFFFF" else "#365A58"
    style["brand"] = {"accent_color": original}
    for section, key in (("captions", "highlight_color"), ("hook", "accent_color"),
                         ("graphics", "accent_color")):
        style[section][key] = original
    accent = readable_accent(style, backdrop)
    assert accent == ("#6E6E6E" if backdrop == "#FFFFFF" else "#688382")
    assert _contrast(accent, backdrop) >= 3
    assert style["brand"]["accent_color"] == original
    assert readable_accent(style, backdrop) == accent


def test_readable_accent_keeps_first_readable_candidate_exactly() -> None:
    style = resolve().data
    style["brand"] = {"accent_color": "#F34e3f"}
    assert readable_accent(style, "#101010") == "#F34e3f"
    style["brand"]["accent_color"] = "#101010"
    style["captions"]["highlight_color"] = "#F34e3f"
    assert readable_accent(style, "#101010") == "#F34e3f"


def test_vertical_number_keeps_existing_ink_fallback() -> None:
    style = resolve().data
    for section, key in (("captions", "highlight_color"), ("hook", "accent_color"),
                         ("graphics", "accent_color")):
        style[section][key] = style["graphics"]["paper_color"]
    plan = plan_blocks("number", {"number": "5", "label": "better nights"}, style,
                       face_box=(450, 350, 500, 700))
    assert plan.blocks[0]["color"] == style["graphics"]["ink_color"]


def test_readable_accent_missing_colors_has_actionable_fix() -> None:
    style = {"captions": {}, "hook": {}, "graphics": {}}
    with pytest.raises(ValidationError) as error:
        readable_accent(style, "#101010")
    assert error.value.suggested_action == (
        "Remove the *accent* marks or set a brand accent colour on the style page.")


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize(("look", "values"), [
    ("photo", {"headline": "*Fuck* this", "kicker": "Shit"}),
    ("hook", {"headline": "Fuck this", "label": "Shit"}),
    ("number", {"number": "5", "label": "Fuck"}),
    ("before-after", {"before": "Fuck", "after": "Shit"}),
])
def test_bleep_masks_cover_copy_before_layout(look, values, on: bool) -> None:
    from yap_studio.sfx import bleep

    style = resolve("preset:editorial_serif")
    style.data["music"]["bleep_swears"] = on
    plan = plan_blocks(look, values, style.data, face_box=(450, 700, 300, 350))
    texts = [block["text"] for block in plan.blocks]
    assert all(bleep.mask_text(text, bleep.listed({})) == text for text in texts) is on
    if on and look == "photo":
        headline = next(block for block in plan.blocks if block["role"] == "headline")
        assert headline["text"][headline["accent_start"]:headline["accent_end"]] == "F***"


def test_bleep_checks_the_actual_cover_blocks(monkeypatch) -> None:
    style = resolve("preset:editorial_serif")
    plan = plan_blocks("photo", {"headline": "Fuck this"}, style.data)
    style.data["music"]["bleep_swears"] = True
    monkeypatch.setattr(render, "plan_blocks", lambda *args, **kwargs: plan)
    with pytest.raises(ValidationError, match="Bleep check failed"):
        render._config(render.CoverAssets(Path("unused.png")), style, "photo", {})


def test_balance_lines_avoids_orphans_and_spreads_width() -> None:
    lines = balance_lines("one two three four five six".split(), len, 14, 3)
    assert lines == ["one two", "three four", "five six"]
    assert balance_lines(["single"], len, 10, 2) == ["single"]
    with pytest.raises(ValidationError, match="shorten"):
        balance_lines(["unbreakableword"], len, 5, 2)


@pytest.mark.parametrize(("look", "values"), [
    ("photo", {}), ("number", {"number": "5"}),
    ("hook", {"headline": "A hook"}), ("before-after", {"before": "$1"}),
])
def test_required_flags(look: str, values: dict) -> None:
    with pytest.raises(ValidationError, match="required"):
        validate_text(look, values)


def test_unused_flags_and_caps() -> None:
    with pytest.raises(ValidationError, match="not used"):
        validate_text("number", {"number": "5", "label": "wins", "inset": "x.png"})
    with pytest.raises(ValidationError, match="shorten to at most 60"):
        validate_text("photo", {"headline": "x" * 61})
    with pytest.raises(ValidationError, match="shorten to at most"):
        validate_text("photo", {"headline": "W" * 55}, font="Instrument Serif")


def test_accent_markup() -> None:
    assert parse_accent("Stop *at night* now") == ("Stop at night now", "at night")
    assert parse_accent("Screen time is *mine*, not theirs") == (
        "Screen time is mine, not theirs", "mine")
    assert parse_accent('She said "*mine*" first') == ('She said "mine" first', "mine")
    assert parse_accent("Try (*mine*) today") == ("Try (mine) today", "mine")
    for text in ("a*b* c", "*one* and *two*", "one *bad", "one **bad**"):
        with pytest.raises(ValidationError, match="accent"):
            parse_accent(text)


@pytest.mark.parametrize("headline", ("go *go* now", "Go *go* now"))
def test_repeated_headline_accents_marked_word(headline: str) -> None:
    plan = plan_blocks("photo", {"headline": headline},
                       resolve("preset:editorial_serif").data)
    lines = [block for block in plan.blocks if block["role"] == "headline"]
    marked = " ".join(block["text"][block["accent_start"]:block["accent_end"]]
                      for block in lines if block["accent_start"] >= 0)
    assert marked == "go"
    assert sum(block["accent_start"] >= 0 for block in lines) == 1
    accent_offset = next(sum(len(prior["text"]) + 1 for prior in lines[:index]) +
                         block["accent_start"] for index, block in enumerate(lines)
                         if block["accent_start"] >= 0)
    assert accent_offset == 3


def test_headline_accent_crosses_line_break() -> None:
    plan = plan_blocks("photo", {"headline": "Start *a better night* now"},
                       resolve("preset:editorial_serif").data)
    lines = [block for block in plan.blocks if block["role"] == "headline"]
    marked = [block["text"][block["accent_start"]:block["accent_end"]]
              for block in lines if block["accent_start"] >= 0]
    assert len(marked) == 2
    assert " ".join(marked) == "a better night"


def test_head_box_from_synthetic_matte() -> None:
    from yap_studio.covers.render import head_box_from_matte

    gray = np.zeros((1920, 1080), dtype=np.uint8)
    yy, xx = np.ogrid[:1920, :1080]
    gray[((xx - 610) / 140) ** 2 + ((yy - 520) / 170) ** 2 <= 1] = 255
    gray[680:1700, 420:800] = 255
    gray[15, 100] = 255
    box = head_box_from_matte(gray)
    assert box is not None
    x, y, width, height = box
    assert 465 <= x <= 490 and 350 <= y <= 370
    assert 240 <= width <= 290 and 300 <= height <= 380
    assert head_box_from_matte(np.zeros_like(gray)) is None


def test_head_box_accepts_close_up_matte() -> None:
    gray = np.zeros((1920, 1080), dtype=np.uint8)
    yy, xx = np.ogrid[:1920, :1080]
    gray[((xx - 540) / 480) ** 2 + ((yy - 530) / 230) ** 2 <= 1] = 255
    gray[760:1920, :] = 255
    box = render.head_box_from_matte(gray)
    assert box is not None
    x, y, width, height = box
    assert 760 <= width <= 920
    assert 290 <= y <= 315 and 80 <= x <= 160
    assert height == round(width * 1.3)


def _raised_hand_matte() -> np.ndarray:
    gray = np.zeros((1920, 1080), dtype=np.uint8)
    yy, xx = np.ogrid[:1920, :1080]
    gray[100:300, 210:310] = 255
    gray[300:850, 240:295] = 255
    gray[((xx - 610) / 140) ** 2 + ((yy - 520) / 170) ** 2 <= 1] = 255
    gray[680:1700, 420:800] = 255
    return gray


def test_head_box_rejects_raised_hand() -> None:
    assert render.head_box_from_matte(_raised_hand_matte()) is None


def test_head_box_rejects_separate_arm_beside_head() -> None:
    gray = _raised_hand_matte()
    gray[100:330, 210:310] = 0
    gray[100:330, 240:295] = 0
    gray[330:580, 240:295] = 255
    assert render.head_box_from_matte(gray) is None


def test_prepared_raised_hand_matte_refuses_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gray = _raised_hand_matte()

    def extract(_cut: Path, _frame: int, output: Path) -> Path:
        output.write_bytes(b"synthetic frame")
        return output

    def matte(_frame: Path, work: Path) -> Path:
        output = work / "matte.png"
        output.write_bytes(b"synthetic matte")
        return output

    def ffmpeg_run(args: list, **_kwargs):
        if "pipe:1" in args:
            return SimpleNamespace(stdout=gray.tobytes())
        Path(args[-1]).write_bytes(b"synthetic cutout")
        return SimpleNamespace(stdout=b"")

    monkeypatch.setattr(pick, "extract_frame", extract)
    monkeypatch.setattr(render.ffmpeg, "run", ffmpeg_run)
    with pytest.raises(ValidationError, match="face cache or person matte"):
        render.prepare_assets(tmp_path / "cut.mp4", 0, tmp_path / "work", "number",
                              matte_provider=matte, face_box=None)


def test_prepared_matte_box_plans_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gray = np.zeros((1920, 1080), dtype=np.uint8)
    yy, xx = np.ogrid[:1920, :1080]
    gray[((xx - 610) / 140) ** 2 + ((yy - 520) / 170) ** 2 <= 1] = 255
    gray[680:1700, 420:800] = 255

    def extract(_cut: Path, _frame: int, output: Path) -> Path:
        output.write_bytes(b"synthetic frame")
        return output

    def matte(_frame: Path, work: Path) -> Path:
        output = work / "matte.png"
        output.write_bytes(b"synthetic matte")
        return output

    def ffmpeg_run(args: list, **_kwargs):
        if "pipe:1" in args:
            return SimpleNamespace(stdout=gray.tobytes())
        Path(args[-1]).write_bytes(b"synthetic cutout")
        return SimpleNamespace(stdout=b"")

    monkeypatch.setattr(pick, "extract_frame", extract)
    monkeypatch.setattr(render.ffmpeg, "run", ffmpeg_run)
    assets = render.prepare_assets(tmp_path / "cut.mp4", 0, tmp_path / "work", "number",
                                   matte_provider=matte, face_box=None)
    assert assets.face_box is not None
    plan = plan_blocks("number", {"number": "5", "label": "NIGHTS"},
                       resolve("preset:editorial_serif").data, face_box=assets.face_box)
    assert plan.speaker_side == "right"


@pytest.mark.parametrize(("look", "inset"), (("number", None),
                                             ("before-after", None),
                                             ("photo", Path("inset.png"))))
def test_missing_face_in_cache_and_matte_explains_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, look: str, inset: Path | None,
) -> None:
    def extract(_cut: Path, _frame: int, output: Path) -> Path:
        output.write_bytes(b"synthetic frame")
        return output

    monkeypatch.setattr(pick, "extract_frame", extract)
    with pytest.raises(ValidationError, match="face cache or person matte"):
        render.prepare_assets(tmp_path / "cut.mp4", 0, tmp_path / "work", look,
                              matte_provider=lambda _frame, _work: None,
                              face_box=None, inset=inset)


def test_geometry_crop_and_top_third() -> None:
    mask = np.zeros((1920, 1080), dtype=np.uint8)
    mask[300:340, 100:200] = 255
    check_geometry(mask, headline_box=(100, 300, 100, 40), look="photo", band=(280, 640))
    mask[260, 100] = 255
    with pytest.raises(ValidationError, match="3:4"):
        check_geometry(mask, headline_box=(100, 300, 100, 40), look="photo", band=(280, 640))
    mask[260, 100] = 0
    with pytest.raises(ValidationError, match="top third"):
        check_geometry(mask, headline_box=(100, 600, 100, 80), look="photo", band=(280, 640))


def test_contrast_arrays() -> None:
    mask = np.ones((4, 4), dtype=bool)
    dark = np.zeros((4, 4, 3), dtype=np.uint8)
    assert check_contrast(mask, dark, "#FFFFFF", minimum=4.5) >= 4.5
    with pytest.raises(ValidationError, match="contrast"):
        check_contrast(mask, np.full_like(dark, 250), "#FFFFFF", minimum=4.5)


def test_speaker_overlap_without_person_at_glyphs() -> None:
    text = np.zeros((4, 8), dtype=np.uint8)
    text[1:3, 1:3] = 255
    person = np.zeros_like(text)
    person[1:3, 5:7] = 255
    blocks = [{"role": "number", "x": 0, "y": 0, "width": 4, "height": 4,
               "opacity": 1}]
    assert render.speaker_overlap(text, person, 0, blocks) == 0


@pytest.mark.parametrize("role", ("number", "label"))
def test_speaker_overlap_counts_shifted_glyph_pixels(role: str) -> None:
    text = np.zeros((4, 8), dtype=np.uint8)
    text[1:3, 2:5] = 255
    person = np.zeros_like(text)
    person[1:3, 0:2] = 255
    blocks = [{"role": role, "x": 1, "y": 1, "width": 4, "height": 2,
               "opacity": 1}]
    assert render.speaker_overlap(text, person, 3, blocks) == 4


@pytest.mark.parametrize("shift", (-8, -3, 3, 8))
def test_speaker_overlap_shift_clears_column(shift: int) -> None:
    text = np.zeros((4, 8), dtype=np.uint8)
    text[1:3, 3:5] = 255
    blocks = [{"role": "number", "x": 2, "y": 0, "width": 4, "height": 4,
               "opacity": 1}]
    assert render.speaker_overlap(text, text, shift, blocks) == 0


@pytest.mark.parametrize(("person_x", "text_x", "shift"), ((7, 0, 1), (0, 7, -1)))
def test_speaker_overlap_drops_pixels_off_canvas(person_x: int, text_x: int,
                                                shift: int) -> None:
    text = np.zeros((4, 8), dtype=np.uint8)
    person = np.zeros_like(text)
    text[:, text_x] = 255
    person[:, person_x] = 255
    blocks = [{"role": "number", "x": 0, "y": 0, "width": 8, "height": 4,
               "opacity": 1}]
    assert render.speaker_overlap(text, person, shift, blocks) == 0


def test_speaker_overlap_uses_block_boxes_and_alpha_thresholds() -> None:
    text = np.array([[183, 184, 91, 92, 31, 32, 255, 255]], dtype=np.uint8)
    person = np.full_like(text, 128)
    person[0, 5] = 127
    blocks = [{"role": "number", "x": 0, "y": 0, "width": 2, "height": 1,
               "opacity": 1},
              {"role": "label", "x": 2, "y": 0, "width": 2, "height": 1,
               "opacity": .5},
              {"role": "label", "x": 4, "y": 0, "width": 2, "height": 1,
               "opacity": .1},
              {"role": "headline", "x": 6, "y": 0, "width": 1, "height": 1,
               "opacity": 1}]
    assert render.speaker_overlap(text, person, 0, blocks) == 2


@pytest.mark.parametrize(("look", "values"), [
    ("photo", {"headline": "A *better* night", "kicker": "REAL PROGRESS"}),
    ("number", {"number": "111111", "label": "better nights"}),
    ("hook", {"headline": "A *better* night", "label": "REAL RESULTS"}),
    ("before-after", {"before": "8 hours", "after": "2 hours", "label": "EACH NIGHT"}),
])
def test_each_look_has_bounded_blocks_at_readable_sizes(look: str, values: dict) -> None:
    style = resolve("preset:editorial_serif").data
    plan = plan_blocks(look, values, style, face_box=(450, 350, 500, 700))
    assert plan.blocks
    for block in plan.blocks:
        assert 60 <= block["x"] < block["x"] + block["width"] <= 1020
        assert 280 <= block["y"] < block["y"] + block["height"] <= 1640
        assert block["role"] and block["font_size"] and block["color"]
        if block["role"] in {"kicker", "label", "proof"}:
            assert 40 <= block["font_size"] <= 48
    if look in {"photo", "hook"}:
        headlines = [block for block in plan.blocks if block["role"] == "headline"]
        assert all(72 <= block["font_size"] <= 150 for block in headlines)
        assert all(block["y"] + block["height"] <= 640 for block in headlines)
    elif look == "number":
        assert 220 <= next(b for b in plan.blocks if b["role"] == "number")["font_size"] <= 440
    else:
        assert 96 <= next(b for b in plan.blocks if b["role"] == "before")["font_size"] <= 140
        assert 160 <= next(b for b in plan.blocks if b["role"] == "after")["font_size"] <= 300


@pytest.mark.parametrize(("face", "speaker_side", "number_x", "inset_side"), [
    ((100, 400, 300, 500), "left", 560, "right"),
    ((650, 400, 300, 500), "right", 60, "left"),
])
def test_speaker_and_inset_use_face_centre(
    face: tuple[int, int, int, int], speaker_side: str, number_x: int, inset_side: str,
) -> None:
    style = resolve("preset:editorial_serif").data
    number = plan_blocks("number", {"number": "5", "label": "NIGHTS"}, style, face_box=face)
    assert number.speaker_side == speaker_side
    assert next(b for b in number.blocks if b["role"] == "number")["x"] == number_x
    centre = face[0] + face[2] / 2 + number.speaker_shift_px
    assert abs(centre - (300 if speaker_side == "left" else 780)) <= 1
    photo = plan_blocks("photo", {"headline": "A better night", "inset": "x.png"},
                        style, face_box=face)
    assert photo.inset_side == inset_side
    x, y, width, height = photo.inset_box
    assert 60 <= x and x + width <= 1020
    assert 280 <= y and y + height <= 1640
    assert photo.inset_scale == 1


def test_selfie_inset_shrinks_to_clear_head_and_text() -> None:
    style = resolve("preset:editorial_serif").data
    face = (450, 885, 500, 385)
    plan = plan_blocks("photo", {"headline": "A better night", "inset": "x.png"},
                       style, face_box=face)
    x, y, width, height = plan.inset_box
    head = head_box(Box(*face))
    assert plan.inset_side == "left"
    assert .7 <= plan.inset_scale < 1
    assert 60 <= x and x + width <= 1020
    assert 280 <= y and y + height <= 1640
    assert x + width <= head.x - 24 or y >= head.y1 + 24
    assert all(x + width <= block["x"] or x >= block["x"] + block["width"] or
               y + height <= block["y"] or y >= block["y"] + block["height"]
               for block in plan.blocks)


def test_inset_refuses_when_lower_corner_cannot_clear_head() -> None:
    with pytest.raises(ValidationError, match=r"--frame N.*omit --inset"):
        plan_blocks("photo", {"headline": "A better night", "inset": "x.png"},
                    resolve("preset:editorial_serif").data,
                    face_box=(0, 1100, 1080, 600))


@pytest.mark.parametrize("face", [(100, 400, 300, 500), (650, 900, 300, 500),
                                   (650, 80, 300, 200), (650, 1500, 300, 300)])
def test_number_block_centres_on_face_and_stays_in_crop(
    face: tuple[int, int, int, int],
) -> None:
    plan = plan_blocks("number", {"number": "5", "label": "NIGHTS"},
                       resolve("preset:editorial_serif").data, face_box=face)
    top = min(b["y"] for b in plan.blocks)
    bottom = max(b["y"] + b["height"] for b in plan.blocks)
    height = bottom - top
    target = max(280 + height / 2, min(1640 - height / 2, face[1] + face[3] / 2))
    assert abs((top + bottom) / 2 - target) <= 1
    assert 280 <= top < bottom <= 1640


def test_small_copy_limit_uses_40_px_minimum() -> None:
    style = resolve("preset:editorial_serif").data
    with pytest.raises(ValidationError, match="40 px"):
        plan_blocks("number", {"number": "5", "label": "W" * 40}, style,
                    face_box=(650, 400, 300, 500))


def test_before_after_moves_above_or_below_face() -> None:
    style = resolve("preset:editorial_serif").data
    values = {"before": "8 hours", "after": "2 hours"}
    above = plan_blocks("before-after", values, style, face_box=(450, 1250, 300, 300))
    assert max(b["y"] + b["height"] for b in above.blocks) <= 1210
    below = plan_blocks("before-after", values, style, face_box=(450, 300, 300, 420))
    assert min(b["y"] for b in below.blocks) >= 760
    with pytest.raises(ValidationError, match="another frame|shorter values"):
        plan_blocks("before-after", values, style, face_box=(450, 570, 300, 800))


def test_suggested_limits_are_measured_from_prefixes() -> None:
    style = resolve("preset:editorial_serif").data
    with pytest.raises(ValidationError, match="shorten to at most") as number_error:
        plan_blocks("number", {"number": "WWWWWW", "label": "NIGHTS"}, style,
                    face_box=(650, 400, 300, 500))
    from yap_studio.render.text_metrics import text_width_em
    expected = max(i for i in range(7)
                   if text_width_em("WWWWWW"[:i], "Instrument Serif", 400) * 220 <= 460)
    assert f"at most {expected} characters" in str(number_error.value)
    headline = "WWWWWWWWW WWWWWWWWW WWWWWWWWW WWWWWWWWW WWWWWWWWW"
    with pytest.raises(ValidationError, match="shorten to at most") as headline_error:
        plan_blocks("photo", {"headline": headline}, style,
                    face_box=(450, 350, 500, 700))
    assert "at most 30 characters" not in str(headline_error.value)


@pytest.mark.parametrize(("look", "values"), [
    ("photo", {"headline": "i " * 29 + "ii", "kicker": "i" * 32}),
    ("number", {"number": "i" * 6, "label": "i" * 30}),
    ("hook", {"headline": "i " * 29 + "ii", "label": "i" * 40}),
    ("before-after", {"before": "i" * 10, "after": "i" * 10,
                      "kicker": "i" * 32, "label": "i" * 40}),
])
def test_copy_at_caps_has_safe_blocks(look: str, values: dict) -> None:
    plan = plan_blocks(look, values, resolve("preset:editorial_serif").data,
                       face_box=(450, 350, 500, 700))
    assert all(60 <= b["x"] and b["x"] + b["width"] <= 1020 and
               280 <= b["y"] and b["y"] + b["height"] <= 1640
               for b in plan.blocks)


def test_block_geometry_uses_its_planned_box() -> None:
    mask = np.zeros((1920, 1080), dtype=np.uint8)
    mask[300:320, 80:100] = 255
    headline = {"role": "headline", "x": 60, "y": 280,
                "width": 960, "height": 100}
    check_geometry(mask, blocks=[headline], band=(280, 640))
    with pytest.raises(ValidationError, match="top third"):
        check_geometry(mask, blocks=[{**headline, "y": 650}], band=(280, 640))
    with pytest.raises(ValidationError, match="planned block"):
        check_geometry(mask, blocks=[{**headline, "x": 200, "width": 300}],
                       band=(280, 640))
