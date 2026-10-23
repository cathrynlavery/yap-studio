"""Delete slots, measured layouts and deterministic native motion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_cards import image as image

from yap_studio import cards
from yap_studio.errors import ValidationError
from yap_studio.render.text_metrics import text_width_em
from yap_studio.style import resolve


def plan(*, frames: int = 90, width: int = 960, **slots) -> cards.CardPlan:
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_width_px"] = width
    return cards.card_layer_plan("jiggle_delete", {
        "items": ["Keep", "Extra", "Finish"], "remove": [2], **slots,
    }, style, frames=frames)


@pytest.mark.parametrize("remove,fix", [
    ([0], "positions within 1..3"), ([4], "positions within 1..3"),
    ([1.5], "whole"), ([2, 2], "once"), ([1, 2, 3], "keep at least one"),
    ([], "1..5 items"), ([True], "finite number"),
])
def test_remove_positions_have_fixes(remove: list, fix: str) -> None:
    with pytest.raises(ValidationError, match=fix) as caught:
        plan(remove=remove)
    assert caught.value.parameter == "slots.remove"


def test_default_and_partial_cues_and_minimum() -> None:
    v = plan(items=["A", "B", "C", "D"], remove=[3, 1]).variables
    assert json.loads(v["cues"]) == pytest.approx([0.9, 1.4])
    v = plan(items=["A", "B", "C", "D"], remove=[3, 1], cues=[1.1]).variables
    assert json.loads(v["cues"]) == pytest.approx([1.1, 1.6])
    with pytest.raises(ValidationError, match="at least 2 s"):
        plan(frames=59)
    assert plan(frames=60, wiggle_at=0, cues=[0.3]).frames == 60


@pytest.mark.parametrize("slots,parameter,fix", [
    ({"cues": [0.2]}, "cues", "seconds within"),
    ({"cues": [1.8]}, "cues", "0.95 s"),
    ({"cues": [1, 1.1]}, "cues", "at most 1"),
    ({"remove": [1, 2], "cues": [1.4, 1.2]}, "cues", "non-decreasing"),
    ({"wiggle_at": -0.1}, "wiggle_at", "seconds within"),
    ({"wiggle_at": 2.5}, "wiggle_at", "seconds within"),
])
def test_cue_windows_refuse_with_a_fix(slots: dict, parameter: str, fix: str) -> None:
    with pytest.raises(ValidationError, match=fix) as caught:
        plan(**slots)
    assert caught.value.parameter == f"slots.{parameter}"


def test_chip_rows_are_measured_centred_and_repacked() -> None:
    v = plan(width=450, items=["First", "Second", "Third", "Fourth"],
             remove=[2, 4]).variables
    items, layouts = json.loads(v["items"]), json.loads(v["layouts"])
    assert [layout["remaining"] for layout in layouts] == [[1, 2, 3, 4], [1, 3, 4], [1, 3]]
    assert len({pos["y_px"] for pos in layouts[0]["positions"].values()}) == 2
    assert len({pos["y_px"] for pos in layouts[-1]["positions"].values()}) == 1
    assert layouts[-1]["positions"]["3"] != layouts[0]["positions"]["3"]
    for layout in layouts:
        rows = {}
        for number in layout["remaining"]:
            position = layout["positions"][str(number)]
            rows.setdefault(position["y_px"], []).append((position, items[number - 1]))
        for row in rows.values():
            left = row[0][0]["x_px"]
            right = row[-1][0]["x_px"] + row[-1][1]["width_px"]
            assert left == pytest.approx(layout["width_px"] - right)
    assert v["label_px"] >= 30
    for item in items:
        assert text_width_em(item["label"], "Inter", 600) * v["label_px"] <= (
            item["width_px"] - 36
        )


def test_labels_and_rows_refuse_at_thirty_pixels() -> None:
    with pytest.raises(ValidationError, match="shorten labels to .* characters"):
        plan(width=240, items=["W" * 24, "A"])
    with pytest.raises(ValidationError, match="remove items"):
        plan(width=450, items=["W" * 10] * 6)


@pytest.mark.parametrize("tiles", [False, True])
def test_label_size_grows_for_short_items_and_keeps_long_items_readable(
    tiles: bool, image: Path,
) -> None:
    short_labels = ["Go", "Stay"]
    long_labels = ["Longer", "Second"] if tiles else ["A longer spoken phrase"] * 6
    short = plan(items=short_labels, remove=[1],
                 images=[image] * len(short_labels) if tiles else [])
    long = plan(items=long_labels, remove=[1],
                images=[image] * len(long_labels) if tiles else [])
    assert 48 < short.variables["label_px"] <= 56
    assert 30 <= long.variables["label_px"] < short.variables["label_px"]
    for layer in (short, long):
        v = layer.variables
        items = json.loads(v["items"])
        positions = json.loads(v["layouts"])[0]["positions"]
        assert min(position["x_px"] for position in positions.values()) == 24
        assert max(position["x_px"] + item["width_px"]
                   for position, item in zip(positions.values(), items, strict=True)) == (
            v["delete_panel_width_px"] - 24
        )
        assert v["panel_height_px"] <= 272
        assert (layer.x, layer.y, layer.width, layer.height) == (20, 280, 1040, 360)
        for item in items:
            label_width = text_width_em(item["label"], "Inter", 600) * v["label_px"]
            assert label_width <= item["width_px"]
            if tiles:
                assert item["picture_px"] == item["width_px"]
                assert item["height_px"] > item["picture_px"]


def test_chip_padding_badges_and_gaps_scale_with_the_label() -> None:
    big = plan(items=["Go", "Stay"] * 3, remove=[1]).variables
    small = plan(items=["A longer spoken phrase"] * 6, remove=[1]).variables
    geometry = []
    for v in (big, small):
        items = json.loads(v["items"])
        positions = json.loads(v["layouts"])[0]["positions"]
        rows = sorted({position["y_px"] for position in positions.values()})
        text = text_width_em(items[0]["label"], "Inter", 600) * v["label_px"]
        geometry.append({
            "padding": items[0]["width_px"] - text,
            "gap": positions["2"]["x_px"] - positions["1"]["x_px"] - items[0]["width_px"],
            "row_gap": rows[1] - rows[0] - items[0]["height_px"],
            "badge": items[0]["keyframes"][18]["badge_px"],
        })
    assert big["label_px"] > small["label_px"]
    assert all(geometry[0][key] > geometry[1][key] for key in geometry[0])
    assert geometry[0]["badge"] / geometry[1]["badge"] == pytest.approx(
        big["label_px"] / small["label_px"], abs=1e-6,
    )


def test_panel_hugs_each_layout_when_large_labels_wrap() -> None:
    layer = plan(items=["Keep", "Duplicate", "Pause", "Finish"], remove=[2])
    v = layer.variables
    items, layouts = json.loads(v["items"]), json.loads(v["layouts"])
    assert layouts[0]["height_px"] > layouts[-1]["height_px"]
    assert layouts[0]["width_px"] > layouts[-1]["width_px"]
    for layout in layouts:
        positions = layout["positions"]
        assert min(position["y_px"] for position in positions.values()) == 24
        assert max(position["y_px"] + items[int(number) - 1]["height_px"]
                   for number, position in positions.items()) == layout["height_px"] - 24
        assert max(position["x_px"] + items[int(number) - 1]["width_px"]
                   for number, position in positions.items()) == layout["width_px"] - 24
    poses = json.loads(v["panel_keyframes"])
    assert len(poses) == layer.frames + 1
    for frame, layout in ((0, layouts[0]), (44, layouts[-1])):
        assert poses[frame]["width_px"] == layout["width_px"]
        assert poses[frame]["height_px"] == layout["height_px"]
    assert (layer.width, layer.height) == (1040, 360)


def test_image_list_assets_and_tile_labels(image: Path, tmp_path: Path) -> None:
    layer = plan(images=[image, image, image])
    assert [name for name, _ in layer.assets] == ["item-1.png", "item-2.png", "item-3.png"]
    items = json.loads(layer.variables["items"])
    assert all(item["picture_px"] == item["width_px"] for item in items)
    assert all(item["height_px"] > item["picture_px"] for item in items)
    prepared = cards.prepare_card(layer, tmp_path / "work")
    assert all((prepared / name).read_bytes() == image.read_bytes() for name, _ in layer.assets)
    with pytest.raises(ValidationError, match="same count as items") as caught:
        plan(images=[image])
    assert caught.value.parameter == "slots.images"
    with pytest.raises(ValidationError, match="shorten labels"):
        plan(items=["W" * 24, "A", "B"], images=[image] * 3)


@pytest.mark.parametrize("value,fix", [
    ([1, 1, 1], "expected an image path"),
    (["bad.svg"] * 3, "use a .png"),
    (["bad\n.png"] * 3, "remove control characters"),
    (["missing.png"] * 3, "existing, readable"),
    ("picture.png", "expected a list"),
    (["missing.png"] * 7, "0..6 items"),
])
def test_image_list_paths_have_the_image_slot_checks(value: object, fix: str) -> None:
    with pytest.raises(ValidationError, match=fix) as caught:
        plan(images=value)
    assert caught.value.parameter.startswith("slots.images")


def test_frame_poses_wiggle_remove_reflow_and_settle() -> None:
    layer = plan()
    v = layer.variables
    items = json.loads(v["items"])
    frames = items[0]["keyframes"]
    assert len(frames) == layer.frames + 1
    assert frames[0]["angle"] == frames[-1]["angle"] == 0
    assert any(abs(frame["angle"]) > 1 for frame in frames[10:25])
    assert frames[12]["angle"] != items[2]["keyframes"][12]["angle"]
    removed = items[1]["keyframes"]
    assert removed[27]["opacity"] == 1
    assert removed[35]["opacity"] == 0
    layouts = json.loads(v["layouts"])
    assert frames[44]["x_px"] == pytest.approx(layouts[-1]["positions"]["1"]["x_px"])
    assert frames[50]["angle"] == frames[50]["badge_opacity"] == 0
    assert removed[28]["badge_dip_px"] > 0
    assert frames[27]["at_s"] == 26.5 / 30


def test_kind_is_slots_only_and_uses_native_animation() -> None:
    from yap_studio.card_project import PROJECT_CARD_TEMPLATES

    assert "jiggle_delete" in cards.CARD_TEMPLATES
    assert "jiggle_delete" not in PROJECT_CARD_TEMPLATES
    source = (cards.template_dir("jiggle_delete") / "template.html").read_text()
    assert '.animate(' in source and "data-no-timeline" in source
    assert "::before" in source and "::after" in source
    assert "steps(1, end)" in source
