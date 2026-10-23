"""Final drawn alpha bounds and portable layout composition."""

from __future__ import annotations

from pathlib import Path

import pytest
import synth

from yap_studio.errors import ValidationError
from yap_studio.faces.clearance import DEFAULT_HAIR, LAYOUT_SCHEMA, Layout
from yap_studio.faces.final import drawn_box, drawn_boxes, final_layout
from yap_studio.faces.model import Box


def test_drawn_box_excludes_faint_border_and_unions_frames(tmp_path: Path) -> None:
    layer = synth.picture_source(
        tmp_path / "card.mov", "color=c=black@0:s=64x48:r=30,format=rgba,"
        "geq=r=255:g=255:b=255:a='if(between(X,10+10*N,29+10*N)*"
        "between(Y,12,23),255,if(between(X,6,43)*between(Y,8,27),51,0))'",
        2, alpha=True,
    )
    assert drawn_box(layer, x=100, y=200, width=64, height=48) == Box(110, 212, 30, 12)
    assert drawn_box(layer, x=100, y=200, width=64, height=48, threshold=51) == \
        Box(106, 208, 38, 20)


def test_drawn_box_empty_layer(tmp_path: Path) -> None:
    layer = synth.picture_source(tmp_path / "empty.mov",
                                 "color=c=black@0:s=64x48:r=30,format=rgba", 2, alpha=True)
    assert drawn_box(layer, x=100, y=200, width=64, height=48) is None
    assert drawn_boxes(layer, x=100, y=200, width=64, height=48) == []


def test_drawn_boxes_group_moving_alpha_and_preserve_empty_frames(tmp_path: Path) -> None:
    layer = synth.picture_source(
        tmp_path / "moving.mov", "color=c=black@0:s=64x48:r=30,format=rgba,"
        "geq=r=255:g=255:b=255:a='if(eq(N,4),0,"
        "if(between(X,10+10*between(N,2,3),29+10*between(N,2,3))*"
        "between(Y,12,23),255,if(between(X,6,43)*between(Y,8,27),51,0)))'",
        7, alpha=True,
    )
    assert drawn_boxes(layer, x=100, y=200, width=64, height=48) == [
        (0, 2, Box(110, 212, 20, 12)),
        (2, 4, Box(120, 212, 20, 12)),
        (5, 7, Box(110, 212, 20, 12)),
    ]
    assert drawn_boxes(layer, x=100, y=200, width=64, height=48, threshold=51) == [
        (0, 4, Box(106, 208, 38, 20)), (5, 7, Box(106, 208, 38, 20)),
    ]


def test_drawn_boxes_limits_alpha_to_a_canvas_region(tmp_path: Path) -> None:
    layer = synth.picture_source(
        tmp_path / "tiles.mov", "color=black@0:s=160x80:r=30,format=rgba,"
        "geq=r=255:g=255:b=255:a='255*(between(X,8,31)*between(Y,10,29)+"
        "between(X,120,151)*between(Y,40,69))'", 2, alpha=True,
    )
    assert drawn_boxes(layer, x=10, y=20, width=160, height=80,
                       region=Box(10, 20, 80, 80)) == [(0, 2, Box(18, 30, 24, 20))]
    assert drawn_boxes(layer, x=10, y=20, width=160, height=80,
                       region=Box(90, 20, 80, 80)) == [(0, 2, Box(130, 60, 32, 30))]


@pytest.mark.parametrize("measured", [True, False])
def test_final_layout_keeps_caption_evidence_and_element_kinds(measured: bool) -> None:
    captions = [{"id": "caption:0", "text": "Small steps", "start_frame": 3, "end_frame": 18,
                 "boxes": [{"x": 80, "y": 1600, "w": 220, "h": 40}],
                 "words": [{"text": "Small", "start_s": 0.1, "end_s": 0.3},
                           {"text": "steps", "start_s": 0.4, "end_s": 0.6}]}]
    source = {"measured": measured, "reason": None if measured else "No glyph measurement.",
              "captions": captions}
    elements = [{"id": kind, "kind": kind, "x": 100, "y": 200, "w": 300, "h": 100,
                 "start_frame": 6, "end_frame": 36} for kind in ("card", "title", "overlay")]
    document = final_layout(source, elements, width=1080, height=1920, clearance_px=24,
                            caption_gap_px=60)
    layout = Layout.from_dict(document)
    assert [element.kind for element in layout.elements] == ["card", "title", "overlay"]
    assert (layout.width, layout.height, layout.clearance_px, layout.hair) == \
        (1080, 1920, 24, DEFAULT_HAIR)
    assert layout.caption_gap_px == document["caption_gap_px"] == 60
    assert document == {"schema": LAYOUT_SCHEMA, "width": 1080, "height": 1920,
                        **source, "elements": elements, "clearance_px": 24,
                        "caption_gap_px": 60, "hair": DEFAULT_HAIR.to_dict()}
    assert document["captions"] is captions


def test_final_layout_refuses_invalid_elements() -> None:
    with pytest.raises(ValidationError, match="kind"):
        final_layout({"measured": True, "reason": None, "captions": []},
                     [{"id": "card", "kind": "graphic", "start_frame": 0, "end_frame": 30}],
                     width=1080, height=1920, clearance_px=24, caption_gap_px=100)
