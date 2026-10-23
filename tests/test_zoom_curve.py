"""Pure camera curves and face framing on the output clock."""

import ast
import math
import re
import subprocess
import sys

import pytest

from yap_studio.faces.model import Box
from yap_studio.zoom_curve import (
    ZoomCurve,
    ZoomKey,
    ZoomMove,
    face_center,
    face_leaves_frame,
    punch_keys,
    push_keys,
)


def curve(keys, start=10, end=70):
    return ZoomCurve(start, end, tuple(keys))


def evaluate(expression, n):
    expression = re.sub(r"\bif\(", "choose(", expression).replace("^", "**")

    # Evaluate only the chosen branch, as ffmpeg does.
    class Choose(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "choose":
                return ast.copy_location(ast.IfExp(*node.args), node)
            return node

    tree = ast.fix_missing_locations(Choose().visit(ast.parse(expression, mode="eval")))
    return eval(compile(tree, "<zoom>", "eval"), {"__builtins__": {}}, {
        "n": n, "PI": math.pi, "sin": math.sin, "cos": math.cos, "pow": pow,
        "lt": lambda a, b: a < b,
    })


@pytest.mark.parametrize("scale", [1.5, 2])
def test_snap_punch_frames_and_moves(scale):
    zoom = curve(punch_keys(10, 70, scale, (0.4, 0.3)))
    assert [key.frame for key in zoom.keys] == [10, 12, 66, 70]
    expected = [1, scale ** (1 - .5**3), scale]
    for frame, value in enumerate(expected, 10):
        assert zoom.at(frame) == pytest.approx((value, .4, .3), abs=1e-12)
    assert zoom.at(40) == (scale, .4, .3)
    for frame in range(66, 71):
        expected_scale = scale * (1 / scale) ** (1 - (1 - (frame - 66) / 4)**3)
        assert zoom.at(frame) == pytest.approx((expected_scale, .4, .3), abs=1e-12)
    assert zoom.moves() == (ZoomMove(10, 12, 1, scale), ZoomMove(66, 70, scale, 1))


def test_cut_punch():
    zoom = curve(punch_keys(10, 70, 1.2, (.5, .42), "cut"))
    assert zoom.at(10) == zoom.at(69) == (1.2, .5, .42)
    assert zoom.at(70) == (1, .5, .42)
    assert zoom.moves() == (ZoomMove(10, 10, 1, 1.2), ZoomMove(70, 70, 1.2, 1))


def test_first_key_includes_cut_entrance():
    zoom = curve([ZoomKey(10, 1.01, (.5, .42), "snap")])
    assert zoom.moves() == (ZoomMove(10, 10, 1, 1.01),)


@pytest.mark.parametrize("ease", ["linear", "ease-in-out"])
@pytest.mark.parametrize("ramp", [None, 30])
def test_push_frames_and_hold(ease, ramp):
    zoom = curve(push_keys(10, 70, 1.3, (.5, .42), ease, ramp))
    duration = ramp or 60
    for frame in range(10, 71):
        t = min(1, (frame - 10) / duration)
        weight = t if ease == "linear" else (1 - math.cos(math.pi * t)) / 2
        assert zoom.at(frame) == pytest.approx((1.3 ** weight, .5, .42), abs=1e-12)
    assert zoom.moves() == (ZoomMove(10, 10 + duration, 1, 1.3),)


def test_mixed_keys_sort_duplicates_pan_and_short_snap():
    zoom = curve([
        ZoomKey(40, 1.3, (.8, .3), "linear"),
        ZoomKey(10, 1, (.4, .3)), ZoomKey(20, 1.2, (.4, .3), "cut"),
        ZoomKey(22, 1.1, (.4, .3), "snap"),
        ZoomKey(60, 1.3, (.8, .7), "ease-in-out"),
        ZoomKey(60, 1.4, (.8, .7), "ease-in-out"),
    ])
    assert zoom.at(0) == (1, .4, .3)
    assert zoom.at(19) == (1, .4, .3)
    assert zoom.at(21) == pytest.approx((1.2 * (1.1 / 1.2) ** .875, .4, .3), abs=1e-12)
    assert zoom.at(31) == pytest.approx((math.sqrt(1.1 * 1.3), .6, .3), abs=1e-12)
    assert zoom.at(50) == pytest.approx((math.sqrt(1.3 * 1.4), .8, .5), abs=1e-12)
    assert zoom.at(80) == (1.4, .8, .7)
    assert zoom.max_scale() == 1.4
    assert zoom.moves() == (ZoomMove(20, 20, 1, 1.2), ZoomMove(20, 22, 1.2, 1.1),
                            ZoomMove(22, 40, 1.1, 1.3), ZoomMove(40, 60, 1.3, 1.4))
    pan = curve([ZoomKey(10, 1, (.2, .3)), ZoomKey(20, 1, (.8, .7), "linear")])
    assert pan.moves() == ()


@pytest.mark.parametrize("keys", [
    punch_keys(10, 70, 1.3, (.4, .3)),
    push_keys(10, 70, 1.2, (.4, .3)),
    (ZoomKey(10, 1, (.2, .3)), ZoomKey(20, 1.2, (.8, .7), "cut"),
     ZoomKey(30, 1.4, (.7, .6), "snap"), ZoomKey(40, 1.2, (.6, .5), "linear"),
     ZoomKey(60, 1.3, (.4, .2), "ease-in-out")),
])
@pytest.mark.parametrize("clock", ["crop", "scale"])
def test_expressions_match_every_frame(keys, clock):
    zoom = curve(keys)
    for first in (0, 17, 41):
        expressions = zoom.expressions(first)
        if clock == "scale":
            expressions = tuple(expr.replace("(n+", "(n-1+") for expr in expressions)
        for frame in range(0, 81):
            n = frame - first + (clock == "scale")
            assert tuple(evaluate(expr, n) for expr in expressions) == pytest.approx(
                zoom.at(frame), abs=1e-12)
    assert "*pow(" in zoom.expressions(0)[0]
    if len({key.center for key in keys}) == 1:
        assert zoom.expressions(0)[1:] == ("0.400000000000", "0.300000000000")


def test_legacy_swell_payload_expression_and_moves():
    zoom = ZoomCurve.from_payload(10, 70, {"peak_scale": 1.08, "center": [.4, .3]})
    assert zoom == ZoomCurve.swell(10, 70, 1.08, (.4, .3))
    assert zoom.expressions(17) == (
        "(1+0.080000000000*sin(PI*(n+7)/60)^2)", "0.400000000000", "0.300000000000")
    assert zoom.max_scale() == 1.08
    assert zoom.moves() == (ZoomMove(10, 40, 1, 1.08), ZoomMove(40, 70, 1.08, 1))
    for frame in range(10, 71):
        assert zoom.at(frame)[0] == 1 + (1.08 - 1) * math.sin(math.pi * (frame - 10) / 60)**2
    keyed = ZoomCurve.from_payload(10, 70, {"curve": [
        {"frame": 10, "scale": 1, "center": [.4, .3], "ease": "cut"}]})
    assert keyed.at(40) == (1, .4, .3)


def test_face_center_union_clamp_and_impossible_fit():
    boxes = {0: (Box(350, 500, 200, 200), Box(800, 1000, 10, 10)),
             1: (Box(450, 600, 200, 200),)}
    center, fits = face_center(boxes, range(2), 1.3)
    assert fits and center == pytest.approx((500 / 1080, 610 / 1920))
    top = {0: (Box(300, 0, 300, 300),)}
    assert face_center(top, range(1), 1.3) == ((450 / 1080, 0), True)
    too_big = {0: (Box(0, 0, 1000, 1800),)}
    assert face_center(too_big, range(1), 1.3)[1] is False
    assert face_center({}, range(2), 1.3) is None


def test_face_leaves_frame_first_bad_frame():
    zoom = curve(push_keys(10, 70, 1.3, (.5, .5)))
    boxes = {frame: (Box(20, 500, 100, 100),) for frame in range(10, 70)}
    expected = next(frame for frame in range(10, 70)
                    if 540 + (20 - 540) * zoom.at(frame)[0] < 0)
    assert face_leaves_frame(zoom, boxes) == (expected, "frame")
    assert face_leaves_frame(zoom, {}) is None


def test_face_helpers_accept_dense_frame_sequences():
    frames = ((Box(300, 0, 300, 300),),) * 30
    assert face_center(frames, range(30), 1.3) == ((450 / 1080, 0), True)
    zoom = ZoomCurve(0, 30, push_keys(0, 30, 1.3, (.5, .5)))
    assert face_leaves_frame(zoom, frames) == (1, "frame")



def test_standalone_import_stays_light():
    subprocess.run([sys.executable, "-c", "import sys; import yap_studio.zoom_curve; "
                    "assert 'numpy' not in sys.modules"], check=True)


def test_face_helpers_do_not_mutate_sparse_frames():
    from collections import defaultdict

    frames = defaultdict(tuple, {10: (Box(400, 400, 100, 100),)})
    before = dict(frames)
    face_center(frames, range(10, 70), 1.3)
    face_leaves_frame(ZoomCurve(10, 70, push_keys(10, 70, 1.3, (.5, .42))), frames)
    assert dict(frames) == before


def test_linear_push_log_symmetry():
    zoom = curve(push_keys(10, 70, 2, (.5, .42)))
    assert zoom.at(40)[0] == pytest.approx(math.sqrt(2))
    duration = 60 * math.log(1.5) / math.log(2)
    assert duration == pytest.approx(60 - 60 * math.log(4 / 3) / math.log(2))
    assert zoom.at(10 + duration)[0] == pytest.approx(1.5)
    assert zoom.at(70 - duration)[0] == pytest.approx(4 / 3)
    assert zoom.at(25)[0] / zoom.at(10)[0] == pytest.approx(
        zoom.at(70)[0] / zoom.at(55)[0])


def test_face_center_caption_clamp_and_margin_fallback():
    boxes = {0: (Box(400, 1100, 200, 300), Box(0, 1800, 20, 20))}
    center, fits = face_center(boxes, range(1), 2, bottom=1600)
    assert fits
    assert center == pytest.approx((500 / 1080, 1224 / 1920))
    assert 2 * 1400 - center[1] * 1920 == pytest.approx(1600 - 24)
    edge = {0: (Box(0, 600, 540, 200),)}
    assert face_center(edge, range(1), 2) == ((0, 660 / 1920), True)
    too_wide = {0: (Box(0, 600, 541, 200),)}
    center, fits = face_center(too_wide, range(1), 2)
    assert not fits and center == pytest.approx((270.5 / 1080, 660 / 1920))
    too_tall = {0: (Box(400, 0, 200, 810),)}
    assert face_center(too_tall, range(1), 2, bottom=1600)[1] is False
    assert face_center({}, range(1), 2, bottom=1600) is None


def test_face_center_caption_bottom_margin_and_fallback():
    boxes = {0: (Box(400, 1100, 200, 300),)}
    center, fits = face_center(boxes, range(1), 2, bottom=1600, bottom_margin=28)
    assert fits
    assert center[0] == pytest.approx(500 / 1080)
    assert 2 * 1400 - center[1] * 1920 == pytest.approx(1600 - 28)
    full_height = {0: (Box(400, 100, 200, 800),)}
    center, fits = face_center(full_height, range(1), 2, bottom=1600, bottom_margin=28)
    assert fits
    assert center == pytest.approx((500 / 1080, 200 / 1920))
    assert 2 * 100 - center[1] * 1920 == pytest.approx(0)
    assert 2 * 900 - center[1] * 1920 == pytest.approx(1600)


def test_face_leaves_frame_captions_and_edge_priority():
    zoom = ZoomCurve(0, 30, punch_keys(0, 30, 2, (.5, .5), "cut"))
    assert face_leaves_frame(zoom, {0: (Box(450, 1100, 100, 200),)}, bottom=1600) == (
        0, "captions")
    assert face_leaves_frame(zoom, {0: (Box(0, 1100, 100, 200),)}, bottom=1600) == (
        0, "frame")
    assert face_leaves_frame(zoom, {0: (Box(400, 500, 200, 300),
                                       Box(0, 1800, 20, 20))}, bottom=1600) is None


@pytest.mark.parametrize("ease", ["cut", "snap"])
def test_face_leaves_frame_only_when_chin_moves_further_down(ease):
    boxes = {frame: (Box(400, 1500, 200, 200),) for frame in range(30)}
    for cy in (1700, 1800):
        zoom = ZoomCurve(0, 30, punch_keys(0, 30, 1.5, (500 / 1080, cy / 1920), ease))
        assert face_leaves_frame(zoom, boxes, bottom=1600) is None
    zoom = ZoomCurve(0, 30, punch_keys(0, 30, 1.5, (500 / 1080, 1400 / 1920), ease))
    assert face_leaves_frame(zoom, boxes, bottom=1600) == (
        0 if ease == "cut" else 1, "captions")


@pytest.mark.parametrize("box", [
    Box(-10, 600, 200, 200), Box(400, -10, 200, 200),
    Box(900, 600, 200, 200), Box(400, 1750, 200, 200),
])
def test_face_leaves_frame_only_when_existing_edge_moves_further_out(box):
    boxes = {frame: (box,) for frame in range(30)}
    held = ZoomCurve(0, 30, (ZoomKey(0, 1, (.5, .5)),))
    assert face_leaves_frame(held, boxes) is None
    zoom = ZoomCurve(0, 30, punch_keys(0, 30, 1.5, (.5, .5)))
    assert face_leaves_frame(zoom, boxes) == (1, "frame")


@pytest.mark.parametrize("zoom", [
    ZoomCurve(10, 70, punch_keys(10, 70, scale, (.5, .42), ease))
    for scale in (1.5, 2) for ease in ("snap", "cut")
] + [
    ZoomCurve(10, 70, push_keys(10, 70, 2, (.5, .42), ease, ramp))
    for ease in ("linear", "ease-in-out") for ramp in (None, 30)
] + [
    ZoomCurve(10, 70, (ZoomKey(10, 1, (.5, .42)),
                      ZoomKey(20, 2, (.5, .42), "snap"),
                      ZoomKey(40, 1.5, (.5, .42), "linear"),
                      ZoomKey(50, 1.8, (.5, .42), "ease-in-out"))),
])
def test_scale_motion_valid_moves(zoom):
    from yap_studio.zoom_curve import declared_moves, scale_motion

    scales = zoom.scales(9, 71)
    assert scales[0] == scales[-1] == 1
    assert scales[1:-1] == [zoom.at(frame)[0] for frame in range(10, 70)]
    assert scale_motion(scales, 9, declared_moves(zoom)) == ()


@pytest.mark.parametrize("end", [69, 70, 71])
def test_scale_motion_legacy_swell(end):
    from yap_studio.zoom_curve import declared_moves, scale_motion

    zoom = ZoomCurve.swell(10, end, 1.08, (.5, .42))
    assert scale_motion(zoom.scales(9, end + 1), 9, declared_moves(zoom)) == ()


def test_scale_motion_flaws_and_tolerance():
    from yap_studio.zoom_curve import MotionFlaw, scale_motion

    assert scale_motion([1, 1.5, 1.5], 10, ()) == (MotionFlaw(11, "jump", math.log(1.5)),)
    move = (ZoomMove(10, 14, 1, 2),)
    backwards = scale_motion([1, 1.5, 1.4, 1.8, 2], 10, move)
    assert len(backwards) == 1 and backwards[0].frame == 12 and backwards[0].kind == "jerk"
    for logs in ([0, .3, .3, .6, math.log(2)], [0, .3, .32, .6, math.log(2)]):
        flaws = scale_motion([math.exp(value) for value in logs], 10, move)
        assert len(flaws) == 1 and flaws[0].frame == 12 and flaws[0].kind == "jerk"
    assert scale_motion([1, math.exp(1e-7)], 10, ()) == ()
    assert scale_motion([1, math.exp(.009)], 10, (), tolerance=.01) == ()
    assert scale_motion([1, math.exp(-1e-7)], 10, move) == ()
    assert scale_motion([1, 2], 10, (ZoomMove(11, 11, 1, 2),)) == ()
    assert scale_motion([1, 1.5, 1.5, 2], 10,
                        (ZoomMove(10, 12, 1, 1.5), ZoomMove(12, 13, 1.5, 2))) == ()
    assert scale_motion([], 10, ()) == ()
    zoom = curve(punch_keys(10, 70, 2, (.5, .42)))
    held = zoom.scales(66, 71)
    held[2] = held[1]
    assert scale_motion(held, 66, zoom.moves())[0] == MotionFlaw(68, "jerk", 0)
    flaws = scale_motion([1, 1.5, 1.5, 1.8, 2], 10,
                         (*move, ZoomMove(11, 11, 1, 1.5)))
    assert len(flaws) == 1 and flaws[0].frame == 12 and flaws[0].kind == "jerk"
