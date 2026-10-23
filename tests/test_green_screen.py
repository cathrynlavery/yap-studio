"""Synthetic green screen geometry, face placement and picture layers."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import synth
from test_assemble_picture import overlay_image
from test_faces_mapping import _cut

from yap_studio import ffmpeg
from yap_studio.faces.mapping import map_cut_faces, map_picture_faces
from yap_studio.faces.model import Box
from yap_studio.stages import green_screen as green
from yap_studio.stages.assemble_picture import _Graph, _source
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline


@pytest.mark.parametrize("layout,box,edges", [
    ("full", (0, 0, 1080, 1920), ()),
    ("bottom", (189, 672, 702, 1248), ("left", "right", "top")),
    ("corner_left", (0, 1056, 486, 864), ("right", "top")),
    ("corner_right", (594, 1056, 486, 864), ("left", "top")),
])
def test_person_geometry(layout, box, edges) -> None:
    assert green.person_box(layout) == box
    assert green.fade_edges(layout) == edges


@pytest.mark.parametrize("payload,seconds", [
    ({}, 0),
    ({"layout": "full", "blend": "none", "enter": "cut", "motion": "none"}, 0),
    ({"layout": "bottom"}, 40),
    ({"blend": "wrap"}, 40),
    ({"enter": "wipe"}, 40),
    ({"motion": "push"}, 40),
])
def test_decode_seconds_counts_only_look_windows(payload: dict, seconds: float) -> None:
    item = ResolvedElement("broll-page", "background", "broll", 30, 180, "front", payload)
    assert green.decode_seconds(item) == seconds
    assert green.decode_seconds(replace(item, end_frame=60)) == seconds / 5


def test_look_graph_limits_per_pixel_expressions_to_single_frames_and_wipe() -> None:
    graph = _Graph()
    item = ResolvedElement("broll-page", "background", "broll", 12, 72, "front", {
        "layout": "bottom", "blend": "wrap", "wipe_frames": 9,
    })
    green.person_layer(graph, "person", "background", item, (1, -1, 2))
    green._wipe(graph, "background_rgba", 9)
    expressions = [chain for chain in graph.filters if "geq=" in chain]
    assert len(expressions) == 3
    assert all("enable='lt(n," in chain for chain in expressions)
    assert sum("enable='lt(n,9)'" in chain for chain in expressions) == 2
    edge = next(chain for chain in expressions if "enable='lt(n,1)'" in chain)
    assert "trim=end_frame=1,loop=loop=-1:size=1" in edge
    assert "trim=end_frame=60" in edge
    assert not any("geq=r=" in chain for chain in graph.filters)
    assert not any("blend=all_expr=" in chain or "lut2=" in chain for chain in graph.filters)
    assert any("scale=270:480:flags=area,gblur=sigma=3,"
               "scale=1080:1920:flags=bicubic" in chain for chain in graph.filters)
    assert any("scale=out_color_matrix=bt709:out_range=full,format=yuv444p" in chain
               for chain in graph.filters)


def test_native_wrap_band_stays_within_one_code() -> None:
    def pixels(native: bool) -> np.ndarray:
        graph = _Graph()
        graph.inputs = ["-f", "lavfi", "-i", "nullsrc=s=256x256,format=gray,geq=lum=X",
                        "-f", "lavfi", "-i", "nullsrc=s=256x256,format=gray,geq=lum=Y"]
        if native:
            output = green._wrap_band(graph, "0:v", "1:v")
        else:
            output = graph.filter("0:v][1:v", "blend=all_expr='"
                                  f"clip(2*A*(255-B)/255,0,255)*{green.WRAP_STRENGTH}'")
        result = ffmpeg.run([
            "ffmpeg", "-v", "error", "-filter_complex_threads", "1", *graph.inputs,
            "-filter_complex", ";".join(graph.filters), "-map", f"[{output}]",
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ], binary=True, timeout_s=60)
        return np.frombuffer(result.stdout, np.uint8).astype(np.int16)

    assert int(np.abs(pixels(True) - pixels(False)).max()) <= 1


def test_native_skin_mask_matches_full_range_bt709_thresholds() -> None:
    def pixels(native: bool) -> np.ndarray:
        graph = _Graph()
        graph.inputs = ["-f", "lavfi", "-i", "nullsrc=s=256x256:r=30,format=gbrp,"
                        "geq=r='X':g='Y':b='mod(13*X+29*Y,256)'"]
        if native:
            output = green._pull(graph, "0:v", (12, -12, 12))
        else:
            original, adjusted, skin = green._split(graph, "0:v", 3)
            adjusted = graph.filter(adjusted, "lutrgb=r='clip(val+12,0,255)':"
                                     "g='clip(val-12,0,255)':b='clip(val+12,0,255)'")
            skin = graph.filter(skin, "scale=out_color_matrix=bt709:out_range=full,"
                                "format=yuv444p,geq=lum='255*between(cb(X,Y),77,127)*"
                                "between(cr(X,Y),133,173)':cb=128:cr=128,extractplanes=y,"
                                "gblur=sigma=1.5,format=gray,"
                                f"lut=c0='val*{green.SKIN_KEEP}',format=gbrp")
            output = graph.filter(f"{adjusted}][{original}][{skin}", "maskedmerge=planes=7")
        result = ffmpeg.run([
            "ffmpeg", "-v", "error", "-filter_complex_threads", "1", *graph.inputs,
            "-filter_complex", ";".join(graph.filters), "-map", f"[{output}]",
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ], binary=True, timeout_s=60)
        return np.frombuffer(result.stdout, np.uint8).astype(np.int16)

    np.testing.assert_array_equal(pixels(True), pixels(False))


def test_wrap_mask_keeps_colour_when_merge_inputs_can_scale() -> None:
    graph = _Graph()
    graph.inputs = ["-f", "lavfi", "-i", "color=red:s=4x4,format=gbrp",
                    "-f", "lavfi", "-i", "color=blue:s=4x4,format=gbrp",
                    "-f", "lavfi", "-i", "nullsrc=s=4x4,format=gray,lut=c0=255"]
    alpha, blurred = green._split(graph, "2:v")
    band = green._wrap_band(graph, alpha, blurred)
    picture = graph.filter("0:v", "scale=4:4")
    background = graph.filter("1:v", "scale=4:4")
    output = graph.filter(f"{picture}][{background}][{band}", "maskedmerge=planes=7")
    result = ffmpeg.run([
        "ffmpeg", "-v", "error", "-filter_complex_threads", "1", *graph.inputs,
        "-filter_complex", ";".join(graph.filters), "-map", f"[{output}]",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ], binary=True, timeout_s=60)
    pixels = np.frombuffer(result.stdout, np.uint8).reshape(-1, 3)
    np.testing.assert_allclose(pixels, np.tile((255, 0, 0), (16, 1)), atol=1)


@pytest.mark.parametrize("layout", ["bottom", "corner_left", "corner_right"])
def test_looped_edge_mask_stays_within_one_alpha_code(layout: str) -> None:
    def pixels(native: bool) -> np.ndarray:
        graph = _Graph()
        graph.inputs = ["-f", "lavfi", "-i", "nullsrc=s=1080x1920:r=30,format=rgba,"
                        "geq=r=255:g=0:b=0:a='mod(X+2*Y+N*17,256)'"]
        item = ResolvedElement("broll-page", "background", "broll", 0, 60, "front",
                               {"layout": layout})
        if native:
            output = graph.filter(green.person_layer(graph, "0:v", None, item, (0, 0, 0)),
                                   "alphaextract")
        else:
            x, y, w, h = green.person_box(layout)
            factors = {"left": "X", "right": "W-1-X", "top": "Y"}
            fade = "*".join(f"clip(({factors[edge]})/24,0,1)" for edge in green.fade_edges(layout))
            output = graph.filter("0:v", f"alphaextract,scale={w}:{h}:flags=bicubic,"
                                   f"geq=lum='lum(X,Y)*{fade}',pad=1080:1920:{x}:{y}:color=black")
        result = ffmpeg.run([
            "ffmpeg", "-v", "error", "-filter_complex_threads", "1", *graph.inputs,
            "-filter_complex", ";".join(graph.filters), "-map", f"[{output}]",
            "-frames:v", "2", "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ], binary=True, timeout_s=60)
        return np.frombuffer(result.stdout, np.uint8).astype(np.int16)

    assert int(np.abs(pixels(True) - pixels(False)).max()) <= 1


def test_wipe_bypass_preserves_pixels_after_entrance() -> None:
    def pixels(native: bool) -> np.ndarray:
        graph = _Graph()
        graph.inputs = ["-f", "lavfi", "-i", "color=red@0.5:s=4x1920:r=30,format=rgba"]
        if native:
            output = green._wipe(graph, "0:v", 9)
        else:
            picture, alpha = green._split(graph, "0:v")
            alpha = graph.filter(alpha, "alphaextract,geq=lum='lum(X,Y)*clip("
                                 "((1-pow(1-min(N+1,9)/9,3))*1984-Y)/64,0,1)'")
            output = graph.filter(f"{picture}][{alpha}", "alphamerge")
        result = ffmpeg.run([
            "ffmpeg", "-v", "error", "-filter_complex_threads", "1", *graph.inputs,
            "-filter_complex", ";".join(graph.filters), "-map", f"[{output}]",
            "-frames:v", "60", "-f", "rawvideo", "-pix_fmt", "rgba", "-",
        ], binary=True, timeout_s=60)
        return np.frombuffer(result.stdout, np.uint8)

    np.testing.assert_array_equal(pixels(True), pixels(False))


@pytest.mark.parametrize("layout", ["bottom", "corner_left", "corner_right"])
def test_scaled_matte_fades_interior_frame_edges_only(layout: str) -> None:
    graph = _Graph()
    graph.inputs = ["-f", "lavfi", "-i", "color=red:s=1080x1920:r=30,format=rgba"]
    item = ResolvedElement("broll-page", "background", "broll", 0, 1, "front", {"layout": layout})
    person = green.person_layer(graph, "0:v", None, item, (0, 0, 0))
    alpha = graph.filter(person, "alphaextract")
    result = ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1", *graph.inputs,
                         "-filter_complex", ";".join(graph.filters), "-map", f"[{alpha}]",
                         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                        timeout_s=60, binary=True)
    pixels = np.frombuffer(result.stdout, np.uint8).reshape(1920, 1080)
    x, y, w, h = green.person_box(layout)
    assert pixels[y, x + w // 2] == 0
    assert pixels[y + 12, x + w // 2] == pytest.approx(127, abs=1)
    assert pixels[y + 24, x + w // 2] == 255
    for edge, column in (("left", x), ("right", x + w - 1)):
        assert pixels[y + h // 2, column] == (0 if edge in green.fade_edges(layout) else 255)
    assert pixels[1919, x + w // 2] == 255


def test_look_is_opt_in_and_windows_are_half_open() -> None:
    assert not green.uses_look({})
    assert not green.uses_look(dict(layout="full", blend="none", enter="cut", motion="none"))
    for field, value in dict(layout="bottom", blend="wrap", enter="wipe", motion="push").items():
        assert green.uses_look({field: value})
    timeline = ResolvedTimeline((ResolvedElement("broll-page", "background", "broll", 2, 11,
                               "front", {"layout": "corner_left", "enter": "wipe"}),),
                                (), (), 12, 0)
    assert green.layout_windows(timeline) == [(2, 11, (0, 1056, 486, 864), 9)]


@pytest.mark.parametrize("framing", ["auto", "keep", "shrink"])
def test_tall_box_maps_faces_and_breaks_phrases_at_both_cuts(framing) -> None:
    from yap_studio.project.model import CutWord

    cut = _cut(30)
    item = ResolvedElement("broll-page", "overlay", "broll", 3, 24, "front",
                           {"placement": "tall_behind", "tall_framing": framing})
    timeline = ResolvedTimeline((item,), (), (), 30, 0)
    mapped, _ = map_picture_faces(cut, timeline, zooms=(), video=cut.video)
    assert mapped.frames[2] == cut.frames[2]
    assert mapped.frames[24] == cut.frames[24]
    for frame in range(3, 24):
        face = cut.frames[frame][0].box
        expected = (Box(108 + .8 * face.x, 384 + .8 * face.y, .8 * face.w, .8 * face.h)
                    if framing == "shrink" else face)
        assert mapped.frames[frame][0].box.to_dict() == expected.to_dict()
    words = [CutWord(str(i), "s1", "word", start, start + .1)
             for i, start in enumerate((0, .1, .7, .8, .9))]
    assert green.caption_breaks(words, green.layout_windows(timeline)) == (
        {"1", "3"} if framing == "shrink" else set())


def test_faces_move_after_wipe_transition_frames() -> None:
    cut = _cut(30)
    window = (3, 24, green.person_box("bottom"), 9)
    kwargs = dict(total_frames=30, end_hold_frames=0, zooms=(), inserts=(), video=cut.video)
    mapped, sources = map_cut_faces(cut, layouts=[window], **kwargs)
    assert mapped.frames[2] == cut.frames[2]
    assert mapped.frames[24] == cut.frames[24]
    for frame in range(3, 24):
        if frame < 12:
            assert mapped.frames[frame] == ()
            continue
        original = cut.frames[frame][0]
        expected = Box(189 + .65 * original.box.x, 672 + .65 * original.box.y,
                       .65 * original.box.w, .65 * original.box.h)
        assert mapped.frames[frame][-1].box == expected
        assert len(mapped.frames[frame]) == 1
    assert sources[1:3] == [
        {"start_frame": 3, "end_frame": 12, "source": "background_wipe"},
        {"start_frame": 12, "end_frame": 24, "source": "background_layout"},
    ]
    assert mapped.gaps == ((3, 12),)
    timeline = ResolvedTimeline((ResolvedElement("broll-page", "background", "broll", 3, 24,
                               "front", {"layout": "bottom", "wipe_frames": 9}),),
                                (), (), 30, 0)
    picture, _ = map_picture_faces(cut, timeline, zooms=(), video=cut.video)
    assert picture.to_dict() == mapped.to_dict()


def test_layout_caption_breaks_use_first_word_at_each_boundary() -> None:
    from yap_studio.project.model import CutWord

    words = [CutWord(str(i), "s1", "word", start, start + .1)
             for i, start in enumerate((.5, 1.1, 1.9, 2.0, 3.8))]
    windows = [(30, 60, green.person_box("corner_right"), 9),
               (60, 120, green.person_box("bottom"), 0)]
    assert green.caption_breaks(words, windows) == {"1", "3"}
    assert green.caption_breaks(words, []) == set()
    assert green.caption_breaks([], windows) == set()


@pytest.mark.parametrize("anchor,frame", [(62.13, 1864), (.03, 1), (.05, 2)])
def test_layout_caption_breaks_snap_anchor_words(anchor: float, frame: int) -> None:
    from yap_studio.project.model import CutWord

    words = [CutWord(name, "s1", name, start, start + .1) for name, start in (
        ("before", anchor - .02), ("anchor", anchor), ("inside", anchor + .2),
        ("end", anchor + 1), ("after", anchor + 1.2),
    )]
    window = (frame, frame + 30, green.person_box("corner_right"), 9)
    assert anchor < frame / 30
    assert green.caption_breaks(words, [window]) == {"anchor", "end"}


@pytest.mark.parametrize("tail", [None, 63.13, 63.14])
def test_layout_phrase_tail_snaps_anchor_start(tail: float | None) -> None:
    from yap_studio.captions import CaptionWord, build_phrases

    phrases = build_phrases([
        CaptionWord("before", "before", 61.7, 61.9),
        CaptionWord("anchor", "corner", 62.13, 62.3),
        CaptionWord("inside", "layout", 62.6, 62.8),
        CaptionWord("after", "after", 63.8, 64.0),
    ], {"max_words_per_phrase": 6, "max_characters_per_phrase": 48},
        forced_breaks={"anchor", "after"})
    window = (1864, 1894, green.person_box("corner_right"), 9)
    inside = next(phrase for phrase in phrases if phrase.words[0].id == "anchor")
    assert inside.start == 62.13 < window[0] / 30
    if tail is None:
        assert inside.end > window[1] / 30
    else:
        inside = replace(inside, end=tail)
        phrases[inside.index] = inside
    clipped = green.clip_layout_phrases(phrases, [window])
    assert clipped[inside.index].end == min(inside.end, window[1] / 30)
    assert clipped[inside.index].words == inside.words


def test_layout_phrase_tail_keeps_last_word_end() -> None:
    from yap_studio.captions import CaptionWord, build_phrases

    phrases = build_phrases([CaptionWord("a", "corner", 1.4, 2.2)],
                            {"max_words_per_phrase": 6, "max_characters_per_phrase": 48})
    window = (30, 60, green.person_box("corner_right"), 9)
    clipped = green.clip_layout_phrases(phrases, [window])
    assert clipped[0].end == 2.2
    assert clipped[0].words == phrases[0].words
    assert green.clip_layout_phrases(phrases, []) == phrases


@pytest.mark.parametrize("layout", ["bottom", "corner_left", "corner_right"])
@pytest.mark.parametrize("word_end", [.3, 1.1])
def test_layout_entrance_clips_preceding_phrase_tail(layout: str, word_end: float) -> None:
    from yap_studio.captions import CaptionWord, build_phrases

    phrases = build_phrases([
        CaptionWord("before", "before", .1, word_end),
        CaptionWord("inside", "inside", 1.3, 1.5),
    ], {"max_words_per_phrase": 6, "max_characters_per_phrase": 48},
        forced_breaks={"inside"})
    window = (30, 60, green.person_box(layout), 9)
    assert phrases[0].end > 1.0
    clipped = green.clip_layout_phrases(phrases, [window])
    assert clipped[0].end == max(1.0, word_end)
    assert clipped[0].words == phrases[0].words
    assert green.clip_layout_phrases(phrases, []) == phrases


def test_caption_avoids_layout_head() -> None:
    import json

    from test_captions_render import _phrases, _style

    from yap_studio.stages.captions import (
        caption_band_geometry,
        caption_layer_plan,
        caption_offsets,
    )

    style = _style(placement="avoid_face", entry_ms=0, background_mode="box")
    geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    cut = _cut(90)
    face = replace(cut.frames[0][0], box=Box(780, (geometry.y - 1056) / .45, 200, 200))
    cut = replace(cut, frames=((face,),) * 90)
    slots = json.loads(caption_layer_plan(_phrases(style), style, frames=90).variables["phrases"])
    moved, _ = map_cut_faces(cut, total_frames=90, end_hold_frames=0, zooms=(), inserts=(),
                            video=cut.video, layouts=[(0, 90, green.person_box("corner_left"), 0)])
    assert caption_offsets(slots, geometry, cut, style)[0]["offset_px"] == 0
    assert caption_offsets(slots, geometry, moved, style)[0]["offset_px"] != 0


def test_pull_chroma_clamp_and_wipe_curve() -> None:
    offset = green.pull_offset((255, 0, 255), (0, 255, 0))
    assert offset == (12, -12, 12)
    offset = green.pull_offset((180, 140, 100), (120, 120, 120))
    assert abs(sum(a * b for a, b in zip(offset, (.2126, .7152, .0722), strict=True))) < 1
    assert green.wipe_alpha(0, 0) == 1
    assert green.wipe_alpha(0, 1919) == 0
    assert green.wipe_alpha(8, 1919) == 1


def _picture(tmp_path: Path, plate: str, background: str, **payload):
    """Render only the new layer builders to a lossless synthetic picture."""
    cut = synth.picture_source(tmp_path / "cut.mov", plate, 18)
    matte = synth.picture_source(tmp_path / "matte.mov",
                                  "color=black:s=1080x1920:r=30,"
                                  "drawbox=x=200:y=200:w=600:h=400:c=white:t=fill", 18)
    image = overlay_image(tmp_path / "background.png", background)
    commands = []
    cut_source, matte_source, bg_source = (
        _source(path, commands, [], owner=path.stem) for path in (cut, matte, image))
    item = ResolvedElement("broll-page", "background", "broll", 0, 18, "front",
                           {"media": {}, "box_px": [0, 0, 1080, 1920], **payload})
    graph = _Graph()
    base = graph.filter(graph.input(cut_source), "settb=1/30,setpts=N")
    base, person = green._split(graph, base)
    alpha = graph.filter(graph.input(matte_source), "settb=1/30,setpts=N,format=gray,"
                         "lut=c0='clip((val-16)*255/80,0,255)',gblur=sigma=1")
    person = graph.filter(f"{person}][{alpha}", "alphamerge")
    _, layers, measurements = green.prepare(graph, [item], {item.id: bg_source},
                                            {"background": person}, cut_source, matte_source,
                                            commands)
    for element, bg, speaker in layers:
        base = graph.overlay(base, bg, element.start_frame, element.end_frame)
        base = graph.overlay(base, speaker, element.start_frame, element.end_frame)
    target = tmp_path / "picture.mkv"
    ffmpeg.run(["ffmpeg", "-v", "error", "-filter_complex_threads", "1", *graph.inputs,
                "-filter_complex", ";".join(graph.filters), "-map", f"[{base}]", "-an",
                "-frames:v", "18", "-threads", "2", "-c:v", "ffv1", "-pix_fmt", "gbrp",
                str(target)], timeout_s=180)
    return target, cut, measurements[0]


def test_light_wrap_only_changes_inside_edge_band(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(green, "PULL_STRENGTH", 0)
    target, cut, measured = _picture(tmp_path, "color=gray:s=1080x1920:r=30",
                                     "color=lime:s=108x192", blend="wrap")
    edge = synth.picture_pixels(target, 10, box=(203, 350, 1, 20)).mean(axis=(0, 1))
    inside = synth.picture_pixels(target, 10, box=(230, 350, 20, 20)).mean(axis=(0, 1))
    original = synth.picture_pixels(cut, 10, box=(230, 350, 20, 20)).mean(axis=(0, 1))
    outside = synth.picture_pixels(target, 10, box=(180, 350, 10, 20)).mean(axis=(0, 1))
    assert edge[1] > inside[1] + 5 and edge[0] < inside[0] - 5
    np.testing.assert_allclose(inside, original, atol=1)
    np.testing.assert_allclose(outside, (0, 255, 0), atol=2)
    assert measured["pull"]["offset_rgb"] == [0, 0, 0]


def test_colour_pull_skin_protection_and_receipt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(green, "WRAP_STRENGTH", 0)
    target, cut, measured = _picture(
        tmp_path, "color=gray:s=1080x1920:r=30,"
        "drawbox=x=200:y=200:w=300:h=400:c=0xc08060:t=fill",
        "color=0xa080c0:s=108x192", blend="wrap")
    pull = measured["pull"]
    assert pull["strength"] == .12 and pull["clamp"] == 12
    assert pull["skin_keep"] == 2 / 3
    assert pull["offset_rgb"] == list(green.pull_offset(pull["background_rgb"], pull["person_rgb"]))
    skin_box, plain_box = (300, 300, 20, 20), (600, 300, 20, 20)
    changes = [synth.picture_pixels(target, 10, box=box).mean(axis=(0, 1))
               - synth.picture_pixels(cut, 10, box=box).mean(axis=(0, 1))
               for box in (skin_box, plain_box)]
    np.testing.assert_allclose(changes[1], pull["offset_rgb"], atol=1)
    np.testing.assert_allclose(changes[0], np.array(pull["offset_rgb"]) / 3, atol=1)
    assert abs(np.dot(changes[1], [.2126, .7152, .0722])) < 1
    assert measured["wrap"] == {"strength": 0, "edge_sigma_px": 6, "blur_sigma_px": 12}


@pytest.mark.parametrize("direction", ["in", "out"])
def test_background_photo_push_reaches_scale(tmp_path: Path, direction: str) -> None:
    target, _, _ = _picture(tmp_path, "color=blue:s=1080x1920:r=30",
                            "color=lime:s=1080x1920,"
                            "drawbox=x=0:y=0:w=100:h=1920:c=magenta:t=fill",
                            motion="push", push_scale=1.08, push_direction=direction)
    first = synth.picture_pixels(target, 0, box=(0, 900, 150, 1))
    last = synth.picture_pixels(target, 17, box=(0, 900, 150, 1))
    before, after = (np.count_nonzero(pixels[0, :, 0] > 127) for pixels in (first, last))
    assert (before, after) == pytest.approx((100, 65) if direction == "in" else (65, 100), abs=3)
