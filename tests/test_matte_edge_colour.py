"""Known colour mixtures and exact lossless edge layers, without real footage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import synth

from yap_studio import ffmpeg
from yap_studio.errors import ProcessingError
from yap_studio.grade import GradeSpec
from yap_studio.matte import edge_colour
from yap_studio.stages import assemble_picture as stage
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline


def write_planes(path: Path, frames, pix_fmt: str) -> Path:
    frames = list(frames)
    h, w = frames[0].shape[:2]
    raw = path.with_suffix(".raw")
    raw.write_bytes(b"".join((np.moveaxis(frame, -1, 0) if frame.ndim == 3 else frame)
                            .tobytes() for frame in frames))
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix_fmt,
                "-s", f"{w}x{h}", "-r", "30", "-i", str(raw), "-c:v", "ffv1",
                "-level", "3", "-fflags", "+bitexact", "-flags:v", "+bitexact", str(path)],
               timeout_s=30)
    raw.unlink()
    return path


def mixture():
    alpha = np.tile(np.clip((np.arange(96) - 30) / 8, 0, 1), (64, 1))
    alpha[24:36, 65:77] = 0.5
    mask = np.round(alpha * 255).astype(np.uint8)
    a = mask[..., None].astype(float) / 255
    foreground = np.array([180, 150, 175])
    background = np.array([95, 45, 40])
    picture = np.round(a * foreground + (1 - a) * background).astype(np.uint8)
    return picture, mask, foreground


@pytest.fixture
def person_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request):
    width, height = getattr(request, "param", (96, 64))
    root = tmp_path / "project"
    (root / "cut").mkdir(parents=True)
    synth.picture_source(root / "cut/cut.mp4", f"color=0x404040:s={width}x{height}:r=30", 12)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, 31], mask[:, 32], mask[:, 33:] = 128, 254, 255
    matte = write_planes(root / "matte.mkv", [mask] * 12, "gray")
    layer = synth.picture_source(root / "blue.mov",
                                 f"color=blue:s={width}x{height}:r=30,format=rgba", 12,
                                 alpha=True)
    blank = synth.picture_source(root / "blank.mov", "color=black@0:s=2x2:r=30,format=rgba",
                                 12, alpha=True)
    monkeypatch.setattr(stage, "WIDTH", width)
    monkeypatch.setattr(stage, "HEIGHT", height)
    monkeypatch.setattr(stage, "CRF", 0)
    monkeypatch.setattr(stage, "ensure_cut_matte", lambda *args, **kwargs: matte)
    monkeypatch.setattr(stage, "measure_behind_layers", lambda *args, **kwargs: ())
    monkeypatch.setattr(stage, "_zoom_inputs", lambda *args: [])
    monkeypatch.setattr(stage, "_insert_inputs", lambda *args: {
        "item": stage._Source(layer, ffmpeg.probe(layer), None)})
    caption = {"path": blank.name, "width": 2, "height": 2, "x": 0, "y": 0,
               "start_frame": 0, "frames": 12}
    return root, caption


@pytest.mark.parametrize("kind", ["background", "graphic"])
def test_person_edge_is_graded_like_the_interior(tmp_path: Path, person_scene, kind: str) -> None:
    root, caption = person_scene
    item = ResolvedElement("item", kind, "test", 0, 12, "behind", {
        "box_px": [0, 0, stage.WIDTH, stage.HEIGHT], "media": {},
    })
    titles = [{"id": "item", "kind": "graphic", "layer": "behind", "path": "blue.mov",
               "width": stage.WIDTH, "height": stage.HEIGHT, "x": 0, "y": 0,
               "start_frame": 0, "frames": 12}] if kind == "graphic" else []
    out = tmp_path / "graded.mp4"
    stage.render_picture(root, out, work_dir=tmp_path / "work", grade=GradeSpec("film"),
                         timeline=ResolvedTimeline((item,), (), (), 12, 0),
                         title_layers=titles, caption_layer=caption)
    pixels = synth.picture_pixels(out, 1).astype(float)
    edge = pixels[16:48, 32].mean(axis=0)
    interior = pixels[16:48, 82:90].mean(axis=(0, 1))
    assert np.abs(interior - 64).max() > 8
    np.testing.assert_allclose(edge, interior, atol=3)


@pytest.mark.parametrize("person_scene", [(1080, 1920)], indirect=True)
@pytest.mark.parametrize("kind", ["background", "graphic"])
@pytest.mark.parametrize("first,last", [(2, 10), (0, 12)])
def test_compare_frames_have_no_edge_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, person_scene,
    kind: str, first: int, last: int,
) -> None:
    root, caption = person_scene
    item = ResolvedElement("item", kind, "test", 0, 12, "behind", {
        "box_px": [0, 0, stage.WIDTH, stage.HEIGHT], "media": {},
    })
    compare = ResolvedElement("compare", "grade_compare", "test", first, last, "front", {})
    timeline = ResolvedTimeline((item, compare), (), (), 12, 0)
    titles = [{"id": "item", "kind": "graphic", "layer": "behind", "path": "blue.mov",
               "width": stage.WIDTH, "height": stage.HEIGHT, "x": 0, "y": 0,
               "start_frame": 0, "frames": 12}] if kind == "graphic" else []
    prepare = stage.prepare_edge_layers
    windows = []

    def layers(*args, **kwargs):
        result = prepare(*args, **kwargs)
        windows.extend((start, start + count) for start, count, _ in result)
        return result

    monkeypatch.setattr(stage, "prepare_edge_layers", layers)
    out = tmp_path / "compare.mp4"
    stage.render_picture(root, out, work_dir=tmp_path / "work", grade=GradeSpec("film"),
                         grade_style={}, timeline=timeline, title_layers=titles,
                         caption_layer=caption)
    assert windows == ([(0, first), (last, 12)] if first else [])
    monkeypatch.setattr(stage, "prepare_edge_layers", lambda *args, **kwargs: [])
    baseline = tmp_path / "baseline.mp4"
    stage.render_picture(root, baseline, work_dir=tmp_path / "baseline", grade=GradeSpec("film"),
                         grade_style={}, timeline=timeline, title_layers=titles,
                         caption_layer=caption)
    for frame in range(first, last):
        np.testing.assert_array_equal(synth.picture_pixels(out, frame, box=(24, 16, 72, 32)),
                                      synth.picture_pixels(baseline, frame, box=(24, 16, 72, 32)))


@pytest.mark.parametrize("width,height,keep_px", [(96, 64, 2), (540, 960, 3), (1080, 1920, 6)])
def test_background_matte_preserves_outer_alpha_and_firms_only_inside(
    tmp_path: Path, width: int, height: int, keep_px: int,
) -> None:
    x = np.arange(width)
    mask = np.tile(np.clip((x - 20) * 16, 0, 255).astype(np.uint8), (height, 1))
    mask[:, 20] = 5
    mask[24:36, 65:77] = 128
    matte = write_planes(tmp_path / "matte.mkv", [mask], "gray")
    filters = [edge_colour.background_matte_filter(width, height),
               "lut=c0='clip((val-16)*255/80,0,255)',gblur=sigma=1"]
    results = []
    for chain in filters:
        raw = ffmpeg.run([
            "ffmpeg", "-v", "error", "-i", str(matte), "-filter_complex",
            f"[0:v]{chain}[alpha]", "-map", "[alpha]", "-pix_fmt", "gray",
            "-f", "rawvideo", "pipe:1",
        ], timeout_s=30, binary=True).stdout
        results.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width))
    actual, firm = results
    keep = x <= 20 + keep_px
    assert np.array_equal(actual[:, keep], mask[:, keep])
    assert np.array_equal(actual[:, ~keep], firm[:, ~keep])
    assert np.all(actual[26:34, 67:75] == 255)


def test_background_composite_keeps_soft_rim_and_opaque_interior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    (root / "cut").mkdir(parents=True)
    synth.picture_source(root / "cut/cut.mp4", "color=gray:s=96x64:r=30", 3)
    mask = np.tile(np.clip((np.arange(96) - 30) * 64, 0, 255).astype(np.uint8), (64, 1))
    mask[24:36, 65:77] = 128
    matte = write_planes(root / "matte.mkv", [mask] * 3, "gray")
    layer = synth.picture_source(root / "blue.mov", "color=blue:s=96x64:r=30", 3)
    blank = synth.picture_source(root / "blank.mov", "color=black@0:s=2x2:r=30,format=rgba",
                                 3, alpha=True)
    monkeypatch.setattr(stage, "WIDTH", 96)
    monkeypatch.setattr(stage, "HEIGHT", 64)
    monkeypatch.setattr(stage, "ensure_cut_matte", lambda *args, **kwargs: matte)
    monkeypatch.setattr(stage, "_zoom_inputs", lambda *args: [])
    monkeypatch.setattr(stage, "_insert_inputs", lambda *args: {
        "blue": stage._Source(layer, ffmpeg.probe(layer), None)})
    item = ResolvedElement("blue", "background", "test", 0, 3, "behind", {
        "box_px": [0, 0, 96, 64], "media": {},
    })
    caption = {"path": blank.name, "width": 2, "height": 2, "x": 0, "y": 0,
               "start_frame": 0, "frames": 3}
    out = tmp_path / "out.mp4"
    stage.render_picture(root, out, work_dir=tmp_path / "work",
                         timeline=ResolvedTimeline((item,), (), (), 3, 0),
                         title_layers=[], caption_layer=caption)
    pixels = synth.picture_pixels(out, 1).astype(float)
    raw = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(out), "-pix_fmt", "gray",
                      "-f", "rawvideo", "pipe:1"], timeout_s=30, binary=True).stdout
    luma = np.frombuffer(raw, dtype=np.uint8).reshape(3, 64, 96)[1].astype(float)
    background_y = luma[16:48, 8:16].mean()
    person_y = luma[16:48, 82:90].mean()
    opacity = (luma[16:48, 32].mean() - background_y) / (person_y - background_y)
    assert abs(opacity - mask[32, 32] / 255) < 0.05
    background = pixels[16:48, 8:16].mean(axis=(0, 1))
    person = pixels[16:48, 82:90].mean(axis=(0, 1))
    rim = pixels[16:48, 32].mean(axis=0)
    np.testing.assert_allclose(pixels[26:34, 67:75].mean(axis=(0, 1)), person, atol=5)
    assert np.linalg.norm(rim - background) > 20
    assert np.linalg.norm(rim - person) > 20


def test_colour_cleanup_removes_tint_and_leaves_interior_and_hard_pixels() -> None:
    picture, mask, foreground = mixture()
    result = edge_colour.clean_edge_colours(picture, mask)
    zone = edge_colour.edge_zone(mask)
    a = mask[..., None] / 255
    new = np.array([60, 180, 90])
    target = a * foreground + (1 - a) * new
    assert np.abs((a * result + (1 - a) * new - target)[zone]).max() <= 3
    assert np.abs((a * picture + (1 - a) * new - target)[zone]).mean() > 8
    assert np.array_equal(result[~zone], picture[~zone])
    assert not zone[24:36, 65:77].any()
    assert np.array_equal(result, edge_colour.clean_edge_colours(picture, mask))
    hard = np.where(mask > 127, 255, 0).astype(np.uint8)
    assert np.array_equal(edge_colour.clean_edge_colours(picture, hard), picture)


def test_edge_layer_exact_window_planes_alpha_and_determinism(tmp_path: Path) -> None:
    picture, mask, _ = mixture()
    video = write_planes(tmp_path / "video.mkv", [picture + i for i in range(5)], "yuv444p")
    matte = write_planes(tmp_path / "matte.mkv", [mask] * 5, "gray")
    paths = [edge_colour.write_edge_layer(video, matte, 2, 2, tmp_path / f"edge-{i}.mkv")
             for i in range(2)]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    info = ffmpeg.probe(paths[0])
    assert (info.video_frames, info.pix_fmt) == (2, "yuva444p")
    raw = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(paths[0]), "-pix_fmt", "yuva444p",
                      "-f", "rawvideo", "pipe:1"], timeout_s=30, binary=True).stdout
    planes = np.frombuffer(raw, dtype=np.uint8).reshape(2, 4, 64, 96)
    zone = edge_colour.edge_zone(mask)
    assert np.array_equal(planes[0, 3], zone.astype(np.uint8) * 255)
    expected = edge_colour.clean_edge_colours(picture + 2, mask)
    assert np.array_equal(planes[0, :3, zone], expected[zone])
    assert np.all(planes[:, 0, ~zone] == 16)
    assert np.all(planes[:, 1:3, ~zone] == 128)


def test_edge_layers_skip_windows_without_edge_pixels(tmp_path: Path) -> None:
    picture, mask, _ = mixture()
    hard = np.where(mask > 127, 255, 0).astype(np.uint8)
    video = write_planes(tmp_path / "video.mkv", [picture] * 4, "yuv444p")
    matte = write_planes(tmp_path / "matte.mkv", [hard, hard, mask, hard], "gray")
    work = tmp_path / "layers"
    out = work / "empty.mkv"
    assert edge_colour.write_edge_layer(video, matte, 0, 2, out) is None
    assert not out.exists()
    assert not list(work.iterdir())
    layers = edge_colour.prepare_edge_layers(video, matte, [(0, 2), (1, 4)], work)
    assert layers == [(1, 3, work / "edge-1-3.mkv")]
    assert list(work.iterdir()) == [layers[0][2]]
    assert ffmpeg.probe(layers[0][2]).video_frames == 3


def test_cleanup_accepts_reused_zone_without_recomputing(monkeypatch: pytest.MonkeyPatch) -> None:
    picture, mask, _ = mixture()
    expected = edge_colour.clean_edge_colours(picture, mask)
    zone = edge_colour.edge_zone(mask)
    before = zone.copy()
    monkeypatch.setattr(edge_colour, "edge_zone", lambda alpha: pytest.fail("zone recomputed"))
    assert np.array_equal(edge_colour.clean_edge_colours(picture, mask, zone=zone), expected)
    assert np.array_equal(zone, before)


def test_edge_layer_computes_zone_once_per_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    picture, mask, _ = mixture()
    video = write_planes(tmp_path / "video.mkv", [picture] * 3, "yuv444p")
    matte = write_planes(tmp_path / "matte.mkv", [mask] * 3, "gray")
    edge_zone = edge_colour.edge_zone
    calls = []

    def zone(alpha):
        calls.append(alpha)
        return edge_zone(alpha)

    monkeypatch.setattr(edge_colour, "edge_zone", zone)
    edge_colour.write_edge_layer(video, matte, 1, 2, tmp_path / "edge.mkv")
    assert len(calls) == 2


@pytest.mark.parametrize("short", ["video", "matte"])
def test_wrong_decode_count_is_clear_and_not_published(tmp_path: Path, short: str) -> None:
    picture, mask, _ = mixture()
    video = write_planes(tmp_path / "video.mkv", [picture] * (2 if short == "video" else 4),
                         "yuv444p")
    matte = write_planes(tmp_path / "matte.mkv", [mask] * (2 if short == "matte" else 4), "gray")
    out = tmp_path / "edge.mkv"
    with pytest.raises(ProcessingError, match="frame"):
        edge_colour.write_edge_layer(video, matte, 1, 3, out)
    assert not out.exists()


def test_edge_layer_preserves_picture_colour_properties(tmp_path: Path) -> None:
    picture, mask, _ = mixture()
    raw_video = write_planes(tmp_path / "raw.mkv", [picture], "yuv444p")
    video = tmp_path / "video.mkv"
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(raw_video), "-vf",
                "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
                "-c:v", "ffv1",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-color_range", "tv", str(video)], timeout_s=30)
    matte = write_planes(tmp_path / "matte.mkv", [mask], "gray")
    layer = edge_colour.write_edge_layer(video, matte, 0, 1, tmp_path / "edge.mkv")
    info = ffmpeg.probe(layer)
    assert (info.color_primaries, info.color_transfer, info.color_space, info.color_range) == (
        "bt709", "bt709", "bt709", "tv")


@pytest.mark.parametrize("kind", ["background", "graphic", "both"])
@pytest.mark.parametrize("hard", [False, True])
def test_render_uses_edge_layers_and_hard_mattes_keep_pixels(
    tmp_path: Path, monkeypatch, kind: str, hard: bool,
) -> None:
    picture, mask, _ = mixture()
    if hard:
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    root = tmp_path / "project"
    (root / "cut").mkdir(parents=True)
    source = write_planes(root / "cut/cut.mkv", [picture] * 4, "yuv444p")
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-c:v", "libx264",
                "-crf", "0", "-preset", "ultrafast", str(root / "cut/cut.mp4")], timeout_s=30)
    matte = write_planes(root / "matte.mkv", [mask] * 4, "gray")
    (root / "cut/spans.json").write_text(json.dumps({"sources": {"a": "a.mov"},
        "segments": [{"source": "a", "start": 0, "end": "2/15"}]}))
    monkeypatch.setattr(stage, "WIDTH", 96)
    monkeypatch.setattr(stage, "HEIGHT", 64)
    monkeypatch.setattr(stage, "ensure_cut_matte", lambda *args, **kwargs: matte)
    monkeypatch.setattr(stage, "measure_behind_layers", lambda *args, **kwargs: ())
    monkeypatch.setattr(stage, "_insert_inputs", lambda *args: {
        "item": stage._Source(layer, ffmpeg.probe(layer), None)})
    monkeypatch.setattr(stage, "_assert_picture", lambda *args: None)
    layer = synth.picture_source(root / "layer.mov", "color=blue:s=96x64:r=30,format=rgba",
                                 4, alpha=True)
    transparent = synth.picture_source(root / "blank.mov", "color=black@0:s=2x2:r=30,format=rgba",
                                       4, alpha=True)
    item = ResolvedElement("item", "background" if kind == "both" else kind,
                          "test", 0, 4, "behind", {"box_px": [0, 0, 96, 64],
                           "media": {}, "motion": "none"})
    items = (item, ResolvedElement("card", "graphic", "test", 0, 4, "behind", {})) \
        if kind == "both" else (item,)
    timeline = ResolvedTimeline(items, (), (), 4, 0)
    caption = {"path": transparent.name, "width": 2, "height": 2, "x": 0, "y": 0,
               "start_frame": 0, "frames": 4}
    titles = [{"id": "card" if kind == "both" else "item", "kind": "graphic",
               "layer": "behind", "path": layer.name,
               "width": 96, "height": 64, "x": 0, "y": 0, "start_frame": 0, "frames": 4}] \
        if kind in {"graphic", "both"} else []
    outputs = []
    for i in range(3):
        if i == 2:
            monkeypatch.setattr(stage, "prepare_edge_layers", lambda *args, **kwargs: [])
        out = tmp_path / f"out-{i}.mp4"
        result = stage.render_picture(root, out, work_dir=tmp_path / f"work-{i}",
            timeline=timeline, title_layers=titles, caption_layer=caption, frames=(1, 2))
        outputs.append(out)
        if i < 2:
            assert any("edge" in " ".join(command) for command in result.commands)
            edge = list((tmp_path / f"work-{i}").glob("*.mkv"))
            if hard:
                assert not edge
            else:
                assert edge and ffmpeg.probe(edge[0]).video_frames == 2
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    a, b = [synth.picture_pixels(out, 0) for out in (outputs[0], outputs[2])]
    if hard:
        assert np.array_equal(a, b)
    else:
        assert np.abs(a.astype(float) - b).mean() > 0.5
