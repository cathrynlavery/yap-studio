"""Synthetic contracts for the project camera caption look."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_camera3d import _silhouette, _words

from yap_studio.render import camera3d
from yap_studio.style import resolve


def test_camera_look_defaults_off() -> None:
    presets = Path(camera3d.__file__).parents[1] / "style/presets"
    for preset in sorted(presets.glob("*.json")):
        assert resolve(f"preset:{preset.stem}").data["captions"]["camera_3d"] is False


def look_plan(**options) -> dict:
    return camera3d.plan_camera3d(_words(), 5.3, _silhouette()[1],
                                 onset_lead=.1, max_words=2, max_characters=18, **options)


def _fallback_style(size=83, position="bottom", top=1503):
    style = deepcopy(resolve("preset:clean_minimal"))
    style.data["captions"].update(camera_3d=True, font="Instrument Serif", size_px=size,
                                  position=position, top_px=top)
    return style


@pytest.mark.parametrize("size, expected", [(83, 83), (120, 84), (40, 60)])
@pytest.mark.parametrize("path", ["band", "torso"])
def test_band_groups_use_the_style_size_and_position(size, expected, path, monkeypatch) -> None:
    from yap_studio.render.camera3d_look import check_display_words, plan_layout

    monkeypatch.setattr(camera3d, "_choose_box", lambda *a, **kw: None)
    if path == "band":
        monkeypatch.setattr(camera3d, "_flat_box", lambda *a, **kw: None)
    plan = look_plan(style=_fallback_style(size), hero_free=[(0, 159)])
    assert all(group["flat"] for group in plan["groups"])
    if path == "band":
        assert all(group["bottom_band"] for group in plan["groups"])
    for group in plan["groups"]:
        assert group["size"] == group["ref_size"] == expected
        assert group["box"][1] == 1503
    assert all(box["h"] >= 60 - 1e-6 for caption in plan_layout(plan)["captions"]
               for box in caption["boxes"])
    check_display_words(plan, _words()["words"])


@pytest.mark.parametrize("position, top", [("middle", 960), ("top", 700)])
def test_flat_fallback_uses_the_flat_caption_position(position, top) -> None:
    from yap_studio.render.camera3d_fallback import lowest_box
    from yap_studio.stages.captions import caption_band_geometry

    style = _fallback_style(position=position, top=top)
    token = camera3d._ACTIVE_STYLE.set(style.data)
    try:
        box = lowest_box("Hi")
        geometry = caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
        expected = geometry.y + geometry.padding
        assert box[1] == expected
    finally:
        camera3d._ACTIVE_STYLE.reset(token)


@pytest.mark.parametrize("block", ["down", "up"])
def test_band_search_starts_at_the_caption_position(block, monkeypatch) -> None:
    from yap_studio.render.camera3d_fallback import _band_box

    style = _fallback_style()
    token = camera3d._ACTIVE_STYLE.set(style.data)
    top_token = camera3d._SAFE_TOP.set(280)
    monkeypatch.setattr(camera3d, "_head_at", lambda *a: [0, 0, 0, 0])
    obstacle = [0, 1450, 1080, 1610 if block == "down" else 1920]
    group = {"text": "Hi", "on_frame": 0, "off_frame": 30}
    try:
        box = _band_box(group, [group], [{"start": 0, "end": 1, "head_box": [0]*4}],
                        {}, [{}] * 30, [], [{"start_frame": 0, "end_frame": 30,
                                             "box": obstacle}])
        assert box[3] - box[1] == 83
        if block == "down":
            assert 1634 <= box[1] <= 1646
        else:
            assert 1331 <= box[1] <= 1343
    finally:
        camera3d._ACTIVE_STYLE.reset(token)
        camera3d._SAFE_TOP.reset(top_token)


def test_flat_torso_uses_the_style_size() -> None:
    token = camera3d._ACTIVE_STYLE.set(_fallback_style().data)
    try:
        box = camera3d._flat_box("Hi", _silhouette()[1], [], 0, 30, look=True)
        assert box is not None and box[3] - box[1] == 83
        assert box[1] == 1503
    finally:
        camera3d._ACTIVE_STYLE.reset(token)


@pytest.mark.parametrize("longer", [False, True])
def test_long_fallback_fits_width_without_small_ink(longer, monkeypatch) -> None:
    import math

    from yap_studio.render.camera3d_look import check_display_words, plan_layout

    style = _fallback_style()
    token = camera3d._ACTIVE_STYLE.set(style.data)
    try:
        count = math.ceil(960 / (70 * camera3d._advance("W", "caption")))
    finally:
        camera3d._ACTIVE_STYLE.reset(token)
    text = "W" * count * (3 if longer else 1)
    words = {"time_base": "output", "words": [
        {"id": "w0", "text": text, "start": .3, "end": .6}]}
    monkeypatch.setattr(camera3d, "_choose_box", lambda *a, **kw: None)
    monkeypatch.setattr(camera3d, "_flat_box", lambda *a, **kw: None)
    plan = camera3d.plan_camera3d(words, 1, _silhouette()[1], style=style, onset_lead=.09,
                                 hero_free=[(0, 30)])
    group = plan["groups"][0]
    assert 60 <= group["size"] < 83
    assert group["box"][2] - group["box"][0] <= 960
    if longer:
        assert group["size"] == 60 and group["tracking"] < 0
    assert all(box["h"] >= 60 - 1e-6 for c in plan_layout(plan)["captions"] for box in c["boxes"])
    check_display_words(plan, words["words"])


def test_depth_that_would_paint_at_fifty_uses_flat_style_captions() -> None:
    from yap_studio.render.camera3d_fallback import recover_groups

    group = {"text": "Hi", "on_frame": 0, "off_frame": 10, "sentence": 0,
             "word_ids": ["w0"], "flat": False, "box": [300, 1000, 400, 1060],
             "size": 60, "ref_size": 60, "depth": 1500, "ref_x": 300, "ref_y": 1040}
    samples = [{"cz": -300, "px": 0, "py": 0, "tx": 0, "ty": 0}] * 11
    sections = [{**_silhouette()[1], "start": 0, "end": 1}]
    heads = [[0, 0, 0, 0]] * 10
    fallbacks = []
    token = camera3d._ACTIVE_STYLE.set(_fallback_style().data)
    try:
        camera3d._stabilize_groups([group], sections, {}, samples, 10, fallbacks,
                                   [[]] * 10, heads=heads)
        recover_groups([group], sections, {}, samples, fallbacks, [[]] * 10, [], heads)
        assert group["flat"] and group["size"] == 83
        assert group["box"][1] == 1503
    finally:
        camera3d._ACTIVE_STYLE.reset(token)


@pytest.mark.parametrize("on", [0, 2])
def test_far_depth_sixty_stays_in_depth_through_the_opening_push(on) -> None:
    from yap_studio.render.camera3d_clearance import reject_unsafe_groups
    from yap_studio.render.camera3d_fallback import recover_groups
    from yap_studio.render.camera3d_geometry import shutter_poses

    frames = 3210
    samples, _ = camera3d._camera_samples([[{"start": 0, "end": 1.4}]], frames)
    depth, landing = 2200, 39
    ref_size = round(60 * (depth - samples[landing]["cz"]) / depth, 5)
    group = {"text": "Hi", "on_frame": on, "off_frame": 40, "sentence": 0,
             "word_ids": ["w0"], "flat": False, "box": [300, 1000, 400, 1060],
             "size": 60, "ref_size": ref_size, "depth": depth, "ref_x": 300, "ref_y": 1040}
    natural = [ref_size * depth / (depth - sample["cz"]) for sample in samples[on:40]]
    assert 54 <= min(natural) < 60
    sections = [{**_silhouette()[1], "start": 0, "end": frames / 30}]
    heads = [[0, 0, 0, 0]] * frames
    fallbacks = []
    token = camera3d._ACTIVE_STYLE.set(_fallback_style().data)
    try:
        camera3d._stabilize_groups([group], sections, {}, samples, frames, fallbacks,
                                   [[]] * frames, heads=heads)
        reject_unsafe_groups([group], samples, heads, fallbacks, top=280)
        recover_groups([group], sections, {}, samples, fallbacks, [[]] * frames, [], heads)
        assert not group["flat"] and group["size"] == 60
        assert all(pose["box"][3] - pose["box"][1] >= 60 - .001 for pose in group["poses"])
        assert min(pose[2] * ref_size for f in range(on, 40)
                   for pose in shutter_poses(group, samples, f)) >= 54
        assert not fallbacks
    finally:
        camera3d._ACTIVE_STYLE.reset(token)


@pytest.mark.parametrize("flat, size, allowed", [
    (False, 54, True), (False, 53.99, False), (False, 50, False),
    (True, 54, False), (True, 60, True),
])
def test_painted_size_tolerance_keeps_the_flat_floor(flat, size, allowed) -> None:
    from yap_studio.render.camera3d_clearance import reject_unsafe_groups
    from yap_studio.render.camera3d_fallback import recover_groups

    samples = [{"cz": 0, "px": 0, "py": 0, "tx": 0, "ty": 0}] * 11
    base = camera3d._metrics()["base"]["caption"]
    group = {"text": "Hi", "on_frame": 0, "off_frame": 10, "sentence": 0,
             "word_ids": ["w0"], "flat": flat, "box": [300, 1000, 400, 1000 + size],
             "size": 60, "ref_size": 60, "depth": 2200, "ref_x": 300, "ref_y": 1040,
             "poses": [{"frame": f, "box": [300, 1000, 400, 1000 + size],
                        "ref_x": 300, "ref_y": 1000 + base * size, "scale": size / 60}
                       for f in range(10)]}
    heads, fallbacks = [[0, 0, 0, 0]] * 10, []
    token = camera3d._ACTIVE_STYLE.set(_fallback_style().data)
    try:
        reject_unsafe_groups([group], samples, heads, fallbacks, top=280)
        assert bool(group.get("unplaced")) is not allowed
        if not allowed:
            recover_groups([group], [], {}, samples, fallbacks, [[]] * 10, [], heads)
            assert group["flat"] and group["size"] == 83
    finally:
        camera3d._ACTIVE_STYLE.reset(token)


def test_camera_caption_sizes_stay_readable_through_whips() -> None:
    from yap_studio.render.camera3d_geometry import shutter_poses
    from yap_studio.render.camera3d_look import check_display_words, plan_layout

    plan = look_plan(style=_fallback_style())
    for group in plan["groups"]:
        assert group["size"] >= 60
        for frame in range(group["on_frame"], group["off_frame"]):
            assert min(pose[2] * group["ref_size"] for pose in shutter_poses(
                group, plan["camera"]["samples"], frame)) >= (
                    60 if group["flat"] else 54) - 1e-6
    assert all(box["h"] >= 54 - 1e-6 for c in plan_layout(plan)["captions"] for box in c["boxes"])
    check_display_words(plan, _words()["words"])


@pytest.mark.parametrize("on", [False, True])
def test_camera_bleep_uses_retained_display_words_once(on) -> None:
    from yap_studio.render.camera3d_look import display_words, plan_layout

    words = _words()
    words["words"][8]["text"] = "ship"
    words["words"][9]["text"] = "it"
    style = deepcopy(resolve("preset:clean_minimal"))
    style.data["music"].update(bleep_swears=on, bleep_words_add=["beautiful"])
    display = display_words(words["words"], style.data["captions"],
                            word_fixes={"ship it": "shit"}, vocabulary=(), gaps=())
    plan = camera3d.plan_camera3d(
        words, 5.3, _silhouette()[1], style=style, onset_lead=.09,
        display_words=display, word_fixes={"shit": "sugar"})
    group, = [g for g in plan["groups"] if "w8" in g["word_ids"]]
    expected = "s***" if on else "shit"
    assert expected in group["text"] and "sugar" not in group["text"]
    assert "w9" in group["word_ids"]
    assert any(w["text"] == expected for w in group["words"])
    assert all(s["hero"] is None or s["hero"]["text"].lower() != "beautiful"
               for s in plan["sentences"])
    assert [span["word"] for span in plan["bleeps"]] == (["Beautiful", "shit"] if on else [])
    layout = plan_layout(plan)
    assert any(w["text"] == expected for c in layout["captions"] for w in c["words"])


def test_masked_camera_words_match_the_spoken_transcript(tmp_path: Path) -> None:
    from yap_studio.checks.caption_layout import read_layout_record, write_layout_record
    from yap_studio.checks.caption_words import check_caption_words
    from yap_studio.render.camera3d_look import plan_layout
    from yap_studio.transcript import Transcript, Word

    media = tmp_path / "final.mp4"
    media.write_bytes(b"synthetic picture")
    style = deepcopy(resolve("preset:clean_minimal"))
    style.data["music"]["bleep_swears"] = True
    write_layout_record(media, plan_layout(look_plan(style=style)))
    record = read_layout_record(media, width=1080, height=1920)
    heard = Transcript([Word(word["text"], word["start"], word["end"])
                        for word in _words()["words"]])
    assert check_caption_words(record.words, heard) == []


@pytest.mark.parametrize("measured", [True, False])
def test_look_clears_the_verified_hair_region(measured: bool) -> None:
    from dataclasses import replace

    from test_faces_mapping import _cut

    from yap_studio.faces.clearance import Layout, check_layout
    from yap_studio.faces.mapping import map_cut_faces
    from yap_studio.faces.model import Box, FaceBox
    from yap_studio.render.camera3d_look import plan_layout

    silhouette = {**_silhouette()[1], "head_box": [360, 650, 720, 1000]}
    face = Box(420, 620, 240, 300) if measured else Box(360, 650, 360, 350)
    faces = replace(_cut(159), frames=((FaceBox(face, .9, "detected"),),) * 159)
    plan = camera3d.plan_camera3d(
        _words(), 5.3, silhouette, onset_lead=.1, faces=faces if measured else None,
        hero_free=[(0, 159)], obstacles=[{"start_frame": 0, "end_frame": 159,
                                         "box": [0, 0, 1080, 400]}])
    mapped, _ = map_cut_faces(faces, total_frames=159, end_hold_frames=0, zooms=(),
                              inserts=(), video=faces.video,
                              camera=camera3d.face_transforms(plan))
    report = check_layout(mapped, Layout.from_dict(plan_layout(plan)), margin_px=24)
    assert report.status == "pass", report.to_dict()


def test_look_rejects_hidden_heroes_without_using_front_or_below(monkeypatch) -> None:
    monkeypatch.setattr(camera3d, "_glyph_visible_share", lambda *a: 0)
    plan = look_plan()
    assert all(sentence["hero"] is None for sentence in plan["sentences"])
    assert sorted(w for g in plan["groups"] for w in g["word_ids"]) == sorted(
        w["id"] for w in _words()["words"])
    pilot = camera3d.plan_camera3d(_words(), 5.3, _silhouette()[1])
    assert any(s["hero"] and s["hero"]["layer"] == "front" for s in pilot["sentences"])


def test_look_does_not_place_heroes_below_cards() -> None:
    plan = look_plan(obstacles=[{"start_frame": 0, "end_frame": 159,
                                 "box": [0, 0, 1080, 1000]}])
    assert all(sentence["hero"] is None for sentence in plan["sentences"])


def test_look_onset_has_margin_below_one_hundred_milliseconds() -> None:
    words = deepcopy(_words())
    for word in words["words"]:
        word["start"] = round(word["start"] * 15) / 15 + .1
        word["end"] = word["start"] + .12
    plan = camera3d.plan_camera3d(words, 5.3, _silhouette()[1], onset_lead=.1)
    items = [*plan["groups"], *(s["hero"] for s in plan["sentences"] if s["hero"])]
    assert any(s["hero"] for s in plan["sentences"])
    for item in items:
        assert 0 <= item["words"][0]["start"] - item["on_frame"] / 30 <= .09


def test_contrast_samples_sharp_ink_at_the_painted_pose(monkeypatch) -> None:
    from yap_studio import ffmpeg
    from yap_studio.matte import check
    from yap_studio.render.camera3d_look import measure_contrast, plan_layout

    plan = look_plan(hero_free=[(0, 159)])
    group = deepcopy(plan["groups"][0])
    group.update(on_frame=0, off_frame=30, fade_start_frame=24, flat=True,
                 ref_size=44, tracking=0)
    base = plan["font_metrics"]["base"]["caption"]
    group["poses"] = [{"frame": f, "box": [x, 1000, x + 150, 1044],
                       "ref_x": x, "ref_y": 1000 + base * 44, "scale": 1}
                      for f in range(30) for x in [130 if 6 <= f <= 10 else 100 + f * 5]]
    plan["groups"], plan["sentences"] = [group], []
    plan["style"]["data"]["captions"]["text_color"] = "#FFFFFF"
    group["halo"] = {"color": "#101010"}

    def pictures(*args, **kwargs):
        for frame in range(30):
            pixels = np.full((1920, 1080, 4), 80, np.uint8)
            if frame in (6, 8, 10):
                pixels[1000:1044, 130:280, :3] = 16
                pixels[1010:1035, 140:270, :3] = 255
            yield pixels

    monkeypatch.setattr(ffmpeg, "probe", lambda *a: None)
    monkeypatch.setattr(check, "_window_frames", pictures)
    caption = measure_contrast(Path("synthetic.mp4"), plan, plan_layout(plan))["captions"][0]
    assert caption["frame"] == 8
    assert caption["ratio"] > 7


def test_group_limits_and_onsets() -> None:
    plan = look_plan()
    words = {word["id"]: word for word in _words()["words"]}
    assert plan["groups"]
    for group in plan["groups"]:
        assert len(group["word_ids"]) <= 2 and len(group["text"]) <= 18
        lead = words[group["word_ids"][0]]["start"] - group["on_frame"] / 30
        assert 0 < lead <= .1 + 1e-9
    for sentence in plan["sentences"]:
        if hero := sentence["hero"]:
            assert hero["layer"] == "behind"
            assert 0 < hero["words"][0]["start"] - hero["on_frame"] / 30 <= .1 + 1e-9


@pytest.mark.parametrize("head_bottom", [1480, 1780])
def test_crowded_look_keeps_every_word(head_bottom: int) -> None:
    from yap_studio.render.camera3d_look import Camera3DRefusal

    silhouette = {**_silhouette()[1], "head_box": [40, 200, 1040, head_bottom],
                  "shoulder_y": 1800}
    style = resolve("preset:clean_minimal")
    try:
        plan = camera3d.plan_camera3d(_words(), 5.3, silhouette, style=style,
                                     onset_lead=.1, hero_free=[(0, 159)])
    except Camera3DRefusal as error:
        assert error.code == "camera3d_refused"
        assert str(error).startswith(
            "3D camera captions need room around you, but there is none at ")
        assert error.suggested_action == "Frame the shot wider, or pick another caption look."
        return
    assert sorted(w["id"] for g in plan["groups"] for w in g["words"]) == sorted(
        w["id"] for w in _words()["words"])
    assert not any(f["action"] == "unplaced" for f in plan["fallbacks"])
    assert any(f["action"] == "bottom band" for f in plan["fallbacks"])
    for group in plan["groups"]:
        assert group["on_frame"] < group["off_frame"]
        assert len(group["poses"]) == group["off_frame"] - group["on_frame"]
        if group.get("bottom_band"):
            assert group["flat"] and 60 <= group["size"] <= 84
            assert group["depth"] == 0
            assert group["box"][3] <= style.data["internal"]["caption_safe_bottom_px"]
        for pose in group["poses"]:
            head = camera3d._head_at(plan["plate"], plan["camera"]["samples"][pose["frame"]],
                                     silhouette["head_box"])
            assert not camera3d._overlap(pose["box"], head, 24)


def test_layout_head_change_uses_only_current_pose() -> None:
    from yap_studio.faces.clearance import NO_HAIR, Layout, check_clearance
    from yap_studio.faces.model import Box
    from yap_studio.render.camera3d_look import group_screen_box, plan_layout

    heads = ([350, 900, 730, 1300], [350, 200, 730, 600])
    sections = [{**_silhouette()[1], "start": start, "end": end, "head_box": head}
                for start, end, head in [(0, 1, heads[0]), (1, 5.3, heads[1])]]
    plan = camera3d.plan_camera3d(_words(), 5.3, sections, onset_lead=.1,
                                 hero_free=[(0, 159)])
    group = plan["groups"][1]
    assert group["on_frame"] < 30 < group["off_frame"]
    faces = []
    for frame, sample in enumerate(plan["camera"]["samples"][:plan["frames"]]):
        head = camera3d._head_at(plan["plate"], sample, heads[frame >= 30])
        faces.append((Box(head[0], head[1], head[2] - head[0], head[3] - head[1]),))
        if group["on_frame"] <= frame < group["off_frame"]:
            assert not camera3d._overlap(group_screen_box(group, sample, frame), head, 24)
    caption = Layout.from_dict(plan_layout(plan)).captions[1]
    assert not check_clearance(faces, (), (caption,), margin_px=24, hair=NO_HAIR)
    assert len(caption.boxes) > 1


@pytest.mark.parametrize("gap", [(24, 42), (25, 41)])
def test_caption_gaps_clamp_groups_heroes_layout_and_subtitles(gap: tuple, monkeypatch) -> None:
    from yap_studio.render.camera3d_look import display_words, plan_layout
    from yap_studio.stages.captions import _srt_text
    from yap_studio.stages.captions_3d import plan_phrases

    monkeypatch.setattr(camera3d, "_glyph_visible_share", lambda *a: 1)
    words = display_words(_words()["words"], resolve("preset:clean_minimal").data["captions"],
                          word_fixes={}, vocabulary=[], gaps=[(gap[0] / 30, gap[1] / 30)])
    plan = camera3d.plan_camera3d(_words(), 5.3, _silhouette()[1], onset_lead=.1,
                                 display_words=words, caption_gaps=[gap])
    items = [*plan["groups"], *(s["hero"] for s in plan["sentences"] if s["hero"])]
    assert any(item["text"] == "stories," for item in items)
    assert any("advances" in item and item["on_frame"] < gap[0] for item in items)
    for item in items:
        assert item["off_frame"] <= gap[0] or item["on_frame"] >= gap[1]
        for frame in range(*gap):
            step_frame = frame // 2 * 2
            assert not item["on_frame"] <= step_frame < item["off_frame"]
    for caption in plan_layout(plan)["captions"]:
        assert caption["end_frame"] <= gap[0] or caption["start_frame"] >= gap[1]
    phrases = plan_phrases(words, resolve("preset:clean_minimal").data["captions"],
                           frames=plan["frames"], gaps=[gap])
    assert all(p.end <= gap[0] / 30 or p.start >= gap[1] / 30 for p in phrases)
    stories = next(p for p in phrases if "stories," in p.text)
    assert stories.end == gap[0] / 30
    assert "Beautiful stories," in _srt_text(phrases)
    assert [w.id for p in phrases for w in p.words] == [w["id"] for w in words]


def test_bottom_band_ends_only_blocking_caption_groups(monkeypatch) -> None:
    from yap_studio.render.camera3d_fallback import _band_box

    group = {"text": "new", "on_frame": 30, "off_frame": 60}
    baseline = camera3d._metrics()["base"]["caption"] * 1920
    prior = {"on_frame": 0, "off_frame": 60, "fade_start_frame": 60,
             "flat": True, "ref_x": 0, "ref_y": baseline, "ref_size": 1920,
             "box": [0, 0, 1080, 1920],
             "poses": [{"frame": f, "box": [0, 0, 1080, 1920], "ref_x": 0,
                        "ref_y": baseline, "scale": 1} for f in range(60)]}
    monkeypatch.setattr(camera3d, "_head_at", lambda *a: [-100, -100, -50, -50])
    sections = [{**_silhouette()[1], "start": 0, "end": 2}]
    box = _band_box(group, [prior, group], sections, {}, [{}] * 60, [[]] * 60, [])
    assert box[3] <= 1740
    assert prior["off_frame"] == 30
    assert prior["fade_start_frame"] == 24 and len(prior["poses"]) == 30


def test_gap_splits_clauses_and_moves_pre_roll_past_the_gap() -> None:
    from yap_studio.render.camera3d_look import display_words

    words = [{"id": f"w{i}", "text": text, "start": start, "end": start + .12}
             for i, (text, start) in enumerate([
                 ("one", .3), ("two", .6), ("three", 1.42), ("four.", 1.7), ("End.", 2.2)])]
    visible = display_words(words, resolve("preset:clean_minimal").data["captions"],
                            word_fixes={}, vocabulary=[], gaps=[(.8, 1.4)])
    plan = camera3d.plan_camera3d({"time_base": "output", "words": words}, 2.6,
                                 _silhouette()[1], display_words=visible, onset_lead=.1,
                                 caption_gaps=[(24, 42)], hero_free=[(0, 78)])
    after = next(g for g in plan["groups"] if "w2" in g["word_ids"])
    assert after["on_frame"] == 42
    assert not {"w0", "w1"} & set(after["word_ids"])
    assert sorted(w["id"] for g in plan["groups"] for w in g["words"]) == [
        w["id"] for w in visible]


def test_gap_resumed_hero_fade_starts_after_its_onset() -> None:
    from yap_studio.render.camera3d_look import clamp_gaps

    hero = {"on_frame": 40, "fade_start_frame": 41, "off_frame": 48}
    clamp_gaps(hero, [(24, 42)])
    assert hero["on_frame"] == 42
    assert 42 <= hero["fade_start_frame"] < hero["off_frame"]


@pytest.mark.parametrize("start", [1.07, 32 / 30])
@pytest.mark.parametrize("heroes", [False, True])
def test_short_gap_keeps_the_following_word_window_positive(start, heroes, monkeypatch) -> None:
    from yap_studio.render.camera3d_look import plan_layout
    from yap_studio.stages.captions_3d import plan_phrases

    monkeypatch.setattr(camera3d, "_glyph_visible_share", lambda *a: 1)
    words = {"time_base": "output", "words": [
        {"id": "w0", "text": "Beautiful.", "start": start, "end": start + .2}]}
    plan = camera3d.plan_camera3d(words, 2, _silhouette()[1], onset_lead=.09,
                                 caption_gaps=[(31, 32)], hero_free=[] if heroes else [(0, 60)])
    item = plan["sentences"][0]["hero"] if heroes else plan["groups"][0]
    assert item["on_frame"] == 32 < item["off_frame"] <= 60
    captions = plan_layout(plan)["captions"]
    assert len(captions) == 1 and captions[0]["start_frame"] == 32
    assert captions[0]["end_frame"] > 32 and captions[0]["boxes"]
    phrase = plan_phrases(words["words"], resolve("preset:clean_minimal").data["captions"],
                          frames=60, gaps=[(31, 32)])[0]
    assert phrase.start == round(start, 3) and phrase.start >= 32 / 30


@pytest.mark.parametrize("words,keep", [([], False), ([{"id": "w", "start": .02}], True)])
def test_gap_only_drops_an_empty_wordless_window(words, keep) -> None:
    from yap_studio.render.camera3d_look import clamp_gaps

    item = {"on_frame": 0, "off_frame": 4, "fade_start_frame": 2, "words": words}
    assert clamp_gaps(item, [(1, 2)]) is keep
    if keep:
        assert item["on_frame"] == 2 < item["off_frame"]


@pytest.mark.parametrize("frames,gap_end,start,kept", [
    (90, 89, 2.97, False), (90, 88, 2.95, True),
    (91, 89, 3.0, True), (90, 90, 3.0, False),
])
def test_gap_at_cut_end_keeps_windows_inside_the_picture(frames, gap_end, start, kept) -> None:
    from yap_studio.render.camera3d_look import plan_layout

    words = {"time_base": "output", "words": [
        {"id": "early", "text": "Hi.", "start": .3, "end": .5},
        {"id": "late", "text": "Late.", "start": start, "end": start + .1}]}
    plan = camera3d.plan_camera3d(words, frames / 30, _silhouette()[1], onset_lead=.09,
                                 caption_gaps=[(24, gap_end)])
    items = [*plan["groups"], *(s["hero"] for s in plan["sentences"] if s["hero"])]
    late = [item for item in items if any(w["id"] == "late" for w in item["words"])]
    assert bool(late) is kept
    for item in items:
        assert 0 <= item["on_frame"] <= (frames - 1) // 2 * 2
        assert item["on_frame"] < item["off_frame"] <= frames
        assert item["off_frame"] <= 24 or item["on_frame"] >= gap_end
    layout = plan_layout(plan)
    assert len(layout["captions"]) == len(items)
    if not kept:
        assert any(item["action"] == "drop" and item["word_ids"] == ["late"]
                   and "cut end" in item["reason"] for item in plan["fallbacks"])


@pytest.mark.parametrize("frames,gap_end,kept", [(90, 89, False), (91, 89, True)])
def test_resumed_hero_window_is_bounded_by_the_cut(frames, gap_end, kept) -> None:
    from yap_studio.render.camera3d_look import clamp_gaps

    hero = {"on_frame": 86, "off_frame": 90, "fade_start_frame": 88,
            "words": [{"id": "late", "start": 2.97}]}
    assert clamp_gaps(hero, [(24, gap_end)], frames=frames) is kept
    if kept:
        assert gap_end <= hero["on_frame"] < hero["off_frame"] <= frames


@pytest.mark.parametrize("blocker", ["head", "hero", "obstacle"])
def test_bottom_band_refuses_when_every_slot_is_blocked(monkeypatch, blocker: str) -> None:
    from yap_studio.render.camera3d_fallback import _band_box
    from yap_studio.render.camera3d_look import Camera3DRefusal

    full = [0, 0, 1080, 1920]
    group = {"text": "new", "on_frame": 30, "off_frame": 60}
    monkeypatch.setattr(camera3d, "_head_at", lambda *a: (
        full if blocker == "head" else [-100, -100, -50, -50]))
    sections = [{**_silhouette()[1], "start": 0, "end": 2}]
    heroes = [[full] if blocker == "hero" else [] for _ in range(60)]
    obstacles = ([{"start_frame": 30, "end_frame": 60, "box": full}]
                 if blocker == "obstacle" else [])
    with pytest.raises(Camera3DRefusal) as error:
        _band_box(group, [group], sections, {}, [{}] * 60, heroes, obstacles)
    assert str(error.value) == "3D camera captions need room around you, but there is none at 0:01."
    assert error.value.suggested_action == "Frame the shot wider, or pick another caption look."


def test_look_asserts_coverage_before_returning(monkeypatch) -> None:
    clauses = camera3d._clauses
    monkeypatch.setattr(camera3d, "_clauses", lambda words, *a: clauses(
        [w for w in words if w["id"] != "w2"], *a))
    with pytest.raises(ValueError, match="w2"):
        look_plan()


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_display_word_coverage_names_missing_or_duplicate_words(change: str) -> None:
    from yap_studio.render.camera3d_look import check_display_words

    plan = look_plan()
    word = plan["groups"][0]["words"][0]
    if change == "missing":
        plan["groups"][0]["words"].remove(word)
    else:
        plan["groups"][0]["words"].append(word)
    with pytest.raises(ValueError, match=word["id"]):
        check_display_words(plan, _words()["words"])


def test_layout_covers_group_entry_rise() -> None:
    from yap_studio.render.camera3d_look import plan_layout

    plan = look_plan()
    group = next(g for g in plan["groups"] if not g["flat"])
    group.update(on_frame=0, off_frame=10)
    group["ref_size"] = 60
    base = plan["font_metrics"]["base"]["caption"]
    group["poses"] = [{"frame": frame, "box": [100, 1000, 300, 1060], "scale": 1,
                       "ref_x": 100, "ref_y": 1000 + base * 60}
                      for frame in range(10)]
    plan["groups"] = [group]
    plan["sentences"] = []
    plan["camera"]["samples"] = [{"cz": 0, "px": 0, "py": 0, "tx": 0, "ty": 0}]
    plan["camera"]["samples"] *= plan["frames"]
    box = plan_layout(plan)["captions"][0]["boxes"][0]
    assert box["y"] + box["h"] >= 1090
    assert "plan.look_version ? 0 : 30" not in camera3d._PAINT_SCRIPT


def test_close_sentence_windows_use_the_same_onset_grid(monkeypatch) -> None:
    monkeypatch.setattr(camera3d, "_glyph_visible_share", lambda *a: 1)
    words = {"time_base": "output", "words": [
        {"id": "w0", "text": "Beautiful.", "start": .3, "end": 1.02},
        {"id": "w1", "text": "Stories.", "start": 1.041, "end": 1.8},
    ]}
    plan = camera3d.plan_camera3d(words, 2, _silhouette()[1], onset_lead=.1)
    first, second = [sentence["hero"] for sentence in plan["sentences"]]
    assert first and second and first["off_frame"] <= second["on_frame"]


@pytest.mark.parametrize("box", [[0, 0, 1080, 900], [350, 1000, 750, 1200]])
def test_camera_obstacles(box: list) -> None:
    plan = look_plan(obstacles=[{"start_frame": 0, "end_frame": 159, "box": box}])
    items = [*plan["groups"], *(s["hero"] for s in plan["sentences"] if s["hero"])]
    assert items
    for item in items:
        for frame in range(item["on_frame"], min(plan["frames"], item["off_frame"])):
            drawn = (camera3d._hero_screen_box(item, plan["camera"]["samples"][frame], frame)
                     if "advances" in item else item["poses"][frame - item["on_frame"]]["box"])
            assert not camera3d._overlap(drawn, box)


def test_painted_text_bounds_clear_other_visible_words() -> None:
    from yap_studio.render.camera3d_geometry import painted_box

    style = deepcopy(resolve("preset:clean_minimal"))
    style.data["captions"]["font"] = "DM Sans"
    silhouette = {**_silhouette()[1], "head_box": [350, 695, 730, 1106]}
    silhouette["rows"] = [[y + 440, left, right] for y, left, right in silhouette["rows"]
                          if y + 440 < 1920]
    plan = camera3d.plan_camera3d(_words(), 5.3, silhouette, style=style, onset_lead=.09)
    items = [*plan["groups"], *(s["hero"] for s in plan["sentences"] if s["hero"])]
    for frame in range(plan["frames"]):
        boxes = []
        for item in items:
            if item["on_frame"] <= frame < item["off_frame"]:
                box = painted_box(item, plan["camera"]["samples"], frame,
                                  hero="advances" in item, metrics=plan["font_metrics"])
                assert all(not camera3d._overlap(box, prior, 24) for prior in boxes), (
                    item["text"], frame)
                boxes.append(box)
    camera3d.check_caption_clearance(plan)


@pytest.mark.parametrize("checker", ["pairs", "plan"])
def test_checks_reject_painted_text_overlap(checker: str) -> None:
    from dataclasses import replace

    from test_faces_mapping import _cut

    from yap_studio.render.camera3d_clearance import check_plan
    from yap_studio.render.camera3d_look import plan_layout

    plan = look_plan(hero_free=[(0, 159)])
    plan["camera"]["samples"] = [{"cz": 0, "px": 0, "py": 0, "tx": 0, "ty": 0}] * 160
    plan["groups"] = deepcopy(plan["groups"][:2])
    plan["sentences"] = []
    base = plan["font_metrics"]["base"]["caption"]
    for index, group in enumerate(plan["groups"]):
        group.update(on_frame=0, off_frame=30, ref_size=60, flat=False)
        group["poses"] = [{"frame": frame, "box": [100, y, 300, y + 60],
                           "ref_x": 100, "ref_y": y + base * 60, "scale": 1}
                          for frame in range(30)
                          for y in [1000 if (frame + index) % 2 else 1200]]
    assert all(not camera3d._overlap(a["box"], b["box"], 24)
               for a, b in zip(*(g["poses"] for g in plan["groups"]), strict=True))
    with pytest.raises(ValueError, match="caption groups overlap"):
        if checker == "pairs":
            camera3d.check_caption_clearance(plan)
        else:
            faces = replace(_cut(159), frames=((),) * 159, gaps=((0, 159),))
            check_plan(plan, plan_layout(plan), faces, 24)


@pytest.mark.parametrize("top", [280, 340])
@pytest.mark.parametrize("low_head", [False, True])
def test_look_painted_words_respect_the_style_safe_top(top: int, low_head: bool) -> None:
    from yap_studio.render.camera3d_geometry import painted_box
    from yap_studio.render.camera3d_look import check_display_words, plan_layout

    style = deepcopy(resolve("preset:clean_minimal"))
    style.data["internal"]["key_text_top_px"] = top
    silhouette = _silhouette()[1]
    if low_head:
        silhouette["head_box"] = [350, 695, 730, 1106]
        silhouette["rows"] = [[y + 440, left, right] for y, left, right in silhouette["rows"]
                              if y + 440 < 1920]
    plan = camera3d.plan_camera3d(_words(), 5.3, silhouette, style=style, onset_lead=.09)
    layout = plan_layout(plan)
    assert all(box["y"] >= top for caption in layout["captions"] for box in caption["boxes"])
    assert any(caption["id"].startswith("hero-") for caption in layout["captions"])
    items = [*plan["groups"], *(s["hero"] for s in plan["sentences"] if s["hero"])]
    whip_frames = {f for whip in plan["camera"]["whips"]
                   for f in range(whip["start_frame"], whip["end_frame"])}
    assert any(item["on_frame"] <= f < item["off_frame"] for f in whip_frames for item in items)
    for item in items:
        for frame in range(item["on_frame"], item["off_frame"]):
            assert painted_box(item, plan["camera"]["samples"], frame,
                               hero="advances" in item, metrics=plan["font_metrics"])[1] >= top
    check_display_words(plan, _words()["words"])


def test_plan_check_rejects_layout_above_the_safe_text_top() -> None:
    from dataclasses import replace

    from test_faces_mapping import _cut

    from yap_studio.errors import YapError
    from yap_studio.render.camera3d_clearance import check_plan
    from yap_studio.render.camera3d_look import plan_layout

    plan = look_plan(hero_free=[(0, 159)])
    layout = plan_layout(plan)
    layout["captions"][0]["boxes"][0]["y"] = 270
    faces = replace(_cut(159), frames=((),) * 159, gaps=((0, 159),))
    with pytest.raises(YapError, match="safe text top"):
        check_plan(plan, layout, faces, 24)


def test_painted_caption_bounds_clear_a_boxed_overlay() -> None:
    from dataclasses import replace

    from test_faces_mapping import _cut

    from yap_studio.render.camera3d_clearance import check_plan
    from yap_studio.render.camera3d_geometry import painted_box
    from yap_studio.render.camera3d_look import plan_layout, planner_windows

    style = resolve("preset:clean_minimal")
    overlay = SimpleNamespace(kind="overlay", start_frame=0, end_frame=159,
                               payload={"box_px": [300, 950, 480, 150]})
    obstacles, _ = planner_windows([overlay], style.data)
    plan = camera3d.plan_camera3d(_words(), 5.3, _silhouette()[1], onset_lead=.09,
                                 style=style, obstacles=obstacles)
    for group in plan["groups"]:
        for frame in range(group["on_frame"], group["off_frame"]):
            assert not camera3d._overlap(painted_box(
                group, plan["camera"]["samples"], frame, metrics=plan["font_metrics"]),
                obstacles[0]["box"]), (group["text"], frame)
    faces = replace(_cut(159), frames=((),) * 159, gaps=((0, 159),))
    check_plan(plan, plan_layout(plan), faces, 24, obstacles=obstacles)


@pytest.mark.parametrize("active", [True, False])
def test_plan_check_rejects_only_active_obstacles(active) -> None:
    from dataclasses import replace

    from test_faces_mapping import _cut

    from yap_studio.errors import YapError
    from yap_studio.render.camera3d_clearance import check_plan
    from yap_studio.render.camera3d_look import plan_layout

    plan = look_plan(hero_free=[(0, 159)])
    layout = plan_layout(plan)
    caption = layout["captions"][0]
    box = caption["boxes"][0]
    first = box["start_frame"] if active else 0
    obstacles = [{"start_frame": first, "end_frame": first + 1,
                  "box": [box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]]}]
    faces = replace(_cut(159), frames=((),) * 159, gaps=((0, 159),))
    if active:
        with pytest.raises(YapError, match="before rendering") as error:
            check_plan(plan, layout, faces, 24, obstacles=obstacles)
        assert error.value.code == "camera3d_caption_clearance"
    else:
        check_plan(plan, layout, faces, 24, obstacles=obstacles)


def test_requested_card_title_and_overlay_windows() -> None:
    from yap_studio.render.camera3d_look import planner_windows

    style = resolve("preset:clean_minimal").data
    gap = style["graphics"]["caption_gap_px"]
    items = [SimpleNamespace(kind=kind, layer="front", start_frame=10, end_frame=50,
                             payload=payload) for kind, payload in [
        ("graphic", {"box": {"x": 60, "y": 200, "width": 960, "height": 500}}),
        ("title", {}), ("overlay", {"box_px": [50, 800, 300, 200]}),
        ("insert", {}), ("transition", {}),
    ]]
    obstacles, hero_free = planner_windows(items, style)
    assert [item["box"] for item in obstacles] == [
        [0, 0, 1080, 700 + gap], [0, 0, 1080, style["internal"]["key_text_bottom_px"] + gap],
        [50 - gap, 800 - gap, 350 + gap, 1000 + gap],
    ]
    assert hero_free == [(10, 50), (10, 50)]
    assert all((item["start_frame"], item["end_frame"]) == (10, 50) for item in obstacles)


@pytest.mark.parametrize("look", ["paper_title", "kinetic"])
def test_title_obstacles_follow_the_planned_title_box(look: str) -> None:
    from dataclasses import replace

    from test_faces_mapping import _cut

    from yap_studio.render.camera3d_geometry import painted_box
    from yap_studio.render.camera3d_look import planner_windows
    from yap_studio.stages.captions import title_layer_plan

    style = resolve("preset:clean_minimal")
    style.data["hook"].update(style=look, position="center")
    lines = ["A new story"]
    title = title_layer_plan("", "story", style, lines=lines, frames=60, start_frame=0)
    item = SimpleNamespace(kind="title", layer="front", start_frame=0, end_frame=60,
                           payload={"lines": lines, "accent": "story"})
    faces = replace(_cut(159), frames=((),) * 159, gaps=((0, 159),))
    obstacles, _ = planner_windows([item], style.data, faces=faces)
    gap = style.data["graphics"]["caption_gap_px"]
    assert obstacles == [{"start_frame": 0, "end_frame": 60, "box": [
        title.x - gap, title.y - gap,
        title.x + title.width + gap, title.y + title.height + gap]}]
    if look == "paper_title":
        assert title.y + title.height > style.data["internal"]["key_text_bottom_px"]
    plan = look_plan(style=style, obstacles=obstacles, hero_free=[(0, 159)])
    for group in plan["groups"]:
        for frame in range(group["on_frame"], min(60, group["off_frame"])):
            assert not camera3d._overlap(painted_box(
                group, plan["camera"]["samples"], frame, metrics=plan["font_metrics"]),
                obstacles[0]["box"])


@pytest.mark.parametrize("placement", ["measured", "missing", "unplaceable"])
def test_title_obstacles_follow_face_adjusted_placement(monkeypatch, placement: str) -> None:
    from dataclasses import replace

    from test_faces_mapping import _cut

    from yap_studio.errors import ValidationError
    from yap_studio.faces.mapping import map_cut_faces
    from yap_studio.faces.model import Box, FaceBox
    from yap_studio.render.camera3d_geometry import painted_box
    from yap_studio.render.camera3d_look import planner_windows
    from yap_studio.stages import graphics
    from yap_studio.stages.captions import title_layer_plan

    style = resolve("preset:clean_minimal")
    style.data["hook"].update(style="quiet_serif", position="center")
    cut = replace(_cut(159), frames=((FaceBox(Box(400, 900, 240, 300), .9, "detected"),),) * 159)
    samples, _ = camera3d._camera_samples(camera3d._sentences(_words()["words"]), 159)
    camera = camera3d.face_transforms({"frames": 159, "camera": {"samples": samples}})
    faces, _ = map_cut_faces(cut, total_frames=159, end_hold_frames=0, zooms=(), inserts=(),
                              video=replace(cut.video, name="picture.mp4"), camera=camera)
    lines = ["A quiet story"]
    title = title_layer_plan("", "story", style, lines=lines, frames=60)
    moved, _ = graphics.place_title(title, style, faces)
    assert moved.y != title.y
    if placement == "unplaceable":
        def blocked(*args):
            raise ValidationError("title.clearance", "no title placement")
        monkeypatch.setattr(graphics, "place_title", blocked)
    item = SimpleNamespace(kind="title", layer="front", start_frame=0, end_frame=60,
                           payload={"lines": lines, "accent": "story"})
    obstacles, _ = planner_windows([item], style.data,
                                   faces=None if placement == "missing" else faces)
    gap = style.data["graphics"]["caption_gap_px"]
    if placement == "measured":
        expected = [moved.x, moved.y, moved.x + moved.width, moved.y + moved.height]
    else:
        expected = [title.x, 0, title.x + title.width,
                    max(title.y + title.height, style.data["internal"]["key_text_bottom_px"])]
    assert obstacles[0]["box"] == [expected[0] - gap, expected[1] - gap,
                                    expected[2] + gap, expected[3] + gap]
    plan = look_plan(style=style, faces=cut, obstacles=obstacles, hero_free=[(0, 159)])
    padded = [moved.x - gap, moved.y - gap,
              moved.x + moved.width + gap, moved.y + moved.height + gap]
    for group in plan["groups"]:
        for frame in range(group["on_frame"], min(60, group["off_frame"])):
            assert not camera3d._overlap(painted_box(
                group, plan["camera"]["samples"], frame, metrics=plan["font_metrics"]), padded)


def test_subtitles_follow_display_words_instead_of_depth_layers(monkeypatch) -> None:
    from yap_studio.captions import CaptionWord, build_phrases
    from yap_studio.stages.captions import _srt_text
    from yap_studio.stages.captions_3d import plan_phrases

    settings = resolve("preset:clean_minimal").data["captions"]
    words = [{"id": f"w{i}", "text": text, "start": .3 + i * .3, "end": .5 + i * .3,
              "source_ids": [f"source-{i}"]}
             for i, text in enumerate("I like stories today. Done.".split())]
    monkeypatch.setattr(camera3d, "_glyph_visible_share", lambda *a: 1)
    plan = camera3d.plan_camera3d({"time_base": "output", "words": words}, 3,
                                 _silhouette()[1], onset_lead=.09)
    assert plan["sentences"][0]["hero"]["text"] == "stories"
    phrases = plan_phrases(words, settings, frames=90)
    expected = build_phrases([CaptionWord(w["id"], w["text"], w["start"], w["end"])
                              for w in words], settings)
    assert [p.to_dict() for p in phrases] == [p.to_dict() for p in expected]
    assert " ".join(p.text for p in phrases) == "I like stories today. Done."
    assert [w.source_ids for p in phrases for w in p.words] == [
        tuple(w["source_ids"]) for w in words]
    assert _srt_text(phrases).index("stories") < _srt_text(phrases).index("today")


@pytest.mark.parametrize("gap", [80, 100])
def test_chapter_text_band_blocks_groups_and_the_window_excludes_heroes(gap) -> None:
    from yap_studio.render.camera3d_geometry import painted_box
    from yap_studio.render.camera3d_look import Camera3DRefusal, planner_windows

    style = deepcopy(resolve("preset:clean_minimal"))
    style.data["graphics"]["caption_gap_px"] = gap
    chapter = SimpleNamespace(kind="chapter", layer="front", start_frame=0, end_frame=159,
                              payload={"title": "A new chapter"})
    obstacles, hero_free = planner_windows([chapter], style.data)
    gap = style.data["graphics"]["caption_gap_px"]
    assert obstacles == [{"start_frame": 0, "end_frame": 159,
                           "box": [0, 240 - gap, 1080, 1560 + gap]}]
    assert hero_free == [(0, 159)]
    if gap == 100:
        with pytest.raises(Camera3DRefusal, match="no room|there is none"):
            look_plan(style=style, obstacles=obstacles, hero_free=hero_free)
        return
    plan = look_plan(style=style, obstacles=obstacles, hero_free=hero_free)
    assert all(sentence["hero"] is None for sentence in plan["sentences"])
    assert sorted(word for group in plan["groups"] for word in group["word_ids"]) == sorted(
        word["id"] for word in _words()["words"])
    for group in plan["groups"]:
        for frame in range(group["on_frame"], group["off_frame"]):
            box = painted_box(group, plan["camera"]["samples"], frame,
                               metrics=plan["font_metrics"])
            assert not camera3d._overlap(box, obstacles[0]["box"])


@pytest.mark.parametrize("kind,payload", [
    ("title", {"style": "kinetic"}), ("graphic", {"card": "gravity_drop"}),
])
def test_kinetic_titles_and_gravity_cards_keep_caption_headroom(kind, payload) -> None:
    from yap_studio.render.camera3d_look import planner_windows

    style = resolve("preset:clean_minimal").data
    item = SimpleNamespace(kind=kind, layer="front", start_frame=10, end_frame=50,
                           payload=payload)
    obstacles, _ = planner_windows([item], style)
    assert obstacles == [{"start_frame": 10, "end_frame": 50, "box": [
        0, 0, 1080, style["internal"]["key_text_bottom_px"] + style["graphics"]["caption_gap_px"]]}]


def test_hero_free_windows_and_skip_heroes() -> None:
    original = look_plan()
    indexes = {i for i, sentence in enumerate(original["sentences"]) if sentence["hero"]}
    assert indexes
    forced = look_plan(skip_heroes=indexes)
    assert all(forced["sentences"][i]["hero"] is None for i in indexes)
    assert sorted(w for g in forced["groups"] for w in g["word_ids"]) == sorted(
        w["id"] for w in _words()["words"])
    blocked = look_plan(hero_free=[(0, 159)])
    assert all(sentence["hero"] is None for sentence in blocked["sentences"])
    assert sorted(word for group in blocked["groups"] for word in group["word_ids"]) == sorted(
        word["id"] for word in _words()["words"])


def test_camera_face_transform_matches_painter() -> None:
    plan = look_plan()
    for frame, transform in enumerate(camera3d.face_transforms(plan)):
        x, y, scale = camera3d._person_transform(plan["camera"]["samples"][frame])
        assert transform == (scale, x, y)


@pytest.mark.parametrize("move", ["punch", "smooth", "step_in"])
def test_camera_joins_are_hard_cuts(tmp_path: Path, move: str) -> None:
    from yap_studio.joins.plan import build_plan
    from yap_studio.joins.render import render_joins
    from yap_studio.stages.timeline import ResolvedTimeline

    timeline = ResolvedTimeline((), (), (), 60, 0, camera_3d=True)
    pieces = [{"take": "t1", "segment_id": "s1", "out_f0": 0, "n_frames": 30,
               "src_f0": 0, "src_f1": 30},
              {"take": "t1", "segment_id": "s1", "out_f0": 30, "n_frames": 30,
               "src_f0": 60, "src_f1": 90}]
    plan = build_plan(tmp_path, timeline, pieces, [], {"jump_cuts": move})
    assert plan.mode == plan.fallback == "hard"
    assert len(plan.joins) == 1 and plan.joins[0].handling == "hard"
    assert render_joins(tmp_path, timeline, tmp_path / "missing.mp4", {"jump_cuts": move}) == (
        [], None)


def test_plan_layout_has_words_windows_and_union_boxes() -> None:
    from yap_studio.render.camera3d_geometry import painted_box
    from yap_studio.render.camera3d_look import plan_layout

    plan = look_plan()
    layout = plan_layout(plan)
    assert layout["measured"] is True
    assert len(layout["captions"]) == len(plan["groups"]) + sum(
        bool(sentence["hero"]) for sentence in plan["sentences"])
    for index, group in enumerate(plan["groups"]):
        caption = next(c for c in layout["captions"] if c["id"] == f"caption-{index}")
        assert caption["start_frame"] == group["on_frame"]
        assert caption["end_frame"] == min(plan["frames"], group["off_frame"])
        assert caption["words"] and caption["boxes"]
        for frame in range(group["on_frame"], group["off_frame"]):
            union, = [box for box in caption["boxes"]
                      if box["start_frame"] <= frame < box["end_frame"]]
            drawn = painted_box(group, plan["camera"]["samples"], frame,
                                  metrics=plan["font_metrics"])
            edges = [union["x"], union["y"], union["x"] + union["w"], union["y"] + union["h"]]
            assert all(abs(a - b) <= 8.02 for a, b in zip(edges, drawn, strict=True))
            assert edges[0] <= drawn[0] and edges[1] <= drawn[1]
            assert edges[2] + 1e-6 >= drawn[2] and edges[3] + 1e-6 >= drawn[3]
    behind = [s["hero"] for s in plan["sentences"] if s["hero"]
              and s["hero"]["layer"] == "behind"]
    assert behind
    assert all(c["face_windows"] == [] for c in layout["captions"] if c["id"].startswith("hero-")
               and any(c["text"] == hero["text"] for hero in behind))


def test_camera_contrast_counts_only_a_drawn_halo() -> None:
    from yap_studio.render.camera3d_look import _halo_contrast

    pixels = np.zeros((1920, 1080, 3), dtype=np.uint8)
    pixels[100:150, 100:200] = [54, 90, 88]
    plain = _halo_contrast(pixels, [100, 100, 200, 150], "#365A58", "#FFFFFF")
    pixels[98:152, 98:202] = 255
    pixels[100:150, 100:200] = [54, 90, 88]
    backed = _halo_contrast(pixels, [100, 100, 200, 150], "#365A58", "#FFFFFF")
    assert plain < 3 and backed > 7


@pytest.mark.parametrize(("kind", "layer", "payload"), [
    ("zoom", "front", {}), ("moment", "front", {}), ("background", "front", {}),
    ("graphic", "behind", {}), ("overlay", "front", {"placement": "tall_behind"}),
    ("graphic", "front", {"template": "big_number"}),
    ("graphic", None, {"template": "big_number"}),
    ("insert", "front", {"media": {"fit": "split"}}),
    ("insert", "front", {"layout": "split"}),
])
def test_incompatible_elements_refuse(kind: str, layer: str, payload: dict) -> None:
    from yap_studio.render.camera3d_look import Camera3DRefusal, refuse_elements

    item = SimpleNamespace(kind=kind, layer=layer, payload=payload, id="agent-one")
    with pytest.raises(Camera3DRefusal) as error:
        refuse_elements([item])
    assert error.value.code == "camera3d_refused"
    assert kind in str(error.value) and "agent-one" in str(error.value)
    assert error.value.suggested_action == (
        "Remove those from the timeline, or pick another caption look.")


@pytest.mark.parametrize("change", [{"width": 720}, {"fps": 24}, {"variable_frame_rate": True}])
def test_bad_cut_refuses(change: dict) -> None:
    from yap_studio.render.camera3d_look import Camera3DRefusal, require_cut

    info = SimpleNamespace(**{"width": 1080, "height": 1920, "fps": 30,
                              "variable_frame_rate": False, **change})
    with pytest.raises(Camera3DRefusal) as error:
        require_cut(info)
    assert error.value.code == "camera3d_refused"
    assert error.value.suggested_action == "Pick another caption look for this video."


def test_missing_person_refusal_names_the_reason() -> None:
    from yap_studio.render.camera3d_look import person_refusal

    error = person_refusal("person matte has no visible person")
    assert error.code == "camera3d_refused"
    assert str(error) == ("3D camera captions need you on camera in every shot: "
                          "person matte has no visible person.")
    assert error.suggested_action == (
        "Pick another caption look, or keep only shots that show you.")


def test_standalone_refuses_before_reading_clip(tmp_path: Path) -> None:
    from dataclasses import replace

    from yap_studio.render.camera3d_look import Camera3DRefusal
    from yap_studio.stages.captions import caption_clip

    style = resolve("preset:clean_minimal")
    data = deepcopy(style.data)
    data["captions"]["camera_3d"] = True
    with pytest.raises(Camera3DRefusal, match="inside a project") as error:
        caption_clip(tmp_path / "missing.mp4", style=replace(style, data=data),
                     out=tmp_path / "out.mp4", words=None, title=None, srt=False, force=False)
    assert error.value.suggested_action == (
        "Run yap-studio run DIR, or pass --style with another look.")


def motion_samples(*, shift: float = 0, flicker: bool = False, flat: bool = False):
    y, x = np.indices((384, 216))
    background = (np.full_like(x, 100) if flat else
                  100 + ((x * 7919 + y * 104729 + x * y * 101) % 89))
    frames, mattes = [], []
    for i in range(10):
        matte = np.zeros((384, 216), dtype=np.uint8)
        matte[120:360, 70 + i:130 + i] = 255
        row = np.roll(background, round(i * shift / 5), axis=1).copy()
        if flicker:
            row = row + (i % 2) * 20
        row[matte > 0] = 220
        frames.append(row.astype(np.uint8))
        mattes.append(matte)
    return np.stack(frames), np.stack(mattes)


def test_each_shot_must_have_a_still_background() -> None:
    from yap_studio.render.camera3d_look import Camera3DRefusal, require_still
    from yap_studio.render.camera3d_motion import measure_samples

    still, matte = motion_samples()
    moving, moving_matte = motion_samples(shift=18)
    picture = np.concatenate([np.tile(still[:1], (50, 1, 1)), moving])
    masks = np.concatenate([np.tile(matte[:1], (50, 1, 1)), moving_matte])
    result = measure_samples(picture, masks, shot_starts=[300])
    assert result["shift_p75_px"] < 4
    assert result["moving"] is True
    assert result["shots"][1]["shift_p75_px"] > 15
    with pytest.raises(Camera3DRefusal, match=r"shot at 0:10.*room shifts about"):
        require_still(result)


@pytest.mark.parametrize("kind", ["insert", "transition"])
def test_picture_replacements_hide_heads_in_planning_and_checks(kind: str) -> None:
    from dataclasses import replace

    from test_faces_mapping import _cut

    from yap_studio.faces.model import Box, FaceBox
    from yap_studio.render.camera3d_clearance import check_plan, head_regions
    from yap_studio.render.camera3d_look import picture_windows, plan_layout

    item = SimpleNamespace(kind=kind, start_frame=0, end_frame=159, payload={})
    windows = picture_windows([item])
    faces = replace(_cut(159), frames=((FaceBox(Box(0, 0, 1080, 1920), .9, "detected"),),) * 159)
    silhouette = {**_silhouette()[1], "head_box": [0, 0, 1080, 1920]}
    plan = camera3d.plan_camera3d(_words(), 5.3, silhouette, onset_lead=.09,
                                 faces=faces, hero_free=windows, replacements=windows)
    heads = head_regions(plan["silhouettes"], plan["plate"], plan["camera"]["samples"],
                         159, faces, plan["style"]["data"], replacements=windows)
    assert all(head == [0, 0, 0, 0] for head in heads)
    check_plan(plan, plan_layout(plan), faces, 24, replacements=windows)


def test_picture_replacements_keep_heads_outside_their_windows() -> None:
    from yap_studio.render.camera3d_clearance import head_regions
    from yap_studio.render.camera3d_look import picture_windows

    elements = [SimpleNamespace(kind="insert", start_frame=30, end_frame=60,
                                payload={"media": {"fit": "cover"}}),
                SimpleNamespace(kind="insert", start_frame=60, end_frame=90,
                                payload={"media": {"fit": "split"}})]
    windows = picture_windows(elements)
    assert windows == [(30, 60)]
    plan = look_plan()
    heads = head_regions(plan["silhouettes"], plan["plate"], plan["camera"]["samples"],
                         159, None, plan["style"]["data"], replacements=windows)
    assert all(head == [0, 0, 0, 0] for head in heads[30:60])
    assert all(head[2] > head[0] for head in heads[:30] + heads[60:])


@pytest.mark.parametrize("blocker", ["head", "obstacle"])
def test_bottom_band_searches_between_blocker_edges(blocker: str, monkeypatch) -> None:
    from yap_studio.render.camera3d_fallback import _band_box

    group = {"text": "Hi", "on_frame": 0, "off_frame": 30}
    sides = [[0, 0, 250, 1920], [480, 0, 1080, 1920]]
    monkeypatch.setattr(camera3d, "_head_at", lambda *a: sides[0] if blocker == "head"
                        else [0, 0, 0, 0])
    obstacles = [{"start_frame": 0, "end_frame": 30, "box": box}
                 for box in sides[1 if blocker == "head" else 0:]]
    box = _band_box(group, [group], [{"start": 0, "end": 1, "head_box": sides[0]}],
                    {}, [{}] * 30, [], obstacles)
    assert 274 <= box[0] < box[2] <= 456


def test_video_letters_reserve_text_instead_of_the_full_surface() -> None:
    from dataclasses import replace

    from yap_studio import letters
    from yap_studio.render.camera3d_look import planner_windows
    from yap_studio.stages.captions import title_layer_plan

    style = resolve("preset:clean_minimal")
    data = deepcopy(style.data)
    data["hook"]["style"] = "video_letters"
    style = replace(style, data=data)
    item = SimpleNamespace(kind="title", layer="front", start_frame=0, end_frame=48,
                           payload={"lines": ["Hi"]})
    title = title_layer_plan("", "", style, lines=["Hi"], frames=48)
    obstacles, _ = planner_windows([item], data)
    gap = data["graphics"]["caption_gap_px"]
    box = obstacles[0]["box"]
    assert box[1] > 0 and box[3] < 1500
    assert box[1] <= letters.CENTER_Y_PX - title.variables["size_px"] / 2 - gap
    assert box[3] >= letters.CENTER_Y_PX + title.variables["size_px"] / 2 + gap
    plan = look_plan(style=style, obstacles=obstacles, hero_free=[(0, 159)])
    assert plan["groups"] and not any(g.get("unplaced") for g in plan["groups"])


@pytest.mark.parametrize("backdrop", ["auto", "paper", "ink", "dim"])
def test_full_frame_backdrops_keep_hero_words_in_front_captions(backdrop, monkeypatch) -> None:
    from yap_studio.render.camera3d_look import planner_windows

    monkeypatch.setattr(camera3d, "_glyph_visible_share", lambda *a: 1)
    card = SimpleNamespace(kind="graphic", layer="front", start_frame=0, end_frame=159,
                           payload={"template": "reel_feed", "slots": {"backdrop": backdrop}})
    obstacles, hero_free = planner_windows([card], resolve("preset:clean_minimal").data)
    assert hero_free == [(0, 159)]
    plan = look_plan(obstacles=obstacles, hero_free=hero_free)
    assert all(sentence["hero"] is None for sentence in plan["sentences"])
    assert sorted(w for g in plan["groups"] for w in g["word_ids"]) == sorted(
        word["id"] for word in _words()["words"])


def test_transparent_full_frame_cards_do_not_hide_heroes() -> None:
    from yap_studio.render.camera3d_look import planner_windows

    card = SimpleNamespace(kind="graphic", layer="front", start_frame=0, end_frame=159,
                           payload={"template": "playing_grid"})
    _, hero_free = planner_windows([card], resolve("preset:clean_minimal").data)
    assert hero_free == []


@pytest.mark.parametrize("options", [{}, {"flicker": True}, {"flat": True}])
def test_still_background_passes(options: dict) -> None:
    from yap_studio.render.camera3d_motion import measure_samples

    result = measure_samples(*motion_samples(**options), shot_starts=())
    assert not result["moving"]
    if options.get("flat"):
        assert result["status"] == "not measurable"
    else:
        assert result["shift_p75_px"] < 1


def test_translating_background_refuses_with_measured_shift() -> None:
    from yap_studio.render.camera3d_look import Camera3DRefusal, require_still
    from yap_studio.render.camera3d_motion import measure_samples

    result = measure_samples(*motion_samples(shift=18), shot_starts=())
    assert result["moving"] and result["shift_p75_px"] > 4
    with pytest.raises(Camera3DRefusal, match="room shifts about .* px every 0.2 s") as error:
        require_still(result)
    assert error.value.code == "camera3d_refused"
    assert error.value.suggested_action == "Record on a tripod, or pick another caption look."


def test_shot_boundaries_are_not_motion() -> None:
    from yap_studio.render.camera3d_motion import measure_samples

    frames, mattes = motion_samples()
    frames[5:] = np.roll(frames[5:], 20, axis=2)
    assert not measure_samples(frames, mattes, shot_starts=(30,))["moving"]


@pytest.mark.parametrize("empty", [False, True])
def test_person_coverage_checks_the_short_opening_shot(empty, monkeypatch) -> None:
    from yap_studio import ffmpeg
    from yap_studio.render.camera3d_look import Camera3DRefusal
    from yap_studio.render.camera3d_matte import sample_matte

    alpha, _ = _silhouette()
    decoded = []

    def stream(command, **kwargs):
        decoded.append(command)
        for frame in range(0, 60, 6):
            yield (np.zeros_like(alpha) if empty and frame < 6 else alpha).tobytes()

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    if empty:
        with pytest.raises(Camera3DRefusal) as error:
            sample_matte(Path("matte.mkv"), 60, (6,))
        assert str(error.value) == (
            "3D camera captions need you on camera in every shot: no person at 0:00.")
        assert error.value.code == "camera3d_refused"
        assert error.value.suggested_action == (
            "Pick another caption look, or keep only shots that show you.")
    else:
        samples, sections = sample_matte(Path("matte.mkv"), 60, (6,))
        assert samples.shape == (10, 384, 216)
        assert sections[0]["start"] == 0 and sections[-1]["end"] == 2
    assert len(decoded) == 1


def test_short_shot_checks_its_middle_frame(monkeypatch) -> None:
    from yap_studio import ffmpeg
    from yap_studio.render.camera3d_look import Camera3DRefusal
    from yap_studio.render.camera3d_matte import sample_matte

    alpha, _ = _silhouette()
    commands = []

    def stream(command, **kwargs):
        commands.append(command)
        for frame in sorted({*range(0, 60, 6), 7, 8, 9}):
            yield (np.zeros_like(alpha) if frame == 8 else alpha).tobytes()

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    with pytest.raises(Camera3DRefusal, match="no person at 0:00"):
        sample_matte(Path("matte.mkv"), 60, (7, 9))
    assert "eq(n\\,8)" in commands[0][commands[0].index("-vf") + 1]


@pytest.mark.parametrize("frames", [60, 69, 78, 91])
def test_shared_matte_keeps_the_pilot_silhouette_cadence(tmp_path, monkeypatch, frames) -> None:
    from yap_studio import ffmpeg
    from yap_studio.render.camera3d_matte import sample_matte

    matte = tmp_path / "matte.mkv"
    boxes = ",".join(
        f"drawbox=x={360 + index * 60}:y={400 + index * 100}:w=240:h=1100:"
        f"color=white:t=fill:enable='between(n,{index * 30},{index * 30 + 29})'"
        for index in range((frames + 29) // 30))
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=1080x1920:r=30",
                "-vf", boxes, "-frames:v", str(frames), "-c:v", "ffv1", "-pix_fmt", "gray",
                str(matte)], timeout_s=30)
    expected = camera3d.summarize_matte(matte, frames / 30)
    decode = ffmpeg.run_stream
    calls = []

    def stream(command, **kwargs):
        calls.append(command)
        yield from decode(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    samples, sections = sample_matte(matte, frames, (7, 9, 36))
    assert sections == expected
    assert sections[-1]["end"] == max(frames / 30, len(sections))
    assert samples.shape == ((frames + 5) // 6, 384, 216)
    assert len(calls) == 1


@pytest.mark.parametrize("frames", [1, 10, 14])
def test_short_matte_keeps_a_silhouette_for_the_plan(tmp_path: Path, frames: int) -> None:
    from yap_studio import ffmpeg
    from yap_studio.render.camera3d_look import plan_layout
    from yap_studio.render.camera3d_matte import sample_matte

    matte = tmp_path / "short.mkv"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=1080x1920:r=30",
                "-vf", "drawbox=x=390:y=400:w=300:h=1520:color=white:t=fill",
                "-frames:v", str(frames), "-c:v", "ffv1", "-pix_fmt", "gray", str(matte)],
               timeout_s=30)
    samples, sections = sample_matte(matte, frames, ())
    assert samples.shape == ((frames + 5) // 6, 384, 216)
    assert len(sections) == 1 and sections[0]["start"] == 0 and sections[0]["end"] == 1
    words = {"time_base": "output", "words": [
        {"id": "w0", "text": "Hi.", "start": 0, "end": frames / 30}]}
    plan = camera3d.plan_camera3d(words, frames / 30, sections, onset_lead=.09)
    layout = plan_layout(plan)
    assert plan["frames"] == frames and len(plan["groups"]) == len(layout["captions"]) == 1
    caption = layout["captions"][0]
    assert 0 <= caption["start_frame"] < caption["end_frame"] <= frames and caption["boxes"]
    assert plan["groups"][0]["word_ids"] == ["w0"]


def test_motion_reuses_the_decoded_matte(monkeypatch) -> None:
    from yap_studio.render import camera3d_motion as motion

    picture, matte = motion_samples()
    decoded = []

    def samples(source):
        decoded.append(source)
        return picture

    monkeypatch.setattr(motion, "_samples", samples)
    result = motion.measure_camera_motion(Path("cut.mp4"), Path("matte.mkv"), (),
                                          matte_samples=matte)
    assert not result["moving"] and decoded == [Path("cut.mp4")]


def test_display_words_keep_fixes_and_skip_only_gap_words() -> None:
    from yap_studio.render.camera3d_look import display_words

    words = deepcopy(_words()["words"])
    result = display_words(words, resolve("preset:clean_minimal").data["captions"],
                           word_fixes={"beautiful": "Wonderful"}, vocabulary=(),
                           gaps=[(.6, .8)])
    assert result[0]["text"] == "Wonderful"
    assert result[0]["id"] == "w0" and result[0]["start"] == .35
    assert "w1" not in {word["id"] for word in result}
    assert "w2" in {word["id"] for word in result}


def test_multiword_fix_stays_intact_on_screen() -> None:
    from yap_studio.render.camera3d_look import display_words

    words = display_words(_words()["words"], resolve("preset:clean_minimal").data["captions"],
                           word_fixes={"beautiful": "Bright Fold"}, vocabulary=(), gaps=[])
    plan = camera3d.plan_camera3d({"time_base": "output", "words": words}, 5.3,
                                  _silhouette()[1], onset_lead=.1)
    items = [*plan["groups"], *(s["hero"] for s in plan["sentences"] if s["hero"])]
    assert any("Bright Fold" in item["text"] for item in items)
    assert all("BrightFold" not in item["text"] for item in items)


def test_display_words_with_equal_times_keep_their_order() -> None:
    words = [{"id": name, "text": text, "start": .5, "end": .6}
             for name, text in (("z", "we"), ("a", "are"), ("b", "it."))]
    plan = camera3d.plan_camera3d({"time_base": "output", "words": words}, 1.5,
                                  _silhouette()[1], display_words=words, onset_lead=.1)
    assert [word for group in plan["groups"] for word in group["word_ids"]] == ["z", "a", "b"]


@pytest.mark.render
def test_project_camera_render_is_repeatable_and_verifies(tmp_path: Path, monkeypatch) -> None:
    import json
    import os

    import synth
    from test_cut import make_project, time_segment, write_edit, write_json
    from test_matte_project import synthetic_matte

    from yap_studio import ffmpeg, receipt
    from yap_studio.faces import detect
    from yap_studio.matte import project as matte_project
    from yap_studio.project.receipts import StageRun, read_receipt
    from yap_studio.project.status import ProjectContext, cut_inputs, ingest_inputs
    from yap_studio.stages import assemble
    from yap_studio.stages.captions import captions
    from yap_studio.stages.graphics import graphics
    from yap_studio.stages.music import music

    if not os.environ.get("YAP_STUDIO_RENDER_TESTS"):
        pytest.skip("set YAP_STUDIO_RENDER_TESTS=1")
    speech, spoken = synth.speech_source(tmp_path / "speech.mov", ["Beautiful stories."],
                                        picture="color", size="1080x1920")
    source = tmp_path / "source.mov"
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(speech), "-vf",
                "tpad=stop_mode=clone:stop_duration=3,trim=end_frame=90", "-af",
                "apad,atrim=end_sample=144000", "-c:v", "libx264", "-c:a", "pcm_s16le",
                "-ar", "48000", str(source)], timeout_s=120)
    rows = [(word.text, word.start, word.end) for word in spoken]
    root = make_project(tmp_path / "project", [source], word_rows=[rows])
    project = json.loads((root / "project.json").read_bytes())
    project.update(style="style.json", asr={"engine": "none"}, music={"mode": "none"},
                   audio={"mode": "preserve"}, platform="none")
    write_json(root / "project.json", project)
    previous = read_receipt(root / "ingest")
    with StageRun(root, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.finish()
    style = deepcopy(resolve("preset:clean_minimal").data)
    style["captions"].update(camera_3d=True, highlight_style="none",
                              font="Instrument Serif", text_color="#FFFFFF")
    style["hook"]["font"] = "Geist"
    style["brand"] = {"accent_color": "#F34E3F"}
    style["grade"].update(look="film", strength=.4)
    write_json(root / "style.json", style)
    write_edit(root, [time_segment("s1", "t1", 0, 3)], end_hold_s=0)
    staged = root / "picture.mp4"
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-vf",
                "tpad=stop_mode=clone:stop_duration=3,trim=end_frame=90,"
                "drawbox=x=390:y=550:w=300:h=1370:color=0xB030A0:t=fill,"
                "eq=brightness='0.03*sin(2*PI*t)':eval=frame",
                "-frames:v", "90", "-an", "-c:v", "libx264", "-crf", "12",
                "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", str(staged)], timeout_s=120)
    voice = root / "voice.wav"
    ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:a:0", "-af",
                "apad,atrim=end_sample=144000", "-ar", "48000", "-c:a", "pcm_f32le", str(voice)],
               timeout_s=120)
    hashes = {"t1": receipt.sha256_file(root / "ingest/t1/words.json")}
    word_file = root / "words.json"
    write_json(word_file, {"schema": "yap-studio/cut-words@1", "time_base": "output",
                           "words_sha256": hashes, "words": [
                               {"id": f"t1:{i}", "segment": "s1", "text": text,
                                "start": start, "end": end}
                               for i, (text, start, end) in enumerate(rows)]})
    spans = root / "spans.json"
    write_json(spans, {"sources": {"t1": "../takes/t1.mov"}, "fps": "30/1",
                       "ranges": [{"source": "t1", "start": 0, "end": 3}]})
    with StageRun(root, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        for name, file in (("cut.mp4", staged), ("voice.wav", voice),
                           ("cut-words.json", word_file), ("spans.json", spans)):
            run.replace_output(root / "cut" / name, file)
        run.finish(extra={"plan": {"total_frames": 90}})

    made = []

    def make_matte(cut, folder, **options):
        made.append(cut)
        data = synthetic_matte(cut, folder, box=(390, 550, 300, 1370),
                                joins=options.get("joins", ()))
        return data

    monkeypatch.setattr(matte_project, "make_matte", make_matte)
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "synthetic person")
    captions(root)
    prep = next((root / ".yap/cache/camera3d").glob("*/prep-record.json"))
    prepared_at = prep.stat().st_mtime_ns
    first = [receipt.sha256_file(root / "captions" / name)
             for name in ("camera3d.mp4", "captions.mov")]
    captions(root, force=True)
    assert len(made) == 1 and prep.stat().st_mtime_ns == prepared_at
    assert first == [receipt.sha256_file(root / "captions" / name)
                     for name in ("camera3d.mp4", "captions.mov")]
    record = read_receipt(root / "captions")
    behind = record["camera_3d"]["behind_words"]
    assert behind and all(item["worst_visible_share"] >= .9 for item in behind)
    alpha = ffmpeg.run(["ffmpeg", "-v", "error", "-ss", "1.2", "-i",
                        str(root / "captions/captions.mov"), "-vf", "alphaextract",
                        "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"],
                       binary=True, timeout_s=60).stdout
    alpha = np.frombuffer(alpha, np.uint8).reshape(1920, 1080)
    assert alpha.min() == 0 and alpha.max() > 200 and alpha[1910, 1070] == 0
    graphics(root)
    music(root)
    result = assemble.assemble(root)
    final = read_receipt(root / "final")
    assert result["status"] == "verified"
    assert final["verify"]["status"] == "pass", final["verify"]["findings"]
    assert final["verify"]["checks"]["picture.grade"]["ran"]
