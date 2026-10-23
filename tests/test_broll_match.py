"""Synthetic colour matching, scoped picture changes and deterministic LUTs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth
from test_assemble_picture import (
    heic_quadrants,
    insert,
    lane,
    layers,
    overlay_image,
    picture_command,
    project,
)
from test_broll_split import measured
from test_planner_broll import catalog_item, phrase_words, requests

from yap_studio import ffmpeg, receipt
from yap_studio.errors import ProcessingError, ValidationError
from yap_studio.faces import project as face_project
from yap_studio.faces.model import Box
from yap_studio.grade import GradeSpec, apply_rgb, graph_filter, rgba_filter
from yap_studio.planner import broll
from yap_studio.stages import assemble_picture as assembly
from yap_studio.stages import broll_match as match
from yap_studio.stages.timeline import ResolvedTimeline, resolve_timeline
from yap_studio.style import resolve_style_data


def stats(median: float = 0.4, spread: float = 0.3, saturation: float = 0.1,
          wb: tuple[float, float] = (1, 1)) -> dict[str, Any]:
    return {"median": median, "spread": spread, "saturation": saturation, "wb": list(wb)}


def test_equal_stats_are_unchanged() -> None:
    assert match.match_parameters(stats(), stats()) == "unchanged"


@pytest.mark.parametrize(("take", "shot", "expected"), [
    (0.9, 0.35, 0.7), (0.1, 0.9, -0.7), (0.9, 0.12, 0), (0.9, 0.25, 0.35),
])
def test_exposure_bounds_and_dark_scene_cap(take: float, shot: float, expected: float) -> None:
    result = match.match_parameters(stats(take), stats(shot))
    assert (0 if result == "unchanged" else result["stops"]) == expected


def test_strength_moves_part_way() -> None:
    result = match.match_parameters(stats(0.5, 0.33, 0.12, (1.08, 0.95)), stats())
    assert result["stops"] == round(0.7 * math.log2((0.5 / 0.4) ** 2.2), 3)
    assert result["contrast"] == round(1.1 ** 0.7, 3)
    assert result["saturation"] == round(1.2 ** 0.7, 3)
    assert 1 < result["gains"][0] / result["gains"][1] < 1.08


def test_sunset_stays_warm_and_gains_preserve_luma() -> None:
    result = match.match_parameters(stats(), stats(wb=(2, 0.5)))
    red, green, blue = result["gains"]
    assert red / green == pytest.approx(0.9, abs=0.002)
    assert blue / green == pytest.approx(1.1, abs=0.002)
    assert np.dot(result["gains"], [0.2126, 0.7152, 0.0722]) == pytest.approx(1, abs=0.001)
    sunset = match.transform(np.array([[0.7, 0.5, 0.3]]), result)
    assert sunset[0, 0] > sunset[0, 1] > sunset[0, 2]


@pytest.mark.parametrize(("take", "expected"), [(0.001, 0.75), (1, 1.3)])
def test_saturation_bounds(take: float, expected: float) -> None:
    assert match.match_parameters(stats(saturation=take), stats())["saturation"] == expected


@pytest.mark.parametrize(("take", "expected"), [(0.02, 0.85), (1, 1.2)])
def test_contrast_bounds(take: float, expected: float) -> None:
    assert match.match_parameters(stats(spread=take), stats())["contrast"] == expected


def test_greyscale_keeps_its_colour() -> None:
    result = match.match_parameters(stats(0.6, wb=(2, 0.5)), stats(saturation=0.02))
    assert result["gains"] == [1, 1, 1]
    assert result["saturation"] == 1


def test_lut_is_content_addressed_and_identity_creates_nothing(tmp_path: Path) -> None:
    params = match.match_parameters(stats(0.5, wb=(1.1, 0.9)), stats())
    first = match.write_lut(tmp_path, params)
    original = first.read_bytes()
    modified = first.stat().st_mtime_ns
    second = match.write_lut(tmp_path, params)
    assert first == second and second.read_bytes() == original
    assert second.stat().st_mtime_ns == modified
    assert first.stem == hashlib.sha256(original).hexdigest()
    rows = original.decode().splitlines()
    assert rows[0] == "LUT_3D_SIZE 33"
    grid = np.array([[0, 0, 0], [1 / 32, 0, 0], [0, 1 / 32, 0], [0, 0, 1 / 32]])
    expected = match.transform(grid, params)
    np.testing.assert_allclose([list(map(float, rows[n].split()))
                               for n in (1, 2, 34, 1090)], expected, atol=1e-6)
    empty = tmp_path / "identity"
    assert match.write_lut(empty, "unchanged") is None
    assert not empty.exists()
    assert match.write_lut(empty, {"stops": 0, "contrast": 1, "gains": [1, 1, 1],
                                   "saturation": 1, "pivot": 0.5}) is None
    assert not empty.exists()


def test_transform_order_and_highlight_shoulder() -> None:
    params = {"stops": 0.5, "contrast": 1.1, "gains": [0.95, 1.01, 1.07],
              "saturation": 0.85, "pivot": 0.4}
    pixel = np.array([[0.9, 0.5, 0.2]])
    linear = pixel ** 2.2 * params["gains"] * 2 ** 0.5
    linear = np.where(linear > 0.8, 0.8 + 0.2 * (1 - np.exp(-(linear - 0.8) / 0.2)), linear)
    expected = 0.4 + (np.clip(linear, 0, 1) ** (1 / 2.2) - 0.4) * 1.1
    luma = expected @ np.array([0.2126, 0.7152, 0.0722])
    expected = np.clip(luma[:, None] + (expected - luma[:, None]) * 0.85, 0, 1)
    np.testing.assert_allclose(match.transform(pixel, params), expected)


def test_lut_path_escaping_and_alpha(tmp_path: Path) -> None:
    params = match.match_parameters(stats(0.5, wb=(1.1, 0.9)), stats())
    lut = match.write_lut(tmp_path / "quote' colon: comma, semi; brackets[] slash\\ space", params)
    source = ["ffmpeg", "-v", "error", "-filter_complex_threads", "1", "-f", "lavfi", "-i",
              "color=0x8ca0b9@0.35:s=16x16,format=rgba", "-filter_complex"]
    def pixels(filters: str) -> np.ndarray:
        result = ffmpeg.run([*source, f"[0:v]{filters}[v]", "-map", "[v]", "-frames:v", "1",
                             "-threads", "1", "-f", "rawvideo", "-"],
                            timeout_s=60, binary=True)
        return np.frombuffer(result.stdout, dtype=np.uint8).reshape(16, 16, 4)
    before = pixels("format=rgba")
    after = pixels(match._lut_filter(lut) + ",format=rgba")
    np.testing.assert_array_equal(before[..., 3], after[..., 3])
    assert np.any(before[..., :3] != after[..., :3])


def test_white_balance_uses_least_saturated_pixels() -> None:
    pixels = np.array([[255, 0, 0]] * 50 + [[128, 128, 128]] * 50, dtype=np.uint8)
    result = match.measure(pixels)
    assert result["wb"] == [1, 1]
    assert result["saturation"] == 0.5
    assert all(value == round(value, 4) for value in result["wb"])


@pytest.fixture(scope="module")
def pictures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("match-pictures")
    return {
        "cut": synth.picture_source(folder / "cut.mov", "nullsrc=s=108x192:r=30,"
                                    "geq=r='170+20*X/W':g='145+15*Y/H':b='125+10*X/W',"
                                    "scale=1080:1920", 60),
        "take": synth.picture_source(folder / "take.mov", "color=0xb9a08c:s=160x90:r=30", 60),
        "insert": synth.picture_source(folder / "insert.mov", "color=0x8ca0b9:s=320x180:r=30", 60),
        "title": synth.picture_source(folder / "title.mov", "color=yellow@0.5:s=40x20:r=30,"
                                      "format=rgba", 1, alpha=True),
        "captions": synth.picture_source(folder / "captions.mov", "color=lime@0:s=40x20:r=30,"
                                         "format=rgba", 60, alpha=True),
    }


@pytest.fixture
def scene(tmp_path: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    root = project(tmp_path / "project", pictures, frames=60, hold_frames=0)
    data = json.loads((root / "project.json").read_bytes())
    data["style"] = "preset:clean_minimal"
    (root / "project.json").write_text(json.dumps(data))
    layers(root, pictures, monkeypatch, titles=[])
    monkeypatch.setattr(assembly, "captions", lambda _: {"layer": {
        "path": "captions.mov", "x": 60, "y": 1800, "width": 40, "height": 20,
        "start_frame": 0, "frames": 60,
    }})
    shot = insert(root, pictures["insert"], 15, 45, identifier="broll-insert")
    over = insert(root, pictures["insert"], 3, 12, identifier="broll-overlay")
    over.update(kind="overlay", placement="box", box=[600, 800, 320, 240], motion="none")
    lane(root, [shot, over], name="broll")
    return root


def style(root: Path, **broll_style: Any) -> None:
    data = json.loads((root / "project.json").read_bytes())
    data["style"] = "style.json"
    (root / "project.json").write_text(json.dumps(data))
    (root / "style.json").write_text(json.dumps({"schema_version": 1, "broll": broll_style}))


def render(root: Path, name: str,
           timeline: ResolvedTimeline | None = None, *,
           grade: GradeSpec | None = None) -> tuple[Path, assembly.PictureResult]:
    work = root / name
    out = work / "picture.mp4"
    return out, assembly.render_picture(root, out, work_dir=work, timeline=timeline, grade=grade)


def frame_hashes(result: assembly.PictureResult, *,
                 box: tuple[int, int, int, int] | None = None,
                 full_chroma: bool = False) -> list[str]:
    command = list(picture_command(result))
    command[0] = "ffmpeg"
    if full_chroma:
        graph_index = command.index("-filter_complex") + 1
        command[graph_index] = command[graph_index].replace(
            f"format={assembly.PIX_FMT}", "format=yuv444p").replace(
                ":format=auto:", ":format=yuv444:")
    if box:
        graph_index, map_index = command.index("-filter_complex") + 1, command.index("-map") + 1
        crop = ":".join(map(str, (box[2], box[3], box[0], box[1])))
        command[graph_index] += f";{command[map_index]}crop={crop}[hash_crop]"
        command[map_index] = "[hash_crop]"
    command = command[:command.index("-c:v")] + [
        "-c:v", "rawvideo", "-f", "framehash", "-",
    ]
    raw = ffmpeg.run(command, timeout_s=120).stdout
    return [line.rsplit(",", 1)[-1].strip() for line in raw.splitlines()
            if line and not line.startswith("#")]


@pytest.mark.parametrize("film", [False, True])
def test_assembly_leaves_take_untouched_and_is_deterministic(scene: Path, film: bool) -> None:
    def overlapping() -> ResolvedTimeline:
        timeline = resolve_timeline(scene).require_valid()
        return replace(timeline, elements=tuple(
            replace(item, start_frame=18, end_frame=42) if item.kind == "overlay" else item
            for item in timeline.elements))

    spec = GradeSpec("film", 1, "style") if film else None
    style(scene, color_match="off")
    off, plain = render(scene, "off", overlapping(), grade=spec)
    style(scene, color_match="on")
    on, matched = render(scene, "on", overlapping(), grade=spec)
    again, repeated = render(scene, "again", overlapping(), grade=spec)
    before, after = frame_hashes(plain), frame_hashes(matched)
    assert len(before) == len(after) == 60
    assert before[:15] == after[:15] and before[45:] == after[45:]
    assert all(a != b for a, b in zip(before[15:45], after[15:45], strict=True))
    target = synth.picture_pixels(off, 0).mean(axis=(0, 1))
    original = synth.picture_pixels(off, 30, box=(100, 500, 100, 100)).mean(axis=(0, 1))
    changed = synth.picture_pixels(on, 30, box=(100, 500, 100, 100)).mean(axis=(0, 1))
    assert np.linalg.norm(changed - target) < np.linalg.norm(original - target)
    assert on.read_bytes() == again.read_bytes()
    assert matched.broll_match == repeated.broll_match
    assert {entry["kind"] for entry in matched.broll_match} == {"insert", "overlay"}
    for entry in matched.broll_match:
        assert entry["lut"] == repeated.broll_match[0]["lut"]
        assert (scene / "on/broll-match" / f"{entry['lut']}.cube").read_bytes() == (
            scene / "again/broll-match" / f"{entry['lut']}.cube").read_bytes()
    decodes = [cmd for cmd in matched.commands if "rawvideo" in cmd]
    assert len([cmd for cmd in decodes if str(scene / "cut/cut.mp4") in cmd]) == 1


@pytest.mark.parametrize("photo", [False, True])
@pytest.mark.parametrize("film", [False, True])
def test_split_matches_only_broll_and_preserves_photo_push(
    scene: Path, pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    photo: bool, film: bool,
) -> None:
    source = pictures["insert"]
    if photo:
        source = overlay_image(scene / "photo.png", "color=0x8ca0b9:s=320x180,"
                               "drawbox=x=128:y=58:w=64:h=64:c=white:t=fill")
    shot = insert(scene, source, 15, 30, identifier="broll-split", fit="split")
    if photo:
        shot.update(motion="push", push_scale=1.08)
    lane(scene, [shot], name="broll")
    faces = measured(((Box(440, 650, 200, 280),),) * 60)
    monkeypatch.setattr(face_project, "cut_faces", lambda *args: faces)
    monkeypatch.setattr(assembly, "PRESET", "ultrafast")
    spec = GradeSpec("film", 1, "style") if film else None
    style(scene, color_match="off")
    off, plain = render(scene, "split-off", grade=spec)
    style(scene, color_match="on")
    on, matched = render(scene, "split-on", grade=spec)

    # Keep full chroma before delivery subsampling mixes pixels across the seam.
    # Include the seam and every lower pixel, without encoding or a tolerance.
    before = frame_hashes(plain, box=(0, 960, 1080, 960), full_chroma=True)
    after = frame_hashes(matched, box=(0, 960, 1080, 960), full_chroma=True)
    assert len(before) == len(after) == 60
    assert before == after
    top_before = frame_hashes(plain, box=(0, 0, 1080, 960))
    top_after = frame_hashes(matched, box=(0, 0, 1080, 960))
    assert top_before[:15] == top_after[:15] and top_before[30:] == top_after[30:]
    assert all(a != b for a, b in zip(top_before[15:30], top_after[15:30], strict=True))
    target = synth.picture_pixels(off, 0).mean(axis=(0, 1))
    original = synth.picture_pixels(off, 22, box=(100, 100, 100, 100)).mean(axis=(0, 1))
    changed = synth.picture_pixels(on, 22, box=(100, 100, 100, 100)).mean(axis=(0, 1))
    assert np.linalg.norm(changed - target) < np.linalg.norm(original - target)
    receipt_data = matched.to_dict()
    layout, = receipt_data["splits"]
    entry, = receipt_data["broll_match"]
    assert layout["id"] == entry["id"] == "broll-split"
    assert layout["seam_y"] == 960 and entry["kind"] == "insert" and entry["lut"]
    assert not plain.broll_match
    if photo:
        # The white marker grows within the top box while her picture remains fixed.
        for output in (off, on):
            start = synth.picture_pixels(output, 15, box=(0, 480, 1080, 2))
            end = synth.picture_pixels(output, 29, box=(0, 480, 1080, 2))
            assert np.count_nonzero(end.mean(axis=2) > 220) > np.count_nonzero(
                start.mean(axis=2) > 220)


@pytest.mark.parametrize("film", [False, True])
def test_film_grades_matched_layers_once_in_order(scene: Path, film: bool) -> None:
    style(scene, color_match="on")
    spec = GradeSpec("film", 1, "style") if film else None
    out, result = render(scene, "matched-film", grade=spec)
    command = picture_command(result)
    graph = command[command.index("-filter_complex") + 1]
    inputs = [Path(command[index + 1]).name
              for index, value in enumerate(command) if value == "-i"]
    nodes = []
    for row in graph.split(";"):
        parsed = re.fullmatch(r"((?:\[[^]]+\])+)(.*)\[([^]]+)\]", row)
        assert parsed is not None
        pads, expression, output = parsed.groups()
        nodes.append((re.findall(r"\[([^]]+)\]", pads), expression, output))

    def branch(filename: str) -> tuple[list, tuple]:
        pad = f"{inputs.index(filename)}:0"
        rows = []
        for _ in nodes:
            consumers = [node for node in nodes if pad in node[0]]
            assert len(consumers) == 1
            node = consumers[0]
            if len(node[0]) > 1:
                return rows, node
            rows.append(node)
            pad = node[2]
        pytest.fail(f"{filename}: no composite in the picture graph")

    insert_rows, insert_mix = branch("broll-insert.mov")
    overlay_rows, overlay_mix = branch("broll-overlay.mov")
    records = {entry["id"]: entry for entry in result.broll_match}
    cube = next(out.parent.glob("grade-v*.cube")) if film else None
    if cube:
        base_filter, overlay_filter = graph_filter(cube), rgba_filter(cube)
        base_grades = [node for node in nodes if node[1] == base_filter]
        assert len(base_grades) == 1 and graph.count(overlay_filter) == 2
        assert base_grades[0][0] == [insert_mix[2]]
        assert overlay_mix[0][0] == base_grades[0][2]
    else:
        assert result.grade is None and not list(out.parent.glob("grade-v*.cube"))
        assert graph.count("lut3d=") == 2
        assert overlay_mix[0][0] == insert_mix[2]
    for identifier, rows in (("broll-insert", insert_rows), ("broll-overlay", overlay_rows)):
        lut = out.parent / "broll-match" / f"{records[identifier]['lut']}.cube"
        expression = ";".join(node[1] for node in rows)
        match_filter = match._lut_filter(lut)
        assert expression.count(match_filter) == 1
        if cube and identifier == "broll-insert":
            assert cube.name not in expression
        elif cube:
            assert expression.count(overlay_filter) == 1
            assert expression.index(match_filter) < expression.index(overlay_filter)
    _, caption_mix = branch("captions.mov")
    assert caption_mix[0][0] == overlay_mix[2]

    shot = synth.picture_pixels(scene / "broll-insert.mov", 15).mean(axis=(0, 1)) / 255
    entry = records["broll-insert"]
    params = match.match_parameters(entry["take"], entry["broll"])
    assert params != "unchanged"
    transformed = match.transform(shot, params)
    expected = (apply_rgb(transformed, spec) if spec else transformed) * 255
    grade_only = (apply_rgb(shot, spec) if spec else shot) * 255
    actual = synth.picture_pixels(out, 30, box=(100, 500, 100, 100)).mean(axis=(0, 1))
    error = np.abs(actual - expected).mean()
    missing = np.abs(actual - grade_only).mean()
    assert error < 3, (actual, expected, error)
    assert missing > 3 and error < missing, (actual, grade_only, missing)


@pytest.mark.parametrize("film", [False, True])
def test_film_matching_leaves_talking_head_frame_hashes_unchanged(scene: Path, film: bool) -> None:
    spec = GradeSpec("film", 1, "style") if film else None
    style(scene, color_match="off")
    _, plain = render(scene, "film-match-off", grade=spec)
    style(scene, color_match="on")
    timeline = resolve_timeline(scene).require_valid()
    _, matched = render(scene, "film-match-on", timeline, grade=spec)
    assert plain.grade == matched.grade
    if film:
        assert matched.grade is not None and matched.grade["look"] == "film"
    else:
        assert matched.grade is None
    before, after = frame_hashes(plain), frame_hashes(matched)
    assert len(before) == len(after) == timeline.total_frames
    windows = [(item.start_frame, item.end_frame) for item in timeline.elements
               if item.kind in {"insert", "overlay"}]
    assert len(windows) == 2
    outside = [frame for frame in range(timeline.total_frames)
               if not any(start <= frame < end for start, end in windows)]
    assert outside and all(before[frame] == after[frame] for frame in outside)
    for start, end in windows:
        assert any(before[frame] != after[frame] for frame in range(start, end))


@pytest.mark.parametrize("film", [False, True])
def test_letters_matches_fill_once_before_plate(scene: Path, monkeypatch: pytest.MonkeyPatch,
                                               film: bool) -> None:
    shot = insert(scene, scene / "broll-insert.mov", 0, 60, identifier="broll-letters")
    shot["letters"] = {"text": "alpha"}
    lane(scene, [shot], name="broll")
    style(scene, color_match="on", emphasis_on=["video_letters"])
    plate = synth.picture_source(
        scene / "letters-plate.mov", "color=0xf0d878:s=108x192:r=30,format=rgba,"
        "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
        "a='if(between(X,24,83)*between(Y,50,119),0,255)',scale=1080:1920", 48, alpha=True)
    monkeypatch.setattr(assembly, "PRESET", "ultrafast")
    out = scene / "letters-match/picture.mp4"
    result = assembly.render_picture(
        scene, out, work_dir=out.parent, grade=GradeSpec("film", 1, "style") if film else None,
        letters_layers=[{"id": "broll-letters", "path": str(plate),
                         "start_frame": 0, "frames": 48}])
    entry, = result.broll_match
    assert entry["id"] == "broll-letters" and entry["kind"] == "insert"
    command = picture_command(result)
    graph = command[command.index("-filter_complex") + 1]
    inputs = [Path(command[index + 1]).name
              for index, value in enumerate(command) if value == "-i"]
    nodes = []
    for row in graph.split(";"):
        parsed = re.fullmatch(r"((?:\[[^]]+\])+)(.*)\[([^]]+)\]", row)
        assert parsed is not None
        pads, expression, output = parsed.groups()
        nodes.append((re.findall(r"\[([^]]+)\]", pads), expression, output))

    def branch(filename: str) -> tuple[list[str], tuple]:
        pad = f"{inputs.index(filename)}:0"
        expressions = []
        for _ in nodes:
            consumers = [node for node in nodes if pad in node[0]]
            assert len(consumers) == 1
            node = consumers[0]
            if len(node[0]) > 1:
                return expressions, node
            expressions.append(node[1])
            pad = node[2]
        pytest.fail(f"{filename}: no composite in the picture graph")

    fill, fill_mix = branch("broll-letters.mov")
    plate_filters, plate_mix = branch(plate.name)
    lut = match._lut_filter(out.parent / "broll-match" / f"{entry['lut']}.cube")
    assert graph.count(lut) == 1 and sum(lut in expression for expression in fill) == 1
    assert not any("lut3d=" in expression for expression in plate_filters)
    assert fill_mix == plate_mix and fill_mix[1].startswith("overlay=")
    assert "between(n,0,47)" in fill_mix[1]
    assert graph.count("lut3d=") == (2 if film else 1)


@pytest.fixture
def card_scene(scene: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    lane(scene, [], name="broll")
    lane(scene, [{"id": "agent-card", "kind": "graphic", "layer": "front",
                  "start": {"time": 0.3}, "hold_s": 1.2,
                  "template": "text_tag", "slots": {"text": "Alpha"}}])
    synth.picture_source(scene / "card.mov", "color=yellow@0:s=352x120:r=30,format=rgba",
                         36, alpha=True)
    window = {"slot": "clip", "index": 0, "x": 96, "y": 116, "width": 128, "height": 88,
              "radius": 12, "start_frame": 12, "frames": 30, "fade_in_frames": 3,
              "fade_out_frames": 3, "source": {
                  "kind": "media", "path": "broll-insert.mov",
                  "sha256": receipt.sha256_file(scene / "broll-insert.mov"),
                  "in_s": 0.2, "speed": 1.5, "focus": [0.25, 0.75],
              }}
    placement = {"id": "agent-card", "path": "card.mov", "x": 80, "y": 100,
                 "width": 352, "height": 120, "start_frame": 9, "frames": 36,
                 "layer": "front", "kind": "graphic", "windows": [
                     window, {**window, "index": 1, "x": 288, "source": {"kind": "self"}},
                 ]}
    monkeypatch.setattr(assembly, "graphics", lambda _: {"elements": [placement]})
    return scene, placement


@pytest.mark.parametrize("film", [False, True])
def test_card_windows_match_media_before_grade_and_leave_self_untouched(
    card_scene: tuple[Path, dict], film: bool,
) -> None:
    root, placement = card_scene
    spec = GradeSpec("film", 1, "style") if film else None
    style(root, color_match="off")
    off, plain = render(root, "card-off", grade=spec)
    style(root, color_match="on")
    on, matched = render(root, "card-on", grade=spec)
    entry, = matched.broll_match
    assert entry["kind"] == "card_window" and entry["id"] == "agent-card window 0"
    command = picture_command(matched)
    graph = command[command.index("-filter_complex") + 1]
    lut = on.parent / "broll-match" / f"{entry['lut']}.cube"
    expression = match._lut_filter(lut) + ",format=rgba"
    assert graph.count(expression) == 1
    row = next(row for row in graph.split(";") if expression in row)
    match_pad = row.rsplit("[", 1)[1].removesuffix("]")
    if film:
        cube = next(on.parent.glob("grade-v*.cube"))
        assert graph.count(f"]{graph_filter(cube)}[") == 1
        assert graph.count(f"]{rgba_filter(cube)}[") == 2
        assert f"[{match_pad}]{rgba_filter(cube)}[" in graph
    else:
        assert graph.count("lut3d=") == 1
    target = synth.picture_pixels(off, 0).mean(axis=(0, 1))
    original = synth.picture_pixels(off, 27, box=(140, 140, 24, 24)).mean(axis=(0, 1))
    actual = synth.picture_pixels(on, 27, box=(140, 140, 24, 24)).mean(axis=(0, 1))
    assert np.linalg.norm(actual - target) < np.linalg.norm(original - target)
    window = placement["windows"][0]
    source = assembly._source(root / window["source"]["path"], [], [], owner="window")
    branch = assembly._Graph()
    pad = assembly._window_clip(branch, source, window, owner="window")
    pad = branch.filter(pad, "select='eq(n,15)'")
    decoded = ffmpeg.run([
        "ffmpeg", "-v", "error", "-filter_complex_threads", "1", *branch.inputs,
        "-filter_complex", ";".join(branch.filters), "-map", f"[{pad}]",
        "-frames:v", "1", "-fps_mode", "passthrough",
        "-pix_fmt", "rgba", "-f", "rawvideo", "-",
    ], binary=True, timeout_s=30)
    _, _, width, height = assembly._window_rect(window)
    pixels = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(height, width, 4)
    assert pixels[height // 2, width // 2, 3] == 255
    shot = pixels[height // 2, width // 2, :3] / 255
    expected = match.transform(shot, match.match_parameters(entry["take"], entry["broll"]))
    # Each LUT returns eight-bit RGB before the layer is composited.
    expected = np.floor(expected * 255) / 255
    if spec:
        expected = np.floor(apply_rgb(expected, spec) * 255) / 255
        assert np.abs(actual - apply_rgb(shot, spec) * 255).mean() > 3
    np.testing.assert_allclose(actual, expected * 255, atol=3)
    before, after = frame_hashes(plain), frame_hashes(matched)
    assert len(before) == len(after) == 60
    assert before[:12] == after[:12] and before[42:] == after[42:]
    assert any(a != b for a, b in zip(before[15:39], after[15:39], strict=True))
    assert frame_hashes(plain, box=(330, 150, 24, 24)) == frame_hashes(
        matched, box=(330, 150, 24, 24))


def test_card_windows_style_off_does_no_measurement(
    card_scene: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = card_scene
    style(root, color_match="off")
    monkeypatch.setattr(match, "_decode", lambda *args, **kwargs: pytest.fail("Matching is off"))
    _, result = render(root, "card-disabled")
    assert not result.broll_match
    assert "lut3d" not in " ".join(picture_command(result))
    assert not any("rawvideo" in command for command in result.commands)


@pytest.mark.parametrize("setting", ["on", "off"])
def test_card_windows_use_resolved_style_without_snapshot_style(
    card_scene: tuple[Path, dict], setting: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = card_scene
    timeline = resolve_timeline(root).require_valid()
    data = json.loads((root / "project.json").read_bytes())
    data["style"] = "styles/owner-review.json"
    (root / "project.json").write_text(json.dumps(data))
    assert not (root / data["style"]).exists()
    resolved_style = resolve_style_data({"schema_version": 1, "broll": {"color_match": setting}})
    if setting == "off":
        monkeypatch.setattr(match, "_decode",
                            lambda *args, **kwargs: pytest.fail("Matching is off"))
    out = root / "snapshot-style/picture.mp4"
    result = assembly.render_picture(root, out, work_dir=out.parent, timeline=timeline,
                                     moment_style=resolved_style)
    if setting == "on":
        entry, = result.broll_match
        assert entry["id"] == "agent-card window 0" and entry["kind"] == "card_window"
        assert "take" in entry and "broll" in entry
        assert "lut3d" in " ".join(picture_command(result))
    else:
        assert not result.broll_match
        assert "lut3d" not in " ".join(picture_command(result))
        assert not any("rawvideo" in command for command in result.commands)


def test_self_card_window_needs_no_match_style_or_decode(
    card_scene: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.project.status import ProjectContext

    root, placement = card_scene
    placement["windows"] = [placement["windows"][1]]
    timeline = resolve_timeline(root).require_valid()
    monkeypatch.setattr(ProjectContext, "render_style",
                        lambda _: pytest.fail("Self windows do not need a matching style"))
    monkeypatch.setattr(match, "_decode", lambda *args, **kwargs: pytest.fail("Self is not B-roll"))
    filters, records, commands = measurement(root, timeline, placements=[placement])
    assert not filters and not records
    assert not any("rawvideo" in command for command in commands)


def test_voiceover_card_window_skips_without_decoding(
    card_scene: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, placement = card_scene
    data = json.loads((root / "project.json").read_bytes())
    data["takes"][0]["voiceover"] = True
    (root / "project.json").write_text(json.dumps(data))
    timeline = resolve_timeline(root).require_valid()
    monkeypatch.setattr(match, "_decode", lambda *args, **kwargs: pytest.fail("No take to match"))
    filters, records, commands = measurement(root, timeline, placements=[placement])
    assert not filters and records == [{"id": "agent-card window 0", "kind": "card_window",
                                       "skipped": "voiceover video"}]
    assert not any("rawvideo" in command for command in commands)


def test_card_windows_share_take_decode_with_elements(
    card_scene: tuple[Path, dict], pictures: dict[str, Path],
) -> None:
    root, placement = card_scene
    lane(root, [insert(root, pictures["insert"], 45, 60, identifier="broll-extra")], name="broll")
    timeline = resolve_timeline(root).require_valid()
    filters, records, commands = measurement(root, timeline, placements=[placement])
    assert set(filters) == {"broll-extra", ("agent-card", 0)}
    assert {record["kind"] for record in records} == {"insert", "card_window"}
    take_decodes = [command for command in commands
                    if "rawvideo" in command and str(root / "cut/cut.mp4") in command]
    assert len(take_decodes) == 1
    assert {0, 17, 27, 37, 56, 30, 47, 52, 57, 59} == {
        frame for record in records for frame in record["take_frames"]
    }


def test_card_window_measurement_uses_timing_focus_and_rounded_alpha(
    card_scene: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, placement = card_scene
    clip = synth.picture_source(
        root / "window.mov", "nullsrc=s=128x64:r=30,"
        "geq=r='40+N+X':g='80+Y':b='190-X'", 120)
    window = placement["windows"][0]
    window.update(width=64, height=64, radius=30, frames=30)
    window["source"].update(path=clip.name, sha256=receipt.sha256_file(clip),
                            in_s=2, speed=1.5, focus=[1, 1])
    sampled = []
    decode = match._decode

    def capture(source, *args, **kwargs):
        pixels = decode(source, *args, **kwargs)
        if source.path == clip:
            sampled.append(pixels)
        return pixels

    monkeypatch.setattr(match, "_decode", capture)
    _, records, commands = measurement(root, resolve_timeline(root).require_valid(),
                                       placements=[placement])
    entry, = records
    pixels, = sampled
    assert pixels.shape == (3, 192, 192, 4)
    visible = pixels[..., 3] >= 128
    assert not visible.all() and visible.mean() > 0.05
    assert entry["broll"] == match.measure(pixels[..., :3][visible])
    assert entry["broll"] != match.measure(pixels[..., :3])
    for index, frame in enumerate((67, 82, 97)):
        expected = synth.picture_pixels(clip, frame, box=(96, 32, 2, 2)).mean(axis=(0, 1))
        np.testing.assert_allclose(pixels[index, 96, 96, :3], expected, atol=3)
    command = next(command for command in commands
                    if "rawvideo" in command and str(clip) in command)
    assert command.index("-ss") < command.index("-i")
    assert command[command.index("-ss") + 1] == "2.000000000000"
    assert command[command.index("-t") + 1] == "2.000000000000"


def test_card_window_measurement_uses_source_crop(card_scene: tuple[Path, dict]) -> None:
    root, placement = card_scene
    timeline = resolve_timeline(root).require_valid()
    window = placement["windows"][0]
    window.update(width=64, height=64, radius=16, frames=30)
    window["source"].update(in_s=0, speed=1, focus=[1, 1], crop=[16, 16, 64, 64])
    records, cubes = [], []
    for index, hidden in enumerate(((255, 0, 0), (0, 0, 255))):
        visible = "between(X,16,79)*between(Y,16,79)"
        clip = synth.picture_source(
            root / f"cropped-{index}.mov", "nullsrc=s=128x96:r=30,"
            f"geq=r='if({visible},140,{hidden[0]})':"
            f"g='if({visible},160,{hidden[1]})':b='if({visible},185,{hidden[2]})'", 60)
        window["source"].update(path=clip.name, sha256=receipt.sha256_file(clip))
        work = f"cropped-window-{index}"
        filters, measured, commands = measurement(root, timeline, work, placements=[placement])
        entry, = measured
        assert set(filters) == {(placement["id"], 0)}
        records.append(entry)
        cubes.append((root / work / "broll-match" / f"{entry['lut']}.cube").read_bytes())
        command = next(command for command in commands if "rawvideo" in command
                       and str(clip) in command)
        geometry = command[command.index("-vf") + 1]
        assert "crop=64:64:16:16:exact=1,scale=" in geometry
    assert records[0]["broll"] == records[1]["broll"]
    assert records[0]["lut"] == records[1]["lut"] and cubes[0] == cubes[1]
    expected = synth.picture_pixels(clip, 15, box=(32, 32, 2, 2))
    assert records[0]["broll"] == match.measure(expected)


def test_on_by_default(scene: Path) -> None:
    assert resolve_style_data({"schema_version": 1})["broll"]["color_match"] == "on"
    timeline = resolve_timeline(scene).require_valid()
    assert {item.kind for item in timeline.elements} == {"insert", "overlay"}
    assert all(item.payload["color_match"] is True for item in timeline.elements)
    _, result = render(scene, "default")
    assert "lut3d" in " ".join(picture_command(result))
    assert {entry["kind"] for entry in result.broll_match} == {"insert", "overlay"}


def test_older_style_without_colour_match_gets_matching(scene: Path) -> None:
    style(scene, still_push="none", avoid_tags=[])
    data = json.loads((scene / "style.json").read_bytes())
    assert "color_match" not in data["broll"]
    timeline = resolve_timeline(scene).require_valid()
    assert all(item.payload["color_match"] is True for item in timeline.elements)
    _, result = render(scene, "older-style")
    assert "lut3d" in " ".join(picture_command(result))
    assert {entry["kind"] for entry in result.broll_match} == {"insert", "overlay"}


def test_style_off_and_no_decode(scene: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    style(scene, color_match="off")
    timeline = resolve_timeline(scene).require_valid()
    assert all("color_match" not in item.payload for item in timeline.elements)
    monkeypatch.setattr(match, "measure", lambda _: pytest.fail("Off must not measure"))
    _, result = render(scene, "default")
    assert "lut3d" not in " ".join(picture_command(result))
    assert not any("rawvideo" in command for command in result.commands)
    assert "broll_match" not in result.to_dict()


@pytest.mark.parametrize(("setting", "override", "expected"), [
    ("on", None, True), ("on", False, False), ("off", True, True),
])
def test_timeline_colour_override(scene: Path, setting: str,
                                  override: bool | None, expected: bool) -> None:
    style(scene, color_match=setting)
    target = scene / "timeline/broll.json"
    data = json.loads(target.read_bytes())
    if override is not None:
        for element in data["elements"]:
            element["color_match"] = override
    target.write_text(json.dumps(data))
    assert all(item.payload["color_match"] is expected
               for item in resolve_timeline(scene).require_valid().elements)


@pytest.mark.parametrize("kind", ["insert", "overlay"])
@pytest.mark.parametrize("value", [True, False, "on", 1, None])
def test_planner_colour_request(tmp_path: Path, kind: str, value: Any) -> None:
    item = catalog_item(tmp_path)
    args = (requests(kind=kind, color_match=value), phrase_words(), [item], {})
    if type(value) is not bool:
        with pytest.raises(ValidationError, match="color_match"):
            broll.plan_broll(*args, frames=60, project_dir=tmp_path)
    else:
        plan = broll.plan_broll(*args, frames=60, project_dir=tmp_path)
        assert plan.lane()["elements"][0]["color_match"] is value


@pytest.mark.parametrize("media_type", ["screenshot", "screen_recording", "clip"])
@pytest.mark.parametrize("setting", ["on", "off", None])
def test_planner_protects_screens(tmp_path: Path, media_type: str, setting: str | None) -> None:
    item = catalog_item(tmp_path)
    item["media_type"] = media_type
    data = {} if setting is None else {"broll": {"color_match": setting}}
    plan = broll.plan_broll(requests(), phrase_words(), [item], data,
                            frames=60, project_dir=tmp_path)
    element = plan.lane()["elements"][0]
    if media_type != "clip":
        assert element["color_match"] is False
    else:
        assert "color_match" not in element


def test_voiceover_only_skips_matching(scene: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    timeline = resolve_timeline(scene).require_valid()
    data = json.loads((scene / "project.json").read_bytes())
    data["takes"][0]["voiceover"] = True
    (scene / "project.json").write_text(json.dumps(data))
    timeline = replace(timeline, elements=tuple(replace(item, payload={
        **item.payload, "color_match": True,
    }) for item in timeline.elements))
    monkeypatch.setattr(match, "measure", lambda _: pytest.fail("No talking head to measure"))
    monkeypatch.setattr(match, "load_edit_list", lambda _: pytest.fail("No cut spans needed"))
    result = assembly.render_picture(scene, scene / "voice/picture.mp4", work_dir=scene / "voice",
                                     timeline=timeline)
    assert "lut3d" not in " ".join(picture_command(result))
    assert not any("rawvideo" in command for command in result.commands)
    assert result.broll_match == tuple(
        {"id": item.id, "kind": item.kind, "skipped": "voiceover video"}
        for item in timeline.elements)
    assert not any("colour match" in warning for warning in result.warnings)


def measurement(scene: Path, timeline: ResolvedTimeline,
                work: str = "measurement", *,
                placements: list[dict] | None = None) -> tuple[dict, list[dict], list[tuple]]:
    commands, conversions = [], []
    sources = assembly._insert_inputs(scene, timeline, commands, conversions)
    cut = assembly._source(scene / "cut/cut.mp4", commands, conversions, owner="cut")
    windows = (assembly._window_sources(scene, placements, cut, commands, conversions)
               if placements else {})
    options = {"placements": placements, "window_sources": windows,
               "window_rect": assembly._window_rect} if placements else {}
    filters, records = match.match_broll(
        scene, timeline, cut, sources, scene / work, prefix=assembly._prefix,
        still_loop=assembly._still_loop, commands=commands, **options)
    return filters, records, commands


def transparent_overlay(scene: Path, name: str, hidden: tuple[int, int, int], *,
                        visible_width: int = 96, hidden_alpha: int = 0) -> None:
    visible = f"lt(X,{visible_width})"
    image = overlay_image(scene / f"{name}.png", "nullsrc=s=192x192,format=rgba,"
                          f"geq=r='if({visible},140,{hidden[0]})':"
                          f"g='if({visible},160,{hidden[1]})':"
                          f"b='if({visible},185,{hidden[2]})':"
                          f"a='if({visible},128,{hidden_alpha})'")
    element = insert(scene, image, 15, 45, identifier="broll-alpha")
    element.update(kind="overlay", placement="box", box=[100, 400, 192, 192],
                   fit="cover", motion="none", color_match=True)
    lane(scene, [element], name="broll")


@pytest.mark.parametrize("hidden_alpha", [0, 127])
def test_overlay_measurement_ignores_hidden_rgb(scene: Path, hidden_alpha: int) -> None:
    records, cubes = [], []
    for index, hidden in enumerate(((255, 0, 0), (0, 0, 255))):
        transparent_overlay(scene, f"alpha-{index}", hidden, hidden_alpha=hidden_alpha)
        work = f"alpha-{index}"
        filters, measured, _ = measurement(scene, resolve_timeline(scene).require_valid(), work)
        assert set(filters) == {"broll-alpha"} and len(measured) == 1
        records.append(measured[0])
        cubes.append((scene / work / "broll-match" / f"{measured[0]['lut']}.cube").read_bytes())
    assert records[0]["broll"] == records[1]["broll"]
    assert records[0]["lut"] == records[1]["lut"]
    assert cubes[0] == cubes[1]
    assert records[0]["broll"] == match.measure(np.array([[140, 160, 185]], dtype=np.uint8))


def test_overlay_with_too_few_visible_pixels_skips_quietly(scene: Path) -> None:
    transparent_overlay(scene, "sparse-alpha", (255, 0, 0), visible_width=3)
    _, result = render(scene, "sparse-overlay")
    assert result.broll_match == ({"id": "broll-alpha", "kind": "overlay",
                                   "skipped": "too few visible pixels"},)
    assert "lut3d" not in " ".join(picture_command(result))
    assert not any("colour match" in warning for warning in result.warnings)


@pytest.mark.parametrize(("kind", "fit", "coloured"), [
    ("insert", "cover", False), ("insert", "contain", True), ("insert", "crop", True),
    ("insert", "split", True),
    ("overlay", "cover", False), ("overlay", "contain", True),
])
def test_measurement_uses_visible_geometry(scene: Path, kind: str,
                                          fit: str, coloured: bool) -> None:
    image = overlay_image(scene / "geometry.png", "color=red:s=320x180,"
                          "drawbox=x=100:y=0:w=120:h=180:c=gray:t=fill")
    element = insert(scene, image, 15, 45, identifier="broll-geometry")
    element["color_match"] = True
    if kind == "insert":
        element["media"]["fit"] = fit
        if fit == "crop":
            element["media"]["crop"] = [0, 0, 80, 180]
    else:
        element.update(kind="overlay", placement="box", box=[200, 300, 108, 192], fit=fit,
                       motion="none")
    lane(scene, [element], name="broll")
    _, records, commands = measurement(scene, resolve_timeline(scene).require_valid())
    assert (records[0]["broll"]["saturation"] > 0.1) is coloured
    assert records[0]["broll"]["median"] > 0.1
    decodes = [cmd for cmd in commands if "rawvideo" in cmd]
    assert decodes[-1][decodes[-1].index("-frames:v") + 1] == "1"
    if fit == "split":
        expression = decodes[-1][decodes[-1].index("-vf") + 1]
        assert "scale=108:96:force_original_aspect_ratio=increase,crop=108:96:exact=1" in (
            expression)


def test_dark_reference_skips_without_broll_decode(scene: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    style(scene, color_match="on")
    original = match._decode
    def dark(*args: Any, **kwargs: Any) -> np.ndarray:
        assert args[0].path == scene / "cut/cut.mp4"
        return original(*args, **kwargs) * 0
    monkeypatch.setattr(match, "_decode", dark)
    _, result = render(scene, "dark")
    assert "lut3d" not in " ".join(picture_command(result))
    assert len(result.broll_match) == 2
    assert all(entry["skipped"] == "talking-head picture is too dark"
               for entry in result.broll_match)
    assert not any("colour match" in warning for warning in result.warnings)


def test_unchanged_keeps_quiet_receipts(scene: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(match, "match_parameters", lambda *_: "unchanged")
    _, result = render(scene, "unchanged")
    assert "lut3d" not in " ".join(picture_command(result))
    assert len(result.broll_match) == 2
    assert all(entry["skipped"] == "unchanged" for entry in result.broll_match)
    assert not any("colour match" in warning for warning in result.warnings)


def test_take_candidates_exclude_voiceover_spans_and_clamp(scene: Path) -> None:
    style(scene, color_match="on")
    timeline = resolve_timeline(scene).require_valid()
    data = json.loads((scene / "project.json").read_bytes())
    data["takes"].append({**data["takes"][0], "id": "t2", "voiceover": True})
    (scene / "project.json").write_text(json.dumps(data))
    spans = json.loads((scene / "cut/spans.json").read_bytes())
    spans["sources"]["t2"] = spans["sources"]["t1"]
    spans["segments"] = [{"source": "t2", "start": "0", "end": "1"},
                         {"source": "t1", "start": "0", "end": "1"}]
    (scene / "cut/spans.json").write_text(json.dumps(spans))
    _, result = render(scene, "mixed", timeline=timeline)
    for entry in result.broll_match:
        assert all(30 <= frame < 60 for frame in entry.get("take_frames", []))
    shot = next(entry for entry in result.broll_match if entry["kind"] == "insert")
    assert shot["take_frames"] == [30, 40, 59]
    overlay = next(entry for entry in result.broll_match if entry["kind"] == "overlay")
    assert overlay["skipped"] == "no talking-head picture near this moment"
    assert not any("colour match" in warning for warning in result.warnings)


def test_clip_samples_follow_source_in_and_speed(scene: Path) -> None:
    source = synth.picture_source(scene / "changing.mov", "color=gray:s=32x32:r=30,"
                                   "drawbox=c=red:t=fill:enable='lt(t,1)',"
                                   "drawbox=c=lime:t=fill:enable='gte(t,1.5)*lt(t,2)',"
                                   "drawbox=c=blue:t=fill:enable='gte(t,2)'", 90)
    element = insert(scene, source, 15, 45, in_s=1, speed=1.5)
    element["color_match"] = True
    lane(scene, [element], name="broll")
    _, records, commands = measurement(scene, resolve_timeline(scene).require_valid())
    expected = match.measure(np.array([synth.picture_pixels(source, frame)
                                       for frame in (37, 52, 67)]))
    for key in ("median", "spread", "saturation"):
        assert records[0]["broll"][key] == pytest.approx(expected[key], abs=0.002)
    decodes = [cmd for cmd in commands if "rawvideo" in cmd]
    assert decodes[-1][decodes[-1].index("-frames:v") + 1] == "3"


def test_prepared_heic_measures_one_frame(scene: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    photo = heic_quadrants(scene / "photo.heic")
    element = insert(scene, photo, 15, 45, identifier="broll-heic")
    element["motion"] = "still"
    lane(scene, [element], name="broll")
    original = ffmpeg.run

    def bounded(command: list[str], *, timeout_s: float, binary: bool = False) -> Any:
        if binary and "rawvideo" in command:
            timeout_s = 30
        return original(command, timeout_s=timeout_s, binary=binary)

    monkeypatch.setattr(ffmpeg, "run", bounded)
    out, result = render(scene, "heic")
    assert out.is_file()
    assert len(result.stills) == 1 and result.stills[0]["input"] == "broll-heic"
    assert len(result.broll_match) == 1 and "broll" in result.broll_match[0]
    command = next(cmd for cmd in result.commands
                   if "-f" in cmd and cmd[cmd.index("-f") + 1] == "rawvideo"
                   and any(arg.endswith("/frame.mkv") for arg in cmd)
                   and "-vf" in cmd)
    assert command[command.index("-frames:v") + 1] == "1"
    expression = command[command.index("-vf") + 1]
    assert expression.startswith("setparams=")
    assert not any(f"{name}=" in expression
                   for name in ("loop", "trim", "setpts", "fps", "tpad", "select"))


@pytest.mark.parametrize(("speed", "frames", "seconds", "timeout"), [
    (1.5, (607, 622, 637), 2, 60), (3, (615, 645, 675), 3.5, 72),
])
def test_clip_measurement_seeks_and_bounds_source_span(
    scene: Path, monkeypatch: pytest.MonkeyPatch, speed: float,
    frames: tuple[int, ...], seconds: float, timeout: float,
) -> None:
    source = synth.picture_source(scene / "long.mov", "color=gray:s=32x32:r=30,"
                                   "drawbox=c=red:t=fill:enable='lt(t,20.5)',"
                                   "drawbox=c=lime:t=fill:enable='gte(t,20.5)*lt(t,21)',"
                                   "drawbox=c=blue:t=fill:enable='gte(t,21)'", 1200)
    element = insert(scene, source, 15, 45, in_s=20, speed=speed)
    lane(scene, [element], name="broll")
    timeline = resolve_timeline(scene).require_valid()
    original = ffmpeg.run
    timeouts = []
    decode = match._decode
    samples = []

    def sampled(*args: Any, **kwargs: Any) -> np.ndarray:
        pixels = decode(*args, **kwargs)
        samples.append(pixels)
        return pixels

    def timed(command: list[str], *, timeout_s: float, binary: bool = False) -> Any:
        if binary and "rawvideo" in command and str(scene / element["media"]["path"]) in command:
            timeouts.append(timeout_s)
        return original(command, timeout_s=timeout_s, binary=binary)

    monkeypatch.setattr(ffmpeg, "run", timed)
    monkeypatch.setattr(match, "_decode", sampled)
    _, records, commands = measurement(scene, timeline)
    command = next(cmd for cmd in commands
                   if "rawvideo" in cmd and str(scene / element["media"]["path"]) in cmd)
    assert command.index("-ss") < command.index("-i")
    assert float(command[command.index("-ss") + 1]) == 20
    assert command.index("-t") < command.index("-i")
    assert float(command[command.index("-t") + 1]) == seconds
    assert "trim=" not in command[command.index("-vf") + 1]
    assert "-threads" not in command
    assert timeouts == [timeout]
    expected = match.measure(np.array([synth.picture_pixels(source, frame) for frame in frames]))
    for key in ("median", "spread", "saturation"):
        assert records[0]["broll"][key] == pytest.approx(expected[key], abs=0.002)
    def single_thread(command: list[str], *, timeout_s: float, binary: bool = False) -> Any:
        if binary and "rawvideo" in command:
            index = command.index("-i")
            command = [*command[:index], "-threads", "1", *command[index:]]
        return original(command, timeout_s=timeout_s, binary=binary)

    monkeypatch.setattr(ffmpeg, "run", single_thread)
    _, repeated, _ = measurement(scene, timeline, "single-thread")
    assert records == repeated
    assert len(samples) == 4
    np.testing.assert_array_equal(samples[0], samples[2])
    np.testing.assert_array_equal(samples[1], samples[3])


@pytest.mark.parametrize("source_name", ["cut.mp4", "broll-insert.mov"])
def test_measurement_failure_names_shot(scene: Path, monkeypatch: pytest.MonkeyPatch,
                                       source_name: str) -> None:
    original = ffmpeg.run

    def failed(command: list[str], *, timeout_s: float, binary: bool = False) -> Any:
        if binary and "rawvideo" in command and any(
            Path(arg).name == source_name for arg in command
        ):
            raise ProcessingError("ffmpeg", -1, "timed out after 60 s", code="ffmpeg_timeout")
        return original(command, timeout_s=timeout_s, binary=binary)

    monkeypatch.setattr(ffmpeg, "run", failed)
    with pytest.raises(assembly.PictureError,
                       match="broll-insert.*: colour measurement failed") as exc:
        measurement(scene, resolve_timeline(scene).require_valid())
    assert isinstance(exc.value.__cause__, ProcessingError)
    assert "set color_match false on this shot and run again" in (
        exc.value.suggested_action.lower())
