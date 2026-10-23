"""Catalog descriptions through word-anchored inserts, overlays and verified delivery."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth
import synth_broll
from test_assemble import project as project
from test_assemble_picture import overlay_image
from test_stage_captions import _disable_face_helper
from test_stage_graphics import graphic

from yap_studio import ffmpeg
from yap_studio.checks.caption_layout import read_layout_record
from yap_studio.media.catalog import add_paths, list_items
from yap_studio.media.describe import DESCRIPTION_SCHEMA, describe_item
from yap_studio.project.receipts import read_receipt
from yap_studio.render import hyperframes
from yap_studio.stages.assemble import assemble
from yap_studio.stages.captions import back_weak_captions, captions
from yap_studio.stages.graphics import graphics
from yap_studio.stages.music import music
from yap_studio.stages.timeline import add_timeline


@pytest.fixture(autouse=True)
def synthetic_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    _disable_face_helper(monkeypatch)
    if request.node.get_closest_marker("render"):
        return

    def render_layer(composition: Path, out: Path, *, frames: int, **kwargs: Any):
        source = (composition / "index.html").read_text()
        width = int(re.search(r'data-width="(\d+)"', source)[1])
        height = int(re.search(r'data-height="(\d+)"', source)[1])
        synth.picture_source(out, f"color=black@0:s={width}x{height}:r=30,format=rgba",
                             frames, alpha=True)
        return hyperframes.LayerResult(out, frames, width, height, frames / 30)

    monkeypatch.setattr(hyperframes, "render_layer", render_layer)


def _flow(project: Path, tmp_path: Path, *, real_layers: bool = False) -> None:
    folder = tmp_path / "media"
    folder.mkdir()
    image = overlay_image(folder / "photo.png")
    clip = synth_broll.clip(folder / "clip.mov", colour="lime", duration=1)
    added = add_paths([folder])
    assert len(added["added"]) == 2 and added["skipped"] == []
    for item in list_items():
        describe_item(item["id"], {"schema": DESCRIPTION_SCHEMA,
                                   "description": f"A {item['kind']} for the spoken idea."})
    items = {Path(item["locations"][0]["path"]).name: item for item in list_items()}
    assert items["photo.png"]["description"]["text"] == "A image for the spoken idea."
    assert items["clip.mov"]["description"]["text"] == "A video for the spoken idea."
    assert items["photo.png"]["duration_s"] is None
    assert items["clip.mov"]["duration_s"] == pytest.approx(1)
    broll = project / "broll"
    broll.mkdir()
    for media in (image, clip):
        shutil.copyfile(media, broll / media.name)
    elements = [{
        "id": "broll-insert", "kind": "insert", "start": {"word": "t1:0", "text": "First"},
        "hold_s": 0.5, "media": {"path": "broll/clip.mov", "sha256": items["clip.mov"]["sha256"]},
        "audio": "mute",
    }, {
        "id": "broll-photo", "kind": "overlay", "start": {"word": "t1:1", "text": "words"},
        "hold_s": 1, "media": {"path": "broll/photo.png", "sha256": items["photo.png"]["sha256"]},
        "placement": "box", "box": [60, 400, 400, 200], "fit": "cover", "motion": "none",
    }]
    fragment = project / "broll-fragment.json"
    fragment.write_text(json.dumps({"schema": "yap-studio/timeline@1", "elements": elements}))
    timeline = add_timeline(project, fragment, lane="broll").require_valid()
    assert [(item.id, item.start_frame, item.end_frame) for item in timeline.elements] == [
        ("broll-insert", 3, 18), ("broll-photo", 18, 48),
    ]
    if real_layers:
        card = graphic(start={"time": 2.0}, hold_s=1, lane="agent")
        agent = project / "graphics-fragment.json"
        agent.write_text(json.dumps({"schema": "yap-studio/timeline@1", "elements": [card]}))
        add_timeline(project, agent, lane="agent").require_valid()
    captions(project)
    graphics(project)
    music(project)
    result = assemble(project)
    if real_layers:
        # Bright synthetic B-roll exercises the automatic halo and saved box choice.
        assert result["caption_halo"]["added"] >= 1
        assert read_receipt(project / "final")["status"] == "verified"
        assert (project / result["out"]).with_suffix(".contrast.json").is_file()
        if read_receipt(project / "final")["caption_contrast"]["weak"]:
            assert back_weak_captions(project, "box")["backed"]
            graphics(project)
            music(project)
            result = assemble(project)
    final = project / result["out"]
    if real_layers:
        contrast = json.loads(final.with_suffix(".contrast.json").read_bytes())
        ratios = [caption["ratio"] for caption in contrast["captions"]]
        assert ratios and all(ratio is not None and ratio >= 3 for ratio in ratios), ratios
    assert read_receipt(project / "final")["status"] == "verified"
    for frame in (2, 3, 17, 18):
        box = (700, 700, 16, 16)
        actual = synth.picture_pixels(final, frame, box=box).mean(axis=(0, 1))
        expected = (0, 255, 0) if 3 <= frame < 18 else \
            synth.picture_pixels(project / "cut/cut.mp4", frame, box=box).mean(axis=(0, 1))
        np.testing.assert_allclose(actual, expected, atol=10)
    for frame, alpha in ((17, 0), (18, 0), (19, 1 / 6), (24, 1), (41, 1), (47, 1 / 6), (48, 0)):
        box = (200, 450, 16, 16)
        base = np.array((0, 255, 0)) if frame == 17 else \
            synth.picture_pixels(project / "cut/cut.mp4", frame, box=box).mean(axis=(0, 1))
        expected = alpha * np.array((255, 0, 0)) + (1 - alpha) * base
        actual = synth.picture_pixels(final, frame, box=box).mean(axis=(0, 1))
        np.testing.assert_allclose(actual, expected, atol=10)
        outside = (500, 450, 16, 16)
        expected_outside = np.array((0, 255, 0)) if frame == 17 else \
            synth.picture_pixels(project / "cut/cut.mp4", frame, box=outside).mean(axis=(0, 1))
        actual_outside = synth.picture_pixels(final, frame, box=outside).mean(axis=(0, 1))
        np.testing.assert_allclose(actual_outside, expected_outside, atol=10)
    layout = read_layout_record(final, width=1080, height=1920)
    (element,) = [item for item in layout.layout.elements if item.kind == "overlay"]
    assert (element.id, element.start_frame, element.end_frame) == ("broll-photo", 20, 47)
    assert layout.element_ids[element.id] == "broll-photo"
    assert layout.labels[element.id] == "photo.png"
    assert element.id not in layout.face_windows
    assert ffmpeg.probe(final).video_frames == 128


def test_catalog_to_word_anchored_insert_and_overlay_delivery(
    project: Path, tmp_path: Path,
) -> None:
    _flow(project, tmp_path)


@pytest.mark.render
def test_catalog_to_delivery_with_real_caption_and_graphics_renderers(
    project: Path, tmp_path: Path,
) -> None:
    _flow(project, tmp_path, real_layers=True)
