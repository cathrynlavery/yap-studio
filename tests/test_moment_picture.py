"""Real synthetic layers, pilot windows and declared sticker freeze receipts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth

from yap_studio import ffmpeg, receipt
from yap_studio.matte import chroma
from yap_studio.matte.make import make_matte
from yap_studio.matte.model import Cleanup
from yap_studio.render import moments
from yap_studio.stages import assemble_picture as stage
from yap_studio.stages.assemble import verification_edl
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline


def decoded(path: Path) -> np.ndarray:
    data = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(path), "-pix_fmt", "yuv444p",
                       "-f", "rawvideo", "pipe:1"], binary=True, timeout_s=60).stdout
    return np.frombuffer(data, np.uint8).reshape(-1, 3, 192, 256).transpose(0, 2, 3, 1)


@pytest.fixture
def picture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    monkeypatch.setattr(stage, "WIDTH", 256)
    monkeypatch.setattr(stage, "HEIGHT", 192)
    # Lossless delivery separates graph correctness from inter-frame quantization.
    monkeypatch.setattr(stage, "CRF", 0)
    root = tmp_path / "project"
    (root / "cut").mkdir(parents=True)
    (root / "project.json").write_text(json.dumps({
        "schema": "yap-studio/project@1", "name": "moments", "style": "style.json",
        "takes": [{"id": "t1", "path": "mask.mov"}],
    }))
    (root / "style.json").write_text(json.dumps({
        "schema_version": 1, "moments": {"styles": list(moments.MOMENT_STYLES)},
    }))
    synth.picture_source(root / "cut/cut.mp4", "testsrc2=s=256x192:r=30", 60)
    mask = synth.picture_source(root / "mask.mov", "color=black:s=256x192:r=30,"
                                 "drawbox=x=88:y=50:w=80:h=100:c=white:t=fill", 60)
    monkeypatch.setattr(stage, "ensure_cut_matte", lambda *args, **kwargs: mask)
    synth.picture_source(root / "front.mov", "color=yellow:s=32x24:r=30", 60, alpha=True)
    synth.picture_source(root / "caption.mov", "color=lime:s=32x24:r=30", 60, alpha=True)
    titles = [{"id": "agent-front", "path": "front.mov", "x": 0, "y": 0, "width": 32,
               "height": 24, "start_frame": 0, "frames": 60, "layer": "front", "kind": "graphic"}]
    caption = {"path": "caption.mov", "x": 224, "y": 168, "width": 32, "height": 24,
               "start_frame": 0, "frames": 60}
    item = ResolvedElement("agent-moment", "moment", "agent", 12, 48, "front",
                           {"style": "color_pop"})
    front = ResolvedElement("agent-front", "graphic", "agent", 0, 60, "front", {})
    timeline = ResolvedTimeline((item, front), (), (), 60, 0)
    return root, mask, timeline, titles, caption


def test_assembly_windows_layers_determinism_and_pilot(picture: tuple) -> None:
    root, _, timeline, titles, caption = picture

    def render(name: str, selected: ResolvedTimeline, frames: tuple | None = None) -> Path:
        out = root / f"{name}/picture.mp4"
        stage.render_picture(root, out, work_dir=out.parent, timeline=selected,
                             title_layers=titles, caption_layer=caption, frames=frames)
        return out

    plain = render("plain", replace(timeline, elements=timeline.elements[1:]))
    full = render("full", timeline)
    repeated = render("repeated", timeline)
    pilot = render("pilot", timeline, (21, 38))
    assert receipt.sha256_file(full) == receipt.sha256_file(repeated)
    before, after = decoded(plain), decoded(full)
    np.testing.assert_array_equal(before[:12], after[:12])
    np.testing.assert_array_equal(before[48:], after[48:])
    assert np.abs(before[30].astype(float) - after[30]).mean() > 5
    np.testing.assert_allclose(after[30, :20, :28], before[30, :20, :28], atol=3)
    np.testing.assert_allclose(after[30, 172:, 228:], before[30, 172:, 228:], atol=3)
    np.testing.assert_array_equal(decoded(pilot), after[21:39])
    layer = next((root / "pilot").glob("moment-*.mkv"))
    assert ffmpeg.probe(layer).video_frames == 18


@pytest.mark.parametrize("style", moments.MOMENT_STYLES[:5])
def test_lossless_layers_and_pilot_equal_full(picture: tuple, style: str) -> None:
    root, mask, timeline, _, _ = picture
    item = replace(timeline.elements[0], payload={"style": style})
    full = moments.render_moments(root / "cut/cut.mp4", mask, [item], root / "full",
                                  accent_color="#FF0000")[0]
    pilot = moments.render_moments(root / "cut/cut.mp4", mask, [item], root / "pilot",
                                   accent_color="#FF0000", frames=(21, 38))[0]
    info = ffmpeg.probe(Path(full["path"]))
    assert info.video_codec == "ffv1" and info.pix_fmt == "yuv444p" and not info.has_audio
    np.testing.assert_array_equal(decoded(Path(full["path"]))[9:27], decoded(Path(pilot["path"])))
    assert (pilot["start_frame"], pilot["frames"]) == (21, 18)


@pytest.mark.parametrize("style", ["color_pop", "background_blur", "spotlight", "sticker_freeze"])
def test_chroma_moment_despills_person_and_preserves_live_endpoints(
    picture: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, style: str,
) -> None:
    root, _, timeline, titles, caption = picture
    green = synth.picture_source(tmp_path / "green.mp4", "color=0x30b050:s=256x192:r=30,"
                                  "drawbox=x=88:y=50:w=80:h=100:c=0xb9d5cc:t=fill,"
                                  "gblur=sigma=2", 60)
    cut = root / "cut/cut.mp4"
    cut.write_bytes(green.read_bytes())
    data = make_matte(cut, root / "matte", cleanup=Cleanup(
        median=False, low=0, high=255, feather_sigma=0, refine=False,
    ))
    assert data.backend["name"] == "chroma-key"
    monkeypatch.setattr(stage, "ensure_cut_matte",
                        lambda *args, **kwargs: root / "matte/matte.mkv")
    moment = replace(timeline.elements[0], payload={"style": style})
    timeline = replace(timeline, elements=(moment, *timeline.elements[1:]))
    out = root / "keyed/picture.mp4"
    stage.render_picture(root, out, work_dir=out.parent, timeline=timeline,
                         title_layers=titles, caption_layer=caption)
    pilot = root / "pilot/picture.mp4"
    stage.render_picture(root, pilot, work_dir=pilot.parent, timeline=timeline,
                         title_layers=titles, caption_layer=caption, frames=(21, 38))
    np.testing.assert_array_equal(decoded(pilot), decoded(out)[21:39])
    monkeypatch.setattr(chroma, "cutout_metadata", lambda _: None)
    plain = root / "plain/picture.mp4"
    stage.render_picture(root, plain, work_dir=plain.parent, timeline=timeline,
                         title_layers=titles, caption_layer=caption)
    before, after = (synth.picture_pixels(path, 30).astype(int) for path in (plain, out))
    np.testing.assert_allclose(after[85:110, 115:140], before[85:110, 115:140], atol=2)
    assert (before[40:160, 70:185, 1] - after[40:160, 70:185, 1]).max() > 3
    if style != "sticker_freeze":
        np.testing.assert_array_equal(decoded(out)[[12, 47]], decoded(plain)[[12, 47]])


@pytest.mark.parametrize("has_picture", [False, True])
def test_sticker_freeze_is_declared_to_final_checks(has_picture: bool) -> None:
    item = ResolvedElement("agent-freeze", "moment", "agent", 30, 66, "front",
                           {"style": "sticker_freeze"})
    timeline = ResolvedTimeline((item,), (), (), 120, 0)
    snapshot = SimpleNamespace(media={}, sources={}, title_layers=[], grade=None,
                               caption_layer={"x": 0, "y": 0, "width": 1, "height": 1,
                                              "start_frame": 0, "frames": 120},
                               timeline=timeline, spans={"ranges": []},
                               **({"picture": None} if has_picture else {}))
    edl = verification_edl(snapshot, base=Path("/tmp"))
    assert {"id": "agent-freeze", "start_frame": 30, "end_frame": 66} in edl["stills"]
    assert json.dumps(edl)


def test_moment_layer_uses_the_active_grade(picture: tuple) -> None:
    from yap_studio.grade import GradeSpec

    root, _, timeline, _, _ = picture
    item = replace(timeline.elements[0], payload={"style": "outline"})
    images = []
    for name, elements in (("plain", ()), ("moment", (item,))):
        out = root / name / "picture.mp4"
        stage.render_picture(root, out, work_dir=out.parent,
            timeline=replace(timeline, elements=elements), title_layers=[],
            caption_layer={"path": "caption.mov", "x": 224, "y": 168, "width": 32,
                           "height": 24, "start_frame": 0, "frames": 60},
            grade=GradeSpec("film"))
        images.append(decoded(out))
    np.testing.assert_array_equal(images[0][30, :24, :64], images[1][30, :24, :64])
