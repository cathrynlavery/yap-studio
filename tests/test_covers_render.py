"""Pinned cover composition checks."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_matte_chroma import VISION_BACKEND, mint_screen_clip

from yap_studio import ffmpeg
from yap_studio.covers import render as cover_render
from yap_studio.covers.render import _default_matte, _rgba, prepare_assets, render_cover
from yap_studio.errors import ValidationError
from yap_studio.matte.make import make_matte
from yap_studio.style import resolve


@pytest.mark.parametrize("backend", ["chroma-key", "vision-person-segmentation"])
def test_cover_cutout_despill_respects_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    plate = mint_screen_clip(tmp_path)

    def matte(frame: Path, folder: Path, **kwargs):
        data = make_matte(frame, folder, **kwargs)
        assert data.backend["name"] == "chroma-key"
        if backend != "chroma-key":
            data = replace(data, backend=VISION_BACKEND.copy())
            (folder / "matte.json").write_text(json.dumps(data.to_dict()), encoding="utf-8")
        return data

    monkeypatch.setattr(cover_render, "make_matte", matte)
    assets = prepare_assets(plate, 0, tmp_path / "prepared", "photo")
    assert assets.cutout is not None and assets.matte is not None
    plain = tmp_path / "plain.png"
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-i", str(assets.frame), "-i", str(assets.matte),
        "-filter_complex", "[0:v]format=rgb24[photo];[1:v]scale=1080:1920,"
        "format=gray[alpha];[photo][alpha]alphamerge,format=rgba[out]",
        "-map", "[out]", "-frames:v", "1", "-pix_fmt", "rgba", str(plain),
    ], timeout_s=60)
    before, after = _rgba(plain).astype(int), _rgba(assets.cutout).astype(int)
    if backend != "chroma-key":
        np.testing.assert_array_equal(after, before)
        return
    original = _rgba(assets.frame).astype(int)
    np.testing.assert_allclose(after[900:1060, 460:620, :3],
                               original[900:1060, 460:620, :3], atol=2)
    np.testing.assert_array_equal(after[:, :, 3], before[:, :, 3])
    assert (before[900:1060, 760:800, 1] - after[900:1060, 760:800, 1]).max() > 3


def _coral_style():
    style = resolve("preset:editorial_serif")
    data = copy.deepcopy(style.data)
    data["brand"] = {"accent_color": "#F34E3F"}
    return replace(style, data=data)


@pytest.mark.render
def test_second_line_accent_selects_vignette(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = np.full((1920, 1080, 4), 155, dtype=np.uint8)
    frame[:, :, 3] = 255
    monkeypatch.setattr(cover_render, "_rgba", lambda _path: frame)
    assets = cover_render.CoverAssets(Path("unused.png"), face_box=(450, 700, 300, 350))
    config = cover_render._config(assets, _coral_style(), "photo", {
        "kicker": "Babysitter rules", "headline": "Their screen time is *my* time"})
    headlines = [b for b in config["blocks"] if b["role"] == "headline"]
    assert len(headlines) >= 2
    assert headlines[0]["accent_start"] < 0
    assert any(b["accent_start"] >= 0 for b in headlines[1:])
    assert config["vignette"] >= 0.7
    assert min(config["predicted_contrast"].values()) >= 3


@pytest.mark.render
@pytest.mark.parametrize("look", ("photo", "number", "hook", "before-after"))
def test_four_looks_render(tmp_path: Path, look: str) -> None:
    from synth import cover_assets

    assets = cover_assets(tmp_path)
    text = {"photo": {"headline": "A *better* night"},
            "number": {"number": "5", "label": "better nights"},
            "hook": {"headline": "A *better* night", "label": "A real result"},
            "before-after": {"before": "8 hours", "after": "2 hours"}}[look]
    style = resolve("preset:editorial_serif")
    result = render_cover(tmp_path, assets, style, look, text)
    assert result.png.is_file()
    assert result.checks["geometry"]["passed"]
    assert result.checks["contrast"]["passed"]
    assert result.checks["geometry"]["text_box"][1] >= 280
    assert result.checks["geometry"]["text_box"][3] <= 1640
    assert result.checks["blocks"]
    for block in result.checks["blocks"]:
        minimum = 3 if block["role"] in {"headline", "number", "after"} else 4.5
        assert block["contrast_p10"] >= minimum
        assert block["measured_box"]
    if look != "number":
        assert result.checks["contrast"]["accent_pixels"] > 0
    else:
        assert result.checks["speaker_clearance"]["overlap_px"] == 0
        actual = _rgba(result.png)[0, 0, :3].astype(int)
        expected = np.array([int(style.data["graphics"]["paper_color"][i:i + 2], 16)
                             for i in (1, 3, 5)])
        assert np.max(np.abs(actual - expected)) <= 4


@pytest.mark.render
def test_number_refuses_text_over_wide_speaker(tmp_path: Path) -> None:
    from synth import cover_assets

    sample = cover_assets(tmp_path)
    ffmpeg.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                "color=c=black:s=1080x1920:r=30", "-vf",
                "drawbox=x=100:y=350:w=860:h=1118:color=white:t=fill,"
                "drawbox=x=0:y=1400:w=1080:h=520:color=white:t=fill,format=gray",
                "-frames:v", "1", str(sample.matte)], timeout_s=120)
    ffmpeg.run(["ffmpeg", "-y", "-i", str(sample.frame), "-i", str(sample.matte),
                "-filter_complex", "[0:v]format=rgb24[p];[1:v]format=gray[a];"
                "[p][a]alphamerge,format=rgba[out]", "-map", "[out]",
                "-frames:v", "1", str(sample.cutout)], timeout_s=120)
    assets = replace(sample, face_box=(100, 350, 860, 1118))
    with pytest.raises(ValidationError, match="number text would cover the speaker") as error:
        render_cover(tmp_path, assets, resolve("preset:editorial_serif"), "number",
                     {"number": "5", "label": "better nights"})
    assert error.value.parameter == "speaker"
    assert not (tmp_path / "cover.png").exists()


@pytest.mark.render
def test_headline_outside_crop_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from synth import cover_assets

    original = cover_render._config
    def moved(*args):
        config = original(*args)
        next(block for block in config["blocks"] if block["role"] == "headline")["y"] = 180
        return config
    monkeypatch.setattr(cover_render, "_config", moved)
    with pytest.raises(ValidationError, match="3:4"):
        render_cover(tmp_path, cover_assets(tmp_path), resolve("preset:editorial_serif"),
                     "photo", {"headline": "A *better* night"})
    assert not (tmp_path / "cover.png").exists()


@pytest.mark.render
def test_one_frame_lossless_matte_is_readable(tmp_path: Path) -> None:
    frame = tmp_path / "frame.png"
    ffmpeg.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                "testsrc2=size=1080x1920:rate=30", "-frames:v", "1", str(frame)],
               timeout_s=120)
    matte = _default_matte(frame, tmp_path)
    assert matte is not None and matte.is_file()
    assert ffmpeg.probe(tmp_path / "one-frame.mov", count_frames=True).video_frames == 1


@pytest.mark.render
def test_assets_accept_injected_matte(tmp_path: Path) -> None:
    from synth import cover_assets

    sample = cover_assets(tmp_path)
    cut = tmp_path / "cut.mov"
    ffmpeg.run(["ffmpeg", "-y", "-loop", "1", "-framerate", "30", "-i",
                str(sample.frame), "-frames:v", "1", "-c:v", "rawvideo", "-pix_fmt",
                "bgra", str(cut)], timeout_s=120)
    assets = prepare_assets(cut, 0, tmp_path / "prepared", "number",
                            matte_provider=lambda _frame, _work: sample.matte,
                            face_box=sample.face_box)
    assert assets.cutout is not None and assets.cutout.is_file()
    assert _rgba(assets.cutout)[0, 0, 3] == 0


@pytest.mark.render
def test_photo_inset_clears_selfie_head(tmp_path: Path) -> None:
    from synth import cover_assets

    sample = cover_assets(tmp_path)
    assets = replace(sample, inset=sample.frame, face_box=(450, 885, 500, 385))
    result = render_cover(tmp_path, assets, resolve("preset:editorial_serif"), "photo",
                          {"headline": "A *better* night", "inset": str(sample.frame)})
    assert result.checks["geometry"]["passed"]
    inset = result.checks["inset"]
    x, y, width, height = inset["box"]
    assert .7 <= inset["scale"] < 1
    assert 60 <= x and x + width <= 1020
    assert 280 <= y and y + height <= 1640
    assert np.any(np.all(_rgba(result.png)[y:y + height, x:x + width, :3] > 248,
                         axis=2))


@pytest.mark.render
def test_second_line_accent_prediction_matches_render(tmp_path: Path) -> None:
    frame = tmp_path / "gray.png"
    ffmpeg.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                "color=c=0x999999:s=1080x1920:r=30", "-frames:v", "1", str(frame)],
               timeout_s=120)
    assets = cover_render.CoverAssets(frame, face_box=(450, 700, 300, 350))
    values = {"kicker": "Babysitter rules",
              "headline": "Their screen time is *my* time"}
    style = _coral_style()
    config = cover_render._config(assets, style, "photo", values)
    result = render_cover(tmp_path, assets, style, "photo", values)
    assert config["vignette"] >= 0.7
    for index, block in enumerate(result.checks["blocks"]):
        assert abs(block["contrast_p10"] - config["predicted_contrast"][str(index)]) < .5


@pytest.mark.render
def test_before_after_keeps_face_visible(tmp_path: Path) -> None:
    from synth import cover_assets

    assets = cover_assets(tmp_path)
    result = render_cover(tmp_path, assets, _coral_style(), "before-after",
                          {"before": "All day", "after": "2 hours", "label": "of screen time"})
    original = _rgba(assets.frame)[850:1050, 500:750, :3].astype(np.float64)
    visible = _rgba(result.png)[850:1050, 500:750, :3].astype(np.float64)
    def luminance(pixels):
        rgb = pixels / 255
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        return np.mean(linear @ np.array([.2126, .7152, .0722]))
    assert luminance(visible) / luminance(original) >= .4
    assert result.checks["darkening"]["base_rgb_scale"] >= .7
    assert result.checks["darkening"]["gradient_strength"] > 0
