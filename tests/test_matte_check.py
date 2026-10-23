"""Native-luma synthetic media for cutout failures and creator review flags."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import test_matte_composite as composite_tests

from yap_studio import ffmpeg, receipt
from yap_studio.errors import ValidationError
from yap_studio.faces.model import VideoIdentity
from yap_studio.matte import check
from yap_studio.matte.check import CutoutThresholds, check_cutout
from yap_studio.matte.composite import CardOverlay, composite_video
from yap_studio.matte.model import Cleanup, MatteData

WIDTH, HEIGHT, FRAMES = 216, 384, 16
FPS = Fraction(10, 1)


def encode(path: Path, luma: np.ndarray, *, gray: bool = False) -> Path:
    """Encode exact native Y values or gray alpha from small test arrays."""
    count, height, width = luma.shape
    fmt = "gray" if gray else "yuv420p"
    raw = path.with_suffix(".raw")
    if gray:
        raw.write_bytes(luma.tobytes())
    else:
        chroma = np.full((count, height * width // 2), 128, dtype=np.uint8)
        raw.write_bytes(np.concatenate((luma.reshape(count, -1), chroma), axis=1).tobytes())
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", fmt,
         "-video_size", f"{width}x{height}", "-framerate", str(FPS), "-i", str(raw),
         "-c:v", "ffv1" if gray else "libx264", *([] if gray else ["-qp", "0"]),
         "-pix_fmt", fmt, str(path)], timeout_s=30,
    )
    raw.unlink()
    return path


def encode_layer(path: Path, alpha: np.ndarray) -> Path:
    """Encode a lossless card with exact, frame-varying alpha."""
    count, height, width = alpha.shape
    raw = path.with_suffix(".raw")
    luma = np.full((count, height * width), 80, dtype=np.uint8)
    chroma = np.full((count, height * width // 2), 128, dtype=np.uint8)
    raw.write_bytes(np.concatenate((luma, chroma, alpha.reshape(count, -1)), axis=1).tobytes())
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "yuva420p",
         "-video_size", f"{width}x{height}", "-framerate", str(FPS), "-i", str(raw),
         "-c:v", "ffv1", "-pix_fmt", "yuva420p", str(path)], timeout_s=30,
    )
    raw.unlink()
    return path


def write_matte(plate: Path, folder: Path, masks: np.ndarray) -> MatteData:
    """Write a measured document; its decoded file may deliberately be short."""
    folder.mkdir()
    matte = encode(folder / "matte.mkv", masks, gray=True)
    info = ffmpeg.probe(plate)
    identity = VideoIdentity(plate.name, receipt.sha256_file(plate), FRAMES, FPS, WIDTH, HEIGHT)
    assert info.video_frames == FRAMES
    data = MatteData(
        "measured", None, identity, {"name": "synthetic-test-matte"},
        Cleanup(median=False, low=0, high=255, feather_sigma=0),
        {"file": matte.name, "sha256": receipt.sha256_file(matte),
         "codec": "ffv1", "pix_fmt": "gray"},
        tuple(float(i / FPS) for i in range(FRAMES)), float(FRAMES / FPS),
        (0.1,) * FRAMES,
    )
    (folder / "matte.json").write_text(json.dumps(data.to_dict()), encoding="utf-8")
    return data


@dataclass
class CheckMedia:
    plate: Path
    matte: Path
    composite: Path
    card: CardOverlay
    luma: np.ndarray
    masks: np.ndarray


def make_media(tmp_path: Path, *, moving: bool = False, defect: str | None = None) -> CheckMedia:
    luma = np.full((FRAMES, HEIGHT, WIDTH), 40, dtype=np.uint8)
    for i in range(FRAMES):
        stripes = np.where(np.arange(64) % 12 < 6, 60, 180)
        luma[i, 80:208, 80:144] = stripes + (i * 4 if moving else 0)
    masks = np.zeros_like(luma)
    masks[:, 80:208, 80:144] = 255
    if defect == "dropout":
        masks[7, 144:176, 112:144] = 0
    elif defect == "flicker":
        masks[1::2, 32:80, 32:192] = 255
    elif defect == "large-flicker":
        masks[1::2, 32:224, 32:192] = 255
    plate = encode(tmp_path / "plate.mp4", luma)
    matte = tmp_path / "matte"
    write_matte(plate, matte, masks)
    layer = tmp_path / "layer.mkv"
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"color=red:size=160x192:rate={FPS},format=yuva420p", "-frames:v", str(FRAMES),
         "-c:v", "ffv1", "-pix_fmt", "yuva420p", str(layer)], timeout_s=30,
    )
    card = CardOverlay("hero", layer, 32, 32, 160, 192, 2, 14, "behind_head")
    out = composite_video(plate, [card], tmp_path / "composite.mp4",
                          matte_dir=matte, lossless=True).path
    return CheckMedia(plate, matte, out, card, luma, masks)


def make_ink_media(tmp_path: Path, *, covered: bool = True) -> CheckMedia:
    """A light panel with dark bars and a matte covering its edge or its ink."""
    luma = np.full((FRAMES, HEIGHT, WIDTH), 40, dtype=np.uint8)
    masks = np.zeros_like(luma)
    masks[:, 208:224, 80:112] = 255  # Plain bottom edge of the panel.
    if covered:
        masks[1, 80:88, 80:112] = 255  # Before the card's window.
        masks[5, 80:82, 80:112] = 255
        masks[7, 80:84, 80:112] = 128  # Worst in-window share: 128 of 512 ink pixels.
        masks[8, 80:88, 80:112] = 127  # Below the person's occlusion threshold.
        masks[14, 80:88, 80:112] = 255  # The half-open end is excluded.
    plate = encode(tmp_path / "plate.mp4", luma)
    matte = tmp_path / "matte"
    write_matte(plate, matte, masks)
    rgba = np.full((12, 192, 160, 4), 220, dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[:, 48:56, 48:80, :3] = 20
    rgba[:, 64:72, 48:80, :3] = 20
    raw = tmp_path / "layer.raw"
    raw.write_bytes(rgba.tobytes())
    layer = tmp_path / "layer.mkv"
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
         "-video_size", "160x192", "-framerate", str(FPS), "-i", str(raw),
         "-c:v", "ffv1", "-pix_fmt", "bgra", str(layer)], timeout_s=30,
    )
    raw.unlink()
    card = CardOverlay("g-skill", layer, 32, 32, 160, 192, 2, 14, "behind_head")
    out = composite_video(plate, [card], tmp_path / "composite.mp4",
                          matte_dir=matte, lossless=True).path
    return CheckMedia(plate, matte, out, card, luma, masks)


@pytest.fixture
def media(tmp_path: Path) -> CheckMedia:
    return make_media(tmp_path)


def results(report: check.CutoutReport) -> dict[str, check.CheckResult]:
    return {result.id: result for result in report.checks}


@pytest.mark.parametrize("reencode", [False, True])
def test_clean_and_crf12_pass(media: CheckMedia, tmp_path: Path, reencode: bool) -> None:
    out = media.composite
    if reencode:
        out = tmp_path / "crf12.mp4"
        ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(media.composite),
                    "-c:v", "libx264", "-crf", "12", str(out)], timeout_s=30)
    report = check_cutout(media.plate, media.matte, out, [media.card])
    assert report.status == "pass" and report.reason is None
    assert all(result.status == "pass" for result in report.checks)
    assert report.cards == report.behind == 1 and report.frames == FRAMES and report.fps == FPS
    document = report.to_dict()
    assert document["schema"] == "yap-studio/cutout-report@1" and document["fps"] == "10/1"
    assert document["thresholds"] == CutoutThresholds().to_dict()
    json.dumps(document, allow_nan=False)
    edges = results(report)["matte.doubled_edges"].data["cards"][0]
    assert edges["outside"]["share"] == edges["solid"]["share"] == 0
    assert edges["outside"]["frame"] == media.card.start_frame
    flicker = results(report)["matte.flicker"].data["cards"][0]
    assert flicker["pairs"] == 11
    assert flicker["median"] == flicker["p95"] == flicker["max"] == 0
    assert flicker["worst_pair"] == {"frames": [2, 3], "time": 0.3}
    occlusion = results(report)["matte.occlusion"].data["cards"][0]
    assert occlusion["share"] == occlusion["pixels"] == 0


@pytest.mark.parametrize("covered", [False, True])
def test_occlusion_reviews_text_but_allows_plain_panel_edge(tmp_path: Path, covered: bool) -> None:
    media = make_ink_media(tmp_path, covered=covered)
    report = check_cutout(media.plate, media.matte, media.composite, [media.card])
    result = results(report)["matte.occlusion"]
    assert report.status == result.status == ("review" if covered else "pass")
    assert all(other.status == "pass" for other in report.checks if other.id != result.id)
    record = result.data["cards"][0]
    frame = 7 if covered else media.card.start_frame
    assert record == {"card": "g-skill", "frame": frame, "time": float(frame / FPS),
                      "share": 0.25 if covered else 0.0, "pixels": 512}
    assert result.detail == (
        f"card g-skill hides {'25.0' if covered else '0.0'}% of its text "
        f"at frame {frame} (0:00.{'70' if covered else '20'}) (limit 2%)"
    )


@pytest.mark.parametrize("limit,status", [(0.25, "pass"), (0.249, "review")])
def test_occlusion_limit_is_strict(tmp_path: Path, limit: float, status: str) -> None:
    media = make_ink_media(tmp_path)
    report = check_cutout(media.plate, media.matte, media.composite, [media.card],
                          CutoutThresholds(occlusion_max=limit))
    assert report.status == results(report)["matte.occlusion"].status == status


def test_ink_uses_opaque_median_and_strict_luma_difference() -> None:
    layer = np.zeros((8, 8, 4), dtype=np.uint8)
    layer[6:] = (200, 200, 200, 255)
    layer[6, 0] = (160, 160, 160, 255)  # Difference exactly 40 is excluded.
    layer[6, 1] = (159, 159, 159, 250)
    layer[6, 2] = (0, 0, 0, 249)  # Transparent pixels do not affect the median or ink.
    layer[6, 3] = (0, 0, 0, 255)
    layer[7, 0] = (241, 241, 241, 255)
    layer[7, 1] = (240, 240, 240, 255)
    ink = check._ink(layer, CutoutThresholds().ink_diff)
    expected = np.zeros((8, 8), dtype=bool)
    expected[6, [1, 3]] = True
    expected[7, 0] = True
    np.testing.assert_array_equal(ink, expected)
    layer[..., 3] = 249
    assert not check._ink(layer, 40).any()


def test_behind_ink_uses_existing_luma_difference_and_glyph_alpha_threshold() -> None:
    layer = np.full((8, 8, 4), (200, 200, 200, 200), dtype=np.uint8)
    layer[0, 0] = (0, 0, 0, 127)  # Transparent corner does not affect the panel median.
    layer[3, 2] = (160, 160, 160, 128)  # Exact difference 40 is excluded.
    layer[3, 3] = (159, 159, 159, 128)
    layer[4, 2] = (240, 240, 240, 128)  # Light ink uses the same strict threshold.
    layer[4, 3] = (241, 241, 241, 128)
    expected = np.zeros((8, 8), dtype=bool)
    expected[3, 3] = expected[4, 3] = True
    np.testing.assert_array_equal(check._behind_ink(layer), expected)
    layer[..., 3] = 127
    assert not check._behind_ink(layer).any()


def test_behind_text_on_transparency_keeps_all_glyph_colours() -> None:
    layer = np.zeros((8, 8, 4), dtype=np.uint8)
    layer[2:6, 1:3] = (255, 255, 255, 255)
    layer[2:6, 5:7] = (20, 20, 20, 128)
    layer[1, 1] = (255, 255, 255, 127)
    np.testing.assert_array_equal(check._behind_ink(layer), layer[..., 3] >= 128)
    layer[2:6, 5:7] = 0
    np.testing.assert_array_equal(check._behind_ink(layer), layer[..., 3] >= 128)


def test_behind_blank_opaque_panel_has_no_ink() -> None:
    layer = np.full((8, 8, 4), (200, 200, 200, 255), dtype=np.uint8)
    assert not check._behind_ink(layer).any()


@pytest.mark.parametrize("part", ["solid", "outside"])
def test_shifted_person_or_outside_patch_fails(
    media: CheckMedia, tmp_path: Path, part: str
) -> None:
    rendered = composite_tests.decode(media.composite, WIDTH, HEIGHT)[0].copy()
    if part == "solid":
        rendered[6, 84:204, 84:140] = media.luma[6, 84:204, 90:146]
    else:
        rendered[6, 250:266, 0:16] = 200
    bad = encode(tmp_path / "bad.mp4", rendered)
    report = check_cutout(media.plate, media.matte, bad, [media.card])
    result = results(report)["matte.doubled_edges"]
    assert report.status == result.status == "fail"
    worst = result.data["cards"][0][part]
    assert worst["frame"] == 6 and worst["time"] == 0.6
    assert worst["share"] > CutoutThresholds().edge_share_max
    assert "frame 6 (0:00.60)" in result.detail


def test_active_front_card_is_excluded(media: CheckMedia, tmp_path: Path) -> None:
    layer = composite_tests.make_layer(tmp_path / "front.mkv", frames=FRAMES)
    front = CardOverlay("front", layer, 0, 272, 48, 48, 4, 10)
    out = composite_video(media.plate, [front, media.card], tmp_path / "front.mp4",
                          matte_dir=media.matte, lossless=True).path
    report = check_cutout(media.plate, media.matte, out, [front, media.card])
    assert report.status == "pass" and report.cards == 2 and report.behind == 1
    # Its rectangle must stop being excluded when its window ends.
    pixels = composite_tests.decode(out, WIDTH, HEIGHT)[0].copy()
    pixels[10, 272:320, :48] = 200
    bad = encode(tmp_path / "inactive-front.mp4", pixels)
    failure = results(check_cutout(media.plate, media.matte, bad, [front, media.card]))
    assert failure["matte.doubled_edges"].data["cards"][0]["outside"]["frame"] == 10


def test_front_card_over_solid_person_passes_but_real_shift_fails(
    media: CheckMedia, tmp_path: Path
) -> None:
    layer = composite_tests.make_layer(tmp_path / "over-person.mkv", frames=FRAMES)
    front = CardOverlay("front", layer, 80, 80, 48, 48, 4, 10)
    cards = [front, media.card]
    out = composite_video(media.plate, cards, tmp_path / "front-person.mp4",
                          matte_dir=media.matte, lossless=True).path
    report = check_cutout(media.plate, media.matte, out, cards)
    assert report.status == "pass"
    assert results(report)["matte.doubled_edges"].status == "pass"
    pixels = composite_tests.decode(out, WIDTH, HEIGHT)[0].copy()
    pixels[6, 144:200, 84:140] = media.luma[6, 144:200, 90:146]
    shifted = encode(tmp_path / "shifted-person.mp4", pixels)
    failure = results(check_cutout(media.plate, media.matte, shifted, cards))["matte.doubled_edges"]
    assert failure.status == "fail"
    assert failure.data["cards"][0]["solid"]["frame"] == 6
    pixels = composite_tests.decode(out, WIDTH, HEIGHT)[0].copy()
    pixels[10, 84:124, 84:124] = 200
    inactive = encode(tmp_path / "inactive-person.mp4", pixels)
    failure = results(check_cutout(media.plate, media.matte, inactive, cards))
    failure = failure["matte.doubled_edges"]
    assert failure.status == "fail"
    assert failure.data["cards"][0]["solid"]["frame"] == 10


@pytest.mark.parametrize("center_alpha", [0, 1, 128, 255])
def test_front_card_alpha_controls_shift_detection(
    media: CheckMedia, tmp_path: Path, center_alpha: int
) -> None:
    alpha = np.full((FRAMES, 48, 48), 255, dtype=np.uint8)
    alpha[2, 8:40, 8:40] = center_alpha  # Layer frame 2 is plate frame 6.
    layer = encode_layer(tmp_path / "front-center.mkv", alpha)
    front = CardOverlay("front", layer, 80, 80, 48, 48, 4, 10)
    cards = [front, media.card]
    out = composite_video(media.plate, cards, tmp_path / "front-center.mp4",
                          matte_dir=media.matte, lossless=True).path
    assert check_cutout(media.plate, media.matte, out, cards).status == "pass"
    pixels = composite_tests.decode(out, WIDTH, HEIGHT)[0].copy()
    pixels[6, 88:120, 88:120] = media.luma[6, 88:120, 94:126]
    shifted = encode(tmp_path / "center-shift.mp4", pixels)
    report = check_cutout(media.plate, media.matte, shifted, cards)
    edges = results(report)["matte.doubled_edges"]
    assert report.status == edges.status == ("fail" if center_alpha == 0 else "pass")
    solid = edges.data["cards"][0]["solid"]
    if center_alpha == 0:
        assert solid["frame"] == 6
        assert solid["share"] > CutoutThresholds().edge_share_max
    else:
        assert solid["share"] == 0


def test_behind_card_transparent_area_detects_outside_patch(
    media: CheckMedia, tmp_path: Path
) -> None:
    alpha = np.zeros((FRAMES, media.card.height, media.card.width), dtype=np.uint8)
    alpha[:, 16:48, 96:144] = 255  # A small panel leaves the rest of the box transparent.
    layer = encode_layer(tmp_path / "small-panel.mkv", alpha)
    card = replace(media.card, layer=layer)
    out = composite_video(media.plate, [card], tmp_path / "small-panel.mp4",
                          matte_dir=media.matte, lossless=True).path
    assert check_cutout(media.plate, media.matte, out, [card]).status == "pass"
    pixels = composite_tests.decode(out, WIDTH, HEIGHT)[0].copy()
    pixels[6, 48:64, 48:64] = 200
    bad = encode(tmp_path / "transparent-patch.mp4", pixels)
    report = check_cutout(media.plate, media.matte, bad, [card])
    edges = results(report)["matte.doubled_edges"]
    assert report.status == edges.status == "fail"
    outside = edges.data["cards"][0]["outside"]
    assert outside["frame"] == 6
    assert outside["share"] > CutoutThresholds().edge_share_max
    assert edges.data["cards"][0]["solid"]["share"] == 0


@pytest.mark.parametrize("large", [False, True])
def test_static_matte_flicker_reports_counts(tmp_path: Path, large: bool) -> None:
    media = make_media(tmp_path, defect="large-flicker" if large else "flicker")
    thresholds = CutoutThresholds(flicker_p95_max=50000) if large else CutoutThresholds()
    report = check_cutout(media.plate, media.matte, media.composite, [media.card], thresholds)
    result = results(report)["matte.flicker"]
    assert report.status == result.status == "fail"
    record = result.data["cards"][0]
    count = 160 * 192 - 64 * 128 if large else 160 * 48
    assert record["median"] == record["p95"] == record["max"] == count
    assert record["worst_pair"] == {"frames": [2, 3], "time": 0.3}
    assert results(report)["matte.doubled_edges"].status == "pass"
    assert results(report)["matte.hand_dropout"].status == "pass"


@pytest.mark.parametrize("moving", [False, True])
def test_one_frame_dropout_requires_motion(tmp_path: Path, moving: bool) -> None:
    media = make_media(tmp_path, moving=moving, defect="dropout")
    report = check_cutout(media.plate, media.matte, media.composite, [media.card])
    assert report.status == ("review" if moving else "pass")
    result = results(report)["matte.hand_dropout"]
    assert result.status == ("review" if moving else "pass")
    flags = result.data["flags"]
    if moving:
        assert len(flags) == 1
        assert flags[0]["card"] == "hero" and flags[0]["frame"] == 7 and flags[0]["time"] == 0.7
        assert flags[0]["dip"] == pytest.approx(0.125)
        assert flags[0]["motion"] > 2
        assert "0:00.70" in result.detail
    else:
        assert flags == []
    assert all(result.status == "pass" for result in report.checks
               if result.id != "matte.hand_dropout")


@pytest.mark.parametrize("kind", ["identity", "hash", "count", "document-count", "dimensions"])
def test_invalid_matte_is_refused(media: CheckMedia, tmp_path: Path, kind: str) -> None:
    document = media.matte / "matte.json"
    data = json.loads(document.read_text())
    if kind == "identity":
        other = encode(tmp_path / "other.mp4", np.zeros_like(media.luma))
        data["video"]["sha256"] = receipt.sha256_file(other)
    elif kind == "hash":
        with (media.matte / "matte.mkv").open("ab") as stream:
            stream.write(b"tampered")
    elif kind == "document-count":
        data["video"]["frames"] -= 1
        data["times"].pop()
        data["coverage"].pop()
    else:
        masks = media.masks[:-1] if kind == "count" else media.masks[:, :-2, :]
        replacement = encode(tmp_path / "replacement.mkv", masks, gray=True)
        (media.matte / "matte.mkv").write_bytes(replacement.read_bytes())
        data["matte"]["sha256"] = receipt.sha256_file(replacement)
    document.write_text(json.dumps(data))
    with pytest.raises(ValidationError) as caught:
        check_cutout(media.plate, media.matte, media.composite, [media.card])
    assert caught.value.parameter == {
        "identity": "matte.video.sha256", "hash": "matte.matte.sha256",
        "count": "matte.matte.frames", "document-count": "matte.video.frames",
        "dimensions": "matte.matte",
    }[kind]


@pytest.mark.parametrize("kind", ["none", "folder", "document", "file", "unavailable", "front"])
def test_skipped_never_passes(media: CheckMedia, tmp_path: Path, kind: str) -> None:
    matte: Path | None = media.matte
    card = media.card
    if kind == "none":
        matte = None
    elif kind == "folder":
        matte = tmp_path / "absent"
    elif kind == "document":
        (matte / "matte.json").unlink()
    elif kind == "file":
        (matte / "matte.mkv").unlink()
    elif kind == "front":
        card = replace(card, placement="front")
        matte = tmp_path / "absent"
    else:
        data = json.loads((matte / "matte.json").read_text())
        data.update(status="unavailable", reason="no segmentation backend", backend={},
                    matte=None, times=None, coverage=None, end_time=None)
        (matte / "matte.json").write_text(json.dumps(data))
    report = check_cutout(media.plate, matte, media.composite, [card])
    assert report.status == "skipped" and report.reason
    assert all(result.status == "skipped" and result.detail == report.reason
               for result in report.checks)
    if kind == "front":
        assert report.reason == "no card sits behind the person" and report.behind == 0
    elif kind == "unavailable":
        assert report.reason == "no segmentation backend"


@pytest.mark.parametrize("kind", ["count", "width", "height"])
def test_composite_geometry_is_refused(media: CheckMedia, tmp_path: Path, kind: str) -> None:
    pixels = media.luma
    if kind == "count":
        pixels = pixels[:-1]
    elif kind == "width":
        pixels = pixels[:, :, :-2]
    else:
        pixels = pixels[:, :-2, :]
    bad = encode(tmp_path / "bad.mp4", pixels)
    with pytest.raises(ValidationError) as caught:
        check_cutout(media.plate, None, bad, [media.card])
    assert caught.value.parameter == f"composite.{'video_frames' if kind == 'count' else kind}"


@pytest.mark.parametrize("field,value", [
    ("x", WIDTH), ("y", HEIGHT), ("end_frame", FRAMES + 1), ("start_frame", FRAMES),
    ("width", 158), ("placement", "back"),
])
def test_cards_are_validated_like_compositing(media: CheckMedia, field: str, value: Any) -> None:
    card = replace(media.card, **{field: value})
    if field == "start_frame":
        card = replace(card, end_frame=FRAMES + 1)
    with pytest.raises(ValidationError) as caught:
        check_cutout(media.plate, media.matte, media.composite, [card])
    assert caught.value.parameter == f"cards[0].{'layer' if field == 'width' else field}"


@pytest.mark.parametrize("placements", [
    ("behind_head",) * 5, ("front",) * 5, ("behind_head",) * 4 + ("front",),
])
def test_five_simultaneous_cards_are_refused_before_decoding(
    media: CheckMedia, monkeypatch: pytest.MonkeyPatch, placements: tuple[str, ...]
) -> None:
    cards = [
        replace(media.card, id=f"card-{index}", start_frame=3 + index, placement=placement)
        for index, placement in enumerate(placements)
    ]

    def unexpected_decode(*args: Any, **kwargs: Any):
        pytest.fail("overlapping cards must be refused before decoding")

    monkeypatch.setattr(ffmpeg, "run_stream", unexpected_decode)
    monkeypatch.setattr(check, "_decoded_count", unexpected_decode)
    with pytest.raises(ValidationError, match="5 cards active at frame 7") as caught:
        check_cutout(media.plate, media.matte, media.composite, cards)
    assert caught.value.parameter == "cards"
    assert "the timeline shows one graphic at a time; check fewer overlapping cards" in str(
        caught.value
    )


def test_four_simultaneous_cards_are_checked(media: CheckMedia) -> None:
    cards = [replace(media.card, id=f"card-{index}") for index in range(4)]
    report = check_cutout(media.plate, media.matte, media.composite, cards)
    assert report.status == "pass" and report.cards == report.behind == 4
    assert all(result.status == "pass" for result in report.checks)
    for name in ("matte.doubled_edges", "matte.flicker"):
        assert [record["card"] for record in results(report)[name].data["cards"]] == [
            card.id for card in cards
        ]


def test_card_limit_counts_half_open_windows(media: CheckMedia) -> None:
    cards = [replace(media.card, id=f"card-{index}", end_frame=8) for index in range(4)]
    cards.append(replace(media.card, id="later", start_frame=8))
    report = check_cutout(media.plate, media.matte, media.composite, cards)
    assert report.status == "pass" and report.cards == report.behind == 5


@pytest.mark.parametrize("name", list(CutoutThresholds().to_dict()))
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "2", None])
def test_thresholds_are_finite_positive_numbers(name: str, value: Any) -> None:
    with pytest.raises(ValidationError) as caught:
        CutoutThresholds.from_dict({name: value})
    assert caught.value.parameter == f"thresholds.{name}"


def test_thresholds_defaults_roundtrip_and_pixel_types() -> None:
    defaults = CutoutThresholds()
    assert defaults.to_dict() == {"edge_diff": 24, "edge_share_max": 0.0005, "flicker_px": 32,
                                  "flicker_p95_max": 1500, "flicker_frame_max": 10000,
                                  "dropout_iou_dip": 0.03, "dropout_motion": 2.0,
                                  "ink_diff": 40, "occlusion_max": 0.02}
    assert CutoutThresholds.from_dict(defaults.to_dict()) == defaults
    for name in ("edge_diff", "flicker_px", "ink_diff"):
        with pytest.raises(ValidationError):
            CutoutThresholds.from_dict({name: 2.5})
    with pytest.raises(ValidationError, match="unknown field"):
        CutoutThresholds.from_dict({"typo": 1})


def test_static_blocks_use_strict_limits_and_partial_borders() -> None:
    first = np.zeros((18, 34), dtype=np.uint8)
    second = first.copy()
    second[:16, :16] = 1  # Mean exactly 1 is not static.
    second[0, 16] = 8  # Maximum exactly 8 is not static.
    second[16:, 32:] = 0
    second[16, 32] = 3  # Partial 2x2 block, mean 0.75 and maximum 3.
    static = check._static(first, second)
    assert not static[:16, :32].any()
    assert static[16:, 32:].all()


def test_erosion_uses_full_frame_and_solid_255() -> None:
    matte = np.full((9, 9), 255, dtype=np.uint8)
    matte[4, 4] = 254
    solid = check._solid(matte)
    assert not solid[2:7, 2:7].any()
    assert solid[0, 0] and solid[-1, -1]


@pytest.mark.parametrize("front_start,failure_point", [
    (None, "_solid"), (0, "_static"), (4, "_static"),
])
def test_streams_are_trimmed_and_closed_on_error(
    media: CheckMedia, monkeypatch: pytest.MonkeyPatch, front_start: int | None, failure_point: str
) -> None:
    stream = ffmpeg.run_stream
    commands: list[list[str]] = []
    closed: list[list[str]] = []

    def tracked(command: list[str], **kwargs: Any):
        commands.append(command)
        try:
            yield from stream(command, **kwargs)
        finally:
            closed.append(command)

    monkeypatch.setattr(ffmpeg, "run_stream", tracked)
    cards = [media.card]
    expected = [
        (media.plate, "trim=start_frame=2:end_frame=14,extractplanes=y", "gray"),
        (media.composite, "trim=start_frame=2:end_frame=14,extractplanes=y", "gray"),
        (media.matte / "matte.mkv", "trim=start_frame=2:end_frame=14,format=gray", "gray"),
        (media.card.layer, "trim=start_frame=0:end_frame=12,format=rgba", "rgba"),
    ]
    if front_start is not None:
        front = replace(media.card, id="front", placement="front", start_frame=front_start,
                        end_frame=10)
        inactive = replace(front, id="inactive", start_frame=14, end_frame=16)
        cards.extend((front, inactive))
        expected.append((
            front.layer, f"trim=start_frame={max(2, front_start) - front_start}:"
            f"end_frame={10 - front_start},alphaextract", "gray",
        ))
    assert check_cutout(media.plate, media.matte, media.composite, cards).status == "pass"
    assert len(commands) == len(closed) == len(expected)
    assert sorted(commands) == sorted(closed)
    for command, (source, conversion, pix_fmt) in zip(commands, expected, strict=True):
        assert command == [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(source.resolve()),
            "-map", "0:v:0", "-an", "-vf", conversion,
            "-fps_mode", "passthrough", "-pix_fmt", pix_fmt,
            "-f", "rawvideo", "pipe:1",
        ]

    def broken(*args: Any, **kwargs: Any):
        raise RuntimeError("measurement interrupted")

    monkeypatch.setattr(check, failure_point, broken)
    commands.clear()
    closed.clear()
    with pytest.raises(RuntimeError, match="measurement interrupted"):
        check_cutout(media.plate, media.matte, media.composite, cards)
    assert len(commands) == len(closed) == (5 if front_start == 0 else 4)
    assert sorted(commands) == sorted(closed)


def test_short_window_marks_unmeasured_checks_skipped(media: CheckMedia) -> None:
    card = replace(media.card, start_frame=5, end_frame=6)
    report = check_cutout(media.plate, media.matte, media.composite, [card])
    assert report.status == "pass"
    assert results(report)["matte.flicker"].status == "skipped"
    assert results(report)["matte.hand_dropout"].status == "skipped"


def test_multiple_card_windows_and_half_open_end(media: CheckMedia, tmp_path: Path) -> None:
    first = replace(media.card, start_frame=2, end_frame=6)
    second = replace(media.card, id="second", start_frame=10, end_frame=14)
    out = composite_video(media.plate, [first, second], tmp_path / "two.mp4",
                          matte_dir=media.matte, lossless=True).path
    pixels = composite_tests.decode(out, WIDTH, HEIGHT)[0].copy()
    pixels[6:10, 250:266, :16] = 200  # End-exclusive gap is never compared.
    gap = encode(tmp_path / "gap.mp4", pixels)
    report = check_cutout(media.plate, media.matte, gap, [first, second])
    assert report.status == "pass" and report.cards == report.behind == 2
    edges = results(report)["matte.doubled_edges"].data["cards"]
    assert [record["outside"]["frame"] for record in edges] == [2, 10]
    flicker = results(report)["matte.flicker"].data["cards"]
    assert [record["pairs"] for record in flicker] == [3, 3]
    occlusion = results(report)["matte.occlusion"].data["cards"]
    assert [(record["card"], record["share"]) for record in occlusion] == [
        ("hero", 0.0), ("second", 0.0),
    ]
    pixels[13, 250:266, :16] = 200
    bad = encode(tmp_path / "second-bad.mp4", pixels)
    report = check_cutout(media.plate, media.matte, bad, [first, second])
    assert report.status == "fail"
    worst = results(report)["matte.doubled_edges"].data["cards"][1]["outside"]
    assert worst["frame"] == 13 and worst["time"] == 1.3


def test_frame_count_is_decoded_and_does_not_trust_probe_metadata(
    media: CheckMedia, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement = encode(tmp_path / "short.mkv", media.masks[:-1], gray=True)
    matte = media.matte / "matte.mkv"
    matte.write_bytes(replacement.read_bytes())
    document = media.matte / "matte.json"
    data = json.loads(document.read_text())
    data["matte"]["sha256"] = receipt.sha256_file(matte)
    document.write_text(json.dumps(data))
    probe = ffmpeg.probe

    def advertised_count(source: Path, *, count_frames: bool = True) -> ffmpeg.MediaInfo:
        info = probe(source, count_frames=count_frames)
        return replace(info, video_frames=FRAMES) if source == matte.resolve() else info

    monkeypatch.setattr(ffmpeg, "probe", advertised_count)
    with pytest.raises(ValidationError, match="decoded count") as caught:
        check_cutout(media.plate, media.matte, media.composite, [media.card])
    assert caught.value.parameter == "matte.matte.frames"


@pytest.mark.parametrize("kind", ["missing", "short", "alpha", "duplicate"])
def test_layer_and_id_validation(media: CheckMedia, tmp_path: Path, kind: str) -> None:
    card = media.card
    if kind == "missing":
        card = replace(card, layer=tmp_path / "absent.mkv")
    elif kind in ("short", "alpha"):
        layer = composite_tests.make_layer(tmp_path / "bad-layer.mkv", frames=1 if kind == "short"
                                            else FRAMES, alpha=kind != "alpha")
        card = replace(card, layer=layer, width=48, height=48)
    cards = [card, card] if kind == "duplicate" else [card]
    with pytest.raises(ValidationError) as caught:
        check_cutout(media.plate, media.matte, media.composite, cards)
    assert caught.value.parameter == ("cards[1].id" if kind == "duplicate" else "cards[0].layer")


def test_empty_card_list_is_skipped(media: CheckMedia) -> None:
    report = check_cutout(media.plate, media.matte, media.composite, [])
    assert report.status == "skipped" and report.cards == report.behind == 0
    assert report.reason == "no card sits behind the person"


def test_fail_has_priority_over_hand_review(tmp_path: Path) -> None:
    media = make_media(tmp_path, moving=True, defect="dropout")
    pixels = composite_tests.decode(media.composite, WIDTH, HEIGHT)[0].copy()
    pixels[7, 250:266, :16] = 200
    bad = encode(tmp_path / "bad.mp4", pixels)
    report = check_cutout(media.plate, media.matte, bad, [media.card])
    assert report.status == "fail"
    assert results(report)["matte.doubled_edges"].status == "fail"
    assert results(report)["matte.hand_dropout"].status == "review"
