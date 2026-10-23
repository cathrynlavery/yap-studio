"""Synthetic contrast, frame alignment, records, and creator findings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from yap_studio import ffmpeg, paths
from yap_studio.checks.caption_contrast import (
    CONTRAST_SCHEMA,
    CONTRAST_SUFFIX,
    check_caption_contrast,
    measure_caption_contrast,
    read_contrast_record,
    word_contrast,
    write_contrast_record,
)
from yap_studio.checks.model import Thresholds
from yap_studio.errors import InputError
from yap_studio.receipt import sha256_file


def contrast_document(ratios: tuple[float | None, ...] = (1.2,)) -> dict:
    return {
        "schema": CONTRAST_SCHEMA, "measured": True, "reason": None,
        "method": {"sample_every": 3, "held_samples": 2, "surround_percentile": 75,
                   "opaque_alpha": 242},
        "captions": [
            {"id": f"caption-{i + 1}", "text": "And I said, no.",
             "start_frame": 1980 + i * 30, "end_frame": 2010 + i * 30,
             "ratio": ratio, "word": 2 if ratio is not None else None,
             "frame": 1989 + i * 30 if ratio is not None else None,
             "words": [{"index": 2, "text": "said,", "ratio": ratio,
                        "frame": 1989 + i * 30 if ratio is not None else None,
                        "samples": 5 if ratio is not None else 0}]}
            for i, ratio in enumerate(ratios)
        ],
    }


def test_contrast_ignores_inactive_caption_boxes(monkeypatch) -> None:
    from yap_studio.checks import caption_contrast as stage

    def frames(path, shape, filters, count):
        yield from (np.zeros(shape, dtype=np.uint8) for _ in range(count))

    monkeypatch.setattr(stage, "_frames", frames)
    monkeypatch.setattr(stage, "_word_contrast", lambda *a, **kw: (21, .1))
    layout = {"measured": True, "captions": [{
        "id": "caption", "text": "Go", "start_frame": 0, "end_frame": 5,
        "words": [{"text": "Go"}], "boxes": [{"x": 0, "y": 0, "w": 20, "h": 20,
                                                   "start_frame": 2, "end_frame": 4}],
    }]}
    result = measure_caption_contrast(Path("final"), Path("layer"), {
        "x": 0, "y": 0, "width": 32, "height": 32, "start_frame": 0, "frames": 5}, layout)
    assert result["captions"][0]["words"][0]["samples"] == 0


def _letters() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    layer = np.zeros((64, 96, 4), dtype=np.uint8)
    layer[24:40, 35:55] = 255
    return layer, (35, 24, 20, 16)


@pytest.mark.parametrize("background,expected", [(255, 1), (0, 21)])
def test_white_letters_use_the_final_picture(background: int, expected: float) -> None:
    layer, box = _letters()
    final = np.full((*layer.shape[:2], 3), background, dtype=np.uint8)
    assert word_contrast(final, layer, box) == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize("ink,ground,halo,expected", [
    ((255, 255, 255), (209, 231, 216), (0, 0, 0), "pass"),
    ((244, 209, 118), (255, 255, 245), (85, 85, 85), "weak"),
])
def test_soft_halo_uses_its_visible_ground(ink, ground, halo, expected) -> None:
    layer, box = _letters()
    layer[21:43, 32:58] = [*halo, 55]
    layer[22:42, 33:57] = [*halo, 145]
    layer[24:40, 35:55] = [*ink, 255]
    alpha = layer[..., 3:4] / 255
    final = np.rint(layer[..., :3] * alpha + np.array(ground) * (1 - alpha)).astype(np.uint8)
    ratio = word_contrast(final, layer, box)
    assert ratio >= 3 if expected == "pass" else ratio < 3


def test_plain_measure_keeps_the_halo_decision() -> None:
    layer, box = _letters()
    layer[21:43, 32:58] = [0, 0, 0, 55]
    layer[22:42, 33:57] = [0, 0, 0, 145]
    layer[24:40, 35:55] = 255
    alpha = layer[..., 3:4] / 255
    final = np.rint(layer[..., :3] * alpha + 240 * (1 - alpha)).astype(np.uint8)
    assert word_contrast(final, layer, box, halo_ground=False) < 3
    assert word_contrast(final, layer, box) >= 3


@pytest.mark.parametrize("held_ink,color", [("text", "#FFFFFF"), ("highlight", "#FF7F50")])
def test_measurement_keeps_the_held_samples_ink(monkeypatch, held_ink, color) -> None:
    from yap_studio.checks import caption_contrast

    layer, box = _letters()
    layers = [layer.copy() for _ in range(3)]
    layers[1][24:40, 35:55, :3] = list(bytes.fromhex(color[1:]))
    pictures = [np.full((64, 96, 3), value, np.uint8) for value in (255, 128, 0)]
    monkeypatch.setattr(caption_contrast, "_frames",
                        lambda path, *args: (item for item in
                                            (pictures if path.name == "final" else layers)))
    x, y, width, height = box
    result = measure_caption_contrast(
        Path("final"), Path("layer"),
        {"x": 0, "y": 0, "width": 96, "height": 64, "start_frame": 0, "frames": 8},
        {"measured": True, "captions": [{
            "id": "caption-0", "text": "Go", "start_frame": 0, "end_frame": 8,
            "words": [{"text": "Go"}], "boxes": [{"x": x, "y": y, "w": width, "h": height}],
        }]}, caption_style={"text_color": "#FFFFFF", "highlight_color": "#FF7F50",
                            "highlight_style": "color"},
    )
    caption = result["captions"][0]
    assert caption["frame"] == 4 and caption["ink"] == held_ink
    assert caption["words"][0]["ink"] == held_ink
    assert result["colors"] == {"text": "#FFFFFF", "highlight": "#FF7F50"}


@pytest.mark.parametrize("mode,text,accent,drawn,role", [
    ("color", "#FFFFFF", "#FF7F50", "#FF7F50", "highlight"),
    ("moving_block", "#FFFFFF", "#FF7F50", "#111111", "highlight"),
    ("highlight", "#FFFFFF", "#FF7F50", "#111111", "highlight"),
    ("pill", "#FFFFFF", "#FF7F50", "#111111", "highlight"),
    ("pill_karaoke", "#FFFFFF", "#FF7F50", "#111111", "highlight"),
    ("none", "#FFFFFF", "#FF7F50", "#FFFFFF", "text"),
    *[(mode, "#FFFFFF", "#FF7F50", "#111111", "text") for mode in
      ("neon_glow", "neon_accent", "gradient_fill", "glitch_rgb", "particle_burst", "texture")],
    ("neon_glow", "#111111", "#FF7F50", "#111111", "text"),
    ("neon_glow", "#888888", "#757575", "#FFFFFF", "text"),
    ("neon_glow", "#888888", "#777777", "#000000", "text"),
])
def test_measurement_classifies_the_ink_the_template_draws(
    monkeypatch, mode, text, accent, drawn, role,
) -> None:
    from yap_studio.checks import caption_contrast

    layer, box = _letters()
    layer[24:40, 35:55, :3] = list(bytes.fromhex(drawn[1:]))
    picture = np.zeros((64, 96, 3), np.uint8)
    monkeypatch.setattr(caption_contrast, "_frames",
                        lambda path, *args: (item for item in
                                            [picture if path.name == "final" else layer]))
    x, y, width, height = box
    result = measure_caption_contrast(
        Path("final"), Path("layer"),
        {"x": 0, "y": 0, "width": 96, "height": 64, "start_frame": 0, "frames": 2},
        {"measured": True, "captions": [{
            "id": "caption-0", "text": "Go", "start_frame": 0, "end_frame": 2,
            "words": [{"text": "Go"}], "boxes": [{"x": x, "y": y, "w": width, "h": height}],
        }]}, caption_style={"text_color": text, "highlight_color": accent,
                            "highlight_text_color": "#111111", "highlight_style": mode},
    )
    caption = result["captions"][0]
    assert caption["ink"] == caption["words"][0]["ink"] == role
    assert result["colors"][role] == drawn


def test_opaque_outline_has_inner_contrast_on_white() -> None:
    layer, box = _letters()
    layer[22:42, 33:57] = [0, 0, 0, 255]
    layer[24:40, 35:55] = 255
    assert word_contrast(np.full((64, 96, 3), 255, np.uint8), layer, box) >= 20


def test_gold_loses_to_bright_stripes_even_when_median_surround_is_dark() -> None:
    layer, box = _letters()
    layer[24:40, 35:55, :3] = [233, 196, 106]
    final = np.zeros((64, 96, 3), np.uint8)
    final[:, np.arange(96) % 5 >= 3] = 255
    assert np.median(final) == 0
    assert word_contrast(final, layer, box) < 2
    assert word_contrast(np.zeros_like(final), layer, box) > 10


def test_translucent_ground_is_read_from_final_not_treated_as_opaque() -> None:
    layer, box = _letters()
    layer[12:52, 23:67] = [0, 0, 0, 220]
    layer[24:40, 35:55] = 255
    alpha = layer[..., 3:4] / 255
    final = np.rint(layer[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    assert word_contrast(final, layer, box) > 10


@pytest.mark.parametrize("ground,expected", [(255, 1.0), (17, 18.88)])
def test_opaque_band_contrast_uses_its_ground(ground, expected) -> None:
    layer, box = _letters()
    layer[:] = [ground, ground, ground, 255]
    layer[24:40, 35:55] = 255
    final = np.zeros((*layer.shape[:2], 3), np.uint8)
    assert word_contrast(final, layer, box) == pytest.approx(expected, abs=0.02)


def test_too_few_opaque_or_ring_pixels_are_unmeasured() -> None:
    layer = np.zeros((20, 20, 4), np.uint8)
    layer[5:10, 5:10] = 255
    final = np.zeros((20, 20, 3), np.uint8)
    assert word_contrast(final, layer, (5, 5, 5, 5)) is None
    layer[:] = 255
    layer[0] = 0
    assert word_contrast(final, layer, (0, 0, 20, 20)) is None


@pytest.mark.parametrize("layer_start,caption_start,caption_end", [(4, 4, 5), (4, 4, 7)])
def test_one_frame_caption_samples_its_start(monkeypatch, layer_start, caption_start, caption_end):
    layer, box = _letters()
    calls = []

    def stream(command, **kwargs):
        calls.append(command)
        frame = caption_start if len(calls) == 1 else caption_start - layer_start
        assert f"select='eq(n,{frame})'" in command[command.index("-vf") + 1]
        yield (np.zeros((*layer.shape[:2], 3), np.uint8)
               if len(calls) == 1 else layer).tobytes()

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    x, y, width, height = box
    layout = {"measured": True, "captions": [{
        "id": "caption-0", "text": "Go", "start_frame": caption_start,
        "end_frame": caption_end, "words": [{"text": "Go"}],
        "boxes": [{"x": x, "y": y, "w": width, "h": height}],
    }]}
    result = measure_caption_contrast(Path("final"), Path("layer"),
                                      {"x": 0, "y": 0, "width": 96, "height": 64,
                                       "start_frame": layer_start, "frames": 1}, layout)
    caption = result["captions"][0]
    assert caption["ratio"] == 21 and caption["frame"] == caption_start
    assert caption["words"][0]["samples"] == 1 and len(calls) == 2


@pytest.mark.parametrize("bad_frames,expected,held_frame", [
    ((5,), 21, 8), ((5, 8), 1, 8),
])
def test_streamed_odd_crop_delayed_layer_and_held_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    bad_frames: tuple[int, ...], expected: float, held_frame: int,
) -> None:
    final = tmp_path / "final.mp4"
    layer = tmp_path / "layer.mov"
    raw = tmp_path / "layer.rgba"
    pixels = np.zeros((20, 41, 64, 4), np.uint8)
    pixels[[1, 4, 7, 10, 13, 17], 12:26, 16:30] = 255
    raw.write_bytes(pixels.tobytes())
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgba",
                "-video_size", "64x41", "-framerate", "30", "-i", str(raw),
                "-c:v", "qtrle", str(layer)], timeout_s=30)
    enable = "+".join(f"eq(n,{frame})" for frame in bad_frames)
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=96x72:r=30",
                "-vf", f"drawbox=c=white:t=fill:enable='{enable}'", "-frames:v", "24",
                "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", str(final)],
               timeout_s=30)
    placement = {"path": layer.name, "x": 7, "y": 11, "width": 64, "height": 41,
                 "start_frame": 4, "frames": 20}
    caption = {"id": "c1", "text": "Go", "start_frame": 4, "end_frame": 20,
               "boxes": [{"x": 23, "y": 23, "w": 14, "h": 14}],
               "words": [{"text": "Go", "start_s": 0.14, "end_s": 0.6}]}
    layout = {"measured": True, "reason": None, "captions": [caption, {
        **caption, "id": "c2", "start_frame": 20, "end_frame": 24,
    }]}
    calls = []
    original = ffmpeg.run_stream

    def tracked(command, **kwargs):
        calls.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run_stream", tracked)
    result = measure_caption_contrast(final, layer, placement, layout)
    assert result["measured"] and result["method"] == contrast_document()["method"]
    first, last = result["captions"]
    assert (first["ratio"], first["frame"], first["word"]) == (expected, held_frame, 0)
    assert first["words"] == [
        {"index": 0, "text": "Go", "ratio": expected, "frame": held_frame, "samples": 5},
    ]
    assert (last["ratio"], last["frame"], last["words"][0]["samples"]) == (21, 21, 1)
    assert len(calls) == 2
    assert all(command[command.index("-fps_mode") + 1] == "passthrough" for command in calls)
    assert "format=rgb24,crop=64:41:7:11" in calls[0][calls[0].index("-vf") + 1]


@pytest.mark.parametrize("which,extra", [(0, b""), (1, b""), (0, b"x"), (1, b"x")])
def test_incomplete_decodes_raise_clear_error(monkeypatch, which, extra) -> None:
    calls = 0

    def stream(*args, **kwargs):
        nonlocal calls
        index = calls
        calls += 1
        channels = 3 if index == 0 else 4
        yield bytes(32 * 33 * channels) if index != which else extra

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    layout = {"measured": True, "captions": [{
        "id": "c1", "text": "Go", "start_frame": 0, "end_frame": 2,
        "boxes": [{"x": 0, "y": 0, "w": 20, "h": 20}], "words": [{"text": "Go"}],
    }]}
    with pytest.raises(ValueError, match="decoded.*requested"):
        measure_caption_contrast(Path("final"), Path("layer"),
                                 {"x": 0, "y": 0, "width": 32, "height": 33,
                                  "start_frame": 0, "frames": 2}, layout)


def test_unmeasured_layout_and_no_captions_need_no_decode(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "run_stream", lambda *a, **k: pytest.fail("unexpected decode"))
    result = measure_caption_contrast(Path("unused"), Path("unused"), {},
                                      {"measured": False, "reason": "no glyphs"})
    assert not result["measured"]
    assert result["reason"] == "caption layout was not measured: no glyphs"
    result = measure_caption_contrast(Path("unused"), Path("unused"), {},
                                      {"measured": True, "captions": []})
    assert result["measured"] and result["captions"] == []


def test_caption_outside_layer_keeps_unmeasured_words(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg, "run_stream", lambda *a, **k: pytest.fail("unexpected decode"))
    result = measure_caption_contrast(Path("unused"), Path("unused"),
                                      {"x": 0, "y": 0, "width": 32, "height": 33,
                                       "start_frame": 10, "frames": 3},
                                      {"measured": True, "captions": [{
                                          "id": "c1", "text": "Go", "start_frame": 0,
                                          "end_frame": 3, "words": [{"text": "Go"}],
                                          "boxes": [{"x": 0, "y": 0, "w": 10, "h": 10}],
                                      }]})
    (caption,) = result["captions"]
    assert result["measured"]
    assert (caption["ratio"], caption["word"], caption["frame"]) == (None, None, None)
    assert caption["words"] == [{"index": 0, "text": "Go", "ratio": None,
                                  "frame": None, "samples": 0}]


@pytest.mark.parametrize("valid_neighbor", [False, True])
def test_word_box_mismatch_keeps_caption_without_sampling_it(monkeypatch, valid_neighbor) -> None:
    layer = np.zeros((41, 64, 4), np.uint8)
    layer[12:26, 16:30] = 255
    calls = []

    def stream(command, **kwargs):
        calls.append(command)
        assert valid_neighbor, "a caption with missing boxes must not be sampled"
        assert "select='eq(n,1)'" in command[command.index("-vf") + 1]
        yield (np.zeros((41, 64, 3), np.uint8) if len(calls) == 1 else layer).tobytes()

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    caption = {"id": "caption-1", "text": "Go now", "start_frame": 4, "end_frame": 8,
               "words": [{"text": "Go"}, {"text": "now"}],
               "boxes": [{"x": 16, "y": 12, "w": 14, "h": 14}]}
    captions = ([{**caption, "id": "caption-0", "text": "Go", "start_frame": 0,
                  "end_frame": 2, "words": [{"text": "Go"}]}] if valid_neighbor else [])
    result = measure_caption_contrast(Path("final"), Path("layer"),
                                      {"x": 0, "y": 0, "width": 64, "height": 41,
                                       "start_frame": 0, "frames": 8},
                                      {"measured": True, "captions": [*captions, caption]})
    assert result["measured"] and result["reason"] is None
    assert len(calls) == (2 if valid_neighbor else 0)
    missing = result["captions"][-1]
    assert (missing["ratio"], missing["word"], missing["frame"]) == (None, None, None)
    assert missing["words"] == [
        {"index": i, "text": word["text"], "ratio": None, "frame": None, "samples": 0}
        for i, word in enumerate(caption["words"])
    ]
    if valid_neighbor:
        assert result["captions"][0]["ratio"] == 21


def test_record_round_trip_and_media_binding(tmp_path: Path) -> None:
    media = tmp_path / "final.mp4"
    media.write_bytes(b"synthetic media")
    document = contrast_document()
    with paths.operation():
        assert read_contrast_record(media) is None
        custom = tmp_path / "custom.contrast.json"
        assert write_contrast_record(media, document, custom) == custom
        assert write_contrast_record(media, document) == media.with_suffix(CONTRAST_SUFFIX)
    assert read_contrast_record(media) == {**document, "media_sha256": sha256_file(media)}
    assert "media_sha256" not in document
    media.write_bytes(b"changed")
    with pytest.raises(InputError, match="media hash does not match"):
        read_contrast_record(media)


@pytest.mark.parametrize("damage", ["json", "schema", "type", "nan", "duplicate", "symlink"])
def test_corrupt_record_is_refused(tmp_path: Path, damage: str) -> None:
    media = tmp_path / "final.mp4"
    media.write_bytes(b"synthetic")
    record = media.with_suffix(CONTRAST_SUFFIX)
    document = contrast_document()
    if damage == "symlink":
        record.symlink_to(tmp_path / "missing.json")
    elif damage == "json":
        record.write_text("{")
    elif damage == "duplicate":
        record.write_text('{"measured":true,"measured":false}')
    else:
        if damage == "schema":
            document["schema"] = "wrong"
        elif damage == "type":
            document["captions"][0]["ratio"] = "1.2"
        else:
            document["captions"][0]["ratio"] = float("nan")
        record.write_text(json.dumps(document))
    with pytest.raises(InputError, match="missing file" if damage == "symlink" else "corrupt"):
        read_contrast_record(media)


@pytest.mark.parametrize("missing", ["ratio", "frame", "word", "start_frame", "words"])
def test_record_missing_measurement_fields_is_corrupt(tmp_path, missing) -> None:
    media = tmp_path / "final.mp4"
    media.write_bytes(b"synthetic")
    document = contrast_document()
    del document["captions"][0][missing]
    document["media_sha256"] = sha256_file(media)
    media.with_suffix(CONTRAST_SUFFIX).write_text(json.dumps(document))
    with pytest.raises(InputError, match="corrupt contrast record"):
        read_contrast_record(media)


@pytest.mark.parametrize("ratio,severity,display", [
    (3, None, None), (1.5, "warn", "1.5"), (2.9, "warn", "2.9"), (2.95, "warn", "2.9"),
    (1.49, "warn", "1.4"), (1.2, "warn", "1.2"), (None, None, None),
])
def test_check_boundaries_and_per_caption_wording(ratio, severity, display) -> None:
    findings, coverage = check_caption_contrast(contrast_document((ratio,)), fps=30,
                                                thresholds=Thresholds())
    assert coverage["unmeasured"] == int(ratio is None)
    if ratio is None:
        assert not coverage["ran"] and coverage["reason"] == "no caption word could be measured"
        assert coverage["note"] == "1 caption had no measurable word (first at 1:06.0)"
    else:
        assert coverage == {"ran": True, "reason": None, "captions": 1, "unmeasured": 0,
                            "flagged": int(severity is not None)}
    if severity is None:
        assert findings == []
        return
    (finding,) = findings
    assert finding.severity == severity and finding.time_s == 66.3
    prefix = 'Caption "And I said, no." at 1:06.3 is hard to read: "said," '
    expected = (f"has {display}:1 contrast with the picture behind it;" if ratio >= 1.5
                else f"nearly vanishes ({display}:1 against the picture behind it);")
    assert finding.message == prefix + expected + " spoken words need at least 3:1."
    assert finding.evidence == {
        "caption_id": "caption-1", "text": "And I said, no.", "word": "said,", "word_index": 2,
        "ratio": ratio, "frame": 1989, "start_frame": 1980, "end_frame": 2010,
        "warn_below": 3, "fail_below": 1.5,
    }
    assert finding.suggested_action == (
        "Back this phrase: yap-studio captions DIR --back shadow (a dark halo) or --back box "
        "(a box behind it), then yap-studio run DIR again."
    )


def test_contrast_coverage_reports_unmeasured_captions() -> None:
    document = contrast_document((4.0, None, None))
    document["captions"][1]["start_frame"] = 123
    findings, coverage = check_caption_contrast(document, fps=30, thresholds=Thresholds())
    assert findings == []
    assert coverage == {"ran": True, "reason": None, "captions": 3, "flagged": 0,
                        "unmeasured": 2,
                        "note": "2 captions had no measurable word (first at 0:04.1)"}


@pytest.mark.parametrize("backing,action", [
    ("shadow", "The halo is not enough here. Put a box behind this phrase: "
     "yap-studio captions DIR --back box, then yap-studio run DIR again."),
    ("box", "Even the box is not enough here; check this moment by eye."),
])
def test_contrast_action_matches_backing(backing, action) -> None:
    document = contrast_document()
    document["captions"][0].update(backing=backing, auto=backing == "shadow")
    findings, _ = check_caption_contrast(document, fps=30, thresholds=Thresholds())
    assert findings[0].severity == "warn" and findings[0].suggested_action == action


@pytest.mark.parametrize("field,value", [("backing", "none"), ("backing", 1),
                                       ("auto", "true")])
def test_record_rejects_invalid_backing_metadata(tmp_path, field, value) -> None:
    media = tmp_path / "final.mp4"
    media.write_bytes(b"synthetic")
    document = contrast_document()
    document["captions"][0][field] = value
    document["media_sha256"] = sha256_file(media)
    media.with_suffix(CONTRAST_SUFFIX).write_text(json.dumps(document))
    with pytest.raises(InputError, match="corrupt contrast record"):
        read_contrast_record(media)


def test_contrast_does_not_run_when_every_caption_is_unmeasured() -> None:
    findings, coverage = check_caption_contrast(contrast_document((None, None)), fps=30,
                                                thresholds=Thresholds())
    assert findings == [] and not coverage["ran"]
    assert coverage["reason"] == "no caption word could be measured"
    assert coverage["unmeasured"] == 2


@pytest.mark.parametrize("warn,label", [(3.0, "3"), (4.5, "4.5")])
def test_wording_uses_the_configured_warn_threshold(warn, label) -> None:
    findings, _ = check_caption_contrast(contrast_document((2.95,)), fps=30,
                                         thresholds=Thresholds(caption_contrast_warn=warn))
    assert findings[0].message.endswith(f"spoken words need at least {label}:1.")
    assert findings[0].evidence["warn_below"] == warn


@pytest.mark.parametrize("ratio,display,severity", [(2.95, "2.9", "warn"), (1.49, "1.4", "warn")])
def test_collapsed_ratios_floor_display_and_keep_exact_evidence(ratio, display, severity) -> None:
    findings, _ = check_caption_contrast(contrast_document((ratio,) * 11), fps=30,
                                         thresholds=Thresholds())
    (finding,) = findings
    assert finding.severity == severity
    assert finding.message.count(f'"said," {display}:1') == 5
    assert all(item["ratio"] == ratio for item in finding.evidence["worst"])
    assert finding.suggested_action == (
        "Most captions are weak on this footage, even with the halo. Put a box behind the weak "
        "phrases with yap-studio captions DIR --back box, or change the caption style "
        "for every phrase: a box ground "
        "(captions.background_mode box) or a stronger shadow (captions.shadow_opacity)."
    )


@pytest.mark.parametrize("count,total,collapsed", [(10, 10, False), (11, 11, True),
                                                  (11, 33, False), (12, 33, True)])
def test_collapse_rule_and_worst_evidence(count, total, collapsed) -> None:
    ratios = tuple(1.0 if i % 2 == 0 else 2.9 for i in range(count)) + (4.0,) * (total - count)
    findings, coverage = check_caption_contrast(contrast_document(ratios), fps=30,
                                                thresholds=Thresholds())
    assert coverage["captions"] == total and coverage["flagged"] == count
    assert len(findings) == (1 if collapsed else count)
    if collapsed:
        (finding,) = findings
        failing = (count + 1) // 2
        assert finding.severity == "warn" and finding.time_s == 66.3
        assert finding.message.startswith(
            f"{count} of {total} captions are hard to read against the picture behind them "
            f'({failing} nearly vanish). Worst: 1:06.3 "said," 1.0:1; '
        )
        assert len(finding.evidence["worst"]) == min(10, count)
        assert finding.message.count('"said,"') == 5
        assert finding.evidence["worst"][0] == {
            "caption_id": "caption-1", "text": "And I said, no.", "word": "said,",
            "time_s": 66.3, "ratio": 1.0,
        }
        assert finding.suggested_action.startswith("Most captions are weak on this footage,")


def test_collapsed_warning_and_coverage_reasons() -> None:
    findings, _ = check_caption_contrast(contrast_document((2.0,) * 11), fps=30,
                                         thresholds=Thresholds())
    assert len(findings) == 1 and findings[0].severity == "warn"
    for record, reason in [
        ({**contrast_document(), "measured": False, "reason": "decode failed"}, "decode failed"),
        (contrast_document(()), "no captions in the layout"),
    ]:
        assert check_caption_contrast(record, fps=30, thresholds=Thresholds()) == (
            [], {"ran": False, "reason": reason, "unmeasured": 0},
        )


def highlight_document(ratios: tuple[float, ...] = (2.12,)) -> dict:
    document = contrast_document(ratios)
    document["colors"] = {"text": "#FFFFFF", "highlight": "#FF7F50"}
    for caption in document["captions"]:
        caption["ink"] = caption["words"][0]["ink"] = "highlight"
        caption["words"].append({"index": 0, "text": "And", "ratio": 4.0,
                                 "frame": caption["frame"], "samples": 5, "ink": "text"})
    return document


@pytest.mark.parametrize("inks", ["highlight", "mixed", "old"])
def test_collapsed_warning_names_only_known_highlight_weakness(inks) -> None:
    document = highlight_document((2.12,) * 11 + (4.0,))
    if inks == "mixed":
        caption = document["captions"][0]
        caption["ink"] = caption["words"][0]["ink"] = "text"
    elif inks == "old":
        del document["colors"]
        for caption in document["captions"]:
            for item in [caption, *caption["words"]]:
                del item["ink"]
    findings, coverage = check_caption_contrast(document, fps=30, thresholds=Thresholds())
    (finding,) = findings
    assert finding.severity == "warn" and coverage["flagged"] == 11
    if inks == "highlight":
        assert finding.message == (
            "The highlighted word is hard to read on this footage: in 11 of 12 captions it "
            "drops to 2.1:1 (#FF7F50 against the picture behind it). "
            "The other words read at 3:1 or better."
        )
        assert finding.suggested_action == (
            "Choose a lighter or darker highlight colour on the style page, or a highlight "
            "style with its own ground such as Word background. If needed, put a box behind "
            "the weak phrases: yap-studio captions DIR --back box, then yap-studio run DIR again."
        )
        assert finding.evidence["worst"][0]["ink"] == "highlight"
    else:
        assert finding.message.startswith(
            "11 of 12 captions are hard to read against the picture behind them "
            "(0 nearly vanish). Worst: "
        )
        assert finding.suggested_action.startswith("Most captions are weak on this footage,")


def test_per_phrase_warning_names_the_weak_highlight() -> None:
    findings, _ = check_caption_contrast(highlight_document(), fps=30, thresholds=Thresholds())
    (finding,) = findings
    assert finding.severity == "warn"
    assert finding.message == (
        'The highlighted word "said," in caption "And I said, no." at 1:06.3 is hard to read: '
        "2.1:1 (#FF7F50 against the picture behind it); spoken words need at least 3:1."
    )
    assert finding.suggested_action.startswith("Choose a lighter or darker highlight colour")
    assert finding.evidence["ink"] == "highlight"


@pytest.mark.parametrize("text_ratio", [2.5, None])
def test_highlight_warning_does_not_claim_unproven_text_contrast(text_ratio) -> None:
    document = highlight_document((2.12,) * 11)
    document["captions"][0]["words"][1]["ratio"] = text_ratio
    findings, _ = check_caption_contrast(document, fps=30, thresholds=Thresholds())
    assert findings[0].message.startswith("The highlighted word is hard to read")
    assert "The other words read" not in findings[0].message


def test_ink_metadata_round_trips_in_the_existing_schema(tmp_path) -> None:
    media = tmp_path / "final.mp4"
    media.write_bytes(b"synthetic")
    document = highlight_document()
    with paths.operation():
        write_contrast_record(media, document)
    assert read_contrast_record(media) == {**document, "media_sha256": sha256_file(media)}
    assert document["schema"] == "yap-studio/caption-contrast@1"


@pytest.mark.parametrize("target", ["caption", "word"])
def test_record_rejects_invalid_ink_metadata(tmp_path, target) -> None:
    media = tmp_path / "final.mp4"
    media.write_bytes(b"synthetic")
    document = highlight_document()
    caption = document["captions"][0]
    (caption if target == "caption" else caption["words"][0])["ink"] = "coral"
    document["media_sha256"] = sha256_file(media)
    media.with_suffix(CONTRAST_SUFFIX).write_text(json.dumps(document))
    with pytest.raises(InputError, match="corrupt contrast record"):
        read_contrast_record(media)
