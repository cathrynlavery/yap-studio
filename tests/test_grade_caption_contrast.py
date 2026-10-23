"""Real caption backing stays above the grade and contrast sees the graded picture."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import synth
from test_captions_render import _style
from test_cut import make_project, time_segment, write_edit, write_json

from yap_studio import ffmpeg, receipt
from yap_studio.captions import phrase_keys
from yap_studio.checks.caption_contrast import measure_caption_contrast
from yap_studio.faces import detect
from yap_studio.grade import GradeSpec, apply_rgb
from yap_studio.project.receipts import StageRun, read_receipt
from yap_studio.project.status import ProjectContext, ingest_inputs
from yap_studio.stages import assemble
from yap_studio.stages.captions import caption_snapshot, captions
from yap_studio.stages.cut import cut
from yap_studio.stages.graphics import graphics
from yap_studio.stages.music import music


@pytest.mark.render
def test_film_preserves_caption_backing_and_measures_graded_picture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "Synthetic picture has no face")
    voice = synth.loudness_master(tmp_path / "voice.wav", frames=72)
    source = tmp_path / "source.mov"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=0x9b9b96:s=180x320:r=30:d=2.4", "-i", str(voice),
        "-vf", "scale=in_color_matrix=bt601:out_color_matrix=bt709", "-frames:v", "72",
        "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-pix_fmt", "yuv444p",
        "-colorspace", "bt709", "-color_trc", "bt709", "-color_primaries", "bt709",
        "-c:a", "pcm_f32le", str(source),
    ], timeout_s=120)
    root = make_project(tmp_path / "project", [source], word_rows=[[
        ("Backed", 0.1, 0.3), ("words.", 0.3, 0.7),
        ("Plain", 1.1, 1.3), ("words.", 1.3, 1.8),
    ]])
    style = _style(highlight="none", size_px=76, text_color="#FFFFFF", shadow_opacity=0,
                   background_mode="none", background_color="#152439", background_opacity=1)
    write_json(root / "style.json", style.data)
    project = json.loads((root / "project.json").read_bytes())
    project.update(style="style.json", asr={"engine": "none"}, music={"mode": "none"},
                   audio={"mode": "preserve"}, grade={"look": "none"})
    write_json(root / "project.json", project)
    previous = read_receipt(root / "ingest")
    with StageRun(root, "ingest") as run:
        for name, digest in ingest_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        run.finish()
    write_edit(root, [time_segment("s1", "t1", 0, 2.4)], end_hold_s=0)
    cut(root)
    phrases = caption_snapshot(ProjectContext(root)).phrases
    assert len(phrases) == 2
    keys = phrase_keys([phrase.word_ids for phrase in phrases])
    project["caption_backing"] = dict(zip(keys, ("box", "none"), strict=True))
    write_json(root / "project.json", project)
    placement = captions(root)["layer"]
    graphics(root)
    music(root)
    layer = root / placement["path"]
    layer_sha256 = receipt.sha256_file(layer)
    layout = json.loads((root / "captions/layout.json").read_bytes())
    assert layout["measured"]
    first = layout["captions"][0]
    frame = (first["start_frame"] + first["end_frame"]) // 2
    rgba = ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(layer), "-vf",
        f"select='eq(n,{frame - placement['start_frame']})',format=rgba", "-frames:v", "1",
        "-f", "rawvideo", "pipe:1",
    ], timeout_s=120, binary=True).stdout
    rgba = np.frombuffer(rgba, np.uint8).reshape(placement["height"], placement["width"], 4)
    opaque = (rgba[..., 3] == 255) & np.all(rgba[..., :3] == (21, 36, 57), axis=-1)
    # Keep clear of the box edge and glyphs, including chroma and encoder ringing.
    backing = np.logical_and.reduce([
        np.roll(opaque, (dy, dx), axis=(0, 1))
        for dy in (-4, 0, 4) for dx in (-4, 0, 4)
    ])
    for box in first["boxes"]:
        x, y = box["x"] - placement["x"], box["y"] - placement["y"]
        backing[max(0, math.floor(y) - 8):math.ceil(y + box["h"]) + 8,
                max(0, math.floor(x) - 8):math.ceil(x + box["w"]) + 8] = False
    assert backing.sum() > 1000
    assert np.abs(apply_rgb(np.array([21, 36, 57]) / 255, GradeSpec("film")) * 255
                  - [21, 36, 57]).mean() > 3

    finals = {}
    for look in ("none", "film"):
        project["grade"] = {"look": look}
        write_json(root / "project.json", project)
        result = assemble.assemble(root)
        final = root / result["out"]
        finals[look] = tmp_path / f"{look}.mp4"
        shutil.copyfile(final, finals[look])
        shutil.copyfile(final.with_suffix(".contrast.json"),
                        finals[look].with_suffix(".contrast.json"))
        assert receipt.sha256_file(layer) == layer_sha256
        record = json.loads(final.with_suffix(".contrast.json").read_bytes())
        assert record["measured"] and record["captions"][0]["backing"] == "box"
        assert not any(caption["auto"] for caption in record["captions"])

    crop = tuple(placement[key] for key in ("x", "y", "width", "height"))
    plain = synth.picture_pixels(finals["none"], frame, box=crop)
    film = synth.picture_pixels(finals["film"], frame, box=crop)
    error = np.abs(film.astype(float) - plain)
    assert error[backing].mean() < 1
    # Allow a small encoder tail while keeping the average well below one RGB level.
    assert np.percentile(error[backing], 99) <= 4
    picture = [synth.picture_pixels(finals[look], frame, box=(100, 300, 80, 80))
               for look in ("none", "film")]
    assert np.abs(picture[1].astype(float) - picture[0]).mean() > 5

    record = json.loads(finals["film"].with_suffix(".contrast.json").read_bytes())
    measured = {look: measure_caption_contrast(final, layer, placement, layout)
                for look, final in finals.items()}
    ratio = record["captions"][1]["ratio"]
    assert ratio == measured["film"]["captions"][1]["ratio"]
    assert abs(ratio - measured["none"]["captions"][1]["ratio"]) > 0.1
    assert record["media_sha256"] == receipt.sha256_file(finals["film"])
    print("Caption backing pixels:", int(backing.sum()),
          "mean error:", error[backing].mean(),
          "99th percentile:", np.percentile(error[backing], 99),
          "picture change:", np.abs(picture[1].astype(float) - picture[0]).mean(),
          "contrast Film/None:", ratio, measured["none"]["captions"][1]["ratio"])
