"""Real alpha plates and deterministic synthetic picture assembly."""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import synth
import synth_broll
from test_assemble_picture import insert, project
from test_letters import style

from yap_studio import ffmpeg, paths
from yap_studio.render import hyperframes
from yap_studio.stages.assemble_picture import render_picture
from yap_studio.stages.captions import prepare_layer_plan
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

pytestmark = pytest.mark.render


def pixels(movie: Path, frame: int, *, alpha: bool = False) -> np.ndarray:
    channels = 4 if alpha else 3
    result = subprocess.run([
        ffmpeg.find_tool("ffmpeg"), "-v", "error", "-i", str(movie), "-vf",
        f"select=eq(n\\,{frame})", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt",
        "rgba" if alpha else "rgb24", "-threads", "1", "pipe:1",
    ], capture_output=True, check=True, timeout=60)
    return np.frombuffer(result.stdout, np.uint8).reshape(1920, 1080, channels)


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    module = importlib.import_module("yap_studio.letters")
    root = tmp_path_factory.mktemp("letters-plate")
    element = ResolvedElement("broll-letters", "insert", "broll", 0, 72, "front",
                              {"letters": {"text": "DOG", "background": "color"}})
    timeline = ResolvedTimeline((element,), (), (), 90, 0)
    with paths.operation():
        placement = module.render_plates(root, timeline, style(), root / "work")[0]
        plan = module.plate_plan("DOG", "Inter", style().data["hook"]["accent_color"])
        composition = prepare_layer_plan(plan, root / "browser")
    install = hyperframes.ensure_installed()
    script = r"""
const fs = require('node:fs');
const {pathToFileURL} = require('node:url');
const puppeteer = require(process.argv[1]);
(async () => {
  const input = JSON.parse(fs.readFileSync(0, 'utf8'));
  const browser = await puppeteer.launch({executablePath: input.browser,
    args: ['--no-sandbox', '--disable-dev-shm-usage'], headless: true});
  try {
    const page = await browser.newPage();
    await page.evaluateOnNewDocument(vars => {
      window.__hyperframes = {getVariables: () => vars};
    }, input.variables);
    await page.goto(pathToFileURL(input.html).href);
    await page.waitForFunction(() => document.querySelector('[data-ready="true"]'));
    process.stdout.write(JSON.stringify(await page.evaluate(() => {
      const animations = document.getAnimations();
      animations.forEach(animation => animation.pause());
      const samples = [...Array(48).keys(), 47, 42, 36, 0].map(frame => {
        animations.forEach(animation => { animation.currentTime = frame / 30 * 1000; });
        const ink = document.getElementById('letter-ink');
        const matrix = new DOMMatrixReadOnly(getComputedStyle(ink).transform);
        return {frame, scale: matrix.a, x: matrix.e, y: matrix.f,
          opacity: Number(getComputedStyle(document.getElementById('plate')).opacity)};
      });
      return {...document.querySelector('[data-composition-id]').dataset, samples};
    })));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
"""
    result = subprocess.run(
        ["node", "-e", script, str(install.prefix / "node_modules/puppeteer-core")],
        input=json.dumps({"browser": install.browser_path,
                          "html": str(composition / "index.html"), "variables": plan.variables}),
        text=True, capture_output=True, check=True, timeout=60)
    return root, placement, json.loads(result.stdout)


def test_plate_alpha_and_final_transparency(plate) -> None:
    _, placement, _ = plate
    first = pixels(Path(placement["path"]), 0, alpha=True)
    assert first[0, 0, 3] == 255
    assert np.count_nonzero(first[:, :, 3] == 0) > 10000
    held = pixels(Path(placement["path"]), 35, alpha=True)[:, :, 3]
    opened = pixels(Path(placement["path"]), 42, alpha=True)[:, :, 3]
    assert np.array_equal(held, first[:, :, 3])
    assert np.count_nonzero(opened == 0) > np.count_nonzero(first[:, :, 3] == 0)
    assert not pixels(Path(placement["path"]), 47, alpha=True)[:, :, 3].any()


def test_zoom_origin_is_deep_inside_a_glyph(plate) -> None:
    _, placement, geometry = plate
    x, y = round(float(geometry["originX"])), round(float(geometry["originY"]))
    first = pixels(Path(placement["path"]), 0, alpha=True)
    assert first[y, x, 3] == 0
    assert float(geometry["originDepth"]) >= .6 * float(geometry["maxDepth"])
    assert 8 <= float(geometry["finalScale"]) <= 400


def test_zoom_and_fade_seek_to_exact_frames_in_both_directions(plate) -> None:
    _, _, geometry = plate
    for sample in geometry["samples"]:
        frame = max(0, sample["frame"] - 36)
        scale = float(geometry["finalScale"]) ** ((frame / 11) ** 2)
        assert sample["scale"] == pytest.approx(scale, rel=1e-5)
        assert sample["x"] == pytest.approx(float(geometry["originX"]) * (1 - scale),
                                            rel=1e-5, abs=.01)
        assert sample["y"] == pytest.approx(float(geometry["originY"]) * (1 - scale),
                                            rel=1e-5, abs=.01)
        assert sample["opacity"] == pytest.approx(min(1, (11 - frame) / 2))


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("letters-media")
    return {
        "cut": synth.picture_source(root / "cut.mov", "color=c=blue:s=1080x1920:r=30", 90),
        "take": synth.picture_source(root / "take.mov", "color=c=blue:s=160x90:r=30", 120),
        "insert": synth.picture_source(root / "insert.mov", "color=c=red:s=320x180:r=30", 90),
        "photo": synth_broll.png(root / "photo.png"),
        "captions": synth.picture_source(root / "clear.mov",
            "color=c=black@0:s=2x2:r=30,format=rgba", 90, alpha=True),
    }


def assemble(root, media, plate, *, background="color", photo=False, title=False, start=0):
    module = importlib.import_module("yap_studio.letters")
    project(root, media, hold_frames=0)
    end = start + 48 if photo else max(84, start + 49)
    authored = insert(root, media["photo" if photo else "insert"], start, end)
    element = ResolvedElement(authored["id"], "insert", "broll", start, end,
                              "front", {"media": authored["media"], "motion": "still",
                                        "letters": {"text": "DOG", "background": background}})
    titles = []
    if title:
        element = ResolvedElement("graphics-title", "title", "graphics", 0, 48, "behind",
                                  {"lines": ["DOG"]})
        titles = [{**plate[1], "id": element.id, "width": 1080, "height": 1920,
                   "layer": "behind", "kind": "title", "letters": True, "x": 0, "y": 0}]
    timeline = ResolvedTimeline((element,), (), (), 90, 0)
    with paths.operation():
        plates = module.render_plates(plate[0], timeline, style(), root / "plates")
        options = dict(work_dir=root / "work", timeline=timeline, title_layers=titles,
                       caption_layer={"path": str(media["captions"]), "x": 0, "y": 0,
                                      "width": 2, "height": 2, "start_frame": 0, "frames": 90},
                       letters_layers=plates)
        out = root / "picture.mp4"
        render_picture(root, out, **options)
    return out, options


@pytest.mark.parametrize("background", ["color", "blur"])
def test_insert_pixels_and_full_screen_after_letters(tmp_path, media, plate, background) -> None:
    movie, _ = assemble(tmp_path, media, plate, background=background)
    mask = pixels(Path(plate[1]["path"]), 0, alpha=True)[:, :, 3]
    y, x = np.argwhere(mask == 0)[len(np.argwhere(mask == 0)) // 2]
    first = pixels(movie, 0)
    assert first[y, x, 0] > 230 and first[y, x, 1] < 15
    if background == "color":
        assert np.allclose(first[20, 20], [54, 90, 88], atol=8)
    else:
        assert 100 < first[20, 20, 0] < 125 and first[20, 20, 1] < 15
    assert pixels(movie, 48)[20, 20, 0] > 230


def test_photo_cuts_back_on_the_next_frame(tmp_path, media, plate) -> None:
    movie, _ = assemble(tmp_path, media, plate, photo=True)
    assert pixels(movie, 47)[20, 20, 1] > 100
    assert pixels(movie, 48)[20, 20, 2] > 230


@pytest.mark.parametrize("background", ["color", "blur"])
def test_mid_video_mask_starts_on_the_insert_frame(tmp_path, media, plate, background) -> None:
    movie, _ = assemble(tmp_path, media, plate, start=37, background=background)
    assert pixels(movie, 36)[20, 20, 2] > 230
    first = pixels(movie, 37)
    if background == "color":
        assert np.allclose(first[20, 20], [54, 90, 88], atol=8)
    else:
        assert 100 < first[20, 20, 0] < 125
    assert pixels(movie, 85)[20, 20, 0] > 230


@pytest.mark.parametrize("background", ["color", "blur"])
def test_late_insert_branch_masks_its_first_frame(tmp_path, media, plate, background) -> None:
    from yap_studio.stages.assemble_picture import _Graph

    module = importlib.import_module("yap_studio.letters")
    graph = _Graph()
    fill = graph.filter("0:v", "trim=end_frame=49,scale=1080:1920,setsar=1,"
                        "settb=1/30,setpts=N+1607")
    mask = graph.filter("1:v", "settb=1/30,setpts=N+1607")
    element = ResolvedElement("broll-late", "insert", "broll", 1607, 1656, "front",
                              {"letters": {"text": "DOG", "background": background}})
    branch = module.composite(graph, fill, mask, element)
    branch = graph.filter(branch, "setpts=PTS-STARTPTS")
    movie = tmp_path / "late.mp4"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-filter_complex_threads", "1",
        "-i", str(media["insert"]), "-i", plate[1]["path"],
        "-filter_complex", ";".join(graph.filters), "-map", f"[{branch}]", "-an",
        "-frames:v", "49", "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", str(movie),
    ], timeout_s=120)
    first = pixels(movie, 0)
    if background == "color":
        assert np.allclose(first[20, 20], [54, 90, 88], atol=8)
    else:
        assert 100 < first[20, 20, 0] < 125
    assert pixels(movie, 48)[20, 20, 0] > 230


def test_title_opens_onto_the_cut(tmp_path, media, plate) -> None:
    movie, _ = assemble(tmp_path, media, plate, title=True)
    first = pixels(movie, 0)
    mask = pixels(Path(plate[1]["path"]), 0, alpha=True)[:, :, 3]
    assert first[:, :, 2][mask == 0].mean() > 230
    assert np.allclose(first[20, 20], [54, 90, 88], atol=8)
    assert pixels(movie, 48)[20, 20, 2] > 230


def test_repeat_render_has_identical_picture_bytes(tmp_path, media, plate) -> None:
    movie, options = assemble(tmp_path, media, plate)
    again = tmp_path / "again.mp4"
    with paths.operation():
        render_picture(tmp_path, again, **{**options, "work_dir": tmp_path / "work-again"})
    assert movie.read_bytes() == again.read_bytes()
