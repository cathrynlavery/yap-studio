"""Exact transition extraction, content caching, local shaders and seek mapping."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth
from test_captions_render import _template_violations

from yap_studio import ffmpeg, paths
from yap_studio.render import hyperframes
from yap_studio.stages import transitions as stage
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

LOOKS = ("whip_pan", "cinematic_zoom", "light_leak")
HASHES = {
    "whip_pan": "97a690de11f6dcebd6356bea84cba1af6ae61bfe03b9b7cc75a81368c93a12c9",
    "cinematic_zoom": "72b3a83d5de6a56fd389044aa15e785dbb40d5402cea6b95684b6b9cec0e4aba",
    "light_leak": "018575074693864c3f0f71380f85f089c172f73717b80787640b2da3615345be",
}


def timeline(look: str = "whip_pan", *, frames: int = 12, join: int = 18,
             total: int = 36) -> ResolvedTimeline:
    """Build a resolved sentence join without invoking unrelated project stages."""
    item = ResolvedElement("agent-transition", "transition", "agent", join - frames // 2,
                           join + frames // 2, "front",
                           {"look": look, "frames": frames, "join_frame": join})
    return ResolvedTimeline((item,), (), (), total, 0)


@pytest.mark.parametrize("look", LOOKS)
def test_transition_template_provenance_and_local_assets(tmp_path: Path, look: str) -> None:
    template = stage.template_file(look)
    source = template.read_text()
    item = look.replace("_", "-")
    metadata = json.loads((template.parent / "template.json").read_text())
    provenance = json.loads((stage.TEMPLATES / "PROVENANCE.json").read_text())
    entry = next(record for record in provenance["files"] if record["item"] == item)
    upstream = f"registry/blocks/{item}/{item}.html"
    assert entry["upstream_path"] == upstream and entry["upstream_sha256"] == HASHES[look]
    assert entry["yap_path"] == f"src/yap_studio/templates/transition_{look}/template.html"
    assert metadata["derived_from"] == [{"item": item, "path": upstream}]
    assert re.match(r'<!doctype html>\s*<!--\s*Derived from HyperFrames registry item', source)
    assert f'Derived from HyperFrames registry item "{item}"' in source
    assert f'({upstream} at commit {provenance["upstream"]["commit"]})' in source
    assert f'{provenance["upstream"]["copyright"]} Apache License 2.0 ' in source
    assert not _template_violations(template)
    assert not _template_violations(template.parent / "template.json")
    (tmp_path / "index.html").write_text(source)
    assert hyperframes.preflight(tmp_path, 12) == 0.399
    assert 'data-no-timeline' in source and "hf-seek" in source and "waitUntil" in source


@pytest.mark.parametrize("look", LOOKS)
@pytest.mark.parametrize("frames", [6, 12, 24])
def test_template_seeks_choose_images_and_progress(look: str, frames: int) -> None:
    """Execute the full script with local image and WebGL stubs, including backwards seeks."""
    node = shutil.which("node")
    assert node is not None, "Node is required for template tests"
    source = stage.template_file(look).read_text()
    script = re.search(r"<script>(.*?)</script>", source, re.DOTALL).group(1)
    harness = r"""
const fs = require('node:fs');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
let serial = 0, active = 0, progress = 0;
const bound = {}, textures = {}, events = {}, samples = [];
const gl = new Proxy({
  createTexture() { return ++serial; },
  getShaderParameter() { return true; }, getProgramParameter() { return true; },
  getUniformLocation(_, name) { return name; },
  activeTexture(unit) { active = unit; },
  bindTexture(_, texture) { bound[active] = texture; },
  texImage2D(...args) { textures[bound[active]] = args[5].src; },
  uniform1f(_, value) { progress = value; },
  drawArrays() { samples.push({from: textures[bound[0]], to: textures[bound[1]], progress}); },
}, { get(target, key) { if (key in target) return target[key];
  if (key === 'TEXTURE0') return 0; if (key === 'TEXTURE1') return 1;
  return /^[A-Z_]+$/.test(key) ? key : () => ({}); } });
const root = {append() {}};
global.Image = class {
  constructor() { this.style = {}; this.naturalWidth = 1080; this.naturalHeight = 1920; }
  set src(value) { this.url = value; Promise.resolve().then(() => this.onload()); }
  get src() { return this.url; }
};
global.window = {
  __hyperframes: {getVariables() {return {frames: input.frames, fps: 30};}},
  addEventListener(name, callback) { events[name] = callback; },
};
global.document = {querySelector() {return root;},
  createElement() {return {getContext() {return gl;}};}};
(async () => {
  eval(input.script);
  for (const time of input.times) {
    const waiting = [];
    events['hf-seek']({detail: {time, waitUntil(promise) {waiting.push(promise);}}});
    if (waiting.length !== 1) throw new Error('Seek did not register image readiness');
    await Promise.all(waiting);
    const sample = samples.at(-1);
    sample.time = time;
  }
  process.stdout.write(JSON.stringify(samples.filter(sample => 'time' in sample)));
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    indices = [*range(frames), frames // 2, 0, frames - 1]
    times = [n / 30 for n in indices]
    result = subprocess.run([node, "-e", harness], input=json.dumps({
        "script": script, "frames": frames, "times": times,
    }), text=True, capture_output=True, check=True, timeout=30)
    samples = json.loads(result.stdout)
    assert len(samples) == len(indices)
    for n, sample in zip(indices, samples, strict=True):
        assert sample["from"] == f"from/{min(n, frames // 2 - 1):03d}.png"
        assert sample["to"] == f"to/{max(0, n - frames // 2):03d}.png"
        assert sample["progress"] == pytest.approx((n + 0.5) / frames)


@pytest.fixture(scope="module")
def counter_cut(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Burn a counter into lossless full-size frames so extraction has an independent oracle."""
    folder = tmp_path_factory.mktemp("transition-counter")
    return synth.counter_source(folder / "cut.mov")


def test_extract_exact_counter_frames(tmp_path: Path, counter_cut: Path) -> None:
    extracted = []
    with paths.operation():
        extracted += stage._extract(counter_cut, tmp_path / "from", 12, 6)
        extracted += stage._extract(counter_cut, tmp_path / "to", 18, 6)
    assert len(extracted) == 12
    for n, png in zip(range(12, 24), extracted, strict=True):
        info = ffmpeg.probe(png)
        assert (info.width, info.height) == (1080, 1920)
        np.testing.assert_array_equal(
            synth.picture_pixels(png, 0, box=(20, 20, 180, 120)),
            synth.picture_pixels(counter_cut, n, box=(20, 20, 180, 120)),
        )


def test_extraction_refuses_missing_frames(tmp_path: Path, counter_cut: Path) -> None:
    with paths.operation(), pytest.raises(stage.TransitionError, match="expected 6"):
        stage._extract(counter_cut, tmp_path / "short", 34, 6)


@pytest.mark.parametrize("join", [6, 30])
def test_transition_renderer_refuses_windows_without_content_neighbors(
    tmp_path: Path, counter_cut: Path, join: int,
) -> None:
    with pytest.raises(stage.TransitionError, match="invalid transition frame window"):
        stage.render_transitions(tmp_path, timeline(join=join), counter_cut, tmp_path / "work")


def test_transition_cache_uses_frame_and_template_content(
    tmp_path: Path, counter_cut: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake(composition: Path, out: Path, **options: Any) -> None:
        assert len(list((composition / "from").glob("*.png"))) == 6
        assert len(list((composition / "to").glob("*.png"))) == 6
        assert hyperframes.preflight(composition, options["frames"]) == 0.399
        assert options["variables"] == {"frames": 12, "fps": 30}
        calls.append(out)
        synth.picture_source(out, "color=c=red:s=1080x1920:r=30", options["frames"])

    monkeypatch.setattr(hyperframes, "render_layer", fake)
    resolved = timeline()
    first = stage.render_transitions(tmp_path, resolved, counter_cut, tmp_path / "work")
    second = stage.render_transitions(tmp_path, resolved, counter_cut, tmp_path / "work")
    assert first == second and len(calls) == 1
    assert first[0]["start_frame"] == 12 and first[0]["frames"] == 12
    assert Path(first[0]["path"]).is_relative_to(tmp_path / ".yap/cache/transitions")
    changed = synth.picture_source(tmp_path / "changed.mov",
                                   "color=c=blue:s=1080x1920:r=30", 36)
    third = stage.render_transitions(tmp_path, resolved, changed, tmp_path / "work")
    assert third[0]["path"] != first[0]["path"] and len(calls) == 2
    template = tmp_path / "changed-template.html"
    template.write_text(stage.template_file("whip_pan").read_text() + "\n<!-- changed -->\n")
    monkeypatch.setattr(stage, "template_file", lambda _: template)
    fourth = stage.render_transitions(tmp_path, resolved, changed, tmp_path / "work")
    assert fourth[0]["path"] != third[0]["path"] and len(calls) == 3
    assert not list((tmp_path / ".yap/cache/transitions").glob(".transition-*"))


def test_no_transitions_needs_no_media_or_cache(tmp_path: Path) -> None:
    assert stage.render_transitions(tmp_path, replace(timeline(), elements=()),
                                    tmp_path / "absent.mp4", tmp_path / "work") == []
    assert not (tmp_path / ".yap").exists()


@pytest.mark.render
@pytest.mark.parametrize("look", LOOKS)
def test_real_transition_solid_sequences(tmp_path: Path, look: str) -> None:
    """Lead-only rendering proves endpoints and cadence for each retained registry shader."""
    cut = synth.picture_source(tmp_path / "cut.mov",
                               "color=c=red:s=1080x1920:r=30,"
                               "drawbox=c=blue:t=fill:enable='gte(n,7)'", 14)
    result = stage.render_transitions(tmp_path, timeline(look, join=7, total=14), cut,
                                      tmp_path / "work")
    movie = Path(result[0]["path"])
    assert ffmpeg.probe(movie).video_frames == 12
    first = synth.picture_pixels(movie, 0).mean(axis=(0, 1))
    last = synth.picture_pixels(movie, 11).mean(axis=(0, 1))
    assert first[0] > 180 and first[0] > first[2] * 3
    assert last[2] > 180 and last[2] > last[0] * 3
