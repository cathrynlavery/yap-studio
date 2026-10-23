"""Offline Motion streak pixels and seek determinism."""

from __future__ import annotations

import json
import math
import subprocess
import threading
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import offline_renderer
import pytest
from test_cards_render import _frame_difference, _read_reference_png, _rgba_frames
from test_cards_render import renderer_install as renderer_install
from test_hf_registry_cards_render import _copy_gate_item

from yap_studio import cards, ffmpeg
from yap_studio.card_motion import SUPPORTED, prepare_builtin, prepare_registry
from yap_studio.gallery import live
from yap_studio.hf_registry import install
from yap_studio.render import hyperframes
from yap_studio.style import resolve

pytestmark = pytest.mark.render


def _plan(kind="quote", motion="streak"):
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_motion"] = motion
    slots = ({"quote": "Ship it", "name": "Sample"} if kind == "quote" else
             {"percent": 72} if kind == "hf:conic-progress-ring" else {"text": "Ship it"})
    return cards.card_layer_plan(kind, slots, style, frames=60)


def _band(plan):
    return replace(plan, x=0, width=1080)


def _rest_band(plan, work, renderer):
    inner = cards.prepare_card(plan, work, install=renderer)
    if plan.template.startswith("hf:"):
        composition = prepare_registry(replace(plan, motion={"side": "right", "enter": False}),
                                       inner, install=renderer)
        page = composition / "index.html"
        source = page.read_text()
        source = source[:source.index("window.__cardMotion=")] + (
            'window.__timelines=window.__timelines||{};window.__timelines["hf-host"]=tl;'
            '</script></body></html>')
        page.write_text(source)
        return composition
    composition = prepare_builtin(replace(plan, motion={"side": "right", "enter": True}),
                                  inner, install=renderer)
    page = composition / "index.html"
    source = page.read_text().replace('data-card-motion="streak" ', "")
    start = source.index('<script>const tl=')
    source = source[:start] + ('<script>const tl=gsap.timeline({paused:true});'
        f'tl.to({{}},{{duration:{hyperframes.layer_duration_s(plan.frames, plan.fps)}}},0);'
        'window.__timelines=window.__timelines||{};window.__timelines.band=tl;'
        '</script></body></html>')
    page.write_text(source)
    return composition


def _snapshots(composition, plan, renderer, tmp_path, seconds):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/index.html":
                data = live.live_html((composition / "index.html").read_text(), {}, "/runtime.js")
                data, kind = data.encode(), "text/html"
            elif self.path == "/runtime.js":
                data, kind = live.runtime_file(renderer).read_bytes(), "text/javascript"
            else:
                return super().do_GET()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(composition)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    prefix = tmp_path / "live"
    command = ["node", str(Path(__file__).parent / "fixtures/motion_capture.mjs"),
               str(renderer.prefix / "node_modules"), renderer.browser_path,
               f"http://127.0.0.1:{server.server_port}/index.html", "1080", str(plan.height),
               str(prefix), str(seconds)]
    if offline_renderer.offline_supported():
        command = ["/usr/bin/sandbox-exec", "-f",
                   str(offline_renderer._profile(tmp_path / "offline-live")), *command]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == 0, result.stdout + result.stderr
    return [_read_reference_png(Path(f"{prefix}-{i}.png")) for i in (0, 1)], \
        json.loads(result.stdout)


def test_real_none_matches_style_without_motion(tmp_path, renderer_install):
    style = resolve("preset:clean_minimal")
    assert style.data["graphics"]["card_motion"] == "none"
    slots = {"quote": "Ship it", "name": "Sample"}
    outputs = []
    for name in ("none", "without"):
        if name == "without":
            del style.data["graphics"]["card_motion"]
        plan = cards.card_layer_plan("quote", slots, style, frames=60)
        assert plan.motion is None
        result = cards.render_card(plan, tmp_path / name, tmp_path / f"{name}.mov",
                                   install=renderer_install)
        outputs.append(_rgba_frames(result.path, plan))
    np.testing.assert_array_equal(*outputs)


def test_real_streak_determinism_rest_and_drawn_entrance(
    tmp_path, renderer_install, _shared_cache_root, monkeypatch,
):
    _copy_gate_item("echo-smear", tmp_path, monkeypatch, _shared_cache_root)
    assert install.installed("echo-smear") is not None
    plan = _plan()
    assert plan.motion is not None
    streams = []
    commands = []
    original = hyperframes.tools.run_tool

    def run(tool, args, **kwargs):
        if tool == "hyperframes" and "render" in args:
            commands.append(args)
        return original(tool, args, **kwargs)

    monkeypatch.setattr(hyperframes.tools, "run_tool", run)
    for repeat in ("a", "b"):
        result = cards.render_card(plan, tmp_path / repeat, tmp_path / f"{repeat}.mov",
                                   install=renderer_install)
        streams.append(_rgba_frames(result.path, _band(plan)))
    np.testing.assert_array_equal(*streams)
    assert commands and all("--no-browser-gpu" in args for args in commands)
    assert all(args[args.index("--workers") + 1] == "1" for args in commands)
    none = _plan(motion="none")
    composition = _rest_band(none, tmp_path / "none", renderer_install)
    result = hyperframes.render_layer(composition, tmp_path / "none.mov", frames=none.frames,
                                       fps=none.fps, variables={}, install=renderer_install,
                                       browser_gpu=False)
    plain = _rgba_frames(result.path, _band(none))
    moving = streams[0]
    np.testing.assert_array_equal(moving[36], plain[36])
    assert not moving[36, :, :plan.x, 3].any()
    assert not moving[36, :, plan.x + plan.width:, 3].any()
    # The first moving frame contains the opaque card and its contrasting text already.
    first = moving[1]
    assert (first[..., 3] == 255).sum() > 100
    assert np.unique(first[first[..., 3] == 255, :3], axis=0).shape[0] > 10
    opaque = first[first[..., 3] == 255, :3].astype(int)
    assert (np.abs(opaque - np.array([34, 34, 34])).max(axis=1) <= 4).sum() > 100
    composition = tmp_path / "a/quote-motion"
    snapshots, report = _snapshots(composition, plan, renderer_install, tmp_path, 57 / 30)
    assert report[0]["from"] < report[0]["time"] < report[1]["from"]
    np.testing.assert_array_equal(*snapshots)
    np.testing.assert_allclose(snapshots[0][..., 3], moving[57, ..., 3], atol=1)
    mean, fraction = _frame_difference(snapshots[0], moving[57])
    # Live screenshots and ProRes differ at text edges, as in the other live layer checks.
    assert mean <= 1 and fraction <= 0.01
    copies = report[0]["copies"]
    assert len(copies) > 2
    assert all(first["opacity"] > second["opacity"] for first, second in
               zip(copies, copies[1:], strict=False))
    assert copies[-1]["opacity"] < 0.01
    assert all(copy["filter"] == "none" for copy in copies)
    edge = math.floor(report[0]["box"]["left"] + cards.PAD_X)
    trail = moving[57, plan.height // 2, :edge - 2, 3]
    trail = trail[trail > 0]
    assert len(trail) >= 10
    assert trail[:5].mean() < trail[-5:].mean()


@pytest.mark.parametrize("name", ["char-slam-explode", "conic-progress-ring"])
def test_real_registry_entrance_is_unchanged(
    name, tmp_path, renderer_install, _shared_cache_root, monkeypatch,
):
    for item in ("echo-smear", name):
        _copy_gate_item(item, tmp_path, monkeypatch, _shared_cache_root)
    outputs = []
    for motion in ("none", "streak"):
        plan = _plan(f"hf:{name}", motion)
        assert (plan.motion is not None) == (motion == "streak")
        if motion == "none":
            composition = _rest_band(plan, tmp_path / motion, renderer_install)
            result = hyperframes.render_layer(composition, tmp_path / f"{motion}.mov",
                frames=plan.frames, fps=plan.fps, variables={}, install=renderer_install,
                browser_gpu=False)
        else:
            result = cards.render_card(plan, tmp_path / motion, tmp_path / f"{motion}.mov",
                                       install=renderer_install)
        outputs.append(_rgba_frames(result.path, _band(plan)))
    plain, streak = outputs
    np.testing.assert_array_equal(streak[:49], plain[:49])
    assert not streak[:49, :, :plan.x, 3].any()
    assert not streak[:49, :, plan.x + plan.width:, 3].any()
    assert not np.array_equal(streak[57], plain[57])
    xs = np.arange(1080)
    def centre(frame):
        alpha = frame[..., 3].sum(axis=0)
        return (alpha * xs).sum() / alpha.sum()
    assert centre(streak[57]) > centre(plain[57]) + 50
    if name == "conic-progress-ring":
        _, reports = _snapshots(tmp_path / "streak/conic-progress-ring-motion", plan,
                                renderer_install, tmp_path, 57 / 30)
        assert reports[0]["copies"]
        assert all(copy["text"] == reports[0]["text"] for copy in reports[0]["copies"])


@pytest.mark.parametrize("kind", sorted(SUPPORTED))
def test_real_supported_panels_arrive_drawn(kind, tmp_path, renderer_install,
                                           _shared_cache_root, monkeypatch):
    _copy_gate_item("echo-smear", tmp_path, monkeypatch, _shared_cache_root)
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_motion"] = "streak"
    slots = json.loads((cards.template_dir(kind) / "preview.json").read_text())["slots"]
    image_slots = [name for name, rule in cards.load_template(kind)["slots"].items()
                   if rule["type"] == "image" and rule["required"]]
    if image_slots:
        image = tmp_path / "sample.png"
        ffmpeg.run(["ffmpeg", "-nostdin", "-f", "lavfi", "-i", "testsrc2=s=640x400:r=1",
                    "-frames:v", "1", "-threads", "1", str(image)], timeout_s=30)
        slots.update({name: image for name in image_slots})
    plan = cards.card_layer_plan(kind, slots, style, frames=90)
    assert plan.motion is not None
    result = cards.render_card(plan, tmp_path / "work", tmp_path / "card.mov",
                               install=renderer_install)
    rgba = _rgba_frames(result.path, _band(plan))
    assert not rgba[0, ..., 3].any()
    if kind == "comment_drop":
        # The bubble is a timed action, not an entrance reveal.
        cue = plan.variables["drop_at"]
        assert not rgba[:math.ceil(cue * plan.fps), ..., 3].any()
        assert (rgba[math.ceil((cue + 0.15) * plan.fps), ..., 3] == 255).sum() > 100
    else:
        assert (rgba[1, ..., 3] == 255).sum() > 100
    assert (rgba[36, ..., 3] == 255).sum() > 100
    assert not rgba[36, :, :plan.x, 3].any()
    assert not rgba[36, :, plan.x + plan.width:, 3].any()


def test_live_streak_arrives_drawn(tmp_path, renderer_install, _shared_cache_root, monkeypatch):
    _copy_gate_item("echo-smear", tmp_path, monkeypatch, _shared_cache_root)
    plan = _plan()
    composition = cards.prepare_card(plan, tmp_path / "host", install=renderer_install)
    _, reports = _snapshots(composition, plan, renderer_install, tmp_path, 1 / 30)
    assert reports[0]["quoteOpacity"] == 1, reports
    assert reports[0]["panelOpacity"] == 1, reports


@pytest.mark.parametrize("cues", [None, [0.3, 1.4, 2.2]], ids=["default", "early-cue"])
def test_live_entrance_copies_keep_rows_hidden_until_their_cues(
    cues, tmp_path, renderer_install, _shared_cache_root, monkeypatch,
):
    _copy_gate_item("echo-smear", tmp_path, monkeypatch, _shared_cache_root)
    style = resolve("preset:clean_minimal")
    style.data["graphics"].update(card_motion="streak", specs_rows="on_cue")
    slots = json.loads((cards.template_dir("specs_list") / "preview.json").read_text())["slots"]
    if cues is not None:
        slots["cues"] = cues
    plan = cards.card_layer_plan("specs_list", slots, style, frames=120)
    assert plan.motion is not None
    cues = json.loads(plan.variables["cues"])
    composition = cards.prepare_card(plan, tmp_path / "host", install=renderer_install)
    page = composition / "card/index.html"
    page.write_text(page.read_text().replace('row.className = "row";',
        'row.className = "row"; row.dataset.probeRow = i;'))
    for frame in range(12):
        time = frame / 30
        _, reports = _snapshots(composition, plan, renderer_install, tmp_path, time)
        for report in reports:
            for index, cue in enumerate(cues):
                if time < cue:
                    assert report["rows"][index] == 0, report
                    assert all(copy["rows"][index] == 0 for copy in report["copies"]), report
            if 0 < time <= 0.1:
                assert len(report["copies"]) > 2, report
            assert report["nestedGroups"] == 0
            assert report["groups"] == 2
    _, reports = _snapshots(composition, plan, renderer_install, tmp_path, 3.9)
    for report in reports:
        assert len(report["copies"]) > 2
        assert all(copy["rows"] == [1, 1, 1] for copy in report["copies"]), report
        assert report["nestedGroups"] == 0


def test_live_chart_bars_are_full_on_the_first_moving_frame(
    tmp_path, renderer_install, _shared_cache_root, monkeypatch,
):
    _copy_gate_item("echo-smear", tmp_path, monkeypatch, _shared_cache_root)
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_motion"] = "streak"
    slots = json.loads((cards.template_dir("chart") / "preview.json").read_text())["slots"]
    plan = cards.card_layer_plan("chart", slots, style, frames=90)
    composition = cards.prepare_card(plan, tmp_path / "host", install=renderer_install)
    _, reports = _snapshots(composition, plan, renderer_install, tmp_path, 1 / 30)
    for report in reports:
        assert report["bars"] == [1, 1, 1], report
        assert report["panelOpacity"] == 1


@pytest.mark.parametrize("kind,on_cue", [("stat", False), ("stat", True), ("specs_list", True)])
def test_real_streak_keeps_animation_after_landing(
    kind, on_cue, tmp_path, renderer_install, _shared_cache_root, monkeypatch,
):
    _copy_gate_item("echo-smear", tmp_path, monkeypatch, _shared_cache_root)
    style = resolve("preset:clean_minimal")
    style.data["graphics"].update(card_motion="streak", specs_rows="on_cue"
                                   if on_cue else "together")
    slots = json.loads((cards.template_dir(kind) / "preview.json").read_text())["slots"]
    if kind == "specs_list":
        slots["cues"] = [0.7, 1.4, 2.2]
    elif on_cue:
        slots.update(rows=["12 first", "25 second"], cues=[0.7, 1.4])
    plan = cards.card_layer_plan(kind, slots, style, frames=120)
    assert plan.motion is not None
    result = cards.render_card(plan, tmp_path / "work", tmp_path / "card.mov",
                               install=renderer_install)
    frames = _rgba_frames(result.path, _band(plan))
    landed, held = frames[13], frames[84]
    assert not np.array_equal(landed[:, plan.x:plan.x + plan.width],
                              held[:, plan.x:plan.x + plan.width])
    for frame in (landed, held):
        assert not frame[:, :plan.x, 3].any()
        assert not frame[:, plan.x + plan.width:, 3].any()


def test_real_quote_matches_none_during_common_hold(
    tmp_path, renderer_install, _shared_cache_root, monkeypatch,
):
    _copy_gate_item("echo-smear", tmp_path, monkeypatch, _shared_cache_root)
    outputs = []
    for motion in ("none", "streak"):
        plan = _plan(motion=motion)
        if motion == "none":
            composition = _rest_band(plan, tmp_path / motion, renderer_install)
            result = hyperframes.render_layer(composition, tmp_path / f"{motion}.mov",
                frames=plan.frames, fps=plan.fps, variables={}, install=renderer_install,
                browser_gpu=False)
        else:
            result = cards.render_card(plan, tmp_path / motion, tmp_path / f"{motion}.mov",
                                       install=renderer_install)
        outputs.append(_rgba_frames(result.path, _band(plan)))
    # Quote's last reveal ends at 0.65 s; the fly-out starts 0.35 s before the end.
    start = math.ceil(0.65 * plan.fps)
    stop = math.ceil(plan.frames - 0.35 * plan.fps)
    assert (start, stop) == (20, 50)
    for frame in range(start, stop):
        np.testing.assert_array_equal(outputs[1][frame, :, plan.x:plan.x + plan.width],
                                      outputs[0][frame, :, plan.x:plan.x + plan.width],
                                      err_msg=f"Quote frame {frame}")
    assert not np.array_equal(outputs[1][stop, :, plan.x:plan.x + plan.width],
                              outputs[0][stop, :, plan.x:plan.x + plan.width]), \
        "Quote frame 50 must be the first difference after the common hold"
