"""Synthetic smooth jump cut decisions, windows, caching and helper checks."""

from __future__ import annotations

import json
import platform
import sys
from contextlib import closing
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth
from test_timeline import project as project
from test_timeline import step_project as step_project

from yap_studio import cli, ffmpeg, receipt, tools
from yap_studio.errors import YapError
from yap_studio.faces.model import Box, FaceBox, FacesData, VideoIdentity
from yap_studio.joins import assembly, helper, plan, render
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline
from yap_studio.style import resolve_style_data


def pieces() -> list[dict]:
    return [dict(segment_id="s1", take="t1", src_f0=0, src_f1=18, out_f0=0, n_frames=18),
            dict(segment_id="s1", take="t1", src_f0=22, src_f1=40, out_f0=18, n_frames=18)]


def timeline(*elements: ResolvedElement) -> ResolvedTimeline:
    return ResolvedTimeline(elements, (), (), 36, 0)


def faces(*, dx: float = 8, height: float = 100, source: str = "detected",
          width: int = 16, video_height: int = 16) -> FacesData:
    boxes = tuple((FaceBox(Box(0 if i < 18 else dx, 0, 80, 100 if i < 18 else height),
                           1, source),) for i in range(36))
    return FacesData("measured", None, VideoIdentity("cut.mp4", "a" * 64, 36, Fraction(30),
                                                  width, video_height), {}, {}, boxes, ())


def pixels(value: int = 100, width: int = 16, height: int = 16) -> bytes:
    return np.full((height, width, 3), value * 257, dtype="<u2").tobytes()


class Blend:
    def __init__(self) -> None:
        self.calls = []

    def interpolate(self, before, after, *, width, height, phases):
        self.calls.append((before, after, width, height, phases))
        a, b = (np.frombuffer(frame, dtype="<u2").astype(float) for frame in (before, after))
        return [np.rint(a * (1 - phase) + b * phase).astype("<u2").tobytes()
                for phase in phases]


@pytest.fixture
def available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper, "unavailable_reason", lambda: None)


def decide(tmp_path, *, mode="smooth", zoom="none", measured=None, values=None,
           parts=None, elements=()):
    return plan.build_plan(tmp_path, timeline(*elements), parts or pieces(), [],
                           {"jump_cuts": mode, "zoom_mode": zoom},
                           faces=measured if measured is not None else faces(),
                           frames=values if values is not None else {14: pixels(), 21: pixels()})


def test_join_kinds_and_contiguous(tmp_path: Path) -> None:
    parts = pieces()
    parts += [dict(segment_id="s2", take="t1", src_f0=45, src_f1=63, out_f0=36, n_frames=18),
              dict(segment_id="s2", take="t2", src_f0=0, src_f1=18, out_f0=54, n_frames=18),
              dict(segment_id="s3", take="t2", src_f0=18, src_f1=36, out_f0=72, n_frames=18)]
    result = decide(tmp_path, mode="hard", parts=parts)
    assert [j.kind for j in result.joins] == ["removal", "story", "take"]
    assert [j.join_frame for j in result.joins] == [18, 36, 54]


def test_shared_jumps_match_join_listing(tmp_path):
    from yap_studio.joins.steps import jumps

    parts = pieces()
    parts += [dict(segment_id="s2", take="t1", src_f0=45, src_f1=63, out_f0=36, n_frames=18),
              dict(segment_id="s2", take="t2", src_f0=0, src_f1=18, out_f0=54, n_frames=18),
              dict(segment_id="s3", take="t2", src_f0=18, src_f1=36, out_f0=72, n_frames=18)]
    old = decide(tmp_path, mode="hard", parts=parts)
    assert [(frame, kind) for frame, kind, _, _ in jumps(parts)] == [
        (j.join_frame, j.kind) for j in old.joins]


def test_step_join_listing_uses_plan_and_ignores_engine_covers(step_project):
    from yap_studio.stages.timeline import resolve_timeline

    resolved = resolve_timeline(step_project)
    parts = json.loads((step_project / "cut/receipt.json").read_bytes())["plan"]["pieces"]
    result = plan.build_plan(step_project, resolved, parts, [], {"jump_cuts": "step_in"})
    assert result.fallback == "hard"
    assert [j.handling for j in result.joins] == ["step_in"] * 5
    assert result.to_dict()["counts"]["step_in"] == 5
    assert [j.reason for j in result.joins] == [j["reason"] for j in resolved.steps["jumps"]]
    assert all(j.covered_by is None for j in result.joins)
    wide = replace(resolved, steps={**resolved.steps,
        "shots": [{**s, "scale": 1, "center": None} for s in resolved.steps["shots"]],
        "jumps": [{**j, "scale_before": 1, "scale_after": 1}
                  for j in resolved.steps["jumps"]]})
    assert all(j.handling == "hard" for j in plan.build_plan(
        step_project, wide, parts, [], {"jump_cuts": "step_in"}).joins)


def test_step_covered_join_keeps_step_reason(step_project):
    from test_timeline import add, zoom

    resolved = add(step_project, zoom(start={"time": 2}, end={"time": 3}), lane="agent")
    parts = json.loads((step_project / "cut/receipt.json").read_bytes())["plan"]["pieces"]
    result = plan.build_plan(step_project, resolved, parts, [], {"jump_cuts": "step_in"})
    assert result.joins[0].handling == "hard"
    assert result.joins[0].reason == "A zoom covers this shot (agent-zoom)."
    assert result.to_dict()["counts"]["hard"] == 1
    assert result.to_dict()["counts"]["step_in"] == 4


def test_step_shift_join_lists_as_step_in(tmp_path):
    from test_step_in import pieces as step_pieces

    from yap_studio.joins.steps import plan_steps

    parts = step_pieces((30, 30, 30))
    boxes = {i: (Box(400 if i < 60 else 0, 400, 200, 250),) for i in range(90)}
    steps = plan_steps(parts, (), 90, (), boxes,
                       [(60, 90, "card", Box(0, 0, 1080, 180))],
                       bottom=1500, bottom_margin=40, card_margin=24).to_dict()
    resolved = replace(timeline(), total_frames=90, steps=steps)
    result = plan.build_plan(tmp_path, resolved, parts, [], {"jump_cuts": "step_in"})
    assert steps["shots"][1]["scale"] == steps["shots"][2]["scale"]
    assert steps["shots"][1]["center"] != steps["shots"][2]["center"]
    assert result.joins[1].handling == "step_in"
    assert result.to_dict()["counts"]["step_in"] == 2


def test_small_move_and_words(tmp_path: Path, available) -> None:
    words = [dict(id="t1:0", text="before", segment="s1", start=0.1, end=0.6),
             dict(id="t1:1", text="after", segment="s1", start=0.6, end=0.9)]
    result = plan.build_plan(tmp_path, timeline(), pieces(), words, {"jump_cuts": "smooth"},
                             faces=faces(), frames={14: pixels(), 21: pixels(102)})
    j = result.to_dict()["joins"][0]
    assert j["handling"] == "smooth" and j["face_shift"] == 0.08
    assert j["face_scale"] == 0 and j["picture_change"] == pytest.approx(2)
    assert j["reason"] == "The face moved 0.08 face heights and the picture changed 2.0."
    assert j["before_word"] == {k: words[0][k] for k in ("id", "text", "segment")}
    assert j["after_word"]["text"] == "after"
    assert result.to_dict()["counts"]["smooth"] == 1


@pytest.mark.parametrize("change,reason", [
    ("shift", "The face moved 0.40 face heights; the limit is 0.25."),
    ("scale", "The face size changed 20%; the limit is 10%."),
    ("picture", "The picture changed 30.0; the limit is 5.5. "
                "A hand or object entering or leaving would ghost."),
])
@pytest.mark.parametrize("zoom,fallback", [("punches", "punch"), ("none", "punch"),
                                           ("punches_and_push", "punch")])
def test_limits(tmp_path: Path, available, change: str, reason: str, zoom: str,
                fallback: str) -> None:
    result = decide(tmp_path, zoom=zoom,
                    measured=faces(dx=40 if change == "shift" else 8,
                                   height=120 if change == "scale" else 100),
                    values={14: pixels(), 21: pixels(130 if change == "picture" else 100)})
    j = result.joins[0]
    assert j.handling == fallback and j.reason == reason
    assert all(getattr(j, key) is not None
               for key in ("face_shift", "face_scale", "picture_change"))


@pytest.mark.parametrize("change", ["shift", "picture", "small"])
def test_smooth_with_default_zooms_off(tmp_path: Path, available, change: str) -> None:
    moves = resolve_style_data({"schema_version": 1, "moves": {"jump_cuts": "smooth"}})[
        "moves"]
    assert moves["zoom_mode"] == "none"
    result = plan.build_plan(
        tmp_path, timeline(), pieces(), [], moves,
        faces=faces(dx=40 if change == "shift" else 8),
        frames={14: pixels(), 21: pixels(130 if change == "picture" else 100)},
    )
    assert result.fallback == "punch"
    assert result.joins[0].handling == ("smooth" if change == "small" else "punch")
    if change == "shift":
        assert result.joins[0].face_shift > 0.25
    elif change == "picture":
        assert result.joins[0].picture_change > 5.5


@pytest.mark.parametrize("mode,expected", [("hard", "hard"), ("punch", "punch"),
                                         ("smooth", "punch"), ("step_in", "hard")])
@pytest.mark.parametrize("zoom", ["none", "punches", "punches_and_push"])
def test_fallback_follows_jump_cut_choice(mode: str, expected: str, zoom: str) -> None:
    assert plan.fallback({"jump_cuts": mode, "zoom_mode": zoom}) == expected


@pytest.mark.parametrize("source", ["interpolated", "unmeasured"])
def test_missing_detected_faces(tmp_path: Path, available, source: str) -> None:
    measured = faces(source="interpolated") if source == "interpolated" else replace(
        faces(), status="unmeasured", reason="No detector.", frames=())
    j = decide(tmp_path, measured=measured).joins[0]
    assert j.handling == "punch" and j.reason == "No face was detected next to the join."


def test_short_piece(tmp_path: Path, available) -> None:
    parts = pieces()
    parts[0].update(n_frames=3, src_f1=3)
    parts[1]["out_f0"] = 3
    j = decide(tmp_path, parts=parts).joins[0]
    assert j.handling == "punch"
    assert j.reason == "A piece next to the join is shorter than the 6-frame window."


@pytest.mark.parametrize("kind", ["zoom", "insert", "transition", "background", "moment",
                                  "grade_compare"])
def test_covered(tmp_path: Path, available, kind: str) -> None:
    item = ResolvedElement("agent-cover", kind, "agent", 17, 18, "front", {})
    j = decide(tmp_path, elements=(item,)).joins[0]
    assert j.handling == "covered" and j.covered_by == item.id
    assert j.reason == f"A {kind} covers this join (agent-cover)."


@pytest.mark.parametrize("mode,handling", [("smooth", "punch"), ("hard", "hard"),
                                         ("punch", "punch")])
def test_window_overlap_does_not_cover_join(tmp_path, available, mode, handling) -> None:
    item = ResolvedElement("early-zoom", "zoom", "agent", 14, 16, "front", {})
    j = decide(tmp_path, mode=mode, elements=(item,)).joins[0]
    assert j.handling == handling and j.covered_by is None
    if mode == "smooth":
        assert j.reason == "A zoom (early-zoom) overlaps the smoothing window."


@pytest.mark.parametrize("start,end", [(18, 20), (16, 18)])
def test_zoom_at_cut_covers_join(tmp_path, available, start, end) -> None:
    item = ResolvedElement("cut-zoom", "zoom", "agent", start, end, "front", {})
    j = decide(tmp_path, elements=(item,)).joins[0]
    assert j.handling == "covered" and j.covered_by == item.id


@pytest.mark.parametrize("start,end", [(18, 36), (0, 18)])
def test_tall_box_covers_both_edge_joins(tmp_path, available, start, end) -> None:
    item = ResolvedElement("broll-page", "overlay", "broll", start, end, "front",
                           {"placement": "tall_behind"})
    j = decide(tmp_path, elements=(item,)).joins[0]
    assert j.handling == "covered" and j.covered_by == item.id


@pytest.mark.parametrize("start,end", [(14, 17), (19, 22)])
def test_tall_box_window_overlap_falls_back(tmp_path, available, start, end) -> None:
    item = ResolvedElement("broll-page", "overlay", "broll", start, end, "front",
                           {"placement": "tall_behind"})
    j = decide(tmp_path, elements=(item,)).joins[0]
    assert j.handling == "punch" and j.covered_by is None
    assert item.id in j.reason and "overlaps the smoothing window" in j.reason


@pytest.mark.parametrize("start,end", [(18, 36), (0, 18), (14, 17), (19, 22)])
def test_headroom_overlay_leaves_smooth_join(tmp_path, available, start, end) -> None:
    item = ResolvedElement("broll-photo", "overlay", "broll", start, end, "front",
                           {"placement": "headroom"})
    j = decide(tmp_path, elements=(item,)).joins[0]
    assert j.handling == "smooth" and j.covered_by is None


@pytest.mark.parametrize("start,end", [(12, 14), (22, 24)])
def test_cover_outside_window_has_no_effect(tmp_path, available, start, end) -> None:
    item = ResolvedElement("outside-zoom", "zoom", "agent", start, end, "front", {})
    assert decide(tmp_path, elements=(item,)).joins[0].handling == "smooth"


@pytest.mark.parametrize("kind", ["overlay", "graphic", "title", "caption"])
def test_top_layers_do_not_cover(tmp_path: Path, available, kind: str) -> None:
    item = ResolvedElement("agent-top", kind, "agent", 14, 22, "front", {})
    assert decide(tmp_path, elements=(item,)).joins[0].handling == "smooth"


@pytest.mark.parametrize("mode", ["hard", "punch"])
def test_plain_modes_need_no_helper_faces_or_frames(tmp_path: Path, monkeypatch, mode) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("Plain joins must not measure or check the helper")
    monkeypatch.setattr(helper, "unavailable_reason", unexpected)
    monkeypatch.setattr(plan, "decode_frames", unexpected)
    monkeypatch.setattr(plan, "cut_faces", unexpected)
    result = plan.build_plan(tmp_path, timeline(), pieces(), [], {"jump_cuts": mode})
    assert result.joins[0].handling == mode
    assert result.joins[0].reason == (
        "The style keeps hard cuts." if mode == "hard"
        else "The style covers jump cuts with a punch.")


@pytest.mark.parametrize("system,version,found", [("darwin", "15.3", "15.3"),
                                                  ("linux", "", "Linux")])
def test_helper_refusal(monkeypatch, system, version, found) -> None:
    monkeypatch.setattr(helper.sys, "platform", system)
    monkeypatch.setattr(helper.platform, "mac_ver", lambda: (version, (), ""))
    reason = helper.unavailable_reason()
    assert found in reason and "15.4" in reason and "hard or punch" in reason


def test_plan_refuses_with_fix(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(helper, "unavailable_reason", lambda: "Update macOS to 15.4 or later.")
    with pytest.raises(YapError, match="15.4") as caught:
        decide(tmp_path)
    assert caught.value.suggested_action


@pytest.fixture
def counter_cut(tmp_path: Path) -> Path:
    return synth.counter_source(tmp_path / "cut.mov")


def test_window_geometry_movie_and_cache(tmp_path: Path, counter_cut, available,
                                       monkeypatch) -> None:
    blend = Blend()
    raw = dict(plan.decode_frames(counter_cut, [14, 21], width=1080, height=1920))
    measured = faces(width=1080, video_height=1920)
    result = plan.build_plan(tmp_path, timeline(), pieces(), [], {"jump_cuts": "smooth"},
                             faces=measured, frames=raw)
    first = render.render_windows(tmp_path, result, cut=counter_cut, interpolator=blend)
    movie = Path(first[0]["path"])
    info = ffmpeg.probe(movie)
    assert (info.video_frames, info.width, info.height, info.fps) == (6, 1080, 1920, 30)
    assert (info.color_primaries, info.color_transfer, info.color_space) == ("bt709",) * 3
    assert not info.has_audio
    assert (first[0]["start_frame"], first[0]["frames"], first[0]["join_frame"]) == (15, 6, 18)
    assert blend.calls == [(raw[14], raw[21], 1080, 1920, [k / 7 for k in range(1, 7)])]
    digest = receipt.sha256_file(movie)
    assert render.render_windows(tmp_path, result, cut=counter_cut, interpolator=blend) == first
    assert len(blend.calls) == 1 and receipt.sha256_file(movie) == digest
    with monkeypatch.context() as changed:
        changed.setattr(plan, "decode_frames", lambda *args, **kwargs:
                        ((index, raw[21]) for index in (14, 21)))
        assert render.render_windows(tmp_path, result, interpolator=blend)[0]["path"] != str(movie)
    movie.write_bytes(b"broken")
    with pytest.raises(YapError, match="cache"):
        render.render_windows(tmp_path, result, cut=counter_cut, interpolator=blend)


_REAL_HELPER = pytest.mark.skipif(
    sys.platform != "darwin" or tools.find_tool("xcrun") is None
    or tuple(int(n) for n in (platform.mac_ver()[0] or "0").split(".")[:2]) < (15, 4),
    reason="real frame conversion needs macOS 15.4 and xcrun",
)


@pytest.mark.helpers
@_REAL_HELPER
def test_real_helper(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(helper, "cache_root", lambda: tmp_path)
    (tmp_path / "helpers").mkdir()
    monkeypatch.setattr(helper, "cache_dir", lambda name: tmp_path / name)
    monkeypatch.setenv("CLANG_MODULE_CACHE_PATH", str(tmp_path / "clang"))
    monkeypatch.setenv("SWIFT_MODULECACHE_PATH", str(tmp_path / "swift"))
    reason = helper.unavailable_reason()
    if reason:
        pytest.skip(reason)
    a = np.zeros((192, 128, 3), dtype="<u2")
    b = a.copy()
    a[50:110, 30:70] = 50000
    b[50:110, 38:78] = 50000
    options = dict(width=128, height=192, phases=[0.25, 0.5, 0.75])
    first = helper.interpolate(a.tobytes(), b.tobytes(), **options)
    second = helper.interpolate(a.tobytes(), b.tobytes(), **options)
    assert len(first) == 3 and all(len(frame) == a.nbytes for frame in first)
    middle = np.frombuffer(first[1], dtype="<u2").reshape(a.shape)
    weights = middle.mean(axis=(0, 2))
    center = np.average(np.arange(128), weights=weights)
    assert 49.5 < center < 57.5
    assert first == second


def test_joins_command_json_is_read_only(project: Path, capsys) -> None:
    record_path = project / "cut/receipt.json"
    record = json.loads(record_path.read_bytes())
    record["plan"]["pieces"] = [
        dict(segment_id="s1", take="t1", src_f0=0, src_f1=180, out_f0=0, n_frames=180),
        dict(segment_id="s2", take="t1", src_f0=200, src_f1=380, out_f0=180, n_frames=180),
    ]
    record_path.write_text(json.dumps(record))
    def documents():
        files = [project / "edit.json", project / "project.json"]
        for name in ("timeline", "cut", "final"):
            files.extend(p for p in (project / name).rglob("*") if p.is_file())
        return {str(p): p.read_bytes() for p in files if p.is_file()}

    before = documents()
    assert cli.main(["joins", str(project), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema"] == "yap-studio/joins@1"
    assert len(data["joins"]) == 1 and data["joins"][0]["kind"] == "story"
    assert data["joins"][0]["handling"] == "hard"
    assert before == documents()


def test_joins_command_refuses_stale_cut(project, capsys) -> None:
    (project / "cut/receipt.json").unlink()
    assert cli.main(["joins", str(project), "--json"]) == 2
    assert "cut" in json.loads(capsys.readouterr().err)["error"]["message"]


@pytest.mark.parametrize("json_output", [False, True])
def test_joins_command_reports_tall_box(project, monkeypatch, capsys, json_output) -> None:
    from test_assemble_picture import overlay_image
    from test_timeline import add, overlay

    from yap_studio.stages import timeline as stage

    record_path = project / "cut/receipt.json"
    record = json.loads(record_path.read_bytes())
    record["plan"]["pieces"] = [
        dict(segment_id="s1", take="t1", src_f0=0, src_f1=180, out_f0=0, n_frames=180),
        dict(segment_id="s2", take="t1", src_f0=200, src_f1=380, out_f0=180, n_frames=180),
    ]
    record_path.write_text(json.dumps(record))
    monkeypatch.setattr(stage, "_picture_cuts", lambda *args: (0, 180, 360))
    page = overlay_image(project / "page.png", "color=white:s=620x2400")
    add(project, overlay(project, id="broll-page", placement="tall_behind",
                         start={"time": 0}, end={"time": 6},
                         media={"path": page.name, "sha256": receipt.sha256_file(page)}))
    assert cli.main(["joins", str(project), *(["--json"] if json_output else [])]) == 0
    output = capsys.readouterr().out
    if json_output:
        join, = json.loads(output)["joins"]
        assert join["handling"] == "covered" and join["covered_by"] == "broll-page"
    else:
        assert "covered" in output and "covers this join (broll-page)" in output


def test_joins_command_smooth_refusal_has_fix(project, monkeypatch, capsys) -> None:
    root = project
    project = json.loads((root / "project.json").read_bytes())
    (root / "smooth.json").write_text(json.dumps({"schema_version": 1,
                                                  "moves": {"jump_cuts": "smooth"}}))
    project["style"] = "smooth.json"
    (root / "project.json").write_text(json.dumps(project))
    record = json.loads((root / "cut/receipt.json").read_bytes())
    record["plan"]["pieces"] = pieces()
    (root / "cut/receipt.json").write_text(json.dumps(record))
    monkeypatch.setattr(helper, "unavailable_reason", lambda: "Update macOS to 15.4 or later.")
    assert cli.main(["joins", str(root), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert "15.4" in error["message"] and error["suggested_action"]


def test_nearest_detected_face_stays_inside_piece(tmp_path, available) -> None:
    measured = faces()
    rows = [(FaceBox(Box(0, 0, 80, 100), 1, "interpolated"),) for _ in range(36)]
    rows[12] = (FaceBox(Box(0, 0, 80, 100), 1, "detected"),)
    rows[19] = (FaceBox(Box(8, 0, 80, 100), 1, "detected"),
                FaceBox(Box(300, 0, 10, 10), 1, "detected"))
    result = decide(tmp_path, measured=replace(measured, frames=tuple(rows)))
    assert result.joins[0].handling == "smooth"
    assert result.joins[0].face_shift == 0.08
    rows[12] = rows[0]
    missing = decide(tmp_path, measured=replace(measured, frames=tuple(rows)))
    assert missing.joins[0].handling == "punch"


def test_geometry_tracks_window_constant(tmp_path, available, monkeypatch) -> None:
    monkeypatch.setattr(plan, "WINDOW_FRAMES", 8)
    j = decide(tmp_path, values={13: pixels(), 22: pixels()}).joins[0]
    assert (j.start_frame, j.frames, j.join_frame) == (14, 8, 18)


@pytest.mark.parametrize("gap,zoom", [(5, "none"), (5, "punches"), (6, "none"), (7, "none")])
def test_smooth_windows_preserve_each_others_boundaries(tmp_path, available, gap, zoom) -> None:
    parts = pieces()
    parts[1].update(n_frames=gap, src_f1=22 + gap)
    parts.append(dict(segment_id="s1", take="t1", src_f0=50, src_f1=68,
                      out_f0=18 + gap, n_frames=18))
    raw = {i: pixels() for i in (14, 21, 14 + gap, 21 + gap)}
    result = decide(tmp_path, parts=parts, values=raw, zoom=zoom)
    expected = ["smooth", "smooth" if gap == 7 else "punch"]
    assert [j.handling for j in result.joins] == expected
    if gap < 7:
        assert result.joins[1].reason == "Another smoothed join is too close."
        assert result.joins[1].picture_change == 0
    placements = [dict(id=f"smooth-{j.index}", join_frame=j.join_frame,
                       start_frame=j.start_frame, frames=j.frames)
                  for j in result.joins if j.handling == "smooth"]
    windows = assembly.verification_windows(placements)
    for i, window in enumerate(windows):
        for other in windows[i + 1:]:
            assert window.end_frame <= other.start_frame
            for boundary in (other.start_frame - 1, other.end_frame):
                assert boundary not in range(window.start_frame, window.end_frame)
            for boundary in (window.start_frame - 1, window.end_frame):
                assert boundary not in range(other.start_frame, other.end_frame)


def test_plan_streams_measurements_without_retaining_raw_frames(tmp_path, available,
                                                              monkeypatch) -> None:
    yielded, closed = [], []

    def reader(*args, **kwargs):
        try:
            for index in (14, 21):
                yielded.append(index)
                yield index, pixels(100 + len(yielded))
        finally:
            closed.append(True)

    monkeypatch.setattr(plan, "decode_frames", reader)
    result = plan.build_plan(tmp_path, timeline(), pieces(), [], {"jump_cuts": "smooth"},
                             faces=faces())
    assert yielded == [14, 21] and closed == [True]
    assert result.joins[0].picture_change == pytest.approx(1)
    assert not hasattr(result, "raw_frames")


@pytest.mark.parametrize("gap,counts", [(7, [2, 3]), (10, [2, 4])])
def test_renderer_streams_one_endpoint_pair_at_a_time(tmp_path, available, monkeypatch,
                                                     gap, counts) -> None:
    parts = pieces()
    parts[1].update(n_frames=gap, src_f1=22 + gap)
    parts.append(dict(segment_id="s1", take="t1", src_f0=50, src_f1=68,
                      out_f0=18 + gap, n_frames=18))
    raw = {i: pixels(100 + (i - 14) // 7) for i in (14, 21, 14 + gap, 21 + gap)}
    result = decide(tmp_path, parts=parts, values=raw)
    yielded, calls, closed = [], [], []

    def reader(cut, indices, **kwargs):
        assert cut == tmp_path / "cut/cut.mp4"
        try:
            for index in sorted(set(indices)):
                yielded.append(index)
                yield index, raw[index]
        finally:
            closed.append(True)

    class Interpolator:
        def interpolate(self, before, after, *, width, height, phases):
            calls.append(len(yielded))
            assert calls[-1] == counts[len(calls) - 1]
            return [pixels(width=width, height=height) for _ in phases]

    def encode(command, **kwargs):
        Path(command[-1]).write_bytes(b"synthetic window")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(plan, "decode_frames", reader)
    monkeypatch.setattr(render.process, "spawn", encode)
    monkeypatch.setattr(render.ffmpeg, "_safe_command", lambda command: command)
    monkeypatch.setattr(render.ffmpeg, "tool_versions", lambda: {"ffmpeg": "synthetic"})
    monkeypatch.setattr(render, "_assert_layer", lambda *args: None)
    placements = render.render_windows(tmp_path, result, interpolator=Interpolator())
    assert len(placements) == 2 and calls == counts and closed == [True]


@pytest.mark.parametrize("payload", [b"", pixels()[:-1], pixels(), pixels() * 3],
                         ids=["missing", "partial", "too_few", "too_many"])
def test_frame_reader_rejects_short_or_wrong_count(tmp_path, monkeypatch, payload) -> None:
    closed = []

    def stream(*args, **kwargs):
        try:
            yield payload
        finally:
            closed.append(True)

    monkeypatch.setattr(plan.ffmpeg, "run_stream", stream)
    with pytest.raises(helper.JoinError, match="frame sequence"):
        list(plan.decode_frames(tmp_path / "cut.mp4", [14, 21], width=16, height=16))
    assert closed == [True]


def test_frame_reader_yields_exact_indices_from_partial_chunks(tmp_path, monkeypatch) -> None:
    a, b = pixels(), pixels(102)
    commands = []

    def stream(command, **kwargs):
        commands.append((command, kwargs))
        yield a[:17]
        yield a[17:] + b[:31]
        yield b[31:]

    monkeypatch.setattr(plan.ffmpeg, "run_stream", stream)
    assert list(plan.decode_frames(tmp_path / "cut.mp4", [21, 14, 21], width=16,
                                   height=16)) == [(14, a), (21, b)]
    command, options = commands[0]
    assert "select=eq(n\\,14)+eq(n\\,21)" in command[command.index("-vf") + 1]
    assert "in_color_matrix=bt709:in_range=tv" in command[command.index("-vf") + 1]
    assert options["timeout_s"] == ffmpeg.DEFAULT_TIMEOUT_S


@pytest.mark.render
def test_frame_reader_stops_after_requested_count(tmp_path, monkeypatch) -> None:
    cut = synth.picture_source(tmp_path / "cut.mov",
                               "color=black:s=16x16:r=30,geq=lum='16+N*5':cb=128:cr=128", 36)
    commands = []
    run_stream = ffmpeg.run_stream

    def stream(command, **kwargs):
        commands.append(command)
        return run_stream(command, **kwargs)

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    selected = dict(plan.decode_frames(cut, [5, 2, 5], width=16, height=16))
    reference = dict(plan.decode_frames(cut, list(range(6)), width=16, height=16))
    assert selected == {index: reference[index] for index in (2, 5)}
    assert selected[2] != selected[5]
    for command, count in zip(commands, (2, 6), strict=True):
        assert command[command.index("-frames:v") + 1] == str(count)


@pytest.mark.render
def test_smooth_decoder_deadline_allows_window_processing(tmp_path, available,
                                                        monkeypatch) -> None:
    cut = synth.picture_source(tmp_path / "cut.mov",
                               "color=black:s=16x16:r=30,geq=lum='16+N*5':cb=128:cr=128", 36)
    parts = pieces()
    parts[1].update(n_frames=10, src_f1=32)
    parts.append(dict(segment_id="s1", take="t1", src_f0=50, src_f1=68,
                      out_f0=28, n_frames=18))
    result = decide(tmp_path, parts=parts,
                    values={index: pixels() for index in (14, 21, 24, 31)})
    elapsed = [0.0]
    run_stream = ffmpeg.run_stream

    def stream(command, **kwargs):
        kwargs["timeout_s"] += 2 - ffmpeg.DEFAULT_TIMEOUT_S
        return run_stream(command, **kwargs)

    class SlowBlend(Blend):
        def interpolate(self, *args, **kwargs):
            elapsed[0] += 5
            return super().interpolate(*args, **kwargs)

    monkeypatch.setattr(ffmpeg, "run_stream", stream)
    monkeypatch.setattr(ffmpeg, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    monkeypatch.setattr(render, "_assert_layer", lambda *args: None)
    blend = SlowBlend()
    placements = render.render_windows(tmp_path, result, cut=cut, interpolator=blend)
    assert len(placements) == len(blend.calls) == 2
    assert all(Path(item["path"]).is_file() for item in placements)


def test_frame_reader_reaps_process_when_closed(tmp_path, monkeypatch) -> None:
    processes = []
    popen = ffmpeg.subprocess.Popen
    script = ("import sys,time;sys.stdout.buffer.write(bytes(1536));"
              "sys.stdout.flush();time.sleep(60)")

    def start(*args, **kwargs):
        child = popen(*args, **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(ffmpeg, "_safe_command", lambda cmd: [sys.executable, "-c", script])
    monkeypatch.setattr(ffmpeg.subprocess, "Popen", start)
    with closing(plan.decode_frames(tmp_path / "cut.mp4", [14, 21], width=16,
                                   height=16)) as reader:
        assert next(reader) == (14, bytes(1536))
    assert len(processes) == 1 and processes[0].poll() is not None


@pytest.mark.parametrize("phases", [[0], [1], [0.5, 0.25], [float("nan")]])
def test_helper_rejects_invalid_phases(phases) -> None:
    with pytest.raises(YapError) as caught:
        helper.interpolate(pixels(), pixels(), width=16, height=16, phases=phases)
    assert caught.value.suggested_action


def test_helper_rejects_wrong_input_count() -> None:
    with pytest.raises(YapError, match="bytes"):
        helper.interpolate(b"", pixels(), width=16, height=16, phases=[0.5])


def test_helper_support_refusal_is_cached(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(helper.platform, "mac_ver", lambda: ("15.4", (), ""))
    monkeypatch.setattr(helper.tools, "find_tool", lambda name: "/usr/bin/xcrun")
    monkeypatch.setattr(helper.tools, "run_tool", lambda *args, **kwargs: None)
    monkeypatch.setattr(helper, "helper_executable", lambda: tmp_path / "helper")
    monkeypatch.setattr(helper, "_support", {})
    calls = []

    def check(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='{"supported":false,"revision":1}')

    monkeypatch.setattr(helper.process, "spawn", check)
    assert "not supported on this Mac" in helper.unavailable_reason()
    assert "hard or punch" in helper.unavailable_reason()
    assert len(calls) == 1


def test_helper_missing_tools(monkeypatch) -> None:
    monkeypatch.setattr(helper.sys, "platform", "darwin")
    monkeypatch.setattr(helper.platform, "mac_ver", lambda: ("15.4", (), ""))
    monkeypatch.setattr(helper, "helper_path", lambda: Path("/absent/smooth-helper"))
    monkeypatch.setattr(helper.tools, "find_tool", lambda name: None)
    assert "xcode-select --install" in helper.unavailable_reason()


@pytest.mark.parametrize("code,output", [(0, b"short"), (3, b"")])
def test_helper_rejects_bad_output(monkeypatch, tmp_path, code, output) -> None:
    monkeypatch.setattr(helper, "helper_executable", lambda: tmp_path / "helper")
    monkeypatch.setattr(helper.process, "spawn", lambda *args, **kwargs: SimpleNamespace(
        returncode=code, stdout=output, stderr=b"Processor not supported."))
    with pytest.raises(YapError) as caught:
        helper.interpolate(pixels(), pixels(), width=16, height=16, phases=[0.5])
    assert caught.value.suggested_action


def test_helper_build_failure_has_fix(monkeypatch) -> None:
    from yap_studio.errors import ProcessingError

    def failed():
        raise ProcessingError("swiftc", 1, "Compiler unavailable.")

    monkeypatch.setattr(helper, "helper_executable", failed)
    with pytest.raises(YapError) as caught:
        helper.interpolate(pixels(), pixels(), width=16, height=16, phases=[0.5])
    assert caught.value.suggested_action


def test_joins_command_refuses_missing_piece_plan(project, capsys) -> None:
    assert cli.main(["joins", str(project), "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert "pieces" in error["message"] and "cut" in error["suggested_action"]


@pytest.mark.parametrize("change,handling", [(5.0, "smooth"), (5.5, "smooth"),
                                           (6.28, "punch"), (6.44, "punch")])
def test_picture_change_ghosting_limit(tmp_path, available, change, handling) -> None:
    after = np.full((16, 16, 3), 100 * 257, dtype="<u2")
    after[:8] += int(change) * 257
    after[8:] += round((2 * change - int(change)) * 257)
    decision = decide(tmp_path, values={14: pixels(), 21: after.tobytes()}).joins[0]
    assert plan.PICTURE_CHANGE_MAX == 5.5
    assert decision.picture_change == pytest.approx(change, abs=0.002)
    assert decision.handling == handling


@pytest.mark.parametrize("mode", ["hard", "punch"])
def test_plain_modes_skip_assemble_work(tmp_path, monkeypatch, mode) -> None:
    def unexpected(*args, **kwargs):
        pytest.fail("Plain cut styles must not read or render join inputs")

    class Context:
        read = unexpected

        def render_style(self):
            return SimpleNamespace(data={"moves": {"jump_cuts": mode}})

    inputs = {"existing": "digest"}
    small = {"existing.json": b"{}"}
    assembly.snapshot_inputs(Context(), inputs, small)
    assert inputs == {"existing": "digest"}
    assert small == {"existing.json": b"{}"}
    monkeypatch.setattr(render, "build_plan", unexpected)
    assert render.render_joins(tmp_path, timeline(), tmp_path / "absent.mp4",
                               {"jump_cuts": mode}) == ([], None)
    assert not list(tmp_path.iterdir())
