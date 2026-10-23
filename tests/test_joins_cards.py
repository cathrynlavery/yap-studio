"""Card handoffs respect the join windows selected for assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from yap_studio import receipt
from yap_studio.cards import CARD_TEMPLATES
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.joins import assembly, plan, render
from yap_studio.project.receipts import projection_sha256
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline, TimelineRefusal


def test_behind_card_guard_invalidates_assembly_but_preserves_window_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(render.helper, "source_digest", lambda: "synthetic helper")
    monkeypatch.setattr(render.platform, "mac_ver", lambda: ("synthetic OS", (), ""))
    monkeypatch.setattr(render.ffmpeg, "tool_versions", lambda: {"ffmpeg": "synthetic encoder"})
    before, after = b"before", b"after"
    phases = [k / 7 for k in range(1, 7)]
    key = projection_sha256({
        "before": hashlib.sha256(before).hexdigest(), "after": hashlib.sha256(after).hexdigest(),
        "phases": phases, "window_frames": plan.WINDOW_FRAMES,
        "helper": "synthetic helper", "macos": "synthetic OS",
        "ffmpeg": "synthetic encoder", "version": 2,
    })
    folder = tmp_path / key
    folder.mkdir()
    movie = folder / "window.mp4"
    movie.write_bytes(b"synthetic cached window")
    (folder / "receipt.json").write_text(json.dumps({
        "schema": "yap-studio/smooth-join-cache@1", "key": key,
        "sha256": receipt.sha256_file(movie),
    }))
    checked = []
    monkeypatch.setattr(render, "_assert_layer",
                        lambda movie, frames: checked.append((movie, frames)))

    class CachedOnly:
        def interpolate(self, *args, **kwargs):
            pytest.fail("The guard must reuse cached join windows")

    join = plan.JoinDecision(0, "removal", 78, 2.6, 75, 6, None, None, "smooth", "")
    first = render.identity({"jump_cuts": "smooth"})
    assert first["behind_card_join_guard"] == render.BEHIND_CARD_JOIN_GUARD_VERSION
    guard = render.BEHIND_CARD_JOIN_GUARD_VERSION
    for version in (guard, guard + 1):
        monkeypatch.setattr(render, "BEHIND_CARD_JOIN_GUARD_VERSION", version)
        assert render._render_window(tmp_path, join, before, after, width=1, height=1,
                                     phases=phases, interpolator=CachedOnly()) == movie
    assert projection_sha256(render.identity({"jump_cuts": "smooth"})) != projection_sha256(first)
    assert checked == [(movie, 6), (movie, 6)]


@pytest.mark.parametrize("handling", ["smooth", "hard", "punch", "covered"])
@pytest.mark.parametrize("ending", ["handoff", "fade"])
@pytest.mark.parametrize("last_frame", [74, 75, 80, 81])
def test_reel_feed_handoff_checks_selected_smooth_join_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    handling: str, ending: str, last_frame: int,
) -> None:
    cut = tmp_path / "cut/cut.mp4"
    cut.parent.mkdir()
    (cut.parent / "receipt.json").write_text(json.dumps({"plan": {"pieces": []}}))
    (cut.parent / "cut-words.json").write_text(json.dumps({"words": []}))
    decisions = plan.JoinPlan("smooth", "punch", [
        plan.JoinDecision(0, "removal", 78, 2.6, 75, 6, None, None, handling, ""),
    ])
    monkeypatch.setattr(render, "build_plan", lambda *args, **kwargs: decisions)
    calls = []

    def rendered(*args, **kwargs):
        calls.append(decisions)
        return []

    monkeypatch.setattr(render, "render_windows", rendered)
    feed = ResolvedElement("agent-feed", "graphic", "agent", last_frame - 74,
                           last_frame + 1, "front",
                           {"template": "reel_feed", "slots": {"ending": ending}})
    timeline = ResolvedTimeline((feed,), (), (), 150, 0)
    blocked = handling == "smooth" and ending == "handoff" and 75 <= last_frame < 81
    if blocked:
        with pytest.raises(TimelineRefusal, match="move the feed.*ending fade") as caught:
            assembly.render_joins(tmp_path, timeline, cut, {"jump_cuts": "smooth"})
        assert caught.value.code == "card_video_handoff"
        assert "agent-feed" in str(caught.value)
        assert "Move the feed" in caught.value.suggested_action
        assert "ending fade" in caught.value.suggested_action
        assert not calls
    else:
        assert assembly.render_joins(tmp_path, timeline, cut,
                                     {"jump_cuts": "smooth"}) == ([], decisions)
        assert calls == [decisions]


@pytest.mark.parametrize("layer,gap,start,end,handling,mode,template,blocked", [
    ("behind", -24, 0, 90, "smooth", "smooth", "playing_grid", True),
    ("behind", -24, 0, 75, "smooth", "smooth", "playing_grid", False),
    ("behind", -24, 81, 120, "smooth", "smooth", "playing_grid", False),
    ("behind", -24, 80, 120, "smooth", "smooth", "playing_grid", True),
    ("behind", 24, 0, 90, "smooth", "smooth", "playing_grid", False),
    ("behind", 48, 0, 90, "smooth", "smooth", "playing_grid", False),
    ("behind", 1, 0, 90, "smooth", "smooth", "playing_grid", True),
    ("behind", 0, 0, 90, "smooth", "smooth", "playing_grid", True),
    ("behind", None, 0, 90, "smooth", "smooth", "playing_grid", True),
    ("front", -24, 0, 90, "smooth", "smooth", "playing_grid", False),
    ("behind", -24, 0, 90, "hard", "smooth", "playing_grid", False),
    ("behind", -24, 0, 90, "punch", "smooth", "playing_grid", False),
    ("behind", -24, 0, 90, "smooth", "hard", "playing_grid", False),
    ("behind", -24, 0, 90, "smooth", "punch", "playing_grid", False),
])
def test_behind_grid_checks_selected_smooth_join_against_clearance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str, gap: float | None,
    start: int, end: int, handling: str, mode: str, template: str, blocked: bool,
) -> None:
    cut = tmp_path / "cut/cut.mp4"
    cut.parent.mkdir()
    (cut.parent / "receipt.json").write_text(json.dumps({"plan": {"pieces": []}}))
    (cut.parent / "cut-words.json").write_text(json.dumps({"words": []}))
    decisions = plan.JoinPlan("smooth", "punch", [
        plan.JoinDecision(0, "removal", 78, 2.6, 75, 6, None, None, handling, ""),
    ])
    monkeypatch.setattr(render, "build_plan", lambda *args, **kwargs: decisions)
    calls = []

    def rendered(*args, **kwargs):
        calls.append(decisions)
        return []

    monkeypatch.setattr(render, "render_windows", rendered)
    grid = ResolvedElement("agent-grid", "graphic", "agent", start, end, layer,
                           {"template": template, "slots": {}})
    timeline = ResolvedTimeline((grid,), (), (), 150, 0)
    placement = {"id": grid.id, "layer": layer, "start_frame": start, "frames": end - start,
                 "clearance": {"status": "behind_head", "margin_px": 24, "gap_px": gap,
                               "tiles": [{"gap_px": 48 if gap is not None else None},
                                         {"gap_px": gap}]}}
    if blocked:
        with pytest.raises(TimelineRefusal, match="move the grid.*layer front") as caught:
            assembly.render_joins(tmp_path, timeline, cut, {"jump_cuts": mode},
                                  placements=[placement])
        assert caught.value.code == "card_grid_behind_join"
        assert "agent-grid" in str(caught.value)
        assert "too close to the speaker" in str(caught.value)
        assert "Move the grid" in caught.value.suggested_action
        assert "layer front" in caught.value.suggested_action
        assert not calls
    else:
        result = assembly.render_joins(tmp_path, timeline, cut, {"jump_cuts": mode},
                                       placements=[placement])
        assert result == ([], decisions if mode == "smooth" else None)
        assert calls == ([decisions] if mode == "smooth" else [])


@pytest.fixture
def card_join(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cut = tmp_path / "cut/cut.mp4"
    cut.parent.mkdir()
    (cut.parent / "receipt.json").write_text(json.dumps({"plan": {"pieces": []}}))
    (cut.parent / "cut-words.json").write_text(json.dumps({"words": []}))
    calls = []

    def rendered(*args, **kwargs):
        calls.append(True)
        return []

    monkeypatch.setattr(render, "render_windows", rendered)

    def run(template, layer="behind", gap=-24, handling="smooth", mode="smooth",
            placement_layer="behind", status="behind_head", recorded=True):
        decisions = plan.JoinPlan("smooth", "punch", [
            plan.JoinDecision(0, "removal", 78, 2.6, 75, 6, None, None, handling, ""),
        ])
        monkeypatch.setattr(render, "build_plan", lambda *args, **kwargs: decisions)
        card = ResolvedElement("card", "graphic", "agent", 0, 90, layer,
                               {"template": template, "slots": {}})
        clearance = {"status": status, "margin_px": 24}
        if gap is not None:
            clearance["gap_px"] = gap
        placements = [{"id": card.id, "layer": placement_layer, "clearance": clearance}]
        return assembly.render_joins(
            tmp_path, ResolvedTimeline((card,), (), (), 150, 0), cut, {"jump_cuts": mode},
            placements=placements if recorded else [])

    return run, calls


@pytest.mark.parametrize("template", [
    name for name in (*CARD_TEMPLATES, *registry_cards.kinds())
    if name not in {"playing_grid", "reel_feed"}
])
@pytest.mark.parametrize("layer,gap,handling,mode,recorded,blocked", [
    ("behind", -24, "smooth", "smooth", True, True),
    ("behind", 24, "smooth", "smooth", True, False),
    ("front", -24, "smooth", "smooth", True, False),
    ("behind", -24, "hard", "smooth", True, False),
    ("behind", -24, "punch", "smooth", True, False),
    ("behind", -24, "smooth", "hard", True, False),
    ("behind", -24, "smooth", "smooth", False, True),
])
def test_behind_cards_check_smooth_join_clearance(
    card_join, template: str, layer: str, gap: int, handling: str, mode: str,
    recorded: bool, blocked: bool,
) -> None:
    run, calls = card_join
    options = dict(layer=layer, gap=gap, handling=handling, mode=mode,
                   placement_layer=layer, recorded=recorded)
    if blocked:
        with pytest.raises(TimelineRefusal) as caught:
            run(template, **options)
        assert caught.value.code == "card_behind_join"
        assert str(caught.value) == (
            "card: the card behind the speaker is too close to her head without person "
            "restoration; move the card off the smooth join, or use layer front")
        assert caught.value.suggested_action == (
            "Move the card off the smooth join, or use layer front.")
        assert not calls
    else:
        run(template, **options)
        assert calls == ([True] if mode == "smooth" else [])


def test_planner_behind_card_does_not_suggest_requested_front_layer(card_join) -> None:
    run, calls = card_join
    with pytest.raises(TimelineRefusal) as caught:
        run("text_tag", layer="front")
    assert caught.value.code == "card_behind_join"
    assert str(caught.value).endswith("move the card off the smooth join")
    assert "layer front" not in str(caught.value)
    assert caught.value.suggested_action == "Move the card off the smooth join."
    assert not calls


def test_big_number_without_recorded_gap_refuses_smooth_join(card_join) -> None:
    run, calls = card_join
    with pytest.raises(TimelineRefusal) as caught:
        run("big_number", gap=None, status="behind_number")
    assert caught.value.code == "card_behind_join"
    assert not calls


def test_recorded_front_layer_takes_precedence_over_timeline(card_join) -> None:
    run, calls = card_join
    run("text_tag", placement_layer="front")
    assert calls == [True]


@pytest.mark.parametrize("margin,gap,blocked", [
    (None, 39, True), (None, 40, False), (None, 48, False),
    (24, 30, False), (48, 40, True), (0, 0, False),
])
def test_behind_grid_uses_style_margin_only_when_clearance_margin_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    margin: int | None, gap: int, blocked: bool,
) -> None:
    (tmp_path / "project.json").write_text(json.dumps({
        "schema": "yap-studio/project@1", "name": "Sample", "style": "style.json",
        "takes": [{"id": "t1", "path": "take.mov"}],
    }))
    (tmp_path / "style.json").write_text(json.dumps({
        "schema_version": 1, "graphics": {"face_clearance_px": 40},
    }))
    cut = tmp_path / "cut/cut.mp4"
    cut.parent.mkdir()
    (cut.parent / "receipt.json").write_text(json.dumps({"plan": {"pieces": []}}))
    (cut.parent / "cut-words.json").write_text(json.dumps({"words": []}))
    decisions = plan.JoinPlan("smooth", "punch", [
        plan.JoinDecision(0, "removal", 78, 2.6, 75, 6, None, None, "smooth", ""),
    ])
    monkeypatch.setattr(render, "build_plan", lambda *args, **kwargs: decisions)
    monkeypatch.setattr(render, "render_windows", lambda *args, **kwargs: [])
    grid = ResolvedElement("agent-grid", "graphic", "agent", 0, 90, "behind",
                           {"template": "playing_grid", "slots": {}})
    timeline = ResolvedTimeline((grid,), (), (), 150, 0)
    clearance = {"gap_px": gap, **({"margin_px": margin} if margin is not None else {})}
    if margin is not None:
        from yap_studio.project.status import ProjectContext

        monkeypatch.setattr(ProjectContext, "render_style",
                            lambda _: pytest.fail("Recorded margin must not read the style"))
    options = {"placements": [{"id": grid.id, "clearance": clearance}]}
    if blocked:
        with pytest.raises(TimelineRefusal, match="too close to the speaker") as caught:
            assembly.render_joins(tmp_path, timeline, cut, {"jump_cuts": "smooth"}, **options)
        assert caught.value.code == "card_grid_behind_join"
    else:
        assert assembly.render_joins(tmp_path, timeline, cut, {"jump_cuts": "smooth"},
                                     **options) == ([], decisions)
