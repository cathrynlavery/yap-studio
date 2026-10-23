"""Chapter contracts, output clocks, palette choices and full-frame plans."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements
from test_timeline import fragment, insert, write, zoom
from test_timeline import project as project

from yap_studio.errors import ValidationError
from yap_studio.project.model import SCHEMAS_DIR, ProjectFileError, parse_document
from yap_studio.project.status import ProjectContext
from yap_studio.stages.captions import (
    _contrast,
    chapter_layer_plan,
    chapter_number,
    chapter_timecode,
)
from yap_studio.stages.graphics import graphics_snapshot
from yap_studio.stages.timeline import add_timeline
from yap_studio.style import ResolvedStyle, resolve, resolve_style_data


def chapter(**changes) -> dict:
    return {"id": "agent-ch-tokens", "kind": "chapter", "lane": "agent", "start": {"word": "t1:0"},
            "title": "How tokens are counted", **changes}


def chapter_style(**sections) -> ResolvedStyle:
    data = resolve_style_data({"schema_version": 1, "name": "Synthetic chapters",
                               "chapters": {"style": "title_card"}, **sections})
    return ResolvedStyle(data["name"], "test", data)


@pytest.mark.parametrize("number,expected", [(1, "01"), (2, "02"), (10, "10"), (100, "100")])
def test_chapter_number(number: int, expected: str) -> None:
    assert chapter_number(number) == expected


@pytest.mark.parametrize("frame,expected", [
    (0, "00:00"), (29, "00:00"), (84 * 30 + 29, "01:24"),
    (3599 * 30, "59:59"), (3600 * 30, "1:00:00"), (3661 * 30, "1:01:01"),
])
def test_chapter_timecode(frame: int, expected: str) -> None:
    assert chapter_timecode(frame) == expected


@pytest.mark.parametrize("ending", [{}, {"hold_s": 1.6}, {"end": {"word": "t1:2"}}])
@pytest.mark.parametrize("name", ["timeline", "timeline-fragment"])
def test_chapter_model_and_schema(ending: dict, name: str) -> None:
    data = {"schema": "yap-studio/timeline@1", "words_sha256": {},
            "elements": [chapter(lane="agent", **ending)]}
    if name == "timeline-fragment":
        data.pop("words_sha256")
    schema = json.loads((SCHEMAS_DIR / f"{name}.schema.json").read_bytes())
    jsonschema.validate(data, schema)
    assert parse_document(name, data, lane="agent").elements[0].fields["title"]
    for title in ("", "x" * 49):
        data["elements"][0]["title"] = title
        with pytest.raises(ProjectFileError):
            parse_document(name, data, lane="agent")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(data, schema)
    data["elements"][0]["title"] = "x" * 48
    jsonschema.validate(data, schema)
    data["elements"][0]["lane"] = "broll"
    with pytest.raises(ProjectFileError):
        parse_document(name, data, lane="broll")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema)


def test_chapter_default_hold_and_anchor(project: Path) -> None:
    write(project / "style.json", chapter_style(chapters={"style": "title_card", "hold_s": 2}).data)
    data = json.loads((project / "project.json").read_bytes())
    data["style"] = "style.json"
    write(project / "project.json", data)
    result = add_timeline(project, fragment(project, chapter()), lane="agent")
    assert result.valid
    item = result.elements[0]
    assert (item.start_frame, item.end_frame, item.layer) == (6, 66, "front")


@pytest.mark.parametrize("hold", [.9, 4.1])
def test_chapter_duration_violation_has_fix(project: Path, hold: float) -> None:
    result = add_timeline(project, fragment(project, chapter(hold_s=hold)), lane="agent")
    assert not result.valid
    finding = next(item for item in result.findings if item.rule == "chapter_hold")
    assert finding.severity == "violation" and finding.suggested_action


def test_chapters_cannot_overlap_even_with_relaxed_visual_limit(project: Path) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["rules"] = {"elements_at_once_max": 3}
    write(project / "project.json", data)
    result = add_timeline(project, fragment(project, chapter(hold_s=2),
        chapter(id="agent-ch-next", start={"word": "t1:1"}, hold_s=2)), lane="agent")
    assert not result.valid
    assert any(item.rule == "chapter_overlap" and item.suggested_action for item in result.findings)


def test_chapter_over_two_inserts_still_exceeds_the_default_cap(project: Path) -> None:
    add_timeline(project, fragment(project, chapter(start={"time": 0}, hold_s=3)),
                 lane="agent").require_valid()
    result = add_timeline(project, fragment(project,
        insert(project, start={"time": 1}, end={"time": 3}),
        insert(project, id="broll-two", start={"time": 1}, end={"time": 3})), lane="broll")
    assert not result.valid
    assert any(item.rule == "visual_overlap"
               and set(item.elements) == {"agent-ch-tokens", "broll-one", "broll-two"}
               for item in result.findings)


@pytest.mark.parametrize("limit", [1, 3])
def test_chapter_over_zoom_stays_exclusive(project: Path, limit: int) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["rules"] = {"elements_at_once_max": limit}
    write(project / "project.json", data)
    result = add_timeline(project, fragment(project,
        chapter(start={"time": 1}, hold_s=1), zoom()), lane="agent")
    assert not result.valid
    assert any(item.rule == "visual_overlap" for item in result.findings)


@pytest.mark.parametrize("caption,paper,ink,background,text", [
    ("#FFFFFF", "#FFFFFF", "#111111", "#111111", "#FFFFFF"),
    ("#111111", "#FFFFFF", "#111111", "#FFFFFF", "#111111"),
    ("#FFFFFF", "#F6F1E7", "#1C1B19", "#1C1B19", "#F6F1E7"),
])
def test_chapter_plan_uses_caption_contrast(caption, paper, ink, background, text) -> None:
    style = chapter_style(captions={"text_color": caption},
                          graphics={"paper_color": paper, "ink_color": ink})
    plan = chapter_layer_plan("How tokens are counted", 2, style, start_frame=84 * 30, frames=48)
    assert (plan.template, plan.x, plan.y, plan.width, plan.height) == ("chapter", 0, 0, 1080, 1920)
    assert (plan.start_frame, plan.frames) == (2520, 48)
    assert plan.variables["number"] == "02" and plan.variables["timecode"] == "01:24"
    assert plan.variables["background_color"] == background
    assert plan.variables["text_color"] == text
    assert plan.variables["accent_color"] == style.data["graphics"]["accent_color"]
    assert plan.variables["font_name"] == style.data["hook"]["font"]


@pytest.mark.parametrize("paper,ink,caption,text", [
    ("#111111", "#111111", "#FFFFFF", "#FFFFFF"),
    ("#111111", "#222222", "#FFFFFF", "#FFFFFF"),
    ("#FFFFFF", "#FFFFFF", "#111111", "#111111"),
    ("#777777", "#888888", "#111111", "#FFFFFF"),
    ("#777777", "#777777", "#111111", "#FFFFFF"),
])
def test_chapter_plan_falls_back_to_readable_text(paper, ink, caption, text) -> None:
    style = chapter_style(captions={"text_color": caption},
                          graphics={"paper_color": paper, "ink_color": ink,
                                    "accent_color": "#AABBCC"})
    plan = chapter_layer_plan("A chapter", 1, style)
    assert plan.variables["text_color"] == text
    wash = plan.variables["background_color"]
    for ground in (0, 255):
        composite = "#" + "".join(f"{round(.7 * int(wash[i:i + 2], 16) + .3 * ground):02X}"
                                   for i in (1, 3, 5))
        assert _contrast(composite, text) >= 4.5
    assert plan.variables["accent_color"] == "#AABBCC"


def test_chapter_off_and_long_title_refusals() -> None:
    with pytest.raises(ValidationError) as caught:
        chapter_layer_plan("Tokens", 1, resolve("preset:clean_minimal"), element_id="agent-ch")
    assert caught.value.parameter == "chapter.agent-ch"
    assert caught.value.detail == ("the style has no chapter title card; choose Title card under "
                                   "chapters on the style page, or remove the chapter")
    with pytest.raises(ValidationError, match=r"shorten the chapter title to \d+ characters"):
        chapter_layer_plan("W" * 48, 1, chapter_style())


def test_chapter_snapshot_numbering_digests_and_receipt(layer_project: Path) -> None:
    write(layer_project / "style.json", chapter_style().data)
    set_elements(layer_project, chapter(start={"time": 2}, hold_s=1),
                 chapter(id="agent-ch-first", start={"time": 0}, hold_s=1))
    snapshot = graphics_snapshot(ProjectContext(layer_project))
    layers = {item.id: item for item in snapshot.titles}
    assert layers["agent-ch-first"].plan.variables["number"] == "01"
    later = layers["agent-ch-tokens"]
    assert later.plan.variables["number"] == "02"
    assert later.placement()["kind"] == "chapter" and later.layer == "front"
    assert later.clearance == {"status": "full_frame",
                               "reason": "chapter text sits over the live picture by design"}
    restored = graphics_snapshot(ProjectContext(layer_project),
                                 placements=[item.placement() for item in snapshot.titles])
    assert restored.inputs == snapshot.inputs
    set_elements(layer_project, chapter(start={"time": 2}, hold_s=1))
    renumbered = graphics_snapshot(ProjectContext(layer_project)).titles[0]
    assert renumbered.plan.start_frame == later.plan.start_frame
    assert renumbered.plan.variables["number"] == "01" and renumbered.digest != later.digest
    set_elements(layer_project, chapter(start={"time": 1.9}, hold_s=1),
                 chapter(id="agent-ch-first", start={"time": 0}, hold_s=1))
    moved = graphics_snapshot(ProjectContext(layer_project))
    assert next(item for item in moved.titles if item.id == later.id).digest != later.digest


def test_chapter_snapshot_off_refusal_names_element(layer_project: Path) -> None:
    set_elements(layer_project, chapter(start={"time": 0}, hold_s=1))
    with pytest.raises(ValidationError) as caught:
        graphics_snapshot(ProjectContext(layer_project))
    assert caught.value.parameter == "chapter.agent-ch-tokens"


def test_chapter_template_declares_plan_variables() -> None:
    root = Path(__file__).parents[1] / "src/yap_studio/templates/chapter"
    source = (root / "template.html").read_text()
    metadata = json.loads((root / "template.json").read_text())
    declared = json.loads(source.split("data-composition-variables='", 1)[1].split("'", 1)[0])
    assert declared == metadata["variables"]
    plan = chapter_layer_plan("Sample chapter", 2, chapter_style())
    assert set(plan.variables) == {item["id"] for item in declared}


def test_chapter_defaults_presets_and_brand_font() -> None:
    from yap_studio.style import SHIPPED_PRESETS

    for preset in SHIPPED_PRESETS:
        assert resolve(f"preset:{preset}").data["chapters"] == {"style": "none", "hold_s": 1.6}
    style = chapter_style(brand={"titles_font": "Gelasio"})
    assert chapter_layer_plan("Tokens", 1, style).variables["font_name"] == "Gelasio"


@pytest.mark.parametrize("hold", [.9, 4.1])
def test_chapter_style_hold_limits(hold: float) -> None:
    with pytest.raises(ValidationError):
        chapter_style(chapters={"hold_s": hold})


@pytest.mark.parametrize("hold,frames", [(1.05, 32), (1.15, 35), (1.6, 48), (2.05, 62)])
def test_chapter_preview_hold_uses_the_timeline_frame_rounding(hold: float, frames: int) -> None:
    style = chapter_style(chapters={"style": "title_card", "hold_s": hold})
    assert chapter_layer_plan("Tokens", 1, style).frames == frames


def test_chapter_font_ignores_caption_font() -> None:
    first = chapter_style(hook={"font": "Gelasio"}, captions={"font": "Inter"})
    second = chapter_style(hook={"font": "Gelasio"}, captions={"font": "Instrument Serif"})
    before = chapter_layer_plan("Small clear steps", 1, first)
    after = chapter_layer_plan("Small clear steps", 1, second)
    assert before == after
    assert before.variables["font_name"] == "Gelasio"
