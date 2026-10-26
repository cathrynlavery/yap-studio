"""Top text requires an answer and a per-video text choice without changing title pixels."""

import copy
import json
from pathlib import Path

import pytest
from test_feedback_batch import feedback_project as feedback_project
from test_home_server import home as home
from test_home_server import request
from test_review_server import _in_process_request
from test_review_server import removed_server as removed_server
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements
from test_text_hook import choice
from test_timeline import write

from yap_studio.brand_words import set_words
from yap_studio.errors import ValidationError
from yap_studio.project.model import parse_document
from yap_studio.project.status import ProjectContext
from yap_studio.stages.captions import title_layer_plan
from yap_studio.stages.graphics import graphics, graphics_snapshot
from yap_studio.style import ResolvedStyle, resolve_style_data
from yap_studio.text_hook import require_top_text


def test_old_styles_keep_their_look_and_are_unanswered():
    for look in ("quiet_serif", "text_hook", "none"):
        hook = resolve_style_data({"schema_version": 1, "hook": {"style": look}})["hook"]
        assert hook["style"] == look and hook["answered"] is False
    with pytest.raises(ValidationError):
        resolve_style_data({"schema_version": 1, "hook": {"answered": "yes"}})


def test_old_whole_video_style_without_choice_keeps_rendering(layer_project):
    data = json.loads((layer_project / "style.json").read_bytes())
    data["hook"] = {"style": "text_hook"}
    write(layer_project / "style.json", data)
    set_elements(layer_project, {"id": "agent-title", "kind": "title",
                                "start": {"time": 0}, "hold_s": 1, "lines": ["Small steps."]})
    assert not (layer_project / "plans/text-hook.json").exists()
    assert not graphics_snapshot(ProjectContext(layer_project)).titles
    assert graphics(layer_project)["elements"] == []
    assert not (layer_project / "plans/text-hook.json").exists()


@pytest.mark.parametrize("raw,displayed", [
    ("Twenty dollars", "$20"), ("sparrowlight", "SparrowLight"),
    ("spoiler", "s******"),
])
def test_opening_choice_checks_displayed_words(layer_project, raw, displayed):
    data = json.loads((layer_project / "style.json").read_bytes())
    data["hook"]["answered"] = True
    if raw == "spoiler":
        data["music"] = {"bleep_swears": True, "bleep_words_add": ["spoiler"]}
    write(layer_project / "style.json", data)
    set_words(["SparrowLight"])
    set_elements(layer_project, {"id": "agent-title", "kind": "title",
                                "start": {"time": 0}, "hold_s": 1, "lines": [raw]})
    write(layer_project / "plans/text-hook.json", {
        **choice(raw, "typed"), "kind": "opening",
    })
    with pytest.raises(ValidationError, match="Use the chosen opening text"):
        require_top_text(ProjectContext(layer_project))
    write(layer_project / "plans/text-hook.json", {
        **choice(displayed, "typed"), "kind": "opening",
    })
    context = ProjectContext(layer_project)
    require_top_text(context)
    title, = graphics_snapshot(context).titles
    assert json.loads(title.plan.variables["lines"]) == [displayed]


def test_none_omits_existing_titles_without_editing_timeline(layer_project):
    set_elements(layer_project, {"id": "agent-title", "kind": "title",
                                "start": {"time": 0}, "hold_s": 1, "lines": ["Small steps."]})
    timelines = {file: file.read_bytes() for file in layer_project.glob("timeline/*.json")}
    assert timelines
    data = json.loads((layer_project / "style.json").read_bytes())
    data["hook"].update(style="none", answered=True)
    write(layer_project / "style.json", data)
    context = ProjectContext(layer_project)
    require_top_text(context)
    assert not any(item.kind == "title" for item in context.resolved_timeline().elements)
    assert graphics(layer_project)["elements"] == []
    assert all(file.read_bytes() == content for file, content in timelines.items())


def test_changelog_describes_top_text_choice():
    changelog = (Path(__file__).parents[1] / "CHANGELOG.md").read_text()
    entry, = [line for line in changelog.splitlines()
              if line.startswith("- ") and "Text at the top" in line]
    for phrase in ("style page", "no default", "Home", "waits for a pick"):
        assert phrase in entry


def test_pick_my_look_guidance_lists_current_steps():
    guide = (Path(__file__).parents[1] / "workspace/AGENTS.md").read_text()
    guidance = guide.split("## Pick my look\n", 1)[1].split("\n## ", 1)[0]
    assert "Font, Captions, Colors, Text at the top, then" in guidance


@pytest.mark.parametrize("look", [
    "paper_title", "quiet_serif", "bold_band", "calm_card", "lockup", "write_on",
    "headline_slam", "shard_reveal", "textured", "shimmer", "stagger", "morph",
    "typewriter", "video_letters", "kinetic",
])
def test_answer_metadata_keeps_every_vertical_title_plan_identical(look):
    data = resolve_style_data({"schema_version": 1, "hook": {"style": look}})
    before = title_layer_plan("Make it", "clear", ResolvedStyle("Test", "test", data))
    answered = copy.deepcopy(data)
    answered["hook"]["answered"] = True
    after = title_layer_plan("Make it", "clear", ResolvedStyle("Test", "test", answered))
    assert before == after


@pytest.mark.render
def test_opening_answer_keeps_rendered_vertical_layer_bytes_identical(tmp_path):
    from yap_studio import paths
    from yap_studio.stages.captions import render_layer_plan

    data = resolve_style_data({"schema_version": 1, "hook": {"style": "quiet_serif"}})
    rendered = []
    with paths.operation():
        for answered in (False, True):
            data["hook"]["answered"] = answered
            plan = title_layer_plan("Make it", "clear", ResolvedStyle("Test", "test", data),
                                    frames=12)
            folder = tmp_path / str(answered)
            output = render_layer_plan(plan, folder, folder / "title.mov").path
            rendered.append(output.read_bytes())
    assert rendered[0] == rendered[1]


@pytest.mark.parametrize("look,kind", [("quiet_serif", "opening"), ("text_hook", "text_hook")])
def test_render_waits_for_chosen_text(layer_project, look, kind):
    data = json.loads((layer_project / "style.json").read_bytes())
    data["hook"] = {"style": look, "answered": True}
    write(layer_project / "style.json", data)
    plan = {**choice(), "kind": kind}
    for value in (None, plan):
        if value:
            write(layer_project / "plans/text-hook.json", value)
        with pytest.raises(ValidationError, match="Choose the text at the top"):
            graphics(layer_project)
    plan["chosen"] = plan["ideas"][0]
    write(layer_project / "plans/text-hook.json", plan)
    if kind == "opening":
        with pytest.raises(ValidationError, match="Use the chosen opening text"):
            require_top_text(ProjectContext(layer_project))
        set_elements(layer_project, {"id": "agent-title", "kind": "title",
                                    "start": {"time": 0}, "hold_s": 1, "lines": [plan["chosen"]]})
    require_top_text(ProjectContext(layer_project))
    plan["kind"] = "opening" if kind == "text_hook" else "text_hook"
    write(layer_project / "plans/text-hook.json", plan)
    with pytest.raises(ValidationError, match="Choose the text at the top"):
        require_top_text(ProjectContext(layer_project))


@pytest.mark.parametrize("answer", ["none", "opening", "whole"])
def test_home_sends_top_text_override_without_changing_style(home, answer):
    payload = {"kind": "edit", "video_type": "voiceover", "takes": [],
               "broll_sources": ["album"], "top_text": answer, "text_hook": "My own text"}
    code, _, body = request(home, "POST", "/api/requests", payload)
    assert code == 201
    sent = json.loads(body)["payload"]
    assert sent["top_text"] == answer
    assert sent.get("text_hook") == (None if answer == "none" else "My own text")
    assert not (home.workspace / "style/my-style.json").exists()


@pytest.mark.parametrize("answer", ["bad", 1, [], {}])
def test_home_refuses_unknown_top_text_answers(home, answer):
    assert request(home, "POST", "/api/requests", {
        "kind": "edit", "video_type": "voiceover", "takes": [],
        "broll_sources": ["album"], "top_text": answer,
    })[0] == 400


@pytest.mark.parametrize("answer", ["none", "opening", "whole", {"stale": True}])
def test_home_drops_top_text_for_horizontal(home, monkeypatch, answer):
    (home.workspace / "inbox/wide.mp4").write_bytes(b"synthetic")
    monkeypatch.setattr(home.state, "metadata", lambda p: ("take", 1.0, 320, 180))
    code, _, body = request(home, "POST", "/api/requests", {
        "kind": "edit", "video_type": "talking_head", "broll_sources": [],
        "takes": ["wide.mp4"], "shape": "horizontal", "captions": True,
        "top_text": answer, "text_hook": "A stale vertical title " * 100,
    })
    assert code == 201
    assert json.loads(body)["payload"] == {
        "video_type": "talking_head", "broll_sources": [], "takes": ["wide.mp4"],
        "album": None, "shape": "horizontal", "captions": True,
    }


def test_review_opening_title_choice_saves_plan_and_note(removed_server):
    target = removed_server.series.project_dir / "plans/text-hook.json"
    target.parent.mkdir(exist_ok=True)
    plan = {**choice(), "kind": "opening"}
    target.write_text(json.dumps(plan))
    assert parse_document("text-hook", plan).to_dict() == plan
    code, _, body = _in_process_request(removed_server, "POST", "/api/comments?version=2", {
        "text_hook": {"text": plan["ideas"][1], "source": "idea"},
    })
    assert code == 201
    assert json.loads(body)["text"] == f'Opening title: "{plan["ideas"][1]}"'
    assert json.loads(target.read_text()) == {**plan, "chosen": plan["ideas"][1]}
