"""Top text requires an answer and a per-video text choice without changing title pixels."""

import copy
import json

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

from yap_studio.errors import ValidationError
from yap_studio.project.model import parse_document
from yap_studio.project.status import ProjectContext
from yap_studio.stages.captions import title_layer_plan
from yap_studio.stages.graphics import graphics
from yap_studio.style import ResolvedStyle, resolve_style_data
from yap_studio.text_hook import require_top_text


def test_old_styles_keep_their_look_and_are_unanswered():
    for look in ("quiet_serif", "text_hook", "none"):
        hook = resolve_style_data({"schema_version": 1, "hook": {"style": look}})["hook"]
        assert hook["style"] == look and hook["answered"] is False
    with pytest.raises(ValidationError):
        resolve_style_data({"schema_version": 1, "hook": {"answered": "yes"}})


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
