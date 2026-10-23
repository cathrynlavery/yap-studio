"""Text hook choices travel through the existing home and review requests."""

import json

import pytest
from test_feedback_batch import feedback_project as feedback_project
from test_home_server import home as home
from test_home_server import request
from test_review_server import _in_process_request
from test_review_server import removed_server as removed_server


@pytest.mark.parametrize("text", ["Make it clear", "Rant"])
def test_home_text_hook_request(home, text):
    payload = {"kind": "edit", "video_type": "voiceover", "takes": [],
               "broll_sources": ["album"], "text_hook": f"  {text}  "}
    code, _, body = request(home, "POST", "/api/requests", payload)
    assert code == 201
    assert json.loads(body)["payload"]["text_hook"] == text


def test_home_shows_text_hook_for_saved_style(home):
    folder = home.workspace / "style"
    folder.mkdir()
    (folder / "my-style.json").write_text(json.dumps({
        "schema_version": 1, "name": "Synthetic", "hook": {"style": "text_hook"},
    }))
    assert json.loads(request(home)[2])["style"]["hook_style"] == "text_hook"


@pytest.mark.parametrize("text", [42, "wide " * 40, "W" * 40])
def test_home_refuses_unfitted_text_hook(home, text):
    payload = {"kind": "edit", "video_type": "voiceover", "takes": [],
               "broll_sources": ["album"], "text_hook": text}
    assert request(home, "POST", "/api/requests", payload)[0] == 400


def test_review_text_hook_choice_is_a_note(removed_server):
    server = removed_server
    target = server.series.project_dir / "plans/text-hook.json"
    target.parent.mkdir(exist_ok=True)
    plan = {"schema": "yap-studio/text-hook@1", "ideas": [
        "Make it clear", "Start with the hard part", "What makes this work?",
    ], "chosen": None, "source": "idea"}
    target.write_text(json.dumps(plan))
    def get():
        return json.loads(_in_process_request(server, "GET", "/api/review?version=2")[2])

    assert get()["text_hook"] == plan
    code, _, body = _in_process_request(server, "POST", "/api/comments?version=2", {
        "text_hook": {"text": plan["ideas"][0], "source": "idea"},
    })
    assert code == 201, body
    assert json.loads(body)["text"] == 'Text hook: "Make it clear"'
    assert json.loads(target.read_text())["chosen"] == "Make it clear"
    assert get()["text_hook"]["chosen"] == "Make it clear"
    assert get()["comments"][0]["text"] == 'Text hook: "Make it clear"'
    assert _in_process_request(server, "POST", "/api/comments?version=2", {
        "text_hook": {"text": "Not an offered idea", "source": "idea"},
    })[0] == 400
    saved = target.read_bytes()
    for bad_source in ([], {}, None, 42):
        assert _in_process_request(server, "POST", "/api/comments?version=2", {
            "text_hook": {"text": "Make it clear", "source": bad_source},
        })[0] == 400
    assert _in_process_request(server, "POST", "/api/comments?version=1", {
        "text_hook": {"text": "Make it clear", "source": "typed"},
    })[0] == 400
    assert _in_process_request(server, "POST", "/api/comments?version=2", {
        "text_hook": {"text": "wide " * 40, "source": "typed"},
    })[0] == 400
    assert target.read_bytes() == saved and len(get()["comments"]) == 1
