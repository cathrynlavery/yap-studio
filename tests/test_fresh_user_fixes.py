"""Regressions from the fresh creator's review and final-render flow."""

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from test_feedback_batch import _cut, _json
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_locate import WORDS
from test_feedback_replies import COMMENTS, _round
from test_planner_cards import fake_render, plan, requests, synthetic_faces

from yap_studio import paths
from yap_studio.faces.clearance import gap_px, head_box
from yap_studio.faces.model import Box
from yap_studio.feedback.batch import FeedbackError, load_batch, set_resulting_version
from yap_studio.feedback.edits import validate_operations
from yap_studio.feedback.replies import reply_status, write_reply
from yap_studio.feedback.versions import save_version
from yap_studio.planner import cards as planner_cards
from yap_studio.project import status as status_module
from yap_studio.project.model import CutWord, CutWords
from yap_studio.review.video_map import video_map
from yap_studio.style import resolve


def test_note_can_be_completed_in_a_later_version(feedback_project, tmp_path, monkeypatch):
    root = feedback_project
    store, _, _ = _round(root, tmp_path / "data", comments=[COMMENTS[0], COMMENTS[2]])
    set_resulting_version(root, "b1", "v2")
    write_reply(root, store, "b1", "v1", "c1", "changed", "Removed the opening")
    with pytest.raises(FeedbackError, match="Nothing changed"):
        write_reply(root, store, "b1", "v1", "c2", "changed", "Removed this word")
    _cut(root, words=CutWords("output", {}, WORDS.words[3:]), label="third render")
    third = save_version(root)
    reply = write_reply(root, store, "b1", "v1", "c2", "changed", "Removed this word",
                        version_id=third.id)
    assert reply.version == 3 and reply.evidence["version"] == "v3"
    assert load_batch(root, "b1")["resulting_version"] == "v2"
    result = reply_status(root, store, "b1")
    assert result["complete"]
    assert result["items"][1]["reply"]["id"] == reply.id
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(store.root))
    monkeypatch.setattr(status_module, "STAGES", ())
    assert not any("reply to" in need for need in status_module.status(root)["needs"])


def test_later_title_is_measured_against_a_raw_story_cut(feedback_project, tmp_path):
    store, _, _ = _round(feedback_project, tmp_path / "data", comments=[COMMENTS[2]],
                         next_words=WORDS)
    set_resulting_version(feedback_project, "b1", "v2")
    layers = feedback_project / "final-layers.json"
    _json(layers, {"schema": "yap-studio/layer-windows@1", "time_base": "output", "layers": [
        {"id": "title", "kind": "title", "text": "Requested title", "start": 0, "end": 3},
    ]})
    _cut(feedback_project, label="third render")
    save_version(feedback_project, layers=layers)
    reply = write_reply(feedback_project, store, "b1", "v1", "c1", "changed", "Added the title",
                        version_id="v3")
    assert reply.evidence["change"]["differences"] == ["layers"]
    assert reply_status(feedback_project, store, "b1")["complete"]


@pytest.mark.parametrize("remaining,expected", [
    (-1, "removed"), (0, "removed"), (1, "partial"), (2, "mapped"),
])
def test_cut_label_measures_requested_words_not_neighbours(
    feedback_project, tmp_path, remaining, expected,
):
    words = CutWords("output", {}, tuple(
        CutWord(f"t1:{i}", "s1", text, i * 0.6 + 0.1, i * 0.6 + 0.5)
        for i, text in enumerate(["before", "cut", "these", "after"])
    ))
    requested = [{"id": word.id, "segment": word.segment, "text": word.text}
                 for word in words.words[1:3]]
    comment = {"kind": "fix", "target": "story", "text": "Cut these words",
               "start_frame": 3, "end_frame": 69,
               "request": {"type": "cut", "start": 0.1, "end": 2.3, "words": requested}}
    after = replace(words, words=(words.words[0], *words.words[1:1 + remaining], words.words[3]))
    if remaining == -1:
        after = replace(words, words=())
    store, _, _ = _round(feedback_project, tmp_path / "data", comments=[comment],
                         own_words=words, next_words=after)
    reply = write_reply(feedback_project, store, "b1", "v1", "c1",
                        "changed" if remaining < 2 else "declined", "Checked the requested words",
                        version_id="v2")
    assert reply.evidence["moment"]["status"] == expected
    assert [word["text"] for word in reply.evidence["change"]["before"]] == ["cut", "these"]


@pytest.mark.parametrize("template,slots", [
    ("steps", {"steps": ["Water", "Sunlight", "One thing"]}),
    ("text_tag", {"text": "One small win"}),
])
def test_sample_presenter_leaves_room_for_ordinary_cards(template, slots, tmp_path, monkeypatch):
    face = Box(324, 603, 484, 484)
    style = resolve("preset:clean_minimal")
    result = plan(requests(template=template, slots=slots),
                  faces=synthetic_faces(face), style=style)
    card = result.cards[0]
    assert card.clearance["status"] == "clear"
    assert gap_px(Box(card.x, card.y, card.width, card.height), head_box(face)) >= 24
    assert card.y + card.height <= style.data["captions"]["top_px"] - 100
    monkeypatch.setattr(planner_cards, "render_card", fake_render)
    with paths.operation():
        rendered = planner_cards.render_layers(result, style, tmp_path / "cards")
    assert rendered[0].height == card.height


def test_edit_skill_has_a_complete_valid_operations_file():
    skill = (Path(__file__).parents[1] / "workspace/.claude/skills/edit-video/SKILL.md").read_text()
    examples = re.findall(r"```json\s*(.*?)```", skill, re.S)
    documents = [json.loads(example) for example in examples]
    document = next(item for item in documents
                    if item.get("schema") == "yap-studio/feedback-operations@1")
    assert validate_operations(document) == document
    assert document["items"][0]["operations"][0]["kind"] == "cut"


def test_story_map_uses_saved_words_and_a_plain_wordless_label(feedback_project):
    manifest = save_version(feedback_project)
    labels = [item["label"] for item in video_map(manifest)[0]["items"]]
    assert labels == ["Start really", "Join", "Start End"]
    wordless = replace(manifest, snapshots={**manifest.snapshots, "cut_words": None})
    assert [item["label"] for item in video_map(wordless)[0]["items"]] == [
        "Part 1", "Join", "Part 2",
    ]
