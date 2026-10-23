"""Per-video generated approval covers inserts and every media card slot."""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager

import pytest
import synth_broll
from pictures_helpers import record_answer
from test_pictures_make import asked as asked
from test_pictures_make import make
from test_pictures_sheet import STYLE
from test_stock_guard_cards import element
from test_stock_sheet import words
from test_timeline import fragment, insert, write
from test_timeline import project as project

from yap_studio import cards, cli, receipt
from yap_studio.errors import OutputPathError, ValidationError
from yap_studio.media.catalog import add_generated_item, load_catalog
from yap_studio.pictures import guard, wavespeed
from yap_studio.project.model import Anchor, Element
from yap_studio.stages import timeline
from yap_studio.stages.timeline import TimelineRefusal, add_timeline


def approve(root, digest):
    write(
        root / "pictures/pictures.json",
        {"schema": guard.PICTURES_SCHEMA, "pictures": [{"sha256": digest}]},
    )


def enable(root, enabled=True):
    record_answer(root)
    write(
        root / "pictures-style.json",
        {"schema_version": 1, "broll": {"generated_pictures": "ask" if enabled else "off"}},
    )
    document = json.loads((root / "project.json").read_text())
    write(root / "project.json", {**document, "style": "pictures-style.json"})


@pytest.mark.parametrize("kind", ["insert", "overlay", "background"])
def test_timeline_add_approval_and_style(project, kind):
    enable(project)
    folder = project / "pictures/files"
    folder.mkdir(parents=True)
    source = synth_broll.png(folder / "abc123.png", size="108x192")
    digest = receipt.sha256_file(source)
    candidate = insert(
        project, kind=kind, media={"path": "pictures/files/abc123.png", "sha256": digest}
    )
    if kind == "overlay":
        candidate["placement"] = "headroom"
    incoming = fragment(project, candidate)
    with pytest.raises(TimelineRefusal, match="not approved") as error:
        add_timeline(project, incoming, lane="broll")
    assert error.value.suggested_action == "Ask with yap-studio pictures ask, then pictures make."
    approve(project, digest)
    add_timeline(project, incoming, lane="broll").require_valid()
    enable(project, enabled=False)
    with pytest.raises(TimelineRefusal, match="Generated pictures are off"):
        add_timeline(project, incoming, lane="broll")


def test_timeline_rechecks_imagery_answer_under_write_lock(project, monkeypatch):
    enable(project)
    add_timeline(project, fragment(project, insert(project)), lane="broll").require_valid()
    target = project / "timeline/broll.json"
    before = target.read_bytes()
    folder = project / "pictures/files"
    folder.mkdir(parents=True)
    source = synth_broll.png(folder / "approved.png", size="108x192")
    digest = receipt.sha256_file(source)
    approve(project, digest)
    incoming = fragment(project, insert(
        project, media={"path": "pictures/files/approved.png", "sha256": digest},
    ))
    original = timeline.project_lock

    @contextmanager
    def revoke_before_lock(root):
        record_answer(root, "no")
        with original(root):
            yield

    monkeypatch.setattr(timeline, "project_lock", revoke_before_lock)
    with pytest.raises(TimelineRefusal, match=(
        "The creator said no to generated pictures for this video."
    )):
        add_timeline(project, incoming, lane="broll")
    assert guard.imagery_answer(project) == "no"
    assert target.read_bytes() == before


def test_timeline_recognizes_catalog_hash_at_another_path(project):
    enable(project)
    source = synth_broll.png(project / "picture.png")
    generated = {
        "provider": "wavespeed",
        "model": wavespeed.MODEL_ID,
        "prompt": "A park",
        "seed": None,
        "size": wavespeed.SIZE,
        "prediction_id": "abc123",
        "cost_usd": 0.027,
        "cost_source": "estimate",
        "created_at": "2026-10-19T12:00:00Z",
    }
    item = add_generated_item(source, generated=generated, description_text="A park")
    renamed = project / "renamed.png"
    shutil.copyfile(source, renamed)
    incoming = fragment(
        project, insert(project, media={"path": renamed.name, "sha256": item["sha256"]})
    )
    with pytest.raises(TimelineRefusal, match="not approved"):
        add_timeline(project, incoming, lane="broll")
    approve(project, item["sha256"])
    add_timeline(project, incoming, lane="broll").require_valid()


@pytest.mark.parametrize("slot_type", ["image", "image_list", "video", "video_list"])
@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("default", [False, True])
def test_all_media_card_slots(tmp_path, monkeypatch, slot_type, as_string, default):
    record_answer(tmp_path)
    folder = tmp_path / "pictures/files"
    folder.mkdir(parents=True)
    source = synth_broll.png(folder / "abc123.png")
    digest = receipt.sha256_file(source)
    media = (
        "pictures/files/abc123.png"
        if as_string
        else {"path": "pictures/files/abc123.png", "sha256": digest}
    )
    value = ["self", media] if slot_type.endswith("_list") else media
    monkeypatch.setattr(cards, "CARD_TEMPLATES", ("synthetic_card",))
    monkeypatch.setattr(
        cards,
        "load_template",
        lambda _: {"slots": {"picture": {"type": slot_type, "default": value}}},
    )
    candidate = element({} if default else {"picture": value}, "synthetic_card")
    with pytest.raises(TimelineRefusal, match="not approved"):
        guard.check_timeline(tmp_path, [candidate], STYLE)
    approve(tmp_path, digest)
    guard.check_timeline(tmp_path, [candidate], STYLE)
    record_answer(tmp_path, "no")
    with pytest.raises(TimelineRefusal, match="creator said no") as error:
        guard.check_timeline(tmp_path, [candidate], STYLE)
    assert error.value.suggested_action == (
        "Carry on with their own media. Ask again only if they bring it up."
    )


def test_project_answer_filters_and_validates_directly(tmp_path):
    item = {"id": "m1", "sha256": "a" * 64, "generated": {}}
    requests = {"broll": [{"item": "m1"}]}
    approve(tmp_path, item["sha256"])
    record_answer(tmp_path)
    assert guard.generated_allowed(STYLE, tmp_path) == (True, None)
    assert guard.filter_approved([item], STYLE, tmp_path) == ([item], [])
    guard.validate_requests(requests, [item], STYLE, tmp_path)
    assert not guard.generated_allowed({}, tmp_path)[0]
    record_answer(tmp_path, "no")
    assert guard.generated_allowed(STYLE, tmp_path) == (
        False, "The creator said no to generated pictures for this video."
    )
    allowed, excluded = guard.filter_approved([item], STYLE, tmp_path)
    assert allowed == [] and excluded == [
        {"item": "m1", "reasons": ["generated picture not approved for this video"]}
    ]
    with pytest.raises(ValidationError) as error:
        guard.validate_requests(requests, [item], STYLE, tmp_path)
    assert error.value.parameter == "broll[0].item"


def test_own_string_path_needs_no_hash(tmp_path, monkeypatch):
    source = synth_broll.png(tmp_path / "own.png")
    monkeypatch.setattr(receipt, "sha256_file", lambda _: pytest.fail("no generated media to hash"))
    candidate = Element("broll-own", "insert", Anchor(time=1), fields={"media": source.name})
    guard.check_timeline(tmp_path, [candidate], {})


def test_own_items_ignore_broken_imagery_answer(tmp_path):
    write(tmp_path / "pictures/imagery.json", {})
    items = [{"id": "m1", "sha256": "a" * 64}, {"id": "m2", "sha256": "b" * 64}]
    requests = {"broll": [{"item": "m1"}]}
    for style in (STYLE, None):
        assert guard.filter_approved(items, style, tmp_path) == (items, [])
        guard.validate_requests(requests, items, style, tmp_path)
    items.append({"id": "m3", "sha256": "c" * 64, "generated": {}})
    with pytest.raises(ValidationError, match="pictures/imagery.json"):
        guard.filter_approved(items, STYLE, tmp_path)
    with pytest.raises(ValidationError, match="pictures/imagery.json"):
        guard.validate_requests(requests, items, STYLE, tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema": guard.PICTURES_SCHEMA, "pictures": [None]},
        {"schema": guard.PICTURES_SCHEMA, "pictures": [{"sha256": 3}]},
    ],
)
def test_approvals_are_strict(tmp_path, payload):
    write(tmp_path / "pictures/pictures.json", payload)
    with pytest.raises(ValidationError) as error:
        guard.approved_pictures(tmp_path)
    assert error.value.parameter == "pictures.pictures"


def test_broll_cross_project_approval_and_request_location(asked, fake_wavespeed, capsys):
    result = make(asked)
    item = load_catalog()["items"][0]
    other = asked / "another-video"
    assert guard.filter_approved([item], STYLE, asked) == ([item], [])
    filtered, excluded = guard.filter_approved([item], STYLE, other)
    assert filtered == []
    assert excluded == [
        {"item": item["id"], "reasons": ["generated picture not approved for this video"]}
    ]
    data = {"schema": "yap-studio/broll-requests@1", "broll": result["requests"]}
    with pytest.raises(ValidationError) as error:
        guard.validate_requests(data, [item], STYLE, other)
    assert error.value.parameter == "broll[0].item"
    guard.validate_requests(data, [item], STYLE, asked)
    cut = write(other / "cut/cut-words.json", words().to_dict())
    style = write(other / "style.json", {"schema_version": 1, **STYLE})
    assert (
        cli.main(["broll", "suggest", "--cut-words", str(cut), "--style", str(style), "--json"])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert not report["phrases"] and report["excluded"] == excluded
    source = write(other / "requests.json", data)
    assert (
        cli.main(
            [
                "broll",
                "plan",
                str(source),
                "--cut-words",
                str(cut),
                "--style",
                str(style),
                "--frames",
                "480",
                "--out",
                str(other / "plan"),
                "--json",
            ]
        )
        == 2
    )
    assert "broll[0].item" in capsys.readouterr().err


def test_cli_unavailable_and_output_errors(tmp_path, monkeypatch, capsys):
    from test_pictures_sheet import requests

    from yap_studio.pictures import command

    record_answer(tmp_path)
    cut = write(tmp_path / "cut/cut-words.json", words().to_dict())
    style = write(tmp_path / "style.json", {"schema_version": 1, **STYLE})
    source = write(tmp_path / "requests.json", requests())
    args = ["pictures", "ask", str(source), "--cut-words", str(cut), "--style", str(style)]
    assert cli.main(args) == 0
    assert wavespeed.SIGNUP_LINE in capsys.readouterr().out
    assert cli.main(["pictures", "make", str(tmp_path), "--pick", "1", "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "pictures_need_yes"

    def refused(*a, **kw):
        raise OutputPathError("synthetic output path")

    monkeypatch.setattr(command, "ask_sheet", refused)
    assert cli.main(args) == 3
    assert "synthetic output path" in capsys.readouterr().err
