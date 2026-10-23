"""Stock approval covers every clip played by a timeline card."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from test_file_drop import clip_source
from test_timeline import approve_stock, fragment, graphic, insert, stock_style, write
from test_timeline import project as project

from yap_studio import cards, cli, receipt, video_type
from yap_studio.errors import ValidationError
from yap_studio.media.catalog import add_stock_item
from yap_studio.project.model import Anchor, Element
from yap_studio.stages.timeline import TimelineRefusal, add_timeline
from yap_studio.stock import guard
from yap_studio.stock.credits import credits

ASK = {"broll": {"stock_clips": "ask"}}


@pytest.fixture(autouse=True)
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "config"
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(root))
    return root


@pytest.fixture
def stock_clip(config: Path, tmp_path: Path) -> dict:
    cache = config / "assets/stock/pexels"
    cache.mkdir(parents=True)
    source = clip_source(cache, "101.mp4")
    item = add_stock_item(source, stock={
        "provider": "pexels", "clip_id": "101",
        "page_url": "https://www.pexels.com/video/101/", "author": "Alex",
        "author_url": "https://www.pexels.com/@alex/", "license": "Pexels License",
        "license_url": "https://www.pexels.com/license/",
        "file_url": "https://videos.pexels.com/video-files/101/clip.mp4",
        "downloaded_at": "2026-10-18T12:00:00Z",
    }, description_text="synthetic stock clip")
    target = tmp_path / "stock.mp4"
    shutil.copyfile(source, target)
    return {"path": target.name, "sha256": item["sha256"]}


def enable_cards(project: Path, *, stock: bool = True) -> None:
    stock_style(project, enabled=stock)
    source = project / "stock-style.json"
    style = json.loads(source.read_bytes())
    style["graphics"] = {"card_kinds_on": ["video_card", "file_drop"]}
    write(source, style)


def card(clip: object, template: str = "video_card") -> dict:
    return graphic(template=template, slots={"clip": clip}, start={"time": 1}, hold_s=1.5)


def element(slots: dict, template: str = "video_card") -> Element:
    return Element("agent-card", "graphic", Anchor(time=1), lane="agent",
                   fields={"template": template, "slots": slots})


def test_cli_refuses_unapproved_card_without_changing_lane(
    project: Path, stock_clip: dict, capsys: pytest.CaptureFixture[str],
) -> None:
    enable_cards(project)
    add_timeline(project, fragment(project, graphic()), lane="agent").require_valid()
    target = project / "timeline/agent.json"
    before = target.read_bytes()
    incoming = fragment(project, card(stock_clip))
    assert cli.main(["timeline", "add", str(incoming), str(project),
                     "--lane", "agent", "--json"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "timeline_refused"
    assert error["message"] == "agent-card uses a stock clip that is not approved for this video."
    assert error["suggested_action"] == "Ask with yap-studio stock search, then stock get."
    assert target.read_bytes() == before


@pytest.mark.parametrize("template", ["video_card", "file_drop"])
def test_approved_card_writes_lane(project: Path, stock_clip: dict, template: str) -> None:
    enable_cards(project)
    approve_stock(project, stock_clip["sha256"])
    clip = stock_clip if template == "video_card" else stock_clip["path"]
    result = add_timeline(project, fragment(project, card(clip, template)), lane="agent")
    result.require_valid()
    assert not any("stock" in finding.message.lower() for finding in result.findings)
    assert (project / "timeline/agent.json").is_file()


def test_approved_card_refuses_style_off(project: Path, stock_clip: dict) -> None:
    enable_cards(project, stock=False)
    approve_stock(project, stock_clip["sha256"])
    with pytest.raises(TimelineRefusal, match="Stock clips are off in this project's style"):
        add_timeline(project, fragment(project, card(stock_clip)), lane="agent")
    assert not (project / "timeline/agent.json").exists()


def test_project_sources_override_stock_guards(project: Path, stock_clip: dict) -> None:
    enable_cards(project)
    approve_stock(project, stock_clip["sha256"])
    manifest = project / "project.json"
    data = json.loads(manifest.read_bytes())
    data["broll_sources"] = ["folder"]
    write(manifest, data)
    with pytest.raises(TimelineRefusal, match="Stock clips are not in this video's B-roll sources"):
        guard.check_timeline(project, [element({"clip": stock_clip})], ASK)
    items = [{"id": "m1", "sha256": stock_clip["sha256"], "stock": {"provider": "pexels"}}]
    request = {"broll": [{"item": "m1"}]}
    with pytest.raises(ValidationError, match="Stock clips are not in this video's B-roll sources"):
        guard.validate_requests(request, items, ASK, project)
    assert guard.filter_approved(items, ASK, project)[0] == []
    data["broll_sources"] = ["stock"]
    write(manifest, data)
    guard.check_timeline(project, [element({"clip": stock_clip})], {})
    guard.validate_requests(request, items, {}, project)
    assert guard.filter_approved(items, {}, project)[0] == items


@pytest.mark.parametrize("lane", ["agent", "broll", "both"])
@pytest.mark.parametrize("existing_default", [False, True])
def test_set_sources_refuses_stock_already_in_timeline(
    project: Path, stock_clip: dict, capsys: pytest.CaptureFixture[str],
    lane: str, existing_default: bool,
) -> None:
    enable_cards(project)
    approve_stock(project, stock_clip["sha256"])
    manifest = project / "project.json"
    data = json.loads(manifest.read_bytes())
    data.update(video_type="talking_head", broll_sources=["stock"])
    write(manifest, data)
    ids = []
    if lane in {"agent", "both"}:
        add_timeline(project, fragment(project, card(stock_clip)), lane="agent").require_valid()
        ids.append("agent-card")
    if lane in {"broll", "both"}:
        add_timeline(project, fragment(project, insert(project, media=stock_clip,
                     start={"time": 5}, end={"time": 6})),
                     lane="broll").require_valid()
        ids.append("broll-one")
    if existing_default:
        video_type.save_default("talking_head", ["stock"])
    default = video_type.defaults_path()
    old_default = default.read_bytes() if default.exists() else None
    before = manifest.read_bytes()
    command = ["video-type", "set", str(project), "--broll", "folder", "--json"]
    assert cli.main(command) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["message"] == "This video's timeline uses stock clips: " + ", ".join(ids) + "."
    assert error["suggested_action"] == (
        "Remove them with yap-studio timeline remove DIR --lane LANE --id ID, "
        "then set the sources again.")
    assert manifest.read_bytes() == before
    assert (default.read_bytes() if default.exists() else None) == old_default
    assert cli.main([*command, "--broll", "stock"]) == 0
    assert json.loads(capsys.readouterr().out)["project"]["broll_sources"] == ["folder", "stock"]


@pytest.mark.parametrize("stock_present", [False, True])
@pytest.mark.parametrize("value", ["media", "self"])
def test_own_video_card_is_unaffected(
    project: Path, request: pytest.FixtureRequest, stock_present: bool, value: str,
) -> None:
    if stock_present:
        request.getfixturevalue("stock_clip")
    enable_cards(project, stock=False)
    clip = ({"path": "source.mov", "sha256": receipt.sha256_file(project / "source.mov")}
            if value == "media" else "self")
    guard.check_timeline(project, [element({"clip": clip})], {})
    result = add_timeline(project, fragment(project, card(clip)), lane="agent")
    assert not any("stock" in finding.message.lower() for finding in result.findings)
    if value == "media":
        result.require_valid()
    if result.valid:
        assert (project / "timeline/agent.json").is_file()


def test_file_drop_refuses_stock_path_and_accepts_own(project: Path, stock_clip: dict) -> None:
    enable_cards(project)
    incoming = fragment(project, card(stock_clip["path"], "file_drop"))
    with pytest.raises(TimelineRefusal, match="agent-card uses a stock clip.*not approved"):
        add_timeline(project, incoming, lane="agent")
    assert not (project / "timeline/agent.json").exists()
    result = add_timeline(project, fragment(project, card("source.mov", "file_drop")), lane="agent")
    result.require_valid()
    assert (project / "timeline/agent.json").is_file()


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("as_string", [False, True])
def test_video_list_checks_each_entry(
    tmp_path: Path, stock_clip: dict, monkeypatch: pytest.MonkeyPatch,
    default: bool, as_string: bool,
) -> None:
    stock = stock_clip["path"] if as_string else stock_clip
    values = ["self", {"path": "own.mov", "sha256": "0" * 64}, stock]
    monkeypatch.setattr(cards, "CARD_TEMPLATES", ("playing_grid",))
    monkeypatch.setattr(cards, "load_template", lambda _: {
        "slots": {"clips": {"type": "video_list", "default": values}},
    })
    candidate = element({} if default else {"clips": values}, "playing_grid")
    with pytest.raises(TimelineRefusal, match="agent-card uses a stock clip.*not approved"):
        guard.check_timeline(tmp_path, [candidate], ASK)
    approve_stock(tmp_path, stock_clip["sha256"])
    guard.check_timeline(tmp_path, [candidate], ASK)


def test_card_without_video_slots_does_not_load_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected():
        pytest.fail("a card without video slots must not load the catalog")

    monkeypatch.setattr(guard, "load_catalog", unexpected)
    guard.check_timeline(tmp_path, [element({"text": "hello"}, "text_tag")], ASK)


@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("default", [False, True])
def test_stock_folder_clip_requires_approval_without_catalog(
    tmp_path: Path, config: Path, monkeypatch: pytest.MonkeyPatch,
    as_string: bool, default: bool,
) -> None:
    cache = config / "assets/stock"
    cache.mkdir(parents=True)
    source = clip_source(cache)
    digest = receipt.sha256_file(source)
    clip = str(source) if as_string else {"path": str(source), "sha256": digest}
    monkeypatch.setattr(cards, "CARD_TEMPLATES", ("default_video",))
    monkeypatch.setattr(cards, "load_template", lambda _: {
        "slots": {"clip": {"type": "video", "default": clip}},
    })
    candidate = element({} if default else {"clip": clip}, "default_video")
    with pytest.raises(TimelineRefusal, match="not approved"):
        guard.check_timeline(tmp_path, [candidate], ASK)
    approve_stock(tmp_path, digest)
    guard.check_timeline(tmp_path, [candidate], ASK)


def test_catalog_and_approvals_load_once(
    tmp_path: Path, stock_clip: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    approve_stock(tmp_path, stock_clip["sha256"])
    loads = []
    for name in ("load_catalog", "approved_clips"):
        original = getattr(guard, name)

        def counted(*args, original=original, name=name):
            loads.append(name)
            return original(*args)

        monkeypatch.setattr(guard, name, counted)
    guard.check_timeline(tmp_path, [element({"clip": stock_clip}),
                                    element({"clip": stock_clip["path"]}, "file_drop")], ASK)
    assert loads == ["load_catalog", "approved_clips"]


@pytest.mark.parametrize("stock_present", [False, True])
def test_string_clip_hashes_only_when_needed(
    tmp_path: Path, request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch, stock_present: bool,
) -> None:
    if stock_present:
        request.getfixturevalue("stock_clip")
    else:
        clip_source(tmp_path, "own.mp4")

    def unexpected(_):
        pytest.fail("missing clips and own clips without catalog stock need no hash")

    monkeypatch.setattr(receipt, "sha256_file", unexpected)
    guard.check_timeline(tmp_path, [element({"clip": "own.mp4"}, "file_drop")], ASK)


@pytest.fixture
def credit_clip(tmp_path: Path, stock_clip: dict) -> dict:
    clip = {**stock_clip, "author": "Alex", "page_url": "https://www.pexels.com/video/101/"}
    write(tmp_path / "stock/clips.json", {"schema": guard.CLIPS_SCHEMA, "clips": [clip]})
    return clip


def agent_cards(root: Path, *elements: dict) -> Path:
    return write(root / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": {}, "elements": list(elements),
    })


@pytest.mark.parametrize("template", ["video_card", "file_drop"])
@pytest.mark.parametrize("both_lanes", [False, True])
@pytest.mark.parametrize("absolute", [False, True])
def test_credits_include_card_stock_once(
    tmp_path: Path, credit_clip: dict, template: str, both_lanes: bool, absolute: bool,
) -> None:
    clip = {"path": credit_clip["path"], "sha256": credit_clip["sha256"]}
    if absolute:
        clip["path"] = str(tmp_path / clip["path"])
    agent_cards(tmp_path, card(clip if template == "video_card" else clip["path"], template))
    if both_lanes:
        write(tmp_path / "timeline/broll.json", {"elements": [{"media": clip}]})
    report = credits(tmp_path)
    assert report["clips"] == [{**credit_clip, "placed": True}]
    line = f"Video by Alex on Pexels: {credit_clip['page_url']}"
    assert report["text"].splitlines() == [line, "Videos provided by Pexels: https://www.pexels.com"]
    assert credits(tmp_path, all_clips=True) == report


@pytest.mark.parametrize("template", ["video_card", "file_drop", "self", "missing"])
def test_credits_ignore_own_card_clips(
    tmp_path: Path, credit_clip: dict, synth_media, template: str,
) -> None:
    own = tmp_path / "own.mov"
    shutil.copyfile(synth_media.path("calib"), own)
    clip = {"path": own.name, "sha256": receipt.sha256_file(own)}
    assert clip["sha256"] != credit_clip["sha256"]
    if template == "self":
        candidate = card("self")
    elif template == "missing":
        candidate = card("missing.mp4", "file_drop")
    else:
        candidate = card(clip if template == "video_card" else clip["path"], template)
    agent_cards(tmp_path, candidate)
    report = credits(tmp_path)
    assert report["clips"] == []
    assert report["text"] == "No stock clips in this video."
    assert credits(tmp_path, all_clips=True)["clips"] == [{**credit_clip, "placed": False}]


@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("as_string", [False, True])
def test_credits_include_video_list_card_clips(
    tmp_path: Path, credit_clip: dict, monkeypatch: pytest.MonkeyPatch,
    default: bool, as_string: bool,
) -> None:
    clip = {"path": credit_clip["path"], "sha256": credit_clip["sha256"]}
    values = ["self", clip["path"] if as_string else clip]
    monkeypatch.setattr(cards, "CARD_TEMPLATES", ("playing_grid",))
    monkeypatch.setattr(cards, "load_template", lambda _: {
        "slots": {"clips": {"type": "video_list", "default": values}},
    })
    candidate = graphic(template="playing_grid", slots={} if default else {"clips": values})
    agent_cards(tmp_path, candidate)
    assert credits(tmp_path)["clips"] == [{**credit_clip, "placed": True}]


@pytest.mark.parametrize("unreadable", [False, True])
def test_credits_report_unreadable_agent_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unreadable: bool,
) -> None:
    source = agent_cards(tmp_path)
    if unreadable:
        original = Path.read_bytes

        def read_bytes(path):
            if path == source:
                raise PermissionError("synthetic unreadable lane")
            return original(path)

        monkeypatch.setattr(Path, "read_bytes", read_bytes)
    else:
        source.write_text("not JSON")
    with pytest.raises(ValidationError, match="cannot read agent placements") as error:
        credits(tmp_path)
    assert error.value.parameter == "timeline.agent"
