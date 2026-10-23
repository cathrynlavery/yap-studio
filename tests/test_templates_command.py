"""Discovery contracts for the additional card kinds."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest
from test_hf_registry import synthetic as synthetic
from test_hf_registry_cards import KIND
from test_hf_registry_cards import registry_card as registry_card
from test_stage_captions import layer_project as layer_project
from test_stage_captions import set_elements
from test_taste import _approved, _body
from test_templates_cli import fake_render as fake_render

from yap_studio import cards, cli, templates_command
from yap_studio.gallery import server as gallery_server
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.render import hyperframes
from yap_studio.style import resolve, write_resolved

NEW_KINDS = (
    "chart", "quote", "progress_stat", "specs_list", "callout_circle", "name_tag",
    "success_check",
    "edit_timeline",
    "storyboard",
    "jiggle_delete",
    "file_drop",
    "video_card",
    "prompt_bar",
    "comment_cta",
    "comment_drop",
    "icon_strike",
    "playing_grid", "reel_feed",
    "gravity_drop",
    "choices",
)


@pytest.fixture(autouse=True)
def isolated_config(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path_factory.mktemp("templates-config")))


@pytest.mark.parametrize("source", [None, "style", "project"])
def test_registry_list_uses_catalog_metadata(registry_card, layer_project, capsys, source):
    set_elements(layer_project)
    style_path = layer_project / "style.json"
    command = ["templates", "list"]
    if source is not None:
        command.extend([f"--{source}", str(style_path if source == "style" else layer_project)])
        assert cli.main([*command, "--json"]) == 0
        listed = json.loads(capsys.readouterr().out)["templates"]
        assert KIND not in [entry["id"] for entry in listed]
        style = json.loads(style_path.read_text())
        style.setdefault("graphics", {})["card_kinds_on"] = [KIND]
        style_path.write_text(json.dumps(style))
    assert cli.main([*command, "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)["templates"]
    entry = listed[-1]
    assert entry["id"] == KIND and entry["label"] == "Sample" and entry["use"] == "words"
    assert entry["slots"] == ["text", "percent"] and entry["required"] == ["text"]
    assert cli.main(command) == 0
    line = capsys.readouterr().out.splitlines()[-1]
    assert KIND in line and "Sample" in line and "words" in line


def test_registry_show_uses_catalog_provenance_without_install(registry_card, capsys):
    registry_catalog, registry_install, _, _, _, _ = registry_card
    shutil.rmtree(registry_install.item_dir("sample"))
    assert cli.main(["templates", "show", KIND, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    entry = registry_catalog.item("sample")
    assert shown["id"] == KIND and shown["version"] == 1
    assert shown["label"] == entry["kind"]["label"] and shown["use"] == entry["kind"]["use"]
    assert shown["derived_from"] == ["sample"] and shown["slots"] == entry["slots"]
    assert shown["preview_slots"] == entry["preview"]
    assert shown["box"] == cards.load_template(KIND)["box"]
    registry = registry_catalog.load_catalog()["registry"]
    assert shown["provenance"]["upstream"] == {
        key: registry[key] for key in ("repository", "commit", "license")}
    provenance, = shown["provenance"]["files"]
    assert provenance["item"] == "sample" and provenance["upstream_path"] == entry["upstream_path"]
    assert "cache at setup" in provenance["changes"] and "not vendored" in provenance["changes"]
    assert cli.main(["templates", "show", KIND]) == 0
    text = capsys.readouterr().out
    for value in ("Sample", "words", registry["repository"], registry["commit"],
                  registry["license"], entry["upstream_path"], "cache at setup", "not vendored"):
        assert value in text


@pytest.mark.parametrize("override", [None, "Custom text"])
def test_registry_preview_uses_card_renderer(
    registry_card, tmp_path, capsys, monkeypatch, override,
):
    observed = []

    def render(composition, out, **kwargs):
        assert kwargs["variables"] == {} and kwargs["browser_gpu"] == hyperframes.CARD_BROWSER_GPU
        assert hyperframes.preflight(composition, kwargs["frames"], kwargs["fps"]) > 0
        observed.append((composition / "index.html").read_text())
        out.write_bytes(b"synthetic alpha")
        return hyperframes.LayerResult(out, kwargs["frames"], 1040, 360, kwargs["frames"] / 30)

    monkeypatch.setattr(hyperframes, "render_layer", render)
    monkeypatch.setattr(templates_command, "_extract_still",
                        lambda movie, still, frame: still.write_bytes(b"synthetic still"))
    command = ["templates", "preview", KIND, "--style", "preset:clean_minimal",
               "--frames", "60", "--out", str(tmp_path / "preview"), "--json"]
    if override is not None:
        slots_file = tmp_path / "slots.json"
        slots_file.write_text(json.dumps({"text": override}))
        command.extend(["--slots", str(slots_file)])
    assert cli.main(command) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["id"] == KIND and result["frames"] == 60 and result["fps"] == 30
    assert result["box"] == {"x": 20, "y": 280, "width": 1040, "height": 360}
    assert Path(result["mov"]).is_file() and Path(result["png"]).is_file()
    host, = observed
    assert (override or "Ship it now") in host and 'data-variable-values=' in host
    assert not list((tmp_path / "preview").glob(".preview-work-*"))


def test_streak_preview_reports_band_box(tmp_path, capsys, fake_render):
    style = resolve("preset:clean_minimal")
    style.data["graphics"]["card_motion"] = "streak"
    source = write_resolved(style, tmp_path / "streak.json")
    assert cli.main(["templates", "preview", "quote", "--style", str(source),
                     "--frames", "60", "--out", str(tmp_path / "preview"), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    call, = fake_render
    plan = call["plan"]
    assert plan.motion is not None
    assert result["box"] == {"x": 0, "y": plan.y, "width": 1080, "height": plan.height}
    assert result["box"] == plan.layer_box()


def test_registry_preview_extends_short_hold(registry_card, tmp_path, capsys, fake_render):
    style = resolve("preset:clean_minimal")
    style.data["graphics"].update(hold_s=1, card_kinds_on=[KIND])
    source = write_resolved(style, tmp_path / "short-hold.json")
    assert cli.main(["templates", "preview", KIND, "--style", str(source),
                     "--out", str(tmp_path / "preview"), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["frames"] == 45 and result["fps"] == 30
    call, = fake_render
    assert call["plan"].frames == 45


def test_registry_preview_reports_missing_item_fix(registry_card, tmp_path, capsys):
    shutil.rmtree(registry_card[1].item_dir("sample"))
    assert cli.main(["templates", "preview", KIND, "--out", str(tmp_path / "preview"),
                     "--style", "preset:clean_minimal", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "registry_item_missing"
    assert error["suggested_action"] == "Run yap-studio renderer install."
    assert not list((tmp_path / "preview").glob(".preview-work-*"))


def test_list_filters_by_project(
    layer_project: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_elements(layer_project)
    base_kinds = tuple(kind for kind in cards.CARD_TEMPLATES if kind not in cards.OPT_IN_KINDS)
    source = layer_project / "styles/x.json"
    source.parent.mkdir()
    (layer_project / "style.json").rename(source)
    project_path = layer_project / "project.json"
    project = json.loads(project_path.read_text())
    project["style"] = "styles/x.json"
    project_path.write_text(json.dumps(project))
    monkeypatch.chdir(layer_project.parent)
    command = ["templates", "list", "--project", str(layer_project), "--json"]
    assert cli.main(command) == 0
    listed = json.loads(capsys.readouterr().out)["templates"]
    assert tuple(item["id"] for item in listed) == base_kinds
    monkeypatch.setattr(cards, "OPT_IN_KINDS", (*cards.OPT_IN_KINDS, "stat"))
    assert cli.main(command) == 0
    listed = json.loads(capsys.readouterr().out)["templates"]
    assert tuple(item["id"] for item in listed) == tuple(
        kind for kind in base_kinds if kind != "stat"
    )
    style = json.loads(source.read_text())
    style["graphics"] = {"card_kinds_on": ["stat"]}
    source.write_text(json.dumps(style))
    assert cli.main(command) == 0
    listed = json.loads(capsys.readouterr().out)["templates"]
    assert tuple(item["id"] for item in listed) == base_kinds


def test_list_project_applies_approved_taste(
    layer_project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    set_elements(layer_project)
    base_kinds = tuple(kind for kind in cards.CARD_TEMPLATES if kind not in cards.OPT_IN_KINDS)
    _approved(_body(["text_tag"], field="graphics.card_kinds_off"))
    assert cli.main(["templates", "list", "--project", str(layer_project), "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)["templates"]
    assert tuple(item["id"] for item in listed) == tuple(
        kind for kind in base_kinds if kind != "text_tag"
    )
    source = layer_project / "style.json"
    assert cli.main(["templates", "list", "--style", str(source), "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)["templates"]
    assert tuple(item["id"] for item in listed) == base_kinds


def test_list_project_and_style_are_mutually_exclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli._parser().parse_args([
            "templates", "list", "--project", str(tmp_path), "--style", "preset:clean_minimal",
        ])
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert "not allowed with argument" in error


def test_list_project_requires_project_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["templates", "list", "--project", str(tmp_path)]) == 2
    error = capsys.readouterr().err
    assert "error:" in error and "fix:" in error and "project.json" in error
    assert "Traceback" not in error


def test_list_filters_by_style(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_kinds = tuple(kind for kind in cards.CARD_TEMPLATES if kind not in cards.OPT_IN_KINDS)
    command = ["templates", "list", "--style", "preset:clean_minimal", "--json"]
    assert cli.main(command) == 0
    assert tuple(item["id"] for item in json.loads(capsys.readouterr().out)["templates"]) == (
        base_kinds
    )
    monkeypatch.setattr(cards, "OPT_IN_KINDS", (*cards.OPT_IN_KINDS, "stat"))
    assert cli.main(command) == 0
    assert tuple(item["id"] for item in json.loads(capsys.readouterr().out)["templates"]) == (
        tuple(kind for kind in base_kinds if kind != "stat")
    )
    assert cli.main(["templates", "list", "--json"]) == 0
    assert tuple(item["id"] for item in json.loads(capsys.readouterr().out)["templates"]) == (
        (*cards.CARD_TEMPLATES, *registry_cards.kinds())
    )


@pytest.mark.parametrize("kind", ("prompt_bar", "comment_cta", "comment_drop", "icon_strike",
                                 "choices"))
@pytest.mark.parametrize("source", ["style", "project"])
def test_list_opt_in_kind_requires_creator_choice(
    kind: str, source: str, layer_project: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    set_elements(layer_project)
    style_path = layer_project / "style.json"
    style = json.loads(style_path.read_text())
    style.setdefault("graphics", {})["card_kinds_off"] = []
    style["graphics"].pop("card_kinds_on", None)
    style_path.write_text(json.dumps(style))
    command = ["templates", "list", f"--{source}",
               str(style_path if source == "style" else layer_project), "--json"]
    assert cli.main(command) == 0
    listed = json.loads(capsys.readouterr().out)["templates"]
    assert not any(item["id"] in cards.OPT_IN_KINDS for item in listed)
    style["graphics"]["card_kinds_on"] = [kind]
    style_path.write_text(json.dumps(style))
    assert cli.main(command) == 0
    listed = json.loads(capsys.readouterr().out)["templates"]
    assert [item["id"] for item in listed if item["id"] in cards.OPT_IN_KINDS] == [kind]


@pytest.mark.parametrize("kind", NEW_KINDS)
def test_new_kind_appears_in_list_and_show(kind: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["templates", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)["templates"]
    entry = next(item for item in listed if item["id"] == kind)
    assert entry["slots"] == list(cards.load_template(kind)["slots"])
    assert cli.main(["templates", "show", kind, "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["preview_slots"] == json.loads(
        (cards.template_dir(kind) / "preview.json").read_text()
    )["slots"]
    assert bool(shown["provenance"]["files"]) == (
        kind not in {"edit_timeline", "storyboard", "jiggle_delete", "file_drop", "gravity_drop",
                     "choices"}
    )
    provenance = json.loads((cards.template_dir(kind).parent / "PROVENANCE.json").read_text())
    expected = [
        {key: entry[key] for key in ("item", "upstream_path", "changes")}
        for entry in provenance["files"]
        if entry["yap_path"] == f"src/yap_studio/templates/{kind}/template.html"
    ]
    assert shown["provenance"]["files"] == expected
    if kind == "gravity_drop":
        assert shown["derived_from"] == []
        assert shown["provenance"] == {"upstream": None, "files": []}
        assert cli.main(["templates", "show", kind]) == 0
        assert "Upstream repository" not in capsys.readouterr().out


def test_prompt_bar_show_explains_timing(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["templates", "show", "prompt_bar"]) == 0
    shown = capsys.readouterr().out
    assert "reserves 1.2 s before fade-out" in shown
    assert "Confirm time in seconds" in shown


def test_comment_cta_show_explains_send_timing(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["templates", "show", "comment_cta"]) == 0
    shown = capsys.readouterr().out
    assert "reserves 1.2 s before fade-out" in shown
    assert "Send time in seconds" in shown


def test_edit_timeline_discovery_and_editor_group(capsys: pytest.CaptureFixture[str]) -> None:
    assert gallery_server._CATEGORIES["Editor"] == (
        "edit_timeline", "storyboard", "jiggle_delete", "file_drop",
    )
    assert list(gallery_server._CATEGORIES).index("Editor") > (
        list(gallery_server._CATEGORIES).index("Media")
    )
    assert cli.main(["templates", "show", "edit_timeline"]) == 0
    shown = capsys.readouterr().out
    assert "view" in shown and "title" in shown
    assert "Upstream" not in shown and "Derived from" not in shown


@pytest.mark.parametrize("kind,minimum_s", [
    ("video_card", 1.5), ("playing_grid", 1.5), ("reel_feed", 2.5),
])
@pytest.mark.parametrize("fps", [24, 30])
@pytest.mark.parametrize("hold_s", [1, 4.25])
def test_video_preview_default_frames_respect_hold_and_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    kind: str, minimum_s: float, fps: int, hold_s: float, fake_render: list[dict],
) -> None:
    from yap_studio import templates_command
    from yap_studio.style import resolve

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["hold_s"] = hold_s
    monkeypatch.setattr(templates_command, "resolve", lambda _: style)
    out = tmp_path / "preview"
    assert cli.main(["templates", "preview", kind, "--fps", str(fps),
                     "--out", str(out), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    expected = max(round(hold_s * fps), math.ceil(minimum_s * fps))
    assert data["frames"] == fake_render[0]["plan"].frames == expected
    assert data["frames"] / fps >= minimum_s
    assert not list(out.glob(".preview-work-*"))


@pytest.mark.parametrize("kind", ["video_card", "playing_grid", "reel_feed"])
def test_video_preview_explicit_frames_stay_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], kind: str,
    fake_render: list[dict],
) -> None:
    from yap_studio import templates_command
    from yap_studio.style import resolve

    style = resolve("preset:clean_minimal")
    style.data["graphics"]["hold_s"] = 4.25
    monkeypatch.setattr(templates_command, "resolve", lambda _: style)
    assert cli.main(["templates", "preview", kind, "--frames", "90", "--out",
                     str(tmp_path / "preview"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == fake_render[0]["plan"].frames == 90


def test_file_drop_show_lists_playback(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["templates", "show", "file_drop", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["slots"]["playback"] == {
        "type": "enum", "required": False, "options": ["still", "play"], "default": "still",
    }
    assert cli.main(["templates", "show", "file_drop"]) == 0
    assert "playback" in capsys.readouterr().out


@pytest.mark.parametrize("playback", ["still", "play"])
def test_file_drop_preview_fills_missing_video(tmp_path: Path, monkeypatch, playback: str) -> None:
    from yap_studio import ffmpeg, templates_command

    seen = []

    def render(plan, work, out):
        seen.append(plan)
        assert "window_fill" not in plan.variables
        assert bool(plan.windows) == (playback == "play")
        assert json.loads(plan.variables["play_window"]) == {}
        info = ffmpeg.probe(plan.poster.source)
        assert info.duration_s == 2
        assert (info.width, info.height) == (640, 400)
        assert (cards.prepare_card(plan, work) / "poster.jpg").is_file()
        out.write_bytes(b"synthetic alpha layer")

    monkeypatch.setattr(cards, "render_card", render)
    monkeypatch.setattr(templates_command, "_extract_still",
                        lambda movie, still, frame: still.write_bytes(b"synthetic preview"))
    slots = tmp_path / "slots.json"
    slots.write_text(json.dumps({"playback": playback, "poster_at": 0}))
    assert cli.main(["templates", "preview", "file_drop", "--style", "preset:clean_minimal",
                     "--frames", "45", "--slots", str(slots),
                     "--out", str(tmp_path / "preview")]) == 0
    assert len(seen) == 1
    assert (tmp_path / "preview/file_drop.mov").is_file()
    assert (tmp_path / "preview/file_drop.png").is_file()
    assert not list((tmp_path / "preview").glob(".preview-work-*"))


@pytest.mark.parametrize("size,expected", [
    ("1080x1920", (360, 640)),
    ("1920x1080", (640, 360)),
])
def test_placeholder_clip_keeps_image_aspect(
    tmp_path: Path, size: str, expected: tuple[int, int],
) -> None:
    from yap_studio import ffmpeg, templates_command

    image = tmp_path / "synthetic.png"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                f"color=c=blue:s={size}", "-frames:v", "1", str(image)], timeout_s=30)
    clip = tmp_path / "synthetic.mp4"
    templates_command.placeholder_clip(clip, image)
    info = ffmpeg.probe(clip)
    assert (info.width, info.height) == expected
    assert info.duration_s == 2


def test_shared_upstream_provenance_uses_each_cards_adaptation(capsys) -> None:
    source = cards.template_dir("video_card").parent / "PROVENANCE.json"
    provenance = json.loads(source.read_text())
    for kind in ("image_card", "video_card"):
        assert cli.main(["templates", "show", kind, "--json"]) == 0
        shown = json.loads(capsys.readouterr().out)
        expected = next(entry["changes"] for entry in provenance["files"]
                        if entry["item"] == "browser-device-stage"
                        and entry["yap_path"] == f"src/yap_studio/templates/{kind}/template.html")
        assert shown["provenance"]["files"][0]["changes"] == expected
