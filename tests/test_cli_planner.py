"""CLI plans can be reviewed and consumed without running a project stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_media_select import CATALOG_SCHEMA
from test_planner_anchors import synthetic_words
from test_planner_broll import item as item
from test_planner_broll import phrase_words
from test_planner_broll import requests as broll_requests
from test_planner_cards import fake_render, requests, synthetic_faces

from yap_studio.cli import main
from yap_studio.faces.model import Box
from yap_studio.media.catalog import catalog_path
from yap_studio.planner import broll, cards, command


@pytest.fixture
def arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    source, words = tmp_path / "requests.json", tmp_path / "cut-words.json"
    source.write_text(json.dumps(requests()))
    words.write_text(json.dumps(synthetic_words().to_dict()))
    return [
        "cards",
        "plan",
        str(source),
        "--cut-words",
        str(words),
        "--frames",
        "900",
        "--out",
        str(tmp_path / "out"),
    ]


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_clear_output_and_files(
    arguments: list[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    face_file = tmp_path / "faces.json"
    face_file.write_text(json.dumps(synthetic_faces().to_dict()))
    argv = arguments + ["--faces", str(face_file)] + (["--json"] if json_output else [])
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        data = json.loads(captured.out)
        assert data["status"] == "ok" and data["cards"][0]["start_frame"] == 372
        assert data["outputs"]
        assert data == json.loads((tmp_path / "out/graphics-plan.json").read_bytes())
    else:
        assert "g-stat stat top 0:12.40 to 0:14.90 (frames 372 to 447) at y 0:" in captured.out
        assert "clear of the head (gap 31 px)" in captured.out
        assert "Plan: ok" in captured.out
        assert "Layers missing for 1 card(s): run again with --render" in captured.out
        for name in ("agent-lane.json", "graphics-plan.json", "layout.json", "cards.json"):
            assert f"Wrote {tmp_path / 'out' / name}" in captured.out
    assert len(list((tmp_path / "out").iterdir())) == 4


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_needs_review_exits_one(
    arguments: list[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    face_file = tmp_path / "faces.json"
    face_file.write_text(json.dumps(synthetic_faces(Box(0, 0, 1080, 1920)).to_dict()))
    argv = arguments + ["--faces", str(face_file)] + (["--json"] if json_output else [])
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        assert json.loads(captured.out)["status"] == "needs_review"
    else:
        assert "blocked:" in captured.out
        assert "Plan: needs review (1 card could not clear the head)" in captured.out
    assert (tmp_path / "out/graphics-plan.json").is_file()


def test_cli_faces_are_optional(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(arguments + ["--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["cards"][0]["clearance"]["status"] == "unchecked"


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_bad_input_exits_two(
    arguments: list[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    (tmp_path / "requests.json").write_text(json.dumps(requests(start={"word": "t1:13"})))
    assert main(arguments + (["--json"] if json_output else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)["error"]["message"] if json_output else captured.err
    assert "cards[0].start.word" in error and "t1:13" in error
    assert not (tmp_path / "out").exists()


def test_cli_foreign_output_exits_three(
    arguments: list[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    target = out / "cards.json"
    target.write_text("foreign")
    assert main(arguments + ["--json"]) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and json.loads(captured.err)["error"]["code"] == "unsafe_path"
    assert target.read_text() == "foreign"


def test_cli_hold_warning_and_style_default(
    arguments: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from yap_studio.style import resolve

    (tmp_path / "requests.json").write_text(
        json.dumps(requests(start={"time": 0}, end={"time": 1}))
    )
    seen = []

    def resolving(spec: str | None):
        seen.append(spec)
        return resolve("preset:electric")

    monkeypatch.setattr(command, "resolve", resolving)
    assert main(arguments) == 0
    assert seen == [None] and "warning: g-stat: hold 1 s is shorter" in capsys.readouterr().out


def test_cli_render_flag_calls_layer_renderer_without_launching_it(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(command, "render_layers", lambda *args: calls.append(args))
    assert main(arguments + ["--render", "--style", "preset:warm_editorial", "--fps", "30"]) == 0
    assert len(calls) == 1 and calls[0][1].name == "Warm Editorial"


def test_cli_replan_changed_card_reports_missing_layers(
    arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cards, "render_card", fake_render)
    assert main(arguments + ["--render", "--style", "preset:clean_minimal"]) == 0
    assert "Layers missing" not in capsys.readouterr().out
    assert main(arguments + ["--style", "preset:clean_minimal"]) == 0
    assert "Layers missing" not in capsys.readouterr().out
    target = tmp_path / "out/layers/g-stat.mov"
    assert target.is_file()
    (tmp_path / "requests.json").write_text(json.dumps(requests(slots={"end": 42})))
    assert main(arguments + ["--style", "preset:clean_minimal"]) == 0
    assert "Layers missing for 1 card(s): run again with --render" in capsys.readouterr().out
    assert not target.exists()
    receipt = json.loads((tmp_path / "out/graphics-plan.json").read_bytes())
    assert receipt["cards"][0]["layer"]["status"] == "missing"


@pytest.fixture
def broll_arguments(item: dict, tmp_path: Path) -> list[str]:
    catalog = catalog_path()
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({
        "schema": CATALOG_SCHEMA, "next_id": 2, "items": [item], "index": {},
    }))
    source, words = tmp_path / "broll-requests.json", tmp_path / "cut-words.json"
    source.write_text(json.dumps(broll_requests()))
    words.write_text(json.dumps(phrase_words().to_dict()))
    return ["broll", "plan", str(source), "--cut-words", str(words),
            "--frames", "60", "--out", str(tmp_path / "out"), "--project", str(tmp_path)]


@pytest.mark.parametrize("json_output", [True, False])
def test_broll_cli_plan_files_and_suggest_is_read_only(
    broll_arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    suggest = ["broll", "suggest", "--cut-words", str(tmp_path / "cut-words.json"),
               "--style", "preset:clean_minimal", "--limit", "1"]
    before = {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()}
    flags = ["--json"] if json_output else []
    assert main(suggest + flags) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        assert json.loads(captured.out)["phrases"][0]["candidates"][0]["item"] == "m1"
    else:
        assert "m1 insert full_frame score" in captured.out
    assert {file: file.read_bytes() for file in tmp_path.rglob("*") if file.is_file()} == before
    assert main(broll_arguments + flags) == 0
    captured = capsys.readouterr()
    if json_output:
        data = json.loads(captured.out)
        assert data["schema"] == broll.PLAN_SCHEMA and data["status"] == "ok"
        assert data == json.loads((tmp_path / "out/broll-plan.json").read_bytes())
    else:
        assert "frames 0 to 30" in captured.out and "Plan: ok" in captured.out
    assert len(list((tmp_path / "out").iterdir())) == 2


@pytest.mark.parametrize("json_output", [True, False])
def test_broll_cli_needs_review_exit_one(
    broll_arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    from yap_studio.style import resolve

    style = resolve("preset:clean_minimal").data
    style["broll"] = {"avoid_tags": ["busy"]}
    style_file = tmp_path / "style.json"
    style_file.write_text(json.dumps(style))
    catalog = json.loads(catalog_path().read_bytes())
    catalog["items"][0]["description"]["tags"] = ["busy"]
    catalog_path().write_text(json.dumps(catalog))
    flags = ["--json"] if json_output else []
    assert main(broll_arguments + ["--style", str(style_file)] + flags) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        assert json.loads(captured.out)["status"] == "needs_review"
    else:
        assert "warning:" in captured.out and "Plan: needs review" in captured.out


@pytest.mark.parametrize("json_output", [True, False])
def test_broll_cli_bad_input_exit_two(
    broll_arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    (tmp_path / "broll-requests.json").write_text(json.dumps(broll_requests(item="m99")))
    assert main(broll_arguments + (["--json"] if json_output else [])) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unknown catalog item" in captured.err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("fields,field", [({"box": [0, 0, 50]}, "box"),
                                         ({"placement": "unknown"}, "placement")])
def test_broll_cli_scroll_malformed_geometry_is_structured(
    broll_arguments: list[str], tmp_path: Path, monkeypatch, capsys, fields: dict, field: str,
) -> None:
    from yap_studio.style import resolve

    style = resolve("preset:clean_minimal")
    style.data["broll"]["overlay_motion"] = "scroll"
    monkeypatch.setattr(command, "resolve", lambda *args: style)
    (tmp_path / "broll-requests.json").write_text(json.dumps(
        broll_requests(kind="overlay", **fields)))
    assert main(broll_arguments + ["--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["type"] == "validation_error"
    assert f"broll[0].{field}" in error["message"]
    assert not (tmp_path / "out").exists()


def test_broll_cli_foreign_output_exit_three(
    broll_arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "broll-lane.json").write_text("foreign")
    assert main(broll_arguments + ["--json"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "unsafe_path"
    assert (out / "broll-lane.json").read_text() == "foreign"


@pytest.mark.parametrize("infer_project", [True, False])
@pytest.mark.parametrize("external", [True, False])
def test_broll_cli_paths_use_project_from_another_directory(
    broll_arguments: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], infer_project: bool, external: bool,
) -> None:
    project = tmp_path / "project"
    (project / "cut").mkdir(parents=True)
    words = project / "cut/cut-words.json"
    words.write_bytes((tmp_path / "cut-words.json").read_bytes())
    source = tmp_path / "dog.mov"
    catalog = json.loads(catalog_path().read_bytes())
    if not external:
        source = source.rename(project / "dog.mov")
        catalog["items"][0]["locations"][0]["path"] = str(source)
        catalog_path().write_text(json.dumps(catalog))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    arguments = broll_arguments[:-2] if infer_project else broll_arguments[:-1] + [str(project)]
    if infer_project:
        arguments[arguments.index("--cut-words") + 1] = str(words)
    assert main(arguments + ["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["broll"][0]["element"]["media"]["path"] == (
        str(source) if external else "dog.mov"
    )


def test_broll_cli_requires_project_for_noncanonical_words_path(
    broll_arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(broll_arguments[:-2] + ["--json"]) == 2
    assert "--project DIR" in json.loads(capsys.readouterr().err)["error"]["message"]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("arguments", [
    ["--limit", "0"], ["--limit", "51"], ["--cut-words", "missing.json"],
])
def test_broll_cli_invalid_suggest_exit_two(
    broll_arguments: list[str], tmp_path: Path, arguments: list[str], capsys,
) -> None:
    assert main(["broll", "suggest", "--cut-words", str(tmp_path / "cut-words.json"),
                 "--json", *arguments]) == 2
    assert "error" in json.loads(capsys.readouterr().err)
