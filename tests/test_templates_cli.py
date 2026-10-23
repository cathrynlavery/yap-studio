"""Template discovery, preview validation, and alpha still extraction."""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from pathlib import Path

import pytest

from yap_studio import cards, cli, ffmpeg, paths, receipt, templates_command
from yap_studio.errors import ProcessingError, YapError
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.render.hyperframes import LayerResult
from yap_studio.style import ResolvedStyle, resolve, write_resolved


@pytest.fixture(autouse=True)
def isolated_style_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


def _sample(template: str) -> dict:
    source = cards.template_dir(template) / "preview.json"
    return json.loads(source.read_text(encoding="utf-8"))["slots"]


def _write_slots(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def render(plan: cards.CardPlan, work_dir: Path, out: Path) -> LayerResult:
        assert work_dir.is_dir()
        calls.append({
            "plan": plan,
            "work": work_dir,
            "assets": {name: ffmpeg.probe(asset, count_frames=False)
                       for name, asset in plan.assets},
        })
        (work_dir / "scratch.txt").write_text("temporary work", encoding="utf-8")
        out.write_bytes(f"fake movie {len(calls)}".encode())
        return LayerResult(out, plan.frames, plan.width, plan.height, plan.frames / plan.fps)

    def extract(movie: Path, still: Path, frame: int) -> None:
        assert movie.is_file()
        assert frame == calls[-1]["plan"].frames * 3 // 5
        still.write_bytes(f"fake still {len(calls)}".encode())

    monkeypatch.setattr(cards, "render_card", render)
    monkeypatch.setattr(templates_command, "_extract_still", extract)
    return calls


@pytest.mark.parametrize("json_output", [False, True])
def test_list_in_library_order(
    capsys: pytest.CaptureFixture[str], json_output: bool,
) -> None:
    flags = ["--json"] if json_output else []
    kinds = (*cards.CARD_TEMPLATES, *registry_cards.kinds())
    assert cli.main(["templates", "list", *flags]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        templates = json.loads(captured.out)["templates"]
        assert [entry["id"] for entry in templates] == list(kinds)
        for entry in templates:
            metadata = cards.load_template(entry["id"])
            assert entry == {
                "id": metadata["id"],
                "version": metadata["version"],
                **({"label": metadata["label"], "use": metadata["use"]}
                   if registry_cards.is_kind(entry["id"]) else {}),
                "derived_from": [item["item"] for item in metadata["derived_from"]],
                "slots": list(metadata["slots"]),
                "required": [name for name, rule in metadata["slots"].items() if rule["required"]],
                "variants": {name: rule["options"] for name, rule in metadata["slots"].items()
                             if rule["type"] == "enum"},
            }
    else:
        lines = captured.out.splitlines()
        assert [line.split()[0] for line in lines] == list(kinds)
        for template, line in zip(kinds, lines, strict=True):
            metadata = cards.load_template(template)
            assert line.split(" | ")[1] == (
                "derived from: " + ", ".join(item["item"] for item in metadata["derived_from"])
            )
            assert line.split(" | ")[2] == "slots: " + ", ".join(
                name + ("*" if rule["required"] else "")
                for name, rule in metadata["slots"].items()
            )


def test_show_terminal_json_matches_contracts(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["templates", "show", "terminal", "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    data = json.loads(captured.out)
    metadata = cards.load_template("terminal")
    provenance = json.loads(
        (cards.template_dir("terminal").parent / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    assert data["id"] == "terminal" and data["version"] == metadata["version"]
    assert data["slots"] == metadata["slots"]
    assert data["slots"]["cadence"]["options"] == ["human", "uniform"]
    assert data["preview_slots"] == _sample("terminal")
    assert data["box"] == metadata["box"]
    assert data["derived_from"] == [entry["item"] for entry in metadata["derived_from"]]
    assert data["provenance"]["upstream"] == {
        key: provenance["upstream"][key] for key in ("repository", "commit", "license")
    }
    for shown, derived in zip(data["provenance"]["files"], metadata["derived_from"], strict=True):
        expected = next(entry for entry in provenance["files"] if entry["item"] == derived["item"])
        assert shown == {key: expected[key] for key in ("item", "upstream_path", "changes")}
        assert shown["upstream_path"] == derived["path"]


@pytest.mark.parametrize("template", ["terminal", "image_card"])
def test_show_text_rules_samples_box_and_provenance(
    template: str, capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["templates", "show", template]) == 0
    text = capsys.readouterr().out
    metadata = cards.load_template(template)
    assert f"{template} (version {metadata['version']})" in text
    for name, rule in metadata["slots"].items():
        assert f"{name}: type={rule['type']}, required={str(rule['required']).lower()}" in text
        assert "default=" + (json.dumps(rule["default"]) if "default" in rule else "(none)") in text
        for key, value in rule.items():
            if key in {"type", "required", "default"}:
                continue
            label = "suffixes" if key == "options" and rule["type"] == "image" else key
            assert f"{label}={json.dumps(value)}" in text
    assert "Preview slots: " + json.dumps(_sample(template)) in text
    assert "Box: " + json.dumps(metadata["box"]) in text
    assert "Upstream repository: " in text and "Pinned commit: " in text
    assert "License: Apache-2.0" in text
    for entry in metadata["derived_from"]:
        assert f"{entry['item']}: {entry['path']}" in text
    assert "Changes: " in text


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("command", ["show", "preview"])
def test_unknown_template_uses_standard_errors(
    command: str, json_output: bool, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    args = ["templates", command, "unknown"]
    out = tmp_path / "out"
    if command == "preview":
        args.extend(["--out", str(out)])
    if json_output:
        args.append("--json")
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        error = json.loads(captured.err)["error"]
        assert error["type"] == "validation_error"
        assert error["code"] == "invalid_parameter"
        assert error["suggested_action"] == "Fix template and retry."
        message = error["message"]
    else:
        assert captured.err.startswith("error: ")
        assert "fix: Fix template and retry." in captured.err
        message = captured.err
    assert all(template in message for template in cards.CARD_TEMPLATES)
    assert fake_render == [] and not out.exists()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_preview_merges_slots_and_reports_plan(
    wrapped: bool, json_output: bool, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    overrides = {"end": 42, "prefix": "+"}
    source = _write_slots(tmp_path / "slots.json", {"slots": overrides} if wrapped else overrides)
    specs: list[str | None] = []

    def resolving(spec: str | None) -> ResolvedStyle:
        specs.append(spec)
        return resolve(spec)

    monkeypatch.setattr(templates_command, "resolve", resolving)
    out = tmp_path / "preview"
    flags = ["--json"] if json_output else []
    assert cli.main([
        "templates", "preview", "stat", "--slots", str(source),
        "--style", "preset:warm_editorial", "--frames", "72", "--fps", "24",
        "--out", str(out), *flags,
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert specs == ["preset:warm_editorial"]
    plan = fake_render[0]["plan"]
    assert plan.frames == 72 and plan.fps == 24
    assert plan.variables["end"] == 42 and plan.variables["prefix"] == "+"
    for name, value in _sample("stat").items():
        if name not in overrides:
            assert plan.variables[name] == value
    expected = {
        "id": "stat", "mov": str(out / "stat.mov"), "png": str(out / "stat.png"),
        "box": {key: getattr(plan, key) for key in ("x", "y", "width", "height")},
        "frames": 72, "fps": 24, "style": "Warm Editorial",
    }
    if json_output:
        assert json.loads(captured.out) == expected
    else:
        assert captured.out.splitlines() == [
            expected["mov"], expected["png"],
            "Box: " + ", ".join(f"{key}={value}" for key, value in expected["box"].items()),
            "Frames: 72; fps: 24", "Style: Warm Editorial",
        ]
    assert (out / "stat.mov").is_file() and (out / "stat.png").is_file()
    assert not list(out.glob(".preview-work-*"))
    assert json.loads(source.read_text()) == ({"slots": overrides} if wrapped else overrides)


def test_preview_default_style_and_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    specs: list[str | None] = []

    def resolving(spec: str | None) -> ResolvedStyle:
        specs.append(spec)
        return resolve(spec)

    monkeypatch.setattr(templates_command, "resolve", resolving)
    assert cli.main(["templates", "preview", "stat", "--out", str(tmp_path / "out"), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert specs == [None]
    assert data["style"] == resolve(None).name
    assert data["frames"] == round(resolve(None).data["graphics"]["hold_s"] * 30)
    assert fake_render[0]["plan"].frames == data["frames"] and data["fps"] == 30


@pytest.mark.parametrize(
    "template,supplied", [("image_card", ()), ("image_compare", ()),
                          ("image_card", ("image",)), ("image_compare", ("before",))],
)
def test_preview_synthesizes_only_missing_image_slots(
    template: str, supplied: tuple[str, ...], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    image = tmp_path / "supplied.png"
    if supplied:
        ffmpeg.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
            "-i", "testsrc2=size=1280x800:rate=1", "-frames:v", "1", str(image),
        ], timeout_s=30)
    source = _write_slots(tmp_path / "slots.json", {name: str(image) for name in supplied})
    generated: list[Path] = []
    placeholder = templates_command._placeholder

    def generating(file: Path) -> None:
        generated.append(file)
        placeholder(file)

    monkeypatch.setattr(templates_command, "_placeholder", generating)
    assert cli.main([
        "templates", "preview", template, "--slots", str(source),
        "--style", "preset:warm_editorial", "--out", str(out), "--json",
    ]) == 0
    data = json.loads(capsys.readouterr().out)
    rules = cards.load_template(template)["slots"]
    image_slots = [name for name, rule in rules.items() if rule["type"] == "image"]
    work = fake_render[0]["work"]
    assert work.parent == out and work.name.startswith(".preview-work-")
    assert generated == [
        work / f"{name}.png" for name in image_slots if name not in supplied
    ]
    plan = fake_render[0]["plan"]
    assert plan.assets == tuple(
        (name + ".png",
         image.resolve() if name in supplied else (work / f"{name}.png").resolve())
        for name in image_slots
    )
    assert all(
        (info.width, info.height) == (1280, 800) for info in fake_render[0]["assets"].values()
    )
    assert data["id"] == template and data["style"] == "Warm Editorial"
    assert not list(out.glob(".preview-work-*"))
    if supplied:
        assert image.is_file()


def test_comment_drop_preview_keeps_initials_without_placeholder_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_render: list[dict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_placeholder(file: Path) -> None:
        pytest.fail(f"optional avatar should not generate {file.name}")

    monkeypatch.setattr(templates_command, "_placeholder", unexpected_placeholder)
    assert cli.main([
        "templates", "preview", "comment_drop", "--out", str(tmp_path / "preview"), "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "comment_drop"
    plan = fake_render[0]["plan"]
    assert plan.assets == () and plan.variables["initials"] == "MC"


def test_choices_preview_uses_options_and_style_colours(
    tmp_path: Path, fake_render: list[dict], capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "style.json"
    source.write_text(json.dumps({"schema_version": 1, "name": "Gold", "graphics": {
        "accent_color": "#E9C46A",
    }}))
    assert cli.main([
        "templates", "preview", "choices", "--style", str(source),
        "--out", str(tmp_path / "preview"), "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "choices"
    plan = fake_render[0]["plan"]
    assert json.loads(plan.variables["options"]) == ["Ten minutes", "One hour", "All day"]
    assert plan.assets == ()
    assert plan.variables["glow_color"] == "#E9C46A"
    assert plan.variables["text_color"] == "#F7EACB"


def test_icon_strike_preview_needs_no_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_render: list[dict],
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_placeholder(file: Path) -> None:
        pytest.fail(f"glyph preview should not generate {file.name}")

    monkeypatch.setattr(templates_command, "_placeholder", unexpected_placeholder)
    assert cli.main([
        "templates", "preview", "icon_strike", "--out", str(tmp_path / "preview"), "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "icon_strike"
    plan = fake_render[0]["plan"]
    assert plan.assets == () and plan.variables["glyph"] == "bell"
    assert plan.variables["label"] == "Less noise"


@pytest.mark.parametrize(
    "kind", ["missing", "malformed", "array", "wrapped_array", "unicode", "nested"],
)
def test_bad_slots_files_exit_two_before_rendering(
    kind: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    source = tmp_path / "slots.json"
    if kind == "malformed":
        source.write_text("{", encoding="utf-8")
    elif kind == "unicode":
        source.write_bytes(b"\xff")
    elif kind == "nested":
        source.write_text("[" * 100000, encoding="utf-8")
    elif kind != "missing":
        _write_slots(source, {"slots": []} if kind == "wrapped_array" else [])
    out = tmp_path / "out"
    assert cli.main([
        "templates", "preview", "stat", "--slots", str(source), "--out", str(out), "--json",
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["type"] == "validation_error" and error["code"] == "invalid_parameter"
    assert "slots" in error["message"] and str(source) in error["message"]
    if kind == "nested":
        assert "JSON" in error["message"]
    assert error["suggested_action"]
    assert fake_render == [] and not out.exists()


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("template,slot,value", [("stat", "end", "invalid"),
                                                ("image_card", "title", 3)])
def test_invalid_slot_names_path_before_rendering(
    template: str, slot: str, value: object, json_output: bool, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    source = _write_slots(tmp_path / "slots.json", {slot: value})
    flags = ["--json"] if json_output else []
    out = tmp_path / "out"
    assert cli.main([
        "templates", "preview", template, "--slots", str(source),
        "--out", str(out), *flags,
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        error = json.loads(captured.err)["error"]
        assert f"slots.{slot}" in error["message"]
        assert error["suggested_action"] == f"Fix slots.{slot} and retry."
    else:
        assert f"error: Bad value for slots.{slot}:" in captured.err
        assert f"fix: Fix slots.{slot} and retry." in captured.err
    assert fake_render == []
    assert not list(out.glob(".preview-work-*"))
    _write_slots(source, {slot: 42 if template == "stat" else "Valid"})
    assert cli.main([
        "templates", "preview", template, "--slots", str(source), "--out", str(out), *flags,
    ]) == 0
    assert len(fake_render) == 1
    assert not list(out.glob(".preview-work-*"))


@pytest.mark.parametrize("flag,value", [("--frames", "0"), ("--fps", "0"),
                                        ("--style", "preset:unknown")])
def test_bad_style_and_timing_exit_two(
    flag: str, value: str, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    assert cli.main([
        "templates", "preview", "stat", flag, value, "--out", str(tmp_path / "out"), "--json",
    ]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert flag.removeprefix("--") in error["message"]
    assert fake_render == []


@pytest.mark.parametrize("existing", ["directory_file", "stat.mov", "stat.png"])
@pytest.mark.parametrize("json_output", [False, True])
def test_output_errors_exit_three_and_preserve_existing_files(
    existing: str, json_output: bool, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    if existing == "directory_file":
        protected = out
    else:
        out.mkdir()
        protected = out / existing
    protected.write_bytes(b"keep this file")
    flags = ["--json"] if json_output else []
    assert cli.main(["templates", "preview", "stat", "--out", str(out), *flags]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        assert json.loads(captured.err)["error"]["code"] in {"unsafe_path", "invalid_output_path"}
    else:
        assert captured.err.startswith("error: ")
    assert fake_render == [] and protected.read_bytes() == b"keep this file"
    assert not list(out.glob(".preview-work-*"))


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_preview_protects_resolved_style_at_receipt_path(
    alias: str, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    source = write_resolved(resolve("preset:warm_editorial"), out / "stat.preview.json")
    style_path = source
    if alias != "direct":
        style_path = tmp_path / "style.json"
        if alias == "symlink":
            style_path.symlink_to(source)
        else:
            style_path.hardlink_to(source)
    original = source.read_bytes()
    assert cli.main([
        "templates", "preview", "stat", "--style", str(style_path),
        "--out", str(out), "--json",
    ]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "invalid_output_path"
    assert "input file" in error["message"]
    assert source.read_bytes() == style_path.read_bytes() == original
    assert fake_render == []
    assert not (out / "stat.mov").exists() and not (out / "stat.png").exists()
    assert not list(out.glob(".preview-work-*"))


@pytest.mark.parametrize("name", ["stat.preview.json", "stat.mov", "stat.png"])
def test_preview_protects_slots_at_output_path(
    name: str, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    source = _write_slots(out / name, {
        "schema": "yap-studio/template-preview@1", "slots": {"end": 42},
    })
    receipt_path = out / "stat.preview.json"
    if source != receipt_path:
        receipt.write_receipt(receipt_path, {
            "schema": "yap-studio/template-preview@1",
            "outputs": [{"path": name, "sha256": receipt.sha256_file(source)}],
        })
    original = source.read_bytes()
    original_receipt = receipt_path.read_bytes()
    assert cli.main([
        "templates", "preview", "stat", "--slots", str(source),
        "--out", str(out), "--json",
    ]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "invalid_output_path"
    assert "input file" in error["message"]
    assert source.read_bytes() == original
    assert receipt_path.read_bytes() == original_receipt
    assert fake_render == []
    assert not list(out.glob(".preview-work-*"))


def test_existing_work_folder_does_not_block_preview(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    (out / "work").mkdir(parents=True)
    protected = out / "work" / "keep.txt"
    protected.write_text("keep this file", encoding="utf-8")
    assert cli.main(["templates", "preview", "stat", "--out", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "stat"
    assert len(fake_render) == 1
    assert protected.read_text(encoding="utf-8") == "keep this file"
    assert not list(out.glob(".preview-work-*"))


@pytest.mark.parametrize("stage", ["placeholder", "render_card", "still"])
def test_failed_preview_cleans_work_and_can_retry(
    stage: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    generated: list[Path] = []
    placeholder = templates_command._placeholder

    def generating(file: Path) -> None:
        placeholder(file)
        assert file.is_file()
        generated.append(file)
        if stage == "placeholder":
            raise ProcessingError("ffmpeg", 1, "placeholder failed")

    def failure(*args: object) -> None:
        raise ProcessingError("ffmpeg", 1, f"{stage} failed")

    rendering = cards.render_card
    extracting = templates_command._extract_still
    monkeypatch.setattr(templates_command, "_placeholder", generating)
    if stage == "render_card":
        monkeypatch.setattr(cards, "render_card", failure)
    elif stage == "still":
        monkeypatch.setattr(templates_command, "_extract_still", failure)
    out = tmp_path / "out"
    args = [
        "templates", "preview", "image_card", "--style", "preset:warm_editorial",
        "--out", str(out), "--json",
    ]
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["type"] == "processing_error"
    assert len(generated) == 1
    assert not list(out.glob(".preview-work-*"))
    assert not (out / "image_card.mov").exists() and not (out / "image_card.png").exists()
    monkeypatch.setattr(templates_command, "_placeholder", placeholder)
    monkeypatch.setattr(cards, "render_card", rendering)
    monkeypatch.setattr(templates_command, "_extract_still", extracting)
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "image_card"
    assert fake_render[-1]["work"] != generated[0].parent
    assert not list(out.glob(".preview-work-*"))


def test_successive_previews_overwrite_owned_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    args = ["templates", "preview", "stat", "--out", str(out), "--json"]
    for run in (1, 2):
        assert cli.main(args) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        data = json.loads(captured.out)
        assert Path(data["mov"]).read_bytes() == f"fake movie {run}".encode()
        assert Path(data["png"]).read_bytes() == f"fake still {run}".encode()
        assert not list(out.glob(".preview-work-*"))
        evidence = json.loads((out / "stat.preview.json").read_text(encoding="utf-8"))
        assert evidence["outputs"] == [
            {"path": name, "sha256": receipt.sha256_file(out / name)}
            for name in ("stat.mov", "stat.png")
        ]
    assert len(fake_render) == 2
    assert fake_render[0]["work"] != fake_render[1]["work"]


@pytest.mark.parametrize("kind", ["symlink", "dangling_symlink", "hardlink", "directory", "fifo"])
def test_unsafe_preview_lock_preserves_outputs_and_can_retry(
    kind: str, tmp_path: Path,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    if kind == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("Named pipes are unavailable on this platform")
    out = tmp_path / "out"
    args = ["templates", "preview", "stat", "--out", str(out), "--json"]
    assert cli.main(args) == 0
    capsys.readouterr()
    originals = {out / name: (out / name).read_bytes()
                 for name in ("stat.mov", "stat.png", "stat.preview.json")}
    outside = tmp_path / "outside.txt"
    if kind != "dangling_symlink":
        outside.write_bytes(b"keep this outside file")
    lock_path = out / ".stat.preview.lock"
    lock_path.unlink()
    if kind in {"symlink", "dangling_symlink"}:
        lock_path.symlink_to(outside)
    elif kind == "hardlink":
        lock_path.hardlink_to(outside)
    elif kind == "directory":
        lock_path.mkdir()
    else:
        os.mkfifo(lock_path)

    assert cli.main(args) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "unsafe_path"
    assert {target: target.read_bytes() for target in originals} == originals
    if kind == "dangling_symlink":
        assert not outside.exists()
    else:
        assert outside.read_bytes() == b"keep this outside file"
    assert len(fake_render) == 1
    assert not list(out.glob(".preview-work-*"))

    if kind == "directory":
        lock_path.rmdir()
    else:
        lock_path.unlink()
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == "" and json.loads(captured.out)["id"] == "stat"
    assert (out / "stat.mov").read_bytes() == b"fake movie 2"
    assert (out / "stat.png").read_bytes() == b"fake still 2"
    assert len(fake_render) == 2
    assert not list(out.glob(".preview-work-*"))


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "mkfifo"),
    reason="Named pipes require POSIX and os.mkfifo",
)
def test_fifo_preview_lock_returns_promptly_and_preserves_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    args = ["templates", "preview", "stat", "--out", str(out), "--json"]
    assert cli.main(args) == 0
    capsys.readouterr()
    originals = {out / name: (out / name).read_bytes()
                 for name in ("stat.mov", "stat.png", "stat.preview.json")}
    lock_path = out / ".stat.preview.lock"
    lock_path.unlink()
    os.mkfifo(lock_path)

    exit_codes: list[int] = []
    worker = threading.Thread(target=lambda: exit_codes.append(cli.main(args)), daemon=True)
    worker.start()
    worker.join(timeout=3)
    assert not worker.is_alive(), "Preview stalled while opening a FIFO lock"
    assert exit_codes == [3]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "unsafe_path"
    assert {target: target.read_bytes() for target in originals} == originals
    assert len(fake_render) == 1
    assert not list(out.glob(".preview-work-*"))


def test_locked_preview_preserves_outputs_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    args = ["templates", "preview", "stat", "--out", str(out), "--json"]
    assert cli.main(args) == 0
    capsys.readouterr()
    originals = {out / name: (out / name).read_bytes()
                 for name in ("stat.mov", "stat.png", "stat.preview.json")}
    lock_path = out / ".stat.preview.lock"
    assert lock_path.is_file() and lock_path not in originals
    lock_inode = lock_path.stat().st_ino
    monkeypatch.setattr(templates_command, "_PREVIEW_LOCK_TIMEOUT_S", 0.02)
    with templates_command._publish_lock(out, "stat", timeout_s=0):
        with templates_command._publish_lock(out, "terminal", timeout_s=0):
            assert (out / ".terminal.preview.lock").is_file()
        started = time.monotonic()
        assert cli.main(args) == 2
        assert time.monotonic() - started < 1
        captured = capsys.readouterr()
        assert captured.out == ""
        error = json.loads(captured.err)["error"]
        assert error["type"] == "processing_error"
        assert error["code"] == "templates_preview_locked"
        assert "Wait" in error["suggested_action"] and "--out" in error["suggested_action"]
        assert {target: target.read_bytes() for target in originals} == originals
        assert len(fake_render) == 1
        assert not list(out.glob(".preview-work-*"))
    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert captured.err == "" and json.loads(captured.out)["id"] == "stat"
    assert (out / "stat.mov").read_bytes() == b"fake movie 2"
    assert (out / "stat.png").read_bytes() == b"fake still 2"
    evidence = json.loads((out / "stat.preview.json").read_bytes())
    assert evidence["outputs"] == [
        {"path": name, "sha256": receipt.sha256_file(out / name)}
        for name in ("stat.mov", "stat.png")
    ]
    assert lock_path.stat().st_ino == lock_inode
    assert len(fake_render) == 2
    assert not list(out.glob(".preview-work-*"))


def test_preview_output_checks_hold_publication_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    targets = {out / name for name in ("stat.mov", "stat.png", "stat.preview.json")}
    checked: list[Path] = []
    validating = paths.validate_output

    def validate(target: Path, *args: object, **kwargs: object) -> Path:
        if target in targets:
            with pytest.raises(YapError) as raised:
                with templates_command._publish_lock(out, "stat", timeout_s=0):
                    pytest.fail("Preview output validation ran without the publication lock")
            assert raised.value.code == "templates_preview_locked"
            checked.append(target)
        return validating(target, *args, **kwargs)

    monkeypatch.setattr(paths, "validate_output", validate)
    assert cli.main(["templates", "preview", "stat", "--out", str(out), "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.err == "" and json.loads(captured.out)["id"] == "stat"
    assert len(fake_render) == 1
    assert all(checked.count(target) >= 2 for target in targets)


def test_interrupted_publication_cleans_work_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    args = ["templates", "preview", "stat", "--out", str(out), "--json"]
    assert cli.main(args) == 0
    capsys.readouterr()
    copying = templates_command.shutil.copyfile

    def failure(source: Path, target: Path) -> None:
        if source.name == "stat.png":
            raise OSError("still publication failed")
        copying(source, target)

    monkeypatch.setattr(templates_command.shutil, "copyfile", failure)
    assert cli.main(args) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_output_path"
    assert (out / "stat.mov").read_bytes() == b"fake movie 2"
    assert (out / "stat.png").read_bytes() == b"fake still 1"
    assert not list(out.glob(".preview-work-*"))
    monkeypatch.setattr(templates_command.shutil, "copyfile", copying)
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "stat"
    assert (out / "stat.mov").read_bytes() == b"fake movie 3"
    assert (out / "stat.png").read_bytes() == b"fake still 3"
    assert len(fake_render) == 3
    assert not list(out.glob(".preview-work-*"))


@pytest.mark.parametrize("name", ["stat.mov", "stat.png"])
def test_modified_preview_outputs_are_protected(
    name: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_render: list[dict],
) -> None:
    out = tmp_path / "out"
    args = ["templates", "preview", "stat", "--out", str(out), "--json"]
    assert cli.main(args) == 0
    capsys.readouterr()
    target = out / name
    target.write_bytes(b"edited output")
    assert cli.main(args) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "unsafe_path"
    assert target.read_bytes() == b"edited output"
    assert len(fake_render) == 1
    assert not list(out.glob(".preview-work-*"))


def _png_rgba(path: Path, width: int, height: int) -> bytes:
    header = path.read_bytes()[:26]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">IIBB", header[16:26]) == (width, height, 8, 6)
    info = ffmpeg.probe(path, count_frames=False)
    assert info.pix_fmt == "rgba" and (info.width, info.height) == (width, height)
    pixels = ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
        "-frames:v", "1", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1",
    ], timeout_s=30, binary=True).stdout
    assert len(pixels) == width * height * 4
    return pixels


def test_preview_real_still_keeps_alpha_and_selects_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    plans: list[cards.CardPlan] = []

    def render(plan: cards.CardPlan, work_dir: Path, out: Path) -> LayerResult:
        plans.append(plan)
        assert work_dir.is_dir()
        ffmpeg.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
            "-i", f"nullsrc=size={plan.width}x{plan.height}:rate={plan.fps},"
            "format=rgba,geq=r=32:g=64:b=96:a='if(eq(N,3),128,64)'",
            "-frames:v", str(plan.frames), "-c:v", "qtrle", "-pix_fmt", "argb", str(out),
        ], timeout_s=30)
        return LayerResult(out, plan.frames, plan.width, plan.height, plan.frames / plan.fps)

    monkeypatch.setattr(cards, "render_card", render)
    out = tmp_path / "out"
    for _ in range(2):
        assert cli.main([
            "templates", "preview", "stat", "--frames", "5", "--fps", "2",
            "--out", str(out), "--json",
        ]) == 0
        data = json.loads(capsys.readouterr().out)
        plan = plans[-1]
        assert data["frames"] == 5 and data["fps"] == 2
        assert (out / "stat.mov").is_file()
        pixels = _png_rgba(out / "stat.png", plan.width, plan.height)
        assert set(pixels[3::4]) == {128}
        assert not list(out.glob(".preview-work-*"))


@pytest.mark.render
def test_real_stat_preview_has_rgba_and_opaque_pixels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    assert cli.main([
        "templates", "preview", "stat", "--style", "preset:warm_editorial",
        "--out", str(out), "--json",
    ]) == 0
    data = json.loads(capsys.readouterr().out)
    plan = cards.card_layer_plan("stat", _sample("stat"), resolve("preset:warm_editorial"))
    assert Path(data["mov"]).is_file() and Path(data["png"]).is_file()
    pixels = _png_rgba(Path(data["png"]), plan.width, plan.height)
    assert max(pixels[3::4]) == 255
    assert not list(out.glob(".preview-work-*"))
