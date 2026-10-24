"""Shared render style resolution, private content boundaries, and revision recovery."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from yap_studio import paths
from yap_studio.errors import InputError, OutputPathError
from yap_studio.faces.model import Box
from yap_studio.project.receipts import StageRun, read_receipt
from yap_studio.project.status import STAGES, ProjectContext, status
from yap_studio.receipt import sha256_file
from yap_studio.style import resolve
from yap_studio.taste import (
    Taste,
    TasteError,
    add_proposals,
    approve,
    load_taste,
    read_taste_record,
    render_style,
    revoke,
    snapshot_at,
    taste_path,
    validate_taste,
    write_taste_record,
)
from yap_studio.taste import render as render_module
from yap_studio.taste.render import TASTE_RECORD_SCHEMA


@pytest.fixture(autouse=True)
def isolated_taste(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


@pytest.fixture(autouse=True)
def no_face_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic renders never invoke the system detector."""
    from test_stage_captions import _disable_face_helper

    _disable_face_helper(monkeypatch)


def _body(
    value: Any = "#123456", *, field: str = "captions.text_color", condition: dict | None = None,
) -> dict:
    return {
        "scope": {"hook": "cards", "chapters": "cards", "graphics": "cards", "music": "music"}.get(
            field.split(".")[0], "captions",
        ),
        "condition": condition,
        "sentence": "Use this private preference for the text.",
        "machine": {"kind": "style_field", "field": field, "value": value},
        "sources": [{
            "type": "comment", "media_sha256": "a" * 64, "comment_id": "c1", "label": "fix",
            "start_frame": 0, "end_frame": 2, "quote": "Make the text easier to read.",
        }],
        "context": "Private notes about this preference.",
    }


def _approved(body: dict | None = None) -> str:
    identity = add_proposals(body or _body())[0].id
    approve(identity)
    return identity


def test_chapter_taste_fields_resolve_and_bind_records(tmp_path: Path) -> None:
    _approved(_body("title_card", field="chapters.style"))
    source = tmp_path / "style.json"
    source.write_text(json.dumps({"schema_version": 1, "chapters": {"hold_s": 2}}))
    application = render_style(source, explicit={"chapters.hold_s": 2})
    assert application.style.data["chapters"]["style"] == "title_card"
    assert "chapters.hold_s" in application.explicit_fields
    media = tmp_path / "synthetic.mp4"
    media.write_bytes(b"synthetic media identity")
    record = write_taste_record(media, application)
    assert read_taste_record(media) is not None
    assert record.is_file()


@pytest.mark.parametrize("placement", [None, "fixed", "avoid_face"])
def test_render_style_preserves_explicit_caption_placement(
    tmp_path: Path, placement: str | None,
) -> None:
    source = tmp_path / "style.json"
    data = {"schema_version": 1, "captions": {}}
    if placement is not None:
        data["captions"]["placement"] = placement
    source.write_text(json.dumps(data))
    # Unrelated taste edits must not opt a legacy standalone clip into measurement.
    _approved()
    style = render_style(source).style
    assert style.data["captions"]["placement"] == (placement or "avoid_face")
    assert style.caption_placement_explicit is (placement is not None)


@pytest.mark.parametrize("method", ["taste", "override"])
@pytest.mark.parametrize("placement", ["fixed", "avoid_face"])
def test_placement_override_is_an_explicit_caption_choice(method: str, placement: str) -> None:
    explicit = None
    if method == "taste":
        _approved(_body(placement, field="captions.placement"))
    else:
        explicit = {"captions.placement": placement}
    style = render_style("preset:clean_minimal", explicit=explicit).style
    assert style.data["captions"]["placement"] == placement
    assert style.caption_placement_explicit


_PROJECT_RULES = [
    ("captions", "captions.text_color", "#123456"),
    ("graphics", "hook.text_color", "#123456"),
    ("graphics", "graphics.accent_color", "#123456"),
    ("music", "music.beat_snap", True),
    ("assemble", "music.level", 0.08),
]


def _project_stage(name: str) -> Any:
    from yap_studio.stages.assemble import assemble
    from yap_studio.stages.captions import captions
    from yap_studio.stages.graphics import graphics
    from yap_studio.stages.music import music

    return {"captions": captions, "graphics": graphics, "music": music, "assemble": assemble}[name]


@pytest.fixture
def project_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    """Use the synthetic stage project and publish bytes through stubbed renderers."""
    from test_stage_captions import make_project, set_elements
    from test_stage_graphics import graphic, title
    from test_timeline import write

    from yap_studio import ffmpeg
    from yap_studio.render.hyperframes import LayerResult
    from yap_studio.stages import assemble, captions, graphics

    root = tmp_path / "project"
    root.mkdir()
    make_project(root)
    set_elements(root, title(hold_s=1), graphic(start={"time": 1.5}))
    previous = read_receipt(root / "cut")
    with StageRun(root, "cut") as run:
        for name, digest in previous["inputs"].items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]})
        for name in ("cut.mp4", "voice.wav"):
            staged = root / f"staged-{name}"
            staged.write_bytes(name.encode())
            run.replace_output(root / "cut" / name, staged)
        spans = write(root / "staged-spans.json", {
            "sources": {"t1": "../source.mov"}, "fps": "30/1",
            "ranges": [{"source": "t1", "start": "0", "end": "3"}],
        })
        run.replace_output(root / "cut/spans.json", spans)
        run.finish(extra={"plan": {"total_frames": 90}})
    captured: dict[str, Any] = {}

    def layer(plan: Any, work: Path, out: Path, **kwargs: Any) -> LayerResult:
        captured[plan.template] = dict(plan.variables)
        out.write_bytes(json.dumps(plan.variables, sort_keys=True).encode())
        return LayerResult(out, plan.frames, plan.width, plan.height, plan.frames / 30)

    def picture(source: Path, out: Path, **kwargs: Any) -> Any:
        out.write_bytes(b"picture")
        return SimpleNamespace(grade=None, grade_compare=(), behind_layers=(), to_dict=lambda: {})

    def sound(source: Path, out: Path, **kwargs: Any) -> Any:
        captured["music"] = dict(kwargs["music"])
        timeline = kwargs["timeline"]
        out.write_bytes(b"sound")
        return SimpleNamespace(samples=(timeline.total_frames + timeline.end_hold_frames) * 1600,
                               click_stem=None,
                               to_dict=lambda: {})

    def mux(args: list[str], **kwargs: Any) -> None:
        Path(args[-1]).write_bytes(b"final media")

    monkeypatch.setattr(captions, "render_layer_plan", layer)
    monkeypatch.setattr(graphics, "render_layer_plan", layer)
    monkeypatch.setattr(graphics, "_render_card", lambda snapshot, work, out: layer(
        snapshot.plan, work, out,
    ))
    monkeypatch.setattr(assemble, "render_picture", picture)
    monkeypatch.setattr(assemble, "render_audio", sound)
    monkeypatch.setattr(assemble, "delivery_asserts", lambda *args, **kwargs: (
        [{"id": "audio.samples", "pass": True}], SimpleNamespace(to_dict=lambda: {}),
    ))
    monkeypatch.setattr(assemble, "verify", lambda *args, **kwargs: {"status": "pass"})
    monkeypatch.setattr(assemble, "drawn_boxes", lambda path, *, x, y, width, height: [
        (0, 90, Box(x, y, width, height)),
    ])
    monkeypatch.setattr(ffmpeg, "run", mux)
    monkeypatch.setattr(ffmpeg, "probe", lambda *args, **kwargs: SimpleNamespace(
        video_frames=135, has_video=True, width=1080, height=1920, color_transfer=None,
        format_name="mov",
    ))
    monkeypatch.setattr(ffmpeg, "tool_versions", lambda: {})
    for name in ("captions", "graphics", "music"):
        _project_stage(name)(root)
    return root, captured


def _stage_state(root: Path, name: str) -> str:
    return next(item["state"] for item in status(root)["stages"] if item["name"] == name)


@pytest.mark.parametrize("stage,field,value", _PROJECT_RULES)
def test_project_stages_apply_pin_and_invalidate_used_taste(
    project_pipeline: tuple[Path, dict], stage: str, field: str, value: Any,
) -> None:
    root, captured = project_pipeline
    render = _project_stage(stage)
    render(root)
    folder = "final" if stage == "assemble" else stage
    baseline = read_receipt(root / folder)["inputs"]
    identity = _approved(_body(value, field=field))
    context = ProjectContext(root)
    section, key = field.split(".")
    assert context.render_style().data[section][key] == value
    assert _stage_state(root, stage) == "stale"
    assert not render(root)["noop"]
    record = read_receipt(root / folder)
    assert record["inputs"] != baseline
    assert record["taste"] == context.taste_application().pin()
    assert record["taste"]["applied"] == [{"id": identity, "field": field}]
    assert record["taste"]["explicit_fields"] == []
    if section == "captions":
        assert captured["captions"][key] == value
    elif section == "hook":
        assert captured["title"][key] == value
    elif section == "graphics":
        assert captured["text_tag"][key] == value
    elif stage == "assemble":
        assert captured["music"][key] == value
    else:
        from yap_studio.project.receipts import projection_sha256

        assert record["inputs"]["style:music:beat_snap"] == projection_sha256(value)
    assert _stage_state(root, stage) == "done"
    assert render(root)["noop"]

    # Rewording preserves the style value while changing the approved rule identity.
    reworded = {**_body(value, field=field), "sentence": "Keep this preferred style value."}
    identity = _approved(reworded)
    assert ProjectContext(root).render_style().sha256() == context.render_style().sha256()
    assert _stage_state(root, stage) == "stale"
    assert not render(root)["noop"]
    assert read_receipt(root / folder)["taste"]["applied"] == [{"id": identity, "field": field}]

    revoke(identity)
    assert _stage_state(root, stage) == "stale"
    assert not render(root)["noop"]
    assert read_receipt(root / folder)["inputs"] == baseline
    assert ProjectContext(root).render_style().sha256() == resolve(root / "style.json").sha256()


@pytest.mark.parametrize("stage", ["captions", "graphics", "music", "assemble"])
@pytest.mark.parametrize("empty", [False, True])
def test_project_no_taste_preserves_existing_inputs_and_receipts(
    project_pipeline: tuple[Path, dict], stage: str, empty: bool,
) -> None:
    root, _ = project_pipeline
    render = _project_stage(stage)
    render(root)
    folder = "final" if stage == "assemble" else stage
    record = read_receipt(root / folder)
    inputs = dict(record["inputs"])
    assert "taste" not in inputs
    # Model an existing receipt that predates project taste support.
    del record["taste"]
    record["outputs"] = [item for item in record["outputs"]
                         if not item["path"].endswith(".taste.json")]
    (root / folder / "receipt.json").write_text(json.dumps(record))
    if empty:
        target = taste_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(Taste().to_json()))
    context = ProjectContext(root)
    assert context.render_style().sha256() == resolve(root / "style.json").sha256()
    spec = next(spec for spec in STAGES if spec.name == stage)
    assert spec.inputs(context) == inputs
    assert _stage_state(root, stage) == "done"
    assert render(root)["noop"]
    assert read_receipt(root / folder) == record


@pytest.mark.parametrize("stage,field", [
    ("captions", "graphics.accent_color"), ("graphics", "captions.text_color"),
    ("music", "music.level"), ("music", "music.dry_open_s"),
    ("assemble", "hook.background_color"),
])
def test_unused_taste_does_not_stale_project_stage(
    project_pipeline: tuple[Path, dict], stage: str, field: str,
) -> None:
    root, _ = project_pipeline
    if stage == "assemble":
        # No title means this hook rule affects no upstream render or final input.
        from test_stage_captions import set_elements
        from test_stage_graphics import graphic

        set_elements(root, graphic(start={"time": 1.5}))
        _project_stage("graphics")(root)
    render = _project_stage(stage)
    render(root)
    _approved(_body(0.08 if field.startswith("music.") else "#123456", field=field))
    assert _stage_state(root, stage) == "done"
    assert render(root)["noop"]


@pytest.mark.parametrize("stage,field,value", _PROJECT_RULES)
def test_project_stage_retains_one_taste_snapshot_during_run(
    project_pipeline: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch,
    stage: str, field: str, value: Any,
) -> None:
    from yap_studio.taste import resolve as taste_resolve

    root, _ = project_pipeline
    _approved(_body(value, field=field))
    application = ProjectContext(root).taste_application()
    reads = []

    def load() -> Taste:
        taste = load_taste()
        reads.append(taste.revision)
        return taste

    monkeypatch.setattr(taste_resolve, "load_taste", load)
    record_input = StageRun.record_inputs
    changed = False

    def changing(run: StageRun, name: str, digest: str) -> None:
        nonlocal changed
        record_input(run, name, digest)
        if not changed:
            changed = True
            replacement = (False if value is True
                           else 0.09 if isinstance(value, float) else "#654321")
            _approved(_body(replacement, field=field))

    monkeypatch.setattr(StageRun, "record_inputs", changing)
    _project_stage(stage)(root)
    folder = "final" if stage == "assemble" else stage
    assert reads == [1]
    assert read_receipt(root / folder)["taste"] == application.pin()
    if stage == "assemble":
        pin = read_taste_record(root / "final/sample.mp4")
        assert pin == {"schema": TASTE_RECORD_SCHEMA,
                       "media_sha256": sha256_file(root / "final/sample.mp4"), **application.pin()}
    assert _stage_state(root, stage) == "stale"


@pytest.mark.parametrize("look", ["quiet_serif", "text_hook"])
@pytest.mark.parametrize("answered", [None, False])
def test_old_project_assembles_without_top_text_choice(
    project_pipeline: tuple[Path, dict], capsys: pytest.CaptureFixture[str], answered: bool | None,
    look: str,
) -> None:
    from yap_studio import cli
    from yap_studio.stages.graphics import graphics_snapshot

    root, _ = project_pipeline
    style = root / "style.json"
    data = json.loads(style.read_bytes())
    assert "answered" not in data["hook"]
    data["hook"]["style"] = look
    if answered is not None:
        data["hook"]["answered"] = answered
    style.write_text(json.dumps(data))
    _project_stage("graphics")(root)
    assert not (root / "plans/text-hook.json").exists()
    titles = graphics_snapshot(ProjectContext(root)).titles
    if look == "text_hook":
        assert not titles
    else:
        assert len(titles) == 1 and titles[0].plan.variables["look"] == look
    unchanged = {file: file.read_bytes() for file in (
        style, *root.glob("graphics/*.mov"), *root.glob("timeline/*.json"),
    )}

    assert cli.main(["assemble", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verified"
    assert (root / result["out"]).read_bytes() == b"final media"
    assert graphics_snapshot(ProjectContext(root)).titles == titles
    assert all(file.read_bytes() == content for file, content in unchanged.items())
    assert not (root / "plans/text-hook.json").exists()


@pytest.mark.parametrize("look", ["quiet_serif", "text_hook"])
@pytest.mark.parametrize("frames", [[], ["--frames", "0-20"]])
def test_assemble_still_requires_chosen_text_after_answer(
    project_pipeline: tuple[Path, dict], capsys: pytest.CaptureFixture[str], frames: list[str],
    look: str,
) -> None:
    from yap_studio import cli

    root, _ = project_pipeline
    style = root / "style.json"
    data = json.loads(style.read_bytes())
    data["hook"].update(style=look, answered=True)
    style.write_text(json.dumps(data))

    assert cli.main(["assemble", str(root), "--json", *frames]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "invalid_parameter"
    assert "Choose the text at the top" in error["message"]
    assert not list(root.glob("final/*.mp4"))
    assert not list(root.glob("pilot/*.mp4"))


def test_project_timeline_uses_taste_defaults_and_explicit_element_duration(
    project_pipeline: tuple[Path, dict],
) -> None:
    from test_stage_captions import set_elements
    from test_stage_graphics import graphic, title

    from yap_studio.stages.graphics import graphics_snapshot

    root, _ = project_pipeline
    _approved(_body(0.7, field="hook.duration_s"))
    _approved(_body(0.7, field="graphics.hold_s"))
    set_elements(root, title(), graphic(start={"time": 1.5}, hold_s=1))
    snapshot = graphics_snapshot(ProjectContext(root))
    assert snapshot.titles[0].plan.frames == 21
    assert snapshot.graphics[0].plan.frames == 30
    set_elements(root, title(hold_s=1), {key: value for key, value in graphic(
        start={"time": 1.5},
    ).items() if key != "hold_s"})
    snapshot = graphics_snapshot(ProjectContext(root))
    assert snapshot.titles[0].plan.frames == 30
    assert snapshot.graphics[0].plan.frames == 21


def test_project_music_settings_keep_precedence_over_style_taste(
    project_pipeline: tuple[Path, dict],
) -> None:
    root, _ = project_pipeline
    data = json.loads((root / "project.json").read_bytes())
    data["music"] = {"mode": "none", "file": "missing-track.wav"}
    (root / "project.json").write_text(json.dumps(data))
    _approved(_body(True, field="music.beat_snap"))
    result = _project_stage("music")(root)
    assert result["mode"] == "none" and result["bed"] is None
    assert read_receipt(root / "music")["taste"]["revision"] == 1


def test_project_file_music_applies_dry_open_taste(
    project_pipeline: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.project.receipts import projection_sha256
    from yap_studio.stages import music

    root, _ = project_pipeline
    data = json.loads((root / "project.json").read_bytes())
    data["music"] = {"mode": "file", "file": "track.wav"}
    (root / "project.json").write_text(json.dumps(data))
    (root / "track.wav").write_bytes(b"synthetic track")
    rendered = []

    def arrange(snapshot: Any, work: Path) -> tuple[Path, dict]:
        rendered.append(snapshot.dry_open_s)
        bed = work / "bed.wav"
        bed.write_bytes(str(snapshot.dry_open_s).encode())
        return bed, {"source": {"sha256": snapshot.source_sha256}}

    monkeypatch.setattr(music, "_arrange", arrange)
    music.music(root)
    _approved(_body(0.5, field="music.dry_open_s"))
    assert _stage_state(root, "music") == "stale"
    assert not music.music(root)["noop"]
    record = read_receipt(root / "music")
    assert rendered[-1] == 0.5
    assert record["inputs"]["style:music:dry_open"] == projection_sha256(0.5)
    assert record["taste"] == ProjectContext(root).taste_application().pin()


def test_project_assembly_owns_taste_record_and_checks_its_bytes(
    project_pipeline: tuple[Path, dict],
) -> None:
    root, _ = project_pipeline
    _approved(_body(0.08, field="music.level"))
    result = _project_stage("assemble")(root)
    media = root / result["out"]
    target = media.with_suffix(".taste.json")
    record = read_receipt(root / "final")
    assert read_taste_record(media) == {
        "schema": TASTE_RECORD_SCHEMA, "media_sha256": sha256_file(media), **record["taste"],
    }
    assert any(item["path"] == "final/sample.taste.json" and item["sha256"] == sha256_file(target)
               for item in record["outputs"])
    target.write_text("{}")
    assert _stage_state(root, "assemble") == "stale"


def test_failed_project_assembly_does_not_publish_taste_record(
    project_pipeline: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.stages import assemble

    root, _ = project_pipeline
    _approved(_body(0.08, field="music.level"))
    monkeypatch.setattr(assemble, "verify", lambda *args, **kwargs: {"status": "fail"})
    with pytest.raises(assemble.AssemblyError, match="verify.status"):
        assemble.assemble(root)
    record = read_receipt(root / "final")
    assert record["status"] == "failed"
    assert record["taste"] == ProjectContext(root).taste_application().pin()
    assert not (root / "final/sample.mp4").exists()
    assert not (root / "final/sample.taste.json").exists()


@pytest.mark.parametrize("matches", [False, True])
def test_project_assembly_verifies_approved_taste_before_publication(
    project_pipeline: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch, matches: bool,
) -> None:
    import importlib
    from fractions import Fraction

    from yap_studio import ffmpeg
    from yap_studio.checks.joins import JoinsResult
    from yap_studio.checks.structure import StructureResult
    from yap_studio.checks.sync import SyncResult
    from yap_studio.stages import assemble

    root, _ = project_pipeline
    verifier = importlib.import_module("yap_studio.verify")
    checker = importlib.import_module("yap_studio.taste.check")
    check = checker.check_taste
    measured_records = []

    def measure(media: Path, *, fields: tuple[str, ...]) -> dict:
        assert fields == ("captions.text_color",)
        measured_records.append(read_taste_record(media))
        return {"captions.text_color": {"value": "#FF0000" if matches else "#FFFFFF", "frames": []}}

    monkeypatch.setattr(checker, "check_taste", lambda media, record: check(
        media, record, measure=measure,
    ))
    monkeypatch.setattr(ffmpeg, "probe", lambda media, **kwargs: SimpleNamespace(
        path=str(media), has_video=True, has_audio=False, fps=Fraction(30), video_frames=135,
        width=1080, height=1920, orientation="portrait", duration_s=4.5, sample_rate=48000,
        color_transfer=None, variable_frame_rate=False, format_name="mov",
    ))
    monkeypatch.setattr(verifier, "check_sync", lambda *args, **kwargs: SyncResult([], None, []))
    monkeypatch.setattr(verifier, "check_joins", lambda *args, **kwargs: JoinsResult([], []))
    monkeypatch.setattr(verifier, "check_structure", lambda *args, **kwargs: StructureResult(
        integrated_lufs=None, lra_lu=None, true_peak_dbtp=None, black=[], freeze=[], frames=135,
        samples=None, sample_rate=None, video_duration_s=4.5, audio_duration_s=None,
        parts_lufs=[], findings=[],
    ))
    monkeypatch.setattr(assemble, "verify", verifier.verify)

    # The existing no-taste report stays unchanged and performs no taste measurement.
    assemble.assemble(root)
    assert read_receipt(root / "final")["verify"]["checks"]["taste.rules"] == {
        "ran": False, "status": "skipped", "reason": "no taste record next to this render",
    }
    assert measured_records == []
    identity = _approved(_body("#FF0000"))
    _project_stage("captions")(root)
    application = ProjectContext(root).taste_application()
    if matches:
        assemble.assemble(root)
    else:
        with pytest.raises(assemble.AssemblyError, match="verify.status"):
            assemble.assemble(root)
    final = root / "final/sample.mp4"
    published_path = final if matches else root / "final/sample.stale.mp4"
    if not matches:
        assert not final.exists()
        assert published_path.is_file()
        assert published_path.with_suffix(".taste.json").is_file()
    record = read_receipt(root / "final")
    assert record["status"] == ("verified" if matches else "failed")
    assert record["verify"]["checks"]["taste.rules"] == {
        "ran": True, "status": "pass" if matches else "fail", "reason": None,
    }
    verified = next(item for item in record["asserts"] if item["id"] == "verify.status")
    assert verified["pass"] == matches
    assert len(measured_records) == 1
    assert measured_records[0] == {
        "schema": TASTE_RECORD_SCHEMA, "media_sha256": sha256_file(published_path),
        **application.pin(),
    }
    published = read_taste_record(published_path)
    if matches:
        assert published == measured_records[0]
    else:
        assert published["revision"] == 0
        assert any(item["check"] == "taste.rules" and item["evidence"]["rule_id"] == identity
                   for item in record["verify"]["findings"])


@pytest.mark.parametrize("command", ["resolve", "add"])
@pytest.mark.parametrize("kind,field", [
    ("title", "hook.duration_s"), ("graphic", "graphics.hold_s"),
])
def test_project_public_timeline_uses_approved_taste_durations(
    project_pipeline: tuple[Path, dict], command: str, kind: str, field: str,
) -> None:
    from test_stage_captions import set_elements
    from test_stage_graphics import graphic, title
    from test_timeline import fragment, write

    from yap_studio.stages.graphics import graphics_snapshot
    from yap_studio.stages.timeline import TimelineRefusal, add_timeline, resolve_timeline

    root, _ = project_pipeline
    section, key = field.split(".")
    style = json.loads((root / "style.json").read_bytes())
    style.setdefault(section, {})[key] = 4.48
    write(root / "style.json", style)
    element = title(start={"time": 0}) if kind == "title" else graphic(start={"time": 0})
    element["lane"] = "agent"
    element.pop("hold_s", None)
    set_elements(root, element)
    assert not resolve_timeline(root).valid
    with pytest.raises(TimelineRefusal):
        graphics_snapshot(ProjectContext(root))
    if command == "add":
        set_elements(root)
        before = (root / "timeline/agent.json").read_bytes()
        assert not add_timeline(root, fragment(root, element), lane="agent").valid
        assert (root / "timeline/agent.json").read_bytes() == before
    _approved(_body(0.7, field=field))
    result = (resolve_timeline(root) if command == "resolve"
              else add_timeline(root, fragment(root, element), lane="agent"))
    result.require_valid()
    snapshot = graphics_snapshot(ProjectContext(root))
    layer = snapshot.titles[0] if kind == "title" else snapshot.graphics[0]
    assert result.elements[0].end_frame - result.elements[0].start_frame == layer.plan.frames == 21
    assert resolve_timeline(root).to_dict() == ProjectContext(root).resolved_timeline().to_dict()


def test_project_public_timeline_uses_approved_taste_beat_snap(
    project_pipeline: tuple[Path, dict],
) -> None:
    from test_stage_captions import set_elements
    from test_stage_graphics import graphic
    from test_timeline import fragment, publish_beats

    from yap_studio.stages.graphics import graphics_snapshot
    from yap_studio.stages.timeline import add_timeline, resolve_timeline

    root, _ = project_pipeline
    element = graphic(lane="agent")
    set_elements(root, element)
    publish_beats(root, (5,))
    assert resolve_timeline(root).elements[0].start_frame == 6
    assert graphics_snapshot(ProjectContext(root)).graphics[0].start_frame == 6
    _approved(_body(True, field="music.beat_snap"))
    publish_beats(root, (5,))
    snapshot = graphics_snapshot(ProjectContext(root))
    assert snapshot.graphics[0].start_frame == 5
    resolved = resolve_timeline(root).require_valid()
    added = add_timeline(root, fragment(root, element), lane="agent").require_valid()
    assert resolved.elements[0].start_frame == added.elements[0].start_frame == 5
    rendered = ProjectContext(root).resolved_timeline()
    assert resolved.to_dict() == added.to_dict() == rendered.to_dict()


@pytest.mark.parametrize("change", ["approve", "revoke", "revision_only"])
def test_timeline_add_refuses_changed_taste_before_write(
    project_pipeline: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    from contextlib import contextmanager

    from test_stage_captions import set_elements
    from test_stage_graphics import title
    from test_timeline import fragment, write

    from yap_studio.project.lock import ConflictError
    from yap_studio.stages import timeline

    root, _ = project_pipeline
    style = json.loads((root / "style.json").read_bytes())
    style["hook"]["duration_s"] = 2 if change == "approve" else 4
    write(root / "style.json", style)
    identity = None if change == "approve" else _approved(_body(2, field="hook.duration_s"))
    set_elements(root)
    incoming = fragment(root, title(lane="agent", start={"time": 0}))
    before = (root / "timeline/agent.json").read_bytes()
    used = ProjectContext(root).taste_application().pin()
    original_lock = timeline.project_lock

    @contextmanager
    def changed_before_write(project_dir: Path):
        with original_lock(project_dir):
            if change == "approve":
                _approved(_body(4, field="hook.duration_s"))
            elif change == "revoke":
                revoke(identity)
            else:
                # Returning to the same approved rules still advances the revision.
                revoke(_approved())
            current = load_taste().snapshot()
            assert current.revision != used["revision"]
            assert (current.sha256 == used["sha256"]) == (change == "revision_only")
            yield

    monkeypatch.setattr(timeline, "project_lock", changed_before_write)
    with pytest.raises(ConflictError, match="taste.json: the base changed before the write"):
        timeline.add_timeline(root, incoming, lane="agent")
    assert (root / "timeline/agent.json").read_bytes() == before


@pytest.mark.parametrize("taste", ["approved", "absent", "empty", "pending"])
def test_timeline_add_commits_with_unchanged_taste(
    project_pipeline: tuple[Path, dict], taste: str,
) -> None:
    from test_stage_captions import set_elements
    from test_stage_graphics import title
    from test_timeline import fragment, write

    from yap_studio.stages.timeline import add_timeline, resolve_timeline

    root, _ = project_pipeline
    style = json.loads((root / "style.json").read_bytes())
    style["hook"]["duration_s"] = 4 if taste == "approved" else 2
    write(root / "style.json", style)
    if taste == "approved":
        _approved(_body(2, field="hook.duration_s"))
    elif taste == "empty":
        write(taste_path(), Taste().to_json())
    elif taste == "pending":
        add_proposals(_body(4, field="hook.duration_s"))
    set_elements(root)
    result = add_timeline(root, fragment(root, title(lane="agent", start={"time": 0})),
                          lane="agent").require_valid()
    assert result.elements[0].end_frame == 60
    assert resolve_timeline(root).to_dict() == result.to_dict()


def test_graphics_revalidates_lane_committed_against_older_taste(
    project_pipeline: tuple[Path, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_stage_captions import set_elements
    from test_stage_graphics import title
    from test_timeline import fragment

    from yap_studio.stages import graphics
    from yap_studio.stages.timeline import TimelineRefusal, add_timeline, resolve_timeline

    root, _ = project_pipeline
    _approved(_body(2, field="hook.duration_s"))
    set_elements(root)
    result = add_timeline(root, fragment(root, title(lane="agent", start={"time": 0})),
                          lane="agent").require_valid()
    assert result.elements[0].end_frame == 60
    lane = (root / "timeline/agent.json").read_bytes()
    previous = (root / "graphics/receipt.json").read_bytes()
    rendered = []
    monkeypatch.setattr(graphics, "render_layer_plan", lambda *args, **kwargs: rendered.append(
        args,
    ))
    _approved(_body(4, field="hook.duration_s"))
    assert not resolve_timeline(root).valid
    with pytest.raises(TimelineRefusal):
        graphics.graphics(root)
    assert rendered == []
    assert (root / "timeline/agent.json").read_bytes() == lane
    assert (root / "graphics/receipt.json").read_bytes() == previous


def test_render_style_applies_only_approved_rules() -> None:
    identity = _approved()
    add_proposals(_body("#654321"))
    base = resolve("preset:clean_minimal")
    before = copy.deepcopy(base.data)
    application = render_style("preset:clean_minimal")
    assert application.style.data["captions"]["text_color"] == "#123456"
    assert application.applied == ({
        "id": identity, "field": "captions.text_color", "value": "#123456",
    },)
    assert application.taste_revision == 1
    assert application.taste_sha256 == load_taste().snapshot().sha256
    assert base.data == before


@pytest.mark.parametrize("spec_kind", ["default", "preset", "path", "resolved"])
def test_render_style_accepts_style_specs(tmp_path: Path, spec_kind: str) -> None:
    base = resolve("preset:clean_minimal")
    spec = None
    if spec_kind == "preset":
        spec = "preset:clean_minimal"
    elif spec_kind == "path":
        spec = tmp_path / "style.json"
        spec.write_text(json.dumps(base.data), encoding="utf-8")
    elif spec_kind == "resolved":
        spec = base
    application = render_style(spec, snapshot=Taste().snapshot())
    assert application.style.data == base.data
    assert application.style.sha256() == base.sha256()


def test_project_instructions_win() -> None:
    identity = _approved()
    application = render_style(explicit={"captions.text_color": "#abcdef"})
    assert application.style.data["captions"]["text_color"] == "#abcdef"
    assert application.applied == ()
    assert application.skipped == ({"id": identity, "reason": "project_override"},)
    assert application.explicit_fields == ("captions.text_color",)


def test_conditional_rule_needs_all_tags() -> None:
    identity = _approved(_body(condition={"text": "For an opening.", "tags": ["opening"]}))
    application = render_style(tags=("screen",))
    assert application.applied == ()
    assert application.skipped == ({"id": identity, "reason": "condition_not_met"},)
    assert render_style(tags=("opening", "screen")).style.data["captions"]["text_color"] == (
        "#123456"
    )


@pytest.mark.parametrize("field,value", [
    ("internal.fps", 24), ("internal.safe_bottom_px", 0), ("name", "Other"),
    ("captions.max_words_per_phrase", 9),
])
def test_engine_limits_hold_for_instructions_and_supplied_rules(field: str, value: Any) -> None:
    with pytest.raises(TasteError):
        render_style(explicit={field: value})
    snapshot = replace(Taste().snapshot(), rules=({"id": "r-123456789abc", **_body(
        value, field=field,
    )},))
    with pytest.raises(TasteError):
        render_style(snapshot=snapshot)


def test_pin_contains_only_identities_and_decision_codes() -> None:
    identity = _approved()
    skipped = _approved(_body(
        "#654321", condition={"text": "Use judgement for this condition.", "tags": []},
    ))
    application = render_style(explicit={"hook.text_color": "#abcdef"})
    pin = application.pin()
    assert pin == {
        "revision": 2, "sha256": load_taste().snapshot().sha256,
        "style_sha256": application.style.sha256(),
        "applied": [{"id": identity, "field": "captions.text_color"}],
        "skipped": [{"id": skipped, "reason": "condition_needs_judgement"}],
        "explicit_fields": ["hook.text_color"],
    }
    text = json.dumps(pin)
    for private in ("value", "sentence", "sources", "context", "#123456", "#654321", "#abcdef"):
        assert private not in text
    pin["skipped"][0]["reason"] = "changed"
    assert application.skipped[0]["reason"] == "condition_needs_judgement"


def test_taste_record_round_trip_and_replacement(tmp_path: Path) -> None:
    _approved()
    media = tmp_path / "out.mp4"
    media.write_bytes(b"first render")
    application = render_style()
    target = write_taste_record(media, application)
    assert target == tmp_path / "out.taste.json"
    expected = {
        "schema": TASTE_RECORD_SCHEMA, "media_sha256": sha256_file(media), **application.pin(),
    }
    assert read_taste_record(media) == expected
    assert json.loads(target.read_text()) == expected
    media.write_bytes(b"second render")
    write_taste_record(media, application)
    assert read_taste_record(media)["media_sha256"] == sha256_file(media)
    assert not list(tmp_path.glob(".yap_tmp_*"))


def test_taste_record_round_trip_with_applied_broll_field(tmp_path: Path) -> None:
    field = "broll.photo_placement"
    identity = _approved({**_body("full_frame", field=field), "scope": "broll"})
    application = render_style()
    assert application.style.data["broll"]["photo_placement"] == "full_frame"
    media = tmp_path / "out.mp4"
    media.write_bytes(b"render with approved B-roll taste")
    target = write_taste_record(media, application)
    expected = {
        "schema": TASTE_RECORD_SCHEMA, "media_sha256": sha256_file(media), **application.pin(),
    }
    assert expected["applied"] == [{"id": identity, "field": field}]
    assert target == media.with_suffix(".taste.json")
    assert json.loads(target.read_text()) == expected
    assert read_taste_record(media) == expected


@pytest.mark.parametrize("matching", [False, True])
def test_staged_taste_record_checks_published_media_receipt(tmp_path: Path, matching: bool) -> None:
    media = tmp_path / "out.mp4"
    media.write_bytes(b"published render")
    original = write_taste_record(media, render_style())
    before = original.read_bytes()
    _approved()
    application = render_style()
    media.with_suffix(".receipt.json").write_text(json.dumps({
        "style": {"sha256": application.style.sha256() if matching else "a" * 64},
        "outputs": [{"sha256": sha256_file(media)}],
    }))
    staged = tmp_path / "staged.taste.json"
    if matching:
        assert write_taste_record(media, application, out=staged) == staged
        assert json.loads(staged.read_bytes()) == {
            "schema": TASTE_RECORD_SCHEMA, "media_sha256": sha256_file(media), **application.pin(),
        }
    else:
        with pytest.raises(InputError, match="render changed while its taste was being recorded"):
            write_taste_record(media, application, out=staged)
        assert not staged.exists()
    assert original.read_bytes() == before


def test_taste_record_matching_receipt_hashes_media_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _approved()
    application = render_style()
    media = tmp_path / "out.mp4"
    media.write_bytes(b"render")
    digest = sha256_file(media)
    media.with_suffix(".receipt.json").write_text(json.dumps({
        "style": {"sha256": application.style.sha256()},
        "outputs": [{"path": media.name, "sha256": digest}],
    }))
    hashed = []

    def hash_once(source: Path) -> str:
        hashed.append(source)
        return sha256_file(source)

    monkeypatch.setattr(render_module, "sha256_file", hash_once)
    target = write_taste_record(media, application)
    assert hashed == [media]
    assert json.loads(target.read_text())["media_sha256"] == digest
    assert read_taste_record(media)["style_sha256"] == application.style.sha256()


def test_taste_record_replacement_uses_receipt_free_copy(tmp_path: Path) -> None:
    first = _approved()
    application = render_style()
    media = tmp_path / "out.mp4"
    original = b"finished render"
    media.write_bytes(original)
    receipt = media.with_suffix(".receipt.json")
    receipt.write_text(json.dumps({
        "style": {"sha256": application.style.sha256()},
        "outputs": [{"path": media.name, "sha256": sha256_file(media)}],
    }))
    target = write_taste_record(media, application)
    before_record = target.read_bytes()
    before_receipt = receipt.read_bytes()
    second = _approved(_body("#FF0000"))
    replacement = render_style()
    assert replacement.style.sha256() != application.style.sha256()
    with pytest.raises(InputError, match="render changed while its taste was being recorded"):
        write_taste_record(media, replacement)

    # Independent copies have no publisher receipt tying them to one render style.
    copied = tmp_path / "check.mp4"
    copied.write_bytes(media.read_bytes())
    write_taste_record(copied, application)
    assert read_taste_record(copied)["applied"] == [{"id": first, "field": "captions.text_color"}]
    write_taste_record(copied, replacement)
    assert read_taste_record(copied) == {
        "schema": TASTE_RECORD_SCHEMA, "media_sha256": sha256_file(media), **replacement.pin(),
    }
    assert read_taste_record(copied)["applied"] == [{"id": second, "field": "captions.text_color"}]
    assert not copied.with_suffix(".receipt.json").exists()
    assert copied.read_bytes() == media.read_bytes() == original
    assert target.read_bytes() == before_record
    assert receipt.read_bytes() == before_receipt


@pytest.mark.parametrize("bad", ["style", "media", "changed_media", "outputs", "missing", "json"])
def test_taste_record_refuses_mismatched_receipt(tmp_path: Path, bad: str) -> None:
    media = tmp_path / "out.mp4"
    media.write_bytes(b"render")
    application = render_style()
    target = write_taste_record(media, application)
    before = target.read_bytes()
    receipt = {
        "style": {"sha256": application.style.sha256()},
        "outputs": [{"path": media.name, "sha256": sha256_file(media)}],
    }
    if bad == "style":
        receipt["style"]["sha256"] = "a" * 64
    elif bad == "media":
        receipt["outputs"][0]["sha256"] = "a" * 64
    elif bad == "changed_media":
        media.write_bytes(b"different render")
    elif bad == "outputs":
        receipt["outputs"] = None
    elif bad == "missing":
        receipt = {}
    media.with_suffix(".receipt.json").write_text("{" if bad == "json" else json.dumps(receipt))
    with pytest.raises(
        InputError, match="render changed while its taste was being recorded",
    ) as exc:
        write_taste_record(media, application)
    assert "render again" in exc.value.message.lower()
    assert target.read_bytes() == before


def test_absent_taste_record(tmp_path: Path) -> None:
    assert read_taste_record(tmp_path / "not-rendered.mp4") is None


@pytest.mark.parametrize("bad", [
    "json", "root", "extra", "missing", "schema", "revision_bool", "revision_negative",
    "media_sha256", "sha256", "style_sha256", "applied_type", "applied_id", "applied_value",
    "applied_field", "skipped_type", "skipped_id", "skipped_reason", "skipped_extra",
    "explicit_type", "explicit_field", "duplicate_key", "unicode",
])
def test_corrupt_taste_records_refused(tmp_path: Path, bad: str) -> None:
    media = tmp_path / "out.mp4"
    media.write_bytes(b"render")
    target = write_taste_record(media, render_style())
    data = json.loads(target.read_text())
    if bad == "json":
        target.write_text("{broken", encoding="utf-8")
    elif bad == "root":
        target.write_text("[]", encoding="utf-8")
    elif bad == "duplicate_key":
        target.write_text(target.read_text().replace(
            '"revision": 0', '"revision": 0, "revision": 1',
        ))
    elif bad == "unicode":
        target.write_bytes(b"\xff")
    else:
        if bad == "extra":
            data["sentence"] = "Private words."
        elif bad == "missing":
            del data["explicit_fields"]
        elif bad == "schema":
            data["schema"] = "unknown"
        elif bad == "revision_bool":
            data["revision"] = True
        elif bad == "revision_negative":
            data["revision"] = -1
        elif bad in {"media_sha256", "sha256", "style_sha256"}:
            data[bad] = "invalid"
        elif bad == "applied_type":
            data["applied"] = {}
        elif bad.startswith("applied_"):
            item = {"id": "r-123456789abc", "field": "captions.text_color"}
            item[bad.removeprefix("applied_")] = 3
            data["applied"] = [item]
        elif bad == "skipped_type":
            data["skipped"] = "none"
        elif bad.startswith("skipped_"):
            item = {"id": "r-123456789abc", "reason": "condition_not_met"}
            item[bad.removeprefix("skipped_")] = 3
            data["skipped"] = [item]
        else:
            data["explicit_fields"] = {} if bad == "explicit_type" else [3]
        target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InputError):
        read_taste_record(media)


def test_mismatched_media_refused(tmp_path: Path) -> None:
    media = tmp_path / "out.mp4"
    media.write_bytes(b"render")
    write_taste_record(media, render_style())
    media.write_bytes(b"different render")
    with pytest.raises(InputError, match="media hash does not match"):
        read_taste_record(media)


@pytest.mark.parametrize("unsafe", ["symlink", "unowned", "input", "media_hardlink"])
def test_record_writer_respects_output_safety(tmp_path: Path, unsafe: str) -> None:
    media = tmp_path / "out.mp4"
    media.write_bytes(b"render")
    target = media.with_suffix(".taste.json")
    original = b"unrelated file"
    if unsafe == "symlink":
        other = tmp_path / "other.json"
        other.write_bytes(original)
        target.symlink_to(other)
    elif unsafe == "media_hardlink":
        target.hardlink_to(media)
        original = media.read_bytes()
    else:
        target.write_bytes(original)
    with paths.operation():
        if unsafe == "input":
            paths.validate_input(target)
        with pytest.raises(OutputPathError):
            write_taste_record(media, render_style())
    assert target.read_bytes() == original
    assert media.read_bytes() == b"render"


def test_record_write_failure_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = tmp_path / "out.mp4"
    media.write_bytes(b"render")
    application = render_style()
    target = write_taste_record(media, application)
    original = target.read_bytes()

    def refuse_replace(source: Path, destination: Path) -> None:
        raise OSError("publication failed")

    monkeypatch.setattr(paths.os, "replace", refuse_replace)
    with pytest.raises(InputError, match="publication failed"):
        write_taste_record(media, application)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".yap_tmp_*"))


def test_snapshot_at_matches_snapshots_across_all_decisions() -> None:
    recorded = {0: Taste().snapshot()}

    def remember() -> None:
        taste = load_taste()
        recorded[taste.revision] = taste.snapshot()

    first_body = _body()
    first = _approved(first_body)
    remember()
    conditional = _approved(_body(condition={"text": "For an opening.", "tags": ["opening"]}))
    remember()
    replacement = _approved(_body("#654321"))
    remember()
    revoke(replacement)
    remember()
    first_body["context"] = "New evidence after revocation."
    assert _approved(first_body) == first
    remember()
    revoke(first)
    remember()
    assert _approved() == first
    remember()
    revoke(conditional)
    remember()
    taste = load_taste()
    before = copy.deepcopy(taste.to_json())
    assert {item["action"] for item in taste.history} == {
        "approved", "superseded", "revoked", "reapproved",
    }
    for revision, expected in recorded.items():
        assert snapshot_at(taste, revision) == expected
    assert taste.to_json() == before
    restored = render_style(snapshot=snapshot_at(taste, 3), tags=("opening",))
    assert restored.taste_revision == 3
    assert restored.taste_sha256 == recorded[3].sha256


@pytest.mark.parametrize("revision", [-1, True, 1.0, "1", 1])
def test_snapshot_at_rejects_invalid_or_future_revision(revision: Any) -> None:
    with pytest.raises(TasteError):
        snapshot_at(Taste(), revision)


def test_migrated_history_mismatch_is_detectable() -> None:
    identity = _approved()
    original = load_taste().snapshot()
    revoke(identity)
    _approved()
    legacy = load_taste().to_json()
    del legacy["history"]
    taste = validate_taste(legacy)
    recovered = snapshot_at(taste, original.revision)
    assert recovered.sha256 != original.sha256
    assert snapshot_at(taste, taste.revision) == taste.snapshot()
