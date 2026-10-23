"""Per-phrase backing contracts, receipt freshness and project rendering."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import jsonschema
import pytest
from test_captions_render import _phrases, _style
from test_cli_captions import fake_layers as fake_layers
from test_stage_captions import make_project, set_elements
from test_stage_captions import no_face_helper as no_face_helper

from yap_studio import captions as phrase_module
from yap_studio import cli, receipt
from yap_studio.captions import CaptionWord
from yap_studio.errors import InputError
from yap_studio.project.model import SCHEMAS_DIR, ProjectFileError, parse_document
from yap_studio.project.receipts import StageRun, projection_sha256, read_receipt
from yap_studio.project.status import ProjectContext, cut_inputs
from yap_studio.render.text_metrics import text_width_em
from yap_studio.stages import captions as stage


@pytest.mark.parametrize("ids,index,key", [
    ([], 4, "caption:4"), (["t1:2"], 0, "caption:t1:2"),
    (["t1:2", "t1:3", "t1:4"], 0, "caption:t1:2-t1:4"),
])
def test_phrase_key(ids, index, key) -> None:
    assert phrase_module.phrase_key(ids, index) == key


def test_backing_occurrence_keys() -> None:
    assert phrase_module.phrase_keys([
        ["t1:0", "t1:1"], [], ["t1:0", "t1:1"], ["t1:4"], ["t1:0", "t1:1"],
    ]) == ["caption:t1:0-t1:1", "caption:1", "caption:t1:0-t1:1#2", "caption:t1:4",
           "caption:t1:0-t1:1#3"]


def test_only_second_repeated_phrase_gets_backing(tmp_path, monkeypatch, fake_layers) -> None:
    from test_caption_contrast import contrast_document

    from yap_studio.stages import assemble

    root = make_project(tmp_path)
    edit_path = root / "edit.json"
    edit = json.loads(edit_path.read_bytes())
    edit["segments"][0]["to"] = {"word": "t1:1"}
    edit["segments"].append({**edit["segments"][0], "id": "s2"})
    edit_path.write_text(json.dumps(edit))
    cut_words = json.loads((root / "cut/cut-words.json").read_bytes())
    for first, second in zip(cut_words["words"][:2], cut_words["words"][2:], strict=True):
        second.update(id=first["id"], text=first["text"], segment="s2")
    staged = root / "repeated-cut-words.json"
    staged.write_text(json.dumps(cut_words))
    previous = read_receipt(root / "cut")
    with StageRun(root, "cut") as run:
        for name, digest in cut_inputs(ProjectContext(root)).items():
            run.record_inputs(name, digest)
        run.outputs.update({item["path"]: item for item in previous["outputs"]
                            if item["path"] != "cut/cut-words.json"})
        run.replace_output(root / "cut/cut-words.json", staged)
        run.finish(extra={"plan": previous["plan"]})
    stage.captions(root)
    phrases = json.loads((root / "captions/phrases.json").read_bytes())["phrases"]
    assert len(phrases) == 2 and phrases[0]["ids"] == phrases[1]["ids"]
    document = contrast_document()
    weak = document["captions"][0]
    weak.update(id="caption-1", text=phrases[1]["text"], start_frame=45, end_frame=75,
                frame=50, word=0, words=[{"index": 0, "text": "simple", "ratio": 1.2,
                                         "frame": 50, "samples": 5}])
    monkeypatch.setattr(assemble, "measure_caption_contrast", lambda *a, **k: document)
    final = root / "synthetic-final.mp4"
    final.write_bytes((root / "cut/cut.mp4").read_bytes())
    layer = stage.caption_snapshot(ProjectContext(root)).plan.placement("captions/captions.mov")
    _, summary = assemble._final_contrast(SimpleNamespace(caption_layer=layer), root, final)
    (root / "final").mkdir()
    (root / "final/receipt.json").write_text(json.dumps({
        "schema": "yap-studio/stage-receipt@1", "status": "failed",
        "inputs": {"media:captions/captions.mov": receipt.sha256_file(root / layer["path"])},
        "caption_contrast": summary,
    }))
    result = stage.back_weak_captions(root, "shadow")
    slots = json.loads(fake_layers[-1]["variables"]["phrases"])
    assert [slot.get("backing") for slot in slots] == [None, "shadow"]
    key = "caption:t1:0-t1:1#2"
    assert summary["weak"][0]["key"] == result["backed"][0]["key"] == key
    assert json.loads((root / "project.json").read_bytes())["caption_backing"] == {key: "shadow"}


@pytest.mark.parametrize("backing", [{}, {"caption:t1:0-t1:2": "shadow", "caption:3": "box"},
                                    {"caption:0": "none"}])
def test_project_backing_round_trip(backing) -> None:
    data = {"schema": "yap-studio/project@1", "name": "sample",
            "takes": [{"id": "t1", "path": "take.mov"}], "caption_backing": backing}
    schema = json.loads((SCHEMAS_DIR / "project.schema.json").read_bytes())
    jsonschema.validate(data, schema)
    project = parse_document("project", data)
    assert project.caption_backing == backing
    assert parse_document("project", project.to_dict()) == project


@pytest.mark.parametrize("backing", [{"caption:0": "invalid"}, {"word:0": "box"},
                                    {"caption:0": None}, []])
def test_project_rejects_invalid_backing(backing) -> None:
    data = {"schema": "yap-studio/project@1", "name": "sample",
            "takes": [{"id": "t1", "path": "take.mov"}], "caption_backing": backing}
    with pytest.raises(ProjectFileError):
        parse_document("project", data)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, json.loads((SCHEMAS_DIR / "project.schema.json").read_bytes()))


def test_caption_plan_backing_slots_and_original_bytes() -> None:
    style = _style()
    phrases = _phrases(style)
    original = stage.caption_layer_plan(phrases, style, frames=90)
    explicit = stage.caption_layer_plan(phrases, style, frames=90, backing=[None] * 3)
    assert explicit == original
    assert json.dumps(explicit.variables) == json.dumps(original.variables)
    backed = stage.caption_layer_plan(phrases, style, frames=90, backing=[None, "shadow", "box"])
    slots = json.loads(backed.variables["phrases"])
    assert [slot.get("backing") for slot in slots] == [None, "shadow", "box"]


@pytest.mark.parametrize("background,opacity", [("none", 0.85), ("band", 0)])
def test_box_backing_plans_breaks_without_changing_plain_slots(background, opacity) -> None:
    texts = "Small steady steps matter".split()
    size = round(920 / text_width_em(" ".join(texts), "Inter", 400))
    style = _style(highlight="none", font="Inter", size_px=size, width_scale=1,
                   letter_spacing_px=0, italic_stress=False, background_mode=background,
                   background_opacity=opacity, background_padding_px=24)
    assert 940 - 2 * 24 < text_width_em(" ".join(texts), "Inter", 400) * size < 940
    phrase = replace(_phrases(style)[0], words=tuple(
        CaptionWord(str(i), text, 0.2 + i * 0.1, 0.3 + i * 0.1)
        for i, text in enumerate(texts)))
    plain = stage.caption_layer_plan([phrase], style, frames=90, backing=[None])
    assert "breaks" not in json.loads(plain.variables["phrases"])[0]
    boxed = stage.caption_layer_plan([phrase], style, frames=90, backing=["box"])
    assert json.loads(boxed.variables["phrases"])[0]["breaks"] == [2]
    phrases = [phrase, _phrases(style)[1]]
    original = stage.caption_layer_plan(phrases, style, frames=90)
    mixed = stage.caption_layer_plan(phrases, style, frames=90, backing=["box", None])
    assert json.dumps(json.loads(mixed.variables["phrases"])[1]) == json.dumps(
        json.loads(original.variables["phrases"])[1])


@pytest.mark.parametrize("backing", [[], [None], [None] * 4, [None, "none", "box"],
                                    [None, True, "box"]])
def test_caption_plan_rejects_invalid_backing(backing) -> None:
    style = _style()
    with pytest.raises(ValueError, match="backing"):
        stage.caption_layer_plan(_phrases(style), style, frames=90, backing=backing)


@pytest.mark.parametrize("mode,opacity", [("none", 1), ("box", 0), ("box", 0.85), ("band", 1)])
def test_backing_band_padding(mode, opacity) -> None:
    tokens = _style(highlight="none", size_px=180, shadow_blur_px=0, shadow_x_px=0,
                    shadow_y_px=0, background_mode=mode, background_opacity=opacity,
                    background_padding_px=24).data["captions"]
    plain = stage.caption_band_geometry(tokens, safe_bottom_px=1740)
    assert stage.caption_band_geometry(tokens, safe_bottom_px=1740, backing=()) == plain
    box = stage.caption_band_geometry(tokens, safe_bottom_px=1740, backing=["box"])
    assert box.padding == plain.padding + (24 if mode == "none" or opacity == 0 else 0)
    shadow = stage.caption_band_geometry(tokens, safe_bottom_px=1740, backing=["shadow"])
    assert shadow.padding == max(plain.padding, math.ceil(0.12 * 180) + 8)


def test_box_backing_uses_fitted_width_without_affecting_next_phrase(monkeypatch) -> None:
    style = _style(background_mode="none", placement="avoid_face")
    geometry = stage.caption_band_geometry(style.data["captions"], safe_bottom_px=1740)
    slots = json.loads(stage.caption_layer_plan(_phrases(style), style, frames=90).variables[
        "phrases"])
    slots[0]["backing"] = "box"
    widths = []
    monkeypatch.setattr(stage, "_phrase_box_width", lambda *args: 123)
    monkeypatch.setattr(stage, "gap_px", lambda element, head: widths.append(element.w) or 999)
    stage.caption_offsets(slots, geometry, None, style,
                          still_faces=[stage.Box(0, 0, 100, 100)])
    assert widths == [123, 940, 940]


@pytest.mark.parametrize("moved", [False, True])
def test_stage_backing_projection_and_render(tmp_path, fake_layers, monkeypatch, moved) -> None:
    root = make_project(tmp_path)
    original = stage.caption_snapshot(ProjectContext(root))
    assert "caption_backing" not in original.inputs
    path = root / "project.json"
    data = json.loads(path.read_bytes())
    data["caption_backing"] = {"caption:unused": "box"}
    path.write_text(json.dumps(data))
    assert stage.caption_snapshot(ProjectContext(root)).inputs == original.inputs
    key = phrase_module.phrase_key(original.phrases[0].word_ids, 0)
    data["caption_backing"][key] = "shadow"
    path.write_text(json.dumps(data))
    backed = stage.caption_snapshot(ProjectContext(root))
    expected = ["shadow"] + [None] * (len(original.phrases) - 1)
    assert backed.inputs["caption_backing"] == projection_sha256(expected)
    if moved:
        def offsets(slots, *args, **kwargs):
            return [{"slot": i, "text": "phrase", "offset_px": -50 if i == 0 else 0,
                     "reason": "moved up"} for i in range(len(slots))]
        monkeypatch.setattr(stage, "caption_offsets", offsets)
    assert not stage.captions(root)["noop"]
    slot = json.loads(fake_layers[-1]["variables"]["phrases"])[0]
    assert slot["backing"] == "shadow"
    assert slot.get("offset_px", 0) == (-50 if moved else 0)
    assert stage.captions(root)["noop"]
    data["caption_backing"][key] = "box"
    path.write_text(json.dumps(data))
    assert not stage.captions(root)["noop"]
    assert json.loads(fake_layers[-1]["variables"]["phrases"])[0]["backing"] == "box"


def _measured_project(tmp_path):
    root = make_project(tmp_path)
    stage.captions(root)
    saved = read_receipt(root / "captions")
    digest = next(item["sha256"] for item in saved["outputs"]
                  if item["path"] == "captions/captions.mov")
    snapshot = stage.caption_snapshot(ProjectContext(root))
    weak = [{"id": "caption-0", "text": snapshot.phrases[0].text,
             "ids": list(snapshot.phrases[0].word_ids)}, {"text": "Old phrase"}]
    final = {"schema": "yap-studio/stage-receipt@1", "status": "failed",
             "inputs": {"media:captions/captions.mov": digest},
             "caption_contrast": {"measured": True, "weak": weak}}
    (root / "final").mkdir()
    (root / "final/receipt.json").write_text(json.dumps(final))
    return root, final, snapshot


def _save_snapshot_halos(root, snapshot):
    keys = phrase_module.phrase_keys([phrase.word_ids for phrase in snapshot.phrases])
    (root / stage.HALO_PATH).write_text(json.dumps({
        "schema": stage.HALO_SCHEMA, "phrases": {
            key: {"text": phrase.text, "ratio": 1.2, "window": window}
            for key, phrase, window in zip(keys, snapshot.phrases, snapshot.windows, strict=True)
        },
    }))
    return keys


def test_caption_windows_use_supplied_timeline(tmp_path, monkeypatch) -> None:
    context = ProjectContext(make_project(tmp_path))
    snapshot = stage.caption_snapshot(context)
    timeline = context.resolved_timeline()
    monkeypatch.setattr(context, "resolved_timeline",
                        lambda: pytest.fail("Caption windows resolved the timeline again"))
    assert stage.caption_windows(context, timeline, snapshot.phrases) == snapshot.windows


def test_snapshot_halo_invalidates_changed_caption_color(tmp_path) -> None:
    root = make_project(tmp_path)
    original = stage.caption_snapshot(ProjectContext(root))
    keys = _save_snapshot_halos(root, original)
    assert stage.caption_snapshot(ProjectContext(root)).backing[0] == "shadow"
    path = root / "style.json"
    style = json.loads(path.read_bytes())
    style["captions"]["text_color"] = "#123456"
    path.write_text(json.dumps(style))
    changed = stage.caption_snapshot(ProjectContext(root))
    assert changed.backing[0] is None and changed.auto[0] is False
    path = root / "project.json"
    project = json.loads(path.read_bytes())
    project["caption_backing"] = {keys[0]: "box"}
    path.write_text(json.dumps(project))
    style["captions"]["text_color"] = "#654321"
    (root / "style.json").write_text(json.dumps(style))
    explicit = stage.caption_snapshot(ProjectContext(root))
    assert explicit.backing[0] == "box" and explicit.auto[0] is False


def test_snapshot_halo_invalidates_only_overlapping_timeline_window(tmp_path) -> None:
    root = make_project(tmp_path)
    original = stage.caption_snapshot(ProjectContext(root))
    _save_snapshot_halos(root, original)
    assert stage.caption_snapshot(ProjectContext(root)).backing == ("shadow", "shadow")
    set_elements(root, {"id": "agent-title", "kind": "title", "start": {"time": 0},
                        "lines": ["Small steps"], "accent": "steps"})
    changed = stage.caption_snapshot(ProjectContext(root))
    assert changed.windows[0] != original.windows[0]
    assert changed.windows[1] == original.windows[1]
    assert changed.backing == (None, "shadow") and changed.auto == (False, True)


def test_snapshot_ignores_halo_without_window(tmp_path) -> None:
    root = make_project(tmp_path)
    original = stage.caption_snapshot(ProjectContext(root))
    key = phrase_module.phrase_keys([phrase.word_ids for phrase in original.phrases])[0]
    (root / stage.HALO_PATH).write_text(json.dumps({
        "schema": stage.HALO_SCHEMA,
        "phrases": {key: {"text": original.phrases[0].text, "ratio": 1.2}},
    }))
    snapshot = stage.caption_snapshot(ProjectContext(root))
    assert snapshot.backing == (None, None) and snapshot.auto == (False, False)
    assert snapshot.plan.variables == original.plan.variables


@pytest.mark.parametrize("window", [None, "a" * 63, "A" * 64, "g" * 64, 123])
def test_caption_halos_reject_invalid_window(tmp_path, window) -> None:
    (tmp_path / ".yap").mkdir()
    (tmp_path / "project.json").write_text(json.dumps({
        "schema": "yap-studio/project@1", "name": "sample",
        "takes": [{"id": "t1", "path": "take.mov"}],
    }))
    (tmp_path / stage.HALO_PATH).write_text(json.dumps({
        "schema": stage.HALO_SCHEMA,
        "phrases": {"caption:0": {"text": "Small steps", "ratio": 1.2, "window": window}},
    }))
    with pytest.raises(InputError, match="invalid caption halo phrase"):
        stage.caption_halos(ProjectContext(tmp_path))


def test_add_caption_halos_refreshes_prunes_and_preserves_matching_entries(
    tmp_path, monkeypatch,
) -> None:
    root = make_project(tmp_path)
    path = root / "project.json"
    project = json.loads(path.read_bytes())
    project["caption_backing"] = {"caption:explicit": "box"}
    path.write_text(json.dumps(project))
    before = path.read_bytes()
    entry = {"text": "Small steps", "ratio": 1.2, "window": "a" * 64}
    saved = {key: dict(entry) for key in (
        "caption:stale", "caption:readable", "caption:gone", "caption:matching",
        "caption:explicit")}
    (root / stage.HALO_PATH).write_text(json.dumps({
        "schema": stage.HALO_SCHEMA, "phrases": saved,
    }))
    windows = {"caption:stale": "b" * 64, "caption:readable": "b" * 64,
               "caption:matching": "a" * 64, "caption:explicit": "b" * 64,
               "caption:new": "c" * 64}
    weak = [{"key": key, "text": "Small steps", "ratio": 1.2} for key in (
        "caption:stale", "caption:new", "caption:matching", "caption:explicit",
        "caption:unknown", None)]
    assert stage.add_caption_halos(root, weak, windows) == ["caption:stale", "caption:new"]
    assert stage.caption_halos(ProjectContext(root)) == {
        "caption:stale": {**entry, "window": "b" * 64},
        "caption:new": {**entry, "window": "c" * 64},
        "caption:matching": entry,
    }
    assert path.read_bytes() == before
    with monkeypatch.context() as unchanged:
        unchanged.setattr(stage.paths, "atomic_write_text",
                          lambda *a, **k: pytest.fail("Unchanged halos were rewritten"))
        assert stage.add_caption_halos(root, weak, windows) == []
    windows["caption:stale"] = "d" * 64
    assert stage.add_caption_halos(root, [], windows) == []
    assert "caption:stale" not in stage.caption_halos(ProjectContext(root))


@pytest.mark.parametrize("explicit,expected", [(None, "shadow"), ("shadow", "shadow"),
                                             ("box", "box"), ("none", None)])
def test_snapshot_halo_respects_explicit_choices(tmp_path, explicit, expected) -> None:
    root = make_project(tmp_path)
    original = stage.caption_snapshot(ProjectContext(root))
    key = phrase_module.phrase_keys([phrase.word_ids for phrase in original.phrases])[0]
    halo_path = root / ".yap/caption-halo.json"
    halo_path.write_text(json.dumps({"schema": "yap-studio/caption-halo@1", "phrases": {
        key: {"text": original.phrases[0].text, "ratio": 1.2, "window": original.windows[0]}}}))
    context = ProjectContext(root)
    retained = context.read(".yap/caption-halo.json")
    if explicit is not None:
        path = root / "project.json"
        data = json.loads(path.read_bytes())
        data["caption_backing"] = {key: explicit}
        path.write_text(json.dumps(data))
        context = ProjectContext(root)
    snapshot = stage.caption_snapshot(context)
    assert snapshot.backing[0] == expected
    assert context.read(".yap/caption-halo.json") == retained
    if expected is None:
        assert snapshot.inputs == original.inputs
        assert snapshot.plan.variables == original.plan.variables
    else:
        assert snapshot.inputs["caption_backing"] == projection_sha256(
            [expected] + [None] * (len(snapshot.phrases) - 1))
    halo_path.unlink()
    assert stage.caption_snapshot(context).backing == snapshot.backing


def test_cli_none_saves_halo_explicit_and_current_weak_choices(tmp_path, capsys) -> None:
    root, _, snapshot = _measured_project(tmp_path)
    path = root / "project.json"
    data = json.loads(path.read_bytes())
    data["caption_backing"] = {"caption:explicit": "shadow", "caption:off": "none"}
    path.write_text(json.dumps(data))
    (root / ".yap/caption-halo.json").write_text(json.dumps({
        "schema": "yap-studio/caption-halo@1",
        "phrases": {"caption:halo": {"text": "Another phrase", "ratio": 2.0}},
    }))
    assert cli.main(["captions", str(root), "--back", "none", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    key = phrase_module.phrase_key(snapshot.phrases[0].word_ids, 0)
    assert json.loads(path.read_bytes())["caption_backing"] == {
        "caption:explicit": "none", "caption:off": "none", "caption:halo": "none", key: "none"}
    assert len(result["backed"]) == 3 and result["skipped"] == 1


@pytest.mark.parametrize("look", ["shadow", "box"])
def test_cli_backs_failed_final_and_keeps_other_keys(tmp_path, capsys, fake_layers, look) -> None:
    root, _, snapshot = _measured_project(tmp_path)
    path = root / "project.json"
    data = json.loads(path.read_bytes())
    data["caption_backing"] = {"caption:other": "shadow"}
    path.write_text(json.dumps(data))
    halo_path = root / ".yap/caption-halo.json"
    halos = {phrase_module.phrase_key(phrase.word_ids, i): {
                 "text": phrase.text, "ratio": 1.2, "window": snapshot.windows[i]}
             for i, phrase in enumerate(snapshot.phrases)}
    halo_path.write_text(json.dumps({"schema": "yap-studio/caption-halo@1", "phrases": halos}))
    assert cli.main(["captions", str(root), "--back", look, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    key = phrase_module.phrase_key(snapshot.phrases[0].word_ids, 0)
    assert result["backed"] == [{"key": key, "text": snapshot.phrases[0].text, "look": look}]
    assert result["skipped"] == 1 and not result["captions"]["noop"]
    assert json.loads(path.read_bytes())["caption_backing"] == {
        "caption:other": "shadow", key: look}
    slots = json.loads(fake_layers[-1]["variables"]["phrases"])
    assert [slot.get("backing") for slot in slots] == [look, "shadow"]
    assert json.loads(halo_path.read_bytes())["phrases"] == halos


@pytest.mark.parametrize("problem", ["missing", "stale", "unmeasured", "no_digest",
                                    "no_receipt", "no_caption_receipt"])
def test_cli_refuses_missing_or_stale_measurement(tmp_path, capsys, problem) -> None:
    root, final, _ = _measured_project(tmp_path)
    if problem == "missing":
        del final["caption_contrast"]
    elif problem == "stale":
        final["inputs"]["media:captions/captions.mov"] = "f" * 64
    elif problem == "no_digest":
        final["inputs"] = {}
    elif problem == "unmeasured":
        final["caption_contrast"]["measured"] = False
    (root / "final/receipt.json").write_text(json.dumps(final))
    if problem == "no_receipt":
        (root / "final/receipt.json").unlink()
    elif problem == "no_caption_receipt":
        (root / "captions/receipt.json").unlink()
    before = (root / "project.json").read_bytes()
    assert cli.main(["captions", str(root), "--back", "shadow", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["suggested_action"] == (
        "Run yap-studio run DIR so the final measures caption contrast, then try again.")
    assert (root / "project.json").read_bytes() == before


def test_cli_turns_backing_off_without_measurement(tmp_path, capsys) -> None:
    root = make_project(tmp_path)
    path = root / "project.json"
    data = json.loads(path.read_bytes())
    data["caption_backing"] = {"caption:t1:0": "box"}
    path.write_text(json.dumps(data))
    (root / ".yap/caption-halo.json").write_text(json.dumps({
        "schema": "yap-studio/caption-halo@1",
        "phrases": {"caption:t1:1": {"text": "A phrase", "ratio": 1.2}},
    }))
    assert cli.main(["captions", str(root), "--back", "none"]) == 0
    assert json.loads(path.read_bytes())["caption_backing"] == {
        "caption:t1:0": "none", "caption:t1:1": "none"}
    assert capsys.readouterr().out.strip() == (
        f"Turned backing off for 2 phrases. Run yap-studio run {root} to render the final.")


def test_cli_backing_refuses_standalone(tmp_path, capsys) -> None:
    assert cli.main(["captions", str(tmp_path / "clip.mov"), "--back", "box", "--json"]) == 2
    assert "project" in json.loads(capsys.readouterr().err)["error"]["message"]


@pytest.mark.parametrize("look", ["shadow", "box"])
def test_cli_backing_human_output_for_verified_final(tmp_path, capsys, look) -> None:
    root, final, _ = _measured_project(tmp_path)
    final["status"] = "verified"
    (root / "final/receipt.json").write_text(json.dumps(final))
    assert cli.main(["captions", str(root), "--back", look]) == 0
    assert capsys.readouterr().out.strip() == (
        f"Backed 1 phrase with a {look}. Run yap-studio run {root} to render the final.")


def test_cli_backing_keeps_project_flag_rules(tmp_path, capsys) -> None:
    root = make_project(tmp_path)
    assert cli.main(["captions", str(root), "--back", "shadow", "--style", "preset:electric",
                     "--json"]) == 2
    assert "pass only DIR" in json.loads(capsys.readouterr().err)["error"]["message"]
