"""Project contracts, generated schemas, initialization, and concurrent writes."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import synth

from yap_studio import cli, paths, receipt
from yap_studio.errors import InputError, OutputPathError, YapError
from yap_studio.project import ProjectRefusal, init_project
from yap_studio.project.lock import ConflictError, project_lock, write_json_cas
from yap_studio.project.model import (
    RULE_DEFAULTS,
    SCHEMAS_DIR,
    SPECS,
    Anchor,
    CutWords,
    Edit,
    Project,
    ProjectFileError,
    Removals,
    TimelineLane,
    Words,
    load_cut_words,
    load_edit,
    load_project,
    load_removals,
    load_timeline,
    load_timeline_fragment,
    load_words,
    parse_document,
)
from yap_studio.style import add, set_default

ROOT = Path(__file__).parents[1]
EXAMPLES = json.loads((ROOT / "tests/fixtures/origin-examples.json").read_bytes())
LOADERS = {"project": load_project, "words": load_words, "edit": load_edit,
           "removals": load_removals, "cut-words": load_cut_words, "timeline": load_timeline,
           "timeline-fragment": load_timeline_fragment}
MODELS = {"project": Project, "words": Words, "edit": Edit, "removals": Removals,
          "cut-words": CutWords, "timeline": TimelineLane, "timeline-fragment": TimelineLane}


def _example(name: str) -> dict[str, Any]:
    return copy.deepcopy(next(item["data"] for item in EXAMPLES if item["contract"] == name))


def _schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMAS_DIR / f"{name}.schema.json").read_bytes())


def test_project_optional_grade() -> None:
    data = _example("project")
    assert parse_document("project", data).grade is None
    assert "grade" not in parse_document("project", data).to_dict()
    data["grade"] = {"look": "film", "strength": 0.7}
    assert parse_document("project", data).grade == data["grade"]
    jsonschema.validate(data, _schema("project"))
    for bad in ({"look": "other"}, {"strength": 2}, {"strength": True}, {"version": 1}, None):
        with pytest.raises(ProjectFileError):
            parse_document("project", {**data, "grade": bad})


@pytest.mark.parametrize("mode", ["preserve", "loudnorm"])
@pytest.mark.parametrize("enabled", [True, False])
def test_project_match_takes_is_optional_boolean(mode: str, enabled: bool) -> None:
    data = _example("project")
    policy = {"mode": mode}
    if mode == "loudnorm":
        policy.update({"integrated_lufs": -14, "true_peak_dbtp": -1.7, "lra_lu": 11})
    data["audio"] = {**policy, "match_takes": enabled}
    assert parse_document("project", data).audio["match_takes"] is enabled
    jsonschema.validate(data, _schema("project"))
    data["audio"]["match_takes"] = "false"
    with pytest.raises(ProjectFileError):
        parse_document("project", data)
    data["audio"] = policy
    assert "match_takes" not in parse_document("project", data).audio


@pytest.mark.parametrize("name", ["timeline", "timeline-fragment"])
@pytest.mark.parametrize("frames", [None, 6, 12.0, 24])
def test_transition_contract_defaults_and_round_trip(name: str, frames: int | None) -> None:
    data = {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "agent-transition", "kind": "transition", "lane": "agent",
        "start": {"word": "t1:2", "segment": "s2"}, "look": "light_leak",
    }]}
    if name == "timeline":
        data["words_sha256"] = {"t1": "a" * 64}
    if frames is not None:
        data["elements"][0]["frames"] = frames
    jsonschema.validate(data, _schema(name))
    normalized = parse_document(name, data).to_dict()
    assert normalized["elements"][0]["frames"] == (frames if frames is not None else 12)
    assert isinstance(normalized["elements"][0]["frames"], int)
    jsonschema.validate(normalized, _schema(name))


@pytest.mark.parametrize("changes", [
    {"frames": 5}, {"frames": 7}, {"frames": 26}, {"frames": True}, {"frames": 12.5},
    {"look": "other"}, {"lane": "broll"}, {"end": {"time": 4}}, {"hold_s": 1},
])
def test_transition_contract_rejects_invalid_fields(changes: dict) -> None:
    data = {"schema": "yap-studio/timeline@1", "elements": [{
        "id": "agent-transition", "kind": "transition", "lane": "agent",
        "start": {"word": "t1:2"}, "look": "whip_pan", **changes,
    }]}
    with pytest.raises(ProjectFileError):
        parse_document("timeline-fragment", data)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, _schema("timeline-fragment"))


@pytest.mark.parametrize("example", EXAMPLES, ids=[item["file"] for item in EXAMPLES])
def test_corrected_origin_examples_load_and_validate(example: dict, tmp_path: Path) -> None:
    name = example["contract"]
    schema = _schema(name)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(example["data"], schema)
    target = tmp_path / example["file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(example["data"]), encoding="utf-8")
    model = LOADERS[name](target)
    assert isinstance(model, MODELS[name])
    with pytest.raises(FrozenInstanceError):
        model.unexpected = True
    normalized = json.loads(json.dumps(model.to_dict()))
    jsonschema.validate(normalized, schema)
    assert isinstance(parse_document(name, normalized), MODELS[name])


def test_schema_generator_reproduces_committed_bytes(tmp_path: Path) -> None:
    subprocess.run([sys.executable, "scripts/gen_project_schemas.py", "--out-dir", str(tmp_path)],
                   cwd=ROOT, stdin=subprocess.DEVNULL, check=True, timeout=30)
    assert {path.name for path in tmp_path.iterdir()} == {
        f"{name}.schema.json" for name in SPECS
    }
    for generated in tmp_path.iterdir():
        assert generated.read_bytes() == (SCHEMAS_DIR / generated.name).read_bytes()
    subprocess.run([sys.executable, "scripts/gen_project_schemas.py", "--check"],
                   cwd=ROOT, stdin=subprocess.DEVNULL, check=True, timeout=30)
    (tmp_path / "project.schema.json").write_text("{}", encoding="utf-8")
    failed = subprocess.run(
        [sys.executable, "scripts/gen_project_schemas.py", "--out-dir", str(tmp_path), "--check"],
        cwd=ROOT, stdin=subprocess.DEVNULL, check=False, capture_output=True, timeout=30,
    )
    assert failed.returncode == 1


@pytest.mark.parametrize("value", [None, "track.wav", "audio/track.wav"])
def test_music_file_accepts_path_or_null(value: str | None, tmp_path: Path) -> None:
    data = _example("project")
    data["music"]["file"] = value
    assert parse_document("project", data).music["file"] == value
    jsonschema.validate(data, _schema("project"))
    data["music"]["file"] = str(tmp_path / "track.wav")
    jsonschema.validate(data, _schema("project"))
    assert parse_document("project", data).music["file"] == str(tmp_path / "track.wav")


@pytest.mark.parametrize("value", [0, True, [], {}, "", "track\x00.wav"])
def test_music_file_rejects_invalid_paths(value: Any) -> None:
    data = _example("project")
    data["music"]["file"] = value
    with pytest.raises(ProjectFileError, match=r"music\.file"):
        parse_document("project", data)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, _schema("project"))


@pytest.mark.parametrize("case", [
    "unknown_rule", "output", "sfx", "word_id", "mixed_anchor", "no_end", "both_ends",
    "unknown_element_field", "empty_title", "bad_crop", "small_blur", "bad_hash",
    "negative_audio_stream", "boolean_number", "fragment_graphic",
])
def test_invalid_contracts_fail_loader_and_schema(case: str) -> None:
    name = "project" if case in {
        "unknown_rule", "output", "negative_audio_stream", "boolean_number",
    } else "timeline"
    if case in {"bad_crop", "small_blur", "bad_hash", "fragment_graphic"}:
        name = "timeline-fragment"
    data = _example(name)
    if case == "unknown_rule":
        data["rules"]["unrecognized"] = 1
    elif case == "output":
        data["output"] = "render.mp4"
    elif case == "negative_audio_stream":
        data["takes"][0]["audio_stream"] = -1
    elif case == "boolean_number":
        data["takes"][0]["crop_x"] = True
    else:
        element = data["elements"][0]
        if case == "sfx":
            element["kind"] = "sfx"
        elif case == "word_id":
            element["start"]["word"] = "invalid"
        elif case == "mixed_anchor":
            element["start"]["time"] = 1
        elif case == "no_end":
            data["elements"] = [{"id": "agent-gap", "kind": "caption_gap",
                                 "start": {"time": 0}}]
        elif case == "both_ends":
            element["end"] = {"time": 3}
        elif case == "unknown_element_field":
            element["unrecognized"] = True
        elif case == "empty_title":
            element["lines"] = []
        elif case == "bad_crop":
            element["media"]["fit"] = "crop"
        elif case == "small_blur":
            element["media"]["blur"] = [[0, 0, 7, 20]]
        elif case == "bad_hash":
            element["media"]["sha256"] = "not-a-hash"
        else:
            element["kind"] = "graphic"
    with pytest.raises(ProjectFileError):
        parse_document(name, data)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, _schema(name))


def test_project_defaults_unknown_keys_and_precise_errors() -> None:
    data = {"schema": "yap-studio/project@1", "name": "sample",
            "takes": [{"id": "t1", "path": "take.mov"}]}
    project = parse_document("project", data)
    assert isinstance(project, Project)
    assert project.language == "en"
    assert project.rules == RULE_DEFAULTS
    assert project.asr == {"engine": "auto", "model": None}
    data["takes"].append({"id": "t1", "path": "other.mov"})
    with pytest.raises(ProjectFileError, match=r"takes\[1\].id"):
        parse_document("project", data)
    edit = _example("edit")
    edit["segments"][1]["from"] = {"word": "wrong"}
    with pytest.raises(ProjectFileError) as caught:
        parse_document("edit", edit)
    assert caught.value.location == "segments[1].from.word"
    assert "edit.json: segments[1].from.word" in caught.value.message
    assert "expected a word id like t1:12" in caught.value.message
    assert caught.value.suggested_action


@pytest.mark.parametrize("kind", ["title", "graphic"])
def test_style_duration_defaults_and_agent_fragments(kind: str, tmp_path: Path) -> None:
    element = {"id": "agent-one", "kind": kind, "lane": "agent", "start": {"word": "t1:0"}}
    element.update({"lines": ["alpha"]} if kind == "title"
                   else {"template": "text_tag", "slots": {"text": "alpha"}})
    data = {"schema": "yap-studio/timeline@1", "elements": [element]}
    jsonschema.validate(data, _schema("timeline-fragment"))
    target = tmp_path / "fragment.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    model = load_timeline_fragment(target, lane="agent")
    assert model.elements[0].hold_s is None and model.elements[0].end is None
    with pytest.raises(ProjectFileError):
        load_timeline_fragment(target)
    element["lane"] = "broll"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, _schema("timeline-fragment"))


def test_pause_rule_defaults_and_overrides() -> None:
    data = {"schema": "yap-studio/project@1", "name": "sample",
            "takes": [{"id": "t1", "path": "take.mov"}],
            "rules": {"max_pause_s": 0.6, "beat_breath_s": 0.5}}
    project = parse_document("project", data)
    assert project.rules["max_pause_s"] == 0.6
    assert project.rules["beat_breath_s"] == 0.5
    assert RULE_DEFAULTS["max_pause_s"] == 0.3
    assert RULE_DEFAULTS["beat_breath_s"] == 0.4
    jsonschema.validate(project.to_dict(), _schema("project"))
    for value in (-0.1, True):
        data["rules"]["max_pause_s"] = value
        with pytest.raises(ProjectFileError, match="rules.max_pause_s"):
            parse_document("project", data)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(data, _schema("project"))


def test_overlay_rule_defaults_and_overrides() -> None:
    data = {"schema": "yap-studio/project@1", "name": "sample",
            "takes": [{"id": "t1", "path": "take.mov"}],
            "rules": {"overlay_min_s": 2.0, "overlay_max_s": 5.0}}
    project = parse_document("project", data)
    assert RULE_DEFAULTS["overlay_min_s"] == 1.0
    assert RULE_DEFAULTS["overlay_max_s"] == 6.0
    assert project.rules["overlay_min_s"] == 2.0
    assert project.rules["overlay_max_s"] == 5.0
    jsonschema.validate(project.to_dict(), _schema("project"))
    for value in (-0.1, True):
        data["rules"]["overlay_min_s"] = value
        with pytest.raises(ProjectFileError, match="rules.overlay_min_s"):
            parse_document("project", data)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(data, _schema("project"))


def test_anchor_defaults_and_normalized_guards(tmp_path: Path) -> None:
    edit = parse_document("edit", _example("edit"))
    assert edit.segments[0].from_.edge == "start"
    assert edit.segments[0].to.edge == "end"
    assert Anchor(word="t1:1", text="OK!").matches("okay")
    assert not Anchor(word="t1:1", text="next").matches("previous")
    assert Anchor(time=1).to_dict() == {"time": 1}
    lane = _example("timeline")
    lane["elements"][0]["lane"] = "broll"
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(lane), encoding="utf-8")
    with pytest.raises(ProjectFileError, match=r"elements\[0\].lane"):
        load_timeline(path)


def test_corrupt_missing_and_non_object_files(tmp_path: Path) -> None:
    target = tmp_path / "project.json"
    for payload in ("[]", "{", "null"):
        target.write_text(payload, encoding="utf-8")
        with pytest.raises(ProjectFileError):
            load_project(target)
    target.unlink()
    with pytest.raises(ProjectFileError):
        load_project(target)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return synth.calib_source(tmp_path / "source.mov", seconds=0.4, size="64x64")


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


def test_init_real_media_relative_paths_and_refusal(source: Path, tmp_path: Path) -> None:
    original = receipt.sha256_file(source)
    project = init_project(tmp_path, takes=[source])
    assert project.style == "preset:clean_minimal"
    assert project.takes[0].path == "source.mov"
    assert project.takes[0].id == "t1"
    assert load_project(tmp_path / "project.json") == project
    jsonschema.validate(json.loads((tmp_path / "project.json").read_bytes()), _schema("project"))
    with pytest.raises(ProjectRefusal) as caught:
        init_project(tmp_path, takes=[source])
    assert caught.value.code == "project_exists"
    assert receipt.sha256_file(source) == original


def test_voiceover_init_and_pickup(source: Path, tmp_path: Path) -> None:
    memo = tmp_path / "memo.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", str(memo)],
                   check=True)
    original = receipt.sha256_file(memo)
    directory = tmp_path / "voice-project"
    project = init_project(directory, takes=[], voiceovers=[memo])
    assert project.takes[0].voiceover is True
    standin = directory / project.takes[0].path
    assert standin.is_file()
    from yap_studio import ffmpeg
    info = ffmpeg.probe(standin)
    assert (info.width, info.height, info.has_audio) == (1080, 1920, True)
    assert (info.color_primaries, info.color_transfer, info.color_space, info.color_range) == (
        "bt709", "bt709", "bt709", "tv")
    updated = init_project(directory, takes=[], voiceovers=[memo])
    assert [take.id for take in updated.takes] == ["t1", "t2"]
    assert load_project(directory / "project.json") == updated
    assert receipt.sha256_file(memo) == original


def test_voiceover_init_rejects_mixed_takes(source: Path, tmp_path: Path) -> None:
    memo = tmp_path / "memo.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", str(memo)],
                   check=True)
    directory = tmp_path / "mixed-project"
    with pytest.raises(InputError, match="cannot mix --take and --voiceover"):
        init_project(directory, takes=[source], voiceovers=[memo])
    assert not directory.exists()


def test_voiceover_cli_accepts_repeatable_memos(source: Path, tmp_path: Path) -> None:
    memo = tmp_path / "memo.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", str(memo)],
                   check=True)
    root = tmp_path / "voice-project"
    assert cli.main(["init", str(root), "--voiceover", str(memo),
                     "--voiceover", str(memo)]) == 0
    assert [take.voiceover for take in load_project(root / "project.json").takes] == [True, True]


def test_init_absolute_external_takes_and_preset(source: Path, tmp_path: Path) -> None:
    directory = tmp_path / "New Story"
    project = init_project(directory, takes=[source, source], style="preset:electric")
    assert project.name == "new-story"
    assert [take.id for take in project.takes] == ["t1", "t2"]
    assert project.takes[0].path == str(source.resolve())
    assert project.style == "preset:electric"
    assert not (directory / "styles").exists()


@pytest.mark.parametrize("selection", ["file", "library", "default"])
def test_init_copies_original_style_snapshot(
    selection: str, source: Path, tmp_path: Path,
) -> None:
    style = tmp_path / "portable.json"
    original = b'{"schema_version": 1, "name": "Portable"}\n'
    style.write_bytes(original)
    spec = str(style)
    if selection in {"library", "default"}:
        entry = add(style, make_default=selection == "default")
        original = entry.path.read_bytes()
        spec = entry.id if selection == "library" else None
    directory = tmp_path / "project"
    project = init_project(directory, takes=[source], style=spec)
    assert project.style == "styles/portable.json"
    snapshot = directory / project.style
    assert snapshot.read_bytes() == original
    style.write_text('{"schema_version": 1, "name": "Changed"}', encoding="utf-8")
    if selection in {"library", "default"}:
        add(style, replace=True)
    assert snapshot.read_bytes() == original


def test_init_library_default_preset(source: Path, tmp_path: Path) -> None:
    set_default("preset:warm_editorial")
    assert init_project(tmp_path / "project", takes=[source]).style == "preset:warm_editorial"


def test_init_input_errors_do_not_write_project(tmp_path: Path) -> None:
    target = tmp_path / "new-project"
    with pytest.raises(InputError):
        init_project(target, takes=[tmp_path / "missing.mov"])
    with pytest.raises(ProjectFileError):
        init_project(target, takes=[])
    silent = tmp_path / "silent.mov"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=32x32:d=0.2",
        "-an", "-c:v", "libx264", str(silent),
    ], check=True, timeout=30)
    with pytest.raises(InputError, match="both picture and audio"):
        init_project(target, takes=[silent])
    assert not (target / "project.json").exists()


def test_init_cli_and_status_exit_codes(
    source: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "cli-project"
    args = ["init", str(target), "--take", str(source), "--name", "named", "--json"]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "named"
    assert cli.main(args) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "project_exists"
    assert cli.main(["status", str(target), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "yap-studio/status@1"
    assert result["stages"][0]["state"] == "missing"
    assert result["next"].startswith("yap-studio ingest ")
    assert cli.main(["status", str(target)]) == 0
    plain = capsys.readouterr().out
    assert "ingest: missing, no receipt" in plain
    assert "\u2014" not in plain and "\u2013" not in plain
    assert cli.main(["init", str(tmp_path / "bad"), "--take", "missing.mov", "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["suggested_action"]
    assert cli.main(["status", str(tmp_path / "missing"), "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_parameter"


def test_cli_maps_compare_and_swap_conflict(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    def conflict(*args: Any, **kwargs: Any) -> None:
        raise ConflictError(tmp_path / "project.json")

    monkeypatch.setattr("yap_studio.project.init_project", conflict)
    assert cli.main(["init", str(tmp_path), "--take", "source.mov", "--json"]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "base_changed"


def test_dead_pid_lock_recovery_and_timeout(tmp_path: Path) -> None:
    directory = tmp_path / ".yap"
    directory.mkdir()
    lock_file = directory / "lock"
    lock_file.write_text('{"pid": 2147483647, "host": "old", "since": "old"}', encoding="utf-8")
    with project_lock(tmp_path):
        assert json.loads(lock_file.read_bytes())["pid"] == os.getpid()
        with ThreadPoolExecutor(max_workers=1) as pool:
            def blocked() -> None:
                with project_lock(tmp_path, timeout_s=0.05):
                    pytest.fail("lock should be held")

            with pytest.raises(YapError) as caught:
                pool.submit(blocked).result(timeout=5)
            assert caught.value.code == "project_locked"
    with project_lock(tmp_path, timeout_s=0):
        pass


def test_two_writers_conflict_and_retry_preserves_both_elements(tmp_path: Path) -> None:
    lane = tmp_path / "timeline" / "agent.json"
    initial = {"schema": "yap-studio/timeline@1", "words_sha256": {}, "elements": []}
    with paths.operation(), project_lock(tmp_path):
        base = write_json_cas(lane, initial, base_sha256=None)
    barrier = threading.Barrier(2)

    def writer(element_id: str) -> bool:
        barrier.wait(timeout=5)
        data = {**initial, "elements": [{"id": element_id}]}
        with paths.operation(), project_lock(tmp_path):
            try:
                write_json_cas(lane, data, base_sha256=base)
                return False
            except ConflictError:
                refreshed = json.loads(lane.read_bytes())
                refreshed["elements"].append({"id": element_id})
                write_json_cas(lane, refreshed, base_sha256=receipt.sha256_file(lane))
                return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(writer, value) for value in ("first", "second")]
        results = [future.result(timeout=10) for future in futures]
    assert sorted(results) == [False, True]
    assert {item["id"] for item in json.loads(lane.read_bytes())["elements"]} == {"first", "second"}


def test_cas_missing_base_and_path_guards(tmp_path: Path) -> None:
    target = tmp_path / "lane.json"
    with paths.operation(), project_lock(tmp_path):
        digest = write_json_cas(target, {"value": 1}, base_sha256=None)
        assert digest == receipt.sha256_file(target)
        with pytest.raises(ConflictError):
            write_json_cas(target, {"value": 2}, base_sha256=None)
        alias = tmp_path / "alias.json"
        alias.symlink_to(target)
        with pytest.raises(OutputPathError):
            write_json_cas(alias, {}, base_sha256=digest)
    with paths.operation(), project_lock(tmp_path):
        paths.validate_input(target)
        with pytest.raises(OutputPathError, match="input file"):
            write_json_cas(target, {}, base_sha256=digest)


@pytest.mark.parametrize("policy", [None, {"mode": "preserve"},
                                     {"mode": "loudnorm", "integrated_lufs": -16,
                                      "true_peak_dbtp": -2, "lra_lu": 8}])
def test_audio_policy_defaults_and_round_trip(policy: dict | None) -> None:
    data = _example("project")
    if policy is not None:
        data["audio"] = policy
    project = parse_document("project", data)
    assert project.audio["mode"] == (policy["mode"] if policy else "loudnorm")
    if policy is None:
        assert project.audio == {"mode": "loudnorm", "integrated_lufs": -14,
                                 "true_peak_dbtp": -1.7, "lra_lu": 11}
    else:
        assert project.audio == policy
    jsonschema.validate(project.to_dict(), _schema("project"))
    assert parse_document("project", project.to_dict()).audio == project.audio


@pytest.mark.parametrize("policy", [{"mode": "other"}, {"mode": "loudnorm"},
                                     {"mode": "preserve", "integrated_lufs": -14},
                                     {"mode": "loudnorm", "integrated_lufs": -80,
                                      "true_peak_dbtp": -2, "lra_lu": 8}])
def test_audio_policy_rejects_invalid_shapes(policy: dict) -> None:
    data = {**_example("project"), "audio": policy}
    with pytest.raises(ProjectFileError):
        parse_document("project", data)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, _schema("project"))
