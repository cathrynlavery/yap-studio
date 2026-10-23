"""Creator taste validation, audit storage, snapshots, and style application."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from yap_studio import cards, paths
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.style import config_root, loader, resolve
from yap_studio.taste import (
    PROPOSALS_SCHEMA,
    SNAPSHOT_SCHEMA,
    TASTE_SCHEMA,
    Proposals,
    Taste,
    TasteError,
    TasteRefusal,
    add_proposals,
    apply_taste,
    approve,
    approve_proposal,
    load_proposals,
    load_taste,
    proposals_path,
    reject,
    reject_proposal,
    revoke,
    rule_id,
    store,
    taste_path,
    validate_body,
    validate_style_field,
)

NOW = "2026-01-02T03:04:05Z"
LATER = "2026-02-03T04:05:06Z"


@pytest.mark.parametrize("mode", ["box", "band"])
@pytest.mark.parametrize("override", ["approved", "explicit"])
def test_disabling_ground_restores_source_color_across_taste_resolutions(
    tmp_path: Path, mode: str, override: str,
) -> None:
    source = tmp_path / "style.json"
    source.write_text(json.dumps({"schema_version": 1, "captions": {
        "text_color": "#FF0000", "background_mode": mode, "background_color": "#000000",
    }, "brand": {"captions_font": "Gelasio", "accent_color": "#123456"}}))
    style = resolve(source)
    before = copy.deepcopy(style.data)
    assert style.data["captions"]["text_color"] == "#FFFFFF"
    snapshot = Taste().snapshot()
    explicit = {"captions.background_mode": "none"}
    if override == "approved":
        snapshot = replace(snapshot, rules=({"id": "r-123456789abc", **_body(
            "none", field="captions.background_mode",
        )},))
        explicit = None
    disabled = apply_taste(style, snapshot, explicit=explicit).style
    assert disabled.data["captions"]["text_color"] == "#FF0000"
    assert disabled.data["captions"]["font"] == "Gelasio"
    assert disabled.data["captions"]["highlight_color"] == "#123456"
    assert disabled.unbranded_data["captions"]["font"] == "Inter"
    enabled = apply_taste(disabled, Taste().snapshot(), explicit={
        "captions.background_mode": mode, "captions.text_color": "#00FF00",
    }).style
    assert enabled.data["captions"]["text_color"] == "#FFFFFF"
    again = apply_taste(enabled, Taste().snapshot(), explicit={
        "captions.background_mode": "none",
    }).style
    assert again.data["captions"]["text_color"] == "#00FF00"
    assert enabled.unbranded_data["captions"]["text_color"] == "#00FF00"
    assert style.data == before


@pytest.fixture(autouse=True)
def isolated_taste(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(store, "_now", lambda: NOW)


def _source(comment_id: str = "c1") -> dict[str, Any]:
    return {
        "type": "comment",
        "media_sha256": "a" * 64,
        "comment_id": comment_id,
        "label": "fix",
        "start_frame": 2,
        "end_frame": 5,
        "quote": "Fewer words.",
    }


def _measurement() -> dict[str, Any]:
    return {
        "type": "measurement",
        "media_sha256": "b" * 64,
        "media_name": "clip.mp4",
        "measure": "captions.face_overlap",
        "frames": [0, 2, 8],
    }


def _body(
    value: Any = 3,
    *,
    condition: dict | None = None,
    field: str = "captions.max_words_per_phrase",
    sentence: str | None = None,
) -> dict:
    section = field.split(".")[0]
    return {
        "scope": {"hook": "cards", "graphics": "cards", "broll": "broll", "music": "music"}.get(
            section, "captions"
        ),
        "condition": condition,
        "sentence": sentence or f"Use {value} for {field}.",
        "machine": {"kind": "style_field", "field": field, "value": value},
        "sources": [_source()],
        "context": "A title over a screen recording.",
    }


def _condition(*tags: str, text: str = "On screen recordings.") -> dict:
    return {"text": text, "tags": list(tags)}


def _file(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _approved(body: dict | None = None) -> str:
    identity = add_proposals(body or _body())[0].id
    approve(identity)
    return identity


def _set(data: dict, dotted: str, value: Any) -> dict:
    result = copy.deepcopy(data)
    parts = dotted.split(".")
    target = result
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    last = parts[-1]
    target[int(last) if isinstance(target, list) else last] = value
    return result


def test_approve_proposal_edit_has_one_save_and_one_write_per_file(monkeypatch) -> None:
    original = _body()
    edited = {**original, "sentence": "Keep caption phrases short."}
    written, saves = [], []
    real_write, real_save = store._write, store._save

    def write(target, data):
        written.append(target)
        return real_write(target, data)

    def save(*args):
        saves.append(args)
        return real_save(*args)

    monkeypatch.setattr(store, "_write", write)
    monkeypatch.setattr(store, "_save", save)
    result = approve_proposal(edited, reject=original)
    assert result.rule["id"] == rule_id(edited) and result.revision == 1
    assert written == [taste_path(), proposals_path()] and len(saves) == 1
    assert load_proposals().proposals[0]["id"] == rule_id(original)
    assert load_proposals().proposals[0]["status"] == "rejected"
    assert load_taste().history[0]["rule"] == edited


def test_approve_proposal_merges_reopened_evidence_and_supersedes(monkeypatch) -> None:
    older = approve_proposal(_body(2))
    body = _body()
    reject_proposal(body)
    monkeypatch.setattr(store, "_now", lambda: LATER)
    result = approve_proposal({**body, "sources": [_source("c2"), _source()]})
    assert result.superseded_ids == (older.rule["id"],) and result.revision == 2
    assert result.rule["sources"] == [_source(), _source("c2")]
    assert result.rule["proposed_at"] == NOW and result.rule["approved_at"] == LATER
    assert load_proposals().proposals == ()
    assert [entry["action"] for entry in load_taste().history] == [
        "approved", "approved", "superseded",
    ]


@pytest.mark.parametrize("operation", ["approve", "reject"])
def test_new_helpers_merge_pending_sources_at_most_fifty(operation: str) -> None:
    body = {**_body(), "sources": [_source(f"c{i}") for i in range(1, 50)]}
    add_proposals(body)
    incoming = {**_body(), "sources": [_source("c1"), _source("c50"), _source("c51")]}
    if operation == "approve":
        record = approve_proposal(incoming).rule
    else:
        record = reject_proposal(incoming)
    assert record["sources"] == [*body["sources"], _source("c50")]


def test_reject_proposal_one_write_and_repeat_writes_nothing(monkeypatch) -> None:
    written = []
    real_write = store._write

    def write(target, data):
        written.append(target)
        return real_write(target, data)

    monkeypatch.setattr(store, "_write", write)
    before = load_taste().snapshot()
    rejected = reject_proposal(_body())
    assert rejected["status"] == "rejected" and rejected["decided_at"] == NOW
    assert written == [proposals_path()]
    assert load_taste().snapshot() == before and not taste_path().exists()
    assert reject_proposal(_body()) == rejected
    assert written == [proposals_path()]


@pytest.mark.parametrize("operation", ["reject", "approve_with_reject"])
def test_new_helpers_refuse_rejecting_approved_without_any_write(monkeypatch, operation) -> None:
    approve_proposal(_body())
    before = load_taste(), load_proposals(), taste_path().read_bytes()
    written = []
    monkeypatch.setattr(store, "_write", lambda *args: written.append(args))
    with pytest.raises(TasteRefusal) as caught:
        if operation == "reject":
            reject_proposal(_body())
        else:
            approve_proposal(_body(4), reject=_body())
    assert caught.value.code == "taste_already_approved"
    assert "revoke" in caught.value.suggested_action
    assert written == []
    assert (load_taste(), load_proposals(), taste_path().read_bytes()) == before


def test_approve_proposal_already_rejected_original_keeps_decision_time(monkeypatch) -> None:
    original = reject_proposal(_body())
    monkeypatch.setattr(store, "_now", lambda: LATER)
    result = approve_proposal({**_body(), "sentence": "Use shorter phrases."}, reject=_body())
    assert result.rule["approved_at"] == LATER
    assert load_proposals().proposals == (original,)


def test_approve_proposal_already_approved_returns_existing_without_writes(monkeypatch) -> None:
    body = _body()
    existing = approve_proposal(body)
    approve_proposal(_body(4, condition=_condition("screen")))
    before = load_taste(), taste_path().read_bytes(), load_proposals()
    written = []
    monkeypatch.setattr(store, "_write", lambda *args: written.append(args))
    result = approve_proposal({**body, "sources": [_source("c2")], "context": "New evidence."})
    assert result.rule == existing.rule
    assert result.superseded_ids == () and result.revision == existing.rule["approved_revision"]
    assert result.revision == 1 and load_taste().revision == 2
    assert written == []
    assert (load_taste(), taste_path().read_bytes(), load_proposals()) == before
    result.rule["sources"][0]["quote"] = "Changed returned copy."
    assert load_taste() == before[0]


def test_approve_proposal_already_approved_rejects_original_with_one_write(monkeypatch) -> None:
    body = _body()
    existing = approve_proposal(body)
    approve_proposal(_body(4, condition=_condition("screen")))
    original = {**body, "sentence": "Original wording."}
    before = load_taste(), taste_path().read_bytes()
    written, saves = [], []
    real_write, real_save = store._write, store._save

    def write(target, data):
        written.append(target)
        return real_write(target, data)

    def save(*args):
        saves.append(args)
        return real_save(*args)

    monkeypatch.setattr(store, "_write", write)
    monkeypatch.setattr(store, "_save", save)
    result = approve_proposal({**body, "sources": [_source("c2")]}, reject=original)
    assert result == existing
    assert result.revision == 1 and load_taste().revision == 2
    assert written == [proposals_path()] and len(saves) == 1
    rejected = load_proposals().proposals[0]
    assert rejected["id"] == rule_id(original) and rejected["status"] == "rejected"
    assert rejected["decided_at"] == NOW
    assert (load_taste(), taste_path().read_bytes()) == before
    assert approve_proposal(body, reject=original) == existing
    assert written == [proposals_path()] and len(saves) == 2


def test_rule_id_uses_exact_canonical_content() -> None:
    body = _body(sentence="Keep café captions short.")
    content = {key: body[key] for key in ("scope", "condition", "sentence", "machine")}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected = "r-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    assert rule_id(body) == expected
    assert rule_id(dict(reversed(list(body.items())))) == expected
    changed_evidence = {**body, "sources": [_measurement()], "context": "Another recording."}
    assert rule_id(changed_evidence) == expected
    assert rule_id(_body(4)) != expected
    assert rule_id({"scope": "pacing", "sentence": "Hold longer."}) == rule_id(
        {"scope": "pacing", "sentence": "Hold longer.", "machine": None, "condition": None}
    )


@pytest.mark.parametrize(
    ("field", "value", "location", "detail"),
    [
        ("scope", "video", "scope", "choose one of"),
        ("sentence", "", "sentence", "length"),
        ("sentence", "x" * 281, "sentence", "length"),
        ("sentence", " Words", "sentence", "stripped"),
        ("sentence", "Two\nlines", "sentence", "no newlines"),
        ("sentence", 3, "sentence", "string"),
        ("condition", {"text": "Only here.", "tags": ["Bad"]}, "condition.tags[0]", "match"),
        ("condition", _condition("a", "a"), "condition.tags[1]", "unique"),
        ("condition", _condition(*[f"t{i}" for i in range(9)]), "condition.tags", "at most 8"),
        ("condition", {"text": "Only here.", "tags": "a"}, "condition.tags", "list"),
        ("condition", {"text": "Only here.", "tags": ["a" * 41]}, "condition.tags[0]", "match"),
        ("condition", {"text": "", "tags": []}, "condition.text", "length"),
        ("condition", {"text": "x" * 201, "tags": []}, "condition.text", "length"),
        ("condition", {"text": " Only here.", "tags": []}, "condition.text", "stripped"),
        ("condition", {"text": "Two\rlines", "tags": []}, "condition.text", "newlines"),
        (
            "condition",
            {"text": "Only here.", "tags": [], "extra": 1},
            "condition.extra",
            "unknown field",
        ),
        ("machine.field", "internal.fps", "machine.field", "engine constraints"),
        ("machine.field", "schema_version", "machine.field", "identity"),
        ("machine.field", "name", "machine.field", "identity"),
        ("machine.field", "captions.unknown", "machine.field", "unknown style field"),
        ("machine.field", "captions", "machine.field", "unknown style field"),
        ("machine.field", "cut.words", "machine.field", "unknown style field"),
        ("machine.value", 9, "machine.value", "between 1 and 8"),
        ("machine.value", True, "machine.value", "integer"),
        (
            "machine",
            {"kind": "style_field", "field": "captions.position", "value": "top"},
            "machine.value",
            "choose one of",
        ),
        (
            "machine",
            {"kind": "style_field", "field": "captions.text_color", "value": "red"},
            "machine.value",
            "color",
        ),
        ("scope", "cards", "scope", "must be 'captions'"),
        (
            "machine",
            {"kind": "filler_word", "word": "So", "position": "leading"},
            "machine.word",
            "match",
        ),
        (
            "machine",
            {"kind": "filler_word", "word": "you  know", "position": "leading"},
            "machine.word",
            "double",
        ),
        (
            "machine",
            {"kind": "filler_word", "word": "so ", "position": "leading"},
            "machine.word",
            "trailing",
        ),
        (
            "machine",
            {"kind": "filler_word", "word": "so", "position": "last"},
            "machine.position",
            "choose one of",
        ),
        (
            "machine",
            {"kind": "filler_word", "word": "so", "position": "leading"},
            "scope",
            "must be 'cut'",
        ),
        (
            "machine",
            {"kind": "placement", "layer": "face", "rule": "clear_of_face"},
            "machine.layer",
            "choose one of",
        ),
        (
            "machine",
            {"kind": "placement", "layer": "captions", "rule": "top"},
            "machine.rule",
            "choose one of",
        ),
        (
            "machine",
            {"kind": "placement", "layer": "cards", "rule": "clear_of_face"},
            "scope",
            "must be 'cards'",
        ),
        ("machine.kind", "other", "machine.kind", "choose one of"),
        ("machine.extra", 1, "machine.extra", "unknown field"),
        ("sources.0.media_sha256", "A" * 64, "sources[0].media_sha256", "match"),
        ("sources.0.media_sha256", "a" * 63, "sources[0].media_sha256", "match"),
        ("sources", [], "sources", "1 to 50"),
        ("sources", [_source()] * 51, "sources", "1 to 50"),
        ("sources.0.comment_id", "", "sources[0].comment_id", "length"),
        ("sources.0.comment_id", "x" * 81, "sources[0].comment_id", "length"),
        ("sources.0.label", "other", "sources[0].label", "choose one of"),
        ("sources.0.start_frame", -1, "sources[0].start_frame", "integer"),
        ("sources.0.start_frame", True, "sources[0].start_frame", "integer"),
        ("sources.0.end_frame", 1, "sources[0].end_frame", "at least 2"),
        ("sources.0.quote", "x" * 501, "sources[0].quote", "length"),
        ("sources.0.extra", 1, "sources[0].extra", "unknown field"),
        ("sources.0.type", "other", "sources[0].type", "choose one of"),
        ("context", "x" * 501, "context", "length"),
        ("extra", True, "extra", "unknown field"),
    ],
)
def test_body_refusals_have_exact_field_locations(
    field: str, value: Any, location: str, detail: str
) -> None:
    with pytest.raises(TasteError) as caught:
        add_proposals(_set(_body(), field, value))
    assert caught.value.parameter == f"proposal.{location}"
    assert detail in caught.value.detail
    assert caught.value.parameter in caught.value.suggested_action
    assert not config_root().exists()


@pytest.mark.parametrize(
    ("field", "value", "location"),
    [
        ("frames", [2, 0], "frames[1]"),
        ("frames", [0, 0], "frames[1]"),
        ("frames", [-1], "frames[0]"),
        ("frames", [True], "frames[0]"),
        ("frames", [], "frames"),
        ("frames", list(range(10001)), "frames"),
        ("media_name", "folder/clip.mp4", "media_name"),
        ("media_name", "folder\\clip.mp4", "media_name"),
        ("media_name", "", "media_name"),
        ("media_name", "x" * 256, "media_name"),
        ("measure", "Bad measure", "measure"),
        ("media_sha256", "b" * 65, "media_sha256"),
        ("extra", 1, "extra"),
    ],
)
def test_measurement_validation(field: str, value: Any, location: str) -> None:
    body = _body()
    body["sources"] = [{**_measurement(), field: value}]
    with pytest.raises(TasteError) as caught:
        add_proposals(body)
    assert caught.value.parameter == f"proposal.sources[0].{location}"


def test_valid_machine_forms_and_nullable_fields() -> None:
    for machine, scope in [
        ({"kind": "filler_word", "word": "you know", "position": "anywhere"}, "cut"),
        ({"kind": "placement", "layer": "title", "rule": "clear_of_face"}, "cards"),
        ({"kind": "placement", "layer": "cards", "rule": "clear_of_face"}, "cards"),
        ({"kind": "placement", "layer": "captions", "rule": "clear_of_face"}, "captions"),
        (None, "pacing"),
    ]:
        assert validate_body({**_body(), "scope": scope, "machine": machine})["machine"] == machine
    body = {"scope": "pacing", "sentence": "Hold longer.", "sources": [_measurement()]}
    identity = add_proposals(body)[0].id
    proposal = load_proposals().pending[0]
    assert proposal["id"] == identity
    assert proposal["condition"] is proposal["machine"] is proposal["context"] is None
    null_source = dict.fromkeys(("label", "start_frame", "end_frame", "quote"))
    validate_body({**_body(), "sources": [{**_source(), **null_source}]})


def test_style_fields_come_from_current_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    schema = copy.deepcopy(loader._schema())
    schema["properties"]["captions"]["properties"]["new_field"] = {
        "type": "integer",
        "minimum": 0,
        "maximum": 2,
    }
    monkeypatch.setattr(loader, "_schema", lambda: schema)
    assert add_proposals(_body(1, field="captions.new_field"))[0].status == "added"
    with pytest.raises(TasteError) as caught:
        add_proposals(_body(3, field="captions.new_field"))
    assert caught.value.parameter == "proposal.machine.value"


def test_style_field_validates_opt_in_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    for kind in cards.OPT_IN_KINDS:
        assert validate_style_field("graphics.card_kinds_on", [kind], "rule") == "graphics"
    assert validate_style_field(
        "graphics.card_kinds_on", ["hf:char-slam-explode"], "rule") == "graphics"
    with pytest.raises(TasteError, match="hf:char-slam-explode") as caught:
        validate_style_field("graphics.card_kinds_on", ["hf:unknown"], "rule")
    assert caught.value.parameter == "rule.value[0]"
    monkeypatch.setattr(cards, "OPT_IN_KINDS", ())
    monkeypatch.setattr(registry_cards, "kinds", lambda: ())
    with pytest.raises(TasteError, match="leave this list empty") as caught:
        validate_style_field("graphics.card_kinds_on", ["unknown"], "rule")
    assert caught.value.parameter == "rule.value[0]"
    monkeypatch.setattr(cards, "OPT_IN_KINDS", ("fake_kind",))
    assert validate_style_field("graphics.card_kinds_on", ["fake_kind"], "rule") == "graphics"
    with pytest.raises(TasteError, match="choose one of fake_kind") as caught:
        validate_style_field("graphics.card_kinds_on", ["unknown"], "rule")
    assert caught.value.parameter == "rule.value[0]"


def test_missing_files_are_empty_and_reads_create_nothing() -> None:
    assert load_taste() == Taste()
    assert load_proposals() == Proposals()
    assert load_taste().snapshot().rules == ()
    assert taste_path() == config_root() / "taste.json"
    assert proposals_path() == config_root() / "taste-proposals.json"
    assert not config_root().exists()


@pytest.mark.parametrize("content", ["[]", "null", "{", "\udcff"])
def test_unreadable_files_raise_taste_errors(content: str) -> None:
    config_root().mkdir()
    taste_path().write_bytes(content.encode("utf-8", errors="surrogateescape"))
    with pytest.raises(TasteError):
        load_taste()
    proposals_path().write_bytes(content.encode("utf-8", errors="surrogateescape"))
    taste_path().unlink()
    with pytest.raises(TasteError):
        load_proposals()


@pytest.mark.parametrize(
    ("field", "value", "location"),
    [
        ("schema", "other", "taste.schema"),
        ("extra", 1, "taste.extra"),
        ("revision", -1, "taste.revision"),
        ("revision", True, "taste.revision"),
        ("updated_at", "2026-02-30T00:00:00Z", "taste.updated_at"),
        ("rules.0.proposed_at", "2026-01-02T03:04:05+00:00", "taste.rules[0].proposed_at"),
        ("rules.0.approved_at", "2026-01-02T03:04:05.123Z", "taste.rules[0].approved_at"),
        ("rules.0.id", "r-000000000000", "taste.rules[0].id"),
        ("rules.0.status", "proposed", "taste.rules[0].status"),
        ("rules.0.approved_revision", 0, "taste.rules[0].approved_revision"),
        ("rules.0.approved_revision", 2, "taste.rules[0].approved_revision"),
        ("rules.0.ended_at", NOW, "taste.rules[0].ended_at"),
        ("rules.0.ended_revision", 1, "taste.rules[0].ended_revision"),
        ("rules.0.superseded_by", "r-000000000000", "taste.rules[0].superseded_by"),
    ],
)
def test_taste_invariants_on_load(field: str, value: Any, location: str) -> None:
    _approved()
    _file(taste_path(), _set(_json(taste_path()), field, value))
    with pytest.raises(TasteError) as caught:
        load_taste()
    assert caught.value.parameter == location


@pytest.mark.parametrize(
    ("changes", "location"),
    [
        ({"ended_at": None}, "ended_at"),
        ({"ended_revision": None}, "ended_revision"),
        ({"ended_revision": 1}, "ended_revision"),
        ({"ended_revision": 3}, "ended_revision"),
        ({"superseded_by": "r-000000000000"}, "superseded_by"),
        ({"status": "superseded", "superseded_by": None}, "superseded_by"),
        ({"status": "superseded", "superseded_by": "r-000000000000"}, "superseded_by"),
    ],
)
def test_ended_rule_invariants_on_load(changes: dict, location: str) -> None:
    identity = _approved()
    revoke(identity)
    data = _json(taste_path())
    data["rules"][0].update(changes)
    _file(taste_path(), data)
    with pytest.raises(TasteError) as caught:
        load_taste()
    assert caught.value.parameter == f"taste.rules[0].{location}"


def test_duplicate_ids_and_self_supersession_are_refused() -> None:
    identity = _approved()
    data = _json(taste_path())
    _file(taste_path(), {**data, "rules": data["rules"] * 2})
    with pytest.raises(TasteError) as caught:
        load_taste()
    assert caught.value.parameter == "taste.rules[1].id"
    data["revision"] = 2
    data["rules"][0].update(
        status="superseded", ended_at=NOW, ended_revision=2, superseded_by=identity
    )
    _file(taste_path(), data)
    with pytest.raises(TasteError) as caught:
        load_taste()
    assert caught.value.parameter == "taste.rules[0].superseded_by"
    taste_path().unlink()
    add_proposals(_body())
    data = _json(proposals_path())
    _file(proposals_path(), {**data, "proposals": data["proposals"] * 2})
    with pytest.raises(TasteError) as caught:
        load_proposals()
    assert caught.value.parameter == "proposals.proposals[1].id"


@pytest.mark.parametrize(
    ("changes", "location"),
    [
        ({"decided_at": NOW}, "decided_at"),
        ({"status": "rejected"}, "decided_at"),
        ({"status": "approved"}, "status"),
        ({"proposed_at": "2026-13-02T03:04:05Z"}, "proposed_at"),
        ({"decided_at": "yesterday"}, "decided_at"),
        ({"extra": 1}, "extra"),
    ],
)
def test_proposal_invariants_on_load(changes: dict, location: str) -> None:
    add_proposals(_body())
    data = _json(proposals_path())
    data["proposals"][0].update(changes)
    _file(proposals_path(), data)
    with pytest.raises(TasteError) as caught:
        load_proposals()
    assert caught.value.parameter == f"proposals.proposals[0].{location}"


def test_add_merge_approve_and_reject_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _body()
    first = add_proposals(body)[0]
    assert first.status == "added"
    other = add_proposals(_body(4))[0]
    monkeypatch.setattr(store, "_now", lambda: LATER)
    merged = add_proposals({**body, "sources": [_source("c2"), _source(), _source("c2")]})[0]
    assert merged.id == first.id and merged.status == "merged"
    proposals = load_proposals().pending
    assert [item["id"] for item in proposals] == [first.id, other.id]
    assert proposals[0]["sources"] == [_source(), _source("c2")]
    assert proposals[0]["proposed_at"] == NOW
    approve(first.id)
    before = taste_path().read_bytes(), proposals_path().read_bytes()
    assert add_proposals(body)[0].status == "already_approved"
    assert (taste_path().read_bytes(), proposals_path().read_bytes()) == before
    rejected = reject(other.id)
    assert rejected["status"] == "rejected" and rejected["decided_at"] == LATER
    assert load_taste().revision == 1
    assert add_proposals(_body(4))[0].status == "previously_rejected"
    assert load_proposals().pending == ()


def test_full_batch_is_validated_before_any_write(tmp_path: Path) -> None:
    data = {
        "schema": PROPOSALS_SCHEMA,
        "proposals": [_body(), _set(_body(4), "machine.value", 100)],
    }
    source = _file(tmp_path / "input.json", data)
    with pytest.raises(TasteError) as caught:
        add_proposals(source)
    assert caught.value.parameter == "proposals[1].machine.value"
    assert not config_root().exists()
    data["proposals"][1] = _body(4)
    assert [item.status for item in add_proposals(data)] == ["added", "added"]


@pytest.mark.parametrize(
    ("changes", "location"),
    [
        ({"id": "r-000000000000"}, "proposal.id"),
        ({"status": "rejected"}, "proposal.status"),
        ({"proposed_at": NOW}, "proposal.proposed_at"),
    ],
)
def test_input_metadata_is_restricted(changes: dict, location: str) -> None:
    with pytest.raises(TasteError) as caught:
        add_proposals({**_body(), **changes})
    assert caught.value.parameter == location
    assert not config_root().exists()


def test_input_identity_and_proposed_status_are_accepted() -> None:
    body = _body()
    result = add_proposals({**body, "id": rule_id(body), "status": "proposed"})
    assert result[0].status == "added"


def test_merging_caps_sources_and_keeps_other_batch_proposals() -> None:
    body = {**_body(), "sources": [_source(f"c{i}") for i in range(50)]}
    add_proposals(body)
    result = add_proposals({
        "schema": PROPOSALS_SCHEMA,
        "proposals": [_body(4), {**_body(), "sources": [_source("new")]}],
    })
    assert [item.status for item in result] == ["added", "merged"]
    assert load_proposals().proposals[0]["sources"] == body["sources"]
    assert load_proposals().proposals[1]["id"] == rule_id(_body(4))


def test_approve_moves_proposal_between_files_and_advances_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = add_proposals(_body())[0].id
    assert _json(proposals_path())["proposals"][0]["status"] == "proposed"
    assert not taste_path().exists()
    monkeypatch.setattr(store, "_now", lambda: LATER)
    result = approve(identity)
    assert result.revision == 1 and result.superseded_ids == ()
    assert result.rule["status"] == "approved"
    taste_data = _json(taste_path())
    assert taste_data == {
        "schema": TASTE_SCHEMA,
        "revision": 1,
        "updated_at": LATER,
        "rules": [result.rule],
        "history": [
            {
                "revision": 1,
                "at": LATER,
                "action": "approved",
                "rule_id": identity,
                "by": None,
                "rule": validate_body(_body()),
            }
        ],
    }
    assert result.rule["proposed_at"] == NOW
    assert result.rule["approved_at"] == LATER
    assert result.rule["approved_revision"] == 1
    assert (
        result.rule["ended_at"]
        is result.rule["ended_revision"]
        is result.rule["superseded_by"]
        is None
    )
    assert _json(proposals_path()) == {"schema": PROPOSALS_SCHEMA, "proposals": []}
    revoked = revoke(identity)
    assert revoked["status"] == "revoked" and revoked["ended_at"] == LATER
    assert revoked["ended_revision"] == 2
    assert load_taste().revision == 2 and load_taste().snapshot().rules == ()


def test_reapproval_keeps_each_approval_body_and_ending() -> None:
    body = _body()
    identity = add_proposals(body)[0].id
    approve(identity)
    first = copy.deepcopy(load_taste().history[0])
    initial_hash = load_taste().snapshot().sha256
    revoke(identity)
    changed = {**body, "context": "Additional evidence for the same preference."}
    add_proposals(changed)
    approve(identity)
    taste = load_taste()
    assert [(entry["action"], entry["revision"]) for entry in taste.history] == [
        ("approved", 1),
        ("revoked", 2),
        ("reapproved", 3),
    ]
    assert taste.history[0] == first and first["rule"] == body
    assert taste.history[-1]["rule"]["context"] == changed["context"]
    assert taste.snapshot().sha256 == initial_hash
    assert len(taste.rules) == 1 and taste.rules[0]["approved_revision"] == 3


def test_superseding_approval_records_both_decisions_in_one_revision() -> None:
    original = _approved()
    replacement = add_proposals(_body(value=4))[0].id
    result = approve(replacement)
    history = load_taste().history
    assert [entry["action"] for entry in history] == ["approved", "approved", "superseded"]
    assert history[1]["revision"] == history[2]["revision"] == result.revision
    assert history[2]["rule_id"] == original and history[2]["by"] == replacement
    assert history[2]["rule"] is None


@pytest.mark.parametrize("change", ["remove", "action", "status", "field"])
def test_tampered_decision_history_is_refused(change: str) -> None:
    identity = _approved()
    revoke(identity)
    add_proposals(_body())
    approve(identity)
    data = _json(taste_path())
    if change == "remove":
        del data["history"][1]
    elif change == "action":
        data["history"][0]["action"] = "reapproved"
    elif change == "status":
        data["history"][1]["action"] = "approved"
    else:
        del data["history"][1]["by"]
    _file(taste_path(), data)
    with pytest.raises(TasteError, match="history"):
        load_taste()


def test_valid_ended_record_cannot_disagree_with_history() -> None:
    identity = _approved()
    revoke(identity)
    data = _json(taste_path())
    data["rules"][0].update(status="approved", ended_at=None, ended_revision=None)
    _file(taste_path(), data)
    with pytest.raises(TasteError, match="does not match decision history"):
        load_taste()


def _legacy_taste(*, gap: bool = False) -> dict[str, Any]:
    """Return exactly the earlier persisted shape, including a replaced rule."""
    active_body, ended_body = _body(3), _body(4)
    active = {
        **active_body,
        "id": rule_id(active_body),
        "status": "approved",
        "proposed_at": NOW,
        "approved_at": NOW,
        "approved_revision": 6 if gap else 2,
        "ended_at": None,
        "ended_revision": None,
        "superseded_by": None,
    }
    ended = {
        **ended_body,
        "id": rule_id(ended_body),
        "status": "superseded",
        "proposed_at": NOW,
        "approved_at": NOW,
        "approved_revision": 2 if gap else 1,
        "ended_at": NOW,
        "ended_revision": 4 if gap else 2,
        "superseded_by": active["id"],
    }
    return {
        "schema": TASTE_SCHEMA,
        "revision": 6 if gap else 2,
        "updated_at": NOW,
        "rules": [active, ended],
    }


@pytest.mark.parametrize("gap", [False, True])
def test_legacy_taste_loads_applies_and_shows_without_writing(
    gap: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    from yap_studio import cli

    legacy = _legacy_taste(gap=gap)
    _file(taste_path(), legacy)
    before = taste_path().read_bytes()
    taste = load_taste()
    assert all(entry["migrated"] is True for entry in taste.history)
    if gap:
        assert [entry["revision"] for entry in taste.history] == [2, 4, 6]
    else:
        assert [(entry["revision"], entry["action"]) for entry in taste.history] == [
            (1, "approved"),
            (2, "approved"),
            (2, "superseded"),
        ]
    assert taste.history[0]["rule"] == _body(4)
    superseded = next(entry for entry in taste.history if entry["action"] == "superseded")
    assert superseded["by"] == legacy["rules"][0]["id"]
    result = apply_taste(resolve("preset:clean_minimal"))
    assert result.style.data["captions"]["max_words_per_phrase"] == 3
    assert result.applied[0]["id"] == legacy["rules"][0]["id"]
    assert cli.main(["taste", "show", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["history"] == list(taste.history)
    assert cli.main(["taste", "show"]) == 0
    assert legacy["rules"][0]["id"] in capsys.readouterr().out
    assert taste_path().read_bytes() == before
    assert not proposals_path().exists() and not (config_root() / ".taste.lock").exists()


@pytest.mark.parametrize("operation", ["approve", "revoke", "reject", "add-proposal"])
@pytest.mark.parametrize("gap", [False, True])
def test_next_mutation_saves_migrated_history(operation: str, gap: bool) -> None:
    legacy = _legacy_taste(gap=gap)
    _file(taste_path(), legacy)
    migrated = copy.deepcopy(list(load_taste().history))
    body = _body(2)
    identity = rule_id(body)
    if operation in {"approve", "reject"}:
        pending = {
            **body,
            "id": identity,
            "status": "proposed",
            "proposed_at": NOW,
            "decided_at": None,
        }
        _file(proposals_path(), Proposals((pending,)).to_json())
    if operation == "approve":
        approve(identity)
    elif operation == "revoke":
        revoke(legacy["rules"][0]["id"])
    elif operation == "reject":
        reject(identity)
    else:
        add_proposals(body)
    saved = _json(taste_path())
    assert saved["history"][: len(migrated)] == migrated
    normal = saved["history"][len(migrated) :]
    if operation in {"approve", "revoke"}:
        assert normal and all("migrated" not in entry for entry in normal)
        assert {entry["revision"] for entry in normal} == {legacy["revision"] + 1}
    else:
        assert not normal and saved["revision"] == legacy["revision"]
    assert load_taste().to_json() == saved


def test_migrated_entry_after_normal_decision_is_refused() -> None:
    _file(taste_path(), _legacy_taste())
    revoke(load_taste().rules[0]["id"])
    data = _json(taste_path())
    data["history"].append(copy.deepcopy(data["history"][0]))
    _file(taste_path(), data)
    with pytest.raises(TasteError, match="migrated entries must precede normal entries"):
        load_taste()


@pytest.mark.parametrize("marker", [False, None, 1, "true"])
def test_migrated_marker_must_be_true(marker: object) -> None:
    _file(taste_path(), _legacy_taste())
    data = load_taste().to_json()
    data["history"][0]["migrated"] = marker
    _file(taste_path(), data)
    with pytest.raises(TasteError, match="must be true"):
        load_taste()


def test_revision_gap_after_migrated_prefix_is_refused() -> None:
    _file(taste_path(), _legacy_taste(gap=True))
    revoke(load_taste().rules[0]["id"])
    data = _json(taste_path())
    data["revision"] += 1
    data["history"][-1]["revision"] += 1
    data["rules"][0]["ended_revision"] += 1
    _file(taste_path(), data)
    with pytest.raises(TasteError, match="every taste revision must have a decision"):
        load_taste()


@pytest.mark.parametrize("operation", [approve, reject, revoke])
def test_decisions_refuse_unknown_ids(operation: Any) -> None:
    with pytest.raises(TasteRefusal) as caught:
        operation("r-000000000000")
    assert caught.value.error_type == "taste_refusal"
    assert caught.value.suggested_action
    assert not taste_path().exists() and not proposals_path().exists()


def test_rejecting_approved_rule_names_revoke_and_other_invalid_transitions_refuse() -> None:
    identity = _approved()
    with pytest.raises(TasteRefusal) as caught:
        reject(identity)
    assert "yap-studio taste revoke" in caught.value.suggested_action
    with pytest.raises(TasteRefusal):
        approve(identity)
    revoke(identity)
    with pytest.raises(TasteRefusal):
        revoke(identity)
    second = add_proposals(_body(4))[0].id
    reject(second)
    with pytest.raises(TasteRefusal):
        reject(second)
    with pytest.raises(TasteRefusal):
        approve(second)


def test_supersession_matches_target_and_equal_condition() -> None:
    first = _approved(_body(3))
    conditional = _approved(_body(4, condition=_condition("screen")))
    replacement = add_proposals(_body(2))[0].id
    result = approve(replacement)
    assert result.revision == 3 and result.superseded_ids == (first,)
    rules = {rule["id"]: rule for rule in load_taste().rules}
    assert rules[conditional]["status"] == "approved"
    assert rules[first]["status"] == "superseded"
    assert rules[first]["ended_revision"] == 3
    assert rules[first]["superseded_by"] == replacement
    assert {rule["id"] for rule in load_taste().snapshot().rules} == {conditional, replacement}
    assert add_proposals(_body(3))[0].status == "added"
    result = approve(first)
    assert result.revision == 4 and result.superseded_ids == (replacement,)
    assert len(load_taste().rules) == 3


@pytest.mark.parametrize(
    ("machine", "replacement", "scope"),
    [
        (
            {"kind": "filler_word", "word": "so", "position": "leading"},
            {"kind": "filler_word", "word": "so", "position": "anywhere"},
            "cut",
        ),
        (
            {"kind": "placement", "layer": "title", "rule": "clear_of_face"},
            {"kind": "placement", "layer": "title", "rule": "clear_of_face"},
            "cards",
        ),
    ],
)
def test_supersession_for_other_machine_targets(
    machine: dict, replacement: dict, scope: str
) -> None:
    first = _approved({**_body(), "scope": scope, "machine": machine})
    second = add_proposals(
        {
            **_body(),
            "scope": scope,
            "machine": replacement,
            "sentence": "Use this preference instead.",
        }
    )[0].id
    assert approve(second).superseded_ids == (first,)


def test_rules_without_machine_never_supersede() -> None:
    first = _approved({**_body(), "machine": None})
    second = _approved({**_body(4), "machine": None})
    assert {rule["id"] for rule in load_taste().snapshot().rules} == {first, second}


def test_reapproving_revoked_rule_reactivates_in_place() -> None:
    identity = _approved()
    revoke(identity)
    assert add_proposals(_body())[0].status == "added"
    result = approve(identity)
    assert result.revision == 3 and result.rule["approved_revision"] == 3
    assert (
        result.rule["ended_at"]
        is result.rule["ended_revision"]
        is result.rule["superseded_by"]
        is None
    )
    assert load_taste().rules == (result.rule,)


def test_snapshot_hash_ignores_evidence_context_revision_and_audit_records() -> None:
    _approved()
    taste = load_taste()
    snapshot = taste.snapshot()
    minimal = {
        key: taste.rules[0][key] for key in ("id", "scope", "condition", "sentence", "machine")
    }
    payload = {"schema": SNAPSHOT_SCHEMA, "rules": [minimal]}
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    assert snapshot.sha256 == expected
    assert snapshot.to_json() == {**payload, "revision": 1, "sha256": expected}
    altered = {
        **taste.rules[0],
        "sources": [_measurement()],
        "context": None,
        "approved_at": LATER,
        "approved_revision": 10,
        "proposed_at": LATER,
    }
    assert Taste(10, LATER, (altered,)).snapshot().sha256 == expected
    audit = {**_body(5), "id": rule_id(_body(5)), "status": "revoked"}
    assert Taste(11, LATER, (*taste.rules, audit)).snapshot().sha256 == expected
    second = _approved(_body(4))
    changed = load_taste().snapshot()
    assert changed.sha256 != expected and [rule["id"] for rule in changed.rules] == [second]
    other_field = _approved(_body(0.1, field="music.level"))
    taste = load_taste()
    assert [rule["id"] for rule in taste.snapshot().rules] == sorted([second, other_field])
    assert replace(taste, rules=tuple(reversed(taste.rules))).snapshot() == taste.snapshot()
    snapshot.to_json()["rules"][0]["machine"]["value"] = 7
    assert snapshot.rules[0]["machine"]["value"] == 3


def test_failed_second_approval_write_recovers_pending_in_memory_and_on_next_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = add_proposals(_body())[0].id
    write = paths.atomic_write_text
    calls = []

    def interrupted(target: Path, text: str) -> Path:
        calls.append(target.name)
        if target == proposals_path():
            raise OSError("interrupted proposal write")
        return write(target, text)

    with monkeypatch.context() as patch:
        patch.setattr(paths, "atomic_write_text", interrupted)
        with pytest.raises(TasteError):
            approve(identity)
    assert calls == ["taste.json", "taste-proposals.json"]
    assert _json(proposals_path())["proposals"][0]["id"] == identity
    assert load_proposals().pending == ()
    assert load_taste().snapshot().rules[0]["id"] == identity
    assert add_proposals(_body())[0].status == "already_approved"
    assert _json(proposals_path())["proposals"] == []


def test_all_mutations_reread_both_files_inside_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    lock = store._lock
    inside = False
    counts = {"taste": 0, "proposals": 0}
    from contextlib import contextmanager

    @contextmanager
    def tracked_lock():
        nonlocal inside
        with lock():
            inside = True
            try:
                yield
            finally:
                inside = False

    taste_reader, proposal_reader = store.load_taste, store._read_proposals

    def taste_read():
        assert inside
        counts["taste"] += 1
        return taste_reader()

    def proposal_read():
        assert inside
        counts["proposals"] += 1
        return proposal_reader()

    monkeypatch.setattr(store, "_lock", tracked_lock)
    monkeypatch.setattr(store, "load_taste", taste_read)
    monkeypatch.setattr(store, "_read_proposals", proposal_read)
    first = add_proposals(_body())[0].id
    approve(first)
    revoke(first)
    second = add_proposals(_body(4))[0].id
    reject(second)
    assert counts == {"taste": 5, "proposals": 5}


@pytest.mark.skipif(os.name == "nt", reason="POSIX lock contention")
def test_lock_timeout_and_release() -> None:
    import fcntl

    with store._lock():
        with pytest.raises(TasteRefusal) as caught:
            with store._lock(timeout_s=0):
                pytest.fail("a second lock was acquired")
    assert caught.value.code == "taste_locked"
    with store._lock(timeout_s=0):
        pass
    descriptor = os.open(config_root() / ".taste.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("marker", [".git", "project.json"])
@pytest.mark.parametrize("on_root", [False, True])
def test_writes_refuse_repositories_and_projects_but_reads_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
    on_root: bool,
) -> None:
    outer = tmp_path / "private"
    root = outer / "nested" / "config"
    root.mkdir(parents=True)
    (root if on_root else outer).joinpath(marker).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(root))
    assert load_taste() == Taste() and load_proposals() == Proposals()
    with pytest.raises(TasteError) as caught:
        add_proposals(_body())
    assert caught.value.parameter == "taste"
    assert "YAP_STUDIO_CONFIG" in caught.value.suggested_action
    assert not (root / ".taste.lock").exists()


def test_home_markers_are_excluded_but_child_markers_are_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".git").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    root = home / "private" / "config"
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(root))
    assert add_proposals(_body())[0].status == "added"
    (root.parent / "project.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TasteError):
        reject(rule_id(_body()))


def test_symlink_ancestor_cannot_bypass_repository_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(repository, target_is_directory=True)
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(alias / "config"))
    with pytest.raises(TasteError):
        add_proposals(_body())
    assert not (repository / "config").exists()


def test_operations_write_only_config_files(tmp_path: Path) -> None:
    first = add_proposals(_body())[0].id
    approve(first)
    second = add_proposals(_body(4))[0].id
    reject(second)
    revoke(first)
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == [
        "config",
        "config/.taste.lock",
        "config/taste-proposals.json",
        "config/taste.json",
    ]


def test_apply_approved_rules_only_and_preserve_input() -> None:
    style = resolve("preset:clean_minimal")
    before = copy.deepcopy(style.to_json())
    _approved(_body(4))
    active = _approved(_body(3))
    revoked = _approved(_body(0.2, field="music.level"))
    revoke(revoked)
    rejected = add_proposals(_body(0.3, field="music.level"))[0].id
    reject(rejected)
    add_proposals(_body(0.4, field="music.level"))
    result = apply_taste(style)
    assert result.style.data["captions"]["max_words_per_phrase"] == 3
    assert result.style.data["music"] == style.data["music"]
    assert result.applied == ({"id": active, "field": "captions.max_words_per_phrase", "value": 3},)
    assert result.skipped == ()
    assert result.style.name == style.name and result.style.source == style.source
    assert result.style.sha256() != style.sha256()
    assert style.to_json() == before
    assert result.taste_revision == 4 and result.taste_sha256 == load_taste().snapshot().sha256
    receipt = result.to_json()
    assert receipt["style"] == result.style.to_json()
    assert receipt["applied"] == list(result.applied)
    receipt["applied"][0]["value"] = 7
    assert result.applied[0]["value"] == 3


def test_apply_condition_tags_and_skip_other_machine_forms() -> None:
    condition_id = _approved(_body(3, condition=_condition("screen", "opening")))
    judgement = _approved(_body(4, condition=_condition()))
    placement = _approved(
        {**_body(), "machine": {"kind": "placement", "layer": "captions", "rule": "clear_of_face"}}
    )
    filler = _approved(
        {
            **_body(),
            "scope": "cut",
            "machine": {"kind": "filler_word", "word": "so", "position": "leading"},
        }
    )
    sentence = _approved({**_body(5), "machine": None})
    style = resolve("preset:clean_minimal")
    result = apply_taste(style, tags=("screen",))
    skipped = {item["id"]: item["reason"] for item in result.skipped}
    assert skipped == {
        condition_id: "condition_not_met",
        judgement: "condition_needs_judgement",
        filler: "not_a_style_rule",
        sentence: "followed_by_agent",
    }
    assert result.enforced == ({"id": placement, "check": "faces.clearance"},)
    assert result.style.data == style.data
    result = apply_taste(style, tags=("screen", "opening", "extra"))
    assert [item["id"] for item in result.applied] == [condition_id]
    assert result.style.data["captions"]["max_words_per_phrase"] == 3


def test_project_overrides_win_and_validate_engine_constraints() -> None:
    identity = _approved()
    style = resolve("preset:clean_minimal")
    result = apply_taste(style, explicit={"captions.max_words_per_phrase": 2, "music.level": 0.1})
    assert result.style.data["captions"]["max_words_per_phrase"] == 2
    assert result.style.data["music"]["level"] == 0.1
    assert result.applied == ()
    assert result.skipped == ({"id": identity, "reason": "project_override"},)
    assert result.explicit_fields == ("captions.max_words_per_phrase", "music.level")
    for field, value, suffix in [
        ("internal.fps", 30, "field"),
        ("name", "Video", "field"),
        ("captions.unknown", 1, "field"),
        ("captions.max_words_per_phrase", 9, "value"),
        ("captions.position", "top", "value"),
        ("captions.text_color", "red", "value"),
    ]:
        with pytest.raises(TasteError) as caught:
            apply_taste(style, explicit={field: value})
        assert caught.value.parameter == f"explicit.{field}.{suffix}"


def test_conflicts_use_tag_specificity_then_lower_id_independent_of_file_order() -> None:
    generic = _approved(_body(7))
    broad = _approved(_body(5, condition=_condition("screen")))
    narrow = _approved(_body(3, condition=_condition("screen", "opening")))
    tie = _approved(_body(2, condition=_condition("screen", "opening", text="For the first card.")))
    snapshot = load_taste().snapshot()
    style = resolve("preset:clean_minimal")
    result = apply_taste(style, snapshot, tags=("screen", "opening"))
    winner = min(narrow, tie)
    assert [item["id"] for item in result.applied] == [winner]
    assert {item["id"]: item["reason"] for item in result.skipped} == {
        identity: "conflict" for identity in (generic, broad, narrow, tie) if identity != winner
    }
    reverse = replace(snapshot, rules=tuple(reversed(snapshot.rules)))
    assert apply_taste(style, reverse, tags=("screen", "opening")).to_json() == result.to_json()
    assert apply_taste(style, snapshot, tags=("screen",)).applied[0]["id"] == broad
    assert apply_taste(style, snapshot).applied[0]["id"] == generic


def test_supplied_snapshot_does_not_read_store(monkeypatch: pytest.MonkeyPatch) -> None:
    from yap_studio.taste import resolve as taste_resolve

    def unexpected():
        pytest.fail("a supplied snapshot read the store")

    monkeypatch.setattr(taste_resolve, "load_taste", unexpected)
    style = resolve("preset:clean_minimal")
    result = apply_taste(style, Taste().snapshot())
    assert result.style.data == style.data and result.style is not style
