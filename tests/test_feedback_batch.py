"""Feedback batches preserve exact bases, carried snapshots, and input guards."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest
from test_feedback_locate import LAYERS, WORDS, _comment
from test_feedback_versions import _receipt_inputs, _render_media

from yap_studio.errors import InputError, ValidationError
from yap_studio.feedback import batch as batch_module
from yap_studio.feedback.batch import (
    FeedbackError,
    StaleBatchError,
    _supersedes,
    build_batch,
    check_inputs,
    latest_batch,
    load_batch,
    set_disposition,
    set_resulting_version,
)
from yap_studio.feedback.versions import LAYERS_SCHEMA, save_version
from yap_studio.project.model import CutWord, CutWords
from yap_studio.receipt import sha256_file
from yap_studio.review.export import comment_to_dict, to_export_json
from yap_studio.review.model import MediaFacts

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


def _json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def _cut(root: Path, *, words: CutWords = WORDS, label="first render",
         segments=("s1", "s2"), amplitude: float | None = None) -> None:
    _json(root / "cut" / "cut-words.json", words.to_dict())
    _json(root / "cut" / "spans.json", {"fps": "30/1", "segments": []})
    media = root / "cut" / "cut.mp4"
    _render_media(media, 120, amplitude=amplitude, label=label)
    _json(root / "cut" / "receipt.json", {
        "schema": "yap-studio/stage-receipt@1", "status": "verified",
        "inputs": _receipt_inputs(root),
        "outputs": [{"path": name, "sha256": sha256_file(root / name)} for name in (
            "cut/cut.mp4", "cut/spans.json", "cut/cut-words.json", "cut/voice.wav"
        ) if (root / name).exists()],
        "plan": {"fps": "30/1", "total_frames": 120, "pieces": [
            {"segment_id": segment, "take": "t1", "out_f0": index * 60, "n_frames": 60,
             "src_t0": "0", "src_t1": "2"} for index, segment in enumerate(segments)
        ]},
    })


@pytest.fixture
def feedback_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    _json(root / "project.json", {"schema": "yap-studio/project@1", "name": "sample-video",
                                 "takes": [{"id": "t1", "path": "takes/t1.mp4"}]})
    _json(root / "edit.json", {"schema": "yap-studio/edit@1", "words_sha256": {},
                               "segments": [], "end_hold_s": 0})
    _json(root / "timeline" / "agent.json", {"schema": "yap-studio/timeline@1", "elements": []})
    _cut(root)
    return root


def _layers(root: Path) -> Path:
    path = root / "windows.json"
    _json(path, {"schema": LAYERS_SCHEMA, "time_base": "output", "layers": [
        {"id": layer.id, "kind": layer.kind, "start": layer.start, "end": layer.end,
         "text": layer.text} for layer in LAYERS
    ]})
    return path


def _export(root: Path, comments, *, version=None) -> dict:
    digest = version.media["sha256"] if version else sha256_file(root / "cut" / "cut.mp4")
    return to_export_json(MediaFacts("render.mp4", digest, 4.0, Fraction(30), True),
                          list(comments), None, NOW)


def test_batch_uses_current_after_rollback(feedback_project: Path) -> None:
    from yap_studio.feedback.release import rollback

    root = feedback_project
    first = save_version(root)
    _cut(root, label="second render")
    second = save_version(root)
    rollback(root, "v1")
    batch = build_batch(root, _export(root, [_comment(target="caption")], version=first))
    assert batch["base_version"] == "v1"
    with pytest.raises(FeedbackError, match=r"v2 is not the current version \(v1\)"):
        build_batch(root, _export(root, [_comment()], version=second))


def test_batch_ids_are_sorted_by_number(feedback_project: Path) -> None:
    from yap_studio.feedback.batch import batch_ids

    folder = feedback_project / "feedback"
    folder.mkdir()
    for name in ("b10.json", "b2.json", "b1.json", "b0.json"):
        (folder / name).write_text("{}")
    assert batch_ids(feedback_project) == ["b1", "b2", "b10"]


def test_build_saved_version_slots_evidence_and_input_hashes(feedback_project: Path) -> None:
    root = feedback_project
    version = save_version(root, layers=_layers(root))
    export = _export(root, [_comment("c1", target="element"),
                            _comment("c2", kind="keep", start_frame=66, end_frame=93)])
    batch = build_batch(root, export, now=lambda: NOW)
    assert load_batch(root, "b1") == batch == latest_batch(root)
    assert batch["schema"] == "yap-studio/feedback@1" and batch["project"] == "sample-video"
    assert batch["base"] == "version" and batch["base_version"] == version.id
    assert batch["created_at"] == "2026-10-05T12:00:00Z"
    assert batch["review"] == {"media_sha256": version.media["sha256"], "media_name": "render.mp4",
                               "reviewer": None, "exported_at": "2026-10-05T12:00:00Z",
                               "series_id": None}
    assert batch["expected_inputs"] == {
        "edit.json": sha256_file(root / "edit.json"), "removals.json": None,
        "timeline/agent.json": sha256_file(root / "timeline" / "agent.json"),
    }
    fix = next(item for item in batch["items"] if item["comment"]["id"] == "c1")
    keep = next(item for item in batch["items"] if item["comment"]["id"] == "c2")
    assert fix["comment"]["revision"] == export["comments"][1]["updated_at"]
    assert fix["location"]["words"][0]["segment"] == "s2"
    assert fix["location"]["segments"] == ["s2"]
    assert [layer["id"] for layer in fix["location"]["layers"]] == ["card", "caption"]
    for item in batch["items"]:
        assert item["disposition"] == "open" and item["operations"] == []
        assert item["evidence"] is None and item["unlocated"] is None and not item["carried"]
    assert fix["constraint"] is None and batch["constraints"] == [keep["constraint"]]
    assert [(word["id"], word["segment"]) for word in keep["constraint"]["words"]] == [
        ("t1:0", "s2"), ("t1:2", "s2"),
    ]
    assert batch["released"] == [] and batch["resulting_version"] is None
    check_inputs(root, batch)


def test_legacy_current_cut_and_layer_override(feedback_project: Path) -> None:
    root = feedback_project
    comment = _comment(kind="keep", target=None, start_frame=18, start_s=0.6)
    data = comment_to_dict(comment)
    data.pop("kind")
    data.pop("target")
    data.pop("request")
    data["label"] = "like"
    export = _export(root, [])
    export.update(version=1, comments=[data])
    batch = build_batch(root, export, layers=_layers(root))
    assert batch["base"] == "cut" and batch["base_version"] is None
    item = batch["items"][0]
    assert item["comment"]["kind"] == "keep" and item["comment"]["target"] is None
    assert item["constraint"]["target"] == "story"
    assert [word["id"] for word in item["constraint"]["words"]] == ["t1:0", "t1:1"]
    assert item["constraint"]["gaps"] == pytest.approx([0.3])
    assert item["location"]["layers_known"] is True
    data["label"] = "change"
    assert build_batch(root, export)["items"][0]["comment"]["kind"] == "fix"


def test_current_export_without_request_still_builds(feedback_project: Path) -> None:
    export = _export(feedback_project, [_comment(target="caption")])
    del export["comments"][0]["request"]
    item = build_batch(feedback_project, export)["items"][0]
    assert item["request"] is None and item["comment"]["request"] is None


def test_batch_export_refuses_invalid_spelling_request(feedback_project: Path) -> None:
    export = _export(feedback_project, [_comment(target="story")])
    export["comments"][0]["request"] = {"type": "spelling", "from": "wrong", "to": "right"}
    with pytest.raises(InputError, match="a spelling fix needs"):
        build_batch(feedback_project, export)
    assert not (feedback_project / "feedback").exists()


def test_batch_carries_restore_request_without_automatic_operations(feedback_project: Path) -> None:
    version = save_version(feedback_project)
    request = {"type": "restore", "range": "s1.0", "take": "t1", "start": 1.0,
               "end": 2.0, "text": "um well"}
    comment = replace(_comment(target="story", start_frame=30, start_s=1.0), request=request)
    batch = build_batch(feedback_project, _export(feedback_project, [comment], version=version))
    item = batch["items"][0]
    assert item["request"] == item["comment"]["request"] == request
    assert item["disposition"] == "open" and item["operations"] == []
    item["request"]["text"] = "changed"
    assert item["comment"]["request"]["text"] == comment.request["text"] == "um well"
    assert load_batch(feedback_project, batch["id"])["items"][0]["request"] == request


@pytest.mark.parametrize("kind,target", [
    (None, None), ("fix", "speech"), ("keep", None), ("keep", "story"),
    ("keep", "video"), ("keep", "caption"), ("keep", "element"),
])
def test_current_cut_batch_skips_decode_without_speech_keeps(
    feedback_project: Path, monkeypatch: pytest.MonkeyPatch, kind: str | None, target: str | None,
) -> None:
    def unexpected_decode(path):
        pytest.fail("audio was decoded without speech delivery evidence to record")

    monkeypatch.setattr(batch_module, "media_envelope", unexpected_decode)
    comments = [_comment(kind=kind, target=target)] if kind else []
    assert build_batch(feedback_project, _export(feedback_project, comments))["base"] == "cut"


@pytest.mark.parametrize("amplitude", [None, 0.1])
def test_current_cut_batch_decodes_once_for_multiple_speech_keeps(
    feedback_project: Path, monkeypatch: pytest.MonkeyPatch, amplitude: float | None,
) -> None:
    root = feedback_project
    _cut(root, amplitude=amplitude)
    decode = batch_module.media_envelope
    calls = []

    def measure(path):
        calls.append(path)
        return decode(path)

    monkeypatch.setattr(batch_module, "media_envelope", measure)
    comments = [_comment("c1", kind="keep", target="speech", start_frame=9),
                _comment("c2", kind="keep", target="speech")]
    batch = build_batch(root, _export(root, comments))
    assert calls == [root / "cut" / "cut.mp4"]
    assert len(batch["constraints"]) == 2
    assert all((constraint["delivery"] is None) == (amplitude is None)
               for constraint in batch["constraints"])


def test_current_cut_carried_speech_keep_uses_saved_delivery(
    feedback_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = feedback_project
    _cut(root, amplitude=0.1)
    first = save_version(root)
    _cut(root, amplitude=0.2, label="next render")
    export = _export(root, [])
    export["series"] = {
        "id": "series", "version": 2, "versions": [
            {"number": 1, "sha256": first.media["sha256"], "version_id": first.id},
            {"number": 2, "sha256": export["media"]["sha256"], "version_id": None},
        ], "threads": [{**comment_to_dict(_comment("c1", kind="keep", target="speech")),
                        "version": 1, "reply": None, "state": "open"}],
    }

    def unexpected_decode(path):
        pytest.fail("a carried keep must use its own saved delivery evidence")

    monkeypatch.setattr(batch_module, "media_envelope", unexpected_decode)
    batch = build_batch(root, export)
    assert batch["base"] == "cut" and batch["items"][0]["carried"]
    assert batch["constraints"][0]["delivery"]["envelope"] == pytest.approx([-23.0] * 30, abs=0.3)


def test_saved_and_explicit_layers_conflict_and_missing_layers_are_unknown(
    feedback_project: Path,
) -> None:
    root = feedback_project
    version = save_version(root, layers=_layers(root))
    export = _export(root, [_comment()])
    with pytest.raises(FeedbackError, match="Both"):
        build_batch(root, export, layers=_layers(root))
    assert not (root / "feedback").exists()
    _cut(root, label="new render")
    batch = build_batch(root, _export(root, [_comment()]))
    assert batch["base"] == "cut" and batch["items"][0]["location"]["layers_known"] is False
    assert version.id == "v1"


def test_new_and_carried_comments_use_their_own_version_snapshots(feedback_project: Path) -> None:
    root = feedback_project
    first = save_version(root, layers=_layers(root))
    shifted = CutWords("output", {}, (
        CutWord("t1:2", "s2", "End", 0.2, 0.5),
        CutWord("t1:0", "s2", "Start", 0.8, 1.1),
        CutWord("t1:0", "s1", "Start", 2.2, 2.5),
        CutWord("t1:1", "s1", "really", 2.8, 3.1),
    ))
    _cut(root, words=shifted, label="retimed render", segments=("s2", "s1"))
    second = save_version(root)
    export = _export(root, [_comment("c1", start_frame=9)], version=second)
    export["series"] = {
        "id": "series-one", "version": 2, "versions": [
            {"number": 1, "sha256": first.media["sha256"], "name": "first.mp4", "version_id": "v1"},
            {"number": 2, "sha256": second.media["sha256"], "name": "next.mp4", "version_id": "v2"},
            {"number": 3, "sha256": "f" * 64, "name": "absent.mp4", "version_id": "v99"},
        ], "threads": [
            {**comment_to_dict(_comment("c1")), "version": 1, "reply": None, "state": "open"},
            {**comment_to_dict(_comment("c2", start_frame=9)), "version": 1,
             "reply": None, "state": "needs_clarification"},
            {**comment_to_dict(_comment("c3")), "version": 3, "reply": None, "state": "open"},
            {**comment_to_dict(_comment("c4")), "version": 1, "reply": None, "state": "declined"},
        ],
    }
    batch = build_batch(root, export)
    own, old, question, missing = batch["items"]
    assert missing["disposition"] == "needs_clarification"
    assert missing["clarification"]["reason"] == "unlocated"
    assert missing["clarification"]["options"] == []
    assert missing["constraint"] is None and missing["operations"] == []
    assert own["comment"]["version"] == "v2" and own["location"]["words"][0]["id"] == "t1:2"
    assert own["location"]["segments"] == ["s2"]
    assert old["comment"]["version"] == "v1" and old["carried"]
    assert [(word["id"], word["segment"]) for word in old["location"]["words"]] == [("t1:0", "s2")]
    assert old["location"]["layers_known"] is True and own["location"]["layers_known"] is False
    assert question["location"]["words"][0]["id"] == "t1:0"
    assert missing["location"] is None and "not saved" in missing["unlocated"]
    assert missing["comment"]["version"] == "v99"
    export["series"]["versions"][0]["version_id"] = None
    assert build_batch(root, export)["items"][1]["comment"]["version"] == "v1"


def test_stale_and_unknown_reviews_are_refused(feedback_project: Path) -> None:
    root = feedback_project
    first = save_version(root)
    _cut(root, label="new render")
    save_version(root)
    with pytest.raises(FeedbackError, match=r"v1 is not the current version \(v2\)"):
        build_batch(root, _export(root, [_comment()], version=first))
    export = _export(root, [])
    export["media"]["sha256"] = "f" * 64
    with pytest.raises(FeedbackError, match="does not have") as error:
        build_batch(root, export)
    assert "yap-studio versions save" in error.value.suggested_action
    assert latest_batch(root) is None


@pytest.mark.parametrize("change", ["edit", "remove_lane", "add_lane", "removals"])
def test_check_inputs_names_all_changed_inputs(feedback_project: Path, change: str) -> None:
    root = feedback_project
    batch = build_batch(root, _export(root, []))
    if change == "edit":
        _json(root / "edit.json", {"changed": True})
        expected = ["edit.json"]
    elif change == "remove_lane":
        (root / "timeline" / "agent.json").unlink()
        expected = ["timeline/agent.json"]
    elif change == "add_lane":
        _json(root / "timeline" / "creator.json", {})
        expected = ["timeline/creator.json"]
    else:
        _json(root / "removals.json", {})
        expected = ["removals.json"]
    with pytest.raises(StaleBatchError) as error:
        check_inputs(root, batch)
    assert error.value.changed == expected
    assert all(filename in str(error.value) for filename in expected)


def test_dispositions_operation_uniqueness_and_resulting_version_once(
    feedback_project: Path,
) -> None:
    root = feedback_project
    first = save_version(root)
    build_batch(root, _export(root, [_comment("c1"), _comment("c2")]))
    set_disposition(root, "b1", "v1", "c1", "applied", operations=[{"id": "op1"}],
                    evidence={"checked": True})
    updated = load_batch(root, "b1")
    assert updated["items"][0]["evidence"] == {"checked": True}
    assert updated["items"][0]["operations"] == [{
        "id": "op1", "inputs_before": updated["expected_inputs"],
        "inputs_after": updated["expected_inputs"],
    }]
    for comment_id in ("c1", "c2"):
        with pytest.raises(FeedbackError, match="operation op1 was already applied"):
            set_disposition(root, "b1", "v1", comment_id, "applied", operations=[{"id": "op1"}])
    with pytest.raises(FeedbackError, match="operation op2 was already applied"):
        set_disposition(root, "b1", "v1", "c2", "applied",
                        operations=[{"id": "op2"}, {"id": "op2"}])
    assert load_batch(root, "b1") == updated
    with pytest.raises(ValidationError, match="disposition"):
        set_disposition(root, "b1", "v1", "c1", "unknown")
    with pytest.raises(ValidationError, match="comment"):
        set_disposition(root, "b1", "v2", "c1", "declined")
    with pytest.raises(ValidationError, match="string id"):
        set_disposition(root, "b1", "v1", "c1", "open", operations=[{"id": 1}])
    with pytest.raises(FeedbackError, match="newer"):
        set_resulting_version(root, "b1", first.id)
    with pytest.raises(InputError):
        set_resulting_version(root, "b1", "v9")
    _cut(root, label="changed cut")
    second = save_version(root)
    assert set_resulting_version(root, "b1", second.id)["resulting_version"] == "v2"
    with pytest.raises(FeedbackError, match="only once"):
        set_resulting_version(root, "b1", second.id)


def test_constraints_carry_forward_and_new_fix_releases_them(feedback_project: Path) -> None:
    root = feedback_project
    first = save_version(root)
    b1 = build_batch(root, _export(root, [_comment(kind="keep", start_frame=66, end_frame=93)]))
    _cut(root, label="next cut")
    second = save_version(root)
    b2 = build_batch(root, _export(root, []))
    assert b2["constraints"] == b1["constraints"]
    b3 = build_batch(root, _export(root, [_comment("c4", target="story")]))
    assert b3["constraints"] == []
    assert b3["released"] == [{"constraint": b1["constraints"][0],
                               "by": {"version": second.id, "id": "c4"}, "batch": "b3"}]
    assert b1["constraints"][0]["comment"] == {
        "version": first.id, "id": "c1", "revision": _comment().updated_at,
    }


def test_concurrent_builds_allocate_new_batches_without_overwriting(feedback_project: Path) -> None:
    root = feedback_project
    export = _export(root, [])
    with ThreadPoolExecutor(max_workers=2) as pool:
        batches = list(pool.map(lambda _: build_batch(root, export), range(2)))
    assert {batch["id"] for batch in batches} == {"b1", "b2"}
    assert latest_batch(root)["id"] == "b2"
    b2 = load_batch(root, "b2")
    b2["id"] = "b5"
    _json(root / "feedback" / "b5.json", b2)
    assert build_batch(root, export)["id"] == "b6"


@pytest.mark.parametrize("update", [
    {"format": "other"}, {"version": 0}, {"version": 3}, {"version": True},
    {"comments": [{"id": "bad"}]}, {"comments": "invalid"}, {"series": {}},
])
def test_invalid_exports_refused_without_a_batch(feedback_project: Path, update: dict) -> None:
    export = _export(feedback_project, [])
    export.update(update)
    with pytest.raises(InputError):
        build_batch(feedback_project, export)
    assert latest_batch(feedback_project) is None


def test_changed_current_media_is_not_silently_located(feedback_project: Path) -> None:
    export = _export(feedback_project, [_comment()])
    (feedback_project / "cut" / "cut.mp4").write_bytes(b"changed behind receipt")
    with pytest.raises(FeedbackError, match="cut/cut.mp4 changed after the verified cut"):
        build_batch(feedback_project, export)


def test_version_without_layers_accepts_explicit_windows(feedback_project: Path) -> None:
    root = feedback_project
    save_version(root)
    batch = build_batch(root, _export(root, [_comment()]), layers=_layers(root))
    assert batch["items"][0]["location"]["layers_known"] is True
    assert [layer["id"] for layer in batch["items"][0]["location"]["layers"]] == ["card", "caption"]


def test_series_positions_resolve_project_version_ids_and_carried_keep_releases(
    feedback_project: Path,
) -> None:
    root = feedback_project
    save_version(root)
    _cut(root, label="second render")
    second = save_version(root)
    _cut(root, label="third render")
    third = save_version(root)
    export = _export(root, [_comment("c4")])
    export["series"] = {
        "id": "series-two", "version": 2, "versions": [
            {"number": 1, "sha256": second.media["sha256"], "name": "second.mp4",
             "version_id": "v2"},
            {"number": 2, "sha256": third.media["sha256"], "name": "third.mp4", "version_id": "v3"},
        ], "threads": [{**comment_to_dict(_comment("c3", kind="keep")), "version": 1,
                        "reply": None, "state": "open"}],
    }
    batch = build_batch(root, export)
    assert [item["comment"]["version"] for item in batch["items"]] == ["v3", "v2"]
    assert batch["constraints"] == []
    assert batch["released"][0]["constraint"]["comment"] == {
        "version": "v2", "id": "c3", "revision": _comment().updated_at,
    }
    assert batch["released"][0]["by"] == {"version": "v3", "id": "c4"}


def test_rebuilding_fresh_keep_does_not_release_it_as_carried(feedback_project: Path) -> None:
    root = feedback_project
    save_version(root)
    export = _export(root, [_comment("c1"), _comment("c2", kind="keep")])
    first = build_batch(root, export)
    second = build_batch(root, export)
    assert second["constraints"] == first["constraints"] and second["released"] == []


def test_current_cut_result_must_follow_the_reviewed_base(feedback_project: Path) -> None:
    root = feedback_project
    first = save_version(root, now=lambda: NOW)
    _cut(root, label="reviewed current cut")
    build_batch(root, _export(root, []), now=lambda: NOW + timedelta(minutes=1))
    with pytest.raises(FeedbackError, match="newer"):
        set_resulting_version(root, "b1", first.id)
    same = save_version(root, now=lambda: NOW + timedelta(minutes=2))
    with pytest.raises(FeedbackError, match="newer"):
        set_resulting_version(root, "b1", same.id)
    _cut(root, label="feedback outcome")
    newer = save_version(root, now=lambda: NOW + timedelta(minutes=3))
    assert set_resulting_version(root, "b1", newer.id)["resulting_version"] == "v3"


def test_variable_rate_export_uses_seconds_on_its_base(feedback_project: Path) -> None:
    root = feedback_project
    comment = _comment(start_frame=None, start_s=2.2)
    facts = MediaFacts("render.mp4", sha256_file(root / "cut" / "cut.mp4"), 4.0, None, False)
    export = to_export_json(facts, [comment], None, NOW)
    item = build_batch(root, export)["items"][0]
    assert item["location"]["start_s"] == item["location"]["end_s"] == 2.2
    assert item["location"]["start_frame"] is item["location"]["end_frame"] is None
    assert [(word["id"], word["segment"]) for word in item["location"]["words"]] == [("t1:0", "s2")]


def test_input_check_names_multiple_changes_and_invalid_disposition_is_atomic(
    feedback_project: Path,
) -> None:
    root = feedback_project
    batch = build_batch(root, _export(root, [_comment()]))
    _json(root / "edit.json", {})
    _json(root / "removals.json", {})
    _json(root / "timeline" / "agent.json", {"changed": True})
    with pytest.raises(StaleBatchError) as error:
        check_inputs(root, batch)
    assert error.value.changed == ["edit.json", "removals.json", "timeline/agent.json"]
    with pytest.raises(ValidationError, match="valid JSON"):
        set_disposition(root, "b1", None, "c1", "applied", evidence=float("nan"))
    assert load_batch(root, "b1") == batch


def test_regression_unrelated_caption_fix_preserves_story_keep(feedback_project: Path) -> None:
    root = feedback_project
    save_version(root, layers=_layers(root))
    first = build_batch(root, _export(root, [_comment("c21", kind="keep", target="story")]))
    second = build_batch(root, _export(root, [_comment("c22", target="caption")]))
    assert second["constraints"] == first["constraints"]
    assert second["released"] == []


def test_regression_fix_in_another_segment_preserves_story_keep(feedback_project: Path) -> None:
    root = feedback_project
    save_version(root)
    first = build_batch(root, _export(root, [_comment("c21", kind="keep", start_frame=9)]))
    # Both ranges have t1:0, but the keep is in s1 and the fix is in s2.
    second = build_batch(root, _export(root, [_comment("c22", target="story")]))
    assert second["constraints"] == first["constraints"]
    assert second["released"] == []


@pytest.mark.parametrize("target", [None, "story", "speech", "caption", "element", "video"])
def test_matching_fresh_fix_releases_same_target_keep(feedback_project: Path, target: str) -> None:
    root = feedback_project
    save_version(root, layers=_layers(root))
    first = build_batch(root, _export(root, [_comment("c21", kind="keep", target=target)]))
    fix_target = "story" if target is None else target
    fix = _comment("c22", target=fix_target, start_frame=0 if target == "video" else 69)
    second = build_batch(root, _export(root, [fix]))
    assert second["constraints"] == []
    assert second["released"] == [{"constraint": first["constraints"][0],
                                    "by": {"version": "v1", "id": "c22"}, "batch": "b2"}]


def _release_export(root: Path, *, edited: bool = False) -> tuple[dict, dict]:
    first = save_version(root)
    keep = _comment("c21", kind="keep", target="story")
    build_batch(root, _export(root, [keep]))
    _cut(root, label="next render")
    second = save_version(root)
    export = _export(root, [_comment("c22", target="story")])
    export["series"] = {
        "id": "series", "version": 2, "versions": [
            {"number": 1, "sha256": first.media["sha256"], "version_id": "v1"},
            {"number": 2, "sha256": second.media["sha256"], "version_id": "v2"},
        ], "threads": [{**comment_to_dict(keep), "version": 1, "reply": None, "state": "open"}],
    }
    released = build_batch(root, export)
    assert released["constraints"] == [] and len(released["released"]) == 1
    export["comments"] = []
    if edited:
        export["series"]["threads"][0].update(
            text="Keep this again", updated_at="2026-10-05T13:00:00Z",
        )
    return released, export


def test_regression_released_keep_stays_released(feedback_project: Path) -> None:
    released, export = _release_export(feedback_project)
    third = build_batch(feedback_project, export)
    assert third["constraints"] == []
    assert third["released"] == released["released"]
    assert third["released"][0]["batch"] == "b2"
    assert third["released"][0]["constraint"]["comment"]["revision"] == _comment().updated_at
    fourth = build_batch(feedback_project, _export(feedback_project, []))
    assert fourth["constraints"] == [] and fourth["released"] == third["released"]


def test_regression_edited_released_keep_becomes_active(feedback_project: Path) -> None:
    _, export = _release_export(feedback_project, edited=True)
    third = build_batch(feedback_project, export)
    assert len(third["constraints"]) == 1 and third["released"] == []
    assert third["constraints"][0]["comment"]["revision"] == "2026-10-05T13:00:00Z"
    assert third["constraints"][0]["text"] == "Keep this again"


def test_regression_build_refuses_changed_cut_words(feedback_project: Path) -> None:
    root = feedback_project
    export = _export(root, [_comment()])
    shifted = CutWords("output", {}, tuple(replace(word, start=word.start + 0.1,
                                                  end=word.end + 0.1) for word in WORDS.words))
    _json(root / "cut" / "cut-words.json", shifted.to_dict())
    with pytest.raises(FeedbackError, match="cut/cut-words.json changed after the verified cut"):
        build_batch(root, export)
    assert not (root / "feedback").exists()


def test_regression_build_refuses_tampered_saved_cut_words(feedback_project: Path) -> None:
    root = feedback_project
    manifest = save_version(root)
    export = _export(root, [_comment()])
    shifted = CutWords("output", {}, tuple(replace(word, start=word.start + 0.1,
                                                  end=word.end + 0.1) for word in WORDS.words))
    _json(manifest.path / "cut-words.json", shifted.to_dict())
    with pytest.raises(InputError, match="snapshot changed after it was saved"):
        build_batch(root, export)
    assert not (root / "feedback").exists()


@pytest.mark.parametrize("filename", [
    "edit.json", "removals.json", "cut/spans.json", "cut/voice.wav",
])
def test_build_refuses_other_stale_cut_files(feedback_project: Path, filename: str) -> None:
    from test_feedback_versions import _voice

    root = feedback_project
    edit = json.loads((root / "edit.json").read_text())
    edit["segments"] = [{"id": "s1", "take": "t1", "from": {"time": 0}, "to": {"time": 2}}]
    _json(root / "edit.json", edit)
    _voice(root)
    _cut(root)
    export = _export(root, [])
    if filename == "removals.json":
        _json(root / filename, {"schema": "yap-studio/removals@1", "removals": [
            {"id": "r1", "take": "t1", "start": 0.3, "end": 0.4,
             "kind": "manual", "status": "applied"},
        ]})
    else:
        (root / filename).write_bytes(b"changed")
    with pytest.raises(FeedbackError, match=f"{filename} changed after the verified cut"):
        build_batch(root, export)
    assert not (root / "feedback").exists()


def test_regression_applied_operation_refuses_stale_batch(feedback_project: Path) -> None:
    root = feedback_project
    build_batch(root, _export(root, [_comment()]))
    batch_path = root / "feedback" / "b1.json"
    before = batch_path.read_bytes()
    edit = json.loads((root / "edit.json").read_text())
    edit["end_hold_s"] = 0.5
    _json(root / "edit.json", edit)
    with pytest.raises(StaleBatchError) as error:
        set_disposition(root, "b1", None, "c1", "applied", operations=[{"id": "op1"}])
    assert error.value.changed == ["edit.json"]
    assert batch_path.read_bytes() == before
    # Marking applied without an operation has the same guard.
    with pytest.raises(StaleBatchError):
        set_disposition(root, "b1", None, "c1", "applied")
    assert batch_path.read_bytes() == before


def test_operation_capture_edit_record_advances_expected_inputs(feedback_project: Path) -> None:
    from yap_studio.feedback.batch import current_inputs

    root = feedback_project
    first = build_batch(root, _export(root, [_comment()]))
    before = current_inputs(root)
    assert before == first["expected_inputs"]
    edit = json.loads((root / "edit.json").read_text())
    edit["end_hold_s"] = 0.5
    _json(root / "edit.json", edit)
    after = current_inputs(root)
    operation = {"id": "op1", "change": {"end_hold_s": 0.5}}
    updated = set_disposition(root, "b1", None, "c1", "applied", operations=[operation],
                              inputs_before=before)
    assert updated["expected_inputs"] == after
    assert updated["items"][0]["operations"] == [{**operation, "inputs_before": before,
                                                "inputs_after": after}]
    assert operation == {"id": "op1", "change": {"end_hold_s": 0.5}}
    batch_bytes = (root / "feedback" / "b1.json").read_bytes()
    with pytest.raises(StaleBatchError, match="edit.json"):
        set_disposition(root, "b1", None, "c1", "applied", operations=[{"id": "op2"}],
                        inputs_before=before)
    assert (root / "feedback" / "b1.json").read_bytes() == batch_bytes
    edit["end_hold_s"] = 1.0
    _json(root / "edit.json", edit)
    final = set_disposition(root, "b1", None, "c1", "open", operations=[{"id": "op2"}],
                            inputs_before=after)
    assert final["expected_inputs"] == current_inputs(root)
    assert final["items"][0]["operations"][1]["inputs_before"] == after


def test_non_operation_dispositions_can_be_recorded_on_stale_inputs(feedback_project: Path) -> None:
    root = feedback_project
    first = build_batch(root, _export(root, [_comment()]))
    _json(root / "edit.json", {})
    updated = set_disposition(root, "b1", None, "c1", "declined", evidence={"reason": "scope"})
    assert updated["expected_inputs"] == first["expected_inputs"]
    assert updated["items"][0]["disposition"] == "declined"


@pytest.mark.parametrize("target", [None, "story", "speech"])
def test_regression_fresh_fix_releases_keep_in_same_pause(
    feedback_project: Path, target: str | None,
) -> None:
    root = feedback_project
    keep = _comment("c31", kind="keep", target=target, start_frame=18)
    first = build_batch(root, _export(root, [keep]))
    assert [(word["id"], word["segment"]) for word in first["constraints"][0]["words"]] == [
        ("t1:0", "s1"), ("t1:1", "s1"),
    ]
    other = _comment("c32", target=target, start_frame=78)
    second = build_batch(root, _export(root, [other]))
    assert second["constraints"] == first["constraints"] and second["released"] == []
    fix = _comment("c33", target=target, start_frame=21)
    third = build_batch(root, _export(root, [fix]))
    assert third["items"][0]["location"]["status"] == "silence"
    assert third["constraints"] == []
    assert third["released"] == [{
        "constraint": first["constraints"][0], "by": {"version": None, "id": "c33"}, "batch": "b3",
    }]


@pytest.mark.parametrize("change", [
    "before_id", "after_id", "before_segment", "after_segment", "before_none", "after_none",
    "pause_false", "pause_missing", "carried", "target", "status", "one_neighbour",
])
def test_pause_fix_requires_both_neighbours_and_pause_flag(change: str) -> None:
    from test_feedback_locate import PIECES

    from yap_studio.feedback.keeps import keep_constraint
    from yap_studio.feedback.locate import locate

    keep = _comment("c31", kind="keep", target="story", start_frame=18)
    location = locate(keep, fps=Fraction(30), duration_s=4, words=WORDS, pieces=PIECES, layers=None)
    constraint = keep_constraint(keep, location, version="v1")
    item = {"comment": {"kind": "fix", "target": "story"}, "carried": False,
            "location": location.to_dict()}
    assert _supersedes(constraint, item)
    if change.endswith("_id") or change.endswith("_segment"):
        neighbour, field = change.split("_")
        item["location"][neighbour][field] = "different"
    elif change.endswith("_none"):
        item["location"][change.split("_")[0]] = None
    elif change == "pause_false":
        constraint["pause"] = False
    elif change == "pause_missing":
        del constraint["pause"]
    elif change == "carried":
        item["carried"] = True
    elif change == "target":
        item["comment"]["target"] = "speech"
    elif change == "status":
        item["location"]["status"] = "outside"
    else:
        constraint["words"] = constraint["words"][:1]
    assert not _supersedes(constraint, item)


def test_pause_keep_can_still_be_released_by_a_fix_on_a_neighbour() -> None:
    from test_feedback_locate import _locate

    from yap_studio.feedback.keeps import keep_constraint

    keep = _comment("c31", kind="keep", target="story", start_frame=18)
    constraint = keep_constraint(keep, _locate(keep), version="v1")
    fix = _comment("c32", target="story", start_frame=9)
    assert _supersedes(constraint, {"comment": comment_to_dict(fix), "carried": False,
                                   "location": _locate(fix).to_dict()})
