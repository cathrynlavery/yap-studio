"""B-roll feedback edits real lanes and preserves the batch transaction guards."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_feedback_apply import _document, _tree
from test_feedback_batch import _export, _json
from test_feedback_locate import _comment
from test_feedback_versions import _render_media
from test_timeline import fragment, graphic, insert, overlay
from test_timeline import project as timeline_fixture

from yap_studio.errors import ValidationError
from yap_studio.feedback import apply as apply_module
from yap_studio.feedback.apply import apply_operations
from yap_studio.feedback.batch import FeedbackError, build_batch, load_batch
from yap_studio.feedback.clarify import automatic_clarification
from yap_studio.feedback.edits import NeedsTimelineError, validate_operations
from yap_studio.feedback.locate import locate
from yap_studio.feedback.versions import LAYERS_SCHEMA, save_version, version_cut_words
from yap_studio.receipt import sha256_file
from yap_studio.stages.timeline import add_timeline, resolve_timeline


def _review_export(root: Path, comments: list) -> dict:
    export = _export(root, comments)
    export["media"]["duration_s"] = 12.0
    return export


@pytest.fixture
def broll_project(tmp_path: Path) -> Path:
    root = timeline_fixture.__wrapped__(tmp_path)
    add_timeline(root, fragment(root, insert(root), overlay(
        root, id="broll-overlay", start={"word": "t1:6"}, end={"word": "t1:7"},
    )), lane="broll").require_valid()
    add_timeline(root, fragment(root, graphic(start={"word": "t1:9"}, hold_s=1)),
                 lane="agent").require_valid()
    _render_media(root / "cut/cut.mp4", 360)
    _json(root / "cut/spans.json", {
        "sources": {"t1": "../source.mov"}, "fps": "30/1",
        "segments": [{"source": "t1", "start": "0", "end": "12"}],
    })
    receipt_path = root / "cut/receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["outputs"] += [{"path": name, "sha256": sha256_file(root / name)}
                           for name in ("cut/cut.mp4", "cut/spans.json")]
    receipt["plan"].update(fps="30/1", pieces=[{
        "segment_id": "s1", "take": "t1", "out_f0": 0, "n_frames": 360,
        "src_t0": "0", "src_t1": "12",
    }])
    _json(receipt_path, receipt)
    resolved = resolve_timeline(root).require_valid()
    _json(root / "layers.json", {
        "schema": LAYERS_SCHEMA, "time_base": "output",
        "layers": [{"id": item.id, "kind": item.kind, "text": None,
                    "start": item.start_s, "end": item.end_s} for item in resolved.elements],
    })
    save_version(root, layers=root / "layers.json")
    build_batch(root, _review_export(root, [
        _comment("c1", target="element"),
        _comment("c2", target="element", start_frame=189),
        _comment("c3", kind="keep", target="story", start_frame=9),
    ]))
    return root


def _op(kind: str, element: str = "broll-one", **fields) -> dict:
    return {"id": "change", "kind": kind, "element": element, **fields}


def _element(root: Path, identifier: str = "broll-one"):
    return next(item for item in resolve_timeline(root).require_valid().elements
                if item.id == identifier)


def _next_version(root: Path):
    _render_media(root / "cut/cut.mp4", 360, label="next render")
    source = root / "cut/receipt.json"
    receipt = json.loads(source.read_bytes())
    next(output for output in receipt["outputs"] if output["path"] == "cut/cut.mp4")[
        "sha256"
    ] = sha256_file(root / "cut/cut.mp4")
    _json(source, receipt)
    return save_version(root)


@pytest.mark.parametrize("element", ["broll-one", "broll-overlay"])
@pytest.mark.parametrize("kind", ["broll_media", "broll_retime", "broll_place", "broll_remove"])
def test_each_operation_validates_and_updates_resolved_lane(
    broll_project: Path, element: str, kind: str,
) -> None:
    root = broll_project
    agent = (root / "timeline/agent.json").read_bytes()
    before = _element(root, element)
    fields = {}
    if kind == "broll_media":
        replacement = root / "replacement.mov"
        replacement.write_bytes((root / "source.mov").read_bytes())
        fields["media"] = {"path": replacement.name, "sha256": sha256_file(replacement),
                           "in_s": 1, "speed": 0.75}
    elif kind == "broll_retime":
        fields = {"start": {"word": "t1:4"}, "end": {"word": "t1:5"}}
    elif kind == "broll_place":
        fields = {"fit": "contain"}
        if element == "broll-overlay":
            fields.update(placement="side_left", motion="none")
    document = _document(_op(kind, element, **fields))
    assert validate_operations(document) == document
    result = apply_operations(root, document)
    assert result["status"] == "applied"
    assert all(finding["severity"] != "violation" for finding in result["findings"])
    assert (root / "timeline/agent.json").read_bytes() == agent
    if kind == "broll_remove":
        assert element not in {item.id for item in resolve_timeline(root).elements}
    else:
        after = _element(root, element)
        if kind == "broll_media":
            assert after.payload["media"]["path"] == "replacement.mov"
            assert after.payload["media"]["speed"] == 0.75
        elif kind == "broll_retime":
            assert (after.start_frame, after.end_frame) == (126, 168)
            assert after.start_frame != before.start_frame
        else:
            media = after.payload if element == "broll-overlay" else after.payload["media"]
            assert media["fit"] == "contain"
            if element == "broll-overlay":
                assert (after.payload["placement"], after.payload["motion"]) == (
                    "side_left", "none",
                )
    version = _next_version(root)
    assert version.id == "v2"
    assert version.snapshots["timeline"]["lanes"]["broll"]["sha256"] == sha256_file(
        root / "timeline/broll.json")


def test_hold_box_and_crop(broll_project: Path) -> None:
    document = _document(
        _op("broll_retime", start={"word": "t1:4", "segment": "s1"}, hold_s=1),
        {**_op("broll_place", fit="crop", crop=[0, 0, 80, 80]), "id": "crop"},
        {**_op("broll_place", "broll-overlay", placement="box", box=[60, 60, 400, 300],
               fit="cover", motion="none"), "id": "box"},
    )
    apply_operations(broll_project, document)
    insert_element = _element(broll_project)
    assert insert_element.end_frame - insert_element.start_frame == 30
    assert insert_element.payload["media"]["crop"] == [0, 0, 80, 80]
    assert _element(broll_project, "broll-overlay").payload["box_px"] == [60, 60, 400, 300]


@pytest.mark.parametrize("fields", [
    {"kind": "broll_retime", "start": {"word": "bad"}, "hold_s": 1},
    {"kind": "broll_retime", "start": {"time": 0}, "hold_s": 0},
    {"kind": "broll_retime", "start": {"time": 0}, "hold_s": 1, "end": {"time": 2}},
    {"kind": "broll_retime", "start": {"time": 0}},
    {"kind": "broll_media", "media": {"path": "x", "sha256": "bad"}},
    {"kind": "broll_media", "media": {"path": "x", "sha256": "0" * 64, "speed": True}},
    {"kind": "broll_place", "placement": "elsewhere"},
    {"kind": "broll_place", "fit": "crop"},
    {"kind": "broll_place", "motion": "fast"},
    {"kind": "broll_place", "box": [0, 0, 0, 20]},
    {"kind": "broll_place"},
    {"kind": "broll_remove", "extra": True},
    {"kind": "broll_other"},
    {"kind": "broll_remove", "comment": "v1:c2"},
])
def test_invalid_contracts(fields: dict) -> None:
    with pytest.raises(ValidationError):
        validate_operations(_document({"id": "change", "element": "broll-one", **fields}))


@pytest.mark.parametrize("element", ["agent-card", "broll-missing"])
def test_non_broll_element_refused(broll_project: Path, element: str) -> None:
    before = _tree(broll_project, lock=False)
    with pytest.raises(NeedsTimelineError):
        apply_operations(broll_project, _document(_op("broll_remove", element)))
    assert _tree(broll_project, lock=False) == before


@pytest.mark.parametrize("dry_run", [False, True])
def test_changed_lane_refuses_with_plain_feedback_error(broll_project: Path, dry_run: bool) -> None:
    source = broll_project / "timeline/broll.json"
    source.write_bytes(source.read_bytes() + b" ")
    before = _tree(broll_project, lock=False)
    with pytest.raises(FeedbackError, match="broll.json changed") as caught:
        apply_operations(broll_project, _document(_op("broll_remove")), dry_run=dry_run)
    assert type(caught.value) is FeedbackError
    assert _tree(broll_project, lock=False) == before


def test_stale_base_refuses_before_lane_hash(broll_project: Path) -> None:
    _next_version(broll_project)
    source = broll_project / "timeline/broll.json"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(FeedbackError, match=r"base v1 is not the current version \(v2\)"):
        apply_operations(broll_project, _document(_op("broll_remove")))


def test_media_hash_must_match_file(broll_project: Path) -> None:
    before = _tree(broll_project, lock=False)
    with pytest.raises(FeedbackError, match="sha256 does not match"):
        apply_operations(broll_project, _document(_op(
            "broll_media", media={"path": "source.mov", "sha256": "0" * 64},
        )))
    assert _tree(broll_project, lock=False) == before


@pytest.mark.parametrize("element", ["broll-one", "broll-overlay"])
@pytest.mark.parametrize("kind", ["broll_remove", "broll_media", "broll_retime", "broll_place"])
def test_keep_declines_operation_and_names_comment(
    broll_project: Path, element: str, kind: str,
) -> None:
    root = broll_project
    start = 69 if element == "broll-one" else 189
    batch = build_batch(root, _review_export(root, [
        _comment("c1", target="element", start_frame=start),
        _comment("c4", target="element", kind="keep", start_frame=start),
    ]))
    fields = {}
    if kind == "broll_media":
        fields = {"media": {"path": "source.mov", "sha256": sha256_file(root / "source.mov")}}
    elif kind == "broll_retime":
        fields = {"start": {"word": "t1:4"}, "hold_s": 1.2}
    elif kind == "broll_place":
        fields = {"fit": "contain"}
    document = _document(_op(kind, element, **fields))
    document["batch"] = batch["id"]
    before = _tree(root)
    preview = apply_operations(root, document, dry_run=True)
    assert _tree(root) == before
    assert preview["dispositions"][0]["keep"]["id"] == "c4"
    lane = (root / "timeline/broll.json").read_bytes()
    apply_operations(root, document)
    item = load_batch(root, batch["id"])["items"][0]
    assert item["disposition"] == "declined" and item["operations"] == []
    assert item["evidence"]["protected_by"]["id"] == "c4"
    assert "v1:c4" in item["evidence"]["reason"]
    assert (root / "timeline/broll.json").read_bytes() == lane


def test_kept_comment_with_invalid_story_edit_does_not_block_other_fix(broll_project: Path) -> None:
    root = broll_project
    batch = build_batch(root, _review_export(root, [
        _comment("c1", target="element", start_frame=69),
        _comment("c2", target="element", start_frame=189),
        _comment("c4", target="element", kind="keep", start_frame=69),
    ]))
    invalid = {"id": "bad-story", "kind": "remove", "segment": "missing"}
    document = _document(_op("broll_remove"), invalid)
    document["batch"] = batch["id"]
    document["items"].append({"comment": "v1:c2", "operations": [{
        "id": "valid-place", "kind": "broll_place", "element": "broll-overlay", "fit": "contain",
    }]})
    protected = _element(root).to_dict()
    before = _tree(root)
    preview = apply_operations(root, document, dry_run=True)
    assert _tree(root) == before
    assert {item["operation"] for item in preview["changes"]
            if item.get("disposition") == "declined"} == {"change", "bad-story"}
    result = apply_operations(root, document)
    assert result["status"] == "applied"
    assert _element(root).to_dict() == protected
    assert _element(root, "broll-overlay").payload["fit"] == "contain"
    by_id = {item["comment"]["id"]: item for item in load_batch(root, batch["id"])["items"]}
    assert by_id["c1"]["disposition"] == "declined" and by_id["c1"]["operations"] == []
    assert by_id["c1"]["evidence"]["protected_by"]["id"] == "c4"
    assert by_id["c2"]["disposition"] == "applied"


def test_dry_run_writes_no_project_bytes(broll_project: Path) -> None:
    before = _tree(broll_project)
    operation = _op("broll_place", "broll-overlay", fit="contain")
    result = apply_operations(broll_project, _document(operation), dry_run=True)
    assert result["status"] == "planned"
    assert all(finding["severity"] != "violation" for finding in result["findings"])
    assert result["broll_operations"] == [{**operation, "comment": "v1:c1"}]
    assert _tree(broll_project) == before


def test_rule_violation_refuses_entire_apply_and_dry_run_reports_it(broll_project: Path) -> None:
    document = _document(_op("broll_retime", start={"word": "t1:6"}, hold_s=1.2))
    before = _tree(broll_project)
    result = apply_operations(broll_project, document, dry_run=True)
    assert result["status"] == "refused"
    assert any(finding["rule"] == "visual_overlap" for finding in result["findings"])
    assert all(str(broll_project) in finding["suggested_action"] for finding in result["findings"])
    assert _tree(broll_project) == before
    with pytest.raises(FeedbackError, match="B-roll timeline refused"):
        apply_operations(broll_project, document)
    assert _tree(broll_project, lock=False) == {
        key: value for key, value in before.items() if key != ".yap/lock"
    }
    assert "application" not in load_batch(broll_project, "b1")


def test_crash_after_lane_write_resumes_once(broll_project: Path, monkeypatch) -> None:
    original = apply_module._write_batch
    document = _document(_op("broll_remove"))

    def crash_on_completion(root, batch):
        if batch["application"]["status"] == "applied":
            raise RuntimeError("crash after lane publication")
        return original(root, batch)

    monkeypatch.setattr(apply_module, "_write_batch", crash_on_completion)
    with pytest.raises(RuntimeError, match="crash after lane publication"):
        apply_operations(broll_project, document)
    pending = load_batch(broll_project, "b1")["application"]
    assert pending["status"] == "pending" and not pending["broll"]["done"]
    after = (broll_project / "timeline/broll.json").read_bytes()
    monkeypatch.setattr(apply_module, "_write_batch", original)
    monkeypatch.setattr(apply_module, "plan_broll",
                        lambda *args, **kwargs: pytest.fail("replanned"))
    assert apply_operations(broll_project, document)["status"] == "applied"
    assert (broll_project / "timeline/broll.json").read_bytes() == after
    item = next(item for item in load_batch(broll_project, "b1")["items"]
                if item["comment"]["id"] == "c1")
    assert len(item["operations"]) == 1


def test_pending_journal_refuses_changed_lane_and_stale_base_first(
    broll_project: Path, monkeypatch,
) -> None:
    original = apply_module.publish_broll
    monkeypatch.setattr(apply_module, "publish_broll",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("crash before lane")))
    document = _document(_op("broll_remove"))
    with pytest.raises(RuntimeError):
        apply_operations(broll_project, document)
    monkeypatch.setattr(apply_module, "publish_broll", original)
    source = broll_project / "timeline/broll.json"
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(FeedbackError, match="broll.json changed"):
        apply_operations(broll_project, document)
    _next_version(broll_project)
    with pytest.raises(FeedbackError, match="base v1 is not the current version"):
        apply_operations(broll_project, document)


@pytest.mark.parametrize(("kind", "identity", "start"), [
    ("insert", "broll-one", 69), ("overlay", "broll-overlay", 189),
])
def test_broll_clarification_names_kind_and_id(
    broll_project: Path, kind: str, identity: str, start: int,
) -> None:
    from fractions import Fraction

    from yap_studio.feedback.versions import load_version, version_layers, version_pieces

    version = load_version(broll_project, "v1")
    comment = _comment(start_frame=start)
    location = locate(comment, fps=Fraction(30), duration_s=12,
                      words=version_cut_words(version), pieces=version_pieces(version),
                      layers=version_layers(version))
    question = automatic_clarification(comment, "v1", location)
    assert {"value": "element", "label": f"The {kind} {identity}"} in question["options"]


def test_keep_cannot_be_used_as_fix_comment(broll_project: Path) -> None:
    with pytest.raises(ValidationError, match="require a fix comment"):
        apply_operations(broll_project, _document(_op("broll_remove"), comment="v1:c3"))


def test_insert_cannot_accept_overlay_placement(broll_project: Path) -> None:
    with pytest.raises(ValidationError, match="only fit and crop"):
        apply_operations(broll_project, _document(_op("broll_place", placement="headroom")))


def test_other_kind_in_broll_lane_refused_with_named_error(broll_project: Path) -> None:
    source = broll_project / "timeline/broll.json"
    lane = json.loads(source.read_bytes())
    lane["elements"][0] = {**graphic(), "id": "broll-one", "lane": "broll"}
    _json(source, lane)
    from yap_studio.feedback.broll import plan_broll

    with pytest.raises(NeedsTimelineError):
        plan_broll(broll_project, ({**_op("broll_remove"), "comment": "v1:c1"},),
                   edit={}, story_changed=False, removal_actions=())


def test_operation_document_is_not_mutated(broll_project: Path) -> None:
    document = _document(_op("broll_remove", comment="v1:c1"))
    expected = copy.deepcopy(document)
    apply_operations(broll_project, document)
    assert document == expected


def test_story_and_broll_changes_share_journal_and_resolve_next_cut(broll_project: Path) -> None:
    from yap_studio.stages import cut

    root = broll_project
    before = _element(root)
    document = _document(
        {"id": "trim", "kind": "trim", "segment": "s1", "edge": "start", "word": "t1:1"},
        _op("broll_place", fit="contain"),
    )
    preview = apply_operations(root, document, dry_run=True)
    assert preview["status"] == "planned"
    assert apply_operations(root, document)["status"] == "applied"
    journal = load_batch(root, "b1")["application"]
    assert journal["edit"]["segments"][0]["from"] == {"word": "t1:1"}
    assert journal["broll"]["done"] and journal["status"] == "applied"
    plan = cut.resolve_cut(root)
    _json(root / "cut/cut-words.json", cut._cut_words(plan))
    _json(root / "cut/spans.json", cut._spans(plan, root / "cut"))
    receipt = json.loads((root / "cut/receipt.json").read_bytes())
    receipt.update(inputs=cut.cut_inputs(plan.context), plan=plan.to_dict())
    for output in receipt["outputs"]:
        output["sha256"] = sha256_file(root / output["path"])
    _json(root / "cut/receipt.json", receipt)
    after = _element(root)
    assert after.start_frame < before.start_frame
    assert after.payload["media"]["fit"] == "contain"


def test_invalid_next_story_timeline_refuses_before_story_write(broll_project: Path) -> None:
    document = _document(
        {"id": "trim", "kind": "trim", "segment": "s1", "edge": "end", "word": "t1:5"},
        _op("broll_place", fit="contain"),
    )
    before = _tree(broll_project, lock=False)
    with pytest.raises(FeedbackError, match="B-roll timeline refused"):
        apply_operations(broll_project, document)
    assert _tree(broll_project, lock=False) == before


def test_box_only_sets_overlay_box_placement(broll_project: Path) -> None:
    apply_operations(broll_project, _document(_op(
        "broll_place", "broll-overlay", box=[60, 60, 400, 300],
    )))
    assert _element(broll_project, "broll-overlay").payload["placement"] == "box"


def test_feedback_stock_swap_refused_until_approved(broll_project: Path) -> None:
    import shutil

    from test_timeline import approve_stock, stock_style

    from yap_studio.stages.timeline import TimelineRefusal
    from yap_studio.style.library import config_root

    root = broll_project
    stock_style(root)
    cache = config_root() / "assets/stock/pexels/101.mp4"
    cache.parent.mkdir(parents=True)
    shutil.copyfile(root / "source.mov", cache)
    digest = sha256_file(cache)
    # The folder exists before the feedback shadow is built, but has no approval yet.
    _json(root / "stock/clips.json", {"schema": "yap-studio/stock-clips@1", "clips": []})
    document = _document(_op("broll_media", media={"path": str(cache), "sha256": digest}))
    before = (root / "timeline/broll.json").read_bytes()
    with pytest.raises(TimelineRefusal, match="not approved"):
        apply_operations(root, document)
    assert (root / "timeline/broll.json").read_bytes() == before
    approve_stock(root, digest)
    assert apply_operations(root, document)["status"] == "applied"
    assert _element(root).payload["media"]["sha256"] == digest
