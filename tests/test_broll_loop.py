"""Catalog proposals, rendered feedback versions and approved B-roll defaults."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth
import synth_broll
import test_assemble as assembly_tests
import test_broll_e2e as broll_tests
from test_assemble_picture import overlay_image
from test_cut import write_json

from yap_studio import ffmpeg
from yap_studio.checks.caption_layout import read_layout_record
from yap_studio.faces.model import Box
from yap_studio.feedback.apply import apply_operations
from yap_studio.feedback.batch import build_batch, load_batch
from yap_studio.feedback.edits import OPERATIONS_SCHEMA
from yap_studio.feedback.taste import ProposalInbox
from yap_studio.feedback.versions import LAYERS_SCHEMA, load_version, save_version
from yap_studio.media.catalog import add_paths, list_items
from yap_studio.media.describe import DESCRIPTION_SCHEMA, describe_item
from yap_studio.planner.broll import (
    REQUESTS_SCHEMA,
    load_requests,
    plan_broll,
    suggest_broll,
    write_broll_plan,
)
from yap_studio.project.model import load_cut_words
from yap_studio.project.receipts import read_receipt
from yap_studio.project.status import ProjectContext
from yap_studio.receipt import sha256_file
from yap_studio.review.export import to_export_json
from yap_studio.review.model import MediaFacts
from yap_studio.review.store import ReviewStore
from yap_studio.stages.assemble import assemble
from yap_studio.stages.captions import captions
from yap_studio.stages.cut import resolve_cut
from yap_studio.stages.graphics import graphics
from yap_studio.stages.music import music
from yap_studio.stages.timeline import add_timeline, resolve_timeline
from yap_studio.taste import decide, inbox_view, load_taste, propose, read_taste_record

DESCRIPTIONS = {
    "clip.mov": ("First scene beside a lake.", ["scenery"]),
    "Screen Recording.mov": ("First steps in a terminal.", ["first", "terminal"]),
    "photo.jpg": ("A photograph of words on a sign.", ["words", "sign"]),
    "Screenshot.png": ("Words in a settings dialog.", ["dialog"]),
}


@pytest.fixture(scope="module")
def catalog_project(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest,
) -> Iterator[Path]:
    workspace = tmp_path_factory.mktemp("broll-loop")
    with pytest.MonkeyPatch.context() as patch:
        # Use the existing transparent layer and unavailable Vision stubs verbatim.
        broll_tests.synthetic_dependencies.__wrapped__(workspace, patch, request)
        assembly_tests.refuse_unexpected_matte_work.__wrapped__(patch)
        # No end hold: saved feedback media must have the cut's frame count.
        root = assembly_tests.project.__wrapped__(workspace, SimpleNamespace(param=0))
        receipt = read_receipt(root / "cut")
        receipt["plan"] = resolve_cut(root).to_dict()
        write_json(root / "cut/receipt.json", receipt)
        folder = root / "broll"
        folder.mkdir()
        synth_broll.clip(folder / "clip.mov", colour="blue", duration=2)
        synth_broll.clip(folder / "Screen Recording.mov", colour="lime", duration=2)
        overlay_image(folder / "photo.jpg")
        synth_broll.png(folder / "Screenshot.png")
        added = add_paths([folder])
        assert len(added["added"]) == 4 and added["skipped"] == []
        for item in list_items():
            description, tags = DESCRIPTIONS[Path(item["locations"][0]["path"]).name]
            describe_item(item["id"], {
                "schema": DESCRIPTION_SCHEMA, "description": description, "tags": tags,
            })
        yield root


@pytest.fixture(scope="module")
def catalog_items(catalog_project: Path) -> list[dict]:
    return list_items()


@pytest.fixture(scope="module")
def suggestions(catalog_project: Path, catalog_items: list[dict]) -> dict:
    return suggest_broll(
        load_cut_words(catalog_project / "cut/cut-words.json"), catalog_items,
        ProjectContext(catalog_project).render_style().data, phrase_words=1, limit=4,
    )


@pytest.fixture(scope="module")
def rendered_project(catalog_project: Path, catalog_items: list[dict], suggestions: dict) -> Path:
    return _render_requests(catalog_project, catalog_items, suggestions)


def _render_requests(root: Path, catalog_items: list[dict], suggestions: dict, *,
                     offset_s: float | None = None) -> Path:
    first, words = suggestions["phrases"]
    screen, photo = first["candidates"][0], words["candidates"][0]
    starts = [dict(phrase["start"]) for phrase in (first, words)]
    if offset_s is not None:
        for start in starts:
            start["offset_s"] = offset_s
    requests = root / "broll-requests.json"
    write_json(requests, {"schema": REQUESTS_SCHEMA, "broll": [{
        "id": "broll-insert", "item": screen["item"], "kind": screen["kind"],
        "start": starts[0], "hold_s": 0.5, "fit": "cover",
        "matches": first["text"],
    }, {
        "id": "broll-photo", "item": photo["item"], "kind": photo["kind"],
        "start": starts[1], "hold_s": 1, "placement": "box",
        "box": [60, 400, 400, 200], "fit": "cover", "motion": "none",
        "matches": words["text"],
    }]})
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        plan = plan_broll(
            load_requests(requests), load_cut_words(root / "cut/cut-words.json"),
            catalog_items, ProjectContext(root).render_style().data, project_dir=root, frames=120,
        )
        outputs = write_broll_plan(plan, root / "broll-plan")
    add_timeline(root, outputs["broll-lane.json"], lane="broll").require_valid()
    _render(root)
    return root


def _render(root: Path) -> Path:
    captions(root)
    graphics(root)
    music(root)
    return root / assemble(root)["out"]


def _final(root: Path) -> Path:
    return root / "final" / f"{ProjectContext(root).project.name}.mp4"


def _save_render(root: Path):
    timeline = resolve_timeline(root).require_valid()
    layers = root / "layers.json"
    write_json(layers, {"schema": LAYERS_SCHEMA, "time_base": "output", "layers": [
        {"id": item.id, "kind": item.kind, "start": item.start_s,
         "end": item.end_s, "text": None} for item in timeline.elements
    ]})
    return save_version(root, media=_final(root), layers=layers)


@pytest.fixture(scope="module")
def corrected_project(
    rendered_project: Path, tmp_path_factory: pytest.TempPathFactory,
    catalog_items: list[dict], suggestions: dict,
) -> tuple[Path, ReviewStore, dict]:
    workspace = tmp_path_factory.mktemp("broll-feedback")
    root = shutil.copytree(rendered_project, workspace / "project")
    _render_requests(root, catalog_items, suggestions, offset_s=0)
    first = _save_render(root)
    review = ReviewStore(workspace / "review")
    facts = MediaFacts(first.media_path.name, first.media["sha256"], 4, Fraction(30), True)
    for kind, frame, text in (
        ("fix", 24, "Move this photo to the right."),
        ("keep", 9, "Keep this screen recording exactly as it is."),
        ("fix", 9, "Remove this screen recording."),
    ):
        review.add_comment(facts.sha256, facts, {
            "kind": kind, "target": "element", "start_frame": frame, "text": text,
        })
    batch = build_batch(root, to_export_json(
        facts, review.comments(facts.sha256), None, datetime.now(UTC),
    ))
    result = apply_operations(root, {"schema": OPERATIONS_SCHEMA, "batch": batch["id"],
                                    "items": [{
        "comment": "v1:c1", "operations": [{
            "id": "move-photo", "kind": "broll_place", "element": "broll-photo",
            "placement": "side_right", "motion": "none",
        }],
    }, {"comment": "v1:c2", "operations": [{"id": "keep-insert", "kind": "keep"}]}, {
        "comment": "v1:c3", "operations": [{
            "id": "remove-kept-insert", "kind": "broll_remove", "element": "broll-insert",
        }],
    }]})
    _render(root)
    _save_render(root)
    return root, review, result


def test_catalog_descriptions_tags_and_media_durations(catalog_items: list[dict]) -> None:
    items = {Path(item["locations"][0]["path"]).name: item for item in catalog_items}
    assert items.keys() == DESCRIPTIONS.keys()
    for name, media_type in (
        ("clip.mov", "clip"), ("Screen Recording.mov", "screen_recording"),
        ("photo.jpg", "photo"), ("Screenshot.png", "screenshot"),
    ):
        item = items[name]
        description, tags = DESCRIPTIONS[name]
        assert item["media_type"] == media_type
        assert item["description"]["text"] == description
        assert item["description"]["tags"] == tags
        if item["kind"] == "video":
            assert item["duration_s"] == pytest.approx(2)
        else:
            assert item["kind"] == "image" and item["duration_s"] is None


def test_suggestions_rank_matching_descriptions_and_anchor_words(
    suggestions: dict, catalog_items: list[dict],
) -> None:
    items = {item["id"]: item for item in catalog_items}
    assert suggestions["schema"] == "yap-studio/broll-suggestions@1"
    assert suggestions["excluded"] == []
    for index, (phrase, text, matched, media_types, kind) in enumerate(zip(
        suggestions["phrases"], ("First", "words"), ("first", "word"),
        (("screen_recording", "clip"), ("photo", "screenshot")),
        ("insert", "overlay"), strict=True,
    )):
        assert phrase["text"] == text and phrase["word_ids"] == [f"t1:{index}"]
        for edge in ("start", "end"):
            assert phrase[edge] == {
                "word": f"t1:{index}", "text": text, "segment": "s1", "edge": edge,
            }
        candidates = phrase["candidates"]
        assert [items[item["item"]]["media_type"] for item in candidates] == list(media_types)
        assert candidates[0]["score"] > candidates[1]["score"]
        for candidate in candidates:
            assert candidate["kind"] == kind and candidate["matched"] == [matched]
            assert f"matches description: {matched}" in candidate["reasons"]
        assert f"matches tags: {matched}" in candidates[0]["reasons"]


def _assert_insert_frames(final: Path, cut: Path, *, frames: tuple = (2, 3, 17, 18),
                          lead_frames: int = 0) -> None:
    for frame in frames:
        box = (700, 700, 16, 16)
        actual = synth.picture_pixels(final, frame, box=box).mean(axis=(0, 1))
        expected = (0, 255, 0) if 3 - lead_frames <= frame < 18 - lead_frames else \
            synth.picture_pixels(cut, frame, box=box).mean(axis=(0, 1))
        np.testing.assert_allclose(actual, expected, atol=10)


def _assert_overlay_frames(final: Path, cut: Path, *, inside: tuple, outside: tuple,
                           lead_frames: int = 0) -> None:
    for frame, alpha in ((17, 0), (18, 0), (19, 1 / 6), (24, 1), (41, 1), (47, 1 / 6), (48, 0)):
        frame -= lead_frames
        base = np.array((0, 255, 0)) if frame == 17 - lead_frames else \
            synth.picture_pixels(cut, frame, box=inside).mean(axis=(0, 1))
        expected = alpha * np.array((255, 0, 0)) + (1 - alpha) * base
        actual = synth.picture_pixels(final, frame, box=inside).mean(axis=(0, 1))
        np.testing.assert_allclose(actual, expected, atol=10)
        expected_outside = np.array((0, 255, 0)) if frame == 17 - lead_frames else \
            synth.picture_pixels(cut, frame, box=outside).mean(axis=(0, 1))
        actual_outside = synth.picture_pixels(final, frame, box=outside).mean(axis=(0, 1))
        np.testing.assert_allclose(actual_outside, expected_outside, atol=10)


def test_requests_plan_fragment_and_render_exact_frames(rendered_project: Path) -> None:
    root = rendered_project
    summary = json.loads((root / "broll-plan/broll-plan.json").read_bytes())
    assert summary["status"] == "ok" and summary["warnings"] == []
    assert summary["outputs"][0]["sha256"] == sha256_file(root / "broll-plan/broll-lane.json")
    timeline = resolve_timeline(root).require_valid()
    # Each start leads its named word by 0.1 s.
    assert [(item.id, item.kind, item.start_frame, item.end_frame)
            for item in timeline.elements] == [
        ("broll-insert", "insert", 0, 15), ("broll-photo", "overlay", 15, 45),
    ]
    final, cut = _final(root), root / "cut/cut.mp4"
    assert read_receipt(root / "final")["status"] == "verified"
    _assert_insert_frames(final, cut, frames=(0, 2, 14, 15, 18), lead_frames=3)
    _assert_overlay_frames(final, cut, inside=(200, 450, 16, 16), outside=(500, 450, 16, 16),
                           lead_frames=3)
    record = read_layout_record(_final(root), width=1080, height=1920)
    (overlay,) = [item for item in record.layout.elements if item.kind == "overlay"]
    assert (overlay.id, overlay.start_frame, overlay.end_frame, overlay.box) == (
        "broll-photo", 17, 44, Box(60, 400, 400, 200),
    )
    assert ffmpeg.probe(final).video_frames == 120


def test_feedback_moves_photo_and_keeps_insert_in_next_render(corrected_project: tuple) -> None:
    root, _, result = corrected_project
    assert result["status"] == "applied"
    assert result["dispositions"][0]["keep"]["id"] == "c2"
    items = {item["comment"]["id"]: item for item in load_batch(root, "b1")["items"]}
    assert items["c1"]["disposition"] == items["c2"]["disposition"] == "applied"
    assert items["c3"]["disposition"] == "declined" and items["c3"]["operations"] == []
    assert items["c3"]["evidence"]["protected_by"]["id"] == "c2"
    assert "v1:c2" in items["c3"]["evidence"]["reason"]
    first, second = (load_version(root, version) for version in ("v1", "v2"))
    assert second.parent == first.id and second.media["sha256"] != first.media["sha256"]
    first_lane = json.loads((first.path / "timeline/broll.json").read_bytes())
    second_lane = json.loads((root / "timeline/broll.json").read_bytes())
    assert second_lane["elements"][0] == first_lane["elements"][0]
    assert second_lane["elements"][1]["placement"] == "side_right"
    assert second.snapshots["timeline"]["lanes"]["broll"]["sha256"] == sha256_file(
        root / "timeline/broll.json")
    final, cut = second.media_path, root / "cut/cut.mp4"
    assert read_receipt(root / "final")["status"] == "verified"
    _assert_insert_frames(final, cut)
    _assert_overlay_frames(final, cut, inside=(700, 900, 16, 16), outside=(200, 450, 16, 16))
    for frame in (3, 9, 17):
        np.testing.assert_allclose(synth.picture_pixels(final, frame, box=(700, 700, 16, 16)),
                                   synth.picture_pixels(first.media_path, frame,
                                                        box=(700, 700, 16, 16)), atol=10)
    record = read_layout_record(_final(root), width=1080, height=1920)
    (overlay,) = [item for item in record.layout.elements if item.kind == "overlay"]
    assert (overlay.id, overlay.start_frame, overlay.end_frame, overlay.box) == (
        "broll-photo", 20, 47, Box(580, 680, 440, 560),
    )


def test_approved_comment_taste_defaults_photo_to_full_frame_insert(
    corrected_project: tuple, catalog_items: list[dict], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous, review, _ = corrected_project
    root = shutil.copytree(previous, tmp_path / "project")
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    body = {
        "scope": "broll", "sentence": "Show photos full frame by default.",
        "machine": {"kind": "style_field", "field": "broll.photo_placement",
                    "value": "full_frame"},
        "comments": [{"version": "v1", "id": "c1"}],
    }
    (proposal,) = propose(root, body, store=review)
    assert proposal["status"] == "added"
    source = ProposalInbox(root).read()[0]["sources"][0]
    assert source["comment_id"] == "c1" and source["label"] == "fix"
    assert source["quote"] == "Move this photo to the right."
    assert source["media_sha256"] == load_version(root, "v1").media["sha256"]
    assert ProjectContext(root).render_style().data["broll"]["photo_placement"] == "overlay"
    assert load_taste().revision == 0
    approved = decide(root, proposal["id"], "approve")
    assert approved["state"] == "approved" and load_taste().revision == 1
    assert inbox_view(root)["proposals"][0]["state"] == "approved"
    style = ProjectContext(root).render_style().data
    assert style["broll"]["photo_placement"] == "full_frame"
    words = load_cut_words(root / "cut/cut-words.json")
    suggestion = suggest_broll(words, catalog_items, style, phrase_words=1, limit=4)
    phrase = suggestion["phrases"][1]
    photo = phrase["candidates"][0]
    assert photo["media_type"] == "photo"
    assert (photo["kind"], photo["placement"], photo["box"]) == ("insert", "full_frame", None)
    requests = {"schema": REQUESTS_SCHEMA, "broll": [{
        "id": "broll-photo", "item": photo["item"],
        "start": {**phrase["start"], "offset_s": 0},
        "hold_s": 1,
    }]}
    plan = plan_broll(requests, words, catalog_items, style, project_dir=root, frames=120)
    (element,) = plan.lane()["elements"]
    assert element["kind"] == "insert" and element["audio"] == "mute"
    assert "placement" not in element and "box" not in element
    assert (plan.broll[0]["start_frame"], plan.broll[0]["end_frame"]) == (18, 48)
    lane = plan.lane()
    fragment = root / "approved-broll.json"
    write_json(fragment, lane)
    add_timeline(root, fragment, lane="broll").require_valid()
    resolved = resolve_timeline(root).require_valid()
    assert [(item.kind, item.start_frame, item.end_frame) for item in resolved.elements] == [
        ("insert", 3, 18), ("insert", 18, 48),
    ]
    final = _render(root)
    for frame in (18, 24, 47, 48):
        box = (200, 900, 16, 16)
        actual = synth.picture_pixels(final, frame, box=box).mean(axis=(0, 1))
        expected = (255, 0, 0) if frame < 48 else synth.picture_pixels(
            root / "cut/cut.mp4", frame, box=box,
        ).mean(axis=(0, 1))
        np.testing.assert_allclose(actual, expected, atol=10)
    _assert_insert_frames(final, root / "cut/cut.mp4", frames=(2, 3, 17))
    receipt = read_receipt(root / "final")
    assert receipt["status"] == "verified"
    record = read_taste_record(final)
    assert record["revision"] == 1
    assert record["applied"] == [{"id": proposal["id"], "field": "broll.photo_placement"}]
    assert record["media_sha256"] == sha256_file(final)
    assert receipt["taste"] == ProjectContext(root).taste_application().pin()
    assert record["applied"] == receipt["taste"]["applied"]
    assert any(item["path"] == final.with_suffix(".taste.json").relative_to(root).as_posix()
               and item["sha256"] == sha256_file(final.with_suffix(".taste.json"))
               for item in receipt["outputs"])
