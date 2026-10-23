"""B-roll suggestions stay pure; only explicit requests become a valid fragment."""

from __future__ import annotations

import contextlib
import copy
import json
from collections import Counter
from pathlib import Path

import pytest
from test_media_select import STAMP, _decision, _item, _speech
from test_planner_anchors import synthetic_words

from yap_studio import paths
from yap_studio.errors import OutputPathError, ValidationError
from yap_studio.planner import broll
from yap_studio.project.model import CutWord, CutWords, parse_document
from yap_studio.receipt import sha256_file
from yap_studio.style import resolve
from yap_studio.taste import add_proposals, apply_taste, approve


def phrase_words(*texts: str) -> CutWords:
    return CutWords(
        "output",
        {"t1": "a" * 64},
        tuple(
            CutWord(f"t1:{index}", "story", text, index * 0.5, (index + 1) * 0.5)
            for index, text in enumerate(texts or ("Dog", "park."))
        ),
    )


def requests(**changes: object) -> dict:
    return {
        "schema": broll.REQUESTS_SCHEMA,
        "broll": [
            {
                "id": "broll-dog",
                "item": "m1",
                "start": {"word": "t1:0", "text": "Dog"},
                "end": {"word": "t1:1", "text": "park."},
                **changes,
            }
        ],
    }


def _project(item: dict) -> Path:
    return Path(item["locations"][-1]["path"]).parent


def catalog_item(tmp_path: Path, number: int = 1, **fields) -> dict:
    source = tmp_path / f"candidate-{number}.mov"
    source.write_bytes(f"catalog media {number}".encode())
    item = _item(
        number, locations=[{"path": str(source), "added_at": STAMP, "seen_at": STAMP}], **fields
    )
    item["sha256"] = sha256_file(source)
    return item


@pytest.mark.parametrize("frames,motion", [(30, "none"), (62, "none"), (63, "scroll")])
def test_default_scroll_fits_resolved_window(tmp_path: Path, frames: int, motion: str) -> None:
    item = catalog_item(tmp_path, kind="image")
    item.update(width=620, height=2400)
    style = {"broll": {"photo_placement": "overlay", "overlay_motion": "scroll"}}
    request = requests(start={"time": 0}, end={"time": frames / 30}, kind="overlay")
    plan = broll.plan_broll(request, phrase_words(), [item], style,
                           project_dir=tmp_path, frames=90)
    assert plan.lane()["elements"][0]["motion"] == motion
    request["broll"][0]["motion"] = "scroll"
    explicit = broll.plan_broll(request, phrase_words(), [item], style,
                               project_dir=tmp_path, frames=90)
    assert explicit.lane()["elements"][0]["motion"] == "scroll"


@pytest.mark.parametrize("offset,motion", [(None, "scroll"), (0, "none")])
def test_default_scroll_uses_window_after_named_lead(tmp_path: Path, offset, motion: str) -> None:
    item = catalog_item(tmp_path, kind="image")
    item.update(width=620, height=2400)
    start = {"word": "t1:1", "text": "park"}
    if offset is not None:
        start["offset_s"] = offset
    data = requests(kind="overlay", start=start, end={"time": 2.5}, matches="park")
    style = {"broll": {"overlay_motion": "scroll"}}
    plan = broll.plan_broll(data, phrase_words("Dog", "park"), [item], style,
                           project_dir=tmp_path, frames=90)
    assert plan.lane()["elements"][0]["motion"] == motion
    duration = plan.broll[0]["end_frame"] - plan.broll[0]["start_frame"]
    assert duration == (63 if offset is None else 60)


def test_short_phrase_suggests_still_page(tmp_path: Path) -> None:
    item = catalog_item(tmp_path, kind="image")
    item.update(width=620, height=2400)
    style = {"broll": {"photo_placement": "overlay", "overlay_motion": "scroll"}}
    suggestion = broll.suggest_broll(phrase_words(), [item], style)["phrases"][0]["candidates"][0]
    assert suggestion["motion"] == suggestion["request"]["motion"] == "none"


@pytest.mark.parametrize("frames,box,motion", [
    (30, "headroom", "none"), (74, "headroom", "scroll"),
    (75, "tall_behind", "scroll"), (120, "tall_behind", "scroll"),
    (240, "tall_behind", "scroll"), (241, "headroom", "scroll"),
    (270, "headroom", "scroll"),
])
def test_implicit_tall_box_fits_plan_and_phrase_windows(
    tmp_path: Path, frames: int, box: str, motion: str,
) -> None:
    item = catalog_item(tmp_path, kind="image")
    item.update(width=1000, height=2400, media_type="screenshot")
    style = {"broll": {"overlay_box": "tall_behind", "overlay_motion": "scroll"}}
    words = CutWords("output", {}, (CutWord("t1:0", "story", "Dog", 0, frames / 30),))
    request = requests(start={"time": 0}, end={"time": frames / 30})
    plan = broll.plan_broll(request, words, [item], style,
                           project_dir=tmp_path, frames=300)
    assert plan.status == "ok"
    element = plan.lane()["elements"][0]
    assert (element["placement"], element["motion"]) == (box, motion)
    candidate = broll.suggest_broll(words, [item], style)["phrases"][0]["candidates"][0]
    assert (candidate["box"], candidate["motion"]) == (box, motion)
    assert (candidate["request"]["box"], candidate["request"]["motion"]) == (box, motion)


@pytest.mark.parametrize("field", ["box", "placement"])
def test_explicit_short_tall_box_is_preserved(tmp_path: Path, field: str) -> None:
    item = catalog_item(tmp_path, kind="image")
    item.update(width=1000, height=2400, media_type="screenshot")
    style = {"broll": {"overlay_box": "tall_behind", "overlay_motion": "scroll"}}
    plan = broll.plan_broll(requests(**{field: "tall_behind"}), phrase_words(), [item], style,
                           project_dir=tmp_path, frames=90)
    element = plan.lane()["elements"][0]
    assert (element["placement"], element["motion"]) == ("tall_behind", "none")


@pytest.mark.parametrize("end,offset,box", [
    (2.9, None, "tall_behind"), (2.9, 0, "headroom"),
    (8.5, None, "headroom"), (8.5, 0, "tall_behind"),
])
def test_implicit_tall_box_uses_window_after_named_lead(
    tmp_path: Path, end: float, offset: int | None, box: str,
) -> None:
    item = catalog_item(tmp_path, kind="image")
    item.update(width=1000, height=2400, media_type="screenshot")
    start = {"word": "t1:1", "text": "park"}
    if offset is not None:
        start["offset_s"] = offset
    request = requests(start=start, end={"time": end}, matches="park")
    plan = broll.plan_broll(request, phrase_words("Dog", "park"), [item],
                           {"broll": {"overlay_box": "tall_behind"}},
                           project_dir=tmp_path, frames=300)
    assert plan.lane()["elements"][0]["placement"] == box
    assert plan.broll[0]["start_frame"] == (12 if offset is None else 15)


def test_fallback_box_determines_default_motion(tmp_path: Path) -> None:
    item = catalog_item(tmp_path, kind="image")
    item.update(width=1000, height=2400, media_type="screenshot")
    style = {"broll": {"overlay_box": "tall_behind", "overlay_motion": "scroll"},
             "internal": {"key_text_bottom_px": 3000}}
    request = requests(start={"time": 0}, end={"time": 9})
    plan = broll.plan_broll(request, phrase_words(), [item], style,
                           project_dir=tmp_path, frames=300)
    element = plan.lane()["elements"][0]
    assert (element["placement"], element["motion"]) == ("headroom", "none")
    words = CutWords("output", {}, (CutWord("t1:0", "story", "Dog", 0, 9),))
    candidate = broll.suggest_broll(words, [item], style)["phrases"][0]["candidates"][0]
    assert (candidate["box"], candidate["motion"]) == ("headroom", "none")


@pytest.mark.parametrize("fields,field", [({"box": [0, 0, 50]}, "box"),
                                         ({"placement": "unknown"}, "placement")])
def test_default_scroll_validates_geometry_before_resolving(
    tmp_path: Path, fields: dict, field: str,
) -> None:
    item = catalog_item(tmp_path, kind="image")
    item.update(width=620, height=2400)
    with pytest.raises(ValidationError, match=rf"broll\[0\].{field}"):
        broll.plan_broll(requests(kind="overlay", **fields), phrase_words(), [item],
                         {"broll": {"overlay_motion": "scroll"}},
                         project_dir=tmp_path, frames=90)


@pytest.mark.parametrize("internal,expected", [({}, "none"),
                                              ({"key_text_bottom_px": 300}, "scroll")])
def test_scroll_geometry_uses_schema_defaults_without_loading_a_preset(
    monkeypatch: pytest.MonkeyPatch, internal: dict, expected: str,
) -> None:
    from yap_studio import style

    monkeypatch.setattr(style, "resolve", lambda *args, **kwargs:
                        pytest.fail("Scroll geometry must not resolve a preset"))
    item = {"kind": "image", "width": 1000, "height": 100}
    data = {"broll": {"overlay_motion": "scroll"}, "internal": internal}
    assert broll._suggested_motion(item, {"placement": "headroom"}, data, 75) == expected


@pytest.mark.parametrize("push,motion", [("in", "push"), ("none", "none")])
def test_green_screen_plan_and_suggestion(tmp_path: Path, push: str, motion: str) -> None:
    item = catalog_item(tmp_path, kind="image")
    item["media_type"] = "screenshot"
    style = {"broll": {"screenshot_placement": "green_screen",
                       "green_screen_layout": "bottom", "still_push": push}}
    planned = broll.plan_broll(requests(), phrase_words(), [item], style,
                               project_dir=tmp_path, frames=60).lane()["elements"][0]
    assert {key: planned[key] for key in ("kind", "layout", "blend", "enter", "motion")} == {
        "kind": "background", "layout": "bottom", "blend": "wrap", "enter": "wipe",
        "motion": motion,
    }
    assert planned["fit"] == "cover"
    overridden = broll.plan_broll(requests(layout="corner_right"), phrase_words(), [item], style,
                                  project_dir=tmp_path, frames=60).lane()["elements"][0]
    assert overridden["layout"] == "corner_right"
    suggestion = broll.suggest_broll(phrase_words(), [item], style)["phrases"][0]["candidates"][0]
    assert (suggestion["kind"], suggestion["layout"], suggestion["fit"]) == (
        "background", "bottom", "cover")


@pytest.mark.parametrize("fields", [
    {"kind": "insert", "layout": "bottom"},
    {"kind": "background", "crop": [0, 0, 10, 10]},
    {"kind": "background", "audio": "mute"},
    {"kind": "background", "box": "headroom"},
    {"placement": []},
    {"kind": "background", "layout": "centre"},
    {"kind": "background", "color_match": True},
])
def test_background_request_field_scope(tmp_path: Path, fields: dict) -> None:
    item = catalog_item(tmp_path)
    with pytest.raises(ValidationError):
        broll.plan_broll(requests(**fields), phrase_words(), [item], {},
                         project_dir=tmp_path, frames=60)


@pytest.mark.parametrize("choice", [{"placement": "green_screen"}, {"kind": "background"}])
def test_explicit_background_request(tmp_path: Path, choice: dict) -> None:
    item = catalog_item(tmp_path)
    result = broll.plan_broll(requests(**choice, layout="corner_left"), phrase_words(), [item], {},
                              project_dir=tmp_path, frames=60).lane()["elements"][0]
    assert (result["kind"], result["layout"], result["motion"]) == (
        "background", "corner_left", "none")


@pytest.mark.parametrize("end,enter", [(.6, "cut"), (.63, "wipe")])
def test_background_entrance_uses_resolved_window_length(
    tmp_path: Path, end: float, enter: str,
) -> None:
    item = catalog_item(tmp_path, kind="image")
    words = CutWords("output", {"t1": "a" * 64}, (
        CutWord("t1:0", "story", "Dog", .03, .2),
        CutWord("t1:1", "story", "park.", .4, end),
    ))
    plan = broll.plan_broll(requests(placement="green_screen"), words, [item], {},
                           project_dir=tmp_path, frames=60)
    assert (plan.broll[0]["start_frame"], plan.broll[0]["end_frame"]) == (1, round(end * 30))
    assert plan.lane()["elements"][0]["enter"] == enter


@pytest.mark.parametrize("motion,push", [("none", "in"), ("push", "none")])
def test_background_request_motion_overrides_style(
    tmp_path: Path, motion: str, push: str,
) -> None:
    item = catalog_item(tmp_path, kind="image")
    plan = broll.plan_broll(requests(kind="background", motion=motion), phrase_words(),
                           [item], {"broll": {"still_push": push}},
                           project_dir=tmp_path, frames=60)
    assert plan.lane()["elements"][0]["motion"] == motion


@pytest.mark.parametrize("kind,motion", [
    ("video", "push"), ("image", "drift"), ("image", None), ("image", []),
])
def test_background_request_refuses_invalid_motion(
    tmp_path: Path, kind: str, motion: object,
) -> None:
    item = catalog_item(tmp_path, kind=kind)
    with pytest.raises(ValidationError, match=r"broll\[0\]\.motion"):
        broll.plan_broll(requests(kind="background", motion=motion), phrase_words(), [item], {},
                         project_dir=tmp_path, frames=60)


@pytest.mark.parametrize("previous_end", [2.0, 59 / 30])
def test_named_word_lead_stops_at_adjacent_planned_shot(tmp_path, previous_end):
    item = catalog_item(tmp_path)
    words = CutWords("output", {"t1": "a" * 64},
                     (CutWord("t1:0", "story", "Dog", 2.0, 2.3),))
    data = {"schema": broll.REQUESTS_SCHEMA, "broll": [
        {"id": "broll-first", "item": "m1", "start": {"time": 0},
         "end": {"time": previous_end}},
        {"id": "broll-second", "item": "m1", "start": {"word": "t1:0"},
         "end": {"time": 3}, "matches": "dog"},
    ]}
    plan = broll.plan_broll(data, words, [item], {}, project_dir=tmp_path, frames=90)
    first, second = plan.broll
    assert first["end_frame"] == second["start_frame"]
    assert second["start_s"] == pytest.approx(previous_end)
    assert second["element"]["start"].get("offset_s", 0) == pytest.approx(previous_end - 2)
    assert "offset_s" not in data["broll"][1]["start"]


@pytest.mark.parametrize(("duration", "in_s", "speed", "fps", "start_frame"), [
    (1, 0, 1, 30, 60),
    (1.05, 0, 1, 30, 59),
    (1.1, 0, 1, 30, 57),
    (2, 0, 1, 30, 57),
    (3, 1, 2, 30, 60),
    (3.1, 1, 2, 30, 59),
    (0.5, 0, 0.5, 30, 60),
    (1.0333333333333332, 0, 1, 30, 60),
    (1.0333333333333334, 0, 1, 30, 59),
    (1, 0, 1, 24, 48),
    (1.05, 0, 1, 24, 47),
    (2, 0, 1, 24, 46),
    (1, 0, 1, 60, 120),
    (1.05, 0, 1, 60, 117),
    (2, 0, 1, 60, 114),
])
def test_named_word_lead_fits_available_source(tmp_path, duration, in_s, speed, fps, start_frame):
    item = catalog_item(tmp_path, duration=duration)
    words = CutWords("output", {}, (CutWord("t1:0", "story", "Dog", 2, 2.3),))
    data = requests(start={"word": "t1:0"}, end={"time": 3}, matches="dog",
                    in_s=in_s, speed=speed)
    original = copy.deepcopy(data)
    plan = broll.plan_broll(data, words, [item], {}, project_dir=tmp_path,
                            frames=3 * fps, fps=fps)
    shot, = plan.broll
    assert plan.status == "ok"
    assert (shot["start_frame"], shot["end_frame"]) == (start_frame, 3 * fps)
    assert shot["out_s"] <= duration
    assert shot["element"]["media"]["in_s"] == in_s
    assert shot["element"]["media"]["speed"] == speed
    assert data == original


def test_named_word_lead_keeps_hold_window_within_source(tmp_path):
    item = catalog_item(tmp_path, duration=0.1)
    words = CutWords("output", {}, (CutWord("t1:0", "story", "Dog", 2, 2.3),))
    data = requests(start={"word": "t1:0"}, hold_s=0.1, matches="dog")
    del data["broll"][0]["end"]
    plan = broll.plan_broll(data, words, [item], {}, project_dir=tmp_path, frames=72, fps=24)
    shot, = plan.broll
    assert shot["start_s"] < 2
    assert shot["duration_s"] == pytest.approx(0.1)
    assert shot["end_frame"] - shot["start_frame"] == 2
    assert shot["out_s"] <= 0.1


@pytest.mark.parametrize(("kind", "speed"), [("image", 1), ("video", "fit")])
def test_named_word_lead_has_no_fixed_source_limit(tmp_path, kind, speed):
    item = catalog_item(tmp_path, kind=kind, duration=1)
    words = CutWords("output", {}, (CutWord("t1:0", "story", "Dog", 2, 2.3),))
    plan = broll.plan_broll(
        requests(start={"word": "t1:0"}, end={"time": 3}, matches="dog", speed=speed),
        words, [item], {}, project_dir=tmp_path, frames=90,
    )
    assert plan.broll[0]["start_s"] == pytest.approx(1.9)


def test_named_word_lead_preserves_authored_offset(tmp_path):
    item = catalog_item(tmp_path, duration=1)
    words = CutWords("output", {}, (CutWord("t1:0", "story", "Dog", 2, 2.3),))
    data = requests(start={"word": "t1:0", "offset_s": -0.1},
                    end={"time": 3}, matches="dog")
    with pytest.raises(ValidationError, match="source window must fit"):
        broll.plan_broll(data, words, [item], {}, project_dir=tmp_path, frames=90)
    assert data["broll"][0]["start"]["offset_s"] == -0.1


@pytest.fixture
def item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    source = tmp_path / "dog.mov"
    source.write_bytes(b"catalog media")
    result = _item(locations=[{"path": str(source), "added_at": STAMP, "seen_at": STAMP}])
    result["sha256"] = sha256_file(source)
    result["provenance"] = {"kind": "own", "source": "camera", "license": None, "url": None}
    return result


def test_suggestions_rank_and_explain_description_tags_speech_and_approval(tmp_path: Path) -> None:
    items = [
        catalog_item(tmp_path),
        catalog_item(
            tmp_path,
            2,
            tags=["dog", "park"],
            speech=_speech("dog"),
            decisions=[
                _decision(verdict="approved"),
            ],
        ),
    ]
    words = phrase_words("Dog", "park.", "Unmatched!")
    before = copy.deepcopy((words, items))
    result = broll.suggest_broll(words, items, {}, limit=1)
    assert result["schema"] == broll.SUGGESTIONS_SCHEMA
    assert len(result["phrases"]) == 1
    phrase = result["phrases"][0]
    assert phrase["text"] == "Dog park." and phrase["word_ids"] == ["t1:0", "t1:1"]
    assert phrase["start"] == {"word": "t1:0", "text": "Dog", "segment": "story", "edge": "start"}
    assert phrase["end"] == {"word": "t1:1", "text": "park.", "segment": "story", "edge": "end"}
    candidate = phrase["candidates"][0]
    assert candidate["item"] == "m2" and candidate["score"] == 7
    assert candidate["kind"] == "insert" and candidate["fit"] == "cover"
    assert (candidate["in_s"], candidate["out_s"]) == (0, 1)
    assert set(candidate["reasons"]) == {
        "matches description: dog, park",
        "matches tags: dog, park",
        "matches speech: dog",
        "whole item approved by d1",
        "placement from style: full_frame",
    }
    assert (words, items) == before


def test_phrase_limit_sentence_ends_and_repeated_id_anchors(item: dict) -> None:
    words = phrase_words("Dog", "park", "dog", "park!", "dog", "park")
    result = broll.suggest_broll(words, [item], {}, phrase_words=3)
    assert [entry["text"] for entry in result["phrases"]] == ["Dog park dog", "park!", "dog park"]
    repeated = CutWords(
        "output",
        {"t1": "a" * 64},
        (
            CutWord("t1:0", "first", "Dog.", 0, 1),
            CutWord("t1:0", "second", "Dog.", 1, 2),
        ),
    )
    assert [
        entry["start"]["segment"]
        for entry in broll.suggest_broll(
            repeated,
            [item],
            {},
        )["phrases"]
    ] == ["first", "second"]


@pytest.mark.parametrize(
    ("media_type", "kind", "placement", "fit"),
    [
        ("clip", "video", "full_frame", "contain"),
        ("screen_recording", "video", "full_frame", "contain"),
        ("photo", "image", "overlay", "cover"),
        ("screenshot", "image", "overlay", "cover"),
    ],
)
def test_media_type_defaults_and_style_overlay_motion(
    tmp_path: Path,
    media_type: str,
    kind: str,
    placement: str,
    fit: str,
) -> None:
    item = catalog_item(tmp_path, kind=kind, orientation="horizontal")
    item["media_type"] = media_type
    candidate = broll.suggest_broll(
        phrase_words(),
        [item],
        {
            "broll": {
                "overlay_box": "side_right",
                "overlay_motion": "none",
            }
        },
    )["phrases"][0]["candidates"][0]
    assert candidate["placement"] == placement and candidate["fit"] == fit
    assert candidate["kind"] == ("insert" if placement == "full_frame" else "overlay")
    assert candidate["box"] == ("side_right" if placement == "overlay" else None)
    assert candidate["motion"] == ("none" if placement == "overlay" else None)
    assert ("in_s" in candidate) == (kind == "video")


def test_avoided_tags_omit_candidates_with_explanation() -> None:
    result = broll.suggest_broll(
        phrase_words(),
        [_item(tags=["busy_background"])],
        {
            "broll": {"avoid_tags": ["busy_background"]},
        },
    )
    assert result["phrases"] == []
    assert result["excluded"] == [{"item": "m1", "reasons": ["avoided tags: busy_background"]}]


def test_suggestions_fill_limit_after_missing_and_changed_top_items(tmp_path: Path) -> None:
    items = [catalog_item(tmp_path, number) for number in range(1, 5)]
    Path(items[0]["locations"][0]["path"]).unlink()
    Path(items[1]["locations"][0]["path"]).write_bytes(b"changed")
    phrase = broll.suggest_broll(phrase_words(), items, {}, limit=2)["phrases"][0]
    assert [entry["item"] for entry in phrase["candidates"]] == ["m3", "m4"]
    assert phrase["skipped"] == [
        {"item": "m1", "reasons": ["media file missing or unreadable"]},
        {"item": "m2", "reasons": ["media file changed"]},
    ]


def test_suggestions_report_phrase_when_all_files_are_missing(item: dict) -> None:
    Path(item["locations"][0]["path"]).unlink()
    phrase = broll.suggest_broll(phrase_words(), [item], {})["phrases"][0]
    assert phrase["candidates"] == []
    assert phrase["skipped"][0]["item"] == item["id"]


@pytest.mark.parametrize("limit", [1, 2])
def test_suggestions_cache_verification_and_stop_at_phrase_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int,
) -> None:
    items = [catalog_item(tmp_path, number) for number in range(1, 7)]
    sources = [Path(item["locations"][0]["path"]) for item in items]
    sources[0].unlink()
    sources[1].write_bytes(b"changed")
    sources[4].write_bytes(b"changed")
    sources[5].unlink()
    hashes = {source: sha256_file(source) for source in sources if source.is_file()}
    hash_calls: Counter[Path] = Counter()
    verified = []
    matching_location = broll._matching_location

    def fake_hash(source: Path) -> str:
        hash_calls[source] += 1
        return hashes[source]

    def verify(item: dict) -> tuple[Path | None, str]:
        verified.append(item["id"])
        return matching_location(item)

    monkeypatch.setattr(broll.receipt, "sha256_file", fake_hash)
    monkeypatch.setattr(broll, "_matching_location", verify)
    words = phrase_words("Dog", "park.", "Dog", "park.", "Dog", "park.")
    phrases = broll.suggest_broll(words, items, {}, limit=limit)["phrases"]
    assert len(phrases) == 3
    for phrase in phrases:
        assert [candidate["item"] for candidate in phrase["candidates"]] == [
            item["id"] for item in items[2:2 + limit]
        ]
        assert phrase["skipped"] == [
            {"item": "m1", "reasons": ["media file missing or unreadable"]},
            {"item": "m2", "reasons": ["media file changed"]},
        ]
    assert verified == [item["id"] for item in items[:2 + limit]]
    assert hash_calls == Counter({source: 1 for source in sources[1:2 + limit]})


def test_suggestions_reverify_locations_on_next_invocation(item: dict) -> None:
    assert broll.suggest_broll(phrase_words(), [item], {})["phrases"][0]["candidates"]
    Path(item["locations"][0]["path"]).write_bytes(b"changed")
    phrase = broll.suggest_broll(phrase_words(), [item], {})["phrases"][0]
    assert phrase["candidates"] == []
    assert phrase["skipped"] == [{"item": "m1", "reasons": ["media file changed"]}]


@pytest.mark.parametrize("kind,placement", [("video", "full_frame"), ("image", "overlay"),
                                          ("image", "full_frame")])
def test_suggested_request_plans_unchanged_with_same_source_window(
    item: dict, kind: str, placement: str,
) -> None:
    item.update(kind=kind, media_type="clip" if kind == "video" else "photo")
    item["duration_s"] = 10 if kind == "video" else None
    item["decisions"] = [_decision(start=0, end=2)] if kind == "video" else []
    words = CutWords("output", {}, (CutWord("t1:0", "story", "Dog.", 0, 1.02),))
    style = {"broll": {f"{item['media_type']}_placement": placement}}
    candidate = broll.suggest_broll(words, [item], style)["phrases"][0]["candidates"][0]
    request = candidate["request"]
    before = copy.deepcopy(request)
    plan = broll.plan_broll({"schema": broll.REQUESTS_SCHEMA, "broll": [request]},
                            words, [item], style, project_dir=_project(item), frames=60)
    assert request == before
    assert plan.broll[0]["end_frame"] == 31
    assert plan.broll[0]["duration_s"] == 1.02
    if kind == "video":
        assert request["in_s"] == candidate["in_s"] == plan.broll[0]["in_s"] == 2
        assert request["speed"] == 1
        assert plan.broll[0]["out_s"] == pytest.approx(candidate["out_s"])
    else:
        assert "in_s" not in request and "speed" not in request
    assert plan.lane()["elements"][0]["kind"] == candidate["kind"]


@pytest.mark.parametrize(("in_s", "duration", "accepted"), [
    (0, 1.1333333333333333, False),
    (0, 1.1333333333333335, True),
    (2, 3.1333333333333333, False),
    (2, 3.1333333333333337, True),
])
def test_suggestions_only_emit_plannable_exact_frame_windows(
    item: dict, in_s: int, duration: float, accepted: bool,
) -> None:
    item["duration_s"] = duration
    item["decisions"] = [_decision(start=0, end=in_s)] if in_s else []
    words = CutWords("output", {}, (CutWord("t1:0", "story", "Dog.", 0, 34 / 30),))
    phrases = broll.suggest_broll(words, [item], {})["phrases"]
    if not accepted:
        assert phrases == []
        return
    candidate = phrases[0]["candidates"][0]
    request = copy.deepcopy(candidate["request"])
    plan = broll.plan_broll(
        {"schema": broll.REQUESTS_SCHEMA, "broll": [request]},
        words, [item], {}, project_dir=_project(item), frames=60,
    )
    assert request == candidate["request"]
    assert (plan.broll[0]["start_frame"], plan.broll[0]["end_frame"]) == (0, 34)
    assert candidate["in_s"] == plan.broll[0]["in_s"] == in_s
    assert candidate["out_s"] == plan.broll[0]["out_s"]


@pytest.mark.parametrize("case", ["missing", "changed"])
def test_plan_refuses_stale_catalog_location_with_refresh_action(item: dict, case: str) -> None:
    source = Path(item["locations"][0]["path"])
    if case == "missing":
        source.unlink()
    else:
        source.write_bytes(b"changed")
    with pytest.raises(ValidationError, match=f"m1:.*{case}.*media add again.*refresh"):
        broll.plan_broll(requests(), phrase_words(), [item], {},
                         project_dir=source.parent, frames=60)


def test_plan_uses_later_matching_location_after_changed_file(item: dict) -> None:
    source = Path(item["locations"][0]["path"])
    changed = source.with_name("changed.mov")
    changed.write_bytes(b"changed")
    item["locations"].insert(0, {"path": str(changed), "added_at": STAMP, "seen_at": STAMP})
    plan = broll.plan_broll(requests(), phrase_words(), [item], {},
                            project_dir=source.parent, frames=60)
    assert plan.lane()["elements"][0]["media"]["path"] == source.name
    assert broll.suggest_broll(phrase_words(), [item], {})["phrases"][0]["skipped"] == []


def test_plan_refuses_clip_shorter_than_snapped_frame_window(item: dict) -> None:
    item["duration_s"] = 1.02
    with pytest.raises(ValidationError, match="source window must fit"):
        broll.plan_broll(requests(start={"time": 0}, end={"time": 1.02}),
                         phrase_words(), [item], {}, project_dir=_project(item), frames=60)


@pytest.mark.parametrize(("fps", "duration", "accepted"), [
    (24, 0.9, False), (24, 1.0, True), (60, 0.9, False), (60, 1.0, True),
])
def test_plan_uses_requested_fps_for_source_consumption(
    item: dict, fps: int, duration: float, accepted: bool,
) -> None:
    item["duration_s"] = duration
    if not accepted:
        with pytest.raises(ValidationError, match="source window must fit"):
            broll.plan_broll(
                requests(), phrase_words(), [item], {}, project_dir=_project(item),
                frames=2 * fps, fps=fps,
            )
        return
    plan = broll.plan_broll(
        requests(), phrase_words(), [item], {}, project_dir=_project(item),
        frames=2 * fps, fps=fps,
    )
    assert plan.fps == fps
    assert (plan.broll[0]["start_frame"], plan.broll[0]["end_frame"]) == (0, fps)
    assert plan.broll[0]["out_s"] == 1.0


def test_approved_taste_rule_changes_screen_recording_suggestion(
    item: dict,
) -> None:
    item["media_type"] = "screen_recording"
    style = resolve("preset:clean_minimal")
    assert (
        broll.suggest_broll(phrase_words(), [item], style.data)["phrases"][0]["candidates"][0][
            "kind"
        ]
        == "insert"
    )
    identity = add_proposals(
        {
            "scope": "broll",
            "condition": None,
            "sentence": "Keep screen recordings in an overlay.",
            "machine": {
                "kind": "style_field",
                "field": "broll.screen_recording_placement",
                "value": "overlay",
            },
            "sources": [
                {
                    "type": "comment",
                    "media_sha256": "a" * 64,
                    "comment_id": "note",
                    "label": "fix",
                    "start_frame": None,
                    "end_frame": None,
                    "quote": "Use an overlay.",
                }
            ],
            "context": None,
        }
    )[0].id
    approve(identity)
    candidate = broll.suggest_broll(phrase_words(), [item], apply_taste(style).style.data)[
        "phrases"
    ][0]["candidates"][0]
    assert candidate["kind"] == "overlay" and candidate["placement"] == "overlay"
    assert "placement from style: overlay" in candidate["reasons"]


def test_plan_valid_fragment_frames_metadata_and_input_purity(item: dict) -> None:
    data = requests(in_s=2, speed=2, matches="Dog in the park", audio={"gain_db": -18})
    before = copy.deepcopy((data, item))
    plan = broll.plan_broll(data, phrase_words(), [item], {}, frames=60, project_dir=_project(item))
    assert plan.status == "ok" and plan.warnings == ()
    fragment = plan.lane()
    parsed = parse_document("timeline-fragment", fragment, lane="broll")
    assert parsed.elements[0].kind == "insert"
    element = fragment["elements"][0]
    assert element["media"] == {
        "path": "dog.mov",
        "sha256": item["sha256"],
        "in_s": 2,
        "speed": 2,
        "fit": "cover",
    }
    assert element["provenance"] == item["provenance"]
    assert element["matches"] == "Dog in the park" and element["audio"] == {"gain_db": -18}
    summary = plan.to_dict()
    assert summary["schema"] == broll.PLAN_SCHEMA and summary["frames"] == 60
    entry = summary["broll"][0]
    assert (entry["start_frame"], entry["end_frame"], entry["out_s"]) == (0, 30, 4)
    assert summary["words_sha256"] == {"t1": "a" * 64}
    element["media"]["fit"] = "contain"
    summary["broll"][0]["element"]["id"] = "changed"
    assert plan.lane()["elements"][0]["id"] == "broll-dog"
    assert plan.lane()["elements"][0]["media"]["fit"] == "cover"
    assert (data, item) == before
    data["broll"][0]["start"]["word"] = "t1:99"
    assert plan.lane()["elements"][0]["start"]["word"] == "t1:0"


def test_chronology_passes_through_and_invalid_value_is_refused(item: dict) -> None:
    plan = broll.plan_broll(requests(chronology="hook"), phrase_words(), [item], {},
                            frames=60, project_dir=_project(item))
    assert plan.lane()["elements"][0]["chronology"] == "hook"
    with pytest.raises(ValidationError, match="chronology"):
        broll.plan_broll(requests(chronology="later"), phrase_words(), [item], {},
                         frames=60, project_dir=_project(item))


@pytest.mark.parametrize("orientation", ["vertical", "horizontal", "square"])
def test_insert_fit_rule_and_override(item: dict, orientation: str) -> None:
    item["orientation"] = orientation
    plan = broll.plan_broll(
        requests(), phrase_words(), [item], {}, frames=60, project_dir=_project(item)
    )
    assert plan.lane()["elements"][0]["media"]["fit"] == (
        "cover" if orientation == "vertical" else "contain"
    )
    plan = broll.plan_broll(
        requests(fit="contain"), phrase_words(), [item], {}, frames=60, project_dir=_project(item)
    )
    assert plan.lane()["elements"][0]["media"]["fit"] == "contain"


def test_insert_crop_is_written_in_source_pixels(item: dict) -> None:
    plan = broll.plan_broll(
        requests(crop=[10, 20, 50, 80]), phrase_words(), [item], {},
        frames=60, project_dir=_project(item),
    )
    media = plan.lane()["elements"][0]["media"]
    assert media["fit"] == "crop" and media["crop"] == [10, 20, 50, 80]


@pytest.mark.parametrize(("changes", "field", "message"), [
    ({"crop": [50, 20, 50, 80]}, "crop", "90x160"),
    ({"kind": "overlay", "crop": [0, 0, 50, 80]}, "crop", "only inserts"),
    ({"fit": "contain", "crop": [0, 0, 50, 80]}, "fit", "crop"),
    ({"fit": "crop"}, "crop", "crop"),
])
def test_bad_crop_requests_name_field(
    item: dict, changes: dict, field: str, message: str,
) -> None:
    with pytest.raises(ValidationError) as error:
        broll.plan_broll(
            requests(**changes), phrase_words(), [item], {},
            frames=60, project_dir=_project(item),
        )
    assert error.value.parameter == f"broll[0].{field}"
    assert message in error.value.detail


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"placement": "side_left", "motion": "none", "fit": "contain"},
        {"placement": "box", "box": [10, 20, 300, 400]},
        {"box": "side_left"},
        {"box": [10, 20, 300, 400]},
    ],
)
def test_overlay_defaults_and_explicit_fields(item: dict, changes: dict) -> None:
    style = {
        "broll": {
            "clip_placement": "overlay",
            "overlay_box": "side_right",
            "overlay_motion": "none",
        }
    }
    plan = broll.plan_broll(
        requests(**changes), phrase_words(), [item], style, frames=60, project_dir=_project(item)
    )
    element = plan.lane()["elements"][0]
    assert parse_document("timeline-fragment", plan.lane(), lane="broll").elements[0].kind == (
        "overlay"
    )
    assert element["placement"] == (
        "box"
        if isinstance(changes.get("box"), list)
        else (changes.get("placement", changes.get("box", "side_right")))
    )
    assert element["motion"] == "none" and "fit" not in element["media"]
    assert "audio" not in element


def test_side_left_request_overrides_full_frame_clip_style(item: dict) -> None:
    assert item["media_type"] == "clip"
    plan = broll.plan_broll(
        requests(placement="side_left"), phrase_words(), [item],
        {"broll": {"clip_placement": "full_frame"}}, frames=60, project_dir=_project(item),
    )
    element = plan.lane()["elements"][0]
    assert element["kind"] == "overlay" and element["placement"] == "side_left"
    assert "audio" not in element


def test_hold_time_anchors_stills_and_unknown_provenance(item: dict) -> None:
    item.update(kind="image", media_type="photo", duration_s=None)
    item["provenance"]["kind"] = None
    data = requests(start={"time": 0.15}, hold_s=2)
    del data["broll"][0]["end"]
    plan = broll.plan_broll(data, phrase_words(), [item], {}, frames=90, project_dir=_project(item))
    assert (plan.broll[0]["start_frame"], plan.broll[0]["end_frame"]) == (5, 65)
    assert "provenance" not in plan.lane()["elements"][0]
    assert plan.broll[0]["provenance"]["kind"] is None


def test_location_uses_first_existing_and_project_relative_path(
    item: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "project.json").write_text("{}")
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    item["locations"].insert(0, {"path": str(tmp_path / "missing.mov")})
    plan = broll.plan_broll(
        requests(), phrase_words(), [item], {}, frames=60, project_dir=_project(item)
    )
    assert plan.lane()["elements"][0]["media"]["path"] == "dog.mov"
    plan = broll.plan_broll(requests(), phrase_words(), [item], {}, frames=60, project_dir=subdir)
    assert plan.lane()["elements"][0]["media"]["path"] == str(tmp_path / "dog.mov")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"item": "m999"}, "unknown catalog item"),
        ({"id": "other"}, "broll- prefix"),
            ({"kind": "graphic"}, "insert, overlay or background"),
        ({"start": {"word": "t1:99"}}, "removed"),
        ({"start": {"word": "t1:0", "text": "Wrong"}}, "guard"),
        ({"start": {"time": 2}}, "window"),
        ({"hold_s": 2}, "exactly one"),
        ({"in_s": -1}, "in_s"),
        ({"speed": 0}, "speed"),
        ({"in_s": 10}, "source window"),
        ({"kind": "overlay", "audio": "mute"}, "only inserts"),
        ({"box": [0, 0, 20, 20]}, "only overlays"),
        ({"extra": True}, "unknown field"),
    ],
)
def test_bad_requests_refused(item: dict, changes: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        broll.plan_broll(
            requests(**changes), phrase_words(), [item], {}, frames=60, project_dir=_project(item)
        )


def test_missing_locations_duplicate_ids_and_overlaps_refused(item: dict) -> None:
    with pytest.raises(ValidationError, match="m1:.*media add again to refresh the catalog"):
        broll.plan_broll(
            requests(),
            phrase_words(),
            [{**item, "locations": []}],
            {},
            frames=60,
            project_dir=_project(item),
        )
    data = requests()
    data["broll"].append(copy.deepcopy(data["broll"][0]))
    with pytest.raises(ValidationError, match="duplicate id"):
        broll.plan_broll(data, phrase_words(), [item], {}, frames=60, project_dir=_project(item))
    data["broll"][1]["id"] = "broll-second"
    data["broll"][1]["kind"] = "overlay"
    with pytest.raises(ValidationError, match="overlap"):
        broll.plan_broll(data, phrase_words(), [item], {}, frames=60, project_dir=_project(item))
    data["broll"][1].update(start={"time": 1}, end={"time": 2})
    assert (
        len(
            broll.plan_broll(
                data, phrase_words(), [item], {}, frames=60, project_dir=_project(item)
            ).broll
        )
        == 2
    )


@pytest.mark.parametrize(
    ("decision", "changes", "refused"),
    [
        (_decision(), {}, True),
        (_decision(start=0.5, end=1.5), {}, True),
        (_decision(start=1, end=2), {}, False),
        (_decision(start=0, end=1), {"in_s": 1}, False),
        (_decision(start=1, end=2), {"speed": 2}, True),
        (_decision(start=8, end=9), {"speed": "fit"}, True),
        (_decision(start=0, end=1, revoked=True), {}, False),
    ],
)
def test_rejected_source_ranges_and_speed(item: dict, decision: dict, changes: dict, refused: bool):
    item["decisions"] = [decision]
    if refused:
        with pytest.raises(ValidationError, match="rejected decision d1"):
            broll.plan_broll(
                requests(**changes),
                phrase_words(),
                [item],
                {},
                frames=60,
                project_dir=_project(item),
            )
    else:
        plan = broll.plan_broll(
            requests(**changes), phrase_words(), [item], {}, frames=60, project_dir=_project(item)
        )
        assert plan.status == "ok"


def test_avoided_explicit_choice_warns_and_requires_review(item: dict) -> None:
    item["description"]["tags"] = ["busy_background"]
    plan = broll.plan_broll(
        requests(),
        phrase_words(),
        [item],
        {
            "broll": {"avoid_tags": ["busy_background"]},
        },
        frames=60,
        project_dir=_project(item),
    )
    assert plan.status == "needs_review" and len(plan.lane()["elements"]) == 1
    assert "busy_background" in plan.warnings[0]


def test_repeated_word_anchor_requires_segment(item: dict) -> None:
    data = requests(start={"word": "t1:12"}, end={"time": 14})
    with pytest.raises(ValidationError, match="ambiguous"):
        broll.plan_broll(
            data, synthetic_words(), [item], {}, frames=900, project_dir=_project(item)
        )
    data["broll"][0]["start"]["segment"] = "story"
    assert (
        broll.plan_broll(
            data, synthetic_words(), [item], {}, frames=900, project_dir=_project(item)
        ).broll[0]["start_frame"]
        == 372
    )


def test_writer_receipts_owned_replacement_and_tamper_refusal(item: dict, tmp_path: Path) -> None:
    out = tmp_path / "out"
    plan = broll.plan_broll(
        requests(), phrase_words(), [item], {}, frames=60, project_dir=_project(item)
    )
    for _ in range(2):
        with paths.operation():
            outputs = broll.write_broll_plan(plan, out)
        manifest = json.loads(outputs["broll-plan.json"].read_bytes())
        assert manifest["outputs"] == [
            {
                "path": "broll-lane.json",
                "sha256": sha256_file(outputs["broll-lane.json"]),
            }
        ]
        assert len(list(out.iterdir())) == 2
    outputs["broll-lane.json"].write_text('{"schema": "yap-studio/foreign@1"}')
    with paths.operation(), pytest.raises(OutputPathError, match="current hash"):
        broll.write_broll_plan(plan, out)


@pytest.mark.parametrize("name", ["broll-lane.json", "broll-plan.json", "receipt.pending.json"])
def test_writer_foreign_outputs_preserved(item: dict, tmp_path: Path, name: str) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / name).write_text("foreign")
    plan = broll.plan_broll(
        requests(), phrase_words(), [item], {}, frames=60, project_dir=_project(item)
    )
    with paths.operation(), pytest.raises(OutputPathError):
        broll.write_broll_plan(plan, out)
    assert (out / name).read_text() == "foreign"
    assert len(list(out.iterdir())) == 1


def test_writer_input_and_symlink_refusal(item: dict, tmp_path: Path) -> None:
    plan = broll.plan_broll(
        requests(), phrase_words(), [item], {}, frames=60, project_dir=_project(item)
    )
    out = tmp_path / "out"
    with paths.operation():
        paths.protect_input(out / "broll-lane.json")
        with pytest.raises(OutputPathError, match="input file"):
            broll.write_broll_plan(plan, out)
    target = tmp_path / "foreign.json"
    target.write_text("foreign")
    (out / "broll-plan.json").symlink_to(target)
    with paths.operation(), pytest.raises(OutputPathError, match="symlink"):
        broll.write_broll_plan(plan, out)
    assert target.read_text() == "foreign"


@pytest.mark.parametrize("existing", [False, True])
def test_writer_recovers_interrupted_publication(
    item: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
) -> None:
    plan = broll.plan_broll(
        requests(), phrase_words(), [item], {}, frames=60, project_dir=_project(item)
    )
    out = tmp_path / "out"
    real_atomic = paths.atomic_output
    if existing:
        previous = broll.plan_broll(
            requests(matches="previous"),
            phrase_words(),
            [item],
            {},
            frames=60,
            project_dir=_project(item),
        )
        with paths.operation():
            broll.write_broll_plan(previous, out)

    @contextlib.contextmanager
    def fail_manifest(target, *args, **kwargs):
        if Path(target).name == "broll-plan.json":
            raise OSError("interrupted")
        with real_atomic(target, *args, **kwargs) as temporary:
            yield temporary

    monkeypatch.setattr(paths, "atomic_output", fail_manifest)
    with paths.operation(), pytest.raises(OutputPathError, match="interrupted"):
        broll.write_broll_plan(plan, out)
    assert (out / "receipt.pending.json").exists()
    assert (out / "broll-plan.json").exists() == existing
    monkeypatch.setattr(paths, "atomic_output", real_atomic)
    with paths.operation():
        broll.write_broll_plan(plan, out)
    assert len(list(out.iterdir())) == 2
    assert json.loads((out / "broll-lane.json").read_bytes()) == plan.lane()
