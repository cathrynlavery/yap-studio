"""Pure source selection and the read-only media pick command."""

from __future__ import annotations

import builtins
import copy
import json
import os
import shutil
import subprocess
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest
import synth_broll as synth

from yap_studio import cli, receipt
from yap_studio.errors import ValidationError
from yap_studio.media import (
    CATALOG_SCHEMA,
    DESCRIPTION_SCHEMA,
    INSERT_MAX_S,
    INSERT_MIN_S,
    MONTAGE_PART_MAX_S,
    MONTAGE_PART_MIN_S,
    REQUEST_SCHEMA,
    SelectionRequest,
    add_paths,
    catalog_path,
    describe_item,
    parse_request,
    select_candidates,
    update_catalog,
    validate_catalog,
)

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
STAMP = "2026-10-05T12:00:00Z"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


def _decision(
    number: int = 1, *, verdict: str = "rejected", start: float | None = None,
    end: float | None = None, revoked: bool = False,
) -> dict:
    return {
        "id": f"d{number}", "verdict": verdict, "start_s": start, "end_s": end,
        "reason": "agent judgment", "by": "agent", "at": STAMP,
        "revoked_at": STAMP if revoked else None, "revoked_by": "agent" if revoked else None,
    }


def _speech(text: str, status: str = "words") -> dict:
    return {
        "status": status, "engine": "test", "model": "test", "words": len(text.split()),
        "text": text, "path": None, "message": None, "at": STAMP,
    }


def _item(
    number: int = 1, *, kind: str = "video", orientation: str = "vertical",
    duration: float = 10.0, text: str | None = "A dog in the park",
    tags: list[str] | None = None, speech: dict | None = None,
    decisions: list[dict] | None = None, locations: list[dict] | None = None,
) -> dict:
    """Build catalog-shaped data and prove it valid before any pure-function test."""
    width, height = {"vertical": (90, 160), "horizontal": (160, 90), "square": (80, 80)}[
        orientation
    ]
    decisions = [] if decisions is None else decisions
    item = {
        "id": f"m{number}", "sha256": f"{number:064x}", "bytes": 100, "kind": kind,
        "media_type": "photo" if kind == "image" else "clip", "media_type_by": "rule",
        "format": "png" if kind == "image" else "mov", "codec": None,
        "width": width, "height": height, "rotation": 0, "orientation": orientation,
        "duration_s": None if kind == "image" else duration,
        "fps": None if kind == "image" else "30/1", "variable_frame_rate": False,
        "color_transfer": None, "hdr": None, "has_audio": False, "camera": None,
        "provenance": {"kind": None, "source": None, "license": None, "url": None},
        "locations": [{
            "path": f"/media/candidate-{number}.mov", "added_at": STAMP, "seen_at": STAMP,
        }] if locations is None else locations,
        "added_at": STAMP,
        "description": None if text is None and tags is None else {
            "text": text, "tags": [] if tags is None else tags,
            "by": "agent", "at": STAMP, "revision": 1,
        },
        "decisions": decisions,
        "next_decision": max((int(entry["id"][1:]) for entry in decisions), default=0) + 1,
        "sheet": None, "speech": speech,
    }
    document = {"schema": CATALOG_SCHEMA, "next_id": number + 1, "items": [item], "index": {}}
    assert validate_catalog(document, Path("catalog.json")) is document
    return item


def _request_data(**changes: object) -> dict:
    return {
        "schema": REQUEST_SCHEMA, "words": "dog park", "length_s": 2.0,
        "placement": "full_frame", **changes,
    }


def _request(**changes: object) -> SelectionRequest:
    return parse_request(_request_data(**changes))


def _span(candidate: dict) -> tuple[float, float]:
    return candidate["in_s"], candidate["out_s"]


def test_request_defaults_and_frozen_dataclass() -> None:
    request = _request(words="  dog park  ")
    assert isinstance(request, SelectionRequest)
    assert request.schema == REQUEST_SCHEMA and request.words == "dog park"
    assert request.length_s == 2.0 and request.placement == "full_frame"
    assert request.orientation == "any" and request.montage is False
    assert request.kinds == ("video", "image") and request.match == "words"
    assert request.exclude == () and request.limit == 5 and request.anchor is None
    with pytest.raises(FrozenInstanceError):
        request.limit = 2


@pytest.mark.parametrize(("montage", "length"), [
    (False, INSERT_MIN_S), (False, INSERT_MAX_S), (False, 2.5),
    (True, MONTAGE_PART_MIN_S), (True, MONTAGE_PART_MAX_S), (True, 1.0),
])
def test_length_bands_inclusive_and_never_stretched(montage: bool, length: float) -> None:
    request = _request(montage=montage, length_s=length)
    candidate = select_candidates([_item()], request)[0]
    assert candidate["length_s"] == length and _span(candidate) == (0.0, length)


@pytest.mark.parametrize(("montage", "length"), [
    (False, 0.999999), (False, 4.000001), (True, 0.699999), (True, 1.300001),
    (False, 0.7), (True, 4.0),
])
def test_length_outside_applicable_band_refused(montage: bool, length: float) -> None:
    with pytest.raises(ValidationError) as caught:
        _request(montage=montage, length_s=length)
    assert caught.value.parameter == "length_s"
    assert "choose anchor words that fit" in str(caught.value)


@pytest.mark.parametrize(("changes", "field"), [
    ({"schema": "other@1"}, "schema"), ({"schema": None}, "schema"),
    ({"extra": 1}, "extra"), ({"words": " "}, "words"), ({"words": 3}, "words"),
    ({"words": "x" * 501}, "words"), ({"length_s": True}, "length_s"),
    ({"length_s": "2"}, "length_s"), ({"length_s": None}, "length_s"),
    ({"length_s": float("nan")}, "length_s"), ({"length_s": float("inf")}, "length_s"),
    ({"length_s": 10 ** 400}, "length_s"), ({"placement": []}, "placement"),
    ({"placement": "box"}, "placement"), ({"orientation": "portrait"}, "orientation"),
    ({"orientation": None}, "orientation"), ({"montage": 1}, "montage"),
    ({"kinds": []}, "kinds"), ({"kinds": "video"}, "kinds"),
    ({"kinds": ["audio"]}, "kinds[0]"), ({"kinds": [None]}, "kinds[0]"),
    ({"match": "meaning"}, "match"), ({"exclude": "m3"}, "exclude"),
    ({"exclude": [3]}, "exclude[0]"), ({"exclude": ["m0"]}, "exclude[0]"),
    ({"limit": True}, "limit"), ({"limit": 2.0}, "limit"), ({"limit": 0}, "limit"),
    ({"limit": 51}, "limit"), ({"anchor": []}, "anchor"), ({"anchor": None}, "anchor"),
    ({"anchor": {"word": object()}}, "anchor.word"),
    ({"anchor": {"time": float("nan")}}, "anchor.time"),
    ({"anchor": {1: "word"}}, "anchor"),
])
def test_request_validation_names_field(changes: dict, field: str) -> None:
    with pytest.raises(ValidationError) as caught:
        _request(**changes)
    assert caught.value.parameter == field


@pytest.mark.parametrize("field", ["schema", "words", "length_s"])
def test_required_request_fields(field: str) -> None:
    data = _request_data()
    del data[field]
    with pytest.raises(ValidationError) as caught:
        parse_request(data)
    assert caught.value.parameter == field


@pytest.mark.parametrize("data", [None, [], "dog", 1])
def test_request_must_be_object(data: object) -> None:
    with pytest.raises(ValidationError, match="request"):
        parse_request(data)


def test_anchor_must_not_be_cyclic() -> None:
    anchor = {}
    anchor["cycle"] = anchor
    with pytest.raises(ValidationError) as caught:
        _request(anchor=anchor)
    assert caught.value.parameter == "anchor"


def test_rejected_union_skipped_window_after_it() -> None:
    item = _item(decisions=[
        _decision(1, start=0.0, end=1.5),
        _decision(2, start=1.0, end=3.0),
        _decision(3, start=3.0, end=4.0),
        _decision(4, start=2.0, end=2.5),
    ])
    candidate = select_candidates([item], _request())[0]
    assert _span(candidate) == (4.0, 6.0)
    assert candidate["approved"] is False and candidate["approved_by"] == []
    assert "skips rejected ranges d1, d2, d3, d4" in candidate["reasons"]


def test_window_ending_exactly_at_rejected_start_allowed() -> None:
    candidate = select_candidates([
        _item(decisions=[_decision(start=2.0, end=4.0)]),
    ], _request())[0]
    assert _span(candidate) == (0.0, 2.0)


def test_any_positive_overlap_disallowed_without_epsilon() -> None:
    candidate = select_candidates([
        _item(decisions=[_decision(start=1.9999999999999998, end=4.0)]),
    ], _request())[0]
    assert _span(candidate) == (4.0, 6.0)


@pytest.mark.parametrize(("decisions", "expected"), [
    ([_decision(start=0.0, end=5.0, revoked=True)], (0.0, 2.0)),
    ([_decision(revoked=True)], (0.0, 2.0)),
    ([_decision(verdict="approved", start=4.0, end=8.0, revoked=True)], (0.0, 2.0)),
])
def test_revoked_decisions_ignored(decisions: list[dict], expected: tuple[float, float]) -> None:
    candidate = select_candidates([_item(decisions=decisions)], _request())[0]
    assert _span(candidate) == expected and candidate["approved"] is False


@pytest.mark.parametrize("kind", ["video", "image"])
def test_whole_item_rejection_excludes_even_when_approved(kind: str) -> None:
    item = _item(kind=kind, decisions=[_decision(), _decision(2, verdict="approved")])
    assert select_candidates([item], _request()) == []


def test_earliest_approved_interval_preferred_and_containing_ids_reported() -> None:
    item = _item(decisions=[
        _decision(1, verdict="approved", start=6.0, end=9.0),
        _decision(2, verdict="approved", start=3.0, end=6.0),
        _decision(3, verdict="approved", start=2.5, end=6.0),
        _decision(4, verdict="approved", start=2.5, end=3.0),
        _decision(5, verdict="approved", start=0.0, end=8.0, revoked=True),
    ])
    candidate = select_candidates([item], _request())[0]
    assert _span(candidate) == (2.5, 4.5) and candidate["approved"] is True
    assert candidate["approved_by"] == ["d3"]
    assert "inside approved range d3 (2.500 to 6.000 s)" in candidate["reasons"]
    candidate = select_candidates([item], _request(length_s=3.0))[0]
    assert _span(candidate) == (2.5, 5.5) and candidate["approved_by"] == ["d3"]


def test_all_approvals_containing_window_reported() -> None:
    item = _item(decisions=[
        _decision(1, verdict="approved", start=4.0, end=8.0),
        _decision(2, verdict="approved", start=4.0, end=9.0),
        _decision(3, verdict="approved", start=5.0, end=9.0),
    ])
    candidate = select_candidates([item], _request())[0]
    assert _span(candidate) == (4.0, 6.0) and candidate["approved_by"] == ["d1", "d2"]


def test_approval_intersected_with_free_time_rejection_wins() -> None:
    item = _item(decisions=[
        _decision(1, verdict="approved", start=0.5, end=8.0),
        _decision(2, start=1.0, end=4.0),
    ])
    candidate = select_candidates([item], _request())[0]
    assert _span(candidate) == (4.0, 6.0) and candidate["approved_by"] == ["d1"]


def test_short_approvals_not_combined_fall_back_to_earliest_free_time() -> None:
    item = _item(decisions=[
        _decision(1, verdict="approved", start=3.0, end=4.0),
        _decision(2, verdict="approved", start=4.0, end=5.0),
    ])
    candidate = select_candidates([item], _request())[0]
    assert _span(candidate) == (0.0, 2.0)
    assert candidate["approved"] is False and candidate["approved_by"] == []
    assert "no approved range fits; earliest clear stretch" in candidate["reasons"]


@pytest.mark.parametrize("kind", ["video", "image"])
def test_whole_item_approval(kind: str) -> None:
    decisions = [_decision(verdict="approved")]
    if kind == "video":
        decisions.append(_decision(2, start=0.0, end=1.0))
    candidate = select_candidates([_item(kind=kind, decisions=decisions)], _request())[0]
    assert candidate["approved"] is True and candidate["approved_by"] == ["d1"]
    assert _span(candidate) == ((1.0, 3.0) if kind == "video" else (0.0, 2.0))
    assert "whole item approved by d1" in candidate["reasons"]


@pytest.mark.parametrize(("duration", "decisions"), [
    (1.99, []), (0.0, []), (10.0, [_decision(start=0.0, end=10.0)]),
    (4.0, [_decision(start=1.5, end=2.5)]),
])
def test_no_window_fits(duration: float, decisions: list[dict]) -> None:
    assert select_candidates([_item(duration=duration, decisions=decisions)], _request()) == []


def test_exact_decimal_window_fits_without_binary_addition_drift() -> None:
    item = _item(duration=1.1, decisions=[_decision(start=0.0, end=0.3)])
    candidate = select_candidates([item], _request(length_s=0.8, montage=True))[0]
    assert _span(candidate) == (0.3, 1.1)


@pytest.mark.parametrize("duration", [1.1333333333333333, 1.1333333333333335])
def test_exact_frame_length_checks_clip_duration(duration: float) -> None:
    length = Fraction(34, 30)
    item = _item(duration=duration)
    request = _request(length_s=float(length))
    assert len(select_candidates([item], request)) == 1
    exact_request = replace(request, exact_length_s=length)
    candidates = select_candidates([item], exact_request)
    assert len(candidates) == int(Fraction(str(duration)) >= length)
    if candidates:
        assert candidates[0]["length_s"] == float(length)
        assert _span(candidates[0]) == (0.0, float(length))
        json.dumps(candidates)


@pytest.mark.parametrize("verdict", ["approved", "rejected"])
def test_exact_frame_length_checks_decision_boundaries(verdict: str) -> None:
    length = Fraction(34, 30)
    decisions = (
        [
            _decision(1, verdict=verdict, start=0.0, end=float(length)),
            _decision(2, verdict=verdict, start=2.0, end=4.0),
        ]
        if verdict == "approved"
        else [_decision(start=float(length), end=2.0)]
    )
    item = _item(decisions=decisions)
    request = replace(_request(length_s=float(length)), exact_length_s=length)
    candidate = select_candidates([item], request)[0]
    assert _span(candidate) == (2.0, float(2 + length))
    assert candidate["approved"] is (verdict == "approved")


def test_submillisecond_length_preserved() -> None:
    candidate = select_candidates([_item(duration=2)], _request(length_s=1.00055))[0]
    assert candidate["out_s"] - candidate["in_s"] == pytest.approx(1.00055, abs=1e-9)
    assert candidate["length_s"] == 1.00055


@pytest.mark.parametrize(("rejected_start", "expected_in"), [(1.00055, 0.0), (1.00045, 1.5)])
def test_submillisecond_length_rejection_boundary(
    rejected_start: float, expected_in: float,
) -> None:
    item = _item(duration=3, decisions=[_decision(start=rejected_start, end=1.5)])
    candidate = select_candidates([item], _request(length_s=1.00055))[0]
    assert candidate["in_s"] == expected_in
    assert candidate["out_s"] == pytest.approx(expected_in + 1.00055, abs=1e-9)


@pytest.mark.parametrize("end", [1.2341, 1.2345, 1.2349])
def test_rounding_start_moves_inward_without_rejected_overlap(end: float) -> None:
    item = _item(decisions=[_decision(start=0.0, end=end)])
    candidate = select_candidates([item], _request())[0]
    start, out = _span(candidate)
    assert start >= end and out - start == pytest.approx(2.0)
    assert start == round(start, 3) and out == round(out, 3)


def test_exact_end_fits_before_rejection_without_rounding_overlap() -> None:
    item = _item(decisions=[_decision(start=2.0006, end=4.0)])
    candidate = select_candidates([item], _request(length_s=2.00055))[0]
    assert _span(candidate) == (0.0, 2.00055)


def test_rounding_drops_window_that_cannot_stay_clear() -> None:
    item = _item(duration=3.0004, decisions=[_decision(start=0.0, end=1.0004)])
    assert select_candidates([item], _request()) == []


def test_rounded_approval_stays_inside_approval_and_clear_of_rejection() -> None:
    item = _item(decisions=[
        _decision(1, verdict="approved", start=1.2344, end=5.0),
        _decision(2, start=0.0, end=1.2344),
        _decision(3, start=5.0, end=8.0),
    ])
    candidate = select_candidates([item], _request())[0]
    assert _span(candidate) == (1.235, 3.235)
    assert candidate["approved"] is True and candidate["approved_by"] == ["d1"]


def test_millisecond_windows_checked_against_rejections() -> None:
    # Vary both edges around rounding boundaries without filesystem fixtures.
    for offset in range(1, 10):
        start = 1 + offset / 10000
        end = 3 + offset / 10000
        item = _item(duration=6, decisions=[
            _decision(1, start=0, end=start), _decision(2, start=end, end=4),
        ])
        for candidate in select_candidates([item], _request()):
            lower, upper = (Fraction(str(value)) for value in _span(candidate))
            for decision in item["decisions"]:
                assert upper <= Fraction(str(decision["start_s"])) or (
                    lower >= Fraction(str(decision["end_s"]))
                )


@pytest.mark.parametrize("orientation", ["vertical", "horizontal", "square", "any"])
def test_orientation_filter_and_full_frame_fit(orientation: str) -> None:
    items = [_item(i, orientation=value) for i, value in enumerate(
        ("vertical", "horizontal", "square"), start=1,
    )]
    candidates = select_candidates(items, _request(orientation=orientation))
    assert len(candidates) == (3 if orientation == "any" else 1)
    for candidate in candidates:
        assert orientation == "any" or candidate["orientation"] == orientation
        assert candidate["fit"] == (
            "cover" if candidate["orientation"] == "vertical" else "contain"
        )


@pytest.mark.parametrize("kinds", [["video"], ["image"], ["video", "image"]])
def test_kinds_and_exclude(kinds: list[str]) -> None:
    items = [_item(), _item(2, kind="image"), _item(3)]
    candidates = select_candidates(items, _request(kinds=kinds, exclude=["m3"]))
    assert [entry["item"] for entry in candidates] == [
        item["id"] for item in items[:2] if item["kind"] in kinds
    ]


def test_still_and_overlay_exact_candidate_keys() -> None:
    item = _item(kind="image", orientation="square")
    candidate = select_candidates([item], _request(placement="overlay"))[0]
    assert candidate == {
        "item": "m1", "sha256": item["sha256"], "kind": "image", "media_type": "photo",
        "orientation": "square", "width": 80, "height": 80, "placement": "overlay",
        "in_s": 0.0, "out_s": 2.0, "length_s": 2.0, "still": True, "fit": None,
        "score": 2, "matched": ["dog", "park"], "approved": False, "approved_by": [],
        "reasons": ["no whole-item approval", "matches description: dog, park",
                    "placement from request: overlay"],
        "locations": ["/media/candidate-1.mov"], "anchor": None,
    }
    video = select_candidates([_item()], _request(placement="overlay"))[0]
    assert video["fit"] is None and video["still"] is False
    assert set(video) == set(candidate)


@pytest.mark.parametrize("words", ["unmatched", "the and very then"])
def test_match_words_versus_any(words: str) -> None:
    items = [_item(text=None)]
    assert select_candidates(items, _request(words=words)) == []
    candidate = select_candidates(items, _request(words=words, match="any"))[0]
    assert candidate["score"] == 0 and candidate["matched"] == []


def test_scoring_plural_folding_stopwords_distinct_terms_and_request_order() -> None:
    item = _item(
        tags=["Dogs", "park dogs", "park"], text="Dog, parks and grass grass",
        speech=_speech("dogs grass owner'S 123"),
    )
    candidate = select_candidates([item], _request(
        words="The parks and DOGS with my dog's grass grass dog's 123 just very then",
    ))[0]
    assert candidate["score"] == 10  # Tags 4, description 3, speech 3.
    assert candidate["matched"] == ["park", "dog", "gras", "123"]
    assert "matches tags: park, dog" in candidate["reasons"]
    assert "matches description: park, dog, gras" in candidate["reasons"]
    assert "matches speech: dog, gras, 123" in candidate["reasons"]


def test_apostrophes_digits_and_unicode_letters_tokenized() -> None:
    item = _item(text="A dog's café 123")
    candidate = select_candidates([item], _request(words="dog's café 123"))[0]
    assert candidate["matched"] == ["dog'", "café", "123"]
    assert candidate["score"] == 3


@pytest.mark.parametrize("status", ["skipped", "failed", "no_speech", "no_audio", "no_engine"])
def test_only_word_status_speech_matches(status: str) -> None:
    item = _item(text=None, speech=_speech("dog", status))
    assert select_candidates([item], _request()) == []


def test_approved_first_then_score_then_numeric_id_and_limit() -> None:
    items = [
        _item(10), _item(2), _item(3, tags=["dog", "park"]),
        _item(11, text=None, decisions=[_decision(verdict="approved")]),
        _item(4, tags=["dog"], decisions=[_decision(verdict="approved")]),
    ]
    candidates = select_candidates(items, _request(match="any", limit=4))
    assert [entry["item"] for entry in candidates] == ["m4", "m11", "m3", "m2"]
    assert [entry["score"] for entry in candidates] == [4, 0, 6, 2]
    assert select_candidates(list(reversed(items)), _request(match="any", limit=4)) == candidates
    assert len(select_candidates(items, _request(match="any", limit=50))) == 5


def test_rank_candidates_returns_full_eligible_order_without_limit() -> None:
    from yap_studio.media import rank_candidates

    items = [_item(number) for number in (6, 2, 4, 3, 1, 5)]
    request = _request(limit=1, exclude=["m3"])
    before = copy.deepcopy(items)
    ranked = rank_candidates(items, request)
    assert [candidate["item"] for candidate in ranked] == ["m1", "m2", "m4", "m5", "m6"]
    assert select_candidates(items, request) == ranked[:1]
    assert items == before


def test_location_order_latest_seen_first_with_last_inserted_tiebreak() -> None:
    locations = [
        {"path": "/media/old.mov", "added_at": STAMP, "seen_at": STAMP},
        {"path": "/media/latest.mov", "added_at": STAMP, "seen_at": "2026-10-05T13:00:00Z"},
        {"path": "/media/tied.mov", "added_at": STAMP, "seen_at": STAMP},
    ]
    candidate = select_candidates([_item(locations=locations)], _request())[0]
    assert candidate["locations"] == ["/media/latest.mov", "/media/tied.mov", "/media/old.mov"]


@pytest.mark.parametrize(("media_type", "kind", "placement"), [
    ("clip", "video", "full_frame"), ("screen_recording", "video", "full_frame"),
    ("photo", "image", "overlay"), ("screenshot", "image", "overlay"),
])
def test_omitted_placement_uses_media_type_defaults(media_type, kind, placement) -> None:
    data = _request_data()
    del data["placement"]
    request = parse_request(data)
    assert request.placement is None
    item = _item(kind=kind)
    item["media_type"] = media_type
    candidate = select_candidates([item], request)[0]
    assert candidate["placement"] == placement
    assert f"placement from style: {placement}" in candidate["reasons"]


def test_style_placement_and_request_override_preserve_candidate_keys() -> None:
    from yap_studio.media.select import rank_candidates

    data = _request_data()
    del data["placement"]
    style = {"clip_placement": "overlay"}
    item = _item()
    candidate = rank_candidates([item], parse_request(data), style_broll=style)[0]
    assert candidate["placement"] == "overlay" and candidate["fit"] is None
    assert "placement from style: overlay" in candidate["reasons"]
    forced = select_candidates([item], _request(), style_broll=style)[0]
    assert forced["placement"] == "full_frame" and forced["fit"] == "cover"
    assert "placement from request: full_frame" in forced["reasons"]
    legacy = select_candidates([item], _request())[0]
    assert "placement from request: full_frame" in legacy["reasons"]
    assert set(forced) == set(legacy)


def test_split_candidate_keeps_photo_kind_and_suggests_insert(tmp_path: Path) -> None:
    from test_planner_broll import catalog_item, phrase_words

    from yap_studio.media.select import rank_candidates
    from yap_studio.planner.broll import suggest_broll

    item = catalog_item(tmp_path, kind="image")
    request = parse_request({**_request_data(), "placement": "split"})
    candidate = rank_candidates([item], request)[0]
    assert candidate["kind"] == "image" and candidate["fit"] == "split"
    suggestion = suggest_broll(phrase_words(), [item], {"broll": {"photo_placement": "split"}})[
        "phrases"][0]["candidates"][0]
    assert suggestion["request"]["kind"] == "insert"
    assert suggestion["request"]["fit"] == "split"


def test_avoid_tags_merge_request_caller_and_style_with_drop_reasons() -> None:
    from yap_studio.media.select import rank_candidates

    items = [_item(1, tags=["BUSY"]), _item(2, tags=["stock"]),
             _item(3, tags=["screen"]), _item(4)]
    before = copy.deepcopy(items)
    dropped = []
    request = _request(avoid_tags=["busy"])
    assert request.avoid_tags == ("busy",)
    candidates = rank_candidates(items, request, avoid_tags=["stock"],
                                 style_broll={"avoid_tags": ["screen"]}, dropped=dropped)
    assert [entry["item"] for entry in candidates] == ["m4"]
    assert dropped == [
        {"item": "m1", "reasons": ["avoided tags: BUSY"]},
        {"item": "m2", "reasons": ["avoided tags: stock"]},
        {"item": "m3", "reasons": ["avoided tags: screen"]},
    ]
    assert items == before


@pytest.mark.parametrize(("value", "field"), [
    ("busy", "avoid_tags"), (None, "avoid_tags"), ([1], "avoid_tags[0]"),
    ([" "], "avoid_tags[0]"), (("busy",), "avoid_tags"),
])
def test_avoid_tags_validation(value, field) -> None:
    with pytest.raises(ValidationError) as error:
        _request(avoid_tags=value)
    assert error.value.parameter == field


def test_purity_determinism_anchors_and_no_input_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    anchor = {
        "start": {"word": "t1:245", "text": "park"},
        "end": {"word": "t1:251", "edge": "end"},
        "opaque": [None, True, 123, 2.5, {"anything": "kept"}],
    }
    data = _request_data(anchor=anchor)
    request = parse_request(data)
    items = [_item(decisions=[_decision(start=0, end=1)]), _item(2, kind="image")]
    before_items, before_data = copy.deepcopy(items), copy.deepcopy(data)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("select_candidates performed I/O")

    with monkeypatch.context() as patched:
        patched.setattr(builtins, "open", forbidden)
        patched.setattr(os, "stat", forbidden)
        patched.setattr(subprocess, "Popen", forbidden)
        first = select_candidates(items, request)
        assert select_candidates(items, request) == first
        assert items == before_items and data == before_data
        assert all(candidate["anchor"] == anchor for candidate in first)
    first[0]["anchor"]["start"]["text"] = "changed"
    assert request.anchor == anchor and first[1]["anchor"] == anchor
    anchor["start"]["text"] = "changed input"
    assert request.anchor == before_data["anchor"]


@pytest.fixture(scope="session")
def sample_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synth.clip(tmp_path_factory.mktemp("select") / "clip.mov", audio=False)


@pytest.fixture
def described_clip(tmp_path: Path, sample_clip: Path) -> Path:
    source = Path(shutil.copyfile(sample_clip, tmp_path / "clip.mov"))
    add_paths([source], now=lambda: NOW)
    describe_item("m1", {
        "schema": DESCRIPTION_SCHEMA, "description": "Walking dogs through the park",
        "tags": ["dog", "park"], "ranges": [{"verdict": "approved", "reason": "steady"}],
    }, now=lambda: NOW)
    return source


def _file(tmp_path: Path, **changes: object) -> Path:
    target = tmp_path / "request.json"
    target.write_text(json.dumps(_request_data(**changes)), encoding="utf-8")
    return target


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_pick_real_media_path_and_no_writes(
    tmp_path: Path, described_clip: Path, as_json: bool, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    anchor = {"start": {"word": "t1:245", "text": "park"}, "end": {"word": "t1:251"}}
    target = _file(tmp_path, anchor=anchor)
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("pick tried to hash or write media")

    monkeypatch.setattr(receipt, "sha256_file", forbidden)
    flag = ["--json"] if as_json else []
    assert cli.main(["media", "pick", "--file", str(target), *flag]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    if as_json:
        result = json.loads(output.out)
        assert result["request"] == _request_data(anchor=anchor) and result["missing"] == []
        candidate = result["candidates"][0]
        assert candidate["item"] == "m1" and candidate["path"] == str(described_clip)
        assert candidate["approved"] is True and candidate["approved_by"] == ["d1"]
        assert candidate["anchor"] == anchor and candidate["score"] == 6
    else:
        assert "m1 0.000 to 2.000 s approved; score 6; matched: dog, park;" in output.out
        assert str(described_clip) in output.out and len(output.out.splitlines()) == 1
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files} == before
    assert [path for path in tmp_path.rglob("*") if path.is_file()] == files


def test_cli_deleted_location_reported_missing_without_catalog_write(
    tmp_path: Path, described_clip: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    target = _file(tmp_path)
    described_clip.unlink()
    before = catalog_path().read_bytes()
    assert cli.main(["media", "pick", "--file", str(target), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["candidates"] == [] and result["missing"] == ["m1"]
    assert cli.main(["media", "pick", "--file", str(target)]) == 0
    assert capsys.readouterr().out == "No candidates.\n"
    assert catalog_path().read_bytes() == before


def test_cli_pick_searches_past_missing_top_five(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    items = []
    for number in range(1, 7):
        source = tmp_path / f"{number}.mov"
        source.write_bytes(b"footage")
        items.append(_item(number, locations=[{
            "path": str(source), "added_at": STAMP, "seen_at": STAMP,
        }]))
        if number < 6:
            source.unlink()
    update_catalog(lambda doc: doc.update(items=items, next_id=7))
    target = _file(tmp_path)
    before = catalog_path().read_bytes()
    assert cli.main(["media", "pick", "--file", str(target), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [candidate["item"] for candidate in result["candidates"]] == ["m6"]
    assert result["candidates"][0]["path"] == str(tmp_path / "6.mov")
    assert result["missing"] == ["m1", "m2", "m3", "m4", "m5"]
    assert cli.main(["media", "pick", "--file", str(target)]) == 0
    assert capsys.readouterr().out.startswith("m6 ")
    assert catalog_path().read_bytes() == before


def test_cli_pick_missing_report_stops_at_last_returned_candidate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    items = []
    for number in range(1, 5):
        source = tmp_path / f"{number}.mov"
        if number in {2, 3}:
            source.write_bytes(b"footage")
        items.append(_item(number, locations=[{
            "path": str(source), "added_at": STAMP, "seen_at": STAMP,
        }]))
    update_catalog(lambda doc: doc.update(items=items, next_id=5))
    target = _file(tmp_path, limit=1)
    assert cli.main(["media", "pick", "--file", str(target), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert [candidate["item"] for candidate in result["candidates"]] == ["m2"]
    assert result["missing"] == ["m1"]


def test_cli_location_fallback_stat_only_even_if_contents_changed(
    tmp_path: Path, described_clip: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    latest = Path(shutil.copyfile(described_clip, tmp_path / "latest.mov"))
    add_paths([latest], now=lambda: NOW + timedelta(seconds=1))
    target = _file(tmp_path)
    # The command intentionally only stats paths; it does not verify content identity.
    latest.write_bytes(b"changed")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("pick tried to hash")

    monkeypatch.setattr(receipt, "sha256_file", forbidden)
    before = catalog_path().read_bytes()
    assert cli.main(["media", "pick", "--file", str(target), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["candidates"][0]["path"] == str(latest)
    latest.unlink()
    assert cli.main(["media", "pick", "--file", str(target), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["candidates"][0]["path"] == str(described_clip)
    assert catalog_path().read_bytes() == before


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("problem", ["missing", "json", "encoding", "schema", "length", "unknown"])
def test_cli_bad_request_exits_two_and_writes_nothing(
    tmp_path: Path, described_clip: Path, problem: str, as_json: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "request.json"
    if problem == "json":
        target.write_text("{bad JSON")
    elif problem == "encoding":
        target.write_bytes(b"\xff")
    elif problem != "missing":
        changes = {
            "schema": {"schema": "bad"}, "length": {"length_s": 4.1}, "unknown": {"extra": 1},
        }[problem]
        target = _file(tmp_path, **changes)
    before = catalog_path().read_bytes()
    flag = ["--json"] if as_json else []
    assert cli.main(["media", "pick", "--file", str(target), *flag]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    expected = "input_error" if problem in {"missing", "json", "encoding"} else "validation_error"
    if as_json:
        assert json.loads(output.err)["error"]["type"] == expected
    elif problem == "missing":
        assert "request file not found" in output.err
    elif problem in {"json", "encoding"}:
        assert "request file is not valid JSON" in output.err
    else:
        assert "Bad value for" in output.err
    assert catalog_path().read_bytes() == before


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_no_catalog_no_candidates_does_not_create_storage(
    tmp_path: Path, as_json: bool, capsys: pytest.CaptureFixture[str],
) -> None:
    target = _file(tmp_path)
    flag = ["--json"] if as_json else []
    assert cli.main(["media", "pick", "--file", str(target), *flag]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    if as_json:
        result = json.loads(output.out)
        assert result == {"request": _request_data(), "candidates": [], "missing": []}
    else:
        assert output.out == "No candidates.\n"
    assert not catalog_path().parent.exists()
