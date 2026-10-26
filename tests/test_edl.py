"""Canonical edit-list model and JSON reader tests."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from yap_studio.edl import (
    EditList,
    Evidence,
    Hold,
    JoinAssertion,
    PictureRegion,
    Segment,
    Source,
    Still,
    load_edit_list,
)
from yap_studio.errors import EditListError, InputError, ValidationError


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_yap_single_source_runs_and_fps(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "edit.json",
        {
            "source": "raw/take-01.mov",
            "fps": "30000/1001",
            "segments": [
                {"start": 0, "end": 1, "text": "opening"},
                {"runs": [[2.42, 3], [4, 5]], "text": "two runs"},
            ],
        },
    )
    edit_list = load_edit_list(path)

    assert edit_list.format == "yap"
    assert edit_list.fps == Fraction(30000, 1001)
    assert edit_list.sources["main"].path == tmp_path / "raw/take-01.mov"
    assert [segment.index for segment in edit_list.segments] == [1, 2, 3]
    assert edit_list.segments[1].start == Fraction(121, 50)
    assert edit_list.segments[1].label == "two runs"
    assert [cut.index for cut in edit_list.cuts()] == [1, 2]
    assert edit_list.to_dict()["segments"][1]["start"] == "121/50"


def test_yap_sources_map_requires_segment_source(tmp_path: Path) -> None:
    good = _write_json(
        tmp_path / "good.json",
        {
            "sources": {"a": "a.mov", "b": "b.mov"},
            "segments": [
                {"source": "a", "start": 0, "end": 1},
                {"source": "b", "start": 2, "end": 3},
            ],
        },
    )
    assert [segment.source_id for segment in load_edit_list(good).segments] == ["a", "b"]

    bad = _write_json(
        tmp_path / "bad.json",
        {"sources": {"a": "a.mov", "b": "b.mov"}, "segments": [{"start": 0, "end": 1}]},
    )
    with pytest.raises(EditListError, match="must name a source"):
        load_edit_list(bad)


def test_yap_planned_boundary_words_round_trip(tmp_path: Path) -> None:
    """Intent survives reading and serialization, including grouped runs."""
    path = _write_json(tmp_path / "words.json", {
        "source": "take.mov",
        "segments": [
            {"start": 0, "end": 1, "first_word": "Bright", "last_word": "hills.",
             "first_word_anchor_s": 0.12, "last_word_anchor_s": 0.8},
            {"runs": [[2, 3], [4, 5], [6, 7]], "first_word": "Seven", "last_word": "garden",
             "first_word_anchor_s": 2.12, "last_word_anchor_s": 6.8},
            {"start": 8, "end": 9},
        ],
    })
    edit = load_edit_list(path)
    expected = [("Bright", "hills."), ("Seven", None), (None, None),
                (None, "garden"), (None, None)]
    assert [(row.first_word, row.last_word) for row in edit.segments] == expected
    assert [(row.first_word_anchor_s, row.last_word_anchor_s) for row in edit.segments] == [
        (Fraction("0.12"), Fraction("0.8")), (Fraction("2.12"), None), (None, None),
        (None, Fraction("6.8")), (None, None),
    ]
    for row, (first, last) in zip(edit.to_dict()["segments"], expected, strict=True):
        assert row.get("first_word") == first
        assert row.get("last_word") == last
    written = _write_json(tmp_path / "round-trip.json", {
        "source": "take.mov",
        "segments": [
            {"start": str(row.start), "end": str(row.end),
             **{key: value for key, value in row.to_dict().items()
                if key in {"first_word", "last_word", "first_word_anchor_s", "last_word_anchor_s"}
                and value is not None}}
            for row in edit.segments
        ],
    })
    assert load_edit_list(written).segments == edit.segments


@pytest.mark.parametrize("field", ["first_word_anchor_s", "last_word_anchor_s"])
@pytest.mark.parametrize("value", [-1, None, True, "nan", "inf", [], {}])
def test_yap_refuses_invalid_word_anchor_times(tmp_path: Path, field: str, value: object) -> None:
    path = _write_json(tmp_path / "bad-anchor.json", {
        "source": "take.mov", "segments": [
            {"start": 0, "end": 1, "first_word": "bright", "last_word": "hills", field: value},
        ],
    })
    with pytest.raises(EditListError, match=field):
        load_edit_list(path)


@pytest.mark.parametrize("field", ["first_word", "last_word"])
@pytest.mark.parametrize("value", ["", "   ", None, False, 42, [], {}])
def test_yap_refuses_invalid_planned_words(tmp_path: Path, field: str, value: object) -> None:
    path = _write_json(tmp_path / "bad.json", {
        "source": "take.mov", "segments": [{"start": 0, "end": 1, field: value}],
    })
    with pytest.raises(EditListError, match=field) as error:
        load_edit_list(path)
    assert error.value.suggested_action


@pytest.mark.parametrize("kind", ["ranges", "fcpxml", "xmeml"])
def test_other_formats_have_no_planned_words(tmp_path: Path, kind: str) -> None:
    if kind == "ranges":
        path = _write_json(tmp_path / "edit.json", {
            "sources": {"main": "take.mov"},
            "ranges": [{"source": "main", "start": 0, "end": 1,
                        "first_word": "ignored", "last_word": "ignored"}],
        })
    else:
        path = tmp_path / f"edit.{kind}"
        body = ('''<fcpxml version="1.10"><resources>
            <format id="r1" frameDuration="1/30s"/>
            <asset id="r2" start="0s"><media-rep src="take.mov"/></asset>
            </resources><library><event><project><sequence format="r1"><spine>
            <asset-clip ref="r2" offset="0s" start="0s" duration="1s"/>
            </spine></sequence></project></event></library></fcpxml>'''
            if kind == "fcpxml" else '''<xmeml version="4"><sequence>
            <rate><timebase>30</timebase><ntsc>FALSE</ntsc></rate><media><video><track>
            <clipitem id="clip-1"><start>0</start><end>30</end><in>0</in><out>30</out>
            <file id="file-1"><pathurl>take.mov</pathurl></file></clipitem>
            </track></video></media></sequence></xmeml>''')
        path.write_text(body, encoding="utf-8")
    assert load_edit_list(path).segments
    assert all(row.first_word is row.last_word is None for row in load_edit_list(path).segments)


def test_yap_source_object_carries_audio_stream(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "edit.json", {
        "sources": {"speech": {"path": "take.mov", "audio_stream": 1}, "legacy": "other.mov"},
        "segments": [{"source": "speech", "start": 0, "end": 1}],
    })
    sources = load_edit_list(path).sources
    assert sources["speech"].audio_stream == 1
    assert sources["speech"].to_dict() == {
        "id": "speech", "path": str(tmp_path / "take.mov"), "audio_stream": 1,
    }
    assert sources["legacy"].audio_stream is None


@pytest.mark.parametrize("stream", [-1, True, 1.5, "1"])
def test_yap_source_object_rejects_invalid_audio_stream(tmp_path: Path, stream: object) -> None:
    path = _write_json(tmp_path / "edit.json", {
        "sources": {"main": {"path": "take.mov", "audio_stream": stream}},
        "segments": [{"start": 0, "end": 1}],
    })
    with pytest.raises(EditListError, match="audio_stream"):
        load_edit_list(path)


def test_ranges_multi_source_overlays_labels_and_missing_files(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "ranges.json",
        {
            "sources": {"take1": "raw/take1.mov", "take2": "/media/project/take2.mov"},
            "ranges": [
                {"source": "take1", "start": 2.42, "end": 6.85, "beat": "HOOK"},
                {"source": "take2", "start": 7, "end": 8, "note": "close"},
            ],
            "overlays": [
                {"file": "graphics/title.mp4", "start_in_output": 0.0, "duration": 5.0},
                {"file": "graphics/bad.mp4", "duration": "unknown"},
            ],
        },
    )
    edit_list = load_edit_list(path)

    assert edit_list.format == "ranges"
    assert edit_list.sources["take1"].path == tmp_path / "raw/take1.mov"
    assert edit_list.sources["take2"].path == Path("/media/project/take2.mov")
    assert [segment.label for segment in edit_list.segments] == ["HOOK", "close"]
    assert edit_list.picture_masks == [(Fraction(0), Fraction(5))]
    assert len(edit_list.warnings) == 1


def test_ranges_holds_are_read_in_output_frames(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "holds.json",
        {
            "sources": {"main": "take.mov"},
            "ranges": [{"source": "main", "start": 0, "end": 2}],
            "holds": [
                {"start_frame": 60, "end_frame": 105, "audio": "silence"},
                {"start_frame": 0, "end_frame": 15},
            ],
        },
    )
    edit_list = load_edit_list(path)
    assert edit_list.holds == [Hold(2, 0, 15, "silence"), Hold(1, 60, 105, "silence")]
    assert edit_list.holds[1].frames == 45
    assert edit_list.to_dict()["holds"] == [
        {"index": 2, "start_frame": 0, "end_frame": 15, "audio": "silence"},
        {"index": 1, "start_frame": 60, "end_frame": 105, "audio": "silence"},
    ]
    without = _write_json(
        tmp_path / "plain.json",
        {"sources": {"main": "take.mov"}, "ranges": [{"source": "main", "start": 0, "end": 1}]},
    )
    assert load_edit_list(without).holds == []


def test_ranges_stills_are_read_in_output_frames(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "stills.json", {
        "sources": {"main": "take.mov"},
        "ranges": [{"source": "main", "start": 0, "end": 2}],
        "stills": [{"id": "broll-diagram", "start_frame": 30, "end_frame": 60}],
    })
    edit = load_edit_list(path)
    assert edit.stills == [Still("broll-diagram", 30, 60)]
    assert edit.to_dict()["stills"] == [
        {"id": "broll-diagram", "start_frame": 30, "end_frame": 60}
    ]
    assert load_edit_list(_write_json(tmp_path / "plain-stills.json", {
        "sources": {"main": "take.mov"},
        "ranges": [{"source": "main", "start": 0, "end": 2}],
    })).stills == []


@pytest.mark.parametrize("stills,message", [
    ({"id": "diagram"}, "stills must be a list"),
    (["diagram"], "Still 1 must be an object"),
    ([{"start_frame": 0, "end_frame": 30}], "Still 1 needs an id"),
    ([{"id": "", "start_frame": 0, "end_frame": 30}], "Still 1 needs an id"),
    ([{"id": "diagram", "start_frame": -1, "end_frame": 30}], "frame 0 or later"),
    ([{"id": "diagram", "start_frame": 30, "end_frame": 30}], "end after it starts"),
    ([{"id": "diagram", "start_frame": 0.5, "end_frame": 30}], "whole frame number"),
])
def test_invalid_stills_are_rejected(tmp_path: Path, stills: object, message: str) -> None:
    path = _write_json(tmp_path / "bad-stills.json", {
        "sources": {"main": "take.mov"},
        "ranges": [{"source": "main", "start": 0, "end": 2}],
        "stills": stills,
    })
    with pytest.raises(EditListError, match=message):
        load_edit_list(path)


@pytest.mark.parametrize(
    "holds,message",
    [
        ({"start_frame": 0, "end_frame": 1}, "holds must be a list"),
        (["hold"], "Hold 1 must be an object"),
        ([{"start_frame": 1.5, "end_frame": 3}], "whole frame number"),
        ([{"start_frame": True, "end_frame": 3}], "whole frame number"),
        ([{"start_frame": 10, "end_frame": 10}], "end after it starts"),
        ([{"start_frame": -1, "end_frame": 10}], "frame 0 or later"),
        ([{"start_frame": 0, "end_frame": 10, "audio": "music"}], 'audio must be "silence"'),
        (
            [{"start_frame": 0, "end_frame": 10}, {"start_frame": 5, "end_frame": 20}],
            "overlap",
        ),
    ],
)
def test_invalid_holds_are_rejected(tmp_path: Path, holds: object, message: str) -> None:
    path = _write_json(
        tmp_path / "bad-holds.json",
        {
            "sources": {"main": "take.mov"},
            "ranges": [{"source": "main", "start": 0, "end": 1}],
            "holds": holds,
        },
    )
    with pytest.raises(EditListError, match=message):
        load_edit_list(path)


def test_ranges_join_assertions_are_read(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "joins.json",
        {
            "sources": {"main": "take.mov"},
            "ranges": [{"source": "main", "start": 0, "end": 2}],
            "joins": [
                {
                    "out_frame": 274,
                    "before": " going kind of viral ",
                    "after": "Now if you don't",
                    "absent": ["mansplainers"],
                    "once": ["kind of viral"],
                },
                {"out_frame": 900, "after": "Next line"},
            ],
        },
    )
    edit_list = load_edit_list(path)
    assert edit_list.joins == [
        JoinAssertion(
            1, 274, "going kind of viral", "Now if you don't", ("mansplainers",), ("kind of viral",)
        ),
        JoinAssertion(2, 900, "", "Next line"),
    ]
    assert edit_list.to_dict()["joins"][0]["absent"] == ["mansplainers"]


@pytest.mark.parametrize(
    "joins,message",
    [
        ({"out_frame": 1}, "joins must be a list"),
        ([{"out_frame": "274", "before": "a"}], "whole frame number"),
        ([{"out_frame": -1, "before": "a"}], "0 or later"),
        ([{"out_frame": 1}], "needs the words"),
        ([{"out_frame": 1, "before": 3}], "must be text"),
        ([{"out_frame": 1, "before": "a", "absent": "b"}], "list of non-empty phrases"),
        ([{"out_frame": 1, "before": "a", "once": [""]}], "list of non-empty phrases"),
    ],
)
def test_invalid_join_assertions_are_rejected(tmp_path: Path, joins: object, message: str) -> None:
    path = _write_json(
        tmp_path / "bad-joins.json",
        {
            "sources": {"main": "take.mov"},
            "ranges": [{"source": "main", "start": 0, "end": 1}],
            "joins": joins,
        },
    )
    with pytest.raises(EditListError, match=message):
        load_edit_list(path)


def test_ranges_composite_and_evidence(tmp_path: Path) -> None:
    digest = "AB" * 32
    path = _write_json(
        tmp_path / "composite.json",
        {
            "sources": {"main": "master.mp4"},
            "ranges": [{"source": "main", "start": 0, "end": 2}],
            "composite": True,
            "evidence": [
                {"path": "checks/cuts.json", "sha256": digest},
                {"path": "/media/project/review.txt", "sha256": "0" * 64},
            ],
        },
    )
    edit_list = load_edit_list(path)
    assert edit_list.composite is True
    assert edit_list.evidence == [
        Evidence(tmp_path / "checks/cuts.json", digest.lower()),
        Evidence(Path("/media/project/review.txt"), "0" * 64),
    ]
    assert edit_list.to_dict()["composite"] is True
    plain = _write_json(
        tmp_path / "plain.json",
        {"sources": {"main": "take.mov"}, "ranges": [{"source": "main", "start": 0, "end": 1}]},
    )
    assert load_edit_list(plain).composite is False
    assert load_edit_list(plain).evidence == []


@pytest.mark.parametrize(
    "extra,message",
    [
        ({"composite": "yes"}, "composite must be true or false"),
        ({"evidence": {"path": "a"}}, "evidence must be a list"),
        ({"evidence": [{"sha256": "0" * 64}]}, "needs a path"),
        ({"evidence": [{"path": "a", "sha256": "abc"}]}, "64 hexadecimal"),
    ],
)
def test_invalid_composite_and_evidence_are_rejected(
    tmp_path: Path, extra: dict, message: str
) -> None:
    path = _write_json(
        tmp_path / "bad.json",
        {
            "sources": {"main": "take.mov"},
            "ranges": [{"source": "main", "start": 0, "end": 1}],
            **extra,
        },
    )
    with pytest.raises(EditListError, match=message):
        load_edit_list(path)


def test_overlay_regions_are_read_in_output_pixels(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "regions.json",
        {
            "sources": {"main": "take.mov"},
            "ranges": [{"source": "main", "start": 0, "end": 2}],
            "overlays": [
                {"start_in_output": 5.967, "duration": 168, "region": [0, 1480, 1080, 200]},
                {"start_in_output": 1, "duration": 2},
                {"start_in_output": 3, "duration": 1, "region": [0, 0, -5, 10]},
            ],
        },
    )
    edit_list = load_edit_list(path)
    assert edit_list.picture_regions == [
        PictureRegion(Fraction("5.967"), Fraction("173.967"), 0.0, 1480.0, 1080.0, 200.0)
    ]
    assert edit_list.picture_regions[0].rect == (0.0, 1480.0, 1080.0, 200.0)
    # A time-only overlay, and one whose region cannot be read, mask the whole picture.
    assert edit_list.picture_masks == [(Fraction(1), Fraction(3)), (Fraction(3), Fraction(4))]
    assert any("region could not be read" in warning for warning in edit_list.warnings)
    assert edit_list.to_dict()["picture_regions"][0]["region"] == [0.0, 1480.0, 1080.0, 200.0]


def test_source_overrides(tmp_path: Path) -> None:
    single = _write_json(
        tmp_path / "single.json",
        {"source": "old.mov", "segments": [{"start": 0, "end": 1}]},
    )
    assert load_edit_list(single, source_overrides=["new.mov"]).sources["main"].path == (
        tmp_path / "new.mov"
    )

    multiple = _write_json(
        tmp_path / "multiple.json",
        {
            "sources": {"a": "a.mov", "b": "b.mov"},
            "ranges": [{"source": "a", "start": 0, "end": 1}],
        },
    )
    overridden = load_edit_list(multiple, source_overrides=["b=/media/project/relinked.mov"])
    assert overridden.sources["b"].path == Path("/media/project/relinked.mov")
    with pytest.raises(ValidationError, match="unknown source id"):
        load_edit_list(multiple, source_overrides=["missing=x.mov"])
    with pytest.raises(ValidationError, match="bare path"):
        load_edit_list(multiple, source_overrides=["x.mov"])


@pytest.mark.parametrize("override", ["new.mov", "main=new.mov"])
def test_source_overrides_preserve_audio_stream(tmp_path: Path, override: str) -> None:
    path = _write_json(
        tmp_path / "edit.json",
        {
            "sources": {"main": {"path": "a.mov", "audio_stream": 1}},
            "segments": [{"start": 0, "end": 1}],
        },
    )
    source = load_edit_list(path, source_overrides=[override]).sources["main"]
    assert source.path == tmp_path / "new.mov"
    assert source.audio_stream == 1


def test_invalid_segment_and_unknown_content_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(EditListError, match="Segment 1"):
        EditList(
            "yap",
            tmp_path / "edit.json",
            {"main": Source("main", None)},
            [Segment(1, "main", Fraction(2), Fraction(1))],
            None,
            [],
            [],
        )

    unknown = tmp_path / "unknown.txt"
    unknown.write_text("not an edit list", encoding="utf-8")
    with pytest.raises(EditListError) as error:
        load_edit_list(unknown)
    assert error.value.code == "unsupported_edit_list"

    with pytest.raises(InputError):
        load_edit_list(tmp_path / "missing.json")


@pytest.mark.parametrize("bundle", [False, True])
def test_edit_lists_larger_than_64_mb_are_rejected_before_reading(
    tmp_path: Path, bundle: bool
) -> None:
    requested = tmp_path / ("large.fcpxmld" if bundle else "large.fcpxml")
    if bundle:
        requested.mkdir()
        source = requested / "Info.fcpxml"
    else:
        source = requested
    with source.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)

    with pytest.raises(InputError, match="Edit list is larger than 64 MB.") as error:
        load_edit_list(requested)

    assert error.value.code == "invalid_input"
