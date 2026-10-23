"""Golden plain-text and JSON review export tests."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from yap_studio.checks.model import format_time
from yap_studio.feedback.batch import _export
from yap_studio.feedback.versions import LayerWindow
from yap_studio.review import (
    Comment,
    MediaFacts,
    comment_from_dict,
    comment_to_dict,
    export_filename,
    format_text,
    format_threads,
    format_threads_summary,
    to_export_json,
)
from yap_studio.review.series import Reply, Thread

FPS = Fraction(30000, 1001)
FACTS = MediaFacts("final-v3.mp4", "a" * 64, 58.625, FPS, True)
EXPORTED_AT = datetime(2026, 9, 29, 15, 4, tzinfo=UTC)


def test_pin_percent_rounds_half_up_in_text_and_threads() -> None:
    comment = replace(_comments()[1], pin={"x": 0.425, "y": 0.125,
                                          "frame": 447, "time_s": 14.9149})
    suffix = " (pinned 43% across, 13% down)"
    assert suffix in format_text(FACTS, [comment], None, EXPORTED_AT.date())
    assert suffix in format_threads([Thread(1, comment, None, "open")], {1: FACTS})


def test_pin_exports_point_range_threads_and_variable_rate() -> None:
    point = replace(_comments()[1], pin={"x": 0.4213, "y": 0.3105,
                                         "frame": 447, "time_s": 14.9149})
    suffix = " (pinned 42% across, 31% down)"
    assert f"[Fix: caption]{suffix} Caption" in format_text(
        FACTS, [point], None, EXPORTED_AT.date())
    assert suffix in format_threads([Thread(1, point, None, "open")], {1: FACTS})
    exported = to_export_json(FACTS, [point], None, EXPORTED_AT)
    assert exported["version"] == 2 and exported["comments"][0]["pin"] == point.pin
    assert _export(exported)[0] == [point]
    ranged = replace(_comments()[3], target="element", pin={"x": 0.8, "y": 0.12,
                                                          "frame": 1356, "time_s": 45.2452})
    assert "[Fix: graphic] (pinned 80% across, 12% down at 0:45.2)" in format_text(
        FACTS, [ranged], None, EXPORTED_AT.date())
    variable = replace(ranged, start_frame=None, end_frame=None,
                       pin={**ranged.pin, "frame": None})
    assert "(pinned 80% across, 12% down at 0:45.2)" in format_text(
        replace(FACTS, cfr=False, fps=None), [variable], None, EXPORTED_AT.date())


def test_stored_pin_shape_is_strict_and_optional_request_stays_optional() -> None:
    record = comment_to_dict(_comments()[1])
    assert "pin" not in record
    pin = {"x": 0.4213, "y": 0.3105, "frame": 447, "time_s": 14.9149}
    for request in (True, False):
        raw = dict(record)
        if not request:
            raw.pop("request")
        assert comment_from_dict({**raw, "pin": pin}).pin == pin
    for bad in (None, {}, {"x": 0, "y": 0, "frame": 447},
                {**pin, "extra": 1}, {**pin, "frame": True}, {**pin, "frame": 448},
                {**pin, "x": float("nan")}, {**pin, "y": 2}, {**pin, "time_s": -1},
                {**pin, "time_s": True}, {**pin, "frame": None}, {**pin, "x": True},
                {**pin, "y": False}, {**pin, "time_s": float("inf")}):
        with pytest.raises(ValueError):
            comment_from_dict({**record, "pin": bad})
    legacy = {key: value for key, value in record.items()
              if key not in ("kind", "target", "request")}
    with pytest.raises(ValueError):
        comment_from_dict({**legacy, "label": "change", "pin": pin}, legacy=True)


@pytest.mark.parametrize(("kind", "label"), [
    ("insert", "B-roll"), ("overlay", "overlay"), ("graphic", "graphic"),
])
def test_known_element_kinds_label_current_and_earlier_exports(kind: str, label: str) -> None:
    comment = replace(_comments()[2], start_frame=66, end_frame=90, start_s=99)
    layers = (LayerWindow("broll-one", kind, 2.1, 3.2, None),)
    assert f"[Keep: {label}]" in format_text(FACTS, [comment], None, date(2026, 9, 29),
                                           layers=layers)
    assert f"[Keep: {label}]" in format_threads(
        [Thread(1, comment, None, "open")], {1: FACTS}, layers_by_version={1: layers},
    )
    assert "[Keep: graphic]" in format_text(FACTS, [comment], None, date(2026, 9, 29))


def test_element_export_ignores_caption_and_nonoverlapping_windows() -> None:
    comment = replace(_comments()[2], start_frame=90, end_frame=None)
    layers = (LayerWindow("broll-one", "insert", 2, float(90 / FPS), None),
              LayerWindow("caption", "caption", 2, 4, "words"))
    assert "[Keep: graphic]" in format_text(FACTS, [comment], None, date(2026, 9, 29),
                                          layers=layers)


def _comments() -> list[Comment]:
    return [
        Comment(
            "c1",
            "fix",
            None,
            'Cut this. Open on "here\'s the thing".',
            0,
            96,
            0.0,
            3.2032,
            "2026-09-29T14:51:10Z",
            "2026-09-29T14:51:10Z",
        ),
        Comment(
            "c2",
            "fix",
            "caption",
            "Caption covers my face here.\nMove it up or drop it for this line.",
            447,
            None,
            14.9149,
            None,
            "2026-09-29T14:52:40Z",
            "2026-09-29T14:52:40Z",
        ),
        Comment(
            "c3",
            "keep",
            "element",
            "This zoom lands. Keep it.",
            966,
            None,
            32.2322,
            None,
            "2026-09-29T14:53:00Z",
            "2026-09-29T14:53:00Z",
        ),
        Comment(
            "c4",
            "fix",
            None,
            "Music is louder than my voice through this bit.",
            1323,
            1538,
            44.1441,
            51.3179,
            "2026-09-29T14:54:00Z",
            "2026-09-29T14:54:00Z",
        ),
    ]


def test_text_matches_golden_example_byte_for_byte() -> None:
    expected = (
        "Feedback on final-v3.mp4 (0:58.6, 29.97 fps)\n"
        "From Alex Reviewer, 2026-09-29\n"
        "4 comments: 3 fixes, 1 keep\n"
        "\n"
        '0:00.0-0:03.2 [Fix] Cut this. Open on "here\'s the thing".\n'
        "0:14.9 [Fix: caption] Caption covers my face here.\n"
        "       Move it up or drop it for this line.\n"
        "0:32.2 [Keep: graphic] This zoom lands. Keep it.\n"
        "0:44.1-0:51.3 [Fix] Music is louder than my voice through this bit.\n"
    )
    assert (
        format_text(FACTS, list(reversed(_comments())), "Alex Reviewer", date(2026, 9, 29))
        == expected
    )


def test_series_text_header_matches_golden() -> None:
    facts = replace(FACTS, name="sample-video-v2.mp4")
    expected = (
        "Feedback on sample-video-v2.mp4, version 2 of 3 (0:58.6, 29.97 fps)\n"
        "1 comment: 1 fix\n\n"
        "0:14.9 [Fix: caption] Caption covers my face here.\n"
        "       Move it up or drop it for this line.\n"
    )
    assert format_text(facts, [_comments()[1]], None, date(2026, 9, 29), version=(2, 3)) == expected


def test_earlier_threads_golden_text_uses_each_version_timeline() -> None:
    comments = _comments()
    threads = [Thread(1, comments[0], None, "open")]
    for number, status in enumerate(("changed", "kept", "declined", "needs_clarification"), 1):
        reply = Reply(f"r{number}", 1, comments[1].id, 2, status,
                      f"Response {number}", None, "2026-10-05T22:00:00Z", "2026-10-05T22:00:00Z")
        threads.append(Thread(1, comments[1], reply, status))
    expected = (
        "Earlier versions: 2 open, 3 replied\n"
        'v1 0:00.0-0:03.2 [Fix] Cut this. Open on "here\'s the thing".\n'
        "  Open\n"
        "v1 0:14.9 [Fix: caption] Caption covers my face here.\n"
        "          Move it up or drop it for this line.\n"
        "  Changed in v2: Response 1\n"
        "v1 0:14.9 [Fix: caption] Caption covers my face here.\n"
        "          Move it up or drop it for this line.\n"
        "  Kept in v2: Response 2\n"
        "v1 0:14.9 [Fix: caption] Caption covers my face here.\n"
        "          Move it up or drop it for this line.\n"
        "  Declined in v2: Response 3\n"
        "v1 0:14.9 [Fix: caption] Caption covers my face here.\n"
        "          Move it up or drop it for this line.\n"
        "  Question in v2: Response 4\n"
    )
    assert format_threads(threads, {1: FACTS, 2: replace(FACTS, fps=Fraction(24))}) == expected
    assert format_threads([], {}) == ""


def test_earlier_threads_summary_counts_notes_and_open_states() -> None:
    comment = _comments()[0]
    reply = Reply("r1", 1, comment.id, 2, "changed", "Done", None,
                  "2026-10-05T22:00:00Z", "2026-10-05T22:00:00Z")
    assert format_threads_summary([]) == ""
    assert format_threads_summary([Thread(1, comment, None, "open")]) == (
        "Earlier versions: 1 note, 1 still open\n"
    )
    assert format_threads_summary([Thread(1, comment, reply, "changed")]) == (
        "Earlier versions: 1 note, all replied\n"
    )
    assert format_threads_summary([
        Thread(1, comment, None, "open"),
        Thread(1, comment, reply, "needs_clarification"),
        Thread(1, comment, reply, "changed"),
    ]) == "Earlier versions: 3 notes, 2 still open\n"


def test_earlier_threads_as_of_hides_replies_from_prior_versions() -> None:
    from yap_studio.review.export import earlier_threads

    comments = _comments()
    timestamp = "2026-10-05T22:00:00Z"
    old = Reply("r1", 1, comments[0].id, 2, "changed", "Done earlier", None,
                timestamp, timestamp)
    current = Reply("r2", 2, comments[1].id, 7, "changed", "Done in v7", None,
                    timestamp, timestamp)
    question = Reply("r3", 3, comments[2].id, 4, "needs_clarification", "Which?", None,
                     timestamp, timestamp)
    threads = [
        Thread(1, comments[0], old, "changed"),
        Thread(2, comments[1], current, "changed"),
        Thread(3, comments[2], question, "needs_clarification"),
        Thread(4, comments[3], None, "open"),
    ]
    facts = {number: FACTS for number in range(1, 8)}
    assert earlier_threads(threads, None) == threads
    assert earlier_threads(threads, 7) == threads[1:]
    default = format_threads(threads, facts)
    assert "Done earlier" in default
    assert format_threads(threads, facts, as_of=None) == default
    selected = format_threads(threads, facts, as_of=7)
    assert (format_threads_summary(threads) + selected).startswith(
        "Earlier versions: 4 notes, 2 still open\nv2 0:14.9 [Fix: caption]"
    )
    assert " open, " not in selected
    assert "Done earlier" not in selected
    assert "Done in v7" in selected
    assert "Question in v4" in selected
    assert "  Open\n" in selected
    assert format_threads([threads[0]], facts, as_of=7) == ""


def _reply_evidence(**fields: object) -> dict:
    return {
        "schema": "yap-studio/reply-evidence@1", "batch": "b1", "comment_version": "v1",
        "version": "v2", "media_sha256": "a" * 64, "disposition": "applied",
        "operations": [{"id": "op1", "type": "cut"}],
        "moment": {"status": "mapped", "start_s": 12.39, "end_s": 12.5,
                   "start_frame": 296, "end_frame": 300},
        "unmapped": None,
        "change": {"before": [{"id": "t1:1", "segment": "s1", "text": "really"}],
                   "after": [], "differences": ["removed", "words"]},
        "keep": None, "question": None, **fields,
    }


@pytest.mark.parametrize("rate", [Fraction(30), None])
def test_range_whose_last_word_was_cut_shows_wrong_join_partial_export(
    rate: Fraction | None,
) -> None:
    evidence = _reply_evidence(moment={"status": "partial", "start_s": 1.0, "end_s": 2.0,
                                      "start_frame": 30 if rate else None,
                                      "end_frame": 60 if rate else None})
    comment = replace(_comments()[1], text="Trim the end")
    reply = Reply("r1", 1, comment.id, 2, "changed", "Trimmed the end", evidence,
                  "2026-10-05T22:00:00Z", "2026-10-05T22:00:00Z")
    assert format_threads([Thread(1, comment, reply, "changed")],
                          {1: FACTS, 2: replace(FACTS, fps=rate, cfr=rate is not None)}) == (
        "Earlier versions: 0 open, 1 replied\n"
        "v1 0:14.9 [Fix: caption] Trim the end\n"
        "  Changed in v2 (partly removed, at 0:01.0): Trimmed the end\n"
    )


@pytest.mark.parametrize(("cfr", "rate", "frame", "moment_status", "status", "keep", "line"), [
    (True, Fraction(24), 296, "mapped", "changed", None, "Changed in v2 at 0:12.3"),
    (True, Fraction(25), 296, "removed", "changed", None,
     "Changed in v2 (removed, join at 0:11.8)"),
    (False, Fraction(25), 296, "mapped", "changed", None, "Changed in v2 at 0:12.3"),
    (False, None, None, "removed", "changed", None, "Changed in v2 (removed, join at 0:12.3)"),
    (True, Fraction(25), None, "time", "changed", None, "Changed in v2 at 0:12.3"),
    (True, None, 296, "mapped", "changed", None, "Changed in v2 at 0:12.3"),
    (True, Fraction(24), 296, "mapped", "kept", "kept",
     "Kept in v2 at 0:12.3 (checked: kept)"),
    (False, Fraction(25), 296, "mapped", "kept", "unmeasured",
     "Kept in v2 at 0:12.3 (checked: not measured)"),
    (False, None, None, "mapped", "kept", "broken", "Kept in v2 at 0:12.3 (checked: broken)"),
    (True, Fraction(24), 296, None, "kept", "unmeasured", "Kept in v2 (checked: not measured)"),
    (True, Fraction(24), 296, "moved", "declined", None, "Declined in v2 at 0:12.3"),
    (True, Fraction(24), 296, "ambiguous", "needs_clarification", None, "Question in v2 at 0:12.3"),
])
def test_reply_evidence_golden_lines_use_reply_version_facts(
    cfr: bool, rate: Fraction | None, frame: int | None, moment_status: str | None,
    status: str, keep: str | None, line: str,
) -> None:
    evidence = _reply_evidence()
    evidence["moment"].update(status=moment_status, start_frame=frame)
    if moment_status is None:
        evidence.update(moment=None, change=None, unmapped="whole video")
    if keep is not None:
        evidence["keep"] = {"status": keep, "detail": "Check detail"}
    comment = replace(_comments()[1], kind="keep" if status == "kept" else "fix",
                      text="Check this caption")
    reply = Reply("r1", 1, comment.id, 2, status, "Response\nMore detail", evidence,
                  "2026-10-05T22:00:00Z", "2026-10-05T22:00:00Z")
    open_count = 1 if status == "needs_clarification" else 0
    expected = (
        f"Earlier versions: {open_count} open, {1 - open_count} replied\n"
        f"v1 0:14.9 [{comment.kind.capitalize()}: caption] Check this caption\n"
        f"  {line}: Response\n"
        + " " * (len(line) + 4) + "More detail\n"
    )
    facts = {1: FACTS, 2: replace(FACTS, fps=rate, cfr=cfr)}
    assert format_threads([Thread(1, comment, reply, status)], facts) == expected


@pytest.mark.parametrize("evidence", [
    None, {"frames": [4, 8], "note": "confirmed"},
    _reply_evidence(schema="yap-studio/reply-evidence@2"),
    {"schema": "yap-studio/reply-evidence@1", "moment": {"start_s": 12.39}},
    _reply_evidence(batch=1), _reply_evidence(comment_version=1), _reply_evidence(version=2),
    _reply_evidence(media_sha256=[]), _reply_evidence(disposition=False),
    _reply_evidence(operations={}), _reply_evidence(operations=[None]),
    _reply_evidence(operations=[{"id": 2}]), _reply_evidence(operations=[{"id": "op1", "type": 2}]),
    _reply_evidence(unmapped=False), _reply_evidence(moment=[]),
    *[_reply_evidence(moment={"status": "mapped", "start_s": 12.39, "end_s": 12.5,
                             "start_frame": 296, "end_frame": 300, key: bad})
      for key, bad in [("status", "unknown"), ("start_s", "12.3"), ("end_s", True),
                       ("start_s", float("nan")), ("end_s", float("inf")), ("start_s", -1),
                       ("start_s", 10**400), ("end_s", 1), ("start_frame", True),
                       ("start_frame", 1.5), ("end_frame", "300"), ("end_frame", -1),
                       ("start_frame", 2**53), ("end_frame", 1)]],
    _reply_evidence(change=[]), _reply_evidence(change={"before": [], "after": []}),
    _reply_evidence(change={"before": {}, "after": [], "differences": []}),
    _reply_evidence(change={"before": [], "after": [None], "differences": []}),
    *[_reply_evidence(change={"before": [{"id": "t1:1", "segment": "s1", "text": "word",
                                         key: 2}], "after": [], "differences": []})
      for key in ("id", "segment", "text")],
    _reply_evidence(change={"before": [], "after": [], "differences": "words"}),
    _reply_evidence(change={"before": [], "after": [], "differences": ["unknown"]}),
    _reply_evidence(keep={}), _reply_evidence(keep={"status": "unknown", "detail": "Why"}),
    _reply_evidence(keep={"status": "kept", "detail": 2}), _reply_evidence(question=[]),
    *[_reply_evidence(question={"question": "Which line?", "answer": None, "option": None, key: 2})
      for key in ("question", "answer", "option")],
])
def test_reply_with_old_or_invalid_evidence_keeps_existing_golden_line(evidence: object) -> None:
    comment = replace(_comments()[1], text="Check this caption")
    reply = Reply("r1", 1, comment.id, 2, "changed", "Moved the caption", evidence,
                  "2026-10-05T22:00:00Z", "2026-10-05T22:00:00Z")
    expected = (
        "Earlier versions: 0 open, 1 replied\n"
        "v1 0:14.9 [Fix: caption] Check this caption\n"
        "  Changed in v2: Moved the caption\n"
    )
    # Legacy evidence needs no facts for the reply version.
    assert format_threads([Thread(1, comment, reply, "changed")], {1: FACTS}) == expected


def test_json_matches_golden_shape_and_frame_values() -> None:
    result = to_export_json(FACTS, list(reversed(_comments()[:3])), "Alex Reviewer", EXPORTED_AT)
    assert result == {
        "format": "yap-studio-feedback",
        "version": 2,
        "media": {
            "name": "final-v3.mp4",
            "sha256": "a" * 64,
            "duration_s": 58.625,
            "fps": "30000/1001",
        },
        "reviewer": "Alex Reviewer",
        "exported_at": "2026-09-29T15:04:00Z",
        "comments": [
            {
                "id": "c1",
                "start_frame": 0,
                "end_frame": 96,
                "start_s": 0.0,
                "end_s": 3.2032,
                "kind": "fix",
                "target": None,
                "text": 'Cut this. Open on "here\'s the thing".',
                "created_at": "2026-09-29T14:51:10Z",
                "updated_at": "2026-09-29T14:51:10Z",
                "request": None,
            },
            {
                "id": "c2",
                "start_frame": 447,
                "end_frame": None,
                "start_s": 14.9149,
                "end_s": None,
                "kind": "fix",
                "target": "caption",
                "text": "Caption covers my face here.\nMove it up or drop it for this line.",
                "created_at": "2026-09-29T14:52:40Z",
                "updated_at": "2026-09-29T14:52:40Z",
                "request": None,
            },
            {
                "id": "c3",
                "start_frame": 966,
                "end_frame": None,
                "start_s": 32.2322,
                "end_s": None,
                "kind": "keep",
                "target": "element",
                "text": "This zoom lands. Keep it.",
                "created_at": "2026-09-29T14:53:00Z",
                "updated_at": "2026-09-29T14:53:00Z",
                "request": None,
            },
        ],
    }
    assert list(result["comments"][0]) == [
        "id",
        "start_frame",
        "end_frame",
        "start_s",
        "end_s",
        "kind",
        "target",
        "text",
        "created_at",
        "updated_at",
        "request",
    ]


def test_comment_dict_round_trip_and_strict_keys() -> None:
    comment = _comments()[0]
    record = comment_to_dict(comment)
    assert comment_from_dict(record) == comment
    for bad in (
        {**record, "extra": 1},
        {key: value for key, value in record.items() if key != "text"},
    ):
        with pytest.raises(ValueError):
            comment_from_dict(bad)


def test_spelling_record_order_old_current_shape_and_exports() -> None:
    request = {"type": "spelling", "from": "tail net", "to": "tailnet"}
    comment = replace(_comments()[1], request=request)
    record = comment_to_dict(comment)
    assert list(record) == ["id", "start_frame", "end_frame", "start_s", "end_s", "kind",
                            "target", "text", "created_at", "updated_at", "request"]
    assert comment_from_dict(record, legacy=False) == comment
    record["request"]["to"] = "another"
    assert comment.request == request
    old = comment_to_dict(_comments()[1])
    del old["request"]
    assert comment_from_dict(old, legacy=False).request is None
    assert to_export_json(FACTS, [comment], None, EXPORTED_AT)["comments"][0]["request"] == request
    text = format_text(FACTS, [comment], None, date(2026, 9, 29))
    assert '[Fix: caption] (spelling: "tail net" -> "tailnet") Caption covers' in text
    assert '[Fix: caption] (spelling: "tail net" -> "tailnet")' in format_threads(
        [Thread(1, comment, None, "open")], {1: FACTS},
    )


def test_cut_request_text_and_json_file_round_trip(tmp_path: Path) -> None:
    request = {"type": "cut", "start": 0.0, "end": 3.2032, "words": [
        {"id": "t1:40", "segment": "s3", "text": "so"},
        {"id": "t1:41", "segment": "s3", "text": "basically"},
    ]}
    comment = replace(_comments()[0], target="story", text="Tighten this", request=request)
    assert comment_from_dict(comment_to_dict(comment), legacy=False) == comment
    exported = to_export_json(FACTS, [comment], None, EXPORTED_AT)
    review_file = tmp_path / "review.json"
    review_file.write_text(json.dumps(exported), encoding="utf-8")
    comments, fps = _export(json.loads(review_file.read_text(encoding="utf-8")))
    assert comments == [comment]
    assert fps == FPS
    assert to_export_json(FACTS, comments, None, EXPORTED_AT) == exported
    line = '0:00.0-0:03.2 [Fix: story] (cut: "so basically") Tighten this\n'
    assert format_text(FACTS, comments, None, date(2026, 9, 29)).endswith(line)
    assert format_threads([Thread(1, comment, None, "open")], {1: FACTS}) == (
        "Earlier versions: 1 open, 0 replied\nv1 " + line + "  Open\n"
    )


@pytest.mark.parametrize("length", [120, 121, 200])
def test_cut_request_text_truncates_after_120_characters(length: int) -> None:
    words = [{"id": "t1:40", "segment": "s3", "text": "so"},
             {"id": "t1:41", "segment": "s3", "text": "x" * (length - 3)}]
    request = {"type": "cut", "start": 0.0, "end": 3.2032, "words": words}
    comment = replace(_comments()[0], target="story", text="Tighten this", request=request)
    quote = '"so ' + ("x" * 114 + '..."' if length > 120 else "x" * 117 + '"')
    assert format_text(FACTS, [comment], None, date(2026, 9, 29)).endswith(
        f"0:00.0-0:03.2 [Fix: story] (cut: {quote}) Tighten this\n"
    )


def test_record_refuses_invalid_request() -> None:
    request = {"type": "spelling", "from": "wrong", "to": "right"}
    for record in (
        {**comment_to_dict(_comments()[0]), "request": request},
        {**comment_to_dict(_comments()[1]), "request": {**request, "to": "wrong"}},
    ):
        with pytest.raises(ValueError):
            comment_from_dict(record)


def test_json_uses_frames_as_canonical_seconds() -> None:
    stale = replace(_comments()[0], start_s=0.5, end_s=3.0)
    exported = to_export_json(FACTS, [stale], None, EXPORTED_AT)
    assert exported["comments"][0]["start_s"] == 0.0
    assert exported["comments"][0]["end_s"] == 3.2032
    assert stale.start_s == 0.5


@pytest.mark.parametrize(("field", "value"), [("kind", "maybe"), ("target", "other")])
def test_current_record_rejects_invalid_kind_or_target(field: str, value: str) -> None:
    record = {**comment_to_dict(_comments()[0]), field: value}
    for legacy in (None, False):
        with pytest.raises(ValueError):
            comment_from_dict(record, legacy=legacy)


@pytest.mark.parametrize(("label", "kind"), [("like", "keep"), ("change", "fix")])
def test_old_record_maps_kind_and_has_no_target(label: str, kind: str) -> None:
    comment = replace(_comments()[0], kind=kind, target=None)
    record = comment_to_dict(comment)
    legacy_record = {key: value for key, value in record.items()
                     if key not in ("kind", "target", "request")}
    legacy_record["label"] = label
    assert comment_from_dict(legacy_record) == comment
    assert comment_from_dict(legacy_record, legacy=True) == comment
    assert comment_from_dict(record, legacy=False) == comment
    for data, legacy in ((legacy_record, False), (record, True)):
        with pytest.raises(ValueError):
            comment_from_dict(data, legacy=legacy)
    with pytest.raises(ValueError):
        comment_from_dict({**legacy_record, "label": "other"})


def test_only_the_two_exact_record_shapes_are_valid() -> None:
    record = comment_to_dict(_comments()[0])
    for bad in (
        {key: value for key, value in record.items() if key not in ("kind", "target")},
        {key: value for key, value in record.items() if key != "target"},
        {**record, "label": "change"},
        {**{key: value for key, value in record.items() if key != "kind"}, "label": "change"},
    ):
        with pytest.raises(ValueError):
            comment_from_dict(bad)


@pytest.mark.parametrize(
    ("kinds", "counts"),
    [
        (["fix"], "1 comment: 1 fix"),
        (["keep"], "1 comment: 1 keep"),
        (["fix", "fix"], "2 comments: 2 fixes"),
        (["keep", "keep"], "2 comments: 2 keeps"),
        (["keep", "fix"], "2 comments: 1 fix, 1 keep"),
    ],
)
def test_counts_singular_plural_order_and_zero_parts(kinds: list[str], counts: str) -> None:
    comments = [
        replace(comment, kind=kind) for comment, kind in zip(_comments(), kinds, strict=False)
    ]
    assert format_text(FACTS, comments, None, date(2026, 9, 29)).splitlines()[1] == counts


@pytest.mark.parametrize(
    ("target", "tag"),
    [(None, "Keep"), ("story", "Keep: story"), ("speech", "Keep: speech"),
     ("caption", "Keep: caption"), ("element", "Keep: graphic"), ("video", "Keep: whole video")],
)
def test_all_target_tags_and_json_fields(target: str | None, tag: str) -> None:
    comment = replace(_comments()[0], kind="keep", target=target)
    assert f"[{tag}]" in format_text(FACTS, [comment], None, date(2026, 9, 29))
    exported = to_export_json(FACTS, [comment], None, EXPORTED_AT)
    assert exported["version"] == 2
    assert exported["comments"][0]["kind"] == "keep"
    assert exported["comments"][0]["target"] == target
    assert "label" not in exported["comments"][0]


def test_floored_times_variable_rate_header_and_full_precision_json() -> None:
    assert format_time(59.99) == "0:59.9"
    assert format_time(3665.2) == "61:05.2"
    facts = replace(FACTS, fps=None, cfr=False, duration_s=3665.2)
    comment = replace(
        _comments()[0], start_frame=None, end_frame=None, start_s=59.99, end_s=3665.199999
    )
    assert format_text(facts, [comment], None, date(2026, 9, 29)) == (
        "Feedback on final-v3.mp4 (61:05.2, variable frame rate)\n"
        "1 comment: 1 fix\n\n"
        '0:59.9-61:05.1 [Fix] Cut this. Open on "here\'s the thing".\n'
    )
    exported = to_export_json(facts, [comment], None, EXPORTED_AT)
    assert exported["media"]["fps"] is None
    assert exported["comments"][0]["start_s"] == 59.99
    assert exported["comments"][0]["end_s"] == 3665.199999
    assert exported["comments"][0]["start_frame"] is None


def test_empty_review_plural_rules_no_reviewer_and_utc_conversion() -> None:
    assert format_text(FACTS, [], None, date(2026, 9, 29)) == (
        "Feedback on final-v3.mp4 (0:58.6, 29.97 fps)\n0 comments\n"
    )
    keeps = [replace(comment, kind="keep") for comment in _comments()[:2]]
    result = format_text(FACTS, keeps, None, date(2026, 9, 29))
    assert "2 comments: 2 keeps\n" in result
    assert "From " not in result
    offset = timezone(timedelta(hours=-5))
    exported = to_export_json(FACTS, [], None, datetime(2026, 9, 29, 10, 4, tzinfo=offset))
    assert exported["exported_at"] == "2026-09-29T15:04:00Z"
    with pytest.raises(ValueError):
        to_export_json(FACTS, [], None, datetime(2026, 9, 29, 15, 4))


def test_export_filename_uses_video_stem() -> None:
    assert export_filename("final-v3.mp4", "txt") == "final-v3-feedback.txt"
    assert export_filename("folder/final-v3.mp4", ".json") == "final-v3-feedback.json"
