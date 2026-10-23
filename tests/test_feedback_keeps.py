"""Keep checks protect scoped words, pauses, captions, and on-screen layers."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest
from test_feedback_batch import _cut, _export, _layers
from test_feedback_batch import feedback_project as feedback_project
from test_feedback_locate import LAYERS, WORDS, _comment, _locate
from test_feedback_versions import _render_media, _voice

from yap_studio.feedback import keeps as keeps_module
from yap_studio.feedback.batch import FeedbackError, build_batch
from yap_studio.feedback.keeps import (
    KeepBrokenError,
    check_keeps,
    check_version,
    element_keep,
    keep_constraint,
    require_keeps,
)
from yap_studio.feedback.versions import LayerWindow, save_version
from yap_studio.project.model import CutWord, CutWords


@pytest.mark.parametrize("kind", ["insert", "overlay"])
def test_element_keep_captures_and_protects_broll_identity(kind: str) -> None:
    comment = _comment("c3", kind="keep", target="element")
    layers = (LayerWindow("broll-one", kind, 2.1, 2.7, None),)
    constraint = keep_constraint(comment, _locate(comment, layers=layers), version="v1")
    assert element_keep([constraint], "broll-one") is constraint
    assert element_keep([constraint], "broll-other") is None
    assert _check(constraint, layers=layers).ok
    assert not _check(constraint, layers=()).ok
    constraint["target"] = "story"
    assert element_keep([constraint], "broll-one") is None


def _constraint(target="story", *, start=66, end=93):
    comment = _comment("c3", kind="keep", target=target, start_frame=start, end_frame=end)
    return keep_constraint(comment, _locate(comment), version="v2")


def _check(constraint, *, words=WORDS, layers=LAYERS):
    return check_keeps([constraint], words, layers, fps=Fraction(30))


def test_missing_word_names_comment_version_text_and_first_missing_word() -> None:
    constraint = _constraint()
    removed = CutWords("output", {}, WORDS.words[:3])
    with pytest.raises(KeepBrokenError) as error:
        require_keeps([constraint], removed, None, fps=Fraction(30))
    message = str(error.value)
    assert 'Keep comment c3 on v2 ("Preserve this material") is broken:' in message
    assert 'word t1:2 "End" in s2 is missing from the new cut' in message
    assert error.value.code == "keep_broken"
    assert "Restore" in error.value.suggested_action and "creator" in error.value.suggested_action
    assert error.value.report.ok is False


def test_story_keep_allows_other_words_to_move_and_kept_run_to_retime() -> None:
    changed = CutWords("output", {}, (
        CutWord("t1:0", "s2", "Start", 0.1, 0.4),
        CutWord("t1:2", "s2", "End", 0.6, 1.0),
        CutWord("t1:1", "s1", "really", 2.0, 2.5),
        CutWord("t1:0", "s1", "Start", 3.0, 3.5),
    ))
    report = require_keeps([_constraint()], changed, None, fps=Fraction(30))
    assert report.ok and report.results[0].status == "kept"
    assert report.to_dict()["ok"] is True


@pytest.mark.parametrize("words", [
    CutWords("output", {}, (WORDS.words[3], WORDS.words[2], WORDS.words[0], WORDS.words[1])),
    CutWords("output", {}, (WORDS.words[0], WORDS.words[2], WORDS.words[1], WORDS.words[3])),
])
def test_reordered_or_interrupted_kept_range_is_broken(words: CutWords) -> None:
    result = _check(_constraint(), words=words).results[0]
    assert result.status == "broken"
    assert "order changed or other words were inserted" in result.detail


@pytest.mark.parametrize("target", [None, "story", "speech"])
def test_silence_keep_records_neighbour_words_and_pause(target: str | None) -> None:
    constraint = _constraint(target, start=18, end=None)
    assert constraint["target"] == (target or "story")
    assert [(word["id"], word["segment"]) for word in constraint["words"]] == [
        ("t1:0", "s1"), ("t1:1", "s1"),
    ]
    assert constraint["gaps"] == pytest.approx([0.3])
    assert constraint["pause"] is True
    assert _check(constraint).ok


@pytest.mark.parametrize("delta,expected", [(0.02, "kept"), (1 / 30, "kept"), (0.1, "broken")])
@pytest.mark.parametrize("target", [None, "story", "speech"])
def test_pause_changes_are_checked_to_one_frame(
    delta: float, expected: str, target: str | None,
) -> None:
    constraint = _constraint(target, start=18, end=None)
    shifted = CutWords("output", {}, (
        WORDS.words[0], CutWord("t1:1", "s1", "really", 0.8 + delta, 1.1 + delta),
        *WORDS.words[2:],
    ))
    result = _check(constraint, words=shifted).results[0]
    assert result.status == (
        "unmeasured" if target == "speech" and expected == "kept" else expected
    )
    if expected == "broken":
        assert "t1:0" in result.detail and "t1:1" in result.detail
        assert "0.3000s" in result.detail and "0.4000s" in result.detail


@pytest.mark.parametrize("target", [None, "story", "speech"])
@pytest.mark.parametrize("new_words", [WORDS, CutWords("output", {}, ())])
def test_keep_without_words_or_neighbours_is_unmeasured(
    target: str | None, new_words: CutWords,
) -> None:
    comment = _comment("c3", kind="keep", target=target, start_frame=18)
    location = _locate(comment, words=CutWords("output", {}, ()))
    constraint = keep_constraint(comment, location, version="v2")
    assert constraint["words"] == constraint["gaps"] == []
    report = require_keeps([constraint], new_words, None, fps=Fraction(30))
    assert report.ok and report.results[0].status == "unmeasured"
    assert report.results[0].detail == "no words to check"


def test_speech_range_records_and_checks_consecutive_word_gaps() -> None:
    constraint = _constraint("speech")
    assert constraint["gaps"] == pytest.approx([0.3])
    assert constraint["pause"] is False
    moved = CutWords("output", {}, (*WORDS.words[:3], CutWord("t1:2", "s2", "End", 3.0, 3.3)))
    assert _check(constraint, words=moved).results[0].status == "broken"


@pytest.mark.parametrize("target", ["caption", "element"])
def test_layer_keeps_unmeasured_without_data_and_checked_with_data(target: str) -> None:
    constraint = _constraint(target)
    assert [layer["id"] for layer in constraint["layers"]] == [
        "caption" if target == "caption" else "card",
    ]
    result = _check(constraint, layers=None).results[0]
    assert result.status == "unmeasured" and result.detail == "no layer data for the new version"
    assert _check(constraint).results[0].status == "kept"
    assert _check(constraint, layers=()).results[0].status == "broken"


def test_caption_text_kind_and_element_kind_or_alignment_break_keeps() -> None:
    caption = _constraint("caption")
    changed = (LayerWindow("caption", "caption", 2.1, 3.2, "Different"),)
    assert _check(caption, layers=changed).results[0].status == "broken"
    changed = (LayerWindow("caption", "title", 2.1, 3.2, "Start End"),)
    assert _check(caption, layers=changed).results[0].status == "broken"
    element = _constraint("element")
    changed = (LayerWindow("card", "zoom", 2.1, 2.7, None),)
    assert _check(element, layers=changed).results[0].status == "broken"
    moved = (LayerWindow("card", "graphic", 1.2, 1.7, None),)
    report = _check(element, layers=moved)
    assert report.results[0].status == "broken" and "overlaps" in report.results[0].detail
    no_words = _constraint("element", start=63, end=None)
    assert no_words["words"] == []
    assert _check(no_words, layers=moved).ok


def test_element_keep_breaks_when_kept_words_move_to_a_renamed_segment() -> None:
    constraint = _constraint("element")
    moved = CutWords("output", {}, (
        CutWord("t1:0", "renamed", "Start", 0.3, 0.6),
        CutWord("t1:2", "renamed", "End", 0.9, 1.2),
    ))
    layers = (LayerWindow("card", "graphic", 0.2, 0.7, None),)
    result = _check(constraint, words=moved, layers=layers).results[0]
    assert result.status == "broken"
    assert "overlaps" in result.detail


def test_video_keep_and_all_broken_results_in_error() -> None:
    video = _constraint("video")
    assert video["words"] == video["gaps"] == video["layers"] == []
    result = _check(video).results[0]
    assert result.status == "unmeasured" and result.detail == "needs a person to check"
    assert _check(video).ok
    story, speech = _constraint(), _constraint("speech")
    speech["comment"] = {"version": "v1", "id": "c7"}
    speech["text"] = "x" * 100
    with pytest.raises(KeepBrokenError) as error:
        require_keeps([story, speech, video], CutWords("output", {}, ()), None, fps=Fraction(30))
    assert "Keep comment c3 on v2" in str(error.value)
    assert "Keep comment c7 on v1" in str(error.value)
    assert '"' + "x" * 60 + '"' in str(error.value)
    assert len(error.value.report.results) == 3


def test_check_version_current_cut_and_saved_version_defaults(feedback_project: Path) -> None:
    root = feedback_project
    first = save_version(root, layers=_layers(root))
    build_batch(root, _export(root, [_comment(kind="keep", start_frame=66, end_frame=93)]))
    assert check_version(root).ok
    assert check_version(root, "b1", first.id).ok
    _cut(root, words=CutWords("output", {}, WORDS.words[:3]), label="dropped kept word")
    second = save_version(root)
    assert not check_version(root).ok and not check_version(root, version_id=second.id).ok
    assert check_version(root, version_id=first.id).ok


def test_check_without_batch_names_the_build_command(feedback_project: Path) -> None:
    with pytest.raises(FeedbackError, match="No feedback batch") as error:
        check_version(feedback_project)
    assert "feedback build" in error.value.suggested_action


@pytest.mark.parametrize("target", ["story", "speech"])
@pytest.mark.parametrize("saved", [False, True])
def test_check_version_skips_envelope_without_delivery_evidence(
    feedback_project: Path, monkeypatch: pytest.MonkeyPatch, target: str, saved: bool,
) -> None:
    root = feedback_project
    version_id = save_version(root).id if saved else None
    build_batch(root, _export(root, [_comment("c1", kind="keep", target=target)]))

    def unexpected_envelope(path):
        pytest.fail("an envelope was measured or read without delivery evidence to compare")

    monkeypatch.setattr(keeps_module, "media_envelope", unexpected_envelope)
    monkeypatch.setattr(keeps_module, "version_envelope", unexpected_envelope)
    result = check_version(root, version_id=version_id).results[0]
    assert result.status == ("unmeasured" if target == "speech" else "kept")


def test_check_version_measures_once_for_delivery_constraints(
    feedback_project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = feedback_project
    _cut(root, amplitude=0.1)
    comments = [_comment("c1", kind="keep", target="speech", start_frame=9),
                _comment("c2", kind="keep", target="speech")]
    build_batch(root, _export(root, comments))
    decode = keeps_module.media_envelope
    calls = []

    def measure(path):
        calls.append(path)
        return decode(path)

    monkeypatch.setattr(keeps_module, "media_envelope", measure)
    assert all(result.status == "kept" for result in check_version(root).results)
    assert calls == [root / "cut" / "cut.mp4"]


def test_regression_story_keep_requires_exact_occurrence() -> None:
    comment = _comment("c21", kind="keep", target="story", start_frame=9)
    constraint = keep_constraint(comment, _locate(comment), version="v1")
    assert _check(constraint).results[0].status == "kept"
    # The same source word remains, but its kept story block has been dropped.
    words = CutWords("output", {}, WORDS.words[2:])
    result = _check(constraint, words=words).results[0]
    assert result.status == "broken"
    assert result.detail == ('word t1:0 "Start" in s1 is missing from the new cut '
                             '(the same word is in s2)')


@pytest.mark.parametrize("target", [None, "speech"])
def test_regression_pause_keep_requires_neighbour_occurrences(target: str | None) -> None:
    constraint = _constraint(target, start=18, end=None)
    changed = CutWords("output", {}, (
        CutWord("t1:0", "s2", "Start", 0.2, 0.5),
        CutWord("t1:1", "s2", "really", 0.8, 1.1),
    ))
    assert _check(constraint, words=changed).results[0].status == "broken"


def test_regression_speech_keep_checks_delivery(feedback_project: Path) -> None:
    root = feedback_project
    _cut(root, amplitude=0.1)
    first = save_version(root)
    build_batch(root, _export(root, [_comment("c21", kind="keep", target="speech",
                                            start_frame=66, end_frame=93)]))
    # Words, gaps and source identities are identical; the render's audio is 4 dB louder.
    _cut(root, amplitude=0.1 * 10 ** (4 / 20), label="louder render")
    second = save_version(root)
    result = check_version(root, "b1", second.id).results[0]
    assert result.status == "broken"
    assert "louder" in result.detail
    assert check_version(root, "b1", first.id).results[0].status == "kept"
    assert check_version(root, "b1").results[0].status == "broken"


def test_regression_speech_keep_without_envelope_is_unmeasured() -> None:
    result = _check(_constraint("speech")).results[0]
    assert result.status == "unmeasured"
    assert result.detail.startswith("words and pauses kept; delivery not measured:")


def test_speech_source_changes_and_missing_target_evidence(feedback_project: Path) -> None:
    root = feedback_project
    _cut(root, amplitude=0.1)
    save_version(root)
    batch = build_batch(root, _export(root, [_comment("c21", kind="keep", target="speech")]))
    delivery = batch["constraints"][0]["delivery"]
    assert delivery["start_s"] == 2.2 and delivery["end_s"] == 2.5
    assert delivery["sources"] == {"t1": {"take": "b" * 64, "words": "a" * 64}}
    assert len(delivery["envelope"]) == 30
    from yap_studio.feedback.audio import ENVELOPE_SCHEMA

    envelope = {"schema": ENVELOPE_SCHEMA, "rate": 100, "db": [-23.0] * 400}
    sources = {"take:t1": "b" * 64, "words:t1": "a" * 64}
    for key in sources:
        changed = {**sources, key: "c" * 64}
        result = check_keeps(batch["constraints"], WORDS, None, fps=Fraction(30),
                             envelope=envelope, sources=changed).results[0]
        assert result.status == "broken" and result.detail == "take t1 changed"
    for missing in ("envelope", "sources"):
        result = check_keeps(batch["constraints"], WORDS, None, fps=Fraction(30),
                             envelope=envelope if missing != "envelope" else None,
                             sources=sources if missing != "sources" else None).results[0]
        assert result.status == "unmeasured"
        assert result.detail.startswith("words and pauses kept; delivery not measured:")


def test_regression_caption_keep_without_reviewed_layers_is_unmeasured() -> None:
    comment = _comment("c21", kind="keep", target="caption")
    constraint = keep_constraint(comment, _locate(comment, layers=None), version="v1")
    changed = (LayerWindow("caption", "caption", 2.1, 3.2, "Changed caption"),)
    report = _check(constraint, layers=changed)
    assert report.results[0].status == "unmeasured"
    assert report.results[0].detail == "no layer data for the reviewed version"
    assert report.ok and report.unmeasured == report.results


@pytest.mark.parametrize("target", ["caption", "element"])
def test_layer_keep_without_protected_layer_is_unmeasured(target: str) -> None:
    comment = _comment("c21", kind="keep", target=target, start_frame=9)
    constraint = keep_constraint(comment, _locate(comment), version="v1")
    assert constraint["layers_known"] and constraint["layers"] == []
    result = _check(constraint).results[0]
    assert result.status == "unmeasured" and result.detail == f"no {target} in the kept range"


def test_carried_speech_keep_uses_its_own_version_delivery(feedback_project: Path) -> None:
    from yap_studio.review.export import comment_to_dict

    root = feedback_project
    _cut(root, amplitude=0.1)
    first = save_version(root)
    _cut(root, amplitude=0.1 * 10 ** (4 / 20), label="louder render")
    second = save_version(root)
    export = _export(root, [])
    export["series"] = {
        "id": "series", "version": 2, "versions": [
            {"number": 1, "sha256": first.media["sha256"], "version_id": "v1"},
            {"number": 2, "sha256": second.media["sha256"], "version_id": "v2"},
        ], "threads": [{**comment_to_dict(_comment("c21", kind="keep", target="speech")),
                        "version": 1, "reply": None, "state": "open"}],
    }
    constraint = build_batch(root, export)["constraints"][0]
    assert constraint["delivery"]["envelope"] == pytest.approx([-23.0] * 30, abs=0.3)
    assert check_version(root, version_id=second.id).results[0].status == "broken"


def test_current_cut_batch_records_speech_delivery(feedback_project: Path) -> None:
    root = feedback_project
    _cut(root, amplitude=0.1)
    batch = build_batch(root, _export(root, [_comment("c21", kind="keep", target="speech")]))
    assert batch["base"] == "cut"
    assert batch["constraints"][0]["delivery"]["envelope"] == pytest.approx([-23.0] * 30, abs=0.3)
    assert check_version(root).results[0].status == "kept"


@pytest.mark.parametrize("saved", [False, True], ids=["current_cut", "saved_v2"])
def test_regression_explicit_media_uses_its_own_delivery(
    feedback_project: Path, tmp_path: Path, saved: bool,
) -> None:
    root = feedback_project
    _voice(root)
    _cut(root, amplitude=0.1)
    quiet_media = tmp_path / "quieter.mp4"
    _render_media(quiet_media, 120, amplitude=0.1 * 10 ** (-8 / 20))
    first = save_version(root, media=quiet_media)
    keep = _comment("c31", kind="keep", target="speech", start_frame=66, end_frame=93)
    build_batch(root, _export(root, [keep], version=first))
    assert check_version(root, version_id=first.id).results[0].status == "kept"
    version_id = save_version(root).id if saved else None
    result = check_version(root, version_id=version_id).results[0]
    assert result.status == "broken"
    assert "louder" in result.detail


def test_regression_speech_keep_measures_full_new_word_span() -> None:
    from yap_studio.feedback.audio import ENVELOPE_SCHEMA

    envelope = {"schema": ENVELOPE_SCHEMA, "rate": 100, "db": [-23.0] * 50}
    sources = {"take:t1": "b" * 64, "words:t1": "a" * 64}
    comment = _comment("c31", kind="keep", target="speech", start_frame=9)
    constraint = keep_constraint(comment, _locate(comment), version="v1",
                                 envelope=envelope, inputs=sources)
    words = CutWords("output", {}, (CutWord("t1:0", "s1", "Start", 0.2, 0.7),))
    new = {**envelope, "db": [-23.0] * 50 + [-10.0] * 20}
    result = check_keeps([constraint], words, None, fps=Fraction(30), envelope=new,
                         sources=sources).results[0]
    assert result.status == "broken"
    assert result.detail == "the kept speech is 0.20 s longer in the new version"
