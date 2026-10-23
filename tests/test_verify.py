"""End-to-end verification orchestration."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import synth
from test_feedback_versions import project as project
from test_taste_render import _body

from yap_studio import audio, ffmpeg
from yap_studio.checks.joins import JoinsResult
from yap_studio.checks.model import CHECK_IDS, TransitionWindow
from yap_studio.checks.sync import SegmentSync, SyncResult
from yap_studio.edl import load_edit_list
from yap_studio.errors import InputError, OutputPathError, ValidationError
from yap_studio.feedback.versions import save_version
from yap_studio.plan import Plan, snap_frame
from yap_studio.report import TOP_LEVEL_KEYS, _check_status, build_report, exit_code
from yap_studio.taste import (
    Taste,
    add_proposals,
    approve,
    read_taste_record,
    render_style,
    write_taste_record,
)
from yap_studio.transcript import ScriptedBackend, Transcript, Word
from yap_studio.verify import _edit_coverage, _verify, verify


def test_click_coverage_reports_measured_signal() -> None:
    coverage = {"join.click": {"ran": True, "reason": None, "click_source": "voice_stem"}}
    assert _check_status("join.click", [], coverage)["click_source"] == "voice_stem"


@pytest.mark.parametrize("record_state", ["measured", "missing", "corrupt"])
def test_verify_caption_contrast_record(tmp_path: Path, record_state: str) -> None:
    from test_caption_contrast import contrast_document

    from yap_studio.checks.caption_contrast import write_contrast_record

    media = synth.calib_source(tmp_path / "final.mov", seconds=1, size="96x72")
    if record_state == "measured":
        write_contrast_record(media, contrast_document())
    elif record_state == "corrupt":
        media.with_suffix(".contrast.json").write_text("{")
    result = verify(str(media), asr="none", loudness="note")
    coverage = result["checks"]["captions.contrast"]
    findings = [item for item in result["findings"] if item["check"] == "captions.contrast"]
    if record_state == "measured":
        assert coverage["ran"] and coverage["status"] == "warn"
        assert len(findings) == 1 and findings[0]["evidence"]["ratio"] == 1.2
        assert findings[0]["severity"] == "warn"
        assert str(media.with_suffix(".contrast.json")) in json.dumps(result["receipt"])
    elif record_state == "missing":
        assert not coverage["ran"]
        assert coverage["reason"] == "no contrast record next to this render"
        assert findings == []
    else:
        assert not coverage["ran"] and "corrupt contrast record" in coverage["reason"]
        assert len(findings) == 1 and findings[0]["severity"] == "warn"


@pytest.mark.parametrize(
    ("cut_sources", "expected"),
    [((None, None), "voice_stem"), (("mix", "voice_stem"), "mixed")],
)
def test_click_coverage_falls_back_when_measured_cuts_lack_sources(
    cut_sources: tuple[str | None, str | None], expected: str,
) -> None:
    cuts = [SimpleNamespace(
        index=index, click_db=-20.0, click_source=source, gap_ms=0.0,
        words=SimpleNamespace(status="pass", reason=None),
        frames=SimpleNamespace(status="pass", reason=None), flash=None, relisten=None,
    ) for index, source in enumerate(cut_sources, start=1)]
    joins = JoinsResult(cuts, [])
    coverage = _edit_coverage(
        SimpleNamespace(has_audio=True), SyncResult([], None, []), joins, None,
        click_source="voice_stem",
    )["join.click"]
    assert coverage["click_source"] == expected
    assert [row["click_source"] for row in coverage["evidence"]] == [
        source or "voice_stem" for source in cut_sources
    ]


def _edit_list(path: Path, source: Path, spans) -> Path:
    path.write_text(
        json.dumps(
            {
                "source": str(source),
                "segments": [{"start": start, "end": end} for start, end in spans],
            }
        )
    )
    return path


def _ranges(path: Path, source: Path, spans, **extra) -> Path:
    path.write_text(
        json.dumps(
            {
                "sources": {"main": str(source)},
                "ranges": [
                    {"source": "main", "start": start, "end": end} for start, end in spans
                ],
                **extra,
            }
        )
    )
    return path


def _declared_script(path: Path) -> Transcript:
    # The join sits after the 1 s of WAV padding and the window's lead-in.
    at = 1.0 + float(path.stem.rsplit("-", 1)[1])
    return Transcript(
        [Word("alpha", at - 0.6, at - 0.4), Word("beta", at - 0.3, at - 0.1),
         Word("gamma", at + 0.05, at + 0.25)]
    )


def _script(path: Path) -> Transcript:
    if path.stem.endswith("left"):
        return Transcript([Word("bright", 1.2, 1.4), Word("planet", 1.5, 1.9)])
    if path.stem.endswith("right"):
        return Transcript([Word("quiet", 2.1, 2.4), Word("garden", 2.5, 2.8)])
    return Transcript(
        [
            Word("bright", 1.2, 1.4),
            Word("planet", 1.5, 1.9),
            Word("quiet", 3.1, 3.4),
            Word("garden", 3.5, 3.8),
        ]
    )


@pytest.fixture(scope="module")
def transition_media(tmp_path_factory: pytest.TempPathFactory):
    """Blend a delayed picture over one join while copying the clean sound."""
    directory = tmp_path_factory.mktemp("verify-transition")
    raw = synth.calib_source(directory / "raw.mov", seconds=8, size="160x90")
    source = directory / "source.mov"
    synth._run(["ffmpeg", "-v", "error", "-i", str(raw), "-vf",
                "noise=alls=35:allf=t+u:all_seed=7", "-c:v", "libx264", "-crf", "0",
                "-preset", "ultrafast", "-c:a", "copy", str(source)])
    edl = _edit_list(directory / "edit.json", source, [(0, 3), (4, 7)])
    clean = synth.render_clean(source, [(0, 3), (4, 7)], directory / "clean.mp4")
    renders = {}
    for shift in (0, -2, 2):
        render = directory / f"blend-{shift}.mp4"
        # blend's N is one-based; the declared output window is [84, 96).
        filters = ("[0:v]split=2[base][other];[other]setpts=PTS+2/(30*TB)[lag];"
                   "[base][lag]blend=all_expr='"
                   f"if(between(N,{85 + shift},{96 + shift}),0.1*A+0.9*B,A)'"
                   ":shortest=1[picture]")
        synth._run(["ffmpeg", "-v", "error", "-i", str(clean), "-filter_complex", filters,
                    "-map", "[picture]", "-map", "0:a:0", "-c:v", "libx264", "-crf", "0",
                    "-preset", "ultrafast", "-c:a", "copy", str(render)])
        renders[shift] = render
    return SimpleNamespace(edl=edl, clean=clean, renders=renders,
                           window=TransitionWindow(90, 84, 96, "join-blend"))


def test_verify_transition_covers_picture_and_keeps_audio(transition_media) -> None:
    media = transition_media
    baseline = verify(str(media.clean), edl=str(media.edl), asr="none", loudness="note",
                      detail="full")
    result = _verify(str(media.renders[0]), edl=str(media.edl), asr="none", loudness="note",
                     detail="full", transitions=[media.window])
    assert result["status"] == "pass", result["findings"]
    assert result["checks"]["picture.transition"]["status"] == "pass"
    for check in ("join.frames", "picture.flash"):
        coverage = result["checks"][check]
        assert coverage["ran"] is False and coverage["status"] == "skipped"
        assert coverage["reason"] == "covered by transition"
        assert coverage["evidence"] == [{"cut": 1, "status": "covered by transition",
                                         "id": "join-blend", "join_frame": 90,
                                         "start_frame": 84, "end_frame": 96}]
    assert result["cuts"][0]["frames"] == "covered by transition"
    for check in ("join.click", "join.gap", "join.words", "join.listen", "join.relisten",
                  "join.second_model"):
        assert result["checks"][check] == baseline["checks"][check]
    for field in ("click_db", "gap_ms", "words", "relisten", "j_end_s", "j_start_s"):
        assert result["cuts"][0]["detail"][field] == baseline["cuts"][0]["detail"][field]
    evidence = result["checks"]["picture.transition"]["evidence"][0]
    assert evidence["actual_frames"] == evidence["expected_frames"] == 180
    assert all(row["matched"] for row in evidence["boundaries"].values())
    assert "right frames" not in result["summary"]["text"]


@pytest.mark.parametrize("shift,side", [(-2, "before"), (2, "after")])
def test_verify_shifted_transition_fails_with_id(transition_media, shift, side) -> None:
    media = transition_media
    result = _verify(str(media.renders[shift]), edl=str(media.edl), asr="none", loudness="note",
                     transitions=[vars(media.window)])
    assert result["status"] == "fail"
    assert result["checks"]["picture.transition"]["status"] == "fail"
    finding = next(row for row in result["findings"] if row["check"] == "picture.transition")
    assert "join-blend" in finding["message"] and side in finding["message"]
    assert finding["evidence"]["boundaries"][side]["matched"] is False


def test_verify_transition_without_declaration_keeps_frame_failures(transition_media) -> None:
    media = transition_media
    result = verify(str(media.renders[0]), edl=str(media.edl), asr="none", loudness="note")
    assert result["status"] == "fail"
    assert result["checks"]["join.frames"]["status"] == "fail"
    assert result["checks"]["picture.transition"] == {
        "ran": False, "status": "skipped", "reason": "no declared transitions",
    }


def test_transition_windows_require_an_edit_list() -> None:
    with pytest.raises(ValidationError, match="transitions: requires an edit list"):
        _verify("unused.mp4", transitions=[TransitionWindow(90, 84, 96, "blend")])


@pytest.mark.parametrize("measured_status,expected", [("pass", "skipped"), ("fail", "fail")])
def test_report_transition_coverage_names_skipped_windows(measured_status, expected) -> None:
    joins = JoinsResult([], [], [
        {"id": "checked-window", "join_frame": 90, "start_frame": 84, "end_frame": 96,
         "status": measured_status, "reason": None},
        {"id": "skipped-window", "join_frame": 180, "start_frame": 174, "end_frame": 186,
         "status": "skipped", "reason": "source not found"},
    ])
    coverage = _edit_coverage(SimpleNamespace(has_audio=True), SyncResult([], None, []),
                              joins, None)
    findings = [{"check": "picture.transition", "severity": "fail"}] \
        if measured_status == "fail" else []
    state = _check_status("picture.transition", findings, coverage)
    assert state["status"] == expected
    assert "partial" in state["note"] and "skipped-window" in state["note"]
    if expected == "skipped":
        assert state["ran"] is False
        assert "skipped-window" in state["reason"] and "source not found" in state["reason"]
    assert [row["status"] for row in state["evidence"]] == [measured_status, "skipped"]


def test_public_verify_forwards_options_without_transition_windows(monkeypatch) -> None:
    module = importlib.import_module("yap_studio.verify")
    implementation = Mock(return_value={"status": "pass"})
    monkeypatch.setattr(module, "_verify", implementation)
    result = verify("render.mp4", edl="edit.json", source=["main=source.mov"], asr="none",
                    asr_model="local", loudness_target=-16, strict=True, report="report.json",
                    artifacts="artifacts", detail="full", platform="none", loudness="note")
    assert result == {"status": "pass"}
    implementation.assert_called_once_with(
        "render.mp4", edl="edit.json", source=["main=source.mov"], asr="none", asr_model="local",
        loudness_target=-16, strict=True, report="report.json", artifacts="artifacts",
        detail="full", platform="none", loudness="note",
    )


@pytest.mark.parametrize(
    "rows,av_offset,drift",
    [
        (
            [SegmentSync(1, "main", av_offset_ms=100, confidence="high")],
            {"ran": True, "reason": None},
            {"ran": False, "reason": "needs two confident sync measurements"},
        ),
        (
            [
                SegmentSync(1, "main", av_offset_ms=100, confidence="low"),
                SegmentSync(2, "main", av_offset_ms=100, confidence="low"),
            ],
            {"ran": False, "reason": "sync measurements were not confident enough"},
            {"ran": False, "reason": "sync measurements were not confident enough"},
        ),
        (
            [SegmentSync(1, "main", confidence="low")],
            {"ran": False, "reason": "sync could not be measured"},
            {"ran": False, "reason": "sync could not be measured"},
        ),
        (
            [
                SegmentSync(1, "main", av_offset_ms=100, confidence="high"),
                SegmentSync(2, "main", av_offset_ms=100, confidence="high"),
            ],
            {"ran": True, "reason": None},
            {"ran": True, "reason": None},
        ),
    ],
)
def test_edit_coverage_uses_only_confident_sync_rows(rows, av_offset, drift) -> None:
    coverage = _edit_coverage(
        SimpleNamespace(has_audio=True),
        SyncResult(rows, None, []),
        SimpleNamespace(cuts=[]),
        None,
    )
    assert coverage["sync.av_offset"] == av_offset
    assert coverage["sync.drift"] == drift


def test_low_confidence_sync_is_skipped_in_report() -> None:
    sync = SyncResult(
        [
            SegmentSync(1, "main", av_offset_ms=100, confidence="low"),
            SegmentSync(2, "main", av_offset_ms=100, confidence="low"),
        ],
        None,
        [],
    )
    joins = SimpleNamespace(cuts=[])
    coverage = _edit_coverage(SimpleNamespace(has_audio=True), sync, joins, None)
    report = build_report(
        media_info=SimpleNamespace(
            path="render.mp4",
            width=320,
            height=180,
            orientation="landscape",
            fps=None,
            duration_s=1.0,
            video_frames=30,
            sample_rate=48_000,
            has_audio=True,
        ),
        edit_list=SimpleNamespace(
            path=Path("edit.json"), format="yap", segments=[], warnings=[]
        ),
        sources=[],
        findings=[],
        structure=SimpleNamespace(
            integrated_lufs=None,
            lra_lu=None,
            true_peak_dbtp=None,
            black=[],
            freeze=[],
        ),
        checks=coverage,
        receipt={},
        target_lufs=-14.0,
        joins=joins,
        sync=sync,
    )
    assert report["checks"]["sync.av_offset"] == {
        "ran": False,
        "status": "skipped",
        "reason": "sync measurements were not confident enough",
    }


@pytest.mark.parametrize("engine", [False, True])
def test_single_span_intent_coverage(tmp_path: Path, engine: bool) -> None:
    """Outer word checks stay visible even when the edit has no joins."""
    edl = tmp_path / "intent.json"
    edl.write_text(json.dumps({"source": "take.mov", "segments": [
        {"start": 0, "end": 3, "first_word": "Bright", "last_word": "hills"},
    ]}), encoding="utf-8")
    plan = Plan.build(load_edit_list(edl), Fraction(30), 48000)
    coverage = _edit_coverage(SimpleNamespace(has_audio=True), SyncResult([], None, []),
                              JoinsResult([], []),
                              ScriptedBackend(lambda _: Transcript([])) if engine else None, plan)
    words = coverage["join.words"]
    assert words["ran"] is engine
    if not engine:
        assert "planned boundary words not checked" in words["reason"]
        assert "no speech engine" in words["reason"]


def test_verify_intent_without_engine_names_skipped_check(synth_media, tmp_path: Path) -> None:
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans)
    data = json.loads(edl.read_bytes())
    data["segments"][1]["first_word"] = "this"
    edl.write_text(json.dumps(data), encoding="utf-8")
    result = verify(str(synth_media.path("clean")), edl=str(edl), asr="none")
    assert set(result) == set(TOP_LEVEL_KEYS)
    assert set(result["checks"]) == set(CHECK_IDS)
    assert result["checks"]["join.words"]["status"] == "pass"
    assert "planned boundary words not checked" in result["checks"]["join.words"]["note"]


@pytest.mark.parametrize("case", ["exact", "connected", "clipped", "missing"])
def test_single_span_outer_findings_reach_report(
    synth_media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    """Cut coverage cannot hide endpoint failures or listen notes on a continuous render."""
    source = synth_media.path("calib")
    edl = tmp_path / "intent.json"
    edl.write_text(json.dumps({"source": str(source), "segments": [
        {"start": 0.1, "end": 29.9, "first_word": "Bright", "last_word": "hills"},
    ]}), encoding="utf-8")
    module = importlib.import_module("yap_studio.verify")
    def script(path: Path) -> Transcript:
        if case == "missing" or path.name.startswith("excluded-"):
            return Transcript([])
        if case == "connected":
            return Transcript([Word("Bright", 2.02, 2.24)] if "first" in path.name
                              else [Word("hills", 2.3, 2.48)])
        word = Word("Bright", 1.94, 2.24) if "first" in path.name else Word("hills", 2.3, 2.55)
        return Transcript([word])

    monkeypatch.setattr(module, "resolve_backend", lambda *_: ScriptedBackend(script))
    decode = audio.decode

    def outer_audio(path: Path, *, sr: int | None = None) -> tuple[int, np.ndarray]:
        rate, sound = decode(path, sr=sr)
        sound = sound.copy()
        sound[:] = 10 ** (-65 / 20)
        start = 0.04 if case == "clipped" else 0.12
        end = 29.98 if case == "clipped" else 29.88
        sound[round(start * rate):round(0.34 * rate)] = 0.1
        sound[round(29.7 * rate):round(end * rate)] = 0.1
        if case == "connected":
            # Neighbouring voice is separated by a dip rather than absolute silence.
            sound[:round(0.08 * rate)] = 0.1
            sound[round(0.08 * rate):round(start * rate)] = 10 ** (-30 / 20)
            sound[round(end * rate):round(29.92 * rate)] = 10 ** (-30 / 20)
            sound[round(29.92 * rate):] = 0.1
        return rate, sound

    monkeypatch.setattr(audio, "decode", outer_audio)
    result = verify(str(source), edl=str(edl), asr="auto")
    assert set(result) == set(TOP_LEVEL_KEYS)
    assert set(result["checks"]) == set(CHECK_IDS)
    assert result["checks"]["join.words"] == {
        "ran": True, "status": "fail" if case == "clipped" else "pass", "reason": None,
    }
    check = "join.listen" if case == "missing" else "join.words"
    findings = [row for row in result["findings"] if row["check"] == check]
    assert len(findings) == (0 if case in ("exact", "connected") else 2)
    if findings:
        assert '"Bright"' in findings[0]["message"] and '"hills"' in findings[1]["message"]
    assert all(row["cut"] is None and row["severity"] == ("fail" if case == "clipped" else "info")
               for row in findings)
    if case == "missing":
        assert result["checks"]["join.listen"] == {"ran": True, "status": "pass", "reason": None}


def test_verify_clean_report_runs_words_without_engine(synth_media, tmp_path: Path) -> None:
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans)
    result = verify(str(synth_media.path("clean")), edl=str(edl), asr="none")
    assert result["status"] != "fail"
    assert result["checks"]["join.words"] == {
        "ran": True,
        "status": "pass",
        "reason": None,
        "note": "Words not named: no speech engine installed; pip install 'yap-studio[asr]'",
    }
    assert result["segments"] is None
    # Holds are only checked where the edit list declares them, and not listed otherwise.
    assert result["checks"]["plan.holds"] == {
        "ran": False,
        "status": "skipped",
        "reason": "no holds in the edit list",
    }
    assert "Declared holds" not in result["summary"]["text"]
    assert result["checks"]["edit_list.evidence"] == {
        "ran": False,
        "status": "skipped",
        "reason": "no evidence in the edit list",
    }
    assert "Attested evidence" not in result["summary"]["text"]


@pytest.fixture(scope="module")
def loudness_render(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a continuous synthetic render near -20 LUFS with a high true peak."""
    return synth.calib_source(
        tmp_path_factory.mktemp("verify-loudness") / "render.mov", seconds=6
    )


@pytest.mark.parametrize("with_edl", [False, True])
@pytest.mark.parametrize("record_kind", [
    "absent", "present", "corrupt", "unmeasured", "history_mismatch",
])
def test_verify_taste_coverage_and_record_warning(
    loudness_render: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    with_edl: bool, record_kind: str,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    media = tmp_path / "render.mov"
    shutil.copyfile(loudness_render, media)
    edl = _ranges(tmp_path / "edit.json", media, [(0, 6)]) if with_edl else None
    if record_kind != "absent":
        identity = add_proposals(_body())[0].id
        approve(identity)
        target = write_taste_record(media, render_style())
        if record_kind == "corrupt":
            target.write_text("{", encoding="utf-8")
    module = importlib.import_module("yap_studio.taste.check")
    check = module.check_taste
    observation = (
        {"value": None, "reason": "No stable visible evidence was found."}
        if record_kind == "unmeasured" else {"value": "#FFFFFF", "frames": []}
    )
    measure = Mock(return_value={"captions.text_color": observation})
    if record_kind == "history_mismatch":
        monkeypatch.setattr(module, "load_taste", Taste)
    monkeypatch.setattr(module, "check_taste", lambda media, record: check(
        media, record, measure=measure,
    ))
    result = verify(str(media), edl=str(edl) if edl else None, asr="none", loudness="note")
    assert set(result["checks"]) == set(CHECK_IDS)
    findings = [row for row in result["findings"] if row["check"] == "taste.rules"]
    if record_kind == "present":
        measure.assert_called_once_with(media, fields=("captions.text_color",))
        assert result["checks"]["taste.rules"] == {
            "ran": True, "status": "fail", "reason": None,
        }
        assert findings[0]["severity"] == "fail"
        assert findings[0]["evidence"]["rule_id"] == identity
        assert result["status"] == "fail"
    elif record_kind == "corrupt":
        measure.assert_not_called()
        assert result["checks"]["taste.rules"] == {
            "ran": True, "status": "warn", "reason": None,
        }
        assert findings[0]["severity"] == "warn"
        assert findings[0]["message"].startswith("corrupt taste record:")
        assert result["status"] == "warn"
    else:
        if record_kind == "unmeasured":
            measure.assert_called_once_with(media, fields=("captions.text_color",))
        else:
            measure.assert_not_called()
        coverage = result["checks"]["taste.rules"]
        assert coverage["ran"] is False and coverage["status"] == "skipped"
        assert findings == []
        if record_kind == "absent":
            assert coverage["reason"] == "no taste record next to this render"
        else:
            if record_kind == "unmeasured":
                assert coverage["reason"] == (
                    "no applied rule could be measured from the picture: "
                    "captions.text_color (No stable visible evidence was found.)"
                )
            else:
                assert coverage["reason"] == (
                    "this computer's taste history does not match the pinned revision 1"
                )
            assert f"Approved taste rules: {coverage['reason']}" in result["summary"]["text"]


def test_verify_saved_version_runs_taste_rules(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    identity = add_proposals(_body())[0].id
    approve(identity)
    media = project / "render.mp4"
    shutil.copyfile(project / "cut" / "cut.mp4", media)
    write_taste_record(media, render_style())
    record = read_taste_record(media)
    manifest = save_version(project, media=media)
    module = importlib.import_module("yap_studio.taste.check")
    check = module.check_taste
    measure = Mock(return_value={"captions.text_color": {"value": "#FFFFFF", "frames": []}})
    monkeypatch.setattr(module, "check_taste", lambda media, record: check(
        media, record, measure=measure,
    ))
    result = verify(str(manifest.media_path), asr="none")
    measure.assert_called_once_with(manifest.media_path, fields=("captions.text_color",))
    assert read_taste_record(manifest.media_path) == record
    assert result["checks"]["taste.rules"] == {"ran": True, "status": "fail", "reason": None}
    findings = [row for row in result["findings"] if row["check"] == "taste.rules"]
    assert findings[0]["evidence"]["rule_id"] == identity


@pytest.mark.parametrize("with_edl", [False, True])
def test_verify_loudness_check_and_note(
    loudness_render: Path, tmp_path: Path, with_edl: bool,
) -> None:
    edl = _ranges(tmp_path / "edit.json", loudness_render, [(0, 6)]) if with_edl else None
    arguments = {
        "edl": str(edl) if edl is not None else None,
        "asr": "none",
        "detail": "full" if with_edl else "summary",
    }
    checked = verify(str(loudness_render), loudness="check", **arguments)
    noted = verify(str(loudness_render), loudness="note", **arguments)
    assert checked["structure"]["integrated_lufs"] == pytest.approx(-20, abs=1)
    checks = {"loudness.integrated", "loudness.true_peak"}
    warnings = {row["check"]: row for row in checked["findings"] if row["check"] in checks}
    notes = {row["check"]: row for row in noted["findings"] if row["check"] in checks}
    assert set(warnings) == set(notes) == checks
    for check in checks:
        assert warnings[check]["severity"] == "warn"
        assert notes[check]["severity"] == "info"
        assert notes[check]["suggested_action"] is None
        assert notes[check]["evidence"] == warnings[check]["evidence"]
        assert "Loudness is set when the video is assembled." in notes[check]["message"]
    integrated = checked["structure"]["integrated_lufs"]
    peak = checked["structure"]["true_peak_dbtp"]
    assert warnings["loudness.integrated"]["message"] == (
        f"At 0:00.0, overall loudness is {integrated:.1f} LUFS. "
        "Normalize the whole video to -14 LUFS."
    )
    assert warnings["loudness.true_peak"]["message"] == (
        f"At 0:00.0, sound peaks at {peak:.1f} dBTP. Limit the true peak to -1 dBTP."
    )
    assert f"{integrated:.1f} LUFS" in notes["loudness.integrated"]["message"]
    assert f"{peak:.1f} dBTP" in notes["loudness.true_peak"]["message"]
    assert noted["status"] == "pass"
    assert noted["summary"]["warn"] == 0
    assert exit_code(noted, strict=True) == 0
    assert list(noted) == list(checked) == TOP_LEVEL_KEYS
    assert set(noted["checks"]) == set(checked["checks"]) == set(CHECK_IDS)
    assert len(noted["checks"]) == len(CHECK_IDS)
    if with_edl:
        nested = [row for row in noted["structure"]["findings"] if row["check"] in checks]
        assert nested == list(notes.values())


def test_loudness_parts_warns_in_both_modes(loudness_render: Path, tmp_path: Path) -> None:
    render = tmp_path / "different-levels.mov"
    ffmpeg.run(
        [
            "ffmpeg", "-nostdin", "-i", str(loudness_render),
            "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
            "-af", "volume=0.8,volume=0.25:enable='gte(t,3)'", "-c:a", "pcm_s16le",
            str(render),
        ],
        timeout_s=60,
    )
    edl = tmp_path / "edit.json"
    edl.write_text(json.dumps({
        "sources": {"first": str(render), "second": str(render)},
        "ranges": [
            {"source": "first", "start": 0, "end": 3},
            {"source": "second", "start": 3, "end": 6},
        ],
    }))
    checked = verify(str(render), edl=str(edl), asr="none", loudness="check")
    noted = verify(str(render), edl=str(edl), asr="none", loudness="note")
    warnings = [row for row in checked["findings"] if row["check"] == "loudness.parts"]
    assert warnings
    assert all(row["severity"] == "warn" for row in warnings)
    assert [row for row in noted["findings"] if row["check"] == "loudness.parts"] == warnings
    assert noted["checks"]["loudness.parts"] == checked["checks"]["loudness.parts"]
    assert noted["status"] != "pass"
    assert not any(row["check"] == "loudness.true_peak" for row in checked["findings"])
    assert any(row["check"] == "loudness.true_peak" for row in noted["findings"])


def test_invalid_loudness_mode_is_actionable() -> None:
    with pytest.raises(ValidationError, match="loudness: must be check or note") as caught:
        verify("render.mp4", loudness="invalid")
    assert caught.value.suggested_action


def test_verify_one_segment_skips_join_checks(synth_media, tmp_path: Path) -> None:
    edl = _edit_list(
        tmp_path / "one-segment.json",
        synth_media.path("calib"),
        [synth_media.spans[0]],
    )

    result = verify(str(synth_media.path("clean")), edl=str(edl), asr="none")

    assert result["cuts"] == []
    for check in (
        "join.click",
        "join.gap",
        "join.words",
        "join.frames",
        "join.listen",
    ):
        assert result["checks"][check] == {
            "ran": False,
            "status": "skipped",
            "reason": "cut coverage: none",
        }
    assert result["edit_list"]["cut_coverage"] == "none"
    assert result["edit_list"]["attested"] == []
    text = result["summary"]["text"]
    assert "  Cut coverage: none. The edit list is one continuous range" in text
    assert "  Cut checks (clicks, gaps, words, frames, listening): cut coverage: none" in text
    assert "no cuts in the edit list" not in text


@pytest.mark.parametrize("use_stem", [False, True])
def test_declared_joins_are_checked_under_join_words(
    synth_media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_stem: bool,
) -> None:
    edl = _ranges(
        tmp_path / "declared.json",
        synth_media.path("calib"),
        [synth_media.spans[0]],
        joins=[{"out_frame": 36, "before": "alpha beta", "after": "gamma"}],
    )
    module = importlib.import_module("yap_studio.verify")
    monkeypatch.setattr(module, "resolve_backend", lambda *_: ScriptedBackend(_declared_script))
    render = synth_media.path("clean")
    stem = synth.float_audio(tmp_path / "no-effects.wav", audio.decode(render)[1])
    seen = []
    declared_check = module.check_declared_joins

    def record(path, *args, **kwargs):
        seen.append(path)
        return declared_check(path, *args, **kwargs)

    monkeypatch.setattr(module, "check_declared_joins", record)
    result = _verify(str(render), edl=str(edl), asr="auto",
                     **({"words_audio": stem} if use_stem else {}))
    assert seen == [stem if use_stem else render]
    words = result["checks"]["join.words"]
    assert words["ran"] is True and words["status"] == "pass"
    assert words["note"] == "1 declared join checked by its words (source: declared)."
    declared = [f for f in result["findings"] if f["check"] == "join.words"]
    assert [f["severity"] for f in declared] == ["info"]
    assert declared[0]["evidence"]["source"] == "declared"
    assert declared[0]["time_s"] == pytest.approx(36 / 30)
    assert result["edit_list"]["cut_coverage"] == "declared"
    assert result["edit_list"]["declared_joins"] == 1
    for check in ("join.click", "join.gap", "join.frames", "join.listen"):
        assert result["checks"][check]["reason"] == "cut coverage: declared"
    text = result["summary"]["text"]
    assert (
        "  Cut coverage: declared. 1 join declared in the edit list was checked by its words."
        in text
    )
    assert "Cut checks (clicks, gaps, frames, listening): cut coverage: declared" in text
    assert "no cuts in the edit list" not in text

    monkeypatch.setattr(module, "resolve_backend", lambda *_: None)
    unchecked = verify(str(synth_media.path("clean")), edl=str(edl), asr="none")
    assert unchecked["checks"]["join.words"] == {
        "ran": False,
        "status": "skipped",
        "reason": "declared join not checked: no speech engine installed; "
        "pip install 'yap-studio[asr]'",
    }


@pytest.mark.parametrize("how", ["composite", "finished_render"])
def test_cuts_are_not_exposed_by_a_composite_edit_list(synth_media, tmp_path: Path, how) -> None:
    render = synth_media.path("clean")
    if how == "composite":
        edl = _ranges(
            tmp_path / "composite.json",
            synth_media.path("calib"),
            [synth_media.spans[0]],
            composite=True,
        )
    else:
        # One range copies a finished render of the same length: the render itself.
        edl = _ranges(tmp_path / "master.json", render, [(0, 19.2)])
    result = verify(str(render), edl=str(edl), asr="none")
    assert result["edit_list"]["cut_coverage"] == "not_exposed"
    assert result["edit_list"]["composite"] is (how == "composite")
    for check in ("join.click", "join.gap", "join.words", "join.frames", "join.listen"):
        assert result["checks"][check]["reason"] == "cut coverage: not exposed"
    text = result["summary"]["text"]
    assert "  Cut coverage: not exposed." in text
    assert "so its cuts cannot be checked from it" in text
    assert "no cuts in the edit list" not in text


def test_attested_cut_checks_are_hashed_and_never_pass(synth_media, tmp_path: Path) -> None:
    evidence = tmp_path / "cut-review.txt"
    evidence.write_text("all 36 cuts reviewed by ear\n")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    other = tmp_path / "other-review.txt"
    other.write_text("changed after it was attested\n")
    edl = _ranges(
        tmp_path / "attested.json",
        synth_media.path("calib"),
        [synth_media.spans[0]],
        composite=True,
        evidence=[
            {"path": evidence.name, "sha256": digest},
            {"path": other.name, "sha256": digest},
            {"path": "missing.txt", "sha256": digest},
        ],
    )
    result = verify(str(synth_media.path("clean")), edl=str(edl), asr="none")
    edit_list = result["edit_list"]
    assert edit_list["cut_coverage"] == "attested"
    assert [row["match"] for row in edit_list["attested"]] == [True, False, None]
    assert [row["found"] for row in edit_list["attested"]] == [True, True, False]
    assert all(row["label"] == "checked outside Yap Studio" for row in edit_list["attested"])
    assert edit_list["attested"][0]["actual_sha256"] == digest
    # Attested cuts stay unchecked here; nothing about them counts as a pass.
    for check in ("join.click", "join.gap", "join.words", "join.frames", "join.listen"):
        assert result["checks"][check] == {
            "ran": False,
            "status": "skipped",
            "reason": "cut coverage: attested",
        }
    text = result["summary"]["text"]
    assert "  Cut coverage: attested. Cuts were checked outside Yap Studio" in text
    assert "not counted as a pass" in text
    assert "cut-review.txt: hash matches (checked outside Yap Studio)" in text
    assert "other-review.txt: hash does not match" in text
    assert "missing.txt: not found" in text
    # A changed or missing evidence file warns; the matching file adds nothing.
    warnings = [f for f in result["findings"] if f["check"] == "edit_list.evidence"]
    assert [f["severity"] for f in warnings] == ["warn", "warn"]
    assert [Path(f["evidence"]["path"]).name for f in warnings] == [
        "other-review.txt",
        "missing.txt",
    ]
    assert "does not match its recorded hash" in warnings[0]["message"]
    assert warnings[0]["evidence"]["actual_sha256"] != digest
    assert "is missing" in warnings[1]["message"]
    assert result["checks"]["edit_list.evidence"] == {
        "ran": True,
        "status": "warn",
        "reason": None,
    }
    assert result["status"] in {"warn", "fail"}


def test_matching_attested_evidence_is_not_a_pass(synth_media, tmp_path: Path) -> None:
    evidence = tmp_path / "cut-review.txt"
    evidence.write_text("all 36 cuts reviewed by ear\n")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    edl = _ranges(
        tmp_path / "attested.json",
        synth_media.path("calib"),
        [synth_media.spans[0]],
        composite=True,
        evidence=[{"path": evidence.name, "sha256": digest.upper()}],
    )
    result = verify(str(synth_media.path("clean")), edl=str(edl), asr="none")
    assert result["edit_list"]["cut_coverage"] == "attested"
    assert not [f for f in result["findings"] if f["check"] == "edit_list.evidence"]
    assert result["checks"]["edit_list.evidence"] == {
        "ran": False,
        "status": "skipped",
        "reason": "checked outside Yap Studio; the recorded hash matches, "
        "not counted as a pass",
    }
    assert (
        "  Attested evidence: checked outside Yap Studio; the recorded hash matches, "
        "not counted as a pass" in result["summary"]["text"]
    )


def test_cuts_count_as_checked_only_when_compared_with_a_source(
    synth_media, tmp_path: Path
) -> None:
    spans = synth_media.spans
    clean = str(synth_media.path("clean"))
    edl = _edit_list(tmp_path / "full.json", synth_media.path("calib"), spans)
    full = verify(clean, edl=str(edl), asr="none")
    assert full["edit_list"]["cut_coverage"] == "checked"
    assert full["edit_list"]["checked_cuts"] == full["edit_list"]["planned_cuts"] == len(spans) - 1
    assert "Cut coverage:" not in full["summary"]["text"]

    missing = verify(
        clean, edl=str(_edit_list(tmp_path / "gone.json", tmp_path / "missing.mov", spans)),
        asr="none",
    )
    assert missing["edit_list"]["cut_coverage"] == "unavailable"
    assert missing["edit_list"]["checked_cuts"] == 0
    assert (
        f"  Cut coverage: unavailable. The edit list has {len(spans) - 1} cuts, but none "
        "could be compared with a source." in missing["summary"]["text"]
    )

    mixed = tmp_path / "mixed.json"
    mixed.write_text(
        json.dumps(
            {
                "sources": {
                    "main": str(synth_media.path("calib")),
                    "gone": str(tmp_path / "x.mov"),
                },
                "ranges": [
                    {"source": "main", "start": spans[0][0], "end": spans[0][1]},
                    {"source": "main", "start": spans[1][0], "end": spans[1][1]},
                    {"source": "gone", "start": spans[2][0], "end": spans[2][1]},
                ],
            }
        )
    )
    partial = verify(clean, edl=str(mixed), asr="none")
    assert partial["edit_list"]["cut_coverage"] == "partial"
    assert (partial["edit_list"]["checked_cuts"], partial["edit_list"]["planned_cuts"]) == (1, 2)
    assert (
        "  Cut coverage: partial. 1 of 2 cuts were checked against their sources; "
        "the others could not be compared." in partial["summary"]["text"]
    )


def test_declared_joins_cover_only_the_planned_cuts_at_their_frames(
    synth_media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spans = synth_media.spans[:3]
    fps = Fraction(30)
    lengths = [
        snap_frame(Fraction(str(end)), fps) - snap_frame(Fraction(str(start)), fps)
        for start, end in spans
    ]
    cut_frames = [lengths[0], lengths[0] + lengths[1]]
    module = importlib.import_module("yap_studio.verify")
    monkeypatch.setattr(module, "resolve_backend", lambda *_: ScriptedBackend(_declared_script))

    def coverage(name: str, frames: list[int]) -> dict:
        joins = [{"out_frame": frame, "before": "alpha beta", "after": "gamma"} for frame in frames]
        # The source is missing, so no planned cut can be compared with it.
        edl = _ranges(tmp_path / f"{name}.json", tmp_path / "missing.mov", spans, joins=joins)
        return verify(str(synth_media.path("clean")), edl=str(edl), asr="auto")["edit_list"]

    every = coverage("every", cut_frames)
    assert (every["cut_coverage"], every["checked_cuts"], every["declared_cuts"]) == (
        "declared",
        0,
        2,
    )
    one = coverage("one", cut_frames[:1])
    assert (one["cut_coverage"], one["declared_cuts"]) == ("partial", 1)
    assert "1 of 2 cuts were checked (0 against their sources, 1 by declared joins)" in (
        one["cut_coverage_reason"]
    )
    elsewhere = coverage("elsewhere", [5])
    assert (elsewhere["cut_coverage"], elsewhere["declared_cuts"]) == ("unavailable", 0)
    assert "no declared join sits at a planned cut" in elsewhere["cut_coverage_reason"]


def test_verify_missing_source_skips_unmeasured_join_gaps(
    synth_media, tmp_path: Path
) -> None:
    edl = _edit_list(tmp_path / "missing-source.json", tmp_path / "missing.mov", synth_media.spans)

    result = verify(
        str(synth_media.path("clean")), edl=str(edl), asr="none", detail="full"
    )

    assert result["cuts"]
    assert all(cut["gap_ms"] is None for cut in result["cuts"])
    landing_notes = [
        finding
        for finding in result["findings"]
        if finding["evidence"].get("landing") == "planned"
    ]
    assert landing_notes
    assert all(
        side["offset_source"] == "none"
        for finding in landing_notes
        for side in (finding["evidence"]["end"], finding["evidence"]["start"])
    )
    assert result["checks"]["join.gap"] == {
        "ran": False,
        "status": "skipped",
        "reason": "audio landings could not be measured",
    }


def test_verify_without_detected_cuts_skips_join_checks(synth_media) -> None:
    result = verify(str(synth_media.path("calib")), asr="none")

    assert result["cuts"] == []
    for check in ("join.click", "join.frames", "join.listen"):
        assert result["checks"][check] == {
            "ran": False,
            "status": "skipped",
            "reason": "no cuts found in the picture",
        }


def test_scripted_words_run_and_full_detail_is_present(
    synth_media,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans)
    module = importlib.import_module("yap_studio.verify")
    monkeypatch.setattr(module, "resolve_backend", lambda *_: ScriptedBackend(_script))
    result = verify(
        str(synth_media.path("clean")),
        edl=str(edl),
        asr="auto",
        detail="full",
    )
    assert result["checks"]["join.words"]["ran"] is True
    assert result["segments"]
    assert all("detail" in cut for cut in result["cuts"])
    assert all("expected" in cut["detail"]["words"] for cut in result["cuts"])
    assert all(isinstance(cut["detail"]["click_peak_dbfs"], float) for cut in result["cuts"])
    assert all(cut["detail"]["click_floor_dbfs"] == -60 for cut in result["cuts"])
    assert list(result) == TOP_LEVEL_KEYS


def test_relative_source_override_is_resolved_from_cwd(
    synth_media,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "calib.mov"
    shutil.copyfile(synth_media.path("calib"), source)
    edl_dir = tmp_path / "lists"
    edl_dir.mkdir()
    edl = _edit_list(edl_dir / "edit.json", edl_dir / "missing.mov", synth_media.spans)
    monkeypatch.chdir(tmp_path)
    result = verify(
        str(synth_media.path("clean")),
        edl=str(edl),
        source=["calib.mov"],
        asr="none",
    )
    assert result["sources"] == [{"id": "main", "path": str(source), "found": True}]


def test_source_override_preserves_selected_audio_stream(synth_media, tmp_path: Path) -> None:
    reference = synth_media.path("calib")
    source = tmp_path / "streams.mov"
    duration = ffmpeg.probe(reference).duration_s
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-i", str(reference), "-f", "lavfi", "-i",
        f"sine=frequency=440:sample_rate=48000:duration={duration}",
        "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0",
        "-c:v", "copy", "-c:a", "pcm_s16le", str(source),
    ], timeout_s=120)
    edl = tmp_path / "edit.json"
    edl.write_text(json.dumps({
        "sources": {"main": {"path": "missing.mov", "audio_stream": 1}},
        "segments": [{"start": start, "end": end} for start, end in synth_media.spans],
    }))
    result = verify(str(synth_media.path("clean")), edl=str(edl), source=[str(source)],
                    asr="none", detail="full")
    assert result["sources"][0]["audio_stream"] == 1
    assert all(row["audio_landing_s"] is not None for row in result["segments"])
    assert result["checks"]["join.words"]["status"] == "pass"


def test_unavailable_audio_stream_reports_missing_join_comparisons(tmp_path: Path) -> None:
    source, words = synth.speech_source(
        tmp_path / "speech.mov",
        [
            "Bright planets circle slowly over quiet hills on calm winter evenings",
            "Seven green boxes wait beside the open door near our garden",
        ],
    )
    spans = [(words[0].start, words[10].end), (words[11].start, words[21].end)]
    render = synth.render_clean(source, spans, tmp_path / "render.mp4")
    edl = tmp_path / "edit.json"
    edl.write_text(json.dumps({
        "sources": {"main": {"path": str(source), "audio_stream": 5}},
        "segments": [{"start": start, "end": end} for start, end in spans],
    }))

    result = verify(str(render), edl=str(edl), asr="none", detail="full")

    assert len(result["segments"]) == 2
    assert all(row["source_id"] == "main" for row in result["segments"])
    assert all(row["reason"] == "no match" for row in result["segments"])
    assert result["edit_list"]["cut_coverage"] == "unavailable"
    assert result["edit_list"]["planned_cuts"] == 1
    assert result["edit_list"]["checked_cuts"] == 0
    for check in ("join.words", "join.frames", "join.gap"):
        assert result["checks"][check]["ran"] is False
        assert result["checks"][check]["status"] == "skipped"
    assert len(result["cuts"]) == 1
    cut = result["cuts"][0]["detail"]
    assert cut["words"]["reason"] == cut["frames"]["reason"] == "source not found"
    assert cut["gap_ms"] is cut["click_db_control"] is None
    assert cut["level_left_db"] is cut["level_right_db"] is None


def _without_audio(source: Path, output: Path) -> None:
    ffmpeg.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            str(output),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )


def _trim_audio_to_five_seconds(source: Path, output: Path) -> None:
    ffmpeg.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
            "-af",
            "atrim=0:5",
            str(output),
        ],
        timeout_s=ffmpeg.DEFAULT_TIMEOUT_S,
    )


@pytest.mark.parametrize("with_edit_list", [False, True])
def test_verify_continues_when_sound_ends_before_picture_cuts(
    synth_media,
    tmp_path: Path,
    with_edit_list: bool,
) -> None:
    render = synth_media.cached(
        "verify-clean-audio-five-seconds",
        lambda output: _trim_audio_to_five_seconds(synth_media.path("clean"), output),
    )
    edl = _edit_list(tmp_path / "short-audio.json", synth_media.path("calib"), synth_media.spans)
    result = verify(
        str(render),
        edl=str(edl) if with_edit_list else None,
        asr="none",
        detail="full",
    )

    before = [cut for cut in result["cuts"] if cut["time_s"] < 5]
    after = [cut for cut in result["cuts"] if cut["time_s"] > 5]
    assert before and after
    assert all(cut["click_db"] is not None for cut in before)
    assert all(cut["click_db"] is None for cut in after)
    assert all(cut["frames"] == "pass" for cut in result["cuts"] if with_edit_list)
    if with_edit_list:
        assert all(cut["detail"]["pause_s"] is not None for cut in before)
        assert all(
            (cut["gap_ms"] is not None) == (cut["landing"] == "measured")
            for cut in before
        )
        assert all(
            cut["gap_ms"] is None
            and cut["detail"]["pause_s"] is None
            and cut["words"] == "skipped"
            and cut["detail"]["words"]["reason"] == "the sound ends before this cut"
            for cut in after
        )
    else:
        assert all(cut["detail"]["pause_s"] is not None for cut in before)
        assert all(cut["detail"]["pause_s"] is None for cut in after)
        searches = next(
            finding
            for finding in result["findings"]
            if finding["check"] == "detect.cuts"
        )["evidence"]["audio_searches"]
        after_indices = {cut["index"] for cut in after}
        skipped = [search for search in searches if search["cut"] in after_indices]
        assert all(search["reason"] == "the sound ends before this cut" for search in skipped)
    length = next(
        finding for finding in result["findings"] if finding["check"] == "structure.av_length"
    )
    assert length["severity"] == "warn"
    assert "earlier than the picture" in length["message"]


@pytest.mark.parametrize("with_edit_list", [False, True])
def test_verify_without_audio_runs_picture_structure_and_platform(
    synth_media,
    tmp_path: Path,
    with_edit_list: bool,
) -> None:
    render = synth_media.cached(
        "verify-clean-no-audio",
        lambda output: _without_audio(synth_media.path("clean"), output),
    )
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans)
    result = verify(
        str(render),
        edl=str(edl) if with_edit_list else None,
        asr="none",
        detail="full",
        platform="youtube",
    )
    for check in (
        "join.click",
        "join.gap",
        "join.words",
        "join.listen",
        "sync.av_offset",
        "sync.drift",
        "plan.samples",
        "structure.av_length",
        "loudness.integrated",
        "loudness.true_peak",
        "loudness.parts",
    ):
        assert result["checks"][check] == {
            "ran": False,
            "status": "skipped",
            "reason": "no audio in the render",
        }
    platform_checks = [check for check in result["checks"] if check.startswith("platform.")]
    assert all(result["checks"][check]["ran"] for check in platform_checks)
    if with_edit_list:
        assert result["checks"]["plan.picture"]["ran"] is True
        assert result["checks"]["join.frames"]["ran"] is True
        assert all(row["click_db"] is row["gap_ms"] is None for row in result["cuts"])
        assert all(
            row["detail"]["click_db"]
            is row["detail"]["click_peak_dbfs"]
            is row["detail"]["gap_ms"]
            is None
            for row in result["cuts"]
        )
    else:
        assert result["checks"]["detect.cuts"]["ran"] is True
        assert result["checks"]["join.frames"]["ran"] is True
        assert all(row["click_db"] is None for row in result["cuts"])
    summary = result["summary"]["text"]
    assert "Loudness not measured" in summary
    assert (
        "  Sound checks (clicks, gaps, words, sync, loudness): no audio in the render"
        in summary
    )
    assert "join.click:" not in summary
    assert "structure.av_length:" not in summary
    assert all(result["checks"][row["check"]]["ran"] for row in result["findings"])
    assert exit_code(result, strict=False) == (1 if result["status"] == "fail" else 0)


@pytest.mark.parametrize("name", ["clean", "perclip", "clean_shift_samples"])
def test_every_finding_belongs_to_a_check_that_ran(synth_media, tmp_path: Path, name: str) -> None:
    edl = _edit_list(tmp_path / f"{name}.json", synth_media.path("calib"), synth_media.spans)
    result = verify(str(synth_media.path(name)), edl=str(edl), asr="none")
    assert all(result["checks"][row["check"]]["ran"] for row in result["findings"])


def test_report_receipt_never_hashes_itself(synth_media, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    expected = {
        "path": str(output),
        "sha256": None,
        "bytes": None,
        "role": "this report",
    }
    for _ in range(2):
        result = verify(str(synth_media.path("clean")), asr="none", report=str(output))
        assert result["receipt"]["outputs"] == [expected]
        assert json.loads(output.read_text())["receipt"]["outputs"] == [expected]


def test_click_audio_is_hashed_in_edit_receipt(synth_media, tmp_path: Path) -> None:
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans[:2])
    digests = []
    for name, value in (("a", 0.0), ("b", 0.1)):
        stem = synth.float_audio(tmp_path / f"{name}.wav", np.full(48000, value))
        report = tmp_path / f"{name}.json"
        result = _verify(str(synth_media.path("clean")), edl=str(edl), click_audio=stem,
                         asr="none", report=str(report))
        recorded = json.loads(report.read_text())
        entry = next(row for row in recorded["receipt"]["inputs"] if row["path"] == str(stem))
        assert entry["sha256"] == hashlib.sha256(stem.read_bytes()).hexdigest()
        assert result["receipt"]["inputs"] == recorded["receipt"]["inputs"]
        digests.append(entry["sha256"])
    assert digests[0] != digests[1]


def test_verify_refuses_existing_transcript_report_before_probe(
    synth_media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "transcript.json"
    output.write_text('{"schema": "yap-studio/transcript@1"}', encoding="utf-8")
    monkeypatch.setattr(
        ffmpeg,
        "probe",
        lambda *_args, **_kwargs: pytest.fail("report validation must run before probing"),
    )

    with pytest.raises(OutputPathError, match="not a yap-studio output"):
        verify(str(synth_media.path("clean")), asr="none", report=str(output))


@pytest.mark.parametrize("use_stem", [False, True])
def test_words_audio_is_forwarded_hashed_and_reported(
    synth_media, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_stem: bool,
) -> None:
    module = importlib.import_module("yap_studio.verify")
    edl = _edit_list(tmp_path / "edit.json", synth_media.path("calib"), synth_media.spans[:2])
    render = synth_media.path("clean")
    stem = synth.float_audio(tmp_path / "no-effects.wav", audio.decode(render)[1])
    seen = []
    check = module.check_joins

    def record(*args, **kwargs):
        seen.append(kwargs.get("words_audio"))
        return check(*args, **kwargs)

    monkeypatch.setattr(module, "check_joins", record)
    result = _verify(str(render), edl=str(edl), asr="none",
                     **({"words_audio": stem} if use_stem else {}))
    assert seen == [stem.resolve() if use_stem else None]
    words = result["checks"]["join.words"]
    if use_stem:
        assert words["words_source"] == "no_effects_stem"
        row, = [row for row in result["receipt"]["inputs"] if row["path"] == str(stem)]
        assert row["sha256"] == hashlib.sha256(stem.read_bytes()).hexdigest()
    else:
        assert "words_source" not in words
        assert not any(row["path"] == str(stem) for row in result["receipt"]["inputs"])


def test_words_audio_is_validated_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "render.mp4"
    media.touch()
    monkeypatch.setattr(ffmpeg, "probe", lambda *_args, **_kwargs: pytest.fail("must not probe"))
    with pytest.raises(InputError, match="missing-stem.wav"):
        _verify(str(media), words_audio=tmp_path / "missing-stem.wav", asr="none")
