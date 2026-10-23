"""Taste CLI parsing, decision round trips, and structured errors."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_review_rounds import round_project as round_project

from yap_studio import cli
from yap_studio.feedback.taste import ProposalInbox
from yap_studio.feedback.versions import load_version
from yap_studio.review.model import MediaFacts
from yap_studio.review.store import ReviewStore
from yap_studio.style import config_root
from yap_studio.taste import (
    add_proposals,
    approve,
    load_taste,
    proposals_path,
    render_style,
    revoke,
    rule_id,
    taste_path,
    write_taste_record,
)


@pytest.fixture(autouse=True)
def isolated_taste(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


def _body(words: int = 3) -> dict:
    return {
        "scope": "captions",
        "sentence": f"Captions show at most {words} words.",
        "machine": {
            "kind": "style_field",
            "field": "captions.max_words_per_phrase",
            "value": words,
        },
        "sources": [
            {
                "type": "comment",
                "media_sha256": "a" * 64,
                "comment_id": "c1",
                "label": "fix",
                "start_frame": None,
                "end_frame": None,
                "quote": None,
            }
        ],
    }


def _file(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_propose_review_comments_text_and_json(
    round_project: Path, tmp_path: Path, monkeypatch, capsys, json_output: bool,
) -> None:
    from fractions import Fraction

    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    version = load_version(round_project, "v1")
    sha = version.media["sha256"]
    review = ReviewStore(tmp_path / "data")
    comment = review.add_comment(sha, MediaFacts("render.mp4", sha, 4, Fraction(30), True),
                                 {"text": "Keep this pace", "kind": "keep", "start_frame": 12})
    source = _file(tmp_path / "from-review.json", {
        "scope": "pacing", "sentence": "Keep this pace.",
        "comments": [{"version": "v1", "id": comment.id}],
    })
    flag = ["--json"] if json_output else []
    command = ["taste", "propose", str(round_project), str(source), *flag]
    assert cli.main(command) == 0
    identity = rule_id(ProposalInbox(round_project).read()[0])
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        assert json.loads(captured.out) == [{"id": identity, "status": "added"}]
    else:
        assert captured.out == f"{identity}: added\n"
    assert cli.main(command) == 0
    captured = capsys.readouterr()
    if json_output:
        assert json.loads(captured.out) == [{"id": identity, "status": "unchanged"}]
    else:
        assert captured.out == f"{identity}: unchanged\n"


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_propose_unknown_comment_exits_two(
    round_project: Path, tmp_path: Path, monkeypatch, capsys, json_output: bool,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_DATA_DIR", str(tmp_path / "data"))
    source = _file(tmp_path / "from-review.json", {
        "scope": "pacing", "sentence": "Keep this pace.",
        "comments": [{"version": "v2", "id": "c7"}],
    })
    flag = ["--json"] if json_output else []
    assert cli.main(["taste", "propose", str(round_project), str(source), *flag]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "no comment c7 on version v2" in captured.err
    if json_output:
        error = json.loads(captured.err)["error"]
        assert error["type"] == "validation_error" and error["suggested_action"]
    assert not ProposalInbox(round_project).path.exists()


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], json_output: bool
) -> None:
    flag = ["--json"] if json_output else []
    body = _body()
    source = _file(tmp_path / "proposal.json", body)
    identity = rule_id(body)
    assert cli.main(["taste", "add-proposal", str(source), *flag]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    if json_output:
        assert json.loads(captured.out) == [{"id": identity, "status": "added"}]
    else:
        assert captured.out == f"{identity}: added\n"
    assert cli.main(["taste", "show", *flag]) == 0
    captured = capsys.readouterr()
    if json_output:
        shown = json.loads(captured.out)
        assert shown["revision"] == 0 and shown["rules"] == []
        assert shown["proposals"][0]["id"] == identity
        assert shown["paths"] == {"taste": str(taste_path()), "proposals": str(proposals_path())}
    else:
        assert "Taste revision 0 (" in captured.out
        assert "0 approved, 1 proposed" in captured.out
        assert f"{identity} [captions]" in captured.out
        assert "captions.max_words_per_phrase = 3" in captured.out
    assert cli.main(["taste", "approve", identity, *flag]) == 0
    captured = capsys.readouterr()
    if json_output:
        approved = json.loads(captured.out)
        assert approved["revision"] == 1 and approved["rule"]["status"] == "approved"
        assert approved["superseded_ids"] == []
    else:
        assert captured.out == f"Approved {identity} at revision 1.\n"
    assert cli.main(["taste", "revoke", identity, *flag]) == 0
    captured = capsys.readouterr()
    if json_output:
        assert json.loads(captured.out)["status"] == "revoked"
    else:
        assert captured.out == f"Revoked {identity} at revision 2.\n"
    second = _body(4)
    _file(source, second)
    assert cli.main(["taste", "add-proposal", str(source), *flag]) == 0
    capsys.readouterr()
    assert cli.main(["taste", "reject", rule_id(second), *flag]) == 0
    captured = capsys.readouterr()
    if json_output:
        assert json.loads(captured.out)["status"] == "rejected"
    else:
        assert captured.out == f"Rejected {rule_id(second)}.\n"
    assert cli.main(["taste", "show", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["revision"] == 2 and shown["rules"] == shown["proposals"] == []
    assert cli.main(["taste", "show", "--all", *flag]) == 0
    captured = capsys.readouterr()
    if json_output:
        shown = json.loads(captured.out)
        assert shown["rules"][0]["status"] == "revoked"
        assert shown["proposals"][0]["status"] == "rejected"
    else:
        assert "[revoked]" in captured.out and "[rejected]" in captured.out


def test_show_all_includes_superseded_rules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "proposal.json"
    for value in (3, 4):
        _file(source, _body(value))
        assert cli.main(["taste", "add-proposal", str(source)]) == 0
        assert cli.main(["taste", "approve", rule_id(_body(value))]) == 0
    capsys.readouterr()
    assert cli.main(["taste", "show", "--all", "--json"]) == 0
    assert [rule["status"] for rule in json.loads(capsys.readouterr().out)["rules"]] == [
        "superseded",
        "approved",
    ]


@pytest.mark.parametrize("command", ["approve", "reject", "revoke"])
def test_unknown_id_exits_three_with_json_error(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["taste", command, "r-000000000000", "--json"]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["type"] == "taste_refusal"
    assert error["code"] in {"taste_not_pending", "taste_not_approved"}
    assert "r-000000000000" in error["message"]
    assert error["suggested_action"]


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("malformed_json", [False, True])
def test_invalid_proposal_exits_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
    malformed_json: bool,
) -> None:
    body = _body(100)
    source = _file(tmp_path / "proposal.json", body)
    if malformed_json:
        source.write_text("{", encoding="utf-8")
    assert (
        cli.main(["taste", "add-proposal", str(source), *(["--json"] if json_output else [])]) == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        error = json.loads(captured.err)["error"]
        assert error["type"] == "validation_error" and error["code"] == "invalid_parameter"
        assert error["suggested_action"]
    else:
        assert "error: " in captured.err and "fix: " in captured.err
    assert not config_root().exists()


def test_text_refusal_includes_fix(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["taste", "revoke", "r-000000000000"]) == 3
    captured = capsys.readouterr()
    assert "error: " in captured.err and "fix: " in captured.err


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("value,expected", [(3, "pass"), (4, "fail"), (None, "not_measured")])
def test_check_rule_exit_code_and_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    json_output: bool, value: int | None, expected: str,
) -> None:
    identity = add_proposals(_body())[0].id
    approve(identity)
    media = tmp_path / "final.mp4"
    media.write_bytes(b"finished render")
    write_taste_record(media, render_style())
    observation = {"value": value, "frames": []} if value is not None else {
        "value": None, "reason": "Needs text recognition.",
    }
    measure = Mock(return_value={"captions.max_words_per_phrase": observation})
    monkeypatch.setattr(import_module("yap_studio.taste.cli"), "measure_style", measure)
    assert cli.main([
        "taste", "check", str(media), *(["--json"] if json_output else []),
    ]) == int(expected == "fail")
    captured = capsys.readouterr()
    assert captured.err == ""
    measure.assert_called_once_with(media, fields=("captions.max_words_per_phrase",))
    if json_output:
        result = json.loads(captured.out)
        assert result["rules"] == [{
            "id": identity, "field": "captions.max_words_per_phrase", "wanted": 3,
            "measured": value, "status": expected,
            "reason": "Needs text recognition." if value is None else None, "frames": [],
        }]
        assert result["coverage"]["ran"] == (value is not None)
        assert len(result["findings"]) == int(expected != "pass")
    else:
        line = f"{identity} captions.max_words_per_phrase: {expected.replace('_', ' ')}"
        if value is None:
            line += " (Needs text recognition.)"
        assert captured.out == line + "\n"
    assert _body()["sentence"] not in captured.out


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("problem", ["absent", "corrupt", "media_changed"])
def test_check_record_errors_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], json_output: bool, problem: str,
) -> None:
    media = tmp_path / "final.mp4"
    media.write_bytes(b"finished render")
    if problem != "absent":
        target = write_taste_record(media, render_style())
        if problem == "corrupt":
            target.write_text("{", encoding="utf-8")
        else:
            media.write_bytes(b"a changed render")
    assert cli.main([
        "taste", "check", str(media), *(["--json"] if json_output else []),
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    reason = {
        "absent": "no taste record next to this render", "corrupt": "corrupt taste record",
        "media_changed": "taste record media hash does not match the media",
    }[problem]
    if json_output:
        error = json.loads(captured.err)["error"]
        assert error["type"] == "input_error"
        assert reason in error["message"]
    else:
        assert "error:" in captured.err and reason in captured.err


def test_check_no_rules_and_receipt_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    media = tmp_path / "final.mp4"
    media.write_bytes(b"finished render")
    write_taste_record(media, render_style())
    assert cli.main(["taste", "check", str(media)]) == 0
    assert capsys.readouterr().out == "no approved taste rule applied to this render\n"
    media.with_suffix(".receipt.json").write_text(
        json.dumps({"style": {"sha256": "0" * 64}}), encoding="utf-8",
    )
    assert cli.main(["taste", "check", str(media), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["rules"] == []
    assert result["findings"][0]["severity"] == "fail"


def test_show_empty_store_is_read_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["taste", "show", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert set(shown) == {"revision", "sha256", "rules", "proposals", "paths", "history"}
    assert shown["history"] == []
    assert shown["revision"] == 0 and shown["sha256"] == load_taste().snapshot().sha256
    assert shown["rules"] == shown["proposals"] == []
    assert not config_root().exists()


def test_taste_help_and_style_parsing_share_main_parser(capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli._parser()
    assert parser.parse_args(["taste", "show", "--json"]).taste_command == "show"
    assert parser.parse_args(["style", "presets", "--json"]).style_command == "presets"
    assert cli.main(["taste", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert all(
        command in help_text for command in ("show", "approve", "reject", "revoke", "add-proposal")
    )
    assert cli.main(["taste"]) == 2


@pytest.mark.parametrize(
    "machine,scope,fragment",
    [
        (
            {"kind": "filler_word", "word": "so", "position": "leading"},
            "cut",
            '(filler word "so", leading)',
        ),
        (
            {"kind": "placement", "layer": "captions", "rule": "clear_of_face"},
            "captions",
            "(placement: captions clear_of_face)",
        ),
    ],
)
def test_show_non_style_machine_forms(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], machine: dict, scope: str, fragment: str
) -> None:
    body = {**_body(), "scope": scope, "machine": machine}
    source = _file(tmp_path / "proposal.json", body)
    assert cli.main(["taste", "add-proposal", str(source)]) == 0
    capsys.readouterr()
    assert cli.main(["taste", "show"]) == 0
    assert fragment in capsys.readouterr().out


def test_approve_names_superseded_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "proposal.json"
    for value in (3, 4):
        _file(source, _body(value))
        assert cli.main(["taste", "add-proposal", str(source)]) == 0
        assert cli.main(["taste", "approve", rule_id(_body(value))]) == 0
    text = capsys.readouterr().out
    assert f"Superseded {rule_id(_body(3))}." in text


@pytest.fixture
def learn_clip(tmp_path: Path) -> Path:
    from yap_studio import ffmpeg

    clip = tmp_path / "vertical.mp4"
    ffmpeg.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=s=180x320:r=30:d=0.8",
            "-vf",
            ",".join(
                f"drawbox=x={x}:y={y}:w=8:h=10:c=white:t=fill"
                for y in (250, 264)
                for x in (45, 75, 105)
            ),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        timeout_s=30,
    )
    return clip


@pytest.mark.parametrize("json_output", [False, True])
def test_learn_cli_outputs_and_refusal(
    learn_clip: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], json_output: bool
) -> None:
    out = tmp_path / "learned"
    args = ["taste", "learn", str(learn_clip), "--out", str(out), "--asr", "none"]
    flags = ["--json"] if json_output else []
    assert cli.main([*args, *flags]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    if json_output:
        data = json.loads(captured.out)
        assert "pacing.cut_frames" in data["measures"]
        assert data["fields"]["captions.text_color"]["status"] == "proposed"
        assert data["paths"]["style"] == str(out / "proposed-style.json")
        assert data["not_measured"]
    else:
        assert "captions.text_color = #" in captured.out
        assert "frames in 1 video)" in captured.out
        assert "Not measured: captions.font:" in captured.out
        assert str(out / "measurements.json") in captured.out
    assert not config_root().exists()
    assert cli.main([*args, *flags]) == 3
    captured = capsys.readouterr()
    if json_output:
        error = json.loads(captured.err)["error"]
        assert error["code"] == "taste_learn_exists" and error["suggested_action"]
    else:
        assert "error:" in captured.err and "fix:" in captured.err
    assert cli.main([*args, "--force", *flags]) == 0
    capsys.readouterr()
    (out / "taste-proposals.json").write_text("{}")
    assert cli.main([*args, "--force", *flags]) == 3
    capsys.readouterr()


def test_learn_horizontal_and_invalid_input_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from yap_studio import ffmpeg

    clip = tmp_path / "horizontal.mp4"
    ffmpeg.run(["ffmpeg", "-f", "lavfi", "-i", "color=s=160x90:d=0.1", str(clip)], timeout_s=30)
    for path in (clip, tmp_path / "missing.mp4"):
        assert (
            cli.main(
                [
                    "taste",
                    "learn",
                    str(path),
                    "--out",
                    str(tmp_path / "out"),
                    "--asr",
                    "none",
                    "--json",
                ]
            )
            == 2
        )
        assert json.loads(capsys.readouterr().err)["error"]["suggested_action"]
    assert not (tmp_path / "out").exists()


def test_import_review_round_spelling_proposals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    feedback = tmp_path / "feedback"
    feedback.mkdir()
    bodies = [
        {
            "scope": "captions",
            "condition": None,
            "sentence": 'Spell "teh" as "the" in captions.',
            "machine": None,
            "sources": [
                {
                    "type": "comment",
                    "media_sha256": "a" * 64,
                    "comment_id": "c3",
                    "label": "fix",
                    "start_frame": None,
                    "end_frame": None,
                    "quote": "teh should be the",
                }
            ],
            "context": 'Caption spelling fix from review. Word fix: "teh" -> "the".',
        },
        {
            "scope": "captions",
            "condition": None,
            "sentence": 'Spell "recieve" as "receive" in captions.',
            "machine": None,
            "sources": [
                {
                    "type": "comment",
                    "media_sha256": "a" * 64,
                    "comment_id": "c4",
                    "label": "fix",
                    "start_frame": 120,
                    "end_frame": 150,
                    "quote": "recieve",
                }
            ],
            "context": 'Caption spelling fix from review. Word fix: "recieve" -> "receive".',
        },
    ]
    source = _file(
        feedback / "taste-proposals.json",
        {"schema": "yap-studio/taste-proposals@1", "proposals": bodies},
    )
    for expected in ("added", "already_present"):
        assert cli.main(["taste", "add-proposal", str(source), "--json"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert [item["status"] for item in result] == [expected, expected]
        assert [item["id"] for item in result] == [rule_id(body) for body in bodies]
    assert cli.main(["taste", "show", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert len(shown["proposals"]) == 2
    assert all(row["status"] == "proposed" for row in shown["proposals"])
    assert shown["rules"] == [] and shown["revision"] == 0


def test_learn_text_reports_rules_and_partial_video_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib

    from yap_studio.taste.learn import LearnResult

    module = importlib.import_module("yap_studio.taste.cli")
    result = LearnResult(
        tmp_path / "proposed-style.json",
        tmp_path / "taste-proposals.json",
        tmp_path / "measurements.json",
        {
            "videos": [{"name": "clip.mp4"}, {"name": "clip.mp4"}],
            "fields": {
                "captions.text_color": {
                    "status": "proposed",
                    "value": "#F8EACA",
                    "per_video": [{"video": 0, "value": "#F8EACA", "frames": [3, 7]}],
                },
                "captions.size_px": {
                    "status": "partial",
                    "value": 80,
                    "reason": "measured in 1 of 2 videos that show captions",
                    "per_video": [{"video": 0, "value": 80, "frames": [3, 7]}],
                },
            },
            "measures": {},
            "proposed_style": {
                "base": {"kind": "schema_defaults", "name": "Schema defaults"},
                "style": {},
                "sources": {
                    "captions.text_color": {
                        "source": "measured",
                        "status": "proposed",
                        "reason": None,
                    },
                    "captions.font": {"source": "base", "status": None, "reason": "Needs OCR."},
                },
            },
            "not_measured": [
                {
                    "field": "captions.text_color",
                    "reason": "No caption.",
                    "video": 1,
                    "name": "clip.mp4",
                }
            ],
        },
        ("Consider shots around 3 seconds long.",),
    )
    monkeypatch.setattr(module, "learn", lambda *args, **kwargs: result)
    args = ["taste", "learn", "first.mp4", "second.mp4", "--out", str(tmp_path)]
    assert cli.main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "Style base: Schema defaults",
        "Measured style fields: captions.text_color",
        "Fields from base: captions.font",
    ]
    assert lines[3:7] == [
        "captions.text_color = #F8EACA (from 2 frames in 1 of 2 videos)",
        "captions.size_px: partial: measured in 1 of 2 videos that show captions",
        "Proposed rule: Consider shots around 3 seconds long.",
        "Not measured in video 2 (clip.mp4): captions.text_color: No caption.",
    ]
    assert lines[-3:] == [
        str(result.style_path),
        str(result.proposals_path),
        str(result.measurements_path),
    ]
    assert cli.main([*args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["proposed_rules"] == list(result.proposed_rules)


def test_show_history_in_json_and_all_text(capsys: pytest.CaptureFixture[str]) -> None:
    identity = add_proposals(_body())[0].id
    approve(identity)
    revoke(identity)
    assert cli.main(["taste", "show", "--json"]) == 0
    assert [entry["action"] for entry in json.loads(capsys.readouterr().out)["history"]] == [
        "approved",
        "revoked",
    ]
    assert cli.main(["taste", "show", "--all"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len([line for line in lines if line.startswith("History revision ")]) == 2
    assert lines[-1].endswith(f"revoked {identity}")


def test_learn_explicit_base_and_outside_style_text(
    learn_clip: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib

    from yap_studio.style import resolve

    module = importlib.import_module("yap_studio.taste.learn")
    video = learn_clip
    monkeypatch.setattr(
        module, "_visual", lambda *args: ({"hook.duration_s": (9.4, [1, 5], {})}, {}, [])
    )
    out = tmp_path / "based"
    assert (
        cli.main(
            [
                "taste",
                "learn",
                str(video),
                "--out",
                str(out),
                "--asr",
                "none",
                "--base",
                "preset:electric",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "Style base: Electric" in text
    assert "hook.duration_s: outside_style: measured 9.4 s, above the style's 8 s limit" in text
    style = json.loads((out / "proposed-style.json").read_text())
    assert style["captions"]["font"] == resolve("preset:electric").data["captions"]["font"]
