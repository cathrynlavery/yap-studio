"""Offline reviewer notes, exact frame ranges, and guarded findings files."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from pathlib import Path

import pytest
import synth

from yap_studio import cli, ffmpeg, paths
from yap_studio.errors import OutputPathError, ValidationError
from yap_studio.receipt import sha256_file
from yap_studio.watch import FINDINGS_SCHEMA, Finding, note_from_file
from yap_studio.watch.findings import (
    KINDS,
    SEVERITIES,
    SOURCES,
    default_findings_path,
    format_findings,
    load_findings,
    parse_findings_input,
    record_findings,
)


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synth.hold_render(tmp_path_factory.mktemp("watch-findings") / "cut.mp4")


def point(at=0.5, **changes) -> dict:
    return {
        "at": at,
        "kind": "other",
        "severity": "medium",
        "reason": "The picture needs attention.",
        "fix": "Adjust the picture.",
        **changes,
    }


def span(start=1.0, end=2.0, **changes) -> dict:
    entry = point(**changes)
    entry.pop("at")
    return {"start": start, "end": end, **entry}


def parse(data) -> tuple[str | None, list[dict]]:
    return parse_findings_input(data, fps=Fraction(30), duration_s=3.5, frames=105)


def note_file(tmp_path: Path, data) -> Path:
    target = tmp_path / "note.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


def note(video: Path, tmp_path: Path, data, *options: str) -> int:
    return cli.main(
        [
            "watch",
            "note",
            str(video),
            "--file",
            str(note_file(tmp_path, data)),
            "--findings",
            str(tmp_path / "saved.findings.json"),
            *options,
        ]
    )


def test_constants_and_default_path(tmp_path: Path) -> None:
    assert FINDINGS_SCHEMA == "yap-studio/watch-findings@1"
    assert KINDS == ("slow", "jarring_cut", "face_covered", "caption_issue", "audio", "other")
    assert SEVERITIES == ("low", "medium", "high")
    assert SOURCES == ("agent", "machine")
    assert default_findings_path(tmp_path / "cut.final.mp4") == tmp_path / "cut.final.findings.json"


def test_note_from_file(video: Path, tmp_path: Path) -> None:
    source = note_file(
        tmp_path,
        {
            "reviewer": "  Reviewer A  ",
            "findings": [span("0:00.40", "0:01.20", kind="face_covered")],
        },
    )
    target = tmp_path / "saved.findings.json"
    result = note_from_file(video, source, findings_path=target)
    assert (result["added"], result["duplicates"], result["removed"], result["total"]) == (
        1,
        0,
        0,
        1,
    )
    assert result["path"] == str(target)
    finding = result["findings"][0]
    assert finding["source"] == "agent" and finding["reviewer"] == "Reviewer A"
    assert (finding["start_s"], finding["end_s"]) == (0.4, 1.2)
    assert (finding["start_frame"], finding["end_frame"]) == (12, 36)
    assert load_findings(video, findings_path=target)["findings"] == result["findings"]
    original = target.read_bytes()
    again = note_from_file(video, source, findings_path=target)
    assert (again["added"], again["duplicates"], again["total"]) == (0, 1, 1)
    assert target.read_bytes() == original
    replaced = note_from_file(video, source, findings_path=target, replace=True)
    assert (replaced["added"], replaced["removed"], replaced["total"]) == (1, 1, 1)
    assert replaced["findings"][0]["id"] == "f2"
    for input_file in (source, video):
        original = input_file.read_bytes()
        with pytest.raises(OutputPathError, match="input"):
            note_from_file(video, source, findings_path=input_file)
        assert input_file.read_bytes() == original


def test_points_ranges_half_open_and_clamping(video: Path, tmp_path: Path) -> None:
    reviewer, entries = parse(
        {
            "reviewer": "  Reviewer A  ",
            "findings": [
                point("0:00.50", reason="  A point.  ", fix="  Adjust it.  "),
                span("0:01.10", "0:02.20"),
                span(0.101, 0.2),
                span(0.101, 0.200001, reason="Near the frame boundary."),
                span(0.101, 0.102),
                span(3.48, 3.53),
                point(3.53),
                point(0.999999),
            ],
        }
    )
    assert reviewer == "Reviewer A"
    assert [(entry["start_frame"], entry["end_frame"]) for entry in entries] == [
        (15, None),
        (33, 66),
        (3, 6),
        (3, 6),
        (3, 4),
        (104, 105),
        (104, None),
        (30, None),
    ]
    assert entries[0]["reason"] == "A point." and entries[0]["fix"] == "Adjust it."
    result = record_findings(
        video, entries, source="agent", reviewer=reviewer, findings_path=tmp_path / "saved.json"
    )
    data = json.loads(Path(result["path"]).read_text())
    assert set(data) == {"schema", "video", "next_id", "findings"}
    assert data["schema"] == FINDINGS_SCHEMA and data["next_id"] == 9
    assert data["video"] == {
        "file": "cut.mp4",
        "sha256": sha256_file(video),
        "fps": "30/1",
        "frames": 105,
        "duration_s": 3.5,
    }
    saved = {entry["id"]: entry for entry in result["findings"]}
    assert (saved["f2"]["start_s"], saved["f2"]["end_s"]) == (1.1, 2.2)
    assert (saved["f6"]["start_s"], saved["f6"]["end_s"]) == (3.48, 3.53)
    finding = Finding.from_dict(saved["f2"])
    assert finding.to_dict() == saved["f2"]
    with pytest.raises(FrozenInstanceError):
        finding.kind = "slow"


def test_hours_and_fractional_fps() -> None:
    _, entries = parse_findings_input(
        [point("1:02:03.40"), span("62:04.1", "1:02:05.25")],
        fps=Fraction(30000, 1001),
        duration_s=4000,
        frames=119880,
    )
    assert entries[0]["start_s"] == 3723.4
    assert entries[0]["start_frame"] == 111590
    assert entries[1]["start_s"] == 3724.1 and entries[1]["end_s"] == 3725.25


def test_all_batch_errors_reported_and_no_write(video: Path, tmp_path: Path, capsys) -> None:
    invalid = [
        point(kind="typo"),
        point(severity="urgent"),
        {key: value for key, value in point().items() if key != "reason"},
        point(fix=" "),
        point(severty="low"),
        span(2, 2),
        span(2, 1),
        {**span(), "at": 1.5},
        point(3.54),
        point(float("nan")),
        point(True),
        point(-1),
        point(float("inf")),
        {**point(), "end": 2},
        {key: value for key, value in span().items() if key != "end"},
        point(reason="bad\ttext", fix="bad\x00text"),
    ]
    with pytest.raises(ValidationError) as raised:
        parse(invalid)
    message = raised.value.message
    fields = [
        "kind",
        "severity",
        "reason",
        "fix",
        "severty",
        "end",
        "end",
        "at",
        "at",
        "at",
        "at",
        "at",
        "at",
        "at",
        "end",
        "reason",
    ]
    assert all(f"findings[{index}].{field}" in message for index, field in enumerate(fields))
    assert "findings[15].fix" in message
    assert note(video, tmp_path, invalid) == 2
    error = capsys.readouterr().err
    assert all(f"findings[{index}].{field}" in error for index, field in enumerate(fields))
    assert not (tmp_path / "saved.findings.json").exists()
    assert note(video, tmp_path, invalid, "--json") == 2
    error = json.loads(capsys.readouterr().err)["error"]["message"]
    assert all(f"findings[{index}].{field}" in error for index, field in enumerate(fields))


def test_invalid_batch_leaves_existing_file_untouched(video: Path, tmp_path: Path, capsys) -> None:
    assert note(video, tmp_path, [point()]) == 0
    capsys.readouterr()
    target = tmp_path / "saved.findings.json"
    before = target.read_bytes()
    assert note(video, tmp_path, [point(1), point(kind="typo")], "--replace") == 2
    assert "findings[1].kind" in capsys.readouterr().err
    assert target.read_bytes() == before


@pytest.mark.parametrize(
    "entry,field",
    [
        (point(None), "at"),
        (point("00:60"), "at"),
        (point("1:60:00"), "at"),
        (point("-1:00"), "at"),
        (point("1.5"), "at"),
        (point("bad"), "at"),
        (point(reason=""), "reason"),
        (point(reason="x" * 1001), "reason"),
        (point(fix="x" * 1001), "fix"),
        (point(fix=False), "fix"),
        (point(reason="x\r\ny"), "reason"),
        (point(reason="x\x7fy"), "reason"),
        (point(kind=[]), "kind"),
        (span(False, 1), "start"),
        (span(0, float("inf")), "end"),
        (span(-1, 2), "start"),
        (span(0, 3.54), "end"),
        ({"kind": "other"}, "start"),
        (point("9" * 5000 + ":00:00"), "at"),
    ],
)
def test_invalid_entry_fields(entry: dict, field: str) -> None:
    with pytest.raises(ValidationError, match=rf"findings\[0\].{field}"):
        parse([entry])


@pytest.mark.parametrize("reviewer", ["", "x" * 65, "bad\nlabel", "bad\tlabel", None, False])
def test_invalid_reviewer(reviewer) -> None:
    with pytest.raises(ValidationError, match="reviewer"):
        parse({"reviewer": reviewer, "findings": [point(kind="bad")]})


@pytest.mark.parametrize("data", [None, {}, "bad", {"findings": {}}, [None]])
def test_invalid_input_shape(data) -> None:
    with pytest.raises(ValidationError, match="findings"):
        parse(data)


def test_limit_reports_all_errors_in_json(video: Path, tmp_path: Path, capsys) -> None:
    entries = [point(reason="") for _ in range(501)]
    assert note(video, tmp_path, entries, "--json") == 2
    error = json.loads(capsys.readouterr().err)["error"]["message"]
    assert "at most 500" in error
    assert all(f"findings[{index}].reason" in error for index in range(501))
    assert not (tmp_path / "saved.findings.json").exists()
    assert len(parse([point() for _ in range(500)])[1]) == 500


def test_multiline_text_is_stripped_and_printed_on_one_line(video: Path, tmp_path: Path) -> None:
    reviewer, entries = parse(
        {
            "reviewer": "  Review  ",
            "findings": [point(reason=" \nFirst\nsecond\n ", fix=" \nFix\nit\n ")],
        }
    )
    result = record_findings(
        video, entries, source="agent", reviewer=reviewer, findings_path=tmp_path / "saved.json"
    )
    assert result["findings"][0]["reason"] == "First\nsecond"
    assert format_findings(result["findings"]) == [
        "0:00.50 | other | medium | agent (Review) | First second | fix: Fix it"
    ]


def test_append_idempotence_and_numeric_id_sort(video: Path, tmp_path: Path) -> None:
    target = tmp_path / "saved.json"
    first = record_findings(video, [point(), point()], source="agent", findings_path=target)
    assert (first["added"], first["duplicates"], first["removed"], first["total"]) == (1, 1, 0, 1)
    original = target.read_bytes()
    again = record_findings(video, [point(0.501)], source="agent", findings_path=target)
    assert (again["added"], again["duplicates"], again["total"]) == (0, 1, 1)
    assert target.read_bytes() == original
    result = record_findings(
        video,
        [point(reason=f"Reason {index}") for index in range(11)],
        source="agent",
        findings_path=target,
    )
    assert result["total"] == 12
    assert [entry["id"] for entry in result["findings"]] == [f"f{i}" for i in range(1, 13)]


def test_replace_one_reviewer_and_never_reuse_ids(video: Path, tmp_path: Path, capsys) -> None:
    assert note(video, tmp_path, {"reviewer": "A", "findings": [point()]}) == 0
    assert note(video, tmp_path, {"reviewer": "B", "findings": [point(1)]}) == 0
    capsys.readouterr()
    assert (
        note(video, tmp_path, {"reviewer": "A", "findings": [point(2)]}, "--replace", "--json") == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert (result["added"], result["removed"], result["total"]) == (1, 1, 2)
    assert [(entry["id"], entry["reviewer"]) for entry in result["findings"]] == [
        ("f2", "B"),
        ("f3", "A"),
    ]
    assert note(video, tmp_path, {"reviewer": "A", "findings": []}, "--replace") == 0
    capsys.readouterr()
    assert note(video, tmp_path, {"reviewer": "A", "findings": [point()]}, "--json") == 0
    result = json.loads(capsys.readouterr().out)
    assert {entry["id"] for entry in result["findings"]} == {"f2", "f4"}
    assert json.loads((tmp_path / "saved.findings.json").read_text())["next_id"] == 5


def test_duplicate_identity_is_per_source(video: Path, tmp_path: Path) -> None:
    target = tmp_path / "saved.json"
    record_findings(video, [point()], source="agent", reviewer="A", findings_path=target)
    result = record_findings(video, [point()], source="agent", reviewer="B", findings_path=target)
    assert result["duplicates"] == 1 and result["added"] == 0
    machine = record_findings(video, [point()], source="machine", findings_path=target)
    assert machine["added"] == 1 and machine["total"] == 2


def test_machine_replaces_all_machines_and_keeps_agents(video: Path, tmp_path: Path) -> None:
    target = tmp_path / "saved.json"
    record_findings(video, [point()], source="agent", findings_path=target)
    entry = {**span(2, 3.5, kind="slow"), "check": "watch.still", "evidence": {"mean": 0.2}}
    first = record_findings(video, [entry], source="machine", reviewer="m1", findings_path=target)
    result = record_findings(video, [entry], source="machine", reviewer="m2", findings_path=target)
    assert (result["added"], result["removed"], result["total"]) == (1, 1, 2)
    assert [finding["id"] for finding in result["findings"]] == ["f1", "f3"]
    assert first["findings"][1]["check"] == "watch.still"
    assert result["findings"][1]["evidence"] == {"mean": 0.2}
    empty = record_findings(video, [], source="machine", findings_path=target)
    assert empty["removed"] == 1 and empty["total"] == 1
    assert empty["findings"][0]["source"] == "agent"


@pytest.mark.parametrize("evidence", [{"x": True}, {"x": float("nan")}, {"x": "a"}, []])
def test_machine_evidence_requires_numbers(video: Path, tmp_path: Path, evidence) -> None:
    target = tmp_path / "saved.json"
    with pytest.raises(ValidationError, match=r"findings\[0\].evidence"):
        record_findings(
            video, [{**point(), "evidence": evidence}], source="machine", findings_path=target
        )
    assert not target.exists()


def test_agent_metadata_and_source_are_validated(video: Path, tmp_path: Path) -> None:
    target = tmp_path / "saved.json"
    with pytest.raises(ValidationError, match="source"):
        record_findings(video, [point()], source="bad", findings_path=target)
    with pytest.raises(ValidationError, match="check"):
        record_findings(
            video, [{**point(), "check": "watch.still"}], source="agent", findings_path=target
        )
    assert not target.exists()


def test_stale_refused_untouched_and_list_warns(video: Path, tmp_path: Path, capsys) -> None:
    changed = tmp_path / "cut.mp4"
    shutil.copyfile(video, changed)
    assert note(changed, tmp_path, [point()]) == 0
    capsys.readouterr()
    target = tmp_path / "saved.findings.json"
    original = target.read_bytes()
    with changed.open("ab") as stream:
        stream.write(b"different-render")
    assert note(changed, tmp_path, [point(1)], "--replace", "--json") == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "stale_findings" and "--findings NEW_PATH" in error["suggested_action"]
    assert target.read_bytes() == original
    assert cli.main(["watch", "list", str(changed), "--findings", str(target)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert "different render" in lines[0] and "1 findings for cut.mp4" in lines[1]
    assert "0:00.50" in lines[2]
    assert cli.main(["watch", "list", str(changed), "--findings", str(target), "--json"]) == 1
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["stale"] is True and len(loaded["findings"]) == 1
    assert target.read_bytes() == original


@pytest.mark.parametrize(
    "damage",
    [
        "json",
        "schema",
        "list",
        "video",
        "next_id",
        "finding",
        "evidence",
        "duplicate",
        "fps",
    ],
)
def test_malformed_existing_refused_untouched(
    video: Path, tmp_path: Path, capsys, damage: str
) -> None:
    assert note(video, tmp_path, [point()]) == 0
    capsys.readouterr()
    target = tmp_path / "saved.findings.json"
    data = json.loads(target.read_text())
    if damage == "json":
        target.write_bytes(b"{malformed")
    else:
        if damage == "schema":
            data["schema"] = "yap-studio/watch-findings@2"
        elif damage == "list":
            data["findings"] = {}
        elif damage == "video":
            data["video"] = None
        elif damage == "next_id":
            data["next_id"] = 1
        elif damage == "finding":
            data["findings"][0]["end_frame"] = False
        elif damage == "evidence":
            data["findings"][0]["evidence"] = {"score": "bad"}
        elif damage == "duplicate":
            data["findings"].append(data["findings"][0])
        elif damage == "fps":
            data["video"]["fps"] = "30/0"
        target.write_text(json.dumps(data))
    original = target.read_bytes()
    assert note(video, tmp_path, [point(1)]) == 3
    assert "malformed findings" in capsys.readouterr().err
    assert target.read_bytes() == original


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_findings_never_overwrites_video(video: Path, tmp_path: Path, capsys, alias: str) -> None:
    target = video
    if alias == "symlink":
        target = tmp_path / "video.json"
        target.symlink_to(video)
    elif alias == "hardlink":
        target = tmp_path / "video.json"
        os.link(video, target)
    original = video.read_bytes()
    assert (
        cli.main(
            [
                "watch",
                "note",
                str(video),
                "--file",
                str(note_file(tmp_path, [point()])),
                "--findings",
                str(target),
                "--json",
            ]
        )
        == 3
    )
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "unsafe_path"
    assert video.read_bytes() == original


def test_input_json_is_protected(video: Path, tmp_path: Path, capsys) -> None:
    source = note_file(tmp_path, [point()])
    original = source.read_bytes()
    assert (
        cli.main(["watch", "note", str(video), "--file", str(source), "--findings", str(source)])
        == 3
    )
    assert "JSON input" in capsys.readouterr().err
    assert source.read_bytes() == original


@pytest.mark.parametrize(
    "content",
    [b"{bad", b"\xff", b" " * (4 * 1024 * 1024 + 1)],
    ids=["invalid-json", "invalid-utf8", "oversize"],
)
def test_bad_input_json_and_size_limit(video: Path, tmp_path: Path, capsys, content: bytes) -> None:
    source = tmp_path / "note.json"
    source.write_bytes(content)
    assert (
        cli.main(
            [
                "watch",
                "note",
                str(video),
                "--file",
                str(source),
                "--findings",
                str(tmp_path / "saved.json"),
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "4 MB" in error if len(content) > 4 * 1024 * 1024 else "UTF-8 JSON" in error
    assert not (tmp_path / "saved.json").exists()


@pytest.mark.parametrize("unsafe", ["directory", "parent", "null", "foreign", "symlink"])
def test_unsafe_output_paths_are_refused(video: Path, tmp_path: Path, capsys, unsafe: str) -> None:
    target = tmp_path / "saved.json"
    if unsafe == "directory":
        target.mkdir()
    elif unsafe == "parent":
        target = tmp_path / "child" / ".." / "saved.json"
    elif unsafe == "null":
        target = Path(str(target) + "\x00")
    elif unsafe == "foreign":
        target.write_text('{"mine": true}')
    elif unsafe == "symlink":
        target.symlink_to(tmp_path / "missing.json")
    assert (
        cli.main(
            [
                "watch",
                "note",
                str(video),
                "--file",
                str(note_file(tmp_path, [point()])),
                "--findings",
                str(target),
            ]
        )
        == 3
    )
    assert capsys.readouterr().err
    if unsafe == "foreign":
        assert target.read_text() == '{"mine": true}'
    elif unsafe == "directory":
        assert list(target.iterdir()) == []
    elif unsafe == "symlink":
        assert target.is_symlink() and not (tmp_path / "missing.json").exists()
    elif unsafe == "parent":
        assert not (tmp_path / "saved.json").exists()


def test_list_text_filter_json_and_acceptance(video: Path, tmp_path: Path, capsys) -> None:
    assert (
        note(
            video,
            tmp_path,
            {
                "reviewer": "Review A",
                "findings": [
                    span(
                        2,
                        3.5,
                        kind="slow",
                        reason="The frozen stretch drags.",
                        fix="Shorten the hold.",
                    ),
                    span(
                        0.4,
                        1.2,
                        kind="face_covered",
                        severity="high",
                        reason="The card covers the face.",
                        fix="Move the card lower.",
                    ),
                ],
            },
        )
        == 0
    )
    output = capsys.readouterr().out
    target = tmp_path / "saved.findings.json"
    assert output == (
        "Recorded 2 findings for cut.mp4 (0 duplicates skipped, 0 replaced). "
        f"Total 2 in {target}.\n"
    )
    record_findings(video, [point(1.5, kind="audio")], source="machine", findings_path=target)
    base = ["watch", "list", str(video), "--findings", str(target)]
    assert cli.main(base) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "3 findings for cut.mp4: 2 from agents, 1 machine"
    assert "0:00.40 to 0:01.20 | face_covered | high | agent (Review A)" in lines[1]
    assert "The card covers the face." in lines[1] and "fix: Move the card lower." in lines[1]
    assert "0:01.50 | audio | medium | machine" in lines[2]
    assert "0:02.00 to 0:03.50 | slow" in lines[3]
    assert all(chr(0x2013) not in line and chr(0x2014) not in line for line in lines)
    for source, count in [("agent", 2), ("machine", 1), ("all", 3)]:
        assert cli.main([*base, "--source", source, "--json"]) == 0
        loaded = json.loads(capsys.readouterr().out)
        assert loaded["exists"] is True and loaded["stale"] is False
        assert loaded["path"] == str(target) and len(loaded["findings"]) == count
        assert source == "all" or all(item["source"] == source for item in loaded["findings"])
    assert cli.main([*base, "--source", "machine"]) == 0
    header = capsys.readouterr().out.splitlines()[0]
    assert header == "1 findings for cut.mp4: 0 from agents, 1 machine"


def test_missing_file_list_and_default_note(video: Path, tmp_path: Path, capsys) -> None:
    local_video = tmp_path / "cut.mp4"
    shutil.copyfile(video, local_video)
    assert cli.main(["watch", "list", str(local_video)]) == 0
    assert capsys.readouterr().out == "No findings for cut.mp4.\n"
    assert cli.main(["watch", "list", str(local_video), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "path": str(tmp_path / "cut.findings.json"),
        "exists": False,
        "stale": False,
        "video": None,
        "findings": [],
    }
    assert (
        cli.main(["watch", "note", str(local_video), "--file", str(note_file(tmp_path, [point()]))])
        == 0
    )
    capsys.readouterr()
    assert default_findings_path(local_video).is_file()
    assert load_findings(local_video)["exists"] is True


def test_deterministic_bytes(video: Path, tmp_path: Path) -> None:
    _, entries = parse([point(2), span(0.5, 1.5, kind="face_covered")])
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    for target in (one, two):
        record_findings(video, entries, source="agent", reviewer="Review", findings_path=target)
    assert one.read_bytes() == two.read_bytes()
    assert [entry["id"] for entry in json.loads(one.read_text())["findings"]] == ["f2", "f1"]


@pytest.mark.parametrize("container_frames", [None, 99])
def test_probe_does_not_count_frames_and_uses_container_count(
    video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, container_frames: int | None
) -> None:
    original = ffmpeg.probe(video, count_frames=False)
    calls = []

    def probe(source, *, count_frames=True):
        calls.append(count_frames)
        return replace(original, video_frames=container_frames)

    monkeypatch.setattr(ffmpeg, "probe", probe)
    target = tmp_path / "saved.json"
    record_findings(video, [point()], source="agent", findings_path=target)
    assert calls == [False]
    assert json.loads(target.read_text())["video"]["frames"] == (container_frames or 105)


def test_atomic_failure_preserves_previous_file(
    video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "saved.json"
    record_findings(video, [point()], source="agent", findings_path=target)
    original = target.read_bytes()

    def fail_replace(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr(paths.os, "replace", fail_replace)
    with pytest.raises(OutputPathError, match="Could not write findings"):
        record_findings(video, [point(2)], source="agent", findings_path=target)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]
