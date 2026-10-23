"""Audio-led projects require complete authored picture before delivery."""

import json
import subprocess
from pathlib import Path

import pytest
import synth
from test_cut import time_segment, write_edit
from test_ingest import FakeBackend, make_media
from test_timeline import fragment, insert
from test_timeline import project as project

from yap_studio import receipt, transcript
from yap_studio.project import init_project
from yap_studio.project.status import ProjectContext, status
from yap_studio.stages import assemble as assemble_stage
from yap_studio.stages import cut as cut_stage
from yap_studio.stages import voiceover
from yap_studio.stages.assemble import (
    AssemblyError,
    _graphic_picture_frames,
    require_voiceover_picture,
)
from yap_studio.stages.captions import captions
from yap_studio.stages.graphics import graphics
from yap_studio.stages.ingest import ingest
from yap_studio.stages.music import music
from yap_studio.stages.timeline import add_timeline, resolve_timeline
from yap_studio.stages.voiceover import skip_standin_picture_checks


@pytest.mark.parametrize("audio_only", [False, True])
@pytest.mark.parametrize("start", [0, 2])
def test_chapter_over_insert_is_one_visual_unit_at_default_limit(
    project: Path, audio_only: bool, start: int,
) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["takes"][0]["voiceover"] = audio_only
    data["style"] = "style.json"
    (project / "style.json").write_text(json.dumps({
        "schema_version": 1, "chapters": {"style": "title_card"},
    }))
    (project / "project.json").write_text(json.dumps(data))
    assert ProjectContext(project).project.rules["elements_at_once_max"] == 1
    chapter = {"id": "agent-chapter", "kind": "chapter", "lane": "agent",
               "start": {"time": start}, "hold_s": 1, "title": "A chapter"}
    before = add_timeline(project, fragment(project, chapter), lane="agent").require_valid()
    if audio_only:
        assert voiceover.uncovered_frames(360, [item.to_dict() for item in before.elements]) == [
            (0, 359)]
        assert any(item.rule == "voiceover_uncovered" for item in before.findings)
    result = add_timeline(project, fragment(project,
        insert(project, start={"time": 0}, end={"time": 12})), lane="broll").require_valid()
    assert not any(item.rule in {"visual_overlap", "voiceover_uncovered"}
                   for item in result.findings)
    require_voiceover_picture(360, [item.to_dict() for item in result.elements])
    assert resolve_timeline(project).valid


def test_voiceover_picture_reports_exact_lane_gap() -> None:
    elements = [
        {"kind": "insert", "start_frame": 0, "end_frame": 30},
        {"id": "card", "kind": "graphic", "start_frame": 40, "end_frame": 90},
    ]
    with pytest.raises(AssemblyError, match=r"frames 30-89") as error:
        require_voiceover_picture(90, elements)
    assert error.value.suggested_action == (
        "Plan B-roll inserts over these frames and add them with yap-studio timeline add "
        "FRAGMENT DIR --lane broll; yap-studio timeline check DIR lists frames still uncovered.")
    require_voiceover_picture(90, elements + [
        {"kind": "insert", "start_frame": 30, "end_frame": 90},
    ])


def test_voiceover_chapter_does_not_cover_picture_gap() -> None:
    elements = [
        {"kind": "insert", "start_frame": 0, "end_frame": 60},
        {"kind": "insert", "start_frame": 90, "end_frame": 150},
    ]
    assert voiceover.uncovered_frames(150, elements) == [(60, 89)]
    with pytest.raises(AssemblyError, match=r"frames 60-89"):
        require_voiceover_picture(150, elements)
    elements.append({"kind": "chapter", "start_frame": 60, "end_frame": 90})
    with pytest.raises(AssemblyError, match=r"frames 60-89"):
        require_voiceover_picture(150, elements)
    assert voiceover.uncovered_frames(150, elements) == [(60, 89)]


def test_voiceover_chapter_does_not_cover_timeline_or_status(project: Path) -> None:
    data = json.loads((project / "project.json").read_bytes())
    data["takes"][0]["voiceover"] = True
    (project / "project.json").write_text(json.dumps(data))
    result = add_timeline(project, fragment(project,
        insert(project, start={"time": 0}, end={"time": 2}),
        insert(project, id="broll-two", start={"time": 3}, end={"time": 12})), lane="broll")
    assert result.valid
    warning = next(item for item in result.findings if item.rule == "voiceover_uncovered")
    assert warning.message == "Voiceover picture uncovered at frames 60-89"
    assert any("frames 60-89 have no B-roll yet" in need for need in status(project)["needs"])
    result = add_timeline(project, fragment(project, {
        "id": "agent-chapter", "kind": "chapter", "lane": "agent", "start": {"time": 2},
        "hold_s": 1, "title": "A chapter",
    }), lane="agent")
    assert result.valid
    warning = next(item for item in result.findings if item.rule == "voiceover_uncovered")
    assert warning.message == "Voiceover picture uncovered at frames 60-89"
    assert not any(item.rule == "voiceover_graphic_only" for item in result.findings)
    assert any("frames 60-89 have no B-roll yet" in need for need in status(project)["needs"])


@pytest.mark.parametrize(("elements", "expected"), [
    ([], [(0, 89)]),
    ([{"kind": "insert", "start_frame": 10, "end_frame": 30},
      {"kind": "insert", "start_frame": 20, "end_frame": 40}], [(0, 9), (40, 89)]),
    ([{"kind": "graphic", "start_frame": 0, "end_frame": 45},
      {"kind": "title", "start_frame": 45, "end_frame": 90}], [(45, 89)]),
    ([{"kind": "insert", "start_frame": 0, "end_frame": 90}], []),
    ([{"kind": "insert", "start_frame": 0, "end_frame": 89}], [(89, 89)]),
])
def test_uncovered_frames(elements: list[dict], expected: list[tuple[int, int]]) -> None:
    assert voiceover.uncovered_frames(90, elements) == expected
    assert voiceover.frame_ranges_text(expected) == ", ".join(f"{a}-{b}" for a, b in expected)


@pytest.mark.parametrize(("kinds", "expected"), [
    (frozenset({"insert"}), [(0, 29)]),
    (frozenset({"graphic"}), [(60, 89)]),
    (frozenset(), [(0, 89)]),
])
def test_uncovered_frames_can_limit_picture_kinds(
    kinds: frozenset[str], expected: list[tuple[int, int]],
) -> None:
    elements = [{"kind": "graphic", "start_frame": 0, "end_frame": 60},
                {"kind": "insert", "start_frame": 30, "end_frame": 90}]
    assert voiceover.uncovered_frames(90, elements) == []
    assert voiceover.uncovered_frames(90, elements, kinds=kinds) == expected


def test_uncovered_end_hold_shares_last_content_frame() -> None:
    elements = [{"kind": "insert", "start_frame": 0, "end_frame": 90}]
    assert voiceover.uncovered_frames(90, elements) == []
    require_voiceover_picture(90, elements, frames=(85, 120))
    assert voiceover.uncovered_frames(0, []) == []


def test_split_does_not_cover_voiceover_standin() -> None:
    elements = [{"id": "broll-split", "kind": "insert", "start_frame": 0, "end_frame": 90,
                 "media": {"fit": "split"}}]
    assert voiceover.uncovered_frames(90, elements) == [(0, 89)]
    with pytest.raises(AssemblyError, match="frames 0-89"):
        require_voiceover_picture(90, elements)


def test_voiceover_timeline_warning_keeps_lane_and_check_valid(project: Path) -> None:
    root = project
    assert not any(item.rule == "voiceover_uncovered" for item in resolve_timeline(root).findings)
    project = json.loads((root / "project.json").read_bytes())
    project["takes"][0]["voiceover"] = True
    (root / "project.json").write_text(json.dumps(project))
    result = add_timeline(root, fragment(root, insert(
        root, start={"time": 1}, end={"time": 2})), lane="broll")
    assert result.valid
    warning = next(item for item in result.findings if item.rule == "voiceover_uncovered")
    assert warning.severity == "warning" and warning.elements == warning.lanes == ()
    assert warning.message == warning.bare_message == (
        "Voiceover picture uncovered at frames 0-29, 60-359")
    assert warning.suggested_action == (
        "Plan B-roll inserts over these frames and add them with yap-studio timeline add "
        "FRAGMENT DIR --lane broll; the last insert ends at the cut end.")
    from yap_studio import cli

    assert cli.main(["timeline", "check", str(root)]) == 0


def test_voiceover_status_points_to_coverage_after_music(tmp_path: Path) -> None:
    from test_stage_captions import make_project
    from test_status import CUT_REVIEW

    root = make_project(tmp_path)
    project = json.loads((root / "project.json").read_bytes())
    project["takes"][0]["voiceover"] = True
    project["music"] = {"mode": "none"}
    (root / "project.json").write_text(json.dumps(project))
    need = ("cover the voiceover with picture: frames 0-89 have no B-roll yet; "
            "plan inserts, then yap-studio timeline add FRAGMENT DIR --lane broll")
    assert status(root)["next"] == CUT_REVIEW
    assert need in status(root)["needs"]
    music(root)
    assert status(root)["next"] == need
    cover = root / "cover.json"
    cover.write_text(json.dumps({"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-cover", "kind": "insert", "start": {"time": 0}, "end": {"time": 3},
        "media": {"path": "source.mov", "sha256": receipt.sha256_file(root / "source.mov")},
    }]}))
    assert add_timeline(root, cover, lane="broll").valid
    result = status(root)
    assert need not in result["needs"]
    assert result["next"].startswith("yap-studio captions ")


def test_graphic_only_covers_frames_with_opaque_canvas() -> None:
    card = {"id": "card", "kind": "graphic", "start_frame": 0, "end_frame": 30}
    with pytest.raises(AssemblyError, match=r"frames 0-29"):
        require_voiceover_picture(30, [card], graphic_coverage={"card": set()})
    require_voiceover_picture(30, [card], graphic_coverage={"card": set(range(30))})


def test_pilot_picture_checks_only_requested_frames() -> None:
    insert = {"kind": "insert", "start_frame": 0, "end_frame": 30}
    require_voiceover_picture(90, [insert], frames=(0, 29))
    with pytest.raises(AssemblyError, match=r"frames 30-40"):
        require_voiceover_picture(90, [insert], frames=(20, 40))
    with pytest.raises(AssemblyError, match=r"frames 30-89"):
        require_voiceover_picture(90, [insert])


def test_pilot_into_end_hold_uses_last_content_frame() -> None:
    covered = {"kind": "insert", "start_frame": 0, "end_frame": 120}
    require_voiceover_picture(120, [covered], frames=(115, 125))
    short = {"kind": "insert", "start_frame": 0, "end_frame": 118}
    with pytest.raises(AssemblyError, match=r"frames 118-125"):
        require_voiceover_picture(120, [short], frames=(115, 125))


def test_rendered_graphic_requires_every_pixel_opaque(tmp_path: Path) -> None:
    class Snapshot:
        class Timeline:
            total_frames = 2
        timeline = Timeline()
        title_layers = []

    snapshot = Snapshot()
    for name, expression in (
        ("partial", "color=c=white@0:s=1080x1920:r=30,format=rgba,"
         "drawbox=x=0:y=0:w=100:h=100:c=white@1:t=fill"),
        ("opaque", "color=c=white@1:s=1080x1920:r=30,format=rgba"),
    ):
        target = tmp_path / f"{name}.mov"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        expression, "-frames:v", "2", "-c:v", "qtrle", "-pix_fmt", "argb",
                        str(target)], check=True)
        snapshot.title_layers = [{"id": name, "kind": "graphic", "path": target.name,
                                  "x": 0, "y": 0, "width": 1080, "height": 1920,
                                  "start_frame": 0, "frames": 2}]
        coverage = _graphic_picture_frames(tmp_path, snapshot, None)
        assert coverage.get(name, set()) == (set(range(2)) if name == "opaque" else set())


def test_standin_check_skip_keeps_audio_failure() -> None:
    report = {"status": "fail", "summary": {"status": "fail"},
              "checks": {"picture.black": {"status": "fail"},
                         "join.words": {"status": "fail"}},
              "findings": [{"check": "picture.black", "severity": "fail"},
                           {"check": "join.words", "severity": "fail"}]}
    filtered = skip_standin_picture_checks(report)
    assert filtered["status"] == "fail"
    assert filtered["findings"] == [{"check": "join.words", "severity": "fail"}]
    assert report["checks"]["picture.black"]["status"] == "fail"


def test_standin_check_skip_keeps_failure_without_finding() -> None:
    report = {"status": "fail", "summary": {"status": "fail"},
              "checks": {"picture.black": {"status": "fail"},
                         "join.words": {"status": "fail"}},
              "findings": [{"check": "picture.black", "severity": "fail"}]}
    filtered = skip_standin_picture_checks(report)
    assert filtered["status"] == filtered["summary"]["status"] == "fail"
    assert filtered["checks"]["join.words"]["status"] == "fail"


def test_pickup_ingest_transcribes_only_new_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_media(tmp_path / "source.mkv")
    memo = tmp_path / "memo.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", str(memo)],
                   check=True)
    root = tmp_path / "voice-project"
    backend = FakeBackend()
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_args: backend)
    init_project(root, takes=[], voiceovers=[memo])
    ingest(root)
    backend.calls.clear()
    pickup_source = make_media(tmp_path / "pickup-source.mkv", seconds=3)
    pickup = tmp_path / "pickup.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(pickup_source), "-vn", str(pickup)],
                   check=True)
    init_project(root, takes=[], voiceovers=[pickup])
    ingest(root)
    assert [[file.name for file in call] for call in backend.calls] == [["t2.wav"]]


@pytest.mark.render
def test_voiceover_final_requires_and_accepts_complete_broll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_media(tmp_path / "source.mkv", seconds=3)
    memo = tmp_path / "memo.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn", str(memo)],
                   check=True)
    root = tmp_path / "voice-project"
    backend = FakeBackend()
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_args: backend)
    init_project(root, takes=[], voiceovers=[memo])
    project = json.loads((root / "project.json").read_bytes())
    project["music"]["mode"] = "none"
    project["audio"] = {"mode": "preserve"}
    (root / "project.json").write_text(json.dumps(project), encoding="utf-8")
    ingest(root)
    init_project(root, takes=[], voiceovers=[memo])
    ingest(root)
    write_edit(root, [time_segment("s1", "t1", 0, 2), time_segment("s2", "t2", 0, 2)])
    real_cut_verify = cut_stage.verify
    monkeypatch.setattr(cut_stage, "verify", lambda **kwargs:
                        real_cut_verify(**{**kwargs, "asr": "none"}))
    assert cut_stage.cut(root)["status"] == "verified"
    captions(root)
    graphics(root)
    music(root)
    with pytest.raises(AssemblyError, match=r"frames 0-119"):
        assemble_stage.assemble(root)
    broll = synth.calib_source(root / "broll.mov", seconds=4)
    fragment = root / "broll.json"
    fragment.write_text(json.dumps({"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-cover", "kind": "insert", "start": {"time": 0},
        "end": {"time": 65 / 30}, "media": {"path": broll.name,
                                     "sha256": receipt.sha256_file(broll)},
    }]}), encoding="utf-8")
    add_timeline(root, fragment, lane="broll").require_valid()
    captions(root)
    graphics(root)
    music(root)
    pilot = assemble_stage.assemble(root, frames=(0, 64))
    assert pilot["status"] == "verified" and pilot["frames"] == [0, 64]
    assert (root / pilot["out"]).is_file()
    with pytest.raises(AssemblyError, match=r"frames 65-119"):
        assemble_stage.assemble(root, frames=(0, 119))
    with pytest.raises(AssemblyError, match=r"frames 65-119"):
        assemble_stage.assemble(root)
    fragment.write_text(json.dumps({"schema": "yap-studio/timeline@1", "elements": [{
        "id": "broll-cover", "kind": "insert", "start": {"time": 0},
        "end": {"time": 4}, "media": {"path": broll.name,
                                     "sha256": receipt.sha256_file(broll)},
    }]}), encoding="utf-8")
    add_timeline(root, fragment, lane="broll").require_valid()
    captions(root)
    graphics(root)
    music(root)
    real_final_verify = assemble_stage.verify
    monkeypatch.setattr(assemble_stage, "verify", lambda *args, **kwargs:
                        real_final_verify(*args, **{**kwargs, "asr": "none"}))
    # Bright calibration B-roll gets a halo and may warn without blocking coverage delivery.
    assert assemble_stage.assemble(root)["status"] == "verified"
