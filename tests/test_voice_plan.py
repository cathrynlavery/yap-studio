"""Free proposals refuse unsafe word and picture ranges."""

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_cut import time_segment, write_edit, write_json
from voice_helpers import calls, linked

from yap_studio import ffmpeg, receipt
from yap_studio.errors import YapError
from yap_studio.voice import list_fixes, propose
from yap_studio.voice.plan import require_picture, validate_words


def test_opaque_graphic_frames_rejects_incomplete_alpha(tmp_path, monkeypatch):
    from yap_studio.stages.assemble import AssemblyError, opaque_graphic_frames

    closed = []

    def alpha():
        try:
            yield b"\xff"
        finally:
            closed.append(True)

    monkeypatch.setattr(ffmpeg, "run_stream", lambda *_, **__: alpha())
    layer = {"id": "agent-cover", "kind": "graphic", "path": "cover.mov", "x": 0, "y": 0,
             "width": 1080, "height": 1920, "start_frame": 60, "frames": 1}
    with pytest.raises(AssemblyError, match="agent-cover: graphic alpha decode is incomplete"):
        opaque_graphic_frames(tmp_path, [layer])
    assert closed == [True]


@pytest.fixture(scope="module")
def opaque_graphic(tmp_path_factory):
    target = tmp_path_factory.mktemp("voice-picture") / "opaque.mov"
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=c=white@1:s=1080x1920:r=30,format=rgba", "-frames:v", "60",
                "-c:v", "qtrle", "-pix_fmt", "argb", str(target)], timeout_s=120)
    return target


def _mouth_graphic(root, monkeypatch, opaque_graphic, *, layer="front", full=True,
                   rendered=True, current=True):
    from yap_studio.project.receipts import projection_sha256
    from yap_studio.project.status import ProjectContext
    from yap_studio.stages import graphics as graphics_stage
    from yap_studio.stages.timeline import ResolvedElement

    project = json.loads((root / "project.json").read_text())
    project["takes"][0]["voiceover"] = False
    write_json(root / "project.json", project)
    element = ResolvedElement("agent-cover", "graphic", "agent", 60, 120, layer, {})
    timeline = SimpleNamespace(elements=[element], total_frames=360)
    monkeypatch.setattr(ProjectContext, "resolved_timeline", lambda _: timeline)
    write_json(root / "timeline/agent.json", {})
    inputs = {"timeline": projection_sha256(element.to_dict())}
    placement = {"id": element.id, "kind": "graphic", "layer": layer,
                 "path": "graphics/agent-cover.mov", "start_frame": 60, "frames": 60,
                 "x": 0, "y": 0, "width": 1080 if full else 200, "height": 1920 if full else 200}
    monkeypatch.setattr(graphics_stage, "graphics_snapshot", lambda *_, **__: SimpleNamespace(
        inputs=inputs, graphics=[SimpleNamespace(placement=lambda: placement)],
    ))
    if rendered:
        target = root / placement["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(opaque_graphic, target)
        write_json(root / "graphics/receipt.json", {
            "schema": "yap-studio/stage-receipt@1", "status": "verified",
            "inputs": inputs if current else {"timeline": "old"}, "elements": [placement],
            "outputs": [{"path": placement["path"], "sha256": receipt.sha256_file(target)}],
        })
    return element, placement


@pytest.mark.parametrize("action", ["propose", "make", "assemble"])
@pytest.mark.parametrize("layer,full,allowed", [("behind", True, False), ("front", False, False),
                                              ("front", True, True)])
def test_voice_picture_requires_front_opaque_canvas(
    voice_project, fake_voice, monkeypatch, opaque_graphic, action, layer, full, allowed,
):
    from yap_studio.stages import assemble as assemble_stage
    from yap_studio.stages import cut as cut_stage
    from yap_studio.voice import make, store

    fix = None
    if action != "propose":
        linked(voice_project, monkeypatch)
        fix = propose(voice_project, "t1:2", "t1:3", "cold plunge")
    element, placement = _mouth_graphic(
        voice_project, monkeypatch, opaque_graphic, layer=layer, full=full,
    )
    if action == "assemble":
        context = cut_stage.resolve_cut(voice_project).context
        snapshot = SimpleNamespace(sfx={"kit": "none"}, title_layers=[placement],
                                   timeline=SimpleNamespace(elements=[element], total_frames=360))
        fix["patch"] = {"start_sample": round(fix["source"]["start_s"] * 48000),
                        "samples": round((fix["source"]["end_s"]
                                          - fix["source"]["start_s"]) * 48000)}
        monkeypatch.setattr(store, "approved_fixes", lambda _: [fix])
        monkeypatch.setattr(store, "patches_for_plan", lambda *_, **__: [(fix, None)])
        monkeypatch.setattr(store, "projection", lambda *_, **__: {"consent": None, "approved": []})
        read = context.json
        monkeypatch.setattr(context, "json", lambda name, *args, **kwargs: (
            {"inputs": {}} if name == "cut/receipt.json" else read(name, *args, **kwargs)))
        monkeypatch.setattr(assemble_stage, "ProjectContext", lambda _: context)
        monkeypatch.setattr(assemble_stage, "validate_voice_chain", lambda *_: None)
        monkeypatch.setattr(assemble_stage, "assembly_snapshot", lambda *_, **__: snapshot)
        monkeypatch.setattr(assemble_stage, "_pilot_phrases", lambda _: (b"words", "hash"))
        monkeypatch.setattr(assemble_stage, "_pilot_window", lambda *_: None)
        monkeypatch.setattr(assemble_stage, "_pilot_assemble",
                            lambda *_, **__: {"status": "verified"})

    def run():
        if action == "propose":
            return propose(voice_project, "t1:2", "t1:3", "cold plunge")
        if action == "make":
            return make(voice_project, ["f1"], yes=True, max_credits=20)
        return assemble_stage._assemble_impl(voice_project, frames=(60, 119))

    if allowed:
        run()
    else:
        with pytest.raises(YapError, match="Your face is on screen") as caught:
            run()
        assert caught.value.code == "voice_mouth_on_screen"
        assert "render the graphics first" not in str(caught.value)
        assert not any(argv[0] == "text-to-speech" for argv in calls(fake_voice))


@pytest.mark.parametrize("rendered,current", [(False, True), (True, False)])
@pytest.mark.parametrize("action", ["propose", "make"])
def test_unrendered_or_stale_front_graphic_refuses_before_payment(
    voice_project, fake_voice, monkeypatch, opaque_graphic, rendered, current, action,
):
    from yap_studio.voice import make

    if action == "make":
        linked(voice_project, monkeypatch)
        propose(voice_project, "t1:2", "t1:3", "cold plunge")
    _mouth_graphic(voice_project, monkeypatch, opaque_graphic, rendered=rendered, current=current)
    with pytest.raises(YapError, match="or render the graphics first") as caught:
        if action == "make":
            make(voice_project, ["f1"], yes=True, max_credits=20)
        else:
            propose(voice_project, "t1:2", "t1:3", "cold plunge")
    assert caught.value.code == "voice_mouth_on_screen"
    assert not any(argv[0] == "text-to-speech" for argv in calls(fake_voice))


@pytest.mark.parametrize("layer,full,frames", [("behind", True, 60), ("front", False, 60),
                                             ("front", True, 10)])
def test_render_hint_needs_a_front_canvas_over_the_whole_moment(
    voice_project, fake_voice, monkeypatch, opaque_graphic, layer, full, frames,
):
    _, placement = _mouth_graphic(voice_project, monkeypatch, opaque_graphic, layer=layer,
                                 full=full, rendered=False)
    placement["frames"] = frames
    with pytest.raises(YapError, match="Your face is on screen") as caught:
        propose(voice_project, "t1:2", "t1:3", "cold plunge")
    assert "render the graphics first" not in str(caught.value)
    assert not any(argv[0] == "text-to-speech" for argv in calls(fake_voice))


@pytest.mark.parametrize("action", ["propose", "make"])
def test_front_canvas_with_transparent_pixels_refuses_before_payment(
    voice_project, fake_voice, monkeypatch, opaque_graphic, action,
):
    from yap_studio.voice import make

    if action == "make":
        linked(voice_project, monkeypatch)
        propose(voice_project, "t1:2", "t1:3", "cold plunge")
    _mouth_graphic(voice_project, monkeypatch, opaque_graphic)
    # Keep the real alpha decoder and make one canvas pixel transparent in every frame.
    target = voice_project / "graphics/agent-cover.mov"
    target.unlink()
    ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=c=white@1:s=1080x1920:r=30,format=rgba,"
                "drawbox=x=0:y=0:w=1:h=1:c=white@0:t=fill:replace=1", "-frames:v", "60",
                "-c:v", "qtrle", "-pix_fmt", "argb", str(target)], timeout_s=120)
    record = json.loads((voice_project / "graphics/receipt.json").read_text())
    record["outputs"][0]["sha256"] = receipt.sha256_file(target)
    write_json(voice_project / "graphics/receipt.json", record)
    with pytest.raises(YapError, match="Your face is on screen") as caught:
        if action == "make":
            make(voice_project, ["f1"], yes=True, max_credits=20)
        else:
            propose(voice_project, "t1:2", "t1:3", "cold plunge")
    assert "render the graphics first" not in str(caught.value)
    assert not any(argv[0] == "text-to-speech" for argv in calls(fake_voice))


def test_propose(voice_project):
    fix = propose(voice_project, "t1:2", "t1:3", "cold plunge", why="Misspoke")
    assert fix["id"] == "f1" and fix["status"] == "proposed"
    assert fix["said"] == "coal flunge" and fix["words"] == ["t1:2", "t1:3"]
    assert fix["estimate"]["characters"] == 11
    assert list_fixes(voice_project)["fixes"] == [fix]


@pytest.mark.parametrize("index,field", [(2, "start"), (3, "end")])
@pytest.mark.parametrize("delta", [-0.002, -0.001, 0, 0.001, 0.002])
def test_word_timing_tolerance(voice_project, index, field, delta):
    from yap_studio.stages.cut import resolve_cut

    fix = propose(voice_project, "t1:2", "t1:3", "cold plunge")
    plan = resolve_cut(voice_project)
    snapshot = plan.source_words["t1"]
    words = list(snapshot.words)
    words[index] = replace(words[index], **{field: getattr(words[index], field) + delta})
    plan.source_words["t1"] = replace(snapshot, words=tuple(words))
    if abs(delta) <= 0.001:
        validate_words(plan, fix)
    else:
        with pytest.raises(YapError, match="The words moved since this fix was made") as caught:
            validate_words(plan, fix)
        assert caught.value.code == "voice_words_changed"


@pytest.mark.parametrize(
    "first,last,say,code",
    [
        ("t1:4", "t1:2", "water", "voice_words_not_contiguous"),
        ("t1:0", "t1:6", "water", "voice_too_many_words"),
        ("t1:2", "t1:3", "a" * 61, "voice_text_too_long"),
        ("t1:2", "t1:3", "one two three four five six seven eight nine", "voice_text_too_long"),
    ],
)
def test_proposal_limits(voice_project, first, last, say, code):
    with pytest.raises(YapError) as caught:
        propose(voice_project, first, last, say)
    assert caught.value.code == code


def test_dropped_and_overlap(voice_project):
    write_edit(voice_project, [time_segment("s1", "t1", 5, 12)])
    with pytest.raises(YapError, match="kept"):
        propose(voice_project, "t1:2", "t1:3", "cold plunge")
    write_edit(voice_project, [time_segment("s1", "t1", 0, 12)])
    propose(voice_project, "t1:2", "t1:3", "cold plunge")
    with pytest.raises(YapError, match="overlap"):
        propose(voice_project, "t1:3", "t1:4", "plunge project")


def test_mouth_times_and_full_frame(voice_project):
    data = json.loads((voice_project / "project.json").read_text())
    data["takes"][0]["voiceover"] = False
    write_json(voice_project / "project.json", data)
    with pytest.raises(YapError, match=r"face is on screen at 0:02.1 to 0:03.9"):
        propose(voice_project, "t1:2", "t1:3", "cold plunge")
    fix = {"id": "f1", "take": "t1", "source": {"start_s": 2.2, "end_s": 3.8}}
    from yap_studio.stages.cut import resolve_cut

    plan = resolve_cut(voice_project)
    insert = {
        "id": "b1",
        "kind": "insert",
        "start_frame": 60,
        "end_frame": 120,
        "media": {"fit": "cover"},
    }
    assert require_picture(plan, fix, elements=[insert])["kind"] == "covered"
    insert["media"]["fit"] = "split"
    with pytest.raises(YapError) as caught:
        require_picture(plan, fix, elements=[insert])
    assert caught.value.code == "voice_mouth_on_screen"


@pytest.mark.parametrize("fit,allowed", [("cover", True), ("split", False)])
@pytest.mark.parametrize("layer", ["front", "behind"])
def test_propose_with_resolved_insert(voice_project, monkeypatch, fit, allowed, layer):
    from types import SimpleNamespace

    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.timeline import ResolvedElement

    data = json.loads((voice_project / "project.json").read_text())
    data["takes"][0]["voiceover"] = False
    write_json(voice_project / "project.json", data)
    write_json(voice_project / "timeline/broll.json", {})
    insert = ResolvedElement("b1", "insert", "broll", 60, 120, layer, {"media": {"fit": fit}})
    monkeypatch.setattr(
        ProjectContext, "resolved_timeline", lambda _: SimpleNamespace(elements=[insert])
    )
    if allowed and layer == "front":
        fix = propose(voice_project, "t1:2", "t1:3", "cold plunge")
        assert fix["picture"]["kind"] == "covered"
    else:
        with pytest.raises(YapError) as caught:
            propose(voice_project, "t1:2", "t1:3", "cold plunge")
        assert caught.value.code == "voice_mouth_on_screen"


def test_broken_store(voice_project):
    write_json(
        voice_project / "voice/fixes.json",
        {"schema": "yap-studio/voice-fixes@1", "fixes": [{"id": "f1"}]},
    )
    with pytest.raises(YapError, match="fixes.json"):
        list_fixes(voice_project)


def test_graphic_requires_opaque_frames_at_assembly(voice_project):
    data = json.loads((voice_project / "project.json").read_text())
    data["takes"][0]["voiceover"] = False
    write_json(voice_project / "project.json", data)
    from yap_studio.stages.cut import resolve_cut

    plan = resolve_cut(voice_project)
    fix = {"id": "f1", "take": "t1", "source": {"start_s": 2.2, "end_s": 3.8}}
    graphic = {"id": "g1", "kind": "graphic", "start_frame": 60, "end_frame": 120}
    with pytest.raises(YapError, match="Your face is on screen"):
        require_picture(plan, fix, elements=[graphic])
    with pytest.raises(YapError) as caught:
        require_picture(plan, fix, elements=[graphic], graphic_coverage={"g1": {60}})
    assert caught.value.code == "voice_mouth_on_screen"
    assert (
        require_picture(
            plan, fix, elements=[graphic], graphic_coverage={"g1": set(range(60, 120))}
        )["kind"]
        == "covered"
    )
