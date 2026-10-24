"""Synthetic inspection exports, verified layer inputs, and Studio command contracts."""

from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from yap_studio import cli, ffmpeg, receipt
from yap_studio.errors import InputError, OutputPathError
from yap_studio.project.status import ProjectContext, cut_inputs, graphics_inputs, ingest_inputs
from yap_studio.render import hyperframes
from yap_studio.stages.captions import caption_snapshot
from yap_studio.stages.graphics import graphics_snapshot
from yap_studio.studio import export as studio

pytestmark = pytest.mark.usefixtures("_no_real_node")

_FAKE_RENDERER = r'''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert all(os.environ[key] == "1" for key in (
    "HYPERFRAMES_NO_TELEMETRY", "HYPERFRAMES_NO_UPDATE_CHECK",
    "HYPERFRAMES_NO_AUTO_INSTALL", "HYPERFRAMES_SKIP_SKILLS"))
assert sys.stdin.buffer.read() == b""
with Path(os.environ["YAP_TEST_STUDIO_LOG"]).open("a") as stream:
    stream.write(json.dumps(args) + "\n")
if args[0] == "lint":
    assert args[1] == "--json"
    assert (Path(args[-1]) / "index.html").is_file()
    print(os.environ.get("YAP_TEST_STUDIO_LINT", json.dumps({
        "ok": True, "errorCount": 0, "warningCount": 1,
        "findings": [{"severity": "warning", "code": "inspection-note"}]})))
elif args[0] == "catch-up":
    assert args[1] == "--json"
    print(os.environ.get("YAP_TEST_STUDIO_CATCH_UP", json.dumps({"turns": [], "files": []})))
elif "--stop" in args:
    print("Stopped")
else:
    assert args[0:2] == ["preview", "--background"]
    port = args[args.index("--port") + 1]
    print("Server http://127.0.0.1:" + port)
    print(os.environ.get("YAP_TEST_STUDIO_URL",
          "\x1b[32mStudio http://127.0.0.1:" + port + "/#project/studio\x1b[0m"))
sys.exit(int(os.environ.get("YAP_TEST_STUDIO_EXIT", "0")))
'''


def write(target: Path, data: Any) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return target


def context(root: Path) -> ProjectContext:
    return ProjectContext(root)


def stage_receipt(root: Path, stage: str, inputs: dict, outputs: list[str], **extra: Any) -> None:
    write(root / stage / "receipt.json", {
        "schema": "yap-studio/stage-receipt@1", "status": "verified", "inputs": inputs,
        "outputs": [{"path": name, "sha256": receipt.sha256_file(root / name)} for name in outputs],
        **extra,
    })


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build exact cut contracts with split story spans and a second source take."""
    from test_stage_captions import _disable_face_helper

    _disable_face_helper(monkeypatch)
    root = tmp_path / "sample"
    root.mkdir()
    for take in ("t1", "t2"):
        (root / f"{take}.mov").write_bytes(take.encode())
    write(root / "project.json", {
        "schema": "yap-studio/project@1", "name": "sample-inspection",
        "style": "style.json", "asr": {"engine": "none"},
        "takes": [{"id": take, "path": f"{take}.mov"} for take in ("t1", "t2")],
    })
    write(root / "style.json", {
        "schema_version": 1, "name": "Inspection style", "captions": {"size_px": 48},
        "hook": {"style": "quiet_serif", "size_px": 48},
    })
    for take in ("t1", "t2"):
        write(root / f"ingest/{take}/words.json", {
            "schema": "yap-studio/words@1", "take": take,
            "take_sha256": receipt.sha256_file(root / f"{take}.mov"), "time_base": "source",
            "audio_offset_s": 0, "engine": "none", "model": None,
            "transcript_sha256": "1" * 64,
            "words": [{"id": f"{take}:0", "text": "Small <b> & steps.",
                       "start": 0.4 if take == "t1" else 0.2,
                       "end": 0.5 if take == "t1" else 0.3}],
        })
    stage_receipt(root, "ingest", ingest_inputs(context(root)),
                  [f"ingest/{take}/words.json" for take in ("t1", "t2")])
    hashes = {take: receipt.sha256_file(root / f"ingest/{take}/words.json")
              for take in ("t1", "t2")}
    write(root / "edit.json", {
        "schema": "yap-studio/edit@1", "words_sha256": hashes, "end_hold_s": 0.5,
        "segments": [
            {"id": "s1", "take": "t1", "from": {"time": 1 / 3}, "to": {"time": 4 / 3}},
            {"id": "s2", "take": "t2", "from": {"time": 0.1}, "to": {"time": 73 / 30}},
        ],
    })
    write(root / "cut/cut-words.json", {
        "schema": "yap-studio/cut-words@1", "time_base": "output", "words_sha256": hashes,
        "words": [
            {"id": "t1:0", "segment": "s1", "text": "Small <b> & steps.",
             "start": 0.066, "end": 0.166},
            {"id": "t2:0", "segment": "s2", "text": "More words.",
             "start": 0.766, "end": 0.866},
        ],
    })
    rows = [("s1", "t1", "1/3", "2/3", 0, 10),
            ("s1", "t1", "1", "4/3", 10, 10),
            ("s2", "t2", "1/10", "73/30", 20, 70)]
    write(root / "cut/spans.json", {
        "sources": {take: f"../{take}.mov" for take in ("t1", "t2")}, "fps": "30/1",
        "segments": [{"source": take, "start": first, "end": last}
                     for _, take, first, last, _, _ in rows],
    })
    (root / "cut/cut.mp4").write_bytes(b"verified synthetic cut")
    stage_receipt(root, "cut", cut_inputs(context(root)),
                  ["cut/cut.mp4", "cut/spans.json", "cut/cut-words.json"], plan={
                      "total_frames": 90,
                      "pieces": [{"segment_id": identifier, "take": take, "src_t0": first,
                                  "src_t1": last, "out_f0": start, "n_frames": count}
                                 for identifier, take, first, last, start, count in rows],
                  })
    return root


def add_layers(root: Path) -> tuple[Any, Any]:
    """Create receipts through the existing stage snapshots, without rendering."""
    write(root / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1",
        "words_sha256": {take: receipt.sha256_file(root / f"ingest/{take}/words.json")
                         for take in ("t1", "t2")},
        "elements": [
            {"id": "agent-title", "kind": "title", "start": {"time": 0.2}, "hold_s": 0.4,
             "lines": ['Small <b> & "steps"'], "accent": "steps"},
            {"id": "agent-tag", "kind": "graphic", "start": {
                "word": "t2:0", "text": "Small <b> & steps.", "edge": "start", "segment": "s2",
                "offset_s": 0.134}, "hold_s": 1,
             "template": "text_tag", "slots": {"text": 'A <b> & "tag"'}},
        ],
    })
    view = context(root)
    caption = caption_snapshot(view)
    graphic = graphics_snapshot(view)
    (root / "captions").mkdir()
    (root / "captions/captions.mov").write_bytes(b"caption layer")
    stage_receipt(root, "captions", caption.inputs, ["captions/captions.mov"],
                  layer=caption.plan.placement("captions/captions.mov"))
    (root / "graphics").mkdir()
    for layer in (*graphic.titles, *graphic.graphics):
        (root / f"graphics/{layer.id}.mov").write_bytes(layer.id.encode())
    stage_receipt(root, "graphics", graphic.inputs,
                  [f"graphics/{layer.id}.mov" for layer in (*graphic.titles, *graphic.graphics)],
                  elements=[layer.placement() for layer in (*graphic.titles, *graphic.graphics)])
    return caption, graphic


@pytest.fixture
def fake_renderer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = tmp_path / "hyperframes"
    executable.write_text(f"#!{sys.executable}\n" + _FAKE_RENDERER, encoding="utf-8")
    executable.chmod(0o755)
    install = hyperframes.RendererInstall(hyperframes.HYPERFRAMES_VERSION, tmp_path,
                                          executable, None, "22.0.0")
    monkeypatch.setattr(hyperframes, "ensure_installed", lambda: install)
    gsap = tmp_path / hyperframes.GSAP_SCRIPT
    gsap.write_text("/* local dependency */", encoding="utf-8")
    monkeypatch.setattr(hyperframes, "gsap_file", lambda _install: gsap)
    log = tmp_path / "commands.jsonl"
    monkeypatch.setenv("YAP_TEST_STUDIO_LOG", str(log))
    return log


class Tags(HTMLParser):
    def __init__(self, target: Path) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.feed(target.read_text(encoding="utf-8"))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))

    def by_id(self, identifier: str) -> dict[str, Any]:
        return next(attrs for _, attrs in self.elements if attrs.get("id") == identifier)


def summary(target: Path) -> dict:
    return json.loads((target / "yap-export.json").read_bytes())


def files(root: Path) -> dict[str, tuple[bytes | str, int]]:
    """Capture file bytes and metadata, including bookkeeping and symlink destinations."""
    return {str(item.relative_to(root)): (
        str(item.readlink()) if item.is_symlink() else item.read_bytes(), item.lstat().st_mtime_ns,
    ) for item in root.rglob("*") if item.is_file() or item.is_symlink()}


def stable_files(root: Path) -> dict[str, tuple[bytes | str, int]]:
    """Exclude only the lock and identity cache the shared resolver may update."""
    return {name: data for name, data in files(root).items()
            if name not in {".yap/lock", ".yap/identity.json"}}


def commands(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_export_structure_and_exact_story_mapping(project: Path, fake_renderer: Path) -> None:
    before = stable_files(project)
    target = studio.export_project(project)
    assert target == project / ".yap/studio"
    tags = Tags(target / "index.html")
    root = tags.by_id("root")
    assert root["data-composition-id"] == "root" and "data-no-timeline" in root
    assert (root["data-width"], root["data-height"], root["data-fps"]) == ("1080", "1920", "30")
    assert float(root["data-duration"]) == 3.5
    videos = [attrs for tag, attrs in tags.elements if tag == "video"]
    assert len(videos) == 2
    for clip, take, first, last, start, duration in zip(
        videos, ("t1", "t2"), ("1/3", "1/10"), ("4/3", "73/30"),
        (Fraction(0), Fraction(2, 3)), (Fraction(2, 3), Fraction(17, 6)), strict=True,
    ):
        assert clip["src"] == "media/cut.mp4" and clip["data-track-index"] == "0"
        assert clip["data-has-audio"] == "true" and "muted" not in clip
        assert float(clip["data-start"]) == float(start)
        assert float(clip["data-media-start"]) == float(start)
        assert float(clip["data-duration"]) == float(duration - Fraction(1, 1000000))
        assert (clip["data-yap-take-id"], clip["data-yap-source-in"],
                clip["data-yap-source-out"]) == (take, first, last)
    assert [clip["data-yap-segment-index"] for clip in videos] == ["0", "1"]
    assert [clip["data-yap-segment-id"] for clip in videos] == ["s1", "s2"]
    assert json.loads(videos[0]["data-yap-source-spans"]) == [
        {"in": "1/3", "out": "2/3"}, {"in": "1", "out": "4/3"},
    ]
    assert float(videos[0]["data-start"]) + float(videos[0]["data-duration"]) < float(
        videos[1]["data-start"]
    )
    assert (target / "media/cut.mp4").is_symlink()
    assert (target / "media/cut.mp4").resolve() == project / "cut/cut.mp4"
    report = summary(target)
    assert report["schema"] == "yap-studio/studio-export@1"
    assert report["project_name"] == "sample-inspection" and len(report["timeline_hash"]) == 64
    assert report["lint"]["findings"][0]["code"] == "inspection-note"
    assert report["stages"]["cut"]["included"]
    for stage in ("captions", "graphics"):
        assert report["stages"][stage] == {"included": False, "reason": "no receipt"}
    assert "Nothing is read back from Studio into the project." in (
        target / "index.html"
    ).read_text(encoding="utf-8")
    after = {name: data for name, data in stable_files(project).items()
             if not name.startswith(".yap/studio/")}
    assert before == after
    assert {item.name for item in target.parent.iterdir()} == {"studio", "lock", "identity.json"}
    assert commands(fake_renderer)[0][:2] == ["lint", "--json"]


def test_layer_hosts_match_receipts_and_exact_variables(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    caption, graphic = add_layers(project)

    def unexpected(*args, **kwargs):
        pytest.fail("exporting existing receipts must not probe or measure media")

    monkeypatch.setattr(ffmpeg, "probe", unexpected)
    monkeypatch.setattr("yap_studio.faces.project.cut_faces", unexpected)
    before = stable_files(project)
    target = studio.export_project(project)
    tags = Tags(target / "index.html")
    layers = [("captions", caption.plan, caption.plan.placement("captions/captions.mov")),
              *((layer.id, layer.plan, layer.placement())
                for layer in (*graphic.titles, *graphic.graphics))]
    tracks = []
    for identifier, plan, placement in layers:
        host = tags.by_id(identifier)
        assert host["data-composition-src"] == f"layers/{identifier}/index.html"
        assert json.loads(host["data-variable-values"]) == plan.variables
        assert float(host["data-start"]) == placement["start_frame"] / 30
        assert float(host["data-duration"]) == placement["frames"] / 30
        assert host["data-yap-start-frame"] == str(placement["start_frame"])
        assert host["data-yap-frames"] == str(placement["frames"])
        for name in ("x", "y", "width", "height"):
            css = {"x": "left", "y": "top"}.get(name, name)
            assert f"{css}:{placement[name]}px" in host["style"]
        assert host["data-yap-take-id"] in {"t1", "t2"}
        assert json.loads(host["data-yap-segments"])
        layer = target / f"layers/{identifier}"
        assert (layer / "index.html").is_file() and (layer / "fonts").is_dir()
        assert not (layer / plan.template).exists()
        tracks.append(host["data-track-index"])
    assert len(set(tracks)) == 3 and "0" not in tracks
    assert (target / "layers/agent-tag" / hyperframes.GSAP_SCRIPT).is_file()
    raw = (target / "index.html").read_text(encoding="utf-8")
    assert 'A <b> & "tag"' not in raw and "&lt;b&gt;" in raw and "&quot;" in raw
    assert all(item["included"] for item in summary(target)["stages"].values())
    assert before == {name: data for name, data in stable_files(project).items()
                      if not name.startswith(".yap/studio/")}


def test_export_omits_verified_camera_captions(project: Path, fake_renderer: Path) -> None:
    from yap_studio.stages.captions_3d import CameraSnapshot

    _, graphic = add_layers(project)
    data = json.loads((project / "style.json").read_bytes())
    data["captions"]["camera_3d"] = True
    write(project / "style.json", data)
    view = context(project)
    caption = caption_snapshot(view)
    assert isinstance(caption, CameraSnapshot)
    (project / "captions/camera3d.mp4").write_bytes(b"camera picture")
    write(project / "captions/camera3d.json", {
        "schema": "yap-studio/camera3d-pilot@1", "frames": 90,
    })
    stage_receipt(project, "captions", caption.inputs, [
        "captions/captions.mov", "captions/camera3d.mp4", "captions/camera3d.json",
    ], picture="captions/camera3d.mp4", layer={
        "path": "captions/captions.mov", "x": 0, "y": 0, "width": 1080, "height": 1920,
        "start_frame": 0, "frames": 90,
    })
    placements = [item.placement() for item in (*graphic.titles, *graphic.graphics)]
    graphic = graphics_snapshot(context(project), placements=placements)
    stage_receipt(project, "graphics", graphic.inputs,
                  [f"graphics/{item['id']}.mov" for item in placements], elements=placements)
    before = stable_files(project)
    target = studio.export_project(project)
    assert summary(target)["stages"]["captions"] == {
        "included": False,
        "reason": "3D camera captions are rendered video, so Studio shows the cut without them",
    }
    assert summary(target)["stages"]["graphics"]["included"]
    assert not (target / "layers/captions").exists()
    for placement in placements:
        assert (target / "layers" / placement["id"] / "index.html").is_file()
    assert (target / "media/cut.mp4").resolve() == project / "cut/cut.mp4"
    assert before == {name: data for name, data in stable_files(project).items()
                      if not name.startswith(".yap/studio/")}


def test_export_omits_graphics_after_cut_identity_changes(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_layers(project)
    (project / "cut/cut.mp4").write_bytes(b"a different verified synthetic cut")
    record = json.loads((project / "cut/receipt.json").read_bytes())
    stage_receipt(project, "cut", record["inputs"],
                  [item["path"] for item in record["outputs"]], plan=record["plan"])

    def unexpected(*args, **kwargs):
        pytest.fail("stale receipts must be detected by identity without measuring media")

    monkeypatch.setattr("yap_studio.faces.project.cut_faces", unexpected)
    target = studio.export_project(project)
    assert not summary(target)["stages"]["graphics"]["included"]
    assert not (target / "layers/agent-title").exists()


@pytest.mark.parametrize("change", ["input", "output", "failed", "pending", "missing", "invalid"])
def test_stale_caption_receipt_is_omitted(
    project: Path, fake_renderer: Path, change: str,
) -> None:
    add_layers(project)
    target = project / "captions/receipt.json"
    record = json.loads(target.read_bytes())
    if change == "input":
        data = json.loads((project / "project.json").read_bytes())
        data["word_fixes"] = {"small": "Little"}
        write(project / "project.json", data)
    elif change == "output":
        (project / "captions/captions.mov").write_bytes(b"changed")
    elif change == "failed":
        write(target, {**record, "status": "failed"})
    elif change == "pending":
        write(project / "captions/receipt.pending.json", record)
    elif change == "missing":
        target.unlink()
    else:
        target.write_text("{invalid", encoding="utf-8")
    exported = studio.export_project(project)
    state = summary(exported)["stages"]["captions"]
    assert not state["included"] and state["reason"]
    assert not (exported / "layers/captions").exists()
    assert summary(exported)["stages"]["graphics"]["included"]


@pytest.mark.parametrize(("field", "fixes"), [
    ("word_fixes", {"small": "Little"}),
    ("word_text_fixes", {"t1:0": "Little"}),
])
def test_word_fixes_keep_plain_graphics_in_export(
    project: Path, fake_renderer: Path, field: str, fixes: dict[str, str],
) -> None:
    _, before = add_layers(project)
    data = json.loads((project / "project.json").read_bytes())
    data[field] = fixes
    write(project / "project.json", data)
    exported = studio.export_project(project)
    stages = summary(exported)["stages"]
    assert stages["graphics"]["included"]
    assert not stages["captions"]["included"]
    assert (exported / "layers/agent-title").is_dir()
    assert (exported / "layers/agent-tag").is_dir()
    assert graphics_inputs(context(project)) == before.inputs


@pytest.mark.parametrize("change", ["input", "output", "missing", "new", "placement", "invalid"])
def test_stale_graphics_receipt_is_omitted(
    project: Path, fake_renderer: Path, change: str,
) -> None:
    add_layers(project)
    if change == "input":
        target = project / "timeline/agent.json"
        data = json.loads(target.read_bytes())
        data["elements"][1]["slots"]["text"] = "Changed tag"
        write(target, data)
    elif change == "output":
        (project / "graphics/agent-title.mov").write_bytes(b"changed")
    elif change == "missing":
        (project / "graphics/receipt.json").unlink()
    elif change == "new":
        target = project / "timeline/agent.json"
        data = json.loads(target.read_bytes())
        data["elements"][1]["id"] = "agent-new-tag"
        write(target, data)
    else:
        target = project / "graphics/receipt.json"
        data = json.loads(target.read_bytes())
        if change == "placement":
            data["elements"][0]["start_frame"] += 1
        else:
            data["elements"] = {}
        write(target, data)
    exported = studio.export_project(project)
    assert not summary(exported)["stages"]["graphics"]["included"]
    assert summary(exported)["stages"]["captions"]["included"]


def test_verified_empty_graphics_stage_is_included(project: Path, fake_renderer: Path) -> None:
    stage_receipt(project, "graphics", graphics_snapshot(context(project)).inputs, [], elements=[])
    target = studio.export_project(project)
    assert summary(target)["stages"]["graphics"]["included"]
    assert not (target / "layers").exists()


def test_verified_empty_captions_are_included_without_a_layer(
    project: Path, fake_renderer: Path,
) -> None:
    from test_stage_captions import set_elements

    from yap_studio.stages.captions import captions

    set_elements(project, {"id": "agent-gap", "kind": "caption_gap",
                           "start": {"time": 0}, "hold_s": 3})
    assert captions(project)["layer"] is None
    target = studio.export_project(project)
    assert summary(target)["stages"]["captions"]["included"]
    assert not (target / "layers").exists()


def test_export_uses_shared_resolution_and_only_writes_allowed_bookkeeping(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_layers(project)
    for name in ("lock", "identity.json"):
        (project / ".yap" / name).unlink()
    before = files(project)
    resolve = ProjectContext.resolved_timeline
    calls = []

    def resolve_shared(self: ProjectContext, *, allow_violations: bool = False) -> Any:
        calls.append(self.project_dir)
        return resolve(self, allow_violations=allow_violations)

    monkeypatch.setattr(ProjectContext, "resolved_timeline", resolve_shared)
    target = studio.export_project(project)
    assert calls == [project, project]
    assert all(item["included"] for item in summary(target)["stages"].values())
    after = {name: data for name, data in files(project).items()
             if not name.startswith(".yap/studio/")}
    assert after.keys() - before.keys() == {".yap/lock", ".yap/identity.json"}
    assert before == {name: data for name, data in after.items() if name in before}


def test_rebuild_swaps_complete_export_and_removes_obsolete_files(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = studio.export_project(project)
    previous_inode = target.stat().st_ino
    (target / "obsolete.txt").write_text("old", encoding="utf-8")
    publish = studio._publish

    def inspect(work: Path, destination: Path) -> None:
        assert (destination / "obsolete.txt").is_file()
        assert (work / "index.html").is_file() and (work / "yap-export.json").is_file()
        publish(work, destination)
        assert destination.stat().st_ino != previous_inode
        assert not work.exists()
        assert not work.with_name(work.name + ".previous").exists()

    monkeypatch.setattr(studio, "_publish", inspect)
    assert studio.export_project(project) == target
    assert not (target / "obsolete.txt").exists()
    assert not any(item.name.startswith(".studio-") for item in target.parent.iterdir())


def test_publish_restores_previous_export_when_new_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, work = tmp_path / "studio", tmp_path / ".studio-build"
    target.mkdir()
    work.mkdir()
    (target / "index.html").write_text("previous export", encoding="utf-8")
    (work / "index.html").write_text("replacement export", encoding="utf-8")
    before = files(target)
    previous = work.with_name(work.name + ".previous")
    rename = os.rename
    calls = []

    def fail_second(source: Path, destination: Path) -> None:
        calls.append((source, destination))
        if source == work:
            assert not target.exists()
            assert files(previous) == before
            raise OSError("new export rename failed")
        rename(source, destination)

    monkeypatch.setattr(studio.os, "rename", fail_second)
    with pytest.raises(OSError, match="new export rename failed"):
        studio._publish(work, target)
    assert calls == [(target, previous), (work, target), (previous, target)]
    assert files(target) == before
    assert (work / "index.html").read_text(encoding="utf-8") == "replacement export"
    assert not previous.exists()


@pytest.mark.parametrize("exit_code", ["0", "1"])
def test_lint_findings_fail_export_and_preserve_previous_export(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch, exit_code: str,
) -> None:
    target = studio.export_project(project)
    before = stable_files(project)
    monkeypatch.setenv("YAP_TEST_STUDIO_LINT", json.dumps({
        "ok": False, "errorCount": 1, "findings": [{"severity": "error", "code": "bad-timing"}],
    }))
    monkeypatch.setenv("YAP_TEST_STUDIO_EXIT", exit_code)
    with pytest.raises(studio.StudioError, match="bad-timing") as caught:
        studio.export_project(project)
    assert caught.value.code == "studio_lint_failed"
    assert target.is_dir() and stable_files(project) == before


def test_publish_failure_preserves_previous_export(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    studio.export_project(project)
    before = stable_files(project)

    def fail(_work: Path, _target: Path) -> None:
        raise OSError("publication refused")

    monkeypatch.setattr(studio, "_publish", fail)
    with pytest.raises(studio.StudioError, match="publication refused"):
        studio.export_project(project)
    assert stable_files(project) == before


@pytest.mark.parametrize("change", ["cut", "source", "missing", "unowned"])
def test_export_requires_current_verified_cut(
    project: Path, fake_renderer: Path, change: str,
) -> None:
    if change == "cut":
        (project / "cut/cut.mp4").write_bytes(b"changed")
    elif change == "source":
        (project / "t1.mov").write_bytes(b"changed")
    elif change == "missing":
        (project / "cut/receipt.json").unlink()
    else:
        target = project / "cut/receipt.json"
        record = json.loads(target.read_bytes())
        record["outputs"] = [item for item in record["outputs"] if item["path"] != "cut/cut.mp4"]
        write(target, record)
    before = stable_files(project)
    with pytest.raises(studio.StudioError):
        studio.export_project(project)
    assert stable_files(project) == before and not fake_renderer.exists()


@pytest.mark.parametrize("change", ["document", "cut", "source"])
def test_changed_snapshot_during_export_refuses_publication(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    run = studio._run

    def change_input(args: list[str], install: Any = None) -> Any:
        result = run(args, install)
        if change == "document":
            target = project / "edit.json"
            data = json.loads(target.read_bytes())
            write(target, {**data, "end_hold_s": 1})
        else:
            (project / ("cut/cut.mp4" if change == "cut" else "t1.mov")).write_bytes(b"changed")
        return result

    monkeypatch.setattr(studio, "_run", change_input)
    with pytest.raises(studio.StudioError, match="changed during export"):
        studio.export_project(project)
    assert not (project / ".yap/studio").exists()


def test_export_refuses_symlink_destination(
    project: Path, fake_renderer: Path, tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.txt").write_text("keep", encoding="utf-8")
    (project / ".yap").mkdir(exist_ok=True)
    (project / ".yap/studio").symlink_to(external, target_is_directory=True)
    before = files(external)
    with pytest.raises(OutputPathError, match="symlink"):
        studio.export_project(project)
    assert files(external) == before


def test_open_parses_deep_link_and_passes_background_port_and_no_open(
    project: Path, fake_renderer: Path,
) -> None:
    result = studio.open_project(project, port=9234, open_browser=False)
    assert result["url"] == "http://127.0.0.1:9234/#project/studio"
    assert result["port"] == 9234
    assert commands(fake_renderer)[-1] == [
        "preview", "--background", "--port", "9234", "--no-open", str(project / ".yap/studio"),
    ]


def test_default_port_binds_loopback_and_default_open_is_enabled(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Socket:
        def __enter__(self) -> Socket:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        def bind(self, address: tuple[str, int]) -> None:
            assert address == ("127.0.0.1", 0)

        def getsockname(self) -> tuple[str, int]:
            return "127.0.0.1", 9123

    monkeypatch.setattr(studio.socket, "socket", lambda *_args: Socket())
    result = studio.open_project(project)
    assert result["port"] == 9123 and "--no-open" not in commands(fake_renderer)[-1]


@pytest.mark.parametrize("port", [0, -1, 65536, True, 1.5])
def test_invalid_port_is_refused_before_export(project: Path, port: Any) -> None:
    before = files(project)
    with pytest.raises(studio.StudioError, match="port"):
        studio.open_project(project, port=port, open_browser=False)
    assert files(project) == before


def test_open_requires_url_with_project_hash(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_TEST_STUDIO_URL", "Studio http://127.0.0.1:9234/")
    with pytest.raises(studio.StudioError, match="#project/<name>"):
        studio.open_project(project, port=9234, open_browser=False)


def test_stop_and_catch_up_commands(project: Path, fake_renderer: Path) -> None:
    target = studio.export_project(project)
    before = files(project)
    assert studio.stop_project(project) == {"stopped": True, "export": str(target)}
    assert studio.catch_up(project) == {"turns": [], "files": []}
    assert commands(fake_renderer)[-2:] == [
        ["preview", "--stop", str(target)], ["catch-up", "--json", str(target)],
    ]
    assert files(project) == before


@pytest.mark.parametrize("operation", [studio.stop_project, studio.catch_up])
def test_session_commands_require_export(project: Path, operation: Any) -> None:
    before = files(project)
    with pytest.raises(InputError, match="no Studio export yet"):
        operation(project)
    assert files(project) == before


@pytest.mark.parametrize("operation", [studio.stop_project, studio.catch_up, studio.open_project])
def test_session_command_exit_status_is_checked(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch, operation: Any,
) -> None:
    studio.export_project(project)
    monkeypatch.setenv("YAP_TEST_STUDIO_EXIT", "3")
    if operation == studio.open_project:
        monkeypatch.setattr(studio, "export_project", lambda _root: project / ".yap/studio")
        arguments = {"port": 9234, "open_browser": False}
    else:
        arguments = {}
    with pytest.raises(studio.StudioError, match=r"\(3\)"):
        operation(project, **arguments)


@pytest.mark.parametrize("payload", ["not-json", "[]"])
def test_catch_up_requires_json_object(
    project: Path, fake_renderer: Path, monkeypatch: pytest.MonkeyPatch, payload: str,
) -> None:
    studio.export_project(project)
    monkeypatch.setenv("YAP_TEST_STUDIO_CATCH_UP", payload)
    with pytest.raises(studio.StudioError, match="JSON object"):
        studio.catch_up(project)


@pytest.mark.parametrize("command", ["export", "open", "stop", "catch-up"])
def test_cli_subcommands_and_json(
    project: Path, fake_renderer: Path, capsys: pytest.CaptureFixture, command: str,
) -> None:
    if command in {"stop", "catch-up"}:
        studio.export_project(project)
    options = ["--port", "9234", "--no-open"] if command == "open" else []
    assert cli.main(["studio", command, str(project), "--json", *options]) == 0
    result = json.loads(capsys.readouterr().out)
    assert isinstance(result, dict)
    if command == "export":
        assert result["schema"] == "yap-studio/studio-export@1"
    elif command == "open":
        assert result["url"].endswith("#project/studio")
    elif command == "stop":
        assert result["stopped"]
    else:
        assert result == {"turns": [], "files": []}


@pytest.mark.parametrize("command", [None, "export", "open", "stop", "catch-up"])
def test_cli_help_explains_inspection_and_no_read_back(
    capsys: pytest.CaptureFixture, command: str | None,
) -> None:
    arguments = ["studio", *([command] if command else []), "--help"]
    assert cli.main(arguments) == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "agent and debugging inspection, not editing" in output
    assert "Nothing is read back from Studio into the project." in output


def test_cli_json_error(project: Path, capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["studio", "catch-up", str(project), "--json"]) == 2
    result = capsys.readouterr()
    assert not result.out and "no Studio export yet" in json.loads(result.err)["error"]["message"]


@pytest.mark.render
def test_real_pinned_renderer_lints_export_and_returns_catch_up_json(project: Path) -> None:
    """The lead runs this opt-in integration against the real pinned renderer."""
    if os.environ.get("YAP_STUDIO_RENDER_TESTS") != "1":
        pytest.skip("set YAP_STUDIO_RENDER_TESTS=1")
    # Only this opt-in test needs decodable media for the real dependency's inspection.
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
        "color=c=black:s=1080x1920:r=30:d=3", "-f", "lavfi", "-i",
        "anullsrc=r=48000:cl=stereo", "-t", "3", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(project / "cut/cut.mp4"),
    ], timeout_s=60)
    target = project / "cut/receipt.json"
    record = json.loads(target.read_bytes())
    for output in record["outputs"]:
        if output["path"] == "cut/cut.mp4":
            output["sha256"] = receipt.sha256_file(project / output["path"])
    write(target, record)
    add_layers(project)
    target = studio.export_project(project)
    assert summary(target)["lint"]["errorCount"] == 0
    assert isinstance(studio.catch_up(project), dict)
    assert hyperframes.ensure_installed().version == "0.8.192"
