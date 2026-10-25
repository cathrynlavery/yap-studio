"""Cover command boundaries and receipt ownership."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yap_studio import cli
from yap_studio.covers import command
from yap_studio.covers.command import check_owned_output, finished_snapshot, require_finished
from yap_studio.covers.pick import Pick, TimelineSnapshot
from yap_studio.covers.render import CoverAssets, RenderedCover
from yap_studio.errors import OutputPathError, ValidationError
from yap_studio.style import resolve


def test_parser_registered(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["cover", "--help"]) == 0
    assert "--look" in capsys.readouterr().out


def test_missing_final_has_fix(tmp_path: Path) -> None:
    with pytest.raises(ValidationError) as exc:
        require_finished(tmp_path)
    assert "yap-studio assemble" in str(exc.value) or "yap-studio assemble" in (
        exc.value.suggested_action or "")


def test_error_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["cover", str(tmp_path), "--headline", "A short headline", "--json"]) == 2
    error = json.loads(capsys.readouterr().err)["error"]
    assert {"type", "code", "message"} <= error.keys()


def test_foreign_png_refused(tmp_path: Path) -> None:
    png = tmp_path / "cover.png"
    png.write_bytes(b"foreign")
    with pytest.raises(OutputPathError):
        check_owned_output(png, tmp_path / "cover.json")


@pytest.mark.parametrize("shape", ["vertical", "horizontal"])
def test_receipt_fields_and_owned_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    shape: str,
) -> None:
    (tmp_path / "project.json").write_text(json.dumps({"name": "sample", "shape": shape}))
    (tmp_path / "final").mkdir()
    (tmp_path / "cut").mkdir()
    (tmp_path / "final/sample.mp4").write_bytes(b"final")
    (tmp_path / "cut/cut.mp4").write_bytes(b"cut")
    (tmp_path / "final/receipt.json").write_text(json.dumps({
        "schema": "yap-studio/stage-receipt@1", "stage": "assemble", "status": "verified",
        "inputs": {"media:cut/cut.mp4": hashlib.sha256(b"cut").hexdigest()},
        "timeline": {"schema": "yap-studio/resolved-timeline@1", "total_frames": 90,
                     "end_hold_frames": 10, "elements": [
                         {"kind": "insert", "start_frame": 30, "end_frame": 40}]},
    }))
    context = SimpleNamespace(project=SimpleNamespace(name="sample"),
                              render_style=lambda: resolve("preset:editorial_serif"),
                              resolved_timeline=lambda: (_ for _ in ()).throw(
                                  AssertionError("live timeline must not be read")))
    monkeypatch.setattr(command, "ProjectContext", lambda _path: context)
    monkeypatch.setattr(command, "cut_faces", lambda *_args: SimpleNamespace(frames=()))
    def picked(_cut, _faces, timeline, **_kwargs):
        assert isinstance(timeline, TimelineSnapshot)
        assert timeline.total_frames == 90 and timeline.replacements == (("insert", 30, 40),)
        assert _kwargs.get("shape", "vertical") == shape
        return Pick(3, "override", [{"frame": 3, "score": 1}])
    monkeypatch.setattr(command, "pick_frame", picked)
    monkeypatch.setattr(command, "prepare_assets", lambda cut, *_args, **_kwargs:
                        CoverAssets(cut, face_box=(450, 350, 300, 390)))
    def rendered(work: Path, *_args) -> RenderedCover:
        png = work / "cover.png"
        png.write_bytes(b"synthetic png")
        return RenderedCover(png, {"geometry": {"passed": True},
                                   "contrast": {"passed": True}}, "test-pin")
    monkeypatch.setattr(command, "render_cover", rendered)
    assert cli.main(["cover", str(tmp_path), "--headline", "A better night", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "yap-studio/cover@1"
    assert result["look"] == "photo" and result["frame"] == 3
    assert result["pick"]["candidates"][0]["score"] == 1
    assert result["style"]["sha256"] and result["renderer_version"] == "test-pin"
    assert result["face_box"] == {"source": "matte", "box": [450, 350, 300, 390]}
    assert result["hashes"]["cut"] and result["hashes"]["final"]
    png = tmp_path / "covers/sample-photo.png"
    check_owned_output(png, png.with_suffix(".json"))
    assert cli.main(["cover", str(tmp_path), "--headline", "A better night"]) == 0
    capsys.readouterr()
    png.write_bytes(b"foreign replacement")
    assert cli.main(["cover", str(tmp_path), "--headline", "A better night"]) == 3
    assert "differs from its cover receipt" in capsys.readouterr().err


def test_changed_cut_after_final_refused(tmp_path: Path) -> None:
    (tmp_path / "cut").mkdir()
    (tmp_path / "cut/cut.mp4").write_bytes(b"new cut")
    document = tmp_path / "receipt.json"
    document.write_text(json.dumps({
        "inputs": {"media:cut/cut.mp4": hashlib.sha256(b"old cut").hexdigest()},
        "timeline": {"schema": "yap-studio/resolved-timeline@1", "total_frames": 90,
                     "end_hold_frames": 0, "elements": []},
    }))
    with pytest.raises(ValidationError, match="the cut changed after the final was assembled") \
            as exc:
        finished_snapshot(document, tmp_path / "cut/cut.mp4", tmp_path)
    assert "yap-studio assemble" in (exc.value.suggested_action or "")


def test_pending_cover_receipt_allows_retry_after_final_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "project.json").write_text('{"name":"sample"}')
    (tmp_path / "final").mkdir()
    (tmp_path / "cut").mkdir()
    (tmp_path / "final/sample.mp4").write_bytes(b"final")
    (tmp_path / "cut/cut.mp4").write_bytes(b"cut")
    (tmp_path / "final/receipt.json").write_text(json.dumps({
        "schema": "yap-studio/stage-receipt@1", "stage": "assemble", "status": "verified",
        "inputs": {"media:cut/cut.mp4": hashlib.sha256(b"cut").hexdigest()},
        "timeline": {"schema": "yap-studio/resolved-timeline@1", "total_frames": 90,
                     "end_hold_frames": 0, "elements": []},
    }))
    context = SimpleNamespace(project=SimpleNamespace(name="sample"),
                              render_style=lambda: resolve("preset:editorial_serif"))
    monkeypatch.setattr(command, "ProjectContext", lambda _path: context)
    monkeypatch.setattr(command, "cut_faces", lambda *_args: SimpleNamespace(frames=()))
    monkeypatch.setattr(command, "pick_frame", lambda *_args, **_kwargs:
                        Pick(3, "override", [{"frame": 3}]))
    monkeypatch.setattr(command, "prepare_assets", lambda cut, *_args, **_kwargs:
                        CoverAssets(cut))
    content = [b"old cover"]

    def rendered(work: Path, *_args) -> RenderedCover:
        png = work / "cover.png"
        png.write_bytes(content[0])
        return RenderedCover(png, {"geometry": {"passed": True}}, "test-pin")

    monkeypatch.setattr(command, "render_cover", rendered)
    args = ["cover", str(tmp_path), "--headline", "A better night"]
    assert cli.main(args) == 0
    content[0] = b"new cover"
    real_write = command.paths.atomic_write_text
    fail = [True]

    def write(path: Path, value: str) -> None:
        if path.name == "sample-photo.json" and fail[0]:
            fail[0] = False
            raise OSError("receipt write failed")
        real_write(path, value)

    monkeypatch.setattr(command.paths, "atomic_write_text", write)
    with pytest.raises(OSError, match="receipt write failed"):
        cli.main(args)
    png = tmp_path / "covers/sample-photo.png"
    pending = tmp_path / "covers/sample-photo.pending.json"
    assert png.read_bytes() == b"new cover"
    assert pending.is_file()
    assert cli.main(args) == 0
    assert png.read_bytes() == b"new cover"
    assert not pending.exists()


def test_pending_cover_receipt_does_not_own_unlisted_hash(tmp_path: Path) -> None:
    png = tmp_path / "sample-photo.png"
    png.write_bytes(b"foreign")
    document = png.with_suffix(".json")
    document.write_text(json.dumps({"schema": "yap-studio/cover@1", "outputs": [
        {"path": str(png), "sha256": hashlib.sha256(b"old").hexdigest()}]}))
    png.with_name("sample-photo.pending.json").write_text(json.dumps({
        "schema": "yap-studio/cover@1", "status": "pending", "outputs": [
            {"path": str(png), "sha256": hashlib.sha256(b"old").hexdigest()},
            {"path": str(png), "sha256": hashlib.sha256(b"new").hexdigest()},
        ],
    }))
    with pytest.raises(OutputPathError, match="differs"):
        check_owned_output(png, document)
