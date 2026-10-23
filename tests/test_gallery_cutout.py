"""Synthetic stills and measured gray mattes exercise the cached person PNG."""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from yap_studio import ffmpeg, receipt, tools
from yap_studio.errors import ProcessingError
from yap_studio.faces.model import VideoIdentity
from yap_studio.gallery import cutout, still
from yap_studio.matte import make, vision
from yap_studio.matte.model import Cleanup, MatteData


def picture(path: Path, *, size: str = "16x24", color: str = "0x3264C8") -> Path:
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                f"color=c={color}:s={size},format=rgb24", "-frames:v", "1", str(path)],
               timeout_s=30)
    return path


def pixels(path: Path, width: int = 16, height: int = 24) -> np.ndarray:
    raw = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                      "-pix_fmt", "rgba", "-f", "rawvideo", "-"],
                     timeout_s=30, binary=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)


@pytest.fixture
def fake_make(monkeypatch):
    monkeypatch.setattr(vision, "unavailable_reason", lambda: None)
    calls = []

    def measured(clip, folder, *, backend):
        assert backend == "vision"
        info = ffmpeg.probe(clip, count_frames=True)
        assert (info.video_codec, info.pix_fmt, info.video_frames, info.fps) == (
            "h264", "yuv420p", 1, Fraction(30))
        calls.append(clip)
        width, height = info.width, info.height
        raw = folder / "mask.gray"
        raw.write_bytes((bytes([255]) * (width // 2) + bytes(width - width // 2)) * height)
        matte = folder / "matte.mkv"
        ffmpeg.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                    "-s", f"{width}x{height}", "-r", "30", "-i", str(raw),
                    "-frames:v", "1", "-c:v", "ffv1", "-pix_fmt", "gray", str(matte)],
                   timeout_s=30)
        return MatteData(
            "measured", None,
            VideoIdentity(clip.name, receipt.sha256_file(clip), 1, Fraction(30), width, height),
            {"name": "synthetic"}, Cleanup(),
            {"file": "matte.mkv", "sha256": receipt.sha256_file(matte),
             "codec": "ffv1", "pix_fmt": "gray"}, (0.0,), 1 / 30, (0.5,),
        )

    return measured, calls


def test_cutout_pixels_record_cache_and_new_still(tmp_path, fake_make):
    measured, calls = fake_make
    source = picture(tmp_path / "still.png")
    out = tmp_path / "gallery"
    image = cutout.prepare_cutout(source, out, make=measured)
    assert image == out / "cutout" / "cutout.png"
    rgba = pixels(image)
    assert np.array_equal(rgba[:, :, :3], pixels(source)[:, :, :3])
    assert np.all(rgba[:, :8, 3] == 255)
    assert np.all(rgba[:, 8:, 3] == 0)
    record = json.loads(image.with_suffix(".json").read_text())
    assert record["schema"] == "yap-studio/gallery-cutout@1"
    assert record["still_sha256"] == receipt.sha256_file(source)
    assert (record["width"], record["height"]) == (16, 24)
    assert MatteData.from_dict(record["matte"]).status == "measured"
    assert cutout.prepare_cutout(source, out, make=measured) == image
    assert len(calls) == 1
    other = picture(tmp_path / "other.png", color="red")
    assert cutout.prepare_cutout(other, out, make=measured) == image
    assert len(calls) == 2
    updated = json.loads(image.with_suffix(".json").read_text())
    assert updated["still_sha256"] == receipt.sha256_file(other)
    assert np.array_equal(pixels(image)[:, :, :3], pixels(other)[:, :, :3])
    assert sorted(item.name for item in image.parent.iterdir()) == ["cutout.json", "cutout.png"]
    assert all(not clip.parent.exists() for clip in calls)


@pytest.mark.parametrize("corrupt", ["{", "[]", "null", "schema", "sha", "missing-png"])
def test_invalid_cache_rebuilds(tmp_path, fake_make, corrupt):
    measured, calls = fake_make
    source = picture(tmp_path / "still.png")
    image = cutout.prepare_cutout(source, tmp_path / "gallery", make=measured)
    record = image.with_suffix(".json")
    if corrupt == "missing-png":
        image.unlink()
    elif corrupt in {"schema", "sha"}:
        data = json.loads(record.read_text())
        data["schema" if corrupt == "schema" else "still_sha256"] = "wrong"
        record.write_text(json.dumps(data))
    else:
        record.write_text(corrupt)
    assert cutout.prepare_cutout(source, tmp_path / "gallery", make=measured) == image
    assert len(calls) == 2


def test_drawing_and_unavailable_vision_write_nothing(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("matting must not run")

    monkeypatch.setattr(vision, "unavailable_reason", forbidden)
    out = tmp_path / "gallery"
    drawing = tmp_path / "drawing.svg"
    drawing.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    assert cutout.prepare_cutout(drawing, out, make=forbidden) is None
    monkeypatch.setattr(vision, "unavailable_reason", lambda: "unavailable here")
    assert cutout.prepare_cutout(picture(tmp_path / "still.png"), out, make=forbidden) is None
    assert not out.exists()
    assert capsys.readouterr().err.splitlines() == [
        "warning: preview cutout unavailable: unavailable here"]


@pytest.mark.parametrize("failure", ["matte", "os", "empty-os", "unmeasured", "merge", "record"])
def test_failure_leaves_no_complete_output(tmp_path, fake_make, monkeypatch, capsys, failure):
    measured, _calls = fake_make
    source = picture(tmp_path / "still.png")
    out = tmp_path / "gallery"

    def failing(clip, folder, **kwargs):
        if failure == "os":
            raise OSError("cannot read matte")
        if failure == "empty-os":
            raise OSError
        if failure == "matte":
            raise ProcessingError("matte", 1, "failed\nmore detail")
        result = measured(clip, folder, **kwargs)
        if failure == "unmeasured":
            return MatteData("unavailable", "no matte", result.video, {}, Cleanup(),
                             None, None, None, None)
        return result

    if failure == "merge":
        run = ffmpeg.run

        def merge_error(command, **kwargs):
            if "-filter_complex" in command:
                Path(command[-1]).write_bytes(b"partial")
                raise OSError("cannot merge")
            return run(command, **kwargs)

        monkeypatch.setattr(ffmpeg, "run", merge_error)
    if failure == "record":
        def record_error(*args, **kwargs):
            raise OSError("cannot write record")
        monkeypatch.setattr(cutout.paths, "atomic_write_text", record_error)
    assert cutout.prepare_cutout(source, out, make=failing) is None
    assert list((out / "cutout").iterdir()) == []
    assert len(capsys.readouterr().err.splitlines()) == 1


def test_odd_still_size_is_kept_when_matte_is_scaled(tmp_path, fake_make):
    measured, _calls = fake_make
    source = picture(tmp_path / "odd.png", size="17x25")
    image = cutout.prepare_cutout(source, tmp_path / "gallery", make=measured)
    rgba = pixels(image, 17, 25)
    assert np.array_equal(rgba[:, :, :3], pixels(source, 17, 25)[:, :, :3])
    assert np.all(rgba[:, 0, 3] == 255) and np.all(rgba[:, -1, 3] == 0)


def test_cutout_subfolder_keeps_gallery_output_usable(tmp_path, fake_make):
    measured, _calls = fake_make
    out = tmp_path / "gallery"
    image = cutout.prepare_cutout(picture(tmp_path / "still.png"), out, make=measured)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    result = still.prepare_preview(inbox, out)
    assert result["source"] == "sample"
    still._check_output_folder(out)
    assert image.is_file()


_REAL_HELPER = pytest.mark.skipif(
    sys.platform != "darwin" or tools.find_tool("xcrun") is None,
    reason="real Vision helper needs macOS and xcrun",
)


@pytest.mark.helpers
@_REAL_HELPER
def test_real_vision_cutout_size(tmp_path, _owner_folders, monkeypatch):
    monkeypatch.setenv("CLANG_MODULE_CACHE_PATH", str(_owner_folders / "clang"))
    monkeypatch.setenv("SWIFT_MODULECACHE_PATH", str(_owner_folders / "swift"))

    def real_make(*args, **kwargs):
        try:
            return make.make_matte(*args, **kwargs)
        except ProcessingError as exc:
            if (exc.code == "person_matte_exit_2"
                    and "sysctlbyname for kern.hv_vmm_present failed" in exc.full_stderr
                    and "Cannot Decode" in exc.full_stderr):
                pytest.skip("the execution sandbox blocks the native macOS video decoder")
            raise

    source = picture(tmp_path / "still.png", size="108x192")
    image = cutout.prepare_cutout(source, tmp_path / "gallery", make=real_make)
    assert image is not None
    info = ffmpeg.probe(image)
    assert (info.width, info.height, info.pix_fmt) == (108, 192, "rgba")
