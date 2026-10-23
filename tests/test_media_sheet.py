"""Real synthetic sheet images and deterministic local speech evidence."""

from __future__ import annotations

import json
import math
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth_broll as synth

from yap_studio import cli, ffmpeg, hdr, receipt, transcript
from yap_studio.errors import (
    DependencyError,
    InputError,
    OutputPathError,
    ProcessingError,
    ValidationError,
)
from yap_studio.media import (
    DESCRIPTION_SCHEMA,
    add_paths,
    catalog,
    catalog_path,
    describe_item,
    get_item,
    make_sheet,
)
from yap_studio.transcript import ScriptedBackend, Transcript, Word

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


def test_shared_frame_extraction_retries_the_previous_frame(tmp_path: Path, monkeypatch) -> None:
    from yap_studio.media.frame import extract_frame

    source, output = tmp_path / "source.mov", tmp_path / "tile.png"
    source.write_bytes(b"synthetic source")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            Path(args[-1]).write_bytes(b"synthetic frame")

    monkeypatch.setattr(ffmpeg, "run", run)
    extract_frame(source, SimpleNamespace(fps=30, rotation=90, video_frames=90), output,
                  time_s=1.5, width=90, height=160,
                  conversion=hdr.HdrConversion("test", [], "convert_bt709,transpose=cclock", 90))
    assert len(calls) == 2
    assert [float(command[command.index("-ss") + 1]) for command in calls] == (
        pytest.approx([1.5, 1.5 - 1 / 30])
    )
    assert calls[-1][calls[-1].index("-vf") + 1].count("transpose=cclock") == 1


def test_shared_frame_extraction_uses_the_heif_grid(tmp_path: Path, monkeypatch) -> None:
    from yap_studio.media.frame import extract_frame

    source, output = tmp_path / "source.heic", tmp_path / "tile.png"
    source.write_bytes(b"synthetic grid")
    commands = []
    monkeypatch.setattr(catalog, "heif_grid", lambda source: {})

    def run(args, **kwargs):
        commands.append(args)
        Path(args[-1]).write_bytes(b"synthetic picture")

    monkeypatch.setattr(ffmpeg, "run", run)
    extract_frame(source, SimpleNamespace(fps=None, rotation=0, video_frames=1), output,
                  time_s=None, width=90, height=160, conversion=None)
    graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert graph.startswith("[0:g:0]scale=")
    assert commands[0][commands[0].index("-map") + 1] == "[out]"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


@pytest.fixture(scope="session")
def sample_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synth.clip(tmp_path_factory.mktemp("sheet") / "clip.mov")


@pytest.fixture
def clip(tmp_path: Path, sample_clip: Path) -> Path:
    media = tmp_path / "media"
    media.mkdir()
    source = Path(shutil.copyfile(sample_clip, media / "clip.mov"))
    add_paths([source])
    return source


def _make(**kwargs: object) -> dict:
    return make_sheet("m1", now=lambda: NOW, **kwargs)


def _assert_grid(result: dict, item: dict, count: int) -> None:
    sheet = result["sheet"]
    info = ffmpeg.probe(sheet["path"], count_frames=False)
    assert info.video_codec == "mjpeg"
    assert (info.width, info.height) == (sheet["width"], sheet["height"])
    assert max(info.width, info.height) <= 1600
    columns = min(2 if item["orientation"] == "horizontal" else 3, count)
    rows = math.ceil(count / columns)
    # Derive the tile dimensions from the decoded sheet, including its 8 px gutters.
    width = (info.width - (columns + 1) * 8) / columns
    height = (info.height - (rows + 1) * 8) / rows
    assert width.is_integer() and height.is_integer()
    assert max(width, height) >= 240
    assert width / height == pytest.approx(item["width"] / item["height"], rel=0.015)
    times = []
    for i, frame in enumerate(sheet["frames"]):
        time = item["duration_s"] * (2 * i + 1) / (2 * count)
        assert frame == {"index": i, "time_s": time, "column": i % columns, "row": i // columns}
        assert 0 < time < item["duration_s"]
        times.append(time)
    assert len(times) == count and times == sorted(set(times))


@pytest.mark.parametrize(("size", "frames"), [("160x90", 6), ("90x160", 6), ("80x80", 12)])
def test_clip_grid_geometry_and_no_files_beside_media(
    tmp_path: Path, size: str, frames: int,
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    source = synth.clip(media / "clip.mov", size=size, audio=False)
    add_paths([source])
    original = source.read_bytes(), source.stat().st_mtime_ns
    result = _make(frames=frames)
    item = get_item("m1")
    _assert_grid(result, item, frames)
    assert result["sheet"]["colour_converted"] is None
    assert item["sheet"] == result["sheet"] and item["speech"] == result["speech"]
    assert set(media.iterdir()) == {source}
    assert (source.read_bytes(), source.stat().st_mtime_ns) == original
    folder = tmp_path / "config" / "assets" / "sheets" / "m1"
    assert result["folder"] == str(folder)
    assert set(path.name for path in folder.iterdir()) == {"sheet.jpg", "receipt.json"}
    saved = json.loads((folder / "receipt.json").read_text())
    assert saved["schema"] == "yap-studio/receipt@1"
    assert saved["media"] == {"id": "m1", "sha256": item["sha256"]}
    assert saved["sheet"] == {"frames": frames, "asr": "auto"}
    assert saved["inputs"][0]["path"] == str(source)
    assert saved["inputs"][0]["sha256"] == item["sha256"]
    assert saved["outputs"][0]["sha256"] == item["sheet"]["sha256"]


def test_rotated_clip_has_upright_vertical_tiles(tmp_path: Path, sample_clip: Path) -> None:
    if not synth.display_rotation_supported():
        pytest.skip("display rotation is unsupported by this encoder")
    source = synth.rotated(sample_clip, tmp_path / "rotated.mov")
    add_paths([source])
    item = get_item("m1")
    assert item["orientation"] == "vertical"
    _assert_grid(_make(asr="none"), item, 6)


def _gray_frame(
    path: Path, size: tuple[int, int], *, time_s: float | None = None, autorotate: bool = True,
) -> np.ndarray:
    width, height = size
    result = ffmpeg.run([
        "ffmpeg", "-v", "error", *([] if autorotate else ["-noautorotate"]),
        *([] if time_s is None else ["-ss", str(time_s)]),
        "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
        "-vf", f"scale={width}:{height}:flags=area,format=gray",
        "-threads", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-",
    ], timeout_s=30, binary=True)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width)


@pytest.mark.parametrize("rotation", [90, -90, 180])
def test_explicit_rotation_matches_ffmpeg_autorotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rotation: int,
) -> None:
    if not synth.display_rotation_supported():
        pytest.skip("display rotation is unsupported by this encoder")
    source = synth.rotated(
        synth.top_bottom_clip(tmp_path / "asymmetric.mov"), tmp_path / "rotated.mov",
        rotation=rotation,
    )
    add_paths([source])
    item = get_item("m1")
    assert item["rotation"] == rotation % 360
    original = ffmpeg.run
    tile = tmp_path / "extracted.png"

    def run(cmd: list[str], **kwargs: object) -> object:
        result = original(cmd, **kwargs)
        if cmd[0] == "ffmpeg" and str(source) in cmd and cmd[-1].endswith(".png"):
            assert "-noautorotate" in cmd[:cmd.index("-i")]
            shutil.copyfile(cmd[-1], tile)
        elif any("tile-%02d.png" in arg for arg in cmd):
            assert "-noautorotate" in cmd[:cmd.index("-i")]
        return result

    monkeypatch.setattr(ffmpeg, "run", run)
    result = _make(frames=1, asr="none")
    _assert_grid(result, item, 1)
    size = (45, 80) if item["orientation"] == "vertical" else (80, 45)
    actual = _gray_frame(tile, size, autorotate=False)
    reference = _gray_frame(source, size, time_s=result["sheet"]["frames"][0]["time_s"])
    error = np.abs(actual.astype(float) - reference.astype(float))
    assert float(error.mean()) < 3.0
    assert float(reference.std()) > 80.0


@pytest.mark.parametrize("rotation", [90, -90])
def test_rotated_hlg_has_vertical_tiles_and_bounded_sheet(tmp_path: Path, rotation: int) -> None:
    if not synth.display_rotation_supported():
        pytest.skip("display rotation is unsupported by this encoder")
    source = synth.rotated(
        synth.hlg(tmp_path / "hlg.mp4", size="128x72"), tmp_path / "rotated.mp4",
        rotation=rotation,
    )
    add_paths([source])
    item = get_item("m1")
    assert item["hdr"] == "hlg" and item["orientation"] == "vertical"
    result = _make()
    _assert_grid(result, item, 6)
    assert type(result["sheet"]["colour_converted"]) is bool
    sheet = result["sheet"]
    picture = _gray_frame(Path(sheet["path"]), (sheet["width"], sheet["height"]), autorotate=False)
    width = (sheet["width"] - 4 * 8) // 3
    height = (sheet["height"] - 3 * 8) // 2
    for frame in sheet["frames"]:
        left, top = 8 + frame["column"] * (width + 8), 8 + frame["row"] * (height + 8)
        # This synthetic pattern has bright pixels in every frame. An empty decode
        # must not silently turn the last grid cell black when tiling reaches EOF.
        assert float(picture[top:top + height, left:left + width].mean()) > 10.0


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("size", ["160x90", "90x160", "80x80"])
def test_short_clip_uses_one_tile_per_actual_frame(tmp_path: Path, count: int, size: str) -> None:
    add_paths([synth.clip(tmp_path / "short.mov", size=size, audio=False, fps=1, duration=count)])
    result = _make(frames=6)
    item = get_item("m1")
    _assert_grid(result, item, count)
    assert [frame["column"] for frame in result["sheet"]["frames"]] == list(range(count))
    assert all(frame["row"] == 0 for frame in result["sheet"]["frames"])
    if count == 1 and item["orientation"] == "vertical":
        assert result["sheet"]["height"] > result["sheet"]["width"]


@pytest.mark.parametrize(("builder", "size", "expected"), [
    (synth.png, "120x80", (120, 80)), (synth.jpeg, "80x120", (80, 120)),
    (synth.png, "2000x1000", (1600, 800)),
])
def test_still_one_tile_aspect_and_no_upscaling(
    tmp_path: Path, builder: object, size: str, expected: tuple[int, int],
) -> None:
    extension = ".png" if builder is synth.png else ".jpg"
    source = builder(tmp_path / f"still{extension}", size=size)
    add_paths([source])
    result = _make()
    info = ffmpeg.probe(result["sheet"]["path"], count_frames=False)
    assert (info.width, info.height) == expected
    assert result["sheet"]["frames"] == [{"index": 0, "time_s": None, "column": 0, "row": 0}]
    assert result["sheet"]["colour_converted"] is None
    assert result["speech"] is None
    assert _make()["noop"] is True


def test_hlg_sheet_records_conversion_boolean(tmp_path: Path) -> None:
    add_paths([synth.hlg(tmp_path / "hlg.mp4")])
    result = _make()
    assert type(result["sheet"]["colour_converted"]) is bool
    assert len(result["sheet"]["frames"]) == 6


def test_failed_hdr_conversion_falls_back_to_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_paths([synth.hlg(tmp_path / "hlg.mp4")])
    selected = hdr.HdrConversion("test", [], "filter_that_does_not_exist")
    monkeypatch.setattr(hdr, "conversion", lambda info: selected)
    assert _make()["sheet"]["colour_converted"] is False
    monkeypatch.setattr(hdr, "conversion", lambda info: (_ for _ in ()).throw(
        hdr.HdrUnsupportedError()
    ))
    assert _make(force=True)["sheet"]["colour_converted"] is False


def test_noop_writes_nothing_force_and_parameter_change_regenerate(clip: Path) -> None:
    result = _make(asr="none")
    files = [path for path in catalog_path().parent.rglob("*") if path.is_file()]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}
    assert _make(asr="none") == {**result, "noop": True}
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files} == before
    forced = make_sheet("m1", asr="none", force=True, now=lambda: NOW + timedelta(seconds=1))
    assert forced["noop"] is False
    sheet_path = Path(result["sheet"]["path"])
    assert sheet_path.stat().st_mtime_ns != before[sheet_path][1]
    changed = _make(frames=3, asr="none")
    assert changed["noop"] is False and len(changed["sheet"]["frames"]) == 3


def test_no_remaining_location_and_replaced_file_raise_input_error(clip: Path) -> None:
    _make(asr="none")
    clip.rename(clip.with_name("moved.mov"))
    with pytest.raises(InputError, match="no catalog location still holds") as caught:
        _make(asr="none")
    assert caught.value.path == str(clip)
    synth.clip(clip, colour="red")
    with pytest.raises(InputError, match="run yap-studio media add"):
        _make(force=True)


def test_latest_matching_location_and_fallback(tmp_path: Path, clip: Path) -> None:
    copy = Path(shutil.copyfile(clip, tmp_path / "new.mov"))
    add_paths([copy])
    result = _make(asr="none")
    saved = json.loads((Path(result["folder"]) / "receipt.json").read_text())
    assert saved["inputs"][0]["path"] == str(copy)
    synth.clip(copy, colour="red")
    result = _make(asr="none", force=True)
    saved = json.loads((Path(result["folder"]) / "receipt.json").read_text())
    assert saved["inputs"][0]["path"] == str(clip)


def test_tampered_sheet_is_not_overwritten(clip: Path) -> None:
    result = _make(backend_factory=lambda name: ScriptedBackend(
        lambda source: Transcript([Word("previous", 0, 1)])
    ))
    folder = Path(result["folder"])
    sheet = Path(result["sheet"]["path"])
    sheet.write_bytes(b"foreign content")
    files = set(folder.iterdir())
    before = {path: path.read_bytes() for path in (*files, catalog_path())}
    with pytest.raises(OutputPathError, match="receipt matching its current hash") as caught:
        _make(force=True, asr="none")
    assert set(folder.iterdir()) == files
    assert {path: path.read_bytes() for path in before} == before
    assert not (folder / "receipt.pending.json").exists()
    assert caught.value.suggested_action == (
        f"Delete the sheet folder {result['folder']}, then run yap-studio media sheet m1 again."
    )


@pytest.mark.parametrize("remove_transcript", [True, False], ids=["deletion", "replacement"])
def test_tampered_transcript_refused_before_any_publication(
    clip: Path, remove_transcript: bool,
) -> None:
    result = _make(backend_factory=lambda name: ScriptedBackend(
        lambda source: Transcript([Word("previous", 0, 1)])
    ))
    folder = Path(result["folder"])
    target = Path(result["speech"]["path"])
    edited = json.loads(target.read_text())
    edited["words"][0]["text"] = "edited"
    target.write_text(json.dumps(edited) + "\n")
    assert Transcript.load(target).text == "edited"
    files = set(folder.iterdir())
    before = {path: path.read_bytes() for path in (*files, catalog_path())}

    with pytest.raises(OutputPathError, match="receipt matching its current hash") as caught:
        _make(
            frames=3, force=True, asr="none" if remove_transcript else "auto",
            backend_factory=lambda name: ScriptedBackend(
                lambda source: Transcript([Word("replacement", 0, 1)])
            ),
        )
    assert set(folder.iterdir()) == files
    assert {path: path.read_bytes() for path in before} == before
    assert not (folder / "receipt.pending.json").exists()
    assert caught.value.suggested_action == (
        f"Delete the sheet folder {result['folder']}, then run yap-studio media sheet m1 again."
    )


@pytest.mark.parametrize("previous", [False, True], ids=["first", "forced"])
def test_sheet_publication_interrupt_before_receipt_recovers(
    clip: Path, monkeypatch: pytest.MonkeyPatch, previous: bool,
) -> None:
    if previous:
        _make(frames=6, asr="none")
    before = catalog_path().read_bytes()
    folder = catalog_path().parent / "sheets" / "m1"

    def interrupt(target: Path, evidence: dict) -> None:
        assert target == folder / "receipt.json"
        assert receipt.sha256_file(folder / "sheet.jpg") == evidence["outputs"][0]["sha256"]
        raise KeyboardInterrupt

    with monkeypatch.context() as patched:
        patched.setattr(receipt, "write_receipt", interrupt)
        with pytest.raises(KeyboardInterrupt):
            _make(frames=3, force=True, asr="none")
    assert catalog_path().read_bytes() == before
    rerun = _make(frames=3, force=True, asr="none")
    assert rerun["noop"] is False
    saved = json.loads((folder / "receipt.json").read_text())
    item = get_item("m1")
    assert item["sheet"] == rerun["sheet"]
    assert item["sheet"]["sha256"] == saved["outputs"][0]["sha256"] == receipt.sha256_file(
        folder / "sheet.jpg"
    )
    assert not (folder / "receipt.pending.json").exists()


@pytest.mark.parametrize("force", [False, True])
def test_sheet_publication_interrupt_after_receipt_recovers(
    clip: Path, monkeypatch: pytest.MonkeyPatch, force: bool,
) -> None:
    _make(frames=6, asr="none")
    folder = catalog_path().parent / "sheets" / "m1"
    pending = folder / "receipt.pending.json"
    original_unlink = Path.unlink

    def interrupt(target: Path, *args: object, **kwargs: object) -> None:
        if target == pending:
            assert (folder / "receipt.json").is_file()
            raise KeyboardInterrupt
        original_unlink(target, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(Path, "unlink", interrupt)
        with pytest.raises(KeyboardInterrupt):
            _make(frames=6, force=True, asr="none")
    assert pending.is_file()
    rerun = _make(frames=6, force=force, asr="none")
    assert rerun["noop"] is False
    saved = json.loads((folder / "receipt.json").read_text())
    assert get_item("m1")["sheet"] == rerun["sheet"]
    assert rerun["sheet"]["sha256"] == saved["outputs"][0]["sha256"] == receipt.sha256_file(
        folder / "sheet.jpg"
    )
    assert not pending.exists()


@pytest.mark.parametrize("remove_transcript", [False, True])
def test_transcript_publication_interrupt_journals_replacement_or_removal(
    clip: Path, monkeypatch: pytest.MonkeyPatch, remove_transcript: bool,
) -> None:
    def factory(name: str) -> ScriptedBackend:
        return ScriptedBackend(lambda source: Transcript([Word("previous", 0, 1)]))

    first = _make(backend_factory=factory)
    folder = Path(first["folder"])
    target = Path(first["speech"]["path"])
    old_hash = receipt.sha256_file(target)
    asr = "none" if remove_transcript else "auto"

    def changed(name: str) -> ScriptedBackend:
        return ScriptedBackend(lambda source: Transcript([Word("changed", 0, 1)]))

    def interrupt(receipt_path: Path, evidence: dict) -> None:
        raise KeyboardInterrupt

    with monkeypatch.context() as patched:
        patched.setattr(receipt, "write_receipt", interrupt)
        with pytest.raises(KeyboardInterrupt):
            _make(frames=3, force=True, asr=asr, backend_factory=changed)
    pending_path = folder / "receipt.pending.json"
    pending = json.loads(pending_path.read_text())
    assert pending["schema"] == "yap-studio/stage-pending@1"
    assert all(Path(entry["path"]).is_absolute() for entry in pending["outputs"])
    assert pending["outputs"][-1] == {
        "path": str(target), "old_sha256": old_hash,
        "new_sha256": None if remove_transcript else receipt.sha256_file(target),
    }
    rerun = _make(frames=3, force=True, asr=asr, backend_factory=changed)
    saved = json.loads((folder / "receipt.json").read_text())
    assert get_item("m1")["sheet"] == rerun["sheet"]
    assert get_item("m1")["speech"] == rerun["speech"]
    assert all(receipt.sha256_file(entry["path"]) == entry["sha256"] for entry in saved["outputs"])
    assert target.exists() is not remove_transcript
    assert not pending_path.exists()


def test_conflicting_receipt_and_pending_journal_refused(
    clip: Path,
) -> None:
    result = _make(asr="none")
    folder = Path(result["folder"])
    pending = folder / "receipt.pending.json"
    saved = json.loads((folder / "receipt.json").read_text())
    pending.write_text(json.dumps({
        "schema": "yap-studio/stage-pending@1", "started_at": saved["started_at"],
        "media": saved["media"], "outputs": [{
            "path": result["sheet"]["path"], "old_sha256": None, "new_sha256": "a" * 64,
        }],
    }))
    before = {path: path.read_bytes() for path in (*folder.iterdir(), catalog_path())}
    with pytest.raises(InputError, match="receipt and journal conflict") as caught:
        _make(force=True, asr="none")
    assert caught.value.path == str(pending)
    assert {path: path.read_bytes() for path in before} == before


def test_speech_words_transcript_receipt_catalog_and_reuse(clip: Path) -> None:
    calls = []

    def scripted(source: Path) -> Transcript:
        calls.append(source)
        return Transcript([Word("Hello", 0.1, 0.5, 0.9), Word("world", 0.5, 1.0, 0.8)])

    def factory(name: str) -> ScriptedBackend:
        return ScriptedBackend(scripted, name="local-test", model="tiny")
    result = _make(backend_factory=factory)
    speech = result["speech"]
    assert calls == [clip]
    assert speech == {
        "status": "words", "engine": "local-test", "model": "tiny", "words": 2,
        "text": "Hello world", "path": str(Path(result["folder"]) / "transcript.json"),
        "message": None, "at": "2026-10-05T12:00:00Z",
    }
    assert Transcript.load(speech["path"]).text == "Hello world"
    assert get_item("m1")["speech"] == speech
    saved = json.loads((Path(result["folder"]) / "receipt.json").read_text())
    assert [entry["path"] for entry in saved["outputs"]] == [
        result["sheet"]["path"], speech["path"],
    ]
    before = {path: path.stat().st_mtime_ns for path in catalog_path().parent.rglob("*")}
    assert _make(backend_factory=factory)["noop"] is True and calls == [clip]
    assert {path: path.stat().st_mtime_ns for path in before} == before
    Path(speech["path"]).unlink()
    assert _make(backend_factory=factory)["noop"] is False and calls == [clip, clip]


def test_no_words_and_stale_transcript_removed(clip: Path) -> None:
    result = _make(backend_factory=lambda name: ScriptedBackend(lambda source: Transcript([
        Word("word", 0, 1),
    ])))
    old_transcript = Path(result["speech"]["path"])
    result = _make(force=True, backend_factory=lambda name: ScriptedBackend(
        lambda source: Transcript([])
    ))
    assert result["speech"]["status"] == "no_speech"
    assert result["speech"]["path"] is None
    assert not old_transcript.exists()
    assert _make()["noop"] is True


@pytest.mark.parametrize("named", [False, True])
def test_missing_engine_clear_status(clip: Path, named: bool) -> None:
    def factory(name: str) -> None:
        if named:
            raise DependencyError.missing_engine(name)
        return None

    result = _make(asr="parakeet-onnx" if named else "auto", backend_factory=factory)
    assert result["speech"]["status"] == "no_engine"
    message = result["speech"]["message"]
    assert "not found" in message if named else "install yap-studio[asr]" in message
    assert not (Path(result["folder"]) / "transcript.json").exists()


def test_no_audio_and_none_skip_backend(tmp_path: Path, clip: Path) -> None:
    def fail(name: str) -> None:
        pytest.fail("backend was called")

    result = _make(asr="none", backend_factory=fail)
    assert result["speech"]["status"] == "skipped"
    assert result["speech"]["message"] == "speech transcription was turned off"
    source = synth.clip(tmp_path / "silent.mov", audio=False)
    add_paths([source])
    assert make_sheet("m2", backend_factory=fail)["speech"]["status"] == "no_audio"


@pytest.mark.parametrize("error", [
    OSError("engine failed"), ProcessingError("asr", 1, "bad audio"),
])
@pytest.mark.parametrize("as_json", [False, True])
def test_failed_speech_keeps_sheet_and_cli_exits_one(
    clip: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    error: Exception, as_json: bool,
) -> None:
    def fail(source: Path) -> Transcript:
        raise error

    monkeypatch.setattr(transcript, "resolve_backend", lambda name: ScriptedBackend(fail))
    flag = ["--json"] if as_json else []
    assert cli.main(["media", "sheet", "m1", *flag]) == 1
    output = capsys.readouterr()
    assert output.err == ""
    assert get_item("m1")["speech"]["status"] == "failed"
    assert Path(get_item("m1")["sheet"]["path"]).is_file()
    if as_json:
        result = json.loads(output.out)
        assert result["speech"]["status"] == "failed" and str(error) in result["speech"]["message"]
    else:
        assert "Speech: failed" in output.out and str(error) in output.out


def test_speech_text_is_capped_and_full_transcript_is_saved(clip: Path) -> None:
    result = _make(backend_factory=lambda name: ScriptedBackend(
        lambda source: Transcript([Word("x" * 4100, 0, 1)])
    ))
    assert len(result["speech"]["text"]) == 4000
    assert len(Transcript.load(result["speech"]["path"]).text) == 4100


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_sheet_words_print_a_capped_preview(
    clip: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], as_json: bool,
) -> None:
    text = "A speech preview " + "x" * 300
    monkeypatch.setattr(transcript, "resolve_backend", lambda name: ScriptedBackend(
        lambda source: Transcript([Word(text, 0, 1)])
    ))
    assert cli.main(["media", "sheet", "m1", *(["--json"] if as_json else [])]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    if as_json:
        assert json.loads(output.out)["speech"]["text"] == text
    else:
        assert output.out.splitlines()[-1] == f"Speech: words; text: {text[:200]}"
    assert get_item("m1")["speech"]["text"] == text


def test_cli_show_readable_details_and_speech_preview(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    source = synth.clip(tmp_path / "camera.mov", camera=True)
    add_paths([source])
    describe_item("m1", {
        "schema": DESCRIPTION_SCHEMA, "description": "A laptop", "tags": ["desk"],
        "media_type": "screen_recording",
        "provenance": {"kind": "own", "source": "phone", "license": None, "url": None},
        "ranges": [
            {"verdict": "approved", "start_s": 0.25, "end_s": 1.5, "reason": "steady"},
            {"verdict": "rejected", "reason": "shaky"},
        ],
    }, now=lambda: NOW)
    describe_item("m1", {"schema": DESCRIPTION_SCHEMA, "revoke": ["d1"]}, now=lambda: NOW)
    text = "A clear speech preview " + "x" * 300
    result = _make(backend_factory=lambda name: ScriptedBackend(
        lambda source: Transcript([Word(text, 0, 1)])
    ))
    assert cli.main(["media", "show", "m1"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "ID: m1",
        "Kind: video; media type: screen_recording (set by agent)",
        "Size: 160x90; orientation: horizontal",
        "Duration: 2.000 s; fps: 30/1",
        "HDR: none", "Audio: yes", "Camera: Example Camera Pocket Model", "Provenance: own",
        "Description: A laptop", "Tags: desk",
        "d1 approved 0.250 to 1.500 s: steady (agent, 2026-10-05T12:00:00Z) (revoked)",
        "d2 rejected whole item: shaky (agent, 2026-10-05T12:00:00Z)",
        f"Sheet: {result['sheet']['path']}",
        f"Speech: words; text: {text[:200]}",
        f"Location (exists): {source}",
    ]
    assert cli.main(["media", "show", "m1", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown == {**get_item("m1"), "locations": [
        {**location, "exists": True} for location in get_item("m1")["locations"]
    ]}
    assert shown["speech"]["text"] == text


@pytest.mark.parametrize("still", [False, True])
def test_cli_show_still_and_hdr_facts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], still: bool,
) -> None:
    source = synth.png(tmp_path / "still.png") if still else synth.hlg(tmp_path / "hlg.mp4")
    add_paths([source])
    assert cli.main(["media", "show", "m1"]) == 0
    output = capsys.readouterr().out
    assert "Audio: no" in output
    if still:
        assert "Kind: image; media type: screenshot" in output
        assert "Duration: still\n" in output
        assert "fps:" not in output
        assert "Speech: not applicable for a still\n" in output
    else:
        assert "HDR: hlg\n" in output
        assert "Speech: not checked yet; run yap-studio media sheet m1\n" in output


def test_speech_interrupt_preserves_previous_outputs_and_rerun_succeeds(clip: Path) -> None:
    result = _make(backend_factory=lambda name: ScriptedBackend(
        lambda source: Transcript([Word("previous", 0, 1)])
    ))
    folder = Path(result["folder"])
    files = [
        folder / "sheet.jpg", folder / "receipt.json", folder / "transcript.json", catalog_path(),
    ]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files}

    def interrupt(source: Path) -> Transcript:
        assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files} == before
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _make(frames=3, force=True, backend_factory=lambda name: ScriptedBackend(interrupt))
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in files} == before
    rerun = _make(frames=3, backend_factory=lambda name: ScriptedBackend(
        lambda source: Transcript([Word("rerun", 0, 1)])
    ))
    assert rerun["noop"] is False and len(rerun["sheet"]["frames"]) == 3
    assert rerun["speech"]["text"] == "rerun"
    assert _make(frames=3)["noop"] is True


@pytest.mark.parametrize("initial_status", ["failed", "no_engine"])
def test_failed_or_unavailable_speech_retries_without_force(
    clip: Path, initial_status: str,
) -> None:
    def fail(source: Path) -> Transcript:
        raise OSError("temporary engine failure")

    def factory(name: str) -> ScriptedBackend | None:
        return ScriptedBackend(fail) if initial_status == "failed" else None

    first = _make(backend_factory=factory)
    assert first["speech"]["status"] == initial_status
    calls = []

    def transcribe(source: Path) -> Transcript:
        calls.append(source)
        return Transcript([Word("available", 0, 1)])

    retried = _make(backend_factory=lambda name: ScriptedBackend(transcribe))
    assert retried["noop"] is False and retried["speech"]["status"] == "words"
    assert calls == [clip]
    assert Transcript.load(retried["speech"]["path"]).text == "available"
    assert _make(backend_factory=lambda name: ScriptedBackend(transcribe))["noop"] is True
    assert calls == [clip]


def test_slow_work_allows_catalog_updates_and_keeps_other_changes(
    clip: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ffmpeg.run
    changed = False

    def run(cmd: list[str], **kwargs: object) -> object:
        nonlocal changed
        if not changed and cmd[0] == "ffmpeg":
            changed = True
            describe_item("m1", {"schema": DESCRIPTION_SCHEMA, "description": "During render"})
        return original(cmd, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", run)

    def scripted(source: Path) -> Transcript:
        describe_item("m1", {"schema": DESCRIPTION_SCHEMA, "tags": ["during speech"]})
        return Transcript([])

    _make(backend_factory=lambda name: ScriptedBackend(scripted))
    description = get_item("m1")["description"]
    assert description["text"] == "During render"
    assert description["tags"] == ["during speech"] and description["revision"] == 2


def test_catalog_hash_change_during_speech_is_not_overwritten(clip: Path) -> None:
    def scripted(source: Path) -> Transcript:
        def mutate(document: dict) -> None:
            document["items"][0]["sha256"] = "a" * 64
            document["index"][str(clip)]["sha256"] = "a" * 64

        catalog.update_catalog(mutate)
        return Transcript([])

    with pytest.raises(InputError, match="catalog media changed"):
        _make(backend_factory=lambda name: ScriptedBackend(scripted))
    assert get_item("m1")["sha256"] == "a" * 64 and get_item("m1")["sheet"] is None


@pytest.mark.parametrize("frames", [0, 13, -1, True, 1.5])
def test_invalid_frames_leave_catalog_unchanged(clip: Path, frames: object) -> None:
    before = catalog_path().read_bytes()
    with pytest.raises(ValidationError, match="frames"):
        _make(frames=frames)
    assert catalog_path().read_bytes() == before


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_sheet_show_and_errors(
    clip: Path, capsys: pytest.CaptureFixture[str], as_json: bool,
) -> None:
    flag = ["--json"] if as_json else []
    assert cli.main(["media", "sheet", "m1", "--asr", "none", "--frames", "3", *flag]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    item = get_item("m1")
    if as_json:
        assert json.loads(output.out)["sheet"] == item["sheet"]
    else:
        assert item["sheet"]["path"] in output.out
        assert "Frames: 0.333 s, 1.000 s, 1.667 s" in output.out
        assert "Speech: skipped: speech transcription was turned off" in output.out
    assert cli.main(["media", "show", "m1"]) == 0
    output = capsys.readouterr().out
    assert f"Sheet: {item['sheet']['path']}" in output
    assert "Speech: skipped: speech transcription was turned off" in output
    for args in (["m99"], ["m1", "--frames", "13"]):
        assert cli.main(["media", "sheet", *args, *flag]) == 2
        output = capsys.readouterr()
        assert output.out == ""
        if as_json:
            assert "error" in json.loads(output.err)
        else:
            assert "error:" in output.err


def test_heif_tile_grid_uses_filter_complex_group_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = synth.png(tmp_path / "still.png", size="120x80")
    image = Path(shutil.copyfile(source, tmp_path / "still.heic"))
    original_probe = catalog._probe_json
    original_run = ffmpeg.run
    captured = []

    def probe(path: Path, options: list[str]) -> dict:
        if "-show_stream_groups" in options:
            return {"stream_groups": [{"type": "Tile Grid", "components": [{
                "width": 120, "height": 80,
            }]}]}
        return original_probe(path, options)

    def run(cmd: list[str], **kwargs: object) -> object:
        if cmd[0] == "ffmpeg" and str(image) in cmd:
            captured.append(cmd)
            assert "-filter_complex" in cmd and cmd[cmd.index("-map") + 1] == "[out]"
            assert cmd[cmd.index("-filter_complex") + 1].startswith("[0:g:0]scale=")
            # A synthetic PNG stands in for the decoded grid; preserve actual rendering.
            cmd = [arg.replace("[0:g:0]", "[0:v:0]") for arg in cmd]
        return original_run(cmd, **kwargs)

    monkeypatch.setattr(catalog, "_probe_json", probe)
    monkeypatch.setattr(ffmpeg, "run", run)
    add_paths([image])
    result = _make()
    assert len(captured) == 1 and (result["sheet"]["width"], result["sheet"]["height"]) == (120, 80)
