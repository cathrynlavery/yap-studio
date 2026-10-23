"""Synthetic cut matte caching and alpha-based timeline occlusion measurements."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import synth
from test_matte_timing import timestamp_plate

from yap_studio import ffmpeg, receipt
from yap_studio.faces.model import VideoIdentity
from yap_studio.matte import check
from yap_studio.matte import project as cache
from yap_studio.matte.model import Cleanup, MatteData


def synthetic_matte(
    cut: Path, folder: Path, *, box: tuple[int, int, int, int] | None = None,
    active_windows: tuple[tuple[int, int], ...] | None = None,
    soft_box: tuple[int, int, int, int] | None = None,
    joins: tuple[int, ...] = (),
) -> MatteData:
    """Write an FFV1 gray mask and its real cut identity without segmentation."""
    folder.mkdir(parents=True, exist_ok=True)
    info = ffmpeg.probe(cut)
    assert info.width and info.height and info.video_frames and info.fps
    x, y, width, height = box or (0, 0, info.width, info.height)
    windows = active_windows if active_windows is not None else ((0, info.video_frames),)
    active = "+".join(f"between(N,{first},{last - 1})" for first, last in windows) or "0"
    person = (f"gte(X,{x})*lt(X,{x + width})*gte(Y,{y})*lt(Y,{y + height})"
              f"*gt({active},0)")
    expression = f"255*({person})"
    if soft_box is not None:
        sx, sy, sw, sh = soft_box
        soft = f"gte(X,{sx})*lt(X,{sx + sw})*gte(Y,{sy})*lt(Y,{sy + sh})"
        expression = f"({person})*(255-115*({soft}))"
    matte = folder / "matte.mkv"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        f"color=black:s={info.width}x{info.height}:r={info.fps},format=gray,"
        f"geq=lum='{expression}'", "-frames:v", str(info.video_frames),
        "-threads", "2", "-c:v", "ffv1", "-pix_fmt", "gray", str(matte),
    ], timeout_s=120)
    data = MatteData(
        "measured", None,
        VideoIdentity(cut.name, receipt.sha256_file(cut), info.video_frames, info.fps,
                      info.width, info.height),
        {"name": "synthetic"}, Cleanup(median=False, low=0, high=255, feather_sigma=0),
        {"file": matte.name, "sha256": receipt.sha256_file(matte),
         "codec": "ffv1", "pix_fmt": "gray"},
        tuple(float(frame / info.fps) for frame in range(info.video_frames)),
        float(info.video_frames / info.fps),
        tuple(width * height / (info.width * info.height)
              if any(first <= frame < last for first, last in windows) else 0.0
              for frame in range(info.video_frames)),
        joins=joins,
    )
    (folder / "matte.json").write_text(json.dumps(data.to_dict()), encoding="utf-8")
    return data


@pytest.fixture
def cut(tmp_path: Path) -> Path:
    return synth.picture_source(tmp_path / "cut.mp4", "color=blue:s=64x96:r=30", 12)


@pytest.mark.parametrize("has_joins", [False, True])
def test_cache_requests_and_legacy_auto(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch, has_joins: bool,
) -> None:
    calls = []
    expected = (6,) if has_joins else ()
    if has_joins:
        (cut.parent / "spans.json").write_text(json.dumps({
            "sources": {"a": "a.mov", "b": "b.mov"}, "segments": [
                {"source": "a", "start": 0, "end": "1/5"},
                {"source": "b", "start": 0, "end": "1/5"},
            ],
        }))

    def make(source: Path, folder: Path, **kwargs) -> MatteData:
        calls.append(kwargs.get("backend", "auto"))
        assert kwargs.get("joins", ()) == expected
        assert ("joins" in kwargs) == has_joins
        data = synthetic_matte(source, folder, joins=expected)
        if kwargs.get("backend") == "vision":
            assert kwargs["requested"] == "person"
            data.backend["requested"] = kwargs["requested"]
            (folder / "matte.json").write_text(json.dumps(data.to_dict()), encoding="utf-8")
        return data

    monkeypatch.setattr(cache, "make_matte", make)
    monkeypatch.setattr("yap_studio.matte.vision.unavailable_reason", lambda: None)
    root = tmp_path / "project"
    first = cache.ensure_cut_matte(root, cut)
    cache.ensure_cut_matte(root, cut, cutout="auto")
    assert calls == ["auto"]
    assert "requested" not in json.loads((first.parent / "matte.json").read_text())["backend"]
    cache.ensure_cut_matte(root, cut, cutout="person")
    cache.ensure_cut_matte(root, cut, cutout="person")
    assert calls == ["auto", "vision"]
    assert json.loads((first.parent / "matte.json").read_text())["backend"]["requested"] == "person"
    cache.ensure_cut_matte(root, cut, cutout="auto")
    assert calls == ["auto", "vision", "auto"]

@pytest.mark.parametrize("has_spans", [False, True])
def test_cut_owns_joins_and_passes_them_when_present(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch, has_spans: bool,
) -> None:
    root = tmp_path / "project"
    (root / "cut").mkdir(parents=True)
    (root / "cut/spans.json").write_text("Not the cut's own edit list")
    if has_spans:
        (cut.parent / "spans.json").write_text(json.dumps({
            "sources": {"a": "a.mov", "b": "b.mov"}, "segments": [
                {"source": "a", "start": 0, "end": "1/5"},
                {"source": "b", "start": 0, "end": "1/5"},
            ],
        }))
    expected = (6,) if has_spans else ()
    built, checked = [], []
    cached = cache._cached

    def make(source: Path, folder: Path, *, joins: tuple[int, ...] = ()) -> MatteData:
        built.append(joins)
        return synthetic_matte(source, folder, joins=joins)

    def check_cached(folder: Path, source: Path, info: ffmpeg.MediaInfo, joins):
        checked.append(joins)
        return cached(folder, source, info, joins)

    monkeypatch.setattr(cache, "make_matte", make)
    monkeypatch.setattr(cache, "_cached", check_cached)
    matte = cache.ensure_cut_matte(root, cut)
    assert cache.ensure_cut_matte(root, cut) == matte
    assert json.loads((matte.parent / "matte.json").read_text())["joins"] == list(expected)
    assert built == [expected] and checked == [expected] * 3


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("planned_frames", [11, 13])
def test_mismatched_cut_spans_refused_before_cache_or_build(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch,
    cached: bool, planned_frames: int,
) -> None:
    root = tmp_path / "project"
    monkeypatch.setattr(cache, "make_matte", synthetic_matte)
    matte = cache.ensure_cut_matte(root, cut) if cached else None
    before = matte.read_bytes() if matte else None
    (cut.parent / "spans.json").write_text(json.dumps({
        "sources": {"a": "a.mov"}, "segments": [
            {"source": "a", "start": 0, "end": f"{planned_frames}/30"},
        ],
    }))
    monkeypatch.setattr(cache, "make_matte", lambda *args, joins=(): pytest.fail("matte build"))
    monkeypatch.setattr(cache, "_cached", lambda *args: pytest.fail("cache lookup"))
    with pytest.raises(cache.CutMatteError, match="frame") as caught:
        cache.ensure_cut_matte(root, cut)
    assert "re-run cut" in str(caught.value).lower()
    assert "constant-rate" not in caught.value.suggested_action
    if matte:
        assert matte.read_bytes() == before
    else:
        assert not (root / ".yap/cache/matte").exists()


def test_cache_reuses_cut_hash_and_rebuilds_changed_cut(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    root = tmp_path / "project"

    def make(source: Path, folder: Path, *, joins=()) -> MatteData:
        digest = receipt.sha256_file(source)
        assert not (root / ".yap/cache/matte" / digest).exists()
        calls.append(digest)
        return synthetic_matte(source, folder)

    monkeypatch.setattr(cache, "make_matte", make)
    first = cache.ensure_cut_matte(root, cut)
    before = first.stat().st_mtime_ns
    assert first.parent == root / ".yap/cache/matte" / receipt.sha256_file(cut)
    assert cache.ensure_cut_matte(root, cut) == first
    assert first.stat().st_mtime_ns == before and len(calls) == 1
    idle_since = cache.time.time() - cache.MATTE_IDLE_SECONDS - 1
    os.utime(first.parent, (idle_since, idle_since))
    other = synth.picture_source(tmp_path / "other.mp4", "color=red:s=64x96:r=30", 12)
    shutil.copyfile(other, cut)
    second = cache.ensure_cut_matte(root, cut)
    assert second.parent != first.parent and len(calls) == 2
    assert not first.parent.exists() and second.is_file()
    assert list(second.parent.parent.iterdir()) == [second.parent]
    assert not list(second.parent.parent.glob(".matte-*"))


def test_publish_retires_hash_directories_atomically_and_preserves_other_entries(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    monkeypatch.setattr(cache, "make_matte", synthetic_matte)
    first = cache.ensure_cut_matte(root, cut)
    parent = first.parent.parent
    notes = parent / "notes"
    notes.mkdir()
    (notes / "keep.txt").write_text("unrelated")
    inflight = parent / ".matte-inflight"
    inflight.mkdir()
    (inflight / "keep.txt").write_text("still building")
    invalid_name = parent / ("g" * 64)
    invalid_name.mkdir()
    (invalid_name / "keep.txt").write_text("not a hash")
    hash_file = parent / ("a" * 64)
    hash_file.write_text("a file, not a cache directory")
    target = tmp_path / "unrelated"
    target.mkdir()
    (target / "keep.txt").write_text("outside the cache")
    link = parent / ("b" * 64)
    link.symlink_to(target, target_is_directory=True)
    damaged = parent / ("c" * 64)
    damaged.mkdir()
    (damaged / "matte.json").write_text("incomplete old matte")
    idle_since = cache.time.time() - cache.MATTE_IDLE_SECONDS - 1
    for entry in (first.parent, damaged):
        os.utime(entry, (idle_since, idle_since))
    other = synth.picture_source(tmp_path / "other.mp4", "color=red:s=64x96:r=30", 12)
    shutil.copyfile(other, cut)
    current = parent / receipt.sha256_file(cut)
    renamed = []
    replace = cache.os.replace

    def observe(source: Path, destination: Path) -> None:
        if source in (first.parent, damaged):
            # Publication is complete before an old entry disappears. The old
            # directory moves as one unit; deletion happens in temporary storage.
            assert (current / "matte.mkv").is_file() and (current / "matte.json").is_file()
            replace(source, destination)
            assert not source.exists() and (destination / "matte.json").is_file()
            renamed.append((source, destination))
        else:
            replace(source, destination)

    monkeypatch.setattr(cache.os, "replace", observe)
    second = cache.ensure_cut_matte(root, cut)
    assert second.parent == current
    assert {source for source, _ in renamed} == {first.parent, damaged}
    assert all(not destination.exists() for _, destination in renamed)
    assert set(parent.iterdir()) == {current, notes, inflight, invalid_name, hash_file, link}
    assert (notes / "keep.txt").read_text() == "unrelated"
    assert (inflight / "keep.txt").read_text() == "still building"
    assert (invalid_name / "keep.txt").read_text() == "not a hash"
    assert hash_file.read_text() == "a file, not a cache directory"
    assert link.is_symlink() and (target / "keep.txt").read_text() == "outside the cache"


@pytest.mark.parametrize("reuse", [False, True])
def test_returned_matte_survives_publication_while_idle_entry_is_retired(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch, reuse: bool,
) -> None:
    root = tmp_path / "project"
    clock = [1_700_000_000.0]
    monkeypatch.setattr(cache.time, "time", lambda: clock[0])
    monkeypatch.setattr(cache, "make_matte", synthetic_matte)
    idle_cut = synth.picture_source(tmp_path / "idle.mp4", "color=green:s=64x96:r=30", 12)
    idle = cache.ensure_cut_matte(root, idle_cut)
    clock[0] += 1
    first = cache.ensure_cut_matte(root, cut)
    assert first.parent.stat().st_mtime == clock[0]
    first_returned = clock[0]
    # An entry used exactly ten minutes ago remains protected. Reusing an older
    # entry refreshes its use time, rather than its publication or movie mtime.
    clock[0] += cache.MATTE_IDLE_SECONDS + (1 if reuse else 0)
    if reuse:
        movie_mtime = first.stat().st_mtime_ns
        assert cache.ensure_cut_matte(root, cut) == first
        assert first.stat().st_mtime_ns == movie_mtime
        first_returned = clock[0]
    other = synth.picture_source(tmp_path / "other.mp4", "color=red:s=64x96:r=30", 12)
    second = cache.ensure_cut_matte(root, other)
    assert first.parent.stat().st_mtime == first_returned
    assert second.parent.stat().st_mtime == clock[0]
    assert not idle.parent.exists()
    assert set(second.parent.parent.iterdir()) == {first.parent, second.parent}
    # The first render can open and decode the path it received after the
    # second render publishes a different cut's matte in the same project.
    assert check._decoded_count(first) == 12


def test_concurrent_publisher_cache_hit_refreshes_use_time(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    now = 1_700_000_000.0
    monkeypatch.setattr(cache.time, "time", lambda: now)
    current = root / ".yap/cache/matte" / receipt.sha256_file(cut)

    def publish_during_build(source: Path, folder: Path, *, joins=()) -> MatteData:
        data = synthetic_matte(source, folder)
        synthetic_matte(source, current)
        idle_since = now - cache.MATTE_IDLE_SECONDS - 1
        os.utime(current, (idle_since, idle_since))
        return data

    monkeypatch.setattr(cache, "make_matte", publish_during_build)
    matte = cache.ensure_cut_matte(root, cut)
    assert matte == current / "matte.mkv"
    assert current.stat().st_mtime == now
    assert list(current.parent.iterdir()) == [current]


@pytest.mark.parametrize("state", [
    "missing", "existing", "published_during_build", "changed_before_publication",
])
def test_expensive_cache_validation_runs_without_project_lock(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch, state: str,
) -> None:
    root = tmp_path / "project"
    current = root / ".yap/cache/matte" / receipt.sha256_file(cut)
    monkeypatch.setattr(cache, "make_matte", synthetic_matte)
    if state == "existing":
        cache.ensure_cut_matte(root, cut)
    staged_hashes = []

    def make(source: Path, folder: Path, *, joins=()) -> MatteData:
        data = synthetic_matte(source, folder)
        assert data.matte is not None
        staged_hashes.append(data.matte["sha256"])
        if state in {"published_during_build", "changed_before_publication"}:
            synthetic_matte(source, current, box=(0, 0, 1, 1))
        return data

    monkeypatch.setattr(cache, "make_matte", make)
    lock_held = False
    project_lock = cache.project_lock
    cached = cache._cached
    validated = cache._validated
    cached_calls = []
    validated_calls = []

    @contextmanager
    def tracked_lock(project: Path) -> Iterator[None]:
        nonlocal lock_held
        with project_lock(project):
            lock_held = True
            try:
                if state == "changed_before_publication":
                    (current / "matte.json").write_text("changed after validation")
                yield
            finally:
                lock_held = False

    def unlocked_cached(folder: Path, source: Path, info: ffmpeg.MediaInfo, joins=()):
        assert not lock_held, "cached media validation holds the project lock"
        cached_calls.append(folder)
        return cached(folder, source, info, joins)

    def unlocked_validated(folder: Path, source: Path, info: ffmpeg.MediaInfo):
        assert not lock_held, "media hashing or decoding holds the project lock"
        validated_calls.append(folder)
        return validated(folder, source, info)

    monkeypatch.setattr(cache, "project_lock", tracked_lock)
    monkeypatch.setattr(cache, "_cached", unlocked_cached)
    monkeypatch.setattr(cache, "_validated", unlocked_validated)
    matte = cache.ensure_cut_matte(root, cut)
    assert not lock_held and matte == current / "matte.mkv"
    assert cached_calls == [current] * (1 if state == "existing" else 2)
    if state != "missing":
        assert current in validated_calls
    if state != "existing":
        assert any(folder.name == "new" for folder in validated_calls)
        assert (receipt.sha256_file(matte) == staged_hashes[0]) == (
            state != "published_during_build"
        )


@pytest.mark.parametrize("change", ["directory", "matte", "document"])
def test_changed_cached_identity_falls_through_to_rebuild(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    root = tmp_path / "project"
    calls = []

    def make(source: Path, folder: Path, *, joins=()) -> MatteData:
        calls.append(folder)
        return synthetic_matte(source, folder)

    monkeypatch.setattr(cache, "make_matte", make)
    matte = cache.ensure_cut_matte(root, cut)
    project_lock = cache.project_lock
    changed = False

    @contextmanager
    def changed_lock(project: Path) -> Iterator[None]:
        nonlocal changed
        with project_lock(project):
            if not changed:
                # Change a validated entry just before its cheap locked check.
                # Each replacement is still valid, so the rebuild proves that
                # the caller did not trust the obsolete validation identity.
                if change == "directory":
                    old = tmp_path / "replaced"
                    matte.parent.rename(old)
                    shutil.copytree(old, matte.parent)
                else:
                    target = matte if change == "matte" else matte.parent / "matte.json"
                    before = target.stat()
                    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
                changed = True
            yield

    monkeypatch.setattr(cache, "project_lock", changed_lock)
    assert cache.ensure_cut_matte(root, cut) == matte
    assert changed and len(calls) == 2
    assert cache._cached(matte.parent, cut, ffmpeg.probe(cut)) is not None
    assert not list(matte.parent.parent.glob(".matte-*"))


def test_entry_changed_during_validation_is_rebuilt(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    calls = []

    def make(source: Path, folder: Path, *, joins=()) -> MatteData:
        calls.append(folder)
        return synthetic_matte(source, folder)

    monkeypatch.setattr(cache, "make_matte", make)
    matte = cache.ensure_cut_matte(root, cut)
    validated = cache._validated
    changed = False

    def change_after_decode(folder: Path, source: Path, info: ffmpeg.MediaInfo) -> Path:
        nonlocal changed
        result = validated(folder, source, info)
        if folder == matte.parent and not changed:
            before = result.stat()
            os.utime(result, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
            changed = True
        return result

    monkeypatch.setattr(cache, "_validated", change_after_decode)
    assert cache.ensure_cut_matte(root, cut) == matte
    assert changed and len(calls) == 2


def test_failed_new_cut_leaves_previous_matte_cache_intact(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    monkeypatch.setattr(cache, "make_matte", synthetic_matte)
    first = cache.ensure_cut_matte(root, cut)
    original = first.read_bytes()
    other = synth.picture_source(tmp_path / "other.mp4", "color=red:s=64x96:r=30", 12)
    shutil.copyfile(other, cut)

    def fail(source: Path, folder: Path, *, joins=()) -> MatteData:
        raise cache.CutMatteError("no segmentation backend")

    monkeypatch.setattr(cache, "make_matte", fail)
    with pytest.raises(cache.CutMatteError, match="no segmentation backend"):
        cache.ensure_cut_matte(root, cut)
    assert first.read_bytes() == original
    assert list(first.parent.parent.iterdir()) == [first.parent]


@pytest.mark.parametrize("damage", ["json", "missing", "hash", "count", "geometry", "identity"])
def test_invalid_cache_is_rebuilt(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    calls = []

    def make(source: Path, folder: Path, *, joins=()) -> MatteData:
        calls.append(folder)
        return synthetic_matte(source, folder)

    monkeypatch.setattr(cache, "make_matte", make)
    root = tmp_path / "project"
    matte = cache.ensure_cut_matte(root, cut)
    document = matte.parent / "matte.json"
    data = json.loads(document.read_text())
    if damage == "json":
        document.write_text("incomplete")
    elif damage == "missing":
        matte.unlink()
    elif damage == "hash":
        data["matte"]["sha256"] = "0" * 64
    elif damage == "identity":
        data["video"]["sha256"] = "0" * 64
    else:
        replacement = tmp_path / "replacement.mkv"
        ffmpeg.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=white:s=32x48:r=30" if damage == "geometry" else "color=white:s=64x96:r=30",
            "-frames:v", "6" if damage == "count" else "12", "-c:v", "ffv1",
            "-pix_fmt", "gray", str(replacement),
        ], timeout_s=30)
        shutil.copyfile(replacement, matte)
        data["matte"]["sha256"] = receipt.sha256_file(matte)
    if damage not in {"json", "missing"}:
        document.write_text(json.dumps(data))
    assert cache.ensure_cut_matte(root, cut) == matte and len(calls) == 2
    assert cache.ensure_cut_matte(root, cut) == matte and len(calls) == 2


def test_failed_rebuild_keeps_old_cache_and_publishes_nothing(
    tmp_path: Path, cut: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "project"
    monkeypatch.setattr(cache, "make_matte", synthetic_matte)
    matte = cache.ensure_cut_matte(root, cut)
    document = matte.parent / "matte.json"
    document.write_text("broken cache")
    original = matte.read_bytes()

    def unavailable(source: Path, folder: Path, *, joins=()) -> MatteData:
        data = synthetic_matte(source, folder)
        data = MatteData("unavailable", "no segmentation backend", data.video, {}, data.cleanup,
                         None, None, None, None)
        (folder / "matte.json").write_text(json.dumps(data.to_dict()))
        return data

    monkeypatch.setattr(cache, "make_matte", unavailable)
    with pytest.raises(cache.CutMatteError, match="no segmentation backend") as caught:
        cache.ensure_cut_matte(root, cut)
    assert "doctor" in caught.value.suggested_action
    assert document.read_text() == "broken cache" and matte.read_bytes() == original
    assert not list(matte.parent.parent.glob(".matte-*"))


def test_variable_rate_cut_refused_before_matte_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cut = timestamp_plate(tmp_path / "irregular.mkv")
    monkeypatch.setattr(cache, "make_matte",
                        lambda *args, joins=(): pytest.fail("unexpected matte work"))
    with pytest.raises(cache.CutMatteError, match="constant-rate") as caught:
        cache.ensure_cut_matte(tmp_path / "project", cut)
    assert "Rebuild the cut" in caught.value.suggested_action
    assert not (tmp_path / "project/.yap/cache/matte").exists()


def test_unusable_cut_refused_with_action(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = synth.float_audio(tmp_path / "voice.wav", np.zeros(1600, dtype=np.float32))
    monkeypatch.setattr(cache, "make_matte",
                        lambda *args, joins=(): pytest.fail("unexpected matte work"))
    with pytest.raises(cache.CutMatteError, match="usable picture") as caught:
        cache.ensure_cut_matte(tmp_path / "project", audio)
    assert caught.value.suggested_action


def test_hidden_half_of_frames_and_strict_time_threshold(tmp_path: Path, cut: Path) -> None:
    synthetic_matte(cut, tmp_path / "matte", active_windows=((0, 6),))
    synth.picture_source(tmp_path / "title.mov", "color=white:s=32x16:r=30,format=rgba,"
                         "drawbox=x=8:y=4:w=16:h=4:c=black:t=fill",
                         12, alpha=True)
    layer = {"id": "title", "path": "title.mov", "x": 8, "y": 8, "width": 32,
             "height": 16, "start_frame": 0, "frames": 12}
    args = (tmp_path, tmp_path / "matte/matte.mkv", ffmpeg.probe(cut), [layer])
    records = check.measure_behind_layers(*args)
    assert records == ({"id": "title", "start_frame": 0, "frames": 12, "hidden_frames": 6,
                        "hidden_share": 0.5, "flag": True},)
    assert not check.measure_behind_layers(*args, hidden_time_share=0.5)[0]["flag"]
    skipped = check.measure_behind_layers(*args, skipped_windows=((0, 3),))[0]
    assert skipped["hidden_frames"] == 3 and skipped["hidden_share"] == 0.25
    assert not skipped["flag"]


@pytest.mark.parametrize("covered", ["panel", "ink"])
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("alpha", [255, 200])
def test_behind_measurement_counts_ink_instead_of_panel(
    tmp_path: Path, cut: Path, covered: str, padding: int, alpha: int,
) -> None:
    # The right-hand panel region contains over half the panel's visible pixels.
    # The two short ink rows occupy under 5% of the panel, but all must be counted.
    box = (20, 8, 20, 16) if covered == "panel" else (12, 12, 6, 6)
    synthetic_matte(cut, tmp_path / "matte", box=box)
    synth.picture_source(
        tmp_path / "card.mov", "color=white:s=32x16:r=30,format=rgba,"
        "drawbox=x=4:y=4:w=6:h=2:c=black:t=fill,"
        "drawbox=x=4:y=8:w=6:h=2:c=black:t=fill,"
        "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
        f"a='{alpha}*gte(X,{padding})*lt(X,{32 - padding})"
        f"*gte(Y,{padding})*lt(Y,{16 - padding})'", 12, alpha=True,
    )
    layer = {"id": "card", "path": "card.mov", "x": 8, "y": 8, "width": 32,
             "height": 16, "start_frame": 0, "frames": 12}
    record = check.measure_behind_layers(
        tmp_path, tmp_path / "matte/matte.mkv", ffmpeg.probe(cut), [layer],
    )[0]
    assert record == {"id": "card", "start_frame": 0, "frames": 12,
                      "hidden_frames": 12 if covered == "ink" else 0,
                      "hidden_share": 1.0 if covered == "ink" else 0.0,
                      "flag": covered == "ink"}


def test_glyph_threshold_empty_alpha_and_clipped_box(
    cut: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two opaque glyphs surrounded by transparent pixels. Half covered is not hidden.
    glyphs = np.zeros((1, 4, 4), dtype=np.uint8)
    glyphs[..., :3] = (255, 0, 0)
    glyphs[..., 3] = (0, 128, 128, 127)
    person = np.zeros((96, 64), dtype=np.uint8)
    person[0, 0] = 128
    layers = [{"id": "edge", "path": "edge.mov", "x": -1, "y": 0, "width": 4,
               "height": 1, "start_frame": 0, "frames": 12}]

    def windows(source: Path, *args: Any, rgba: bool = False, **kwargs: Any):
        for frame in range(12):
            yield glyphs if rgba and frame < 6 else (
                np.zeros_like(glyphs) if rgba else person
            )

    monkeypatch.setattr(check, "_window_frames", windows)
    args = (cut.parent, cut, ffmpeg.probe(cut), layers)
    assert check.measure_behind_layers(*args)[0]["hidden_frames"] == 0
    assert check.measure_behind_layers(*args, hidden_glyph_share=0.49)[0]["hidden_frames"] == 6
