"""Private offline inventory adoption, catalog mapping, search, and CLI behavior."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from yap_studio import cli, ffmpeg, paths
from yap_studio.errors import DependencyError, ProcessingError, ValidationError
from yap_studio.media import inventory
from yap_studio.media.catalog import add_paths, catalog_path, load_catalog, update_catalog
from yap_studio.media.inventory import inventory_root, search_inventory, sync_inventory
from yap_studio.render import hyperframes
from yap_studio.render.hyperframes import HYPERFRAMES_VERSION, RENDERER_ENV, RendererInstall

pytestmark = pytest.mark.usefixtures("_no_real_node")

STAMP = "2026-10-06T12:00:00Z"
_FAKE_RENDERER = r'''import json, os, re, sys
from pathlib import Path

args = sys.argv[1:]
assert args[:2] == ["media-use", "resolve"]
root = Path(args[args.index("--project") + 1])
home = Path(os.environ["HYPERFRAMES_MEDIA_HOME"])
with Path(os.environ["YAP_TEST_INVENTORY_LOG"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"argv": args, "env": {key: os.environ.get(key)
                                                  for key in env_names}}) + "\n")
controls = json.loads(Path(os.environ["YAP_TEST_INVENTORY_CONTROLS"]).read_text())
operation = "adopt" if "--adopt" in args else "search"
mode = controls.get(operation, "success")
if mode == "nonzero":
    print(json.dumps({"ok": True, "assets": []}))
    sys.exit(9)
if mode == "empty":
    sys.exit(0)
if mode == "stderr_json":
    print(json.dumps({"ok": True, "assets": []}), file=sys.stderr)
    sys.exit(0)
if mode == "invalid_json":
    print("invalid inventory output")
    sys.exit(0)
if mode == "false":
    print(json.dumps({"ok": False}))
    sys.exit(0)
if mode == "nonobject":
    print("[]")
    sys.exit(0)
(home / ".media").mkdir(parents=True, exist_ok=True)
(home / ".media" / "cache.json").write_text(json.dumps({"project": str(root)}))
manifest = root / ".media" / "manifest.jsonl"
if operation == "adopt":
    assets, counters = [], {"video": 0, "image": 0}
    video_extensions = {".mp4", ".webm", ".mov"}
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico"}
    for source in sorted((root / "assets").rglob("*")):
        extension = source.suffix.lower()
        if not source.is_file() or extension not in video_extensions | image_extensions:
            continue
        kind = "image" if extension in image_extensions else "video"
        counters[kind] += 1
        asset = {"id": f"{kind}_{counters[kind]:03d}", "type": kind,
                 "path": source.relative_to(root).as_posix(), "source": "existing",
                 "description": re.sub(r"[-_]+", " ", source.stem),
                 "width": 64, "height": 36,
                 "provenance": {"provider": "local", "adopted": True}}
        if kind == "video":
            asset["duration"] = 2
        assets.append(asset)
    if mode == "unknown":
        assets.append({"id": "video_999", "type": "video", "path": "assets/unknown.mp4"})
    elif mode == "duplicate_path":
        assets.append({**assets[0], "id": "video_999"})
    elif mode == "duplicate_id":
        assets[1]["id"] = assets[0]["id"]
    elif mode == "wrong_type":
        assets[0]["type"] = "image" if assets[0]["type"] == "video" else "video"
    elif mode == "omitted":
        assets.pop()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(asset) + "\n" for asset in assets))
    (manifest.parent / "index.md").write_text("Local media inventory\n")
    response = {"ok": True, "adopted": len(assets), "assets": assets}
    if mode == "missing_list":
        del response["assets"]
    elif mode == "invalid_entry":
        response["assets"] = [None]
else:
    assert "--candidates" in args
    kind = args[args.index("--type") + 1]
    assets = [json.loads(line) for line in manifest.read_text().splitlines()]
    candidates = [{**asset, "scope": "project", "score": 1 - index / 10}
                  for index, asset in enumerate(reversed(assets)) if asset["type"] == kind]
    if controls.get("extra_candidates"):
        candidates.insert(0, {**candidates[0], "scope": "global"})
        candidates.insert(1, {**candidates[0], "scope": "project",
                              "id": "video_999", "path": "assets/stale.mp4"})
    response = {"ok": True, "candidates": candidates, "truncated": False}
    if mode == "missing_list":
        del response["candidates"]
    elif mode == "invalid_entry":
        response["candidates"] = [None]
print(json.dumps(response))
'''


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str | None:
    """Keep both creator data and derived inventory inside synthetic test storage."""
    configured_cache = os.environ.get("YAP_STUDIO_CACHE")
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setenv("YAP_STUDIO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("HYPERFRAMES_MEDIA_HOME", str(tmp_path / "must-not-use"))
    return configured_cache


@dataclass(frozen=True)
class FakeRenderer:
    """A local media resolver with inspectable calls and configurable responses."""

    install: RendererInstall
    log: Path
    controls: Path

    def configure(self, **values: object) -> None:
        self.controls.write_text(json.dumps(values), encoding="utf-8")

    def calls(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]


@pytest.fixture
def fake_renderer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRenderer:
    """Write a fake executable using the renderer tests' executable pattern."""
    executable = tmp_path / "hyperframes"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        f"env_names = {['HYPERFRAMES_MEDIA_HOME', *RENDERER_ENV]!r}\n" + _FAKE_RENDERER,
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    fake = FakeRenderer(
        RendererInstall(HYPERFRAMES_VERSION, tmp_path, executable, None, "v22.0.0"),
        tmp_path / "calls.jsonl", tmp_path / "controls.json",
    )
    fake.configure()
    monkeypatch.setenv("YAP_TEST_INVENTORY_LOG", str(fake.log))
    monkeypatch.setenv("YAP_TEST_INVENTORY_CONTROLS", str(fake.controls))
    monkeypatch.setattr(hyperframes, "ensure_installed", lambda: fake.install)
    return fake


def _item(
    number: int, source: Path, *, kind: str = "video", text: str | None = None,
) -> dict:
    """Build a complete synthetic catalog item for the real catalog validator."""
    return {
        "id": f"m{number}", "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "bytes": source.stat().st_size, "kind": kind,
        "media_type": "photo" if kind == "image" else "clip", "media_type_by": "rule",
        "format": source.suffix[1:].lower(), "codec": None,
        "width": 64, "height": 36, "rotation": 0, "orientation": "horizontal",
        "duration_s": None if kind == "image" else 2,
        "fps": None if kind == "image" else "30/1", "variable_frame_rate": False,
        "color_transfer": None, "hdr": None, "has_audio": False, "camera": None,
        "provenance": {"kind": "own", "source": "camera", "license": None, "url": None},
        "locations": [{"path": str(source), "added_at": STAMP, "seen_at": STAMP}],
        "added_at": STAMP,
        "description": None if text is None else {
            "text": text, "tags": [], "by": "creator", "at": STAMP, "revision": 1,
        },
        "decisions": [], "next_decision": 1, "sheet": None, "speech": None,
    }


def _source(
    tmp_path: Path, number: int = 1, *, name: str | None = None, kind: str = "video",
    text: str | None = None,
) -> dict:
    source = tmp_path / (name or f"desk-{number}.{'png' if kind == 'image' else 'mp4'}")
    source.write_bytes(f"synthetic media {number}".encode())
    return _item(number, source, kind=kind, text=text)


def _catalog(*items: dict) -> None:
    def mutate(document: dict) -> None:
        document["items"] = copy.deepcopy(list(items))
        document["next_id"] = max((int(item["id"][1:]) for item in items), default=0) + 1
        document["index"] = {}

    update_catalog(mutate)


def _snapshot(root: Path) -> dict[str, bytes | str]:
    return {
        path.relative_to(root).as_posix(): str(path.readlink()) if path.is_symlink()
        else path.read_bytes()
        for path in root.rglob("*") if path.is_symlink() or path.is_file()
    }


def _assert_private_calls(fake: FakeRenderer) -> None:
    for call in fake.calls():
        args = call["argv"]
        assert "--local-only" in args and "--json" in args
        project = Path(args[args.index("--project") + 1])
        media_home = Path(call["env"]["HYPERFRAMES_MEDIA_HOME"])
        assert media_home == project / "home"
        assert media_home.is_relative_to(inventory_root().parent)
        assert media_home != Path.home()
        assert all(call["env"][key] == value for key, value in RENDERER_ENV.items())
    assert not (fake.log.parent / "must-not-use").exists()


def test_sync_names_links_metadata_and_first_existing_location(
    tmp_path: Path, fake_renderer: FakeRenderer,
) -> None:
    first = _source(tmp_path, name="Original.MOV", text="  DESK / Shot___Take 2!!  ")
    original = Path(first["locations"][0]["path"])
    first["locations"].insert(0, {
        "path": str(tmp_path / "missing.mov"), "added_at": STAMP, "seen_at": STAMP,
    })
    later = tmp_path / "later.mov"
    later.write_bytes(b"later location")
    first["locations"].append({"path": str(later), "added_at": STAMP, "seen_at": STAMP})
    second = _source(tmp_path, 2, name="Still__At Home.PNG", kind="image")
    third = _source(tmp_path, 3, text="A" * 59 + " / " + "Z" * 20)
    fourth = _source(tmp_path, 4, text="!!!")
    fifth = _source(tmp_path, 5, name="Fallback.Stem.JPG", kind="image")
    fifth["description"] = {"text": None, "tags": [], "by": "creator", "at": STAMP,
                            "revision": 1}
    sixth = _source(tmp_path, 6, text="B" * 80)
    _catalog(first, second, third, fourth, fifth, sixth)
    before = catalog_path().read_bytes()

    result = sync_inventory(install=fake_renderer.install)

    assert inventory_root() == tmp_path / "cache" / "media-inventory"
    expected = {
        "m1": "assets/videos/desk-shot-take-2-m1.mov",
        "m2": "assets/images/still-at-home-m2.png",
        "m3": "assets/videos/" + "a" * 59 + "-m3.mp4",
        "m4": "assets/videos/media-m4.mp4",
        "m5": "assets/images/fallback-stem-m5.jpg",
        "m6": "assets/videos/" + "b" * 60 + "-m6.mp4",
    }
    assert {item["id"]: item["path"] for item in result["items"]} == expected
    for item in (first, second, third, fourth, fifth, sixth):
        link = inventory_root() / expected[item["id"]]
        assert link.is_symlink()
        source = original if item is first else Path(item["locations"][0]["path"])
        assert link.readlink() == source
        assert link.read_bytes() == source.read_bytes()
    assert result["schema"] == "yap-studio/media-inventory@1"
    assert result["renderer"] == {"name": "hyperframes", "version": HYPERFRAMES_VERSION}
    assert datetime.strptime(result["synced_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    assert len(result["plan_sha256"]) == 64
    assert "catalog_sha256" not in result
    assert result["skipped"] == []
    assert json.loads((inventory_root() / "inventory.json").read_text()) == result
    assert (inventory_root() / ".media" / "manifest.jsonl").is_file()
    assert (inventory_root() / ".media" / "index.md").is_file()
    assert (inventory_root() / "home" / ".media" / "cache.json").is_file()
    assert catalog_path().read_bytes() == before
    assert len(fake_renderer.calls()) == 1
    _assert_private_calls(fake_renderer)
    assert not list(inventory_root().parent.glob(".media-inventory-*"))


def test_sync_skips_missing_files_and_empty_location_lists(
    tmp_path: Path, fake_renderer: FakeRenderer,
) -> None:
    first, missing, no_locations = (_source(tmp_path, number) for number in (1, 2, 3))
    Path(missing["locations"][0]["path"]).unlink()
    no_locations["locations"] = []
    _catalog(first, missing, no_locations)

    result = sync_inventory()

    assert [item["id"] for item in result["items"]] == ["m1"]
    assert result["skipped"] == [
        {"id": "m2", "reason": "no file on disk"},
        {"id": "m3", "reason": "no file on disk"},
    ]
    assert len(list((inventory_root() / "assets").rglob("*.mp4"))) == 1


def test_sync_skips_unadoptable_formats_and_links_supported_clip(
    tmp_path: Path, fake_renderer: FakeRenderer,
) -> None:
    photo = _source(tmp_path, 1, name="photo.HEIC", kind="image")
    unsupported_clip = _source(tmp_path, 2, name="clip.MKV")
    supported_clip = _source(tmp_path, 3, name="clip.MOV")
    _catalog(photo, unsupported_clip, supported_clip)
    before = catalog_path().read_bytes()

    result = sync_inventory(install=fake_renderer.install)

    assert result["items"] == [{
        "id": "m3", "hf_id": "video_001", "type": "video",
        "path": "assets/videos/clip-m3.mov", "description": "clip m3",
    }]
    assert result["skipped"] == [
        {"id": "m1", "reason": "format not adopted by the HyperFrames inventory (.heic)"},
        {"id": "m2", "reason": "format not adopted by the HyperFrames inventory (.mkv)"},
    ]
    links = [path for path in (inventory_root() / "assets").rglob("*") if path.is_symlink()]
    assert links == [inventory_root() / "assets" / "videos" / "clip-m3.mov"]
    assert links[0].readlink() == Path(supported_clip["locations"][0]["path"])
    assert json.loads((inventory_root() / "inventory.json").read_text()) == result
    assert catalog_path().read_bytes() == before
    _assert_private_calls(fake_renderer)


def test_sync_empty_catalog(fake_renderer: FakeRenderer) -> None:
    result = sync_inventory()
    assert result["items"] == result["skipped"] == []
    assert (inventory_root() / "assets" / "videos").is_dir()
    assert (inventory_root() / "assets" / "images").is_dir()
    assert not catalog_path().exists()


def test_sync_chooses_first_existing_adoptable_location(
    tmp_path: Path, fake_renderer: FakeRenderer,
) -> None:
    photo = _source(tmp_path, name="photo.HEIC", kind="image")
    first_source = Path(photo["locations"][0]["path"])
    second_source = tmp_path / "photo.JPG"
    second_source.write_bytes(first_source.read_bytes())
    third_source = tmp_path / "later.PNG"
    third_source.write_bytes(first_source.read_bytes())
    photo["locations"].extend(
        {"path": str(source), "added_at": STAMP, "seen_at": STAMP}
        for source in (second_source, third_source)
    )
    _catalog(photo)

    result = sync_inventory()

    assert result["skipped"] == []
    assert result["items"][0]["path"] == "assets/images/photo-m1.jpg"
    assert (inventory_root() / result["items"][0]["path"]).readlink() == second_source


def test_sync_unsupported_reason_uses_first_existing_location(
    tmp_path: Path, fake_renderer: FakeRenderer,
) -> None:
    photo = _source(tmp_path, name="photo.HEIC", kind="image")
    photo["locations"].insert(0, {
        "path": str(tmp_path / "missing.JPG"), "added_at": STAMP, "seen_at": STAMP,
    })
    later = tmp_path / "photo.TIFF"
    later.write_bytes(b"synthetic later photo")
    photo["locations"].append({"path": str(later), "added_at": STAMP, "seen_at": STAMP})
    _catalog(photo)

    result = sync_inventory()

    assert result["items"] == []
    assert result["skipped"] == [{
        "id": "m1", "reason": "format not adopted by the HyperFrames inventory (.heic)",
    }]


@pytest.mark.parametrize("operation", ["sync", "search"])
@pytest.mark.parametrize("fresh", [False, True])
def test_operations_timeout_on_held_lock_without_changing_inventory(
    tmp_path: Path, fake_renderer: FakeRenderer, monkeypatch: pytest.MonkeyPatch,
    operation: str, fresh: bool,
) -> None:
    _catalog(_source(tmp_path))
    if fresh:
        sync_inventory()
    root = inventory_root()
    before = _snapshot(root) if fresh else None
    calls_before = len(fake_renderer.calls())
    monkeypatch.setattr(inventory, "_LOCK_TIMEOUT_S", 0.01)

    with inventory._inventory_lock(root):
        with pytest.raises(DependencyError) as caught:
            if operation == "sync":
                sync_inventory()
            else:
                search_inventory("desk", kind="video")
        assert caught.value.code == "media_inventory_locked"
        assert caught.value.suggested_action and "Wait" in caught.value.suggested_action
        assert root.exists() == fresh
        assert (root / "home").exists() == fresh
        assert not list(root.parent.glob(".media-inventory-*"))
        assert len(fake_renderer.calls()) == calls_before
        if fresh:
            assert _snapshot(root) == before
    # A timeout must close its descriptor and leave subsequent operations usable.
    assert search_inventory("desk", kind="video")[0]["id"] == "m1"


@pytest.mark.parametrize("operation", ["sync", "search"])
def test_lock_is_held_through_publish_and_renderer_resolution(
    tmp_path: Path, fake_renderer: FakeRenderer, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    _catalog(_source(tmp_path))
    publish = inventory._publish
    resolve = inventory._resolve
    published, resolutions = [], []
    lock_file = inventory_root().parent / ".media-inventory.lock"

    def assert_locked() -> None:
        assert lock_file.is_file()
        with pytest.raises(DependencyError) as caught:
            with inventory._inventory_lock(inventory_root(), timeout_s=0):
                pytest.fail("The operation released its inventory lock too early")
        assert caught.value.code == "media_inventory_locked"

    def checked_publish(staging: Path, root: Path) -> None:
        assert_locked()
        identity = lock_file.stat().st_ino
        publish(staging, root)
        assert lock_file.stat().st_ino == identity
        assert_locked()
        published.append(root)

    def checked_resolve(root: Path, install: RendererInstall, arguments: list[str]) -> dict:
        assert_locked()
        result = resolve(root, install, arguments)
        assert_locked()
        resolutions.append(arguments)
        return result

    monkeypatch.setattr(inventory, "_publish", checked_publish)
    monkeypatch.setattr(inventory, "_resolve", checked_resolve)
    if operation == "sync":
        sync_inventory()
    else:
        search_inventory("desk", kind="video")
    assert published == [inventory_root()]
    assert len(resolutions) == (1 if operation == "sync" else 2)
    # The lock must also be released after a successful complete operation.
    with inventory._inventory_lock(inventory_root(), timeout_s=0):
        pass


def test_search_racing_sync_times_out_then_uses_completed_inventory(
    tmp_path: Path, fake_renderer: FakeRenderer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = _source(tmp_path)
    _catalog(item)
    sync_inventory()
    item["description"] = {
        "text": "Updated desk", "tags": [], "by": "creator", "at": STAMP, "revision": 1,
    }
    _catalog(item)
    publish = inventory._publish
    entered, release = threading.Event(), threading.Event()
    monkeypatch.setattr(inventory, "_LOCK_TIMEOUT_S", 0.05)

    def paused_publish(staging: Path, root: Path) -> None:
        entered.set()
        assert release.wait(5), "The test did not release the paused sync"
        publish(staging, root)

    monkeypatch.setattr(inventory, "_publish", paused_publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        syncing = pool.submit(sync_inventory)
        try:
            assert entered.wait(5), "Sync did not reach publish"
            searching = pool.submit(search_inventory, "desk", kind="video")
            with pytest.raises(DependencyError) as caught:
                searching.result(timeout=5)
            assert caught.value.code == "media_inventory_locked"
            assert not syncing.done()
        finally:
            release.set()
        result = syncing.result(timeout=5)
    assert result["items"][0]["path"] == "assets/videos/updated-desk-m1.mp4"
    assert search_inventory("desk", kind="video")[0]["description"] == "updated desk m1"
    assert sum("--adopt" in call["argv"] for call in fake_renderer.calls()) == 2
    assert not list(inventory_root().parent.glob(".media-inventory-*"))


def test_search_syncs_missing_root_directly_without_creating_home_first(
    tmp_path: Path, fake_renderer: FakeRenderer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _catalog(_source(tmp_path))
    sync_body = inventory._sync_inventory
    synced = []

    def checked_sync(
        root: Path, plan: inventory._MirrorPlan, install: RendererInstall,
    ) -> dict:
        assert not root.exists()
        assert not (root / "home").exists()
        synced.append(root)
        return sync_body(root, plan, install)

    def fail_public_sync(**kwargs: object) -> dict:
        pytest.fail("Search called the public sync and would re-enter the file lock")

    monkeypatch.setattr(inventory, "_sync_inventory", checked_sync)
    monkeypatch.setattr(inventory, "sync_inventory", fail_public_sync)
    assert search_inventory("desk", kind="video")[0]["id"] == "m1"
    assert synced == [inventory_root()]


@pytest.mark.parametrize("mode", [
    "nonzero", "empty", "stderr_json", "invalid_json", "false", "nonobject",
    "missing_list", "invalid_entry",
])
def test_failed_adoption_preserves_previous_and_cleans_staging(
    tmp_path: Path, fake_renderer: FakeRenderer, mode: str,
) -> None:
    _catalog(_source(tmp_path))
    sync_inventory()
    previous = _snapshot(inventory_root())
    fake_renderer.configure(adopt=mode)

    with pytest.raises(ProcessingError) as caught:
        sync_inventory()

    assert caught.value.code == "media_inventory_failed"
    assert caught.value.suggested_action and "media inventory sync" in caught.value.suggested_action
    if mode == "nonzero":
        assert caught.value.returncode == 9
    assert _snapshot(inventory_root()) == previous
    assert not list(inventory_root().parent.glob(".media-inventory-*"))
    _assert_private_calls(fake_renderer)


@pytest.mark.parametrize("mode", ["unknown", "duplicate_path", "duplicate_id", "wrong_type",
                                 "omitted"])
def test_adoption_requires_complete_unique_catalog_mapping(
    tmp_path: Path, fake_renderer: FakeRenderer, mode: str,
) -> None:
    _catalog(_source(tmp_path), _source(tmp_path, 2))
    fake_renderer.configure(adopt=mode)
    with pytest.raises(ProcessingError) as caught:
        sync_inventory()
    assert caught.value.code == "media_inventory_mismatch"
    assert caught.value.suggested_action
    assert not inventory_root().exists()
    assert not list(inventory_root().parent.glob(".media-inventory-*"))


def test_successful_sync_replaces_old_inventory_without_leftovers(
    tmp_path: Path, fake_renderer: FakeRenderer,
) -> None:
    _catalog(_source(tmp_path))
    sync_inventory()
    old_link = next((inventory_root() / "assets" / "videos").iterdir())
    second = _source(tmp_path, 2, kind="image")
    _catalog(second)

    result = sync_inventory()

    assert [item["id"] for item in result["items"]] == ["m2"]
    assert not old_link.exists()
    assert not list(inventory_root().parent.glob(".media-inventory-*"))


def test_publish_rename_failure_restores_previous_inventory(
    tmp_path: Path, fake_renderer: FakeRenderer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _catalog(_source(tmp_path))
    sync_inventory()
    previous = _snapshot(inventory_root())
    replace = Path.replace

    def fail_publish(source: Path, target: Path) -> Path:
        if source.name.startswith(".media-inventory-") and target == inventory_root():
            if not source.name.startswith(".media-inventory-old-"):
                raise OSError("synthetic rename failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_publish)
    with pytest.raises(OSError, match="synthetic rename failure"):
        sync_inventory()
    assert _snapshot(inventory_root()) == previous
    assert not list(inventory_root().parent.glob(".media-inventory-*"))


def test_inventory_write_failure_cleans_staging_and_preserves_previous(
    tmp_path: Path, fake_renderer: FakeRenderer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _catalog(_source(tmp_path))
    sync_inventory()
    previous = _snapshot(inventory_root())

    def fail_write(*args: object, **kwargs: object) -> Path:
        raise OSError("synthetic write failure")

    monkeypatch.setattr(paths, "atomic_write_text", fail_write)
    with pytest.raises(OSError, match="synthetic write failure"):
        sync_inventory()
    assert _snapshot(inventory_root()) == previous
    assert not list(inventory_root().parent.glob(".media-inventory-*"))


@pytest.mark.parametrize("code", ["hyperframes_timeout", "hyperframes_exec_failed"])
def test_process_failure_cleans_staging_and_suggests_action(
    tmp_path: Path, fake_renderer: FakeRenderer, monkeypatch: pytest.MonkeyPatch, code: str,
) -> None:
    _catalog(_source(tmp_path))
    sync_inventory()
    previous = _snapshot(inventory_root())

    def fail_spawn(*args: object, **kwargs: object) -> None:
        raise ProcessingError("hyperframes", -1, "synthetic process failure", code=code)

    monkeypatch.setattr(inventory.process, "spawn", fail_spawn)
    with pytest.raises(ProcessingError) as caught:
        sync_inventory()
    assert caught.value.code == code
    assert caught.value.suggested_action
    assert _snapshot(inventory_root()) == previous
    assert not list(inventory_root().parent.glob(".media-inventory-*"))


@pytest.mark.parametrize("kind", ["video", "image"])
def test_search_maps_ordered_candidates_and_drops_global_and_unknown(
    tmp_path: Path, fake_renderer: FakeRenderer, kind: str,
) -> None:
    _catalog(_source(tmp_path, kind=kind), _source(tmp_path, 2, kind=kind))
    fake_renderer.configure(extra_candidates=True)

    matches = search_inventory("desk shot", kind=kind, install=fake_renderer.install)

    assert [item["id"] for item in matches] == ["m2", "m1"]
    assert [item["hf_id"] for item in matches] == [f"{kind}_002", f"{kind}_001"]
    assert [item["description"] for item in matches] == ["desk 2 m2", "desk 1 m1"]
    assert [item["score"] for item in matches] == [1, 0.9]
    assert all(item["duration_s"] == (2 if kind == "video" else None) for item in matches)
    assert all((item["width"], item["height"]) == (64, 36) for item in matches)
    calls = fake_renderer.calls()
    assert len(calls) == 2 and "--adopt" in calls[0]["argv"]
    assert calls[1]["argv"] == [
        "media-use", "resolve", "--type", kind, "--intent", "desk shot", "--candidates",
        "--local-only", "--json", "--project", str(inventory_root()),
    ]
    _assert_private_calls(fake_renderer)


def test_search_syncs_missing_and_stale_but_reuses_fresh_and_reordered_catalog(
    tmp_path: Path, fake_renderer: FakeRenderer,
) -> None:
    first, second = _source(tmp_path), _source(tmp_path, 2)
    _catalog(first)
    assert [item["id"] for item in search_inventory("desk", kind="video")] == ["m1"]
    assert len(fake_renderer.calls()) == 2
    first_digest = json.loads((inventory_root() / "inventory.json").read_text())["plan_sha256"]
    search_inventory("desk", kind="video")
    assert len(fake_renderer.calls()) == 3

    _catalog(first, second)
    assert [item["id"] for item in search_inventory("desk", kind="video")] == ["m2", "m1"]
    second_digest = json.loads((inventory_root() / "inventory.json").read_text())["plan_sha256"]
    assert second_digest != first_digest
    assert len(fake_renderer.calls()) == 5

    _catalog(second, first)
    search_inventory("desk", kind="video")
    assert len(fake_renderer.calls()) == 6
    first["sha256"] = "f" * 64
    _catalog(first, second)
    search_inventory("desk", kind="video")
    assert len(fake_renderer.calls()) == 8

    _catalog(first)
    assert [item["id"] for item in search_inventory("desk", kind="video")] == ["m1"]
    assert len(fake_renderer.calls()) == 10
    assert sum("--adopt" in call["argv"] for call in fake_renderer.calls()) == 4
    _assert_private_calls(fake_renderer)


@pytest.mark.parametrize("change", ["description", "location", "deleted"])
def test_search_resyncs_changed_description_location_or_availability(
    tmp_path: Path, fake_renderer: FakeRenderer, change: str,
) -> None:
    item = _source(tmp_path, text="Original desk")
    source = Path(item["locations"][0]["path"])
    _catalog(item)
    before = sync_inventory()
    selected_source = source
    if change == "description":
        item["description"]["text"] = "Updated desk"
        _catalog(item)
    elif change == "location":
        selected_source = tmp_path / "moved.MP4"
        selected_source.write_bytes(source.read_bytes())
        item["locations"] = [{"path": str(selected_source), "added_at": STAMP, "seen_at": STAMP}]
        _catalog(item)
    else:
        source.unlink()

    matches = search_inventory("desk", kind="video")

    after = json.loads((inventory_root() / "inventory.json").read_text())
    assert after["plan_sha256"] != before["plan_sha256"]
    assert load_catalog()["items"][0]["sha256"] == item["sha256"]
    assert sum("--adopt" in call["argv"] for call in fake_renderer.calls()) == 2
    if change == "deleted":
        assert matches == []
        assert after["items"] == []
        assert after["skipped"] == [{"id": "m1", "reason": "no file on disk"}]
    else:
        assert matches[0]["id"] == "m1"
        assert after["skipped"] == []
        link = inventory_root() / after["items"][0]["path"]
        assert link.readlink() == selected_source
        assert matches[0]["description"] == (
            "updated desk m1" if change == "description" else "original desk m1"
        )
    # The same plan stays fresh after the resync, even for an empty inventory.
    search_inventory("desk", kind="video")
    assert len(fake_renderer.calls()) == 4
    assert sum("--adopt" in call["argv"] for call in fake_renderer.calls()) == 2
    _assert_private_calls(fake_renderer)


@pytest.mark.parametrize("extension", ["MP4", "MKV"])
def test_search_resyncs_newly_available_files_and_changed_skip_reasons(
    tmp_path: Path, fake_renderer: FakeRenderer, extension: str,
) -> None:
    item = _source(tmp_path, name=f"desk.{extension}")
    source = Path(item["locations"][0]["path"])
    content = source.read_bytes()
    _catalog(item)
    source.unlink()
    before = sync_inventory()
    assert before["skipped"] == [{"id": "m1", "reason": "no file on disk"}]
    source.write_bytes(content)

    matches = search_inventory("desk", kind="video")

    after = json.loads((inventory_root() / "inventory.json").read_text())
    assert after["plan_sha256"] != before["plan_sha256"]
    assert sum("--adopt" in call["argv"] for call in fake_renderer.calls()) == 2
    if extension == "MP4":
        assert matches[0]["id"] == "m1"
        assert after["skipped"] == []
    else:
        assert matches == []
        assert after["skipped"] == [{
            "id": "m1", "reason": "format not adopted by the HyperFrames inventory (.mkv)",
        }]


@pytest.mark.parametrize("mode", ["nonzero", "empty", "invalid_json", "false", "missing_list",
                                 "invalid_entry"])
def test_search_failure_has_suggested_action(
    tmp_path: Path, fake_renderer: FakeRenderer, mode: str,
) -> None:
    _catalog(_source(tmp_path))
    sync_inventory()
    fake_renderer.configure(search=mode)
    with pytest.raises(ProcessingError) as caught:
        search_inventory("desk", kind="video")
    assert caught.value.code == "media_inventory_failed"
    assert caught.value.suggested_action
    assert len(fake_renderer.calls()) == 2


def test_invalid_kind_does_not_run_renderer(fake_renderer: FakeRenderer) -> None:
    with pytest.raises(ValidationError, match="choose video or image"):
        search_inventory("desk", kind="audio")
    assert fake_renderer.calls() == []
    assert not inventory_root().exists()


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_inventory_sync_and_search(
    tmp_path: Path, fake_renderer: FakeRenderer, capsys: pytest.CaptureFixture[str], as_json: bool,
) -> None:
    video = _source(tmp_path)
    image = _source(tmp_path, 2, kind="image")
    missing = _source(tmp_path, 3)
    Path(missing["locations"][0]["path"]).unlink()
    _catalog(video, image, missing)
    flag = ["--json"] if as_json else []

    assert cli.main(["media", "inventory", "sync", *flag]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    if as_json:
        result = json.loads(output.out)
        assert {item["id"] for item in result["items"]} == {"m1", "m2"}
        assert result["skipped"] == [{"id": "m3", "reason": "no file on disk"}]
    else:
        assert output.out == "Mirrored 2, skipped 1.\n"
    for kind, item_id in (("video", "m1"), ("image", "m2")):
        assert cli.main(["media", "inventory", "search", "--type", kind, "desk", *flag]) == 0
        output = capsys.readouterr()
        assert output.err == ""
        if as_json:
            assert [item["id"] for item in json.loads(output.out)] == [item_id]
        else:
            assert output.out == f"{item_id} {'1.0' if kind == 'video' else '0.9'} desk " + (
                f"{item_id[1:]} {item_id}\n"
            )
    assert len(fake_renderer.calls()) == 3
    _assert_private_calls(fake_renderer)


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_empty_search_and_renderer_failure(
    fake_renderer: FakeRenderer, capsys: pytest.CaptureFixture[str], as_json: bool,
) -> None:
    flag = ["--json"] if as_json else []
    assert cli.main(["media", "inventory", "search", "--type", "image", "desk", *flag]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == [] if as_json else output.out == "No candidates.\n"
    fake_renderer.configure(adopt="nonzero")
    assert cli.main(["media", "inventory", "sync", *flag]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    if as_json:
        error = json.loads(output.err)["error"]
        assert error["code"] == "media_inventory_failed"
        assert "media inventory sync" in error["suggested_action"]
    else:
        assert "error:" in output.err and "fix:" in output.err


@pytest.mark.render
def test_real_pinned_inventory_mapping_and_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_storage: str | None,
) -> None:
    """Opt-in adoption and search using two synthetic media files and the pinned tool."""
    # Keep the configured renderer cache for installation before isolating the inventory.
    with monkeypatch.context() as installation:
        if isolated_storage:
            installation.setenv("YAP_STUDIO_CACHE", isolated_storage)
        else:
            installation.delenv("YAP_STUDIO_CACHE")
        install = hyperframes.ensure_installed()
    video, image = tmp_path / "desk-video.MP4", tmp_path / "desk-image.PNG"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:size=64x36:rate=30:duration=1",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
    ], timeout_s=60)
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:size=64x36", "-frames:v", "1", str(image),
    ], timeout_s=60)
    added = add_paths([video, image])
    assert len(added["added"]) == 2
    catalog = load_catalog()
    ids = {item["kind"]: item["id"] for item in catalog["items"]}

    result = sync_inventory(install=install)

    assert result["skipped"] == []
    assert {item["id"] for item in result["items"]} == set(ids.values())
    for item in result["items"]:
        assert item["hf_id"].startswith(item["type"] + "_")
        assert (inventory_root() / item["path"]).is_symlink()
    for kind in ("video", "image"):
        matches = search_inventory("desk", kind=kind, install=install)
        assert ids[kind] in {item["id"] for item in matches}
        assert all((item["width"], item["height"]) == (64, 36) for item in matches)
