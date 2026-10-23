"""Recoverable watch pack publication and strict receipt ownership."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from test_watch_pack import media as media
from test_watch_pack import prepared as prepared

from yap_studio import cli, paths
from yap_studio.errors import OutputPathError
from yap_studio.receipt import sha256_file
from yap_studio.watch import build_pack, load_pack
from yap_studio.watch import pack as pack_module

PENDING_SCHEMA = "yap-studio/watch-pack-pending@1"


def _snapshot(directory: Path) -> dict[str, bytes | str | None]:
    """Record files, directories, and link destinations without reading through links."""
    return {
        item.relative_to(directory).as_posix(): (
            str(item.readlink())
            if item.is_symlink()
            else None
            if item.is_dir()
            else item.read_bytes()
        )
        for item in directory.rglob("*")
    }


def _journal(directory: Path) -> dict:
    """Describe a publication paused before any existing output was replaced."""
    outputs = [
        *load_pack(directory)["outputs"],
        {"path": "pack.json", "sha256": sha256_file(directory / "pack.json")},
    ]
    return {
        "schema": PENDING_SCHEMA,
        "outputs": [
            {**item, "old_sha256": item["sha256"], "new_sha256": item["sha256"]}
            for item in outputs
        ],
    }


def _write_journal(directory: Path, data: dict) -> None:
    (directory / "pack.pending.json").write_text(json.dumps(data), encoding="utf-8")


def _assert_pack_refused_untouched(
    video: Path, directory: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Require a CLI refusal before any pack file changes."""
    before = _snapshot(directory)
    assert cli.main([
        "watch", "pack", str(video), "--out", str(directory), "--asr", "none", "--json",
    ]) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "unsafe_path"
    assert _snapshot(directory) == before


def test_pending_index_cannot_claim_modified_output(
    media: dict[str, Path],
    prepared: tuple[dict, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "pack"
    shutil.copytree(prepared[1], out)
    pending = _journal(out)
    modified = b"keep this edited guide\n"
    (out / "pack.md").write_bytes(modified)
    index = load_pack(out)
    for item in index["outputs"]:
        if item["path"] == "pack.md":
            item["sha256"] = sha256_file(out / "pack.md")
    (out / "pack.json").write_text(json.dumps(index), encoding="utf-8")
    for item in pending["outputs"]:
        if item["path"] == "pack.json":
            item["new_sha256"] = sha256_file(out / "pack.json")
    _write_journal(out, pending)
    _assert_pack_refused_untouched(media["video"], out, capsys)
    assert (out / "pack.md").read_bytes() == modified


def test_pending_index_hash_must_match_journal(
    media: dict[str, Path],
    prepared: tuple[dict, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "pack"
    shutil.copytree(prepared[1], out)
    _write_journal(out, _journal(out))
    (out / "pack.json").write_text(json.dumps(load_pack(out)), encoding="utf-8")
    _assert_pack_refused_untouched(media["video"], out, capsys)


@pytest.mark.parametrize("name", ["pack.md", "pack.json"])
def test_pending_present_file_needs_journal_entry(
    media: dict[str, Path],
    prepared: tuple[dict, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    out = tmp_path / "pack"
    shutil.copytree(prepared[1], out)
    pending = _journal(out)
    pending["outputs"] = [item for item in pending["outputs"] if item["path"] != name]
    _write_journal(out, pending)
    _assert_pack_refused_untouched(media["video"], out, capsys)


@pytest.mark.parametrize(
    "name,old_hash", [("pack.md", "0" * 64), ("pack.md", None), ("verify.json", "0" * 64)]
)
def test_pending_old_hashes_must_match_old_index(
    media: dict[str, Path],
    prepared: tuple[dict, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    old_hash: str | None,
) -> None:
    out = tmp_path / "pack"
    shutil.copytree(prepared[1], out)
    pending = _journal(out)
    if name == "verify.json":
        pending["outputs"].append(
            {"path": name, "sha256": old_hash, "old_sha256": old_hash, "new_sha256": None}
        )
    else:
        for item in pending["outputs"]:
            if item["path"] == name:
                item.update(sha256=old_hash, old_sha256=old_hash)
    _write_journal(out, pending)
    _assert_pack_refused_untouched(media["video"], out, capsys)


@pytest.mark.parametrize("point", ["journal", "second-output", "second-strip", "index", "cleanup"])
def test_interrupted_publication_resumes_identically(
    media: dict[str, Path],
    prepared: tuple[dict, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    point: str,
) -> None:
    out = tmp_path / "pack"
    shutil.copytree(prepared[1], out)
    before = _snapshot(out)
    replace_file = paths.os.replace
    copy_file = pack_module.shutil.copyfile
    copied = 0

    def replace_then_interrupt(source, target):
        replace_file(source, target)
        if Path(target) == out / ("pack.pending.json" if point == "journal" else "pack.json"):
            raise OSError("interrupted publication")

    def copy_or_interrupt(source, target):
        nonlocal copied
        copied += 1
        if (
            point == "second-output" and copied == 2
            or point == "second-strip" and Path(source).name == "strip-002.png"
            or point == "index" and Path(source).name == "pack.json"
        ):
            Path(target).write_bytes(b"partial copy")
            raise OSError("interrupted publication")
        return copy_file(source, target)

    with monkeypatch.context() as patch:
        if point in {"journal", "cleanup"}:
            patch.setattr(paths.os, "replace", replace_then_interrupt)
        else:
            patch.setattr(pack_module.shutil, "copyfile", copy_or_interrupt)
        with pytest.raises(OSError, match="interrupted publication"):
            build_pack(
                media["video"], edl=media["edl"], out=out, every_s=0.5, words=media["words"]
            )

    pending = json.loads((out / "pack.pending.json").read_text())
    assert pending["schema"] == PENDING_SCHEMA
    assert (out / "pack.json").exists() == (point in {"journal", "cleanup"})
    if point == "journal":
        after = _snapshot(out)
        after.pop("pack.pending.json")
        assert after == before
    elif point in {"second-output", "second-strip"}:
        failed = "pack.md" if point == "second-output" else "strips/strip-002.png"
        assert (out / failed).read_bytes() == before[failed]
    for item in pending["outputs"]:
        name = item["path"]
        assert item["sha256"] == item["old_sha256"]
        assert item["old_sha256"] == (
            sha256_file(prepared[1] / name) if name in before else None
        )
        assert item["new_sha256"] is not None
    assert any(item["old_sha256"] is None for item in pending["outputs"])

    fresh = tmp_path / "fresh"
    expected = build_pack(
        media["video"], edl=media["edl"], out=fresh, every_s=0.5, words=media["words"]
    )
    assert {item["path"] for item in pending["outputs"]} == {
        name for name, content in _snapshot(fresh).items() if content is not None
    }
    assert all(
        item["new_sha256"] == sha256_file(fresh / item["path"])
        for item in pending["outputs"]
    )
    # The retry may itself fail after replacing its journal. It must still be resumable.
    if point == "second-strip":
        with monkeypatch.context() as patch:
            patch.setattr(paths.os, "replace", replace_then_interrupt)
            # Interrupt after writing the retry's journal, while no index exists.
            point = "journal"
            with pytest.raises(OSError, match="interrupted publication"):
                build_pack(
                    media["video"], edl=media["edl"], out=out,
                    every_s=0.5, words=media["words"],
                )
    assert cli.main([
        "watch", "pack", str(media["video"]), "--edl", str(media["edl"]),
        "--out", str(out), "--every", "0.5", "--words", str(media["words"]), "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == expected
    assert not (out / "pack.pending.json").exists()
    assert _snapshot(out) == _snapshot(fresh)


@pytest.mark.parametrize("receipt", ["pack.json", "pack.pending.json"])
@pytest.mark.parametrize("name", ["notes.txt", "keep/me.json"])
def test_forged_ownership_refused_without_deletion(
    media: dict[str, Path],
    prepared: tuple[dict, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    receipt: str,
    name: str,
) -> None:
    out = tmp_path / "pack"
    shutil.copytree(prepared[1], out)
    foreign = out / name
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"keep these unrelated notes\n")
    digest = sha256_file(foreign)
    data = load_pack(out) if receipt == "pack.json" else _journal(out)
    entry = {"path": name, "sha256": digest}
    if receipt == "pack.pending.json":
        entry.update(old_sha256=digest, new_sha256=None)
    data["outputs"].append(entry)
    (out / receipt).write_text(json.dumps(data), encoding="utf-8")
    before = _snapshot(out)
    assert cli.main(
        ["watch", "pack", str(media["video"]), "--out", str(out), "--asr", "none", "--json"]
    ) == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "unsafe_path"
    assert _snapshot(out) == before
    assert foreign.read_bytes() == b"keep these unrelated notes\n"


@pytest.mark.parametrize(
    "name",
    [
        "../outside.json", "/outside.json", "unknown.json", "strips/strip-01.png",
        "strips/strip-001.png/child", "strips/../motion.json", "./pack.md",
        "pack.pending.json",
    ],
)
def test_unsafe_journal_paths_refused_untouched(
    media: dict[str, Path],
    prepared: tuple[dict, Path],
    tmp_path: Path,
    name: str,
) -> None:
    out = tmp_path / "pack"
    shutil.copytree(prepared[1], out)
    pending = _journal(out)
    pending["outputs"].append(
        {"path": name, "sha256": None, "old_sha256": None, "new_sha256": "0" * 64}
    )
    _write_journal(out, pending)
    before = _snapshot(out)
    with pytest.raises(OutputPathError) as exc:
        build_pack(media["video"], out=out, asr="none")
    assert exc.value.code == "unsafe_path"
    assert _snapshot(out) == before


@pytest.mark.parametrize("linked", ["folder", "strips", "output", "index", "journal", "temporary"])
def test_symlinks_refused_without_touching_destination(
    media: dict[str, Path],
    prepared: tuple[dict, Path],
    tmp_path: Path,
    linked: str,
) -> None:
    out = tmp_path / "pack"
    outside = tmp_path / "outside"
    shutil.copytree(prepared[1], outside)
    shutil.copytree(prepared[1], out)
    if linked == "folder":
        shutil.rmtree(out)
        out.symlink_to(outside, target_is_directory=True)
    elif linked == "strips":
        shutil.rmtree(out / "strips")
        (out / "strips").symlink_to(outside / "strips", target_is_directory=True)
    else:
        name = {
            "output": "pack.md", "index": "pack.json", "journal": "pack.pending.json",
            "temporary": ".yap_watch_tmp_abc12345",
        }[linked]
        if linked == "journal":
            _write_journal(outside, _journal(outside))
        elif linked == "temporary":
            (outside / name).write_bytes(b"external file")
        (out / name).unlink(missing_ok=True)
        (out / name).symlink_to(outside / name)
    before, external = _snapshot(out), _snapshot(outside)
    with pytest.raises(OutputPathError) as exc:
        build_pack(media["video"], out=out, asr="none")
    assert exc.value.code == "unsafe_path"
    assert _snapshot(out) == before
    assert _snapshot(outside) == external


def test_only_strips_directory_is_allowed(media: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "pack"
    (out / "unrelated").mkdir(parents=True)
    with pytest.raises(OutputPathError) as exc:
        build_pack(media["video"], out=out, asr="none")
    assert exc.value.code == "unsafe_path"
    assert _snapshot(out) == {"unrelated": None}


@pytest.mark.parametrize(
    "malformed", ["json", "object", "schema", "outputs", "entry", "digest", "alias", "duplicate"]
)
def test_malformed_journal_refused_untouched(
    media: dict[str, Path],
    prepared: tuple[dict, Path],
    tmp_path: Path,
    malformed: str,
) -> None:
    out = tmp_path / "pack"
    shutil.copytree(prepared[1], out)
    pending = _journal(out)
    if malformed == "schema":
        pending["schema"] = "unsupported"
    elif malformed == "outputs":
        pending["outputs"] = {}
    elif malformed == "entry":
        pending["outputs"][0].pop("old_sha256")
    elif malformed == "digest":
        pending["outputs"][0]["new_sha256"] = "invalid hash"
    elif malformed == "alias":
        pending["outputs"][0]["sha256"] = "0" * 64
    elif malformed == "duplicate":
        pending["outputs"].append(pending["outputs"][0])
    _write_journal(out, pending)
    if malformed in {"json", "object"}:
        (out / "pack.pending.json").write_text("{" if malformed == "json" else "[]")
    before = _snapshot(out)
    with pytest.raises(OutputPathError) as exc:
        build_pack(media["video"], out=out, asr="none")
    assert exc.value.code == "unsafe_path"
    assert _snapshot(out) == before


def test_interrupted_deletion_resumes_identically(
    media: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "pack"
    dense = build_pack(
        media["video"], edl=media["edl"], out=out, every_s=0.5, words=media["words"]
    )
    stale = out / "strips/strip-003.png"
    stale_hash = sha256_file(stale)
    unlink = Path.unlink

    def delete_then_interrupt(target, *args, **kwargs):
        unlink(target, *args, **kwargs)
        if target == stale:
            raise OSError("interrupted deletion")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", delete_then_interrupt)
        with pytest.raises(OSError, match="interrupted deletion"):
            build_pack(media["video"], edl=media["edl"], out=out, asr="none")
    assert not stale.exists()
    assert not (out / "pack.json").exists()
    pending = json.loads((out / "pack.pending.json").read_text())
    entries = {item["path"]: item for item in pending["outputs"]}
    assert entries["strips/strip-003.png"] == {
        "path": "strips/strip-003.png", "sha256": stale_hash,
        "old_sha256": stale_hash, "new_sha256": None,
    }
    assert {name for name, item in entries.items() if item["new_sha256"] is None} == {
        *(strip["image"] for strip in dense["timeline"]["strips"][2:]),
        "transcript.json", "transcript.txt",
    }
    fresh = tmp_path / "fresh"
    expected = build_pack(media["video"], edl=media["edl"], out=fresh, asr="none")
    result = build_pack(media["video"], edl=media["edl"], out=out, asr="none")
    assert result == expected
    assert _snapshot(out) == _snapshot(fresh)


@pytest.mark.parametrize("change", ["modified", "symlink"])
def test_deletions_recheck_current_file(
    media: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    out = tmp_path / "pack"
    build_pack(media["video"], edl=media["edl"], out=out, every_s=0.5, asr="none")
    stale = out / "strips/strip-003.png"
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"external image")
    copy_file = pack_module.shutil.copyfile

    def copy_then_change(source, target):
        result = copy_file(source, target)
        if Path(source).name == "strip-002.png":
            if change == "symlink":
                stale.unlink()
                stale.symlink_to(outside)
            else:
                stale.write_bytes(b"changed externally")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(pack_module.shutil, "copyfile", copy_then_change)
        with pytest.raises(OutputPathError) as exc:
            build_pack(media["video"], edl=media["edl"], out=out, asr="none")
    assert exc.value.code == "unsafe_path"
    assert outside.read_bytes() == b"external image"
    if change == "symlink":
        assert stale.is_symlink() and stale.readlink() == outside
    else:
        assert stale.read_bytes() == b"changed externally"


def test_publication_temporaries_stay_inside_pack(
    media: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "pack"
    make_temporary = pack_module.tempfile.mkstemp
    created = []

    def observe(*args, **kwargs):
        descriptor, name = make_temporary(*args, **kwargs)
        temporary = Path(name)
        created.append(temporary)
        assert temporary.parent.resolve() == out.resolve()
        assert re.fullmatch(r"\.yap_watch_tmp_[a-z0-9_]{8}", temporary.name)
        return descriptor, name

    monkeypatch.setattr(pack_module.tempfile, "mkstemp", observe)
    build_pack(media["video"], edl=media["edl"], out=out, asr="none")
    assert created and not any(item.exists() for item in created)
    assert not list(out.glob(".yap_watch_tmp_*"))


@pytest.mark.parametrize("state", ["empty", "published", "pending", "pending-only"])
def test_leftover_temporaries_are_journaled_then_removed(
    media: dict[str, Path],
    prepared: tuple[dict, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    out = tmp_path / "pack"
    if state == "empty":
        out.mkdir()
    else:
        shutil.copytree(prepared[1], out)
        if state in {"pending", "pending-only"}:
            _write_journal(out, _journal(out))
        if state == "pending-only":
            (out / "pack.json").unlink()
            (out / "motion.json").unlink()
    leftovers = [out / ".yap_watch_tmp_abc0123_", out / ".yap_watch_tmp_z9y8x7w6"]
    for item in leftovers:
        item.write_bytes(b"interrupted write")
    index_present = (out / "pack.json").exists()
    hashes = {item.name: sha256_file(item) for item in leftovers}
    unlink = Path.unlink
    removed = []

    def observe_delete(target, *args, **kwargs):
        if target in leftovers:
            journal = json.loads((out / "pack.pending.json").read_text())
            entries = {item["path"]: item for item in journal["outputs"]}
            assert entries[target.name] == {
                "path": target.name, "sha256": hashes[target.name],
                "old_sha256": hashes[target.name], "new_sha256": None,
            }
            # Cleanup happens before the old index is removed or outputs are replaced.
            assert (out / "pack.json").exists() == index_present
            assert all((out / name).read_bytes() == (prepared[1] / name).read_bytes()
                       for name in ("pack.md", "strips/strip-001.png") if (out / name).exists())
            removed.append(target)
        return unlink(target, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", observe_delete)
    result = build_pack(media["video"], edl=media["edl"], out=out, words=media["words"])
    assert result == prepared[0]
    assert set(removed) == set(leftovers)
    assert _snapshot(out) == _snapshot(prepared[1])


@pytest.mark.parametrize(
    "name",
    [
        ".yap_watch_tmp_abc1234", ".yap_watch_tmp_abc123456", ".yap_watch_tmp_ABC12345",
        ".yap_watch_tmp_abc12345.png", ".yap_watch_tmp_", "strips/.yap_watch_tmp_abc12345",
    ],
)
def test_similar_temporary_names_stay_foreign(
    media: dict[str, Path], prepared: tuple[dict, Path], tmp_path: Path, name: str
) -> None:
    out = tmp_path / "pack"
    shutil.copytree(prepared[1], out)
    (out / name).write_bytes(b"keep this foreign file")
    before = _snapshot(out)
    with pytest.raises(OutputPathError) as exc:
        build_pack(media["video"], out=out, asr="none")
    assert exc.value.code == "unsafe_path"
    assert _snapshot(out) == before


def test_interrupted_first_publication_resumes(
    media: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "pack"
    copy_file = pack_module.shutil.copyfile

    def copy_or_interrupt(source, target):
        if Path(source).name == "pack.md":
            Path(target).write_bytes(b"partial copy")
            raise OSError("interrupted first publication")
        return copy_file(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(pack_module.shutil, "copyfile", copy_or_interrupt)
        with pytest.raises(OSError, match="interrupted first publication"):
            build_pack(media["video"], edl=media["edl"], out=out, asr="none")
    assert not (out / "pack.json").exists()
    pending = json.loads((out / "pack.pending.json").read_text())
    assert all(item["old_sha256"] is None for item in pending["outputs"])
    fresh = tmp_path / "fresh"
    expected = build_pack(media["video"], edl=media["edl"], out=fresh, asr="none")
    result = build_pack(media["video"], edl=media["edl"], out=out, asr="none")
    assert result == expected
    assert _snapshot(out) == _snapshot(fresh)
