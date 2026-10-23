"""Guarded path and atomic write tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from yap_studio import paths
from yap_studio.errors import InputError, OutputPathError
from yap_studio.paths import (
    atomic_output,
    atomic_write_text,
    operation,
    validate_input,
    validate_output,
    validate_output_dir,
)


def _report(status: str) -> str:
    return (
        '{"schema": "yap-studio/verify-report@1", '
        f'"status": "{status}", "media": {{}}, "summary": {{}}, '
        '"checks": {}, "findings": []}'
    )


@pytest.mark.parametrize("schema", ["yap-studio/stage-receipt@1", "yap-studio/stage-pending@1"])
def test_fillers_receipt_owns_root_removals_with_matching_hash(tmp_path: Path,
                                                             schema: str) -> None:
    target = tmp_path / "removals.json"
    target.write_text('{"schema":"yap-studio/removals@1","removals":[]}')
    folder = tmp_path / "fillers"
    folder.mkdir()
    owner = folder / "receipt.json"
    owner.write_text(json.dumps({"schema": schema, "outputs": [{
        "path": "removals.json", "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }]}))
    with operation():
        assert validate_output(target, owner_receipts=(owner,)) == target


@pytest.mark.parametrize("stage,filename,listed", [
    ("cut", "removals.json", "../removals.json"),
    ("cut", "removals.json", "removals.json"),
    ("fillers", "edit.json", "../edit.json"),
    ("fillers", "edit.json", "edit.json"),
    ("fillers", "project.json", "../project.json"),
    ("fillers", "project.json", "project.json"),
])
def test_stage_receipt_refuses_unlisted_root_outputs(tmp_path: Path, stage: str,
                                                   filename: str, listed: str) -> None:
    target = tmp_path / filename
    target.write_text('{"schema":"yap-studio/removals@1","removals":[]}')
    folder = tmp_path / stage
    folder.mkdir()
    owner = folder / "receipt.json"
    owner.write_text(json.dumps({"schema": "yap-studio/stage-receipt@1", "outputs": [{
        "path": listed, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }]}))
    with operation(), pytest.raises(OutputPathError):
        validate_output(target, owner_receipts=(owner,))


def test_fillers_root_output_hash_mismatch_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "removals.json"
    target.write_text('{"schema":"yap-studio/removals@1","removals":[]}')
    folder = tmp_path / "fillers"
    folder.mkdir()
    owner = folder / "receipt.json"
    owner.write_text(json.dumps({"schema": "yap-studio/stage-receipt@1", "outputs": [{
        "path": "removals.json", "sha256": "0" * 64,
    }]}))
    with operation(), pytest.raises(OutputPathError):
        validate_output(target, owner_receipts=(owner,))


def test_operation_records_are_empty_nested_and_restored(tmp_path: Path) -> None:
    handles = (paths._INPUT_REALPATHS, paths._OUTPUT_REALPATHS, paths._CREATED_REALPATHS)
    for handle in handles:
        assert not handle.get()
        paths._record(handle, str(tmp_path / "outside"))
        assert not handle.get()
    with operation():
        for handle in handles:
            assert isinstance(handle.get(), frozenset)
            assert handle.get() == frozenset()
            paths._record(handle, str(tmp_path / "outer"))
        for exceptional in (False, True):
            try:
                with operation():
                    for handle in handles:
                        assert handle.get() == set()
                        paths._record(handle, str(tmp_path / "inner"))
                    if exceptional:
                        raise RuntimeError("stop")
            except RuntimeError:
                pass
            for handle in handles:
                assert handle.get() == {str(tmp_path / "outer")}
    for handle in handles:
        assert not handle.get()
    with pytest.raises(RuntimeError), operation():
        paths._record(paths._INPUT_REALPATHS, str(tmp_path / "exception"))
        raise RuntimeError("stop")
    assert not paths._INPUT_REALPATHS.get()


def test_operation_records_do_not_cross_threads(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)

    def run(index: int) -> None:
        assert not paths._INPUT_REALPATHS.get()
        with operation():
            own = str(tmp_path / str(index))
            for handle in (paths._INPUT_REALPATHS, paths._OUTPUT_REALPATHS,
                           paths._CREATED_REALPATHS):
                paths._record(handle, own)
            barrier.wait(timeout=10)
            for handle in (paths._INPUT_REALPATHS, paths._OUTPUT_REALPATHS,
                           paths._CREATED_REALPATHS):
                assert handle.get() == {own}
        assert not paths._INPUT_REALPATHS.get()

    with operation(), ThreadPoolExecutor(max_workers=2) as executor:
        paths._record(paths._INPUT_REALPATHS, str(tmp_path / "parent"))
        futures = [executor.submit(run, index) for index in range(2)]
        for future in futures:
            future.result(timeout=15)


@pytest.mark.parametrize("kind", ["nul", "missing", "directory"])
def test_validate_input_refusals(tmp_path: Path, kind: str) -> None:
    target = str(tmp_path / "missing")
    if kind == "nul":
        target += "\x00"
    elif kind == "directory":
        target = str(tmp_path)
    with operation(), pytest.raises(InputError) as caught:
        validate_input(target)
    assert caught.value.path == target
    assert caught.value.code == "invalid_input"
    assert caught.value.reason == (
        "the path contains a NUL character" if kind == "nul" else "no file at this path"
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode 000 files")
def test_validate_input_unreadable(tmp_path: Path) -> None:
    target = tmp_path / "unreadable.mov"
    target.write_bytes(b"synthetic")
    target.chmod(0)
    try:
        with pytest.raises(InputError) as caught:
            validate_input(target)
        assert caught.value.reason.startswith("cannot open it for reading (")
        assert "Permission denied" in caught.value.reason
    finally:
        target.chmod(0o600)


def test_validate_input_resolves_and_records_unless_output(tmp_path: Path) -> None:
    source = tmp_path / "source.mov"
    source.write_bytes(b"synthetic")
    alias = tmp_path / "alias.mov"
    alias.symlink_to(source)
    resolved = os.path.realpath(source)
    with operation():
        assert validate_input(alias) == Path(resolved)
        assert paths._INPUT_REALPATHS.get() == {resolved}
    with operation():
        validate_output(source.with_suffix(".json"))
        paths._record(paths._OUTPUT_REALPATHS, resolved)
        assert validate_input(alias) == Path(resolved)
        assert paths._INPUT_REALPATHS.get() == set()


@pytest.mark.parametrize("tail,message", [
    ("nul\x00.json", "Output path contains a NUL character."),
    ("child/../output.json",
     "Output path uses '..' to step up a folder; give the path without it."),
    ("child\\..\\output.json",
     "Output path uses '..' to step up a folder; give the path without it."),
])
def test_guard_invalid_spelling(tmp_path: Path, tail: str, message: str) -> None:
    with pytest.raises(OutputPathError) as caught:
        paths._guard_write(str(tmp_path) + "/" + tail)
    assert caught.value.code == "invalid_output_path"
    assert str(caught.value) == message


@pytest.mark.parametrize("alias_kind", ["realpath", "symlink", "hardlink"])
def test_guard_input_alias_precedes_other_checks(tmp_path: Path, alias_kind: str) -> None:
    source = tmp_path / "source.json"
    source.write_text(_report("old"))
    target = tmp_path / "alias.json"
    if alias_kind == "symlink":
        target.symlink_to(source)
    elif alias_kind == "hardlink":
        os.link(source, target)
    else:
        target = source
    with operation(), pytest.raises(OutputPathError) as caught:
        validate_input(source)
        paths._guard_write(target)
    assert caught.value.code == "invalid_output_path"
    assert str(caught.value) == "Output path is an input file of this command."


def test_guard_plain_symlink(tmp_path: Path) -> None:
    target = tmp_path / "alias.json"
    target.symlink_to(tmp_path / "missing.json")
    with pytest.raises(OutputPathError) as caught:
        paths._guard_write(target)
    assert caught.value.code == "unsafe_path"
    assert str(caught.value) == "Output path is a symlink; give the real file path."


@pytest.mark.parametrize("spelling", ["x.json/.", "x.json/./", "x.json//", "./x.json"])
@pytest.mark.parametrize("kind", ["foreign", "symlink", "hardlink"])
@pytest.mark.parametrize("atomic", [False, True])
def test_path_normalized_spelling_preserves_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str, kind: str, atomic: bool,
) -> None:
    source = tmp_path / "source.json"
    source.write_text("protected input")
    target = tmp_path / "x.json"
    if kind == "symlink":
        target.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, target)
    else:
        target.write_text("foreign output")
    original = target.read_text()
    metadata = target.stat()
    monkeypatch.chdir(tmp_path)
    with operation():
        if kind == "hardlink":
            validate_input(source)
        with pytest.raises(OutputPathError) as plain:
            validate_output(target)
        if kind != "foreign":
            with pytest.raises(OutputPathError) as guarded:
                paths._guard_write(spelling)
            assert guarded.value.code == plain.value.code
            assert str(guarded.value) == str(plain.value)
        with pytest.raises(OutputPathError) as caught:
            if atomic:
                with atomic_output(spelling):
                    pytest.fail("An unsafe output reached the write body")
            else:
                validate_output(spelling).write_text("overwritten")
        assert caught.value.code == plain.value.code
        assert str(caught.value) == str(plain.value)
    assert target.read_text() == original
    assert source.read_text() == "protected input"
    assert target.is_symlink() is (kind == "symlink")
    assert os.path.samestat(target.stat(), metadata)
    assert not list(tmp_path.glob(".yap_tmp_*"))


@pytest.mark.parametrize("target", [
    "/etc/yap-probe.json", "/usr/bin/yap-probe", "/System/yap-probe",
    "/private/var/db/yap-probe", "/proc/yap-probe", "/lib/yap-probe",
])
def test_guard_system_folders(target: str) -> None:
    import sys

    if sys.platform != "darwin" and target.startswith(("/System/", "/private/")):
        pytest.skip("macOS folder")
    if sys.platform == "darwin" and target.startswith(("/proc/", "/lib/")):
        pytest.skip("Linux folder")
    with pytest.raises(OutputPathError) as caught:
        paths._guard_write(target)
    assert caught.value.code == "unsafe_path"
    assert str(caught.value) == "Output path is inside a system folder."


@pytest.mark.parametrize("relative", [
    ".ssh", ".ssh/id", ".aws/credentials", ".config/x", ".kube/config", ".gnupg/key",
    ".azure/config", ".oci/config", ".docker/config.json",
])
def test_guard_sensitive_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                              relative: str) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(OutputPathError) as caught:
        paths._guard_write(tmp_path / relative)
    assert caught.value.code == "unsafe_path"
    assert str(caught.value) == "Output path is inside a home folder that holds keys or settings."


@pytest.mark.skipif(paths.sys.platform != "darwin", reason="macOS file-name normalization")
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("insensitive", [False, True])
def test_guard_sensitive_home_unicode_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool, insensitive: bool,
) -> None:
    home = tmp_path / "caf\u00e9"
    (home / ".ssh").mkdir(parents=True)
    if existing:
        (home / ".ssh" / "key").write_text("synthetic key data")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(paths, "_case_insensitive_volume", lambda _: insensitive)
    target = tmp_path / "cafe\u0301" / ".ssh" / "key"
    assert target.exists() is existing
    with pytest.raises(OutputPathError) as caught:
        paths._guard_write(target)
    assert caught.value.code == "unsafe_path"
    assert str(caught.value) == "Output path is inside a home folder that holds keys or settings."
    if existing:
        assert target.read_text() == "synthetic key data"
    assert not list(tmp_path.glob(".yap_tmp_*"))


def test_guard_uses_real_home_and_folder_boundaries(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    alias = tmp_path / "home-alias"
    alias.symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("HOME", str(alias))
    with pytest.raises(OutputPathError, match="home folder"):
        paths._guard_write(home / ".ssh" / "key")
    for target in (home / ".ssh-safe" / "x", home / "ordinary" / ".aws" / "x"):
        assert paths._guard_write(target) == (str(target), os.path.realpath(target))
    parent_alias = tmp_path / "folder-alias"
    parent_alias.symlink_to(home, target_is_directory=True)
    target = parent_alias / "ordinary.json"
    assert paths._guard_write(target) == (str(target), str(home / "ordinary.json"))


@pytest.mark.parametrize("insensitive", [False, True])
def test_guard_case_rules_and_allowed_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, insensitive: bool,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.setattr(paths, "_case_insensitive_volume", lambda _: insensitive, raising=False)
    for target in ("/sYsTeM/yap-probe", str(tmp_path / ".SSH" / "key")):
        if insensitive:
            with pytest.raises(OutputPathError) as caught:
                paths._guard_write(target)
            assert caught.value.code == "unsafe_path"
        else:
            assert paths._guard_write(target) == (target, os.path.realpath(target))
    for target in ("/private/var/folders/yap-probe", "/private/tmp/yap-probe",
                   "/tmp/yap-probe", "/var/folders/yap-probe",
                   "/Systematic/yap-probe", str(tmp_path / "ordinary.json")):
        assert paths._guard_write(target) == (target, os.path.realpath(target))


def test_output_dir_records_created_folder(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "output"
    with operation():
        assert validate_output_dir(target) == target
        assert target.is_dir()
        assert paths._OUTPUT_REALPATHS.get() == {os.path.realpath(target)}


def test_protect_input_refuses_writing_missing_input(tmp_path: Path) -> None:
    source = tmp_path / "missing.json"
    assert not source.exists()
    with paths.operation():
        paths.protect_input(source)
        with pytest.raises(OutputPathError, match="input file") as caught:
            atomic_write_text(source, '{"schema": "yap-studio/example@1"}')
    assert caught.value.code == "invalid_output_path"
    assert not source.exists()


def test_output_rejects_existing_non_report_json(tmp_path: Path) -> None:
    target = tmp_path / "package.json"
    target.write_text('{"name": "ordinary-package"}', encoding="utf-8")
    with pytest.raises(OutputPathError) as caught:
        validate_output(target)
    assert caught.value.code == "unsafe_path"
    assert str(caught.value) == "Existing file is not a yap-studio output; choose a new path."


def test_atomic_output_replaces_existing_yap_json_by_default(tmp_path: Path) -> None:
    target = tmp_path / "review.json"
    target.write_text('{"schema": "yap-studio/review@1", "status": "old"}', encoding="utf-8")
    replacement = '{"schema": "yap-studio/review@1", "status": "new"}'
    atomic_write_text(target, replacement)
    assert target.read_text(encoding="utf-8") == replacement


def test_atomic_output_replaces_existing_verify_report_in_strict_mode(
    tmp_path: Path,
) -> None:
    target = tmp_path / "report.json"
    target.write_text(_report("old"), encoding="utf-8")
    replacement = _report("new")

    atomic_write_text(target, replacement, existing_json="verify-report")

    assert target.read_text(encoding="utf-8") == replacement


@pytest.mark.parametrize(
    "content",
    [
        '{"schema": "yap-studio/transcript@1", "media": {}, "summary": {}, '
        '"checks": {}, "findings": []}',
        '{"schema": "yap-studio/verify-report@1"}',
    ],
)
def test_output_rejects_other_or_forged_yap_json(tmp_path: Path, content: str) -> None:
    target = tmp_path / "report.json"
    target.write_text(content, encoding="utf-8")

    with pytest.raises(OutputPathError, match="not a yap-studio output"):
        validate_output(target, existing_json="verify-report")


def test_output_rejects_malformed_json(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    target.write_text('{"schema":', encoding="utf-8")
    with pytest.raises(OutputPathError, match="not a yap-studio output") as caught:
        validate_output(target)
    assert caught.value.code == "unsafe_path"


def test_output_accepts_named_artifact_with_wav_header(tmp_path: Path) -> None:
    target = tmp_path / "cut-03-render.wav"
    target.write_bytes(b"RIFF\x04\x00\x00\x00WAVE")
    with atomic_output(target, frozenset({".wav"})) as temporary:
        temporary.write_bytes(b"RIFF\x04\x00\x00\x00WAVEnew")
    assert target.read_bytes() == b"RIFF\x04\x00\x00\x00WAVEnew"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("song.wav", b"RIFF\x04\x00\x00\x00WAVE"),
        ("cut-03-render.wav", b"not a wave file"),
    ],
)
def test_output_rejects_non_artifact_wav(tmp_path: Path, name: str, content: bytes) -> None:
    target = tmp_path / name
    target.write_bytes(content)
    with pytest.raises(OutputPathError, match="not a yap-studio output") as caught:
        validate_output(target, frozenset({".wav"}))
    assert caught.value.code == "unsafe_path"


def test_output_rejects_non_report_json_hardlink_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "package.json"
    content = '{"name": "ordinary-package"}'
    source.write_text(content, encoding="utf-8")
    target = tmp_path / "report.json"
    os.link(source, target)
    with pytest.raises(OutputPathError, match="not a yap-studio output") as caught:
        atomic_write_text(target, '{"schema": "yap-studio/verify-report@1"}')
    assert caught.value.code == "unsafe_path"
    assert source.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("kind,message", [
    ("directory", "Output path is an existing folder."),
    ("receipt", "Existing file is not listed in an owner receipt matching its current hash; "
     "choose a new path."),
    ("suffix", "Existing file type is not one this command replaces; choose a new path."),
    ("foreign", "Existing file is not a yap-studio output; choose a new path."),
])
def test_output_refusal_codes(tmp_path: Path, kind: str, message: str) -> None:
    target = tmp_path / ("target.bin" if kind == "suffix" else "target.json")
    if kind == "directory":
        target.mkdir()
    else:
        target.write_text(_report("old") if kind == "receipt" else "foreign")
    receipts = (tmp_path / "missing-receipt.json",) if kind == "receipt" else ()
    with pytest.raises(OutputPathError) as caught:
        validate_output(target, owner_receipts=receipts)
    assert caught.value.code == "unsafe_path"
    assert str(caught.value) == message


def test_output_records_and_operation_can_rewrite_its_atomic_files(tmp_path: Path) -> None:
    target = tmp_path / "output.json"
    with operation():
        assert validate_output(target) == target
        assert paths._OUTPUT_REALPATHS.get() == {os.path.realpath(target)}
        atomic_write_text(target, "arbitrary first bytes")
        assert os.path.realpath(target) in paths._CREATED_REALPATHS.get()
        assert validate_input(target) == target.resolve()
        atomic_write_text(target, "arbitrary replacement")
        assert target.read_text() == "arbitrary replacement"
    with pytest.raises(OutputPathError, match="not a yap-studio output"):
        validate_output(target)


def test_created_file_still_requires_allowed_suffix_and_receipt(tmp_path: Path) -> None:
    target = tmp_path / "output.bin"
    with operation():
        with atomic_output(target) as temporary:
            temporary.write_bytes(b"new")
        with pytest.raises(OutputPathError, match="file type"):
            validate_output(target)
        validate_output(target, frozenset({".bin"}))
        with pytest.raises(OutputPathError, match="current hash"):
            validate_output(target, frozenset({".bin"}), owner_receipts=[tmp_path / "missing"])


@pytest.mark.parametrize("mask", [0o000, 0o027, 0o077])
def test_atomic_temporary_location_permissions_and_replace(tmp_path: Path, mask: int) -> None:
    import stat

    target = tmp_path / "new" / "nested" / "output.json"
    previous_umask = os.umask(mask)
    try:
        with atomic_output(target) as temporary:
            assert temporary.parent == target.parent
            assert temporary.name.startswith(".yap_tmp_")
            assert temporary.suffix == ".json"
            assert stat.S_IMODE(temporary.stat().st_mode) == 0o600
            temporary.write_text(_report("new"))
            temporary_stat = temporary.stat()
            assert not target.exists()
    finally:
        os.umask(previous_umask)
    assert target.read_text() == _report("new")
    assert target.stat().st_ino == temporary_stat.st_ino
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(target.parent.glob(".yap_tmp_*"))
    with atomic_output(target) as temporary:
        temporary.write_text(_report("replacement"))
        assert target.read_text() == _report("new")
        replacement_stat = temporary.stat()
    assert target.stat().st_ino == replacement_stat.st_ino
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert target.read_text() == _report("replacement")


@pytest.mark.parametrize("with_receipt", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_atomic_relative_target_survives_working_folder_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupted: bool, with_receipt: bool,
) -> None:
    from contextlib import nullcontext

    original = tmp_path / "original"
    original.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    target = original / "output.json"
    target.write_text(_report("old"))
    if with_receipt:
        _owner(original / "owner.json", target)
    owners = ("owner.json",) if with_receipt else ()
    unrelated = other / "output.json"
    unrelated.write_text("unrelated output")
    monkeypatch.chdir(original)
    expectation = pytest.raises(KeyboardInterrupt) if interrupted else nullcontext()
    with expectation, atomic_output("output.json", owner_receipts=owners) as temporary:
        temporary.write_text(_report("new"))
        monkeypatch.chdir(other)
        if interrupted:
            raise KeyboardInterrupt
    assert target.read_text() == _report("old" if interrupted else "new")
    assert unrelated.read_text() == "unrelated output"
    assert not list(original.glob(".yap_tmp_*"))
    assert not list(other.glob(".yap_tmp_*"))
    assert temporary.is_absolute()
    assert temporary.parent == original.resolve()


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt, GeneratorExit])
def test_atomic_interruption_preserves_old_file_and_cleans(tmp_path: Path,
                                                         exception: type[BaseException]) -> None:
    target = tmp_path / "output.json"
    target.write_text(_report("old"))
    with pytest.raises(exception), atomic_output(target) as temporary:
        temporary.write_text("partial")
        raise exception()
    assert target.read_text() == _report("old")
    assert not list(tmp_path.glob(".yap_tmp_*"))


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo", "missing"])
def test_atomic_rejects_changed_temporary_type(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "output.json"
    target.write_text(_report("old"))
    other = tmp_path / "other.json"
    other.write_text("untouched")
    with pytest.raises(OutputPathError) as caught, atomic_output(target) as temporary:
        temporary.unlink()
        if kind == "directory":
            temporary.mkdir()
        elif kind == "symlink":
            temporary.symlink_to(other)
        elif kind == "fifo":
            os.mkfifo(temporary)
    assert caught.value.code == "unsafe_path"
    if kind != "symlink":
        assert str(caught.value) == "Temporary output changed type during the write."
    else:
        assert "symlink" in str(caught.value)
    assert target.read_text() == _report("old") and other.read_text() == "untouched"
    assert not list(tmp_path.glob(".yap_tmp_*"))


def test_atomic_target_recheck_refuses_new_input(tmp_path: Path) -> None:
    target = tmp_path / "output.json"
    target.write_text(_report("old"))
    with operation(), pytest.raises(OutputPathError, match="input file"):
        with atomic_output(target) as temporary:
            temporary.write_text("replacement")
            paths.protect_input(target)
    assert target.read_text() == _report("old")
    assert not list(tmp_path.glob(".yap_tmp_*"))


def test_atomic_uses_checked_destination_after_target_becomes_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextvars import ContextVar

    source = tmp_path / "input.json"
    source.write_text("protected input")
    target = tmp_path / "output.json"
    target.write_text(_report("old"))
    target_realpath = os.path.realpath(target)
    record = paths._record
    body_finished = False
    swapped = False

    def record_and_swap(which: ContextVar, realpath: str) -> None:
        nonlocal swapped
        record(which, realpath)
        if which is paths._OUTPUT_REALPATHS and realpath == target_realpath and body_finished:
            assert not swapped
            target.unlink()
            target.symlink_to(source)
            swapped = True

    monkeypatch.setattr(paths, "_record", record_and_swap)
    with operation():
        validate_input(source)
        with atomic_output(target) as temporary:
            temporary.write_text(_report("new"))
            body_finished = True
    assert swapped
    assert source.read_text() == "protected input"
    assert not target.is_symlink()
    assert target.read_text() == _report("new")
    assert not list(tmp_path.glob(".yap_tmp_*"))


@pytest.mark.parametrize("separator", [os.sep, os.sep * 3])
def test_guard_trailing_separator_on_symlink(tmp_path: Path, separator: str) -> None:
    source = tmp_path / "ordinary.json"
    source.write_text("ordinary content")
    alias = tmp_path / "alias.json"
    alias.symlink_to(source)
    with pytest.raises(OutputPathError) as caught:
        paths._guard_write(str(alias) + separator)
    assert caught.value.code == "unsafe_path"
    assert str(caught.value) == "Output path is a symlink; give the real file path."


@pytest.mark.parametrize("separator", [os.sep, os.sep * 3])
def test_output_trailing_separator_preserves_existing_file_checks(
    tmp_path: Path, separator: str,
) -> None:
    target = tmp_path / "ordinary.json"
    target.write_text('{"name": "ordinary"}')
    with pytest.raises(OutputPathError) as bare:
        validate_output(target)
    with pytest.raises(OutputPathError) as trailing:
        validate_output(str(target) + separator)
    assert trailing.value.code == bare.value.code == "unsafe_path"
    assert str(trailing.value) == str(bare.value)
    assert target.read_text() == '{"name": "ordinary"}'


def test_empty_output_path_means_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert paths._guard_write("") == ("", os.path.realpath(tmp_path))
    with pytest.raises(OutputPathError) as caught:
        validate_output("")
    assert caught.value.code == "unsafe_path"
    assert str(caught.value) == "Output path is an existing folder."
    with operation():
        assert validate_output_dir("") == Path("")
        assert paths._OUTPUT_REALPATHS.get() == {os.path.realpath(tmp_path)}


@pytest.mark.parametrize("atomic", [False, True])
def test_existing_fifo_is_refused_without_replacement(tmp_path: Path, atomic: bool) -> None:
    target = tmp_path / "output.json"
    os.mkfifo(target)
    original = target.stat()
    with pytest.raises(OutputPathError) as caught:
        if atomic:
            with atomic_output(target) as temporary:
                temporary.write_text("synthetic output")
        else:
            validate_output(target)
    assert caught.value.code == "unsafe_path"
    assert str(caught.value) == "Output path is not a regular file."
    assert target.exists()
    assert target.is_fifo()
    assert os.path.samestat(target.stat(), original)
    assert not list(tmp_path.glob(".yap_tmp_*"))


@pytest.mark.parametrize("record_name", [
    "_INPUT_REALPATHS", "_OUTPUT_REALPATHS", "_CREATED_REALPATHS",
])
def test_record_contains_exact_path_needs_no_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_name: str,
) -> None:
    target = tmp_path / "output.json"
    target.write_text("synthetic output")
    resolved = os.path.realpath(target)
    handle = getattr(paths, record_name)
    calls = []
    stat = os.stat

    def counted_stat(*args, **kwargs):
        calls.append(args)
        return stat(*args, **kwargs)

    with operation():
        paths._record(handle, resolved)
        with monkeypatch.context() as patched:
            patched.setattr(paths.os, "stat", counted_stat)
            found = paths._record_contains(handle, resolved)
        assert found
        assert calls == []


@pytest.mark.parametrize("separator", [os.sep, os.sep * 3])
def test_output_trailing_separator_uses_stripped_suffix_and_keeps_given_path(
    tmp_path: Path, separator: str,
) -> None:
    target = tmp_path / "output.json"
    target.write_text(_report("old"))
    given = str(target) + separator
    assert paths._guard_write(given) == (given, os.path.realpath(target))
    assert validate_output(given) == Path(given)
    with atomic_output(given) as temporary:
        assert temporary.suffix == ".json"
        temporary.write_text(_report("new"))
    assert target.read_text() == _report("new")
    assert paths._guard_write("/") == ("/", os.path.realpath("/"))


@pytest.mark.skipif(os.sep != "/", reason="POSIX file name")
def test_output_trailing_backslash_is_a_separate_file(tmp_path: Path) -> None:
    original = tmp_path / "result.json"
    original.write_text("protected original")
    target = tmp_path / "result.json\\"
    assert paths._guard_write(target) == (str(target), os.path.realpath(target))
    with operation():
        paths.protect_input(original)
        assert validate_output(target) == target
        with atomic_output(target) as temporary:
            temporary.write_text("separate output")
    assert target.read_text() == "separate output"
    assert original.read_text() == "protected original"
    assert not list(tmp_path.glob(".yap_tmp_*"))


@pytest.mark.skipif(paths.sys.platform != "darwin", reason="macOS file-name normalization")
@pytest.mark.parametrize("insensitive", [False, True])
def test_missing_input_unicode_alias_blocks_atomic_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, insensitive: bool,
) -> None:
    monkeypatch.setattr(paths, "_case_insensitive_volume", lambda _: insensitive)
    source = tmp_path / "caf\u00e9.json"
    target = tmp_path / "cafe\u0301.json"
    with operation():
        paths.protect_input(source)
        with pytest.raises(OutputPathError) as caught:
            with atomic_output(target) as temporary:
                temporary.write_text("synthetic output")
    assert caught.value.code == "invalid_output_path"
    assert str(caught.value) == "Output path is an input file of this command."
    assert not source.exists()
    assert not target.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("insensitive", [False, True])
def test_guard_missing_input_case_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, insensitive: bool,
) -> None:
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.setattr(paths, "_case_insensitive_volume", lambda _: insensitive, raising=False)
    source = tmp_path / "input.json"
    target = tmp_path / "INPUT.JSON"
    with operation():
        paths.protect_input(source)
        if insensitive:
            with pytest.raises(OutputPathError) as caught:
                paths._guard_write(target)
            assert caught.value.code == "invalid_output_path"
            assert str(caught.value) == "Output path is an input file of this command."
        else:
            assert paths._guard_write(target) == (str(target), os.path.realpath(target))
        assert paths._INPUT_REALPATHS.get() == {os.path.realpath(source)}


@pytest.mark.parametrize("insensitive", [False, True])
@pytest.mark.parametrize("identical", [False, True])
@pytest.mark.parametrize("role", ["input", "output", "created"])
def test_existing_record_case_alias_uses_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, insensitive: bool,
    identical: bool, role: str,
) -> None:
    monkeypatch.setattr(paths, "_case_insensitive_volume", lambda _: insensitive, raising=False)
    target = tmp_path / "OUT.JSON"
    target.write_text("arbitrary output bytes")
    recorded = os.path.realpath(tmp_path / "out.json")
    stat = os.stat
    recorded_metadata = list(target.stat())
    if not identical:
        recorded_metadata[1] += 1
    recorded_stat = os.stat_result(recorded_metadata)

    def case_stat(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and os.fspath(path) == recorded:
            return recorded_stat
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(paths.os, "stat", case_stat)
    assert os.path.samefile(target, recorded) is identical
    with operation():
        if role == "input":
            paths._record(paths._INPUT_REALPATHS, recorded)
            if identical:
                with pytest.raises(OutputPathError, match="input file"):
                    paths._guard_write(target)
            else:
                assert paths._guard_write(target) == (str(target), os.path.realpath(target))
        elif role == "output":
            paths._record(paths._OUTPUT_REALPATHS, recorded)
            assert validate_input(target) == target.resolve()
            assert (os.path.realpath(target) in paths._INPUT_REALPATHS.get()) is not identical
        else:
            paths._record(paths._CREATED_REALPATHS, recorded)
            if identical:
                assert validate_output(target) == target
            else:
                with pytest.raises(OutputPathError, match="not a yap-studio output"):
                    validate_output(target)


@pytest.mark.parametrize("result,insensitive", [(0, True), (1, False), (-1, False)])
def test_case_volume_queries_nearest_existing_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: int, insensitive: bool,
) -> None:
    folder = tmp_path / "existing"
    folder.mkdir()
    target = folder / "missing" / "nested" / "output.json"
    calls = []

    def pathconf(queried_folder, name):
        calls.append((os.fspath(queried_folder), name))
        return result

    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.setitem(paths.os.pathconf_names, "PC_CASE_SENSITIVE", 12345)
    monkeypatch.setattr(paths.os, "pathconf", pathconf)
    assert paths._case_insensitive_volume(str(target)) is insensitive
    assert calls == [(str(folder), 12345)]


def test_case_volume_uses_macos_numeric_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def pathconf(folder, name):
        calls.append((os.fspath(folder), name))
        return 0

    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.delitem(paths.os.pathconf_names, "PC_CASE_SENSITIVE", raising=False)
    monkeypatch.setattr(paths.os, "pathconf", pathconf)
    assert paths._case_insensitive_volume(str(tmp_path / "output.json"))
    assert calls == [(str(tmp_path), 11)]


@pytest.mark.parametrize("named_setting", [False, True])
def test_case_volume_linux_never_queries_pathconf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named_setting: bool,
) -> None:
    calls = []

    def pathconf(folder, name):
        calls.append((os.fspath(folder), name))
        return 0

    monkeypatch.setattr(paths.sys, "platform", "linux")
    if named_setting:
        monkeypatch.setitem(paths.os.pathconf_names, "PC_CASE_SENSITIVE", 11)
    else:
        monkeypatch.delitem(paths.os.pathconf_names, "PC_CASE_SENSITIVE", raising=False)
    monkeypatch.setattr(paths.os, "pathconf", pathconf)
    with operation():
        paths.protect_input(tmp_path / "input.json")
        paths.protect_input(tmp_path / "caf\u00e9.json")
        for target in (tmp_path / "INPUT.JSON", tmp_path / "cafe\u0301.json"):
            assert paths._guard_write(target) == (str(target), os.path.realpath(target))
    assert not paths._case_insensitive_volume(str(tmp_path / "output.json"))
    assert calls == []


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_case_volume_pathconf_error_uses_exact_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception],
) -> None:
    def pathconf(*args):
        raise error("unsupported setting")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.setitem(paths.os.pathconf_names, "PC_CASE_SENSITIVE", 12345)
    monkeypatch.setattr(paths.os, "pathconf", pathconf)
    assert not paths._case_insensitive_volume(str(tmp_path / "INPUT.JSON"))
    with operation():
        paths.protect_input(tmp_path / "input.json")
        for target in (str(tmp_path / "INPUT.JSON"), str(tmp_path / ".SSH" / "key"),
                       "/sYsTeM/yap-probe"):
            assert paths._guard_write(target) == (target, os.path.realpath(target))


def test_validate_input_recorded_output_hardlink_is_not_recorded(tmp_path: Path) -> None:
    target = tmp_path / "output.json"
    alias = tmp_path / "readback.json"
    with operation():
        atomic_write_text(target, "synthetic output")
        os.link(target, alias)
        assert validate_input(alias) == alias.resolve()
        assert not paths._INPUT_REALPATHS.get()
        atomic_write_text(target, "replacement")
    assert target.read_text() == "replacement"
    assert alias.read_text() == "synthetic output"


@pytest.mark.parametrize("threaded", [False, True])
def test_operation_copied_context_keeps_records_in_child(
    tmp_path: Path, threaded: bool,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    handles = (paths._INPUT_REALPATHS, paths._OUTPUT_REALPATHS, paths._CREATED_REALPATHS)
    outer = os.path.realpath(tmp_path / "outer")
    child = os.path.realpath(tmp_path / "child")

    def record_child() -> None:
        for handle in handles:
            assert handle.get() == {outer}
            paths._record(handle, child)
            assert handle.get() == {outer, child}

    with operation():
        for handle in handles:
            paths._record(handle, outer)
        copied = copy_context()
        if threaded:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(copied.run, record_child).result(timeout=10)
        else:
            copied.run(record_child)
        for handle in handles:
            assert handle.get() == {outer}
        assert copied.run(paths._INPUT_REALPATHS.get) == {outer, child}
    for handle in handles:
        assert not handle.get()


def test_atomic_body_can_read_own_temporary(tmp_path: Path) -> None:
    target = tmp_path / "output.json"
    with operation(), atomic_output(target) as temporary:
        temporary.write_text("synthetic output")
        assert validate_input(temporary) == temporary.resolve()
    assert target.read_text() == "synthetic output"
    assert not list(tmp_path.glob(".yap_tmp_*"))


def _owner(path: Path, target: Path, *, relative: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "yap-studio/receipt@1",
                "outputs": [
                    {
                        "path": target.name if relative else str(target),
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("relative", [True, False])
@pytest.mark.parametrize("suffix", [".mp4", ".srt", ".bin"])
def test_atomic_output_replaces_receipt_owned_file(
    tmp_path: Path,
    relative: bool,
    suffix: str,
) -> None:
    target = tmp_path / ("owned" + suffix)
    target.write_bytes(b"original")
    owner = _owner(tmp_path / "run.receipt.json", target, relative=relative)
    with operation(), atomic_output(target, owner_receipts=[owner]) as temporary:
        temporary.write_bytes(b"replacement")
    assert target.read_bytes() == b"replacement"


@pytest.mark.parametrize("reason", ["changed", "missing", "other", "invalid_schema", "malformed"])
def test_receipt_ownership_refuses_unverified_file(tmp_path: Path, reason: str) -> None:
    target = tmp_path / "owned.mp4"
    target.write_bytes(b"original")
    owner = _owner(tmp_path / "run.receipt.json", target)
    if reason == "changed":
        target.write_bytes(b"changed")
    elif reason == "missing":
        owner.unlink()
    elif reason == "other":
        other = tmp_path / "other.mp4"
        other.write_bytes(target.read_bytes())
        _owner(owner, other)
    elif reason == "invalid_schema":
        data = json.loads(owner.read_text())
        data["schema"] = "foreign/receipt@1"
        owner.write_text(json.dumps(data))
    else:
        owner.write_text("[]")
    original = target.read_bytes()
    with operation(), pytest.raises(OutputPathError):
        with atomic_output(target, owner_receipts=[owner]) as temporary:
            temporary.write_bytes(b"replacement")
    assert target.read_bytes() == original


def test_ownership_accepts_one_of_several_receipts(tmp_path: Path) -> None:
    target = tmp_path / "owned.mp4"
    target.write_bytes(b"original")
    owner = _owner(tmp_path / "owner.json", target)
    assert validate_output(target, owner_receipts=[tmp_path / "missing.json", owner]) == target


def test_owned_input_and_symlink_still_refused(tmp_path: Path) -> None:
    target = tmp_path / "owned.mp4"
    target.write_bytes(b"original")
    owner = _owner(tmp_path / "owner.json", target)
    with operation(), pytest.raises(OutputPathError, match="input file"):
        validate_input(target)
        validate_output(target, owner_receipts=[owner])
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(target)
    with operation(), pytest.raises(OutputPathError, match="symlink"):
        validate_output(alias, owner_receipts=[owner])


def test_atomic_output_rechecks_owned_hash_before_replace(tmp_path: Path) -> None:
    target = tmp_path / "owned.mp4"
    target.write_bytes(b"original")
    owner = _owner(tmp_path / "owner.json", target)
    with operation(), pytest.raises(OutputPathError):
        with atomic_output(target, owner_receipts=[owner]) as temporary:
            temporary.write_bytes(b"replacement")
            target.write_bytes(b"external change")
    assert target.read_bytes() == b"external change"


@pytest.mark.parametrize("entry", ["absolute", "relative", "symlink", "hardlink"])
def test_receipt_cannot_own_files_outside_its_folder(tmp_path: Path, entry: str) -> None:
    folder = tmp_path / "receipts"
    folder.mkdir()
    target = tmp_path / "outside.mp4"
    target.write_bytes(b"original")
    owner = _owner(folder / "owner.json", target, relative=False)
    if entry != "absolute":
        document = json.loads(owner.read_text())
        if entry == "relative":
            document["outputs"][0]["path"] = "../outside.mp4"
        elif entry == "symlink":
            (folder / "alias.mp4").symlink_to(target)
            document["outputs"][0]["path"] = "alias.mp4"
        else:
            os.link(target, folder / "alias.mp4")
            document["outputs"][0]["path"] = "alias.mp4"
        owner.write_text(json.dumps(document))
    with operation(), pytest.raises(OutputPathError):
        validate_output(target, owner_receipts=[owner])
    assert target.read_bytes() == b"original"


def test_input_refusal_precedes_matching_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio import paths

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source media")
    owner = _owner(tmp_path / "owner.json", source)

    def unexpected_ownership(*args: object) -> bool:
        pytest.fail("input refusal must precede receipt ownership")

    monkeypatch.setattr(paths, "_is_owned_output", unexpected_ownership)
    with operation(), pytest.raises(OutputPathError, match="input file"):
        validate_input(source)
        validate_output(source, owner_receipts=[owner])
    assert source.read_bytes() == b"source media"


@pytest.mark.parametrize("which", ["old_sha256", "new_sha256"])
def test_pending_journal_owns_either_matching_hash(tmp_path: Path, which: str) -> None:
    stage = tmp_path / "cut"
    stage.mkdir()
    target = stage / "cut.mp4"
    target.write_bytes(b"owned bytes")
    pending = stage / "receipt.pending.json"
    pending.write_text(json.dumps({
        "schema": "yap-studio/stage-pending@1", "stage": "cut", "outputs": [{
            "path": "cut/cut.mp4", "old_sha256": None, "new_sha256": "0" * 64,
            which: hashlib.sha256(target.read_bytes()).hexdigest(),
        }],
    }), encoding="utf-8")
    with operation(), atomic_output(target, owner_receipts=[pending]) as temporary:
        temporary.write_bytes(b"rerun complete")
    assert target.read_bytes() == b"rerun complete"


def test_stage_receipt_paths_are_relative_to_project(tmp_path: Path) -> None:
    stage = tmp_path / "ingest"
    stage.mkdir()
    target = stage / "words.json"
    target.write_text('{"schema":"yap-studio/words@1"}', encoding="utf-8")
    owner = stage / "receipt.json"
    data = {"schema": "yap-studio/stage-receipt@1", "outputs": [{
        "path": "ingest/words.json", "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }]}
    owner.write_text(json.dumps(data), encoding="utf-8")
    assert validate_output(target, owner_receipts=[owner]) == target
    target.write_text('{"schema":"yap-studio/words@1","changed":true}', encoding="utf-8")
    with operation(), pytest.raises(OutputPathError):
        validate_output(target, owner_receipts=[owner])


def test_pending_journal_does_not_own_hand_edited_output(tmp_path: Path) -> None:
    stage = tmp_path / "cut"
    stage.mkdir()
    target = stage / "cut.mp4"
    target.write_bytes(b"hand edited")
    pending = stage / "receipt.pending.json"
    pending.write_text(json.dumps({
        "schema": "yap-studio/stage-pending@1", "outputs": [{
            "path": "cut/cut.mp4", "old_sha256": "0" * 64, "new_sha256": "1" * 64,
        }],
    }), encoding="utf-8")
    with operation(), pytest.raises(OutputPathError):
        validate_output(target, owner_receipts=[pending])
