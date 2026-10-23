"""Project cut face caches with deterministic detectors."""

import hashlib
import json
import platform
from pathlib import Path

import pytest
from test_faces_detect import FakeDetector, make_clip

from yap_studio.errors import OutputPathError
from yap_studio.faces import detect
from yap_studio.faces.project import FACE_EVERY, cut_faces, cut_faces_identity


@pytest.mark.parametrize("reason", [None, "unavailable in this test"])
def test_cut_faces_identity_does_not_measure(
    monkeypatch: pytest.MonkeyPatch, reason: str | None,
) -> None:
    monkeypatch.setattr(detect, "unavailable_reason", lambda: reason)
    monkeypatch.setattr(platform, "mac_ver", lambda: ("15.7", ("", "", ""), ""))

    def unexpected(*args, **kwargs):
        pytest.fail("face identity must not measure video")

    monkeypatch.setattr(detect, "detect_faces", unexpected)
    assert cut_faces_identity("a" * 64) == {
        "cut": "a" * 64,
        "helper": hashlib.sha256(detect.helper_source().read_bytes()).hexdigest(),
        "every": FACE_EVERY, "macos": "15.7", "available": reason is None,
    }


def test_cut_faces_cache_hit_and_cut_digest_miss(tmp_path: Path) -> None:
    cut = make_clip(tmp_path / "cut.mp4")
    detector = FakeDetector()
    measured = cut_faces(tmp_path, cut, detector=detector)
    assert measured.status == "measured"
    assert detector.calls == [(cut, FACE_EVERY)]
    helper_sha = hashlib.sha256(detect.helper_source().read_bytes()).hexdigest()[:16]
    target = tmp_path / ".yap/cache/faces" / (
        f"{measured.video.sha256}-{helper_sha}-macos{platform.mac_ver()[0] or 'unknown'}"
        f"-every{FACE_EVERY}.json"
    )
    assert json.loads(target.read_text()) == measured.to_dict()
    assert cut_faces(tmp_path, cut, detector=detector) == measured
    assert len(detector.calls) == 1
    cut.unlink()
    make_clip(cut, frames=32)
    detector.frames = 32
    assert cut_faces(tmp_path, cut, detector=detector).video.frames == 32
    assert len(detector.calls) == 2
    assert len(list(target.parent.glob("*.json"))) == 2


def test_os_update_remeasures_cut_and_changes_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cut = make_clip(tmp_path / "cut.mp4")
    detector = FakeDetector()
    monkeypatch.setattr(platform, "mac_ver", lambda: ("15.7", ("", "", ""), ""))
    measured = cut_faces(tmp_path, cut, detector=detector)
    before = cut_faces_identity(measured.video.sha256)
    assert cut_faces(tmp_path, cut, detector=detector) == measured
    assert len(detector.calls) == 1
    monkeypatch.setattr(platform, "mac_ver", lambda: ("15.7.1", ("", "", ""), ""))
    after = cut_faces_identity(measured.video.sha256)
    assert after == {**before, "macos": "15.7.1"}
    assert cut_faces(tmp_path, cut, detector=detector) == measured
    assert len(detector.calls) == 2
    assert cut_faces(tmp_path, cut, detector=detector) == measured
    assert len(detector.calls) == 2
    names = [target.name for target in (tmp_path / ".yap/cache/faces").glob("*.json")]
    assert len(names) == 2
    assert any("-macos15.7-" in name for name in names)
    assert any("-macos15.7.1-" in name for name in names)


@pytest.mark.parametrize("invalid", ["sha", "sampling", "status", "json"])
def test_invalid_cached_document_is_remeasured(tmp_path: Path, invalid: str) -> None:
    cut = make_clip(tmp_path / "cut.mp4")
    detector = FakeDetector()
    cut_faces(tmp_path, cut, detector=detector)
    target = next((tmp_path / ".yap/cache/faces").glob("*.json"))
    data = json.loads(target.read_text())
    if invalid == "sha":
        data["video"]["sha256"] = "0" * 64
    elif invalid == "sampling":
        data["sampling"]["every"] = 1
    elif invalid == "status":
        data.update(status="unmeasured", reason="unavailable", frames=[], gaps=[])
    target.write_text("{" if invalid == "json" else json.dumps(data))
    assert cut_faces(tmp_path, cut, detector=detector).status == "measured"
    assert len(detector.calls) == 2


def test_unavailable_detection_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cut = make_clip(tmp_path / "cut.mp4")
    monkeypatch.setattr(detect, "unavailable_reason", lambda: "unavailable in this test")
    first = cut_faces(tmp_path, cut)
    assert (first.status, first.reason) == ("unmeasured", "unavailable in this test")
    assert not (tmp_path / ".yap/cache/faces").exists()
    cut_faces(tmp_path, cut, detector=FakeDetector())
    assert cut_faces(tmp_path, cut).status == "unmeasured"


def test_cache_rejects_symlink_escape(tmp_path: Path) -> None:
    cut = make_clip(tmp_path / "cut.mp4")
    folder = tmp_path / ".yap/cache"
    folder.mkdir(parents=True)
    (folder / "faces").symlink_to(tmp_path / "external")
    with pytest.raises(OutputPathError):
        cut_faces(tmp_path, cut, detector=FakeDetector())



def test_cached_only_faces_never_measure_or_create_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.faces.project import cached_cut_faces

    cut = make_clip(tmp_path / "cut.mp4")
    assert cached_cut_faces(tmp_path, cut) is None
    assert not (tmp_path / ".yap/cache/faces").exists()
    measured = cut_faces(tmp_path, cut, detector=FakeDetector())
    monkeypatch.setattr(detect, "detect_faces", lambda *args, **kwargs: pytest.fail("measurement"))
    assert cached_cut_faces(tmp_path, cut) == measured
    monkeypatch.setattr(platform, "mac_ver", lambda: ("99.0", ("", "", ""), ""))
    assert cached_cut_faces(tmp_path, cut) is None


@pytest.mark.parametrize("measure", [False, True])
def test_face_cache_uses_supplied_digest_without_rehashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measure: bool,
) -> None:
    from yap_studio import receipt
    from yap_studio.faces.project import cached_cut_faces

    cut = make_clip(tmp_path / "cut.mp4")
    detector = FakeDetector()
    measured = cut_faces(tmp_path, cut, detector=detector)
    sha256_file = receipt.sha256_file

    def hash_file(source):
        if Path(source) == cut:
            pytest.fail("supplied digest must avoid hashing the cut")
        return sha256_file(source)

    monkeypatch.setattr(receipt, "sha256_file", hash_file)
    cached = (cut_faces(tmp_path, cut, detector=detector, digest=measured.video.sha256)
              if measure else cached_cut_faces(tmp_path, cut, digest=measured.video.sha256))
    assert cached == measured
    assert detector.calls == [(cut, FACE_EVERY)]
