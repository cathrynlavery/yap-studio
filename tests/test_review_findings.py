"""Hash-bound reviewer findings for plain media and saved project versions."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from yap_studio import ffmpeg
from yap_studio.receipt import sha256_file
from yap_studio.review import findings as review_findings
from yap_studio.review.findings import findings_for_media, note_target
from yap_studio.watch import findings as watch_findings
from yap_studio.watch.findings import default_findings_path, record_findings


@pytest.fixture(scope="module")
def render(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("review-findings") / "render.mp4"
    ffmpeg.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=blue:size=64x36:rate=30", "-frames:v", "90", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(path),
    ], timeout_s=30)
    return path


@pytest.fixture
def media(render: Path, tmp_path: Path) -> Path:
    path = tmp_path / "render.mp4"
    shutil.copyfile(render, path)
    return path


def _point(at: float = 0.5, **changes) -> dict:
    return {"at": at, "kind": "slow", "severity": "medium",
            "reason": "First note", "fix": "Tighten the pause.", **changes}


def _project_media(render: Path, tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    cut = project / "cut" / "cut.mp4"
    saved = project / "versions" / "v1" / "cut.mp4"
    for path in (cut, saved):
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(render, path)
    return project, cut, saved


def test_next_to_media_findings_found(media: Path) -> None:
    agent = record_findings(media, [{"start": 0.4, "end": 1.2, "kind": "face_covered",
                                    "severity": "high", "reason": "The face is covered.",
                                    "fix": "Move the text."}],
                            source="agent", reviewer="Reviewer A")
    machine = record_findings(media, [_point(2, kind="audio", check="loudness",
                                             evidence={"jump_lu": 9.5})], source="machine")
    result = findings_for_media(media, sha256_file(media))
    assert result == {"available": True, "file": "render.findings.json", "stale": [],
                      "items": [{**agent["findings"][0], "target": "element"},
                                {**machine["findings"][1], "target": "speech"}]}
    assert str(media.parent) not in json.dumps(result)


def test_missing_findings_returns_empty(media: Path) -> None:
    assert findings_for_media(media, sha256_file(media)) == {
        "available": False, "file": None, "stale": [], "items": [],
    }


def test_project_cut_findings_found_for_saved_version(render: Path, tmp_path: Path) -> None:
    project, cut, saved = _project_media(render, tmp_path)
    written = record_findings(cut, [_point()], source="agent", reviewer="Reviewer A")
    result = findings_for_media(saved, sha256_file(saved), project_dir=project)
    assert result == {"available": True, "file": "cut/cut.findings.json", "stale": [],
                      "items": [{**written["findings"][0], "target": "story"}]}
    assert str(project) not in json.dumps(result)


def test_another_render_listed_as_stale(media: Path) -> None:
    record_findings(media, [_point()], source="agent")
    result = findings_for_media(media, "a" * 64)
    assert result == {"available": False, "file": None,
                      "stale": ["render.findings.json"], "items": []}
    assert str(media.parent) not in json.dumps(result)


def test_stale_saved_findings_fall_back_to_matching_cut(render: Path, tmp_path: Path) -> None:
    project, cut, saved = _project_media(render, tmp_path)
    record_findings(saved, [_point(reason="An older render.")], source="agent")
    path = default_findings_path(saved)
    data = json.loads(path.read_text())
    data["video"]["sha256"] = "a" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    written = record_findings(cut, [_point()], source="agent")
    result = findings_for_media(saved, sha256_file(saved), project_dir=project)
    assert result == {"available": True, "file": "cut/cut.findings.json",
                      "stale": ["versions/v1/cut.findings.json"],
                      "items": [{**written["findings"][0], "target": "story"}]}
    assert str(project) not in json.dumps(result)


def test_malformed_saved_findings_fall_back_to_matching_cut(render: Path, tmp_path: Path) -> None:
    project, cut, saved = _project_media(render, tmp_path)
    default_findings_path(saved).write_text("{broken", encoding="utf-8")
    written = record_findings(cut, [_point()], source="agent")
    result = findings_for_media(saved, sha256_file(saved), project_dir=project)
    assert result == {"available": True, "file": "cut/cut.findings.json", "stale": [],
                      "items": [{**written["findings"][0], "target": "story"}]}


def test_matching_saved_findings_keep_stale_cut_file_name(render: Path, tmp_path: Path) -> None:
    project, cut, saved = _project_media(render, tmp_path)
    written = record_findings(saved, [_point()], source="agent")
    record_findings(cut, [_point()], source="agent")
    path = default_findings_path(cut)
    data = json.loads(path.read_text())
    data["video"]["sha256"] = "a" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    result = findings_for_media(saved, sha256_file(saved), project_dir=project)
    assert result == {"available": True, "file": "versions/v1/cut.findings.json",
                      "stale": ["cut/cut.findings.json"],
                      "items": [{**written["findings"][0], "target": "story"}]}


def test_first_matching_file_wins(render: Path, tmp_path: Path) -> None:
    project, cut, saved = _project_media(render, tmp_path)
    written = record_findings(saved, [_point(reason="Saved version note.")], source="agent")
    record_findings(cut, [_point(reason="Cut note.")], source="agent")
    result = findings_for_media(saved, sha256_file(saved), project_dir=project)
    assert result == {"available": True, "file": "versions/v1/cut.findings.json", "stale": [],
                      "items": [{**written["findings"][0], "target": "story"}]}
    assert str(project) not in json.dumps(result)


def test_duplicate_candidates_list_stale_once(render: Path, tmp_path: Path) -> None:
    project, cut, _ = _project_media(render, tmp_path)
    record_findings(cut, [_point()], source="agent")
    assert findings_for_media(cut, "a" * 64, project_dir=project) == {
        "available": False, "file": None, "stale": ["cut/cut.findings.json"], "items": [],
    }


@pytest.mark.parametrize("damage", ["json", "schema", "finding", "evidence", "utf8"])
def test_malformed_findings_report_error(media: Path, damage: str) -> None:
    record_findings(media, [_point()], source="agent")
    path = default_findings_path(media)
    data = json.loads(path.read_text())
    if damage == "json":
        path.write_text("{broken", encoding="utf-8")
    elif damage == "utf8":
        path.write_bytes(b"\xff")
    else:
        if damage == "schema":
            data["schema"] = "foreign/findings@1"
        elif damage == "finding":
            data["findings"][0]["start_frame"] = -1
        else:
            data["findings"][0]["evidence"] = {"score": float("nan")}
        path.write_text(json.dumps(data), encoding="utf-8")
    original = path.read_bytes()
    result = findings_for_media(media, sha256_file(media))
    assert result == {"available": False, "file": None, "stale": [], "items": [],
                      "error": "render.findings.json: malformed findings file"}
    assert str(media.parent) not in json.dumps(result)
    assert path.read_bytes() == original


def test_malformed_cut_keeps_stale_saved_file_name(render: Path, tmp_path: Path) -> None:
    project, cut, saved = _project_media(render, tmp_path)
    record_findings(saved, [_point()], source="agent")
    default_findings_path(cut).write_text("{broken", encoding="utf-8")
    result = findings_for_media(saved, "a" * 64, project_dir=project)
    assert result == {"available": False, "file": None,
                      "stale": ["versions/v1/cut.findings.json"], "items": [],
                      "error": "cut/cut.findings.json: malformed findings file"}
    assert str(project) not in json.dumps(result)


def test_malformed_saved_findings_keep_stale_cut_file_name(render: Path, tmp_path: Path) -> None:
    project, cut, saved = _project_media(render, tmp_path)
    default_findings_path(saved).write_text("{broken", encoding="utf-8")
    record_findings(cut, [_point()], source="agent")
    result = findings_for_media(saved, "a" * 64, project_dir=project)
    assert result == {"available": False, "file": None, "stale": ["cut/cut.findings.json"],
                      "items": [],
                      "error": "versions/v1/cut.findings.json: malformed findings file"}


def test_first_error_reported_when_no_candidate_matches(render: Path, tmp_path: Path) -> None:
    project, cut, saved = _project_media(render, tmp_path)
    for path in (default_findings_path(saved), default_findings_path(cut)):
        path.write_text("{broken", encoding="utf-8")
    result = findings_for_media(saved, sha256_file(saved), project_dir=project)
    assert result == {"available": False, "file": None, "stale": [], "items": [],
                      "error": "versions/v1/cut.findings.json: malformed findings file"}


@pytest.mark.parametrize("changed", ["mtime", "size"])
def test_cached_findings_reread_when_file_changes(media: Path, monkeypatch, changed: str) -> None:
    record_findings(media, [_point()], source="agent")
    path = default_findings_path(media)
    sha = sha256_file(media)
    original = path.stat()
    read = watch_findings.read_findings
    reads = []

    def counted(path: Path) -> dict | None:
        reads.append(path)
        return read(path)

    monkeypatch.setattr(review_findings, "read_findings", counted)
    first = findings_for_media(media, sha)
    assert findings_for_media(media, sha) == first
    assert reads == [path]
    if changed == "mtime":
        path.write_text(path.read_text().replace("First note", "Other note"), encoding="utf-8")
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
        assert path.stat().st_size == original.st_size
    else:
        record_findings(media, [_point(1, reason="Another note.")], source="agent")
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert path.stat().st_size != original.st_size
    updated = findings_for_media(media, sha)
    assert updated != first
    assert findings_for_media(media, sha) == updated
    assert reads == [path, path]


def test_cached_findings_reread_after_same_size_replacement_with_preserved_mtime(
    media: Path,
) -> None:
    record_findings(media, [_point()], source="agent")
    path = default_findings_path(media)
    sha = sha256_file(media)
    original = path.stat()
    assert findings_for_media(media, sha)["items"][0]["reason"] == "First note"

    replacement = path.with_suffix(".replacement")
    replacement.write_text(path.read_text().replace("First note", "Other note"), encoding="utf-8")
    os.replace(replacement, path)
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    replaced = path.stat()
    assert replaced.st_size == original.st_size
    assert replaced.st_mtime_ns == original.st_mtime_ns
    assert replaced.st_ino != original.st_ino
    updated = findings_for_media(media, sha)
    assert updated["items"][0]["reason"] == "Other note"
    assert findings_for_media(media, sha) == updated


def test_cached_malformed_file_reread_after_repair(media: Path, monkeypatch) -> None:
    path = default_findings_path(media)
    path.write_text("{broken", encoding="utf-8")
    sha = sha256_file(media)
    read = watch_findings.read_findings
    reads = []

    def counted(path: Path) -> dict | None:
        reads.append(path)
        return read(path)

    monkeypatch.setattr(review_findings, "read_findings", counted)
    assert "error" in findings_for_media(media, sha)
    assert "error" in findings_for_media(media, sha)
    assert reads == [path]
    path.unlink()
    record_findings(media, [_point()], source="agent")
    assert findings_for_media(media, sha)["available"]
    assert reads == [path, path]


def test_cached_items_are_independent(media: Path) -> None:
    record_findings(media, [_point(check="still", evidence={"duration_s": 2})], source="machine")
    sha = sha256_file(media)
    first = findings_for_media(media, sha)
    first["items"][0]["reason"] = "Changed by caller"
    first["items"][0]["evidence"]["duration_s"] = 99
    again = findings_for_media(media, sha)
    assert again["items"][0]["reason"] == "First note"
    assert again["items"][0]["evidence"] == {"duration_s": 2}


def test_items_sorted_by_frame_then_numeric_id(media: Path) -> None:
    entries = [_point(2)] + [_point(0.5, reason=f"Note {index}") for index in range(2, 11)]
    record_findings(media, entries, source="agent")
    path = default_findings_path(media)
    data = json.loads(path.read_text())
    data["findings"].reverse()
    path.write_text(json.dumps(data), encoding="utf-8")
    items = findings_for_media(media, sha256_file(media))["items"]
    assert [item["id"] for item in items] == [f"f{index}" for index in range(2, 11)] + ["f1"]
    assert [item["start_frame"] for item in items] == [15] * 9 + [60]


def test_supplied_hash_does_not_hash_or_probe_media(media: Path, monkeypatch) -> None:
    record_findings(media, [_point()], source="agent")
    sha = sha256_file(media)

    def unexpected(*_args, **_kwargs):
        pytest.fail("review findings must use the supplied media hash")

    monkeypatch.setattr(watch_findings, "sha256_file", unexpected)
    monkeypatch.setattr(ffmpeg, "probe", unexpected)
    media.unlink()
    assert findings_for_media(media, sha)["available"]


def test_media_outside_project_uses_file_name_only(media: Path, tmp_path: Path) -> None:
    record_findings(media, [_point()], source="agent")
    result = findings_for_media(media, sha256_file(media), project_dir=tmp_path / "project")
    assert result["available"] and result["file"] == "render.findings.json"
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize(("kind", "target"), [
    ("slow", "story"), ("jarring_cut", "story"), ("face_covered", "element"),
    ("behind_hidden", "element"),
    ("caption_issue", "caption"), ("audio", "speech"), ("other", None), ("unknown", None),
])
def test_note_target(kind: str, target: str | None) -> None:
    assert note_target(kind) == target


def _layer_checks(pass_name: str, message: str, *, time: float | None = None,
                  severity: str = "warning") -> dict:
    passes = {name: {"status": "passed", "errors": 0, "warnings": 0, "findings": []}
              for name in ("lint", "runtime", "layout", "motion", "contrast")}
    passes[pass_name] = {
        "status": "errors" if severity == "error" else "warnings",
        "errors": int(severity == "error"), "warnings": int(severity == "warning"),
        "findings": [{"severity": severity, "code": "text", "message": message,
                      "time": time, "selector": ".text"}],
    }
    return {"schema": "yap-studio/layer-checks@1", "status": "findings", "passes": passes}


def _write_receipt(project: Path, stage: str, data: dict) -> None:
    path = project / stage / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "yap-studio/stage-receipt@1",
                                "status": "verified", **data}), encoding="utf-8")


@pytest.mark.parametrize("collapsed", [False, True])
@pytest.mark.parametrize("frame,expected", [(9, 10), (14, 14), (31, 29)])
def test_final_caption_contrast_finding_and_layer_skip(
    checked_project: tuple[Path, Path], collapsed: bool, frame: int, expected: int,
) -> None:
    project, media = checked_project
    data = json.loads((project / "final/receipt.json").read_bytes())
    evidence = {"caption_id": "c1", "text": "Read this", "word": "Read", "word_index": 0,
                "ratio": 1.2, "frame": frame, "start_frame": 10, "end_frame": 30}
    if collapsed:
        evidence = {"flagged": 11, "worst": [{"caption_id": "c1", "text": "Read this",
                                               "word": "Read", "time_s": frame / 30,
                                               "ratio": 1.2}]}
    data["caption_contrast"] = {"weak": [{"id": "c1", "start_frame": 10, "end_frame": 30}]}
    data["verify"] = {"checks": {"captions.contrast": {"ran": True}}, "findings": [{
        "check": "captions.contrast", "severity": "fail" if collapsed else "warn",
        "message": "Words are hard to read", "suggested_action": "Back this phrase",
        "evidence": evidence,
    }]}
    _write_receipt(project, "final", data)
    captions = json.loads((project / "captions/receipt.json").read_bytes())
    captions["layer_checks"]["passes"]["contrast"] = {
        "status": "skipped", "reason": "no text on an opaque background in this layer",
    }
    _write_receipt(project, "captions", captions)
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    item = next(item for item in result["items"] if item["kind"] == "caption_contrast")
    assert note_target("caption_contrast") == item["target"] == "caption"
    assert item["source"] == "machine"
    assert (item["start_frame"], item["end_frame"]) == (expected, expected + 1)
    assert item["severity"] == ("error" if collapsed else "warning")
    assert item["evidence"] == evidence and item["reason"] == "Words are hard to read"
    assert item["fix"] == "Back this phrase"
    assert result["layer_checks"][0]["skipped"] == []
    data["verify"]["checks"]["captions.contrast"]["ran"] = False
    _write_receipt(project, "final", data)
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    assert result["layer_checks"][0]["skipped"][0]["pass"] == "contrast"


@pytest.fixture
def checked_project(render: Path, tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    media = project / "final" / "sample.mp4"
    media.parent.mkdir(parents=True)
    shutil.copyfile(render, media)
    hashes = {"captions/captions.mov": "c" * 64, "graphics/agent-title.mov": "d" * 64,
              "graphics/agent-card.mov": "e" * 64}
    _write_receipt(project, "final", {
        "stage": "assemble", "outputs": [{"path": "final/sample.mp4",
                                           "sha256": sha256_file(media)}],
        "inputs": {f"media:{path}": digest for path, digest in hashes.items()},
        "timeline": {"fps": "30/1"},
    })
    _write_receipt(project, "captions", {
        "layer": {"path": "captions/captions.mov", "start_frame": 0, "frames": 90},
        "outputs": [{"path": "captions/captions.mov", "sha256": hashes["captions/captions.mov"]}],
        "layer_checks": _layer_checks("layout", "Caption overlaps", time=0.5),
    })
    _write_receipt(project, "graphics", {
        "elements": [{"id": "agent-title", "kind": "title", "path": "graphics/agent-title.mov",
                      "start_frame": 6, "frames": 33},
                     {"id": "agent-card", "kind": "graphic", "path": "graphics/agent-card.mov",
                      "start_frame": 45, "frames": 30}],
        "outputs": [{"path": path, "sha256": digest} for path, digest in hashes.items()
                    if path.startswith("graphics/")],
        "layer_checks": {"agent-title": _layer_checks("runtime", "Title failed", time=0.2,
                                                     severity="error"),
                         "agent-card": _layer_checks("contrast", "Card contrast is low")},
    })
    return project, media


@pytest.fixture
def behind_project(checked_project: tuple[Path, Path]) -> tuple[Path, Path]:
    project, media = checked_project
    data = json.loads((project / "final/receipt.json").read_text())
    data["behind_layers"] = [
        {"id": "agent-card", "start_frame": 45, "frames": 30, "hidden_frames": 12,
         "hidden_share": 0.4, "flag": True},
        {"id": "agent-title", "start_frame": 6, "frames": 33, "hidden_frames": 6,
         "hidden_share": 6 / 33, "flag": False},
    ]
    _write_receipt(project, "final", data)
    for stage in ("captions", "graphics"):
        data = json.loads((project / stage / "receipt.json").read_text())
        data.pop("layer_checks")
        _write_receipt(project, stage, data)
    return project, media


def test_flagged_behind_card_has_output_frames_text_and_evidence(
    behind_project: tuple[Path, Path],
) -> None:
    project, media = behind_project
    record_findings(media, [_point(1.5)], source="agent")
    data = json.loads((project / "final/receipt.json").read_text())
    data["timeline"]["fps"] = "24/1"
    data["behind_layers"][0]["frames"] = 20
    data["behind_layers"][0]["hidden_frames"] = 8
    _write_receipt(project, "final", data)
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    assert result["available"]
    assert [item["id"] for item in result["items"]] == ["f1", "l1"]
    assert result["items"][1] == {
        "id": "l1", "kind": "behind_hidden", "source": "machine", "reviewer": None,
        "severity": "warning", "start_frame": 45, "end_frame": 65,
        "start_s": 45 / 24, "end_s": 65 / 24,
        "reason": "Card agent-card is mostly hidden behind the speaker for 40 percent of its time",
        "fix": "Move the card or set it in front of the speaker", "target": "element",
        "evidence": data["behind_layers"][0],
    }


def test_behind_warning_is_available_without_layer_checks(
    behind_project: tuple[Path, Path],
) -> None:
    project, media = behind_project
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    assert result["available"] and len(result["items"]) == 1
    assert result["items"][0]["kind"] == "behind_hidden"
    assert "layer_checks" not in result


def test_moved_graphics_placement_keeps_measured_behind_timing(
    behind_project: tuple[Path, Path],
) -> None:
    project, media = behind_project
    digest = sha256_file(media)
    before = findings_for_media(media, digest, project_dir=project)
    data = json.loads((project / "graphics/receipt.json").read_text())
    card = next(layer for layer in data["elements"] if layer["id"] == "agent-card")
    card.update(start_frame=9, frames=60)
    _write_receipt(project, "graphics", data)
    after = findings_for_media(media, digest, project_dir=project)
    assert after == before
    assert (after["items"][0]["start_frame"], after["items"][0]["end_frame"]) == (45, 75)


@pytest.mark.parametrize("value", [None, False, 1, 0.4, "bad", {}, {"flag": True}])
def test_non_list_behind_measurements_keep_all_other_findings(
    checked_project: tuple[Path, Path], value: object,
) -> None:
    project, media = checked_project
    record_findings(media, [_point()], source="agent")
    digest = sha256_file(media)
    before = findings_for_media(media, digest, project_dir=project)
    data = json.loads((project / "final/receipt.json").read_text())
    data["behind_layers"] = value
    _write_receipt(project, "final", data)
    assert findings_for_media(media, digest, project_dir=project) == before


_VALID_BEHIND_MEASUREMENT = {
    "id": "agent-title", "start_frame": 6, "frames": 33,
    "hidden_frames": 33, "hidden_share": 1.0, "flag": True,
}


@pytest.mark.parametrize("entry", [
    None, False, 1, "bad", [], {},
    *[{key: value for key, value in _VALID_BEHIND_MEASUREMENT.items() if key != missing}
      for missing in ("id", "frames", "start_frame", "hidden_share", "flag")],
    *[{**_VALID_BEHIND_MEASUREMENT, field: value} for field, value in [
        ("id", None), ("id", 1), ("id", []), ("id", {}), ("id", ""),
        ("frames", None), ("frames", "33"), ("frames", True), ("frames", 33.0),
        ("frames", 0), ("frames", -1), ("frames", float("inf")),
        ("start_frame", None), ("start_frame", "6"), ("start_frame", True),
        ("start_frame", 6.0), ("start_frame", -1), ("start_frame", float("nan")),
        ("hidden_share", None), ("hidden_share", "1.0"), ("hidden_share", True),
        ("hidden_share", []), ("hidden_share", -0.1), ("hidden_share", 1.1),
        ("hidden_share", float("nan")), ("hidden_share", float("inf")),
        ("hidden_share", float("-inf")), ("hidden_share", 10 ** 400),
        ("flag", None), ("flag", "true"), ("flag", 1), ("flag", []),
        ("hidden_frames", float("nan")), ("hidden_frames", float("inf")),
        ("hidden_frames", True), ("hidden_frames", -1), ("hidden_frames", 34),
    ]],
])
def test_bad_behind_entries_keep_valid_measurements_and_all_other_findings(
    checked_project: tuple[Path, Path], entry: object,
) -> None:
    project, media = checked_project
    record_findings(media, [_point()], source="agent")
    data = json.loads((project / "final/receipt.json").read_text())
    data["behind_layers"] = [_VALID_BEHIND_MEASUREMENT]
    _write_receipt(project, "final", data)
    digest = sha256_file(media)
    before = findings_for_media(media, digest, project_dir=project)
    assert len(before["items"]) == 5
    assert before["items"][0]["kind"] == "behind_hidden"
    data["behind_layers"] = [entry, _VALID_BEHIND_MEASUREMENT, entry]
    _write_receipt(project, "final", data)
    assert findings_for_media(media, digest, project_dir=project) == before


def test_each_flagged_card_merges_with_layer_checks_in_time_order(
    checked_project: tuple[Path, Path],
) -> None:
    project, media = checked_project
    data = json.loads((project / "final/receipt.json").read_text())
    data["behind_layers"] = [
        {"id": "agent-card", "start_frame": 45, "frames": 30, "hidden_frames": 12,
         "hidden_share": 0.4, "flag": True},
        {"id": "agent-title", "start_frame": 6, "frames": 33, "hidden_frames": 33,
         "hidden_share": 1, "flag": True},
    ]
    _write_receipt(project, "final", data)
    record_findings(media, [_point(1.5)], source="agent")
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    assert [(item["id"], item["kind"], item["start_frame"])
            for item in result["items"]] == [
        ("l1", "behind_hidden", 6), ("l2", "layer_check", 12),
        ("l3", "layer_check", 15), ("f1", "slow", 45),
        ("l4", "behind_hidden", 45), ("l5", "layer_check", 45),
    ]
    assert result["items"][0]["end_frame"] == 39
    assert result["items"][0]["reason"] == (
        "Card agent-title is mostly hidden behind the speaker for 100 percent of its time"
    )


@pytest.mark.parametrize("flag", [False, None])
def test_unflagged_behind_cards_add_nothing(
    behind_project: tuple[Path, Path], flag: bool | None,
) -> None:
    project, media = behind_project
    data = json.loads((project / "final/receipt.json").read_text())
    if flag is None:
        data["behind_layers"][0].pop("flag")
    else:
        data["behind_layers"][0]["flag"] = flag
    _write_receipt(project, "final", data)
    assert findings_for_media(media, sha256_file(media), project_dir=project) == {
        "available": False, "file": None, "stale": [], "items": [],
    }


def test_non_final_media_has_no_behind_findings(behind_project: tuple[Path, Path]) -> None:
    project, media = behind_project
    assert findings_for_media(media, "a" * 64, project_dir=project) == {
        "available": False, "file": None, "stale": [], "items": [],
    }


@pytest.mark.parametrize("damage", ["hash", "status", "schema"])
def test_stale_final_receipt_has_no_behind_findings(
    behind_project: tuple[Path, Path], damage: str,
) -> None:
    project, media = behind_project
    data = json.loads((project / "final/receipt.json").read_text())
    if damage == "hash":
        data["outputs"][0]["sha256"] = "a" * 64
    elif damage == "status":
        data["status"] = "failed"
    else:
        data["schema"] = "foreign/stage-receipt@1"
    _write_receipt(project, "final", data)
    assert findings_for_media(media, sha256_file(media), project_dir=project) == {
        "available": False, "file": None, "stale": [], "items": [],
    }


def test_changed_graphics_receipt_has_no_behind_findings(
    behind_project: tuple[Path, Path],
) -> None:
    project, media = behind_project
    data = json.loads((project / "graphics/receipt.json").read_text())
    for output in data["outputs"]:
        output["sha256"] = "a" * 64
    _write_receipt(project, "graphics", data)
    assert findings_for_media(media, sha256_file(media), project_dir=project) == {
        "available": False, "file": None, "stale": [], "items": [],
        "layer_stale": ["graphics/receipt.json"],
    }


def test_final_layer_findings_merge_in_time_order_with_targets(
    checked_project: tuple[Path, Path],
) -> None:
    project, media = checked_project
    record_findings(media, [_point(0.4)], source="agent")
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    assert result["available"] and result["stale"] == []
    assert [(item["id"], item["start_frame"], item["end_frame"], item["target"])
            for item in result["items"]] == [
        ("f1", 12, None, "story"), ("l1", 12, 13, "element"),
        ("l2", 15, 16, "caption"), ("l3", 45, 75, "element"),
    ]
    layers = result["items"][1:]
    assert [item["kind"] for item in layers] == ["layer_check"] * 3
    assert [(item["reason"], item["pass"], item["severity"]) for item in layers] == [
        ("Title failed", "runtime", "error"), ("Caption overlaps", "layout", "warning"),
        ("Card contrast is low", "contrast", "warning"),
    ]
    assert [(item["start_s"], item["end_s"]) for item in layers] == [
        (0.4, 13 / 30), (0.5, 16 / 30), (1.5, 2.5),
    ]
    assert result["layer_checks"] == [
        {"path": "captions/captions.mov", "status": "findings", "skipped": []},
        {"path": "graphics/agent-title.mov", "status": "findings", "skipped": []},
        {"path": "graphics/agent-card.mov", "status": "findings", "skipped": []},
    ]
    assert str(project) not in json.dumps(result)


@pytest.mark.parametrize("stage", ["captions", "graphics"])
@pytest.mark.parametrize("severity", ["warning", "error"])
def test_lint_pass_stays_in_receipts_without_review_items_or_count(
    checked_project: tuple[Path, Path], stage: str, severity: str,
) -> None:
    project, media = checked_project
    digest = sha256_file(media)
    before = findings_for_media(media, digest, project_dir=project)
    path = project / stage / "receipt.json"
    data = json.loads(path.read_bytes())
    checks = data["layer_checks"] if stage == "captions" else data["layer_checks"]["agent-card"]
    lint = _layer_checks("lint", "This HTML composition file has 925 lines.",
                         severity=severity)["passes"]["lint"]
    lint["errors" if severity == "error" else "warnings"] = 2
    checks["passes"]["lint"] = lint
    _write_receipt(project, stage, data)
    saved = path.read_bytes()

    result = findings_for_media(media, digest, project_dir=project)

    assert len(result["items"]) == len(before["items"]) == 3
    assert result == before
    assert [item["pass"] for item in result["items"]] == ["runtime", "layout", "contrast"]
    assert path.read_bytes() == saved
    assert json.loads(saved)["layer_checks"] == data["layer_checks"]


@pytest.mark.parametrize(("stage", "path", "message"), [
    ("captions", "captions/captions.mov", "Caption overlaps"),
    ("graphics", "graphics/agent-card.mov", "Card contrast is low"),
])
def test_changed_layer_receipt_is_stale_and_contributes_no_items(
    checked_project: tuple[Path, Path], stage: str, path: str, message: str,
) -> None:
    project, media = checked_project
    receipt_path = project / stage / "receipt.json"
    data = json.loads(receipt_path.read_text())
    next(output for output in data["outputs"] if output["path"] == path)["sha256"] = "a" * 64
    _write_receipt(project, stage, data)
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    assert result["stale"] == []
    assert result["layer_stale"] == [f"{stage}/receipt.json"]
    assert result["available"]
    assert len(result["items"]) == 2
    assert all(item["reason"] != message for item in result["items"])
    assert all(layer["path"] != path for layer in result["layer_checks"])


def test_non_final_media_has_no_layer_findings(checked_project: tuple[Path, Path]) -> None:
    project, media = checked_project
    result = findings_for_media(media, "a" * 64, project_dir=project)
    assert result == {"available": False, "file": None, "stale": [], "items": []}


def test_saved_copy_of_matching_final_has_layer_findings(
    checked_project: tuple[Path, Path],
) -> None:
    project, media = checked_project
    saved = project / "saved.mp4"
    shutil.copyfile(media, saved)
    assert len(findings_for_media(saved, sha256_file(saved), project_dir=project)["items"]) == 3


def test_layer_summary_keeps_skipped_passes_explicit_without_findings(
    checked_project: tuple[Path, Path],
) -> None:
    project, media = checked_project
    data = json.loads((project / "captions/receipt.json").read_text())
    checks = data["layer_checks"]
    checks["status"] = "incomplete"
    checks["passes"]["layout"] = {"status": "passed", "findings": []}
    skipped = [{"pass": "motion", "reason": "motion check not enabled"},
               {"pass": "contrast", "reason": "no text on an opaque background in this layer; "
                "captions are measured on the final"}]
    for check in skipped:
        checks["passes"][check["pass"]] = {"status": "skipped", "reason": check["reason"],
                                           "findings": []}
    _write_receipt(project, "captions", data)
    _write_receipt(project, "graphics", {"elements": [], "outputs": [], "layer_checks": {}})
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    assert result["available"] and result["items"] == []
    assert result["layer_checks"] == [{"path": "captions/captions.mov", "status": "incomplete",
                                      "skipped": skipped}]


def test_layer_ids_sort_numerically_and_info_findings_are_ignored(
    checked_project: tuple[Path, Path],
) -> None:
    project, media = checked_project
    data = json.loads((project / "captions/receipt.json").read_text())
    check = data["layer_checks"]["passes"]["layout"]
    check["findings"] *= 12
    check["findings"].append({"severity": "info", "message": "Informational", "time": 0})
    _write_receipt(project, "captions", data)
    items = findings_for_media(media, sha256_file(media), project_dir=project)["items"]
    assert [item["id"] for item in items if item["target"] == "caption"] == [
        f"l{index}" for index in range(2, 14)
    ]
    assert len(items) == 14


@pytest.mark.parametrize(("time", "expected"), [(-1, 0), (3, 89), (20, 89)])
def test_timed_layer_findings_stay_inside_layer(
    checked_project: tuple[Path, Path], time: float, expected: int,
) -> None:
    project, media = checked_project
    data = json.loads((project / "captions/receipt.json").read_text())
    data["layer_checks"]["passes"]["layout"]["findings"][0]["time"] = time
    _write_receipt(project, "captions", data)
    items = findings_for_media(media, sha256_file(media), project_dir=project)["items"]
    item = next(item for item in items if item["target"] == "caption")
    assert (item["start_frame"], item["end_frame"]) == (expected, expected + 1)


@pytest.mark.parametrize("layer_kind", ["caption", "title", "graphic"])
def test_layer_note_target(layer_kind: str) -> None:
    assert note_target("layer_check", layer_kind=layer_kind) == (
        "caption" if layer_kind == "caption" else "element"
    )


def test_skipped_pass_summary_survives_overall_findings_status(
    checked_project: tuple[Path, Path],
) -> None:
    project, media = checked_project
    data = json.loads((project / "captions/receipt.json").read_text())
    data["layer_checks"]["passes"]["motion"] = {
        "status": "skipped", "errors": 0, "warnings": 0, "findings": [],
        "reason": "motion check not enabled",
    }
    _write_receipt(project, "captions", data)
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    assert len(result["items"]) == 3
    assert result["layer_checks"][0] == {
        "path": "captions/captions.mov", "status": "findings",
        "skipped": [{"pass": "motion", "reason": "motion check not enabled"}],
    }


@pytest.mark.parametrize(
    ("errors", "warnings", "listed_errors", "listed_warnings", "truncated", "message", "severity"),
    [
        (4, 0, 1, 0, 0, "3 more runtime errors not listed", "error"),
        (0, 3, 0, 1, 2, "2 more runtime warnings not listed", "warning"),
        (3, 4, 1, 1, 3, "2 more runtime errors and 3 more runtime warnings not listed", "error"),
        (2, 0, 0, 0, 0, "2 more runtime errors not listed", "error"),
        (0, 2, 0, 0, 0, "2 more runtime warnings not listed", "warning"),
        (1, 1, 1, 0, 1, "1 more runtime warning not listed", "warning"),
        (1, 0, 1, 0, 0, None, None),
        (0, 0, 0, 0, 2, None, None),
    ],
)
def test_unlisted_pass_counts_add_one_whole_layer_finding(
    checked_project: tuple[Path, Path], errors: int, warnings: int, listed_errors: int,
    listed_warnings: int, truncated: int, message: str | None, severity: str | None,
) -> None:
    project, media = checked_project
    data = json.loads((project / "graphics/receipt.json").read_text())
    findings = [{"severity": level, "message": f"Listed {level}", "time": 0.2}
                for level, count in (("error", listed_errors), ("warning", listed_warnings))
                for _ in range(count)]
    data["layer_checks"]["agent-title"]["passes"]["runtime"] = {
        "status": "errors" if errors else "warnings" if warnings else "passed",
        "errors": errors, "warnings": warnings, "findings": findings,
        "findings_truncated": truncated,
    }
    _write_receipt(project, "graphics", data)
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    items = [item for item in result["items"] if item["pass"] == "runtime"]
    assert len(items) == len(findings) + int(message is not None)
    listed = [item for item in items if item["reason"].startswith("Listed")]
    assert all((item["start_frame"], item["end_frame"]) == (12, 13) for item in listed)
    if message is not None:
        unlisted = next(item for item in items if item["reason"] == message)
        assert unlisted["severity"] == severity
        assert unlisted["target"] == "element" and unlisted["kind"] == "layer_check"
        assert (unlisted["start_frame"], unlisted["end_frame"]) == (6, 39)
        assert (unlisted["start_s"], unlisted["end_s"]) == (0.2, 1.3)


def test_layer_stale_receipts_are_separate_from_reviewer_stale_files(
    checked_project: tuple[Path, Path],
) -> None:
    project, media = checked_project
    record_findings(media, [_point()], source="agent")
    path = default_findings_path(media)
    notes = json.loads(path.read_text())
    notes["video"]["sha256"] = "a" * 64
    path.write_text(json.dumps(notes), encoding="utf-8")
    data = json.loads((project / "graphics/receipt.json").read_text())
    for output in data["outputs"]:
        output["sha256"] = "f" * 64
    _write_receipt(project, "graphics", data)
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    assert result["available"] and len(result["items"]) == 1
    assert result["stale"] == ["final/sample.findings.json"]
    assert result["layer_stale"] == ["graphics/receipt.json"]
    assert result["items"][0]["target"] == "caption"


def test_layer_stale_receipts_remain_when_no_findings_are_available(
    checked_project: tuple[Path, Path],
) -> None:
    project, media = checked_project
    for stage in ("captions", "graphics"):
        data = json.loads((project / stage / "receipt.json").read_text())
        for output in data["outputs"]:
            output["sha256"] = "f" * 64
        _write_receipt(project, stage, data)
    result = findings_for_media(media, sha256_file(media), project_dir=project)
    assert result == {
        "available": False, "file": None, "stale": [], "items": [],
        "layer_stale": ["captions/receipt.json", "graphics/receipt.json"],
    }
