"""Ingest, preflight, and word-listing commands and their exit contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_ingest import FakeBackend, add_references, make_media, make_project

from yap_studio import transcript
from yap_studio.cli import main
from yap_studio.stages.ingest import ingest
from yap_studio.transcript import Word


@pytest.fixture
def fixture_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, FakeBackend]:
    """Create CLI inputs with a deterministic speech backend."""
    fake = FakeBackend()
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_args: fake)
    root = make_project(tmp_path, [make_media(tmp_path / "take.mkv")])
    return root, fake


def test_ingest_and_words_text(
    fixture_project: tuple[Path, FakeBackend],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = fixture_project
    assert main(["ingest", str(root)]) == 0
    assert "Ingest verified." in capsys.readouterr().out
    assert main(["words", str(root)]) == 0
    assert capsys.readouterr().out == (
        "t1:0-t1:5  0.200-2.000  Here's the thing nobody tells you.\n"
    )
    assert main(["ingest", str(root)]) == 0
    assert "Ingest unchanged." in capsys.readouterr().out


def test_cli_force_refreshes_shared_transcript(fixture_project, tmp_path, capsys):
    root, backend = fixture_project
    assert main(["ingest", str(root)]) == 0
    other = make_project(tmp_path / "other", [root / "take.mkv"])
    backend.calls.clear()
    assert main(["ingest", str(other), "--json"]) == 0
    assert backend.calls == []
    capsys.readouterr()
    assert main(["ingest", str(other), "--force", "--json"]) == 0
    assert len(backend.calls) == 1
    assert json.loads(capsys.readouterr().out)["noop"] is False


def test_ingest_words_and_status_json(
    fixture_project: tuple[Path, FakeBackend],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = fixture_project
    assert main(["ingest", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "yap-studio/ingest@1"
    assert result["takes"][0]["id"] == "t1"
    assert main(["words", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["words"][0]["id"] == "t1:0"
    assert result["time_base"] == "source"
    assert main(["status", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["next"] == "write edit.json segments (read `yap-studio words DIR`)"


def test_status_has_empty_warnings_after_clean_ingest(
    fixture_project: tuple[Path, FakeBackend],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = fixture_project
    assert main(["ingest", str(root)]) == 0
    capsys.readouterr()
    assert main(["status", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["warnings"] == []


def test_default_directory(
    fixture_project: tuple[Path, FakeBackend],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = fixture_project
    monkeypatch.chdir(root)
    assert main(["ingest"]) == 0
    capsys.readouterr()
    assert main(["words"]) == 0
    assert "t1:0-t1:5" in capsys.readouterr().out


def test_refusal_exit_and_force_orphans(
    fixture_project: tuple[Path, FakeBackend],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, fake = fixture_project
    ingest(root)
    add_references(root)
    fake.words = [Word("Changed", 0.2, 0.4)]
    fake.model = "fixture-v2"
    assert main(["ingest", str(root), "--asr", "parakeet-mlx", "--json"]) == 3
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "words_referenced"
    assert "--force" in error["suggested_action"]
    assert main(["ingest", str(root), "--force"]) == 0
    output = capsys.readouterr().out
    assert output.count("orphan:") == 6
    assert "timeline/agent.json" in output
    assert "next: yap-studio edit rebase DIR" in output


def test_preflight_pass_and_fail(
    fixture_project: tuple[Path, FakeBackend],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, fake = fixture_project
    assert main(["ingest", "--preflight", str(root / "take.mkv")]) == 0
    assert capsys.readouterr().out.startswith("PASS\n")
    fake.words = []
    silence = make_media(root / "silence.mkv", sound="silence")
    assert main(["ingest", "--preflight", str(silence)]) == 1
    output = capsys.readouterr().out
    assert output.startswith("FAIL\n")
    assert "at_least_five_words" in output
    assert main(["ingest", "--preflight", str(silence), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


def test_input_errors_exit_two(
    fixture_project: tuple[Path, FakeBackend],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = fixture_project
    assert main(["words", str(root), "--json"]) == 2
    assert "run yap-studio ingest first" in capsys.readouterr().err
    assert main(["words", str(root), "--cut"]) == 2
    assert "run yap-studio cut first" in capsys.readouterr().err
    assert main(["ingest", str(root), "--preflight", str(root / "take.mkv")]) == 2
    assert "without DIR" in capsys.readouterr().err
    no_audio = make_media(root / "no-audio.mkv", sound="none")
    assert main(["ingest", "--preflight", str(no_audio), "--json"]) == 2
    assert "the take has no audio" in capsys.readouterr().err


def test_copy_and_asr_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen = []
    fake = FakeBackend()

    def resolve(name: str, _model: str | None = None) -> FakeBackend:
        seen.append(name)
        return fake

    monkeypatch.setattr(transcript, "resolve_backend", resolve)
    take = make_media(tmp_path / "external.mkv")
    root = make_project(tmp_path / "project", [take])
    assert main(["ingest", str(root), "--copy", "--asr", "parakeet-mlx", "--json"]) == 0
    assert seen == ["parakeet-mlx"]
    assert json.loads(capsys.readouterr().out)["takes"][0]["copy"] == "ingest/takes/t1.mkv"


def test_plain_warnings_printed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_args: FakeBackend())
    take = make_media(tmp_path / "hdr.mkv", transfer="arib-std-b67", fps=24, sound="tone")
    root = make_project(tmp_path, [take])
    assert main(["ingest", str(root)]) == 0
    output = capsys.readouterr().out
    assert "warning: t1: HDR" in output
    assert "24 fps source" in output
    assert "music under the voice" in output
