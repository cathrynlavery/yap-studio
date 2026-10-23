"""Creator vocabulary storage and commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yap_studio import cli
from yap_studio.brand_words import add_words, list_words, remove_words, set_words
from yap_studio.errors import ValidationError


def test_add_remove_set_and_cli_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    assert add_words(["Sparrowlight", "PaperKettle"]) == ["Sparrowlight", "PaperKettle"]
    assert add_words(["SparrowLight"]) == ["PaperKettle", "SparrowLight"]
    target = tmp_path / "config/brand/words.json"
    assert json.loads(target.read_text()) == {
        "schema": "yap-studio/brand-words@1", "words": ["PaperKettle", "SparrowLight"]}
    assert remove_words(["PAPERKETTLE"]) == ["SparrowLight"]
    source = tmp_path / "words.txt"
    source.write_text("# creator spelling\nInstrument Serif\n\nsparrowlight\nSparrowLight\n")
    assert cli.main(["brand", "words", "set", str(source)]) == 0
    capsys.readouterr()
    assert cli.main(["brand", "words", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == ["Instrument Serif", "SparrowLight"]
    assert list_words() == ["Instrument Serif", "SparrowLight"]
    assert cli.main(["brand", "words", "add", "PaperKettle"]) == 0
    assert cli.main(["brand", "words", "remove", "sparrowlight"]) == 0
    assert list_words() == ["Instrument Serif", "PaperKettle"]
    capsys.readouterr()


def test_validation_and_cli_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    for bad in ("", "x" * 61, "bad\nword"):
        with pytest.raises(ValidationError):
            add_words([bad])
    with pytest.raises(ValidationError):
        set_words([f"Word {i}" for i in range(501)])
    assert cli.main(["brand", "words", "set", str(tmp_path / "missing.txt")]) == 2
    assert "error:" in capsys.readouterr().err
    bad_root = tmp_path / "not-a-directory"
    bad_root.write_text("x")
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(bad_root))
    assert cli.main(["brand", "words", "add", "Brand"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_add_file_merges_with_stored_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    source = tmp_path / "words.txt"
    source.write_text("# spelling\n\nSparrowLight\n")
    assert cli.main(["brand", "words", "add", "Yap Studio"]) == 0
    assert cli.main(["brand", "words", "add", "--file", str(source)]) == 0
    assert list_words() == ["Yap Studio", "SparrowLight"]
    assert cli.main(["brand", "words", "add", "PaperKettle", "--file", str(source)]) == 0
    assert list_words() == ["Yap Studio", "PaperKettle", "SparrowLight"]


def test_cli_add_requires_words_or_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["brand", "words", "add"]) == 2
    assert "words or --file" in capsys.readouterr().err.lower()
