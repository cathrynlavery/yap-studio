"""`taste learn --save`: the learned style joins the library and its rules wait for approval."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_cli_taste import _body, isolated_taste  # noqa: F401
from test_cli_taste import learn_clip as learn_clip

from yap_studio import cli
from yap_studio.style import get_default, list_entries, set_default
from yap_studio.style.library import library_dir
from yap_studio.taste import load_proposals, rule_id
from yap_studio.taste.learn import LearnResult
from yap_studio.taste.learned import LEARNED_NAME, save_learned, source_path
from yap_studio.taste.model import TasteError
from yap_studio.taste.store import proposals_path


def test_save_adds_the_learned_style_without_making_it_default(
    learn_clip: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "learned"
    args = ["taste", "learn", str(learn_clip), "--out", str(out), "--asr", "none", "--save"]
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert (f"Saved style: {LEARNED_NAME} (learned) from 1 finished video. "
            "It is not your default until you pick it.") in text
    assert [(entry.id, entry.name, entry.is_default) for entry in list_entries()] == [
        ("learned", LEARNED_NAME, False)]
    assert get_default() is None
    source = json.loads(source_path("learned").read_text())
    assert source["schema"] == "yap-studio/learned-source@1"
    assert source["videos"] == 1 and source["names"] == ["vertical.mp4"]
    saved = json.loads((library_dir() / "learned.json").read_text())
    proposed = json.loads((out / "proposed-style.json").read_text())
    assert saved == {**proposed, "name": LEARNED_NAME}

    assert cli.main([*args, "--force", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["saved"]["style"]["id"] == "learned" and data["saved"]["style"]["videos"] == 1
    assert [entry.id for entry in list_entries()] == ["learned"]


def test_plain_learn_still_saves_nothing(learn_clip: Path, tmp_path: Path) -> None:
    assert cli.main(["taste", "learn", str(learn_clip), "--out", str(tmp_path / "o"),
                     "--asr", "none"]) == 0
    assert not library_dir().exists()


def _result(tmp_path: Path, proposals: list[dict], color: str = "#F8EACA") -> LearnResult:
    from yap_studio.style import PRESETS_DIR

    style = json.loads((PRESETS_DIR / "clean_minimal.json").read_text())
    style["captions"]["text_color"] = color
    style_path = tmp_path / f"style-{color[1:]}.json"
    style_path.write_text(json.dumps(style))
    proposals_path = tmp_path / f"proposals-{color[1:]}.json"
    proposals_path.write_text(json.dumps({"schema": "yap-studio/taste-proposals@1",
                                          "proposals": proposals}))
    measurements = {
        "created_at": "2026-10-12T10:00:00+00:00",
        "videos": [{"name": f"v{index}.mp4"} for index in range(6)],
        "fields": {}, "measures": {}, "not_measured": [],
        "proposed_style": {"base": {"name": "Clean minimal"}, "style": style, "sources": {
            "captions.text_color": {"source": "measured", "status": "proposed", "reason": None},
        }},
    }
    return LearnResult(style_path, proposals_path, tmp_path / "measurements.json", measurements)


def test_save_queues_rules_for_approval(tmp_path: Path) -> None:
    saved = save_learned(_result(tmp_path, [_body()]))
    assert saved["rules"] == [{"id": rule_id(_body()), "status": "added"}]
    assert [item["id"] for item in load_proposals().pending] == [rule_id(_body())]
    assert saved["style"]["videos"] == 6 and saved["style"]["default"] is False
    assert json.loads(source_path("learned").read_text())["measured_fields"] == [
        "captions.text_color"]


def test_relearning_never_changes_a_picked_default(tmp_path: Path) -> None:
    first = save_learned(_result(tmp_path, []))
    set_default(first["style"]["id"])
    before = (library_dir() / "learned.json").read_text()
    second = save_learned(_result(tmp_path, [], color="#112233"))
    assert second["style"]["id"] == "learned-2" and second["style"]["default"] is False
    assert get_default() == "learned"
    assert (library_dir() / "learned.json").read_text() == before
    assert sorted(entry.id for entry in list_entries()) == ["learned", "learned-2"]
    again = save_learned(_result(tmp_path, []))
    assert again["style"]["id"] == "learned" and again["style"]["default"] is True


def test_relearning_skips_creator_styles_with_learned_ids(tmp_path: Path) -> None:
    first = save_learned(_result(tmp_path, []))
    set_default(first["style"]["id"])
    creator = library_dir() / "learned-2.json"
    creator.write_bytes(_result(tmp_path, [], color="#112233").style_path.read_bytes())
    before = creator.read_bytes()

    saved = save_learned(_result(tmp_path, [], color="#445566"))

    assert saved["style"]["id"] == "learned-3"
    assert creator.read_bytes() == before
    assert not source_path("learned-2").exists()


def test_save_skips_creator_style_named_learned(tmp_path: Path) -> None:
    creator = library_dir() / "learned.json"
    creator.parent.mkdir(parents=True)
    creator.write_bytes(_result(tmp_path, [], color="#112233").style_path.read_bytes())
    before = creator.read_bytes()

    saved = save_learned(_result(tmp_path, []))

    assert saved["style"]["id"] == "learned-2"
    assert creator.read_bytes() == before
    assert not source_path("learned").exists()


def test_save_skips_another_source_record(tmp_path: Path) -> None:
    source = source_path("learned")
    source.parent.mkdir(parents=True)
    source.write_bytes(b'{"schema":"yap-studio/other@1","value":1}\n')
    before = source.read_bytes()

    saved = save_learned(_result(tmp_path, []))

    assert saved["style"]["id"] == "learned-2"
    assert source.read_bytes() == before
    assert not (library_dir() / "learned.json").exists()


def test_malformed_proposals_store_leaves_library_unchanged(tmp_path: Path) -> None:
    save_learned(_result(tmp_path, []))
    style = library_dir() / "learned.json"
    source = source_path("learned")
    before = style.read_bytes(), source.read_bytes()
    proposals_path().write_text('{"schema":"yap-studio/wrong@1"}')

    with pytest.raises(TasteError):
        save_learned(_result(tmp_path, [], color="#112233"))

    assert (style.read_bytes(), source.read_bytes()) == before


def test_failed_rule_import_restores_previous_style_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_learned(_result(tmp_path, []))
    style = library_dir() / "learned.json"
    source = source_path("learned")
    before = style.read_bytes(), source.read_bytes()
    result = _result(tmp_path, [], color="#112233")
    result.measurements["created_at"] = "2026-10-13T10:00:00+00:00"

    def fail(_path: Path) -> None:
        raise RuntimeError("forced rule import failure")

    monkeypatch.setattr("yap_studio.taste.learned.add_proposals", fail)
    with pytest.raises(RuntimeError, match="forced rule import failure"):
        save_learned(result)

    assert (style.read_bytes(), source.read_bytes()) == before
