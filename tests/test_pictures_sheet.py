"""Prompt validation, own media gaps and immutable cost sheets."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from pictures_helpers import record_answer
from test_media_select import STAMP, _item
from test_stock_provider import calls
from test_stock_sheet import words

from yap_studio.errors import ValidationError, YapError
from yap_studio.pictures import guard, wavespeed
from yap_studio.pictures.sheet import REQUESTS_SCHEMA, ask_sheet, money
from yap_studio.receipt import sha256_file

STYLE = {"broll": {"generated_pictures": "ask"}}


def requests(count=1):
    return {
        "schema": REQUESTS_SCHEMA,
        "lines": [
            {
                "id": f"pic-park{i}",
                "start": {"word": "t1:0"},
                "end": {"word": "t1:1"},
                "prompt": f"A quiet park at dusk, view {i}",
            }
            for i in range(count)
        ],
    }


def ask(root, **kwargs):
    record_answer(root)
    return ask_sheet(
        kwargs.pop("requests", requests()),
        kwargs.pop("cut_words", words()),
        kwargs.pop("catalog_items", []),
        kwargs.pop("style", STYLE),
        project_dir=root,
        **kwargs,
    )


def test_missing_cli_is_one_line_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(wavespeed, "estimate", lambda _: pytest.fail("missing CLI must not price"))
    root = tmp_path / "video"
    result = ask(root)
    assert result["status"] == "unavailable" and result["reason"] == "missing_cli"
    assert "\n" not in result["message"]
    assert result["message"].endswith(wavespeed.SIGNUP_LINE)
    assert "own footage" in result["message"]
    assert not (root / "pictures/sheets").exists()
    assert not (root / "pictures/pictures.json").exists()
    assert result["setup"] == [wavespeed.INSTALL_LINE, wavespeed.KEY_LINE]


@pytest.mark.parametrize(
    "field,value,location",
    [
        ("schema", "bad", "schema"),
        ("lines", [], "lines"),
        ("extra", 1, "requests.extra"),
        ("line.extra", 1, "lines[0].extra"),
        ("line.id", "bad", "lines[0].id"),
        ("line.prompt", "ab", "lines[0].prompt"),
        ("line.prompt", "bad\nline", "lines[0].prompt"),
        ("line.prompt", "x" * 401, "lines[0].prompt"),
        ("line.start", {"time": 0}, "lines[0].start"),
        ("line.start", {"word": "t1:0", "bad": 1}, "lines[0].start.bad"),
        ("line.end", {"word": "t1:99"}, "lines[0].end.word"),
        ("line.not_fitting", ["x"], "lines[0].not_fitting"),
    ],
)
def test_validation_locations(tmp_path, fake_wavespeed, field, value, location):
    data = requests()
    if field.startswith("line."):
        data["lines"][0][field[5:]] = value
    else:
        data[field] = value
    with pytest.raises(ValidationError) as error:
        ask(tmp_path / "video", requests=data)
    assert error.value.parameter == location
    assert not (tmp_path / "video/pictures/sheets").exists()
    assert not (tmp_path / "video/pictures/pictures.json").exists()


def test_duplicate_count_time_base_and_window(tmp_path, fake_wavespeed):
    data = requests()
    data["lines"] *= 2
    with pytest.raises(ValidationError, match=r"lines\[1\].id"):
        ask(tmp_path, requests=data)
    with pytest.raises(ValidationError, match="1 to 8"):
        ask(tmp_path, requests=requests(9))
    with pytest.raises(ValidationError, match="cut_words.time_base"):
        ask(tmp_path, cut_words=replace(words(), time_base="source"))
    short = replace(
        words(),
        words=tuple(replace(w, start=w.start / 100, end=w.end / 100) for w in words().words),
    )
    with pytest.raises(ValidationError, match=r"lines\[0\].end.*span 1 to 4"):
        ask(tmp_path, cut_words=short)


def test_sheet_prompts_prices_total_and_no_generation(tmp_path, fake_wavespeed):
    data = requests(3)
    data["lines"][0]["prompt"] = "A park <b>& trees"
    result = ask(tmp_path, requests=data)
    folder = tmp_path / "pictures/sheets/s1"
    assert json.loads((folder / "ask-sheet.json").read_text()) == result
    assert [p["estimate_usd"] for p in result["lines"]] == [0.027] * 3
    assert result["total_usd"] == 0.081
    assert result["ask"][0]["question"] == (
        'Generate a picture for "Dog park." (0:12.4 to 0:15.1)? About $0.027.'
    )
    assert result["ask"][0]["options"][0]["code"] == "1"
    html = (folder / "ask-sheet.html").read_text()
    assert "A park &lt;b&gt;&amp; trees" in html and "A park <b>" not in html
    assert "Total for every line: about $0.081" in html
    assert "Made with WaveSpeed (" + wavespeed.MODEL_ID + ", 1080 x 1920)" in html
    assert "Nothing is made until you say yes." in html
    prices = [a for a in calls(fake_wavespeed) if a[0] == "price"]
    assert len(prices) == 3 and not [a for a in calls(fake_wavespeed) if a[0] == "run"]
    for argv in prices:
        assert wavespeed.MODEL_ID in argv and "size=" + wavespeed.SIZE in argv
    before = (folder / "ask-sheet.json").read_bytes()
    assert ask(tmp_path)["sheet"] == "s2"
    assert (folder / "ask-sheet.json").read_bytes() == before
    assert [money(v) for v in (0, 1, 0.027, 0.08)] == ["$0.00", "$1.00", "$0.027", "$0.08"]


def test_own_fit_skips_and_other_sources_do_not(tmp_path, fake_wavespeed):
    source = tmp_path / "own.mov"
    source.write_bytes(b"synthetic own footage")
    item = _item(3, locations=[{"path": str(source), "added_at": STAMP, "seen_at": STAMP}])
    item["sha256"] = sha256_file(source)
    result = ask(tmp_path, catalog_items=[item])
    assert not result["lines"] and "m3" in result["skipped"][0]["reason"]
    assert not [a for a in calls(fake_wavespeed) if a[0] == "price"]
    data = requests()
    data["lines"][0]["not_fitting"] = ["m3"]
    assert ask(tmp_path, requests=data, catalog_items=[item])["lines"]
    for key in ("stock", "generated"):
        assert ask(tmp_path, catalog_items=[{**item, key: {}}])["lines"]
    source.unlink()
    assert ask(tmp_path, catalog_items=[item])["lines"]


def test_style_and_project_sources(tmp_path, fake_wavespeed, monkeypatch):
    with pytest.raises(YapError) as error:
        ask(tmp_path, style={})
    assert error.value.code == "pictures_off"
    source = tmp_path / "project.json"
    source.write_text(json.dumps({"broll_sources": ["folder"]}))
    assert guard.generated_allowed(STYLE, tmp_path) == (True, None)
    assert guard.generated_allowed({}, tmp_path) == (
        False, "Generated pictures are off in this style."
    )
    monkeypatch.setattr(wavespeed, "status", lambda: "signed_out")
    assert ask(tmp_path)["reason"] == "signed_out"
    assert not (tmp_path / "pictures/sheets").exists()
    assert not (tmp_path / "pictures/pictures.json").exists()
    source.write_text("unreadable JSON")
    assert guard.generated_allowed(STYLE, tmp_path) == (True, None)


def test_estimate_failure_writes_nothing(tmp_path, fake_wavespeed, monkeypatch):
    monkeypatch.setenv("FAKE_WAVESPEED_PRICE_EXIT", "7")
    result = ask(tmp_path / "video")
    assert result["reason"] == "rate_limited"
    assert not (tmp_path / "video/pictures/sheets").exists()
    assert not (tmp_path / "video/pictures/pictures.json").exists()
