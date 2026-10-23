"""Consent, spend limits, cached generation and catalog provenance."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import synth_broll
from test_pictures_sheet import STYLE, ask, requests
from test_stock_provider import calls
from test_stock_sheet import words
from test_timeline import write

from yap_studio import cli, receipt
from yap_studio.errors import InputError, ValidationError, YapError
from yap_studio.media import catalog
from yap_studio.pictures import guard, wavespeed
from yap_studio.pictures.make import make_pictures

NOW = datetime(2026, 10, 19, 12, tzinfo=UTC)


@pytest.fixture
def asked(tmp_path, fake_wavespeed):
    root = tmp_path / "video"
    root.mkdir()
    write(root / "cut/cut-words.json", words().to_dict())
    write(root / "style.json", {"schema_version": 1, **STYLE})
    write(
        root / "project.json",
        {
            "schema": "yap-studio/project@1",
            "name": "video",
            "style": "style.json",
            "takes": [{"id": "t1", "path": "own.mov"}],
            "asr": {"engine": "none"},
        },
    )
    ask(root, requests=requests(4))
    return root


def make(root, numbers=("1",), **kwargs):
    return make_pictures(
        root,
        list(numbers),
        yes=kwargs.pop("yes", True),
        max_cost=kwargs.pop("max_cost", 1.0),
        now=NOW,
        **kwargs,
    )


def runs(log):
    return [a for a in calls(log) if a[0] == "run" and "--dry-run" not in a]


def test_consent_picks_and_ceiling_preflight(asked, fake_wavespeed):
    with pytest.raises(YapError) as error:
        make(asked, yes=False)
    assert error.value.code == "pictures_need_yes"
    assert error.value.suggested_action == (
        "Show the ask sheet with its costs, ask, then pass --yes and --max-cost."
    )
    for picks in (("1", "1"), ("9",), (), ("1", "2", "3", "4")):
        with pytest.raises(ValidationError) as error:
            make(asked, picks)
        assert error.value.parameter == "pick"
    for ceiling in (None, True, 0, -1, 10.01, float("nan"), float("inf")):
        with pytest.raises(ValidationError) as error:
            make(asked, max_cost=ceiling)
        assert error.value.parameter == "max_cost"
    with pytest.raises(YapError) as error:
        make(asked, ("1", "2"), max_cost=0.05)
    assert error.value.code == "pictures_over_ceiling" and "$0.054" in str(error.value)
    assert not runs(fake_wavespeed)


def test_dry_run_no_consent_no_ceiling_no_writes(asked, fake_wavespeed):
    before = {str(p): p.read_bytes() for p in asked.rglob("*") if p.is_file()}
    result = make(asked, ("1", "2"), yes=False, max_cost=None, dry_run=True)
    assert result["status"] == "dry_run" and len(result["previews"]) == 2
    assert all("Authorization" not in p["request"] for p in result["previews"])
    assert all("POST https://" in p["request"] for p in result["previews"])
    assert before == {str(p): p.read_bytes() for p in asked.rglob("*") if p.is_file()}
    assert not runs(fake_wavespeed)
    assert all("--dry-run" in a for a in calls(fake_wavespeed) if a[0] == "run")


def test_success_catalog_record_cost_request_and_cache(asked, fake_wavespeed):
    result = make(asked)
    assert result["status"] == "ok" and not result["failed"], result
    entry = result["made"][0]
    target = asked / entry["path"]
    assert target.is_file() and receipt.sha256_file(target) == entry["sha256"]
    assert target.parent == asked / "pictures/files"
    assert result["spent_usd"] == 0.027
    assert entry["charged_usd"] == 0.027 and entry["cost_source"] == "charged"
    assert entry["approved_at"] == entry["created_at"] == "2026-10-19T12:00:00Z"
    assert result["requests"] == [entry["request"]]
    assert entry["request"] == {
        "id": "broll-pic-park0",
        "item": "m1",
        "kind": "insert",
        "start": {"word": "t1:0"},
        "end": {"word": "t1:1"},
        "audio": "mute",
    }
    assert guard.approved_pictures(asked) == [entry]
    item = catalog.load_catalog()["items"][0]
    assert item["generated"] == {
        "provider": "wavespeed",
        "model": wavespeed.MODEL_ID,
        "prompt": entry["prompt"],
        "seed": None,
        "size": wavespeed.SIZE,
        "prediction_id": entry["prediction_id"],
        "cost_usd": 0.027,
        "cost_source": "charged",
        "created_at": entry["created_at"],
    }
    assert item["provenance"] == {
        "kind": "generated",
        "source": "WaveSpeed",
        "license": "WaveSpeed Terms of Service",
        "url": "https://wavespeed.ai/static/terms",
    }
    assert item["description"]["tags"] == ["generated_picture"]
    assert item["description"]["by"] == "generated"
    assert "access_key" not in json.dumps(entry)
    argv = runs(fake_wavespeed)[0]
    assert "--no-learn" in argv and "--download" not in argv
    assert any(a.startswith("--download=") for a in argv)
    before = calls(fake_wavespeed)
    cached = make(asked, max_cost=0.001)
    assert cached["cached"] == [entry] and cached["spent_usd"] == 0
    assert not cached["made"]
    assert calls(fake_wavespeed) == before
    assert not list((asked / "pictures").glob(".make-*"))


@pytest.mark.parametrize("setting", ["FAKE_WAVESPEED_BILLINGS_EXIT", "FAKE_WAVESPEED_NO_MATCH"])
def test_unknown_charge_uses_run_price(asked, fake_wavespeed, monkeypatch, setting):
    monkeypatch.setenv(setting, "1")
    monkeypatch.setenv("FAKE_WAVESPEED_RUN_PRICE", "0.025")
    result = make(asked)
    entry = result["made"][0]
    assert entry["charged_usd"] is None and entry["cost_source"] == "estimate"
    assert entry["cost_usd"] == result["spent_usd"] == 0.025


def test_unknown_run_price_falls_back_to_fresh_estimate(asked, fake_wavespeed, monkeypatch):
    monkeypatch.setenv("FAKE_WAVESPEED_RUN_PRICE", "nan")
    monkeypatch.setenv("FAKE_WAVESPEED_NO_MATCH", "1")
    result = make(asked)
    assert result["made"][0]["cost_usd"] == result["spent_usd"] == 0.027


@pytest.mark.parametrize("ceiling,picks", [(0.054, ["1", "2"]), (0.081, ["1", "2", "3"])])
def test_higher_charge_stops_later_lines(asked, fake_wavespeed, monkeypatch, ceiling, picks):
    monkeypatch.setenv("FAKE_WAVESPEED_CHARGED", "0.05")
    result = make(asked, picks, max_cost=ceiling)
    assert len(result["made"]) == len(runs(fake_wavespeed)) == 1
    assert result["spent_usd"] == 0.05
    assert result["failed"] == [
        {
            "number": int(number),
            "reason": "stopped: WaveSpeed charged $0.05 for line 1, more than its $0.027 quote",
        }
        for number in picks[1:]
    ]


def test_reprice_all_before_spend(asked, fake_wavespeed, monkeypatch):
    monkeypatch.setenv("FAKE_WAVESPEED_PRICE", "0.04")
    with pytest.raises(YapError) as error:
        make(asked, ("1", "2"), max_cost=0.054)
    assert error.value.code == "pictures_over_ceiling" and not runs(fake_wavespeed)
    count = 0

    def fail_second(prompt):
        nonlocal count
        count += 1
        if count == 2:
            raise wavespeed.PicturesUnavailable("offline")
        return 0.027

    monkeypatch.setattr(wavespeed, "estimate", fail_second)
    assert make(asked, ("1", "2"))["reason"] == "offline"
    assert not runs(fake_wavespeed)


@pytest.mark.parametrize(
    "setting", ["FAKE_WAVESPEED_FAILED", "FAKE_WAVESPEED_LOST", "FAKE_WAVESPEED_SKIP_DOWNLOAD"]
)
def test_failed_or_missing_download_keeps_recovery(asked, fake_wavespeed, monkeypatch, setting):
    monkeypatch.setenv(setting, "1")
    result = make(asked)
    failure = result["failed"][0]
    assert wavespeed.prediction_id(failure["prediction_id"])
    assert failure["recover"] == "wavespeed-pp-cli prediction-results " + failure["prediction_id"]
    assert not catalog.load_catalog()["items"] and not result["made"]
    assert len(runs(fake_wavespeed)) == 1
    assert result["spent_usd"] == 0.027
    assert not list((asked / "pictures").glob(".make-*"))


def test_submission_without_id_stops_next_pick(asked, fake_wavespeed, monkeypatch):
    original = wavespeed.generate
    monkeypatch.setenv("FAKE_WAVESPEED_RUN_EXIT", "5")

    def generate(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            monkeypatch.delenv("FAKE_WAVESPEED_RUN_EXIT", raising=False)

    monkeypatch.setattr(wavespeed, "generate", generate)
    result = make(asked, ("1", "2"), max_cost=0.054)
    assert result["failed"] == [
        {"number": 1, "reason": "synthetic submission refused"},
        {"number": 2, "reason": "stopped: line 1 may have been charged. "
         "Check wavespeed-pp-cli predictions, then pass --new to make it again."},
    ]
    assert not result["made"]
    assert result["spent_usd"] == 0.027
    assert len(runs(fake_wavespeed)) == 1
    assert not [argv for argv in calls(fake_wavespeed) if argv[0] == "billings"]


def test_make_off_sources_and_unavailable(asked, fake_wavespeed, monkeypatch):
    write(asked / "style.json", {"schema_version": 1})
    with pytest.raises(YapError) as error:
        make(asked)
    assert error.value.code == "pictures_off"
    write(asked / "style.json", {"schema_version": 1, **STYLE})
    monkeypatch.setenv("FAKE_WAVESPEED_AUTH_EXIT", "4")
    assert make(asked)["reason"] == "signed_out"
    assert not runs(fake_wavespeed)


def test_corrupt_cache_generates_again(asked, fake_wavespeed):
    first = make(asked)["made"][0]
    (asked / first["path"]).write_bytes(b"changed")
    second = make(asked)
    assert second["made"] and not second["cached"] and len(runs(fake_wavespeed)) == 2
    assert len(guard.approved_pictures(asked)) == 1


@pytest.mark.parametrize(
    "problem",
    ["outside", "empty", "symlink", "bad_bytes", "video", "bad_id", "failed", "no_downloads"],
)
def test_reject_invalid_downloads(asked, monkeypatch, problem):
    def generate(prompt, staging, *, dry_run):
        source = staging / "out.png"
        if problem == "outside":
            source = asked / "outside.png"
        if problem == "empty":
            source.touch()
        elif problem == "bad_bytes":
            source.write_bytes(b"not a picture")
        elif problem == "video":
            synth_broll.clip(source.with_suffix(".mov"))
            source.with_suffix(".mov").rename(source)
        else:
            synth_broll.png(source)
        if problem == "symlink":
            target = staging / "real.png"
            source.rename(target)
            source.symlink_to(target)
        return {
            "prediction": {
                "data": {
                    "id": "../bad" if problem == "bad_id" else "known123",
                    "status": "failed" if problem == "failed" else "completed",
                    "error": "synthetic refusal",
                }
            },
            "downloads": [] if problem == "no_downloads" else [{"path": str(source)}],
        }

    monkeypatch.setattr(wavespeed, "generate", generate)
    result = make(asked)
    assert result["failed"] and not result["made"]
    assert not catalog.load_catalog()["items"]
    assert not list((asked / "pictures").glob(".make-*"))
    if problem != "bad_id":
        assert result["failed"][0]["prediction_id"] == "known123"
    else:
        assert result["spent_usd"] == 0


def test_catalog_validation_and_creator_item(tmp_path):
    source = synth_broll.png(tmp_path / "own.png")
    catalog.add_paths([source], now=lambda: NOW)
    original = copy.deepcopy(catalog.load_catalog()["items"][0])
    generated = {
        "provider": "wavespeed",
        "model": wavespeed.MODEL_ID,
        "prompt": "A park",
        "seed": None,
        "size": wavespeed.SIZE,
        "prediction_id": "abc123",
        "cost_usd": 0.027,
        "cost_source": "estimate",
        "created_at": original["added_at"],
    }
    item = catalog.add_generated_item(
        source, generated=generated, description_text="A park", now=NOW
    )
    assert "generated" not in item and item == original
    item["generated"] = generated
    document = {"schema": catalog.CATALOG_SCHEMA, "next_id": 2, "items": [item], "index": {}}
    assert catalog.validate_catalog(document, tmp_path / "catalog.json") == document
    for key, value in [
        ("provider", "bad/name"),
        ("model", ""),
        ("seed", True),
        ("cost_usd", float("inf")),
        ("cost_usd", -1),
        ("cost_source", "unknown"),
        ("created_at", "yesterday"),
        ("extra", 1),
    ]:
        invalid = copy.deepcopy(document)
        invalid["items"][0]["generated"][key] = value
        with pytest.raises(InputError):
            catalog.validate_catalog(invalid, tmp_path / "catalog.json")


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_ask_make_and_failed_exit(asked, fake_wavespeed, monkeypatch, capsys, json_output):
    cut = write(asked / "cut/cut-words.json", words().to_dict())
    source = write(asked / "plans/pictures.json", requests())
    extra = ["--json"] if json_output else []
    assert cli.main(["pictures", "ask", str(source), "--cut-words", str(cut), *extra]) == 0
    output = capsys.readouterr()
    if json_output:
        result = json.loads(output.out)
        assert Path(result["html"]).is_file() and Path(result["json"]).is_file()
        saved = json.loads(Path(result["json"]).read_text())
        assert "html" not in saved and "json" not in saved
    else:
        assert "about $0.027" in output.out and "Total: about $0.027" in output.out
    argv = ["pictures", "make", str(asked), "--pick", "1", "--yes", "--max-cost", "1", *extra]
    assert cli.main(argv) == 0
    output = capsys.readouterr()
    if not json_output:
        assert "1: made as m1 ($0.027 charged)" in output.out
        assert "Spent: $0.027 of $1.00" in output.out
    assert cli.main(argv) == 0
    capsys.readouterr()
    monkeypatch.setenv("FAKE_WAVESPEED_FAILED", "1")
    assert cli.main([*argv, "--sheet", "s1", "--pick", "2"]) == 1
    capsys.readouterr()
