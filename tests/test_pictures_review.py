"""Regression cases for paid attempts, recuts and safe failure messages."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace

import pytest
import synth_broll
from test_pictures_make import asked as asked
from test_pictures_make import make, runs
from test_pictures_sheet import ask, requests
from test_stock_provider import calls
from test_stock_sheet import words
from test_timeline import write

from yap_studio import cli, paths, tools
from yap_studio.errors import ProcessingError, ValidationError, YapError
from yap_studio.media.catalog import load_catalog, update_catalog
from yap_studio.pictures import wavespeed
from yap_studio.pictures.attempts import load_attempts, save_attempt
from yap_studio.planner.broll import REQUESTS_SCHEMA, plan_broll
from yap_studio.project.model import CutWord
from yap_studio.project.status import ProjectContext


def attempts(root):
    return json.loads((root / "pictures/attempts.json").read_text())["attempts"]


@pytest.mark.parametrize("placement", [None, "split"])
@pytest.mark.parametrize("delivery", ["made", "cached", "recovered"])
@pytest.mark.parametrize("image_format", ["png", "jpeg"])
def test_returned_requests_follow_photo_placement(
    asked, fake_wavespeed, monkeypatch, placement, delivery, image_format,
):
    image = synth_broll.png if image_format == "png" else synth_broll.jpeg
    photo = image(asked.parent / f"synthetic.{image_format}", size="108x192")
    monkeypatch.setenv("FAKE_WAVESPEED_IMAGE", str(photo))
    style_file = asked / "style.json"
    data = json.loads(style_file.read_text())
    if placement is not None:
        data["broll"]["photo_placement"] = placement
        write(style_file, data)
    if delivery == "recovered":
        monkeypatch.setenv("FAKE_WAVESPEED_SKIP_DOWNLOAD", "1")
        assert make(asked)["failed"]
        monkeypatch.delenv("FAKE_WAVESPEED_SKIP_DOWNLOAD")
    elif delivery == "cached":
        assert make(asked)["made"]
    result = make(asked)
    assert result[delivery]
    request = result["requests"][0]
    items = load_catalog()["items"]
    plan = plan_broll(
        {"schema": REQUESTS_SCHEMA, "broll": result["requests"]},
        words(), items, ProjectContext(asked).render_style().data,
        project_dir=asked, frames=480,
    )
    element = plan.lane()["elements"][0]
    assert element["kind"] == "insert"
    assert element["media"]["fit"] == ("split" if placement == "split" else "cover")
    assert items[0]["kind"] == "image" and items[0]["media_type"] == "photo"
    assert "color_match" not in element
    record = json.loads((asked / "pictures/pictures.json").read_text())["pictures"][0]
    assert record["request"] == result[delivery][0]["request"] == request
    assert len(runs(fake_wavespeed)) == 1


def test_timeout_submission_id_recovers_without_another_run(asked, fake_wavespeed, monkeypatch):
    monkeypatch.setenv("FAKE_WAVESPEED_TIMEOUT", "1")
    monkeypatch.setattr(wavespeed, "GENERATION_TIMEOUT_S", 0.5)
    first = make(asked)
    identifier = json.loads(fake_wavespeed.with_suffix(".ids.json").read_text())[0]
    assert attempts(asked)[0]["prediction_id"] == identifier
    assert first["failed"][0]["prediction_id"] == identifier
    assert len(runs(fake_wavespeed)) == 1 and first["spent_usd"] == 0.027
    monkeypatch.delenv("FAKE_WAVESPEED_TIMEOUT")
    recovered = make(asked)
    assert recovered["recovered"][0]["prediction_id"] == identifier
    assert not recovered["made"] and recovered["spent_usd"] == 0
    assert len(runs(fake_wavespeed)) == 1 and attempts(asked) == []


@pytest.mark.parametrize("answer", ["no", None])
def test_batch_stops_when_imagery_is_revoked(asked, fake_wavespeed, monkeypatch, answer):
    module = importlib.import_module("yap_studio.pictures.make")
    original = module._obtain
    after_revocation = []

    def obtain(*args, **kwargs):
        result = original(*args, **kwargs)
        if result["entry"] and result["entry"]["number"] == 1:
            if answer == "no":
                assert cli.main(["pictures", "answer", str(asked), "no", "--json"]) == 0
            else:
                (asked / "pictures/imagery.json").unlink()
            after_revocation.extend(calls(fake_wavespeed))
        return result

    monkeypatch.setattr(module, "_obtain", obtain)
    result = make(asked, ("1", "2", "3"))
    assert len(runs(fake_wavespeed)) == 1
    assert [entry["number"] for entry in result["made"]] == [1]
    assert result["spent_usd"] == 0.027
    assert result["failed"] == [
        {"number": number, "reason":
         "stopped: the creator no longer wants generated pictures for this video"}
        for number in (2, 3)
    ]
    assert calls(fake_wavespeed) == after_revocation


def test_uncertain_submission_counts_quote_and_stops_batch(asked, fake_wavespeed, monkeypatch):
    monkeypatch.setenv("FAKE_WAVESPEED_LOST_ID", "1")
    result = make(asked, ("1", "2"))
    assert len(runs(fake_wavespeed)) == 1
    assert result["spent_usd"] == 0.027 and not result["made"]
    assert result["failed"][1] == {
        "number": 2,
        "reason": "stopped: line 1 may have been charged. Check wavespeed-pp-cli predictions, "
                  "then pass --new to make it again.",
    }
    assert len(json.loads(fake_wavespeed.with_suffix(".ids.json").read_text())) == 1
    assert load_attempts(asked) == [{
        "sheet": "s1", "number": 1, "prediction_id": None,
        "quote_usd": 0.027, "started_at": "2026-10-19T12:00:00Z",
    }]
    before = calls(fake_wavespeed)
    retry = make(asked)
    assert retry["failed"] and not retry["made"] and retry["spent_usd"] == 0
    assert "pass --new" in retry["failed"][0]["reason"]
    assert calls(fake_wavespeed) == before
    monkeypatch.delenv("FAKE_WAVESPEED_LOST_ID")
    assert make(asked, new=True)["made"]
    assert len(runs(fake_wavespeed)) == 2 and load_attempts(asked) == []


@pytest.mark.parametrize("setting", ["FAKE_WAVESPEED_SKIP_DOWNLOAD", "FAKE_WAVESPEED_LOST"])
def test_retry_recovers_paid_attempt_without_second_run(
    asked, fake_wavespeed, monkeypatch, setting
):
    monkeypatch.setenv(setting, "1")
    first = make(asked)
    identifier = first["failed"][0]["prediction_id"]
    assert attempts(asked)[0]["prediction_id"] == identifier
    monkeypatch.delenv(setting)
    recovered = make(asked)
    assert recovered["recovered"][0]["prediction_id"] == identifier
    assert recovered["spent_usd"] == 0 and not recovered["made"]
    assert len(runs(fake_wavespeed)) == 1 and attempts(asked) == []
    argv = calls(fake_wavespeed)
    assert any(a[:2] == ["prediction-results", identifier] for a in argv)
    assert any(a[0] == "download" and any(v.startswith("--output=") for v in a) for a in argv)
    for args in argv:
        if args[0] in {"prediction-results", "download"}:
            assert all(flag in args for flag in wavespeed.FLAGS)


def test_catalog_failure_recovers_without_second_run(asked, fake_wavespeed, monkeypatch):
    module = importlib.import_module("yap_studio.pictures.make")
    original = module.add_generated_item

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(module, "add_generated_item", original)
        raise OSError("synthetic catalog write failure")

    monkeypatch.setattr(module, "add_generated_item", fail_once)
    assert make(asked)["failed"]
    assert make(asked)["recovered"]
    assert len(runs(fake_wavespeed)) == 1 and attempts(asked) == []


def test_record_write_failure_recovers_without_second_run(asked, fake_wavespeed, monkeypatch):
    original = paths.atomic_write_text

    def fail_record(target, *args, **kwargs):
        if target == asked / "pictures/pictures.json":
            monkeypatch.setattr(paths, "atomic_write_text", original)
            raise OSError("synthetic record write failure")
        return original(target, *args, **kwargs)

    monkeypatch.setattr(paths, "atomic_write_text", fail_record)
    first = make(asked)
    assert first["failed"] and attempts(asked)[0]["prediction_id"]
    second = make(asked)
    assert second["recovered"] and second["spent_usd"] == 0
    assert len(runs(fake_wavespeed)) == 1


@pytest.mark.parametrize("problem", ["processing", "download"])
def test_unfinished_recovery_keeps_attempt(asked, fake_wavespeed, monkeypatch, problem):
    monkeypatch.setenv("FAKE_WAVESPEED_SKIP_DOWNLOAD", "1")
    identifier = make(asked)["failed"][0]["prediction_id"]
    monkeypatch.delenv("FAKE_WAVESPEED_SKIP_DOWNLOAD")
    setting = (
        "FAKE_WAVESPEED_RECOVERY_STATUS"
        if problem == "processing"
        else "FAKE_WAVESPEED_DOWNLOAD_EXIT"
    )
    monkeypatch.setenv(setting, "processing" if problem == "processing" else "5")
    result = make(asked)
    assert result["failed"][0]["prediction_id"] == identifier
    assert result["spent_usd"] == 0 and attempts(asked)[0]["prediction_id"] == identifier
    assert len(runs(fake_wavespeed)) == 1
    monkeypatch.delenv(setting)
    assert make(asked)["recovered"]


@pytest.mark.parametrize("exit_code", [0, 5])
def test_recovery_failed_prediction_drops_attempt(asked, fake_wavespeed, monkeypatch, exit_code):
    monkeypatch.setenv("FAKE_WAVESPEED_SKIP_DOWNLOAD", "1")
    make(asked)
    monkeypatch.delenv("FAKE_WAVESPEED_SKIP_DOWNLOAD")
    monkeypatch.setenv("FAKE_WAVESPEED_RECOVERY_STATUS", "failed")
    monkeypatch.setenv("FAKE_WAVESPEED_RESULTS_EXIT", str(exit_code))
    result = make(asked)
    assert result["failed"][0]["reason"] == (
        "WaveSpeed reported the earlier attempt failed; run pictures make again to try once more."
    )
    assert attempts(asked) == [] and len(runs(fake_wavespeed)) == 1
    assert make(asked)["made"] and len(runs(fake_wavespeed)) == 2


def test_recovery_falls_back_to_original_quote(asked, fake_wavespeed, monkeypatch):
    monkeypatch.setenv("FAKE_WAVESPEED_SKIP_DOWNLOAD", "1")
    make(asked)
    monkeypatch.delenv("FAKE_WAVESPEED_SKIP_DOWNLOAD")
    monkeypatch.setenv("FAKE_WAVESPEED_NO_MATCH", "1")
    monkeypatch.setenv("FAKE_WAVESPEED_PRICE", "0.5")
    monkeypatch.setenv("FAKE_WAVESPEED_RUN_PRICE", "0.4")
    result = make(asked, max_cost=0.001)
    assert result["recovered"][0]["cost_usd"] == 0.027
    assert result["recovered"][0]["cost_source"] == "estimate" and result["spent_usd"] == 0


def test_attempt_exists_before_submission(asked, fake_wavespeed, monkeypatch):
    original = wavespeed.generate

    def generate(*args, **kwargs):
        assert attempts(asked) == [
            {
                "sheet": "s1",
                "number": 1,
                "prediction_id": None,
                "quote_usd": 0.027,
                "started_at": "2026-10-19T12:00:00Z",
            }
        ]
        return original(*args, **kwargs)

    monkeypatch.setattr(wavespeed, "generate", generate)
    assert make(asked)["made"]


@pytest.mark.parametrize(
    "payload",
    [{}, {"schema": "yap-studio/picture-attempts@1", "attempts": [{"prediction_id": "bad/id"}]}],
)
def test_attempts_are_read_strictly(asked, fake_wavespeed, payload):
    write(asked / "pictures/attempts.json", payload)
    with pytest.raises(ValidationError) as error:
        make(asked)
    assert error.value.parameter == "pictures.attempts" and not runs(fake_wavespeed)


def test_failed_prediction_drops_attempt(asked, fake_wavespeed, monkeypatch):
    monkeypatch.setenv("FAKE_WAVESPEED_FAILED", "1")
    assert make(asked)["failed"]
    assert attempts(asked) == []
    monkeypatch.delenv("FAKE_WAVESPEED_FAILED")
    assert make(asked)["made"] and len(runs(fake_wavespeed)) == 2


def test_unknown_attempt_needs_new_before_paying(asked, fake_wavespeed):
    write(
        asked / "pictures/attempts.json",
        {
            "schema": "yap-studio/picture-attempts@1",
            "attempts": [
                {
                    "sheet": "s1",
                    "number": 1,
                    "prediction_id": None,
                    "quote_usd": 0.027,
                    "started_at": "2026-10-19T12:00:00Z",
                }
            ],
        },
    )
    result = make(asked)
    assert result["spent_usd"] == 0 and not result["made"]
    assert result["failed"] == [
        {
            "number": 1,
            "reason": (
                "A previous attempt for line 1 may have been charged. "
                "Check wavespeed-pp-cli predictions, then pass --new to make it again."
            ),
        }
    ]
    assert not runs(fake_wavespeed)
    assert make(asked, new=True)["made"] and len(runs(fake_wavespeed)) == 1


def test_cli_new_discards_uncertain_attempt(asked, fake_wavespeed, capsys):
    write(
        asked / "pictures/attempts.json",
        {
            "schema": "yap-studio/picture-attempts@1",
            "attempts": [
                {
                    "sheet": "s1",
                    "number": 1,
                    "prediction_id": None,
                    "quote_usd": 0.027,
                    "started_at": "2026-10-19T12:00:00Z",
                }
            ],
        },
    )
    assert (
        cli.main(
            [
                "pictures",
                "make",
                str(asked),
                "--pick",
                "1",
                "--yes",
                "--new",
                "--max-cost",
                "0.027",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["made"]
    assert len(runs(fake_wavespeed)) == 1 and attempts(asked) == []


def test_above_quote_charge_stops_next_picture(asked, fake_wavespeed, monkeypatch):
    monkeypatch.setenv("FAKE_WAVESPEED_CHARGED", "0.04")
    result = make(asked, ("1", "2"))
    assert len(result["made"]) == len(runs(fake_wavespeed)) == 1
    assert result["spent_usd"] == 0.04
    assert result["failed"] == [
        {
            "number": 2,
            "reason": ("stopped: WaveSpeed charged $0.04 for line 1, more than its $0.027 quote"),
        }
    ]


@pytest.mark.parametrize("state", ["signed_out", "missing_cli"])
def test_all_cached_works_without_auth(asked, fake_wavespeed, monkeypatch, state):
    entry = make(asked)["made"][0]
    if state == "signed_out":
        monkeypatch.setenv("FAKE_WAVESPEED_AUTH_EXIT", "4")
    else:
        monkeypatch.setattr(tools, "find_tool", lambda _: None)
    before = calls(fake_wavespeed)
    result = make(asked)
    assert result["cached"] == [entry] and result["spent_usd"] == 0
    assert calls(fake_wavespeed) == before


def test_cached_picture_after_project_rename_registers_location(asked, fake_wavespeed):
    entry = make(asked)["made"][0]
    original = load_catalog()["items"][0]
    record = (asked / "pictures/pictures.json").read_bytes()
    before = calls(fake_wavespeed)
    renamed = asked.with_name("renamed-video")
    asked.rename(renamed)
    result = make(renamed)
    assert result["cached"] == [entry] and result["spent_usd"] == 0
    assert not result["made"] and not result["recovered"] and not result["failed"]
    assert calls(fake_wavespeed) == before and len(runs(fake_wavespeed)) == 1
    items = load_catalog()["items"]
    plan = plan_broll(
        {"schema": REQUESTS_SCHEMA, "broll": result["requests"]},
        words(), items, ProjectContext(renamed).render_style().data,
        project_dir=renamed, frames=480,
    )
    assert plan.lane()["elements"][0]["media"]["path"] == entry["path"]
    assert len(items) == 1 and items[0]["id"] == original["id"] == entry["item"]
    assert items[0]["generated"] == original["generated"]
    assert str((renamed / entry["path"]).resolve()) in {
        location["path"] for location in items[0]["locations"]
    }
    assert (renamed / "pictures/pictures.json").read_bytes() == record


@pytest.mark.parametrize("change", ["missing", "invalid"])
@pytest.mark.parametrize("preflight", ["ceiling", "attempt"])
def test_bad_cached_metadata_obeys_preflight(asked, fake_wavespeed, change, preflight):
    make(asked)
    source = asked / "pictures/pictures.json"
    data = json.loads(source.read_text())
    if change == "missing":
        data["pictures"][0].pop("model")
    else:
        data["pictures"][0]["cost_usd"] = -1
    write(source, data)
    record = source.read_bytes()
    if preflight == "attempt":
        save_attempt(asked, "s1", 1, {
            "sheet": "s1", "number": 1, "prediction_id": None,
            "quote_usd": 0.027, "started_at": "2026-10-19T12:00:00Z",
        })
        before = calls(fake_wavespeed)
        result = make(asked)
        assert not result["cached"] and not result["made"] and result["spent_usd"] == 0
        assert "pass --new" in result["failed"][0]["reason"]
        assert calls(fake_wavespeed) == before
    else:
        with pytest.raises(YapError) as error:
            make(asked, max_cost=0.001)
        assert error.value.code == "pictures_over_ceiling"
    assert len(runs(fake_wavespeed)) == 1
    assert source.read_bytes() == record


def test_cached_request_uses_rebuilt_catalog_id(asked, fake_wavespeed):
    entry = make(asked)["made"][0]
    original = load_catalog()["items"][0]
    source = asked / "pictures/pictures.json"
    record = source.read_bytes()
    update_catalog(lambda document: document.update(items=[], index={}, next_id=2))
    before = calls(fake_wavespeed)
    result = make(asked)
    items = load_catalog()["items"]
    assert len(items) == 1 and items[0]["id"] != entry["item"]
    assert result["cached"][0]["item"] == items[0]["id"]
    assert result["cached"][0]["request"]["item"] == items[0]["id"]
    assert result["requests"][0]["item"] == items[0]["id"]
    assert items[0]["generated"] == original["generated"]
    assert result["spent_usd"] == 0 and not result["made"]
    assert calls(fake_wavespeed) == before and len(runs(fake_wavespeed)) == 1
    plan = plan_broll(
        {"schema": REQUESTS_SCHEMA, "broll": result["requests"]},
        words(), items, ProjectContext(asked).render_style().data,
        project_dir=asked, frames=480,
    )
    assert plan.broll[0]["item"] == items[0]["id"]
    assert source.read_bytes() == record


def test_recut_dropping_internal_word_refuses_before_spend(asked, fake_wavespeed):
    cut_words = replace(
        words(),
        words=(
            CutWord("t1:0", "story", "Dog", 12.4, 13.0),
            CutWord("t1:2", "story", "quiet", 13.0, 13.4),
            words().words[1],
        ),
    )
    ask(asked, cut_words=cut_words)
    assert words().words[0].id == "t1:0"
    before = calls(fake_wavespeed)
    with pytest.raises(
        ValidationError, match="line 1 changed since sheet s2; run pictures ask again"
    ):
        make(asked)
    assert calls(fake_wavespeed) == before


def test_sheet_records_spanned_word_ids(asked):
    data = json.loads((asked / "pictures/sheets/s1/ask-sheet.json").read_text())
    assert data["lines"][0]["word_ids"] == ["t1:0", "t1:1"]
    page = (asked / "pictures/sheets/s1/ask-sheet.html").read_text()
    assert "ceiling is checked against WaveSpeed's quotes before each picture" in page
    assert "run stops if a charge comes in above its quote" in page


@pytest.fixture
def external_sheet(asked, capsys):
    source = write(asked.parent / "external-cut-words.json", words().to_dict())
    (asked / "cut/cut-words.json").unlink()
    request_file = write(asked / "plans/pictures.json", requests())
    assert cli.main([
        "pictures", "ask", str(request_file), "--cut-words", str(source),
        "--project", str(asked), "--json",
    ]) == 0
    data = json.loads(capsys.readouterr().out)
    return source, data["sheet"]


def test_make_uses_external_cut_words(asked, external_sheet, fake_wavespeed, capsys):
    source, sheet = external_sheet
    assert not (asked / "cut/cut-words.json").exists()
    assert cli.main([
        "pictures", "make", str(asked), "--pick", "1", "--yes", "--max-cost", "1", "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["made"]) == len(runs(fake_wavespeed)) == 1
    saved = json.loads((asked / f"pictures/sheets/{sheet}/ask-sheet.json").read_text())
    assert saved["schema"] == "yap-studio/picture-ask-sheet@1"
    assert saved["cut_words"] == str(source.resolve())


@pytest.mark.parametrize("cut_words", ["cut/cut-words.json", "plans/custom-cut-words.json"])
def test_make_after_project_rename(asked, fake_wavespeed, capsys, cut_words):
    source = write(asked / cut_words, words().to_dict())
    request_file = write(asked / "plans/pictures.json", requests())
    assert cli.main([
        "pictures", "ask", str(request_file), "--cut-words", str(source),
        "--project", str(asked), "--json",
    ]) == 0
    sheet = json.loads(capsys.readouterr().out)["sheet"]
    renamed = asked.with_name("renamed-video")
    asked.rename(renamed)
    assert cli.main([
        "pictures", "make", str(renamed), "--pick", "1", "--yes", "--max-cost", "1", "--json",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["made"]) == len(runs(fake_wavespeed)) == 1
    saved = json.loads((renamed / f"pictures/sheets/{sheet}/ask-sheet.json").read_text())
    assert saved["cut_words"] == cut_words


@pytest.mark.parametrize("change", ["text", "missing"])
def test_external_cut_words_stale_before_cli(asked, external_sheet, fake_wavespeed, change):
    source, sheet = external_sheet
    write(asked / "cut/cut-words.json", words().to_dict())
    if change == "missing":
        source.unlink()
    else:
        data = json.loads(source.read_text())
        data["words"][0]["text"] = "Bird"
        write(source, data)
    before = calls(fake_wavespeed)
    with pytest.raises(ValidationError) as error:
        make(asked)
    assert error.value.parameter == "pick"
    assert str(error.value).endswith(f"line 1 changed since sheet {sheet}; run pictures ask again")
    assert calls(fake_wavespeed) == before


def test_older_sheet_without_cut_words_uses_project_cut(asked, fake_wavespeed):
    source = asked / "pictures/sheets/s1/ask-sheet.json"
    data = json.loads(source.read_text())
    data.pop("cut_words", None)
    write(source, data)
    assert make(asked)["made"]
    assert len(runs(fake_wavespeed)) == 1


@pytest.mark.parametrize("change", ["missing", "anchor", "window"])
def test_stale_sheet_refuses_before_any_cli_call(asked, fake_wavespeed, change):
    source = asked / "cut/cut-words.json"
    data = json.loads(source.read_text())
    if change == "missing":
        source.unlink()
    else:
        if change == "anchor":
            data["words"].pop()
        else:
            data["words"][-1]["end"] = 19
        write(source, data)
    before = calls(fake_wavespeed)
    with pytest.raises(ValidationError) as error:
        make(asked)
    assert error.value.parameter == "pick"
    assert str(error.value).endswith("line 1 changed since sheet s1; run pictures ask again")
    assert calls(fake_wavespeed) == before


def test_recut_changed_internal_id_with_same_text(asked, fake_wavespeed):
    cut_words = replace(
        words(),
        words=(
            CutWord("t1:0", "story", "Dog", 12.4, 13.0),
            CutWord("t1:2", "story", "quiet", 13.0, 13.4),
            words().words[1],
        ),
    )
    ask(asked, cut_words=cut_words)
    changed = cut_words.to_dict()
    changed["words"][1]["id"] = "t1:9"
    write(asked / "cut/cut-words.json", changed)
    before = calls(fake_wavespeed)
    with pytest.raises(ValidationError, match="line 1 changed since sheet s2"):
        make(asked)
    assert calls(fake_wavespeed) == before


def test_older_sheet_compares_text_and_cached_skips_recut(asked, fake_wavespeed):
    source = asked / "pictures/sheets/s1/ask-sheet.json"
    data = json.loads(source.read_text())
    for line in data["lines"]:
        line.pop("word_ids")
    write(source, data)
    assert make(asked)["made"]
    (asked / "cut/cut-words.json").unlink()
    before = calls(fake_wavespeed)
    assert make(asked)["cached"]
    assert calls(fake_wavespeed) == before


def test_later_stale_pick_refuses_whole_run(asked, fake_wavespeed):
    source = asked / "pictures/sheets/s1/ask-sheet.json"
    data = json.loads(source.read_text())
    data["lines"][1]["end"] = {"word": "t1:99"}
    write(source, data)
    before = calls(fake_wavespeed)
    with pytest.raises(ValidationError, match="line 2 changed since sheet s1"):
        make(asked, ("1", "2"))
    assert calls(fake_wavespeed) == before


@pytest.mark.parametrize("error_type", [YapError, OSError])
def test_generation_failure_hides_command_arguments(tmp_path, monkeypatch, error_type):
    prompt = "A synthetic prompt that must stay private"

    def fail(*args, **kwargs):
        raise error_type(f"wavespeed-pp-cli run --prompt={prompt} --download={tmp_path}/out.png")

    monkeypatch.setattr(tools, "run_tool", fail)
    with pytest.raises(wavespeed.GenerationError) as error:
        wavespeed.generate(prompt, tmp_path, dry_run=False)
    assert prompt not in str(error.value) and str(tmp_path) not in str(error.value)


def test_empty_stderr_failure_uses_exit_code_only(tmp_path, monkeypatch):
    prompt = "Synthetic private prompt"

    def fail(*args, **kwargs):
        raise ProcessingError(f"wavespeed-pp-cli run {prompt} {tmp_path}", 5, "")

    monkeypatch.setattr(tools, "run_tool", fail)
    with pytest.raises(wavespeed.GenerationError) as error:
        wavespeed.generate(prompt, tmp_path, dry_run=False)
    assert str(error.value) == "WaveSpeed CLI exited with code 5"


@pytest.mark.parametrize("operation", ["prediction-results", "download"])
@pytest.mark.parametrize("error_type", [YapError, OSError, ProcessingError])
def test_recovery_failures_hide_command_arguments(tmp_path, monkeypatch, operation, error_type):
    command = f"wavespeed-pp-cli {operation} synthetic-private {tmp_path}/out.png"

    def fail(*args, **kwargs):
        if error_type is ProcessingError:
            raise ProcessingError(command, 5, command)
        raise error_type(command)

    monkeypatch.setattr(tools, "run_tool", fail)
    with pytest.raises(wavespeed.GenerationError) as error:
        if operation == "prediction-results":
            wavespeed.prediction_results("known123")
        else:
            wavespeed.download("https://cdn.example.test/known123.png", tmp_path)
    assert "synthetic-private" not in str(error.value) and str(tmp_path) not in str(error.value)


@pytest.mark.parametrize(
    "keyword",
    [
        "Authorization",
        "BEARER",
        "api key",
        "api_key",
        "api-key",
        "apikey",
        "TOKEN",
        "secret",
        "password",
        "credential",
        "cookie",
        "xi-api-key",
        "X-Api-Key",
    ],
)
def test_redaction_drops_every_credential_line(keyword):
    assert wavespeed.redact(f"before\n{keyword}: synthetic-sensitive\nafter") == "before\nafter"
