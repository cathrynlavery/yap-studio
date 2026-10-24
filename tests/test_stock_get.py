"""Consent, independent downloads, catalog provenance, approval guards, and credits."""

from __future__ import annotations

import copy
import importlib
import json
import shutil
from pathlib import Path

import pytest
from test_media_select import _item
from test_stock_provider import calls
from test_stock_sheet import STYLE, requests, search, words

from yap_studio import cli, receipt
from yap_studio.errors import OutputPathError, ValidationError, YapError
from yap_studio.media import catalog
from yap_studio.receipt import sha256_file


@pytest.fixture
def picked(tmp_path, monkeypatch, fake_pexels):
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    root = tmp_path / "video"
    search(root)
    fetched = []
    sources = {}

    def fetch(url, target, *, max_bytes, timeout_s):
        assert max_bytes == 600 * 1024 * 1024 and timeout_s == 900
        fetched.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        import synth_broll

        data = json.loads((root / 'stock/sheets/s1/pick-sheet.json').read_bytes())
        file = next((c['file'] for p in data['phrases'] for c in p['candidates']
                     if c['file']['url'] == url), data['phrases'][0]['candidates'][0]['file'])
        size = f"{file['width']}x{file['height']}"
        if size not in sources:
            sources[size] = synth_broll.clip(tmp_path / f'stock-{size}.mp4', size=size, duration=4)
        shutil.copyfile(sources[size], target)

    return root, fetch, fetched


def get(root, codes, fetch, **kwargs):
    from yap_studio.stock.get import get_picks
    from yap_studio.stock.pexels import PexelsProvider

    return get_picks(
        root, codes, provider=kwargs.pop("provider", PexelsProvider()), fetch=fetch, **kwargs
    )


def test_consent_unknown_duplicate_no_download(picked):
    root, fetch, fetched = picked
    with pytest.raises(YapError) as error:
        get(root, ["1a"], fetch, yes=False)
    assert error.value.code == "stock_needs_yes"
    for codes, message in [(["1a", "9z"], "9z"), (["1a", "1a"], "same clip")]:
        with pytest.raises(ValidationError, match=message):
            get(root, codes, fetch, yes=True)
    assert not fetched and not (root / "stock/clips.json").exists()


def test_download_catalog_request_and_offline_cache(picked, fake_pexels):
    root, fetch, fetched = picked
    result = get(root, ["1a"], fetch, yes=True)
    assert len(fetched) == 1 and "/101/" in fetched[0]
    assert len(result["downloaded"]) == 1 and not result["failed"], result
    request = result["requests"][0]
    assert request == {
        "id": "broll-stock-dog",
        "item": "m1",
        "kind": "insert",
        "start": {"word": "t1:0"},
        "end": {"word": "t1:1"},
        "fit": "cover",
        "in_s": 1,
        "speed": 1,
        "audio": "mute",
    }
    item = catalog.load_catalog()["items"][0]
    assert item["stock"]["clip_id"] == "101"
    assert item["provenance"] == {
        "kind": "licensed",
        "source": "Pexels",
        "license": "Pexels License",
        "url": item["stock"]["page_url"],
    }
    assert item["description"]["tags"] == ["stock_clip"]
    assert item["description"]["text"] == "dog park (stock clip from Pexels)"
    assert item["description"]["by"] == "stock"
    assert Path(item["locations"][0]["path"]).name.startswith("101-")
    records = json.loads((root / "stock/clips.json").read_text())["clips"]
    assert records[0]["request"] == request and records[0]["sha256"] == item["sha256"]
    before = calls(fake_pexels)
    cached = get(root, ["1a"], fetch, yes=True)
    assert calls(fake_pexels) == before and len(fetched) == 1 and len(cached["cached"]) == 1
    assert len(json.loads((root / "stock/clips.json").read_text())["clips"]) == 1
    assert cached["cached"][0]["cached"] is True


@pytest.mark.parametrize("module_name", ["yap_studio.stock.get", "yap_studio.planner.broll"])
def test_stock_hash_uses_restored_receipt_after_import(picked, monkeypatch, module_name):
    module = importlib.import_module(module_name)
    root, fetch, _ = picked
    get(root, ["1a"], fetch, yes=True)
    item = catalog.load_catalog()["items"][0]
    target = Path(item["locations"][0]["path"])

    def forbidden(_):
        raise AssertionError("stock import retained a patched hash function")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(receipt, "sha256_file", forbidden)
            importlib.reload(module)
        if module_name == "yap_studio.stock.get":
            assert module._cached(target, "pexels", "101") == item
        else:
            assert module._matching_location(item) == (target.resolve(), "")
    finally:
        importlib.reload(module)


def test_failed_pick_does_not_stop_rest(picked):
    root, fetch, fetched = picked

    def partial(url, *a, **kw):
        if "/101/" in url:
            raise OSError("synthetic offline")
        return fetch(url, *a, **kw)

    second_phrase_sheet(root)
    result = get(root, ["1a", "2b"], partial, yes=True)
    assert result["failed"][0]["code"] == "1a"
    assert result["downloaded"][0]["code"] == "2b" and len(fetched) == 1


def test_bad_video_is_removed(picked):
    root, _, _ = picked
    targets = []

    def invalid(url, target, **kw):
        target.parent.mkdir(parents=True, exist_ok=True)
        targets.append(target)
        target.write_bytes(b"not video")

    result = get(root, ["1a"], invalid, yes=True)
    assert result["failed"] and not targets[0].exists()
    assert not catalog.load_catalog()["items"]


def test_catalog_old_and_invalid_stock(tmp_path):
    item = _item()
    document = {"schema": catalog.CATALOG_SCHEMA, "next_id": 2, "items": [item], "index": {}}
    assert catalog.validate_catalog(document, tmp_path / "catalog.json") is document
    item["stock"] = {"provider": "bad"}
    with pytest.raises(YapError):
        catalog.validate_catalog(document, tmp_path / "catalog.json")


@pytest.mark.parametrize("described", [False, True])
def test_same_bytes_preserves_creator_item_and_records_stock_approval(
    picked,
    synth_media,
    described,
    capsys,
):
    from yap_studio.stock.credits import credits

    root, fetch, _ = picked
    source = root / "own.mp4"
    data = json.loads((root / 'stock/sheets/s1/pick-sheet.json').read_bytes())
    fetch(data['phrases'][0]['candidates'][0]['file']['url'], source,
          max_bytes=600 * 1024 * 1024, timeout_s=900)
    catalog.add_paths([source])
    provenance = {"kind": "own", "source": "my camera", "license": None, "url": None}
    description = (
        {
            "text": "Our dog on a family walk",
            "tags": ["family", "dog"],
            "by": "creator",
            "at": "2026-10-18T12:00:00Z",
            "revision": 3,
        }
        if described
        else None
    )
    catalog.update_catalog(
        lambda data: data["items"][0].update(provenance=provenance, description=description)
    )
    original = copy.deepcopy(catalog.load_catalog()["items"][0])
    result = get(root, ["1a"], fetch, yes=True)
    assert not result["failed"], result
    item = catalog.load_catalog()["items"][0]
    assert len(item["locations"]) == 2
    assert {key: value for key, value in item.items() if key != "locations"} == {
        key: value for key, value in original.items() if key != "locations"
    }
    assert "stock" not in item and item["provenance"] == provenance
    entry = json.loads((root / "stock/clips.json").read_text())["clips"][0]
    assert entry["item"] == "m1" and entry["sha256"] == item["sha256"]
    assert entry["provider"] == "pexels" and entry["clip_id"] == "101"
    assert credits(root, all_clips=True)["clips"][0]["item"] == "m1"
    if described:
        other = root / "other-video"
        cut = other / "cut/cut-words.json"
        cut.parent.mkdir(parents=True)
        cut.write_text(json.dumps(words().to_dict()))
        style = other / "style.json"
        style.write_text('{"schema_version": 1}')
        base = ["--cut-words", str(cut), "--style", str(style), "--json"]
        assert cli.main(["broll", "suggest", *base]) == 0
        assert json.loads(capsys.readouterr().out)["phrases"][0]["candidates"][0]["item"] == "m1"
        request_file = other / "requests.json"
        request_file.write_text(
            json.dumps({"schema": "yap-studio/broll-requests@1", "broll": result["requests"]})
        )
        assert cli.main(
            [
                "broll",
                "plan",
                str(request_file),
                *base,
                "--frames",
                "480",
                "--out",
                str(other / "plan"),
            ]
        ) in {0, 1}
        assert json.loads(capsys.readouterr().out)["broll"][0]["item"] == "m1"
        assert not (other / "stock/clips.json").exists()


def test_guards_and_credits(picked, capsys):
    from yap_studio.stock.credits import credits
    from yap_studio.stock.guard import filter_approved, validate_requests

    root, fetch, _ = picked
    assert credits(root)["text"] == "No stock clips in this video."
    result = get(root, ["1a"], fetch, yes=True)
    item = catalog.load_catalog()["items"][0]
    pending = root.parent / "other"
    filtered, excluded = filter_approved([item], STYLE, pending)
    assert not filtered and excluded[0]["reasons"] == ["stock clip not approved for this video"]
    data = {"schema": "yap-studio/broll-requests@1", "broll": result["requests"]}
    with pytest.raises(ValidationError, match=r"broll\[0\].item.*not approved"):
        validate_requests(data, [item], STYLE, pending)
    with pytest.raises(ValidationError, match="stock clips are off"):
        validate_requests(data, [item], {}, root)
    assert filter_approved([item], STYLE, root) == ([item], [])
    validate_requests(data, [item], STYLE, root)
    assert credits(root)["clips"] == []
    assert credits(root, all_clips=True)["clips"][0]["placed"] is False
    (root / "timeline").mkdir()
    (root / "timeline/broll.json").write_text(
        json.dumps(
            {"elements": [{"media": {"sha256": sha256_file(Path(item["locations"][0]["path"]))}}]}
        )
    )
    report = credits(root)
    assert report["clips"][0]["placed"] is True
    assert "Video by Alex <b>& on Pexels:" in report["text"]
    assert report["text"].endswith("Videos provided by Pexels: https://www.pexels.com")
    assert cli.main(["stock", "credits", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report


def test_cli_search_get_errors_and_unavailable(tmp_path, capsys, monkeypatch):
    from yap_studio.stock import command

    source = tmp_path / "requests.json"
    source.write_text(json.dumps(requests()))
    cut = tmp_path / "cut/cut-words.json"
    cut.parent.mkdir()
    cut.write_text(json.dumps(words().to_dict()))
    style = tmp_path / "style.json"
    style.write_text(json.dumps({"schema_version": 1, **STYLE}))
    argv = ["stock", "search", str(source), "--cut-words", str(cut), "--style", str(style)]
    assert cli.main([*argv, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"
    assert cli.main(["stock", "get", str(tmp_path), "--pick", "1a", "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "stock_needs_yes"

    def refused(*a, **kw):
        raise OutputPathError("synthetic path")

    monkeypatch.setattr(command, "search_sheet", refused)
    assert cli.main(argv) == 3
    assert "error: synthetic path" in capsys.readouterr().err


def test_cli_broll_approval_gate(picked, capsys):
    root, fetch, _ = picked
    result = get(root, ["1a"], fetch, yes=True)
    cut = root / "cut/cut-words.json"
    cut.parent.mkdir()
    cut.write_text(json.dumps(words().to_dict()))
    style = root / "style.json"
    style.write_text(json.dumps({"schema_version": 1, **STYLE}))
    source = root / "requests.json"
    source.write_text(
        json.dumps({"schema": "yap-studio/broll-requests@1", "broll": result["requests"]})
    )
    approvals = root / "stock/clips.json"
    approved_text = approvals.read_text()
    approvals.unlink()
    base = ["--cut-words", str(cut), "--style", str(style), "--json"]
    assert cli.main(["broll", "suggest", *base]) == 0
    suggested = json.loads(capsys.readouterr().out)
    assert not suggested["phrases"]
    assert suggested["excluded"][0]["reasons"] == ["stock clip not approved for this video"]
    plan = ["broll", "plan", str(source), *base, "--frames", "480", "--out", str(root / "plan")]
    assert cli.main(plan) == 2
    assert "broll[0].item" in capsys.readouterr().err
    approvals.write_text(approved_text)
    assert cli.main(["broll", "suggest", *base]) == 0
    assert json.loads(capsys.readouterr().out)["phrases"][0]["candidates"][0]["item"] == "m1"
    assert cli.main(plan) in {0, 1}
    assert json.loads(capsys.readouterr().out)["broll"][0]["item"] == "m1"
    style.write_text('{"schema_version": 1}')
    assert cli.main(plan) == 2
    assert "stock clips are off in this style" in capsys.readouterr().err
    assert cli.main(["broll", "suggest", *base]) == 0
    assert json.loads(capsys.readouterr().out)["excluded"]


def test_cache_corruption_refresh_and_changed_file_id(picked, fake_pexels):
    from dataclasses import replace

    from yap_studio.stock.pexels import PexelsProvider

    root, fetch, fetched = picked
    result = get(root, ["1a"], fetch, yes=True)
    target = Path(result["downloaded"][0]["path"])
    target.write_bytes(b"corrupt cache")

    class Changed(PexelsProvider):
        def details(self, clip_id):
            clip = super().details(clip_id)
            return replace(
                clip,
                files=(
                    replace(clip.files[0], file_id="999", url="https://videos.pexels.com/new.mp4"),
                ),
            )

    result = get(root, ["1a"], fetch, yes=True, provider=Changed())
    assert result["downloaded"] and len(fetched) == 2
    assert fetched[-1] == "https://videos.pexels.com/new.mp4"
    assert catalog.load_catalog()["items"][0]["stock"]["file_url"] == fetched[-1]


def test_cache_hit_without_cli_or_fetch(picked, monkeypatch):
    from yap_studio import tools

    root, fetch, fetched = picked
    get(root, ["1a"], fetch, yes=True)

    def forbidden(*a, **kw):
        pytest.fail("cache must work without CLI or fetch")

    monkeypatch.setattr(tools, "run_tool", forbidden)
    result = get(root, ["1a"], forbidden, yes=True)
    assert result["cached"] and len(fetched) == 1


def test_media_show_stock_line(picked, capsys):
    root, fetch, _ = picked
    get(root, ["1a"], fetch, yes=True)
    assert cli.main(["media", "show", "m1"]) == 0
    assert (
        "Stock: Pexels clip 101 by Alex <b>&, Pexels License, https://www.pexels.com/video/101/"
    ) in capsys.readouterr().out


def test_short_probe_refuses_pick(picked, monkeypatch):
    from dataclasses import replace

    from yap_studio import ffmpeg

    root, fetch, _ = picked
    original = ffmpeg.probe
    monkeypatch.setattr(
        ffmpeg,
        "probe",
        lambda *a, **kw: replace(original(*a, **kw), duration_s=0.5, video_duration_s=0.5),
    )
    result = get(root, ["1a"], fetch, yes=True)
    assert result["failed"] and "lasting the phrase" in result["failed"][0]["reason"]
    assert not (catalog.catalog_path().parent / "stock/pexels/101.mp4").exists()


def test_catalog_valid_stock_and_bad_fields(picked):
    root, fetch, _ = picked
    get(root, ["1a"], fetch, yes=True)
    item = catalog.load_catalog()["items"][0]
    document = {"schema": catalog.CATALOG_SCHEMA, "next_id": 2, "items": [item], "index": {}}
    assert catalog.validate_catalog(document, Path("catalog.json")) is document
    for key, value in [
        ("page_url", "http://www.pexels.com/a"),
        ("author", 10),
        ("provider", "bad/provider"),
        ("downloaded_at", "yesterday"),
    ]:
        old = item["stock"][key]
        item["stock"][key] = value
        with pytest.raises(YapError):
            catalog.validate_catalog(document, Path("catalog.json"))
        item["stock"][key] = old


@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_cli_get_status_and_stdout(picked, monkeypatch, capsys, json_output, fails):
    from yap_studio.stock import command

    root, fetch, _ = picked

    def download(url, target, **kwargs):
        if fails and "/101/" in url:
            raise OSError("synthetic download failed")
        fetch(url, target, **kwargs)

    second_phrase_sheet(root)
    monkeypatch.setattr(command, "fetch", download)
    argv = ["stock", "get", str(root), "--pick", "1a", "--pick", "2b", "--yes"]
    if json_output:
        argv.append("--json")
    status = cli.main(argv)
    output = capsys.readouterr()
    assert not output.err
    if json_output:
        result = json.loads(output.out)
        assert bool(result["failed"]) is fails
        assert len(result["downloaded"]) == (1 if fails else 2)
    else:
        assert "2b: downloaded as" in output.out
        if fails:
            assert "1a: synthetic download failed" in output.out
    assert status == (1 if fails else 0)


def second_phrase_sheet(root):
    data = requests()
    data["phrases"].append({**data["phrases"][0], "id": "stock-second"})
    return search(root, requests=data)


def test_get_refuses_two_clips_for_one_phrase_before_cli_or_fetch(picked, monkeypatch):
    from yap_studio import tools

    root, _, _ = picked

    def forbidden(*args, **kwargs):
        pytest.fail("conflicting choices must not call the CLI or download")

    monkeypatch.setattr(tools, "run_tool", forbidden)
    with pytest.raises(
        ValidationError, match=("pick one clip per phrase: 1a and 1b are both for stock-dog")
    ):
        get(root, ["1a", "1b"], forbidden, yes=True)
    assert not (root / "stock/clips.json").exists()
