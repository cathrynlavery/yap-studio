"""Requests, own-footage gaps, and the creator's static pick sheet."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
from test_media_select import STAMP, _item
from test_stock_provider import calls

from yap_studio.errors import ValidationError, YapError
from yap_studio.project.model import CutWord, CutWords
from yap_studio.receipt import sha256_file

STYLE = {"broll": {"stock_clips": "ask"}}


def words():
    return CutWords(
        "output",
        {"t1": "a" * 64},
        (
            CutWord("t1:0", "story", "Dog", 12.4, 13.4),
            CutWord("t1:1", "story", "park.", 13.4, 15.1),
        ),
    )


def requests():
    return {
        "schema": "yap-studio/stock-requests@1",
        "phrases": [
            {
                "id": "stock-dog",
                "start": {"word": "t1:0"},
                "end": {"word": "t1:1"},
                "query": "dog park",
            }
        ],
    }


def preview(url, target, *, max_bytes, timeout_s):
    assert max_bytes == 5 * 1024 * 1024 and timeout_s == 30
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"synthetic picture")


def search(tmp_path, **kwargs):
    from yap_studio.stock.pexels import PexelsProvider
    from yap_studio.stock.sheet import search_sheet

    return search_sheet(
        kwargs.pop("requests", requests()),
        kwargs.pop("cut_words", words()),
        kwargs.pop("catalog_items", []),
        kwargs.pop("style", STYLE),
        project_dir=tmp_path,
        provider=kwargs.pop("provider", PexelsProvider()),
        fetch=kwargs.pop("fetch", preview),
        **kwargs,
    )


def test_project_sources_override_stock_style(tmp_path, fake_pexels):
    from yap_studio.project.model import Project, Take

    target = tmp_path / "project.json"
    data = Project("story", (Take("t1", "take.mov"),)).to_dict()
    target.write_text(json.dumps({**data, "broll_sources": ["folder"]}))
    message = "Stock clips are not in this video's B-roll sources"
    with pytest.raises(YapError, match=message) as error:
        search(tmp_path)
    assert "video-type set DIR" in error.value.suggested_action
    target.write_text(json.dumps({**data, "broll_sources": ["stock"]}))
    assert search(tmp_path, style={"broll": {"stock_clips": "off"}})["status"] == "ok"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "bad"),
        ("phrases", []),
        ("phrases", [requests()["phrases"][0]] * 9),
        ("extra", 1),
        ("phrases[0].extra", 1),
        ("phrases[0].id", "dog"),
        ("phrases[0].query", "a"),
        ("phrases[0].query", "a\nb"),
        ("phrases[0].query", "x" * 81),
        ("phrases[0].not_fitting", "m1"),
        ("phrases[0].start", {"time": 12.4}),
        ("phrases[0].end", {"word": "t1:99"}),
        ("phrases[0].start", {"word": "t1:0", "bad": 1}),
    ],
)
def test_request_errors_name_field(tmp_path, fake_pexels, field, value):
    data = requests()
    if field.startswith("phrases[0]."):
        data["phrases"][0][field.split(".", 1)[1]] = value
    else:
        data[field] = value
    with pytest.raises(ValidationError) as error:
        search(tmp_path / "project", requests=data)
    assert field in str(error.value)
    assert not (tmp_path / "project").exists()


def test_duplicate_window_and_per_phrase(tmp_path, fake_pexels):
    data = requests()
    data["phrases"] *= 2
    with pytest.raises(ValidationError, match=r"phrases\[1\].id"):
        search(tmp_path, requests=data)
    short = replace(
        words(),
        words=(CutWord("t1:0", "story", "Dog", 0, 0.1), CutWord("t1:1", "story", "park", 0.1, 0.3)),
    )
    with pytest.raises(ValidationError, match="span 1 to 4 seconds"):
        search(tmp_path, cut_words=short)
    for n in (0, 4, 5, 6, True):
        with pytest.raises(ValidationError, match="per_phrase.*1 to 3"):
            search(tmp_path, per_phrase=n)


def test_off_and_unavailable_write_nothing(tmp_path):
    from yap_studio.stock.provider import StockUnavailable

    root = tmp_path / "project"
    with pytest.raises(YapError) as error:
        search(root, style={})
    assert error.value.code == "stock_off" and not root.exists()
    result = search(root)
    assert result["status"] == "unavailable" and result["reason"] == "missing_cli"
    assert result["setup"] and not root.exists()

    class Offline:
        label = "Pexels"

        def status(self):
            return "ready"

        def search(self, *a, **kw):
            raise StockUnavailable("offline")

    result = search(root, provider=Offline())
    assert result["reason"] == "offline" and not root.exists()


def test_sheet_fields_codes_two_searches_and_html(tmp_path, fake_pexels):
    result = search(tmp_path, per_phrase=3)
    assert result["sheet"] == "s1" and result["status"] == "ok"
    root = tmp_path / "stock/sheets/s1"
    assert json.loads((root / "pick-sheet.json").read_text()) == result
    phrase = result["phrases"][0]
    assert phrase["text"] == "Dog park."
    assert phrase["window_s"] == 2.7
    assert [c["code"] for c in phrase["candidates"]] == ["1a", "1b"]
    assert [c["clip_id"] for c in phrase["candidates"]] == ["101", "102"]
    assert phrase["candidates"][1]["crop_width"] == 1215
    assert phrase["candidates"][0]["in_s"] == 1
    assert result["ask"][0]["question"] == ('For "Dog park." (0:12.4 to 0:15.1), which stock clip?')
    assert result["ask"][0]["options"][-1] == {"code": None, "label": "None, keep my own footage"}
    html = (root / "pick-sheet.html").read_text()
    assert "Alex &lt;b&gt;&amp;" in html and "Alex <b>&" not in html
    assert "Videos provided by Pexels" in html and 'href="https://www.pexels.com"' in html
    assert "thumbs/1a.jpg" in html and "<script" not in html and "crop-frame" in html
    assert "Stock clip 1a by Alex &lt;b&gt;&amp;" in html
    searches = [a for a in calls(fake_pexels) if a[:2] == ["videos", "search"]]
    assert len(searches) == 2 and "portrait" in searches[0] and "large" in searches[1]
    assert "--orientation" not in searches[1]
    assert search(tmp_path, per_phrase=1)["sheet"] == "s2"
    assert len([a for a in calls(fake_pexels) if a[:2] == ["videos", "search"]]) == 3


def test_gap_check_not_fitting_and_stock_ignored(tmp_path, fake_pexels):
    source = tmp_path / "own.mov"
    source.write_bytes(b"own footage")
    item = _item(3, locations=[{"path": str(source), "added_at": STAMP, "seen_at": STAMP}])
    item["sha256"] = sha256_file(source)
    result = search(tmp_path, catalog_items=[item])
    assert result["phrases"] == [] and "m3" in result["skipped"][0]["reason"]
    assert not [a for a in calls(fake_pexels) if a[:2] == ["videos", "search"]]
    data = requests()
    data["phrases"][0]["not_fitting"] = ["m3"]
    assert search(tmp_path, requests=data, catalog_items=[item])["phrases"]
    item["stock"] = {"provider": "pexels"}
    assert search(tmp_path, catalog_items=[item])["phrases"]


def test_preview_failure_and_untrusted_links(tmp_path, fake_pexels):
    from yap_studio.stock.pexels import PexelsProvider

    class Untrusted(PexelsProvider):
        def search(self, *a, **kw):
            return [
                replace(
                    super().search(*a, **kw)[0],
                    page_url="https://evil.com/<b>",
                    author_url="http://www.pexels.com/a",
                    license_url="javascript:bad",
                )
            ]

    def failed(*a, **kw):
        raise OSError("offline")

    data = search(tmp_path, provider=Untrusted(), fetch=failed, per_phrase=1)
    assert data["phrases"][0]["candidates"][0]["thumb"] is None
    html = (tmp_path / "stock/sheets/s1/pick-sheet.html").read_text()
    assert "No preview" in html
    assert 'href="https://evil' not in html and 'href="http:' not in html
    assert 'href="javascript:' not in html


def test_later_search_failure_writes_nothing(tmp_path, fake_pexels):
    from yap_studio.stock.pexels import PexelsProvider
    from yap_studio.stock.provider import StockUnavailable

    class LaterFailure(PexelsProvider):
        count = 0

        def search(self, *a, **kw):
            self.count += 1
            if self.count == 2:
                raise StockUnavailable("rate_limited")
            return super().search(*a, **kw)

    root = tmp_path / "project"
    assert search(root, provider=LaterFailure())["reason"] == "rate_limited"
    assert not root.exists()


def test_multiple_phrases_and_no_results(tmp_path, fake_pexels):
    from yap_studio.stock.pexels import PexelsProvider

    data = copy.deepcopy(requests())
    data["phrases"].append({**data["phrases"][0], "id": "stock-second"})
    result = search(tmp_path, requests=data, per_phrase=1)
    assert result["phrases"][1]["candidates"][0]["code"] == "2a"

    class Empty(PexelsProvider):
        def search(self, *a, **kw):
            return []

    result = search(tmp_path, provider=Empty())
    assert result["phrases"][0]["note"] == "Nothing suitable found; try other search words"
    assert result["ask"] == []


@pytest.mark.parametrize("limit", [4, 5])
def test_per_phrase_refuses_more_than_three(tmp_path, fake_pexels, limit):
    with pytest.raises(ValidationError, match="per_phrase.*1 to 3"):
        search(tmp_path, per_phrase=limit)
    assert not calls(fake_pexels)[1:]
    assert not (tmp_path / "stock").exists()


def test_cli_help_explains_per_phrase_limit(capsys):
    from yap_studio import cli

    assert cli.main(["stock", "search", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "1 to 3" in help_text and "default: 3" in help_text


def test_previews_fit_fixed_height_boxes_and_label_only_landscape(tmp_path, fake_pexels):
    import re

    search(tmp_path)
    markup = (tmp_path / "stock/sheets/s1/pick-sheet.html").read_text()
    portrait, landscape = re.findall(r"<article>.*?</article>", markup, flags=re.DOTALL)
    assert (
        '<div class="media" style="aspect-ratio: 1080/1920; '
        'width: min(100%, calc(var(--preview-h) * 0.5625))">'
    ) in portrait
    assert (
        '<div class="media" style="aspect-ratio: 3840/2160; '
        'width: min(100%, calc(var(--preview-h) * 1.7778))">'
    ) in landscape
    assert "Cropped to the white frame" not in portrait
    assert "Cropped to the white frame" in landscape
    assert "crop-frame" not in portrait
    assert landscape.index('class="crop-frame"') < landscape.index("</div></div>")
    assert landscape.index("</div></div>") < landscape.index("Cropped to the white frame")


def test_preview_css_centres_images_at_desktop_and_mobile_heights(tmp_path, fake_pexels):
    import re

    search(tmp_path)
    markup = (tmp_path / "stock/sheets/s1/pick-sheet.html").read_text()
    root = re.search(r":root\{([^}]+)\}", markup).group(1)
    preview_rule = re.search(r"\.preview\{([^}]+)\}", markup).group(1)
    media_rule = re.search(r"\.media\{([^}]+)\}", markup).group(1)
    image_rule = re.search(r"\.media img\{([^}]+)\}", markup).group(1)
    assert "--preview-h:320px" in root
    for declaration in (
        "height:var(--preview-h)",
        "display:flex",
        "align-items:center",
        "justify-content:center",
        "background:#e9e9e6",
    ):
        assert declaration in preview_rule
    for declaration in ("position:relative", "overflow:hidden", "border-radius:10px"):
        assert declaration in media_rule
    for declaration in ("width:100%", "height:100%", "object-fit:cover"):
        assert declaration in image_rule
    assert "@media(max-width:560px){:root{--preview-h:240px}" in markup
    assert "border:2px solid white" in markup
    assert "rgba(0,0,0,.55)" in markup


@pytest.mark.parametrize(("end", "length"), [(44.3, "3.0"), (45.0, "3.7")])
def test_phrase_heading_quotes_and_one_decimal_length(tmp_path, fake_pexels, end, length):
    from yap_studio.stock.sheet import render_html

    result = search(tmp_path)
    phrase = result["phrases"][0]
    phrase.update(text="<b>& strawberry", start_s=41.3, end_s=end, window_s=end - 41.3)
    markup = render_html(result, "project")
    assert "<h2>1. “&lt;b&gt;&amp; strawberry”</h2>" in markup
    assert f"0:41.3 to 0:{end:.1f} ({length} s)" in markup


@pytest.mark.parametrize("ready", [True, False])
def test_cli_search_paths_are_absolute_and_only_in_stdout(
    tmp_path,
    fake_pexels,
    monkeypatch,
    capsys,
    ready,
):
    from yap_studio import cli
    from yap_studio.stock import command

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setenv("FAKE_PEXELS_AUTH_EXIT", "0" if ready else "4")
    monkeypatch.setattr(command, "fetch", preview)
    project = tmp_path / "project"
    cut = project / "cut/cut-words.json"
    cut.parent.mkdir(parents=True)
    cut.write_text(json.dumps(words().to_dict()))
    source = project / "requests.json"
    source.write_text(json.dumps(requests()))
    style = project / "style.json"
    style.write_text(json.dumps({"schema_version": 1, **STYLE}))
    assert (
        cli.main(
            [
                "stock",
                "search",
                str(source),
                "--cut-words",
                str(cut),
                "--style",
                str(style),
                "--json",
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert not output.err
    result = json.loads(output.out)
    if ready:
        from pathlib import Path

        folder = project / "stock/sheets" / result["sheet"]
        assert result["html"] == str((folder / "pick-sheet.html").resolve())
        assert result["json"] == str((folder / "pick-sheet.json").resolve())
        assert Path(result["html"]).is_file()
        saved = json.loads(Path(result["json"]).read_text())
        assert "html" not in saved and "json" not in saved
        assert {key: value for key, value in result.items() if key not in {"html", "json"}} == saved
    else:
        assert result["status"] == "unavailable"
        assert "html" not in result and "json" not in result


@pytest.mark.parametrize("state", ["deleted", "changed"])
def test_gap_check_searches_when_own_file_is_unusable(tmp_path, fake_pexels, state):
    source = tmp_path / "own.mov"
    source.write_bytes(b"own footage")
    item = _item(3, locations=[{"path": str(source), "added_at": STAMP, "seen_at": STAMP}])
    item["sha256"] = sha256_file(source)
    if state == "deleted":
        source.unlink()
    else:
        source.write_bytes(b"different footage")
    result = search(tmp_path, catalog_items=[item])
    assert result["phrases"] and not result["skipped"]
    assert any(argv[:2] == ["videos", "search"] for argv in calls(fake_pexels))


def test_gap_skip_reason_lists_only_files_that_still_match(tmp_path, fake_pexels):
    source = tmp_path / "own.mov"
    source.write_bytes(b"own footage")
    good = _item(7, locations=[{"path": str(source), "added_at": STAMP, "seen_at": STAMP}])
    good["sha256"] = sha256_file(source)
    stale = _item(
        3, locations=[{"path": str(tmp_path / "gone.mov"), "added_at": STAMP, "seen_at": STAMP}]
    )
    result = search(tmp_path, catalog_items=[stale, good])
    assert result["skipped"][0]["reason"] == (
        "your own media may fit: m7; look at them, or list them in not_fitting"
    )


def test_cli_search_uses_project_render_style_when_style_is_omitted(
    tmp_path,
    fake_pexels,
    monkeypatch,
    capsys,
):
    from yap_studio import cli
    from yap_studio.stock import command
    from yap_studio.style import resolve

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(command, "fetch", preview)
    assert resolve(None).data["broll"]["stock_clips"] == "off"
    project = tmp_path / "video"
    cut = project / "cut/cut-words.json"
    cut.parent.mkdir(parents=True)
    cut.write_text(json.dumps(words().to_dict()))
    source = project / "requests.json"
    source.write_text(json.dumps(requests()))
    style = project / "style.json"
    style.write_text(json.dumps({"schema_version": 1, **STYLE}))
    (project / "project.json").write_text(
        json.dumps(
            {
                "schema": "yap-studio/project@1",
                "name": "video",
                "style": "style.json",
                "takes": [{"id": "t1", "path": "source.mov"}],
                "asr": {"engine": "none"},
            }
        )
    )
    argv = ["stock", "search", str(source), "--cut-words", str(cut), "--json"]
    assert cli.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ok"
    assert any(call[:2] == ["videos", "search"] for call in calls(fake_pexels))
    off = project / "off.json"
    off.write_text('{"schema_version": 1}')
    assert cli.main([*argv, "--style", str(off)]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "stock_off"
