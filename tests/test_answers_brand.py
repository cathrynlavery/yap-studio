"""Brand answers become saved styles and readable render plans."""

from __future__ import annotations

import json
import time
from email.message import Message
from pathlib import Path

import pytest

from yap_studio import cards, cli
from yap_studio.render import fonts, user_fonts
from yap_studio.stages.captions import caption_layer_plan, title_layer_plan
from yap_studio.style import loader, site_colors
from yap_studio.style.answers import ANSWERS_SCHEMA, apply_answers
from yap_studio.style.library import library_dir, save_pick


def _file(tmp_path: Path, brand: dict) -> Path:
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"schema": ANSWERS_SCHEMA, "brand": brand}))
    return path


def test_brand_answers_before_a_look_stay_with_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    brand = {"captions_font": "Alata", "titles_font": "Gelasio", "cards_font": "Inter",
             "accent_color": "#114488"}
    file = _file(tmp_path, brand)
    result = apply_answers(file, workspace=tmp_path)
    assert result["style_id"] is None and result["style_saved"] is False
    assert set(result["fields_set"]) == {f"brand.{key}" for key in brand}
    assert json.loads(file.read_text())["brand"] == brand
    assert not library_dir().exists()
    assert not (tmp_path / "style/my-style.json").exists()


def test_website_accent_before_a_look_is_kept_in_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from yap_studio.style.answers import load_answers, style_changes
    from yap_studio.style.library import get_default

    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(site_colors, "site_colors", lambda _url: {
        "site": "example.test", "suggestion": {"accent": "#114488"}, "colors": [],
    })
    file = _file(tmp_path, {"website": "https://example.test"})
    result = apply_answers(file, workspace=tmp_path)
    assert result["style_saved"] is False and result["style_id"] is None
    assert style_changes(load_answers(file), loader.resolve().unbranded_data)["brand"] == {
        "accent_color": "#114488",
    }
    assert get_default() is None and not library_dir().exists()
    assert not (tmp_path / "style/my-style.json").exists()


def test_from_answers_brand_reads_in_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My look")
    brand = {"captions_font": "Alata", "titles_font": "Gelasio",
             "cards_font": "Inter", "accent_color": "#AABBCC"}
    assert cli.main(["style", "from-answers", str(_file(tmp_path, brand))]) == 0
    output = capsys.readouterr().out
    assert "Brand written" in output
    assert "Accent from answers" in output
    style = loader.resolve()
    assert style.unbranded_data["brand"] == brand | {"logo": ""}
    caption = caption_layer_plan([], style, frames=30)
    title = title_layer_plan("Short", "title", style)
    assert caption.variables["font_family"] == fonts.font_stack("Alata")
    assert caption.variables["highlight_color"] == "#AABBCC"
    assert title.variables["font_family"] == fonts.font_stack("Gelasio")
    assert title.variables["accent_color"] == "#AABBCC"
    slots = json.loads((cards.template_dir("text_tag") / "preview.json").read_text())["slots"]
    card = cards.card_layer_plan("text_tag", slots, style, frames=90)
    assert card.variables["font_family"] == fonts.font_stack("Inter")
    assert card.variables["accent_color"] == "#AABBCC"


def test_from_answers_saves_split_bundled_fonts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My look")
    brand = {"captions_font": "Instrument Serif", "titles_font": "Geist", "cards_font": "Geist"}
    result = apply_answers(_file(tmp_path, brand))
    assert not result["pending"]
    saved = json.loads((library_dir() / f"{result['style_id']}.json").read_text())
    assert saved["brand"] == brand | {"accent_color": "", "logo": ""}
    style = loader.resolve(result["style_id"])
    assert style.data["brand"] == saved["brand"]
    assert [style.data[section]["font"] for section in ("captions", "hook", "graphics")] == [
        "Instrument Serif", "Geist", "Geist",
    ]


def test_unknown_font_is_pending_and_creator_font_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My look")
    file = _file(tmp_path, {"captions_font": "Unknown Face", "titles_font": "Gelasio"})
    result = apply_answers(file)
    assert "Unknown Face" in result["pending"]["brand"]
    assert "brand.titles_font" in result["fields_set"]
    saved = json.loads((library_dir() / f"{result['style_id']}.json").read_text())
    assert not saved["brand"].get("captions_font")

    user_fonts.add_font_files([fonts.FONTS_DIR / "alata-regular.woff2"], family="Creator Face")
    file = _file(tmp_path, {"captions_font": "Creator Face"})
    result = apply_answers(file)
    assert not result["pending"]
    assert loader.resolve(result["style_id"]).data["captions"]["font"] == "Creator Face"


def test_accent_change_keeps_unbranded_look(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    save_pick({"schema_version": 1, "captions": {"font": "Inter"},
               "hook": {"font": "Gelasio"}, "graphics": {"font": "Alata"},
               "brand": {"captions_font": "Alata", "titles_font": "Inter",
                         "cards_font": "Gelasio", "accent_color": "#223344"}}, name="Start")
    result = apply_answers(_file(tmp_path, {"accent_color": "#AABBCC"}))
    assert result["fields_set"] == ["brand.accent_color"]
    saved = json.loads((library_dir() / f"{result['style_id']}.json").read_text())
    assert [saved[section]["font"] for section in ("captions", "hook", "graphics")] == [
        "Inter", "Gelasio", "Alata",
    ]
    assert saved["brand"] == {"captions_font": "Alata", "titles_font": "Inter",
                              "cards_font": "Gelasio", "accent_color": "#AABBCC", "logo": ""}


def test_empty_brand_answers_clear_saved_choices_and_null_preserves_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    original = save_pick({"schema_version": 1, "captions": {"font": "Inter"},
                          "hook": {"font": "Gelasio"}, "graphics": {"font": "Inter"},
                          "brand": {"captions_font": "Alata", "titles_font": "Alata",
                                    "cards_font": "Alata", "accent_color": "#223344"}},
                         name="Start")
    null_result = apply_answers(_file(tmp_path, {"captions_font": None,
                                                  "titles_font": None,
                                                  "cards_font": None,
                                                  "accent_color": None}))
    assert null_result["fields_set"] == []
    assert null_result["style_id"] == original.id
    assert loader.resolve().data["brand"]["captions_font"] == "Alata"
    assert loader.resolve().data["brand"]["accent_color"] == "#223344"

    cleared = apply_answers(_file(tmp_path, {"captions_font": "", "titles_font": "",
                                             "cards_font": "", "accent_color": ""}))
    assert set(cleared["fields_set"]) == {
        "brand.captions_font", "brand.titles_font", "brand.cards_font", "brand.accent_color",
    }
    saved = json.loads((library_dir() / f"{cleared['style_id']}.json").read_text())
    assert saved["brand"] == {"captions_font": "", "titles_font": "", "cards_font": "",
                              "accent_color": "", "logo": ""}
    style = loader.resolve(cleared["style_id"])
    assert caption_layer_plan([], style, frames=30).variables["font_family"] == (
        fonts.font_stack("Inter")
    )


def test_website_color_and_explicit_accent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My look")
    html = '<link rel="stylesheet" href="/site.css"><style>body{background:#F4F3EE}</style>'
    css = ':root{--brand-accent:#C03A62}a:hover{color:#C03A62}'
    calls = []

    def fetch(url: str, limit: int, timeout: float) -> tuple[str, str]:
        calls.append(url)
        return (html if url.endswith("example.test") else css), url

    monkeypatch.setattr(site_colors, "_get", fetch)
    result = apply_answers(_file(tmp_path, {"website": "https://example.test"}))
    assert calls == ["https://example.test", "https://example.test/site.css"]
    assert result["website_colors"]["used"] == "#C03A62"
    assert result["website_colors"]["colors"]
    assert loader.resolve(result["style_id"]).data["brand"]["accent_color"] == "#C03A62"

    calls.clear()
    result = apply_answers(_file(tmp_path, {"website": "https://example.test",
                                            "accent_color": "#112233"}))
    assert calls == []
    assert result["website_colors"] is None
    assert loader.resolve(result["style_id"]).data["brand"]["accent_color"] == "#112233"


def test_empty_accent_does_not_fetch_website(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(site_colors, "site_colors", lambda _url: pytest.fail("website fetched"))
    result = apply_answers(_file(tmp_path, {"website": "https://example.test",
                                            "accent_color": ""}))
    assert result["website_colors"] is None
    assert "brand.accent_color" not in result["fields_set"]


def test_website_error_is_pending_and_other_brand_is_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My look")

    def fail(url: str, limit: int, timeout: float) -> tuple[str, str]:
        raise site_colors.FetchError("The site did not respond.")

    monkeypatch.setattr(site_colors, "_get", fail)
    result = apply_answers(_file(tmp_path, {"website": "https://example.test",
                                            "captions_font": "Alata"}))
    assert result["pending"]["brand"] == (
        "Could not read colors from https://example.test: The site did not respond. "
        "Pick an accent in the gallery."
    )
    assert result["fields_set"] == ["brand.captions_font"]
    assert loader.resolve(result["style_id"]).data["brand"]["captions_font"] == "Alata"


@pytest.mark.parametrize("website", ["http://[bad", "https://example.test:bad"])
def test_malformed_website_is_pending_and_font_is_saved(
    website: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My look")
    result = apply_answers(_file(tmp_path, {"website": website,
                                            "captions_font": "Alata"}))
    assert website in result["pending"]["brand"]
    assert result["fields_set"] == ["brand.captions_font"]
    assert loader.resolve(result["style_id"]).data["brand"]["captions_font"] == "Alata"


def test_website_cli_reports_accent_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    save_pick(loader.resolve("preset:clean_minimal").unbranded_data, name="My look")
    monkeypatch.setattr(site_colors, "_get", lambda url, limit, timeout: (
        '<meta name="theme-color" content="#C03A62">', url,
    ))
    assert cli.main(["style", "from-answers",
                     str(_file(tmp_path, {"website": "https://example.test"}))]) == 0
    output = capsys.readouterr().out
    assert "Brand written: brand.accent_color." in output
    assert "Accent from example.test: #C03A62." in output


def test_gallery_choice_without_brand_fields_is_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    result = apply_answers(_file(tmp_path, {"pick_in_gallery": True}))
    assert result["style_id"] is None
    assert result["pending"]["brand"] == "Pick your brand in the gallery."


@pytest.mark.parametrize("address", ["127.0.0.1", "10.2.3.4", "192.168.1.5", "::1"])
def test_site_colors_refuses_nonpublic_addresses(
    address: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(site_colors.socket, "getaddrinfo",
                        lambda *_args, **_kwargs: [(None, None, None, None, (address, 443))])
    with pytest.raises(site_colors.FetchError, match="not a public website"):
        site_colors._public_ip("example.test", 443)


def test_site_colors_caps_response_size_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        headers = Message()

        def __init__(self) -> None:
            self.requests: list[int] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def geturl(self) -> str:
            return "https://example.test"

        def read1(self, size: int) -> bytes:
            self.requests.append(size)
            return b"a" * size

    response = Response()
    monkeypatch.setattr(site_colors._OPENER, "open", lambda *_args, **_kwargs: response)
    site_colors._LOOKUP.deadline = time.monotonic() + 5
    try:
        body, _ = site_colors._get("https://example.test", 100)
        assert len(body) == 100
        assert response.requests == [100]
        site_colors._LOOKUP.deadline = time.monotonic() - 1
        with pytest.raises(site_colors.FetchError, match="took too long"):
            site_colors._get("https://example.test", 100)
    finally:
        del site_colors._LOOKUP.deadline
