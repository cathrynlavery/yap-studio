"""Brand resolution and creator fonts across the render plans."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from yap_studio import cards, cli, ffmpeg
from yap_studio.render import fonts, user_fonts
from yap_studio.stages.captions import (
    caption_layer_plan,
    layer_assets,
    prepare_layer_plan,
    render_layer_plan,
    title_layer_plan,
)
from yap_studio.stages.graphics import _card_assets
from yap_studio.style import ResolvedStyle
from yap_studio.style.loader import (
    PRESETS_DIR,
    SHIPPED_PRESETS,
    StyleError,
    apply_brand,
    load_style,
    resolve,
    resolve_style_data,
    unbranded,
)

ROOT = Path(__file__).parents[1]
PROPOSED = ROOT / "docs/design/style-page/proposed-presets"


@pytest.fixture
def creator_font(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    source = fonts.FONTS_DIR / "alata-regular.woff2"
    target = tmp_path / "creator.woff2"
    shutil.copyfile(source, target)
    return target


def _style(**brand: str) -> ResolvedStyle:
    data = resolve_style_data({"schema_version": 1, "brand": brand})
    return ResolvedStyle(data["name"], "test", data)


def _sfnt_face(
    path: Path, family: str, subfamily: str, weight: int, *, italic: bool = False,
    variable: bool = False,
) -> Path:
    """Write the name and OS/2 tables needed to identify a static face."""
    names = []
    strings = bytearray()
    for name_id, value in ((1, family), (2, subfamily)):
        raw = value.encode("utf-16-be")
        names.append(struct.pack(">HHHHHH", 3, 1, 0x409, name_id, len(raw), len(strings)))
        strings.extend(raw)
    name = struct.pack(">HHH", 0, len(names), 6 + 12 * len(names)) + b"".join(names) + strings
    os2 = bytearray(64)
    struct.pack_into(">H", os2, 4, weight)
    struct.pack_into(">H", os2, 62, int(italic))
    tables = [(b"name", name), (b"OS/2", os2)]
    if variable:
        tables.append((b"fvar", b""))
    offset = 12 + 16 * len(tables)
    header = bytearray(b"\x00\x01\x00\x00" + struct.pack(">HHHH", len(tables), 0, 0, 0))
    body = bytearray()
    for tag, data in tables:
        header.extend(struct.pack(">4sIII", tag, 0, offset, len(data)))
        body.extend(data)
        offset += len(data)
    path.write_bytes(header + body)
    return path


def test_brand_schema_and_existing_presets() -> None:
    brand = {"captions_font": "A", "titles_font": "B", "cards_font": "C",
             "accent_color": "#123ABC", "logo": "logo.png"}
    assert resolve_style_data({"schema_version": 1, "brand": brand})["brand"] == brand
    assert resolve_style_data({"schema_version": 1, "brand": {"accent_color": ""}})
    for bad in ("#12345", "red"):
        with pytest.raises(StyleError, match="expected a color like #1A2B3C, or empty"):
            resolve_style_data({"schema_version": 1, "brand": {"accent_color": bad}})
    with pytest.raises(StyleError, match="unknown field"):
        resolve_style_data({"schema_version": 1, "brand": {"unknown": "value"}})
    for path in [*(PRESETS_DIR / f"{name}.json" for name in SHIPPED_PRESETS),
                 *(p for p in PROPOSED.glob("*.json") if p.name != "order.json")]:
        original = json.loads(path.read_text(encoding="utf-8"))
        resolved = load_style(path)
        assert resolved["brand"] == {key: "" for key in brand}
        assert resolved == unbranded(original)


def test_apply_brand_is_idempotent_and_keeps_empty_looks() -> None:
    original = resolve_style_data({"schema_version": 1})
    assert apply_brand(original) == original
    for accent, ink in (("#111111", "#FFFFFF"), ("#EEEEEE", "#111111")):
        data = resolve_style_data({"schema_version": 1, "brand": {
            "captions_font": "Caption Face", "titles_font": "Title Face",
            "cards_font": "Card Face", "accent_color": accent}})
        assert [data[s]["font"] for s in ("captions", "hook", "graphics")] == [
            "Caption Face", "Title Face", "Card Face"]
        assert [data[s]["accent_color" if s != "captions" else "highlight_color"]
                for s in ("captions", "hook", "graphics")] == [accent] * 3
        assert data["captions"]["highlight_text_color"] == ink
        assert apply_brand(data) == data
        assert resolve_style_data(data) == data
        own = unbranded({"schema_version": 1, "brand": data["brand"],
                         "captions": {"font": "Inter", "highlight_color": "#365A58"}})
        assert own["brand"] == data["brand"]
        assert own["captions"]["font"] == "Inter"
        assert own["captions"]["highlight_color"] == "#365A58"


def test_resolved_style_keeps_its_own_look(tmp_path: Path) -> None:
    raw = {"schema_version": 1, "captions": {"font": "Inter"},
           "brand": {"captions_font": "Gelasio"}}
    source = tmp_path / "branded.json"
    source.write_text(json.dumps(raw))
    assert unbranded(raw)["captions"]["font"] == "Inter"
    style = resolve(source)
    assert style.data["captions"]["font"] == "Gelasio"
    assert style.unbranded_data["captions"]["font"] == "Inter"
    assert "unbranded_data" not in style.data


def test_concurrent_font_imports_keep_both_faces(
    creator_font: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = user_fonts._metadata
    rendezvous = threading.Barrier(2)

    def competing_read() -> list[user_fonts.UserFont]:
        result = original()
        if not result:
            try:
                rendezvous.wait(timeout=1)
            except threading.BrokenBarrierError:
                pass
        return result

    monkeypatch.setattr(user_fonts, "_metadata", competing_read)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(user_fonts.add_font_files, [creator_font], family=name)
                   for name in ("First Face", "Second Face")]
        for future in futures:
            future.result()
    assert {face.family for face in user_fonts.list_fonts()} == {"First Face", "Second Face"}


@pytest.mark.parametrize("family", ["Inter", "iNtEr"])
def test_bundled_font_family_is_refused(creator_font: Path, family: str) -> None:
    with pytest.raises(user_fonts.UserFontError,
                       match=f"{family} is a bundled font; choose another name with --family"):
        user_fonts.add_font_files([creator_font], family=family)


@pytest.mark.parametrize("bad_name", ["../escaped.woff2", "user-BAD-12345678-normal-400.woff2"])
def test_unsafe_generated_font_name_is_refused(
    creator_font: Path, monkeypatch: pytest.MonkeyPatch, bad_name: str,
) -> None:
    monkeypatch.setattr(user_fonts, "_stored_name", lambda *args: bad_name)
    with pytest.raises(user_fonts.UserFontError, match="unsafe font file name"):
        user_fonts.add_font_files([creator_font], family="Creator Face")
    assert not (user_fonts.user_fonts_dir().parent / "escaped.woff2").exists()


def test_all_template_plans_read_brand(tmp_path: Path) -> None:
    style = _style(captions_font="Caption Face", titles_font="Title Face",
                   cards_font="Card Face", accent_color="#AABBCC")
    caption = caption_layer_plan([], style, frames=30)
    title = title_layer_plan("Short", "title", style)
    assert caption.variables["font_family"] == fonts.font_stack("Caption Face")
    assert caption.variables["highlight_color"] == "#AABBCC"
    assert title.variables["font_family"] == fonts.font_stack("Title Face")
    assert title.variables["accent_color"] == "#AABBCC"
    for card in cards.CARD_TEMPLATES:
        slots = json.loads((cards.template_dir(card) / "preview.json").read_text())["slots"]
        rules = cards.load_template(card)["slots"]
        image_slots = [key for key, rule in rules.items()
                       if rule["type"] == "image"]
        image_lists = [key for key, rule in rules.items()
                       if rule["type"] == "image_list" and not slots.get(key)]
        if image_slots or image_lists:
            image = tmp_path / "image.png"
            if not image.exists():
                ffmpeg.run(["ffmpeg", "-nostdin", "-f", "lavfi", "-i",
                            "testsrc2=size=640x400:rate=1", "-frames:v", "1", "-threads", "1",
                            str(image)], timeout_s=30)
            slots.update({name: image for name in image_slots})
            if image_lists:
                # Use short labels for the image tile sample.
                slots["items"] = [f"{index + 1:02}" for index in range(len(slots["items"]))]
            slots.update({name: [image] * len(slots["items"]) for name in image_lists})
        video_slots = [key for key, rule in rules.items() if rule["type"] == "video"
                       and "sources" not in rule]
        if video_slots:
            from test_file_drop import clip_source

            clip = clip_source(tmp_path)
            slots.update({name: clip for name in video_slots})
        slots = cards.preview_videos(card, slots, tmp_path / card)
        plan = cards.card_layer_plan(card, slots, style, frames=90)
        assert plan.variables["font_family"] == fonts.font_stack("Card Face")
        expected_mono = cards.MONO_FONT if card == "terminal" else "Card Face"
        assert plan.variables["mono_family"] == fonts.font_stack(expected_mono)
        assert (cards.MONO_FONT in plan.fonts) == (card == "terminal")
        assert plan.variables["accent_color"] == "#AABBCC"


def test_template_plans_use_split_bundled_fonts(tmp_path: Path) -> None:
    style = _style(captions_font="Instrument Serif", titles_font="Geist", cards_font="Geist")
    caption = caption_layer_plan([], style, frames=30)
    title = title_layer_plan("Short", "title", style)
    card = cards.card_layer_plan("text_tag", {"text": "Hello"}, style, frames=75)
    assert caption.variables["font_family"] == fonts.font_stack("Instrument Serif")
    assert title.variables["font_family"] == fonts.font_stack("Geist")
    assert card.variables["font_family"] == fonts.font_stack("Geist")
    assert "Geist" in card.fonts
    assert "Instrument Serif" not in card.fonts
    prepared = cards.prepare_card(card, tmp_path / "cards")
    assert (prepared / "fonts" / "geist-variable.woff2").read_bytes() == (
        fonts.FONTS_DIR / "geist-variable.woff2"
    ).read_bytes()


def test_creator_font_assets_and_receipt(creator_font: Path, tmp_path: Path) -> None:
    face = user_fonts.add_font_files([creator_font], family="Creator Face")[0]
    assert face.file.startswith("user-")
    assert face.file not in {item.file for font in fonts.BUNDLED.values() for item in font.faces}
    assert face.family == "Creator Face"
    assert user_fonts.find("Creator Face") == [face]
    assert user_fonts.list_fonts() == [face]
    assert (user_fonts.user_fonts_dir() / face.file).read_bytes() == creator_font.read_bytes()
    assert json.loads((user_fonts.user_fonts_dir() / "fonts.json").read_text())["schema"] == (
        "yap-studio/user-fonts@1")
    with pytest.raises(user_fonts.UserFontError):
        user_fonts.add_font_files([fonts.FONTS_DIR / "anton-sc-regular.woff2"],
                                  family="Creator Face")
    css = fonts.font_face_css(["Creator Face"])
    assert "Yap Studio User Creator Face" in css
    assert f'fonts/{face.file}' in css
    assert fonts.font_stack("Creator Face") == '"Yap Studio User Creator Face", sans-serif'
    assert fonts.resolve_font("Creator Face").warning is None
    assert (tmp_path / "copied" / face.file) in fonts.copy_fonts(
        ["Creator Face"], tmp_path / "copied")
    style = _style(captions_font="Creator Face", titles_font="Creator Face",
                   cards_font="Creator Face")
    caption = caption_layer_plan([], style, frames=30)
    prepared = prepare_layer_plan(caption, tmp_path / "layers")
    assert (prepared / "fonts" / face.file).read_bytes() == creator_font.read_bytes()
    card = cards.card_layer_plan("text_tag", {"text": "Hello"}, style, frames=75)
    prepared_card = cards.prepare_card(card, tmp_path / "cards")
    assert (prepared_card / "fonts" / face.file).is_file()
    context = SimpleNamespace(read=lambda path: Path(path).read_bytes())
    before = layer_assets("captions", "Creator Face", context).font_identity()
    card_before = _card_assets(card, context).font_identity()
    user_fonts.add_font_files([fonts.FONTS_DIR / "anton-sc-regular.woff2"],
                              family="Creator Face", replace=True)
    after = layer_assets("captions", "Creator Face", context).font_identity()
    card_after = _card_assets(card, context).font_identity()
    assert before != after
    assert card_before != card_after


def test_creator_font_validation(creator_font: Path, tmp_path: Path) -> None:
    inferred = tmp_path / "CreatorSerif-Italic.woff2"
    inferred.write_bytes(creator_font.read_bytes())
    assert user_fonts.add_font_files([inferred])[0].family == "Creator Serif"
    invalid = tmp_path / "invalid.woff2"
    invalid.write_bytes(b"not a font")
    with pytest.raises(user_fonts.UserFontError, match="invalid font signature"):
        user_fonts.add_font_files([invalid], family="Invalid")
    oversized = tmp_path / "large.woff2"
    with oversized.open("wb") as stream:
        stream.write(b"wOF2")
        stream.truncate(user_fonts.MAX_BYTES + 1)
    with pytest.raises(user_fonts.UserFontError, match="larger than 20 MB"):
        user_fonts.add_font_files([oversized], family="Large")
    wrong_suffix = tmp_path / "font.bin"
    wrong_suffix.write_bytes(creator_font.read_bytes())
    with pytest.raises(user_fonts.UserFontError, match="WOFF2, WOFF, TTF or OTF"):
        user_fonts.add_font_files([wrong_suffix], family="Wrong")


def test_regular_and_bold_faces_coexist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    regular = _sfnt_face(tmp_path / "Brand-Regular.ttf", "Brand", "Regular", 400)
    bold = _sfnt_face(tmp_path / "Brand-Bold.ttf", "Brand", "Bold", 700)
    faces = user_fonts.add_font_files([regular, bold])
    assert [face.weight for face in faces] == ["400", "700"]
    assert len({face.file for face in faces}) == 2
    assert user_fonts.list_fonts() == faces
    css = fonts.font_face_css(["Brand"])
    for face in faces:
        assert f'fonts/{face.file}' in css
        assert f"font-weight: {face.weight};" in css
    changed_bold = _sfnt_face(bold, "Brand", "Bold", 700)
    with changed_bold.open("ab") as stream:
        stream.write(b"updated")
    replacement = user_fonts.add_font_files([changed_bold], replace=True)[0]
    assert [face.weight for face in user_fonts.list_fonts()] == ["400", "700"]
    assert replacement.file == faces[1].file
    assert replacement.sha256 != faces[1].sha256


def test_os2_italic_weight_clamp_and_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))
    italic = _sfnt_face(tmp_path / "Static.otf", "Static", "Regular", 1200, italic=True)
    variable = _sfnt_face(tmp_path / "Variable.ttf", "Variable", "Regular", 0,
                          variable=True)
    faces = user_fonts.add_font_files([italic, variable])
    assert [(face.style, face.weight) for face in faces] == [
        ("italic", "1000"), ("normal", "100 900")]


@pytest.mark.parametrize(("stem", "weight", "style"), [
    ("Face-UltraLight", "200", "normal"),
    ("Face-SemiBold", "600", "normal"),
    ("Face-ExtraBoldItalic", "800", "italic"),
    ("Face-Black", "900", "normal"),
    ("Face-VF", "100 900", "normal"),
    ("Face-Oblique", "400", "italic"),
])
def test_woff2_stem_faces(
    creator_font: Path, tmp_path: Path, stem: str, weight: str, style: str,
) -> None:
    source = tmp_path / f"{stem}.woff2"
    source.write_bytes(creator_font.read_bytes())
    face = user_fonts.add_font_files([source])[0]
    assert (face.family, face.style, face.weight) == ("Face", style, weight)


def test_folder_add_skips_hidden_and_nonfont_files(
    creator_font: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    folder = tmp_path / "faces"
    folder.mkdir()
    assert cli.main(["brand", "fonts", "add", str(folder), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    (folder / "B-Bold.woff2").write_bytes(creator_font.read_bytes())
    (folder / "A-Regular.woff2").write_bytes(creator_font.read_bytes())
    (folder / ".Hidden.woff2").write_bytes(creator_font.read_bytes())
    (folder / "notes.txt").write_text("notes")
    (folder / "nested").mkdir()
    (folder / "nested" / "C-Light.woff2").write_bytes(creator_font.read_bytes())
    assert cli.main(["brand", "fonts", "add", str(folder), "--json"]) == 0
    faces = json.loads(capsys.readouterr().out)
    assert [(face["family"], face["weight"]) for face in faces] == [
        ("A", "400"), ("B", "700")]


def test_brand_fonts_cli(creator_font: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["brand", "fonts", "add", str(creator_font), "--family", "CLI Face"]) == 0
    assert cli.main(["brand", "fonts", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out.split("\n", 1)[1])[0]["family"] == "CLI Face"
    assert cli.main(["brand", "fonts", "add", str(creator_font.parent / "missing.woff2"),
                     "--family", "Missing"]) == 2
    assert "error:" in capsys.readouterr().err


def test_brand_parser_keeps_font_implementation_lazy() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from yap_studio.cli import _parser; "
         "_parser(); assert 'yap_studio.render.user_fonts' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.render
def test_branded_creator_caption_renders(creator_font: Path, tmp_path: Path) -> None:
    from test_captions_render import _alpha_frames

    from yap_studio.captions import CaptionWord, build_phrases

    user_fonts.add_font_files([creator_font], family="Creator Face")
    style = _style(captions_font="Creator Face", accent_color="#3388AA")
    phrases = build_phrases([CaptionWord("one", "Hello", 0.2, 0.7)], style.data["captions"])
    plan = caption_layer_plan(phrases, style, frames=30)
    out = render_layer_plan(plan, tmp_path / "render", tmp_path / "caption.mov").path
    assert _alpha_frames(out, plan)[15].max() > 0
    assert (tmp_path / "render" / "captions" / "fonts" /
            user_fonts.find("Creator Face")[0].file).is_file()
