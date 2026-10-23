"""Opt-in letters plans, spoken copy, timing, and portable requests."""

from __future__ import annotations

import copy
import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_planner_broll import catalog_item, phrase_words, requests
from test_stage_captions import layer_project as layer_project
from test_timeline import project as build_timeline_project

from yap_studio.errors import ValidationError
from yap_studio.planner.broll import plan_broll
from yap_studio.project.model import ProjectFileError, parse_document
from yap_studio.project.receipts import projection_sha256
from yap_studio.stages.captions import title_layer_plan
from yap_studio.stages.graphics import TitleSnapshot
from yap_studio.stages.timeline import ResolvedElement
from yap_studio.style.loader import ResolvedStyle, resolve_style_data


def letters():
    return importlib.import_module("yap_studio.letters")


@pytest.fixture
def timeline_project(tmp_path: Path) -> Path:
    return build_timeline_project.__wrapped__(tmp_path)


def style(*, opening: bool = True, emphasis: bool = False) -> ResolvedStyle:
    data = resolve_style_data({
        "schema_version": 1, "name": "Synthetic letters",
        "hook": {"style": "video_letters" if opening else "quiet_serif", "font": "Inter"},
        "broll": {"emphasis_on": ["video_letters"] if emphasis else []},
    })
    return ResolvedStyle(data["name"], "test", data)


def item(*, start: int = 0, frames: int = 48, text: str = "Dog park", **options):
    return ResolvedElement("broll-letters", "insert", "broll", start, start + frames,
                           "front", {"letters": {"text": text, **options}})


def findings(element=None, *, selected=None, image=False, count=1, words=None):
    return letters().check(element or item(), (selected or style()).data,
                           words or phrase_words(), image=image, mid_count=count)


def test_fit_one_line_and_cap() -> None:
    assert letters().fit("I", "Inter") == (["I"], 640)
    lines, size = letters().fit("Picture", "Inter")
    assert lines == ["PICTURE"] and 150 <= size <= 640


def test_fit_two_lines_choose_larger() -> None:
    module = letters()
    lines, size = module.fit("Dog park", "Inter")
    assert lines == ["DOG", "PARK"]
    from yap_studio.render.text_metrics import text_width_em
    assert size > int(960 / text_width_em("DOG PARK", "Inter", 900))


@pytest.mark.parametrize(("text", "token"), [
    ("İT", "it"), ("ÉCOLE", "ecole"), ("naïve", "naive"), ("e\u0301cole", "ecole"),
], ids=["dotted_i", "precomposed_accent", "diaeresis", "decomposed_accent"])
def test_fit_unicode_accents_count_as_one_word(text: str, token: str) -> None:
    module = letters()
    lines, size = module.fit(text, "Inter")
    assert lines == [text.upper()]
    assert size >= module.MIN_SIZE_PX
    assert module._tokens(text) == [token]


def test_fit_dotted_i_pair_stacks_after_hyphen() -> None:
    lines, size = letters().fit("İT-PARK", "Inter")
    assert lines == ["İT-", "PARK"]
    assert size >= letters().MIN_SIZE_PX


@pytest.mark.parametrize("separator", ["-", "/", " "], ids=["hyphen", "slash", "spaced"])
def test_fit_pair_stacks_above_floor_and_preserves_punctuation(separator: str) -> None:
    from yap_studio.render.text_metrics import text_width_em

    module = letters()
    word = "WWWW"
    text = f"{word}{separator}{word}"
    expected = [word + separator.rstrip(), word]
    one_line_size = int(module.TEXT_WIDTH_PX / text_width_em(text, "Inter", module.WEIGHT))
    stacked_size = int(module.TEXT_WIDTH_PX / max(
        text_width_em(line, "Inter", module.WEIGHT) for line in expected))
    assert one_line_size < module.MIN_SIZE_PX <= stacked_size
    assert module.fit(text, "Inter") == (expected, stacked_size)


def test_fit_floor_names_required_size_and_shorter_word() -> None:
    with pytest.raises(ValidationError, match=r"needs \d+ px.*150 px.*about \d+ letters"):
        letters().fit("W" * 40, "Inter")


def test_timeline_fit_fix_names_a_shorter_spoken_word() -> None:
    result = findings(item(text="W" * 40), words=phrase_words("W" * 40, "Dog"))
    fix = next(f.message for f in result if f.rule == "letters_fit")
    assert 'Try "Dog" from the spoken line.' in fix and "150 px" in fix


@pytest.mark.parametrize("text", ["", "dog park again"])
def test_text_refuses_other_word_counts(text: str) -> None:
    result = findings(item(text=text))
    assert "letters_text" in {f.rule for f in result}
    assert "one or two words from the spoken line" in next(
        f.message for f in result if f.rule == "letters_text")
    with pytest.raises(ValidationError, match="one or two words from the spoken line"):
        letters().fit(text, "Inter")


@pytest.mark.parametrize(("lines", "rule"), [
    (["Dog park", "again"], "letters_text"),
    (["dog runs"], "letters_words"),
    (["W" * 40], "letters_fit"),
], ids=["letters_text", "letters_words", "letters_fit"])
def test_title_findings_name_lines(lines, rule) -> None:
    title = ResolvedElement("agent-title", "title", "agent", 0, 48,
                            "front", {"lines": lines})
    result = letters().check_timeline([title], style().data, phrase_words(), {})
    message = next(f.message for f in result if f.rule == rule)
    assert "lines" in message
    assert "letters.text" not in message


def test_spoken_words_ignore_case_and_punctuation_and_require_consecutive() -> None:
    assert not findings(item(text="DOG, park!"))
    result = findings(item(text="dog runs"))
    assert [f.rule for f in result] == ["letters_words"]
    assert "Dog park." in result[0].message and "Change" in result[0].message
    assert "consecutive words from the spoken line" in result[0].message


@pytest.mark.parametrize(("text", "spoken", "expected"), [
    ("DOG-PARK", ("Dog", "park"), ["dog", "park"]),
    ("DOG PARK", ("Dog-park",), ["dog", "park"]),
    ("DOG/PARK", ("Dog", "park"), ["dog", "park"]),
    ("HERE’S,DOG", ("Here's", "dog"), ["heres", "dog"]),
], ids=["hyphenated_copy", "hyphenated_spoken", "slash", "curly_contraction"])
def test_spoken_words_normalize_separators_and_contractions(text, spoken, expected) -> None:
    assert letters()._tokens(text) == expected
    assert letters()._tokens(" ".join(spoken)) == expected
    assert not findings(item(text=text), words=phrase_words(*spoken))


@pytest.mark.parametrize(("text", "spoken", "token"), [
    ("e\u0301", "é", "e"), ("é", "e\u0301", "e"),
    ("İT", "it", "it"), ("NAÏVE!", "naive", "naive"),
    ("ecole", "ÉCOLE", "ecole"),
], ids=["decomposed_copy", "decomposed_spoken", "dotted_i", "diaeresis", "accented_spoken"])
def test_spoken_words_ignore_unicode_accents(text: str, spoken: str, token: str) -> None:
    assert letters()._tokens(text) == [token]
    assert letters()._tokens(spoken) == [token]
    assert not findings(item(text=text), words=phrase_words(spoken))


@pytest.mark.parametrize("separator", ["-", "/"], ids=["hyphen", "slash"])
def test_text_counts_punctuation_separated_words(separator) -> None:
    text = separator.join(["Dog", "park"])
    assert not findings(item(text=text))
    letters().fit(text, "Inter")
    text += separator + "again"
    result = findings(item(text=text), words=phrase_words("Dog", "park", "again"))
    assert "letters_text" in {f.rule for f in result}
    with pytest.raises(ValidationError, match="one or two words from the spoken line"):
        letters().fit(text, "Inter")


def test_spoken_words_window_is_widened_by_90_frames() -> None:
    assert not findings(item(start=100), selected=style(emphasis=True))
    assert "letters_words" in {f.rule for f in findings(
        item(start=140), selected=style(emphasis=True))}


@pytest.mark.parametrize(("opening", "emphasis", "start", "allowed"), [
    (False, False, 0, False), (True, False, 29, True), (True, False, 30, False),
    (False, True, 0, True), (False, True, 45, True),
])
def test_style_gating(opening, emphasis, start, allowed) -> None:
    rules = {f.rule for f in findings(item(start=start), selected=style(
        opening=opening, emphasis=emphasis))}
    assert ("letters_off" not in rules) is allowed


@pytest.mark.parametrize(("image", "frames", "allowed"), [
    (False, 47, False), (False, 48, True), (False, 60, True),
    (True, 47, False), (True, 48, True), (True, 49, False),
])
def test_lengths_and_photo_hold_fix(image, frames, allowed) -> None:
    result = findings(item(frames=frames), image=image)
    assert ("letters_length" not in {f.rule for f in result}) is allowed
    if image and not allowed:
        assert "hold_s: 1.6" in next(f.message for f in result if f.rule == "letters_length")


def test_custom_lengths_and_mid_count_warning() -> None:
    assert letters().phase_frames({"word_hold_s": 1, "reveal_frames": 10}) == 40
    assert letters().phase_frames({"word_hold_s": 1.5, "reveal_frames": 14}) == 59
    for options in ({"word_hold_s": "bad"}, {"reveal_frames": "bad"}, ["bad"]):
        with pytest.raises(ValidationError, match="Set"):
            letters().phase_frames(options)
    result = findings(item(start=30), selected=style(emphasis=True), count=3)
    assert [(f.rule, f.severity) for f in result] == [("letters_count", "warning")]


@pytest.mark.parametrize("options", [
    {"text": "Dog"}, {"text": "Dog park", "background": "blur", "word_hold_s": 1.5,
                          "reveal_frames": 14},
])
def test_schema_accepts_letters(options) -> None:
    letters()
    element = {"id": "broll-dog", "kind": "insert", "start": {"time": 0}, "hold_s": 1.6,
               "media": {"path": "dog.mov", "sha256": "a" * 64}, "letters": options}
    parsed = parse_document("timeline", {"schema": "yap-studio/timeline@1",
                                         "words_sha256": {},
                                         "elements": [element]}, lane="broll")
    assert parsed.elements[0].fields["letters"] == options


@pytest.mark.parametrize("options", [
    {}, {"text": ""}, {"text": "x" * 41}, {"text": "Dog", "background": "noise"},
    {"text": "Dog", "word_hold_s": .9}, {"text": "Dog", "word_hold_s": 1.6},
    {"text": "Dog", "reveal_frames": 9}, {"text": "Dog", "reveal_frames": 15},
    {"text": "Dog", "reveal_frames": 12.5}, {"text": "Dog", "other": True},
])
def test_schema_rejects_letters(options) -> None:
    letters()
    with pytest.raises(ProjectFileError):
        parse_document("timeline", {"schema": "yap-studio/timeline@1", "words_sha256": {},
                                    "elements": [{
            "id": "broll-dog", "kind": "insert", "start": {"time": 0}, "hold_s": 1.6,
            "media": {"path": "dog.mov", "sha256": "a" * 64}, "letters": options,
        }]}, lane="broll")


def test_planner_copies_letters_defaults_cover_and_photo_hold(tmp_path: Path) -> None:
    letters()
    candidate = catalog_item(tmp_path, kind="image", orientation="horizontal")
    request = requests(kind="insert", letters={"text": "Dog park"})
    del request["broll"][0]["end"]
    before = copy.deepcopy(request)
    result = plan_broll(request, phrase_words(), [candidate], style().data,
                        project_dir=tmp_path, frames=120)
    element = result.broll[0]["element"]
    assert element["letters"] == {"text": "Dog park"}
    assert element["media"]["fit"] == "cover" and element["hold_s"] == 1.6
    assert request == before


@pytest.mark.parametrize("kind", ["insert", "overlay"])
def test_planner_refuses_disabled_letters_and_overlays(tmp_path: Path, kind) -> None:
    module = letters()
    candidate = catalog_item(tmp_path)
    with pytest.raises(ValidationError) as error:
        plan_broll(requests(kind=kind, letters={"text": "Dog"}), phrase_words(),
                   [candidate], style(opening=kind == "overlay").data,
                   project_dir=tmp_path, frames=120)
    assert (module.permission_message(0) if kind == "insert" else "insert") in str(error.value)


def test_planner_checks_permission_before_missing_duration(tmp_path: Path) -> None:
    candidate = catalog_item(tmp_path)
    request = requests(kind="insert", letters={"text": "Dog"})
    del request["broll"][0]["end"]
    with pytest.raises(ValidationError, match=letters().permission_message(0)):
        plan_broll(request, phrase_words(), [candidate], style(opening=False).data,
                   project_dir=tmp_path, frames=120)


@pytest.mark.parametrize("start", [37, 1607])
@pytest.mark.parametrize("background", ["color", "blur"])
def test_composite_window_uses_insert_frames(start, background) -> None:
    from yap_studio.stages.assemble_picture import _Graph

    graph = _Graph()
    letters().composite(graph, "fill", "plate", item(start=start, background=background))
    assert "enable='between(n,0,47)'" in graph.filters[-1]


def test_timeline_title_defaults_to_letters_phase(timeline_project: Path) -> None:
    from test_timeline import fragment

    from yap_studio.stages import timeline

    selected = style()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(timeline, "_style", lambda context: selected.data)
        result = timeline.add_timeline(timeline_project, fragment(timeline_project, {
            "id": "agent-title", "kind": "title", "lane": "agent", "start": {"time": 0},
            "lines": ["alpha"],
        }), lane="agent")
    assert result.elements[0].end_frame == 48
    assert not any(f.rule.startswith("letters_") for f in result.findings)


def test_plate_cache_identity_and_content_key(tmp_path: Path, monkeypatch) -> None:
    from yap_studio.stages import captions
    from yap_studio.stages.captions import LayerAssets
    from yap_studio.stages.timeline import ResolvedTimeline

    module = letters()
    calls = []
    font_bytes = [b"font one"]

    def render(plan, work, out, **kwargs):
        calls.append(plan)
        out.write_bytes(b"alpha movie")

    monkeypatch.setattr(captions, "render_layer_plan", render)
    monkeypatch.setattr(module, "_assert_layer", lambda movie, frames: None)
    monkeypatch.setattr(module, "plate_assets", lambda font: LayerAssets(
        "plate", b'{"version":"1"}', "font", (("font.woff2", font_bytes[0]),)))
    timeline = ResolvedTimeline((item(),), (), (), 90, 0)
    first = module.render_plates(tmp_path, timeline, style(), tmp_path / "work")
    assert module.render_plates(tmp_path, timeline, style(), tmp_path / "work") == first
    assert len(calls) == 1
    font_bytes[0] = b"font two"
    second = module.render_plates(tmp_path, timeline, style(), tmp_path / "work")
    assert len(calls) == 2 and first[0]["path"] != second[0]["path"]
    Path(second[0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValidationError, match="Remove the incomplete cache"):
        module.render_plates(tmp_path, timeline, style(), tmp_path / "work")
    inputs = module.assembly_inputs(item(), style().data)
    monkeypatch.setattr(module, "WORD_HOLD_S", 1.3)
    assert module.assembly_inputs(item(), style().data) != inputs


def test_opening_and_mid_video_tokens_follow_brand() -> None:
    data = resolve_style_data({"schema_version": 1, "brand": {
        "titles_font": "Gelasio", "cards_font": "Inter", "accent_color": "#123456"}})
    opening = letters().tokens_for(data, 29)
    middle = letters().tokens_for(data, 30)
    assert (opening["font"], middle["font"]) == ("Gelasio", "Inter")
    assert opening["accent_color"] == middle["accent_color"] == "#123456"


def test_bleep_masks_plate_before_fit_and_records_drawn_lines(tmp_path: Path, monkeypatch) -> None:
    from yap_studio.stages import captions
    from yap_studio.stages.captions import LayerAssets
    from yap_studio.stages.timeline import ResolvedTimeline

    module = letters()
    plans = []

    def render(plan, work, out, **kwargs):
        plans.append(plan)
        out.write_bytes(b"alpha movie")

    monkeypatch.setattr(captions, "render_layer_plan", render)
    monkeypatch.setattr(module, "_assert_layer", lambda movie, frames: None)
    monkeypatch.setattr(module, "plate_assets", lambda font: LayerAssets(
        "plate", b'{"version":"1"}', "font", (("font.woff2", b"font"),)))
    selected = style()
    element = item(text="Fuck")
    timeline = ResolvedTimeline((element,), (), (), 90, 0)
    before = module.assembly_inputs(element, selected.data)
    off, = module.render_plates(tmp_path, timeline, selected, tmp_path / "work")
    assert "display_text" not in off and json.loads(plans[-1].variables["lines"]) == ["FUCK"]
    selected.data["music"]["bleep_swears"] = True
    on, = module.render_plates(tmp_path, timeline, selected, tmp_path / "work")
    assert on["display_text"] == json.loads(plans[-1].variables["lines"]) == ["F***"]
    assert on["kind"] == "insert" and on["path"] != off["path"]
    assert module.assembly_inputs(element, selected.data) != before
    selected.data["music"]["bleep_words_remove"] = ["fuck"]
    removed, = module.render_plates(tmp_path, timeline, selected, tmp_path / "work")
    assert removed["display_text"] == ["FUCK"]
    selected.data["music"]["bleep_swears"] = False
    assert module.assembly_inputs(element, selected.data) == before
    again, = module.render_plates(tmp_path, timeline, selected, tmp_path / "work")
    assert again == off


def test_graphics_title_plate_skips_face_placement_and_marks_receipt(
    layer_project: Path, monkeypatch,
) -> None:
    from test_stage_captions import set_elements

    from yap_studio.project.status import ProjectContext
    from yap_studio.stages import graphics

    selected = json.loads((layer_project / "style.json").read_bytes())
    selected["hook"]["style"] = "video_letters"
    (layer_project / "style.json").write_text(json.dumps(selected))
    set_elements(layer_project, {"id": "agent-title", "kind": "title", "start": {"time": 0},
                                 "lines": ["Small", "steps"]})
    monkeypatch.setattr(graphics, "place_title", lambda *args: pytest.fail("Plate moved"))
    snapshot = graphics.graphics_snapshot(ProjectContext(layer_project))
    placement = snapshot.titles[0].placement()
    assert placement["kind"] == "title"
    assert placement["letters"] is True
    assert (placement["x"], placement["y"], placement["width"], placement["height"]) == (
        0, 0, 1080, 1920)
    assert snapshot.titles[0].plan.template == "letters"
    assert "word_times" not in snapshot.titles[0].plan.variables
    assert set(snapshot.inputs) == {"title:agent-title", "faces"}
    assert snapshot.inputs["title:agent-title"] == snapshot.titles[0].digest
    restored = graphics.graphics_snapshot(ProjectContext(layer_project), placements=[placement])
    assert restored.titles[0].placement() == placement
    assert restored.inputs == snapshot.inputs


def test_letters_title_placement_keeps_the_timeline_kind() -> None:
    plan = title_layer_plan("Small", "steps", style(), frames=48)
    placement = TitleSnapshot("agent-title", "behind", plan, "a" * 64, {}).placement()
    assert plan.template == "letters"
    assert placement["kind"] == "title"
    assert placement["letters"] is True


def test_letters_title_receipt_resize_needs_no_word_times(layer_project: Path) -> None:
    from test_stage_captions import set_elements

    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.graphics import graphics_snapshot

    selected = json.loads((layer_project / "style.json").read_bytes())
    selected["hook"]["style"] = "video_letters"
    (layer_project / "style.json").write_text(json.dumps(selected))
    set_elements(layer_project, {"id": "agent-title", "kind": "title", "start": {"time": 0},
                                 "lines": ["Small", "steps"]})
    first = graphics_snapshot(ProjectContext(layer_project)).titles[0]
    placement = first.placement()
    placement["clearance"]["size_px"] = first.plan.variables["size_px"] - 10
    restored = graphics_snapshot(ProjectContext(layer_project), placements=[placement]).titles[0]
    assert restored.plan.variables == first.plan.variables
    assert "word_times" not in restored.plan.variables


@pytest.mark.parametrize(("plain", "accent"), [
    ("many words in a gallery preview", ""), ("", "W" * 80), ("", ""),
    ("words", "dog park"),
])
def test_gallery_preview_never_raises(plain, accent) -> None:
    letters()
    plan = title_layer_plan(plain, accent, style())
    assert plan.template == "letters" and (plan.width, plan.height) == (1080, 1920)
    assert plan.frames == 48 and len(json.loads(plan.variables["lines"])) in {1, 2}


def test_explicit_title_refuses_fit_and_uses_opening_tokens() -> None:
    letters()
    with pytest.raises(ValidationError, match="150 px"):
        title_layer_plan("", "", style(), lines=["W" * 40])
    plan = title_layer_plan("", "", style(), lines=["Dog", "park"], frames=40)
    assert plan.variables["hold_frames"] == 30
    assert plan.variables["color"] == style().data["hook"]["accent_color"]


def test_title_default_length_and_spoken_checks() -> None:
    title = replace(item(), id="agent-title", kind="title", lane="agent",
                    payload={"lines": ["Dog", "park"]})
    assert not findings(title)
    assert "letters_length" in {f.rule for f in findings(replace(title, end_frame=60))}


def test_existing_title_receipt_bytes_stay_unchanged() -> None:
    letters()
    expected = {
        "paper_title": "d2fb5985fa43cfb886662eb05d03df98c089c1b3e45af798f22be6d49fe516f9",
        "quiet_serif": "92bc1f6470729ac49b30438989b698ce99d49d4744d726e25edcb5dbc4b63619",
        "bold_band": "8fb80dc0e0b688effd5d9b2ddcc7a4debe6f444589c3f420405e7ad971ca81b8",
        "calm_card": "781dadcfb670d85a3aecc9aa037162362849eb4fc0429b0c2e4771fcc6b20706",
        "lockup": "c134385f4717c9bc09355d07cc229e2bd5ddd85be5b46d5390b9619ef29169a1",
        "write_on": "ff72c1f073b302e5214396220d2efd70059a51046196f50592f728b023a61a52",
        "headline_slam": "5234e4853721a39407dec3a39ddc2f050627baf33f5e86876abd835ba74f5bcf",
        "shard_reveal": "12d34d252c9b0f4e725e28e24d37cdfc4f7c6fbb882c9cd651cc86d22f3c9cca",
        "textured": "ff72c1f073b302e5214396220d2efd70059a51046196f50592f728b023a61a52",
        "shimmer": "ff72c1f073b302e5214396220d2efd70059a51046196f50592f728b023a61a52",
        "stagger": "3d26bea8321acdad8ab97b15e35e3744093683bc4b2d34dbf0df66b7c74b3737",
        "morph": "ff72c1f073b302e5214396220d2efd70059a51046196f50592f728b023a61a52",
        "typewriter": "4b1a4a7c98c3ef3464b106b1070a037a766cfda41d9105db82634bc1983f0144",
    }
    for look, digest in expected.items():
        data = resolve_style_data({"schema_version": 1, "name": "Synthetic title",
                                   "hook": {"style": look}})
        plan = title_layer_plan("Make it", "clear", ResolvedStyle(data["name"], "test", data),
                                frames=48)
        assert projection_sha256(TitleSnapshot(
            "graphics-title", "behind", plan, "a" * 64, {}).placement()) == digest
