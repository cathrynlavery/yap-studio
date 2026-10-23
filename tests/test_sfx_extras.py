"""Opt-in and requested ElevenLabs cues use cached audio only."""

from __future__ import annotations

import json
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import synth
from test_sfx import _calls, _item
from test_sfx import fake as fake

from yap_studio import cards, cli, elevenlabs, ffmpeg, receipt, sfx
from yap_studio.project.model import ProjectFileError, parse_document
from yap_studio.sfx.place import Cue, candidates, space
from yap_studio.stages.assemble_audio import decode, render_audio
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline
from yap_studio.style import StyleError, resolve_style_data

RECIPES = [
    ("typing", 2.0, "typing", 2,
     "Soft, clean laptop keyboard typing: a short, steady run of light key presses, "
     "close and dry, no music, quiet enough to sit under calm speech."),
    ("message-ping", 1.0, "message", 3,
     "A soft, short, friendly message notification ping, clean and modern, no music, "
     "quiet enough to sit under calm speech."),
    ("camera-shutter", 1.0, "photo", 2,
     "A single crisp camera shutter click, close and clean, no music."),
    ("applause", 3.0, "request", 9,
     "A small audience applauding warmly for a few seconds, natural room sound, "
     "no cheering, no whistles, no music."),
    ("laugh", 2.5, "request", 9,
     "A small group of people laughing warmly together for a moment, "
     "natural room sound, no music."),
    ("cheer", 2.5, "request", 9,
     "A small crowd cheering and whooping happily for a moment, natural room sound, no music."),
    ("record-scratch", 1.2, "request", 9,
     "A short vinyl record scratch, a quick comedic stop, no music after it."),
]


def test_exact_extras_recipes_and_missing_listing() -> None:
    kit = sfx.load_kit("extras")
    assert kit["schema"] == sfx.SCHEMA
    assert kit["description"] == "More ElevenLabs sounds you turn on, or ask for"
    assert kit["sounds"] == [dict(name=name, duration_s=duration, rule=rule, priority=priority,
                                  prompt=prompt, gain_db=0, prompt_influence=0.5)
                             for name, duration, rule, priority, prompt in RECIPES]
    assert all(sound["state"] == "missing" for sound in sfx.list_kit("extras")["sounds"])


def test_extras_cli_only_overlay_and_peak_from_decoded_cache(
    fake: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["sfx", "kit", "--kit", "extras", "--only", "typing", "--yes",
                     "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["generated"] == ["typing"] and len(_calls(fake)) == 1
    sound = listing["sounds"][0]
    audio = decode(Path(sound["cache_path"]), [], channels=2)
    assert sound["peak_s"] == np.argmax(np.max(np.abs(audio), axis=1)) / 48000
    overlay = sfx.sfx_cache_root() / "kits/extras.json"
    overlay.parent.mkdir(parents=True)
    overlay.write_text(json.dumps({**sfx.load_kit("extras"), "sounds": [
        {**sfx.load_kit("extras")["sounds"][0], "gain_db": -2}]}))
    assert sfx.list_kit("extras")["sounds"][0]["state"] == "stale"
    assert cli.main(["sfx", "kit", "--kit", "extras", "--yes", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["generated"]) == 6
    assert len(_calls(fake)) == 7  # The changed gain reuses the same paid request.


@pytest.mark.parametrize("extras", [["typing", "typing"], ["applause"], ["unknown"]])
def test_style_rejects_invalid_extras(extras: list[str]) -> None:
    with pytest.raises(StyleError):
        resolve_style_data({"schema_version": 1, "music": {"sound_extras": extras}})


def test_style_defaults_and_auto_switches() -> None:
    music = resolve_style_data({"schema_version": 1})["music"]
    assert music["sound_extras"] == [] and music["sound_effects_fallback"] == "made"
    assert resolve_style_data({"schema_version": 1, "music": {
        "sound_extras": ["typing", "message-ping", "camera-shutter"]}})["music"][
            "sound_extras"] == ["typing", "message-ping", "camera-shutter"]


def test_extras_cli_reuses_existing_request_audio_without_generated_index(
    fake: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    typing = sfx.list_kit("extras")["sounds"][0]
    cached = Path(typing["cache_path"])
    cached.parent.mkdir(parents=True)
    source = tmp_path / "cached.wav"
    synth.float_audio(source, np.full((4800, 2), .1, dtype=np.float32))
    cached.write_bytes(source.read_bytes())
    original_hash = receipt.sha256_file(cached)
    assert not (sfx.sfx_cache_root() / "generated.json").exists()
    assert cli.main(["sfx", "kit", "--kit", "extras", "--yes", "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert len(_calls(fake)) == 6
    assert all(typing["prompt"] not in call for call in _calls(fake))
    assert listing["generated"] == [recipe[0] for recipe in RECIPES[1:]]
    assert all(sound["state"] == "fresh" for sound in listing["sounds"])
    assert receipt.sha256_file(cached) == original_hash


@pytest.mark.parametrize("indexed", [False, True])
def test_extras_regenerates_cache_with_stream_but_no_samples(
    fake: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    indexed: bool,
) -> None:
    sound = sfx.list_kit("extras")["sounds"][0]
    target = Path(sound["cache_path"])
    target.parent.mkdir(parents=True)
    with wave.open(str(target), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(48000)
    assert ffmpeg.probe(target, count_frames=False).has_audio
    if indexed:
        sfx._write_json(sfx.sfx_cache_root() / "generated.json", {
            "extras/typing": sound["recipe_sha256"]})
    assert sfx.list_kit("extras")["sounds"][0]["state"] == "missing"
    args = ["sfx", "kit", "--kit", "extras", "--only", "typing", "--json"]
    assert cli.main(args) == 2
    capsys.readouterr()
    assert _calls(fake) == []
    pending = sfx.sfx_cache_root() / "pending.json"
    sfx._write_json(pending, {"state": "unreconciled"})
    assert cli.main([*args, "--yes"]) == 2
    capsys.readouterr()
    assert _calls(fake) == []
    pending.unlink()
    generate_sound = elevenlabs.generate_sound

    def guarded_generate(*args, **kwargs):
        assert json.loads(pending.read_text())["state"] == "pending"
        return generate_sound(*args, **kwargs)

    monkeypatch.setattr(elevenlabs, "generate_sound", guarded_generate)
    assert cli.main([*args, "--yes"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(_calls(fake)) == 1 and result["generated"] == ["typing"]
    assert result["sounds"][0]["state"] == "fresh"
    assert len(decode(target, [], channels=2)) > 0
    assert not pending.exists()
    assert cli.main([*args, "--yes"]) == 0
    assert json.loads(capsys.readouterr().out)["generated"] == []
    assert len(_calls(fake)) == 1


@pytest.mark.parametrize(("kind", "slots", "name"), [
    ("prompt_bar", {"request": "Hello world"}, "typing"),
    ("prompt_bar", {"request": "Hello world", "word_cues": [0.9, 1.4]}, "typing"),
    ("terminal", {"command": "echo hello", "cadence": "uniform", "type_at": 0.7}, "typing"),
    ("terminal", {"command": "echo hello", "word_cues": [0.9, 1.4]}, "typing"),
    ("comment_cta", {"keyword": "Hello"}, "typing"),
    ("comment_cta", {"keyword": "Hello", "cues": [1.7]}, "message-ping"),
    ("comment_drop", {"text": "Hi", "drop_at": 0.35}, "message-ping"),
    ("image_card", {}, "camera-shutter"),
    ("image_compare", {}, "camera-shutter"),
])
def test_auto_extras_opt_in_and_exact_picture_moment(kind: str, slots: dict, name: str) -> None:
    item = _item(kind, slots, end=210)
    timeline = ResolvedTimeline((item,), (), (), 240, 45)
    sounds = [{**sound, "state": "fresh", "peak_s": 0.23}
              for sound in sfx.load_kit("extras")["sounds"]]
    assert candidates(timeline, sounds) == []
    selected = candidates(timeline, sounds, sound_extras=[name])
    assert len(selected) == 1 and selected[0].sound == name
    if name == "typing" or kind == "comment_cta":
        rules = cards.load_template(kind)["slots"]
        values = {key: slots.get(key, rule.get("default")) for key, rule in rules.items()}
        if kind == "terminal":
            schedule = cards._terminal_schedule(values["command"], values["output"],
                values["cadence"], values["type_at"], values["word_cues"], values["cues"],
                (item.end_frame - item.start_frame - 1) / 30)
        else:
            schedule = cards._typing_schedule(values["request" if kind == "prompt_bar" else
                "keyword"], values["cadence"], values["type_at"], values["word_cues"],
                values["cues"], (item.end_frame - item.start_frame - 1) / 30)
        moment = (next(at for at, count in schedule["rows"] if count > 0)
                  if name == "typing" else schedule["confirm_at"])
    else:
        moment = slots["drop_at"] + 0.6 if kind == "comment_drop" else 0
    assert selected[0].time_s == pytest.approx(1 + moment)
    assert selected[0].peak_s == (0 if name == "typing" else 0.23)
    if name == "typing":
        assert selected[0].end_s == pytest.approx(min(
            item.end_s, max(1 + schedule["type_end"], selected[0].time_s + .35)))
    assert candidates(timeline, [{**sound, "state": "missing"} for sound in sounds],
                      sound_extras=[name]) == []


def test_enabled_extra_replaces_same_card_cue_and_requests_cannot_be_spaced_out() -> None:
    existing = Cue("old", "card:in", "card", 1, 2, "card_in")
    extra = Cue("camera-shutter", "photo", "card", 1, 2, "photo")
    kept, dropped = space([extra, replace(existing, priority=4)], 2.5)
    assert kept == [extra]
    assert dropped[0]["reason"] == "enabled_extra"
    assert dropped[0]["kept_cue"]["sound"] == "camera-shutter"
    # Other cards and nearby moments retain the usual priority rules.
    assert space([extra, replace(existing, element_id="other", priority=4)], 2.5)[0] == [
        replace(existing, element_id="other", priority=4)]
    assert space([extra, replace(existing, time_s=1.1, priority=4)], 2.5)[0] == [
        replace(existing, time_s=1.1, priority=4)]
    requests = [Cue("laugh", "request", "a", 1, 1, "request"),
                Cue("cheer", "request", "b", 1.1, 1, "request")]
    kept, dropped = space([existing, *requests], 2.5)
    assert kept == requests and dropped[0]["cue"]["sound"] == "old"


@pytest.mark.parametrize(("kind", "slots"), [
    ("comment_drop", {"drop_at": .3}), ("comment_cta", {"keyword": "Hello"}),
])
def test_message_ping_replaces_soft_chime_on_same_comment(kind: str, slots: dict) -> None:
    timeline = ResolvedTimeline((_item(kind, slots),), (), (), 300, 45)
    sounds = [*sfx.load_kit("soft")["sounds"], *sfx.load_kit("extras")["sounds"]]
    cues = candidates(timeline, sounds, sound_extras=["message-ping"])
    ping = next(cue for cue in cues if cue.sound == "message-ping")
    kept, dropped = space(cues, 2.5)
    assert ping in kept
    assert any(item["cue"]["sound"] == "chime" and item["cue"]["time_s"] == ping.time_s
               and item["reason"] == "enabled_extra" for item in dropped)


def test_request_rule_alone_excludes_automatic_placement() -> None:
    timeline = ResolvedTimeline((_item("image_card", {}),), (), (), 150, 45)
    recipe = sfx.load_kit("extras")["sounds"][3]
    assert candidates(timeline, [recipe]) == []
    # A creator's recipe can reuse a name with a different rule.
    assert candidates(timeline, [{**recipe, "rule": "photo"}],
                      sound_extras=["applause"])[0].sound == "applause"


@pytest.mark.parametrize("name", [item[0] for item in RECIPES] +
                         ["whoosh", "click", "tick-roll", "chime"])
def test_sound_element_contract(name: str) -> None:
    item = {"id": "agent-sound", "kind": "sound", "lane": "agent", "start": {
        "word": "t1:3"}, "sound": name, "gain_db": -2, "align": "peak"}
    document = parse_document("timeline-fragment", {
        "schema": "yap-studio/timeline@1", "elements": [item]})
    assert document.elements[0].end is None and document.elements[0].fields["sound"] == name


@pytest.mark.parametrize("changes", [{"sound": "unknown"}, {"gain_db": -13}, {"gain_db": 7},
    {"hold_s": 1}, {"end": {"time": 2}}, {"align": "bad"}, {"lane": "broll"}])
def test_invalid_sound_element(changes: dict) -> None:
    with pytest.raises(ProjectFileError):
        parse_document("timeline-fragment", {"schema": "yap-studio/timeline@1", "elements": [
            {"id": "agent-sound", "kind": "sound", "lane": "agent", "start": {"time": 1},
             "sound": "applause", **changes}]})


@pytest.mark.parametrize("align", ["start", "peak"])
@pytest.mark.parametrize("gain_db", [-12, 0, 6])
@pytest.mark.parametrize("name", ["applause", "whoosh"])
def test_requested_sound_peak_level_and_silent_hold(tmp_path: Path, align: str,
                                                  gain_db: float, name: str) -> None:
    rate = 48000
    t = np.arange(rate * 4) / rate
    synth.float_audio(tmp_path / "cut/voice.wav", (.2 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32))
    sound_path = tmp_path / "sound.wav"
    audio = np.zeros((rate, 2), dtype=np.float32)
    audio[rate // 4] = 0.8
    synth.float_audio(sound_path, audio)
    recipe = next(item for item in sfx.load_kit("extras" if name == "applause" else "soft")[
        "sounds"] if item["name"] == name)
    sound = {**recipe, **({"peak_s": .25} if name == "applause" else {}), "state": "fresh",
             "cache_path": str(sound_path), "audio_sha256": receipt.sha256_file(sound_path),
             "recipe_sha256": "0" * 64}
    item = ResolvedElement("agent-sound", "sound", "agent", 60, 61, "front", {
        "sound": name, "align": align, "gain_db": gain_db})
    tail = replace(item, id="agent-tail", start_frame=119, end_frame=120)
    timeline = ResolvedTimeline((item, tail), (), (), 120, 45)
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=tmp_path / "work",
        timeline=timeline, policy={"mode": "preserve"}, music={},
        sfx={"kit": "extras", "ready": True, "sounds": [sound]})
    cue = result.sfx["cues"][0]
    assert cue["time_s"] == 2 - (.25 if align == "peak" else 0)
    assert cue["peak_at_s"] == 2 + (.25 if align == "start" else 0)
    assert cue["relative_db"] == pytest.approx(min(-10 + gain_db, -8))
    stem = np.fromfile(tmp_path / "work/sfx.f32le", dtype="<f4").reshape(-1, 2)
    assert not stem[4 * rate:].any()


@pytest.mark.parametrize(("kit", "name"), [("soft", "whoosh"), ("extras", "applause")])
def test_requested_peak_is_measured_after_content_end_trim(
    tmp_path: Path, kit: str, name: str,
) -> None:
    rate = 48000
    t = np.arange(rate * 4) / rate
    synth.float_audio(tmp_path / "cut/voice.wav", (.2 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32))
    audio = np.zeros((rate, 2), dtype=np.float32)
    audio[rate // 100] = .4
    audio[3 * rate // 4] = .8  # The source peak falls beyond the content end.
    source = synth.float_audio(tmp_path / "sound.wav", audio)
    sound = next(item for item in sfx.load_kit(kit)["sounds"] if item["name"] == name)
    sound = {**sound, "state": "fresh", "cache_path": str(source), "peak_s": .75,
             "audio_sha256": receipt.sha256_file(source), "recipe_sha256": "0" * 64}
    item = ResolvedElement("agent-sound", "sound", "agent", 105, 106, "front", {
        "sound": name})
    timeline = ResolvedTimeline((item,), (), (), 120, 45)
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=tmp_path / "work",
        timeline=timeline, policy={"mode": "preserve"}, music={},
        sfx={"kit": kit, "ready": True, "sounds": [sound]})
    stem = np.fromfile(tmp_path / "work/sfx.f32le", dtype="<f4").reshape(-1, 2)
    peak_at = int(np.argmax(np.max(np.abs(stem), axis=1))) / rate
    assert result.sfx["cues"][0]["peak_at_s"] == pytest.approx(peak_at)
    assert peak_at == pytest.approx(3.51)
    assert not stem[4 * rate:].any()


@pytest.mark.parametrize(("kind", "slots", "name", "moment"), [
    ("comment_drop", {"drop_at": 0.3}, "message-ping", 1.9),
    ("image_compare", {}, "camera-shutter", 1.0),
])
def test_only_ready_extra_mixes_with_missing_base_kit(
    tmp_path: Path, kind: str, slots: dict, name: str, moment: float,
) -> None:
    t = np.arange(48000 * 5) / 48000
    synth.float_audio(tmp_path / "cut/voice.wav", (.2 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32))
    sound_path = tmp_path / "sound.wav"
    audio = np.zeros((48000, 2), dtype=np.float32)
    audio[12000] = .8
    synth.float_audio(sound_path, audio)
    sound = next(item for item in sfx.load_kit("extras")["sounds"] if item["name"] == name)
    sound = {**sound, "peak_s": .25, "state": "fresh", "cache_path": str(sound_path),
             "recipe_sha256": "0" * 64, "audio_sha256": receipt.sha256_file(sound_path)}
    missing = {**sound, "name": "laugh", "rule": "request", "state": "missing"}
    timeline = ResolvedTimeline((_item(kind, slots, end=120),), (), (), 150, 45)
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=tmp_path / "work",
        timeline=timeline, policy={"mode": "preserve"}, music={},
        sfx={"kit": "soft", "ready": False, "sounds": [], "sound_extras": [name],
             "additional_sounds": [sound, missing]})
    assert len(result.sfx["cues"]) == 1
    cue = result.sfx["cues"][0]
    assert cue["time_s"] == pytest.approx(moment - .25)
    assert cue["peak_at_s"] == pytest.approx(moment)
    assert cue["relative_db"] == pytest.approx(-10)


@pytest.mark.parametrize("kind", ["prompt_bar", "terminal", "comment_cta"])
@pytest.mark.parametrize("sound_duration", [.1, 2.0])
def test_typing_starts_with_text_and_trims_only_at_typing_end(
    tmp_path: Path, kind: str, sound_duration: float,
) -> None:
    rate = 48000
    t = np.arange(rate * 5) / rate
    synth.float_audio(tmp_path / "cut/voice.wav", (.2 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32))
    # A loudest point late in the file must not pull typing before the text.
    audio = np.full((round(rate * sound_duration), 2), .1, dtype=np.float32)
    audio[-1] = .8
    source = tmp_path / "typing.wav"
    synth.float_audio(source, audio)
    sound = {**sfx.load_kit("extras")["sounds"][0], "state": "fresh",
             "cache_path": str(source), "peak_s": sound_duration - 1 / rate,
             "recipe_sha256": "0" * 64, "audio_sha256": receipt.sha256_file(source)}
    text_slot = {"prompt_bar": "request", "terminal": "command", "comment_cta": "keyword"}[
        kind]
    item = _item(kind, {text_slot: "Hello world", "word_cues": [.8, 1.4]}, end=150)
    timeline = ResolvedTimeline((item,), (), (), 150, 45)
    cue = candidates(timeline, [sound], sound_extras=["typing"])[0]
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=tmp_path / "work",
        timeline=timeline, policy={"mode": "preserve"}, music={},
        sfx={"kit": "soft", "ready": False, "sounds": [], "sound_extras": ["typing"],
             "additional_sounds": [sound]})
    record = result.sfx["cues"][0]
    assert record["time_s"] == pytest.approx(1.8)
    assert record["peak_s"] == 0
    assert record["relative_db"] == pytest.approx(-10)
    stem = np.fromfile(tmp_path / "work/sfx.f32le", dtype="<f4").reshape(-1, 2)
    start = round(1.8 * rate)
    assert not stem[:start].any() and stem[start].any()
    if sound_duration == 2:
        end = round(cue.end_s * rate)
        fade = round(.06 * rate)
        assert record["trim"] == {"duration_s": (end - start) / rate, "fade_out_s": .06}
        assert not stem[end:].any()
        assert np.allclose(stem[end - fade:end, 0] / stem[start, 0],
                           np.linspace(1, 0, fade), atol=1e-6)
        assert start <= round(record["peak_at_s"] * rate) < end
    else:
        assert "trim" not in record
        assert not stem[start + len(audio):].any()
        assert stem[start + len(audio) - 1, 0] > stem[start, 0]


@pytest.mark.parametrize("kind", ["terminal", "prompt_bar", "comment_cta"])
@pytest.mark.parametrize("sound_duration", [.1, 2.0])
@pytest.mark.parametrize("short_card", [False, True])
def test_one_chunk_typing_has_audible_minimum_and_stays_on_card(
    tmp_path: Path, kind: str, sound_duration: float, short_card: bool,
) -> None:
    rate = 48000
    t = np.arange(rate * 5) / rate
    synth.float_audio(tmp_path / "cut/voice.wav", (.2 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32))
    audio = np.full((round(sound_duration * rate), 2), .1, dtype=np.float32)
    source = synth.float_audio(tmp_path / "typing.wav", audio)
    sound = {**sfx.load_kit("extras")["sounds"][0], "state": "fresh", "peak_s": 0,
             "cache_path": str(source), "recipe_sha256": "0" * 64,
             "audio_sha256": receipt.sha256_file(source)}
    text_slot = {"terminal": "command", "prompt_bar": "request", "comment_cta": "keyword"}[
        kind]
    card_end = (84 if kind == "terminal" else 96) if short_card else 150
    item = _item(kind, {text_slot: "ls"}, end=card_end,
                 override={"shift_s": 1} if short_card else None)
    timeline = ResolvedTimeline((item,), (), (), 150, 45)
    cue = candidates(timeline, [sound], sound_extras=["typing"])[0]
    expected_end = min(cue.time_s + .35, item.end_s)
    assert cue.end_s == pytest.approx(expected_end)
    if short_card and kind == "terminal":
        assert expected_end < cue.time_s + .35
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=tmp_path / "work",
        timeline=timeline, policy={"mode": "preserve"}, music={},
        sfx={"kit": "extras", "ready": True, "sounds": [sound], "sound_extras": ["typing"]})
    record = result.sfx["cues"][0]
    stem = np.fromfile(tmp_path / "work/sfx.f32le", dtype="<f4").reshape(-1, 2)
    start = round(cue.time_s * rate)
    end = min(start + len(audio), round(expected_end * rate))
    assert end > start and stem[start:end].any()
    assert not stem[:start].any() and not stem[end:].any()
    fade = min(end - start, round(.06 * rate))
    assert np.allclose(stem[end - fade:end, 0] / stem[start, 0],
                       np.linspace(1, 0, fade), atol=1e-6)
    assert record["relative_db"] == pytest.approx(-10)
    if end - start < len(audio):
        assert record["trim"] == {"duration_s": (end - start) / rate,
                                  "fade_out_s": fade / rate}
    else:
        assert "trim" not in record
        assert record["fade_out_s"] == fade / rate


@pytest.mark.parametrize(("kind", "scenario"), [
    (kind, scenario) for kind in ["terminal", "prompt_bar", "comment_cta"]
    for scenario in ["minus_one", "overlaps_start", "inside_negative", "inside_positive"]
] + [("terminal", scenario) for scenario in [
    "touches_start", "overlaps_end", "touches_end", "after_card"]])
def test_shifted_typing_is_intersected_with_card_and_empty_cues_are_receipted(
    tmp_path: Path, kind: str, scenario: str,
) -> None:
    rate = 48000
    t = np.arange(rate * 8) / rate
    synth.float_audio(tmp_path / "cut/voice.wav", (.2 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32))
    source = synth.float_audio(tmp_path / "typing.wav",
                               np.full((rate * 2, 2), .1, dtype=np.float32))
    sound = {**sfx.load_kit("extras")["sounds"][0], "state": "fresh", "peak_s": 0,
             "cache_path": str(source), "recipe_sha256": "0" * 64,
             "audio_sha256": receipt.sha256_file(source)}
    text_slot = {"terminal": "command", "prompt_bar": "request", "comment_cta": "keyword"}[
        kind]
    slots = {text_slot: "ls"}
    if scenario in {"overlaps_end", "touches_end", "after_card"}:
        slots["type_at"] = 3.1
    item = _item(kind, slots, start=60, end=180)
    timeline = ResolvedTimeline((item,), (), (), 240, 45)
    original = candidates(timeline, [sound], sound_extras=["typing"])[0]
    shift = {
        "minus_one": -1,
        "overlaps_start": item.start_s - original.time_s - .15,
        "inside_negative": -.25,
        "inside_positive": 1,
        "touches_start": item.start_s - original.end_s,
        "overlaps_end": item.end_s - original.time_s - .15,
        "touches_end": item.end_s - original.time_s,
        "after_card": item.end_s - original.time_s + .1,
    }[scenario]
    assert -1 <= shift <= 1
    shifted = replace(item, payload={**item.payload, "sfx": {"shift_s": shift}})
    timeline = replace(timeline, elements=(shifted,))
    cues = candidates(timeline, [sound], sound_extras=["typing"])
    intersection_start = max(item.start_s, original.time_s + shift)
    intersection_end = min(item.end_s, original.end_s + shift)
    empty = intersection_end <= intersection_start + 1e-9
    if empty:
        kept, dropped = space(cues, 2.5)
        assert kept == [] and len(dropped) == 1
        assert dropped[0]["reason"] == "outside_card"
        assert dropped[0]["cue"]["time_s"] == pytest.approx(original.time_s + shift)
        assert dropped[0]["cue"]["sound"] == "typing"
        assert dropped[0]["cue"]["element_id"] == item.id
        assert space(cues, 2.5) == (kept, dropped)
        assert space(list(cues), 2.5) == (kept, dropped)
        entrance = Cue("old", "card:in", item.id, item.start_s, 9, "card_in")
        assert space([*cues, entrance], 2.5) == ([entrance], dropped)
    else:
        assert len(cues) == 1
        cue = cues[0]
        expected_end = min(item.end_s, intersection_start + .35)
        assert cue.time_s == pytest.approx(intersection_start)
        assert cue.end_s == pytest.approx(expected_end)
        assert space(cues, 2.5) == ([cue], [])
    # Match assembly's required-sound selection before the mixer sees the extras.
    required = {cue.sound for cue in candidates(timeline, [sound], sound_extras=["typing"])}
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=tmp_path / "work",
        timeline=timeline, policy={"mode": "preserve"}, music={},
        sfx={"kit": "soft", "ready": False, "sounds": [], "sound_extras": ["typing"],
             "additional_sounds": [sound] if "typing" in required else []})
    stem = np.fromfile(tmp_path / "work/sfx.f32le", dtype="<f4").reshape(-1, 2)
    assert not stem[:item.start_frame * 1600].any()
    assert not stem[item.end_frame * 1600:].any()
    if empty:
        assert not stem.any() and result.sfx["cues"] == []
        assert result.sfx["dropped"] == dropped
    else:
        start, end = round(intersection_start * rate), round(expected_end * rate)
        record = result.sfx["cues"][0]
        assert stem[start:end].any() and not stem[:start].any() and not stem[end:].any()
        assert record["time_s"] == pytest.approx(intersection_start, abs=1 / rate)
        assert record["trim"] == {"duration_s": (end - start) / rate, "fade_out_s": .06}
        assert record["relative_db"] == pytest.approx(-10)
        assert np.allclose(stem[end - 2880:end, 0] / stem[start, 0],
                           np.linspace(1, 0, 2880), atol=1e-6)
