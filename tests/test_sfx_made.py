"""Recipe synthesis, peak placement and account-free sound contracts."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import synth

from yap_studio import cli, elevenlabs, ffmpeg, sfx
from yap_studio.project.model import AUDIO_DEFAULTS
from yap_studio.sfx import made
from yap_studio.sfx.place import Cue, candidates, space
from yap_studio.stages.assemble_audio import AudioError, decode, render_audio
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline
from yap_studio.style import resolve_style_data

NAMES = ("whoosh", "swoosh-in", "swoosh-out", "click", "pop", "hit", "riser",
         "tick-roll", "chime")


@pytest.mark.parametrize("name", NAMES)
def test_render_is_repeatable_and_peaks_on_recipe(name: str) -> None:
    recipe = next(sound for sound in made.kit()["sounds"] if sound["name"] == name)
    first = made.render(name).copy()
    made.render.cache_clear()
    second = made.render(name)
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32 and first.shape == (round(recipe["duration_s"] * 48000), 2)
    assert np.isfinite(first).all() and 0 < np.max(np.abs(first)) <= 1
    np.testing.assert_array_equal(first[:, 0], first[:, 1])
    assert abs(made.peak_time(name) - recipe["peak_s"]) <= .010
    assert made.wav_bytes(name) == made.wav_bytes(name)
    with wave.open(io.BytesIO(made.wav_bytes(name))) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (48000, 2, 2)
        assert wav.getnframes() == len(first)
    digest = subprocess.check_output([
        sys.executable, "-c",
        f"from yap_studio.sfx import made; print(made.audio_sha256({name!r}))",
    ], text=True).strip()
    assert digest == made.audio_sha256(name) == hashlib.sha256(first.tobytes()).hexdigest()


def test_listing_and_recipe_identity() -> None:
    listing = made.listing()
    assert listing["kit"] == "made" and listing["label"] == "Made by Yap Studio"
    assert {sound["name"] for sound in listing["sounds"]} == set(NAMES)
    assert {sound["rule"] for sound in listing["sounds"]} == (
        sfx.RULES - {"typing", "message", "photo", "request"})
    for sound in listing["sounds"]:
        assert sound["peak_s"] == made.peak_time(sound["name"])
        assert sound["made"] and sound["state"] == "fresh" and sound["cache_path"] is None
        assert len(sound["recipe_sha256"]) == len(sound["audio_sha256"]) == 64
    original = made.kit()["sounds"][0]
    assert made.recipe_sha256(original) != made.recipe_sha256({**original, "seed": 99})


@pytest.mark.parametrize(("field", "value"), [
    ("duration_s", -1), ("peak_s", 2), ("seed", True), ("rule", "bad"),
    ("gain_db", float("nan")), ("priority", 0), ("layers", []), ("extra", 1),
])
def test_invalid_sound_names_field(field: str, value: object) -> None:
    sound = {**made.kit()["sounds"][0], field: value}
    with pytest.raises(sfx.SfxError, match=field):
        made._check_sound(sound, "sound")


@pytest.mark.parametrize(("field", "value"), [
    ("source", "bad"), ("level", -1), ("from_hz", 24001),
    ("filter", {"type": "bandpass", "from_hz": 100, "to_hz": 500, "q": 0}),
    ("envelope", {"start_s": 0, "attack_s": -1, "hold_s": 0, "release_s": .1}),
    ("repeat", {"count": 0, "every_s": .1, "decay": .9}),
])
def test_invalid_layer_names_field(field: str, value: object) -> None:
    sound = copy.deepcopy(made.kit()["sounds"][0])
    sound["layers"][0][field] = value
    with pytest.raises(sfx.SfxError, match=field):
        made._check_sound(sound, "sound")


def test_reserved_kit_cli_and_no_account_or_cache(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(elevenlabs, "account_state", lambda: pytest.fail("Account queried"))
    root = sfx.sfx_cache_root()
    assert not root.exists()
    assert cli.main(["sfx", "list", "--kit", "made", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["sounds"]) == 9
    assert cli.main(["sfx", "kit", "--kit", "made", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "fresh" and result["generated"] == [] and result["notices"]
    assert cli.main(["sfx", "list", "--kit", "made"]) == 0
    output = capsys.readouterr().out
    assert "transition" in output and "\u2014" not in output
    assert f"peak {made.peak_time('whoosh'):g} s" in output
    assert cli.main(["sfx", "add", "x", "--kit", "made", "--prompt", "x",
                     "--duration", "1", "--rule", "pop", "--json"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "sfx_builtin_kit"
    assert not root.exists()
    with pytest.raises(sfx.SfxError, match="ElevenLabs"):
        sfx.load_kit("made")
    overlay = root / "kits/made.json"
    overlay.parent.mkdir(parents=True)
    overlay.write_text("invalid overlay")
    assert sfx.list_kit("made") == made.listing()


def _element(kind: str, slots: dict | None = None, *, start: int = 60,
             end: int = 180) -> ResolvedElement:
    payload = ({"join_frame": 90} if kind == "transition" else {} if kind == "chapter"
               else {"template": kind, "slots": slots or {}})
    return ResolvedElement("moment", kind if kind in {"transition", "chapter"} else "graphic",
                           "agent", start, end, "front", payload)


@pytest.mark.parametrize(("rule", "element", "event", "moment"), [
    ("transition", _element("transition"), "section", 3),
    ("card_in", _element("checklist", {"items": ["A"]}), "card:in", 2),
    ("card_out", _element("checklist", {"items": ["A"]}), "card:out", 5.7),
    ("pop", _element("flow", {"nodes": ["A", "B"], "cues": [1, 2]}), "flow:1", 3 + 1 / 6),
    ("land", _element("gravity_drop", {"drops": [{"at": .5}]}), "gravity_drop:land", 2.5),
    ("chapter", _element("chapter"), "chapter", 2),
    ("item", _element("checklist", {"items": ["A"]}), "checklist:1", 2.3),
    ("count_start", _element("stat"), "stat:start", 2),
    ("settle", _element("stat"), "stat:final", 3.72),
])
def test_mixed_peak_lands_on_event(
    tmp_path: Path, rule: str, element: ResolvedElement, event: str, moment: float,
) -> None:
    listing = made.listing()
    listing["sounds"] = [sound for sound in listing["sounds"] if sound["rule"] == rule]
    timeline = ResolvedTimeline((element,), (), (), 240, 45)
    planned = [cue for cue in candidates(timeline, listing["sounds"], flow_look="tags")
               if cue.event == event]
    assert len(planned) == 1 and planned[0].time_s == pytest.approx(moment)
    assert planned[0].peak_s == made.peak_time(planned[0].sound)
    _check_mix(tmp_path, timeline, listing, event, moment)


def _check_mix(tmp_path: Path, timeline: ResolvedTimeline, listing: dict,
               event: str, moment: float) -> None:
    rate = 48000
    time = np.arange(timeline.total_frames * 1600) / rate
    synth.float_audio(tmp_path / "cut/voice.wav",
                      (.2 * np.cos(2 * np.pi * 220 * time)).astype(np.float32))
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=tmp_path / "work",
                          timeline=timeline, policy={"mode": "preserve"}, music={},
                          sfx={**listing, "ready": True}, flow_look="tags",
                          sfx_settings={"sfx_min_gap_s": 0})
    record = next(cue for cue in result.sfx["cues"] if cue["event"] == event)
    stem = np.fromfile(tmp_path / "work/sfx.f32le", dtype="<f4").reshape(-1, 2)
    # Inspect the actual mixed stem near this event, away from the independent end cue.
    first, last = max(0, round((moment - .04) * rate)), round((moment + .04) * rate)
    peak = (first + np.argmax(np.abs(stem[first:last, 0]))) / rate
    assert abs(peak - moment) <= .001
    assert abs(record["peak_at_s"] - moment) <= .001
    assert record["peak_s"] == made.peak_time(record["sound"])
    gains = {sound["name"]: sound["gain_db"] for sound in listing["sounds"]}
    assert all(abs(cue["relative_db"] - min(-10 + gains[cue["sound"]], -8)) < .1
               for cue in result.sfx["cues"])
    actual_db = 20 * np.log10(np.max(np.abs(stem[first:last])) / .2)
    assert abs(actual_db - min(-10 + gains[record["sound"]], -8)) < .1
    assert not stem[timeline.total_frames * 1600:].any()


def test_head_and_tail_trim(tmp_path: Path) -> None:
    timeline = ResolvedTimeline((_element("checklist", {"items": ["A"]}, start=0, end=60),),
                                (), (), 60, 45)
    listing = made.listing()
    listing["sounds"] = [sound for sound in listing["sounds"] if sound["rule"] == "card_in"]
    _check_mix(tmp_path, timeline, listing, "card:in", 0)
    late = _element("checklist", {"items": ["A"]}, start=59, end=60)
    timeline = ResolvedTimeline((late,), (), (), 60, 45)
    _check_mix(tmp_path / "tail", timeline, listing, "card:in", 59 / 30)


def test_changed_pcm_refused(tmp_path: Path) -> None:
    listing = made.listing()
    listing["sounds"] = [{**listing["sounds"][1], "audio_sha256": "0" * 64}]
    timeline = ResolvedTimeline((_element("checklist", {"items": ["A"]}),), (), (), 240, 45)
    synth.float_audio(tmp_path / "cut/voice.wav", np.full(240 * 1600, .2, dtype=np.float32))
    with pytest.raises(AudioError, match="changed"):
        render_audio(tmp_path, tmp_path / "out.m4a", work_dir=tmp_path / "work",
                     timeline=timeline, policy={"mode": "preserve"}, music={},
                     sfx={**listing, "ready": True})


@pytest.mark.parametrize("fallback", ["none", "made"])
def test_schema_made_and_fallback(fallback: str) -> None:
    style = resolve_style_data({"schema_version": 1, "music": {
        "sound_effects": "made", "sound_effects_fallback": fallback}})
    assert style["music"]["sound_effects_fallback"] == fallback
    with pytest.raises(ValueError):
        resolve_style_data({"schema_version": 1, "music": {"sound_effects_fallback": "bad"}})


@pytest.mark.parametrize("gain_db", [-4, 12])
def test_gain_and_voice_ceiling(tmp_path: Path, gain_db: float) -> None:
    listing = made.listing()
    listing["sounds"] = [{**sound, "gain_db": gain_db} for sound in listing["sounds"]
                         if sound["rule"] == "item"]
    timeline = ResolvedTimeline((_element("checklist", {"items": ["A"]}),), (), (), 240, 45)
    _check_mix(tmp_path, timeline, listing, "checklist:1", 2.3)


def test_end_sound_finishes_before_cut(tmp_path: Path) -> None:
    listing = made.listing()
    listing["sounds"] = [sound for sound in listing["sounds"] if sound["rule"] == "settle"]
    timeline = ResolvedTimeline((), (), (), 120, 45)
    sound = listing["sounds"][0]
    moment = 4 - sound["duration_s"] - .05 + sound["peak_s"]
    cue, = candidates(timeline, listing["sounds"])
    assert cue.time_s == pytest.approx(moment)
    _check_mix(tmp_path, timeline, listing, "end", moment)
    stem = np.fromfile(tmp_path / "work/sfx.f32le", dtype="<f4").reshape(-1, 2)
    assert not stem[round(3.95 * 48000):].any()


def test_render_does_not_move_audio_to_the_designed_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    original = made.render("whoosh").copy()
    data = made.kit()
    data["sounds"][0]["peak_s"] += .05
    try:
        with monkeypatch.context() as patch:
            patch.setattr(made, "_kit", lambda: data)
            made.render.cache_clear()
            np.testing.assert_array_equal(made.render("whoosh"), original)
    finally:
        made.render.cache_clear()


def test_priorities_favour_card_contents() -> None:
    assert {sound["name"]: sound["priority"] for sound in made.kit()["sounds"]} == {
        "click": 2, "pop": 2, "swoosh-in": 1, "swoosh-out": 1, "riser": 5,
        "chime": 4, "hit": 4, "whoosh": 3, "tick-roll": 2,
    }


def test_short_plain_card_keeps_its_swoosh_pair() -> None:
    item = _element("text_tag", {"text": "A calm card"}, start=60, end=135)
    timeline = ResolvedTimeline((item,), (), (), 450, 45)
    kept, dropped = space(candidates(timeline, made.listing()["sounds"]), 2.5)
    assert [(cue.rule, cue.time_s) for cue in kept if cue.element_id == item.id] == [
        ("card_in", 2), ("card_out", 4.2)]
    assert not [row for row in dropped if row["cue"]["element_id"] == item.id]


def test_checklist_keeps_click_instead_of_both_swooshes() -> None:
    item = _element("checklist", {"items": ["A"]}, start=60, end=135)
    timeline = ResolvedTimeline((item,), (), (), 450, 45)
    kept, dropped = space(candidates(timeline, made.listing()["sounds"]), 2.5)
    click, = [cue for cue in kept if cue.element_id == item.id]
    assert click.sound == "click"
    removed = {row["cue"]["sound"]: row for row in dropped}
    assert removed["swoosh-in"]["reason"] == "minimum_gap"
    assert removed["swoosh-out"]["reason"] == "card_in_dropped"
    assert removed["swoosh-in"]["kept_cue"] == removed["swoosh-out"]["kept_cue"] == click.to_dict()


@pytest.mark.parametrize("out_priority", [1, 3])
def test_long_card_loses_exit_when_another_cards_click_drops_entrance(out_priority: int) -> None:
    plain = replace(_element("text_tag", start=60, end=300), id="plain")
    checklist = replace(_element("checklist", {"items": ["A"]}, start=54, end=129), id="rows")
    timeline = ResolvedTimeline((plain, checklist), (), (), 450, 45)
    cues = [replace(cue, priority=out_priority) if cue.element_id == "plain"
            and cue.rule == "card_out" else cue
            for cue in candidates(timeline, made.listing()["sounds"])]
    kept, dropped = space(cues, 2.5)
    click, = [cue for cue in kept if cue.sound == "click"]
    assert not any(cue.element_id == plain.id for cue in kept)
    removed = {row["cue"]["sound"]: row for row in dropped
               if row["cue"]["element_id"] == plain.id}
    assert removed["swoosh-in"]["reason"] == "minimum_gap"
    assert removed["swoosh-out"]["reason"] == "card_in_dropped"
    assert removed["swoosh-out"]["kept_cue"] == click.to_dict()


@pytest.mark.parametrize("out_priority", [1, 3])
def test_dropped_card_exit_cannot_block_later_cards_pair(out_priority: int) -> None:
    first = replace(_element("text_tag", start=60, end=300), id="first")
    second = replace(_element("text_tag", start=300, end=390), id="second")
    timeline = ResolvedTimeline((first, second), (), (), 420, 45)
    sounds = made.listing()["sounds"]
    chime = next(sound for sound in sounds if sound["name"] == "chime")
    opening = Cue(chime["name"], "opening:settle", "opening", 1.1,
                  chime["priority"], chime["rule"], chime["peak_s"])
    cues = candidates(timeline, [sound for sound in sounds
                                 if sound["rule"] in {"card_in", "card_out"}])
    cues = [replace(cue, priority=out_priority) if cue.element_id == first.id
            and cue.rule == "card_out" else cue for cue in cues]
    kept, dropped = space([opening, *cues], 2.5)
    assert [(cue.element_id, cue.rule) for cue in kept] == [
        ("opening", "settle"), ("second", "card_in"), ("second", "card_out")]
    assert [cue.time_s for cue in kept] == pytest.approx([1.1, 10, 12.7])
    assert {row["cue"]["element_id"] for row in dropped} == {first.id}
    removed = {row["cue"]["sound"]: row for row in dropped}
    assert removed["swoosh-in"]["reason"] == "minimum_gap"
    assert removed["swoosh-out"]["cue"]["time_s"] == pytest.approx(9.7)
    assert removed["swoosh-out"]["reason"] == "card_in_dropped"
    assert removed["swoosh-in"]["kept_cue"] == removed["swoosh-out"]["kept_cue"] == (
        opening.to_dict())


def test_different_cards_swooshes_still_obey_gap() -> None:
    first = replace(_element("text_tag", start=60, end=135), id="first")
    second = replace(_element("text_tag", start=150, end=225), id="second")
    timeline = ResolvedTimeline((first, second), (), (), 450, 45)
    sounds = [sound for sound in made.listing()["sounds"]
              if sound["rule"] in {"card_in", "card_out"}]
    kept, dropped = space(candidates(timeline, sounds), 2.5)
    assert [cue.element_id for cue in kept] == ["first", "first"]
    assert {row["cue"]["element_id"] for row in dropped} == {"second"}
    entrance = next(row for row in dropped if row["cue"]["event"] == "card:in")
    assert entrance["reason"] == "minimum_gap"
    assert entrance["kept_cue"] == kept[1].to_dict()


def test_exit_without_any_entrance_candidate_is_kept() -> None:
    item = _element("text_tag", start=60, end=135)
    timeline = ResolvedTimeline((item,), (), (), 450, 45)
    sounds = [sound for sound in made.listing()["sounds"] if sound["rule"] == "card_out"]
    kept, dropped = space(candidates(timeline, sounds), 2.5)
    assert [cue.rule for cue in kept] == ["card_out"]
    assert not dropped


@pytest.mark.parametrize("mode", ["preserve", "loudnorm"])
@pytest.mark.parametrize("bed", [False, True])
def test_words_stem_is_the_same_mix_without_effects(
    tmp_path: Path, mode: str, bed: bool,
) -> None:
    count, hold = 4 * 48000, 15 * 1600
    time = np.arange(count) / 48000
    voice = (.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    synth.float_audio(tmp_path / "cut/voice.wav", np.column_stack((voice, voice)))
    clip = synth.float_audio(tmp_path / "clip.wav", .03 * np.sin(2 * np.pi * 770 * time))
    clips = [{"id": "insert", "path": clip.name, "start_frame": 30, "frames": 15,
              "gain_db": -6}]
    track = None
    if bed:
        music_pcm = .02 * np.sin(2 * np.pi * 880 * time)
        padded = np.concatenate((music_pcm, np.zeros(hold)))
        track = synth.float_audio(tmp_path / "bed.wav", np.column_stack((padded, padded)))
    music = resolve_style_data({"schema_version": 1})["music"]
    timeline = ResolvedTimeline((_element("transition", start=54, end=66),), (), (), 120, 15)
    listing = made.listing()
    listing["sounds"] = [sound for sound in listing["sounds"] if sound["rule"] == "transition"]
    work = tmp_path / "with"
    result = render_audio(tmp_path, tmp_path / "with.m4a", work_dir=work,
                          timeline=timeline, policy=AUDIO_DEFAULTS if mode == "loudnorm"
                          else {"mode": "preserve"}, music=music, bed=track, clips=clips,
                          sfx={**listing, "ready": True})
    assert result.words_stem is not None and result.words_stem.is_file()
    assert result.sfx["cues"]
    render_audio(tmp_path, tmp_path / "without.m4a", work_dir=tmp_path / "without",
                 timeline=timeline, policy={"mode": "preserve"}, music=music,
                 bed=track, clips=clips)
    reference = tmp_path / "without/mix.wav"
    mixed = work / "mix.wav"
    if mode == "loudnorm":
        normalised = work / f"normalised-{result.normalization['iterations'] - 1}.wav"
        mixed = normalised
        command = next(command for command in result.commands if command[-1] == str(normalised))
        second = command[command.index("-af") + 1]
        expected = tmp_path / "expected.wav"
        ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(reference),
                    "-af", second, "-c:a", "pcm_f32le", str(expected)],
                   timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)
        reference = expected
    stem = decode(result.words_stem, [], channels=2)
    reference_pcm = decode(reference, [], channels=2)
    tolerance = 1e-3
    if bed:
        # ffmpeg's sidechaincompress tail varies with graph frame scheduling.
        np.testing.assert_allclose(stem, reference_pcm, rtol=0, atol=tolerance)
    else:
        np.testing.assert_array_equal(stem, reference_pcm)
    assert len(stem) == count + hold and not stem[count:].any()
    peak = round(result.sfx["cues"][0]["peak_at_s"] * 48000)
    near = slice(max(0, peak - 2400), min(count, peak + 2401))
    with_effects = decode(mixed, [], channels=2)
    assert np.max(np.abs(stem[near] - with_effects[near])) > 10 * tolerance
    assert "words_stem" not in result.to_dict()


@pytest.mark.parametrize("case", ["off", "no_cues", "pilot"])
def test_words_stem_absent_without_final_effects(tmp_path: Path, case: str) -> None:
    time = np.arange(4 * 48000) / 48000
    voice = (.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    synth.float_audio(tmp_path / "cut/voice.wav", np.column_stack((voice, voice)))
    elements = (_element("transition", start=54, end=66),) if case == "pilot" else ()
    timeline = ResolvedTimeline(elements, (), (), 120, 15)
    listing = made.listing()
    listing["sounds"] = [sound for sound in listing["sounds"] if sound["rule"] == "transition"]
    frames = (30, 89) if case == "pilot" else None
    result = render_audio(tmp_path, tmp_path / "out.m4a", work_dir=tmp_path / "work",
                          timeline=timeline, policy={"mode": "preserve"}, music={},
                          frames=frames, sfx={**listing, "ready": True} if case != "off" else None)
    assert result.words_stem is None
    assert not list((tmp_path / "work").glob("no-effects*"))
    if case != "pilot":
        render_audio(tmp_path, tmp_path / "baseline.m4a", work_dir=tmp_path / "baseline",
                     timeline=timeline, policy={"mode": "preserve"}, music={})
        assert (tmp_path / "out.m4a").read_bytes() == (tmp_path / "baseline.m4a").read_bytes()
