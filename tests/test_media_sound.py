"""Find sound events in small synthetic catalog clips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import synth_broll as synth

from yap_studio import cli, ffmpeg
from yap_studio.errors import InputError, ValidationError
from yap_studio.media import add_paths, sound, update_catalog
from yap_studio.media.sound import find_sound
from yap_studio.transcript import Transcript, Word


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


def _clip(path: Path, bursts: tuple[tuple[float, float], ...] = ()) -> Path:
    """Mux low noise and delayed tones with a small test picture."""
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "testsrc2=size=64x64:rate=10:duration=3", "-f", "lavfi", "-i",
        "anoisesrc=color=white:amplitude=0.001:sample_rate=16000:duration=3",
    ]
    for _start, _volume in bursts:
        command.extend([
            "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=16000:duration=0.3",
        ])
    filters = [
        f"[{index}:a]volume={volume},afade=t=out:st=0:d=0.3,"
        f"adelay={round(start * 1000)}[tone{index}]"
        for index, (start, volume) in enumerate(bursts, start=2)
    ]
    if bursts:
        inputs = "[1:a]" + "".join(f"[tone{index}]" for index in range(2, 2 + len(bursts)))
        filters.append(f"{inputs}amix=inputs={len(bursts) + 1}:duration=first:normalize=0[a]")
        command.extend(["-filter_complex", ";".join(filters), "-map", "0:v:0", "-map", "[a]"])
    else:
        command.extend(["-map", "0:v:0", "-map", "1:a:0"])
    command.extend(["-c:v", "mpeg4", "-c:a", "aac", str(path)])
    ffmpeg.run(command, timeout_s=30)
    return path


def _add(source: Path) -> None:
    add_paths([source])


def test_levels_stream_chunk_boundaries_match_whole_array_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.concatenate([
        np.full(1600, 0.5, dtype="<f4"),
        np.linspace(-0.25, 0.25, 1600, dtype="<f4"),
        np.array([0.1, -0.2, 0.3], dtype="<f4"),
    ])
    raw = samples.tobytes()
    chunks = [raw[:3], raw[3:1611], raw[1611:6405], raw[6405:]]
    timeouts: list[float] = []

    def stream(command: list[str], *, timeout_s: float) -> object:
        assert command[0] == "ffmpeg" and timeout_s >= ffmpeg.DEFAULT_TIMEOUT_S
        timeouts.append(timeout_s)
        yield from chunks

    monkeypatch.setattr(sound.ffmpeg, "run", lambda *args, **kwargs: pytest.fail(
        "_levels must stream decoded audio"))
    monkeypatch.setattr(sound.ffmpeg, "run_stream", stream)
    levels, duration = sound._levels(tmp_path / "synthetic.mov")
    windows = samples[:3200].reshape(2, 1600)
    rms = np.sqrt(np.mean(np.square(windows), axis=1, dtype=np.float64))
    tail = np.sqrt(np.mean(np.square(samples[3200:]), dtype=np.float64))
    expected = 20 * np.log10(np.maximum(np.append(rms, tail), 1e-12))
    np.testing.assert_array_equal(levels, expected)
    assert duration == len(samples) / sound.SAMPLE_RATE
    sound._levels(tmp_path / "synthetic.mov", duration_hint_s=4_000)
    assert timeouts[1] == max(ffmpeg.DEFAULT_TIMEOUT_S, 120 + 1.5 * 4_000)


def test_one_burst_finds_peak_and_insert_start(tmp_path: Path) -> None:
    _add(_clip(tmp_path / "one.mp4", ((1.0, 1.0),)))
    result = find_sound("m1")
    assert result["id"] == "m1"
    assert len(result["moments"]) == 1
    moment = result["moments"][0]
    assert abs(moment["peak_s"] - 1.0) <= 0.1
    assert moment["start_s"] <= moment["peak_s"] < moment["end_s"]
    assert moment["peak_dbfs"] > -40
    assert moment["above_floor_db"] >= 12
    assert moment["in_s"] == round(max(0, moment["peak_s"] - 1.0), 2)


def test_two_bursts_are_in_time_order_and_top_keeps_louder(tmp_path: Path) -> None:
    _add(_clip(tmp_path / "two.mp4", ((0.6, 0.25), (2.0, 1.0))))
    moments = find_sound("m1")["moments"]
    assert len(moments) == 2
    assert [moment["peak_s"] for moment in moments] == sorted(
        moment["peak_s"] for moment in moments
    )
    assert abs(moments[0]["peak_s"] - 0.6) <= 0.1
    assert abs(moments[1]["peak_s"] - 2.0) <= 0.1
    assert find_sound("m1", top=1)["moments"] == [moments[1]]


def test_quiet_only_has_no_moments(tmp_path: Path) -> None:
    _add(_clip(tmp_path / "quiet.mp4"))
    assert find_sound("m1")["moments"] == []


def test_nearby_bursts_merge_into_one_moment(tmp_path: Path) -> None:
    _add(_clip(tmp_path / "nearby.mp4", ((0.6, 0.5), (1.0, 1.0))))
    moments = find_sound("m1")["moments"]
    assert len(moments) == 1
    assert moments[0]["start_s"] <= 0.6
    assert moments[0]["end_s"] >= 1.2


@pytest.mark.parametrize("still", [False, True])
def test_no_sound_errors(tmp_path: Path, still: bool) -> None:
    source = synth.png(tmp_path / "still.png") if still else synth.clip(
        tmp_path / "silent.mov", audio=False
    )
    _add(source)
    reason = "no sound: still image" if still else "no sound: clip has no audio track"
    with pytest.raises(InputError, match=reason):
        find_sound("m1")


def test_transcript_groups_words_at_gaps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _add(_clip(tmp_path / "speech.mp4"))
    words_path = tmp_path / "transcript.json"
    Transcript([
        Word("Plunge", 0.2, 0.5), Word("is", 0.5, 0.6),
        Word("done.", 0.7, 0.9), Word("Splash!", 1.7, 2.0),
    ]).save(words_path)

    def set_speech(document: dict) -> None:
        document["items"][0]["speech"] = {
            "status": "words", "engine": "local-test", "model": "tiny", "words": 4,
            "text": "Plunge is done. Splash!", "path": str(words_path),
            "message": None, "at": "2026-10-05T12:00:00Z",
        }

    update_catalog(set_speech)
    result = find_sound("m1")
    assert result["speech"] == [
        {"start_s": 0.2, "end_s": 0.9, "text": "Plunge is done."},
        {"start_s": 1.7, "end_s": 2.0, "text": "Splash!"},
    ]
    assert "speech_note" not in result
    assert cli.main(["media", "sound", "m1"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        '0.20 s to 0.90 s: "Plunge is done."',
        '1.70 s to 2.00 s: "Splash!"',
    ]


def test_no_transcript_has_note(tmp_path: Path) -> None:
    _add(_clip(tmp_path / "without-words.mp4"))
    result = find_sound("m1")
    assert result["speech"] is None
    assert "yap-studio media sheet m1" in result["speech_note"]


def test_changed_catalog_file_is_refused(tmp_path: Path) -> None:
    source = _clip(tmp_path / "changed.mp4")
    _add(source)
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(InputError, match="no catalog location still holds this file"):
        find_sound("m1")


@pytest.mark.parametrize("top", [0, 21, 1.5])
def test_top_bounds(tmp_path: Path, top: int | float) -> None:
    _add(_clip(tmp_path / "top.mp4"))
    with pytest.raises(ValidationError, match="top"):
        find_sound("m1", top=top)


def test_cli_text_json_and_unknown_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _add(_clip(tmp_path / "cli.mp4", ((1.0, 1.0),)))
    assert cli.main(["media", "sound", "m1", "--top", "1"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert "peak" in lines[0] and "dBFS" in lines[0] and "insert in_s" in lines[0]
    assert "yap-studio media sheet m1" in lines[1]

    assert cli.main(["media", "sound", "m1", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["id"] == "m1" and len(result["moments"]) == 1
    assert result["speech"] is None

    assert cli.main(["media", "sound", "m99", "--json"]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"]["code"] == "unknown_media"
