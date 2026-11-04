"""Voice samples assembled from retained speech."""

import copy
import json
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import synth
from test_cut import make_project, time_segment, write_edit, write_json
from voice_helpers import calls

from yap_studio import audio, ffmpeg, receipt
from yap_studio.errors import YapError
from yap_studio.stages.cut import FADE_SAMPLES, _audio_piece, resolve_cut
from yap_studio.voice import consent, create, link, propose, samples
from yap_studio.voice.model import checked_file, validate, write
from yap_studio.voice.samples import _Candidate, _runs


@pytest.fixture(scope="module")
def speech(tmp_path_factory):
    return synth.speech_source(
        tmp_path_factory.mktemp("sample-speech") / "speech.mov",
        ["These are my own recordings of a quiet morning in the garden with bright light."] * 18,
        size="160x90", noise_dbfs=-60,
    )


@pytest.fixture
def short_pieces(tmp_path, speech):
    source, words = speech
    root = make_project(
        tmp_path / "project", [source],
        word_rows=[[(word.text, word.start, word.end) for word in words]],
    )
    project = json.loads((root / "project.json").read_text())
    project["takes"][0]["voiceover"] = True
    write_json(root / "project.json", project)
    end = ffmpeg.probe(source).video_duration_s
    segments = [
        time_segment(f"s{i}", "t1", first, first + 4)
        for i, first in enumerate(range(0, int(end) - 4, 5), 1)
    ]
    write_edit(root, segments)
    return root


def test_samples_join_short_kept_pieces(short_pieces, tmp_path):
    plan = resolve_cut(short_pieces)
    assert len(plan.pieces) > 20
    assert all(float(piece.src_t1 - piece.src_t0) <= 6 for piece in plan.pieces)
    data = samples(short_pieces)
    assert 4 <= len(data["samples"]) <= 6
    assert 45 <= sum(row["duration_s"] for row in data["samples"]) <= 150
    sounds = [_audio_piece(piece, plan) for piece in plan.pieces]
    offsets = np.cumsum([0, *[len(sound) for sound in sounds]])
    joined, _ = audio.splice(
        np.concatenate(sounds), list(zip(offsets[:-1], offsets[1:], strict=True)), FADE_SAMPLES,
    )
    for row in data["samples"]:
        assert 15 <= row["duration_s"] <= 30
        assert len(row["ranges"]) > 1
        assert row["start_s"] == row["ranges"][0][0]
        assert row["end_s"] == row["ranges"][-1][1]
        assert sum(b - a for a, b in row["ranges"]) == pytest.approx(row["duration_s"], abs=0.05)
        first = row["ranges"][0][0]
        index = next(i for i, piece in enumerate(plan.pieces)
                     if float(piece.src_t0) <= first < float(piece.src_t1))
        start = offsets[index] + round((first - float(plan.pieces[index].src_t0)) * 48000)
        count = round(row["duration_s"] * 48000)
        expected = joined[start:start + count]
        raw = tmp_path / "expected.f32le"
        raw.write_bytes(expected.astype("<f4").tobytes())
        converted = ffmpeg.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", "48000", "-ac", "1",
            "-i", str(raw), "-ar", "44100", "-f", "f32le", "-",
        ], binary=True, timeout_s=120)
        reference = np.frombuffer(converted.stdout, dtype="<f4")
        _, decoded = audio.decode(short_pieces / row["path"], sr=44100)
        assert len(decoded) == len(reference)
        # Compare on the saved sample clock, allowing only 16-bit PCM rounding.
        assert np.max(np.abs(decoded - reference)) <= 1 / 32768


@pytest.fixture
def six_samples(short_pieces, monkeypatch):
    module = import_module("yap_studio.voice.samples")
    candidates = [_Candidate(1, "t1", [[i * 15, (i + 1) * 15]],
                             np.full((15 * 48000, 1), 0.1, dtype=np.float32))
                  for i in range(6)]
    choices = iter((candidates, candidates[:4], candidates))
    monkeypatch.setattr(module, "_choose", lambda _: next(choices))
    return short_pieces


def test_samples_reexport_six_then_four_then_six(six_samples, monkeypatch):
    module = import_module("yap_studio.voice.samples")
    lock = module.project_lock
    locked = False

    @contextmanager
    def track_lock(root):
        nonlocal locked
        with lock(root):
            locked = True
            try:
                yield
            finally:
                locked = False

    original_unlink = type(six_samples).unlink

    def checked_unlink(target, *args, **kwargs):
        if target.is_relative_to(six_samples / "voice/samples"):
            assert locked
        return original_unlink(target, *args, **kwargs)

    monkeypatch.setattr(module, "project_lock", track_lock)
    monkeypatch.setattr(type(six_samples), "unlink", checked_unlink)
    for count in (6, 4, 6):
        data = samples(six_samples)
        assert len(data["samples"]) == count
        assert {target.name for target in (six_samples / "voice/samples").rglob("*.wav")} == {
            f"s{i:02}.wav" for i in range(1, count + 1)
        }
        assert json.loads((six_samples / "voice/samples.json").read_text()) == data
        assert all(receipt.sha256_file(six_samples / row["path"]) == row["sha256"]
                   for row in data["samples"])


def test_flat_export_moves_to_a_folder_and_links_current_files(
    six_samples, fake_voice, monkeypatch,
):
    first = samples(six_samples)
    old_files = []
    for row in first["samples"]:
        old = six_samples / row["path"]
        flat = six_samples / "voice/samples" / old.name
        old.rename(flat)
        row["path"] = str(flat.relative_to(six_samples))
        old_files.append(flat)
    write(six_samples, "samples", first)
    unlisted = six_samples / "voice/samples/s07.wav"
    unlisted.write_bytes(b"another file")
    current = samples(six_samples)
    assert len(current["samples"]) == 4
    assert all(not target.exists() for target in old_files)
    assert unlisted.read_bytes() == b"another file"
    assert json.loads((six_samples / "voice/samples.json").read_text()) == current
    for row in current["samples"]:
        target = checked_file(six_samples, row)
        assert target.parent.parent == six_samples / "voice/samples"
        assert target.parent.name.startswith("g") and len(target.parent.name) == 17
    consent(six_samples, yes=True, voice_name="My voice")
    preview = create(six_samples, dry_run=True)
    assert preview["samples"] == [row["path"] for row in current["samples"]]
    assert not any(args[:2] == ["voices", "add"] for args in calls(fake_voice))
    provider = [{"file_name": Path(row["path"]).with_suffix(".mp3").name,
                 "size_bytes": row["size_bytes"]} for row in current["samples"]]
    provider[0]["size_bytes"] += 1
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_SAMPLES", json.dumps(provider))
    with pytest.raises(YapError) as caught:
        link(six_samples, name=preview["name"], yes=True)
    assert caught.value.code == "voice_samples_differ"
    for row in current["samples"]:
        assert str(six_samples / row["path"]) in caught.value.suggested_action
    provider[0]["size_bytes"] -= 1
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_SAMPLES", json.dumps(provider))
    monkeypatch.setenv("FAKE_ELEVENLABS_VOICE_NAME", preview["name"])
    assert link(six_samples, name=preview["name"], yes=True)["samples"] == current["samples"]


@pytest.mark.parametrize("during_export", [False, True])
@pytest.mark.parametrize("binding", [None, "samples", "voice", "both"])
def test_first_export_after_consent(six_samples, monkeypatch, during_export, binding):
    module = import_module("yap_studio.voice.samples")

    def give_consent():
        record = consent(six_samples, yes=True)
        if binding in {"samples", "both"}:
            record["samples"] = joined_record()["samples"]
        if binding in {"voice", "both"}:
            record["voice"].update(id_sha256="a" * 64, made="linked", at="now")
        write(six_samples, "consent", record)

    if during_export:
        lock = module.project_lock

        @contextmanager
        def consent_before_publication(root):
            give_consent()
            with lock(root):
                yield

        monkeypatch.setattr(module, "project_lock", consent_before_publication)
    else:
        give_consent()
    if binding:
        with pytest.raises(YapError) as caught:
            samples(six_samples)
        assert caught.value.code == "voice_samples_in_use"
        assert not (six_samples / "voice/samples.json").exists()
        assert not (six_samples / "voice/samples").exists()
    else:
        data = samples(six_samples)
        assert len(data["samples"]) == 6
        assert all(receipt.sha256_file(six_samples / row["path"]) == row["sha256"]
                   for row in data["samples"])
        record = json.loads((six_samples / "voice/consent.json").read_text())
        assert record["status"] == "given" and record["samples"] == []
        assert record["voice"]["id_sha256"] is None


@pytest.mark.parametrize("failure", ["owner", "sample_write", "record"])
@pytest.mark.parametrize("legacy", [False, True])
def test_sample_publication_failure_keeps_previous_set(six_samples, monkeypatch, failure, legacy):
    module = import_module("yap_studio.voice.samples")
    first = samples(six_samples)
    if legacy:
        for row in first["samples"]:
            target = six_samples / row["path"]
            flat = six_samples / "voice/samples" / target.name
            target.rename(flat)
            row["path"] = str(flat.relative_to(six_samples))
        write(six_samples, "samples", first)
    original = {row["path"]: (six_samples / row["path"]).read_bytes() for row in first["samples"]}
    second = six_samples / first["samples"][1]["path"]
    candidates = [_Candidate(1, "t1", [[i * 15, (i + 1) * 15]],
                             np.full((15 * 48000, 1), 0.2, dtype=np.float32))
                  for i in range(6)]
    monkeypatch.setattr(module, "_choose", lambda _: candidates)
    if failure == "owner":
        second.write_bytes(original[first["samples"][1]["path"]] + b"hand edit")
    with monkeypatch.context() as patch:
        if failure == "sample_write":
            atomic_output = module.paths.atomic_output

            @contextmanager
            def fail_second(target, *args, **kwargs):
                if target.name == "s02.wav":
                    raise OSError("sample publication failed")
                with atomic_output(target, *args, **kwargs) as staged:
                    yield staged

            patch.setattr(module.paths, "atomic_output", fail_second)
        elif failure == "record":
            write_record = module.write

            def fail_record(root, kind, data):
                if kind == "samples":
                    raise OSError("sample record publication failed")
                return write_record(root, kind, data)

            patch.setattr(module, "write", fail_record)
        with pytest.raises((YapError, OSError)):
            samples(six_samples)
    assert json.loads((six_samples / "voice/samples.json").read_text()) == first
    assert {str(target.relative_to(six_samples))
            for target in (six_samples / "voice/samples").rglob("*.wav")} == set(original)
    for relative, content in original.items():
        target = six_samples / relative
        assert target.read_bytes() == content + (b"hand edit" if failure == "owner"
                                                and target == second else b"")
    if failure == "owner":
        second.write_bytes(original[first["samples"][1]["path"]])
    for row in first["samples"]:
        checked_file(six_samples, row)
    latest = samples(six_samples)
    assert latest != first
    for row in latest["samples"]:
        checked_file(six_samples, row)
    assert consent(six_samples, yes=True)["samples"] == latest["samples"]


def test_samples_reexport_leaves_changed_and_unlisted_files(six_samples):
    first = samples(six_samples)
    changed = six_samples / first["samples"][4]["path"]
    unlisted = six_samples / "voice/samples/s07.wav"
    changed.write_bytes(b"changed sample")
    unlisted.write_bytes(b"another file")
    data = samples(six_samples)
    assert len(data["samples"]) == 4
    assert changed.read_bytes() == b"changed sample"
    assert unlisted.read_bytes() == b"another file"
    assert not (six_samples / first["samples"][5]["path"]).exists()
    assert data["notices"] == [f"Left {first['samples'][4]['path']} because it changed."]


@pytest.mark.parametrize("duration", [75, 95])
def test_long_runs_split_evenly_without_short_tail(short_pieces, duration):
    write_edit(short_pieces, [time_segment("s1", "t1", 0, duration)])
    data = samples(short_pieces)
    lengths = [row["duration_s"] for row in data["samples"]]
    assert 4 <= len(lengths) <= 6
    assert min(lengths) >= 15 and max(lengths) <= 30
    assert max(lengths) - min(lengths) <= 1 / 48000
    assert sum(lengths) == pytest.approx(duration, abs=1 / 48000)


@pytest.mark.parametrize("status", ["proposed", "failed", "rejected"])
def test_samples_break_runs_at_fixes(short_pieces, fake_voice, status):
    plan = resolve_cut(short_pieces)
    piece = plan.pieces[len(plan.pieces) // 2]
    word = next(word for word in plan.source_words["t1"].words
                if word.id in piece.words
                and float(piece.src_t0) + 0.1 <= word.start < word.end <= float(piece.src_t1) - 0.1)
    fix = propose(short_pieces, word.id, word.id, "morning")
    stored = json.loads((short_pieces / "voice/fixes.json").read_text())
    stored["fixes"][0]["status"] = status
    write_json(short_pieces / "voice/fixes.json", stored)
    data = samples(short_pieces)
    assert 4 <= len(data["samples"]) <= 6
    blocked = float(piece.src_t0), float(piece.src_t1)
    if status == "rejected":
        assert any(a < blocked[1] and b > blocked[0]
                   for row in data["samples"] for a, b in row["ranges"])
    else:
        for row in data["samples"]:
            assert row["end_s"] <= blocked[0] or row["start_s"] >= blocked[1]
            assert not any(a < fix["source"]["end_s"] + 0.1
                           and b > fix["source"]["start_s"] - 0.1 for a, b in row["ranges"])


@pytest.mark.parametrize("status", ["proposed", "ready", "approved", "failed"])
def test_fix_padding_excludes_whole_adjacent_pieces(status):
    pieces = [SimpleNamespace(take="t1", src_t0=i * 4, src_t1=(i + 1) * 4) for i in range(4)]
    fix = {"take": "t1", "status": status, "source": {"start_s": 7.95, "end_s": 7.99}}
    assert _runs(SimpleNamespace(pieces=pieces), [fix]) == [[pieces[0]], [pieces[3]]]


def joined_record():
    return {
        "schema": "yap-studio/voice-samples@1",
        "samples": [
            {"take": "t1", "start_s": i * 25, "end_s": i * 25 + 21,
             "duration_s": 20, "ranges": [[i * 25, i * 25 + 10], [i * 25 + 11, i * 25 + 21]],
             "path": f"voice/samples/s{i + 1:02}.wav", "sha256": "1" * 64, "size_bytes": 100}
            for i in range(4)
        ],
    }


def test_joined_sample_validation_uses_audio_duration(short_pieces):
    data = joined_record()
    # Removed source gaps may make the source span longer than a 30 s sample.
    data["samples"][0].update(end_s=40, ranges=[[0, 10], [30, 40]])
    assert validate(short_pieces, "samples", data) == data


@pytest.mark.parametrize("field,value", [
    ("duration_s", 14), ("duration_s", 31), ("duration_s", 19.9),
    ("ranges", [[0, 11], [10, 21]]), ("ranges", [[11, 21], [0, 10]]),
    ("ranges", [[-1, 9], [11, 21]]), ("ranges", [[0, 10], [500, 510]]),
    ("ranges", [[0, 10], [11, "21"]]), ("ranges", []), ("ranges", None),
    ("take", "missing"), ("start_s", 1), ("end_s", 22),
])
def test_joined_sample_validation_rejects_broken_ranges(short_pieces, field, value):
    data = copy.deepcopy(joined_record())
    data["samples"][0][field] = value
    with pytest.raises(YapError) as caught:
        validate(short_pieces, "samples", data)
    assert caught.value.code == "voice_record_invalid"


def test_old_sample_records_stay_readable(short_pieces):
    data = joined_record()
    for row in data["samples"]:
        row.pop("duration_s")
        row.pop("ranges")
        row["end_s"] = row["start_s"] + 15
    assert validate(short_pieces, "samples", data) == data
