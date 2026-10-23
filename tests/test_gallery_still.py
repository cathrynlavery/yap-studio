"""Synthetic checks for gallery stills and opening words."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yap_studio import ffmpeg, tools
from yap_studio.cli import main
from yap_studio.errors import ProcessingError, ValidationError
from yap_studio.gallery import still
from yap_studio.transcript import ScriptedBackend, Transcript, Word


def clip(path: Path, *, size: str = "108x192") -> Path:
    """Make a small, steady picture with exact candidate times."""
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"smptebars=size={size}:rate=10", "-f", "lavfi", "-i",
         "anullsrc=channel_layout=mono:sample_rate=16000", "-t", "3.1",
         "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(path)],
        timeout_s=30,
    )
    return path


class Faces:
    """Return one scripted face per candidate."""

    def __init__(self, *faces: still.FaceLandmarks | None) -> None:
        self.faces = faces

    def measure(self, images: list[Path]) -> list[list[still.FaceLandmarks]]:
        assert len(images) == len(self.faces)
        return [[face] if face else [] for face in self.faces]


def face(*, y: float = 50, open_eye: float = 0.3) -> still.FaceLandmarks:
    return still.FaceLandmarks(150, y, 180, 200, open_eye, open_eye, 0.0, 0.0, 0.9)


def test_newest_take(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    old = inbox / "old.MOV"
    new = inbox / "new.mp4"
    old.touch()
    new.touch()
    (inbox / ".hidden.mp4").touch()
    (inbox / "notes.txt").touch()
    (inbox / "folder.m4v").mkdir()
    os.utime(old, (100, 100))
    os.utime(new, (200, 200))
    assert still.newest_take(inbox) == new
    assert still.newest_take(tmp_path / "missing") is None
    assert still.newest_take(tmp_path) is None


def _words(text: str, *, start: float = 0.0, step: float = 0.3) -> list[Word]:
    return [Word(token, start + index * step, start + index * step + 0.2)
            for index, token in enumerate(text.split())]


@pytest.mark.parametrize("ending", ["word.", "word!", "word?", 'word."', "word?)"])
def test_first_sentence_end_marks_and_retimes(ending: str) -> None:
    opening = _words("These five words make a complete sentence", start=2.111, step=1.1)
    opening[-1] = Word(ending, opening[-1].start, opening[-1].end)
    result = still.first_sentence([*opening, *_words("Next five words make another sentence.",
                                                     start=12.0)])
    assert [(word.id, word.text) for word in result] == [
        (f"w{index}", word.text) for index, word in enumerate(opening)
    ]
    assert [(word.start, word.end) for word in result] == [
        (0.3, 0.5), (1.4, 1.6), (2.5, 2.7), (3.6, 3.8),
        (4.7, 4.9), (5.8, 6.0), (6.9, 7.1),
    ]


def test_first_sentence_silence_boundary() -> None:
    opening = _words("These words stay together", step=1.5)
    later = _words("This later sentence has enough words.", start=7.0)
    result = still.first_sentence([*opening, *later])
    assert [word.text for word in result] == [word.text for word in later]


def test_first_sentence_skips_short_sentence() -> None:
    words = [*_words("Small start."), *_words("This next sentence has enough words.", start=1.0)]
    assert [word.text for word in still.first_sentence(words)] == [
        "This", "next", "sentence", "has", "enough", "words."
    ]


@pytest.mark.parametrize("opener", [
    "All right.", "All right, do you know what this is?", "Okay so here is the thing.",
    "Hey everyone, welcome back.", "Um, let me show you.",
    "Alright, here is something useful today.", "Ok, here is something useful today.",
    "So, here is something useful today.", "Hi, here is something useful today.",
    "Hello, here is something useful today.", "Uh, here is something useful today.",
    "Umm, here is something useful today.", "Uhh, here is something useful today.",
    "Well, here is something useful today.", "Yeah, here is something useful today.",
])
def test_first_sentence_skips_opener(opener: str) -> None:
    later = _words("This useful sentence explains the actual idea.", start=6.0)
    result = still.first_sentence([*_words(opener), *later])
    assert [word.text for word in result] == [word.text for word in later]


def test_first_sentence_keeps_long_sentence_to_end_mark() -> None:
    words = _words("Let me tell you about the time before rideshare when we used to have "
                   "to call a car and wait outside for ages before it finally arrived.")
    assert len(words) > 14
    assert [word.text for word in still.first_sentence(words)] == [word.text for word in words]


def test_first_sentence_safety_cap_is_40_words() -> None:
    words = _words(" ".join(["word"] * 44 + ["end."]))
    assert len(still.first_sentence(words)) == 40


def test_first_sentence_skips_open_sentence_at_window_end() -> None:
    complete = _words("Here is the complete useful thought.")
    open_words = _words("This next thought continues beyond the recording window", start=37.6)
    assert [word.text for word in still.first_sentence(
        [*complete, *open_words], window_end_s=40.0
    )] == [word.text for word in complete]
    assert [word.text for word in still.first_sentence(
        open_words, window_end_s=40.0
    )] == [word.text for word in open_words]


def test_first_sentence_falls_back_to_longest_sentence() -> None:
    words = [*_words("One two three."), *_words("Four five six seven.", start=2.0)]
    assert [word.text for word in still.first_sentence(words)] == [
        "Four", "five", "six", "seven."
    ]
    tie = [*_words("One two three."), *_words("Four five six.", start=2.0)]
    assert [word.text for word in still.first_sentence(tie)] == ["One", "two", "three."]
    opener_words = [*_words("Okay one two three four five."),
                    *_words("Um one two three four five six.", start=3.0)]
    assert [word.text for word in still.first_sentence(opener_words)] == [
        "Um", "one", "two", "three", "four", "five", "six."
    ]


def test_first_sentence_empty() -> None:
    assert still.first_sentence([]) == []


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("opening_sentence", [
    "Okay here is a little opener.",
    "This opening sentence explains the useful idea.",
])
def test_prepare_take_cached_sentence_comes_from_the_opening(
    tmp_path, monkeypatch, cached, opening_sentence,
):
    from test_ingest import make_media
    from test_transcript_shared import CountingBackend

    from yap_studio.transcript import Transcript, shared

    take = make_media(tmp_path / "take.mkv", seconds=44, sound="silence", fps=2)
    backend = CountingBackend()
    backend.words = _words(opening_sentence)
    later = _words("This later sentence explains the useful idea.", start=41)
    if cached:
        digest = shared.take_digest(take)
        shared.store(backend, shared.Request(digest, 0, take),
                     Transcript([*backend.words, *later], duration_s=44))
    observed = {}

    def pick(_take, _out, *, words, detector):
        observed["words"] = words
        return SimpleNamespace(path=still.sample_still(), time_s=1,
                               scores={"faces": []}, reasons=())

    monkeypatch.setattr(still, "pick_still", pick)
    result = still.prepare_take(take, tmp_path / "out", backend=backend)
    assert result["transcript"] == ("take" if cached else "opening")
    assert result["sentence"] == opening_sentence
    assert len(backend.calls) == (0 if cached else 1)
    assert observed["words"] == backend.words
    assert json.loads((tmp_path / "out/preview.json").read_text())["transcript"] == (
        result["transcript"]
    )


@pytest.mark.parametrize("duration", [3.1, 40.0, 44.0])
def test_prepare_take_cached_words_use_the_opening_window(tmp_path, monkeypatch, duration):
    from test_ingest import make_media
    from test_transcript_shared import CountingBackend

    from yap_studio.transcript import shared

    take = make_media(tmp_path / "take.mkv", seconds=duration, sound="silence", fps=2)
    backend = CountingBackend()
    info = ffmpeg.probe(take)
    window_end = min(40.0, info.audio_duration_s or info.duration_s or 40.0)
    opener = _words("Okay here is a little opener.", step=0.1)
    unfinished = _words("This useful thought continues past", start=window_end - 0.9, step=0.2)
    crossing = Word("the", window_end - 0.05, window_end + 0.1)
    outside = Word("window.", window_end, window_end + 0.3)
    words = [*opener, *unfinished, crossing, outside]
    request = shared.Request(shared.take_digest(take), 0, take)
    shared.store(backend, request, Transcript(words, duration_s=duration))
    observed = {}

    def pick(_take, _out, *, words, detector):
        observed["words"] = words
        return SimpleNamespace(path=still.sample_still(), time_s=1,
                               scores={"faces": []}, reasons=())

    monkeypatch.setattr(still, "pick_still", pick)
    result = still.prepare_take(take, tmp_path / "out", backend=backend)
    assert observed["words"] == [*opener, *unfinished, crossing]
    assert result["sentence"] == "Okay here is a little opener."
    assert result["transcript"] == "take"
    assert backend.calls == []


def test_ingest_remembers_long_take_for_gallery(tmp_path, monkeypatch):
    from test_ingest import make_media, make_project
    from test_transcript_shared import CountingBackend

    from yap_studio import receipt, transcript
    from yap_studio.stages.ingest import ingest
    from yap_studio.transcript import shared

    take = make_media(tmp_path / "take.mkv", seconds=44, sound="silence", fps=2)
    backend = CountingBackend()
    monkeypatch.setattr(transcript, "resolve_backend", lambda *_: backend)
    root = make_project(tmp_path / "project", [take])
    ingest(root)
    assert backend.durations == [44.0]
    backend.calls.clear()
    monkeypatch.setattr(receipt, "sha256_file", lambda _: pytest.fail("unexpected gallery hash"))
    monkeypatch.setattr(still, "pick_still", lambda *_args, **_kwargs: SimpleNamespace(
        path=still.sample_still(), time_s=1, scores={"faces": []}, reasons=(),
    ))
    result = still.prepare_take(take, tmp_path / "preview", backend=backend)
    assert result["transcript"] == "take"
    assert backend.calls == []
    assert shared.known_digest(take) is not None
    with take.open("ab") as output:
        output.write(b"changed")
    assert shared.known_digest(take) is None


def test_pick_prefers_open_eyes_clear_band_and_speech(tmp_path: Path) -> None:
    take = clip(tmp_path / "take.mp4")
    detector = Faces(face(open_eye=0.1), face(y=500), face(), face())
    chosen = still.pick_still(take, tmp_path / "out", detector=detector,
                              band=(900, 1300), words=[Word("speaking", 2, 2.2)])
    assert chosen.source == "take"
    assert chosen.time_s == 2.0
    assert chosen.reasons == ()
    assert (ffmpeg.probe(chosen.path).width, ffmpeg.probe(chosen.path).height) == (1080, 1920)
    record = json.loads((chosen.path.parent / "still.json").read_text())
    assert record["schema"] == "yap-studio/gallery-still@1"
    assert len(record["candidates"]) == 4
    assert not list(chosen.path.parent.glob("*.png"))
    assert not list(chosen.path.parent.glob("still-frames-*"))


def test_pick_uses_eye_openness_relative_to_take(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    take = clip(tmp_path / "take.mp4")
    values = iter([10.0, 100.0, 9.0, 8.0])
    monkeypatch.setattr(still, "_sharpness", lambda _frame, _face: next(values))
    chosen = still.pick_still(
        take, tmp_path / "out", detector=Faces(
            face(open_eye=0.4), face(open_eye=0.25),
            face(open_eye=0.4), face(open_eye=0.4),
        ), band=(900, 1300),
    )
    record = json.loads((chosen.path.parent / "still.json").read_text())
    assert chosen.time_s == 1.0
    assert record["eyes_open_threshold"] == pytest.approx(0.32)
    assert record["candidates"][1]["eyes_open"] is False


def test_pick_accepts_uniformly_narrow_eyes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    take = clip(tmp_path / "take.mp4")
    values = iter([10.0, 100.0, 9.0, 8.0])
    monkeypatch.setattr(still, "_sharpness", lambda _frame, _face: next(values))
    chosen = still.pick_still(
        take, tmp_path / "out", detector=Faces(*(face(open_eye=0.25) for _ in range(4))),
        band=(900, 1300),
    )
    record = json.loads((chosen.path.parent / "still.json").read_text())
    assert chosen.time_s == 1.5
    assert record["eyes_open_threshold"] == 0.2
    assert all(candidate["eyes_open"] for candidate in record["candidates"])


def test_pick_rounds_candidate_and_result_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    take = clip(tmp_path / "take.mp4")
    values = iter([1.0, 100.0, 1.0, 1.0, 1.0, 1.0])
    monkeypatch.setattr(still, "_sharpness", lambda _frame, _face: next(values))
    chosen = still.pick_still(
        take, tmp_path / "out", detector=Faces(*(face() for _ in range(6))),
        step_s=1 / 3, band=(900, 1300),
    )
    record = json.loads((chosen.path.parent / "still.json").read_text())
    assert [candidate["time_s"] for candidate in record["candidates"]] == [
        1.0, 1.333, 1.667, 2.0, 2.333, 2.667,
    ]
    assert chosen.time_s == record["time_s"] == chosen.scores["time_s"] == 1.333


def test_published_still_is_scored_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    take = tmp_path / "changing.mp4"
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=108x192:rate=7:duration=3.1", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(take)],
        timeout_s=30,
    )
    candidate = tmp_path / "scored.png"

    class RecordingFaces(Faces):
        def measure(self, images: list[Path]) -> list[list[still.FaceLandmarks]]:
            candidate.write_bytes(images[3].read_bytes())
            return super().measure(images)

    values = iter([1.0, 1.0, 1.0, 100.0, 1.0, 1.0])
    monkeypatch.setattr(still, "_sharpness", lambda _frame, _face: next(values))
    chosen = still.pick_still(
        take, tmp_path / "out", detector=RecordingFaces(*(None for _ in range(6))),
        step_s=1 / 3,
    )
    assert chosen.time_s == 2.0

    def pixels(path: Path) -> np.ndarray:
        raw = ffmpeg.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
             "-vf", "scale=108:192", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
            timeout_s=30, binary=True,
        ).stdout
        return np.frombuffer(raw, dtype=np.uint8).astype(np.int16)

    assert np.mean(np.abs(pixels(candidate) - pixels(chosen.path))) < 3


@pytest.mark.parametrize(
    ("faces", "expected_time", "reason_count"),
    [
        ((face(y=500), face(y=500), face(y=500), face(y=500)), 1.0, 1),
        ((face(open_eye=0.1), face(open_eye=0.1), face(open_eye=0.1),
          face(open_eye=0.1)), 1.0, 2),
        ((None, None, None, None), 1.0, 3),
    ],
)
def test_pick_relaxes_in_order(
    tmp_path: Path, faces: tuple[still.FaceLandmarks | None, ...],
    expected_time: float, reason_count: int,
) -> None:
    take = clip(tmp_path / "take.mp4")
    chosen = still.pick_still(take, tmp_path / "out", detector=Faces(*faces),
                              band=(900, 1300))
    assert chosen.time_s == expected_time
    assert len(chosen.reasons) == reason_count


def test_landscape_is_centre_cropped(tmp_path: Path) -> None:
    take = clip(tmp_path / "landscape.mp4", size="192x108")
    chosen = still.pick_still(take, tmp_path / "out", detector=Faces(None, None, None, None))
    info = ffmpeg.probe(chosen.path)
    assert (info.width, info.height) == (1080, 1920)


def test_single_frame_take_uses_middle_candidate(tmp_path: Path) -> None:
    take = tmp_path / "short.mp4"
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         "smptebars=size=108x192:rate=10", "-frames:v", "1", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(take)],
        timeout_s=30,
    )
    chosen = still.pick_still(take, tmp_path / "out", detector=Faces(None))
    assert chosen.time_s == 0.05
    assert chosen.path.is_file()


def test_no_face_fallback_uses_sharpest_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    take = clip(tmp_path / "take.mp4")
    values = iter([10.0, 9.0, 8.0, 7.0])
    monkeypatch.setattr(still, "_sharpness", lambda _frame, _face: next(values))
    chosen = still.pick_still(
        take, tmp_path / "out", detector=Faces(None, None, None, None),
        words=[Word("late", 2.4, 2.6)],
    )
    assert chosen.time_s == 1.0
    assert chosen.reasons[-1] == "No face in the opening; picked the sharpest frame."


def test_prepare_preview_empty_and_take(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    sample = still.prepare_preview(inbox, tmp_path / "sample")
    assert sample["source"] == "sample"
    assert Path(sample["still"]).read_bytes() == still.sample_still().read_bytes()
    assert json.loads((tmp_path / "sample" / "preview.json").read_text()) == sample
    clip(inbox / "take.mp4")
    backend = ScriptedBackend(lambda _path: Transcript([
        Word("Hello", 0.5, 0.7), Word("there.", 0.8, 1.1), Word("Next", 1.2, 1.4)
    ]))
    result = still.prepare_preview(inbox, tmp_path / "preview", detector=Faces(
        face(), face(), face(), face()), backend=backend)
    assert result["source"] == "take"
    assert result["sentence"] == "Hello there."
    assert len(result["words"]) == 2
    assert result["face_boxes"] == [{"x": 300, "y": 100, "w": 360, "h": 400}]
    assert json.loads((tmp_path / "preview" / "preview.json").read_text()) == result
    direct = still.prepare_take(inbox / "take.mp4", tmp_path / "direct", detector=Faces(
        face(), face(), face(), face()), transcribe=False)
    assert direct["source"] == "take"
    assert json.loads((tmp_path / "direct" / "preview.json").read_text()) == direct


def test_preview_retains_every_face_from_the_selected_still(tmp_path: Path) -> None:
    take = clip(tmp_path / "take.mp4")
    first, second = face(), still.FaceLandmarks(380, 400, 80, 100, None, None, None, None, 0.9)

    class TwoFaces:
        def measure(self, images):
            return [[first, second] for _ in images]

    result = still.prepare_take(take, tmp_path / "out", detector=TwoFaces(), transcribe=False)
    assert result["face_boxes"] == [
        {"x": 300, "y": 100, "w": 360, "h": 400},
        {"x": 760, "y": 800, "w": 160, "h": 200},
    ]


@pytest.mark.parametrize("sample", [False, True])
def test_refuses_foreign_still_jpg(
    tmp_path: Path, sample: bool
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    image = out / "still.jpg"
    image.write_bytes(b"personal photo")
    if sample:
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        def run() -> dict:
            return still.prepare_preview(inbox, out)
    else:
        take = clip(tmp_path / "take.mp4")

        def run() -> dict:
            return still.prepare_take(
                take, out, detector=Faces(*(None for _ in range(4))), transcribe=False
            )
    with pytest.raises(ValidationError) as exc:
        run()
    assert exc.value.parameter == "--out"
    assert image.read_bytes() == b"personal photo"
    assert not (out / "still.svg").exists()


@pytest.mark.parametrize("sample", [False, True])
def test_second_run_into_owned_folder(tmp_path: Path, sample: bool) -> None:
    out = tmp_path / "out"
    if sample:
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        def run() -> dict:
            return still.prepare_preview(inbox, out)

        marker = "yap-studio/gallery-preview@1"
        record = out / "preview.json"
    else:
        take = clip(tmp_path / "take.mp4")

        def run() -> dict:
            return still.prepare_take(
                take, out, detector=Faces(*(None for _ in range(4))), transcribe=False
            )

        marker = "yap-studio/gallery-still@1"
        record = out / "still.json"
    first = run()
    second = run()
    assert first["source"] == second["source"]
    assert Path(second["still"]).is_file()
    assert json.loads(record.read_text())["schema"] == marker


def test_cached_landmark_helper_needs_no_xcrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = tmp_path / "landmarks"
    helper.write_bytes(b"cached")
    helper.chmod(0o755)
    face_detect = importlib.import_module("yap_studio.faces.detect")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(still, "_helper_path", lambda: helper)
    monkeypatch.setattr(face_detect, "helper_path", lambda: tmp_path / "missing")
    monkeypatch.setattr(still.tools, "find_tool", lambda _name: None)
    assert still.landmarks_unavailable_reason() is None
    assert isinstance(still.default_landmark_detector(), still.VisionLandmarkDetector)


def test_face_box_cache_does_not_enable_landmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = tmp_path / "face-boxes"
    helper.write_bytes(b"cached")
    helper.chmod(0o755)
    face_detect = importlib.import_module("yap_studio.faces.detect")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(still, "_helper_path", lambda: tmp_path / "missing")
    monkeypatch.setattr(face_detect, "helper_path", lambda: helper)
    monkeypatch.setattr(still.tools, "find_tool", lambda _name: None)
    assert "Xcode command line tools" in still.landmarks_unavailable_reason()
    assert still.default_landmark_detector() is None


def test_video_only_take_skips_transcription(tmp_path: Path) -> None:
    take = tmp_path / "silent.mp4"
    ffmpeg.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=108x192:rate=10:duration=3.1", "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(take)],
        timeout_s=30,
    )
    backend = ScriptedBackend(lambda _path: pytest.fail("speech backend was called"))
    result = still.prepare_take(
        take, tmp_path / "out", detector=Faces(*(None for _ in range(4))), backend=backend
    )
    assert result["sentence"] is None
    assert Path(result["still"]).is_file()


def test_transcribe_without_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(still.transcript, "resolve_backend", lambda _name: None)
    assert still.transcribe_opening(tmp_path / "missing.mp4", tmp_path / "out") is None


def test_transcribe_opening_default_is_40_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(still.ffmpeg, "probe", lambda _take: SimpleNamespace(has_audio=True))
    monkeypatch.setattr(still.ffmpeg, "run", lambda args, **_kwargs: commands.append(args))
    backend = ScriptedBackend(lambda _wav: Transcript([]))

    assert still.transcribe_opening(tmp_path / "take.mp4", tmp_path / "out", backend=backend)
    assert commands[0][commands[0].index("-t") + 1] == "40.0"


@pytest.mark.parametrize("line", [
    '{"image":"x","width":540,"height":960,"faces":[{"x":true}]}',
    '{"image":"x","width":540,"height":960,"faces":[{"x":500,"y":0,"w":60,"h":10,"left_eye":0.2,"right_eye":0.2,"yaw":null,"roll":null,"confidence":0.9}]}',
    '{"image":"x","width":540,"height":960,"faces":"bad"}',
])
def test_bad_helper_output(line: str) -> None:
    with pytest.raises(ProcessingError):
        still._parse_output(line, [Path("x")])


def test_cli_missing_input_and_sample(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["style", "still", "--out", str(tmp_path / "out")]) == 2
    error = capsys.readouterr().err
    assert "error: Bad value for TAKE: provide TAKE or --inbox DIR" in error
    assert "fix: Fix TAKE and retry." in error
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    assert main(["style", "still", "--inbox", str(inbox), "--out", str(tmp_path / "out")]) == 0
    assert "still.jpg" in capsys.readouterr().out
    assert main(["style", "still", "--inbox", str(inbox), "--out", str(tmp_path / "json"),
                 "--asr", "faster-whisper", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["source"] == "sample"


def test_cli_refuses_foreign_output_with_out_fix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "still.jpg").write_bytes(b"personal photo")

    assert main(["style", "still", "--inbox", str(inbox), "--out", str(out)]) == 2
    error = capsys.readouterr().err
    assert "output folder has files this command did not write" in error
    assert "fix: Fix --out and retry." in error


@pytest.mark.parametrize(
    ("extra", "fix"),
    [
        (["take.mp4", "--inbox", "inbox"], "TAKE"),
        (["--inbox", "missing"], "--inbox"),
    ],
)
def test_cli_invalid_input_options_have_parameter_fix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], fix: str
) -> None:
    assert main(["style", "still", *extra, "--out", str(tmp_path / "out")]) == 2
    assert f"fix: Fix {fix} and retry." in capsys.readouterr().err


_REAL_HELPER = pytest.mark.skipif(
    sys.platform != "darwin" or tools.find_tool("xcrun") is None,
    reason="real Vision helper needs macOS and xcrun",
)


@pytest.mark.helpers
@_REAL_HELPER
def test_real_helper_on_synthetic_frame(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    ffmpeg.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "smptebars=size=540x960:rate=1", "-frames:v", "1", str(image)],
               timeout_s=30)
    assert isinstance(still.VisionLandmarkDetector().measure([image])[0], list)
