"""Chapter names and clocks stay bound to the delivered render."""

from fractions import Fraction
from math import ceil

import pytest
import synth
from test_verify import _ranges

from yap_studio.verify import verify
from yap_studio.youtube_chapters import chapter_list


@pytest.mark.parametrize("fps", ["60/1", "30000/1001"])
def test_chapter_list_uses_evidence_clock(fps):
    rate = Fraction(fps)
    rows = [{"start_frame": ceil(second * rate), "title": title}
            for second, title in [(0, "Start"), (10, "Middle"), (20, "End")]]
    assert chapter_list(rows, ceil(30 * rate), fps=fps) == (
        "0:00 Start\n0:10 Middle\n0:20 End\n", None)
    text, fix = chapter_list(rows, ceil(30 * rate) - 1, fps=fps)
    assert text is None and "10 seconds" in fix


@pytest.mark.parametrize("fps", [0, -30, True, "bad", "NaN", "Infinity", "1/0", None])
def test_chapter_list_rejects_invalid_clock(fps):
    text, fix = chapter_list([{"start_frame": 0, "title": "Start"}], 900, fps=fps)
    assert text is None and "frame rate" in fix


@pytest.mark.parametrize("fps", ["60/1", "30000/1001"])
@pytest.mark.parametrize("clock", ["matching", "legacy", "media_mismatch", "edl_mismatch"])
def test_verify_chapter_clock(tmp_path, fps, clock):
    rate = Fraction(fps)
    media = synth.calib_source(tmp_path / "source.mov", seconds=31, fps=fps, size="96x72")
    evidence = {"chapters": [{"start_frame": ceil(second * rate), "title": title}
                             for second, title in [(0, "Start"), (10, "Middle"), (20, "End")]],
                "total_frames": round(31 * rate), "fps": fps}
    if clock == "legacy":
        evidence.pop("fps")
    elif clock == "media_mismatch":
        evidence["fps"] = "30/1"
    edl = _ranges(tmp_path / "edit.json", media, [(0, 31)],
                  fps="30/1" if clock == "edl_mismatch" else fps, youtube_chapters=evidence)
    report = verify(str(media), edl=str(edl), asr="none", loudness="note", detail="full")
    check = report["checks"]["platform.chapters"]
    assert check["ran"]
    if clock in {"matching", "legacy"}:
        assert check["status"] == "pass", report["findings"]
        assert report["youtube_chapters"] == "0:00 Start\n0:10 Middle\n0:20 End\n"
    else:
        assert check["status"] == "warn"
        assert "youtube_chapters" not in report
        finding, = [row for row in report["findings"] if row["check"] == "platform.chapters"]
        assert "frame rate" in finding["suggested_action"]


@pytest.mark.parametrize("evidence", [None, {}, {"chapters": []}])
@pytest.mark.parametrize("with_edl", [False, True])
def test_verify_skips_absent_chapters_with_platform(tmp_path, evidence, with_edl):
    media = synth.calib_source(tmp_path / "source.mov", seconds=1, size="96x72")
    edl = _ranges(tmp_path / "edit.json", media, [(0, 1)], youtube_chapters=evidence)
    report = verify(str(media), edl=str(edl) if with_edl else None, platform="youtube",
                    asr="none", loudness="note", detail="full")
    assert report["checks"]["platform.chapters"] == {
        "ran": False, "status": "skipped", "reason": "no chapters in this video"}
