"""B-roll chronology, visual similarity, and source reuse warnings."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import synth_broll as synth
from test_timeline import add, insert
from test_timeline import project as project

from yap_studio import receipt
from yap_studio.stages import timeline


def shot(root: Path, name: str, start: float, source: Path, **changes: object) -> dict:
    item = insert(root, id=name, start={"time": start}, end={"time": start + 0.6},
                  media={"path": source.name, "sha256": receipt.sha256_file(source),
                         "in_s": 0, "speed": 1}, **changes)
    return item


def findings(result: timeline.ResolvedTimeline, rule: str) -> list:
    return [item for item in result.findings if item.rule == rule]


@pytest.mark.parametrize("dates,exceptions,expected", [
    ((1, 2, 3), ({}, {}, {}), 0),
    ((3, 1, 2), ({}, {}, {}), 1),
    ((4, 1, 2), ({}, {}, {}), 2),
    ((9, 1, 2), ({"chronology": "hook"}, {}, {}), 0),
    ((1, 9, 2), ({}, {"chronology": "before_after"}, {}), 0),
])
def test_chronology(project: Path, monkeypatch: pytest.MonkeyPatch,
                    dates: tuple[int, ...], exceptions: tuple[dict, ...], expected: int) -> None:
    sources = [synth.png(project / f"shot-{index}.png") for index in range(3)]
    stamps = {source.name: datetime(2023, 5, day, 12)
              for source, day in zip(sources, dates, strict=True)}
    monkeypatch.setattr(timeline, "capture_time", lambda path, **kwargs: stamps[Path(path).name])
    items = [shot(project, f"broll-{index}", index, source, **exception)
             for index, (source, exception) in enumerate(zip(sources, exceptions, strict=True))]
    result = add(project, *items)
    warnings = findings(result, "broll_order")
    assert len(warnings) == expected
    assert all(warning.severity == "warning" for warning in warnings)
    if expected:
        assert "captured 2023-05" in warnings[0].message
        assert "before broll-" in warnings[0].message


def test_chronology_slack_and_undated(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources = [synth.png(project / f"other-{index}.png") for index in range(3)]
    stamps = {sources[0].name: datetime(2023, 5, 3, 18),
              sources[1].name: datetime(2023, 5, 2, 19), sources[2].name: None}
    monkeypatch.setattr(timeline, "capture_time", lambda path, **kwargs: stamps[Path(path).name])
    result = add(project, *(shot(project, f"broll-{i}", i, source)
                            for i, source in enumerate(sources)))
    assert findings(result, "broll_order") == []


def test_real_dated_jpeg_orders_end_to_end(project: Path) -> None:
    import struct

    from test_media_captured import tiff_date

    sources = []
    for index, day in enumerate((19, 17)):
        source = synth.jpeg(project / f"dated-{index}.jpg")
        exif = b"Exif\x00\x00" + tiff_date("II").replace(b"2023:06:17", f"2023:06:{day}".encode())
        data = source.read_bytes()
        source.write_bytes(data[:2] + b"\xff\xe1" + struct.pack(">H", len(exif) + 2)
                           + exif + data[2:])
        sources.append(source)
    result = add(project, *(shot(project, f"broll-{i}", i, source)
                            for i, source in enumerate(sources)))
    assert len(findings(result, "broll_order")) == 1


def test_near_duplicates_and_distinct_pictures(project: Path) -> None:
    sources = []
    for index in range(3):
        source = project / f"same-{index}.png"
        synth.synth._run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                          f"color=c=green:s=80x80,drawbox=x={index}:y=0:w=2:h=2:"
                          "color=lime:t=fill", "-frames:v", "1", str(source)])
        sources.append(source)
    different = project / "different.png"
    synth.synth._run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                      "color=c=red:s=80x80", "-frames:v", "1", str(different)])
    result = add(project, *(shot(project, f"broll-{i}", i, source)
                            for i, source in enumerate([*sources, different])))
    groups = findings(result, "broll_near_duplicate")
    assert len(groups) == 1
    assert groups[0].elements == ("broll-0", "broll-1", "broll-2")


def test_near_duplicate_pair_and_distinct_pair(project: Path) -> None:
    first = synth.png(project / "first.png")
    second = project / "second.png"
    synth.synth._run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                      "color=c=green:s=80x80,drawbox=x=1:y=1:w=2:h=2:color=lime:t=fill",
                      "-frames:v", "1", str(second)])
    close = add(project, shot(project, "broll-first", 0, first),
                shot(project, "broll-second", 1, second))
    assert len(findings(close, "broll_near_duplicate")) == 1
    timeline.remove_timeline(project, lane="broll", ids=("broll-first", "broll-second"))
    red = project / "red.png"
    synth.synth._run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                      "color=c=red:s=80x80", "-frames:v", "1", str(red)])
    distinct = add(project, shot(project, "broll-first", 0, first),
                   shot(project, "broll-red", 1, red))
    assert findings(distinct, "broll_near_duplicate") == []


@pytest.mark.parametrize("chronology", ["hook", "before_after"])
def test_near_duplicate_includes_chronology_exception(project: Path, chronology: str) -> None:
    first = synth.png(project / "before.png")
    second = synth.png(project / "after.png")
    second.write_bytes(second.read_bytes() + b"different file bytes")
    result = add(project, shot(project, "broll-before", 0, first,
                               chronology=chronology),
                 shot(project, "broll-after", 1, second))
    warnings = findings(result, "broll_near_duplicate")
    assert len(warnings) == 1
    assert warnings[0].elements == ("broll-before", "broll-after")


def test_near_duplicate_compares_displayed_crop(project: Path) -> None:
    sources = []
    for name, border in (("red-border", "red"), ("blue-border", "blue")):
        source = project / f"{name}.png"
        synth.synth._run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                          f"color=c={border}:s=80x80,"
                          "drawbox=x=20:y=20:w=40:h=40:color=green:t=fill",
                          "-frames:v", "1", str(source)])
        sources.append(source)
    assert receipt.sha256_file(sources[0]) != receipt.sha256_file(sources[1])

    items = []
    for index, source in enumerate(sources):
        item = shot(project, f"broll-{index}", index, source)
        item["media"].update(fit="crop", crop=[20, 20, 40, 40])
        items.append(item)
    warnings = findings(add(project, *items), "broll_near_duplicate")
    assert len(warnings) == 1
    assert warnings[0].elements == ("broll-0", "broll-1")


def test_near_duplicate_distinguishes_displayed_crops(project: Path) -> None:
    first = project / "split-first.png"
    synth.synth._run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                      "color=c=red:s=80x80,drawbox=x=40:y=0:w=40:h=80:color=blue:t=fill",
                      "-frames:v", "1", str(first)])
    second = project / "split-second.png"
    second.write_bytes(first.read_bytes() + b"different file bytes")
    assert receipt.sha256_file(first) != receipt.sha256_file(second)

    left = shot(project, "broll-left", 0, first)
    left["media"].update(fit="crop", crop=[0, 0, 40, 80])
    right = shot(project, "broll-right", 1, second)
    right["media"].update(fit="crop", crop=[40, 0, 40, 80])
    assert findings(add(project, left, right), "broll_near_duplicate") == []


def test_clip_fingerprint_uses_middle_of_source_window(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = [synth.clip(project / f"clip-{i}.mov", colour=colour, duration=3)
               for i, colour in enumerate(("red", "blue"))]
    times = []

    def thumbnail(path: Path, *, sha256: str, time_s: float) -> bytes:
        times.append(time_s)
        return bytes([0 if Path(path) == sources[0] else 255] * 192)

    monkeypatch.setattr(timeline, "capture_time", lambda path, **kwargs: None)
    monkeypatch.setattr(timeline, "fingerprint", thumbnail)
    items = [insert(project, id=f"broll-{i}", start={"time": i}, end={"time": i + 0.6},
                    media={"path": source.name, "sha256": receipt.sha256_file(source),
                           "in_s": 0.4, "speed": 2})
             for i, source in enumerate(sources)]
    assert findings(add(project, *items), "broll_near_duplicate") == []
    assert times == [1.0, 1.0]


def test_reuse_three_warns_twice_does_not(project: Path) -> None:
    source = synth.png(project / "repeat.png")
    twice = add(project, *(shot(project, f"broll-{i}", i, source) for i in range(2)))
    assert findings(twice, "broll_reuse") == []
    thrice = add(project, *(shot(project, f"broll-{i}", i, source) for i in range(3)))
    warnings = findings(thrice, "broll_reuse")
    assert len(warnings) == 1
    assert warnings[0].severity == "warning" and "3 times" in warnings[0].message
