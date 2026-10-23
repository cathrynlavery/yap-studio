"""End-to-end exit contract through the installed console script."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import pytest
import synth

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.e2e


@dataclass(frozen=True)
class ExitFixtures:
    root: Path
    source: Path
    spans: list[tuple[float, float]]
    click: Path
    clean: Path
    shifted: Path
    click_vertical: Path
    clean_vertical: Path
    ranges: Path
    xmeml: Path
    click_oracle: synth.Oracle
    clean_oracle: synth.Oracle
    shifted_oracle: synth.Oracle


def _ranges_json(path: Path, source: Path, spans: list[tuple[float, float]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "sources": {"main": str(source)},
                "ranges": [
                    {"source": "main", "start": start, "end": end}
                    for start, end in spans
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _xmeml_xml(path: Path, source: Path, spans: list[tuple[float, float]]) -> Path:
    clips = []
    output_frame = 0
    for index, (start, end) in enumerate(spans, 1):
        source_in = round(start * 30)
        source_out = round(end * 30)
        duration = source_out - source_in
        file_node = (
            f'<file id="file-main"><pathurl>{escape(str(source))}</pathurl></file>'
            if index == 1
            else '<file id="file-main"/>'
        )
        clips.append(
            f"""
        <clipitem id="clip-{index}">
          <name>Segment {index}</name>
          <rate><timebase>30</timebase><ntsc>FALSE</ntsc></rate>
          <start>{output_frame}</start><end>{output_frame + duration}</end>
          <in>{source_in}</in><out>{source_out}</out>
          {file_node}
        </clipitem>"""
        )
        output_frame += duration
    document = f"""<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="4">
  <sequence id="sequence-1">
    <name>Calibration edit</name>
    <rate><timebase>30</timebase><ntsc>FALSE</ntsc></rate>
    <media><video><track>{''.join(clips)}
      </track></video><audio/></media>
  </sequence>
</xmeml>
"""
    path.write_text(document, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def exit_fixtures(
    synth_media: object,
    tmp_path_factory: pytest.TempPathFactory,
) -> ExitFixtures:
    root = tmp_path_factory.mktemp("exit")
    spans = synth.default_spans()
    source = synth_media.path("calib")
    click = synth_media.path("perclip_click3")
    clean = synth_media.path("clean")
    shifted = synth_media.path("clean_shift_start")
    click_vertical = synth_media.path("perclip_click3_crop")
    clean_vertical = synth_media.path("clean_crop")
    ranges = _ranges_json(root / "edit.json", source, spans)
    xmeml = _xmeml_xml(root / "edit.xml", source, spans)
    click_oracle = synth.oracle(click, spans)
    clean_oracle = synth.oracle(clean, spans)
    shifted_oracle = synth.oracle(shifted, spans)
    assert click_oracle.slope_ms_per_cut is not None
    assert clean_oracle.slope_ms_per_cut is not None
    assert click_oracle.slope_ms_per_cut >= 10
    assert abs(clean_oracle.slope_ms_per_cut) <= 1
    return ExitFixtures(
        root=root,
        source=source,
        spans=spans,
        click=click,
        clean=clean,
        shifted=shifted,
        click_vertical=click_vertical,
        clean_vertical=clean_vertical,
        ranges=ranges,
        xmeml=xmeml,
        click_oracle=click_oracle,
        clean_oracle=clean_oracle,
        shifted_oracle=shifted_oracle,
    )


@pytest.fixture(scope="module")
def console_script() -> Path:
    found = shutil.which("yap-studio")
    candidate = Path(found) if found else Path(sys.executable).parent / "yap-studio"
    if not candidate.is_file():
        pytest.skip(
            "yap-studio console script is not on PATH or beside the test Python executable"
        )
    return candidate


def _verify(
    console_script: Path,
    media: Path,
    *args: str | Path,
    asr: str = "none",
) -> tuple[int, dict]:
    command = [
        str(console_script),
        "verify",
        str(media),
        *(str(arg) for arg in args),
        "--json",
        "--asr",
        asr,
        "--detail",
        "full",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = result.stdout
    if not output:
        output = result.stderr.splitlines()[-1] if result.stderr else ""
    assert output, f"no JSON response\nstderr:\n{result.stderr}"
    try:
        report = json.loads(output)
    except json.JSONDecodeError:
        pytest.fail(f"invalid JSON on stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result.returncode, report


def _findings(report: dict, check: str, *, cut: int | None = None) -> list[dict]:
    return [
        row
        for row in report["findings"]
        if row["check"] == check and (cut is None or row["cut"] == cut)
    ]


def _drift(report: dict) -> float:
    rows = _findings(report, "sync.drift")
    assert len(rows) == 1
    return rows[0]["evidence"]["drift_ms_per_cut"]


def _assert_clean(report: dict) -> None:
    assert not [row for row in report["findings"] if row["severity"] == "fail"]
    assert report["checks"]["join.click"]["status"] == "pass"
    assert report["checks"]["join.frames"]["status"] == "pass"
    assert all(row["click_db"] < 0 for row in report["cuts"])
    assert all(row["detail"]["frames"]["status"] == "pass" for row in report["cuts"])
    offsets = [row["av_offset_ms"] for row in report["segments"]]
    assert all(offset is not None and abs(offset) <= 1 for offset in offsets)


def test_exit_contract(console_script: Path, exit_fixtures: ExitFixtures) -> None:
    fixture = exit_fixtures
    edl = ("--edl", fixture.ranges)

    click_code, click = _verify(console_script, fixture.click, *edl)
    assert click_code == 1
    assert click["checks"]["join.click"]["status"] == "fail"
    assert click["checks"]["sync.av_offset"]["status"] == "fail"
    assert _findings(click, "join.click", cut=3)[0]["severity"] == "fail"
    assert _findings(click, "join.frames", cut=3)[0]["severity"] == "fail"
    assert _drift(click) == pytest.approx(fixture.click_oracle.slope_ms_per_cut, abs=2)

    clean_code, clean = _verify(console_script, fixture.clean, *edl)
    assert clean_code == 0
    _assert_clean(clean)

    xml_code, xml = _verify(console_script, fixture.clean, "--edl", fixture.xmeml)
    assert xml_code == 0
    assert xml["edit_list"]["format"] == "xmeml"
    _assert_clean(xml)

    shifted_code, shifted = _verify(console_script, fixture.shifted, *edl)
    assert shifted_code == 1
    expected_shifts = [
        fixture.shifted_oracle.segment_offsets_ms[index]
        - fixture.clean_oracle.segment_offsets_ms[index]
        for index in sorted(fixture.clean_oracle.segment_offsets_ms)
    ]
    expected_shift = sum(expected_shifts) / len(expected_shifts)
    shifted_offsets = [
        row["av_offset_ms"] for row in shifted["segments"] if row["av_offset_ms"] is not None
    ]
    assert shifted_offsets
    assert all(offset == pytest.approx(expected_shift, abs=1) for offset in shifted_offsets)
    assert shifted["checks"]["sync.av_offset"]["status"] == "fail"

    vertical_click_code, vertical_click = _verify(
        console_script, fixture.click_vertical, *edl
    )
    assert vertical_click_code == 1
    assert _findings(vertical_click, "join.click", cut=3)[0]["severity"] == "fail"
    assert vertical_click["checks"]["sync.drift"]["status"] == "warn"
    assert _findings(vertical_click, "sync.drift")

    vertical_clean_code, vertical_clean = _verify(
        console_script, fixture.clean_vertical, *edl
    )
    assert vertical_clean_code == 0
    _assert_clean(vertical_clean)

    detected_click_code, detected_click = _verify(console_script, fixture.click)
    assert detected_click_code == 1
    assert detected_click["summary"]["cuts"] == 7
    assert _findings(detected_click, "join.click", cut=3)[0]["severity"] == "fail"
    assert [row["detail"]["cut"]["frame"] for row in detected_click["cuts"]] == list(
        range(72, 505, 72)
    )

    detected_clean_code, detected_clean = _verify(console_script, fixture.clean)
    assert detected_clean_code == 0
    assert detected_clean["summary"]["cuts"] == 7
    assert [row["detail"]["cut"]["frame"] for row in detected_clean["cuts"]] == list(
        range(72, 505, 72)
    )

    shorts_vertical_code, shorts_vertical = _verify(
        console_script,
        fixture.clean_vertical,
        *edl,
        "--platform",
        "shorts",
    )
    assert shorts_vertical_code == 0
    assert shorts_vertical["checks"]["platform.shape"]["status"] == "pass"

    shorts_horizontal_code, shorts_horizontal = _verify(
        console_script,
        fixture.clean,
        *edl,
        "--platform",
        "shorts",
    )
    assert shorts_horizontal_code == 1
    assert shorts_horizontal["checks"]["platform.shape"]["status"] == "fail"

    youtube_code, youtube = _verify(
        console_script,
        fixture.clean,
        *edl,
        "--platform",
        "youtube",
    )
    assert youtube_code == 0
    assert youtube["checks"]["platform.shape"]["status"] == "pass"

    words_code, words = _verify(
        console_script,
        fixture.clean,
        *edl,
        asr="auto",
    )
    if words_code == 2:
        assert words["error"]["code"] == "speech_engine_failed"
        words_code, words = _verify(console_script, fixture.clean, *edl)
    assert words_code == 0
    word_check = words["checks"]["join.words"]
    if words["receipt"]["speech"] is None:
        assert word_check == {
            "ran": True,
            "status": "pass",
            "reason": None,
            "note": (
                "Words not named: no speech engine installed; "
                "pip install 'yap-studio[asr]'"
            ),
        }
    else:
        assert word_check["status"] == "skipped"
        assert word_check["reason"] in {"no speech near the cut", "no speech near the cuts"}
