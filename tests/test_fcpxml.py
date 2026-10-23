"""FCPXML reader tests."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from yap_studio.edl import load_edit_list
from yap_studio.edl.fcpxml import parse_time
from yap_studio.errors import EditListError


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _document(primary: str, *, version: str = "1.10", resources: str = "") -> str:
    default_resources = """
      <format id="r1" frameDuration="1001/30000s"/>
      <asset id="r2" start="0s"><media-rep kind="original-media" src="raw/take.mov"/></asset>
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="{version}">
  <resources>{resources or default_resources}</resources>
  <library><event><project>
    <sequence format="r1"><spine>{primary}</spine></sequence>
  </project></event></library>
</fcpxml>
"""


def test_parse_time_is_exact_and_rejects_bad_values() -> None:
    assert parse_time("1001/30000s") == Fraction(1001, 30000)
    assert parse_time("5s") == 5
    assert parse_time(None) == 0
    with pytest.raises(EditListError):
        parse_time("five")


def test_version_110_media_rep_and_relative_path(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "edit.fcpxml",
        _document('<asset-clip ref="r2" offset="0s" start="2s" duration="3s" name="A"/>'),
    )
    edit_list = load_edit_list(path)

    assert edit_list.fps == Fraction(30000, 1001)
    assert edit_list.sources["r2"].path == tmp_path / "raw/take.mov"
    assert edit_list.segments[0].start == 2
    assert edit_list.segments[0].end == 5
    assert edit_list.segments[0].label == "A"


def test_version_18_asset_src_and_timecode_start(tmp_path: Path) -> None:
    resources = """
      <format id="r1" frameDuration="1/24s"/>
      <asset id="r2" src="/media/project/take.mov" start="3600s"/>
    """
    path = _write(
        tmp_path / "edit.fcpxml",
        _document(
            '<asset-clip ref="r2" offset="10s" start="3601s" duration="2s"/>',
            version="1.8",
            resources=resources,
        ).replace('<sequence format="r1">', '<sequence format="r1" tcStart="10s">'),
    )
    segment = load_edit_list(path).segments[0]
    assert segment.start == 1
    assert segment.end == 3
    assert segment.output_start == 0


def test_gap_and_clip_with_video_child(tmp_path: Path) -> None:
    primary = """
      <gap offset="0s" duration="2s" name="pause"/>
      <clip offset="2s" start="5s" duration="3s" name="nested">
        <video ref="r2" offset="5s" start="11s" duration="3s"/>
      </clip>
    """
    edit_list = load_edit_list(_write(tmp_path / "edit.fcpxml", _document(primary)))
    gap, clip = edit_list.segments
    assert gap.is_gap
    assert gap.duration == 2
    assert clip.source_id == "r2"
    assert clip.start == 11
    assert clip.end == 14
    assert [cut.index for cut in edit_list.cuts()] == [1]


def test_clip_with_video_child_subtracts_asset_timecode_start(tmp_path: Path) -> None:
    resources = """
      <format id="r1" frameDuration="1/24s"/>
      <asset id="r2" src="/media/project/take.mov" start="3600s"/>
    """
    primary = """
      <clip offset="0s" start="0s" duration="2s">
        <video ref="r2" offset="0s" start="3601s" duration="2s"/>
      </clip>
    """
    path = _write(tmp_path / "edit.fcpxml", _document(primary, resources=resources))

    segment = load_edit_list(path).segments[0]
    assert segment.start == 1
    assert segment.end == 3


def test_connected_lanes_create_masks_and_warnings(tmp_path: Path) -> None:
    primary = """
      <asset-clip ref="r2" offset="2s" start="10s" duration="5s">
        <asset-clip lane="1" ref="r2" offset="12s" start="20s" duration="2s"/>
        <audio lane="-1" ref="r2" offset="10s" start="10s" duration="1s"/>
      </asset-clip>
    """
    edit_list = load_edit_list(_write(tmp_path / "edit.fcpxml", _document(primary)))
    assert edit_list.picture_masks == [(Fraction(4), Fraction(6))]
    assert len(edit_list.warnings) == 2


def test_nested_connected_lanes_use_their_parent_timeline(tmp_path: Path) -> None:
    primary = """
      <asset-clip ref="r2" offset="0s" start="10s" duration="5s">
        <asset-clip lane="1" ref="r2" offset="12s" start="20s" duration="2s">
          <asset-clip lane="1" ref="r2" offset="21s" start="30s" duration="1s"/>
        </asset-clip>
      </asset-clip>
    """

    edit_list = load_edit_list(_write(tmp_path / "edit.fcpxml", _document(primary)))

    assert edit_list.picture_masks == [
        (Fraction(2), Fraction(4)),
        (Fraction(3), Fraction(4)),
    ]


def test_retimed_connected_clip_records_picture_mask(tmp_path: Path) -> None:
    primary = """
      <asset-clip ref="r2" offset="2s" start="10s" duration="5s">
        <asset-clip lane="1" ref="r2" offset="12s" start="20s" duration="2s"
          name="slow B-roll">
          <timeMap>
            <timept time="0s" value="0s"/>
            <timept time="2s" value="4s"/>
          </timeMap>
        </asset-clip>
      </asset-clip>
    """

    edit_list = load_edit_list(_write(tmp_path / "edit.fcpxml", _document(primary)))

    assert len(edit_list.segments) == 1
    assert edit_list.picture_masks == [(Fraction(4), Fraction(6))]
    assert edit_list.warnings == ["Connected lane 1 clip masks the primary picture."]


@pytest.mark.parametrize(
    "primary",
    [
        """
          <asset-clip ref="r2" offset="0s" start="0s" duration="1s" name="2x">
            <timeMap>
              <timept time="0s" value="0s"/>
              <timept time="1s" value="2s"/>
            </timeMap>
          </asset-clip>
        """,
        """
          <clip offset="0s" start="0s" duration="1s" name="2x">
            <video ref="r2" offset="0s" start="0s" duration="1s">
              <timeMap>
                <timept time="0s" value="0s"/>
                <timept time="1s" value="2s"/>
              </timeMap>
            </video>
          </clip>
        """,
    ],
)
def test_retimed_clip_is_rejected(tmp_path: Path, primary: str) -> None:
    path = _write(tmp_path / "edit.fcpxml", _document(primary))

    with pytest.raises(EditListError) as error:
        load_edit_list(path)

    assert error.value.code == "unsupported_edit_list"
    assert error.value.message == (
        "Clip '2x' is retimed (its timeline length differs from its source length). "
        "Retimed clips are not supported yet."
    )
    assert error.value.suggested_action == (
        "Render speed changes into a new source clip first."
    )


@pytest.mark.parametrize("element", ["ref-clip", "sync-clip", "mc-clip", "transition"])
def test_unsupported_elements_have_structured_error(tmp_path: Path, element: str) -> None:
    path = _write(
        tmp_path / "edit.fcpxml",
        _document(f'<{element} offset="0s" duration="1s"/>'),
    )
    with pytest.raises(EditListError) as error:
        load_edit_list(path)
    assert error.value.code == "unsupported_edit_list"
    assert error.value.suggested_action


@pytest.mark.parametrize(
    "declaration",
    ["<!DOCTYPE fcpxml>", '<!ENTITY sample "value">'],
)
def test_unsafe_xml_declarations_are_refused(tmp_path: Path, declaration: str) -> None:
    path = _write(tmp_path / "edit.fcpxml", declaration + _document(""))
    with pytest.raises(EditListError) as error:
        load_edit_list(path)
    assert error.value.code == "unsupported_edit_list"


def test_percent_encoded_local_file_url(tmp_path: Path) -> None:
    resources = """
      <format id="r1" frameDuration="1/25s"/>
      <asset id="r2" src="file://localhost/media/project/clip%20one.mov"/>
    """
    path = _write(
        tmp_path / "edit.fcpxml",
        _document(
            '<asset-clip ref="r2" offset="0s" duration="1s"/>', resources=resources
        ),
    )
    assert load_edit_list(path).sources["r2"].path == Path("/media/project/clip one.mov")


def test_fcpxmld_bundle_reads_info_document(tmp_path: Path) -> None:
    bundle = tmp_path / "project.fcpxmld"
    bundle.mkdir()
    _write(
        bundle / "Info.fcpxml",
        _document('<asset-clip ref="r2" offset="0s" duration="1s"/>'),
    )
    assert load_edit_list(bundle).segments[0].duration == 1
