"""xmeml reader tests."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from yap_studio.edl import load_edit_list
from yap_studio.errors import EditListError
from yap_studio.plan import Plan


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _document(
    video_tracks: str,
    *,
    timebase: int = 30,
    ntsc: bool = False,
    audio_tracks: str = "",
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="4">
  <sequence id="sequence-1">
    <name>Edit</name>
    <rate><timebase>{timebase}</timebase><ntsc>{str(ntsc).upper()}</ntsc></rate>
    <media>
      <video>{video_tracks}</video>
      <audio>{audio_tracks}</audio>
    </media>
  </sequence>
</xmeml>
"""


def _clip(
    clip_id: str,
    *,
    start: int,
    end: int,
    source_in: int,
    source_out: int,
    file: str,
    name: str = "take.mov",
    rate: str = "",
    enabled: str = "",
) -> str:
    return f"""
      <clipitem id="{clip_id}">
        <name>{name}</name>{rate}
        <start>{start}</start><end>{end}</end>
        <in>{source_in}</in><out>{source_out}</out>{enabled}
        {file}
      </clipitem>
    """


def _file(file_id: str = "file-1", pathurl: str = "raw/take.mov") -> str:
    return f'<file id="{file_id}"><pathurl>{pathurl}</pathurl></file>'


def test_integer_rate_file_reference_and_percent_encoded_path(tmp_path: Path) -> None:
    first = _clip(
        "clip-1",
        start=0,
        end=72,
        source_in=12,
        source_out=84,
        file=_file(pathurl="file://localhost/media/project/take%20one.mov"),
        name="First",
    )
    second = _clip(
        "clip-2",
        start=72,
        end=144,
        source_in=114,
        source_out=186,
        file='<file id="file-1"/>',
        name="Second",
    )
    path = _write(tmp_path / "edit.xml", _document(f"<track>{first}{second}</track>"))

    edit_list = load_edit_list(path)

    assert edit_list.format == "xmeml"
    assert edit_list.fps == 30
    assert edit_list.sources["file-1"].path == Path("/media/project/take one.mov")
    assert [(item.start, item.end) for item in edit_list.segments] == [
        (Fraction(2, 5), Fraction(14, 5)),
        (Fraction(19, 5), Fraction(31, 5)),
    ]
    assert [item.output_start for item in edit_list.segments] == [0, Fraction(12, 5)]
    assert [item.label for item in edit_list.segments] == ["First", "Second"]


def test_ntsc_sequence_and_clip_rates_are_exact(tmp_path: Path) -> None:
    rate = "<rate><timebase>24</timebase><ntsc>TRUE</ntsc></rate>"
    clip = _clip(
        "clip-1",
        start=30,
        end=60,
        source_in=24,
        source_out=48,
        file=_file(pathurl="/media/project/take.mov"),
        rate=rate,
    )
    path = _write(
        tmp_path / "edit.xml",
        _document(f"<track>{clip}</track>", timebase=30, ntsc=True),
    )

    edit_list = load_edit_list(path)

    assert edit_list.fps == Fraction(30000, 1001)
    assert edit_list.segments[0].start == Fraction(1001, 1000)
    assert edit_list.segments[0].end == Fraction(1001, 500)
    assert edit_list.segments[0].output_start == Fraction(1001, 1000)


def test_relative_path_overlay_masks_disabled_clip_and_split_edit(tmp_path: Path) -> None:
    first = _clip(
        "clip-1",
        start=0,
        end=72,
        source_in=0,
        source_out=72,
        file=_file(),
    )
    disabled = _clip(
        "clip-disabled",
        start=72,
        end=90,
        source_in=72,
        source_out=90,
        file='<file id="file-1"/>',
        enabled="<enabled>FALSE</enabled>",
    )
    second = _clip(
        "clip-2",
        start=72,
        end=144,
        source_in=72,
        source_out=144,
        file='<file id="file-1"/>',
    )
    overlay = _clip(
        "clip-overlay",
        start=30,
        end=60,
        source_in=0,
        source_out=30,
        file='<file id="file-1"/>',
    )
    audio_first = _clip(
        "audio-1",
        start=0,
        end=75,
        source_in=0,
        source_out=75,
        file='<file id="file-1"/>',
    )
    audio_second = _clip(
        "audio-2",
        start=75,
        end=144,
        source_in=75,
        source_out=144,
        file='<file id="file-1"/>',
    )
    document = _document(
        f"<track>{first}{disabled}{second}</track><track>{overlay}</track>",
        audio_tracks=f"<track>{audio_first}{audio_second}</track>",
    )
    edit_list = load_edit_list(_write(tmp_path / "edit.xml", document))

    assert edit_list.sources["file-1"].path == tmp_path / "raw/take.mov"
    assert edit_list.picture_masks == [(Fraction(1), Fraction(2))]
    assert len(edit_list.segments) == 2
    assert any("Disabled clip" in warning for warning in edit_list.warnings)
    assert any("Video track 2" in warning for warning in edit_list.warnings)
    assert any("cut 1 (0:02.4)" in warning for warning in edit_list.warnings)


def test_first_enabled_video_track_is_primary(tmp_path: Path) -> None:
    old = _clip(
        "old-edit",
        start=0,
        end=30,
        source_in=0,
        source_out=30,
        file=_file(),
        name="Old",
    )
    final = _clip(
        "final-edit",
        start=0,
        end=60,
        source_in=90,
        source_out=150,
        file='<file id="file-1"/>',
        name="Final",
    )
    tracks = f"<track><enabled>FALSE</enabled>{old}</track><track>{final}</track>"

    edit_list = load_edit_list(_write(tmp_path / "edit.xml", _document(tracks)))

    assert [segment.label for segment in edit_list.segments] == ["Final"]
    assert edit_list.segments[0].start == 3
    assert edit_list.picture_masks == []
    assert "Disabled video track 1 was ignored." in edit_list.warnings


def test_disabled_overlay_clip_and_track_do_not_add_masks(tmp_path: Path) -> None:
    primary = _clip(
        "primary",
        start=0,
        end=90,
        source_in=0,
        source_out=90,
        file=_file(),
    )
    disabled_clip = _clip(
        "disabled-overlay",
        start=10,
        end=20,
        source_in=10,
        source_out=20,
        file='<file id="file-1"/>',
        enabled="<enabled>FALSE</enabled>",
    )
    disabled_track_clip = _clip(
        "disabled-track-overlay",
        start=30,
        end=40,
        source_in=30,
        source_out=40,
        file='<file id="file-1"/>',
    )
    tracks = (
        f"<track>{primary}</track>"
        f"<track>{disabled_clip}</track>"
        f"<track><enabled>FALSE</enabled>{disabled_track_clip}</track>"
    )

    edit_list = load_edit_list(_write(tmp_path / "edit.xml", _document(tracks)))

    assert edit_list.picture_masks == []
    assert "Disabled clip 'disabled-overlay' was ignored." in edit_list.warnings
    assert "Disabled video track 3 was ignored." in edit_list.warnings
    assert not any("clips mask" in warning for warning in edit_list.warnings)


@pytest.mark.parametrize(
    ("body", "action"),
    [
        (
            "<track><transitionitem><start>10</start><end>20</end></transitionitem></track>",
            "Remove dissolves or transitions",
        ),
        (
            '<track><clipitem id="nested"><sequence id="inside"/></clipitem></track>',
            "Flatten nested sequences",
        ),
    ],
)
def test_unsupported_edits_have_structured_errors(
    tmp_path: Path, body: str, action: str
) -> None:
    path = _write(tmp_path / "edit.xml", _document(body))
    with pytest.raises(EditListError) as error:
        load_edit_list(path)
    assert error.value.code == "unsupported_edit_list"
    assert action in (error.value.suggested_action or "")


@pytest.mark.parametrize("declaration", ["<!DOCTYPE xmeml>", '<!ENTITY sample "value">'])
def test_unsafe_xml_declarations_are_refused(tmp_path: Path, declaration: str) -> None:
    path = _write(tmp_path / "edit.xml", declaration + _document("<track/>"))
    with pytest.raises(EditListError) as error:
        load_edit_list(path)
    assert error.value.code == "unsupported_edit_list"


def test_transition_boundary_without_transition_resolves_from_neighbors(
    tmp_path: Path,
) -> None:
    first = _clip(
        "clip-1",
        start=0,
        end=30,
        source_in=0,
        source_out=30,
        file=_file(),
    )
    second = _clip(
        "clip-2",
        start=-1,
        end=60,
        source_in=30,
        source_out=60,
        file='<file id="file-1"/>',
    )
    path = _write(tmp_path / "edit.xml", _document(f"<track>{first}{second}</track>"))
    edit_list = load_edit_list(path)
    assert edit_list.segments[1].output_start == 1


def test_unresolved_transition_boundary_is_rejected(tmp_path: Path) -> None:
    clip = _clip(
        "clip-1",
        start=-1,
        end=30,
        source_in=0,
        source_out=30,
        file=_file(),
    )
    path = _write(tmp_path / "edit.xml", _document(f"<track>{clip}</track>"))
    with pytest.raises(EditListError) as error:
        load_edit_list(path)
    assert error.value.code == "unsupported_edit_list"


def test_retimed_clip_is_rejected(tmp_path: Path) -> None:
    clip = _clip(
        "clip-1",
        start=0,
        end=30,
        source_in=0,
        source_out=60,
        file=_file(),
        name="2x",
    )
    path = _write(tmp_path / "edit.xml", _document(f"<track>{clip}</track>"))

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


@pytest.mark.parametrize(
    ("timebase", "ntsc", "frame_duration"),
    [(30, False, "1/30s"), (30, True, "1001/30000s")],
)
def test_xmeml_and_fcpxml_produce_the_same_plan(
    tmp_path: Path, timebase: int, ntsc: bool, frame_duration: str
) -> None:
    first = _clip(
        "clip-1",
        start=0,
        end=72,
        source_in=12,
        source_out=84,
        file=_file(pathurl="/media/project/take.mov"),
        name="First",
    )
    second = _clip(
        "clip-2",
        start=72,
        end=144,
        source_in=114,
        source_out=186,
        file='<file id="file-1"/>',
        name="Second",
    )
    xmeml_path = _write(
        tmp_path / "edit.xml",
        _document(f"<track>{first}{second}</track>", timebase=timebase, ntsc=ntsc),
    )
    scale = Fraction(1001, 30000) if ntsc else Fraction(1, 30)
    fcpxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.10">
  <resources>
    <format id="r1" frameDuration="{frame_duration}"/>
    <asset id="file-1" src="/media/project/take.mov"/>
  </resources>
  <library><event><project><sequence format="r1"><spine>
    <asset-clip ref="file-1" offset="0s" start="{12 * scale}s"
      duration="{72 * scale}s" name="First"/>
    <asset-clip ref="file-1" offset="{72 * scale}s" start="{114 * scale}s"
      duration="{72 * scale}s" name="Second"/>
  </spine></sequence></project></event></library>
</fcpxml>
"""
    fcpxml_path = _write(tmp_path / "edit.fcpxml", fcpxml)

    xmeml_list = load_edit_list(xmeml_path)
    fcpxml_list = load_edit_list(fcpxml_path)
    xmeml_plan = Plan.build(xmeml_list, xmeml_list.fps or 30, 48_000)
    fcpxml_plan = Plan.build(fcpxml_list, fcpxml_list.fps or 30, 48_000)

    xmeml_values = [
        (item.src_f0, item.src_f1, item.n_frames, item.out_f0)
        for item in xmeml_plan.segments
    ]
    fcpxml_values = [
        (item.src_f0, item.src_f1, item.n_frames, item.out_f0)
        for item in fcpxml_plan.segments
    ]
    assert xmeml_values == fcpxml_values
    assert xmeml_plan.total_frames == fcpxml_plan.total_frames
