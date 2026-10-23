"""Independent PNG decoding and contact strip layout tests."""

from __future__ import annotations

import struct
import zlib
from fractions import Fraction

import numpy as np
import pytest

from yap_studio.errors import ValidationError
from yap_studio.watch import images


def decode_png(raw: bytes) -> np.ndarray:
    """Check PNG chunks and CRCs, then decode the RGB filter-zero scanlines."""
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    position = 8
    compressed = bytearray()
    width = height = 0
    kinds = []
    while position < len(raw):
        length = struct.unpack_from(">I", raw, position)[0]
        kind = raw[position + 4 : position + 8]
        data = raw[position + 8 : position + 8 + length]
        crc = struct.unpack_from(">I", raw, position + 8 + length)[0]
        assert crc == zlib.crc32(kind + data)
        kinds.append(kind)
        if kind == b"IHDR":
            width, height, depth, colour, compression, filters, interlace = struct.unpack(
                ">IIBBBBB",
                data,
            )
            assert (depth, colour, compression, filters, interlace) == (8, 2, 0, 0, 0)
        elif kind == b"IDAT":
            compressed.extend(data)
        position += length + 12
    assert kinds == [b"IHDR", b"IDAT", b"IEND"]
    assert position == len(raw)
    rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(height, width * 3 + 1)
    assert not rows[:, 0].any()
    return rows[:, 1:].reshape(height, width, 3)


def test_png_round_trip() -> None:
    pixels = np.random.default_rng(123).integers(0, 256, (17, 31, 3), dtype=np.uint8)
    encoded = images.png_bytes(pixels)
    np.testing.assert_array_equal(decode_png(encoded), pixels)
    assert images.png_bytes(pixels) == encoded
    np.testing.assert_array_equal(decode_png(images.png_bytes(pixels[:, ::-1])), pixels[:, ::-1])


def test_glyphs_and_unknown_characters() -> None:
    assert set(images.FONT) == set("0123456789:.C<> ")
    assert all(
        len(rows) == 7 and all(0 <= bits < 32 for bits in rows) for rows in images.FONT.values()
    )
    for char in images.FONT:
        band = np.zeros((24, 20, 3), dtype=np.uint8)
        images.draw_label(band, char)
        assert band.any() == (char != " ")
    with pytest.raises(ValidationError, match="unsupported character"):
        images.draw_label(band, "?")


def test_label_changes_only_the_band() -> None:
    canvas = np.zeros((80 + images.LABEL_HEIGHT, 180, 3), dtype=np.uint8)
    images.draw_label(canvas[80:], "C3< 1:05.33")
    assert not canvas[:80].any()
    assert canvas[80:].any()


@pytest.mark.parametrize(
    "size, expected",
    [
        ((1080, 1920), (180, 320)),
        ((1920, 1080), (320, 180)),
        ((101, 101), (320, 320)),
        ((10, 1000), (4, 320)),
    ],
)
def test_dimensions(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert images.tile_dimensions(*size) == expected


@pytest.mark.parametrize("columns", [1, 2, 3, 6, 12])
def test_layout_keeps_cut_pairs(columns: int) -> None:
    cuts = [{"index": 1, "frame": 30}, {"index": 2, "frame": 61}]
    tiles = images.timeline_tiles(100, Fraction(30), 1.0, cuts)
    assert all(tile["cut_roles"] == [] for tile in tiles if tile["kind"] == "sample")
    assert all(tile["cut_roles"] for tile in tiles if tile["kind"] != "sample")
    assert len({tile["frame"] for tile in tiles}) == len(tiles)
    assert next(tile for tile in tiles if tile["frame"] == 30)["kind"] == "cut_after"
    rows = images.layout_strips(tiles, columns)
    assert [tile["frame"] for row in rows for tile in row] == [tile["frame"] for tile in tiles]
    for cut in cuts:
        pair = [
            (row_index, tile["column"])
            for row_index, row in enumerate(rows)
            for tile in row
            if tile["cut"] == cut["index"]
        ]
        assert pair[0][0] == pair[1][0]
        assert pair[0][1] + 1 == pair[1][1]
    for row in rows:
        assert len(row) <= max(columns, 2)
        assert [tile["column"] for tile in row] == list(range(1, len(row) + 1))


def test_consecutive_cuts_share_one_tile_and_one_strip() -> None:
    tiles = images.timeline_tiles(
        60, Fraction(30), 1.0, [{"index": 1, "frame": 30}, {"index": 2, "frame": 31}]
    )
    shared = next(tile for tile in tiles if tile["frame"] == 30)
    assert len(shared["cut_roles"]) == 2
    rows = images.layout_strips(tiles, 2)
    assert any([tile["frame"] for tile in row] == [29, 30, 31] for row in rows)


def test_strip_pixels_and_time_carry() -> None:
    pixels = np.full((320, 180, 3), 77, dtype=np.uint8)
    tiles = [
        {
            "frame": 0,
            "time_s": 59.99,
            "kind": "cut_before",
            "cut": 3,
            "cut_roles": [{"cut": 3, "kind": "cut_before"}],
        },
        {
            "frame": 1,
            "time_s": 60.02,
            "kind": "cut_after",
            "cut": 3,
            "cut_roles": [{"cut": 3, "kind": "cut_after"}],
        },
    ]
    strip = images.compose_strip(tiles, {0: pixels, 1: pixels})
    assert strip.shape == (344, 362, 3)
    assert tuple(strip[0, 0]) == images.BEFORE_COLOUR
    assert tuple(strip[0, 182]) == images.AFTER_COLOUR
    np.testing.assert_array_equal(
        strip[:, 180:182], np.full((344, 2, 3), images.BACKGROUND, dtype=np.uint8)
    )
    np.testing.assert_array_equal(strip[3:317, 3:177], pixels[3:317, 3:177])
    assert images.format_time(59.999) == "1:00.00"
    assert images.format_time(65.33) == "1:05.33"
    assert images.format_time(110 / 30) == "0:03.67"
    assert images.format_time(111 / 30) == "0:03.70"


def test_long_cut_label_fits_vertical_tile() -> None:
    label = "C100< " + images.format_time(605.33)
    assert label == "C100< 10:05.33"
    narrow = np.zeros((images.LABEL_HEIGHT, 180, 3), dtype=np.uint8)
    wide = np.zeros((images.LABEL_HEIGHT, 240, 3), dtype=np.uint8)
    images.draw_label(narrow, label)
    images.draw_label(wide, label)
    np.testing.assert_array_equal(narrow, wide[:, :180])
    assert narrow.any() and not wide[:, 180:].any()


def test_tile_limit_counts_unique_frames_and_cut_sides() -> None:
    assert len(images.timeline_tiles(5000, Fraction(30), 1 / 30, [])) == 5000
    # A tiny interval repeats frames, so it must remain bounded after deduplication.
    assert len(images.timeline_tiles(120, Fraction(30), 1e-300, [])) == 120
    for every_s in (1 / 30, 1e-300):
        with pytest.raises(ValidationError) as exc:
            images.timeline_tiles(5001, Fraction(30), every_s, [])
        assert "5000" in exc.value.message and "--every" in exc.value.suggested_action
    # Samples alone fit, but the last pair of cut frames adds a new frame.
    with pytest.raises(ValidationError):
        images.timeline_tiles(10000, Fraction(30), 2 / 30, [{"index": 1, "frame": 9999}])
    assert [tile["frame"] for tile in images.timeline_tiles(120, Fraction(30), 1e308, [])] == [0]
