"""Strict cleanup settings and versioned matte document behavior."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from yap_studio.errors import InputError, ValidationError
from yap_studio.faces.model import VideoIdentity
from yap_studio.matte.model import MATTE_SCHEMA, Cleanup, MatteData, load_matte


def test_cleanup_can_exclude_temporal_filter() -> None:
    assert Cleanup().filter_chain(temporal=False) == Cleanup(median=False).filter_chain()


@pytest.fixture
def measured() -> MatteData:
    return MatteData(
        "measured",
        None,
        VideoIdentity("clip.mp4", "a" * 64, 3, Fraction(30), 16, 24),
        {
            "name": "vision-person-segmentation",
            "quality": "accurate",
            "revision": 1,
            "os": "macOS",
            "helper_sha256": "b" * 64,
            "mask_width": 12,
            "mask_height": 18,
        },
        Cleanup(),
        {"file": "matte.mkv", "sha256": "c" * 64, "codec": "ffv1", "pix_fmt": "gray"},
        (0.0, 0.04, 0.09),
        0.15,
        (0.0, 1.0, 0.5),
        ("variable timing",),
    )


@pytest.mark.parametrize(
    ("cleanup", "expected"),
    [
        (
            Cleanup(),
            "tpad=start=1:stop=1:start_mode=clone:stop_mode=clone,tmedian=radius=1,"
            "lut=c0='clip((val-8)*255/(247-8),0,255)',gblur=sigma=1",
        ),
        (Cleanup(median=False), "lut=c0='clip((val-8)*255/(247-8),0,255)',gblur=sigma=1"),
        (
            Cleanup(feather_sigma=0),
            "tpad=start=1:stop=1:start_mode=clone:stop_mode=clone,tmedian=radius=1,"
            "lut=c0='clip((val-8)*255/(247-8),0,255)'",
        ),
        (Cleanup(False, 0, 255, 0), "lut=c0='clip((val-0)*255/(255-0),0,255)'"),
        (
            Cleanup(False, 4, 200, 0.125),
            "lut=c0='clip((val-4)*255/(200-4),0,255)',gblur=sigma=0.125",
        ),
        (
            Cleanup(feather_sigma=0.000001),
            "tpad=start=1:stop=1:start_mode=clone:stop_mode=clone,tmedian=radius=1,"
            "lut=c0='clip((val-8)*255/(247-8),0,255)',gblur=sigma=0.000001",
        ),
    ],
)
def test_cleanup_filter_chain(cleanup: Cleanup, expected: str) -> None:
    assert cleanup.filter_chain() == expected
    assert Cleanup.from_dict(cleanup.to_dict()) == cleanup


@pytest.mark.parametrize(
    ("updates", "location"),
    [
        ({"median": 1}, "cleanup.median"),
        ({"low": True}, "cleanup.low"),
        ({"low": -1}, "cleanup.low"),
        ({"low": 8.0}, "cleanup.low"),
        ({"high": 256}, "cleanup.high"),
        ({"high": 8}, "cleanup.high"),
        ({"high": "247"}, "cleanup.high"),
        ({"feather_sigma": True}, "cleanup.feather_sigma"),
        ({"feather_sigma": -0.1}, "cleanup.feather_sigma"),
        ({"feather_sigma": 8.1}, "cleanup.feather_sigma"),
        ({"feather_sigma": float("nan")}, "cleanup.feather_sigma"),
        ({"feather_sigma": float("inf")}, "cleanup.feather_sigma"),
        ({"feather_sigma": "1"}, "cleanup.feather_sigma"),
    ],
)
def test_cleanup_rejects_invalid_values(updates: dict[str, Any], location: str) -> None:
    for parse in (
        lambda: Cleanup(**updates),
        lambda: Cleanup.from_dict({**Cleanup().to_dict(), **updates}),
    ):
        with pytest.raises(ValidationError) as caught:
            parse()
        assert caught.value.parameter == location


@pytest.mark.parametrize("key", ["median", "low", "high", "feather_sigma"])
def test_cleanup_requires_all_settings(key: str) -> None:
    payload = Cleanup().to_dict()
    del payload[key]
    with pytest.raises(ValidationError) as caught:
        Cleanup.from_dict(payload)
    assert caught.value.parameter == f"cleanup.{key}"


def test_matte_roundtrip(measured: MatteData, tmp_path: Path) -> None:
    payload = measured.to_dict()
    assert payload["schema"] == MATTE_SCHEMA
    assert payload["video"]["fps"] == "30/1"
    assert payload["times"] == [0.0, 0.04, 0.09]
    assert MatteData.from_dict(json.loads(json.dumps(payload))) == measured
    document = tmp_path / "matte.json"
    document.write_text(json.dumps(payload), encoding="utf-8")
    assert load_matte(document) == measured


def test_empty_frames_roundtrip_and_legacy_default(measured: MatteData, tmp_path: Path) -> None:
    guarded = replace(measured, empty_frames=(0, 2))
    payload = guarded.to_dict()
    assert payload["empty_frames"] == [0, 2]
    document = tmp_path / "matte.json"
    document.write_text(json.dumps(payload), encoding="utf-8")
    assert load_matte(document) == guarded
    legacy = measured.to_dict()
    del legacy["empty_frames"]
    assert MatteData.from_dict(legacy).empty_frames == ()


@pytest.mark.parametrize("indices", [(-1,), (3,), (True,), (1.0,), ("1",), (1, 1), (2, 0)])
def test_empty_frames_rejects_invalid_indices(measured: MatteData, indices: tuple) -> None:
    with pytest.raises(ValidationError) as caught:
        replace(measured, empty_frames=indices)
    assert caught.value.parameter == "matte.empty_frames"
    payload = measured.to_dict()
    payload["empty_frames"] = list(indices)
    with pytest.raises(ValidationError) as caught:
        MatteData.from_dict(payload)
    assert caught.value.parameter == "matte.empty_frames"


@pytest.mark.parametrize("value", [None, "", {}, (0,), [0]])
def test_empty_frames_rejects_wrong_container(measured: MatteData, value: Any) -> None:
    if type(value) is not tuple:
        with pytest.raises(ValidationError, match="empty_frames"):
            replace(measured, empty_frames=value)
    if type(value) is not list:
        with pytest.raises(ValidationError, match="empty_frames"):
            MatteData.from_dict({**measured.to_dict(), "empty_frames": value})


def test_hyperframes_metadata_roundtrip_and_validation(measured: MatteData) -> None:
    data = replace(
        measured,
        backend={
            "name": "hyperframes-remove-background",
            "version": "0.8.134",
            "alignment": 1.0,
        },
    )
    assert MatteData.from_dict(data.to_dict()) == data
    for key, value in (
        ("version", ""),
        ("version", 3),
        ("alignment", True),
        ("alignment", -0.1),
        ("alignment", 1.1),
        ("alignment", float("nan")),
    ):
        with pytest.raises(ValidationError) as caught:
            replace(data, backend={**data.backend, key: value})
        assert caught.value.parameter == f"matte.backend.{key}"


@pytest.mark.parametrize(
    ("keys", "value", "location"),
    [
        (("schema",), "yap-studio/matte@2", "matte.schema"),
        (("status",), "unmeasured", "matte.status"),
        (("status",), True, "matte.status"),
        (("reason",), "unexpected", "matte.reason"),
        (("times",), [0.0], "matte.times"),
        (("coverage",), [0.5], "matte.coverage"),
        (("times",), None, "matte.times"),
        (("times",), [0.0, 0.1, 0.05], "matte.times"),
        (("times", 1), True, "matte.times[1]"),
        (("coverage", 1), True, "matte.coverage[1]"),
        (("coverage", 1), 1.1, "matte.coverage[1]"),
        (("coverage", 1), float("nan"), "matte.coverage[1]"),
        (("end_time",), 0.01, "matte.end_time"),
        (("end_time",), True, "matte.end_time"),
        (("matte",), None, "matte.matte"),
        (("matte", "sha256"), "bad", "matte.matte.sha256"),
        (("matte", "file"), "../matte.mkv", "matte.matte.file"),
        (("matte", "codec"), "h264", "matte.matte.codec"),
        (("matte", "pix_fmt"), "rgb24", "matte.matte.pix_fmt"),
        (("backend", "revision"), True, "matte.backend.revision"),
        (("backend", "quality"), "fast", "matte.backend.quality"),
        (("backend", "mask_width"), 0, "matte.backend.mask_width"),
        (("backend", "helper_sha256"), "bad", "matte.backend.helper_sha256"),
        (("warnings",), [True], "matte.warnings"),
        (("video", "frames"), True, "video.frames"),
    ],
)
def test_matte_rejects_invalid_document_fields(
    measured: MatteData, keys: tuple[str | int, ...], value: Any, location: str
) -> None:
    payload = measured.to_dict()
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValidationError) as caught:
        MatteData.from_dict(payload)
    assert caught.value.parameter == location


@pytest.mark.parametrize(
    "key",
    [
        "schema",
        "status",
        "reason",
        "video",
        "backend",
        "cleanup",
        "matte",
        "times",
        "end_time",
        "coverage",
        "warnings",
    ],
)
def test_matte_requires_document_fields(measured: MatteData, key: str) -> None:
    payload = measured.to_dict()
    del payload[key]
    with pytest.raises(ValidationError) as caught:
        MatteData.from_dict(payload)
    assert caught.value.parameter == f"matte.{key}"


def test_unavailable_roundtrip(measured: MatteData) -> None:
    unavailable = replace(
        measured,
        status="unavailable",
        reason="no Vision",
        backend={},
        matte=None,
        times=None,
        end_time=None,
        coverage=None,
    )
    assert MatteData.from_dict(unavailable.to_dict()) == unavailable
    for key, value in (
        ("reason", None),
        ("reason", " "),
        ("matte", measured.matte),
        ("backend", measured.backend),
        ("times", measured.times),
        ("coverage", measured.coverage),
        ("end_time", measured.end_time),
        ("empty_frames", (0,)),
    ):
        with pytest.raises(ValidationError) as caught:
            replace(unavailable, **{key: value})
        assert caught.value.parameter == f"matte.{key}"


def test_matte_allows_future_backend_metadata(measured: MatteData) -> None:
    future = replace(measured, backend={"name": "future-backend", "settings": {"version": 2}})
    assert MatteData.from_dict(future.to_dict()) == future


def test_load_matte_reports_path(tmp_path: Path) -> None:
    document = tmp_path / "matte.json"
    with pytest.raises(InputError, match="matte.json"):
        load_matte(document)
    for content in (b"not json", b"\xff", b"{}"):
        document.write_bytes(content)
        with pytest.raises(ValidationError, match="matte.json"):
            load_matte(document)
