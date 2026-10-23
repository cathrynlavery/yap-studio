"""Agent judgments validate completely before the catalog is changed."""

from __future__ import annotations

import copy
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import synth_broll as synth

from yap_studio import cli
from yap_studio.errors import InputError, ValidationError
from yap_studio.media import (
    DESCRIPTION_SCHEMA,
    add_paths,
    catalog_path,
    describe_item,
    get_item,
    read_description_file,
)

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
STAMP = "2026-10-05T12:00:00Z"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YAP_STUDIO_CONFIG", str(tmp_path / "config"))


@pytest.fixture(scope="session")
def sample_clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return synth.clip(tmp_path_factory.mktemp("describe") / "clip.mov", audio=False)


@pytest.fixture
def item(tmp_path: Path, sample_clip: Path) -> dict:
    source = Path(shutil.copyfile(sample_clip, tmp_path / "clip.mov"))
    add_paths([source], now=lambda: NOW)
    return get_item("m1")


def _describe(**changes: object) -> dict:
    return describe_item("m1", {"schema": DESCRIPTION_SCHEMA, **changes}, now=lambda: NOW)


def test_description_tags_types_provenance_and_append_only_decisions(item: dict) -> None:
    result = _describe(
        description="  Hands typing\non a laptop  ", tags=[" Laptop ", "TYPING", "laptop"],
        media_type="screen_recording",
        provenance={"kind": "own", "source": "phone", "license": None, "url": None},
        ranges=[
            {"verdict": "approved", "start_s": 0.25, "end_s": 1.0, "reason": " steady "},
            {"verdict": "rejected", "reason": "camera shake throughout"},
        ],
    )
    assert result["id"] == item["id"]
    assert result["description"] == {
        "text": "Hands typing\non a laptop", "tags": ["laptop", "typing"],
        "by": "agent", "at": STAMP, "revision": 1,
    }
    assert result["decisions_added"] == ["d1", "d2"] and result["revoked"] == []
    first = get_item("m1")
    assert result["item"] == first
    assert first["media_type"] == "screen_recording" and first["media_type_by"] == "agent"
    assert first["provenance"] == {
        "kind": "own", "source": "phone", "license": None, "url": None,
    }
    assert first["decisions"] == [
        {"id": "d1", "verdict": "approved", "start_s": 0.25, "end_s": 1.0,
         "reason": "steady", "by": "agent", "at": STAMP, "revoked_at": None, "revoked_by": None},
        {"id": "d2", "verdict": "rejected", "start_s": None, "end_s": None,
         "reason": "camera shake throughout", "by": "agent", "at": STAMP,
         "revoked_at": None, "revoked_by": None},
    ]
    second = _describe(tags=["\tDesk "], ranges=[{"verdict": "approved", "reason": "useful"}])
    assert second["description"] == {**result["description"], "tags": ["desk"], "revision": 2}
    assert second["decisions_added"] == ["d3"]
    assert second["item"]["decisions"][:2] == first["decisions"]
    assert second["item"]["next_decision"] == 4
    third = _describe(description="A desk")
    assert third["description"]["tags"] == ["desk"]
    assert third["description"]["revision"] == 3


def test_tags_first_description_and_revoke_preserve_history(item: dict) -> None:
    result = _describe(tags=[], ranges=[{"verdict": "approved", "reason": "useful"}])
    assert result["description"]["text"] is None
    before = copy.deepcopy(result["item"]["decisions"])
    instant = NOW + timedelta(hours=1)
    revoked = describe_item(
        "m1", {"schema": DESCRIPTION_SCHEMA, "by": "reviewer", "revoke": ["d1"]},
        now=lambda: instant,
    )
    assert revoked["revoked"] == ["d1"] and revoked["decisions_added"] == []
    assert revoked["item"]["decisions"] == [
        {**before[0], "revoked_at": "2026-10-05T13:00:00Z", "revoked_by": "reviewer"},
    ]
    assert revoked["item"]["next_decision"] == 2
    assert revoked["description"] == result["description"]


@pytest.mark.parametrize(("changes", "field"), [
    ({"schema": None, "tags": []}, "schema"),
    ({"extra": 1, "tags": []}, "extra"),
    ({}, "data"),
    ({"by": "agent"}, "data"),
    ({"by": "", "tags": []}, "by"),
    ({"by": "x" * 65, "tags": []}, "by"),
    ({"by": "a\nb", "tags": []}, "by"),
    ({"description": "  "}, "description"),
    ({"description": "x" * 2001}, "description"),
    ({"description": "line\ttab"}, "description"),
    ({"description": 2}, "description"),
    ({"tags": "desk"}, "tags"),
    ({"tags": [""]}, "tags[0]"),
    ({"tags": ["x" * 41]}, "tags[0]"),
    ({"tags": ["desk"] * 51}, "tags"),
    ({"tags": [False]}, "tags[0]"),
    ({"media_type": "photo"}, "media_type"),
    ({"media_type": []}, "media_type"),
    ({"provenance": {"kind": "own"}}, "provenance"),
    ({"provenance": {"kind": "own", "source": None, "license": None, "url": "ftp://host"}},
     "provenance.url"),
    ({"provenance": {"kind": "own", "source": None, "license": None, "url": "https://"}},
     "provenance.url"),
    ({"provenance": {"kind": "own", "source": None, "license": None, "url": "https://host:bad"}},
     "provenance.url"),
    ({"provenance": {"kind": "own", "source": "s" * 201, "license": None, "url": None}},
     "provenance.source"),
    ({"ranges": [{"verdict": "maybe", "reason": "useful"}]}, "ranges[0].verdict"),
    ({"ranges": [{"verdict": "approved", "reason": " "}]}, "ranges[0].reason"),
    ({"ranges": [{"verdict": "approved"}]}, "ranges[0].reason"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "extra": 1}]}, "ranges[0].extra"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "start_s": 0, "end_s": 3}]},
     "ranges[0]"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "start_s": 1, "end_s": 1}]},
     "ranges[0]"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "start_s": 2, "end_s": 1}]},
     "ranges[0]"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "start_s": -1, "end_s": 1}]},
     "ranges[0]"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "start_s": None}]}, "ranges[0]"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "start_s": 0, "end_s": None}]},
     "ranges[0].end_s"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "start_s": False, "end_s": 1}]},
     "ranges[0].start_s"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "start_s": 0, "end_s": float("inf")}]},
     "ranges[0].end_s"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "start_s": float("nan"), "end_s": 1}]},
     "ranges[0].start_s"),
    ({"ranges": [{"verdict": "approved", "reason": "ok", "start_s": 0, "end_s": 10 ** 400}]},
     "ranges[0].end_s"),
    ({"ranges": {}}, "ranges"),
    ({"revoke": ["d99"]}, "revoke[0]"),
    ({"revoke": ["d1"]}, "revoke[0]"),
    ({"revoke": ["d2", "d2"]}, "revoke[1]"),
])
def test_invalid_payload_leaves_catalog_bytes_unchanged(
    item: dict, changes: dict, field: str,
) -> None:
    _describe(ranges=[{"verdict": "approved", "reason": "ok"}] * 2)
    _describe(revoke=["d1"])
    before = catalog_path().read_bytes()
    with pytest.raises(ValidationError) as caught:
        _describe(**changes)
    assert field in caught.value.parameter
    assert catalog_path().read_bytes() == before


def test_missing_schema_and_non_object_leave_catalog_unchanged(item: dict) -> None:
    for data in ({"description": "desk"}, [], None):
        before = catalog_path().read_bytes()
        with pytest.raises(ValidationError):
            describe_item("m1", data)
        assert catalog_path().read_bytes() == before


def test_late_invalid_revoke_does_not_apply_earlier_fields(item: dict) -> None:
    before = catalog_path().read_bytes()
    with pytest.raises(ValidationError):
        _describe(description="new text", ranges=[{"verdict": "approved", "reason": "ok"}],
                  revoke=["d99"])
    assert catalog_path().read_bytes() == before


def test_duration_tolerance_and_explicit_whole_item_nulls(item: dict) -> None:
    result = _describe(ranges=[
        {"verdict": "approved", "reason": "ok", "start_s": 0, "end_s": 2.001},
        {"verdict": "rejected", "reason": "ok", "start_s": None, "end_s": None},
    ])
    assert result["decisions_added"] == ["d1", "d2"]
    assert result["item"]["decisions"][1]["start_s"] is None


def test_still_accepts_whole_item_only_and_fitting_type(tmp_path: Path) -> None:
    add_paths([synth.png(tmp_path / "still.png")])
    _describe(media_type="photo", ranges=[{"verdict": "approved", "reason": "ok"}])
    before = catalog_path().read_bytes()
    for changes in (
        {"media_type": "clip"},
        {"ranges": [{"verdict": "approved", "reason": "ok", "start_s": 0, "end_s": 1}]},
    ):
        with pytest.raises(ValidationError):
            _describe(**changes)
        assert catalog_path().read_bytes() == before


def test_file_loader(tmp_path: Path) -> None:
    source = tmp_path / "description.json"
    with pytest.raises(InputError, match="description file not found"):
        read_description_file(source)
    source.write_text("{invalid}")
    with pytest.raises(InputError, match="not valid JSON"):
        read_description_file(source)
    source.write_text(json.dumps({"schema": DESCRIPTION_SCHEMA, "tags": ["desk"]}))
    assert read_description_file(source) == {"schema": DESCRIPTION_SCHEMA, "tags": ["desk"]}


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_describe_show_and_unknown_id(
    tmp_path: Path, item: dict, capsys: pytest.CaptureFixture[str], as_json: bool,
) -> None:
    source = tmp_path / "judgment.json"
    flag = ["--json"] if as_json else []
    source.write_text(json.dumps({
        "schema": DESCRIPTION_SCHEMA, "description": "A laptop", "tags": ["desk"],
        "ranges": [{"verdict": "approved", "reason": "steady"}],
    }))
    assert cli.main(["media", "describe", "m1", "--file", str(source), *flag]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    if as_json:
        assert json.loads(output.out)["decisions_added"] == ["d1"]
    else:
        assert "Description revision 1" in output.out and "Added decisions: d1" in output.out
    source.write_text(json.dumps({"schema": DESCRIPTION_SCHEMA, "revoke": ["d1"]}))
    assert cli.main(["media", "describe", "m1", "--file", str(source), *flag]) == 0
    output = capsys.readouterr().out
    if as_json:
        assert json.loads(output)["revoked"] == ["d1"]
    else:
        assert "Revoked decisions: d1" in output
    assert cli.main(["media", "show", "m1"]) == 0
    output = capsys.readouterr().out
    assert "Description: A laptop" in output and "Tags: desk" in output
    assert "d1 approved whole item: steady (agent," in output and " (revoked)" in output
    assert cli.main(["media", "show", "m1", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    stored = get_item("m1")
    assert shown == {**stored, "locations": [
        {**location, "exists": True} for location in stored["locations"]
    ]}
    assert cli.main(["media", "describe", "m99", "--file", str(source), *flag]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    if as_json:
        assert json.loads(output.err)["error"]["code"] == "unknown_media"
    else:
        assert "Unknown media id: m99" in output.err
    source.write_text(json.dumps({"schema": DESCRIPTION_SCHEMA, "tags": [""]}))
    before = catalog_path().read_bytes()
    assert cli.main(["media", "describe", "m1", "--file", str(source), *flag]) == 2
    capsys.readouterr()
    assert catalog_path().read_bytes() == before
