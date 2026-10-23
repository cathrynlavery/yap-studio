"""Eyedropper style and project contracts."""

import json

import jsonschema
import pytest

from yap_studio.errors import YapError
from yap_studio.project.model import SCHEMAS_DIR, parse_document
from yap_studio.style import SCHEMA_PATH, resolve_style_data


def test_style_eyedropper_colour_round_trip():
    data = {"schema_version": 1,
            "grade": {"look": "eyedropper", "color": "#1E6BD6", "strength": 0.5}}
    assert resolve_style_data(data)["grade"] == {**data["grade"], "slider": False}
    jsonschema.validate(data, json.loads(SCHEMA_PATH.read_text()))


def test_style_eyedropper_starts_at_light():
    data = {"schema_version": 1, "grade": {"look": "eyedropper", "color": "#1E6BD6"}}
    assert resolve_style_data(data)["grade"]["strength"] == 0.5


@pytest.mark.parametrize("grade", [
    {"look": "eyedropper"}, {"look": "eyedropper", "color": None},
    {"look": "eyedropper", "color": "blue"}, {"look": "eyedropper", "color": "#808080"},
])
def test_style_bad_eyedropper_refused(grade):
    with pytest.raises(YapError) as error:
        resolve_style_data({"schema_version": 1, "grade": grade})
    assert error.value.suggested_action


@pytest.mark.parametrize("selection", [
    {"color": "#1E6BD6"}, {"sample": {"time_s": 0.1, "x": 0.2, "y": 0.3}},
])
def test_project_eyedropper_forms(selection):
    data = {"schema": "yap-studio/project@1", "name": "grade",
            "takes": [{"id": "t1", "path": "take.mov"}],
            "grade": {"look": "eyedropper", **selection}}
    result = parse_document("project", data)
    assert result.grade == data["grade"]
    jsonschema.validate(data, json.loads((SCHEMAS_DIR / "project.schema.json").read_text()))


@pytest.mark.parametrize("selection", [
    {}, {"color": "#1E6BD6", "sample": {"time_s": 0, "x": 0.5, "y": 0.5}},
    {"sample": {"time_s": -1, "x": 0.5, "y": 0.5}},
    {"sample": {"time_s": 0, "x": 1.1, "y": 0.5}}, {"color": None},
])
def test_project_eyedropper_requires_one_valid_selection(selection):
    with pytest.raises(YapError) as error:
        parse_document("project", {"schema": "yap-studio/project@1", "name": "grade",
                                  "takes": [{"id": "t1", "path": "take.mov"}],
                                  "grade": {"look": "eyedropper", **selection}})
    assert error.value.suggested_action
