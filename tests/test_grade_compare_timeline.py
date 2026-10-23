"""Compare windows obey the picture lane and editorial constraints."""

import json

import pytest
from test_timeline import fragment, overlay, transition, transition_story, write
from test_timeline import project as project  # noqa: F401

from yap_studio import receipt
from yap_studio.stages.timeline import add_timeline, resolve_timeline


def compare(identifier="agent-compare", start=1, duration=3):
    return {"id": identifier, "kind": "grade_compare", "lane": "agent",
            "start": {"time": start}, "hold_s": duration}


def set_grade(root, look="film", strength=1):
    data = json.loads((root / "project.json").read_text())
    data["grade"] = {"look": look, "strength": strength}
    write(root / "project.json", data)


@pytest.mark.parametrize("duration", [2, 8])
def test_compare_schema_and_visual_limit(project, duration):
    set_grade(project)
    added = add_timeline(project, fragment(project, compare(duration=duration)), lane="agent")
    assert added.valid
    assert added.elements[0].kind == "grade_compare"


@pytest.mark.parametrize("duration", [1.9, 8.1])
def test_compare_length(project, duration):
    set_grade(project)
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1", "words_sha256": {},
                                           "elements": [compare(duration=duration)]})
    row = next(row for row in resolve_timeline(project).findings
               if row.rule == "grade_compare_length")
    assert row.severity == "violation"
    assert row.message == "agent-compare (lane agent): grade compare length must be 2 to 8 seconds"
    assert "yap-studio timeline add" in row.suggested_action


@pytest.mark.parametrize("look,strength", [("none", 1), ("film", 0)])
def test_compare_requires_active_grade(project, look, strength):
    set_grade(project, look, strength)
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1", "words_sha256": {},
                                           "elements": [compare()]})
    row = next(row for row in resolve_timeline(project).findings
               if row.rule == "grade_compare_grade")
    assert row.message == "agent-compare (lane agent): grade compare needs an active grade"
    assert row.severity == "violation"


def test_compare_overlap_and_count(project):
    set_grade(project)
    write(project / "timeline/agent.json", {
        "schema": "yap-studio/timeline@1", "words_sha256": {}, "elements": [
            compare("agent-one", 0, 3), compare("agent-two", 2, 3), compare("agent-three", 7, 3)]})
    rows = resolve_timeline(project).findings
    overlap = next(row for row in rows if row.rule == "grade_compare_overlap")
    assert overlap.message == (
        "agent-one (lane agent), agent-two (lane agent): grade compare cannot overlap overlays, "
        "backgrounds, transitions or another grade compare")
    count = next(row for row in rows if row.rule == "grade_compare_count")
    assert count.message == (
        "agent-one (lane agent), agent-two (lane agent), agent-three (lane agent): "
        "at most two grade compares are allowed per video")
    assert overlap.severity == count.severity == "violation"


@pytest.mark.parametrize("kind", ["overlay", "background", "transition"])
def test_compare_refuses_other_picture_effects(project, kind):
    set_grade(project)
    if kind == "transition":
        transition_story(project)
        elements = [compare(start=5), transition()]
    else:
        other = overlay(project, start={"time": 5}, end={"time": 8}, lane="broll")
        other["kind"] = kind
        if kind == "background":
            other.pop("placement")
        write(project / "timeline/broll.json", {"schema": "yap-studio/timeline@1",
                                               "words_sha256": {}, "elements": [other]})
        elements = [compare(start=5)]
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1",
                                           "words_sha256": {"t1": receipt.sha256_file(
                                               project / "ingest/t1/words.json")},
                                           "elements": elements})
    rows = resolve_timeline(project).findings
    row = next(row for row in rows if row.rule == "grade_compare_overlap")
    assert row.severity == "violation"
    owner = "agent-transition (lane agent)" if kind == "transition" else "broll-one (lane broll)"
    assert row.message == (f"{owner}, agent-compare (lane agent): grade compare cannot overlap "
                           "overlays, backgrounds, transitions or another grade compare")


@pytest.mark.parametrize("kind", ["zoom", "insert"])
def test_compare_allows_zoom_and_insert_without_counting_a_card(project, kind):
    from test_timeline import insert

    set_grade(project)
    data = json.loads((project / "project.json").read_text())
    data["rules"] = {"elements_at_once_max": 1}
    write(project / "project.json", data)
    if kind == "insert":
        other = insert(project, start={"time": 1}, end={"time": 4}, lane="broll")
        write(project / "timeline/broll.json", {"schema": "yap-studio/timeline@1",
                                               "words_sha256": {}, "elements": [other]})
        elements = [compare()]
    else:
        elements = [compare(), {"id": "agent-zoom", "lane": "agent", "kind": "zoom",
                                "start": {"time": 1}, "end": {"time": 3},
                                "peak_scale": 1.04, "center": [0.5, 0.5]}]
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1",
                                           "words_sha256": {}, "elements": elements})
    assert not any(row.rule in {"grade_compare_overlap", "visual_overlap"}
                   for row in resolve_timeline(project).findings)


@pytest.mark.parametrize("name", ["timeline", "timeline-fragment"])
def test_compare_schema_has_only_timing_fields(name):
    import jsonschema

    from yap_studio.project.model import ProjectFileError, json_schema, parse_document

    data = {"schema": "yap-studio/timeline@1", "words_sha256": {}, "elements": [compare()]}
    if name == "timeline-fragment":
        data.pop("words_sha256")
    jsonschema.validate(data, json_schema(name))
    assert parse_document(name, data).elements[0].kind == "grade_compare"
    for extra in ({"lane": "broll"}, {"sfx": False}, {"look": "film"}):
        bad = {**data, "elements": [{**compare(), **extra}]}
        with pytest.raises(ProjectFileError):
            parse_document(name, bad)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, json_schema(name))


def test_compare_uses_style_grade_and_respects_project_override(project):
    data = json.loads((project / "project.json").read_text())
    data["style"] = "styles/film.json"
    write(project / "styles/film.json", {"schema_version": 1,
                                        "grade": {"look": "film", "slider": True}})
    write(project / "project.json", data)
    write(project / "timeline/agent.json", {"schema": "yap-studio/timeline@1",
                                           "words_sha256": {}, "elements": [compare()]})
    assert not any(row.rule == "grade_compare_grade" for row in resolve_timeline(project).findings)
    set_grade(project, "none")
    assert any(row.rule == "grade_compare_grade" for row in resolve_timeline(project).findings)
