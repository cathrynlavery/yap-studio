"""9:16 safe areas for placed text and pictures, and HyperFrames layer findings in verify."""

from __future__ import annotations

import pytest
import synth

from yap_studio.checks.caption_layout import read_layout_record, write_layout_record
from yap_studio.checks.safe_area import check_safe_area, edges_crossed
from yap_studio.errors import InputError
from yap_studio.faces import detect
from yap_studio.faces.model import Box
from yap_studio.render.layer_checks import _pass, check_status, gated_findings
from yap_studio.verify import _layer_coverage, verify


@pytest.fixture(autouse=True)
def portable_face_detector(monkeypatch):
    monkeypatch.setattr(detect, "unavailable_reason",
                        lambda: "System detector unavailable in tests")
    monkeypatch.setattr(detect, "default_detector",
                        lambda: pytest.fail("Tests must not invoke the system face detector"))


@pytest.mark.parametrize("box,expected", [
    # Inside every zone.
    (Box(200, 600, 500, 400), {"tiktok": [], "reels": [], "shorts": []}),
    # Yap's caption band sits below every zone's bottom (1248 to 1260 px).
    (Box(350, 1499, 380, 94), {"tiktok": ["bottom"], "reels": ["bottom"], "shorts": ["bottom"]}),
    # TikTok's lower band stops at 780 px for its right-hand buttons; the others allow more.
    (Box(300, 1000, 550, 100), {"tiktok": ["right"], "reels": [], "shorts": []}),
    # Above every band only the top edge counts; side edges apply where a band is.
    (Box(20, 100, 300, 120), {"tiktok": ["top"], "reels": ["top"], "shorts": ["top"]}),
    (Box(20, 300, 300, 120), {"tiktok": ["left"], "reels": ["left"], "shorts": ["left"]}),
])
def test_edges_match_hyperframes_zones(box, expected):
    assert {zone: edges_crossed(box, 1080, 1920, zone) for zone in expected} == expected


def _record(tmp_path, *, width=1080, height=1920, elements=(), captions=(), **extra):
    media = tmp_path / "final.mp4"
    media.write_bytes(b"final bytes")
    write_layout_record(media, {
        "schema": "yap-studio/clearance-layout@1", "width": width, "height": height,
        "measured": True, "reason": None, "elements": list(elements),
        "captions": list(captions), **extra,
    })
    record = read_layout_record(media, width=width, height=height)
    assert record is not None
    return record


def _element(identity, kind, x, y, w=300, h=200):
    return {"id": identity, "kind": kind, "x": x, "y": y, "w": w, "h": h,
            "start_frame": 30, "end_frame": 90}


CAPTION = {"id": "caption-1", "text": "Small steps", "start_frame": 0, "end_frame": 60,
           "words": [{"text": "Small", "start_s": 0.1, "end_s": 0.5},
                     {"text": "steps", "start_s": 0.5, "end_s": 1.0}],
           "boxes": [{"x": 350, "y": 1499, "w": 380, "h": 94}]}


def test_cards_fail_captions_and_pictures_warn_backgrounds_skip(tmp_path):
    record = _record(tmp_path, captions=[CAPTION], elements=[
        _element("hook", "title", 40, 120),
        _element("inside", "card", 300, 600),
        _element("photo", "overlay", 700, 1300),
        _element("backdrop", "background", 0, 0, 1080, 1920),
    ])
    findings, coverage = check_safe_area(record, fps=30, platform="none")
    got = {(f.evidence["target_id"], f.severity) for f in findings}
    assert got == {("hook", "fail"), ("photo", "warn"), ("caption-1", "warn")}
    hook = next(f for f in findings if f.evidence["target_id"] == "hook")
    assert hook.evidence["zones"] == {"tiktok": ["left", "top"], "reels": ["left", "top"]}
    assert "TikTok (left and top edges)" in hook.message and hook.time_s == 1.0
    assert coverage == {"ran": True, "reason": None, "zones": ["tiktok", "reels"]}


def test_shorts_only_with_its_preset(tmp_path):
    # Above the Shorts top edge (288 px) but inside TikTok (240) and Reels (268.8).
    record = _record(tmp_path, elements=[_element("hook", "title", 200, 270, 600, 300)])
    assert check_safe_area(record, fps=30, platform="none")[0] == []
    assert check_safe_area(record, fps=30, platform="reels")[0] == []
    findings, _ = check_safe_area(record, fps=30, platform="shorts")
    assert [f.evidence["zones"] for f in findings] == [{"shorts": ["top"]}]


def test_platform_checks_only_its_zone(tmp_path):
    record = _record(tmp_path, elements=[_element("wide", "card", 300, 1000, 550, 100)])
    assert check_safe_area(record, fps=30, platform="shorts")[0] == []
    findings, coverage = check_safe_area(record, fps=30, platform="tiktok")
    assert [(f.severity, f.evidence["zones"], f.evidence["text_under_tiktok_buttons"])
            for f in findings] == [("warn", {"tiktok": ["right"]}, True)]
    assert "Text under TikTok buttons" in findings[0].message
    assert coverage["zones"] == ["tiktok"]


def test_house_card_width_warns_on_sides(tmp_path):
    # Yap's 960 px card panel (frame x 60-1020) in the key text band, past TikTok's x 120-960.
    record = _record(tmp_path, elements=[_element("stat", "card", 60, 300, 960, 270)])
    [finding] = check_safe_area(record, fps=30, platform="none")[0]
    assert finding.severity == "warn"
    assert finding.evidence["zones"] == {"tiktok": ["left", "right"], "reels": ["left", "right"]}
    assert finding.evidence["text_under_tiktok_buttons"] is False
    assert "Only its sides cross" in finding.message


def test_landscape_render_is_not_checked(tmp_path):
    record = _record(tmp_path, width=1920, height=1080,
                     elements=[_element("hook", "title", 0, 0)])
    assert check_safe_area(record, fps=30, platform="none") == (
        [], {"ran": False, "reason": "not a 9:16 render"})


def _checks(*findings):
    return {"schema": "yap-studio/layer-checks@1", "passes": {
        "layout": {"status": "warnings", "findings": list(findings)},
        "lint": {"status": "skipped", "findings": []},
    }}


def test_gated_findings_keep_only_face_safe_area_and_audio_codes():
    checks = _checks(
        {"severity": "info", "code": "text_over_face", "message": "covers 40% of a face",
         "time": 1.5},
        {"severity": "warning", "code": "outside_safe_zone", "message": "outside", "time": None},
        {"severity": "warning", "code": "clipped_text", "message": "not gated", "time": 0.2},
        {"severity": "error", "code": "audio_src_not_found", "message": "silent", "time": -1},
    )
    found = gated_findings(checks, layer_id="hook", start_frame=60)
    assert [(item["code"], item["layer_time_s"]) for item in found] == [
        ("text_over_face", 1.5), ("outside_safe_zone", 0.0), ("audio_src_not_found", 0.0)]
    assert all(item["start_frame"] == 60 and item["pass"] == "layout" for item in found)
    assert gated_findings({"schema": "other"}, layer_id="x", start_frame=0) == []
    assert gated_findings(None, layer_id="x", start_frame=0) == []


def test_layout_record_refuses_malformed_layer_findings(tmp_path):
    with pytest.raises(InputError, match="layer findings"):
        _record(tmp_path, layer_findings=[{"layer_id": "hook", "code": "text_over_face"}])


def test_verify_fails_on_safe_area_and_layer_findings(tmp_path):
    clip = synth.picture_source(tmp_path / "final.mp4", "testsrc2=s=180x320:r=30", 90)
    finding = gated_findings(_checks(
        {"severity": "info", "code": "text_over_face", "message": "covers 40% of a face",
         "time": 0.5}), layer_id="hook", start_frame=30)
    write_layout_record(clip, {
        "schema": "yap-studio/clearance-layout@1", "width": 180, "height": 320,
        "measured": True, "reason": None, "captions": [],
        "elements": [{"id": "hook", "kind": "title", "x": 4, "y": 4, "w": 60, "h": 30,
                      "start_frame": 30, "end_frame": 60}],
        "layer_findings": finding, "layer_check_status": {"hook": "findings"},
    })
    report = verify(str(clip), asr="none")
    fails = {(item["check"], item["severity"]) for item in report["findings"]
             if item["check"] in {"platform.safe_area", "layers.hyperframes"}}
    assert fails == {("platform.safe_area", "fail"), ("layers.hyperframes", "fail")}
    layer = next(item for item in report["findings"] if item["check"] == "layers.hyperframes")
    assert layer["time_s"] == pytest.approx(1.5)
    assert report["status"] == "fail"
    assert report["checks"]["platform.safe_area"]["ran"] is True
    assert report["checks"]["layers.hyperframes"]["ran"] is True


def test_nttd_title_card_panel_edge_warns(tmp_path):
    # The NTTD opener card as drawn in the 04-the-terminal v6 opener: its panel crosses
    # TikTok's side edges and the Reels top by up to 10 px, while its padded text is inside.
    record = _record(tmp_path, elements=[_element("nttd-title", "title", 110, 262, 860, 343)])
    findings, _ = check_safe_area(record, fps=30, platform="none")
    assert [(f.severity, f.evidence["zones"], f.evidence["text_area_crosses"])
            for f in findings] == [
        ("warn", {"tiktok": ["left", "right"], "reels": ["top"]}, False)]
    assert "outer 24 px" in findings[0].message


@pytest.mark.parametrize("y,h", [(10, 40), (1800, 40), (1700, 30)])
def test_short_title_past_top_or_bottom_fails(tmp_path, y, h):
    # Shorter than twice the panel inset, entirely above TikTok's top or below Reels' bottom.
    record = _record(tmp_path, elements=[_element("tag", "title", 200, y, 500, h)])
    [finding] = check_safe_area(record, fps=30, platform="none")[0]
    assert finding.severity == "fail"


def test_gated_finding_after_the_display_limit_is_kept(tmp_path):
    other = [{"severity": "warning", "code": "clipped_text", "message": f"clip {index}"}
             for index in range(30)]
    gated = {"severity": "warning", "code": "outside_safe_zone", "message": "outside"}
    result = _pass({"ok": True, "findings": [*other, gated]}, tmp_path, "layer", "layout")
    assert len(result["findings"]) == 26 and result["findings"][-1]["code"] == "outside_safe_zone"
    assert result["findings_truncated"] == 5
    found = gated_findings({"schema": "yap-studio/layer-checks@1",
                            "passes": {"layout": result}}, layer_id="card", start_frame=0)
    assert [item["code"] for item in found] == ["outside_safe_zone"]


def test_layer_check_status_and_coverage():
    assert check_status({"schema": "yap-studio/layer-checks@1", "status": "incomplete"}) == \
        "incomplete"
    assert check_status(None) == "missing"
    assert _layer_coverage(None)["ran"] is False
    assert _layer_coverage({}) == {"ran": False, "reason": "no caption or graphics layers"}
    assert _layer_coverage({"captions": "missing"}) == {
        "ran": False, "reason": "HyperFrames checks did not run on any layer"}
    assert _layer_coverage({"captions": "passed", "hook": "incomplete"}) == {
        "ran": True, "reason": None, "note": "not fully checked: hook (incomplete)"}


def test_layout_record_refuses_unknown_layer_status(tmp_path):
    with pytest.raises(InputError, match="layer_check_status"):
        _record(tmp_path, layer_check_status={"hook": "fine"})
