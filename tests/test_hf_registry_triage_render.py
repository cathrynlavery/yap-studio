"""Render registry drafts through the card adapter with external networking denied."""

from __future__ import annotations

import pytest
from test_hf_registry import CORE
from test_hf_registry import synthetic as synthetic
from test_hf_registry_render import offline_registry_renderer as offline_registry_renderer
from test_hf_registry_render import renderer_install as renderer_install
from test_hf_registry_triage import classify
from test_hf_registry_triage import records as records
from test_hf_registry_triage import registry as registry
from test_hf_registry_triage import registry_add as registry_add

from yap_studio.hf_registry import catalog, install, triage

pytestmark = pytest.mark.render


def test_fixture_cards_pass_and_keep_stills(
    offline_registry_renderer, registry_add, records, tmp_path,
):
    fake, _, audit, _ = registry_add
    selected = {name: records[name] for name in ("titlecard-calm", "conic-progress-ring",
                                                "caption-kinetic-slam")}
    drafts = {name: {**record["entry"], "status": "on"} for name, record in selected.items()}
    with catalog.trial(drafts):
        install.install_items(list(drafts), install=fake)
    triage.check(selected, install=offline_registry_renderer, work=tmp_path / "checks",
                 stills=tmp_path)
    for name, record in selected.items():
        assert record["check"]["status"] == "pass", (name, record["check"])
        still = tmp_path / f"{name}.png"
        assert still.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(audit.read_text().splitlines()) == 3
    assert not list((tmp_path / "checks").iterdir())


def test_empty_composition_fails(offline_registry_renderer, synthetic, tmp_path):
    _, registry_install, _, fake, add, _ = synthetic
    source = (
        f'<script src="{CORE}"></script>'
        '<div data-composition-id="sample" data-duration="3" '
        'data-composition-variables=\'[{"id":"text","type":"string","default":"Empty"}]\'>'
        '</div><script>window.__timelines = window.__timelines || {};'
        'window.__timelines.sample = gsap.timeline({paused:true});</script>'
    )
    main = add(source=source)
    record = classify(source=source)
    record["entry"]["type"] = record["type"] = "block"
    record["entry"]["upstream_path"] = "registry/blocks/sample/sample.html"
    record["entry"]["main"] = main
    record["entry"]["files"] = catalog.item("sample")["files"]
    with catalog.trial({"sample": {**record["entry"], "status": "on"}}):
        registry_install.install_items(["sample"], install=fake)
    triage.check({"sample": record}, install=offline_registry_renderer,
                 work=tmp_path / "checks", stills=tmp_path)
    assert record["check"] == {"status": "fail",
                               "detail": "Nothing drew in the card panel at 1.8 s."}
    assert (tmp_path / "sample.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
