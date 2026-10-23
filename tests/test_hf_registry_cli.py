"""Registry setup and read-only command line reports."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yap_studio import cli
from yap_studio.errors import DependencyError
from yap_studio.render import hyperframes

pytestmark = pytest.mark.usefixtures("_no_real_node")


def modules():
    from yap_studio.hf_registry import catalog, install

    return catalog, install


def test_registry_list_json_and_plain(monkeypatch, capsys):
    catalog, install = modules()
    names = list(catalog.items())
    states = {"ready": names[:1], "missing": names[1:2], "changed": names[2:]}
    monkeypatch.setattr(install, "status", lambda: states)
    monkeypatch.setattr(install, "install_items", lambda **kw: pytest.fail("must not install"))
    assert cli.main(["registry", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        **states,
        "commit": catalog.load_catalog()["registry"]["commit"],
    }
    assert cli.main(["registry", "list"]) == 0
    lines = capsys.readouterr().out
    for state, entries in states.items():
        for name in entries:
            assert f"{name} {state}" in lines


@pytest.mark.parametrize(
    ("states", "line"),
    [
        ({"ready": ["a", "b"], "missing": [], "changed": []}, "registry ready (2 items)"),
        ({"ready": [], "missing": ["a", "b"], "changed": []}, "registry missing 2"),
        ({"ready": [], "missing": ["c"], "changed": ["a", "b"]}, "registry changed: a, b"),
    ],
)
def test_doctor_reports_registry_without_installing(monkeypatch, capsys, states, line):
    _, install = modules()
    monkeypatch.setattr(install, "status", lambda: states)
    monkeypatch.setattr(install, "install_items", lambda **kw: pytest.fail("must not install"))
    monkeypatch.setattr(
        hyperframes, "ensure_installed", lambda **kw: pytest.fail("must not install")
    )
    cli.main(["doctor", "--json"])
    assert json.loads(capsys.readouterr().out)["registry"] == states
    cli.main(["doctor"])
    assert line in capsys.readouterr().out.splitlines()


@pytest.mark.parametrize("failure", [False, True])
def test_renderer_install_runs_registry_after_renderer(monkeypatch, capsys, failure):
    _, install = modules()
    calls = []
    renderer = SimpleNamespace(version=hyperframes.HYPERFRAMES_VERSION)

    def ensure(**kwargs):
        calls.append("renderer")
        return renderer

    def registry(**kwargs):
        assert kwargs["install"] is renderer
        calls.append("registry")
        if failure:
            raise DependencyError(
                "registry failed",
                error_type="dependency_error",
                code="registry_item_changed",
                suggested_action="try setup again",
            )
        return [{"name": "a", "status": "installed", "files": 1}]

    monkeypatch.setattr(hyperframes, "ensure_installed", ensure)
    monkeypatch.setattr(install, "install_items", registry)
    assert cli.main(["renderer", "install"]) == (2 if failure else 0)
    assert calls == ["renderer", "registry"]
    output = capsys.readouterr()
    assert f"Renderer {renderer.version} and browser ready." in output.out
    if failure:
        assert "error: registry failed" in output.err and "fix: try setup again" in output.err
    else:
        assert "HyperFrames registry items ready: 1." in output.out


def test_registry_install_command(monkeypatch, capsys):
    _, install = modules()
    calls = []
    monkeypatch.setattr(
        install,
        "install_items",
        lambda **kw: calls.append(kw) or [{"name": "a", "status": "already", "files": 1}],
    )
    assert cli.main(["registry", "install"]) == 0
    assert len(calls) == 1
    assert "HyperFrames registry items ready: 1." in capsys.readouterr().out
