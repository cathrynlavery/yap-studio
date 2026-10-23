"""Pinned registry metadata, isolated installs, and offline localization."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from yap_studio.errors import DependencyError, ValidationError, YapError
from yap_studio.render import fonts, hyperframes, user_fonts

pytestmark = pytest.mark.usefixtures("_no_real_node")
COMMIT = "d15a274c198cf18bc1df177cbe20df95392a8947"
URL = f"https://raw.githubusercontent.com/heygen-com/hyperframes/{COMMIT}/registry"
PLUGIN = b"/* synthetic license https://gsap.com */\nwindow.CustomEase = {};"
CORE = "https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"
CUSTOM = "https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/CustomEase.min.js"


def registry_modules():
    from yap_studio.hf_registry import catalog, install, localize

    return catalog, install, localize


def test_helper_localizes_only_script_and_is_not_a_card(synthetic, tmp_path):
    from yap_studio.hf_registry import cards

    catalog, install, localize, renderer, add, _ = synthetic
    body = 'window.attachEchoSmear = function () {};'
    main = add("sample-helper", source=f'<!-- synthetic header --><script>{body}</script>')
    catalog.item("sample-helper")["status"] = "helper"
    install.install_items(install=renderer)
    assert install.status()["ready"] == ["sample-helper"]
    assert "hf:sample-helper" not in cards.kinds()
    with pytest.raises(ValidationError):
        cards.item_name("hf:sample-helper")
    result = localize.localize("sample-helper", tmp_path / "host", install=renderer)
    script = (tmp_path / "host" / result.main).read_text()
    assert result.main.endswith(".js")
    assert "Modified by Yap Studio" in script
    assert body in script and "<script>" not in script and "synthetic header" not in script
    assert "http" not in script
    assert (install.item_files("sample-helper") / main).read_text().startswith("<!-- synthetic")
    (install.item_dir("sample-helper") / main).write_text("changed")
    assert install.installed("sample-helper") is None


@pytest.mark.parametrize("field", [None, "kind", "slots", "preview"])
def test_catalog_helper_contract(tmp_path, monkeypatch, field):
    catalog, _, _ = registry_modules()
    data = copy.deepcopy(catalog.load_catalog())
    entry = {"type": "component", "status": "helper",
             "upstream_path": "registry/components/sample-helper/sample-helper.html",
             "main": "compositions/components/sample-helper.html",
             "files": {"compositions/components/sample-helper.html": "0" * 64}}
    if field:
        entry[field] = {}
    data["items"]["sample-helper"] = entry
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        if field:
            with pytest.raises(ValidationError, match=field):
                catalog.load_catalog()
        else:
            assert catalog.load_catalog()["items"]["sample-helper"] == entry
    finally:
        catalog.load_catalog.cache_clear()


def test_helper_keeps_legacy_card_catalog_bytes(monkeypatch):
    catalog, _, _ = registry_modules()
    original = (b'{\n  "items": {\n    "sample": {"status": "on", "kind": {"label": "Sample"}}'
                b'\n  }\n}\n')
    source = original.replace(b'\n  }',
        b',\n    "sample-helper": {"status": "helper"}\n  }', 1)
    monkeypatch.setattr(catalog, "load_catalog", lambda: json.loads(source))
    assert catalog.card_catalog_source(source) == original
    assert catalog.card_catalog() == json.loads(original)


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog, install, localize = registry_modules()
    original = copy.deepcopy(catalog.load_catalog())
    original["items"] = {}
    payloads: dict[str, dict[str, str]] = {}
    audit = tmp_path / "calls.jsonl"
    control = tmp_path / "control.json"
    executable = tmp_path / "fake-hyperframes"
    configuration = {
        "registry": URL,
        "paths": {
            "blocks": "compositions",
            "components": "compositions/components",
            "assets": "assets",
        },
    }
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        f"control = json.loads(Path({str(control)!r}).read_text())\n"
        "name = sys.argv[2]\nroot = Path(sys.argv[4])\n"
        "assert sys.argv[1:] == ['add', name, '--dir', str(root), '--json', '--no-clipboard']\n"
        f"assert os.environ['HOME'] == {str(install.registry_home())!r}\n"
        "assert os.environ['HYPERFRAMES_NO_TELEMETRY'] == '1'\n"
        "assert Path(os.environ['HYPERFRAMES_HISTORY_ROOT']).is_dir()\n"
        f"expected_config = {configuration!r}\n"
        "assert json.loads((root / 'hyperframes.json').read_text()) == expected_config\n"
        f"with Path({str(audit)!r}).open('a') as stream:\n"
        "    stream.write(json.dumps({'name': name, 'history': "
        "os.environ['HYPERFRAMES_HISTORY_ROOT']}) + '\\n')\n"
        "written = []\n"
        "for relative, content in control['payloads'][name].items():\n"
        "    target = root / relative\n    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_text(content + ('changed' if control.get('mode') == 'hash' else ''))\n"
        "    written.append(str(target))\n"
        "mode = control.get('mode')\n"
        "if mode == 'extra':\n"
        "    (root / 'extra.js').write_text('extra')\n"
        "    written.append(str(root / 'extra.js'))\n"
        "if mode == 'missing': written = []\n"
        "if mode == 'outside': written = [str(root.parent / 'outside.html')]\n"
        "if mode == 'symlink':\n"
        "    target = root / next(iter(control['payloads'][name]))\n"
        "    target.unlink()\n"
        "    target.symlink_to(root.parent / 'outside.html')\n"
        "print(json.dumps({'ok': mode not in ['error', 'exit'], 'written': written, "
        "'error': 'synthetic add failed'}))\n"
        "sys.exit(7 if mode == 'exit' else 0)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    renderer = hyperframes.RendererInstall(
        hyperframes.HYPERFRAMES_VERSION, tmp_path / "renderer", executable, None, "v26.1.0"
    )
    plugin = renderer.prefix / "node_modules/gsap/dist/CustomEase.min.js"
    plugin.parent.mkdir(parents=True)
    plugin.write_bytes(PLUGIN)
    monkeypatch.setattr(
        hyperframes,
        "PINNED_GSAP_PLUGINS",
        {
            "CustomEase.min.js": hashlib.sha256(PLUGIN).hexdigest(),
        },
    )
    monkeypatch.setattr(catalog, "load_catalog", lambda: original)
    monkeypatch.setattr(hyperframes, "ensure_installed", lambda **kwargs: renderer)

    def add(name="sample", source="<div>synthetic item</div>", *, mode=None, files=None):
        main = f"compositions/{name}.html"
        contents = files or {main: source}
        payloads[name] = contents
        original["items"][name] = {
            "type": "block",
            "upstream_path": f"registry/blocks/{name}/{name}.html",
            "main": main,
            "files": {p: hashlib.sha256(s.encode()).hexdigest() for p, s in contents.items()},
            "status": "pilot",
        }
        control.write_text(json.dumps({"payloads": payloads, "mode": mode}))
        return main

    return catalog, install, localize, renderer, add, audit


def test_catalog_shipped_items():
    catalog, _, _ = registry_modules()
    data = catalog.load_catalog()
    assert data["schema"] == "yap-studio/hf-registry-catalog@1"
    assert data["hyperframes_version"] == hyperframes.HYPERFRAMES_VERSION
    assert data["registry"]["commit"] == COMMIT
    assert catalog.registry_url() == URL
    assert set(catalog.items()) == {"char-slam-explode", "conic-progress-ring", "echo-smear"}
    for name, entry in catalog.items().items():
        assert catalog.item(name) == entry
        assert entry["status"] == ("helper" if name == "echo-smear" else "on")
        assert entry["upstream_path"] == f"registry/{entry['type']}s/{name}/{name}.html"
        assert entry["main"] in entry["files"]
        for file, digest in entry["files"].items():
            assert not Path(file).is_absolute() and ".." not in Path(file).parts
            assert re.fullmatch("[0-9a-f]{64}", digest)
    with pytest.raises(
        ValidationError, match="char-slam-explode.*conic-progress-ring"
    ):
        catalog.item("unknown")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("hyperframes_version", "0.0.1", "0.0.1.*0.8.184"),
        ("commit", "bad", "commit"),
        ("url", "https://example.com/registry", "url"),
        ("path", "../escape.html", "path"),
        ("hash", "bad", "hash"),
        ("type", "wrong", "type"),
        ("main", "missing.html", "main"),
        ("upstream_path", "registry/blocks/other/other.html", "upstream_path"),
        ("status", "wrong", "status"),
    ],
)
def test_catalog_rejects_invalid_metadata(tmp_path, monkeypatch, field, value, message):
    catalog, _, _ = registry_modules()
    data = copy.deepcopy(catalog.load_catalog())
    entry = data["items"]["char-slam-explode"]
    if field == "hyperframes_version":
        data[field] = value
    elif field in {"commit", "url"}:
        data["registry"][field] = value
    elif field == "path":
        entry["files"] = {value: "0" * 64}
    elif field == "hash":
        entry["files"][entry["main"]] = value
    else:
        entry[field] = value
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        with pytest.raises(ValidationError, match=message):
            catalog.load_catalog()
    finally:
        catalog.load_catalog.cache_clear()


def test_catalog_accepts_future_item_keys(tmp_path, monkeypatch):
    catalog, _, _ = registry_modules()
    data = copy.deepcopy(catalog.load_catalog())
    data["items"]["char-slam-explode"]["label"] = "Future label"
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(catalog, "CATALOG_PATH", source)
    catalog.load_catalog.cache_clear()
    try:
        assert catalog.load_catalog()["items"]["char-slam-explode"]["label"] == "Future label"
    finally:
        catalog.load_catalog.cache_clear()


def test_install_publishes_once_and_isolates_home(synthetic, tmp_path, monkeypatch):
    catalog, install, _, renderer, add, audit = synthetic
    home = tmp_path / "untouched-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    main = add()
    assert install.item_dir("sample") == install.items_root() / "sample"
    assert install.items_root().name == f"hyperframes-{renderer.version}-{COMMIT[:12]}"
    assert install.install_items() == [{"name": "sample", "status": "installed", "files": 1}]
    root = install.item_files("sample")
    metadata = json.loads((root / "install.json").read_text())
    assert metadata == {
        "schema": "yap-studio/hf-registry-install@1",
        "name": "sample",
        "commit": COMMIT,
        "files": catalog.item("sample")["files"],
    }
    assert (root / main).read_text() == "<div>synthetic item</div>"
    assert install.install_items(["sample"], install=renderer) == [
        {"name": "sample", "status": "already", "files": 1}
    ]
    calls = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(calls) == 1
    assert not Path(calls[0]["history"]).exists()
    assert list(home.iterdir()) == []
    assert not list(install.items_root().glob(".staging-*"))


@pytest.mark.parametrize("mode", ["hash", "extra", "missing", "outside", "symlink"])
def test_install_refuses_changed_payload_and_cleans_staging(synthetic, mode):
    _, install, _, renderer, add, _ = synthetic
    main = add(mode=mode)
    with pytest.raises(DependencyError) as caught:
        install.install_items(install=renderer)
    assert caught.value.code == "registry_item_changed"
    assert "sample" in caught.value.message
    assert main in caught.value.message or "extra.js" in caught.value.message or "outside" in mode
    assert not install.item_dir("sample").exists()
    assert not list(install.items_root().glob(".staging-*"))


@pytest.mark.parametrize("mode", ["error", "exit"])
def test_install_reports_cli_error(synthetic, mode):
    _, install, _, renderer, add, _ = synthetic
    add(mode=mode)
    with pytest.raises(DependencyError, match="synthetic add failed"):
        install.install_items(install=renderer)
    assert not install.item_dir("sample").exists()
    assert not list(install.items_root().glob(".staging-*"))


def test_status_and_item_files_distinguish_missing_and_changed(synthetic):
    _, install, _, renderer, add, _ = synthetic
    add("ready")
    add("missing")
    main = add("changed")
    assert install.status() == {
        "ready": [],
        "missing": ["ready", "missing", "changed"],
        "changed": [],
    }
    with pytest.raises(DependencyError) as caught:
        install.item_files("missing")
    assert caught.value.code == "registry_item_missing"
    assert "yap-studio renderer install" in caught.value.suggested_action
    install.install_items(["ready", "changed"], install=renderer)
    (install.item_dir("changed") / main).write_text("altered")
    assert install.installed("changed") is None
    assert install.status() == {"ready": ["ready"], "missing": ["missing"], "changed": ["changed"]}
    with pytest.raises(DependencyError) as caught:
        install.item_files("changed")
    assert caught.value.code == "registry_item_changed"
    assert str(install.item_dir("changed")) in caught.value.suggested_action


@pytest.mark.parametrize("change", ["metadata", "file-symlink", "folder-symlink"])
def test_installed_rejects_metadata_and_escaping_symlinks(synthetic, tmp_path, change):
    _, install, _, renderer, add, _ = synthetic
    main = add()
    install.install_items(install=renderer)
    root = install.item_dir("sample")
    if change == "metadata":
        metadata = json.loads((root / "install.json").read_text())
        metadata["commit"] = "0" * 40
        (root / "install.json").write_text(json.dumps(metadata))
    elif change == "file-symlink":
        outside = tmp_path / "outside.html"
        outside.write_bytes((root / main).read_bytes())
        (root / main).unlink()
        (root / main).symlink_to(outside)
    else:
        outside = tmp_path / "outside-item"
        root.rename(outside)
        root.symlink_to(outside, target_is_directory=True)
    assert install.installed("sample") is None
    assert install.status()["changed"] == ["sample"]


def item_html(shape):
    content = (
        f'<script src="{CORE}"></script>\n<script src="{CUSTOM}"></script>\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700'
        '&amp;family=Space+Grotesk&amp;family=JetBrains+Mono" rel="stylesheet">\n'
        '<style>@import url("https://fonts.googleapis.com/css2?family=Montserrat");\n'
        "p {font-family: Montserrat, system-ui, sans-serif;}\n"
        '.space {font-family: "Space Grotesk", sans-serif;}</style>\n'
        "<p style=\"font-family: 'JetBrains Mono', monospace\">Test</p>\n"
        '<script>document.createElementNS("http://www.w3.org/2000/svg", "svg");</script>\n'
    )
    if shape == "document":
        content = f"<!doctype html><html><head>{content}</head><body></body></html>"
    elif shape == "template":
        content = f"<html data-composition-id='test'><template>{content}</template></html>"
    return "<!-- hyperframes-registry-item: sample -->\n" + content


@pytest.mark.parametrize("shape", ["document", "template", "fragment"])
def test_localize_shapes_aliases_fonts_and_preserves_cache(synthetic, tmp_path, shape):
    _, install, localize, renderer, add, _ = synthetic
    source = item_html(shape)
    main = add(source=source)
    install.install_items(install=renderer)
    dest = tmp_path / "localized"
    result = localize.localize("sample", dest, fonts_by_role={"cards": "Gelasio"}, install=renderer)
    text = (dest / result.main).read_text()
    assert result.main == main
    assert set(result.families) == {"Montserrat", "Space Grotesk", "JetBrains Mono"}
    assert result.plugins == ("CustomEase.min.js",)
    assert (
        CORE not in text and "fonts.googleapis.com" not in text and "fonts.gstatic.com" not in text
    )
    assert "CustomEase.min.js" not in text
    assert (dest / "CustomEase.min.js").read_bytes() == PLUGIN
    assert "data-yap-fonts" not in text
    assert "<style" not in result.font_css
    for family, target in [
        ("Montserrat", "Gelasio"),
        ("Space Grotesk", "Gelasio"),
        ("JetBrains Mono", "Space Mono"),
    ]:
        for face in fonts.BUNDLED[target].faces:
            assert f'font-family: "{family}"' in result.font_css
            assert f'url("fonts/{face.file}")' in result.font_css
            assert (dest / "fonts" / face.file).read_bytes() == (
                fonts.FONTS_DIR / face.file
            ).read_bytes()
            assert f"font-weight: {face.weight}" in result.font_css
            assert f"font-style: {face.style}" in result.font_css
    assert not ((dest / result.main).parent / "fonts").exists()
    assert not ((dest / result.main).parent / "CustomEase.min.js").exists()
    head = localize.host_head(result)
    assert head == (
        '<script src="gsap.min.js"></script>\n'
        '<script src="CustomEase.min.js"></script>\n'
        f"<style data-yap-fonts>\n{result.font_css}\n</style>"
    )
    hyperframes._validate_css(result.font_css, dest, dest / "index.html")
    parser = hyperframes._CompositionParser(dest, dest / "index.html")
    parser.feed(f"<!doctype html><html><head>{head}</head><body></body></html>")
    parser.close()
    assert text.startswith(
        "<!-- hyperframes-registry-item: sample -->\n<!-- Modified by Yap Studio:"
    )
    assert "registry/blocks/sample/sample.html" in text and COMMIT in text
    assert "Copyright 2026 HeyGen, Inc., Apache License 2.0." in text
    assert (install.item_files("sample") / main).read_text() == source
    hyperframes.scan_files(dest, dest.rglob("*"))


def test_localize_creator_font(synthetic, tmp_path, monkeypatch):
    _, install, localize, renderer, add, _ = synthetic
    creator = tmp_path / "creator-fonts"
    creator.mkdir()
    face = user_fonts.UserFont("Creator", "user-creator.woff2", "italic", "300 800", "0" * 64)
    (creator / face.file).write_bytes(b"synthetic font bytes")
    monkeypatch.setattr(user_fonts, "find", lambda name: [face] if name == "Creator" else [])
    monkeypatch.setattr(user_fonts, "user_fonts_dir", lambda: creator)
    add(source='<p style="font-family: Montserrat">Test</p>')
    install.install_items(install=renderer)
    dest = tmp_path / "localized"
    result = localize.localize("sample", dest, fonts_by_role={"cards": "Creator"}, install=renderer)
    text = (dest / result.main).read_text()
    assert "data-yap-fonts" not in text
    assert 'font-family: "Montserrat"' in result.font_css
    assert "font-weight: 300 800" in result.font_css and "font-style: italic" in result.font_css
    assert (dest / "fonts" / face.file).read_bytes() == b"synthetic font bytes"


@pytest.mark.parametrize(
    ("source", "code"),
    [
        (
            f'<script src="{CORE.replace("3.14.2", "3.15.0")}"></script>',
            "registry_item_gsap_version",
        ),
        (
            f'<script src="{CUSTOM.replace("CustomEase", "Unknown")}"></script>',
            "registry_item_gsap_plugin",
        ),
        ('<script src="https://example.com/x.js"></script>', "registry_item_external"),
    ],
)
def test_localize_refuses_unsupported_or_external_scripts(synthetic, tmp_path, source, code):
    _, install, localize, renderer, add, _ = synthetic
    main = add(source=source)
    install.install_items(install=renderer)
    with pytest.raises(YapError) as caught:
        localize.localize("sample", tmp_path / "localized", install=renderer)
    assert caught.value.code == code
    assert "sample" in caught.value.message
    if code == "registry_item_external":
        assert main + ":" in caught.value.message
        assert "https://example.com/x.js" in caught.value.message
        assert "cannot render offline" in caught.value.suggested_action


def test_localize_refuses_changed_plugin(synthetic, tmp_path):
    _, install, localize, renderer, add, _ = synthetic
    add(source=item_html("document"))
    install.install_items(install=renderer)
    (renderer.prefix / "node_modules/gsap/dist/CustomEase.min.js").write_bytes(b"changed")
    with pytest.raises(DependencyError) as caught:
        localize.localize("sample", tmp_path / "localized", install=renderer)
    assert caught.value.code == "registry_item_gsap_plugin"


@pytest.mark.parametrize("target", ["fonts", "plugin"])
def test_localize_refuses_destination_symlink_escape(synthetic, tmp_path, target):
    _, install, localize, renderer, add, _ = synthetic
    add(source=item_html("document"))
    install.install_items(install=renderer)
    dest = tmp_path / "localized"
    dest.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if target == "fonts":
        (dest / "fonts").symlink_to(outside, target_is_directory=True)
    else:
        (dest / "CustomEase.min.js").symlink_to(outside / "plugin.js")
    with pytest.raises(YapError):
        localize.localize("sample", dest, install=renderer)
    assert list(outside.iterdir()) == []


def test_localize_default_font_and_font_urls_without_declarations(synthetic, tmp_path):
    _, install, localize, renderer, add, _ = synthetic
    add(
        source=(
            '<style>@import "https://fonts.googleapis.com/css?family=Only+In+Import:400";</style>'
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link href="https://fonts.googleapis.com/css2?family=Only+In+Link" rel="stylesheet">'
            '<script>const a = {fontFamily: "Script Family, sans-serif"}; '
            'const b = {"font-family": "Other Script"};</script>'
            '<div style="font-family:var(--font),system-ui,inherit,unset,initial"></div>'
        )
    )
    install.install_items(install=renderer)
    dest = tmp_path / "localized"
    result = localize.localize("sample", dest, install=renderer)
    assert set(result.families) == {
        "Only In Import",
        "Only In Link",
        "Script Family",
        "Other Script",
    }
    text = (dest / result.main).read_text()
    assert "fonts.googleapis.com" not in text
    assert "geist-variable.woff2" in result.font_css
    assert "geist-variable.woff2" not in text


def test_localize_component_paths(synthetic, tmp_path):
    catalog, install, localize, renderer, add, _ = synthetic
    main = "compositions/components/sample.html"
    add(files={main: item_html("template")})
    entry = catalog.item("sample")
    entry.update(
        type="component", main=main, upstream_path="registry/components/sample/sample.html"
    )
    install.install_items(install=renderer)
    dest = tmp_path / "localized"
    result = localize.localize("sample", dest, install=renderer)
    text = (dest / result.main).read_text()
    assert "CustomEase.min.js" not in text
    assert "data-yap-fonts" not in text
    assert 'url("fonts/geist-variable.woff2")' in result.font_css
    assert (dest / "CustomEase.min.js").read_bytes() == PLUGIN
    assert (dest / "fonts/geist-variable.woff2").is_file()
    assert not (dest / main).parent.joinpath("fonts").exists()
    assert "../" not in text


def test_localize_scans_other_copied_files(synthetic, tmp_path):
    _, install, localize, renderer, add, _ = synthetic
    add(
        files={
            "compositions/sample.html": "<p>Text</p>",
            "assets/hidden.css": "p{background:url(https://example.com/hidden.png)}",
        }
    )
    install.install_items(install=renderer)
    with pytest.raises(YapError) as caught:
        localize.localize("sample", tmp_path / "localized", install=renderer)
    assert caught.value.code == "registry_item_external"
    assert "assets/hidden.css:1" in caught.value.message
    assert "https://example.com/hidden.png" in caught.value.message


def test_concurrent_installs_run_add_once(synthetic):
    from concurrent.futures import ThreadPoolExecutor

    _, install, _, renderer, add, audit = synthetic
    add()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(install.install_items, install=renderer) for _ in range(2)]
        results = [future.result() for future in futures]
    assert sorted(result[0]["status"] for result in results) == ["already", "installed"]
    assert len(audit.read_text().splitlines()) == 1


def test_localize_multiline_font_stack(synthetic, tmp_path):
    _, install, localize, renderer, add, _ = synthetic
    add(
        source=(
            '<style>p {font-family: Montserrat,\n "Space Grotesk",\n '
            "JetBrains Mono, var(--family), sans-serif;}</style>"
        )
    )
    install.install_items(install=renderer)
    result = localize.localize("sample", tmp_path / "localized", install=renderer)
    assert set(result.families) == {"Montserrat", "Space Grotesk", "JetBrains Mono"}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("p{font-family:var(--font-display, Inter, system-ui, sans-serif)}", {"Inter"}),
        (
            'p{font-family:var(--font-display)} :root{--font-display:"Space Grotesk",sans-serif}',
            {"Space Grotesk"},
        ),
        (
            ':root{--font-display:var(--brand, var(--fallback, "Space Grotesk", sans-serif))}'
            "p{font-family:var(--font-display)}",
            {"Space Grotesk"},
        ),
        (
            ':root{--font-display:var(--brand, "Inter", sans-serif);'
            "--brand:var(--fallback, JetBrains Mono, monospace)}"
            "p{font-family:var(--font-display)}",
            {"Inter", "JetBrains Mono"},
        ),
        (
            "p{font-family:\"Space Grotesk\", Inter, 'JetBrains Mono', sans-serif}",
            {"Space Grotesk", "Inter", "JetBrains Mono"},
        ),
        (
            "p{font-family:var(--font-display, serif, sans-serif, monospace, cursive, fantasy,"
            "system-ui, ui-sans-serif, ui-serif, ui-monospace, ui-rounded, emoji, math, fangsong,"
            "-apple-system, BlinkMacSystemFont, inherit, initial, unset, revert, revert-layer)}",
            set(),
        ),
        (
            ":root{--font-display:var(--font-display, Inter)}p{font-family:var(--font-display)}",
            {"Inter"},
        ),
        (':root{--unrelated:"Unused Font"}p{font-family:Inter !important}', {"Inter"}),
        ('p{font-family:"A, B", sans-serif}', {"A, B"}),
    ],
)
def test_font_families_resolve_variables_and_filter_generics(source, expected):
    _, _, localize = registry_modules()
    assert set(localize._families(source)) == expected


def test_font_shorthand_is_not_requested_by_pinned_compiler():
    _, _, localize = registry_modules()
    # HyperFrames 0.8.181 only requests font-family and data-font-family declarations.
    assert localize._families('p{font:italic 700 24px/1.2 "Shorthand Only", sans-serif}') == []


def test_localize_resolves_font_properties_across_style_surfaces(synthetic, tmp_path):
    _, install, localize, renderer, add, _ = synthetic
    add(
        source=(
            "<html><head><style>p{font-family:var(--font-display)}"
            "h1{font-family:var(--title-font)}</style>"
            '<style>:root{--title-font:var(--brand, "Space Grotesk", sans-serif);'
            "--unused-font:Unused}</style></head><body>"
            '<p style="--font-display:Inter,sans-serif">Words</p>'
            "<h1>Title</h1>"
            '<div data-font-family="var(--mono, JetBrains Mono, monospace)">Code</div>'
            "</body></html>"
        )
    )
    install.install_items(install=renderer)
    dest = tmp_path / "localized"
    result = localize.localize("sample", dest, install=renderer)
    assert set(result.families) == {"Inter", "Space Grotesk", "JetBrains Mono"}
    text = (dest / result.main).read_text()
    assert "data-yap-fonts" not in text
    for family in result.families:
        assert f'font-family: "{family}";' in result.font_css
    hyperframes.scan_files(dest, dest.rglob("*"))


def test_font_families_ignore_comments_without_changing_quoted_names():
    _, _, localize = registry_modules()
    source = (
        ":root{--display:Inter; /* --display:Ignored; */}"
        'p{font-family:var(--display), "Font /* Label */", sans-serif}'
    )
    assert set(localize._families(source)) == {"Inter", "Font /* Label */"}


def test_host_head_keeps_plugin_order_and_omits_empty_font_style():
    _, _, localize = registry_modules()
    localized = localize.Localized(
        main="compositions/sample.html",
        families=(),
        plugins=("First.min.js", "Second.min.js"),
        font_css="",
    )
    assert localize.host_head(localized) == (
        '<script src="gsap.min.js"></script>\n'
        '<script src="First.min.js"></script>\n'
        '<script src="Second.min.js"></script>'
    )


def test_host_head_without_fonts_or_plugins(synthetic, tmp_path):
    _, install, localize, renderer, add, _ = synthetic
    add()
    install.install_items(install=renderer)
    localized = localize.localize("sample", tmp_path / "localized", install=renderer)
    assert localized.font_css == ""
    assert localized.plugins == ()
    assert localize.host_head(localized) == '<script src="gsap.min.js"></script>'
