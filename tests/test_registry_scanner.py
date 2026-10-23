"""Namespace literals and pinned plugins share the render preflight rules."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from yap_studio.errors import YapError
from yap_studio.render import hyperframes

NAMESPACES = [
    "http://www.w3.org/2000/svg",
    "http://www.w3.org/1999/xlink",
    "http://www.w3.org/1999/xhtml",
    "http://www.w3.org/XML/1998/namespace",
    "http://www.w3.org/2000/xmlns/",
]


@pytest.mark.parametrize("namespace", NAMESPACES)
def test_namespace_literal_and_attribute_are_allowed(tmp_path: Path, namespace: str):
    (tmp_path / "index.html").write_text(
        f'<div data-namespace="{namespace}"></div>'
        f'<script>document.createElementNS("{namespace}", "svg");</script>'
    )
    hyperframes._scan_composition_tree(tmp_path)


@pytest.mark.parametrize(
    "script",
    [
        'const ns = "http://www.w3.org/2000/svg/../x";',
        'const ns = "http://www.w3.org/TR/";',
        'const ns = "prefixhttp://www.w3.org/2000/svg";',
        'const ns = "http://www.w3.org/2000/svg" + "/other";',
        'fetch("http://www.w3.org/2000/svg");',
        'document.createElementNS("http://www.w3.org/2000/svg", "svg"); fetch("/x");',
    ],
)
def test_other_namespace_urls_and_fetch_are_refused(tmp_path, script):
    (tmp_path / "index.html").write_text(f"<script>{script}</script>")
    with pytest.raises(YapError):
        hyperframes._scan_composition_tree(tmp_path)


@pytest.mark.parametrize("altered", [False, True])
def test_only_pinned_plugin_bytes_skip_scan(tmp_path, monkeypatch, altered):
    original = b"/* synthetic license https://gsap.com */"
    monkeypatch.setattr(
        hyperframes,
        "PINNED_GSAP_PLUGINS",
        {
            "CustomEase.min.js": hashlib.sha256(original).hexdigest(),
        },
        raising=False,
    )
    (tmp_path / "index.html").write_text('<script src="CustomEase.min.js"></script>')
    (tmp_path / "CustomEase.min.js").write_bytes(original + (b" altered" if altered else b""))
    if altered:
        with pytest.raises(YapError, match="https://gsap.com"):
            hyperframes._scan_composition_tree(tmp_path)
    else:
        hyperframes._scan_composition_tree(tmp_path)


def test_plugin_checksum_is_pinned():
    assert hyperframes.PINNED_GSAP_PLUGINS == {
        "CustomEase.min.js": "2dae8ef6daba66ce7b527b1f46a5695ba325b21ad5b4d73e4a843753c65415f6",
    }


@pytest.mark.parametrize(
    "reference",
    [
        "../fonts/local.woff2",
        "fonts/../fonts/local.woff2",
        "%2e%2e/fonts/local.woff2",
    ],
)
def test_nested_item_refuses_parent_segments_even_for_existing_assets(tmp_path, reference):
    (tmp_path / "fonts").mkdir()
    (tmp_path / "fonts/local.woff2").write_bytes(b"synthetic font")
    (tmp_path / "compositions").mkdir()
    (tmp_path / "compositions/fonts").mkdir()
    (tmp_path / "compositions/fonts/local.woff2").write_bytes(b"synthetic font")
    (tmp_path / "compositions/item.html").write_text(
        f"<style>@font-face{{font-family:Local;src:url({reference})}}</style>"
    )
    (tmp_path / "index.html").write_text(
        '<div data-composition-src="compositions/item.html"></div>'
    )
    with pytest.raises(YapError) as caught:
        hyperframes._scan_composition_tree(tmp_path)
    assert caught.value.code == "layer_external_asset"


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize(
    "method",
    [
        "createElementNS",
        "setAttributeNS",
        "getAttributeNS",
        "removeAttributeNS",
        "hasAttributeNS",
        "getElementsByTagNameNS",
        "createAttributeNS",
    ],
)
def test_namespace_literal_is_allowed_as_first_dom_namespace_argument(tmp_path, quote, method):
    source = f'document.{method}( {quote}{NAMESPACES[0]}{quote}, "path");'
    hyperframes._validate_script(source, tmp_path, tmp_path / "item.js")


@pytest.mark.parametrize("namespace", NAMESPACES)
@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("keyword", ["var", "let", "const"])
def test_namespace_declaration_is_allowed_when_all_uses_are_namespace_arguments(
    tmp_path,
    namespace,
    quote,
    keyword,
):
    source = (
        f"{keyword} NS = {quote}{namespace}{quote};\n"
        'const path = document.createElementNS(NS, "circle");\n'
        'path.setAttributeNS(NS, "role", "value");\n'
        'document.getElementsByTagNameNS(NS, "g");'
    )
    hyperframes._validate_script(source, tmp_path, tmp_path / "item.js")


@pytest.mark.parametrize("identifier", ["svg", "$NS", "namespace$"])
def test_namespace_declaration_checks_identifier_tokens(tmp_path, identifier):
    source = (
        f'var {identifier} = "{NAMESPACES[0]}";\n'
        'const identifierSuffix = "unrelated";\n'
        f'document.createElementNS({identifier}, "g");'
    )
    hyperframes._validate_script(source, tmp_path, tmp_path / "item.js")


@pytest.mark.parametrize(
    "tail",
    [
        '; document.createElementNS(NS, "g");',
        ', element = document.createElementNS(NS, "g");',
        '\ndocument.createElementNS(NS, "g");',
        "",
    ],
)
def test_namespace_simple_declaration_terminators_are_allowed(tmp_path, tail):
    hyperframes._validate_script(
        f'var NS = "{NAMESPACES[0]}"{tail}',
        tmp_path,
        tmp_path / "item.js",
    )


@pytest.mark.parametrize(
    "source",
    [
        'const image = new Image(); image.src = "NAMESPACE";',
        'image.href = "NAMESPACE";',
        'var NS = "NAMESPACE"; img.src = NS;',
        'var NS = "NAMESPACE"; document.createElementNS(NS, "g"); fetch(NS);',
        'var NS = "NAMESPACE"; document.createElementNS(NS, "g"); image.src = NS;',
        'var NS = "NAMESPACE"; x = NS + "/a";',
        'var NS = "NAMESPACE"; document.createElementNS(NS + "/a", "g");',
        'var NS = "NAMESPACE"; const url = `${NS}`; image.src = url;',
        'var $NS = "NAMESPACE"; image.src = $NS;',
        'var namespace$ = "NAMESPACE"; image.src = namespace$;',
        'x = "NAMESPACE" + "/a";',
        'const NS = value || "NAMESPACE";',
        'const settings = {namespace: "NAMESPACE"};',
        'document.createElementNS(value || "NAMESPACE", "g");',
        'document.createElementNS("NAMESPACE" + "/a", "g");',
        'var NS = "NAMESPACE"\n + "/a"; document.createElementNS(NS, "g");',
    ],
)
def test_namespace_literals_are_refused_outside_the_surveyed_forms(tmp_path, source):
    source = source.replace("NAMESPACE", NAMESPACES[0])
    with pytest.raises(YapError) as caught:
        hyperframes._validate_script(source, tmp_path, tmp_path / "item.js")
    assert caught.value.code == "layer_network_script"


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize(
    "form",
    [
        'document.createElementNS(URL, "path");',
        'var NS = URL; document.createElementNS(NS, "path");',
        'let NS = URL; document.createElementNS(NS, "path");',
        'const NS = URL; document.createElementNS(NS, "path");',
        "image.src = URL;",
        "const settings = {namespace: URL};",
    ],
)
def test_other_w3_urls_are_refused_in_every_form(tmp_path, quote, form):
    source = form.replace("URL", f"{quote}http://www.w3.org/TR/{quote}")
    with pytest.raises(YapError) as caught:
        hyperframes._validate_script(source, tmp_path, tmp_path / "item.js")
    assert caught.value.code == "layer_network_script"
