"""Classify HyperFrames registry items and write the triage file."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from yap_studio import cards, ffmpeg
from yap_studio.errors import DependencyError, YapError
from yap_studio.hf_registry import catalog, localize
from yap_studio.hf_registry import install as registry_install
from yap_studio.hf_registry.catalog import TRIAGE_SCHEMA
from yap_studio.style import resolve

STYLE = "preset:clean_minimal"
MIN_S = 1.5
FRAMES = 90
STILL_FRAME = 54


class _Metadata(HTMLParser):
    def __init__(self, source: str) -> None:
        super().__init__()
        self.root: dict | None = None
        self.variables: list[dict] = []
        self.elements: list[tuple[str, dict]] = []
        self.in_script = False
        self.feed(source)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        self.in_script = tag == "script"
        if self.root is None and "data-composition-id" in attributes:
            self.root = attributes
        if "data-composition-variables" in attributes:
            declarations = json.loads(attributes["data-composition-variables"])
            if not isinstance(declarations, list) or not all(
                isinstance(rule, dict) and isinstance(rule.get("id"), str)
                and isinstance(rule.get("type"), str) for rule in declarations
            ):
                raise ValueError("expected variable declarations")
            self.variables.extend(declarations)

    def handle_endtag(self, tag: str) -> None:
        self.elements.append(("/" + tag, {}))
        self.in_script = False

    def handle_data(self, data: str) -> None:
        if data.strip() and not self.in_script:
            self.elements.append(("text", {}))

    def handle_decl(self, decl: str) -> None:
        self.elements.append(("declaration", {}))

    def handle_pi(self, data: str) -> None:
        self.elements.append(("instruction", {}))


def _integral(value: object) -> bool:
    return type(value) is int or (type(value) is float and value.is_integer())


def classify(name, manifest, files, source, *, error=None) -> dict:
    """Describe an item's usable content without installing or rendering it."""
    kind = manifest["type"].removeprefix("hyperframes:")
    record = {"type": kind, "use": "none", "reason": ""}
    if kind == "example":
        record["reason"] = "An example project; HyperFrames installs it with init, not add."
        return record
    if manifest.get("deprecated"):
        record["reason"] = "Deprecated upstream."
        return record
    if error is not None:
        record["reason"] = f"HyperFrames add failed: {error[:200]}"
        return record
    try:
        metadata = _Metadata(source or "")
    except (ValueError, TypeError):
        record["reason"] = "Its variable declarations are not valid JSON."
        return record
    if metadata.root is None:
        if (kind == "component" and len(metadata.elements) == 2
                and metadata.elements[0][0] == "script" and metadata.elements[1][0] == "/script"
                and "src" not in metadata.elements[0][1]):
            record.update(use="helper",
                          reason="A script-only snippet that built-in templates can call.")
        else:
            record["reason"] = ("A paste-in snippet without its own composition; "
                                "the adapter mounts compositions only.")
            return record
    slots, preview, variables = {}, {}, {}
    content = [rule for rule in metadata.variables
               if metadata.root is not None and rule.get("role", "content") == "content"]
    pictures = [rule["id"] for rule in content if rule["type"] in ("image", "video")]
    if pictures:
        record["reason"] = "Needs pictures: " + ", ".join(pictures) + "."
        return record
    for rule in content:
        slot = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", rule["id"]).replace("-", "_").lower()
        if re.fullmatch(r"[a-z][a-z0-9_]*", slot) is None or slot in slots:
            continue
        default = rule.get("default")
        if rule["type"] == "string":
            sample = default if isinstance(default, str) and default.strip() else "Sample text"
            slots[slot] = {"type": "string", "required": True, "min_length": 1,
                           "max_length": max(12, len(sample))}
        elif (rule["type"] == "number" and rule.get("role") == "content" and _integral(default)
              and ("step" not in rule or _integral(rule["step"]))):
            low = int(rule["min"]) if _integral(rule.get("min")) else 0
            high = int(rule["max"]) if _integral(rule.get("max")) else max(100, int(default))
            slots[slot] = {"type": "integer", "required": True, "minimum": low, "maximum": high}
            sample = max(low, min(high, int(default)))
        elif rule["type"] == "enum" and rule.get("role") == "content":
            options = [option["value"] if isinstance(option, dict) else option
                       for option in rule.get("options", [])]
            if not options:
                continue
            sample = default if default in options else options[0]
            slots[slot] = {"type": "enum", "required": False, "options": options, "default": sample}
        else:
            continue
        preview[slot] = sample
        variables[rule["id"]] = "{" + slot + "}"
    tags = set(manifest.get("tags", []))
    category = "Words"
    if record["use"] != "helper":
        if "caption-style" in tags:
            record.update(use="caption", reason="Caption look with its own sample words; "
                          "using it for captions needs a caption port.")
        elif not slots:
            record["reason"] = ("No slot for the creator's words or numbers; "
                                "it would show its own sample content.")
            return record
        elif any(rule["type"] == "integer" for rule in slots.values()):
            record["use"], category = "card", "Numbers"
        elif (tags & {"typography", "title-card", "text", "text-effect", "text-effects", "headline"}
              and "lower-third" not in tags):
            record["use"] = "title"
        else:
            record["use"] = "card"
            if "chat" in tags:
                category = "Chat"
            elif tags & {"code", "terminal", "code-animation", "developer"}:
                category = "Editor"
        if record["use"] != "caption":
            parts = []
            for slot_type, label in (("integer", "number"), ("string", "text"), ("enum", "choice")):
                names = [slot for slot, rule in slots.items() if rule["type"] == slot_type]
                if names:
                    parts.append(f"{label} slot{'s' if len(names) > 1 else ''} " + ", ".join(names))
            record["reason"] = record["use"].capitalize() + ": " + "; ".join(parts) + "."
    main = next(file["target"] for file in manifest["files"]
                if file["type"] in ("hyperframes:snippet", "hyperframes:composition"))
    entry = {"type": kind, "upstream_path": f"registry/{kind}s/{name}/{name}.html",
             "main": main, "files": files}
    record.update(entry=entry, check={"status": "pending", "detail": ""})
    if record["use"] == "helper":
        return record
    entry.update(readability="halo", mount="fill", tokens={}, min_s=MIN_S,
                 slots=slots, preview=preview, variables=variables)
    root = metadata.root
    width, height = root.get("data-width") or "", root.get("data-height") or ""
    if (re.fullmatch(r"[0-9]+", width) and re.fullmatch(r"[0-9]+", height)
            and int(width) > 0 and int(height) > 0):
        w, h = int(width), int(height)
        entry.update(mount="scale", stage=[w, h], content=[0, 0, w, h], scale_max=1)
    for token in sorted(catalog.color_tokens(source) - catalog.HOST_TOKENS.keys()):
        lower = token.lower()
        if any(word in lower for word in ("accent", "brand", "highlight", "rim", "glow", "caret")):
            color = "accent"
        elif any(word in lower for word in ("bg", "background", "surface", "panel", "border",
                                             "track", "card")):
            color = "accent_tint"
        else:
            color = "paper"
        entry["tokens"][token] = color
    uses = []
    for rule in slots.values():
        if rule["type"] == "string":
            uses.append(f"text up to {rule['max_length']} characters")
        elif rule["type"] == "integer":
            uses.append(f"a number from {rule['minimum']} to {rule['maximum']}")
        else:
            uses.append("one of " + ", ".join(rule["options"]))
    entry["kind"] = {"category": category,
                     "role": "cards" if record["use"] == "card" else "titles",
                     "label": manifest["title"],
                     "use": ("a caption look with its own sample words"
                             if record["use"] == "caption" else "; ".join(uses))}
    style = resolve(STYLE).data
    entry["min_panel"] = [style["graphics"]["card_width_px"],
                          style["internal"]["key_text_bottom_px"]
                          - style["internal"]["key_text_top_px"] - cards.PAD_TOP - cards.PAD_BOTTOM]
    return record


def _fetch(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except (OSError, ValueError) as exc:
        raise DependencyError(
            f"Could not fetch the HyperFrames registry: {exc}",
            error_type="dependency_error", code="registry_triage_fetch",
            suggested_action="Check the network and run the triage again.",
        ) from exc


def survey(registry_url, *, executable, work, home, timeout_s=300, fetch=None) -> dict[str, dict]:
    """Stage and classify every item at the pinned registry URL."""
    fetch = fetch or _fetch
    index = json.loads(fetch(f"{registry_url}/registry.json"))
    work.mkdir(parents=True, exist_ok=True)
    records = {}
    for item in index["items"]:
        name = item["name"]
        kind = item["type"].removeprefix("hyperframes:")
        manifest = json.loads(fetch(f"{registry_url}/{kind}s/{name}/registry-item.json"))
        if kind == "example":
            records[name] = classify(name, manifest, None, None)
            continue
        with (
            tempfile.TemporaryDirectory(dir=work, prefix="item-") as temporary,
            tempfile.TemporaryDirectory(dir=work, prefix="history-") as history,
        ):
            staging = Path(temporary).resolve()
            try:
                response = registry_install.stage_item(
                    name, executable, staging, registry_url, home=home,
                    history=Path(history), timeout_s=timeout_s,
                )
            except YapError as exc:
                records[name] = classify(name, manifest, None, None, error=exc.message)
                continue
            written = response.get("written", [])
            if not isinstance(written, list):
                written = []
            if any(not isinstance(file, str) or not Path(file).is_absolute()
                   or not Path(file).resolve().is_relative_to(staging) for file in written):
                records[name] = {"type": kind, "use": "none",
                                 "reason": "HyperFrames add wrote outside its staging folder."}
                continue
            files = {Path(file).relative_to(staging).as_posix():
                     hashlib.sha256(Path(file).read_bytes()).hexdigest() for file in written}
            main = next((file["target"] for file in manifest["files"]
                         if file["type"] in ("hyperframes:snippet", "hyperframes:composition")),
                        None)
            if main not in files:
                records[name] = {"type": kind, "use": "none",
                                 "reason": "HyperFrames add did not write its main file."}
                continue
            records[name] = classify(name, manifest, files,
                                     (staging / main).read_text(encoding="utf-8"))
    return dict(sorted(records.items()))


def merge(previous, records) -> dict[str, dict]:
    """Keep completed checks only while an item's use and draft entry are unchanged."""
    for name, record in records.items():
        old = (previous or {}).get(name, {})
        if (record["use"] != "none" and old.get("use") == record["use"]
                and old.get("entry") == record["entry"]
                and old.get("check", {}).get("status", "pending") != "pending"):
            record["check"] = old["check"].copy()
    return records


def verdict(rgba) -> tuple[str, str]:
    """Judge whether the still draws inside the transparent card band."""
    alpha = rgba[cards.PAD_TOP:-cards.PAD_BOTTOM, cards.PAD_X:-cards.PAD_X, 3]
    drawn = np.count_nonzero(alpha > 0)
    if drawn < 100:
        return "fail", "Nothing drew in the card panel at 1.8 s."
    if np.count_nonzero(alpha >= 250) >= alpha.size * 0.9:
        return "fail", "An opaque backdrop covers the card band."
    return "pass", f"Draws on {round(100 * drawn / alpha.size)}% of the panel."


def check(records, *, install, work, stills=None) -> dict[str, dict]:
    """Test each usable draft through the card adapter or offline script localizer."""
    from yap_studio.templates_command import _extract_still

    drafts = {name: {**record["entry"],
                     "status": "helper" if record["use"] == "helper" else "on"}
              for name, record in records.items() if record["use"] != "none"}
    work.mkdir(parents=True, exist_ok=True)
    with catalog.trial(drafts):
        for name in sorted(drafts):
            record = records[name]
            directory = work / name
            movie = directory / f"{name}.mov"
            try:
                registry_install.install_items([name], install=install)
                if record["use"] == "helper":
                    localize.localize(name, directory, install=install)
                    status, detail = "pass", "Its script runs offline."
                else:
                    plan = cards.card_layer_plan("hf:" + name, record["entry"]["preview"],
                                                 resolve(STYLE), frames=FRAMES)
                    cards.render_card(plan, directory, movie, install=install)
                    still = (stills if stills is not None else directory) / f"{name}.png"
                    still.parent.mkdir(parents=True, exist_ok=True)
                    _extract_still(movie, still, STILL_FRAME)
                    raw = ffmpeg.run(
                        ["ffmpeg", "-v", "error", "-i", str(still), "-frames:v", "1",
                         "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"],
                        timeout_s=90, binary=True,
                    ).stdout
                    rgba = np.frombuffer(raw, dtype=np.uint8).reshape(plan.height, plan.width, 4)
                    status, detail = verdict(rgba)
            except (YapError, OSError, ValueError) as exc:
                message = exc.message if isinstance(exc, YapError) else str(exc)
                status, detail = "fail", (message.splitlines() or [""])[0][:300]
            finally:
                shutil.rmtree(directory, ignore_errors=True)
            record["check"] = {"status": status, "detail": detail}
            print(name, record["use"], status, file=sys.stderr)
    return records


def document(records, *, commit, version) -> dict:
    """Build a triage document in registry name order."""
    return {"schema": TRIAGE_SCHEMA, "hyperframes_version": version, "commit": commit,
            "style": STYLE, "items": dict(sorted(records.items()))}


def dumps(document) -> str:
    """Keep each sorted item on one line so registry bumps have small diffs."""
    header = json.dumps({key: value for key, value in document.items() if key != "items"}, indent=2)
    lines = ["    " + json.dumps(name) + ": "
             + json.dumps(record, sort_keys=True, separators=(",", ":"))
             for name, record in sorted(document["items"].items())]
    return header[:-2] + ',\n  "items": {\n' + ",\n".join(lines) + "\n  }\n}\n"
