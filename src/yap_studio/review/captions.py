"""Read caption hit targets from the reviewed render's saved records."""

from collections import Counter
from fractions import Fraction
from pathlib import Path

from yap_studio.captions import _stripped
from yap_studio.faces.clearance import Layout
from yap_studio.feedback.removed import _words
from yap_studio.feedback.versions import VersionManifest, _snapshot_json, version_pieces
from yap_studio.project.model import load_project


def caption_phrases(root: Path, manifest: VersionManifest) -> dict:
    """Pair displayed words and measured boxes with their preserved source ids."""
    empty = {"available": False, "phrases": []}
    if not manifest.snapshots.get("caption_layout"):
        return empty
    layout = _snapshot_json(manifest, "caption_layout")
    Layout.from_dict(layout)
    if not layout.get("measured"):
        return empty
    source = {}
    inputs = version_pieces(manifest).get("inputs", {})
    # Only takes this version's cut read have saved word hashes; unused takes are not checked.
    takes = [take for take in load_project(root / "project.json").takes
             if f"words:{take.id}" in inputs]
    if not takes:
        return empty
    for take in takes:
        words = _words(root, take.id, inputs)
        if words is None:
            return empty
        source.update((word.id, word.text) for word in words.words)
    counts = Counter(_stripped(text) for text in source.values())
    if any(not isinstance(word.get("id"), str) or word["id"] not in source
           or not isinstance(word.get("source_ids"), list) or not word["source_ids"]
           or word["source_ids"][0] != word["id"]
           or any(not isinstance(identifier, str) or identifier not in source
                  for identifier in word["source_ids"])
           for shown in layout["captions"] for word in shown["words"]):
        return empty
    result = []
    for shown in layout["captions"]:
        words = []
        for word in shown["words"]:
            identifier = word["id"]
            heard = source[identifier]
            editable = len(word["source_ids"]) == 1
            words.append({"id": identifier, "text": word["text"], "heard": heard,
                          "start": word["start_s"], "end": word["end_s"],
                          "editable": editable,
                          "reason": None if editable else
                          "This caption joins several words. Fix it with a note.",
                          "repeats": counts[_stripped(heard)]})
        if words:
            result.append({"start": shown["start_frame"] / float(Fraction(manifest.media["fps"])),
                           "end": shown["end_frame"] / float(Fraction(manifest.media["fps"])),
                           "text": shown["text"], "boxes": shown["boxes"], "words": words})
    return {"available": True, "width": layout["width"], "height": layout["height"],
            "phrases": result}
