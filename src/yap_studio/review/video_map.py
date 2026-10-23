"""Read-only output-time lanes from immutable review version records."""

from dataclasses import replace
from fractions import Fraction

from yap_studio.feedback.versions import (
    VersionManifest,
    _snapshot_json,
    version_cut_words,
    version_layers,
    version_pieces,
)
from yap_studio.project.model import Anchor, parse_document
from yap_studio.style import resolve, resolve_style_data

LANE_NAMES = ("Story", "Captions", "Titles and Text hook", "Cards", "B-roll and backgrounds",
              "Moments", "Music and sounds")
KINDS = {
    "caption": (1, "caption"), "title": (2, "element"), "chapter": (2, "element"),
    "graphic": (3, "element"), "insert": (4, "video"), "overlay": (4, "element"),
    "background": (4, "video"), "zoom": (5, "video"), "moment": (5, "video"),
    "music_level": (6, "video"), "sound": (6, "video"),
}


def video_map(manifest: VersionManifest | None) -> list[dict]:
    """Keep absent records empty and never resolve against live project inputs."""
    is_cut = manifest is not None and manifest.media["source"] == "cut/cut.mp4"
    lanes = [{"name": name, "items": []} for name in (LANE_NAMES[:2] if is_cut else LANE_NAMES)]
    if manifest is None:
        return lanes
    duration = manifest.media["duration_s"]

    def add(lane, start, end, label, target):
        if lane >= len(lanes) or end < 0 or start >= duration:
            return
        start, end = max(0, start), min(duration, end)
        if end < start:
            return
        lanes[lane]["items"].append({"start": start, "end": end,
                                    "label": " ".join(label.split())[:80], "target": target})

    if manifest.snapshots.get("cut_pieces"):
        saved = version_pieces(manifest)
        fps = Fraction(saved["fps"])
        for index, piece in enumerate(saved["pieces"]):
            start = float(piece["out_f0"] / fps)
            if index:
                add(0, start, start, "Join", "story")
            add(0, start, float((piece["out_f0"] + piece["n_frames"]) / fps),
                piece["segment"], "story")

    authored = {}
    timeline = manifest.snapshots.get("timeline")
    for entry in timeline["lanes"].values() if timeline and not is_cut else ():
        document = _snapshot_json(replace(manifest, snapshots={"lane": entry}), "lane")
        for item in parse_document("timeline", document).elements:
            authored[item.id] = item

    def label(item):
        fields = item.fields
        return (" ".join(fields.get("lines", [])) or fields.get("title")
                or fields.get("text") or (fields.get("style") or fields.get("sound")
                or fields.get("template") or item.kind).replace("_", " ").capitalize())

    represented = set()
    layers = version_layers(manifest) if manifest.snapshots.get("layers") else ()
    for item in layers or ():
        if item.kind not in KINDS:
            continue
        lane, target = KINDS[item.kind]
        text = item.text or (label(authored[item.id]) if item.id in authored else
                             item.kind.replace("_", " ").capitalize())
        add(lane, item.start, item.end, text, target)
        represented.add(item.id)

    if is_cut or not authored:
        return lanes
    words = version_cut_words(manifest).words if manifest.snapshots.get("cut_words") else ()
    style = None
    entry = manifest.snapshots.get("style")
    if entry:
        style = (resolve_style_data(_snapshot_json(manifest, "style")) if entry.get("file")
                 else resolve(entry["ref"]).data)

    def seconds(anchor: Anchor):
        if anchor.time is not None:
            return anchor.time
        matches = [word for word in words if word.id == anchor.word
                   and (anchor.segment is None or word.segment == anchor.segment)]
        if len(matches) != 1:
            return None
        word = matches[0]
        return (word.start if anchor.edge == "start" else word.end) + anchor.offset_s

    for item in authored.values():
        if item.id in represented or item.kind not in KINDS:
            continue
        start = seconds(item.start)
        if start is None:
            continue
        end = seconds(item.end) if item.end is not None else None
        if item.kind == "title" and style is not None and style["hook"]["style"] == "text_hook":
            start, end = 0, duration
        elif item.kind == "sound":
            end = start
        elif item.end is None:
            hold = item.hold_s
            if hold is None and style is not None:
                hold = (style["hook"]["duration_s"] if item.kind == "title" else
                        style["chapters"]["hold_s"] if item.kind == "chapter" else
                        style["graphics"]["hold_s"])
            if hold is not None:
                end = start + hold
        if end is not None:
            lane, target = KINDS[item.kind]
            add(lane, start, end, label(item), target)
    for lane in lanes:
        lane["items"].sort(key=lambda item: (item["start"], item["end"]))
    return lanes
