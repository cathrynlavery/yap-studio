"""Find removed source stretches from immutable version snapshots."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import cast

from yap_studio.errors import InputError
from yap_studio.feedback.versions import VersionManifest, _snapshot_json, version_pieces
from yap_studio.project.model import Removals, Words, parse_document


def _words(project_dir: Path, take: str, inputs: dict) -> Words | None:
    path = project_dir / "ingest" / take / "words.json"
    expected = inputs.get(f"words:{take}")
    if expected is None:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if hashlib.sha256(data).hexdigest() != expected:
        return None
    try:
        document = json.loads(data)
    except (UnicodeError, ValueError) as exc:
        raise InputError(str(path), "cannot read valid JSON") from exc
    return cast(Words, parse_document("words", document, filename=str(path)))


def removed_ranges(
    project_dir: Path, manifest: VersionManifest, *, min_s: float = 1 / 30,
) -> list[dict]:
    """List forward gaps in the saved cut, with hash-matched source words only."""
    saved = version_pieces(manifest)
    pieces = saved["pieces"]
    if len(pieces) < 2:
        return []
    removals = ()
    if manifest.snapshots["removals"] is not None:
        removals = cast(Removals, parse_document(
            "removals", cast(dict, _snapshot_json(manifest, "removals")),
        )).removals
    fps = Fraction(saved["fps"])
    minimum = Fraction(min_s)
    transcripts: dict[str, Words | None] = {}
    ranges = []
    for index, (p, q) in enumerate(zip(pieces, pieces[1:], strict=False)):
        start, end = Fraction(p["src_t1"]), Fraction(q["src_t0"])
        if p["take"] != q["take"] or end <= start or end - start < minimum:
            continue
        take = p["take"]
        overlaps = []
        for removal in removals:
            if removal.status != "applied" or removal.take != take:
                continue
            overlap = min(end, Fraction(str(removal.end))) - max(
                start, Fraction(str(removal.start)),
            )
            if overlap > 0:
                overlaps.append((removal, overlap))
        kind = (max(overlaps, key=lambda item: item[1])[0].kind if overlaps else
                "story" if p["segment"] != q["segment"] else "trim")
        if take not in transcripts:
            transcripts[take] = _words(Path(project_dir), take, saved.get("inputs", {}))
        transcript = transcripts[take]
        words = [{"id": word.id, "text": word.text} for word in transcript.words
                 if start <= (Fraction(str(word.start)) + Fraction(str(word.end))) / 2 < end
                 ] if transcript is not None else []
        ranges.append({
            "id": f'{p["segment"]}.{index}', "take": take,
            "start": float(start), "end": float(end),
            "at_s": float(Fraction(q["out_f0"]) / fps), "at_frame": q["out_f0"],
            "kind": kind, "removal_ids": [item.id for item, _ in overlaps],
            "words": words, "text": " ".join(word["text"] for word in words) or None,
            "words_known": transcript is not None,
        })
    return ranges
