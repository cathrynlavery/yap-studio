"""Cache number layers and check their placement against the cut person matte."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yap_studio import cards, ffmpeg, paths, receipt
from yap_studio.errors import DependencyError, ValidationError
from yap_studio.faces.clearance import face_boxes_per_frame, face_clearance_px
from yap_studio.faces.model import Box
from yap_studio.faces.placement import top_band
from yap_studio.faces.project import cut_faces
from yap_studio.matte.check import _behind_ink, _window_frames
from yap_studio.matte.ink import covered_ink_pixels
from yap_studio.planner.number_behind import place_number
from yap_studio.project.receipts import projection_sha256
from yap_studio.render import fonts, hyperframes
from yap_studio.stages.captions import LayerAssets
from yap_studio.stages.number_visibility import NUMBER_INK_ALPHA_THRESHOLD

if TYPE_CHECKING:
    from yap_studio.stages.timeline import ResolvedElement
    from yap_studio.style import ResolvedStyle


def _number_assets(plan: cards.CardPlan) -> LayerAssets:
    template = cards.template_dir(plan.template)
    return LayerAssets(paths.validate_input(template / "template.html").read_text(),
                       paths.validate_input(template / "template.json").read_bytes(),
                       fonts.font_face_css(plan.fonts),
                       tuple((name, paths.validate_input(source).read_bytes())
                             for name, source in fonts.font_files(plan.fonts)))


def _number_key(plan: cards.CardPlan, assets: LayerAssets) -> str:
    return projection_sha256({
        "variables": plan.variables, "width": plan.width, "height": plan.height,
        "frames": plan.frames, "fps": plan.fps, "fonts": assets.font_identity(),
        "template": assets.template_identity(
            workers=hyperframes.RENDER_WORKERS, browser_gpu=hyperframes.CARD_BROWSER_GPU),
    })


def cached_number_layer(
    plan: cards.CardPlan, project_dir: Path, *, assets: LayerAssets | None = None,
) -> tuple[hyperframes.LayerResult, bool]:
    """Share position-independent card pixels between placement and publication."""
    assets = assets or _number_assets(plan)
    key = _number_key(plan, assets)
    cache = paths.validate_output_dir(project_dir / ".yap/cache/number-layers")
    target, record = cache / f"{key}.mov", cache / f"{key}.json"
    if target.exists():
        try:
            saved = json.loads(record.read_bytes())
            if receipt.sha256_file(target) != saved["sha256"]:
                raise ValueError("changed cache bytes")
            return hyperframes.LayerResult(paths.validate_input(target), plan.frames,
                plan.width, plan.height, saved["duration_s"], saved["checks"]), False
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValidationError("big_number.cache",
                                  "cached number layer changed; clear its cache") from exc
    work = paths.validate_output_dir(project_dir / ".yap/work")
    with tempfile.TemporaryDirectory(prefix="number-", dir=work) as folder:
        rendered = cards.render_card(plan, Path(folder), Path(folder) / "number.mov")
        if key != _number_key(plan, _number_assets(plan)):
            raise ValidationError("graphic.inputs", "template or font changed; run graphics again")
        with paths.atomic_output(target) as temporary:
            shutil.copyfile(rendered.path, temporary)
        digest = receipt.sha256_file(target)
        paths.atomic_write_text(record, json.dumps({
            "schema": "yap-studio/number-layer@1", "sha256": digest,
            "duration_s": rendered.duration_s, "checks": rendered.checks,
            "outputs": [{"path": target.name, "sha256": digest}],
        }, sort_keys=True) + "\n")
    return replace(rendered, path=target), True


def publish_number_layer(
    plan: cards.CardPlan, project_dir: Path | None, out: Path, *, assets: LayerAssets | None = None,
) -> hyperframes.LayerResult:
    """Publish the exact cached layer whose ink placement measured."""
    if project_dir is None:
        with tempfile.TemporaryDirectory(prefix="number-", dir=out.parent) as folder:
            return cards.render_card(plan, Path(folder), out)
    rendered, _ = cached_number_layer(plan, project_dir, assets=assets)
    with paths.atomic_output(out) as temporary:
        shutil.copyfile(rendered.path, temporary)
    return replace(rendered, path=out)


def place_number_graphic(
    card: cards.CardPlan, element: ResolvedElement, style: ResolvedStyle,
    faces: Sequence[Sequence[Box]] | None, matte: Path, caption_top: int, *,
    face_windows: tuple[tuple[int, int], ...] = (), project_dir: Path,
) -> tuple[cards.CardPlan, str, dict]:
    """Decode matte and rendered ink once per size, then search their exact coverage."""
    # Transitions clear mapped faces; recover cut faces to keep the front fallback clear.
    if faces is not None and face_windows and not all(faces):
        source = cut_faces(project_dir, project_dir / "cut/cut.mp4")
        if source.status == "measured":
            cut = face_boxes_per_frame(source)[element.start_frame:element.end_frame]
            faces = [current or original for current, original in zip(faces, cut, strict=True)]
    with closing(_window_frames(matte, ffmpeg.probe(matte), element.start_frame,
                                element.end_frame, matte=True)) as stream:
        mattes = tuple(stream)
    options = {"caption_top": caption_top - style.data["graphics"]["caption_gap_px"],
               "band": top_band(style.data), "margin_px": face_clearance_px(style.data),
               "identifier": element.id, "face_windows": face_windows}
    last_size = None
    ink_frames: tuple[np.ndarray, ...] = ()
    totals: tuple[int, ...] = ()
    render_count = 0

    def visible(candidate: cards.CardPlan) -> float:
        nonlocal last_size, ink_frames, totals, render_count
        size = candidate.variables["value_px"]
        if size != last_size:
            rendered, created = cached_number_layer(candidate, project_dir)
            render_count += int(created)
            with closing(_window_frames(rendered.path, ffmpeg.probe(rendered.path),
                                        0, candidate.frames, rgba=True)) as stream:
                ink_frames = tuple(_behind_ink(pixels, alpha_threshold=NUMBER_INK_ALPHA_THRESHOLD)
                                   for pixels in stream)
            totals = tuple(int(np.count_nonzero(ink)) for ink in ink_frames)
            if not any(totals):
                raise ValidationError("big_number.ink",
                                      "number layer has no visible ink; render again")
            last_size = size
        return min((1 - covered_ink_pixels(ink, alpha, candidate.x, candidate.y) / total
                    for ink, alpha, total in zip(ink_frames, mattes, totals, strict=True) if total),
                   default=1.0)

    try:
        placed, placement, evidence = place_number(card, faces, mattes,
                                                    ink_visible_share=visible, **options)
    except DependencyError as exc:
        placed, placement, evidence = place_number(card, faces, mattes, **options)
        evidence.update(ink_status="unavailable", ink_reason=exc.code,
                        reason=evidence["reason"]
                        + "; renderer unavailable; final ink check applies")
    evidence["render_count"] = render_count
    return placed, placement, evidence


def number_warnings(elements: list[dict]) -> list[str]:
    """Recover fallback notices from saved clearance evidence as well as new plans."""
    return [item["clearance"]["warning"] for item in elements
            if item.get("clearance", {}).get("warning")]
