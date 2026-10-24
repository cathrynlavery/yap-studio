"""Publish the project camera caption picture, front layer and measurements."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from yap_studio import ffmpeg, paths, receipt
from yap_studio.captions import CaptionWord, Phrase, build_phrases
from yap_studio.errors import YapError
from yap_studio.faces.clearance import face_clearance_px
from yap_studio.faces.mapping import map_picture_faces
from yap_studio.faces.project import cut_faces, cut_faces_identity
from yap_studio.matte import chroma
from yap_studio.matte.check import measure_behind_layers
from yap_studio.matte.model import Cleanup
from yap_studio.matte.project import CutMatteError, ensure_cut_matte
from yap_studio.matte.refine import shot_starts
from yap_studio.project.lock import ConflictError, project_lock
from yap_studio.project.receipts import StageRun, projection_sha256, read_pending, read_receipt
from yap_studio.project.status import ProjectContext
from yap_studio.render import camera3d, fonts, hyperframes
from yap_studio.render.camera3d_clearance import check_plan
from yap_studio.render.camera3d_look import (
    PLAN_VERSION,
    add_halos,
    display_words,
    person_refusal,
    picture_windows,
    plan_layout,
    planner_windows,
    refuse_elements,
    require_cut,
    require_still,
)
from yap_studio.render.camera3d_matte import sample_matte
from yap_studio.render.camera3d_motion import measure_camera_motion
from yap_studio.sfx import bleep


@dataclass(frozen=True)
class CameraSnapshot:
    """Cheap retained display words and identities, without matte preparation."""

    words: list[dict]
    inputs: dict[str, str]
    vocabulary: tuple[str, ...]
    frames: int


def retire_outputs(root: Path, run: StageRun, previous: dict | None,
                   names: Sequence[str]) -> None:
    """Retire owned caption outputs that the new result no longer needs."""
    with project_lock(root):
        records = [read_pending(run.stage_dir) or {}]
        if previous and previous.get("status") == "verified":
            records.append(previous)
        owned = {item["path"] for record in records for item in record.get("outputs", [])}
        for name in names:
            target = root / "captions" / name
            if f"captions/{name}" in owned and (target.exists() or target.is_symlink()):
                old = receipt.sha256_file(target)
                paths.validate_output(target, owner_receipts=(run.pending_path,))
                run._journal_output({"path": f"captions/{name}", "old_sha256": old,
                                     "new_sha256": None})
                if receipt.sha256_file(target) != old:
                    raise ConflictError(target)
                target.unlink()


def snapshot(context: ProjectContext) -> CameraSnapshot:
    from yap_studio.brand_words import list_words

    style = context.render_style().data
    timeline = context.resolved_timeline(allow_violations=True)
    words = context.json("cut/cut-words.json")["words"]
    vocabulary = tuple(list_words())
    gaps = [(item.start_s, item.end_s) for item in timeline.elements if item.kind == "caption_gap"]
    words = display_words(words, style["captions"], word_fixes=context.project.word_fixes,
                          word_text_fixes=context.project.word_text_fixes,
                          vocabulary=vocabulary, gaps=gaps)
    words = [word for word in words if word["start"] < timeline.total_frames / 30
             and word["end"] > 0]
    families = (style["captions"]["font"], style["hook"]["font"])
    inputs = {
        "style:captions": projection_sha256(style["captions"]),
        "brand:words": projection_sha256(vocabulary),
        "word_fixes": projection_sha256(context.project.word_fixes),
        **({"word_text_fixes": projection_sha256(context.project.word_text_fixes)}
           if context.project.word_text_fixes else {}),
        "camera3d:plan": projection_sha256({
            "version": PLAN_VERSION, "frames": timeline.total_frames,
            "hero_font": style["hook"]["font"],
            "hero_exclusions": sorted(bleep.listed(style["music"])),
            "title": style["hook"],
            "accent": style.get("brand", {}).get("accent_color") or style["hook"].get(
                "accent_color", style["captions"]["highlight_color"]),
            "headroom": style["internal"]["key_text_bottom_px"],
            "gap": style["graphics"]["caption_gap_px"], "words": words,
            "windows": [item.to_dict() for item in timeline.elements if item.kind in {
                "insert", "transition", "graphic", "title", "chapter", "overlay", "caption_gap"}],
        }),
        "cut/cut.mp4": context.identity.sha256(context.project_dir / "cut/cut.mp4"),
        "faces": projection_sha256(cut_faces_identity(
            context.identity.sha256(context.project_dir / "cut/cut.mp4"))),
        "style:face_clearance": projection_sha256(face_clearance_px(style)),
        "matte": projection_sha256({"backend": "auto", "quality": "accurate",
                                    "cleanup": Cleanup().to_dict(),
                                    "cutout": chroma.project_request(context.project_dir)}),
        "fonts": projection_sha256({"css": fonts.font_face_css(families),
            "faces": {name: hashlib.sha256(context.read(str(source))).hexdigest()
                      for name, source in fonts.font_files(families)}}),
        "renderer": projection_sha256({"version": hyperframes.HYPERFRAMES_VERSION,
            "workers": hyperframes.RENDER_WORKERS, "browser_gpu": hyperframes.CAPTION_BROWSER_GPU,
            "paint": hashlib.sha256(camera3d._PAINT_SCRIPT.encode()).hexdigest(),
            "assets": {name: hashlib.sha256(context.read(str(camera3d.TEMPLATE / name))).hexdigest()
                       for name in ("cam3d.js", "tables.js", "metrics.json")}}),
        **context.taste_inputs("captions"),
    }
    if bleep.enabled(style):
        phrases = plan_phrases(words, style["captions"], frames=timeline.total_frames,
                               gaps=[(round(start * 30), round(end * 30)) for start, end in gaps],
                               music=style["music"])
        inputs["bleep"] = projection_sha256({**bleep.projection(style["music"]),
                                              "phrases": [phrase.to_dict() for phrase in phrases]})
    spans = context.project_dir / "cut/spans.json"
    if spans.is_file():
        inputs["cut/spans.json"] = hashlib.sha256(context.read("cut/spans.json")).hexdigest()
    return CameraSnapshot(words, inputs, vocabulary, timeline.total_frames)


def prepare(root: Path, cut: Path, matte: Path) -> tuple[Path, Path]:
    key = projection_sha256({"cut": receipt.sha256_file(cut),
                             "matte": receipt.sha256_file(matte), "prep_version": 3})
    folder = paths.validate_output_dir(root / ".yap/cache/camera3d" / key)
    plate, person, _ = camera3d.prepare_plate(cut, folder, matte=matte)
    return plate, person


def render_outputs(plan: dict, plate: Path, person: Path, cut: Path, work: Path) -> dict:
    """Render one opaque camera picture and the independent BT.709 front ink."""
    frames = plan["frames"]
    composition = camera3d.build_composition(plan, plate, person, work / "picture",
                                            grain=False, mode="picture")
    renderer = hyperframes.ensure_installed()
    flags = ["--no-best-effort", "--quiet", "--fps", "30", "--workers",
             str(hyperframes.RENDER_WORKERS), "--experimental-fast-capture=false",
             *(["--no-browser-gpu"] if not hyperframes.CAPTION_BROWSER_GPU else []),
             "--player-ready-timeout", "240000", "--crf", "12"]
    picture = work / "camera3d.mp4"
    retry = hyperframes.render_with_retry(partial(
        hyperframes._run_renderer, renderer.executable,
        ["render", str(composition.absolute()), "--format", "mp4", *flags,
         "-o", str(picture.absolute())], timeout_s=180 + frames * 1080 * 1920 / 1e6))
    camera3d._check_media(picture, width=1080, height=1920, frames=frames)
    info = ffmpeg.probe(picture)
    if info.has_audio or info.variable_frame_rate or info.video_codec != "h264":
        raise ValueError("camera picture must be constant-rate silent H.264")
    front = camera3d.build_composition(plan, plate, person, work / "front",
                                      grain=False, mode="front")
    result = hyperframes.render_layer(front, work / "captions.mov", frames=frames,
                                     variables={}, workers=hyperframes.RENDER_WORKERS,
                                     browser_gpu=hyperframes.CAPTION_BROWSER_GPU, checks=False)
    return {"flags": flags, "retries": [retry] if retry else [],
            "front_retries": result.render_retries}


def check_behind(plan: dict, matte: Path, plate: Path, person: Path, cut: Path,
                 work: Path) -> tuple[dict, ...]:
    """Measure real hero ink against the camera transformed shared person matte."""
    heroes = [(index, sentence["hero"]) for index, sentence in enumerate(plan["sentences"])
              if sentence["hero"] and sentence["hero"]["layer"] == "behind"]
    if not heroes:
        return ()
    composition = camera3d.build_composition(plan, plate, person, work / "behind",
                                            grain=False, mode="behind")
    layer = work / "behind.mov"
    hyperframes.render_layer(composition, layer, frames=plan["frames"], variables={},
                             workers=hyperframes.RENDER_WORKERS,
                             browser_gpu=hyperframes.CAPTION_BROWSER_GPU, checks=False)
    specs = [{"id": f"hero-{index}", "start_frame": hero["on_frame"],
              "layer_start_frame": hero["on_frame"],
              "frames": hero["off_frame"] - hero["on_frame"], "path": layer.name,
              "x": 0, "y": 0, "width": 1080, "height": 1920} for index, hero in heroes]
    return measure_behind_layers(
        work, matte, ffmpeg.probe(cut), specs, hidden_glyph_share=.1, hidden_time_share=0,
        require_ink=True, report_worst=True,
        covered_pixels=partial(camera3d._render_person_covered, camera=plan["camera"]["samples"]))


def plan_phrases(words: list[dict], captions: dict, *, frames: int,
                 gaps: Sequence[tuple[int, int]] = (), music: dict | None = None) -> list[Phrase]:
    """Export retained display words in spoken order with the flat phrase limits."""
    breaks = {word["id"] for before, word in zip(words, words[1:], strict=False)
              if any(before["start"] * 30 < first and word["start"] * 30 >= last
                     for first, last in gaps)}
    phrases = build_phrases([CaptionWord(w["id"], w["text"], w["start"], w["end"])
                             for w in words], captions, forced_breaks=breaks)
    sources = {w["id"]: tuple(w.get("source_ids", [w["id"]])) for w in words}
    result = []
    for phrase in phrases:
        end = min(frames / 30, phrase.end,
                  *(first / 30 for first, _ in gaps if first / 30 > phrase.words[-1].start))
        result.append(replace(phrase, start=max(0, phrase.start), end=end,
                              words=tuple(replace(w, source_ids=sources[w.id])
                                          for w in phrase.words)))
    if music and bleep.enabled({"music": music}):
        masked = iter(bleep.mask_display_words([w for p in result for w in p.words],
                                                bleep.listed(music)))
        result = [replace(p, words=tuple(next(masked) for _ in p.words)) for p in result]
    return result


def title_faces(faces, words: list[dict], timeline):
    """Place titles against the same camera and face windows as the final picture."""
    frames = timeline.total_frames
    samples, _ = camera3d._camera_samples(camera3d._sentences(words), frames)
    camera = camera3d.face_transforms({"frames": frames, "camera": {"samples": samples}})
    return map_picture_faces(
        faces, timeline, zooms=(), camera=camera,
        video=replace(faces.video, name="picture.mp4",
                      frames=frames + timeline.end_hold_frames))[0]


def captions(context: ProjectContext, *, force: bool = False,
              retained: CameraSnapshot | None = None) -> dict:
    from yap_studio.stages.captions import _srt_text, caption_spelling

    root = context.project_dir
    retained = snapshot(context) if retained is None else retained
    timeline = context.resolved_timeline(allow_violations=True)
    refuse_elements(timeline.elements)
    timeline.require_valid()
    cut = root / "cut/cut.mp4"
    info = ffmpeg.probe(cut, count_frames=True)
    require_cut(info, cut)
    run = StageRun(root, "captions")
    layer = {"path": "captions/captions.mov", "x": 0, "y": 0, "width": 1080, "height": 1920,
             "start_frame": 0, "frames": retained.frames}
    result = {"status": "verified", "out": layer["path"], "frames": retained.frames,
              "layer": layer, "picture": "captions/camera3d.mp4", "srt": "captions/captions.srt",
              "layout": "captions/layout.json", "receipt": "captions/receipt.json"}
    if not force and run.up_to_date(retained.inputs):
        previous = read_receipt(root / "captions") or {}
        return {**result, "phrases": previous["phrases"],
                "camera_3d": previous["camera_3d"], "noop": True}
    try:
        matte = ensure_cut_matte(root, cut, cutout=chroma.project_request(root))
    except (CutMatteError, ValueError) as exc:
        raise person_refusal(str(exc)) from exc
    spans = root / "cut/spans.json"
    starts = shot_starts(spans, frames=retained.frames) if spans.is_file() else ()
    try:
        matte_samples, silhouettes = sample_matte(matte, retained.frames, starts)
    except ValueError as exc:
        raise person_refusal(str(exc)) from exc
    motion = measure_camera_motion(cut, matte, starts, matte_samples=matte_samples)
    require_still(motion)
    luminance = camera3d.sample_cut_luminance(cut, retained.frames / 30)
    style = context.render_style()
    faces = cut_faces(root, cut)
    obstacles, hero_free = planner_windows(
        timeline.elements, style.data, vocabulary=retained.vocabulary,
        faces=title_faces(faces, retained.words, timeline))
    replacements = picture_windows(timeline.elements)
    gaps = [(item.start_frame, item.end_frame) for item in timeline.elements
            if item.kind == "caption_gap"]
    work_root = paths.validate_output_dir(root / ".yap/work")
    with tempfile.TemporaryDirectory(prefix="captions3d-", dir=work_root) as folder, run:
        work = Path(folder)
        for name, digest in retained.inputs.items():
            run.record_inputs(name, digest)
        removed = set()
        unselected = set()
        attempts = []
        behind_attempts = []
        for attempt in range(2):
            plan = camera3d.plan_camera3d(
                {"time_base": "output", "words": retained.words}, retained.frames / 30,
                silhouettes, luminance=luminance, matte=matte, style=style, onset_lead=.09,
                max_words=min(4, style.data["captions"]["max_words_per_phrase"]),
                max_characters=style.data["captions"]["max_characters_per_phrase"],
                obstacles=obstacles, hero_free=hero_free, skip_heroes=removed | unselected,
                display_words=retained.words, faces=faces, caption_gaps=gaps,
                replacements=replacements)
            if unselected:
                plan["fallbacks"] = [item for item in plan["fallbacks"]
                                     if item["action"] != "hero removed"
                                     or item["sentence"] not in unselected]
            plan["look_version"] = PLAN_VERSION
            add_halos(plan)
            layout = plan_layout(plan)
            check_plan(plan, layout, faces, face_clearance_px(style.data), obstacles=obstacles,
                        replacements=replacements)
            if attempt == 0:
                plate, person = prepare(root, cut, matte)
            output = work / f"attempt-{attempt}"
            output.mkdir()
            run.progress("rendering camera captions", .1 + attempt * .4)
            attempts.append(render_outputs(plan, plate, person, cut, output))
            behind = check_behind(plan, matte, plate, person, cut, output)
            behind_attempts.append(list(behind))
            failed = {int(item["id"].split("-")[1]) for item in behind if item["flag"]}
            if not failed:
                break
            if attempt:
                raise YapError("Camera caption ink is still hidden after removing hidden heroes.",
                               code="camera3d_behind_words")
            removed.update(failed)
            unselected = {i for i, sentence in enumerate(plan["sentences"]) if not sentence["hero"]}
        plan["measurements"] = {"camera_motion": motion, "behind_words": list(behind),
                                "behind_attempts": behind_attempts}
        phrases = plan_phrases(retained.words, style.data["captions"],
                               frames=retained.frames, gaps=gaps, music=style.data["music"])
        layout = plan_layout(plan)
        summary = {"groups": len(plan["groups"]),
                   "heroes": [{"sentence": i, "text": s["hero"]["text"],
                               "layer": s["hero"]["layer"]}
                              for i, s in enumerate(plan["sentences"]) if s["hero"]],
                   "fallbacks": plan["fallbacks"], "camera_motion": motion,
                   "removed_heroes": sorted(removed), "renderer": attempts,
                   "behind_words": list(behind)}
        spelling = caption_spelling(phrases, retained.vocabulary)
        documents = {"camera3d.json": plan, "layout.json": layout,
                     "phrases.json": {"schema": "yap-studio/phrases@1", "time_base": "output",
                                      "fps": 30, "layer": layer,
                                      "phrases": [phrase.to_dict() for phrase in phrases]}}
        for name, document in documents.items():
            (output / name).write_text(json.dumps(document, indent=2, ensure_ascii=False,
                                                  allow_nan=False) + "\n", encoding="utf-8")
        (output / "captions.srt").write_text(_srt_text(phrases), encoding="utf-8")
        for name in ("captions.mov", "camera3d.mp4", "camera3d.json", "layout.json",
                     "phrases.json", "captions.srt"):
            run.replace_output(root / "captions" / name, output / name)
        run.finish(extra={"layer": layer, "picture": result["picture"], "camera_3d": summary,
                          "phrases": len(phrases), "caption_spelling": spelling,
                          "caption_layout": {"measured": True, "reason": None,
                                             "captions": len(layout["captions"])},
                          "taste": context.taste_application().pin()})
    return {**result, "phrases": len(phrases), "camera_3d": summary, "noop": False}
