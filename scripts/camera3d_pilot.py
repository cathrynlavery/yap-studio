"""Render one whole vertical cut with the camera caption pilot."""

from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    import os
    import shlex

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(
        "scripts/camera3d_pilot.py needs Python 3.11 or newer. "
        f"Run it with: cd {shlex.quote(root)} && uv run python scripts/camera3d_pilot.py"
    )

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from yap_studio import ffmpeg
from yap_studio.checks.model import status_of
from yap_studio.project.status import ProjectContext
from yap_studio.render import hyperframes
from yap_studio.render.camera3d import (
    build_composition,
    check_pilot_render,
    plan_camera3d,
    prepare_plate,
    render_pilot,
    sample_cut_luminance,
    summarize_matte,
)
from yap_studio.style.library import get_default
from yap_studio.style.loader import resolve as resolve_style


def select_style(project: Path, spec: str | None):
    """Use the explicit style, then the saved default, then the project style."""
    if spec:
        return resolve_style(spec)
    saved = get_default()
    if saved:
        return resolve_style(saved)
    if (project / "project.json").is_file():
        return ProjectContext(project).render_style()
    return resolve_style("preset:clean_minimal")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--no-grain", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="write the plan without rendering")
    parser.add_argument("--style", help="style file, library id, or preset spec")
    parser.add_argument(
        "--workers", type=int, default=hyperframes.RENDER_WORKERS,
        help=(f"renderer workers (default: {hyperframes.RENDER_WORKERS}; "
              "faster overrides may shift frames)"),
    )
    parser.add_argument(
        "--keep", type=Path, metavar="DIR", help="store the composition and prep assets in DIR"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    started = time.monotonic()

    def progress(stage: str) -> None:
        print(f"[{time.monotonic() - started:.1f}s] {stage}", flush=True)

    project = args.project_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        parser.error("OUT must end in .mp4")
    cut = project / "cut/cut.mp4"
    words_file = project / "cut/cut-words.json"
    if not cut.is_file() or not words_file.is_file():
        parser.error("PROJECT_DIR needs cut/cut.mp4 and cut/cut-words.json")
    words = json.loads(words_file.read_text(encoding="utf-8"))
    info = ffmpeg.probe(cut, count_frames=True)
    if not info.video_frames:
        parser.error("the cut has no verified video frame count")
    folder = args.keep.expanduser().resolve() if args.keep else output.with_suffix(".camera3d")
    folder.mkdir(parents=True, exist_ok=True)
    style = select_style(project, args.style)
    print(f"Style: {style.source} ({style.name})", flush=True)
    plate, person, gray = prepare_plate(cut, folder / "prep", progress=progress)
    progress("plan")
    silhouette = summarize_matte(gray, info.video_frames / 30)
    luminance = sample_cut_luminance(cut, info.video_frames / 30)
    plan = plan_camera3d(words, info.video_frames / 30, silhouette,
                         luminance=luminance, matte=gray, style=style,
                         word_fixes=(ProjectContext(project).project.word_fixes
                                     if (project / "project.json").is_file() else {}))
    output.parent.mkdir(parents=True, exist_ok=True)
    plan_file = output.with_suffix(".plan.json")
    plan_file.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    if args.plan_only:
        for sentence in plan.get("sentences", ()):
            hero = sentence.get("hero")
            if hero:
                print(f"Hero {hero['text']}: {hero['placement']} "
                      f"min visible {hero['min_visible_share']:.3f}", flush=True)
        print(f"Plan: {plan_file}\nFallbacks: {len(plan['fallbacks'])}", flush=True)
        return 0
    progress("build")
    composition_dir = folder / "composition"
    staging = Path(tempfile.mkdtemp(prefix=".composition-build-", dir=folder))
    retired = folder / f".composition-old-{uuid4().hex}"
    try:
        build_composition(plan, plate, person, staging, grain=not args.no_grain)
        had_previous = composition_dir.exists()
        if had_previous:
            composition_dir.rename(retired)
        try:
            staging.rename(composition_dir)
        except OSError:
            if had_previous:
                retired.rename(composition_dir)
            raise
        if had_previous:
            shutil.rmtree(retired)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    composition = composition_dir
    print(f"Composition: {composition}", flush=True)
    render_pilot(composition, cut, output, plan["frames"], workers=args.workers,
                 bleeps=plan.get("bleeps"), progress=progress)
    findings = (check_pilot_render(plan, output, gray, plate, person, folder,
                                   grain=not args.no_grain, workers=args.workers)
                if "sentences" in plan else [])
    checks = output.with_suffix(".checks.json")
    checks.write_text(json.dumps({"schema": "yap-studio/camera3d-checks@1",
                                  "status": status_of(findings),
                                  "findings": [item.to_dict() for item in findings]},
                                 indent=2) + "\n", encoding="utf-8")
    print(f"Video: {output}\nPlan: {plan_file}\nFallbacks: {len(plan['fallbacks'])}",
          flush=True)
    return 1 if status_of(findings) == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
