"""Cache measured faces of overlay-free project cuts."""

from __future__ import annotations

import hashlib
import os
import platform
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from yap_studio import paths, receipt
from yap_studio.errors import YapError
from yap_studio.faces import detect
from yap_studio.faces.mapping import map_picture_faces, picture_zooms
from yap_studio.faces.model import FacesData, VideoIdentity, load_faces

if TYPE_CHECKING:
    from yap_studio.project.status import ProjectContext
    from yap_studio.stages.timeline import ResolvedTimeline

FACE_EVERY = 2


def _camera_transforms(
    root: Path, timeline: ResolvedTimeline, context: ProjectContext | None = None,
) -> list[tuple[float, float, float]] | None:
    from yap_studio.project.status import ProjectContext, captions_inputs
    from yap_studio.render.camera3d import face_transforms
    from yap_studio.stages.assemble import StaleStageError

    if context is None:
        context = ProjectContext(root)
    if not context.render_style().data["captions"].get("camera_3d", False):
        return None
    try:
        record = context.json("captions/receipt.json")
    except FileNotFoundError:
        record = {}
    if not record.get("picture") and not (root / "captions/camera3d.json").exists():
        return None
    try:
        plan = context.json("captions/camera3d.json")
        digest = hashlib.sha256(context.read("captions/camera3d.json")).hexdigest()
        owned = any(item.get("path") == "captions/camera3d.json"
                    and item.get("sha256") == digest for item in record.get("outputs", []))
        if (not owned or plan.get("frames") != timeline.total_frames
                or record.get("schema") != "yap-studio/stage-receipt@1"
                or record.get("status") != "verified"
                or (root / "captions/receipt.pending.json").exists()
                or record.get("inputs") != captions_inputs(context)):
            raise StaleStageError("captions")
        return face_transforms(plan)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StaleStageError("captions") from exc


def picture_faces(
    project_dir: Path, timeline: ResolvedTimeline, context: ProjectContext | None = None, *,
    cut: FacesData | None = None,
) -> FacesData:
    """Plan against the same cut cache and source-aware zoom geometry as assembly."""
    from yap_studio.stages.assemble_picture import _source, _zoom_inputs
    from yap_studio.stages.broll_split import split_layouts, split_windows

    root = Path(project_dir)
    zooms = []
    camera = (_camera_transforms(root, timeline, context)
              if any((root / name).exists() for name in
                     ("captions/camera3d.json", "captions/receipt.json")) else None)
    cut = cut if cut is not None else cut_faces(root, root / "cut/cut.mp4")
    if cut.status == "measured" and any(item.kind == "zoom" for item in timeline.elements):
        source = _source(root / "cut/cut.mp4", [], [], owner="cut")
        zooms = picture_zooms(_zoom_inputs(root, timeline, source, [], [], []), source)
    return map_picture_faces(
        cut, timeline, zooms=zooms, splits=split_windows(timeline, split_layouts(root, timeline)),
        camera=camera,
        video=VideoIdentity("picture.mp4", cut.video.sha256,
                            timeline.total_frames + timeline.end_hold_frames,
                            Fraction(30), cut.video.width, cut.video.height),
    )[0]


def _helper_sha256() -> str:
    return hashlib.sha256(detect.helper_source().read_bytes()).hexdigest()


def cut_faces_identity(cut_sha256: str) -> dict:
    """Identify face inputs without probing or measuring the cut."""
    return {"cut": cut_sha256, "helper": _helper_sha256(), "every": FACE_EVERY,
            "macos": platform.mac_ver()[0],
            "available": detect.unavailable_reason() is None}


def _cut_faces_path(project_dir: Path, digest: str) -> Path:
    macos = platform.mac_ver()[0] or "unknown"
    return Path(project_dir) / ".yap/cache/faces" / (
        f"{digest}-{_helper_sha256()[:16]}-macos{macos}-every{FACE_EVERY}.json")


def cached_cut_faces(
    project_dir: Path, cut: Path, *, digest: str | None = None,
) -> FacesData | None:
    """Read an existing valid cache without measuring or creating directories."""
    try:
        source = paths.validate_input(cut)
        digest = receipt.sha256_file(source) if digest is None else digest
        cached = load_faces(_cut_faces_path(project_dir, digest))
        if (cached.status == "measured" and cached.video.sha256 == digest
                and cached.sampling.get("every") == FACE_EVERY):
            return cached
    except (YapError, OSError, ValueError):
        pass
    return None


def cut_faces(
    project_dir: Path, cut: Path, *, detector: detect.FaceDetector | None = None,
    digest: str | None = None,
) -> FacesData:
    """Reuse a validated cut/helper/sampling document, or atomically measure it."""
    with paths.operation():
        if detector is None and detect.unavailable_reason() is not None:
            return detect.detect_faces(cut, every=FACE_EVERY)
        source = paths.validate_input(cut)
        digest = receipt.sha256_file(source) if digest is None else digest
        target = _cut_faces_path(project_dir, digest)
        cache = paths.validate_output_dir(target.parent)
        paths._guard_write(target)
        try:
            cached = load_faces(target)
            if (cached.status == "measured" and cached.video.sha256 == digest
                    and cached.sampling.get("every") == FACE_EVERY):
                return cached
        except (YapError, OSError, ValueError):
            pass
        measured = detect.detect_faces(source, every=FACE_EVERY, detector=detector)
        if measured.status == "measured" and measured.video.sha256 == digest:
            with tempfile.TemporaryDirectory(prefix=".faces-", dir=cache) as folder:
                staged = Path(folder) / "faces.json"
                detect.write_faces(staged, measured)
                paths._guard_write(target)
                os.replace(staged, target)
        return measured
