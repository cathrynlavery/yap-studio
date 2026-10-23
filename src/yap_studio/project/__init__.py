"""Project initialization and the versioned edit-pipeline contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

from yap_studio import ffmpeg, paths
from yap_studio import video_type as video_choices
from yap_studio.errors import InputError, YapError
from yap_studio.project.lock import ConflictError, project_lock, write_json_cas
from yap_studio.project.model import Project, Take, load_project, parse_document
from yap_studio.style import resolve
from yap_studio.style.library import library_dir
from yap_studio.style.loader import resolve_style_data

__all__ = ["ConflictError", "Project", "Take", "init_project", "load_project"]


class ProjectRefusal(YapError):
    """Refuse initialization over an existing project."""

    def __init__(self) -> None:
        super().__init__(
            "project.json already exists.", error_type="project_refusal", code="project_exists",
            suggested_action="Choose another project folder or open the existing project.",
        )


def _slug(value: str, *, limit: int = 80) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")[:limit] or "project"


def init_project(
    project_dir: Path, *, takes: Sequence[Path], style: str | None = None, name: str | None = None,
    answers: Path | None = None, voiceovers: Sequence[Path] = (),
    video_type: str | None = None, broll_sources: Sequence[str] | None = None,
) -> Project:
    """Probe read-only takes and atomically initialize an isolated project style."""
    if takes and voiceovers:
        raise InputError(str(project_dir), "cannot mix --take and --voiceover in one project")
    directory = Path(project_dir).absolute()
    target = directory / "project.json"
    existing = (load_project(target) if target.is_file() and not target.is_symlink()
                and voiceovers else None)
    if (target.exists() or target.is_symlink()) and existing is None:
        raise ProjectRefusal()
    if existing and (takes or style or name or answers or video_type is not None
                     or broll_sources is not None or
                     not all(take.voiceover for take in existing.takes)):
        raise ProjectRefusal()
    defaults = {}
    answer_data = None
    if answers is not None:
        from yap_studio.style.answers import load_answers, project_defaults

        answer_data = load_answers(answers)
        defaults = project_defaults(answer_data)
    kind, chosen_sources = (video_choices.resolve(
        voiceover=bool(voiceovers), video_type=video_type, broll_sources=broll_sources,
        default=(video_choices.load_default()
                 if video_type is None or broll_sources is None else None))
        if existing is None else (None, None))
    with paths.operation():
        sources = list(existing.takes) if existing else []
        for index, source in enumerate(takes, 1):
            validated = paths.validate_input(source)
            media = ffmpeg.probe(validated)
            if not media.has_video or not media.has_audio:
                raise InputError(str(source), "a take must have both picture and audio")
            try:
                stored = validated.relative_to(directory.resolve()).as_posix()
            except ValueError:
                stored = str(validated)
            sources.append(Take(f"t{index}", stored))
        pending: list[tuple[Path, Path, float]] = []
        for source in voiceovers:
            validated = paths.validate_input(source)
            media = ffmpeg.probe(validated)
            if not media.has_audio or media.has_video or not media.duration_s:
                raise InputError(str(source), "a voiceover must contain audio and no picture")
            take_id = f"t{len(sources) + 1}"
            relative = f".yap/voiceover/{take_id}.mp4"
            sources.append(Take(take_id, relative, voiceover=True))
            pending.append((validated, directory / relative, media.duration_s))
        if existing:
            with project_lock(directory):
                if load_project(target) != existing:
                    raise ConflictError(target)
                for source, standin, duration in pending:
                    _standin(source, standin, duration)
                project = parse_document("project", {**existing.to_dict(), "takes": [
                    {key: value for key, value in vars(take).items()
                     if value is not None and (key != "voiceover" or value)}
                    for take in sources
                ]})
                write_json_cas(target, project.to_dict(),
                               base_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            return project
        selected = resolve(style)
        original: bytes | None = None
        copied: Path | None = None
        if selected.source.startswith("preset:"):
            stored_style = selected.source
            if style is None and answer_data is not None:
                from yap_studio.style.answers import style_changes

                data = copy.deepcopy(selected.unbranded_data)
                for section, values in style_changes(answer_data, data).items():
                    if section != "pending":
                        data[section] = {**data.get(section, {}), **values}
                if data != selected.unbranded_data:
                    resolve_style_data(data)
                    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
                    original = text.encode("utf-8")
                    preset = selected.source.removeprefix("preset:")
                    copied = directory / "styles" / f"{preset}.json"
                    stored_style = copied.relative_to(directory).as_posix()
        else:
            source = (library_dir() / f"{selected.source.removeprefix('library:')}.json"
                      if selected.source.startswith("library:") else Path(selected.source))
            source = paths.validate_input(source)
            original = source.read_bytes()
            if resolve_style_data(json.loads(original)) != selected.data:
                raise ConflictError(source)
            copied = directory / "styles" / f"{_slug(source.stem, limit=64)}.json"
            stored_style = copied.relative_to(directory).as_posix()
        project = Project(name=name if name is not None else _slug(directory.name),
                          takes=tuple(sources), style=stored_style, video_type=kind,
                          broll_sources=(tuple(chosen_sources)
                                         if chosen_sources is not None else None))
        document = project.to_dict()
        for key, value in defaults.items():
            document[key] = {**document[key], **value} if key in {"rules", "music"} else value
        project = parse_document("project", document)
        paths.validate_output_dir(directory)
        with project_lock(directory):
            if target.exists() or target.is_symlink():
                raise ProjectRefusal()
            if copied is not None and original is not None:
                with paths.atomic_output(copied) as temporary:
                    temporary.write_bytes(original)
            for source, standin, duration in pending:
                _standin(source, standin, duration)
            write_json_cas(target, project.to_dict(), base_sha256=None)
        return project


def _standin(source: Path, target: Path, duration: float) -> None:
    """Make a deterministic vertical picture while preserving the memo audio."""
    paths.validate_output(target, allowed_existing_suffixes=frozenset())
    paths.validate_output_dir(target.parent)
    with tempfile.NamedTemporaryFile(suffix=".mp4", dir=target.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        ffmpeg.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:r=30",
                    "-i", str(source), "-t", str(duration), "-map", "0:v:0", "-map", "1:a:0",
                    "-vf", "setparams=color_primaries=bt709:color_trc=bt709:"
                    "colorspace=bt709:range=tv",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-colorspace", "bt709", "-color_primaries", "bt709",
                    "-color_trc", "bt709", "-color_range", "tv",
                    "-c:a", "aac", "-shortest", str(temporary)],
                   timeout_s=max(ffmpeg.DEFAULT_TIMEOUT_S, duration * 4))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
