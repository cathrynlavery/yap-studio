"""Build one token-scoped review video from files or saved project versions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from yap_studio.errors import InputError, ValidationError
from yap_studio.feedback.versions import (
    complete_pending_rollback,
    load_versions,
    version_cut_words,
)
from yap_studio.review.media import file_identity, probe_media
from yap_studio.review.series import SeriesEntry
from yap_studio.review.server import ReviewSeries, ReviewVersion, ReviewVideo
from yap_studio.review.store import ReviewStore


def version_video(
    values: Sequence[str], store: ReviewStore, expand: Callable[[Sequence[str]], list[Path]]
) -> ReviewVideo:
    """Validate ordered media and launch identities, then create or extend its series."""
    projects = [Path(value) for value in values
                if Path(value).is_dir() and (Path(value) / "project.json").is_file()]
    if projects and len(values) != 1:
        raise ValidationError("media", "cannot mix a project folder with other versions")
    if projects:
        complete_pending_rollback(projects[0])
    manifests = load_versions(projects[0]) if projects else ()
    if projects and not manifests:
        error = InputError(str(projects[0]), "no saved project versions")
        error.suggested_action = "Run yap-studio versions save first."
        raise error
    files = [item.media_path for item in manifests] if projects else expand(values)
    if not projects and files and not any(Path(value).is_dir() for value in values):
        first_hash = store.hash_media(files[0]).sha256
        existing, indexed = store.series_and_paths_containing(first_hash)
        if existing is not None:
            position = next(item.number for item in existing.versions
                            if item.sha256 == first_hash)
            earlier = []
            prefix = existing.versions[:position - 1]
            if any(store.hash_media(path).sha256 in {item.sha256 for item in prefix}
                   for path in files[1:]):
                prefix = ()
            for saved in prefix:
                found = None
                for candidate in indexed.get(saved.sha256, []):
                    try:
                        if (candidate.is_file()
                                and store.hash_media(candidate).sha256 == saved.sha256):
                            found = candidate
                            break
                    except InputError:
                        continue
                if found is None:
                    raise ValidationError(
                        "media", f"missing version {saved.number}: {saved.name}; list it first, "
                        "oldest first, or open the project folder",
                    )
                earlier.append(found)
            files = [*earlier, *files]
    if not projects and len(files) < 2:
        raise ValidationError(
            "media", "--versions needs at least two files, oldest first, or a project folder"
        )
    versions, entries = [], []
    supplied_paths: dict[Path, list[str]] = {}
    for value in values:
        supplied_paths.setdefault(Path(value).resolve(), []).append(value)
    seen: dict[str, str] = {}
    for index, path in enumerate(files):
        manifest = manifests[index] if projects else None
        result = store.hash_media(path)
        if manifest is not None and result.sha256 != manifest.media["sha256"]:
            raise InputError(str(path), "it changed after it was saved")
        supplied = supplied_paths.get(path.resolve(), [])
        display_path = supplied.pop(0) if supplied else str(path)
        if result.sha256 in seen:
            raise ValidationError(
                "media", f"{seen[result.sha256]} and {display_path} have identical contents; "
                "each version must differ",
            )
        seen[result.sha256] = display_path
        media = probe_media(path)
        if file_identity(path) != result.identity:
            raise InputError(str(path), "the file changed while it was being opened; try again")
        if manifest is not None:
            media = replace(media, name=f"{manifest.project}-{manifest.id}{path.suffix}")
        store.comments(result.sha256)
        version_id = manifest.id if manifest is not None else None
        project = manifest.project if manifest is not None else None
        entries.append(SeriesEntry(result.sha256, media.name, version_id, project))
        versions.append(ReviewVersion(
            index + 1, media, result.sha256, result.identity,
            version_cut_words(manifest) if manifest is not None else None, version_id,
        ))
    series = store.open_series(entries)
    link = store.series_link(series.id)
    newest = versions[-1]
    return ReviewVideo(newest.media, newest.sha256, previous_sha=None,
                       token=link["token"],
                       media_identity=newest.media_identity,
                       series=ReviewSeries(series.id, tuple(versions),
                                           projects[0].resolve() if projects else None,
                                           preferred_port=link["port"]))
