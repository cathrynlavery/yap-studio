"""Extract project card stills only when a layer needs rendering."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from yap_studio import ffmpeg, hdr, paths, receipt
from yap_studio.card_project import CardShot
from yap_studio.errors import ValidationError
from yap_studio.grade import rgba_filter
from yap_studio.media.frame import extract_frame
from yap_studio.project.receipts import projection_sha256


@dataclass(frozen=True)
class CardPoster:
    """Retain a clip's exact poster time until preparation needs its picture."""

    source: Path
    time_s: str
    sha256: str | None = None
    cache_root: Path | None = None


def cached_still(project_dir: Path, shot: CardShot, *, width: int, height: int,
                 verified_sources: dict[Path, tuple[str, int, int, int]] | None = None) -> Path:
    """Cache one converted picture by source identity, exact time and dimensions."""
    if shot.source is None or shot.source_sha256 is None:
        raise ValidationError("storyboard.shot", "use a shot with a retained source identity")
    return cached_frame(project_dir, shot.source, shot.source_sha256,
                        source_frame=shot.source_frame, source_time=shot.source_time,
                        width=width, height=height, verified_sources=verified_sources)


def cached_frame(project_dir: Path, source: str | Path, source_sha256: str | None, *,
                 source_frame: int | None = None, source_time: str | None = None,
                 width: int, height: int,
                 source_kind: str = "shot",
                 verified_sources: dict[Path, tuple[str, int, int, int]] | None = None,
                 grade_identity: dict | None = None, cube: Path | None = None) -> Path:
    """Reuse a converted frame without putting source paths in the cache identity."""
    source = paths.validate_input(project_dir / source)
    if verified_sources is None:
        verified_sources = {}
    if source not in verified_sources:
        stamp = source.stat()
        verified_sources[source] = (receipt.sha256_file(source), stamp.st_size,
                                    stamp.st_mtime_ns, stamp.st_ctime_ns)
    if source_sha256 is None:
        source_sha256 = verified_sources[source][0]

    def check_source() -> None:
        stamp = source.stat()
        if verified_sources[source] != (source_sha256, stamp.st_size,
                                         stamp.st_mtime_ns, stamp.st_ctime_ns):
            raise ValidationError("graphic.inputs",
                                  f"{source_kind} source changed; run graphics again")

    check_source()
    info = ffmpeg.probe(source)
    try:
        conversion = hdr.conversion(info)
    except hdr.HdrUnsupportedError:
        conversion = None
    identity = {"source_sha256": source_sha256, "source_frame": source_frame,
                "source_time": source_time}
    key = projection_sha256({**identity, "width": width, "height": height, "rule": 1,
                             "conversion": conversion.describe() if conversion else "none",
                             **({"grade": grade_identity} if grade_identity is not None else {})})
    folder = paths.validate_output_dir(project_dir / ".yap/cache/card-stills")
    target = folder / f"{key}.{'png' if cube is not None else 'jpg'}"
    if target.is_file() and target.stat().st_size:
        return paths.validate_input(target)
    time = (Fraction(source_frame, 30) if source_frame is not None else
            Fraction(source_time) if source_time is not None else None)
    with paths.atomic_output(target) as temporary:
        if cube is None:
            extract_frame(source, info, temporary, time_s=float(time) if time is not None else None,
                          width=width, height=height, conversion=conversion)
        else:
            with tempfile.TemporaryDirectory(dir=folder) as directory:
                original = Path(directory) / "poster.png"
                extract_frame(source, info, original,
                              time_s=float(time) if time is not None else None,
                              width=width, height=height, conversion=conversion)
                ffmpeg.run(["ffmpeg", "-v", "error", "-y", "-i", str(original),
                            "-vf", f"format=rgb24,{rgba_filter(cube)}", "-frames:v", "1",
                            "-pix_fmt", "rgb24", "-threads", "1", "-update", "1", str(temporary)],
                           timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)
        check_source()
    return target
