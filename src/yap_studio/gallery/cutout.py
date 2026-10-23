"""Cache a still with person alpha from the local Vision matte helper."""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from yap_studio import ffmpeg, paths, receipt
from yap_studio.errors import ProcessingError, YapError
from yap_studio.matte import vision
from yap_studio.matte.make import make_matte
from yap_studio.matte.model import MatteData

_SCHEMA = "yap-studio/gallery-cutout@1"


def prepare_cutout(
    still: Path, out_dir: Path, *, make: Callable[..., MatteData] = make_matte,
) -> Path | None:
    """Reuse a matching PNG or build one with Vision, without a model download."""
    if still.suffix.lower() == ".svg":
        return None
    published = False
    image = out_dir / "cutout" / "cutout.png"
    record = image.with_suffix(".json")
    try:
        reason = vision.unavailable_reason()
        if reason is not None:
            print("warning: preview cutout unavailable: " + " ".join(reason.splitlines()),
                  file=sys.stderr)
            return None
        digest = receipt.sha256_file(still)
        try:
            cached = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError, RecursionError):
            cached = None
        if (isinstance(cached, dict) and cached.get("schema") == _SCHEMA
                and cached.get("still_sha256") == digest and image.is_file()):
            return image
        # Invalidate both names before rebuilding a missing or mismatched entry.
        image.unlink(missing_ok=True)
        record.unlink(missing_ok=True)
        with paths.operation():
            paths.protect_input(still)
            folder = paths.validate_output_dir(image.parent)
            info = ffmpeg.probe(still, count_frames=False)
            if not info.width or not info.height:
                raise ProcessingError("ffprobe", 0, "cannot read the preview still size")
            with tempfile.TemporaryDirectory(prefix=".prepare-", dir=folder) as temporary:
                scratch = Path(temporary)
                clip = scratch / "still.mp4"
                ffmpeg.run([
                    "ffmpeg", "-v", "error", "-framerate", "30", "-i", str(still),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-frames:v", "1",
                    "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
                ], timeout_s=30)
                matte = make(clip, scratch, backend="vision")
                if matte.status != "measured" or matte.matte is None:
                    raise ProcessingError(
                        "cutout", 0, matte.reason or "person matte was not measured")
                payload = {"schema": _SCHEMA, "still_sha256": digest,
                           "width": info.width, "height": info.height, "matte": matte.to_dict()}
                with paths.atomic_output(image) as output:
                    ffmpeg.run([
                        "ffmpeg", "-y", "-v", "error", "-i", str(still),
                        "-i", str(scratch / matte.matte["file"]), "-filter_complex",
                        f"[0:v]format=rgb24[rgb];[1:v]scale={info.width}:{info.height},"
                        "format=gray[alpha];[rgb][alpha]alphamerge",
                        "-frames:v", "1", "-pix_fmt", "rgba", str(output),
                    ], timeout_s=30)
                published = True
                paths.atomic_write_text(record, json.dumps(payload, indent=2) + "\n")
        return image
    except (YapError, OSError) as exc:
        if published:
            with suppress(OSError):
                image.unlink(missing_ok=True)
        detail = (str(exc).splitlines() or [type(exc).__name__])[0]
        print(f"warning: preview cutout could not be prepared: {detail}", file=sys.stderr)
        return None
