"""Local contact sheets and speech evidence for cataloged footage."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from yap_studio import ffmpeg, hdr, paths, receipt, transcript
from yap_studio.canvas import canvas
from yap_studio.errors import (
    DependencyError,
    InputError,
    OutputPathError,
    ValidationError,
    YapError,
)
from yap_studio.media import catalog
from yap_studio.media.frame import extract_frame
from yap_studio.style.library import config_root

_PADDING = 8
_NO_ENGINE = (
    "no local speech engine is installed; install yap-studio[asr] to transcribe speech"
)


def _source(item: dict, document: dict) -> Path:
    locations = sorted(
        enumerate(item["locations"]), key=lambda pair: (pair[1]["seen_at"], pair[0]), reverse=True
    )
    for _, location in locations:
        candidate = Path(location["path"])
        try:
            if not candidate.is_file():
                continue
            digest, _ = catalog._hash_media(candidate, document["index"].get(str(candidate)))
            if digest == item["sha256"]:
                return paths.validate_input(candidate)
        except (OSError, InputError):
            continue
    last = locations[0][1]["path"] if locations else item["id"]
    raise InputError(
        last, "no catalog location still holds this file; run yap-studio media add on its folder"
    )


def _reusable(item: dict, folder: Path, params: dict) -> bool:
    if (folder / "receipt.pending.json").exists():
        return False
    try:
        saved = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or saved.get("schema") != "yap-studio/receipt@1":
            return False
        evidence = saved.get("catalog", item)
        sheet, speech = evidence["sheet"], evidence["speech"]
        if sheet is None or sheet["path"] != str(folder / "sheet.jpg"):
            return False
        if item["kind"] == "video" and speech is None:
            return False
        if speech is not None and speech["status"] in {"failed", "no_engine"}:
            return False
        if saved.get("media") != {"id": item["id"], "sha256": item["sha256"]}:
            return False
        if saved.get("sheet") != params:
            return False
        outputs = saved.get("outputs")
        if not isinstance(outputs, list) or any(not isinstance(entry, dict) for entry in outputs):
            return False
        recorded = {entry.get("path"): entry.get("sha256") for entry in outputs}
        if receipt.sha256_file(folder / "sheet.jpg") != sheet["sha256"]:
            return False
        if recorded.get(sheet["path"]) != sheet["sha256"]:
            return False
        if speech is not None and speech["status"] == "words":
            target = folder / "transcript.json"
            if speech["path"] != str(target) or recorded.get(str(target)) is None:
                return False
            if receipt.sha256_file(target) != recorded[str(target)]:
                return False
        item.update(sheet=sheet, speech=speech)
        return True
    except (OSError, ValueError, UnicodeError, TypeError, KeyError, RecursionError):
        return False


def _publication_record(target: Path) -> dict | None:
    try:
        saved = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise InputError(str(target), "cannot read sheet publication bookkeeping") from exc
    allowed = {str(target.parent / name) for name in ("sheet.jpg", "transcript.json")}
    if (
        not isinstance(saved, dict)
        or saved.get("schema") not in ("yap-studio/receipt@1", "yap-studio/stage-pending@1")
        or not isinstance(saved.get("outputs"), list)
        or any(
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or entry["path"] not in allowed
            for entry in saved["outputs"]
        )
    ):
        raise InputError(str(target), "invalid sheet publication bookkeeping")
    return saved


def _start_publication(receipt_path: Path, pending_path: Path, evidence: dict) -> dict:
    """Carry ownership forward and recover a crash after the final receipt write."""
    previous = _publication_record(receipt_path)
    pending = _publication_record(pending_path)
    if previous is not None:
        if pending is not None:
            latest = {
                entry["path"]: entry.get("new_sha256", entry.get("sha256"))
                for entry in pending["outputs"]
            }
            try:
                matching = (
                    previous.get("started_at") == pending.get("started_at")
                    and previous.get("media") == pending.get("media")
                    and all(
                        entry["sha256"] == latest.get(entry["path"])
                        and receipt.sha256_file(entry["path"]) == entry["sha256"]
                        for entry in previous["outputs"]
                    )
                )
            except (OSError, KeyError):
                matching = False
            if not matching:
                raise InputError(str(pending_path), "receipt and journal conflict")
            pending_path.unlink()
        paths.validate_output(receipt_path)
        paths.validate_output(pending_path)
        os.replace(receipt_path, pending_path)
        pending = previous
    pending = pending or {"outputs": []}
    pending.update({
        "schema": "yap-studio/stage-pending@1", "started_at": evidence["started_at"],
        "media": evidence["media"],
    })
    paths.atomic_write_text(pending_path, json.dumps(pending, indent=2) + "\n")
    return pending


def _journal_output(target: Path, digest: str | None, pending_path: Path, pending: dict) -> None:
    """Validate ownership before recording the hashes allowed across a replacement."""
    paths.validate_output(target, owner_receipts=[target.parent / "receipt.json", pending_path])
    pending["outputs"].append({
        "path": str(target.absolute()),
        "old_sha256": receipt.sha256_file(target) if target.exists() else None,
        "new_sha256": digest,
    })
    paths.atomic_write_text(pending_path, json.dumps(pending, indent=2) + "\n")


def _tile_size(item: dict, columns: int, rows: int) -> tuple[int, int]:
    box_width = (1600 - (columns + 1) * _PADDING) // columns
    box_height = (1600 - (rows + 1) * _PADDING) // rows
    factor = min(box_width / item["width"], box_height / item["height"])
    # Even tile dimensions also work with JPEG encoders using chroma subsampling.
    return (
        max(2, int(item["width"] * factor) // 2 * 2),
        max(2, int(item["height"] * factor) // 2 * 2),
    )


def _render_clip(
    source: Path, item: dict, count: int, output: Path, *, shape: str = "vertical",
) -> tuple[list[dict], bool | None]:
    info = ffmpeg.probe(source)
    if info.video_frames is not None and info.video_frames > 0:
        count = min(count, info.video_frames)
    duration = item["duration_s"]
    if duration <= 0:
        raise InputError(str(source), "clip has no positive duration; add a playable clip")
    wide = shape == "horizontal"
    columns = min(3 if wide else 2 if item["orientation"] == "horizontal" else 3, count)
    rows = math.ceil(count / columns)
    width, height = _tile_size(item, columns, rows)
    if wide:
        frame = canvas(shape)
        width, height = frame.width // 4, frame.height // 4
    frames = [{
        "index": i, "time_s": duration * (2 * i + 1) / (2 * count),
        "column": i % columns, "row": i // columns,
    } for i in range(count)]
    conversion = None
    colour_converted = None if item["hdr"] is None else False
    if item["hdr"] is not None:
        try:
            conversion = hdr.conversion(info)
        except hdr.HdrUnsupportedError:
            pass
    with tempfile.TemporaryDirectory(prefix="yap-sheet-") as temporary:
        tiles = Path(temporary)

        def extract(selected: hdr.HdrConversion | None) -> None:
            for frame in frames:
                tile = tiles / f"tile-{frame['index']:02d}.png"
                extract_frame(source, info, tile, time_s=frame["time_s"], width=width,
                              height=height, conversion=selected, rotation=item["rotation"])

        try:
            extract(conversion)
            if conversion is not None:
                colour_converted = True
        except (YapError, OSError):
            if conversion is None:
                raise
            # Re-extract every tile so the sheet has one consistent colour treatment.
            extract(None)
        ffmpeg.run([
            "ffmpeg", "-v", "error", "-y", "-framerate", "1",
            "-noautorotate", "-i", str(tiles / "tile-%02d.png"), "-map", "0:v:0",
            "-vf", f"tile={columns}x{rows}:nb_frames={count}:padding={_PADDING}:"
            f"margin={_PADDING}:color=black" + (
                ",scale=1600:900:force_original_aspect_ratio=decrease,"
                "pad=1600:900:(ow-iw)/2:(oh-ih)/2,setsar=1" if wide else ""),
            "-frames:v", "1", "-threads", "1", "-q:v", "2", "-update", "1", str(output),
        ], timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)
    return frames, colour_converted


def _render_still(source: Path, output: Path, *, shape: str = "vertical") -> None:
    stream = "0:v:0"
    if catalog.heif_grid(source) is not None:
        stream = "0:g:0"
    # The decoder's upright dimensions drive scaling, including HEIF grid pictures.
    filters = f"[{stream}]scale=w='min(1600,iw)':h='min(1600,ih)':"
    filters += "force_original_aspect_ratio=decrease,setsar=1[out]"
    if shape == "horizontal":
        filters = (f"[{stream}]scale=1600:900:force_original_aspect_ratio=decrease,"
                   "pad=1600:900:(ow-iw)/2:(oh-ih)/2,setsar=1[out]")
    ffmpeg.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(source),
        "-filter_complex", filters, "-map", "[out]", "-frames:v", "1",
        "-threads", "1", "-q:v", "2", "-update", "1", str(output),
    ], timeout_s=ffmpeg.DEFAULT_TIMEOUT_S)


def _speech(
    source: Path, item: dict, asr: str, at: str,
    backend_factory: Callable[[str], transcript.Backend | None] | None,
) -> tuple[dict | None, transcript.Transcript | None]:
    if item["kind"] == "image":
        return None, None
    result = {
        "status": "no_audio", "engine": None, "model": None, "words": 0,
        "text": "", "path": None, "message": None, "at": at,
    }
    if not item["has_audio"]:
        return result, None
    if asr == "none":
        result.update(status="skipped", message="speech transcription was turned off")
        return result, None
    try:
        backend = (backend_factory or transcript.resolve_backend)(asr)
    except DependencyError as exc:
        result.update(status="no_engine", message=exc.message)
        return result, None
    if backend is None:
        result.update(status="no_engine", message=_NO_ENGINE)
        return result, None
    result.update(engine=backend.name, model=backend.model)
    try:
        words = backend.transcribe_many([source])[source]
    except (YapError, OSError) as exc:
        result.update(status="failed", message=str(exc))
        return result, None
    if not words.words:
        result["status"] = "no_speech"
        return result, None
    result.update(status="words", words=len(words.words), text=words.text[:4000])
    return result, words


def make_sheet(
    item_id: str, *, frames: int = 6, asr: str = "auto", force: bool = False,
    backend_factory: Callable[[str], transcript.Backend | None] | None = None,
    now: Callable[[], datetime] | None = None,
    shape: str = "vertical",
) -> dict:
    """Render local viewing evidence and store it with one short catalog update."""
    canvas(shape)
    if type(frames) is not int or not 1 <= frames <= 12:
        raise ValidationError("frames", "must be an integer from 1 to 12")
    if asr not in ("auto", *transcript.ENGINES, "none"):
        raise ValidationError(
            "asr", "choose auto, parakeet-mlx, parakeet-onnx, faster-whisper, or none"
        )
    clock = now or (lambda: datetime.now(UTC))
    with paths.operation():
        document = catalog.load_catalog()
        item = next((entry for entry in document["items"] if entry["id"] == item_id), None)
        if item is None:
            raise catalog.UnknownMediaError(item_id)
        source = _source(item, document)

        def store(current: dict) -> None:
            found = next((entry for entry in current["items"] if entry["id"] == item_id), None)
            if found is None:
                raise catalog.UnknownMediaError(item_id)
            if found["sha256"] != item["sha256"]:
                raise InputError(str(source), "catalog media changed; run media sheet again")
            found.update(sheet=sheet, speech=speech)

        folder = (config_root() / "assets" / "sheets" / item_id).absolute()
        params = {"frames": frames, "asr": asr}
        if shape == "horizontal":
            folder = folder / shape
            params["shape"] = shape
        previous_evidence = item["sheet"], item["speech"]
        if not force and _reusable(item, folder, params):
            sheet, speech = item["sheet"], item["speech"]
            if (sheet, speech) != previous_evidence:
                catalog.update_catalog(store)
            return {
                "id": item_id, "sheet": item["sheet"], "speech": item["speech"],
                "noop": True, "folder": str(folder),
            }
        paths.validate_output_dir(folder)
        sheet_path, transcript_path = folder / "sheet.jpg", folder / "transcript.json"
        receipt_path = folder / "receipt.json"
        pending_path = folder / "receipt.pending.json"
        started = clock()
        at = started.replace(tzinfo=started.tzinfo or UTC).astimezone(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with tempfile.TemporaryDirectory(prefix="yap-sheet-") as temporary:
            rendered = Path(temporary) / "sheet.jpg"
            if item["kind"] == "video":
                tiles, colour_converted = _render_clip(source, item, frames, rendered, shape=shape)
            else:
                _render_still(source, rendered, shape=shape)
                tiles = [{"index": 0, "time_s": None, "column": 0, "row": 0}]
                colour_converted = None
            speech, words = _speech(source, item, asr, at, backend_factory)
            outputs = [rendered]
            rendered_transcript = Path(temporary) / "transcript.json"
            if words is not None:
                words.save(rendered_transcript)
                speech["path"] = str(transcript_path)
                outputs.append(rendered_transcript)
            info = ffmpeg.probe(rendered, count_frames=False)
            # Hash the source and staged bytes before replacing any published output.
            evidence = receipt.build_receipt(
                [source], outputs, started, clock(),
                extra={"media": {"id": item_id, "sha256": item["sha256"]}, "sheet": params},
            )
            if evidence["inputs"][0]["sha256"] != item["sha256"]:
                raise InputError(
                    str(source), "media changed while making its sheet; run media add again"
                )
            evidence["outputs"][0]["path"] = str(sheet_path)
            if words is not None:
                evidence["outputs"][1]["path"] = str(transcript_path)
            sheet = {
                "path": str(sheet_path), "sha256": evidence["outputs"][0]["sha256"],
                "width": info.width, "height": info.height, "frames": tiles,
                "colour_converted": colour_converted, "at": at,
            }
            evidence["catalog"] = {"sheet": sheet, "speech": speech}
            paths.validate_output(receipt_path)
            owners = [receipt_path, pending_path]
            try:
                # Check all replacements and deletions before changing publication state.
                paths.validate_output(sheet_path, owner_receipts=owners)
                if words is not None or transcript_path.exists():
                    paths.validate_output(transcript_path, owner_receipts=owners)
                pending = _start_publication(receipt_path, pending_path, evidence)
                _journal_output(sheet_path, sheet["sha256"], pending_path, pending)
                with paths.atomic_output(sheet_path, owner_receipts=owners) as output:
                    shutil.copyfile(rendered, output)
                if words is not None:
                    _journal_output(
                        transcript_path, evidence["outputs"][1]["sha256"], pending_path, pending,
                    )
                    with paths.atomic_output(transcript_path, owner_receipts=owners) as output:
                        shutil.copyfile(rendered_transcript, output)
                elif transcript_path.exists():
                    _journal_output(transcript_path, None, pending_path, pending)
                    transcript_path.unlink()
            except OutputPathError as exc:
                error = OutputPathError(exc.message, code=exc.code)
                error.suggested_action = (
                    f"Delete the sheet folder {folder}, "
                    f"then run yap-studio media sheet {item_id} again."
                )
                raise error from exc
            receipt.write_receipt(receipt_path, evidence)
            pending_path.unlink()

        catalog.update_catalog(store)
        return {
            "id": item_id, "sheet": sheet, "speech": speech, "noop": False, "folder": str(folder),
        }
