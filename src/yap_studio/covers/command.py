"""Command line cover generation for a finished project."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from yap_studio import paths, receipt
from yap_studio.covers.layout import validate_text
from yap_studio.covers.pick import TimelineSnapshot, pick_frame
from yap_studio.covers.render import prepare_assets, render_cover
from yap_studio.errors import OutputPathError, ValidationError, YapError
from yap_studio.faces.project import cut_faces
from yap_studio.project.status import ProjectContext

SCHEMA = "yap-studio/cover@1"


def require_finished(project_dir: Path, name: str | None = None) -> tuple[Path, Path]:
    """Require assembled picture and its successful stage receipt."""
    project_name = name or project_dir.name
    final = project_dir / "final" / f"{project_name}.mp4"
    document = project_dir / "final/receipt.json"
    if not final.is_file() or not document.is_file():
        error = ValidationError("project", "finished video or final receipt missing; "
                                "run yap-studio assemble DIR")
        error.suggested_action = f"yap-studio assemble {project_dir}"
        raise error
    try:
        record = json.loads(document.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        record = None
    if not isinstance(record, dict) or (
        record.get("schema"), record.get("stage"), record.get("status")
    ) != ("yap-studio/stage-receipt@1", "assemble", "verified"):
        error = ValidationError("project", "final receipt is not a verified assembly")
        error.suggested_action = f"yap-studio assemble {project_dir}"
        raise error
    return final, document


def finished_snapshot(document: Path, cut: Path, project_dir: Path) -> TimelineSnapshot:
    """Bind cover frames to the cut and timeline recorded by assembly."""
    try:
        record = json.loads(document.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ValidationError("final receipt", "readable assembly receipt required") from exc
    inputs = record.get("inputs") if isinstance(record, dict) else None
    expected = inputs.get("media:cut/cut.mp4") if isinstance(inputs, dict) else None
    if not isinstance(expected, str) or receipt.sha256_file(cut) != expected:
        error = ValidationError("cut", "the cut changed after the final was assembled")
        error.suggested_action = f"yap-studio assemble {project_dir}"
        raise error
    try:
        return TimelineSnapshot.from_document(record.get("timeline"))
    except ValidationError as exc:
        exc.suggested_action = f"yap-studio assemble {project_dir}"
        raise


def check_owned_output(png: Path, document: Path) -> None:
    """Refuse replacement unless a cover receipt owns this PNG."""
    pending = document.with_name(f"{document.stem}.pending.json")
    if not png.exists() and not document.exists() and not pending.exists():
        return

    def read(path: Path) -> dict | None:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError, RecursionError):
            return None
        return saved if isinstance(saved, dict) and saved.get("schema") == SCHEMA else None

    saved = read(document)
    staged = read(pending)
    if staged is not None and staged.get("status") != "pending":
        staged = None
    if png.exists():
        digest = receipt.sha256_file(png)
        for record in (saved, staged):
            if record is not None and isinstance(record.get("outputs"), list) and any(
                item.get("path") == str(png) and item.get("sha256") == digest
                for item in record["outputs"] if isinstance(item, dict)
            ):
                return
        if saved is not None or staged is not None:
            raise OutputPathError("Existing cover PNG differs from its cover receipt.")
    elif saved is not None or staged is not None:
        return
    raise OutputPathError("Cover output exists without a readable cover receipt.")


def _run(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            started = datetime.now(UTC)
            root = Path(args.project_dir).expanduser().resolve()
            name = args.name
            if name is not None and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name) is None:
                raise ValidationError("name", "use 1..80 letters, digits, hyphens or underscores")
            # Flag errors do not depend on the presence of a project.
            values = {key: getattr(args, key) for key in (
                "headline", "kicker", "number", "label", "before", "after", "inset")}
            validate_text(args.look, values, measure=False)
            project_name = root.name
            project_file = root / "project.json"
            if project_file.is_file():
                try:
                    project_data = json.loads(project_file.read_text(encoding="utf-8"))
                    project_name = project_data["name"]
                except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
                    raise ValidationError("project.json", "readable project name required") from exc
            final, final_receipt = require_finished(root, project_name)
            if not project_file.is_file():
                raise ValidationError("project", "project.json is missing")
            cut = paths.validate_input(root / "cut/cut.mp4")
            final = paths.validate_input(final)
            final_receipt = paths.validate_input(final_receipt)
            timeline = finished_snapshot(final_receipt, cut, root)
            context = ProjectContext(root)
            style = context.render_style()
            inset = paths.validate_input(Path(args.inset).expanduser()) if args.inset else None
            output_name = name or context.project.name
            png = root / "covers" / f"{output_name}-{args.look}.png"
            document = png.with_suffix(".json")
            pending = png.with_name(f"{png.stem}.pending.json")
            check_owned_output(png, document)
            faces = cut_faces(root, cut)
            chosen = pick_frame(cut, faces, timeline, frame=args.frame)
            work_root = paths.validate_output_dir(root / ".yap/work")
            with tempfile.TemporaryDirectory(prefix="cover-", dir=work_root) as folder:
                work = Path(folder)
                face = (max(faces.frames[chosen.frame],
                            key=lambda item: item.box.w * item.box.h).box
                        if chosen.frame < len(faces.frames) and faces.frames[chosen.frame]
                        else None)
                box = (face.x, face.y, face.w, face.h) if face else None
                assets = prepare_assets(cut, chosen.frame, work, args.look,
                                        inset=inset, face_box=box)
                rendered = render_cover(work, assets, style, args.look,
                                        {key: value for key, value in values.items()
                                         if value is not None})
                matte_hash = receipt.sha256_file(assets.matte) if assets.matte else None
                paths.validate_output_dir(png.parent)
                outputs = ([{"path": str(png), "sha256": receipt.sha256_file(png)}]
                           if png.exists() else [])
                outputs.append({"path": str(png),
                                "sha256": receipt.sha256_file(rendered.png)})
                paths.atomic_write_text(pending, json.dumps({
                    "schema": SCHEMA, "status": "pending", "outputs": outputs,
                }, indent=2) + "\n")
                with paths.atomic_output(png, allowed_existing_suffixes=frozenset(),
                                         owner_receipts=(document, pending)) as temporary:
                    shutil.copyfile(rendered.png, temporary)
            ended = datetime.now(UTC)
            extra = {
                "schema": SCHEMA, "look": args.look, "frame": chosen.frame,
                "time_s": round(chosen.frame / 30, 6),
                "pick": {"reason": chosen.reason, "candidates": chosen.candidates},
                "text": {key: value for key, value in values.items() if value is not None},
                "style": {"name": style.name, "sha256": style.sha256()},
                "face_box": ({"source": "faces" if box is not None else "matte",
                              "box": list(assets.face_box)} if assets.face_box else None),
                "hashes": {"cut": receipt.sha256_file(cut),
                           "final": receipt.sha256_file(final),
                           "inset": receipt.sha256_file(inset) if inset else None,
                           "matte": matte_hash},
                "renderer_version": rendered.renderer_version,
                "checks": rendered.checks,
            }
            report = receipt.build_receipt(
                [cut, final, final_receipt, *( [inset] if inset else [])], [png],
                started, ended, extra=extra)
            paths.atomic_write_text(document, json.dumps(report, indent=2,
                                                         ensure_ascii=False) + "\n")
            pending.unlink()
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if isinstance(exc, OutputPathError) else 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    else:
        print(f"Wrote covers/{png.name}")
        print(f"Wrote covers/{document.name}")
    return 0


def add_cover_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register a single still-cover command."""
    parser = subparsers.add_parser("cover", help="make a checked cover for a finished project")
    parser.add_argument("project_dir", metavar="DIR")
    parser.add_argument("--look", choices=("photo", "number", "hook", "before-after"),
                        default="photo")
    parser.add_argument("--frame", type=int, metavar="N")
    parser.add_argument("--name")
    for flag in ("headline", "kicker", "number", "label", "before", "after", "inset"):
        parser.add_argument(f"--{flag}")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=_run)
