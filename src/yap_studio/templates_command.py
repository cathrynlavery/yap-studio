"""Inspect card templates and render standalone previews."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from yap_studio import cards, ffmpeg, paths, receipt
from yap_studio.errors import InputError, OutputPathError, ValidationError, YapError
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.hf_registry import catalog
from yap_studio.style import cards_on, resolve

_PREVIEW_LOCK_TIMEOUT_S = 30.0


def _summary(template: str) -> dict:
    metadata = cards.load_template(template)
    rules = metadata["slots"]
    return {
        "id": metadata["id"],
        "version": metadata["version"],
        **{key: metadata[key] for key in ("label", "use") if key in metadata},
        "derived_from": [entry["item"] for entry in metadata["derived_from"]],
        "slots": list(rules),
        "required": [name for name, rule in rules.items() if rule["required"]],
        "variants": {
            name: rule["options"] for name, rule in rules.items() if rule["type"] == "enum"
        },
    }


def _preview_slots(template: str) -> dict[str, object]:
    if registry_cards.is_kind(template):
        return registry_cards.preview_slots(template)
    source = cards.template_dir(template) / "preview.json"
    return json.loads(source.read_text(encoding="utf-8"))["slots"]


def _details(template: str) -> dict:
    metadata = cards.load_template(template)
    if registry_cards.is_kind(template):
        name = registry_cards.item_name(template)
        entry = catalog.item(name)
        provenance = {"upstream": catalog.load_catalog()["registry"]}
        derived = [{"item": name, "upstream_path": entry["upstream_path"],
                    "changes": "Installed into the cache at setup, not vendored; "
                               "scripts and fonts are localized for rendering."}]
    else:
        source = cards.template_dir(template).parent / "PROVENANCE.json"
        provenance = (json.loads(source.read_text(encoding="utf-8"))
                      if metadata["derived_from"] else {"files": []})
        files = {(entry["item"], entry["upstream_path"]): entry for entry in provenance["files"]
                 if entry["yap_path"] == f"src/yap_studio/templates/{template}/template.html"}
        derived = [files[(entry["item"], entry["path"])] for entry in metadata["derived_from"]]
    return {
        "id": metadata["id"],
        "version": metadata["version"],
        **{key: metadata[key] for key in ("label", "use") if key in metadata},
        "derived_from": [entry["item"] for entry in metadata["derived_from"]],
        "slots": metadata["slots"],
        "preview_slots": _preview_slots(template),
        "box": metadata["box"],
        "provenance": {
            "upstream": {
                key: provenance["upstream"][key] for key in ("repository", "commit", "license")
            } if derived else None,
            "files": [
                {key: entry[key] for key in ("item", "upstream_path", "changes")}
                for entry in derived
            ],
        },
    }


def _slot_overrides(file: str | None) -> dict[str, object]:
    if file is None:
        return {}
    try:
        source = paths.validate_input(Path(file).expanduser())
        data = json.loads(source.read_text(encoding="utf-8"))
    except InputError as exc:
        raise ValidationError("slots", f"cannot read slots file {file}: {exc.reason}") from exc
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ValidationError("slots", f"slots file {file} must contain readable JSON") from exc
    if not isinstance(data, dict):
        raise ValidationError("slots", f"slots file {file} must contain an object of slot values")
    if "slots" in data:
        data = data["slots"]
    if not isinstance(data, dict):
        raise ValidationError("slots", f"slots file {file}: slots must be an object of slot values")
    return data


def _placeholder(out: Path) -> None:
    with paths.atomic_output(out) as temporary:
        ffmpeg.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                "-i", "testsrc2=size=1280x800:rate=1", "-frames:v", "1",
                "-c:v", "png", str(temporary),
            ],
            timeout_s=30,
        )


def placeholder_clip(out: Path, image: Path | None = None) -> None:
    """Make a short synthetic preview clip, optionally from the gallery still."""
    input_args = (["-loop", "1", "-i", str(image)] if image is not None else
                  ["-f", "lavfi", "-i", "testsrc2=size=640x400:rate=30"])
    video_filter = (
        "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1"
        if image is not None else
        "scale=640:400:force_original_aspect_ratio=decrease,"
        "pad=640:400:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    with paths.atomic_output(out) as temporary:
        ffmpeg.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *input_args,
                    "-vf", video_filter, "-t", "2", "-r", "30",
                    "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(temporary)],
                   timeout_s=30)


def _extract_still(movie: Path, still: Path, frame: int) -> None:
    with paths.atomic_output(still) as temporary:
        ffmpeg.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(movie),
                "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1",
                "-c:v", "png", "-pix_fmt", "rgba", str(temporary),
            ],
            timeout_s=30,
        )


@contextmanager
def _publish_lock(out: Path, template: str, timeout_s: float) -> Iterator[None]:
    """Serialize final output checks and publication, retaining the lock file."""
    lock_path = out / f".{template}.preview.lock"
    unsafe_message = f"Preview lock must be a regular file with a single link in {out}."
    if lock_path.parent != out:
        raise OutputPathError(unsafe_message, code="unsafe_path")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if os.name == "nt":
        if lock_path.is_symlink():
            raise OutputPathError(unsafe_message, code="unsafe_path")
    else:
        flags |= os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK, errno.EISDIR}:
            raise OutputPathError(unsafe_message, code="unsafe_path") from exc
        raise
    locked = False
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise OutputPathError(unsafe_message, code="unsafe_path")
        if os.name == "nt":
            import msvcrt
        else:
            import fcntl

        deadline = time.monotonic() + timeout_s
        while True:
            try:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise YapError(
                        f"Another preview holds the output lock for {template} in {out}.",
                        error_type="processing_error",
                        code="templates_preview_locked",
                        suggested_action="Wait for that run to finish, or choose a new --out.",
                    ) from exc
                time.sleep(min(0.05, remaining))
        locked = True
        yield
    finally:
        try:
            if locked:
                if os.name == "nt":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _publish_preview(work: Path, outputs: tuple[Path, ...], receipt_path: Path) -> None:
    owners = [receipt_path]
    for target in outputs:
        paths.validate_output(target, owner_receipts=owners)
    paths.validate_output(receipt_path)
    evidence = {
        "schema": "yap-studio/template-preview@1",
        "outputs": [
            {"path": target.name, "sha256": receipt.sha256_file(work / target.name)}
            for target in outputs
        ],
    }
    try:
        previous = (
            json.loads(receipt_path.read_bytes()).get("outputs", [])
            if receipt_path.exists() else []
        )
    except ValueError as exc:
        raise OutputPathError(
            "Preview receipt must contain valid JSON.", code="unsafe_path"
        ) from exc
    if not isinstance(previous, list):
        raise OutputPathError("Preview receipt outputs must be a list.", code="unsafe_path")
    # Retain both generations' ownership until every replacement succeeds.
    receipt.write_receipt(receipt_path, {**evidence, "outputs": previous + evidence["outputs"]})
    for target in outputs:
        with paths.atomic_output(target, owner_receipts=owners) as temporary:
            shutil.copyfile(work / target.name, temporary)
    receipt.write_receipt(receipt_path, evidence)


def _preview(args: argparse.Namespace) -> dict:
    metadata = cards.load_template(args.id)
    slots = {**_preview_slots(args.id), **_slot_overrides(args.slots)}
    style = resolve(args.style)
    frames = (cards.default_preview_frames(args.id, slots, style, fps=args.fps)
              if args.frames is None else args.frames)
    if not style.source.startswith(("preset:", "library:")):
        paths.validate_input(style.source)
    out = Path(args.out).expanduser()
    work: Path | None = None
    movie, still = out / f"{args.id}.mov", out / f"{args.id}.png"
    receipt_path = out / f"{args.id}.preview.json"
    try:
        paths.validate_output_dir(out)
        work = Path(tempfile.mkdtemp(prefix=".preview-work-", dir=out))
        paths.validate_output_dir(work)
        for name, rule in metadata["slots"].items():
            if rule["type"] == "image" and rule["required"] and name not in slots:
                placeholder = work / f"{name}.png"
                _placeholder(placeholder)
                slots[name] = placeholder
            if rule["type"] == "video" and "sources" not in rule and name not in slots:
                placeholder = work / f"{name}.mp4"
                placeholder_clip(placeholder)
                slots[name] = placeholder
        slots = cards.preview_videos(args.id, slots, work, frames=frames, fps=args.fps)
        plan = cards.card_layer_plan(args.id, slots, style, frames=frames, fps=args.fps)
        if plan.template == "file_drop":
            plan = replace(plan, variables={**plan.variables, "play_window": "{}"})
        if "window_fill" in plan.variables:
            plan = replace(plan, variables={**plan.variables, "window_fill": "preview"})
        with _publish_lock(out, args.id, _PREVIEW_LOCK_TIMEOUT_S):
            paths.validate_output(movie, owner_receipts=[receipt_path])
            paths.validate_output(still, owner_receipts=[receipt_path])
            paths.validate_output(receipt_path)
            rendered_movie, rendered_still = work / movie.name, work / still.name
            cards.render_card(plan, work, rendered_movie)
            _extract_still(rendered_movie, rendered_still, plan.frames * 3 // 5)
            _publish_preview(work, (movie, still), receipt_path)
    except OSError as exc:
        raise OutputPathError(f"Cannot write preview outputs: {exc}") from exc
    finally:
        if work is not None:
            try:
                shutil.rmtree(work)
            except OSError as exc:
                raise OutputPathError(f"Cannot remove preview work directory: {exc}") from exc
    return {
        "id": args.id,
        "mov": str(movie),
        "png": str(still),
        "box": plan.layer_box(),
        "frames": plan.frames,
        "fps": plan.fps,
        "style": style.name,
    }


def _print_details(data: dict) -> None:
    print(f"{data['id']} (version {data['version']})")
    for key in ("label", "use"):
        if key in data:
            print(f"{key.title()}: {data[key]}")
    if data["derived_from"]:
        print("Derived from: " + ", ".join(data["derived_from"]))
    for name, rule in data["slots"].items():
        fields = [
            f"type={rule['type']}",
            f"required={str(rule['required']).lower()}",
            "default=" + (json.dumps(rule["default"]) if "default" in rule else "(none)"),
        ]
        for key, value in rule.items():
            if key not in {"type", "required", "default"}:
                label = ("suffixes" if key == "options" and rule["type"] in {"image", "video"}
                         else key)
                fields.append(f"{label}={json.dumps(value)}")
        print(f"  {name}: " + ", ".join(fields))
    print("Preview slots: " + json.dumps(data["preview_slots"]))
    print("Box: " + json.dumps(data["box"]))
    upstream = data["provenance"]["upstream"]
    if upstream:
        print(f"Upstream repository: {upstream['repository']}")
        print(f"Pinned commit: {upstream['commit']}")
        print(f"License: {upstream['license']}")
    for entry in data["provenance"]["files"]:
        commit = entry.get("upstream_commit")
        suffix = f" at commit {commit}" if commit else ""
        print(f"  {entry['item']}: {entry['upstream_path']}{suffix}")
        print(f"    Changes: {entry['changes']}")


def _templates(args: argparse.Namespace) -> int:
    try:
        with paths.operation():
            if args.templates_command == "list":
                kinds = (*cards.CARD_TEMPLATES, *registry_cards.kinds())
                if args.project is not None:
                    from yap_studio.project.status import ProjectContext

                    project = Path(args.project).expanduser()
                    try:
                        context = ProjectContext(project)
                    except FileNotFoundError as exc:
                        raise ValidationError(
                            "project", f"project.json not found in {project}",
                        ) from exc
                    kinds = cards_on(context.render_style())
                elif args.style is not None:
                    style = resolve(args.style)
                    if not style.source.startswith(("preset:", "library:")):
                        paths.validate_input(style.source)
                    kinds = cards_on(style)
                data = {"templates": [_summary(template) for template in kinds]}
            elif args.templates_command == "show":
                data = _details(args.id)
            else:
                data = _preview(args)
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if isinstance(exc, OutputPathError) else 2
    if args.json:
        print(json.dumps(data, allow_nan=False))
    elif args.templates_command == "list":
        for template in data["templates"]:
            slots = [name + ("*" if name in template["required"] else "")
                     for name in template["slots"]]
            print(
                f"{template['id']} | derived from: {', '.join(template['derived_from'])}"
                f" | slots: {', '.join(slots)}"
                + (f" | {template['label']}: {template['use']}" if "label" in template else "")
            )
    elif args.templates_command == "show":
        _print_details(data)
    else:
        print(data["mov"])
        print(data["png"])
        print("Box: " + ", ".join(f"{key}={value}" for key, value in data["box"].items()))
        print(f"Frames: {data['frames']}; fps: {data['fps']}")
        print(f"Style: {data['style']}")
    return 0


def add_templates_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register template discovery, details, and standalone previews."""
    parser = subparsers.add_parser("templates", help="inspect and preview card templates")
    commands = parser.add_subparsers(dest="templates_command", required=True)
    listing = commands.add_parser("list", help="list card templates and their slots")
    source = listing.add_mutually_exclusive_group()
    source.add_argument("--style", metavar="SPEC",
                        help="use a standalone style without project taste rules")
    source.add_argument("--project", metavar="DIR",
                        help="use the project's render style and approved taste")
    showing = commands.add_parser("show", help="show slot rules, samples, and provenance")
    preview = commands.add_parser("preview", help="render a card movie and PNG still")
    for command in (showing, preview):
        command.add_argument("id", metavar="ID")
    for command in (listing, showing, preview):
        command.add_argument("--json", action="store_true")
        command.set_defaults(func=_templates)
    preview.add_argument("--style", metavar="SPEC")
    preview.add_argument("--slots", metavar="FILE.json")
    preview.add_argument("--frames", type=int, metavar="N")
    preview.add_argument("--fps", type=int, default=30, metavar="N")
    preview.add_argument("--out", required=True, metavar="DIR")
