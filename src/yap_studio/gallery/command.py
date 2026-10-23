"""Run the local style gallery."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

from yap_studio.errors import ValidationError, YapError

_STILL_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
IDLE_AFTER_SAVE_S = 120


def pick(args: argparse.Namespace) -> int:
    """Serve the gallery until Ctrl-C."""
    from yap_studio import background
    from yap_studio.gallery import detach

    if args.stop or args.detach:
        if not background.posix_supported():
            background.print_error(YapError(
                "background style pages need macOS or Linux", code="background_unsupported",
                suggested_action="run style pick without --detach and keep it open",
            ), json_output=args.json)
            return 2
        if args.stop:
            incompatible = (args.detach or args.phone or args.host is not None
                            or args.port is not None or args.inbox is not None
                            or args.still is not None or args.style is not None
                            or args.workspace is not None)
            if incompatible:
                background.print_error(ValidationError(
                    "stop", "cannot combine --stop with --detach, --phone, --host, --port, "
                    "--inbox, --still, --style or --workspace",
                ), json_output=args.json)
                return 2
            return detach.stop(json_output=args.json)
        return detach.start(args)
    if args.port is None:
        args.port = 0
    # Keep the renderer and server out of CLI parser construction.
    from yap_studio.gallery import still
    from yap_studio.gallery.cutout import prepare_cutout
    from yap_studio.gallery.live import SAMPLE_SENTENCE
    from yap_studio.gallery.server import GalleryServer
    from yap_studio.render import hyperframes
    from yap_studio.style import (
        PRESETS_DIR,
        StyleError,
        current_default_data,
        library_dir,
        read_style_file,
        resolve,
        unbranded,
    )
    from yap_studio.transcript import resolve_backend
    from yap_studio.transcript.take_job import TakeJobs

    jobs = TakeJobs()
    work: tempfile.TemporaryDirectory | None = None
    server: GalleryServer | None = None
    cleanup_mapping = None
    phone_via = None
    public_base = None
    stopping = False
    serving = False

    def stop_on_sigterm(_signum: int, _frame: object) -> None:
        nonlocal stopping
        if serving:
            raise KeyboardInterrupt
        stopping = True

    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        if args.phone and args.host is not None:
            raise ValidationError("phone", "use --phone or --host, not both")
        install = hyperframes.installed()
        if install is None:
            raise ValidationError("renderer", "the pinned renderer is missing; "
                                  "run yap-studio renderer install")
        gsap = hyperframes.gsap_file(install)
        inbox = Path(args.inbox).expanduser() if args.inbox else None
        workspace = Path(args.workspace).expanduser() if args.workspace else None
        if inbox is None and args.still is None and Path("inbox").is_dir():
            inbox = Path("inbox")
            if workspace is None:
                workspace = Path(".")
        if inbox is not None and not inbox.is_dir():
            raise ValidationError("inbox", "choose an existing inbox folder")
        if args.still is not None:
            image = Path(args.still).expanduser()
            if not image.is_file() or image.suffix.lower() not in _STILL_SUFFIXES:
                raise ValidationError("still", "use an existing PNG, JPEG or WebP image")
        else:
            image = still.sample_still()
        if workspace is None:
            work = tempfile.TemporaryDirectory(prefix="yap-studio-gallery-")
            work_dir = Path(work.name)
        else:
            work_dir = workspace / ".yap-studio" / "gallery"
            work_dir.mkdir(parents=True, exist_ok=True)
        newest = still.newest_take(inbox) if inbox is not None else None
        preview = {"source": "sample", "sentence": SAMPLE_SENTENCE, "words": [],
                   "pending": newest is not None, "generation": 0, "transcript": None}
        if inbox is not None:
            previous = work_dir / "preview.json"
            if previous.is_file():
                try:
                    cached = json.loads(previous.read_text(encoding="utf-8"))
                    cached_image = Path(cached["still"])
                    take_stat = newest.stat() if newest is not None else None
                    if (cached.get("schema") == "yap-studio/gallery-preview@1"
                            and cached.get("source") == "take"
                            and cached.get("take") == str(newest)
                            and take_stat is not None
                            and cached.get("take_size") == take_stat.st_size
                            and cached.get("take_mtime_ns") == take_stat.st_mtime_ns
                            and cached_image.is_file()):
                        image, preview = cached_image, {**cached, "generation": 0}
                except (OSError, ValueError, KeyError, TypeError):
                    pass
        # Keep the look's own colors in the editable style beneath the brand layer.
        if args.style:
            source = resolve(args.style).source
        else:
            current_default_data()
            source = resolve().source
        if source.startswith("preset:"):
            raw = read_style_file(PRESETS_DIR / f"{source.removeprefix('preset:')}.json")
        elif source.startswith("library:"):
            raw = read_style_file(library_dir() / f"{source.removeprefix('library:')}.json")
        else:
            raw = read_style_file(source)
        selected = unbranded(raw)
        if not args.style and source.startswith("preset:") and workspace is not None:
            from yap_studio.style.answers import load_answers, style_changes

            answer_path = workspace / "style/answers.json"
            try:
                changes = (style_changes(load_answers(answer_path), selected)
                           if answer_path.is_file() else {})
            except (StyleError, OSError, ValueError):
                changes = {}
            for section, values in changes.items():
                if section != "pending":
                    selected[section] = {**selected.get(section, {}), **values}
        from_library = (source.startswith("library:") or (
            not source.startswith("preset:")
            and Path(source).resolve().parent == library_dir().resolve()
        ))

        def start(host: str, allowed_hosts: tuple[str, ...] = ()) -> GalleryServer:
            return GalleryServer(image, selected, work_dir, install=install, gsap=gsap,
                                 workspace=workspace, host=host, port=args.port,
                                 preview=preview, allowed_hosts=allowed_hosts,
                                 from_library=from_library)

        if args.phone:
            from yap_studio.review import phone

            name = phone.tailscale_name()
            if name:
                server = start("127.0.0.1", (name,))
                try:
                    public_base, cleanup_mapping = phone.tailscale_origin(server.port)
                    phone_via = "tailscale"
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    server.server_close()
                    server = None
                    print(f"warning: Tailscale link failed ({exc}); using Wi-Fi instead",
                          file=sys.stderr)
            if server is None:
                address = phone.lan_address()
                if address is None:
                    raise phone.no_network()
                server = start(address)
                phone_via = "wifi"
        else:
            server = start(args.host or "127.0.0.1")
        def prepare() -> None:
            take = None
            if inbox is not None:
                take, failed = newest, False
                try:
                    result = still.prepare_preview(inbox, work_dir)
                    server.update_preview(result)
                except (OSError, YapError, ValueError) as exc:
                    server.fail_preview()
                    print(f"error: preview could not be prepared: {exc}", file=sys.stderr)
                    failed = True
                else:
                    take = result.get("take") if result.get("source") == "take" else None
                try:
                    if take:
                        jobs.start(Path(take), resolve_backend("auto"))
                except (OSError, YapError, ValueError) as exc:
                    print(f"warning: take transcript could not be started: {exc}", file=sys.stderr)
                if failed:
                    return
            generation, current_image = server.preview_image()
            if (cutout := prepare_cutout(current_image, work_dir)) is not None:
                server.update_cutout(generation, cutout)
            if take and args.still is None:
                from yap_studio.matte import chroma

                try:
                    detected = chroma.detect(Path(take))
                    server.update_screen(generation, detected["screen"] if detected else None)
                except (YapError, OSError, ValueError) as exc:
                    print(f"warning: screen could not be checked: {exc}", file=sys.stderr)
        threading.Thread(target=prepare, daemon=True).start()
    except (YapError, OSError, ValueError) as exc:
        action = getattr(exc, "suggested_action", None)
        if "renderer" in str(exc).lower():
            fix = "run yap-studio renderer install"
        else:
            fix = action or "check the paths and style, then try again"
        if args.detached_record is not None:
            if not isinstance(exc, YapError):
                exc = YapError(str(exc), code="style_pick_start_failed")
            exc.suggested_action = fix
            background.print_error(exc, json_output=True)
        else:
            print(f"error: {getattr(exc, 'message', str(exc))}", file=sys.stderr)
            print(f"fix: {fix}", file=sys.stderr)
        try:
            if server is not None:
                server.server_close()
        finally:
            try:
                if cleanup_mapping is not None:
                    cleanup_mapping()
            finally:
                try:
                    jobs.close()
                except OSError as exc:
                    print(f"warning: take transcript could not be stopped: {exc}", file=sys.stderr)
                finally:
                    try:
                        if work is not None:
                            work.cleanup()
                    finally:
                        if previous_sigterm is not None:
                            signal.signal(signal.SIGTERM, previous_sigterm)
        return 2
    url = server.url
    if public_base is not None:
        local = urlsplit(url)
        url = f"{public_base}{local.path}?{local.query}"
    idle_done = threading.Event()
    try:
        result = {"url": url}
        if phone_via is not None:
            result["phone"] = {"via": phone_via, "url": url}
        if args.json:
            print(json.dumps(result), flush=True)
        elif phone_via is not None:
            print(f"Open on your phone: {phone.link_label(phone_via)}: {url}", flush=True)
            print("Press Ctrl-C to stop.", flush=True)
        else:
            print(url, flush=True)
            print("Press Ctrl-C to stop.", flush=True)
        if args.detached_record is not None:
            background.write_record(Path(args.detached_record), result, cwd=os.getcwd())
        if stopping:
            raise KeyboardInterrupt
        if not args.no_open:
            webbrowser.open(url)
        serving = True
        if stopping:
            raise KeyboardInterrupt
        if args.detached_record is not None:
            def stop_when_idle() -> None:
                while not idle_done.wait(min(1, IDLE_AFTER_SAVE_S / 2)):
                    if (server.saved_at is not None
                            and time.monotonic() - server.last_request >= IDLE_AFTER_SAVE_S):
                        server.shutdown()
                        return
            threading.Thread(target=stop_when_idle, daemon=True).start()
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        idle_done.set()
        serving = False
        try:
            server.server_close()
        finally:
            try:
                if cleanup_mapping is not None:
                    cleanup_mapping()
            finally:
                if previous_sigterm is not None:
                    signal.signal(signal.SIGTERM, previous_sigterm)
                try:
                    jobs.close()
                except OSError as exc:
                    print(f"warning: take transcript could not be stopped: {exc}", file=sys.stderr)
                finally:
                    try:
                        if work is not None:
                            work.cleanup()
                    finally:
                        if args.detached_record is not None:
                            background.remove_record(Path(args.detached_record))
    return 0


def add_pick_parser(style_commands: argparse._SubParsersAction, port_type) -> None:
    """Register `style pick` without importing gallery internals."""
    parser = style_commands.add_parser("pick", help="pick a style over a still from your take")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--inbox", metavar="DIR")
    source.add_argument("--still", metavar="IMAGE")
    parser.add_argument("--style", metavar="SPEC")
    parser.add_argument("--workspace", metavar="DIR")
    parser.add_argument("--host", metavar="HOST")
    parser.add_argument("--phone", action="store_true",
                        help="print a phone link through Tailscale or the same Wi-Fi")
    parser.add_argument("--port", type=port_type)
    parser.add_argument("--detach", action="store_true", help="keep serving in the background")
    parser.add_argument("--stop", action="store_true", help="stop background style pages")
    parser.add_argument("--detached-record", help=argparse.SUPPRESS)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=pick)
