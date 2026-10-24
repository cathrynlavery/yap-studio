"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import signal
import subprocess
import sys
import threading
import webbrowser
from collections.abc import Sequence
from contextlib import redirect_stderr
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import numpy as np

from yap_studio import install_info, paths, setup_checks, tools
from yap_studio.checks.platform import PLATFORMS
from yap_studio.errors import DependencyError, InputError, ValidationError, YapError
from yap_studio.ffmpeg import check_version, has_filters, resolve_tool, tool_versions
from yap_studio.project.receipts import failed_assert_lines
from yap_studio.report import exit_code, summarize
from yap_studio.style import (
    SHIPPED_PRESETS,
    StyleRefusal,
    add,
    create,
    get_default,
    list_entries,
    load_style,
    read_style_file,
    resolve,
    save_pick,
    set_default,
    write_resolved,
)
from yap_studio.transcript import available_engines, resolve_backend
from yap_studio.verify import verify

if TYPE_CHECKING:
    from yap_studio.render.hyperframes import RendererInstall
    from yap_studio.review.server import ReviewVideo
    from yap_studio.review.store import ReviewStore

_REQUIRED_FILTERS = ("ebur128", "blackdetect", "freezedetect", "aresample")
_ASR_CHOICES = ("auto", "parakeet-mlx", "parakeet-onnx", "faster-whisper", "none")
_PLATFORM_CHOICES = ("none", *PLATFORMS)


class _VersionAction(argparse.Action):
    """Resolve install identity only when the version flag is used."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ) -> None:
        """Print the running install identity and exit successfully."""
        print(install_info.version_text())
        parser.exit(status=0)


def _mcp(_args: argparse.Namespace) -> int:
    try:
        from yap_studio.mcp_server import run_stdio

        return run_stdio()
    except DependencyError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2


def _verify(args: argparse.Namespace) -> int:
    logger = logging.getLogger("yap_studio")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("yap-studio: %(message)s"))
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        result = verify(
            media=args.media,
            edl=args.edl,
            source=args.source,
            asr=args.asr,
            asr_model=args.asr_model,
            loudness_target=args.loudness_target,
            loudness=args.loudness,
            platform=args.platform,
            strict=args.strict,
            report=args.report,
            artifacts=args.artifacts,
            detail=args.detail,
        )
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(summarize(result))
    return exit_code(result, args.strict)


def _doctor(args: argparse.Namespace) -> int:
    """Check the installed engine and the dependencies needed by editing."""
    lines = [f"install  {install_info.version_text()}"]
    running_root = install_info.source_checkout()
    running_version = install_info.running_version(running_root)
    warnings: list[str] = []
    checkout = install_info.checkout_version(Path.cwd())
    if checkout is not None:
        root, version = checkout
        if (
            install_info.compare_versions(running_version, version) == -1
            and running_root != root
        ):
            warnings.append(
                f"the installed yap-studio {running_version} is older "
                f"than this checkout ({version}). Run the checkout with uv run, or reinstall "
                f"with: uv tool install --force '{root}[asr,mcp]'"
            )
    cwd = Path.cwd()
    workspace = install_info.workspace_version(cwd)
    workspace_matches = workspace is not None and workspace[1] == running_version
    if workspace is not None and not workspace_matches:
        warnings.append(
            f"workspace version {workspace[1]} differs from installed yap-studio "
            f"{running_version}. Run python3 setup.py from the workspace."
        )
    service = setup_checks.synced_service(cwd)
    free_gb = setup_checks.free_disk_gb(cwd)
    disk_enough = free_gb >= setup_checks.MIN_FREE_GB
    lines.extend(f"warning: {warning}" for warning in warnings)
    lines.append(f"folder   {cwd}   {service or 'local'}")
    lines.append(f"disk     {free_gb:.1f} GB free; {setup_checks.MIN_FREE_GB} GB needed")
    if workspace is not None:
        lines.append(f"workspace {'matches' if workspace_matches else 'version differs'}")
    problems: list[YapError | str] = []
    tool_paths: dict[str, str | None] = {}
    tool_ok: dict[str, bool] = {}
    for name in ("ffmpeg", "ffprobe"):
        tool_paths[name] = None
        tool_ok[name] = False
        try:
            tool_paths[name] = resolve_tool(name)
            check_version(tool_paths[name])
            tool_ok[name] = True
        except DependencyError as exc:
            problems.append(exc)
    versions = tool_versions()
    for name in ("ffmpeg", "ffprobe"):
        lines.append(f"{name:<9}{tool_paths[name] or 'not found'}   {versions[name] or 'unknown'}")
        if tool_paths[name] is not None and versions[name] is None:
            problems.append(f"could not read the {name} version")

    filters = {name: False for name in _REQUIRED_FILTERS}
    supported_ffmpeg = tool_paths["ffmpeg"] is not None and not any(
        isinstance(problem, DependencyError) and problem.code == "ffmpeg_too_old"
        for problem in problems
    )
    if supported_ffmpeg:
        try:
            filters = has_filters(_REQUIRED_FILTERS)
        except YapError as exc:
            problems.append(exc)
    missing_filters = [name for name, available in filters.items() if not available]
    if missing_filters and supported_ffmpeg:
        problems.append("missing required filters: " + ", ".join(missing_filters))
    filter_text = ", ".join(
        f"{name} {'ok' if available else 'missing'}" for name, available in filters.items()
    )
    if tool_paths["ffmpeg"] is not None and not supported_ffmpeg:
        filter_text = "not checked (ffmpeg is too old)"
    lines.append(f"filters  {filter_text}")
    lines.append(f"python   {platform.python_version()}")
    lines.append(f"numpy    {np.__version__}")
    speech = available_engines()
    auto = resolve_backend()
    model_ready = setup_checks.speech_model_ready(auto)
    cutout_reason = setup_checks.cutout_reason()
    if auto is not None and not model_ready:
        warning = ("speech model is not downloaded yet; first ingest downloads it. "
                   "Run yap-studio speech download to prepare it now.")
        warnings.append(warning)
        lines.append(f"warning: {warning}")
    if cutout_reason is not None:
        warning = f"cutout helper unavailable: {cutout_reason}"
        warnings.append(warning)
        lines.append(f"warning: {warning}")
    engine_text = ", ".join(
        f"{name} {'yes' if available else 'no'}" for name, available in speech.items()
    )
    lines.append(f"speech   {engine_text}; auto picks {auto.name if auto is not None else 'none'}")
    lines.append(f"model    {'ready' if model_ready else 'missing'}")
    lines.append(f"cutout   {'ready' if cutout_reason is None else cutout_reason}")
    from yap_studio.render import hyperframes

    node = tools.find_tool("node")
    npm = tools.find_tool("npm")
    node_version = None
    if node is not None:
        try:
            node_version = tools.run_tool("node", ["--version"], timeout_s=5).stdout.strip()
        except (YapError, OSError, ValueError):
            pass
    match = re.fullmatch(r"v?(\d+)\.\d+\.\d+(?:[-+].*)?", node_version or "")
    node_ok = npm is not None and match is not None and int(match[1]) >= tools.MIN_NODE_MAJOR
    renderer = hyperframes._npm_installed()
    browser_ok = hyperframes.installed() is not None
    lines.append(f"node     {node_version or 'not found'}")
    lines.append(f"renderer {'ready' if renderer is not None and browser_ok else 'missing'}")
    from yap_studio.hf_registry import install as registry_install

    registry = registry_install.status()
    if registry["changed"]:
        lines.append("registry changed: " + ", ".join(registry["changed"]))
    elif registry["missing"]:
        lines.append(f"registry missing {len(registry['missing'])}")
    else:
        lines.append(f"registry ready ({len(registry['ready'])} items)")
    from yap_studio import elevenlabs

    elevenlabs_cli = tools.find_tool("elevenlabs-pp-cli") is not None
    elevenlabs_signed_in = False
    if elevenlabs_cli:
        try:
            tools.run_tool("elevenlabs-pp-cli", ["auth", "status", "--json"], timeout_s=10)
            elevenlabs_signed_in = True
        except (YapError, OSError, ValueError):
            pass
    audio_state = ("ready" if elevenlabs_signed_in else
                   "signed out" if elevenlabs_cli else "not installed")
    elevenlabs_setup = ([] if elevenlabs_signed_in else
                        elevenlabs.doctor_setup_lines(cli_available=elevenlabs_cli))
    lines.append(f"audio    ElevenLabs CLI {audio_state}")
    lines.extend(f"note: {line}" for line in elevenlabs_setup)
    from yap_studio.stock.pexels import PexelsProvider

    stock_cli = tools.find_tool("pexels-pp-cli") is not None
    stock_signed_in = stock_cli and PexelsProvider().status() == "ready"
    stock_state = ("ready" if stock_signed_in else "signed out" if stock_cli else "not installed")
    lines.append(f"stock    Pexels CLI {stock_state} (optional, for stock clips)")
    from yap_studio.pictures import wavespeed

    pictures_state = wavespeed.status()
    pictures_cli = pictures_state != "missing_cli"
    pictures_signed_in = pictures_state == "ready"
    pictures_text = ("ready" if pictures_signed_in else
                     "signed out" if pictures_cli else "not installed")
    pictures_setup = ([] if pictures_signed_in else
                      ([wavespeed.INSTALL_LINE] if not pictures_cli else [])
                      + [wavespeed.SIGNUP_LINE, wavespeed.KEY_LINE])
    lines.append(f"pictures WaveSpeed CLI {pictures_text} (optional, for generated pictures)"
                 + ("" if pictures_signed_in else ". " + wavespeed.SIGNUP_LINE))
    photos = None
    if sys.platform == "darwin":
        osxphotos = tools.find_tool("osxphotos") is not None
        photos = {"osxphotos": osxphotos, "optional": True}
        lines.append("photos   osxphotos " + ("ready" if osxphotos else
                     "not installed (optional, for Photos albums: uv tool install osxphotos)"))
    from yap_studio.review import phone

    tailscale_binary = phone._tailscale_binary()
    phone_name = phone.tailscale_name() if tailscale_binary else None
    tailscale_state = ("connected" if phone_name else
                       "installed" if tailscale_binary else "missing")
    # Review tries Tailscale first, then falls back to Wi-Fi if Serve fails.
    phone_via = "tailscale" if phone_name else "wifi" if phone.lan_address() else None
    phone_text = (f"Tailscale ({phone_name})" if phone_name else
                  "Wi-Fi link (Tailscale not connected)" if phone_via and tailscale_binary else
                  "Wi-Fi link (Tailscale not set up)" if phone_via else
                  "no network a phone can reach (connect to Wi-Fi or set up Tailscale)")
    lines.append(f"phone    {phone_text}")
    from yap_studio.product_feedback import availability

    feedback = availability()
    lines.append(f"feedback {feedback['mode']}; endpoint "
                 f"{'available' if feedback['endpoint'] else 'not set'}; gh "
                 f"{'ready' if feedback['gh_logged_in'] else 'unavailable'}")
    # Verify needs only ffmpeg; --edit also fails on the editing dependencies.
    edit_missing = [name for name, ok in (
        ("node", node_ok), ("renderer", renderer is not None), ("browser", browser_ok),
        ("speech", auto is not None),
        ("folder", service is None), ("disk", disk_enough),
    ) if not ok]
    ready_to_edit = not problems and not edit_missing
    lines.append(f"edit     {'ready' if ready_to_edit else 'not ready'}")
    report = {
        "install": {"version": running_version, "build": install_info.build_id(),
                    "text": install_info.version_text()},
        "workspace": None if workspace is None else {
            "version": workspace[1], "matches": workspace_matches},
        "folder": {"path": str(cwd), "synced": service is not None, "service": service},
        "disk": {"free_gb": round(free_gb, 1), "min_gb": setup_checks.MIN_FREE_GB,
                 "enough": disk_enough},
        "warnings": warnings,
        "ffmpeg": {"available": tool_ok["ffmpeg"] and versions["ffmpeg"] is not None,
                   "path": tool_paths["ffmpeg"], "version": versions["ffmpeg"]},
        "ffprobe": {"available": tool_ok["ffprobe"] and versions["ffprobe"] is not None,
                    "path": tool_paths["ffprobe"], "version": versions["ffprobe"]},
        "filters": filters,
        "node": {"available": node_ok, "version": node_version, "path": node},
        "renderer": {"available": renderer is not None, "browser": browser_ok,
                     "version": hyperframes.HYPERFRAMES_VERSION},
        "registry": registry,
        "elevenlabs": {"cli": elevenlabs_cli, "signed_in": elevenlabs_signed_in,
                       "setup": elevenlabs_setup},
        "stock": {"provider": "pexels", "cli": stock_cli,
                  "signed_in": stock_signed_in, "optional": True},
        "pictures": {"provider": wavespeed.PROVIDER, "cli": pictures_cli,
                     "signed_in": pictures_signed_in, "optional": True, "setup": pictures_setup},
        "photos": photos,
        "speech": {"available": auto is not None, "engines": speech, "model": model_ready},
        "cutout": {"available": cutout_reason is None, "reason": cutout_reason},
        "phone": {"tailscale": tailscale_state, "name": phone_name, "via": phone_via},
        "feedback": feedback,
        "problems": [str(problem) for problem in problems],
        "edit_missing": edit_missing,
        "ready_to_edit": ready_to_edit,
    }
    failed = bool(problems) or (args.edit and not ready_to_edit)
    if args.json:
        print(json.dumps(report, indent=2))
        return 2 if failed else 0
    print("\n".join(lines))
    if problems:
        problem = problems[0]
        suggestion = ""
        if isinstance(problem, YapError) and problem.suggested_action:
            suggestion = f" {problem.suggested_action}"
        print(f"problem: {problem}{suggestion}")
        return 2
    if args.edit and not ready_to_edit:
        print(f"problem: not ready to edit; missing {', '.join(edit_missing)}")
        return 2
    print("ok: ready to edit" if args.edit else "ok: ffmpeg is usable")
    return 0


def _renderer_install(_args: argparse.Namespace) -> int:
    """Install the pinned renderer and its browser using the engine cache."""
    from yap_studio.render.hyperframes import ensure_installed

    try:
        result = ensure_installed(notice=print)
    except YapError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    print(f"Renderer {result.version} and browser ready.")
    return _registry_install(_args, install=result)


def _registry_install(
    _args: argparse.Namespace, *, install: RendererInstall | None = None,
) -> int:
    """Install the catalog's pinned registry items into the cache."""
    from yap_studio.hf_registry.install import install_items

    try:
        results = install_items(install=install, notice=print)
    except YapError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    print(f"HyperFrames registry items ready: {len(results)}.")
    return 0


def _registry_list(args: argparse.Namespace) -> int:
    """List registry items and their local cache state."""
    from yap_studio.hf_registry import catalog
    from yap_studio.hf_registry.install import status

    states = status()
    if args.json:
        report = {**states, "commit": catalog.load_catalog()["registry"]["commit"]}
        print(json.dumps(report, indent=2))
    else:
        by_name = {name: state for state, names in states.items() for name in names}
        for name in catalog.items():
            print(f"{name} {by_name[name]}")
    return 0


def _speech_download(_args: argparse.Namespace) -> int:
    """Fetch the selected speech model before the first ingest."""
    backend = resolve_backend()
    if backend is None:
        error = DependencyError.missing_engine("speech engine")
        print(f"error: {error.message}", file=sys.stderr)
        print(f"fix: {error.suggested_action}", file=sys.stderr)
        return 2
    if setup_checks.speech_model_ready(backend):
        print("Speech model ready.")
        return 0
    try:
        backend.prefetch()
    except Exception as exc:
        print(f"error: could not download the speech model: {exc}", file=sys.stderr)
        return 2
    print(f"Downloaded speech model {backend.model} for {backend.name}.")
    return 0


def _review_media_paths(values: Sequence[str]) -> list[Path]:
    """Expand each directory's direct video files while keeping argument order."""
    extensions = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
    media_paths: list[Path] = []
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            try:
                children = sorted(
                    (
                        child for child in candidate.iterdir()
                        if child.is_file() and child.suffix.lower() in extensions
                    ),
                    key=lambda child: child.name,
                )
            except OSError as exc:
                raise InputError(value, f"could not read directory: {exc.strerror}") from exc
            if not children:
                raise InputError(value, "no video files directly inside this directory")
            media_paths.extend(paths.validate_input(child) for child in children)
        else:
            media_paths.append(paths.validate_input(value))
    return media_paths


def _prepare_review(args: argparse.Namespace) -> tuple[ReviewStore, list[ReviewVideo], list[str]]:
    """Validate and load review videos before a server is started."""
    from yap_studio.review.launch import version_video
    from yap_studio.review.media import file_identity, probe_media
    from yap_studio.review.server import ReviewVideo
    from yap_studio.review.store import ReviewStore, data_dir

    allowed_hosts = list(args.allow_host)
    if args.public_base is not None:
        allowed_hosts.append(urlsplit(args.public_base).netloc)
    media_paths = [] if args.versions else _review_media_paths(args.media)
    store = ReviewStore(data_dir())
    videos: list[ReviewVideo] = [version_video(args.media, store, _review_media_paths)] \
        if args.versions else []
    seen: set[str] = set()
    for media_path in media_paths:
        result = store.hash_media(media_path)
        if result.sha256 in seen:
            raise ValidationError(
                "media", f"duplicate video: {media_path.name}; "
                "give each video's contents only once"
            )
        seen.add(result.sha256)
        media = probe_media(media_path)
        if file_identity(media_path) != result.identity:
            raise InputError(
                str(media_path), "the file changed while it was being opened; try again"
            )
        store.comments(result.sha256)
        if result.previous_sha256 is not None:
            store.comments(result.previous_sha256)
        videos.append(
            ReviewVideo(
                media, result.sha256, result.previous_sha256, media_identity=result.identity
            )
        )
    if args.reviewer is not None:
        store.set_reviewer(args.reviewer)
    store.reviewer()
    return store, videos, allowed_hosts


def _review_error(exc: YapError, *, json_output: bool) -> None:
    """Print one review error in the requested format."""
    if json_output:
        print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
    else:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.suggested_action:
            print(f"fix: {exc.suggested_action}", file=sys.stderr)


def _review(args: argparse.Namespace) -> int:
    """Start local review sessions for videos on one server."""
    from yap_studio.review import detach
    from yap_studio.review.server import ReviewServer

    if (args.stop or args.detach) and not detach.posix_supported():
        _review_error(YapError(
            "background reviews need macOS or Linux", code="background_unsupported",
            suggested_action="run review without --detach and keep it open",
        ), json_output=args.json)
        return 2
    if args.stop:
        incompatible = (args.detach or args.phone or args.public_base is not None
                        or args.versions or args.port is not None)
        if incompatible:
            error = ValidationError("stop", "cannot combine --stop with --detach, --phone, "
                                    "--public-base, --versions or --port")
            _review_error(error, json_output=args.json)
            return 2
        return detach.stop(args.media if args.media else None, json_output=args.json)
    if not args.media:
        _review_error(YapError("give at least one video or project folder", code="missing_media"),
                      json_output=args.json)
        return 2
    try:
        store, videos, allowed_hosts = _prepare_review(args)
    except YapError as exc:
        _review_error(exc, json_output=args.json)
        return 2
    if args.detach:
        return detach.start(args, newest_sha=videos[0].sha256 if args.versions else None)
    if args.port is None:
        args.port = 0
    first = videos[0]

    def start(host: str, hosts: list[str]) -> ReviewServer:
        return ReviewServer(
            first.media, first.sha256, first.previous_sha, store, host=host,
            port=args.port, token=first.token, media_identity=first.media_identity,
            allowed_hosts=hosts, additional_videos=videos[1:], series=first.series,
        )

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
    server: ReviewServer | None = None
    cleanup_mapping = None
    record_written = False
    try:
        public_base = args.public_base
        phone_via = None
        try:
            if args.phone:
                from yap_studio.review import phone

                if args.public_base is not None:
                    raise ValidationError("phone", "use --phone or --public-base, not both")
                name = phone.tailscale_name()
                phone_via = "tailscale" if name else "wifi"
                if name:
                    # Tailscale serve forwards the tailnet name to the loopback server.
                    server = start("127.0.0.1", [*allowed_hosts, name])
                    try:
                        public_base, cleanup_mapping = phone.tailscale_origin(server.port)
                    except (OSError, subprocess.SubprocessError, ValueError) as exc:
                        server.server_close()
                        server = None
                        print(f"warning: Tailscale link failed ({exc}); using Wi-Fi instead",
                              file=sys.stderr)
                        phone_via = "wifi"
                if phone_via == "wifi":
                    address = phone.lan_address()
                    if address is None:
                        raise phone.no_network()
                    server = start(address, allowed_hosts)
            else:
                server = start("127.0.0.1", allowed_hosts)
        except OSError as exc:
            _review_error(YapError(
                f"could not listen on port {args.port}: {exc}", code="listen_failed",
            ), json_output=args.json)
            return 2
        except YapError as exc:
            _review_error(exc, json_output=args.json)
            return 2

        def public_url(url: str) -> str:
            if public_base is None:
                return url
            local = urlsplit(url)
            return f"{public_base}{local.path}?{local.query}"

        review_url = public_url(server.url)
        # Behind Tailscale or a public base the server is on loopback, so this Mac can stay on
        # 127.0.0.1, where Yap Studio home keeps its cookie.
        video_links = [
            {"name": video.media.name, "sha256": video.sha256, "url": public_url(url),
             **({"local_url": url} if public_base is not None else {})}
            for video, url in zip(videos, server.video_urls, strict=True)
        ]
        document = {"list": review_url if server.list_url else None, "videos": video_links}
        if phone_via is not None:
            document["phone"] = {"via": phone_via, "url": review_url}
            server.phone = document["phone"]
        if args.json:
            print(json.dumps(document, ensure_ascii=False), flush=True)
        elif phone_via is not None:
            print(f"Open on your phone: {phone.link_label(phone_via)}: {review_url}", flush=True)
            print("Press Ctrl-C to stop.", flush=True)
        else:
            if server.list_url is None:
                print(f"Reviewing {first.media.name} at {review_url}", flush=True)
            else:
                print(f"Review list at {review_url}", flush=True)
                for item in video_links:
                    print(f"{item['name']}  {item['url']}", flush=True)
            print("Press Ctrl-C to stop.", flush=True)

        if args.detached_record is not None:
            detach.write_record(Path(args.detached_record), args.media, args.versions, document)
            record_written = True
        if stopping:
            raise KeyboardInterrupt
        if not args.no_open:
            webbrowser.open(review_url)
        serving = True
        if stopping:
            raise KeyboardInterrupt
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        serving = False
        try:
            if server is not None:
                server.server_close()
        finally:
            try:
                if record_written:
                    record_path = Path(args.detached_record)
                    record_path.unlink(missing_ok=True)
                    record_path.with_suffix(".log").unlink(missing_ok=True)
                if cleanup_mapping is not None:
                    cleanup_mapping()
            finally:
                if previous_sigterm is not None:
                    signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


def _review_allow_host(value: str) -> str:
    """Validate a proxy host before opening the media or review store."""
    from yap_studio.review.server import parse_allowed_host

    try:
        parse_allowed_host(value)
    except ValidationError as exc:
        raise argparse.ArgumentTypeError(exc.detail) from exc
    return value


def _review_public_base(value: str) -> str:
    """Require a plain HTTP origin for the printed review link."""
    detail = "must be an http:// or https:// origin with no path, query or fragment"
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not value.startswith(f"{parsed.scheme}://")
            or parsed.path
            or "?" in value
            or "#" in value
            or any(character.isspace() or ord(character) < 32 for character in value)
        ):
            raise ValueError(detail)
        _review_allow_host(parsed.netloc)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(detail) from exc
    return value


def _review_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer from 0 to 65535") from exc
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be an integer from 0 to 65535")
    return port


def _style(args: argparse.Namespace) -> int:
    """Validate, resolve, or manage styles in the creator's library."""
    try:
        command = args.style_command
        if command == "questions":
            from yap_studio.style.answers import load_answers, questions

            saved = (
                load_answers(args.answers)
                if args.answers and Path(args.answers).exists() else None
            )
            result = questions(saved)
            lines = []
            for index, row in enumerate(result, 1):
                choices = ", ".join(
                    f"{choice['label']}{' (recommended)' if choice['recommended'] else ''}"
                    for choice in row["choices"]
                )
                current = (f" [current: {json.dumps(row['current'], ensure_ascii=False)}]"
                           if row["current"] is not None else "")
                lines.append(f"{index}. {row['prompt']}" + (f" {choices}" if choices else "")
                             + current)
            text = "\n".join(lines)
        elif command == "from-answers":
            from yap_studio.style.answers import apply_answers

            result = apply_answers(args.file, workspace=args.workspace)
            style_text = (
                "Style choices are kept with your answers until you save a look."
                if not result["style_saved"] and result["fields_set"] else
                f"Default style: {result['style_id']}." if result["fields_set"] else
                "Style unchanged."
            )
            brand_fields = [field for field in result["fields_set"]
                            if field.startswith("brand.")]
            brand_text = (f"Brand {'written' if result['style_saved'] else 'choices'}: "
                          f"{', '.join(brand_fields)}. "
                          if brand_fields else "")
            accent_text = ""
            if "brand.accent_color" in brand_fields:
                colors = result["website_colors"]
                accent_text = (f"Accent from {colors['site']}: {colors['used']}. "
                               if colors and colors["used"] else "Accent from answers. ")
            text = (f"{style_text} Fields set: {', '.join(result['fields_set']) or 'none'}. "
                    + brand_text + accent_text
                    + f"Rules approved: {len(result['rules_approved'])}; "
                    f"revoked: {len(result['rules_revoked'])}. "
                    f"Project defaults: "
                    f"{json.dumps(result['project_defaults'], ensure_ascii=False)}."
                    + (" Pending: " + "; ".join(result["pending"].values())
                       if result["pending"] else ""))
        elif command == "validate":
            load_style(args.file)
            result = {"status": "valid"}
            text = "valid"
        elif command in {"show", "resolve"}:
            style = resolve(args.spec)
            if command == "resolve" and args.out is not None:
                output = write_resolved(style, args.out)
                result = {"path": str(output), "sha256": style.sha256()}
                text = f"wrote {output}"
            else:
                print(style.text(), end="")
                return 0
        elif command == "presets":
            result = [
                {"id": preset, "name": resolve(f"preset:{preset}").name}
                for preset in SHIPPED_PRESETS
            ]
            text = "\n".join(f"{entry['id']}: {entry['name']}" for entry in result)
        elif command == "new":
            if args.list:
                result = [
                    {"id": preset, "name": resolve(f"preset:{preset}").name}
                    for preset in SHIPPED_PRESETS
                ]
                text = "\n".join(f"{entry['id']}: {entry['name']}" for entry in result)
            elif args.name is None:
                raise InputError("style.new", "provide a name or use --list")
            else:
                output = create(args.name, preset=args.preset, out=args.out)
                result = {"path": str(output), "name": args.name, "preset": args.preset}
                text = f"wrote {output}"
        elif command == "add":
            entry = add(args.file, make_default=args.default, replace=args.replace)
            result = {**asdict(entry), "path": str(entry.path)}
            text = f"{entry.id}: {entry.name}" + (" (default)" if entry.is_default else "")
        elif command == "save":
            entry = save_pick(
                read_style_file(args.file), name=args.name,
                workspace=Path(args.workspace) if args.workspace is not None else None,
            )
            result = {**asdict(entry), "path": str(entry.path),
                      "workspace_copy": str(entry.workspace_copy) if entry.workspace_copy else None}
            text = f'Saved "{entry.name}" as your style ({entry.id}). New videos use it.'
            if entry.workspace_copy is not None:
                text += f"\nCopy: {entry.workspace_copy}"
        elif command == "list":
            entries = list_entries()
            result = [{**asdict(entry), "path": str(entry.path)} for entry in entries]
            text = "\n".join(
                f"{entry.id}: {entry.name}" + (" (default)" if entry.is_default else "")
                for entry in entries
            ) or "No saved styles."
        else:
            if args.spec is not None:
                set_default(args.spec)
            default = get_default() or "preset:clean_minimal"
            result = {"default_style": default}
            text = default
    except YapError as exc:
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if isinstance(exc, StyleRefusal) or exc.code == "unsafe_path" else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(text)
    return 0


def _captions(args: argparse.Namespace) -> int:
    """Render project captions or caption a standalone vertical clip."""
    from yap_studio.stages.captions import (
        CaptionRefusal,
        _publish_lock,
        back_weak_captions,
        caption_clip,
        caption_error,
        captions,
    )
    from yap_studio.taste.render import render_style, write_taste_record

    try:
        project_mode = (Path(args.clip) / "project.json").is_file()
        if project_mode:
            if (any(value is not None for value in (args.style, args.out, args.words, args.title))
                    or args.srt or args.asr != "auto"):
                raise InputError("captions", "project mode uses project words, timeline and style; "
                                 "pass only DIR, --force, --json and --back")
            result = (back_weak_captions(Path(args.clip), args.back, force=args.force)
                      if args.back is not None else captions(Path(args.clip), force=args.force))
        else:
            if args.back is not None:
                raise InputError("back", "caption backing requires a project directory")
            if args.out is None:
                raise InputError("out", "standalone captions requires --out OUT.mp4")
            application = render_style(args.style)
            result = caption_clip(
                Path(args.clip), style=application.style, out=Path(args.out),
                words=Path(args.words) if args.words is not None else None,
                title=args.title, srt=args.srt, force=args.force, asr=args.asr,
            )
            with _publish_lock(Path(args.out), 120 + 20 * result["frames"] / 30):
                write_taste_record(Path(args.out), application)
            result["taste"] = application.pin()
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = caption_error(exc)
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if isinstance(exc, CaptionRefusal) or exc.code in {
            "unsafe_path", "timeline_refused", "stale_cut", "stage_running", "base_changed",
        } else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    elif project_mode and args.back is not None:
        count = len(result["backed"])
        noun = "phrase" if count == 1 else "phrases"
        if args.back == "none":
            print(f"Turned backing off for {count} {noun}. "
                  f"Run yap-studio run {args.clip} to render the final.")
        else:
            print(f"Backed {count} {noun} with a {args.back}. "
                  f"Run yap-studio run {args.clip} to render the final.")
    elif project_mode:
        print("Captions unchanged." if result["noop"] else
              "No caption phrases; skipped caption layer." if result["layer"] is None else
              f"Wrote {result['out']}: {result['frames']} frames, {result['phrases']} phrases.")
        print(f"SRT: {result['srt']}")
        print(f"Receipt: {result['receipt']}")
    else:
        print(f"Wrote {result['out']}: {result['frames']} frames, "
              f"{result['phrases']} phrases, style {result['style']['name']}.")
        if result["srt"]:
            print(f"SRT: {result['srt']}")
        print(f"Receipt: {result['receipt']}")
        if application.applied or application.skipped:
            print(f"Taste: revision {application.taste_revision}, "
                  f"{len(application.applied)} rules applied")
        else:
            print("Taste: none approved")
        for warning in result.get("warnings", []):
            print(f"warning: {warning}")
    return 0


def _graphics(args: argparse.Namespace) -> int:
    """Render timeline titles in the project's saved style."""
    from yap_studio.stages.graphics import graphics

    try:
        result = graphics(Path(args.directory), force=args.force)
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot read or render graphics: {exc}")
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if exc.code in {
            "unsafe_path", "timeline_refused", "stale_cut", "stage_running", "base_changed",
        } else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print("Graphics unchanged." if result["noop"] else
              f"Rendered {len(result['elements'])} title layers.")
        print(f"Receipt: {result['receipt']}")
    return 0


def _music(args: argparse.Namespace) -> int:
    """Prepare supplied music, approved generated music, or voice-only assembly."""
    from yap_studio.stages.music import MusicError, music

    try:
        if args.other_computer_stopped and not args.confirm_retry:
            raise MusicError("--other-computer-stopped requires --confirm-retry.",
                             code="music_retry_confirmation_required",
                             suggested_action="Add --confirm-retry and try again.")
        if not args.dry_run and (args.file is not None or args.description is not None
                                 or args.tags is not None or args.yes):
            import hashlib

            from yap_studio.project.lock import project_lock, write_json_cas
            from yap_studio.project.model import parse_document

            root = Path(args.directory).resolve()
            source = root / "project.json"
            original = source.read_bytes()
            data = json.loads(original)
            settings = dict(data.get("music", {}))
            if args.file is not None:
                settings.update(mode="file", file=args.file)
            if args.description is not None:
                settings["description"] = args.description
            if args.tags is not None:
                settings["tags"] = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
            if args.yes:
                settings["mode"] = "elevenlabs"
            data["music"] = settings
            parse_document("project", data)
            with paths.operation(), project_lock(root):
                write_json_cas(source, data, base_sha256=hashlib.sha256(original).hexdigest())
        result = music(Path(args.directory), force=args.force, new=args.new,
                       confirm_retry=args.confirm_retry,
                       other_computer_stopped=args.other_computer_stopped,
                       dry_run=args.dry_run,
                       preview_mode="elevenlabs" if args.yes and args.dry_run else None,
                       preview_description=args.description,
                       preview_tags=([tag.strip() for tag in args.tags.split(",")
                                      if tag.strip()] if args.tags is not None else None))
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot prepare music: {exc}")
        notices = getattr(exc, "notices", [])
        if args.json:
            print(json.dumps({"error": exc.to_dict(), "notices": notices}), file=sys.stderr)
        else:
            for line in notices:
                print(line, file=sys.stderr)
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if exc.code in {"unsafe_path", "timeline_refused", "stale_cut",
                                "stage_running", "base_changed"} else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        for line in result.get("notices", []):
            print(line, file=sys.stderr)
        if result["status"] == "dry_run":
            print(json.dumps(result["requests"], indent=2, ensure_ascii=False))
            return 0
        print("Music unchanged." if result["noop"] else
              "Music bed prepared." if result["bed"] else "Music: none; use voice only.")
        print(f"Receipt: {result['receipt']}")
    return 0


def _frame_range(value: str) -> tuple[int, int]:
    """Parse an inclusive non-negative output frame window."""
    match = re.fullmatch(r"([0-9]+)-([0-9]+)", value)
    if match is None or int(match[1]) > int(match[2]):
        raise argparse.ArgumentTypeError("expected A-B with 0 <= A <= B")
    return int(match[1]), int(match[2])


def _assemble(args: argparse.Namespace) -> int:
    """Publish and verify the project's final picture and audio."""
    from yap_studio.stages.assemble import assemble

    try:
        result = assemble(Path(args.directory), force=args.force,
                          allow_violations=args.allow_violations, frames=args.frames)
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot assemble final: {exc}")
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        if exc.code == "assembly_checks_failed":
            return 1
        return 3 if exc.code in {"pilot_range_refused", "unsafe_path", "timeline_refused",
                                "stale_cut", "stage_running", "base_changed"} else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        if result.get("refreshed"):
            print(f"Re-ran stale stages: {', '.join(result['refreshed'])}.")
        for line in result.get("notices", []):
            print(line, file=sys.stderr)
        if result.get("pilot"):
            first, last = result["frames"]
            print("Pilot unchanged." if result["noop"] else
                  f"Verified pilot (frames {first}-{last}, not a final): {result['out']}")
        else:
            print("Final unchanged." if result["noop"] else f"Verified final: {result['out']}")
        print(f"Receipt: {result['receipt']}")
    return 0


def _project(args: argparse.Namespace) -> int:
    """Initialize a project or report the agent's next action."""
    from yap_studio.project import init_project
    from yap_studio.project.status import status

    try:
        if args.command == "init":
            project = init_project(Path(args.directory), takes=[Path(take) for take in args.take],
                                   voiceovers=[Path(memo) for memo in args.voiceover],
                                   style=args.style, name=args.name,
                                   shape=args.shape,
                                   video_type=args.video_type, broll_sources=args.broll,
                                   answers=Path(args.answers) if args.answers else None)
            result = project.to_dict()
            text = f"Initialized {project.name} in {args.directory}"
        else:
            from yap_studio.stages import run_detach

            if args.wait is not None:
                run_detach.wait(Path(args.directory).resolve(), args.wait)
            result = status(Path(args.directory))
            from yap_studio.video_type import describe

            lines = ["video: " + describe(result["video"]["type"],
                                         result["video"]["broll_sources"])]
            lines.extend(run_detach.status_lines(result["run"]) if result["run"] else [])
            lines.extend(f"{stage['name']}: {stage['state']}, {stage['reason']}"
                         for stage in result["stages"])
            lines.extend(f"needs: {item}" for item in result["needs"])
            lines.extend(f"needs human: {item}" for item in result["needs_human"])
            lines.extend(f"warning: {item}" for item in result["warnings"])
            if any(stage["name"] == "ingest" and stage["state"] != "done"
                   for stage in result["stages"]):
                lines.extend(f"transcript: {item['take']} cached by {item['engine']}; "
                             "ingest reuses it"
                             for item in result.get("transcripts", []))
            if "cutout" in result:
                lines.append(f"cutout: {result['cutout']}")
            lines.append(f"next: {result['next'] or 'none'}")
            text = "\n".join(lines)
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot read or write project: {exc}")
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if exc.code in {"project_exists", "base_changed", "unsafe_path"} else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(text)
    return 0


def _word_fixes(args: argparse.Namespace) -> int:
    """List or atomically change transcript display corrections."""
    from yap_studio import receipt
    from yap_studio.project.lock import project_lock, write_json_cas
    from yap_studio.project.model import load_project, parse_document
    from yap_studio.project.word_fixes import change_word_fix

    try:
        selected = args.word is not None or args.heard is not None
        if args.list:
            if selected or args.to is not None or args.remove:
                raise ValidationError("words fix",
                                      "use --list without a selector, --to or --remove")
        elif ((args.word is not None) == (args.heard is not None)
              or args.remove == (args.to is not None)):
            raise ValidationError("words fix", "choose --word ID or --heard TEXT, "
                                  "then --to TEXT or --remove")
        if any(value is not None and not value.strip() for value in (args.heard, args.to)):
            raise ValidationError("words fix", "heard and replacement text must not be blank")
        with paths.operation():
            root = Path(args.directory).expanduser().resolve()
            target = root / "project.json"
            project = load_project(target)
            if not args.list:
                with project_lock(root):
                    digest = receipt.sha256_file(target)
                    data = json.loads(target.read_bytes())
                    _, key = change_word_fix(root, data, word=args.word, heard=args.heard,
                                             to=args.to, remove=args.remove)
                    project = parse_document("project", data)
                    write_json_cas(target, data, base_sha256=digest)
            result = {"word_text_fixes": project.word_text_fixes, "word_fixes": project.word_fixes}
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot read or write word fixes: {exc}")
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
        return 3 if exc.code in {"base_changed", "unsafe_path", "project_locked"} else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    elif args.remove:
        print(f"Removed fix: {key}")
    else:
        for field, label in (("word_text_fixes", "Word"), ("word_fixes", "Heard")):
            for key, value in result[field].items():
                print(f"{label}: {key} -> {value}")
        if not any(result.values()):
            print("No word fixes.")
    return 0


def _ingest_words(args: argparse.Namespace) -> int:
    """Analyze takes, check a recording, or print compact word ids."""
    from yap_studio.stages.ingest import IngestRefusal, ingest, preflight, words_listing

    try:
        if args.command == "words":
            result = words_listing(Path(args.directory), cut=args.cut)
            text = "\n".join(result["lines"])
        elif args.preflight:
            if args.force or args.copy or args.directory != ".":
                raise InputError("preflight",
                                 "use --preflight CLIP without DIR, --force, or --copy")
            result = preflight(Path(args.preflight), asr=args.asr or "auto")
            lines = [result["status"]]
            lines.extend(f"failed: {check}" for check in result["failed_checks"])
            lines.extend(result["notes"])
            text = "\n".join(lines)
        else:
            result = ingest(Path(args.directory), force=args.force, copy=args.copy, asr=args.asr)
            untranscribed = any(
                take["speech_check"]["untranscribed"] for take in result["takes"]
            )
            lines = [
                "Ingest finished: some speech has no words." if untranscribed
                else "Ingest unchanged." if result["noop"] else "Ingest verified."
            ]
            lines.extend(f"warning: {warning}" for warning in result["warnings"])
            lines.extend(f"orphan: {item['file']} {item.get('segment', item.get('element'))} "
                         f"{item['anchor']} {item['word']} {item['old_word']}"
                         for item in result["orphans"])
            if result["orphans"]:
                lines.append("next: yap-studio edit rebase DIR")
            text = "\n".join(lines)
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot read or write ingest data: {exc}")
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        refused = isinstance(exc, IngestRefusal) or exc.code in {"base_changed", "unsafe_path"}
        return 3 if refused else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(text)
    return 1 if result.get("status") == "FAIL" else 0


def _cut_edit(args: argparse.Namespace) -> int:
    """Render the story or print the same resolution without writing."""
    from yap_studio.stages.cut import cut, edit_check

    try:
        if args.command == "edit":
            result = edit_check(Path(args.directory))
            lines = [f"{piece['segment_id']} {piece['take']} "
                     f"{piece['src_t0']} to {piece['src_t1']}: {piece['n_frames']} frames"
                     for piece in result["pieces"]]
            lines.extend(f"  note: {note}" for piece in result["pieces"] for note in piece["notes"])
            lines.extend(f"{segment['id']}: {segment['text']} ({', '.join(segment['words'])})"
                         for segment in result["segments"])
            lines.append(f"Total: {result['total_frames']} frames, {result['total_samples']} "
                         f"samples, {result['duration_s']:.3f} s")
            lines.append(f"End hold: {result['end_hold_s']:g} s from "
                         f"{result['end_hold_source']}, applied at assembly")
            lines.extend(f"listen: frame {join['frame']}, {join['left']} to {join['right']}"
                         for join in result["listen_joins"])
        else:
            result = cut(Path(args.directory), force=args.force)
            lines = ["Cut unchanged." if result["noop"] else f"Cut {result['status']}."]
            if levels := result.get("take_levels"):
                if not levels["match_takes"]:
                    lines.append("Take levels kept as recorded (match_takes is false).")
                elif any(take["gain_db"] != 0 for take in levels["takes"].values()):
                    gains = ", ".join(
                        f"{take} {row['gain_db']:+.1f} dB"
                        for take, row in levels["takes"].items()
                        if take != levels["reference"] or levels["headroom_db"] != 0
                    )
                    action = ("adjusted toward" if any(take.get("limited")
                              for take in levels["takes"].values()) else "matched to")
                    lines.append(f"Take levels {action} {levels['reference']}: {gains}")
            lines.extend(f"failed: {line}" for line in failed_assert_lines(
                result.get("asserts", [])))
            lines.append(result["out"])
        lines.extend(f"warning: {warning}" for warning in result["warnings"])
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot read or render the cut: {exc}")
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print("\n".join(lines))
    return 1 if result.get("status") == "failed" else 0


def _edit_schema(args: argparse.Namespace) -> int:
    """Print the edit contract without opening a project."""
    from yap_studio.project.edit_format import edit_schema_text
    from yap_studio.project.model import json_schema

    if args.json:
        print(json.dumps(json_schema("edit"), indent=2))
    else:
        print(edit_schema_text())
    return 0


def _edit_rebase(args: argparse.Namespace) -> int:
    """Report and apply source-time word moves across authored files."""
    from yap_studio.stages.rebase import rebase_edit

    try:
        result = rebase_edit(Path(args.directory), dry_run=args.dry_run,
                             pin_unmatched=args.pin_unmatched)
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot rebase edit: {exc}")
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
        return 3 if exc.code in {"base_changed", "project_locked"} else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    elif result["status"] == "cut_failed":
        print("error: the cut does not resolve after rebase: "
              f"{result['resolve_error']}; restored {', '.join(result['restored'])}")
    else:
        if result["status"] == "unchanged":
            print("Edit already matches the current words.")
        for name, counts in result["files"].items():
            if name in result["changed"] or any(counts[key] for key in (
                    "reanchored", "pinned", "unmatched")):
                print(f"{name}: {counts['reanchored']} moved, {counts['unchanged']} unchanged, "
                      f"{counts['pinned']} pinned, {counts['unmatched']} unmatched")
                if counts["moved"]:
                    print("  " + ", ".join(counts["moved"]))
        for item in result["missing_snapshots"]:
            print(f"missing snapshot: {item['file']} {item['take']} {item['hash']}")
        for item in (*result["unmatched"], *result["pinned"]):
            owner = item.get("segment", item.get("element", item.get("removal", "")))
            action = "pinned" if item in result["pinned"] else "unmatched"
            if "reason" in item and "word" not in item:
                print(f"{action}: {item['file']} {owner} {item['anchor']} "
                      f"(take {item['take']}); {item['reason']}")
                continue
            nearest = ", ".join(f"{word['id']} {word['text']!r} {word['start']:g}s"
                                 for word in item["nearest"])
            print(f"{action}: {item['file']} {owner} {item['anchor']} "
                  f"{item['word']} {item['text']!r} {item['time']}s; nearest: {nearest}"
                  + (f"; now {item['new_word']} {item['new_text']!r}"
                     if item.get("new_word") else "")
                  + (f"; {item['note']}" if item.get("note") else ""))
        for item in result["kept"]:
            edge = "start" if item["edge"] == "from" else "end"
            frames = f"{item['frames']:+d}" if item["frames"] is not None else "unknown"
            print(f"kept: {item['segment']} {edge} at {item['time']:.3f} s "
                  f"(new words would move it {frames} frames)")
        for item in result["shifted"]:
            if item.get("estimated"):
                print(f"estimated shift: {item['segment']} {item['anchor']} "
                      f"{item['old_s']:g}s -> {item['new_s']:g}s")
                continue
            edges = []
            for label in ("start", "end"):
                move = item[f"{label}_frames"]
                old_s, new_s = item[f"{label}_old_s"], item[f"{label}_new_s"]
                if move:
                    edges.append(f"{label} {old_s:.3f} s -> {new_s:.3f} s "
                                 f"({move:+d} frames)")
                elif old_s is None or new_s is None:
                    edges.append(f"{label} {'added' if old_s is None else 'removed'}")
            detail = " ".join(edges) or "source pieces changed"
            print(f"moved: {item['segment']} {detail}; "
                  f"{item['frames_old']} -> {item['frames_new']} output frames, "
                  f"{item['pieces_old']} -> {item['pieces_new']} pieces")
        if result["cut_change"] is not None:
            change = result["cut_change"]
            print(f"cut: {change['frames_old']} -> {change['frames_new']} frames")
        elif result["edges_compared"] not in {"no changes", "rebase not written"}:
            print(f"cut comparison: {result['edges_compared']}")
        if result["status"] == "dry_run":
            print("Dry run; no files written. Edges are checked on a real run.")
        print(result["next"])
    return 1 if result["status"] in {"unmatched", "moved", "cut_failed"} else 0


def _fillers(args: argparse.Namespace) -> int:
    """Test joins or change a single removal decision under the project lock."""
    from yap_studio.stages.fillers import (
        _check_removals_receipt,
        add_manual,
        fillers,
        set_removal,
    )

    try:
        if args.fillers_command == "set":
            result = set_removal(Path(args.directory), args.identifier, args.removal_status,
                                 reason=args.reason)
            lines = [f"{result['id']}: {args.removal_status}"]
        elif args.fillers_command == "add":
            take, start, end = args.manual
            result = add_manual(Path(args.directory), take, float(start), float(end),
                                reason=args.reason)
            lines = [f"{result['id']}: manual applied"]
        else:
            expected_max_pause_s = None
            if args.max_pause is not None:
                import hashlib
                import math

                from yap_studio.project.lock import pid_alive, project_lock, write_json_cas
                from yap_studio.project.model import parse_document

                try:
                    max_pause = float(args.max_pause)
                except ValueError as exc:
                    raise ValidationError(
                        "--max-pause", "must be a finite nonnegative number of seconds",
                    ) from exc
                if not math.isfinite(max_pause) or max_pause < 0:
                    raise ValidationError(
                        "--max-pause", "must be a finite nonnegative number of seconds",
                    )
                root = Path(args.directory).resolve()
                with paths.operation(), project_lock(root):
                    folder = root / "fillers"
                    progress = folder / "progress.json"
                    if (progress.exists()
                            and pid_alive(json.loads(progress.read_bytes()).get("pid"))):
                        raise YapError("fillers is already running.", code="stage_running",
                                       suggested_action="Wait for the stage and retry.")
                    if (folder / "receipt.pending.json").exists():
                        raise InputError("fillers", "unfinished stage journal; rerun fillers first")
                    _check_removals_receipt(root)
                    removals = root / "removals.json"
                    if removals.exists():
                        parse_document("removals", json.loads(removals.read_bytes()))
                    source = root / "project.json"
                    original = source.read_bytes()
                    data = json.loads(original)
                    project = parse_document("project", data)
                    if project.rules["max_pause_s"] != max_pause:
                        data["rules"] = {**data.get("rules", {}), "max_pause_s": max_pause}
                        parse_document("project", data)
                        write_json_cas(source, data,
                                       base_sha256=hashlib.sha256(original).hexdigest())
                expected_max_pause_s = max_pause
            result = fillers(Path(args.directory), apply=args.apply, force=args.force,
                             expected_max_pause_s=expected_max_pause_s)
            lines = ["Fillers unchanged."] if result["noop"] else []
            for item in result["entries"]:
                if item["status"] in {"applied", "rejected"}:
                    words = " ".join(item.get("evidence", {}).get(
                        "removed_words", item.get("words", []),
                    )) or ("manual span" if item["kind"] == "manual" else "pause")
                    reason = item.get("reason") or item.get("evidence", {}).get("words", "pass")
                    lines.append(f"{item['id']} {item['kind']} {item['status']} {words}: {reason}")
                if item.get("listen"):
                    lines.append(f"listen: {item['id']} {item['take']} "
                                 f"{item['start']:.3f} to {item['end']:.3f}")
            lines.extend(
                f"{note} ({result['skipped']})"
                if note == "fillers skipped: no speech engine" else note
                for note in result["notes"]
            )
            counts = result["counts"]["status"]
            lines.append(f"Fillers: {counts.get('applied', 0)} applied, "
                         f"{counts.get('proposed', 0)} proposed, "
                         f"{counts.get('rejected', 0)} rejected, {result['skipped']} skipped.")
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot read or write fillers data: {exc}")
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if exc.code in {"base_changed", "unsafe_path"} else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print("\n".join(lines))
    return 0


def _timeline(args: argparse.Namespace) -> int:
    """Resolve timeline lanes or transact one writer's fragment."""
    from yap_studio.stages.timeline import (
        TimelineRefusal,
        add_timeline,
        remove_timeline,
        resolve_timeline,
    )

    try:
        if args.timeline_command == "remove":
            result = remove_timeline(Path(args.directory), lane=args.lane,
                                     ids=tuple(args.id or ()))
            valid = True
            lines = [f"Removed {len(result['removed'])} elements from lane {args.lane}."]
        else:
            timeline = (add_timeline(Path(args.directory), Path(args.fragment), lane=args.lane,
                                     replace_lane=args.replace_lane)
                        if args.timeline_command == "add"
                        else resolve_timeline(Path(args.directory)))
            result = timeline.to_dict()
            valid = timeline.valid
            lines = [f"Timeline: {len(timeline.elements)} elements; "
                     f"{'passed' if valid else 'refused'}."]
            if timeline.steps is not None and args.timeline_command in {"check", "show"}:
                summary = timeline.steps["summary"]
                lines.append(f"Step in: {summary['changed']} of {summary['jumps']} "
                             "jump cuts change the framing.")
            if args.timeline_command == "show":
                lines.extend(f"{item.id} ({item.lane}, {item.kind}): "
                             f"frames {item.start_frame} to {item.end_frame}, "
                             f"{item.start_s:g} to {item.end_s:g} s" for item in timeline.elements)
            groups = {}
            for index, item in enumerate(timeline.findings):
                bare = item.bare_message or item.message.partition(": ")[2]
                separate = index if len(item.elements) != 1 else None
                groups.setdefault((item.rule, item.severity, bare, separate), []).append(item)
            for (_, severity, bare, _), grouped in groups.items():
                if len(grouped) == 1:
                    item = grouped[0]
                    lines.append(f"{severity}: {item.message}\nfix: {item.suggested_action}")
                    continue
                ids = list(dict.fromkeys(name for item in grouped for name in item.elements))
                count = len(ids)
                shown = ", ".join(ids[:3])
                if len(ids) > 3:
                    shown += f" and {len(ids) - 3} more"
                action = grouped[0].suggested_action
                if any(item.suggested_action != action for item in grouped[1:]):
                    action += f"; the same for the other {count - 1} elements"
                lines.append(f"{severity}: {bare} ({count} elements: {shown})\n"
                             f"fix: {action}")
            lines.extend(f"snapped: {item['element']} {item['anchor']}: "
                         f"{item['from_word']} to {item['word']} in {item['segment']}"
                         for item in timeline.snapped_anchors)
            lines.extend(f"beat: {item['element']} entrance frame {item['from_frame']} "
                         f"to {item['to_frame']}" for item in timeline.beat_snaps)
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot read or write timeline: {exc}")
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if isinstance(exc, TimelineRefusal) or exc.code in {
            "base_changed", "unsafe_path", "project_locked",
        } else 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print("\n".join(lines))
    return 0 if valid else 3


def _run(args: argparse.Namespace) -> int:
    """Run every stage in order and stop at the first failed check with its fix."""
    from yap_studio import background
    from yap_studio.stages import run_detach
    from yap_studio.stages.run import run

    try:
        if args.stop and (args.detach or args.note is not None or args.frames is not None
                          or args.allow_violations):
            raise ValidationError("stop", "cannot combine --stop with --detach, --note, "
                                  "--frames or --allow-violations")
        if (args.detach or args.stop or args.detached_run) and not background.posix_supported():
            raise YapError("background runs need macOS or Linux", code="background_unsupported",
                           suggested_action="run without --detach and keep it open")
        if args.stop:
            return run_detach.stop(Path(args.directory), json_output=args.json)
        if args.detach:
            return run_detach.start(args)
        if args.detached_run:
            return run_detach.child(args)
        with run_detach.run_lock(Path(args.directory)):
            run_detach.clear_stale(Path(args.directory).resolve())
            result = run(Path(args.directory), allow_violations=args.allow_violations,
                         note=args.note, frames=args.frames,
                         on_step=None if args.json else run_detach.show_step)
    except (YapError, OSError, ValueError) as exc:
        if not isinstance(exc, YapError):
            exc = InputError(args.directory, f"cannot run the project: {exc}")
        if args.json:
            print(json.dumps({"error": exc.to_dict()}), file=sys.stderr)
        else:
            print(f"error: {exc.message}", file=sys.stderr)
            if exc.suggested_action:
                print(f"fix: {exc.suggested_action}", file=sys.stderr)
        return 3 if exc.code == "run_running" else 2
    stopped = result["stopped"]
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return stopped["exit"] if stopped else 0
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    for notice in result["notices"]:
        print(notice, file=sys.stderr)
    if stopped:
        print(f"stopped at {stopped['step']}: {stopped['message']}", file=sys.stderr)
        if stopped.get("suggested_action"):
            print(f"fix: {stopped['suggested_action']}", file=sys.stderr)
        print(f"next: {result['next']} (finished stages are skipped)")
        return stopped["exit"]
    if result.get("pilot"):
        first, last = args.frames
        print(f"Pilot: {result['pilot']} (frames {first}-{last}). "
              "Not a final and not saved as a version.")
        print(f"next: {result['next']}")
        return 0
    print(f"Finished: {result['final']}, saved as {result['version']}.")
    print(f"next: {result['next']}")
    return 0


def _wait_seconds(value: str) -> int:
    seconds = int(value)
    if not 1 <= seconds <= 3600:
        raise argparse.ArgumentTypeError("wait must be from 1 to 3600 seconds")
    return seconds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yap-studio")
    parser.add_argument(
        "--version", action=_VersionAction, nargs=0, dest=argparse.SUPPRESS,
        help="show program's version number and exit",
    )
    subparsers = parser.add_subparsers(dest="command")
    init_parser = subparsers.add_parser("init", help="initialize a project from raw takes")
    init_parser.add_argument("directory", metavar="DIR")
    init_parser.add_argument("--take", action="append", default=[], metavar="FILE")
    init_parser.add_argument("--voiceover", action="append", default=[], metavar="MEMO.m4a")
    from yap_studio.video_type import BROLL_SOURCES, VIDEO_TYPES

    init_parser.add_argument("--video-type", choices=VIDEO_TYPES)
    init_parser.add_argument("--shape", choices=("vertical", "horizontal"), default="vertical",
                             help="delivery shape; horizontal supports talking head and captions")
    init_parser.add_argument("--broll", action="append", choices=(*BROLL_SOURCES, "none"))
    init_parser.add_argument("--style", metavar="SPEC")
    init_parser.add_argument("--name", metavar="NAME")
    init_parser.add_argument("--answers", metavar="FILE")
    init_parser.add_argument("--json", action="store_true")
    init_parser.set_defaults(func=_project)
    status_parser = subparsers.add_parser("status", help="show stage states and the next action")
    status_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
    status_parser.add_argument("--json", action="store_true")
    status_parser.add_argument("--wait", type=_wait_seconds, metavar="SECONDS")
    status_parser.set_defaults(func=_project)
    ingest_parser = subparsers.add_parser("ingest", help="analyze takes and produce source words")
    ingest_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
    ingest_parser.add_argument("--force", action="store_true")
    ingest_parser.add_argument("--copy", action="store_true")
    ingest_parser.add_argument("--asr", choices=_ASR_CHOICES[:-1], metavar="ENGINE")
    ingest_parser.add_argument("--preflight", metavar="CLIP")
    ingest_parser.add_argument("--json", action="store_true")
    ingest_parser.set_defaults(func=_ingest_words)
    words_parser = subparsers.add_parser(
        "words", help="print compact transcripts with word ids",
        epilog="Corrections: words fix DIR --word ID --to TEXT; words fix DIR --list.",
    )
    words_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
    words_parser.add_argument("--cut", action="store_true")
    words_parser.add_argument("--json", action="store_true")
    words_parser.set_defaults(func=_ingest_words)
    fillers_parser = subparsers.add_parser(
        "fillers", help="test joins and propose removals; --apply applies passing cuts",
        epilog="Decisions: fillers set DIR ID STATUS; fillers add DIR --manual TAKE START END.",
    )
    fillers_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
    fillers_parser.add_argument("--apply", action="store_true",
                                help="apply verified cuts; a join that could not be verified "
                                     "stays proposed")
    fillers_parser.add_argument("--force", action="store_true")
    fillers_parser.add_argument("--max-pause", metavar="S",
                                help="save the maximum pause in seconds to project rules")
    fillers_parser.add_argument("--json", action="store_true")
    fillers_parser.set_defaults(func=_fillers, fillers_command=None)
    timeline_parser = subparsers.add_parser("timeline", help="manage word-anchored timeline lanes")
    timeline_commands = timeline_parser.add_subparsers(dest="timeline_command", required=True)
    for command in ("add", "remove", "show", "check"):
        command_parser = timeline_commands.add_parser(command)
        if command == "add":
            command_parser.add_argument("fragment", metavar="FRAGMENT")
            command_parser.add_argument("--replace-lane", action="store_true")
        command_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
        command_parser.add_argument("--json", action="store_true")
        command_parser.set_defaults(func=_timeline)
        if command in {"add", "remove"}:
            command_parser.add_argument("--lane", required=True, choices=("agent", "broll"))
        if command == "remove":
            command_parser.add_argument("--id", action="append", metavar="ID")
    cut_parser = subparsers.add_parser("cut", help="render and verify the ordered story")
    cut_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
    cut_parser.add_argument("--force", action="store_true")
    cut_parser.add_argument("--json", action="store_true")
    cut_parser.set_defaults(func=_cut_edit)
    music_parser = subparsers.add_parser("music", help="prepare a music bed or voice only")
    music_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
    music_parser.add_argument("--force", action="store_true")
    music_parser.add_argument("--file", metavar="TRACK", help="save and prepare a supplied track")
    music_parser.add_argument("--description", metavar="TEXT")
    music_parser.add_argument("--tags", metavar="A,B")
    music_parser.add_argument("--yes", action="store_true")
    music_parser.add_argument("--new", action="store_true")
    music_parser.add_argument("--confirm-retry", action="store_true")
    music_parser.add_argument("--other-computer-stopped", action="store_true")
    music_parser.add_argument("--dry-run", action="store_true")
    music_parser.add_argument("--json", action="store_true")
    music_parser.set_defaults(func=_music)
    assemble_parser = subparsers.add_parser("assemble", help="render and verify the final video")
    assemble_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
    assemble_parser.add_argument("--json", action="store_true")
    assemble_parser.add_argument("--force", action="store_true")
    assemble_parser.add_argument("--allow-violations", action="store_true")
    assemble_parser.add_argument("--frames", type=_frame_range, metavar="A-B",
                                 help="render only frames A-B (inclusive, 30 fps output frames) "
                                      "as a pilot; never a final or a saved version")
    assemble_parser.set_defaults(func=_assemble)
    run_parser = subparsers.add_parser(
        "run", help="run every stage, stop at the first failed check, resume on rerun",
        epilog="Stages: ingest, fillers, cut, music, captions, graphics, assemble, verify, "
               "words, version. Finished stages are skipped, so a rerun resumes where it stopped.",
    )
    run_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
    run_parser.add_argument("--note", metavar="TEXT", help="note for the saved version")
    run_parser.add_argument("--allow-violations", action="store_true",
                            help="publish a final whose only failures are layout overlaps")
    run_parser.add_argument("--json", action="store_true")
    run_parser.add_argument("--detach", action="store_true", help="run in the background")
    run_parser.add_argument("--stop", action="store_true",
                            help="stop this project's background run")
    run_parser.add_argument("--detached-run", help=argparse.SUPPRESS)
    run_parser.add_argument("--frames", type=_frame_range, metavar="A-B",
                            help="render only frames A-B (inclusive, 30 fps output frames) "
                                 "as a pilot; never a final or a saved version")
    run_parser.set_defaults(func=_run)
    edit_parser = subparsers.add_parser("edit", help="describe and inspect the story (edit.json)")
    edit_commands = edit_parser.add_subparsers(dest="edit_command", required=True)
    check_parser = edit_commands.add_parser("check", help="resolve pieces without rendering")
    check_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
    check_parser.add_argument("--json", action="store_true")
    check_parser.set_defaults(func=_cut_edit)
    schema_parser = edit_commands.add_parser(
        "schema", help="print the edit.json format and an example"
    )
    schema_parser.add_argument("--json", action="store_true")
    schema_parser.set_defaults(func=_edit_schema)
    rebase_parser = edit_commands.add_parser("rebase", help="move edits onto current source words")
    rebase_parser.add_argument("directory", nargs="?", default=".", metavar="DIR")
    rebase_parser.add_argument("--dry-run", action="store_true")
    rebase_parser.add_argument("--pin-unmatched", action="store_true")
    rebase_parser.add_argument("--json", action="store_true")
    rebase_parser.set_defaults(func=_edit_rebase)
    doctor = subparsers.add_parser("doctor", help="check runtime dependencies")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument(
        "--edit", action="store_true",
        help="also fail unless Node, the renderer and its browser, and speech are ready",
    )
    doctor.set_defaults(func=_doctor)
    renderer_parser = subparsers.add_parser("renderer", help="prepare the pinned renderer")
    renderer_parser.add_argument("install", choices=("install",))
    renderer_parser.set_defaults(func=_renderer_install)
    registry_parser = subparsers.add_parser("registry", help="prepare pinned registry items")
    registry_commands = registry_parser.add_subparsers(dest="registry_command", required=True)
    registry_install = registry_commands.add_parser("install", help="install items into the cache")
    registry_install.set_defaults(func=_registry_install)
    registry_list = registry_commands.add_parser("list", help="list items and cache state")
    registry_list.add_argument("--json", action="store_true")
    registry_list.set_defaults(func=_registry_list)
    speech_parser = subparsers.add_parser("speech", help="prepare the speech model")
    speech_parser.add_argument("download", choices=("download",))
    speech_parser.set_defaults(func=_speech_download)
    mcp_parser = subparsers.add_parser("mcp", help="run the MCP stdio server")
    mcp_parser.set_defaults(func=_mcp)
    verify_parser = subparsers.add_parser("verify", help="verify a finished video render")
    verify_parser.add_argument("media", metavar="MEDIA")
    verify_parser.add_argument("--edl", metavar="FILE")
    verify_parser.add_argument("--source", action="append", metavar="PATH_OR_ID=PATH")
    verify_parser.add_argument("--asr", choices=_ASR_CHOICES, default="auto")
    verify_parser.add_argument("--asr-model", metavar="NAME")
    verify_parser.add_argument("--loudness-target", type=float, default=-14.0, metavar="LUFS")
    verify_parser.add_argument("--loudness", choices=("check", "note"), default="check")
    verify_parser.add_argument("--platform", choices=_PLATFORM_CHOICES, default="none")
    verify_parser.add_argument("--strict", action="store_true")
    verify_parser.add_argument("--json", action="store_true")
    verify_parser.add_argument("--report", metavar="PATH")
    verify_parser.add_argument("--artifacts", metavar="DIR")
    verify_parser.add_argument("--detail", choices=("summary", "full"), default="summary")
    verify_parser.set_defaults(func=_verify)
    review_parser = subparsers.add_parser("review", help="review a render and leave timed comments")
    review_parser.add_argument("media", nargs="*", metavar="MEDIA")
    review_parser.add_argument(
        "--versions", action="store_true",
        help="treat MEDIA as versions of one video, oldest first, or one project folder "
        "with saved versions",
    )
    review_parser.add_argument("--port", type=_review_port, metavar="N")
    review_parser.add_argument("--no-open", action="store_true")
    review_parser.add_argument("--reviewer", metavar="NAME")
    review_parser.add_argument(
        "--json", action="store_true", help="print review links as one JSON line"
    )
    review_parser.add_argument(
        "--allow-host", type=_review_allow_host, action="append", default=[], metavar="NAME[:PORT]",
        help="allow a Host header forwarded by a trusted local proxy (repeatable)",
    )
    review_parser.add_argument(
        "--public-base", type=_review_public_base, metavar="URL",
        help="print and open the review link on this HTTP or HTTPS origin; also allow its host",
    )
    review_parser.add_argument(
        "--phone", action="store_true",
        help="print one link for your phone: Tailscale when it is connected, otherwise "
        "this computer's Wi-Fi address with a private token",
    )
    review_parser.add_argument("--detach", action="store_true",
                               help="keep serving after this command returns")
    review_parser.add_argument("--stop", action="store_true",
                               help="stop matching background reviews, or all without MEDIA")
    review_parser.add_argument("--detached-record", help=argparse.SUPPRESS)
    review_parser.set_defaults(func=_review)
    from yap_studio.feedback.cli import register as _register_loop_commands
    _register_loop_commands(subparsers)
    captions_parser = subparsers.add_parser("captions", help="render project or clip captions")
    captions_parser.add_argument("clip", metavar="DIR|CLIP", nargs="?", default=".")
    captions_parser.add_argument("--style", metavar="S")
    captions_parser.add_argument("--out", metavar="OUT.mp4")
    captions_parser.add_argument("--words", metavar="W")
    captions_parser.add_argument("--title", metavar="a|b")
    captions_parser.add_argument("--srt", action="store_true")
    captions_parser.add_argument("--force", action="store_true")
    captions_parser.add_argument("--back", choices=("shadow", "box", "none"),
                                 help="back only phrases flagged by the final contrast check")
    captions_parser.add_argument("--asr", choices=_ASR_CHOICES[:-1], default="auto")
    captions_parser.add_argument("--json", action="store_true")
    captions_parser.set_defaults(func=_captions)

    graphics_parser = subparsers.add_parser("graphics", help="render project title layers")
    graphics_parser.add_argument("directory", metavar="DIR", nargs="?", default=".")
    graphics_parser.add_argument("--force", action="store_true")
    graphics_parser.add_argument("--json", action="store_true")
    graphics_parser.set_defaults(func=_graphics)
    style_parser = subparsers.add_parser("style", help="validate and manage video styles")
    style_commands = style_parser.add_subparsers(dest="style_command", required=True)
    for command in (
        "validate", "show", "presets", "resolve", "add", "save", "list", "default", "new",
        "questions", "from-answers",
    ):
        command_parser = style_commands.add_parser(command)
        command_parser.add_argument("--json", action="store_true")
        command_parser.set_defaults(func=_style)
        if command in {"validate", "add", "save", "from-answers"}:
            command_parser.add_argument("file", metavar="FILE")
        if command == "questions":
            command_parser.add_argument("--answers", metavar="FILE")
        if command in {"show", "resolve", "default"}:
            command_parser.add_argument("spec", nargs="?", metavar="SPEC")
        if command == "resolve":
            command_parser.add_argument("--out", metavar="PATH")
        if command == "add":
            command_parser.add_argument("--default", action="store_true")
            command_parser.add_argument("--replace", action="store_true")
        if command == "save":
            command_parser.add_argument("--name", metavar="NAME")
        if command in {"save", "from-answers"}:
            command_parser.add_argument("--workspace", metavar="DIR")
        if command == "new":
            command_parser.add_argument("name", nargs="?", metavar="NAME")
            command_parser.add_argument("--preset", choices=SHIPPED_PRESETS,
                                        default="clean_minimal")
            command_parser.add_argument("--out", metavar="FILE")
            command_parser.add_argument("--list", action="store_true")
    from yap_studio.gallery.still_command import add_still_parser

    add_still_parser(style_commands)
    from yap_studio.gallery.command import add_pick_parser

    add_pick_parser(style_commands, _review_port)
    from yap_studio.homepage.command import add_parser as add_home_parser

    add_home_parser(subparsers, _review_port)
    from yap_studio.brand_command import add_brand_parser

    add_brand_parser(subparsers)
    from yap_studio.faces.command import add_faces_parser

    add_faces_parser(subparsers)
    from yap_studio.joins.command import add_joins_parser

    add_joins_parser(subparsers)
    from yap_studio.matte.command import add_matte_parser

    add_matte_parser(subparsers)
    from yap_studio.planner.command import add_broll_parser, add_cards_parser

    add_cards_parser(subparsers)
    from yap_studio.covers.command import add_cover_parser
    add_cover_parser(subparsers)
    add_broll_parser(subparsers)
    from yap_studio.stock.command import register as add_stock_parser

    add_stock_parser(subparsers)
    from yap_studio.pictures.command import register as add_pictures_parser

    add_pictures_parser(subparsers)
    from yap_studio.templates_command import add_templates_parser

    add_templates_parser(subparsers)
    from yap_studio.media.commands import register as _register_media

    _register_media(subparsers)
    from yap_studio.studio.commands import register as _register_studio

    _register_studio(subparsers)
    from yap_studio.watch.command import add_watch_parser

    add_watch_parser(subparsers)
    from yap_studio.taste.cli import add_parser as add_taste_parser

    add_taste_parser(subparsers)
    from yap_studio.checks.fillers_gold import add_parser as add_fillers_score_parser

    add_fillers_score_parser(subparsers)
    from yap_studio.sfx.commands import register as register_sfx

    register_sfx(subparsers)
    from yap_studio.grade_command import add_parser as add_grade_parser

    add_grade_parser(subparsers)
    from yap_studio.video_type_command import add_parser as add_video_type_parser

    add_video_type_parser(subparsers)
    return parser


def _argument_parser(arguments: list[str]) -> tuple[argparse.ArgumentParser, list[str]]:
    """Select the parser and arguments for ordinary commands or filler decisions."""
    if arguments[:1] == ["feedback"] and (
        len(arguments) < 2 or arguments[1] not in {
            "build", "apply", "check", "reply", "replies", "ask", "answer",
            "list", "send", "setting", "--help", "-h",
        }
    ):
        from yap_studio.product_feedback import record_parser

        parser = record_parser()
        arguments = arguments[1:]
    elif arguments[:2] == ["words", "fix"]:
        parser = argparse.ArgumentParser(prog="yap-studio words fix")
        parser.add_argument("directory", metavar="DIR")
        parser.add_argument("--word", metavar="ID")
        parser.add_argument("--heard", metavar="TEXT")
        parser.add_argument("--to", metavar="TEXT")
        parser.add_argument("--list", action="store_true")
        parser.add_argument("--remove", action="store_true")
        parser.add_argument("--json", action="store_true")
        parser.set_defaults(func=_word_fixes)
        arguments = arguments[2:]
    elif len(arguments) >= 2 and arguments[0] == "fillers" and arguments[1] in {"set", "add"}:
        command = arguments[1]
        parser = argparse.ArgumentParser(prog=f"yap-studio fillers {command}")
        parser.add_argument("directory", metavar="DIR")
        if command == "set":
            parser.add_argument("identifier", metavar="ID")
            parser.add_argument("removal_status", choices=("proposed", "applied", "rejected"))
        else:
            parser.add_argument("--manual", nargs=3, required=True,
                                metavar=("TAKE", "START", "END"))
        parser.add_argument("--reason", metavar="TEXT")
        parser.add_argument("--json", action="store_true")
        parser.set_defaults(func=_fillers, fillers_command=command)
        arguments = arguments[2:]
    else:
        parser = _parser()
    return parser, arguments


def _main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface."""
    tools.add_homebrew_path()
    parser, arguments = _argument_parser(list(sys.argv[1:] if argv is None else argv))
    if arguments[:1] == ["review"] and "--json" in arguments:
        captured = StringIO()
        parse_exit = None
        with redirect_stderr(captured):
            try:
                args = parser.parse_args(arguments)
            except SystemExit as exc:
                parse_exit = exc
        if parse_exit is not None:
            if parse_exit.code:
                message = captured.getvalue().rsplit(": error: ", 1)[-1].strip()
                _review_error(YapError(message, error_type="validation_error",
                                       code="invalid_argument"), json_output=True)
            return int(parse_exit.code or 0)
        return int(args.func(args))
    try:
        args = parser.parse_args(arguments)
    except SystemExit as exc:
        return int(exc.code or 0)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    return int(func(args))


def main(argv: Sequence[str] | None = None) -> int:
    """Capture a surfaced Yap error only when its command actually fails."""
    from yap_studio.errors import _command_error
    from yap_studio.product_feedback import automatic

    arguments = list(sys.argv[1:] if argv is None else argv)
    observed: list[YapError] = []
    token = _command_error.set(observed)
    failed = False
    try:
        result = _main(arguments)
        failed = result != 0
        return result
    except YapError as exc:
        observed[:] = [exc]
        failed = True
        raise
    finally:
        _command_error.reset(token)
        if failed and observed:
            automatic(arguments[0] if arguments else "unknown", observed[-1].code)
