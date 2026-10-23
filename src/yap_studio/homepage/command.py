"""CLI entry points for creator home and its background request listener."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlsplit

from yap_studio import background
from yap_studio.errors import ValidationError, YapError


def _result(result: dict, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    elif "event" in result:
        event = result["event"]
        if event == "request":
            print(f"Request {result['request']['id']}: {result['next']}", flush=True)
        elif event == "timeout":
            print("No new requests or review updates.", flush=True)
        else:
            detail = result.get("batch") or result.get("version")
            print(f"{result['project']}: {event} {detail}.", flush=True)
    else:
        print(f"Request {result['id']}: {result['state']}.", flush=True)


def run(args: argparse.Namespace) -> int:
    """Serve, stop, wait for or acknowledge a home page request."""
    # Keep CLI parser construction free of the bridge and feedback imports.
    from yap_studio.homepage import bridge, detach

    try:
        if args.home_command == "wait":
            _result(
                bridge.wait(bridge.workspace_dir(args.workspace or "."), args.timeout), args.json
            )
            return 0
        if args.home_command == "ack":
            _result(
                bridge.ack(
                    bridge.workspace_dir(args.workspace or "."),
                    args.id,
                    args.state,
                    project=args.project,
                    message=args.message,
                ),
                args.json,
            )
            return 0
        if args.stop:
            if args.detach or args.phone or args.host is not None or args.port is not None:
                raise ValidationError("stop", "Use --stop with only --workspace and --json.")
            workspace = Path(args.workspace).expanduser().resolve() if args.workspace else None
            return detach.stop(workspace=workspace, json_output=args.json)
        workspace = bridge.workspace_dir(args.workspace or ".")
        if args.phone and args.host is not None:
            raise ValidationError("phone", "Use --phone or --host, not both.")
        if args.detach:
            if not background.posix_supported():
                raise YapError(
                    "Background home pages need macOS or Linux.",
                    code="background_unsupported",
                    suggested_action="Run home without --detach and keep it open.",
                )
            return detach.start(args, workspace)
        return _serve(args, workspace)
    except (YapError, OSError, ValueError, subprocess.SubprocessError) as exc:
        if not isinstance(exc, YapError):
            exc = YapError(
                "The home page could not start. Check the workspace and try again.",
                code="home_start_failed",
            )
        background.print_error(exc, json_output=args.json)
        return 2


def _serve(args: argparse.Namespace, workspace: Path) -> int:
    from yap_studio.homepage import detach
    from yap_studio.homepage.server import HomeServer
    from yap_studio.review import phone

    server = None
    cleanup = None
    previous = {}

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                previous[signum] = signal.signal(signum, stop)
        public = None
        via = None
        if args.phone:
            name = phone.tailscale_name()
            if name:
                server = HomeServer(workspace, allowed_hosts=(name,), port=args.port or 0)
                try:
                    public, cleanup = phone.tailscale_origin(server.port)
                    via = "tailscale"
                except (OSError, ValueError, subprocess.SubprocessError):
                    server.server_close()
                    server = None
            if server is None:
                address = phone.lan_address()
                if address is None:
                    raise phone.no_network()
                server = HomeServer(
                    workspace, host="0.0.0.0", port=args.port or 0, allowed_hosts=(address,)
                )
                public, via = f"http://{address}:{server.port}", "wifi"
        else:
            server = HomeServer(workspace, host=args.host or "127.0.0.1", port=args.port or 0)
        phone_link = None
        if public:
            local = urlsplit(server.url)
            phone_link = {"via": via, "url": f"{public}{local.path}?{local.query}"}
        server.state.phone = phone_link
        result = {"url": server.url, "phone": phone_link}
        if args.detached_record:
            background.write_record(
                Path(args.detached_record), result, cwd=str(workspace), workspace=str(workspace)
            )
        detach.show(result, json_output=args.json, no_open=args.no_open)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if server:
                server.server_close()
            if cleanup:
                cleanup()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            if args.detached_record:
                background.remove_record(Path(args.detached_record))
    return 0


def add_parser(subparsers: argparse._SubParsersAction, port_type) -> None:
    """Register home and its request bridge commands without loading the HTTP server."""
    parser = subparsers.add_parser("home", help="open your creator workspace home")
    parser.set_defaults(func=run)
    parser.add_argument("--workspace", metavar="DIR")
    parser.add_argument("--phone", action="store_true")
    parser.add_argument("--host", metavar="HOST")
    parser.add_argument("--port", type=port_type)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--detached-record", help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="home_command")
    waiter = commands.add_parser("wait", help="wait for one creator request or review update")
    waiter.add_argument("--workspace", metavar="DIR", default=argparse.SUPPRESS)
    waiter.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    waiter.add_argument("--timeout", type=float, default=3000)
    acknowledge = commands.add_parser("ack", help="acknowledge a creator request")
    acknowledge.add_argument("id")
    acknowledge.add_argument("--workspace", metavar="DIR", default=argparse.SUPPRESS)
    acknowledge.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    acknowledge.add_argument("--state", required=True, choices=("claimed", "done", "failed"))
    acknowledge.add_argument("--project", metavar="DIR")
    acknowledge.add_argument("--message", metavar="TEXT")
