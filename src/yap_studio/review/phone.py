"""Find the link a phone can open: Tailscale when it is set up, otherwise the local Wi-Fi."""

from __future__ import annotations

import ipaddress
import json
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

from yap_studio.errors import YapError

_TAILSCALE_APP = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


class PhoneLinkError(YapError):
    """No network path from a phone to this computer."""

    def __init__(self, message: str, suggested_action: str) -> None:
        super().__init__(message, error_type="phone_link", code="phone_link_unavailable",
                         suggested_action=suggested_action)


def link_label(via: str) -> str:
    """Describe the phone link transport beside its URL."""
    return "Encrypted (Tailscale)" if via == "tailscale" else "Not encrypted (same Wi-Fi)"


def is_lan_address(value: str) -> bool:
    """Accept a private IPv4 address a phone on the same network can reach."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return (address.version == 4 and address.is_private and not address.is_loopback
            and not address.is_link_local and not address.is_unspecified
            and address not in _CGNAT)


def _hardware_lan_address() -> str | None:
    """Find a private address on a macOS hardware port, preferring Wi-Fi."""
    try:
        result = subprocess.run(["networksetup", "-listallhardwareports"], capture_output=True,
                                text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode:
        return None
    ports: list[tuple[str, str]] = []
    hardware_port = ""
    for line in result.stdout.splitlines():
        if line.startswith("Hardware Port: "):
            hardware_port = line.removeprefix("Hardware Port: ").strip()
        elif line.startswith("Device: ") and hardware_port:
            device = line.removeprefix("Device: ").strip()
            if device and not device.startswith(("utun", "ipsec", "ppp", "gif", "stf", "bridge",
                                                 "awdl", "llw")):
                ports.append((hardware_port, device))
            hardware_port = ""
    for _, device in sorted(ports, key=lambda item: item[0] != "Wi-Fi"):
        try:
            result = subprocess.run(["ipconfig", "getifaddr", device], capture_output=True,
                                    text=True, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            continue
        address = result.stdout.strip()
        if result.returncode == 0 and is_lan_address(address):
            return address
    return None


def lan_address() -> str | None:
    """Return this computer's address on its local network, or None when it has none."""
    if sys.platform == "darwin":
        address = _hardware_lan_address()
        if address is not None:
            return address
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # Connecting a UDP socket sends nothing; it only picks the outgoing interface.
            probe.connect(("10.254.254.254", 9))
            address = probe.getsockname()[0]
    except OSError:
        return None
    return address if is_lan_address(address) else None


def _tailscale_binary() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    return _TAILSCALE_APP if Path(_TAILSCALE_APP).is_file() else None


def _tailscale(binary: str, *args: str) -> str:
    timeout = 5 if args == ("status", "--json") else 20
    result = subprocess.run([binary, *args], capture_output=True, text=True, timeout=timeout,
                            check=False)
    if result.returncode:
        raise OSError((result.stderr or result.stdout).strip() or f"exit {result.returncode}")
    return result.stdout


def tailscale_name() -> str | None:
    """Return this computer's Tailscale name when Tailscale is installed and connected."""
    binary = _tailscale_binary()
    if binary is None:
        return None
    for attempt in range(3):
        try:
            status = json.loads(_tailscale(binary, "status", "--json"))
            if not isinstance(status, dict):
                raise ValueError("expected a status object")
            if status.get("BackendState") != "Running":
                return None
            own = status.get("Self") or {}
            if not isinstance(own, dict):
                raise ValueError("expected a Self object")
            name = str(own.get("DNSName") or "").rstrip(".")
            return name or None
        except (OSError, ValueError, subprocess.TimeoutExpired):
            if attempt < 2:
                time.sleep(1)
    return None


def remove_tailscale_mapping(binary: str, external_port: int) -> None:
    """Remove a mapping started by this review, warning if removal fails."""
    try:
        _tailscale(binary, "serve", f"--https={external_port}", "off")
    except (OSError, subprocess.SubprocessError):
        print(f"warning: could not remove Tailscale mapping; "
              f"run tailscale serve --https={external_port} off", file=sys.stderr)


def tailscale_origin(port: int) -> tuple[str, Callable[[], None] | None]:
    """Find or start a tailnet-only HTTPS mapping and return its origin and cleanup."""
    binary = _tailscale_binary()
    name = tailscale_name()
    if binary is None or name is None:
        raise OSError("Tailscale is not connected")
    target = f"http://127.0.0.1:{port}"
    try:
        config = json.loads(_tailscale(binary, "serve", "status", "--json") or "{}") or {}
    except ValueError as exc:
        raise OSError(f"unreadable tailscale serve status: {exc}") from exc
    web = config.get("Web") or {}
    tcp = config.get("TCP") or {}
    funnel = config.get("AllowFunnel") or {}
    for host_port, value in web.items():
        external = host_port.rsplit(":", 1)[-1]
        if ((value.get("Handlers") or {}).get("/", {}).get("Proxy") == target
                and (tcp.get(external) or {}).get("HTTPS") and not funnel.get(host_port)):
            origin = f"https://{name}" if external == "443" else f"https://{name}:{external}"
            return origin, None
    occupied = ({int(item.rsplit(":", 1)[-1]) for item in web}
                | {int(item) for item in tcp}
                | {int(item.rsplit(":", 1)[-1]) for item, enabled in funnel.items() if enabled})
    external_port = port if port not in occupied else next(
        candidate for candidate in range(18000, 19000) if candidate not in occupied)
    _tailscale(binary, "serve", "--bg", f"--https={external_port}", target)
    cleanup = partial(remove_tailscale_mapping, binary, external_port)
    return f"https://{name}:{external_port}", cleanup


def no_network() -> PhoneLinkError:
    """Explain that neither Tailscale nor a local network is available."""
    return PhoneLinkError(
        "No network your phone can reach: this computer is not on Wi-Fi or a local network, "
        "and Tailscale is not connected.",
        "Join the same Wi-Fi as your phone, or set up Tailscale on both, then run this again.",
    )
