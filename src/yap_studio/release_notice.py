"""Optional, cached release notices for Home and doctor."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import tempfile
import threading
import time
from urllib.request import Request, urlopen

from yap_studio import install_info, paths
from yap_studio.cache import cache_dir, cache_root
from yap_studio.style.library import _config, _save_lock, config_root

_LATEST = "https://api.github.com/repos/cathrynlavery/yap-studio/releases/latest"
_RELEASES = "https://github.com/cathrynlavery/yap-studio/releases/tag/"
_TAG = re.compile(r"v\d+\.\d+\.\d+")
_DAY = 86400
_FILE = "release-notice.json"


def setting(value: bool) -> None:
    """Save the preference without replacing other creator choices."""
    with _save_lock(), paths.operation():
        config = _config()
        config["update_notice"] = value
        paths.atomic_write_text(config_root() / "config.json", json.dumps(config) + "\n")


def _enabled() -> bool:
    return (os.environ.get("YAP_STUDIO_NO_UPDATE_NOTICE") != "1"
            and _config().get("update_notice") is not False)


def _read() -> dict | None:
    try:
        data = json.loads((cache_root() / _FILE).read_text(encoding="utf-8"))
        checked = data["checked_at"]
        tag = data["latest_tag"]
        if (type(checked) not in (float, int) or not math.isfinite(checked)
                or not 0 <= checked <= time.time()
                or tag is not None and (not isinstance(tag, str) or not _TAG.fullmatch(tag))):
            return None
        return data
    except Exception:
        return None


def _fresh(data: dict | None) -> bool:
    return data is not None and time.time() - data["checked_at"] < _DAY


def _write(data: dict) -> None:
    folder = cache_dir()
    # Replace even corrupt cache JSON, which the normal output guard refuses to overwrite.
    with tempfile.TemporaryDirectory(dir=folder) as temporary, paths.operation():
        saved = paths.atomic_write_text(os.path.join(temporary, _FILE), json.dumps(data) + "\n")
        os.replace(saved, folder / _FILE)


def _refresh() -> None:
    if _fresh(_read()):
        return
    descriptor = os.open(cache_dir() / ".release-notice.lock",
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        cached = _read()
        if _fresh(cached):
            return
        data = {"checked_at": time.time(), "latest_tag": cached["latest_tag"] if cached else None}
        _write(data)
        request = Request(_LATEST, headers={"User-Agent": "Yap-Studio"})
        with urlopen(request, timeout=3) as response:
            tag = json.loads(response.read(65536))["tag_name"]
        if isinstance(tag, str) and _TAG.fullmatch(tag):
            data["latest_tag"] = tag
            _write(data)


def notice(*, refresh: bool = False) -> dict | None:
    """Read only the cache unless a bounded foreground refresh is requested."""
    try:
        if not _enabled():
            return None
        if refresh:
            _refresh()
        data = _read()
        tag = data["latest_tag"] if data else None
        version = install_info.running_version(None)
        if tag and install_info.compare_versions(tag[1:], version) == 1:
            return {"text": f'Yap Studio {tag[1:]} is out. Say "update Yap Studio".',
                    "url": _RELEASES + tag}
    except Exception:
        pass
    return None


def refresh_async() -> None:
    """Refresh stale state without putting network work on the Home request thread."""
    try:
        if _enabled() and not _fresh(_read()):
            threading.Thread(target=lambda: notice(refresh=True), daemon=True).start()
    except Exception:
        pass


def _command(args: argparse.Namespace) -> int:
    setting(args.value == "on")
    print(f"Update notices {args.value}.")
    return 0


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("update-notice", help="turn release notices on or off")
    parser.add_argument("value", choices=("on", "off"))
    parser.set_defaults(func=_command)
