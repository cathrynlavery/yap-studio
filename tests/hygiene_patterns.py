"""Shared repository and release archive hygiene patterns."""

from __future__ import annotations

import re

SYNC_SERVICE_FILES = {"src/yap_studio/setup_checks.py", "workspace/setup.py"}


def _machine_path_patterns() -> list[re.Pattern[str]]:
    slash = "/"
    return [
        re.compile(re.escape(slash + "Users" + slash) + r"[^/\s]+"),
        re.compile(re.escape(slash + "home" + slash) + r"[^/\s]+"),
        re.compile(re.escape(slash + "Volumes" + slash) + r"[^/\s]+"),
    ]


def _sync_service_patterns() -> list[re.Pattern[str]]:
    space = " "
    return [
        re.compile("Drop" + "box", re.IGNORECASE),
        re.compile("Cloud" + "Storage", re.IGNORECASE),
        re.compile("i" + "Cloud" + space + "Drive", re.IGNORECASE),
        re.compile("One" + "Drive", re.IGNORECASE),
        re.compile("Google" + space + "Drive", re.IGNORECASE),
    ]


def _secret_tool_patterns() -> list[re.Pattern[str]]:
    slash = "/"
    return [
        re.compile(r"\b" + "op" + r"\s+" + "read" + r"\b"),
        re.compile("op" + ":" + slash + slash),
        re.compile(r"\b" + "bw" + r"\s+" + "get" + r"\b"),
        re.compile(r"\b" + "lpass" + r"\s+" + "show" + r"\b"),
    ]


MACHINE_PATTERNS = _machine_path_patterns()
SECRET_TOOL_PATTERNS = _secret_tool_patterns()
SYNC_SERVICE_PATTERNS = _sync_service_patterns()


def _has_machine_details(line: str, relative: str) -> bool:
    patterns = MACHINE_PATTERNS + SECRET_TOOL_PATTERNS
    if relative not in SYNC_SERVICE_FILES:
        patterns += SYNC_SERVICE_PATTERNS
    return any(pattern.search(line) for pattern in patterns)


def _video_tool_patterns() -> list[re.Pattern[str]]:
    return [
        re.compile(re.escape("video" + "-use"), re.IGNORECASE),
        re.compile(r"\b" + re.escape("video" + "use") + r"\b", re.IGNORECASE),
        re.compile(re.escape("open" + "chatcut"), re.IGNORECASE),
        re.compile(re.escape("diffusion" + " studio"), re.IGNORECASE),
        re.compile(re.escape("open" + "montage"), re.IGNORECASE),
        re.compile(re.escape("kino" + "cut"), re.IGNORECASE),
        re.compile(re.escape("kyan" + "ite"), re.IGNORECASE),
        re.compile(r"\b" + re.escape("open" + "cut") + r"\b", re.IGNORECASE),
        re.compile(re.escape("vel" + "orn"), re.IGNORECASE),
        re.compile(re.escape("cap" + "cut"), re.IGNORECASE),
        re.compile(re.escape("pi" + "reel"), re.IGNORECASE),
        re.compile(re.escape("butter" + "cut"), re.IGNORECASE),
        re.compile(r"(?<![a-z])" + re.escape("prem" + "iere") + r"(?![a-z])", re.IGNORECASE),
        re.compile(r"(?<![a-z])" + re.escape("desc" + "ript") + r"(?![a-z])", re.IGNORECASE),
        re.compile(r"(?<![a-z])" + re.escape("dav" + "inci") + r"(?![a-z])", re.IGNORECASE),
        re.compile(r"\b" + re.escape("final " + "cut") + r"\b", re.IGNORECASE),
    ]


VIDEO_TOOL_PATTERNS = _video_tool_patterns()
