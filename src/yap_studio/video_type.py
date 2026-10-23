"""Video recipes and the creator's allowed B-roll sources."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from yap_studio import paths
from yap_studio.errors import InputError, OutputPathError, ValidationError

try:
    import fcntl
except ImportError:  # Windows has no fcntl.
    fcntl = None

VIDEO_TYPES = ("talking_head", "voiceover", "green_screen", "screen_recording")
BROLL_SOURCES = ("folder", "album", "stock")
USES_BROLL = ("talking_head", "voiceover", "green_screen")
QUESTIONS_SCHEMA = "yap-studio/video-type@1"
DEFAULTS_SCHEMA = "yap-studio/video-defaults@1"
STOCK_ACTION = ("Ask the creator first, then run yap-studio video-type set DIR with --broll stock "
                "and the video's other sources.")

QUESTIONS = [
    {"id": "video_type", "header": "Video type", "prompt": "What are you making?",
     "note": "Each video can be a different kind. Your answer is remembered as the default "
             "for the next video.",
     "kind": "choice", "choices": [
        {"value": "talking_head", "label": "Talking head",
         "description": "You on camera. B-roll, cards and moments where they help.",
         "recommended": True},
        {"value": "voiceover", "label": "Voiceover",
         "description": "Your voice over your photos and clips, in the order you shot them.",
         "recommended": False},
        {"value": "green_screen", "label": "Green screen",
         "description": "You cut out over screenshots, photos or clips.",
         "recommended": False},
        {"value": "screen_recording", "label": "Screen recording",
         "description": "Your screen is the picture, with your voice over it.",
         "recommended": False},
     ]},
    {"id": "broll_sources", "header": "B-roll", "prompt": "Where can the agent get B-roll?",
     "note": "Pick everything it may use. It uses nothing else.",
     "kind": "multi", "only_for": ["talking_head", "voiceover", "green_screen"], "choices": [
        {"value": "folder", "label": "Inbox or a folder",
         "description": "Clips and photos you drop in or point to.", "recommended": True},
        {"value": "album", "label": "A Photos album",
         "description": "You name the album. Shots follow the day you took them.",
         "recommended": True},
        {"value": "stock", "label": "Stock clips",
         "description": "Free clips found online. You pick each one before it is used.",
         "recommended": False},
        {"value": "none", "label": "None", "description": "Only what is in your take.",
         "recommended": False, "only_for": ["talking_head", "green_screen"]},
     ]},
]


def defaults_path() -> Path:
    """Keep video defaults beside the strict reviewer settings record."""
    from yap_studio.review.store import data_dir

    return data_dir() / "video-defaults.json"


def parse_sources(values: Sequence[str]) -> list[str]:
    """Validate CLI sources and return a deduplicated canonical list."""
    if isinstance(values, str) or any(value not in (*BROLL_SOURCES, "none") for value in values):
        raise ValidationError("broll", "choose folder, album, stock or none")
    if "none" in values and any(value != "none" for value in values):
        raise ValidationError("broll", "none cannot be combined with another source")
    return [source for source in BROLL_SOURCES if source in values]


def check_fit(video_type: str, broll_sources: Sequence[str] | None, *, voiceover: bool) -> None:
    """Refuse a recipe that cannot use the project's takes or pictures."""
    if video_type not in VIDEO_TYPES:
        raise ValidationError("video_type", "choose " + ", ".join(VIDEO_TYPES))
    message = action = None
    if video_type == "voiceover" and not voiceover:
        message = "A voiceover starts from an audio memo, and this project has video takes."
        action = ("Record the voice as a memo and put it in inbox/, or choose talking_head, "
                  "green_screen or screen_recording for this video.")
    elif video_type in {"talking_head", "green_screen"} and voiceover:
        label = "talking head" if video_type == "talking_head" else "green screen"
        message = f"A {label} needs a video take, and this project has audio memos."
        action = "Choose voiceover or screen_recording for this video."
    elif video_type == "voiceover" and broll_sources is not None and not broll_sources:
        message = "A voiceover needs pictures to show."
        action = "Choose folder, album or stock with --broll."
    if message:
        error = ValidationError("video_type", message)
        error.suggested_action = action
        raise error


def load_default() -> dict | None:
    """Read a complete default, naming the file when it needs repair."""
    target = defaults_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if (not isinstance(data, dict)
                or set(data) != {"schema", "video_type", "broll_sources"}
                or data["schema"] != DEFAULTS_SCHEMA
                or not isinstance(data["broll_sources"], list)
                or any(source not in BROLL_SOURCES for source in data["broll_sources"])
                or len(data["broll_sources"]) != len(set(data["broll_sources"]))):
            raise ValueError("wrong shape")
        check_fit(data["video_type"], data["broll_sources"],
                  voiceover=data["video_type"] == "voiceover")
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        error = InputError(str(target), "invalid video defaults JSON or shape")
        error.suggested_action = ("Fix this file, or run yap-studio video-type set "
                                  "--type TYPE --broll SOURCE again.")
        raise error from exc
    return {**data, "broll_sources": parse_sources(data["broll_sources"])}


def _storage_error(exc: OSError) -> OutputPathError:
    error = OutputPathError(f"Cannot write video defaults in {defaults_path()}: {exc}")
    error.suggested_action = "Set YAP_STUDIO_DATA_DIR to a writable folder."
    return error


@contextmanager
def defaults_lock() -> Iterator[None]:
    """Serialize default changes across projects and standalone answers."""
    target = defaults_path().with_suffix(".lock")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        lock = target.open("a+b")
    except OSError as exc:
        raise _storage_error(exc) from exc
    with lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _save_default_unlocked(video_type: str, broll_sources: Sequence[str], *,
                           only_if_missing: bool = False) -> bool:
    """Write an answer while the caller holds defaults_lock."""
    sources = parse_sources(broll_sources)
    check_fit(video_type, sources, voiceover=video_type == "voiceover")
    if only_if_missing and load_default() is not None:
        return False
    target = defaults_path()
    try:
        paths.atomic_write_text(target, json.dumps({"schema": DEFAULTS_SCHEMA,
            "video_type": video_type, "broll_sources": sources}, indent=2) + "\n")
    except OSError as exc:
        raise _storage_error(exc) from exc
    return True


def save_default(video_type: str, broll_sources: Sequence[str], *,
                 only_if_missing: bool = False) -> bool:
    """Atomically save an answer, optionally keeping another process's first answer."""
    with defaults_lock():
        return _save_default_unlocked(video_type, broll_sources, only_if_missing=only_if_missing)


def resolve(*, voiceover: bool, video_type: str | None = None,
            broll_sources: Sequence[str] | None = None,
            default: dict | None = None) -> tuple[str | None, list[str] | None]:
    """Use explicit choices, then a default that fits the files."""
    sources = parse_sources(broll_sources) if broll_sources is not None else None
    kind = video_type
    if kind is None and default is not None:
        kind = default["video_type"]
        try:
            check_fit(kind, None, voiceover=voiceover)
        except ValidationError:
            kind = "voiceover" if voiceover else "talking_head"
    if sources is None:
        if kind == "screen_recording":
            sources = []
        elif default is not None and kind is not None:
            try:
                check_fit(kind, default["broll_sources"], voiceover=voiceover)
            except ValidationError:
                pass
            else:
                sources = parse_sources(default["broll_sources"])
    if kind is not None:
        check_fit(kind, sources, voiceover=voiceover)
    return kind, sources


def describe(video_type: str | None, broll_sources: Sequence[str] | None) -> str:
    """Describe both choices on one status line."""
    if video_type is None:
        kind = "not chosen yet (yap-studio video-type show)"
    else:
        kind = video_type.replace("_", " ")
    if broll_sources is None:
        return kind if video_type is None else f"{kind}; B-roll not chosen yet"
    if not broll_sources:
        return f"{kind}; no B-roll"
    labels = {"folder": "your inbox or a folder", "album": "a Photos album", "stock": "stock clips"}
    names = [labels[source] for source in broll_sources]
    sources = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"{kind}; B-roll from {sources}"


def stock_allowed(style: Mapping, project_dir: Path | None) -> tuple[bool, str | None]:
    """Recorded project sources take precedence over the older style permission."""
    if project_dir is not None:
        try:
            data = json.loads((Path(project_dir) / "project.json").read_bytes())
        except (OSError, ValueError):
            data = None
        if isinstance(data, dict) and isinstance(data.get("broll_sources"), list):
            allowed = "stock" in data["broll_sources"]
            return allowed, (None if allowed else
                             "Stock clips are not in this video's B-roll sources.")
    return style.get("broll", {}).get("stock_clips") == "ask", None
