"""Local HTTP gallery for live style choices and a saved default."""

from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import re
import secrets
import tempfile
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

import numpy as np

from yap_studio import cards, ffmpeg
from yap_studio.captions import CaptionWord
from yap_studio.errors import ValidationError, YapError
from yap_studio.faces.model import Box
from yap_studio.gallery import live
from yap_studio.grade import (
    EYEDROPPER_DEFAULT_STRENGTH,
    GradeError,
    GradeSpec,
    apply_rgb,
    color_info,
)
from yap_studio.hf_registry import cards as registry_cards
from yap_studio.hf_registry import catalog as registry_catalog
from yap_studio.hf_registry import install as registry_install
from yap_studio.local_http import LocalHTTPServer
from yap_studio.render import fonts, hyperframes, user_fonts
from yap_studio.render.hyperframes import RendererInstall
from yap_studio.sfx import bleep
from yap_studio.style import (
    PRESETS_DIR,
    SCHEMA_PATH,
    SHIPPED_PRESETS,
    ResolvedStyle,
    list_entries,
    read_style_file,
    resolve_style_data,
    save_pick,
    unbranded,
    validate_style_data,
)
from yap_studio.style.library import find_by_name
from yap_studio.style.site_colors import FetchError, site_colors

_STATIC = Path(__file__).parent / "static"
_STATIC_TYPES = {
    "pick.html": "text/html; charset=utf-8", "pick.css": "text/css; charset=utf-8",
    "pick.js": "text/javascript; charset=utf-8",
    "choices.mjs": "text/javascript; charset=utf-8",
    "shot.jpg": "image/jpeg",
}
_ASSET = re.compile(r"[A-Za-z0-9_.-]+\Z")
_LAYER = re.compile(r"/layer/([0-9a-f]{64})/([0-9]+)/(.*)\Z")
_MAX_BODY = 1024 * 1024
_CACHE_LIMIT = 128
_STILL_CACHE_LIMIT = 8
_STILL_MAX_WIDTH = 1080
_STILL_MAX_HEIGHT = 1920
_CATEGORIES = {
    "Words": ("text_tag", "checklist", "steps", "flow", "terminal", "quote", "specs_list",
              "name_tag", "success_check", "gravity_drop", "choices"),
    "Numbers": ("stat", "chart", "progress_stat", "big_number"),
    "Media": ("image_card", "image_compare", "transfer", "callout_circle", "icon_strike"),
    "Chat": ("prompt_bar", "comment_cta", "comment_drop"),
    "Editor": ("edit_timeline", "storyboard", "jiggle_delete", "file_drop"),
    "Video": ("video_card", "playing_grid", "reel_feed"),
}
_PAGE_CSP = ("default-src 'self'; img-src 'self' data:; font-src 'self'; "
             "style-src 'self'; script-src 'self'; "
             "frame-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
             "frame-ancestors 'none'")
_LAYER_CSP = ("default-src 'self'; img-src 'self' data:; font-src 'self'; "
              "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
              "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'")
_LAYER_404_SCRIPT = """<script>
window.addEventListener('error', async (event) => {
  const asset = event.target.currentSrc || event.target.src || event.target.href;
  if (!asset) return;
  try {
    const url = new URL(asset, location.href);
    if (url.origin !== location.origin) return;
    if ((await fetch(url, {method: 'HEAD'})).status === 404) {
      parent.postMessage({type: 'gallery-layer-404', url: url.href}, location.origin);
    }
  } catch { /* A network error can recover on the next request. */ }
}, true);
</script>"""


class _GalleryHTTPServer(LocalHTTPServer):
    # A tab can open many composition frames and their assets together.
    request_queue_size = 128


class _RequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        self.status, self.message = status, message


class GalleryServer:
    """Serve one creator's still and prepare bounded live compositions on demand."""

    def __init__(
        self, image: Path, style: dict, work_dir: Path, *, install: RendererInstall,
        gsap: Path | None = None, workspace: Path | None = None,
        host: str = "127.0.0.1", port: int = 0, preview: dict | None = None,
        allowed_hosts: tuple[str, ...] = (), from_library: bool = False,
    ) -> None:
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValidationError("port", "must be from 0 to 65535")
        validate_style_data(style)
        self.style = style
        self.still = Path(image)
        self.preview = preview or {"source": "sample", "sentence": live.SAMPLE_SENTENCE,
                                   "words": [], "generation": 0}
        self.preview = {**self.preview, "screen": None, "screen_checked": False}
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace
        self.from_library = from_library
        self.last_request = time.monotonic()
        self.saved_at: float | None = None
        self.allowed_hosts = {name.lower() for name in allowed_hosts}
        self.runtime = live.runtime_file(install)
        self.gsap = gsap
        self.token = secrets.token_urlsafe(24)
        self.host = host
        self._lock = threading.RLock()
        self._cutout: Path | None = None
        self._prepare_lock = threading.Lock()
        self._site_colors_lock = threading.Lock()
        self._still_lock = threading.Lock()
        self._still_cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._still_rgb_cache: OrderedDict[tuple[int, Path], np.ndarray] = OrderedDict()
        self._cutout_lock = threading.Lock()
        self._cutout_cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._cutout_rgba_cache: OrderedDict[tuple[int, Path], np.ndarray] = OrderedDict()
        self._site_colors_cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_limit = _CACHE_LIMIT
        self._cache: OrderedDict[str, tuple[tempfile.TemporaryDirectory, list[live.LiveLayer]]] = (
            OrderedDict()
        )
        self._http = _GalleryHTTPServer((host, port), self._handler_type())
        self._http.daemon_threads = True
        self.port = self._http.server_port
        link_host = "127.0.0.1" if host == "0.0.0.0" else host
        self.url = f"http://{link_host}:{self.port}/?t={quote(self.token, safe='')}"

    def serve_forever(self) -> None:
        self._http.serve_forever()

    def shutdown(self) -> None:
        self._http.shutdown()

    def server_close(self) -> None:
        self._http.server_close()
        with self._lock:
            for folder, _ in self._cache.values():
                folder.cleanup()
            self._cache.clear()
            self._still_cache.clear()
            self._still_rgb_cache.clear()
            self._cutout_cache.clear()
            self._cutout_rgba_cache.clear()

    def update_preview(self, result: dict) -> None:
        """Publish a completed still and sentence atomically to request threads."""
        image = Path(result["still"])
        if not image.is_file():
            raise ValidationError("preview", "the prepared still is missing")
        with self._lock:
            self.still = image
            self.preview = {**result, "generation": self.preview["generation"] + 1,
                            "screen": None, "screen_checked": False}
            self._cutout = None

    def update_cutout(self, generation: int, path: Path) -> None:
        """Publish a cutout only while its still generation is current."""
        with self._lock:
            if generation == self.preview["generation"] and path.is_file():
                self._cutout = path

    def update_screen(self, generation: int, screen: str | None) -> None:
        """Publish a screen check only while its still generation is current."""
        with self._lock:
            if generation == self.preview["generation"]:
                self.preview.update(screen=screen, screen_checked=True)

    def preview_image(self) -> tuple[int, Path]:
        """Read the current generation and still together."""
        with self._lock:
            return self.preview["generation"], self.still

    def cutout_image(self, query: dict[str, list[str]]) -> bytes:
        """Return a bounded, cached PNG grade with the original person alpha."""
        if set(query) - {"g", "t", "look", "strength", "color"} or any(
            len(values) != 1 for values in query.values()
        ):
            raise ValidationError("cutout", "use each supported cutout parameter once")
        generation = query.get("g", ["0"])[0]
        if not generation.isdecimal():
            raise ValidationError("g", "use a non-negative generation number")
        try:
            strength = float(query.get("strength", ["1"])[0])
        except ValueError:
            raise ValidationError("strength", "use a number from 0 to 1") from None
        spec = GradeSpec.from_dict({"look": query.get("look", ["none"])[0],
                                    "strength": strength, "color": query.get("color", [None])[0]})
        with self._lock:
            image = self._cutout
            if (int(generation) != self.preview["generation"] or image is None
                    or not image.is_file()):
                raise _RequestError(404, "cutout not found")
            if not spec.active:
                try:
                    return image.read_bytes()
                except FileNotFoundError:
                    raise _RequestError(404, "cutout not found") from None
        key = (int(generation), image, spec.look, spec.strength, spec.color)
        with self._cutout_lock:
            if key not in self._cutout_cache:
                rgba = self._cutout_rgba(image, int(generation))
                graded = rgba.copy()
                graded[..., :3] = np.rint(apply_rgb(rgba[..., :3] / 255, spec) * 255).astype(
                    np.uint8)
                height, width, _ = rgba.shape
                with tempfile.TemporaryDirectory(
                    prefix="grade-cutout-", dir=self.work_dir,
                ) as folder:
                    raw = Path(folder) / "person.rgba"
                    raw.write_bytes(graded.tobytes())
                    result = ffmpeg.run([
                        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgba",
                        "-video_size", f"{width}x{height}", "-i", str(raw), "-frames:v", "1",
                        "-c:v", "png", "-pix_fmt", "rgba", "-threads", "1", "-f", "image2pipe", "-",
                    ], timeout_s=30, binary=True).stdout
                self._cutout_cache[key] = result
                if len(self._cutout_cache) > _STILL_CACHE_LIMIT:
                    self._cutout_cache.popitem(last=False)
            self._cutout_cache.move_to_end(key)
            with self._lock:
                if int(generation) != self.preview["generation"] or image != self._cutout:
                    raise _RequestError(404, "cutout not found")
                return self._cutout_cache[key]

    def _cutout_rgba(self, image: Path, generation: int) -> np.ndarray:
        """Decode the original PNG once per generation, under the cutout lock."""
        key = (generation, image)
        if key not in self._cutout_rgba_cache:
            info = ffmpeg.probe(image, count_frames=False)
            if not info.width or not info.height:
                raise GradeError("Cannot read this cutout.", "Try another preview still.")
            decoded = ffmpeg.run([
                "ffmpeg", "-v", "error", "-i", str(image), "-frames:v", "1",
                "-pix_fmt", "rgba", "-f", "rawvideo", "-",
            ], timeout_s=30, binary=True).stdout
            self._cutout_rgba_cache[key] = np.frombuffer(decoded, dtype=np.uint8).reshape(
                info.height, info.width, 4)
            if len(self._cutout_rgba_cache) > 2:
                self._cutout_rgba_cache.popitem(last=False)
        self._cutout_rgba_cache.move_to_end(key)
        return self._cutout_rgba_cache[key]

    def fail_preview(self) -> None:
        """Keep the sample visible and report a failed background preparation."""
        from yap_studio.gallery.still import sample_still

        with self._lock:
            self.still = sample_still()
            self._cutout = None
            self.preview = {"source": "sample", "sentence": live.SAMPLE_SENTENCE,
                            "words": [], "pending": False,
                            "screen": None, "screen_checked": False,
                            "message": "Could not read your newest take; showing the sample still.",
                            "generation": self.preview["generation"] + 1}

    def _registry_kinds(self) -> dict:
        kinds = {}
        for kind in registry_cards.kinds():
            name = registry_cards.item_name(kind)
            if registry_install.installed(name) is None:
                continue
            entry = registry_catalog.item(name)
            kinds[kind] = {"category": entry["kind"]["category"], "opt_in": True,
                           "off": cards.card_kind_off(self.style, kind),
                           "label": entry["kind"]["label"], "registry": True}
        return kinds

    def state(self) -> dict:
        from yap_studio.homepage.link import for_path
        from yap_studio.video_type import load_default

        try:
            default = load_default()
        except (YapError, OSError, ValueError):
            default = None
        music_answer = None
        if self.workspace is not None:
            try:
                answers = json.loads((self.workspace / "style/answers.json").read_text(
                    encoding="utf-8"))
                if (isinstance(answers, dict)
                        and answers.get("music") in ("none", "quiet", "upbeat")):
                    music_answer = answers["music"]
            except (OSError, ValueError):
                pass
        presets = []
        for name in SHIPPED_PRESETS:
            data = unbranded(read_style_file(PRESETS_DIR / f"{name}.json"))
            presets.append({"id": name, "name": data["name"], "style": data})
        library = []
        for entry in sorted(list_entries(), key=lambda item: (
            -item.path.stat().st_mtime_ns, item.id
        )):
            source = entry.path.with_name(f"{entry.id}.source.json")
            try:
                count = json.loads(source.read_text(encoding="utf-8")).get("videos")
            except (OSError, ValueError):
                count = None
            library.append({"id": entry.id, "name": entry.name,
                            "videos": count, "style": unbranded(read_style_file(entry.path))})
        with self._lock:
            preview = dict(self.preview)
            image = self.still
            cutout = f"/cutout?g={preview['generation']}" if self._cutout is not None else None
        return {
            "schema": json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
            "presets": presets, "library": library, "style": self.style,
            "from_library": self.from_library,
            "music_answer": music_answer,
            "video_type": default["video_type"] if default else None,
            "bleep_builtin": list(bleep.builtin_words()),
            "home": for_path(self.workspace),
            "fonts": {"bundled": list(fonts.BUNDLED),
                      "creator": sorted({face.family for face in user_fonts.list_fonts()})},
            "card_kinds": {
                **{kind: {"category": category, "opt_in": kind in cards.OPT_IN_KINDS,
                          "off": cards.card_kind_off(self.style, kind)}
                   for category, kinds in _CATEGORIES.items() for kind in kinds},
                **self._registry_kinds(),
            },
            "preview": {**preview, "still": "/still?g=" + str(preview["generation"]),
                        "cutout": cutout,
                        "grade_color": self.grade_color(scene=(image, preview["generation"])),
                        "grade_default_strength": EYEDROPPER_DEFAULT_STRENGTH,
                        "sentence": preview.get("sentence") or live.SAMPLE_SENTENCE},
        }

    def _words(self, record: dict | None = None) -> list[CaptionWord]:
        if record is None:
            with self._lock:
                record = self.preview
        if record.get("words"):
            return [CaptionWord(f"w{index}", item["text"], item["start"], item["end"])
                    for index, item in enumerate(record["words"])]
        return live.sample_words(record.get("sentence") or live.SAMPLE_SENTENCE)

    def website_colors(self, url: str) -> dict:
        """Cache engine lookups for this server run, including concurrent requests."""
        if len(url) > 2048:
            raise FetchError("Use a website address like example.com.")
        key = url.strip()
        with self._site_colors_lock:
            if key not in self._site_colors_cache:
                self._site_colors_cache[key] = site_colors(key)
                if len(self._site_colors_cache) > _CACHE_LIMIT:
                    self._site_colors_cache.popitem(last=False)
            self._site_colors_cache.move_to_end(key)
            return self._site_colors_cache[key]

    def _card_image(self) -> Path:
        """Give image cards the same drawing when the current still is SVG."""
        if self.still.suffix.lower() != ".svg":
            return self.still
        return Path(__file__).with_name("sample-still.jpg")

    def still_image(self, query: dict[str, list[str]]) -> tuple[bytes, str]:
        """Serve the unchanged still or a bounded, cached RGB grade of its generation."""
        allowed = {"g", "look", "strength", "color", "t", "slider", "text_color", "ink_color"}
        if set(query) - allowed or any(
            len(values) != 1 for values in query.values()
        ):
            raise ValidationError("still", "use each supported still parameter once")
        generation = query.get("g", ["0"])[0]
        if not generation.isdecimal():
            raise ValidationError("g", "use a non-negative generation number")
        try:
            strength = float(query.get("strength", ["1"])[0])
        except ValueError:
            raise ValidationError("strength", "use a number from 0 to 1") from None
        spec = GradeSpec.from_dict({"look": query.get("look", ["none"])[0],
                                    "strength": strength, "color": query.get("color", [None])[0]})
        if query.get("slider", ["0"])[0] not in {"0", "1"}:
            raise ValidationError("slider", "use 0 or 1")
        slider = query.get("slider", ["0"])[0] == "1"
        if slider and not spec.active:
            raise ValidationError("slider", "pick an active grade first")
        from yap_studio.grade import COLOR_PATTERN, compare_still

        default_text = self.style.get("captions", {}).get("text_color", "#FFFFFF")
        default_ink = self.style.get("graphics", {}).get("ink_color", "#222222")
        text = query.get("text_color", [default_text])[0]
        ink = query.get("ink_color", [default_ink])[0]
        if any(COLOR_PATTERN.fullmatch(colour) is None for colour in (text, ink)):
            raise ValidationError("slider", "use hex colours like #FFFFFF for the handle")
        with self._lock:
            image = self.still
            key = (self.preview["generation"], spec.look, spec.strength, spec.color,
                   slider, text, ink)
        if "look" in query and image.suffix.lower() == ".svg":
            image = Path(__file__).with_name("sample-still.jpg")
        mime = mimetypes.guess_type(image.name)[0]
        if mime not in {"image/jpeg", "image/png", "image/webp", "image/svg+xml"}:
            raise _RequestError(404, "still not found")
        if not spec.active:
            return image.read_bytes(), mime
        with self._still_lock:
            if key in self._still_cache:
                self._still_cache.move_to_end(key)
                return self._still_cache[key], "image/jpeg"
            rgb = self._still_rgb(image, key[0])
            height, width, _ = rgb.shape
            graded = np.rint(apply_rgb(rgb / 255, spec) * 255).astype(np.uint8)
            if slider:
                graded = compare_still(rgb, graded, {"captions": {"text_color": text},
                                                     "graphics": {"ink_color": ink}})
            with tempfile.TemporaryDirectory(prefix="grade-still-", dir=self.work_dir) as folder:
                raw = Path(folder) / "still.rgb"
                raw.write_bytes(graded.tobytes())
                result = ffmpeg.run([
                    "ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", f"{width}x{height}", "-i", str(raw),
                    "-frames:v", "1", "-c:v", "mjpeg", "-q:v", "2", "-pix_fmt", "yuvj444p",
                    "-f", "image2pipe", "-",
                ], timeout_s=30, binary=True).stdout
            self._still_cache[key] = result
            if len(self._still_cache) > _STILL_CACHE_LIMIT:
                self._still_cache.popitem(last=False)
            return result, "image/jpeg"

    def _still_rgb(self, image: Path, generation: int) -> np.ndarray:
        """Decode a bounded ungraded raster, while the caller holds the still lock."""
        key = (generation, image)
        if key not in self._still_rgb_cache:
            info = ffmpeg.probe(image, count_frames=False)
            if not info.width or not info.height:
                raise GradeError("Cannot read this still.", "Pick a colour with the hex field.")
            scale = min(1, _STILL_MAX_WIDTH / info.width, _STILL_MAX_HEIGHT / info.height)
            width, height = max(1, round(info.width * scale)), max(1, round(info.height * scale))
            decoded = ffmpeg.run([
                "ffmpeg", "-v", "error", "-i", str(image), "-frames:v", "1",
                "-vf", f"scale={width}:{height}:flags=area",
                "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
            ], timeout_s=30, binary=True).stdout
            self._still_rgb_cache[key] = np.frombuffer(decoded, dtype=np.uint8).reshape(
                height, width, 3)
            if len(self._still_rgb_cache) > 2:
                self._still_rgb_cache.popitem(last=False)
        self._still_rgb_cache.move_to_end(key)
        return self._still_rgb_cache[key]

    def grade_color(self, color: str | None = None, *,
                    scene: tuple[Path, int] | None = None) -> dict:
        """Describe a pick or the current scene's average top-third colour."""
        if color is None:
            if scene is None:
                with self._lock:
                    scene = (self.still, self.preview["generation"])
            image, generation = scene
            if image.suffix.lower() == ".svg":
                image = Path(__file__).with_name("sample-still.jpg")
            try:
                with self._still_lock:
                    rgb = self._still_rgb(image, generation)
                    mean = np.rint(rgb[:max(1, len(rgb) // 3)].mean(axis=(0, 1))).astype(int)
            except (YapError, OSError):
                return {"color": None, "chroma": 0, "colorful": False,
                        "message": "Cannot read this still. Pick a colour with the hex field."}
            color = "#" + "".join(f"{value:02X}" for value in mean)
        info = color_info(color)
        if not info["colorful"]:
            info["message"] = GradeError(
                "This colour is too grey. Pick a more colourful point or colour.").message
        return info

    def layers(self, data: dict) -> dict:
        raw, tile = data.get("style"), data.get("tile", {})
        if not isinstance(raw, dict) or not isinstance(tile, dict):
            raise ValidationError("layers", "style and tile must be objects")
        validate_style_data(raw)
        tab = tile.get("tab", "preview")
        if tab not in {"preview", "captions", "title", "chapter", "cards"}:
            raise ValidationError("tile.tab", "choose preview, captions, title, chapter or cards")
        branded = tile.get("branded", True)
        if type(branded) is not bool:
            raise ValidationError("tile.branded", "expected a boolean")
        style_data = resolve_style_data(raw) if branded else unbranded(raw)
        if tab == "chapter" and style_data["chapters"]["style"] == "none":
            raise ValidationError("chapters.style", "turn on chapter cards to preview them")
        look = tile.get("look")
        if look is not None and not isinstance(look, str):
            raise ValidationError("tile.look", "expected a look name")
        if tab == "captions" and look is not None:
            allowed = json.loads(SCHEMA_PATH.read_text())["properties"]["captions"]["properties"][
                "highlight_style"]["enum"]
            if look not in allowed:
                raise ValidationError("tile.look", "unknown caption look")
            style_data["captions"]["highlight_style"] = look
        if tab == "title" and look is not None:
            allowed = json.loads(SCHEMA_PATH.read_text())["properties"]["hook"]["properties"][
                "style"]["enum"]
            if look not in allowed:
                raise ValidationError("tile.look", "unknown title look")
            style_data["hook"]["style"] = look
        if tab == "cards" and look is not None:
            if look not in {"paper", "minimal", "dark", "bright"}:
                raise ValidationError("tile.look", "unknown card style")
            style_data["graphics"]["style"] = look
        card = tile.get("card", "text_tag")
        if (not isinstance(card, str)
                or (card not in cards.CARD_TEMPLATES and card not in self._registry_kinds())):
            raise ValidationError("tile.card", "unknown card kind")
        step = tile.get("step", "looks")
        if step not in {"looks", "start", "templates", "font", "captions", "colors", "more",
                        "words", "highlight", "title", "cards", "grade", "opening", "broll",
                        "moves", "moments", "sound"}:
            raise ValidationError("tile.step", "unknown style step")
        section = tile.get("section", "cards")
        if not isinstance(section, str) or section not in {"title", "cards"}:
            raise ValidationError("tile.section", "choose title or cards")
        showing_cards = (step in {"colors", "cards"}
                         or (step == "more" and section == "cards")
                         or (step == "font" and tile.get("section") == "cards"))
        if tab == "preview":
            card = ("text_tag" if showing_cards
                    and not cards.card_kind_off(style_data, "text_tag") else None)
        elif tab in {"captions", "title", "chapter"}:
            card = None
        with self._lock:
            preview = self.preview
        words = self._words(preview)
        face_boxes = ([Box.from_dict(box) for box in preview["face_boxes"]]
                      if "face_boxes" in preview else None)
        # Bleep changes drawn text; other sound choices have no visual layer.
        visual_style = {key: value for key, value in style_data.items()
                        if key not in {"music", "grade"}}
        bleep_on = bleep.enabled(style_data)
        bleep_words = bleep.listed(style_data["music"])
        if bleep_on or bleep_words != frozenset(bleep.builtin_words()):
            visual_style["music"] = {"bleep_swears": bleep_on,
                                     "bleep_words": sorted(bleep_words)}
        digest = hashlib.sha256(json.dumps([visual_style, [vars(w) for w in words], tile,
                                            preview["generation"], preview.get("face_boxes")],
                                          sort_keys=True).encode()).hexdigest()
        with self._lock:
            entry = self._cache.get(digest)
            if entry is not None:
                self._cache.move_to_end(digest)
        if entry is None:
            with self._prepare_lock:
                with self._lock:
                    entry = self._cache.get(digest)
                    if entry is not None:
                        self._cache.move_to_end(digest)
                if entry is None:
                    # Preparation can run for seconds; leave request and cache locks free.
                    folder = tempfile.TemporaryDirectory(prefix="layer-", dir=self.work_dir)
                    try:
                        slots = live.registry_card_slots(card, words) if card is not None else {}
                        if card == "text_tag":
                            # A title-style sign using only the creator's own words.
                            text = live.sentence_card_text(
                                words, ResolvedStyle(style_data["name"], "gallery", style_data))
                            if text is None:
                                card, slots = None, {}
                            else:
                                slots = {"text": text, "variant": "sign"}
                        if card in {"image_card", "callout_circle"}:
                            slots["image"] = str(self._card_image())
                        if card == "callout_circle":
                            # The template sample label is wider than some styles' circle.
                            slots["label"] = "Here"
                        elif card == "image_compare":
                            image = str(self._card_image())
                            slots.update(before=image, after=image,
                                         before_label="", after_label="")
                        selected = {"captions": "captions", "title": "title", "chapter": "chapter",
                                    "cards": "card"}.get(tab)
                        layers = live.live_layers(
                            ResolvedStyle(style_data["name"], "gallery", style_data), words,
                            Path(folder.name), card=card, slots=slots, only=selected,
                            show_title=(not (tab == "preview" and showing_cards)
                                        or (step == "font" and card is None)),
                            face_boxes=face_boxes,
                            sample_image=(self._card_image()
                                          if card in {"storyboard", "file_drop"} else None),
                        )
                        if style_data["captions"].get("camera_3d"):
                            layers = [layer for layer in layers if layer.name != "captions"]
                    except BaseException:
                        folder.cleanup()
                        raise
                    with self._lock:
                        self._cache[digest] = folder, layers
                        self._cache.move_to_end(digest)
                        if len(self._cache) > self._cache_limit:
                            _, (old, _) = self._cache.popitem(last=False)
                            old.cleanup()
                    entry = folder, layers
        layers = entry[1]
        return {"frame": {"width": 1080, "height": 1920}, "layers": [
            {"name": layer.name, "template": layer.template,
             "src": f"/layer/{digest}/{index}/", "x": layer.x, "y": layer.y,
             "width": layer.width, "height": layer.height, "frames": layer.frames,
             "fps": layer.fps, "duration_s": layer.frames / layer.fps,
             "poster_s": live.poster_time(layer),
             "start_s": (min(words[0].start + 0.05, words[0].end)
                         if tab == "captions" else
                         min(1.5, layer.frames / layer.fps * 0.5)
                         if tab in {"title", "chapter", "cards"} else 0)}
            for index, layer in enumerate(layers)
        ]}

    def layer_file(self, digest: str, index: int, rest: str) -> tuple[bytes, str, str]:
        with self._lock:
            entry = self._cache.get(digest)
            if entry is None or not 0 <= index < len(entry[1]):
                raise _RequestError(404, "layer not found")
            self._cache.move_to_end(digest)
            layer = entry[1][index]
        parts = (rest or "index.html").split("/")
        if len(parts) > 6 or any(
            part.startswith(".") or _ASSET.fullmatch(part) is None for part in parts
        ):
            raise _RequestError(404, "file not found")
        target = layer.composition
        for part in parts:
            target /= part
            if target.is_symlink():
                raise _RequestError(404, "file not found")
        if rest == hyperframes.GSAP_SCRIPT and self.gsap is not None:
            return self.gsap.read_bytes(), "text/javascript; charset=utf-8", ""
        if not target.is_file() or not target.resolve().is_relative_to(layer.composition.resolve()):
            raise _RequestError(404, "file not found")
        if parts == ["index.html"]:
            source = target.read_text(encoding="utf-8")
            source = live.live_html(source, layer.variables, "/hf/runtime.js")
            source = source.replace("<head>", "<head>" + _LAYER_404_SCRIPT, 1)
            return source.encode(), (
                "text/html; charset=utf-8"), _LAYER_CSP
        mime = {".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf",
                ".otf": "font/otf", ".js": "text/javascript; charset=utf-8",
                ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml", ".json": "application/json",
                ".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}.get(
                    target.suffix.lower())
        if mime is None:
            raise _RequestError(404, "file not found")
        return target.read_bytes(), mime, _LAYER_CSP if target.suffix.lower() == ".html" else ""

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        gallery = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *args: object) -> None:
                pass

            def send(self, status: int, payload: bytes, mime: str, csp: str = "") -> None:
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cache-Control", "no-store")
                if self.path.startswith("/?t="):
                    self.send_header("Set-Cookie", f"yap_gallery={gallery.token}; HttpOnly; "
                                     "SameSite=Strict; Path=/")
                if csp:
                    self.send_header("Content-Security-Policy", csp)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)

            def result(self, status: int, data: dict) -> None:
                self.send(status, (json.dumps(data, ensure_ascii=False) + "\n").encode(),
                          "application/json; charset=utf-8")

            def body(self) -> dict:
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != (
                    "application/json"
                ):
                    raise _RequestError(415, "Content-Type must be application/json")
                length = self.headers.get("Content-Length", "")
                if not length.isdecimal():
                    raise _RequestError(400, "Content-Length is required")
                if int(length) > _MAX_BODY:
                    raise _RequestError(413, "request is too large")
                try:
                    data = json.loads(self.rfile.read(int(length)))
                except (ValueError, UnicodeError):
                    raise _RequestError(400, "invalid JSON") from None
                if not isinstance(data, dict):
                    raise _RequestError(400, "expected a JSON object")
                return data

            def route(self) -> None:
                parsed = urlsplit(self.path)
                route = unquote(parsed.path)
                query = parse_qs(parsed.query, keep_blank_values=True)
                host = self.headers.get("Host", "")
                valid_hosts = {f"{gallery.host}:{gallery.port}"}
                if gallery.host == "127.0.0.1":
                    valid_hosts.add(f"localhost:{gallery.port}")
                elif gallery.host == "0.0.0.0":
                    valid_hosts.add(f"127.0.0.1:{gallery.port}")
                host_ok = host in valid_hosts
                allowed_name, separator, allowed_port = host.lower().partition(":")
                if (allowed_name in gallery.allowed_hosts
                        and (not separator or allowed_port.isdecimal())):
                    host_ok = True
                if gallery.host not in {"127.0.0.1", "localhost", "::1"}:
                    host_ok = bool(re.fullmatch(r"[A-Za-z0-9.-]+:" + str(gallery.port), host))
                if not host_ok:
                    raise _RequestError(403, "invalid Host header")
                if (route == "/api/site-colors" or self.command == "POST" or gallery.host not in {
                    "127.0.0.1", "localhost", "::1"
                } or gallery.allowed_hosts):
                    token = (self.headers.get("X-Gallery-Token") if self.command == "POST"
                             else (query.get("t", [None])[0] or next(
                                 (part.split("=", 1)[1] for part in
                                  self.headers.get("Cookie", "").split("; ")
                                  if part.startswith("yap_gallery=")), None)))
                    if token is None or not hmac.compare_digest(token, gallery.token):
                        raise _RequestError(403, "gallery token required")
                if self.command == "POST":
                    if route not in {"/api/layers", "/api/save"}:
                        raise _RequestError(404, "route not found")
                    origin = self.headers.get("Origin")
                    if origin is not None and origin not in {
                        f"http://{host}", f"https://{host}"
                    }:
                        raise _RequestError(403, "cross-origin write refused")
                    data = self.body()
                    if route == "/api/layers":
                        self.result(200, gallery.layers(data))
                    else:
                        raw = data.get("style")
                        if not isinstance(raw, dict):
                            raise _RequestError(400, "style must be an object")
                        validate_style_data(raw)
                        name = data.get("name")
                        if name is not None and not isinstance(name, str):
                            raise _RequestError(400, "name must be text")
                        name = (name if name is not None else unbranded(raw)["name"]).strip()
                        replace = data.get("replace")
                        if replace is not None and not isinstance(replace, str):
                            raise _RequestError(400, "replace must be text")
                        with gallery._lock:
                            existing = find_by_name(name)
                            replacement = next((entry for entry in list_entries()
                                                if entry.id == replace), None)
                            identical = existing is not None and (
                                unbranded({**raw, "name": existing.name})
                                == unbranded(read_style_file(existing.path)))
                            if ((replace is None and existing and not identical)
                                    or (replace is not None and (replacement is None
                                        or replacement.name.strip().lower() != name.lower()))):
                                self.result(409, {
                                    "error": "You already have a style called "
                                             f"“{existing.name if existing else name}”. "
                                             "Update it, or choose another name.",
                                    "code": "style_name_taken",
                                    "id": existing.id if existing else None,
                                })
                                return
                            if identical and replace is None:
                                replace, name = existing.id, existing.name
                            saved = save_pick(raw, name=name, workspace=gallery.workspace,
                                              replace=replace)
                            # Keep editable source colors; layers apply brand/ground contrast.
                            gallery.style = unbranded({**raw, "name": saved.name})
                            gallery.from_library = True
                            gallery.saved_at = time.monotonic()
                        workspace_copy = saved.workspace_copy
                        if workspace_copy is not None and gallery.workspace is not None:
                            try:
                                workspace_copy = workspace_copy.relative_to(gallery.workspace)
                            except ValueError:
                                pass
                        self.result(200, {"id": saved.id, "name": saved.name,
                                          "revision": saved.revision,
                                          "workspace_copy": str(workspace_copy)
                                          if workspace_copy else None})
                    return
                if self.command not in {"GET", "HEAD"}:
                    raise _RequestError(405, "method not allowed")
                if route == "/":
                    self.send(200, (_STATIC / "pick.html").read_bytes(),
                              _STATIC_TYPES["pick.html"], _PAGE_CSP)
                elif route.startswith("/static/") and route[8:] in _STATIC_TYPES:
                    name = route[8:]
                    self.send(200, (_STATIC / name).read_bytes(), _STATIC_TYPES[name])
                elif route == "/static/fonts.css":
                    names = [*fonts.BUNDLED, *(face.family for face in user_fonts.list_fonts())]
                    self.send(200, fonts.font_face_css(names, url_prefix="/fonts/").encode(),
                              "text/css; charset=utf-8")
                elif route.startswith("/fonts/"):
                    names = [*fonts.BUNDLED, *(face.family for face in user_fonts.list_fonts())]
                    source = dict(fonts.font_files(names)).get(route[7:])
                    if source is None or not source.is_file():
                        raise _RequestError(404, "font not found")
                    mime = {".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf",
                            ".otf": "font/otf"}[source.suffix.lower()]
                    self.send(200, source.read_bytes(), mime)
                elif route == "/api/state":
                    self.result(200, gallery.state())
                elif route == "/api/preview":
                    self.result(200, gallery.state()["preview"])
                elif route == "/api/grade-color":
                    if set(query) - {"color", "t"} or any(len(v) != 1 for v in query.values()):
                        raise ValidationError("grade.color", "use color once")
                    self.result(200, gallery.grade_color(query.get("color", [None])[0]))
                elif route == "/api/site-colors":
                    origin = self.headers.get("Origin")
                    if origin is not None and origin not in {
                        f"http://{host}", f"https://{host}"
                    }:
                        raise _RequestError(403, "cross-origin write refused")
                    try:
                        self.result(200, gallery.website_colors(query.get("url", [""])[0]))
                    except FetchError as exc:
                        self.result(400, {"error": str(exc)})
                elif route == "/still":
                    body, mime = gallery.still_image(query)
                    self.send(200, body, mime)
                elif route == "/cutout":
                    self.send(200, gallery.cutout_image(query), "image/png")
                elif route == "/hf/runtime.js":
                    self.send(200, gallery.runtime.read_bytes(), "text/javascript; charset=utf-8")
                elif (match := _LAYER.fullmatch(route)) is not None:
                    body, mime, csp = gallery.layer_file(match[1], int(match[2]), match[3])
                    self.send(200, body, mime, csp)
                else:
                    raise _RequestError(404, "file not found")

            def dispatch(self) -> None:
                gallery.last_request = time.monotonic()
                try:
                    try:
                        self.route()
                    except _RequestError as exc:
                        self.result(exc.status, {"error": exc.message})
                    except YapError as exc:
                        self.result(400, {"error": exc.message})
                    except (OSError, ValueError, TypeError):
                        self.result(500, {"error": "internal server error"})
                except (BrokenPipeError, ConnectionResetError):
                    # Browsers cancel stale tile loads when a tab changes.
                    pass

            def do_GET(self) -> None:
                self.dispatch()

            def do_HEAD(self) -> None:
                self.dispatch()

            def do_POST(self) -> None:
                self.dispatch()

        return Handler
