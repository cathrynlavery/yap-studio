"""Validate setup answers and map them to style, project and taste settings."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from yap_studio.checks.platform import load_presets
from yap_studio.errors import ValidationError
from yap_studio.style import loader

ANSWERS_SCHEMA = "yap-studio/style-answers@1"
_PLATFORMS = ("tiktok", "reels", "shorts", "youtube")
_CUTS = {"tight": (0.2, 0.3), "natural": (0.3, 0.4), "relaxed": (0.45, 0.6)}
_MUSIC = {"none": 0, "quiet": 0.04, "upbeat": 0.09}


def _choices(*items: tuple[str, str, bool]) -> list[dict[str, Any]]:
    return [{"value": value, "label": label, "recommended": recommended}
            for value, label, recommended in items]


QUESTIONS = [
    {"id": "finished_videos", "prompt": "Finished videos you like?",
     "answer_key": "finished_videos",
     "kind": "videos", "choices": _choices(("yes", "Yes, pick a folder", True),
                                           ("not_yet", "Not yet", False))},
    {"id": "brand", "prompt": "What is your brand?", "answer_key": "brand", "kind": "brand",
     "choices": _choices(("gallery", "Pick in the next step", True),
                         ("website", "Paste my website", False))},
    {"id": "about", "prompt": "What are your videos about, and who are they for? "
     "Type your answer, or skip for now.",
     "answer_key": "about", "kind": "text", "choices": []},
    {"id": "platforms", "prompt": "Where do you post?", "answer_key": "platforms",
     "kind": "multi", "choices": _choices(("tiktok", "TikTok", True),
                                            ("reels", "Reels", True),
                                            ("shorts", "Shorts", True),
                                            ("youtube", "YouTube", False))},
    {"id": "cuts", "prompt": "How tight are your cuts?", "answer_key": "cuts",
     "kind": "choice", "choices": _choices(("natural", "Natural", True),
                                              ("tight", "Tight", False),
                                              ("relaxed", "Relaxed", False))},
    {"id": "music", "prompt": "Music under your voice?", "answer_key": "music",
     "kind": "choice", "choices": _choices(("none", "None", True),
                                              ("quiet", "Quiet", False),
                                              ("upbeat", "Upbeat", False))},
    {"id": "word_fixes", "prompt": "Words the transcript gets wrong? "
     "Type them as heard -> correct, or skip for now.",
     "answer_key": "word_fixes", "kind": "list", "choices": []},
    {"id": "never", "prompt": "What should never happen? Type any additional rules.",
     "answer_key": "never",
     "kind": "toggles", "choices": _choices(("cover_face", "Never cover my face", True),
                                               ("emoji", "No emoji", False),
                                               ("all_caps", "No all-caps", False))},
]


def _object(value: Any, allowed: set[str], location: str) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(location, "expected object")
    for key in value:
        if key not in allowed:
            raise ValidationError(f"{location}.{key}", "unknown field")
    return value


def _string(value: Any, location: str, *, maximum: int | None = None) -> None:
    if (not isinstance(value, str) or (maximum is not None and len(value) > maximum)
            or not value.strip()):
        limit = f" of at most {maximum} characters" if maximum is not None else ""
        raise ValidationError(location, f"expected a non-empty string{limit}")
    if value != value.strip() or "\n" in value or "\r" in value:
        raise ValidationError(location, "use one trimmed line")


def _choice(value: Any, options: tuple[str, ...], location: str) -> None:
    if not isinstance(value, str) or value not in options:
        raise ValidationError(location, f"choose one of {', '.join(options)}")


def _nullable_string(data: dict, key: str, location: str) -> None:
    if key in data and data[key] is not None and not isinstance(data[key], str):
        raise ValidationError(f"{location}.{key}", "expected string or null")


def _strings(value: Any, location: str, *, limit: int | None = None,
             maximum: int | None = None) -> None:
    if not isinstance(value, list) or (limit is not None and len(value) > limit):
        detail = f"expected up to {limit} items" if limit is not None else "expected array"
        raise ValidationError(location, detail)
    for index, item in enumerate(value):
        _string(item, f"{location}[{index}]", maximum=maximum)


def validate_answers(value: Any) -> dict[str, Any]:
    """Return a validated copy with precise locations for invalid fields."""
    data = copy.deepcopy(_object(value, {"schema", *(q["answer_key"] for q in QUESTIONS)},
                                 "answers"))
    if data.get("schema") != ANSWERS_SCHEMA:
        raise ValidationError("answers.schema", f"must be {ANSWERS_SCHEMA!r}")
    if "finished_videos" in data:
        videos = _object(data["finished_videos"], {"answer", "videos", "learned_style"},
                         "answers.finished_videos")
        if "answer" not in videos:
            raise ValidationError("answers.finished_videos.answer", "missing required field")
        _choice(videos["answer"], ("yes", "not_yet"), "answers.finished_videos.answer")
        if "videos" in videos:
            if not isinstance(videos["videos"], list):
                raise ValidationError("answers.finished_videos.videos", "expected array")
            for index, video in enumerate(videos["videos"]):
                if not isinstance(video, str):
                    raise ValidationError(
                        f"answers.finished_videos.videos[{index}]", "expected string"
                    )
        _nullable_string(videos, "learned_style", "answers.finished_videos")
    if "brand" in data:
        brand = _object(data["brand"], {"website", "accent_color", "captions_font",
                                         "titles_font", "cards_font", "pick_in_gallery"},
                        "answers.brand")
        for key in ("website", "captions_font", "titles_font", "cards_font"):
            _nullable_string(brand, key, "answers.brand")
        if "accent_color" in brand and brand["accent_color"] is not None:
            color = brand["accent_color"]
            if not isinstance(color, str) or (color != "" and
                    re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is None):
                raise ValidationError("answers.brand.accent_color",
                                      "expected an empty string or a color like #1A2B3C")
        if "pick_in_gallery" in brand and not isinstance(brand["pick_in_gallery"], bool):
            raise ValidationError("answers.brand.pick_in_gallery", "expected boolean")
    if "about" in data:
        about = data["about"]
        if not isinstance(about, str) or len(about) > 200 or any(c in about for c in "\r\n"):
            raise ValidationError("answers.about", "expected one line of at most 200 characters")
    if "platforms" in data:
        platforms = data["platforms"]
        if not isinstance(platforms, list) or len(platforms) > len(_PLATFORMS):
            raise ValidationError("answers.platforms", "expected up to four platforms")
        for index, platform in enumerate(platforms):
            _choice(platform, _PLATFORMS, f"answers.platforms[{index}]")
            if platform in platforms[:index]:
                raise ValidationError(f"answers.platforms[{index}]", "duplicate platform")
    for key, options in (("cuts", tuple(_CUTS)), ("music", tuple(_MUSIC))):
        if key in data:
            _choice(data[key], options, f"answers.{key}")
    if "word_fixes" in data:
        fixes = _object(data["word_fixes"], set(data["word_fixes"])
                        if isinstance(data["word_fixes"], dict) else set(), "answers.word_fixes")
        if len(fixes) > 100:
            raise ValidationError("answers.word_fixes", "expected up to 100 entries")
        for heard, correct in fixes.items():
            _string(heard, "answers.word_fixes key")
            _string(correct, f"answers.word_fixes.{heard}")
    if "never" in data:
        never = _object(data["never"], {"cover_face", "emoji", "all_caps", "other"},
                        "answers.never")
        for key in ("cover_face", "emoji", "all_caps"):
            if key in never and not isinstance(never[key], bool):
                raise ValidationError(f"answers.never.{key}", "expected boolean")
        if "other" in never:
            if isinstance(never["other"], str):
                never["other"] = [never["other"]] if never["other"] else []
            _strings(never["other"], "answers.never.other", limit=10, maximum=200)
    return data


def load_answers(path: str | Path) -> dict[str, Any]:
    """Read an answers document and validate it before use."""
    file = Path(path).expanduser()
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValidationError(str(file), f"cannot read answers JSON: {exc}") from exc
    return validate_answers(data)


def questions(answers: dict | None = None) -> list[dict[str, Any]]:
    """Return ordered questions with any saved answer beside each prompt."""
    saved = validate_answers(answers) if answers is not None else {}
    return [{**copy.deepcopy(row), "current": copy.deepcopy(saved.get(row["answer_key"]))}
            for row in QUESTIONS]


def style_changes(answers: dict, base_data: dict) -> dict[str, Any]:
    """Return style edits and any font choices that need attention."""
    from yap_studio.render import fonts, user_fonts

    data = validate_answers(answers)
    changes: dict[str, Any] = {}
    never = data.get("never", {})
    if "cover_face" in never:
        changes["captions"] = {
            "placement": "avoid_face" if never["cover_face"] else "fixed",
        }
    if "music" in data:
        changes["music"] = {"level": _MUSIC[data["music"]]}
    brand_answers = data.get("brand", {})
    properties = loader._schema()["properties"]["brand"]["properties"]
    brand = {}
    pending = []
    for key in ("captions_font", "titles_font", "cards_font"):
        value = brand_answers.get(key)
        if value is None:
            continue
        if value == "" or (len(value) <= properties[key]["maxLength"]
                and (value in fonts.BUNDLED or user_fonts.find(value))):
            brand[key] = value
        else:
            pending.append(
                f"Font {value!r} for {key.replace('_', ' ')} is not available. "
                "Choose a bundled font or add it with yap-studio brand fonts add."
            )
    if brand_answers.get("accent_color") is not None:
        brand["accent_color"] = brand_answers["accent_color"]
    if brand:
        changes["brand"] = brand
    if pending:
        changes["pending"] = {"brand": " ".join(pending)}
    return changes


def project_defaults(answers: dict) -> dict[str, Any]:
    """Map answered project preferences onto project.json fields."""
    data = validate_answers(answers)
    defaults: dict[str, Any] = {}
    if "cuts" in data:
        pause, breath = _CUTS[data["cuts"]]
        defaults["rules"] = {"max_pause_s": pause, "beat_breath_s": breath}
    if "platforms" in data:
        vertical = [item for item in data["platforms"] if item != "youtube"]
        if vertical:
            presets = load_presets()
            defaults["platform"] = min(
                vertical, key=lambda item: presets[item]["duration"]["max_s"]
            )
    if "music" in data:
        defaults["music"] = ({"mode": "none"} if data["music"] == "none" else
                             {"mode": "auto", "tags": [data["music"]]})
    if "word_fixes" in data:
        defaults["word_fixes"] = data["word_fixes"]
    return defaults


def taste_rules(answers: dict) -> list[dict[str, Any]]:
    """Build approved rule bodies from explicit never answers."""
    never = validate_answers(answers).get("never", {})
    rows: list[dict[str, Any]] = []

    def add(scope: str, sentence: str, machine: dict | None, answer: str) -> None:
        rows.append({"scope": scope, "condition": None, "sentence": sentence,
                     "machine": machine, "sources": [{"type": "answer", "question": "never",
                                                    "answer": answer}], "context": None})

    if never.get("cover_face"):
        for layer in ("captions", "title", "cards"):
            add("captions" if layer == "captions" else "cards",
                f"Keep {layer} clear of my face.",
                {"kind": "placement", "layer": layer, "rule": "clear_of_face"},
                "Never cover my face")
    if never.get("emoji"):
        add("cards", "No emoji on screen.", None, "No emoji")
    if never.get("all_caps"):
        add("captions", "No all-caps words on screen.", None, "No all-caps")
    for sentence in never.get("other", []):
        add("cut", sentence, None, sentence)
    return rows


def apply_answers(path: str | Path, workspace: str | Path | None = None) -> dict[str, Any]:
    """Apply answered style and taste choices, returning the setup summary."""
    from yap_studio.style.library import get_default, save_pick
    from yap_studio.style.site_colors import FetchError, site_colors
    from yap_studio.taste.model import rule_id
    from yap_studio.taste.store import approve_proposal, load_taste, revoke

    answers = load_answers(path)
    base = loader.resolve().unbranded_data
    changes = style_changes(answers, base)
    pending = changes.pop("pending", {})
    brand_answers = answers.get("brand", {})
    website_colors = None
    website = brand_answers.get("website")
    if website and brand_answers.get("accent_color") is None:
        try:
            website_colors = site_colors(website)
            accent = website_colors["suggestion"]["accent"]
            website_colors = {**website_colors, "used": accent}
            if accent:
                changes.setdefault("brand", {})["accent_color"] = accent
            else:
                note = "No accent color found on that site. Pick an accent in the gallery."
                pending["brand"] = f"{pending.get('brand', '')} {note}".strip()
        except FetchError as exc:
            note = f"Could not read colors from {website}: {exc} Pick an accent in the gallery."
            pending["brand"] = f"{pending.get('brand', '')} {note}".strip()
    if brand_answers.get("pick_in_gallery") and not changes.get("brand") and "brand" not in pending:
        pending["brand"] = "Pick your brand in the gallery."
    fields = [f"{section}.{key}" for section, values in changes.items()
              for key, value in values.items() if base.get(section, {}).get(key) != value]
    default = get_default()
    style_id = default if default and not default.startswith("preset:") else None
    if style_id is None and website_colors and website_colors["used"]:
        from yap_studio import paths

        answers["brand"]["accent_color"] = website_colors["used"]
        with paths.operation():
            paths.atomic_write_text(Path(path).expanduser(),
                                    json.dumps(answers, indent=2, ensure_ascii=False) + "\n")
    if fields and style_id is not None:
        data = copy.deepcopy(base)
        for section, values in changes.items():
            data[section] = {**data.get(section, {}), **values}
        style_id = save_pick(data, name=base["name"], workspace=workspace,
                             replace=style_id).id

    wanted = taste_rules(answers)
    wanted_ids = {rule_id(body) for body in wanted}
    revoked = []
    if "never" in answers:
        for rule in load_taste().rules:
            if (rule["status"] == "approved" and rule["id"] not in wanted_ids
                    and rule["sources"]
                    and all(source["type"] == "answer" and source["question"] == "never"
                            for source in rule["sources"])):
                revoke(rule["id"])
                revoked.append(rule["id"])
    approved = []
    for body in wanted:
        identity = rule_id(body)
        existing = load_taste().rules
        machine = body["machine"]
        if machine is not None and any(
            rule["status"] == "approved" and rule["id"] != identity
            and rule["machine"] == machine and rule["condition"] == body["condition"]
            and any(source["type"] != "answer" for source in rule["sources"])
            for rule in existing
        ):
            pending["taste"] = "A review-backed rule already covers one setup choice."
            continue
        if not any(rule["id"] == identity and rule["status"] == "approved"
                   for rule in existing):
            approve_proposal(body)
            approved.append(identity)
    finished = answers.get("finished_videos", {})
    if finished.get("videos") and not finished.get("learned_style"):
        pending["finished_videos"] = (
            "Run yap-studio taste learn VIDEO... --save and record learned_style."
        )
    if answers.get("platforms") == ["youtube"]:
        pending["platform"] = (
            "YouTube alone has no vertical project preset; platform omitted."
        )
    return {"style_id": style_id, "style_saved": style_id is not None,
            "fields_set": fields, "rules_approved": approved,
            "rules_revoked": revoked, "project_defaults": project_defaults(answers),
            "pending": pending, "website_colors": website_colors}
