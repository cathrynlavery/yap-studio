"""Ask, answer, and list durable questions attached to feedback items."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path

from yap_studio.errors import ValidationError
from yap_studio.feedback.locate import Location
from yap_studio.project.lock import project_lock
from yap_studio.review.model import Comment


class UnknownQuestionError(ValidationError):
    """Report a missing batch or comment reference."""

    def __init__(self) -> None:
        super().__init__("comment", "comment reference is not in this batch")


class CarriedQuestionError(ValidationError):
    """Refuse an answer that has already been sent in another batch."""

    def __init__(self) -> None:
        super().__init__("clarification", "this answer has already been carried")


def _clarification(question: str, options: list[dict], reason: str) -> dict:
    return {"question": question, "options": copy.deepcopy(options), "reason": reason,
            "answer": None, "option": None, "answered_at": None, "carried_to": None}


def automatic_clarification(
    comment: Comment, version: str | None, location: Location | None,
) -> dict | None:
    """Ask for a missing moment or an ambiguous fix scope before making changes."""
    if comment.kind == "keep":
        return None
    if location is None or location.status == "outside":
        return _clarification(
            f"I could not find this moment in {version or 'the current cut'}. "
            "Which part do you mean?", [], "unlocated",
        )
    if comment.target is not None:
        return None
    options = []
    if location.words:
        options.append({"value": "story", "label": "The words"})
    if location.layers_known:
        caption = next((layer for layer in location.layers if layer.kind == "caption"), None)
        element = next((layer for layer in location.layers if layer.kind != "caption"), None)
        if caption is not None:
            label = f'The caption "{caption.text}"' if caption.text else "The caption"
            options.append({"value": "caption", "label": label})
        if element is not None:
            options.append({"value": "element", "label": f"The {element.kind} {element.id}"})
    return (_clarification("Which do you want changed here?", options, "ambiguous_target")
            if len(options) > 1 else None)


def _item(root: Path, batch_id: str, version: str | None, comment_id: str) -> tuple[dict, dict]:
    from yap_studio.feedback.batch import _batch_path, load_batch

    if not _batch_path(root, batch_id).is_file():
        raise UnknownQuestionError()
    batch = load_batch(root, batch_id)
    item = next((item for item in batch["items"]
                 if (item["comment"]["version"], item["comment"]["id"]) == (version, comment_id)),
                None)
    if item is None:
        raise UnknownQuestionError()
    return batch, item


def ask_clarification(
    project_dir: Path, batch_id: str, comment_version: str | None, comment_id: str,
    question: str, *, options: list[dict] | None = None,
) -> dict:
    """Attach an agent's question while refusing items with recorded operations."""
    from yap_studio.feedback.batch import _write_batch

    if (not isinstance(question, str) or not 1 <= len(question) <= 500
            or not question.strip()):
        raise ValidationError("question", "must contain 1 to 500 characters")
    options = [] if options is None else options
    values = set()
    if not isinstance(options, list):
        raise ValidationError("options", "must be an array")
    for option in options:
        if (not isinstance(option, dict) or set(option) != {"value", "label"}
                or any(not isinstance(option[key], str) or not option[key].strip()
                       for key in ("value", "label")) or option["value"] in values):
            raise ValidationError("options", "must have distinct values and nonempty labels")
        values.add(option["value"])
    root = Path(project_dir).resolve()
    with project_lock(root):
        batch, item = _item(root, batch_id, comment_version, comment_id)
        if (item.get("clarification") or {}).get("carried_to") is not None:
            raise CarriedQuestionError()
        if item["operations"]:
            raise ValidationError("operations", "cannot ask after operations were recorded")
        item["disposition"] = "needs_clarification"
        item["clarification"] = _clarification(question.strip(), options, "agent")
        item["constraint"] = None
        batch["constraints"] = [constraint for constraint in batch["constraints"]
                                if constraint["comment"] != {
                                    "version": comment_version, "id": comment_id,
                                    "revision": item["comment"]["revision"],
                                }]
        _write_batch(root, batch)
        return batch


def answer_clarification(
    project_dir: Path, batch_id: str, comment_version: str | None, comment_id: str,
    *, answer: str | None = None, option: str | None = None,
) -> dict:
    """Save one answer immediately under the project lock."""
    from yap_studio.feedback.batch import _write_batch

    if answer is not None:
        if not isinstance(answer, str) or not 1 <= len(answer.strip()) <= 2000:
            raise ValidationError("answer", "must contain 1 to 2000 characters")
        answer = answer.strip()
    if answer is None and option is None:
        raise ValidationError("answer", "an answer or option is required")
    root = Path(project_dir).resolve()
    with project_lock(root):
        batch, item = _item(root, batch_id, comment_version, comment_id)
        clarification = item.get("clarification")
        if clarification is not None and clarification["carried_to"] is not None:
            raise CarriedQuestionError()
        if (clarification is None or clarification["answered_at"] is not None
                or item["disposition"] != "needs_clarification"):
            raise ValidationError("clarification", "this item has no open clarification")
        if option is not None and (
            not isinstance(option, str)
            or option not in {item["value"] for item in clarification["options"]}
        ):
            raise ValidationError("option", "must be one of this question's option values")
        clarification.update(answer=answer, option=option,
                             answered_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
        _write_batch(root, batch)
        return batch


def _questions(project_dir: Path, *, answered: bool) -> list[dict]:
    from yap_studio.feedback.batch import _batch_numbers, load_batch

    root = Path(project_dir)
    questions = []
    for number in sorted(_batch_numbers(root)):
        batch = load_batch(root, f"b{number}")
        for item in batch["items"]:
            clarification = item.get("clarification")
            if (clarification is None or clarification["carried_to"] is not None
                    or (clarification["answered_at"] is not None) != answered):
                continue
            question = {"batch": batch["id"], "comment": {
                "version": item["comment"]["version"], "id": item["comment"]["id"],
            }, **{key: copy.deepcopy(clarification[key]) for key in
                  ("question", "options", "reason")}}
            if answered:
                question.update({key: clarification[key] for key in
                                 ("answer", "option", "answered_at")})
            questions.append(question)
    return questions


def pending_questions(project_dir: Path) -> list[dict]:
    """List questions that still need an answer."""
    return _questions(project_dir, answered=False)


def answered_questions(project_dir: Path) -> list[dict]:
    """List saved answers waiting to be carried into a batch."""
    return _questions(project_dir, answered=True)
