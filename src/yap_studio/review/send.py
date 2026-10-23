"""Run one durable round through the project feedback batch builder."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from yap_studio.errors import ValidationError, YapError
from yap_studio.feedback.batch import FeedbackError, build_batch
from yap_studio.feedback.spelling import apply_word_fixes
from yap_studio.feedback.taste import ProposalInbox
from yap_studio.feedback.versions import current_version
from yap_studio.review.export import series_export, to_export_json
from yap_studio.review.model import conform_comment

if TYPE_CHECKING:
    from yap_studio.review.server import ReviewSeries, ReviewVersion
    from yap_studio.review.store import ReviewStore


def send_round(
    store: ReviewStore, series: ReviewSeries, version: ReviewVersion, round_id: str,
    *, now: datetime | None = None, instance: str | None = None,
) -> None:
    """Update persisted progress and results, containing every sending failure."""
    batch_id = None
    applying = False
    try:
        round = store.update_round(series.id, round_id, instance=instance)
        batch_id = round.batch
        if series.project_dir is None:
            raise ValidationError(
                "project", "Open the project folder with --versions to send notes.",
            )
        if version.number != round.version:
            raise ValidationError("version", "does not match the round's version")
        current = current_version(series.project_dir)
        if current is not None and version.version_id != current.id:
            raise FeedbackError(
                f"{version.version_id} is not the current version ({current.id}).",
                suggested_action="Review the current saved version.",
            )
        if batch_id is None:
            store.update_round(series.id, round_id, step="Locating notes", instance=instance)
            facts = version.media.facts(version.sha256)
            saved = {item.id: item for item in store.comments(version.sha256)}
            comments = []
            for ref in round.comments:
                comment = saved.get(ref["id"])
                if comment is None or comment.updated_at != ref["revision"]:
                    raise ValidationError(
                        "comments", "These notes changed. Send the edited drafts again.",
                    )
                comments.append(conform_comment(comment, facts))
            export = to_export_json(facts, comments, store.reviewer(), now or datetime.now(UTC))
            export["series"] = series_export(store, series, version.number)
            store.update_round(series.id, round_id, step="Writing the batch", instance=instance)
            batch_id = build_batch(series.project_dir, export,
                                   origin={"series": series.id, "round": round.id},
                                   answers=round.answers)["id"]
        applying = True
        store.update_round(series.id, round_id, batch=batch_id, step="Applying spelling fixes",
                           instance=instance)
        apply_word_fixes(series.project_dir, batch_id, sink=ProposalInbox(series.project_dir))
        store.update_round(series.id, round_id, status="sent", batch=batch_id, step=None,
                           instance=instance)
    except Exception as exc:
        message = (str(exc).strip() if applying and str(exc).strip()
                   else "Could not send these notes. Try again.")
        error = ({"code": exc.code, "message": exc.message} if isinstance(exc, YapError) else
                 {"code": "internal_error", "message": message})
        try:
            store.update_round(series.id, round_id, status="failed", step=None, batch=batch_id,
                               error=error, instance=instance)
        except Exception:
            # A storage failure leaves sending recoverable at the next startup.
            pass
