"""Public creator taste validation, persistence, and application API."""

from __future__ import annotations

from yap_studio.taste.check import TasteCheck, check_taste
from yap_studio.taste.inbox import UnknownProposalError, decide, inbox_view, propose
from yap_studio.taste.learn import LearnResult, learn, measure_style
from yap_studio.taste.model import (
    PROPOSALS_SCHEMA,
    SNAPSHOT_SCHEMA,
    TASTE_SCHEMA,
    AddResult,
    ApproveResult,
    Proposals,
    Taste,
    TasteError,
    TasteRefusal,
    TasteSnapshot,
    rule_id,
    snapshot_at,
    validate_body,
    validate_proposals,
    validate_style_field,
    validate_taste,
)
from yap_studio.taste.render import read_taste_record, render_style, write_taste_record
from yap_studio.taste.resolve import TasteApplication, apply_taste
from yap_studio.taste.store import (
    add_proposals,
    approve,
    approve_proposal,
    load_proposals,
    load_taste,
    proposals_path,
    reject,
    reject_proposal,
    revoke,
    taste_path,
)

__all__ = [
    "PROPOSALS_SCHEMA",
    "SNAPSHOT_SCHEMA",
    "TASTE_SCHEMA",
    "AddResult",
    "ApproveResult",
    "LearnResult",
    "Proposals",
    "Taste",
    "TasteApplication",
    "TasteCheck",
    "TasteError",
    "TasteRefusal",
    "TasteSnapshot",
    "UnknownProposalError",
    "add_proposals",
    "apply_taste",
    "approve",
    "approve_proposal",
    "check_taste",
    "decide",
    "inbox_view",
    "learn",
    "load_proposals",
    "load_taste",
    "measure_style",
    "proposals_path",
    "propose",
    "read_taste_record",
    "reject",
    "reject_proposal",
    "render_style",
    "revoke",
    "rule_id",
    "snapshot_at",
    "taste_path",
    "validate_body",
    "validate_proposals",
    "validate_style_field",
    "validate_taste",
    "write_taste_record",
]
