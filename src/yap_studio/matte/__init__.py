"""Person matte documents, streaming segmentation, and card compositing."""

from __future__ import annotations

from yap_studio.matte.check import CheckResult, CutoutReport, CutoutThresholds, check_cutout
from yap_studio.matte.composite import (
    DEFAULT_PLACEMENT,
    PLACEMENTS,
    CardOverlay,
    CompositeResult,
    Placement,
    composite_graph,
    composite_video,
)
from yap_studio.matte.model import MATTE_SCHEMA, Cleanup, MatteData, load_matte

__all__ = [
    "MATTE_SCHEMA", "Cleanup", "MatteData", "load_matte", "Placement", "PLACEMENTS",
    "DEFAULT_PLACEMENT", "CardOverlay", "CompositeResult", "composite_graph", "composite_video",
    "CheckResult", "CutoutReport", "CutoutThresholds", "check_cutout",
]
