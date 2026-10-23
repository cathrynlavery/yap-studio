"""Match retained speech with one static gain per take at cut time."""

from __future__ import annotations

import copy
import math
import struct
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.checks.structure import _loudness
from yap_studio.project.model import Project

# Bump when measurement or gain rules change.
TAKE_LEVEL_RULE_VERSION = 1
GAIN_LIMIT_DB = 12.0
PEAK_CEILING_DBFS = -1.0
SAMPLE_RATE = 48_000


def rule_identity() -> dict[str, float]:
    """Identify the measurement and gain rules used by the cut."""
    return {
        "TAKE_LEVEL_RULE_VERSION": TAKE_LEVEL_RULE_VERSION,
        "GAIN_LIMIT_DB": GAIN_LIMIT_DB,
        "PEAK_CEILING_DBFS": PEAK_CEILING_DBFS,
    }


def match_enabled(project: Project) -> bool:
    """Match takes unless the project asks to keep their recorded levels."""
    return project.audio.get("match_takes", True)


def word_audio(sound: np.ndarray, spans: list[tuple[Fraction, Fraction]],
               start: Fraction, end: Fraction) -> np.ndarray:
    """Keep word windows inside one piece without counting overlaps twice."""
    windows = []
    for first, last in spans:
        first, last = max(start, first), min(end, last)
        if last <= first:
            continue
        low = min(len(sound), max(0, round((first - start) * SAMPLE_RATE)))
        high = min(len(sound), max(0, round((last - start) * SAMPLE_RATE)))
        if high > low:
            windows.append((low, high))
    merged: list[tuple[int, int]] = []
    for first, last in sorted(windows):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return np.concatenate([sound[first:last] for first, last in merged]) if merged else sound[:0]


def _peak(sound: np.ndarray) -> float | None:
    """Measure the sample peak, including sound outside word windows."""
    peak = float(np.max(np.abs(sound), initial=0))
    return 20 * math.log10(peak) if peak else None


def measure(sound: np.ndarray, work: Path, name: str) -> tuple[float | None, float | None]:
    """Measure gated loudness and sample peak over the supplied PCM."""
    frames = np.asarray(sound, dtype="<f4")
    channels = frames.shape[1]
    target = work / f"take-level-{name}.wav"
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + frames.nbytes, b"WAVE", b"fmt ", 16,
        3, channels, SAMPLE_RATE, SAMPLE_RATE * channels * 4, channels * 4, 32,
        b"data", frames.nbytes,
    )
    with target.open("wb") as output:
        output.write(header)
        output.write(frames.tobytes())
    result = ffmpeg.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-i", str(target),
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        timeout_s=max(60.0, 30.0 + 6.0 * len(frames) / SAMPLE_RATE),
    )
    lufs, _, _ = _loudness(result.stderr)
    # An empty ebur128 gate reports its -70 LUFS floor.
    if lufs is not None and lufs <= -70:
        lufs = None
    return lufs, _peak(frames)


def measure_take(sound: np.ndarray, speech: np.ndarray, work: Path, name: str) -> dict[str, Any]:
    """Prefer retained words for loudness and always use kept PCM for peak."""
    lufs = None
    if len(speech) >= round(0.4 * SAMPLE_RATE):
        lufs, _ = measure(speech, work, f"{name}-words")
    basis = "words" if lufs is not None else "kept"
    if basis == "kept":
        lufs, _ = measure(sound, work, f"{name}-kept")
    return {"seconds": len(sound) / SAMPLE_RATE, "lufs": lufs,
            "peak_dbfs": _peak(sound), "basis": basis}


def plan(takes: dict[str, dict[str, Any]], *, enabled: bool) -> dict[str, Any]:
    """Plan bounded gains in cut order and keep the measurements in a receipt."""
    reference = next((take for take, row in takes.items() if row["lufs"] is not None), None)
    reference_lufs = takes[reference]["lufs"] if reference is not None else None
    rows = {}
    for take, measurement in takes.items():
        lufs = measurement["lufs"]
        needed = reference_lufs - lufs if reference_lufs is not None and lufs is not None else None
        gain = max(-GAIN_LIMIT_DB, min(GAIN_LIMIT_DB, needed)) if enabled and needed else 0.0
        rows[take] = {
            **measurement,
            "gain_db": gain,
            "needed_db": needed,
            "limited": "gain" if enabled and needed is not None and abs(needed) > GAIN_LIMIT_DB
            else None,
        }
    raised = [row for row in rows.values() if row["gain_db"] > 0 and row["peak_dbfs"] is not None]
    excess = max((row["peak_dbfs"] + row["gain_db"] - PEAK_CEILING_DBFS for row in raised),
                 default=0.0)
    room = min((row["gain_db"] + GAIN_LIMIT_DB for row in rows.values()), default=0.0)
    for row in raised:
        if row["peak_dbfs"] > PEAK_CEILING_DBFS:
            room = min(room, row["gain_db"])
    headroom = -min(max(0.0, excess), room) or 0.0
    for row in rows.values():
        row["gain_db"] += headroom
    # Keep peak checks for takes that reached zero gain during the shared shift.
    for row in raised:
        if row["peak_dbfs"] + row["gain_db"] > PEAK_CEILING_DBFS + 1e-9:
            gain = min(
                row["gain_db"], max(0.0, PEAK_CEILING_DBFS - row["peak_dbfs"]),
            )
            if row["gain_db"] - gain > 1e-9:
                row["limited"] = "peak"
            row["gain_db"] = gain
    for row in rows.values():
        row["gain_db"] = round(row["gain_db"], 1) or 0.0
        while (row["gain_db"] > 0 and row["peak_dbfs"] is not None
               and row["peak_dbfs"] + row["gain_db"] > PEAK_CEILING_DBFS + 1e-9):
            row["gain_db"] = round(row["gain_db"] - 0.1, 1) or 0.0
        for key in ("seconds", "lufs", "peak_dbfs", "needed_db"):
            if row[key] is not None:
                row[key] = round(row[key], 2) or 0.0
    return {"rule": rule_identity(), "match_takes": enabled, "reference": reference,
            "headroom_db": round(headroom, 2) or 0.0, "takes": rows}


def warnings(record: dict) -> list[str]:
    """Ask for a recording check when gain or peak limits prevent matching."""
    result = []
    for take, row in record["takes"].items():
        if row["limited"] == "peak":
            reference = record["reference"]
            reference_gain = record["takes"][reference]["gain_db"]
            remaining = row["needed_db"] - (row["gain_db"] - reference_gain)
            result.append(
                f"Take {take} stays {abs(remaining):.1f} LU "
                f"{'quieter' if remaining > 0 else 'louder'} than take {reference}, "
                "because raising it more would clip. "
                "Check that it is the right recording, or record it again."
            )
        elif row["limited"] == "gain":
            result.append(
                f"Take {take} is {abs(row['needed_db']):.1f} LU "
                f"{'quieter' if row['needed_db'] > 0 else 'louder'} "
                f"than take {record['reference']}; "
                f"the cut changed it by the {GAIN_LIMIT_DB:g} dB limit. "
                "Check that it is the right recording, or record it again."
            )
    return result


def explain_part_jumps(report: dict, record: dict | None) -> dict:
    """Explain remaining jumps using the take levels measured by the cut."""
    if not record:
        return report
    result = copy.deepcopy(report)
    changed = False
    for finding in [*result.get("findings", []),
                    *result.get("structure", {}).get("findings", [])]:
        if finding["check"] != "loudness.parts":
            continue
        evidence = finding.get("evidence", {})
        takes = [evidence.get("left_source"), evidence.get("right_source")]
        rows = record["takes"]
        if any(take not in rows for take in takes):
            continue
        limit = record["rule"]["GAIN_LIMIT_DB"]
        limited = next((take for take in takes if rows[take]["limited"]), None)
        unmeasured = next((take for take in takes if rows[take]["lufs"] is None), None)
        if not record["match_takes"]:
            action = ('The takes keep their recorded levels because "match_takes" is false '
                      'in the project audio settings. Remove it and cut again '
                      'to even out the takes.')
        elif limited is not None:
            if rows[limited]["limited"] == "peak":
                action = (f"Take {limited} could only be raised part of the way to match take "
                          f"{record['reference']} before it would clip. "
                          "Check that it is the right recording, or record it again.")
            else:
                action = (f"Take {limited} needed more than the {limit:g} dB limit to match take "
                          f"{record['reference']}. Check that it is the right recording, "
                          "or record it again.")
        elif unmeasured is not None:
            action = (f"Take {unmeasured} has too little speech to measure, "
                      "so it keeps its recorded level.")
        else:
            deviations = [evidence[f"{side}_lufs"] - (rows[take]["lufs"] + rows[take]["gain_db"])
                          for side, take in zip(("left", "right"), takes, strict=True)]
            side = max(range(2), key=lambda index: abs(deviations[index]))
            deviation = deviations[side]
            action = (f"The takes are already matched, so the jump is inside take {takes[side]}: "
                      f"the part {'before' if side == 0 else 'after'} the cut is "
                      f"{abs(deviation):.1f} LU {'louder' if deviation > 0 else 'quieter'} "
                      "than the rest of that take. Listen at this cut, "
                      "and record the line again if it stands out.")
        old_action = finding["suggested_action"]
        finding["message"] = finding["message"].removesuffix(old_action) + action
        finding["suggested_action"] = action
        changed = True
    if changed and "text" in result.get("summary", {}) and "media" in result:
        from yap_studio.report import summarize

        result["summary"]["text"] = summarize(result)
    return result
