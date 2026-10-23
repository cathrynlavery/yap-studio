"""Extract sound cues from resolved picture events and apply spacing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from yap_studio.cards import (
    FLOW_ARROW_DRAW_S,
    FLOW_BOX_DRAW_S,
    FLOW_NODE_REVEAL_S,
    FLOW_PANEL_IN_S,
    FLOW_REST_S,
    FLOW_TAGS_POP_S,
    STEPS_CHIP_IN_S,
    STEPS_PANEL_IN_S,
    STEPS_REST_S,
    STEPS_STAGGER_S,
    _flow_tags_cues,
    _row_cues,
    _terminal_schedule,
    _typing_schedule,
    load_template,
)
from yap_studio.render.hyperframes import layer_duration_s
from yap_studio.sfx import AUTO_EXTRAS
from yap_studio.stages.timeline import ResolvedElement, ResolvedTimeline

TRANSITION_LEAD_S = 0.2


@dataclass(frozen=True)
class Cue:
    """One candidate sound on the output clock."""

    sound: str
    event: str
    element_id: str
    time_s: float
    priority: int
    rule: str
    peak_s: float = 0.0
    gain_db: float = 0.0
    align_peak: bool = False
    end_s: float | None = None
    fade_out_s: float = 0.0
    drop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return receipt-safe cue identity."""
        return {"sound": self.sound, "event": self.event, "element_id": self.element_id,
                "time_s": self.time_s, **({"peak_s": self.peak_s} if self.peak_s else {})}


def _card_typing_schedule(item: ResolvedElement) -> dict[str, Any]:
    """Use the same text timing as the rendered card."""
    kind = item.payload["template"]
    slots = item.payload["slots"]
    rules = load_template(kind)["slots"]
    values = {name: slots[name] if name in slots else rule["default"]
              for name, rule in rules.items()}
    duration = layer_duration_s(item.end_frame - item.start_frame)
    if kind == "terminal":
        return _terminal_schedule(
            values["command"], values["output"], values["cadence"], values["type_at"],
            values["word_cues"], values["cues"], duration)
    return _typing_schedule(
        values["request" if kind == "prompt_bar" else "keyword"], values["cadence"],
        values["type_at"], values["word_cues"], values["cues"], duration)


def _template_events(
    item: ResolvedElement, *, flow_look: str = "boxes", specs_rows: str = "together",
) -> list[tuple[tuple[str, ...], str, float]]:
    """Mirror the shipped card template timings on a resolved card window."""
    start = item.start_s
    duration = item.end_s - start
    if item.kind == "transition":
        return [(("transition",), "section", item.payload["join_frame"] / 30)]
    if item.kind == "chapter":
        return [(("chapter",), "chapter", start)]
    if item.kind != "graphic":
        return []
    kind = item.payload["template"]
    slots = item.payload["slots"]
    out_start = duration - min(0.3, duration * 0.15)
    if kind == "checklist":
        count = len(slots["items"])
        natural_end = 0.3 + max(0, count - 1) * 0.35 + 0.65
        scale = min(1, out_start / natural_end)
        explicit = slots.get("cues", [])
        previous = 0.0
        times = []
        for index in range(count):
            at = (explicit[index] if index < len(explicit) else
                  min(out_start, previous + 0.35 * scale) if index and explicit else
                  (0.3 + index * 0.35) * scale)
            times.append((("item",), f"checklist:{index + 1}", start + at))
            previous = at
        return times
    if kind == "steps":
        count = len(slots["steps"])
        entrance_end = (STEPS_PANEL_IN_S + STEPS_CHIP_IN_S +
                        (count - 1) * STEPS_STAGGER_S)
        latest = out_start - STEPS_REST_S
        gap = (latest - entrance_end) / (count - 1)
        explicit = slots.get("cues", [])
        cues = []
        for index in range(count):
            at = (explicit[index] if index < len(explicit) else
                  max(cues[-1] if cues else 0, min(latest, entrance_end + index * gap)))
            cues.append(at)
        return [(("item",), f"steps:{index + 1}", start + at)
                for index, at in enumerate(cues)]
    if kind == "flow":
        count = len(slots["nodes"])
        if flow_look == "tags":
            cues = _flow_tags_cues(slots.get("cues", []),
                                   layer_duration_s(item.end_frame - item.start_frame), count)
            return [(("pop", "item"), f"flow:{index + 1}", start + at + FLOW_TAGS_POP_S)
                    for index, at in enumerate(cues)]
        first = FLOW_PANEL_IN_S + FLOW_ARROW_DRAW_S
        latest = min(out_start - FLOW_NODE_REVEAL_S - FLOW_REST_S,
                     duration * 0.6 - FLOW_NODE_REVEAL_S)
        gap = (latest - first) / max(1, count - 1)
        explicit = slots.get("cues", [])
        times = []
        for index in range(count):
            at = (explicit[index] if index < len(explicit) else
                  max(times[-1][2] - start - FLOW_BOX_DRAW_S if times else 0,
                      min(latest, first + index * gap)))
            times.append((("item",), f"flow:{index + 1}", start + at + FLOW_BOX_DRAW_S))
        return times
    if kind == "specs_list":
        if specs_rows != "together":
            cues = _row_cues({"cues": slots.get("cues", [])}, len(slots["rows"]),
                             layer_duration_s(item.end_frame - item.start_frame))
            return [(("item",), f"specs_list:{index + 1}", start + at)
                    for index, at in enumerate(cues)]
        return [(("item",), f"specs_list:{index + 1}", start + 0.2 + index * 0.13)
                for index in range(len(slots["rows"]))]
    if kind == "chart":
        bars = [(("item",), f"chart:{index + 1}", start + 0.3 + index * 0.12)
                for index in range(len(slots["bars"]))]
        return [(("count_start",), "chart:start", start + 0.3),
                (("settle",), "chart:final", start + 0.8 + (len(bars) - 1) * 0.12), *bars]
    if kind == "stat":
        end = slots.get("end", 0)
        reveal = slots.get("reveal", "count")
        digit_count = sum(char.isdigit() for char in f"{end:,}")
        wheel_end = 0.38 + 1.1 + max(0, digit_count - 1) * 0.05
        slot_end = 0.38 + 0.52 + max(0, digit_count - 1) * 0.06
        reveal_end = (wheel_end + 0.33 if reveal == "wheel" else
                      slot_end + 0.33 if reveal == "slot" else 2.05)
        scale = min(1, out_start / reveal_end)
        landed = (wheel_end if reveal == "wheel" else
                  slot_end if reveal == "slot" else 1.72) * scale
        return [(("count_start",), "stat:start", start),
                (("settle",), "stat:final", start + landed)]
    if kind == "progress_stat":
        return [(("count_start",), "progress_stat:start", start + 0.25),
                (("settle",), "progress_stat:final", start + 1.10)]
    if kind == "success_check":
        return [(("settle",), "success_check", start + 0.8)]
    if kind == "file_drop":
        from yap_studio.file_cards import FILE_DROP_LAND_S

        return [(("land", "settle"), "file_drop",
                 start + slots.get("drop_at", 0.15) + FILE_DROP_LAND_S)]
    if kind == "jiggle_delete":
        from yap_studio.jiggle_cards import removal_cues

        cues = removal_cues(slots, duration)
        return [(("pop", "item"), f"jiggle_delete:{position}", start + at)
                for position, at in zip(slots["remove"], cues, strict=True)]
    if kind in {"prompt_bar", "comment_cta", "terminal"}:
        schedule = _card_typing_schedule(item)
        if kind == "terminal":
            events = []
        else:
            events = [(("settle",), kind, start + schedule["confirm_at"])]
            if kind == "comment_cta":
                events.append((("message",), kind, start + schedule["confirm_at"]))
        first = next((at for at, count in schedule["rows"] if count > 0), None)
        if first is not None:
            events.append((("typing",), f"{kind}:typing", start + first))
        return events
    if kind == "comment_drop":
        at = slots.get("drop_at", load_template(kind)["slots"]["drop_at"]["default"])
        return [(("land", "settle"), kind, start + at + 0.6),
                (("message",), kind, start + at + 0.6)]
    if kind in {"image_card", "image_compare"}:
        return [(("photo",), kind, start)]
    if kind == "icon_strike":
        at = slots.get("strike_at", load_template(kind)["slots"]["strike_at"]["default"])
        return [(("land", "settle"), kind, start + at + 0.28)]
    if kind == "gravity_drop":
        return [(("land", "settle"), "gravity_drop:land", start + (
                 (drop["at"]["frame"] - item.start_frame) / 30
                 if isinstance(drop["at"], dict) else drop["at"]))
                for drop in slots["drops"]]
    return []


def _event_times(
    item: ResolvedElement, *, flow_look: str = "boxes", specs_rows: str = "together",
) -> list[tuple[tuple[str, ...], str, float]]:
    """Return true picture moments with most-specific-first rule chains."""
    events = _template_events(item, flow_look=flow_look, specs_rows=specs_rows)
    if item.kind == "graphic":
        duration = item.end_s - item.start_s
        out_start = duration - min(0.3, duration * 0.15)
        events = [(("card_in",), "card:in", item.start_s), *events,
                  (("card_out",), "card:out", item.start_s + out_start)]
    return events


def candidates(timeline: ResolvedTimeline, sounds: list[dict[str, Any]], *,
               flow_look: str = "boxes", specs_rows: str = "together",
               sound_extras: list[str] | tuple[str, ...] = (),
               requested_sounds: list[dict[str, Any]] | None = None) -> list[Cue]:
    """Match recipe rules to picture events, honoring per-element overrides."""
    by_rule: dict[str, list[dict[str, Any]]] = {}
    by_name = {sound["name"]: sound for sound in sounds
               if sound.get("state", "fresh") == "fresh"}
    requests = (by_name if requested_sounds is None else
                {sound["name"]: sound for sound in requested_sounds
                 if sound.get("state", "fresh") == "fresh"})
    for sound in sounds:
        if sound["name"] not in by_name or sound.get("request_only"):
            continue
        if (sound["rule"] in {"typing", "message", "photo", "request"}
                or sound["name"] in AUTO_EXTRAS):
            if sound["name"] not in sound_extras or sound["rule"] == "request":
                continue
        by_rule.setdefault(sound["rule"], []).append(sound)
    result = []
    cut_s = timeline.total_frames / 30
    for item in timeline.elements:
        if item.kind == "sound":
            sound = requests.get(item.payload["sound"])
            if sound is not None and 0 <= item.start_s < cut_s:
                result.append(Cue(sound["name"], "request", item.id, item.start_s,
                                  sound["priority"], "request",
                                  sound.get("peak_s", 0.0)
                                  if item.payload.get("align") == "peak" else 0.0,
                                  item.payload.get("gain_db", 0.0),
                                  item.payload.get("align") == "peak"))
            continue
        override = item.payload.get("sfx")
        if override is False:
            continue
        shift = override.get("shift_s", 0) if isinstance(override, dict) else 0
        for chain, event, time in _event_times(item, flow_look=flow_look,
                                               specs_rows=specs_rows):
            rule = next((rule for rule in chain if rule in by_rule), None)
            for sound in by_rule.get(rule, []):
                at = time + shift
                if rule == "transition" and "peak_s" not in sound:
                    at -= TRANSITION_LEAD_S
                end_s = None
                fade_out_s = 0.0
                if rule == "typing":
                    typing_end = item.start_s + shift + _card_typing_schedule(item)["type_end"]
                    card_end = min(item.end_s, cut_s)
                    end_s = min(card_end, max(typing_end, at + 0.35))
                    start_s = max(item.start_s, at)
                    if end_s <= start_s + 1e-9:
                        result.append(Cue(
                            sound["name"], event, item.id, at, sound["priority"], rule,
                            end_s=end_s, drop_reason="outside_card"))
                        continue
                    # Extend only an interval that overlaps the card, keeping the minimum inside it.
                    at = start_s
                    end_s = min(card_end, max(end_s, at + 0.35))
                    if min(typing_end, card_end) - at < 0.35:
                        fade_out_s = 0.06
                if 0 <= at < cut_s:
                    result.append(Cue(sound["name"], event, item.id, at,
                                      sound["priority"], rule,
                                      0.0 if rule == "typing" else sound.get("peak_s", 0.0),
                                      end_s=end_s, fade_out_s=fade_out_s))
    for sound in by_rule.get("settle", []):
        start = cut_s - sound["duration_s"] - 0.05
        if start >= 0:
            result.append(Cue(sound["name"], "end", "end", start + sound.get("peak_s", 0.0),
                              sound["priority"], "settle", sound.get("peak_s", 0.0)))
    return sorted(result, key=lambda cue: cue.time_s)


def space(cues: list[Cue], min_gap_s: float) -> tuple[list[Cue], list[dict[str, Any]]]:
    """Keep higher priority cues while preserving card and count pairs."""
    dropped = [{"cue": cue.to_dict(), "reason": cue.drop_reason}
               for cue in cues if cue.drop_reason]
    cues = [cue for cue in cues if not cue.drop_reason]

    def paired(left: Cue, right: Cue) -> bool:
        if left.element_id != right.element_id:
            return False
        if {left.rule, right.rule} == {"card_in", "card_out"}:
            return True
        return ({left.rule, right.rule} == {"count_start", "settle"} and
                {left.event.rpartition(":")[2], right.event.rpartition(":")[2]}
                == {"start", "final"} and
                left.event.rpartition(":")[0] == right.event.rpartition(":")[0])

    entrance_priorities: dict[str, int] = {}
    for cue in cues:
        if cue.rule == "card_in":
            entrance_priorities[cue.element_id] = min(
                cue.priority, entrance_priorities.get(cue.element_id, cue.priority))

    def order(cue: Cue) -> tuple[bool, int, float, bool, bool]:
        deferred = cue.rule == "card_out" and cue.element_id in entrance_priorities
        priority = cue.priority
        if deferred:
            priority = min(priority, entrance_priorities[cue.element_id])
        return (cue.rule != "request", -priority, cue.time_s, deferred,
                cue.rule in {"typing", "message", "photo"})

    kept: list[Cue] = []
    entrances: set[str] = set()
    lost_entrances: dict[str, Cue] = {}
    # Opted-in extras replace only the same card's cue at the exact picture moment.
    # Apply this before priority spacing, so a higher-priority card cue cannot win it back.
    extras = [cue for cue in cues if cue.rule in {"typing", "message", "photo"}]
    remaining = []
    for cue in cues:
        replacement = next((extra for extra in extras
                            if cue.rule not in {"request", "typing", "message", "photo"}
                            and extra.element_id == cue.element_id
                            and abs(extra.time_s - cue.time_s) < 1e-9), None)
        if replacement is None:
            remaining.append(cue)
        else:
            dropped.append({"cue": cue.to_dict(), "kept_cue": replacement.to_dict(),
                            "reason": "enabled_extra"})
            if cue.rule == "card_in":
                lost_entrances[cue.element_id] = replacement
    for cue in sorted(remaining, key=order):
        blocker = lost_entrances.get(cue.element_id) if cue.rule == "card_out" else None
        conflict = blocker or next((other for other in kept
                                    if abs(other.time_s - cue.time_s) < min_gap_s
                                    and not (cue.rule == other.rule == "request")
                                    and not paired(other, cue)), None)
        if conflict is None:
            kept.append(cue)
            if cue.rule == "card_in":
                entrances.add(cue.element_id)
                lost_entrances.pop(cue.element_id, None)
        else:
            if cue.rule == "card_in" and cue.element_id not in entrances:
                lost_entrances.setdefault(cue.element_id, conflict)
            dropped.append({"cue": cue.to_dict(), "kept_cue": conflict.to_dict(),
                            "reason": "card_in_dropped" if blocker else "minimum_gap"})
    return sorted(kept, key=lambda item: item.time_s), dropped
