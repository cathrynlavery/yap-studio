"""Build a soft person redraw around number cards without changing other cards."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yap_studio.stages.assemble_picture import _Graph
    from yap_studio.stages.timeline import ResolvedTimeline

REDRAW_PAD_FRAMES = 15  # Extend the person redraw by half a second on each side at 30 fps.
REDRAW_FADE_FRAMES = 9  # Fade the redraw over 0.3 seconds at each outer boundary.
OPACITY_SIZE_PX = 2  # Evaluate the envelope on a tiny even plane before expanding it.
OPAQUE_ALPHA = 255  # Multiply opacity on the same eight-bit scale as the person alpha.


def number_redraw(graph: _Graph, person: str, layers: list[dict],
                  timeline: ResolvedTimeline, *, width: int, height: int,
                  fps: int) -> tuple[str, str]:
    """Union old exact windows with padded number windows and a matching alpha envelope."""
    numbers = {item.id for item in timeline.elements
               if item.kind == "graphic" and item.payload.get("template") == "big_number"}
    windows = []
    envelopes = []
    changed = False
    for layer in layers:
        first = layer["start_frame"]
        last = first + layer["frames"]
        if layer["id"] in numbers:
            changed = True
            fade_first, fade_last = first - REDRAW_PAD_FRAMES, last + REDRAW_PAD_FRAMES
            first = max(0, fade_first)
            last = min(timeline.total_frames, fade_last)
            # Clip the envelope with the video so a card at an edge still gets a solid person.
            envelope = (f"max(0,min(1,min((N-({fade_first}))/{REDRAW_FADE_FRAMES},"
                        f"({fade_last}-N)/{REDRAW_FADE_FRAMES})))")
        else:
            envelope = f"between(N,{first},{last - 1})"
        windows.append(f"between(n,{first},{last - 1})")
        envelopes.append(envelope)
    if changed:
        envelope = envelopes[0]
        for value in envelopes[1:]:
            envelope = f"max({envelope},{value})"
        index = len(graph.filters)
        colour, alpha, opacity = (f"number_{kind}_{index}" for kind in
                                   ("colour", "alpha", "opacity"))
        graph.filters.append(f"[{person}]split=2[{colour}][{alpha}]")
        alpha = graph.filter(alpha, "alphaextract")
        graph.filters.append(
            f"color=white:s={OPACITY_SIZE_PX}x{OPACITY_SIZE_PX}:r={fps},format=gray,"
            f"trim=end_frame={timeline.total_frames},settb=1/{fps},setpts=N,"
            f"geq=lum='{OPAQUE_ALPHA}*({envelope})',"
            f"scale={width}:{height}:flags=neighbor,format=gray[{opacity}]"
        )
        alpha = graph.filter(f"{alpha}][{opacity}", "blend=all_mode=multiply:shortest=1")
        person = graph.filter(f"{colour}][{alpha}", "alphamerge=ts_sync_mode=nearest")
    return person, "+".join(windows) or "0"
