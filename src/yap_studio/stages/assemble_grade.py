"""Composite a grade comparison before text layers and person restoration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from yap_studio.grade import HANDLE_FADE_FRAMES, compare_x_expr, graph_filter, handle_rgba

if TYPE_CHECKING:
    from yap_studio.stages.assemble_picture import _Graph
    from yap_studio.stages.timeline import ResolvedElement


def grade_picture(graph: _Graph, base: str, cube: Path, windows: list[ResolvedElement],
                  style: dict, work: Path) -> tuple[str, tuple[dict, ...]]:
    """Keep one raw branch per window and apply the cube once to the other branch."""
    raw_labels = [f"compare_raw_{i}" for i in range(len(windows))]
    graph.filters.append(f"[{base}]split={len(windows)+1}[compare_grade]"
                         + "".join(f"[{label}]" for label in raw_labels))
    base = graph.filter("compare_grade", graph_filter(cube))
    width, height, pixels = handle_rgba(style)
    handle = work / "grade-handle.rgba"
    handle.write_bytes(pixels)
    records = []
    for index, (item, raw) in enumerate(zip(windows, raw_labels, strict=True)):
        start, end = item.start_frame, item.end_frame
        frames = end - start
        x = compare_x_expr(start, frames, 1080)
        mask = f"compare_mask_{index}"
        graph.filters.append(
            "color=c=black:s=2160x1920:r=30,format=gbrp16le,"
            "drawbox=x=0:y=0:w=1080:h=1920:color=white:t=fill,"
            f"crop=1080:1920:x='1080-({x})':y=0,extractplanes=r,format=gray,"
            f"split=3[{mask}_0][{mask}_1][{mask}_2]"
        )
        # Copy the binary RGB mask into all three YUV planes. The graded branch
        # stays in its original format, including frames outside the window.
        mask = graph.filter(f"{mask}_0][{mask}_1][{mask}_2",
                            "mergeplanes=mapping=0x001020:format=yuv444p")
        raw = graph.filter(raw, "format=yuv444p")
        base = graph.filter(f"{base}][{raw}][{mask}",
                            f"maskedmerge=planes=7:enable='between(n,{start},{end-1})'")
        number = graph.count
        graph.count += 1
        graph.owners[number] = (f"{item.id} (grade handle)", item)
        graph.inputs.extend(["-f", "rawvideo", "-pixel_format", "rgba", "-video_size",
                             f"{width}x{height}", "-framerate", "30", "-stream_loop", "-1",
                             "-i", str(handle)])
        layer = graph.filter(
            f"{number}:v:0", f"trim=end_frame={frames},settb=1/30,setpts=N,"
            f"fade=t=in:alpha=1:start_frame=0:nb_frames={HANDLE_FADE_FRAMES},"
            f"fade=t=out:alpha=1:start_frame={frames-1-HANDLE_FADE_FRAMES}:"
            f"nb_frames={HANDLE_FADE_FRAMES},setpts=PTS+{start}",
        )
        # Overlay's frame counter differs from crop's; PTS is the output clock.
        handle_x = re.sub(r"\bn\b", "(t*30)", x)
        base = graph.overlay(base, layer, start, end, x=f"'{handle_x}-{width/2:g}'",
                             enable=f"gte(t,{start}/30)*lt(t,{end}/30)")
        records.append({"id": item.id, "start_frame": start, "end_frame": end})
    return base, tuple(records)
