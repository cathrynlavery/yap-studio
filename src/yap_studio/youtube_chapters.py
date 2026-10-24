"""YouTube chapter text from the resolved edit clock."""

from __future__ import annotations

from fractions import Fraction


def chapter_list(rows: list[dict], total_frames: int, *, fps: str | Fraction | int,
                 opening: str = "Opening"
                 ) -> tuple[str | None, str | None]:
    """Validate the displayed seconds too: YouTube receives whole-second starts."""
    if not rows:
        return None, None
    try:
        rate = Fraction(str(fps))
        if rate <= 0:
            raise ValueError
    except (ValueError, ZeroDivisionError):
        return None, "Use a valid positive frame rate for the chapter times."
    if (type(total_frames) is not int or total_frames <= 0 or not isinstance(rows, list)
            or not isinstance(opening, str)
            or any(not isinstance(row, dict) or type(row.get("start_frame")) is not int
                   or row["start_frame"] < 0 or not isinstance(row.get("title"), str)
                   for row in rows)):
        return None, "Give every chapter a name and a valid start time inside the video."
    chapters = sorted((row["start_frame"] // rate, " ".join(row["title"].split())) for row in rows)
    if min(row["start_frame"] for row in rows) > 0:
        chapters.insert(0, (0, " ".join(opening.split()) or "Opening"))
    if len(chapters) < 3:
        return None, "Add at least three chapters, starting at 0:00."
    ends = [time for time, _ in chapters[1:]] + [total_frames / rate]
    if chapters[0][0] != 0 or any(end - start < 10 for (start, _), end
                                in zip(chapters, ends, strict=True)):
        return None, ("Space chapters at least 10 seconds apart, "
                      "including the opening and final chapter.")
    if any(not title for _, title in chapters):
        return None, "Give every chapter a short name."

    def clock(seconds: int) -> str:
        minutes, second = divmod(seconds, 60)
        hours, minute = divmod(minutes, 60)
        return f"{hours}:{minute:02d}:{second:02d}" if hours else f"{minute}:{second:02d}"

    return "".join(f"{clock(start)} {title}\n" for start, title in chapters), None


def timeline_chapters(timeline, title_layers: list[dict]) -> dict | None:
    """Retain chapter evidence with the delivery, so later edits cannot change it."""
    from yap_studio.stages.timeline import FPS

    if not any(item.kind == "chapter" for item in timeline.elements):
        return None
    display = {layer["id"]: " ".join(layer["display_text"]) for layer in title_layers
               if layer["kind"] in {"title", "chapter"}}
    rows = [{"start_frame": item.start_frame, "title": display[item.id]}
            for item in timeline.elements if item.kind == "chapter"]
    opening = next((display[item.id] for item in timeline.elements
                    if item.kind == "title" and item.start_frame == 0), "Opening")
    return {"chapters": rows, "total_frames": timeline.total_frames + timeline.end_hold_frames,
            "opening": opening, "fps": f"{FPS.numerator}/{FPS.denominator}"}


def version_chapters(manifest) -> str | None:
    """Use the selected version's frozen report, never a subsequently edited lane."""
    from yap_studio.feedback.versions import version_report

    if manifest is None or not manifest.media["source"].startswith("final/"):
        return None
    report = version_report(manifest)
    return ((report or {}).get("qa") or {}).get("youtube_chapters")
