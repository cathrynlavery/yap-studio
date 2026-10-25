"""Select a sharp, face-forward cut frame on the final picture clock."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from yap_studio import ffmpeg
from yap_studio.errors import ValidationError
from yap_studio.faces.model import FaceBox, FacesData
from yap_studio.gallery import still


@dataclass(frozen=True)
class TimelineSnapshot:
    """Only the final receipt's picture windows needed for cover selection."""

    total_frames: int
    end_hold_frames: int
    replacements: tuple[tuple[str, int, int], ...]

    @classmethod
    def from_document(cls, data: Any) -> TimelineSnapshot:
        """Read frame windows from a rendered timeline snapshot."""
        if not isinstance(data, dict) or data.get("schema") != "yap-studio/resolved-timeline@1":
            raise ValidationError("timeline", "final receipt has no resolved timeline snapshot")
        total = data.get("total_frames")
        hold = data.get("end_hold_frames")
        elements = data.get("elements")
        if (type(total) is not int or total < 1 or type(hold) is not int or hold < 0
                or not isinstance(elements, list)):
            raise ValidationError("timeline", "final receipt has invalid frame counts")
        replacements = []
        for element in elements:
            if not isinstance(element, dict):
                raise ValidationError("timeline", "final receipt has an invalid element")
            kind = element.get("kind")
            start = element.get("start_frame")
            end = element.get("end_frame")
            if (not isinstance(kind, str) or type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= total + hold):
                raise ValidationError("timeline", "final receipt has an invalid frame window")
            if kind in {"insert", "transition"}:
                replacements.append((kind, start, end))
        return cls(total, hold, tuple(replacements))


@dataclass(frozen=True)
class Pick:
    """Selected frame and the measures that led to it."""

    frame: int
    reason: str
    candidates: list[dict]
    face_box: tuple[float, float, float, float] | None = None


def eligible_frame(frame: int, timeline: TimelineSnapshot) -> str | None:
    """Explain why a final-clock frame does not show the cut speaker."""
    if frame < 0 or frame >= timeline.total_frames + timeline.end_hold_frames:
        return "outside the final video"
    if frame >= timeline.total_frames:
        return "end hold repeats the last cut frame"
    for kind, start, end in timeline.replacements:
        if start <= frame < end:
            return f"{kind} replaces the speaker"
    return None


def extract_frame(cut: Path, frame: int, output: Path, *, shape: str = "vertical") -> Path:
    """Save an exact decode-order frame at cover dimensions."""
    size = "1280:720" if shape == "horizontal" else "1080:1920"
    # Finished cuts are 30 fps; integer-second seeking keeps the frame clock exact.
    seek = ["-ss", str(frame // 30)] if shape == "horizontal" else []
    selected = frame % 30 if shape == "horizontal" else frame
    graph = (f"select=eq(n\\,{selected}),"
             f"scale={size}:force_original_aspect_ratio=increase,crop={size}")
    ffmpeg.run(["ffmpeg", "-y", "-v", "error", *seek, "-i", str(cut),
                "-vf", graph, "-frames:v", "1", "-pix_fmt", "rgb24", str(output)],
               timeout_s=120)
    if not output.is_file():
        raise ValidationError("frame", f"frame {frame} could not be extracted; "
                              "choose another frame")
    return output


def _gray(path: Path, *, shape: str = "vertical") -> np.ndarray:
    raw = ffmpeg.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                      "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"], timeout_s=60,
                     binary=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(
        (720, 1280) if shape == "horizontal" else (1920, 1080))


def _face(faces: tuple[FaceBox, ...], width: int, height: int) -> FaceBox | None:
    eligible = [face for face in faces if face.box.x > 0 and face.box.y > 0
                and face.box.x1 < width and face.box.y1 < height]
    return max(eligible, key=lambda face: face.box.w * face.box.h, default=None)


def pick_frame(
    cut: Path | None, faces: FacesData, timeline: TimelineSnapshot, *, frame: int | None = None,
    image_loader: Callable[[int], np.ndarray] | None = None,
    landmark_detector: still.LandmarkDetector | None = None,
    choose_default_detector: bool = True,
    shape: str = "vertical",
    layout: Callable[[tuple], object] | None = None,
) -> Pick:
    """Rank large faces, then reject closed eyes and soft face crops."""
    if frame is not None:
        reason = eligible_frame(frame, timeline)
        if reason:
            raise ValidationError("frame", f"frame {frame} is not eligible: {reason}; "
                                  "choose a speaker frame")
        if frame >= faces.video.frames:
            raise ValidationError("frame", "frame is outside the cut; choose a lower frame")
        box = None
        if shape == "horizontal":
            from yap_studio.covers.wide import check_face, thumbnail_face

            face = (_face(faces.frames[frame], faces.video.width, faces.video.height)
                    if frame < len(faces.frames) else None)
            if face is None:
                raise ValidationError("face", "wide thumbnail needs a face; choose another frame")
            box = (face.box.x, face.box.y, face.box.w, face.box.h)
            mapped = thumbnail_face(face.box, faces.video.width, faces.video.height)
            target = (mapped.x, mapped.y, mapped.w, mapped.h)
            check_face([], target)
            if layout is not None:
                layout(target)
        return Pick(frame, "--frame override", [{"frame": frame, "override": True}], box)
    if faces.status != "measured":
        raise ValidationError("faces", f"faces unavailable: {faces.reason}; choose --frame N")
    sampled = []
    count = min(timeline.total_frames, len(faces.frames))
    # shortcut: 120 wide samples or one per speaker window, batch-decode for denser picks.
    step = max(15, ((count + 1799) // 1800) * 15) if shape == "horizontal" else 15
    indices = list(range(0, count, step))
    if shape == "horizontal":
        seeds = []
        cursor = 0
        for _, start, end in sorted(timeline.replacements, key=lambda window: window[1]):
            if cursor < min(start, count):
                seeds.append(cursor)
            cursor = max(cursor, end)
        if cursor < count:
            seeds.append(cursor)
        rest = [i for i in indices if i not in seeds and eligible_frame(i, timeline) is None]
        budget = max(0, 120 - len(seeds))
        if len(rest) > budget:
            rest = [rest[int(i)] for i in np.linspace(0, len(rest) - 1, budget)]
        indices = sorted(set(seeds + rest))
    for index in indices:
        if eligible_frame(index, timeline) is not None:
            continue
        face = _face(faces.frames[index], faces.video.width, faces.video.height)
        if face is not None:
            sampled.append((index, face))
    if not sampled:
        raise ValidationError("frame", "no eligible face frame; try --frame N on the speaker")
    sampled.sort(key=lambda item: item[1].box.w * item[1].box.h, reverse=True)
    detector = landmark_detector
    if detector is None and choose_default_detector and image_loader is None:
        detector = still.default_landmark_detector()
    reason = ("face landmarks unavailable; picked by face size and sharpness"
              if detector is None else "largest sharp face with open eyes")
    if shape == "horizontal":
        reason += "; ranked lighting and room beside the face across the finished cut"
    with tempfile.TemporaryDirectory(prefix="cover-pick-") as directory:
        folder = Path(directory)
        rows = []
        for start in range(0, len(sampled), 24):
            images = []
            batch = []
            for index, face in sampled[start:start + 24]:
                path = folder / f"frame-{index:05d}.png"
                if image_loader is None:
                    if cut is None:
                        raise ValueError("cut is needed to extract frames")
                    extract_frame(cut, index, path, shape=shape)
                    pixels = _gray(path, shape=shape)
                else:
                    pixels = image_loader(index)
                images.append(path)
                box = face.box
                if shape == "horizontal":
                    from yap_studio.covers.wide import thumbnail_face

                    box = thumbnail_face(box, faces.video.width, faces.video.height)
                crop = still.FaceLandmarks(box.x, box.y, box.w, box.h, None, None,
                                           None, None, face.confidence)
                batch.append({"frame": index, "face_box": box.to_dict(),
                              "face_area": round(box.w * box.h, 2),
                              "sharpness": round(still._sharpness(pixels, crop), 3)})
                if shape == "horizontal":
                    batch[-1]['source_face_box'] = (face.box.x, face.box.y, face.box.w, face.box.h)
                    batch[-1]['face_box'] = dict(x=box.x, y=box.y, w=box.w, h=box.h)
                    patch = pixels[max(0, int(box.y)):int(box.y1),
                                   max(0, int(box.x)):int(box.x1)]
                    batch[-1].update(face_luma=round(float(np.mean(patch)), 2),
                                     headline_room=round(max(box.x, 1280 - box.x1) / 1280, 4))
            if detector is not None:
                found = detector.measure(images)
                if len(found) != len(batch):
                    raise ValidationError("landmarks", "detector returned the wrong frame count")
                for row, landmarks in zip(batch, found, strict=True):
                    largest = max(landmarks, key=lambda face: face.w * face.h, default=None)
                    row["eyes"] = None if largest is None else [largest.left_eye, largest.right_eye]
                    row["yaw"] = None if largest is None else largest.yaw
            else:
                for row in batch:
                    row["eyes"] = None
                    row["yaw"] = None
            rows.extend(batch)
            eye_values = [value for row in rows if row["eyes"] is not None
                          and all(value is not None for value in row["eyes"])
                          for value in row["eyes"]]
            threshold = (max(still.EYES_OPEN,
                             still.EYES_RELATIVE * float(np.median(eye_values)))
                         if eye_values else still.EYES_OPEN) if detector is not None else None
            best_sharpness = max(row["sharpness"] for row in rows)
            best_area = max(row["face_area"] for row in rows)
            for row in rows:
                eyes = row["eyes"]
                row["eyes_open"] = (None if threshold is None else eyes is not None
                                    and all(value is not None and value >= threshold
                                            for value in eyes))
                row["motion_blur"] = (best_sharpness > 0 and
                                      row["sharpness"] < best_sharpness * 0.35)
                row["score"] = round(row["face_area"] / best_area +
                                     (row["sharpness"] / best_sharpness if best_sharpness else 0) -
                                     min(1, abs(row["yaw"] or 0) / 0.5) * 0.25, 4)
                if shape == "horizontal":
                    row["well_lit"] = 45 <= row["face_luma"] <= 225
                    row["score"] = round(row["score"] + 2 * row["headline_room"] +
                                         (0.5 if row["well_lit"] else -0.5), 4)
            if shape == "horizontal" and start + 24 < len(sampled):
                continue
            viable = [row for row in rows if not row["motion_blur"] and
                      row["eyes_open"] is not False]
            if viable:
                for chosen in sorted(viable, key=lambda row: (-row["score"], row["frame"])):
                    if shape == "horizontal":
                        from yap_studio.covers.wide import check_face

                        box = tuple(chosen['face_box'][k] for k in ('x', 'y', 'w', 'h'))
                        try:
                            check_face([], box)
                            if layout is not None:
                                layout(box)
                        except ValidationError as exc:
                            chosen['placement_error'] = exc.message
                            continue
                    return Pick(chosen["frame"], reason, rows, chosen.get('source_face_box'))
                raise ValidationError("frame", "no ranked frame has room for this layout; "
                                      "shorten text or choose another frame")
        raise ValidationError("frame", "no sharp frame with open eyes; try --frame N")
