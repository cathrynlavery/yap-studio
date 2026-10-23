"""Plan and check the same head regions used by final face verification."""

from dataclasses import replace

from yap_studio.checks.caption_layout import LayoutRecord, check_caption_cards
from yap_studio.errors import YapError
from yap_studio.faces.clearance import Element, Layout, check_layout, face_clearance_px, head_box
from yap_studio.faces.mapping import map_cut_faces
from yap_studio.faces.model import Box, FacesData
from yap_studio.render import camera3d as camera
from yap_studio.render.camera3d_geometry import painted_box, shutter_poses


def head_regions(sections: list[dict], plate: dict, samples: list[dict],
                 frames: int, faces: FacesData | None, style: dict, *,
                 replacements: list[tuple[int, int]] = ()) -> list[list[float]]:
    """Union matte heads with mapped faces, retaining room for timed box unions."""
    result = []
    margin = face_clearance_px(style) + 8.1
    for frame, sample in enumerate(samples[:frames]):
        if any(first <= frame < last for first, last in replacements):
            result.append([0, 0, 0, 0])
            continue
        head = camera._head_at(plate, sample, camera._section_at(sections, frame)["head_box"])
        boxes = [head]
        if faces is not None and faces.status == "measured":
            x, y, scale = camera._person_transform(sample)
            for face in faces.frames[frame]:
                box = head_box(Box(x + scale * face.box.x, y + scale * face.box.y,
                                   scale * face.box.w, scale * face.box.h))
                boxes.append([box.x, box.y, box.x1, box.y1])
        else:
            box = head_box(Box(head[0], head[1], head[2] - head[0], head[3] - head[1]))
            boxes.append([box.x, box.y, box.x1, box.y1])
        result.append([min(box[0] for box in boxes) - margin,
                       min(box[1] for box in boxes) - margin,
                       max(box[2] for box in boxes) + margin,
                       max(box[3] for box in boxes) + margin])
    return result


def reject_unsafe_groups(groups: list[dict], samples: list[dict], heads: list,
                         fallbacks: list[dict], obstacles: list[dict] = (),
                         heroes: list = (), *, top: float = camera.SAFE[1]) -> None:
    """Send painted head, text, obstacle or safe top conflicts to the bottom band."""
    for index, group in enumerate(groups):
        if group.get("unplaced"):
            continue
        minimum = (camera.CAPTION_MIN_SIZE if group["flat"]
                   else camera.CAPTION_MIN_PAINTED_SIZE)
        for frame in range(group["on_frame"], group["off_frame"]):
            box = painted_box(group, samples, frame)
            reason = None
            if box[1] < top:
                reason = f"paint crosses safe text top at frame {frame}"
            elif min(pose[2] * group["ref_size"] for pose in shutter_poses(
                    group, samples, frame)) < minimum:
                reason = f"paint size below {minimum:g} px at frame {frame}"
            elif camera._overlap(box, heads[frame]):
                reason = f"entry crosses head at frame {frame}"
            elif any(item["start_frame"] <= frame < item["end_frame"]
                     and camera._overlap(box, item["box"], 8.1) for item in obstacles):
                reason = f"paint crosses obstacle at frame {frame}"
            elif ((heroes and any(camera._overlap(box, hero, 24) for hero in heroes[frame]))
                  or any(not prior.get("unplaced")
                         and prior["on_frame"] <= frame < prior["off_frame"]
                         and camera._overlap(box, painted_box(prior, samples, frame), 24)
                         for prior in groups[:index])):
                reason = f"paint crosses text at frame {frame}"
            if reason:
                camera._unplace_group(group, fallbacks, reason)
                break


def _check_obstacles(layout: Layout, obstacles: list[dict]) -> None:
    elements = []
    for index, item in enumerate(obstacles):
        x, y, right, bottom = item["box"]
        elements.append(Element(f"obstacle-{index}", "overlay", Box(x, y, right - x, bottom - y),
                                item["start_frame"], item["end_frame"]))
    record = LayoutRecord(replace(layout, elements=tuple(elements), caption_gap_px=0),
                          True, None, {})
    findings = check_caption_cards(record, fps=30)
    if findings:
        ids = ", ".join(sorted({item.evidence["caption_id"] for item in findings}))
        raise YapError(
            f"3D camera captions overlap a card, title or overlay before rendering: {ids}.",
            code="camera3d_caption_clearance",
            suggested_action="Move the card, title or overlay, or pick another caption look.")


def check_plan(plan: dict, layout: dict, faces: FacesData, margin: float, *,
                obstacles: list[dict] = (), replacements: list[tuple[int, int]] = ()) -> None:
    """Stop before rendering if final face or card checks would reject this layout."""
    geometry = Layout.from_dict(layout)
    _check_obstacles(geometry, obstacles)
    mapped, _ = map_cut_faces(faces, total_frames=plan["frames"], end_hold_frames=0,
                              zooms=(), inserts=replacements, video=faces.video,
                              camera=camera.face_transforms(plan))
    report = check_layout(mapped, geometry, margin_px=margin)
    if report.violations:
        ids = ", ".join(sorted({item.target_id for item in report.violations}))
        raise YapError(f"3D camera caption face clearance failed before rendering: {ids}.",
                       code="camera3d_face_clearance",
                       suggested_action="Frame the shot wider, or pick another caption look.")
    top = plan["style"]["data"]["internal"]["key_text_top_px"]
    if any(box["y"] < top for caption in layout["captions"] for box in caption["boxes"]):
        raise YapError("3D camera captions cross the safe text top before rendering.",
                       code="camera3d_caption_clearance",
                       suggested_action="Frame the shot wider, or pick another caption look.")
    camera.check_caption_clearance(plan)
    camera.check_hero_caption_clearance(plan)
