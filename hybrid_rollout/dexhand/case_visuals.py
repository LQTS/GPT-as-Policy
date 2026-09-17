"""Image overlays and contact sheets for persistent DexHand cases."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


_AXIS_COLORS = {
    "X": (220, 45, 45),
    "Y": (35, 170, 70),
    "Z": (45, 100, 230),
}


def _camera_basis(view: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eye = np.asarray(view["eye"], dtype=np.float64)
    target = np.asarray(view["target"], dtype=np.float64)
    z_axis = eye - target
    z_axis /= np.linalg.norm(z_axis)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(up_hint, z_axis)
    if np.linalg.norm(x_axis) < 1.0e-8:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(up_hint, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return x_axis, y_axis, z_axis


def _project_point(point: np.ndarray, view: dict, width: int, height: int) -> np.ndarray:
    eye = np.asarray(view["eye"], dtype=np.float64)
    x_axis, y_axis, z_axis = _camera_basis(view)
    relative = np.asarray(point, dtype=np.float64) - eye
    camera = np.array(
        [
            np.dot(relative, x_axis),
            np.dot(relative, y_axis),
            np.dot(relative, z_axis),
        ]
    )
    depth = -camera[2]
    if depth <= 1.0e-6:
        raise ValueError("World-axis point is behind the configured camera.")
    focal_length = 34.0
    horizontal_aperture = 24.0
    focal_pixels = focal_length * float(width) / horizontal_aperture
    return np.array(
        [
            0.5 * width + focal_pixels * camera[0] / depth,
            0.5 * height - focal_pixels * camera[1] / depth,
        ]
    )


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    *,
    width: int,
) -> None:
    start_xy = tuple(float(value) for value in start)
    end_xy = tuple(float(value) for value in end)
    draw.line((start_xy, end_xy), fill=(255, 255, 255), width=width + 4)
    draw.line((start_xy, end_xy), fill=(15, 15, 15), width=width + 2)
    draw.line((start_xy, end_xy), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_length = 10.0
    head_angle = math.radians(28.0)
    left = (
        end[0] - head_length * math.cos(angle - head_angle),
        end[1] - head_length * math.sin(angle - head_angle),
    )
    right = (
        end[0] - head_length * math.cos(angle + head_angle),
        end[1] - head_length * math.sin(angle + head_angle),
    )
    draw.polygon((end_xy, left, right), fill=color)


def annotate_world_axes(
    image: Image.Image,
    view: dict,
    *,
    target_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Image.Image:
    """Draw a camera-correct world-axis triad in the lower-left corner."""
    output = image.convert("RGB").copy()
    width, height = output.size
    draw = ImageDraw.Draw(output)
    panel_height = min(150, height - 8)
    panel_width = min(166, width - 8)
    panel = (4, height - panel_height - 4, panel_width + 4, height - 4)
    draw.rectangle(panel, fill=(250, 250, 250), outline=(25, 25, 25), width=2)
    draw.text((12, height - panel_height + 4), "WORLD", fill=(15, 15, 15))

    reference = np.asarray(view["target"], dtype=np.float64)
    projected_origin = _project_point(reference, view, width, height)
    axis_vectors = {}
    unit_axes = np.eye(3, dtype=np.float64)
    for label, axis in zip(("X", "Y", "Z"), unit_axes):
        projected_end = _project_point(reference + 0.1 * axis, view, width, height)
        axis_vectors[label] = projected_end - projected_origin

    max_length = max(float(np.linalg.norm(vector)) for vector in axis_vectors.values())
    scale = 52.0 / max(max_length, 1.0e-8)
    origin = np.array((85.0, float(height - 69)), dtype=np.float64)
    target = np.asarray(target_axis, dtype=np.float64)
    target /= max(float(np.linalg.norm(target)), 1.0e-8)
    highlighted = ("X", "Y", "Z")[int(np.argmax(target))]
    highlighted = highlighted if float(target[int(np.argmax(target))]) > 0.99 else ""

    for label in ("X", "Y", "Z"):
        vector = axis_vectors[label] * scale
        vector_length = float(np.linalg.norm(vector))
        if 0.0 < vector_length < 12.0:
            vector *= 12.0 / vector_length
        end = origin + vector
        is_target = label == highlighted
        _draw_arrow(
            draw,
            origin,
            end,
            _AXIS_COLORS[label],
            width=5 if is_target else 3,
        )
        text = f"+{label}" + (" target" if is_target else "")
        offset = np.sign(vector) * 4.0
        position = end + offset
        draw.text(
            tuple(float(value) for value in position),
            text,
            fill=_AXIS_COLORS[label],
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    draw.ellipse(
        (origin[0] - 3, origin[1] - 3, origin[0] + 3, origin[1] + 3),
        fill=(15, 15, 15),
    )
    return output


def make_case_preview(
    case_id: str,
    paths: dict[str, Path],
    output: Path,
    camera_views: tuple[dict, ...],
    *,
    target_speed: float,
) -> Path:
    """Compose annotated camera views into one selection preview."""
    images = [Image.open(paths[view["name"]]).convert("RGB") for view in camera_views]
    width = sum(image.width for image in images)
    header = 34
    canvas = Image.new("RGB", (width, max(image.height for image in images) + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), f"{case_id}   target: world +Z   speed: {target_speed:.1f} rad/s", fill="black")
    x = 0
    for view, image in zip(camera_views, images):
        canvas.paste(image, (x, header))
        draw.rectangle((x, header, x + 116, header + 22), fill="white")
        draw.text((x + 6, header + 5), view["name"], fill="black")
        x += image.width
    path = output / "preview.png"
    canvas.save(path)
    return path


def make_contact_sheet(entries: list[dict], output: Path) -> Path:
    """Create an annotated front-view index for choosing cases."""
    columns = 4
    tile_width = 320
    tile_height = 276
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, entry in enumerate(entries):
        image = Image.open(entry["images"]["front"]).convert("RGB")
        image.thumbnail((tile_width, 240))
        x = (slot % columns) * tile_width
        y = (slot // columns) * tile_height
        sheet.paste(image, (x + (tile_width - image.width) // 2, y))
        label = f"{entry['case_id']}  {entry['split_role']}  group={entry['grasp_group']}"
        draw.text((x + 6, y + 246), label, fill="black")
    path = output / "contact_sheet.png"
    sheet.save(path)
    return path


__all__ = ["annotate_world_axes", "make_case_preview", "make_contact_sheet"]
