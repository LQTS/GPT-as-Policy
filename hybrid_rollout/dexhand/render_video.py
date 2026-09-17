#!/usr/bin/env python3
"""Render recorded DexHand multi-view observations with object-pose telemetry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

from hybrid_rollout.robodojo.io import write_json


WIDTH, HEIGHT = 1600, 900
BG, CARD = "#0c1422", "#152136"
INK, MUTED = "#edf3fb", "#94a7c0"
CYAN, AMBER, RED = "#5bdacc", "#ffbd59", "#ff6b6b"
FONT_PATH = Path(__file__).parents[1] / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
VIEW_NAMES = ("front", "opposite", "top")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def orientation_delta_deg(initial: list[float], current: list[float]) -> float:
    q0_norm = math.sqrt(sum(value * value for value in initial))
    q1_norm = math.sqrt(sum(value * value for value in current))
    cosine = abs(sum(a * b for a, b in zip(initial, current, strict=True)) / (q0_norm * q1_norm))
    return math.degrees(2.0 * math.acos(min(1.0, cosine)))


def load_samples(controller: Path) -> tuple[dict, dict, list[dict]]:
    run = read_json(controller / "run.json")
    result = read_json(controller / "result.json")
    samples = []
    initial_position = None
    initial_quaternion = None
    for directory in sorted((controller / "observations").iterdir()):
        observation_path = directory / "observation.json"
        if not directory.is_dir() or not observation_path.is_file():
            continue
        packet = read_json(observation_path)
        state = packet["state"]
        position = state["object_pose_palm"]["position_m"]
        quaternion = state["object_pose_palm"]["quaternion_wxyz"]
        if initial_position is None:
            initial_position = position
            initial_quaternion = quaternion
        history = packet.get("history", [])
        samples.append(
            {
                "observation": int(directory.name),
                "step_id": packet["step_id"],
                "decisions_used": packet["decisions_used"],
                "position_m": position,
                "position_delta_mm": [
                    1000.0 * (value - origin)
                    for value, origin in zip(position, initial_position, strict=True)
                ],
                "quaternion_wxyz": quaternion,
                "orientation_delta_deg": orientation_delta_deg(initial_quaternion, quaternion),
                "linear_velocity_m_s": state["object_velocity_palm"]["linear_m_s"],
                "angular_velocity_rad_s": state["object_velocity_palm"]["angular_rad_s"],
                "contacts": state["fingertip_contacts"],
                "metrics": state["native_command_metrics"],
                "last_action": history[-1] if history else None,
                "images": {
                    name: str(directory / f"{name}_rgb.png") for name in VIEW_NAMES
                },
            }
        )
    if not samples:
        raise ValueError(f"No recorded observations found in {controller}")
    return run, result, samples


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def fitted_lines(text: str, face: ImageFont.FreeTypeFont, width: int, limit: int) -> list[str]:
    lines, current = [], ""
    for word in str(text).split():
        candidate = word if not current else current + " " + word
        if current and face.getlength(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > limit:
        lines = lines[:limit]
        while lines[-1] and face.getlength(lines[-1] + "...") > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "..."
    return lines


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int = 22,
         color: str = INK, width: int | None = None, lines: int = 1) -> None:
    face = font(size)
    rendered = fitted_lines(value, face, width or WIDTH - xy[0] - 20, lines)
    for index, line in enumerate(rendered):
        draw.text((xy[0], xy[1] + index * (size + 7)), line, font=face, fill=color)


def vector(values: list[float], digits: int = 4) -> str:
    return "[" + ", ".join(f"{value:+.{digits}f}" for value in values) + "]"


def draw_chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str,
               series: list[tuple[str, str, list[float]]], visible: int) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=14, fill=CARD)
    text(draw, (left + 16, top + 10), title, 18, MUTED, width=right - left - 32)
    plot = (left + 45, top + 46, right - 15, bottom - 24)
    x0, y0, x1, y1 = plot
    all_values = [value for _, _, values in series for value in values]
    low, high = min(all_values + [0.0]), max(all_values + [0.0])
    padding = max((high - low) * 0.12, 1.0e-6)
    low, high = low - padding, high + padding
    zero_y = y1 - (0.0 - low) / (high - low) * (y1 - y0)
    draw.line((x0, zero_y, x1, zero_y), fill="#40516a", width=1)
    count = len(series[0][2])
    for label, color, values in series:
        points = []
        for index, value in enumerate(values[:visible]):
            x = x0 if count == 1 else x0 + index * (x1 - x0) / (count - 1)
            y = y1 - (value - low) / (high - low) * (y1 - y0)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        label_x = left + 16 + series.index((label, color, values)) * 120
        text(draw, (label_x, bottom - 22), label, 15, color, width=110)
    text(draw, (left + 6, y0 - 8), f"{high:+.3f}", 13, MUTED, width=70)
    text(draw, (left + 6, y1 - 8), f"{low:+.3f}", 13, MUTED, width=70)


def render_frame(run: dict, result: dict, samples: list[dict], index: int) -> Image.Image:
    sample = samples[index]
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    title = f"GPT-6 Astra direct DexHand  |  step {sample['step_id']} / {run['max_episode_steps']}"
    text(draw, (20, 14), title, 27, INK, width=1180)
    text(draw, (1220, 18), f"sim {sample['step_id'] * run['control_dt_s']:.2f} s", 22, CYAN)

    image_width, image_height = 512, 384
    for view_index, name in enumerate(VIEW_NAMES):
        left = 20 + view_index * 524
        picture = Image.open(sample["images"][name]).convert("RGB")
        picture = picture.resize((image_width, image_height), Image.Resampling.LANCZOS)
        canvas.paste(picture, (left, 62))
        draw.rounded_rectangle((left + 10, 72, left + 165, 106), radius=8, fill="#09111d")
        text(draw, (left + 20, 76), name.upper(), 17, INK, width=135)

    y = 462
    draw.rounded_rectangle((20, y, 520, 880), radius=14, fill=CARD)
    text(draw, (38, y + 14), "Object pose · palm frame", 20, CYAN)
    text(draw, (38, y + 52), "XYZ m       " + vector(sample["position_m"]), 18)
    text(draw, (38, y + 83), "ΔXYZ mm     " + vector(sample["position_delta_mm"], 2), 18, AMBER)
    text(draw, (38, y + 114), "Quat wxyz   " + vector(sample["quaternion_wxyz"]), 18)
    text(draw, (38, y + 145), f"Orientation Δ {sample['orientation_delta_deg']:+.2f} deg", 18)
    text(draw, (38, y + 184), "Linear m/s  " + vector(sample["linear_velocity_m_s"]), 18, MUTED)
    text(draw, (38, y + 215), "Angular r/s " + vector(sample["angular_velocity_rad_s"]), 18, MUTED)
    metrics = sample["metrics"]
    text(draw, (38, y + 254), f"Axis rotation {metrics.get('actual_rotation_rad', 0.0):+.5f} rad", 20, AMBER)
    text(draw, (38, y + 287), f"Axis speed    {metrics.get('actual_axis_speed', 0.0):+.4f} rad/s", 20)
    active = [name for name, present in sample["contacts"].items() if present]
    text(draw, (38, y + 326), "Contacts  " + (", ".join(active) if active else "none"), 18, CYAN)
    text(draw, (38, y + 361), f"Decision {sample['decisions_used']}  ·  observation {sample['observation']}", 17, MUTED)

    deltas = [[item["position_delta_mm"][axis] for item in samples] for axis in range(3)]
    rotations = [item["metrics"].get("actual_rotation_rad", 0.0) for item in samples]
    draw_chart(draw, (536, y, 1048, y + 200), "Position delta · mm",
               [("X", RED, deltas[0]), ("Y", CYAN, deltas[1]), ("Z", AMBER, deltas[2])], index + 1)
    draw_chart(draw, (536, y + 216, 1048, 880), "Signed target-axis rotation · rad",
               [("rotation", AMBER, rotations)], index + 1)

    draw.rounded_rectangle((1064, y, 1580, 880), radius=14, fill=CARD)
    action = sample["last_action"]
    if action is None:
        text(draw, (1084, y + 18), "Initial observation", 23, CYAN)
        text(draw, (1084, y + 60), "No physical action has executed yet.", 19, MUTED, width=474, lines=2)
    else:
        text(draw, (1084, y + 18), f"Astra decision {action['decision'] + 1}", 23, CYAN)
        nonzero = sum(abs(value) > 1.0e-12 for value in action["joint_delta"])
        text(draw, (1084, y + 58),
             f"Nonzero joints {nonzero}/22  ·  executed {action['executed_steps']} steps", 18, AMBER)
        text(draw, (1084, y + 96), "Public rationale", 17, MUTED)
        for row, line in enumerate(fitted_lines(action["reason"], font(19), 474, 8)):
            draw.text((1084, y + 126 + row * 29), line, font=font(19), fill=INK)
        clipped = action.get("clipped_joint_indices", [])
        text(draw, (1084, y + 372), "Clipped joints: " + (str(clipped) if clipped else "none"), 17, MUTED)
    if index == len(samples) - 1:
        color = CYAN if result.get("signed_rotation_rad", 0.0) > 0 else AMBER
        text(draw, (1084, y + 390), "Recorded rollout end", 17, color)
    return canvas


def render_video(controller: Path, output: Path, fps: int, seconds_per_observation: float) -> dict:
    controller = controller.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing video: {output}")
    run, result, samples = load_samples(controller)
    trajectory_path = output.with_name("object_trajectory.json")
    manifest_path = output.with_name("video_manifest.json")
    poster_path = output.with_name("rollout_multiview_poster.png")
    if any(path.exists() for path in (trajectory_path, manifest_path, poster_path)):
        raise FileExistsError("Refusing existing video sidecar")
    write_json(
        trajectory_path,
        {
            "schema": "dexhand_astra.object_trajectory.v1",
            "coordinate_frame": "palm",
            "source": str(controller),
            "task": run["task"],
            "seed": run["seed"],
            "samples": [{key: value for key, value in sample.items() if key not in ("images", "last_action")}
                        for sample in samples],
        },
    )
    hold = max(1, round(fps * seconds_per_observation))
    with tempfile.TemporaryDirectory(prefix="dexhand_video_") as temporary:
        frames = Path(temporary)
        serial = 0
        final_frame = None
        for index in range(len(samples)):
            frame = render_frame(run, result, samples, index)
            final_frame = frame
            for _ in range(hold * (2 if index == len(samples) - 1 else 1)):
                frame.save(frames / f"{serial:06d}.png")
                serial += 1
        final_frame.save(poster_path)
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                "-framerate", str(fps), "-i", str(frames / "%06d.png"),
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-threads", "1", str(output),
            ],
            check=True,
        )
    manifest = {
        "schema": "dexhand_astra.video_manifest.v1",
        "status": "completed",
        "source": str(controller),
        "video": str(output),
        "poster": str(poster_path),
        "trajectory": str(trajectory_path),
        "fps": fps,
        "seconds_per_observation": seconds_per_observation,
        "observation_count": len(samples),
        "encoded_frame_count": serial,
        "interpolated_physics_frames": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seconds-per-observation", type=float, default=1.25)
    args = parser.parse_args()
    if args.fps < 1 or args.seconds_per_observation <= 0:
        parser.error("fps and seconds-per-observation must be positive")
    output = args.output or args.controller / "rollout_multiview.mp4"
    print(json.dumps(render_video(args.controller, output, args.fps, args.seconds_per_observation), indent=2))


if __name__ == "__main__":
    main()
