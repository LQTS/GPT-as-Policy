"""Continuous annotated video for DexHand Astra rollouts."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


class FfmpegWriter:
    """Stream RGB frames to an H.264 MP4."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for DexHand rollout videos.")
        self.path = path
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def append(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input pipe is closed.")
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace")
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")


def _make_orientation_axes(prim_path: str) -> VisualizationMarkers:
    colors = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 0.85, 0.1),
        "z": (0.05, 0.25, 1.0),
    }
    shaft_length = 0.055
    cone_height = 0.018
    markers = {}
    for axis, color in colors.items():
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6)
        markers[f"{axis}_shaft"] = sim_utils.CylinderCfg(
            radius=0.003,
            height=shaft_length,
            axis=axis.upper(),
            visual_material=material,
        )
    for axis, color in colors.items():
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6)
        markers[f"{axis}_head"] = sim_utils.ConeCfg(
            radius=0.0075,
            height=cone_height,
            axis=axis.upper(),
            visual_material=material,
        )
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=prim_path, markers=markers))


def _orientation_axis_poses(
    base_positions: torch.Tensor,
    frame_orientations: torch.Tensor,
    local_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    marker_count = local_offsets.shape[0]
    rotations = frame_orientations[:, None, :].expand(-1, marker_count, -1).reshape(-1, 4)
    offsets = local_offsets[None, :, :].expand(base_positions.shape[0], -1, -1).reshape(-1, 3)
    positions = base_positions[:, None, :].expand(-1, marker_count, -1).reshape(-1, 3)
    positions = positions + math_utils.quat_apply(rotations, offsets)
    return positions, rotations


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    root = Path(__file__).resolve().parents[1]
    path = root / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
    return ImageFont.truetype(path, size=size) if path.is_file() else ImageFont.load_default()


def _rgb_array(camera) -> np.ndarray:
    rgb = camera.data.output["rgb"][0]
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.dtype != torch.uint8:
        rgb = rgb.to(torch.float32)
        if float(rgb.max().item()) <= 2.0:
            rgb = rgb * 255.0
        rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
    array = rgb.detach().cpu().numpy()
    if array.size == 0 or float(array.std()) < 1.0:
        raise RuntimeError("Rendered DexHand video frame is empty or nearly uniform.")
    return array


class OrientationVideoRecorder:
    """Record the front view with object and moving-reference coordinate frames."""

    def __init__(
        self,
        raw,
        camera,
        output: Path,
        *,
        target_axis: tuple[float, float, float],
        target_speed: float,
        intro_frames: int = 15,
        outro_frames: int = 15,
    ) -> None:
        self.raw = raw
        self.camera = camera
        self.output = Path(output)
        self.target_axis = target_axis
        self.target_speed = target_speed
        self.intro_frames = intro_frames
        self.outro_frames = outro_frames
        self.object = raw.scene["object"]
        self.command = raw.command_manager.get_term("rotation")
        self.object_axes = _make_orientation_axes("/Visuals/DexHandAstraObjectFrame")
        self.reference_axes = _make_orientation_axes("/Visuals/DexHandAstraReferenceFrame")
        shaft_center = 0.055 / 2.0
        head_center = 0.055 + 0.018 / 2.0
        self.marker_offsets = torch.tensor(
            (
                (shaft_center, 0.0, 0.0),
                (0.0, shaft_center, 0.0),
                (0.0, 0.0, shaft_center),
                (head_center, 0.0, 0.0),
                (0.0, head_center, 0.0),
                (0.0, 0.0, head_center),
            ),
            device=raw.device,
            dtype=torch.float32,
        )
        self.reference_lift = torch.tensor((0.0, 0.0, 0.11), device=raw.device)
        self.marker_indices = torch.arange(6, device=raw.device)
        self.writer = None
        self.last_frame = None
        self.frame_count = 0

    def update_axes(self) -> None:
        object_positions, object_orientations = _orientation_axis_poses(
            self.object.data.root_pos_w,
            self.object.data.root_quat_w,
            self.marker_offsets,
        )
        reference_positions, reference_orientations = _orientation_axis_poses(
            self.object.data.root_pos_w + self.reference_lift,
            self.command.target_quat_w,
            self.marker_offsets,
        )
        self.object_axes.visualize(
            translations=object_positions,
            orientations=object_orientations,
            marker_indices=self.marker_indices,
        )
        self.reference_axes.visualize(
            translations=reference_positions,
            orientations=reference_orientations,
            marker_indices=self.marker_indices,
        )

    def capture(self, *, case_id: str, step: int, rotation_error: float) -> None:
        self.update_axes()
        self.raw.sim.render()
        self.camera.update(dt=float(self.raw.step_dt))
        image = Image.fromarray(_rgb_array(self.camera)).convert("RGB")
        header = 64
        output = Image.new("RGB", (image.width, image.height + header), (18, 18, 18))
        output.paste(image, (0, header))
        draw = ImageDraw.Draw(output)
        draw.text(
            (10, 5),
            f"Astra | {case_id} | step {step:04d} | rotation error {rotation_error:.3f} rad",
            font=_font(16),
            fill=(245, 245, 245),
        )
        axis = ", ".join(f"{value:g}" for value in self.target_axis)
        draw.text(
            (10, 34),
            (
                "Lower RGB: OBJECT frame | Upper RGB: REFERENCE frame | "
                f"target: WORLD [{axis}] @ {self.target_speed:g} rad/s"
            ),
            font=_font(14),
            fill=(245, 245, 245),
        )
        frame = np.asarray(output)
        if self.writer is None:
            self.writer = FfmpegWriter(
                self.output / "rollout.mp4",
                output.width,
                output.height,
                1.0 / float(self.raw.step_dt),
            )
            Image.fromarray(frame).save(self.output / "first_frame.png")
            for _ in range(self.intro_frames):
                self.writer.append(frame)
        self.writer.append(frame)
        self.last_frame = frame
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is None:
            return
        if self.last_frame is not None:
            Image.fromarray(self.last_frame).save(self.output / "last_frame.png")
            for _ in range(self.outro_frames):
                self.writer.append(self.last_frame)
        self.writer.close()
        self.writer = None


__all__ = ["OrientationVideoRecorder"]
