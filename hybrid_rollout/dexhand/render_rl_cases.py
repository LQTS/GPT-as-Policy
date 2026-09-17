#!/usr/bin/env python3
"""Render selected D3 RL policies from persistent fixed-world-axis cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from hybrid_rollout.dexhand.profiles import PROFILES, get_profile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ptrack-root", type=Path, required=True)
parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
parser.add_argument("--session", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--case-state", type=Path, nargs="+", required=True)
parser.add_argument("--output-name", default="rl_d3_world_z")
parser.add_argument("--video-length", type=int, default=600)
parser.add_argument("--intro-frames", type=int, default=15)
parser.add_argument("--outro-frames", type=int, default=15)
parser.add_argument("--camera-width", type=int, default=640)
parser.add_argument("--camera-height", type=int, default=480)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()

args.ptrack_root = args.ptrack_root.expanduser().resolve()
args.session = args.session.expanduser().resolve()
args.checkpoint = args.checkpoint.expanduser().resolve()
args.case_state = [path.expanduser().resolve() for path in args.case_state]
profile = get_profile(args.profile)
args.grasp_bank = profile.grasp_bank(args.ptrack_root)
args.task = profile.task
if profile.fixed_world_axis is None:
    parser.error("RL case rendering requires a fixed-world-axis profile.")
if args.video_length < 1 or args.intro_frames < 0 or args.outro_frames < 0:
    parser.error("Video length must be positive and hold-frame counts must be non-negative.")
for required in (args.ptrack_root, args.session, args.checkpoint, args.grasp_bank, *args.case_state):
    if not required.exists():
        parser.error(f"Required path does not exist: {required}")
if not (args.session / "params" / "agent.yaml").is_file():
    parser.error(f"Session agent config does not exist: {args.session / 'params' / 'agent.yaml'}")
if shutil.which("ffmpeg") is None:
    parser.error("ffmpeg is required to stream the annotated MP4.")

sys.path.insert(0, str(args.ptrack_root))
sys.path.insert(0, str(args.ptrack_root / "source" / "ConTrack"))
os.environ["CONTRACK_SHARPA_GRASP_BANK"] = str(args.grasp_bank)
args.enable_cameras = True
sys.argv = [sys.argv[0], *hydra_args]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401,E402
import ConTrack.tasks  # noqa: F401,E402
from rsl_rl_contrack.runners import DistillationRunner, OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from scripts.tools.sharpa_camera import camera_quat_opengl_wxyz  # noqa: E402
from ConTrack.tasks.manager_based.sharpa_in_hand_rotation.mdp.dynamic_target import (  # noqa: E402
    integrate_world_angular_velocity,
)

from hybrid_rollout.dexhand.camera_views import CAMERA_VIEWS  # noqa: E402
from hybrid_rollout.dexhand.case_state import apply_case_state, load_case_state  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_orientation_axes(prim_path: str) -> VisualizationMarkers:
    """Create RGB coordinate axes matching the established D3 videos."""
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
    return VisualizationMarkers(
        VisualizationMarkersCfg(prim_path=prim_path, markers=markers)
    )


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


class FfmpegWriter:
    """Stream RGB frames to an H.264 MP4 without buffering the episode in memory."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        command = [
            shutil.which("ffmpeg"),
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
        ]
        self.process = subprocess.Popen(
            command,
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


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = repo_root / "hybrid_rollout" / "assets" / "fonts" / "NotoSansCJKsc-Regular.otf"
    return ImageFont.truetype(path, size=size) if path.is_file() else ImageFont.load_default()


def _camera_frame(camera, *, case_id: str, step: int, rot_error: float) -> np.ndarray:
    rgb = camera.data.output["rgb"][0]
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.dtype != torch.uint8:
        rgb = rgb.to(torch.float32)
        if float(rgb.max().item()) <= 2.0:
            rgb = rgb * 255.0
        rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
    image = Image.fromarray(rgb.detach().cpu().numpy()).convert("RGB")
    header = 64
    output = Image.new("RGB", (image.width, image.height + header), (18, 18, 18))
    output.paste(image, (0, header))
    draw = ImageDraw.Draw(output)
    draw.text(
        (10, 5),
        f"D3 RL | {case_id} | step {step:04d} | rotation error {rot_error:.3f} rad",
        font=_font(16),
        fill=(245, 245, 245),
    )
    draw.text(
        (10, 34),
        "Lower RGB: OBJECT frame | Upper RGB: REFERENCE frame | target: WORLD +Z @ 1.0 rad/s",
        font=_font(14),
        fill=(245, 245, 245),
    )
    return np.asarray(output)


def _configure_camera(env_cfg) -> None:
    view = CAMERA_VIEWS[0]
    eye = np.asarray(view["eye"], dtype=np.float32)
    target = np.asarray(view["target"], dtype=np.float32)
    camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{view['prim_name']}",
        offset=CameraCfg.OffsetCfg(
            pos=tuple(float(value) for value in eye),
            rot=camera_quat_opengl_wxyz(eye, target),
            convention="opengl",
        ),
        data_types=["rgb"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=34.0,
            focus_distance=2.0,
            horizontal_aperture=24.0,
            clipping_range=(0.01, 20.0),
        ),
        width=args.camera_width,
        height=args.camera_height,
    )
    setattr(env_cfg.scene, view["scene_key"], camera)


def _configure_env(env_cfg, agent_cfg: dict) -> None:
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1
    env_cfg.log_dir = str(args.session)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.observations.critic.enable_corruption = False
    env_cfg.commands.rotation.grasp_bank_path = str(args.grasp_bank)
    env_cfg.commands.rotation.grasp_bank_probability = 1.0
    env_cfg.commands.rotation.grasp_sampling_mode = "state"
    env_cfg.commands.rotation.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.rotation.speed_stages = (profile.target_speed,)
    env_cfg.commands.rotation.min_speed_ratio = 1.0
    env_cfg.commands.rotation.success_tolerance = profile.success_tolerance
    env_cfg.events.reset_hand_root.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    agent_cfg["device"] = args.device
    agent_cfg["seed"] = args.seed
    _configure_camera(env_cfg)


def _stack(values: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    return np.stack(values, axis=0) if values else np.empty((0, *shape), dtype=np.float32)


def _use_fixed_world_velocity(command, velocity_w: torch.Tensor) -> None:
    """Keep the continuous target velocity fixed in the world frame."""

    def update_command(term) -> None:
        _, palm_quat_w = term._palm_pose_w()
        term.target_ang_vel_w.copy_(velocity_w.expand_as(term.target_ang_vel_w))
        term.target_ang_vel_p.copy_(
            math_utils.quat_apply_inverse(palm_quat_w, term.target_ang_vel_w)
        )
        term.target_quat_w.copy_(
            integrate_world_angular_velocity(
                term.target_quat_w,
                term.target_ang_vel_w,
                term._step_dt,
            )
        )

    command._update_command = types.MethodType(update_command, command)


@hydra_task_config(args.task, args.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    runner_cfg = yaml.safe_load((args.session / "params" / "agent.yaml").read_text())
    _configure_env(env_cfg, runner_cfg)
    gym_env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=runner_cfg["clip_actions"])
    runner_class = runner_cfg["class_name"]
    runner = (
        OnPolicyRunner(env, runner_cfg, log_dir=None, device=runner_cfg["device"])
        if runner_class == "OnPolicyRunner"
        else DistillationRunner(env, runner_cfg, log_dir=None, device=runner_cfg["device"])
    )
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw = env.unwrapped
    obj = raw.scene["object"]
    hand = raw.scene["hand"]
    command = raw.command_manager.get_term("rotation")
    action_term = raw.action_manager.get_term("joint_pos")
    camera = raw.scene[CAMERA_VIEWS[0]["scene_key"]]
    object_axes = _make_orientation_axes("/Visuals/DexHandObjectFrame")
    reference_axes = _make_orientation_axes("/Visuals/DexHandReferenceFrame")
    shaft_center = 0.055 / 2.0
    head_center = 0.055 + 0.018 / 2.0
    marker_offsets = torch.tensor(
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
    reference_lift = torch.tensor((0.0, 0.0, 0.11), device=raw.device)
    marker_indices = torch.arange(6, device=raw.device)
    expected_velocity = torch.tensor(
        profile.fixed_world_axis, dtype=torch.float32, device=raw.device
    ) * profile.target_speed
    _use_fixed_world_velocity(command, expected_velocity)
    checkpoint_sha256 = _sha256(args.checkpoint)
    fps = 1.0 / float(raw.step_dt)

    def update_axes() -> None:
        object_positions, object_orientations = _orientation_axis_poses(
            obj.data.root_pos_w,
            obj.data.root_quat_w,
            marker_offsets,
        )
        reference_positions, reference_orientations = _orientation_axis_poses(
            obj.data.root_pos_w + reference_lift,
            command.target_quat_w,
            marker_offsets,
        )
        object_axes.visualize(
            translations=object_positions,
            orientations=object_orientations,
            marker_indices=marker_indices,
        )
        reference_axes.visualize(
            translations=reference_positions,
            orientations=reference_orientations,
            marker_indices=marker_indices,
        )

    for case_path in args.case_state:
        case = load_case_state(case_path)
        if case["profile"] != args.profile:
            raise ValueError(
                f"Case {case_path} uses profile {case['profile']!r}, expected {args.profile!r}."
            )
        case_id = str(case["case_id"])
        output = case_path.parent / args.output_name
        result_path = output / "result.json"
        if result_path.is_file() and args.skip_existing:
            print(f"SKIP_EXISTING {result_path}", flush=True)
            continue
        if output.exists():
            raise FileExistsError(f"Case video output already exists: {output}")
        output.mkdir(parents=True)

        env.reset()
        apply_case_state(raw, command, action_term, case)
        raw.episode_length_buf.zero_()
        runner.alg.policy.reset(
            torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device)
        )
        obs = env.get_observations()
        if not bool(
            torch.allclose(
                command.target_ang_vel_w[0],
                expected_velocity,
                atol=1.0e-5,
                rtol=0.0,
            )
        ):
            raise RuntimeError("Restored command does not match the fixed world target.")

        video_path = output / "rollout.mp4"
        writer = FfmpegWriter(
            video_path,
            args.camera_width,
            args.camera_height + 64,
            fps,
        )
        joint_pos = []
        joint_vel = []
        object_root_state = []
        target_quat = []
        target_ang_vel_w = []
        actions = []
        rewards = []
        rotation_errors = []
        at_goal = []
        object_quats = []
        dropped = False
        non_finite = False
        timed_out = False
        last_frame = None

        try:
            for step in range(args.video_length):
                velocity_error = torch.linalg.vector_norm(
                    command.target_ang_vel_w[0] - expected_velocity
                )
                if float(velocity_error.item()) > 1.0e-5:
                    raise RuntimeError(
                        f"Fixed world target drifted by {float(velocity_error.item())}."
                    )
                rot_error = math_utils.quat_error_magnitude(
                    obj.data.root_quat_w,
                    command.target_quat_w,
                )[0]
                joint_pos.append(hand.data.joint_pos[0].detach().cpu().numpy().copy())
                joint_vel.append(hand.data.joint_vel[0].detach().cpu().numpy().copy())
                object_root_state.append(
                    obj.data.root_state_w[0].detach().cpu().numpy().copy()
                )
                target_quat.append(
                    command.target_quat_w[0].detach().cpu().numpy().copy()
                )
                target_ang_vel_w.append(
                    command.target_ang_vel_w[0].detach().cpu().numpy().copy()
                )
                rotation_errors.append(float(rot_error.item()))
                at_goal.append(float(rot_error.item()) < profile.success_tolerance)
                object_quats.append(obj.data.root_quat_w[0].detach().clone())

                update_axes()
                raw.sim.render()
                camera.update(dt=float(raw.step_dt))
                frame = _camera_frame(
                    camera,
                    case_id=case_id,
                    step=step,
                    rot_error=float(rot_error.item()),
                )
                if step == 0:
                    Image.fromarray(frame).save(output / "first_frame.png")
                    for _ in range(args.intro_frames):
                        writer.append(frame)
                writer.append(frame)
                last_frame = frame

                with torch.no_grad():
                    action = policy(obs)
                obs, reward, dones, extras = env.step(action)
                runner.alg.policy.reset(dones)
                actions.append(action[0].detach().cpu().numpy().copy())
                rewards.append(float(reward[0].item()))
                if bool(dones[0].item()):
                    dropped = bool(
                        raw.termination_manager.get_term("object_dropped")[0].item()
                    )
                    non_finite = bool(
                        raw.termination_manager.get_term("non_finite")[0].item()
                    )
                    time_outs = extras.get("time_outs")
                    timed_out = (
                        bool(time_outs[0].item()) if time_outs is not None else False
                    )
                    break
            if last_frame is not None:
                Image.fromarray(last_frame).save(output / "last_frame.png")
                for _ in range(args.outro_frames):
                    writer.append(last_frame)
        finally:
            writer.close()

        pose_signed = 0.0
        pose_absolute = 0.0
        pose_perpendicular = 0.0
        for previous, current in zip(object_quats, object_quats[1:]):
            delta = math_utils.quat_mul(current, math_utils.quat_conjugate(previous))
            rotation = math_utils.axis_angle_from_quat(delta)
            parallel = float(torch.dot(rotation, expected_velocity).item()) / profile.target_speed
            perpendicular = torch.linalg.vector_norm(
                rotation - parallel * expected_velocity / profile.target_speed
            )
            pose_signed += parallel
            pose_absolute += abs(parallel)
            pose_perpendicular += float(perpendicular.item())

        np.savez_compressed(
            output / "rollout_state.npz",
            joint_position=_stack(joint_pos, tuple(hand.data.joint_pos.shape[1:])),
            joint_velocity=_stack(joint_vel, tuple(hand.data.joint_vel.shape[1:])),
            object_root_state=_stack(object_root_state, (13,)),
            target_quaternion_wxyz=_stack(target_quat, (4,)),
            target_angular_velocity_world=_stack(target_ang_vel_w, (3,)),
            action=_stack(actions, (action_term.action_dim,)),
            reward=np.asarray(rewards, dtype=np.float32),
            rotation_error_rad=np.asarray(rotation_errors, dtype=np.float32),
            at_goal=np.asarray(at_goal, dtype=np.bool_),
        )
        steps = len(actions)
        result = {
            "schema": "dexhand.rl_case_video.v1",
            "case_id": case_id,
            "profile": args.profile,
            "source_case": str(case_path),
            "source": case["source"],
            "task": args.task,
            "policy": {
                "session": str(args.session),
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "inference": "deterministic",
            },
            "target": {
                "axis_frame": "world",
                "axis_unit_vector": list(profile.fixed_world_axis),
                "speed_rad_s": profile.target_speed,
            },
            "video": {
                "path": str(video_path),
                "width": args.camera_width,
                "height": args.camera_height + 64,
                "fps": fps,
                "simulation_frames": len(rotation_errors),
                "intro_frames": args.intro_frames,
                "outro_frames": args.outro_frames,
                "camera_view": CAMERA_VIEWS[0]["name"],
                "object_frame": "lower RGB axes",
                "reference_frame": "upper RGB axes",
            },
            "metrics": {
                "executed_steps": steps,
                "duration_s": steps * float(raw.step_dt),
                "at_goal_step_rate": float(np.mean(at_goal)) if at_goal else 0.0,
                "mean_rotation_error_rad": (
                    float(np.mean(rotation_errors)) if rotation_errors else 0.0
                ),
                "signed_rotation_rad": pose_signed,
                "absolute_axis_rotation_rad": pose_absolute,
                "perpendicular_rotation_rad": pose_perpendicular,
                "dropped": dropped,
                "non_finite": non_finite,
                "timed_out": timed_out,
            },
            "artifacts": {
                "video": str(video_path),
                "rollout_state": str(output / "rollout_state.npz"),
                "first_frame": str(output / "first_frame.png"),
                "last_frame": str(output / "last_frame.png"),
            },
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            "RL_CASE_COMPLETE "
            + json.dumps(
                {
                    "case_id": case_id,
                    "video": str(video_path),
                    **result["metrics"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
