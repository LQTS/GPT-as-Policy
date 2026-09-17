#!/usr/bin/env python3
"""Generate persistent fixed-world-axis initial cases for DexHand evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from isaaclab.app import AppLauncher

from .profiles import PROFILES, get_profile


WORLD_Z_PROFILES = tuple(
    name for name, profile in PROFILES.items() if profile.fixed_world_axis == (0.0, 0.0, 1.0)
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ptrack-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", choices=WORLD_Z_PROFILES, required=True)
parser.add_argument("--num-cases", type=int, default=20)
parser.add_argument("--seed", type=int, default=20260917)
parser.add_argument("--camera-width", type=int, default=640)
parser.add_argument("--camera-height", type=int, default=480)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.ptrack_root = args.ptrack_root.expanduser().resolve()
args.output = args.output.expanduser().resolve()
profile = get_profile(args.profile)
args.grasp_bank = profile.grasp_bank(args.ptrack_root)
args.heldout_reference = profile.heldout_reference(args.ptrack_root)
args.task = profile.task
if args.num_cases < 1:
    parser.error("--num-cases must be positive")
if args.output.exists():
    parser.error(f"Output already exists: {args.output}")

sys.path.insert(0, str(args.ptrack_root))
sys.path.insert(0, str(args.ptrack_root / "source" / "ConTrack"))
os.environ["CONTRACK_SHARPA_GRASP_BANK"] = str(args.grasp_bank)
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
from PIL import Image
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sensors.camera import CameraCfg
from isaaclab_tasks.utils import load_cfg_from_registry

import isaaclab_tasks  # noqa: F401,E402
import ConTrack.tasks  # noqa: F401,E402
from ConTrack.tasks.manager_based.sharpa_in_hand_rotation.mdp.rotation_grasp_bank import (  # noqa: E402
    load_grasp_bank,
)
from scripts.tools.sharpa_camera import camera_quat_opengl_wxyz  # noqa: E402

from .camera_views import CAMERA_VIEWS  # noqa: E402
from .case_state import (  # noqa: E402
    capture_case_state,
    configure_fixed_world_axis,
    save_case_state,
)
from .case_visuals import (  # noqa: E402
    annotate_world_axes,
    make_case_preview,
    make_contact_sheet,
)
from hybrid_rollout.robodojo.io import write_json  # noqa: E402


def select_case_indices(bank: dict, count: int, seed: int) -> list[int]:
    """Select a deterministic group-diverse batch from a grasp bank."""
    groups = torch.as_tensor(bank["tensors"]["grasp_group"], dtype=torch.long)
    total = int(groups.numel())
    count = min(count, total)
    if count == total:
        return list(range(total))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected = []
    group_values = torch.unique(groups, sorted=True)
    group_order = group_values[torch.randperm(group_values.numel(), generator=generator)]
    for group in group_order:
        members = torch.nonzero(groups == group, as_tuple=False).flatten()
        chosen = members[torch.randint(members.numel(), (1,), generator=generator)]
        selected.append(int(chosen.item()))
        if len(selected) == count:
            return selected

    remaining = torch.tensor(
        [index for index in range(total) if index not in set(selected)], dtype=torch.long
    )
    order = torch.randperm(remaining.numel(), generator=generator)
    selected.extend(int(value) for value in remaining[order[: count - len(selected)]])
    return selected


def source_grasp_id(bank: dict, index: int) -> int | None:
    """Return the original grasp identifier when the bank records one."""
    values = bank["metadata"].get("source_grasp_ids")
    return int(values[index]) if values is not None else None


def heldout_source_ids(bank: dict | None) -> set[int]:
    """Return source IDs represented in an optional held-out reference bank."""
    if bank is None:
        return set()
    values = bank["metadata"].get("source_grasp_ids")
    return {int(value) for value in values} if values is not None else set()


def split_role(bank: dict, source_id: int | None, heldout_ids: set[int]) -> str:
    """Label whether a generated case is a true held-out state."""
    bank_role = str(bank["metadata"].get("split_role", "")).lower()
    if bank_role in {"test", "heldout", "held_out"}:
        return "heldout"
    if source_id is not None and source_id in heldout_ids:
        return "heldout"
    return "train_overlap"


def make_env():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.observations.policy.enable_corruption = False
    cfg.observations.critic.enable_corruption = False
    cfg.commands.rotation.grasp_bank_path = str(args.grasp_bank)
    cfg.commands.rotation.grasp_bank_probability = 1.0
    cfg.commands.rotation.grasp_sampling_mode = "state"
    cfg.commands.rotation.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.rotation.speed_stages = (profile.target_speed,)
    cfg.commands.rotation.min_speed_ratio = 1.0
    cfg.events.reset_hand_root.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    for view in CAMERA_VIEWS:
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
        setattr(cfg.scene, view["scene_key"], camera)
    return gym.make(args.task, cfg=cfg)


def render_views(raw, output: Path) -> dict[str, Path]:
    """Render and save all configured camera views."""
    raw.sim.render()
    paths = {}
    for view in CAMERA_VIEWS:
        camera = raw.scene[view["scene_key"]]
        camera.update(dt=float(raw.step_dt))
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
            raise RuntimeError(f"Rendered case view {view['name']!r} is empty.")
        raw_path = output / f"{view['name']}_rgb_raw.png"
        image = Image.fromarray(array)
        image.save(raw_path)
        path = output / f"{view['name']}_rgb.png"
        annotate_world_axes(
            image,
            view,
            target_axis=profile.fixed_world_axis,
        ).save(path)
        paths[view["name"]] = path
    return paths


def git_provenance(root: Path) -> dict:
    """Return exact PTrack source provenance."""
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"root": str(root), "commit": commit, "dirty": dirty}


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    bank = load_grasp_bank(args.grasp_bank, require_stable=True)
    heldout_bank = (
        load_grasp_bank(args.heldout_reference, require_stable=True)
        if args.heldout_reference is not None
        else None
    )
    heldout_ids = heldout_source_ids(heldout_bank)
    selected = select_case_indices(bank, args.num_cases, args.seed)

    env = make_env()
    raw = env.unwrapped
    env.reset(seed=args.seed)
    hand = raw.scene["hand"]
    obj = raw.scene["object"]
    command = raw.command_manager.get_term("rotation")
    action_term = raw.action_manager.get_term("joint_pos")
    env_ids = torch.tensor([0], dtype=torch.long, device=raw.device)
    nominal_hand_root = hand.data.root_state_w.clone()
    nominal_joint_pos = hand.data.joint_pos.clone()
    nominal_joint_vel = torch.zeros_like(hand.data.joint_vel)
    finger_ids = action_term._joint_ids
    tensors = bank["tensors"]
    entries = []

    for ordinal, bank_index in enumerate(selected):
        case_id = f"case_{ordinal:03d}"
        case_dir = args.output / case_id
        case_dir.mkdir()

        finger_pos = torch.as_tensor(
            tensors["finger_joint_pos"][bank_index],
            dtype=torch.float32,
            device=raw.device,
        ).unsqueeze(0)
        joint_pos = nominal_joint_pos.clone()
        joint_pos[:, finger_ids] = finger_pos
        hand.write_root_pose_to_sim(nominal_hand_root[:, :7], env_ids=env_ids)
        hand.write_root_velocity_to_sim(nominal_hand_root[:, 7:], env_ids=env_ids)
        hand.write_joint_state_to_sim(joint_pos, nominal_joint_vel, env_ids=env_ids)
        action_term.seed_joint_targets(finger_pos, env_ids)

        raw.sim.forward()
        palm_pos_w, palm_quat_w = command._palm_pose_w()
        object_pos_p = torch.as_tensor(
            tensors["object_pos_palm"][bank_index],
            dtype=torch.float32,
            device=raw.device,
        ).unsqueeze(0)
        object_quat_p = torch.as_tensor(
            tensors["object_quat_palm"][bank_index],
            dtype=torch.float32,
            device=raw.device,
        ).unsqueeze(0)
        object_pos_w = palm_pos_w + math_utils.quat_apply(palm_quat_w, object_pos_p)
        object_quat_w = math_utils.quat_mul(palm_quat_w, object_quat_p)
        obj.write_root_pose_to_sim(
            torch.cat((object_pos_w, object_quat_w), dim=-1), env_ids=env_ids
        )
        obj.write_root_velocity_to_sim(
            torch.zeros(1, 6, dtype=torch.float32, device=raw.device), env_ids=env_ids
        )
        raw.sim.forward()
        configure_fixed_world_axis(
            raw, command, profile.fixed_world_axis, profile.target_speed
        )

        source_id = source_grasp_id(bank, bank_index)
        role = split_role(bank, source_id, heldout_ids)
        stability = {
            key: float(torch.as_tensor(tensors[key][bank_index]).item())
            for key in (
                "hold_steps",
                "position_drift",
                "rotation_drift",
                "capture_linear_speed",
                "capture_angular_speed",
                "min_contacts",
            )
        }
        source = {
            "grasp_bank": str(args.grasp_bank),
            "grasp_bank_content_sha256": bank.get("content_sha256"),
            "bank_index": bank_index,
            "source_grasp_id": source_id,
            "grasp_group": int(torch.as_tensor(tensors["grasp_group"][bank_index]).item()),
            "split_role": role,
            "stability": stability,
        }
        case = capture_case_state(
            raw,
            command,
            action_term,
            case_id=case_id,
            profile_name=args.profile,
            source=source,
            axis=profile.fixed_world_axis,
            speed=profile.target_speed,
        )
        state_path = save_case_state(case, case_dir / "initial_state.pt")
        np.savez_compressed(
            case_dir / "state.npz",
            **{
                key: value.detach().cpu().numpy()
                for key, value in case["state"].items()
            },
        )
        images = render_views(raw, case_dir)
        preview = make_case_preview(
            case_id,
            images,
            case_dir,
            CAMERA_VIEWS,
            target_speed=profile.target_speed,
        )
        raw_images = {
            view["name"]: str(case_dir / f"{view['name']}_rgb_raw.png")
            for view in CAMERA_VIEWS
        }
        metadata = {
            "schema": "dexhand.initial_case.metadata.v1",
            "case_id": case_id,
            "profile": args.profile,
            "source": source,
            "target": case["target"],
            "artifacts": {
                "initial_state": str(state_path),
                "state_npz": str(case_dir / "state.npz"),
                "preview": str(preview),
                "images": {name: str(path) for name, path in images.items()},
                "raw_images": raw_images,
            },
        }
        write_json(case_dir / "metadata.json", metadata)
        entries.append(
            {
                "case_id": case_id,
                "bank_index": bank_index,
                "source_grasp_id": source_id,
                "grasp_group": source["grasp_group"],
                "split_role": role,
                "stability": stability,
                "state": str(state_path),
                "preview": str(preview),
                "images": {name: str(path) for name, path in images.items()},
                "raw_images": raw_images,
            }
        )

    contact_sheet = make_contact_sheet(entries, args.output)
    manifest = {
        "schema": "dexhand.initial_case_set.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile_name": args.profile,
        "profile": profile.record(args.ptrack_root),
        "selection": {
            "method": "deterministic_group_diverse_random",
            "seed": args.seed,
            "requested_count": args.num_cases,
            "generated_count": len(entries),
            "available_bank_states": int(tensors["finger_joint_pos"].shape[0]),
        },
        "target": {
            "axis_frame": "world",
            "axis_unit_vector": list(profile.fixed_world_axis),
            "speed_rad_s": profile.target_speed,
        },
        "ptrack": git_provenance(args.ptrack_root),
        "contact_sheet": str(contact_sheet),
        "visualization": {
            "coordinate_frame": "world",
            "axis_colors": {"X": "red", "Y": "green", "Z": "blue"},
            "target_axis": list(profile.fixed_world_axis),
            "target_axis_emphasized": True,
            "raw_images_preserved": True,
        },
        "cases": entries,
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
