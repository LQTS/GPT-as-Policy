"""Persistent initial-state snapshots for controlled DexHand evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import isaaclab.utils.math as math_utils


CASE_SCHEMA = "dexhand.initial_case.v1"


def normalized_axis(axis: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    """Return a finite unit-length three-vector."""
    value = torch.as_tensor(axis, dtype=torch.float64)
    if value.shape != (3,) or not bool(torch.isfinite(value).all()):
        raise ValueError("Rotation axis must contain three finite values.")
    norm = float(torch.linalg.vector_norm(value).item())
    if norm <= 0.0:
        raise ValueError("Rotation axis must be non-zero.")
    return tuple(float(item) for item in (value / norm).tolist())


def configure_fixed_world_axis(raw, command, axis, speed: float) -> None:
    """Set a continuous-rotation command to a fixed world-frame axis."""
    if speed <= 0.0:
        raise ValueError("Target speed must be positive.")
    unit_axis = torch.tensor(
        normalized_axis(axis), dtype=torch.float32, device=raw.device
    ).unsqueeze(0)
    target_ang_vel_w = unit_axis * float(speed)
    palm_pos_w, palm_quat_w = command._palm_pose_w()

    command.target_pos_w.copy_(palm_pos_w + command._goal_offset)
    command.target_quat_w.copy_(raw.scene["object"].data.root_quat_w)
    command.target_ang_vel_w.copy_(target_ang_vel_w)
    command.target_ang_vel_p.copy_(
        math_utils.quat_apply_inverse(palm_quat_w, target_ang_vel_w)
    )
    command._previous_object_quat_w.copy_(raw.scene["object"].data.root_quat_w)
    command._pose_delta_angular_velocity_w.zero_()
    command._pose_delta_step = -1
    command._episode_success.zero_()
    command._exclude_next_metric.fill_(True)


def capture_case_state(
    raw,
    command,
    action_term,
    *,
    case_id: str,
    profile_name: str,
    source: dict[str, Any],
    axis: tuple[float, float, float],
    speed: float,
) -> dict[str, Any]:
    """Capture all simulator and controller state needed to replay one initial case."""
    hand = raw.scene["hand"]
    obj = raw.scene["object"]
    state = {
        "hand_root_state_w": hand.data.root_state_w[0].detach().cpu().clone(),
        "hand_joint_pos": hand.data.joint_pos[0].detach().cpu().clone(),
        "hand_joint_vel": hand.data.joint_vel[0].detach().cpu().clone(),
        "object_root_state_w": obj.data.root_state_w[0].detach().cpu().clone(),
        "action_raw": action_term._raw_actions[0].detach().cpu().clone(),
        "action_processed": action_term._processed_actions[0].detach().cpu().clone(),
        "action_previous_targets": action_term._previous_targets[0].detach().cpu().clone(),
        "target_pos_w": command.target_pos_w[0].detach().cpu().clone(),
        "target_quat_w": command.target_quat_w[0].detach().cpu().clone(),
        "target_ang_vel_w": command.target_ang_vel_w[0].detach().cpu().clone(),
        "target_ang_vel_p": command.target_ang_vel_p[0].detach().cpu().clone(),
    }
    case = {
        "schema": CASE_SCHEMA,
        "case_id": case_id,
        "profile": profile_name,
        "source": dict(source),
        "target": {
            "axis_frame": "world",
            "axis_unit_vector": normalized_axis(axis),
            "speed_rad_s": float(speed),
        },
        "joint_order": list(action_term._joint_names),
        "state": state,
    }
    validate_case_state(case)
    return case


def validate_case_state(case: dict[str, Any]) -> None:
    """Validate a persistent initial-state snapshot."""
    if not isinstance(case, dict) or case.get("schema") != CASE_SCHEMA:
        raise ValueError(f"Unsupported DexHand case schema: {case.get('schema')!r}")
    target = case.get("target")
    if not isinstance(target, dict) or target.get("axis_frame") != "world":
        raise ValueError("DexHand case target must use the world axis frame.")
    normalized_axis(target.get("axis_unit_vector"))
    if float(target.get("speed_rad_s", 0.0)) <= 0.0:
        raise ValueError("DexHand case target speed must be positive.")

    state = case.get("state")
    if not isinstance(state, dict):
        raise TypeError("DexHand case state must be a dictionary.")
    expected_shapes = {
        "hand_root_state_w": (13,),
        "object_root_state_w": (13,),
        "target_pos_w": (3,),
        "target_quat_w": (4,),
        "target_ang_vel_w": (3,),
        "target_ang_vel_p": (3,),
    }
    for key, shape in expected_shapes.items():
        value = torch.as_tensor(state.get(key))
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"DexHand case field {key!r} must be finite with shape {shape}.")
    for key in (
        "hand_joint_pos",
        "hand_joint_vel",
        "action_raw",
        "action_processed",
        "action_previous_targets",
    ):
        value = torch.as_tensor(state.get(key))
        if value.ndim != 1 or value.numel() == 0 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"DexHand case field {key!r} must be a finite vector.")

    target_axis = torch.tensor(
        normalized_axis(target["axis_unit_vector"]), dtype=torch.float32
    )
    expected_velocity = target_axis * float(target["speed_rad_s"])
    saved_velocity = torch.as_tensor(state["target_ang_vel_w"], dtype=torch.float32)
    if not bool(torch.allclose(saved_velocity, expected_velocity, atol=1.0e-5, rtol=0.0)):
        raise ValueError("Saved target angular velocity does not match the case target.")


def save_case_state(case: dict[str, Any], path: Path) -> Path:
    """Validate and save a case snapshot."""
    validate_case_state(case)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(case, output)
    return output


def load_case_state(path: Path) -> dict[str, Any]:
    """Load and validate a case snapshot."""
    case_path = Path(path).expanduser().resolve()
    if not case_path.is_file():
        raise FileNotFoundError(f"DexHand case state does not exist: {case_path}")
    case = torch.load(case_path, map_location="cpu", weights_only=False)
    validate_case_state(case)
    return case


def apply_case_state(raw, command, action_term, case: dict[str, Any]) -> None:
    """Restore a captured state into a one-environment simulator."""
    validate_case_state(case)
    if raw.num_envs != 1:
        raise ValueError("DexHand case replay requires exactly one environment.")
    if list(case["joint_order"]) != list(action_term._joint_names):
        raise ValueError("DexHand case joint order does not match the current action term.")

    device = raw.device
    state = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for key, value in case["state"].items()
    }
    hand = raw.scene["hand"]
    obj = raw.scene["object"]
    env_ids = torch.tensor([0], dtype=torch.long, device=device)

    if state["hand_joint_pos"].shape != hand.data.joint_pos.shape:
        raise ValueError("DexHand case hand joint dimension does not match the current task.")
    if state["action_previous_targets"].shape != action_term._previous_targets.shape:
        raise ValueError("DexHand case action dimension does not match the current task.")

    hand.write_root_pose_to_sim(state["hand_root_state_w"][:, :7], env_ids=env_ids)
    hand.write_root_velocity_to_sim(state["hand_root_state_w"][:, 7:], env_ids=env_ids)
    hand.write_joint_state_to_sim(
        state["hand_joint_pos"], state["hand_joint_vel"], env_ids=env_ids
    )
    obj.write_root_pose_to_sim(state["object_root_state_w"][:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(state["object_root_state_w"][:, 7:], env_ids=env_ids)

    action_term._raw_actions.copy_(state["action_raw"])
    action_term._processed_actions.copy_(state["action_processed"])
    action_term._previous_targets.copy_(state["action_previous_targets"])
    command.target_pos_w.copy_(state["target_pos_w"])
    command.target_quat_w.copy_(state["target_quat_w"])
    command.target_ang_vel_w.copy_(state["target_ang_vel_w"])
    command.target_ang_vel_p.copy_(state["target_ang_vel_p"])
    command._previous_object_quat_w.copy_(state["object_root_state_w"][:, 3:7])
    command._pose_delta_angular_velocity_w.zero_()
    command._pose_delta_step = -1
    command._episode_success.zero_()
    command._exclude_next_metric.fill_(True)
    raw.sim.forward()


__all__ = [
    "CASE_SCHEMA",
    "apply_case_state",
    "capture_case_state",
    "configure_fixed_world_axis",
    "load_case_state",
    "normalized_axis",
    "save_case_state",
    "validate_case_state",
]
