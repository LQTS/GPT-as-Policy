"""Single-environment PTrack runtime presented to the Astra policy tools."""

from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
import torch
import isaaclab.utils.math as math_utils

from hybrid_rollout.robodojo.io import InputError, write_json

from .protocol import ACTION_DIM, accumulate_action, validate_action


def _values(tensor: torch.Tensor, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in tensor.detach().cpu().flatten().tolist()]


def _normalized_joint_positions(position: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    lower = limits[..., 0]
    upper = limits[..., 1]
    return (2.0 * (position - lower) / (upper - lower) - 1.0).clamp(-1.0, 1.0)


class DexHandRollout:
    """Expose a bounded decision loop around an already-created Isaac Lab environment."""

    def __init__(self, env, output: Path, *, task: str, seed: int, max_decisions: int) -> None:
        if max_decisions < 1:
            raise ValueError("max_decisions must be positive")
        self.env = env
        self.raw = env.unwrapped
        if self.raw.num_envs != 1:
            raise ValueError("Astra direct smoke requires exactly one environment")
        self.output = Path(output)
        (self.output / "observations").mkdir()
        self.task = task
        self.seed = seed
        self.max_decisions = max_decisions
        self.phase = "start"
        self.tick = 0
        self.request = None
        self.history = []
        self.result = None
        self.action_term = self.raw.action_manager.get_term("joint_pos")
        if self.action_term.action_dim != ACTION_DIM:
            raise ValueError(
                f"Expected {ACTION_DIM} finger actions, got {self.action_term.action_dim}"
            )
        self.joint_names = list(self.action_term._joint_names)
        self.command = self.raw.command_manager.get_term("rotation")
        self.camera = self.raw.scene["render_camera"]
        self.current_action = [0.0] * ACTION_DIM
        self.signed_angle = 0.0
        self.positive_angle = 0.0
        self.reverse_angle = 0.0
        self.perpendicular_angle = 0.0
        self.terminated = False
        self.dropped = False
        self.non_finite = False
        self.step_dt = float(self.raw.step_dt)
        self.max_episode_steps = int(round(self.raw.max_episode_length))

    def next_call(self) -> dict | None:
        if self.phase == "start":
            return {"tool": "dexhand_start"}
        if self.phase == "act":
            return {
                "tool": "dexhand_act",
                "request_id": self.request["request_id"],
                "observation_path": self.request["observation_path"],
            }
        return None

    def _require_phase(self, expected: str) -> None:
        if self.phase != expected:
            raise InputError(f"Expected phase {expected}, current phase is {self.phase}")

    def _render(self, directory: Path) -> dict:
        self.raw.sim.render()
        self.camera.update(dt=self.step_dt)
        rgb = self.camera.data.output["rgb"][0]
        if rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        if rgb.dtype != torch.uint8:
            rgb = rgb.to(torch.float32)
            if float(rgb.max().item()) <= 2.0:
                rgb = rgb * 255.0
            rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
        array = rgb.detach().cpu().numpy()
        path = directory / "front_rgb.png"
        Image.fromarray(array).save(path)
        if array.size == 0 or float(array.std()) < 1.0:
            raise RuntimeError("Rendered DexHand RGB frame is empty or nearly uniform")
        return {"name": "front", "path": str(path), "size": [array.shape[1], array.shape[0]]}

    def _contacts(self) -> dict[str, bool]:
        contacts = {}
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            positions = self.raw.scene[f"{finger}_contact"].data.contact_pos_w[:, 0]
            if positions.ndim == 3:
                positions = positions[:, 0]
            contacts[finger] = bool(torch.isfinite(positions).all(dim=-1)[0].item())
        return contacts

    def _state(self) -> dict:
        hand = self.raw.scene["hand"]
        obj = self.raw.scene["object"]
        ids = self.action_term._joint_ids
        palm_pos, palm_quat = self.command._palm_pose_w()
        object_pos_p = math_utils.quat_apply_inverse(
            palm_quat, obj.data.root_pos_w - palm_pos
        )
        object_quat_p = math_utils.quat_mul(
            math_utils.quat_conjugate(palm_quat), obj.data.root_quat_w
        )
        target_pos_p = math_utils.quat_apply_inverse(
            palm_quat, self.command.target_pos_w - palm_pos
        )
        target_quat_p = math_utils.quat_mul(
            math_utils.quat_conjugate(palm_quat), self.command.target_quat_w
        )
        object_linvel_p = math_utils.quat_apply_inverse(
            palm_quat, obj.data.root_lin_vel_w
        )
        object_angvel_p = math_utils.quat_apply_inverse(
            palm_quat, obj.data.root_ang_vel_w
        )
        target_axis_p = math_utils.quat_apply_inverse(
            palm_quat, self.command.target_ang_vel_w
        )
        metrics = {
            key: round(float(value[0].item()), 6)
            for key, value in self.command.metrics.items()
        }
        return {
            "joint_order": self.joint_names,
            "joint_position_rad": _values(hand.data.joint_pos[:, ids][0]),
            "joint_velocity_rad_s": _values(hand.data.joint_vel[:, ids][0]),
            "normalized_target": [round(value, 6) for value in self.current_action],
            "object_pose_palm": {
                "position_m": _values(object_pos_p[0]),
                "quaternion_wxyz": _values(object_quat_p[0]),
            },
            "target_pose_palm": {
                "position_m": _values(target_pos_p[0]),
                "quaternion_wxyz": _values(target_quat_p[0]),
            },
            "object_velocity_palm": {
                "linear_m_s": _values(object_linvel_p[0]),
                "angular_rad_s": _values(object_angvel_p[0]),
            },
            "target_axis_velocity_palm_rad_s": _values(target_axis_p[0]),
            "fingertip_contacts": self._contacts(),
            "native_command_metrics": metrics,
        }

    def _packet(self, *, result: dict | None = None) -> dict:
        index = len(self.history)
        directory = self.output / "observations" / f"{index:03d}"
        directory.mkdir()
        image = self._render(directory)
        state = self._state()
        np.savez_compressed(
            directory / "state.npz",
            joint_position=np.asarray(state["joint_position_rad"], dtype=np.float32),
            joint_velocity=np.asarray(state["joint_velocity_rad_s"], dtype=np.float32),
            normalized_target=np.asarray(state["normalized_target"], dtype=np.float32),
            object_position_palm=np.asarray(
                state["object_pose_palm"]["position_m"], dtype=np.float32
            ),
            object_quaternion_palm=np.asarray(
                state["object_pose_palm"]["quaternion_wxyz"], dtype=np.float32
            ),
        )
        packet = {
            "schema": "dexhand_astra.observation.v1",
            "task": self.task,
            "task_context": {
                "objective": (
                    "Continuously rotate the held object in the positive commanded object-local "
                    "axis direction while preserving the grasp and limiting perpendicular rotation."
                ),
                "object_local_axis": list(self.command.cfg.object_axis),
                "positive_direction": "right-hand rule",
                "target_angular_speed_rad_s": float(self.command.cfg.angular_speed),
                "episode_horizon_steps": self.max_episode_steps,
            },
            "seed": self.seed,
            "step_id": self.tick,
            "remaining_episode_steps": max(0, self.max_episode_steps - self.tick),
            "decisions_used": len(self.history),
            "max_decisions": self.max_decisions,
            "control_dt_s": self.step_dt,
            "state": state,
            "images": [image],
            "history": self.history,
            "rollout_finished": result is not None,
            "result": result,
        }
        observation_path = directory / "observation.json"
        write_json(observation_path, packet)
        if result is None:
            request_id = uuid.uuid4().hex
            self.request = {
                "request_id": request_id,
                "observation_path": str(observation_path),
            }
            packet.update(self.request)
            write_json(self.output / f"request_{index:03d}.json", {**packet, "images": [image]})
        return packet

    def start(self, **arguments) -> dict:
        self._require_phase("start")
        if arguments:
            raise InputError("dexhand_start takes no arguments")
        torch.manual_seed(self.seed)
        self.env.reset(seed=self.seed)
        position = self.raw.scene["hand"].data.joint_pos[:, self.action_term._joint_ids]
        normalized = _normalized_joint_positions(position, self.action_term._joint_limits)
        self.current_action = normalized[0].detach().cpu().tolist()
        self.phase = "act"
        write_json(
            self.output / "run.json",
            {
                "schema": "dexhand_astra.run.v1",
                "evaluation_method": "astra_direct_joint_delta",
                "task": self.task,
                "seed": self.seed,
                "action_dim": ACTION_DIM,
                "action_space": "normalized_joint_target_delta",
                "max_decisions": self.max_decisions,
                "control_dt_s": self.step_dt,
                "max_episode_steps": self.max_episode_steps,
                "observation_modalities": ["front_rgb", "named_proprio", "object_target_state"],
            },
        )
        write_json(self.output / "history.json", self.history)
        return self._packet()

    def _measure_step(self, angular_velocity_w: torch.Tensor, target_axis_w: torch.Tensor) -> None:
        axis = target_axis_w[0]
        axis = axis / torch.linalg.vector_norm(axis)
        velocity = angular_velocity_w[0]
        parallel = float(torch.dot(velocity, axis).item())
        perpendicular = float(torch.linalg.vector_norm(velocity - parallel * axis).item())
        angle = parallel * self.step_dt
        self.signed_angle += angle
        self.positive_angle += max(angle, 0.0)
        self.reverse_angle += max(-angle, 0.0)
        self.perpendicular_angle += perpendicular * self.step_dt

    def act(self, response: object) -> dict:
        self._require_phase("act")
        action = validate_action(response, self.request["request_id"])
        decision = len(self.history)
        target, clipped = accumulate_action(self.current_action, action["joint_delta"])
        self.current_action = target
        write_json(self.output / f"response_{decision:03d}.json", action)
        executed_steps = 0
        for _ in range(action["repeat_steps"]):
            angular_velocity = self.raw.scene["object"].data.root_ang_vel_w.clone()
            target_axis = self.command.target_ang_vel_w.clone()
            tensor = torch.tensor(
                [self.current_action], dtype=torch.float32, device=self.raw.device
            )
            self.env.step(tensor)
            self._measure_step(angular_velocity, target_axis)
            self.tick += 1
            executed_steps += 1
            terminated = bool(self.raw.reset_terminated[0].item())
            if terminated:
                self.terminated = True
                self.dropped = bool(
                    self.raw.termination_manager.get_term("object_dropped")[0].item()
                )
                self.non_finite = bool(
                    self.raw.termination_manager.get_term("non_finite")[0].item()
                )
                break
        self.history.append(
            {
                "decision": decision,
                "request_id": action["request_id"],
                "reason": action["reason"],
                "joint_delta": action["joint_delta"],
                "normalized_target": [round(value, 6) for value in self.current_action],
                "clipped_joint_indices": clipped,
                "requested_steps": action["repeat_steps"],
                "executed_steps": executed_steps,
                "end_step": self.tick,
                "terminated": self.terminated,
            }
        )
        write_json(self.output / "history.json", self.history)
        exhausted = len(self.history) >= self.max_decisions
        result = None
        if self.terminated or exhausted:
            result = self.finish("native_termination" if self.terminated else "decision_budget")
        return self._packet(result=result)

    def finish(self, reason: str) -> dict:
        if self.result is not None:
            return self.result
        total_axis = self.positive_angle + self.reverse_angle
        denominator = total_axis + self.perpendicular_angle
        self.result = {
            "schema": "dexhand_astra.result.v1",
            "reason": reason,
            "complete": True,
            "partial_horizon": self.tick < self.max_episode_steps,
            "steps": self.tick,
            "decisions": len(self.history),
            "survived_window": not self.terminated,
            "dropped": self.dropped,
            "non_finite": self.non_finite,
            "signed_rotation_rad": self.signed_angle,
            "signed_turns": self.signed_angle / (2.0 * math.pi),
            "reverse_rotation_fraction": self.reverse_angle / total_axis if total_axis else 0.0,
            "perpendicular_rotation_rad": self.perpendicular_angle,
            "axis_purity": total_axis / denominator if denominator else 0.0,
            "artifacts": {
                "run": str(self.output / "run.json"),
                "history": str(self.output / "history.json"),
                "observations": str(self.output / "observations"),
            },
        }
        self.phase = "done"
        write_json(self.output / "result.json", self.result)
        return self.result

    def close(self) -> None:
        self.env.close()


__all__ = ["DexHandRollout"]
