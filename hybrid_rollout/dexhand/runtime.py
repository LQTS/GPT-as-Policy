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

from ConTrack.tasks.manager_based.sharpa_in_hand_rotation.evaluation import (
    DynamicRotationEvalAccumulator,
    quaternion_geodesic_error,
)
from ConTrack.tasks.manager_based.sharpa_in_hand_rotation.mdp.dynamic_target import (
    signed_dominant_axis_bins,
)
from hybrid_rollout.robodojo.io import InputError, write_json

from .camera_views import CAMERA_VIEWS
from .protocol import (
    ACTION_DIM,
    accumulate_action,
    rotation_target_context,
    validate_action,
)


def _values(tensor: torch.Tensor, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in tensor.detach().cpu().flatten().tolist()]


def _normalized_joint_positions(position: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    lower = limits[..., 0]
    upper = limits[..., 1]
    return (2.0 * (position - lower) / (upper - lower) - 1.0).clamp(-1.0, 1.0)


class DexHandRollout:
    """Expose a bounded decision loop around an already-created Isaac Lab environment."""

    def __init__(
        self,
        env,
        output: Path,
        *,
        task: str,
        seed: int,
        max_decisions: int,
        success_tolerance: float = 0.1,
        warmup_steps: int = 20,
        provenance: dict | None = None,
    ) -> None:
        if max_decisions < 1:
            raise ValueError("max_decisions must be positive")
        self.env = env
        self.raw = env.unwrapped
        if self.raw.num_envs != 1:
            raise ValueError("Astra direct evaluation requires exactly one environment")
        self.output = Path(output)
        (self.output / "observations").mkdir()
        self.task = task
        self.seed = seed
        self.max_decisions = max_decisions
        self.provenance = dict(provenance or {})
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
        self.cameras = [
            (view["name"], self.raw.scene[view["scene_key"]]) for view in CAMERA_VIEWS
        ]
        self.current_action = [0.0] * ACTION_DIM
        self.signed_angle = 0.0
        self.positive_angle = 0.0
        self.reverse_angle = 0.0
        self.perpendicular_angle = 0.0
        self.terminated = False
        self.timed_out = False
        self.dropped = False
        self.non_finite = False
        self.step_dt = float(self.raw.step_dt)
        self.max_episode_steps = int(round(self.raw.max_episode_length))
        self.metrics = DynamicRotationEvalAccumulator(
            num_envs=1,
            device=self.raw.device,
            step_dt=self.step_dt,
            success_tolerance=success_tolerance,
            warmup_steps=warmup_steps,
        )

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

    def _render(self, directory: Path) -> list[dict]:
        self.raw.sim.render()
        images = []
        for name, camera in self.cameras:
            camera.update(dt=self.step_dt)
            rgb = camera.data.output["rgb"][0]
            if rgb.shape[-1] > 3:
                rgb = rgb[..., :3]
            if rgb.dtype != torch.uint8:
                rgb = rgb.to(torch.float32)
                if float(rgb.max().item()) <= 2.0:
                    rgb = rgb * 255.0
                rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
            array = rgb.detach().cpu().numpy()
            path = directory / f"{name}_rgb.png"
            Image.fromarray(array).save(path)
            if array.size == 0 or float(array.std()) < 1.0:
                raise RuntimeError(f"Rendered DexHand RGB frame {name!r} is empty or nearly uniform")
            images.append(
                {"name": name, "path": str(path), "size": [array.shape[1], array.shape[0]]}
            )
        return images

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
        target_axis_p = self.command.target_ang_vel_p
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
            "fingertip_contacts_valid": self.tick > 0,
            "fingertip_contacts": self._contacts(),
            "native_command_metrics_valid": self.tick > 0,
            "native_command_metrics": metrics,
        }

    def _packet(self, *, result: dict | None = None) -> dict:
        index = len(self.history)
        directory = self.output / "observations" / f"{index:03d}"
        directory.mkdir()
        images = self._render(directory)
        state = self._state()
        target_context = rotation_target_context(
            state["target_axis_velocity_palm_rad_s"]
        )
        if hasattr(self.command.cfg, "object_axis"):
            axis = [float(value) for value in self.command.cfg.object_axis]
            norm = math.sqrt(sum(value * value for value in axis))
            target_context.update(
                configured_axis_frame="initialized_object",
                configured_axis_unit_vector=[round(value / norm, 6) for value in axis],
            )
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
                    "Track the continuously moving target orientation while preserving the grasp. "
                    "The current target angular velocity is expressed in the palm frame."
                ),
                "rotation_target": target_context,
                "episode_horizon_steps": self.max_episode_steps,
            },
            "seed": self.seed,
            "step_id": self.tick,
            "remaining_episode_steps": max(0, self.max_episode_steps - self.tick),
            "decisions_used": len(self.history),
            "max_decisions": self.max_decisions,
            "control_dt_s": self.step_dt,
            "state": state,
            "images": images,
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
            write_json(self.output / f"request_{index:03d}.json", packet)
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
                "success_tolerance_rad": self.metrics.success_tolerance,
                "warmup_steps": self.metrics.warmup_steps,
                "ptrack": self.provenance,
                "camera_views": [dict(view) for view in CAMERA_VIEWS],
                "observation_modalities": [
                    "front_rgb",
                    "opposite_rgb",
                    "top_rgb",
                    "named_proprio",
                    "object_target_state",
                ],
            },
        )
        write_json(self.output / "history.json", self.history)
        return self._packet()

    def _measure_step(self, angular_velocity_w: torch.Tensor, target_axis_w: torch.Tensor) -> None:
        axis = target_axis_w[0]
        norm = torch.linalg.vector_norm(axis)
        if float(norm.item()) <= 0.0:
            return
        axis = axis / norm
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
            target_axis = math_utils.quat_apply(
                self.command._palm_pose_w()[1], self.command.target_ang_vel_p
            )
            rot_dist = quaternion_geodesic_error(
                self.raw.scene["object"].data.root_quat_w,
                self.command.target_quat_w,
            )
            target_speed = torch.linalg.vector_norm(
                self.command.target_ang_vel_p, dim=-1
            )
            axis_bin, moving_target = signed_dominant_axis_bins(
                self.command.target_ang_vel_p
            )
            tensor = torch.tensor(
                [self.current_action], dtype=torch.float32, device=self.raw.device
            )
            _, _, terminated, truncated, _ = self.env.step(tensor)
            self._measure_step(angular_velocity, target_axis)
            self.tick += 1
            executed_steps += 1
            terminated_now = bool(terminated[0].item())
            truncated_now = bool(truncated[0].item())
            object_dropped = self.raw.termination_manager.get_term(
                "object_dropped"
            ).clone()
            non_finite = self.raw.termination_manager.get_term("non_finite").clone()
            self.metrics.step(
                rot_dist,
                target_speed,
                axis_bin,
                moving_target,
                terminated,
                object_dropped=object_dropped,
                non_finite=non_finite,
            )
            if terminated_now or truncated_now:
                self.terminated = terminated_now
                self.timed_out = truncated_now
                self.dropped = bool(object_dropped[0].item())
                self.non_finite = bool(non_finite[0].item())
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
                "timed_out": self.timed_out,
            }
        )
        write_json(self.output / "history.json", self.history)
        exhausted = len(self.history) >= self.max_decisions
        result = None
        if self.terminated or self.timed_out or exhausted:
            if self.terminated:
                reason = "native_termination"
            elif self.timed_out:
                reason = "native_timeout"
            else:
                reason = "decision_budget"
            result = self.finish(reason)
        return self._packet(result=result)

    def finish(self, reason: str) -> dict:
        if self.result is not None:
            return self.result
        total_axis = self.positive_angle + self.reverse_angle
        denominator = total_axis + self.perpendicular_angle
        dynamic_metrics = self.metrics.finalize()
        self.result = {
            "schema": "dexhand_astra.result.v1",
            "reason": reason,
            "complete": True,
            "partial_horizon": self.tick < self.max_episode_steps,
            "native_episode_complete": self.terminated or self.timed_out,
            "steps": self.tick,
            "decisions": len(self.history),
            "survived_window": not self.terminated,
            "dropped": self.dropped,
            "non_finite": self.non_finite,
            "timed_out": self.timed_out,
            "signed_rotation_rad": self.signed_angle,
            "signed_turns": self.signed_angle / (2.0 * math.pi),
            "reverse_rotation_fraction": self.reverse_angle / total_axis if total_axis else 0.0,
            "perpendicular_rotation_rad": self.perpendicular_angle,
            "axis_purity": total_axis / denominator if denominator else 0.0,
            "dynamic_rotation_metrics": dynamic_metrics,
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
