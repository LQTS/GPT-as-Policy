"""Host-validated action contract for direct Sharpa finger control."""

from __future__ import annotations

import math

from hybrid_rollout.robodojo.io import InputError
from hybrid_rollout.robodojo.robodojo_server.validation import validate_public_language


ACTION_DIM = 22
MAX_JOINT_DELTA = 0.1
MAX_REPEAT_STEPS = 5


def _object(properties: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def response_schema() -> dict:
    """Return the dynamic-tool schema exposed to the policy agent."""
    return _object(
        {
            "request_id": {"type": "string"},
            "joint_delta": {
                "type": "array",
                "items": {
                    "type": "number",
                    "minimum": -MAX_JOINT_DELTA,
                    "maximum": MAX_JOINT_DELTA,
                },
                "minItems": ACTION_DIM,
                "maxItems": ACTION_DIM,
            },
            "repeat_steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_REPEAT_STEPS,
            },
            "reason": {"type": "string"},
        }
    )


def tool_specs() -> list[dict]:
    """Return the two additive app-server tools for one DexHand rollout."""
    return [
        {
            "type": "function",
            "name": "dexhand_start",
            "description": "Start the single authorized Sharpa episode and return RGB plus named state.",
            "inputSchema": _object({}),
        },
        {
            "type": "function",
            "name": "dexhand_act",
            "description": (
                "Execute one bounded 22-joint delta for 1-5 control steps, then return "
                "a fresh RGB/state observation and native metrics."
            ),
            "inputSchema": _object({"response": response_schema()}),
        },
    ]


def validate_action(response: object, request_id: str) -> dict:
    """Validate a model-authored action before any simulator step occurs."""
    if not isinstance(response, dict):
        raise InputError("response must be an object")
    required = {"request_id", "joint_delta", "repeat_steps", "reason"}
    if set(response) != required:
        raise InputError(f"response fields must be exactly {sorted(required)}")
    if response["request_id"] != request_id:
        raise InputError("response request_id does not match the current observation")
    reason = response["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise InputError("reason must contain brief visible evidence and action purpose")
    try:
        validate_public_language(response)
    except ValueError as error:
        raise InputError(str(error)) from error
    repeat_steps = response["repeat_steps"]
    if isinstance(repeat_steps, bool) or not isinstance(repeat_steps, int):
        raise InputError("repeat_steps must be an integer")
    if not 1 <= repeat_steps <= MAX_REPEAT_STEPS:
        raise InputError(f"repeat_steps must be in [1, {MAX_REPEAT_STEPS}]")
    delta = response["joint_delta"]
    if not isinstance(delta, list) or len(delta) != ACTION_DIM:
        raise InputError(f"joint_delta must contain exactly {ACTION_DIM} values")
    values = []
    for index, value in enumerate(delta):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputError(f"joint_delta[{index}] must be a number")
        value = float(value)
        if not math.isfinite(value):
            raise InputError(f"joint_delta[{index}] must be finite")
        if abs(value) > MAX_JOINT_DELTA + 1.0e-12:
            raise InputError(
                f"joint_delta[{index}]={value} exceeds +/-{MAX_JOINT_DELTA}"
            )
        values.append(value)
    return {
        "request_id": request_id,
        "joint_delta": values,
        "repeat_steps": repeat_steps,
        "reason": reason.strip(),
    }


def accumulate_action(current: list[float], delta: list[float]) -> tuple[list[float], list[int]]:
    """Accumulate a bounded delta and report any absolute-target saturation."""
    if len(current) != ACTION_DIM or len(delta) != ACTION_DIM:
        raise ValueError(f"current and delta must both have length {ACTION_DIM}")
    target = []
    clipped = []
    for index, (old, change) in enumerate(zip(current, delta, strict=True)):
        value = float(old) + float(change)
        bounded = min(1.0, max(-1.0, value))
        if bounded != value:
            clipped.append(index)
        target.append(bounded)
    return target, clipped


def rotation_target_context(angular_velocity: list[float]) -> dict:
    """Describe one non-zero palm-frame angular-velocity command."""
    if not isinstance(angular_velocity, list) or len(angular_velocity) != 3:
        raise ValueError("angular_velocity must contain exactly three values")
    values = [float(value) for value in angular_velocity]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("angular_velocity must be finite")
    speed = math.sqrt(sum(value * value for value in values))
    if speed <= 0.0:
        raise ValueError("angular_velocity must be non-zero")
    return {
        "axis_frame": "palm",
        "axis_unit_vector": [round(value / speed, 6) for value in values],
        "angular_velocity_rad_s": [round(value, 6) for value in values],
        "angular_speed_rad_s": round(speed, 6),
        "positive_direction": "right-hand rule",
    }


__all__ = [
    "ACTION_DIM",
    "MAX_JOINT_DELTA",
    "MAX_REPEAT_STEPS",
    "accumulate_action",
    "rotation_target_context",
    "response_schema",
    "tool_specs",
    "validate_action",
]
