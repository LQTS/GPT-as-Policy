"""Host-validated action contract for direct Sharpa finger control."""

from __future__ import annotations

import math

from hybrid_rollout.robodojo.io import InputError
from hybrid_rollout.robodojo.robodojo_server.validation import validate_public_language


ACTION_DIM = 22
MAX_JOINT_DELTA = 0.1
MAX_REPEAT_STEPS = 10


def _object(properties: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _repeat_bounds(fixed_repeat_steps: int | None) -> tuple[int, int]:
    if fixed_repeat_steps is None:
        return 1, MAX_REPEAT_STEPS
    if not 1 <= fixed_repeat_steps <= MAX_REPEAT_STEPS:
        raise ValueError(f"fixed_repeat_steps must be in [1, {MAX_REPEAT_STEPS}]")
    return fixed_repeat_steps, fixed_repeat_steps


def response_schema(fixed_repeat_steps: int | None = None) -> dict:
    """Return the dynamic-tool schema exposed to the policy agent."""
    repeat_minimum, repeat_maximum = _repeat_bounds(fixed_repeat_steps)
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
                "minimum": repeat_minimum,
                "maximum": repeat_maximum,
            },
            "reason": {"type": "string"},
        }
    )


def tool_specs(fixed_repeat_steps: int | None = None) -> list[dict]:
    """Return the two additive app-server tools for one DexHand rollout."""
    cadence = (
        f"exactly {fixed_repeat_steps} control steps"
        if fixed_repeat_steps is not None
        else "1-10 control steps"
    )
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
                f"Execute one bounded 22-joint delta for {cadence}, then return "
                "a fresh RGB/state observation and native metrics."
            ),
            "inputSchema": _object({"response": response_schema(fixed_repeat_steps)}),
        },
    ]


def validate_action(
    response: object, request_id: str, fixed_repeat_steps: int | None = None
) -> dict:
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
    repeat_minimum, repeat_maximum = _repeat_bounds(fixed_repeat_steps)
    if not repeat_minimum <= repeat_steps <= repeat_maximum:
        if fixed_repeat_steps is not None:
            raise InputError(f"repeat_steps must equal {fixed_repeat_steps} for this rollout")
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


__all__ = [
    "ACTION_DIM",
    "MAX_JOINT_DELTA",
    "MAX_REPEAT_STEPS",
    "accumulate_action",
    "response_schema",
    "tool_specs",
    "validate_action",
]
