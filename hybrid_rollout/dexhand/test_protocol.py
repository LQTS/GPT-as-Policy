import math
from pathlib import Path

import pytest

from hybrid_rollout.robodojo.io import InputError

from .protocol import ACTION_DIM, accumulate_action, tool_specs, validate_action


def valid_response():
    return {
        "request_id": "request",
        "joint_delta": [0.0] * ACTION_DIM,
        "repeat_steps": 3,
        "reason": "The grasp is stable; test a small coordinated flexion change.",
    }


def test_valid_action_and_saturation_report():
    action = validate_action(valid_response(), "request")
    assert action["repeat_steps"] == 3
    target, clipped = accumulate_action([0.95] * ACTION_DIM, [0.1] * ACTION_DIM)
    assert target == [1.0] * ACTION_DIM
    assert clipped == list(range(ACTION_DIM))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(request_id="stale"),
        lambda value: value.update(joint_delta=[0.0] * (ACTION_DIM - 1)),
        lambda value: value["joint_delta"].__setitem__(0, 0.1001),
        lambda value: value["joint_delta"].__setitem__(0, math.nan),
        lambda value: value.update(repeat_steps=True),
        lambda value: value.update(repeat_steps=11),
        lambda value: value.update(extra="not allowed"),
    ],
)
def test_invalid_actions_are_rejected_before_execution(mutate):
    response = valid_response()
    mutate(response)
    with pytest.raises(InputError):
        validate_action(response, "request")


def test_dynamic_tools_expose_only_start_and_bounded_act():
    specs = tool_specs()
    assert [spec["name"] for spec in specs] == ["dexhand_start", "dexhand_act"]
    delta = specs[1]["inputSchema"]["properties"]["response"]["properties"]["joint_delta"]
    assert delta["minItems"] == delta["maxItems"] == ACTION_DIM


def test_policy_inherits_pinned_project_model_without_rpc_dependency():
    from .policy import EFFORT, MODEL, agent_config

    assert (MODEL, EFFORT) == ("gpt-6-astra", "xhigh")
    config = agent_config(Path("/tmp/audit"), Path("/tmp/agent"))
    assert config["model"] == MODEL
    assert config["model_reasoning_effort"] == EFFORT
