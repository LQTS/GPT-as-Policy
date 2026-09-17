"""Persistent Codex app-server driver for the DexHand Astra rollout."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from hybrid_rollout.robodojo.io import InputError, write_json
from hybrid_rollout.robodojo.settings import (
    DEFAULT_CODEX_IMAGE_MAX_EDGE,
    EFFORT,
    MODEL,
    PROVIDER,
)
from hybrid_rollout.robodojo.skill.image_preview import image_max_edge, prepare_image
from hybrid_rollout.robodojo.skill.network_recovery import (
    NETWORK_CONTINUE_DELAYS,
    closed_network_turn,
    is_network_error,
)
from hybrid_rollout.robodojo.skill.transport import StdioAppServer

from .protocol import tool_specs


SKILL_ROOT = Path(__file__).parent / "dexhand-astra-rollout"
CONTROLLER_VERSION = "dexhand_direct_smoke_v1"
NATIVE_WORK_ITEMS = frozenset(
    (
        "commandExecution",
        "fileChange",
        "imageView",
        "webSearch",
        "mcpToolCall",
        "collabAgentToolCall",
    )
)


def toml_value(value: object) -> str:
    """Encode a value for the Codex CLI's TOML `-c` syntax."""
    if isinstance(value, dict):
        return "{" + ", ".join(
            json.dumps(key) + " = " + toml_value(item) for key, item in value.items()
        ) + "}"
    return json.dumps(value)


def agent_config(audit: Path, agent: Path) -> dict:
    """Use the same pinned app-server configuration as the project rollout."""
    state_root = Path(
        os.environ.get("ROLLOUT_CODEX_STATE_DIR", os.environ.get("CODEX_HOME", str(audit)))
    )
    return {
        "model": MODEL,
        "model_provider": PROVIDER,
        "model_reasoning_effort": EFFORT,
        "sqlite_home": str(state_root / "runtime_db"),
        "log_dir": str(state_root / "runtime_logs"),
        "default_permissions": "rollout_agent",
        "features.shell_tool": True,
        "features.view_image": True,
        "shell_environment_policy.inherit": "all",
        "permissions.rollout_agent.extends": ":workspace",
        "permissions.rollout_agent.filesystem": {
            str(audit.parent): "read",
            str(agent): "write",
        },
    }


def content_items(packet: dict, *, images: bool, max_image_edge: int) -> list[dict]:
    """Attach RGB previews without changing the recorded full-resolution image."""
    visible_packet = packet
    attachments = []
    if images:
        visible_packet = dict(packet, images=[])
        for item in packet.get("images", []):
            descriptor, payload = prepare_image(item, max_image_edge)
            visible_packet["images"].append(descriptor)
            data = base64.b64encode(payload).decode("ascii")
            attachments.append(
                {"type": "inputImage", "imageUrl": "data:image/png;base64," + data}
            )
    return [
        {"type": "inputText", "text": json.dumps(visible_packet, separators=(",", ":"))},
        *attachments,
    ]


def prepare_workspace(audit: Path) -> Path:
    """Create the writable policy workspace and install its local rollout skill."""
    audit = audit.resolve()
    agent = audit / "agent"
    agent.mkdir()
    (agent / "scratch").mkdir()
    context = agent / "context"
    context.mkdir()
    shutil.copy2(SKILL_ROOT / "references" / "action_contract.md", context / "action_contract.md")
    local_skill = agent / ".agents" / "skills" / "dexhand-astra-rollout"
    (local_skill / "references").mkdir(parents=True)
    shutil.copy2(SKILL_ROOT / "SKILL.md", local_skill / "SKILL.md")
    shutil.copy2(
        SKILL_ROOT / "references" / "action_contract.md",
        local_skill / "references" / "action_contract.md",
    )
    write_json(
        agent / "workspace.json",
        {
            "controller_output": str(audit.parent),
            "robot_profile": "sharpa_left_22d",
            "evaluation_method": "astra_direct_joint_delta",
            "observations_path": str(audit.parent / "observations"),
            "history_path": str(audit.parent / "history.json"),
            "notes_path": str(agent / "NOTES.md"),
            "scratch": str(agent / "scratch"),
            "baseline_full_conversation_available": False,
        },
    )
    (agent / "AGENTS.md").write_text(
        "# DexHand policy workspace\n\n"
        "Use the installed dexhand-astra-rollout skill for this single episode. "
        "Read context/action_contract.md and workspace.json. Host-owned observations, "
        "requests, responses, and histories are read-only; write only NOTES.md or scratch/. "
        "Do not start another simulator or reset/replay physical actions. Use English for "
        "public output and do not expose private chain-of-thought.\n"
    )
    (agent / "NOTES.md").write_text(
        "# Episode working memory\n\nNo observations yet. Record only concise confirmed state and corrections.\n"
    )
    return agent


def rejected_input(error: InputError, rollout) -> dict:
    """Return correction context for a call rejected before physics execution."""
    packet = {
        "error": str(error),
        "error_type": "recoverable_tool_input",
        "no_execution": True,
        "retryable": True,
        "step_id": rollout.tick,
        "next_call": rollout.next_call(),
        "correction": "Correct the arguments and retry the current call; do not reset the episode.",
    }
    if rollout.request is not None:
        packet["request_id"] = rollout.request["request_id"]
    return packet


class DexHandCodexPolicy:
    """Run the pinned project model/settings with DexHand-specific dynamic tools."""

    def __init__(
        self,
        workspace: Path,
        codex: str,
        *,
        timeout: int = 900,
        transport_factory=StdioAppServer,
    ) -> None:
        self.image_max_edge = image_max_edge(
            os.environ.get("CODEX_IMAGE_MAX_EDGE", str(DEFAULT_CODEX_IMAGE_MAX_EDGE))
        )
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=False)
        self.agent_workspace = prepare_workspace(self.workspace)
        self.timeout = timeout
        skill = (SKILL_ROOT / "SKILL.md").read_text()
        contract = (SKILL_ROOT / "references" / "action_contract.md").read_text()
        self.prompt = (
            skill
            + "\n\n# Action contract\n\n"
            + contract
            + "\n\nAgent working directory: "
            + str(self.agent_workspace)
            + "\nResolve context/ and workspace.json relative to that directory.\n"
        )
        self.prompt_sha256 = hashlib.sha256(self.prompt.encode()).hexdigest()
        (self.workspace / "SKILL.md").write_text(skill)
        (self.workspace / "PROMPT.md").write_text(self.prompt)
        specs = tool_specs()
        write_json(self.workspace / "tools.json", specs)
        version = subprocess.run(
            [codex, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        config = agent_config(self.workspace, self.agent_workspace)
        argv = [codex, "app-server", "--stdio", "--strict-config"]
        for key, value in config.items():
            argv += ["-c", key + "=" + toml_value(value)]
        write_json(self.workspace / "launch.json", {"argv": argv, "config": config})
        self.transport = transport_factory(argv, self.workspace)
        try:
            self.transport.request(
                "initialize",
                {
                    "clientInfo": {"name": "dexhand-astra-policy", "version": "1"},
                    "capabilities": {"experimentalApi": True},
                },
                timeout,
            )
            self.transport.notify("initialized", {})
            response = self.transport.request(
                "thread/start",
                {
                    "cwd": str(self.agent_workspace),
                    "model": MODEL,
                    "modelProvider": PROVIDER,
                    "config": {"model_reasoning_effort": EFFORT},
                    "developerInstructions": self.prompt,
                    "dynamicTools": specs,
                    "ephemeral": False,
                    "allowProviderModelFallback": False,
                    "approvalPolicy": "never",
                    "approvalsReviewer": "auto_review",
                    "permissions": "rollout_agent",
                    "runtimeWorkspaceRoots": [str(self.agent_workspace)],
                },
                timeout,
            )
            if response.get("model") != MODEL or response.get("reasoningEffort") != EFFORT:
                raise RuntimeError(
                    f"Codex model/effort differs from required {MODEL}/{EFFORT}: "
                    f"{response.get('model')}/{response.get('reasoningEffort')}"
                )
            self.thread_id = response["thread"]["id"]
            write_json(
                self.workspace / "worker.json",
                {
                    "model": MODEL,
                    "model_provider": PROVIDER,
                    "reasoning_effort": EFFORT,
                    "controller_version": CONTROLLER_VERSION,
                    "codex_version": version,
                    "thread_id": self.thread_id,
                    "pid": self.transport.process.pid,
                    "prompt_sha256": self.prompt_sha256,
                    "allow_provider_model_fallback": False,
                    "tools": [spec["name"] for spec in specs],
                    "codex_image_max_edge": self.image_max_edge,
                    "policy_images_resized": False,
                },
            )
        except BaseException:
            self.close()
            raise

    def _turn(self, text: str) -> str:
        result = self.transport.request(
            "turn/start",
            {
                "threadId": self.thread_id,
                "model": MODEL,
                "effort": EFFORT,
                "approvalPolicy": "never",
                "approvalsReviewer": "auto_review",
                "permissions": "rollout_agent",
                "cwd": str(self.agent_workspace),
                "runtimeWorkspaceRoots": [str(self.agent_workspace)],
                "input": [{"type": "text", "text": text}],
            },
            self.timeout,
        )
        return result["turn"]["id"]

    def _network_continue(
        self, rollout, old_turn: str, error: object, consecutive: int, serial: int
    ) -> str:
        if not is_network_error(error) or consecutive >= len(NETWORK_CONTINUE_DELAYS):
            raise RuntimeError(f"Codex network continue exhausted or ineligible: {error}")
        delay = NETWORK_CONTINUE_DELAYS[consecutive]
        record = {
            "thread_id": self.thread_id,
            "previous_turn_id": old_turn,
            "error": error,
            "step_id": rollout.tick,
            "next_call": rollout.next_call(),
            "delay_seconds": delay,
            "consecutive_attempt": consecutive + 1,
            "physical_actions_replayed": 0,
        }
        path = self.workspace / f"network_continue_{serial:04d}.json"
        write_json(path, record)
        for _ in range(delay):
            time.sleep(1)
        return self._turn(
            "Continue the same rollout after a network error. The simulator and executed "
            "actions are unchanged; do not reset or repeat actions. Next call: "
            + json.dumps(rollout.next_call())
        )

    def run(self, rollout) -> None:
        turn_id = self._turn(
            "Act as the autonomous policy for this single Sharpa simulation rollout. "
            "Use the rollout tools plus normal file/image/calculation tools as useful. "
            "The user authorizes sending this episode's RGB views, named joint state, object/target "
            "pose, contacts, and same-episode history to OpenAI Codex. First call: "
            + json.dumps(rollout.next_call())
        )
        call_index = 0
        continuations = 0
        network_error = None
        network_consecutive = 0
        network_serial = 0
        completed_turns = 0
        token_limit = int(os.environ.get("CODEX_MAX_TOTAL_TOKENS", "0"))
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise TimeoutError("Codex made no completed tool call before timeout")
                event = self.transport.next_message(remaining)
            except TimeoutError:
                if not network_error:
                    raise
                response = self.transport.request(
                    "thread/read",
                    {"threadId": self.thread_id, "includeTurns": True},
                    min(30, self.timeout),
                )
                turn = closed_network_turn(response, self.thread_id, turn_id)
                if turn is None:
                    raise
                event = {
                    "method": "turn/completed",
                    "params": {"threadId": self.thread_id, "turn": turn},
                }
            method = event.get("method")
            params = event.get("params", {})
            if method == "thread/tokenUsage/updated" and params.get("threadId") == self.thread_id:
                usage = params.get("tokenUsage", {})
                write_json(
                    self.workspace.parent / "token_usage.json",
                    {
                        "model": MODEL,
                        "effort": EFFORT,
                        "provider": PROVIDER,
                        "usage": usage,
                        "configured_limit": token_limit,
                    },
                )
                if token_limit and usage.get("total", {}).get("totalTokens", 0) >= token_limit:
                    raise RuntimeError("Configured total-token budget reached")
            elif method == "item/tool/call":
                if params.get("threadId") != self.thread_id or params.get("turnId") != turn_id:
                    raise RuntimeError("Tool call belongs to another thread or turn")
                name = params["tool"]
                arguments = params["arguments"]
                write_json(
                    self.workspace / f"call_{call_index:04d}_request.json",
                    {"call_id": params["callId"], "tool": name, "arguments": arguments},
                )
                handlers = {"dexhand_start": rollout.start, "dexhand_act": rollout.act}
                try:
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as error:
                            raise InputError(f"Tool arguments must be JSON: {error}") from error
                    if name not in handlers:
                        raise InputError("Unknown host service")
                    if not isinstance(arguments, dict):
                        raise InputError("Tool arguments must be an object")
                    packet = handlers[name](**arguments)
                except InputError as error:
                    packet = rejected_input(error, rollout)
                    success = False
                else:
                    success = True
                    network_error = None
                    network_consecutive = 0
                write_json(self.workspace / f"call_{call_index:04d}_result.json", packet)
                self.transport.reply(
                    event["id"],
                    {
                        "success": success,
                        "contentItems": content_items(
                            packet,
                            images=success,
                            max_image_edge=self.image_max_edge,
                        ),
                    },
                )
                call_index += 1
                deadline = time.monotonic() + self.timeout
            elif method in ("item/started", "item/completed"):
                if params.get("threadId") != self.thread_id or params.get("turnId") != turn_id:
                    continue
                item = params.get("item", {})
                kind = item.get("type")
                if kind in NATIVE_WORK_ITEMS or (
                    kind == "agentMessage" and method == "item/completed"
                ):
                    with (self.workspace / "agent_events.jsonl").open("a") as stream:
                        stream.write(json.dumps({"method": method, **params}, ensure_ascii=False) + "\n")
                    if kind in NATIVE_WORK_ITEMS:
                        deadline = time.monotonic() + self.timeout
            elif method == "turn/completed" and params.get("threadId") == self.thread_id:
                turn = params.get("turn", {})
                if turn.get("id") != turn_id:
                    continue
                write_json(self.workspace / f"turn_{completed_turns:02d}_completed.json", turn)
                completed_turns += 1
                failure = turn.get("error") or network_error
                if turn.get("status") == "failed" and is_network_error(failure):
                    if rollout.phase == "done":
                        return
                    turn_id = self._network_continue(
                        rollout, turn_id, failure, network_consecutive, network_serial
                    )
                    network_consecutive += 1
                    network_serial += 1
                    network_error = None
                    deadline = time.monotonic() + self.timeout
                    continue
                if turn.get("status") != "completed":
                    raise RuntimeError(f"Codex turn ended: {turn.get('status')}")
                if rollout.phase == "done":
                    return
                if continuations >= 2:
                    raise RuntimeError("Codex ended repeatedly before completing the rollout")
                continuations += 1
                turn_id = self._turn(
                    "The same episode is unfinished. Continue the skill; next call: "
                    + json.dumps(rollout.next_call())
                )
                deadline = time.monotonic() + self.timeout
            elif method in ("error", "turn/failed"):
                if params.get("threadId") not in (None, self.thread_id) or params.get(
                    "turnId"
                ) not in (None, turn_id):
                    continue
                if is_network_error(params.get("error", params)):
                    network_error = params.get("error", params)
                if method == "error" and params.get("willRetry") is True:
                    continue
                if network_error:
                    if rollout.phase == "done":
                        return
                    deadline = min(deadline, time.monotonic() + 30)
                    continue
                raise RuntimeError(f"Codex error: {params}")
            elif "id" in event and "method" in event:
                raise RuntimeError(f"Unexpected Codex capability request: {method}")

    def close(self) -> None:
        transport = getattr(self, "transport", None)
        if transport is not None:
            transport.close()


__all__ = ["CONTROLLER_VERSION", "DexHandCodexPolicy"]
