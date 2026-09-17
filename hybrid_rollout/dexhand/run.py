#!/usr/bin/env python3
"""Run one GPT-6 Astra direct-control smoke episode in PTrack Sharpa."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import traceback

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-Sharpa-Benchmark-Cylinder-Rotation-A-Axis-v0"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ptrack-root", type=Path, required=True)
parser.add_argument("--grasp-bank", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--codex", type=Path, required=True)
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-decisions", type=int, default=3)
parser.add_argument("--controller-timeout", type=int, default=900)
parser.add_argument("--camera-width", type=int, default=640)
parser.add_argument("--camera-height", type=int, default=480)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.ptrack_root = args.ptrack_root.expanduser().resolve()
args.grasp_bank = args.grasp_bank.expanduser().resolve()
args.output = args.output.expanduser().resolve()
args.codex = args.codex.expanduser().resolve()
for required in (args.ptrack_root, args.grasp_bank, args.codex):
    if not required.exists():
        parser.error(f"Required path does not exist: {required}")
if args.max_decisions < 1:
    parser.error("--max-decisions must be positive")
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
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg
from isaaclab_tasks.utils import load_cfg_from_registry

import isaaclab_tasks  # noqa: F401,E402
import ConTrack.tasks  # noqa: F401,E402
from scripts.tools.sharpa_camera import camera_quat_opengl_wxyz  # noqa: E402

from .camera_views import CAMERA_VIEWS  # noqa: E402
from .policy import CONTROLLER_VERSION, DexHandCodexPolicy  # noqa: E402
from .runtime import DexHandRollout  # noqa: E402
from hybrid_rollout.robodojo.io import write_json  # noqa: E402
from hybrid_rollout.robodojo.settings import EFFORT, MODEL  # noqa: E402


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


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    worker = None
    rollout = None

    def terminate(signum, frame):
        raise KeyboardInterrupt("Stopping this owned DexHand rollout")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        env = make_env()
        rollout = DexHandRollout(
            env,
            args.output,
            task=args.task,
            seed=args.seed,
            max_decisions=args.max_decisions,
        )
        worker = DexHandCodexPolicy(
            args.output / "codex_workspace",
            str(args.codex),
            timeout=args.controller_timeout,
        )
        write_json(
            args.output / "astra_settings.json",
            {
                "source": "hybrid_rollout.robodojo.settings",
                "model": MODEL,
                "reasoning_effort": EFFORT,
                "provider_fallback": False,
                "prompt_sha256": worker.prompt_sha256,
            },
        )
        worker.run(rollout)
    except BaseException:
        write_json(
            args.output / "failure.json",
            {
                "error": traceback.format_exc(),
                "controller_version": CONTROLLER_VERSION,
                "step_id": rollout.tick if rollout else 0,
                "completed": False,
            },
        )
        if rollout is not None and rollout.phase != "done":
            rollout.finish("controller_error")
        raise
    finally:
        if worker is not None:
            worker.close()
        if rollout is not None:
            rollout.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
