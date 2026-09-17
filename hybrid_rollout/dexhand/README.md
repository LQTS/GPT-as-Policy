# DexHand Astra direct-control evaluation

This adapter runs one PTrack Sharpa environment and lets the repository-pinned
GPT-6 Astra policy issue bounded 22-joint target deltas. It uses the unchanged
`hybrid_rollout.robodojo.settings` model, provider, reasoning effort, Responses
transport, and no-fallback policy. It does not load an RL checkpoint or modify
PTrack source.

The default task is `Isaac-Sharpa-Benchmark-Cylinder-Rotation-A-Axis-v0` from
PTrack branch `sharpa-rl-only-five-tasks`, with target speed 0.5 rad/s, seed 42,
one environment, and three model decisions. It is deliberately a partial-horizon
integration check, not a benchmark score. The adapter also supports PTrack
continuous-rotation tasks that expose `speed_stages`.

Run the simulator, grasp-bank, camera, observation, and one no-op control-step
preflight without an Astra model call:

```bash
PREFLIGHT_ONLY=1 RUN_ID=cylinder_a_seed42_preflight_01 \
  hybrid_rollout/dexhand/run_local.sh
```

Run the three-decision Astra smoke after preparing an isolated managed Codex
profile:

```bash
RUN_ID=cylinder_a_seed42_smoke_01 hybrid_rollout/dexhand/run_local.sh
```

Override `PTRACK_ROOT`, `ISAACLAB_PYTHON`, `CODEX_BIN`, `GRASP_BANK`,
`TARGET_SPEED`, `SUCCESS_TOLERANCE`, or `WARMUP_STEPS` when using another
validated installation or protocol. A full episode requires an explicitly
reviewed `MAX_DECISIONS` budget; the default remains three.

Each observation includes synchronized front-oblique, opposite-oblique, and
top-oblique RGB views, named proprioception, object/target state, and the current
target angular velocity in the palm frame. For fixed object-axis benchmarks the
packet also records the configured initialized-object axis. Results include the
same dynamic rotation accumulator used by PTrack (`at_goal`, rotation distance,
signed-axis coverage, drop, non-finite, and survival metrics).

Artifacts are written below
`.runtime/robodojo_mixed_control/results/dexhand/<RUN_ID>/controller/`. `run.json`
records the exact PTrack commit and dirty state, task, grasp bank, target speed,
camera setup, tolerance, and warmup. The controller directory also contains the
Astra settings, prompt hash, RGB/state observations, validated model responses,
executed target history, token usage, and terminal or partial-window metrics.
