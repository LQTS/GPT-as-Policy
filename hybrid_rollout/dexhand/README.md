# DexHand Astra direct-control evaluation

This adapter runs one PTrack Sharpa environment and lets the repository-pinned
GPT-6 Astra policy issue bounded 22-joint target deltas. It uses the unchanged
`hybrid_rollout.robodojo.settings` model, provider, reasoning effort, Responses
transport, and no-fallback policy. It does not load an RL checkpoint or modify
PTrack source.

Every run must select an explicit profile. There is no implicit A-axis task:

| Profile | Purpose |
| --- | --- |
| `cylinder_a_axis_smoke` | Minimal A-axis integration smoke; not a benchmark score. |
| `cylinder_d3_heldout` | Held-out continuous-rotation comparison against the selected D3 RL policy. |
| `cylinder_world_z_cases` | Persistent cylinder cases rotating about world +Z at 1.0 rad/s. |
| `cuboid_world_z_cases` | Persistent cuboid cases rotating about world +Z at 1.0 rad/s. |

`cylinder_d3_heldout` fixes the task, held-out grasp bank, group grasp sampling,
1.0 rad/s target speed, 0.005 m wrist-position range, 0.0872665 rad wrist-rotation
range, 0.1 rad success tolerance, and 20-step metric warmup to the existing D3
evaluation conditions. The resolved values and PTrack commit are stored in
`run.json`.

Generate deterministic case sets before running Astra:

```bash
python -m hybrid_rollout.dexhand.prepare_cases \
  --ptrack-root /mnt/liuqingtao/PTrack-dynamic-rotation \
  --output /mnt/liuqingtao/PTrack/outputs/astra_evaluation/world_z_cases_20260917/cylinder \
  --profile cylinder_world_z_cases --num-cases 20 \
  --device cuda:0 --headless --enable_cameras

python -m hybrid_rollout.dexhand.prepare_cases \
  --ptrack-root /mnt/liuqingtao/PTrack-dynamic-rotation \
  --output /mnt/liuqingtao/PTrack/outputs/astra_evaluation/world_z_cases_20260917/cuboid \
  --profile cuboid_world_z_cases --num-cases 20 \
  --device cuda:0 --headless --enable_cameras
```

Each case contains an exact `initial_state.pt`, a portable `state.npz`,
source and stability metadata, raw three-camera images, and a combined preview.
Annotated images show a projected world-frame triad (X red, Y green, Z blue)
with the target +Z axis emphasized; unmodified frames remain as `*_rgb_raw.png`.
The set manifest and contact sheet make case selection auditable. The cuboid
bank currently contains six compatible states; only cases marked `heldout`
are strict held-out evaluations.

Run the simulator, grasp-bank, camera, observation, and one no-op control-step
preflight without an Astra model call:

```bash
PROFILE=cylinder_d3_heldout \
PREFLIGHT_ONLY=1 \
RUN_ID=cylinder_d3_heldout_seed42_preflight_01 \
  hybrid_rollout/dexhand/run_local.sh
```

Replay a selected persistent case by setting both its matching profile and
state path:

```bash
PROFILE=cylinder_world_z_cases \
CASE_STATE=/path/to/case_000/initial_state.pt \
RUN_ID=cylinder_world_z_case000_astra_01 \
  hybrid_rollout/dexhand/run_local.sh
```

Run a three-decision A-axis integration smoke after preparing the isolated
managed Codex profile and explicitly authorizing the episode data transfer:

```bash
PROFILE=cylinder_a_axis_smoke \
RUN_ID=cylinder_a_seed42_smoke_01 \
  hybrid_rollout/dexhand/run_local.sh
```

Set `CODEX_BIN` to a current shared CLI when a compute node's system Codex is
too old for GPT-6 Astra. Override `PTRACK_ROOT`, `ISAACLAB_PYTHON`, or
`ROLLOUT_SHARED_ROOT` only when using another validated installation. A longer
episode requires an explicitly reviewed `MAX_DECISIONS` budget; the default
remains three.

Each observation includes synchronized front-oblique, opposite-oblique, and
top-oblique RGB views, named proprioception, object/target state, and the current
target angular velocity in the palm frame. Results use the same dynamic rotation
accumulator as PTrack and report at-goal rate, rotation distance, signed-axis
coverage, drop, non-finite, survival, signed rotation, reverse rotation, and axis
purity.

Artifacts are written below
`.runtime/robodojo_mixed_control/results/dexhand/<RUN_ID>/controller/`. The
controller directory records the task profile, PTrack provenance, Astra settings,
prompt hash, RGB/state observations, validated responses, executed target history,
token usage, and terminal or partial-window metrics.
