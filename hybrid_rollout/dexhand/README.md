# DexHand Astra direct-control smoke

This adapter runs one PTrack Sharpa environment and lets the repository-pinned
GPT-6 Astra policy issue bounded 22-joint target deltas. It uses the unchanged
`hybrid_rollout.robodojo.settings` model, provider, reasoning effort, Responses
transport, and no-fallback policy. It does not load an RL checkpoint or modify
PTrack source.

The default smoke is `Cylinder-Rotation-A-Axis`, seed 42, one environment, and
three model decisions. It is deliberately a partial-horizon integration check,
not a benchmark score.

From the GPT-as-Policy repository root:

```bash
RUN_ID=cylinder_a_seed42_smoke_01 hybrid_rollout/dexhand/run_local.sh
```

The default isolated login is `codex_a` under the shared runtime prepared by the
profile manager. Override `ROLLOUT_SHARED_ROOT`, `ROLLOUT_AUTH_PROFILE`,
`ROLLOUT_CODEX_HOME_DIR`, `PTRACK_ROOT`, `ISAACLAB_PYTHON`, or `CODEX_BIN` only
when using an equivalently validated local installation.

Each observation includes synchronized front-oblique, opposite-oblique, and
top-oblique RGB views. Artifacts are written below
`.runtime/robodojo_mixed_control/results/dexhand/<RUN_ID>/controller/`, including
the exact Astra settings, prompt hash, RGB/state observations, validated model
responses, executed target history, token usage, and partial-window rotation metrics.
By default the launcher also writes `rollout_multiview.mp4`,
`rollout_multiview_poster.png`, and `object_trajectory.json`. Set
`MAKE_VIDEO=0` only when video postprocessing is intentionally disabled.
