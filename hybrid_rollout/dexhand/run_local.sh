#!/usr/bin/env bash
set -euo pipefail

DEXHAND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd "$DEXHAND_DIR/../.." && pwd)}"
ASTRA_ASSESS_ROOT="${ASTRA_ASSESS_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
PTRACK_ROOT="${PTRACK_ROOT:-$ASTRA_ASSESS_ROOT/PTrack_lqt}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-/home/galbot/miniconda3/envs/isaaclab_311/bin/python}"
CODEX_BIN="${CODEX_BIN:-/home/galbot/.nvm/versions/node/v20.20.2/bin/codex}"
ROLLOUT_SHARED_ROOT="${ROLLOUT_SHARED_ROOT:-$ASTRA_ASSESS_ROOT/.runtime/robodojo_mixed_control}"
ROLLOUT_AUTH_PROFILE="${ROLLOUT_AUTH_PROFILE:-codex_a}"
ROLLOUT_CODEX_HOME_DIR="${ROLLOUT_CODEX_HOME_DIR:-$ROLLOUT_SHARED_ROOT/private/auth_profiles/$ROLLOUT_AUTH_PROFILE/codex_home}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROLLOUT_SHARED_ROOT/results/dexhand}"
RUN_ID="${RUN_ID:-cylinder_a_seed42_smoke}"
TASK="${TASK:-Isaac-Sharpa-Benchmark-Cylinder-Rotation-A-Axis-v0}"
SEED="${SEED:-42}"
MAX_DECISIONS="${MAX_DECISIONS:-3}"
HORIZON_STEPS="${HORIZON_STEPS:-0}"
FIXED_REPEAT_STEPS="${FIXED_REPEAT_STEPS:-0}"
MAKE_VIDEO="${MAKE_VIDEO:-1}"
GRASP_BANK="${GRASP_BANK:-$PTRACK_ROOT/outputs/sharpa_dynamic/cylinder_recoverable_grasps_train80_v1.pt}"
OUTPUT="$RESULTS_ROOT/$RUN_ID/controller"

for path in "$PTRACK_ROOT" "$ISAACLAB_PYTHON" "$CODEX_BIN" "$GRASP_BANK" \
    "$ROLLOUT_CODEX_HOME_DIR/auth.json" "$ROLLOUT_CODEX_HOME_DIR/config.toml"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
[[ "$ROLLOUT_AUTH_PROFILE" == codex_* ]] || {
    echo "DexHand smoke currently requires an isolated managed ChatGPT profile" >&2
    exit 2
}
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 2; }

mkdir -p "$RESULTS_ROOT/$RUN_ID"
export CODEX_HOME="$ROLLOUT_CODEX_HOME_DIR"
export ROLLOUT_AUTH_PROFILE ROLLOUT_SHARED_ROOT
export PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE_ROOT:$PTRACK_ROOT:$PTRACK_ROOT/source/ConTrack${PYTHONPATH:+:$PYTHONPATH}"

"$ISAACLAB_PYTHON" -m hybrid_rollout.dexhand.run \
    --ptrack-root "$PTRACK_ROOT" \
    --grasp-bank "$GRASP_BANK" \
    --output "$OUTPUT" \
    --codex "$CODEX_BIN" \
    --task "$TASK" \
    --seed "$SEED" \
    --max-decisions "$MAX_DECISIONS" \
    --horizon-steps "$HORIZON_STEPS" \
    --fixed-repeat-steps "$FIXED_REPEAT_STEPS" \
    --device cuda:0 \
    --headless \
    --enable_cameras

[[ -f "$OUTPUT/result.json" ]] || {
    echo "DexHand rollout exited without result.json: $OUTPUT" >&2
    exit 4
}

if [[ "$MAKE_VIDEO" == 1 ]]; then
    "$ISAACLAB_PYTHON" -m hybrid_rollout.dexhand.render_video --controller "$OUTPUT"
fi
