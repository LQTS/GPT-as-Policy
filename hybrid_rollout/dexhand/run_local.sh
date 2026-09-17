#!/usr/bin/env bash
set -euo pipefail

DEXHAND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd "$DEXHAND_DIR/../.." && pwd)}"
ASTRA_ASSESS_ROOT="${ASTRA_ASSESS_ROOT:-$(cd "$CODE_ROOT/.." && pwd)}"
PTRACK_ROOT="${PTRACK_ROOT:-$ASTRA_ASSESS_ROOT/PTrack-dynamic-rotation}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-$ASTRA_ASSESS_ROOT/miniconda3/envs/dexhot/bin/python}"
CODEX_BIN="${CODEX_BIN:-/usr/local/bin/codex}"
ISAACSIM_SETUP="${ISAACSIM_SETUP:-/isaac-sim/setup_conda_env.sh}"
ROLLOUT_SHARED_ROOT="${ROLLOUT_SHARED_ROOT:-$ASTRA_ASSESS_ROOT/.runtime/robodojo_mixed_control}"
ROLLOUT_AUTH_PROFILE="${ROLLOUT_AUTH_PROFILE:-codex_a}"
ROLLOUT_CODEX_HOME_DIR="${ROLLOUT_CODEX_HOME_DIR:-$ROLLOUT_SHARED_ROOT/private/auth_profiles/$ROLLOUT_AUTH_PROFILE/codex_home}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROLLOUT_SHARED_ROOT/results/dexhand}"
PROFILE="${PROFILE:?Set PROFILE to an explicit DexHand evaluation profile}"
RUN_ID="${RUN_ID:?Set RUN_ID to a unique result identifier}"
SEED="${SEED:-42}"
MAX_DECISIONS="${MAX_DECISIONS:-3}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
OUTPUT="$RESULTS_ROOT/$RUN_ID/controller"

for path in "$PTRACK_ROOT" "$ISAACLAB_PYTHON" "$CODEX_BIN" "$ISAACSIM_SETUP"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
if [[ "$PREFLIGHT_ONLY" != 1 ]]; then
    for path in "$ROLLOUT_CODEX_HOME_DIR/auth.json" "$ROLLOUT_CODEX_HOME_DIR/config.toml"; do
        [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
    done
    [[ "$ROLLOUT_AUTH_PROFILE" == codex_* ]] || {
        echo "DexHand rollout requires an isolated managed ChatGPT profile" >&2
        exit 2
    }
fi
[[ ! -e "$OUTPUT" ]] || { echo "Refusing existing output: $OUTPUT" >&2; exit 2; }

mkdir -p "$RESULTS_ROOT/$RUN_ID"
export CODEX_HOME="$ROLLOUT_CODEX_HOME_DIR"
export ROLLOUT_AUTH_PROFILE ROLLOUT_SHARED_ROOT
set +u
source "$ISAACSIM_SETUP"
set -u

export PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE_ROOT:$PTRACK_ROOT:$PTRACK_ROOT/source/ConTrack${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
    --ptrack-root "$PTRACK_ROOT"
    --output "$OUTPUT"
    --codex "$CODEX_BIN"
    --profile "$PROFILE"
    --seed "$SEED"
    --max-decisions "$MAX_DECISIONS"
    --device cuda:0
    --headless
    --enable_cameras
)
[[ "$PREFLIGHT_ONLY" == 1 ]] && ARGS+=(--preflight-only)

"$ISAACLAB_PYTHON" -m hybrid_rollout.dexhand.run "${ARGS[@]}"

[[ -f "$OUTPUT/result.json" ]] || {
    echo "DexHand rollout exited without result.json: $OUTPUT" >&2
    exit 4
}
