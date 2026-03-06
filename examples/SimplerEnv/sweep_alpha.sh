#!/usr/bin/env bash
# sweep_alpha.sh — Sweep alpha values for DDCD and record success rates.
#
# For each alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
#   1. Launch DDCD server (alpha=0.0 uses vanilla server)
#   2. Run rollout_policy.py for google_robot_pick_coke_can (100 episodes)
#   3. Parse success rate and append to CSV
#   4. Kill server and proceed to next alpha
#
# Usage:
#   chmod +x examples/SimplerEnv/sweep_alpha.sh
#   ./examples/SimplerEnv/sweep_alpha.sh
#
# Prerequisites:
#   - SimplerEnv setup completed (see examples/SimplerEnv/README.md)
#   - Model checkpoint available (default: nvidia/GR00T-N1.6-fractal)

set -euo pipefail

# ===== Configuration =====
MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.6-fractal}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-OXE_GOOGLE}"
DEVICE="${DEVICE:-cuda}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"
N_EPISODES="${N_EPISODES:-100}"
N_ENVS="${N_ENVS:-5}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
ENV_NAME="${ENV_NAME:-simpler_env_google/google_robot_pick_coke_can}"
N_ACTION_STEPS="${N_ACTION_STEPS:-1}"
CLAMP_RATIO="${CLAMP_RATIO:-0.3}"

RESULTS_DIR="${RESULTS_DIR:-results}"
CSV_FILE="${RESULTS_DIR}/alpha_sweep.csv"
SIMPLER_PYTHON="gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"

ALPHAS=(0.0 0.5 1.0 1.5 2.0)

# ===== Setup =====
mkdir -p "${RESULTS_DIR}"
echo "alpha,success_rate,successes,total,env_name" > "${CSV_FILE}"

echo "=============================================="
echo "  DDCD Alpha Sweep"
echo "  Model: ${MODEL_PATH}"
echo "  Env:   ${ENV_NAME}"
echo "  Episodes: ${N_EPISODES}"
echo "  Alphas: ${ALPHAS[*]}"
echo "=============================================="

for ALPHA in "${ALPHAS[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "  Running alpha=${ALPHA}"
    echo "----------------------------------------------"

    # Launch server in background
    if [ "${ALPHA}" = "0.0" ]; then
        echo "  Using vanilla server (alpha=0.0)"
        uv run python gr00t/eval/run_gr00t_server.py \
            --model-path "${MODEL_PATH}" \
            --embodiment-tag "${EMBODIMENT_TAG}" \
            --device "${DEVICE}" \
            --host "${HOST}" \
            --port "${PORT}" \
            --use-sim-policy-wrapper &
    else
        echo "  Using DDCD server (alpha=${ALPHA}, clamp_ratio=${CLAMP_RATIO})"
        uv run python examples/SimplerEnv/run_contrastive_server.py \
            --model-path "${MODEL_PATH}" \
            --embodiment-tag "${EMBODIMENT_TAG}" \
            --device "${DEVICE}" \
            --host "${HOST}" \
            --port "${PORT}" \
            --alpha "${ALPHA}" \
            --clamp-ratio "${CLAMP_RATIO}" \
            --use-sim-policy-wrapper &
    fi
    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID}"

    # Wait for server to be ready
    echo "  Waiting for server to start..."
    sleep 30

    # Run evaluation
    echo "  Running rollout (${N_EPISODES} episodes)..."
    ROLLOUT_LOG="${RESULTS_DIR}/rollout_alpha_${ALPHA}.log"
    ${SIMPLER_PYTHON} gr00t/eval/rollout_policy.py \
        --n_episodes "${N_EPISODES}" \
        --policy_client_host "${HOST}" \
        --policy_client_port "${PORT}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
        --env_name "${ENV_NAME}" \
        --n_action_steps "${N_ACTION_STEPS}" \
        --n_envs "${N_ENVS}" \
        2>&1 | tee "${ROLLOUT_LOG}"

    # Parse success rate from log
    # Look for patterns like "Success rate: X/Y (Z%)" or "success_rate: 0.XX"
    SUCCESS_RATE=$(grep -oP 'success[_ ]rate[:\s]*\K[0-9.]+' "${ROLLOUT_LOG}" | tail -1 || echo "N/A")
    SUCCESSES=$(grep -oP '\K[0-9]+(?=/[0-9]+ )' "${ROLLOUT_LOG}" | tail -1 || echo "N/A")
    TOTAL=$(grep -oP '[0-9]+/\K[0-9]+' "${ROLLOUT_LOG}" | tail -1 || echo "N/A")

    echo "  alpha=${ALPHA} => success_rate=${SUCCESS_RATE}, successes=${SUCCESSES}, total=${TOTAL}"
    echo "${ALPHA},${SUCCESS_RATE},${SUCCESSES},${TOTAL},${ENV_NAME}" >> "${CSV_FILE}"

    # Kill server
    echo "  Killing server (PID=${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    sleep 5
done

echo ""
echo "=============================================="
echo "  Sweep complete! Results saved to: ${CSV_FILE}"
echo "=============================================="
cat "${CSV_FILE}"
