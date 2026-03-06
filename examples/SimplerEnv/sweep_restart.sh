#!/usr/bin/env bash
# sweep_restart.sh — Sweep Restart Sampling hyperparameters and record success rates.
#
# Four sweep modes:
#   --sweep grid:   t_restart × t_back grid (K=1, total_steps=10)
#   --sweep K:      K=0,1,2,3 with fixed t_restart/t_back
#   --sweep method: sdedit vs interpolation vs scaled
#   --sweep fair:   vanilla 5/10/15 vs restart 5/10 NFE (fair comparison)
#
# Usage:
#   chmod +x examples/SimplerEnv/sweep_restart.sh
#   ./examples/SimplerEnv/sweep_restart.sh --sweep grid
#
# Prerequisites:
#   - SimplerEnv setup completed (see examples/SimplerEnv/README.md)
#   - Model checkpoint available

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

RESULTS_DIR="${RESULTS_DIR:-results}"
SIMPLER_PYTHON="gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"

SWEEP_MODE="${1:---sweep}"
if [ "${SWEEP_MODE}" = "--sweep" ]; then
    SWEEP_MODE="${2:-grid}"
else
    SWEEP_MODE="${SWEEP_MODE#--sweep=}"
fi

# ===== Helper functions =====

launch_restart_server() {
    local t_restart="$1"
    local t_back="$2"
    local K="$3"
    local steps_restart="$4"
    local total_steps="$5"
    local noise_method="$6"

    uv run python examples/SimplerEnv/run_restart_server.py \
        --model-path "${MODEL_PATH}" \
        --embodiment-tag "${EMBODIMENT_TAG}" \
        --device "${DEVICE}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --use-sim-policy-wrapper \
        --t-restart "${t_restart}" \
        --t-back "${t_back}" \
        --restart-K "${K}" \
        --steps-restart "${steps_restart}" \
        --total-steps "${total_steps}" \
        --noise-method "${noise_method}" &
    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID}"
    sleep 30
}

launch_vanilla_server() {
    local num_steps="${1:-4}"
    # Vanilla server uses default num_inference_timesteps from the model.
    # For fair comparison we use restart with K=0.
    uv run python examples/SimplerEnv/run_restart_server.py \
        --model-path "${MODEL_PATH}" \
        --embodiment-tag "${EMBODIMENT_TAG}" \
        --device "${DEVICE}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --use-sim-policy-wrapper \
        --t-restart 0.5 \
        --t-back 0.0 \
        --restart-K 0 \
        --steps-restart 1 \
        --total-steps "${num_steps}" \
        --noise-method sdedit &
    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID} (vanilla, steps=${num_steps})"
    sleep 30
}

run_eval() {
    local log_file="$1"
    ${SIMPLER_PYTHON} gr00t/eval/rollout_policy.py \
        --n_episodes "${N_EPISODES}" \
        --policy_client_host "${HOST}" \
        --policy_client_port "${PORT}" \
        --max_episode_steps "${MAX_EPISODE_STEPS}" \
        --env_name "${ENV_NAME}" \
        --n_action_steps "${N_ACTION_STEPS}" \
        --n_envs "${N_ENVS}" \
        2>&1 | tee "${log_file}"
}

parse_and_record() {
    local log_file="$1"
    local csv_file="$2"
    local label="$3"

    SUCCESS_RATE=$(grep -oP 'success[_ ]rate[:\s]*\K[0-9.]+' "${log_file}" | tail -1 || echo "N/A")
    SUCCESSES=$(grep -oP '\K[0-9]+(?=/[0-9]+ )' "${log_file}" | tail -1 || echo "N/A")
    TOTAL=$(grep -oP '[0-9]+/\K[0-9]+' "${log_file}" | tail -1 || echo "N/A")

    echo "  ${label} => success_rate=${SUCCESS_RATE}, successes=${SUCCESSES}, total=${TOTAL}"
    echo "${label},${SUCCESS_RATE},${SUCCESSES},${TOTAL},${ENV_NAME}" >> "${csv_file}"
}

kill_server() {
    echo "  Killing server (PID=${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    sleep 5
}

# ===== Sweep modes =====

sweep_grid() {
    local CSV_FILE="${RESULTS_DIR}/restart_grid_sweep.csv"
    mkdir -p "${RESULTS_DIR}"
    echo "label,success_rate,successes,total,env_name" > "${CSV_FILE}"

    T_RESTARTS=(0.3 0.5 0.6 0.7)
    T_BACKS=(0.0 0.1 0.2 0.3)

    echo "=============================================="
    echo "  Restart Grid Sweep (t_restart × t_back)"
    echo "  Model: ${MODEL_PATH}"
    echo "  Env:   ${ENV_NAME}"
    echo "  K=1, steps_restart=3, total_steps=10"
    echo "=============================================="

    for T_RESTART in "${T_RESTARTS[@]}"; do
        for T_BACK in "${T_BACKS[@]}"; do
            # Skip invalid combos where t_back >= t_restart
            if (( $(echo "${T_BACK} >= ${T_RESTART}" | bc -l) )); then
                echo "  Skipping t_restart=${T_RESTART}, t_back=${T_BACK} (invalid)"
                continue
            fi

            LABEL="tr${T_RESTART}_tb${T_BACK}"
            echo ""
            echo "----------------------------------------------"
            echo "  Running t_restart=${T_RESTART}, t_back=${T_BACK}"
            echo "----------------------------------------------"

            launch_restart_server "${T_RESTART}" "${T_BACK}" 1 3 10 sdedit
            ROLLOUT_LOG="${RESULTS_DIR}/rollout_${LABEL}.log"
            run_eval "${ROLLOUT_LOG}"
            parse_and_record "${ROLLOUT_LOG}" "${CSV_FILE}" "${LABEL}"
            kill_server
        done
    done

    echo ""
    echo "=============================================="
    echo "  Grid sweep complete! Results: ${CSV_FILE}"
    echo "=============================================="
    cat "${CSV_FILE}"
}

sweep_K() {
    local CSV_FILE="${RESULTS_DIR}/restart_K_sweep.csv"
    mkdir -p "${RESULTS_DIR}"
    echo "label,success_rate,successes,total,env_name" > "${CSV_FILE}"

    KS=(0 1 2 3)
    T_RESTART="${T_RESTART:-0.6}"
    T_BACK="${T_BACK:-0.3}"
    STEPS_RESTART="${STEPS_RESTART:-3}"
    TOTAL_STEPS="${TOTAL_STEPS:-10}"

    echo "=============================================="
    echo "  Restart K Sweep"
    echo "  t_restart=${T_RESTART}, t_back=${T_BACK}"
    echo "  steps_restart=${STEPS_RESTART}, total_steps=${TOTAL_STEPS}"
    echo "=============================================="

    for K in "${KS[@]}"; do
        LABEL="K${K}"
        echo ""
        echo "----------------------------------------------"
        echo "  Running K=${K}"
        echo "----------------------------------------------"

        launch_restart_server "${T_RESTART}" "${T_BACK}" "${K}" "${STEPS_RESTART}" "${TOTAL_STEPS}" sdedit
        ROLLOUT_LOG="${RESULTS_DIR}/rollout_${LABEL}.log"
        run_eval "${ROLLOUT_LOG}"
        parse_and_record "${ROLLOUT_LOG}" "${CSV_FILE}" "${LABEL}"
        kill_server
    done

    echo ""
    echo "=============================================="
    echo "  K sweep complete! Results: ${CSV_FILE}"
    echo "=============================================="
    cat "${CSV_FILE}"
}

sweep_method() {
    local CSV_FILE="${RESULTS_DIR}/restart_method_sweep.csv"
    mkdir -p "${RESULTS_DIR}"
    echo "label,success_rate,successes,total,env_name" > "${CSV_FILE}"

    METHODS=(sdedit interpolation scaled)
    T_RESTART="${T_RESTART:-0.6}"
    T_BACK="${T_BACK:-0.3}"

    echo "=============================================="
    echo "  Restart Method Sweep"
    echo "  t_restart=${T_RESTART}, t_back=${T_BACK}, K=1"
    echo "=============================================="

    for METHOD in "${METHODS[@]}"; do
        LABEL="method_${METHOD}"
        echo ""
        echo "----------------------------------------------"
        echo "  Running noise_method=${METHOD}"
        echo "----------------------------------------------"

        launch_restart_server "${T_RESTART}" "${T_BACK}" 1 3 10 "${METHOD}"
        ROLLOUT_LOG="${RESULTS_DIR}/rollout_${LABEL}.log"
        run_eval "${ROLLOUT_LOG}"
        parse_and_record "${ROLLOUT_LOG}" "${CSV_FILE}" "${LABEL}"
        kill_server
    done

    echo ""
    echo "=============================================="
    echo "  Method sweep complete! Results: ${CSV_FILE}"
    echo "=============================================="
    cat "${CSV_FILE}"
}

sweep_fair() {
    local CSV_FILE="${RESULTS_DIR}/restart_fair_sweep.csv"
    mkdir -p "${RESULTS_DIR}"
    echo "label,success_rate,successes,total,env_name" > "${CSV_FILE}"

    echo "=============================================="
    echo "  Fair NFE Comparison: Vanilla vs Restart"
    echo "=============================================="

    # Vanilla baselines: 5, 10, 15 NFE
    for STEPS in 5 10 15; do
        LABEL="vanilla_${STEPS}nfe"
        echo ""
        echo "----------------------------------------------"
        echo "  Running vanilla (${STEPS} NFE)"
        echo "----------------------------------------------"

        launch_vanilla_server "${STEPS}"
        ROLLOUT_LOG="${RESULTS_DIR}/rollout_${LABEL}.log"
        run_eval "${ROLLOUT_LOG}"
        parse_and_record "${ROLLOUT_LOG}" "${CSV_FILE}" "${LABEL}"
        kill_server
    done

    # Restart with matched NFE budgets
    # 5 NFE: phase1=2, K=1, steps_restart=2, phase3=1
    LABEL="restart_5nfe"
    echo ""
    echo "----------------------------------------------"
    echo "  Running restart (5 NFE)"
    echo "----------------------------------------------"
    launch_restart_server 0.5 0.2 1 2 5 sdedit
    ROLLOUT_LOG="${RESULTS_DIR}/rollout_${LABEL}.log"
    run_eval "${ROLLOUT_LOG}"
    parse_and_record "${ROLLOUT_LOG}" "${CSV_FILE}" "${LABEL}"
    kill_server

    # 10 NFE: phase1=4, K=1, steps_restart=3, phase3=3
    LABEL="restart_10nfe"
    echo ""
    echo "----------------------------------------------"
    echo "  Running restart (10 NFE)"
    echo "----------------------------------------------"
    launch_restart_server 0.6 0.3 1 3 10 sdedit
    ROLLOUT_LOG="${RESULTS_DIR}/rollout_${LABEL}.log"
    run_eval "${ROLLOUT_LOG}"
    parse_and_record "${ROLLOUT_LOG}" "${CSV_FILE}" "${LABEL}"
    kill_server

    echo ""
    echo "=============================================="
    echo "  Fair sweep complete! Results: ${CSV_FILE}"
    echo "=============================================="
    cat "${CSV_FILE}"
}

# ===== Main dispatch =====

case "${SWEEP_MODE}" in
    grid)
        sweep_grid
        ;;
    K)
        sweep_K
        ;;
    method)
        sweep_method
        ;;
    fair)
        sweep_fair
        ;;
    *)
        echo "Unknown sweep mode: ${SWEEP_MODE}"
        echo "Usage: $0 --sweep {grid|K|method|fair}"
        exit 1
        ;;
esac
