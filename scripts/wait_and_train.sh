#!/bin/bash
# Wait for all depth generation GPUs to finish, then launch DepthMem training.
set -e

DEPTH_DIR="/mnt/md1/solee/bridge_depth_maps"
EXPECTED_EPISODES=53192
CHECK_INTERVAL=60  # seconds

echo "=== Waiting for depth map generation to complete ==="
echo "Expected: ${EXPECTED_EPISODES} episodes with .done markers"
echo "Checking every ${CHECK_INTERVAL}s..."
echo ""

while true; do
    # Count .done markers
    DONE_COUNT=$(find "${DEPTH_DIR}" -name ".done" 2>/dev/null | wc -l)
    TIMESTAMP=$(date '+%H:%M:%S')
    echo "[${TIMESTAMP}] Progress: ${DONE_COUNT} / ${EXPECTED_EPISODES} episodes done"

    if [ "${DONE_COUNT}" -ge "${EXPECTED_EPISODES}" ]; then
        echo ""
        echo "=== All depth maps generated! ==="
        break
    fi

    # Also check if all 4 depth GPU tmux sessions have exited
    RUNNING=0
    for gpu_id in 0 1 2 3; do
        if tmux has-session -t "depth_gpu${gpu_id}" 2>/dev/null; then
            RUNNING=$((RUNNING + 1))
        fi
    done

    if [ "${RUNNING}" -eq 0 ] && [ "${DONE_COUNT}" -gt 0 ]; then
        echo ""
        echo "=== All depth GPU sessions have exited (${DONE_COUNT} episodes done) ==="
        break
    fi

    sleep ${CHECK_INTERVAL}
done

echo ""
echo "=== Starting DepthMem fine-tuning ==="
echo "Time: $(date)"
cd /home/solee/Isaac-GR00T
bash scripts/finetune_depthmem.sh
