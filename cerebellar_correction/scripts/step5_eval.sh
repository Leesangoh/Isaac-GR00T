#!/bin/bash
# Step 5: Run GR00T + Intent Cerebellum eval server for SimplerEnv
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

CEREBELLUM_CKPT=${CEREBELLUM_CKPT:-checkpoints/cerebellum_intent}
CORRECTION_ALPHA=${CORRECTION_ALPHA:-1.0}
MAX_CORRECTION=${MAX_CORRECTION:-0.15}

LD_PRELOAD=.venv/lib/libglibc_compat.so \
uv run python cerebellar_correction/eval/run_cerebellum_server.py \
    --model-path /mnt/md1/solee/checkpoints/GR00T-N1.6-bridge \
    --embodiment-tag OXE_WIDOWX \
    --cerebellum-ckpt ${CEREBELLUM_CKPT} \
    --correction-alpha ${CORRECTION_ALPHA} \
    --max-correction ${MAX_CORRECTION} \
    --device cuda
