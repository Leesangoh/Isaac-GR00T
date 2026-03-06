#!/bin/bash
# Step 5a: Run GR00T + Intent Passthrough server (for client-side cerebellum)
# Server returns raw action chunks + intent vectors. No correction applied.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

LD_PRELOAD=.venv/lib/libglibc_compat.so \
uv run python cerebellar_correction/eval/run_intent_server.py \
    --model-path /mnt/md1/solee/checkpoints/GR00T-N1.6-bridge \
    --embodiment-tag OXE_WIDOWX \
    --device cuda
