#!/bin/bash
# Step 1: Extract intent vectors + VLA actions from GR00T (single forward pass)
# ~6-8 hours with 4 GPUs. Use --max_episodes 5 for smoke test.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

LD_PRELOAD=.venv/lib/libglibc_compat.so \
uv run python cerebellar_correction/data/extract_all.py \
    --dataset_path /mnt/md1/solee/data/bridge_lerobot \
    --model_path /mnt/md1/solee/checkpoints/GR00T-N1.6-bridge \
    --intent_dir data/bridge_intents \
    --vla_dir data/vla_actions \
    --gpu_ids 0,1,2,3 \
    --decode_workers 32 \
    --inference_batch_size 32 \
    --image_size 224
