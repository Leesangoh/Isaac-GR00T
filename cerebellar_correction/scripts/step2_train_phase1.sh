#!/bin/bash
# Step 2: Train Phase 1 — DINOv2+LoRA encoder + self-attention forward model (EMA)
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

LD_PRELOAD=.venv/lib/libglibc_compat.so \
uv run python cerebellar_correction/training/train_phase1.py \
    --dataset_path /mnt/md1/solee/data/bridge_lerobot \
    --intent_dir data/bridge_intents \
    --output_dir checkpoints/cerebellum_intent/phase1 \
    --cache_dir /mnt/md1/solee/data/phase1_cache \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --encoder_lr 1e-4 \
    --vicreg_lambda 5.0 \
    --ema_tau 0.996 \
    --num_epochs 50 \
    --decode_workers 64 \
    --num_workers 8 \
    --device cuda \
    --wandb_project cerebellum \
    --wandb_run_name phase1_intent_ema
