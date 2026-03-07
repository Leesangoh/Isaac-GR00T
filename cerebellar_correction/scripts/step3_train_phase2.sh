#!/bin/bash
# Step 3: Train Phase 2 — Patch-level correction with attention pooling (v2)
# Frozen DINOv2 + frozen TransitionViT, trains pooling + correction net
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

LD_PRELOAD=.venv/lib/libglibc_compat.so \
uv run python cerebellar_correction/training/train_phase2.py \
    --dataset_path /mnt/md1/solee/data/bridge_lerobot \
    --intent_dir data/bridge_intents \
    --vla_action_dir data/vla_actions \
    --phase1_dir checkpoints/cerebellum_intent/phase1 \
    --output_dir checkpoints/cerebellum_intent/phase2 \
    --max_correction 0.15 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --num_epochs 50 \
    --decode_workers 32 \
    --num_workers 4 \
    --device cuda \
    --wandb_project cerebellum \
    --wandb_run_name phase2_patch_correction_v2
