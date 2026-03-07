#!/bin/bash
# Step 2: Train Phase 1 — Patch-level Transition ViT (v2, DINO-WM inspired)
# Frozen DINOv2 encoder, 49 patch tokens, 4-layer ViT predictor
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

LD_PRELOAD=.venv/lib/libglibc_compat.so \
uv run python cerebellar_correction/training/train_phase1.py \
    --dataset_path /mnt/md1/solee/data/bridge_lerobot \
    --intent_dir data/bridge_intents \
    --output_dir checkpoints/cerebellum_intent/phase1 \
    --cache_dir /mnt/md1/solee/data/phase1_cache \
    --batch_size 128 \
    --learning_rate 3e-4 \
    --intent_proj_lr 5e-4 \
    --weight_decay 0.01 \
    --warmup_epochs 5 \
    --proprio_lambda 0.1 \
    --num_layers 4 \
    --num_heads 8 \
    --ffn_dim 1536 \
    --num_epochs 100 \
    --decode_workers 80 \
    --num_workers 8 \
    --device cuda \
    --wandb_project cerebellum \
    --wandb_run_name phase1_patch_vit_v2
