#!/bin/bash
# Experiment 2: Multi-frame H=3 + Factorized Spatiotemporal + Stale Intent (chunk_size=8)
# SpatiotemporalTransitionViT: MEM-style factorized attention, H=3 frames
# Note: history_len=3 → 3x DINOv2 encoding per batch → reduce batch_size if OOM
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

LD_PRELOAD=.venv/lib/libglibc_compat.so \
uv run python cerebellar_correction/training/train_phase1.py \
    --dataset_path /mnt/md1/solee/data/bridge_lerobot \
    --intent_dir data/bridge_intents \
    --output_dir checkpoints/exp2_multiframe_stale \
    --cache_dir /mnt/md1/solee/data/phase1_cache \
    --experiment_mode multiframe \
    --chunk_size 8 \
    --history_len 3 \
    --batch_size 512 \
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
    --num_workers 32 \
    --device cuda \
    --wandb_project icac-phase1 \
    --wandb_run_name exp2_multiframe_stale
