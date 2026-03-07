#!/bin/bash
# Experiment 1: Action-Conditioned Oracle + Stale Intent (chunk_size=8)
# Diagnostic: tests whether forward prediction bottleneck is intent signal quality
# ActionConditionedTransitionViT: 179 tokens = [49 patches, 1 proprio, 1 action, 128 intent]
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

LD_PRELOAD=.venv/lib/libglibc_compat.so \
uv run python cerebellar_correction/training/train_phase1.py \
    --dataset_path /mnt/md1/solee/data/bridge_lerobot \
    --intent_dir data/bridge_intents \
    --vla_dir data/vla_actions \
    --output_dir checkpoints/exp1_action_stale \
    --cache_dir /mnt/md1/solee/data/phase1_cache \
    --experiment_mode action_conditioned \
    --chunk_size 8 \
    --batch_size 2048 \
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
    --wandb_run_name exp1_action_stale
