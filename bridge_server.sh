#!/bin/bash
set -x -e

# flash_attn GLIBC compat
export LD_PRELOAD=.venv/lib/libglibc_compat.so

# PhysREPA tsalign ViT-G checkpoint (default)
MODEL_PATH="${1:-/mnt/md1/solee/checkpoints/GR00T-N1.6-physrepa-tsalign-vitg}"

uv run python gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag OXE_WIDOWX \
    --use-sim-policy-wrapper
