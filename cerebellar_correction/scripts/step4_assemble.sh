#!/bin/bash
# Step 4: Assemble Phase 1 + Phase 2 into a single checkpoint
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

LD_PRELOAD=.venv/lib/libglibc_compat.so \
uv run python cerebellar_correction/training/assemble_checkpoint.py \
    --phase1_dir checkpoints/cerebellum_intent/phase1 \
    --phase2_dir checkpoints/cerebellum_intent/phase2 \
    --output_path checkpoints/cerebellum_intent/cerebellum_assembled.pt
