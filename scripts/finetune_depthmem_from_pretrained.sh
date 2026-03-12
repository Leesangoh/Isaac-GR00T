#!/bin/bash
# DepthMem fine-tuning from PRETRAINED checkpoint (not bridge-finetuned)
#
# Starts from nvidia/GR00T-N1.6-3B base model so depth channel
# co-adapts with RGB from scratch on bridge data.
#
# Prerequisites:
#   1. Generate depth maps: python scripts/generate_depth_maps.py \
#        --data_dir /path/to/bridge_lerobot --output_dir /path/to/depth_maps
#   2. HuggingFace access to nvidia/GR00T-N1.6-3B
#
# Usage:
#   bash scripts/finetune_depthmem_from_pretrained.sh

set -x -e

# === Environment ===
export LD_PRELOAD=.venv/lib/libglibc_compat.so
PYTHON=".venv/bin/python"
TORCHRUN=".venv/bin/torchrun"

# === Paths ===
DATA_DIR="/mnt/md1/solee/data/bridge_lerobot"
DEPTH_DIR="/mnt/md1/solee/bridge_depth_maps"
MODEL_PATH="nvidia/GR00T-N1.6-3B"
OUTPUT_DIR="/mnt/md1/solee/checkpoints/GR00T-N1.6-depthmem-pretrained"

# === Hardware ===
export NUM_GPUS=4

# === DepthMem settings ===
NUM_TEMPORAL_FRAMES=16

# === Training hyperparameters ===
MAX_STEPS=20000
GLOBAL_BATCH_SIZE=16        # micro_batch = 16/4GPUs = 4 per GPU
GRAD_ACCUM=4                # effective batch = 16 × 4 = 64 per step
LR=1e-4
WARMUP_RATIO=0.05
WEIGHT_DECAY=1e-5
STATE_DROPOUT=0.8
ACTION_HORIZON=8

echo "=== DepthMem Fine-tuning (from pretrained) ==="
echo "Data:    ${DATA_DIR}"
echo "Depths:  ${DEPTH_DIR}"
echo "Model:   ${MODEL_PATH}"
echo "Output:  ${OUTPUT_DIR}"
echo "GPUs:    ${NUM_GPUS}"
echo "Frames:  ${NUM_TEMPORAL_FRAMES}"
echo "Action:  ${ACTION_HORIZON}"
echo ""

# Step 1: Check depth maps exist
if [ ! -d "${DEPTH_DIR}" ] || [ -z "$(ls -A ${DEPTH_DIR} 2>/dev/null)" ]; then
    echo "ERROR: Depth maps not found at ${DEPTH_DIR}"
    echo "Run: python scripts/generate_depth_maps.py --data_dir ${DATA_DIR} --output_dir ${DEPTH_DIR}"
    exit 1
fi

# Step 2: Fine-tune with DepthMem
$TORCHRUN --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path "${MODEL_PATH}" \
    --dataset_path "${DATA_DIR}" \
    --embodiment_tag OXE_WIDOWX \
    --num_gpus $NUM_GPUS \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps $MAX_STEPS \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --state_dropout_prob $STATE_DROPOUT \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --use_wandb \
    --gradient_checkpointing \
    --action_horizon $ACTION_HORIZON \
    --depthmem_enabled \
    --depthmem_num_temporal_frames $NUM_TEMPORAL_FRAMES \
    --depthmem_depth_dir "${DEPTH_DIR}" \
    --depthmem_lora_rank 16 \
    --video_backend ffmpeg \
    --shard_size 128

echo "Fine-tuning complete. Checkpoint saved to ${OUTPUT_DIR}"
