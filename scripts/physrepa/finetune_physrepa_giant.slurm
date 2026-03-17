#!/bin/bash

#SBATCH --job-name=physrepa_giant_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200000M
#SBATCH --time=3-00:00:00
#SBATCH --output=logs_physrepa/%x_%j.out
#SBATCH --error=logs_physrepa/%x_%j.err
#SBATCH --export=ALL

set -eo pipefail

# ── Config ───────────────────────────────────────────────────────────────
DOCKER_IMAGE="dltkdh9845/gr00t-physrepa:latest"
CODE_DIR="/home2/leesangoh/Isaac-GR00T"
DATA_DIR="/home2/leesangoh/datasets/bridge_lerobot"
MODEL_PATH="/home2/leesangoh/checkpoints/GR00T-N1.6-3B"
OUTPUT_DIR="/home2/leesangoh/checkpoints/GR00T-N1.6-physrepa-vitg"
FEATURES_DIR="/home2/leesangoh/features/vjepa2_vitg"
export WANDB_NAME="physrepa_giant_bridge_100k"
export WANDB_API_KEY="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"

# ── Training hyperparams (A100 80GB × 4) ────────────────────────────────
NUM_GPUS=4
MAX_STEPS=100000
GLOBAL_BATCH_SIZE=512
SAVE_STEPS=1000

# ── GPU / CPU detection ─────────────────────────────────────────────────
UUIDLIST=$(nvidia-smi -L | cut -d '(' -f 2 | awk '{print$2}' | tr -d ")" | paste -s -d, -)
GPULIST=\"device=${UUIDLIST}\"
CPULIST=$(grep "Cpus_allowed_list" /proc/self/status | awk '{print $2}')

echo "GPUs: ${UUIDLIST}"
echo "CPUs: ${CPULIST}"
nvidia-smi

mkdir -p logs_physrepa

# ── Container naming & cleanup ──────────────────────────────────────────
CONTAINER_NAME="${SLURM_JOB_USER}_job_${SLURM_JOB_ID}_physrepa_giant"
cleanup() {
  docker rm -f ${CONTAINER_NAME} 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

docker pull ${DOCKER_IMAGE}

# ── Run training in Docker ──────────────────────────────────────────────
docker run \
  --gpus ${GPULIST} \
  --cpuset-cpus ${CPULIST} \
  --ipc=host \
  --network host \
  --name ${CONTAINER_NAME} \
  -e WANDB_API_KEY \
  -e WANDB_NAME \
  -e NCCL_TIMEOUT=3600 \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
  -v ${CODE_DIR}:/workspace/gr00t \
  -v ${DATA_DIR}:/data/bridge_lerobot \
  -v ${OUTPUT_DIR}:/checkpoints/GR00T-N1.6-physrepa-vitg \
  -v ${FEATURES_DIR}:/data/vjepa2_features \
  ${DOCKER_IMAGE} \
  bash -lc '
    set -euo pipefail
    cd /workspace/gr00t

    torchrun --nproc_per_node='"${NUM_GPUS}"' --master_port=29502 \
      gr00t/experiment/launch_finetune.py \
      --base_model_path nvidia/GR00T-N1.6-3B \
      --dataset_path /data/bridge_lerobot \
      --embodiment_tag OXE_WIDOWX \
      --num_gpus '"${NUM_GPUS}"' \
      --output_dir /checkpoints/GR00T-N1.6-physrepa-vitg \
      --max_steps '"${MAX_STEPS}"' \
      --global_batch_size '"${GLOBAL_BATCH_SIZE}"' \
      --learning_rate 1e-4 \
      --warmup_ratio 0.05 \
      --weight_decay 1e-5 \
      --state_dropout_prob 0.8 \
      --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
      --dataloader_num_workers 4 \
      --save_steps '"${SAVE_STEPS}"' \
      --save_total_limit 5 \
      --use_wandb \
      --physrepa-enabled \
      --physrepa-lambda 0.5 \
      --physrepa-vjepa-features-dir /data/vjepa2_features \
      --physrepa-vjepa-layer 10 \
      --physrepa-vjepa-dim 1408 \
      --physrepa-align-layers 0 1 2 3 4 5 6 7 8
  '
