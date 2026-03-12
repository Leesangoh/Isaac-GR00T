#!/bin/bash

#SBATCH --job-name=depthmem_pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200000M
#SBATCH --time=3-00:00:00
#SBATCH --output=logs_depthmem_pt/%x_%j.out
#SBATCH --error=logs_depthmem_pt/%x_%j.err
#SBATCH --export=ALL

set -eo pipefail

# ── Config ───────────────────────────────────────────────────────────────
DOCKER_IMAGE="dltkdh9845/gr00t-depthmem:latest"
CODE_DIR="/home2/leesangoh/Isaac-GR00T"     # gr00t code (bind-mounted, editable)
DATA_DIR="/home2/leesangoh/datasets/bridge_lerobot"
DEPTH_DIR="/home2/leesangoh/datasets/bridge_depth_maps"
MODEL_PATH="nvidia/GR00T-N1.6-3B"           # HuggingFace pretrained (auto-download)
OUTPUT_DIR="/home2/leesangoh/checkpoints/GR00T-N1.6-depthmem-pretrained"
HF_CACHE="/home2/leesangoh/.cache/huggingface"
export WANDB_API_KEY="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"

# ── Training hyperparams (A100 80GB × 4) ────────────────────────────────
NUM_GPUS=4
MAX_STEPS=108000         # ~4.5 epochs
GLOBAL_BATCH_SIZE=32     # micro_batch = 32/4GPUs = 8 per GPU (8×16frames = 128 imgs/fwd)
GRAD_ACCUM=2             # effective batch = 32 × 2 = 64
SAVE_STEPS=1000          # checkpoint every 1000 steps

# ── GPU / CPU detection ─────────────────────────────────────────────────
UUIDLIST=$(nvidia-smi -L | cut -d '(' -f 2 | awk '{print$2}' | tr -d ")" | paste -s -d, -)
GPULIST=\"device=${UUIDLIST}\"
CPULIST=$(grep "Cpus_allowed_list" /proc/self/status | awk '{print $2}')

echo "GPUs: ${UUIDLIST}"
echo "CPUs: ${CPULIST}"
nvidia-smi

mkdir -p logs_depthmem_pt

# ── Container naming & cleanup ──────────────────────────────────────────
CONTAINER_NAME="${SLURM_JOB_USER}_job_${SLURM_JOB_ID}_depthmem_pt"
cleanup() {
  docker rm -f ${CONTAINER_NAME} 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT

docker pull ${DOCKER_IMAGE}

# ── Run training in Docker ──────────────────────────────────────────────
# Model is auto-downloaded from HuggingFace on first run and cached in HF_CACHE.
# Resume is automatic: if OUTPUT_DIR has a checkpoint, training continues from it.
docker run \
  --gpus ${GPULIST} \
  --cpuset-cpus ${CPULIST} \
  --ipc=host \
  --network host \
  --name ${CONTAINER_NAME} \
  -e WANDB_API_KEY \
  -e NCCL_TIMEOUT=3600 \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 \
  -e HF_HOME=/cache/huggingface \
  -v ${CODE_DIR}:/workspace/gr00t \
  -v ${DATA_DIR}:/data/bridge_lerobot \
  -v ${DEPTH_DIR}:/data/bridge_depth_maps \
  -v ${OUTPUT_DIR}:/checkpoints/GR00T-N1.6-depthmem-pretrained \
  -v ${HF_CACHE}:/cache/huggingface \
  ${DOCKER_IMAGE} \
  bash -lc '
    set -euo pipefail
    cd /workspace/gr00t

    torchrun --nproc_per_node='"${NUM_GPUS}"' --master_port=29500 \
      gr00t/experiment/launch_finetune.py \
      --base_model_path '"${MODEL_PATH}"' \
      --dataset_path /data/bridge_lerobot \
      --embodiment_tag OXE_WIDOWX \
      --num_gpus '"${NUM_GPUS}"' \
      --output_dir /checkpoints/GR00T-N1.6-depthmem-pretrained \
      --max_steps '"${MAX_STEPS}"' \
      --global_batch_size '"${GLOBAL_BATCH_SIZE}"' \
      --gradient_accumulation_steps '"${GRAD_ACCUM}"' \
      --learning_rate 1e-4 \
      --warmup_ratio 0.05 \
      --weight_decay 1e-5 \
      --state_dropout_prob 0.8 \
      --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
      --dataloader_num_workers 4 \
      --save_steps '"${SAVE_STEPS}"' \
      --save_total_limit 4 \
      --use_wandb \
      --gradient_checkpointing \
      --action_horizon 8 \
      --depthmem_enabled \
      --depthmem_num_temporal_frames 16 \
      --depthmem_depth_dir /data/bridge_depth_maps \
      --depthmem_lora_rank 16 \
      --video_backend ffmpeg \
      --shard_size 128
  '
