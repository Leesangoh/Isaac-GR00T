#!/bin/bash
# Step 5b: Run client-side cerebellum eval with per-step correction
# Connects to step5a server, loads DINOv2+correction_net locally, corrects every step.
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/../.." && pwd)"

CEREBELLUM_CKPT=${CEREBELLUM_CKPT:-checkpoints/cerebellum_intent}
CORRECTION_ALPHA=${CORRECTION_ALPHA:-1.0}
MAX_CORRECTION=${MAX_CORRECTION:-0.15}

gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python \
    cerebellar_correction/eval/run_cerebellum_client.py \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --cerebellum_ckpt ${CEREBELLUM_CKPT} \
    --correction_alpha ${CORRECTION_ALPHA} \
    --max_correction ${MAX_CORRECTION} \
    --env_name simpler_env_widowx/widowx_stack_cube \
    --n_episodes 50 \
    --max_episode_steps 300 \
    --n_envs 1 \
    --n_action_steps 1
