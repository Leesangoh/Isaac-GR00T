#!/bin/bash
set -x -e

# Default: widowx_carrot_on_plate, 200 episodes, 5 parallel envs
ENV_NAME="${1:-simpler_env_widowx/widowx_carrot_on_plate}"
N_EPISODES="${2:-200}"
N_ENVS="${3:-5}"

gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes "$N_EPISODES" \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps 300 \
    --env_name "$ENV_NAME" \
    --n_action_steps 1 \
    --n_envs "$N_ENVS"
