gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 5 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=300 \
    --env_name simpler_env_widowx/widowx_stack_cube \
    --n_action_steps 8 \
    --n_envs 5