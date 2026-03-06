"""Client-side rollout with local cerebellar correction (Option 3).

Connects to a GR00T+IntentPassthrough server, receives action chunks + intent,
and applies per-step cerebellar correction locally using DINOv2 + correction net.

Usage:
    gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python \
        cerebellar_correction/eval/run_cerebellum_client.py \
        --policy_client_host 127.0.0.1 \
        --policy_client_port 5555 \
        --cerebellum_ckpt checkpoints/cerebellum_intent \
        --env_name simpler_env_widowx/widowx_stack_cube
"""

import argparse
import uuid

from gr00t.eval.rollout_policy import (
    MultiStepConfig,
    VideoConfig,
    WrapperConfigs,
    run_rollout_gymnasium_policy,
)
import numpy as np

from cerebellar_correction.policy.client_cerebellum_policy import ClientSideCerebellumPolicy


def main():
    parser = argparse.ArgumentParser(
        description="Client-side cerebellum eval with per-step correction"
    )
    parser.add_argument("--policy_client_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_client_port", type=int, default=5555)
    parser.add_argument(
        "--cerebellum_ckpt",
        type=str,
        default="checkpoints/cerebellum_intent",
    )
    parser.add_argument("--correction_alpha", type=float, default=1.0)
    parser.add_argument("--max_correction", type=float, default=0.15)
    parser.add_argument("--cerebellum_device", type=str, default="cuda")
    parser.add_argument(
        "--env_name",
        type=str,
        default="simpler_env_widowx/widowx_stack_cube",
    )
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--max_episode_steps", type=int, default=300)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--n_action_steps", type=int, default=1)

    args = parser.parse_args()

    video_dir = (
        f"cerebellum_eval_{args.env_name.split('/')[-1]}_a{args.correction_alpha}_{uuid.uuid4()}"
    )

    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=args.max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = ClientSideCerebellumPolicy(
        host=args.policy_client_host,
        port=args.policy_client_port,
        cerebellum_ckpt=args.cerebellum_ckpt,
        device=args.cerebellum_device,
        correction_alpha=args.correction_alpha,
        max_correction=args.max_correction,
    )

    results = run_rollout_gymnasium_policy(
        env_name=args.env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
    )

    print("Video saved to:", wrapper_configs.video.video_dir)
    print("Results:", results[0])
    print("Success rate:", np.mean(results[1]))


if __name__ == "__main__":
    main()
