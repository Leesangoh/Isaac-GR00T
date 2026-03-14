"""
Phase 2: Multi-Sample Action Extraction
=========================================
For each observation, run GR00T N1.6 inference K times to collect
action variability samples.

GR00T stochasticity sources:
  1. state_dropout_prob=0.8 (proprioception masked 80% of the time)
  2. Flow matching initial noise (torch.randn)

Both are resampled every forward pass → K different action samples per observation.

Usage:
  python -m ucm_analysis.phase2_extract_samples \
    --model_path /mnt/md1/solee/checkpoints/GR00T-N1.6-bridge \
    --dataset_path /mnt/md1/solee/data/bridge_lerobot \
    --n_episodes 50 --K 30 --device cuda:3 \
    --output_dir ucm_analysis/results/multi_samples
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


def parse_observation(obs_dict, modality_configs):
    """Convert step data to policy-expected observation format."""
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        for key in modality_configs[modality].modality_keys:
            if modality == "language":
                parsed_key = key
            else:
                parsed_key = f"{modality}.{key}"
            arr = obs_dict[parsed_key]
            if isinstance(arr, str):
                new_obs[modality][key] = [[arr]]
            else:
                new_obs[modality][key] = arr[None, :]
    return new_obs


def prepare_observation(traj, step_idx, modality_configs, embodiment_tag, loader):
    """Prepare a single observation from trajectory data."""
    data_point = extract_step_data(traj, step_idx, modality_configs, embodiment_tag)

    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)
    for language_key in loader.modality_configs["language"].modality_keys:
        obs[language_key] = data_point.text

    return parse_observation(obs, loader.modality_configs)


def extract_action_chunk(action_dict, action_keys):
    """Convert policy output action dict to (T, D) numpy array.
    action_keys must be in canonical order: [x, y, z, roll, pitch, yaw, gripper]
    """
    arrays = []
    for key in action_keys:  # Use original order, NOT sorted
        arr = action_dict[key][0]  # Remove batch dim → (T, 1)
        arrays.append(arr)
    return np.concatenate(arrays, axis=-1)  # (T, D)


def get_expert_action_chunk(traj, step_idx, action_keys, n_steps=8):
    """Get 8-step expert action chunk starting at step_idx.
    action_keys must be in canonical order: [x, y, z, roll, pitch, yaw, gripper]
    """
    T = len(traj)
    chunk = np.zeros((n_steps, len(action_keys)), dtype=np.float32)
    for s in range(n_steps):
        idx = min(step_idx + s, T - 1)
        row = traj.iloc[idx]
        for d, key in enumerate(action_keys):  # Use original order
            col_name = f"action.{key}"
            val = row[col_name]
            if hasattr(val, '__len__'):
                chunk[s, d] = val[0] if len(val) == 1 else val
            else:
                chunk[s, d] = float(val)
    return chunk


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Multi-sample action extraction")
    parser.add_argument("--model_path", default="/mnt/md1/solee/checkpoints/GR00T-N1.6-bridge")
    parser.add_argument("--dataset_path", default="/mnt/md1/solee/data/bridge_lerobot")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--K", type=int, default=30, help="Number of samples per observation")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output_dir", default="ucm_analysis/results/multi_samples")
    parser.add_argument("--step_stride", type=int, default=3,
                        help="Sample every N-th timestep to reduce compute")
    parser.add_argument("--episode_offset", type=int, default=0,
                        help="Start from this episode index")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 2: Multi-Sample Action Extraction")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Episodes: {args.n_episodes}, K: {args.K}, stride: {args.step_stride}")
    print(f"Device: {args.device}")

    # Load policy
    print("\nLoading GR00T policy...")
    t0 = time.time()
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.OXE_WIDOWX,
        model_path=args.model_path,
        device=args.device,
    )
    print(f"Policy loaded in {time.time() - t0:.1f}s")

    # Get modality configs
    modality_configs = policy.get_modality_config()
    obs_modality_configs = deepcopy(modality_configs)
    obs_modality_configs.pop("action")
    action_keys = modality_configs["action"].modality_keys

    # Load dataset
    print("Loading dataset loader...")
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_configs,
        video_backend="ffmpeg",
    )
    n_total_episodes = len(loader)
    print(f"Dataset has {n_total_episodes} episodes")

    # Select episode indices (spread across dataset for diversity)
    np.random.seed(42)
    episode_indices = np.random.choice(
        n_total_episodes, size=min(args.n_episodes, n_total_episodes), replace=False
    )
    episode_indices.sort()
    print(f"Selected {len(episode_indices)} episodes")

    # Extraction loop
    total_inferences = 0
    total_time = 0
    extraction_start = time.time()

    for ep_count, ep_idx in enumerate(episode_indices):
        ep_idx = int(ep_idx)
        output_path = os.path.join(args.output_dir, f"episode_{ep_idx:06d}.npz")

        # Skip if already extracted
        if os.path.exists(output_path):
            print(f"[{ep_count+1}/{len(episode_indices)}] Episode {ep_idx} already exists, skipping")
            continue

        # Load trajectory
        try:
            traj = loader[ep_idx]
        except Exception as e:
            print(f"[{ep_count+1}/{len(episode_indices)}] Episode {ep_idx} load failed: {e}")
            continue

        traj_length = len(traj)

        # Select timesteps (with stride to reduce compute)
        timestep_indices = list(range(0, traj_length, args.step_stride))
        n_timesteps = len(timestep_indices)

        # Allocate arrays
        all_samples = np.zeros((n_timesteps, args.K, 8, 7), dtype=np.float32)
        expert_actions = np.zeros((n_timesteps, 8, 7), dtype=np.float32)
        timestep_ids = np.array(timestep_indices, dtype=np.int32)

        ep_start = time.time()

        for t_idx, step_idx in enumerate(timestep_indices):
            # Prepare observation (done once per timestep)
            try:
                obs = prepare_observation(traj, step_idx, obs_modality_configs,
                                          EmbodimentTag.OXE_WIDOWX, loader)
            except Exception as e:
                print(f"  Step {step_idx} obs prep failed: {e}")
                continue

            # Get expert action chunk
            expert_actions[t_idx] = get_expert_action_chunk(
                traj, step_idx, action_keys, n_steps=8
            )

            # Run K inference samples
            for k in range(args.K):
                with torch.no_grad():
                    action, info = policy.get_action(obs)

                # Convert action dict to array
                action_chunk = extract_action_chunk(action, action_keys)
                # Ensure we get 8 steps x 7 dims
                all_samples[t_idx, k] = action_chunk[:8, :7]
                total_inferences += 1

        ep_time = time.time() - ep_start
        total_time += ep_time

        # Save
        np.savez_compressed(
            output_path,
            samples=all_samples,        # (n_timesteps, K, 8, 7)
            expert=expert_actions,      # (n_timesteps, 8, 7)
            timestep_ids=timestep_ids,  # (n_timesteps,)
            episode_idx=ep_idx,
            traj_length=traj_length,
        )

        avg_ms = (ep_time / (n_timesteps * args.K) * 1000) if (n_timesteps * args.K) > 0 else 0
        elapsed = time.time() - extraction_start
        remaining_eps = len(episode_indices) - (ep_count + 1)
        eta = (elapsed / (ep_count + 1)) * remaining_eps if ep_count > 0 else 0

        print(f"[{ep_count+1}/{len(episode_indices)}] Episode {ep_idx}: "
              f"{n_timesteps} steps x {args.K} samples = {n_timesteps * args.K} inferences "
              f"in {ep_time:.1f}s ({avg_ms:.1f}ms/inf), ETA: {eta/60:.0f}min")

    print(f"\nDone! Total: {total_inferences} inferences in {total_time:.0f}s")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
