"""Offline intent vector extraction from GR00T on BridgeData.

Runs GR00T forward pass on each timestep of each episode and saves
the backbone hidden state (mean-pooled to 2048-dim intent vector).

Output: one .pt file per episode in output_dir/
    {"episode_id": str, "intents": (T, 2048), "task_str": str}

Supports multi-GPU via subprocess (same pattern as generate_vla_actions).
Resume-safe: skips already-processed episodes.

Usage:
    python cerebellar_correction/data/extract_intents.py \
        --dataset_path /mnt/md1/solee/data/bridge_lerobot \
        --model_path /mnt/md1/solee/checkpoints/GR00T-N1.6-bridge \
        --output_dir data/bridge_intents \
        --gpu_ids 0,1,2
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from tqdm import tqdm


log = logging.getLogger(__name__)


def _load_episode_tasks(dataset_path: Path) -> dict[int, str]:
    episode_tasks = {}
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        with open(episodes_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                tasks = entry.get("tasks", [])
                task_str = tasks[0] if tasks else ""
                episode_tasks[entry["episode_index"]] = task_str
    return episode_tasks


def _decode_episode_minimal(args):
    """Decode episode frames and proprios (reuse feature_extractor pattern)."""
    parquet_path, video_dir, image_size = args
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        episode_id = Path(parquet_path).stem

        # Extract proprios
        state_cols = [c for c in df.columns if c.startswith("observation.state")]
        if state_cols:
            proprios = df[state_cols].values.astype(np.float32)
        else:
            proprios = np.zeros((len(df), 8), dtype=np.float32)

        # Decode video frames
        video_path_col = [c for c in df.columns if c.endswith("video_path")]

        if video_path_col:
            rel_path = df[video_path_col[0]].iloc[0]
            video_path = os.path.join(video_dir, rel_path)
        else:
            # Try to find video file matching episode
            ep_idx = episode_id.replace("episode_", "")
            candidates = list(Path(video_dir).rglob(f"*{ep_idx}*"))
            if not candidates:
                return None
            video_path = str(candidates[0])

        import av

        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            if img.shape[0] != image_size or img.shape[1] != image_size:
                from PIL import Image

                img = np.array(
                    Image.fromarray(img).resize((image_size, image_size), Image.BILINEAR)
                )
            frames.append(img)
        container.close()

        frames = np.stack(frames, axis=0)
        ep_len = min(len(frames), len(proprios))
        frames = frames[:ep_len]
        proprios = proprios[:ep_len]

        # Extract actions
        action_cols = [c for c in df.columns if c.startswith("action")]
        if action_cols:
            actions = df[action_cols].values[:ep_len].astype(np.float32)
        else:
            actions = np.zeros((ep_len, 7), dtype=np.float32)

        return {
            "episode_id": episode_id,
            "frames": frames,
            "proprios": proprios,
            "actions": actions,
        }
    except Exception as e:
        log.warning("Failed to decode %s: %s", parquet_path, e)
        return None


def _run_single_gpu(
    gpu_id: int,
    model_path: str,
    dataset_path: str,
    output_dir: str,
    parquet_files: list[str],
    image_size: int,
    decode_workers: int,
    inference_batch_size: int = 64,
):
    """Extract intents on a single GPU for a subset of episodes."""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    from cerebellar_correction.models.intent_extractor import IntentExtractor

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{gpu_id}"
    log.info("[GPU %d] Loading GR00T on %s...", gpu_id, device)
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.OXE_WIDOWX,
        model_path=model_path,
        device=device,
        strict=False,
    )
    intent_extractor = IntentExtractor(policy.model)

    episode_tasks = _load_episode_tasks(dataset_path)
    video_dir = dataset_path / "videos"

    worker_args = [(p, str(video_dir), image_size) for p in parquet_files]
    pool = mp.Pool(processes=decode_workers)
    results_iter = pool.imap_unordered(_decode_episode_minimal, worker_args, chunksize=4)

    done = 0
    skipped = 0
    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]

    pbar = tqdm(total=len(parquet_files), desc=f"GPU {gpu_id} intents", position=gpu_id)

    for result in results_iter:
        pbar.update(1)

        if result is None:
            skipped += 1
            continue

        episode_id = result["episode_id"]
        out_path = output_dir / f"{episode_id}.pt"
        if out_path.exists():
            skipped += 1
            continue

        frames = result["frames"]  # (T, H, W, 3) uint8
        proprios = result["proprios"]  # (T, proprio_dim)
        ep_len = len(frames)

        try:
            ep_idx = int(episode_id.replace("episode_", ""))
        except ValueError:
            ep_idx = -1

        task_str = episode_tasks.get(ep_idx, "")
        if not task_str:
            task_str = "manipulate object"

        # Batched inference for intent extraction
        all_intents = []
        for start in range(0, ep_len, inference_batch_size):
            end = min(start + inference_batch_size, ep_len)
            b = end - start

            # Build observation batch
            batch_frames = frames[start:end]  # (B, H, W, 3)
            batch_proprios = proprios[start:end]  # (B, D)

            video = {"image_0": batch_frames[:, np.newaxis, ...].astype(np.uint8)}
            state = {}
            for i, key in enumerate(state_keys):
                if i < batch_proprios.shape[1]:
                    vals = batch_proprios[:, i : i + 1]
                else:
                    vals = np.zeros((b, 1), dtype=np.float32)
                state[key] = vals[:, np.newaxis, :].astype(np.float32)
            language = {"annotation.human.action.task_description": [[task_str]] * b}
            obs = {"video": video, "state": state, "language": language}

            try:
                policy.get_action(obs)
                intent_batch = intent_extractor.get_intent()  # (B, 2048)
                all_intents.append(intent_batch.cpu())
            except Exception as e:
                log.warning(
                    "Intent extraction failed for %s [%d-%d]: %s", episode_id, start, end, e
                )
                all_intents.append(torch.zeros(b, 2048))

        intents = torch.cat(all_intents, dim=0)  # (T, 2048)

        torch.save(
            {"episode_id": episode_id, "intents": intents, "task_str": task_str},
            out_path,
        )
        done += 1

    pbar.close()
    pool.close()
    pool.join()
    log.info("[GPU %d] Done: %d, Skipped: %d", gpu_id, done, skipped)


def extract_intents(
    dataset_path: str,
    model_path: str,
    output_dir: str,
    gpu_ids: list[int] | None = None,
    decode_workers: int = 16,
    image_size: int = 224,
    max_episodes: int = 0,
):
    if gpu_ids is None:
        gpu_ids = [0]

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = len(gpu_ids)

    data_dir = dataset_path / "data"
    if not data_dir.exists():
        data_dir = dataset_path
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    log.info("Found %d episode parquet files", len(parquet_files))

    already_done = set(p.stem for p in output_dir.glob("episode_*.pt"))
    remaining = [p for p in parquet_files if p.stem not in already_done]
    log.info("Skipping %d already processed, %d remaining", len(already_done), len(remaining))

    if max_episodes > 0:
        remaining = remaining[:max_episodes]

    if not remaining:
        log.info("All episodes already processed.")
        return

    remaining_strs = [str(p) for p in remaining]

    if num_gpus == 1:
        _run_single_gpu(
            gpu_ids[0],
            model_path,
            str(dataset_path),
            str(output_dir),
            remaining_strs,
            image_size,
            max(decode_workers, 1),
        )
    else:
        per_gpu = [[] for _ in range(num_gpus)]
        for i, p in enumerate(remaining_strs):
            per_gpu[i % num_gpus].append(p)

        workers_per_gpu = max(decode_workers // num_gpus, 1)
        procs = []
        for i, gpu_id in enumerate(gpu_ids):
            if not per_gpu[i]:
                continue
            list_file = output_dir / f".gpu{gpu_id}_episodes.json"
            with open(list_file, "w") as f:
                json.dump(per_gpu[i], f)

            cmd = [
                sys.executable,
                __file__,
                "--dataset_path",
                str(dataset_path),
                "--model_path",
                model_path,
                "--output_dir",
                str(output_dir),
                "--gpu_ids",
                str(gpu_id),
                "--decode_workers",
                str(workers_per_gpu),
                "--image_size",
                str(image_size),
                "--_episode_list",
                str(list_file),
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            proc = subprocess.Popen(cmd, env=env)
            procs.append((gpu_id, proc))

        for gpu_id, proc in procs:
            ret = proc.wait()
            if ret != 0:
                log.error("GPU %d subprocess exited with code %d", gpu_id, ret)

        for i, gpu_id in enumerate(gpu_ids):
            list_file = output_dir / f".gpu{gpu_id}_episodes.json"
            if list_file.exists():
                list_file.unlink()

    # Save metadata
    all_ids = sorted(p.stem for p in output_dir.glob("episode_*.pt"))
    metadata = {"num_episodes": len(all_ids), "intent_dim": 2048, "model_path": model_path}
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Intent extraction complete: %d episodes", len(all_ids))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Extract intent vectors from GR00T")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--decode_workers", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--_episode_list", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]

    if args._episode_list:
        mp.set_start_method("spawn", force=True)
        with open(args._episode_list) as f:
            episode_list = json.load(f)
        _run_single_gpu(
            gpu_id=0,
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            parquet_files=episode_list,
            image_size=args.image_size,
            decode_workers=args.decode_workers,
        )
    else:
        if len(gpu_ids) == 1:
            mp.set_start_method("spawn", force=True)
        extract_intents(
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            output_dir=args.output_dir,
            gpu_ids=gpu_ids,
            decode_workers=args.decode_workers,
            image_size=args.image_size,
            max_episodes=args.max_episodes,
        )
