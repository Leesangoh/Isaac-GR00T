"""Offline extraction of intent token sequences + VLA actions from GR00T on BridgeData.

Single forward pass per timestep extracts both:
  1. Intent token sequences (backbone features, (T, 128, 2048) fp16) + attention masks (T, 128)
  2. VLA action chunks (8-step, 7-dim) from the action head

Output: two directories with one .pt file per episode:
  intent_dir/episode_NNNNNN.pt:
      {"episode_id": str, "intent_tokens": (T, 128, 2048) float16,
       "attention_masks": (T, 128) bool, "task_str": str}
  vla_dir/episode_NNNNNN.pt:
      {"episode_id": str, "action_expert": (T, 7), "action_vla_chunks": (T, 8, 7),
       "proprios": (T, 8)}

Optimized pipeline:
  - Prefetch thread overlaps video decoding with GPU inference
  - Cross-episode mega-batching maximizes GPU utilization
  - Background thread for saving .pt files (non-blocking I/O)
  - Multi-GPU via subprocess with LD_PRELOAD inheritance

Supports multi-GPU via subprocess. Resume-safe: skips already-processed episodes.

Usage:
    python cerebellar_correction/data/extract_all.py \
        --dataset_path /mnt/md1/solee/data/bridge_lerobot \
        --model_path /mnt/md1/solee/checkpoints/GR00T-N1.6-bridge \
        --intent_dir data/bridge_intents \
        --vla_dir data/vla_actions \
        --gpu_ids 0,1,2,3
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import numpy as np
import torch
from tqdm import tqdm


log = logging.getLogger(__name__)

CHUNK_SIZE = 8
ACTION_DIM = 7
# WidowX action key order — must match parquet column order (x,y,z,roll,pitch,yaw,gripper)
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


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


def _decode_episode(args):
    """Decode episode frames, proprios, and actions from parquet + video."""
    parquet_path, video_dir, image_size = args
    try:
        import av
        import pandas as pd

        parquet_path = Path(parquet_path)
        df = pd.read_parquet(parquet_path)
        episode_id = parquet_path.stem
        chunk_name = parquet_path.parent.name

        # Actions
        if "action" in df.columns:
            actions = np.stack(df["action"].values).astype(np.float32)
        else:
            action_cols = sorted(c for c in df.columns if c.startswith("action."))
            if action_cols:
                actions = df[action_cols].values.astype(np.float32)
            else:
                return None

        # Proprios
        if "observation.state" in df.columns:
            proprios = np.stack(df["observation.state"].values).astype(np.float32)
        else:
            state_cols = sorted(c for c in df.columns if c.startswith("observation.state."))
            if state_cols:
                proprios = df[state_cols].values.astype(np.float32)
            else:
                proprios = np.zeros((len(df), 8), dtype=np.float32)

        # Video
        video_path = (
            Path(video_dir) / chunk_name / "observation.images.image_0" / f"{episode_id}.mp4"
        )
        if not video_path.exists():
            return None

        num_frames = len(df)
        container = av.open(str(video_path))
        frames = []
        for frame in container.decode(video=0):
            f = frame.reformat(width=image_size, height=image_size).to_ndarray(format="rgb24")
            frames.append(f)
            if len(frames) >= num_frames:
                break
        container.close()

        if len(frames) == 0:
            return None

        while len(frames) < num_frames:
            frames.append(frames[-1])

        frames_np = np.stack(frames)
        ep_len = min(len(frames_np), len(proprios), len(actions))

        return {
            "episode_id": episode_id,
            "frames": frames_np[:ep_len],
            "proprios": proprios[:ep_len],
            "actions": actions[:ep_len],
        }
    except Exception as e:
        log.warning("Failed to decode %s: %s", parquet_path, e)
        return None


def _extract_chunks_batched(action_dict: dict) -> np.ndarray:
    """Extract (B, 8, 7) action chunks from GR00T's batched output dict.

    Uses ACTION_KEYS order to match parquet expert action column order.
    """
    if len(action_dict) == 1:
        arr = next(iter(action_dict.values()))  # (B, 8, D)
        return arr[:, :, :ACTION_DIM].astype(np.float32)
    chunks = [action_dict[k] for k in ACTION_KEYS if k in action_dict]
    chunk = np.concatenate(chunks, axis=-1)
    return chunk[:, :, :ACTION_DIM].astype(np.float32)


def _build_obs_batch(frames, proprios, task_str, state_keys):
    """Build GR00T observation dict from numpy arrays."""
    b = len(frames)
    video = {"image_0": frames[:, np.newaxis, ...].astype(np.uint8)}
    state = {}
    for i, key in enumerate(state_keys):
        if i < proprios.shape[1]:
            vals = proprios[:, i : i + 1]
        else:
            vals = np.zeros((b, 1), dtype=np.float32)
        state[key] = vals[:, np.newaxis, :].astype(np.float32)
    language = {"annotation.human.action.task_description": [[task_str]] * b}
    return {"video": video, "state": state, "language": language}


def _saver_worker(save_queue: queue.Queue):
    """Background thread that saves files without blocking GPU.

    Items are (path, data) tuples. If path ends with .npz, saves with np.savez;
    otherwise uses torch.save.
    """
    while True:
        item = save_queue.get()
        if item is None:
            break
        path, data = item
        path = Path(path)
        if path.suffix == ".npz":
            np.savez(path, **data)
        else:
            torch.save(data, path)
        save_queue.task_done()


def _run_single_gpu(
    gpu_id: int,
    model_path: str,
    dataset_path: str,
    intent_dir: str,
    vla_dir: str,
    parquet_files: list[str],
    image_size: int,
    decode_workers: int,
    inference_batch_size: int = 32,
    prefetch_size: int = 64,
):
    """Extract intents + VLA actions on a single GPU.

    Optimized pipeline:
      1. Decode pool fills a prefetch queue (overlaps with GPU)
      2. Main thread collects frames across episodes into mega-batches
      3. GPU inference on mega-batches (better utilization than per-episode)
      4. Background thread saves .pt files (non-blocking I/O)
    """
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    from cerebellar_correction.models.intent_extractor import IntentExtractor

    dataset_path = Path(dataset_path)
    intent_dir = Path(intent_dir)
    vla_dir = Path(vla_dir)
    intent_dir.mkdir(parents=True, exist_ok=True)
    vla_dir.mkdir(parents=True, exist_ok=True)

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
    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]

    # --- Prefetch thread: decode episodes ahead of GPU ---
    prefetch_q: queue.Queue = queue.Queue(maxsize=prefetch_size)
    worker_args = [(p, str(video_dir), image_size) for p in parquet_files]

    def _prefetch_producer():
        pool = mp.Pool(processes=decode_workers)
        for result in pool.imap_unordered(_decode_episode, worker_args, chunksize=8):
            prefetch_q.put(result)
        prefetch_q.put(None)  # sentinel
        pool.close()
        pool.join()

    prefetch_thread = threading.Thread(target=_prefetch_producer, daemon=True)
    prefetch_thread.start()

    # --- Background saver thread ---
    save_q: queue.Queue = queue.Queue(maxsize=128)
    saver_thread = threading.Thread(target=_saver_worker, args=(save_q,), daemon=True)
    saver_thread.start()

    done = 0
    skipped = 0
    pbar = tqdm(total=len(parquet_files), desc=f"GPU {gpu_id}", position=gpu_id)

    # --- Mega-batch accumulator ---
    # Collect frames from multiple episodes, run inference in large batches
    buf_frames = []  # list of frame arrays
    buf_proprios = []
    buf_task_strs = []
    buf_meta = []  # (episode_id, start_in_buf, end_in_buf, task_str, actions, proprios)
    buf_total = 0

    def _flush_buffer():
        """Run inference on accumulated buffer and save results."""
        nonlocal buf_frames, buf_proprios, buf_task_strs, buf_meta, buf_total, done

        if not buf_meta:
            return

        # Concatenate all frames across episodes
        all_frames = np.concatenate(buf_frames, axis=0)
        all_proprios = np.concatenate(buf_proprios, axis=0)
        total = len(all_frames)

        # Pre-allocate output arrays
        all_intent_tokens = []
        all_attention_masks = []
        all_vla_chunks = np.zeros((total, CHUNK_SIZE, ACTION_DIM), dtype=np.float32)

        # Find the dominant task string (most frames use same task within a mega-batch)
        # We need per-frame task strings for GR00T, but episodes in the buffer
        # may have different tasks. Process in sub-batches by task_str.
        # Build index mapping: for each position in all_frames, which task_str?
        frame_task_strs = []
        for ep_id, start, end, task_str, _, _ in buf_meta:
            frame_task_strs.extend([task_str] * (end - start))

        # Run inference in chunks of inference_batch_size
        for batch_start in range(0, total, inference_batch_size):
            batch_end = min(batch_start + inference_batch_size, total)
            b = batch_end - batch_start

            batch_frames = all_frames[batch_start:batch_end]
            batch_proprios = all_proprios[batch_start:batch_end]
            # Use first frame's task_str for the batch (frames within inference_batch_size
            # are usually from same or consecutive episodes with same task)
            task_str = frame_task_strs[batch_start]

            obs = _build_obs_batch(batch_frames, batch_proprios, task_str, state_keys)

            try:
                action_dict, _ = policy.get_action(obs)
                all_vla_chunks[batch_start:batch_end] = _extract_chunks_batched(action_dict)
                tokens, masks = intent_extractor.get_intent_tokens()
                all_intent_tokens.append(tokens.half().cpu())  # fp16 to save memory
                all_attention_masks.append(masks.cpu())
            except Exception as e:
                log.warning("Inference failed [%d-%d]: %s", batch_start, batch_end, e)
                max_t = intent_extractor.MAX_TOKENS
                all_intent_tokens.append(torch.zeros(b, max_t, 2048, dtype=torch.float16))
                all_attention_masks.append(torch.zeros(b, max_t, dtype=torch.bool))

        all_tokens_t = torch.cat(all_intent_tokens, dim=0)
        all_masks_t = torch.cat(all_attention_masks, dim=0)

        # Scatter results back to per-episode outputs and save
        for ep_id, start, end, task_str, actions_expert, ep_proprios in buf_meta:
            ep_tokens = all_tokens_t[start:end]  # (T, 128, 2048) fp16
            ep_masks = all_masks_t[start:end]  # (T, 128) bool
            vla_chunks = all_vla_chunks[start:end]
            actions_trimmed = actions_expert[:, :ACTION_DIM].astype(np.float32)

            save_q.put(
                (
                    intent_dir / f"{ep_id}.npz",
                    {
                        "intent_tokens": ep_tokens.numpy(),  # (T, 128, 2048) fp16
                        "attention_masks": ep_masks.numpy(),  # (T, 128) bool
                    },
                )
            )
            save_q.put(
                (
                    vla_dir / f"{ep_id}.pt",
                    {
                        "episode_id": ep_id,
                        "action_expert": torch.from_numpy(actions_trimmed),
                        "action_vla_chunks": torch.from_numpy(vla_chunks.copy()),
                        "proprios": torch.from_numpy(ep_proprios.astype(np.float32)),
                    },
                )
            )
            done += 1

        # Clear buffer
        buf_frames.clear()
        buf_proprios.clear()
        buf_task_strs.clear()
        buf_meta.clear()
        buf_total = 0

    # --- Main loop: drain prefetch queue into mega-batches ---
    while True:
        result = prefetch_q.get()

        if result is None:
            # End of decode stream — flush remaining
            _flush_buffer()
            pbar.update(0)
            break

        pbar.update(1)

        if result is None:
            skipped += 1
            continue

        episode_id = result["episode_id"]
        intent_path = intent_dir / f"{episode_id}.npz"
        vla_path = vla_dir / f"{episode_id}.pt"

        if intent_path.exists() and vla_path.exists():
            skipped += 1
            continue

        frames = result["frames"]
        proprios = result["proprios"]
        actions_expert = result["actions"]
        ep_len = len(frames)

        try:
            ep_idx = int(episode_id.replace("episode_", ""))
        except ValueError:
            ep_idx = -1
        task_str = episode_tasks.get(ep_idx, "") or "manipulate object"

        # Add to mega-batch buffer
        start_in_buf = buf_total
        end_in_buf = buf_total + ep_len
        buf_frames.append(frames)
        buf_proprios.append(proprios)
        buf_task_strs.append(task_str)
        buf_meta.append((episode_id, start_in_buf, end_in_buf, task_str, actions_expert, proprios))
        buf_total += ep_len

        # Flush when buffer is large enough for a good mega-batch
        if buf_total >= inference_batch_size * 4:
            _flush_buffer()

    pbar.close()

    # Wait for all saves to complete
    save_q.join()
    save_q.put(None)
    saver_thread.join()

    prefetch_thread.join()
    log.info("[GPU %d] Done: %d, Skipped: %d", gpu_id, done, skipped)


def extract_all(
    dataset_path: str,
    model_path: str,
    intent_dir: str,
    vla_dir: str,
    gpu_ids: list[int] | None = None,
    decode_workers: int = 16,
    image_size: int = 224,
    max_episodes: int = 0,
    inference_batch_size: int = 32,
):
    if gpu_ids is None:
        gpu_ids = [0]

    dataset_path = Path(dataset_path)
    intent_dir_p = Path(intent_dir)
    vla_dir_p = Path(vla_dir)
    intent_dir_p.mkdir(parents=True, exist_ok=True)
    vla_dir_p.mkdir(parents=True, exist_ok=True)

    num_gpus = len(gpu_ids)

    data_dir = dataset_path / "data"
    if not data_dir.exists():
        data_dir = dataset_path
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    log.info("Found %d episode parquet files", len(parquet_files))

    # Skip episodes that have BOTH intent and VLA already done
    intent_done = set(p.stem for p in intent_dir_p.glob("episode_*.pt"))
    vla_done = set(p.stem for p in vla_dir_p.glob("episode_*.pt"))
    both_done = intent_done & vla_done
    remaining = [p for p in parquet_files if p.stem not in both_done]
    log.info("Skipping %d already processed, %d remaining", len(both_done), len(remaining))

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
            intent_dir,
            vla_dir,
            remaining_strs,
            image_size,
            max(decode_workers, 1),
            inference_batch_size=inference_batch_size,
        )
    else:
        per_gpu = [[] for _ in range(num_gpus)]
        for i, p in enumerate(remaining_strs):
            per_gpu[i % num_gpus].append(p)

        workers_per_gpu = max(decode_workers // num_gpus, 4)
        procs = []
        for i, gpu_id in enumerate(gpu_ids):
            if not per_gpu[i]:
                continue
            list_file = intent_dir_p / f".gpu{gpu_id}_episodes.json"
            with open(list_file, "w") as f:
                json.dump(per_gpu[i], f)

            cmd = [
                sys.executable,
                __file__,
                "--dataset_path",
                str(dataset_path),
                "--model_path",
                model_path,
                "--intent_dir",
                intent_dir,
                "--vla_dir",
                vla_dir,
                "--gpu_ids",
                str(gpu_id),
                "--decode_workers",
                str(workers_per_gpu),
                "--image_size",
                str(image_size),
                "--inference_batch_size",
                str(inference_batch_size),
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
            list_file = intent_dir_p / f".gpu{gpu_id}_episodes.json"
            if list_file.exists():
                list_file.unlink()

    # Save metadata for both
    intent_ids = sorted(p.stem for p in intent_dir_p.glob("episode_*.npz"))
    with open(intent_dir_p / "metadata.json", "w") as f:
        json.dump(
            {
                "num_episodes": len(intent_ids),
                "intent_dim": 2048,
                "max_tokens": 128,
                "format": "token_sequence",
                "dtype": "float16",
                "model_path": model_path,
            },
            f,
            indent=2,
        )

    vla_ids = sorted(p.stem for p in vla_dir_p.glob("episode_*.pt"))
    with open(vla_dir_p / "metadata.json", "w") as f:
        json.dump(
            {
                "num_episodes": len(vla_ids),
                "model_path": model_path,
                "chunk_size": CHUNK_SIZE,
                "action_dim": ACTION_DIM,
            },
            f,
            indent=2,
        )

    log.info("Extraction complete: %d intents, %d VLA actions", len(intent_ids), len(vla_ids))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Extract intents + VLA actions from GR00T")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--intent_dir", type=str, required=True)
    parser.add_argument("--vla_dir", type=str, required=True)
    parser.add_argument("--gpu_ids", type=str, default="0")
    parser.add_argument("--decode_workers", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--inference_batch_size", type=int, default=32)
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
            intent_dir=args.intent_dir,
            vla_dir=args.vla_dir,
            parquet_files=episode_list,
            image_size=args.image_size,
            decode_workers=args.decode_workers,
            inference_batch_size=args.inference_batch_size,
        )
    else:
        if len(gpu_ids) == 1:
            mp.set_start_method("spawn", force=True)
        extract_all(
            dataset_path=args.dataset_path,
            model_path=args.model_path,
            intent_dir=args.intent_dir,
            vla_dir=args.vla_dir,
            gpu_ids=gpu_ids,
            decode_workers=args.decode_workers,
            image_size=args.image_size,
            max_episodes=args.max_episodes,
            inference_batch_size=args.inference_batch_size,
        )
