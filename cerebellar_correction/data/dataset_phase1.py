"""Phase 1 dataset: Raw image pairs + intent token sequences for joint encoder + forward model.

Loads video frames into RAM at startup using multiprocessing, joins with
pre-extracted intent token sequences by episode_id. Returns consecutive pairs:
(frame_t, frame_t1, proprio_t, proprio_t1, intent_tokens, attention_mask).

Supports disk caching: first run decodes videos and saves paired arrays to cache_dir.
Subsequent runs skip video decoding entirely and load from cache (~2 min vs ~3 hours).

Intent tokens are stored as float16 in RAM (~987GB for full BridgeData).
Float32 conversion happens per-batch in the training loop.

Frames are stored as uint8 HWC numpy arrays (~108GB for full BridgeData).
Float conversion + CHW transpose happens per-sample in __getitem__.
"""

import json
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cerebellar_correction.data.feature_extractor import _decode_episode


log = logging.getLogger(__name__)

MAX_INTENT_TOKENS = 128
INTENT_DIM = 2048
CACHE_VERSION = 1


class BridgePhase1Dataset(Dataset):
    """Dataset of consecutive raw-frame pairs + intent tokens for Phase 1 training.

    Decodes all episodes in parallel at init, stores frames as uint8 HWC
    numpy arrays in RAM. Intent tokens stored as float16.

    With cache_dir set, saves decoded frames/proprios/masks on first run.
    Subsequent runs load from cache (skipping video decoding) and only
    load intent tokens from .npz files using parallel threads.

    Returns (frame_t, frame_t1, proprio_t, proprio_t1, intent_tokens, attention_mask).
    """

    def __init__(
        self,
        dataset_path: str,
        intent_dir: str,
        proprio_dim: int = 8,
        image_size: int = 98,
        decode_workers: int = 32,
        cache_dir: str = "",
    ):
        intent_dir = Path(intent_dir)
        cache_dir = Path(cache_dir) if cache_dir else None

        if cache_dir and self._load_from_cache(cache_dir, intent_dir, proprio_dim, decode_workers):
            return

        # Full decode path (first run or no cache)
        self._decode_and_build(
            dataset_path, intent_dir, proprio_dim, image_size, decode_workers
        )

        # Save cache for next time
        if cache_dir:
            self._save_to_cache(cache_dir)

    def _load_from_cache(
        self,
        cache_dir: Path,
        intent_dir: Path,
        proprio_dim: int,
        load_workers: int,
    ) -> bool:
        """Try to load from disk cache. Returns True on success."""
        meta_path = cache_dir / "cache_meta.json"
        if not meta_path.exists():
            log.info("No cache found at %s, will decode from scratch", cache_dir)
            return False

        with open(meta_path) as f:
            meta = json.load(f)

        if meta.get("version") != CACHE_VERSION:
            log.info("Cache version mismatch, will re-decode")
            return False

        total_pairs = meta["total_pairs"]
        h, w = meta["h"], meta["w"]
        ep_order = meta["episode_order"]  # list of (ep_id, n_pairs)

        log.info("Loading from cache: %d pairs, %d episodes", total_pairs, len(ep_order))

        # Load frames/proprios/masks from cache (mmap for speed)
        log.info("Loading cached frames (~%.1f GB)...", total_pairs * 2 * h * w * 3 / 1e9)
        frames_t_hwc = np.load(cache_dir / "frames_t.npy", mmap_mode="r")
        frames_t1_hwc = np.load(cache_dir / "frames_t1.npy", mmap_mode="r")
        proprio_buf = np.load(cache_dir / "proprio_t.npy")
        proprio1_buf = np.load(cache_dir / "proprio_t1.npy")
        attn_mask_buf = np.load(cache_dir / "attn_masks.npy")
        log.info("Cached frames/proprios/masks loaded")

        # Load intent tokens in parallel (I/O bound → threads)
        log.info(
            "Loading intent tokens from %d episodes (~%.1f GB)...",
            len(ep_order),
            total_pairs * MAX_INTENT_TOKENS * INTENT_DIM * 2 / 1e9,
        )
        intent_tokens_buf = np.empty(
            (total_pairs, MAX_INTENT_TOKENS, INTENT_DIM), dtype=np.float16
        )

        def _load_intent(task):
            ep_id, offset, n_pairs = task
            data = np.load(intent_dir / f"{ep_id}.npz")
            tokens = data["intent_tokens"]  # (T, 128, 2048) fp16
            intent_tokens_buf[offset : offset + n_pairs] = tokens[:n_pairs]

        tasks = []
        offset = 0
        for ep_id, n_pairs in ep_order:
            tasks.append((ep_id, offset, n_pairs))
            offset += n_pairs

        n_threads = min(load_workers, 64)
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(tqdm(
                pool.map(_load_intent, tasks),
                total=len(tasks),
                desc="Loading intent tokens",
            ))

        # Copy frames from mmap to RAM for fast random access during training
        log.info("Copying frames to RAM...")
        self.frames_t = np.array(frames_t_hwc)
        self.frames_t1 = np.array(frames_t1_hwc)
        del frames_t_hwc, frames_t1_hwc

        self.proprio_t = torch.from_numpy(proprio_buf)
        self.proprio_t1 = torch.from_numpy(proprio1_buf)
        self.intent_tokens = intent_tokens_buf
        self.attention_masks = attn_mask_buf

        log.info("Cache load complete: %d pairs", total_pairs)
        return True

    def _save_to_cache(self, cache_dir: Path):
        """Save frames/proprios/masks to disk cache."""
        cache_dir.mkdir(parents=True, exist_ok=True)

        log.info("Saving cache to %s (~%.1f GB)...", cache_dir,
                 (self.frames_t.nbytes + self.frames_t1.nbytes +
                  self.proprio_t.numpy().nbytes * 2 + self.attention_masks.nbytes) / 1e9)

        np.save(cache_dir / "frames_t.npy", self.frames_t)
        np.save(cache_dir / "frames_t1.npy", self.frames_t1)
        np.save(cache_dir / "proprio_t.npy", self.proprio_t.numpy())
        np.save(cache_dir / "proprio_t1.npy", self.proprio_t1.numpy())
        np.save(cache_dir / "attn_masks.npy", self.attention_masks)

        meta = {
            "version": CACHE_VERSION,
            "total_pairs": len(self.frames_t),
            "h": self.frames_t.shape[1],
            "w": self.frames_t.shape[2],
            "episode_order": self._episode_order,
        }
        with open(cache_dir / "cache_meta.json", "w") as f:
            json.dump(meta, f)

        log.info("Cache saved")

    def _decode_and_build(
        self,
        dataset_path: str,
        intent_dir: Path,
        proprio_dim: int,
        image_size: int,
        decode_workers: int,
    ):
        """Full decode path: decode videos + load intents + build paired arrays."""
        dataset_path = Path(dataset_path)

        intent_ids = {p.stem for p in intent_dir.glob("episode_*.npz")}
        log.info("Found %d intent episodes", len(intent_ids))

        data_dir = dataset_path / "data"
        if not data_dir.exists():
            data_dir = dataset_path
        parquet_files = sorted(data_dir.glob("**/*.parquet"))
        log.info("Found %d episode parquet files", len(parquet_files))

        if len(parquet_files) == 0:
            raise FileNotFoundError(
                f"No parquet files found in {data_dir}. "
                "Make sure the dataset is downloaded in LeRobot V2 format."
            )

        video_dir = str(dataset_path / "videos")
        worker_args = [(str(p), video_dir, image_size) for p in parquet_files]
        log.info("Decoding %d episodes with %d workers...", len(parquet_files), decode_workers)

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=decode_workers)

        episodes = []
        total_pairs = 0
        skipped = 0
        episode_order = []  # for cache metadata

        for result in tqdm(
            pool.imap_unordered(_decode_episode, worker_args, chunksize=8),
            total=len(parquet_files),
            desc="Decoding episodes",
        ):
            if result is None:
                skipped += 1
                continue

            ep_id = result.get("episode_id", "")
            if ep_id not in intent_ids:
                skipped += 1
                continue

            frames = result["frames"]
            ep_len = len(frames)
            if ep_len < 2:
                skipped += 1
                continue

            intent_data = np.load(intent_dir / f"{ep_id}.npz")
            intent_tokens = torch.from_numpy(intent_data["intent_tokens"])
            attn_masks = torch.from_numpy(intent_data["attention_masks"])

            min_len = min(ep_len, len(intent_tokens))
            if min_len < 2:
                skipped += 1
                continue

            result["intent_tokens"] = intent_tokens[:min_len]
            result["attention_masks"] = attn_masks[:min_len]
            result["frames"] = frames[:min_len]
            result["proprios"] = result["proprios"][:min_len]

            n_pairs = min_len - 1
            total_pairs += n_pairs
            episode_order.append((ep_id, n_pairs))
            episodes.append(result)

        pool.close()
        pool.join()
        log.info(
            "Decoded %d episodes (%d pairs), skipped %d",
            len(episodes),
            total_pairs,
            skipped,
        )

        if len(episodes) == 0:
            raise RuntimeError("No valid episodes found after decoding + intent join.")

        # Pass 2: pre-allocate contiguous arrays and fill
        h, w = episodes[0]["frames"].shape[1], episodes[0]["frames"].shape[2]
        log.info("Allocating %.1f GB for frames...", total_pairs * 2 * 3 * h * w / 1e9)
        log.info(
            "Allocating %.1f GB for intent tokens (fp16)...",
            total_pairs * MAX_INTENT_TOKENS * INTENT_DIM * 2 / 1e9,
        )

        frames_t_hwc = np.empty((total_pairs, h, w, 3), dtype=np.uint8)
        frames_t1_hwc = np.empty((total_pairs, h, w, 3), dtype=np.uint8)
        proprio_buf = np.empty((total_pairs, proprio_dim), dtype=np.float32)
        proprio1_buf = np.empty((total_pairs, proprio_dim), dtype=np.float32)
        intent_tokens_buf = np.empty(
            (total_pairs, MAX_INTENT_TOKENS, INTENT_DIM), dtype=np.float16
        )
        attn_mask_buf = np.empty((total_pairs, MAX_INTENT_TOKENS), dtype=np.bool_)

        offset = 0
        for result in tqdm(episodes, desc="Building pairs"):
            frames = result["frames"]
            proprios = result["proprios"][:, :proprio_dim]
            ep_tokens = result["intent_tokens"].numpy()
            ep_masks = result["attention_masks"].numpy()
            n = len(frames) - 1

            frames_t_hwc[offset : offset + n] = frames[:-1]
            frames_t1_hwc[offset : offset + n] = frames[1:]
            proprio_buf[offset : offset + n] = proprios[:-1]
            proprio1_buf[offset : offset + n] = proprios[1:]
            intent_tokens_buf[offset : offset + n] = ep_tokens[:-1]
            attn_mask_buf[offset : offset + n] = ep_masks[:-1]

            offset += n

            result["frames"] = None
            result["proprios"] = None
            result["intent_tokens"] = None
            result["attention_masks"] = None

        del episodes

        self.frames_t = frames_t_hwc
        self.frames_t1 = frames_t1_hwc
        self.proprio_t = torch.from_numpy(proprio_buf)
        self.proprio_t1 = torch.from_numpy(proprio1_buf)
        self.intent_tokens = intent_tokens_buf
        self.attention_masks = attn_mask_buf
        self._episode_order = episode_order  # for cache saving

        log.info("Loaded %d pairs into RAM", total_pairs)
        log.info(
            "RAM usage: ~%.1f GB frames, ~%.1f GB intent tokens",
            self.frames_t.nbytes * 2 / 1e9,
            self.intent_tokens.nbytes / 1e9,
        )

    def __len__(self):
        return len(self.frames_t)

    def __getitem__(self, idx):
        return {
            "frame_t": torch.from_numpy(self.frames_t[idx].transpose(2, 0, 1).copy()),
            "frame_t1": torch.from_numpy(self.frames_t1[idx].transpose(2, 0, 1).copy()),
            "proprio_t": self.proprio_t[idx],
            "proprio_t1": self.proprio_t1[idx],
            "intent_tokens": torch.from_numpy(self.intent_tokens[idx].copy()),
            "attention_mask": torch.from_numpy(self.attention_masks[idx].copy()),
        }
