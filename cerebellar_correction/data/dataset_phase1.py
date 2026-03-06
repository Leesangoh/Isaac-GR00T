"""Phase 1 dataset: Raw image pairs + intent token sequences for joint encoder + forward model.

Loads video frames into RAM at startup using multiprocessing, joins with
pre-extracted intent token sequences by episode_id. Returns consecutive pairs:
(frame_t, frame_t1, proprio_t, proprio_t1, intent_tokens, attention_mask).

Intent tokens are stored as float16 in RAM (~820GB for full BridgeData).
Float32 conversion happens per-batch in the training loop.

Frames are stored as uint8 HWC numpy arrays (~46GB for full BridgeData).
Float conversion + CHW transpose happens per-sample in __getitem__.
"""

import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cerebellar_correction.data.feature_extractor import _decode_episode


log = logging.getLogger(__name__)

MAX_INTENT_TOKENS = 128
INTENT_DIM = 2048


class BridgePhase1Dataset(Dataset):
    """Dataset of consecutive raw-frame pairs + intent tokens for Phase 1 training.

    Decodes all episodes in parallel at init, stores frames as uint8 HWC
    numpy arrays in RAM. Intent tokens stored as float16.

    Returns (frame_t, frame_t1, proprio_t, proprio_t1, intent_tokens, attention_mask).
    """

    def __init__(
        self,
        dataset_path: str,
        intent_dir: str,
        proprio_dim: int = 8,
        image_size: int = 98,
        decode_workers: int = 32,
    ):
        dataset_path = Path(dataset_path)
        intent_dir = Path(intent_dir)

        # Load intent episode IDs for joining
        intent_ids = {p.stem for p in intent_dir.glob("episode_*.npz")}
        log.info("Found %d intent episodes", len(intent_ids))

        # Find all episode parquet files
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

        # Decode all episodes in parallel
        worker_args = [(str(p), video_dir, image_size) for p in parquet_files]
        log.info("Decoding %d episodes with %d workers...", len(parquet_files), decode_workers)

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=decode_workers)

        # Pass 1: decode episodes, join with intents, count pairs
        episodes = []
        total_pairs = 0
        skipped = 0
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

            frames = result["frames"]  # (T, H, W, 3) uint8
            ep_len = len(frames)
            if ep_len < 2:
                skipped += 1
                continue

            # Load intent tokens for this episode
            intent_data = np.load(intent_dir / f"{ep_id}.npz")
            intent_tokens = torch.from_numpy(intent_data["intent_tokens"])  # (T, 128, 2048) fp16
            attn_masks = torch.from_numpy(intent_data["attention_masks"])  # (T, 128) bool

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
        intent_tokens_buf = np.empty((total_pairs, MAX_INTENT_TOKENS, INTENT_DIM), dtype=np.float16)
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

            # Free early
            result["frames"] = None
            result["proprios"] = None
            result["intent_tokens"] = None
            result["attention_masks"] = None

        del episodes

        self.frames_t = frames_t_hwc
        self.frames_t1 = frames_t1_hwc
        self.proprio_t = torch.from_numpy(proprio_buf)
        self.proprio_t1 = torch.from_numpy(proprio1_buf)
        self.intent_tokens = intent_tokens_buf  # keep as numpy fp16 to save RAM
        self.attention_masks = attn_mask_buf

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
            "intent_tokens": torch.from_numpy(self.intent_tokens[idx].copy()),  # (128, 2048) fp16
            "attention_mask": torch.from_numpy(self.attention_masks[idx].copy()),  # (128,) bool
        }
