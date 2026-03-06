"""Phase 2 dataset: Raw image pairs + intent tokens + VLA actions for correction training.

Loads video frames into RAM, joins with pre-extracted intent token sequences
and VLA actions. Returns:
(frame_t, frame_t1, action_expert, action_vla, proprio_t, intent_tokens, attention_mask, chunk_step).

Uses ALL 8 chunk steps per timestep. For chunk step k at timestep t:
  - vla_chunks[t, k] is VLA's prediction for timestep t+k
  - action_expert[t+k] is ground truth at timestep t+k
  - frame_t = frames[t+k], frame_t1 = frames[t+k+1] (visual state at execution time)
  - proprio_t = proprios[t+k] (proprio at execution time)
  - intent_tokens = tokens[t] (from chunk generation time, stays fixed within chunk)

Phase 2 uses the frozen Phase 1 encoder to compute features on-the-fly.
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

CHUNK_SIZE = 8
MAX_INTENT_TOKENS = 128
INTENT_DIM = 2048


class BridgePhase2Dataset(Dataset):
    """Dataset for Phase 2: correction training with VLA errors + intent tokens.

    Decodes all episodes in parallel at init, joins with intent tokens and VLA actions.
    Uses all 8 chunk steps per timestep for ~8x more training data.
    """

    def __init__(
        self,
        dataset_path: str,
        intent_dir: str,
        vla_action_dir: str,
        action_dim: int = 7,
        proprio_dim: int = 8,
        image_size: int = 98,
        chunk_size: int = CHUNK_SIZE,
        decode_workers: int = 32,
    ):
        dataset_path = Path(dataset_path)
        intent_dir = Path(intent_dir)
        vla_action_dir = Path(vla_action_dir)

        # Load available episode IDs
        intent_ids = {p.stem for p in intent_dir.glob("episode_*.npz")}
        vla_ids = {p.stem for p in vla_action_dir.glob("episode_*.pt")}
        common_ids = intent_ids & vla_ids
        log.info("Intent: %d, VLA: %d, Common: %d", len(intent_ids), len(vla_ids), len(common_ids))

        # Find parquet files
        data_dir = dataset_path / "data"
        if not data_dir.exists():
            data_dir = dataset_path
        parquet_files = sorted(data_dir.glob("**/*.parquet"))
        log.info("Found %d episode parquet files", len(parquet_files))

        if len(parquet_files) == 0:
            raise FileNotFoundError(f"No parquet files found in {data_dir}.")

        video_dir = str(dataset_path / "videos")

        # Decode episodes in parallel
        worker_args = [(str(p), video_dir, image_size) for p in parquet_files]
        log.info("Decoding %d episodes with %d workers...", len(parquet_files), decode_workers)

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=decode_workers)

        # Pass 1: decode, join, expand chunks, count pairs
        pair_records = []
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
            if ep_id not in common_ids:
                skipped += 1
                continue

            frames = result["frames"]  # (T, H, W, 3) uint8
            ep_len = len(frames)
            if ep_len < 2:
                skipped += 1
                continue

            intent_data = np.load(intent_dir / f"{ep_id}.npz")
            vla_data = torch.load(
                vla_action_dir / f"{ep_id}.pt", map_location="cpu", weights_only=False
            )

            intent_tokens = torch.from_numpy(intent_data["intent_tokens"])  # (T, 128, 2048) fp16
            attn_masks = torch.from_numpy(intent_data["attention_masks"])  # (T, 128) bool
            vla_chunks = vla_data["action_vla_chunks"][:, :, :action_dim].float()  # (T, 8, 7)
            proprios = result["proprios"][:, :proprio_dim].astype(np.float32)
            actions_expert = result["actions"][:, :action_dim].astype(np.float32)

            min_len = min(ep_len, len(intent_tokens), len(vla_chunks))
            if min_len < 2:
                skipped += 1
                continue

            # Count pairs from all chunk steps
            n_pairs_ep = 0
            for k in range(chunk_size):
                max_t = min_len - k - 1
                if max_t <= 0:
                    break
                n_pairs_ep += max_t

            total_pairs += n_pairs_ep
            pair_records.append(
                {
                    "frames": frames[:min_len],
                    "proprios": proprios[:min_len],
                    "actions_expert": actions_expert[:min_len],
                    "intent_tokens": intent_tokens[:min_len],
                    "attention_masks": attn_masks[:min_len],
                    "vla_chunks": vla_chunks[:min_len],
                    "ep_len": min_len,
                }
            )

        pool.close()
        pool.join()
        log.info(
            "Decoded %d episodes (%d pairs), skipped %d",
            len(pair_records),
            total_pairs,
            skipped,
        )

        if len(pair_records) == 0:
            raise RuntimeError("No valid episodes found.")

        # Pass 2: allocate and fill with chunk expansion
        h, w = pair_records[0]["frames"].shape[1], pair_records[0]["frames"].shape[2]
        log.info("Allocating %.1f GB for frames...", total_pairs * 2 * 3 * h * w / 1e9)
        log.info(
            "Allocating %.1f GB for intent tokens (fp16)...",
            total_pairs * MAX_INTENT_TOKENS * INTENT_DIM * 2 / 1e9,
        )

        frames_t_hwc = np.empty((total_pairs, h, w, 3), dtype=np.uint8)
        frames_t1_hwc = np.empty((total_pairs, h, w, 3), dtype=np.uint8)
        action_expert_buf = np.empty((total_pairs, action_dim), dtype=np.float32)
        action_vla_buf = np.empty((total_pairs, action_dim), dtype=np.float32)
        proprio_buf = np.empty((total_pairs, proprio_dim), dtype=np.float32)
        intent_tokens_buf = np.empty((total_pairs, MAX_INTENT_TOKENS, INTENT_DIM), dtype=np.float16)
        attn_mask_buf = np.empty((total_pairs, MAX_INTENT_TOKENS), dtype=np.bool_)
        chunk_step_buf = np.empty((total_pairs, 1), dtype=np.float32)

        offset = 0
        for rec in tqdm(pair_records, desc="Building chunk-expanded pairs"):
            frames = rec["frames"]
            proprios = rec["proprios"]
            actions_expert = rec["actions_expert"]
            tokens_np = rec["intent_tokens"].numpy()
            masks_np = rec["attention_masks"].numpy()
            vla_chunks_np = rec["vla_chunks"].numpy()
            ep_len = rec["ep_len"]

            for k in range(chunk_size):
                max_t = ep_len - k - 1
                if max_t <= 0:
                    break

                n = max_t
                exec_idx = np.arange(k, k + n)  # execution times
                gen_idx = np.arange(n)  # chunk generation times

                frames_t_hwc[offset : offset + n] = frames[exec_idx]
                frames_t1_hwc[offset : offset + n] = frames[exec_idx + 1]
                action_expert_buf[offset : offset + n] = actions_expert[exec_idx]
                action_vla_buf[offset : offset + n] = vla_chunks_np[gen_idx, k]
                proprio_buf[offset : offset + n] = proprios[exec_idx]
                intent_tokens_buf[offset : offset + n] = tokens_np[gen_idx]
                attn_mask_buf[offset : offset + n] = masks_np[gen_idx]
                chunk_step_buf[offset : offset + n, 0] = k

                offset += n

            # Free early
            rec["frames"] = None
            rec["proprios"] = None
            rec["intent_tokens"] = None
            rec["attention_masks"] = None
            rec["vla_chunks"] = None

        del pair_records

        self.frames_t = frames_t_hwc
        self.frames_t1 = frames_t1_hwc
        self.action_expert = torch.from_numpy(action_expert_buf)
        self.action_vla = torch.from_numpy(action_vla_buf)
        self.proprio_t = torch.from_numpy(proprio_buf)
        self.intent_tokens = intent_tokens_buf  # numpy fp16
        self.attention_masks = attn_mask_buf
        self.chunk_step = torch.from_numpy(chunk_step_buf)

        step0_count = (self.chunk_step == 0).sum().item()
        log.info(
            "Loaded %d pairs (step0: %d, all steps: ~%.1fx)",
            total_pairs,
            step0_count,
            total_pairs / max(step0_count, 1),
        )
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
            "action_expert": self.action_expert[idx],
            "action_vla": self.action_vla[idx],
            "proprio_t": self.proprio_t[idx],
            "intent_tokens": torch.from_numpy(self.intent_tokens[idx].copy()),  # (128, 2048) fp16
            "attention_mask": torch.from_numpy(self.attention_masks[idx].copy()),  # (128,) bool
            "chunk_step": self.chunk_step[idx],
        }
