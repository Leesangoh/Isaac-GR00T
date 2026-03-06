"""Video decoding utility for BridgeData episodes.

Provides _decode_episode() for multiprocessing-based video frame extraction.
Used by dataset_phase1.py and dataset_phase2.py to load raw frames into RAM.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _decode_episode(args: tuple) -> dict | None:
    """Decode one episode's video + parquet in a worker process.

    Returns dict with numpy arrays (no torch tensors to avoid IPC issues).
    """
    parquet_path, video_dir, image_size = args
    parquet_path = Path(parquet_path)

    try:
        import av

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

        # Video frames
        video_path = (
            Path(video_dir) / chunk_name / "observation.images.image_0" / f"{episode_id}.mp4"
        )
        if not video_path.exists():
            return None

        num_frames = len(df)
        container = av.open(str(video_path))
        frames = []
        for frame in container.decode(video=0):
            sz = image_size if image_size > 98 else 98
            f = frame.reformat(width=sz, height=sz).to_ndarray(format="rgb24")
            frames.append(f)
            if len(frames) >= num_frames:
                break
        container.close()

        if len(frames) == 0:
            return None

        while len(frames) < num_frames:
            frames.append(frames[-1])

        frames_np = np.stack(frames)  # (T, H, W, 3) uint8

        return {
            "episode_id": episode_id,
            "frames": frames_np,
            "actions": actions,
            "proprios": proprios,
        }
    except Exception:
        return None
