"""Loads pre-extracted V-JEPA 2 features for PhysREPA training."""

from functools import lru_cache
from pathlib import Path

from safetensors.torch import load_file
import torch


class PhysREPAFeatureLoader:
    """Loads pre-extracted V-JEPA 2 features given (episode_idx, timestep)."""

    def __init__(self, features_dir, vjepa_layer=10, window_size=16, window_stride=4):
        self.features_dir = Path(features_dir)
        self.vjepa_layer = vjepa_layer
        self.window_size = window_size
        self.window_stride = window_stride

    @lru_cache(maxsize=512)
    def _load_episode(self, episode_idx):
        path = self.features_dir / f"{episode_idx:06d}.safetensors"
        if not path.exists():
            return None
        return load_file(path)

    def get_feature(self, episode_idx, timestep):
        """Get V-JEPA feature for a given episode and timestep."""
        data = self._load_episode(episode_idx)
        if data is None:
            return None

        window_idx = timestep // self.window_stride
        window_starts = data.get("window_starts")
        if window_starts is not None:
            max_window = int(window_starts.shape[0]) - 1
        else:
            max_window = 0
            for key in data:
                if key.startswith(f"layer_{self.vjepa_layer}_window_"):
                    w = int(key.split("_")[-1])
                    max_window = max(max_window, w)

        window_idx = min(window_idx, max_window)
        key = f"layer_{self.vjepa_layer}_window_{window_idx}"
        if key not in data:
            return None
        return data[key]

    def get_batch_features(self, metadata_list, device, dtype):
        """Get V-JEPA features for a batch.

        Args:
            metadata_list: list of dicts with 'episode_idx' and 'step_index' keys
            device: torch device
            dtype: torch dtype

        Returns:
            [B, vjepa_dim] tensor or None if any features are missing
        """
        features = []
        for meta in metadata_list:
            ep_idx = meta.get("episode_idx")
            step_idx = meta.get("step_index")
            if ep_idx is None or step_idx is None:
                return None
            feat = self.get_feature(ep_idx, step_idx)
            if feat is None:
                return None
            features.append(feat)
        return torch.stack(features).to(device=device, dtype=dtype)
