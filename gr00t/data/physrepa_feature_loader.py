"""Loads pre-extracted V-JEPA 2 features for PhysREPA training."""

from functools import lru_cache
import logging
from pathlib import Path

from safetensors.torch import load_file
import torch
import torch.nn.functional as F


class PhysREPAFeatureLoader:
    """Loads pre-extracted V-JEPA 2 features given (episode_idx, timestep)."""

    def __init__(
        self, features_dir, vjepa_layer=10, window_size=16, window_stride=4, global_means_path=None
    ):
        self.features_dir = Path(features_dir)
        self.vjepa_layer = vjepa_layer
        self.window_size = window_size
        self.window_stride = window_stride

        # Load global means for centering if provided
        self.global_mean = None
        if global_means_path is not None:
            global_means_path = Path(global_means_path)
            if global_means_path.exists():
                global_means = torch.load(global_means_path, map_location="cpu", weights_only=True)
                if self.vjepa_layer in global_means:
                    self.global_mean = global_means[self.vjepa_layer].float()
                    logging.info(
                        f"PhysREPA: loaded global mean for layer {self.vjepa_layer} "
                        f"(norm={self.global_mean.norm():.4f}) from {global_means_path}"
                    )
                else:
                    logging.warning(
                        f"PhysREPA: global_means.pt does not contain layer {self.vjepa_layer}. "
                        f"Available: {list(global_means.keys())}. Centering DISABLED."
                    )
            else:
                logging.warning(
                    f"PhysREPA: global_means_path {global_means_path} not found. Centering DISABLED."
                )
        self._sanity_checked = False

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

    def _center(self, feat):
        """Subtract global mean from feature vector."""
        if self.global_mean is not None:
            return feat.float() - self.global_mean
        return feat

    def get_batch_features(self, metadata_list, device, dtype):
        """Get V-JEPA features for a batch, with optional global mean centering.

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
            features.append(self._center(feat))
        batch = torch.stack(features).to(device=device, dtype=dtype)

        # One-time sanity check: verify centered features have ~0 mean cosine similarity
        if not self._sanity_checked and self.global_mean is not None and len(features) >= 4:
            self._sanity_checked = True
            with torch.no_grad():
                normed = F.normalize(batch.float(), dim=-1)
                # Pairwise cosine sim for first min(B, 16) samples
                n = min(batch.shape[0], 16)
                cos_matrix = normed[:n] @ normed[:n].T
                # Exclude diagonal
                mask = ~torch.eye(n, dtype=torch.bool, device=cos_matrix.device)
                off_diag = cos_matrix[mask]
                logging.info(
                    f"PhysREPA sanity check (centered features, layer {self.vjepa_layer}): "
                    f"pairwise cosine sim mean={off_diag.mean():.4f}, std={off_diag.std():.4f} "
                    f"(should be ~0 if centering is effective)"
                )

        return batch
