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
        self,
        features_dir,
        vjepa_layer=10,
        window_size=16,
        window_stride=4,
        global_means_path=None,
        action_horizon=16,
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
        self.action_horizon = action_horizon
        self.timestepwise = False
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

        # Select the latest window that still covers the timestep (past~current physics)
        # ceil((timestep - window_size + 1) / stride), clamped to [0, max_window]
        window_idx = max(0, -(-(timestep - self.window_size + 1) // self.window_stride))
        max_window = self._get_max_window(data)
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

    def _get_max_window(self, data):
        """Get the maximum window index for an episode's data."""
        window_starts = data.get("window_starts")
        if window_starts is not None:
            return int(window_starts.shape[0]) - 1
        max_window = 0
        for key in data:
            if key.startswith(f"layer_{self.vjepa_layer}_window_"):
                w = int(key.split("_")[-1])
                max_window = max(max_window, w)
        return max_window

    def get_feature_sequence(self, episode_idx, step_index):
        """Get V-JEPA features for each action timestep in the horizon.

        For action token t+i (i=0..action_horizon-1), looks up the V-JEPA window
        covering frame (step_index + 1 + i).

        Args:
            episode_idx: episode index
            step_index: current timestep

        Returns:
            [action_horizon, vjepa_dim] tensor or None if features are missing
        """
        data = self._load_episode(episode_idx)
        if data is None:
            return None

        max_window = self._get_max_window(data)

        # Cache centered features per window_idx to avoid redundant centering
        centered_cache = {}
        sequence = []
        for i in range(self.action_horizon):
            future_frame = step_index + 1 + i
            # Select the latest window that still covers future_frame (past~current physics).
            # Window W covers frames [W*stride, W*stride + window_size - 1].
            # We want the largest W such that W*stride + window_size - 1 >= future_frame,
            # i.e. W <= (future_frame) // stride  (frame is within or before window end).
            # But also W*stride <= future_frame (window starts at or before the frame).
            # The latest valid W is: future_frame // stride, but capped so the window
            # still covers future_frame: W*stride + window_size - 1 >= future_frame.
            # Simplest: W = (future_frame - window_size + 1 + stride - 1) // stride
            #            = ceil((future_frame - window_size + 1) / stride)
            window_idx = max(0, -(-(future_frame - self.window_size + 1) // self.window_stride))
            window_idx = min(window_idx, max_window)

            if window_idx not in centered_cache:
                key = f"layer_{self.vjepa_layer}_window_{window_idx}"
                if key not in data:
                    return None
                centered_cache[window_idx] = self._center(data[key])

            sequence.append(centered_cache[window_idx])

        return torch.stack(sequence)  # [action_horizon, vjepa_dim]

    def get_batch_features(self, metadata_list, device, dtype):
        """Get V-JEPA features for a batch, with optional global mean centering.

        When self.timestepwise is True, returns per-action-token features [B, H, vjepa_dim].
        When False, returns a single feature per sample [B, vjepa_dim] (original behavior).

        Args:
            metadata_list: list of dicts with 'episode_idx' and 'step_index' keys
            device: torch device
            dtype: torch dtype

        Returns:
            [B, H, vjepa_dim] or [B, vjepa_dim] tensor, or None if any features are missing
        """
        if self.timestepwise:
            return self._get_batch_features_timestepwise(metadata_list, device, dtype)
        return self._get_batch_features_meanpool(metadata_list, device, dtype)

    def _get_batch_features_meanpool(self, metadata_list, device, dtype):
        """Original mean-pool mode: one V-JEPA feature per sample → [B, vjepa_dim]."""
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
            self._log_centering_check(batch)

        return batch

    def _get_batch_features_timestepwise(self, metadata_list, device, dtype):
        """Timestep-wise mode: per-action-token features → [B, H, vjepa_dim]."""
        features = []
        for meta in metadata_list:
            ep_idx = meta.get("episode_idx")
            step_idx = meta.get("step_index")
            if ep_idx is None or step_idx is None:
                return None
            feat_seq = self.get_feature_sequence(ep_idx, step_idx)
            if feat_seq is None:
                return None
            features.append(feat_seq)
        batch = torch.stack(features).to(device=device, dtype=dtype)  # [B, H, vjepa_dim]

        # One-time sanity check
        if not self._sanity_checked and self.global_mean is not None and len(features) >= 4:
            self._sanity_checked = True
            self._log_centering_check(batch[:, 0, :])
            # Verify targets differ across timesteps within a sample
            with torch.no_grad():
                sample_targets = batch[0]  # [H, vjepa_dim]
                ts_normed = F.normalize(sample_targets.float(), dim=-1)
                ts_cos = ts_normed @ ts_normed.T  # [H, H]
                ts_mask = ~torch.eye(self.action_horizon, dtype=torch.bool, device=ts_cos.device)
                ts_off_diag = ts_cos[ts_mask]
                n_unique = (ts_off_diag < 0.9999).sum().item()
                logging.info(
                    f"PhysREPA temporal check: across {self.action_horizon} timesteps, "
                    f"{n_unique}/{ts_off_diag.numel()} pairs differ "
                    f"(inter-timestep cosine sim mean={ts_off_diag.mean():.4f})"
                )

        return batch

    def _log_centering_check(self, features_2d):
        """Log pairwise cosine similarity sanity check for centered features."""
        with torch.no_grad():
            normed = F.normalize(features_2d.float(), dim=-1)
            n = min(features_2d.shape[0], 16)
            cos_matrix = normed[:n] @ normed[:n].T
            mask = ~torch.eye(n, dtype=torch.bool, device=cos_matrix.device)
            off_diag = cos_matrix[mask]
            logging.info(
                f"PhysREPA sanity check (centered features, layer {self.vjepa_layer}): "
                f"pairwise cosine sim mean={off_diag.mean():.4f}, std={off_diag.std():.4f} "
                f"(should be ~0 if centering is effective)"
            )
