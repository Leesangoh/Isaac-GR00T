"""Prediction-error-based action correction network.

Takes the intent-aware prediction error (which already encodes how much
the current trajectory deviates from the goal) and outputs a bounded
action correction.

Does NOT receive intent as input — the prediction error already encodes
goal-relative deviation because the forward model was intent-conditioned.

Receives chunk_step (0-7, normalized to [0,1]) so it can learn to apply
stronger corrections for later chunk steps where VLA predictions drift more.
"""

import torch
import torch.nn.functional as F
from torch import nn


class AttentionWeightedPooling(nn.Module):
    """Attention-weighted pooling of per-patch prediction errors.

    Learns which patches carry the most informative prediction errors
    (e.g., patches near the robot arm or manipulated object).

    Input: (B, N_patches, feature_dim) — per-patch prediction error
    Output: (B, feature_dim) — weighted sum

    Parameters: ~50K.
    """

    def __init__(self, feature_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, patch_errors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_errors: (B, N_patches, feature_dim)
        Returns:
            pooled: (B, feature_dim)
        """
        scores = self.score_mlp(patch_errors)  # (B, N_patches, 1)
        weights = F.softmax(scores, dim=1)  # (B, N_patches, 1)
        return (weights * patch_errors).sum(dim=1)  # (B, feature_dim)


class CorrectionNetwork(nn.Module):
    """Maps prediction error + planned action + proprio + chunk_step to action correction.

    Input: concat(prediction_error [384], action_vla [7], proprio [8], chunk_step [1]) = 400
    Output: Δa [7], bounded by tanh × max_correction
    """

    def __init__(
        self,
        feature_dim: int = 384,
        action_dim: int = 7,
        proprio_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 3,
        max_correction: float = 0.1,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.max_correction = max_correction
        self.chunk_size = chunk_size
        input_dim = feature_dim + action_dim + proprio_dim + 1  # +1 for chunk_step

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        prediction_error: torch.Tensor,
        action_vla: torch.Tensor,
        proprio: torch.Tensor,
        chunk_step: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            prediction_error: (B, 384) intent-aware visual prediction error
            action_vla: (B, 7) VLA's planned action
            proprio: (B, 8) current proprioception
            chunk_step: (B, 1) normalized chunk step [0, 1]. If None, defaults to 0.
        Returns:
            delta_a: (B, 7) bounded action correction
        """
        if chunk_step is None:
            chunk_step = torch.zeros(prediction_error.shape[0], 1, device=prediction_error.device)
        else:
            # Normalize to [0, 1]
            chunk_step = chunk_step / max(self.chunk_size - 1, 1)
        x = torch.cat([prediction_error, action_vla, proprio, chunk_step], dim=-1)
        raw = self.net(x)
        return torch.tanh(raw) * self.max_correction
