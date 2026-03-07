"""Integrated Patch-Level ICAC Module (v2).

Combines all sub-modules for inference:
    1. IntentExtractor — hooks GR00T backbone (no params)
    2. CerebellumVisualEncoder — Frozen DINOv2 → 49 patch tokens
    3. TransitionViT — (patch_tokens, proprio, intent) → predicted z_{t+1} patches
    4. ProprioForwardModel — MLP: (proprio, intent_pooled) → Δproprio
    5. AttentionWeightedPooling — per-patch prediction error → 384-dim
    6. CorrectionNetwork — (pooled_error, action, proprio, step) → Δa

Inference flow (per control step):
    1. GR00T forward → action_chunk + intent tokens (per chunk, 1 forward pass)
    2. Frozen DINOv2 → Z_t (49×384 patch tokens)
    3. TransitionViT prediction using cached intent tokens
    4. Compute patch-level prediction error from previous step (1-step delay)
    5. Attention-weighted pooling → 384-dim error signal
    6. Correction network → Δa
    7. Execute action_vla + Δa
"""

from dataclasses import dataclass

import torch
from torch import nn

from cerebellar_correction.models.correction_net import AttentionWeightedPooling, CorrectionNetwork
from cerebellar_correction.models.forward_model import ProprioForwardModel, TransitionViT
from cerebellar_correction.models.intent_extractor import IntentExtractor
from cerebellar_correction.models.visual_encoder import CerebellumVisualEncoder


@dataclass
class CerebellumConfig:
    # Visual encoder (frozen)
    visual_backbone: str = "dinov2_vits14"
    image_size: int = 98
    num_patches: int = 49  # 7×7 for 98/14

    # Dimensions
    feature_dim: int = 384
    intent_dim: int = 2048
    action_dim: int = 7
    proprio_dim: int = 8

    # Transition ViT
    transition_num_layers: int = 4
    transition_num_heads: int = 8
    transition_ffn_dim: int = 1536
    transition_dropout: float = 0.1
    max_intent_tokens: int = 128

    # Attention pooling
    pooling_hidden_dim: int = 128

    # Correction network
    correction_hidden_dim: int = 256
    correction_num_layers: int = 3
    max_correction: float = 0.15


class PatchCerebellumModule(nn.Module):
    """Patch-level cerebellar correction module for inference (v2)."""

    def __init__(self, config: CerebellumConfig):
        super().__init__()
        self.config = config

        # Frozen DINOv2 encoder (patch-level output)
        self.visual_encoder = CerebellumVisualEncoder(
            backbone=config.visual_backbone,
            use_lora=False,
            input_size=config.image_size,
        )
        # Freeze encoder
        self.visual_encoder.eval()
        for p in self.visual_encoder.parameters():
            p.requires_grad = False

        # Transition ViT: predicts z_{t+1} patches
        self.transition_vit = TransitionViT(
            feature_dim=config.feature_dim,
            proprio_dim=config.proprio_dim,
            intent_dim=config.intent_dim,
            num_patches=config.num_patches,
            num_layers=config.transition_num_layers,
            num_heads=config.transition_num_heads,
            ffn_dim=config.transition_ffn_dim,
            dropout=config.transition_dropout,
            max_intent_tokens=config.max_intent_tokens,
        )

        # Proprio forward model
        self.proprio_forward = ProprioForwardModel(
            proprio_dim=config.proprio_dim,
            intent_dim=config.intent_dim,
        )

        # Attention-weighted pooling for prediction error
        self.error_pooling = AttentionWeightedPooling(
            feature_dim=config.feature_dim,
            hidden_dim=config.pooling_hidden_dim,
        )

        # Correction network
        self.correction_net = CorrectionNetwork(
            feature_dim=config.feature_dim,
            action_dim=config.action_dim,
            proprio_dim=config.proprio_dim,
            hidden_dim=config.correction_hidden_dim,
            num_layers=config.correction_num_layers,
            max_correction=config.max_correction,
        )

        # Intent extractor (set via set_groot_model)
        self.intent_extractor: IntentExtractor | None = None

        # Runtime state
        self._prev_z_patches: torch.Tensor | None = None  # (1, 49, 384)
        self._prev_z_predicted: torch.Tensor | None = None  # (1, 49, 384) predicted from prev step
        self._prev_proprio: torch.Tensor | None = None
        self._current_intent_tokens: torch.Tensor | None = None
        self._current_intent_mask: torch.Tensor | None = None

    def set_groot_model(self, groot_model: nn.Module):
        """Attach intent extractor hook to GR00T backbone."""
        self.intent_extractor = IntentExtractor(groot_model)

    def on_new_chunk(
        self,
        intent_tokens: torch.Tensor,
        intent_attention_mask: torch.Tensor,
    ):
        """Called at the start of each new action chunk."""
        self._current_intent_tokens = intent_tokens.detach()
        self._current_intent_mask = intent_attention_mask.detach()

    def reset(self):
        """Full reset at episode start."""
        self._prev_z_patches = None
        self._prev_z_predicted = None
        self._prev_proprio = None
        self._current_intent_tokens = None
        self._current_intent_mask = None

    @torch.no_grad()
    def correct(
        self,
        image_current: torch.Tensor,
        proprio_current: torch.Tensor,
        action_planned: torch.Tensor,
        chunk_step: int = 0,
    ) -> torch.Tensor:
        """Compute corrected action for a single control step.

        Args:
            image_current: (1, 3, H, W) float32 in [0, 1]
            proprio_current: (1, 8)
            action_planned: (1, 7) VLA's planned action
            chunk_step: which step within the action chunk (0-7)

        Returns:
            action_corrected: (1, 7)
        """
        # Get current patch tokens
        z_patches = self.visual_encoder.forward_patches(image_current)  # (1, 49, 384)

        # Compute prediction error from previous step's prediction
        if self._prev_z_predicted is not None:
            # E = Z_t_actual - Ẑ_t_predicted (per-patch error)
            patch_error = z_patches - self._prev_z_predicted  # (1, 49, 384)
            prediction_error = self.error_pooling(patch_error)  # (1, 384)
        else:
            prediction_error = torch.zeros(
                1, self.config.feature_dim, device=image_current.device
            )

        # Compute correction
        chunk_step_t = torch.tensor(
            [[chunk_step]], dtype=torch.float32, device=image_current.device
        )
        delta_a = self.correction_net(
            prediction_error=prediction_error,
            action_vla=action_planned,
            proprio=proprio_current,
            chunk_step=chunk_step_t,
        )

        # Predict z_{t+1} for next step's error computation
        if self._current_intent_tokens is not None:
            self._prev_z_predicted = self.transition_vit(
                z_patches,
                proprio_current,
                self._current_intent_tokens,
                self._current_intent_mask,
            )
        else:
            self._prev_z_predicted = None

        self._prev_z_patches = z_patches
        self._prev_proprio = proprio_current

        return action_planned + delta_a

    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu"):
        """Load trained weights from assembled checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            if "transition_vit" in ckpt:
                self.transition_vit.load_state_dict(ckpt["transition_vit"])
            if "proprio_forward" in ckpt:
                self.proprio_forward.load_state_dict(ckpt["proprio_forward"])
            if "error_pooling" in ckpt:
                self.error_pooling.load_state_dict(ckpt["error_pooling"])
            if "correction_net" in ckpt:
                self.correction_net.load_state_dict(ckpt["correction_net"])
