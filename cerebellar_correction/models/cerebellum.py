"""Integrated Intent-Conditioned Cerebellar Correction Module.

Combines all sub-modules for inference:
    1. IntentExtractor — hooks GR00T backbone (no params)
    2. CerebellumVisualEncoder — DINOv2 + LoRA → z_t
    3. IntentForwardModel — self-attention: (z_t, proprio, intent_tokens, mask) → Δz_ideal
    4. ProprioForwardModel — MLP: (proprio, intent_pooled) → Δproprio
    5. CorrectionNetwork — (pred_error, action, proprio) → Δa

Inference flow (per control step):
    1. GR00T forward → action_chunk + intent tokens (per chunk, 1 forward pass)
    2. DINOv2 → z_t
    3. Self-attention forward model prediction using cached intent tokens
    4. Compute prediction error from previous step (1-step delay)
    5. Correction network → Δa
    6. Execute action_vla + Δa
"""

from dataclasses import dataclass

import torch
from torch import nn

from cerebellar_correction.models.correction_net import CorrectionNetwork
from cerebellar_correction.models.forward_model import IntentForwardModel, ProprioForwardModel
from cerebellar_correction.models.intent_extractor import IntentExtractor
from cerebellar_correction.models.visual_encoder import CerebellumVisualEncoder


@dataclass
class CerebellumConfig:
    # Visual encoder
    visual_backbone: str = "dinov2_vits14"
    image_size: int = 98
    use_visual_lora: bool = True
    visual_lora_rank: int = 8
    ema_tau: float = 0.996

    # Dimensions (verified from GR00T checkpoint)
    feature_dim: int = 384
    intent_dim: int = 2048
    action_dim: int = 7
    proprio_dim: int = 8

    # Forward model (self-attention)
    forward_num_layers: int = 2
    forward_num_heads: int = 6
    forward_ffn_dim: int = 1536
    forward_dropout: float = 0.1
    max_intent_tokens: int = 128

    # Correction network
    correction_hidden_dim: int = 256
    correction_num_layers: int = 3
    max_correction: float = 0.1


class IntentCerebellumModule(nn.Module):
    """Full cerebellar correction module for inference."""

    def __init__(self, config: CerebellumConfig):
        super().__init__()
        self.config = config

        self.visual_encoder = CerebellumVisualEncoder(
            backbone=config.visual_backbone,
            use_lora=config.use_visual_lora,
            lora_rank=config.visual_lora_rank,
            input_size=config.image_size,
        )

        self.forward_model = IntentForwardModel(
            feature_dim=config.feature_dim,
            proprio_dim=config.proprio_dim,
            intent_dim=config.intent_dim,
            num_layers=config.forward_num_layers,
            num_heads=config.forward_num_heads,
            ffn_dim=config.forward_ffn_dim,
            dropout=config.forward_dropout,
            max_intent_tokens=config.max_intent_tokens,
        )

        self.proprio_forward = ProprioForwardModel(
            proprio_dim=config.proprio_dim,
            intent_dim=config.intent_dim,
        )

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
        self._prev_z: torch.Tensor | None = None
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
        """Called at the start of each new action chunk.

        Args:
            intent_tokens: (B, 128, 2048) full backbone token sequence
            intent_attention_mask: (B, 128) True for active tokens

        Does NOT reset prev_z/prev_proprio — the last error from the
        previous chunk can still inform the first correction of this chunk.
        """
        self._current_intent_tokens = intent_tokens.detach()
        self._current_intent_mask = intent_attention_mask.detach()

    def reset(self):
        """Full reset at episode start."""
        self._prev_z = None
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
            action_planned: (1, 7) VLA's planned action for this step
            chunk_step: which step within the action chunk (0-7)

        Returns:
            action_corrected: (1, 7)
        """
        z_current = self.visual_encoder(image_current)

        if self._prev_z is not None and self._current_intent_tokens is not None:
            delta_z_ideal = self.forward_model(
                self._prev_z,
                self._prev_proprio,
                self._current_intent_tokens,
                self._current_intent_mask,
            )
            delta_z_actual = z_current - self._prev_z
            prediction_error = delta_z_actual - delta_z_ideal
        else:
            prediction_error = torch.zeros(1, self.config.feature_dim, device=image_current.device)

        chunk_step_t = torch.tensor(
            [[chunk_step]], dtype=torch.float32, device=image_current.device
        )
        delta_a = self.correction_net(
            prediction_error=prediction_error,
            action_vla=action_planned,
            proprio=proprio_current,
            chunk_step=chunk_step_t,
        )

        self._prev_z = z_current
        self._prev_proprio = proprio_current

        return action_planned + delta_a

    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu"):
        """Load trained weights from Phase 1 + Phase 2 checkpoints."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

        if "config" in ckpt:
            pass  # config already set at init

        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            # Try loading sub-module checkpoints individually
            if "visual_encoder" in ckpt:
                self.visual_encoder.load_state_dict(ckpt["visual_encoder"])
            if "forward_model" in ckpt:
                self.forward_model.load_state_dict(ckpt["forward_model"])
            if "proprio_forward" in ckpt:
                self.proprio_forward.load_state_dict(ckpt["proprio_forward"])
            if "correction_net" in ckpt:
                self.correction_net.load_state_dict(ckpt["correction_net"])
