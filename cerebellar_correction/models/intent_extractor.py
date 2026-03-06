"""Intent vector extraction from GR00T VLM backbone via forward hook.

Extracts the VLM hidden state (backbone_features) during GR00T's forward pass.
Supports two modes:
  - get_intent_tokens(): full token sequence (B, 128, 2048) + mask for self-attention forward model
  - get_intent_pooled(): mean-pooled (B, 2048) for ProprioForwardModel and other MLP-based modules

GR00T backbone output:
    backbone_features: (B, S, 2048) — variable-length (typically 103-164 tokens)
    image_mask: (B, S)
    attention_mask: (B, S)

get_intent_tokens() pads/truncates to MAX_TOKENS=128 for consistent downstream shapes.

D_intent = 2048 (Qwen3-1.7B hidden_size, confirmed from checkpoint config).
"""

import torch
from torch import nn


class IntentExtractor(nn.Module):
    """Extracts intent vectors from GR00T's Eagle backbone via forward hook.

    Registers a hook on the backbone module to capture backbone_features
    after each forward pass.

    Not a trainable module — just reads from GR00T's frozen representations.
    """

    INTENT_DIM = 2048
    MAX_TOKENS = 128  # 81 image + up to 47 text (covers 99.84% of BridgeData)

    def __init__(self, groot_model: nn.Module):
        super().__init__()
        self._hook_data: dict[str, torch.Tensor] = {}
        self._hook_handle = groot_model.backbone.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # EagleBackbone.forward() returns BatchFeature with these keys
        self._hook_data["features"] = output["backbone_features"].detach()
        self._hook_data["image_mask"] = output["image_mask"].detach()
        self._hook_data["attention_mask"] = output["backbone_attention_mask"].detach()

    def get_intent_tokens(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full token sequence for self-attention forward model.

        Pads or truncates to MAX_TOKENS (128) since the backbone may produce
        variable-length sequences depending on input.

        Returns:
            intent_tokens: (B, MAX_TOKENS, 2048) — backbone features, padded/truncated
            attention_mask: (B, MAX_TOKENS) — True for active (non-padding) tokens
        """
        features = self._hook_data["features"]  # (B, S, 2048)
        attn_mask = self._hook_data["attention_mask"]  # (B, S)
        S = features.shape[1]

        if S == self.MAX_TOKENS:
            return features, attn_mask
        elif S > self.MAX_TOKENS:
            return features[:, : self.MAX_TOKENS, :], attn_mask[:, : self.MAX_TOKENS]
        else:
            B, _, D = features.shape
            pad_len = self.MAX_TOKENS - S
            feat_pad = features.new_zeros(B, pad_len, D)
            mask_pad = attn_mask.new_zeros(B, pad_len)
            return (
                torch.cat([features, feat_pad], dim=1),
                torch.cat([attn_mask, mask_pad], dim=1),
            )

    def get_intent_pooled(self, pooling: str = "all_active") -> torch.Tensor:
        """Mean-pooled intent vector for MLP-based modules (ProprioForwardModel etc).

        Args:
            pooling: "all_active" (image+text), "image_only", or "text_only"

        Returns:
            (B, 2048) intent vector.
        """
        features = self._hook_data["features"]  # (B, S, 2048)
        image_mask = self._hook_data["image_mask"]  # (B, S)
        attn_mask = self._hook_data["attention_mask"]  # (B, S)

        if pooling == "all_active":
            mask = attn_mask
        elif pooling == "image_only":
            mask = image_mask
        elif pooling == "text_only":
            mask = ~image_mask & attn_mask
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        mask_f = mask.unsqueeze(-1).float()  # (B, S, 1)
        pooled = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return pooled  # (B, 2048)

    def get_intent(self, pooling: str = "all_active") -> torch.Tensor:
        """Backward-compatible alias for get_intent_pooled()."""
        return self.get_intent_pooled(pooling)

    def remove_hook(self):
        self._hook_handle.remove()

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "IntentExtractor has no forward(). "
            "Call get_intent_tokens() or get_intent_pooled() after GR00T forward."
        )
