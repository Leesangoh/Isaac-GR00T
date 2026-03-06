"""DINOv2 ViT-S visual encoder with optional LoRA and EMA target encoder.

Outputs 384-dim features (DINOv2 native dim, no projection layer).
Uses patch token mean pooling (not CLS) — verified to be better for dynamics
prediction in prior experiments (16.8% vs 13.8% improvement over baseline).

EMA target encoder prevents temporal collapse (Jointly VICReg alone caused
49x Δz norm reduction; EMA τ=0.996 maintained dz≈0.88, z_std≈1.07).
"""

import copy
import sys
import types

import torch
from torch import nn
from torchvision import transforms as T


FEATURE_DIM = 384  # DINOv2 ViT-S native dimension


class CerebellumVisualEncoder(nn.Module):
    """DINOv2 ViT-S encoder with optional LoRA adapters.

    Input: (B, 3, 98, 98) float32 images in [0, 1]
    Output: (B, 384) patch-mean-pooled features
    """

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        use_lora: bool = True,
        lora_rank: int = 8,
        input_size: int = 98,
    ):
        super().__init__()
        self.feature_dim = FEATURE_DIM
        self.input_size = input_size

        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone)

        if use_lora:
            # Temporarily stub flash_attn_2_cuda if unavailable so peft import succeeds.
            # Remove stub afterward so it doesn't break GR00T's real flash attention.
            _stub_added = False
            if "flash_attn_2_cuda" not in sys.modules:
                try:
                    import flash_attn_2_cuda  # noqa: F401
                except (ImportError, OSError):
                    sys.modules["flash_attn_2_cuda"] = types.ModuleType("flash_attn_2_cuda")
                    _stub_added = True

            from peft import LoraConfig, get_peft_model

            if _stub_added:
                del sys.modules["flash_attn_2_cuda"]

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["qkv"],
                lora_dropout=0.1,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

        self.norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) float32 in [0, 1]
        Returns:
            z: (B, 384)
        """
        # Resize to expected input size if needed (inference images may differ)
        if image.shape[-2] != self.input_size or image.shape[-1] != self.input_size:
            image = torch.nn.functional.interpolate(
                image, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
            )
        x = self.norm(image)
        features = self.backbone.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]  # (B, N_patches, 384)
        return patch_tokens.mean(dim=1)  # (B, 384)


class EMAEncoder:
    """Exponential moving average copy of the online encoder (no gradients).

    Used as the target encoder in BYOL/I-JEPA style training to prevent
    temporal collapse. The target produces z_t+1 supervision while the
    online encoder produces z_t with gradient flow.
    """

    def __init__(self, online_encoder: CerebellumVisualEncoder, tau: float = 0.996):
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.tau = tau

    @torch.no_grad()
    def update(self, online_encoder: CerebellumVisualEncoder):
        """EMA update: θ_target = τ * θ_target + (1-τ) * θ_online."""
        for p_online, p_target in zip(
            online_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data.mul_(self.tau).add_(p_online.data, alpha=1.0 - self.tau)

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        self.target_encoder.eval()
        return self.target_encoder(image)
