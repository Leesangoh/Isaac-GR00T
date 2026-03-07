"""Intent-conditioned forward models.

TransitionViT (v2): Patch-level ViT transition model (DINO-WM inspired).
  Takes 49 DINOv2 patch tokens + proprio + intent tokens → predicts z_{t+1} patches.
  178 input tokens, 4-layer ViT, outputs first 49 tokens as predicted next state.

IntentForwardModel (v1): Self-attention predictor (I-JEPA style).
  Takes mean-pooled z_t + proprio + intent tokens → predicts Δz_ideal.
  Kept for backward compatibility.

ProprioForwardModel: MLP (unchanged).
  Uses pooled intent (B, 2048) — self-attention is overkill for 8-dim proprio.
"""

import torch
from torch import nn


class TransitionViT(nn.Module):
    """Patch-level ViT transition model for visual state prediction (DINO-WM inspired).

    Token composition (N = 49 + 1 + K, K ≤ 128):
      [patch_0, ..., patch_48, proprio, intent_0, ..., intent_K]

    Self-attention across all tokens, then read out first 49 positions
    as predicted z_{t+1} patch tokens.

    Parameters: ~7.4M (4 layers × 384-dim).
    Latency: ~1.5ms on RTX A6000 (178 tokens × 384-dim × 4 layers).
    """

    def __init__(
        self,
        feature_dim: int = 384,
        proprio_dim: int = 8,
        intent_dim: int = 2048,
        num_patches: int = 49,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
        max_intent_tokens: int = 128,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_patches = num_patches
        self.max_intent_tokens = max_intent_tokens

        # Token-type embeddings (3 types: patch, proprio, intent)
        self.patch_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.proprio_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.intent_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Projections
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, feature_dim),
            nn.GELU(),
        )
        self.intent_proj = nn.Linear(intent_dim, feature_dim)

        # Spatial position embedding for patch tokens only (7×7 grid)
        self.patch_pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output head: LayerNorm on patch positions
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        z_patches: torch.Tensor,
        proprio: torch.Tensor,
        intent_tokens: torch.Tensor,
        intent_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next visual state patch tokens.

        Args:
            z_patches: (B, 49, 384) current DINOv2 patch tokens
            proprio: (B, 8) current proprioception
            intent_tokens: (B, 128, 2048) GR00T backbone tokens
            intent_attention_mask: (B, 128) True for active tokens

        Returns:
            z_next_pred: (B, 49, 384) predicted next-state patch tokens
        """
        B = z_patches.shape[0]

        # 1. Build tokens with type embeddings
        patch_tokens = z_patches + self.patch_token_embed + self.patch_pos_embed
        proprio_token = self.proprio_proj(proprio).unsqueeze(1) + self.proprio_token_embed
        intent_projected = self.intent_proj(intent_tokens) + self.intent_token_embed

        # 2. Concatenate: [49 patches, 1 proprio, 128 intent] = 178 tokens
        tokens = torch.cat([patch_tokens, proprio_token, intent_projected], dim=1)

        # 3. Build attention mask: patches + proprio always active, intent follows mask
        prefix_mask = torch.ones(B, self.num_patches + 1, dtype=torch.bool, device=z_patches.device)
        full_mask = torch.cat([prefix_mask, intent_attention_mask], dim=1)
        padding_mask = ~full_mask  # True = ignore for PyTorch TransformerEncoder

        # 4. Self-attention
        output = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # 5. Read out first 49 positions → predicted z_{t+1}
        patch_output = output[:, : self.num_patches, :]
        return self.output_norm(patch_output)


class IntentForwardModel(nn.Module):
    """I-JEPA style self-attention predictor for visual state change.

    Token composition (N = 2 + num_active_intent_tokens):
      [z_t_proj, proprio_proj, intent_1, intent_2, ..., intent_K]
      - z_t_proj: (B, 1, 384) — DINOv2 visual feature + learnable token embed
      - proprio_proj: (B, 1, 384) — proprioception projected + token embed
      - intent_k: (B, K, 384) — GR00T backbone tokens projected (K ≤ 128)

    Self-attention output at z_t position → Δz_ideal [384].

    Latency: ~0.3ms on RTX 4090 (130 tokens × 384-dim × 2 layers).
    Parameters: ~3.5M.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        proprio_dim: int = 8,
        intent_dim: int = 2048,
        num_layers: int = 2,
        num_heads: int = 6,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
        max_intent_tokens: int = 128,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_intent_tokens = max_intent_tokens

        # Learnable token type embeddings
        self.z_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.proprio_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.intent_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Projection layers (each modality → feature_dim tokens)
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, feature_dim),
            nn.GELU(),
        )
        self.intent_proj = nn.Linear(intent_dim, feature_dim)

        # Learnable positional embedding: [z_t, proprio, intent_0, ..., intent_127]
        self.pos_embed = nn.Parameter(torch.randn(1, 2 + max_intent_tokens, feature_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output head: z_t position → Δz_ideal
        self.output_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        proprio: torch.Tensor,
        intent_tokens: torch.Tensor,
        intent_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict ideal visual feature change conditioned on intent.

        Args:
            z: (B, 384) current visual feature
            proprio: (B, 8) current proprioception
            intent_tokens: (B, 128, 2048) full GR00T backbone tokens
            intent_attention_mask: (B, 128) True for active tokens

        Returns:
            delta_z_ideal: (B, 384)
        """
        B = z.shape[0]

        # 1. Build tokens
        z_token = z.unsqueeze(1) + self.z_token_embed  # (B, 1, 384)
        proprio_token = self.proprio_proj(proprio).unsqueeze(1) + self.proprio_token_embed
        intent_projected = self.intent_proj(intent_tokens) + self.intent_token_embed

        # 2. Concatenate: [z_t, proprio, intent_1, ..., intent_128]
        tokens = torch.cat([z_token, proprio_token, intent_projected], dim=1)  # (B, 130, 384)

        # 3. Add positional embedding
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]

        # 4. Attention mask: z_t, proprio always active; intent follows mask
        prefix_mask = torch.ones(B, 2, dtype=torch.bool, device=z.device)
        full_mask = torch.cat([prefix_mask, intent_attention_mask], dim=1)  # (B, 130)
        # PyTorch TransformerEncoder: src_key_padding_mask True = padding (ignore)
        padding_mask = ~full_mask

        # 5. Self-attention
        output = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # 6. Read z_t position output
        z_output = output[:, 0, :]  # (B, 384)

        # 7. Output head → Δz_ideal
        return self.output_head(z_output)


class ProprioForwardModel(nn.Module):
    """Predicts ideal proprioception change conditioned on pooled intent.

    Input: concat(proprio [8], intent_pooled [2048]) = 2056
    Output: Δproprio [8]

    Uses pooled intent — self-attention is overkill for 8-dim output.
    """

    def __init__(
        self,
        proprio_dim: int = 8,
        intent_dim: int = 2048,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim + intent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proprio_dim),
        )

    def forward(self, proprio: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([proprio, intent], dim=-1)
        return self.net(x)
