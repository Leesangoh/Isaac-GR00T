"""Intent-conditioned forward models.

IntentForwardModel: Self-attention predictor (I-JEPA style).
  Takes z_t, proprio, and FULL intent token sequence (B, 128, 2048) as input.
  Each modality is projected to 384-dim tokens, then self-attention lets them
  interact. z_t position output → Δz_ideal.

  Previous MLP version had 2440→256 bottleneck (9.5x compression), which
  destroyed most of the 2048-dim intent information. Self-attention preserves
  individual token relationships and lets z_t selectively attend to intent.

ProprioForwardModel: MLP (unchanged).
  Uses pooled intent (B, 2048) — self-attention is overkill for 8-dim proprio.
"""

import torch
from torch import nn


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
