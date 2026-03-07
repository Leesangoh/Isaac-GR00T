"""Intent-conditioned forward models.

TransitionViT (v2): Patch-level ViT transition model (DINO-WM inspired).
  Takes 49 DINOv2 patch tokens + proprio + intent tokens → predicts z_{t+1} patches.
  178 input tokens, 4-layer ViT, outputs first 49 tokens as predicted next state.

ActionConditionedTransitionViT: Oracle diagnostic model.
  Same as TransitionViT but adds expert action (7-dim) as an additional token.
  179 input tokens: [49 patches, 1 proprio, 1 action, 128 intent].
  Tests whether deterministic action signal resolves the intent bottleneck.

SpatiotemporalTransitionViT: Multi-frame transition model (MEM-inspired).
  Takes H frames × 49 patches with factorized spatiotemporal attention.
  Spatial attention within each frame, temporal attention across frames for
  same patch position (causal). Temporal attention at layers 2, 4.

IntentForwardModel (v1): Self-attention predictor (I-JEPA style).
  Takes mean-pooled z_t + proprio + intent tokens → predicts Δz_ideal.
  Kept for backward compatibility.

ProprioForwardModel: MLP (unchanged).
  Uses pooled intent (B, 2048) — self-attention is overkill for 8-dim proprio.
"""

import math

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


# ======================================================================
# Experiment 1: Action-conditioned oracle
# ======================================================================


class ActionConditionedTransitionViT(nn.Module):
    """TransitionViT variant with expert action token for oracle diagnostic.

    Token composition (N = 49 + 1 + 1 + K, K <= 128):
      [patch_0, ..., patch_48, proprio, action, intent_0, ..., intent_K]

    The action token provides deterministic next-state information that intent
    tokens may lack. This model tests whether the forward prediction bottleneck
    is in intent signal quality vs. model capacity.

    Parameters: ~7.5M (slightly more than TransitionViT due to action projection).
    """

    def __init__(
        self,
        feature_dim: int = 384,
        proprio_dim: int = 8,
        action_dim: int = 7,
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

        # Token-type embeddings (4 types: patch, proprio, action, intent)
        self.patch_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.proprio_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.action_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.intent_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Projections
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, feature_dim),
            nn.GELU(),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, feature_dim),
            nn.GELU(),
        )
        self.intent_proj = nn.Linear(intent_dim, feature_dim)

        # Spatial position embedding for patch tokens
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

        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        z_patches: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        intent_tokens: torch.Tensor,
        intent_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next visual state patch tokens with action conditioning.

        Args:
            z_patches: (B, 49, 384) current DINOv2 patch tokens
            proprio: (B, 8) current proprioception
            action: (B, 7) expert action at current timestep
            intent_tokens: (B, 128, 2048) GR00T backbone tokens (possibly stale)
            intent_attention_mask: (B, 128) True for active tokens

        Returns:
            z_next_pred: (B, 49, 384) predicted next-state patch tokens
        """
        B = z_patches.shape[0]

        # 1. Build tokens with type embeddings
        patch_tokens = z_patches + self.patch_token_embed + self.patch_pos_embed
        proprio_token = self.proprio_proj(proprio).unsqueeze(1) + self.proprio_token_embed
        action_token = self.action_proj(action).unsqueeze(1) + self.action_token_embed
        intent_projected = self.intent_proj(intent_tokens) + self.intent_token_embed

        # 2. Concatenate: [49 patches, 1 proprio, 1 action, 128 intent] = 179 tokens
        tokens = torch.cat([patch_tokens, proprio_token, action_token, intent_projected], dim=1)

        # 3. Build attention mask: patches + proprio + action always active
        prefix_mask = torch.ones(B, self.num_patches + 2, dtype=torch.bool, device=z_patches.device)
        full_mask = torch.cat([prefix_mask, intent_attention_mask], dim=1)
        padding_mask = ~full_mask

        # 4. Self-attention
        output = self.transformer(tokens, src_key_padding_mask=padding_mask)

        # 5. Read out first 49 positions -> predicted z_{t+1}
        patch_output = output[:, : self.num_patches, :]
        return self.output_norm(patch_output)


# ======================================================================
# Experiment 2: Multi-frame spatiotemporal
# ======================================================================


class SpatiotemporalTransitionViT(nn.Module):
    """Multi-frame ViT with MEM-style factorized spatiotemporal attention.

    Takes H frames of DINOv2 patch tokens and uses factorized attention:
      - All layers: spatial attention within each frame
      - Layers 2, 4 (1-indexed): additive composition of spatial + temporal attention
        using same QKV weights (MEM Appendix C, Eq. 3)

    Temporal attention: same patch position across frames, causal (frame t sees <= t).
    Additive composition: attn_out = spatial_attn(x) + temporal_attn(x), then MLP.

    Token layout: [frame_0_patches, ..., frame_{H-1}_patches, proprio, intent_0, ..., intent_K]
    Total: H*49 + 1 + K tokens

    Only reads the last frame's 49 patches as output (predicted z_{t+1}).

    Key design choices (following MEM):
      - Sinusoidal temporal position embedding with e(0)=0 (current frame = no shift)
      - Additive attention: spatial_attn + temporal_attn with same QKV weights
      - Causal temporal mask: frame t cannot attend to future frames
      - Proprio and intent tokens have global attention (attend to/from all tokens)

    Parameters: ~7.6M (4 layers x 384-dim).
    Latency: ~2.0ms on RTX A6000 with H=3 (vs ~1.5ms for single-frame).
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
        history_len: int = 3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_patches = num_patches
        self.max_intent_tokens = max_intent_tokens
        self.history_len = history_len
        self.num_layers = num_layers

        # Token-type embeddings
        self.patch_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.proprio_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.intent_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Projections
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, feature_dim),
            nn.GELU(),
        )
        self.intent_proj = nn.Linear(intent_dim, feature_dim)

        # Spatial position embedding for patches (shared across frames)
        self.patch_pos_embed = nn.Parameter(torch.randn(1, num_patches, feature_dim) * 0.02)

        # Sinusoidal temporal position embedding (MEM: e(0)=0 for current frame)
        temporal_embed = self._build_sinusoidal_temporal_embedding(history_len, feature_dim)
        self.register_buffer("temporal_pos_embed", temporal_embed.unsqueeze(0).unsqueeze(2))
        # shape: (1, H, 1, feature_dim)

        # Layer components: nn.MultiheadAttention + MLP (not TransformerEncoderLayer)
        # because additive attention requires calling attn twice with different masks
        self.attn_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()  # pre-attn norm
        self.norm2_layers = nn.ModuleList()  # pre-mlp norm

        for _ in range(num_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    feature_dim, num_heads, dropout=dropout, batch_first=True,
                )
            )
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(feature_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, feature_dim),
                nn.Dropout(dropout),
            ))
            self.norm1_layers.append(nn.LayerNorm(feature_dim))
            self.norm2_layers.append(nn.LayerNorm(feature_dim))

        # Which layers add temporal attention (0-indexed: layers 1, 3 = "layers 2, 4")
        self.temporal_layer_indices = {1, 3}

        self.output_norm = nn.LayerNorm(feature_dim)

        # Cache for attention masks (built once per forward, reused)
        self._cached_masks = {}

    @staticmethod
    def _build_sinusoidal_temporal_embedding(history_len: int, dim: int) -> torch.Tensor:
        """Build sinusoidal temporal position embedding with e(0)=0.

        MEM Appendix C: ẑ_{p,t}^{l-1} = z_{p,t}^{l-1} + e(t)
        timesteps: t ∈ {-(H-1), ..., -1, 0}
        Current frame t=0 → e(0) = 0 (identical to single-image case).
        """
        timesteps = torch.arange(-(history_len - 1), 1, dtype=torch.float32)  # e.g., [-2, -1, 0]

        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        args = timesteps[:, None] * freqs[None, :]  # (H, half_dim)
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (H, dim)

        # Guarantee e(0) = 0: subtract the last row (t=0) from all rows
        embeddings = embeddings - embeddings[-1:]

        return embeddings  # (H, dim), last row is all-zero

    def _build_attention_masks(
        self,
        H: int,
        K_active_max: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build spatial-only and temporal-only attention masks.

        Args:
            H: number of history frames
            K_active_max: max intent tokens (for mask size)
            device: torch device

        Returns:
            spatial_mask: (N, N) additive mask for spatial-only attention
            temporal_only_mask: (N, N) additive mask for temporal-only attention
        """
        P = self.num_patches
        n_patch_total = H * P
        n_global = 1 + self.max_intent_tokens
        N = n_patch_total + n_global

        # === Spatial-only mask ===
        spatial_mask = torch.full((N, N), float("-inf"), device=device)

        for f in range(H):
            # Patches in frame f can attend to each other
            start = f * P
            end = start + P
            spatial_mask[start:end, start:end] = 0.0

        # Global tokens: bidirectional with everything
        spatial_mask[n_patch_total:, :] = 0.0  # global → all
        spatial_mask[:, n_patch_total:] = 0.0  # all → global

        # === Temporal-only mask (NO intra-frame spatial, only cross-frame same-position) ===
        temporal_only_mask = torch.full((N, N), float("-inf"), device=device)

        for p in range(P):
            for f_q in range(H):
                q_idx = f_q * P + p
                for f_k in range(f_q + 1):  # causal: attend to frames <= f_q
                    k_idx = f_k * P + p  # same patch position only
                    temporal_only_mask[q_idx, k_idx] = 0.0

        # Global tokens: bidirectional with everything
        temporal_only_mask[n_patch_total:, :] = 0.0
        temporal_only_mask[:, n_patch_total:] = 0.0

        return spatial_mask, temporal_only_mask

    def forward(
        self,
        history_patches: torch.Tensor,
        proprio: torch.Tensor,
        intent_tokens: torch.Tensor,
        intent_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next visual state from multi-frame history.

        Args:
            history_patches: (B, H, 49, 384) DINOv2 patch tokens for H frames
                             Ordered oldest-to-newest: frame 0 = oldest, frame H-1 = current
            proprio: (B, 8) current proprioception
            intent_tokens: (B, 128, 2048) GR00T backbone tokens (possibly stale)
            intent_attention_mask: (B, 128) True for active tokens

        Returns:
            z_next_pred: (B, 49, 384) predicted next-state patch tokens
        """
        B, H, P, D = history_patches.shape
        assert P == self.num_patches and D == self.feature_dim

        # 1. Add spatial + sinusoidal temporal position embeddings to patch tokens
        patch_tokens = (
            history_patches
            + self.patch_token_embed
            + self.patch_pos_embed.unsqueeze(1)       # (1, 1, P, D) broadcast over frames
            + self.temporal_pos_embed[:, :H, :, :]    # (1, H, 1, D) sinusoidal, e(0)=0
        )  # (B, H, P, D)

        # Flatten frames: (B, H*P, D)
        patch_tokens = patch_tokens.reshape(B, H * P, D)

        # 2. Build global tokens
        proprio_token = self.proprio_proj(proprio).unsqueeze(1) + self.proprio_token_embed
        intent_projected = self.intent_proj(intent_tokens) + self.intent_token_embed

        # 3. Concatenate: [H*49 patches, 1 proprio, 128 intent]
        tokens = torch.cat([patch_tokens, proprio_token, intent_projected], dim=1)
        N = tokens.shape[1]

        # 4. Build key_padding_mask: patches + proprio always active, intent follows mask
        prefix_mask = torch.ones(B, H * P + 1, dtype=torch.bool, device=tokens.device)
        full_mask = torch.cat([prefix_mask, intent_attention_mask], dim=1)
        padding_mask = ~full_mask  # True = ignore

        # 5. Build attention masks (cached for efficiency)
        cache_key = (H, self.max_intent_tokens, tokens.device)
        if cache_key not in self._cached_masks:
            self._cached_masks[cache_key] = self._build_attention_masks(
                H, self.max_intent_tokens, tokens.device
            )
        spatial_mask, temporal_only_mask = self._cached_masks[cache_key]

        # Trim masks to actual sequence length if needed
        spatial_mask_t = spatial_mask[:N, :N]
        temporal_only_mask_t = temporal_only_mask[:N, :N]

        # 6. Run through layers: pre-norm, attention (additive for temporal layers), MLP
        x = tokens
        for i in range(self.num_layers):
            attn = self.attn_layers[i]
            mlp = self.mlp_layers[i]
            norm1 = self.norm1_layers[i]
            norm2 = self.norm2_layers[i]

            # Pre-LayerNorm
            x_norm = norm1(x)

            if i in self.temporal_layer_indices:
                # MEM Additive Composition: same QKV weights, different masks
                spatial_out, _ = attn(
                    x_norm, x_norm, x_norm,
                    attn_mask=spatial_mask_t,
                    key_padding_mask=padding_mask,
                )
                temporal_out, _ = attn(
                    x_norm, x_norm, x_norm,
                    attn_mask=temporal_only_mask_t,
                    key_padding_mask=padding_mask,
                )
                attn_out = spatial_out + temporal_out
            else:
                # Spatial only
                attn_out, _ = attn(
                    x_norm, x_norm, x_norm,
                    attn_mask=spatial_mask_t,
                    key_padding_mask=padding_mask,
                )

            # Residual + MLP
            x = x + attn_out
            x = x + mlp(norm2(x))

        # 7. Read out LAST frame's 49 patches -> predicted z_{t+1}
        last_frame_start = (H - 1) * P
        last_frame_end = H * P
        patch_output = x[:, last_frame_start:last_frame_end, :]
        return self.output_norm(patch_output)
