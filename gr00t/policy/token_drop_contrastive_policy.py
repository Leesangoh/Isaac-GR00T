"""VLM Token-Drop Contrastive Decoding for GR00T N1.6.

Applies contrastive correction at the velocity level during each denoising step,
using a token-dropped version of VL embeddings as the amateur signal.

The VLM (Eagle backbone) runs only once. The token drop creates a degraded copy
of vl_embeds, and at each denoising step:

    v_full    = DiT(x_t, t, vl_embeds_full)      -- full velocity
    v_amateur = DiT(x_t, t, vl_embeds_dropped)   -- degraded velocity
    v_corrected = v_full + alpha * (v_full - v_amateur)

Three drop variants:
    A (vision):    Zero/mean/noise vision tokens -> amplify visual grounding
    B (language):  Zero/mean/noise language tokens -> amplify task instruction following
    C (attention): Drop top-K most attended tokens (via cross-attention weights)
                   -> amplify the model's most informative signals

Overhead: 2x DiT per step (1x full + 1x dropped). VLM runs only 1x.
Variant C (static) adds 1 extra DiT forward pass for attention weight capture.

Related work:
    - CFG (Ho & Salimans, 2022): score-level contrast (unconditional vs conditional)
    - VCD (Leng et al., 2024): visual contrastive decoding (image degradation)
    - PCD (arXiv 2505.13255): policy contrastive decoding (object masking at pixel level)
    - Autoguidance (Karras et al., NeurIPS 2024): same-model guidance
"""

from contextlib import contextmanager
from enum import Enum
import logging
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EmbodimentTag

from .gr00t_policy import Gr00tPolicy


logger = logging.getLogger(__name__)


class DropVariant(Enum):
    """Token drop strategy for creating the amateur signal."""

    VISION = "vision"  # Variant A: drop vision tokens
    LANGUAGE = "language"  # Variant B: drop language tokens
    ATTENTION = "attention"  # Variant C: drop top-K most attended tokens


class _WeightCaptureProcessor:
    """Wraps an attention processor to also capture cross-attention weights.

    Computes Q*K^T separately to extract attention weights, then delegates
    the actual forward computation to the original processor (e.g. SDPA).
    Only captures weights for cross-attention (encoder_hidden_states is not None).
    """

    def __init__(self, original_processor):
        self.original_processor = original_processor
        self.last_cross_attn_weights = None

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs
    ):
        if encoder_hidden_states is not None:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)

            batch_size = query.shape[0]
            inner_dim = query.shape[-1]
            head_dim = inner_dim // attn.heads

            q = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            k = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * (head_dim**-0.5)
            # Skip attention_mask here: it has been preprocessed by diffusers'
            # prepare_attention_mask() into [B*heads, 1, K] format which is
            # incompatible with our [B, heads, Q, K] scores.  Padding tokens
            # are filtered out by backbone_attention_mask in the caller.
            weights = torch.softmax(scores, dim=-1)
            self.last_cross_attn_weights = weights.detach()

        return self.original_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )


class TokenDropContrastivePolicy(Gr00tPolicy):
    """GR00T N1.6 policy with VLM token-drop contrastive decoding.

    Creates an amateur signal by dropping (zeroing/replacing) vision or language
    tokens from the VLM output, then applies velocity-level contrastive correction
    at each denoising step. The VLM runs only once; overhead is ~2x DiT per step.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        alpha: float = 1.0,
        variant: str | DropVariant = DropVariant.VISION,
        drop_value: str = "zero",
        top_k_ratio: float = 0.3,
        static_attention: bool = True,
        clamp_ratio: float = 0.3,
        verbose: bool = False,
    ):
        """Initialize TokenDropContrastivePolicy.

        Args:
            embodiment_tag: Robot embodiment type.
            model_path: Path to pretrained model checkpoint.
            device: Device for inference (e.g. 'cuda:0').
            strict: Whether to enforce strict input validation.
            alpha: Contrastive amplification factor. 0.0 = vanilla, 1.0 = standard CFG.
            variant: Drop strategy - "vision", "language", or "attention".
            drop_value: Replacement for dropped tokens - "zero", "mean", or "noise".
            top_k_ratio: Variant C only: fraction of tokens to drop (0.0-1.0).
            static_attention: Variant C only: reuse first-step attention weights.
            clamp_ratio: Maximum deviation ratio for clamping. 0 disables clamping.
            verbose: Log per-step delta statistics.
        """
        super().__init__(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=strict,
        )
        self.alpha = alpha
        if isinstance(variant, str):
            self.variant = DropVariant(variant)
        else:
            self.variant = variant
        self.drop_value = drop_value
        self.top_k_ratio = top_k_ratio
        self.static_attention = static_attention
        self.clamp_ratio = clamp_ratio
        self.verbose = verbose

        self._action_head = self._find_action_head()
        self._original_num_inference_timesteps = self._action_head.num_inference_timesteps
        logger.info(
            "Token-drop CD initialized: alpha=%.2f, variant=%s, drop_value=%s, N=%d",
            self.alpha,
            self.variant.value,
            self.drop_value,
            self._original_num_inference_timesteps,
        )

    # ------------------------------------------------------------------
    # Action head discovery (shared with ContrastiveGr00tPolicy)
    # ------------------------------------------------------------------

    def _find_action_head(self) -> torch.nn.Module:
        """Locate the action head submodule that owns the denoising loop."""
        for name, module in self.model.named_modules():
            if hasattr(module, "num_inference_timesteps") and hasattr(
                module, "get_action_with_features"
            ):
                logger.info("Found action head at: %s", name)
                return module
        raise RuntimeError(
            "Could not find action head with num_inference_timesteps "
            "and get_action_with_features in self.model"
        )

    # ------------------------------------------------------------------
    # Velocity computation (shared with ContrastiveGr00tPolicy)
    # ------------------------------------------------------------------

    def _compute_velocity(
        self,
        action_head: torch.nn.Module,
        actions: torch.Tensor,
        t_discretized: int,
        vl_embeds: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        """Compute predicted velocity at a given discretized timestep.

        Replicates the inner loop body of the action head's denoising loop:
        encode actions -> add pos embed -> concat state -> DiT forward -> decode.
        """
        batch_size = actions.shape[0]
        device = actions.device

        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
        action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)

        if action_head.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if action_head.config.use_alternate_vl_dit:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_tensor,
            )

        pred = action_head.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_head.action_horizon :]
        return pred_velocity

    # ------------------------------------------------------------------
    # Token dropping
    # ------------------------------------------------------------------

    def _create_dropped_embeds(
        self,
        vl_embeds: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        """Create amateur VL embeddings by dropping vision or language tokens.

        Args:
            vl_embeds: Full VL embeddings from backbone. (B, seq_len, D)
            backbone_output: Backbone output with image_mask and attention_mask.

        Returns:
            Modified VL embeddings with dropped tokens. (B, seq_len, D)
        """
        image_mask = backbone_output.image_mask  # [B, seq_len] bool
        attn_mask = backbone_output.backbone_attention_mask  # [B, seq_len] bool

        if self.variant == DropVariant.VISION:
            drop_mask = image_mask  # drop vision tokens
        elif self.variant == DropVariant.LANGUAGE:
            drop_mask = (~image_mask) & attn_mask  # drop language tokens (not padding)
        else:
            raise ValueError("Use _compute_attention_drop_mask for ATTENTION variant")

        if self.verbose:
            n_dropped = drop_mask.sum().item()
            n_total = attn_mask.sum().item()
            logger.info(
                "Token drop (%s): %d / %d tokens (%.1f%%)",
                self.variant.value,
                n_dropped,
                n_total,
                100.0 * n_dropped / max(n_total, 1),
            )

        return self._apply_drop(vl_embeds, drop_mask, attn_mask)

    def _apply_drop(
        self,
        vl_embeds: torch.Tensor,
        drop_mask: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply drop_value replacement to masked positions.

        Args:
            vl_embeds: VL embeddings. (B, seq_len, D)
            drop_mask: Boolean mask where True = drop this token. (B, seq_len)
            attn_mask: Boolean mask where True = valid (non-padding) token. (B, seq_len)

        Returns:
            Modified VL embeddings with dropped tokens replaced. (B, seq_len, D)
        """
        vl_dropped = vl_embeds.clone()

        if self.drop_value == "zero":
            vl_dropped[drop_mask] = 0
        elif self.drop_value == "mean":
            # Replace with mean embedding of all valid tokens
            mean_embed = vl_embeds[attn_mask].mean(dim=0)
            vl_dropped[drop_mask] = mean_embed
        elif self.drop_value == "noise":
            # Replace with random noise scaled to match embedding statistics
            embed_scale = vl_embeds[attn_mask].std()
            vl_dropped[drop_mask] = torch.randn_like(vl_dropped[drop_mask]) * embed_scale
        else:
            raise ValueError(f"Unknown drop_value: {self.drop_value!r}")

        return vl_dropped

    # ------------------------------------------------------------------
    # Variant C: attention-weighted token drop
    # ------------------------------------------------------------------

    def _compute_attention_drop_mask(
        self,
        action_head: torch.nn.Module,
        actions: torch.Tensor,
        vl_embeds: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        """Compute drop mask using cross-attention weights (Variant C).

        Temporarily swaps attention processors to capture cross-attention weights,
        runs one DiT forward pass, aggregates weights across layers, and returns
        a boolean mask for the top-K most attended tokens.

        Args:
            action_head: The action head module.
            actions: Current noised actions (initial noise). (B, action_horizon, action_dim)
            vl_embeds: Full VL embeddings. (B, seq_len, D)
            state_features: Encoded state features. (B, state_horizon, D)
            embodiment_id: Embodiment IDs. (B,)
            backbone_output: Backbone output with masks.

        Returns:
            Boolean drop mask. (B, seq_len) where True = drop this token.
        """
        from diffusers.models.attention import Attention

        # Step 1: Find cross-attention Attention modules and swap processors
        original_processors = {}
        capture_processors = {}

        for name, module in action_head.model.named_modules():
            if isinstance(module, Attention) and hasattr(module, "to_q"):
                original_processors[name] = module.processor
                capture = _WeightCaptureProcessor(module.processor)
                capture_processors[name] = capture
                module.processor = capture

        try:
            # Step 2: Run one forward pass at t=0 to capture attention weights
            self._compute_velocity(
                action_head,
                actions,
                0,
                vl_embeds,
                state_features,
                embodiment_id,
                backbone_output,
            )

            # Step 3: Aggregate attention weights across layers
            all_importance = []
            for name, proc in capture_processors.items():
                if proc.last_cross_attn_weights is not None:
                    # [B, heads, Q, K] -> mean over batch, heads, queries -> [K]
                    importance = proc.last_cross_attn_weights.mean(dim=[0, 1, 2])
                    all_importance.append(importance)

            if not all_importance:
                logger.warning(
                    "No cross-attention weights captured; "
                    "falling back to embedding norm for importance"
                )
                importance = vl_embeds.norm(dim=-1).mean(dim=0)  # [seq_len]
            else:
                importance = torch.stack(all_importance).mean(dim=0)  # [seq_len]

            # Step 4: Top-K drop mask
            seq_len = importance.shape[0]
            k = max(1, int(seq_len * self.top_k_ratio))
            topk_indices = importance.topk(k).indices
            drop_mask = torch.zeros(seq_len, dtype=torch.bool, device=vl_embeds.device)
            drop_mask[topk_indices] = True
            # Expand to batch dimension
            drop_mask = drop_mask.unsqueeze(0).expand(vl_embeds.shape[0], -1)

            if self.verbose:
                attn_mask = backbone_output.backbone_attention_mask
                n_dropped = drop_mask[0].sum().item()
                n_total = attn_mask[0].sum().item()
                n_vision = backbone_output.image_mask[0].sum().item()
                n_vision_dropped = (drop_mask[0] & backbone_output.image_mask[0]).sum().item()
                logger.info(
                    "Attention drop: %d / %d tokens (%.1f%%), of which %d / %d are vision tokens",
                    n_dropped,
                    n_total,
                    100.0 * n_dropped / max(n_total, 1),
                    n_vision_dropped,
                    n_vision,
                )

            return drop_mask

        finally:
            # Step 5: Restore original processors
            for name, module in action_head.model.named_modules():
                if name in original_processors:
                    module.processor = original_processors[name]

    # ------------------------------------------------------------------
    # Velocity-level contrastive correction
    # ------------------------------------------------------------------

    def _apply_velocity_contrastive(
        self, v_amateur: torch.Tensor, v_full: torch.Tensor
    ) -> torch.Tensor:
        """Apply contrastive correction to velocity predictions.

        Computes: v_corrected = v_full + alpha * (v_full - v_amateur)

        Args:
            v_amateur: Velocity from dropped embeddings. (B, T, D)
            v_full: Velocity from full embeddings. (B, T, D)

        Returns:
            Contrastively corrected velocity. (B, T, D)
        """
        delta = v_full - v_amateur
        v_corrected = v_full + self.alpha * delta

        if self.clamp_ratio > 0:
            max_abs = (1.0 + self.clamp_ratio) * v_full.abs().clamp(min=1e-6)
            v_corrected = v_corrected.clamp(-max_abs, max_abs)

        if self.verbose:
            delta_norm = delta.norm(dim=-1).mean().item()
            correction_norm = (self.alpha * delta).norm(dim=-1).mean().item()
            full_norm = v_full.norm(dim=-1).mean().item()
            logger.info(
                "Token-drop CD velocity: delta=%.4f, correction=%.4f, v_full=%.4f, ratio=%.4f",
                delta_norm,
                correction_norm,
                full_norm,
                correction_norm / max(full_norm, 1e-8),
            )

        return v_corrected

    # ------------------------------------------------------------------
    # Main inference override
    # ------------------------------------------------------------------

    @contextmanager
    def _contrastive_inference_ctx(self):
        """Context manager that monkey-patches model.get_action for token-drop CD.

        Inside the patched get_action:
        1. Runs backbone (VLM) once.
        2. Encodes state features once.
        3. Creates dropped VL embeddings (amateur signal).
        4. Runs custom denoising loop with per-step velocity-level contrast.
        5. Returns actions in standard output format.
        """
        original_get_action = self.model.get_action
        action_head = self._action_head
        N = self._original_num_inference_timesteps
        policy_self = self

        def patched_get_action(inputs: dict) -> BatchFeature:
            # Step 1: Prepare inputs and run backbone ONCE
            backbone_inputs, action_inputs = policy_self.model.prepare_input(inputs)
            backbone_outputs = policy_self.model.backbone(backbone_inputs)

            # Step 2: Encode state features ONCE (includes vlln normalization)
            features = action_head._encode_features(backbone_outputs, action_inputs)
            vl_embeds = features.backbone_features
            state_features = features.state_features
            embodiment_id = action_inputs.embodiment_id

            # Step 3: Initialize noise
            batch_size = vl_embeds.shape[0]
            device = vl_embeds.device
            actions = torch.randn(
                size=(
                    batch_size,
                    action_head.config.action_horizon,
                    action_head.action_dim,
                ),
                dtype=vl_embeds.dtype,
                device=device,
            )

            # Step 4: Create dropped embeddings (amateur signal)
            if policy_self.alpha != 0.0:
                if policy_self.variant == DropVariant.ATTENTION:
                    drop_mask = policy_self._compute_attention_drop_mask(
                        action_head,
                        actions,
                        vl_embeds,
                        state_features,
                        embodiment_id,
                        backbone_outputs,
                    )
                    attn_mask = backbone_outputs.backbone_attention_mask
                    vl_embeds_dropped = policy_self._apply_drop(vl_embeds, drop_mask, attn_mask)
                else:
                    vl_embeds_dropped = policy_self._create_dropped_embeds(
                        vl_embeds, backbone_outputs
                    )
            else:
                vl_embeds_dropped = None  # Not used when alpha=0

            dt = 1.0 / N

            # Step 5: Denoising loop with per-step velocity correction
            for k in range(N):
                t_cont = k / float(N)
                t_discretized = int(t_cont * action_head.num_timestep_buckets)

                # Compute v_full: velocity with full VL embeddings
                v_full = policy_self._compute_velocity(
                    action_head,
                    actions,
                    t_discretized,
                    vl_embeds,
                    state_features,
                    embodiment_id,
                    backbone_outputs,
                )

                if policy_self.alpha != 0.0:
                    # Compute v_amateur: velocity with dropped VL embeddings
                    v_amateur = policy_self._compute_velocity(
                        action_head,
                        actions,
                        t_discretized,
                        vl_embeds_dropped,
                        state_features,
                        embodiment_id,
                        backbone_outputs,
                    )
                    v_corrected = policy_self._apply_velocity_contrastive(v_amateur, v_full)
                else:
                    v_corrected = v_full

                # Euler integration
                actions = actions + dt * v_corrected

            return BatchFeature(
                data={
                    "action_pred": actions,
                    "backbone_features": vl_embeds,
                    "state_features": state_features,
                }
            )

        try:
            self.model.get_action = patched_get_action
            yield
        finally:
            self.model.get_action = original_get_action

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute actions with token-drop contrastive decoding.

        The parent class _get_action handles observation processing and action
        decoding. We only intercept the model.get_action call within it.
        """
        with self._contrastive_inference_ctx():
            return super()._get_action(observation, options)
