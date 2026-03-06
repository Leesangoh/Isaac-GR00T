"""Velocity-Level Contrastive Decoding (DDCD v2) for GR00T N1.6.

This module implements ContrastiveGr00tPolicy, which enhances GR00T N1.6 inference
by applying contrastive correction at the velocity level during each denoising step,
analogous to Classifier-Free Guidance in diffusion models.

At each denoising step k of the N-step loop:

    v_full = model(x_k, t=t_k)       -- velocity with correct timestep (informed)
    v_rough = model(x_k, t=0)        -- velocity pretending it's step 0 (uninformed)
    v_corrected = v_full + alpha * (v_full - v_rough)

Key insight: At step 0, t_k = 0, so v_full == v_rough and no correction is applied.
Correction naturally kicks in at steps 1, 2, ... where the model benefits from
knowing t > 0.  This keeps results on the valid action manifold, unlike output-level
correction which can push results outside it.

Overhead: N extra DiT forward passes (one per step with t=0, skipping step 0).
The backbone (VLM) still runs only once.

Inspired by Contrastive Decoding (Li 2022), Classifier-Free Guidance (Ho 2022),
and Autoguidance (Karras et al., NeurIPS 2024).
"""

from contextlib import contextmanager
import logging
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EmbodimentTag

from .gr00t_policy import Gr00tPolicy


logger = logging.getLogger(__name__)


class ContrastiveGr00tPolicy(Gr00tPolicy):
    """GR00T N1.6 policy with velocity-level contrastive decoding.

    Intercepts model.get_action to run a custom denoising loop where each
    step's velocity prediction is contrastively corrected by comparing the
    informed prediction (at correct timestep t_k) against an uninformed
    prediction (at t=0).

    The backbone (VLM) forward pass runs only once; only the DiT forward
    pass is doubled per denoising step (skipping step 0 where t_k=0 already),
    keeping overhead at ~(N-1) extra DiT evaluations for an N-step loop.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        alpha: float = 1.0,
        clamp_ratio: float = 0.3,
        verbose: bool = False,
    ):
        """Initialize ContrastiveGr00tPolicy.

        Args:
            embodiment_tag: Robot embodiment type.
            model_path: Path to pretrained model checkpoint.
            device: Device for inference (e.g. 'cuda:0').
            strict: Whether to enforce strict input validation.
            alpha: Contrastive amplification factor. 0.0 = vanilla, 1.0 = 2x refinement.
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
        self.clamp_ratio = clamp_ratio
        self.verbose = verbose

        self._action_head = self._find_action_head()
        self._original_num_inference_timesteps = self._action_head.num_inference_timesteps
        logger.info(
            "DDCD v2 (velocity-level) initialized: alpha=%.2f, clamp_ratio=%.2f, N=%d",
            self.alpha,
            self.clamp_ratio,
            self._original_num_inference_timesteps,
        )

    def _find_action_head(self) -> torch.nn.Module:
        """Locate the action head submodule that owns the denoising loop.

        Searches self.model recursively for a module with both
        num_inference_timesteps and get_action_with_features attributes.

        Returns:
            The action head module.

        Raises:
            RuntimeError: If no suitable action head is found.
        """
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

        Args:
            action_head: The action head module.
            actions: Current noised action tensor. (B, action_horizon, action_dim)
            t_discretized: Discretized timestep bucket value.
            vl_embeds: Vision-language embeddings from backbone. (B, seq_len, dim)
            state_features: Encoded state features. (B, state_horizon, dim)
            embodiment_id: Embodiment IDs. (B,)
            backbone_output: Full backbone output (for image_mask, attention_mask).

        Returns:
            Predicted velocity tensor. (B, action_horizon, action_dim)
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

    def _apply_velocity_contrastive(
        self, v_rough: torch.Tensor, v_full: torch.Tensor
    ) -> torch.Tensor:
        """Apply contrastive correction to velocity predictions.

        Computes: v_corrected = v_full + alpha * (v_full - v_rough)

        Optionally clamps the result so that each element does not exceed
        (1 + clamp_ratio) * |v_full| in absolute value, preventing runaway
        extrapolation.

        Args:
            v_rough: Velocity prediction at t=0 (uninformed). (B, T, D)
            v_full: Velocity prediction at correct timestep (informed). (B, T, D)

        Returns:
            Contrastively corrected velocity. (B, T, D)
        """
        delta = v_full - v_rough
        v_corrected = v_full + self.alpha * delta

        if self.clamp_ratio > 0:
            max_abs = (1.0 + self.clamp_ratio) * v_full.abs().clamp(min=1e-6)
            v_corrected = v_corrected.clamp(-max_abs, max_abs)

        if self.verbose:
            delta_norm = delta.norm(dim=-1).mean().item()
            correction_norm = (self.alpha * delta).norm(dim=-1).mean().item()
            full_norm = v_full.norm(dim=-1).mean().item()
            logger.info(
                "DDCD velocity step: delta_norm=%.4f, correction_norm=%.4f, "
                "v_full_norm=%.4f, ratio=%.4f",
                delta_norm,
                correction_norm,
                full_norm,
                correction_norm / max(full_norm, 1e-8),
            )

        return v_corrected

    @contextmanager
    def _contrastive_inference_ctx(self):
        """Context manager that monkey-patches self.model.get_action for velocity-level DDCD.

        Inside the patched get_action:
        1. Runs backbone (VLM) forward pass once.
        2. Encodes state features once.
        3. Runs a custom denoising loop where each step applies velocity-level
           contrastive correction before the Euler integration step.
        4. Returns actions in the standard output format.

        The original get_action is always restored, even if an exception occurs.
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

            # Step 3: Initialize noise (same as original denoising loop)
            batch_size = vl_embeds.shape[0]
            device = vl_embeds.device
            actions = torch.randn(
                size=(batch_size, action_head.config.action_horizon, action_head.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
            )

            dt = 1.0 / N

            # Step 4: Custom denoising loop with per-step velocity correction
            for k in range(N):
                t_cont = k / float(N)
                t_discretized = int(t_cont * action_head.num_timestep_buckets)

                # Compute v_full: velocity with correct timestep (informed prediction)
                v_full = policy_self._compute_velocity(
                    action_head,
                    actions,
                    t_discretized,
                    vl_embeds,
                    state_features,
                    embodiment_id,
                    backbone_outputs,
                )

                if k > 0 and policy_self.alpha != 0.0:
                    # Compute v_rough: velocity pretending it's step 0 (uninformed)
                    v_rough = policy_self._compute_velocity(
                        action_head,
                        actions,
                        0,
                        vl_embeds,
                        state_features,
                        embodiment_id,
                        backbone_outputs,
                    )
                    # Apply velocity-level contrastive correction
                    v_corrected = policy_self._apply_velocity_contrastive(v_rough, v_full)
                else:
                    # At k=0: t_discretized=0, so v_full==v_rough, no correction needed.
                    # Also skip if alpha=0 (pure vanilla behavior).
                    v_corrected = v_full

                # Euler integration with corrected velocity
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
        """Compute actions with velocity-level DDCD via monkey-patched model.get_action.

        The parent class _get_action handles observation processing and action
        decoding. We only intercept the model.get_action call within it.

        Args:
            observation: Batched observation dictionary.
            options: Optional parameters (currently unused).

        Returns:
            Tuple of (actions_dict, info_dict).
        """
        with self._contrastive_inference_ctx():
            return super()._get_action(observation, options)
