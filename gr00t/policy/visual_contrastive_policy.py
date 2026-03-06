"""Visual Contrastive Decoding (VCD) for GR00T N1.6.

Applies contrastive correction at the velocity level using noisy-image
VLM features as the amateur signal.  Unlike token-drop CD (which modifies
embeddings post-VLM), VCD adds Gaussian noise to pixel_values and runs
the VLM backbone twice to obtain two distinct sets of VL embeddings:

    vl_embeds_full  = VLM(clean_image, language)
    vl_embeds_noisy = VLM(noisy_image, language)

    # DiT denoising loop (velocity level)
    v_full    = DiT(x_t, t, vl_embeds_full)
    v_amateur = DiT(x_t, t, vl_embeds_noisy)
    v_corrected = v_full + alpha * (v_full - v_amateur)

The amateur "sees" a degraded version of the scene at the pixel level,
so the contrastive signal amplifies the model's visual understanding —
the part of the action that depends on clear visual perception.

Overhead: 2x VLM + 2x DiT per denoising step.

Related work:
    - VCD (Leng et al., 2024): Visual Contrastive Decoding for LVLMs
    - PCD (arXiv 2505.13255): Policy Contrastive Decoding (object masking)
    - CFG (Ho & Salimans, 2022): Classifier-Free Guidance
"""

from contextlib import contextmanager
import logging
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EmbodimentTag

from .gr00t_policy import Gr00tPolicy


logger = logging.getLogger(__name__)


class VisualContrastivePolicy(Gr00tPolicy):
    """GR00T N1.6 policy with Visual Contrastive Decoding.

    Runs the VLM backbone twice (clean + noisy images), then applies
    velocity-level contrastive correction at each denoising step using
    the noisy-image features as the amateur signal.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        alpha: float = 1.0,
        noise_std: float = 0.5,
        clamp_ratio: float = 0.3,
        verbose: bool = False,
    ):
        """Initialize VisualContrastivePolicy.

        Args:
            embodiment_tag: Robot embodiment type.
            model_path: Path to pretrained model checkpoint.
            device: Device for inference (e.g. 'cuda:0').
            strict: Whether to enforce strict input validation.
            alpha: Contrastive amplification factor. 0.0 = vanilla.
            noise_std: Gaussian noise std added to normalized pixel_values.
                Typical range 0.1-1.0 (pixel_values are normalized ~[-1, 1]).
            clamp_ratio: Maximum deviation ratio for velocity clamping.
                0 disables clamping.
            verbose: Log per-step delta statistics.
        """
        super().__init__(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=strict,
        )
        self.alpha = alpha
        self.noise_std = noise_std
        self.clamp_ratio = clamp_ratio
        self.verbose = verbose

        self._action_head = self._find_action_head()
        self._original_num_inference_timesteps = self._action_head.num_inference_timesteps
        logger.info(
            "VCD initialized: alpha=%.2f, noise_std=%.2f, clamp_ratio=%.2f, N=%d",
            self.alpha,
            self.noise_std,
            self.clamp_ratio,
            self._original_num_inference_timesteps,
        )

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
        """Compute predicted velocity at a given discretized timestep."""
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
        self, v_amateur: torch.Tensor, v_full: torch.Tensor
    ) -> torch.Tensor:
        """Apply contrastive correction: v_corrected = v_full + alpha * (v_full - v_amateur)."""
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
                "VCD velocity: delta=%.4f, correction=%.4f, v_full=%.4f, ratio=%.4f",
                delta_norm,
                correction_norm,
                full_norm,
                correction_norm / max(full_norm, 1e-8),
            )

        return v_corrected

    @contextmanager
    def _contrastive_inference_ctx(self):
        """Context manager that monkey-patches model.get_action for VCD.

        Inside the patched get_action:
        1. Prepares inputs once.
        2. Runs VLM backbone twice: clean images -> vl_embeds_full,
           noisy images -> vl_embeds_noisy.
        3. Encodes state features once.
        4. Runs custom denoising loop with per-step velocity-level contrast.
        """
        original_get_action = self.model.get_action
        action_head = self._action_head
        N = self._original_num_inference_timesteps
        policy_self = self

        def patched_get_action(inputs: dict) -> BatchFeature:
            # Step 1: Prepare inputs (handles vlm_content collation, device transfer)
            backbone_inputs, action_inputs = policy_self.model.prepare_input(inputs)

            # Step 2a: Run backbone with clean images
            backbone_outputs_clean = policy_self.model.backbone(backbone_inputs)

            if policy_self.alpha != 0.0:
                # Step 2b: Create noisy pixel_values and run backbone again.
                # pixel_values may be a list of tensors (one per image) or a
                # single stacked tensor, depending on the Eagle processor.
                pv = backbone_inputs["pixel_values"]
                if isinstance(pv, list):
                    noisy_pixel_values = [
                        t + torch.randn_like(t) * policy_self.noise_std for t in pv
                    ]
                else:
                    noisy_pixel_values = pv + (torch.randn_like(pv) * policy_self.noise_std)
                noisy_backbone_inputs = BatchFeature(
                    data={
                        "input_ids": backbone_inputs["input_ids"],
                        "attention_mask": backbone_inputs["attention_mask"],
                        "pixel_values": noisy_pixel_values,
                    }
                )
                backbone_outputs_noisy = policy_self.model.backbone(noisy_backbone_inputs)

                if policy_self.verbose:
                    if isinstance(pv, list):
                        clean_norm = sum(t.norm().item() ** 2 for t in pv) ** 0.5
                        noise_norm = (
                            sum((n - t).norm().item() ** 2 for n, t in zip(noisy_pixel_values, pv))
                            ** 0.5
                        )
                    else:
                        clean_norm = pv.norm().item()
                        noise_norm = (noisy_pixel_values - pv).norm().item()
                    pixel_snr = clean_norm / max(noise_norm, 1e-8)
                    logger.info(
                        "VCD pixel noise: std=%.3f, SNR=%.2f",
                        policy_self.noise_std,
                        pixel_snr,
                    )

            # Step 3: Encode features from clean backbone (state_features + vlln)
            features = action_head._encode_features(backbone_outputs_clean, action_inputs)
            vl_embeds_full = features.backbone_features
            state_features = features.state_features
            embodiment_id = action_inputs.embodiment_id

            # Apply vlln to noisy backbone features
            if policy_self.alpha != 0.0:
                vl_embeds_noisy = action_head.vlln(backbone_outputs_noisy.backbone_features)
            else:
                vl_embeds_noisy = None

            # Step 4: Initialize noise
            batch_size = vl_embeds_full.shape[0]
            device = vl_embeds_full.device
            actions = torch.randn(
                size=(
                    batch_size,
                    action_head.config.action_horizon,
                    action_head.action_dim,
                ),
                dtype=vl_embeds_full.dtype,
                device=device,
            )

            dt = 1.0 / N

            # Step 5: Denoising loop with velocity-level contrast
            for k in range(N):
                t_cont = k / float(N)
                t_discretized = int(t_cont * action_head.num_timestep_buckets)

                v_full = policy_self._compute_velocity(
                    action_head,
                    actions,
                    t_discretized,
                    vl_embeds_full,
                    state_features,
                    embodiment_id,
                    backbone_outputs_clean,
                )

                if policy_self.alpha != 0.0:
                    v_amateur = policy_self._compute_velocity(
                        action_head,
                        actions,
                        t_discretized,
                        vl_embeds_noisy,
                        state_features,
                        embodiment_id,
                        backbone_outputs_clean,
                    )
                    v_corrected = policy_self._apply_velocity_contrastive(v_amateur, v_full)
                else:
                    v_corrected = v_full

                actions = actions + dt * v_corrected

            return BatchFeature(
                data={
                    "action_pred": actions,
                    "backbone_features": vl_embeds_full,
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
        """Compute actions with Visual Contrastive Decoding."""
        with self._contrastive_inference_ctx():
            return super()._get_action(observation, options)
