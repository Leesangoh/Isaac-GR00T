"""Restart Sampling for GR00T N1.6 Flow Matching.

Improves sample quality by injecting stochastic noise mid-trajectory and
re-denoising, exploiting stochastic contraction to reduce ODE discretization
error.  Unlike contrastive decoding approaches (DDCD, Token-Drop, VCD), this
requires no "amateur" signal — it is a pure sampling-time enhancement.

Algorithm:
    Phase 1: Standard ODE      t=0 → t_restart     (steps_phase1 steps)
    Phase 2: K restart cycles:
      2a. Re-noise:            t_restart → t_back   (add noise, 0 NFE)
      2b. Re-denoise ODE:      t_back → t_restart   (steps_restart steps)
    Phase 3: Standard ODE      t_restart → t=1.0    (steps_phase3 steps)

    Total NFE = steps_phase1 + K × steps_restart + steps_phase3

Three re-noise methods:
    - sdedit:        x_back = x + (t_current - t_target) × z
    - interpolation: x_back = (1-t_target)×z + t_target×x
    - scaled:        x_back = α×x + σ×z  where α = t_target/t_current,
                     σ = √(1-α²)×(1-t_target)

Reference:
    Xu et al., "Restart Sampling for Improving Generative Processes",
    NeurIPS 2023.
"""

from contextlib import contextmanager
import logging
import math
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.data.embodiment_tags import EmbodimentTag

from .gr00t_policy import Gr00tPolicy


logger = logging.getLogger(__name__)

NOISE_METHODS = ("sdedit", "interpolation", "scaled")


class RestartSamplingPolicy(Gr00tPolicy):
    """GR00T N1.6 policy with Restart Sampling.

    Runs a 3-phase denoising process: standard ODE → restart cycles
    (re-noise + re-denoise) → standard ODE to completion.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        # Restart config
        t_restart: float = 0.6,
        t_back: float = 0.3,
        restart_K: int = 1,
        steps_restart: int = 3,
        total_steps: int = 10,
        noise_method: str = "sdedit",
        verbose: bool = False,
    ):
        """Initialize RestartSamplingPolicy.

        Args:
            embodiment_tag: Robot embodiment type.
            model_path: Path to pretrained model checkpoint.
            device: Device for inference (e.g. 'cuda:0').
            strict: Whether to enforce strict input validation.
            t_restart: Restart point in [0, 1] (0=noise, 1=data).
            t_back: Re-noise target (must be < t_restart).
            restart_K: Number of restart cycles. 0 = vanilla ODE.
            steps_restart: ODE steps per restart cycle.
            total_steps: Total NFE budget.
            noise_method: Re-noise method: "sdedit", "interpolation", or "scaled".
            verbose: Log per-step statistics.
        """
        super().__init__(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=strict,
        )
        assert 0.0 < t_restart < 1.0, f"t_restart must be in (0, 1), got {t_restart}"
        assert 0.0 <= t_back < t_restart, (
            f"t_back must be in [0, t_restart), got {t_back} >= {t_restart}"
        )
        assert restart_K >= 0, f"restart_K must be >= 0, got {restart_K}"
        assert noise_method in NOISE_METHODS, (
            f"noise_method must be one of {NOISE_METHODS}, got {noise_method}"
        )

        self.t_restart = t_restart
        self.t_back = t_back
        self.restart_K = restart_K
        self.steps_restart = steps_restart
        self.total_steps = total_steps
        self.noise_method = noise_method
        self.verbose = verbose

        self._action_head = self._find_action_head()
        self._original_num_inference_timesteps = self._action_head.num_inference_timesteps

        # Compute step allocation
        self._steps_phase1 = max(1, round(self.t_restart * self.total_steps))
        self._steps_phase3 = max(
            1, self.total_steps - self._steps_phase1 - self.restart_K * self.steps_restart
        )
        self._actual_nfe = (
            self._steps_phase1 + self.restart_K * self.steps_restart + self._steps_phase3
        )

        logger.info(
            "RestartSampling initialized: t_restart=%.2f, t_back=%.2f, K=%d, "
            "steps_restart=%d, total_steps=%d, noise_method=%s",
            self.t_restart,
            self.t_back,
            self.restart_K,
            self.steps_restart,
            self.total_steps,
            self.noise_method,
        )
        logger.info(
            "  Step allocation: phase1=%d, restart=%d×%d, phase3=%d (actual NFE=%d)",
            self._steps_phase1,
            self.restart_K,
            self.steps_restart,
            self._steps_phase3,
            self._actual_nfe,
        )

    # ------------------------------------------------------------------
    # Internals
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

    def _restart_forward(
        self,
        actions: torch.Tensor,
        t_current: float,
        t_target: float,
    ) -> torch.Tensor:
        """Re-noise actions from t_current back to t_target.

        Args:
            actions: Current denoised actions at t_current.
            t_current: Current time (e.g. t_restart).
            t_target: Target time to re-noise to (e.g. t_back, must be < t_current).

        Returns:
            Re-noised actions at t_target.
        """
        z = torch.randn_like(actions)

        if self.noise_method == "sdedit":
            # SDEdit-style: add noise proportional to time difference
            actions_back = actions + (t_current - t_target) * z
        elif self.noise_method == "interpolation":
            # Flow matching forward process interpolation
            actions_back = (1.0 - t_target) * z + t_target * actions
        elif self.noise_method == "scaled":
            # Scaled noise injection preserving signal-to-noise
            alpha = t_target / t_current
            sigma = math.sqrt(1.0 - alpha**2) * (1.0 - t_target)
            actions_back = alpha * actions + sigma * z
        else:
            raise ValueError(f"Unknown noise_method: {self.noise_method}")

        if self.verbose:
            noise_norm = (actions_back - actions).norm(dim=-1).mean().item()
            action_norm = actions.norm(dim=-1).mean().item()
            logger.info(
                "Restart re-noise (%s): t=%.3f→%.3f, noise_norm=%.4f, action_norm=%.4f",
                self.noise_method,
                t_current,
                t_target,
                noise_norm,
                action_norm,
            )

        return actions_back

    @contextmanager
    def _restart_inference_ctx(self):
        """Context manager that monkey-patches model.get_action for restart sampling.

        Inside the patched get_action:
        1. Prepares inputs and runs backbone once.
        2. Encodes features (state + VL embeddings).
        3. Runs 3-phase denoising: ODE → restart cycles → ODE.
        """
        original_get_action = self.model.get_action
        action_head = self._action_head
        policy_self = self

        def patched_get_action(inputs: dict) -> BatchFeature:
            # Step 1: Prepare inputs
            backbone_inputs, action_inputs = policy_self.model.prepare_input(inputs)

            # Step 2: Run backbone
            backbone_output = policy_self.model.backbone(backbone_inputs)

            # Step 3: Encode features
            features = action_head._encode_features(backbone_output, action_inputs)
            vl_embeds = features.backbone_features
            state_features = features.state_features
            embodiment_id = action_inputs.embodiment_id

            # Step 4: Initialize noise
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

            num_timestep_buckets = action_head.num_timestep_buckets
            steps_phase1 = policy_self._steps_phase1
            steps_phase3 = policy_self._steps_phase3
            K = policy_self.restart_K
            steps_restart = policy_self.steps_restart
            t_restart = policy_self.t_restart
            t_back = policy_self.t_back

            nfe_count = 0

            # ---- Phase 1: Standard ODE t=0 → t_restart ----
            dt1 = t_restart / steps_phase1
            for i in range(steps_phase1):
                t_cont = i * dt1
                t_discretized = int(t_cont * num_timestep_buckets)
                v = policy_self._compute_velocity(
                    action_head,
                    actions,
                    t_discretized,
                    vl_embeds,
                    state_features,
                    embodiment_id,
                    backbone_output,
                )
                actions = actions + dt1 * v
                nfe_count += 1

            if policy_self.verbose:
                logger.info(
                    "Phase 1 done: %d steps, t=0→%.3f, action_norm=%.4f",
                    steps_phase1,
                    t_restart,
                    actions.norm(dim=-1).mean().item(),
                )

            # ---- Phase 2: K restart cycles ----
            for k in range(K):
                # 2a. Re-noise: t_restart → t_back
                actions = policy_self._restart_forward(actions, t_restart, t_back)

                # 2b. Re-denoise ODE: t_back → t_restart
                dt_r = (t_restart - t_back) / steps_restart
                for i in range(steps_restart):
                    t_cont = t_back + i * dt_r
                    t_discretized = int(t_cont * num_timestep_buckets)
                    v = policy_self._compute_velocity(
                        action_head,
                        actions,
                        t_discretized,
                        vl_embeds,
                        state_features,
                        embodiment_id,
                        backbone_output,
                    )
                    actions = actions + dt_r * v
                    nfe_count += 1

                if policy_self.verbose:
                    logger.info(
                        "Restart cycle %d/%d done: %d steps, t=%.3f→%.3f→%.3f, action_norm=%.4f",
                        k + 1,
                        K,
                        steps_restart,
                        t_restart,
                        t_back,
                        t_restart,
                        actions.norm(dim=-1).mean().item(),
                    )

            # ---- Phase 3: Standard ODE t_restart → 1.0 ----
            dt3 = (1.0 - t_restart) / steps_phase3
            for i in range(steps_phase3):
                t_cont = t_restart + i * dt3
                t_discretized = int(t_cont * num_timestep_buckets)
                v = policy_self._compute_velocity(
                    action_head,
                    actions,
                    t_discretized,
                    vl_embeds,
                    state_features,
                    embodiment_id,
                    backbone_output,
                )
                actions = actions + dt3 * v
                nfe_count += 1

            if policy_self.verbose:
                logger.info(
                    "Phase 3 done: %d steps, t=%.3f→1.0, action_norm=%.4f, total_NFE=%d",
                    steps_phase3,
                    t_restart,
                    actions.norm(dim=-1).mean().item(),
                    nfe_count,
                )

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
        """Compute actions with Restart Sampling."""
        with self._restart_inference_ctx():
            return super()._get_action(observation, options)

    @property
    def nfe(self) -> int:
        """Return the actual number of function evaluations per inference call."""
        return self._actual_nfe
