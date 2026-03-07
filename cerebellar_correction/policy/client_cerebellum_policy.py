"""Client-side policy with local cerebellar correction (Option 3).

Wraps a PolicyClient that talks to a GR00T+IntentPassthrough server.
The server returns action chunks (B, 8, D) + intent vectors (B, 2048).

This policy:
1. Calls server once per 8-step chunk to get actions + intent
2. Loads cerebellum (DINOv2 22M + correction net) locally on client GPU
3. Returns 1 corrected action per get_action() call using fresh observation
4. Use with MultiStepWrapper(n_action_steps=1)

Per-step latency: ~3ms (DINOv2 ViT-S/14 forward + correction net).
"""

import logging
from pathlib import Path
from typing import Any

from gr00t.policy.policy import BasePolicy
from gr00t.policy.server_client import PolicyClient
import numpy as np
import torch

from cerebellar_correction.models.cerebellum import CerebellumConfig, PatchCerebellumModule


log = logging.getLogger(__name__)


class ClientSideCerebellumPolicy(BasePolicy):
    """Client-side policy that applies per-step cerebellar correction locally.

    The server runs GR00T and returns raw action chunks + intent vectors.
    This client caches the 8-step chunk and returns 1 corrected action per call,
    applying correction using the current observation at each step.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        cerebellum_ckpt: str = "checkpoints/cerebellum_intent",
        device: str = "cuda",
        correction_alpha: float = 1.0,
        max_correction: float = 0.15,
        chunk_size: int = 8,
        image_key: str = "video.image_0",
        state_keys: tuple[str, ...] = (
            "state.x",
            "state.y",
            "state.z",
            "state.roll",
            "state.pitch",
            "state.yaw",
            "state.pad",
            "state.gripper",
        ),
        timeout_ms: int = 30000,
        strict: bool = False,
    ):
        super().__init__(strict=strict)
        self.device = device
        self.correction_alpha = correction_alpha
        self.chunk_size = chunk_size
        self.image_key = image_key
        self.state_keys = state_keys

        # Remote GR00T server
        self.client = PolicyClient(host=host, port=port, timeout_ms=timeout_ms, strict=False)

        # Local cerebellum
        config = CerebellumConfig(max_correction=max_correction)
        self.cerebellum = PatchCerebellumModule(config)

        ckpt_dir = Path(cerebellum_ckpt)
        assembled = ckpt_dir / "cerebellum_assembled.pt"
        if assembled.exists():
            self.cerebellum.load_checkpoint(str(assembled), device="cpu")
            log.info("Loaded assembled cerebellum from %s", assembled)
        else:
            self._load_components(ckpt_dir)

        self.cerebellum.to(device).eval()
        total_params = sum(p.numel() for p in self.cerebellum.parameters())
        log.info(
            "Client-side cerebellum: %d params on %s, alpha=%.2f, chunk=%d",
            total_params,
            device,
            correction_alpha,
            chunk_size,
        )

        # Chunk cache
        self._cached_action: dict[str, Any] | None = None
        self._cached_info: dict[str, Any] = {}
        self._action_key: str | None = None
        self._chunk_step = 0

    def _load_components(self, ckpt_dir: Path):
        """Load cerebellum from individual phase checkpoints."""
        phase1 = ckpt_dir / "phase1"
        if not phase1.exists():
            phase1 = ckpt_dir

        for name, submod in [
            ("transition_vit_best.pt", self.cerebellum.transition_vit),
            ("proprio_forward_best.pt", self.cerebellum.proprio_forward),
        ]:
            path = phase1 / name
            if path.exists():
                submod.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

        phase2 = ckpt_dir / "phase2"
        if not phase2.exists():
            phase2 = ckpt_dir
        for name, submod in [
            ("error_pooling_best.pt", self.cerebellum.error_pooling),
            ("correction_net_best.pt", self.cerebellum.correction_net),
        ]:
            path = phase2 / name
            if path.exists():
                submod.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        log.info("Loaded cerebellum components from %s", ckpt_dir)

    def _extract_image(self, observation: dict[str, Any]) -> torch.Tensor:
        """Extract current image as (1, 3, H, W) float32 [0,1]."""
        img = observation[self.image_key]  # (B, T, H, W, C) uint8
        frame = img[0, -1]  # (H, W, C) last timestep
        t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        return t.to(self.device)

    def _extract_proprio(self, observation: dict[str, Any]) -> torch.Tensor:
        """Extract (1, proprio_dim) float32 proprio."""
        parts = []
        for key in self.state_keys:
            val = observation[key]  # (B, T, D)
            parts.append(val[0, -1])
        proprio = np.concatenate(parts)
        return torch.from_numpy(proprio).unsqueeze(0).float().to(self.device)

    def _find_action_key(self, action: dict[str, Any]) -> str:
        """Find the main action key in the action dict."""
        if self._action_key is not None:
            return self._action_key
        for candidate in ["joint_position", "action.joint_position"]:
            if candidate in action:
                self._action_key = candidate
                return candidate
        self._action_key = next(iter(action))
        return self._action_key

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get a single corrected action step.

        Calls server every chunk_size steps to get a new action chunk + intent.
        Returns 1 corrected action per call using the current observation.
        """
        # Fetch new chunk from server if needed
        if self._cached_action is None or self._chunk_step >= self.chunk_size:
            action, info = self.client._get_action(observation, options)
            self._cached_action = action
            self._cached_info = info

            # Extract intent tokens from server response (full sequence for self-attention)
            tokens_np = info.get("intent_tokens")
            mask_np = info.get("intent_attention_mask")
            if tokens_np is not None and mask_np is not None:
                intent_tokens = torch.from_numpy(tokens_np).float().to(self.device)
                intent_mask = torch.from_numpy(mask_np).to(self.device)
                self.cerebellum.on_new_chunk(intent_tokens, intent_mask)
            else:
                log.warning("No intent_tokens in server response — correction may be degraded")

            self._chunk_step = 0

        # Get current step's action from cached chunk
        action_key = self._find_action_key(self._cached_action)
        action_chunk = self._cached_action[action_key]  # (B, T, D)
        B, T, D = action_chunk.shape
        step = min(self._chunk_step, T - 1)

        # Extract single step: (B, 1, D)
        step_action = action_chunk[:, step : step + 1, :].copy()

        # Apply cerebellar correction
        image = self._extract_image(observation)
        proprio = self._extract_proprio(observation)
        planned = torch.from_numpy(step_action[0, 0, :7]).unsqueeze(0).float().to(self.device)

        corrected = self.cerebellum.correct(
            image_current=image,
            proprio_current=proprio,
            action_planned=planned,
            chunk_step=step,
        )

        # Blend with alpha
        if self.correction_alpha < 1.0:
            corrected = planned + self.correction_alpha * (corrected - planned)

        step_action[0, 0, :7] = corrected.cpu().numpy()[0]
        self._chunk_step += 1

        # Return single-step action (B, 1, D)
        single_action = {action_key: step_action}
        return single_action, self._cached_info

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self._cached_action = None
        self._cached_info = {}
        self._chunk_step = 0
        self.cerebellum.reset()
        return self.client.reset(options)

    def get_modality_config(self):
        return self.client.get_modality_config()

    def check_observation(self, observation: dict[str, Any]) -> None:
        pass  # Client-side, no strict checking

    def check_action(self, action: dict[str, Any]) -> None:
        pass
