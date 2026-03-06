"""Policy wrapper that applies intent-conditioned cerebellar correction.

Wraps Gr00tSimPolicyWrapper. On each get_action call:
1. GR00T forward pass -> action chunk + intent vector (via hook)
2. DINOv2 -> visual features for cerebellar correction
3. Per-step correction using intent-aware prediction error

The client sees the same interface — no changes needed.
"""

import logging
from pathlib import Path
from typing import Any

from gr00t.policy.policy import BasePolicy, PolicyWrapper
import numpy as np
import torch

from cerebellar_correction.models.cerebellum import CerebellumConfig, IntentCerebellumModule
from cerebellar_correction.models.intent_extractor import IntentExtractor


log = logging.getLogger(__name__)


class IntentCerebellumPolicyWrapper(PolicyWrapper):
    """Wraps Gr00tSimPolicyWrapper with intent-conditioned cerebellar correction.

    On each get_action:
    1. Gets GR00T's planned action chunk (triggers backbone hook -> intent)
    2. Extracts intent via IntentExtractor
    3. Applies cerebellar correction to the first action step
    4. Returns modified action chunk
    """

    def __init__(
        self,
        policy: BasePolicy,
        groot_model: torch.nn.Module,
        cerebellum_ckpt: str,
        device: str = "cuda",
        correction_alpha: float = 1.0,
        max_correction: float = 0.15,
        image_key: str = "image_0",
        state_keys: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"),
    ):
        super().__init__(policy, strict=policy.strict)
        self.device = device
        self.image_key = image_key
        self.state_keys = state_keys
        self.correction_alpha = correction_alpha

        # Intent extractor (hooks GR00T backbone)
        self.intent_extractor = IntentExtractor(groot_model)
        log.info("Intent extractor hook attached to GR00T backbone")

        # Load cerebellum
        ckpt_dir = Path(cerebellum_ckpt)
        config = CerebellumConfig(max_correction=max_correction)
        self.cerebellum = IntentCerebellumModule(config)

        if (ckpt_dir / "cerebellum_assembled.pt").exists():
            self.cerebellum.load_checkpoint(str(ckpt_dir / "cerebellum_assembled.pt"), device="cpu")
            log.info("Loaded assembled cerebellum from %s", ckpt_dir)
        else:
            self._load_components(ckpt_dir)

        self.cerebellum.to(device).eval()

        total_params = sum(p.numel() for p in self.cerebellum.parameters())
        log.info("Cerebellum loaded: %d params, alpha=%.2f", total_params, correction_alpha)

    def _load_components(self, ckpt_dir: Path):
        """Load cerebellum from individual phase checkpoints."""
        phase1 = ckpt_dir / "phase1"
        if not phase1.exists():
            phase1 = ckpt_dir

        fm_path = phase1 / "forward_model_best.pt"
        if fm_path.exists():
            self.cerebellum.forward_model.load_state_dict(
                torch.load(fm_path, map_location="cpu", weights_only=True)
            )

        pf_path = phase1 / "proprio_forward_best.pt"
        if pf_path.exists():
            self.cerebellum.proprio_forward.load_state_dict(
                torch.load(pf_path, map_location="cpu", weights_only=True)
            )

        phase2 = ckpt_dir / "phase2"
        if not phase2.exists():
            phase2 = ckpt_dir

        cn_path = phase2 / "correction_net_best.pt"
        if cn_path.exists():
            self.cerebellum.correction_net.load_state_dict(
                torch.load(cn_path, map_location="cpu", weights_only=True)
            )

        log.info("Loaded cerebellum components from %s", ckpt_dir)

    def _extract_image(self, observation: dict[str, Any]) -> torch.Tensor:
        """Extract current image as (1, 3, H, W) float32 [0,1]."""
        key = f"video.{self.image_key}"
        img = observation[key]  # (B, T, H, W, C) uint8
        frame = img[0, -1]  # (H, W, C) last timestep
        t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        return t.to(self.device)

    def _extract_proprio(self, observation: dict[str, Any]) -> torch.Tensor:
        """Extract (1, proprio_dim) float32 proprio."""
        parts = []
        for key in self.state_keys:
            flat_key = f"state.{key}"
            val = observation[flat_key]  # (B, T, D)
            parts.append(val[0, -1])
        proprio = np.concatenate(parts)
        return torch.from_numpy(proprio).unsqueeze(0).float().to(self.device)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get GR00T action with intent-conditioned cerebellar correction."""
        # GR00T forward pass (triggers intent hook)
        action, info = self.policy._get_action(observation, options)

        # Extract intent tokens from the hook (full sequence for self-attention)
        intent_tokens, intent_mask = self.intent_extractor.get_intent_tokens()
        self.cerebellum.on_new_chunk(intent_tokens, intent_mask)

        # Extract current observation for cerebellum
        image = self._extract_image(observation)
        proprio = self._extract_proprio(observation)

        # Find action array
        action_arr = None
        action_key_used = None
        for candidate in ["joint_position", "action.joint_position"]:
            if candidate in action:
                action_arr = action[candidate]
                action_key_used = candidate
                break
        if action_arr is None:
            action_key_used = next(iter(action))
            action_arr = action[action_key_used]

        B, T, D = action_arr.shape

        # Correct first action step
        first_action = torch.from_numpy(action_arr[0, 0:1, :7]).float().to(self.device)

        corrected = self.cerebellum.correct(
            image_current=image,
            proprio_current=proprio,
            action_planned=first_action,
        )

        # Blend with alpha
        if self.correction_alpha < 1.0:
            corrected = first_action + self.correction_alpha * (corrected - first_action)

        action_arr[0, 0, :7] = corrected.cpu().numpy()[0]
        action[action_key_used] = action_arr

        return action, info

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.cerebellum.reset()
        return self.policy.reset(options)

    def get_modality_config(self):
        return self.policy.get_modality_config()
