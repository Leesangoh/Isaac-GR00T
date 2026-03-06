"""Server-side policy wrapper that extracts intent vectors without applying correction.

For Option 3 (client-side cerebellum): the server runs GR00T and returns
raw action chunks + intent vectors. The client loads the cerebellum locally
and applies per-step correction with fresh observations.
"""

import logging
from typing import Any

from gr00t.policy.policy import BasePolicy, PolicyWrapper
import torch

from cerebellar_correction.models.intent_extractor import IntentExtractor


log = logging.getLogger(__name__)


class IntentPassthroughPolicyWrapper(PolicyWrapper):
    """Wraps GR00T to extract intent vectors and pass them in info dict.

    No correction is applied — the raw VLA actions are returned alongside
    the intent vector so the client can apply correction locally.
    """

    def __init__(
        self,
        policy: BasePolicy,
        groot_model: torch.nn.Module,
        device: str = "cuda",
    ):
        super().__init__(policy, strict=policy.strict)
        self.device = device
        self.intent_extractor = IntentExtractor(groot_model)
        log.info("IntentPassthroughPolicyWrapper: hook attached, intent will be in info dict")

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get GR00T action and extract intent vector into info dict."""
        action, info = self.policy._get_action(observation, options)

        # Pass full token sequence for client-side self-attention forward model
        intent_tokens, intent_mask = self.intent_extractor.get_intent_tokens()
        info["intent_tokens"] = intent_tokens.half().cpu().numpy()  # (B, 128, 2048) fp16
        info["intent_attention_mask"] = intent_mask.cpu().numpy()  # (B, 128) bool

        return action, info

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.policy.reset(options)

    def get_modality_config(self):
        return self.policy.get_modality_config()
