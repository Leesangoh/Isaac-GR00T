"""MultiStepWrapper with intent-conditioned cerebellar correction at each sub-step.

Subclasses the standard MultiStepWrapper to intercept each action within
a chunk and apply real-time correction based on intent-aware prediction errors.

This is the correct way to implement per-step correction as specified in the
design doc: each step gets a fresh observation, DINOv2 encoding, prediction
error computation, and correction — not just step 0.

Requires the cerebellum (DINOv2 + correction net) to be available on the
eval machine. The intent is extracted from GR00T's forward pass and stays
fixed within a chunk.
"""

import logging

from gr00t.eval.sim.wrapper.multistep_wrapper import (
    MultiStepWrapper,
    aggregate,
    compress_dict_list,
    dict_take_last_n,
)
import gymnasium as gym
import numpy as np
import torch

from cerebellar_correction.models.cerebellum import PatchCerebellumModule
from cerebellar_correction.models.intent_extractor import IntentExtractor


log = logging.getLogger(__name__)


class PatchCerebellumMultiStepWrapper(MultiStepWrapper):
    """MultiStepWrapper that applies intent-conditioned cerebellar correction at each sub-step.

    During action chunk execution, each individual action is corrected
    using the PatchCerebellumModule. The intent vector is extracted once
    per chunk (from GR00T's forward pass) and reused for all steps.
    """

    def __init__(
        self,
        env,
        cerebellum: PatchCerebellumModule,
        intent_extractor: IntentExtractor,
        video_delta_indices,
        state_delta_indices,
        n_action_steps,
        max_episode_steps=None,
        reward_agg_method="max",
        terminate_on_success=False,
        correction_alpha: float = 1.0,
        device: str = "cuda",
        state_keys: list[str] | None = None,
        action_keys: list[str] | None = None,
        image_key: str | None = None,
    ):
        super().__init__(
            env=env,
            video_delta_indices=video_delta_indices,
            state_delta_indices=state_delta_indices,
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            reward_agg_method=reward_agg_method,
            terminate_on_success=terminate_on_success,
        )
        self.cerebellum = cerebellum
        self.intent_extractor = intent_extractor
        self.correction_alpha = correction_alpha
        self.device = device

        self._state_keys = state_keys
        self._action_keys = action_keys
        self._image_key = image_key

        self.correction_magnitudes = []

    def _discover_state_keys(self, observation: dict) -> list[str]:
        if self._state_keys is not None:
            return self._state_keys
        keys = sorted(k for k in observation if k.startswith("state."))
        self._state_keys = keys
        log.info("Auto-discovered state keys: %s", keys)
        return keys

    def _discover_action_keys(self, action_dict: dict) -> list[str]:
        if self._action_keys is not None:
            return self._action_keys
        keys = sorted(action_dict.keys())
        self._action_keys = keys
        log.info("Auto-discovered action keys: %s", keys)
        return keys

    def _discover_image_key(self, observation: dict) -> str:
        if self._image_key is not None:
            return self._image_key
        for k in observation:
            if k.startswith("video"):
                self._image_key = k
                log.info("Auto-discovered image key: %s", k)
                return k
        self._image_key = ""
        return ""

    def _obs_to_image_tensor(self, observation: dict) -> torch.Tensor:
        key = self._discover_image_key(observation)
        if not key or key not in observation:
            return torch.zeros(1, 3, 98, 98, device=self.device)
        img = observation[key]  # (H, W, 3) uint8
        t = torch.from_numpy(img.astype(np.float32)).div_(255.0)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _obs_to_proprio_tensor(self, observation: dict) -> torch.Tensor:
        keys = self._discover_state_keys(observation)
        values = []
        for key in keys:
            val = observation.get(key, 0.0)
            if isinstance(val, np.ndarray):
                values.extend(val.flatten().tolist())
            elif isinstance(val, (list, tuple)):
                values.extend([float(v) for v in val])
            else:
                values.append(float(val))
        return torch.tensor([values], dtype=torch.float32, device=self.device)

    def _action_dict_to_tensor(self, act: dict) -> torch.Tensor:
        keys = self._discover_action_keys(act)
        values = []
        for key in keys:
            val = act[key]
            if isinstance(val, np.ndarray):
                values.extend(val.flatten().tolist())
            else:
                values.append(float(val))
        return torch.tensor([values], dtype=torch.float32, device=self.device)

    def _tensor_to_action_dict(self, tensor: torch.Tensor, original: dict) -> dict:
        keys = self._discover_action_keys(original)
        flat = tensor.cpu().numpy().flatten()
        result = {}
        idx = 0
        for key in keys:
            orig_val = original[key]
            if isinstance(orig_val, np.ndarray):
                n = orig_val.size
                result[key] = flat[idx : idx + n].astype(orig_val.dtype).reshape(orig_val.shape)
                idx += n
            else:
                result[key] = float(flat[idx])
                idx += 1
        return result

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.cerebellum.reset()
        self.correction_magnitudes = []
        return obs, info

    def step(self, action):
        """Execute action chunk with per-step cerebellar correction.

        For each step in the chunk:
        1. Get current observation from env
        2. Extract intent (once per chunk, from GR00T's last forward pass)
        3. Apply cerebellar correction using intent-aware prediction error
        4. Execute corrected action in env
        """
        states = []
        rewards = []
        dones = []

        # Extract intent tokens from GR00T's last forward pass (stays fixed within chunk)
        intent_tokens, intent_mask = self.intent_extractor.get_intent_tokens()
        self.cerebellum.on_new_chunk(intent_tokens, intent_mask)

        for step in range(self.n_action_steps):
            act = {}
            for key, value in action.items():
                act[key] = value[step, :]

            if len(self.done) > 0 and self.done[-1]:
                break

            # Apply cerebellar correction using latest observation
            if len(self.obs) > 0:
                current_obs = self.obs[-1]

                image_tensor = self._obs_to_image_tensor(current_obs)
                proprio_tensor = self._obs_to_proprio_tensor(current_obs)
                action_tensor = self._action_dict_to_tensor(act)

                corrected_tensor = self.cerebellum.correct(
                    image_current=image_tensor,
                    proprio_current=proprio_tensor,
                    action_planned=action_tensor[:, :7],  # first 7 dims
                    chunk_step=step,
                )

                # Blend with alpha
                original_7 = action_tensor[:, :7]
                blended = original_7 + self.correction_alpha * (corrected_tensor - original_7)

                # Write back corrected values
                full_corrected = action_tensor.clone()
                full_corrected[:, :7] = blended

                delta = (full_corrected - action_tensor).abs().mean().item()
                self.correction_magnitudes.append(delta)
                act = self._tensor_to_action_dict(full_corrected, act)

            # Execute corrected action
            observation, reward, done, truncated, info = gym.Wrapper.step(self, act)

            env_state = {"states": [], "model": []}
            states.append(env_state["states"])
            rewards.append(reward)
            dones.append(done)
            self.obs.append(observation)
            self.reward.append(reward)

            if (self.max_episode_steps is not None) and (
                len(self.reward) >= self.max_episode_steps
            ):
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.video_delta_indices, self.state_delta_indices)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, "max")
        info = dict_take_last_n(self.info, self.n_action_steps)
        states = np.array(states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        info["states"] = states
        info["rewards"] = rewards
        info["model"] = env_state["model"]
        info["actions"] = action
        info["dones"] = dones

        if "intermediate_signals" in info:
            info["intermediate_signals"] = compress_dict_list(list(info["intermediate_signals"]))

        if self.correction_magnitudes:
            info["mean_correction_magnitude"] = np.mean(
                self.correction_magnitudes[-self.n_action_steps :]
            )

        if self.terminate_on_success and any(info["success"]):
            done = True

        return observation, reward, done, truncated, info
