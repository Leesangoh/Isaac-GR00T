"""DepthMem Policy: Gr00t policy with depth channel and temporal attention.

Extends Gr00tPolicy to:
1. Maintain an RGB frame buffer for Video Depth Anything
2. Generate temporally consistent depth maps at each step
3. Pass RGBD (4ch) frames through SigLIP2 with temporal attention
4. Manage temporal KV cache for efficient inference
"""

from collections import deque
import logging
import os
import sys
from typing import Any

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData

from .gr00t_policy import Gr00tPolicy, _rec_to_dtype


logger = logging.getLogger(__name__)


class DepthMemPolicy(Gr00tPolicy):
    """Policy with depth channel and temporal attention support.

    On top of Gr00tPolicy, this class:
    - Maintains an RGB frame buffer (deque of T frames)
    - Runs Video Depth Anything Small on the buffer each step
    - Concatenates depth as 4th channel
    - Passes num_temporal_frames through the VLM pipeline
    - Manages temporal KV cache for SigLIP2 temporal attention layers
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        num_temporal_frames: int = 6,
        depth_model_size: str = "small",
        depth_resolution: int = 224,
        use_temporal_kv_cache: bool = True,
        compile_depth_model: bool = True,
        save_attention_map: bool = False,
        attention_map_dir: str = "./attention_maps",
    ):
        """Initialize DepthMem policy.

        Args:
            embodiment_tag: Robot embodiment tag.
            model_path: Path to pretrained model checkpoint.
            device: Device for inference.
            num_temporal_frames: T - number of frames to maintain in buffer.
            depth_model_size: Video Depth Anything model size ('small', 'base', 'large').
            depth_resolution: Input resolution for depth model.
            use_temporal_kv_cache: Whether to cache temporal attention KVs.
            compile_depth_model: Whether to torch.compile the depth model.
        """
        super().__init__(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=strict,
            save_attention_map=save_attention_map,
            attention_map_dir=attention_map_dir,
        )

        self.num_temporal_frames = num_temporal_frames
        self.depth_resolution = depth_resolution
        self.use_temporal_kv_cache = use_temporal_kv_cache

        # RGB frame buffer: stores recent frames for depth estimation
        self.rgb_buffer = deque(maxlen=num_temporal_frames)

        # Temporal KV cache for SigLIP2 temporal attention
        self.temporal_kv_cache = {} if use_temporal_kv_cache else None

        # Load Video Depth Anything
        self.depth_model = self._load_depth_model(depth_model_size, device, compile_depth_model)

        logger.info(
            f"DepthMem policy initialized: T={num_temporal_frames}, "
            f"depth_model={depth_model_size}, resolution={depth_resolution}"
        )

    def _load_depth_model(self, model_size: str, device, compile_model: bool):
        """Load Video Depth Anything model."""
        vda_path = os.path.join(os.path.dirname(__file__), "..", "..", "Video-Depth-Anything")
        sys.path.insert(0, vda_path)

        from video_depth_anything.video_depth import VideoDepthAnything

        model_configs = {
            "small": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "base": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "large": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }
        encoder_map = {"small": "vits", "base": "vitb", "large": "vitl"}

        config = model_configs[model_size]
        model = VideoDepthAnything(**config)

        ckpt_path = os.path.join(
            vda_path, "checkpoints", f"video_depth_anything_{encoder_map[model_size]}.pth"
        )
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

        if isinstance(device, int):
            device = f"cuda:{device}"
        model = model.to(device).eval()

        if compile_model:
            try:
                model = torch.compile(model, mode="max-autotune")
                logger.info("Depth model compiled with max-autotune")
            except Exception as e:
                logger.warning(f"torch.compile failed for depth model: {e}")

        return model

    def _generate_depth_maps(self, rgb_frames: np.ndarray) -> np.ndarray:
        """Generate temporally consistent depth maps for a batch of frames.

        Args:
            rgb_frames: [T, H, W, 3] uint8 RGB frames.

        Returns:
            depth_maps: [T, H, W] float32 normalized depth maps in [0, 1].
        """
        with torch.no_grad():
            depths, _ = self.depth_model.infer_video_depth(
                rgb_frames,
                target_fps=-1,
                input_size=self.depth_resolution,
                device=str(self.model.device),
            )

        # Normalize to [0, 1]
        d_min, d_max = depths.min(), depths.max()
        if d_max - d_min > 1e-6:
            depths = (depths - d_min) / (d_max - d_min)
        else:
            depths = np.zeros_like(depths)

        return depths.astype(np.float32)

    def reset(self):
        """Reset buffers for a new episode."""
        self.rgb_buffer.clear()
        if self.temporal_kv_cache is not None:
            self.temporal_kv_cache.clear()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute actions with temporal KV cache support.

        When KV cache is enabled and warm (non-empty), only the current frame
        is encoded through SigLIP2 — past frames' KVs are retrieved from cache.
        This avoids re-encoding T frames every step (O(T) → O(1) per step).

        When cache is disabled or cold (first step), falls back to full
        T-frame encoding via the parent class.
        """
        if not self.use_temporal_kv_cache or not self.temporal_kv_cache:
            # First step or cache disabled: encode all T frames (parent behavior)
            result = super()._get_action(observation, options)
            # After first full encoding, populate cache if enabled
            # The cache gets populated inside the model's temporal_attention layers
            # via the temporal_kv_cache dict reference — but only if we inject it.
            # For the first step, we do a full re-run with cache injection below.
            if self.use_temporal_kv_cache and not self.temporal_kv_cache:
                # Re-run with cache dict injected to warm it up
                self._warm_up_cache(observation)
            return result

        # KV cache is warm: encode only current frame (T=1)
        # Extract observation data for attention map visualization
        if self.save_attention_map:
            self._current_obs_images = [
                observation["video"][k][0, -1] for k in self.modality_configs["video"].modality_keys
            ]
            self._current_instruction = observation["language"][self.language_key][0][0]

        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []
        states = []

        for obs in unbatched_observations:
            # Create VLAStepData with only the current frame + its depth
            vla_step_data = self._to_vla_step_data_current_only(obs)
            states.append(vla_step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Inject temporal KV cache and set T=1 (current frame only)
        collated_inputs["temporal_kv_cache"] = self.temporal_kv_cache
        collated_inputs["num_temporal_frames"] = 1

        with self._attention_capture_ctx() as attn_ctx:
            with torch.inference_mode():
                model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        if attn_ctx is not None:
            self._flush_attention_data(attn_ctx)

        # Trim cache to keep only last (T-1) frames' KVs
        self._trim_kv_cache()

        # Decode actions
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, {}

    def _warm_up_cache(self, observation: dict[str, Any]):
        """Run a forward pass with cache injection to populate KV cache.

        Called once after the first step to warm up the cache. Subsequent steps
        will use cached KVs and only encode the current frame.
        """
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Inject the (empty) cache dict — temporal_attention will populate it
        collated_inputs["temporal_kv_cache"] = self.temporal_kv_cache

        with torch.inference_mode():
            self.model.get_action(**collated_inputs)

        logger.info(f"KV cache warmed up with {len(self.temporal_kv_cache)} layer entries")

    def _trim_kv_cache(self):
        """Trim KV cache to keep only the last (T-1) frames' worth of KVs.

        Each cache entry has shape [B*N, T_cached, D]. We keep only the last
        (num_temporal_frames - 1) time steps, so the next step's current frame
        makes it T total.
        """
        max_cached = self.num_temporal_frames - 1
        for cache_key in self.temporal_kv_cache:
            k, v = self.temporal_kv_cache[cache_key]
            if k.shape[1] > max_cached:
                self.temporal_kv_cache[cache_key] = (
                    k[:, -max_cached:].contiguous(),
                    v[:, -max_cached:].contiguous(),
                )

    def _to_vla_step_data_current_only(self, observation: dict[str, Any]) -> VLAStepData:
        """Create VLAStepData with only the current frame and its depth.

        Used when KV cache is warm — past frames are already cached, so we
        only need to encode the current frame through SigLIP2.
        """
        video_keys = list(observation["video"].keys())
        primary_key = video_keys[0] if video_keys else None

        if primary_key is not None:
            current_frames = observation["video"][primary_key]
            if current_frames.ndim == 4:
                for t in range(current_frames.shape[0]):
                    self.rgb_buffer.append(current_frames[t])
            elif current_frames.ndim == 3:
                self.rgb_buffer.append(current_frames)

        step_data = VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

        # Generate depth for current frame only (but use full buffer for temporal consistency)
        if len(self.rgb_buffer) > 0:
            buffer_frames = np.stack(list(self.rgb_buffer))
            depth_maps = self._generate_depth_maps(buffer_frames)
            # Only pass the current (last) frame's depth
            step_data.depth_maps = depth_maps[-1:]  # [1, H, W]
            step_data.num_temporal_frames = 1

        return step_data

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert observation to VLAStepData, adding depth maps.

        Override to inject depth information into the processing pipeline.
        """
        # Get the RGB images from the video observation
        video_keys = list(observation["video"].keys())
        primary_key = video_keys[0] if video_keys else None

        if primary_key is not None:
            # Get current frame: observation video is [T, H, W, C]
            current_frames = observation["video"][primary_key]
            # Add current frame(s) to buffer
            if current_frames.ndim == 4:  # [T, H, W, C]
                for t in range(current_frames.shape[0]):
                    self.rgb_buffer.append(current_frames[t])
            elif current_frames.ndim == 3:  # [H, W, C]
                self.rgb_buffer.append(current_frames)

        # Create base VLAStepData
        step_data = VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

        # Generate depth maps from buffer
        if len(self.rgb_buffer) > 0:
            buffer_frames = np.stack(list(self.rgb_buffer))  # [T', H, W, 3]
            depth_maps = self._generate_depth_maps(buffer_frames)
            step_data.depth_maps = depth_maps
            step_data.num_temporal_frames = len(self.rgb_buffer)

        return step_data
