"""Gr00t Policy implementation for inference.

This module provides the core policy classes for running Gr00t models:
- Gr00tPolicy: Base policy class for model inference
- Gr00tSimPolicyWrapper: Wrapper for compatibility with existing Gr00t simulation environments
"""

from contextlib import contextmanager
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.types import MessageType, ModalityConfig, VLAStepData

from .policy import BasePolicy, PolicyWrapper


logger = logging.getLogger(__name__)


def _rec_to_dtype(x: Any, dtype: torch.dtype) -> Any:
    """Recursively convert all floating point tensors in a nested structure to the given dtype.

    Args:
        x: Input data structure (tensor, dict, list, or other)
        dtype: Target torch dtype for floating point tensors

    Returns:
        Data structure with floating point tensors converted to target dtype

    Warning:
        Non-floating point tensors will be left as is.
    """
    if isinstance(x, torch.Tensor) and torch.is_floating_point(x):
        return x.to(dtype=dtype)
    # Handle dict-like objects (tianshou.BatchFeature is not dict but has items() method)
    elif isinstance(x, dict) or hasattr(x, "items"):
        return {k: _rec_to_dtype(v, dtype) for k, v in x.items()}  # type: ignore
    elif isinstance(x, list):
        return [_rec_to_dtype(v, dtype) for v in x]
    else:
        return x


class AttentionWeightCaptureProcessor:
    """Attention processor that captures weights via manual scaled dot-product.

    Drop-in replacement for ``AttnProcessor2_0`` that computes attention
    explicitly (Q @ K^T / sqrt(d) -> softmax) so the weight matrix can be
    extracted.  The captured weights are stored in ``self.captured_weights``
    after each forward call.
    """

    def __init__(self):
        self.captured_weights = None  # [B, heads, Q_len, K_len]

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # Prepare attention mask (same logic as AttnProcessor2_0)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Q, K, V projections
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape for multi-head: (B, seq, heads*head_dim) -> (B, heads, seq, head_dim)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Manual attention: Q @ K^T / sqrt(d)
        scale = head_dim**-0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))
            else:
                attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights.softmax(dim=-1)

        # Store captured weights [B, heads, Q_len, K_len]
        self.captured_weights = attn_weights.detach()

        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Output projection + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class Gr00tPolicy(BasePolicy):
    """Core policy class for Gr00t model inference.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with pretrained VLA processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag,
        model_path: str,
        *,
        device: int | str,
        strict: bool = True,
        save_attention_map: bool = False,
        attention_map_dir: str = "./attention_maps",
    ):
        """Initialize the Gr00t Policy.

        Args:
            embodiment_tag: The embodiment tag defining the robot/environment type
            model_path: Path to the pretrained model checkpoint directory
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
            save_attention_map: Save cross-attention heatmaps from DiT action head.
            attention_map_dir: Directory for attention map images.
        """
        # Import this to register all models.
        import gr00t.model  # noqa: F401

        super().__init__(strict=strict)
        model_dir = Path(model_path)

        # Load the pretrained model and move to target device with bfloat16 precision
        model = AutoModel.from_pretrained(model_dir)
        model.eval()  # Set model to evaluation mode
        model.to(device=device, dtype=torch.bfloat16)
        self.model = model

        # Load the processor for input/output transformation
        self.processor: BaseProcessor = AutoProcessor.from_pretrained(model_dir)
        self.processor.eval()

        # Store embodiment-specific configurations
        self.embodiment_tag = embodiment_tag
        self.modality_configs = self.processor.get_modality_configs()[self.embodiment_tag.value]
        self.collate_fn = self.processor.collator

        # Extract and validate language configuration
        # Currently only supports single language input per timestep
        language_keys = self.modality_configs["language"].modality_keys
        language_delta_indices = self.modality_configs["language"].delta_indices
        assert len(language_keys) == 1, "Only one language key is supported"
        assert len(language_delta_indices) == 1, "Only one language delta index is supported"
        self.language_key = language_keys[0]

        # Attention map state
        self.save_attention_map = save_attention_map
        self.attention_map_dir = attention_map_dir
        self._attn_episode_counter = 0
        self._attn_rows: list = []  # buffered row images for current episode

        if save_attention_map:
            os.makedirs(attention_map_dir, exist_ok=True)
            self._action_head = self._find_action_head()
            logger.info("Attention map saving enabled, output dir: %s", attention_map_dir)

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(self, observation: dict[str, Any]) -> VLAStepData:
        """Convert a single observation into a VLAStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            VLAStepData object ready for processor input
        """
        return VLAStepData(
            images=observation["video"],
            states=observation["state"],
            actions={},  # No ground truth actions during inference
            text=observation["language"][self.language_key][0],
            embodiment=self.embodiment_tag,
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate that the observation has the correct structure and types.

        This method ensures that all required modalities are present and that their
        data types, shapes, and dimensions match the model's expectations.

        Expected observation structure:
            - video: dict[str, np.ndarray[np.uint8, (B, T, H, W, C)]]
                - B: batch size
                - T: temporal horizon (number of frames)
                - H, W: image height and width
                - C: number of channels (must be 3 for RGB)
            - state: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: temporal horizon (number of state observations)
                - D: state dimension
            - language: dict[str, list[list[str]]]
                - Shape: (B, T) where each element is a string
                - T: temporal horizon (typically 1 for language)

        Args:
            observation: Dictionary containing video, state, and language modalities

        Raises:
            AssertionError: If any validation check fails
        """
        # Check that observation contains all required top-level modality keys
        for modality in ["video", "state", "language"]:
            assert modality in observation, f"Observation must contain a '{modality}' key"
            assert isinstance(observation[modality], dict), (
                f"Observation '{modality}' must be a dictionary. Got {type(observation[modality])}: {observation[modality]}"
            )

        # Track batch size across modalities to ensure consistency
        bs = -1

        # ===== VIDEO VALIDATION =====
        # Validate each video stream defined in the modality config
        for video_key in self.modality_configs["video"].modality_keys:
            # Set or verify batch size consistency across all video keys
            if bs == -1:
                bs = len(observation["video"][video_key])
            else:
                assert len(observation["video"][video_key]) == bs, (
                    f"Video key '{video_key}' must have batch size {bs}. Got {len(observation['video'][video_key])}"
                )

            # Check that the expected video key exists in the observation
            assert video_key in observation["video"], (
                f"Video key '{video_key}' must be in observation"
            )

            batched_video = observation["video"][video_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(self.modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(self.modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Validate each state stream defined in the modality config
        for state_key in self.modality_configs["state"].modality_keys:
            # Set or verify batch size consistency across all state keys
            if bs == -1:
                bs = len(observation["state"][state_key])
            else:
                assert len(observation["state"][state_key]) == bs, (
                    f"State key '{state_key}' must have batch size {bs}. Got {len(observation['state'][state_key])}"
                )

            # Check that the expected state key exists in the observation
            assert state_key in observation["state"], (
                f"State key '{state_key}' must be in observation"
            )

            batched_state = observation["state"][state_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(self.modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(self.modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Validate each language stream defined in the modality config
        for language_key in self.modality_configs["language"].modality_keys:
            # Set or verify batch size consistency (language uses len instead of .shape)
            if bs == -1:
                bs = len(observation["language"][language_key])
            else:
                assert len(observation["language"][language_key]) == bs, (
                    f"Language key '{language_key}' must have batch size {bs}. Got {len(observation['language'][language_key])}"
                )

            # Check that the expected language key exists in the observation
            assert language_key in observation["language"], (
                f"Language key '{language_key}' must be in observation"
            )

            batched_language: list[list[str]] = observation["language"][language_key]

            # Verify outer structure is a list (batch dimension)
            assert isinstance(batched_language, list), (
                f"Language key '{language_key}' must be a list. Got {type(batched_language)}"
            )

            # Validate each batch item
            for batch_item in batched_language:
                # Verify temporal dimension matches expected horizon
                assert len(batch_item) == len(self.modality_configs["language"].delta_indices), (
                    f"Language key '{language_key}'s horizon must be {len(self.modality_configs['language'].delta_indices)}. Got {len(batched_language)}"
                )

                # Verify inner structure is also a list (temporal dimension)
                assert isinstance(batch_item, list), (
                    f"Language batch item must be a list. Got {type(batch_item)}"
                )

                # Current implementation expects exactly one language instruction per timestep
                assert len(batch_item) == 1, (
                    f"Language batch item must have exactly one item. Got {len(batch_item)}"
                )

                # Verify the instruction itself is a string
                assert isinstance(batch_item[0], str), (
                    f"Language batch item must be a string. Got {type(batch_item[0])}"
                )

    # ------------------------------------------------------------------
    # Attention map capture helpers
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

    def _get_target_cross_attn_info(self) -> dict[str, Any]:
        """Identify which DiT blocks to capture cross-attention from.

        For standard DiT: returns the last cross-attention block index.
        For AlternateVLDiT: returns the last image-attending and last
        text-attending block indices separately.
        """
        dit_model = self._action_head.model
        blocks = dit_model.transformer_blocks
        is_alternate = self._action_head.config.use_alternate_vl_dit

        if is_alternate:
            n = dit_model.attend_text_every_n_blocks
            last_image_idx = None
            last_text_idx = None
            for idx in range(len(blocks)):
                if idx % 2 == 0:  # cross-attention block
                    if idx % (2 * n) == 0:
                        last_text_idx = idx
                    else:
                        last_image_idx = idx
            return {
                "is_alternate": True,
                "image_block_idx": last_image_idx,
                "text_block_idx": last_text_idx,
            }
        else:
            last_cross_idx = None
            for idx in range(len(blocks)):
                if blocks[idx].cross_attention_dim is not None:
                    last_cross_idx = idx
            return {
                "is_alternate": False,
                "cross_block_idx": last_cross_idx,
            }

    def _swap_attn_processors(self, info: dict[str, Any]):
        """Replace attention processors on target blocks with capture processors.

        Returns (block_indices, capture_processors, original_processors).
        """
        blocks = self._action_head.model.transformer_blocks
        block_indices = []
        capture_procs = []
        orig_procs = []

        if info["is_alternate"]:
            for idx in [info["image_block_idx"], info["text_block_idx"]]:
                if idx is not None:
                    block_indices.append(idx)
                    orig_procs.append(blocks[idx].attn1.processor)
                    cap = AttentionWeightCaptureProcessor()
                    blocks[idx].attn1.set_processor(cap)
                    capture_procs.append(cap)
        else:
            idx = info["cross_block_idx"]
            if idx is not None:
                block_indices.append(idx)
                orig_procs.append(blocks[idx].attn1.processor)
                cap = AttentionWeightCaptureProcessor()
                blocks[idx].attn1.set_processor(cap)
                capture_procs.append(cap)

        return block_indices, capture_procs, orig_procs

    def _restore_attn_processors(
        self,
        block_indices: list[int],
        original_processors: list,
    ):
        """Restore original attention processors after capture."""
        blocks = self._action_head.model.transformer_blocks
        for idx, proc in zip(block_indices, original_processors):
            blocks[idx].attn1.set_processor(proc)

    @staticmethod
    def _find_contiguous_runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
        """Find (start, end) indices of contiguous True runs."""
        runs = []
        in_run = False
        start = 0
        for i, val in enumerate(mask_1d):
            if val and not in_run:
                start = i
                in_run = True
            elif not val and in_run:
                runs.append((start, i))
                in_run = False
        if in_run:
            runs.append((start, len(mask_1d)))
        return runs

    @contextmanager
    def _attention_capture_ctx(self):
        """Context manager that wraps backbone and swaps DiT attention processors.

        When ``save_attention_map`` is False the context manager is a no-op
        (yields ``None``).  Otherwise it:

        1. Wraps ``self.model.backbone.forward`` to capture backbone output
           (needed for ``image_mask`` / ``backbone_attention_mask``).
        2. Swaps attention processors on target DiT cross-attention blocks
           with :class:`AttentionWeightCaptureProcessor` instances.  Because
           the capture processor overwrites ``captured_weights`` on every
           call, after the denoising loop finishes the stored tensor
           corresponds to the **last** denoising step.
        3. Yields a context dict with the captured data.
        4. Restores all originals in the ``finally`` block.
        """
        if not self.save_attention_map:
            yield None
            return

        # Wrap backbone to capture its output
        orig_fwd = self.model.backbone.forward
        ctx: dict[str, Any] = {
            "backbone_output": None,
            "capture_procs": [],
            "attn_info": {},
        }

        def _capturing_fwd(*a, **kw):
            out = orig_fwd(*a, **kw)
            ctx["backbone_output"] = out
            return out

        self.model.backbone.forward = _capturing_fwd

        # Swap attention processors on target blocks
        attn_info = self._get_target_cross_attn_info()
        ctx["attn_info"] = attn_info
        block_indices, capture_procs, orig_procs = self._swap_attn_processors(attn_info)
        ctx["capture_procs"] = capture_procs

        try:
            yield ctx
        finally:
            self.model.backbone.forward = orig_fwd
            self._restore_attn_processors(block_indices, orig_procs)

    def _flush_attention_data(self, ctx: dict[str, Any]):
        """Generate an attention row and append it to the episode buffer.

        The accumulated grid image is saved to disk periodically (every 10
        steps) so the file is always reasonably up-to-date even if
        ``reset()`` is never called.
        """
        try:
            backbone_out = ctx.get("backbone_output")
            capture_procs = ctx.get("capture_procs", [])
            attn_info = ctx.get("attn_info", {})
            if backbone_out is None or len(capture_procs) == 0:
                return
            captured_attn = capture_procs[0].captured_weights
            if captured_attn is None:
                return
            captured_attn_text = None
            if attn_info.get("is_alternate") and len(capture_procs) > 1:
                captured_attn_text = capture_procs[1].captured_weights
            row = self._make_attention_row(
                captured_attn,
                backbone_out.image_mask,
                backbone_out.backbone_attention_mask,
                attn_weights_text=captured_attn_text,
                obs_images=self._current_obs_images,
                instruction=self._current_instruction,
            )
            self._attn_rows.append(row)
            # Save periodically to keep file up-to-date
            n = len(self._attn_rows)
            if n <= 3 or n % 10 == 0:
                self._save_episode_image()
        except Exception:
            logger.exception("Failed to save attention map")

    _ATTN_CELL = 96  # thumbnail size (pixels) for each attention cell
    _ATTN_TEXT_W = 400  # width of text attention bar

    def _make_attention_row(
        self,
        attn_weights: torch.Tensor,
        image_mask: torch.Tensor,
        backbone_attn_mask: torch.Tensor,
        attn_weights_text: torch.Tensor | None = None,
        obs_images: list[np.ndarray] | None = None,
        instruction: str | None = None,
    ):
        """Generate one row: camera heatmap (t=0) + text attention bar.

        Only the first action token (t=0) is visualised since
        ``n_action_steps=1`` means only that token is executed.

        Returns a PIL Image of shape ``(cell_h, CELL + TEXT_W, 3)``.
        """
        from matplotlib import cm as mpl_cm
        from PIL import Image, ImageDraw, ImageFont

        CELL = self._ATTN_CELL
        TEXT_W = self._ATTN_TEXT_W
        instruction = instruction or ""

        if obs_images is None or len(obs_images) == 0:
            obs_images = [np.zeros((CELL, CELL, 3), dtype=np.uint8)]

        # Average across heads -> [Q, S], use first action token (index 1)
        attn_avg = attn_weights.float().mean(dim=1)[0].cpu().numpy()  # [Q, S]
        t0_attn = attn_avg[1]  # [S] — first action token

        img_mask = image_mask[0].cpu().numpy().astype(bool)
        valid_mask = backbone_attn_mask[0].cpu().numpy().astype(bool)
        text_mask = (~img_mask) & valid_mask
        cam_runs = self._find_contiguous_runs(img_mask)
        num_cameras = max(min(len(cam_runs), len(obs_images)), 1)
        cell_h = num_cameras * CELL

        # -- Text attention source --
        if attn_weights_text is not None:
            t0_text_attn = attn_weights_text.float().mean(dim=1)[0].cpu().numpy()[1]
        else:
            t0_text_attn = t0_attn
        text_positions = np.where(text_mask)[0]

        # -- Global normalization across image + text --
        # Collect all valid attention values to compute a single min/max
        all_attn_parts = []
        for cam_idx in range(num_cameras):
            if cam_idx < len(cam_runs):
                start, end = cam_runs[cam_idx]
                all_attn_parts.append(t0_attn[start:end])
        if len(text_positions) > 0:
            all_attn_parts.append(t0_text_attn[text_positions])
        if all_attn_parts:
            all_vals = np.concatenate(all_attn_parts)
            g_min, g_max = float(all_vals.min()), float(all_vals.max())
        else:
            g_min, g_max = 0.0, 1.0
        g_range = g_max - g_min if g_max > g_min else 1.0

        # -- Camera heatmap for t=0 --
        cam_img = Image.new("RGB", (CELL, cell_h), (0, 0, 0))
        for cam_idx in range(num_cameras):
            orig = obs_images[min(cam_idx, len(obs_images) - 1)]
            if cam_idx < len(cam_runs):
                start, end = cam_runs[cam_idx]
                n_patches = end - start
                patch_attn = t0_attn[start:end]

                pn = (patch_attn - g_min) / g_range

                grid_side = int(math.sqrt(n_patches))
                if grid_side * grid_side == n_patches:
                    attn_2d = pn.reshape(grid_side, grid_side)
                else:
                    for h in range(int(math.sqrt(n_patches)), 0, -1):
                        if n_patches % h == 0:
                            attn_2d = pn.reshape(h, n_patches // h)
                            break
                    else:
                        attn_2d = pn.reshape(1, n_patches)

                img_h, img_w = orig.shape[:2]
                attn_up = (
                    np.array(
                        Image.fromarray((attn_2d * 255).astype(np.uint8)).resize(
                            (img_w, img_h), Image.BILINEAR
                        )
                    ).astype(np.float32)
                    / 255.0
                )
                heatmap = (mpl_cm.jet(attn_up)[:, :, :3] * 255).astype(np.uint8)
                overlay = (0.5 * orig.astype(np.float32) + 0.5 * heatmap.astype(np.float32)).astype(
                    np.uint8
                )
                thumb = Image.fromarray(overlay).resize((CELL, CELL), Image.BILINEAR)
            else:
                thumb = Image.fromarray(orig).resize((CELL, CELL), Image.BILINEAR)
            cam_img.paste(thumb, (0, cam_idx * CELL))

        # -- Text attention bar --
        words = instruction.split() if instruction else []

        try:
            font_word = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13
            )
        except (OSError, IOError):
            font_word = ImageFont.load_default()

        text_bar = Image.new("RGB", (TEXT_W, cell_h), (255, 255, 255))
        if len(text_positions) > 0 and len(words) > 0:
            draw = ImageDraw.Draw(text_bar)
            t_attn = t0_text_attn[text_positions]
            # Use same global normalization
            t_attn_norm = (t_attn - g_min) / g_range

            num_tokens = len(t_attn_norm)
            num_words = len(words)
            word_attns = []
            for w_idx in range(num_words):
                s = int(w_idx * num_tokens / num_words)
                e = max(int((w_idx + 1) * num_tokens / num_words), s + 1)
                word_attns.append(float(t_attn_norm[s : min(e, num_tokens)].mean()))

            # Measure word widths
            word_widths = []
            for word in words:
                bbox = draw.textbbox((0, 0), word + " ", font=font_word)
                word_widths.append(bbox[2] - bbox[0])
            total_text_w = sum(word_widths)
            scale = min(TEXT_W / total_text_w, 1.0) if total_text_w > 0 else 1.0

            x = 0
            y_text = (cell_h - 15) // 2
            for w_idx, word in enumerate(words):
                w = int(word_widths[w_idx] * scale)
                if w <= 0:
                    continue
                rgb = mpl_cm.jet(word_attns[w_idx])[:3]
                bg = tuple(int(c * 255) for c in rgb)
                draw.rectangle([x, 0, x + w - 1, cell_h - 1], fill=bg)
                fg = (255, 255, 255) if sum(bg) < 384 else (0, 0, 0)
                draw.text((x + 2, y_text), word, fill=fg, font=font_word)
                x += w

        # Combine: [camera heatmap | text bar]
        row_w = CELL + TEXT_W
        row = Image.new("RGB", (row_w, cell_h), (255, 255, 255))
        row.paste(cam_img, (0, 0))
        row.paste(text_bar, (CELL, 0))
        return row

    def _save_episode_image(self):
        """Combine all buffered rows into a grid and save to disk."""
        from PIL import Image, ImageDraw, ImageFont

        if not self._attn_rows:
            return

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except (OSError, IOError):
            font = ImageFont.load_default()

        row_w, cell_h = self._attn_rows[0].size
        num_steps = len(self._attn_rows)

        LABEL_W = 36
        HEADER_H = 16

        total_w = LABEL_W + row_w
        total_h = HEADER_H + num_steps * cell_h

        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Column headers
        draw.text((LABEL_W + 2, 1), "image attn", fill=(100, 100, 100), font=font)
        draw.text(
            (LABEL_W + self._ATTN_CELL + 4, 1),
            "text attn",
            fill=(100, 100, 100),
            font=font,
        )

        # Paste rows with step labels on the left
        for s, row in enumerate(self._attn_rows):
            y = HEADER_H + s * cell_h
            draw.text((1, y + cell_h // 2 - 5), f"s{s}", fill=(0, 0, 0), font=font)
            canvas.paste(row, (LABEL_W, y))

        filename = f"attn_{self._attn_episode_counter:05d}.jpg"
        filepath = os.path.join(self.attention_map_dir, filename)
        canvas.save(filepath, quality=85)
        logger.info("Saved attention map: %s (%d steps)", filepath, num_steps)

    # ------------------------------------------------------------------

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to VLAStepData and process
        3. Collate into model input batch
        4. Run model inference (with optional attention capture)
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        # Extract observation data for attention map visualization
        if self.save_attention_map:
            self._current_obs_images = [
                observation["video"][k][0, -1]  # [H, W, 3] uint8
                for k in self.modality_configs["video"].modality_keys
            ]
            self._current_instruction = observation["language"][self.language_key][0][0]

        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the VLA processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed_inputs.append(self.processor(messages))

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 4: Run model inference to predict actions
        # The attention capture ctx wraps backbone and swaps DiT processors
        # so the last denoising step's attention weights are retained.
        with self._attention_capture_ctx() as attn_ctx:
            with torch.inference_mode():
                model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        # Save attention map (overwrites same file within an episode)
        if attn_ctx is not None:
            self._flush_attention_data(attn_ctx)

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in self.modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), self.embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, {}

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate that the action has the correct structure and types.

        This method ensures that all required action keys are present and that their
        data types, shapes, and dimensions match the model's action space.

        Expected action structure:
            - action: dict[str, np.ndarray[np.float32, (B, T, D)]]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension (e.g., joint positions, velocities, gripper state)

        Args:
            action: Dictionary containing action arrays for each action key

        Raises:
            AssertionError: If any validation check fails
        """
        # Validate each action key defined in the modality config
        for action_key in self.modality_configs["action"].modality_keys:
            # Check that the expected action key exists
            assert action_key in action, f"Action key '{action_key}' must be in action"

            action_arr = action[action_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(self.modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(self.modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        return self.modality_configs

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        When ``save_attention_map`` is enabled, saves the accumulated
        episode image, clears the buffer, and increments the episode
        counter so the next episode writes to a new file.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        if self.save_attention_map:
            if self._attn_rows:
                self._save_episode_image()
                self._attn_rows = []
            self._attn_episode_counter += 1
        return {}


class Gr00tSimPolicyWrapper(PolicyWrapper):
    """Wrapper for Gr00tPolicy to enable compatibility with existing Gr00t simulation environments.

    This wrapper is specifically designed for retro-fitting the Gr00t policy with the current
    Gr00t simulation environment interface. It handles the transformation between the flat
    observation format used by Gr00t sim environments (with keys like 'video.camera_name',
    'state.joint_positions') and the nested format expected by Gr00tPolicy.

    **Important**: If you are using other environments, custom robots, or building new environments,
    you should use `Gr00tPolicy` directly and format your observations according to its interface.
    This wrapper is only needed for compatibility with the existing Gr00t sim infrastructure.

    Key transformations performed by this wrapper:
    - Observation keys: 'video.cam' -> observation['video']['cam']
    - Observation keys: 'state.joints' -> observation['state']['joints']
    - Language keys: 'task' or 'annotation.human.coarse_action' -> observation['language']['task']
    - Action keys: action['joints'] -> 'action.joints'
    """

    def __init__(self, policy: Gr00tPolicy, *, strict: bool = True):
        """Initialize the wrapper around a Gr00tPolicy instance.

        Args:
            policy: The Gr00tPolicy instance to wrap
            strict: Whether to enforce strict validation (default: True)
        """
        super().__init__(policy, strict=strict)
        self.policy: Gr00tPolicy = policy
        assert len(self.policy.modality_configs["language"].delta_indices) == 1, (
            "Only one language delta index is supported"
        )

    def check_observation(self, observation: dict[str, Any]) -> None:
        """Validate observation from Gr00t sim environment format.

        This validation is specific to the flat observation format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_observation which expects nested dicts, this expects flat keys.

        Expected observation structure (Gr00t sim format):
            - Flat keys like 'video.camera_name': np.ndarray[np.uint8, (B, T, H, W, C)]
            - Flat keys like 'state.state_name': np.ndarray[np.float32, (B, T, D)]
            - Language keys: tuple[str] or list[str] with shape (B,)
                - Key can be 'task' or 'annotation.human.coarse_action' (for DC envs)

        Args:
            observation: Flat observation dictionary from Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # ===== VIDEO VALIDATION =====
        # Check video modalities with flat key format: 'video.camera_name'
        for video_key in modality_configs["video"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"video.{video_key}"
            assert parsed_key in observation, f"Video key '{parsed_key}' must be in observation"

            batched_video = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_video, np.ndarray), (
                f"Video key '{video_key}' must be a numpy array. Got {type(batched_video)}"
            )

            # Verify dtype is uint8 (standard for image data, range 0-255)
            assert batched_video.dtype == np.uint8, (
                f"Video key '{video_key}' must be a numpy array of type np.uint8. Got {batched_video.dtype}"
            )

            # Verify shape has 5 dimensions: (B, T, H, W, C)
            assert batched_video.ndim == 5, (
                f"Video key '{video_key}' must be a numpy array of shape (B, T, H, W, C), got {batched_video.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_video.shape[1] == len(modality_configs["video"].delta_indices), (
                f"Video key '{video_key}'s horizon must be {len(modality_configs['video'].delta_indices)}. Got {batched_video.shape[1]}"
            )

            # Verify channel dimension is 3 (RGB images)
            assert batched_video.shape[-1] == 3, (
                f"Video key '{video_key}'s channel 'C' must be 3. Got {batched_video.shape[-1]}"
            )

        # ===== STATE VALIDATION =====
        # Check state modalities with flat key format: 'state.state_name'
        for state_key in modality_configs["state"].modality_keys:
            # Construct flat key expected in Gr00t sim environment
            parsed_key = f"state.{state_key}"
            assert parsed_key in observation, f"State key '{parsed_key}' must be in observation"

            batched_state = observation[parsed_key]

            # Verify data type is numpy array
            assert isinstance(batched_state, np.ndarray), (
                f"State key '{state_key}' must be a numpy array. Got {type(batched_state)}"
            )

            # Verify dtype is float32 (standard for continuous state values)
            assert batched_state.dtype == np.float32, (
                f"State key '{state_key}' must be a numpy array of type np.float32. Got {batched_state.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert batched_state.ndim == 3, (
                f"State key '{state_key}' must be a numpy array of shape (B, T, D), got {batched_state.shape}"
            )

            # Verify temporal dimension matches the expected horizon from config
            assert batched_state.shape[1] == len(modality_configs["state"].delta_indices), (
                f"State key '{state_key}'s horizon must be {len(modality_configs['state'].delta_indices)}. Got {batched_state.shape[1]}"
            )

        # ===== LANGUAGE VALIDATION =====
        # Check language modalities (special handling for DC environment compatibility)
        for language_key in modality_configs["language"].modality_keys:
            # PATCH: Legacy compatibility for DC environments
            # DC envs use 'annotation.human.coarse_action' instead of 'task'
            if language_key == "task" and "annotation.human.coarse_action" in observation:
                language_key = "annotation.human.coarse_action"
            # /PATCH

            # Check that the expected language key exists
            assert language_key in observation, (
                f"Language key '{language_key}' must be in observation"
            )

            # In Gr00t sim format, language is a tuple of strings (B,)
            batched_language: tuple[str] | list[str] = observation[language_key]  # (B,)

            # Verify outer structure is a tuple (batch dimension)
            assert isinstance(batched_language, (tuple, list)), (
                f"Language key '{language_key}' must be a tuple or list. Got {type(batched_language)}"
            )

            # Verify each batch item is a string
            assert isinstance(batched_language[0], str), (
                f"Language batch item must be a string. Got {type(batched_language[0])}"
            )

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Transform Gr00t sim observation format and compute actions.

        This method transforms the flat observation format from Gr00t sim environments
        into the nested format expected by Gr00tPolicy, computes actions, and transforms
        them back to the flat format expected by Gr00t sim environments.

        Input format (Gr00t sim):
            - Flat keys: 'video.camera_name', 'state.state_name'
            - Language: tuple[str] (B,)

        Output format (Gr00t sim):
            - Flat keys: 'action.action_name'

        Args:
            observation: Flat observation dictionary from Gr00t sim environment
            options: Optional parameters (currently unused)

        Returns:
            Tuple of (flat_actions_dict, info_dict)
        """
        # Transform flat observation format to nested format expected by Gr00tPolicy
        new_obs = {}
        for modality in ["video", "state", "language"]:
            new_obs[modality] = {}
            for key in self.policy.modality_configs[modality].modality_keys:
                if modality == "language":
                    # PATCH: Legacy compatibility for DC environments
                    if key == "task" and "annotation.human.coarse_action" in observation:
                        parsed_key = "annotation.human.coarse_action"
                    # /PATCH
                    else:
                        parsed_key = key
                else:
                    # Construct flat key (e.g., 'video.camera' or 'state.joints')
                    parsed_key = f"{modality}.{key}"

                arr = observation[parsed_key]

                # Transform to nested format
                if modality == "language":
                    # Convert from tuple[str] or list[str] (B,) to list[list[str]] (B, 1)
                    # Each element becomes a list with one string for temporal dimension
                    new_obs[modality][key] = [[str(item)] for item in arr]
                else:
                    # Video and state arrays are already in correct format (B, T, ...)
                    new_obs[modality][key] = arr

        # Compute actions using the underlying Gr00tPolicy
        action, info = self.policy.get_action(new_obs, options)

        # Transform actions back to flat format for Gr00t sim environment
        # action['joints'] -> 'action.joints'
        return {f"action.{key}": action[key] for key in action}, info

    def check_action(self, action: dict[str, Any]) -> None:
        """Validate action in Gr00t sim environment format.

        This validation is specific to the flat action format used by Gr00t sim environments.
        Unlike Gr00tPolicy.check_action which expects nested dicts, this expects flat keys.

        Expected action structure (Gr00t sim format):
            - Flat keys like 'action.action_name': np.ndarray[np.float32, (B, T, D)]
                - B: batch size
                - T: action horizon (number of future action steps)
                - D: action dimension

        Args:
            action: Flat action dictionary for Gr00t sim environment

        Raises:
            AssertionError: If any validation check fails
        """
        modality_configs = self.get_modality_config()

        # Validate each action key defined in the modality config
        for action_key in modality_configs["action"].modality_keys:
            # Construct flat key expected in Gr00t sim environment (e.g., 'action.joints')
            parsed_key = f"action.{action_key}"
            assert parsed_key in action, f"Action key '{parsed_key}' must be in action"

            action_arr = action[parsed_key]

            # Verify data type is numpy array
            assert isinstance(action_arr, np.ndarray), (
                f"Action key '{action_key}' must be a numpy array. Got {type(action_arr)}"
            )

            # Verify dtype is float32 (standard for continuous actions)
            assert action_arr.dtype == np.float32, (
                f"Action key '{action_key}' must be a numpy array of type np.float32. Got {action_arr.dtype}"
            )

            # Verify shape has 3 dimensions: (B, T, D)
            assert action_arr.ndim == 3, (
                f"Action key '{action_key}' must be a numpy array of shape (B, T, D), got {action_arr.shape}"
            )

            # Verify action horizon matches the expected temporal dimension from config
            assert action_arr.shape[1] == len(modality_configs["action"].delta_indices), (
                f"Action key '{action_key}'s horizon must be {len(modality_configs['action'].delta_indices)}. Got {action_arr.shape[1]}"
            )

    def get_modality_config(self) -> dict[str, ModalityConfig]:
        """Get the modality configuration from the underlying policy.

        Returns:
            Dictionary mapping modality names to their configurations
        """
        return self.policy.get_modality_config()
