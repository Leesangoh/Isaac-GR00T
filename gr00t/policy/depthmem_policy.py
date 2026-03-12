"""DepthMem Policy: Gr00t policy with depth channel and temporal attention.

Extends Gr00tPolicy to:
1. Maintain an RGB frame buffer for Video Depth Anything
2. Generate temporally consistent depth maps at each step
3. Pass RGBD (4ch) frames through SigLIP2 with temporal attention
"""

import copy
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
    - Passes T temporal frames through the VLM pipeline with temporal attention
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
        compile_depth_model: bool = False,
        save_attention_map: bool = False,
        attention_map_dir: str = "./attention_maps",
        save_video_dir: str | None = None,
        video_fps: int = 10,
    ):
        """Initialize DepthMem policy.

        Args:
            embodiment_tag: Robot embodiment tag.
            model_path: Path to pretrained model checkpoint.
            device: Device for inference.
            num_temporal_frames: T - number of frames to maintain in buffer.
            depth_model_size: Video Depth Anything model size ('small', 'base', 'large').
            depth_resolution: Input resolution for depth model.
            compile_depth_model: Whether to torch.compile the depth model.
            save_video_dir: If set, save RGB|Depth side-by-side videos per episode.
            video_fps: FPS for saved videos.
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

        # Deep copy modality_configs so we don't modify the processor's internal state.
        # Override video delta_indices to match the T frames sent by the client.
        self.modality_configs = copy.deepcopy(self.modality_configs)
        self.modality_configs["video"].delta_indices = list(range(-(num_temporal_frames - 1), 1))

        # Load checkpoint with LoRA merging and patch embedding fix.
        # AutoModel.from_pretrained doesn't handle PEFT keys, so we load manually.
        self._load_depthmem_checkpoint(model_path)

        # Per-env episode video recording (RGB | Depth side-by-side)
        self.save_video_dir = save_video_dir
        self.video_fps = video_fps
        self._episode_frames: dict[int, list[np.ndarray]] = {}
        self._episode_counter = 0
        if save_video_dir:
            os.makedirs(save_video_dir, exist_ok=True)
            print(f"[DepthMem] Episode videos will be saved to {save_video_dir}")

        # Load Video Depth Anything
        self.depth_model = self._load_depth_model(depth_model_size, device, compile_depth_model)

        logger.info(
            f"DepthMem policy initialized: T={num_temporal_frames}, "
            f"depth_model={depth_model_size}, resolution={depth_resolution}"
        )

    def _load_depthmem_checkpoint(self, model_path: str):
        """Load DepthMem checkpoint with LoRA merging and patch embedding fix.

        The checkpoint was saved with PEFT/LoRA wrappers, so keys have prefixes like
        'base_model.model.' and LoRA weights as 'lora_A.default.weight' / 'lora_B.default.weight'.
        AutoModel.from_pretrained doesn't handle these, so we:
        1. Load all checkpoint tensors
        2. Map PEFT-prefixed keys to model keys
        3. Merge LoRA adapters: W_merged = W_base + (lora_B @ lora_A) * scale
        4. Handle the 3ch→4ch patch embedding expansion
        5. Load the merged state dict into the model
        """
        from pathlib import Path
        import re

        from safetensors import safe_open

        ckpt_dir = Path(model_path)

        # Step 1: Load all checkpoint tensors
        ckpt_state = {}
        for sf_file in sorted(ckpt_dir.glob("model*.safetensors")):
            with safe_open(str(sf_file), framework="pt") as f:
                for key in f.keys():
                    ckpt_state[key] = f.get_tensor(key)
        print(f"[DepthMem] Loaded {len(ckpt_state)} tensors from checkpoint")

        # Step 2: Separate into base_layer, lora, and normal keys
        base_layer_map = {}  # model_key -> tensor
        lora_a_map = {}  # model_key -> tensor
        lora_b_map = {}  # model_key -> tensor
        normal_map = {}  # model_key -> tensor

        def strip_peft_prefix(key):
            """Remove PEFT wrapper prefixes to get the model key."""
            key = re.sub(r"\.base_model\.model\.", ".", key)
            key = re.sub(r"^base_model\.model\.", "", key)
            return key

        for ck, tensor in ckpt_state.items():
            if ".base_layer." in ck:
                # e.g. backbone.model.xxx.base_model.model.xxx.self_attn.q_proj.base_layer.weight
                model_key = strip_peft_prefix(ck).replace(".base_layer.", ".")
                base_layer_map[model_key] = tensor
            elif ".lora_A.default." in ck:
                model_key = strip_peft_prefix(ck).replace(".lora_A.default.", ".")
                lora_a_map[model_key] = tensor
            elif ".lora_B.default." in ck:
                model_key = strip_peft_prefix(ck).replace(".lora_B.default.", ".")
                lora_b_map[model_key] = tensor
            else:
                model_key = strip_peft_prefix(ck)
                normal_map[model_key] = tensor

        print(
            f"[DepthMem] Keys: {len(normal_map)} normal, "
            f"{len(base_layer_map)} base_layer, "
            f"{len(lora_a_map)} lora_A, {len(lora_b_map)} lora_B"
        )

        # Step 3: Merge LoRA into base weights
        # LoRA formula: W_merged = W_base + (B @ A) * scaling
        # Default LoRA scaling = alpha / rank (typically alpha=rank, so scaling=1.0)
        lora_scaling = 1.0  # alpha=rank by default in PEFT
        merged = {}

        # Start with base_layer weights
        for key, tensor in base_layer_map.items():
            merged[key] = tensor.clone()

        # Add LoRA contribution
        lora_merged_count = 0
        for key in lora_a_map:
            if key in lora_b_map and key in merged:
                A = lora_a_map[key]  # [rank, in_features]
                B = lora_b_map[key]  # [out_features, rank]
                merged[key] = merged[key] + (B @ A) * lora_scaling
                lora_merged_count += 1

        print(f"[DepthMem] Merged {lora_merged_count} LoRA adapters into base weights")

        # Add normal (non-PEFT) weights
        merged.update(normal_map)

        # Step 4: Handle patch embedding 3ch→4ch
        model_params = dict(self.model.named_parameters())
        patch_embed_key = None
        for name in model_params:
            if "patch_embedding.weight" in name:
                patch_embed_key = name
                break

        if patch_embed_key and patch_embed_key in merged:
            ckpt_shape = merged[patch_embed_key].shape
            model_shape = model_params[patch_embed_key].shape
            if ckpt_shape != model_shape and ckpt_shape[1] == 784:
                # Need to expand model's patch_embedding from 3ch to 4ch
                # patch_embed_key is like "...embeddings.patch_embedding.weight"
                # Navigate to "patch_embedding" module (2 levels up from ".weight")
                module_parts = patch_embed_key.rsplit(".", 1)[0].split(".")
                parent = self.model
                for part in module_parts[:-1]:
                    parent = getattr(parent, part)
                old_module = getattr(parent, module_parts[-1])
                device = old_module.weight.device
                dtype = old_module.weight.dtype
                new_embed = torch.nn.Linear(784, ckpt_shape[0], bias=old_module.bias is not None)
                new_embed = new_embed.to(device=device, dtype=dtype)
                setattr(parent, module_parts[-1], new_embed)
                print(f"[DepthMem] Expanded patch_embedding: {model_shape} -> {ckpt_shape}")

        # Step 5: Load merged weights into model
        model_state = self.model.state_dict()
        loaded, skipped, shape_mismatch = 0, 0, 0
        for key, tensor in merged.items():
            if key in model_state:
                if model_state[key].shape == tensor.shape:
                    model_state[key] = tensor
                    loaded += 1
                else:
                    print(
                        f"[DepthMem] Shape mismatch: {key} "
                        f"model={model_state[key].shape} ckpt={tensor.shape}"
                    )
                    shape_mismatch += 1
            else:
                skipped += 1

        self.model.load_state_dict(model_state, strict=False)
        self.model.to(device=self.model.device, dtype=torch.bfloat16)
        print(
            f"[DepthMem] Loaded {loaded} params, skipped {skipped}, shape_mismatch {shape_mismatch}"
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

    @staticmethod
    def _colorize_depth(depth: np.ndarray) -> np.ndarray:
        """Convert a [H, W] float32 depth map in [0,1] to an [H, W, 3] uint8 inferno colormap."""
        import matplotlib.cm as cm

        colored = cm.inferno(depth)[:, :, :3]  # [H, W, 3] float in [0,1]
        return (colored * 255).astype(np.uint8)

    def _save_episode_video(self, frames: list[np.ndarray] | None = None):
        """Save RGB|Depth frames as a side-by-side mp4 (h264)."""
        if frames is None:
            return
        if not frames:
            return

        import subprocess

        path = os.path.join(self.save_video_dir, f"episode_{self._episode_counter:04d}.mp4")
        H, W, _ = frames[0].shape

        # Pipe raw frames to ffmpeg for h264 encoding
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(self.video_fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "fast",
            path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
        print(f"[DepthMem] Saved episode video ({len(frames)} frames): {path}")

    def _flush_video(self, env_idx: int | None = None):
        """Save any accumulated episode frames.

        Args:
            env_idx: If given, flush only that env. Otherwise flush all envs.
        """
        if not self.save_video_dir:
            return
        indices = [env_idx] if env_idx is not None else list(self._episode_frames.keys())
        for idx in indices:
            frames = self._episode_frames.get(idx, [])
            if frames:
                self._save_episode_video(frames)
                self._episode_counter += 1
                self._episode_frames[idx] = []

    def _flush_all_videos(self):
        """Save all remaining episode videos (called on shutdown)."""
        self._flush_video()

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset for a new episode."""
        result = super().reset(options)
        self._flush_all_videos()
        return result

    def __del__(self):
        """Save any remaining episode video on cleanup."""
        self._flush_all_videos()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute actions with temporal depth and attention.

        The client (MultiStepWrapper) sends T consecutive video frames via
        video_delta_indices=[-T+1, ..., 0]. This method generates depth maps
        per env and passes T RGBD frames through the model.
        """
        T = self.num_temporal_frames
        video_keys = self.modality_configs["video"].modality_keys
        primary_key = video_keys[0]
        batched_video = observation["video"][primary_key]  # [B, T, H, W, C]
        B = batched_video.shape[0]

        # Step 1: Generate depth maps per env from the T frames sent by client
        per_env_depth = []
        for b in range(B):
            temporal_frames = batched_video[b]  # [T, H, W, C]
            depth_maps = self._generate_depth_maps(temporal_frames)  # [T, H, W]
            per_env_depth.append(depth_maps)

            # Record RGB | Depth side-by-side frame for episode video
            if self.save_video_dir:
                if b not in self._episode_frames:
                    self._episode_frames[b] = []
                current_frame = temporal_frames[-1]  # current frame
                depth_colored = self._colorize_depth(depth_maps[-1])
                if depth_colored.shape[:2] != current_frame.shape[:2]:
                    import cv2

                    depth_colored = cv2.resize(
                        depth_colored, (current_frame.shape[1], current_frame.shape[0])
                    )
                side_by_side = np.concatenate([current_frame, depth_colored], axis=1)
                self._episode_frames[b].append(side_by_side)

        # Step 2: Unbatch, create VLAStepData with per-env depth, process
        if self.save_attention_map:
            self._current_obs_images = [
                observation["video"][k][0, -1]
                for k in self.modality_configs["video"].modality_keys
            ]
            self._current_instruction = observation["language"][self.language_key][0][0]

        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []
        states = []

        for b, obs in enumerate(unbatched_observations):
            step_data = VLAStepData(
                images=obs["video"],
                states=obs["state"],
                actions={},
                text=obs["language"][self.language_key][0],
                embodiment=self.embodiment_tag,
            )
            step_data.depth_maps = per_env_depth[b]  # per-env depth [T, H, W]
            step_data.num_temporal_frames = T

            states.append(step_data.states)
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
            processed_inputs.append(self.processor(messages))

        collated_inputs = self.collate_fn(processed_inputs)
        collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

        # Step 3: Run model inference
        with self._attention_capture_ctx() as attn_ctx:
            with torch.inference_mode():
                model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()

        if attn_ctx is not None:
            self._flush_attention_data(attn_ctx)

        # Step 4: Decode actions
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
