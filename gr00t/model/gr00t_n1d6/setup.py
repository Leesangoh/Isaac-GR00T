import json
import logging
from pathlib import Path

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.model.registry import register_model
import numpy as np
from termcolor import colored
import torch
from transformers import AutoModel, AutoProcessor


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class Gr00tN1d6Pipeline(ModelPipeline):
    model_class = Gr00tN1d6
    processor_class = Gr00tN1d6Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""

        # Build transformers loading kwargs from training config

        if self.config.training.start_from_checkpoint is not None:
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            # Initialize mask_tokens if they are not present in the base checkpoint
            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)

            if mask_token_missing and model.action_head.mask_token is not None:
                # Initialize mask_token
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            # DepthMem: expand patch embedding and enable temporal attention
            if getattr(self.config.model, "depthmem_enabled", False):
                self._enable_depthmem(model)

        else:
            model = self.model_class(
                self.config.model, transformers_loading_kwargs=self.transformers_loading_kwargs
            )

        print(colored(f"Model Config: {model.config}", "yellow"))
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        print("Model: ", model)

        return model

    def _enable_depthmem(self, model):
        """Enable DepthMem: expand patch embeddings 3ch→4ch, configure temporal attention, add LoRA.

        After loading from a 3ch checkpoint, this method:
        1. Updates the vision config for 4ch + temporal attention
        2. Replaces the 3ch patch_embedding Linear with a 4ch version (RGB weights copied, depth zero-init)
        3. Updates encoder temporal_attention_layers
        4. Applies LoRA to SigLIP2 vision encoder and Qwen3 LLM
        """
        T = self.config.model.depthmem_num_temporal_frames
        temporal_layers = getattr(self.config.model, "depthmem_temporal_attention_layers", [7, 14, 21, 26])
        lora_rank = getattr(self.config.model, "depthmem_lora_rank", 16)
        patch_size = 14
        rgb_dim = 3 * patch_size * patch_size  # 588
        rgbd_dim = 4 * patch_size * patch_size  # 784

        # Step 1: Update vision_config on the Eagle sub-model
        # Path: backbone.model.vision_model.vision_model (Siglip2VisionModel wraps inner model)
        vision_model = model.backbone.model.vision_model.vision_model
        vision_config = vision_model.config
        vision_config.num_channels = 4
        vision_config.num_temporal_frames = T
        vision_config.temporal_attention_layers = temporal_layers

        # Step 2: Replace patch_embedding Linear (3ch→4ch)
        embeddings = vision_model.embeddings
        old_patch = embeddings.patch_embedding  # nn.Linear(588, hidden_dim)
        hidden_dim = old_patch.out_features
        has_bias = old_patch.bias is not None

        new_patch = torch.nn.Linear(rgbd_dim, hidden_dim, bias=has_bias)
        with torch.no_grad():
            new_patch.weight.data[:, :rgb_dim] = old_patch.weight.data
            new_patch.weight.data[:, rgb_dim:] = 0  # zero-init depth channel
            if has_bias:
                new_patch.bias.data.copy_(old_patch.bias.data)
        new_patch = new_patch.to(dtype=old_patch.weight.dtype, device=old_patch.weight.device)
        embeddings.patch_embedding = new_patch
        logging.info(f"[DepthMem] Expanded patch_embedding: {rgb_dim}→{rgbd_dim} (depth zero-init)")

        # Step 3: Update encoder's temporal_attention_layers set
        # (the encoder caches this from config at __init__, so we update it directly)
        encoder = vision_model.encoder
        encoder.temporal_attention_layers = set(temporal_layers)
        logging.info(
            f"[DepthMem] Temporal attention enabled at layers {temporal_layers} with T={T}"
        )

        # Step 4: Apply LoRA to vision encoder and LLM
        if lora_rank > 0:
            eagle_model = model.backbone.model
            eagle_model.wrap_backbone_lora(
                r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05
            )
            logging.info(f"[DepthMem] Vision LoRA applied (rank={lora_rank})")

            eagle_model.wrap_llm_lora(
                r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05
            )
            logging.info(f"[DepthMem] LLM LoRA applied (rank={lora_rank})")

            # Ensure patch_embedding stays trainable (PEFT freezes base model params)
            # Find patch_embedding in the PEFT-wrapped vision model
            for name, param in eagle_model.vision_model.named_parameters():
                if "patch_embedding" in name:
                    param.requires_grad = True
                    logging.info(f"[DepthMem] Kept {name} trainable")

    def _get_statistics(self) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""

        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                use_relative_action=self.model_config.use_relative_action,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        print(
            colored(
                f"These are all the processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}",
                "yellow",
            )
        )
        if get_rank() == 0:
            with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d6Config, Gr00tN1d6Pipeline)
