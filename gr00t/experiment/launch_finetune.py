# Launch finetuning for N1.6 on "single node".
# This script tries to provide a similar user experience as current OSS.

import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.video_backend = ft_config.video_backend

    # Gradient checkpointing
    if ft_config.gradient_checkpointing:
        config.training.gradient_checkpointing = True

    # Override action horizon if specified
    if ft_config.action_horizon is not None:
        config.model.action_horizon = ft_config.action_horizon

    # DepthMem configuration
    if ft_config.depthmem_enabled:
        T = ft_config.depthmem_num_temporal_frames
        config.model.depthmem_enabled = True
        config.model.depthmem_num_temporal_frames = T
        config.data.depthmem_depth_dir = ft_config.depthmem_depth_dir
        config.data.depthmem_num_temporal_frames = T
        config.data.allow_padding = True  # needed for temporal window at episode start

        # Override video delta_indices to load T temporal frames
        if embodiment_tag in config.data.modality_configs:
            video_cfg = config.data.modality_configs[embodiment_tag].get("video")
            if video_cfg is not None:
                video_cfg.delta_indices = list(range(-(T - 1), 1))
                print(f"[DepthMem] Video delta_indices set to {video_cfg.delta_indices}")
        config.model.depthmem_lora_rank = ft_config.depthmem_lora_rank
        print(f"[DepthMem] Enabled: T={T}, depth_dir={ft_config.depthmem_depth_dir}, lora_rank={ft_config.depthmem_lora_rank}")

    run(config)
