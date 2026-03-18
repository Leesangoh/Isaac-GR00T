set -x -e

export NUM_GPUS=4

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path /mnt/md1/solee/data/bridge_lerobot \
    --embodiment_tag OXE_WIDOWX \
    --num_gpus $NUM_GPUS \
    --output_dir /mnt/md1/solee/checkpoints/physrepa_vitl/ \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps 20000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 512 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --state_dropout_prob 0.8 \
    --physrepa-enabled \
    --physrepa-lambda 0.5 \
    --physrepa-vjepa-features-dir /mnt/md1/solee/features/vjepa2_vitl/ \
    --physrepa-vjepa-layer 10 \
    --physrepa-align-layers 0 1 2 3 4 5 6 7 8 \
    --physrepa-global-means-path /mnt/md1/solee/features/vjepa2_vitl/global_means.pt
