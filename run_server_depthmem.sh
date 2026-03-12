LD_PRELOAD=.venv/lib/libglibc_compat.so \
.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path /mnt/md1/solee/checkpoints/GR00T-N1.6-depthmem/checkpoint-5000 \
    --embodiment-tag OXE_WIDOWX \
    --use-sim-policy-wrapper \
    --denoising-steps ${DENOISING_STEPS:-4} \
    --depthmem \
    --depthmem-num-temporal-frames 16 \
    --depthmem-depth-model-size small \
    --depthmem-depth-resolution 224 \
    --depthmem-save-video-dir ./eval_videos
