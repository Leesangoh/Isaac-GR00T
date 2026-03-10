LD_PRELOAD=.venv/lib/libglibc_compat.so \
.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path /mnt/md1/solee/checkpoints/GR00T-N1.6-bridge \
    --embodiment-tag OXE_WIDOWX \
    --use-sim-policy-wrapper \
    --denoising-steps ${DENOISING_STEPS:-4}
# --save-attention-map --attention-map-dir ./attention_maps
