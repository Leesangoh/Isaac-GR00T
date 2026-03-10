"""
Offline depth map generation for DepthMem training.

Generates temporally consistent depth maps using Video Depth Anything Small
for all RGB frames in a LeRobot V2 dataset (bridge_lerobot format).

Processes episodes as complete units to ensure temporal consistency.
Output mirrors the video directory structure with .npy depth files per frame.

Usage:
    python scripts/generate_depth_maps.py \
        --data_dir /mnt/md1/solee/data/bridge_lerobot \
        --output_dir /mnt/md1/solee/bridge_depth_maps \
        --video_key observation.images.image_0 \
        --resolution 224 \
        --num_workers 1

Output structure:
    output_dir/
      chunk-000/
        episode_000000/
          frame_000000.npy   # float16, [H, W], range [0,1]
          frame_000001.npy
          ...
          depth_stats.npz    # d_min, d_max, num_frames
          .done              # marker file
        episode_000001/
        ...
      chunk-001/
        ...
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_vda_model(model_size: str = "small", device: str = "cuda"):
    """Load Video Depth Anything model."""
    vda_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Video-Depth-Anything")
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

    ckpt_path = os.path.join(vda_path, "checkpoints", f"video_depth_anything_{encoder_map[model_size]}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VDA checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()

    # torch.compile for faster inference
    try:
        model.forward = torch.compile(model.forward, mode="max-autotune")
        print("torch.compile applied (max-autotune)")
    except Exception as e:
        print(f"torch.compile failed, using eager mode: {e}")

    return model


def read_video_frames(video_path: str):
    """Read video frames using imageio+ffmpeg (handles AV1 codec)."""
    import imageio
    reader = imageio.get_reader(video_path)
    frames = [frame for frame in reader]
    reader.close()
    return np.array(frames)  # [T, H, W, 3]


def process_episode(
    model,
    video_path: str,
    output_dir: str,
    resolution: int = 224,
    device: str = "cuda",
):
    """Process a single episode video: read frames, generate depth, save per-frame .npy."""
    os.makedirs(output_dir, exist_ok=True)

    # Check if already done
    done_marker = os.path.join(output_dir, ".done")
    if os.path.exists(done_marker):
        return 0, True  # already processed

    # Read video frames
    frames = read_video_frames(video_path)  # [T, H, W, 3]
    num_frames = len(frames)
    if num_frames == 0:
        return 0, False

    # Run Video Depth Anything
    with torch.no_grad():
        depths, _ = model.infer_video_depth(
            frames, target_fps=-1, input_size=resolution, device=device
        )
    # depths: [T, H, W] float64

    # Normalize per episode to [0, 1]
    d_min = float(depths.min())
    d_max = float(depths.max())
    if d_max - d_min > 1e-6:
        depths_norm = (depths - d_min) / (d_max - d_min)
    else:
        depths_norm = np.zeros_like(depths)

    # Save each frame as float16 .npy
    for i in range(num_frames):
        np.save(
            os.path.join(output_dir, f"frame_{i:06d}.npy"),
            depths_norm[i].astype(np.float16),
        )

    # Save stats
    np.savez(
        os.path.join(output_dir, "depth_stats.npz"),
        d_min=d_min, d_max=d_max, num_frames=num_frames,
    )

    # Mark done
    Path(done_marker).touch()
    return num_frames, False


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for DepthMem training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to LeRobot V2 dataset (e.g., /mnt/md1/solee/data/bridge_lerobot)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for depth maps")
    parser.add_argument("--video_key", type=str, default="observation.images.image_0",
                        help="Video key to process")
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["small", "base", "large"])
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_chunk", type=int, default=0)
    parser.add_argument("--end_chunk", type=int, default=-1,
                        help="End chunk (exclusive), -1 for all")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU index for multi-GPU parallel. Auto-splits chunks across GPUs.")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Total number of GPUs for parallel processing.")
    args = parser.parse_args()

    # Multi-GPU: auto-assign chunk range per GPU
    if args.gpu_id is not None:
        args.device = f"cuda:{args.gpu_id}"

    # Read dataset info
    info_path = os.path.join(args.data_dir, "meta", "info.json")
    with open(info_path) as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    chunks_size = info["chunks_size"]
    total_chunks = info["total_chunks"]
    video_path_template = info["video_path"]

    end_chunk = args.end_chunk if args.end_chunk > 0 else total_chunks

    # Multi-GPU: split chunk range evenly
    if args.gpu_id is not None and args.num_gpus > 1:
        all_chunks = list(range(args.start_chunk, end_chunk))
        per_gpu = len(all_chunks) // args.num_gpus
        remainder = len(all_chunks) % args.num_gpus
        start = args.gpu_id * per_gpu + min(args.gpu_id, remainder)
        end = start + per_gpu + (1 if args.gpu_id < remainder else 0)
        args.start_chunk = all_chunks[start] if start < len(all_chunks) else end_chunk
        end_chunk = all_chunks[end - 1] + 1 if end <= len(all_chunks) else end_chunk
        print(f"GPU {args.gpu_id}/{args.num_gpus}: chunks [{args.start_chunk}, {end_chunk})")

    print(f"Dataset: {total_episodes} episodes, {total_chunks} chunks, chunk_size={chunks_size}")
    print(f"Processing chunks [{args.start_chunk}, {end_chunk})")
    print(f"Video key: {args.video_key}")
    print(f"Resolution: {args.resolution}")
    print(f"Output: {args.output_dir}")
    print()

    # Load model
    print(f"Loading Video Depth Anything ({args.model_size})...")
    model = load_vda_model(args.model_size, args.device)
    print("Model loaded.\n")

    os.makedirs(args.output_dir, exist_ok=True)

    total_frames_processed = 0
    total_skipped = 0
    total_errors = 0
    start_time = time.time()

    for chunk_idx in range(args.start_chunk, end_chunk):
        chunk_name = f"chunk-{chunk_idx:03d}"
        chunk_video_dir = os.path.join(args.data_dir, "videos", chunk_name, args.video_key)

        if not os.path.isdir(chunk_video_dir):
            print(f"Skipping {chunk_name}: no video dir")
            continue

        # List episodes in this chunk
        episode_files = sorted([
            f for f in os.listdir(chunk_video_dir) if f.endswith(".mp4")
        ])

        pbar = tqdm(episode_files, desc=f"{chunk_name}", unit="ep")
        for ep_file in pbar:
            ep_name = ep_file.replace(".mp4", "")
            video_path = os.path.join(chunk_video_dir, ep_file)
            ep_output_dir = os.path.join(args.output_dir, chunk_name, ep_name)

            try:
                n_frames, was_cached = process_episode(
                    model, video_path, ep_output_dir,
                    resolution=args.resolution, device=args.device,
                )
                if was_cached:
                    total_skipped += 1
                else:
                    total_frames_processed += n_frames
            except Exception as e:
                total_errors += 1
                tqdm.write(f"  ERROR {ep_name}: {e}")
                continue

            elapsed = time.time() - start_time
            pbar.set_postfix(
                frames=total_frames_processed,
                skip=total_skipped,
                err=total_errors,
                fps=f"{total_frames_processed / max(elapsed, 1):.0f}",
            )

    elapsed = time.time() - start_time
    print(f"\nDone! {total_frames_processed} frames in {elapsed:.0f}s "
          f"({total_frames_processed / max(elapsed, 1):.0f} frames/s)")
    print(f"Skipped (cached): {total_skipped}, Errors: {total_errors}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
