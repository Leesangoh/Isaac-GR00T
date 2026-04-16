"""Offline V-JEPA 2 feature extraction for PhysREPA.

Extracts hidden-state features from V-JEPA 2 (Large or Giant) at specified
layers for all episodes in a LeRobot-format dataset. Features are saved as
per-episode safetensors files.

Usage (Large):
    python scripts/physrepa/extract_vjepa2_features.py \
        --model_size large \
        --dataset_path /mnt/md1/solee/data/bridge_lerobot \
        --output_dir /mnt/md1/solee/features/vjepa2_vitl/ \
        --checkpoint /mnt/md1/solee/checkpoints/vjepa2/vitl.pt

Usage (Giant):
    python scripts/physrepa/extract_vjepa2_features.py \
        --model_size giant \
        --dataset_path /mnt/md1/solee/data/bridge_lerobot \
        --output_dir /mnt/md1/solee/features/vjepa2_vitg/ \
        --checkpoint /mnt/md1/solee/checkpoints/vjepa2/vitg-384.pt
"""

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from safetensors.torch import save_file
import torch
from torchvision import transforms
from tqdm import tqdm


# Model configurations
MODEL_CONFIGS = {
    "large": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "img_size": 256,
        "default_out_layers": [8, 10, 12, 23],
        "extra_kwargs": {
            "use_sdpa": True,
            "use_SiLU": False,
            "wide_SiLU": True,
            "uniform_power": False,
            "use_rope": True,
        },
        "checkpoint_key": "target_encoder",
    },
    "giant": {
        "embed_dim": 1408,
        "depth": 40,
        "num_heads": 22,
        "img_size": 384,
        "default_out_layers": [13, 16, 20, 39],
        "extra_kwargs": {
            "use_sdpa": True,
            "use_SiLU": False,
            "wide_SiLU": True,
            "uniform_power": False,
            "use_rope": True,
        },
        "checkpoint_key": "target_encoder",
    },
}

ARCH_FACTORY = {
    "large": "vit_large",
    "giant": "vit_giant_xformers",
}


def load_vjepa2_model(checkpoint_path, model_size, out_layers, device):
    """Load V-JEPA 2 model (Large or Giant)."""
    import sys

    sys.path.insert(0, "/home/solee/vjepa2")
    sys.path.insert(0, "/home/solee/vjepa2/src")

    import models.vision_transformer as vit_module

    cfg = MODEL_CONFIGS[model_size]
    factory_fn = getattr(vit_module, ARCH_FACTORY[model_size])
    model = factory_fn(
        patch_size=16,
        img_size=(cfg["img_size"], cfg["img_size"]),
        num_frames=64,
        tubelet_size=2,
        out_layers=out_layers,
        **cfg["extra_kwargs"],
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    ckpt_key = cfg["checkpoint_key"]
    if ckpt_key in state_dict:
        state_dict = state_dict[ckpt_key]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    elif "encoder" in state_dict:
        state_dict = state_dict["encoder"]

    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    model = model.to(device).eval()

    print(
        f"Loaded V-JEPA 2 {model_size}: embed_dim={cfg['embed_dim']}, "
        f"depth={cfg['depth']}, img_size={cfg['img_size']}, out_layers={out_layers}"
    )
    return model, cfg["img_size"]


def get_episode_video_paths(dataset_path):
    """Get episode count and video dir from a LeRobot dataset."""
    dataset_path = Path(dataset_path)
    meta_path = dataset_path / "meta"

    episodes_path = meta_path / "episodes.jsonl"
    if not episodes_path.exists():
        import pandas as pd

        episodes_path = meta_path / "episodes.parquet"
        episodes = pd.read_parquet(episodes_path)
        num_episodes = len(episodes)
    else:
        with open(episodes_path) as f:
            episodes = [json.loads(line) for line in f]
        num_episodes = len(episodes)

    video_dir = dataset_path / "videos"
    if not video_dir.exists():
        raise FileNotFoundError(f"No videos directory found at {video_dir}")

    return num_episodes, video_dir


def find_video_path(video_dir, ep_idx, video_key="observation.images.image_0"):
    """Find video file for an episode, supporting LeRobot chunked layout."""
    ep_filename = f"episode_{ep_idx:06d}.mp4"

    # Try chunked layout first (1000 episodes per chunk)
    chunk_idx = ep_idx // 1000
    chunk_dir = video_dir / f"chunk-{chunk_idx:03d}"
    if chunk_dir.exists():
        video_path = chunk_dir / video_key / ep_filename
        if video_path.exists():
            return video_path

    # Fallback: try chunk-000 (merged datasets may put all episodes in chunk-000)
    if chunk_idx != 0:
        chunk0_dir = video_dir / "chunk-000"
        if chunk0_dir.exists():
            video_path = chunk0_dir / video_key / ep_filename
            if video_path.exists():
                return video_path

    # Try flat layout
    video_path = video_dir / video_key / ep_filename
    if video_path.exists():
        return video_path

    video_path = video_dir / ep_filename
    if video_path.exists():
        return video_path

    return None


def load_video_frames(video_path, max_frames=None):
    """Load video frames using ffmpeg subprocess (supports AV1 etc.)."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True, text=True,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        print(f"Failed to probe video {video_path}")
        return None

    parts = probe.stdout.strip().split(",")
    w, h = int(parts[0]), int(parts[1])

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"Failed to decode video {video_path}: {result.stderr.decode()[:200]}")
        return None

    raw = result.stdout
    frame_size = w * h * 3
    num_frames = len(raw) // frame_size
    if num_frames == 0:
        return None

    if max_frames:
        num_frames = min(num_frames, max_frames)

    frames = np.frombuffer(raw[: num_frames * frame_size], dtype=np.uint8).reshape(
        num_frames, h, w, 3
    )
    return torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)


def extract_features_for_episode(
    model, frames, out_layers, window_size=16, window_stride=4, device="cuda",
    batch_size=1,
):
    """Extract V-JEPA 2 features for one episode using sliding windows (batched)."""
    T = frames.shape[0]
    if T < window_size:
        pad = window_size - T
        frames = torch.cat([frames[:1].expand(pad, -1, -1, -1), frames], dim=0)
        T = window_size

    window_starts = list(range(0, T - window_size + 1, window_stride))
    if not window_starts:
        window_starts = [0]

    results = {}
    results["window_starts"] = torch.tensor(window_starts, dtype=torch.int32)

    # Process clips in batches for GPU throughput
    for batch_start in range(0, len(window_starts), batch_size):
        batch_end = min(batch_start + batch_size, len(window_starts))
        batch_clips = []
        for w_idx in range(batch_start, batch_end):
            start = window_starts[w_idx]
            clip = frames[start : start + window_size]
            clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
            batch_clips.append(clip)

        batch = torch.stack(batch_clips).to(device)  # (B, C, T, H, W)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(batch)

        for b_offset in range(batch_end - batch_start):
            w_idx = batch_start + b_offset
            for layer_idx, layer_out in zip(out_layers, outputs):
                pooled = layer_out[b_offset].mean(dim=0).half().cpu()
                results[f"layer_{layer_idx}_window_{w_idx}"] = pooled

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract V-JEPA 2 features for PhysREPA")
    parser.add_argument(
        "--model_size", type=str, choices=["large", "giant"], default="large",
        help="V-JEPA 2 model size: large (ViT-L, 1024d) or giant (ViT-G, 1408d)",
    )
    parser.add_argument("--dataset_path", type=str, default="/mnt/md1/solee/data/bridge_lerobot")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out_layers", type=int, nargs="+", default=None)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--window_stride", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--end_episode", type=int, default=-1)
    parser.add_argument("--video_key", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1, help="Clips per GPU batch (higher = faster, more VRAM)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]

    if args.checkpoint is None:
        if args.model_size == "giant":
            args.checkpoint = "/mnt/md1/solee/checkpoints/vjepa2/vitg-384.pt"
        else:
            args.checkpoint = "/mnt/md1/solee/checkpoints/vjepa2/vitl.pt"

    if args.output_dir is None:
        suffix = "vitl" if args.model_size == "large" else "vitg"
        args.output_dir = f"/mnt/md1/solee/features/vjepa2_{suffix}/"

    if args.out_layers is None:
        args.out_layers = cfg["default_out_layers"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = f"cuda:{args.gpu_id}"

    print(f"Loading V-JEPA 2 {args.model_size} from {args.checkpoint}...")
    model, img_size = load_vjepa2_model(args.checkpoint, args.model_size, args.out_layers, device)

    normalize = transforms.Compose(
        [
            transforms.Resize(img_size, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    num_episodes, video_dir = get_episode_video_paths(args.dataset_path)
    video_key = args.video_key or "observation.images.image_0"
    print(f"Found {num_episodes} episodes, video_key: {video_key}")

    end_episode = args.end_episode if args.end_episode > 0 else num_episodes

    for ep_idx in tqdm(range(args.start_episode, end_episode), desc="Extracting features"):
        out_path = output_dir / f"{ep_idx:06d}.safetensors"
        if out_path.exists() and not args.overwrite:
            continue

        video_path = find_video_path(video_dir, ep_idx, video_key)
        if video_path is None:
            print(f"Video not found for episode {ep_idx}, skipping")
            continue

        frames = load_video_frames(video_path)
        if frames is None:
            print(f"Failed to load episode {ep_idx}, skipping")
            continue

        frames = torch.stack([normalize(f) for f in frames])

        results = extract_features_for_episode(
            model, frames, args.out_layers, args.window_size, args.window_stride, device,
            batch_size=args.batch_size,
        )

        save_file(results, str(out_path))

    print(f"Done! Features saved to {output_dir}")


if __name__ == "__main__":
    main()
