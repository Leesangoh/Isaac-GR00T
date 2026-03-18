#!/usr/bin/env python
"""Compute per-layer global mean vectors from pre-extracted V-JEPA 2 features.

Outputs a single `global_means.pt` file per model containing:
    {layer_id (int): mean_vector (1-D tensor)}

Usage:
    python scripts/physrepa/compute_global_means.py \
        --features-dir /mnt/md1/solee/features/vjepa2_vitl \
        --layers 8 10 12 23

    python scripts/physrepa/compute_global_means.py \
        --features-dir /mnt/md1/solee/features/vjepa2_vitg \
        --layers 13 16 20 39
"""

import argparse
from pathlib import Path

from safetensors.torch import load_file
import torch
import torch.nn.functional as F


def compute_global_means(features_dir: Path, layers: list[int]) -> dict[int, torch.Tensor]:
    """Compute global mean vector per layer across all samples and windows."""
    files = sorted(features_dir.glob("*.safetensors"))
    print(f"Found {len(files)} feature files in {features_dir}")

    # Accumulators: running sum + count per layer
    running_sum: dict[int, torch.Tensor] = {}
    running_count: dict[int, int] = {}

    for i, f in enumerate(files):
        data = load_file(f)
        for layer_id in layers:
            # Collect all windows for this layer
            for key, tensor in data.items():
                if key.startswith(f"layer_{layer_id}_window_"):
                    vec = tensor.float()  # fp16 -> fp32 for precision
                    if layer_id not in running_sum:
                        running_sum[layer_id] = torch.zeros_like(vec)
                        running_count[layer_id] = 0
                    running_sum[layer_id] += vec
                    running_count[layer_id] += 1

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(files)} files...")

    # Compute means
    global_means = {}
    for layer_id in layers:
        count = running_count[layer_id]
        mean = running_sum[layer_id] / count
        global_means[layer_id] = mean
        print(f"  Layer {layer_id}: mean computed from {count} vectors, norm={mean.norm():.4f}")

    return global_means


def sanity_check(features_dir: Path, global_means: dict[int, torch.Tensor], layers: list[int]):
    """Quick sanity check: sample centered features should have ~0 mean cosine similarity."""
    import random

    files = sorted(features_dir.glob("*.safetensors"))
    sample_files = random.sample(files, min(200, len(files)))

    for layer_id in layers:
        centered_vecs = []
        for f in sample_files:
            data = load_file(f)
            key = f"layer_{layer_id}_window_0"
            if key in data:
                vec = data[key].float() - global_means[layer_id]
                centered_vecs.append(vec)

        if len(centered_vecs) < 2:
            continue

        vecs = torch.stack(centered_vecs)
        vecs_normed = F.normalize(vecs, dim=-1)

        # Pairwise cosine similarity (sample 500 pairs)
        n = len(vecs_normed)
        indices = torch.randint(0, n, (500, 2))
        cos_sims = F.cosine_similarity(
            vecs_normed[indices[:, 0]], vecs_normed[indices[:, 1]], dim=-1
        )
        print(
            f"  Layer {layer_id} centered cosine sim: "
            f"mean={cos_sims.mean():.4f}, std={cos_sims.std():.4f}, "
            f"min={cos_sims.min():.4f}, max={cos_sims.max():.4f}"
        )

        # Also check raw (un-centered) for comparison
        raw_vecs = []
        for f in sample_files:
            data = load_file(f)
            key = f"layer_{layer_id}_window_0"
            if key in data:
                raw_vecs.append(data[key].float())
        raw = torch.stack(raw_vecs)
        raw_normed = F.normalize(raw, dim=-1)
        raw_cos = F.cosine_similarity(raw_normed[indices[:, 0]], raw_normed[indices[:, 1]], dim=-1)
        print(
            f"  Layer {layer_id} RAW cosine sim:      "
            f"mean={raw_cos.mean():.4f}, std={raw_cos.std():.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute global mean vectors for V-JEPA 2 features"
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory containing pre-extracted .safetensors feature files",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layer IDs to compute means for",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for global_means.pt (default: <features-dir>/global_means.pt)",
    )
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_path = Path(args.output) if args.output else features_dir / "global_means.pt"

    print(f"Computing global means for layers {args.layers}")
    print(f"Features directory: {features_dir}")
    print(f"Output: {output_path}")
    print()

    global_means = compute_global_means(features_dir, args.layers)

    print(f"\nSaving to {output_path}")
    torch.save(global_means, output_path)

    print("\n--- Sanity check: centered vs raw cosine similarity ---")
    sanity_check(features_dir, global_means, args.layers)

    print("\nDone!")


if __name__ == "__main__":
    main()
