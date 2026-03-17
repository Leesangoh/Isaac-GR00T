"""Sanity check: verify extracted V-JEPA 2 features are diverse (not collapsed)."""

import argparse
from pathlib import Path
import random

from safetensors.torch import load_file
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", type=str, default="/mnt/md1/solee/features/vjepa2_vitl/")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--max_cosine_sim", type=float, default=0.95)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    files = sorted(features_dir.glob("*.safetensors"))
    print(f"Found {len(files)} feature files")

    if len(files) == 0:
        print("FAIL: No feature files found")
        return

    vectors = []
    sampled_files = random.sample(files, min(len(files), args.num_samples * 2))

    for f in sampled_files:
        if len(vectors) >= args.num_samples:
            break
        data = load_file(str(f))
        for key, val in data.items():
            if key.startswith(f"layer_{args.layer}_window_"):
                vectors.append(val.float())
                if len(vectors) >= args.num_samples:
                    break

    print(f"Collected {len(vectors)} feature vectors")

    if len(vectors) < 2:
        print("FAIL: Not enough vectors to compute similarity")
        return

    vecs = torch.stack(vectors)
    vecs = F.normalize(vecs, dim=-1)

    n = len(vecs)
    num_pairs = min(10000, n * (n - 1) // 2)
    idx1 = torch.randint(0, n, (num_pairs,))
    idx2 = torch.randint(0, n, (num_pairs,))
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    sims = (vecs[idx1] * vecs[idx2]).sum(dim=-1)

    print(f"\nCosine similarity statistics (layer {args.layer}):")
    print(f"  Mean: {sims.mean().item():.4f}")
    print(f"  Std:  {sims.std().item():.4f}")
    print(f"  Min:  {sims.min().item():.4f}")
    print(f"  Max:  {sims.max().item():.4f}")

    if sims.mean().item() > args.max_cosine_sim:
        print(f"\nFAIL: Mean cosine similarity {sims.mean():.4f} > {args.max_cosine_sim}")
        print("Features may have collapsed!")
    else:
        print(f"\nPASS: Features are diverse (mean cosine sim < {args.max_cosine_sim})")


if __name__ == "__main__":
    main()
