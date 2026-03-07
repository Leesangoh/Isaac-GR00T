"""Assemble Phase 1 + Phase 2 checkpoints into a single cerebellum checkpoint (v2).

Phase 1 provides: TransitionViT, ProprioForwardModel
Phase 2 provides: AttentionWeightedPooling, CorrectionNetwork
Visual encoder is frozen DINOv2 (no trained weights to save).

Usage:
    python cerebellar_correction/training/assemble_checkpoint.py \
        --phase1_dir checkpoints/cerebellum_intent/phase1 \
        --phase2_dir checkpoints/cerebellum_intent/phase2 \
        --output_path checkpoints/cerebellum_intent/cerebellum_assembled.pt
"""

import argparse
import logging
from pathlib import Path

import torch

from cerebellar_correction.models.cerebellum import CerebellumConfig, PatchCerebellumModule


log = logging.getLogger(__name__)


def assemble(phase1_dir: str, phase2_dir: str, output_path: str, max_correction: float = 0.15):
    phase1_dir = Path(phase1_dir)
    phase2_dir = Path(phase2_dir)

    config = CerebellumConfig(max_correction=max_correction)
    module = PatchCerebellumModule(config)

    # Phase 1: TransitionViT
    vit_path = phase1_dir / "transition_vit_best.pt"
    if vit_path.exists():
        module.transition_vit.load_state_dict(
            torch.load(vit_path, map_location="cpu", weights_only=True)
        )
        log.info("Loaded TransitionViT from %s", vit_path)

    # Phase 1: ProprioForwardModel
    pf_path = phase1_dir / "proprio_forward_best.pt"
    if pf_path.exists():
        module.proprio_forward.load_state_dict(
            torch.load(pf_path, map_location="cpu", weights_only=True)
        )
        log.info("Loaded ProprioForwardModel from %s", pf_path)

    # Phase 2: AttentionWeightedPooling
    pool_path = phase2_dir / "error_pooling_best.pt"
    if pool_path.exists():
        module.error_pooling.load_state_dict(
            torch.load(pool_path, map_location="cpu", weights_only=True)
        )
        log.info("Loaded AttentionWeightedPooling from %s", pool_path)

    # Phase 2: CorrectionNetwork
    cn_path = phase2_dir / "correction_net_best.pt"
    if cn_path.exists():
        module.correction_net.load_state_dict(
            torch.load(cn_path, map_location="cpu", weights_only=True)
        )
        log.info("Loaded CorrectionNetwork from %s", cn_path)

    # Save assembled
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "config": {
                "feature_dim": config.feature_dim,
                "intent_dim": config.intent_dim,
                "action_dim": config.action_dim,
                "proprio_dim": config.proprio_dim,
                "num_patches": config.num_patches,
                "max_correction": config.max_correction,
                "transition_num_layers": config.transition_num_layers,
                "transition_num_heads": config.transition_num_heads,
                "transition_ffn_dim": config.transition_ffn_dim,
                "max_intent_tokens": config.max_intent_tokens,
                "pooling_hidden_dim": config.pooling_hidden_dim,
            },
        },
        output_path,
    )

    trainable_params = sum(
        p.numel() for p in module.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in module.parameters())
    log.info(
        "Assembled checkpoint saved to %s (%d trainable / %d total params)",
        output_path, trainable_params, total_params,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Assemble cerebellum checkpoint (v2)")
    parser.add_argument("--phase1_dir", type=str, required=True)
    parser.add_argument("--phase2_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_correction", type=float, default=0.15)
    args = parser.parse_args()

    assemble(args.phase1_dir, args.phase2_dir, args.output_path, args.max_correction)
