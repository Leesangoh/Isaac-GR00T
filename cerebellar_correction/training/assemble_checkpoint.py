"""Assemble Phase 1 + Phase 2 checkpoints into a single cerebellum checkpoint.

Phase 1 provides: visual encoder (DINOv2+LoRA), forward model, proprio forward model
Phase 2 provides: correction network

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

from cerebellar_correction.models.cerebellum import CerebellumConfig, IntentCerebellumModule


log = logging.getLogger(__name__)


def assemble(phase1_dir: str, phase2_dir: str, output_path: str):
    phase1_dir = Path(phase1_dir)
    phase2_dir = Path(phase2_dir)

    config = CerebellumConfig()
    module = IntentCerebellumModule(config)

    # Phase 1: visual encoder
    enc_path = phase1_dir / "encoder_best.pt"
    if enc_path.exists():
        module.visual_encoder.load_state_dict(
            torch.load(enc_path, map_location="cpu", weights_only=True)
        )
        log.info("Loaded visual encoder from %s", enc_path)

    # Phase 1: forward models
    fm_path = phase1_dir / "forward_model_best.pt"
    if fm_path.exists():
        module.forward_model.load_state_dict(
            torch.load(fm_path, map_location="cpu", weights_only=True)
        )
        log.info("Loaded forward model from %s", fm_path)

    pf_path = phase1_dir / "proprio_forward_best.pt"
    if pf_path.exists():
        module.proprio_forward.load_state_dict(
            torch.load(pf_path, map_location="cpu", weights_only=True)
        )
        log.info("Loaded proprio forward from %s", pf_path)

    # Phase 2: correction net
    cn_path = phase2_dir / "correction_net_best.pt"
    if cn_path.exists():
        module.correction_net.load_state_dict(
            torch.load(cn_path, map_location="cpu", weights_only=True)
        )
        log.info("Loaded correction net from %s", cn_path)

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
                "max_correction": config.max_correction,
                "forward_num_layers": config.forward_num_layers,
                "forward_num_heads": config.forward_num_heads,
                "max_intent_tokens": config.max_intent_tokens,
            },
        },
        output_path,
    )

    total_params = sum(p.numel() for p in module.parameters())
    log.info("Assembled checkpoint saved to %s (%d params)", output_path, total_params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Assemble cerebellum checkpoint")
    parser.add_argument("--phase1_dir", type=str, required=True)
    parser.add_argument("--phase2_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    assemble(args.phase1_dir, args.phase2_dir, args.output_path)
