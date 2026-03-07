"""Phase 2: Patch-level correction network training (v2).

Uses frozen Phase 1 modules (DINOv2 + TransitionViT) to compute per-patch
prediction errors, then trains AttentionWeightedPooling + CorrectionNetwork.

Prediction error: E = Z_{t+1} - Ẑ_{t+1}  (49×384, per-patch)
Pooled error: e = attention_pool(E) → 384-dim
Correction: Δa = correction_net(e, a_vla, proprio, chunk_step)
Target: a_expert - a_vla (actual VLA errors)

Usage:
    python cerebellar_correction/training/train_phase2.py \
        --dataset_path /mnt/md1/solee/data/bridge_lerobot \
        --intent_dir data/bridge_intents \
        --vla_action_dir data/vla_actions \
        --phase1_dir checkpoints/cerebellum_intent/phase1 \
        --output_dir checkpoints/cerebellum_intent/phase2 \
        --max_correction 0.15 --num_epochs 50
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from cerebellar_correction.data.dataset_phase2 import BridgePhase2Dataset
from cerebellar_correction.models.correction_net import AttentionWeightedPooling, CorrectionNetwork
from cerebellar_correction.models.forward_model import TransitionViT
from cerebellar_correction.models.visual_encoder import CerebellumVisualEncoder


log = logging.getLogger(__name__)


def train_phase2(
    dataset_path: str,
    intent_dir: str,
    vla_action_dir: str,
    phase1_dir: str,
    output_dir: str,
    feature_dim: int = 384,
    action_dim: int = 7,
    proprio_dim: int = 8,
    intent_dim: int = 2048,
    num_patches: int = 49,
    transition_num_layers: int = 4,
    transition_num_heads: int = 8,
    transition_ffn_dim: int = 1536,
    hidden_dim: int = 256,
    num_layers: int = 3,
    pooling_hidden_dim: int = 128,
    max_correction: float = 0.15,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 50,
    val_split: float = 0.1,
    device: str = "cuda",
    num_workers: int = 4,
    decode_workers: int = 32,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir = Path(phase1_dir)

    use_wandb = bool(wandb_project)
    if use_wandb:
        import wandb

        wandb.init(
            project=wandb_project,
            name=wandb_run_name or "phase2_patch_correction",
            config={
                "phase": "2-patch-correction-v2",
                "feature_dim": feature_dim,
                "action_dim": action_dim,
                "intent_dim": intent_dim,
                "num_patches": num_patches,
                "hidden_dim": hidden_dim,
                "pooling_hidden_dim": pooling_hidden_dim,
                "max_correction": max_correction,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
            },
        )

    # Dataset
    log.info("Loading Phase 2 dataset...")
    dataset = BridgePhase2Dataset(
        dataset_path=dataset_path,
        intent_dir=intent_dir,
        vla_action_dir=vla_action_dir,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        decode_workers=decode_workers,
    )

    # Log VLA error distribution
    errors = dataset.action_expert - dataset.action_vla
    log.info("VLA error distribution:")
    for d in range(errors.shape[1]):
        dim_errors = errors[:, d]
        log.info(
            "  Dim %d: mean=%.4f std=%.4f |mean|=%.4f p95=%.4f max=%.4f",
            d,
            dim_errors.mean().item(),
            dim_errors.std().item(),
            dim_errors.abs().mean().item(),
            dim_errors.abs().quantile(0.95).item(),
            dim_errors.abs().max().item(),
        )

    within_bound = (errors.abs() <= max_correction).float().mean()
    log.info(
        "max_correction=%.3f covers %.1f%% of VLA errors", max_correction, within_bound.item() * 100
    )

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    log.info("Train: %d pairs, Val: %d pairs", train_size, val_size)

    # === Frozen models from Phase 1 ===
    # Frozen DINOv2 encoder (patch-level)
    visual_encoder = CerebellumVisualEncoder(
        backbone="dinov2_vits14",
        use_lora=False,
    ).to(device)
    visual_encoder.eval()
    for p in visual_encoder.parameters():
        p.requires_grad = False
    log.info("Frozen DINOv2 encoder loaded")

    # Frozen Transition ViT from Phase 1
    transition_vit = TransitionViT(
        feature_dim=feature_dim,
        proprio_dim=proprio_dim,
        intent_dim=intent_dim,
        num_patches=num_patches,
        num_layers=transition_num_layers,
        num_heads=transition_num_heads,
        ffn_dim=transition_ffn_dim,
    ).to(device)
    vit_path = phase1_dir / "transition_vit_best.pt"
    transition_vit.load_state_dict(
        torch.load(vit_path, map_location=device, weights_only=True)
    )
    transition_vit.eval()
    for p in transition_vit.parameters():
        p.requires_grad = False
    log.info("Loaded frozen TransitionViT from %s", vit_path)

    # === Trainable modules ===
    # Attention-weighted pooling
    error_pooling = AttentionWeightedPooling(
        feature_dim=feature_dim,
        hidden_dim=pooling_hidden_dim,
    ).to(device)

    # Correction network
    correction_net = CorrectionNetwork(
        feature_dim=feature_dim,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_correction=max_correction,
    ).to(device)

    pooling_count = sum(p.numel() for p in error_pooling.parameters())
    correction_count = sum(p.numel() for p in correction_net.parameters())
    log.info("Trainable params — Pooling: %d, Correction: %d, Total: %d",
             pooling_count, correction_count, pooling_count + correction_count)

    trainable_params = list(error_pooling.parameters()) + list(correction_net.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # === Training ===
        error_pooling.train()
        correction_net.train()
        train_loss = 0.0
        train_steps = 0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=False
        ):
            frames_t = batch["frame_t"].to(device, dtype=torch.float32).div_(255.0)
            frames_t1 = batch["frame_t1"].to(device, dtype=torch.float32).div_(255.0)
            action_expert = batch["action_expert"].to(device)
            action_vla = batch["action_vla"].to(device)
            proprio_t = batch["proprio_t"].to(device)
            intent_tokens = batch["intent_tokens"].to(device).float()
            attention_mask = batch["attention_mask"].to(device)
            chunk_step = batch["chunk_step"].to(device)

            correction_target = action_expert - action_vla

            # Compute patch-level prediction error (frozen)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z_t_patches = visual_encoder.forward_patches(frames_t)    # (B, 49, 384)
                z_t1_patches = visual_encoder.forward_patches(frames_t1)  # (B, 49, 384)

                z_t1_pred = transition_vit(
                    z_t_patches, proprio_t, intent_tokens, attention_mask
                )

                # Per-patch prediction error
                patch_error = (z_t1_patches - z_t1_pred).float()  # (B, 49, 384)

            # Trainable: pooling + correction
            prediction_error = error_pooling(patch_error)  # (B, 384)

            delta_a = correction_net(
                prediction_error=prediction_error,
                action_vla=action_vla,
                proprio=proprio_t,
                chunk_step=chunk_step,
            )

            loss = F.mse_loss(delta_a, correction_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train = train_loss / max(train_steps, 1)

        # === Validation ===
        error_pooling.eval()
        correction_net.eval()
        val_loss = 0.0
        val_steps = 0
        val_correction_norms = []
        val_pooling_weights = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [val]", leave=False
            ):
                frames_t = batch["frame_t"].to(device, dtype=torch.float32).div_(255.0)
                frames_t1 = batch["frame_t1"].to(device, dtype=torch.float32).div_(255.0)
                action_expert = batch["action_expert"].to(device)
                action_vla = batch["action_vla"].to(device)
                proprio_t = batch["proprio_t"].to(device)
                intent_tokens = batch["intent_tokens"].to(device).float()
                attention_mask = batch["attention_mask"].to(device)
                chunk_step = batch["chunk_step"].to(device)

                correction_target = action_expert - action_vla

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z_t_patches = visual_encoder.forward_patches(frames_t)
                    z_t1_patches = visual_encoder.forward_patches(frames_t1)

                    z_t1_pred = transition_vit(
                        z_t_patches, proprio_t, intent_tokens, attention_mask
                    )
                    patch_error = (z_t1_patches - z_t1_pred).float()

                prediction_error = error_pooling(patch_error)

                delta_a = correction_net(
                    prediction_error=prediction_error,
                    action_vla=action_vla,
                    proprio=proprio_t,
                    chunk_step=chunk_step,
                )

                loss = F.mse_loss(delta_a, correction_target)
                val_loss += loss.item()
                val_steps += 1
                val_correction_norms.append(delta_a.norm(dim=-1))

                # Track pooling attention weights
                scores = error_pooling.score_mlp(patch_error)
                weights = F.softmax(scores, dim=1).squeeze(-1)  # (B, 49)
                val_pooling_weights.append(weights)

        avg_val = val_loss / max(val_steps, 1)

        all_norms = torch.cat(val_correction_norms)
        correction_cov = all_norms.std() / (all_norms.mean() + 1e-8)

        # Pooling weight statistics
        all_weights = torch.cat(val_pooling_weights)  # (N, 49)
        weight_entropy = -(all_weights * (all_weights + 1e-8).log()).sum(dim=-1).mean()
        top_patch = all_weights.mean(dim=0).argmax().item()

        log.info(
            "Epoch %d/%d | Train=%.6f | Val=%.6f | CoV=%.3f | "
            "Entropy=%.2f TopPatch=%d | LR=%.2e",
            epoch + 1, num_epochs,
            avg_train, avg_val, correction_cov.item(),
            weight_entropy.item(), top_patch, scheduler.get_last_lr()[0],
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/correction_loss": avg_train,
                    "val/correction_loss": avg_val,
                    "val/correction_l2_cov": correction_cov.item(),
                    "val/correction_l2_mean": all_norms.mean().item(),
                    "val/pooling_entropy": weight_entropy.item(),
                    "val/pooling_top_patch": top_patch,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch + 1,
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(error_pooling.state_dict(), output_dir / "error_pooling_best.pt")
            torch.save(correction_net.state_dict(), output_dir / "correction_net_best.pt")
            log.info("  -> Best model saved (val_loss=%.6f)", avg_val)

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "error_pooling": error_pooling.state_dict(),
                    "correction_net": correction_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                output_dir / f"checkpoint_epoch{epoch + 1}.pt",
            )

    torch.save(error_pooling.state_dict(), output_dir / "error_pooling_final.pt")
    torch.save(correction_net.state_dict(), output_dir / "correction_net_final.pt")
    log.info("Phase 2 training complete. Best val_loss=%.6f", best_val_loss)

    if use_wandb:
        wandb.log({"best_val_loss": best_val_loss})
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Phase 2: Patch-level correction training (v2)")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--intent_dir", type=str, required=True)
    parser.add_argument("--vla_action_dir", type=str, required=True)
    parser.add_argument("--phase1_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--feature_dim", type=int, default=384)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--proprio_dim", type=int, default=8)
    parser.add_argument("--intent_dim", type=int, default=2048)
    parser.add_argument("--num_patches", type=int, default=49)
    parser.add_argument("--transition_num_layers", type=int, default=4)
    parser.add_argument("--transition_num_heads", type=int, default=8)
    parser.add_argument("--transition_ffn_dim", type=int, default=1536)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--pooling_hidden_dim", type=int, default=128)
    parser.add_argument("--max_correction", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--decode_workers", type=int, default=32)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    args = parser.parse_args()

    train_phase2(**vars(args))
