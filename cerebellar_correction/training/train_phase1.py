"""Phase 1: Joint DINOv2+LoRA + intent-conditioned forward model training with EMA.

Trains:
  - Online encoder (DINOv2 ViT-S + LoRA): produces z_t with gradient flow
  - Target encoder (EMA copy, no gradient): produces z_t1_target
  - IntentForwardModel: (z_t, proprio, intent) -> delta_z_ideal
  - ProprioForwardModel: (proprio, intent) -> delta_proprio

Loss:
  L_visual = MSE(delta_z_ideal, delta_z_target)   where delta_z_target = z_t1_target - z_t.detach()
  L_proprio = MSE(delta_proprio_pred, delta_proprio_actual)
  L_vicreg = VICReg_variance(z_t)                 feature collapse prevention
  L_total = L_visual + L_proprio + lambda * L_vicreg

EMA prevents temporal collapse (tau=0.996). Separate lower LR for LoRA params.

Usage:
    python cerebellar_correction/training/train_phase1.py \
        --dataset_path /mnt/md1/solee/data/bridge_lerobot \
        --intent_dir data/bridge_intents \
        --output_dir checkpoints/cerebellum_intent/phase1 \
        --num_epochs 50 --batch_size 256
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from cerebellar_correction.data.dataset_phase1 import BridgePhase1Dataset
from cerebellar_correction.models.forward_model import IntentForwardModel, ProprioForwardModel
from cerebellar_correction.models.visual_encoder import CerebellumVisualEncoder, EMAEncoder


log = logging.getLogger(__name__)


def _get_lora_state_dict(encoder: CerebellumVisualEncoder) -> dict:
    """Extract only LoRA adapter weights from the encoder."""
    lora_state = {}
    for name, param in encoder.backbone.named_parameters():
        if "lora" in name.lower():
            lora_state[name] = param.data.clone()
    return lora_state


def _save_lora_weights(encoder: CerebellumVisualEncoder, path: Path):
    torch.save(_get_lora_state_dict(encoder), path)


def train_phase1(
    dataset_path: str,
    intent_dir: str,
    output_dir: str,
    feature_dim: int = 384,
    proprio_dim: int = 8,
    intent_dim: int = 2048,
    hidden_dim: int = 256,
    num_layers: int = 3,
    lora_rank: int = 8,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    encoder_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_epochs: int = 50,
    vicreg_lambda: float = 5.0,
    ema_tau: float = 0.996,
    val_split: float = 0.1,
    device: str = "cuda",
    num_workers: int = 4,
    decode_workers: int = 32,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(wandb_project)
    if use_wandb:
        import wandb

        wandb.init(
            project=wandb_project,
            name=wandb_run_name or "phase1_intent_ema",
            config={
                "phase": "1-intent-EMA",
                "feature_dim": feature_dim,
                "proprio_dim": proprio_dim,
                "intent_dim": intent_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "lora_rank": lora_rank,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "encoder_lr": encoder_lr,
                "vicreg_lambda": vicreg_lambda,
                "ema_tau": ema_tau,
                "num_epochs": num_epochs,
            },
        )

    # Dataset (decodes all frames into RAM)
    log.info("Loading dataset from %s (this may take a while)...", dataset_path)
    dataset = BridgePhase1Dataset(
        dataset_path=dataset_path,
        intent_dir=intent_dir,
        proprio_dim=proprio_dim,
        decode_workers=decode_workers,
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

    # === Models ===
    online_encoder = CerebellumVisualEncoder(
        backbone="dinov2_vits14",
        use_lora=True,
        lora_rank=lora_rank,
    ).to(device)
    online_encoder.train()

    # EMA target encoder
    ema_enc = EMAEncoder(online_encoder, tau=ema_tau)
    ema_enc.target_encoder.to(device)
    log.info("EMA target encoder created (tau=%.4f)", ema_tau)

    forward_model = IntentForwardModel(
        feature_dim=feature_dim,
        proprio_dim=proprio_dim,
        intent_dim=intent_dim,
        num_layers=num_layers,
        num_heads=6,
        ffn_dim=1536,
    ).to(device)

    proprio_forward = ProprioForwardModel(
        proprio_dim=proprio_dim,
        intent_dim=intent_dim,
    ).to(device)

    # Separate param groups: encoder LoRA gets lower LR
    encoder_params = []
    for name, param in online_encoder.named_parameters():
        if param.requires_grad:
            encoder_params.append(param)

    param_groups = [
        {
            "params": list(forward_model.parameters()) + list(proprio_forward.parameters()),
            "lr": learning_rate,
        },
        {
            "params": encoder_params,
            "lr": encoder_lr,
        },
    ]

    encoder_count = sum(p.numel() for p in encoder_params)
    forward_count = sum(p.numel() for p in forward_model.parameters())
    proprio_count = sum(p.numel() for p in proprio_forward.parameters())
    log.info(
        "Trainable params — Encoder: %d, Forward: %d, Proprio: %d",
        encoder_count,
        forward_count,
        proprio_count,
    )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # === Training ===
        online_encoder.train()
        forward_model.train()
        proprio_forward.train()
        train_vis_loss = 0.0
        train_prop_loss = 0.0
        train_var_loss = 0.0
        train_z_std_sum = 0.0
        train_dz_norm_sum = 0.0
        train_steps = 0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=False
        ):
            frames_t = batch["frame_t"].to(device, dtype=torch.float32).div_(255.0)
            frames_t1 = batch["frame_t1"].to(device, dtype=torch.float32).div_(255.0)
            proprio_t = batch["proprio_t"].to(device)
            proprio_t1 = batch["proprio_t1"].to(device)
            intent_tokens = batch["intent_tokens"].to(device).float()  # (B, 128, 2048)
            attention_mask = batch["attention_mask"].to(device)  # (B, 128)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Online encoder: z_t (gradient flows)
                z_t = online_encoder(frames_t)  # (B, 384)

                # EMA target encoder: z_t1_target (no gradient)
                with torch.no_grad():
                    z_t1_target = ema_enc.encode(frames_t1)  # (B, 384)

                # Forward model: self-attention with full intent token sequence
                delta_z_target = z_t1_target - z_t.detach()  # stop_grad on z_t for target
                delta_z_pred = forward_model(z_t, proprio_t, intent_tokens, attention_mask)
                visual_loss = F.mse_loss(delta_z_pred, delta_z_target)

                # Proprio forward model: uses pooled intent (MLP)
                mask_f = attention_mask.unsqueeze(-1).float()  # (B, 128, 1)
                intent_pooled = (intent_tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                delta_proprio_target = proprio_t1 - proprio_t
                delta_proprio_pred = proprio_forward(proprio_t, intent_pooled)
                proprio_loss = F.mse_loss(delta_proprio_pred, delta_proprio_target)

                # VICReg variance loss on z_t
                z_std = z_t.float().std(dim=0)  # (384,)
                variance_loss = F.relu(1.0 - z_std).mean()

                total_loss = visual_loss + proprio_loss + vicreg_lambda * variance_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            all_params = (
                list(online_encoder.parameters())
                + list(forward_model.parameters())
                + list(proprio_forward.parameters())
            )
            torch.nn.utils.clip_grad_norm_([p for p in all_params if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            ema_enc.update(online_encoder)

            with torch.no_grad():
                dz_norm = delta_z_target.float().norm(dim=-1).mean().item()

            train_vis_loss += visual_loss.item()
            train_prop_loss += proprio_loss.item()
            train_var_loss += variance_loss.item()
            train_z_std_sum += z_std.mean().item()
            train_dz_norm_sum += dz_norm
            train_steps += 1

        scheduler.step()

        avg_train_vis = train_vis_loss / max(train_steps, 1)
        avg_train_prop = train_prop_loss / max(train_steps, 1)
        avg_train_var = train_var_loss / max(train_steps, 1)
        avg_train_z_std = train_z_std_sum / max(train_steps, 1)
        avg_train_dz = train_dz_norm_sum / max(train_steps, 1)

        # === Validation ===
        online_encoder.eval()
        forward_model.eval()
        proprio_forward.eval()
        val_vis_loss = 0.0
        val_prop_loss = 0.0
        val_var_loss = 0.0
        val_z_std_sum = 0.0
        val_dz_norm_sum = 0.0
        val_baseline_loss = 0.0
        val_steps = 0

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [val]", leave=False
            ):
                frames_t = batch["frame_t"].to(device, dtype=torch.float32).div_(255.0)
                frames_t1 = batch["frame_t1"].to(device, dtype=torch.float32).div_(255.0)
                proprio_t = batch["proprio_t"].to(device)
                proprio_t1 = batch["proprio_t1"].to(device)
                intent_tokens = batch["intent_tokens"].to(device).float()
                attention_mask = batch["attention_mask"].to(device)

                z_t = online_encoder(frames_t)
                z_t1_target = ema_enc.encode(frames_t1)

                delta_z_target = z_t1_target - z_t.detach()
                delta_z_pred = forward_model(z_t, proprio_t, intent_tokens, attention_mask)
                visual_loss = F.mse_loss(delta_z_pred, delta_z_target)

                mask_f = attention_mask.unsqueeze(-1).float()
                intent_pooled = (intent_tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                delta_proprio_target = proprio_t1 - proprio_t
                delta_proprio_pred = proprio_forward(proprio_t, intent_pooled)
                proprio_loss = F.mse_loss(delta_proprio_pred, delta_proprio_target)

                z_std = z_t.float().std(dim=0)
                variance_loss = F.relu(1.0 - z_std).mean()

                dz_norm = delta_z_target.float().norm(dim=-1).mean().item()

                # Baseline: predict delta_z = 0
                baseline = F.mse_loss(torch.zeros_like(delta_z_target), delta_z_target)

                val_vis_loss += visual_loss.item()
                val_prop_loss += proprio_loss.item()
                val_var_loss += variance_loss.item()
                val_z_std_sum += z_std.mean().item()
                val_dz_norm_sum += dz_norm
                val_baseline_loss += baseline.item()
                val_steps += 1

        avg_val_vis = val_vis_loss / max(val_steps, 1)
        avg_val_prop = val_prop_loss / max(val_steps, 1)
        avg_val_var = val_var_loss / max(val_steps, 1)
        avg_val_total = avg_val_vis + avg_val_prop
        avg_val_z_std = val_z_std_sum / max(val_steps, 1)
        avg_val_dz = val_dz_norm_sum / max(val_steps, 1)
        avg_val_baseline = val_baseline_loss / max(val_steps, 1)
        improvement_pct = (avg_val_baseline - avg_val_vis) / max(avg_val_baseline, 1e-8) * 100

        log.info(
            "Epoch %d/%d | Train vis=%.6f prop=%.6f var=%.6f z_std=%.3f dz=%.3f | "
            "Val vis=%.6f prop=%.6f z_std=%.3f dz=%.3f baseline=%.6f improv=%.1f%% | LR=%.2e",
            epoch + 1,
            num_epochs,
            avg_train_vis,
            avg_train_prop,
            avg_train_var,
            avg_train_z_std,
            avg_train_dz,
            avg_val_vis,
            avg_val_prop,
            avg_val_z_std,
            avg_val_dz,
            avg_val_baseline,
            improvement_pct,
            scheduler.get_last_lr()[0],
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/visual_loss": avg_train_vis,
                    "train/proprio_loss": avg_train_prop,
                    "train/variance_loss": avg_train_var,
                    "train/z_std_mean": avg_train_z_std,
                    "train/dz_norm": avg_train_dz,
                    "val/visual_loss": avg_val_vis,
                    "val/proprio_loss": avg_val_prop,
                    "val/variance_loss": avg_val_var,
                    "val/total_loss": avg_val_total,
                    "val/z_std_mean": avg_val_z_std,
                    "val/dz_norm": avg_val_dz,
                    "val/baseline_loss": avg_val_baseline,
                    "val/improvement_pct": improvement_pct,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch + 1,
            )

        # Save best
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            # Save encoder (full state for loading in Phase 2)
            torch.save(online_encoder.state_dict(), output_dir / "encoder_best.pt")
            _save_lora_weights(online_encoder, output_dir / "encoder_lora_best.pt")
            torch.save(
                {"norm": online_encoder.norm.state_dict()},
                output_dir / "encoder_norm_best.pt",
            )
            torch.save(forward_model.state_dict(), output_dir / "forward_model_best.pt")
            torch.save(proprio_forward.state_dict(), output_dir / "proprio_forward_best.pt")
            log.info("  -> Best model saved (val_loss=%.6f)", avg_val_total)

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "online_encoder": online_encoder.state_dict(),
                    "forward_model": forward_model.state_dict(),
                    "proprio_forward": proprio_forward.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                output_dir / f"checkpoint_epoch{epoch + 1}.pt",
            )

    # Save final
    torch.save(online_encoder.state_dict(), output_dir / "encoder_final.pt")
    torch.save(forward_model.state_dict(), output_dir / "forward_model_final.pt")
    torch.save(proprio_forward.state_dict(), output_dir / "proprio_forward_final.pt")
    log.info("Phase 1 training complete. Best val_loss=%.6f", best_val_loss)

    if use_wandb:
        wandb.log({"best_val_loss": best_val_loss})
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Phase 1: Intent-conditioned forward model (EMA)")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--intent_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--feature_dim", type=int, default=384)
    parser.add_argument("--proprio_dim", type=int, default=8)
    parser.add_argument("--intent_dim", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=256, help="(unused, kept for compat)")
    parser.add_argument("--num_layers", type=int, default=2, help="Self-attention layers")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--vicreg_lambda", type=float, default=5.0)
    parser.add_argument("--ema_tau", type=float, default=0.996)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--decode_workers", type=int, default=32)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    args = parser.parse_args()

    train_phase1(**vars(args))
