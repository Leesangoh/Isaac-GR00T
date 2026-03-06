"""Phase 2: Train correction network using actual VLA errors + intent.

Uses the frozen Phase 1 encoder to compute DINOv2 features on-the-fly from
raw images. The frozen intent-conditioned forward model generates prediction
errors, and the correction network learns to map them to action corrections.

Forward model: (z_t, proprio, intent) -> delta_z_ideal  (FROZEN from Phase 1)
Visual encoder: DINOv2+LoRA (FROZEN from Phase 1, on-the-fly feature computation)
Correction target: action_expert - action_vla  (actual VLA errors)
Correction input: intent-aware prediction error + action_vla + proprio + chunk_step

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
from cerebellar_correction.models.correction_net import CorrectionNetwork
from cerebellar_correction.models.forward_model import IntentForwardModel
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
    hidden_dim: int = 256,
    num_layers: int = 3,
    max_correction: float = 0.15,
    forward_hidden_dim: int = 256,
    forward_num_layers: int = 3,
    lora_rank: int = 8,
    batch_size: int = 256,
    learning_rate: float = 5e-4,
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
            name=wandb_run_name or "phase2_intent_correction",
            config={
                "phase": "2-intent",
                "feature_dim": feature_dim,
                "action_dim": action_dim,
                "intent_dim": intent_dim,
                "hidden_dim": hidden_dim,
                "max_correction": max_correction,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
            },
        )

    # Dataset (decodes frames into RAM)
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

    # Load frozen visual encoder from Phase 1
    visual_encoder = CerebellumVisualEncoder(
        backbone="dinov2_vits14",
        use_lora=True,
        lora_rank=lora_rank,
    ).to(device)
    encoder_path = phase1_dir / "encoder_best.pt"
    visual_encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    visual_encoder.eval()
    for p in visual_encoder.parameters():
        p.requires_grad = False
    log.info("Loaded frozen visual encoder from %s", encoder_path)

    # Load frozen forward models from Phase 1 (self-attention)
    forward_model = IntentForwardModel(
        feature_dim=feature_dim,
        proprio_dim=proprio_dim,
        intent_dim=intent_dim,
        num_layers=forward_num_layers,
        num_heads=6,
        ffn_dim=1536,
    ).to(device)
    forward_model.load_state_dict(
        torch.load(phase1_dir / "forward_model_best.pt", map_location=device, weights_only=True)
    )
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad = False

    log.info("Loaded frozen intent-conditioned forward model from Phase 1")

    # Correction network (to train)
    correction_net = CorrectionNetwork(
        feature_dim=feature_dim,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_correction=max_correction,
    ).to(device)

    log.info("Correction net params: %d", sum(p.numel() for p in correction_net.parameters()))

    optimizer = torch.optim.AdamW(
        correction_net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
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
            intent_tokens = batch["intent_tokens"].to(device).float()  # (B, 128, 2048)
            attention_mask = batch["attention_mask"].to(device)  # (B, 128)
            chunk_step = batch["chunk_step"].to(device)

            correction_target = action_expert - action_vla

            # Compute features on-the-fly with frozen encoder
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z_t = visual_encoder(frames_t)
                z_t1 = visual_encoder(frames_t1)

                # Self-attention forward model with full intent tokens
                delta_z_predicted = forward_model(
                    z_t.float(), proprio_t, intent_tokens, attention_mask
                )
                delta_z_actual = (z_t1 - z_t).float()
                prediction_error = delta_z_actual - delta_z_predicted

            # Correction prediction
            delta_a = correction_net(
                prediction_error=prediction_error,
                action_vla=action_vla,
                proprio=proprio_t,
                chunk_step=chunk_step,
            )

            loss = F.mse_loss(delta_a, correction_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(correction_net.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train = train_loss / max(train_steps, 1)

        # Validation
        correction_net.eval()
        val_loss = 0.0
        val_steps = 0
        val_correction_norms = []

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
                    z_t = visual_encoder(frames_t)
                    z_t1 = visual_encoder(frames_t1)

                delta_z_predicted = forward_model(
                    z_t.float(), proprio_t, intent_tokens, attention_mask
                )
                delta_z_actual = (z_t1 - z_t).float()
                prediction_error = delta_z_actual - delta_z_predicted

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

        avg_val = val_loss / max(val_steps, 1)

        all_norms = torch.cat(val_correction_norms)
        correction_cov = all_norms.std() / (all_norms.mean() + 1e-8)

        log.info(
            "Epoch %d/%d | Train loss=%.6f | Val loss=%.6f | Corr CoV=%.3f | LR=%.2e",
            epoch + 1,
            num_epochs,
            avg_train,
            avg_val,
            correction_cov.item(),
            scheduler.get_last_lr()[0],
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/correction_loss": avg_train,
                    "val/correction_loss": avg_val,
                    "val/correction_l2_cov": correction_cov.item(),
                    "val/correction_l2_mean": all_norms.mean().item(),
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch + 1,
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(correction_net.state_dict(), output_dir / "correction_net_best.pt")
            log.info("  -> Best correction model saved (val_loss=%.6f)", avg_val)

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "correction_net": correction_net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                output_dir / f"checkpoint_epoch{epoch + 1}.pt",
            )

    torch.save(correction_net.state_dict(), output_dir / "correction_net_final.pt")
    log.info("Phase 2 training complete. Best val_loss=%.6f", best_val_loss)

    if use_wandb:
        wandb.log({"best_val_loss": best_val_loss})
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Phase 2: Intent-conditioned correction training")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--intent_dir", type=str, required=True)
    parser.add_argument("--vla_action_dir", type=str, required=True)
    parser.add_argument("--phase1_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--feature_dim", type=int, default=384)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--proprio_dim", type=int, default=8)
    parser.add_argument("--intent_dim", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--max_correction", type=float, default=0.15)
    parser.add_argument(
        "--forward_hidden_dim", type=int, default=256, help="(unused, kept for compat)"
    )
    parser.add_argument("--forward_num_layers", type=int, default=2, help="Self-attention layers")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
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
