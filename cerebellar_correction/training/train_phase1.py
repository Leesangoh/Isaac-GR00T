"""Phase 1: Patch-level Transition ViT training (v2, DINO-WM inspired).

Trains:
  - TransitionViT: (patch_tokens, proprio, intent) -> predicted z_{t+1} patches
  - ProprioForwardModel: (proprio, intent_pooled) -> Δproprio

Visual encoder is FROZEN DINOv2 ViT-S/14 (no LoRA, no EMA).
Patch tokens: 49 patches × 384-dim from 98×98 input (7×7 grid).

Loss:
  L_visual = (1/49) Σ ||ẑ_{t+1}^i - z_{t+1}^i||²   (per-patch MSE)
  L_proprio = ||Δp_pred - (p_{t+1} - p_t)||²
  L_total = L_visual + λ_proprio × L_proprio

Experiment modes:
  - baseline: Original TransitionViT with fresh intent (default)
  - action_conditioned: ActionConditionedTransitionViT with expert action + stale intent
  - multiframe: SpatiotemporalTransitionViT with H-frame history + stale intent

Usage:
    # Baseline (original)
    python cerebellar_correction/training/train_phase1.py \
        --dataset_path /mnt/md1/solee/data/bridge_lerobot \
        --intent_dir data/bridge_intents \
        --output_dir checkpoints/cerebellum_intent/phase1

    # Experiment 1: Action-conditioned oracle + stale intent
    python cerebellar_correction/training/train_phase1.py \
        --dataset_path /mnt/md1/solee/data/bridge_lerobot \
        --intent_dir data/bridge_intents \
        --vla_dir data/vla_actions \
        --output_dir checkpoints/exp1_action_stale \
        --experiment_mode action_conditioned \
        --chunk_size 8

    # Experiment 2: Multi-frame H=3 + spatiotemporal + stale intent
    python cerebellar_correction/training/train_phase1.py \
        --dataset_path /mnt/md1/solee/data/bridge_lerobot \
        --intent_dir data/bridge_intents \
        --output_dir checkpoints/exp2_multiframe_stale \
        --experiment_mode multiframe \
        --chunk_size 8 \
        --history_len 3
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from cerebellar_correction.data.dataset_phase1 import BridgePhase1Dataset
from cerebellar_correction.models.forward_model import (
    ActionConditionedTransitionViT,
    ProprioForwardModel,
    SpatiotemporalTransitionViT,
    TransitionViT,
)
from cerebellar_correction.models.visual_encoder import CerebellumVisualEncoder


log = logging.getLogger(__name__)


def train_phase1(
    dataset_path: str,
    intent_dir: str,
    output_dir: str,
    cache_dir: str = "",
    feature_dim: int = 384,
    proprio_dim: int = 8,
    action_dim: int = 7,
    intent_dim: int = 2048,
    num_patches: int = 49,
    num_layers: int = 4,
    num_heads: int = 8,
    ffn_dim: int = 1536,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    intent_proj_lr: float = 5e-4,
    weight_decay: float = 0.01,
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    proprio_lambda: float = 0.1,
    val_split: float = 0.1,
    device: str = "cuda",
    num_workers: int = 4,
    decode_workers: int = 32,
    wandb_project: str = "",
    wandb_run_name: str = "",
    # === Experiment extensions ===
    experiment_mode: str = "baseline",
    chunk_size: int = 0,
    vla_dir: str = "",
    history_len: int = 1,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate experiment mode
    valid_modes = {"baseline", "action_conditioned", "multiframe"}
    if experiment_mode not in valid_modes:
        raise ValueError(f"Unknown experiment_mode '{experiment_mode}'. Must be one of {valid_modes}")

    if experiment_mode == "action_conditioned" and not vla_dir:
        raise ValueError("action_conditioned mode requires --vla_dir")
    if experiment_mode == "multiframe" and history_len < 2:
        raise ValueError("multiframe mode requires --history_len >= 2")

    log.info("=== Experiment mode: %s ===", experiment_mode)
    if chunk_size > 0:
        log.info("Stale intent enabled: chunk_size=%d", chunk_size)

    use_wandb = bool(wandb_project)
    if use_wandb:
        import wandb

        run_name = wandb_run_name or f"phase1_{experiment_mode}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "phase": "1-patch-vit-v2",
                "experiment_mode": experiment_mode,
                "feature_dim": feature_dim,
                "proprio_dim": proprio_dim,
                "action_dim": action_dim,
                "intent_dim": intent_dim,
                "num_patches": num_patches,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "ffn_dim": ffn_dim,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "intent_proj_lr": intent_proj_lr,
                "weight_decay": weight_decay,
                "warmup_epochs": warmup_epochs,
                "proprio_lambda": proprio_lambda,
                "num_epochs": num_epochs,
                "chunk_size": chunk_size,
                "history_len": history_len,
            },
        )

    # === Dataset ===
    log.info("Loading dataset from %s...", dataset_path)
    dataset = BridgePhase1Dataset(
        dataset_path=dataset_path,
        intent_dir=intent_dir,
        proprio_dim=proprio_dim,
        decode_workers=decode_workers,
        cache_dir=cache_dir,
        chunk_size=chunk_size,
        vla_dir=vla_dir if experiment_mode == "action_conditioned" else "",
        history_len=history_len if experiment_mode == "multiframe" else 1,
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
    # Frozen DINOv2 encoder (patch-level output)
    encoder = CerebellumVisualEncoder(
        backbone="dinov2_vits14",
        use_lora=False,
        input_size=98,
    ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    log.info("Frozen DINOv2 encoder loaded (21M params, no LoRA)")

    # Select transition model based on experiment mode
    if experiment_mode == "action_conditioned":
        transition_model = ActionConditionedTransitionViT(
            feature_dim=feature_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            intent_dim=intent_dim,
            num_patches=num_patches,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_intent_tokens=128,
        ).to(device)
        model_name = "ActionConditionedTransitionViT"
    elif experiment_mode == "multiframe":
        transition_model = SpatiotemporalTransitionViT(
            feature_dim=feature_dim,
            proprio_dim=proprio_dim,
            intent_dim=intent_dim,
            num_patches=num_patches,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_intent_tokens=128,
            history_len=history_len,
        ).to(device)
        model_name = "SpatiotemporalTransitionViT"
    else:
        transition_model = TransitionViT(
            feature_dim=feature_dim,
            proprio_dim=proprio_dim,
            intent_dim=intent_dim,
            num_patches=num_patches,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            max_intent_tokens=128,
        ).to(device)
        model_name = "TransitionViT"

    # Proprio forward model (uses pooled intent)
    proprio_forward = ProprioForwardModel(
        proprio_dim=proprio_dim,
        intent_dim=intent_dim,
    ).to(device)

    model_count = sum(p.numel() for p in transition_model.parameters())
    proprio_count = sum(p.numel() for p in proprio_forward.parameters())
    log.info(
        "%s params: %d, Proprio: %d, Total: %d",
        model_name, model_count, proprio_count, model_count + proprio_count,
    )

    # Optimizer with separate lr for intent projection
    intent_proj_params = list(transition_model.intent_proj.parameters())
    intent_proj_ids = {id(p) for p in intent_proj_params}
    model_other_params = [p for p in transition_model.parameters() if id(p) not in intent_proj_ids]

    param_groups = [
        {"params": model_other_params, "lr": learning_rate},
        {"params": intent_proj_params, "lr": intent_proj_lr},
        {"params": list(proprio_forward.parameters()), "lr": learning_rate},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # Warmup + cosine schedule
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")

    # =================================================================
    # Helper: run transition model forward (handles all experiment modes)
    # =================================================================
    def forward_transition(batch_data, z_t_patches):
        """Run transition model forward, dispatching by experiment mode.

        Args:
            batch_data: dict with all batch tensors on device
            z_t_patches: (B, 49, 384) current frame's patch tokens

        Returns:
            z_t1_pred: (B, 49, 384) predicted next-state patches
        """
        intent_tokens = batch_data["intent_tokens"]
        attention_mask = batch_data["attention_mask"]
        proprio_t = batch_data["proprio_t"]

        if experiment_mode == "action_conditioned":
            return transition_model(
                z_t_patches, proprio_t, batch_data["action"],
                intent_tokens, attention_mask,
            )
        elif experiment_mode == "multiframe":
            # Encode all history frames
            history_frames = batch_data["history_frames"]  # (B, H, 3, h, w)
            B, H, C, h, w = history_frames.shape
            # Flatten for batch encoding
            flat_frames = history_frames.reshape(B * H, C, h, w)
            with torch.no_grad():
                flat_patches = encoder.forward_patches(flat_frames)  # (B*H, 49, 384)
            history_patches = flat_patches.reshape(B, H, num_patches, feature_dim)

            return transition_model(
                history_patches, proprio_t,
                intent_tokens, attention_mask,
            )
        else:
            return transition_model(
                z_t_patches, proprio_t,
                intent_tokens, attention_mask,
            )

    # =================================================================
    # Training loop
    # =================================================================
    for epoch in range(num_epochs):
        transition_model.train()
        proprio_forward.train()
        train_vis_loss = 0.0
        train_prop_loss = 0.0
        train_steps = 0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [train]", leave=False
        ):
            # Move everything to device
            batch_data = {}
            batch_data["proprio_t"] = batch["proprio_t"].to(device)
            batch_data["proprio_t1"] = batch["proprio_t1"].to(device)
            batch_data["intent_tokens"] = batch["intent_tokens"].to(device).float()
            batch_data["attention_mask"] = batch["attention_mask"].to(device)

            frames_t = batch["frame_t"].to(device, dtype=torch.float32).div_(255.0)
            frames_t1 = batch["frame_t1"].to(device, dtype=torch.float32).div_(255.0)

            if "action" in batch:
                batch_data["action"] = batch["action"].to(device)
            if "history_frames" in batch:
                batch_data["history_frames"] = (
                    batch["history_frames"].to(device, dtype=torch.float32).div_(255.0)
                )

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Frozen encoder → patch tokens
                with torch.no_grad():
                    z_t_patches = encoder.forward_patches(frames_t)    # (B, 49, 384)
                    z_t1_patches = encoder.forward_patches(frames_t1)  # (B, 49, 384) target

                # Transition model predicts z_{t+1}
                z_t1_pred = forward_transition(batch_data, z_t_patches)

                # Per-patch MSE
                visual_loss = F.mse_loss(z_t1_pred, z_t1_patches)

                # Proprio forward model
                intent_tokens = batch_data["intent_tokens"]
                attention_mask = batch_data["attention_mask"]
                mask_f = attention_mask.unsqueeze(-1).float()
                intent_pooled = (intent_tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                delta_proprio_target = batch_data["proprio_t1"] - batch_data["proprio_t"]
                delta_proprio_pred = proprio_forward(batch_data["proprio_t"], intent_pooled)
                proprio_loss = F.mse_loss(delta_proprio_pred, delta_proprio_target)

                total_loss = visual_loss + proprio_lambda * proprio_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            all_params = list(transition_model.parameters()) + list(proprio_forward.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_vis_loss += visual_loss.item()
            train_prop_loss += proprio_loss.item()
            train_steps += 1

        scheduler.step()

        avg_train_vis = train_vis_loss / max(train_steps, 1)
        avg_train_prop = train_prop_loss / max(train_steps, 1)

        # === Validation ===
        transition_model.eval()
        proprio_forward.eval()
        val_vis_loss = 0.0
        val_prop_loss = 0.0
        val_baseline_loss = 0.0
        val_per_patch_losses = torch.zeros(num_patches, device=device)
        val_steps = 0

        # Stale intent: per-chunk-position loss tracking
        if chunk_size > 0:
            chunk_pos_loss_sum = torch.zeros(chunk_size, device=device)
            chunk_pos_count = torch.zeros(chunk_size, device=device)

        # Multiframe: spatial vs temporal attention norm tracking
        if experiment_mode == "multiframe":
            spatial_norm_sum = 0.0
            temporal_norm_sum = 0.0
            st_norm_count = 0

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [val]", leave=False
            ):
                batch_data = {}
                batch_data["proprio_t"] = batch["proprio_t"].to(device)
                batch_data["proprio_t1"] = batch["proprio_t1"].to(device)
                batch_data["intent_tokens"] = batch["intent_tokens"].to(device).float()
                batch_data["attention_mask"] = batch["attention_mask"].to(device)

                frames_t = batch["frame_t"].to(device, dtype=torch.float32).div_(255.0)
                frames_t1 = batch["frame_t1"].to(device, dtype=torch.float32).div_(255.0)

                if "action" in batch:
                    batch_data["action"] = batch["action"].to(device)
                if "history_frames" in batch:
                    batch_data["history_frames"] = (
                        batch["history_frames"].to(device, dtype=torch.float32).div_(255.0)
                    )

                z_t_patches = encoder.forward_patches(frames_t)
                z_t1_patches = encoder.forward_patches(frames_t1)

                z_t1_pred = forward_transition(batch_data, z_t_patches)

                visual_loss = F.mse_loss(z_t1_pred, z_t1_patches)

                # Per-patch loss for monitoring
                per_patch = (z_t1_pred - z_t1_patches).float().pow(2).mean(dim=(0, 2))  # (49,)
                val_per_patch_losses += per_patch

                # Baseline: predict z_{t+1} = z_t (copy current state)
                baseline = F.mse_loss(z_t_patches, z_t1_patches)

                intent_tokens = batch_data["intent_tokens"]
                attention_mask = batch_data["attention_mask"]
                mask_f = attention_mask.unsqueeze(-1).float()
                intent_pooled = (intent_tokens * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                delta_proprio_target = batch_data["proprio_t1"] - batch_data["proprio_t"]
                delta_proprio_pred = proprio_forward(batch_data["proprio_t"], intent_pooled)
                proprio_loss = F.mse_loss(delta_proprio_pred, delta_proprio_target)

                val_vis_loss += visual_loss.item()
                val_prop_loss += proprio_loss.item()
                val_baseline_loss += baseline.item()
                val_steps += 1

                # --- Extra: per-chunk-position loss ---
                if chunk_size > 0 and "chunk_position" in batch:
                    chunk_pos = batch["chunk_position"].to(device)  # (B,)
                    per_sample_loss = (z_t1_pred - z_t1_patches).float().pow(2).mean(dim=(1, 2))
                    for cp in range(chunk_size):
                        mask_cp = chunk_pos == cp
                        if mask_cp.any():
                            chunk_pos_loss_sum[cp] += per_sample_loss[mask_cp].sum()
                            chunk_pos_count[cp] += mask_cp.sum()

                # --- Extra: spatial vs temporal attention norms (multiframe) ---
                if experiment_mode == "multiframe" and val_steps <= 1:
                    # Probe first batch: run one temporal layer to get spatial/temporal norms
                    history_frames = batch_data["history_frames"]
                    B_p, H_p, C_p, h_p, w_p = history_frames.shape
                    flat_f = history_frames.reshape(B_p * H_p, C_p, h_p, w_p)
                    flat_p = encoder.forward_patches(flat_f)
                    hist_p = flat_p.reshape(B_p, H_p, num_patches, feature_dim)

                    # Build tokens same as forward
                    pt = (hist_p + transition_model.patch_token_embed
                          + transition_model.patch_pos_embed.unsqueeze(1)
                          + transition_model.temporal_pos_embed[:, :H_p, :, :])
                    pt = pt.reshape(B_p, H_p * num_patches, feature_dim)
                    proprio_tok = (transition_model.proprio_proj(batch_data["proprio_t"]).unsqueeze(1)
                                   + transition_model.proprio_token_embed)
                    intent_proj = (transition_model.intent_proj(batch_data["intent_tokens"])
                                   + transition_model.intent_token_embed)
                    tokens_probe = torch.cat([pt, proprio_tok, intent_proj], dim=1)

                    x_norm = transition_model.norm1_layers[1](tokens_probe)
                    cache_key = (H_p, transition_model.max_intent_tokens, tokens_probe.device)
                    sp_m, tp_m = transition_model._cached_masks.get(
                        cache_key,
                        transition_model._build_attention_masks(
                            H_p, transition_model.max_intent_tokens, tokens_probe.device
                        ),
                    )
                    N_p = tokens_probe.shape[1]
                    prefix_m = torch.ones(B_p, H_p * num_patches + 1,
                                          dtype=torch.bool, device=tokens_probe.device)
                    full_m = torch.cat([prefix_m, batch_data["attention_mask"]], dim=1)
                    pad_m = ~full_m

                    s_out, _ = transition_model.attn_layers[1](
                        x_norm, x_norm, x_norm,
                        attn_mask=sp_m[:N_p, :N_p], key_padding_mask=pad_m)
                    t_out, _ = transition_model.attn_layers[1](
                        x_norm, x_norm, x_norm,
                        attn_mask=tp_m[:N_p, :N_p], key_padding_mask=pad_m)

                    spatial_norm_sum = s_out.float().norm().item()
                    temporal_norm_sum = t_out.float().norm().item()
                    st_norm_count = 1

        avg_val_vis = val_vis_loss / max(val_steps, 1)
        avg_val_prop = val_prop_loss / max(val_steps, 1)
        avg_val_total = avg_val_vis + proprio_lambda * avg_val_prop
        avg_val_baseline = val_baseline_loss / max(val_steps, 1)
        improvement_pct = (avg_val_baseline - avg_val_vis) / max(avg_val_baseline, 1e-8) * 100

        avg_per_patch = val_per_patch_losses / max(val_steps, 1)
        patch_loss_std = avg_per_patch.std().item()
        patch_loss_max_idx = avg_per_patch.argmax().item()

        current_lr = scheduler.get_last_lr()[0]

        log.info(
            "Epoch %d/%d [%s] | Train vis=%.6f prop=%.6f | "
            "Val vis=%.6f prop=%.6f baseline=%.6f improv=%.1f%% | "
            "Patch std=%.4f max_idx=%d | LR=%.2e",
            epoch + 1, num_epochs, experiment_mode,
            avg_train_vis, avg_train_prop,
            avg_val_vis, avg_val_prop, avg_val_baseline, improvement_pct,
            patch_loss_std, patch_loss_max_idx, current_lr,
        )

        if use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/visual_loss": avg_train_vis,
                "train/proprio_loss": avg_train_prop,
                "train/total_loss": avg_train_vis + proprio_lambda * avg_train_prop,
                "val/visual_loss": avg_val_vis,
                "val/proprio_loss": avg_val_prop,
                "val/total_loss": avg_val_total,
                "val/baseline_loss": avg_val_baseline,
                "val/improvement_pct": improvement_pct,
                "val/patch_loss_std": patch_loss_std,
                "val/patch_loss_max_idx": patch_loss_max_idx,
                "lr": current_lr,
            }

            # --- Stale intent: per-chunk-position loss ---
            if chunk_size > 0:
                for cp in range(chunk_size):
                    if chunk_pos_count[cp] > 0:
                        log_dict[f"val/chunk_pos_{cp}_loss"] = (
                            chunk_pos_loss_sum[cp] / chunk_pos_count[cp]
                        ).item()
                # Staleness degradation: loss at last position vs first
                if chunk_pos_count[0] > 0 and chunk_pos_count[-1] > 0:
                    loss_fresh = (chunk_pos_loss_sum[0] / chunk_pos_count[0]).item()
                    loss_stale = (chunk_pos_loss_sum[-1] / chunk_pos_count[-1]).item()
                    log_dict["val/stale_degradation"] = loss_stale - loss_fresh
                    log_dict["val/stale_degradation_pct"] = (
                        (loss_stale - loss_fresh) / max(loss_fresh, 1e-8) * 100
                    )

            # --- Multiframe: spatial vs temporal attention norms ---
            if experiment_mode == "multiframe" and st_norm_count > 0:
                log_dict["val/spatial_attn_norm"] = spatial_norm_sum / st_norm_count
                log_dict["val/temporal_attn_norm"] = temporal_norm_sum / st_norm_count
                total_norm = spatial_norm_sum + temporal_norm_sum
                if total_norm > 0:
                    log_dict["val/temporal_ratio"] = temporal_norm_sum / total_norm

            wandb.log(log_dict, step=epoch + 1)

        # Save best
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            torch.save(transition_model.state_dict(), output_dir / "transition_vit_best.pt")
            torch.save(proprio_forward.state_dict(), output_dir / "proprio_forward_best.pt")
            log.info("  -> Best model saved (val_loss=%.6f)", avg_val_total)

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt = {
                "epoch": epoch + 1,
                "experiment_mode": experiment_mode,
                "transition_model": transition_model.state_dict(),
                "proprio_forward": proprio_forward.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(ckpt, output_dir / f"checkpoint_epoch{epoch + 1}.pt")

    # Save final
    torch.save(transition_model.state_dict(), output_dir / "transition_vit_final.pt")
    torch.save(proprio_forward.state_dict(), output_dir / "proprio_forward_final.pt")
    log.info("Phase 1 training complete [%s]. Best val_loss=%.6f", experiment_mode, best_val_loss)

    if use_wandb:
        wandb.log({"best_val_loss": best_val_loss})
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Phase 1: Patch-level Transition ViT (v2)")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--intent_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="",
                        help="Disk cache for decoded frames.")
    parser.add_argument("--feature_dim", type=int, default=384)
    parser.add_argument("--proprio_dim", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--intent_dim", type=int, default=2048)
    parser.add_argument("--num_patches", type=int, default=49)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--ffn_dim", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--intent_proj_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--proprio_lambda", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--decode_workers", type=int, default=32)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    # Experiment extensions
    parser.add_argument("--experiment_mode", type=str, default="baseline",
                        choices=["baseline", "action_conditioned", "multiframe"],
                        help="Experiment variant to run")
    parser.add_argument("--chunk_size", type=int, default=0,
                        help="Action chunk size for stale intent (0 = fresh intent)")
    parser.add_argument("--vla_dir", type=str, default="",
                        help="Directory with VLA action .pt files (for action_conditioned)")
    parser.add_argument("--history_len", type=int, default=1,
                        help="Number of history frames (for multiframe, default=1)")
    args = parser.parse_args()

    train_phase1(**vars(args))
