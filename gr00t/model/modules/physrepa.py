"""PhysREPA: Physics-informed REPA alignment loss.

Projects DiT early-layer hidden states to align with V-JEPA 2 PEZ representations
via negative cosine similarity loss (following REPA paper).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysREPAHead(nn.Module):
    """Projection heads + alignment loss for PhysREPA."""

    def __init__(self, dit_dim=1536, vjepa_dim=1024, align_layers=None):
        super().__init__()
        if align_layers is None:
            align_layers = list(range(9))
        self.align_layers = align_layers

        # Per-layer 2-layer MLP projector
        self.projectors = nn.ModuleDict(
            {
                str(i): nn.Sequential(
                    nn.Linear(dit_dim, dit_dim),
                    nn.GELU(),
                    nn.Linear(dit_dim, vjepa_dim),
                )
                for i in align_layers
            }
        )

    def forward(self, all_hidden_states, vjepa_target):
        """Compute alignment loss between DiT hidden states and V-JEPA features.

        Supports two modes based on vjepa_target shape:
        - Mean-pool mode [B, vjepa_dim]: pool all DiT tokens, align to single target.
        - Timestep-wise mode [B, H, vjepa_dim]: align each action token individually.

        Args:
            all_hidden_states: list of [B, seq_len, dit_dim] from DiT.
                Index 0 = input, index i+1 = output of block i.
            vjepa_target: [B, vjepa_dim] or [B, H, vjepa_dim] detached V-JEPA features.

        Returns:
            Tuple of (scalar loss, metrics dict).
            Loss is -cos_sim (REPA paper convention). Range [-1, 1].
        """
        if vjepa_target.dim() == 3:
            return self._forward_timestepwise(all_hidden_states, vjepa_target)
        return self._forward_meanpool(all_hidden_states, vjepa_target)

    def _forward_meanpool(self, all_hidden_states, vjepa_target):
        """Original mean-pool alignment: pool all tokens → align to [B, vjepa_dim]."""
        losses = []
        metrics = {}
        vjepa_target = vjepa_target.detach()

        for layer_idx_str, proj in self.projectors.items():
            layer_idx = int(layer_idx_str)
            h = all_hidden_states[layer_idx + 1]  # [B, seq_len, dit_dim]
            h_pooled = h.mean(dim=1)  # [B, dit_dim]
            h_proj = proj(h_pooled)  # [B, vjepa_dim]
            cos_sim = F.cosine_similarity(h_proj, vjepa_target, dim=-1)  # [B]
            losses.append(-cos_sim.mean())

            metrics[f"repa/layer_{layer_idx}_cosine_sim"] = cos_sim.mean().detach()
            metrics[f"dit/layer_{layer_idx}_hidden_norm"] = h_pooled.detach().norm(dim=-1).mean()

        loss = sum(losses) / len(losses)
        metrics["repa/mean_cosine_sim"] = -loss.detach()
        return loss, metrics

    def _forward_timestepwise(self, all_hidden_states, vjepa_target):
        """Per-action-token alignment: each action token aligns to its V-JEPA target."""
        losses = []
        metrics = {}
        vjepa_target = vjepa_target.detach()
        action_horizon = vjepa_target.shape[1]

        for layer_idx_str, proj in self.projectors.items():
            layer_idx = int(layer_idx_str)
            h = all_hidden_states[layer_idx + 1]  # [B, state_horizon+H, dit_dim]
            h_actions = h[:, -action_horizon:, :]  # [B, H, dit_dim]
            h_proj = proj(h_actions)  # [B, H, vjepa_dim]
            cos_sims = F.cosine_similarity(h_proj, vjepa_target, dim=-1)  # [B, H]
            losses.append(-cos_sims.mean())

            metrics[f"repa/layer_{layer_idx}_cosine_sim"] = cos_sims.mean().detach()
            metrics[f"dit/layer_{layer_idx}_hidden_norm"] = h_actions.detach().norm(dim=-1).mean()

        loss = sum(losses) / len(losses)
        metrics["repa/mean_cosine_sim"] = -loss.detach()

        # Per-timestep metrics from last layer
        with torch.no_grad():
            ts_cos = cos_sims.mean(dim=0)  # [H]
            metrics["repa/per_timestep_cosine_sim_min"] = ts_cos.min()
            metrics["repa/per_timestep_cosine_sim_max"] = ts_cos.max()

        return loss, metrics

    def get_monitoring_metrics(self):
        """Get projection head weight/grad norms for monitoring."""
        metrics = {}
        for layer_idx_str, proj in self.projectors.items():
            layer_idx = int(layer_idx_str)
            # Weight norm
            w_norm = sum(p.data.norm().item() ** 2 for p in proj.parameters()) ** 0.5
            metrics[f"repa/proj_head_{layer_idx}_weight_norm"] = w_norm
            # Grad norm (from previous backward, if available)
            g_norm_sq = 0.0
            has_grad = False
            for p in proj.parameters():
                if p.grad is not None:
                    g_norm_sq += p.grad.data.norm().item() ** 2
                    has_grad = True
            if has_grad:
                metrics[f"repa/proj_head_{layer_idx}_grad_norm"] = g_norm_sq**0.5
        return metrics
