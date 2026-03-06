# GR00T N1.6 Architecture Verification

Verified 2026-03-05 from `/mnt/md1/solee/checkpoints/GR00T-N1.6-bridge/`.

## Summary

```
========================================
GR00T N1.6 Architecture Verification
========================================

1. VLM Hidden Dimension:  2048
2. Language Model:         Qwen3-1.7B (hidden_size=2048, 28 layers → truncated to 16)
3. Tokenizer:              Qwen3 (vocab_size=151,680)
4. Vision Encoder:         SigLIP2 (hidden_size=1152, patch_size=14, 27 layers)
5. VLM Output Shape:       (B, 108, 2048)   [81 image + 27 text tokens for 224px input]
6. DiT Action Head:        AlternateVLDiT (32 layers, hidden=1024, input_emb=1536)
7. Action Chunk:            length=8 (WidowX), action_dim=7
8. Action Horizon (max):    50 (config), 8 (WidowX delta_indices)
9. Proprio Dimension:       8 (x, y, z, roll, pitch, yaw, pad, gripper)
10. Denoising Steps:        4 (flow matching)
11. Total Params:           3.29B (Backbone 1.87B + Action Head 1.42B)

D_intent = 2048

========================================
```

## Detailed Architecture

### Model Structure

```
Gr00tN1d6
├── backbone: EagleBackbone
│   └── model: Eagle3_VLForConditionalGeneration
│       ├── vision_model: Siglip2VisionModel (1152-dim, 27 layers, patch=14)
│       ├── mlp1: Sequential (vision projector, downsample_ratio=0.5)
│       └── language_model: Qwen3ForCausalLM (2048-dim, truncated 28→16 layers)
│
└── action_head: Gr00tN1d6ActionHead
    ├── vlln: LayerNorm(2048)       ← applied to backbone features
    ├── state_encoder: CategorySpecificMLP (128 → 1536)
    ├── action_encoder: MultiEmbodimentActionEncoder (128 → 1536)
    ├── model: AlternateVLDiT (32 layers, 32 heads, head_dim=48)
    ├── action_decoder: CategorySpecificMLP (1024 → 128)
    └── position_embedding: Embedding(1024, 1536)
```

### Forward Pass Flow

```
1. Image (224×224) → SigLIP2 → (B, 324, 1152)
2. SigLIP features → Eagle MLP connector (downsample 0.5) → (B, 81, 2048)
3. Text → Qwen3 tokenizer → Qwen3 LM (16 layers) → (B, 27, 2048)
4. Concat → VLM output: (B, 108, 2048) = backbone_features
5. VLLN(backbone_features) → vl_embeds: (B, 108, 2048)
6. State → state_encoder → state_features: (B, 1, 1536)
7. For t in range(4):  # denoising loop
   a. action_encoder(noisy_actions, t) → action_features: (B, 8, 1536)
   b. cat(state_features, action_features) → sa_embs: (B, 9, 1536)
   c. AlternateVLDiT(sa_embs, vl_embeds) → (B, 9, 1024)
   d. action_decoder → velocity: (B, 8, 128)
   e. actions += dt * velocity
8. actions[:, :8, :7] → action chunk output
```

### Token Breakdown (224×224 single image input)

| Token Type | Count | Source |
|------------|-------|--------|
| Image | 81 | SigLIP2 (18×18=324 patches, downsampled 0.5→9×9=81) |
| Text | 27 | Qwen3 tokenizer (task instruction) |
| **Total** | **108** | backbone_features seq_len |

### WidowX (oxe_widowx) Config

- **Video**: 1 camera (`image_0`), delta_indices=[0] (current frame only)
- **State**: 8 dims (x, y, z, roll, pitch, yaw, pad, gripper), delta_indices=[0]
- **Action**: 7 dims (x, y, z, roll, pitch, yaw, gripper), delta_indices=[0..7] (8-step chunk)
- **Language**: `annotation.human.action.task_description`
- **Action normalization**: mean_std for xyz/rpy, absolute representation

## Intent Vector Extraction Plan

### Hook Location

```python
# Option 1: Hook on backbone output (after Eagle, before VLLN)
model.backbone.register_forward_hook(hook_fn)
# Output: backbone_features (B, 108, 2048), image_mask, attention_mask

# Option 2: Hook after VLLN (inside action_head)
model.action_head.vlln.register_forward_hook(hook_fn)
# Output: (B, 108, 2048) after LayerNorm
```

**Recommended: Option 1 (backbone output)** — gives access to all metadata (image_mask, attention_mask) for flexible pooling.

### Pooling Options

| Method | Norm | Description |
|--------|------|-------------|
| `features[attn_mask].mean(0)` | 58.5 | Mean over all active tokens |
| `features[img_mask].mean(0)` | 77.5 | Mean over image tokens only |
| `features[~img_mask & attn_mask].mean(0)` | 15.2 | Mean over text tokens only |
| `features[attn_mask][-1]` | 326.0 | Last token (autoregressive summary) |

**Recommended: Mean over all active tokens** — captures both visual scene and language intent.

### Intent Vector Specification

```
D_intent = 2048
Shape: (B, 2048) after mean pooling
Source: backbone_features with attention_mask pooling
```

## Key Config Values (from checkpoint config.json)

```json
{
  "backbone_embedding_dim": 2048,
  "action_horizon": 50,        // max across all embodiments
  "max_action_dim": 128,
  "max_state_dim": 128,
  "hidden_size": 1024,          // DiT hidden
  "input_embedding_dim": 1536,  // DiT input
  "num_inference_timesteps": 4,
  "select_layer": 16,           // LM layers kept
  "state_dropout_prob": 0.8,
  "use_alternate_vl_dit": true,
  "use_relative_action": true,
  "apply_sincos_state_encoding": true
}
```
