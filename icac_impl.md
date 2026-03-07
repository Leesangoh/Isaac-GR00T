# Patch-Level ICAC Architecture v2

> **날짜**: 2026-03-07
> **변경 사유**: Frozen DINOv2 mean-pooled (v1) 실패 → patch-level transition model로 전환
> **참고**: DINO-WM (Zhou et al., ICML 2025) 아키텍처 기반, intent conditioning 추가

---

## 변경 요약 (v1 → v2)

| 항목 | v1 (기존) | v2 (새로운) |
|---|---|---|
| Visual encoder | DINOv2 + LoRA + EMA | **Frozen DINOv2** (no LoRA, no EMA) |
| Feature 형태 | mean-pooled 384-dim (1개 벡터) | **49 patch tokens × 384-dim** |
| Forward model | 2-layer Transformer, 130 tokens | **ViT transition model, ~178 tokens** |
| 예측 대상 | Δz (변화량) | **z_{t+1} (절대값)** |
| Prediction error | (z_{t+1}-z_t) - Δz_ideal (384-dim) | **z_{t+1} - ẑ_{t+1} → attn pool (384-dim)** |
| Action 입력 | ❌ 안 넣음 | ❌ 안 넣음 (동일) |
| Intent 입력 | ✅ 128 tokens | ✅ 128 tokens (동일) |
| VICReg | λ=5.0 | **불필요** (frozen encoder) |

---

## 전체 파이프라인

```
═══════════════════════════════════════════════════════════════
                    PHASE 1: Forward Model 학습
═══════════════════════════════════════════════════════════════

  Image o_t (96×96)                    Image o_{t+1} (96×96)
       │                                      │
       ▼                                      ▼
  ┌──────────────┐                    ┌──────────────┐
  │ Frozen DINOv2│                    │ Frozen DINOv2│
  │ ViT-S/14     │                    │ ViT-S/14     │
  │ (21M, frozen)│                    │ (21M, frozen)│
  └──────────────┘                    └──────────────┘
       │                                      │
       ▼                                      ▼
  Z_t ∈ R^{49×384}                   Z_{t+1} ∈ R^{49×384}  (target)
       │
       │  GR00T Eagle3 (frozen, offline pre-extracted)
       │       │
       │       ▼
       │  intent_tokens ∈ R^{128×2048}
       │       │
       │       ▼ Linear projection
       │  intent_proj ∈ R^{128×384}
       │       │
       ├───────┤  proprio_t (8-dim)
       │       │       │
       │       │       ▼ Linear projection
       │       │  proprio_proj ∈ R^{1×384}
       │       │
       ▼       ▼
  ┌─────────────────────────────────────┐
  │     Transition ViT (Predictor)      │
  │                                     │
  │  Input tokens (178 total):          │
  │    - 49 patch tokens (Z_t)          │
  │    - 1 proprio token                │
  │    - 128 intent tokens              │
  │                                     │
  │  + Learnable token-type embeddings  │
  │  + Learnable position embeddings    │
  │    (patch tokens에만)               │
  │                                     │
  │  ViT: 4 layers, 8 heads, 384-dim   │
  │  MLP dim: 1536                      │
  │  Pre-LayerNorm                      │
  │                                     │
  │  Output: 처음 49개 token readout    │
  └─────────────────────────────────────┘
       │
       ▼
  Ẑ_{t+1} ∈ R^{49×384}  (predicted)
       │
       ▼
  L_visual = ||Ẑ_{t+1} - Z_{t+1}||²   (per-patch MSE, mean over patches)


═══════════════════════════════════════════════════════════════
                    PHASE 2: Correction Network 학습
═══════════════════════════════════════════════════════════════

  Phase 1의 모든 모듈 frozen 상태에서:

  Prediction Error (per-patch):
  E = Z_{t+1} - Ẑ_{t+1}              # 49×384

       │
       ▼
  ┌─────────────────────────────────┐
  │  Attention-Weighted Pooling     │
  │                                 │
  │  score_i = MLP(E_i) → scalar   │  (learnable, patch별 중요도)
  │  w_i = softmax(scores)          │
  │  e = Σ w_i × E_i               │  → 384-dim
  └─────────────────────────────────┘
       │
       ▼
  e ∈ R^{384}  (pooled prediction error)
       │
       ├── action_t (7-dim)
       ├── proprio_t (8-dim)
       ├── chunk_step (1-dim)
       │
       ▼
  ┌─────────────────────────────────┐
  │  Correction Network             │
  │  3-layer MLP, hidden 256-dim    │
  │  Input: 384+7+8+1 = 400-dim    │
  │  Output: Δa_t (7-dim)           │
  │  Bounded: tanh × α (α=0.15)    │
  └─────────────────────────────────┘
       │
       ▼
  a_corrected = a_t + Δa_t
```

---

## 모듈별 상세 설계

### 1. Observation Model — Frozen DINOv2 ViT-S/14

- **모델**: `dinov2_vits14` (facebookresearch/dinov2)
- **파라미터**: ~21M, **완전히 frozen** (LoRA 없음, EMA 없음)
- **입력**: 98×98 RGB (96×96에서 DINOv2 patch size 14에 맞게 조정)
  - 98 / 14 = 7 → 7×7 = **49 patch tokens**
- **출력**: Z_t ∈ R^{49×384}
- **특징**: pre-extracted 가능 (frozen이므로 매번 계산 불필요, offline 저장)

### 2. Intent Projection — Linear

- **입력**: intent_tokens ∈ R^{128×2048} (GR00T Eagle3 backbone output, offline pre-extracted)
- **투영**: Linear(2048, 384) — bias 포함
- **출력**: intent_proj ∈ R^{128×384}
- **파라미터**: 2048 × 384 + 384 ≈ **0.79M**
- **이유**: intent는 이미 VLM에서 인코딩됨. MLP 불필요, dimension 맞추기만.

### 3. Proprio Projection — Linear

- **입력**: proprio_t ∈ R^{8} (joint positions + gripper state)
- **투영**: Linear(8, 384)
- **출력**: proprio_proj ∈ R^{1×384}
- **파라미터**: ~3K (무시할 수준)

### 4. Transition ViT (Predictor) — 핵심 모듈

- **아키텍처**: ViT (decoder-only style은 아님, 일반 encoder)
  - Depth: **4 layers** (DINO-WM 6에서 축소, 1-step만 예측하므로)
  - Heads: **8**
  - Hidden dim: **384**
  - MLP dim: **1536** (4× hidden)
  - Activation: GELU
  - Normalization: Pre-LayerNorm
  - Dropout: 0.1

- **입력 토큰 구성** (총 178개):
  | Token type | 개수 | Dim | Source |
  |---|---|---|---|
  | Patch tokens | 49 | 384 | Frozen DINOv2(o_t) |
  | Proprio token | 1 | 384 | Linear(proprio_t) |
  | Intent tokens | 128 | 384 | Linear(intent_tokens) |

- **Embeddings**:
  - Token-type embedding: 3종 (patch / proprio / intent), learnable, 384-dim
  - Position embedding: patch tokens에만 적용 (7×7 spatial, learnable)
  - Intent tokens: position embedding 없음 (VLM 내부 순서가 이미 있으므로)

- **출력**: 처음 49개 token → Ẑ_{t+1} ∈ R^{49×384}

- **파라미터 추정**:
  - Self-attention per layer: 4 × 384² = 589K (Q, K, V, O)
  - MLP per layer: 2 × 384 × 1536 = 1.18M
  - Per layer total: ~1.77M
  - 4 layers: **~7.1M**
  - Embeddings + LayerNorm: ~0.3M
  - **Total: ~7.4M**

### 5. Attention-Weighted Pooling (Phase 2에서 학습)

- **입력**: E ∈ R^{49×384} (per-patch prediction error)
- **Score MLP**: Linear(384, 128) → ReLU → Linear(128, 1) → Softmax over patches
- **출력**: e = Σ w_i × E_i ∈ R^{384}
- **파라미터**: ~50K
- **의미**: error가 큰 patch (로봇 팔/물체 근처)에 높은 가중치

### 6. Correction Network (Phase 2에서 학습)

- v1과 동일 (변경 없음)
- **입력**: pooled_error (384) + action_t (7) + proprio_t (8) + chunk_step (1) = **400-dim**
- **구조**: 3-layer MLP, hidden 256-dim, ReLU
- **출력**: Δa_t (7-dim), bounded by tanh × α (α=0.15)
- **파라미터**: ~0.5M

---

## 파라미터 요약

| 모듈 | 파라미터 | 학습 여부 |
|---|---|---|
| DINOv2 ViT-S/14 | 21M | ❄️ Frozen |
| GR00T Eagle3 | 1.87B | ❄️ Frozen (offline) |
| Intent projection (Linear) | 0.79M | ✅ Phase 1 |
| Proprio projection (Linear) | 0.003M | ✅ Phase 1 |
| Transition ViT (4-layer) | 7.4M | ✅ Phase 1 |
| Attn-weighted pooling | 0.05M | ✅ Phase 2 |
| Correction Network | 0.5M | ✅ Phase 2 |
| **Total trainable** | **~8.7M** | |

- GR00T (3.29B) 대비 **~0.26%** — 여전히 경량
- v1 (~3M) 대비 증가했으나, ViT transition model이 patch-level dynamics를 학습하기 위해 필요한 크기
- DINO-WM (19M)의 절반 이하 (49 patches vs 196 patches로 인한 자연스러운 축소)

---

## 학습 상세

### Phase 1: Transition ViT 학습 (자기지도학습)

**목적**: intent 기반으로 다음 frame의 visual patch tokens 예측

```
L_visual = (1/N) Σ_{i=1}^{49} ||ẑ_{t+1}^i - z_{t+1}^i||²

where:
  z_{t+1} = frozen_dinov2(o_{t+1})   (target, no gradient)
  ẑ_{t+1} = transition_vit(z_t, proprio_t, intent_tokens)
  z_t = frozen_dinov2(o_t)            (input, no gradient through encoder)
```

**Proprio branch** (선택적, 유지):
```
L_proprio = ||Δp_pred - (p_{t+1} - p_t)||²

where Δp_pred = MLP(proprio_t, mean(intent_proj))
```

**Total loss**:
```
L_total = L_visual + λ_proprio × L_proprio

λ_proprio = 0.1 (visual이 주, proprio는 보조)
```

> **VICReg 제거**: frozen encoder이므로 collapse 걱정 없음

**Hyperparameters (DINO-WM 기반 + 우리 상황 조정)**:

| Hyperparameter | 값 | 비고 |
|---|---|---|
| Optimizer | AdamW | DINO-WM과 동일 |
| Transition ViT lr | 3e-4 | DINO-WM "Decoder lr" |
| Intent projection lr | 5e-4 | DINO-WM "Action encoder lr" 참고 |
| Weight decay | 0.01 | |
| LR scheduler | Cosine annealing | |
| Warmup | 5 epochs | |
| Batch size | 128 | DINO-WM 32 → 우리 데이터 53K로 더 크므로 증가 |
| Epochs | 100 | DINO-WM과 동일 |
| Mixed precision | bfloat16 | |
| Gradient clipping | 1.0 | |

**데이터 전처리**:
- DINOv2 patch features **offline pre-extraction** (frozen이므로 가능!)
  - 전체 1.47M frames → 49×384 features로 저장
  - 학습 시 이미지 로딩 불필요 → **대폭 빨라짐**
- Intent tokens도 offline pre-extracted (기존과 동일)
- Proprio는 dataset에서 직접 로딩

**Monitoring metrics**:
| Metric | 의미 | 기대값 |
|---|---|---|
| val_visual_loss | 예측 정확도 | 꾸준히 감소 |
| improvement% | (baseline - loss) / baseline × 100 | >50% 목표 |
| per_patch_loss | patch별 loss 분포 | 로봇 근처 patch가 높아야 정상 |
| baseline (Δz=0) | z_t를 그대로 출력 시 loss | 안정적 상수 (frozen이므로!) |

> **핵심 차이**: frozen encoder이므로 baseline이 완전 상수! v1에서 baseline이 2.6~7.3으로 진동하던 문제 완전 해결.

### Phase 2: Correction Network 학습

**Phase 1 모듈 전부 frozen 후:**

```
L_correction = ||Δa_pred - (a_expert - a_VLA)||²

where:
  E = z_{t+1} - ẑ_{t+1}                    (49×384, frozen forward model)
  e = attention_weighted_pool(E)             (384-dim)
  Δa_pred = correction_net(e, a_t, proprio, step)
```

| Hyperparameter | 값 |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Epochs | 50 |

---

## Inference 시 파이프라인

```
매 action chunk 시작 시:
  1. GR00T forward → action chunk [a_1, ..., a_8] + intent tokens 추출

매 control step t (chunk 내):
  2. DINOv2(o_t) → Z_t (49×384)                    ~2.5ms
  3. Transition ViT(Z_t, proprio, intent) → Ẑ_{t+1} ~1.5ms
  4. 실제 관측 후: E = Z_{t+1} - Ẑ_{t+1}
  5. e = attention_pool(E) → 384-dim                 ~0.1ms
  6. Δa = correction_net(e, a_t, proprio, step)      ~0.1ms
  7. a_corrected = a_{t+1} + Δa (다음 step에 적용)

총 오버헤드: ~4.2ms (200ms budget의 ~2%)
```

> Note: one-step delay는 v1과 동일. step t의 prediction error가 step t+1을 보정.

---

## v1 대비 장점

1. **Frozen encoder → 안정적 학습**: EMA 불안정, baseline 진동, cross-space 버그 모두 제거
2. **Patch-level → spatial dynamics 포착**: mean pooling으로 희석되던 로봇/물체 움직임을 patch 단위로 잡음
3. **Offline feature extraction 가능**: frozen이므로 patch features를 미리 계산해놓으면 학습 속도 대폭 향상
4. **Baseline 상수**: frozen encoder에서 같은 이미지는 항상 같은 feature → improvement% 측정이 신뢰할 수 있음
5. **DINO-WM 검증 완료**: frozen DINOv2 patch features의 dynamics modeling은 ICML 2025에서 검증됨

## 잠재적 리스크

1. **49 patches로 충분한가?**: DINO-WM은 196 patches (14×14). 우리 49 (7×7)는 spatial resolution이 낮음
   - Mitigation: 로봇 manipulation 이미지는 96×96로도 충분할 수 있음 (배경 단순)
2. **Intent token 128개가 너무 많은가?**: 178 tokens 중 128이 intent → ViT computation의 대부분
   - Mitigation: intent token을 pooling/sampling으로 줄일 수 있음 (16~32개)
3. **~8.7M은 "경량"인가?**: v1의 3M 대비 3배. 하지만 GR00T 3.29B의 0.26%이므로 여전히 경량 주장 가능
