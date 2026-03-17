# PhysREPA Implementation Prompt

> **목적**: GR00T N1.6 bridge-finetuned checkpoint에 PhysREPA alignment loss를 추가한 finetuning 구현
> **작성일**: 2026-03-17

---

## 프로젝트 개요

### 배경: REPA (Representation Alignment)

REPA (ICLR 2025 Oral)는 image generation DiT를 학습할 때, DiT의 early layer hidden states를 pretrained encoder (DINOv2)의 representation과 cosine similarity loss로 align하여 학습 수렴을 크게 가속시킨 방법이다 (SiT-XL/2 28L에서 first 8 layers에 alignment → 17.5x speedup).

### PhysREPA: REPA를 VLA + Physics로 확장

PhysREPA는 이 REPA 아이디어를 **VLA (Vision-Language-Action) 모델의 action head**에 적용하되, alignment target을 DINOv2 (semantic)가 아닌 **V-JEPA 2 (physics)** 로 교체한다.

V-JEPA 2는 self-supervised video encoder인데, "Interpreting Physics in Video World Models" (2602.07050) 논문에서 V-JEPA 2의 **중간 layer (~1/3-0.5 depth)**에 물리 변수 (speed, direction, object permanence 등)가 linearly decodable한 **Physics Emergence Zone (PEZ)** 이 존재함이 밝혀졌다. PEZ에서는 motion direction이 갑자기 decode 가능해지고, speed/acceleration은 초기 layer부터 이미 decodable하다. 이 physics representation은 distributed, high-dimensional (40-80 features) 형태이며, physics-specific한 local attention head가 PEZ에서 출현한다.

PhysREPA는 이 PEZ layer의 representation을 VLA action head의 early layers에 주입하여, VLA가 pixel generation 없이 physics knowledge를 활용하도록 한다.

### 현재 단계

GR00T N1.6 (32-layer DiT action head)의 bridge-finetuned checkpoint에서, BridgeData V2로 추가 finetuning할 때 PhysREPA alignment loss를 추가하는 것이 첫 번째 실험 (Story 2: finetuning improvement).

---

## 환경

- **GR00T 코드**: `/home/solee/Isaac-GR00T/`
- **V-JEPA 2 코드**: `/home/solee/Isaac-GR00T/vjepa2/` 에 clone
  - V-JEPA 2 GitHub repo: https://github.com/facebookresearch/vjepa2
  - 설치는 repo의 README.md 지침을 따를 것
- **BridgeData V2**: `/mnt/md1/solee/data/bridge_lerobot` (53K episodes, LeRobot format, 이미 다운로드 완료)
- **GR00T checkpoint**: `/mnt/md1/solee/checkpoints/GR00T-N1.6-bridge` (bridge-finetuned)
- **V-JEPA 2 checkpoint 저장**: `/mnt/md1/solee/checkpoints/vjepa2/` (새로 생성, 다운로드한 weight 저장)
- **V-JEPA 2 pre-extracted features 저장**: `/mnt/md1/solee/features/vjepa2/` (새로 생성)
- **PhysREPA checkpoints 저장**: `/mnt/md1/solee/checkpoints/physrepa/` (새로 생성)
- **GPU**: A6000 48GB × 4

---

## 작업 순서

### 1. V-JEPA 2 Feature Pre-extraction

V-JEPA 2 repo (https://github.com/facebookresearch/vjepa2) 를 `/home/solee/Isaac-GR00T/vjepa2/`에 clone하고, repo의 README.md 지침을 따라 설치. Model checkpoint는 `/mnt/md1/solee/checkpoints/vjepa2/`에 저장.

**V-JEPA 2-L (Large, 24 layers) 사용.**

BridgeData V2 전체 (53K episodes)에 대해 V-JEPA 2-L encoder의 특정 layer features를 pre-extract하여 disk에 저장. Training 시에는 이 pre-extracted features를 로드하여 alignment target으로 사용 (매 training step마다 V-JEPA 2 forward pass를 하지 않기 위함).

#### V-JEPA 2 Input 요구사항 (⚠️ 중요)

**V-JEPA 2는 video model이다. Single frame이 아니라 16 consecutive frames를 input으로 받는다.**
- Tubelet embedding (temporal stride 2) → 8 temporal tokens
- Spatial: 14×14 grid (16×16 patches) → 196 spatial tokens
- Total: 8 × 196 = 1568 tokens per video clip

BridgeData V2의 각 episode (~36 timesteps)에서 16-frame sliding window (stride 4)로 clip을 추출하여 V-JEPA 2에 통과시킨다. 각 clip에서 특정 layer의 hidden state를 추출하고, 1568 tokens을 spatiotemporal mean-pooling하여 single vector로 만든 뒤 저장.

#### 추출할 Layer (V-JEPA 2-L, 24 layers 기준)

| Layer | Fraction | 역할 |
|---|---|---|
| **Layer 10** | 0.42 | **PEZ peak — primary alignment target** |
| Layer 8 | 0.33 | PEZ transition 시작 (ablation) |
| Layer 12 | 0.50 | PEZ peak 후반 (ablation) |
| Layer 23 (last) | 1.0 | Output (ablation, PEZ vs output 비교) |

출처: "Interpreting Physics in Video World Models" (2602.07050). V-JEPA 2에서 ~1/3 depth에서 physics emerge 시작, ~0.4-0.5에서 peak.

#### Sanity Check (추출 후 반드시 수행)

Pre-extracted features에서 random 1000개를 뽑아 pair-wise cosine similarity를 측정. 만약 mean cosine similarity가 0.95 이상이면 mean-pooling으로 인해 features가 너무 유사해진 것 → token subset sampling이나 attention-weighted pooling 등 대안 검토 필요.

저장 위치: `/mnt/md1/solee/features/vjepa2/`

### 2. GR00T Finetuning에 PhysREPA Loss 추가

GR00T N1.6의 **기존 bridge finetuning script를 기반으로** 다음을 추가:

#### 2.1 DiT Intermediate Hidden States 캡처

GR00T DiT (32 layers)의 **first 9 layers (layer 0-8, ~30%)**에 forward hook을 설치하여 중간 hidden states를 캡처.

**왜 first ~30%인가**: REPA (ICLR 2025)에서 SiT-XL/2 (28L)의 first 8 layers에만 alignment을 적용했다 (= ~30%). Later layers는 task-specific representation으로 specialization되어 있으므로 external representation과 align하면 오히려 방해.

DiT block의 class 이름, hidden_dim 등은 Isaac-GR00T 코드를 직접 분석하여 파악할 것.

#### 2.2 PhysREPA Alignment Loss

각 align layer마다 **독립적인 2-layer MLP projection head** (linear → GELU → linear)를 두고, DiT hidden state를 V-JEPA 2 representation space로 project한 뒤 **cosine similarity loss**로 align.

```
PhysREPA loss = - mean over layers [ mean over batch [ cos_sim(proj(dit_h), vjepa_target) ] ]
total_loss = flow_matching_loss + λ * physrepa_loss
```

- λ = 0.5 (REPA convention, 시작점). 1.0도 ablation.
- Projection head는 표현력을 의도적으로 제한 (2-layer MLP) → DiT 자체가 변해야 loss가 줄어드도록. 만약 projection head가 너무 강력하면 DiT는 안 변하고 MLP만 학습하는 collapse가 발생할 수 있다.
- V-JEPA 2 target은 반드시 detach (frozen, gradient 안 흘림).
- DiT hidden state도 mean-pool over action tokens 후 project.

#### 2.3 Training Data에서 V-JEPA 2 Feature 매칭

Training sample의 (episode_id, timestep) → 해당하는 pre-extracted V-JEPA 2 feature를 로드. Timestep과 V-JEPA 2 window의 매핑 (stride 4 기준 가장 가까운 window) 처리 필요.

기존 BridgeData DataLoader를 수정하여 V-JEPA 2 feature도 함께 반환하도록.

#### 2.4 기존 Finetuning Script 기반

기존 GR00T bridge finetuning script의 설정 (lr, batch size, optimizer 등)을 그대로 유지. **추가되는 것은 alignment loss 관련 설정만:**
- `align_layers`: `[0,1,2,3,4,5,6,7,8]` (DiT first 9 layers)
- `vjepa_layer`: `10` (V-JEPA 2-L primary PEZ layer)
- `lambda_repa`: `0.5`
- `vjepa_features_dir`: `/mnt/md1/solee/features/vjepa2/`

### 3. 실행

1. **Small-scale test** (5K episodes, ~1000 steps) — loss가 뜨는지, OOM 없는지, gradient 흐르는지, projection head collapse 없는지 확인
2. **Baseline** — PhysREPA 없이 동일 조건 finetuning (λ=0)
3. **PhysREPA** — λ=0.5, vjepa_layer=10

Logging: flow_loss, repa_loss, total_loss, per-layer cosine similarity 반드시 log.

**Projection head collapse 체크**: repa_loss는 줄어드는데 flow_loss에 긍정적 영향 없으면 → projection head만 학습되고 DiT는 안 변한 것. Collapse 의심.

---

## 핵심 주의사항 요약

1. **V-JEPA 2는 16-frame video input 필요** — single frame 불가
2. **V-JEPA 2-L 사용** (24 layers). PEZ primary layer = **layer 10** (fraction 0.42)
3. **GR00T DiT의 first 9 layers (0-8)에만 alignment** — REPA convention, first ~30%
4. **Projection head는 2-layer MLP** — collapse 방지를 위해 표현력 제한
5. **V-JEPA 2 target은 .detach()** — frozen, gradient 안 흘림
6. **Pre-extracted features의 diversity 반드시 체크** — mean cosine sim > 0.95이면 mean-pooling 문제
7. **기존 finetuning script 설정을 건드리지 말 것** — alignment loss 관련 설정만 추가

---
*PhysREPA Implementation Prompt — 2026-03-17*