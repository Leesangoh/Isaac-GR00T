# PhysREPA 가설 검증 실험 Prompt

> **목적**: PhysREPA alignment가 flow_loss에 영향을 주지 않는 원인을 검증하는 두 가지 실험
> **작성일**: 2026-03-19

---

## 배경

5개 run (baseline, MeanPool ViT-L/G, TS-Align ViT-L/G) 모두 40K steps 학습 후 flow_loss가 0.200~0.236으로 거의 동일. Alignment 자체는 잘 학습됨 (cosine sim 0.93+), DiT representation도 크게 변함 (layer 8 drift 33%), 그러나 action prediction에 도움이 안 됨.

두 가지 가설을 검증하기 위한 실험을 설정한다.

---

## 실험 A: DiT Random Init에서 From Scratch 학습

### 가설
현재 PhysREPA가 안 되는 이유는 이미 수렴한 bridge-finetuned checkpoint에서 시작하기 때문. 이미 형성된 representation을 바꾸려 하면 later layers (9-31)가 보상하여 같은 output을 만듦. REPA 원논문은 random init에서 시작하여 alignment이 representation 형성을 가이드했고 그래서 효과가 있었음.

### 설정

**4개 run 비교:**

| Run | DiT Init | PhysREPA | 설명 |
|---|---|---|---|
| A1 | Random init | ❌ 없음 | From scratch baseline |
| A2 | Random init | ✅ TS-Align ViT-L | From scratch + PhysREPA |
| (기존) B1 | Bridge-finetuned | ❌ 없음 | 기존 baseline |
| (기존) B2 | Bridge-finetuned | ✅ TS-Align ViT-L | 기존 TS-Align |

A1, A2만 새로 돌리면 됨. B1, B2는 이미 있음 (260319_Baseline_NoPhysREPA_Final.csv, 260319_TSAlign_ViTL_Final.csv).

### 구체적 구현

1. **DiT action head만 random init**. VLM (Cosmos-Reason-2B)은 frozen pretrained 그대로 유지.
   - GR00T finetuning 코드에서 DiT weights를 로드하지 않고 random initialization하는 옵션을 찾거나 만들기.
   - VLM backbone은 반드시 pretrained weights 유지 (frozen).
   - 나머지 (state encoder, action encoder 등)도 가능하면 random init. 핵심은 DiT가 BridgeData에 대한 representation을 처음부터 학습해야 한다는 것.

2. **학습 설정**: 기존 bridge finetuning config 그대로 (lr, optimizer, batch size 등). 단, from scratch이므로 수렴에 더 많은 step이 필요할 수 있음 → **최소 40K steps, 가능하면 80K steps**.

3. **A2에서 PhysREPA 설정**: 기존 TS-Align ViT-L과 동일.
   - align_layers=[0-8], vjepa_layer=10, λ=0.5
   - Global mean centering 적용
   - Timestep-wise alignment (action token별 개별 V-JEPA 2 window)

4. **핵심 metric**: 같은 flow_loss 도달까지의 step 수. REPA 논문의 17.5x speedup처럼, A2가 A1보다 빠르게 수렴하면 가설 4 확인.

### WandB
- Run name: `physrepa_fromscratch_baseline`, `physrepa_fromscratch_tsalign_vitl`
- 기존과 동일한 logging (flow_loss, repa_loss, layer별 cosine_sim, DiT norm drift 등)

---

## 실험 B: λ Extreme (강한 Alignment 압력)

### 가설
현재 λ=0.5에서 alignment 압력이 약해서 later layers가 쉽게 보상(compensate)하고 있음. λ를 극단적으로 키우면 보상이 불가능해질 수 있음.

### 설정

| Run | λ | 기대 결과 |
|---|---|---|
| (기존) | 0.5 | flow_loss = baseline과 동일 |
| B1 | 2.0 | ? |
| B2 | 5.0 | ? |

**기존 bridge-finetuned checkpoint에서 시작**, TS-Align ViT-L 설정 그대로. λ만 변경.

### 해석 가이드

| 결과 | 의미 |
|---|---|
| flow_loss 나빠짐 | Alignment이 representation을 바꾸고 있지만, 그 방향이 action prediction에 해로움 |
| flow_loss 여전히 동일 | Later layers의 보상 능력이 매우 강하거나, alignment이 orthogonal한 방향에서 작동 |
| flow_loss 개선됨 | λ=0.5가 너무 약했을 뿐, 더 강한 alignment이 효과적 |

### 구체적 구현

1. 기존 TS-Align ViT-L 코드에서 `lambda_repa` 설정만 변경.
2. **40K steps** (기존과 동일).
3. λ가 크면 total_loss = flow_loss + λ * repa_loss에서 repa_loss 비중이 커지므로, gradient magnitude 변화에 주의. Gradient clipping이 있다면 그 설정 확인.

### WandB
- Run name: `physrepa_tsalign_vitl_lambda2.0`, `physrepa_tsalign_vitl_lambda5.0`

---

## 환경 (동일)

- **GR00T 코드**: `/home/solee/Isaac-GR00T/`
- **V-JEPA 2 pre-extracted features**: `/mnt/md1/solee/features/vjepa2_vitl/`
- **Global means**: `/mnt/md1/solee/features/vjepa2_vitl/global_means.pt`
- **BridgeData V2**: `/mnt/md1/solee/data/bridge_lerobot`
- **GPU**: A6000 48GB × 4

---

## 우선순위

**실험 B (λ extreme)를 먼저** — 코드 변경이 λ 값 하나뿐이라 즉시 시작 가능. 결과가 빠르게 나옴.

**실험 A (from scratch)는 병렬로** — DiT random init 구현이 필요하고 학습도 오래 걸리므로, 실험 B를 먼저 세팅한 후 구현 시작.

---
*PhysREPA Hypothesis Test Prompt — 2026-03-19*

