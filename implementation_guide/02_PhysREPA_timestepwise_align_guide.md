# PhysREPA Timestep-wise Alignment Implementation Prompt

> **목적**: 현재 mean-pool 기반 PhysREPA alignment을 timestep-wise temporal alignment으로 변경
> **작성일**: 2026-03-18

---

## 배경: 왜 바꾸는가

### 현재 방식의 문제

현재 PhysREPA는 DiT의 모든 token (1 state + 16 action = 17개)을 **mean-pool**하여 single vector로 만든 후, V-JEPA 2의 single mean-pooled vector와 cosine similarity로 align한다.

```python
# 현재 코드 (physrepa_alignment.py 52번째 줄 부근)
h = all_hidden_states[layer_idx + 1]  # [B, H+1, 1536]
h_pooled = h.mean(dim=1)              # [B, 1536]
h_proj = proj(h_pooled)               # [B, 1024]
```

문제점:
1. **Semantic mismatch**: Token 0은 robot joint state이지 scene physics가 아님. Action tokens는 미래 행동 계획. V-JEPA 2 feature는 과거 scene physics. 이 세 가지를 평균내서 align하는 건 의미가 불명확.
2. **Signal 희석**: 17개 token → 1 vector로 평균하면 개별 action token의 정보가 소실.
3. **Temporal correspondence 부재**: 16개 action token이 각각 다른 미래 시점 (t+1, t+2, ..., t+16)을 담당하지만, 전부 같은 하나의 physics target에 align.
4. **실험 결과**: mean-pool alignment으로 학습했을 때 flow_loss가 baseline (PhysREPA 없음)과 동일 (±0.02 차이, noise 범위).

### 새 방식: Timestep-wise Temporal Alignment

**핵심 아이디어**: REPA에서 spatial patch correspondence (DINOv2 patch n ↔ SiT token n)를 temporal timestep correspondence로 치환.

**물리적 motivation**: a_t는 본질적으로 velocity command (4Hz에서 0.25초간의 speed + direction). V-JEPA 2 PEZ가 encode하는 것도 speed + direction (R² 0.97). **같은 물리량을 다른 modality로 표현**한 것이므로, 같은 시점의 representation이 aligned되어야 한다.

각 action token t+i에 대해, 해당 시점을 포함하는 V-JEPA 2 window feature를 **개별** alignment target으로 사용:

```
action token t+1  ←align→  V-JEPA 2 feature from window covering t+1
action token t+2  ←align→  V-JEPA 2 feature from window covering t+2
...
action token t+16 ←align→  V-JEPA 2 feature from window covering t+16
```

이렇게 하면:
- 각 action token이 자기 시점에 맞는 고유한 physics target을 받음 (signal 희석 없음)
- State token (robot joint state)은 alignment에서 제외
- REPA의 patch-wise cosine sim 평균과 구조적으로 동일한 형태

---

## 구현 상세

### 1. V-JEPA 2 Feature 매칭 로직 변경

**현재**: 각 training sample에 대해 **하나의** V-JEPA 2 feature를 로드 (observation 시점의 window).

**변경**: 각 training sample에 대해 **H개의** V-JEPA 2 features를 로드 (각 action timestep에 대응하는 window).

V-JEPA 2 features는 `/mnt/md1/solee/features/vjepa2_vitl/`에 sliding window (stride 4)로 pre-extracted 되어 있다. Episode가 ~36 timesteps이면 window가 약 8개 존재 (window 0: frames [0-15], window 1: frames [4-19], ...).

**Timestep → Window 매칭**: action token t+i에 대해, 가장 가까운 (nearest) V-JEPA 2 window를 찾아 매칭. Stride 4이므로 action token 4개가 같은 V-JEPA window를 공유할 수 있음 — 이건 정상. 16개 전부 같은 target보다 훨씬 나음.

구체적 매핑 예시 (observation at frame t=20, stride 4):
- action t+1 (frame 21) → nearest window covering frame 21 → window starting at frame 8: [8-23] 또는 frame 12: [12-27]
- action t+4 (frame 24) → nearest window covering frame 24 → window starting at frame 12: [12-27]
- action t+16 (frame 36) → nearest window covering frame 36 → window starting at frame 24: [24-39]

매칭 전략: 각 action timestep의 frame index를 계산하고, pre-extracted windows 중 해당 frame을 포함하는 (또는 가장 가까운) window의 feature를 선택.

**DataLoader 수정**: 현재 (episode_id, timestep) → 1개 V-JEPA feature를 반환하는 구조를 → H개 features tensor (shape: [H, feature_dim])를 반환하도록 수정.

### 2. Alignment Loss 계산 변경

**현재**:
```python
# 각 align layer에서:
h = all_hidden_states[layer_idx + 1]     # [B, H+1, 1536]
h_pooled = h.mean(dim=1)                 # [B, 1536]
h_proj = proj(h_pooled)                  # [B, 1024]
loss += -cos_sim(h_proj, vjepa_target)   # vjepa_target: [B, 1024]
```

**변경**:
```python
# 각 align layer에서:
h = all_hidden_states[layer_idx + 1]      # [B, H+1, 1536]
h_actions = h[:, 1:, :]                   # [B, H, 1536] — state token 제외, action tokens만
h_proj = proj(h_actions)                  # [B, H, 1024] — 각 action token을 개별 project
# vjepa_targets: [B, H, 1024] — 각 action timestep에 대응하는 V-JEPA 2 feature
cos_sims = F.cosine_similarity(h_proj, vjepa_targets, dim=-1)  # [B, H]
loss += -cos_sims.mean()                  # H개 timestep의 cosine sim 평균
```

**Projection head 변경**: 현재 projection head가 (1536 → 1024) mapping인데, 이건 그대로 사용 가능. 각 action token에 **동일한** projection head를 share해서 적용 (token 위치와 무관한 mapping).

### 3. Global Mean Centering 유지

이전 실험에서 확인된 mean bias 문제 (raw cosine sim 0.989)는 여전히 존재. V-JEPA 2 features를 로드할 때 반드시 global mean을 빼서 centered features를 사용할 것. Global mean은 `/mnt/md1/solee/features/vjepa2_vitl/global_means.pt`에 저장되어 있음.

### 4. 나머지는 동일

- align_layers: [0,1,2,3,4,5,6,7,8] (DiT first 9 layers)
- vjepa_layer: 10 (V-JEPA 2-L PEZ peak)
- lambda_repa: 0.5
- 각 layer마다 독립 projection head (2-layer MLP)
- V-JEPA 2 target은 .detach()

---

## 환경

- **GR00T 코드**: `/home/solee/Isaac-GR00T/`
- **V-JEPA 2 pre-extracted features**: `/mnt/md1/solee/features/vjepa2_vitl/`
- **Global means**: `/mnt/md1/solee/features/vjepa2_vitl/global_means.pt`
- **GPU**: A6000 48GB × 4

---

## WandB Logging

기존 logging에 추가:
- `train/repa/per_timestep_cosine_sim`: H개 timestep별 cosine sim의 분포 (mean, min, max) — 특정 timestep에서만 alignment이 잘 되는지 확인

## Sanity Check

학습 시작 전 또는 첫 몇 step에서:
1. `vjepa_targets` shape이 [B, H, 1024]인지 확인
2. 같은 batch 내에서 서로 다른 timestep의 target이 실제로 다른지 확인 (stride 4 내에서는 같을 수 있지만, stride 간격을 넘으면 달라야 함)
3. Centered features인지 확인 (mean cosine sim ~0)

---
*PhysREPA Timestep-wise Alignment Implementation Prompt — 2026-03-18*