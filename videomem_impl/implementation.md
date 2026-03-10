# DepthMem: Implementation Prompt for Claude Code

---

## 1. 연구 개요 (Top-Down)

### 1.1 무엇을 만드는가

**DepthMem**은 GR00T N1.6 VLA에 두 가지를 추가하는 연구입니다:

1. **Depth Channel**: Video Depth Anything Small이 생성한 temporally consistent depth map을 SigLIP2 vision encoder의 4번째 input channel로 추가
2. **Spatiotemporal Attention**: MEM(Physical Intelligence, 2026) 논문의 video encoder 기법을 SigLIP2에 적용하여, T개 과거 프레임의 RGB+D 정보를 temporal attention으로 압축

이 두 가지가 합쳐지면: "T개 과거 프레임의 temporally consistent depth가 temporal attention을 통해 current frame의 representation에 압축" → VLA가 depth dynamics (depth velocity, time-to-contact)를 implicit하게 학습 → 정밀한 grasping과 self-correction 능력 획득.

### 1.2 왜 하는가

GR00T N1.6은 green cube stacking에서 99% grasping 실패. 원인: systematic depth bias (물체보다 카메라 쪽에서 gripper를 닫음). Denoising steps 4→1024 증가해도 개선 없음 → action head가 아닌 VLM representation 문제.

기존 depth+VLA 논문들(DepthVLA, SpatialVLA 등 6+개)은 전부 **per-frame independent depth** → temporal consistency 없음 → depth velocity나 time-to-contact 계산 불가. 우리는 Video Depth Anything의 temporally consistent depth + MEM-style temporal attention으로 이 gap을 채움.

### 1.3 베이스 모델 구조 (GR00T N1.6)

```
현재 GR00T N1.6 inference:

RGB [3, 224, 224] (single frame)
  ↓
SigLIP2 Vision Encoder (27 layers, 1152-dim, patch_size=14)
  - Patch Embedding: nn.Linear(3×14×14=588, 1152)
  - Window attention (14×14) at most layers
  - Full attention at layers [7, 14, 21, 26]
  - RoPE 2D positional embedding
  - Output: [num_patches=256, 1152]
  ↓
Eagle3 VLM (Qwen3-1.7B, 28 layers, 2048-dim)
  - MLP connector maps 1152 → 2048
  - Text tokens + image patch tokens + proprio tokens
  ↓
DiT Action Head (32 layers, Flow Matching, ~860M)
  - Gradient가 VLM으로 역전파 안됨
  - State-relative action chunks 출력
  ↓
Robot Action
```

---

## 2. 수정된 아키텍처 (DepthMem)

```
DepthMem inference (with temporal attention):

과거 T frames: RGB [T, 3, 224, 224]
  ↓
Video Depth Anything Small (frozen, ~9ms for 16 frames @224×224, torch.compile)
  ↓
Depth maps [T, 1, 224, 224] (temporally consistent)
  ↓
Channel concat per frame: [T, 4, 224, 224]  (RGB + Depth)
  ↓
SigLIP2 Vision Encoder (수정됨)
  - Patch Embedding: nn.Linear(4×14×14=784, 1152)    ← 변경 ①
  - Layers 0-6: Spatial attention (각 frame 독립)
  - Layer 7: Spatial + Temporal attention              ← 변경 ②
  - Layers 8-13: Spatial attention
  - Layer 14: Spatial + Temporal attention              ← 변경 ②
  - Layers 15-20: Spatial attention
  - Layer 21: Spatial + Temporal attention              ← 변경 ②
  - Layers 22-25: Spatial attention
  - Layer 26: Spatial + Temporal attention              ← 변경 ②
  - 최종: 과거 frame tokens DROP, current frame tokens만 출력
  - Output: [num_patches=256, 1152]  (token 수 변화 없음!)
  ↓
Eagle3 VLM (Qwen3-1.7B) — 변경 없음
  ↓
DiT Action Head — 변경 없음
  ↓
Robot Action
```

### 2.1 변경 ① — Patch Embedding (4ch)

```python
# 기존
self.patch_embedding = nn.Linear(3 * 14 * 14, 1152)   # nn.Linear(588, 1152)

# 변경
self.patch_embedding = nn.Linear(4 * 14 * 14, 1152)   # nn.Linear(784, 1152)
```

Config에서 `num_channels: 3 → 4`로 변경하면 자동 반영.

Weight 초기화:
```python
new_weight[:, :588] = pretrained_weight   # RGB 부분 복사
new_weight[:, 588:] = 0                   # Depth 부분 zero-init
```

### 2.2 변경 ② — Spatiotemporal Attention (MEM 방식)

**MEM 논문의 핵심 설계를 그대로 적용:**

SigLIP2의 full attention layers [7, 14, 21, 26]에 temporal attention 추가.

```
기존 SigLIP2 Layer (spatial only):
  Input: [B*T, num_patches, 1152]
  → Spatial Attention: 각 frame 내 패치 간 bidirectional attention
  → FFN
  → Output: [B*T, num_patches, 1152]

수정된 Layer (spatial + temporal):
  Input: [B*T, num_patches, 1152]
  → Spatial Attention: 각 frame 내 패치 간 bidirectional attention (기존과 동일)
  → Temporal Attention (추가):
      - reshape: [B, T, num_patches, 1152]
      - 각 patch position (i)에 대해: 시간축 T에 대한 causal self-attention
      - 즉 patch i의 frame t는 frame 0..t까지만 attend (causal mask)
      - reshape back: [B*T, num_patches, 1152]
  → FFN
  → Output: [B*T, num_patches, 1152]
```

**Temporal Attention의 핵심 특성 (MEM 논문에서 가져옴):**
- **New learnable parameter: 0개.** 기존 spatial attention의 QKV weight를 temporal에서도 재사용.
- **Temporal position encoding**: Fixed sinusoidal encoding을 token에 더함 (not learned).
  - t=0 (현재 frame)일 때 sinusoidal encoding = 0 → single-frame 모델과 동일한 output 보장
- **Additive**: spatial attention output에 temporal attention output을 더함 (residual 방식)
- **Complexity**: O(Kn² + nK²) — joint space-time O(n²K²) 대비 훨씬 효율적
  - K = num frames, n = num patches

```python
# Temporal Attention 의사코드
def temporal_attention(x, T, num_patches):
    """
    x: [B*T, num_patches, dim]
    """
    B_T, N, D = x.shape
    B = B_T // T

    # reshape to [B, T, N, D]
    x = x.view(B, T, N, D)

    # Add sinusoidal temporal position encoding
    # t=0 current frame → encoding=0, t=-1 → encoding=sin/cos(1), ...
    temp_pos = sinusoidal_temporal_encoding(T, D)  # [T, D], fixed, not learned
    x = x + temp_pos.unsqueeze(0).unsqueeze(2)     # [B, T, N, D]

    # Per-patch temporal attention with causal mask
    # transpose to [B*N, T, D] for attention along T dimension
    x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)

    # Self-attention along T (causal mask: frame t can attend to 0..t)
    causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()  # upper tri = True = masked
    attn_output = self_attention(x, attn_mask=causal_mask)  # reuse spatial attn QKV weights

    # reshape back to [B*T, N, D]
    attn_output = attn_output.reshape(B, N, T, D).permute(0, 2, 1, 3).reshape(B * T, N, D)

    return attn_output
```

**최종 layer 후 past frame token drop:**
```python
# encoding 완료 후
# x: [B*T, num_patches, 1152]
x = x.view(B, T, num_patches, 1152)
x = x[:, -1, :, :]  # current frame만 선택: [B, num_patches, 1152]
# 이 tokens만 Qwen3으로 전달 → token 수 변화 없음
```

### 2.3 추론 시 Feature Caching (효율성)

학습 시에는 T frames를 한꺼번에 처리하면 됨. 추론 시에는 매 step마다 T frames를 다시 인코딩하는 건 비효율적이므로 **과거 frame의 intermediate features를 캐싱:**

```
추론 Step t:
  1. Video Depth Anything: RGB buffer [T frames] → Depth maps (9ms, 전체 재처리 필요)
  2. SigLIP2:
     - 과거 frames (t-K ~ t-1): 각 temporal attention layer의 KV가 이미 캐싱됨
     - 현재 frame (t): patch embedding + spatial attention (fresh 계산)
     - Temporal attention: current frame Q × cached past KV → attend
     - Current frame의 KV를 캐시에 추가, 가장 오래된 것 제거
  3. Current frame tokens만 Qwen3으로 전달

캐시 구조 (각 temporal attention layer마다):
  cached_keys:   [B, T-1, num_patches, head_dim]
  cached_values: [B, T-1, num_patches, head_dim]
```

이 방식으로 추론 시 SigLIP2의 spatial attention은 **current frame 1개만** 계산하면 됨. Temporal attention만 cached past에 attend.

---

## 3. Training Strategy

### 3.1 학습 대상 파라미터

> **[2026-03-11 구현 결과 업데이트]** 실제 측정값으로 수정. LoRA target_modules는 Eagle3 코드에 이미 있는 `wrap_backbone_lora()`/`wrap_llm_lora()` 메서드를 사용.

| 컴포넌트 | 학습 방식 | 파라미터 수 (실측) | 비고 |
|---|---|---|---|
| Video Depth Anything Small | **Frozen** (항상) | 0 | Offline 사전 생성, 학습에 로드 불필요 |
| Patch Embedding (4ch 전체) | **Full fine-tune** | **904K** | RGB+Depth 전체 trainable (depth부분 zero-init) |
| SigLIP2 (attention + MLP) | **LoRA** (rank=16) | **8.9M** | target: q/k/v/out_proj + mlp.fc1/fc2 |
| SigLIP2 나머지 | **Frozen** | 0 | |
| Temporal Position Encoding | **Fixed** (not learned) | 0 | Sinusoidal, MEM과 동일 |
| Qwen3 (LM backbone) | **LoRA** (rank=16) | **10.0M** | target: q/k/v/o_proj + mlp gate/down/up |
| MLP Connector (1152→2048) | **Frozen** | 0 | |
| DiT Action Head | **Fine-tune** | **1,419M** (기존과 동일) | GR00T 표준 post-training 방식 |

**총 새로운 학습 파라미터: ~19.8M (LoRA 18.9M + patch embedding 0.9M)**
DiT 1,419M은 기존 GR00T fine-tuning에서도 학습하는 부분이므로 추가 비용 아님.
전체 3.31B 파라미터 중 trainable 1,438M (43.5%), frozen 1,867M (56.5%).

### 3.2 학습 데이터

- 기존 GR00T fine-tuning 데이터 (BridgeData V2 등)
- **추가 작업**: 모든 RGB 프레임에 대해 Video Depth Anything Small로 depth map을 offline 사전 생성
  - Episode 단위로 처리 (temporal consistency 보장)
  - 저장 형식: `.npy`, float16, [H, W], range [0, 1]
  - 16 frames @ 224×224 → ~9ms → 10만 프레임도 ~15분

### 3.3 학습 절차

> **[2026-03-11 구현 반영]** 실제 구현된 학습 파이프라인으로 업데이트.

```
Step 1: Depth Map 사전 생성 (완료/진행 중)
  - scripts/generate_depth_maps.py: 4 GPU 병렬, ~22 fps/GPU
  - Episode 단위 Video Depth Anything Small 실행 (temporal consistency)
  - 출력: {depth_dir}/chunk-{:03d}/episode_{:06d}/frame_{:06d}.npy (float16, [256,256])
  - BridgeData V2: 53,192 episodes, ~1.89M frames, 약 5시간 소요

Step 2: Pre-trained Weight 로딩 + 확장 (setup.py _enable_depthmem)
  - Bridge-finetuned checkpoint 로드 (3ch, from_pretrained)
  - Patch embedding: nn.Linear(588,1152) → nn.Linear(784,1152) 동적 교체
    - RGB weights 복사, depth weights zero-init
  - Encoder temporal_attention_layers 설정: {7, 14, 21, 26}
  - LoRA 적용: wrap_backbone_lora(r=16) + wrap_llm_lora(r=16)
  - Patch embedding requires_grad=True 유지 (PEFT가 freeze하므로 명시적 설정)

Step 3: Fine-tuning
  - Data: RGB+Depth 4ch, T=16 frames per sample (15 past + 1 current)
  - Video backend: ffmpeg (bridge는 AV1 codec, torchcodec 미지원)
  - SigLIP2: T frames를 한꺼번에 처리 (spatial + temporal attention)
  - 과거 frame tokens drop → current tokens만 Qwen3으로 전달
  - Loss: 기존 GR00T action prediction loss (flow matching MSE)
  - Optimizer: AdamW, LR=1e-4, warmup_ratio=0.05, weight_decay=1e-5
  - Batch: global_batch_size=256, gradient_checkpointing=True
  - Schedule: 20K steps, save every 1K steps
  - Hardware: A6000 48GB × 4 (ZeRO-2)
  - Augmentation: depth maps에 RGB와 동일한 spatial transform (albumentations replay)
  - Logging: wandb (project: finetune-gr00t-n1d6)
```

---

## 4. 레포지토리 구조 & 수정 파일

```
Isaac-GR00T/  (dmemory branch)
├── Video-Depth-Anything/                    # submodule (이미 추가됨)
├── gr00t/
│   ├── model/
│   │   ├── modules/
│   │   │   ├── nvidia/Eagle-Block2A-2B-v2/
│   │   │   │   ├── config.json              # ★ num_channels: 3→4
│   │   │   │   └── modeling_siglip2.py      # ★ temporal attention 추가
│   │   │   │                                #   Siglip2VisionTransformer 클래스에
│   │   │   │                                #   temporal attn 로직 추가
│   │   │   │                                #   convert_images_to_patches()는
│   │   │   │                                #   num_channels를 config에서 읽으므로
│   │   │   │                                #   자동 반영될 것 (확인 필요)
│   │   │   ├── eagle_backbone.py            # ★ T frames 입력 지원
│   │   │   │                                #   SigLIP2에 T frames 넘기고
│   │   │   │                                #   current frame tokens만 반환
│   │   │   ├── dit.py                       #   (변경 없음)
│   │   │   └── flowmatching_modules.py      #   (변경 없음)
│   │   ├── gr00t_n1d6/
│   │   │   ├── gr00t_n1d6.py               # ★ weight 로딩 시 patch embedding 확장
│   │   │   │                                #   LoRA 설정 추가
│   │   │   ├── processing_gr00t_n1d6.py    # ★ depth concat in _get_vlm_inputs()
│   │   │   │                                #   T frames 로딩 처리
│   │   │   └── image_augmentations.py       # ★ depth-aware augmentation
│   │   └── registry.py
│   ├── data/
│   │   ├── dataset/                         # ★ depth map 로딩 추가
│   │   │                                    #   episode 내 T frames 샘플링
│   │   └── types.py                         # ★ depth 필드 추가
│   ├── configs/
│   │   ├── model/                           # ★ depth + temporal config
│   │   └── training/                        # ★ LoRA config
│   ├── experiment/
│   │   └── trainer.py                       #   (변경 최소)
│   └── policy/                              # ★ 추론 시 depth buffer + feature cache
├── scripts/
│   └── generate_depth_maps.py               # ★ 신규: offline depth 생성
└── CLAUDE.md                                #   (참고용)
```

---

## 5. 구현 작업 목록 (순서대로)

### 작업 1: SigLIP2에 Temporal Attention 추가

**파일**: `gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/modeling_siglip2.py`

이것이 가장 핵심이고 가장 복잡한 작업.

**5.1.1** `Siglip2VisionConfig`에 temporal 관련 config 추가:
```python
num_temporal_frames: int = 6          # 과거 프레임 수 (5 past + 1 current)
temporal_attention_stride: int = 7    # 매 N번째 layer에서 temporal attention
                                      # (layers 7, 14, 21, 26에서 동작하도록)
temporal_attention_layers: List[int] = [7, 14, 21, 26]  # 또는 명시적으로 지정
```

**5.1.2** Sinusoidal Temporal Position Encoding 함수 추가:
```python
def get_sinusoidal_temporal_encoding(num_frames, dim):
    """
    Fixed sinusoidal encoding for temporal positions.
    Current frame (t=0) → encoding = 0 (중요! single-frame과 동일한 output 보장)
    Past frames → non-zero encoding
    Returns: [num_frames, dim]
    """
    positions = torch.arange(num_frames - 1, -1, -1)  # [T-1, T-2, ..., 1, 0]
    # t=0 (current) → position 0 → sin(0)=0, cos(0)=1... 하지만 MEM은 encoding=0으로 처리
    # MEM 방식: position 0일 때 전체 encoding을 0으로 설정
    encoding = torch.zeros(num_frames, dim)
    for i in range(num_frames - 1):  # past frames만
        pos = positions[i].float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        encoding[i, 0::2] = torch.sin(pos * div_term)
        encoding[i, 1::2] = torch.cos(pos * div_term)
    # encoding[-1] (current frame) = 0 vector
    return encoding
```

**5.1.3** `Siglip2EncoderLayer`(또는 해당 클래스)에 temporal attention 로직 추가:
- Full attention layers [7, 14, 21, 26]에서만 temporal attention 활성화
- **기존 spatial attention의 QKV projection weight를 재사용** (new param 0개)
- Spatial attention output + temporal attention output (additive residual)

**5.1.4** `Siglip2VisionTransformer.forward()`에서:
- Input shape 변경: [B, C, H, W] → [B, T, C, H, W] 또는 [B*T, C, H, W]
- 모든 T frames를 patch embedding 통과
- Spatial+temporal attention layers 처리
- 최종: past frame tokens drop, current tokens만 반환

### 작업 2: Patch Embedding 4ch 확장

**파일**: `config.json`, `modeling_siglip2.py`

- config.json: `"num_channels": 4`
- modeling_siglip2.py: `convert_images_to_patches()`가 `num_channels`를 config에서 읽는지 확인
- 하드코딩된 `3`이 있으면 `self.config.num_channels`로 교체

### 작업 3: Weight 로딩 & LoRA 설정

**파일**: `gr00t/model/gr00t_n1d6/gr00t_n1d6.py`

3ch→4ch patch embedding 확장 함수:
```python
def expand_patch_embedding_for_depth(state_dict):
    """
    Pre-trained 3ch weight를 4ch로 확장.
    정확한 state_dict key는 model.state_dict().keys()에서 확인 필요.
    """
    # patch embedding key 찾기 (예: "backbone.vision_model.embeddings.patch_embedding.weight")
    for key in state_dict:
        if "patch_embedding.weight" in key:
            old_w = state_dict[key]  # [1152, 588]
            new_w = torch.zeros(1152, 784, dtype=old_w.dtype, device=old_w.device)
            new_w[:, :588] = old_w
            state_dict[key] = new_w
            print(f"Expanded {key}: {old_w.shape} → {new_w.shape}")
    return state_dict
```

LoRA 설정 (PEFT 라이브러리 사용, 이미 dependency에 있음):
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj",  # SigLIP2 attention
        # Qwen3 attention도 포함 — 정확한 module name은 코드에서 확인
    ],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
```

### 작업 4: Eagle Backbone 수정

**파일**: `gr00t/model/modules/eagle_backbone.py`

Eagle backbone이 SigLIP2를 호출하는 wrapper. T frames 입력을 처리하도록 수정:
```python
def forward(self, pixel_values, ...):
    # pixel_values: [B*T, 4, 224, 224] 또는 [B, T, 4, 224, 224]
    # SigLIP2에 전달
    # SigLIP2가 temporal attention 처리 후 current frame tokens만 반환
    # 반환: [B, num_patches, 1152]
```

### 작업 5: Data Pipeline 수정

**파일**: `gr00t/data/dataset/`, `gr00t/data/types.py`

- `VLAStepData`에 depth 필드 추가
- Dataset 로더에서 T frames 샘플링 + 대응하는 depth map 로딩
- Episode 내에서 1Hz stride로 T frames 선택 (MEM default: T=6, 1초 간격)

```python
# 각 training step에서:
# 1. Episode에서 current frame t 선택
# 2. Past frames: t-5, t-4, t-3, t-2, t-1 (1Hz stride)
# 3. 각 frame에 대해 RGB + pre-generated depth map 로딩
# 4. RGB 3ch + Depth 1ch = 4ch per frame
# 5. T=6 frames → [6, 4, 224, 224] tensor
```

### 작업 6: Image Processing & Augmentation

**파일**: `processing_gr00t_n1d6.py`, `image_augmentations.py`

- `_get_vlm_inputs()`: T frames의 RGB+Depth를 [T, 4, 224, 224]로 구성
- Augmentation: color augmentation은 RGB 3ch에만, spatial augmentation은 4ch 전체에 동일 적용
- **중요**: Eagle3 processor에서 PIL Image 변환하는 부분 주의. 4ch tensor가 PIL로 변환되면 depth 손실. 이 경로를 우회하거나 수정 필요.

### 작업 7: Offline Depth 생성 스크립트

**파일**: `scripts/generate_depth_maps.py` (신규 생성)

```python
"""
학습 데이터의 모든 RGB 프레임에 대해 Video Depth Anything Small로 depth 생성.

반드시 EPISODE 단위로 처리해야 temporal consistency 보장.
(프레임 단독 처리 시 per-frame depth와 동일 → 의미 없음)

Usage:
    python scripts/generate_depth_maps.py \
        --data_dir /path/to/lerobot_dataset \
        --model_size small \
        --resolution 224

출력: 각 frame에 대응하는 depth map
  - 형식: .npy, float16
  - Shape: [H, W]
  - Range: [0, 1] (min-max normalized per episode)
  - 경로: 원본 image 경로 기반 (예: images/ → depths/)
"""
```

### 작업 8: Config 통합

**파일**: `gr00t/configs/model/`, `gr00t/configs/training/`

```yaml
# Model config
depthmem:
  enabled: true
  num_channels: 4                    # RGB(3) + Depth(1)
  depth_mode: "offline"              # "offline" or "online"
  temporal:
    enabled: true
    num_frames: 6                    # 5 past + 1 current
    stride: 1                        # 1 second stride (at data collection Hz)
    attention_layers: [7, 14, 21, 26]  # full attention layers in SigLIP2

# Training config
lora:
  rank: 16
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj"]
  dropout: 0.05

training:
  patch_embedding_depth_lr: 1e-3     # depth 부분은 zero에서 시작하므로 높은 LR
  lora_lr: 1e-4                      # LoRA는 일반적인 fine-tuning LR
  dit_lr: 1e-4                       # DiT action head
```

### 작업 9: Inference Pipeline

**파일**: `gr00t/policy/`

추론 시 두 가지 캐시 관리:
1. **RGB frame buffer**: Video Depth Anything에 넘기기 위한 최근 T frames
2. **SigLIP2 temporal KV cache**: 과거 frame의 attention KV (temporal attention layers마다)

```python
class DepthMemPolicy:
    def __init__(self, model, depth_model, num_frames=6):
        self.model = model
        self.depth_model = torch.compile(depth_model)  # Video Depth Anything Small
        self.rgb_buffer = deque(maxlen=num_frames)
        self.temporal_kv_cache = {}  # layer_idx → (cached_keys, cached_values)

    def get_action(self, rgb_frame, language_instruction):
        # 1. RGB buffer에 추가
        self.rgb_buffer.append(rgb_frame)

        # 2. Video Depth Anything: 전체 buffer 재처리 (temporal consistency 보장)
        rgb_stack = torch.stack(list(self.rgb_buffer))  # [T, 3, 224, 224]
        depth_maps = self.depth_model(rgb_stack)         # [T, 1, 224, 224], ~9ms

        # 3. RGB + Depth concat
        rgbd_stack = torch.cat([rgb_stack, depth_maps], dim=2)  # [T, 4, 224, 224]

        # 4. SigLIP2 forward (temporal KV cache 사용)
        current_tokens = self.model.vision_encode(
            rgbd_stack,
            temporal_kv_cache=self.temporal_kv_cache  # 과거 frame KV 재사용
        )
        # current_tokens: [1, 256, 1152]

        # 5. 나머지 forward (Qwen3 → DiT)
        action = self.model.predict_action(current_tokens, language_instruction)
        return action
```

---

## 6. 기술적 주의사항

### 6.1 SigLIP2 Patch Embedding은 nn.Linear (Conv2d 아님)
이미지를 `convert_images_to_patches()`로 flatten한 후 Linear projection.
`config.num_channels` 변경 시 자동 반영되어야 하지만, 하드코딩된 `3`이 있으면 수정 필요.

### 6.2 Eagle3 Processor의 PIL Image 변환 — 해결됨

> **[2026-03-11]** 확인 결과: 4ch RGBD를 PIL로 변환하는 문제는 발생하지 않음.
> 구현 방식: RGB 3ch만 Eagle processor를 통과 → depth는 별도로 collator에서 수집 → model의 `_concat_depth_to_pixel_values()`에서 normalized RGB pixel_values에 depth를 4번째 채널로 concat.
> Depth는 PIL 변환을 우회하며, `F.interpolate`로 RGB spatial size에 맞춰 resize됨.

### 6.3 Temporal Attention에서 QKV Weight 재사용
MEM의 핵심 설계: temporal attention에 **별도의 QKV weight를 두지 않음**. Spatial attention의 QKV weight를 그대로 재사용. 이것이 "new learnable parameter 0개"를 가능하게 함. 구현 시 같은 projection layer를 두 번 호출하는 방식으로 구현.

### 6.4 Video Depth Anything의 Episode 단위 처리
- 학습: offline으로 episode 전체를 한번에 처리하여 depth 생성/저장
- 추론: frame buffer 유지, 매 step마다 전체 buffer를 Video Depth Anything에 넘김 (9ms)
- 단일 프레임만 넣으면 temporal consistency 없음 → per-frame depth estimator와 동일 → 의미 없음

### 6.5 LoRA Target Modules — 확인 완료

> **[2026-03-11]** Eagle3 코드에 이미 `wrap_backbone_lora()`와 `wrap_llm_lora()` 메서드가 있음 (modeling_eagle3_vl.py line 150-173). 별도 구현 불필요.
> - SigLIP2: `['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj', 'mlp.fc1', 'mlp.fc2']`
> - Qwen3: `['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']`

### 6.6 DiT Action Head의 Gradient — ~~역전파 안됨~~ → **역전파 됨** (수정)

> **[2026-03-11 코드 검증 결과]** 원래 문서에서 "DiT gradient가 VLM backbone으로 역전파되지 않음"이라고 기술했으나, **이것은 틀렸음.**
>
> 실제 코드(`gr00t_n1d6.py` line 597-598): `backbone_outputs = self.backbone(backbone_inputs)` → `action_outputs = self.action_head(backbone_outputs, action_inputs)` — **detach() 없음.**
>
> Action head의 `forward()` (line 262): `encoder_hidden_states=vl_embeds`로 DiT cross-attention에 backbone output이 직접 들어감. Loss에서 backward 시 gradient가 VLM backbone까지 흐름.
>
> **Gradient flow 경로:**
> ```
> Action Loss (MSE) → DiT (cross-attention) → Qwen3 LoRA → MLP Projector → SigLIP2 LoRA
>   → Current frame: 직접 gradient
>   → Past 15 frames: temporal attention을 통한 간접 gradient
>   → Patch Embedding (4ch): depth channel 학습
> ```
>
> 따라서 LoRA가 action loss에서 직접 학습 신호를 받으며, 별도의 VLM loss 없이도 DepthMem이 학습됨.

---

## 7. 검증 체크리스트

```
모델 수정: (2026-03-11 검증)
- [x] 4ch input [16, 4, 224, 224]이 SigLIP2를 정상 통과 ✓
- [x] T=16 frames temporal attention 포함 정상 통과 ✓
- [x] Temporal attention 후 current frame tokens만 출력 (shape: [1, 256, 1152]) ✓
- [x] 3ch checkpoint → 동적 patch embedding 교체 (detach 없이 from_pretrained 후 nn.Linear 교체) ✓
      ⚠️ config.json을 4ch로 바꾸면 from_pretrained에서 size mismatch 크래시 → 반드시 3ch 유지 후 동적 교체
- [x] Missing keys: 0, Mismatched keys: 0 확인 ✓
- [ ] T=1 single frame backward compatibility 미검증
- [x] LoRA 정상 적용: SigLIP2 8.9M + Qwen3 10.0M + patch_embed 904K ✓

데이터: (2026-03-11 검증)
- [x] Depth map .npy 로딩 정상 (256x256, float16) ✓
- [x] ffmpeg video backend으로 AV1 디코딩 정상 ✓
- [x] T=16 frames 로딩: video delta_indices=[-15,...,0] ✓
- [x] Spatial augmentation (albumentations replay)을 depth에도 동일 적용 ✓
- [x] Color jitter는 depth에 미적용 (올바른 동작) ✓

학습:
- [ ] 학습 정상 시작 및 loss 감소 (depth 생성 완료 후 자동 실행 대기 중)
- [ ] Depth channel의 patch embedding weight가 0에서 점점 커지는지

추론:
- [x] depthmem_policy.py: RGB buffer + VDA depth 생성 구현 ✓
- [ ] Temporal KV cache 추론 시 model forward에 전달 미구현 (학습에는 영향 없음)
- [ ] 추론 latency 미측정
```

---

## 8. 구현 우선순위

```
Week 1: 핵심 모델 수정
  1. SigLIP2 temporal attention 구현 (작업 1 — 가장 중요, 가장 복잡)
  2. Patch embedding 4ch 확장 (작업 2)
  3. Weight 로딩 + LoRA 설정 (작업 3)
  4. Eagle backbone 수정 (작업 4)
  5. 단위 테스트: forward pass 검증

Week 2: 데이터 & 학습
  6. Offline depth 생성 스크립트 (작업 7)
  7. Data pipeline 수정 (작업 5)
  8. Augmentation 수정 (작업 6)
  9. Config 통합 (작업 8)
  10. Fine-tuning 실행

Week 3: 추론 & 평가
  11. Inference pipeline + KV cache (작업 9)
  12. 실시간 테스트
  13. Baseline 비교 실험
```