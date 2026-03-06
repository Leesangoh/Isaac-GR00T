# Claude Code Prompt: Intent-Conditioned Cerebellar Action Correction 구현

---

## 개요

GR00T N1.6 VLA의 action chunk를 real-time으로 보정하는 **Intent-Conditioned Cerebellar Correction Module**을 구현한다.

이 모듈은 사람의 소뇌(cerebellum)에서 영감을 받았으며, **prediction error 기반**으로 action을 보정한다.

**핵심 차별점**: 이전 구현(action-conditioned forward model)과 달리, Forward model이 **intent (VLM hidden state)**를 받아서 "이 goal을 향해 진행 중이면 다음 상태는 이래야 한다"를 예측한다. Action은 Forward model에 **들어가지 않는다**.

---

## 이전 구현이 실패한 이유 (필독!)

이전에 3가지 approach를 시도했고 **모두 실패**했다. 새 구현은 이 근본 원인을 해결하기 위한 것이다.

### 실패 원인 1: Action-Conditioned Forward Model의 한계

이전 Forward model: `(z_t, action, proprio) → Δz_predicted`

이 모델이 잘 학습되면:
- VLA가 **어떤** action을 내든 (맞든 틀리든), forward model은 "그 action을 실행하면 이렇게 변하겠네" 하고 **정확하게 예측**
- prediction_error = actual_z_t+1 - predicted_z_t+1 ≈ 0 (항상!)
- Correction network: "수정할 게 없네" → **VLA가 실패해도 correction이 0**

**비유**: 사람이 "이 방향으로 걸으면 저기 도착해"라고 예측하는데, 실제로 저기 도착하면 "맞네!" → 하지만 그 방향이 **목적지와 반대**여도 예측 자체는 맞음. 목적지(intent)를 모르니까.

### 실패 원인 2: Perturbation-based Phase 2의 분포 불일치

이전 Phase 2: expert action + random noise → correction target = -noise

- 학습: random Gaussian noise를 보정하는 법을 학습
- Inference: VLA의 **systematic bias** (특정 방향으로 치우침)를 보정해야 함
- 결과: 모든 방법에서 **constant bias correction**만 출력 (상황과 무관하게 항상 같은 Δa)

### 실패 원인 3: 실험 결과 요약

| 방법 | Forward Model | Phase 2 | 결과 |
|------|-------------|---------|------|
| Frozen DINOv2 (CLS token) | val_vis=0.360 (baseline 0.417, 13.8% 개선) | Perturbation | Constant bias |
| Frozen DINOv2 (patch mean) | val_vis=0.079 (baseline 0.095, 16.8% 개선) | Perturbation | Constant bias |
| R3M frozen | val_vis=0.000267 (baseline 0.000264, **-1%**) | - | R3M features 자체가 temporally invariant → 실패 |
| Jointly DINOv2+LoRA (VICReg λ=5) | val_vis=0.00019 | Perturbation | Temporal collapse (Δz norm 49x 감소) + Constant bias |
| EMA DINOv2+LoRA (τ=0.996) | val_vis=0.002, dz≈0.88, z_std≈1.07 | Perturbation | Forward model은 가장 좋았지만, Phase 2에서 여전히 constant bias. 로봇 팔 난동. |

### 해결책: Intent-Conditioned Forward Model

Forward model에서 **action을 제거**하고 **intent (VLM hidden state)**를 넣는다:

`Forward model: (z_t, proprio, intent) → Δz_ideal`

- intent가 "이 goal을 향해 가야 해"를 알려줌
- Forward model은 "지금 상태에서 이 goal을 향해 가면 다음 상태는 이래야 해"를 예측
- VLA가 잘못된 action을 내면 → 실제 z_t+1 ≠ predicted → **의미 있는 prediction error**
- Correction network가 이 error를 보고 action을 보정

**생물학적 근거**: 실제 소뇌의 mossy fibers는 efference copy + prefrontal/parietal cortex의 goal/intent 정보를 **함께** 전달한다. 소뇌의 forward model도 intent를 받아서 예측한다.

---

## GR00T N1.6 아키텍처 (확인 완료)

```
Gr00tN1d6
├── backbone: EagleBackbone (Eagle3_VLForConditionalGeneration)
│   ├── vision_model: Siglip2VisionModel (1152-dim, 27 layers, patch=14)
│   ├── mlp1: Sequential (vision projector, downsample 0.5 → 81 image tokens)
│   └── language_model: Qwen3ForCausalLM (2048-dim, truncated 28→16 layers)
│
└── action_head: Gr00tN1d6ActionHead
    ├── vlln: LayerNorm(2048)
    ├── state_encoder: CategorySpecificMLP (128 → 1536)
    ├── model: AlternateVLDiT (32 layers, 1024-dim, 4 denoising steps)
    └── action_decoder: CategorySpecificMLP (1024 → 128)

VLM Output: (B, 108, 2048) — 81 image tokens + 27 text tokens
D_intent = 2048 (backbone_features mean pooled with attention_mask)
Action: 7-dim (x, y, z, roll, pitch, yaw, gripper), 8-step chunk
Proprio/State: 8-dim (x, y, z, roll, pitch, yaw, pad, gripper)
Total Params: 3.29B (Backbone 1.87B + Action Head 1.42B)
```

---

## 아키텍처 상세

### 전체 Inference Flow

```
┌──────────────────────────────────────────────────────────────┐
│  GR00T N1.6 (완전 Frozen, 3.29B params)                       │
│                                                              │
│  image(224×224) + language                                    │
│    → SigLIP2 → (B,324,1152) → Eagle MLP → (B,81,2048)        │
│    → Qwen3 (16 layers) → (B,27,2048)                         │
│    → concat → backbone_features: (B,108,2048)                │
│                          │                                    │
│                    intent_vec [2048]                           │
│                    (backbone hook + attn_mask mean pooling)    │
│                                                              │
│    → VLLN → DiT (32 layers, 4 denoising steps)               │
│                          │                                    │
│                    action_vla (8-step chunk, 7-dim)            │
│                                                              │
│  ※ 매 action chunk마다 1회 forward pass                       │
│  ※ intent_vec은 chunk 내 8 step 동안 동일하게 사용 (1.6초)      │
│     → chunk 내 staleness는 허용 (VLA도 동일 observation으로 계획) │
└───────────────────┬──────────────────────┬───────────────────┘
                    │                      │
                    ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Cerebellar Module (매 control step 실행, 목표 < 5ms)          │
│                                                              │
│  Step 1: DINOv2+LoRA (online encoder) → z_t [384]            │
│                                                              │
│  Step 2: Forward Model (intent-conditioned, action 없음!)     │
│          (z_t [384], proprio_t [8], intent_vec [2048])         │
│          → Δz_ideal [384]                                     │
│          "이 intent를 향해 가면 다음 변화는 이래야 해"            │
│                                                              │
│  Step 3: action_vla_t 실행 → 다음 image 관측 → z_t+1           │
│          prediction_error = (z_t+1 - z_t) - Δz_ideal          │
│                                                              │
│  Step 4: Correction Network (intent 받지 않음!)                │
│          (prediction_error [384], action_vla_{t+1} [7],        │
│           proprio_{t+1} [8]) → Δa [7]                         │
│          prediction_error가 이미 intent 정보를 내포하므로        │
│                                                              │
│  Step 5: 실행 = action_vla_{t+1} + Δa                         │
│                                                              │
│  ※ 1-step delay: t에서의 error가 t+1의 correction을 만듦       │
│    → 소뇌도 동일: 결과를 관측해야 error를 계산할 수 있음          │
│    → chunk 첫 step은 correction 없이 실행 (이전 chunk error 사용 가능) │
└──────────────────────────────────────────────────────────────┘
```

### Module 1: Intent Extractor

```python
"""
GR00T forward pass 시 VLM hidden state를 hook으로 추출하는 모듈.

별도 학습 없음 — GR00T 내부의 intermediate representation을 가져오기만 함.
매 action chunk마다 1회 실행 (GR00T forward pass와 동시에).

** 확인 완료 **:
D_intent = 2048 (GR00T N1.6 backbone = Eagle3, VLM hidden dim = 2048)
Language model = Qwen3-1.7B (28 layers, truncated to 16)
Vision = SigLIP2 (1152-dim, patch_size=14)
VLM output = (B, 108, 2048): 81 image tokens + 27 text tokens

Hook 위치: model.backbone (EagleBackbone)
→ output: (backbone_features [B,108,2048], image_mask [B,108], attention_mask [B,108])
→ attention_mask로 active token만 mean pooling → intent_vec [2048]
"""

class IntentExtractor:
    """
    역할: GR00T VLM의 backbone output을 hook으로 추출하여 intent vector 생성
    입력: GR00T 모델 (Gr00tN1d6)
    출력: intent_vec [2048]
    학습: 없음 (frozen GR00T에서 추출만)
    갱신 주기: action chunk마다 1회 (GR00T forward pass와 동시에)

    GR00T 내부 forward flow:
    1. image → SigLIP2 → (B, 324, 1152)
    2. SigLIP features → Eagle MLP (downsample 0.5) → (B, 81, 2048)
    3. text → Qwen3 tokenizer → Qwen3 LM (16 layers) → (B, 27, 2048)
    4. concat → backbone_features: (B, 108, 2048)
    우리는 4번의 output을 hook으로 가져옴.
    """

    def __init__(self, groot_model):
        """
        Args:
            groot_model: 로드된 Gr00tN1d6 모델
        """
        self.groot_model = groot_model
        self._hook_data = {}

        # Hook 위치: model.backbone (EagleBackbone)
        # backbone의 forward()는 (backbone_features, image_mask, attention_mask) 반환
        groot_model.backbone.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """
        EagleBackbone.forward() output:
          backbone_features: (B, 108, 2048)
          image_mask: (B, 108) — True for image tokens
          attention_mask: (B, 108) — True for active (non-padding) tokens
        """
        backbone_features, image_mask, attention_mask = output
        self._hook_data['features'] = backbone_features.detach()
        self._hook_data['image_mask'] = image_mask.detach()
        self._hook_data['attention_mask'] = attention_mask.detach()

    def get_intent(self, pooling: str = "all_active") -> torch.Tensor:
        """
        GR00T forward pass 후 호출. Hook에 저장된 backbone features를 pooling.

        Args:
            pooling: pooling 방식
                - "all_active": attention_mask가 True인 모든 token의 mean (권장)
                  → image context + language intent 모두 포함
                - "image_only": image token만 mean → visual scene 정보
                - "text_only": text token만 mean → pure language intent

        Returns:
            intent_vec: (B, 2048)
        """
        features = self._hook_data['features']       # (B, 108, 2048)
        image_mask = self._hook_data['image_mask']    # (B, 108)
        attn_mask = self._hook_data['attention_mask'] # (B, 108)

        if pooling == "all_active":
            # attention_mask로 active token만 mean pooling
            mask = attn_mask.unsqueeze(-1).float()  # (B, 108, 1)
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pooling == "image_only":
            mask = image_mask.unsqueeze(-1).float()
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pooling == "text_only":
            text_mask = (~image_mask & attn_mask).unsqueeze(-1).float()
            pooled = (features * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        return pooled  # (B, 2048)
```

### Module 2: Visual Encoder (DINOv2 ViT-S + LoRA + EMA)

```python
"""
DINOv2 ViT-S를 사용한 visual feature extractor.

이전 실험에서의 교훈:
1. CLS token보다 patch token mean이 dynamics prediction에 훨씬 나음
   (CLS: 13.8% 개선 vs Patch mean: 16.8% 개선)
   → DINO-WM, VAT 논문에서도 patch token이 spatial information을 더 잘 보존함을 확인
2. Projection layer (384→256) 없이 DINOv2의 native 384-dim 출력을 그대로 사용
   → 이전에 projection이 학습 안 되는 문제 있었음 (randomly initialized, frozen)
3. EMA target encoder가 temporal collapse 없이 가장 좋은 결과를 보임
   (Jointly VICReg만으로는 temporal collapse 발생)
"""

class CerebellumVisualEncoder(nn.Module):
    """
    역할: 이미지에서 384-dim visual feature 추출
    구조: DINOv2 ViT-S (22M params) + LoRA adapters

    Online encoder: 학습됨 (Phase 1에서 forward model과 jointly)
    EMA target encoder: no gradient, τ=0.996으로 online에서 EMA 업데이트

    입력: image [98×98] (14의 배수, 96에 가장 가까운 값)
    출력: z [384] (49 patch tokens의 mean pooling)
    """

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        use_lora: bool = True,          # EMA 방식에서는 LoRA 사용
        lora_rank: int = 8,
        input_size: int = 98,           # 14 * 7 = 98
    ):
        super().__init__()
        self.feature_dim = 384          # DINOv2 ViT-S native dim, 변경 불가
        self.input_size = input_size

        # Online encoder (trainable with LoRA)
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone)

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=["qkv"],
                lora_dropout=0.1,
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)

        # Projection 없음! DINOv2 384-dim 그대로 사용.

        # Image preprocessing
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) float32, [0, 1] 범위
        Returns:
            z: (B, 384) — patch token mean pooling
        """
        x = self.transform(image)

        # Patch tokens 추출 (CLS 제외)
        # DINOv2 forward_features → dict with 'x_norm_patchtokens'
        features = self.backbone.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']   # (B, 49, 384) for 98×98 input

        # Mean pooling over spatial patches
        z = patch_tokens.mean(dim=1)  # (B, 384)

        return z


class EMAEncoder:
    """
    EMA Target Encoder — gradient를 받지 않는 online encoder의 slow copy.

    이전 실험 결과:
    - Jointly DINOv2+LoRA (VICReg λ=5): temporal collapse 발생 (Δz norm 49x 감소)
    - EMA (τ=0.996): collapse 없이 안정적 학습 (dz≈0.88, z_std≈1.07)

    BYOL/I-JEPA 스타일:
    - Online encoder (trainable) → z_t 생성 (forward model 입력)
    - Target encoder (EMA, no grad) → z_t+1_target 생성 (forward model supervision)
    - 이렇게 하면 encoder가 Δz를 0으로 만들어 prediction을 trivial하게 만드는 temporal collapse를 방지
    """

    def __init__(self, online_encoder: CerebellumVisualEncoder, tau: float = 0.996):
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.tau = tau

    @torch.no_grad()
    def update(self, online_encoder: CerebellumVisualEncoder):
        """매 training step마다 호출"""
        for p_online, p_target in zip(online_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data = self.tau * p_target.data + (1 - self.tau) * p_online.data

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """z_t+1_target 생성용"""
        return self.target_encoder(image)
```

### Module 3: Forward Model (Intent-Conditioned State Predictor)

```python
"""
*** 이 모듈이 이전과 가장 크게 달라진 핵심 ***

이전: (z_t, action, proprio) → Δz_predicted  (action-conditioned dynamics)
지금: (z_t, proprio, intent)  → Δz_ideal     (intent-conditioned goal prediction)

Action을 받지 않는다!

왜 action을 제거하는가:
- Action-conditioned forward model은 VLA가 어떤 action을 내든 정확히 예측 → error ≈ 0 항상
- Intent-conditioned면 "이 goal을 향해 가면 다음은 이래야 해" 예측
- VLA가 잘못된 action → 실제 z_t+1이 intent-based prediction과 다름 → 의미 있는 error

왜 intent가 Forward model에 들어가고 Correction network에는 안 들어가는가:
- 실제 소뇌에서도 mossy fibers (intent + motor command)는 forward model (Purkinje cells)에 들어감
- Climbing fibers (error signal)는 이미 goal-relative error를 담고 있음
- Correction network는 "이 error가 있으면 이만큼 수정해"만 하면 됨
- prediction_error 자체가 이미 intent 정보를 내포 (forward model이 intent 기반으로 예측했으므로)
"""

class IntentForwardModel(nn.Module):
    """
    Intent-conditioned 1-step forward prediction.

    "이 intent를 달성하려는 과정에서, 현재 상태가 이렇다면,
     이상적으로 다음 상태 변화는 이래야 한다"를 예측.

    입력: concat(z_t [384], proprio_t [8], intent_vec [2048])
          → total: 384 + 8 + 2048 = 2440
    출력: Δz_ideal [384]

    학습: Expert trajectory에서 (z_t, proprio_t, intent) → (z_t+1 - z_t) 학습
          Expert의 다음 상태 = intent에 맞는 이상적 다음 상태 (expert이니까)

    ※ intent_dim=2048은 확인 완료 (GR00T backbone = Eagle3, Qwen3-1.7B hidden_size)
    ※ proprio_dim=8 (WidowX: x, y, z, roll, pitch, yaw, pad, gripper)
    """

    def __init__(
        self,
        feature_dim: int = 384,         # DINOv2 native dim
        proprio_dim: int = 8,           # WidowX: 8-dim (x,y,z,rpy,pad,gripper)
        intent_dim: int = 2048,         # GR00T backbone hidden dim (확인 완료)
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()

        input_dim = feature_dim + proprio_dim + intent_dim  # 384 + 8 + 2048 = 2440

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, feature_dim))  # output: 384

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,              # (B, 384) — current visual feature
        proprio: torch.Tensor,         # (B, 8) — current proprioception
        intent: torch.Tensor,          # (B, 2048) — VLM backbone hidden state
    ) -> torch.Tensor:
        """
        Returns:
            delta_z_ideal: (B, 384) — "이 intent를 향해 가면 이만큼 변해야 해"
        """
        x = torch.cat([z, proprio, intent], dim=-1)
        delta_z_ideal = self.net(x)
        return delta_z_ideal


class ProprioForwardModel(nn.Module):
    """
    Proprioception forward model — intent-conditioned.

    이전과 달리 action 대신 intent를 받음.
    "이 intent를 향해 가면 proprio가 이렇게 변해야 해"

    입력: concat(proprio_t [8], intent_vec [2048]) → total: 2056
    출력: Δproprio_predicted [8]

    ※ proprio_dim=8: x, y, z, roll, pitch, yaw, pad, gripper
    """

    def __init__(
        self,
        proprio_dim: int = 8,           # WidowX: 8-dim
        intent_dim: int = 2048,         # GR00T backbone hidden dim
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(proprio_dim + intent_dim, hidden_dim),  # 2056 → 128
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proprio_dim),               # 128 → 8
        )

    def forward(self, proprio: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio: (B, 8)
            intent: (B, 2048)
        Returns:
            delta_proprio: (B, 8)
        """
        x = torch.cat([proprio, intent], dim=-1)
        return self.net(x)
```

### Module 4: Correction Network

```python
"""
Prediction error 기반 action correction.

입력: (prediction_error, action_vla, proprio) — intent 받지 않음!
출력: Δa [7]

왜 intent를 안 받는가:
- prediction_error = (z_t+1_actual - z_t) - Δz_ideal
- Δz_ideal은 forward model이 intent를 보고 예측한 것
- 따라서 prediction_error 자체가 이미 "intent 대비 얼마나 벗어났는지"를 담고 있음
- Correction network는 그냥 "이 error가 있고, 이 action이 계획되어 있으니 이만큼 수정해"만 하면 됨
- 입력 차원을 줄여서 학습 안정성 확보

이전 실패와의 차이:
- 이전: perturbation-based (random noise correction 학습) → constant bias만 출력
- 지금: VLA의 실제 action + intent-aware prediction error → 상황별 다른 correction 가능
"""

class CorrectionNetwork(nn.Module):
    """
    입력: concat(prediction_error [384], action_vla [7], proprio [8])
          → total: 384 + 7 + 8 = 399
    출력: Δa [7], tanh × max_correction

    ※ action_dim=7 (correction output), proprio_dim=8 (input state)
    """

    def __init__(
        self,
        feature_dim: int = 384,         # prediction error dim = visual feature dim
        action_dim: int = 7,            # correction output dim
        proprio_dim: int = 8,           # WidowX: 8-dim state
        hidden_dim: int = 256,
        num_layers: int = 3,
        max_correction: float = 0.1,
    ):
        super().__init__()

        input_dim = feature_dim + action_dim + proprio_dim  # 384 + 7 + 8 = 399

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.net = nn.Sequential(*layers)
        self.max_correction = max_correction

    def forward(
        self,
        prediction_error: torch.Tensor,    # (B, 384) — intent-aware visual prediction error
        action_vla: torch.Tensor,           # (B, 7) — VLA's planned action
        proprio: torch.Tensor,              # (B, 8) — current proprioception
    ) -> torch.Tensor:
        """
        Returns:
            delta_a: (B, 7) — bounded action correction
        """
        x = torch.cat([prediction_error, action_vla, proprio], dim=-1)
        raw = self.net(x)
        delta_a = torch.tanh(raw) * self.max_correction
        return delta_a
```

### Module 5: 통합 Cerebellum Module

```python
"""
전체 모듈 통합 — Inference 시 사용.

Inference flow (매 control step):
1. GR00T forward pass → action_chunk + intent_vec (chunk 시작 시 1회)
2. DINOv2 → z_t
3. Forward model: (z_t, proprio, intent) → Δz_ideal
4. 이전 step의 prediction error 계산 (1-step delay)
5. Correction network: (prev_error, action_vla, proprio) → Δa
6. 실행: action_vla + Δa
"""

class IntentCerebellumModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Visual encoder (online + EMA)
        self.visual_encoder = CerebellumVisualEncoder(
            backbone=config.visual_backbone,
            use_lora=config.use_visual_lora,
            lora_rank=config.visual_lora_rank,
            input_size=config.image_size,
        )
        self.ema_encoder = EMAEncoder(self.visual_encoder, tau=config.ema_tau)

        # Intent extractor (GR00T hook, 학습 없음)
        # groot_model은 외부에서 주입
        self.intent_extractor = None  # set_groot_model()에서 초기화

        # Forward model (intent-conditioned, action 없음!)
        self.forward_model = IntentForwardModel(
            feature_dim=384,
            proprio_dim=8,              # WidowX state dim
            intent_dim=2048,            # GR00T backbone hidden dim
            hidden_dim=config.forward_hidden_dim,
            num_layers=config.forward_num_layers,
        )

        # Proprio forward model
        self.proprio_forward = ProprioForwardModel(
            proprio_dim=8,
            intent_dim=2048,
            hidden_dim=128,
        )

        # Correction network (intent 안 받음)
        self.correction_net = CorrectionNetwork(
            feature_dim=384,
            action_dim=7,               # WidowX action dim
            proprio_dim=8,              # WidowX state dim
            hidden_dim=config.correction_hidden_dim,
            num_layers=config.correction_num_layers,
            max_correction=config.max_correction,
        )

        # State tracking
        self.prev_z = None
        self.prev_proprio = None
        self.prev_prediction_error = None
        self.current_intent = None

    def set_groot_model(self, groot_model):
        """GR00T 모델 연결 및 intent extractor 초기화"""
        self.intent_extractor = IntentExtractor(groot_model)

    def on_new_chunk(self, intent_vec: torch.Tensor):
        """
        새 action chunk 시작 시 호출.
        GR00T forward pass 후 intent_extractor.get_intent()로 뽑은 intent를 저장.

        prev_z, prev_proprio는 reset하지 않음!
        이전 chunk의 마지막 error를 새 chunk 첫 step에 사용할 수 있도록.
        """
        self.current_intent = intent_vec.detach()

    def reset(self):
        """에피소드 시작 시 완전 초기화"""
        self.prev_z = None
        self.prev_proprio = None
        self.prev_prediction_error = None
        self.current_intent = None

    @torch.no_grad()
    def correct(
        self,
        image_current: torch.Tensor,        # (1, 3, H, W)
        proprio_current: torch.Tensor,       # (1, 8)
        action_planned: torch.Tensor,        # (1, 7)
    ) -> torch.Tensor:
        """
        매 control step마다 호출.

        Returns:
            action_corrected: (1, 7)
        """
        # 1. Visual feature 추출
        z_current = self.visual_encoder(image_current)  # (1, 384)

        # 2. Prediction error 계산 (1-step delay)
        if self.prev_z is not None and self.current_intent is not None:
            # 이전 step에서의 forward model 예측
            delta_z_ideal = self.forward_model(
                self.prev_z, self.prev_proprio, self.current_intent
            )
            delta_z_actual = z_current - self.prev_z
            prediction_error = delta_z_actual - delta_z_ideal
        else:
            # 에피소드 첫 step → error 없음 → correction 없음
            prediction_error = torch.zeros(1, 384, device=image_current.device)

        # 3. Correction
        delta_a = self.correction_net(
            prediction_error=prediction_error,
            action_vla=action_planned,
            proprio=proprio_current,
        )

        action_corrected = action_planned + delta_a

        # 4. State 업데이트
        self.prev_z = z_current.detach()
        self.prev_proprio = proprio_current.detach()
        self.prev_prediction_error = prediction_error.detach()

        return action_corrected
```

---

## Training Pipeline

### Phase 1: Forward Model + Visual Encoder (EMA, Intent-Conditioned)

```python
"""
Phase 1: Intent-conditioned forward model 학습

핵심 변화 (이전 대비):
1. Forward model이 action 대신 intent를 받음
2. Intent는 GR00T forward pass에서 VLM hidden state를 뽑아서 사용
3. Intent 추출은 오프라인으로 미리 해놓을 수 있음 (속도를 위해)
4. Visual encoder는 EMA 방식으로 학습 (temporal collapse 방지)

학습 데이터: BridgeData expert trajectories
각 sample: (image_t, image_t+1, proprio_t, proprio_t+1, intent_vec, language_instruction)

Loss:
  L_visual = MSE(Δz_ideal, Δz_target)
  L_proprio = MSE(Δproprio_pred, Δproprio_actual)
  L_vicreg = VICReg_variance(z_t)  — feature collapse 방지
  L_total = L_visual + L_proprio + λ * L_vicreg

EMA:
  z_t = online_encoder(img_t)           ← gradient 흐름
  z_t+1_target = ema_encoder(img_t+1)   ← no gradient
  Δz_target = z_t+1_target - z_t
"""

# ===== 사전 단계: Intent 오프라인 추출 =====

def extract_intents_offline(groot_model, dataset_path, output_path, hook_layer_name):
    """
    BridgeData의 모든 에피소드에서 GR00T forward pass를 돌려
    intent vector를 미리 추출하여 저장.

    각 에피소드의 각 action chunk 시작점마다 intent를 저장.
    (chunk 내 step들은 같은 intent 공유)

    저장 형식:
    {
        'episode_0': {
            'intents': tensor (T, D_intent),  # T = 에피소드 길이
            'language': "pick up the green cube..."
        },
        ...
    }

    ※ 이 과정은 시간이 걸림 (GR00T forward pass 필요)
    ※ 하지만 1회만 하면 됨
    """
    intent_extractor = IntentExtractor(groot_model, hook_layer_name)

    # BridgeData 로드
    dataset = load_bridge_dataset(dataset_path)

    all_intents = {}
    for episode_idx, episode in enumerate(dataset):
        intents = []
        for t in range(len(episode)):
            obs = episode.get_observation(t)
            with torch.no_grad():
                # GR00T forward pass (action은 안 씀, intent만 추출)
                _ = groot_model.get_action(obs)
                intent_vec = intent_extractor.get_intent()
                intents.append(intent_vec.cpu())

        all_intents[f'episode_{episode_idx}'] = {
            'intents': torch.stack(intents),
            'language': episode.language_instruction,
        }

    torch.save(all_intents, output_path)


# ===== Phase 1 Training Loop =====

def train_phase1(config):
    """
    Phase 1: Forward Model + Visual Encoder 학습 (EMA 방식)
    """

    # Dataset: (image_t, image_t+1, proprio_t, proprio_t+1, intent_vec)
    # intent_vec는 오프라인 추출된 것 사용
    dataset = IntentForwardModelDataset(
        dataset_path=config.dataset_path,         # /mnt/md1/solee/data/bridge_lerobot
        intent_path=config.intent_path,            # data/bridge_intents.pt (오프라인 추출)
        image_size=config.image_size,              # 98
    )
    dataloader = DataLoader(dataset, batch_size=config.phase1.batch_size, shuffle=True)

    # Models
    online_encoder = CerebellumVisualEncoder(
        use_lora=True, lora_rank=config.visual_lora_rank
    ).cuda()
    ema_encoder = EMAEncoder(online_encoder, tau=config.ema_tau)

    forward_model = IntentForwardModel(
        feature_dim=384,
        proprio_dim=8,                     # WidowX state
        intent_dim=2048,                   # GR00T backbone hidden
        hidden_dim=config.forward_hidden_dim,
    ).cuda()

    proprio_forward = ProprioForwardModel(
        proprio_dim=8,
        intent_dim=2048,
    ).cuda()

    # Optimizer — online encoder LoRA + forward model + proprio forward
    optimizer = torch.optim.AdamW([
        {'params': online_encoder.parameters(), 'lr': config.phase1.encoder_lr},    # LoRA는 작은 lr
        {'params': forward_model.parameters(), 'lr': config.phase1.learning_rate},
        {'params': proprio_forward.parameters(), 'lr': config.phase1.learning_rate},
    ], weight_decay=config.phase1.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.phase1.num_epochs
    )

    # VICReg variance loss
    def vicreg_variance_loss(z, target_std=1.0):
        """Feature dimension별 std가 target_std 이상이 되도록 강제"""
        std = z.std(dim=0)  # (384,)
        return torch.mean(torch.relu(target_std - std))

    # Training loop
    for epoch in range(config.phase1.num_epochs):
        for batch in dataloader:
            img_t = batch['image_t'].cuda()
            img_t1 = batch['image_t1'].cuda()
            proprio_t = batch['proprio_t'].cuda()
            proprio_t1 = batch['proprio_t1'].cuda()
            intent = batch['intent'].cuda()         # (B, D_intent)

            # Online encoder → z_t (gradient 흐름)
            z_t = online_encoder(img_t)              # (B, 384)

            # EMA encoder → z_t+1_target (no gradient)
            with torch.no_grad():
                z_t1_target = ema_encoder.encode(img_t1)  # (B, 384)

            # Forward model prediction
            delta_z_target = z_t1_target - z_t.detach()   # target은 stop_grad(z_t) 사용
            delta_z_ideal = forward_model(z_t, proprio_t, intent)

            # Proprio forward model
            delta_proprio_target = proprio_t1 - proprio_t
            delta_proprio_pred = proprio_forward(proprio_t, intent)

            # Losses
            visual_loss = F.mse_loss(delta_z_ideal, delta_z_target)
            proprio_loss = F.mse_loss(delta_proprio_pred, delta_proprio_target)
            vicreg_loss = vicreg_variance_loss(z_t)

            loss = visual_loss + proprio_loss + config.vicreg_lambda * vicreg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            ema_encoder.update(online_encoder)

        scheduler.step()

        # Logging
        # 반드시 로깅할 것:
        # - visual_loss, proprio_loss, vicreg_loss
        # - z_t.std(dim=0).mean() → z_std (1.0 근처여야 함, collapse 감지)
        # - delta_z_target.norm(dim=-1).mean() → dz_norm (0이면 temporal collapse)
        # - delta_z_ideal.norm(dim=-1).mean() → pred_dz_norm

        # Δz=0 baseline과 비교
        baseline_loss = F.mse_loss(
            torch.zeros_like(delta_z_target), delta_z_target
        )
        improvement = (baseline_loss - visual_loss) / baseline_loss * 100

        print(f"Epoch {epoch}: vis={visual_loss:.6f}, proprio={proprio_loss:.6f}, "
              f"vicreg={vicreg_loss:.4f}, z_std={z_t.std(dim=0).mean():.3f}, "
              f"dz_norm={delta_z_target.norm(dim=-1).mean():.3f}, "
              f"baseline={baseline_loss:.6f}, improvement={improvement:.1f}%")

    # Save
    torch.save({
        'online_encoder': online_encoder.state_dict(),
        'ema_encoder': ema_encoder.target_encoder.state_dict(),
        'forward_model': forward_model.state_dict(),
        'proprio_forward': proprio_forward.state_dict(),
    }, f"{config.checkpoint_dir}/phase1_best.pt")
```

### Phase 2: Correction Network 학습 (VLA-based)

```python
"""
Phase 2: Correction network 학습

이전과의 핵심 차이:
1. Perturbation (random noise) 대신 **VLA의 실제 action**을 사용
2. Forward model이 intent-conditioned이므로, prediction error가 "goal 대비 얼마나 벗어났는지"를 담음
3. correction_target = action_expert - action_vla (VLA가 expert와 다른 만큼 보정)

사전 요구사항:
- Phase 1에서 학습된 forward model + visual encoder (frozen)
- 오프라인 추출된 intent vectors
- 오프라인 생성된 VLA actions (BridgeData에서 GR00T inference)

학습 데이터 구성:
매 timestep마다:
  - image_t, image_t+1 (BridgeData expert trajectory)
  - action_expert (BridgeData에서)
  - action_vla (GR00T inference 결과)
  - proprio_t, proprio_t+1
  - intent_vec

Prediction error 계산:
  z_t = online_encoder(img_t)
  Δz_ideal = forward_model(z_t, proprio_t, intent)   ← intent 기반 예측
  Δz_actual = online_encoder(img_t+1) - z_t           ← 실제 expert trajectory의 변화
  prediction_error = Δz_actual - Δz_ideal

  ※ 여기서 Δz_actual은 expert trajectory 기반이므로 Δz_ideal과 비슷할 것.
  ※ 하지만 prediction error가 0이 아닌 이유: forward model이 완벽하지 않으므로.

  ** 중요 **: Phase 2에서 prediction error의 역할은 "학습 시 correction network에게
  prediction error라는 입력 형태를 학습시키는 것". Inference 시에는 VLA action에 의한
  실제 observation이 들어오므로 prediction error가 커질 수 있고, 그때 correction이 작동.

Correction target:
  correction_target = action_expert - action_vla
  → VLA가 expert와 다른 만큼을 보정해야 함
"""

# ===== 사전 단계: VLA Action 오프라인 생성 =====

def generate_vla_actions(groot_model, dataset_path, output_path):
    """
    BridgeData의 각 에피소드에서 GR00T inference를 돌려
    VLA가 실제로 어떤 action을 내는지 저장.

    각 timestep마다:
    - observation (image + language + proprio) 입력
    - GR00T가 action chunk (8 steps) 출력
    - 첫 번째 action만 저장 (또는 전체 chunk 저장)

    저장 형식:
    {
        'episode_0': {
            'actions_vla': tensor (T, 7),      # VLA가 낸 action
            'actions_expert': tensor (T, 7),   # BridgeData expert action
            'proprios': tensor (T, 8),
        },
        ...
    }
    """
    # GR00T로 BridgeData 전체에 대해 inference
    # 각 timestep에서 현재 observation을 넣고 action chunk의 첫 action을 저장
    pass  # 구현 필요


# ===== Phase 2 Training Loop =====

def train_phase2(config):
    """
    Phase 2: Correction network 학습 (VLA-based, intent-aware)
    """

    # 학습된 Phase 1 모델 로드 (frozen)
    checkpoint = torch.load(f"{config.checkpoint_dir}/phase1_best.pt")

    online_encoder = CerebellumVisualEncoder(use_lora=True).cuda()
    online_encoder.load_state_dict(checkpoint['online_encoder'])
    online_encoder.eval()
    for p in online_encoder.parameters():
        p.requires_grad = False

    forward_model = IntentForwardModel(
        feature_dim=384, proprio_dim=8, intent_dim=2048
    ).cuda()
    forward_model.load_state_dict(checkpoint['forward_model'])
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad = False

    # Correction network (학습 대상)
    correction_net = CorrectionNetwork(
        feature_dim=384, action_dim=7, proprio_dim=8,
        hidden_dim=config.correction_hidden_dim,
        max_correction=config.max_correction,
    ).cuda()

    # Dataset
    dataset = VLACorrectionDataset(
        dataset_path=config.dataset_path,
        vla_actions_path=config.vla_actions_path,    # data/vla_actions.pt
        intent_path=config.intent_path,               # data/bridge_intents.pt
        image_size=config.image_size,
    )
    dataloader = DataLoader(dataset, batch_size=config.phase2.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        correction_net.parameters(),
        lr=config.phase2.learning_rate,
        weight_decay=config.phase2.weight_decay,
    )

    for epoch in range(config.phase2.num_epochs):
        for batch in dataloader:
            img_t = batch['image_t'].cuda()
            img_t1 = batch['image_t1'].cuda()
            action_expert = batch['action_expert'].cuda()
            action_vla = batch['action_vla'].cuda()
            proprio_t = batch['proprio_t'].cuda()
            intent = batch['intent'].cuda()

            with torch.no_grad():
                # Visual features
                z_t = online_encoder(img_t)
                z_t1 = online_encoder(img_t1)

                # Intent-conditioned prediction
                delta_z_ideal = forward_model(z_t, proprio_t, intent)
                delta_z_actual = z_t1 - z_t
                prediction_error = delta_z_actual - delta_z_ideal

            # Correction target
            correction_target = action_expert - action_vla  # (B, 7)

            # Correction prediction
            delta_a = correction_net(prediction_error, action_vla, proprio_t)

            # Loss
            loss = F.mse_loss(delta_a, correction_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Logging
        # - correction_loss
        # - correction magnitude: delta_a.norm(dim=-1).mean()
        # - correction target magnitude: correction_target.norm(dim=-1).mean()
        # - per-dimension correction 분포 (constant bias인지 확인!)
        #   → delta_a.std(dim=0)이 작으면 constant bias → 문제!
        #   → delta_a.std(dim=0)이 크면 상황별 다른 correction → 원하는 결과

        correction_std = delta_a.std(dim=0).mean()
        target_std = correction_target.std(dim=0).mean()

        print(f"Epoch {epoch}: loss={loss:.6f}, "
              f"correction_std={correction_std:.4f}, target_std={target_std:.4f}")

    torch.save(correction_net.state_dict(), f"{config.checkpoint_dir}/phase2_correction.pt")
```

---

## Config

```yaml
# cerebellar_correction/configs/cerebellum_config.yaml

# ===== Visual Encoder =====
visual_backbone: "dinov2_vits14"
image_size: 98                        # 14 * 7, DINOv2 patch size의 배수
use_visual_lora: true
visual_lora_rank: 8
ema_tau: 0.996

# ===== Intent (확인 완료) =====
intent_dim: 2048                      # GR00T backbone hidden dim (Qwen3-1.7B)
intent_pooling: "all_active"          # attention_mask 기반 mean pooling (image+text)

# ===== Forward Models =====
forward_hidden_dim: 256
forward_num_layers: 3
vicreg_lambda: 5.0                    # 이전 EMA 실험에서 λ=5.0이 안정적

# ===== Correction Network =====
correction_hidden_dim: 256
correction_num_layers: 3
max_correction: 0.1

# ===== Robot (WidowX / BridgeData) =====
action_dim: 7                         # x, y, z, roll, pitch, yaw, gripper
proprio_dim: 8                        # x, y, z, roll, pitch, yaw, pad, gripper
control_frequency: 5                  # Hz

# ===== Paths =====
dataset_path: "/mnt/md1/solee/data/bridge_lerobot"
groot_model_path: "/mnt/md1/solee/checkpoints/GR00T-N1.6-bridge"
checkpoint_dir: "checkpoints/cerebellum_v2"
intent_path: "data/bridge_intents.pt"
vla_actions_path: "data/vla_actions.pt"

# ===== Phase 1: Forward Model + Encoder =====
phase1:
  batch_size: 256
  learning_rate: 1.0e-3               # forward model
  encoder_lr: 1.0e-4                  # LoRA (더 작게)
  weight_decay: 1.0e-4
  num_epochs: 50
  optimizer: "adamw"
  scheduler: "cosine"

# ===== Phase 2: Correction Network =====
phase2:
  batch_size: 256
  learning_rate: 5.0e-4
  weight_decay: 1.0e-4
  num_epochs: 50

# ===== Evaluation =====
eval:
  env_name: "simpler_env_widowx/widowx_stack_cube"
  num_episodes: 50
```

---

## Evaluation

```python
"""
GR00T + Intent-Conditioned Cerebellum 통합 eval server.

Inference flow:
1. SimplerEnv에서 observation 수신
2. GR00T forward pass → action chunk (8 steps) + intent vector (hook)
3. 매 step마다:
   a. 현재 image로 DINOv2 → z_t
   b. Forward model: (z_t, proprio, intent) → Δz_ideal
   c. 이전 step의 prediction error 계산
   d. Correction network: (error, action_vla, proprio) → Δa
   e. action_corrected = action_vla + Δa 실행
4. 8 step 완료 후 새로운 observation으로 GR00T 재계획
"""

class IntentCerebellumGR00TServer:
    def __init__(self, config):
        # GR00T (완전 frozen)
        self.groot = load_groot_policy(config.groot_model_path)
        for p in self.groot.parameters():
            p.requires_grad = False
        self.groot.eval()

        # Cerebellum
        self.cerebellum = IntentCerebellumModule(config)
        self.cerebellum.load_checkpoint(config.checkpoint_dir)
        self.cerebellum.set_groot_model(self.groot)
        self.cerebellum.eval()

    def run_episode(self, env):
        obs = env.reset()
        self.cerebellum.reset()

        done = False
        total_steps = 0

        while not done and total_steps < env.max_steps:
            # GR00T forward pass → action chunk + intent
            action_chunk = self.groot.get_action(obs)
            intent_vec = self.cerebellum.intent_extractor.get_intent()
            self.cerebellum.on_new_chunk(intent_vec)

            # Action chunk 실행 (매 step correction 적용)
            chunk_length = len(action_chunk)  # 보통 8
            for step_idx in range(chunk_length):
                if done:
                    break

                action_planned = action_chunk[step_idx]
                image_current = obs['image']
                proprio_current = obs['proprio']

                # Correction 적용
                action_corrected = self.cerebellum.correct(
                    image_current=preprocess_image(image_current),
                    proprio_current=proprio_current,
                    action_planned=action_planned,
                )

                # 환경에 corrected action 실행
                obs, reward, done, info = env.step(action_corrected)
                total_steps += 1

        return info.get('success', False)
```

---

## 프로젝트 디렉토리 구조

```
cerebellar_correction/
├── models/
│   ├── __init__.py
│   ├── visual_encoder.py          # CerebellumVisualEncoder + EMAEncoder
│   ├── intent_extractor.py        # IntentExtractor (GR00T hook)
│   ├── forward_model.py           # IntentForwardModel + ProprioForwardModel
│   ├── correction_net.py          # CorrectionNetwork
│   └── cerebellum.py              # IntentCerebellumModule (통합)
├── data/
│   ├── __init__.py
│   ├── extract_intents.py         # GR00T에서 intent vector 오프라인 추출
│   ├── generate_vla_actions.py    # GR00T에서 VLA action 오프라인 생성
│   ├── dataset_phase1.py          # Phase 1 dataset (image pairs + intent)
│   └── dataset_phase2.py          # Phase 2 dataset (image pairs + intent + VLA actions)
├── training/
│   ├── __init__.py
│   ├── train_phase1.py            # Forward model + encoder (EMA)
│   └── train_phase2.py            # Correction network (VLA-based)
├── eval/
│   ├── __init__.py
│   ├── run_cerebellum_server.py   # GR00T + Cerebellum 통합 eval server
│   └── rollout_cerebellum.py      # SimplerEnv rollout
├── configs/
│   └── cerebellum_config.yaml     # 전체 config
├── scripts/
│   ├── extract_intents.sh         # Intent 추출 실행
│   ├── generate_vla_actions.sh    # VLA action 생성 실행
│   ├── train_phase1.sh            # Phase 1 학습
│   ├── train_phase2.sh            # Phase 2 학습
│   └── eval.sh                    # Evaluation
└── ARCHITECTURE_NOTES.md          # 02_verify에서 확인한 아키텍처 정보
```

---

## 실행 순서 요약

```
0. [완료] D_intent = 2048 확인 (Qwen3-1.7B hidden_size)
   Hook: model.backbone, pooling: attention_mask mean

1. Intent 오프라인 추출
   → scripts/extract_intents.sh
   → data/bridge_intents.pt 생성

2. VLA action 오프라인 생성
   → scripts/generate_vla_actions.sh
   → data/vla_actions.pt 생성

3. Phase 1 학습 (forward model + encoder)
   → scripts/train_phase1.sh
   → checkpoints/cerebellum_v2/phase1_best.pt

4. Phase 2 학습 (correction network)
   → scripts/train_phase2.sh
   → checkpoints/cerebellum_v2/phase2_correction.pt

5. SimplerEnv 평가
   → scripts/eval.sh
```

---

## 주의사항

1. **D_intent = 2048 (확인 완료)**: Hook 위치는 `model.backbone` (EagleBackbone). Pooling은 attention_mask 기반 mean.
2. **GR00T는 완전 freeze**: 어떤 parameter도 수정하지 않음. Hook으로 hidden state만 추출.
3. **GR00T 코드 수정 금지**: `gr00t/` 디렉토리 하위 코드는 건드리지 않음.
4. **Feature dim = 384**: DINOv2 ViT-S의 native dim. Projection 없음.
5. **EMA τ=0.996**: 이전 실험에서 검증된 값.
6. **VICReg λ=5.0**: 이전 실험에서 λ=5.0이 collapse 없이 안정적이었음.
7. **체크포인트 경로**: 이전과 구분하기 위해 `checkpoints/cerebellum_v2/` 사용.
8. **wandb 로깅 필수**: 모든 training에서 z_std, dz_norm, correction_std를 로깅하여 collapse/constant bias 감지.