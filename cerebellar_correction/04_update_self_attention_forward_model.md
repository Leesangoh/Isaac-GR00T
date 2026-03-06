# 04. 업데이트: Self-Attention Forward Model + Full Intent Token Sequence

---

## 변경 이유

03.md의 IntentForwardModel은 MLP 기반이었다:
- `concat(z_t [384], proprio [8], intent_vec [2048]) = 2440 → hidden 256`
- **문제**: 2440→256은 9.5x 압축. 2048-dim intent 정보가 대부분 유실됨.
- MLP는 모든 token을 하나의 vector로 뭉쳐서 받으므로, 108개 intent token 간의 관계(image-text interaction)가 이미 소실된 상태.

**해결**: I-JEPA 스타일 self-attention predictor로 교체.
- z_t, proprio, intent tokens를 **개별 토큰**으로 유지
- Self-attention으로 서로 attend → intent가 현재 state에 맞게 contextualize
- 110 tokens × 384-dim × 2 layers → **latency < 0.5ms** (전체 budget 대비 무시 가능)

---

## 변경 1: IntentExtractor — Full Token Sequence 반환

### 이전 (03.md)
```python
# get_intent() → mean pooled (B, 2048)
def get_intent(self, pooling="all_active") -> torch.Tensor:
    ...
    return pooled  # (B, 2048)
```

### 변경 후
```python
class IntentExtractor:
    """
    변경점:
    - get_intent()는 이제 2가지 모드 지원
    - get_intent_tokens(): full token sequence 반환 (self-attention forward model용)
    - get_intent_pooled(): mean pooled 반환 (ProprioForwardModel 등에서 사용)

    self-attention forward model에서 108개 token을 개별 토큰으로 넣어야 하므로,
    pooling 전의 raw token sequence가 필요.
    """

    def __init__(self, groot_model):
        self.groot_model = groot_model
        self._hook_data = {}
        groot_model.backbone.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        backbone_features, image_mask, attention_mask = output
        self._hook_data['features'] = backbone_features.detach()
        self._hook_data['image_mask'] = image_mask.detach()
        self._hook_data['attention_mask'] = attention_mask.detach()

    def get_intent_tokens(self) -> tuple:
        """
        Forward model (self-attention)용 — full token sequence 반환.

        Returns:
            intent_tokens: (B, 108, 2048) — raw backbone features (padding 포함)
            attention_mask: (B, 108) — True for active (non-padding) tokens
        """
        return (
            self._hook_data['features'],       # (B, 108, 2048)
            self._hook_data['attention_mask'],  # (B, 108)
        )

    def get_intent_pooled(self, pooling: str = "all_active") -> torch.Tensor:
        """
        ProprioForwardModel 등 MLP 기반 모듈용 — pooled vector 반환.
        (기존 get_intent()와 동일)

        Returns:
            intent_vec: (B, 2048)
        """
        features = self._hook_data['features']
        attn_mask = self._hook_data['attention_mask']

        if pooling == "all_active":
            mask = attn_mask.unsqueeze(-1).float()
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pooling == "image_only":
            image_mask = self._hook_data['image_mask']
            mask = image_mask.unsqueeze(-1).float()
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif pooling == "text_only":
            image_mask = self._hook_data['image_mask']
            text_mask = (~image_mask & attn_mask).unsqueeze(-1).float()
            pooled = (features * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        return pooled  # (B, 2048)
```

---

## 변경 2: Intent 오프라인 추출 — 모든 Step에서 Full Token Sequence 저장

### 이전 (03.md)
```python
# intent_vec만 저장 (B, 2048) — pooled
intent_vec = intent_extractor.get_intent()
intents.append(intent_vec.cpu())
```

### 변경 후

**모든 timestep에서 추출한다** (chunk 시작점만이 아님!).
- chunk 시작점만 추출하면 중간 step들이 학습 데이터로 쓸 수 없음
- 모든 step에서 뽑아야 (image_t, image_t+1, proprio_t, intent_t) 쌍을 최대한 활용 가능
- 저장 용량은 크지만 /mnt/md1/solee에 충분한 공간 있음

```python
def extract_intents_offline(groot_model, dataset_path, output_dir):
    """
    BridgeData의 **모든 에피소드, 모든 timestep**에서 GR00T forward pass를 돌려
    full intent token sequence를 추출하여 저장.

    ※ 모든 step에서 추출하는 이유:
      - chunk 시작점만 추출하면 중간 step의 (image, intent) 쌍이 학습에서 빠짐
      - 각 step의 observation이 다르므로 intent도 미묘하게 다름
      - forward model 학습 데이터를 최대화하려면 모든 step 필요

    저장 형식: numpy memmap (디스크에 저장, 학습 시 RAM에 전체 로드)
    저장 위치: /mnt/md1/solee/bridge_intent_tokens/
    구조:
      /mnt/md1/solee/bridge_intent_tokens/
        intent_tokens.npy      # memmap, shape (total_steps, 108, 2048), dtype=float16
        attention_masks.npy    # memmap, shape (total_steps, 108), dtype=bool
        episode_index.json     # {"episode_0": {"start": 0, "end": 42}, ...}

    저장 용량 추정:
      - per step: 108 × 2048 × 2 bytes (fp16) = ~432 KB
      - BridgeData ~50K episodes × ~40 steps = ~2M steps
      - intent_tokens: 2M × 432KB = ~820 GB
      - attention_masks: 2M × 108 bytes = ~200 MB (무시 가능)
      - 총합: ~820 GB → /mnt/md1/solee에 저장
      - 학습 시: RAM (1.5TB)에 전부 로드하여 사용 (random access ~1000x faster than disk)
    """
    import numpy as np
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    intent_extractor = IntentExtractor(groot_model)
    dataset = load_bridge_dataset(dataset_path)

    # === 1단계: 전체 step 수 계산 (memmap 크기 결정) ===
    total_steps = 0
    episode_lengths = {}
    for episode_idx, episode in enumerate(dataset):
        ep_len = len(episode)
        episode_lengths[f'episode_{episode_idx}'] = ep_len
        total_steps += ep_len

    print(f"Total steps: {total_steps}, Episodes: {len(dataset)}")
    print(f"Estimated storage: {total_steps * 432 / 1024 / 1024:.1f} GB")

    # === 2단계: memmap 파일 생성 ===
    intent_tokens_mmap = np.memmap(
        output_dir / 'intent_tokens.npy',
        dtype=np.float16,
        mode='w+',
        shape=(total_steps, 108, 2048),
    )
    attention_masks_mmap = np.memmap(
        output_dir / 'attention_masks.npy',
        dtype=np.bool_,
        mode='w+',
        shape=(total_steps, 108),
    )

    # === 3단계: 추출 및 저장 ===
    episode_index = {}
    global_step = 0

    for episode_idx, episode in enumerate(dataset):
        ep_start = global_step

        for t in range(len(episode)):
            obs = episode.get_observation(t)
            with torch.no_grad():
                _ = groot_model.get_action(obs)
                tokens, mask = intent_extractor.get_intent_tokens()

                # GPU → CPU → numpy → memmap에 직접 쓰기
                intent_tokens_mmap[global_step] = tokens[0].cpu().numpy().astype(np.float16)
                attention_masks_mmap[global_step] = mask[0].cpu().numpy()

            global_step += 1

        episode_index[f'episode_{episode_idx}'] = {
            'start': ep_start,
            'end': global_step,
            'language': episode.language_instruction,
        }

        if episode_idx % 100 == 0:
            # 중간중간 flush (crash 대비)
            intent_tokens_mmap.flush()
            attention_masks_mmap.flush()
            print(f"Episode {episode_idx}/{len(dataset)}, step {global_step}/{total_steps}")

    # === 4단계: 인덱스 저장 ===
    intent_tokens_mmap.flush()
    attention_masks_mmap.flush()

    with open(output_dir / 'episode_index.json', 'w') as f:
        json.dump(episode_index, f, indent=2)

    print(f"Done! Saved {global_step} steps to {output_dir}")
    print(f"intent_tokens.npy: {global_step * 108 * 2048 * 2 / 1024**3:.1f} GB")
```

---

## 변경 3: IntentForwardModel — MLP → Self-Attention

### 이전 (03.md)
```python
class IntentForwardModel(nn.Module):
    # MLP: concat(z_t, proprio, intent_vec) = 2440 → hidden 256 → 384
    # 문제: 9.5x bottleneck, intent 정보 유실
```

### 변경 후: Self-Attention Predictor (I-JEPA style)
```python
class IntentForwardModel(nn.Module):
    """
    I-JEPA 스타일 self-attention predictor.

    이전 MLP 대비 장점:
    1. 2440→256 bottleneck 제거 — 각 token이 384-dim 유지
    2. Intent tokens 간 관계 보존 — image-text cross attention 자연스럽게 발생
    3. z_t, proprio가 intent tokens를 선택적으로 attend
    4. Intent tokens도 z_t를 보고 contextualize (cross-attention 대비 양방향)
       → "현재 state가 이러니까 이 intent에서 중요한 부분은 이거"

    Token 구성 (N = 2 + num_active_intent_tokens):
      [z_t_proj, proprio_proj, intent_1, intent_2, ..., intent_K]
      - z_t_proj: (B, 1, 384) — DINOv2 visual feature projected
      - proprio_proj: (B, 1, 384) — proprioception projected
      - intent_k: (B, K, 384) — GR00T backbone tokens projected (K ≤ 108)
      - attention_mask로 padding 처리 (language instruction 길이 가변)

    Self-attention 후 z_t position의 output → Δz_ideal [384]

    설계 결정:
    - d_model = 384 (DINOv2 native dim과 일치, projection overhead 최소화)
    - num_layers = 2 (110 tokens은 작은 sequence, 2 layers면 충분)
    - num_heads = 6 (384 / 64 = 6)
    - FFN dim = 1536 (4x expansion, standard)

    Latency: ~0.3ms on RTX 4090 (전체 inference budget 200ms의 0.15%)
    Parameter count: ~2 × (4 × 384² + 2 × 384 × 1536) ≈ 3.5M params
    """

    def __init__(
        self,
        feature_dim: int = 384,         # DINOv2 native dim = self-attention d_model
        proprio_dim: int = 8,           # WidowX: 8-dim
        intent_dim: int = 2048,         # GR00T backbone hidden dim
        num_layers: int = 2,
        num_heads: int = 6,             # 384 / 64 = 6
        ffn_dim: int = 1536,            # 4x expansion
        dropout: float = 0.1,
        max_intent_tokens: int = 108,   # GR00T: 81 image + 27 text (max)
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_intent_tokens = max_intent_tokens

        # === Projection layers (각 modality → 384-dim tokens) ===

        # z_t는 이미 384-dim → identity (projection 불필요)
        # 하지만 learnable token embedding을 더해서 "이건 visual state token이다" 표시
        self.z_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # proprio: 8 → 384
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, feature_dim),
            nn.GELU(),
        )
        self.proprio_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # intent: 2048 → 384 (per token)
        self.intent_proj = nn.Linear(intent_dim, feature_dim)
        self.intent_token_embed = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # === Positional encoding ===
        # z_t, proprio는 고정 position (0, 1), intent tokens는 position (2, 3, ..., 109)
        # learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, 2 + max_intent_tokens, feature_dim) * 0.02
        )  # (1, 110, 384)

        # === Transformer layers ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,            # (B, N, D) format
            norm_first=True,             # Pre-LN (더 안정적 학습)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # === Output head ===
        # z_t position output → Δz_ideal
        self.output_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self,
        z: torch.Tensor,                      # (B, 384) — current visual feature
        proprio: torch.Tensor,                 # (B, 8) — current proprioception
        intent_tokens: torch.Tensor,           # (B, 108, 2048) — full GR00T backbone tokens
        intent_attention_mask: torch.Tensor,   # (B, 108) — True for active tokens
    ) -> torch.Tensor:
        """
        Returns:
            delta_z_ideal: (B, 384) — "이 intent를 향해 가면 이만큼 변해야 해"

        주의: 이전 MLP 버전과 시그니처가 다름!
        - 이전: forward(z, proprio, intent_vec)  ← intent_vec (B, 2048) pooled
        - 현재: forward(z, proprio, intent_tokens, intent_attention_mask) ← full sequence
        """
        B = z.shape[0]

        # 1. Token 생성
        # z_t token: (B, 1, 384)
        z_token = z.unsqueeze(1) + self.z_token_embed  # (B, 1, 384)

        # proprio token: (B, 1, 384)
        proprio_token = self.proprio_proj(proprio).unsqueeze(1) + self.proprio_token_embed

        # intent tokens: (B, 108, 384)
        intent_projected = self.intent_proj(intent_tokens) + self.intent_token_embed

        # 2. Concatenate: [z_t, proprio, intent_1, ..., intent_108]
        tokens = torch.cat([z_token, proprio_token, intent_projected], dim=1)  # (B, 110, 384)

        # 3. Add positional embedding
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # 4. Attention mask 구성
        # z_t, proprio는 항상 active → True
        # intent tokens는 attention_mask에 따라 active/padding
        prefix_mask = torch.ones(B, 2, dtype=torch.bool, device=z.device)  # z_t, proprio
        full_mask = torch.cat([prefix_mask, intent_attention_mask], dim=1)  # (B, 110)

        # PyTorch TransformerEncoder는 src_key_padding_mask를 받음
        # True = padding (ignore) → 반전 필요
        padding_mask = ~full_mask  # True where padding

        # 5. Self-attention
        output = self.transformer(tokens, src_key_padding_mask=padding_mask)  # (B, 110, 384)

        # 6. z_t position의 output만 읽기
        z_output = output[:, 0, :]  # (B, 384) — 첫 번째 token = z_t

        # 7. Output head → Δz_ideal
        delta_z_ideal = self.output_head(z_output)  # (B, 384)

        return delta_z_ideal
```

---

## 변경 4: ProprioForwardModel — Intent Pooled 사용 (유지)

ProprioForwardModel은 MLP 그대로 유지한다.
- proprio (8-dim) 예측에는 self-attention이 과도함
- 대신 intent_tokens의 pooled version (2048-dim) 사용

**변경 없음**, 단 입력이 명시적으로 pooled intent임을 표시:
```python
# ProprioForwardModel 호출 시:
# training loop 내에서 직접 pooling
mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, 108, 1)
intent_pooled = (intent_tokens * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
# → (B, 2048)
delta_proprio = proprio_forward(proprio_t, intent_pooled)
```

---

## 변경 5: Phase 1 Training Loop 수정

### 주요 변경점
1. Dataset이 `intent_tokens (B, 108, 2048)` + `attention_mask (B, 108)` 반환
2. IntentForwardModel 호출 시 token sequence + mask 전달
3. ProprioForwardModel은 pooled intent 사용

```python
def train_phase1(config):
    """
    Phase 1 변경점:
    - IntentForwardModelDataset이 이제 intent_tokens + attention_mask 반환
    - forward_model() 호출 시그니처 변경
    """

    dataset = IntentForwardModelDataset(
        dataset_path=config.dataset_path,
        intent_path=config.intent_path,     # /mnt/md1/solee/bridge_intent_tokens/
        image_size=config.image_size,
    )
    dataloader = DataLoader(dataset, batch_size=config.phase1.batch_size, shuffle=True)

    # Models (IntentForwardModel이 self-attention으로 변경됨)
    online_encoder = CerebellumVisualEncoder(use_lora=True, lora_rank=config.visual_lora_rank).cuda()
    ema_encoder = EMAEncoder(online_encoder, tau=config.ema_tau)

    forward_model = IntentForwardModel(
        feature_dim=384,
        proprio_dim=8,
        intent_dim=2048,
        num_layers=2,           # ← NEW: self-attention layers
        num_heads=6,            # ← NEW
        ffn_dim=1536,           # ← NEW
    ).cuda()

    proprio_forward = ProprioForwardModel(
        proprio_dim=8, intent_dim=2048,
    ).cuda()

    optimizer = torch.optim.AdamW([
        {'params': online_encoder.parameters(), 'lr': config.phase1.encoder_lr},
        {'params': forward_model.parameters(), 'lr': config.phase1.learning_rate},
        {'params': proprio_forward.parameters(), 'lr': config.phase1.learning_rate},
    ], weight_decay=config.phase1.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.phase1.num_epochs)

    def vicreg_variance_loss(z, target_std=1.0):
        std = z.std(dim=0)
        return torch.mean(torch.relu(target_std - std))

    for epoch in range(config.phase1.num_epochs):
        for batch in dataloader:
            img_t = batch['image_t'].cuda()
            img_t1 = batch['image_t1'].cuda()
            proprio_t = batch['proprio_t'].cuda()
            proprio_t1 = batch['proprio_t1'].cuda()
            intent_tokens = batch['intent_tokens'].cuda().float()   # (B, 108, 2048) — fp16→fp32
            attention_mask = batch['attention_mask'].cuda()          # (B, 108) ← NEW

            # Online encoder → z_t
            z_t = online_encoder(img_t)
            with torch.no_grad():
                z_t1_target = ema_encoder.encode(img_t1)

            # Forward model — self-attention (시그니처 변경!)
            delta_z_target = z_t1_target - z_t.detach()
            delta_z_ideal = forward_model(
                z_t, proprio_t,
                intent_tokens,        # ← (B, 108, 2048) full sequence
                attention_mask,       # ← (B, 108) padding mask
            )

            # Proprio forward — pooled intent 사용
            # attention_mask로 mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, 108, 1)
            intent_pooled = (intent_tokens * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            # → (B, 2048)

            delta_proprio_target = proprio_t1 - proprio_t
            delta_proprio_pred = proprio_forward(proprio_t, intent_pooled)

            # Losses (동일)
            visual_loss = F.mse_loss(delta_z_ideal, delta_z_target)
            proprio_loss = F.mse_loss(delta_proprio_pred, delta_proprio_target)
            vicreg_loss = vicreg_variance_loss(z_t)
            loss = visual_loss + proprio_loss + config.vicreg_lambda * vicreg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_encoder.update(online_encoder)

        scheduler.step()
```

---

## 변경 6: Phase 2 Training Loop 수정

Phase 1과 동일한 패턴으로 intent_tokens + attention_mask 전달.

```python
def train_phase2(config):
    # ... (모델 로드 동일) ...

    for batch in dataloader:
        img_t = batch['image_t'].cuda()
        img_t1 = batch['image_t1'].cuda()
        action_expert = batch['action_expert'].cuda()
        action_vla = batch['action_vla'].cuda()
        proprio_t = batch['proprio_t'].cuda()
        intent_tokens = batch['intent_tokens'].cuda().float()   # (B, 108, 2048)
        attention_mask = batch['attention_mask'].cuda()          # (B, 108)

        with torch.no_grad():
            z_t = online_encoder(img_t)
            z_t1 = online_encoder(img_t1)

            # Self-attention forward model (시그니처 변경!)
            delta_z_ideal = forward_model(
                z_t, proprio_t,
                intent_tokens,        # ← full sequence
                attention_mask,       # ← padding mask
            )
            delta_z_actual = z_t1 - z_t
            prediction_error = delta_z_actual - delta_z_ideal

        # 나머지 동일
        correction_target = action_expert - action_vla
        delta_a = correction_net(prediction_error, action_vla, proprio_t)
        loss = F.mse_loss(delta_a, correction_target)
        ...
```

---

## 변경 7: IntentCerebellumModule (Inference) 수정

```python
class IntentCerebellumModule(nn.Module):
    """
    변경점:
    - on_new_chunk()이 intent_tokens + attention_mask를 받음
    - correct()에서 forward_model 호출 시 token sequence 전달
    """

    def on_new_chunk(self, intent_tokens: torch.Tensor, intent_attention_mask: torch.Tensor):
        """
        새 action chunk 시작 시 호출.

        이전: on_new_chunk(intent_vec: Tensor)  — (B, 2048) pooled
        현재: on_new_chunk(intent_tokens, intent_attention_mask)  — (B, 108, 2048) + (B, 108)
        """
        self.current_intent_tokens = intent_tokens.detach()
        self.current_intent_mask = intent_attention_mask.detach()

    @torch.no_grad()
    def correct(self, image_current, proprio_current, action_planned):
        z_current = self.visual_encoder(image_current)

        if self.prev_z is not None and self.current_intent_tokens is not None:
            delta_z_ideal = self.forward_model(
                self.prev_z, self.prev_proprio,
                self.current_intent_tokens,       # ← full sequence
                self.current_intent_mask,          # ← padding mask
            )
            delta_z_actual = z_current - self.prev_z
            prediction_error = delta_z_actual - delta_z_ideal
        else:
            prediction_error = torch.zeros(1, 384, device=image_current.device)

        delta_a = self.correction_net(
            prediction_error=prediction_error,
            action_vla=action_planned,
            proprio=proprio_current,
        )

        action_corrected = action_planned + delta_a

        self.prev_z = z_current.detach()
        self.prev_proprio = proprio_current.detach()
        return action_corrected

    def reset(self):
        self.prev_z = None
        self.prev_proprio = None
        self.prev_prediction_error = None
        self.current_intent_tokens = None
        self.current_intent_mask = None
```

---

## 변경 8: Dataset 클래스 — RAM 전체 로드

**RAM 1.5TB 사용 가능** → intent_tokens (~820GB)를 전부 RAM에 올린다.
memmap의 random access (shuffle 시 page fault ~50-100μs)보다 RAM direct access (~100ns)가 **~1000배 빠름**.

저장은 디스크 (`/mnt/md1/solee/`)에 하되, 학습 시작 시 전부 RAM으로 로드.

```python
class IntentForwardModelDataset(Dataset):
    """
    변경점:
    - 디스크의 numpy 파일을 학습 시작 시 RAM에 전부 로드
    - RAM 1.5TB에서 ~820GB intent_tokens + ~200MB masks = 여유 있음
    - random shuffle 시 disk I/O 없이 순수 메모리 접근 → 학습 속도 최대화
    - episode_index.json으로 에피소드별 step 범위 관리
    """

    def __init__(self, dataset_path, intent_dir, image_size=98):
        """
        Args:
            dataset_path: BridgeData lerobot 경로
            intent_dir: /mnt/md1/solee/bridge_intent_tokens/ (저장 디렉토리)
        """
        import json
        from pathlib import Path

        self.data = load_bridge_dataset(dataset_path)
        self.image_size = image_size

        intent_dir = Path(intent_dir)

        # episode index 로드
        with open(intent_dir / 'episode_index.json', 'r') as f:
            self.episode_index = json.load(f)

        total_steps = sum(
            v['end'] - v['start'] for v in self.episode_index.values()
        )

        # === RAM에 전체 로드 ===
        # np.load with mmap_mode=None (default) → 전부 RAM에 올림
        # 또는 memmap으로 열고 np.array()로 복사
        print(f"Loading intent_tokens ({total_steps} steps, "
              f"~{total_steps * 108 * 2048 * 2 / 1024**3:.0f} GB) into RAM...")

        mmap_tokens = np.memmap(
            intent_dir / 'intent_tokens.npy',
            dtype=np.float16, mode='r',
            shape=(total_steps, 108, 2048),
        )
        self.intent_tokens = np.array(mmap_tokens)  # memmap → RAM 복사
        del mmap_tokens  # memmap 핸들 해제

        mmap_masks = np.memmap(
            intent_dir / 'attention_masks.npy',
            dtype=np.bool_, mode='r',
            shape=(total_steps, 108),
        )
        self.attention_masks = np.array(mmap_masks)
        del mmap_masks

        print(f"Loaded! intent_tokens: {self.intent_tokens.shape}, "
              f"attention_masks: {self.attention_masks.shape}")

        # 학습용 sample 리스트 구성: (episode_idx, step_t, global_step_t)
        # 각 sample = (image_t, image_t+1, proprio_t, proprio_t+1, intent_tokens_t, attention_mask_t)
        self.samples = []
        for ep_key, ep_info in self.episode_index.items():
            ep_idx = int(ep_key.split('_')[1])
            start = ep_info['start']
            end = ep_info['end']
            ep_len = end - start
            # 연속 쌍 (t, t+1) → t는 0부터 ep_len-2까지
            for local_t in range(ep_len - 1):
                self.samples.append((ep_idx, local_t, start + local_t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, local_t, global_t = self.samples[idx]

        # image, proprio 로드 (기존과 동일)
        episode = self.data[ep_idx]
        img_t = load_and_resize_image(episode, local_t, self.image_size)       # (3, 98, 98)
        img_t1 = load_and_resize_image(episode, local_t + 1, self.image_size)  # (3, 98, 98)
        proprio_t = episode.get_proprio(local_t)                                # (8,)
        proprio_t1 = episode.get_proprio(local_t + 1)                           # (8,)

        # intent tokens — RAM에서 직접 읽기 (zero disk I/O)
        intent_tokens = torch.from_numpy(
            self.intent_tokens[global_t].copy()  # copy()로 writable tensor
        )  # (108, 2048) float16
        attention_mask = torch.from_numpy(
            self.attention_masks[global_t].copy()
        )  # (108,) bool

        return {
            'image_t': img_t,                # (3, 98, 98)
            'image_t1': img_t1,              # (3, 98, 98)
            'proprio_t': proprio_t,          # (8,)
            'proprio_t1': proprio_t1,        # (8,)
            'intent_tokens': intent_tokens,  # (108, 2048) float16 → training loop에서 .float()
            'attention_mask': attention_mask, # (108,) bool
        }
```

---

## 변경 9: Config 업데이트

```yaml
# cerebellar_correction/configs/cerebellum_config.yaml

# ===== Forward Model (변경됨!) =====
# 이전: MLP (forward_hidden_dim: 256, forward_num_layers: 3)
# 현재: Self-Attention
forward_model_type: "self_attention"    # ← NEW
forward_num_layers: 2                   # self-attention layers (이전: MLP layers 3)
forward_num_heads: 6                    # 384 / 64
forward_ffn_dim: 1536                   # 4x expansion
forward_dropout: 0.1
max_intent_tokens: 108                  # GR00T: 81 image + 27 text

# 삭제:
# forward_hidden_dim: 256  ← MLP 전용, 더 이상 사용 안 함

# ===== Intent (변경됨!) =====
intent_dim: 2048
intent_format: "token_sequence"         # ← NEW: "pooled" → "token_sequence"

# ===== Paths (변경됨!) =====
dataset_path: "/mnt/md1/solee/data/bridge_lerobot"
groot_model_path: "/mnt/md1/solee/checkpoints/GR00T-N1.6-bridge"
checkpoint_dir: "/mnt/md1/solee/checkpoints/cerebellum_v2"
intent_dir: "/mnt/md1/solee/bridge_intent_tokens"    # ← NEW: memmap 디렉토리
vla_actions_path: "/mnt/md1/solee/data/vla_actions.pt"

# 삭제:
# intent_path: "data/bridge_intents.pt"  ← 단일 .pt 파일 → memmap 디렉토리로 변경
```

---

## 변경 요약 체크리스트

Claude Code에게 전달할 때 이 순서대로 수정:

1. [ ] **IntentExtractor**: `get_intent()` → `get_intent_tokens()` + `get_intent_pooled()` 분리
2. [ ] **extract_intents_offline()**: 전면 재작성
   - 모든 timestep에서 추출 (chunk 시작점만이 아님!)
   - numpy memmap으로 저장 (torch.save가 아님!)
   - 저장 위치: `/mnt/md1/solee/bridge_intent_tokens/`
   - 저장 형식: `intent_tokens.npy` (float16), `attention_masks.npy` (bool), `episode_index.json`
3. [ ] **IntentForwardModel**: MLP 전체 삭제 → Self-Attention Predictor로 교체
   - 입력 시그니처: `(z, proprio, intent_vec)` → `(z, proprio, intent_tokens, intent_attention_mask)`
4. [ ] **ProprioForwardModel**: 변경 없음 (pooled intent 계속 사용, training loop에서 pooling)
5. [ ] **CorrectionNetwork**: 변경 없음
6. [ ] **Phase 1 training loop**: forward_model 호출부 수정 (token sequence 전달)
7. [ ] **Phase 2 training loop**: 동일하게 수정
8. [ ] **IntentCerebellumModule**: on_new_chunk(), correct() 시그니처 수정
9. [ ] **Dataset 클래스**: torch.load → numpy memmap 기반으로 전면 재작성
10. [ ] **Config**: 경로 변경 (intent_path → intent_dir), forward model 파라미터 변경
11. [ ] **기존 bridge_intents.pt 삭제 가능** (새 memmap 형식으로 재추출 필요)

**변경하지 않는 것들**:
- CerebellumVisualEncoder (DINOv2+LoRA+EMA) — 그대로
- EMAEncoder — 그대로
- CorrectionNetwork — 그대로
- ProprioForwardModel — 그대로 (pooled intent 사용)
- VICReg variance loss — 그대로
- Phase 2의 correction_target = action_expert - action_vla — 그대로
