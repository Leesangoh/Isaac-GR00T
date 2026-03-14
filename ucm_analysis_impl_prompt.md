# UCM Analysis of VLA Action Prediction Errors — Implementation Prompt

## 프로젝트 개요

**연구 질문**: VLA (Vision-Language-Action) 모델의 action prediction error가 random noise인가, 아니면 task-irrelevant 차원에 집중된 structured variability인가?

**배경**: GR00T N1.6 VLA는 BridgeData V2에서 per-step R²=0.10-0.29 (연속 action 차원)의 낮은 예측 정확도를 보이지만, 실제 manipulation task는 상당히 성공적으로 수행한다. Motor neuroscience의 Uncontrolled Manifold (UCM) 가설 (Scholz & Schöner, 1999)에 따르면, 이 "error"는 task 결과에 영향을 주지 않는 null space에 집중된 기능적 variability일 수 있다.

**목표**: 기존 VLA Error Analysis 데이터를 활용하여 UCM 분해를 수행하고, VLA error의 구조를 정량적으로 분석한다.

---

## 기존 인프라 & 데이터

### 코드 위치
- **Isaac-GR00T**: `https://github.com/NVIDIA/Isaac-GR00T` (설치 완료 가정)
- **작업 레포**: 새 깃 브랜치에서 작업 (기존 cerebellar_correction 코드 없음, clean state)
- **ucm_analysis 코드**: 이 프로젝트를 위해 새로 작성

### 데이터 경로
- **BridgeData V2 원본**: `/mnt/md1/solee/data/bridge_lerobot`
  - LeRobot 형식의 BridgeData V2 (~53K episodes)
  - parquet 파일들, `action` column에 expert action [x, y, z, roll, pitch, yaw, gripper]
- **VLA actions (GR00T output)**: `/mnt/md1/solee/features/vla_actions`
  - BridgeData의 각 observation에 대해 GR00T N1.6이 예측한 action 저장
  - 40,855 episodes 추출 완료
  - **중요**: 파일 구조 및 포함된 데이터를 먼저 확인해야 함 (아래 Phase 0 참조)
  - 가능한 형식: `.pt` (PyTorch tensor) 또는 `.npz` (numpy)
  - 포함 가능 데이터: expert action, VLA predicted action (8-step chunk), intent tokens 등

### Action 형식
```
[x, y, z, roll, pitch, yaw, gripper] × 8 steps
```
- Position (x,y,z): End-effector Cartesian deltas
- Orientation (roll, pitch, yaw): Euler angle deltas
- Gripper: Binary 0(close)/1(open)
- Action chunk: 8 steps → 56D vector (7D × 8 steps)

### VLA Error Analysis 핵심 수치 (1,000 episodes, 36,027 timesteps)

| Dim | R² | VLA_std/Expert_std | MAE/Expert_std |
|---------|-------|---------------------|-----------------|
| x | 0.177 | 0.594 | 0.605 |
| y | 0.233 | 0.606 | 0.569 |
| z | 0.288 | 0.611 | 0.573 |
| roll | 0.141 | 0.539 | 0.606 |
| pitch | 0.105 | 0.567 | 0.637 |
| yaw | 0.157 | 0.553 | 0.588 |
| gripper | 0.667 | 1.000 | 0.168 |

핵심: 모든 연속 차원에서 R²=0.10-0.29 (매우 낮음), VLA_std가 Expert_std의 54-61% (conservative bias).

### GR00T N1.6 Architecture (관련 부분만)
- Backbone: Eagle3 (SigLIP2 + Qwen3 1.7B) → [B, seq_len, 2048]
- Action Head: DiT (32 layers) + Flow Matching (4 denoising steps, Euler integration dt=0.25)
- `state_dropout_prob=0.8`: inference에서도 항상 active → 80% 확률로 proprioception masked
- Inference stochasticity: random initial noise + state dropout → 매 inference마다 다른 action sample

---

## 구현 계획

### Phase 0: 데이터 준비 및 기존 데이터 확인

**목표**: 기존 추출 데이터의 정확한 구조를 확인하고, UCM 분석에 필요한 형태로 정리

**작업**:

1. VLA actions 디렉토리 탐색 및 파일 구조 확인
   ```bash
   # 디렉토리 구조 확인
   ls /mnt/md1/solee/features/vla_actions/ | head -20
   ls /mnt/md1/solee/features/vla_actions/ | wc -l

   # 파일 하나 로드해서 구조 확인
   python -c "
   import torch, glob, os, numpy as np
   base = '/mnt/md1/solee/features/vla_actions'

   # .pt 파일 탐색
   pt_files = sorted(glob.glob(f'{base}/**/*.pt', recursive=True))[:3]
   for f in pt_files:
       data = torch.load(f, map_location='cpu', weights_only=False)
       print(f'{f}: type={type(data)}')
       if isinstance(data, dict):
           for k, v in data.items():
               shape = v.shape if hasattr(v, 'shape') else type(v)
               print(f'  {k}: {shape}')

   # .npz 파일 탐색
   npz_files = sorted(glob.glob(f'{base}/**/*.npz', recursive=True))[:3]
   for f in npz_files:
       data = np.load(f)
       print(f'{f}: keys={list(data.keys())}')
       for k in data.keys():
           print(f'  {k}: shape={data[k].shape}, dtype={data[k].dtype}')
   "
   ```
   **확인해야 할 것**:
   - VLA predicted action이 포함되어 있는가? (8-step chunk 전체? 아니면 step 0만?)
   - Expert action도 함께 저장되어 있는가?
   - Normalization 상태: normalized (GR00T 내부 값) vs unnormalized (물리적 단위)

2. BridgeData V2 원본 구조 확인
   ```bash
   ls /mnt/md1/solee/data/bridge_lerobot/ | head -20

   python -c "
   import pandas as pd, glob
   base = '/mnt/md1/solee/data/bridge_lerobot'
   pq_files = sorted(glob.glob(f'{base}/**/*.parquet', recursive=True))[:2]
   for f in pq_files:
       df = pd.read_parquet(f)
       print(f'{f}: columns={list(df.columns)}, shape={df.shape}')
       if 'action' in df.columns:
           print(f'  action[0] shape: {df[\"action\"].iloc[0].shape if hasattr(df[\"action\"].iloc[0], \"shape\") else type(df[\"action\"].iloc[0])}')
   "
   ```

3. 두 데이터소스 간 episode 매칭 방식 확인
   - vla_actions의 파일명 ↔ bridge_lerobot의 episode ID 매칭 로직 파악

**산출물**: 데이터 로더 유틸리티 (`ucm_analysis/data_loader.py`)

---

### Phase 1: VLA-Expert Error의 UCM 분해 (기존 데이터, 추가 inference 불필요)

**이 Phase의 핵심 아이디어**:

기존에 추출된 VLA prediction과 Expert action의 차이 (error = a_VLA - a_expert)를 여러 timestep에 걸쳐 수집하고, 이 error 벡터들을 task-relevant / task-irrelevant 부분으로 분해한다.

**Task Variable 정의**:

BridgeData V2의 pick-and-place task에서, action chunk 실행의 결과로 중요한 것:

*Task Variable 1 (TV1): Cumulative EE Displacement*
```
TV1 = [Σᵢ Δxᵢ, Σᵢ Δyᵢ, Σᵢ Δzᵢ]  (3D)
```
해석: 8 step chunk 실행 후 end-effector가 어디로 이동했는가.

*Task Variable 2 (TV2): Cumulative EE Displacement + Orientation*
```
TV2 = [Σᵢ Δxᵢ, Σᵢ Δyᵢ, Σᵢ Δzᵢ, Σᵢ Δrollᵢ, Σᵢ Δpitchᵢ, Σᵢ Δyawᵢ]  (6D)
```
해석: chunk 실행 후 end-effector의 position + orientation 변화.

*Task Variable 3 (TV3): Per-Step Position (all 8 steps)*
```
TV3 = [Δx₁, Δy₁, Δz₁, Δx₂, Δy₂, Δz₂, ..., Δx₈, Δy₈, Δz₈]  (24D)
```
해석: 모든 step의 position이 task-relevant. Null space = orientation + gripper 변동.

**UCM 분해 알고리즘**:

```python
import numpy as np
from scipy.linalg import null_space

def compute_ucm_decomposition(errors, J):
    """
    UCM decomposition of error vectors.

    Args:
        errors: (N, D) array of error vectors (VLA - Expert), D=56 for 7D×8steps
        J: (task_dim, D) Jacobian matrix mapping action chunk to task variable

    Returns:
        V_ucm: variance per DOF in the uncontrolled manifold (null space of J)
        V_ort: variance per DOF in the orthogonal complement (range space of J.T)
        ratio: V_ucm / V_ort
    """
    N, D = errors.shape

    # 1. Compute null space of J (= UCM)
    UCM_basis = null_space(J)  # (D, d_ucm) where d_ucm = D - rank(J)
    d_ucm = UCM_basis.shape[1]

    # 2. Compute orthogonal complement (= ORT)
    # ORT basis = column space of J.T
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    rank = np.sum(S > 1e-10)
    ORT_basis = Vt[:rank].T  # (D, d_ort) where d_ort = rank(J)
    d_ort = ORT_basis.shape[1]

    # 3. Project each error onto UCM and ORT
    # proj_UCM(e) = UCM_basis @ (UCM_basis.T @ e)
    proj_ucm = errors @ UCM_basis  # (N, d_ucm)
    proj_ort = errors @ ORT_basis  # (N, d_ort)

    # 4. Compute variance per DOF
    V_ucm = np.sum(proj_ucm ** 2) / (N * d_ucm)
    V_ort = np.sum(proj_ort ** 2) / (N * d_ort)

    ratio = V_ucm / V_ort if V_ort > 0 else float('inf')

    return V_ucm, V_ort, ratio, d_ucm, d_ort


def build_jacobian_tv1(n_steps=8, action_dim=7):
    """
    Jacobian for TV1: Cumulative EE Displacement = [Σ Δxᵢ, Σ Δyᵢ, Σ Δzᵢ]

    Action chunk layout: [x₁,y₁,z₁,r₁,p₁,w₁,g₁, x₂,y₂,..., g₈]
    Total dimension: 7 × 8 = 56

    J is (3, 56): ∂TV/∂action
    - ∂(Σ Δxᵢ)/∂(Δxⱼ) = 1 for all j
    - ∂(Σ Δxᵢ)/∂(Δyⱼ) = 0, etc.
    """
    D = n_steps * action_dim  # 56
    J = np.zeros((3, D))
    for step in range(n_steps):
        base = step * action_dim
        J[0, base + 0] = 1.0  # ∂(Σx)/∂(xᵢ)
        J[1, base + 1] = 1.0  # ∂(Σy)/∂(yᵢ)
        J[2, base + 2] = 1.0  # ∂(Σz)/∂(zᵢ)
    return J


def build_jacobian_tv2(n_steps=8, action_dim=7):
    """
    Jacobian for TV2: Cumulative Position + Orientation
    TV = [Σ Δxᵢ, Σ Δyᵢ, Σ Δzᵢ, Σ Δrollᵢ, Σ Δpitchᵢ, Σ Δyawᵢ]
    J is (6, 56)
    """
    D = n_steps * action_dim
    J = np.zeros((6, D))
    for step in range(n_steps):
        base = step * action_dim
        for dim in range(6):  # x,y,z,roll,pitch,yaw
            J[dim, base + dim] = 1.0
    return J


def build_jacobian_tv3(n_steps=8, action_dim=7):
    """
    Jacobian for TV3: All per-step positions
    TV = [Δx₁,Δy₁,Δz₁,...,Δx₈,Δy₈,Δz₈] (24D)
    J is (24, 56): identity for position dims, zero for orientation/gripper
    """
    D = n_steps * action_dim
    tv_dim = n_steps * 3
    J = np.zeros((tv_dim, D))
    for step in range(n_steps):
        base_action = step * action_dim
        base_tv = step * 3
        J[base_tv + 0, base_action + 0] = 1.0  # xᵢ
        J[base_tv + 1, base_action + 1] = 1.0  # yᵢ
        J[base_tv + 2, base_action + 2] = 1.0  # zᵢ
    return J
```

**분석 단계**:

1. **데이터 로드**: 기존 추출된 VLA action chunks와 Expert action chunks를 로드
   - 각 timestep에서: `a_VLA` (7×8 = 56D), `a_expert` (7×8 = 56D)
   - Error: `e = a_VLA - a_expert` (56D)
   - 주의: 현재 데이터가 chunk step 0의 action만 가지고 있을 수 있음. 8-step chunk 전체가 필요.

2. **전체 UCM 분해**: 모든 timestep의 error를 모아서 세 가지 Task Variable에 대해 UCM 분해
   ```python
   errors = load_all_errors()  # (N_total, 56)

   for tv_name, build_J in [("TV1_position", build_jacobian_tv1),
                              ("TV2_pos_orient", build_jacobian_tv2),
                              ("TV3_per_step_pos", build_jacobian_tv3)]:
       J = build_J()
       V_ucm, V_ort, ratio, d_ucm, d_ort = compute_ucm_decomposition(errors, J)
       print(f"{tv_name}: V_UCM/dof={V_ucm:.6f}, V_ORT/dof={V_ort:.6f}, "
             f"ratio={ratio:.2f}, UCM_dim={d_ucm}, ORT_dim={d_ort}")
   ```

3. **Task Phase별 분석**: Episode를 phase로 세그먼트하고 phase별 UCM ratio 비교
   - Phase 세그먼트 heuristic:
     - **Approach**: gripper open (>0.5), large position changes
     - **Pre-grasp**: gripper open, small position changes (approaching object)
     - **Grasp**: gripper closes (transition from open to close)
     - **Transport**: gripper closed, large position changes
     - **Place**: gripper opens (transition from close to open)
   - 각 phase에서 UCM ratio가 다를 것으로 예상
   - 가설: Grasp phase에서 V_ORT가 상대적으로 높음 (task-relevant precision 필요)

4. **통계 검정**: UCM ratio가 1과 유의하게 다른지 검정
   - 귀무가설 H₀: V_UCM/dof = V_ORT/dof (uniform variance distribution)
   - Bootstrap 또는 permutation test로 p-value 계산
   - Episode 단위로 UCM ratio를 계산하여 분포 확인

5. **Chunk Step별 분석**: Action chunk 내 step 0~7에서 UCM 구조가 어떻게 변하는지
   - Later steps는 더 uncertain → V_UCM도 V_ORT도 커지겠지만, ratio는?

**산출물**:
- `ucm_analysis/phase1_error_decomposition.py` — 메인 분석 스크립트
- `ucm_analysis/task_variables.py` — Task Variable 정의 및 Jacobian 생성
- `ucm_analysis/visualization.py` — 결과 시각화
- 결과 plots: UCM ratio bar charts, phase별 분석, error projection scatter plots

---

### Phase 2: Multi-Sample Action Extraction (새 inference 필요)

**목표**: 동일 observation에서 GR00T N1.6을 K번 inference하여 action variability 수집

**왜 필요한가**: Phase 1은 "VLA prediction vs Expert" 차이의 구조를 분석한다. Phase 2는 "VLA 자체의 variability" 구조를 분석한다. 둘 다 UCM 가설을 검증하지만 다른 각도에서:
- Phase 1: VLA의 error가 task-irrelevant한가?
- Phase 2: VLA의 internal variability가 task-irrelevant한가?

**구현**:

```python
import torch
import numpy as np
from gr00t.policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

def extract_multi_samples(
    model_path: str,
    dataset_path: str,
    n_episodes: int = 200,
    K: int = 50,  # samples per observation
    output_dir: str = "ucm_analysis/multi_samples/",
    device: str = "cuda:0",
):
    """
    각 observation에서 K번 GR00T inference를 수행하여 action variability 수집.

    GR00T의 stochasticity 소스:
    1. state_dropout_prob=0.8 → 80% 확률로 proprioception masked
    2. Flow matching의 initial noise (torch.randn)

    두 소스 모두 매 inference마다 다른 seed → K개 다른 action sample.
    """
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.OXE_WIDOWX,
        model_path=model_path,
        device=device,
    )

    # BridgeData V2 로드 (LeRobot format from /mnt/md1/solee/data/bridge_lerobot)
    dataset = load_bridge_dataset(dataset_path, n_episodes=n_episodes)

    os.makedirs(output_dir, exist_ok=True)

    for ep_idx in range(n_episodes):
        episode = dataset[ep_idx]
        T = len(episode)

        # 각 timestep에서 K번 샘플링
        all_samples = np.zeros((T, K, 8, 7), dtype=np.float32)  # (T, K, chunk_steps, action_dim)
        expert_actions = np.zeros((T, 8, 7), dtype=np.float32)

        for t in range(T):
            obs = prepare_observation(episode, t)  # 기존 extract_all.py의 observation 준비 로직 참고

            for k in range(K):
                with torch.no_grad():
                    action_output = policy.model.get_action(obs)
                    # action_output["action_pred"]: (1, 16, 29) 또는 (1, 8, 7)
                    # BridgeData/WidowX는 7D action × 8 steps
                    action_chunk = action_output["action_pred"][0].cpu().numpy()

                    # Unnormalize (processor의 unnormalize 사용)
                    action_chunk = unnormalize_action(action_chunk, policy.processor)

                    all_samples[t, k] = action_chunk[:8, :7]  # 8 steps × 7 dims

            expert_actions[t] = get_expert_action_chunk(episode, t)  # 8-step expert chunk

        np.savez_compressed(
            f"{output_dir}/episode_{ep_idx:06d}.npz",
            samples=all_samples,      # (T, K, 8, 7)
            expert=expert_actions,     # (T, 8, 7)
        )
        print(f"Episode {ep_idx}: {T} timesteps × {K} samples extracted")
```

**계산 비용 예측**:
- GR00T inference: ~64ms per forward pass
- 200 episodes × ~36 timesteps/episode × 50 samples = 360,000 inference calls
- 총 시간: 360,000 × 64ms ≈ 6.4시간 (single GPU)
- A6000 4대 병렬 → ~1.6시간

**산출물**: `ucm_analysis/extract_multi_samples.py`

---

### Phase 3: Multi-Sample UCM 분석

**Phase 2의 데이터를 이용한 UCM 분석**:

```python
def analyze_multi_sample_ucm(data_dir, task_variable="TV1"):
    """
    동일 observation에서의 K개 action sample의 variability를 UCM 분해.

    Phase 1과의 차이:
    - Phase 1: 여러 timestep에 걸친 VLA-Expert error의 UCM 구조
    - Phase 3: 동일 timestep에서의 VLA 자체 variability의 UCM 구조
    """
    J = build_jacobian(task_variable)  # Task variable에 맞는 Jacobian

    results = []

    for episode_file in sorted(glob(f"{data_dir}/episode_*.npz")):
        data = np.load(episode_file)
        samples = data["samples"]   # (T, K, 8, 7)
        expert = data["expert"]     # (T, 8, 7)

        T, K, n_steps, action_dim = samples.shape
        D = n_steps * action_dim  # 56

        for t in range(T):
            # K개 샘플을 56D 벡터로 flatten
            action_vecs = samples[t].reshape(K, D)  # (K, 56)

            # 평균 제거
            mean_action = action_vecs.mean(axis=0)
            deviations = action_vecs - mean_action  # (K, 56)

            # UCM 분해
            V_ucm, V_ort, ratio, d_ucm, d_ort = compute_ucm_decomposition(deviations, J)

            results.append({
                "episode": episode_file,
                "timestep": t,
                "V_ucm": V_ucm,
                "V_ort": V_ort,
                "ratio": ratio,
                "K": K,
                "expert_action_norm": np.linalg.norm(expert[t].flatten()),
                "gripper_state": expert[t, 0, 6],  # gripper at step 0
            })

    return pd.DataFrame(results)
```

**추가 분석**:

1. **Expert Action Direction 분석**:
   ```python
   # Expert action 방향에서의 variance vs 직교 방향에서의 variance
   expert_dir = expert_action / np.linalg.norm(expert_action)  # unit vector
   var_along_expert = np.var(deviations @ expert_dir)
   var_ortho_expert = (np.sum(np.var(deviations, axis=0)) - var_along_expert) / (D - 1)
   ```
   이건 "VLA가 expert와 같은 방향으로 불확실한가?"를 검정

2. **Principal Component 분석**:
   ```python
   # K samples의 PCA → 주요 variability 방향 확인
   cov = (deviations.T @ deviations) / K
   eigenvalues, eigenvectors = np.linalg.eigh(cov)
   # 상위 eigenvalues의 eigenvectors가 UCM에 align되는지 확인
   ```

3. **Gripper Binary 분리**:
   - Gripper는 binary (0/1)이므로 continuous UCM 분석에서 제외
   - Position + orientation만으로 분석 (48D = 6D × 8 steps)

---

### Phase 4: 시각화

**필수 Plots**:

1. **UCM Ratio Bar Chart**: TV1, TV2, TV3 각각에 대한 V_UCM/V_ORT ratio
   - Horizontal line at ratio=1 (null hypothesis: uniform variance)
   - Error bars: episode 간 standard error

2. **Phase별 UCM Ratio**: Approach / Pre-grasp / Grasp / Transport / Place 별 ratio
   - 가설: Grasp에서 ratio가 가장 낮음 (task-relevant precision 필요)

3. **UCM Ratio Temporal Profile**: Episode 내 timestep에 따른 ratio 변화
   - Episode 시작(approach) → 중간(grasp) → 끝(place) trajectory

4. **Error Projection Scatter Plot**:
   - X축: ||proj_ORT(error)||, Y축: ||proj_UCM(error)||
   - 색: task success / failure (SimplerEnv에서 검증 시)
   - 가설: 성공 episode는 ORT가 작고 UCM이 큼

5. **Variance Eigenspectrum**: Error covariance의 eigenvalue 분포
   - UCM 방향과 ORT 방향에서의 eigenvalue 비교

6. **2D Projection Visualization**:
   - K개 action sample을 PCA 2D로 투영
   - UCM 방향 vs ORT 방향 overlay
   - Expert action 위치 표시

---

### 프로젝트 구조

**config.py 기본 설정**:
```python
# config.py
BRIDGE_DATA_DIR = "/mnt/md1/solee/data/bridge_lerobot"
VLA_ACTIONS_DIR = "/mnt/md1/solee/features/vla_actions"
GROOT_MODEL_PATH = "/path/to/groot-n1.6-bridge"  # Phase 2에서 필요, 실제 경로로 수정

N_STEPS = 8          # action chunk length
ACTION_DIM = 7       # [x, y, z, roll, pitch, yaw, gripper]
CONTINUOUS_DIMS = 6   # gripper 제외
```

```
ucm_analysis/
├── README.md
├── requirements.txt              # numpy, scipy, pandas, matplotlib, seaborn, torch
├── config.py                     # 경로 설정, 하이퍼파라미터
├── data_loader.py                # 기존 데이터 로드 유틸리티
├── task_variables.py             # Task Variable 정의 & Jacobian 생성
├── ucm_decomposition.py          # 핵심 UCM 분해 알고리즘
├── phase1_error_analysis.py      # Phase 1: VLA-Expert error UCM 분해
├── phase2_extract_samples.py     # Phase 2: Multi-sample extraction
├── phase3_sample_analysis.py     # Phase 3: Multi-sample UCM 분석
├── visualization.py              # 모든 시각화
├── phase_segmentation.py         # Episode를 task phase로 분할
├── statistical_tests.py          # 통계 검정 (bootstrap, permutation)
└── results/                      # 분석 결과 저장
    ├── figures/
    └── tables/
```

---

## 실행 순서

### Step 1: 데이터 확인 (Phase 0)
```bash
# VLA actions 구조 확인
ls /mnt/md1/solee/features/vla_actions/ | head -20
ls /mnt/md1/solee/features/vla_actions/ | wc -l

python -c "
import torch, glob, os, numpy as np
base = '/mnt/md1/solee/features/vla_actions'

# 모든 파일 형식 탐색
for ext in ['*.pt', '*.npz', '*.npy']:
    files = sorted(glob.glob(f'{base}/**/{ext}', recursive=True))
    if files:
        print(f'\n=== {ext} files: {len(files)} total ===')
        for f in files[:3]:
            if ext == '*.pt':
                data = torch.load(f, map_location='cpu', weights_only=False)
                print(f'{os.path.basename(f)}: type={type(data)}')
                if isinstance(data, dict):
                    for k, v in data.items():
                        print(f'  {k}: {v.shape if hasattr(v, \"shape\") else type(v)}')
            else:
                data = np.load(f, allow_pickle=True)
                print(f'{os.path.basename(f)}: keys={list(data.keys()) if hasattr(data, \"keys\") else \"array\"}')
                if hasattr(data, 'keys'):
                    for k in data.keys():
                        print(f'  {k}: shape={data[k].shape}, dtype={data[k].dtype}')

# BridgeData 원본 확인
print('\n=== BridgeData V2 ===')
bridge_base = '/mnt/md1/solee/data/bridge_lerobot'
os.system(f'ls {bridge_base}/ | head -20')
"
```

**중요**: VLA predicted action chunk (8 steps × 7 dims)이 기존 데이터에 포함되어 있는지 확인해야 함.
- 포함되어 있으면: 바로 Phase 1 진행 가능
- Expert action만 있으면: GR00T inference를 다시 돌려 VLA action을 추출해야 함
- 이 경우 Phase 2의 multi-sample extraction과 합쳐서 한번에 수행하는 것이 효율적

### Step 2: Phase 1 실행
```bash
cd ucm_analysis
python phase1_error_analysis.py \
    --vla_dir /mnt/md1/solee/features/vla_actions \
    --bridge_dir /mnt/md1/solee/data/bridge_lerobot \
    --output_dir results/phase1/
```

### Step 3: 결과 해석
- **V_UCM/V_ORT >> 1**: UCM 가설 지지 → Phase 2 진행
- **V_UCM/V_ORT ≈ 1**: Uniform variance → 가설 기각, 다른 설명 필요
- **V_UCM/V_ORT << 1**: VLA가 task-relevant 방향에서 더 variable → 예상 외, 흥미로운 발견

### Step 4: Phase 2-3 (Phase 1 결과가 positive일 경우)
```bash
python phase2_extract_samples.py \
    --model_path /path/to/groot-n1.6-bridge \
    --bridge_dir /mnt/md1/solee/data/bridge_lerobot \
    --n_episodes 200 --K 50 \
    --output_dir results/multi_samples/

python phase3_sample_analysis.py --data_dir results/multi_samples/ --output_dir results/phase3/
```

---

## 핵심 확인 사항

1. **기존 데이터에 VLA prediction의 full 8-step chunk가 있는가?**
   - `/mnt/md1/solee/features/vla_actions`의 파일 구조를 먼저 확인
   - 이전 VLA Error Analysis에서는 chunk step 0~7 모두 분석했으므로, 8-step chunk가 저장되어 있을 가능성 높음
   - 없으면: Isaac-GR00T를 이용하여 BridgeData에 대해 VLA inference를 다시 수행하여 추출

2. **Action의 normalization 상태**
   - 분석은 unnormalized (실제 물리적 단위) action에서 수행해야 함
   - GR00T 내부는 normalized space에서 작동 → 추출 시 unnormalize 필수
   - Normalization stats: VLA Error Analysis Section 8 참고

3. **Gripper 처리**
   - Gripper는 binary (0/1) → continuous UCM 분석에 부적합
   - Position (3D × 8) + Orientation (3D × 8) = 48D로 분석
   - Gripper는 별도 분석 (per-step gripper accuracy는 이미 R²=0.67으로 높음)

4. **Chunk Step 0만 있는 경우의 대안**
   - 만약 기존 데이터가 chunk step 0만 저장했다면:
   - 7D error vector에 대한 UCM 분석도 가능 (단, 차원이 낮아 UCM의 의미가 약해짐)
   - TV: [Δx, Δy, Δz] (3D) → null space 4D (orientation + gripper 방향)
   - 이것만으로도 "position error vs orientation error" 구조는 분석 가능

---

## 예상 결과 시나리오

### 시나리오 A: V_UCM >> V_ORT (가설 지지)
- **의미**: VLA error는 task-irrelevant dimension에 집중. VLA는 "중요한 곳에서만 정확".
- **다음 단계**: UCM-aware correction (task-relevant error만 교정) 설계 → Phase 4
- **논문 기여**: VLA의 error 구조에 대한 새로운 이해 + principled correction method

### 시나리오 B: V_UCM ≈ V_ORT (가설 기각)
- **의미**: VLA error는 균일하게 분포. 특별한 구조 없음.
- **다음 단계**: VLA success의 다른 설명 탐색 (e.g., error cancellation over trajectory)
- **논문 기여**: Negative result이지만, "VLA는 UCM을 활용하지 않는다"는 발견도 가치 있음

### 시나리오 C: Phase별로 다른 패턴
- **의미**: Approach에서는 V_UCM >> V_ORT이지만 Grasp에서는 ≈ 1
- **다음 단계**: Phase-adaptive correction (phase별 다른 전략)
- **논문 기여**: "VLA error 구조는 task phase에 따라 변한다" — 가장 흥미로운 결과

---

## 주의 사항

- 이 분석의 Jacobian은 **linear approximation**이다. 실제 robot dynamics는 nonlinear이지만, action이 small delta인 BridgeData에서는 linear approximation이 합리적.
- UCM 분석에서 dimension normalization이 핵심이다. V_UCM과 V_ORT를 각각의 DOF 수로 나누지 않으면, 단순히 차원 수 차이 때문에 V_UCM > V_ORT가 나올 수 있다 (trivial result).
- **Gripper를 포함하면 분석이 왜곡된다** — binary variable은 continuous variance 분석에 맞지 않으므로 반드시 제외.