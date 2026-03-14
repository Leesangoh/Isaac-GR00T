# Phase 4: UCM Validation & Deep-Dive Experiments

## 개요

Phase 1 (error UCM)과 Phase 3 (multi-sample variability UCM)의 결과가 나왔다. 핵심 수치:

| | TV1 (cum. pos) | TV2 (cum. pos+orient) | TV3 (per-step pos) |
|---|---|---|---|
| Phase 1 (error) | 3.35 | 0.41 (inverse) | 20.54 |
| Phase 3 (variability) | 11.37 | 1.54 | 19.11 |

Phase 1-3의 결과에는 **아직 해결되지 않은 critical한 의문**들이 있다. 이 Phase 4는 그 의문들을 해소하고, 논문에 쓸 수 있는 수준의 evidence를 만드는 것이 목표다.

**해결해야 할 핵심 의문 3가지:**

1. **Confound 검증**: TV3 ratio=20이 "orientation이 position보다 noisy하다"의 trivial한 restatement가 아닌지? → Surrogate test로 검증
2. **TV2 Dissociation의 의미**: Phase 1에서 0.41, Phase 3에서 1.54 — 이 차이가 "orientation error는 bias이고 noise가 아니다"를 정말로 증명하는지? → Orientation axis별 분해로 검증
3. **PCA-UCM alignment의 chance-level**: TV1의 alignment=0.997이 trivial한지 (UCM이 45/48차원이므로 random도 0.9375) → Chance baseline 계산 및 비교

---

## 실험 1: Surrogate Permutation Test (최우선)

### 목적

실제 UCM ratio가 dimension-wise variance 차이(position vs orientation의 단순한 스케일 차이)로 설명 가능한지 검증한다.

**귀무가설 H₀**: UCM ratio는 error의 방향적 구조가 아닌, dimension별 variance 크기 차이에서 발생한다.

### 방법

```python
import numpy as np
from scipy.linalg import null_space

def surrogate_test(errors, J, n_permutations=1000, seed=42):
    """
    Surrogate permutation test for UCM ratio.

    핵심 아이디어:
    각 dimension 내에서 timestep 순서를 독립적으로 shuffle한다.
    이렇게 하면 dimension별 variance는 보존되지만,
    dimension 간의 공분산 구조(= 방향적 구조)는 파괴된다.

    만약 실제 ratio가 surrogate distribution보다 유의하게 높으면,
    ratio가 단순한 variance scale 차이가 아닌 방향적 구조에서 온다는 증거.

    Args:
        errors: (N, D) — Phase 1이면 VLA-Expert error, Phase 3이면 deviations
        J: (task_dim, D) — Jacobian
        n_permutations: surrogate 생성 횟수
        seed: random seed

    Returns:
        actual_ratio: 실제 UCM ratio
        surrogate_ratios: (n_permutations,) surrogate ratio들
        p_value: actual_ratio가 surrogate distribution에서 나올 확률
        percentile: actual_ratio의 percentile
    """
    rng = np.random.RandomState(seed)
    N, D = errors.shape

    # 실제 ratio 계산
    actual_ratio = compute_ucm_ratio(errors, J)

    # Surrogate 생성
    surrogate_ratios = np.zeros(n_permutations)
    for i in range(n_permutations):
        # 각 dimension 내에서 독립적으로 shuffle
        shuffled = errors.copy()
        for d in range(D):
            rng.shuffle(shuffled[:, d])

        surrogate_ratios[i] = compute_ucm_ratio(shuffled, J)

    # p-value: actual ratio보다 큰 surrogate의 비율
    p_value = np.mean(surrogate_ratios >= actual_ratio)
    percentile = np.mean(surrogate_ratios <= actual_ratio) * 100

    return actual_ratio, surrogate_ratios, p_value, percentile


def compute_ucm_ratio(errors, J):
    """기존 UCM decomposition에서 ratio만 반환"""
    N, D = errors.shape
    UCM_basis = null_space(J)
    d_ucm = UCM_basis.shape[1]

    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    rank = np.sum(S > 1e-10)
    ORT_basis = Vt[:rank].T
    d_ort = ORT_basis.shape[1]

    proj_ucm = errors @ UCM_basis
    proj_ort = errors @ ORT_basis

    V_ucm = np.sum(proj_ucm ** 2) / (N * d_ucm)
    V_ort = np.sum(proj_ort ** 2) / (N * d_ort)

    return V_ucm / V_ort if V_ort > 0 else float('inf')
```

### 실행

Phase 1과 Phase 3 **모두에 대해** surrogate test를 수행한다.

```python
# Phase 1: VLA-Expert error
for tv_name, J in [("TV1", J_tv1), ("TV2", J_tv2), ("TV3", J_tv3)]:
    actual, surrogates, p, pct = surrogate_test(phase1_errors, J, n_permutations=1000)
    print(f"Phase1 {tv_name}: actual={actual:.2f}, surrogate_mean={surrogates.mean():.2f}, "
          f"surrogate_95th={np.percentile(surrogates, 95):.2f}, p={p:.4f}, percentile={pct:.1f}%")

# Phase 3: Multi-sample deviations (per-timestep)
# Phase 3은 timestep별로 K=30 samples의 deviations가 있으므로,
# 모든 timestep의 deviations를 concatenate하여 surrogate test
for tv_name, J in [("TV1", J_tv1), ("TV2", J_tv2), ("TV3", J_tv3)]:
    actual, surrogates, p, pct = surrogate_test(phase3_all_deviations, J, n_permutations=1000)
    print(f"Phase3 {tv_name}: actual={actual:.2f}, surrogate_mean={surrogates.mean():.2f}, "
          f"surrogate_95th={np.percentile(surrogates, 95):.2f}, p={p:.4f}, percentile={pct:.1f}%")
```

### 해석 기준

- **p < 0.01 (actual > surrogate 99%)**: 방향적 구조가 확인됨. ratio는 trivial하지 않다.
- **p > 0.05**: ratio가 dimension-wise variance 차이로 설명 가능. Finding이 artifact일 가능성 높음.
- TV3에 대해 p < 0.01이 나오지 않으면: **Phase 1/3의 TV3 결과는 논문에 쓸 수 없다.**

### 시각화

```
[Histogram]
- X축: surrogate ratio 분포
- Y축: frequency
- 빨간 수직선: actual ratio
- 제목에 p-value 표시
- Phase 1, Phase 3 × TV1, TV2, TV3 = 6개 subplot
- 파일: results/phase4/surrogate_test_histograms.png
```

### 예상 소요 시간

- Phase 1: 1000 permutations × 3 TVs = ~5분 (N이 크지만 UCM decomposition은 O(N×D²))
- Phase 3: concatenated deviations가 더 크므로 ~15분
- 총: ~20분

---

## 실험 2: TV2 Orientation Axis별 분해

### 목적

Phase 1에서 TV2 ratio=0.41 (inverse UCM), Phase 3에서 1.54. 이 dissociation의 원인을 orientation 3축(roll, pitch, yaw) 수준에서 규명한다.

**핵심 질문**: orientation error의 task-relevant bias가 roll/pitch/yaw 중 어디에 집중되어 있는가?

### 방법

TV2를 position(3D) + orientation(3D)로 나누지 말고, **orientation만의 sub-task variable**을 새로 정의한다.

```python
def build_jacobian_orientation_only(n_steps=8, action_dim=7):
    """
    Task variable: cumulative orientation change [Σ roll, Σ pitch, Σ yaw] (3D)
    Action space: 48D (6D × 8 steps, gripper 제외)
    UCM: 45D, ORT: 3D
    """
    D = n_steps * (action_dim - 1)  # 48 (gripper 제외)
    J = np.zeros((3, D))
    for step in range(n_steps):
        base = step * 6  # 6D per step (x,y,z,roll,pitch,yaw)
        J[0, base + 3] = 1.0  # roll
        J[1, base + 4] = 1.0  # pitch
        J[2, base + 5] = 1.0  # yaw
    return J


def build_jacobian_single_axis(axis_idx, n_steps=8, action_dim=7):
    """
    Task variable: cumulative change along single orientation axis (1D)
    axis_idx: 3=roll, 4=pitch, 5=yaw (within 6D per-step action)

    이렇게 하면 각 axis가 개별적으로 얼마나 task-relevant한 error를 만드는지 볼 수 있다.
    UCM: 47D, ORT: 1D
    """
    D = n_steps * (action_dim - 1)  # 48
    J = np.zeros((1, D))
    for step in range(n_steps):
        base = step * 6
        J[0, base + (axis_idx - 1)] = 1.0  # axis_idx is 3,4,5 for roll,pitch,yaw; subtract 1 for 0-indexed within 6D
    return J
```

**주의**: single axis Jacobian (1D task variable → 47D UCM, 1D ORT)은 dimensionality 비대칭이 극단적이다. 이 경우 surrogate test가 특히 중요하므로, **실험 2는 실험 1의 surrogate test와 함께 실행**한다.

### 추가 분석: Per-axis Error 통계

UCM 분해와 별개로, 더 직접적인 분석:

```python
def orientation_error_analysis(phase1_errors, phase3_deviations, n_steps=8):
    """
    각 orientation axis별로:
    1. Phase 1 error의 mean (bias) 과 std (noise)
    2. Phase 3 variability의 std
    3. Bias-to-noise ratio = |mean_error| / std_variability

    Bias-to-noise ratio가 높은 axis = systematic bias가 dominant
    Bias-to-noise ratio가 낮은 axis = stochastic noise가 dominant
    """
    results = {}
    for axis_name, axis_offset in [("roll", 3), ("pitch", 4), ("yaw", 5)]:
        # 48D에서 해당 axis의 indices (각 step마다)
        indices = [step * 6 + axis_offset for step in range(n_steps)]

        # Phase 1: error (VLA - Expert)
        axis_errors = phase1_errors[:, indices]  # (N, 8)
        mean_error = np.mean(axis_errors, axis=0)  # (8,) per-step mean bias
        std_error = np.std(axis_errors, axis=0)    # (8,) per-step noise

        # Phase 3: variability (deviations from VLA mean)
        axis_var = phase3_deviations[:, indices]  # (M, 8)
        std_var = np.std(axis_var, axis=0)         # (8,) per-step internal variability

        # Bias-to-noise ratio
        bias_noise_ratio = np.abs(mean_error) / (std_var + 1e-10)

        results[axis_name] = {
            "mean_bias_per_step": mean_error,
            "error_std_per_step": std_error,
            "variability_std_per_step": std_var,
            "bias_noise_ratio_per_step": bias_noise_ratio,
            "overall_bias_noise_ratio": np.mean(bias_noise_ratio),
        }

    return results
```

### 해석 기준

- **Bias-to-noise ratio >> 1인 axis**: VLA가 이 axis에서 consistently wrong direction으로 예측. Correction이 필요하고 가능하다 (systematic이니까).
- **Bias-to-noise ratio ≈ 1인 axis**: bias와 noise가 비슷. Correction 효과가 제한적.
- **BridgeData V2 사전 지식**: tabletop manipulation에서는 yaw가 dominant할 가능성 높음 (gripper의 회전이 grasp 방향을 결정). Roll/pitch는 table plane에 대해 비교적 constrained.

### 시각화

```
[Figure 1: Per-axis bias-noise decomposition]
- 3×1 subplot (roll, pitch, yaw)
- 각 subplot: X축=chunk step (0-7), Y축=magnitude
- 3개 선: |mean_bias| (빨강), error_std (파랑), variability_std (초록)
- 파일: results/phase4/orientation_axis_decomposition.png

[Figure 2: Cumulative bias direction]
- 3D arrow plot: [Σ mean_roll_error, Σ mean_pitch_error, Σ mean_yaw_error]
- Phase별 (approach, pre_grasp, grasp, transport, place) 다른 색
- 파일: results/phase4/orientation_bias_direction.png
```

---

## 실험 3: PCA-UCM Alignment Chance-Level Baseline

### 목적

Phase 3의 PCA-UCM alignment (모든 PC > 0.94)가 trivial한지 검증한다.

### 방법

48D 공간에서 UCM이 d_ucm차원일 때, random unit vector의 expected UCM alignment:

```python
def compute_chance_alignment(D, d_ucm, n_random=10000, seed=42):
    """
    D-dimensional 공간에서 d_ucm-dimensional subspace에 대한
    random vector의 expected alignment.

    Analytical: E[alignment] = d_ucm / D
    하지만 distribution도 필요하므로 Monte Carlo로 계산.

    alignment = ||proj_UCM(v)||² / ||v||²
    """
    rng = np.random.RandomState(seed)

    # Random UCM basis (실제 UCM basis와 같은 차원)
    # 실제로는 J로부터 계산한 UCM basis를 사용해야 함

    alignments = np.zeros(n_random)
    for i in range(n_random):
        v = rng.randn(D)
        v = v / np.linalg.norm(v)
        # UCM_basis는 실제 Jacobian에서 계산한 것을 사용
        proj = UCM_basis @ (UCM_basis.T @ v)
        alignments[i] = np.dot(proj, proj)  # ||proj||² (v is unit vector이므로 분모=1)

    return {
        "analytical_mean": d_ucm / D,
        "empirical_mean": np.mean(alignments),
        "empirical_std": np.std(alignments),
        "ci_95": (np.percentile(alignments, 2.5), np.percentile(alignments, 97.5)),
    }
```

### 각 TV에 대한 chance-level

| TV | D | d_ucm | d_ort | Chance alignment | 관측값 (PC0) |
|---|---|---|---|---|---|
| TV1 | 48 | 45 | 3 | 45/48 = **0.9375** | 0.997 |
| TV2 | 48 | 42 | 6 | 42/48 = **0.875** | 0.945 |
| TV3 | 48 | 24 | 24 | 24/48 = **0.500** | 0.980 |

**사전 판단:**
- TV1: 관측값 0.997 vs chance 0.9375 — margin이 있지만 좁다. Monte Carlo CI를 봐야 판단 가능.
- TV2: 관측값 0.945 vs chance 0.875 — moderate margin.
- TV3: 관측값 0.980 vs chance 0.500 — **압도적**. 이건 확실히 non-trivial.

### 시각화

```
[Figure: PCA-UCM alignment with chance baseline]
- X축: PC index (0-9)
- Y축: UCM alignment
- 실선: 실제 alignment (Phase 3)
- 점선: chance level (d_ucm/D)
- shaded: chance level의 95% CI (Monte Carlo)
- 3개 subplot: TV1, TV2, TV3
- 파일: results/phase4/pca_ucm_alignment_with_baseline.png
```

---

## 실험 4: Expert Action Magnitude vs Variability 관계

### 목적

Phase 3에서 Expert Direction Alignment ratio=3.33은 "VLA가 direction보다 magnitude에 uncertain하다"는 해석을 지지한다. 이게 expert action의 크기와 상관이 있는지 확인한다.

### 방법

```python
def magnitude_vs_variability(phase3_data):
    """
    각 timestep에서:
    - expert_action_norm: expert action chunk의 L2 norm
    - total_variance: K samples의 total variance
    - direction_variance: expert 방향 projection의 variance
    - magnitude_ratio: direction_variance / orthogonal_variance
    - ucm_ratio: 기존 UCM ratio

    이들의 correlation을 분석한다.
    """
    results = []
    for timestep_data in phase3_data:
        samples = timestep_data["samples"]  # (K, 48)
        expert = timestep_data["expert"]    # (48,)

        expert_norm = np.linalg.norm(expert)
        expert_dir = expert / (expert_norm + 1e-10)

        deviations = samples - samples.mean(axis=0)
        total_var = np.sum(np.var(samples, axis=0))

        # Expert 방향의 variance
        proj_expert = deviations @ expert_dir
        var_along = np.var(proj_expert)
        var_ortho = (total_var - var_along) / (47)  # 나머지 47차원에 평균

        results.append({
            "expert_norm": expert_norm,
            "total_var": total_var,
            "var_along_expert": var_along,
            "var_ortho_expert": var_ortho,
            "direction_ratio": var_along / (var_ortho + 1e-10),
        })

    df = pd.DataFrame(results)

    # Correlation
    corr_norm_var = df["expert_norm"].corr(df["total_var"])
    corr_norm_ratio = df["expert_norm"].corr(df["direction_ratio"])

    return df, corr_norm_var, corr_norm_ratio
```

### 해석 기준

- **expert_norm과 total_var가 양의 상관**: action이 클수록 VLA가 더 uncertain. Flow matching의 denoising에서 large action이 더 어렵다는 증거.
- **expert_norm과 direction_ratio가 양의 상관**: 큰 action일수록 magnitude uncertainty가 상대적으로 더 커진다. Conservative bias의 메커니즘을 설명.
- **상관없음**: magnitude uncertainty가 action 크기와 무관. 다른 설명이 필요.

### 시각화

```
[Figure 1: Scatter]
- X축: expert action norm
- Y축: total variance
- 색: task phase (approach=파랑, grasp=빨강, transport=초록, place=보라)
- regression line
- r, p-value 표시
- 파일: results/phase4/magnitude_vs_variance_scatter.png

[Figure 2: Binned analysis]
- Expert action norm을 10개 bin으로 나누기
- 각 bin의 mean direction_ratio + error bar
- 파일: results/phase4/magnitude_vs_direction_ratio.png
```

---

## 실행 계획

### 코드 구조

```
ucm_analysis/
├── (기존 Phase 0-3 코드들)
├── phase4_surrogate_test.py        # 실험 1
├── phase4_orientation_decomp.py    # 실험 2
├── phase4_pca_baseline.py          # 실험 3
├── phase4_magnitude_analysis.py    # 실험 4
├── phase4_run_all.py               # 모든 실험 순차 실행
└── results/phase4/
    ├── surrogate_test_histograms.png
    ├── surrogate_test_results.csv
    ├── orientation_axis_decomposition.png
    ├── orientation_bias_direction.png
    ├── orientation_axis_results.csv
    ├── pca_ucm_alignment_with_baseline.png
    ├── pca_baseline_results.csv
    ├── magnitude_vs_variance_scatter.png
    ├── magnitude_vs_direction_ratio.png
    └── magnitude_analysis_results.csv
```

### 실행 순서 및 의존성

```
실험 1 (Surrogate) ──────────────────────────────────────┐
실험 3 (PCA baseline) ──── 동시 실행 가능 ────────────────┤
실험 4 (Magnitude) ──── 동시 실행 가능 ───────────────────┤
                                                          ▼
                                               결과 종합 판단
                                                          │
                                         ┌────────────────┴────────────────┐
                                         ▼                                 ▼
                              Surrogate PASS                    Surrogate FAIL
                                         │                                 │
                                         ▼                                 │
                              실험 2 (Orientation) 실행            Phase 1/3의 ratio는
                                         │                       artifact. TV2 dissociation과
                                         ▼                       phase-dependent pattern만으로
                                    논문 narrative 확정           논문 구성 가능한지 판단
```

### 데이터 의존성

- **실험 1, 3, 4**: Phase 1의 error 데이터 + Phase 3의 multi-sample 데이터 필요
  - Phase 1: 기존 `results/phase1/` 에서 error 행렬 로드
  - Phase 3: 기존 `results/phase3/` 또는 multi-sample raw data에서 로드
- **실험 2**: Phase 1 error + Phase 3 deviations + phase segmentation 정보

### 예상 소요 시간

| 실험 | 계산 시간 | 비고 |
|---|---|---|
| 실험 1 (Surrogate) | ~20-30분 | 1000 permutations × 6 조건 (2 phases × 3 TVs) |
| 실험 2 (Orientation) | ~5분 | axis별 통계 + 소규모 UCM decomposition |
| 실험 3 (PCA baseline) | ~5분 | Monte Carlo 10,000 random vectors |
| 실험 4 (Magnitude) | ~5분 | Phase 3 데이터 재분석 |
| **총계** | **~35-45분** | 실험 1,3,4 병렬 시 ~30분 |

---

## 결과에 따른 분기 시나리오

### 시나리오 A: Surrogate test 통과 (p < 0.01 for TV3)

Phase 1/3의 UCM ratio가 non-trivial함이 확인됨. 논문의 main claim을 그대로 유지:

> "VLA errors exhibit structured variability aligned with task-irrelevant manifolds, not merely reflecting dimension-wise variance differences."

이 경우 **실험 2의 orientation 분해**가 paper narrative의 핵심이 됨:
- Position: UCM-structured (functional variability) → 교정 불필요
- Orientation: systematic bias (not noise) → task-relevant correction 필요
- 이 비대칭이 UCM-aware correction의 design principle

### 시나리오 B: Surrogate test 실패 (p > 0.05 for TV3)

TV3 ratio=20은 artifact. 하지만 **phase-dependent variation은 여전히 valid** (같은 dimension-wise variance에서 phase에 따라 ratio가 다르므로). 논문을 재구성:

> "While absolute UCM ratios are partially explained by dimension-wise variance structure, the phase-dependent modulation of error directionality reveals that VLA prediction errors are contextually structured."

이 경우 Phase 1의 phase별 패턴 + TV2 dissociation이 main finding이 되고, 절대적 ratio 값은 de-emphasize.

### 시나리오 C: TV2 orientation 분해에서 yaw가 dominant

BridgeData V2 tabletop task의 특성상 yaw가 grasp 방향을 결정하므로, yaw bias가 크면:

> "VLA systematically fails to predict grasp-critical rotation (yaw), while non-critical rotation axes (roll, pitch) show lower bias — demonstrating that the failure mode is task-specific, not a generic orientation problem."

이건 correction 방법 설계에 직접적 시사점: **yaw만 선택적으로 correction하는 것이 최적.**

---

## 최종 산출물 체크리스트

Phase 4 완료 후 확보해야 하는 것:

- [ ] Surrogate test 결과 (6개 조건): actual ratio, surrogate mean, p-value, histogram
- [ ] Orientation axis별 bias-noise ratio (roll, pitch, yaw × 8 steps)
- [ ] PCA-UCM alignment with chance baseline (3 TVs × 10 PCs)
- [ ] Expert magnitude vs variability correlation (scatter + binned)
- [ ] 논문 narrative 최종 결정 (시나리오 A/B/C 중 어디에 해당하는지)
- [ ] Phase 5 (UCM-aware correction 설계) 진행 여부 결정

---

## 참고: 기존 데이터 경로

- BridgeData V2 원본: `/mnt/md1/solee/data/bridge_lerobot`
- VLA actions: `/mnt/md1/solee/features/vla_actions`
- Phase 1 결과: `ucm_analysis/results/phase1/`
- Phase 3 결과: `ucm_analysis/results/phase3/`
- 기존 UCM 분석 코드: `ucm_analysis/` 디렉토리 전체

---
*Phase 4 Validation Experiments — UCM Analysis of VLA Action Prediction*