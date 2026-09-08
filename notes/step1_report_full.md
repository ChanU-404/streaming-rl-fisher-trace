# Step 1 재현 보고서 — Full Algorithm 3 재측정 (Ant-v4, 3 seed pilot)

이전 보고서([step1_report.md](step1_report.md))에서 제기한 핵심 가설 — Appendix B의 단순화된 수식(`‖g(s,a)‖²` 분모)이 아니라, 실제 Algorithm 3의 RMSProp 대각 정규화(ρ_t=1/v_hat), σ̄_t EMA, `safe_delta`(적응형 클리핑+정규화)를 모두 포함해야 논문 수치(0.590)에 가까워질 것이다 — 를 검증하기 위해 `src/measure_bias_full.py` + `src/run_step1_full.py`로 3 seed 파일럿을 재실행했다.

## 방법

체크포인트 시점(1,000,000 스텝)에 옵티마이저 내부 상태(`rmsprop_v_hat`, `sigma`, `t_step`, `clip_ema_sq`, `delta_abs_ema` 등)를 함께 동결 저장한 뒤, 각 (s, 1000개 행동) 샘플에 대해 실제 Algorithm 3 메커니즘을 그대로 재현:

```
g_i          = ∇_θ log π(a_i|s)
g_precond_i  = g_i / v_hat                         (동결된 RMSProp 정규화)
sigma_i      = <g_i, g_precond_i>
alpha_i      = eta_policy / sqrt(sigma_bar_frozen * sigma_i)
safe_delta_i = frozen_clip_normalize(TD_error_i)    (동결된 적응형 클리핑+정규화)

v_unbiased(s)    = mean_i[ safe_delta_i · 1       · g_precond_i ]
v_intentional(s) = mean_i[ safe_delta_i · alpha_i · g_precond_i ]
cos_sim(s)       = cosine(v_unbiased(s), v_intentional(s))
```

이전 버전과 달리 `‖g‖²` 대신 실측 σ_τ=⟨g,ρg⟩를 쓰고, `safe_delta`(클리핑+정규화)까지 포함한다. 나머지(체크포인트 시점, λ=0 등가성, 샘플 수)는 이전 점검에서 이미 "문제 없음"으로 판정된 부분이라 그대로 유지했다.

## 결과

| | Mean | Std | Median | 20th pct | 5th pct | 1st pct |
|---|---|---|---|---|---|---|
| **논문 Table 4 (Ant-v4, 30 seed)** | 0.590 | 0.302 | 0.628 | 0.342 | 0.025 | -0.261 |
| **이전 재현 (Appendix B 단순화, 3 seed)** | 0.876 | 0.140 | 0.928 | 0.786 | 0.584 | 0.376 |
| **본 재현 (Full Algorithm 3, pooled 3 seed, N=3000)** | **0.756** | 0.252 | 0.828 | 0.585 | 0.247 | -0.138 |
| seed 0 | 0.757 | 0.247 | 0.827 | 0.596 | 0.254 | -0.042 |
| seed 1 | 0.781 | 0.236 | 0.857 | 0.627 | 0.316 | -0.164 |
| seed 2 | 0.729 | 0.268 | 0.802 | 0.538 | 0.156 | -0.144 |

통과 기준: mean 0.52~0.66. **여전히 미달**(0.756, 상한을 0.096 초과)이지만, RMSProp 정규화 + safe_delta를 포함하자 편향 측정치가 목표 방향(0.590)으로 뚜렷하게 이동했다:

- 상한 초과폭: 0.876 - 0.66 = **0.216** → 0.756 - 0.66 = **0.096** (약 56% 감소)
- std, 5th/1st percentile도 논문 값(0.302, 0.025, -0.261)에 근접하는 방향으로 이동 (0.140→0.252, 0.584→0.247, 0.376→-0.138) — 이전 측정에서 3 seed가 좁은 범위에 몰려 있던 것(0.870~0.885)과 달리 상태별 변동폭이 커져 논문의 넓은 분포(std 0.302)에 더 가까워졌다.

이는 가설의 방향이 맞았음을 강하게 시사한다: ρ_t·safe_delta를 포함하지 않은 단순화 수식은 실제보다 편향을 과소평가(코사인 유사도 과대평가)하고 있었다.

## 남은 간극

0.756 vs 0.590, 아직 0.166 차이가 남아 있고 통과 기준(≤0.66)에도 못 미친다. 조사하지 않은 채 남겨둔 요소:

1. **σ̄_t 동결 방식**: 현재는 체크포인트 종료 시점의 EMA 값(bias-correction 포함)을 1000×1000 branching 전체에 고정 사용. 논문이 실제로 어떤 시점/방식의 σ̄_t를 Table 4 측정에 썼는지는 원문에 명시되어 있지 않음.
2. **30 seed vs 3 seed**: 파일럿 규모(3 seed)라 seed 노이즈 자체가 아직 크게 작용할 수 있음 — 다만 이전 파일럿과 달리 이번엔 seed간 분산(0.729~0.781)이 이미 커서, seed 수를 늘리면 평균이 더 낮아질지 높아질지는 예측하기 어려움.
3. 그 외 옵티마이저 세부 항목(예: 관측/보상 정규화 통계의 동결 시점, gradient clipping 등)의 추가 점검 여지.

## 다음 단계 (사용자 확인 대기)

지시서 규칙에 따라 코드를 더 수정하기 전에 여기서 멈춘다. 제안:

- (a) 이 결과를 "부분 확증"으로 보고 30 seed 본 실험으로 확대해 통계적으로 재확인, 또는
- (b) σ̄_t 동결 방식 등 남은 가설을 추가로 점검한 뒤 30 seed로 확대.

## 산출물

- [results/step1_bias_baseline_full.csv](../results/step1_bias_baseline_full.csv) — seed별 + pooled 요약 통계
- `results/step1_raw_full/Ant-v4_seed{0,1,2}.csv` — 상태별 원시 코사인 유사도 (git 미추적, 로컬 보관)
- `results/checkpoints_full/Ant-v4_seed{0,1,2}_1000000steps.pt` — 체크포인트, 옵티마이저 내부 상태 포함 (git 미추적, 로컬 보관)
