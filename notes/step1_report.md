# Step 1 재현 보고서 — 파일럿 (Ant-v4, 3 seed)

## 결과 요약

| Environment | Mean | Std | Median | 20th pct | 5th pct | 1st pct | 비고 |
|---|---|---|---|---|---|---|---|
| **논문 Table 4 (Ant-v4, 30 seed)** | 0.590 | 0.302 | 0.628 | 0.342 | 0.025 | -0.261 | 목표값 |
| **본 재현 (pooled, 3 seed, N=3000)** | **0.876** | 0.140 | 0.928 | 0.786 | 0.584 | 0.376 | 실측값 |
| seed 0 (N=1000) | 0.885 | 0.130 | 0.935 | 0.804 | 0.624 | 0.397 | |
| seed 1 (N=1000) | 0.870 | 0.146 | 0.923 | 0.764 | 0.581 | 0.366 | |
| seed 2 (N=1000) | 0.873 | 0.144 | 0.929 | 0.781 | 0.560 | 0.375 | |

**통과 기준: mean 0.52~0.66 → 미달 (0.876, 통과 기준 상한을 0.22 이상 초과).** 3개 seed 모두 좁은 범위(0.870~0.885)에 몰려 있어, 이는 seed 노이즈가 아니라 측정 공식/프로토콜 자체가 논문과 다르다는 강한 신호다.

원본 코드(`baseline/optimizer.py`)와 지시서 수식이 정확히 일치함을 Step 0에서 이미 확인했으므로, 학습 알고리즘 자체의 구현 오류 가능성은 낮다고 보고, 아래 순서(체크포인트 시점 → λ 전환 → 샘플 수)로 점검했다.

## 점검 1: 체크포인트 시점

- 정확히 1,000,000 환경 스텝까지 학습 후 그 시점의 네트워크 가중치·정규화 통계(`obs_stats.mean/var`, `reward_stats.var`)를 스냅샷하여 사용. 논문의 "1M environment step mark, mid-training rising phase"와 동일한 지점.
- 학습 스크립트는 원본 `intentional_ac.py`의 로직을 그대로 재사용(`Actor`/`Critic`/`IntentionalAC`/옵티마이저를 import, 재구현 아님). λ=0.8, γ=0.99, η_actor=0.05, η_critic=0.5, entropy_coeff=0.01(원본 기본값) 사용.
- **판정: 문제 없음.** 스텝 수 자체는 사양과 정확히 일치.

## 점검 2: λ 전환

- 측정 시 λ=0을 가정하고, 적격흔적을 아예 구현하지 않고 매 (s,a)마다 `g(s,a) = ∇_θ log π(a|s)`를 직접 사용했다. λ=0일 때 `e_t = γλ·e_{t-1} + g_t = g_t`이므로 이는 수학적으로 정확히 동일하다.
- **판정: 문제 없음** (적어도 적격흔적 부분에 한해서는).
- **단, 다음 사실을 발견**: `baseline/optimizer.py`에서 σ̄_t의 EMA 갱신은 `self.sigma += (1 - γλ)·(norm_grad - self.sigma)`로, γλ에 직접 묶여 있다. λ=0이면 (1-γλ)=1이 되어 **σ̄_t가 매 스텝 현재 샘플의 norm_grad로 완전히 치환**된다(스무딩 없음). 즉 "λ=0 측정 모드"에서 실제 알고리즘을 그대로 돌리면 σ̄_t도 순간값이 되어버리는데, 이것이 논문이 의도한 그대로인지, 아니면 논문의 실험 코드는 σ̄_t/v_hat을 학습 종료 시점 값으로 **동결**해서 쓰는지가 원문에 명시되어 있지 않다. 이는 아래 "핵심 가설"과 연결된다.

## 점검 3: 샘플 수

- 상태 1,000개 × 상태당 행동 1,000개, 정확히 사양대로. 3 seed(파일럿) → 3,000 샘플. 논문은 30 seed → 30,000 샘플.
- 3개 seed의 결과가 0.870~0.885로 매우 좁게 몰려 있음 — 표준편차 대비 seed간 편차가 작아 통계적 노이즈로는 설명되지 않는 편향된(systematic) 차이로 판단된다.
- **판정: 샘플 수 부족이 원인일 가능성은 낮음.**

## 핵심 가설: RMSProp 대각 정규화(ρ_t) 누락

측정에 사용한 정확한 수식은 **Appendix B**(`app:bias`)의 예시 수식을 그대로 구현한 것이다:

```
v_unbiased(s)    = E_a[ A(s,a) · g(s,a) ]
v_intentional(s) = E_a[ A(s,a)/‖g(s,a)‖² · g(s,a) ]
```

이 수식에는 RMSProp류 항목별 대각 정규화 ρ_t(=1/v_hat)가 없다. 그런데 논문 본문 Section 4(`sec:policy`) 말미, Appendix B를 인용하기 직전 문단("Traces and diagonal normalization")은 다음과 같이 말한다:

> "As in Section 3, we use eligibility traces and RMSProp-style diagonal normalization in our streaming implementations... **these mechanisms modify the update direction** (e.g., replacing g_t by a trace z_t and applying an entry-wise preconditioner ρ_t), **and the intentional update then chooses α_t using the same principle applied to the resulting direction.**"

즉 실제 Algorithm 3(및 `optimizer.py`)의 정규화 분모는 `‖g‖²`가 아니라 `⟨ρ_t g_t, g_t⟩`이고, 업데이트 방향 자체도 `g`가 아니라 `g/v_hat`이다. Appendix B의 `‖g(s,a)‖²` 수식은 편향의 **메커니즘을 설명하기 위한 단순화된 예시**일 가능성이 높고, Table 4가 실제로 측정한 것은 (그렇게 명시되어 있진 않지만) 학습된 실제 모델의 **진짜 σ_τ = ⟨g_τ,ρ_τg_τ⟩ 및 e/v_hat 메커니즘**일 가능성이 있다.

이 가설이 맞다면, 내가 뺀 ρ_t(각 파라미터 좌표별로 다른, 학습 이력에 의존하는 가중치)가 실제로는 편향을 **더 키우는 방향**으로 작용해야 논문 수치(0.59)에 가까워진다 — 지금 내 측정치(0.876)가 논문보다 훨씬 "편향이 적은" 쪽으로 나온 것과 방향이 일치한다.

부차적으로 확인이 필요한 것: `safe_delta`(적응형 클리핑 + 정규화 EMA)도 배제했는데, 이 역시 Appendix B의 단순화된 수식에는 없는 요소다. 이것까지 포함해야 하는지는 불확실하다.

## 다음 단계 (사용자 확인 대기)

지시서 규칙("코드 수정 전에 원인부터 보고")에 따라, 아직 코드를 수정하지 않고 여기서 멈춘다. 다음 시도로 제안하는 것:

1. `baseline/optimizer.py`의 `IntentionalOptimizerPolicy`를 실제로 재사용하여, 체크포인트 시점의 `rmsprop_v_hat`, `sigma`, `t_step` 등 옵티마이저 내부 상태를 함께 저장·동결한 뒤, 각 (s,a) 샘플에 대해 **진짜 σ_τ = ⟨g_τ,ρ_τg_τ⟩**를 사용해 "unbiased(ρ 포함, α=1)" vs "intentional(실제 α_t)"의 코사인 유사도를 재측정.
2. `safe_delta`(클리핑+정규화) 포함 여부는 별도로 ablation하여 어느 조합이 논문 수치(0.52~0.66)에 부합하는지 확인.

이 재구현에는 옵티마이저 내부 상태를 동결하는 방식(측정 중 계속 갱신할지 여부)에 대한 추가 설계 결정이 필요하고, 파일럿 재실행에 다시 seed당 약 1시간이 소요된다. 진행 여부를 사용자에게 확인 후 착수한다.

## 산출물

- [results/step1_bias_baseline.csv](../results/step1_bias_baseline.csv) — seed별 요약 통계
- `results/step1_raw_pilot/Ant-v4_seed{0,1,2}.csv` — 상태별 원시 코사인 유사도 (git 미추적, 로컬 보관)
- `results/checkpoints/Ant-v4_seed{0,1,2}_1000000steps.pt` — 체크포인트 (git 미추적, 재현용 로컬 보관)
