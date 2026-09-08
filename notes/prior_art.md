# 선점 확인 보고서 (Step 0)

## 검색 범위

본 조사는 아래 4개 버킷에 대해 병렬로 선행연구 검색을 수행했다.

1. **fisher_trace_stepsize** — Fisher 정보량의 trace(또는 E_a[‖g(a\|s)‖²])를 스텝사이즈의 분모로 사용한 선행연구 탐색
2. **action_independent_stepsize_bias** — 정책 경사법에서 "행동 독립적(action-independent), 상태 종속적(state-dependent)" 스텝사이즈를 명시적으로 다룬 선행연구 탐색 (자연 정책 경사, natural actor-critic 계열 포함)
3. **nlms_policy_gradient** — NLMS(Normalized LMS) 스타일의 정규화 스텝사이즈를 스트리밍/적격 흔적(eligibility trace) 기반 액터-크리틱에 적용한 선행연구 탐색 (Mahmood/Sutton 계열의 IDBD-AutoStep-TIDBD-Metatrace 포함)
4. **muon_modular_norm_lamb** — Muon, Modular Norm, LAMB 등 "구조적으로 의미 있는 단위로 정규화"하는 최신 최적화기 계열이 동일 아이디어를 이미 구현했는지 탐색

이와 별도로, 대상 논문(arXiv:2604.19033)의 원문 **Appendix A(Fisher/KL 유도), Appendix B(2-행동 반례), Section 9(공개 문제)**를 직접 재확인했다.

## 유사 연구 목록

- **Limitations of the Empirical Fisher Approximation for Natural Gradient Descent** (Kunstner, Balles, Hennig; NeurIPS 2019) — *같음*: "표본화된 결과(레이블/행동)의 그래디언트 외적"을 "모델 자신의 예측 분포에 대한 기댓값(true Fisher)"으로 대체해야 한다는 것과 정확히 동일한 구조적 편향을 지적함 / *다름*: 지도학습 분류 문제이며, 전체(또는 K-FAC 근사) Fisher 행렬을 프리컨디셔너로 다룰 뿐 스칼라 trace를 스텝사이즈 분모로 쓰지 않고, 스트리밍/RL 맥락도 Hutchinson 단일-역전파 추정기도 없음 / *우리 기여가 남는 지점*: 이 진단을 스트리밍 정책 경사법의 스텝사이즈 분모 문제로, 그리고 구체적 단일-역전파 추정기로 처음 구현.

- **A Natural Policy Gradient / Natural Actor-Critic** (Kakade, 2001; Peters & Schaal, 2008; Bhatnagar, Sutton, Ghavamzadeh, Lee, 2009 — 후자는 Sutton 본인이 공저) — *같음*: F(θ;s)=E_{a~π}[g(a\|s)g(a\|s)ᵀ]가 정확히 행동에 대해 기댓값을 취한, 행동 독립적 객체이며 이를 정책 경사에 사용하고, Bhatnagar 등은 완전한 스트리밍/온라인 설정임 / *다름*: 전체 Fisher 행렬(또는 그 역행렬)을 업데이트 "방향"을 재구성하는 프리컨디셔너로 쓰지, 스칼라 trace를 스텝사이즈 "크기"의 분모로 쓰지 않음. 호환 특성(compatible features) 기반 추정이며 Hutchinson류 단일-역전파 trace 추정기가 아니고, RMSProp 스타일 rho_t나 σ_t 정식화와 무관함 / *우리 기여가 남는 지점*: trace만을 스칼라 정규화 상수로 분리해 쓰는 최소 개입, 그리고 output-space 공분산 샘플링을 통한 단일-역전파 추정기는 이 계열에 없음.

- **Trust Region Policy Optimization** (Schulman, Levine, Moritz, Jordan, Abbeel; ICML 2015) — *같음*: Pearlmutter R-operator 스타일의 단일 역전파 트릭으로 Fisher-vector product를 저비용으로 얻어 행동-독립적 스텝사이즈/신뢰영역을 구성한다는 계산 전략의 정신이 같음 / *다름*: 전체 g^T F^{-1} g 이차형식(CG 반복 다회)을 쓰지 trace만 쓰지 않고, KL의 해석적 적분성에 의존하지 E_a[...] 명시적 추정이 아니며, 대규모 배치·신뢰영역 방법으로 스트리밍/배치크기-1/적격흔적 설정이 아님 / *우리 기여가 남는 지점*: 단일 샘플, 단일 역전파, output-space Hutchinson 추정으로 tr F(θ;s)를 직접 얻어 σ_t를 대체하는 구체적 메커니즘.

- **The Optimal Reward Baseline for Gradient-Based Reinforcement Learning** (Weaver & Tao; UAI 2001) — *같음*: E[‖∇log π(a\|s)‖²]와 정확히 동일한 양(궤적 평균 형태)을 정책 경사 관련 정규화 상수로 이미 사용함 / *다름*: 전역 스칼라 가산적 보상 베이스라인이지, 상태별 곱셈적 스텝사이즈 분모가 아니며, 상태 조건부도 아니고 적격흔적/스트리밍/단일-역전파 추정기와 무관함 / *우리 기여가 남는 지점*: 동일 수학적 양을 완전히 다른 위치(스텝사이즈 분모, 상태 조건부)와 완전히 다른 추정 메커니즘(Hutchinson 단일 역전파)으로 재도입.

- **Randomized Advantage Transformation (RAT)** 및 **Rank-1 Approximation of Inverse Fisher for NPG** (2026, arXiv:2605.18591 / arXiv:2601.18626) — *같음*: 대상 논문과 동일 시기(2026)에, Fisher 기반 자연 경사 효과를 명시적 역행렬 계산 없이 역전파 트릭으로 저비용 획득하려는 동일한 최신 문제의식 / *다름*: 미니배치 PPO류 루프에서 전체 업데이트 "방향"(Woodbury 항등식 반복 풀이, 또는 랭크-1 프리컨디셔너)을 재구성하는 것이지, 스트리밍 단일 샘플의 스칼라 스텝사이즈 분모 문제가 아니며 Hutchinson 단일-역전파 trace 추정기도 아님 / *우리 기여가 남는 지점*: 미니배치가 아닌 배치크기-1 스트리밍 설정, 방향이 아닌 크기(스칼라 분모) 문제, 그리고 훨씬 더 가벼운 output-space 공분산 샘플링 추정기.

- **ACKTR** (Wu, Mansimov, Liao, Grosse, Ba; NeurIPS 2017), **RLS-A2C/RLSNA2C** (Wang et al., 2022), **AutoStep/TIDBD/Metatrace** (Mahmood, Sutton 계열) — 참고용 인접 계열로 함께 확인. 모두 "행동 독립적 정규화" 또는 "적격흔적 기반 온라인 스텝사이즈 적응"이라는 문제의식은 공유하지만, Fisher trace를 스텝사이즈 분모로 쓰거나 Hutchinson 단일-역전파 추정기를 쓰는 조합은 없음. 특히 "NLMS + policy gradient" 검색은 대상 논문 자신(arXiv:2604.19033)을 계속 재검색 결과로 노출시켰다 — 이 조합이 최신/신규임을 방증.

## 논문 원문 확인

arXiv:2604.19033 원문(HTML 미러, 반복 교차확인)을 직접 재확인한 결과는 다음과 같다.

- **Appendix A**: Fisher 정보행렬 F(θ_t;s) := E_{a~π_θt(·\|s)}[g(a\|s)g(a\|s)ᵀ]가 이미 논문 자체 표기법으로 명시적으로 정의되어 있으며, 이는 KL(π_θt(·\|s) ‖ π_θt+1(·\|s)) ≈ (1/2)E[(Δ log π(a\|s))²] 유도(2차 근사)에 사용된다. 그러나 **실제 알고리즘(Algorithm 3)의 스텝사이즈 분모 σ_τ(식 11)는 σ_τ := ⟨g_τ, ρ_τ g_τ⟩로, 단일 표본화된 행동 a_τ의 그래디언트만을 사용**한다. 즉 논문은 상태 조건부 기댓값(Fisher)을 이미 정의해 두고도, 계산/스트리밍 상 이유로 실제 알고리즘에서는 이를 단일 표본으로 대체한다 — 이것이 바로 편향의 근원이다. (실제 `baseline/optimizer.py`의 `sigma`/`z_sum`/`step_size` 계산이 이 식과 정확히 일치함을 코드 레벨에서도 별도 확인함.)

- **Appendix B**: 2-행동 반례를 명시적으로 제시한다. 원문 인용(수식): "A(s,a1)>A(s,a2)>0 이면서 A(s,a1)/‖g(s,a1)‖² < A(s,a2)/‖g(s,a2)‖²가 가능하다." 결론 인용(원문 발췌, 9단어): "the normalized update can favor a2 even though a1 is better." — 즉 더 나쁜 행동 a2가 더 큰 정규화 업데이트를 받을 수 있음을 저자들 스스로 증명한다.

- **Section 9**: 이 문제를 "action-dependence of policy normalization"이라는 이름으로 명시적 공개 문제(open issue)로 남겨둔다. 원문은 "action-dependent step-size rules can reweight actions inside the expectation"이라 설명하고, 바람직한 방향으로 "It would be preferable to obtain the same control over the scale of one-step policy change with an action-independent step size (state-dependent at most)"라고 명시한다 — 이는 과제 설명에서 제시된 "행동 독립적(action-independent), 상태 종속(state-dependent)" 해법 방향과 거의 동일한 표현이다.

(주의: 원문 확인은 WebFetch를 통한 HTML 미러 교차조회로 이루어졌으며, PDF 텍스트 레이어의 byte-level 검증은 하지 못했다. 서로 다른 각도의 독립적 질의에서 동일한 정확한 문구·수식·용어가 일관되게 재현되어 신뢰도는 높지만, 출판/인용에 그대로 쓸 경우 PDF 원문 재확인을 권장한다.)

## 판정

4개 버킷 전체에서, 동일한 제안(즉 (i) tr F(θ;s) = E_a[‖g(a\|s)‖²]를 스텝사이즈 σ_t의 분모로 명시적으로 대체하고, (ii) 이를 대상 논문 Appendix B의 행동-의존적 편향에 대한 해법으로 명시적으로 프레이밍하며, (iii) 배치크기-1·적격흔적·리플레이버퍼 없는 스트리밍 액터-크리틱에서 작동하고, (iv) output-space 공분산 F_out에서 v를 샘플링해 ⟨z,v⟩를 한 번 역전파하는 Hutchinson/R-operator식 단일-역전파 추정기로 계산하는) 선행연구는 발견되지 않았다.

가장 근접한 선행연구는 두 갈래로 나뉜다. 첫째, Fisher 정보량을 이용해 행동 독립적 정책 업데이트를 구성하는 계열(Kakade의 자연 정책 경사, Bhatnagar·Sutton의 Natural Actor-Critic, TRPO, ACKTR, 그리고 2026년의 RAT·rank-1 inverse-Fisher)로, 이들은 모두 전체 Fisher 행렬(또는 그 역행렬)을 업데이트 **방향**을 재구성하는 프리컨디셔너로 사용할 뿐, trace만을 분리해 스텝사이즈 **크기**의 분모로 쓰지 않으며, 우리가 제안하는 단일-역전파 output-space 샘플링 추정기도 사용하지 않는다. 둘째, 동일한 구조적 편향(표본화된 결과의 그래디언트 외적 vs. 진짜 기댓값)을 지도학습 맥락에서 진단한 Kunstner et al.(2019)과, 정확히 같은 수학적 양 E[‖g‖²]를 전혀 다른 용도(가산적 보상 베이스라인)로 사용한 Weaver & Tao(2001)가 있으나, 둘 다 RL 스텝사이즈 분모 문제나 스트리밍/적격흔적 설정, Hutchinson 추정기를 다루지 않는다.

더불어 대상 논문 자신의 Appendix A가 F(θ;s)를 이미 정의해 두고도 Algorithm 3에서는 계산 비용상의 이유로 단일 표본 σ_τ로 대체했고, Section 9에서 "action-independent (state-dependent at most)" 스텝사이즈가 바람직하다고 명시적으로 미해결 과제로 남겨둔 점은, 우리의 제안이 논문 저자들 스스로도 인지했지만 닫지 못한 간극을 정확히 메운다는 것을 뒷받침한다.

**진행 권고 (proceed).** Step 1로 진행한다.

---

## 부록: 환경/인프라 사전 점검 (Step 0 병행 작업)

- `baseline/` 에 공식 저장소(https://github.com/sharifnassab/Intentional_RL) 클론 완료 (git hash: 클론 시점 HEAD, `git -C baseline log --oneline -1` 참고).
- 로컬 환경: Windows 11, Python 3.12.10, GPU 없음(Intel UHD 730 내장 그래픽만, CPU 전용), CPU 13th Gen Intel i5-13400 (10 코어/16 스레드).
- `requirements.txt`는 Linux/macOS 전용으로 명시되어 있으나(`torch==2.7.1`, `mujoco==3.8.1`, `gymnasium[atari,mujoco]==1.2.2` 등), Python 3.12 + Windows 조합에서도 전체 설치가 예외 없이 성공했고(`.venv`), `Ant-v4` 환경 생성·스텝 실행까지 스모크 테스트 통과.
- 원본 `intentional_ac.py`로 Ant-v4 20,000 스텝 실행 시간을 측정한 결과 약 2.5ms/스텝. 1M 스텝 기준 seed당 약 40분(단일 프로세스, CPU 전용) 소요 예상. 10코어를 활용해 여러 seed를 프로세스 병렬로 돌리면 파일럿(3 seed) 규모는 벽시계 기준 약 40분 내외로 단축 가능.
