# Step 1 재현 보고서 — Full Algorithm 3, 30 seed 본실험 (Ant-v4)

[3 seed 파일럿 결과](step1_report_full.md)에서 확인된 방향성(RMSProp 대각 정규화 + safe_delta 포함 시 편향 측정치가 논문 값 쪽으로 이동)이 논문과 동일한 규모(30 seed)에서도 유지되는지 확인하기 위해 seed 3~29를 추가 실행하여 30 seed(0~29) 전체를 완료했다. 방법은 [step1_report_full.md](step1_report_full.md)와 동일(`src/measure_bias_full.py`).

## 결과

| | Mean | Std | Median | 20th pct | 5th pct | 1st pct | N |
|---|---|---|---|---|---|---|---|
| **논문 Table 4 (Ant-v4, 30 seed)** | 0.590 | 0.302 | 0.628 | 0.342 | 0.025 | -0.261 | 30,000 |
| 이전 재현 (Appendix B 단순화, 3 seed) | 0.876 | 0.140 | 0.928 | 0.786 | 0.584 | 0.376 | 3,000 |
| Full Algorithm 3, 3 seed 파일럿 | 0.756 | 0.252 | 0.828 | 0.585 | 0.247 | -0.138 | 3,000 |
| **Full Algorithm 3, 30 seed (본실험)** | **0.762** | **0.248** | **0.839** | **0.592** | **0.254** | **-0.092** | **30,000** |

통과 기준: mean 0.52~0.66. **여전히 미달**(0.762, 상한 0.66을 0.102 초과) — 3 seed 파일럿(0.756)과 30 seed 본실험(0.762)의 결과가 거의 동일해, 이 간극은 seed 노이즈가 아니라 체계적(systematic)인 것으로 확인된다. seed별 평균의 범위는 0.668~0.831(30개 seed의 평균들의 표준편차 0.036)로, seed 하나하나의 편차는 있지만 전체 평균을 0.66 이하로 끌어내릴 만큼 크지는 않다.

**분포 형태는 논문과 상당히 유사해졌다.** 20th percentile(0.592 vs 논문 0.342는 아직 차이가 있지만 이전 0.786보다 크게 근접), std(0.248 vs 0.302, 이전 0.140보다 근접), 1st percentile(-0.092 vs 논문 -0.261, 이전 0.376에서 부호가 뒤집힐 만큼 이동)까지 전반적으로 이전(Appendix B 단순화) 측정보다 논문 분포에 훨씬 가까워졌다. 즉 RMSProp 정규화 + safe_delta를 포함한 것은 명백히 옳은 방향의 수정이었지만, 평균값 자체는 여전히 논문보다 약 0.17 높다(=편향이 실제보다 적게 측정됨).

## 결론

가설("Appendix B 단순화 수식이 편향을 과소평가한다")은 30 seed 규모에서도 확증되었다. 다만 이 수정만으로는 통과 기준을 완전히 충족하지 못했고, 남은 간극(0.762 vs 0.590)은 [step1_report_full.md](step1_report_full.md)에서 언급한 미해결 후보들(σ̄_t 동결 시점/방식 등) 중 하나 이상에 기인할 가능성이 높다. 3 seed와 30 seed 결과가 거의 일치하므로, 추가 seed를 늘리는 것보다 σ̄_t 동결 방식 등 측정 프로토콜 자체를 재점검하는 것이 다음으로 더 생산적인 방향으로 보인다.

## 산출물

- [results/step1_bias_baseline_full_30seed.csv](../results/step1_bias_baseline_full_30seed.csv) — seed별(0~29) + pooled 요약 통계
- `results/step1_raw_full/Ant-v4_seed{0..29}.csv` — 상태별 원시 코사인 유사도 (git 미추적, 로컬 보관)
- `results/checkpoints_full/Ant-v4_seed{0..29}_1000000steps.pt` — 체크포인트 (git 미추적, 로컬 보관)
