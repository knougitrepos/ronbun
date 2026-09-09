# 낮은 FPIR 목표 초과 원인 분석

작성일: 2026-09-09 KST. 분석 대상: step11, `ec9d7f2dafecc3b4f636ba2f8df29a4e78595e28`.

## 핵심 결론

**1% 조건은 단순한 fit/safety 분할 문제만으로 설명하기 어렵고, calibration→test 전이 문제가 두드러진다. 5% 조건은 분할 민감성과 전이 문제가 함께 존재한다.**

이 결론은 ArcFace × SurvFace × PQ m128 ADC의 고정 calibration cohort·gallery·codec에 대한 탐색적 진단이다. 인물 구성, 영상 조건, gallery 및 codec 일반화 중 어느 요인이 최종 원인인지는 아직 분리하지 못했다. 압축 전 원본 점수에서도 목표 초과가 있어 PQ/FIQA만의 문제로 볼 수 없다.

기존 노트북, 연구 Python 모듈, 완료 run, 이전 결과는 변경하지 않았다. 새 분석 스크립트와 그 결과만 이 디렉터리에 추가했다. 아래 권장 구현은 이번 작업에 포함하지 않았다.

## 1. 무엇을 비교했는가

- FPIR: non-mated query의 최고 검색 점수가 고정 threshold 이상인 비율.
- 목표 1%, 5%; 비교 기준 10%.
- calibration non-mated 98,825장 / 실제 identity 2,319명.
- test non-mated 121,736장. 실제 identity 라벨은 없으므로 121,736명을 의미하지 않는다.
- TPIR20: mated 60,423장 / 3,000명을 대상으로 genuine-score-topk-v2 정의 유지.
- calibration gallery 생성 seed 8972, frozen codec, 저장된 모든 검색 점수 고정.
- 내부 fit/safety seed: 0–17, 42, 8972의 20개. 결과를 보기 전에 목록을 고정했다.
- S/L 모두 같은 분할. fit/safety 70/30 identity 분할, FIQA 2 bins, shrinkage 200, 최소 그룹 non-mated 100 유지.

아래 최소–최대는 같은 데이터를 재분할한 기술통계다. 95% CI, 독립 실험 20회, 새로운 데이터셋의 성공 확률로 해석하면 안 된다. 좋은 seed를 선택하지 않았다.

## 2. 1%는 20개 내부 분할 모두 목표 초과

| 방법 | test FPIR 최소–최대 | 중앙값 | 목표 충족 분할 |
|---|---:|---:|---:|
| Global empirical | 1.480–1.480% | 1.480% | 0/20 |
| Global safe | 1.126–1.468% | 1.393% | 0/20 |
| FIQA-S | 1.098–1.452% | 1.383% | 0/20 |
| FIQA-L | 1.170–1.456% | 1.406% | 0/20 |

Global empirical은 전체 calibration을 이용하므로 내부 seed와 무관하다. 다른 세 방법은 분할에 따라 FPIR가 달라지지만, 현재 cohort에서 내부 fit/safety 분할을 바꾸는 것만으로는 확인한 20개 seed 중 목표를 충족한 경우가 없었다.

**이는 모든 가능한 seed/다른 gallery에서 목표 달성이 불가능하다는 뜻이 아니다.** 아래처럼 calibration의 사용 집단 자체를 바꾸면 드물게 목표가 충족되는 경우도 있다. 판정은 '분할 민감성이 없음'이 아니라 '단순한 내부 분할 문제만으로는 충분히 설명되지 않음'이다.

근거: [seed_ranges.csv](audit-v1/seed_ranges.csv), [seed_metrics.csv](audit-v1/seed_metrics.csv).

## 3. 5%는 분할에 따라 성공·실패가 바뀜

| 방법 | test FPIR 최소–최대 | 중앙값 | 목표 충족 분할 |
|---|---:|---:|---:|
| Global empirical | 5.315–5.315% | 5.315% | 0/20 |
| Global safe | 4.537–5.306% | 5.198% | 4/20 |
| FIQA-S | 4.623–5.325% | 5.210% | 4/20 |
| FIQA-L | 4.533–5.250% | 5.122% | 5/20 |

5%에서 분할 민감성은 실제로 관찰된다. 그러나 대부분의 분할 및 중앙값은 목표를 초과한다. '좋은 seed를 고르면 해결'이 아니라 분할 안정성과 운영 집단 전이를 동시에 검증해야 한다.

10% 조건에서는 네 방법 모두 20/20개 내부 분할에서 목표를 충족했다. 따라서 모든 operating point가 같은 방향으로 실패하는 단순한 전체 점수 이동으로 일반화하지 않는다.

## 4. 두 FIQA bin의 비중 차이만으로 설명되지 않음

FIQA-L, 기존 seed 8972, target 1%:

| 그룹 | safety 그룹 비중 | test 그룹 비중 | safety FPIR | test FPIR |
|---|---:|---:|---:|---:|
| Low | 47.9155% | 47.8560% | 0.9457% | 1.3955% |
| High | 52.0845% | 52.1440% | 0.9986% | 1.4115% |

비중은 거의 같지만 두 그룹 내부의 초과율이 모두 커졌다. 정확한 분해는 다음과 같다.

`R_test − R_safety = Σ(p_test − p_safety)·r_safety + Σp_test·(r_test − r_safety)`

- 전체 차이: +0.43057358%p.
- 그룹 구성 비중 변화: 약 +0.00003148%p.
- 그룹 내부 초과율 변화: 약 +0.43054210%p.

현재 2-bin 수준에서는 대부분 그룹 내부 변화다. 더 세밀한 품질/인물 구성 차이를 배제한 것은 아니다. 또한 safety 데이터는 최종 threshold를 정하는 데 쓰였으므로, safety 목표 충족 자체는 독립 검증이 아니다.

5%에서는 low 그룹의 test FPIR가 5.4585%, high는 4.9356%였다. 1%와 동일하게 두 그룹 모두 실패했다고 설명하면 안 된다.

근거: [group_transfer.csv](audit-v1/group_transfer.csv), [gap_decomposition.csv](audit-v1/gap_decomposition.csv).

## 5. 독립 threshold holdout과 인물 집중도 추가 검사

### 첫 holdout 검사

outer seed 314159로 calibration identity 약 20%를 threshold 적합에서 제외했다. 실제 non-mated holdout은 20,085장 / 499명이었다. 나머지 calibration에 대해서만 다시 20개 내부 fit/safety 분할을 평가했다.

이 holdout의 FPIR가 test보다 지속적으로 낮았다. 다만 첫 holdout 자체가 쉬운 구성일 수 있으므로 이 한 결과만으로 일반화하지 않았다. 이 조건의 1% test에서는 Global-safe/S/L이 각각 1/20개 내부 분할에서 목표를 충족했다. calibration 사용 집단을 바꾸면 결과도 바뀐다는 민감성 근거다.

### 다른 holdout으로 반복한 보조 검사

첫 holdout 결과를 본 뒤, outer seeds `(0, 1, 2, 3, 314159)`와 inner seeds `(42, 8972)`를 모두 교차한 탐색적 후속 검사를 했다. 이 목록에서 유리한 결과를 선택하지 않았으며, 주 분석과 구분한다.

| 방법 | 1%: test − holdout FPIR 범위 | 5%: test − holdout FPIR 범위 |
|---|---:|---:|
| Global safe | +0.359–+0.760%p | +0.167–+1.580%p |
| FIQA-S | +0.349–+0.687%p | +0.415–+1.563%p |
| FIQA-L | +0.374–+0.738%p | +0.134–+1.579%p |

모든 보조 조합에서 test FPIR가 holdout보다 높았다. 하나의 쉬운 holdout만의 결과는 아니라는 근거지만, 고정된 calibration gallery와 codec에 대한 결론이라는 제한은 남는다.

### calibration tail은 일부 인물에 집중

전체 calibration의 global empirical 1% threshold에서 초과 988건 중:

- 최대 기여 1명: 6.58%.
- 상위 10명: 26.82%.
- 이 상위 10명의 전체 non-mated query 비중: 2.86%.
- 초과 사례가 있는 identity: 299명 / 전체 2,319명.

인물별 초과 사례의 집중은 fit/safety 배정에 따른 변동에 기여할 수 있다. 이 관찰만으로 특정 인물을 제외하거나 query를 독립 표본으로 간주하지 않는다.

근거: [holdout metrics](holdout-sensitivity-v1/metrics.csv), [calibration tail concentration](holdout-sensitivity-v1/calibration_tail_concentration.csv).

## 6. 압축 전에도 실패: PQ/FIQA만의 원인은 아님

| 목표 FPIR | 원본 calibration FPIR | 원본 test FPIR | 원본 test 초과 건수 |
|---|---:|---:|---:|
| 1% | 0.9997% | 1.5041% | 1,831 / 121,736 |
| 5% | 4.9997% | 5.3517% | 6,515 / 121,736 |
| 10% | 9.9995% | 9.5321% | 11,604 / 121,736 |

원본 cosine 전용 calibration threshold를 적용해 raw test ledger에서 초과 건수를 재집계했고 저장된 결과와 일치했다. 원본 calibration threshold는 hash 검증된 기존 진단 값을 사용했으며 이번에 calibration 검색을 재실행하지 않았다. cosine threshold를 ADC에 적용한 것이 아니다.

양쪽 gallery template 수는 3,000개로 같고 enrollment 이미지 수는 calibration 60,480 / test 60,294장이다. 단순한 gallery 개수 불일치가 확인된 것은 아니다. 그러나 실제 identity와 이미지, score tail은 다른 집단이다.

근거: [origin_control.json](origin_control.json), [재집계 스크립트](audit_origin_control.py).

## 7. 검증 범위와 남은 질문

현재 검증 수준: **한계를 명시하면 공유 가능한 진단**. 확정적 세부 인과 설명이나 새로운 표본에서의 FPIR 보장으로는 사용할 수 없다.

- 입력 artifact와 raw test ledger의 SHA-256 검증, 분석 전후 source hash 불변 확인.
- calibration/test, holdout/fit identity 비중복 확인; S/L 입력 순서·점수 일치 확인.
- seed 8972의 FPIR/TPIR를 기존 산출물 및 canonical evaluator와 대조해 일치.
- mixture + within-group = 전체 격차를 모든 조합에서 재검증.
- 관련 회귀 테스트 37개 통과.
- 주 분석 약 288초. embedding·FIQA inference·Grad-CAM·ADC search 재실행 없음.

calibration은 training identity로 구성한 watch-list와 known-unknown probe이고, test는 공식 gallery와 unknown-unknown probe다. 이번에 바꾼 것은 내부 fit/safety 및 threshold용 calibration subset이다. gallery 생성 seed 8972나 codec fit을 바꾸지 않았다.

아직 분리되지 않은 원인은 query 집단/영상 조건 차이, gallery 구성 차이, 현재 FIQA bin 내부 이질성 및 codec 일반화다. 모든 원인을 '데이터셋 품질' 또는 '압축 오류' 하나로 확정하지 않는다.

## 8. 다음 구현에 주는 결론

사용자가 정한 순서인 **① 공통 CI → ② 분할 안정성 → ③ 보고 통합**을 유지할 수 있다. 단, 목적을 구분해야 한다.

1. 공통 인물 단위 TPIR CI는 통계 보고의 일관성을 개선한다. FPIR 점추정이나 실패를 직접 교정하지 않는다. 실제 identity가 없는 test non-mated에 cluster FPIR CI를 적용했다고 표시하지 않는다.
2. 분할 안정성에서는 내부 partition seed와 calibration cohort/gallery seed를 분리해 기록한다. 독립 holdout, 범위/목표 충족 수, threshold 및 TPIR 변화를 함께 보고하고 best-seed 선택을 금지한다. 추가적인 gallery 변경은 검색 점수 재계산이 필요한 별도 실험이다.
3. 통합 보고는 실제 FPIR, 목표 충족, TPIR, CI 방식, 내부/외부 seed와 source hash를 함께 표시한다.

추론 설정이나 bootstrap 횟수를 늘리는 것만으로 이번 목표 FPIR 초과가 해결되는 것은 아니다.

## 재현 자료

- [주 분석 스크립트](analyze.py): source 검증, 20개 내부 분할, 독립 holdout, 품질 그룹 분해.
- [주 분석 manifest](audit-v1/manifest.json): 전체 입력 hash, 조건, checks, 산출물 hash.
- [후속 holdout 스크립트](holdout_sensitivity.py): 5 outer × 2 inner 보조 검사.
- [원본 대조군 스크립트](audit_origin_control.py).
- [보고서 payload](artifact.json): 보고서 도구 입력. 렌더링 미완료이며 아래 제한을 참조.

실행은 저장소 루트에서 Python 3.11로 각 스크립트를 실행한다. 완료 출력 디렉터리/파일이 이미 있으면 덮어쓰지 않고 거부한다. 새 실행 시 분석 디렉터리를 별도로 복사/지정해 기존 근거를 보존한다.

### 보고서 표시 도구 제한

Data Analytics MCP 보고서 검증과 portable HTML 빌더가 모두 `source must include the actual SQL query text` 오류로 Python/CSV 출처를 거부했다. 실행하지 않은 SQL을 만들어 출처로 표시하지 않았다. HTML/MCP 보고서가 완성됐다고 주장하지 않으며, 이 Markdown과 실제 CSV/JSON을 검토용 파일 기록으로 제공한다. 예정했던 1% 최소 FPIR 막대그래프는 표로 대체했다. 계산 검증과 보고서 렌더링 제한은 별개다.
