# 핵심 요약 — 1개 seed

아래 표의 수치는 각 데이터셋의 12조건(4 FR × 3 PQ) 중 해당 조건의 수입니다. FPIR·TPIR를 pooling한 값이 아닙니다.

## 모든 seed에서 목표 FPIR를 충족한 조건 수 / 12

```csv
dataset_id,target_fpir,global_safe,fiqa_2bin,fiqa_5bin,continuous_fiqa,plus_outside,plus_entropy,plus_both
lfw,0.01,1,0,1,1,1,2,1
lfw,0.05,2,2,3,3,2,3,2
lfw,0.1,1,1,1,1,1,1,1
lfw,0.2,1,1,1,1,1,1,1
lfw,0.3,2,1,1,1,1,1,1
rfw_custom,0.01,10,11,12,10,10,12,10
rfw_custom,0.05,11,12,12,11,10,9,10
rfw_custom,0.1,11,11,12,11,12,12,12
rfw_custom,0.2,10,12,12,10,10,10,10
rfw_custom,0.3,11,11,12,11,10,11,10
survface,0.01,7,5,5,5,6,7,6
survface,0.05,7,9,8,7,9,9,7
survface,0.1,12,12,12,12,12,12,12
survface,0.2,12,12,12,12,12,12,12
survface,0.3,12,12,12,12,12,12,12
```

## FIQA 대비 saliency 효과를 읽는 순서

목표 충족 횟수만으로 효과를 판정하지 않습니다. conditions/의 paired 표에서 reference_method=baseline, metric=tpir_at_rank_k를 확인하세요.
먼저 both_target_met_seeds와 두 방법의 실제 FPIR 범위를 확인하고, delta_min/median/max 및 positive/negative_ci_seeds를 함께 읽습니다.
두 방법이 모두 목표를 충족해도 동일 실제 FPIR는 아닙니다. seed들은 독립 반복이 아니며 다중 비교 보정도 없습니다.
일괄적인 효과 있음/없음 판정이나 좋은 조건만의 선정 없이, 모든 조건의 크기·손실·CI·fallback을 상세 표에 보존했습니다.
