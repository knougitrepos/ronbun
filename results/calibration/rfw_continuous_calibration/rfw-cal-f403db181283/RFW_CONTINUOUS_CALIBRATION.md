# RFW 1:1 Continuous FIQA Threshold Calibration 안내서

## 1. 연구 질문과 비교 조건
본 연구는 사전학습된 얼굴 임베딩 및 fine-tuned 모델을 대상으로,
Product Quantization(PQ m=128, 64, 32) 압축 시 발생하는 점수 분포 변화에 대해
Continuous FIQA(구간별 선형 basis를 사용하는 스플라인 분위수 회귀와 safety offset)가
Global Safe 및 FIQA 5-bin 대비 어떤 차이를 만드는지
RFW 공식 4개 인종 그룹(African, Asian, Caucasian, Indian) 10-fold 1:1 검증 프로토콜에서 평가합니다.

- **비교 조건**:
  - 압축: Origin 512D float32, PQ m128 b8, PQ m64 b8, PQ m32 b8
  - 보정: Global Safe, FIQA 5-bin, Continuous FIQA
  - 목표 FMR: 0.001, 0.01, 0.05, 0.10

## 2. 데이터 분할과 역할 분리
1. **공식 10-fold 분할 보존**:
   각 fold를 held-out test로 두고, 나머지 9개 fold를 calibration에 사용합니다.
2. **양쪽 endpoint identity 역할 일치 계약 (Two-Endpoint Identity Disjointness)**:
   1:1 pair 검증에서 impostor pair는 서로 다른 identity를 가집니다.
   임의의 impostor pair가 fit과 safety에 걸치는 경우 발생하는 identity 누수를 차단하기 위해,
   좌/우 identity를 동일한 hash 시드로 'fit'/'safety'로 분할하고,
   양쪽 모두 동일한 역할에 배정된 pair만 calibration에 유지합니다.
   불일치하는 cross-partition pair는 calibration에서 제외하며 제외 수량과 분모를 기록합니다.
3. **PQ Development 분리**:
   PQ 코덱은 반드시 RFW test 평가 데이터와 엄격히 분리된 모델별 development 임베딩(LFW disjoint non-test 1,549장)으로만 학습합니다.
   평가 데이터셋 벡터와의 byte hash 교집합 검사를 통해 어떠한 슬라이스나 재배열 누수도 원천 차단합니다.

## 3. 품질 정의: LFW와의 차이
- **LFW**: 1:1 pair 평가에 query/reference 비대칭 압축을 적용(비대칭 DB 검색 모사), query-only FIQA($q_\text{pair} = q_\text{query}$).
- **RFW**: 순수 1:1 대칭 검증(symmetric min FIQA, $q_\text{pair} = \min(q_\text{left}, q_\text{right})$).
두 데이터셋의 품질 정의는 프로토콜 목적이 다르므로 동일한 설정으로 취급하지 않습니다.

## 4. 점수 공간과 임계값
- Origin: Cosine similarity [-1, 1]
- PQ ADC: Asymmetric distance computation, negative squared L2 [-4, 0]
Cosine과 PQ ADC는 별도의 점수 공간이며, 임계값 범위도 독립적으로 적합됩니다.

## 5. 지표 해석 및 감사 추적 (Audit Trail)
- **realized_fmr**: test fold에서 실제로 관측된 False Match Rate.
- **tar**: True Accept Rate (genuine pair 중 임계값을 통과한 비율).
- **tar_gain_vs_global**: 동일 조건에서 Continuous FIQA가 Global Safe 대비 달성한 TAR 차이 ($TAR_\text{continuous} - TAR_\text{global}$).
- **target_met_on_test**: realized_fmr <= target_fmr 여부.
- **pooled_false_accepts / pooled_impostor_pairs**: 10개 fold에 걸친 풀링된 오인식 수 및 분모.
- **fold_metrics.csv**: 요약 모드(`KEEP_RAW_RESULTS=False`)에서도 fold별 오류 수·분모·신뢰구간 감사 추적을 위해 필수 보존됩니다.
- **주의사항**:
  - realized_fmr <= target_fmr는 경험적 표본에서의 달성이며 수학적/통계적 FMR 보장이 아닙니다 (`formal_fmr_guarantee = False`).
  - 그룹 간 성능 차이를 보고하되, 이를 공정성 개선이나 보장으로 해석하지 않습니다 (`fairness_guarantee = False`).

## 6. 실행 방법 및 CLI
```powershell
# 1. 계획 및 설정 검증 (Dry run)
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml

# 2. 정식 실행 (실제 RFW 데이터 및 실모델 가중치 기본 적용)
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute

# 3. 고속 축소 검증 (Quick smoke: African 2-fold, 1,200쌍)
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute --quick-smoke

# 4. CI/회귀 테스트용 합성 데이터 실행
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute --synthetic
```
