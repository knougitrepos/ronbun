# RFW 1:1 Continuous FIQA Threshold Calibration: 실행과 결과 해석

## 1. 연구 질문과 범위

본 실험은 사전학습된 얼굴 인식 모델(ArcFace, AdaFace, MagFace 등)에서 추출된 512D 얼굴 임베딩을 Product Quantization(PQ m=128, 64, 32, 8-bit)으로 압축했을 때 발생하는 점수 왜곡에 대해, Continuous FIQA 기반 임계값 보정이 Global Safe 및 FIQA 5-bin 대비 어떤 성능과 임계값 적응 특성을 나타내는지 평가합니다.

- **프로토콜**: RFW (Racial Faces in the Wild) 공식 1:1 verification 프로토콜을 엄격히 준수합니다.
  - 4개 인종 그룹: African, Asian, Caucasian, Indian.
  - 각 그룹별 10-fold cross-validation.
  - 1개 fold당 300 genuine pair + 300 impostor pair = 600 pairs.
  - 그룹당 6,000 pairs, 4개 그룹 총 24,000 pairs.
- **프로토콜 분리 원칙**:
  - **RFW-Official**: 1:1 verification 프로토콜 (본 평가 대상). TAR@FMR, Realized FMR, Target Met Rate, Accuracy를 보고하며, SurvFace의 1:N open-set identification 지표(TPIR@20, FPIR, Rank/Threshold failure)나 TinyFace의 closed-set rank-k 지표와 혼용하지 않습니다.
  - **RFW-Custom**: 비공식 1:N open-set diagnostic. RFW-Official과 artifact 및 protocol UID를 공유하지 않으며, 상호 전이를 수행하지 않습니다.
- **비교 행렬**:
  - 표현(압축): Origin 512D float32, PQ 512D m128 b8, PQ 512D m64 b8, PQ 512D m32 b8.
  - 보정 방법: Global Safe (전역 분위수 안전 임계값), FIQA 5-bin (품질 분위수 5구간 계단식 임계값), Continuous FIQA (구간별 선형 basis를 사용하는 스플라인 분위수 회귀와 safety offset).
  - 목표 FMR: 0.001 (0.1%), 0.01 (1%), 0.05 (5%), 0.10 (10%).
- **공정성 및 보장 한계**:
  - 인종 그룹별로 독립 평가 및 집계를 수행하되, 이를 공정성 개선이나 인종 간 동등성 보장으로 주장하지 않습니다 (`fairness_guarantee = False`).
  - Held-out test fold에서의 `target_met_rate` 달성은 경험적 표본 관측 결과일 뿐, 수학적/통계적 FMR 안전 보장이 아닙니다 (`formal_fmr_guarantee = False`).

---

## 2. 데이터 분할과 역할 분리 (Two-Endpoint Identity Disjointness)

1. **공식 10-fold 분할**:
   각 iteration에서 1개 fold를 held-out test로 두고, 나머지 9개 fold를 threshold calibration 풀로 사용합니다.
2. **양쪽 Endpoint Identity Disjointness (두 끝점 인물 분리)**:
   - 1:1 verification에서 impostor pair는 서로 다른 두 인물 $(id_{\text{left}}, id_{\text{right}})$로 구성됩니다.
   - calibration 데이터 내에서 임의로 pair를 fit과 safety로 분할할 경우, 동일한 인물이 fit pair와 safety pair에 걸쳐 나타나는 identity leakage(인물 누수)가 발생합니다.
   - 이를 원천 차단하기 위해, calibration 대상의 모든 identity를 고정 SHA-256 해시 함수를 통해 결정론적으로 `fit` (70%) 또는 `safety` (30%) 역할로 배정합니다.
   - **양쪽 인물이 모두 `fit`에 속한 pair만 fit에 사용하고, 양쪽 인물이 모두 `safety`에 속한 pair만 safety에 사용합니다.**
   - 한쪽은 `fit`, 다른 쪽은 `safety`에 속하는 cross-partition impostor pair는 calibration에서 완전히 제외합니다.
   - 제외된 pair 수량과 전체 분모는 `cross_partition_pairs_excluded`, `calibration_pairs_retained` 메타데이터에 투명하게 기록됩니다.
3. **PQ Development 분리 및 한계 명시**:
   - PQ 코덱(codebook 및 subquantizers)은 LFW View-2 test pair와 분리된 모델별 development 임베딩(LFW disjoint non-test 1,549장)에서만 학습됩니다.
   - RFW 평가 임베딩과의 byte hash 교집합 검사를 통해 동일 임베딩 벡터의 유출을 차단합니다.
   - **한계 명시**: LFW 개발 데이터의 인물과 RFW 평가 데이터 인물 간의 cross-dataset identity 분리 여부는 공식적으로 검증되지 않았습니다 (`identity_overlap_verified = False`).

---

## 3. 품질 정의: LFW와의 차이점

- **LFW 1:1 검증**: 1:1 pair 평가에 query/reference 비대칭 압축을 적용(비대칭 DB 검색 모사)하고, 쿼리의 단일 품질 점수를 사용 ($q_{\text{pair}} = q_{\text{left}}$).
- **RFW 1:1 검증**: 두 얼굴 간의 대칭적 동일인 여부를 판별하는 순수 1:1 대칭 검증 프로토콜이므로, 대칭 품질 연산자(`quality_mode="symmetric_min"`)를 기본값으로 사용:
  $$q_{\text{pair}} = \min(q_{\text{left}}, q_{\text{right}})$$
- 두 프로토콜은 목적과 데이터 구조가 상이하므로 품질 정의를 임의로 통일하거나 전이하지 않습니다.

---

## 4. 점수 공간과 임계값 격자 (Score Space Isolation)

- **Origin 512D**: Cosine similarity 점수 공간, 유효 점수 범위 $[-1.0, 1.0]$.
- **PQ ADC (m=128, 64, 32)**: Asymmetric Distance Computation 점수 공간, Negative squared L2 distance (Lookup table 합산), 유효 점수 범위 $[-4.0, 0.0]$.
- Cosine 점수 공간과 PQ ADC 점수 공간은 완전히 상이한 척도(scale)와 통계적 특성을 가집니다.
- 따라서 Cosine 임계값을 PQ ADC에 그대로 전용(`frozen_origin`)하지 않으며, 각 점수 공간에 특화된 candidate grid 범위 내에서 독립적으로 calibration을 수행합니다.

---

## 5. 저장 공간 회계 (Storage Accounting)

각 압축 프로파일의 저장 효율은 템플릿 페이로드와 공유 코드북을 명확히 구분하여 계산합니다:

| 프로파일 | 템플릿 페이로드 | 공유 코드북 크기 | 12,000장 기준 총 저장량 | 압축 배율 (vs Origin) |
| :--- | :--- | :--- | :--- | :--- |
| **Origin 512D** | 2,048 Bytes (float32) | 0 Bytes | 24.576 MB | 1.00x |
| **PQ m128 b8** | 128 Bytes (uint8) | 524,288 Bytes (0.524 MB) | 2.060 MB | 11.93x |
| **PQ m64 b8** | 64 Bytes (uint8) | 524,288 Bytes (0.524 MB) | 1.292 MB | 19.02x |
| **PQ m32 b8** | 32 Bytes (uint8) | 524,288 Bytes (0.524 MB) | 0.908 MB | 27.06x |

*참고: 위 수치는 순수 벡터 데이터 및 코드북 저장량이며, DB row 오버헤드, 인덱스 구조, FIQA 메타데이터, 원본 fallback 저장 비용은 포함되지 않습니다.*

---

## 6. 실행 방법 및 CLI

### 설정 파일
`configs/experiments/rfw_continuous_calibration.yaml`

### CLI 실행
```powershell
# 1. 계획 및 설정 검증 (Dry run)
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml

# 2. 정식 실행 (실제 RFW 아카이브 및 모델 가중치 기본 적용)
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute

# 3. 고속 축소 검증 (Quick smoke: African 2-fold 축소 검증, 1,200쌍)
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute --quick-smoke

# 4. 상세 원본 결과 보존 실행
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute --quick-smoke --keep-raw-results

# 5. CI/회귀 테스트용 합성 데이터 실행
py -3.11 -m research.experiments.rfw_continuous_calibration --config configs/experiments/rfw_continuous_calibration.yaml --execute --synthetic
```

---

## 7. 결과 산출물 구조 및 해석 순서

실험 완료 시 `results/calibration/rfw_continuous_calibration/<run_id>/` 디렉터리에 다음 산출물이 생성됩니다:

1. `CHATGPT_SUMMARY.md`: LLM 분석용 경량 요약본. 실행 설정, 주요 메트릭 표, 주의사항 수록.
2. `RFW_CONTINUOUS_CALIBRATION.md`: 본 해석 안내서 불변 사본.
3. `run_manifest.json`: 실행 상태(`completed`), config SHA-256, 결과 파일 해시, `formal_fmr_guarantee = False`, `identity_overlap_verified = False` 선언.
4. `group_summary.csv`: 인종 그룹 × 압축 프로파일 × 보정 방법 × 목표 FMR별 `mean_realized_fmr`, `mean_tar`, `target_met_rate`, 풀링된 오류/분모.
5. `comparison_table.csv`: Global Safe 대비 FIQA 5-bin 및 Continuous FIQA의 TAR gain 집계 (`tar_gain_vs_global`).
6. `fold_metrics.csv`: fold별 세부 결과 및 신뢰구간 (감사 추적을 위해 항상 보존).
7. `raw_pair_evaluations.csv.gz` (옵션 `KEEP_RAW_RESULTS=True` 시에만 생성): pair별 점수, 품질, 판정 상세 테이블.
8. `fitted_calibration_models.pkl` (옵션 `KEEP_RAW_RESULTS=True` 시에만 생성): 각 fold별 학습된 Continuous/Conditional 모델 파라미터.

### 지표 해석 요령
- **mean_realized_fmr**: test fold에서 관측된 실제 FMR의 평균. 목표 FMR와 일치 여부 점검.
- **target_met_rate**: test fold 중 `realized_fmr <= target_fmr` 조건을 만족한 fold의 비율.
- **mean_tar**: test fold에서의 True Accept Rate 평균.
- **tar_gain_vs_global**: 동일 조건에서 Continuous FIQA가 Global Safe 대비 달성한 TAR 차이 ($TAR_{\text{continuous}} - TAR_{\text{global}}$).
- **pooled_false_accepts / pooled_impostor_pairs**: 10개 fold에 걸친 풀링된 오인식 수 및 분모.
