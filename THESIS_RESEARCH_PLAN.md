# 석사논문 연구 및 구현 준비 계획

## 1. 최종 연구 방향

본 연구는 **PostgreSQL/pgvector 기반 압축 얼굴 임베딩 검색에서 압축으로 인해 변형되는 유사도 분포와 미등록 인물 거부 임계값을 경량 calibration으로 보정할 수 있는지**를 분석한다.

품질 적응형 얼굴 템플릿 집계는 독립적인 핵심 novelty로 주장하지 않고, 압축 환경에서 검색과 미등록 인물 거부를 안정화하는 보조 구성 및 ablation 축으로 둔다.

연구 대상은 정렬된 얼굴 crop이며, 얼굴 임베딩 모델은 사전학습된 `ArcFace/InsightFace`를 동결하여 사용한다. 대규모 모델 재학습은 수행하지 않는다.

시스템은 다음 두 상황을 모두 처리한다.

1. **등록 인물 검색**
   - 질의 인물이 데이터베이스에 존재한다.
   - Rank-1과 Top-K 검색 정확도를 평가한다.

2. **미등록 인물 거부**
   - 질의 인물이 데이터베이스에 존재하지 않는다.
   - 시스템은 잘못된 등록 인물을 반환하지 않고 `unknown`으로 거부해야 한다.

핵심 연구 질문은 다음과 같다.

> 압축된 ArcFace 얼굴 임베딩을 PostgreSQL/pgvector 기반 검색 시스템에 저장할 때, 압축으로 변형되는 유사도 분포를 경량 calibration으로 보정하여 등록 인물 식별 성능과 미등록 인물 거부 성능을 동시에 보존할 수 있는가?

## 2. 연구 범위와 제약

- 공개 데이터셋만 사용한다.
- 수천 명, 수만 장 수준의 중간 규모 데이터셋을 대상으로 한다.
- 정렬된 얼굴 crop끼리 검색한다.
- 데이터베이스 등록은 인물당 1장을 기본으로 한다.
- 동일 인물의 등록 이미지가 여러 장이면 하나의 대표 임베딩으로 결합한다.
- GPU는 GTX 1080 Ti를 사용하므로 ArcFace나 대형 품질 모델을 재학습하지 않는다.
- 사전학습 모델 추론, PCA/PQ 학습, 경량 보정 모델 학습만 수행한다.
- 일반 이미지 모델의 배경 편향과 segmentation 기반 보정은 본 연구 범위에서 제외한다.

## 3. 핵심 Novelty

### 3.1 이상치 제거와 품질 적응형 얼굴 템플릿 집계

이 구성은 핵심 novelty가 아니라 압축 인지형 거부 보정을 보조하는 실험 축이다. 단일 등록 이미지와 단순 평균을 기준선으로 두고, 품질 기반 템플릿이 압축 전후의 검색 안정성과 open-set 거부 성능을 추가로 개선하는지 확인한다.

동일 인물의 등록 이미지가 여러 장인 경우 단순 평균만 사용하지 않는다.

1. ArcFace 임베딩 사이의 코사인 거리를 이용해 템플릿 중심에서 크게 벗어난 샘플을 제거한다.
2. 남은 얼굴에 품질 점수를 부여한다.
3. 품질 점수에 따라 가중 평균한다.
4. 최종 대표 임베딩을 L2 정규화한다.

품질 측정 방식은 다음 두 종류를 비교한다.

- **규칙 기반 품질 점수**
  - 해상도
  - 흐림 정도
  - 얼굴 검출 신뢰도
  - yaw/pitch 기반 자세

- **사전학습 FIQA 품질 점수**
  - 공개된 사전학습 Face Image Quality Assessment 모델을 동결하여 사용한다.

다음 집계 방식을 ablation으로 비교한다.

| 집계 방식 | 설명 |
| --- | --- |
| Single enrollment | 등록 이미지 1장을 그대로 사용 |
| Mean | 여러 임베딩의 단순 평균 |
| Outlier + Mean | 이상치 제거 후 균등 평균 |
| Outlier + Rule quality | 이상치 제거 후 규칙 기반 품질 가중 평균 |
| Outlier + FIQA quality | 이상치 제거 후 FIQA 품질 가중 평균 |

### 3.2 압축 인지형 미등록 인물 거부 보정

고정 코사인 유사도 임계값만으로 미등록 여부를 판단하지 않는다. 압축으로 유사도 분포가 변하는 현상을 고려하여 등록 확률을 보정한다.

본 연구의 핵심 기법은 **Quality-Compression Aware Threshold Calibration(QCU-ATC)**로 부르되, 복잡한 end-to-end 학습 모델이 아니라 다음 feature를 사용하는 경량 accept/reject calibration 모듈로 정의한다.

```text
x_q = [
  s_1,
  s_1 - s_2,
  Q_q,
  Q_T,
  Var(T_i),
  enrollment_count,
  E_rec_norm,
  compression_profile_one_hot
]

y_hat = g_theta(x_q)
L = L_BCE
```

각 feature의 의미는 다음과 같다.

| Feature | 의미 |
| --- | --- |
| `s_1` | top-1 후보와의 cosine similarity |
| `s_1 - s_2` | top-1과 top-2 후보의 score margin |
| `Q_q` | 질의 이미지 품질 |
| `Q_T` | top-1 후보 템플릿의 평균 품질 |
| `Var(T_i)` | top-1 후보 템플릿 내부 분산 |
| `enrollment_count` | 해당 identity의 등록 이미지 수 |
| `E_rec_norm` | 압축 방식별 z-score 정규화된 재구성 오차 |
| `compression_profile_one_hot` | original/PCA/PQ/PCA+PQ 등 압축 방식 categorical feature |

PCA와 PQ의 재구성 오차는 분포와 의미가 다르므로 raw `E_rec`를 하나의 공통 변수처럼 사용하지 않는다. 각 압축 profile별 평균과 표준편차로 `E_rec_norm = (E_rec - mu_C) / sigma_C`를 계산하고, 압축 방식 one-hot feature를 함께 넣는다.

보정 모델은 ArcFace를 재학습하지 않고, 다음 모델 복잡도 ablation으로 비교한다.

| 모델 | 목적 |
| --- | --- |
| Fixed global threshold | 가장 단순한 기준선 |
| Per-compression threshold | 압축 profile별 임계값 보정 기준선 |
| Logistic regression | 기본 제안 모델 |
| Shallow MLP | 약한 비선형 calibration 효과 확인 |
| Random forest 또는 LightGBM | 선택 실험. feature interaction 확인용 |

본문 핵심 학습 손실은 binary cross entropy 하나로 제한한다. `rank loss`와 `consistency loss`는 본문 필수 구성으로 넣지 않고, 성능 향상과 구현 여유가 있을 때만 선택 ablation 또는 부록으로 둔다. consistency loss를 다룰 경우에는 압축 전후 cosine score 차이를 줄이는 `score consistency`인지, 압축 전후 보정 확률 차이를 줄이는 `decision consistency`인지 명시해야 한다.

비교 대상은 다음과 같다.

| 거부 방식 | 설명 |
| --- | --- |
| Global threshold | 모든 조건에 동일한 임계값 |
| Per-compression threshold | 압축 프로파일별 개별 임계값 |
| QCU-ATC / learned calibration | 품질, 템플릿 통계, 압축 방식별 정규화 오차를 이용한 등록 확률 보정 |

### 3.3 제한적인 품질 적응형 압축 정책

모든 대표 템플릿에 동일한 압축률을 적용하지 않고 템플릿의 식별 위험에 따라 압축 프로파일을 선택한다.

이 요소는 핵심 기여가 아니라 선택 확장이다. 본문 핵심 결과가 충분히 확보된 뒤에만 수행하며, 구현 범위가 커지면 Phase D 또는 부록으로 내린다.

초기 압축 정책은 다음처럼 제한한다.

| 템플릿 상태 | 압축 정책 |
| --- | --- |
| 고품질, 낮은 내부 분산 | 강한 압축 |
| 중간 품질 또는 중간 분산 | PCA-256D |
| 단일 등록, 저품질 또는 높은 분산 | 원본 512D 또는 약한 압축 |

이 정책은 새로운 대형 신경망으로 학습하지 않는다. 개발 세트에서 저장공간을 최소화하면서 허용 가능한 `DIR` 또는 Rank-1 손실을 만족하도록 임계값을 선택한다.

이 요소는 핵심 기여인 압축 인지형 거부 보정을 보조하는 제한적 확장으로 둔다.

## 4. Novelty 수준에 대한 안전한 주장

품질 기반 템플릿 집계, 얼굴 품질 평가, open-set 얼굴 식별, PCA/PQ는 각각 기존 연구가 존재한다. 따라서 각 기술을 단순히 결합했다는 주장만으로는 충분하지 않다.

본 연구의 차별점은 다음 조합에 둔다.

> 동결된 ArcFace 기반 중간 규모 얼굴 검색에서 이상치 제거 및 품질 적응형 템플릿 집계를 적용하고, 압축으로 변형된 유사도 분포를 품질·템플릿 분산·압축 오차를 이용해 보정하여 등록 인물 식별과 미등록 인물 거부를 동시에 평가한다.

논문에서 피해야 할 주장은 다음과 같다.

- 최초의 품질 기반 얼굴 템플릿 집계
- 최초의 open-set 얼굴 식별
- 최초의 얼굴 임베딩 압축
- 완전히 새로운 얼굴 인식 모델

안전한 기여 주장은 다음과 같다.

1. PostgreSQL/pgvector 기반 얼굴 임베딩 검색에서 PCA/PQ 압축이 closed-set 검색 정확도, open-set 미등록 거부, 유사도 분포, 임계값 안정성에 미치는 영향을 분석한다.
2. top-1 유사도, score margin, 얼굴 품질, 템플릿 분산, 압축 방식별 정규화 재구성 오차를 이용한 경량 압축 인지형 미등록 거부 보정 방법을 제안한다.
3. 단일 등록, 단순 평균, 이상치 제거, 품질 기반 템플릿 집계가 압축 전후 성능에 주는 영향을 ablation으로 검증한다.
4. PostgreSQL/pgvector 환경에서 실제 검색 지연시간, 저장비용, HNSW recall-latency trade-off를 함께 측정한다.

## 5. 데이터셋과 분할 원칙

### 5.1 데이터 규모

- 수천 명의 identity
- 수만 장의 정렬 얼굴 이미지
- 공개 데이터셋만 사용

### 5.2 분할 원칙

identity 누수를 방지하기 위해 다음 세 집합의 identity를 서로 겹치지 않게 분리한다.

1. **Development**
   - PCA/PQ 학습
   - 이상치 제거 계수와 품질 가중치 결정
   - 보정 모델 학습
   - 압축 정책 임계값 결정

2. **Calibration**
   - 목표 FPIR에 맞는 최종 거부 임계값 결정
   - 보정 확률의 calibration 확인

3. **Test**
   - 모든 최종 결과 보고
   - 개발 및 임계값 선택에 사용하지 않음

Test probe는 다음 세 종류로 구성한다.

- 등록 identity의 새로운 이미지
- **Known unknown**: 같은 공개 데이터셋 또는 같은 도메인에서 오지만 gallery에는 등록하지 않은 identity 이미지
- **Unknown unknown**: 학습, calibration, gallery 구성에 전혀 사용하지 않은 별도 hold-out identity 또는 별도 공개 데이터셋의 identity 이미지

`known unknown`과 `unknown unknown`은 최종 결과에서 분리 보고한다. 이 구분 없이 모든 미등록 인물을 하나로 합치면, 단순 threshold가 데이터 분포가 비슷한 미등록 인물에서 취약한지 검증하기 어렵다.

### 5.3 등록 장수 실험

다음 조건을 분리하여 평가한다.

- 인물당 등록 이미지 1장
- 인물당 등록 이미지 2장
- 인물당 등록 이미지 5장
- 실제 가용 이미지 수에 따른 혼합 등록

## 6. 처리 파이프라인

```text
공개 얼굴 데이터셋
    -> 정렬 얼굴 crop 검증
    -> ArcFace 512D 임베딩 추출
    -> 얼굴 품질 점수 계산
       -> 규칙 기반 점수
       -> FIQA 점수
    -> identity별 이상치 제거
    -> 대표 템플릿 생성
    -> 압축 프로파일 생성
       -> 원본 512D
       -> PCA-256D
       -> 선택적 강한 압축
    -> PostgreSQL/pgvector 저장 및 HNSW 검색
    -> 등록 후보 점수 및 템플릿 통계 추출
    -> 압축 인지형 등록 확률 보정
    -> identity 반환 또는 unknown 거부
```

## 7. 압축 프로파일과 PQ 구현 제약

현재 저장소의 PQ 코드는 `EmbeddingPQ.codes`의 `LargeBinary`에 코드를 저장한다. 이 값은 pgvector의 HNSW 인덱스로 직접 검색할 수 없다.

따라서 실험을 다음처럼 구분한다.

### PostgreSQL/pgvector 직접 검색 프로파일

- ArcFace 512D `vector`
- PCA-256D `vector`
- pgvector 버전이 지원하면 PCA-256D `halfvec`
- 필요하면 binary quantization 기반 후보 검색과 원본 벡터 재정렬

### PQ 보조 실험

- Faiss `IndexPQ`를 이용한 검색
- 또는 PQ 복원 벡터를 이용한 정확도 손실 분석
- PQ code 저장량과 codebook 크기를 분리 측정

PQ 복원 벡터를 일반 `vector(256)`로 저장하고 이를 PQ 저장공간이라고 보고해서는 안 된다. 검색 가능한 DB 표현과 실제 압축 code 저장량을 구분한다.

구현 초기에 pgvector 버전과 `halfvec`/binary quantization 지원 여부를 확인한 뒤 최종 압축 프로파일을 고정한다.

## 8. 평가 지표

### 8.1 등록 인물 검색

- Rank-1
- Rank-5
- CMC
- mAP

### 8.2 미등록 인물 거부

- DIR@FPIR
- FNIR@FPIR
- FPIR
- AUROC
- FPR@95TPR
- ROC 또는 DET curve
- Expected Calibration Error
- Brier score

주요 목표 FPIR은 개발 세트에서 고정한 뒤 Test에서 보고한다.
모든 open-set 지표는 전체 unknown뿐 아니라 `known unknown`과 `unknown unknown`으로 나누어 보고한다.

### 8.3 압축 및 시스템 효율

- 벡터당 저장 byte
- 전체 임베딩 및 인덱스 크기
- 압축률
- 압축 재구성 오차
- 인덱스 생성 시간
- P50/P95 질의시간
- HNSW exact-search 대비 recall

### 8.4 통계 검증

- identity 단위 bootstrap 95% 신뢰구간
- 동일 probe에 대한 paired bootstrap
- 고품질/중간품질/저품질 구간별 성능
- 등록 장수별 성능

## 9. 핵심 실험 매트릭스

모든 조합을 무제한으로 늘리지 않는다. 다음 순서로 실험한다.

### Phase A: 대표 템플릿 집계

- Single
- Mean
- Outlier + Mean
- Outlier + Rule quality
- Outlier + FIQA quality

### Phase B: 압축

- 512D 원본
- PCA-256D
- PostgreSQL에서 직접 검색 가능한 강한 압축 프로파일 1개
- PQ는 보조 실험

### Phase C: 미등록 거부

- Global threshold
- Per-compression threshold
- Logistic regression calibration
- Shallow MLP calibration
- Feature ablation
  - reconstruction error 제외
  - quality feature 제외
  - template dispersion 제외
  - compression categorical feature 제외

### Phase D: 제한적 적응형 압축

- Uniform compression
- Quality-aware profile selection

먼저 각 Phase의 최선 설정을 선택한 뒤 다음 Phase와 결합한다. 처음부터 모든 Cartesian product를 실행하지 않는다.

## 10. 앞으로 novelty를 높이는 방향

현재 연구가 완료된 뒤 다음 순서로 확장할 수 있다.

1. **품질 적응형 비트 예산 최적화**
   - 고정된 2~3개 프로파일 선택을 넘어 identity별 저장 bit를 최적화한다.

2. **분포 이동 강건성**
   - 개발 데이터셋과 다른 공개 얼굴 데이터셋에서 보정 모델과 임계값의 전이 가능성을 평가한다.

3. **불확실성 기반 보정**
   - 로지스틱 회귀 대신 conformal prediction 또는 uncertainty-aware calibration을 적용한다.

4. **동적 Gallery 변화**
   - identity 추가와 삭제 후 보정 임계값 및 HNSW 성능 변화를 분석한다.

5. **공정성 분석**
   - 데이터셋에 허용된 demographic annotation이 있을 경우 품질·압축 정책이 집단별 오류율을 악화시키는지 검증한다.

이번 석사논문에서는 1~5를 모두 구현하지 않는다. 핵심 범위는 압축 인지형 거부 보정과 그 ablation이며, 품질 적응형 집계는 보조 실험, 제한적 압축 프로파일 선택은 선택 확장으로 둔다.

## 11. 구현 진행 순서

실험은 계산 로직을 담은 `research/` 모듈과 그 모듈을 순서대로 호출하는 노트북으로 분리한다. 데이터·모델·원본 임베딩 생성은 `notebooks/prerequisite/`, 압축·open-set·Grad-CAM 실험은 `notebooks/experiments/`, 집계는 `notebooks/reports/`에 두며 공통 위험 로직은 복제하지 않고 `research/`에서 공유한다. DB 연결 정보는 `configs/database.yaml`과 Git에서 제외되는 `configs/database.local.yaml`에 두며 노트북에 직접 작성하지 않는다.

LFW 기본 흐름은 다음과 같다.

1. `notebooks/prerequisite/embeddings/lfw/00_protocol_and_run_freeze.ipynb`
   - 데이터셋 manifest와 identity-disjoint development/calibration/test 분할을 확정한다.
   - 등록 probe, known unknown, unknown unknown 구성을 고정한다.
   - 전체 설정 hash와 Git 상태를 기록하고 새 실험 run을 생성한다.
2. `notebooks/prerequisite/embeddings/lfw/01_arcface_embedding_extraction.ipynb`
   - ArcFace 512D 원본 임베딩과 얼굴 검출·품질 메타데이터를 추출한다.
   - 원본 임베딩을 PostgreSQL의 원본 테이블에 저장한다.
3. `notebooks/experiments/compression/lfw/00_compressor_fit.ipynb`
   - development split만 사용해 PCA와 보조 PQ codebook을 학습한다.
   - 압축 프로파일, 학습 입력 hash, 모델 checksum을 기록한다.
4. `notebooks/experiments/compression/lfw/01_compressed_materialization_and_index.ipynb`
   - DB의 원본 임베딩을 PCA 등 검색 가능한 표현으로 변환해 별도 테이블에 저장한다.
   - test/calibration identity의 원본 512D template과 PCA-256 retrieval template을 각각 `template_embedding_512`와 `template_embedding_256`에 저장한다.
   - pgvector HNSW 인덱스 존재 여부와 저장 byte를 확인한다. 실제 index build time은 빈 테이블에 미리 생성된 전역 index의 `IF NOT EXISTS` 시간을 사용하지 않고, 별도의 깨끗한 DB snapshot 실험에서 측정한다.
   - PQ code는 pgvector 검색 벡터로 취급하지 않고 Faiss/복원 오차 보조 실험으로 분리한다.
5. `notebooks/experiments/open_set/lfw/00_probe_search_and_certification.ipynb`
   - calibration split의 origin-512와 PCA-256 pgvector exact 검색 결과로 점수 공간별 목표 FPIR threshold를 고정한다.
   - test probe마다 origin-512 exact, PCA-256 exact, PCA-256 HNSW Top-K를 분리 실행한다.
   - candidate recall, 압축 rank inversion, HNSW rank inversion, threshold crossing, P50/P95 latency를 기록한다.
   - certificate는 원본 query 512D와 reconstructed template 512D를 사용하여 query angular error를 0으로 둔다.
   - HNSW 후보 집합의 certificate는 전역 보증으로 주장하지 않으며 candidate recall과 함께 보고한다.
   - BCE 기반 logistic calibration은 profile별 threshold 시스템 baseline이 안정화된 뒤 추가한다.
6. `notebooks/experiments/open_set/lfw/01_evaluation_and_visualization.ipynb`
   - DIR@FPIR, FNIR@FPIR, Rank-K, calibration, 저장량, 지연시간을 계산한다.
   - bootstrap 신뢰구간, 결과 표, 실패 사례, 논문용 그림을 생성한다.

SurvFace 흐름은 공식 MAT/CSV 순서, 3,000개 gallery identity의 모든 이미지를 평균한 template, registered 및 unknown-unknown probe만을 사용한다. 공식 test는 PCA/PQ 또는 calibration 학습에 사용하지 않고 development 데이터에서 동결한 모델을 가져온다. 평가는 rank-20과 TPIR@FPIR 0.1/0.2/0.3 및 AUC를 포함한다.

재사용되거나 잘못 바꾸면 결과 전체에 영향을 주는 DB 처리, ArcFace 추론, 압축, 검색, calibration, 지표 계산은 `.py` 모듈로 유지한다. 노트북은 설정 고정, 단계 호출, 결과 검토만 담당한다.

실험 기록은 `runs/YYYY/MM/DD/YYYYMMDD-RNNN-<config-hash>_<name>/`에 저장한다. 각 run에는 비밀정보를 제거한 manifest, 구조화된 JSONL 로그, phase별 attempt, 산출물 checksum을 남긴다. 완료된 run은 덮어쓰지 않는다. 중단 후에는 커널을 재시작하고 bootstrap과 입력 검증부터 실행하며, 상위 단계 hash가 달라졌다면 영향을 받는 단계부터 새 attempt 또는 새 run으로 다시 수행한다.

### 11.1 2026-07-14 구현 상태

- `research/experiments/lfw_pgvector.py`에 identity template materialization, origin/PCA별 calibration exact 검색, test exact/HNSW 후보 검색, candidate recall, 원본 query certificate, 선택적 origin exact fallback 측정을 구현했다.
- 실제 PostgreSQL smoke test에서는 임시 test/calibration template scope를 생성하여 exact/HNSW 검색과 query angular error 0을 확인한 뒤 임시 행을 모두 삭제했다.
- 이 smoke test는 30개 calibration probe와 15개 test probe의 기능 검증이므로 논문 결과로 사용하지 않는다.
- 다음 정식 실행은 변경된 config hash로 00부터 새 LFW run을 만들고 05까지 순서대로 수행한다.
- 정식 LFW baseline이 확보된 뒤 per-compression/logistic calibration, `ef_search`·candidate K sweep, PQ Faiss baseline, SurvFace 공식 실험 순으로 확장한다.

## 12. 제목 후보

추천 제목:

> `PostgreSQL/pgvector 기반 압축 얼굴 임베딩 검색에서 유사도 분포 보정을 통한 미등록 인물 거부 성능 개선`

대안:

1. `압축 얼굴 임베딩 검색에서 유사도 보정을 통한 미등록 인물 거부 성능 개선`
2. `압축 얼굴 임베딩의 Open-Set 검색을 위한 경량 점수 보정`
3. `PostgreSQL/pgvector 기반 얼굴 검색에서 임베딩 압축과 미등록 거부 성능 분석`

## 13. 최종 논문 주장

> 본 연구는 정렬된 얼굴 crop으로 구성된 PostgreSQL/pgvector 검색 환경에서 ArcFace 임베딩 압축이 유사도 분포와 open-set 임계값 안정성에 미치는 영향을 분석하고, top-1 유사도, score margin, 얼굴 품질, 템플릿 내부 분산, 압축 방식별 정규화 재구성 오차를 이용한 경량 calibration으로 압축 이후의 등록 인물 식별과 미등록 인물 거부 성능을 보존할 수 있는지 검증한다.

## 14. 연관 논문 우선순위

자세한 읽기 순서와 논문별 사용 위치는 [`docs/related_papers.md`](docs/related_papers.md)에 정리한다. 우선순위는 다음 축으로 정한다.

1. 본 연구의 핵심 주장인 open-set 미등록 거부와 압축 이후 score calibration에 직접 연결되는가
2. 실험 프로토콜, 지표, 데이터 분할을 설계하는 데 필수인가
3. ArcFace, FIQA, 템플릿 집계, PCA/PQ, HNSW를 이해하는 사전지식인가
4. 구현 또는 비교 baseline으로 바로 사용할 수 있는가
