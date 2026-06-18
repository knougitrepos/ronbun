# 품질 및 압축 인지형 얼굴 검색 연구 설계

## 1. 목적

정렬된 얼굴 crop을 사용하는 PostgreSQL/pgvector 검색 시스템에서 다음 문제를 해결한다.

1. 동일 인물의 등록 이미지가 여러 장일 때 저품질 이미지와 이상치가 대표 임베딩을 오염시키는 문제
2. PCA, 저정밀 표현 또는 PQ가 유사도 분포를 변경하여 미등록 인물 거부 임계값을 불안정하게 만드는 문제
3. 모든 identity에 동일한 압축률을 적용하여 쉬운 템플릿에는 저장공간을 낭비하고 어려운 템플릿에는 식별 정보를 과도하게 손실하는 문제

ArcFace는 동결하고, 경량 통계 처리와 calibration만 학습한다.

## 2. 연구 가설

- H1: 이상치 제거 후 품질 가중 평균은 단순 평균보다 다중 등록 템플릿의 Rank-1과 open-set DIR을 개선한다.
- H2: 압축 프로파일별 고정 임계값보다 품질·분산·압축 오차를 함께 사용하는 보정 모델이 목표 FPIR에서 높은 DIR을 제공한다.
- H3: 템플릿 품질과 내부 분산에 따라 제한된 압축 프로파일을 선택하면 균일 압축보다 동일 저장공간에서 높은 식별 성능을 제공한다.
- H4: FIQA 품질 점수와 규칙 기반 품질 점수는 데이터 품질 구간에 따라 서로 다른 장단점을 보인다.

## 3. 시스템 경계

### 포함

- 공개 얼굴 데이터셋
- 정렬된 얼굴 crop
- ArcFace 512D 임베딩
- 인물당 1장 또는 다중 등록 템플릿
- 등록 인물 1:N 검색
- 미등록 인물 거부
- PostgreSQL/pgvector exact 및 HNSW 검색
- PCA와 PostgreSQL에서 검색 가능한 저정밀 표현
- PQ 보조 실험

### 제외

- ArcFace 재학습
- 대형 딥러닝 calibration 모델
- 일반 이미지 모델의 배경 편향
- segmentation 기반 영역 가중
- 얼굴 검출이 포함된 원본 장면 검색
- 실시간 영상 추적

## 4. 데이터 모델

### Image record

- image ID
- identity ID
- source dataset
- split
- file path
- ArcFace embedding
- 규칙 기반 품질 점수와 세부 지표
- FIQA 점수

### Identity template

- identity ID
- 집계 방식
- 사용된 이미지 ID
- 제외된 이상치 이미지 ID
- 등록 이미지 수
- 평균 품질
- 내부 분산
- 대표 임베딩
- 압축 프로파일
- 압축 재구성 오차

### Search observation

- probe ID
- known/unknown 여부
- top-1 identity와 score
- top-2 score와 margin
- probe quality
- matched template statistics
- compression profile
- calibrated registration probability
- 최종 identity 또는 unknown

## 5. 구성요소

### 5.1 Dataset protocol

identity 단위로 development, calibration, test를 분리한다. Test identity나 이미지는 PCA/PQ, 품질 결합식, 보정 모델 또는 임계값 선택에 사용하지 않는다.

### 5.2 Quality scoring

모든 scorer는 `score(image, face_metadata) -> [0, 1]` 인터페이스를 따른다.

- Rule-based scorer: 해상도, Laplacian blur, detection score, yaw/pitch
- FIQA scorer: 사전학습 모델을 고정된 PyTorch/ONNX adapter로 실행

각 raw 품질 값은 development split의 분위수로 정규화하고 calibration/test에는 고정된 변환을 적용한다.

### 5.3 Robust aggregation

1. L2 정규화된 임베딩의 pairwise cosine distance를 계산한다.
2. 총 거리의 중앙값이 최소인 medoid를 선택한다.
3. medoid 거리에 median absolute deviation 기준을 적용해 이상치를 제거한다.
4. retained sample이 1개이면 해당 벡터를 사용한다.
5. 2개 이상이면 균등 또는 품질 가중 평균 후 L2 정규화한다.

품질 가중치는 `softmax(q_i / temperature)`를 사용하고 temperature는 development split에서 선택한다.

### 5.4 Compression

검색 가능한 압축 프로파일은 독립된 테이블 또는 명시적인 profile column으로 관리한다.

- `arcface_512_f32`
- `pca_256_f32`
- `pca_256_f16` 또는 현재 pgvector 버전에서 지원되는 동등한 profile

PQ는 별도의 Faiss 평가기로 관리한다. PQ code byte와 codebook byte를 저장공간에 포함한다. PQ 복원 벡터를 pgvector에 저장한 경우 검색 편의를 위한 복원 표현임을 명시하고 PQ 저장량으로 계산하지 않는다.

### 5.5 Search

각 profile에 대해 exact search 결과를 기준값으로 저장한다. HNSW는 동일 distance와 동일 Top-K 조건에서 평가한다. profile을 혼합하는 적응형 정책은 profile별 후보 검색 후 원본 cosine과 호환되는 보정 score로 통합한다.

### 5.6 Unknown rejection calibration

보정 feature는 다음으로 고정한다.

- top-1 cosine similarity
- top-1/top-2 margin
- probe quality
- template mean quality
- template dispersion
- enrollment count
- compression profile one-hot
- reconstruction error

첫 구현은 표준화된 feature를 입력으로 받는 L2-regularized logistic regression을 사용한다. 비교군은 global threshold와 per-profile threshold다.

Calibration split에서 목표 FPIR별 threshold를 선택하고 Test split에서 고정한다.

### 5.7 Adaptive compression policy

세 개 이하의 profile만 사용한다. Development split에서 quality, dispersion, enrollment count에 대한 threshold grid를 검색한다.

목적함수는 다음 우선순위를 따른다.

1. 목표 FPIR에서 DIR 손실 상한 만족
2. Rank-1 손실 상한 만족
3. 평균 template byte 최소화

정확도 제약을 만족하지 못하면 더 약한 압축으로 fallback한다.

## 6. 오류 처리

- 얼굴 검출 실패 이미지는 manifest에 실패 사유를 기록하고 제외한다.
- NaN 또는 zero-norm 임베딩은 저장하지 않는다.
- FIQA 모델이 없거나 checksum이 다르면 실험을 중단한다.
- identity에 유효 이미지가 하나도 없으면 해당 identity를 등록하지 않는다.
- 이상치 제거 후 모든 이미지가 제거되지 않도록 medoid는 항상 유지한다.
- PCA/PQ 모델은 development split에서만 fit되었는지 metadata로 검증한다.
- calibration model의 입력 feature schema가 다르면 실행을 중단한다.

## 7. 평가 설계

### 등록 검색

- Rank-1/5
- CMC
- mAP

### 미등록 거부

- DIR@FPIR
- FNIR@FPIR
- FPIR
- ECE
- Brier score

### 시스템

- P50/P95 latency
- index build time
- table/index byte
- vector byte
- HNSW recall against exact search

### 분석 축

- enrollment count 1/2/5/mixed
- probe quality tertile
- template quality tertile
- aggregation method
- compression profile
- rejection method

## 8. 검증 전략

- 수학·통계 단위는 synthetic embedding으로 단위 테스트한다.
- 데이터 누수 방지는 manifest validation test로 검증한다.
- SQL 저장과 검색은 작은 PostgreSQL integration fixture로 검증한다.
- full experiment 전에 20 identity smoke dataset으로 end-to-end 검증한다.
- 동일 probe 결과를 사용한 paired bootstrap으로 95% 신뢰구간을 계산한다.

## 9. 승인된 구현 순서

1. 데이터 manifest와 split validator
2. embedding/quality extraction
3. robust template aggregation
4. compression profiles
5. exact/HNSW search evaluator
6. open-set protocol
7. compression-aware calibrator
8. adaptive compression policy
9. experiment runner와 결과 표

## 10. 성공 기준

연구 성공은 제안 방법이 항상 모든 지표에서 최고인 것으로 정의하지 않는다.

다음 중 하나 이상을 통계적으로 확인하면 유효한 연구 결과로 본다.

- 동일 FPIR에서 DIR 개선
- 동일 Rank-1 허용 손실에서 저장공간 절감
- 저품질 probe에서 unknown 오수락 감소
- FIQA와 규칙 기반 점수의 적용 조건 차이 규명
- 압축 프로파일별 score shift와 threshold drift 정량화
