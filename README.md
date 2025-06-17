### [2024-06-09] 구조 개선 1단계 진행중
- FeatureExtractor 추상 클래스 및 ArcFaceFeatureExtractor 구현, embedding_service.py 통합 리팩토링 완료
- origin_feature_extractor.py는 더 이상 사용하지 않음(구조 안내 주석만 유지)
- 다양한 임베딩 모델 확장 가능 구조로 변경됨

1. 이미지에서 특징 추출 arcface 사용
2. wavelet,dct,pca,pq quantize로 특징 변환
3. recall 등의 기준으로 정확도 판단
4. PostgreSQL 의 pgvector 를 사용하여
cosine 유사도 비교로 업로드된 이미지를 유사도 판단
5. sota 모델과의 개선 및 변화사항 비교 파악
6. 위와 같은 연구용 파이프라인 관리용 웹임

### [2024-06-09] 구조 개선 2단계 진행중
- vector_transformer.py 신규 생성, PCA/DCT/Wavelet/Quantize/정규화 등 벡터 변환 함수 통합
- 기존 zscore_normalization.py, L2_normarlization.py는 안내 주석만 유지(기능 통합)
- transform_vector 함수로 파라미터 기반 동적 변환 지원

### [2024-06-09] 구조 개선 3단계 진행중
- DB/db_utils.py에 add_image_embedding 함수 추가, 여러 벡터/메타 정보 일괄 저장 구조로 개선
- ImageEmbeddings 테이블의 origin/pca/dct/wavelet/quantized 벡터와 각종 params, 메타 정보 저장 일관성 강화

### [2024-06-09] 구조 개선 4단계 진행중
- similarity_utils.py에 search_similar_vectors 함수 추가, cosine/euclidean/inner product 등 다양한 거리 계산 방식 및 HNSW/Brute Force 전략 분기 구조 설계
- metric 파라미터 및 K값 기반 자동 전략 선택 구조 반영(실제 DB 연동/인덱스는 후속 구현)

### [2024-06-09] 구조 개선 5단계 진행중
- extract_router.py에 FastAPI BackgroundTasks 적용, 대량 특징추출 비동기 처리 구조로 개선
- 실제 추출/DB저장 로직은 서비스 함수로 분리, 라우트는 트리거 역할만 담당

### [2024-06-09] 구조 개선 6단계 진행중
- 중복/불필요 파일(origin_feature_extractor.py, zscore_normalization.py, L2_normarlization.py) 삭제
- 명명 규칙(영문, PEP8) 및 주석/문서화 정리, 코드 일관성 강화

### [2024-06-09] 구조 개선 7단계 진행중
- config.yaml 기반 실험 조합 구조 설계, config.py에 예시 구조/주석 추가
- Hydra 등 설정 관리 도구 연동 및 실험 파라미터 동적 로딩 구조 반영(후속 적용 가능)

### [2024-06-09] 구조 개선 8단계 진행중
- 마이크로서비스 아키텍처, Docker 기반 컨테이너화, API 표준화 등 장기 확장성 구조 설계 방향 반영
- 얼굴 검출/임베딩/검색 등 서비스 분리, 컨테이너화 및 표준 API 명세화 예시 문서화 예정

### [2024-06-10] Embedding_128 지원 및 실험 자동화 연동
- Embedding_128 테이블 및 저장/조회 기능(core/schemas.py, core/database.py) 추가
- 128차원 벡터(DCT, PCA 등)도 256/512와 동일하게 저장/조회/실험 자동화 가능
- 실험 자동화(Optuna+MLflow) 구조상 벡터 차원(128/256/512 등) 확장에 문제 없음
- 예시: repository.add_embedding_128(...), repository.get_embeddings_128(...) 사용

### [ArcFace origin(원본) 벡터 저장 원칙]
- ArcFace 등 임베딩 모델의 원본 벡터(origin)는 Embedding_512 테이블에 저장
- vector_type='origin', parameters={}로 구분하여 저장(예시 코드 참고)
- 변환 벡터(DCT, Wavelet 등)와 동일한 구조로 관리되어 실험 자동화/검색/확장에 유리

# 프로젝트 구조 및 실험 자동화 가이드

## 1. 전체 개요 및 목적

- 다양한 임베딩 모델(ArcFace 등)과 벡터 변환(PCA, DCT, Wavelet, Quantize 등), PostgreSQL+pgvector 기반 DB, Optuna+MLflow 실험 자동화 등 최신 연구/서비스 구조를 반영한 얼굴 임베딩/검색 시스템입니다.
- 기능별 계층적 구조, 전략 패턴, DAO/Repository, config.yaml 기반 설정, 실험 자동화, 확장성/유지보수성에 중점을 둡니다.

---

## 2. 실험 자동화 전/후 필수 과정

### 2-1. 실험 자동화 전(사전 준비)
- **모든 이미지에 대해 특징 추출 및 다양한 벡터 변환(PCA, DCT, Wavelet 등)을 미리 수행**
- 실험에 필요한 모든 파라미터 조합(level, mode, keep_dim 등)에 맞는 벡터를 **DB에 저장**
- (예: vec_origin, vec_pca, vec_dct, vec_wavelet, ... 필드 활용)
- **이유:**  
  - 실험 자동화(Optuna/MLflow)에서 빠르고 일관성 있게 평가하려면, 변환된 벡터가 DB에 미리 준비되어 있어야 함  
  - 실험마다 변환/저장하면 속도가 느려지고, 결과의 일관성/재현성이 떨어짐

### 2-2. 실험 자동화/평가(Optuna + MLflow)
- Optuna trial마다 DB에서 해당 파라미터 조합(level, mode 등)에 맞는 벡터를 꺼내와 검색/평가
- 결과(Recall, 쿼리 시간 등)를 MLflow에 기록, 다양한 조합을 자동 반복 실험
- **이유:**  
  - 실험 자동화는 "검색/평가"에만 집중, 변환/저장 오버헤드 없이 빠르고 신뢰성 있게 반복 가능

---

## 3. 전체 실행 방법

### 3-1. 벡터 변환 및 DB 저장 (사전 준비)
```bash
python scripts/batch_vector_transform_and_save.py
```
- 모든 이미지에 대해 특징 추출 및 다양한 변환(level, mode 등) 결과를 DB에 저장

### 3-2. 실험 자동화/평가
```bash
python experiments/run_optimization.py
mlflow ui --port 5001
```
- 실험 자동화(Optuna + MLflow) 실행, MLflow UI에서 실험별 결과 비교

---

## 4. 디렉토리 구조

```
/project-root
├── app/                  # FastAPI 웹 애플리케이션
├── core/                 # 핵심 ML/DB 로직
│   ├── config.yaml       # 설정 파일
│   ├── config.py         # 설정 로더
│   ├── database.py       # DAO/Repository
│   └── pipeline/
│       ├── vector_pipeline.py  # 객체지향 파이프라인
│       └── transformers/ # 전략 패턴 기반 변환기
├── experiments/          # 실험 자동화 스크립트
├── data/                 # 데이터셋
├── tests/                # 단위/통합 테스트
├── requirements.txt
└── README.md
```

---

## 5. 설정 관리
- 모든 실험/운영/개발 파라미터는 `core/config.yaml`에서 일원 관리
- `ConfigLoader`로 코드에서 손쉽게 접근

---

## 6. ML 파이프라인 및 DB 추상화

- `VectorPipeline` 클래스 기반, 각 단계별 메서드로 체인 호출
- 벡터 변환은 `BaseTransformer`/`PCATransformer`/`DCTTransformer`/`WaveletTransformer` 등 전략 패턴 적용
- **Wavelet, DCT 변환은 level(분해 레벨), mode(고주파/저주파), keep_dim 등 다양한 파라미터 조합 실험이 가능하도록 설계됨**
- DCT, PCA 등 128/256/512 등 다양한 차원 벡터를 실험/저장/검색에 활용 가능
- `VectorRepository`(DAO/Repository 패턴)로 DB 연동 캡슐화

---

## 7. 실험 자동화/MLflow

- `experiments/run_optimization.py`에서 Optuna + MLflow 기반 실험 자동화
- 파라미터/메트릭/아티팩트 기록, MLflow UI로 실험 비교
- **Wavelet, DCT level/mode 등 다양한 파라미터 조합 실험 자동화 지원**

---

## 8. Embedding_128 등 다양한 차원 벡터 사용 방법

### 8-1. 원본(ArcFace) 벡터 저장 예시
```python
# ArcFace origin(원본) 벡터 저장
repository.add_embedding_512(
    image_id=1,
    vector_type='origin',
    parameters={},
    embedding=arcface_vector,  # 512차원
    created_at=...
)
```

### 8-2. 벡터 조회 예시
```python
# 128차원 벡터 조회
results = repository.get_embeddings_128(image_id=1, vector_type='dct', param_filter={'keep_dim': 128})
for emb in results:
    print(emb.embedding)
```

### 8-3. 실험 자동화에서 활용
- 실험 파이프라인에서 128/256/512 등 원하는 차원의 벡터를 DB에서 조회하여 평가에 활용
- 예시:
```python
# objective 함수 내에서
vecs_128 = repository.get_embeddings_128(...)
# 벡터 검색/평가 후 mlflow.log_metric 등 기록
```

### 8-4. batch_vector_transform_and_save.py에서 자동 저장
- DCT_KEEP_DIMS = [32, 64, 128] 등으로 128차원 벡터도 자동 저장
- repository.add_embedding_128(...) 호출로 일괄 저장

---

## 9. 참고/확장

- 벡터 변환/저장 자동화 스크립트, DB 스키마, 실험 자동화/DB 연동 예시는 core/ 및 experiments/ 디렉토리 참고
- config.yaml에서 실험 파라미터/DB 정보 등 일원 관리
- 확장/실험/운영이 모두 용이한 구조

---

이 구조를 기반으로 확장, 실험, 운영이 모두 용이합니다.