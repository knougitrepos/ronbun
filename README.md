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

# 얼굴 임베딩/벡터 변환/실험 자동화 시스템

## 주요 특징
- ArcFace 등 임베딩 모델 기반 특징 추출
- Wavelet, DCT 등 다양한 벡터 변환 지원
- DB(PGVector) 기반 대용량 벡터 저장/검색
- Optuna+MLflow 실험 자동화
- Celery+Redis 기반 분산/중단-재시작 지원
- 모든 큐 등록/중복 체크/분산 처리 통합 (batch_vector_transform_and_save.py)

---

## 1. 설치 및 환경 준비

```bash
pip install -r requirements.txt
```
- requirements.txt에 Celery, Redis, numpy, opencv-python 등 필수 패키지 포함

---

## 2. 실행 방법 (분산/중단-재시작 지원)

1. **Redis 실행**
2. **Celery 워커 실행**
   ```bash
   celery -A core.celery_app.celery_app worker --loglevel=info
   ```
3. **벡터 변환/DB 저장 작업 등록**
   ```bash
   python scripts/batch_vector_transform_and_save.py
   ```
   - 중단/재시작 시에도 이 파일만 다시 실행하면 미처리만 이어서 처리됨

---

## 3. 주요 변환 정책 요약

| 변환 종류   | transform_method | 저장 차원 | 저장 함수           | 파라미터 예시           | 비고                |
|:-----------|:----------------|:----------|:--------------------|:------------------------|:--------------------|
| ArcFace    | None            | 512       | add_embedding_512   | -                       | 원본                |
| Wavelet    | 'wavelet'       | 256       | add_embedding_256   | level, mode             | 지원/저장           |
| DCT        | 'dct'           | 128, 256  | add_embedding_128/256| keep_dim, mode         | 지원/저장           |
| PCA        | 'pca'           | (미정)    | (미정)              | n_components 등         | 구조상 확장 가능    |
| PQ         | 'pq'            | (미정)    | (미정)              | n_subvectors, n_bits 등 | 구조상 확장 가능    |

- Wavelet, DCT는 실제 저장/지원, PCA, PQ는 구조상 확장 가능(현재 스크립트 미사용)
- 자세한 변환 정책은 wavelet_dct_notice.md 참고

---

## 4. 폴더/파일 구조

```
/project-root
├── core/                 # 핵심 ML/DB/설정/파이프라인
├── scripts/              # 벡터 변환/DB 저장 스크립트 (batch_vector_transform_and_save.py)
├── experiments/          # 실험 자동화(Optuna+MLflow)
├── requirements.txt
├── README.md
└── ...
```

---

## 5. 기타 참고
- submit_vector_tasks.py는 더 이상 사용하지 않으므로 삭제됨
- batch_vector_transform_and_save.py 하나로 큐 등록/중복 체크/분산 처리 통합
- 실험 자동화, 변환 정책, DB 구조 등은 README와 wavelet_dct_notice.md 참고

---

문의/확장/실험 관련 문의는 언제든 환영합니다.