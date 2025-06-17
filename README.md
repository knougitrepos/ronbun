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

## 1. 시스템 개요 및 구조
- ArcFace 등 임베딩 모델 기반 특징 추출, Wavelet/DCT/PCA/PQ 등 다양한 벡터 변환 지원
- PostgreSQL + pgvector 기반 대용량 벡터 저장/검색
- Optuna + MLflow 실험 자동화, Celery + Redis 분산/중단-재시작 지원
- 모든 변환/실험/저장 정책은 코드와 문서(wavelet_dct_notice.md 등)에 동기화
- 추상 클래스(FeatureExtractor, BaseTransformer) 기반 계층적 구조, 유지보수성 강화
- 폴더/파일/클래스/함수명은 기능 의미를 따르며, PEP8 및 일관성 준수

---

## 2. 환경 및 설치
- requirements.txt에 모든 필수 패키지 기록(추가시 반드시 동기화)
- 주요 패키지: fastapi, uvicorn, celery, redis, numpy, opencv-python, insightface, torch, pywavelets, faiss-cpu, mlflow, optuna 등

```bash
pip install -r requirements.txt
```

---

## 3. 실행 및 자동화
- run_server.cmd로 FastAPI 서버 실행(uvicorn main:app --reload --host 0.0.0.0 --port 5000)
- Redis, Celery 워커, batch_vector_transform_and_save.py로 분산/중단-재시작 지원
- batch_vector_transform_and_save.py 하나로 큐 등록/중복 체크/분산 처리 통합

---

## 4. 추상화 및 계층 구조
- features/embedding_service.py: FeatureExtractor(추상), ArcFaceFeatureExtractor(구현)
- core/pipeline/transformers/base.py: BaseTransformer(추상), wavelet/dct/pca/pq 등 각 변환기 구현
- core/pipeline/vector_pipeline.py: VectorPipeline에서 동적 변환/파라미터 처리, 유지보수성 강화

---

## 5. 변환 정책 및 예시
- DCT: keep_dim(128/256), mode('low'/'high'), add_embedding_128/256
- Wavelet: level(1/2/3), mode('low'/'high'), add_embedding_256
- PCA: n_components(0.95 등), add_embedding_128/256, joblib 코드북
- PQ: d/M/nbits, add_embedding_pq, faiss 코드북
- 모든 정책/예시는 wavelet_dct_notice.md에 상세 기술

---

## 6. DB 저장 정책
- Embedding_512: ArcFace origin(원본) 벡터, vector_type='origin', parameters={}
- Embedding_128/256: DCT, Wavelet, PCA 등 변환 벡터, vector_type/parameters로 구분
- EmbeddingPQ(권장): PQ 코드 저장, codes(LargeBinary/JSON), parameters로 구분
- VectorRepository에서 add_embedding_128/256/512/pq 등 함수 제공

---

## 7. 코드북 저장/로드 정책
- codebook/ 폴더에 변환명_차원(joblib/faiss) 규칙으로 저장
- PCA: joblib, PQ: faiss 공식 write_index/read_index
- 예시 코드 및 정책은 wavelet_dct_notice.md 참고

---

## 8. 실험 자동화/분산 처리
- Optuna + MLflow로 실험 자동화, 벡터 차원(128/256/512 등) 확장 지원
- Celery + Redis로 대량 이미지 분산 처리, 중단/재시작 안전
- 모든 큐/중복 체크/분산 처리 로직은 batch_vector_transform_and_save.py에 통합

---

## 9. 로그 및 에러 처리
- 모든 주요 함수/클래스에서 로그 출력 및 에러 메시지 명확화
- 로그는 실험 자동화, 분산 처리, DB 저장 등에서 일관되게 사용

---

## 10. 문서화 및 유지보수 원칙
- 코드 변경/구조/정책/실험/변환/DB/코드북 등 모든 내용은 README.md, wavelet_dct_notice.md 등 문서에 즉시 반영
- 코드와 문서가 항상 동기화되도록 관리
- 폴더/파일/클래스/함수명은 기능 의미를 따르며, 계층적/추상화 구조로 유지보수성 강화

---

## 11. 예시 코드/실행
- 각 변환/저장/코드북/DB/실험 자동화 예시는 README.md 및 wavelet_dct_notice.md에 상세 기술
- 실제 사용 예시는 scripts/batch_vector_transform_and_save.py, core/pipeline/vector_pipeline.py 등 참고

---

### [적용 방법]
- 위 내용을 README.md, wavelet_dct_notice.md에 반영/정리
- requirements.txt, run_server.cmd 등 환경 파일도 항상 동기화
- 새로운 기능/정책/구조 추가시 반드시 문서에 즉시 반영

문의/확장/실험 관련 문의는 언제든 환영합니다.

---

## 12. 실험/운영 단계별 전체 절차 및 실행 가이드

### 1. 사전 준비
- **환경 준비**: requirements.txt 설치, DB(PostgreSQL) 실행, config.yaml 등 환경설정 완료
- **DB 준비**: PostgreSQL 서버가 반드시 실행 중이어야 함
- **Celery 워커 실행**: (터미널에서)
  ```bash
  celery -A core.pipeline.vector_tasks worker --loglevel=info
  ```
  - 반드시 위와 같이 -A 옵션에 태스크가 정의된 모듈을 지정해야 함
  - core/celery_app.py에서 vector_tasks 임포트하지 않도록 유지(순환참조 방지)
  - 태스크 정의 파일(vector_tasks.py)에서는 from core.celery_app import celery_app만 사용
- **벡터 변환/특징 추출 작업 등록**:
  - (1) **터미널에서 직접 실행**
    - **유의:** 반드시 프로젝트 루트(ronbun)에서 실행하거나, 아래처럼 PYTHONPATH를 명시적으로 지정해야 내부 모듈 import 에러가 발생하지 않습니다.
    - 예시(Git Bash/리눅스/WSL):
      ```bash
      export PYTHONPATH=$(pwd)
      python scripts/batch_vector_transform_and_save.py
      ```
    - 예시(Windows CMD):
      ```cmd
      set PYTHONPATH=%cd%
      python scripts/batch_vector_transform_and_save.py
      ```
  - (2) **웹에서 실행**: [벡터 변환 작업 실행] 버튼 클릭 또는 `/automation/run-batch-transform` (POST)
  - 이 작업은 대량 이미지의 특징 추출/벡터 변환을 Celery 큐에 등록합니다.
  - **자동 추출/저장 대상:**
    - ArcFace 원본(512차원)
    - Wavelet(level=1~3, mode=low/high) → 6가지 조합
    - DCT(keep_dim=32/64/128/256, mode=low/high) → 8가지 조합
    - PCA(128/256)
    - PQ(128/256, M=16, nbits=8)
  - **파라미터를 변경하고 싶으면 소스코드 내 리스트(WAVELET_LEVELS, DCT_KEEP_DIMS 등)를 직접 수정해야 함**
  - **Celery 워커가 반드시 실행 중이어야 실제 작업이 처리됨**
- **MLflow UI 실행**: (터미널에서)
  ```bash
  mlflow ui --port 5001 --backend-store-uri ./mlruns --host 127.0.0.1
  ```
  또는 웹에서 "MLflow UI 실행" 버튼 사용 (구현된 경우)

### 2. 특징 추출 및 벡터 변환 작업 등록
- **웹에서**: [벡터 변환 작업 실행] 버튼 클릭 (또는 `/automation/run-batch-transform` POST)
- **내부 동작**: scripts/batch_vector_transform_and_save.py가 실행되어, Celery 큐에 대량 이미지 특징 추출/변환 작업이 등록됨
- **Celery 워커가 반드시 실행 중이어야 실제 작업이 처리됨**
- **진행상황 확인**: [작업 상태 모니터링] 버튼 또는 `/automation/status`에서 각 변환별 진행률 확인

### 3. 특징 추출/변환 결과 저장
- **DB에**: ArcFace 원본, DCT, Wavelet, PCA, PQ 등 다양한 벡터가 자동 저장됨
- **코드/정책**: 변환별 파라미터, 저장 테이블, 예시 코드는 wavelet_dct_notice.md 참고

### 4. 실험 자동화 및 recall 테스트
- **실험 자동화(Optuna+MLflow)**: (터미널에서)
  ```bash
  python optuna_mlflow/run_optuna.py
  ```
- **실험 결과**: mlruns/ 폴더에 기록, MLflow UI(웹)에서 시각적으로 비교/분석 가능
- **실험 파라미터/정책**: optuna_mlflow/experiment.py, run_optuna.py 등 참고

### 5. 기타
- **DB/코드북/실험 정책**: README.md, wavelet_dct_notice.md, MLFLOW_instructions.md 등 문서 참고
- **운영상 주의**: Celery 워커, DB, MLflow UI 등은 항상 별도 프로세스/터미널에서 실행되어야 함
- **문제 발생 시**: 로그, 상태 모니터링, 에러 메시지, DB 상태 등 확인

---

### [Celery 워커 태스크 등록 및 에러 해결 실전 가이드]

#### 1. 반드시 아래와 같이 Celery 워커를 실행해야 태스크가 정상 등록됨
```bash
celery -A core.pipeline.vector_tasks worker --loglevel=info
```
- `-A` 옵션에 태스크가 정의된 모듈(`core.pipeline.vector_tasks`)을 지정해야 함
- core/celery_app.py에서는 태스크 임포트하지 않음(순환참조 방지)
- 태스크 정의 파일(vector_tasks.py)에서는 from core.celery_app import celery_app만 사용

#### 2. [tasks] 아래에 `core.pipeline.vector_tasks.process_image`가 보이면 정상 등록된 것임
- Celery 워커 실행 로그에서 `[tasks]` 아래에 태스크가 보이지 않으면 등록이 안 된 것임

#### 3. 'Received unregistered task of type ...' 에러가 발생하는 경우
- 워커가 해당 태스크를 등록하지 않은 상태에서 큐에 등록된 작업을 받아서 생기는 현상임
- 주로 아래와 같은 원인
  - 워커 실행 시 -A 옵션을 잘못 지정 (예: core.celery_app 등)
  - 코드 수정 후 워커를 재시작하지 않음
  - PYTHONPATH/실행 위치가 불일치

#### 4. 해결 방법
- 반드시 워커를 아래 명령어로 실행
  ```bash
  celery -A core.pipeline.vector_tasks worker --loglevel=info
  ```
- 코드 수정(특히 태스크 정의/등록 구조 변경) 후 워커를 재시작
- 큐 등록 스크립트도 같은 PYTHONPATH에서 실행 (예: export PYTHONPATH=$(pwd))
- 워커와 큐 등록 스크립트 모두 프로젝트 루트에서 실행 권장

#### 5. 실전 운영 체크리스트
- [ ] 워커 실행 명령어가 정확한지 확인
- [ ] 워커 실행 로그에 태스크가 등록되어 있는지 확인
- [ ] 큐 등록 스크립트와 워커의 PYTHONPATH/실행 위치가 일치하는지 확인
- [ ] 에러 발생 시 위 항목을 모두 점검

---

### [운영 중인 모든 Celery 프로세스 종료(강제 kill) 방법]

#### Windows에서 모든 celery 프로세스 종료
```cmd
# PowerShell
Get-Process celery | Stop-Process -Force

# 또는 명령 프롬프트(cmd)
taskkill /F /IM celery.exe
```

#### Linux/WSL/Mac에서 모든 celery 프로세스 종료
```bash
pkill -f 'celery'
# 또는
ps aux | grep celery
# 위 명령으로 PID 확인 후
kill -9 <PID>
```

- 여러 워커/백그라운드 celery 프로세스가 남아 있을 때 위 명령으로 모두 종료 가능
- 작업 중인 큐/워커가 모두 종료되므로, 재시작 전 반드시 필요한 작업이 없는지 확인

---
