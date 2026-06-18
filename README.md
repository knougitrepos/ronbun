# Thesis Image Retrieval Project

이 저장소는 **PostgreSQL/pgvector 기반 얼굴 검색에서 품질 적응형 템플릿 집계와 압축 인지형 미등록 인물 거부를 연구**하기 위한 석사논문 프로젝트입니다.

자세한 연구 방향은 [THESIS_RESEARCH_PLAN.md](THESIS_RESEARCH_PLAN.md)를 기준으로 합니다.

## 연구 초점

핵심 연구 질문은 다음과 같습니다.

> 품질 적응형 템플릿 집계와 압축 인지형 점수 보정은 압축된 ArcFace 임베딩 기반 얼굴 검색에서 등록 인물 식별과 미등록 인물 거부 성능을 동시에 보존할 수 있는가?

연구는 정렬된 얼굴 crop, 동결된 ArcFace, 공개 데이터셋을 사용합니다. 핵심 novelty는 **이상치 제거 및 품질 가중 템플릿 집계와 품질·템플릿 분산·압축 오차를 이용한 미등록 인물 거부 보정**입니다.

## 실험군

핵심 실험은 세 단계로 구성합니다.

| 단계 | 비교 항목 |
| --- | --- |
| 템플릿 집계 | Single, Mean, Outlier+Mean, 규칙 기반 품질 가중, FIQA 품질 가중 |
| 압축 | ArcFace 512D, PCA-256D, PostgreSQL 검색 가능한 강한 압축 프로파일 |
| 미등록 거부 | 전역 임계값, 압축별 임계값, 품질·압축 인지형 보정 |

PQ는 pgvector HNSW가 `LargeBinary` code를 직접 검색하지 못하므로 Faiss 또는 복원 오차 기반 보조 실험으로 분리합니다.

## 주요 구성

- `main.py`: FastAPI 애플리케이션 진입점
- `features/`: ArcFace 기반 임베딩 추출 및 벡터 변환 유틸리티
- `core/`: 설정, DB 스키마, 벡터 저장/변환 파이프라인
- `routes/`: 웹 라우트
- `similarity/`, `recall/`: 검색 및 평가 관련 유틸리티
- `notebooks/origin_extractor.ipynb`: ArcFace 원본 임베딩 추출 실험
- `notebooks/arcface_grad_cam.ipynb`: ArcFace 분석용 노트북
- `THESIS_RESEARCH_PLAN.md`: 논문 방향, novelty, 실험군, 평가 지표 정리
- `docs/superpowers/specs/2026-06-19-quality-compression-aware-face-search-design.md`: 승인된 연구 설계
- `docs/superpowers/plans/2026-06-19-quality-compression-aware-face-search.md`: 구현 계획

## 평가 지표

등록 검색:

- `Rank-1`
- `Rank-5`
- `mAP`

미등록 거부:

- `DIR@FPIR`
- `FNIR@FPIR`
- `Expected Calibration Error`
- `Brier Score`

시스템 효율:

- `Query Time`
- `Storage Size`
- `Index Build Time`

압축 및 강건성:

- `Compression Ratio`
- `Reconstruction Error`
- 압축 전후 식별 및 미등록 거부 성능
- 품질 구간별 성능

## 로컬 산출물 관리

다음 항목은 연구 소스가 아니라 로컬 산출물이므로 Git과 이전 대상에서 제외합니다.

- `.conda/`
- `downloaded_datasets/`
- `dbdata/`
- `mlruns/`
- `codebook/`
- `tmp/`
- `log/`
- `temp/`
- `static/extract_example/`

## 실행 개요

의존성 설치:

```powershell
pip install -r requirements.txt
```

PostgreSQL/pgvector 실행:

```powershell
docker compose up -d db
```

FastAPI 서버 실행:

```powershell
uvicorn main:app --reload
```

## 이전 위치

현재 연구 정리본은 `C:\thesis`에도 복사되어 있습니다. 해당 위치에는 Git 이력과 `thesis1` 브랜치가 함께 보존되어 있으며, 대용량 로컬 산출물은 제외되어 있습니다.
