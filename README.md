# Thesis Image Retrieval Project

이 저장소는 **PostgreSQL/pgvector 기반 인물 이미지 검색에서 얼굴 전용 임베딩의 압축 가능성과 검색 효율을 분석**하기 위한 연구 프로젝트입니다.

자세한 연구 방향은 [THESIS_RESEARCH_PLAN.md](THESIS_RESEARCH_PLAN.md)를 기준으로 합니다.

## 연구 초점

핵심 연구 질문은 다음과 같습니다.

> PostgreSQL/pgvector 기반 인물 이미지 검색에서 ArcFace 임베딩의 PCA/PQ 압축은 검색 정확도를 유지하면서 저장공간과 질의시간을 줄일 수 있는가?

본 연구의 novelty는 새로운 정규화 수식을 주장하는 것이 아니라, **얼굴 전용 임베딩을 실제 관계형 데이터베이스 검색 환경에서 압축했을 때의 정확도-속도-저장공간 trade-off를 정량적으로 분석**하는 데 있습니다.

## 실험군

핵심 실험군은 4개로 제한합니다.

| 실험군 | 목적 |
| --- | --- |
| ArcFace 512D | 얼굴 전용 임베딩 원본 기준선 |
| ArcFace PCA-256D | 실용적 차원 축소 기준선 |
| ArcFace PCA-256D + PQ | 강한 압축에서 정확도와 효율 trade-off 확인 |
| General Model 512D | 일반 이미지 모델 ablation 및 실패 양상 비교 |

DCT/Wavelet 및 autoencoder 계열 실험은 본 논문의 핵심 실험군에서 제외하고, 필요 시 향후 연구나 부록 후보로만 다룹니다.

## 주요 구성

- `main.py`: FastAPI 애플리케이션 진입점
- `features/`: ArcFace 기반 임베딩 추출 및 벡터 변환 유틸리티
- `core/`: 설정, DB 스키마, 벡터 저장/변환 파이프라인
- `routes/`: 웹 라우트
- `similarity/`, `recall/`: 검색 및 평가 관련 유틸리티
- `notebooks/origin_extractor.ipynb`: ArcFace 원본 임베딩 추출 실험
- `notebooks/arcface_grad_cam.ipynb`: ArcFace 분석용 노트북
- `THESIS_RESEARCH_PLAN.md`: 논문 방향, novelty, 실험군, 평가 지표 정리

## 평가 지표

검색 정확도:

- `Recall@1`
- `Recall@5`
- `Recall@K`
- `mAP`

시스템 효율:

- `Query Time`
- `Storage Size`
- `Index Build Time`

압축 분석:

- `Compression Ratio`
- `Accuracy Loss`
- 원본 대비 성능 유지율

보조 통합 지표:

```text
A_norm = Recall_compressed / Recall_original
T_norm = Time_original / Time_compressed
S_norm = Storage_original / Storage_compressed

NRES = A_norm^0.5 * T_norm^0.25 * S_norm^0.25
```

`NRES`는 핵심 novelty가 아니라 결과를 요약하기 위한 보조 지표로 사용합니다.

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
