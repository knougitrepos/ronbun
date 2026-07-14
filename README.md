# Compressed Open-Set Face Search Thesis

이 저장소는 `PostgreSQL/pgvector`에 저장한 압축 ArcFace 임베딩의 검색 성능, 저장 효율, open-set 미등록 인물 거부 성능을 재현 가능하게 비교하기 위한 석사논문 실험 코드입니다. 연구 방향의 기준 문서는 [THESIS_RESEARCH_PLAN.md](THESIS_RESEARCH_PLAN.md)입니다.

핵심 질문은 다음과 같습니다.

> 임베딩 압축으로 변형된 유사도 분포와 거부 임계값을 경량 calibration으로 보정하여 등록 인물 식별과 미등록 인물 거부 성능을 함께 보존할 수 있는가?

## 연구 범위

- 동결된 `ArcFace/InsightFace`로 512차원 원본 임베딩을 추출합니다.
- pgvector에서 직접 비교 가능한 기준선은 원본 512D와 PCA 256D입니다.
- `Product Quantization(PQ)` code는 `LargeBinary`이며 pgvector HNSW가 직접 검색할 수 없습니다. PQ는 Faiss 검색 또는 복원 오차 분석용 보조 실험으로 분리합니다.
- 보정의 기본 모델은 BCE 기반의 경량 모델입니다. 대규모 모델 재학습, Beta-VAE, 웹 서비스, MLOps 파이프라인은 연구 범위가 아닙니다.
- 등록 probe, known unknown, unknown unknown을 분리하여 평가합니다.

## 디렉터리 구조

| 경로 | 역할 |
| --- | --- |
| `configs/database.yaml` | Git에 기록하는 비밀정보 없는 DB 기본 설정 |
| `configs/database.local.yaml` | 로컬 DB 비밀번호·설정 override. Git에서 제외 |
| `configs/experiments/` | 실험 조건과 고정된 프로토콜 설정 |
| `research/database/` | DB 설정 로딩, 연결, 스키마, 저장소 |
| `research/embeddings/` | InsightFace 로딩과 ArcFace 임베딩 추출 |
| `research/compression/` | PCA 및 보조 PQ 학습·변환 |
| `research/search/` | open-set 검색과 압축 오차 인증 |
| `research/protocols/` | identity-disjoint 데이터 분할 |
| `research/calibration/` | 임계값 선택과 경량 거부 보정 |
| `research/evaluation/` | 식별, FPIR/DIR, calibration 지표 |
| `research/templates/` | 템플릿 집계 ablation |
| `research/runtime/` | 날짜·회차별 실행 기록과 안정적인 실행 코드 |
| `notebooks/` | 순서대로 실행하는 여섯 개의 실험 runbook |
| `runs/` | 실행별 manifest, 로그, phase 산출물. Git에서 제외 |
| `results/paper/` | 검증 후 논문 표·그림으로 선별한 결과 |

복잡하거나 재사용되어야 하는 계산, DB 처리, 압축, 검색, 지표 코드는 `research/`의 `.py`에 둡니다. 노트북은 설정을 고정하고 각 단계를 호출하며 결과를 확인하는 얇은 실행 절차입니다.

## 환경 준비

프로젝트 루트에서 다음 명령을 실행합니다.

```powershell
py -m pip install -r requirements.txt
```

DB 기본값은 `configs/database.yaml`에 있습니다. 비밀번호는 환경 변수 사용을 권장합니다.

```powershell
$env:RONBUN_DB_PASSWORD = "<local-password>"
py scripts/setup_postgres_pgvector.py --dry-run
py scripts/setup_postgres_pgvector.py
```

필요하면 Git에서 제외되는 `configs/database.local.yaml`에 로컬 override를 둘 수 있습니다. 환경 변수 `RONBUN_DB_HOST`, `RONBUN_DB_PORT`, `RONBUN_DB_NAME`, `RONBUN_DB_USER`, `RONBUN_DB_PASSWORD`가 가장 높은 우선순위를 갖습니다. 로그와 manifest에는 비밀번호를 기록하지 않습니다.

전체 설정을 실행하지 않고 검사하려면 다음 CLI를 사용합니다.

```powershell
py experiments/run_face_search_study.py `
  --config configs/experiments/face_search.yaml `
  --dry-run
```

## 데이터 매니페스트 준비

| 데이터셋 | 준비 노트북 | 기본 출력 폴더 |
| --- | --- | --- |
| LFW deep-funneled | `notebooks/data_preparation/prepare_lfw_manifest.ipynb` | `data/interim/lfw/` |
| QMUL-SurvFace-v1 | `notebooks/data_preparation/prepare_survface_manifest.ipynb` | `data/interim/survface/` |

두 노트북 모두 기본 `WRITE_OUTPUTS=False`에서 전체 데이터와 프로토콜을 검사하되 파일은 저장하지 않습니다. 검증 통과 후 `True`로 바꾸어 위에서 아래로 다시 실행합니다. LFW 결과는 일반 00 노트북에 연결할 수 있습니다. SurvFace 결과는 공식 gallery/mated/unmated 역할을 보존하므로, gallery를 다시 표본추출하는 일반 00 노트북 대신 전용 공식 프로토콜 어댑터에서 사용해야 합니다.

## 노트북 실행 순서

| 순서 | 노트북 | 산출물 |
| --- | --- | --- |
| 00 | `00_protocol_and_run_freeze.ipynb` | 데이터 분할, 설정 hash, 새 run 고정 |
| 01 | `01_arcface_embedding_extraction.ipynb` | 원본 ArcFace 임베딩과 추출 메타데이터 |
| 02 | `02_compressor_fit.ipynb` | development split으로 학습한 PCA/PQ 프로파일 |
| 03 | `03_compressed_materialization_and_index.ipynb` | 압축 벡터 materialization과 pgvector 인덱스 |
| 04 | `04_probe_search_and_certification.ipynb` | probe 검색, open-set feature, 인증 결과 |
| 05 | `05_evaluation_and_visualization.ipynb` | 최종 지표, 표, 그림 |

각 노트북은 위에서 아래로 실행합니다. 중단 후 재개할 때는 임의의 중간 셀부터 시작하지 말고 커널을 재시작한 뒤 bootstrap과 입력 검증 셀을 먼저 실행합니다. 이전 단계의 설정 hash나 산출물 checksum이 달라졌으면 새 run을 만들고 영향을 받는 단계부터 다시 실행합니다.

## 실행 기록

새 실험은 KST 날짜와 일일 회차를 사용해 다음처럼 저장됩니다.

```text
runs/YYYY/MM/DD/YYYYMMDD-RNNN-<config-hash>_<experiment-name>/
```

각 run에는 비밀정보를 제거한 `run_manifest.json`, `logs/events.jsonl`, phase별 재시도 기록, 산출물, 그림, 모델이 저장됩니다. 완료된 run은 덮어쓰지 않으며 재시도는 새 attempt로 기록합니다. 논문에 사용할 결과만 checksum과 출처를 확인한 뒤 `results/paper/`로 선별합니다.

## 주요 평가 지표

- 등록 검색: Rank-1, Rank-5, CMC
- open-set: DIR@FPIR, FNIR@FPIR, known/unknown 분리 결과
- calibration: AUROC, ECE, Brier score
- 효율: 저장 byte, 압축률, 인덱스 생성 시간, P50/P95 검색 시간, HNSW recall
