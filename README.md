# Compressed Open-Set Face Search Thesis

이 저장소는 `PostgreSQL/pgvector`에 저장한 압축 ArcFace 임베딩의 검색 성능, 저장 효율, open-set 미등록 인물 거부 성능을 재현 가능하게 비교하기 위한 석사논문 실험 코드입니다. 연구 방향의 기준 문서는 [THESIS_RESEARCH_PLAN.md](THESIS_RESEARCH_PLAN.md)입니다.

핵심 질문은 다음과 같습니다.

> 임베딩 압축으로 변형된 유사도 분포와 거부 임계값을 경량 calibration으로 보정하여 등록 인물 식별과 미등록 인물 거부 성능을 함께 보존할 수 있는가?

---
## 연구방향 정리

### 고정 baseline

> **압축된 ArcFace 임베딩의 1:N open-set 검색에서 압축 오차와 검색 점수 특성을 이용해 결정 안정성을 평가하고, 불확실한 질의에 대해서만 saliency 특징과 원본 512차원 검색을 단계적으로 적용하는 위험 제약 기반 선택적 검색 방법**

### 핵심 구성

1. InsightFace SCRFD 검출 및 5-point 정렬  
2. 동일한 PyTorch ArcFace로 512차원 임베딩 추출  
3. PCA-448/384/256/128 및 PQ 압축  
4. PostgreSQL/pgvector HNSW 후보 검색  
5. score·margin·angular error 기반 1차 보정  
6. 불확실 질의만 pair-conditioned Grad-CAM 적용  
7. landmark 영역별 saliency 특징 추출  
8. accept/reject/defer 결정  
9. 최종 불확실 질의만 원본 exact fallback  
10. FPIR(False Positive Identification Rate, 오인 수락 식별률)·DIR(Detection and Identification Rate, 탐지 및 식별률)·저장량·지연시간 제약을 만족하는 최소 압축 차원 선택

기존 연구의 핵심이 DB 알고리즘 자체가 아니라 **압축으로 흔들리는 유사도 분포와 open-set 임계값을 분석·보정하는 것**이라는 방향과도 일치합니다.
---

## 진행 원칙

Grad-CAM과 saliency는 처음부터 핵심 방법으로 확정하지 않고 **검증 대상 특징**으로 둡니다.

```text
압축오차 기반 baseline
→ saliency 추가 ablation
→ 잔차·분산 예측 개선 확인
→ fallback 및 전체 latency 개선 확인
→ 효과가 재현될 때 핵심 기여로 승격
```

효과가 없거나 계산비용이 지나치게 크면 saliency는 오류 분석용 보조 실험으로 내리고, **위험 제약 기반 압축 프로파일 선택과 결정 안정성 보정**을 주 기여로 유지합니다.

## 우선 구현 순서

1. 동일 ArcFace의 PyTorch 추론과 기존 ONNX 출력 일치 검증  
2. 4분면 pair-conditioned Grad-CAM 최소 구현  
3. 영역 가림으로 Grad-CAM 신뢰성 확인  
4. 압축 점수 잔차와 saliency 특징의 상관성 분석  
5. 기존 보정 모델 대비 saliency 추가 ablation  
6. 효과 확인 후 106-point landmark 영역으로 확장  
7. 선택적 Grad-CAM과 원본 fallback을 포함한 전체 시스템 평가  

이 기준을 이후 코드 구조, 실험 설계, 관련연구 분류 및 논문 기여 분석의 기본 전제로 삼으면 됩니다.
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
| `notebooks/lfw/`, `notebooks/survface/` | 데이터셋별로 분리한 준비 및 00~05 실험 runbook |
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
  --config configs/experiments/lfw_face_search.yaml `
  --dry-run
```

## 데이터 매니페스트 준비

| 데이터셋 | 준비 노트북 | 기본 출력 폴더 |
| --- | --- | --- |
| LFW deep-funneled | `notebooks/lfw/data_preparation.ipynb` | `data/interim/lfw/` |
| QMUL-SurvFace-v1 | `notebooks/survface/data_preparation.ipynb` | `data/interim/survface/` |

두 노트북 모두 기본 `WRITE_OUTPUTS=False`에서 전체 데이터와 프로토콜을 검사하되 파일은 저장하지 않습니다. 검증 통과 후 `True`로 바꾸어 위에서 아래로 다시 실행합니다. LFW와 SurvFace는 이후 단계도 서로 다른 폴더에서 실행합니다. SurvFace는 공식 gallery/mated/unmated 역할과 `protocol_index`를 보존하며 gallery를 다시 표본추출하지 않습니다.

## 노트북 실행 순서

LFW는 `notebooks/lfw/`에서 `data_preparation` 뒤 00~05를 순서대로 실행합니다. 설정은 `configs/experiments/lfw_face_search.yaml`, 새 run은 `runs/lfw/`에 기록합니다.

| 순서 | LFW 노트북 | 산출물 |
| --- | --- | --- |
| 준비 | `data_preparation.ipynb` | identity-disjoint manifest와 ID 목록 |
| 00 | `00_protocol_and_run_freeze.ipynb` | 데이터 분할, 설정 hash, 새 run 고정 |
| 01 | `01_arcface_embedding_extraction.ipynb` | 원본 ArcFace 임베딩과 추출 메타데이터 |
| 02 | `02_compressor_fit.ipynb` | development split으로 학습한 PCA/PQ 프로파일 |
| 03 | `03_compressed_materialization_and_index.ipynb` | 압축 벡터 materialization과 pgvector 인덱스 |
| 04 | `04_probe_search_and_certification.ipynb` | 세 probe 유형의 검색·보정·인증 결과 |
| 05 | `05_evaluation_and_visualization.ipynb` | 최종 지표, 표, 그림 |

SurvFace는 `notebooks/survface/`에서 공식 프로토콜 전용 파일을 실행합니다. 설정은 `configs/experiments/survface_face_search.yaml`, run은 `runs/survface/`에 기록합니다. 공식 test로 PCA/PQ 또는 calibration을 학습하지 않으며, 02에서 development 데이터로 학습된 외부 frozen run을 명시해야 합니다.

| 순서 | SurvFace 노트북 | 핵심 차이 |
| --- | --- | --- |
| 준비 | `data_preparation.ipynb` | 공식 MAT 순서와 gallery/mated/unmated 역할 보존 |
| 00 | `00_official_protocol_and_run_freeze.ipynb` | 공식 protocol과 checksum 고정, known unknown 0건 |
| 01 | `01_official_arcface_embedding_extraction.ipynb` | 242,453개 공식 행을 순서대로 추출하고 실패 분모 기록 |
| 02 | `02_external_compressor_import.ipynb` | 공식 test 학습 금지, 외부 development-trained 모델 고정 |
| 03 | `03_official_compressed_materialization_and_index.ipynb` | 압축 materialization과 ID별 `official_all` 평균 template |
| 04 | `04_official_probe_search.ipynb` | 3,000 template 대상 exact/HNSW rank-20 검색 분리 |
| 05 | `05_official_evaluation_and_visualization.ipynb` | TPIR@FPIR 0.1/0.2/0.3, rank-20, AUC 및 성공 분모 |

현재 SurvFace 04는 `official_all` template 생성과 공식 순서 보존을 강제하는
공통 search API가 연결되기 전에는 `EXECUTE_STAGE=True`에서 명확히 중단합니다.
일반 LFW 검색 결과를 공식 SurvFace 결과처럼 재사용하지 않습니다.

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
