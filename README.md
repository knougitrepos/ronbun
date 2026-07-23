# Face Embedding Compression Characterization Thesis

이 저장소는 사전학습된 얼굴 인식(FR) checkpoint의 **post-hoc 임베딩 압축 특성**을 재현 가능하게 비교하기 위한 석사논문 실험 코드입니다. Step 1의 압축 기준은 [architect/20260720.md](architect/20260720.md), PyTorch 모델 비교와 별도 Grad-CAM 후속 분석의 현재 기준은 [architect/20260723.md](architect/20260723.md)입니다. [THESIS_RESEARCH_PLAN.md](THESIS_RESEARCH_PLAN.md)는 이전 시스템 연구 방향을 보존한 문서입니다.

핵심 질문은 다음과 같습니다.

> ArcFace, AdaFace, MagFace 계열 checkpoint의 512D 얼굴 임베딩에 독립적인 PCA 또는 PQ 압축을 적용할 때 embedding geometry, score, rank 및 open-set 성능이 모델·데이터 품질·압축 강도에 따라 어떻게 변하는가?

---

## 연구 방향

### Step 1 원칙

1. 시스템 파이프라인 완성보다 FR 임베딩의 압축 후 특성 분석을 우선합니다.
2. 원본 512D는 offline 비교 기준선이며, 압축 결과를 원본 검색 결과로 바꾸는 fallback은 사용하지 않습니다.
3. PCA와 PQ는 독립 실험군입니다. PCA 출력에 PQ를 다시 적용하지 않습니다.
4. PCA는 `384/256/128/64/32D` 차원 축소 실험입니다.
5. PQ는 원본 512D를 입력으로 하고 `m × nbits` code budget으로 정의합니다.
6. LFW와 QMUL-SurvFace는 현재 핵심 정량 데이터셋입니다. 새로 확보한 RFW는 조건부 1:1 평가, BalancedFace는 개발·보정 후보로 분리하며 현재 정량 목록에 자동 승격하지 않습니다.
7. `MODE`, `DATA_FRACTION`, `SEED`로 실행량을 제어하되 identity와 공식 open-set 역할을 보존합니다.
8. 데이터셋별 노트북은 실행 절차만 담당하고 공통 결과 표·그림은 `notebooks/common/`에서 생성합니다.

### 구현 상태

| 항목 | 상태 |
| --- | --- |
| InsightFace ArcFace 512D 추출 | 구현됨 |
| PCA-only 384/256/128/64/32 | Step 1 구현 |
| 원본 512D PQ-only | Step 1 구현 |
| `dev/real` 및 identity-aware fraction | Step 1 구현 |
| LFW manifest와 파생 open-set 진단 | 구현됨 |
| SurvFace 공식 gallery/mated/unmated manifest | 구현됨 |
| RFW 공식 4그룹×10-fold 1:1 protocol manifest | 구현·로컬 검증됨 |
| BalancedFace RFW-overlap 제거·identity development/calibration index | 구현·로컬 검증됨 |
| BalancedFace RecordIO image materialization | 미구현 |
| 현재 MS1MV2 checkpoint의 RFW headline 평가 | overlap gate로 차단 |
| PyTorch FR 공통 adapter | Step 2 구현됨 |
| ArcFace/AdaFace/MagFace 실제 checkpoint 등록 | 미구현 |
| pair-conditioned Grad-CAM·사례 선택·faithfulness core | Step 2 구현됨 |
| 실제 FR checkpoint Grad-CAM 결과 | 미구현 |
| 세 모델 공정성 게이트 | 검증 필요 |

Step 2의 adapter 코드가 존재하더라도 실제 checkpoint 경로, hash, 전처리와 target layer를 등록하고 smoke test를 통과하기 전에는 해당 모델을 실행 가능하다고 간주하지 않습니다. 서로 다른 공개 checkpoint를 사용할 경우 backbone, 학습 데이터 및 전처리 차이가 섞이므로 결과를 loss 함수의 인과효과가 아닌 checkpoint 수준 비교로 제한합니다. FR 모델을 새로 학습하지 않습니다.

## 연구 범위

- 원본 512D, PCA-only 및 PQ-only 결과를 같은 probe/template 단위로 비교합니다.
- reconstruction MSE, angular error, cosine-score drift, threshold crossing, rank inversion, DIR/TPIR@FPIR 및 저장 byte를 기록합니다.
- `agreement_with_origin`과 ground-truth 정확도를 분리합니다.
- 원본 threshold 고정 결과와 압축별 threshold 재보정 결과를 함께 보고합니다.
- PostgreSQL/pgvector와 HNSW는 PCA 검색의 보조 시스템 측정으로 유지할 수 있지만 Step 1의 주 기여는 아닙니다.
- 기존 certification/fallback 코드는 과거 run 재현용으로 보존하며 새 Step 1 실행 경로에서는 사용하지 않습니다.

## 디렉터리 구조

| 경로 | 역할 |
| --- | --- |
| `configs/database.yaml` | Git에 기록하는 비밀정보 없는 DB 기본 설정 |
| `configs/database.local.yaml` | 로컬 DB 비밀번호·설정 override. Git에서 제외 |
| `configs/experiments/` | 실험 조건과 고정된 프로토콜 설정 |
| `configs/datasets/` | 로컬 데이터 source, checksum, integrity 및 역할 기록 |
| `docs/datasets/` | 데이터 취득·무결성·누수·이용 경계 문서 |
| `research/database/` | DB 설정 로딩, 연결, 스키마, 저장소 |
| `research/embeddings/` | InsightFace 로딩과 ArcFace 임베딩 추출 |
| `research/embeddings/pytorch/` | Step 2 PyTorch FR 공통 adapter와 checkpoint provenance |
| `research/explainability/gradcam/` | pair target Grad-CAM, 사례 선택과 faithfulness |
| `research/compression/` | 독립 PCA family와 원본 512D PQ 학습·변환 |
| `research/search/` | 이전 시스템 실험의 open-set 검색·인증·fallback 보존 |
| `research/protocols/` | identity-disjoint 데이터 분할 |
| `research/calibration/` | 임계값 선택과 경량 거부 보정 |
| `research/evaluation/` | fallback 없는 paired 압축 특성·검색 비교와 FPIR/DIR 지표 |
| `research/templates/` | 템플릿 집계 ablation |
| `research/runtime/` | 날짜·회차별 실행 기록과 안정적인 실행 코드 |
| `notebooks/lfw/`, `notebooks/survface/` | 데이터셋별 준비·추출 및 Step 1 실행 runbook |
| `notebooks/rfw/`, `notebooks/balancedface/` | RFW protocol 및 BalancedFace 개발 source 준비 runbook |
| `notebooks/common/` | 두 데이터셋의 공통 결과 schema 검사·표·그림 |
| `notebooks/model_validation/` | Step 2 checkpoint 등록·전처리·출력 검증 |
| `notebooks/lfw/gradcam/` | 정량 분석과 분리된 LFW Grad-CAM 후속 runbook |
| `notebooks/database/` | `run_uid`와 allowlist 기반 PostgreSQL 선택 정리 runbook |
| `runs/` | 실행별 manifest, 로그, phase 산출물. Git에서 제외 |
| `results/paper/` | 검증 후 논문 표·그림으로 선별한 결과 |

복잡하거나 재사용되어야 하는 계산, DB 처리, 압축, 검색, 지표 코드는 `research/`의 `.py`에 둡니다. 노트북은 설정을 고정하고 각 단계를 호출하며 결과를 확인하는 얇은 실행 절차입니다.

## 환경 준비

프로젝트 루트에서 다음 명령을 실행합니다.

```powershell
py -m pip install -r requirements.txt
```

Step 1의 압축·평가 모듈 자체는 NumPy/Faiss artifact로 실행할 수 있습니다. 다만 현재 데이터셋별 06 노트북은 기존 extraction run의 원본 임베딩을 PostgreSQL에서 읽으므로 해당 run을 사용할 때는 `configs/database.yaml`과 다음 설정이 필요합니다.

```powershell
$env:RONBUN_DB_PASSWORD = "<local-password>"
py scripts/setup_postgres_pgvector.py --dry-run
py scripts/setup_postgres_pgvector.py
```

필요하면 Git에서 제외되는 `configs/database.local.yaml`에 로컬 override를 둘 수 있습니다. 환경 변수 `RONBUN_DB_HOST`, `RONBUN_DB_PORT`, `RONBUN_DB_NAME`, `RONBUN_DB_USER`, `RONBUN_DB_PASSWORD`가 가장 높은 우선순위를 갖습니다. 로그와 manifest에는 비밀번호를 기록하지 않습니다.

리팩토링 전후의 DB 행이 섞였을 때는
`notebooks/database/selective_cleanup.ipynb`에서 먼저 테이블·`run_uid`별
행 수를 읽기 전용으로 확인합니다. 이 도구는 정확한 `run_uid`와 허용된
테이블만 선택하며, 공유 `images`와 조건 없는 전체 테이블 삭제는 지원하지
않습니다. 기본값에서는 DB 연결과 DELETE가 모두 꺼져 있고 완료 run 삭제도
차단됩니다.

Step 1 공통 설정은 `configs/experiments/step1_embedding_compression.yaml`입니다. ArcFace만 현재 실행 가능하며 AdaFace와 MagFace는 adapter가 추가될 때까지 계획 상태입니다.

Step 2를 실행하려면 기본 환경을 설치한 뒤 로컬 driver에 맞는 PyTorch wheel을 준비합니다.

```powershell
py -m pip install -r requirements-step2.txt
```

GPU wheel은 로컬 NVIDIA driver와 맞아야 하므로 필요할 때 PyTorch 공식 설치 선택기의 명령을 우선합니다. Step 2 설정은 `configs/experiments/step2_pytorch_gradcam.yaml`이며, 기본값은 checkpoint 미등록·`execute_stage: false`·`write_outputs: false`로 실패 폐쇄형입니다.

## Step 2 실행 경계

1. `notebooks/model_validation/`에서 checkpoint 출처·hash·전처리·512D 출력·raw norm과 target layer를 검증합니다.
2. 고정된 공통 정렬 crop에서 모델별 PyTorch embedding을 새 lineage로 생성합니다.
3. Step 1과 같은 독립 PCA-only/PQ-only 정량 실험을 모델별로 다시 수행합니다.
4. 정량 결과에서 사례 목록을 먼저 고정한 뒤 `notebooks/lfw/gradcam/`을 실행합니다.
5. Grad-CAM은 원본 query–gallery cosine target을 설명하며 hard PQ를 미분하지 않습니다.

기존 ONNX Step 1과 PyTorch Step 2는 동일 checkpoint parity가 검증되지 않으면 합치지 않습니다. 완료된 Step 1 run은 읽기 전용 기준선으로 유지합니다.

## 데이터 매니페스트 준비

| 데이터셋 | 준비 노트북 | 기본 출력 폴더 |
| --- | --- | --- |
| LFW deep-funneled | `notebooks/lfw/data_preparation.ipynb` | `data/interim/lfw/` |
| QMUL-SurvFace-v1 | `notebooks/survface/data_preparation.ipynb` | `data/interim/survface/` |
| RFW | `notebooks/rfw/data_preparation.ipynb` | `data/interim/rfw/` |
| BUPT-BalancedFace | `notebooks/balancedface/data_preparation.ipynb` | `data/interim/balancedface/` |

모든 준비 노트북은 기본 `WRITE_OUTPUTS=False`에서 데이터와 프로토콜을 검사하되 파일은 저장하지 않습니다. 검증 통과 후 `True`로 바꾸어 위에서 아래로 다시 실행합니다. SurvFace는 `training_set`을 identity-disjoint development/calibration으로 분리하고, 별도의 official manifest에서는 gallery/mated/unmated 역할과 `protocol_index`를 보존합니다.

RFW와 BalancedFace는 반드시 위 표의 순서로 실행합니다. RFW는 공식 1:1 verification test이므로 PCA/PQ를 fit하거나 DIR/FPIR 공식 결과로 사용하지 않습니다. BalancedFace는 RFW와 겹치는 provider identity를 제거한 뒤 development에서 압축기를 fit하고 calibration에서 threshold를 보정하는 후보이며 최종 test가 아닙니다. BalancedFace JPG archive는 정상 파일로 교체됐지만 Asian/Indian의 가변 해상도·정렬 품질 검증이 필요합니다. RecordIO를 선택할 경우에는 PyTorch용 decoder가 아직 미구현이므로 `source_index_manifest.csv`를 실제 image manifest로 해석하면 안 됩니다. 상세 상태는 [RFW/BalancedFace 데이터 기록](docs/datasets/RFW_BALANCEDFACE.md)을 따릅니다.

## 노트북 실행 순서

각 노트북 상단에서 `MODE`, `DATA_FRACTION`, `SEED`를 먼저 고정합니다. 전체 논문 결과는 `MODE="real"`, `DATA_FRACTION=1.0`만 해당하며, 작은 fraction은 identity 단위의 결정론적 개발 실행입니다.

기존 00~05는 `thesis3` DB/certification 결과를 재현하기 위해 파일명을 유지합니다. Step 1의 주 실행은 데이터 준비·원본 임베딩 추출 후 각 데이터셋의 06을 사용합니다.

| 순서 | LFW 노트북 | 산출물 |
| --- | --- | --- |
| 준비 | `data_preparation.ipynb` | identity-disjoint manifest와 ID 목록 |
| 00 | `00_protocol_and_run_freeze.ipynb` | 데이터 분할, 설정 hash, 새 run 고정 |
| 01 | `01_arcface_embedding_extraction.ipynb` | 원본 ArcFace 임베딩과 추출 메타데이터 |
| 02~05 | 기존 파일 | 이전 DB/certification run 재현용; Step 1에서는 fallback 경로 실행 금지 |
| 06 | `06_step1_compression_characterization.ipynb` | PCA-only/PQ-only 학습, 원본 대비 paired 특성·검색 결과 |

SurvFace의 PCA/PQ는 `training_manifest.csv`의 development split에서만 학습하고 calibration에서 설정을 고정한 뒤 official test에 적용합니다. official gallery/mated/unmated 행으로 compressor를 fit하지 않습니다.

| 순서 | SurvFace 노트북 | 핵심 차이 |
| --- | --- | --- |
| 준비 | `data_preparation.ipynb` | 공식 MAT 순서와 gallery/mated/unmated 역할 보존 |
| 00 | `00_official_protocol_and_run_freeze.ipynb` | 공식 protocol과 checksum 고정, known unknown 0건 |
| 01 | `01_official_arcface_embedding_extraction.ipynb` | 242,453개 공식 행을 순서대로 추출하고 실패 분모 기록 |
| 02~05 | 기존 파일 | 이전 외부-compressor/DB 공식 실험 재현용 |
| 06 | `06_step1_compression_characterization.ipynb` | training fit과 official-test fallback-free 비교 |

두 06 산출물을 만든 뒤 `notebooks/common/cross_dataset_results.ipynb`에서 동일 schema로 집계합니다. 공통 노트북은 fallback 열이나 fallback 사용 행을 Step 1 결과로 허용하지 않습니다.

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
