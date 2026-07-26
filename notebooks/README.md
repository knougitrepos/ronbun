# 재현 가능한 노트북 실행 순서

노트북은 계산 구현이 아니라 `research/`의 검증된 Python 함수를 순서대로 호출하는
runbook이다. 저장된 노트북은 출력과 실행 번호를 포함하지 않으며, 항상 커널을
재시작한 뒤 위에서 아래로 실행한다.

기본 실험값은 다음과 같다.

- `DATA_FRACTION = 1.0`
- `EXECUTE_STAGE = True`
- `WRITE_OUTPUTS = True`
- `OVERWRITE = True`

`OVERWRITE=True`는 같은 단계에서 정식 결과를 두세 개 만들기 위한 옵션이 아니다.
완전한 대체 결과를 먼저 만든 뒤 그 단계의 canonical 결과 하나만 교체한다. 반면
`RunStore`에서 `COMPLETED`가 기록된 run은 계속 불변이다. open-set threshold나
보정 방식이 바뀌면 설정이 달라진 새 run을 만들고, 한 run 안에서는 결과를 하나만
유지한다.

## 1. Prerequisite

### 데이터

다음 순서로 필요한 데이터셋만 준비한다.

1. `prerequisite/datasets/00_lfw_data_preparation.ipynb`
2. `prerequisite/datasets/01_survface_data_preparation.ipynb`
3. `prerequisite/datasets/02_rfw_data_preparation.ipynb`
4. `prerequisite/datasets/03_balancedface_data_preparation.ipynb`
5. `prerequisite/datasets/04_lfw_aligned_crop_materialization.ipynb`

RFW는 공식 1:1 verification test이며 PCA/PQ fit 데이터가 아니다. BalancedFace는
RFW 중복 identity를 제거한 development/calibration 후보이며 최종 test가 아니다.

### 모델

1. `prerequisite/models/00_checkpoint_registration.ipynb`
2. `prerequisite/models/01_preprocessing_and_model_smoke.ipynb`

checkpoint 출처, SHA-256, preprocessing, 512D 출력, raw norm과 target layer가
검증되어야 이후 Step 2가 진행된다. 실행 플래그가 `True`여도 누락되거나 검증되지
않은 입력은 그대로 실패하므로 입력 검증은 fail-closed이다.

### 임베딩

LFW:

1. `prerequisite/embeddings/lfw/00_protocol_and_run_freeze.ipynb`
2. `prerequisite/embeddings/lfw/01_arcface_embedding_extraction.ipynb`

SurvFace:

1. `prerequisite/embeddings/survface/00_official_protocol_and_run_freeze.ipynb`
2. `prerequisite/embeddings/survface/01_official_arcface_embedding_extraction.ipynb`

임베딩의 canonical 저장소는 PostgreSQL/pgvector이며, run manifest와 CSV index가
model/config/input hash를 연결한다. NPY/NPZ shard는 배열 자체가 필요할 때만 쓰고
CSV 또는 JSON manifest를 반드시 동반한다.

## 2. Experiments

### 압축 특성화

LFW:

1. `experiments/compression/lfw/00_compressor_fit.ipynb`
2. `experiments/compression/lfw/01_compressed_materialization_and_index.ipynb`
3. `experiments/compression/lfw/02_step1_compression_characterization.ipynb`

SurvFace:

1. `experiments/compression/survface/00_external_compressor_import.ipynb`
2. `experiments/compression/survface/01_official_compressed_materialization_and_index.ipynb`
3. `experiments/compression/survface/02_step1_compression_characterization.ipynb`

### Open-set 검색과 평가

LFW:

1. `experiments/open_set/lfw/00_probe_search_and_certification.ipynb`
2. `experiments/open_set/lfw/01_evaluation_and_visualization.ipynb`

SurvFace:

1. `experiments/open_set/survface/00_official_probe_search.ipynb`
2. `experiments/open_set/survface/01_official_evaluation_and_visualization.ipynb`

과거 exact-fallback 재현 경로와 현재 fallback-free 압축 특성화 경로는 섞지 않는다.
보정 조건을 수정한 반복 실험은 새 config hash/run_id로 기록한다.

### Grad-CAM

세부 순서와 경계는 `experiments/gradcam/README.md`를 따른다. 이 하위에서도
embedding/LOO 생성은 `prerequisite/`, 실제 saliency·compression 결합은
`experiment/`로 분리한다.

## 3. Reports

1. `reports/00_cross_dataset_results.ipynb`

보고용 결과는 CSV/Markdown 표와 그림으로 내보낸다. notebook cell output은
정식 실험 기록으로 사용하지 않는다.

## 4. Maintenance

1. `maintenance/00_selective_cleanup.ipynb`

삭제·격리 노트북은 실험 기본값의 예외다. preview, confirmation token 및 명시적
실행 승인을 유지하며 자동 실행 기본값을 적용하지 않는다.

## 결과 저장 형식

- PostgreSQL/pgvector: 임베딩, 압축 벡터, 검색·보정에 필요한 정규화된 상세 행
- CSV: 사람이 확인할 sample index, metric, 집계표, 실패 목록
- JSON/JSONL: run manifest, hash·lineage, 구조화 event log
- 일반 log: 장시간 실행의 진행·경고·실패 문맥
- NPY/NPZ: heatmap·embedding shard처럼 표 형식이 부적합한 배열만 저장

새 hash 기반 폴더 규칙을 별도로 도입하지 않는다. 기존 `RunStore`의
`run_id + config_hash + input hash + phase attempt`를 추적 기준으로 사용한다.
