# 데이터셋별 노트북 실행 안내

노트북은 계산 구현이 아니라 `research/`의 검증된 Python 함수를 순서대로 호출하는
runbook이다. 최상위 폴더는 데이터셋, 그 아래 숫자 접두사는 실행 단계를 뜻한다.
저장된 노트북은 출력과 실행 번호를 포함하지 않으며 항상 커널을 재시작한 뒤
위에서 아래로 실행한다.

```text
notebooks/
  lfw/
    00_data_preparation/
    01_embeddings/
    02_compression/
    03_open_set/
    04_gradcam/
      prerequisite/
      experiment/
  survface/
    00_data_preparation/
    01_embeddings/
    02_compression/
    03_open_set/
    04_gradcam/
      prerequisite/
      experiment/
  rfw/
    00_data_preparation/
  balancedface/
    00_data_preparation/
  common/
    model_preparation/
    reports/
    maintenance/
```

RFW는 현재 공식 1:1 verification test 준비까지만 구현되어 있고 PCA/PQ fit
데이터가 아니다. BalancedFace는 RFW 중복 identity를 제거한
development/calibration 후보이며 최종 test가 아니다. 구현되지 않은 단계를
빈 폴더나 자리표시자 노트북으로 만들지 않는다.

## 기본 실행 계약

일반 준비·실험·보고 노트북의 기본값은 다음과 같다.

- `DATA_FRACTION = 1.0`
- `EXECUTE_STAGE = True`
- `WRITE_OUTPUTS = True`
- `OVERWRITE = True`

`OVERWRITE=True`는 같은 단계의 canonical 결과 하나를 완전한 새 결과로 교체한다.
완료된 `RunStore` run은 수정하지 않는다. open-set threshold나 보정 방법이
달라지면 새 config hash와 run ID로 실행한다. `common/maintenance/`는 파괴적
작업을 포함하므로 이 기본값의 예외이며 preview와 confirmation을 계속 요구한다.

Step 4의 aligned-crop·landmark·Grad-CAM 노트북은 장시간 GPU 실행이므로 별도
예외다. `configs/experiments/step2_pytorch_gradcam.yaml`의
`execute_stage=true`, `write_outputs=true`, `overwrite=false`를 읽는다.
현재 값은 LFW·SurvFace 전체 재실험용 실행 commit 프로필이며,
`allow_dirty=false`이므로 commit 후 clean worktree에서만 새 run을 시작한다.

## LFW

일반 실행은 다음 폴더 순서를 따른다.

1. `lfw/00_data_preparation/`
2. `lfw/01_embeddings/`
3. `lfw/02_compression/`
4. `lfw/03_open_set/`
5. `common/reports/00_cross_dataset_results.ipynb`

PyTorch Step 2와 Grad-CAM은 다음 순서를 따른다.

1. `lfw/00_data_preparation/00_data_preparation.ipynb`
2. `lfw/00_data_preparation/01_aligned_crop_materialization.ipynb`
3. `lfw/00_data_preparation/02_landmark_region_materialization.ipynb`
4. `common/model_preparation/00_checkpoint_registration.ipynb`
5. `common/model_preparation/01_preprocessing_and_model_smoke.ipynb`
6. `lfw/04_gradcam/prerequisite/`
7. `lfw/04_gradcam/experiment/`

Grad-CAM의 세부 순서는 [LFW Grad-CAM 안내](lfw/04_gradcam/README.md)를 따른다.

## SurvFace

다음 폴더를 숫자 순서대로 실행한다.

1. `survface/00_data_preparation/`
2. `survface/01_embeddings/`
3. `survface/02_compression/`
4. `survface/03_open_set/`
5. `survface/04_gradcam/prerequisite/`
6. `survface/04_gradcam/experiment/`
7. `common/reports/00_cross_dataset_results.ipynb`

공식 gallery/mated/unmated 역할과 순서를 유지하며, official test에서 압축기나
threshold를 학습하지 않는다. `00_data_preparation/01`에서 전체 aligned crop,
`00_data_preparation/02`에서 전체 106-point landmark bundle을 먼저 생성한다.
`02_compression/00`은 SurvFace training development에서 PCA/PQ를 학습하고,
`02_compression/01`은 frozen model로 전체 run을 materialize한다. PQ code는
pgvector vector가 아니다. `03_open_set/00`은 origin/PCA-256의 exact/HNSW
네 조합을 하나의 공식-order 결과로 만든다. Grad-CAM은 registered/unmated
target 적격 표본 전체를 사용한다.

SurvFace 장시간 반복 단계는 batch마다 checkpoint를 유지하되 notebook log는
약 10% 경계에서만 출력한다. 전체 LFW·SurvFace 데이터 실험은 사용자가 각
노트북을 직접 실행하며 Codex나 일괄 CLI가 자동으로 시작하지 않는다.

## Step 4 데이터셋별 재실행

공용 `notebooks/step4` 폴더나 단일 일괄 CLI는 사용하지 않는다. LFW와
SurvFace의 각 노트북은 `research/experiments/step4_workflow.py`에 있는 하나의
단계 함수만 호출한다. 이전 단계 artifact가 없거나 lineage가 다르면 다음
단계는 fail-closed로 중단한다.

정식 실행은 clean commit과 CUDA/ONNX CUDA provider 확인 후 데이터셋별
`00_data_preparation`부터 순서대로 수행한다. LFW와 SurvFace는 별도의
immutable run이며 geometry association과 protocol/threshold별 retrieval
association도 서로 다른 artifact로 저장한다.

## RFW와 BalancedFace

1. `rfw/00_data_preparation/00_data_preparation.ipynb`
2. `balancedface/00_data_preparation/00_data_preparation.ipynb`

BalancedFace 준비 단계가 RFW source identity 목록을 읽으므로 이 둘은 위 순서를
지켜야 한다. RFW를 현재 정량 headline 결과로 승격하거나 BalancedFace를 최종
test로 해석하지 않는다.

## 결과 저장 형식

- PostgreSQL/pgvector: 임베딩, 압축 벡터, 검색·보정 상세 행
- CSV: 사람이 확인할 sample index, metric, 집계표, 실패 목록
- JSON/JSONL: run manifest, hash·lineage, 구조화 event log
- 일반 log: 장시간 실행의 진행·경고·실패 문맥
- NPY/NPZ: heatmap·embedding shard처럼 표 형식이 부적합한 배열

새 hash 폴더 규칙은 추가하지 않는다. 기존 `RunStore`의
`run_id + config_hash + input hash + phase attempt`를 추적 기준으로 사용한다.
