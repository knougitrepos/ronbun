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
3. `common/model_preparation/00_checkpoint_registration.ipynb`
4. `common/model_preparation/01_preprocessing_and_model_smoke.ipynb`
5. `lfw/04_gradcam/prerequisite/`
6. `lfw/04_gradcam/experiment/`

Grad-CAM의 세부 순서는 [LFW Grad-CAM 안내](lfw/04_gradcam/README.md)를 따른다.

## SurvFace

다음 폴더를 숫자 순서대로 실행한다.

1. `survface/00_data_preparation/`
2. `survface/01_embeddings/`
3. `survface/02_compression/`
4. `survface/03_open_set/`
5. `common/reports/00_cross_dataset_results.ipynb`

공식 gallery/mated/unmated 역할과 순서를 유지하며, official test에서 압축기나
threshold를 학습하지 않는다.

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
