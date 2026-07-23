# Step 1 dataset runbooks

현재 Step 1에서 연결된 데이터셋은 LFW와 QMUL-SurvFace-v1입니다. 데이터셋별
노트북은 데이터 준비, 임베딩 추출, 독립 압축군 실행, 평가를 재현하는 얇은
runbook이며 공통 계산은 `research/`의 Python 모듈을 호출합니다.

모든 노트북의 첫 설정 셀에서 다음 세 값을 먼저 고정합니다.

- `MODE`: 빠른 점검용 `dev` 또는 논문 결과용 `real`
- `DATA_FRACTION`: identity 단위로 선택할 비율 `(0, 1]`
- `SEED`: 같은 비율에서 동일 identity를 선택하기 위한 seed

`MODE="real"`, `DATA_FRACTION=1.0`인 실행만 전체 데이터 논문 결과로
취급합니다. 작은 비율은 같은 seed에서 큰 비율의 부분집합이 되며, 이미지를
무작위로 자르지 않습니다.

## 압축 실험군

- PCA: 원본 512D에서 각각 384/256/128/64/32D로 독립 투영합니다.
- PQ: PCA 결과가 아닌 원본 512D에 직접 학습하고 적용합니다.
- `PCA -> PQ` 결합군은 Step 1에 포함하지 않습니다.
- 원본 512D 재탐색으로 defer 결과를 대체하는 fallback은 사용하지 않습니다.

## LFW

`notebooks/lfw/data_preparation.ipynb`에서 development, calibration, test를
identity-disjoint하게 만든 뒤 각 역할 안에서 `DATA_FRACTION`을 적용합니다.
이후 임베딩과 압축 노트북을 실행합니다. Step 1 설정의 기준 파일은
`configs/experiments/step1_embedding_compression.yaml`입니다.

기존 `03_compressed_materialization_and_index.ipynb`와
`04_probe_search_and_certification.ipynb`는 thesis3 DB 시스템 결과 재현을 위해
보존합니다. 특히 04는 과거 origin exact fallback을 포함하므로 Step 1에서는
실행할 수 없게 guard되어 있습니다.

## QMUL-SurvFace-v1

`notebooks/survface/data_preparation.ipynb`는 공식 gallery/mated/unmated 역할과
순서를 먼저 검증합니다. `real, 1.0`은 공식 protocol 전체를 그대로 사용하고,
`dev`에서는 역할을 섞지 않은 identity-aware 부분집합만 만듭니다. 공식 test로
PCA/PQ codebook이나 threshold를 학습하지 않으며 필요한 학습 통계는
development 데이터에서 고정해야 합니다.

기존 `02_external_compressor_import.ipynb`와 03~05는 LFW 압축기 전이 및 DB
시스템 재현용입니다. Step 1의 같은-dataset 주 실험은
`06_step1_compression_characterization.ipynb`에서 SurvFace training
development/calibration과 공식 test를 분리해 수행합니다.

## RFW와 BUPT-BalancedFace

새 데이터는 다음 순서로 준비합니다.

1. `rfw/data_preparation.ipynb`
2. `balancedface/data_preparation.ipynb`

RFW 노트북은 4개 그룹의 공식 10-fold 1:1 pair, image list와 landmark를 검증하고 `source_identities.txt`를 기록합니다. RFW는 evaluation test이므로 PCA/PQ fit이나 DIR/FPIR open-set 공식 결과에 사용하지 않습니다.

BalancedFace 노트북은 위 RFW identity artifact를 필수 입력으로 받아 겹치는 provider identity를 전부 제거한 뒤, 그룹별 identity 단위로 development/calibration을 나눕니다. BalancedFace는 FR 모델 재학습 데이터나 최종 test가 아닙니다.

`data/raw/RFW-balancedface/images/Equalizedface.tar.gz`는 정상 파일로 교체되어 EOF 검사를 통과했습니다. 다만 JPG와 RecordIO 목록은 이미지 14장·identity 1개 차이가 있으며 Asian/Indian JPG에는 가변 해상도 이미지가 포함됩니다. JPG 경로는 공통 정렬·그룹별 coverage 검증 전까지 비활성이고, RecordIO 경로는 metadata index까지만 구현되어 decoder/materializer가 아직 없습니다. 따라서 `data/interim/balancedface/source_index_manifest.csv`는 실제 image path manifest가 아닙니다.

## 공통 결과

데이터셋별 평가가 끝난 뒤
`notebooks/common/cross_dataset_results.ipynb`에서 결과 manifest와 표를 읽어
동일한 열 정의로 집계하고 시각화합니다. 이 노트북은 fallback 열이 포함된
artifact를 Step 1 결과로 받아들이지 않습니다.

## 재시작 원칙

중단 후 임의 셀부터 실행하지 말고 커널을 재시작한 뒤 처음부터 실행합니다.
입력 hash, scope 또는 상위 단계 artifact checksum이 달라지면 새 run을 만들고
영향받는 단계부터 다시 수행합니다.

## Step 2 PyTorch 모델 및 Grad-CAM 후속 분석

Step 2는 기존 데이터셋별 Step 1 노트북을 수정하지 않고 다음 두 폴더를
추가합니다.

- `model_validation/`: ArcFace/AdaFace/MagFace의 checkpoint 등록, 전처리
  명세, 512D/raw norm/L2 출력 및 target layer를 검증합니다.
- `lfw/gradcam/`: 완료된 PyTorch 정량 결과에서 사례를 고정한 뒤 pair
  cosine Grad-CAM과 occlusion faithfulness를 분석합니다.

실행 순서는 다음과 같습니다.

1. `model_validation/00_checkpoint_registration.ipynb`
2. `model_validation/01_preprocessing_and_model_smoke.ipynb`
3. 별도 Step 2 정량 embedding·PCA/PQ run 완료
4. `lfw/gradcam/00_source_and_model_freeze.ipynb`
5. `lfw/gradcam/01_case_selection.ipynb`
6. `lfw/gradcam/02_pair_gradcam_generation.ipynb`
7. `lfw/gradcam/03_saliency_feature_analysis.ipynb`
8. `lfw/gradcam/04_faithfulness_and_report.ipynb`

Grad-CAM 폴더는 정량 압축 코드를 복사하지 않습니다. 입력 result hash와 선택
case manifest를 고정하고 `research/explainability/gradcam/`의 공통 함수를
호출합니다. 실제 checkpoint가 등록되기 전에는 상단 기본값
`EXECUTE_STAGE=False`, `WRITE_OUTPUTS=False`를 유지합니다.

## PostgreSQL 선택 정리

리팩토링 전후의 서로 다른 `run_uid` 행이 같은 테이블에 공존할 때는
`database/selective_cleanup.ipynb`를 사용합니다.

1. `CONNECT_TO_DATABASE=True`, `EXECUTE_DELETE=False`로 전체 테이블 및
   `run_uid`별 행 수를 먼저 확인합니다.
2. 정확한 `RUN_UID`와 허용된 테이블 그룹 또는 개별 테이블을 선택합니다.
3. 미리보기 행 수와 완료 run 보호 상태를 확인합니다.
4. 실제 폐기가 필요할 때만 출력된 확인 문자열을 복사하고
   `EXECUTE_DELETE=True`로 처음부터 다시 실행합니다.

이 노트북은 조건 없는 전체 테이블 삭제, 임의 SQL 및 공유 `images` 삭제를
지원하지 않습니다. 완료된 run은 기본적으로 차단되며 보호를 해제해도 자동
백업되지는 않습니다.
