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

## Step 2 PyTorch 모델 및 원본 공간 특징 분석

Step 2는 기존 데이터셋별 Step 1 노트북을 수정하지 않고 다음 두 폴더를
추가합니다.

- `model_validation/`: ArcFace/AdaFace/MagFace의 checkpoint 등록, 전처리
  명세, 512D/raw norm/L2 출력 및 target layer를 검증합니다.
- `lfw/gradcam/`: 모든 선택 이미지의 원본 embedding과 LOO cosine
  Grad-CAM 특징을 먼저 추출하고, 이후 PCA/PQ 민감도와 결합합니다.

실행 순서는 다음과 같습니다.

1. `model_validation/00_checkpoint_registration.ipynb`
2. `model_validation/01_preprocessing_and_model_smoke.ipynb`
3. `lfw/gradcam/00_source_and_model_freeze.ipynb`
4. `lfw/gradcam/01_origin_embedding_and_loo_templates.ipynb`
5. `lfw/gradcam/02_population_gradcam_extraction.ipynb`
6. `lfw/gradcam/03_saliency_feature_validation.ipynb`
7. `lfw/gradcam/04_step2_compression_characterization.ipynb`
8. `lfw/gradcam/05_saliency_compression_join.ipynb`
9. `lfw/gradcam/06_representative_case_visualization.ipynb`

Pass A는 모든 선택 표본의 임베딩을 보존하고 Pass B는 동일인
leave-one-out target이 있는 모든 표본에 Grad-CAM을 수행합니다. singleton과
identity 누락 표본은 다른 target으로 바꾸지 않고 부적격 상태로 남깁니다.
압축 코드는 Grad-CAM 추출에 복사하지 않으며 두 결과는 원본 embedding lineage로
엄격히 결합합니다. 사례 선택은 06의 시각화 용도에만 존재합니다. 실제
checkpoint와 공통 aligned bundle이 등록되기 전에는 상단 기본값
`EXECUTE_STAGE=False`, `WRITE_OUTPUTS=False`를 유지합니다.

## run 단위 complete reset

리팩토링 뒤 동일한 `run_uid`의 DB, 임베딩 전처리·중간 artifact와 평가 결과가
섞이지 않게 하려면 `database/selective_cleanup.ipynb`를 사용합니다. 권장
모드는 `RESET_MODE="complete_run_reset"`입니다.

1. `RUN_UID`를 입력하고 `CONNECT_TO_DATABASE=False`,
   `EXECUTE_RESET=False`, 빈 `CONFIRMATION_TOKEN`으로 Run All 합니다.
2. `CONNECT_TO_DATABASE=True`로 바꾸고 다시 Run All 하여 하나의 digest에
   묶인 DB 행, exact-owner run bundle, manifest 소유 result bundle 및 일치하는
   active pointer를 미리보기합니다.
3. 완료 run, `results/paper` 등 promoted result, lineage를 완전히 검증할 수
   없는 고아 run은 각각
   `ALLOW_COMPLETED_RUN_RESET`,
   `ALLOW_PROMOTED_RESULTS_RESET`,
   `ALLOW_UNVERIFIED_LINEAGE_RESET` 없이는 실행할 수 없습니다.
4. 미리보기 대상과 보존 대상을 확인한 뒤 새 확인 문자열 전체를
   `CONFIRMATION_TOKEN`에 복사하고 `EXECUTE_RESET=True`로 처음부터 다시
   실행합니다.
5. 실행 후 DB의 해당 run 행이 0인지 확인하고
   `runs/database_cleanup/quarantine/<operation>/payload/`와 감사 JSON을
   확인합니다.

로컬 대상은 영구 삭제하지 않고 격리합니다. DB 삭제는 한 transaction에서
수행하며 로컬 이동 callback이 실패하면 DB를 rollback하고 이미 이동한 파일을
복원합니다. PostgreSQL과 파일시스템은 하나의 원자적 transaction이 아니므로
실행 중 프로세스를 강제 종료하지 마십시오.

항상 보존되는 범위는 공유 `images`, `data/raw`, common aligned crop와 dataset
manifest, checkpoint/model registry, 다른 run 및 cleanup 감사 기록입니다.
PCA 64D/32D와 Step 2 Grad-CAM/LOO artifact처럼 별도 DB 테이블이 없는 결과도
exact-owner run artifact이면 함께 격리됩니다. 반대로 소유권이 확인되지 않는
임의 Step 2 경로는 자동 선택하지 않습니다. DB row snapshot은 생성하지 않아
감사 JSON만으로 embedding row를 복구할 수 없습니다.

특정 PostgreSQL 테이블만 선택해야 하는 전문 유지보수는
`RESET_MODE="advanced_database_cleanup"`을 사용합니다. 기존 allowlist,
행 수 재검증, 완료 run 보호와 확인 문자열 계약은 유지되며 조건 없는 전체
테이블·임의 SQL·공유 `images` 삭제는 지원하지 않습니다.
