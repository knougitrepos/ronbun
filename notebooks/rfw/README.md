# RFW 실행 구조

RFW는 두 개의 명시적으로 분리된 프로토콜로 사용한다.

- **RFW-Official**: 공식 4-group/10-fold pair 기반 supplementary 1:1 verification. TAR/FAR/Accuracy/EER를 보고하며 open-set DIR/FPIR로 재명명하지 않는다.
- **RFW-Custom**: 공식 pair/fold를 사용하지 않고 RFW identity/image metadata를 deterministic development/calibration/test 및 gallery/probe로 나눈 비공식 1:N open-set diagnostic. `official_result_eligible=false`이고 checkpoint training identity overlap은 `UNKNOWN`이다.

아래 RFW 전용 노트북은 RFW-Official 경로다. RFW-Custom은 공통 Step 4 runner와 report에서 LFW/SurvFace와 같은 PCA/PQ·Grad-CAM·compact 흐름을 사용하되 artifact/protocol UID를 Official과 공유하지 않는다.

두 실행 인터페이스를 제공한다.

1. all-in-one: `00_rfw_all_in_one.ipynb`
   - protocol, origin embedding, 명시적 frozen-codec 선택, 1:1 evaluation을 한 번에 계획·실행·재개한다.
   - 기본값은 실행하지 않는 preflight이며 `EXECUTE`, 승인 flag와 stage flag를 모두 켜야 장시간 단계가 시작된다.
2. 절차적 단일 단계: 아래 세 노트북을 순서대로 실행한다.

1. `00_data_preparation/00_data_preparation.ipynb`: 공식 4개 group, 10-fold, 24,000 pair protocol과 source identity 목록을 고정한다.
2. `01_embeddings/00_rfw_origin_embedding_extraction.ipynb`: aligned BIN을 pair batch 단위로 streaming decode하고 선택한 FR 모델의 512D origin embeddings를 immutable artifact로 저장한다.
3. `02_compression/00_rfw_frozen_codec_verification.ipynb`: 사용자가 명시한 완료 LFW 또는 SurvFace run에서 model UID와 SHA가 일치하는 frozen PCA/PQ codec만 적용한다. RFW에서 codec을 fit하지 않는다. codec이 아직 없으면 origin-only baseline까지만 생성할 수 있다.

각 노트북은 `EXECUTE_STAGE`, `WRITE_OUTPUTS`, `REUSE_COMPLETED`, `ARTIFACT_STORAGE_MODE` 같은 실행 변수를 유지한다. 공통 all-in-one runner와 report notebook은 그대로 유지하며, RFW 단계는 독립된 step-by-step runbook이다.

2026-08-09 이전 completed LFW/SurvFace paper run에는 fitted codec 파일이 보존되지 않았다. 압축 transfer에는 새 코드로 완료되어 `frozen_codec_manifest.json`을 가진 같은 model UID의 run이 필요하다. `CODEC_SOURCE_RUN_DIRS`는 사용자가 직접 고정하며 과거 또는 최신 run을 자동 선택하지 않는다.

공통 `notebooks/common/orchestration/00_batch_experiment_runner.ipynb`의 `DATASET_IDS`에는 `rfw_custom`을 넣을 수 있다. 선택적인 `RUN_RFW_VERIFICATION`은 RFW-Official supplementary 평가이며 별도다. `notebooks/common/reports/00_cross_dataset_results.ipynb`는 RFW-Custom을 open-set join에 포함하고 RFW-Official TAR/FAR/EER 표는 별도 section/CSV로 보존한다.

`notebooks/common/orchestration/cross_dataset_calibration_transfer.ipynb`는 calibration source와 target을 독립적으로 선택한다. 같은 physical RFW population을 공유하는 Custom→Official 전이는 명시적 same-domain diagnostic으로만 허용하며 strict external-transfer 또는 strict unseen-identity 증거로 사용하지 않는다.

BalancedFace는 Step 7 범위에서 유예한다.
