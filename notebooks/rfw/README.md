# RFW 실행 구조

RFW는 이 연구에서 supplementary 1:1 face verification 데이터셋으로만 사용한다. RFW 결과를 LFW/SurvFace의 1:N open-set DIR@FPIR 결과와 합치거나 open-set headline으로 표현하지 않는다.

실행 순서는 다음과 같다.

1. `00_data_preparation/00_data_preparation.ipynb`: 공식 4개 group, 10-fold, 24,000 pair protocol과 source identity 목록을 고정한다.
2. `01_embeddings/00_rfw_origin_embedding_extraction.ipynb`: aligned BIN을 pair batch 단위로 streaming decode하고 선택한 FR 모델의 512D origin embeddings를 immutable artifact로 저장한다.
3. `02_compression/00_rfw_frozen_codec_verification.ipynb`: LFW 또는 SurvFace에서 fit하고 SHA를 고정한 PCA/PQ codec만 적용한다. RFW에서 codec을 fit하지 않는다.

각 노트북은 `EXECUTE_STAGE`, `WRITE_OUTPUTS`, `REUSE_COMPLETED`, `ARTIFACT_STORAGE_MODE` 같은 실행 변수를 유지한다. 공통 all-in-one runner와 report notebook은 그대로 유지하며, RFW 단계는 독립된 step-by-step runbook이다.

현재 completed LFW/SurvFace paper run에는 fitted codec 파일이 보존되지 않았다. 따라서 frozen-codec notebook은 SHA-pinned codec artifact가 새로 발행될 때까지 `CODEC_CONFIGS`를 비워 두고 fail-closed한다. 과거 run을 자동 선택하지 않는다.

BalancedFace는 Step 7 범위에서 유예한다.
