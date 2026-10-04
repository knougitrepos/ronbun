# 현재 LFW 기본 경로: 1:1 pair verification (2026-10-04)

UCFace의 LFW 6,000쌍 검증 방향을 반영했습니다. closed-set identification/BLUFR과 구분합니다.
LFW 준비 00의 `pair_verification`에서 목록을 준비하고, calibration 03의 동일 모드에서
`EXECUTE=False`로 점검한 뒤 `True`로 실행합니다. 기본은 4모델×10 folds×20 seeds=800 jobs이며
Origin+PQ 3종×3보정방법×5 FMR 목표를 유지합니다. 기존 7개 development 비율은 이 새 프로토콜에 적용하지 않습니다.
[전용 실행·지표·보존 안내](C:/ronbun/notebooks/calibration/LFW_PAIR_VERIFICATION.md)를 따르세요.

이미 완료된 전체 resize FR/FIQA를 재사용하고 PQ·pair score·보정만 새로 계산합니다.
`KEEP_RAW_RESULTS=False`는 전체 완료/요약 검증 후 이번 캠페인의 상세 checkpoint를 정리합니다.
부분 실행은 재개용 checkpoint를 남깁니다. 결과는 FMR/TAR이며 SurvFace 1:N TPIR/FPIR와 섞지 않습니다.
공통 batch 00은 점검, 01은 선택적 PQ/점수 준비, report는 명시한 pair report 읽기입니다.
기존 완료 결과와 아래 legacy/BLUFR 모드는 보존되며 명시적으로 선택할 때만 실행합니다.

---

## 이전 경로 안내 (보관)

# 공통 실험 오케스트레이션

현재 실행 메뉴는 [전체 안내](C:/ronbun/notebooks/README.md)를 따른다.
이 폴더의 노트북을 항상 순서대로 모두 실행하지 않는다.

## LFW 공개 분할 기반 보정

현재 기본 YAML은 `lfw_blufr_calibration_resize.yaml`이다. LFW 준비 00의 `blufr_resize_inputs`에서
전체 FR/FIQA 새 입력을 생성한 뒤 사용한다. 실행 순서는 [전체 안내](C:/ronbun/notebooks/README.md)를 따른다.

`blufr_calibration`은 공통 batch 00에서 읽기 전용 점검, 공통 calibration 01에서 선택적 입력 준비를 수행한다.
실제 보정은 calibration 03에서 입력 준비까지 이어서 처리하므로 00→01→03 순서가 필수는 아니다.
03에서 `EXECUTE=False`로 검사한 후 `ready=True`일 때 실행한다. 이전 detected/aligned source는 모델별 38장 누락 상태였으며 새 resize source를 사용한다.
01의 `LFW_MAX_NEW_INPUT_JOBS`는 입력 작업 수(전체 280개)를 제한한다.
공개 10 trials·7비율·100명 calibration gallery·1,000명 test gallery 설정과 선행조건은
[전체 안내](C:/ronbun/notebooks/README.md)를 따른다. 별도 benchmark Python은 보존하되 노트북에서 호출하지 않는다.
아래 내용은 `legacy` 모드의 기존 workflow다. 새 모드에서 다른 데이터셋이나 GPU 추출을 실행하지 않는다.

## Source run 생성과 입력 고정

00_batch_experiment_runner.ipynb는 모델 하나를 선택하여 DATASET_IDS의 source run을 생성하거나 재사용한다.
현재 설정의 dataset은 LFW, SurvFace, RFW-Custom, TinyFace다.
4 checkpoint × 3 open-set dataset 비교에는 12개 완료 run이 필요하다.
TinyFace는 별도의 보조 closed-set 평가다.

LFW/SurvFace의 원시 manifest가 없으면 활성 데이터 준비 노트북을 먼저 실행한다.
checkpoint 등록/smoke, aligned crop·landmark·Step 4 단계는 공통 runner가 호출한다.
과거 개별 단계와 DB 실험은 [보관 폴더](C:/ronbun/notebooks/_archive/README.md)에 남아 있다.
과거 pgvector exact/HNSW 실험이 현재 PQ ADC 실험으로 완전히 대체된 것은 아니다.

## 실행 tier와 모델 선택

| tier | LFW | SurvFace | RFW-Custom | TinyFace |
|---|---:|---:|---:|---:|
| quick 기본 비율 | 10% | 2% | 10% | 10% |
| full | 100% | 100% | 100% | 100% |

QUICK_DATA_FRACTIONS의 값을 바꾸면 effective config와 plan에 기록한다.
full은 이 사전과 관계없이 전체 비율이다. quick을 논문 최종 결과로 취급하지 않는다.

| MODEL_NAME | profile | checkpoint |
|---|---|---|
| arc | arcface_ms1mv3_r100 | models/arcface/ms1mv3_r100_backbone.pth |
| ada | adaface_ms1mv3_r100 | models/adaface/adaface_ir101_ms1mv3.ckpt |
| mag | magface_ms1mv2_iresnet100 | models/magface/magface_ms1mv2.pth |
| edge | edgeface_webface12m_xs_gamma_06 | models/edgeface/edgeface_xs_gamma_06.pt |

profile·checkpoint·model UID를 함께 확인한다. checkpoint마다 독립 run을 사용한다.

## 실행 및 재사용 설정

1. DATASET_IDS, MODEL_NAME, RUN_TIER, 모델 경로, 목표 FPIR를 확인한다.
2. EXECUTE, ACKNOWLEDGE_LOCAL_EXECUTION, START_NEW_RUN, COMPLETED_RUN_OVERRIDES를 확인한다.
3. 모델 등록/smoke 결과와 plan의 준비 상태·UID·출처를 확인한다.
4. 실행할 조건과 저장 위치를 확인한 뒤 Kernel Restart → Run All을 사용한다.

EXECUTE=False여도 그 앞의 prepare_common_model_checkpoint 호출은
run_smoke_validation=True이므로 필요할 때 GPU 검증과 registry/검증 결과 저장을 수행한다.
이 flag를 완전한 읽기 전용 모드로 설명하지 않는다.

완료 source를 재사용하려면 COMPLETED_RUN_OVERRIDES에 명시한 run을 사용한다.
START_NEW_RUN=True는 새 source 계산을 요청하는 설정이다.
현재 저장값을 확인하고 실행하며, 의도하지 않은 독립 반복을 만들지 않는다.
quick은 source/diff lineage를 기록하고, full 새 실행은 clean worktree 검사를 요구한다.

RUN_SEARCH_SPACE_REFRESH, RUN_FAITHFULNESS와 그 범위도 확인한다.
FAITHFULNESS_MAXIMUM_SAMPLES=None은 제한 없이 전체 후보, 양의 정수는 샘플 상한이다.
results_only도 모든 heatmap 생성·저장을 끄는 설정은 아니다.

RUN_FINAL_REPORT=True이면 활성 common/reports/00_cross_dataset_results.ipynb를 호출한다.
WRITE_FINAL_REPORT는 계산과 별도로 보고 파일 저장을 제어한다.
CROSS_MODEL_RUN_MATRIX는 12개 완료 run을 명시적으로 연결하는 비교 입력이다.
공통 보고의 직접 실행에서 model UID로 선택한 후보도 확인 후 명시적으로 고정한다.

## 보정 행렬과 보고서 집계

01_batch_fiqa_saliency_calibration.ipynb는 명시한 MODEL_RUN_MATRIX를 입력으로
3 dataset × 4 FR × PQ m128/m64/m32의 36조건을 비교한다.
EXECUTE=False는 입력 검사, True는 필요한 입력 준비와 보정/보고 실행이다.
첫 설정 셀의 현재 값이 기준이다. source batch의 새 run이 자동 전달되지는 않는다.
완료 조건/seed/family 결과는 설정·hash 검증 후 재사용한다.

01_batch_fiqa_saliency_calibration_compact.ipynb는 명시한 완료 matrix report를 읽어 집계한다.
새 fitting·GPU 추론을 수행하지 않지만 파생 compact 파일은 기록한다.
원본/PQ 대조의 요약은 calibration 03의 자체 분석 묶음을 사용한다.

cross_dataset_calibration_transfer.ipynb는 별도의 전이 실험이다.
source/target·physical dataset·score space·프로토콜을 확인한다.
RFW-Custom 1:N과 Official 1:1을 구분하고, 동일 물리 RFW population의 전이는
명시적 same-domain diagnostic 범위에서 해석한다.

## 수동 복구와 보조 실험

LFW/SurvFace 단계별 디버깅은 _archive/의 해당 dataset 수동 노트북을 사용한다.
초기 manifest 생성은 활성 lfw/survface 준비 경로를 사용한다.
RFW-Official은 활성 rfw/00_rfw_all_in_one.ipynb가 대표 진입점이며,
보관된 RFW 단계별 notebook은 독립 실행 창이다.
전체 source 완료만으로 형식적 FPIR 보장·ANN/DB 반복 benchmark·새 성능 기여가 검증되는 것은 아니다.
