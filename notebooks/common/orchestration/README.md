# 공통 실험 오케스트레이션

현재 실행 메뉴는 [전체 안내](C:/ronbun/notebooks/README.md)를 따른다.
이 폴더의 노트북을 항상 순서대로 모두 실행하지 않는다.

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
