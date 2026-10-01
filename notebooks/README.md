# 노트북 실행 안내

현재 활성 노트북은 12개이고, 선택 실험·과거 수동 경로 40개는
[보관 안내](C:/ronbun/notebooks/_archive/README.md)에 정리했다.
파일 번호는 각 workflow 안의 이름이다. 전체 노트북을 번호순으로 모두 실행하지 않는다.

## LFW 공개 기준과 별도 보정 실험 (2026-10-01)

첫 코드 셀의 `LFW_PROTOCOL_MODE` 기본값은 `legacy`다. 기존 설정·결과를 유지한다.
아래 두 경로는 서로 다른 연구 질문이며 같은 결과로 합치지 않는다.

| 목적 | 노트북과 설정 | 실행 범위 |
|---|---|---|
| BLUFR 공개 목록 준비 | LFW 준비: `LFW_PROTOCOL_MODE="blufr_benchmark"`, 최초에만 `DOWNLOAD_BLUFR_CONFIG=True` | SHA가 고정된 MAT 다운로드·이미지 목록 대응 검증 |
| BLUFR 압축 기준 평가 | 공통 batch 00: `LFW_PROTOCOL_MODE="blufr_benchmark"` | 기존 `MODEL_NAME`의 한 모델, YAML의 10 trials; `EXECUTE=False`로 누락 확인 후 실행 |
| 기존 LFW 분할의 보정 조건 개선 | calibration 03: `LFW_PROTOCOL_MODE="matched_calibration"` | `MODELS`의 LFW만, 기존 test 유지·calibration gallery/enrollment를 test와 일치 |
| 새 보정 입력만 먼저 준비 | 공통 calibration 01: `LFW_PROTOCOL_MODE="matched_calibration"` | 선택 단계. 03에서도 누락된 matched 입력을 준비함 |
| 완료 결과 읽기 | 공통 보고: `LFW_PROTOCOL_MODE="protocol_report"`, `LFW_PROTOCOL_REPORT_DIR` 명시 | 새 실행이 반환한 `report_dir`; 자동 latest 선택 없음 |

모듈 변경 후 **Kernel Restart → Run All**한다. BLUFR의 trial과 03의 `PARTITION_SEEDS`는 서로 다르다.
03 matched 모드에서도 `EXECUTE`, `MODELS`, `PARTITION_SEEDS`, `FIT_SETTINGS`, `MAX_NEW_JOBS_PER_RUN`,
`BLAS_THREADS`, 메모리 점검값을 사용한다. `DATASETS`, `REUSE_PQ_MODELS`, `SOURCE_REPORT_DIR`는 legacy 경로용이다.
matched 모드는 기존 PQ codebook을 재사용하지만 threshold 모델은 새로 적합한다.
기존 전체 3-dataset 행렬은 legacy 모드로 그대로 실행한다.

**현재 BLUFR 실행 전제는 충족되지 않았다.** 네 모델 모두 13,233장 중 38장의 임베딩이 없다.
공통 batch의 새 모드가 누락 목록을 표시하며 정식 실행을 차단한다. 이미지를 제외하거나 기존 결과를 BLUFR로 이름만 바꾸지 않는다.
공개 목록은 저자 배포 링크 접근 실패 후 toolkit mirror에서 확인했으며 출처·SHA는
`configs/experiments/lfw_blufr.yaml`에 명시했다. 저자 인증 여부는 별도 한계다.

BLUFR 결과는 `results/lfw_blufr/`, matched 결과는 `results/calibration/lfw_matched/`에 저장한다.
완료 job은 SQLite에서 재사용하며, 채팅에는 반환된 `chat_dir`의 `START_HERE.md`와 `analysis.zip`을 사용한다.
공개 benchmark는 test curve 성능이고, matched 결과는 별도 calibration에서 정한 threshold의 test 성능이다.

## 현재 원본/PQ 실험을 이어갈 때

[calibration 03](C:/ronbun/notebooks/calibration/03_origin_vs_pq_fiqa_calibration.ipynb)을 사용한다.
명시한 완료 source run·PQ/FIQA 입력·PQ 모델 보고서가 준비되어 있으면 다른 batch나
calibration 노트북을 먼저 재실행할 필요가 없다.

- 실험 범위: DATASETS, MODELS, 압축 프로파일, FPIR, PARTITION_SEEDS.
- 출력 표 선택: DISPLAY_*. 실험 조건을 제한하지 않는다.
- 이번 호출의 새 계산 수: MAX_NEW_JOBS_PER_RUN. 전체 계획은 유지한다.
- 완료 checkpoint는 설정·hash 검증 후 재사용한다. 부분 결과는 완료 범위를 확인한다.
- 원본/PQ 상세 보고와 채팅 분석 ZIP은 자체 생성한다. PQ matrix compact는 다른 보고서용이다.

## 목적별 실행 메뉴

| 목적 | 활성 진입점 | 사용 조건 |
|---|---|---|
| LFW 최초 manifest 준비 | [LFW 준비](C:/ronbun/notebooks/lfw/00_data_preparation/00_data_preparation.ipynb) | 신규 환경 또는 원본/분할 변경 |
| SurvFace training·official manifest 준비 | [SurvFace 준비](C:/ronbun/notebooks/survface/00_data_preparation/00_data_preparation.ipynb) | 신규 환경 또는 원본 프로토콜 변경 |
| 모델별 source run 생성/재사용 | [공통 batch 00](C:/ronbun/notebooks/common/orchestration/00_batch_experiment_runner.ipynb) | 새 모델·데이터·프로토콜 또는 미완료 source 작업 |
| 전체 PQ FIQA+saliency 행렬 | [공통 calibration 01](C:/ronbun/notebooks/common/orchestration/01_batch_fiqa_saliency_calibration.ipynb) | 고정된 12개 완료 run, 36개 PQ 조건 |
| 완료 PQ matrix 보고서 요약 | [PQ compact](C:/ronbun/notebooks/common/orchestration/01_batch_fiqa_saliency_calibration_compact.ipynb) | fitting/GPU 재실행 없이 채팅용 집계 |
| 원본/PQ 동일 보정 대조 | [calibration 03](C:/ronbun/notebooks/calibration/03_origin_vs_pq_fiqa_calibration.ipynb) | 현재 우선 대조 실험 |
| margin/distortion/runner-up 단일 조건 ablation | [calibration 01](C:/ronbun/notebooks/calibration/01_fiqa_continuous_retrieval_conditioned_calibration.ipynb) | 별도의 retrieval feature 질문 |
| dataset 간 보정 전이 | [전이 실험](C:/ronbun/notebooks/common/orchestration/cross_dataset_calibration_transfer.ipynb) | 별도 전이 질문 |
| 압축 이후 FPIR 실패 진단 | [FPIR 진단](C:/ronbun/notebooks/diagnostics/00_compression_fpir_failure_diagnosis.ipynb) | false-accept 교체·점수 꼬리 분석 |
| RFW-Official 1:1 보조 평가 | [RFW all-in-one](C:/ronbun/notebooks/rfw/00_rfw_all_in_one.ipynb) | Custom 1:N과 별도 평가 |
| 여러 완료 run의 압축·검색 보고 | [공통 보고](C:/ronbun/notebooks/common/reports/00_cross_dataset_results.ipynb) | 공통 batch에서 자동 호출 또는 명시적 입력으로 보고 |
| DB/run reset | [관리 도구](C:/ronbun/notebooks/common/maintenance/00_selective_cleanup.ipynb) | 특정 run 폐기·격리. 일반 notebook 정리에 사용하지 않음 |

## 새 환경의 준비와 입력 연결

1. 데이터 원본과 모델 checkpoint를 준비한다.
2. LFW/SurvFace manifest가 없으면 위의 최초 준비 노트북을 실행한다.
   원본 목록·분할을 검증한 뒤 WRITE_OUTPUTS에 따라 기록한다.
   공통 runner가 이 원시 manifest 준비까지 대신하지 않는다.
3. 공통 batch 00에서 모델·데이터셋·실행 tier·run 선택·출력 정책을 확인한다.
   checkpoint 등록/smoke, aligned crop·landmark 및 Step 4 단계를 공통 runner가 호출한다.
4. 생성한 run의 완료 상태와 조건을 확인하고 calibration의 명시적 run matrix에 연결한다.
   공통 batch의 새 출력이 다음 calibration 노트북에 자동 전달되지는 않는다.
5. 목적에 맞는 calibration 또는 보고 진입점을 실행한다.

공통 batch 00은 모델 하나씩 선택한다. 현재 데이터셋 설정은 LFW/SurvFace/RFW-Custom/TinyFace다.
전체 PQ calibration은 LFW/RFW-Custom/SurvFace × 4 FR × PQ m128/m64/m32이며 TinyFace는 보조 평가다.
RFW-Custom은 raw/ aligned-bin archive에서 bundle을 구성하므로 보관된 RFW-Official 준비 노트북이
Custom 경로의 필수 선행 단계는 아니다.

## 실행·재사용 계약

첫 설정 셀을 확인한 후 Kernel Restart → Run All을 사용한다.
저장 출력과 실행 번호는 과거 기록이며 현재 설정의 결과나 완료 상태를 증명하지 않는다.
노트북마다 실행·쓰기 flag가 다르므로 같은 기본값을 일괄 가정하지 않는다.

공통 batch 00의 EXECUTE=False는 주요 pipeline 실행을 끄지만, 앞선 checkpoint 등록과
run_smoke_validation=True 호출은 필요할 때 GPU 검증·registry/검증 결과 저장을 수행할 수 있다.
읽기 전용 여부를 이 flag 하나로 판단하지 않는다.
START_NEW_RUN, 완료 run override, faithfulness 범위도 함께 확인한다.

공통 보고는 실행기가 정확한 경로로 호출하므로 이동하지 않는다.
독립 실행의 model UID 기반 자동 후보 선택과 배치의 명시적 run 주입을 구분한다.
논문에 사용할 run은 선택 결과를 확인해 RUN_IDS/MODEL_RUN_MATRIX로 고정한다.
완료 run과 content-addressed 결과를 덮어쓰지 않는다.

상세 설정은 [orchestration 안내](C:/ronbun/notebooks/common/orchestration/README.md),
프로토콜 구분은 [RFW 안내](C:/ronbun/notebooks/rfw/README.md)를 따른다.
