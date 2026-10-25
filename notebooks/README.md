# 실행 순서: 01 → 02 → 03 → 04

활성 노트북은 네 개다. `configs/experiments/facex_workflow.yaml`을 공유한다.
첫 설정 셀의 DATASETS, CONFIG_PATH, EXECUTE, KEEP_RAW_RESULTS, OUTPUT_ROOT를 확인한다.
모듈 변경 후 Kernel Restart → Run All로 실행한다.

FaceX-Zoo와 ronbun은 별도 저장소와 연구 실행 단위를 유지한다.
모델 학습·export는 FaceX-Zoo에서 수행하고, ronbun은 export/sidecar와 분할 기록을 파일로 받아 실험한다.
ronbun의 모델 추론은 FaceX-Zoo Python 소스를 import하거나 학습 코드를 호출하지 않는다.

| 순서 | 노트북 | 역할 |
|---|---|---|
| 01 | [데이터·분할](01_data_preparation.ipynb) | FaceX-Zoo와 같은 hash 고정 역할 manifest 및 5개 데이터셋 상태 확인 |
| 02 | [입력](02_inputs.ipynb) | 등록된 MobileFaceNet/Resnet1024D로 CUDA 임베딩·raw norm 생성, threshold 평가에 필요한 품질 추출·재사용 |
| 03 | [실험](03_experiments.ipynb) | Origin/PCA/PQ·Global-safe/Continuous 또는 데이터셋별 rank 평가 |
| 04 | [결과](04_results.ipynb) | 정확한 REPORT_DIR의 완료·무결성·지표·paired 비교 확인 |

03은 `STUDY_STAGE="baseline"`부터 실행한다. 완료 후 `calibration_size`와 정확한
`BASELINE_REPORT`를 지정하면 같은 모델·codec으로 fit/safety 크기를 분석한다.
이 단계는 DATASETS에서 SurvFace/LFW/RFW만 선택하며, 최종 safety/test를 제외한 내부 validation을 쓴다.
TinyFace/SCFace는 rank 평가를 유지한다. FR/calibration 비율 재학습은 자동 실행하지 않는다.

FaceX-Zoo에서는 `fine-tune/notebooks/00 → 01 → 02 → 03 → 04`로 진행한다.
00은3개 파인튜닝 데이터셋과 backbone별 새 namespace를 설정한다.
03이 공유 역할 목록을 생성하고 해당 FR 인물만 학습한다. 04는 완료 모델을 export한다.
학습되지 않은 모델을 추측하거나 최신 checkpoint를 자동 선택하지 않는다.

완료 export를 등록한 후 YAML의 models에 정확한 spec 경로와 training_dataset을 지정한다.
`py -3.11 -m research.embeddings.facex_zoo EXPORT --registry-root REGISTRY_DIR`
기본 models가 비어 있는 것은 잘못된 checkpoint를 자동 사용하는 일을 막기 위한 설정이다.
LFW/RFW에서는 FR을 학습하지 않는다. 각 FR 모델은 학습한 데이터셋과 LFW/RFW를 평가한다.
DATASETS에서 일부만 선택해 작은 작업부터 실행할 수 있다.
선택한 데이터셋에 맞는 등록 모델이 없으면 준비 완료로 표시하지 않으며 실행 전에 중단한다.
데이터셋 목록을 확장한 경우 FaceX-Zoo 03/04가 출력한 `handoff_path`를 YAML의 handoff에 명시한다.
기존 handoff.json은 최초 snapshot으로 보존하며 새 버전은 handoffs/<hash>.json에 생성된다.

SurvFace80/20 및 fit/safety 분할은 사전 고정 후보다. TinyFace/SCFace는 rank,
LFW/RFW는 공식pair를 유지한 추가 calibration 실험으로 해석한다.
CR-FIQA가 기본 품질 신호이며 raw_norm은 명시적으로 선택하는 별도 조건이다.
SCFace/TinyFace는 작은 PQ codebook을 사용하므로 동일m이 동일 저장량을 뜻하지 않는다.

분할 수량·점수 공간·CI·보존·재개·ChatGPT 업로드 순서는
[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)를 따른다. False 보존 설정은 이번 실행의 상세 결과에만 적용한다.
입력 이미지·학습 checkpoint·이전 완료 결과는 삭제하지 않는다. 입력 cache와 codec도 별도 보존한다.
기존 노트북은 [legacy/](legacy/README.md)에 보존한다. 직전 LFW 전용4개도
`legacy/lfw_sequential_v2/`에 보존했다. 현재 순차 실행에서는 열 필요가 없다.
과거 DB512D/모델/평가 결과는 새1024D 연구 run과 별도 namespace로 유지된다.
