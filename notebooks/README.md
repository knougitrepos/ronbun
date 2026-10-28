# 실행 순서: 01 → 02 → 03 → 04

활성 노트북은 네 개다. `configs/experiments/facex_workflow.yaml`을 공유한다.
첫 설정 셀에서 데이터·모델 루트와 해당 단계의 설정만 확인한다.
01은 상태 검사만 하므로 EXECUTE/REPORT_DIR을 사용하지 않는다. 02·03은 EXECUTE=False가 기본이며,
03에서 KEEP_RAW_RESULTS를 지정한다. 04에서만 완료 결과의 REPORT_DIR을 직접 지정한다.
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
03은 외부의 고정 역할 목록을 검증하고 해당 FR 인물만 학습한다. 04는 완료 모델을 export한다.
학습되지 않은 모델을 추측하거나 최신 checkpoint를 자동 선택하지 않는다.

완료 export를 등록한 후 YAML의 models에 정확한 spec 경로와 training_dataset을 지정한다.
`py -3.11 -m research.embeddings.facex_zoo EXPORT --registry-root REGISTRY_DIR`
기본 models가 비어 있는 것은 잘못된 checkpoint를 자동 사용하는 일을 막기 위한 설정이다.
LFW/RFW에서는 FR을 학습하지 않는다. 각 FR 모델은 학습한 데이터셋과 LFW/RFW를 평가한다.
DATASETS에서 일부만 선택해 작은 작업부터 실행할 수 있다.
선택한 데이터셋에 맞는 등록 모델이 없으면 준비 완료로 표시하지 않으며 실행 전에 중단한다.
현재 공유 index는 `F:/face_datasets/protocols/shared_v1/handoff.json`이다.
두 저장소는 같은 역할 CSV와 인물 hash를 검증한다. 이전 저장소 내부의 index/완료 결과는 보존한다.
분할이 바뀌면 기존 파일을 덮어쓰지 않고 새로운 namespace와 정확한 handoff를 명시한다.

## 외부 데이터·모델 저장소

기본값은 데이터 `F:/face_datasets`, 모델 `F:/models`이다. `FACE_DATASET_ROOT`와
`FACE_MODEL_ROOT` 환경변수로 바꿀 수 있다. Linux 기본값은 `/data/face_datasets`, `/data/models`이며,
서버의 실제 마운트 위치로 설정한 뒤 커널을 시작한다. 활성 YAML의 `${DATASET_ROOT}`,
`${MODEL_ROOT}`, `${PROJECT_ROOT}`는 설정 로더가 해석한다.

현재 데이터의 이미지 내용과 학습/보정/test 역할은 변경하지 않았다. LFW의 이미지 manifest와 pairs,
RFW의 tar 참조 manifest와 공식 pair 표는 각 데이터셋의 `protocols/shared_v1`에 있다.
RFW는 압축 해제 없이 tar 내부를 읽는다. SurvFace/TinyFace/SCFace의 portable v2 프로토콜은
같은 데이터셋의 `protocols/shared_v1/<식별자>`에 있고, 원래 roles/train-list 바이트를 보존한다.

현재 로컬 자료를 처음 전환한 재현 명령은 다음과 같다. 기존 외부 사본을 먼저 검사해야 한다.
이 명령만 두 저장소의 위치를 입력받으며, 이후 실행에는 상대 저장소 소스가 필요하지 않다.

```powershell
py -3.11 -m research.datasets.shared_storage --project-root C:/ronbun --facex-root F:/FaceX-Zoo --dataset-root F:/face_datasets --model-root F:/models
```

이 명령은 원본 이미지와 기존 완료 결과를 이동/삭제하지 않고 새 메타데이터를 게시한다.
반복 실행은 source hash와 완료 bundle을 검증한다. 기존 bundle과 다른 입력은 새 버전이 필요하다.
새 서버에서는 이미 준비된 외부 데이터 tree를 그대로 마운트하고 01부터 검사한다.

새 압축 해제/전처리 결과의 README.md와 derivation.json은 같은 실행 기록에서 생성된다.
원본 hash, 처리 조건, 코드 버전, 개수, 완료/부분 상태를 확인한다. 과거 생성 이력을 추정하지 않는다.
공유 결과는 목적지별 lock을 사용하고 완료 상태 확인 후 읽는다. 잠금이 남으면 기록된 host/PID와
실제 프로세스를 확인하며 자동으로 훔치거나 삭제하지 않는다.

02의 입력/품질 cache는 재개와 검증에 필요한 자료다. 03의 KEEP_RAW_RESULTS는 평가 상세 결과를
제어하며 데이터셋 원본·공유 전처리 결과·모델·02 cache를 삭제하지 않는다. 외부 출력은 Git 밖에 있다.
기존 512D 모델의 보관과 현재 1024D 실험 등록은 별개이며, models={}이면 아직 실행 모델이 없는 상태다.
모델 등록은 `--registry-root F:/models/ronbun_registry`처럼 외부 경로를 명시할 수 있다.

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
