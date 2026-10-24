# 실행 순서: 01 → 02 → 03 → 04

현재 실행할 노트북은 이 폴더의 네 개뿐입니다. **LFW 1:1 pair verification**을 순서대로 진행합니다.
각 파일에서 첫 설정 셀을 확인하고 **Kernel Restart → Run All**을 실행하세요.

| 순서 | 파일 | 수행 내용 |
|---|---|---|
| 01 | [데이터 준비](01_data_preparation.ipynb) | image manifest와 고정 6,000 pair 목록 준비·분할 검증 |
| 02 | [입력 준비·점검](02_inputs.ipynb) | 완료 임베딩/FIQA 재사용 및 hash·coverage 확인, 필요 시 CUDA 입력 생성 |
| 03 | [보정·평가](03_calibration.ipynb) | Origin/PQ별 보정·평가 및 분석 요약 생성 |
| 04 | [결과 확인](04_results.ipynb) | 명시한 보고서의 상태·무결성·FMR/TAR/Accuracy 확인 |

## 설정과 단계 연결

01–03의 `CONFIG_PATH`는 같은 YAML을 사용합니다. 기본값은
`configs/experiments/lfw_pair_verification.yaml`입니다. 모델·fold·seed·압축 프로파일·입력 출처는
이 파일에서 관리하며, 기본 행렬은 4모델 × 10 folds × 20 seeds = 800 jobs입니다.
이 숫자는 계획이며 실제 완료 범위는 report manifest로 확인합니다.

- **기존 입력을 재사용할 때:** 01에서 목록을 검증하고, 02의 `PREPARE_INPUTS=False`로
  입력을 확인합니다. `ready=True`일 때 03으로 진행합니다.
- **입력이 없는 새 환경:** 원본 이미지와 모델 registry/checkpoint를 준비합니다.
  01에서 `CREATE_IMAGE_MANIFEST=True`, 필요 시 `DOWNLOAD_LFW_PAIRS=True`를 설정합니다.
  02에서 `PREPARE_INPUTS=True`, 목록이 없으면 `DOWNLOAD_INPUT_INDEX=True`로 설정합니다.
  입력 생성 YAML과 pair YAML의 source 경로가 일치해야 하며 CUDA가 없으면 중단합니다.
  생성이 끝나면 다시 재사용 설정으로 돌아갈 수 있습니다.
- **보정 실험:** 03에서 먼저 `EXECUTE=False`로 점검하고 `True`로 실행합니다.
  `MAX_NEW_JOBS_PER_RUN=1`은 한 job 점검, `None`은 전체 미완료 계산입니다.
- **결과 연결:** 03에서 출력한 정확한 경로를 04의 `REPORT_DIR`에 복사합니다.
  파일 수정 시각으로 최신 보고서를 자동 선택하지 않습니다.

01의 manifest 생성은 기존 canonical writer를 사용합니다. manifest의 과거 split 열은
pair 프로토콜의 역할 배정에 사용하지 않습니다. 02의 입력 생성 YAML 이름에 BLUFR가 남아 있지만
전체 이미지 목록/resize producer만 사용하며 BLUFR 평가를 실행하지 않습니다.

## 보존과 해석

03의 `KEEP_RAW_RESULTS=False`는 전체 완료와 요약 검증 후 이번 campaign 상세 checkpoint를 정리합니다.
부분 실행 checkpoint, 해석 안내, 요약·설정·출처는 보존합니다. 입력 이미지·모델·FR/FIQA 입력은 삭제하지 않습니다.
`OUTPUT_ROOT`로 출력 드라이브를 변경할 수 있습니다. 상세 NPY·품질 CSV·개별 결과는 Git에서 제외하고,
검증된 집계와 경량 manifest를 추적합니다.

이번 정리에서 해석 안내의 경로와 해당 Python 참조가 바뀌었습니다. 이 파일들도 구현 hash에
포함되므로 이전 코드로 만든 campaign과 새 실행의 UID가 달라집니다. 이전 결과는 그대로 보존되며
04에서 정확한 report 경로로 읽을 수 있습니다. 이전 checkpoint를 새 campaign으로 복사하지 않습니다.

LFW의 FMR/TAR와 1:N FPIR/TPIR는 구분합니다. 주요 표·counts·신뢰구간·완료 상태의 의미는
[결과 해석 안내](LFW_PAIR_VERIFICATION.md)를 따르세요.

## FaceX-Zoo MobileFaceNet 1024D 연결

FaceX-Zoo의 **완료된 비-smoke backbone export**와 같은 이름의 `_meta.json`을 준비한다.
원본 export를 수정하거나 TorchScript로 변환할 필요가 없다. 등록 모듈이 SHA-256, 완료 상태,
head·dataset·source config·전처리·가중치 차원을 검증한다.

```powershell
py -3.11 -m research.embeddings.facex_zoo "F:/path/to/EXPORT.pt" --registry-root "runs/facex_model_registry"
py -3.11 -m research.experiments.facex_lfw_config --model "surv_arc_s8972=REGISTERED_SPEC.json" --output-dir "configs/experiments/facex_lfw_run1"
```

`EXPORT.pt`는 실제 export 경로, `REGISTERED_SPEC.json`은 첫 명령이 출력한 정확한 등록 파일로 바꾼다.
여러 checkpoint는 `--model alias=path`를 반복한다. 같은 학습 head라도 dataset·seed가 다르면
서로 다른 alias를 사용한다. `--artifact-root F:/ronbun_results/<이름>`으로 출력 위치를 지정할 수 있다.
설정 생성은 학습·추출을 실행하지 않는다. 기존 설정 폴더를 덮어쓰지 않는다.

01–03의 `CONFIG_PATH`를 생성된 `pairs.yaml`로 바꾼다. 02는 `INPUT_CONFIG_PATH=None`이면
그 YAML의 `input_config_path`를 읽는다. 새 입력은 `PREPARE_INPUTS=True`로 CUDA에서 생성한다.
LFW resize producer는 RGB를 공급하므로 기본 RGB-source 등록을 사용하며 adapter가 BGR로 변환한다.
모델 입력은 112×112, `(pixel−127.5)/128`이다. 이미지 crop/alignment 정책까지 FaceX 학습과
동일하다고 주장하는 것은 아니며, LFW는 기존 deepfunneled whole-image resize 정책을 유지한다.

원본 임베딩·raw norm·PQ codec·source UID의 차원을 검증한다. PQ는 `pq_1024_m128_b8`처럼
원본 차원을 기록한다. 04는 운영 결과와 paired 비교를 함께 표시한다. Continuous-FIQA 개선은
Global 대비 TAR 차이와 양쪽 실제 FMR를 같이 읽는다. 커널을 재시작한 뒤 01부터 실행한다.

현재 01–04는 **LFW 1:1 실행점**이다. 모든 데이터셋용 통합 launcher는 아니다.
공통 artifact/Step2 검색 평가 및 TinyFace 모듈도 1024D를 지원하지만, 기존 DB `vector(512)`
경로·512D 고정 과거 orchestrator는 그대로 분리되어 있다. DB에 1024D를 넣으면 사전 오류로 중단한다.
RFW 공식 pair Continuous-FIQA와 SCFace 실행 경로가 이 연결로 새로 구현되는 것은 아니다.
SurvFace 학습 전 holdout 권고는 [분할 보고서](../publish/FaceXZoo_학습분할_권고_20261022_v1.md)를 따른다.
그 권고를 FaceX-Zoo training list에 적용한 상태는 아니다.

## 이전 노트북

기존 52개 노트북과 과거 안내는 모두 [legacy/](legacy/README.md)에 보관했습니다.
현재 순차 실행에서는 열거나 실행할 필요가 없습니다. 이전 1:N/SurvFace/RFW/진단/관리 경로는
과거 실험의 재현을 위해 보존합니다. 완료 run·입력 데이터·checkpoint는 이동하거나 삭제하지 않았습니다.
