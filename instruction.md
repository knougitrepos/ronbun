# 실험 실행 순서

이 문서는 `step2` 브랜치에서 현재 존재하는 파일을 어떤 순서로 실행해야
하는지 설명한다. 현재 연구 단위는 ArcFace/AdaFace/MagFace **사전학습
checkpoint별 임베딩 압축 민감도 비교**이며, FR 모델을 새로 학습하지 않는다.

## 1. 가장 먼저 확인할 사항

프로젝트 루트는 `D:\ronbun`이며 PowerShell에서 다음을 실행한다.

```powershell
cd D:\ronbun
git branch --show-current
py -m pip install -r requirements.txt
py -m pip install -r requirements-step2.txt
```

- branch가 `step2`인지 확인한다.
- GPU용 PyTorch는 로컬 NVIDIA driver와 맞는 공식 wheel을 선택한다.
- 실제 checkpoint를 등록하기 전에는 노트북의 `EXECUTE_STAGE=False`,
  `WRITE_OUTPUTS=False`를 유지한다.
- 노트북은 중간 셀부터 실행하지 않는다. 커널을 재시작한 뒤 항상 위에서
  아래로 실행한다.
- 기존 `runs/step1` 결과와 완료된 run은 수정하거나 덮어쓰지 않는다.

## 2. 공통 실행 변수

각 노트북 상단에서 다음 값을 먼저 확인한다.

```python
MODEL_NAME = "arcface"  # "arcface", "adaface", "magface"
MODE = "dev"            # 빠른 점검은 "dev", 전체 논문 실행은 "real"
DATA_FRACTION = 0.10    # identity 단위 사용 비율
SEED = 42               # 부분집합 및 사례 선택 재현 seed
EXECUTE_STAGE = False   # 실제 계산을 시작할 때만 True
WRITE_OUTPUTS = False   # 검증 후 artifact를 저장할 때만 True
```

권장 실행 방식은 다음과 같다.

1. 경로와 설정을 채우고 `EXECUTE_STAGE=True`, `WRITE_OUTPUTS=False`로 검증한다.
2. 검증이 통과하면 커널을 재시작한다.
3. 같은 설정에서 `EXECUTE_STAGE=True`, `WRITE_OUTPUTS=True`로 처음부터 다시
   실행하여 새 artifact를 저장한다.

`MODE="real"`, `DATA_FRACTION=1.0`인 실행만 전체 데이터 논문 결과로
취급한다.

## 3. Step 2 기준 설정 확인

다음 파일은 실행 파일이 아니라 실험 범위와 기본값을 정하는 설정 파일이다.

1. `configs/experiments/step2_pytorch_gradcam.yaml`
2. `architect/20260723.md`

설정에서 반드시 확인할 항목:

- 비교 모델: ArcFace, AdaFace, MagFace
- embedding 차원: 512
- PCA-only 차원: 384/256/128/64/32
- PQ 입력: PCA 결과가 아닌 원본 512D
- `exact_fallback: false`
- Grad-CAM target: `origin_pair_cosine`
- hard PQ 연산은 미분하지 않음

현재 YAML의 checkpoint, loader, 전처리 및 target layer 값은 의도적으로
`null`이다. 실제 파일을 확인하지 않고 임의의 값으로 채우면 안 된다.

## 4. 데이터 준비

### 4.1 LFW

가장 먼저 다음 파일을 실행한다.

1. `notebooks/lfw/data_preparation.ipynb`

주요 출력:

- `data/interim/lfw/face_manifest.csv`
- identity-disjoint development/calibration/test 정보

### 4.2 SurvFace

LFW 검증 후 다음 파일을 실행한다.

1. `notebooks/survface/data_preparation.ipynb`

주요 출력:

- `data/interim/survface/training_manifest.csv`
- `data/interim/survface/official_manifest.csv`
- 공식 gallery/mated/unmated 순서와 역할

SurvFace 공식 test 행으로 PCA/PQ 또는 threshold를 학습하면 안 된다.

### 4.3 현재 Step 2 데이터 준비의 미구현 부분

Step 2 모델 비교에 필요한 다음 산출물을 만드는 전용 단계는 아직 구현되지
않았다.

- 공통 정렬 crop manifest:
  `data/interim/common/aligned_112_manifest.parquet`
- smoke test용 `aligned_faces` NPZ
- Grad-CAM 사례용 query/gallery pair bundle

따라서 데이터 준비 노트북만 실행했다고 Step 2 입력이 모두 만들어지는 것은
아니다. 이 연결 단계가 추가되기 전에는 실제 전체 Step 2 실행으로 진행하지
않는다.

## 5. 모델별 checkpoint 등록

ArcFace부터 시작하고, 성공한 뒤 AdaFace와 MagFace에 같은 절차를 반복한다.

1. `notebooks/model_validation/00_checkpoint_registration.ipynb`

노트북에서 다음 값을 직접 채운다.

- `CHECKPOINT_PATH`
- `CHECKPOINT_SOURCE_URL`
- `IMPLEMENTATION_REPOSITORY`
- `MODULE_FACTORY`
- `TARGET_LAYER`
- `SOURCE_COLOR_ORDER`
- `MODEL_COLOR_ORDER`
- `CHANNEL_MEAN`
- `CHANNEL_STD`

`MODULE_FACTORY`는 다음 형식의 로컬 Python callable이어야 한다.

```text
package.module:function
```

해당 함수는 등록된 `ModelSpec`을 받아 실제 `torch.nn.Module`을 반환해야 한다.
저장소에는 ArcFace/AdaFace/MagFace 공식 checkpoint 형식을 임의로 추측하는
loader가 포함되어 있지 않다.

저장 시 생성되는 manifest:

```text
runs/step2/model_registry/<model_uid>.json
```

이 manifest에는 checkpoint SHA-256, 전처리 hash, model UID와 target layer가
기록된다. 기존 manifest를 수정하지 않고 설정이 바뀌면 새 UID로 등록한다.

## 6. 모델 smoke test

checkpoint 등록 직후 다음 파일을 실행한다.

1. `notebooks/model_validation/01_preprocessing_and_model_smoke.ipynb`

필요한 입력:

- `MODEL_SPEC_PATH`: 앞 단계에서 생성한 JSON
- `SMOKE_CROPS_NPZ`: `aligned_faces` 키를 가진 uint8
  `[N, 112, 112, 3]` 배열
- `DEVICE`: 최초에는 `cpu`, 확인 후 필요하면 `cuda`

검증 항목:

- raw embedding shape가 `[N, 512]`인가
- L2 정규화 전 `raw_norm`이 보존되는가
- normalized embedding의 norm이 1인가
- checkpoint와 전처리 hash가 유지되는가
- 지정한 Grad-CAM target layer가 실제로 존재하는가

ArcFace smoke test가 통과한 뒤 AdaFace, MagFace 순으로 반복한다.

## 7. Step 2 정량 압축 실험

모델 smoke test 다음에는 각 PyTorch checkpoint에 대해 다음 순서가 필요하다.

1. 고정된 동일 crop에서 원본 512D embedding 추출
2. development split에서 모델별 PCA/PQ 학습
3. calibration/test에 고정 적용
4. geometry, score, rank, threshold 및 open-set 지표 계산
5. fallback-free paired/retrieval 결과 저장

그러나 이 과정을 실행하는 **전용 Step 2 정량 runner 또는 notebook은 아직
구현되지 않았다.**

기존 파일
`notebooks/lfw/06_step1_compression_characterization.ipynb`는 현재 ONNX
ArcFace Step 1 기준선용이므로 PyTorch 세 모델 Step 2 결과 생성기로 간주하면
안 된다.

Grad-CAM으로 진행하기 전에 최소한 다음 artifact가 필요하다.

- `paired_metrics.parquet`
  - `sample_id`
  - `compression_family`
  - `compression_profile`
  - `angular_error_rad`
  - `origin_fallback_used=False`
- `retrieval_metrics.parquet`
  - `query_id`
  - `top1_score_drift`
  - `agreement_with_origin`
  - `threshold_crossing`
  - `origin_fallback_used=False`
- 각 행의 `model_uid`와 checkpoint/preprocessing provenance

따라서 현재의 정상적인 중단 지점은 **모델 smoke test 완료 후**이다. 전용
PyTorch 정량 실행 단계가 구현되고 실제 artifact가 생성된 뒤 8절로 진행한다.

## 8. LFW Grad-CAM 후속 분석

Step 2 정량 artifact가 완성된 뒤에만 다음 순서로 실행한다.

### 8.1 입력과 모델 동결

1. `notebooks/lfw/gradcam/00_source_and_model_freeze.ipynb`

필요한 입력:

- `MODEL_SPEC_PATH`
- `PAIRED_METRICS_PATH`
- `RETRIEVAL_METRICS_PATH`
- `ALIGNED_CROP_MANIFEST_PATH`
- 새 `FREEZE_MANIFEST_PATH`

모든 입력 hash와 `model_uid`를 고정하며 fallback 사용 행이 있으면 중단한다.

### 8.2 분석 사례 선택

2. `notebooks/lfw/gradcam/01_case_selection.ipynb`

profile별로 다음 사례를 결정적으로 선택한다.

- `stable`
- `high_error`
- `rank_flip`
- `threshold_crossing`

출력은 새 `CASE_MANIFEST_PATH`에 저장한다.

### 8.3 pair bundle 준비

다음 노트북 전에 case manifest 순서와 정확히 일치하는 NPZ가 필요하다.

```text
case_id            문자열 배열
query_images       uint8 [N,112,112,3]
gallery_templates  float32 [N,512]
```

이 pair bundle을 생성하는 전용 notebook은 아직 구현되지 않았다. 임의로
순서를 맞추지 말고 `case_id`로 결합한 뒤 순서를 검증해야 한다.

### 8.4 pair-conditioned Grad-CAM

3. `notebooks/lfw/gradcam/02_pair_gradcam_generation.ipynb`

- query branch에만 gradient를 흘린다.
- gallery template은 detach한다.
- target은 원본 embedding cosine이다.
- PCA/PQ 또는 hard code를 직접 미분하지 않는다.

### 8.5 saliency 기술 통계

4. `notebooks/lfw/gradcam/03_saliency_feature_analysis.ipynb`

- saliency entropy
- 중앙 50% 집중도
- profile 및 사례군별 기술 통계

중앙 영역을 검증된 얼굴 영역 mask라고 표현하면 안 된다.

### 8.6 faithfulness 검증

5. `notebooks/lfw/gradcam/04_faithfulness_and_report.ipynb`

- high-saliency occlusion
- low-saliency control
- random control
- 원본 pair cosine score 감소량 비교

high-saliency 영역을 가린 결과가 control보다 일관되게 강하지 않으면 Grad-CAM
그림을 압축 오류의 원인 설명 근거로 사용하지 않는다.

## 9. 기존 Step 1 ONNX 기준선을 다시 실행할 경우

Step 2와 별개의 기준선 재현 순서이다.

### LFW

1. `notebooks/lfw/data_preparation.ipynb`
2. `notebooks/lfw/00_protocol_and_run_freeze.ipynb`
3. `notebooks/lfw/01_arcface_embedding_extraction.ipynb`
4. `notebooks/lfw/06_step1_compression_characterization.ipynb`

### SurvFace

1. `notebooks/survface/data_preparation.ipynb`
2. `notebooks/survface/00_official_protocol_and_run_freeze.ipynb`
3. `notebooks/survface/01_official_arcface_embedding_extraction.ipynb`
4. `notebooks/survface/06_step1_compression_characterization.ipynb`

두 데이터셋의 정량 결과가 완성되면 마지막에 다음 파일을 실행한다.

1. `notebooks/common/cross_dataset_results.ipynb`

기존 LFW 02~05와 SurvFace 02~05는 과거 DB/certification/fallback 시스템 재현
경로이다. 현재 Step 1/Step 2 주 실험 순서에 포함하지 않는다. 특히 origin
exact fallback을 포함한 파일을 새 압축 특성 결과 생성에 사용하지 않는다.

## 10. 현재 권장 진행 순서 요약

```text
환경 설치
  -> LFW 데이터 manifest 준비
  -> 공통 aligned crop 생성 단계 구현 필요
  -> ArcFace checkpoint 등록
  -> ArcFace smoke test
  -> AdaFace checkpoint 등록 및 smoke test
  -> MagFace checkpoint 등록 및 smoke test
  -> PyTorch 세 모델 정량 압축 runner 구현 필요
  -> 모델별 PCA-only/PQ-only 정량 실행
  -> fallback-free 결과 동결
  -> Grad-CAM case 선택
  -> case pair bundle 생성 단계 구현 필요
  -> pair Grad-CAM
  -> saliency 통계
  -> occlusion faithfulness
  -> 검증된 결과만 논문용 results로 선별
```

현재 바로 시작할 파일은
`notebooks/model_validation/00_checkpoint_registration.ipynb`이며, 실제
Step 2 전체 실행을 위해 다음 구현 우선순위는 공통 aligned crop 생성,
PyTorch 정량 압축 runner, Grad-CAM pair bundle 생성 순서이다.
