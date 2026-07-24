# 실험 실행 순서

이 문서는 `step2` 브랜치에서 현재 존재하는 파일을 어떤 순서로 실행해야
하는지 설명한다. 현재 연구 단위는 ArcFace/AdaFace/MagFace **사전학습
checkpoint별 임베딩 압축 민감도 비교**이며, FR 모델을 새로 학습하지 않는다.

## 1. 가장 먼저 확인할 사항

프로젝트 루트는 이전 `D:\ronbun`에서 `C:\ronbun`으로 이전되었다. 앞으로 PowerShell에서 다음을 실행한다.

```powershell
cd C:\ronbun
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
2. `architect/20260724.md`의 population Grad-CAM 절

설정에서 반드시 확인할 항목:

- 비교 모델: ArcFace, AdaFace, MagFace
- embedding 차원: 512
- PCA-only 차원: 384/256/128/64/32
- PQ 입력: PCA 결과가 아닌 원본 512D
- `exact_fallback: false`
- Pass A embedding 범위: 모든 선택 이미지
- Grad-CAM target: `origin_leave_one_out_identity_cosine`
- Pass B Grad-CAM 범위: 동일인 LOO target이 있는 모든 선택 이미지
- singleton/identity 미공개 표본: 행·임베딩 유지, target 부적격 사유 기록
- hard PQ 연산은 미분하지 않음
- 대표 사례 선택: 전체 결합 분석 뒤 시각화 전용

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

### 4.3 RFW

RFW는 다음 파일에서 먼저 공식 1:1 protocol을 검증한다.

1. `notebooks/rfw/data_preparation.ipynb`

주요 출력:

- `data/interim/rfw/image_manifest.csv`
- `data/interim/rfw/pair_protocol.csv`
- `data/interim/rfw/landmarks.csv`
- `data/interim/rfw/source_identities.txt`
- `data/interim/rfw/_SUCCESS`

RFW는 1:N open-set 데이터가 아니며 RFW test로 PCA/PQ를 fit하지 않는다.
`MODE="real"`에서는 `DATA_FRACTION=1.0`만 허용한다. 현재 세 모델 후보가
MS1MV2로 표기되어 있으므로 protocol manifest를 만들 수는 있지만 RFW
headline 모델 평가는 checkpoint overlap gate에서 차단된다.

### 4.4 BUPT-BalancedFace(Equalizedface)

RFW의 전체 source identity artifact를 만든 뒤 다음 파일을 실행한다.

1. `notebooks/balancedface/data_preparation.ipynb`

주요 출력:

- `data/interim/balancedface/source_index_manifest.csv`
- `data/interim/balancedface/excluded_rfw_overlap_identities.csv`
- `data/interim/balancedface/_SUCCESS`

BalancedFace는 FR 모델 재학습이나 최종 test에 사용하지 않는다. RFW와 겹치는
provider identity를 제거한 뒤 development는 PCA/PQ fit, calibration은
threshold 보정 후보로만 사용한다.

`data/raw/RFW-balancedface/images/Equalizedface.tar.gz`는 정상 파일로
교체되어 EOF 검사를 통과했다. 다만 Asian/Indian JPG에는 가변 해상도 이미지가
포함되므로 공통 얼굴 정렬과 그룹별 성공률 검증 전에는 embedding 입력으로
사용하지 않는다. RecordIO 경로는 source index까지만 구현되어 있고 decoder가
아직 없다. 따라서 위 source index를 실제 image manifest로 사용하면 안 된다.

### 4.5 현재 Step 2 데이터 준비의 미구현 부분

Step 2 모델 비교에 필요한 다음 산출물을 만드는 전용 단계는 아직 구현되지
않았다.

- 공통 정렬 crop manifest:
  `data/interim/common/aligned_112_manifest.parquet`
- 공통 정렬 crop 배열:
  uint8 `[N,112,112,3]` `.npy`
- smoke test용 aligned crop bundle
- 랜드마크 영역을 사용할 경우 정렬 좌표계의 검증된 dense landmark/face mask

과거의 Grad-CAM 사례용 query/gallery pair bundle은 더 이상 선행 입력이 아니다.
전체 원본 embedding을 먼저 추출한 뒤 코드가 동일인 leave-one-out template을
생성한다.

따라서 데이터 준비 노트북만 실행했다고 Step 2 입력이 모두 만들어지는 것은
아니다. 이 연결 단계가 추가되기 전에는 실제 전체 Step 2 실행으로 진행하지
않는다.

## 5. 모델별 checkpoint 등록

ArcFace부터 시작하고, 성공한 뒤 AdaFace와 MagFace에 같은 절차를 반복한다.

1. `notebooks/model_validation/00_checkpoint_registration.ipynb`

노트북에서 다음 값만 직접 확인·지정한다.

- `CHECKPOINT_PATH`

`CHECKPOINT_SOURCE_URL`은 선택 모델의 공식 model-zoo 페이지를 기본값으로
사용하며, 실제 파일을 다른 공식 링크에서 받았다면 그 링크로 바꾼다.
architecture, repository, `MODULE_FACTORY`, target layer, model color order,
mean/std는 `configs/experiments/step2_pytorch_gradcam.yaml`에서 모델별로 자동
선택된다. 공통 crop source 형식도 `RGB uint8 NHWC`로 고정되어 자동 입력된다.
현재 내장 factory는 다음 세 개다.

```text
research.embeddings.pytorch.official_loaders:load_arcface_checkpoint
research.embeddings.pytorch.official_loaders:load_adaface_checkpoint
research.embeddings.pytorch.official_loaders:load_magface_checkpoint
```

각 factory는 등록된 `ModelSpec`을 받아 공식 repository와 state-dict key가
호환되는 inference backbone을 만들고, 누락·추가·shape mismatch가 하나라도
있으면 checkpoint 로드를 중단한다. AdaFace는 정규화 전 512D를 반환하여
`raw_norm`이 소실되지 않도록 한다.

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

- `MODEL_REGISTRY_ROOT`: 기본 `runs/step2/model_registry`
- `MODEL_NAME`: `arcface`, `adaface`, `magface`
- `MODEL_UID`: 같은 family가 하나이면 `None`; 둘 이상이면 정확한 UID
- `SMOKE_INPUT_PATH`: 기본 `None`. 이 경우 LFW manifest가 가리키는
  deep-funneled 이미지에서 identity가 겹치지 않는 소량 표본을 자동 선택하고
  112×112 입력을 메모리에서 생성한다. 이미 검증된 uint8 `[N,112,112,3]`
  `.npy` 또는 `aligned_faces` 키의 `.npz`가 있으면 선택적으로 지정한다.
- `DEVICE`: 최초에는 `cpu`, 확인 후 필요하면 `cuda`

자동 LFW 입력은 checkpoint 로드·전처리·출력 shape·target layer만 확인하는
smoke 전용이다. 해당 resize 결과를 정량 압축 실험이나 population Grad-CAM의
공통 정렬 crop으로 사용하지 않는다.

검증 항목:

- raw embedding shape가 `[N, 512]`인가
- L2 정규화 전 `raw_norm`이 보존되는가
- normalized embedding의 norm이 1인가
- checkpoint와 전처리 hash가 유지되는가
- 지정한 Grad-CAM target layer가 실제로 존재하는가

ArcFace smoke test가 통과한 뒤 AdaFace, MagFace 순으로 반복한다. 해당 family의
ModelSpec이 정확히 하나이면 자동 선택하고, 없거나 둘 이상이면 실패 폐쇄형으로
중단한다.

이후 LFW Grad-CAM 00도 같은 registry에서 `MODEL_NAME`과 선택적 `MODEL_UID`로
ModelSpec을 자동 선택한다. 01은 freeze manifest의 exact `model_uid`, 02는
Pass-A artifact의 exact `model_uid`를 사용하므로 ModelSpec JSON 경로를 다시
입력하지 않는다.

## 7. LFW 원본 embedding·Grad-CAM population 추출

모델 smoke test가 끝나면 압축 결과나 사례 목록보다 먼저 다음 노트북을
실행한다.

### 7.1 입력·범위·모델 동결

1. `notebooks/lfw/gradcam/00_source_and_model_freeze.ipynb`

필요한 입력:

- 검증된 `MODEL_SPEC_PATH`
- LFW source manifest
- 공통 `ALIGNED_CROP_MANIFEST_PATH`
- uint8 `[N,112,112,3]` `ALIGNED_FACES_NPY_PATH`
- 새 selected manifest와 freeze manifest 출력 경로

`DATA_FRACTION`은 split별 identity 단위로 적용한다. 이 단계에는 paired/retrieval
압축 결과가 필요하지 않다.

### 7.2 Pass A와 동일인 LOO template

2. `notebooks/lfw/gradcam/01_origin_embedding_and_loo_templates.ipynb`

- 모든 선택 이미지의 raw 512D, raw norm, unit embedding을 추출한다.
- 같은 `template_scope_id`와 identity의 다른 이미지 embedding 합에서 자기
  embedding을 빼고 정규화한다.
- singleton 또는 identity 누락 표본은 embedding과 행을 유지하고
  `saliency_target_eligible=False` 및 사유를 기록한다.
- 다른 target을 대신 사용하지 않는다.

### 7.3 Pass B population Grad-CAM

3. `notebooks/lfw/gradcam/02_population_gradcam_extraction.ipynb`

- 모든 LOO-eligible 이미지에 대해 query branch만 미분한다.
- detached LOO template과 원본 embedding cosine을 scalar target으로 사용한다.
- Pass A와 Pass B embedding cosine 및 target score를 재검증한다.
- normalized/raw/ReLU CAM, channel weight, 4사분면·공간 요약값과 occlusion
  faithfulness를 immutable shard로 저장한다.
- full activation/gradient는 기본적으로 영구 저장하지 않는다.

### 7.4 coverage·공간 특징·faithfulness 검증

4. `notebooks/lfw/gradcam/03_saliency_feature_validation.ipynb`

- 전체 선택 행 수, LOO 적격률, heatmap 유효률을 함께 보고한다.
- high-saliency, low-saliency, sample-id seeded random occlusion score drop을
  비교한다.
- 검증된 dense landmark 또는 face mask가 없으면 눈·볼·턱·얼굴 외부 수치를
  임의 생성하지 않고 결측으로 남긴다.

LFW 현재 manifest 기준 13,233장 중 singleton identity 이미지 4,069장은 동일인
LOO target을 만들 수 없다. 따라서 원본 embedding coverage는 100%지만 예상
Grad-CAM target coverage는 9,164장, 약 69.25%이다. 실제 selected fraction마다
coverage를 다시 계산해 artifact에 기록한다.

## 8. Step 2 압축·전체 결합·대표 사례

### 8.1 정량 압축 결과 연결

5. `notebooks/lfw/gradcam/04_step2_compression_characterization.ipynb`

동일한 `origin_embedding_artifact_uid`의 원본 512D를 사용해 모델별로 다음을
생성해야 한다.

1. development split에서 PCA-only와 원본 512D PQ-only 학습
2. 고정된 compressor를 calibration/test에 적용
3. geometry, score, rank, threshold 및 open-set 지표 계산
4. fallback-free `paired_metrics.parquet`와 `retrieval_metrics.parquet` 저장

그러나 실제 PCA/PQ fitting과 open-set 평가를 수행하는 **전용 PyTorch Step 2
정량 runner는 아직 구현되지 않았다.** 04 노트북은 임의로 fitting 코드를
복제하지 않고 runner 산출물의 fallback·profile·lineage를 검사한다. runner
출력 경로가 없으면 정상적으로 중단한다.

기존 `notebooks/lfw/06_step1_compression_characterization.ipynb`는 ONNX ArcFace
Step 1 기준선용이므로 PyTorch 세 모델 Step 2 결과 생성기로 간주하면 안 된다.

### 8.2 전체 표본 결합·관계 분석

6. `notebooks/lfw/gradcam/05_saliency_compression_join.ipynb`

- 결합 키:
  `extraction_uid + dataset_id + sample_id + model_uid`
- 필수 lineage:
  `origin_embedding_artifact_uid`
- embedding distortion은 전체 이미지, retrieval 민감도는 protocol query
  subset일 수 있으며 availability를 별도로 기록한다.
- 모델과 압축 profile을 pooling하지 않고 identity-cluster bootstrap으로
  관계를 추정한다.
- Grad-CAM 특징을 embedding에 이어 붙이거나 PCA/PQ 입력으로 사용하지 않는다.

### 8.3 마지막 대표 사례 시각화

7. `notebooks/lfw/gradcam/06_representative_case_visualization.ipynb`

전체 결합 분석이 끝난 뒤에만 `stable`, `high_error`, `rank_flip`,
`threshold_crossing` 예시를 결정적으로 선택한다. 이미 저장된 heatmap을 읽어
그림을 만들며 Grad-CAM을 다시 계산하지 않는다. 사례 그림은 집단 통계의 보조
설명이고 압축 변화의 인과 증거가 아니다.

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
  -> SurvFace 공식/training manifest 준비
  -> RFW 공식 1:1 protocol 준비
  -> BalancedFace RFW-overlap 제거 source index 준비
  -> BalancedFace source 선택
     ├─ JPG: 공통 alignment/materializer 구현 필요
     └─ RecordIO: decoder/materializer 구현 필요
  -> 공통 aligned crop 생성 단계 구현 필요
  -> ArcFace checkpoint 등록
  -> ArcFace smoke test
  -> AdaFace checkpoint 등록 및 smoke test
  -> MagFace checkpoint 등록 및 smoke test
  -> 전체 선택 이미지 Pass A 원본 512D 추출
  -> split/identity별 leave-one-out template 생성
  -> 모든 eligible 이미지 Pass B Grad-CAM
  -> 공간 특징 및 전량 faithfulness 검증
  -> PyTorch 세 모델 정량 압축 runner 구현 필요
  -> 모델별 PCA-only/PQ-only 정량 실행
  -> fallback-free 결과 동결
  -> 원본 공간 특징과 압축 민감도 strict join
  -> 모델·profile별 identity bootstrap 관계 분석
  -> 마지막 대표 사례 선택·기존 heatmap 시각화
  -> 검증된 결과만 논문용 results로 선별
```

새 데이터 준비는 `notebooks/rfw/data_preparation.ipynb`부터 시작하고 그 다음
`notebooks/balancedface/data_preparation.ipynb`를 실행한다. 모델 쪽 독립 작업은
`notebooks/model_validation/00_checkpoint_registration.ipynb`부터 시작한다.
실제 Step 2 전체 실행을 위해 다음 구현 우선순위는 BalancedFace source 선택과
alignment/decoder materializer, 공통 aligned crop 생성, PyTorch 정량 압축
runner 순서이다. 사례 pair bundle은 더 이상 선행 artifact가 아니다.

## 11. 특정 run을 일관되게 리셋할 경우

연구 실행 순서와 별개의 유지보수 절차이다. 다음 노트북을 사용한다.

1. `notebooks/database/selective_cleanup.ipynb`

일반적인 리팩토링·재실행 전 정리에는 DB만 선택 삭제하지 말고
`RESET_MODE="complete_run_reset"`을 사용한다.

```python
RESET_MODE = "complete_run_reset"
RUN_UID = ""
CONNECT_TO_DATABASE = False
EXECUTE_RESET = False
ALLOW_COMPLETED_RUN_RESET = False
ALLOW_PROMOTED_RESULTS_RESET = False
ALLOW_UNVERIFIED_LINEAGE_RESET = False
CONFIRMATION_TOKEN = ""
```

권장 순서:

1. `RUN_UID`만 정확히 입력하고 나머지 안전 기본값을 유지한 채 커널을
   재시작하여 위에서 아래까지 실행한다.
2. DB 연결이 필요한 preview 단계에서만 `CONNECT_TO_DATABASE=True`로 바꾸고
   다시 처음부터 실행한다.
3. 미리보기에서 다음 대상을 각각 확인한다.
   - 원본·압축·PQ image embedding, template, split/search/calibration 및
     `research_runs` 부모 행
   - exact-owner `runs/**/<run>`의 임베딩 전처리·추출, PCA/PQ
     model/codebook, Grad-CAM/LOO, 평가·그림·로그
   - manifest의 `run_uid`가 일치하는 `results/**/result_manifest.json` bundle
   - 내용이 같은 run을 가리키는 `active_run.json`
4. 완료 run이면 정말 폐기할 때만 `ALLOW_COMPLETED_RUN_RESET=True`, 논문용
   결과가 포함되면 `ALLOW_PROMOTED_RESULTS_RESET=True`, 소유권 manifest가
   불완전한 고아 lineage이면 별도 확인 후
   `ALLOW_UNVERIFIED_LINEAGE_RESET=True`로 새 계획을 만든다.
5. override 또는 대상 파일이 달라질 때마다 이전 확인 문자열을 버리고
   plan digest를 새로 만든다.
6. 미리보기의 DB identity, 행 수, 로컬 경로·파일 수·byte, 보존 대상과
   격리 경로를 확인한다.
7. 출력된 확인 문자열 전체를 `CONFIRMATION_TOKEN`에 붙이고
   `EXECUTE_RESET=True`로 바꾼다.
8. writer와 실행 노트북이 모두 중지됐는지 확인한 뒤 커널을 재시작하고
   위에서 아래까지 다시 실행한다.
9. DB 재조회 결과와
   `runs/database_cleanup/quarantine/<operation>/payload/` 및 감사 JSON을
   확인한다.

complete reset의 로컬 파일은 영구 삭제되지 않고 격리된다. 그러나 DB row
snapshot은 만들지 않으므로 삭제된 vector row는 감사 JSON만으로 복구할 수
없다. 필요하면 reset 전에 PostgreSQL backup 또는 해당 run의 재생성 가능성을
확인한다.

다음 항목은 override로도 자동 선택하지 않는다.

- 공유 `images`
- `data/raw`
- 공유 `data/interim`의 common aligned crop와 dataset manifest
- checkpoint/model registry
- 다른 run
- `runs/database_cleanup`의 감사·격리 기록
- exact-owner manifest가 없는 임의 Step 2 경로

특정 DB 테이블만 전문가가 정리해야 할 때는
`RESET_MODE="advanced_database_cleanup"`을 선택한다. 이 호환 모드에서만
`ADVANCED_TABLE_GROUPS_SELECTED`, `ADVANCED_TABLE_NAMES_SELECTED`,
`ALLOW_COMPLETED_RUN_RESET`, `EXECUTE_RESET`을 사용하며 임의 SQL, 임의
테이블, 조건 없는 전체 테이블 삭제와 `images` 삭제는 여전히 지원하지 않는다.
