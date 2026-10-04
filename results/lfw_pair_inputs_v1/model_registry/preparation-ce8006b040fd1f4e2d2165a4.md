# LFW input preparation

Summary: `preparation-ce8006b040fd1f4e2d2165a4.json`

# LFW 1:1 검증: 실행과 결과 해석

## 연구 질문과 범위

UCFace 본문 §IV-A의 LFW open-set verification, 보충자료 §II-A의 동일인 3,000쌍 +
다른 사람 3,000쌍을 참고한다. closed-set identification이나 BLUFR 1:N 실험이 아니다.
UCFace의 학습·checkpoint·전처리·미공개 threshold 선택 구현을 그대로 재현했다는 뜻도 아니다.
압축 및 Continuous FIQA 후보정은 이번 연구에서 추가한 평가 조건이다.
SurvFace 1:N과 기존 BLUFR 완료 결과는 이 변경으로 수정하지 않는다.

## 실행

1. `configs/experiments/lfw_pair_verification.yaml`에서 원본 LFW-deepfunneled 폴더와
   ArcFace/AdaFace/MagFace/EdgeFace, CR-FIQA L checkpoint 경로를 확인한다.
   **과거 face_manifest, BLUFR MAT, 모델 registry, 완료 embedding/FIQA는 필요 없다.**
   원본 이미지와 사전학습 checkpoint는 사용자가 준비한다. 누락된 pairs.txt만 SHA를 검증하여 다운로드한다.
2. `notebooks/calibration/03_origin_vs_pq_fiqa_calibration.ipynb`만으로 처음부터 시작할 수 있다.
   `LFW_PROTOCOL_MODE="pair_verification"`, `EXECUTE=False`로 원본·가중치·CUDA와 계획을 확인한다.
   입력이 아직 없으면 `ready=False`와 준비 안내가 나오며 예전 입력을 찾도록 요구하지 않는다.
3. `EXECUTE=True`로 실행하면 원본 목록 생성 → 모델 등록 → 전체 resize → FR 4모델/FIQA 추출 →
   pair 역할 분리 → development PQ 학습 → 보정 → test → 요약·해석 안내까지 순서대로 실행한다.
   입력 추출은 CUDA, PQ/보정은 CPU다. GPU가 없으면 CPU로 자동 전환하지 않는다.
   `MAX_NEW_JOBS_PER_RUN`은 보정 job 수만 제한하며 초기 전체 입력 추출은 제한하지 않는다.
4. 입력 단계를 따로 실행하려면 LFW 준비 00 또는 공통 batch 00의 `EXECUTE=True`를 사용한다.
   두 노트북은 같은 `prepare_pair_inputs`를 사용한다. 공통 calibration 01의
   `LFW_PAIR_PREPARE=True`는 입력 추출과 PQ/점수 준비까지 하고 보정은 하지 않는다.
   이 세 노트북은 선택 사항이며 반드시 모두 거칠 필요는 없다.
5. 완료 `report_dir`를 common/reports 또는 compact 노트북의 `LFW_PROTOCOL_REPORT_DIR`에 지정한다.
   자동 latest 선택 없이 hash를 검증하여 읽는다. pair report 자체에 이미 집계 ZIP이 있다.
   cross-dataset transfer는 별도 선택 실험이며 새 LFW 1:1→SurvFace 1:N threshold 전이는 허용하지 않는다.

모든 새 경로는 같은 YAML을 읽는다. 모델/fold/seed/output 설정의 `None`은 YAML 값을 따른다.
입력 목록은 `data/interim/lfw/pair_verification_v1/population.csv`, resize는 같은 디렉터리의
`whole_resize_112/`, 등록 정보·임베딩·FIQA는 `results/lfw_pair_inputs_v1/`에 생성된다.
기존 BLUFR/resize source를 덮어쓰지 않는다. 같은 새 경로의 완료 입력은 hash·설정 검증 후 재사용한다.
원본 내용, checkpoint, 전처리, 추출 batch/device 계약이 바뀌면 새 명시적 입력 경로를 사용한다.

CLI도 전체 경로를 사용한다:

```powershell
py -3.11 -m research.experiments.lfw_pair_verification --download-pairs
py -3.11 -m research.experiments.lfw_pair_verification --execute --download-pairs --max-new-jobs 1 --no-keep-raw-results
```

`--output-root D:/...`는 평가 출력만 변경한다. 입력 산출물 위치는 YAML의 `image_manifest`,
`resize_inputs`, `inputs.registry_root`, `source_runs`, `fiqa_root`를 함께 지정한다.
입력 준비는 최소 4 GiB 여유 공간을 확인한다. 기본 배치는 32, CUDA 장치명을 기록한다.
Python 모듈 변경 후 **Kernel Restart → Run All**한다. 모델 파인튜닝은 수행하지 않는다.

## 추가 checkpoint / fine-tuned baseline 등록

기본 YAML의 네 pretrained 조건은 유지한다. 같은 ArcFace 계열이라도 다른 가중치는
`arcface_ft_<이름>` 같은 **별칭과 별도 checkpoint SHA/UID**로 추가한다.
단순히 기존 `arcface`의 checkpoint 파일만 바꾸면 원래 baseline을 잃으므로 거절한다.

`research.experiments.lfw_baselines`의 등록 도구가 새 YAML을 만들며 원래 YAML은 수정하지 않는다.
다음은 **파일과 출처가 확보된 뒤 실제 값으로 바꾸어 실행할 예시**다. 지금 가중치가 있다는 뜻이 아니다.

```powershell
py -3.11 -m research.experiments.lfw_baselines --alias arcface_ft_external --kind fine_tuned --profile arcface_ms1mv3_r100 --checkpoint D:/weights/actual_finetuned_backbone.pth --source-url https://author.example/checkpoint --training-dataset ms1mv3-plus-external-train --parent-baseline arcface --fine-tuning-data external-training-split --lfw-identity-overlap unknown --overlap-evidence unverified --no-evaluation-used-for-training-or-selection --selection-evidence author-documented-training-and-validation-only --output-config configs/experiments/lfw_pair_arcface_ft.yaml --step4-output-config configs/experiments/step4_arcface_ft.yaml
```

- 등록은 학습/다운로드를 하지 않는다. checkpoint SHA를 고정하고 기존 조건을 보존한 새 설정·출력 경로를 만든다.
- fine-tuning 데이터, parent baseline, 출처, 인물 overlap 상태/근거와 평가 데이터를 학습·checkpoint 선택에 쓰지 않았다는 선언을 요구한다.
  확인된 LFW 평가 인물 중복 또는 평가 데이터 사용은 이 1:1 실험에서 거절한다.
  `unknown`은 숨기지 않고 요약에 남기며 `disjoint` 선언만으로 미관측 인물 성능이 입증되지는 않는다.
- 기존 공식 loader와 호환되는 **완전한 512D backbone checkpoint**를 대상으로 한다. classifier만 든 파일,
  별도 LoRA adapter 또는 입력별 조건부 모듈이 필요한 모델은 추가 loader 구현/실제 추론 검증 전에는 사용할 수 없다.
  파일 등록 성공은 추론 호환성이나 성능 개선의 증명이 아니다. 실행 시 엄격한 state-dict 로드와 CUDA 추론을 거친다.
- 모든 baseline은 같은 이미지·공식 pair·development/fit/safety/test·FIQA를 사용한다.
  새 가중치마다 임베딩, PQ codebook, pair score와 보정함수를 새로 만든다.
  원본/PQ × Global-safe/FIQA 5-bin/Continuous FIQA는 각 baseline 안에서 유지한다.
- 생성한 YAML을 관련 노트북의 `LFW_PAIR_CONFIG_PATH`로 지정한다. 모델/fold/seed를 `None`으로 두면
  원래 네 모델과 추가 baseline을 포함한 전체 행렬을 읽는다. fine-tuned 조건을 선택할 때 parent도 함께 선택해야 한다.
- baseline 숫자는 그 checkpoint를 현재 전처리·프로토콜로 평가해 얻는다. 다른 논문의 수치를 실행 baseline에 대입하지 않는다.

보고서의 `baseline_catalog`는 별칭·UID·SHA·학습 출처·parent·overlap 선언을 보존한다.
`backbone_contrasts`는 같은 fold/seed/표현/방법/목표에서 새 checkpoint−parent의 TAR 차이와
**양쪽 실제 FMR**를 함께 기록한다. `fiqa_gain_difference`는 두 checkpoint에서 측정한
`FIQA TAR − Global TAR`의 차이로, 모델 교체와 FIQA 추가 효과를 분리하는 기술 통계다.
같은 목표가 같은 실제 FMR를 뜻하지 않으며, 이 표에는 checkpoint 간 paired CI나 인과효과 보장을 붙이지 않는다.
표의 seed min/median/max는 공유 test에 대한 기술 요약이다. 동일 backbone 내 paired CI는 기존 `paired` 표에서 읽는다.

### SurvFace 1:N에서 같은 checkpoint를 사용할 때

`--step4-output-config`는 같은 모델 metadata/SHA/전처리·fine-tuning 출처를 1:N용 모델 설정으로 내보낸다.
LFW pair, codebook, threshold를 내보내지는 않는다. SurvFace의 개발/보정/test 계약으로 별도 source run을 생성해야 한다.

1. 공통 batch 00에서 `LFW_PROTOCOL_MODE="legacy"`, `DATASET_IDS=("survface",)`를 명시한다.
   `STEP4_MODEL_CONFIG_PATH`에 export한 YAML, `MODEL_PROFILE_BY_NAME`의 해당 family에 새 별칭,
   `MODEL_WEIGHT_PATHS`에 그 checkpoint를 지정한다. 같은 설정이 등록·smoke test·source plan까지 전달된다.
2. 생성한 **명시적 완료 source run**을 공통 보정 01/평가 03의 legacy run matrix에 별도 별칭으로 넣는다.
   `EXPECTED_MODEL_UIDS={별칭: 실제 model UID, ...}`와 FR_MODELS/MODELS를 같은 행렬로 설정한다.
   기존 default UID 검사를 건너뛰는 방식이 아니라 선언한 UID와 source/freeze/codec/score lineage를 대조한다.
3. 해당 SurvFace 임베딩으로 PQ를 새로 학습하고 보정 점수를 재생성한 뒤 threshold를 재적합한다.
   LFW FMR/TAR와 별도로 TPIR@20·실제 FPIR·순위/threshold 실패를 보고한다.

이 연결의 모델 정보 전달 및 UID/codec 검사는 테스트한다. 선택한 fine-tuned checkpoint가 없으면
실제 SurvFace source 생성과 성능은 미검증이다. SurvFace 학습/평가 인물 중복도 별도 확인해야 한다.

## 역할 분리와 비교 행렬

- 고정 mirror의 View-2 pairs.txt SHA-256은 YAML과 코드에서 지정한 출처와 함께 manifest에 남는다.
  매번 10×(300 genuine+300 impostor), 이미지 대응, label, fold identity 비중첩을 검증한다.
  mirror는 FaceNet 저장소이며 UCFace 저자가 배포한 파일이라는 주장을 하지 않는다.
- 공식 6,000쌍에 등장하지 않는 identity의 모든 이미지가 development다. 현재 입력은 1,468명·1,549장이다.
  개발·보정·평가 identity를 분리한다. 새 population의 `split`은 전부 `population`이며 실제 역할은 공식 pair identity에서만 결정한다.
- 모델별 PQ 3종을 development에서 한 번 학습한다. 모든 fold/seed에서 같은 codebook을 사용한다.
  1,549장은 8-bit PQ의 하한 256장보다 많지만 Faiss 권장 9,984장보다 적다. 과적합 위험을 숨기지 않는다.
- test는 매번 공식 한 fold 600쌍 그대로다. 보정 후보는 나머지 9개 fold 5,400쌍이다.
- 보정 identity를 hash로 fit/safety에 할당한다. 양쪽 얼굴이 같은 partition에 속한 pair만 사용한다.
  양쪽이 다른 partition에 속하는 impostor pair는 보정에서 제외하고 수를 inventory에 기록한다.
  test pair는 제외하지 않는다. Global-safe, FIQA 5-bin, Continuous FIQA에 동일 보정 pair를 사용한다.
- query는 공식 pair의 **왼쪽 원본**이며 reference는 **오른쪽 원본 또는 PQ code**다.
  품질은 왼쪽 query의 CR-FIQA L이다. 방향을 결과에 따라 바꾸거나 양쪽 점수 평균으로 대체하지 않는다.
- 4모델×10 folds×20 seeds=800 jobs. 각 job은 Origin+PQ m128/m64/m32×3방법×5 FMR 목표다.
  기존 1:N의 7개 development 비율은 이 프로토콜에서 폐기하며 조용히 일부 비율만 고르지 않는다.
- raw cosine과 negative squared-L2 ADC는 각자 보정한다. ADC는 PQ lookup-table 합이며 decoded cosine이 아니다.
  Origin 512D float32 payload=2,048 bytes, PQ payload=128/64/32 bytes; 각 codebook은 524,288 bytes다.
  이 수치에 DB row/index 비용·FIQA·원본 fallback 비용은 포함되지 않는다. DB latency 실험은 아니다.

## 결과 파일과 지표

`START_HERE.md` → `manifest.json` → `operating_<model>_<profile>.csv` → `paired_<model>_<profile>.csv` 순서로 읽는다.
`analysis.zip`은 다음 **집계 표**만 담는다. manifest는 각 멤버의 행 수/hash/bytes/토큰 추정치를 기록한다.
전체가 크면 ZIP을 풀어 모델/프로파일/표 단위로 올린다. 진단은 fitted target FMR까지 분할하여
한 CSV에 특정 모델·프로파일·적합 목표의 10 folds를 함께 담는다. 결과가 좋은 fold만 선택하지 않는다.
기본 토큰 추정은 UTF-8 bytes/4, 보수적 상한은 bytes다.

- **operating**: calibration에서 고정한 보정함수의 test `realized_fmr`, `tar`, `fnmr`,
  `operating_accuracy`, 오류·성공 수와 분모, `target_met_seed_count`, CI 끝점의 seed별 min/median/max.
  FMR=false accepts/impostor pairs, TAR=true accepts/genuine pairs, FNMR=1−TAR. 값은 0~1 비율이다.
  목표 충족은 반올림 전 `realized_fmr <= target_fmr`로 계산한다. TPIR/FPIR이나 rank failure는 사용하지 않는다.
- **accuracy_cv**: 각 fold의 나머지 9개 fold 전체에서 Accuracy를 최대화한 **global threshold**를 선택한
  별도 표다. pair 수가 같으므로 10 folds의 accuracy 평균은 6,000쌍 pooled accuracy와 같다.
  20개 seed에 동일한 이 표는 중복 제거한다. Continuous FIQA의 accuracy 최적화를 뜻하지 않는다.
  FMR 목표에 맞춘 `operating_accuracy`와 이 accuracy를 혼동하지 않는다.
- **accuracy_summary**: seed 중복을 제거한 fold Accuracy의 평균·표본 표준편차와 pooled counts다.
  `all_ten_folds_complete=False`이면 부분 fold 결과이며 정식 10-fold Accuracy로 제시하지 않는다.
- **diagnostic**: 한 목표에서 학습한 `score − threshold(FIQA)`를 고정한 뒤 test curve를 기술한다.
  `requested_fmr` 이하의 가능한 operating point를 사용하고 `achieved_fmr`와 정수 counts를 기록한다.
  동점을 분리하지 않으며 선형 보간하지 않는다. 이는 운영용 threshold 선택이나 calibration 성공의 증거가 아니다.
  fitted_target_fmr가 다른 함수의 점들을 하나의 ROC로 이어 붙이지 않는다.
- **paired**: 동일 test pair에서 방법−Global 및 PQ−Origin의 TAR 차이와 두 실제 FMR를 함께 표시한다.
  같은 목표 FMR가 실제 FMR 일치를 뜻하지 않는다. 각 seed의 genuine identity bootstrap CI는
  2,000회이며 seed 간 CI 끝점 범위는 통합 CI가 아니다.
- **baseline_catalog / backbone_contrasts**: 위 추가 baseline 등록 절의 출처와 checkpoint 비교 표다.
- **inventory**: fold/seed별 fit/safety genuine·impostor·identity 수, 제외 pair 수, assignment hash.

## 불확실성과 해석 한계

TAR/paired TAR CI는 genuine identity 단위 재표집이다. FMR의 Wilson 구간은 **pair 독립을 가정한 명목 구간**이며
동일 인물이 여러 impostor pair에 등장하는 의존성을 해결한 구간이 아니다. 모델 재학습/threshold 적합 불확실성도
포함하지 않는다. 20 seeds는 같은 test를 공유하고 각 fold의 보정 데이터도 다른 fold와 겹친다.
seed/fold의 표준편차를 독립 반복에 대한 통계적 유의성이나 일반화 보장으로 해석하지 않는다.
한 fold의 impostor는 300쌍이므로 FMR 분해능은 1/300이다. 0.1% 진단은 0 false accept 조건으로 내려갈 수 있다.
전체 3,000 impostor에서도 0.1%는 3회 수준이다. 극저 FMR 성능의 정밀한 추정에 적합하지 않다.

현재 입력은 deepfunneled 전체 이미지를 112×112로 resize한 별도 전처리다. ArcFace 5점 정렬/공식 checkpoint
benchmark와 동일하지 않다. 사전학습 checkpoint와 LFW 인물 overlap은 미검증이다.
LFW 결과만으로 1:N 미등록자 거절, gallery 크기 전이, Top-k, DB 검색 성능을 주장하지 않는다.
테스트 결과로 전처리·프로파일·보정방법을 선택하면 탐색임을 명시하고 별도 검증이 필요하다.

## 보존, 재개, Git

첫 설정 셀의 `KEEP_RAW_RESULTS=False`가 기본이며 Python/CLI에도 동일 옵션이 있다.
입력 준비에서 이 옵션은 완료 FR source와 FIQA를 검증한 뒤 **중복 FR 추출 shard SQLite**를 정리한다.
실패·중단 또는 다른 설정 모델이 미완료이면 공유 shard는 남는다. 기존 완료 추출의 보존 옵션만
False로 바꿔도 과거 shard를 소급 삭제하지 않는다. 실제 보존 여부는 준비 JSON에 기록한다. 전체 population CSV, resized 이미지, 모델 registry, 임베딩, FIQA는
다음 단계와 반복 평가에 필요한 **재사용 입력/cache**로 유지하며 이 옵션의 삭제 대상이 아니다.
따라서 False가 생성 파일 전체 삭제나 0-byte 보관을 의미하지 않는다. 이 입력들을 별도로 제거하면
원본 이미지와 checkpoint에서 GPU 추출부터 다시 해야 한다. per-image 입력/cache도 Git에서는 제외한다.
입력 manifest와 완료 준비 기록에는 hash, CUDA 장치, 보존 정책과 경로가 남는다.

True면 job별 개별 test decision, fitted model, pair score, PQ codebook을 해당 campaign의
`raw/checkpoint.sqlite3`에 남긴다. False면 개별 decision을 쓰지 않고, 재개에 필요한 점수·모델·집계 checkpoint를
실행 중 유지하다가 **전체 완료 + 요약 round-trip/hash 검증 후에만** 그 campaign 파일을 정리한다.
부분 실행/실패 checkpoint는 보존한다. 설정 변경은 다른 campaign UID를 만들며 과거 결과를 소급 삭제하지 않는다.
완료 요약·해석 안내·설정·출처/hash는 양쪽 모드에서 유지된다. 요약만으로 새 cutoff/개별 pair 분석/새 bootstrap을
계산할 수 없으며 해당 경우 보정과 점수 계산을 재실행해야 한다. 입력 이미지/사전학습 모델/FR·FIQA source는 삭제하지 않는다.
출력 전에 가용 공간과 대략적인 예상 저장량을 점검한다. 실제 SQLite bytes도 반환한다.
기본 출력의 `raw/`와 `.staging-*`는 Git에서 제외하고 요약은 포함한다. 다른 출력 위치가 저장소 내부라면
그 raw 경로도 구체적으로 ignore해야 한다. SQLite/ZIP이라는 이유로 상세 원본을 커밋하지 않는다.

`status=partial`과 `completed_jobs/expected_jobs`는 항상 함께 확인한다. `_SUCCESS`는 해당 snapshot의
무결성을 뜻하며 전체 캠페인 완료 여부는 manifest.status로 판별한다.
