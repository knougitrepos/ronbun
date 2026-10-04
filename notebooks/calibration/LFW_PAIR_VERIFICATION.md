# LFW 1:1 검증: 실행과 결과 해석

## 연구 질문과 범위

UCFace 본문 §IV-A의 LFW open-set verification, 보충자료 §II-A의 동일인 3,000쌍 +
다른 사람 3,000쌍을 참고한다. closed-set identification이나 BLUFR 1:N 실험이 아니다.
UCFace의 학습·checkpoint·전처리·미공개 threshold 선택 구현을 그대로 재현했다는 뜻도 아니다.
압축 및 Continuous FIQA 후보정은 이번 연구에서 추가한 평가 조건이다.
SurvFace 1:N과 기존 BLUFR 완료 결과는 이 변경으로 수정하지 않는다.

## 실행

1. `notebooks/lfw/00_data_preparation/00_data_preparation.ipynb`의 기본 `pair_verification`
   모드에서 고정 SHA의 pairs.txt를 준비한다. `DOWNLOAD_LFW_PAIRS=True`는 누락된 목록만 받는다.
2. 주 실행점은 `notebooks/calibration/03_origin_vs_pq_fiqa_calibration.ipynb`다.
   `LFW_PROTOCOL_MODE="pair_verification"`, `EXECUTE=False`로 coverage와 800-job 계획을 확인한다.
   설정은 `configs/experiments/lfw_pair_verification.yaml`이며 모델·fold·seed는 첫 설정 셀에서 선택한다.
3. `EXECUTE=True`로 실행한다. `MAX_NEW_JOBS_PER_RUN`은 전체 행렬을 유지하면서 새 계산만 제한한다.
   중단 후 같은 설정으로 실행하면 검증된 완료 job을 재사용한다. Python 수정 후에는 커널을 재시작한다.
4. 명시적인 `report_dir`를 common/reports의 `pair_verification_report`에 넣어 다시 읽는다.
   자동 latest 선택을 하지 않는다. batch 00은 점검, batch calibration 01은 선택적 PQ/점수 준비다.

CLI도 동일하다:

```powershell
py -3.11 -m research.experiments.lfw_pair_verification --download-pairs
py -3.11 -m research.experiments.lfw_pair_verification --execute --max-new-jobs 1 --no-keep-raw-results
```

`--output-root D:/...`로 결과 위치를 바꿀 수 있다. source 파일 위치는 YAML의 명시적 경로다.
기존 전체 resize FR/FIQA source를 hash 검증하여 재사용하므로 이 workflow에는 GPU 재추론이 없다.
Faiss PQ 학습/ADC, 보정은 CPU다. 학습/보정/test 점수는 새로 계산하고 기존 1:N threshold를 전용하지 않는다.

## 역할 분리와 비교 행렬

- 고정 mirror의 View-2 pairs.txt SHA-256은 YAML과 코드에서 지정한 출처와 함께 manifest에 남는다.
  매번 10×(300 genuine+300 impostor), 이미지 대응, label, fold identity 비중첩을 검증한다.
  mirror는 FaceNet 저장소이며 UCFace 저자가 배포한 파일이라는 주장을 하지 않는다.
- 공식 6,000쌍에 등장하지 않는 identity의 모든 이미지가 development다. 현재 입력은 1,468명·1,549장이다.
  개발·보정·평가 identity를 분리한다. 기존 face_manifest의 과거 `split` 열은 이번 역할 결정에 사용하지 않는다.
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
