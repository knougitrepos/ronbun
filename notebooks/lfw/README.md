# 현재 LFW 기본 경로: 1:1 pair verification (2026-10-04)

UCFace의 LFW 6,000쌍 검증 방향을 반영했습니다. closed-set identification/BLUFR과 구분합니다.
공통 YAML에 원본 이미지·checkpoint를 지정하고, calibration 03의 `pair_verification`에서
`EXECUTE=False`로 점검한 뒤 `True`로 실행합니다. 기본은 4모델×10 folds×20 seeds=800 jobs이며
Origin+PQ 3종×3보정방법×5 FMR 목표를 유지합니다. 기존 7개 development 비율은 이 새 프로토콜에 적용하지 않습니다.
[전용 실행·지표·보존 안내](C:/ronbun/notebooks/calibration/LFW_PAIR_VERIFICATION.md)를 따르세요.

원본 목록·모델 registry·FR/FIQA가 없어도 03에서 자동 생성합니다. 과거 BLUFR MAT/manifest는 필요 없습니다.
새 입력은 `data/interim/lfw/pair_verification_v1/`, `results/lfw_pair_inputs_v1/`에 저장합니다.
`KEEP_RAW_RESULTS=False`는 전체 완료/요약 검증 후 이번 캠페인의 상세 checkpoint를 정리합니다.
부분 실행은 재개용 checkpoint를 남깁니다. 결과는 FMR/TAR이며 SurvFace 1:N TPIR/FPIR와 섞지 않습니다.
LFW 준비 00/공통 batch 00은 선택적 전체 입력 추출, 01은 선택적 FR/FIQA/PQ/점수 준비입니다.
report/compact는 명시한 pair report를 읽습니다. False에서도 재사용 FR/FIQA 입력은 유지합니다.
기존 완료 결과와 아래 legacy/BLUFR 모드는 보존되며 명시적으로 선택할 때만 실행합니다.

---

## 이전 경로 안내 (보관)

**아래의 ‘현재/새 실험’은 당시의 표현이며, 현재 기본값은 위의 1:1 경로입니다.**

# LFW 전체 resize 입력과 BLUFR 분할 기반 보정

현재 새 실험은 준비 00에서 `blufr_resize_inputs`, `EXECUTE=False`로 검사한 후
`EXECUTE=True`, `RESIZE_ONLY=False`, batch32로 FR 4모델·FIQA를 전체 재추출한다.
13,233장을 동일한 전체 이미지 112×112 resize로 처리한다. 완료 후 03의
`configs/experiments/lfw_blufr_calibration_resize.yaml`에서 ready=True를 확인하고 보정을 실행한다.
기존 임베딩/FIQA와 새 입력을 혼합하지 않는다. 아래 목록 준비와 과거 source 설명은 참고용이다.

공개 목록이 없다면 [준비 노트북](C:/ronbun/notebooks/lfw/00_data_preparation/00_data_preparation.ipynb)의
`LFW_PROTOCOL_MODE="blufr_lists"`, `DOWNLOAD_BLUFR_CONFIG=True`로 Kernel Restart → Run All한다.
SHA가 고정된 MAT를 받고 원본 manifest와 대응을 확인한다. 기존 분할 파일은 덮어쓰지 않는다.

목록이 준비되어 있다면 calibration 03의 `blufr_calibration`, `EXECUTE=False`로 직접 점검한다.
이전 detected/aligned 고정 source는 공개 13,233장 중 임베딩/FIQA 38장이 부족했다. 새 resize source는 별도 YAML에서 읽는다.
준비 노트북의 목록 다운로드가 얼굴 검출/임베딩 누락까지 복구하는 것은 아니다.

공개 train/test·gallery/probe를 유지하고 train 내부 7가지 압축 학습:보정 비율을 비교한다.
실행 순서와 설정은 [전체 안내](C:/ronbun/notebooks/README.md)를 따른다.
별도 benchmark는 노트북에서 실행하지 않으며 Python 파일만 보존한다.

원본 face_manifest.csv 자체가 없는 신규 환경에서만 준비 노트북의 `legacy` 모드가 필요하다.
기존 source 생성은 공통 batch의 `legacy` 경로를 사용한다. 수동 단계와 DB 실험은
[보관 안내](C:/ronbun/notebooks/_archive/README.md)에 있다.
