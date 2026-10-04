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

# 공통 실행·보고·관리

- orchestration/: source run 생성/재사용, 전체 PQ 보정, PQ 결과 집계, 별도 전이 실험.
- reports/: 여러 완료 run의 보고. 실행기가 notebook을 직접 호출하므로 경로를 유지한다.
- maintenance/: 특정 DB/run 폐기·격리. 일반 실험 실행 순서에 포함하지 않는다.

현재 실행 메뉴는 [노트북 안내](C:/ronbun/notebooks/README.md)를 따른다.
독립 checkpoint 등록·smoke 노트북은 _archive/common/model_preparation/에 보관한다.
일반 source 생성에서는 공통 runner가 모델 준비 함수를 호출한다.
