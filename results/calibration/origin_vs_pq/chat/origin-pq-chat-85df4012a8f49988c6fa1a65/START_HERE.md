# 원본/PQ FIQA 분석 자료

실험 상태: **partial** — 예정 240개 중 20개 job 완료.
누락 job 수: 220. 완료된 조건만 집계했으며 미완료 결과를 0으로 채우지 않았습니다.

ChatGPT에 `START_HERE.md`와 `analysis.zip`을 함께 첨부하십시오. ZIP 처리가 지원되지 않으면
압축을 풀고 `INDEX.csv`에서 원하는 데이터셋/FR 모델의 CSV를 골라 첨부하십시오.
각 CSV는 기본 125행 이하, 약 64 KiB 기준으로 나눕니다.
하나의 긴 행은 잘라 버리지 않습니다. 모든 관측된 target/method/profile 조건과 모든 페이지를 보존합니다.

- `operating_performance`: FPIR/TPIR, 오류·성공 수와 분모, 목표 충족 seed 수/비율.
- `operating_failures`: Top-K 밖 순위 실패와 Top-K 안 threshold 실패, rank 상한.
- `operating_ci_endpoints`: 개별 seed CI의 하한/상한이 seed에 따라 어떻게 달라지는지.
- `paired_effects`: 같은 cohort의 candidate-minus-reference 비교와 각 실제 FPIR.
- `operating_interactions`: (PQ FIQA 이득) − (원본 FIQA 이득), 네 조건의 실제 FPIR.
- `diagnostic_curves`, `diagnostic_interactions`: 고정 보정 모델의 test 곡선 보간 진단.
- `partition_inventory`: calibration fit/safety 인원·query 수의 seed별 범위.
- `coverage.json`: 전체 job 완료 범위. `provenance.json`: 입력·설정·원본 checkpoint 출처.

`min/median/max`는 같은 test cohort를 공유하는 calibration seed들의 기술 통계입니다.
seed별 분모·오류 수를 합치지 마십시오. 개별 seed CI endpoint의 범위는 통합 CI가 아닙니다.
목표 충족은 반올림 전 FPIR로 판정합니다. 같은 목표 FPIR은 같은 실제 FPIR을 뜻하지 않습니다.
TPIR@K는 genuine identity의 Top-K 포함과 genuine score의 threshold 통과를 모두 요구합니다.
진단 곡선은 test 보간이며 deployment threshold를 선택하지 않고 CI·수학적 FPIR 보장을 제공하지 않습니다.
원본 cosine과 PQ ADC는 서로 다른 score space입니다. checkpoint 학습 데이터 overlap은 별도 확인이 필요합니다.

이 묶음은 분석용 요약이며 seed별 모델 JSON·전체 seed 결과를 중복 저장하지 않습니다.
그 원자료는 provenance에 연결된 checkpoint/보고서에 남고 source table SHA-256으로 연결됩니다.
해석 요청 예: “모든 target/method/profile을 확인하고, 목표 FPIR 충족과 실제 FPIR을 함께 비교하며,
원본 대비 PQ의 추가 FIQA 이득 및 순위/threshold 실패를 구분해 분석해 주세요.”
