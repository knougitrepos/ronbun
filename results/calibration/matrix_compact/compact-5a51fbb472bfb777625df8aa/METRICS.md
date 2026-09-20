# 읽는 방법과 지표

한 번에 START_HERE.md 또는 CSV 한 페이지만 요청하세요. 전체 폴더/ZIP/원본 CSV를 한꺼번에 채팅으로 읽지 마세요.
CSV의 모든 비율·차이는 0–1 단위이며 0.1은 10%입니다. 대상 FPIR은 설정값이고 실제 FPIR은 측정된 오수락률입니다.

- split_count / seeds: 같은 calibration 집합을 나눈 seed 수. 독립 실험 반복 횟수가 아닙니다.
- target_met_split_count: 실제 FPIR가 목표 이하인 seed 수. 미래 FPIR 보장이 아닙니다.
- fpir_min/median/max, tpir_min/median/max: seed별 관측 범위/중앙값. TPIR은 genuine-score 기준 Rank-20입니다.
- non_mated/mated: test 분모. false_accept / true_identification 최소·최대는 seed별 성공 수입니다.
- CI low_min/high_max: 기존 seed별 95% CI 끝점들의 범위입니다. 20-seed 통합 CI가 아니며 CI를 새로 계산하지 않았습니다.
- paired delta: candidate − reference. TPIR 양수는 증가, FPIR 양수는 오수락 증가입니다.
- positive_ci_seeds / negative_ci_seeds: 기존 paired CI가 각각 0보다 완전히 큰/작은 seed 수입니다. seed 간 검정이나 다중 비교 보정이 아닙니다.
- both_target_met_seeds: 두 방법 모두 목표 FPIR를 충족한 seed 수. 두 방법의 실제 FPIR가 같다는 의미는 아닙니다.
- both_met_positive_ci_seeds: 위 조건을 만족하면서 paired CI가 양수인 seed 수. FPIR metric에서 양수는 이득이 아닙니다.
- saliency paired reference의 baseline은 continuous_fiqa입니다. FIQA 방법 간 비교도 모두 보존했습니다.
- faithfulness_status=failed는 진단 실패를 그대로 표시한 것입니다. threshold 성능 개선이나 인과성을 입증하지 않습니다.
- fallback_*는 무효 saliency에서 FIQA-only로 처리한 query의 최대 수/비율입니다. FIQA-only 행의 빈 칸은 해당 없음입니다.

단일 seed 8972 보고서와 20-seed 보고서는 중복 seed가 있으므로 서로 합쳐 평균내지 않습니다.
데이터셋/모델/PQ를 pooling한 FPIR·TPIR를 만들지 않았습니다. 전체 현황의 condition-seed 수는 기술적 집계입니다.
이 compact는 모든 조건·방법·목표·seed를 집계하지만 개별 seed의 수치, fitted 계수, query 기록을 무손실 복제하지는 않습니다.
정확한 개별 수치·추가 분석은 원본 matrix reports 또는 기존 analysis_archive ZIP에서 확인하세요. 원본은 수정하지 않습니다.
파일 크기는 참고값이며 크기를 맞추기 위해 내용을 삭제하지 않았습니다. 조건별 표를 함께 유지하고 행 수가 많을 때만 페이지를 나눕니다.
연결 도구의 추가 출력/기존 대화 길이에 따른 토큰 한도까지 보장하지는 않습니다. 여러 파일을 한꺼번에 출력하지 마세요.
