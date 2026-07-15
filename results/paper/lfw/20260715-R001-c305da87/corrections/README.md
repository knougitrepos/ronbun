# R001 certified metric correction

기존 `evaluation_metrics.json`의 `certified_accept_correctness`와
`certified_reject_correctness`는 ground truth 정확도가 아니라 origin exact 결정과의
일치도입니다.

인증 결정의 실제 ground-truth 평가는
`certified_ground_truth_v2_826f890203fb/`의 JSON과 CSV를 사용합니다. 기존 run과
04 검색 artifact는 불변 상태로 보존했으며, 검색·latency 등 이 오류와 무관한 기존
지표는 변경하지 않았습니다.
