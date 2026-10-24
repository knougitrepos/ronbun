# 이전 노트북 보관

현재 실행은 [상위 폴더의 01 → 02 → 03 → 04](../README.md)를 사용하세요.
이 폴더의 56개 노트북은 이전 실행 방식과 과거 실험 재현용입니다.

- `pipeline/`: 직전 정리의 다중 프로토콜 진입점 4개.
- `common/`, `calibration/`, `diagnostics/`, `survface/`, `rfw/`: 선택 실험·보조 평가·관리 8개.
- `_archive/`: 이전부터 보관하던 수동 단계 40개.
- `lfw_sequential_v2/`: 직전 LFW 전용 순차 노트북 4개.
- [기존 workflow 상세](LEGACY_WORKFLOWS.md): 과거 설정과 실행 계약.

`scripts/run_integrated_postprocessing.py`의 기존 1:N 자동 보고는
`pipeline/04_results.ipynb`를 호출합니다. 현재 상위 폴더의 04는 5개 데이터셋 workflow의 결과를 읽습니다.
보관 노트북의 저장 출력은 과거 기록이며 현재 실행 완료의 근거가 아닙니다.
