# Calibration matrix 분석 자료와 Git 보관 규칙

작성일: 2026-09-20 (KST)

## 분석 진입점

- `reports/matrix-report-2e2ab02ddb1d039ef1084383`: 36조건 × 두 방법군, 단일 seed 8972의 완료 보고서(72작업).
- `reports/matrix-report-d690567c6f4f7e7ad44ef19e`: 36조건 × 두 방법군 × 20개 seed의 완료 보고서(1,440작업).
- 각 보고서의 `method_summary.csv`는 실제 FPIR·TPIR와 목표 충족 여부, `paired_comparisons.csv`는 기준 방법 대비 paired 비교, `split_summary.csv`는 분할별 변동 요약이다. 개별 seed 행도 통합 CSV에 남아 있다. `manifest.json`에 원본 결과 경로와 해시가 기록된다.
- 완료 여부는 파일·manifest 기준이다. FIQA+saliency의 효과나 목표 FPIR 충족을 의미하지 않으며 실제 지표와 진단을 확인해야 한다.

## 상세 증거 보관본

`analysis_archive/matrix-evidence-7c6031beee4a3d56a3a2e163.zip`은 위 보고서가 참조한 1,440개 고유 결과와 재사용 receipt를 포함한다. 작업별 fitted `models.csv`, 진단·fallback·세부 비교 CSV, manifest 및 receipt 등 총 12,188개 파일을 원래 `jobs/...` 상대 경로로 보존했다. 개별 파일 SHA-256은 ZIP 안의 `inventory.json`, ZIP 자체 SHA-256과 보고서 연결은 동명의 외부 JSON에 있다. 원래 producer/입력 해시를 변경하지 않았다.

기존 결과 분석은 통합 CSV에서 시작하고, 보정 계수나 개별 query fallback 등의 추가 분석에 ZIP을 사용한다. 기존 로컬 작업 폴더가 있다면 그대로 읽으면 된다. 새 checkout에서 확장이 필요하면 먼저 별도 디렉터리에 ZIP을 풀고 해시를 확인한다. 저장된 절대 경로는 원래 실행 환경의 출처 기록이므로 다른 PC에서 그대로 실행 경로로 사용할 수 없다.

## 로컬에만 유지하는 파일

- `jobs/`: 재시작용 확장 체크포인트. 상세 증거가 ZIP에 보존되므로 수천 개 파일을 개별 Git 항목으로 추가하지 않는다.
- `../saliency_bindings/**/saliency_features.csv`, `faithfulness_rows.csv`: 원래 saliency 입력을 PQ 조건에 연결하며 복사한 대용량 행 자료. 분석에 필요하면 로컬 파일을 사용하거나 동일 설정의 원본 입력에서 재생성한다. 원본 입력·연결 manifest 및 요약은 Git에 남긴다.
- Calibration의 `.staging-*` 및 임시 파일: 미완료 또는 이전 저장 실패 상태다. 최종 결과로 해석하지 않는다.

Ignore는 삭제가 아니다. 로컬 데이터와 노트북의 checkpoint 재사용은 유지된다. Git 저장소만으로 전체 이미지·임베딩·saliency 입력을 복원할 수 있는 것은 아니므로 기존 로컬 데이터 보관은 필요하다.

## 다음 실행 결과를 보관할 때

저장소 루트에서 `py -3.11 scripts/package_calibration_analysis.py`를 실행한다. 모든 완료 보고서와 참조 결과·파일 해시를 검사하고, 새 결과 집합이면 새 UID의 ZIP/JSON을 작성한다. 기존 보관본은 덮어쓰지 않는다. 미완료 보고서나 보고서에 포함되지 않은 작업 파일이 있으면 오류로 중단한다. 성공 후 `reports/`, `analysis_archive/`, 필요한 입력 manifest와 출처 기록을 Git에 추가한다. 아직 완료되지 않은 실행의 `jobs/`는 Git에 보관되지 않으므로 로컬 백업을 유지한다.

2026-09-20 정리에서는 실험 파일을 삭제하거나 재계산하지 않고 Git의 추가 대상만 조정했다.
