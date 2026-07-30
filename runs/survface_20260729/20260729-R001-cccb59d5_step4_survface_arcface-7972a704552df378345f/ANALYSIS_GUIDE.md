# SurvFace Step 4 Grad-CAM 결과 안내

## 실행 범위

- Run ID: `20260729-R001-cccb59d5`
- 상태: `completed`
- 데이터셋: QMUL-SurvFace v1
- 모델: `arcface-7972a704552df378345f`
- 완료 단계: `00_source_and_model_freeze`부터
  `06_representative_case_visualization`까지

이 디렉터리는 완료된 작업 run이다. Git에는 결론 확인과 후속 통계 분석에
필요한 선별 결과만 저장하고, 재생성 가능한 대형 작업 산출물은 로컬 전용으로
유지한다.

## 먼저 확인할 결과

1. `artifacts/step2_workflow/step4_summary.json`
   - 전체 실행 범위와 최종 행 수
2. `artifacts/step2_workflow/saliency_validation.json`
   - Grad-CAM 선택·유효성·semantic mask 범위
3. `artifacts/step2_workflow/saliency_geometry_associations.csv`
   - saliency feature와 압축 기하 오차의 association
4. `artifacts/step2_workflow/saliency_retrieval_associations.csv`
   - saliency feature와 검색 성능 변화의 association
5. `artifacts/step2_workflow/representative_cases.csv`
   - stable, high-error, rank-flip, threshold-crossing 대표 사례
6. `artifacts/step2_workflow/saliency_population/saliency_features.csv`
   - 이미지 단위 saliency feature 재분석용 표

CSV 다섯 개는 Git LFS로 추적한다. `selected_manifest.csv`는 분석 모집단과
원본 표본 계보를 고정한다.

## Git에 포함하지 않는 로컬 전용 파일

- `paired_embedding_metrics.csv`
- `retrieval_metrics.csv`
- `prepared_population/embedding_shards/*.npz`
- `prepared_population/sample_index.csv`
- `saliency_population/heatmap_shards/*.npz`
- `figures/*.png`

앞의 두 CSV는 고급 재분석 입력이지만 각각 약 1.27 GiB와 12.73 GiB이므로
로컬 전용으로 둔다. NPZ shard는 임베딩·heatmap 재생성 입력이다.
대표 사례 PNG는 원 데이터셋 얼굴 픽셀을 포함하므로 저장소에 재배포하지
않고 `representative_cases.csv`만 버전 관리한다.

## 삭제한 중복·파생 산출물

- `saliency_geometry_join.csv`
- `saliency_retrieval_join.csv`

두 파일은 saliency feature와 원본 metric ledger를 단순 결합한 대형 파생
테이블이다. 완료된 A002 phase와 최종 association CSV를 확인한 뒤 삭제했다.
필요하면 로컬 전용 `saliency_features.csv`, `paired_embedding_metrics.csv`,
`retrieval_metrics.csv`에서 다시 생성할 수 있다.

세부 보존 파일의 크기와 SHA-256, 제외·삭제 사유는
`retention_manifest.json`을 따른다.
