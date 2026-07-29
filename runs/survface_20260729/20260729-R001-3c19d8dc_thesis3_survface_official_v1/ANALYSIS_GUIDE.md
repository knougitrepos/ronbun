# SurvFace 비-Grad-CAM 실험 결과 보존본

## 범위

- 실행 ID: `20260729-R001-3c19d8dc`
- 데이터셋/프로토콜: QMUL-SurvFace v1 공식 open-set identification
- 보존 단계: `00` 프로토콜 동결부터 `05` 공식 평가·시각화까지
- 제외 단계: `04_gradcam`
- 공식 검색 완료 시도: `04_official_probe_search/A002`
- 공식 평가 완료 시도: `05_official_evaluation_and_visualization/A001`

원본 실행의 최상위 `run_manifest.json`은 평가 노트북의
`FINALIZE_RUN=False` 설정 때문에 `running`으로 남아 있다. 그러나 이 보존본에
포함된 비-Grad-CAM 단계의 phase manifest는 모두 `completed`이다. 중단된 검색
시도 `04_official_probe_search/A001`의 임시 CSV는 보존하지 않는다.

## 먼저 확인할 결과

1. `artifacts/05_official_evaluation_and_visualization/official_metrics_A001.csv`
   - 압축 프로파일과 검색 방식별 공식 TPIR/FPIR 결과
2. `artifacts/05_official_evaluation_and_visualization/tpir20_fpir_curves_A001.csv`
   - 임계값별 TPIR@Rank-20/FPIR 곡선
3. `figures/tpir20_fpir_curve_A001.png`
   - 공식 평가 곡선 그림
4. `artifacts/04_official_probe_search/official_search_summary_A002.json`
   - 검색 조합, probe 수, 완료 범위와 검색 산출물 해시
5. `artifacts/03_survface_compressed_materialization_and_index/survface_materialization_summary_A001.json`
   - PCA/PQ materialization과 DB 저장 범위 요약

## 대용량 원시 결과

- `official_top20_search_matrix_A002.csv`
  - 공식 probe별 top-20 검색 결과이며 평가를 다시 계산할 때 사용한다.
- `survface_pca_sweep_measurements_A001.csv`
  - PCA 차원 sweep의 이미지별 압축 오차 측정값이다.
- `survface_primary_compression_measurements_A001.csv`
  - 대표 압축 프로파일의 이미지별 측정값이다.

위 CSV는 Git LFS로 추적한다. 일반 Git blob으로 저장하지 않는다.

## 의도적으로 제외한 파일

- `phases/**` 아래의 artifact 중복 복사본
- 중단된 검색 시도 A001의 `.tmp`
- 전체 이미지별 임베딩 추출 ledger
- 원본/정렬 얼굴 이미지와 데이터셋 파일
- PostgreSQL/pgvector DB 자체

실제 결과 파일의 SHA-256과 크기는 각 단계의 `phase_manifest.json`에 기록되어
있다.
