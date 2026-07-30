# LFW Step 2 결과 분석 가이드

- Run ID: `20260727-R001-3dc29af8`
- 모델: `arcface-7972a704552df378345f`
- 범위: LFW 전체, `mode=real`, fraction 1.0, seed 42
- 상태: completed
- 정렬 성공 표본: 13,195
- Grad-CAM 적격 표본: 9,133
- 유효 heatmap: 9,122

## 먼저 분석할 파일

1. `artifacts/step2_workflow/saliency_validation.json`
   - Grad-CAM 유효성·faithfulness의 작은 요약 파일이다.
   - 일반 텍스트 편집기로 바로 열 수 있다.
2. `artifacts/step2_workflow/saliency_compression_associations.csv`
   - 주 분석 결과다. 900행이며 saliency 특징과 압축 민감도 지표의
     Spearman 상관계수, identity-cluster bootstrap 95% 신뢰구간을 담는다.
   - GitHub 연결에서도 원문을 읽을 수 있도록 일반 Git 텍스트로 추적하며,
     Excel 또는 pandas로 열 수 있다.
3. `artifacts/step2_workflow/representative_cases.csv`
   - 283개 대표 사례의 선택 근거와 지표를 담는다.
   - `artifacts/step2_workflow/figures/`의 PNG와 `case_id`로 연결된다.
   - GitHub 연결에서 표와 그림을 함께 검토할 수 있도록 일반 Git 텍스트로
     추적한다.
4. `run_manifest.json`과
   `artifacts/step2_workflow/freeze_manifest.json`
   - 실행 설정, 모델·입력 checksum, lineage를 확인할 때 사용한다.

## 상세 분석용 파일

| 파일 | 행 수 | 용도 |
|---|---:|---|
| `paired_embedding_metrics.csv` | 131,950 | PCA/PQ별 embedding 왜곡, 각도 오차 |
| `retrieval_metrics.csv` | 43,280 | 1:N 검색 점수·순위·threshold 변화 |
| `saliency_compression_join.csv` | 153,590 | saliency와 압축 결과의 전체 결합표 |
| `saliency_population/saliency_features.csv` | 13,195 | 표본별 Grad-CAM 특징·faithfulness |
| `selected_manifest.csv` | 13,195 | 분석 표본과 split·identity |

위 파일 중 `saliency_compression_join.csv`와 `retrieval_metrics.csv`는
일반 편집기나 Excel로 전체를 여는 용도가 아니다. pandas, Polars,
DuckDB 또는 PostgreSQL로 필요한 열과 조건만 조회한다.

## 직접 분석하지 않아도 되는 파일

- `prepared_population/embedding_shards/*.npz`
  - 원본 embedding과 LOO template의 재현·검증용 binary shard다.
- `saliency_population/heatmap_shards/*.npz`
  - 전체 Grad-CAM heatmap의 재현·재시각화용 binary shard다.
- `logs/events.jsonl`
  - 실행 감사와 오류 추적용 event log다.
- `COMPLETED`
  - 완료 상태 marker다.

NPZ는 손상되거나 해석 불가능한 파일이 아니라 NumPy 압축 배열이다.
일반 텍스트 편집기로 열지 않고 `numpy.load(..., allow_pickle=False)`로 읽는다.

## 현재 해석 시 주의점

- semantic mask가 제공되지 않아 `semantic_masked_sample_count=0`이다.
  따라서 눈·코·입과 같은 semantic region 결론은 아직 내릴 수 없다.
- high-saliency occlusion median drop은 약 `0.0551`, low-saliency는
  `0.0475`, random은 `0.0440`이다. faithfulness 차이는 후속 통계 검정과
  compression profile별 분석이 필요하다.
- 논문의 우선 분석표는 association CSV이며, 대용량 join CSV는 그 결과를
  재검증하거나 새로운 하위집단 분석을 할 때만 사용한다.

## GitHub 공개 방식 변경

- 2026-07-30: 핵심 association CSV와 대표 사례 CSV를 Git LFS 포인터가 아닌
  일반 Git 텍스트로 전환했다.
- 이 변경은 완료된 실험 값이나 행 내용을 수정하지 않고 저장 방식만
  변경한 것이다. 대형 상세 CSV와 NPZ shard는 계속 Git LFS로 유지한다.
