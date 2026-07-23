# LFW 원본 공간 특징·압축 민감도 분석

이 폴더는 모든 선택 이미지의 원본 임베딩과 원본 모델 Grad-CAM 특징을 먼저
추출하고, 그 뒤 동일한 512D 임베딩의 PCA-only/PQ-only 변화와 결합하는 Step 2
runbook입니다. 압축 오류 사례를 먼저 고른 뒤 일부 이미지만 설명하는 이전 흐름은
폐기했습니다.

실행 순서:

1. `00_source_and_model_freeze.ipynb`
2. `01_origin_embedding_and_loo_templates.ipynb`
3. `02_population_gradcam_extraction.ipynb`
4. `03_saliency_feature_validation.ipynb`
5. `04_step2_compression_characterization.ipynb`
6. `05_saliency_compression_join.ipynb`
7. `06_representative_case_visualization.ipynb`

핵심 경계:

- Pass A는 선택된 모든 이미지의 raw 512D, raw norm, unit embedding을 만든다.
- Grad-CAM target은 같은 split·identity의 다른 이미지 평균인
  `origin_leave_one_out_identity_cosine`이다.
- LOO target이 없는 singleton 또는 identity 미공개 표본도 행과 임베딩은
  보존하고 `saliency_target_eligible=False`와 사유를 기록한다.
- Pass B는 모든 eligible 이미지에 수행하며 Pass A/B embedding 일치를 검사한다.
- Grad-CAM 특징은 임베딩에 붙이지 않고 표 수준에서 압축 민감도와 결합한다.
- PCA는 원본 512D에서 차원별로 독립 적용하고 PQ도 원본 512D에 직접 적용한다.
  PCA 결과에 PQ를 연쇄 적용하지 않는다.
- 원본 fallback은 사용하지 않는다.
- 결합 키는 `extraction_uid + dataset_id + sample_id + model_uid`이고
  `origin_embedding_artifact_uid`도 반드시 같아야 한다.
- 사례 선택은 전체 결합 분석 뒤 그림을 만드는 마지막 단계에서만 수행한다.
- 전체 activation·gradient는 영구 저장하지 않는다. native CAM과 수치 특징은
  immutable shard로 저장하고 중간 tensor는 명시적 debug subset에만 허용한다.
- 랜드마크/face mask가 없으면 눈·볼·턱·얼굴 외부 값을 임의로 추정하지 않는다.

각 노트북은 새 커널에서 위에서 아래로 실행합니다. 모든 기본값은
`EXECUTE_STAGE=False`, `WRITE_OUTPUTS=False`이며 입력 hash나 범위가 바뀌면 기존
artifact를 덮어쓰지 않고 새 lineage를 만듭니다.

현재 실제 실행 blocker는 공통 정렬 crop bundle, 세 모델의 검증된 checkpoint와
target layer, 전용 PyTorch Step 2 PCA/PQ 정량 runner입니다. 코드와 노트북이
존재한다는 사실만으로 실제 모델 결과가 생성됐다고 간주하지 않습니다.
