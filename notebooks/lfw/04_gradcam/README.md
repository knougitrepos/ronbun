# LFW Step 2 Grad-CAM 실행 순서

Grad-CAM은 원본 embedding 공간의 population saliency를 먼저 만들고, 같은
embedding lineage의 PCA-only/PQ-only 민감도와 결합한다.

## Prerequisite

1. `prerequisite/00_source_and_model_freeze.ipynb`
2. `prerequisite/01_origin_embedding_and_loo_templates.ipynb`

첫 단계는 앞서 생성한 aligned-crop·106-point landmark bundle, checkpoint,
preprocessing과 선택 population을 고정한다. 두 번째 단계는 모든 선택 표본의
원본 512D embedding과 leave-one-out identity template을 만든다. singleton 등
target 부적격 표본은 삭제하지 않고 사유를 기록한다.

## Experiment

1. `experiment/00_population_gradcam_extraction.ipynb`
2. `experiment/01_saliency_feature_validation.ipynb`
3. `experiment/02_step2_compression_characterization.ipynb`
4. `experiment/03_saliency_compression_join.ipynb`
5. `experiment/04_representative_case_visualization.ipynb`

압축 metric CSV는 같은 PyTorch 원본 embedding lineage에서 생성되어야 한다.
ONNX Step 1 결과를 checkpoint parity 확인 없이 섞지 않는다. 결합 키는
`extraction_uid + dataset_id + sample_id + model_uid`이며
`origin_embedding_artifact_uid`가 일치해야 한다.

표 형식 artifact는 CSV로 기록한다. embedding/heatmap shard는 NPY/NPZ를 유지하되
CSV index와 JSON manifest, checksum을 동반한다. 새 run은
`runs/lfw_YYYYMMDD/` 아래에 생성되며, 그 날짜 root의 `active_run.json`이
가리키는 미완료 RunStore run에는 canonical 결과 한 세트만 존재한다. 같은
설정으로 노트북 00을 다시 실행하면 그 run을 재사용하며, 다른 모델·설정의
미완료 run을 조용히 덮어쓰지 않는다. 마지막 시각화가 저장되면 run을 완료
상태로 고정한다.

각 노트북은 해당 단계의 공통 Python 함수 하나만 호출한다. 실행값은 노트북에
하드코딩하지 않고
`configs/experiments/step2_pytorch_gradcam.yaml`의 `execution`, `gradcam`,
`joint_analysis`에서 읽는다. 현재 기본값은 `mode=real`,
`data_fraction=1.0`, `execute_stage=true`, `write_outputs=true`,
`overwrite=false`, `allow_dirty=false`, `device=cuda`인 전체 재실험용
프로필이다. 입력 checkpoint, 모델 UID, aligned bundle
또는 lineage가 불완전하면 실행은 즉시 중단된다.
