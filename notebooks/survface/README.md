# SurvFace 노트북

다음 순서를 유지한다.

1. `00_data_preparation/00_data_preparation.ipynb`
2. `00_data_preparation/01_aligned_crop_materialization.ipynb`
3. `00_data_preparation/02_landmark_region_materialization.ipynb`
4. `01_embeddings/00_official_protocol_and_run_freeze.ipynb`
5. `01_embeddings/01_official_arcface_embedding_extraction.ipynb`
6. `02_compression/00_compressor_fit.ipynb`
7. `02_compression/01_official_compressed_materialization_and_index.ipynb`
8. `02_compression/02_step1_compression_characterization.ipynb`
9. `03_open_set/00_official_probe_search.ipynb`
10. `03_open_set/01_official_evaluation_and_visualization.ipynb`
11. `04_gradcam/prerequisite/`
12. `04_gradcam/experiment/`

공식 MAT/CSV의 gallery, registered probe, unknown-unknown probe 역할과 순서를
보존한다. 압축기와 threshold는 development/calibration에서만 정하고 official
test는 평가에만 사용한다.

QMUL-SurvFace 원본은 이미 잘라낸 저해상도 얼굴 이미지다. 따라서
`00_data_preparation/01`은 얼굴 검출기를 다시 적용하지 않고 모든 source
face crop을 동일한 bilinear 112×112 입력으로 materialize한다. 한 장이라도
누락되면 `_SUCCESS`를 만들지 않는다. 106-point 단계는 이 이미지에서 예측한
눈·코·입 anchor를 사용한다.

2026-07-28에 detector 방식으로 만든 기존
`data/interim/step4/survface/aligned_112`는 source 463,341장 중 755장만
포함하므로 현재 계약에서 재사용할 수 없다. 사용자가 이 폴더를 먼저
`bk/20260728_survface_detector_partial/aligned_112`로 보존한 다음
`00_data_preparation/01`과 `02`를 다시 실행한다. `01` 완료 조건은
`source=aligned=463341`, `failed=0`이다.

이미 `01` 결과가 위 완료 조건과
`preprocessing.mode=official_face_crop_resize`, `detector.enabled=false`를
만족하면 다시 생성하지 않고 `02`부터 재개한다. 극저해상도 표본에서 일부
눈·코·입·볼·턱 마스크만 비는 경우에는 그 부위를 측정 불가(`NaN`)로 기록하고
나머지 부위와 표본 처리를 계속한다. 단, 얼굴 전체/얼굴 외부 마스크가 비거나
landmark 좌표가 비정상인 경우에는 계속 실패 처리한다.

`01_embeddings/01`도 원본 저해상도 파일에 detector를 다시 적용하지 않는다.
완료된 `aligned_faces.npy`를 검증·등록된
`arcface-7972a704552df378345f` PyTorch checkpoint에 batch 64로 직접 입력한다.
DB 임베딩과 후속 Grad-CAM은 같은 model UID, checkpoint hash와 preprocess
hash를 사용해야 한다. 이 입력·모델 계보가 바뀌면 기존 run에 이어 쓰지 않고
`01_embeddings/00`에서 새 run을 만든다.

현재 실행 commit은 모든 단계에서 `MODE=real`, `DATA_FRACTION=1.0`을
사용한다. compressor는 같은 SurvFace training development에서 한 번 학습하며,
과거 LFW compressor 전이 노트북은 활성 실행 순서에서 제거했다. 공식 검색은
`origin_512`와 `pca_256`의 exact/HNSW 네 조합, `PROBE_LIMIT=None`과
`FULL_RUN_ACKNOWLEDGEMENT="SURVFACE_FULL_SEARCH"`로 registered 및
unknown-unknown probe 전체를 처리한다.

장시간 단계의 notebook 출력은 시작·완료와 약 10% 경계만 표시한다. batch
commit/checkpoint 주기는 그대로 유지되며 로그 주기만 줄어든다. 전체 데이터
실험은 일괄 CLI나 Codex가 대신 실행하지 않고 사용자가 위 노트북을 순서대로
직접 실행한다.

## Step 4 Grad-CAM 순서

위 1~10을 완료한 뒤 `04_gradcam`에서 다음을 실행한다.

1. `prerequisite/00_source_and_model_freeze.ipynb`
2. `prerequisite/01_origin_embedding_and_top1_gallery_templates.ipynb`
3. `experiment/00_population_gradcam_extraction.ipynb`
4. `experiment/01_saliency_feature_validation.ipynb`
5. `experiment/02_step2_compression_characterization.ipynb`
6. `experiment/03_saliency_compression_join.ipynb`
7. `experiment/04_representative_case_visualization.ipynb`

Grad-CAM target은 frozen origin 공간에서 선택한 top-1 official gallery
template이다. registered와 unmated의 의미를 pooling하지 않는다. saliency
상한은 `null`이므로 target 적격 probe 전체를 사용하며, compression/open-set
평가는 공식 protocol 전체를 사용한다.
