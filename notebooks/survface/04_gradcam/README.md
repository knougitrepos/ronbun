# SurvFace Step 4 Grad-CAM 실행 순서

SurvFace는 LFW와 동일하게 prerequisite와 experiment를 분리하되 공식
gallery/mated/unmated protocol 경계를 유지한다.

## Prerequisite

1. `prerequisite/00_source_and_model_freeze.ipynb`
2. `prerequisite/01_origin_embedding_and_top1_gallery_templates.ipynb`

첫 단계는 전체 training/official source, aligned crop, 106-point landmark,
검증된 ModelSpec과 공식 protocol 순서를 고정한다. 두 번째 단계는 전체
원본 512D embedding을 만들고 registered/unmated probe마다 frozen origin
top-1 gallery target을 고정한다.

## Experiment

1. `experiment/00_population_gradcam_extraction.ipynb`
2. `experiment/01_saliency_feature_validation.ipynb`
3. `experiment/02_step2_compression_characterization.ipynb`
4. `experiment/03_saliency_compression_join.ipynb`
5. `experiment/04_representative_case_visualization.ipynb`

압축기는 training development에서만 학습하고 threshold는 training
calibration에서만 고정한다. official gallery/mated/unmated는 평가에만
사용한다. Grad-CAM 적격 표본 전체 coverage와 선택 hash를 기록한다.
마지막 노트북이 대표 사례와 summary를 저장한 뒤 run을 완료 상태로 고정한다.
