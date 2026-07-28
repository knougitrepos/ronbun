# LFW 노트북

LFW 실험은 `00_data_preparation`부터 숫자 순서대로 실행한다.

1. `00_data_preparation`: manifest와 공통 aligned crop 생성
2. `01_embeddings`: protocol/run 고정과 ArcFace 원본 임베딩 추출
3. `02_compression`: PCA/PQ fit, materialization, fallback-free 특성화
4. `03_open_set`: Step 1 fallback-free open-set 결과 checksum·지표 검증과
   읽기 전용 시각화
5. `04_gradcam`: PyTorch 원본 공간 saliency와 압축 민감도 결합

`04_gradcam` 전에 `00_data_preparation/02_landmark_region_materialization.ipynb`와
`../common/model_preparation/`의 checkpoint 등록·smoke 검증을 완료해야 한다.
세부 전체 순서는 [상위 실행 안내](../README.md)를 따른다.
