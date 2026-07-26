# SurvFace 노트북

다음 순서를 유지한다.

1. `00_data_preparation`
2. `01_embeddings`
3. `02_compression`
4. `03_open_set`

공식 MAT/CSV의 gallery, registered probe, unknown-unknown probe 역할과 순서를
보존한다. 압축기와 threshold는 development/calibration에서만 정하고 official
test는 평가에만 사용한다.
