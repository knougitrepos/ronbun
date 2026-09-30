> 보관된 수동 실행 계약이다. 현재 주실험 순서는 [활성 노트북 안내](C:/ronbun/notebooks/README.md)를 따른다.
> LFW/SurvFace 최초 manifest 준비는 활성 경로에 있다. 아래의 과거 설정/단계는 현재 코드와 입력을 확인한 뒤 사용한다.

# Step 2 model prerequisite

1. `00_checkpoint_registration.ipynb`
2. `01_preprocessing_and_model_smoke.ipynb`

ArcFace, AdaFace, MagFace checkpoint의 출처와 SHA-256, backbone, 학습 데이터,
preprocessing, target layer를 등록한 뒤 실제 loader로 RGB uint8 NHWC 입력의
512D 출력과 raw norm을 검증한다.

기본 실행값은 `DATA_FRACTION=1.0`, `EXECUTE_STAGE=True`,
`WRITE_OUTPUTS=True`, `OVERWRITE=True`이다. 이 값은 검증을 우회하지 않는다.
checkpoint 경로나 provenance가 없거나 family 선택이 모호하면 fail-closed로
중단된다. 동일한 ModelSpec 재등록은 idempotent하며, 같은 경로에 다른 내용이
있으면 덮어쓰지 않는다.
