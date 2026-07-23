# Step 2 model validation

이 폴더는 ArcFace, AdaFace, MagFace의 실제 PyTorch checkpoint를 정량
실험에 투입하기 전에 검증하는 runbook입니다.

1. `00_checkpoint_registration.ipynb`
   - checkpoint 출처와 로컬 SHA-256
   - backbone·학습 데이터 표기
   - 정확한 전처리와 target layer
   - 외부 repository 전용 `package.module:function` loader
   - 불변 `ModelSpec` 등록
2. `01_preprocessing_and_model_smoke.ipynb`
   - 실제 loader와 checkpoint 로드
   - raw 512D, raw norm, L2-normalized 512D 출력
   - target layer 해석 가능 여부

코드가 존재한다는 이유로 checkpoint가 검증된 것은 아닙니다. 두 노트북의
기본값은 `EXECUTE_STAGE=False`, `WRITE_OUTPUTS=False`이며, 실제 경로와
논문/공식 저장소 근거를 채운 뒤 위에서 아래로 실행합니다.
