# Step 2 model validation

이 폴더는 ArcFace, AdaFace, MagFace의 실제 PyTorch checkpoint를 정량
실험에 투입하기 전에 검증하는 runbook입니다.

1. `00_checkpoint_registration.ipynb`
   - checkpoint 출처와 로컬 SHA-256
   - backbone·학습 데이터 표기
   - 모델별 고정 전처리와 target layer 자동 선택
   - 내장된 공식 repository 호환 `package.module:function` loader
   - 불변 `ModelSpec` 등록
2. `01_preprocessing_and_model_smoke.ipynb`
   - `MODEL_NAME`과 선택적 `MODEL_UID`로 registry의 ModelSpec 자동 선택
   - 별도 입력이 없으면 LFW deep-funneled manifest에서 smoke 입력 자동 생성
   - 실제 loader와 checkpoint 로드
   - raw 512D, raw norm, L2-normalized 512D 출력
   - target layer 해석 가능 여부

코드가 존재한다는 이유로 checkpoint가 검증된 것은 아닙니다. 두 노트북의
기본값은 `EXECUTE_STAGE=False`, `WRITE_OUTPUTS=False`입니다. 실제 checkpoint
경로를 지정한 뒤 위에서 아래로 실행합니다. 공통 crop source 형식은
`RGB uint8 NHWC`로 설정에서 자동 선택됩니다.
같은 모델 family의 checkpoint를 둘 이상 등록하면 smoke notebook에서
`MODEL_UID`를 명시해야 합니다.

자동 생성되는 112×112 이미지는 checkpoint 동작 확인 전용입니다. 정량 실험과
Grad-CAM population은 별도의 검증된 공통 aligned-crop artifact를 사용합니다.
