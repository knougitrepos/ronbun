# UCFace baseline 방향 교정 및 중단 모델 보존

작성: 2026-11-15 KST. 대상: `F:/FaceX-Zoo`와 `C:/ronbun`, branch `refacto1`.

코드 수정과 경로 검증은 완료했다. UCFace와 같은 성능을 얻었다거나 새 baseline 전체의 학습을 완료했다는 보고서는 아니다. 정식 학습은 재개하지 않았다. 변경된 기본 목록에는 `continuous_margin`이 없으며 기존 가중치와 실행 결과를 보존했다.

## 1. 바로잡은 문제

사용자의 1차 목표는 UCFace Table I의 일반 loss baseline 비교다. 성능 저하 원인 가설에 해당하는 `margin_am=1-cos(0.6)` 대조 조건을 기본 모델 목록에 넣은 것은 잘못된 범위 확대였다. 단순 이름 변경으로 해결하지 않고 다음 실행 경로를 수정했다.

| 항목 | 교정 후 동작 |
|---|---|
| 기본 모델 목록 | SCFace/TinyFace/SurvFace 각각 ArcFace, CosFace, MagFace, AdaFace; seed 8972 |
| ArcFace | 기존 `margin_arc=0.6`, `margin_am=0`, scale 64 유지; continuous-margin 별도 진단만 허용 |
| 이전 v1 계획 | 읽기·보존 가능, 자동 재실행 불가; 명시적 목적이 있는 새 v2 사용 |
| 03·04 기본값 | `ucface_review`, `EXECUTE=False`; True만으로 미확인 모델을 학습하지 않음 |
| 기존 모델 재사용 | 완료·parent·seed·학습 설정·분할·전처리·SHA를 검증한 것만 재사용 |
| ronbun 전달 | 모델 목적과 재현 한계를 YAML에 저장; 진단 모델을 baseline으로 혼합하면 거부 |
| 새 비교 순서 | Origin만 기본 생성; 이후 새 YAML에서 PCA/PQ·보정 비교 |

`ucface_review`는 실행 승인용 기능이 아니라 재현 조건의 현재 미확인 상태를 표시하는 읽기 전용 경로다. `selected`는 기존 연구 조건을 이어가는 별도 경로다. 아직 새 UCFace 전체 재현 학습 경로가 완성된 것은 아니다.

## 2. 저자 이메일과 논문 대조

사용자가 2026-11-15 이 대화에서 전달한 제1저자 답변은 대부분의 baseline에 FaceX-Zoo 코드를 참고했고, Table I은 MobileFaceNet과 VGGFace2 pretraining을 사용했다는 추가 근거다. 저자는 원본 저장장치에 접근할 수 없어 기억에 의존한다고 밝혔다. 따라서 이를 정확한 code revision·가중치·SCFace 인물 명단 또는 전처리 실패 정책의 확인으로 확대하지 않는다.

이메일의 MobileFaceNet 512D와 출판본의 1024D가 충돌한다. 로컬 `F:/FaceX-Zoo/paper/UCFace.pdf`의 PDF 6쪽, 인쇄면 7181, IV-A.3을 텍스트와 렌더링으로 확인했다. 본문은 모든 주요 실험이 1024D라고 명시한다. `paper/부록.pdf` PDF 4쪽의 Table XVII와 하단 설명은 MobileFaceNet 512D를 별도 비교로 제시하고 Table I의 1024D와 구분한다. **Table I을 목표로 할 때 1024D를 유지한다.** 512D로 바꾸어 parent와 모든 모델을 다시 만들지 않는다. 저자의 기억 오류인지 별도 실험에 대한 설명인지는 확정하지 않는다.

주요 기준: MobileFaceNet 1024D, Adam, batch 64, 40 epoch, lr 0.001, 10 epoch마다 ×0.1, dropout 0.6; ArcFace m=0.6/s=64, CosFace m=0.3/s=64, AdaFace m=0.4/h=0.333/s=64, MagFace (10,110,0.45,0.8,20). parent ArcFace m=0.35/s=32는 FaceX-Zoo에 따른 로컬 선택이며 UCFace가 명시한 parent 값으로 부르지 않는다. 기본 `margin_am=0`은 [FaceX-Zoo ArcFace 원본 구현](https://github.com/JDAI-CV/FaceX-Zoo/blob/main/head/ArcFace.py)과 일치한다.

FaceX-Zoo [학습 안내](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/training_mode)는 112×112 정렬 입력을 권고한다. 현재 VGGFace2는 제공된 5점 landmark와 FaceX cropper를 사용한다. SCFace 두 눈/PFLD TinyFace는 로컬 정렬 정책이며, SurvFace resize-only는 landmark 정렬 수행과 구분해야 한다. 현 정렬기가 저자의 정확한 정렬기라고 확인된 것은 아니다.

출판 근거는 [공저자 논문 목록](https://chengyawlow.github.io/publications/)의 DOI 10.1109/TIFS.2024.3426973, TIFS 2024이다. 공개 논문·저자 페이지 및 코드 검색 범위에서 저자 실험의 원본 체크포인트와 전체 구현을 확보하지 못했다. 공개되지 않았다고 단정하는 것은 아니다.

## 3. 현재 보존된 후보 상태

다음 숫자는 **교정된 로컬 선택 목록 12개**의 상태다. 저장소에 존재하는 모든 과거 모델 수나 UCFace 재현 성공 수가 아니다.

| 데이터셋 | 현재 FR 학습 | 논문 부록 Table IX | 선택 모델 상태 |
|---|---|---|---|
| SCFace | 50명 / 800장 | 50명 / 800장 | 정렬 조건 4개 모두 완료; 저자 인물 명단 동일성 미확인 |
| TinyFace | 2,570명 / 7,804장 | 2,570명 / 7,804장 | ArcFace 완료, CosFace 재개 가능, MagFace/AdaFace 미시작 |
| SurvFace | 4,255명 / 176,205장 | 5,319명 / 220,888장 | 기존 일반 loss 4개 완료; 논문 전체 FR 분할과 다름 |

TinyFace CosFace는 epoch 20까지 완료하고 epoch 21의 122 batch 중 42 batch까지 저장했다. 다음 batch index는 42(사람이 세면 43번째)다.

```text
F:/models/facex/selected/ucface_recipes_v1/checkpoint/tinyface_cosface_pfld_five_point_alignment_s8972/finetune_tinyface_cosface_seed8972/finetune_mobilefacenet_1024d_cosface_recovery_e21_b42_9b3aff4a.pt
SHA256: 9c6afc833183c45fd4da70d48f12829492a7dbdf81dcebb24263792d5e77c647
config_hash: 2a5335c7a1e0b98a9a38a2b4d9489147303a30b43fd427fb82c3649f4e218ab3
```

CPU에서 backbone/head/optimizer/scheduler/RNG 실제 복원을 확인했다. 해당 정식 학습의 optimizer step은 추가하지 않았다. 새 SCFace CosFace/MagFace/AdaFace 3개는 40 epoch 완료 상태다. SurvFace continuous-margin은 미시작으로, 변경된 margin으로 학습한 부분을 폐기해야 하는 상황은 아니다.

## 4. 남은 재현 문제와 재실행 경계

1. **SurvFace 분할과 전처리**: 논문 전체 FR 학습 비교는 5,319명 전용 protocol과 별도 출력 경로가 필요하다. 이 경우 기존 보정용 1,064명은 FR-unseen이 아니므로 기존 독립 보정 실험에 그대로 투입할 수 없다. 현재 4,255명 모델은 그 연구 목적의 별도 기준선으로 보존한다. full-FR 전처리·모델은 이번에 생성하지 않았다.
2. **SCFace 인물 목록과 정렬 세부 구현**: 수량 일치가 같은 인물·같은 전처리를 뜻하지 않는다. 저자 원본 명단/코드가 없으면 공개 근거를 따르는 로컬 재구현으로 명시해야 한다. 성능을 보고 분할을 선택하지 않는다.
3. **Parent/수치 정책**: 기존 VGGFace2 parent의 1024D·제공 landmark 정렬을 유지한다. 공유 parent, 정확한 증강 수치, target-only FP32/zero-radicand 보호는 로컬 선택을 포함한다. 이번에 source-fingerprint 대상 학습 코드를 바꾸지 않아 동일 조건 복구를 유지했다.
4. **결과 비교**: 현재 1 seed와 논문 5회 평균은 구분한다. TinyFace는 distractor 유무, SCFace는 거리별/전체 Rank-1, SurvFace는 동일 FPIR에서 TPIR@20을 비교한다. 시험 ROC의 기술적 비교를 독립 calibration 성능과 섞지 않는다. 동일 숫자를 보장하는 설정은 없다.

이 차이를 해결했다고 표시하지 않도록 검토 모드를 기본값으로 둔다. 디스크의 수정된 노트북을 다시 열고 커널을 재시작한 후 Run All로 현재 계획을 확인할 수 있다. 기존 연구 조건의 재개는 `CAMPAIGN="selected"`이며, 그 선택이 UCFace 동일 재현을 의미하지 않는다.

## 5. 구현·검증 근거

이번 변경 전 HEAD는 FaceX-Zoo `d5a6cba000a192cfe6e91b4d76b0ad75338e37da`, ronbun `79c203885ab3b32aec984408943064e799ce1ab3`다. 두 저장소에는 기존 미커밋 변경이 있었으며 보존했다. 이번 작업은 커밋/푸시하지 않았다.

| 검증 | 결과 |
|---|---|
| FaceX selected/controls 테스트 | 15 passed; 합성 중단/재개 가중치 일치 포함 |
| FaceX notebook_workflow 테스트 | 8 passed; cropper의 기존 deprecation warning 2개 |
| ronbun inventory/alignment 테스트 | 18 passed; 목적·margin·위조 claim·분할·Origin 우선·노트북 전달 검사 |
| 03·04 실제 코드 셀 Run All | 계획 모드 통과, 학습/export 없음, True 실행 차단 확인 |
| 현재 12개 선택 checkpoint 검사 | 재사용 9 / 재개 1 / 신규 2 |
| 실제 producer → consumer | 완료 SCFace ArcFace 1개 export 바이트 유지, v2 inventory/등록/YAML/handoff 검증 |
| 정식 학습·전수 임베딩·압축/보정 실험 | 미실행 |

합계 41개는 중복 실행을 제외한 해당 테스트 수다. 전체 저장소 모든 테스트나 논문 정확도 검증을 뜻하지 않는다. 합성 테스트의 짧은 학습과 사용자의 정식 학습 재개는 구분한다.

주 변경 코드: FaceX `fine-tune/modules/ucface_reference.py`, `selected_campaign.py`, `ucface_controls.py`, `process_runner.py`, `scripts/run_campaign.py`, 03·04 및 README; ronbun `research/experiments/facex_inventory.py`, 02 및 WORKFLOW_GUIDE. 기존 완료 모델, 원본/전처리 이미지, 과거 run과 YAML은 덮어쓰지 않았다.

검증 산출물 루트: `F:/models/facex/diagnostics/ucface_reproduction_review_20261115/`

- `interruption_verified.json`: 복구 상태·SHA 및 실제 CPU 복원.
- `corrected_plan.json`: 선택 파일, 12개 run 상태, 분할/차원 재현 검토.
- `notebook_checks.json`: 03·04 계획 실행과 실행 차단 확인.
- `handoff_check/inventories/43bb72c6f711ff7a5bc9085f184154b4b2cb3c1e2fc51fe48bea0a8f37bb8fad.json`: 기존 SCFace ArcFace 한 개만의 전달 검증. 전체 12개 완료 inventory가 아니다.
- `handoff_check/ronbun_origin.yaml`, `origin_verification.json`: 압축 전 Origin만 포함하는 실제 검증 설정. 정식 결과가 아니다.

논문 파일 SHA-256: UCFace.pdf `ad96d1534b8336a5c1d7ae7450e1d7d0e05901de3b230f7f0b6c69106e047173`; 부록.pdf `1db2267e5367120d257c83abc93626df8beeda586b75b4e6c6dc289c29d80905`.

## 6. 보존과 분석 요약

기존 `KEEP_RAW_RESULTS` 정책은 유지한다. 입력 원본·전처리·parent·과거 완료 checkpoint는 삭제 대상이 아니다. 상세 모델 파일과 실행 원본은 외부 F 드라이브에 보존하며 Git에 넣지 않는다. 이 보고서와 `publish/support/ucface_baseline_correction_20261115/verification.json`은 채팅 분석용 경량 요약이다. 요약은 상태·출처·검증 범위를 설명하며 원본 점수나 학습 곡선을 대신하지 않는다. 파일 크기와 문자 수/3 기준의 대략적 텍스트 토큰 수는 해당 JSON에 기록한다. 결과가 아직 없는 모델의 성능을 추정해 채우지 않는다.
