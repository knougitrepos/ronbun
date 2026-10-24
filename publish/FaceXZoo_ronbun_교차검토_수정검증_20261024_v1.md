# FaceX-Zoo / ronbun 교차검토 수정·검증

작성: 2026-10-24 KST. 대상: C:/ronbun step14와 F:/FaceX-Zoo adjust1의 현재 작업 트리.
HEAD: ronbun `e1997a1c78c48627084202d5b691f203fb157266`, FaceX-Zoo `b7cc6f4772ae09591e0d5b759ca41ec9088ba40b`.
기존 미커밋 변경을 보존했다. 이번 작업에서 staging/commit/push 및 정식 학습은 수행하지 않았다.

## 결론

Argon 교차분석의 핵심 결함 지적은 현재 코드에서 확인되었으며 수정했다. 데이터 분할 수량도
현재 handoff/roles와 pair partition으로 다시 검증했다. 두 저장소는 개별 운용하며 파일만 전달한다.
FaceX-Zoo에서 생성한 1024D export를 ronbun에서 사용하는 경계와 별도 Git 이력을 유지했다.

수정 후 ronbun 관련 회귀 **251개**, FaceX fine-tune **185개**가 통과했다. 이는 코드 동작과
소규모/합성 입력 검증이며 FR 성능이나 Continuous-FIQA 개선, 최적 데이터 분할의 증거가 아니다.

## 지적별 조치

| 지적 | 현재 조치 | 검증 근거 |
|---|---|---|
| 선택한 SCFace에 맞는 모델 없이 준비 완료 표시 | 데이터셋별 모델 매칭·조기 오류, 무관한 모델 제외 | 실제 inspect/prepare/run 경로의 오류 회귀 |
| 완료 입력 cache의 원본 변경 미검출 | 원본 바이트와 완료 image ledger 전수 대조, 불일치 시 중단 | 이미지 변경 후 재사용 거부, 기존 완료 manifest 보존 |
| 입력/결과 hash의 하위 계산 코드 누락 | static import 의존성 재귀 추적 + 동적 FaceX loader 명시 | rejection/origin_vs_pq/sample_efficiency/metrics 및 backbone/adapter/CR-FIQA 변경 시 UID 변경 |
| 03/04 HEADS 하드코딩 | 저장 HEADS·head 파라미터 유지 | ArcFace만/세 head/CosFace만 설정의 노트북 전체 코드 실행 |
| 잘못된 준비·export 안내 | 새 campaign의 부모 준비·학습 필요, export 의미 수정 | nbformat/AST 및 계획 모드 실행 |
| legacy 개수·LFW 전용 안내 잔존 | 56개 보관 안내, 현재 5개 데이터셋 결과 단계로 수정 | legacy 목록과 이전 HEAD notebook 보존 비교 |
| RFW tar 멤버 탐색 비용 | forward stream + TarInfo 추출 | gzip fixture 30장 1회 open, getmember/getmembers 호출 금지 검증 |
| CR-FIQA 반복 추론 | 동일 입력 조건의 품질 cache 공유, rank/codec-only 입력은 생략 | 두 FR 모델에서 한 번의 품질 계산, checkpoint/변환 변경 분리, 변조·재개 검사 |

주요 구현: `research/experiments/facex_inputs.py`, `facex_workflow.py`,
`research/runtime/source_fingerprints.py`. FaceX 변경은 `fine-tune/notebooks/`, 안내와 관련 테스트에 한정된다.
원본 dataset, upstream backbone/head, 기존 완료 모델/결과는 변경하지 않았다.

## 병목 주장에 대한 정확한 해석

FaceX `_run_one`과 `_export_one`이 `bind_aligned_dataset`의 기본 전수 검증을 호출하는 것은 사실이다.
SurvFace 176,205장에 4 heads × 5 seeds의 학습·export를 각 1회 수행하면 40회의 검증으로
**7,048,200개 record 검사**가 발생한다. record마다 원본/전처리 hash와 전처리 decode를 검사한다.
이는 호출 구조에 따른 계산이며 측정된 실제 소요 시간이 아니다. 검증 정책은 유지했다.

기존 RFW 입력도 이미 archive offset 순으로 정렬했으므로, “매 이미지마다 gzip을 처음부터
되감는다”는 해석은 과도하다. 다만 문자열 extractfile이 member 목록을 탐색하고 최초 목록을
전체 읽는 비용은 있었다. 새 구현은 ordered stream으로 처리한다. 역방향 임의 접근은 재open한다.

CR-FIQA 공유는 FR 모델과 무관한 고정 품질 추론에만 적용한다. FR embedding·raw_norm과 압축기는
모델별로 유지한다. 품질 값의 조작이나 fit/test 교환은 없다. 실제 전체 데이터 처리 시간·GPU
절감률은 측정하지 않았으므로 숫자로 주장하지 않는다.

## 데이터셋 분할

아래는 현재 로컬 frozen protocol 및 seed=8972의 실행 코드 집계다. 이번 수정에서 비율을 바꾸지 않았다.

| 데이터셋 | FaceX FR 파인튜닝 | PCA/PQ development | Continuous-FIQA fit / safety | 최종 평가 |
|---|---|---|---|---|
| SurvFace | 4,255명 / 176,205장 | FR 내부 1,255명 / 52,227장 | FR 밖 fit 744명 / 33,754장, safety 320명 / 10,929장 | gallery 3,000명 / 60,294장, 등록 probe 60,423장, 비등록 probe 121,736장; TPIR@20/FPIR |
| TinyFace | 2,570명 / 7,804장 | 같은 training 7,804장 | rank 평가에는 적용하지 않음 | gallery 4,443장, probe 3,728장, distractor 153,428장; rank |
| SCFace | 50명 / 800장 | 같은 training 800장 | rank 평가에는 적용하지 않음 | 다른 80명, 정면 gallery 80장, probe 1,200장; 전체/거리별 rank |
| LFW | 사용하지 않음 | 해당 모델의 외부 development에서 codec 고정 | outer fold별 나머지 9 folds에서 identity 분리; 유효 impostor fit 1,245–1,288쌍 / safety 232–260쌍 | 10 folds, fold당 genuine 300 + impostor 300; TAR/FMR |
| RFW | 사용하지 않음 | 해당 모델의 외부 development에서 codec 고정 | 그룹·outer fold별 test identity purge 후 분리; impostor fit 612–855쌍 / safety 104–184쌍 | 4그룹 × 10 folds, fold당 600쌍; 그룹별 TAR/FMR |

SurvFace FR 176,205장은 PQ 52,227장 + calibration gallery 61,433장 + 등록 calibration probe
62,545장의 합이다. gallery/probe의 3,000명은 같은 FR 학습 인물이다. 비등록 fit/safety 1,064명은
downstream FR 학습에서 제외된다. 따라서 calibration 전체가 FR-unseen이라는 주장은 하지 않는다.
FR 176,205 + 비등록 fit 33,754 + safety 10,929 = 공식 training 220,888장이다.

RFW 그룹별 impostor fit/safety 범위: African 773–855 / 124–154, Asian 612–685 / 104–140,
Caucasian 722–791 / 160–184, Indian 752–804 / 132–163. 9 folds의 5,400쌍은 후보 수이며
purge 및 양쪽 endpoint 역할 분리 후 실제 임계값 학습 수와 다르다. 50개 outer 평가에서
fit/safety/test의 라벨된 identity 교집합이 없는지 검사했다.

SCFace 50/80 ID는 seed8972 연구자 분할이다. UCFace의 정확한 저자 ID 목록과 동일하다는 증거는 없다.
SurvFace 비등록 probe 28,476장과 TinyFace distractor 153,428장은 신원 라벨이 없어 실제 인물
중복까지 검증하지 못했다. 부모 사전학습 인물 overlap도 별도 감사가 필요하다.
PQ 비트는 Surv8/Tiny7/SC4로 다르므로 code 크기·codebook 비용을 함께 보고해야 한다.

FaceX의 기존 수치 정책에도 한계가 남는다. `test_adjust1.py::test_target_only_boundary_contract`는
ArcFace/MagFace의 target cosine이 정확히 ±1인 조건에서 원래 sqrt 수식의 비유한 gradient가
남는 것을 확인한다. 테스트 통과가 모든 입력에서 NaN이 발생하지 않는다는 뜻은 아니다.
이번 수정에서는 기존 target-only original sqrt 정책을 바꾸지 않았다.

## 노트북 정리와 실행 상태

ronbun 활성 경로는 `01_data_preparation → 02_inputs → 03_experiments → 04_results`이고
기존 notebook 56개는 legacy에 보존했다. FaceX는 `00 설정 → 01 부모 입력 → 02 부모 학습 →
03 후속 학습 → 04 export`다. 각 단계를 새 커널에서 시작할 수 있도록 설정을 파일로 읽는다.
기본 EXECUTE=False와 저장된 출력 없는 상태를 유지한다. UI의 시각적 렌더링 검사는 수행하지 않았고,
형식·Python AST·안전 기본 실행 및 설정 전달을 검사했다.
HEAD의 기존 52개 중 47개는 JSON 전체가 같고, 5개는 이전 정리에서 안내 추가·문서 경로·표시
문구가 바뀐 보관본이다. 나머지 4개는 직전 LFW 순차 노트북이다. 이번 수정에서 legacy notebook
본문을 변경하지 않았으며 README의 개수·역할 안내만 고쳤다.

현재 ronbun YAML의 models는 비어 있어 실제 모델을 등록해야 한다. FaceX research_v1 설정에는
새 checkpoint namespace와 전체 전처리 보고서가 아직 준비되지 않았다. 검증한 handoff 역할 파일이
있다는 것과 저장된 campaign 설정이 그 protocol을 선택하여 준비·학습을 끝냈다는 것은 별개다.
03에서 준비 후 정확한 campaign.handoff_path와 완료 export/spec을 명시적으로 전달해야 한다.

이번 수정만으로 FaceX의 검증된 동일 조건 완료 모델을 다시 학습할 필요는 없다. ronbun은 코드
fingerprint 변경 때문에 02의 새 입력 UID/index와 03의 새 결과를 생성한다. 기존 완료 결과는 그대로
남으며 현재 코드 결과로 재명명하지 않는다. 모듈 수정 후 notebook 커널을 재시작한다.

## 후속 질문 세션에 남기는 선택지

1. **FaceX 데이터 무결성 검증 주기**: A. 매 학습/export 전 전수 검증 유지(현재 적용).
   B. 별도 불변 데이터 snapshot을 먼저 만들고 최초 전수 검증을 공유하는 정책 설계.
   B는 변경 감지 범위·추가 저장 공간·중단 후 재검증을 정해야 하므로 답변 전 구현하지 않는다.
2. **분할 최적성 실험 범위**: A. 현재 고정 분할로 기준 실험을 먼저 수행.
   B. 여러 FR/calibration 비율에서 별도 모델을 학습하는 민감도 실험을 추가.
   현재 fit-only sample_efficiency는 B를 대신하지 않으며 최적 FR 비율을 증명하지 않는다.
   모델별 FR 성능 변화와 safety tail 표본 불확실성을 실제 평가해야 한다.
3. **경계 gradient 수치 정책**: A. 현재 수식을 유지하고 학습 중 비유한 값에 대한 중단·진단 유지.
   B. target 경계 안정화 수식을 별도 recipe로 구현·검증하고 새 provenance의 모델을 생성.
   B는 단순 노트북 수정과 달리 학습 계산을 바꾸므로 사용자 답변 전 적용하지 않는다.

이 항목들은 이번에 수정한 실행 결함과 구분한다. 저장소 통합이나 test 데이터의 calibration 전용 전환은
어느 선택지에도 포함하지 않는다.

## 검증 근거

- `publish/support/facex_audit_fixes_20261024/validation.json`: 현재 코드 hash, Git 경계, 테스트 XML hash,
  notebook/설정 상태, protocol 출처와 집계 요약.
- 같은 폴더 `dataset_roles.csv`, `pair_partitions.csv`: 원본 대용량 결과가 아닌 역할/50개 partition 집계.
- `tmp/facex_fixes_20261024/final_tests.xml`: ronbun 251 passed, failures/errors/skipped 0.
- `F:/FaceX-Zoo/fine-tune/data/repo_independent_review_20261024/final_tests.xml`: 185 passed,
  failures/errors/skipped 0. 기존 cropper의 SimilarityTransform API deprecation warning 9건.
- 입력/결과 코드 및 새 테스트 targeted Ruff 통과, 두 저장소 git diff --check 통과.

MobileFaceNet·Resnet IR-50 × 4 head 호환성은 합성 가중치 검증이다. 별도 consumer 프로세스에서
FaceX checkout 파일 접근을 차단한 검증도 포함한다. 실제 모델 학습 완료·전체 CUDA 추출·DB 검색
성능·Continuous-FIQA 개선은 이번 작업에서 실행하거나 입증하지 않았다.
