# FaceX-Zoo → ronbun 실험 실행과 결과 해석

2026-10-23 KST. 현재 진입점은 01 → 02 → 03 → 04이다. 첫 설정 셀의
CONFIG_PATH, DATASETS, OUTPUT_ROOT, EXECUTE, KEEP_RAW_RESULTS를 확인하고 커널을 재시작한다.

## 합의한 실행 단계 — 2026-10-25

03의 `STUDY_STAGE="baseline"`으로 현재 고정 분할을 먼저 평가한다. 기준 실행이 완료된 후
`STUDY_STAGE="calibration_size"`, `BASELINE_REPORT=<정확한 완료 report_dir>`로 별도 분석한다.
YAML/직접 Python 호출도 같은 `study.stage`, `study.baseline_report` 또는 함수 인자를 사용한다.
두 번째 단계는 기준 보고서의 완료/hash, 모델 UID, protocol, quality, 입력 lineage를 검사하고
그 보고서에 저장된 PCA/PQ codec을 로드한다. codec을 다시 fit하거나 FR 모델을 재학습하지 않는다.
기준 보고서와 다른 조건이면 새 기준 실험이 필요하다. 이전 보고서를 덮어쓰지 않는다.

표본 수 분석은 SurvFace/LFW/RFW만 선택한다. TinyFace/SCFace는 기존 rank 평가를 유지하며
임의 threshold 프로토콜을 추가하지 않는다. LFW/RFW는 여전히 FR 학습에 사용하지 않는다.
SurvFace는 원래 calibration-fit 인물, pair 데이터는 outer-test 인물 제거 후 원래 fit 인물만 사용한다.
그 내부를 inner fit/safety/validation으로 identity-disjoint 분리하며 pair는 양쪽 인물이 같은 역할일
때만 남긴다. 최종 safety와 최종 test 점수는 이 진단에 사용하지 않는다.

`sample_efficiency.fit_fractions`, `safety_fractions`는 서로 독립적으로 변화시키는 **내부 pool의 인물 비율**이다.
전체 FR/calibration 배분 비율이 아니며 실제 이미지/쌍·인물 수를 metrics.csv에 함께 기록한다.
기본 5 folds, inner safety 0.3, 비율 0.25/0.5/0.75/1.0은 분석용 후보이며 최적값이 아니다.
각 fold에서 validation은 고정되고 비율별 subset은 중첩된다. 이를 독립 반복 표본으로 합산하지 않는다.

Stage 2 읽기 순서: HOW_TO_READ.md → chatgpt_summary.json → overview.csv → 관심 조건의 summary_parts.
각 part는 모델·데이터셋·group·outer fold·압축·목표 비율 단위이며, 그 조건의 모든 inner fold와 fit/safety 비율을 남긴다.
조건·파일 수·바이트·추정 토큰량은 summary index에 기록한다. 전체 metrics.csv 업로드는 선택 사항이다.
`fit_samples/safety_samples/validation_samples`는 `sample_unit=query`이면 probe 수, `pair`이면 쌍 수다.
`metric`은 FPIR 또는 FMR이며 `validation_false_accepts / validation_samples`가 관측 비율이다.
`validation_target_met`는 반올림 전 관측 비율 판정이다. `*_expected_tail_count=N*target`은 tail 표본
규모 진단이고 보장이나 충분 표본의 인증이 아니다. Wilson 구간은 행 단위 설명용으로, 인물 의존성이나
중첩 subset 간 독립성을 가정한 확정적 보장으로 읽지 않는다.
`insufficient_rows`도 표에 보존한다. inner fit/safety 각 20개 미만 또는 validation 부재 조건은
모델을 fit하지 않았다는 뜻이며, 완료 artifact가 모든 조건의 유효한 추정을 의미하지 않는다.
이 단계는 Global-safe와 Continuous-FIQA의 FPIR/FMR·pinball 안정성 진단이며 test TPIR/TAR 개선을
선택하거나 입증하지 않는다. 최종 기준 실험의 TPIR@20/TAR와 구분한다.

Stage 2는 집계표만 생성한다. KEEP_RAW_RESULTS와 무관하게 상세 scores/identities를 새로 저장하지 않으며
기준 run의 입력·codec·모델은 보존한다. manifest에 기준 run ID/hash와 보존 정책을 기록한다.
중단된 `.partial`은 보존하며 같은 명시적 조건으로 재실행하면 진단을 재계산한다. 완료 run은 재사용한다.
FR/calibration 배분 비교는 이 진단에서 필요성을 검토한 다음 별도 protocol·namespace와 새 FR 모델로
계획한다. 자동 비율 sweep이나 test 결과에 따른 자동 조건 선택은 구현하지 않는다.

FaceX-Zoo의 새 parent 및 ArcFace/MagFace run은 `target_only_fp32_zero_radicand_eps_v1` 정책이다.
정답 클래스에서 FP32의 `1-c*c == 0`일 때만 epsilon으로 대체하고 MagFace norm 경로를 유지한다.
이는 저자 공식 수정이 아닌 로컬 수치 정책이다. 모델 sidecar/source_config의 정책·hash가 전달 경계이며,
ronbun은 FaceX-Zoo 학습 코드를 import하거나 loss를 재구현하지 않는다.

FaceX-Zoo의 모델 생성 연구와 ronbun의 모델 활용 실험은 별개다.
두 저장소를 합치거나 상대 저장소의 학습 모듈을 import하지 않는다.
전달 경계는 export(.pt), sidecar(_meta.json), 명시적으로 선택한 데이터 역할 기록이다.
모델 파일은 복사해 등록할 수도 있다. 입력 데이터와 protocol이 참조하는 파일은 설정된 위치에서 읽는다.

## 실행 준비

FaceX-Zoo 00에서 연구 프로필과 backbone을 선택하고 03에서 학습하며 04에서 export한다.
SurvFace·TinyFace·SCFace만 FR을 학습한다. LFW·RFW는 FR 학습에 쓰지 않는다.
FaceX-Zoo가 생성하는 `<DATASET_ROOT>/protocols/shared_v1/handoff.json`은 정확한
프로토콜 경로·hash를 고정한다. ronbun이 독립적으로 인물을 다시 나누지 않는다.
새로 준비한 데이터셋을 추가하면 FaceX-Zoo는 handoffs/<hash>.json에 새 불변 index를 생성하고
03/04의 출력과 저장 설정 campaign.handoff_path에 정확한 경로를 남긴다.
ronbun YAML의 handoff를 그 경로로 지정한다. 최초 handoff.json과 이전 version은 덮어쓰지 않으며
자동으로 최신 index를 선택하지 않는다. 같은 데이터셋의 분할 변경은 별도 namespace를 사용한다.
SCFace 50/80은 seed8972의 연구 분할이며 저자의 정확한 ID 분할 재현이라고 주장하지 않는다.

완료된 비-smoke export를 `py -3.11 -m research.embeddings.facex_zoo EXPORT --registry-root DIR`로
등록하고 `configs/experiments/facex_workflow.yaml`의 models에 alias, spec, training_dataset을 지정한다.
등록할 정식 학습 export가 없으면 01의 상태는 model_specs_required이다. 모델을 추측하거나 최신 파일을 선택하지 않는다.
02는 CUDA에서 1024D unit embedding, raw norm, CR-FIQA(L)를 생성한다. CR-FIQA가 기본이며
raw_norm 품질 신호는 YAML에서 명시적으로 선택하는 별도 실험이다. 변형은 별도 입력·결과 UID를 만든다.
입력 이미지는 export의 preprocessing.alignment_policy를 읽어 전체 이미지를112×112 resize한다.
bicubic 또는 bilinear를 동일하게 적용하고 실제 정책을 cache key/progress/manifest에 기록한다.
정책 누락·지원하지 않는 보간/크기/geometry는 입력 추출 전에 거부한다. 임의 기본값으로 대체하지 않는다.
detection/landmark alignment를 수행했다고 표현하지 않는다.
VGGFace2 부모의 제공 landmark 정렬과 downstream resize-only 전처리는 구별한다.

## 연구 분할과 한계

SurvFace FR4255명176205장, PQ-development1255명52227장, calibration gallery3000명61433장,
등록자 진단probe62545장, 비등록자fit744명33754장/safety320명10929장이다.
Gallery/PQ는 FR 학습 인물의 하위 역할이다. 비등록자fit/safety는 downstream FR에서 제외되지만
calibration gallery/등록자probe는 FR-seen이다. 공식 test gallery/probe는 FR-unseen이므로
분포가 완전히 일치하지 않는다. 부모 모델의 identity overlap은 미감사이다.
80/20 및70/30은 사전 고정 후보이며 최적값 또는 수학적 FPIR 보장이 아니다.

SurvFace gallery는 calibration/test 모두 이미지 unit embedding 평균 후 L2 정규화한 identity template이다.
공식 MATLAB의 평균·euclidean과 정확히 동일한 점수 recipe를 재현한다고 주장하지 않는다.
TinyFace는 이미지 gallery, distractor 포함/제외 양쪽 rank1/5/10/20을 보고한다.
SCFace는 정면 gallery80장과 가시광probe1200장의 전체·거리별rank를 보고한다.
TinyFace/SCFace rank 실험에 threshold에 의한 순위 개선을 주장하지 않는다.

LFW/RFW는 공식test쌍·fold를 유지하고 추가 calibration 절차를 적용한다.
Outer test 인물과 겹치는 후보 쌍은 제거하고 두 endpoint가 같은fit/safety역할일 때만 유지한다.
9fold5400쌍은 후보 수이며 실제fit/safety수와 다르다. 특히RFW의 safety가 매우 작을 수 있다.
Fold test impostor300쌍의 FMR 한 건은0.333%p이다. 반복seed는 독립 표본 수를 늘리지 않는다.
RFW 입력은 네 그룹 각각 10 folds와 fold당 genuine/impostor 300/300을 모두 검사한다.
정답 표기는 양 끝 인물 ID의 일치 여부와 같아야 하며, 빈 ID·self-pair·image/identity 불일치는
입력 단계에서 거절한다. 이 구조 검증이 공식 배포 파일 자체의 출처 인증을 대신하지는 않는다.
PCA/PQ는 해당 checkpoint의 외부development에서 학습·고정하며 LFW/RFW로 전이한다.
다른 checkpoint 사이에1024D가 같다는 이유로 codec을 공유하지 않는다.

작은 calibration subset에서 1% 목표를 해석할 때는 `validation_samples`와
`validation_false_accepts`를 먼저 확인한다. 오류 한 건의 변화는 `100/N` %p이고,
예를 들어 20쌍이면 5%p이다. 0건 관측을 오류율 0 또는 1% 이하 보장으로 해석하지 않는다.
Stage 2의 최소 fit/safety 20개 기준은 계산 가능 여부만 검사한다. 안정적인 tail 추정이나
최적 FR/calibration 배분을 인증하지 않으며, 불충분 조건을 제거해 좋은 결과만 보고하지 않는다.

PQ 기본 연구설정은 SurvFace8bit/TinyFace7bit/SCFace4bit이다. Faiss 경험적 최소39개/centroid를
사전 검사한다. 이 검사는 충분성 증명이 아니다. bit수가 다르므로 동일m도 payload bytes가 다르다.
비교 시 `ceil(m*nbits/8)`과 codebook 비용을 함께 확인한다. PCA는 centered rank 상한을 검사한다.

## 결과 읽기

완료 조건은 run_manifest.json status=completed와 COMPLETED 파일 및 출력hash 일치이다.
추가로 manifest의 contract에서 계산한 UID, 저장된 run_uid와 디렉터리명이 같아야 한다.
자동 재사용은 현재 요청 contract와도 일치해야 한다. 다른 조건의 report를 이름만 바꾸어
복사하면 거부한다. 이동·백업 복원 시 원래 UID 디렉터리명을 유지한다.
진행 중/실패한 실행은 명시적인 `.partial` 폴더에 남고 완료 결과로 읽지 않는다.
RUNNING.lock에는PID가 있다. 비정상 종료 후에는 해당 프로세스가 종료됐는지 확인한 뒤 잠금을 처리한다.
초기 manifest 쓰기 등 일반 예외에서는 해당 실행이 획득한 lock을 해제한다.
datasets는 설정과 실행 인자 모두 중복 없는 비어 있지 않은 목록이어야 한다.

먼저 HOW_TO_READ.md, chatgpt_summary.json, overview.csv 순으로 읽는다.
그다음 summary_parts가 지정하는 모델·데이터셋별 metrics/paired를 선택한다.
overview는 fold별 trial 수를 합산한 설명용 요약이며 반복 인물을 독립 표본으로 취급한 CI를 만들지 않는다.
metrics에는 모든 선택 모델·데이터셋·압축·목표조건이 포함된다. 분모·성공/오류수·미완료 작업을 확인한다.
SurvFace의 TPIR@20은 genuine identity의 Top20 포함과 genuine score의 threshold 통과를 모두 요구한다.
rank_failure_count는 threshold와 무관한 순위 실패이고 threshold_failure_count는 Top20 안에서의 거부이다.
Continuous는 비등록자 조건부quantile을 학습하며 TPIR을 직접 최대화하거나 순위를 복구하지 않는다.

주 비교는 동일fit/safety에서 학습한 global_safe 대비 continuous_fiqa의 TPIR 또는 TAR 차이다.
목표 FPIR/FMR와 realized 값은 다를 수 있다. target_met_on_test/both_target_met를 반올림 전 수치로 확인한다.
목표 초과 조건도 표에서 제외하지 않으며 FPIR 상승을 TPIR 개선과 함께 해석한다.
Origin cosine, PCA cosine, PQ negative_squared_l2_adc는 별도 점수 공간에서 각각 보정한다.
2026-11-06부터 Global threshold 및 Continuous의 held-out safety offset에 사용하는
공통 경험적 quantile은 다음 거부 score의 바로 위 float64 값(nextafter)을 선택한다.
score >= threshold 규칙에서 동점은 함께 처리하고, calibration 오인식률 상한을
넘지 않는 가장 허용적인 경계를 사용한다. Continuous는 기존대로 offset을 0 이상으로
제한하고 실제 덧셈 후 판정도 확인한다. 이 변경은 test FPIR/FMR 보장을 뜻하지 않는다.
이전 threshold로 계산한 완료 결과는 보존한다. 새 방법 비교는 동일 partition과 조건에서
Global과 Continuous를 함께 다시 보정·평가하고 새 UID에 저장한다. 이전 점수와 새 임계값의
결과를 같은 조건으로 섞어 보고하지 않는다.
paired TPIR 구간은 genuine 인물 단위 bootstrap이며 gallery/threshold 고정 조건부 구간이다.
FPIR/FMR Wilson 구간은 probe/pair 수준의 설명용 구간으로 반복 인물 의존성을 완전히 반영하지 않는다.
작은 tail 표본 또는 분포 이동에 대해 수학적 오류율 보장을 주장하지 않는다.

## 보존과 요약

KEEP_RAW_RESULTS=True는 이번 실행의 상세 per-query 결과를 보존한다. False는 상세 파일 생성을 줄이고
요약 검증 후 해당 실행의 details만 정리한다. 입력 데이터 원본·사전학습/학습checkpoint·입력embedding cache·codec·
재개용 job요약과 최종manifest/집계/모델계수는 보존한다. 기존 완료 run은 옵션 변경으로 삭제하지 않는다.
요약만 있으면 운영지표·paired차이·모델계수·출처를 확인할 수 있다. 임의의 샘플·threshold·subgroup 재분석은
상세가 없으면 입력cache에서 점수를 재계산해야 한다. 입력cache까지 수동 정리했다면 이미지부터 재추출한다.
상세 보존 대상은 SurvFace의 query score·threshold·판단, TinyFace/SCFace의 probe별 Top-20 내
genuine rank·score·거리·distractor 조건, LFW/RFW의 pair score·threshold·판단·양쪽 image/identity다.
rank가 비어 있으면 Top-20 밖이거나 genuine이 없다는 뜻이며 전체 gallery 순위를 저장한 것이 아니다.
LFW/RFW 상세는 group/fold별 decisions.csv로 나눈다. 모든 상세 파일은 완료 manifest와
작업별 재개 기록의 hash 검증 대상이다. 빈 paired 표는 rank-only 실행에서 정상이다.
입력 population hash는 추출 호출당 한 번 계산하고 각 배치의 진행 기록에서 재사용한다.
완료본을 읽을 때도 population.csv 내용의 hash와 manifest의 population_hash를 대조한다.
중단 재개는 배치별 저장 임베딩·raw_norm·quality와 이미지 hash ledger의 검증 기록을
확인한 뒤 수행한다. 검증된 마지막 배치 이후의 미확정 꼬리는 다시 계산한다.
기록이 없거나 변경된 과거 partial은 자동으로 신뢰하지 않는다. 기존 파일을 보존하고
별도 output_root/새 artifact에서 재추출한다. 원본 이미지 불변 전제만으로 파생 cache의
무결성이 확보되지는 않는다.
디스크 사전 검사는 완료 cache 재사용 확인 뒤 artifact 잠금 안에서 수행한다.
이미 할당한 부분 파일을 고려해 추가 공간을 추정한다. estimated_input_bytes는 전체
벡터 payload 규모를 설명하는 수치이며 재사용 시 요구하는 추가 공간과 다르다.
완료 입력 cache를 02에서 재사용할 때도 image_hashes.jsonl과 현재 원본 바이트를 전수 대조한다.
동일 경로의 원본 내용이 바뀌면 중단하며 이전 완료 artifact를 덮어쓰지 않는다. 의도적으로 원본을
교체했다면 별도 output_root에서 새 입력·실험을 생성한다. 이미 고정한 입력을 읽는 03/04는
원본 이미지의 현재 존재 여부와 별개로 artifact 자체의 manifest/hash를 검증한다.
입력·결과 UID에는 실제 Python 하위 의존 코드의 hash가 포함된다. 동적 FaceX loader도 명시적으로
포함하며 다른 저장소의 소스를 import하지 않는다. 코드가 바뀌면 새 입력 index와 결과 UID를 쓴다.
CR-FIQA 품질 점수는 이미지 순서·변환·품질 checkpoint·코드가 같은 경우 output_root/quality에서
공유한다. 각 FR checkpoint의 embedding/raw_norm/codec은 따로 만든다. 공유 품질을 사용할 때도
각 이미지 원본 hash를 대조한다. TinyFace·SCFace rank 및 외부 codec 준비만 수행하는 입력에는
품질 추론과 quality.npy 생성을 생략하고 manifest에 source=not_required를 기록한다.
공유 품질 cache는 입력 재사용용 상세 artifact로 Git에서 제외하고 KEEP_RAW_RESULTS와 별개로
보존한다. 삭제하면 다음 입력 추출에서 품질 추론이 다시 필요하다. 중단된 품질 결과를 완료 cache로
게시하지 않는다. RFW tar 입력은 archive 순서로 읽으며 임의의 역방향 접근은 새 스트림으로 재개한다.
데이터 크기와 GPU속도를 먼저 확인한다. checkpoint·seed 전체 행렬은 자동 실행하지 않는다.
모듈 변경 후에는 커널을 재시작하고 입력·codec·점수·보정을 새 UID에서 생성한다.

02의 PERSIST_ORIGIN_TO_DB=True는 검증된 입력cache를 별도 embedding_1024 테이블로 저장한다.
기존 images와pgvector 초기화가 필요하며 512D테이블을 변경하지 않는다. 기본 평가는 파일/Faiss 경로다.
DB 저장·검색 API의 검증과 PostgreSQL 검색 성능 벤치마크 완료는 구분한다.

과거 sample_efficiency.enabled=True의 inline fit-only 진단은 새 workflow에서 사용하지 않는다.
상단의 별도 calibration_size 단계와 그 metrics.csv를 사용한다.

### ArcFace 대조 실험 실행

SCFace·TinyFace는 02와 03 첫 셀의 `CONFIG_PATH`를 `configs/experiments/facex_alignment_controls.yaml`로 지정하고 `DATASETS = None`으로 실행한다. YAML은 등록된 두 모델과 평가할 데이터셋을 고정한다. 입력 변환은 모델 export의 `affine.csv`에서 이미지 ID로 찾고 원본 바이트 해시를 확인한다. 두 데이터셋은 1:N 순위 평가이며 품질 임계값을 학습하지 않는다. 04에서는 03이 반환한 정확한 `REPORT_DIR`를 지정한다.

2026-11-15부터 SurvFace `continuous_margin`은 기본 모델 생성/논문 baseline 비교에서 제외했다. 기존 진단 모듈은 명시적인 대조 연구용으로만 남기며 새 baseline의 필수 단계로 실행하지 않는다. 저자 답변과 FaceX-Zoo의 기본 ArcFace 수식에 없는 변경을 기본 조건에 추가하지 않는다.

04의 순위 표는 Origin 대비 압축 후 Rank-1 변화(%p)를 보여준다. SurvFace의 Origin test ROC 표는 시험 집합에서 측정한 기술 통계이며, 배포용 임계값은 별도 보정 인물에서 결정한다. `compression_change_pp`는 각 점수 공간에서 다시 보정한 Global-safe의 Origin 대비 TPIR@20 변화다. `calibration_change_pp`는 같은 모델·압축 조건에서 Continuous-FIQA와 Global-safe의 TPIR@20 차이다. 두 방법의 실제 FPIR 및 `both_target_met`를 함께 읽는다. 이전 보고서는 당시 저장한 열만 표시한다.
# FaceX 선택 모델 목록 연결

FaceX 03에서 필요한 dataset·recipe·head·seed만 선택하고 04에서 전체 선택 모델의 완료 export를 검증하면 `MODEL_INVENTORY` JSON 경로를 출력한다. ronbun 02 첫 설정 셀의 `MODEL_INVENTORY`에 그 경로를 넣고, `INVENTORY_CONFIG_PATH`에 새 YAML 파일명을 지정한다. `EXECUTE=True`일 때 `research.experiments.facex_inventory.write_inventory_config`가 전체 목록·export 해시·학습 분할·정렬 파일을 검사하고 기존 registry에 등록한다. `EXECUTE=False`에서는 등록·추출하지 않는다. `MODEL_INVENTORY=None`이면 기존 `CONFIG_PATH` 방식을 그대로 사용한다.

새 v2 inventory는 `INVENTORY_PURPOSE="local_protocol_baseline"`로 읽는다. `exact_ucface_reproduction=False`, parent/전처리 출처 한계 및 데이터셋별 논문/현재 학습 수량을 YAML의 `model_scope`와 최종 report config에 보존한다. `diagnostic_control`은 명시적 대조 모델에만 쓰고, 과거 v1 inventory는 `legacy_local_control`을 명시해야 읽을 수 있다. continuous-margin recipe 또는 ArcFace `margin_am != 0`을 일반 baseline으로 등록하려 하면 차단한다. 기존 등록 ModelSpec/YAML과 완료 결과는 변경하지 않는다.

`INVENTORY_ORIGIN_ONLY=True`가 새 inventory 설정의 기본값이다. 생성된 YAML은 각 데이터셋의 `{family: origin}`만 포함하므로 PCA/PQ를 학습하지 않는다. 먼저 원본 cosine 성능을 확인한다. SurvFace에서 독립 calibration 임계값의 TPIR와 논문 비교용 test ROC는 서로 다른 평가라는 점을 유지한다. Origin 비교 후 압축 실험은 `INVENTORY_ORIGIN_ONLY=False`와 별도의 `INVENTORY_CONFIG_PATH`로 만들고 같은 모델·cohort의 Origin/PCA/PQ를 비교한다. `MODEL_INVENTORY=None`으로 읽는 기존 YAML은 이 옵션으로 바뀌지 않는다.

등록 후 출력한 YAML을 03의 `CONFIG_PATH`로 사용하고 `DATASETS=None`, `STUDY_STAGE="baseline"`에서 시작한다. 여기서 baseline은 ronbun 실험의 기준 단계이며 UCFace 논문 재현 인증을 의미하지 않는다. 04는 새 실험의 정확한 report 경로로 분석한다. 이전 결과를 자동으로 최신 결과에 연결하지 않는다. 기존 export를 재사용하면 모델 UID도 유지된다. 입력 재사용 여부는 population·모델·전처리·소스 해시를 확인하는 기존 cache 계약이 최종 판정한다. 신규 모델에는 새로운 임베딩과 각 모델의 PCA/PQ·보정 학습이 필요하다.

목록 변경 시 inventory와 YAML을 새 이름으로 생성한다. 이미 존재하는 다른 YAML은 덮어쓰지 않는다. 현재 생성기는 SCFace/TinyFace/SurvFace의 각 native 데이터셋 평가를 선택하며 LFW/RFW 학습을 만들지 않는다. 원본 보존은 기준 YAML의 `keep_raw_results`를 이어받으며, 압축 전 Origin 성능과 압축 후 성능·보정 효과는 별도로 읽는다. 새 모델이 UCFace 수치에 도달했거나 보정이 개선됐다는 결론은 실제 완료 결과로 판단한다.

노트북 대신 등록만 실행할 수도 있다:

```powershell
py -3.11 -m research.experiments.facex_inventory --inventory "완료 inventory.json 전체 경로" --baseline configs/experiments/facex_workflow.yaml --registry F:/models/facex/ronbun_registry --output F:/ronbun_results/facex_workflow/configs/selected_models_v2_origin.yaml
```

CLI 역시 Origin만 기본으로 생성한다. Origin 확인 후 별도 압축 설정을 만들 때 `--with-compression`을 사용한다.
