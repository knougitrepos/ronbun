# FaceX-Zoo·ronbun 실행 오류 5건 수정 및 검증

작성일: 2026-10-23 KST. 상태: 코드 수정 및 아래 범위의 검증 완료. 정식 학습·전체 성능 실험 미실행.

## 결론과 저장소 경계

앞선 `FaceXZoo_ronbun_코드검토_20261023_v1.md`에서 확인한 P1 두 건과 P2 세 건을 수정했다.
완료·중단 후 재개·상세 보존·데이터셋 추가·전처리 전달 경로에 회귀 검증을 추가했다.
이번 검증은 해당 결함의 해결 근거이며 전체 연구 실험의 완료나 인식 성능 개선을 의미하지 않는다.

FaceX-Zoo와 ronbun은 별개의 저장소, Git 이력, 실행 환경과 연구 책임을 유지한다.

| 저장소 | 책임 | 전달 경계 |
|---|---|---|
| F:/FaceX-Zoo | 모델 학습·파인튜닝·export 및 학습 데이터 역할 기록 | export 가중치, sidecar와 선택한 분할 manifest 파일을 생산 |
| C:/ronbun | 생성된 모델로 임베딩·압축·보정·평가 | 전달받은 파일의 hash·차원·전처리·분할을 검증하여 사용 |

저장소 통합, submodule 추가, 상대 저장소의 학습 모듈을 runtime import하는 변경은 하지 않았다.
ronbun의 모델 구현과 loader는 ronbun 안에 있고, FaceX-Zoo 학습 환경을 실행하지 않고 파일을 읽는다.
FaceX-Zoo의 split_reference는 명시적인 입력 CSV이며 별도 위치에 복사한 파일로 설정할 수 있다.
이것은 코드 통합이나 ronbun Python 환경에 대한 학습 의존성이 아니다.

검증 기준은 C:/ronbun `step14` / HEAD `e1997a1c78c48627084202d5b691f203fb157266`,
F:/FaceX-Zoo `adjust1` / HEAD `b7cc6f4772ae09591e0d5b759ca41ec9088ba40b`와 각각의 미커밋 변경이다.
HEAD만으로 이번 코드를 재현할 수 없으므로 지원 요약에 대상 소스 SHA-256을 별도 기록했다.
기존 작업 변경을 보존했고 Git staging·commit·push는 수행하지 않았다.

## 결함별 수정과 확인

| 기존 문제 | 수정 | 확인한 실행 범위 |
|---|---|---|
| P1. Windows 빈 paired CSV 때문에 TinyFace/SCFace 완료·재개 실패 | optional CSV reader에서 pandas EmptyDataError를 빈 표로 처리. 누락 파일·기타 파싱 오류와 hash 오류는 계속 거부 | 두 데이터셋 각각 KEEP_RAW_RESULTS True/False에서 rank job 뒤 강제 중단 → CRLF-only CSV → job 재실행 없이 재개 → 완료 결과 읽기·재사용 |
| P1. 매 배치 전체 population hash 재계산 | 입력 population을 고정한 다음 호출당 한 번 계산하여 cache key·progress·manifest에서 재사용 | batch 1/3/7 단위 테스트 및 실제 CUDA 2배치 추출 각각 hash 호출 1회. 중단 후 재개와 hash ledger 검증도 통과 |
| P2. 보존 옵션이 SurvFace 이외 상세 결과에 적용되지 않음 | TinyFace/SCFace probe별 rank·score·조건, LFW/RFW pair별 score·threshold·판단·양쪽 ID 저장. 상세 파일을 job 재개 기록과 완료 manifest의 hash 검증에 포함 | rank 및 pair 평가의 True/False 검사. pair 상세의 true/false accept 수가 집계 지표와 일치 |
| P2. 준비 데이터셋 추가 시 handoff.json 불변 게시 충돌 | 최초 handoff.json을 보존하고 handoffs/<내용 SHA-256>.json으로 새 불변 목록 게시. 저장 설정 campaign.handoff_path에 정확한 새 경로 반영 | 실제 SCFace 역할 준비 후 TinyFace 추가 → 두 이전 snapshot 보존 → ronbun에서 새 목록·역할 hash 읽기. 기존 데이터셋 교체 및 변조 거부 테스트 |
| P2. export의 bilinear 정책을 ronbun이 무시 | 검증한 export metadata에서 resize-only 정책을 읽고 bicubic/bilinear 적용. cache key·progress·manifest에 기록. 누락·미지원 정책 거부 | 두 보간법의 픽셀 변환 검사, metadata 변조 거부, 실제 SCFace 이미지 4장씩 CUDA 추출 |

빈 표 처리는 정상적으로 결과가 없는 optional paired 표에 한정한다. 누락·변조된 결과를 정상으로 간주하지 않는다.
TinyFace/SCFace 상세의 rank는 검색한 Top-20 안에서의 genuine rank다. 빈 rank는 Top-20 밖 또는 genuine 부재를 뜻하며,
전체 gallery의 정확한 순위를 저장했다고 해석하면 안 된다.
LFW/RFW 상세는 group/fold별 파일로 저장하고 같은 fold의 모든 보정 방법과 압축 조건을 포함한다.
KEEP_RAW_RESULTS=False에서도 입력 원본·모델·입력 cache·codec·집계·최소 출처를 보존한다.

추가로 입력 재개 시 Windows memmap을 닫고 미완료 ledger 뒤쪽을 정리하며,
CSV를 읽을 때 SCFace의 원본 ID `001` 같은 앞자리 0을 유지하도록 보완했다.

## 검증 결과와 근거

| 검사 | 결과 | 범위와 해석 |
|---|---|---|
| ronbun 관련 회귀 테스트 | 230 passed, 실패·오류·skip 0 | 관련 19개 테스트 파일. 이번 수정 회귀 25개와 export 정책 검사 2개 포함. 저장소 전체 pytest 실행이라는 뜻은 아님 |
| FaceX-Zoo fine-tune/tests 전체 | 181 passed, 실패·오류·skip 0, warnings 9 | handoff 회귀 4개 포함 |
| producer/export/consumer 호환성 | MobileFaceNet·Resnet IR-50 × 4 heads 통과 | 합성 가중치. 형식·1024D·수치 계약 검사이며 학습 성능 검증 아님 |
| 분리 consumer 프로세스 | MobileFaceNet·Resnet IR-50 각각 통과 | export를 FaceX-Zoo 밖 임시 위치로 전달하고 새 ronbun Python 프로세스에서 F:/FaceX-Zoo 파일 접근을 차단한 상태로 1×1024 추론 |
| 실제 GPU 입력 추출 | bicubic·bilinear 각각 SCFace 4장, 1024D 출력 | GTX 1080 Ti, PyTorch 2.7.1+cu118. FR 가중치는 합성, 품질 모델은 공식 CR-FIQA(L). FaceX-Zoo Python 소스 접근 차단 |
| 실제 데이터 목록 확장 | SCFace 50명/800장 → TinyFace 2,570명/7,804장 추가 | 새 임시 namespace에서 감사. 기존 운영 handoff·원본 데이터 변경 없음 |
| 선택한 변경 파일 lint / 양쪽 git diff --check | 통과 | 기존 전체 저장소 스타일 규칙의 전수 검사는 아님 |

분리 consumer 프로세스 검사는 `test_facex_crossrepo.py`의 기존 8조합 테스트 안에 포함되어 있다.
따라서 별도 2개 테스트로 더해서 통과 수를 부풀리지 않는다. 선행 소규모 테스트 실행도 230/181에 중복 합산하지 않는다.
단위 테스트의 CUDA stub은 실제 GPU 검증과 구분한다.
hash 반복 제거는 호출 횟수로 확인했으며, 전체 SurvFace의 처리 시간이나 성능 향상을 측정하지 않았다.

근거 파일:

- 경량 검증 요약: `C:/ronbun/publish/support/facex_review_fixes_20261023/validation.json`
- ronbun JUnit: `C:/ronbun/tmp/facex_review_fixes_20261023/ronbun_tests.xml`
- FaceX-Zoo JUnit: `F:/FaceX-Zoo/fine-tune/data/research_v1/review_fixes_validation_20261023/facex_tests.xml`
- GPU 검증: `C:/ronbun/tmp/facex_review_fixes_20261023/gpu_validation.json`
- handoff 검증: `C:/ronbun/tmp/facex_review_fixes_20261023/handoff_validation.json`
- FaceX-Zoo 합성 export 생산 스크립트: `F:/FaceX-Zoo/fine-tune/data/research_v1/review_fixes_validation_20261023/export_fixture.py`
- 독립 ronbun consumer 검증 스크립트: `C:/ronbun/tmp/facex_review_fixes_20261023/validate_artifacts.py`

상세 점수·합성 checkpoint·JUnit 등의 진단 원본은 로컬에 두고, Git에는 이 보고서와 경량 검증 요약만 추가로 노출했다.
경량 요약은 원본을 ZIP으로 포장한 것이 아니라 통과 수·검증 범위·출처 hash를 기록한 것이다.

## 실행 시 적용 사항과 남은 범위

1. 두 저장소의 모듈이 바뀌었으므로 사용하는 노트북 커널을 재시작한다. 활성 순차 단계는 그대로 유지했다.
2. FaceX-Zoo에서 데이터셋을 추가한 경우 03/04의 protocol 출력 또는 저장 설정의 `campaign.handoff_path`를 확인하고,
   ronbun YAML의 `handoff`에 그 불변 경로를 명시한다. 자동 최신 선택은 하지 않는다.
3. FaceX-Zoo에서 완성한 정식 export와 sidecar를 ronbun에 등록한다. 기본 YAML의 `models: {}`는 그대로이며
   검증용 합성 모델을 정식 모델로 자동 등록하지 않았다. resize-only 정책이 누락된 오래된 export는 새 입력 workflow에서 거부된다.
4. 변경된 코드·전처리 계약은 새 입력/실험 UID를 만든다. 기존 완료 산출물을 덮어쓰지 않고 필요한 입력·평가를 새 UID에서 생성한다.
   과거 미보존 상세를 옵션만 바꿔 소급 복원할 수 없으며, 상세가 필요하면 해당 평가를 입력 cache에서 재계산해야 한다.

FR fine-tune 대상 SurvFace·TinyFace·SCFace, 평가 전용 LFW·RFW, 1024D 차원과 기존 데이터 분할은 이번 수정에서 바꾸지 않았다.
SurvFace의 주 비교는 같은 점수 공간·cohort에서 Global-safe 대비 Continuous-FIQA의 TPIR@20과 실제 FPIR다.
TinyFace/SCFace rank 실험과 LFW/RFW pair 실험의 지표 구분도 유지한다.
최적 분할 검증, 부모 학습 인물 overlap 감사, 정식 FR 학습, 전체 population 추출,
전체 인식·압축·Continuous-FIQA 성능 실험은 이번 수정 검증에 포함되지 않았다.

변경 경로는 ronbun의 `research/experiments/facex_inputs.py`, `facex_workflow.py`, 해당 테스트·안내·설정 주석과
FaceX-Zoo의 `fine-tune/modules/research_protocol.py`, `finetune_protocol.py`, 해당 테스트·안내다.
FaceX-Zoo upstream backbone/head와 원본 데이터, 기존 완료 결과를 수정하지 않았다.

실행·결과 해석 안내: `C:/ronbun/notebooks/WORKFLOW_GUIDE.md` 및 `F:/FaceX-Zoo/fine-tune/RESEARCH_HANDOFF.md`.
