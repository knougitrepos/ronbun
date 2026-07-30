# 연구 범위

- 얼굴 이미지의 ArcFace 임베딩을 압축한 상태로 PostgreSQL/pgvector에 저장하고 검색 성능과 저장 효율을 평가한다.
- 주요 압축 방법은 PCA와 Product Quantization이며, 1:N Open-set Face Identification/Verification에서 압축에 따른 점수·순위 변화와 threshold 보정을 연구한다.
- 실험은 Intel Core i5-10600K, GeForce GTX 1080 Ti, RAM 64GB에서 수행 가능한 범위를 우선한다.

# 실행 환경 및 하드웨어 가속 원칙

- 2026-07-27 현재 검증된 로컬 GPU 환경은 GeForce GTX 1080 Ti, PyTorch `2.7.1+cu118`, PyTorch CUDA runtime `11.8`이다. 프로젝트 의존성과 wheel은 우선 CUDA 11.8 및 GTX 1080 Ti(Pascal, compute capability 6.1) 호환성을 유지한다.
- CUDA 12 계열이나 검증되지 않은 PyTorch·ONNX Runtime 조합으로 임의 업그레이드하지 않는다. 변경이 필요하면 GTX 1080 Ti 지원, CUDA runtime, cuDNN 및 기존 checkpoint smoke test를 먼저 검증한다.
- PyTorch inference, embedding 추출, Grad-CAM처럼 CUDA를 지원하는 연산은 기본적으로 `cuda` 장치를 우선한다. 실행 전에 `torch.cuda.is_available()`과 실제 device name을 확인하고 run log 또는 manifest에 기록한다.
- ONNX Runtime과 InsightFace 작업은 `CUDAExecutionProvider`를 우선하고 `CPUExecutionProvider`를 명시적 fallback으로 둔다. `ctx_id=0`만으로 GPU 사용을 판단하지 말고 생성된 session의 실제 provider를 확인한다.
- 대규모 정식 실행에서 CUDA 사용을 기대했는데 CUDA provider가 적용되지 않으면 조용히 CPU 전체 실행으로 전환하지 않는다. 즉시 중단하거나 명확한 경고와 사용자 승인 후 CPU 실행한다.
- 얼굴 정렬처럼 일부 모듈만 필요한 작업은 InsightFace의 `allowed_modules`를 사용해 detection 등 필요한 모델만 로드한다. 불필요한 recognition, gender/age, landmark 모델을 함께 실행하지 않는다.
- GPU 가속을 적용해도 이미지 decode, SHA-256, CSV/JSON 기록, pandas 처리 등 CPU 작업은 남는다. GPU 사용률이 일정하지 않다는 이유만으로 가속 실패로 판단하지 말고 실제 provider와 단계별 시간을 확인한다.
- GTX 1080 Ti의 11GB VRAM 범위에서 batch size를 설정하고, 첫 실제 run에서 OOM 여부와 처리량을 측정한다. batch size 또는 provider 변경은 config와 실행 기록에 남긴다.
- Faiss, PostgreSQL/pgvector 등 현재 설치본이 CPU 전용인 구성요소를 GPU로 가장하지 않는다. GPU 구현이 실제 설치·검증된 경우에만 가속됐다고 기록한다.

# 논문 조사 원칙

- 최상위 학회와 Q1 저널을 우선하고 필요한 경우 Q2까지 확대한다.
- 컴퓨터 비전 분야에서는 IEEE TPAMI 등 신뢰도 높은 출처를 우선한다.
- arXiv와 OpenReview 자료는 참고할 수 있지만 동료심사 여부와 정식 출판 여부를 구분하여 기록한다.
- IEEE Access는 필요에 따라 참고하고, MDPI 등 평가가 엇갈리는 출판사는 우선순위를 낮춘다.
- novelty를 주장할 때는 `최초`라고 단정하지 말고 조사 범위와 기존 연구 대비 정확한 차이를 명시한다.

# Novelty 분석 기록 규칙

- 사용자가 연구 novelty, 관련 연구 중복, 차별성 또는 돌파 방안을 분석해 달라고 요청하면 결과를 저장소 루트의 `novelty` 폴더에 Markdown으로 기록한다.
- 파일명은 `1.md`, `2.md`, `3.md`처럼 양의 정수를 순차적으로 사용한다. 새 분석을 시작할 때 기존 숫자 파일 중 최댓값을 확인하고 다음 번호를 사용한다. 기존 파일을 덮어쓰지 않는다.
- 각 문서에는 작성일, 분석 대상 코드/실험 run, 핵심 결론, 직접 경쟁 논문, 중복되는 기여, 남은 차별점, 수학적 정의, 실험으로 검증할 항목을 포함한다.
- 논문은 정식 출판 여부와 venue를 확인하고 가능한 한 공식 논문 페이지나 DOI를 기록한다. 프리프린트만 존재하면 이를 명시한다.
- 새 분석에서는 이전 `novelty/*.md`를 먼저 읽고, 이전 문서에서 지적된 약점과 보강 항목이 현재 코드와 최신 실험에서 실제로 개선됐는지 확인한다.
- 개선 여부를 `개선됨`, `부분 개선`, `미개선`, `검증 불가` 중 하나로 판정하고 코드·설정·실험 artifact 근거를 함께 기록한다.
- 새로운 아이디어를 제안하는 것만으로 `개선됨`이라고 판정하지 않는다. 구현 또는 실험 결과로 확인되지 않은 내용은 `미구현` 또는 `검증 불가`로 구분한다.
- 새 novelty가 MRQ의 압축 오차 기반 reranking, KWS의 quantization 기반 score calibration, 기존 Neyman-Pearson/conformal threshold 또는 uncertainty rejection을 단순히 재명명한 것인지 반드시 점검한다.
- 현재 연구의 우선 차별화 방향은 압축 얼굴 임베딩 1:N open-set DB 검색에서 원본 점수·순위 불확실성, 그룹별 FPIR 보장, DIR 보존, accept/reject/exact-fallback 비용과 압축 프로파일을 공동 최적화하는 것이다.
- 분석 채팅 답변과 저장된 Markdown의 결론이 서로 다르지 않도록 하며, 변경된 경우 변경 이유와 새 근거를 문서에 남긴다.

# Architecture 변경 기록 규칙

- 연구 구조, 실험 단계, 데이터 흐름, 모델·DB 경계, 평가 프로토콜 또는 핵심 모듈 책임에 중요한 변경이 생기면 저장소 루트의 `architect` 폴더에 Markdown으로 기록한다.
- 파일명은 한국 표준시 기준 `YYYYMMDD.md` 형식을 사용한다. 같은 날짜의 파일이 이미 있으면 기존 내용을 지우지 않고 변경 시각과 제목을 가진 새 절을 뒤에 추가한다.
- architecture 작업을 시작하기 전에 최신 `architect/*.md`와 관련된 이전 날짜 문서를 읽어, 새 결정이 기존 결정의 유지·보완·폐기 중 무엇인지 확인한다.
- 각 기록에는 작성일, 상태(`제안`, `구현 중`, `구현됨`, `검증됨`), 변경 배경, 결정 내용, 현재 구현과 목표 구조의 차이, 데이터·모듈 흐름, 영향받는 코드·설정·노트북·DB·artifact, 재실행 범위, 검증 기준, 미해결 위험을 포함한다.
- 아이디어나 계획만 존재할 때는 `구현됨` 또는 `검증됨`으로 기록하지 않는다. 코드, 설정, 테스트 및 실험 artifact 근거를 구분해 남긴다.
- 중요한 방향이 바뀌면 이전 기록을 덮어쓰지 않고 새 날짜 문서에서 폐기 또는 변경된 결정과 그 이유를 명시한다.
- architecture 기록과 `THESIS_RESEARCH_PLAN.md`, `novelty/*.md`, 실제 코드가 충돌하면 충돌 사실을 숨기지 말고 어떤 문서가 현재 기준인지 명시한다.
