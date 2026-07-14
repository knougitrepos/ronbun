# Dataset-specific experiment notebooks

노트북은 데이터셋별로 완전히 분리합니다. 공통 계산·DB·압축·검색 코드는
`research/`의 Python 모듈을 호출하며, 한 데이터셋의 노트북을 다른 데이터셋에
그대로 사용하지 않습니다.

## LFW

`notebooks/lfw/`의 `data_preparation.ipynb`와 00~05를 위에서 아래로
실행합니다. 새 run의 설정은 `configs/experiments/lfw_face_search.yaml`, 기록
위치는 `runs/lfw/`입니다. 분리 전에 생성한 진행 중 run은 기존
`runs/active_run.json`을 fallback으로 읽을 수 있습니다.

LFW 평가에는 `registered`, `known_unknown`, `unknown_unknown` 세 probe 유형을
유지합니다. PCA/PQ는 development에서만 학습하고 calibration cutoff는
calibration split에서 선택하며 test는 최종 평가에만 사용합니다.

## QMUL-SurvFace-v1

`notebooks/survface/`의 `data_preparation.ipynb`와 공식 프로토콜 전용 00~05를
실행합니다. 설정은 `configs/experiments/survface_face_search.yaml`, 기록 위치는
`runs/survface/`입니다.

공식 gallery, mated probe, unmated probe의 순서와 `protocol_index`를 바꾸지
않습니다. 공식 프로토콜에는 known unknown이 없습니다. Gallery identity마다
모든 성공 임베딩을 평균한 `official_all` template을 만들고, 추출 실패 수와
영향받은 identity를 원래 공식 분모와 함께 기록합니다.

SurvFace 공식 test로 PCA/PQ 또는 calibration을 학습하면 누수입니다. 02에서
development 데이터로 학습해 동결한 외부 run을 명시해야 하며, 준비되지 않았을
때는 이후 단계를 실행하지 않습니다.

## 재시작 원칙

중단 후 임의의 셀부터 실행하지 말고 커널을 재시작한 뒤 bootstrap/preflight
셀부터 다시 실행합니다. 입력 hash나 상위 단계 artifact checksum이 달라지면
새 run을 만들고 영향을 받는 단계부터 다시 수행합니다.
