# 공통 실험 오케스트레이션

`00_batch_experiment_runner.ipynb`는 LFW와 SurvFace의 데이터셋별 legacy
노트북을 대체하거나 삭제하지 않는다. 검증된 `research/` Python 단계 함수를
공통 설정으로 순차 호출하는 선택적 상위 실행기다.

## 고정 실행 등급

| 등급 | LFW | SurvFace | 용도 |
|---|---:|---:|---|
| `quick` | 10% | 2% | 실제 데이터·모델 기반 흐름 및 경향 확인 |
| `full` | 100% | 100% | 전체 protocol 실행 |

fraction은 노트북에서 임의로 바꾸지 않는다. quick 표본은 identity-aware,
role-preserving 방식으로 선택되며 논문 최종 수치로 사용하지 않는다.

## 안전한 사용 순서

1. 첫 변수 셀에서 `DATASET_ID`와 `RUN_TIER`를 선택한다.
2. 기본 `EXECUTE=False` 상태로 전체 실행하여 plan과 preflight를 확인한다.
3. `ready_to_execute_pipeline=True`인지 확인한다.
4. 실제 장시간 실험을 시작할 때만 `EXECUTE=True`와
   `ACKNOWLEDGE_LOCAL_EXECUTION=True`를 설정한다.
5. 커널을 재시작하고 전체 실행한다.

완료된 phase는 건너뛰고 미완료 phase부터 재개한다. 같은 plan의 완료 run이
이미 있으면 새 run을 자동 생성하지 않는다. 의도적인 독립 반복일 때만
`START_NEW_RUN=True`를 사용한다.

## 현재 구현 경계

공통 dispatcher는 기존 Step 4의 aligned crop, landmark, origin embedding,
Grad-CAM, compression characterization, saliency join, 대표 사례 생성 단계를
지원한다.

다음 `evaluation_contract_v1` 항목은 아직 제안·구현 전이므로 현재 full 실행의
완료만으로 논문 최종 비교가 되지 않는다.

- PQ exhaustive ADC
- 선택 profile IVF-PQ 시스템 실험
- official/DB baseline 전체 행렬
- calibration 100/500/1,000명
- 전체 FPIR target 행렬
- 동일 commit과 동일 model UID의 LFW/SurvFace full 재실행

세부 단계의 디버깅이나 수동 복구는 기존 `notebooks/lfw` 또는
`notebooks/survface` 순차 노트북을 사용한다.
