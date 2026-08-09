# 공통 실험 오케스트레이션

4개 checkpoint × 3개 open-set dataset은 checkpoint별 batch run으로 생성한다. 12개 완료
run이 준비되면 `CROSS_MODEL_RUN_MATRIX`에 각 run directory를 명시하여 공통 보고서의
통합 표를 만든다. 자동 latest 선택은 하지 않으며 일부 model/dataset만 채운 matrix는
fail-closed로 거부한다.

`00_batch_experiment_runner.ipynb`는 LFW, SurvFace, RFW-Custom의 데이터셋별
실행을 공통화하되 기존 LFW/SurvFace legacy
노트북을 대체하거나 삭제하지 않는다. 검증된 `research/` Python 단계 함수를
공통 설정으로 순차 호출하는 선택적 상위 실행기다.

## 실행 등급과 데이터 비율

| 등급 | LFW | SurvFace | RFW-Custom | 용도 |
|---|---:|---:|---:|---|
| `quick` | 10% | 2% | 10% | 실제 데이터·모델 기반 흐름 및 경향 확인 |
| `full` | 100% | 100% | 100% | 전체 protocol 실행 |

quick 비율은 노트북 첫 변수 셀의 다음 사전에서 명시적으로 설정한다.

```python
QUICK_DATA_FRACTIONS = {
    "lfw": 0.10,
    "survface": 0.02,
    "rfw_custom": 0.10,
}
```

변경한 quick 비율은 effective config와 plan에 기록된다. quick 표본은
identity-aware, role-preserving 방식으로 선택되며 논문 최종 수치로 사용하지
않는다. `full`은 이 사전의 값과 관계없이 항상 `data_fraction=1.0`이다.

## 모델과 가중치 선택

`MODEL_NAME`은 `arc`, `ada`, `mag`, `edge` 중 하나다. 노트북의
`MODEL_PROFILE_BY_NAME`과 `MODEL_WEIGHT_PATHS`에서 선택할 profile과 로컬
가중치 경로를 함께 설정한다. 한 plan은 한 dataset·tier·checkpoint만
담으므로 네 모델 비교는 별칭을 바꾸어 독립 run으로 각각 실행한다.

| 별칭 | 기본 profile | 기본 가중치 |
|---|---|---|
| `arc` | `arcface_ms1mv3_r100` | `models/arcface/ms1mv3_r100_backbone.pth` |
| `ada` | `adaface_ms1mv3_r100` | `models/adaface/adaface_ir101_ms1mv3.ckpt` |
| `mag` | `magface_ms1mv2_iresnet100` | `models/magface/magface_ms1mv2.pth` |
| `edge` | `edgeface_webface12m_xs_gamma_06` | `models/edgeface/edgeface_xs_gamma_06.pt` |

profile과 checkpoint는 같은 family, architecture, training dataset이어야 한다.
선택한 checkpoint의 SHA-256과 전처리 계약은 `model_uid`로 등록되고 effective
config에 정확히 고정된다. 같은 family에 여러 checkpoint가 등록되어도 이
UID와 경로가 일치하는 모델만 사용한다.

AdaFace의 `models/adaface/adaface_ir101_ms1mv2.ckpt`를 사용하려면 profile도
`adaface_ms1mv2_r100`으로 함께 바꿔야 한다. 다만 현재 이 bridge profile은
`run_gradcam=false`이므로 공통 Step 4 Grad-CAM 실행기에서는 fail-closed된다.

## 안전한 사용 순서

1. 첫 변수 셀에서 `DATASET_ID`, `RUN_TIER`, `QUICK_DATA_FRACTIONS`,
   `MODEL_NAME`, profile, 가중치 경로를 확인한다.
2. 선택 모델 등록 및 smoke test 결과에서 `model_uid`, checkpoint SHA-256,
   profile을 확인한다.
3. 실행 전 점검만 할 때는 `EXECUTE=False`로 전체 실행하여 plan과 preflight를
   확인한다.
4. `ready_to_execute_pipeline=True`인지 확인한다.
5. 실제 장시간 실험을 시작할 때만 `EXECUTE=True`와
   `ACKNOWLEDGE_LOCAL_EXECUTION=True`를 설정한다.
6. 커널을 재시작하고 전체 실행한다.

`quick`은 개발 중인 dirty working tree를 허용하지만 현재 local commit,
tracked diff hash, untracked-content hash를 plan과 run config에 고정한다.
plan 생성 뒤 source가 달라지면 실행을 중단한다. `full`은 논문 비교 lineage를
위해 clean local commit을 계속 요구한다. 이 검사는 GitHub나 원격 CI와
무관하다.

완료된 phase는 건너뛰고 미완료 phase부터 재개한다. 같은 plan의 완료 run이
이미 있으면 새 run을 자동 생성하지 않는다. 의도적인 독립 반복일 때만
`START_NEW_RUN=True`를 사용한다.

SurvFace Quick 공식 protocol은 `source_protocol_index`에 전체 protocol의
원래 역할별 순서를 보존하고, `protocol_index`에는 선택 부분집합 안의
0부터 연속된 역할별 순서를 기록한다. 2026-07-31에 확인된 phase 04 인덱스
공백 오류는 dataset·tier·설정·완료 phase·실패 메시지가 모두 일치할 때만
기존 phase 00~03을 보존하고 phase 04부터 제한적으로 재개한다. 코드 수정 뒤
기존 커널에서 마지막 셀만 다시 실행하지 말고 반드시 커널을 재시작한 뒤
첫 셀부터 전체 실행한다.

## 현재 구현 경계

공통 dispatcher는 기존 Step 4의 aligned crop, landmark, origin embedding,
Grad-CAM, compression characterization, saliency join, 대표 사례 생성 단계를
지원한다.

현재 Step 2에는 PCA direct/reconstruction, PQ reconstruction, exhaustive PQ
ADC가 구현되어 있다. 다만 이전 완료 run에는 새 검색 공간 행이 없으므로
`scripts/refresh_step4_search_spaces.py`의 SHA-검증 compact 파생 평가 또는
새 clean-source full run이 필요하다. 다음 항목은 여전히 검증 전이므로 현재
full 실행의 완료만으로 논문 최종 비교가 되지 않는다.

- Faiss IVF-PQ 및 PostgreSQL/pgvector IVFFlat
- ANN parameter sweep
- BalancedFace
- uncertainty/defer 및 고위험/저위험 query 실험
- calibration 100/500/1,000명
- 전체 FPIR target 행렬
- 동일 commit과 동일 model UID의 LFW/SurvFace full 재실행
- warm-up/repeat 기반 latency benchmark

RFW-Custom은 공식 RFW pair benchmark를 재명명한 것이 아니다. 동일 원천 영상을
identity-disjoint development/calibration/test와 1:N gallery/probe로 구성한 별도
custom protocol이며, official RFW는 계속 1:1 verification으로만 보고한다.

세부 단계의 디버깅이나 수동 복구는 기존 `notebooks/lfw`,
`notebooks/survface`, `notebooks/rfw` 순차 노트북을 사용한다.

## Cross-dataset calibration transfer

`cross_dataset_calibration_transfer.ipynb`는 calibration source와 평가 target을
독립적으로 선택한다. `DATASET_IDS`는 target 목록이며 source→target 방향별 결과를
평균내지 않는다. codec source, calibration source, physical dataset, score-space,
protocol과 overlap 상태가 manifest에서 모두 일치해야 한다. RFW official pair에
외부 calibration을 적용한 결과는 official 내부 9-fold benchmark가 아니라 별도
external-calibration diagnostic으로 기록한다.

## SurvFace Grad-CAM faithfulness 파생 평가

완료된 full SurvFace run의 heatmap과 frozen target template을 변경하지 않고
`scripts/derive_survface_faithfulness.py`로 high-saliency, low-saliency,
random occlusion 결과를 별도 생성한다. 표본은 protocol role, 모델별 raw-norm
사분위, 원본 target-score 사분위의 32개 층에서 2,048개를 결정적으로 선택한다.
ArcFace·AdaFace·MagFace 결과는
`scripts/summarize_survface_faithfulness_models.py`가 동일 조건인지 검증한 뒤 한
표로 결합한다. 이 평가는 threshold-independent이므로 SurvFace FPIR calibration
transfer 실패와 분리되지만, 현재 파생 evaluator가 dirty source이므로 논문 최종
수치로 승격하려면 clean commit에서 새 output version으로 재실행해야 한다.
