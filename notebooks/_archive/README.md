# 📦 notebooks/_archive/

> **아카이브 일자**: 2026-09-30  
> **사유**: `common/orchestration/` 및 `calibration/` 노트북이 전체 파이프라인을 통합 자동화하면서, 데이터셋별 개별 노트북이 더 이상 독립 실행 필요 없음

## 아카이브 분류

### Thin Runbook (14개)
- `lfw/04_gradcam/`, `survface/04_gradcam/` 하위 전체
- 4KB 수준, `step4_workflow.py` 함수 1개 호출만 수행
- `00_batch_experiment_runner.ipynb`이 동일 함수를 직접 호출하여 완전 대체

### Step 1 레거시 (20개)
- `lfw/01_embeddings/`, `lfw/02_compression/`, `lfw/03_open_set/`
- `survface/01_embeddings/`, `survface/02_compression/`, `survface/03_open_set/`
- PostgreSQL + ONNX 기반 구 파이프라인, Step 4(PyTorch + 파일 기반)로 대체됨
- 과거 실험 결과가 출력으로 보존되어 있어 **논문 참고 시 활용 가능**

### 배치에 흡수된 프로토타입 (2개)
- `calibration/00_fiqa_conditioned_threshold_calibration.ipynb` → `01_batch`에 36조건 흡수
- `calibration/02_saliency_incremental_threshold_calibration.ipynb` → `01_batch`에 36조건 흡수

### 오케스트레이션에 통합된 공통 도구 (2개)
- `common/model_preparation/00_checkpoint_registration.ipynb` → `prepare_common_model_checkpoint()`
- `common/model_preparation/01_preprocessing_and_model_smoke.ipynb` → `prepare_common_model_checkpoint()`

### 미완성/유예 (1개)
- `balancedface/00_data_preparation/00_data_preparation.ipynb` — RecordIO 미구현

### 오케스트레이터에 통합된 개별 단계 (3개)
- `rfw/00_data_preparation/`, `rfw/01_embeddings/`, `rfw/02_compression/`
- `rfw/00_rfw_all_in_one.ipynb`에 모든 기능이 통합됨

## 복원 방법

```bash
# 특정 파일 복원
git mv notebooks/_archive/<path> notebooks/<path>

# 전체 복원
git mv notebooks/_archive/* notebooks/
```
