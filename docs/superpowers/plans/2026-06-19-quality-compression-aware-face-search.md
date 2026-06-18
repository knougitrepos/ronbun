# Quality and Compression-Aware Face Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible ArcFace template-search experiment that compares robust quality-aware aggregation, evaluates known and unknown probes, calibrates rejection under compression, and measures PostgreSQL/pgvector efficiency.

**Architecture:** Add a research pipeline beside the existing FastAPI application instead of coupling experiments to web routes. Pure NumPy/scikit-learn modules handle protocols, aggregation, compression, calibration, and statistics; SQLAlchemy models persist templates and run results; a CLI composes the modules. Every learned transform is fit only on the development split and serialized with split metadata.

**Tech Stack:** Python 3, NumPy, pandas, scikit-learn, PyTorch/ONNX Runtime for FIQA inference, InsightFace ArcFace, SQLAlchemy, PostgreSQL, pgvector, Faiss, pytest.

---

## Planned file structure

```text
research/
  __init__.py
  protocol.py                 # manifest schema, identity-disjoint splits, probe/gallery construction
  quality/
    __init__.py
    base.py                   # quality scorer protocol and score record
    rule_based.py             # blur, resolution, pose, detection score
    fiqa.py                   # frozen pretrained FIQA adapter
  templates/
    __init__.py
    aggregation.py            # medoid/MAD outlier removal and weighted pooling
  compression/
    __init__.py
    profiles.py               # original, PCA, low-precision and reconstruction metadata
  search/
    __init__.py
    open_set.py               # ranking, unknown rejection and metrics
  calibration/
    __init__.py
    rejection.py              # global, per-profile and logistic calibrators
  policy/
    __init__.py
    adaptive_compression.py   # constrained profile selection
  statistics.py               # paired bootstrap confidence intervals
experiments/
  run_face_search_study.py    # phase-based CLI
  configs/
    face_search.yaml          # frozen experiment configuration
scripts/
  build_face_manifest.py      # public dataset manifest generation
  extract_face_research_data.py
tests/
  research/
    test_protocol.py
    test_quality.py
    test_aggregation.py
    test_compression.py
    test_open_set.py
    test_rejection.py
    test_adaptive_compression.py
    test_statistics.py
  integration/
    test_face_template_repository.py
```

### Task 1: Establish the research package and test harness

**Files:**
- Modify: `requirements.txt`
- Create: `research/__init__.py`
- Create: `tests/research/test_protocol.py`
- Create: `research/protocol.py`

- [ ] **Step 1: Add test and experiment dependencies**

Add these dependencies once, removing the existing duplicate `faiss-cpu`, `matplotlib`, `scikit-learn`, and `seaborn` entries:

```text
pytest>=8.0
pyyaml>=6.0
onnxruntime-gpu>=1.17
```

Keep `faiss-cpu`, `scikit-learn`, `matplotlib`, and `seaborn` as single entries.

- [ ] **Step 2: Write the failing identity-leakage test**

```python
# tests/research/test_protocol.py
import pandas as pd
import pytest

from research.protocol import validate_identity_disjoint_splits


def test_rejects_identity_leakage_between_splits():
    manifest = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "identity_id": ["id-1", "id-1", "id-2"],
            "split": ["development", "test", "calibration"],
            "image_path": ["a.jpg", "b.jpg", "c.jpg"],
        }
    )

    with pytest.raises(ValueError, match="identity leakage"):
        validate_identity_disjoint_splits(manifest)
```

- [ ] **Step 3: Run the test to verify it fails**

Run:

```powershell
pytest tests/research/test_protocol.py::test_rejects_identity_leakage_between_splits -v
```

Expected: collection fails because `research.protocol` does not exist.

- [ ] **Step 4: Implement the minimal split validator**

```python
# research/protocol.py
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"image_id", "identity_id", "split", "image_path"}
ALLOWED_SPLITS = {"development", "calibration", "test"}


def validate_identity_disjoint_splits(manifest: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"missing manifest columns: {sorted(missing)}")

    unknown = set(manifest["split"]).difference(ALLOWED_SPLITS)
    if unknown:
        raise ValueError(f"unknown split names: {sorted(unknown)}")

    split_counts = manifest.groupby("identity_id")["split"].nunique()
    leaked = split_counts[split_counts > 1].index.tolist()
    if leaked:
        raise ValueError(f"identity leakage detected: {leaked[:10]}")
```

- [ ] **Step 5: Run the focused test and package import**

Run:

```powershell
pytest tests/research/test_protocol.py -v
python -c "import research.protocol"
```

Expected: one test passes and the import exits with code 0.

- [ ] **Step 6: Commit**

```powershell
git add requirements.txt research tests/research/test_protocol.py
git commit -m "test: establish face research protocol package"
```

### Task 2: Build deterministic gallery and known/unknown probe protocols

**Files:**
- Modify: `research/protocol.py`
- Modify: `tests/research/test_protocol.py`
- Create: `scripts/build_face_manifest.py`
- Create: `experiments/configs/face_search.yaml`

- [ ] **Step 1: Write tests for gallery and unknown identities**

```python
from research.protocol import build_open_set_protocol


def test_builds_gallery_known_and_unknown_probes_without_overlap():
    manifest = pd.DataFrame(
        {
            "image_id": ["a1", "a2", "b1", "b2", "u1", "u2"],
            "identity_id": ["a", "a", "b", "b", "u", "u"],
            "split": ["test"] * 6,
            "image_path": [f"{value}.jpg" for value in ["a1", "a2", "b1", "b2", "u1", "u2"]],
        }
    )

    protocol = build_open_set_protocol(
        manifest,
        gallery_identities=["a", "b"],
        enrollment_count=1,
        seed=7,
    )

    assert set(protocol.gallery["identity_id"]) == {"a", "b"}
    assert set(protocol.known_probes["identity_id"]) == {"a", "b"}
    assert set(protocol.unknown_probes["identity_id"]) == {"u"}
    assert set(protocol.gallery["image_id"]).isdisjoint(protocol.known_probes["image_id"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/research/test_protocol.py::test_builds_gallery_known_and_unknown_probes_without_overlap -v
```

Expected: FAIL because `build_open_set_protocol` is undefined.

- [ ] **Step 3: Implement the protocol object**

```python
# append to research/protocol.py
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class OpenSetProtocol:
    gallery: pd.DataFrame
    known_probes: pd.DataFrame
    unknown_probes: pd.DataFrame


def build_open_set_protocol(
    manifest: pd.DataFrame,
    gallery_identities: Sequence[str],
    enrollment_count: int,
    seed: int,
) -> OpenSetProtocol:
    if enrollment_count < 1:
        raise ValueError("enrollment_count must be positive")

    test_rows = manifest.loc[manifest["split"] == "test"].copy()
    known = test_rows.loc[test_rows["identity_id"].isin(gallery_identities)]
    unknown = test_rows.loc[~test_rows["identity_id"].isin(gallery_identities)]

    gallery = (
        known.groupby("identity_id", group_keys=False)
        .sample(n=enrollment_count, random_state=seed, replace=False)
        .sort_values(["identity_id", "image_id"])
    )
    known_probes = known.loc[~known["image_id"].isin(gallery["image_id"])]
    if known_probes.empty:
        raise ValueError("known probe set is empty after enrollment")

    return OpenSetProtocol(
        gallery=gallery.reset_index(drop=True),
        known_probes=known_probes.reset_index(drop=True),
        unknown_probes=unknown.reset_index(drop=True),
    )
```

- [ ] **Step 4: Add a frozen YAML configuration**

```yaml
# experiments/configs/face_search.yaml
seed: 20260619
dataset:
  manifest_path: artifacts/manifests/face_search.csv
  enrollment_counts: [1, 2, 5]
  mixed_enrollment: true
embedding:
  model: buffalo_l
  dimension: 512
quality:
  rule_temperature: 0.2
  fiqa_temperature: 0.2
aggregation:
  mad_multiplier: 3.0
compression:
  pca_dimensions: [256]
  pq:
    enabled: true
    subquantizers: 16
    bits: 8
open_set:
  target_fpirs: [0.001, 0.01]
bootstrap:
  samples: 2000
```

- [ ] **Step 5: Implement manifest generation as a deterministic CLI**

`scripts/build_face_manifest.py` must accept `--input-root`, `--output`, `--seed`, `--development-ratio`, and `--calibration-ratio`. It must derive `identity_id` from the immediate parent directory, assign identities rather than images to splits, write CSV, then call `validate_identity_disjoint_splits`.

- [ ] **Step 6: Run tests and a temporary manifest smoke test**

Run:

```powershell
pytest tests/research/test_protocol.py -v
python scripts/build_face_manifest.py --help
```

Expected: all protocol tests pass and the CLI prints usage.

- [ ] **Step 7: Commit**

```powershell
git add research/protocol.py tests/research/test_protocol.py scripts/build_face_manifest.py experiments/configs/face_search.yaml
git commit -m "feat: add identity-disjoint face search protocol"
```

### Task 3: Implement rule-based and frozen FIQA scoring

**Files:**
- Create: `research/quality/__init__.py`
- Create: `research/quality/base.py`
- Create: `research/quality/rule_based.py`
- Create: `research/quality/fiqa.py`
- Create: `tests/research/test_quality.py`
- Modify: `core/config.yaml`

- [ ] **Step 1: Write failing rule-quality tests**

```python
# tests/research/test_quality.py
import numpy as np

from research.quality.rule_based import RuleQualityScorer, RuleQualityStats


def test_rule_quality_prefers_sharp_frontal_high_resolution_face():
    scorer = RuleQualityScorer()
    good = RuleQualityStats(
        width=224,
        height=224,
        laplacian_variance=250.0,
        detection_score=0.99,
        yaw=2.0,
        pitch=1.0,
    )
    bad = RuleQualityStats(
        width=64,
        height=64,
        laplacian_variance=10.0,
        detection_score=0.60,
        yaw=35.0,
        pitch=25.0,
    )

    assert 0.0 <= scorer.score(good) <= 1.0
    assert scorer.score(good) > scorer.score(bad)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/research/test_quality.py -v
```

Expected: FAIL because the quality package does not exist.

- [ ] **Step 3: Implement a stable rule score**

```python
# research/quality/rule_based.py
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RuleQualityStats:
    width: int
    height: int
    laplacian_variance: float
    detection_score: float
    yaw: float
    pitch: float


class RuleQualityScorer:
    def score(self, stats: RuleQualityStats) -> float:
        resolution = min(math.sqrt(stats.width * stats.height) / 224.0, 1.0)
        sharpness = min(math.log1p(max(stats.laplacian_variance, 0.0)) / math.log1p(250.0), 1.0)
        detection = min(max(stats.detection_score, 0.0), 1.0)
        pose = math.exp(-(abs(stats.yaw) + abs(stats.pitch)) / 45.0)
        return float(0.25 * resolution + 0.30 * sharpness + 0.20 * detection + 0.25 * pose)
```

- [ ] **Step 4: Define the FIQA adapter contract**

```python
# research/quality/fiqa.py
from pathlib import Path
from collections.abc import Callable
import numpy as np


class FrozenFiqaScorer:
    def __init__(self, model_path: Path, infer: Callable[[Path, np.ndarray], float]):
        if not model_path.is_file():
            raise FileNotFoundError(f"FIQA model not found: {model_path}")
        self.model_path = model_path
        self._infer = infer

    def score(self, aligned_bgr: np.ndarray) -> float:
        value = float(self._infer(self.model_path, aligned_bgr))
        if not np.isfinite(value):
            raise ValueError("FIQA returned a non-finite score")
        return float(np.clip(value, 0.0, 1.0))
```

The production `infer` function must wrap one frozen public FIQA checkpoint. Record its paper, repository URL, license, SHA-256, preprocessing, and output normalization in `models/fiqa/MODEL_CARD.md`. Do not commit model weights.

- [ ] **Step 5: Add model configuration**

```yaml
# append under model in core/config.yaml
  fiqa_path: ./models/fiqa/model.onnx
  fiqa_sha256: ""
```

An empty checksum is allowed only in unit tests. The extraction CLI must require a non-empty checksum.

- [ ] **Step 6: Run quality tests**

Run:

```powershell
pytest tests/research/test_quality.py -v
```

Expected: all quality tests pass.

- [ ] **Step 7: Commit**

```powershell
git add research/quality tests/research/test_quality.py core/config.yaml
git commit -m "feat: add face quality scoring interfaces"
```

### Task 4: Implement robust template aggregation

**Files:**
- Create: `research/templates/__init__.py`
- Create: `research/templates/aggregation.py`
- Create: `tests/research/test_aggregation.py`

- [ ] **Step 1: Write failing aggregation tests**

```python
# tests/research/test_aggregation.py
import numpy as np

from research.templates.aggregation import aggregate_template


def test_removes_directional_outlier_and_normalizes_template():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, -0.02],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    qualities = np.array([0.8, 0.9, 0.7, 1.0], dtype=np.float32)

    result = aggregate_template(
        embeddings,
        qualities,
        remove_outliers=True,
        quality_weighted=True,
        mad_multiplier=3.0,
        temperature=0.2,
    )

    assert result.retained_indices.tolist() == [0, 1, 2]
    assert result.removed_indices.tolist() == [3]
    assert np.isclose(np.linalg.norm(result.embedding), 1.0)
    assert result.embedding[0] > 0.99
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/research/test_aggregation.py -v
```

Expected: FAIL because `aggregate_template` does not exist.

- [ ] **Step 3: Implement medoid/MAD filtering and weighted pooling**

```python
# research/templates/aggregation.py
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AggregationResult:
    embedding: np.ndarray
    retained_indices: np.ndarray
    removed_indices: np.ndarray
    dispersion: float
    mean_quality: float


def _l2_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("zero-norm embedding")
    return values / norms


def aggregate_template(
    embeddings: np.ndarray,
    qualities: np.ndarray,
    *,
    remove_outliers: bool,
    quality_weighted: bool,
    mad_multiplier: float,
    temperature: float,
) -> AggregationResult:
    vectors = _l2_rows(np.asarray(embeddings, dtype=np.float32))
    quality = np.asarray(qualities, dtype=np.float32)
    if len(vectors) != len(quality) or len(vectors) == 0:
        raise ValueError("embeddings and qualities must have equal non-zero length")

    distances = 1.0 - vectors @ vectors.T
    medoid = int(np.argmin(np.median(distances, axis=1)))
    medoid_distance = distances[medoid]
    retained = np.arange(len(vectors))

    if remove_outliers and len(vectors) >= 3:
        center = float(np.median(medoid_distance))
        mad = float(np.median(np.abs(medoid_distance - center)))
        limit = center + mad_multiplier * max(mad, 1e-6)
        retained = np.flatnonzero(medoid_distance <= limit)
        if medoid not in retained:
            retained = np.sort(np.append(retained, medoid))

    kept_vectors = vectors[retained]
    kept_quality = quality[retained]
    if quality_weighted and len(retained) > 1:
        logits = kept_quality / temperature
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
    else:
        weights = np.full(len(retained), 1.0 / len(retained), dtype=np.float32)

    pooled = np.sum(kept_vectors * weights[:, None], axis=0)
    pooled /= np.linalg.norm(pooled)
    removed = np.setdiff1d(np.arange(len(vectors)), retained)
    dispersion = float(np.mean(1.0 - kept_vectors @ pooled))
    return AggregationResult(
        embedding=pooled.astype(np.float32),
        retained_indices=retained,
        removed_indices=removed,
        dispersion=dispersion,
        mean_quality=float(kept_quality.mean()),
    )
```

- [ ] **Step 4: Run aggregation tests**

Run:

```powershell
pytest tests/research/test_aggregation.py -v
```

Expected: all aggregation tests pass.

- [ ] **Step 5: Commit**

```powershell
git add research/templates tests/research/test_aggregation.py
git commit -m "feat: add robust quality-aware face template aggregation"
```

### Task 5: Add searchable compression profiles and honest storage accounting

**Files:**
- Create: `research/compression/__init__.py`
- Create: `research/compression/profiles.py`
- Create: `tests/research/test_compression.py`
- Modify: `core/schemas.py`

- [ ] **Step 1: Write failing PCA profile tests**

```python
# tests/research/test_compression.py
import numpy as np

from research.compression.profiles import PcaProfile


def test_pca_profile_round_trip_reports_reconstruction_error():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(30, 8)).astype(np.float32)
    profile = PcaProfile(name="pca4", dimensions=4)
    profile.fit(train, split_name="development")

    compressed = profile.transform(train[:2])
    restored = profile.inverse_transform(compressed)

    assert compressed.vectors.shape == (2, 4)
    assert compressed.storage_bytes == 2 * 4 * 4
    assert restored.shape == (2, 8)
    assert compressed.reconstruction_error >= 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/research/test_compression.py -v
```

Expected: FAIL because the compression profile is missing.

- [ ] **Step 3: Implement the PCA profile**

```python
# research/compression/profiles.py
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class CompressedBatch:
    vectors: np.ndarray
    profile_name: str
    storage_bytes: int
    reconstruction_error: float


class PcaProfile:
    def __init__(self, name: str, dimensions: int):
        self.name = name
        self.dimensions = dimensions
        self.model = PCA(n_components=dimensions, random_state=0)
        self.fit_split = None

    def fit(self, values: np.ndarray, split_name: str) -> None:
        if split_name != "development":
            raise ValueError("compression models must be fit on development split")
        self.model.fit(values)
        self.fit_split = split_name

    def transform(self, values: np.ndarray) -> CompressedBatch:
        if self.fit_split != "development":
            raise RuntimeError("profile is not fitted")
        vectors = self.model.transform(values).astype(np.float32)
        restored = self.model.inverse_transform(vectors)
        error = float(np.mean((values - restored) ** 2))
        return CompressedBatch(
            vectors=vectors,
            profile_name=self.name,
            storage_bytes=int(vectors.nbytes),
            reconstruction_error=error,
        )

    def inverse_transform(self, values: CompressedBatch) -> np.ndarray:
        return self.model.inverse_transform(values.vectors).astype(np.float32)
```

- [ ] **Step 4: Add explicit face template tables**

Add SQLAlchemy models to `core/schemas.py`:

```python
class FaceTemplate(Base):
    __tablename__ = "face_templates"
    id = Column(Integer, primary_key=True)
    identity_id = Column(Text, nullable=False, index=True)
    aggregation_method = Column(String(64), nullable=False)
    compression_profile = Column(String(64), nullable=False)
    enrollment_count = Column(Integer, nullable=False)
    retained_count = Column(Integer, nullable=False)
    mean_quality = Column(Float, nullable=False)
    dispersion = Column(Float, nullable=False)
    reconstruction_error = Column(Float, nullable=False, default=0.0)
    embedding_512 = Column(Vector(512))
    embedding_256 = Column(Vector(256))
    parameters = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP)


class FaceSearchRun(Base):
    __tablename__ = "face_search_runs"
    id = Column(Integer, primary_key=True)
    run_name = Column(Text, nullable=False)
    config = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=False)
    artifact_path = Column(Text)
    created_at = Column(TIMESTAMP)
```

Do not store PQ codes in `FaceTemplate`. Keep PQ artifacts in the experiment artifact directory and record their path in run metadata.

- [ ] **Step 5: Run compression tests and schema import**

Run:

```powershell
pytest tests/research/test_compression.py -v
python -c "from core.schemas import FaceTemplate, FaceSearchRun"
```

Expected: tests pass and schema import exits 0.

- [ ] **Step 6: Commit**

```powershell
git add research/compression tests/research/test_compression.py core/schemas.py
git commit -m "feat: add face template compression profiles"
```

### Task 6: Implement known ranking and open-set metrics

**Files:**
- Create: `research/search/__init__.py`
- Create: `research/search/open_set.py`
- Create: `tests/research/test_open_set.py`

- [ ] **Step 1: Write failing open-set metric tests**

```python
# tests/research/test_open_set.py
import numpy as np

from research.search.open_set import evaluate_open_set


def test_open_set_metrics_count_unknown_false_accepts():
    scores = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.70, 0.60],
            [0.40, 0.30],
        ],
        dtype=np.float32,
    )
    probe_ids = np.array(["a", "b", "unknown-1", "unknown-2"])
    gallery_ids = np.array(["a", "b"])

    metrics = evaluate_open_set(scores, probe_ids, gallery_ids, threshold=0.65)

    assert metrics.rank1 == 1.0
    assert metrics.dir == 1.0
    assert metrics.fpir == 0.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/research/test_open_set.py -v
```

Expected: FAIL because `evaluate_open_set` is missing.

- [ ] **Step 3: Implement ranking and rejection metrics**

```python
# research/search/open_set.py
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OpenSetMetrics:
    rank1: float
    dir: float
    fpir: float
    fnir: float


def evaluate_open_set(
    scores: np.ndarray,
    probe_ids: np.ndarray,
    gallery_ids: np.ndarray,
    threshold: float,
) -> OpenSetMetrics:
    top_index = np.argmax(scores, axis=1)
    top_score = scores[np.arange(len(scores)), top_index]
    predicted = gallery_ids[top_index]
    known = np.isin(probe_ids, gallery_ids)
    accepted = top_score >= threshold
    correct = predicted == probe_ids

    rank1 = float(correct[known].mean())
    dir_value = float((accepted[known] & correct[known]).mean())
    fpir = float(accepted[~known].mean())
    fnir = 1.0 - dir_value
    return OpenSetMetrics(rank1=rank1, dir=dir_value, fpir=fpir, fnir=fnir)
```

- [ ] **Step 4: Add threshold selection**

Implement `threshold_at_target_fpir(unknown_top_scores, target_fpir)` using the conservative empirical quantile. Test that the selected threshold does not exceed the requested FPIR on calibration data.

- [ ] **Step 5: Run open-set tests**

Run:

```powershell
pytest tests/research/test_open_set.py -v
```

Expected: all open-set tests pass.

- [ ] **Step 6: Commit**

```powershell
git add research/search tests/research/test_open_set.py
git commit -m "feat: add open-set face search evaluation"
```

### Task 7: Implement compression-aware unknown rejection calibration

**Files:**
- Create: `research/calibration/__init__.py`
- Create: `research/calibration/rejection.py`
- Create: `tests/research/test_rejection.py`

- [ ] **Step 1: Write failing calibration tests**

```python
# tests/research/test_rejection.py
import numpy as np

from research.calibration.rejection import RejectionCalibrator


def test_calibrator_outputs_registration_probabilities():
    features = np.array(
        [
            [0.90, 0.20, 0.9, 0.8, 0.01, 5.0, 0.0],
            [0.82, 0.12, 0.7, 0.7, 0.03, 2.0, 0.1],
            [0.45, 0.01, 0.3, 0.4, 0.12, 1.0, 0.3],
            [0.40, 0.02, 0.2, 0.5, 0.15, 1.0, 0.4],
        ],
        dtype=np.float64,
    )
    labels = np.array([1, 1, 0, 0])
    calibrator = RejectionCalibrator().fit(features, labels, split_name="development")

    probabilities = calibrator.predict_proba(features)

    assert probabilities.shape == (4,)
    assert probabilities[:2].mean() > probabilities[2:].mean()
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/research/test_rejection.py -v
```

Expected: FAIL because `RejectionCalibrator` is missing.

- [ ] **Step 3: Implement standardized logistic calibration**

```python
# research/calibration/rejection.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RejectionCalibrator:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=0,
                    ),
                ),
            ]
        )
        self.fit_split = None

    def fit(self, features: np.ndarray, labels: np.ndarray, split_name: str):
        if split_name != "development":
            raise ValueError("calibrator must be fitted on development split")
        self.pipeline.fit(features, labels)
        self.fit_split = split_name
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.fit_split != "development":
            raise RuntimeError("calibrator is not fitted")
        return self.pipeline.predict_proba(features)[:, 1]
```

- [ ] **Step 4: Define and freeze feature order**

Add this constant and use it whenever feature matrices are built:

```python
CALIBRATION_FEATURES = (
    "top1_similarity",
    "top1_top2_margin",
    "probe_quality",
    "template_quality",
    "template_dispersion",
    "enrollment_count",
    "reconstruction_error",
)
```

Compression profile is represented by one-hot columns appended in sorted profile-name order. Serialize both ordered lists beside the model.

- [ ] **Step 5: Add calibration metrics**

Implement Brier score using `sklearn.metrics.brier_score_loss` and ECE with 10 fixed-width bins. Add tests for a perfectly calibrated synthetic case and an overconfident case.

- [ ] **Step 6: Run calibration tests**

Run:

```powershell
pytest tests/research/test_rejection.py -v
```

Expected: all rejection tests pass.

- [ ] **Step 7: Commit**

```powershell
git add research/calibration tests/research/test_rejection.py
git commit -m "feat: calibrate unknown rejection under compression"
```

### Task 8: Implement constrained adaptive compression

**Files:**
- Create: `research/policy/__init__.py`
- Create: `research/policy/adaptive_compression.py`
- Create: `tests/research/test_adaptive_compression.py`

- [ ] **Step 1: Write failing policy tests**

```python
# tests/research/test_adaptive_compression.py
from research.policy.adaptive_compression import CompressionPolicy


def test_policy_preserves_low_quality_template_and_compresses_stable_template():
    policy = CompressionPolicy(
        high_quality_threshold=0.8,
        low_dispersion_threshold=0.03,
        minimum_enrollment_for_strong_compression=2,
    )

    assert policy.choose(quality=0.9, dispersion=0.01, enrollment_count=5) == "strong"
    assert policy.choose(quality=0.5, dispersion=0.01, enrollment_count=5) == "pca256"
    assert policy.choose(quality=0.9, dispersion=0.20, enrollment_count=5) == "original"
    assert policy.choose(quality=0.9, dispersion=0.01, enrollment_count=1) == "original"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/research/test_adaptive_compression.py -v
```

Expected: FAIL because the policy module is missing.

- [ ] **Step 3: Implement the deterministic policy**

```python
# research/policy/adaptive_compression.py
from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionPolicy:
    high_quality_threshold: float
    low_dispersion_threshold: float
    minimum_enrollment_for_strong_compression: int

    def choose(self, quality: float, dispersion: float, enrollment_count: int) -> str:
        if enrollment_count < self.minimum_enrollment_for_strong_compression:
            return "original"
        if dispersion > self.low_dispersion_threshold:
            return "original"
        if quality >= self.high_quality_threshold:
            return "strong"
        return "pca256"
```

- [ ] **Step 4: Add constrained grid selection**

Implement `select_policy(candidates, development_results, max_rank1_loss, max_dir_loss)`. Filter candidates violating either accuracy constraint, then select the candidate with minimum mean bytes. Raise `ValueError` if none satisfy the constraints.

- [ ] **Step 5: Run policy tests**

Run:

```powershell
pytest tests/research/test_adaptive_compression.py -v
```

Expected: all policy tests pass.

- [ ] **Step 6: Commit**

```powershell
git add research/policy tests/research/test_adaptive_compression.py
git commit -m "feat: add quality-adaptive compression policy"
```

### Task 9: Add repository integration and pgvector benchmarks

**Files:**
- Modify: `core/database.py`
- Create: `tests/integration/test_face_template_repository.py`
- Create: `research/search/pgvector_search.py`

- [ ] **Step 1: Write a repository integration test**

The test must:

1. skip when `TEST_DATABASE_URL` is absent;
2. create tables in an isolated test schema;
3. insert two `FaceTemplate` rows with normalized vectors;
4. run exact cosine Top-1;
5. assert the expected identity;
6. remove the test schema in `finally`.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/integration/test_face_template_repository.py -v
```

Expected: SKIP without `TEST_DATABASE_URL`, or FAIL with it because repository methods are missing.

- [ ] **Step 3: Add repository methods**

Add methods to `VectorRepository`:

```python
def add_face_template(self, **values):
    row = FaceTemplate(**values)
    self.session.add(row)
    self.session.commit()
    self.session.refresh(row)
    return row


def search_face_templates_256(self, query_vector, limit=10):
    distance = FaceTemplate.embedding_256.cosine_distance(query_vector)
    return (
        self.session.query(FaceTemplate, distance.label("distance"))
        .filter(FaceTemplate.embedding_256.isnot(None))
        .order_by(distance)
        .limit(limit)
        .all()
    )
```

Add an equivalent 512D method. Keep dimensions separate so PostgreSQL can use correctly typed HNSW indexes.

- [ ] **Step 4: Add exact/HNSW timing utility**

`research/search/pgvector_search.py` must warm up each query set, execute at least 30 measured repetitions, use `time.perf_counter_ns`, and report median, P95, exact-result overlap, PostgreSQL table bytes, and index bytes.

- [ ] **Step 5: Run integration and unit tests**

Run:

```powershell
pytest tests/research -v
pytest tests/integration/test_face_template_repository.py -v
```

Expected: all unit tests pass; integration passes with a configured DB or skips for the documented reason.

- [ ] **Step 6: Commit**

```powershell
git add core/database.py research/search/pgvector_search.py tests/integration/test_face_template_repository.py
git commit -m "feat: persist and benchmark compressed face templates"
```

### Task 10: Build the phase-based experiment runner and statistical reports

**Files:**
- Create: `research/statistics.py`
- Create: `tests/research/test_statistics.py`
- Create: `experiments/run_face_search_study.py`
- Create: `scripts/extract_face_research_data.py`
- Modify: `README.md`

- [ ] **Step 1: Write a failing paired-bootstrap test**

```python
# tests/research/test_statistics.py
import numpy as np

from research.statistics import paired_bootstrap_difference


def test_paired_bootstrap_is_reproducible():
    baseline = np.array([0, 1, 0, 1, 0, 1], dtype=np.float64)
    proposed = np.array([1, 1, 1, 1, 0, 1], dtype=np.float64)

    first = paired_bootstrap_difference(baseline, proposed, samples=500, seed=9)
    second = paired_bootstrap_difference(baseline, proposed, samples=500, seed=9)

    assert first == second
    assert first.point_estimate > 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
pytest tests/research/test_statistics.py -v
```

Expected: FAIL because the statistics module is missing.

- [ ] **Step 3: Implement paired bootstrap**

```python
# research/statistics.py
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    point_estimate: float
    lower: float
    upper: float


def paired_bootstrap_difference(
    baseline: np.ndarray,
    proposed: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> ConfidenceInterval:
    if baseline.shape != proposed.shape:
        raise ValueError("paired samples must have identical shape")
    rng = np.random.default_rng(seed)
    differences = proposed - baseline
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    draws = differences[indices].mean(axis=1)
    return ConfidenceInterval(
        point_estimate=float(differences.mean()),
        lower=float(np.quantile(draws, 0.025)),
        upper=float(np.quantile(draws, 0.975)),
    )
```

- [ ] **Step 4: Implement the experiment CLI**

The CLI must expose these commands:

```powershell
python experiments/run_face_search_study.py validate --config experiments/configs/face_search.yaml
python experiments/run_face_search_study.py aggregate --config experiments/configs/face_search.yaml
python experiments/run_face_search_study.py compress --config experiments/configs/face_search.yaml
python experiments/run_face_search_study.py calibrate --config experiments/configs/face_search.yaml
python experiments/run_face_search_study.py evaluate --config experiments/configs/face_search.yaml
```

Each command writes immutable artifacts under:

```text
artifacts/face_search/<config-hash>/<phase>/
```

Every phase metadata file must contain Git commit, config SHA-256, dataset manifest SHA-256, split name, input artifact hashes, start/end timestamps, and random seed.

- [ ] **Step 5: Implement extraction CLI**

`scripts/extract_face_research_data.py` must:

1. load the manifest;
2. validate split isolation;
3. extract ArcFace embeddings;
4. compute rule and FIQA scores;
5. write Parquet shards;
6. persist face detector confidence and landmarks;
7. resume by image ID without duplicating rows;
8. record failed images separately.

- [ ] **Step 6: Document execution order**

Add a README section with the five experiment commands, PostgreSQL prerequisites, FIQA model-card requirement, and the rule that Test results cannot be inspected while selecting hyperparameters.

- [ ] **Step 7: Run full verification**

Run:

```powershell
pytest tests/research -v
python -m compileall research experiments scripts
python experiments/run_face_search_study.py --help
git diff --check
```

Expected:

- all research tests pass;
- compileall exits 0;
- CLI help lists all five phases;
- `git diff --check` produces no output.

- [ ] **Step 8: Remove verification caches**

Remove only generated `__pycache__` directories under `research`, `experiments`, `scripts`, and `tests`, after verifying each resolved path stays inside `D:\ronbun`.

- [ ] **Step 9: Commit**

```powershell
git add research experiments scripts tests README.md
git commit -m "feat: add reproducible face search study runner"
```

## Final research validation checklist

- [ ] Development, calibration, and test identities are disjoint.
- [ ] PCA/PQ and normalization parameters are fit only on development identities.
- [ ] Unknown probe identities never appear in the gallery.
- [ ] Global threshold, per-profile threshold, and proposed calibration use the same probe scores.
- [ ] Thresholds are selected on calibration and frozen for Test.
- [ ] PQ code bytes and codebook bytes are reported separately from reconstructed pgvector storage.
- [ ] Exact and HNSW results use identical distance definitions.
- [ ] All main metric differences include identity-level paired bootstrap intervals.
- [ ] Failed detections and excluded outliers are reported rather than silently dropped.
- [ ] Configuration, dataset manifest, model checksum, and Git commit are recorded for every reported table.
