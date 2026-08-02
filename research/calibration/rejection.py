from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from research.search.open_set import CALIBRATION_FEATURE_COLUMNS


def _target(frame: pd.DataFrame) -> np.ndarray:
    if "y_true_accept" not in frame.columns:
        raise ValueError("y_true_accept column is required")
    return frame["y_true_accept"].astype(int).to_numpy()


def choose_threshold(
    scores: np.ndarray,
    is_mated: np.ndarray,
    top1_correct: np.ndarray,
    target_fpir: float,
) -> float:
    if not 0.0 <= target_fpir <= 1.0:
        raise ValueError("target_fpir must be between 0 and 1")
    scores = np.asarray(scores, dtype=float)
    is_mated = np.asarray(is_mated, dtype=bool)
    top1_correct = np.asarray(top1_correct, dtype=bool)
    if not (len(scores) == len(is_mated) == len(top1_correct)):
        raise ValueError("scores, is_mated, and top1_correct must have equal length")
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    best_threshold = float(np.inf)
    best_dir = -1.0
    for threshold in thresholds:
        accepted = scores >= threshold
        non_mated = ~is_mated
        fpir = float(np.mean(accepted[non_mated])) if non_mated.any() else 0.0
        if fpir > target_fpir:
            continue
        directory_identification_rate = (
            float(np.mean(accepted[is_mated] & top1_correct[is_mated]))
            if is_mated.any()
            else 0.0
        )
        if directory_identification_rate > best_dir:
            best_dir = directory_identification_rate
            best_threshold = float(threshold)
    return best_threshold


def choose_non_mated_fpir_threshold(
    scores: np.ndarray,
    is_mated: np.ndarray,
    target_fpir: float,
) -> float:
    """Choose the most permissive tie-preserving non-mated-only threshold."""

    if not 0.0 <= target_fpir <= 1.0:
        raise ValueError("target_fpir must be between 0 and 1")
    score_values = np.asarray(scores, dtype=float)
    mated_values = np.asarray(is_mated, dtype=bool)
    if score_values.ndim != 1 or mated_values.ndim != 1:
        raise ValueError("scores and is_mated must be one-dimensional")
    if len(score_values) != len(mated_values) or len(score_values) == 0:
        raise ValueError("scores and is_mated must have equal non-zero length")
    if not np.isfinite(score_values).all():
        raise ValueError("scores must be finite")
    non_mated_scores = score_values[~mated_values]
    if len(non_mated_scores) == 0:
        raise ValueError("non-mated scores are required")

    unique_scores, counts = np.unique(non_mated_scores, return_counts=True)
    descending_scores = unique_scores[::-1]
    accepted_counts = np.cumsum(counts[::-1])
    feasible = accepted_counts / len(non_mated_scores) <= float(target_fpir) + 1e-15
    if not feasible.any():
        return float(np.nextafter(descending_scores[0], np.inf))
    return float(descending_scores[np.flatnonzero(feasible)[-1]])


def _decision_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if "probe_type" not in frame.columns or "top1_correct" not in frame.columns:
        raise ValueError("probe_type and top1_correct columns are required")
    return (
        frame["probe_type"].astype(str).eq("registered").to_numpy(),
        frame["top1_correct"].astype(bool).to_numpy(),
    )


class GlobalThresholdCalibrator:
    def __init__(self, target_fpir: float = 0.01):
        self.target_fpir = target_fpir
        self.threshold: float | None = None

    def fit(self, frame: pd.DataFrame) -> "GlobalThresholdCalibrator":
        is_mated, top1_correct = _decision_labels(frame)
        self.threshold = choose_threshold(
            frame["top1_score"].astype(float).to_numpy(),
            is_mated,
            top1_correct,
            self.target_fpir,
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.threshold is None:
            raise ValueError("calibrator is not fit")
        return (frame["top1_score"].astype(float).to_numpy() >= self.threshold).astype(int)


class PerCompressionThresholdCalibrator:
    def __init__(self, target_fpir: float = 0.01):
        self.target_fpir = target_fpir
        self.thresholds: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame) -> "PerCompressionThresholdCalibrator":
        self.thresholds = {}
        for profile, group in frame.groupby("compression_profile"):
            is_mated, top1_correct = _decision_labels(group)
            self.thresholds[str(profile)] = choose_threshold(
                group["top1_score"].astype(float).to_numpy(),
                is_mated,
                top1_correct,
                self.target_fpir,
            )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.thresholds:
            raise ValueError("calibrator is not fit")
        predictions = []
        for _, row in frame.iterrows():
            threshold = self.thresholds[str(row["compression_profile"])]
            predictions.append(int(float(row["top1_score"]) >= threshold))
        return np.asarray(predictions, dtype=int)


class _ConstantProbabilityModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=np.float32)

    def predict(self, frame: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(frame) >= threshold).astype(int)


class _LearnedCalibrator:
    estimator_cls = LogisticRegression

    def __init__(self, **estimator_kwargs):
        self.estimator_kwargs = estimator_kwargs
        self.feature_columns = CALIBRATION_FEATURE_COLUMNS
        self.pipeline: Pipeline | _ConstantProbabilityModel | None = None

    def _build_pipeline(self) -> Pipeline:
        numeric = list(CALIBRATION_FEATURE_COLUMNS)
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric),
                ("profile", OneHotEncoder(handle_unknown="ignore"), ["compression_profile"]),
            ]
        )
        estimator = self.estimator_cls(**self.estimator_kwargs)
        return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])

    def fit(self, frame: pd.DataFrame) -> "_LearnedCalibrator":
        missing = set(CALIBRATION_FEATURE_COLUMNS + ["compression_profile"]).difference(frame.columns)
        if missing:
            raise ValueError(f"missing calibration columns: {sorted(missing)}")
        labels = _target(frame)
        if len(set(labels.tolist())) == 1:
            self.pipeline = _ConstantProbabilityModel(float(labels[0]))
            return self
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(frame[CALIBRATION_FEATURE_COLUMNS + ["compression_profile"]], labels)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ValueError("calibrator is not fit")
        if isinstance(self.pipeline, _ConstantProbabilityModel):
            return self.pipeline.predict_proba(frame)
        probabilities = self.pipeline.predict_proba(frame[CALIBRATION_FEATURE_COLUMNS + ["compression_profile"]])
        return probabilities[:, 1].astype(np.float32)

    def predict(self, frame: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(frame) >= threshold).astype(int)


class LogisticRegressionCalibrator(_LearnedCalibrator):
    estimator_cls = LogisticRegression

    def __init__(self, **estimator_kwargs):
        defaults = {"max_iter": 1000, "solver": "lbfgs"}
        defaults.update(estimator_kwargs)
        super().__init__(**defaults)


class ShallowMLPCalibrator(_LearnedCalibrator):
    estimator_cls = MLPClassifier

    def __init__(self, **estimator_kwargs):
        defaults = {
            "hidden_layer_sizes": (8,),
            "activation": "relu",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 0,
        }
        defaults.update(estimator_kwargs)
        super().__init__(**defaults)
