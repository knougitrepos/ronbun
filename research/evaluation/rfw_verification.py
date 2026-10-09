from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from research.datasets.rfw import (
    RFW_GROUPS,
    RFW_OFFICIAL_FOLD_COUNT,
    RFW_OFFICIAL_GENUINE_PER_FOLD,
)
from research.evaluation.metrics import wilson_score_interval



_PAIR_COLUMNS = {
    "pair_id",
    "rfw_group",
    "fold_index",
    "left_image_id",
    "right_image_id",
    "is_genuine",
}


@dataclass(frozen=True)
class RFWVerificationResult:
    """Official-style RFW 9-fold calibration and held-out evaluation."""

    pair_scores: pd.DataFrame
    fold_metrics: pd.DataFrame
    group_summary: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class EmpiricalEER:
    """Discrete empirical EER estimate for one held-out score population."""

    eer: float
    threshold: float
    far: float
    frr: float
    absolute_far_frr_gap: float
    method: str = "heldout_scores_minimum_absolute_far_frr_v1"


def _validate_pair_structure(
    pairs: pd.DataFrame,
    *,
    strict_official: bool,
) -> None:
    missing = sorted(_PAIR_COLUMNS - set(pairs.columns))
    if missing:
        raise ValueError(f"RFW pairs are missing required columns: {missing}")
    if pairs.empty:
        raise ValueError("RFW pairs must be non-empty")
    if strict_official:
        if set(pairs["rfw_group"].astype(str)) != set(RFW_GROUPS):
            raise ValueError("strict RFW evaluation requires all four official groups")
        counts = pairs.groupby(
            ["rfw_group", "fold_index", "is_genuine"]
        ).size()
        expected_index = pd.MultiIndex.from_product(
            [RFW_GROUPS, range(RFW_OFFICIAL_FOLD_COUNT), [False, True]],
            names=["rfw_group", "fold_index", "is_genuine"],
        )
        expected = pd.Series(
            RFW_OFFICIAL_GENUINE_PER_FOLD,
            index=expected_index,
            dtype=np.int64,
        )
        if not counts.reindex(expected_index, fill_value=0).equals(expected):
            raise ValueError(
                "strict RFW evaluation requires 300 genuine and 300 impostor "
                "pairs per group/fold"
            )


def _validate_inputs(
    pairs: pd.DataFrame,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    *,
    strict_official: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    _validate_pair_structure(pairs, strict_official=strict_official)
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("RFW embeddings must be a non-empty 2D matrix")
    if len(image_ids) != len(matrix):
        raise ValueError("image_ids and embeddings must have equal length")
    resolved_ids = [str(value) for value in image_ids]
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError("RFW image_ids must be unique")
    if not np.isfinite(matrix).all():
        raise ValueError("RFW embeddings must be finite")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("RFW embeddings must have positive L2 norm")
    lookup = {image_id: index for index, image_id in enumerate(resolved_ids)}
    referenced = set(pairs["left_image_id"].astype(str)).union(
        pairs["right_image_id"].astype(str)
    )
    absent = sorted(referenced - set(lookup))
    if absent:
        raise ValueError(
            f"RFW pair images are missing embeddings: {absent[:3]}"
        )
    return matrix / norms[:, None], lookup


def _accuracy(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    return float(np.mean((scores >= threshold) == labels))


def empirical_equal_error_rate(
    scores: Sequence[float],
    labels: Sequence[bool],
) -> EmpiricalEER:
    """Estimate EER from held-out scores without using calibration folds.

    Candidate thresholds preserve observed-score ties and include reject-all and
    accept-all sentinels. The selected point minimizes ``abs(FAR - FRR)``; EER
    is the mean of FAR and FRR at that empirical operating point.
    """

    values = np.asarray(scores, dtype=np.float64)
    genuine = np.asarray(labels, dtype=bool)
    if values.ndim != 1 or genuine.ndim != 1 or len(values) != len(genuine):
        raise ValueError("EER scores and labels must be aligned 1D vectors")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("EER scores must be finite and non-empty")
    impostor = ~genuine
    if not genuine.any() or not impostor.any():
        raise ValueError("EER requires genuine and impostor scores")
    unique = np.unique(values)[::-1]
    thresholds = np.r_[
        np.nextafter(unique[0], np.inf),
        unique,
        np.nextafter(unique[-1], -np.inf),
    ]
    accepted = values[:, None] >= thresholds[None, :]
    fars = np.mean(accepted[impostor], axis=0)
    frrs = np.mean(~accepted[genuine], axis=0)
    gaps = np.abs(fars - frrs)
    index = int(np.argmin(gaps))
    return EmpiricalEER(
        eer=float((fars[index] + frrs[index]) / 2.0),
        threshold=float(thresholds[index]),
        far=float(fars[index]),
        frr=float(frrs[index]),
        absolute_far_frr_gap=float(gaps[index]),
    )


def _select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    predictions = scores[:, None] >= thresholds[None, :]
    accuracies = np.mean(predictions == labels[:, None], axis=0)
    return float(thresholds[int(np.argmax(accuracies))])


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    seed: int,
    repeats: int,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) < 2:
        raise ValueError("RFW fold bootstrap requires at least two values")
    rng = np.random.default_rng(int(seed))
    indexes = rng.integers(0, len(data), size=(int(repeats), len(data)))
    means = np.mean(data[indexes], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _threshold_grid(
    thresholds: Sequence[float] | None,
    score_space: str = "cosine",
) -> np.ndarray:
    if thresholds is not None:
        grid = np.asarray(thresholds, dtype=np.float64)
    elif score_space in ("negative_squared_l2_adc", "negative_squared_l2"):
        grid = np.linspace(-4.0, 0.0, 4001, dtype=np.float64)
    else:
        grid = np.linspace(-1.0, 1.0, 4001, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
        raise ValueError("thresholds must be a finite 1D sequence with >=2 values")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("thresholds must be strictly increasing")
    return grid


def _evaluate_scored_pairs(
    scored: pd.DataFrame,
    *,
    score_column: str,
    thresholds: np.ndarray,
    score_space: str,
    bootstrap_seed: int,
    bootstrap_repeats: int,
) -> RFWVerificationResult:
    if int(bootstrap_repeats) < 100:
        raise ValueError("bootstrap_repeats must be at least 100")
    fold_rows: list[dict[str, Any]] = []
    for group, group_rows in scored.groupby("rfw_group", sort=True):
        folds = sorted(int(value) for value in group_rows["fold_index"].unique())
        if len(folds) < 2:
            raise ValueError(f"RFW group {group!r} requires at least two folds")
        for heldout_fold in folds:
            train = group_rows.loc[group_rows["fold_index"] != heldout_fold]
            test = group_rows.loc[group_rows["fold_index"] == heldout_fold]
            if train.empty or test.empty:
                raise ValueError(
                    "RFW train and held-out fold partitions must be non-empty"
                )
            train_scores = train[score_column].to_numpy(dtype=np.float64)
            train_labels = train["is_genuine"].to_numpy(dtype=bool)
            test_scores = test[score_column].to_numpy(dtype=np.float64)
            test_labels = test["is_genuine"].to_numpy(dtype=bool)
            threshold = _select_threshold(train_scores, train_labels, thresholds)
            accepted = test_scores >= threshold
            genuine = test_labels
            impostor = ~genuine
            if not genuine.any() or not impostor.any():
                raise ValueError(
                    "each RFW held-out fold must contain genuine and impostor pairs"
                )
            eer = empirical_equal_error_rate(test_scores, test_labels)
            fold_rows.append(
                {
                    "rfw_group": str(group),
                    "heldout_fold": heldout_fold,
                    "threshold": threshold,
                    "train_pair_count": int(len(train)),
                    "test_pair_count": int(len(test)),
                    "accuracy": _accuracy(test_scores, test_labels, threshold),
                    "tar": float(np.mean(accepted[genuine])),
                    "far": float(np.mean(accepted[impostor])),
                    "eer": eer.eer,
                    "eer_threshold": eer.threshold,
                    "eer_far": eer.far,
                    "eer_frr": eer.frr,
                    "eer_absolute_far_frr_gap": eer.absolute_far_frr_gap,
                    "eer_threshold_source": "heldout_fold_scores_and_labels",
                    "eer_method": eer.method,
                }
            )

    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["rfw_group", "heldout_fold"]
    ).reset_index(drop=True)
    group_rows: list[dict[str, Any]] = []
    for group_index, (group, values) in enumerate(
        fold_metrics.groupby("rfw_group", sort=True)
    ):
        row: dict[str, Any] = {
            "rfw_group": str(group),
            "fold_count": int(len(values)),
            "eer_threshold_source": "heldout_fold_scores_and_labels",
            "eer_method": "heldout_scores_minimum_absolute_far_frr_v1",
        }
        for metric in ("accuracy", "tar", "far", "eer"):
            metric_values = values[metric].to_numpy(dtype=np.float64)
            low, high = _bootstrap_mean_ci(
                metric_values,
                seed=int(bootstrap_seed) + group_index * 17 + len(metric),
                repeats=bootstrap_repeats,
            )
            row[f"mean_{metric}"] = float(np.mean(metric_values))
            row[f"std_{metric}"] = float(np.std(metric_values, ddof=0))
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        group_rows.append(row)
    group_summary = pd.DataFrame.from_records(group_rows)
    group_accuracies = group_summary["mean_accuracy"].to_numpy(dtype=np.float64)
    group_eers = group_summary["mean_eer"].to_numpy(dtype=np.float64)
    summary = {
        "protocol": "rfw_official_groupwise_10fold_verification",
        "threshold_policy": "other_9_folds_when_strict_official",
        "eer_threshold_policy": "heldout_scores_minimum_absolute_far_frr_v1",
        "eer_threshold_source": "heldout_fold_scores_and_labels",
        "eer_uses_internal_9fold_threshold": False,
        "score_space": str(score_space),
        "open_set_protocol": False,
        "codec_fit_on_rfw": False,
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_repeats": int(bootstrap_repeats),
        "pair_count": int(len(scored)),
        "image_count": int(
            len(set(scored["left_image_id"]).union(scored["right_image_id"]))
        ),
        "macro_group_accuracy": float(np.mean(group_accuracies)),
        "macro_group_eer": float(np.mean(group_eers)),
        "group_accuracy_gap": float(
            np.max(group_accuracies) - np.min(group_accuracies)
        ),
        "group_eer_gap": float(np.max(group_eers) - np.min(group_eers)),
    }
    return RFWVerificationResult(
        pair_scores=scored,
        fold_metrics=fold_metrics,
        group_summary=group_summary,
        summary=summary,
    )


def evaluate_rfw_pair_scores(
    pairs: pd.DataFrame,
    *,
    scores: Sequence[float],
    score_space: str,
    thresholds: Sequence[float],
    strict_official: bool = True,
    bootstrap_seed: int = 8972,
    bootstrap_repeats: int = 2000,
) -> RFWVerificationResult:
    """Evaluate externally computed pair scores with fold-isolated thresholds."""

    _validate_pair_structure(pairs, strict_official=strict_official)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(pairs),) or not np.isfinite(values).all():
        raise ValueError("RFW pair scores must be a finite vector aligned to pairs")
    if int(bootstrap_repeats) < 100:
        raise ValueError("bootstrap_repeats must be at least 100")
    scored = pairs.copy().reset_index(drop=True)
    scored["pair_score"] = values
    return _evaluate_scored_pairs(
        scored,
        score_column="pair_score",
        thresholds=_threshold_grid(thresholds, score_space=score_space),
        score_space=score_space,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repeats=int(bootstrap_repeats),
    )


def evaluate_rfw_10fold(
    pairs: pd.DataFrame,
    *,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    codec: Any = None,
    thresholds: Sequence[float] | None = None,
    strict_official: bool = True,
    bootstrap_seed: int = 8972,
    bootstrap_repeats: int = 2000,
) -> RFWVerificationResult:
    """Evaluate 1:1 verification without fitting a codec on RFW.

    For each demographic group and held-out fold, the threshold is selected
    from the other folds only. If codec is provided, asymmetric ADC is used;
    otherwise cosine similarity is evaluated.
    """
    if codec is None:
        normalized, lookup = _validate_inputs(
            pairs,
            image_ids,
            embeddings,
            strict_official=strict_official,
        )
        scored = pairs.copy().reset_index(drop=True)
        left_indices = np.asarray(
            [lookup[str(value)] for value in scored["left_image_id"]], dtype=np.int64
        )
        right_indices = np.asarray(
            [lookup[str(value)] for value in scored["right_image_id"]], dtype=np.int64
        )
        scored["cosine_score"] = np.sum(
            normalized[left_indices] * normalized[right_indices], axis=1
        )
        return _evaluate_scored_pairs(
            scored,
            score_column="cosine_score",
            thresholds=_threshold_grid(thresholds, score_space="cosine"),
            score_space="cosine",
            bootstrap_seed=bootstrap_seed,
            bootstrap_repeats=int(bootstrap_repeats),
        )

    _validate_pair_structure(pairs, strict_official=strict_official)
    adc_scores = compute_rfw_pair_scores(
        pairs,
        image_ids=image_ids,
        embeddings=embeddings,
        codec=codec,
    )
    scored = pairs.copy().reset_index(drop=True)
    scored["adc_score"] = adc_scores
    return _evaluate_scored_pairs(
        scored,
        score_column="adc_score",
        thresholds=_threshold_grid(thresholds, score_space="negative_squared_l2_adc"),
        score_space="negative_squared_l2_adc",
        bootstrap_seed=bootstrap_seed,
        bootstrap_repeats=int(bootstrap_repeats),
    )


def compute_rfw_pair_scores(
    pairs: pd.DataFrame,
    *,
    image_ids: Sequence[str],
    embeddings: np.ndarray,
    codec: Any = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Compute pair scores: cosine similarity if codec is None, asymmetric ADC if codec is given.

    Left image is uncompressed query; right image is stored reference (8-bit PQ codes).
    """
    matrix = np.asarray(embeddings, dtype=np.float32)
    lookup = {str(img_id): i for i, img_id in enumerate(image_ids)}
    left_ids = [str(x) for x in pairs["left_image_id"]]
    right_ids = [str(x) for x in pairs["right_image_id"]]
    missing_left = [x for x in left_ids if x not in lookup]
    missing_right = [x for x in right_ids if x not in lookup]
    if missing_left or missing_right:
        raise ValueError(
            f"RFW pair images are missing embeddings: {(missing_left + missing_right)[:3]}"
        )
    left = np.asarray([lookup[x] for x in left_ids], dtype=np.int64)
    right = np.asarray([lookup[x] for x in right_ids], dtype=np.int64)
    if int(batch_size) < 1:
        raise ValueError("batch_size must be a positive integer")

    if codec is None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0.0) or not np.isfinite(norms).all():
            raise ValueError("cosine embeddings must be non-zero and finite")
        normalized = matrix / norms
        return np.einsum("ij,ij->i", normalized[left], normalized[right]).astype(np.float64)

    nbits = getattr(codec, "nbits", None)
    pq = getattr(codec, "pq", getattr(getattr(codec, "index", None), "pq", None))
    if nbits is None and pq is not None:
        nbits = getattr(pq, "nbits", None)
    if nbits != 8:
        raise ValueError("asymmetric ADC requires 8-bit subquantizers")
    if pq is None or not hasattr(pq, "compute_distance_tables"):
        raise ValueError("codec does not provide Faiss product quantizer compute_distance_tables")
    m = getattr(codec, "m", getattr(pq, "M", None))
    if m is None:
        raise ValueError("cannot determine number of subquantizers m from codec")
    import faiss

    result = []
    for start in range(0, len(pairs), batch_size):
        stop = min(len(pairs), start + batch_size)
        a = np.ascontiguousarray(matrix[left[start:stop]], dtype=np.float32)
        target = matrix[right[start:stop]]
        if hasattr(codec, "encode"):
            codes = codec.encode(target)
        elif hasattr(codec, "sa_encode"):
            codes = codec.sa_encode(np.ascontiguousarray(target))
        elif hasattr(getattr(codec, "index", None), "sa_encode"):
            codes = codec.index.sa_encode(np.ascontiguousarray(target))
        else:
            raise ValueError("codec must implement encode or sa_encode")
        tables = np.empty((len(a), m, 256), dtype=np.float32)
        pq.compute_distance_tables(len(a), faiss.swig_ptr(a), faiss.swig_ptr(tables))
        distances = np.take_along_axis(tables, codes[..., None].astype(np.int64), axis=2)
        result.extend(-distances[..., 0].sum(axis=1, dtype=np.float64))
    return np.asarray(result, dtype=np.float64)


def compute_rfw_pair_qualities(
    pairs: pd.DataFrame,
    fiqa_scores: Mapping[str, float] | pd.Series | np.ndarray,
    *,
    quality_mode: str = "symmetric_min",
) -> np.ndarray:
    """Compute pair qualities for RFW verification pairs.

    Quality modes:
    - "symmetric_min" (default for RFW): q_pair = min(q_left, q_right). In 1:1 verification,
      both images are arbitrary endpoints, and verification confidence is bounded by the
      lower-quality face image of the pair.
    - "query_left": q_pair = q_left. Query-only FIQA, matching the asymmetric database
      retrieval protocol in LFW (where left is the incoming probe and right is the stored reference).

    Distinction note:
    LFW evaluates query-only FIQA for asymmetric database retrieval simulation, whereas
    RFW defaults to symmetric pair quality min(q_left, q_right). These represent different
    application protocols and should not be conflated as identical settings.
    """
    if quality_mode not in ("symmetric_min", "query_left"):
        raise ValueError(
            f"unsupported quality_mode: {quality_mode!r}, must be 'symmetric_min' or 'query_left'"
        )
    if isinstance(fiqa_scores, (pd.Series, Mapping)):
        left_q = pairs["left_image_id"].astype(str).map(fiqa_scores).to_numpy(dtype=np.float64)
        right_q = pairs["right_image_id"].astype(str).map(fiqa_scores).to_numpy(dtype=np.float64)
        if not np.isfinite(left_q).all() or not np.isfinite(right_q).all():
            raise ValueError("missing or non-finite FIQA scores for RFW pair images")
        if quality_mode == "query_left":
            return left_q
        return np.minimum(left_q, right_q)
    fiqa_arr = np.asarray(fiqa_scores, dtype=np.float64)
    if len(fiqa_arr) != len(pairs):
        raise ValueError("fiqa_scores must be a mapping by image_id or an aligned array of pair qualities")
    if not np.isfinite(fiqa_arr).all():
        raise ValueError("missing or non-finite FIQA scores for RFW pair images")
    return fiqa_arr


@dataclass(frozen=True)
class RFWContinuousCalibrationResult:
    """Official 9-fold calibration + held-out fold evaluation result across demographic groups."""

    fold_metrics: pd.DataFrame
    group_summary: pd.DataFrame
    summary: dict[str, Any]
    models: dict[tuple[str, int, str, float], Any]
    raw_pair_evaluations: pd.DataFrame | None = None


def evaluate_rfw_continuous_fiqa_10fold(
    pairs: pd.DataFrame,
    *,
    scores: Sequence[float] | np.ndarray,
    pair_qualities: Sequence[float] | np.ndarray,
    target_fmrs: Sequence[float] = (0.001, 0.01, 0.05, 0.1),
    methods: Sequence[str] = ("global_safe", "fiqa_5bin", "continuous_fiqa"),
    score_space: str = "negative_squared_l2_adc",
    compression_profile: str = "pq_512_m128_b8",
    strict_official: bool = True,
    bootstrap_seed: int = 8972,
    bootstrap_repeats: int = 2000,
    safety_fraction: float = 0.3,
    knot_quantiles: tuple[float, float] = (1 / 3, 2 / 3),
    smoothing: float = 0.01,
    ridge: float = 0.001,
    max_iterations: int = 2000,
    margin_slope_cap: float = 0.95,
    return_raw_pairs: bool = False,
) -> RFWContinuousCalibrationResult:
    """Evaluate 9-fold threshold models and held-out fold TAR/FMR across RFW demographic groups."""
    from research.calibration.conditional import (
        assign_quality_groups,
        fit_conditional_threshold,
        fit_global_threshold,
    )
    from research.calibration.continuous import fit_continuous_threshold

    _validate_pair_structure(pairs, strict_official=strict_official)
    score_arr = np.asarray(scores, dtype=np.float64)
    quality_arr = np.asarray(pair_qualities, dtype=np.float64)
    if score_arr.shape != (len(pairs),) or not np.isfinite(score_arr).all():
        raise ValueError("RFW pair scores must be a finite vector aligned to pairs")
    if quality_arr.shape != (len(pairs),) or not np.isfinite(quality_arr).all():
        raise ValueError("RFW pair qualities must be a finite vector aligned to pairs")
    if int(bootstrap_repeats) < 100:
        raise ValueError("bootstrap_repeats must be at least 100")

    df = pairs.copy().reset_index(drop=True)
    df["pair_score"] = score_arr
    df["pair_quality"] = quality_arr

    fold_rows: list[dict[str, Any]] = []
    fitted_models: dict[tuple[str, int, str, float], Any] = {}
    pair_eval_frames: list[pd.DataFrame] = []

    total_candidates = 0
    total_cross_excluded = 0
    total_retained = 0

    for group, group_rows in df.groupby("rfw_group", sort=True):
        folds = sorted(int(v) for v in group_rows["fold_index"].unique())
        if len(folds) < 2:
            raise ValueError(f"RFW group {group!r} requires at least two folds")
        for heldout_fold in folds:
            train = group_rows.loc[group_rows["fold_index"] != heldout_fold]
            test = group_rows.loc[group_rows["fold_index"] == heldout_fold]
            if train.empty or test.empty:
                raise ValueError("RFW train and held-out fold partitions must be non-empty")

            # Two-endpoint identity partition for calibration:
            # Map both left and right endpoints to deterministic role ('fit' or 'safety')
            # Retain only pairs where both endpoints share the exact same role to prevent
            # identity leakage between fit and safety.
            col_left = (
                "left_identity_id"
                if "left_identity_id" in train.columns
                else ("left_identity" if "left_identity" in train.columns else "left_image_id")
            )
            col_right = (
                "right_identity_id"
                if "right_identity_id" in train.columns
                else ("right_identity" if "right_identity" in train.columns else "right_image_id")
            )
            left_ids = train[col_left].astype(str)
            right_ids = train[col_right].astype(str)

            def _role(ident: str) -> str:
                digest = hashlib.sha256(f"{int(bootstrap_seed)}:{ident}".encode("utf-8")).digest()
                uniform = int.from_bytes(digest[:8], "big") / float(2**64)
                return "safety" if uniform < safety_fraction else "fit"

            left_roles = left_ids.map(_role)
            right_roles = right_ids.map(_role)
            retained = left_roles.eq(right_roles)
            cross_excluded = int((~retained).sum())
            retained_count = int(retained.sum())
            total_candidates += len(train)
            total_cross_excluded += cross_excluded
            total_retained += retained_count

            cal_train = train.loc[retained].copy()
            cal_roles = left_roles[retained]
            fit_pairs = cal_train.loc[cal_roles.eq("fit")]
            safety_pairs = cal_train.loc[cal_roles.eq("safety")]
            fit_impostors = int((~fit_pairs["is_genuine"]).sum())
            safety_impostors = int((~safety_pairs["is_genuine"]).sum())
            fit_genuines = int(fit_pairs["is_genuine"].sum())
            safety_genuines = int(safety_pairs["is_genuine"].sum())

            if fit_impostors < 20 or safety_impostors < 20:
                raise ValueError(
                    f"RFW group {group!r} heldout fold {heldout_fold} calibration partition has insufficient "
                    f"non-mated pairs (fit={fit_impostors}, safety={safety_impostors}, minimum 20 required). "
                    f"Excluded {cross_excluded} cross-partition pairs out of {len(train)} candidates."
                )

            cal = pd.DataFrame({
                "sample_id": cal_train["pair_id"].astype(str),
                "identity_id": cal_train[col_left].astype(str),
                "is_mated": cal_train["is_genuine"].astype(bool),
                "score": cal_train["pair_score"].to_numpy(dtype=np.float64),
                "fiqa_score": cal_train["pair_quality"].to_numpy(dtype=np.float64),
            })
            col_tst = col_left if col_left in test.columns else "left_image_id"
            tst = pd.DataFrame({
                "sample_id": test["pair_id"].astype(str),
                "identity_id": test[col_tst].astype(str),
                "is_mated": test["is_genuine"].astype(bool),
                "score": test["pair_score"].to_numpy(dtype=np.float64),
                "fiqa_score": test["pair_quality"].to_numpy(dtype=np.float64),
            })

            labels = tst["is_mated"].to_numpy(dtype=bool)
            npos, nneg = int(labels.sum()), int((~labels).sum())
            if not npos or not nneg:
                raise ValueError("each RFW held-out fold must contain genuine and impostor pairs")

            for target_fmr in target_fmrs:
                target_fpir = float(target_fmr)
                for method in methods:
                    if method == "global_safe":
                        model = fit_global_threshold(
                            cal,
                            target_fpir=target_fpir,
                            score_space=score_space,
                            partition_column="identity_id",
                            partition_seed=bootstrap_seed,
                            safety_fraction=safety_fraction,
                        )
                        tau = np.full(len(tst), model.global_final_threshold)
                    elif method == "fiqa_5bin":
                        model = fit_conditional_threshold(
                            cal,
                            bin_count=5,
                            target_fpir=target_fpir,
                            score_space=score_space,
                            partition_column="identity_id",
                            partition_seed=bootstrap_seed,
                            safety_fraction=safety_fraction,
                        )
                        groups = assign_quality_groups(
                            tst["fiqa_score"],
                            cutpoints=model.quality_cutpoints,
                            labels=model.group_labels,
                        )
                        tau = np.asarray([model.thresholds[g] for g in groups])
                    elif method == "continuous_fiqa":
                        model = fit_continuous_threshold(
                            cal,
                            target_fpir=target_fpir,
                            features=("fiqa_score",),
                            score_space=score_space,
                            partition_seed=bootstrap_seed,
                            safety_fraction=safety_fraction,
                            knot_quantiles=knot_quantiles,
                            smoothing=smoothing,
                            ridge=ridge,
                            max_iterations=max_iterations,
                            margin_slope_cap=margin_slope_cap,
                        )
                        tau = model.predict(tst)
                    else:
                        raise ValueError(f"unsupported calibration method: {method!r}")

                    fitted_models[(str(group), heldout_fold, method, target_fpir)] = model

                    accepted = tst["score"].to_numpy(dtype=np.float64) >= tau
                    if return_raw_pairs:
                        pair_eval_frames.append(pd.DataFrame({
                            "pair_id": test["pair_id"].astype(str).to_numpy(),
                            "rfw_group": str(group),
                            "heldout_fold": int(heldout_fold),
                            "method": str(method),
                            "target_fmr": float(target_fpir),
                            "compression_profile": str(compression_profile),
                            "score_space": str(score_space),
                            "is_genuine": labels,
                            "score": tst["score"].to_numpy(dtype=np.float64),
                            "pair_quality": tst["fiqa_score"].to_numpy(dtype=np.float64),
                            "threshold": tau,
                            "accepted": accepted,
                        }))
                    true_accepts = int(accepted[labels].sum())
                    false_accepts = int(accepted[~labels].sum())
                    false_rejects = npos - true_accepts
                    true_rejects = nneg - false_accepts
                    tar = float(true_accepts / npos)
                    realized_fmr = float(false_accepts / nneg)
                    accuracy = float((true_accepts + true_rejects) / len(tst))
                    target_met = bool(realized_fmr <= target_fpir)
                    low_fmr, high_fmr = wilson_score_interval(false_accepts, nneg)

                    fold_rows.append({
                        "rfw_group": str(group),
                        "heldout_fold": heldout_fold,
                        "method": method,
                        "target_fmr": target_fpir,
                        "compression_profile": compression_profile,
                        "score_space": score_space,
                        "train_candidate_pairs": int(len(train)),
                        "cross_partition_pairs_excluded": cross_excluded,
                        "calibration_pairs_retained": retained_count,
                        "train_pair_count": retained_count,
                        "fit_pair_count": int(len(fit_pairs)),
                        "safety_pair_count": int(len(safety_pairs)),
                        "fit_impostor_count": fit_impostors,
                        "safety_impostor_count": safety_impostors,
                        "fit_genuine_count": fit_genuines,
                        "safety_genuine_count": safety_genuines,
                        "test_pair_count": int(len(test)),
                        "genuine_pair_count": npos,
                        "impostor_pair_count": nneg,
                        "true_accepts": true_accepts,
                        "false_accepts": false_accepts,
                        "false_rejects": false_rejects,
                        "true_rejects": true_rejects,
                        "accuracy": accuracy,
                        "tar": tar,
                        "realized_fmr": realized_fmr,
                        "target_met_on_test": target_met,
                        "fmr_wilson95_low": low_fmr,
                        "fmr_wilson95_high": high_fmr,
                        "threshold_min": float(np.min(tau)),
                        "threshold_max": float(np.max(tau)),
                    })

    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["rfw_group", "method", "target_fmr", "heldout_fold"]
    ).reset_index(drop=True)

    group_rows: list[dict[str, Any]] = []
    for (group, method, target_fmr), values in fold_metrics.groupby(
        ["rfw_group", "method", "target_fmr"], sort=True
    ):
        accs = values["accuracy"].to_numpy(dtype=np.float64)
        tars = values["tar"].to_numpy(dtype=np.float64)
        fmrs = values["realized_fmr"].to_numpy(dtype=np.float64)

        acc_low, acc_high = _bootstrap_mean_ci(accs, seed=int(bootstrap_seed) + 1, repeats=bootstrap_repeats)
        tar_low, tar_high = _bootstrap_mean_ci(tars, seed=int(bootstrap_seed) + 2, repeats=bootstrap_repeats)
        fmr_low, fmr_high = _bootstrap_mean_ci(fmrs, seed=int(bootstrap_seed) + 3, repeats=bootstrap_repeats)

        group_rows.append({
            "rfw_group": str(group),
            "method": str(method),
            "target_fmr": float(target_fmr),
            "compression_profile": compression_profile,
            "score_space": score_space,
            "fold_count": int(len(values)),
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs, ddof=0)),
            "accuracy_ci95_low": acc_low,
            "accuracy_ci95_high": acc_high,
            "mean_tar": float(np.mean(tars)),
            "std_tar": float(np.std(tars, ddof=0)),
            "tar_ci95_low": tar_low,
            "tar_ci95_high": tar_high,
            "mean_realized_fmr": float(np.mean(fmrs)),
            "std_realized_fmr": float(np.std(fmrs, ddof=0)),
            "realized_fmr_ci95_low": fmr_low,
            "realized_fmr_ci95_high": fmr_high,
            "target_met_count": int(values["target_met_on_test"].sum()),
            "target_met_rate": float(values["target_met_on_test"].mean()),
            "pooled_true_accepts": int(values["true_accepts"].sum()),
            "pooled_false_accepts": int(values["false_accepts"].sum()),
            "pooled_genuine_pairs": int(values["genuine_pair_count"].sum()),
            "pooled_impostor_pairs": int(values["impostor_pair_count"].sum()),
            "pooled_realized_fmr": (
                float(values["false_accepts"].sum() / values["impostor_pair_count"].sum())
                if int(values["impostor_pair_count"].sum()) > 0
                else 0.0
            ),
            "pooled_tar": (
                float(values["true_accepts"].sum() / values["genuine_pair_count"].sum())
                if int(values["genuine_pair_count"].sum()) > 0
                else 0.0
            ),
            "mean_safety_pairs_retained": float(values["safety_pair_count"].mean()),
            "mean_cross_partition_pairs_excluded": float(values["cross_partition_pairs_excluded"].mean()),
        })

    group_summary = pd.DataFrame(group_rows).sort_values(
        ["rfw_group", "method", "target_fmr"]
    ).reset_index(drop=True)

    summary: dict[str, Any] = {
        "protocol": "rfw_official_groupwise_10fold_continuous_calibration",
        "compression_profile": compression_profile,
        "score_space": score_space,
        "methods": list(methods),
        "target_fmrs": [float(t) for t in target_fmrs],
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_repeats": int(bootstrap_repeats),
        "total_pairs": len(pairs),
        "total_calibration_candidates": total_candidates,
        "total_cross_partition_pairs_excluded": total_cross_excluded,
        "total_calibration_pairs_retained": total_retained,
        "calibration_retention_rate": float(total_retained / total_candidates) if total_candidates > 0 else 1.0,
        "formal_fmr_guarantee": False,
    }

    raw_pairs_df = (
        pd.concat(pair_eval_frames, ignore_index=True)
        if (return_raw_pairs and pair_eval_frames)
        else None
    )

    return RFWContinuousCalibrationResult(
        fold_metrics=fold_metrics,
        group_summary=group_summary,
        summary=summary,
        models=fitted_models,
        raw_pair_evaluations=raw_pairs_df,
    )

