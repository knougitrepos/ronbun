from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.explainability.gradcam.features import summarize_saliency_features
from research.explainability.gradcam.metrics import occlude_by_saliency
from research.explainability.gradcam.optional import require_torch
from research.explainability.gradcam.pair import PairCosineGradCAM
from research.explainability.gradcam.templates import (
    LeaveOneOutTemplateBundle,
    build_leave_one_out_identity_templates,
)


ProgressCallback = Callable[[str, dict[str, object]], None]


def _emit(progress: ProgressCallback | None, message: str, **details: object) -> None:
    if progress is not None:
        progress(message, details)


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _text(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _aligned_face_batch(aligned_faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(aligned_faces)
    if (
        faces.ndim != 4
        or faces.shape[0] == 0
        or faces.shape[-1] != 3
        or faces.dtype != np.uint8
    ):
        raise ValueError(
            "aligned_faces must be non-empty uint8 NHWC RGB/BGR source crops"
        )
    return faces


def _origin_embedding_artifact_uid(
    *,
    extraction_uid: str,
    model_uid: str,
    sample_ids: np.ndarray,
    normalized_embeddings: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for value in (extraction_uid, model_uid):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for sample_id in sample_ids.astype(str):
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\0")
    digest.update(
        np.ascontiguousarray(normalized_embeddings, dtype=np.float32).tobytes()
    )
    return f"origin-embedding-{digest.hexdigest()[:24]}"


@dataclass(frozen=True)
class PreparedPopulationInputs:
    extraction_uid: str
    dataset_id: str
    sample_ids: np.ndarray
    identity_ids: np.ndarray
    scope_ids: np.ndarray
    raw_embeddings: np.ndarray
    raw_norms: np.ndarray
    normalized_embeddings: np.ndarray
    loo_templates: LeaveOneOutTemplateBundle
    model_uid: str
    checkpoint_sha256: str
    preprocess_hash: str
    origin_embedding_artifact_uid: str

    def __post_init__(self) -> None:
        row_count = len(self.sample_ids)
        if row_count == 0:
            raise ValueError("prepared population must not be empty")
        if len(set(self.sample_ids.astype(str))) != row_count:
            raise ValueError("prepared sample_ids must be unique")
        for name, values in (
            ("identity_ids", self.identity_ids),
            ("scope_ids", self.scope_ids),
            ("raw_norms", self.raw_norms),
        ):
            if len(values) != row_count:
                raise ValueError(f"{name} must align with sample_ids")
        if self.raw_embeddings.shape != self.normalized_embeddings.shape:
            raise ValueError("raw and normalized embedding matrices must align")
        if self.raw_embeddings.ndim != 2 or len(self.raw_embeddings) != row_count:
            raise ValueError("embedding matrices must have one row per sample")
        if not np.all(np.isfinite(self.raw_embeddings)):
            raise ValueError("raw_embeddings must contain only finite values")
        if not np.all(np.isfinite(self.normalized_embeddings)):
            raise ValueError("normalized_embeddings must contain only finite values")
        if not np.all(np.isfinite(self.raw_norms)) or np.any(self.raw_norms <= 0):
            raise ValueError("raw_norms must be positive and finite")
        observed_raw_norms = np.linalg.norm(
            self.raw_embeddings.astype(np.float64),
            axis=1,
        )
        if not np.allclose(
            observed_raw_norms,
            self.raw_norms,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("raw_norms do not match raw_embeddings")
        unit_norms = np.linalg.norm(
            self.normalized_embeddings.astype(np.float64),
            axis=1,
        )
        if not np.allclose(unit_norms, 1.0, rtol=0.0, atol=1e-4):
            raise ValueError("normalized_embeddings must have unit row norms")
        if len(self.loo_templates.sample_ids) != row_count:
            raise ValueError("leave-one-out templates must align with samples")
        if not np.array_equal(
            self.sample_ids.astype(str),
            self.loo_templates.sample_ids.astype(str),
        ):
            raise ValueError("leave-one-out template sample order changed")
        if self.loo_templates.model_uid != self.model_uid:
            raise ValueError("leave-one-out template model_uid changed")
        for name in (
            "extraction_uid",
            "dataset_id",
            "model_uid",
            "checkpoint_sha256",
            "preprocess_hash",
            "origin_embedding_artifact_uid",
        ):
            _text(getattr(self, name), name=name)

    def sample_frame(self) -> pd.DataFrame:
        frame = self.loo_templates.metadata_frame()
        frame.insert(0, "dataset_id", self.dataset_id)
        frame.insert(0, "extraction_uid", self.extraction_uid)
        frame["origin_embedding_artifact_uid"] = self.origin_embedding_artifact_uid
        frame["checkpoint_sha256"] = self.checkpoint_sha256
        frame["preprocess_hash"] = self.preprocess_hash
        frame["raw_embedding_norm"] = self.raw_norms.astype(
            np.float64,
            copy=False,
        )
        return frame


@dataclass(frozen=True)
class PopulationSaliencyResult:
    extraction_uid: str
    dataset_id: str
    model_uid: str
    origin_embedding_artifact_uid: str
    saliency_spec_uid: str
    target_layer: str
    features: pd.DataFrame
    heatmap_sample_ids: np.ndarray
    normalized_heatmaps: np.ndarray
    raw_cams: np.ndarray
    relu_cams: np.ndarray
    channel_weights: np.ndarray
    pass_b_raw_embeddings: np.ndarray
    pass_b_raw_norms: np.ndarray
    pass_b_normalized_embeddings: np.ndarray
    activations: np.ndarray | None = None
    gradients: np.ndarray | None = None

    def __post_init__(self) -> None:
        heatmap_count = len(self.heatmap_sample_ids)
        if len(set(self.heatmap_sample_ids.astype(str))) != heatmap_count:
            raise ValueError("heatmap_sample_ids must be unique")
        for name, values in (
            ("normalized_heatmaps", self.normalized_heatmaps),
            ("raw_cams", self.raw_cams),
            ("relu_cams", self.relu_cams),
            ("channel_weights", self.channel_weights),
            ("pass_b_raw_embeddings", self.pass_b_raw_embeddings),
            ("pass_b_raw_norms", self.pass_b_raw_norms),
            ("pass_b_normalized_embeddings", self.pass_b_normalized_embeddings),
        ):
            if len(values) != heatmap_count:
                raise ValueError(f"{name} must align with heatmap_sample_ids")
        required_feature_columns = {
            "extraction_uid",
            "dataset_id",
            "sample_id",
            "model_uid",
            "origin_embedding_artifact_uid",
            "saliency_spec_uid",
            "saliency_target_eligible",
            "heatmap_available",
            "heatmap_index",
        }
        missing = sorted(required_feature_columns.difference(self.features.columns))
        if missing:
            raise ValueError(f"population saliency features missing: {missing}")
        if self.features.empty:
            raise ValueError("population saliency features must not be empty")
        if self.features.duplicated(
            ["extraction_uid", "dataset_id", "sample_id", "model_uid"]
        ).any():
            raise ValueError("population saliency feature keys must be unique")
        frozen_values = {
            "extraction_uid": self.extraction_uid,
            "dataset_id": self.dataset_id,
            "model_uid": self.model_uid,
            "origin_embedding_artifact_uid": self.origin_embedding_artifact_uid,
            "saliency_spec_uid": self.saliency_spec_uid,
        }
        for column, expected in frozen_values.items():
            if not self.features[column].astype(str).eq(str(expected)).all():
                raise ValueError(f"feature column {column} changed provenance")
        available = self.features["heatmap_available"].astype(bool)
        available_ids = set(self.features.loc[available, "sample_id"].astype(str))
        if available_ids != set(self.heatmap_sample_ids.astype(str)):
            raise ValueError(
                "heatmap feature availability differs from heatmap_sample_ids"
            )
        observed_indices = sorted(
            self.features.loc[available, "heatmap_index"].astype(int).tolist()
        )
        if observed_indices != list(range(heatmap_count)):
            raise ValueError("heatmap_index must form one contiguous range")
        for name in (
            "extraction_uid",
            "dataset_id",
            "model_uid",
            "origin_embedding_artifact_uid",
            "saliency_spec_uid",
            "target_layer",
        ):
            _text(getattr(self, name), name=name)


def prepare_population_saliency_inputs(
    adapter: Any,
    aligned_faces: np.ndarray,
    *,
    sample_ids: Sequence[object] | np.ndarray,
    identity_ids: Sequence[object] | np.ndarray,
    scope_ids: Sequence[object] | np.ndarray,
    extraction_uid: str,
    dataset_id: str,
    embedding_batch_size: int = 64,
    require_all_eligible: bool = False,
    progress: ProgressCallback | None = None,
) -> PreparedPopulationInputs:
    """Pass A: extract every origin embedding, then build LOO templates."""

    faces = _aligned_face_batch(aligned_faces)
    batch_size = _positive_integer(
        embedding_batch_size,
        name="embedding_batch_size",
    )
    raw_parts: list[np.ndarray] = []
    norm_parts: list[np.ndarray] = []
    normalized_parts: list[np.ndarray] = []
    model_uids: set[str] = set()
    checkpoint_hashes: set[str] = set()
    preprocess_hashes: set[str] = set()
    for start in range(0, len(faces), batch_size):
        output = adapter.embed(faces[start : start + batch_size])
        raw_parts.append(np.asarray(output.raw_embedding, dtype=np.float32))
        norm_parts.append(np.asarray(output.raw_norm, dtype=np.float32))
        normalized_parts.append(
            np.asarray(output.normalized_embedding, dtype=np.float32)
        )
        model_uids.add(str(output.model_uid))
        checkpoint_hashes.add(str(output.checkpoint_sha256))
        preprocess_hashes.add(str(output.preprocess_hash))
        _emit(
            progress,
            "origin embedding extraction",
            processed=min(start + batch_size, len(faces)),
            total=len(faces),
        )
    if len(model_uids) != 1:
        raise ValueError("embedding batches produced different model_uid values")
    if len(checkpoint_hashes) != 1 or len(preprocess_hashes) != 1:
        raise ValueError("embedding provenance changed between batches")

    raw_embeddings = np.concatenate(raw_parts, axis=0)
    raw_norms = np.concatenate(norm_parts, axis=0)
    normalized_embeddings = np.concatenate(normalized_parts, axis=0)
    model_uid = next(iter(model_uids))
    loo = build_leave_one_out_identity_templates(
        sample_ids,
        identity_ids,
        normalized_embeddings,
        model_uid=model_uid,
        scope_ids=scope_ids,
        require_all_eligible=require_all_eligible,
    )
    origin_uid = _origin_embedding_artifact_uid(
        extraction_uid=_text(extraction_uid, name="extraction_uid"),
        model_uid=model_uid,
        sample_ids=loo.sample_ids,
        normalized_embeddings=normalized_embeddings,
    )
    return PreparedPopulationInputs(
        extraction_uid=str(extraction_uid).strip(),
        dataset_id=_text(dataset_id, name="dataset_id"),
        sample_ids=loo.sample_ids,
        identity_ids=loo.identity_ids,
        scope_ids=loo.scope_ids,
        raw_embeddings=raw_embeddings,
        raw_norms=raw_norms,
        normalized_embeddings=normalized_embeddings,
        loo_templates=loo,
        model_uid=model_uid,
        checkpoint_sha256=next(iter(checkpoint_hashes)),
        preprocess_hash=next(iter(preprocess_hashes)),
        origin_embedding_artifact_uid=origin_uid,
    )


def _slice_region_masks(
    region_masks: Mapping[str, np.ndarray] | Any | None,
    *,
    row_indices: np.ndarray,
    total_rows: int,
    image_size: tuple[int, int],
) -> dict[str, np.ndarray] | None:
    if region_masks is None:
        return None
    build_masks = getattr(region_masks, "build_region_masks", None)
    if callable(build_masks):
        sample_count = getattr(region_masks, "sample_count", total_rows)
        if int(sample_count) != total_rows:
            raise ValueError(
                "region mask provider row count must match aligned/prepared samples"
            )
        generated = build_masks(row_indices, image_size=image_size)
        return {str(name): np.asarray(mask) for name, mask in generated.items()}
    sliced: dict[str, np.ndarray] = {}
    for name, mask in region_masks.items():
        values = np.asarray(mask)
        if values.ndim == 3 and values.shape[0] == total_rows:
            sliced[str(name)] = values[row_indices]
        else:
            sliced[str(name)] = values
    return sliced


def _neutral_source_pixel(adapter: Any) -> np.ndarray:
    preprocessing = adapter.spec.preprocessing
    neutral = np.asarray(
        preprocessing.channel_mean,
        dtype=np.float32,
    )
    if neutral.shape != (3,) or not np.all(np.isfinite(neutral)):
        raise ValueError("preprocessing channel_mean must contain 3 finite values")
    if preprocessing.source_color_order != preprocessing.model_color_order:
        neutral = neutral[::-1]
    return neutral


def _embed_unit(adapter: Any, images: np.ndarray, *, batch_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        parts.append(
            np.asarray(
                adapter.embed(images[start : start + batch_size]).normalized_embedding,
                dtype=np.float32,
            )
        )
    return np.concatenate(parts, axis=0)


def _faithfulness_columns(
    adapter: Any,
    images: np.ndarray,
    heatmaps: np.ndarray,
    templates: np.ndarray,
    sample_ids: np.ndarray,
    origin_scores: np.ndarray,
    *,
    fraction: float,
    random_repeats: int,
    seed: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("faithfulness_fraction must be in (0, 1]")
    repeats = _positive_integer(random_repeats, name="random_repeats")
    neutral = _neutral_source_pixel(adapter)
    high_images = occlude_by_saliency(
        images,
        heatmaps,
        fraction=fraction,
        strategy="high_saliency",
        fill_value=neutral,
        seed=seed,
        sample_ids=sample_ids,
    )
    low_images = occlude_by_saliency(
        images,
        heatmaps,
        fraction=fraction,
        strategy="low_saliency",
        fill_value=neutral,
        seed=seed,
        sample_ids=sample_ids,
    )
    high_scores = np.sum(
        _embed_unit(adapter, high_images, batch_size=batch_size) * templates,
        axis=1,
    )
    low_scores = np.sum(
        _embed_unit(adapter, low_images, batch_size=batch_size) * templates,
        axis=1,
    )
    random_score_parts = []
    for repeat in range(repeats):
        random_images = occlude_by_saliency(
            images,
            heatmaps,
            fraction=fraction,
            strategy="random",
            fill_value=neutral,
            seed=seed + repeat,
            sample_ids=sample_ids,
        )
        random_score_parts.append(
            np.sum(
                _embed_unit(adapter, random_images, batch_size=batch_size) * templates,
                axis=1,
            )
        )
    random_scores = np.stack(random_score_parts, axis=0)
    random_mean = np.mean(random_scores, axis=0)
    return {
        "faithfulness_occlusion_fraction": np.full(
            len(images),
            fraction,
            dtype=np.float64,
        ),
        "high_saliency_occlusion_score_drop": origin_scores - high_scores,
        "low_saliency_occlusion_score_drop": origin_scores - low_scores,
        "random_occlusion_score_drop": origin_scores - random_mean,
        "random_occlusion_score_drop_std": np.std(
            origin_scores[None, :] - random_scores,
            axis=0,
            ddof=0,
        ),
        "faithfulness_gain_over_low_saliency": low_scores - high_scores,
        "faithfulness_gain_over_random": random_mean - high_scores,
    }


def extract_population_gradcam(
    adapter: Any,
    aligned_faces: np.ndarray,
    prepared: PreparedPopulationInputs,
    *,
    gradcam_batch_size: int = 4,
    region_masks: Mapping[str, np.ndarray] | Any | None = None,
    region_mask_uid: str | None = None,
    capture_intermediates: bool = False,
    minimum_pass_repeat_cosine: float = 0.99999,
    faithfulness_fraction: float | None = 0.10,
    faithfulness_random_repeats: int = 5,
    faithfulness_seed: int = 42,
    faithfulness_batch_size: int = 32,
    saliency_sample_mask: Sequence[bool] | np.ndarray | None = None,
    progress: ProgressCallback | None = None,
) -> PopulationSaliencyResult:
    """Pass B: Grad-CAM every LOO-eligible image before compression.

    All selected samples retain a feature record. Samples without an identity
    LOO target keep NaN saliency fields and their explicit target status. No
    singleton or unlabeled sample is silently assigned a different target.
    """

    faces = _aligned_face_batch(aligned_faces)
    if len(faces) != len(prepared.sample_ids):
        raise ValueError("aligned_faces must align with prepared samples")
    batch_size = _positive_integer(
        gradcam_batch_size,
        name="gradcam_batch_size",
    )
    minimum_repeat = float(minimum_pass_repeat_cosine)
    if not np.isfinite(minimum_repeat) or not -1.0 <= minimum_repeat <= 1.0:
        raise ValueError("minimum_pass_repeat_cosine must be in [-1, 1]")
    if region_masks is not None and not str(region_mask_uid or "").strip():
        raise ValueError(
            "region_mask_uid is required when semantic region masks are supplied"
        )
    if str(adapter.spec.model_uid) != prepared.model_uid:
        raise ValueError(
            "Pass-B adapter model_uid differs from the frozen Pass-A model_uid"
        )
    if str(adapter.spec.checkpoint.sha256) != prepared.checkpoint_sha256:
        raise ValueError(
            "Pass-B adapter checkpoint differs from the frozen Pass-A checkpoint"
        )
    if str(adapter.spec.preprocessing.preprocess_hash) != prepared.preprocess_hash:
        raise ValueError(
            "Pass-B preprocessing differs from the frozen Pass-A preprocessing"
        )
    target_layer_name = _text(
        adapter.spec.target_layer,
        name="target_layer",
    )
    if saliency_sample_mask is None:
        selected_for_saliency = np.ones(len(faces), dtype=bool)
    else:
        selected_for_saliency = np.asarray(saliency_sample_mask)
        if (
            selected_for_saliency.ndim != 1
            or len(selected_for_saliency) != len(faces)
            or selected_for_saliency.dtype.kind != "b"
        ):
            raise ValueError(
                "saliency_sample_mask must be a boolean vector aligned with samples"
            )
        selected_for_saliency = selected_for_saliency.astype(bool, copy=False)
    eligible_indices = np.flatnonzero(
        prepared.loo_templates.eligible & selected_for_saliency
    )
    if len(eligible_indices) == 0:
        raise ValueError("no samples have an eligible leave-one-out target")

    analyzer = PairCosineGradCAM(
        adapter.model,
        adapter.target_layer,
        embedding_extractor=adapter.select_embedding_tensor,
    )
    torch = require_torch()
    result_parts: dict[str, list[np.ndarray]] = {
        "heatmaps": [],
        "raw_cams": [],
        "relu_cams": [],
        "channel_weights": [],
        "raw_embeddings": [],
        "raw_norms": [],
        "normalized_embeddings": [],
        "target_scores": [],
        "gradient_l2": [],
        "cam_mass": [],
        "valid_heatmap": [],
    }
    activation_parts: list[np.ndarray] = []
    gradient_parts: list[np.ndarray] = []
    for start in range(0, len(eligible_indices), batch_size):
        row_indices = eligible_indices[start : start + batch_size]
        query_tensor = adapter.preprocess(faces[row_indices])
        template_tensor = torch.from_numpy(
            prepared.loo_templates.templates[row_indices]
        ).to(adapter.device)
        generated = analyzer.generate(
            query_tensor,
            template_tensor,
            batch_mode="single" if len(row_indices) == 1 else "independent",
            target_space="origin_embedding",
            target_name=prepared.loo_templates.target_name,
            capture_intermediates=capture_intermediates,
        )
        for key, value in (
            ("heatmaps", generated.heatmaps),
            ("raw_cams", generated.raw_cams),
            ("relu_cams", generated.relu_cams),
            ("channel_weights", generated.channel_weights),
            ("raw_embeddings", generated.raw_embeddings),
            ("raw_norms", generated.raw_norms),
            ("normalized_embeddings", generated.normalized_embeddings),
            ("target_scores", generated.target_scores),
            ("gradient_l2", generated.gradient_l2),
            ("cam_mass", generated.cam_mass),
            ("valid_heatmap", generated.valid_heatmap),
        ):
            result_parts[key].append(np.asarray(value))
        if generated.activations is not None:
            activation_parts.append(generated.activations)
        if generated.gradients is not None:
            gradient_parts.append(generated.gradients)
        _emit(
            progress,
            "population Grad-CAM extraction",
            processed=min(start + batch_size, len(eligible_indices)),
            total=len(eligible_indices),
        )

    combined = {
        name: np.concatenate(parts, axis=0) for name, parts in result_parts.items()
    }
    pass_repeat_cosine = np.sum(
        combined["normalized_embeddings"]
        * prepared.normalized_embeddings[eligible_indices],
        axis=1,
    )
    if np.any(pass_repeat_cosine < minimum_repeat):
        minimum_observed = float(np.min(pass_repeat_cosine))
        raise ValueError(
            "Pass-B Grad-CAM embeddings do not reproduce Pass-A origin "
            f"embeddings; minimum cosine={minimum_observed:.8f}"
        )
    score_delta = np.abs(
        combined["target_scores"]
        - prepared.loo_templates.target_scores[eligible_indices]
    )
    if np.any(score_delta > 1e-5):
        raise ValueError(
            "Grad-CAM target scores differ from frozen LOO template scores; "
            f"maximum absolute difference={float(np.max(score_delta)):.6g}"
        )

    sliced_masks = _slice_region_masks(
        region_masks,
        row_indices=eligible_indices,
        total_rows=len(faces),
        image_size=tuple(int(value) for value in combined["heatmaps"].shape[-2:]),
    )
    spatial = summarize_saliency_features(
        combined["heatmaps"],
        region_masks=sliced_masks,
    )
    eligible_sample_ids = prepared.sample_ids[eligible_indices].astype(str)
    spatial.insert(0, "sample_id", eligible_sample_ids)
    spatial["gradcam_target_score"] = combined["target_scores"]
    spatial["gradcam_gradient_l2"] = combined["gradient_l2"]
    spatial["gradcam_cam_mass"] = combined["cam_mass"]
    spatial["gradcam_valid_heatmap"] = combined["valid_heatmap"].astype(bool)
    spatial["pass_a_pass_b_embedding_cosine"] = pass_repeat_cosine
    spatial["pass_a_pass_b_target_score_abs_diff"] = score_delta

    if faithfulness_fraction is not None:
        faithfulness = _faithfulness_columns(
            adapter,
            faces[eligible_indices],
            combined["heatmaps"],
            prepared.loo_templates.templates[eligible_indices],
            eligible_sample_ids,
            combined["target_scores"].astype(np.float64),
            fraction=float(faithfulness_fraction),
            random_repeats=faithfulness_random_repeats,
            seed=int(faithfulness_seed),
            batch_size=_positive_integer(
                faithfulness_batch_size,
                name="faithfulness_batch_size",
            ),
        )
        for name, values in faithfulness.items():
            spatial[name] = values

    spec_payload = {
        "algorithm": "gradcam",
        "version": 2,
        "extraction_uid": prepared.extraction_uid,
        "origin_embedding_artifact_uid": prepared.origin_embedding_artifact_uid,
        "model_uid": prepared.model_uid,
        "target_layer": target_layer_name,
        "target_name": prepared.loo_templates.target_name,
        "region_mask_uid": str(region_mask_uid or "none"),
        "faithfulness_fraction": faithfulness_fraction,
        "faithfulness_random_repeats": int(faithfulness_random_repeats),
        "saliency_selected_sample_sha256": hashlib.sha256(
            "\x1f".join(
                prepared.sample_ids[selected_for_saliency].astype(str)
            ).encode("utf-8")
        ).hexdigest(),
    }
    spec_digest = hashlib.sha256(
        json.dumps(
            spec_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    saliency_spec_uid = f"saliency-{spec_digest[:24]}"

    features = prepared.sample_frame()
    features["saliency_spec_uid"] = saliency_spec_uid
    features["target_layer"] = target_layer_name
    features["region_mask_uid"] = str(region_mask_uid or "none")
    features["saliency_sample_selected"] = selected_for_saliency
    features["heatmap_available"] = False
    features["heatmap_index"] = -1
    feature_columns = [column for column in spatial if column != "sample_id"]
    for column in feature_columns:
        if pd.api.types.is_bool_dtype(spatial[column]):
            features[column] = pd.Series(
                pd.array([pd.NA] * len(features), dtype="boolean")
            )
        else:
            features[column] = np.nan
    row_by_sample = {
        str(sample_id): index
        for index, sample_id in enumerate(features["sample_id"].astype(str))
    }
    for heatmap_index, row in spatial.iterrows():
        destination_index = row_by_sample[str(row["sample_id"])]
        features.at[destination_index, "heatmap_available"] = True
        features.at[destination_index, "heatmap_index"] = int(heatmap_index)
        for column in feature_columns:
            features.at[destination_index, column] = row[column]
    features["heatmap_available"] = features["heatmap_available"].astype(bool)
    features["heatmap_index"] = features["heatmap_index"].astype(np.int64)

    return PopulationSaliencyResult(
        extraction_uid=prepared.extraction_uid,
        dataset_id=prepared.dataset_id,
        model_uid=prepared.model_uid,
        origin_embedding_artifact_uid=prepared.origin_embedding_artifact_uid,
        saliency_spec_uid=saliency_spec_uid,
        target_layer=target_layer_name,
        features=features,
        heatmap_sample_ids=eligible_sample_ids,
        normalized_heatmaps=combined["heatmaps"].astype(np.float32, copy=False),
        raw_cams=combined["raw_cams"].astype(np.float32, copy=False),
        relu_cams=combined["relu_cams"].astype(np.float32, copy=False),
        channel_weights=combined["channel_weights"].astype(np.float32, copy=False),
        pass_b_raw_embeddings=combined["raw_embeddings"].astype(
            np.float32,
            copy=False,
        ),
        pass_b_raw_norms=combined["raw_norms"].astype(np.float32, copy=False),
        pass_b_normalized_embeddings=combined["normalized_embeddings"].astype(
            np.float32,
            copy=False,
        ),
        activations=(
            np.concatenate(activation_parts, axis=0) if activation_parts else None
        ),
        gradients=(np.concatenate(gradient_parts, axis=0) if gradient_parts else None),
    )
