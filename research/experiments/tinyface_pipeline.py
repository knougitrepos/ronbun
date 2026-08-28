from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from research.compression.profiles import (
    ORIGIN_512,
    PCACompressor,
    PQCompressor,
    pq_profile_name,
    validate_pq_sdc_settings,
)
from research.datasets.tinyface import (
    TINYFACE_DATASET_ID,
    TinyFaceOfficialBundle,
    build_tinyface_official_bundle,
    select_tinyface_protocol_fraction,
)
from research.embeddings import read_model_spec
from research.embeddings.registry import create_pytorch_adapter_from_spec
from research.evaluation.tinyface import (
    TINYFACE_RANKS,
    evaluate_tinyface_identification,
    load_tinyface_completed_evaluation,
    paired_tinyface_deltas,
)
from research.experiments.pipeline_runner import (
    DEFAULT_RUN_TIER,
    QUICK_DATA_FRACTIONS,
    SUPPORTED_RUN_TIERS,
)
from research.runtime.hashing import canonical_sha256, sha256_file
from research.runtime.run_store import RunStore, inspect_git_provenance


TINYFACE_RUN_TIERS = SUPPORTED_RUN_TIERS
TINYFACE_QUICK_DATA_FRACTION = float(QUICK_DATA_FRACTIONS["tinyface"])
TINYFACE_DEFAULT_PCA_DIMENSIONS = (384, 256, 128, 64, 32)
TINYFACE_DEFAULT_PQ_SETTINGS = (
    {"m": 8, "nbits": 8},
    {"m": 16, "nbits": 8},
    {"m": 32, "nbits": 8},
    {"m": 64, "nbits": 8},
    {"m": 128, "nbits": 8},
)
TINYFACE_DEFAULT_PQ_SDC_SETTINGS = ((128, 8),)
ProgressCallback = Callable[[str, dict[str, object]], None]
TINYFACE_NATIVE_PQ_AUDIT_VERSION = "native_decoded_top20_score_audit_v1"


def tinyface_condition_keys(
    *,
    pca_dimensions: tuple[int, ...],
    pq_settings: tuple[Mapping[str, int], ...],
    pq_sdc_settings: tuple[tuple[int, int], ...],
) -> tuple[tuple[str, str], ...]:
    """Return the exact profile/search-mode contract for a TinyFace run."""

    normalized_pq = tuple(
        (int(setting["m"]), int(setting["nbits"])) for setting in pq_settings
    )
    normalized_sdc = validate_pq_sdc_settings(
        normalized_pq,
        pq_sdc_settings,
        source_dim=512,
    )
    selected_sdc = set(normalized_sdc)
    keys: list[tuple[str, str]] = [(ORIGIN_512, "origin_cosine")]
    for dimension in pca_dimensions:
        profile = f"pca_{int(dimension)}"
        keys.extend(
            (
                (profile, "pca_direct_cosine"),
                (profile, "pca_reconstruction_cosine"),
            )
        )
    for m, nbits in normalized_pq:
        profile = pq_profile_name(m, nbits)
        keys.extend(
            (
                (profile, "pq_reconstruction_cosine"),
                (profile, "pq_one_sided_cosine"),
                (profile, "pq_adc_exhaustive"),
            )
        )
        if (m, nbits) in selected_sdc:
            keys.append((profile, "pq_sdc_exhaustive"))
    if len(set(keys)) != len(keys):
        raise ValueError("TinyFace condition keys must be unique")
    return tuple(keys)


@dataclass(frozen=True)
class TinyFaceExperimentPlan:
    project_root: Path
    raw_root: Path
    run_root: Path
    run_tier: str
    data_fraction: float
    seed: int
    model_name: str
    model_profile: str
    model_uid: str
    model_spec_path: Path
    device: str
    embedding_batch_size: int
    query_batch_size: int
    gallery_batch_size: int
    pca_dimensions: tuple[int, ...]
    pq_settings: tuple[dict[str, int], ...]
    pq_sdc_settings: tuple[tuple[int, int], ...]
    source_protocol_uid: str
    protocol_uid: str
    selected_role_counts: dict[str, int]
    paper_eligible: bool
    plan_id: str

    @property
    def dataset_id(self) -> str:
        return TINYFACE_DATASET_ID

    def config(self) -> dict[str, Any]:
        condition_keys = tinyface_condition_keys(
            pca_dimensions=self.pca_dimensions,
            pq_settings=self.pq_settings,
            pq_sdc_settings=self.pq_sdc_settings,
        )
        return {
            "pipeline_id": "tinyface_official_compression_pipeline_v1",
            "dataset_id": TINYFACE_DATASET_ID,
            "protocol_uid": self.protocol_uid,
            "source_protocol_uid": self.source_protocol_uid,
            "protocol_kind": "official_closed_set_1_to_n_with_distractors",
            "open_set_protocol": False,
            "fpir_tpir_metrics_applicable": False,
            "run_tier": self.run_tier,
            "data_fraction": self.data_fraction,
            "seed": self.seed,
            "model_name": self.model_name,
            "model_profile": self.model_profile,
            "model_uid": self.model_uid,
            "model_spec_path": str(self.model_spec_path),
            "preprocessing_mode": "official_face_crop_resize",
            "device": self.device,
            "embedding_batch_size": self.embedding_batch_size,
            "query_batch_size": self.query_batch_size,
            "gallery_batch_size": self.gallery_batch_size,
            "pca_dimensions": list(self.pca_dimensions),
            "pq_settings": [dict(value) for value in self.pq_settings],
            "pq_sdc_settings": [
                {"m": m, "nbits": nbits} for m, nbits in self.pq_sdc_settings
            ],
            "pq_search_modes": [
                "pq_reconstruction_cosine",
                "pq_one_sided_cosine",
                "pq_adc_exhaustive",
                *(["pq_sdc_exhaustive"] if self.pq_sdc_settings else []),
            ],
            "expected_condition_count": len(condition_keys),
            "selected_role_counts": dict(self.selected_role_counts),
            "paper_eligible": self.paper_eligible,
            "official_metrics": ["mean_average_precision", "rank_1", "rank_5", "rank_10", "rank_20"],
            "compression_fit_role": "development_pool",
            "plan_id": self.plan_id,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.config(),
            "project_root": str(self.project_root),
            "raw_root": str(self.raw_root),
            "run_root": str(self.run_root),
        }


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _selected_bundle(plan: TinyFaceExperimentPlan) -> tuple[TinyFaceOfficialBundle, pd.DataFrame]:
    bundle = build_tinyface_official_bundle(plan.raw_root, strict_official=True)
    if bundle.protocol_uid != plan.source_protocol_uid:
        raise ValueError(
            "TinyFace protocol changed after the plan was built: "
            f"expected={plan.source_protocol_uid}, actual={bundle.protocol_uid}"
        )
    selected = select_tinyface_protocol_fraction(
        bundle.manifest,
        data_fraction=plan.data_fraction,
        seed=plan.seed,
    )
    observed = {
        str(role): int(count)
        for role, count in selected["protocol_role"].value_counts().items()
    }
    if observed != plan.selected_role_counts:
        raise ValueError(
            "TinyFace selected scope changed after planning: "
            f"expected={plan.selected_role_counts}, actual={observed}"
        )
    selected_protocol_uids = set(selected["protocol_uid"].astype(str))
    if selected_protocol_uids != {plan.protocol_uid}:
        raise ValueError(
            "TinyFace selected protocol UID changed after planning: "
            f"expected={plan.protocol_uid}, actual={sorted(selected_protocol_uids)}"
        )
    return bundle, selected


def build_tinyface_experiment_plan(
    *,
    project_root: str | Path,
    run_tier: str = DEFAULT_RUN_TIER,
    seed: int,
    model_name: str,
    model_profile: str,
    model_uid: str,
    model_spec_path: str | Path,
    device: str = "cuda",
    quick_data_fraction: float = TINYFACE_QUICK_DATA_FRACTION,
    embedding_batch_size: int = 256,
    query_batch_size: int = 32,
    gallery_batch_size: int = 8_192,
    pca_dimensions: tuple[int, ...] = TINYFACE_DEFAULT_PCA_DIMENSIONS,
    pq_settings: tuple[Mapping[str, int], ...] = TINYFACE_DEFAULT_PQ_SETTINGS,
    pq_sdc_settings: tuple[tuple[int, int], ...] = (
        TINYFACE_DEFAULT_PQ_SDC_SETTINGS
    ),
) -> TinyFaceExperimentPlan:
    root = Path(project_root).expanduser().resolve()
    tier = str(run_tier).strip().lower()
    if tier not in TINYFACE_RUN_TIERS:
        raise ValueError(f"run_tier must be one of {TINYFACE_RUN_TIERS}")
    fraction = 1.0 if tier == "full" else float(quick_data_fraction)
    if not 0.0 < fraction <= 1.0 or (tier == "full" and fraction != 1.0):
        raise ValueError("TinyFace quick fraction must be in (0, 1] and full must be 1.0")
    raw_root = root / "data" / "raw" / "tinyface"
    bundle = build_tinyface_official_bundle(raw_root, strict_official=True)
    selected = select_tinyface_protocol_fraction(
        bundle.manifest,
        data_fraction=fraction,
        seed=int(seed),
    )
    selected_counts = {
        str(role): int(count)
        for role, count in selected["protocol_role"].value_counts().items()
    }
    selected_protocol_uid = str(selected["protocol_uid"].iloc[0])
    spec_path = Path(model_spec_path).expanduser().resolve()
    spec = read_model_spec(spec_path, verify_checkpoint=True)
    if spec.model_uid != str(model_uid):
        raise ValueError(
            f"TinyFace model UID mismatch: expected={model_uid}, actual={spec.model_uid}"
        )
    dimensions = tuple(int(value) for value in pca_dimensions)
    if len(set(dimensions)) != len(dimensions) or any(
        value < 1 or value > 512 for value in dimensions
    ):
        raise ValueError("TinyFace PCA dimensions must be unique values in [1, 512]")
    normalized_pq = tuple(
        {"m": int(value["m"]), "nbits": int(value["nbits"])}
        for value in pq_settings
    )
    for setting in normalized_pq:
        if 512 % setting["m"] != 0 or setting["nbits"] < 1:
            raise ValueError(f"invalid TinyFace PQ setting: {setting}")
    normalized_pq_sdc = validate_pq_sdc_settings(
        ((value["m"], value["nbits"]) for value in normalized_pq),
        pq_sdc_settings,
        source_dim=512,
    )
    tinyface_condition_keys(
        pca_dimensions=dimensions,
        pq_settings=normalized_pq,
        pq_sdc_settings=normalized_pq_sdc,
    )
    plan_payload = {
        "dataset": TINYFACE_DATASET_ID,
        "source_protocol_uid": bundle.protocol_uid,
        "protocol_uid": selected_protocol_uid,
        "tier": tier,
        "fraction": fraction,
        "seed": int(seed),
        "model_uid": spec.model_uid,
        "model_profile": str(model_profile),
        "device": str(device),
        "selected_role_counts": selected_counts,
        "pca_dimensions": dimensions,
        "pq_settings": normalized_pq,
        "pq_sdc_settings": normalized_pq_sdc,
    }
    return TinyFaceExperimentPlan(
        project_root=root,
        raw_root=raw_root,
        run_root=root / "runs" / "tinyface",
        run_tier=tier,
        data_fraction=fraction,
        seed=int(seed),
        model_name=str(model_name),
        model_profile=str(model_profile),
        model_uid=spec.model_uid,
        model_spec_path=spec_path,
        device=str(device),
        embedding_batch_size=_positive_integer(embedding_batch_size, name="embedding_batch_size"),
        query_batch_size=_positive_integer(query_batch_size, name="query_batch_size"),
        gallery_batch_size=_positive_integer(gallery_batch_size, name="gallery_batch_size"),
        pca_dimensions=dimensions,
        pq_settings=normalized_pq,
        pq_sdc_settings=normalized_pq_sdc,
        source_protocol_uid=bundle.protocol_uid,
        protocol_uid=selected_protocol_uid,
        selected_role_counts=selected_counts,
        paper_eligible=(tier == "full" and fraction == 1.0),
        plan_id=canonical_sha256(plan_payload),
    )


def inspect_tinyface_experiment_plan(plan: TinyFaceExperimentPlan) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    try:
        _, selected = _selected_bundle(plan)
        checks["official_protocol_valid"] = True
        checks["selected_scope_valid"] = len(selected) == sum(plan.selected_role_counts.values())
    except Exception as exc:
        checks["official_protocol_valid"] = False
        checks["selected_scope_valid"] = False
        checks["protocol_error"] = f"{type(exc).__name__}: {exc}"
    try:
        spec = read_model_spec(plan.model_spec_path, verify_checkpoint=True)
        checks["model_spec_verified"] = spec.model_uid == plan.model_uid
    except Exception as exc:
        checks["model_spec_verified"] = False
        checks["model_error"] = f"{type(exc).__name__}: {exc}"
    if plan.device.startswith("cuda"):
        try:
            import torch

            checks["cuda_available"] = bool(torch.cuda.is_available())
            checks["cuda_device_name"] = (
                torch.cuda.get_device_name(0) if checks["cuda_available"] else None
            )
        except Exception as exc:
            checks["cuda_available"] = False
            checks["cuda_error"] = f"{type(exc).__name__}: {exc}"
    else:
        checks["cuda_available"] = "not_required"
    git = inspect_git_provenance(plan.project_root, run_root=plan.run_root)
    checks["git_policy_satisfied"] = not bool(git.get("dirty"))
    required = (
        checks.get("official_protocol_valid") is True,
        checks.get("selected_scope_valid") is True,
        checks.get("model_spec_verified") is True,
        checks.get("git_policy_satisfied") is True,
        checks.get("cuda_available") in {True, "not_required"},
    )
    return {
        "dataset_id": TINYFACE_DATASET_ID,
        "protocol_uid": plan.protocol_uid,
        "paper_eligible": plan.paper_eligible,
        "ready_to_execute_pipeline": all(required),
        "checks": checks,
        "readiness": {"checks": checks},
        "git": git,
        "selected_role_counts": plan.selected_role_counts,
        "metrics": ["mean_average_precision", *[f"rank_{rank}" for rank in TINYFACE_RANKS]],
        "open_set_metrics": "not_applicable",
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resize_batch(paths: list[Path], *, height: int, width: int) -> np.ndarray:
    faces = np.empty((len(paths), height, width, 3), dtype=np.uint8)
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            faces[index] = np.asarray(
                image.convert("RGB").resize((width, height), resample=Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
    return faces


def _extract_embeddings(
    plan: TinyFaceExperimentPlan,
    selected: pd.DataFrame,
    *,
    artifact_root: Path,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    embeddings_path = artifact_root / "origin_embeddings.npy"
    rows_path = artifact_root / "embedding_rows.csv"
    manifest_path = artifact_root / "embedding_manifest.json"
    if embeddings_path.is_file() and rows_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("model_uid") != plan.model_uid
            or manifest.get("plan_id") != plan.plan_id
            or manifest.get("row_count") != len(selected)
            or sha256_file(embeddings_path) != manifest.get("embedding_sha256")
            or sha256_file(rows_path) != manifest.get("rows_sha256")
        ):
            raise ValueError("existing TinyFace embedding artifact does not match the plan")
        rows = pd.read_csv(rows_path)
        if rows["image_id"].astype(str).tolist() != selected["image_id"].astype(str).tolist():
            raise ValueError("TinyFace embedding row order drifted")
        if "source_image_sha256" not in rows.columns:
            raise ValueError("TinyFace embedding ledger lacks source image SHA-256 values")
        current_hashes = [sha256_file(Path(value)) for value in selected["image_path"]]
        if current_hashes != rows["source_image_sha256"].astype(str).tolist():
            raise ValueError("TinyFace source images changed after embedding extraction")
        return np.load(embeddings_path, mmap_mode="r"), rows

    artifact_root.mkdir(parents=True, exist_ok=True)
    spec = read_model_spec(plan.model_spec_path, verify_checkpoint=True)
    adapter = create_pytorch_adapter_from_spec(spec, device=plan.device)
    temporary = embeddings_path.with_name(f".{embeddings_path.name}.{os.getpid()}.tmp")
    matrix = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected), spec.embedding_dim),
    )
    started = perf_counter()
    source_hashes: list[str] = []
    for start in range(0, len(selected), plan.embedding_batch_size):
        stop = min(len(selected), start + plan.embedding_batch_size)
        paths = [Path(value) for value in selected.iloc[start:stop]["image_path"]]
        source_hashes.extend(sha256_file(path) for path in paths)
        faces = _resize_batch(
            paths,
            height=spec.preprocessing.input_height,
            width=spec.preprocessing.input_width,
        )
        output = adapter.embed(faces)
        matrix[start:stop] = output.normalized_embedding
        matrix.flush()
        if progress is not None:
            progress(
                "TinyFace embedding extraction",
                {
                    "phase": "tinyface_embedding_extraction",
                    "completed": stop,
                    "total": len(selected),
                    "fraction": stop / len(selected),
                }
            )
    del matrix
    os.replace(temporary, embeddings_path)
    row_columns = [
        "image_id", "identity_id", "split", "protocol_role", "protocol_index",
        "source_relative_path", "official_result_eligible", "protocol_uid",
    ]
    embedding_rows = selected[row_columns].copy()
    embedding_rows["source_image_sha256"] = source_hashes
    embedding_rows.to_csv(rows_path, index=False, encoding="utf-8", lineterminator="\n")
    _atomic_json(
        manifest_path,
        {
            "artifact_type": "tinyface_origin_embeddings_v1",
            "plan_id": plan.plan_id,
            "model_uid": plan.model_uid,
            "model_spec_sha256": sha256_file(plan.model_spec_path),
            "protocol_uid": plan.protocol_uid,
            "row_count": len(selected),
            "embedding_dimension": spec.embedding_dim,
            "embedding_dtype": "float32",
            "preprocessing_mode": "official_face_crop_resize",
            "resize_interpolation": "PIL.Image.Resampling.BILINEAR",
            "elapsed_seconds": perf_counter() - started,
            "embedding_sha256": sha256_file(embeddings_path),
            "rows_sha256": sha256_file(rows_path),
            "source_image_inventory_sha256": canonical_sha256(
                embedding_rows[["image_id", "source_image_sha256"]].to_dict(
                    orient="records"
                )
            ),
        },
    )
    return np.load(embeddings_path, mmap_mode="r"), pd.read_csv(rows_path)


def _role_arrays(
    embeddings: np.ndarray,
    rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    development_indexes = np.flatnonzero(rows["protocol_role"].eq("development_pool"))
    gallery_indexes = np.flatnonzero(
        rows["protocol_role"].isin(["gallery_match", "gallery_distractor"])
    )
    query_indexes = np.flatnonzero(rows["protocol_role"].eq("registered_probe"))
    return (
        np.ascontiguousarray(embeddings[development_indexes], dtype=np.float32),
        np.ascontiguousarray(embeddings[query_indexes], dtype=np.float32),
        np.ascontiguousarray(embeddings[gallery_indexes], dtype=np.float32),
        rows.iloc[query_indexes]["identity_id"].astype(str).to_numpy(),
        rows.iloc[gallery_indexes]["identity_id"].astype(str).to_numpy(),
        rows.iloc[query_indexes]["image_id"].astype(str).to_numpy(),
    )


def _evaluate_condition(
    *,
    plan: TinyFaceExperimentPlan,
    condition: dict[str, Any],
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_image_ids: np.ndarray,
    score_kind: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    result = evaluate_tinyface_identification(
        query_vectors,
        gallery_vectors,
        query_identity_ids=query_ids,
        gallery_identity_ids=gallery_ids,
        query_image_ids=query_image_ids,
        score_kind=score_kind,  # type: ignore[arg-type]
        query_batch_size=plan.query_batch_size,
        gallery_batch_size=plan.gallery_batch_size,
        compute_device=plan.device,
    )
    summary = {**condition, **result.summary}
    per_query = result.per_query.copy()
    for key in (
        "compression_family", "compression_profile", "search_mode",
        "query_representation", "gallery_representation", "distance_function",
    ):
        per_query[key] = condition[key]
    return summary, per_query


def _native_top20_success(indices: np.ndarray, query_ids: np.ndarray, gallery_ids: np.ndarray) -> np.ndarray:
    return _native_rank_successes(indices, query_ids, gallery_ids)[20]


def _native_rank_successes(
    indices: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
) -> dict[int, np.ndarray]:
    ranked = np.asarray(indices, dtype=np.int64)
    queries = np.asarray(query_ids, dtype=object).reshape(-1)
    gallery = np.asarray(gallery_ids, dtype=object).reshape(-1)
    if ranked.ndim != 2 or len(ranked) != len(queries):
        raise ValueError("native PQ indices must be a query-aligned 2D matrix")
    if ranked.shape[1] < max(TINYFACE_RANKS):
        raise ValueError("native PQ audit requires at least the official top-20")
    if np.any(ranked < 0) or np.any(ranked >= len(gallery)):
        raise ValueError("native PQ indices escaped the gallery range")
    matches = gallery[ranked] == queries[:, None]
    return {
        rank: np.any(matches[:, :rank], axis=1)
        for rank in TINYFACE_RANKS
    }


def _native_decoded_score_equivalence_audit(
    *,
    search_mode: str,
    native_query_vectors: np.ndarray,
    decoded_gallery_vectors: np.ndarray,
    native_distances: np.ndarray,
    native_indices: np.ndarray,
    batch_size: int = 64,
) -> dict[str, Any]:
    queries = np.asarray(native_query_vectors, dtype=np.float32)
    gallery = np.asarray(decoded_gallery_vectors, dtype=np.float32)
    distances = np.asarray(native_distances, dtype=np.float32)
    indices = np.asarray(native_indices, dtype=np.int64)
    if queries.ndim != 2 or gallery.ndim != 2 or queries.shape[1] != gallery.shape[1]:
        raise ValueError("native PQ audit vectors must be compatible 2D matrices")
    if distances.ndim != 2 or indices.shape != distances.shape:
        raise ValueError("native PQ audit distances and indices must have equal 2D shape")
    if len(distances) != len(queries) or np.any(indices < 0) or np.any(indices >= len(gallery)):
        raise ValueError("native PQ audit result rows or gallery indices are invalid")
    if not np.isfinite(distances).all() or np.any(distances < -1e-5):
        raise ValueError("native PQ audit distances must be finite squared L2 values")
    batch = int(batch_size)
    if isinstance(batch_size, bool) or batch < 1:
        raise ValueError("native PQ audit batch_size must be a positive integer")

    float32_epsilon = float(np.finfo(np.float32).eps)
    dimension = int(queries.shape[1])
    accumulation_bound = dimension * float32_epsilon
    relative_tolerance = float(
        2.0 * accumulation_bound / (1.0 - accumulation_bound)
    )
    absolute_tolerance = float(8.0 * float32_epsilon)
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    violation_count = 0
    checked_count = int(distances.size)
    for start in range(0, len(queries), batch):
        stop = min(len(queries), start + batch)
        selected = gallery[indices[start:stop]].astype(np.float64, copy=False)
        query = queries[start:stop].astype(np.float64, copy=False)[:, None, :]
        difference = query - selected
        reference = np.einsum(
            "qkd,qkd->qk",
            difference,
            difference,
            dtype=np.float64,
            optimize=True,
        )
        observed = distances[start:stop].astype(np.float64, copy=False)
        absolute_error = np.abs(observed - reference)
        allowed_error = absolute_tolerance + relative_tolerance * np.abs(reference)
        violation_count += int(np.count_nonzero(absolute_error > allowed_error))
        maximum_absolute_error = max(
            maximum_absolute_error,
            float(np.max(absolute_error, initial=0.0)),
        )
        relative_error = absolute_error / np.maximum(
            np.abs(reference),
            absolute_tolerance,
        )
        maximum_relative_error = max(
            maximum_relative_error,
            float(np.max(relative_error, initial=0.0)),
        )
    if violation_count:
        raise RuntimeError(
            f"TinyFace {search_mode} native distances are not decoded-centroid "
            f"score-equivalent: violations={violation_count}/{checked_count}, "
            f"max_abs={maximum_absolute_error:.9g}"
        )
    return {
        "native_score_equivalence_audit_version": TINYFACE_NATIVE_PQ_AUDIT_VERSION,
        "native_score_equivalence_checked_count": checked_count,
        "native_score_equivalence_violation_count": 0,
        "native_score_equivalence_max_absolute_error": maximum_absolute_error,
        "native_score_equivalence_max_relative_error": maximum_relative_error,
        "native_score_equivalence_relative_tolerance": relative_tolerance,
        "native_score_equivalence_absolute_tolerance": absolute_tolerance,
        "native_score_equivalence_passed": True,
    }


def _audit_native_pq_search(
    *,
    search_mode: str,
    native_query_vectors: np.ndarray,
    decoded_gallery_vectors: np.ndarray,
    native_distances: np.ndarray,
    native_indices: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    decoded_ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(decoded_ledger) != len(query_ids):
        raise ValueError("decoded TinyFace ledger is not query-aligned")
    summary = _native_decoded_score_equivalence_audit(
        search_mode=search_mode,
        native_query_vectors=native_query_vectors,
        decoded_gallery_vectors=decoded_gallery_vectors,
        native_distances=native_distances,
        native_indices=native_indices,
    )
    native_by_rank = _native_rank_successes(
        native_indices,
        query_ids,
        gallery_ids,
    )
    audited = decoded_ledger.copy()
    denominator = int(len(audited))
    summary["native_rank_tie_policy"] = "faiss_native_index_order"
    summary["decoded_rank_tie_policy"] = "stable_gallery_index"
    summary["decoded_native_rank_exact_match_required"] = False
    for rank, native_success in native_by_rank.items():
        column = f"rank_{rank}_success"
        if column not in audited.columns:
            raise ValueError(f"decoded TinyFace ledger is missing {column}")
        decoded_success = audited[column].astype(bool).to_numpy()
        mismatch = native_success != decoded_success
        native_only = native_success & ~decoded_success
        decoded_only = ~native_success & decoded_success
        audited[f"native_rank_{rank}_success"] = native_success
        audited[f"decoded_native_rank_{rank}_mismatch"] = mismatch
        summary.update(
            {
                f"native_rank_{rank}": float(native_success.mean()),
                f"native_rank_{rank}_success_count": int(native_success.sum()),
                f"native_rank_{rank}_denominator": denominator,
                f"decoded_native_rank_{rank}_mismatch_count": int(mismatch.sum()),
                f"decoded_native_rank_{rank}_native_only_count": int(native_only.sum()),
                f"decoded_native_rank_{rank}_decoded_only_count": int(decoded_only.sum()),
                f"decoded_native_rank_{rank}_agreement_rate": float(1.0 - mismatch.mean()),
            }
        )
    return audited, summary


def _run_compression_evaluation(
    plan: TinyFaceExperimentPlan,
    embeddings: np.ndarray,
    rows: pd.DataFrame,
    *,
    artifact_root: Path,
    run_id: str,
) -> dict[str, Any]:
    development, queries, gallery, query_ids, gallery_ids, query_image_ids = _role_arrays(embeddings, rows)
    summaries: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []

    origin_condition = {
        "compression_family": "origin",
        "compression_profile": ORIGIN_512,
        "search_mode": "origin_cosine",
        "query_representation": "origin_float32",
        "gallery_representation": "origin_float32",
        "distance_function": "cosine_similarity",
        "gallery_payload_bytes_per_vector": int(queries.shape[1] * 4),
        "query_payload_bytes_per_vector": int(queries.shape[1] * 4),
        "codec_parameter_bytes": 0,
    }
    summary, origin_ledger = _evaluate_condition(
        plan=plan,
        condition=origin_condition,
        query_vectors=queries,
        gallery_vectors=gallery,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
        query_image_ids=query_image_ids,
        score_kind="cosine",
    )
    summaries.append(summary)
    ledgers.append(origin_ledger)

    for dimension in plan.pca_dimensions:
        compressor = PCACompressor(n_components=dimension, random_state=plan.seed).fit(development)
        query_profile = compressor.transform_profile(queries)
        gallery_profile = compressor.transform_profile(gallery)
        codec_bytes = int(gallery_profile.metadata["codec_parameter_bytes"])
        common = {
            "compression_family": "pca",
            "compression_profile": f"pca_{dimension}",
            "gallery_payload_bytes_per_vector": int(dimension * 4),
            "query_payload_bytes_per_vector": int(dimension * 4),
            "codec_parameter_bytes": codec_bytes,
            "mean_gallery_angular_error": float(np.mean(gallery_profile.angular_error)),
            "p95_gallery_angular_error": float(np.quantile(gallery_profile.angular_error, 0.95)),
        }
        for suffix, query_vectors, gallery_vectors, query_repr, gallery_repr in (
            (
                "pca_direct_cosine", query_profile.vectors, gallery_profile.vectors,
                f"pca_{dimension}_float32", f"pca_{dimension}_float32",
            ),
            (
                "pca_reconstruction_cosine", query_profile.reconstructed_vectors,
                gallery_profile.reconstructed_vectors,
                f"pca_{dimension}_reconstruction_512", f"pca_{dimension}_reconstruction_512",
            ),
        ):
            condition = {
                **common,
                "search_mode": suffix,
                "query_representation": query_repr,
                "gallery_representation": gallery_repr,
                "distance_function": "cosine_similarity",
            }
            pca_summary, pca_ledger = _evaluate_condition(
                plan=plan,
                condition=condition,
                query_vectors=np.asarray(query_vectors),
                gallery_vectors=np.asarray(gallery_vectors),
                query_ids=query_ids,
                gallery_ids=gallery_ids,
                query_image_ids=query_image_ids,
                score_kind="cosine",
            )
            pca_summary.update(paired_tinyface_deltas(origin_ledger, pca_ledger))
            summaries.append(pca_summary)
            ledgers.append(pca_ledger)

    for setting in plan.pq_settings:
        m = int(setting["m"])
        nbits = int(setting["nbits"])
        sdc_selected = (m, nbits) in set(plan.pq_sdc_settings)
        compressor = PQCompressor(
            source_dim=queries.shape[1],
            m=m,
            nbits=nbits,
            source_profile=ORIGIN_512,
            random_state=plan.seed,
        ).fit(development)
        query_profile = compressor.transform_profile(queries)
        gallery_profile = compressor.transform_profile(gallery)
        profile = pq_profile_name(m, nbits)
        common = {
            "compression_family": "pq",
            "compression_profile": profile,
            "gallery_payload_bytes_per_vector": int(gallery_profile.metadata["code_bytes"]),
            "codec_parameter_bytes": int(gallery_profile.metadata["codebook_bytes"]),
            "mean_gallery_angular_error": float(np.mean(gallery_profile.angular_error)),
            "p95_gallery_angular_error": float(np.quantile(gallery_profile.angular_error, 0.95)),
            "pq_m": m,
            "pq_nbits": nbits,
        }
        pq_conditions = [
            (
                "pq_reconstruction_cosine", query_profile.reconstructed_vectors,
                gallery_profile.reconstructed_vectors, "pq_reconstruction_512",
                "pq_reconstruction_512", "cosine", int(gallery_profile.metadata["code_bytes"]),
            ),
            (
                "pq_one_sided_cosine", queries, gallery_profile.reconstructed_vectors,
                "origin_float32", "pq_reconstruction_512", "cosine", int(queries.shape[1] * 4),
            ),
            (
                "pq_adc_exhaustive", queries, gallery_profile.reconstructed_vectors,
                "origin_float32", "pq_codes", "negative_squared_l2", int(queries.shape[1] * 4),
            ),
        ]
        if sdc_selected:
            pq_conditions.append(
                (
                    "pq_sdc_exhaustive", query_profile.reconstructed_vectors,
                    gallery_profile.reconstructed_vectors, "pq_codes", "pq_codes",
                    "negative_squared_l2", int(gallery_profile.metadata["code_bytes"]),
                )
            )
        for mode, query_vectors, gallery_vectors, query_repr, gallery_repr, score_kind, query_bytes in pq_conditions:
            condition = {
                **common,
                "search_mode": mode,
                "query_representation": query_repr,
                "gallery_representation": gallery_repr,
                "distance_function": (
                    "cosine_similarity" if score_kind == "cosine" else "squared_l2_distance"
                ),
                "query_payload_bytes_per_vector": query_bytes,
                "native_compressed_search": mode in {"pq_adc_exhaustive", "pq_sdc_exhaustive"},
                "official_metric_implementation": (
                    "decoded_centroid_exact_ranking_with_native_top20_audit"
                    if mode in {"pq_adc_exhaustive", "pq_sdc_exhaustive"}
                    else "exact_reconstructed_vector_ranking"
                ),
            }
            pq_summary, pq_ledger = _evaluate_condition(
                plan=plan,
                condition=condition,
                query_vectors=np.asarray(query_vectors),
                gallery_vectors=np.asarray(gallery_vectors),
                query_ids=query_ids,
                gallery_ids=gallery_ids,
                query_image_ids=query_image_ids,
                score_kind=score_kind,
            )
            if mode == "pq_adc_exhaustive":
                native_distances, native_indices, native_metrics = compressor.search_adc_with_metrics(
                    queries, np.asarray(gallery_profile.codes), top_k=20
                )
                pq_ledger, native_audit = _audit_native_pq_search(
                    search_mode=mode,
                    native_query_vectors=queries,
                    decoded_gallery_vectors=np.asarray(
                        gallery_profile.reconstructed_vectors
                    ),
                    native_distances=native_distances,
                    native_indices=native_indices,
                    query_ids=query_ids,
                    gallery_ids=gallery_ids,
                    decoded_ledger=pq_ledger,
                )
                pq_summary.update({f"native_{key}": value for key, value in native_metrics.items()})
                pq_summary.update(native_audit)
            elif mode == "pq_sdc_exhaustive":
                native_distances, native_indices, native_metrics = compressor.search_sdc_with_metrics(
                    queries, np.asarray(gallery_profile.codes), top_k=20
                )
                pq_ledger, native_audit = _audit_native_pq_search(
                    search_mode=mode,
                    native_query_vectors=np.asarray(
                        query_profile.reconstructed_vectors
                    ),
                    decoded_gallery_vectors=np.asarray(
                        gallery_profile.reconstructed_vectors
                    ),
                    native_distances=native_distances,
                    native_indices=native_indices,
                    query_ids=query_ids,
                    gallery_ids=gallery_ids,
                    decoded_ledger=pq_ledger,
                )
                pq_summary.update({f"native_{key}": value for key, value in native_metrics.items()})
                pq_summary.update(native_audit)
            pq_summary.update(paired_tinyface_deltas(origin_ledger, pq_ledger))
            summaries.append(pq_summary)
            ledgers.append(pq_ledger)

    summary_frame = pd.DataFrame.from_records(summaries)
    summary_frame.insert(0, "dataset_id", TINYFACE_DATASET_ID)
    summary_frame.insert(1, "model_uid", plan.model_uid)
    summary_frame.insert(2, "run_id", run_id)
    summary_frame["protocol_uid"] = plan.protocol_uid
    summary_frame["run_tier"] = plan.run_tier
    summary_frame["paper_eligible"] = plan.paper_eligible
    summary_frame["official_result_eligible"] = plan.paper_eligible
    summary_frame["fpir_tpir_metrics_applicable"] = False
    expected_condition_keys = set(
        tinyface_condition_keys(
            pca_dimensions=plan.pca_dimensions,
            pq_settings=plan.pq_settings,
            pq_sdc_settings=plan.pq_sdc_settings,
        )
    )
    observed_condition_keys = set(
        zip(
            summary_frame["compression_profile"].astype(str),
            summary_frame["search_mode"].astype(str),
        )
    )
    if (
        observed_condition_keys != expected_condition_keys
        or len(summary_frame) != len(expected_condition_keys)
    ):
        raise RuntimeError(
            "TinyFace condition contract mismatch: "
            f"expected={sorted(expected_condition_keys)}, "
            f"observed={sorted(observed_condition_keys)}"
        )
    summary_frame["gallery_payload_bytes_total"] = (
        summary_frame["gallery_payload_bytes_per_vector"] * len(gallery)
    )
    summary_frame["gallery_storage_bytes_total"] = (
        summary_frame["gallery_payload_bytes_total"]
        + summary_frame["codec_parameter_bytes"]
    )
    summary_frame["gallery_storage_bytes_per_vector_amortized"] = (
        summary_frame["gallery_storage_bytes_total"] / len(gallery)
    )

    per_query_frame = pd.concat(ledgers, ignore_index=True)
    per_query_frame.insert(0, "dataset_id", TINYFACE_DATASET_ID)
    per_query_frame.insert(1, "model_uid", plan.model_uid)
    per_query_frame.insert(2, "run_id", run_id)
    per_query_frame["protocol_uid"] = plan.protocol_uid

    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / "condition_summary.csv"
    per_query_path = artifact_root / "per_query.csv"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.12g")
    per_query_frame.to_csv(per_query_path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.12g")
    outputs = {
        path.name: {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (summary_path, per_query_path)
    }
    manifest = {
        "schema_version": 2,
        "artifact_type": "tinyface_official_compression_evaluation_v1",
        "source_run_id": run_id,
        "dataset_id": TINYFACE_DATASET_ID,
        "model_uid": plan.model_uid,
        "protocol_uid": plan.protocol_uid,
        "protocol_kind": "official_closed_set_1_to_n_with_distractors",
        "open_set_protocol": False,
        "non_mated_probe_count": 0,
        "fpir_tpir_metrics_applicable": False,
        "run_tier": plan.run_tier,
        "paper_eligible": plan.paper_eligible,
        "official_result_eligible": plan.paper_eligible,
        "preprocessing_mode": "official_face_crop_resize",
        "compression_fit_role": "development_pool",
        "score_spaces_calibrated": False,
        "score_space_calibration_reason": "official protocol has no non-mated probe/FPIR operating point",
        "native_pq_validation_contract": {
            "version": TINYFACE_NATIVE_PQ_AUDIT_VERSION,
            "score_equivalence": (
                "native top-20 squared-L2 distances versus float64 direct "
                "distance over decoded centroids"
            ),
            "rank_agreement": (
                "reported as an audit because native Faiss and decoded GPU "
                "ranking can resolve quantization ties differently"
            ),
            "rank_exact_match_required": False,
        },
        "pq_sdc_settings": [
            {"m": m, "nbits": nbits} for m, nbits in plan.pq_sdc_settings
        ],
        "condition_count": len(summary_frame),
        "query_count": len(queries),
        "gallery_count": len(gallery),
        "outputs": outputs,
    }
    _atomic_json(artifact_root / "tinyface_evaluation_manifest.json", manifest)
    return manifest


def run_tinyface_experiment(
    plan: TinyFaceExperimentPlan,
    *,
    execution_acknowledged: bool,
    start_new_run: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if execution_acknowledged is not True:
        raise RuntimeError("TinyFace execution requires explicit acknowledgement")
    preflight = inspect_tinyface_experiment_plan(plan)
    if preflight["ready_to_execute_pipeline"] is not True:
        raise RuntimeError(f"TinyFace preflight failed: {preflight['checks']}")
    config = plan.config()
    run = (
        RunStore.create(
            experiment_name=f"tinyface_official_{plan.model_uid}",
            config=config,
            root=plan.run_root,
            repo_root=plan.project_root,
            allow_dirty=False,
        )
        if start_new_run
        else RunStore.create_or_reuse_active(
            experiment_name=f"tinyface_official_{plan.model_uid}",
            config=config,
            root=plan.run_root,
            repo_root=plan.project_root,
            allow_dirty=False,
        )
    )
    try:
        bundle, selected = _selected_bundle(plan)
        run.record_input(plan.model_spec_path, role="model_spec")
        run.record_input(Path(read_model_spec(plan.model_spec_path).checkpoint.path), role="model_checkpoint")
        for name in ("gallery_match_img_ID_pairs.mat", "probe_img_ID_pairs.mat"):
            run.record_input(plan.raw_root / "Testing_Set" / name, role=f"tinyface_{name}")

        protocol_root = run.run_dir / "artifacts" / "tinyface_protocol"
        protocol_root.mkdir(parents=True, exist_ok=True)
        selected_path = protocol_root / "selected_manifest.csv"
        protocol_manifest_path = protocol_root / "protocol_manifest.json"
        if not selected_path.exists():
            selected.to_csv(selected_path, index=False, encoding="utf-8", lineterminator="\n")
        if not protocol_manifest_path.exists():
            _atomic_json(
                protocol_manifest_path,
                {
                    **bundle.summary,
                    "source_protocol_uid": bundle.protocol_uid,
                    "selected_protocol_uid": plan.protocol_uid,
                    "run_tier": plan.run_tier,
                    "data_fraction": plan.data_fraction,
                    "selected_role_counts": plan.selected_role_counts,
                    "paper_eligible": plan.paper_eligible,
                    "selected_manifest_sha256": sha256_file(selected_path),
                },
            )

        embedding_root = run.run_dir / "artifacts" / "tinyface_embeddings"
        embeddings, embedding_rows = _extract_embeddings(
            plan,
            selected,
            artifact_root=embedding_root,
            progress=progress,
        )
        evaluation_root = run.run_dir / "artifacts" / "tinyface_official"
        evaluation_manifest = _run_compression_evaluation(
            plan,
            embeddings,
            embedding_rows,
            artifact_root=evaluation_root,
            run_id=run.run_id,
        )
        run.complete()
        return {
            "status": "completed",
            "run_id": run.run_id,
            "run_dir": str(run.run_dir),
            "model_uid": plan.model_uid,
            "protocol_uid": plan.protocol_uid,
            "paper_eligible": plan.paper_eligible,
            "condition_count": evaluation_manifest["condition_count"],
        }
    except BaseException as exc:
        run.fail(exc)
        raise


def reuse_completed_tinyface_run(
    plan: TinyFaceExperimentPlan,
    run_dir: str | Path,
) -> dict[str, Any]:
    evaluation = load_tinyface_completed_evaluation(run_dir)
    run_manifest = json.loads((evaluation.root / "run_manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("config_hash") != canonical_sha256(plan.config()):
        raise ValueError("selected TinyFace completed run does not match the current plan")
    if evaluation.manifest.get("model_uid") != plan.model_uid:
        raise ValueError("selected TinyFace run model UID differs from the plan")
    if evaluation.manifest.get("protocol_uid") != plan.protocol_uid:
        raise ValueError("selected TinyFace run protocol UID differs from the plan")
    return {
        "status": "already_completed",
        "run_id": str(run_manifest["run_id"]),
        "run_dir": str(evaluation.root),
        "model_uid": plan.model_uid,
        "protocol_uid": plan.protocol_uid,
        "paper_eligible": bool(evaluation.manifest.get("paper_eligible")),
        "condition_count": int(evaluation.manifest["condition_count"]),
    }
