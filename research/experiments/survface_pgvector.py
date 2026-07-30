from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from research.compression import ORIGIN_512, pca_profile_dimension
from research.database.connection import ensure_vector_indexes, session_scope
from research.database.models import PCA_EMBEDDING_MODELS, Embedding512, Image
from research.database.repository import VectorRepository
from research.protocols import build_survface_official_protocol


ProgressCallback = Callable[[str, dict[str, object]], None]
SURVFACE_PROTOCOL_NAME = "qmul-survface-v1-official-open-set-identification"


def _emit(progress: ProgressCallback | None, message: str, **details: object) -> None:
    if progress is not None:
        progress(message, details)


def _canonical_path(value: object) -> str:
    return os.path.normcase(str(Path(str(value)).expanduser().resolve(strict=False)))


def _l2_normalize(vector: object) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("embedding must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("embedding norm must be positive")
    return values / norm


def _embedding_model(compression_profile: str):
    profile = str(compression_profile)
    if profile == ORIGIN_512:
        return Embedding512, 512
    if not profile.startswith("pca_"):
        raise ValueError("compression_profile must be origin_512 or a PCA profile")
    dimension = pca_profile_dimension(profile, allow_legacy=True)
    try:
        return PCA_EMBEDDING_MODELS[dimension], dimension
    except KeyError as exc:
        raise ValueError(
            f"{profile} has no PostgreSQL embedding table"
        ) from exc


def _load_db_embeddings(
    engine: Engine,
    *,
    run_uid: str,
    compression_profile: str,
    required_paths: Iterable[str],
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    model, dimension = _embedding_model(compression_profile)
    required = {_canonical_path(path) for path in required_paths}
    records: list[dict[str, object]] = []
    with session_scope(engine) as session:
        query = (
            session.query(model, Image)
            .join(Image, model.image_id == Image.id)
            .filter(
                model.run_uid == str(run_uid),
                model.vector_type == str(compression_profile),
            )
        )
        total_rows = int(query.count())
        for scanned, (embedding, image) in enumerate(
            query.yield_per(int(batch_size)),
            start=1,
        ):
            canonical = _canonical_path(image.image_path)
            if canonical not in required:
                pass
            else:
                records.append(
                    {
                        "image_path_key": canonical,
                        "embedding": _l2_normalize(embedding.embedding),
                        "parameters": dict(embedding.parameters or {}),
                    }
                )
            if scanned % int(batch_size) == 0 or scanned == total_rows:
                _emit(
                    progress,
                    "SurvFace DB embedding load",
                    processed=scanned,
                    total=total_rows,
                    matched=len(records),
                    compression_profile=compression_profile,
                )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError(
            f"no {compression_profile} embeddings were found for run {run_uid}"
        )
    if frame["image_path_key"].duplicated().any():
        duplicates = frame.loc[
            frame["image_path_key"].duplicated(keep=False), "image_path_key"
        ].head(5)
        raise ValueError(f"duplicate database embeddings: {duplicates.tolist()}")
    observed_dimension = {len(value) for value in frame["embedding"]}
    if observed_dimension != {dimension}:
        raise ValueError(
            f"{compression_profile} embedding dimension mismatch: "
            f"{sorted(observed_dimension)} != {[dimension]}"
        )
    return frame


def _attach_embeddings(
    manifest: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> pd.DataFrame:
    selected = manifest.copy()
    selected["image_path_key"] = selected["image_path"].map(_canonical_path)
    lookup = embeddings.set_index("image_path_key")
    missing = selected.loc[
        ~selected["image_path_key"].isin(lookup.index),
        ["image_id", "image_path"],
    ]
    if not missing.empty:
        examples = missing.head(5).to_dict(orient="records")
        raise RuntimeError(
            f"{len(missing)} official SurvFace images have no database embedding; "
            f"examples={examples}"
        )
    selected["embedding"] = selected["image_path_key"].map(lookup["embedding"])
    return selected


def _model_uid(
    embeddings: pd.DataFrame,
    *,
    run_uid: str,
    compression_profile: str,
) -> str:
    candidates = {
        str(parameters.get("model_uid") or parameters.get("model"))
        for parameters in embeddings["parameters"]
        if parameters.get("model_uid") or parameters.get("model")
    }
    if len(candidates) > 1:
        raise ValueError(f"multiple embedding models are mixed: {sorted(candidates)}")
    source = next(iter(candidates), f"run-{run_uid}")
    return f"survface:{source}:{compression_profile}"


def build_survface_official_templates(
    gallery: pd.DataFrame,
) -> pd.DataFrame:
    required = {"identity_id", "image_id", "embedding"}
    missing = required.difference(gallery.columns)
    if missing:
        raise ValueError(f"gallery is missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for identity_id, rows in gallery.groupby("identity_id", sort=False):
        matrix = np.stack(rows["embedding"]).astype(np.float32)
        template = _l2_normalize(matrix.mean(axis=0))
        records.append(
            {
                "identity_id": str(identity_id),
                "embedding": template,
                "source_image_ids": rows["image_id"].astype(str).tolist(),
                "enrollment_count": int(len(rows)),
                "variance": float(
                    np.mean(np.sum((matrix - template[None, :]) ** 2, axis=1))
                ),
            }
        )
    templates = pd.DataFrame.from_records(records)
    if templates.empty:
        raise ValueError("official gallery produced no identity templates")
    return templates


def _template_scope(
    *,
    run_uid: str,
    compression_profile: str,
    model_uid: str,
    enrollment_policy: str,
    enrollment_target: int,
) -> dict[str, object]:
    return {
        "run_uid": str(run_uid),
        "protocol_name": SURVFACE_PROTOCOL_NAME,
        "vector_type": str(compression_profile),
        "aggregation_method": "mean",
        "enrollment_policy": str(enrollment_policy),
        "enrollment_target": int(enrollment_target),
        "model_uid": str(model_uid),
    }


def _materialize_templates(
    engine: Engine,
    *,
    templates: pd.DataFrame,
    scope: dict[str, object],
    compression_profile: str,
    progress: ProgressCallback | None,
) -> dict[str, int]:
    _, dimension = _embedding_model(compression_profile)
    counts = {"inserted": 0, "skipped": 0}
    with session_scope(engine) as session:
        repository = VectorRepository(session)
        for index, row in enumerate(templates.itertuples(index=False), start=1):
            values = {
                **scope,
                "identity_id": str(row.identity_id),
                "source_image_ids": list(row.source_image_ids),
                "enrollment_count": int(row.enrollment_count),
                "embedding": np.asarray(row.embedding, dtype=np.float32).tolist(),
                "quality": None,
                "variance": float(row.variance),
                "angular_error": 0.0 if compression_profile == ORIGIN_512 else None,
                "reconstruction_error_norm": (
                    0.0 if compression_profile == ORIGIN_512 else None
                ),
                "parameters": {
                    "dataset": "qmul-survface-v1",
                    "gallery_policy": "official_all",
                },
            }
            if compression_profile == ORIGIN_512:
                _, action = repository.upsert_template_512(**values)
            else:
                _, action = repository.upsert_pca_template(dimension, **values)
            counts[action] += 1
            if index % 250 == 0:
                _emit(
                    progress,
                    "SurvFace official templates materialized",
                    processed=index,
                    total=len(templates),
                )
        _emit(
            progress,
            "SurvFace official templates materialized",
            processed=len(templates),
            total=len(templates),
        )
    ensure_vector_indexes(engine)
    return counts


def _search_method(
    repository: VectorRepository,
    compression_profile: str,
):
    if compression_profile == ORIGIN_512:
        return repository.find_similar_templates_512
    dimension = pca_profile_dimension(compression_profile, allow_legacy=True)
    return lambda query, **kwargs: repository.find_similar_pca_templates(
        dimension, query, **kwargs
    )


def run_survface_official_search(
    engine: Engine,
    *,
    run_uid: str,
    manifest: pd.DataFrame,
    compression_profile: str,
    search_mode: str,
    top_k: int,
    enrollment_policy: str,
    enrollment_target: int,
    output_path: str | Path,
    batch_size: int = 256,
    probe_limit_per_role: int | None = None,
    progress: ProgressCallback | None = None,
    _prepared_profile_cache: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Materialize official-all templates and search SurvFace in official order."""

    if not str(run_uid).strip():
        raise ValueError("run_uid must not be empty")
    if search_mode not in {"exact", "hnsw"}:
        raise ValueError("search_mode must be exact or hnsw")
    if isinstance(top_k, bool) or int(top_k) < 1:
        raise ValueError("top_k must be positive")
    if isinstance(batch_size, bool) or int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    if enrollment_policy != "official_all" or int(enrollment_target) != 0:
        raise ValueError(
            "SurvFace official search requires official_all and enrollment_target=0"
        )
    if probe_limit_per_role is not None and (
        isinstance(probe_limit_per_role, bool) or int(probe_limit_per_role) < 1
    ):
        raise ValueError("probe_limit_per_role must be positive or None")

    profile_key = str(compression_profile)
    protocol = build_survface_official_protocol(manifest)
    if not protocol.known_unknown_probes.empty:
        raise ValueError("SurvFace official protocol must not contain known_unknown")
    official = pd.concat(
        [
            protocol.gallery,
            protocol.registered_probes,
            protocol.unknown_unknown_probes,
        ],
        ignore_index=True,
        sort=False,
    )
    cached = (
        _prepared_profile_cache.get(profile_key)
        if _prepared_profile_cache is not None
        else None
    )
    if cached is not None:
        embeddings = cached["embeddings"]
        if not isinstance(embeddings, pd.DataFrame):
            raise TypeError("cached SurvFace embeddings are invalid")
    else:
        embeddings = _load_db_embeddings(
            engine,
            run_uid=str(run_uid),
            compression_profile=profile_key,
            required_paths=official["image_path"].astype(str),
            batch_size=int(batch_size),
            progress=progress,
        )
        if _prepared_profile_cache is not None:
            _prepared_profile_cache[profile_key] = {
                "embeddings": embeddings,
            }
    official = _attach_embeddings(official, embeddings)
    gallery = official.loc[official["protocol_role"].eq("gallery")].copy()
    templates = build_survface_official_templates(gallery)
    if len(templates) < int(top_k):
        raise ValueError("top_k exceeds the official gallery identity count")
    model_uid = _model_uid(
        embeddings,
        run_uid=str(run_uid),
        compression_profile=str(compression_profile),
    )
    scope = _template_scope(
        run_uid=str(run_uid),
        compression_profile=str(compression_profile),
        model_uid=model_uid,
        enrollment_policy=enrollment_policy,
        enrollment_target=int(enrollment_target),
    )
    template_actions = _materialize_templates(
        engine,
        templates=templates,
        scope=scope,
        compression_profile=str(compression_profile),
        progress=progress,
    )

    role_frames = [
        (
            "registered",
            official.loc[official["protocol_role"].eq("registered_probe")].copy(),
        ),
        (
            "unknown_unknown",
            official.loc[
                official["protocol_role"].eq("unknown_unknown_probe")
            ].copy(),
        ),
    ]
    if probe_limit_per_role is not None:
        role_frames = [
            (role, frame.head(int(probe_limit_per_role)).copy())
            for role, frame in role_frames
        ]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "probe_type",
        "protocol_index",
        "query_id",
        "query_identity_id",
        "ranked_identities",
        "ranked_distances",
        "query_elapsed_ms",
        "compression_profile",
        "search_mode",
        "model_uid",
    ]
    total = sum(len(frame) for _, frame in role_frames)
    written = 0
    latency_sum = 0.0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            with session_scope(engine) as session:
                repository = VectorRepository(session)
                search = _search_method(repository, str(compression_profile))
                for probe_type, frame in role_frames:
                    ordered = frame.sort_values("protocol_index", kind="stable")
                    for row in ordered.itertuples(index=False):
                        ranked = search(
                            np.asarray(row.embedding, dtype=np.float32),
                            **scope,
                            top_k=int(top_k),
                            search_mode=str(search_mode),
                        )
                        if len(ranked) != int(top_k):
                            raise RuntimeError(
                                f"expected top-{top_k}, received {len(ranked)} rows"
                            )
                        elapsed = float(ranked[0]["query_elapsed_ms"])
                        latency_sum += elapsed
                        writer.writerow(
                            {
                                "probe_type": probe_type,
                                "protocol_index": int(row.protocol_index),
                                "query_id": str(row.image_id),
                                "query_identity_id": str(row.identity_id),
                                "ranked_identities": json.dumps(
                                    [str(item["identity_id"]) for item in ranked],
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                                "ranked_distances": json.dumps(
                                    [float(item["distance"]) for item in ranked],
                                    separators=(",", ":"),
                                ),
                                "query_elapsed_ms": elapsed,
                                "compression_profile": str(compression_profile),
                                "search_mode": str(search_mode),
                                "model_uid": model_uid,
                            }
                        )
                        written += 1
                        if written % int(batch_size) == 0:
                            handle.flush()
                            _emit(
                                progress,
                                "SurvFace official probes searched",
                                processed=written,
                                total=total,
                            )
            _emit(
                progress,
                "SurvFace official probes searched",
                processed=written,
                total=total,
            )
        temporary_path.replace(destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    counts = {role: int(len(frame)) for role, frame in role_frames}
    return {
        "output_path": str(destination),
        "rows": int(written),
        "probe_counts": counts,
        "gallery_image_count": int(len(gallery)),
        "gallery_identity_count": int(len(templates)),
        "template_actions": template_actions,
        "compression_profile": str(compression_profile),
        "search_mode": str(search_mode),
        "top_k": int(top_k),
        "model_uid": model_uid,
        "mean_query_elapsed_ms": (
            float(latency_sum / written) if written else None
        ),
        "official_order_preserved": True,
        "known_unknown_count": 0,
    }


def run_survface_official_search_matrix(
    engine: Engine,
    *,
    run_uid: str,
    manifest: pd.DataFrame,
    compression_profiles: Iterable[str],
    search_modes: Iterable[str],
    top_k: int,
    enrollment_policy: str,
    enrollment_target: int,
    output_path: str | Path,
    batch_size: int = 256,
    probe_limit_per_role: int | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run the complete profile/mode matrix into one official-order CSV."""

    profiles = tuple(str(value) for value in compression_profiles)
    modes = tuple(str(value) for value in search_modes)
    if not profiles or len(set(profiles)) != len(profiles):
        raise ValueError("compression_profiles must be non-empty and unique")
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("search_modes must be non-empty and unique")
    invalid_modes = sorted(set(modes).difference({"exact", "hnsw"}))
    if invalid_modes:
        raise ValueError(f"unsupported search modes: {invalid_modes}")

    protocol = build_survface_official_protocol(manifest)
    expected_counts = {
        "registered": int(len(protocol.registered_probes)),
        "unknown_unknown": int(len(protocol.unknown_unknown_probes)),
    }
    if probe_limit_per_role is None:
        expected_run_counts = expected_counts
    else:
        expected_run_counts = {
            role: min(count, int(probe_limit_per_role))
            for role, count in expected_counts.items()
        }
    combinations = [
        (profile, mode) for profile in profiles for mode in modes
    ]
    profile_indices = {
        profile: index for index, profile in enumerate(profiles)
    }
    prepared_profile_cache: dict[str, dict[str, object]] = {}
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    def matrix_progress(
        combination_index: int,
        profile: str,
        mode: str,
    ) -> ProgressCallback:
        def report(message: str, details: dict[str, object]) -> None:
            payload = dict(details)
            if "processed" in payload and "total" in payload:
                local_processed = int(payload.pop("processed"))
                local_total = int(payload.pop("total"))
                if message == "SurvFace DB embedding load":
                    unit_index = profile_indices[profile]
                    unit_count = len(profiles)
                else:
                    unit_index = combination_index
                    unit_count = len(combinations)
                progress_details = {
                    "processed": unit_index * local_total + local_processed,
                    "total": unit_count * local_total,
                    "compression_profile": profile,
                    "search_mode": mode,
                    **payload,
                }
            else:
                progress_details = {
                    "compression_profile": profile,
                    "search_mode": mode,
                    **payload,
                }
            _emit(progress, message, **progress_details)

        return report

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.stem}-matrix-",
        dir=destination.parent,
    ) as temporary_dir:
        temporary_root = Path(temporary_dir)
        for combination_index, (profile, mode) in enumerate(combinations):
            part_path = temporary_root / f"{profile}_{mode}.csv"
            summary = run_survface_official_search(
                engine,
                run_uid=run_uid,
                manifest=manifest,
                compression_profile=profile,
                search_mode=mode,
                top_k=top_k,
                enrollment_policy=enrollment_policy,
                enrollment_target=enrollment_target,
                output_path=part_path,
                batch_size=batch_size,
                probe_limit_per_role=probe_limit_per_role,
                progress=(
                    matrix_progress(combination_index, profile, mode)
                    if progress is not None
                    else None
                ),
                _prepared_profile_cache=prepared_profile_cache,
            )
            frame = pd.read_csv(part_path)
            actual_counts = {
                str(role): int(count)
                for role, count in frame["probe_type"].value_counts().items()
            }
            official_complete = bool(
                probe_limit_per_role is None
                and actual_counts == expected_counts
            )
            if actual_counts != expected_run_counts:
                raise RuntimeError(
                    f"SurvFace search coverage mismatch for {profile}/{mode}: "
                    f"{actual_counts} != {expected_run_counts}"
                )
            frame["official_complete"] = official_complete
            frame["metric_label"] = (
                "qmul-survface-v1-official-order-cosine-pgvector-adaptation"
            )
            frames.append(frame)
            summaries.append(
                {
                    **summary,
                    "actual_counts": actual_counts,
                    "expected_counts": expected_counts,
                    "official_complete": official_complete,
                }
            )

    combined = pd.concat(frames, ignore_index=True)
    temporary_output = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        combined.to_csv(
            temporary_output,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        os.replace(temporary_output, destination)
    finally:
        temporary_output.unlink(missing_ok=True)
    return {
        "output_path": str(destination),
        "rows": int(len(combined)),
        "combinations": summaries,
        "compression_profiles": list(profiles),
        "search_modes": list(modes),
        "combination_count": len(combinations),
        "expected_counts_per_combination": expected_counts,
        "all_official_complete": all(
            bool(summary["official_complete"]) for summary in summaries
        ),
        "known_unknown_count": 0,
        "official_order_preserved": True,
    }
