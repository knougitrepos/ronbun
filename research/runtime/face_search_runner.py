from __future__ import annotations

import argparse
import shutil
import json
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from research.runtime.hashing import canonical_sha256, sha256_file
from research.runtime.redaction import redact
from research.runtime.run_store import RunStore
from research.search.open_set import (
    CERTIFICATION_CANDIDATE_SCOPES,
    build_certified_search_features,
    summarize_certified_search_features,
)

PHASES = ("protocol", "templates", "compression", "search", "certification", "calibration")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    return config


def hash_config(config: dict[str, Any]) -> str:
    return canonical_sha256(redact(config))


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def csv_row_count(path: Path) -> int:
    return int(len(pd.read_csv(path)))


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise ValueError("certification_global_claim must be a boolean value")
    if pd.isna(value):
        raise ValueError("certification_global_claim must be a boolean value")
    if value == True:
        return True
    if value == False:
        return False
    if isinstance(value, Number) and value in {0, 1}:
        return bool(value)
    raise ValueError("certification_global_claim must be a boolean value")


def _positive_integer_values(values: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(values, errors="coerce")
    finite = (
        numbers.notna()
        & (numbers != float("inf"))
        & (numbers != float("-inf"))
    )
    if (
        (not finite.all())
        or (numbers <= 0).any()
        or ((numbers % 1) != 0).any()
    ):
        raise ValueError(
            "certification_candidate_count and certification_gallery_size "
            "must be positive integers"
        )
    return numbers.astype(int)


def certified_feature_scope_metadata(features: pd.DataFrame) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if "certification_candidate_scope" not in features.columns:
        return metadata

    scopes = features["certification_candidate_scope"].astype(str)
    unknown_scopes = sorted(set(scopes).difference(CERTIFICATION_CANDIDATE_SCOPES))
    if unknown_scopes:
        raise ValueError(
            f"unknown certification_candidate_scope values: {unknown_scopes}"
        )

    candidate_set_rows = scopes == "candidate_set"
    exhaustive_rows = scopes == "exhaustive"
    if candidate_set_rows.any():
        required = {
            "certification_candidate_count",
            "certification_gallery_size",
            "certification_global_claim",
        }
        missing = sorted(required.difference(features.columns))
        if missing:
            raise ValueError(
                "certification_gallery_size is required when "
                "certification_candidate_scope contains candidate_set; "
                "certification_candidate_count and certification_global_claim "
                "are also required; "
                f"missing columns: {missing}"
            )
        candidate_features = features.loc[candidate_set_rows]
        candidate_counts = _positive_integer_values(
            candidate_features["certification_candidate_count"]
        )
        gallery_sizes = _positive_integer_values(
            candidate_features["certification_gallery_size"]
        )
        if (gallery_sizes < candidate_counts).any():
            raise ValueError(
                "certification_gallery_size must be at least "
                "certification_candidate_count"
            )
        global_claims = candidate_features["certification_global_claim"].apply(
            _bool_value
        )
        if global_claims.any():
            raise ValueError(
                "candidate_set rows cannot set certification_global_claim to true"
            )
    if exhaustive_rows.any():
        required = {
            "certification_candidate_count",
            "certification_gallery_size",
            "certification_global_claim",
        }
        if required.issubset(features.columns):
            exhaustive_features = features.loc[exhaustive_rows]
            candidate_counts = _positive_integer_values(
                exhaustive_features["certification_candidate_count"]
            )
            gallery_sizes = _positive_integer_values(
                exhaustive_features["certification_gallery_size"]
            )
            if (gallery_sizes != candidate_counts).any():
                raise ValueError(
                    "exhaustive rows must have certification_gallery_size "
                    "equal to certification_candidate_count"
                )
            global_claims = exhaustive_features["certification_global_claim"].apply(
                _bool_value
            )
            if (~global_claims).any():
                raise ValueError(
                    "exhaustive rows must set certification_global_claim to true"
                )

    scope_counts = {
        str(scope): int(count)
        for scope, count in scopes.value_counts(sort=False).items()
    }
    metadata["certification_candidate_scope_counts"] = scope_counts
    metadata["certification_candidate_scope"] = (
        next(iter(scope_counts)) if len(scope_counts) == 1 else "mixed"
    )

    for column in ["certification_candidate_count", "certification_gallery_size"]:
        if column in features.columns and features[column].nunique(dropna=False) == 1:
            metadata[column] = int(features[column].iloc[0])

    if (
        "certification_global_claim" in features.columns
        and features["certification_global_claim"].nunique(dropna=False) == 1
    ):
        metadata["certification_global_claim"] = _bool_value(
            features["certification_global_claim"].iloc[0]
        )
    return metadata


def certification_method_payload(certification: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_name": "angular_error_bound_open_set",
        "score_type": "cosine_similarity",
        "angular_error_unit": "radian",
        "threshold": float(certification["threshold"]),
        "fallback_profile": str(certification.get("fallback_profile", "origin_512")),
        "assumptions": [
            "query and template embeddings are treated as unit vectors before cosine scoring",
            "template angular_error upper-bounds the angle between compressed and reference template vectors",
            "query_angular_error is zero unless the probe row explicitly provides angular_error",
            "bounds are certified over the supplied candidate vectors only",
        ],
        "bound_formula": {
            "approximate_angle": "arccos(clip(dot(query, compressed_template), -1, 1))",
            "total_angular_error": "template_angular_error + query_angular_error",
            "lower_bound": "cos(min(pi, approximate_angle + total_angular_error))",
            "upper_bound": "cos(max(0, approximate_angle - total_angular_error))",
        },
        "decision_rules": {
            "reject": "max(upper_bounds) < threshold",
            "accept": (
                "lower_bound[top_compressed_candidate] >= threshold and "
                "lower_bound[top_compressed_candidate] > max(upper_bounds[other_candidates])"
            ),
            "defer": "all remaining cases",
        },
        "fallback_rule": (
            "defer rows may be resolved by exact full-precision cosine scoring "
            "when template fallback_embedding values are available and the query "
            "is either supplied as fallback_embedding or has zero query_angular_error"
        ),
        "candidate_scope_caveat": (
            "candidate_set certificates are valid only over the supplied candidate vectors; "
            "approximate pgvector/HNSW candidate recall must be reported separately for full-gallery claims"
        ),
    }


def expand_phases(phase: str) -> list[str]:
    if phase == "all":
        return list(PHASES)
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    return [phase]


def run_root(config: dict[str, Any]) -> Path:
    run_cfg = config.get("run", {})
    return Path(run_cfg.get("root", run_cfg.get("artifact_root", "runs")))


def validate_config(config: dict[str, Any], *, dry_run: bool) -> None:
    required_sections = {"run", "dataset", "protocol", "compression", "certification", "calibration"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    certification = config["certification"]
    if certification.get("enabled", True):
        threshold = certification.get("threshold")
        if threshold is None:
            raise ValueError("certification.threshold is required when certification is enabled")
        threshold_value = float(threshold)
        if threshold_value < -1.0 or threshold_value > 1.0:
            raise ValueError("certification.threshold must be a cosine threshold in [-1, 1]")
    if not dry_run:
        manifest = Path(config["dataset"]["manifest_path"])
        if not manifest.exists():
            raise FileNotFoundError(f"manifest file not found: {manifest}")


def write_run_artifacts(config: dict[str, Any], config_hash: str, out_dir: Path) -> None:
    """Write a redacted, human-readable copy beside the immutable run manifest."""
    payload = {"config_hash": config_hash, "config": redact(config)}
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_vector_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "embedding" not in frame.columns:
        raise ValueError(f"missing embedding column in vector CSV: {path}")
    for vector_column in ["embedding", "fallback_embedding"]:
        if vector_column in frame.columns:
            frame[vector_column] = frame[vector_column].apply(
                lambda value: json.loads(value) if isinstance(value, str) else value
            )
    return frame


def write_phase_artifacts(
    config: dict[str, Any],
    *,
    config_hash: str,
    phases: list[str],
    out_dir: Path,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    written_metadata: list[dict[str, Any]] = []
    for phase in phases:
        phase_dir = out_dir / "artifacts" / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []
        metadata_extra: dict[str, Any] = {}
        if phase == "search":
            search_config = config.get("search", {})
            input_features_path = search_config.get("input_certified_features_path")
            if input_features_path:
                source = Path(str(input_features_path))
                if not source.exists():
                    raise FileNotFoundError(f"search certified feature CSV not found: {source}")
                destination = phase_dir / "certified_features.csv"
                shutil.copyfile(source, destination)
                outputs.append("certified_features.csv")
                copied_features = pd.read_csv(destination)
                metadata_extra["certified_features_rows"] = int(len(copied_features))
                metadata_extra["certified_features_sha256"] = file_sha256(destination)
                metadata_extra.update(certified_feature_scope_metadata(copied_features))
            elif search_config.get("probes_path") and search_config.get("templates_path"):
                probes_path = Path(str(search_config["probes_path"]))
                templates_path = Path(str(search_config["templates_path"]))
                if not probes_path.exists():
                    raise FileNotFoundError(f"search probes CSV not found: {probes_path}")
                if not templates_path.exists():
                    raise FileNotFoundError(f"search templates CSV not found: {templates_path}")
                features = build_certified_search_features(
                    _read_vector_csv(probes_path),
                    _read_vector_csv(templates_path),
                    compression_profile=str(search_config.get("compression_profile", "origin_512")),
                    threshold=float(config["certification"]["threshold"]),
                    top_k=int(search_config.get("top_k", 2)),
                    candidate_scope=str(search_config.get("candidate_scope", "exhaustive")),
                    gallery_size=(
                        search_config["gallery_size"]
                        if search_config.get("gallery_size") is not None
                        else None
                    ),
                )
                destination = phase_dir / "certified_features.csv"
                features.to_csv(destination, index=False)
                outputs.append("certified_features.csv")
                metadata_extra["certified_features_rows"] = csv_row_count(destination)
                metadata_extra["certified_features_sha256"] = file_sha256(destination)
                metadata_extra.update(certified_feature_scope_metadata(features))

        if phase == "certification":
            outputs.extend(["certification_config.json", "certification_method.json"])
            certification = config["certification"]
            input_features_path = certification.get("input_features_path")
            if not input_features_path:
                default_features = out_dir / "artifacts" / "search" / "certified_features.csv"
                if default_features.exists():
                    input_features_path = str(default_features)
            payload = {
                "enabled": bool(certification.get("enabled", True)),
                "threshold": float(certification["threshold"]),
                "fallback_profile": str(certification.get("fallback_profile", "origin_512")),
                "input_features_path": str(input_features_path) if input_features_path else None,
                "decision_columns": [
                    "certified_decision",
                    "certified_identity",
                    "certified_fallback_required",
                ],
                "bound_columns": [
                    "certified_top1_lower_bound",
                    "certified_top1_upper_bound",
                    "certified_top1_bound_width",
                    "certified_max_upper_bound",
                    "certified_max_other_upper_bound",
                    "certified_top1_threshold_margin",
                    "certified_rank_margin",
                    "certified_reject_margin",
                    "certified_decision_margin",
                ],
                "angular_error_columns": [
                    "certified_query_angular_error",
                    "certified_top1_template_angular_error",
                    "certified_top1_total_angular_error",
                    "certified_top1_approximate_angle",
                ],
                "fallback_columns": [
                    "fallback_used",
                    "fallback_query_source",
                    "fallback_template_source",
                    "fallback_decision",
                    "fallback_identity",
                    "fallback_top1_score",
                ],
                "final_decision_columns": [
                    "final_decision",
                    "final_identity",
                    "final_decision_source",
                ],
                "scope_columns": [
                    "certification_candidate_scope",
                    "certification_candidate_count",
                    "certification_gallery_size",
                    "certification_global_claim",
                ],
            }
            with (phase_dir / "certification_config.json").open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            with (phase_dir / "certification_method.json").open("w", encoding="utf-8") as handle:
                json.dump(certification_method_payload(certification), handle, ensure_ascii=False, indent=2)
            if input_features_path:
                feature_path = Path(str(input_features_path))
                if not feature_path.exists():
                    raise FileNotFoundError(f"certification feature CSV not found: {feature_path}")
                features = pd.read_csv(feature_path)
                summary = summarize_certified_search_features(features)
                outputs.append("certification_summary.json")
                metadata_extra["input_features_rows"] = int(len(features))
                metadata_extra["input_features_sha256"] = file_sha256(feature_path)
                metadata_extra.update(certified_feature_scope_metadata(features))
                with (phase_dir / "certification_summary.json").open("w", encoding="utf-8") as handle:
                    json.dump(summary, handle, ensure_ascii=False, indent=2)

        metadata = {
            "run_id": run_id,
            "phase": phase,
            "status": "completed",
            "config_hash": config_hash,
            "manifest_path": str(config["dataset"]["manifest_path"]),
            "manifest_sha256": file_sha256(Path(config["dataset"]["manifest_path"])),
            "outputs": outputs,
        }
        if phase == "certification":
            input_features_path = config["certification"].get("input_features_path")
            if not input_features_path:
                default_features = out_dir / "artifacts" / "search" / "certified_features.csv"
                if default_features.exists():
                    input_features_path = str(default_features)
            if input_features_path:
                metadata["input_features_path"] = str(input_features_path)
        if phase == "search" and config.get("search", {}).get("input_certified_features_path"):
            metadata["input_certified_features_path"] = str(config["search"]["input_certified_features_path"])
        metadata.update(metadata_extra)
        with (phase_dir / "phase_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        written_metadata.append(metadata)
    return written_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Run compressed face-search research phases.")
    parser.add_argument("--config", required=True, help="Path to face search YAML config.")
    parser.add_argument(
        "--phase",
        choices=("all",) + PHASES,
        default="all",
        help="Phase to run. Use all for the full v1 pipeline.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print planned execution only.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    validate_config(config, dry_run=args.dry_run)
    phases = expand_phases(args.phase)
    config_hash = hash_config(config)

    root = run_root(config)
    run: RunStore | None = None
    if not args.dry_run:
        run = RunStore.create(
            experiment_name=str(config.get("run", {}).get("name", "face_search_study")),
            config=config,
            root=root,
        )
        out_dir = run.run_dir
        try:
            write_run_artifacts(config, config_hash, out_dir)
            configured_inputs: list[tuple[str | Path, str]] = [
                (config_path, "experiment_config"),
                (config["dataset"]["manifest_path"], "dataset_manifest"),
            ]
            for key in ("input_certified_features_path", "probes_path", "templates_path"):
                value = config.get("search", {}).get(key)
                if value:
                    configured_inputs.append((value, f"search_{key}"))
            certification_input = config.get("certification", {}).get("input_features_path")
            if certification_input:
                configured_inputs.append((certification_input, "certification_input_features"))
            seen_inputs: set[Path] = set()
            for input_path, role in configured_inputs:
                resolved = Path(input_path).resolve()
                if resolved in seen_inputs:
                    continue
                run.record_input(resolved, role=role)
                seen_inputs.add(resolved)
            for phase in phases:
                with run.phase(phase) as phase_context:
                    metadata = write_phase_artifacts(
                        config,
                        config_hash=config_hash,
                        phases=[phase],
                        out_dir=out_dir,
                        run_id=run.run_id,
                    )[0]
                    phase_context.details.update(
                        {
                            "metadata_path": str(
                                Path("artifacts") / phase / "phase_metadata.json"
                            ),
                            "outputs": metadata["outputs"],
                        }
                    )
            run.complete()
        except BaseException as error:
            run.fail(error)
            raise
    else:
        out_dir = root

    print(f"dry_run={args.dry_run}")
    print(f"phase={args.phase}")
    print(f"phases={','.join(phases)}")
    if config.get("certification", {}).get("enabled", True):
        print(f"certification_threshold={float(config['certification']['threshold'])}")
    print(f"config_hash={config_hash}")
    print(f"run_root={root}")
    if run is not None:
        print(f"run_id={run.run_id}")
        print(f"artifact_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
