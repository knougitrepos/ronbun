import json
import hashlib
import subprocess
import sys

import pandas as pd
import pytest


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_face_search_cli_dry_run_accepts_all_phase_config():
    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            "experiments/configs/face_search.yaml",
            "--phase",
            "all",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "dry_run=True" in result.stdout
    assert "phase=all" in result.stdout
    assert "phases=protocol,templates,compression,search,certification,calibration" in result.stdout
    assert "certification_threshold=0.8" in result.stdout
    assert "config_hash=" in result.stdout


def test_face_search_cli_writes_certification_phase_artifacts(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: artifact_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "certification",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = artifact_root / "artifact_test"
    phase_dir = run_dir / "certification"
    metadata = json.loads((phase_dir / "phase_metadata.json").read_text(encoding="utf-8"))
    certification_config = json.loads((phase_dir / "certification_config.json").read_text(encoding="utf-8"))
    certification_method = json.loads((phase_dir / "certification_method.json").read_text(encoding="utf-8"))

    assert (run_dir / "run_config.json").is_file()
    assert metadata["phase"] == "certification"
    assert metadata["status"] == "planned"
    assert metadata["config_hash"]
    assert metadata["manifest_path"] == manifest.as_posix()
    assert metadata["outputs"] == ["certification_config.json", "certification_method.json"]
    assert certification_config["threshold"] == 0.82
    assert certification_config["fallback_profile"] == "origin_512"
    assert certification_config["decision_columns"] == [
        "certified_decision",
        "certified_identity",
        "certified_fallback_required",
    ]
    assert certification_config["bound_columns"] == [
        "certified_top1_lower_bound",
        "certified_top1_upper_bound",
        "certified_top1_bound_width",
        "certified_max_upper_bound",
        "certified_max_other_upper_bound",
        "certified_top1_threshold_margin",
        "certified_rank_margin",
        "certified_reject_margin",
        "certified_decision_margin",
    ]
    assert certification_config["angular_error_columns"] == [
        "certified_query_angular_error",
        "certified_top1_template_angular_error",
        "certified_top1_total_angular_error",
        "certified_top1_approximate_angle",
    ]
    assert certification_config["fallback_columns"] == [
        "fallback_used",
        "fallback_query_source",
        "fallback_template_source",
        "fallback_decision",
        "fallback_identity",
        "fallback_top1_score",
    ]
    assert certification_config["final_decision_columns"] == [
        "final_decision",
        "final_identity",
        "final_decision_source",
    ]
    assert certification_config["scope_columns"] == [
        "certification_candidate_scope",
        "certification_candidate_count",
        "certification_gallery_size",
        "certification_global_claim",
    ]
    assert certification_method["method_name"] == "angular_error_bound_open_set"
    assert certification_method["score_type"] == "cosine_similarity"
    assert certification_method["angular_error_unit"] == "radian"
    assert certification_method["bound_formula"]["approximate_angle"] == "arccos(clip(dot(query, compressed_template), -1, 1))"
    assert certification_method["bound_formula"]["lower_bound"] == "cos(min(pi, approximate_angle + total_angular_error))"
    assert certification_method["bound_formula"]["upper_bound"] == "cos(max(0, approximate_angle - total_angular_error))"
    assert certification_method["decision_rules"]["reject"] == "max(upper_bounds) < threshold"
    assert certification_method["decision_rules"]["accept"] == (
        "lower_bound[top_compressed_candidate] >= threshold and "
        "lower_bound[top_compressed_candidate] > max(upper_bounds[other_candidates])"
    )
    assert "candidate_set" in certification_method["candidate_scope_caveat"]


def test_face_search_cli_writes_certification_summary_from_feature_csv(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    features = tmp_path / "certified_features.csv"
    features.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "probe_type",
                        "certified_decision",
                        "certified_fallback_required",
                        "certification_candidate_scope",
                        "certification_candidate_count",
                        "certification_gallery_size",
                        "certification_global_claim",
                    ]
                ),
                "registered,accept,false,candidate_set,10,100,false",
                "known_unknown,reject,false,candidate_set,10,100,false",
                "unknown_unknown,defer,true,candidate_set,10,100,false",
                "registered,accept,false,candidate_set,10,100,false",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: summary_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                f"  input_features_path: {features.as_posix()}",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "certification",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    phase_dir = artifact_root / "summary_test" / "certification"
    metadata = json.loads((phase_dir / "phase_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((phase_dir / "certification_summary.json").read_text(encoding="utf-8"))

    assert metadata["outputs"] == [
        "certification_config.json",
        "certification_method.json",
        "certification_summary.json",
    ]
    assert metadata["input_features_path"] == features.as_posix()
    assert metadata["certification_candidate_scope"] == "candidate_set"
    assert metadata["certification_candidate_scope_counts"] == {"candidate_set": 4}
    assert metadata["certification_candidate_count"] == 10
    assert metadata["certification_gallery_size"] == 100
    assert metadata["certification_global_claim"] is False
    assert summary["total"] == 4
    assert summary["decision_counts"] == {"accept": 2, "reject": 1, "defer": 1}
    assert summary["certification_coverage"] == 0.75
    assert summary["fallback_rate"] == 0.25
    assert summary["candidate_scope_counts"] == {"candidate_set": 4}
    assert summary["by_probe_type"]["registered"]["decision_counts"] == {
        "accept": 2,
        "reject": 0,
        "defer": 0,
    }


def test_face_search_cli_hands_search_certified_features_to_certification_by_default(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    source_features = tmp_path / "search_features.csv"
    source_features.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "probe_type",
                        "certified_decision",
                        "certified_fallback_required",
                        "certification_candidate_scope",
                        "certification_candidate_count",
                        "certification_gallery_size",
                        "certification_global_claim",
                    ]
                ),
                "registered,accept,false,candidate_set,20,200,false",
                "known_unknown,reject,false,candidate_set,20,200,false",
                "unknown_unknown,defer,true,candidate_set,20,200,false",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: handoff_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "search:",
                f"  input_certified_features_path: {source_features.as_posix()}",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = artifact_root / "handoff_test"
    search_features = run_dir / "search" / "certified_features.csv"
    search_metadata = json.loads((run_dir / "search" / "phase_metadata.json").read_text(encoding="utf-8"))
    certification_metadata = json.loads(
        (run_dir / "certification" / "phase_metadata.json").read_text(encoding="utf-8")
    )
    certification_summary = json.loads(
        (run_dir / "certification" / "certification_summary.json").read_text(encoding="utf-8")
    )

    assert search_features.read_text(encoding="utf-8") == source_features.read_text(encoding="utf-8")
    assert search_metadata["certified_features_rows"] == 3
    assert search_metadata["certified_features_sha256"] == _sha256(search_features)
    assert search_metadata["certification_candidate_scope"] == "candidate_set"
    assert search_metadata["certification_candidate_scope_counts"] == {"candidate_set": 3}
    assert search_metadata["certification_candidate_count"] == 20
    assert search_metadata["certification_gallery_size"] == 200
    assert search_metadata["certification_global_claim"] is False
    assert certification_metadata["input_features_path"] == str(search_features)
    assert certification_metadata["input_features_rows"] == 3
    assert certification_metadata["input_features_sha256"] == _sha256(search_features)
    assert certification_metadata["certification_candidate_scope"] == "candidate_set"
    assert certification_metadata["certification_candidate_scope_counts"] == {"candidate_set": 3}
    assert certification_metadata["certification_candidate_count"] == 20
    assert certification_metadata["certification_gallery_size"] == 200
    assert certification_metadata["certification_global_claim"] is False
    assert certification_summary["total"] == 3
    assert certification_summary["decision_counts"] == {"accept": 1, "reject": 1, "defer": 1}
    assert certification_summary["candidate_scope_counts"] == {"candidate_set": 3}


def test_face_search_cli_rejects_precomputed_candidate_set_without_gallery_size(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    features = tmp_path / "certified_features.csv"
    features.write_text(
        "\n".join(
            [
                "probe_type,certified_decision,certified_fallback_required,certification_candidate_scope",
                "registered,accept,false,candidate_set",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: invalid_precomputed_certification_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                f"  input_features_path: {features.as_posix()}",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "certification",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "certification_gallery_size is required when certification_candidate_scope contains candidate_set" in result.stderr


def test_face_search_cli_rejects_search_handoff_candidate_set_without_gallery_size(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    source_features = tmp_path / "search_features.csv"
    source_features.write_text(
        "\n".join(
            [
                "probe_type,certified_decision,certified_fallback_required,certification_candidate_scope",
                "registered,accept,false,candidate_set",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: invalid_precomputed_search_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "search:",
                f"  input_certified_features_path: {source_features.as_posix()}",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "certification_gallery_size is required when certification_candidate_scope contains candidate_set" in result.stderr


def test_face_search_cli_rejects_precomputed_unknown_candidate_scope(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    features = tmp_path / "certified_features.csv"
    features.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "probe_type",
                        "certified_decision",
                        "certified_fallback_required",
                        "certification_candidate_scope",
                        "certification_candidate_count",
                        "certification_gallery_size",
                        "certification_global_claim",
                    ]
                ),
                "registered,accept,false,approximate_gallery,10,100,false",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: invalid_scope_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                f"  input_features_path: {features.as_posix()}",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "certification",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unknown certification_candidate_scope values" in result.stderr


def test_face_search_cli_rejects_precomputed_exhaustive_scope_with_partial_gallery(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    features = tmp_path / "certified_features.csv"
    features.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "probe_type",
                        "certified_decision",
                        "certified_fallback_required",
                        "certification_candidate_scope",
                        "certification_candidate_count",
                        "certification_gallery_size",
                        "certification_global_claim",
                    ]
                ),
                "registered,accept,false,exhaustive,10,100,false",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: invalid_exhaustive_scope_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                f"  input_features_path: {features.as_posix()}",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "certification",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "exhaustive rows must have certification_gallery_size equal to "
        "certification_candidate_count"
    ) in result.stderr


def test_face_search_cli_rejects_precomputed_candidate_set_global_claim(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    features = tmp_path / "certified_features.csv"
    features.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "probe_type",
                        "certified_decision",
                        "certified_fallback_required",
                        "certification_candidate_scope",
                        "certification_candidate_count",
                        "certification_gallery_size",
                        "certification_global_claim",
                    ]
                ),
                "registered,accept,false,candidate_set,10,100,true",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: invalid_global_claim_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                f"  input_features_path: {features.as_posix()}",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "certification",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "candidate_set rows cannot set certification_global_claim to true" in result.stderr


def test_face_search_cli_rejects_precomputed_invalid_global_claim_value(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    features = tmp_path / "certified_features.csv"
    features.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "probe_type",
                        "certified_decision",
                        "certified_fallback_required",
                        "certification_candidate_scope",
                        "certification_candidate_count",
                        "certification_gallery_size",
                        "certification_global_claim",
                    ]
                ),
                "registered,accept,false,candidate_set,10,100,maybe",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: invalid_global_claim_value_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                f"  input_features_path: {features.as_posix()}",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "certification",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "certification_global_claim must be a boolean value" in result.stderr


def test_face_search_cli_rejects_precomputed_gallery_smaller_than_candidate_count(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    features = tmp_path / "certified_features.csv"
    features.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "probe_type",
                        "certified_decision",
                        "certified_fallback_required",
                        "certification_candidate_scope",
                        "certification_candidate_count",
                        "certification_gallery_size",
                        "certification_global_claim",
                    ]
                ),
                "registered,accept,false,candidate_set,100,10,false",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: invalid_gallery_count_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                f"  input_features_path: {features.as_posix()}",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "certification",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "certification_gallery_size must be at least certification_candidate_count" in result.stderr


@pytest.mark.parametrize(
    ("candidate_count", "gallery_size"),
    [
        ("0", "100"),
        ("10.5", "100"),
    ],
)
def test_face_search_cli_rejects_precomputed_non_positive_or_fractional_candidate_counts(
    tmp_path,
    candidate_count,
    gallery_size,
):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    features = tmp_path / "certified_features.csv"
    features.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "probe_type",
                        "certified_decision",
                        "certified_fallback_required",
                        "certification_candidate_scope",
                        "certification_candidate_count",
                        "certification_gallery_size",
                        "certification_global_claim",
                    ]
                ),
                f"registered,accept,false,candidate_set,{candidate_count},{gallery_size},false",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: invalid_candidate_count_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "certification:",
                "  enabled: true",
                "  threshold: 0.82",
                "  fallback_profile: origin_512",
                f"  input_features_path: {features.as_posix()}",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "certification",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert (
        "certification_candidate_count and certification_gallery_size "
        "must be positive integers"
    ) in result.stderr


def test_face_search_cli_generates_certified_features_from_probe_and_template_csv(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    probes = tmp_path / "probes.csv"
    templates = tmp_path / "templates.csv"
    pd.DataFrame(
        {
            "image_id": ["qa", "qu"],
            "identity_id": ["a", "u"],
            "probe_type": ["registered", "unknown_unknown"],
            "embedding": ["[1.0, 0.0]", "[-1.0, 0.0]"],
            "quality": [0.7, 0.5],
            "reconstruction_error_norm": [0.0, 0.0],
            "angular_error": [0.0, 0.0],
        }
    ).to_csv(probes, index=False)
    pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": ["[1.0, 0.0]", "[0.0, 1.0]"],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.05, 0.05],
        }
    ).to_csv(templates, index=False)
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: generated_search_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "search:",
                f"  probes_path: {probes.as_posix()}",
                f"  templates_path: {templates.as_posix()}",
                "  compression_profile: pca_2",
                "  top_k: 2",
                "  candidate_scope: candidate_set",
                "  gallery_size: 100",
                "certification:",
                "  enabled: true",
                "  threshold: 0.80",
                "  fallback_profile: origin_512",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = artifact_root / "generated_search_test"
    features = pd.read_csv(run_dir / "search" / "certified_features.csv")
    search_metadata = json.loads((run_dir / "search" / "phase_metadata.json").read_text(encoding="utf-8"))
    generated_features_path = run_dir / "search" / "certified_features.csv"
    summary = json.loads((run_dir / "certification" / "certification_summary.json").read_text(encoding="utf-8"))

    assert list(features["query_id"]) == ["qa", "qu"]
    assert list(features["certified_decision"]) == ["accept", "reject"]
    assert list(features["certified_fallback_required"]) == [False, False]
    assert list(features["certification_candidate_scope"]) == ["candidate_set", "candidate_set"]
    assert list(features["certification_candidate_count"]) == [2, 2]
    assert list(features["certification_gallery_size"]) == [100, 100]
    assert list(features["certification_global_claim"]) == [False, False]
    assert search_metadata["certification_candidate_scope"] == "candidate_set"
    assert search_metadata["certification_candidate_count"] == 2
    assert search_metadata["certification_gallery_size"] == 100
    assert search_metadata["certification_global_claim"] is False
    assert search_metadata["certified_features_rows"] == 2
    assert search_metadata["certified_features_sha256"] == _sha256(generated_features_path)
    assert summary["total"] == 2
    assert summary["decision_counts"] == {"accept": 1, "reject": 1, "defer": 0}
    assert summary["candidate_scope_counts"] == {"candidate_set": 2}


def test_face_search_cli_requires_gallery_size_for_candidate_set_generation(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    probes = tmp_path / "probes.csv"
    templates = tmp_path / "templates.csv"
    pd.DataFrame(
        {
            "image_id": ["qa"],
            "identity_id": ["a"],
            "probe_type": ["registered"],
            "embedding": ["[1.0, 0.0]"],
            "quality": [0.7],
            "reconstruction_error_norm": [0.0],
            "angular_error": [0.0],
        }
    ).to_csv(probes, index=False)
    pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": ["[1.0, 0.0]", "[0.0, 1.0]"],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.05, 0.05],
        }
    ).to_csv(templates, index=False)
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: missing_gallery_size_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "search:",
                f"  probes_path: {probes.as_posix()}",
                f"  templates_path: {templates.as_posix()}",
                "  compression_profile: pca_2",
                "  top_k: 2",
                "  candidate_scope: candidate_set",
                "certification:",
                "  enabled: true",
                "  threshold: 0.80",
                "  fallback_profile: origin_512",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "gallery_size is required when candidate_scope is candidate_set" in result.stderr


def test_face_search_cli_rejects_fractional_gallery_size_for_candidate_set_generation(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    probes = tmp_path / "probes.csv"
    templates = tmp_path / "templates.csv"
    pd.DataFrame(
        {
            "image_id": ["qa"],
            "identity_id": ["a"],
            "probe_type": ["registered"],
            "embedding": ["[1.0, 0.0]"],
            "quality": [0.7],
            "reconstruction_error_norm": [0.0],
            "angular_error": [0.0],
        }
    ).to_csv(probes, index=False)
    pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": ["[1.0, 0.0]", "[0.0, 1.0]"],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.05, 0.05],
        }
    ).to_csv(templates, index=False)
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: fractional_gallery_size_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "search:",
                f"  probes_path: {probes.as_posix()}",
                f"  templates_path: {templates.as_posix()}",
                "  compression_profile: pca_2",
                "  top_k: 2",
                "  candidate_scope: candidate_set",
                "  gallery_size: 100.5",
                "certification:",
                "  enabled: true",
                "  threshold: 0.80",
                "  fallback_profile: origin_512",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "gallery_size must be a positive integer" in result.stderr


def test_face_search_cli_generates_final_decisions_with_fallback_embeddings(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id,identity_id,split,image_path\n", encoding="utf-8")
    probes = tmp_path / "probes.csv"
    templates = tmp_path / "templates.csv"
    pd.DataFrame(
        {
            "image_id": ["qa"],
            "identity_id": ["a"],
            "probe_type": ["registered"],
            "embedding": ["[1.0, 0.0]"],
            "fallback_embedding": ["[1.0, 0.0]"],
            "quality": [0.7],
            "reconstruction_error_norm": [0.0],
            "angular_error": [0.0],
        }
    ).to_csv(probes, index=False)
    pd.DataFrame(
        {
            "identity_id": ["a", "b"],
            "embedding": ["[0.80, 0.60]", "[0.78, 0.62]"],
            "fallback_embedding": ["[1.0, 0.0]", "[0.0, 1.0]"],
            "quality": [0.9, 0.8],
            "variance": [0.01, 0.02],
            "enrollment_count": [2, 1],
            "angular_error": [0.20, 0.20],
        }
    ).to_csv(templates, index=False)
    config = tmp_path / "face_search.yaml"
    artifact_root = tmp_path / "artifacts"
    config.write_text(
        "\n".join(
            [
                "run:",
                "  name: fallback_search_test",
                f"  artifact_root: {artifact_root.as_posix()}",
                "dataset:",
                f"  manifest_path: {manifest.as_posix()}",
                "protocol:",
                "  split_seed: 7",
                "compression:",
                "  profiles: [origin_512, pca_256]",
                "search:",
                f"  probes_path: {probes.as_posix()}",
                f"  templates_path: {templates.as_posix()}",
                "  compression_profile: pca_2",
                "  top_k: 2",
                "certification:",
                "  enabled: true",
                "  threshold: 0.70",
                "  fallback_profile: origin_512",
                "calibration:",
                "  target_fpir: 0.01",
                "output:",
                "  save_csv: true",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            str(config),
            "--phase",
            "all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dir = artifact_root / "fallback_search_test"
    features = pd.read_csv(run_dir / "search" / "certified_features.csv")
    summary = json.loads((run_dir / "certification" / "certification_summary.json").read_text(encoding="utf-8"))

    assert list(features["certified_decision"]) == ["defer"]
    assert list(features["fallback_used"]) == [True]
    assert list(features["fallback_decision"]) == ["accept"]
    assert list(features["final_decision"]) == ["accept"]
    assert list(features["final_decision_source"]) == ["exact_fallback"]
    assert summary["decision_counts"] == {"accept": 0, "reject": 0, "defer": 1}
    assert summary["final_decision_counts"] == {"accept": 1, "reject": 0, "defer": 0}
    assert summary["fallback_resolution_rate"] == 1.0
