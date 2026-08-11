from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"

EXPECTED_NOTEBOOKS = {
    "lfw/00_data_preparation": {
        "00_data_preparation.ipynb",
        "01_aligned_crop_materialization.ipynb",
        "02_landmark_region_materialization.ipynb",
    },
    "lfw/01_embeddings": {
        "00_protocol_and_run_freeze.ipynb",
        "01_arcface_embedding_extraction.ipynb",
    },
    "lfw/02_compression": {
        "00_compressor_fit.ipynb",
        "01_compressed_materialization_and_index.ipynb",
        "02_step1_compression_characterization.ipynb",
    },
    "lfw/03_open_set": {
        "00_probe_search_and_certification.ipynb",
        "01_evaluation_and_visualization.ipynb",
    },
    "lfw/04_gradcam/prerequisite": {
        "00_source_and_model_freeze.ipynb",
        "01_origin_embedding_and_loo_templates.ipynb",
    },
    "lfw/04_gradcam/experiment": {
        "00_population_gradcam_extraction.ipynb",
        "01_saliency_feature_validation.ipynb",
        "02_step2_compression_characterization.ipynb",
        "03_saliency_compression_join.ipynb",
        "04_representative_case_visualization.ipynb",
    },
    "survface/00_data_preparation": {
        "00_data_preparation.ipynb",
        "01_aligned_crop_materialization.ipynb",
        "02_landmark_region_materialization.ipynb",
    },
    "survface/01_embeddings": {
        "00_official_protocol_and_run_freeze.ipynb",
        "01_official_arcface_embedding_extraction.ipynb",
    },
    "survface/02_compression": {
        "00_compressor_fit.ipynb",
        "01_official_compressed_materialization_and_index.ipynb",
        "02_step1_compression_characterization.ipynb",
    },
    "survface/03_open_set": {
        "00_official_probe_search.ipynb",
        "01_official_evaluation_and_visualization.ipynb",
    },
    "survface/04_gradcam/prerequisite": {
        "00_source_and_model_freeze.ipynb",
        "01_origin_embedding_and_top1_gallery_templates.ipynb",
    },
    "survface/04_gradcam/experiment": {
        "00_population_gradcam_extraction.ipynb",
        "01_saliency_feature_validation.ipynb",
        "02_step2_compression_characterization.ipynb",
        "03_saliency_compression_join.ipynb",
        "04_representative_case_visualization.ipynb",
    },
    "rfw/00_data_preparation": {"00_data_preparation.ipynb"},
    "rfw/01_embeddings": {"00_rfw_origin_embedding_extraction.ipynb"},
    "rfw/02_compression": {"00_rfw_frozen_codec_verification.ipynb"},
    "rfw": {"00_rfw_all_in_one.ipynb"},
    "balancedface/00_data_preparation": {"00_data_preparation.ipynb"},
    "common/model_preparation": {
        "00_checkpoint_registration.ipynb",
        "01_preprocessing_and_model_smoke.ipynb",
    },
    "common/reports": {"00_cross_dataset_results.ipynb"},
    "common/maintenance": {"00_selective_cleanup.ipynb"},
    "common/orchestration": {
        "00_batch_experiment_runner.ipynb",
        "cross_dataset_calibration_transfer.ipynb",
    },
}


def _source(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n".join(cell.source for cell in notebook.cells)


def test_notebooks_are_grouped_by_dataset_then_execution_stage() -> None:
    assert not list(NOTEBOOK_ROOT.glob("*.ipynb"))
    assert not (NOTEBOOK_ROOT / "step4").exists()
    actual_directories = {
        path.parent.relative_to(NOTEBOOK_ROOT).as_posix()
        for path in NOTEBOOK_ROOT.rglob("*.ipynb")
    }
    assert actual_directories == set(EXPECTED_NOTEBOOKS)
    for relative, expected_names in EXPECTED_NOTEBOOKS.items():
        directory = NOTEBOOK_ROOT / relative
        assert {path.name for path in directory.glob("*.ipynb")} == expected_names


def test_step4_notebooks_are_dataset_specific_single_stage_runbooks() -> None:
    expected = {
        "00_source_and_model_freeze.ipynb": "freeze_step4_source_and_model",
        "00_population_gradcam_extraction.ipynb": (
            "extract_step4_population_gradcam"
        ),
        "01_saliency_feature_validation.ipynb": "validate_step4_saliency",
        "02_step2_compression_characterization.ipynb": (
            "characterize_step4_compression"
        ),
        "03_saliency_compression_join.ipynb": (
            "analyze_step4_saliency_compression"
        ),
        "04_representative_case_visualization.ipynb": (
            "finalize_step4_representative_cases"
        ),
    }
    origin_names = {
        "lfw": "01_origin_embedding_and_loo_templates.ipynb",
        "survface": "01_origin_embedding_and_top1_gallery_templates.ipynb",
    }
    stage_functions = {
        *expected.values(),
        "extract_step4_origin_embeddings",
    }
    for dataset_id in ("lfw", "survface"):
        root = NOTEBOOK_ROOT / dataset_id / "04_gradcam"
        notebooks = [
            *(root / "prerequisite").glob("*.ipynb"),
            *(root / "experiment").glob("*.ipynb"),
        ]
        for path in notebooks:
            source = _source(path)
            expected_function = (
                "extract_step4_origin_embeddings"
                if path.name == origin_names[dataset_id]
                else expected[path.name]
            )
            assert f'DATASET_ID = "{dataset_id}"' in source
            assert expected_function in source
            assert "execution_acknowledged=True" in source
            assert sum(name in source for name in stage_functions) == 1
            assert "run_step4_experiment" not in source


def test_common_orchestration_notebook_preserves_quick_full_contract() -> None:
    path = (
        NOTEBOOK_ROOT
        / "common"
        / "orchestration"
        / "00_batch_experiment_runner.ipynb"
    )
    source = _source(path)

    for phrase in (
        "DATASET_ID =",
        "RUN_TIER =",
        "QUICK_DATA_FRACTIONS = {",
        '"lfw":',
        '"survface":',
        '"rfw_custom":',
        "TARGET_FPIRS = (0.10, 0.01)",
        "MODEL_NAME =",
        "arcface_ms1mv3_r100",
        "adaface_ms1mv3_r100",
        "magface_ms1mv2_iresnet100",
        "edgeface_webface12m_xs_gamma_06",
        "models/arcface/ms1mv3_r100_backbone.pth",
        "models/adaface/adaface_ir101_ms1mv3.ckpt",
        "models/magface/magface_ms1mv2.pth",
        "models/edgeface/edgeface_xs_gamma_06.pt",
        "FULL_DATA_FRACTION",
        "prepare_common_model_checkpoint",
        "run_smoke_validation=True",
        "build_common_experiment_plan",
            "quick_data_fractions=QUICK_DATA_FRACTIONS",
            'ARTIFACT_STORAGE_MODE = "results_only"',
            "artifact_storage_mode=ARTIFACT_STORAGE_MODE",
        "inspect_common_experiment_plan",
        "run_common_step4_experiment",
        "ACKNOWLEDGE_LOCAL_EXECUTION =",
        'raise RuntimeError(f"preflight 실패: {FAILED_CHECKS}")',
        "milestone_percent=10",
        "heartbeat_seconds=None",
    ):
        assert phrase in source
    assert "DATA_FRACTION =" not in source
    assert "%run" not in source
    for phrase in (
        "DATASET_IDS = (",
        "postprocess_completed_run",
        "reuse_completed_run_for_plan",
        "COMPLETED_RUN_OVERRIDES = {",
        "run_cross_dataset_report_notebook",
        "RUN_SEARCH_SPACE_REFRESH = True",
        "RUN_FAITHFULNESS = True",
        "RUN_FINAL_REPORT = True",
        "START_NEW_RUN =",
        "RUN_RFW_VERIFICATION = False",
        "frozen_codec_specs_from_completed_run",
        "rfw_frozen_codec_evaluation_uid",
        "rfw_evaluation_dir=",
        "CROSS_MODEL_RUN_MATRIX = {",
        "cross_model_run_matrix=",
    ):
        assert phrase in source
    assert source.index("for dataset_id, plan in PLANS.items()") < (
        source.index(
            "for dataset_id, execution in EXECUTION_RESULTS.items()"
        )
    )


def test_survface_saliency_join_runbook_exposes_long_run_progress() -> None:
    path = (
        NOTEBOOK_ROOT
        / "survface"
        / "04_gradcam"
        / "experiment"
        / "03_saliency_compression_join.ipynb"
    )
    source = _source(path)

    assert "ProgressReporter" in source
    assert "heartbeat_seconds=60" in source
    assert '"bootstrap_batch_size": 4' in source
    assert 'bootstrap_rank_strategy": "weighted_rerank"' in source
    assert 'progress=PROGRESS.callback(key_prefix="step4-05:")' in source


def test_all_notebooks_are_valid_restartable_and_output_free() -> None:
    for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        assert notebook.metadata["ronbun"]["restart_policy"] == (
            "restart_kernel_and_run_all"
        )
        assert notebook.metadata["ronbun"]["outputs_committed"] is False
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(cell.source, f"{path.name}:cell-{index}", "exec")
            assert cell.execution_count is None
            assert cell.outputs == []


def test_experiment_defaults_use_full_data_and_execute() -> None:
    for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
        if "maintenance" in path.parts:
            continue
        source = _source(path)
        if "DATA_FRACTION =" in source:
            assert (
                "DATA_FRACTION = 1.0" in source
                or 'DATA_FRACTION = float(EXECUTION["data_fraction"])' in source
            )
        if "EXECUTE_STAGE =" in source:
            assert (
                "EXECUTE_STAGE = True" in source
                or 'EXECUTE_STAGE = bool(EXECUTION["execute_stage"])' in source
            )
        if "WRITE_OUTPUTS =" in source:
            assert (
                "WRITE_OUTPUTS = True" in source
                or 'WRITE_OUTPUTS = bool(EXECUTION["write_outputs"])' in source
            )
        if "OVERWRITE =" in source:
            assert (
                "OVERWRITE = True" in source
                or 'OVERWRITE = bool(EXECUTION["overwrite"])' in source
            )


def test_survface_notebooks_preserve_official_protocol_boundaries() -> None:
    paths = [
        *(NOTEBOOK_ROOT / "survface" / "01_embeddings").glob("*.ipynb"),
        *(NOTEBOOK_ROOT / "survface" / "02_compression").glob("*.ipynb"),
        *(NOTEBOOK_ROOT / "survface" / "03_open_set").glob("*.ipynb"),
    ]
    sources = "\n".join(_source(path) for path in paths)
    for phrase in (
        "official_all",
        "known_unknown",
        "TPIR",
        "FPIR",
        "development",
        "official_test_fit",
    ):
        assert phrase in sources


def test_rfw_notebooks_preserve_frozen_codec_verification_boundary() -> None:
    origin = _source(
        NOTEBOOK_ROOT
        / "rfw"
        / "01_embeddings"
        / "00_rfw_origin_embedding_extraction.ipynb"
    )
    evaluation = _source(
        NOTEBOOK_ROOT
        / "rfw"
        / "02_compression"
        / "00_rfw_frozen_codec_verification.ipynb"
    )
    for phrase in (
        "extract_rfw_origin_embeddings",
        "EXPECTED_ARCHIVE_SHA256",
        'ARTIFACT_STORAGE_MODE = "results_only"',
        "HORIZONTAL_FLIP_TTA = False",
    ):
        assert phrase in origin
    for phrase in (
        "frozen_codec_specs_from_completed_run",
        "evaluate_rfw_frozen_codecs",
        "CODEC_SOURCE_RUN_DIRS",
        'ARTIFACT_STORAGE_MODE = "results_only"',
        "fit_on_rfw=false",
        "DIR@FPIR",
    ):
        assert phrase in evaluation


def test_step1_characterization_and_report_remain_fallback_free() -> None:
    for dataset in ("lfw", "survface"):
        path = (
            NOTEBOOK_ROOT
            / dataset
            / "02_compression"
            / "02_step1_compression_characterization.ipynb"
        )
        source = _source(path)
        phrases = [
            "paired_embedding_metrics",
            "compare_cosine_retrieval",
            "origin_fallback_used",
            "storage_bytes_per_embedding",
            "codebook_bytes",
        ]
        if dataset == "survface":
            phrases.extend(
                [
                    "apply_retrieval_thresholds",
                    "load_survface_compressor_bundle",
                ]
            )
        else:
            phrases.extend(
                [
                    "origin_threshold=origin_threshold",
                    "compressed_threshold=operating_compressed_threshold",
                ]
            )
        for phrase in phrases:
            assert phrase in source

    lfw_open_set = _source(
        NOTEBOOK_ROOT
        / "lfw"
        / "03_open_set"
        / "00_probe_search_and_certification.ipynb"
    )
    assert "EXECUTE_LEGACY_FALLBACK" not in lfw_open_set
    assert "origin_fallback_used" in lfw_open_set
    assert "verified_read_only" in lfw_open_set
    assert "dir_rank1_recomputed" in lfw_open_set
    assert "fpir_recomputed" in lfw_open_set

    lfw_visualization = _source(
        NOTEBOOK_ROOT
        / "lfw"
        / "03_open_set"
        / "01_evaluation_and_visualization.ipynb"
    )
    assert "visualized_read_only" in lfw_visualization
    assert "origin_fallback_used" in lfw_visualization

    report = _source(
        NOTEBOOK_ROOT / "common" / "reports" / "00_cross_dataset_results.ipynb"
    )
    for phrase in (
        "origin_fallback_used",
        "origin_decision_threshold",
        "compressed_decision_threshold",
        "profile_storage schema mismatch",
        "PREFER_MULTI_FPIR_SEARCH_SPACE",
        "REQUIRE_HOMOGENEOUS_POSTPROCESSING",
        "ALLOW_LEGACY_POSTPROCESSING",
        "search_space_v4_multi_fpir",
        "step4_search_space_multi_fpir_v4",
        '"target_fpir"',
        "accept_to_reject_count",
        "reject_to_accept_count",
        "summary_claim_status",
        "INCLUDE_FAITHFULNESS",
        "load_selected_faithfulness_artifacts",
        "faithfulness_summary_all.csv",
        "RFW_EVALUATION_DIR",
        "load_rfw_frozen_codec_evaluation",
        "RFW_PROFILE_SUMMARY",
        "supplementary_1to1_verification",
        "lfw_survface_fpir_appendix.csv",
        "origin_false_accept_count",
        "compressed_false_accept_count",
        "origin_fpir_denominator",
        "compressed_fpir_denominator",
        "origin_realized_fpir",
        "compressed_realized_fpir",
        "origin_fpir_wilson95_low",
        "compressed_fpir_wilson95_high",
        "compressed_minus_origin_fpir_paired_bootstrap95_low",
        "compressed_minus_origin_dir_rank1_paired_bootstrap95_high",
        '"checkpoint_overlap_status": "UNKNOWN"',
        '"strict_unseen_identity_evidence": False',
        "MODEL_RUN_MATRIX",
        "REQUIRE_COMPLETE_MODEL_MATRIX",
        "load_cross_model_open_set_matrix",
        "CROSS_MODEL_JOINED",
        "4 checkpoints × 3 open-set datasets",
    ):
        assert phrase in report


def test_cross_dataset_transfer_notebook_is_protocol_aware_and_selectable() -> None:
    source = _source(
        NOTEBOOK_ROOT
        / "common"
        / "orchestration"
        / "cross_dataset_calibration_transfer.ipynb"
    )

    for phrase in (
        'CALIBRATION_SOURCE_DATASET_IDS',
        'DATASET_IDS = tuple(',
        'globals().get("DATASET_IDS", ("survface", "lfw"))',
        "build_cross_dataset_calibration_plan",
        "evaluate_external_calibration_transfer",
        "evaluate_rfw_official_internal_baseline",
        "ALLOW_SAME_DOMAIN_DIAGNOSTIC",
        "same-domain diagnostic",
        "maximum-gallery-score",
        "pair-score TAR/FAR/EER",
        "EdgeFace–RFW overlap은 `UNKNOWN`",
        "EXECUTE_TRANSFER",
        "WRITE_TRANSFER_OUTPUTS",
    ):
        assert phrase in source
