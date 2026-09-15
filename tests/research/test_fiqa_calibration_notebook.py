import json
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.runtime.hashing import canonical_sha256


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "calibration"
    / "00_fiqa_conditioned_threshold_calibration.ipynb"
)


def test_fiqa_calibration_notebook_is_valid_and_preserves_historical_outputs():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    identifiers = [str(cell["id"]) for cell in cells]

    assert len(identifiers) == len(set(identifiers))
    assert notebook["nbformat"] == 4
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None or isinstance(cell["execution_count"], int)
            assert isinstance(cell["outputs"], list)
            if cell["id"].startswith("multibin-"):
                assert cell["execution_count"] is None
                assert cell["outputs"] == []


def test_fiqa_calibration_notebook_preserves_execution_and_saliency_contracts():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    full_source = "\n".join(str(cell["source"]) for cell in notebook["cells"])

    for flag in (
        "RUN_MODEL_SMOKE = False",
        "RUN_FIQA_INFERENCE = False",
        "WRITE_FIQA_ARTIFACT = False",
        "RUN_MULTIBIN_DIAGNOSTICS = False",
        "WRITE_MULTIBIN_DIAGNOSTICS = False",
        "RUN_MULTIBIN_SPLIT_STABILITY = False",
        "WRITE_MULTIBIN_SPLIT_STABILITY = False",
        "OVERWRITE_OUTPUTS = False",
    ):
        assert flag in full_source

    assert "fiqa_2bin_conservative_shrunk_safe" in full_source
    assert "`identity_id` SHA-256" in full_source
    assert "pq_512_m128_b8" in full_source
    assert "pq_adc_exhaustive" in full_source
    assert "source_model_uid" in full_source
    assert "startswith('arcface-')" not in full_source
    assert "MODEL_SMOKE_SAMPLE_COUNT = FIQA_BATCH_SIZE" in full_source
    assert "FIQA_SHARD_SIZE = 8192" in full_source
    assert "materialize_aligned_bundle_score_artifact" in full_source
    assert "reused_verified" in full_source
    assert "WRITE=True이면 대응하는 RUN도 True" in full_source
    assert "기존 FIQA artifact가 있습니다" not in full_source
    assert "기존 condition score artifact가 있습니다" not in full_source
    assert "기존 calibration artifact가 있습니다" not in full_source
    assert "Saliency 1차 목적" in full_source
    assert "Saliency 2차 목적" in full_source
    assert "outside_face_attention" in full_source
    assert "saliency_entropy" in full_source
    assert "recalibrated_minus_frozen_rate" in full_source
    assert "recalibrated_minus_frozen_rho" in full_source
    assert "mask &= frame['saliency_feature']" not in full_source
    assert "random" in full_source
    assert "assess_saliency_faithfulness_reliability" in full_source
    assert "`Random`은 threshold feature가 아니라" in full_source
    assert "load_selected_faithfulness_artifacts" in full_source
    assert "saliency_correction_enabled" in full_source
    assert "saliency_reliability_weight" in full_source
    assert "High−Random" in full_source
    assert "test leakage" in full_source
    assert "formal FPIR guarantee가 아닙니다" in full_source
    assert "not applicable" in full_source


def test_split_stability_cell_uses_top_settings_and_is_independent_of_priority_stage():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c["id"] == "split-stability")
    source = "".join(cell["source"])
    context = {"display": lambda *args: None, "Markdown": str,
               "RUN_SPLIT_STABILITY": False, "WRITE_SPLIT_STABILITY": False}
    # No condition, source run, or 8.1 variables needed when disabled.
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), context)
    assert context["split_stability"] is None
    assert "priority_diagnostics" not in source
    with pytest.raises(ValueError, match="WRITE requires"):
        exec(compile(source, str(NOTEBOOK_PATH), "exec"),
             {**context, "WRITE_SPLIT_STABILITY": True})
    with pytest.raises(RuntimeError, match="v2 condition"):
        exec(compile(source, str(NOTEBOOK_PATH), "exec"),
             {**context, "RUN_SPLIT_STABILITY": True, "condition_tables": None})


def test_user_settings_are_defined_once_in_first_code_cell():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert cells[0]["id"] == "user-configuration"
    settings = {}
    for cell in cells:
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if (name.startswith(("RUN_", "WRITE_", "LOAD_", "CLUSTER_BOOTSTRAP_", "SPLIT_STABILITY_"))
                                or name in {"QUALITY_BIN_COUNTS", "QUALITY_BIN_COUNT", "PARTITION_SEED",
                                            "SALIENCY_FEATURE_PATH", "SALIENCY_MINIMUM_COVERAGE",
                                            "SALIENCY_REQUESTED_FEATURES", "INTEGRATED_EVIDENCE_SOURCES"}):
                            settings.setdefault(name, []).append(cell["id"])
    assert settings and all(ids == ["user-configuration"] for ids in settings.values())
    assert "QUALITY_BIN_COUNTS" in settings


@pytest.mark.parametrize("stage", ["multibin-diagnostics", "multibin-split-stability"])
def test_multibin_cells_can_be_disabled_without_inputs(stage):
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "".join(next(c for c in notebook["cells"] if c["id"] == stage)["source"])
    context = {"display": lambda *args: None, "Markdown": str,
               "RUN_MULTIBIN_DIAGNOSTICS": False, "WRITE_MULTIBIN_DIAGNOSTICS": False,
               "RUN_MULTIBIN_SPLIT_STABILITY": False, "WRITE_MULTIBIN_SPLIT_STABILITY": False}
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), context)


def test_fiqa_notebook_separates_metric_versions_and_documents_ci_scope():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "CONDITION_OUTPUT_DIR = LEGACY_CONDITION_DIR / METRIC_CONTRACT" in source
    assert "/ METRIC_CONTRACT / f'partition-{PARTITION_SEED}'" in source
    assert "upgrade_condition_score_artifact(LEGACY_CONDITION_DIR, SOURCE_RUN_DIR)" in source
    assert "정답 identity의 점수 자체가 threshold 이상" in source
    assert "query 단위·고정 threshold" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("stale_field", [None, "partition_seed", "shrinkage_strength", "condition_hash"])
def test_fiqa_notebook_reuse_checks_settings_and_manifest_hashes(stale_field):
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["id"] == "threshold-comparison")
    tree = ast.parse("".join(cell["source"]))
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "comparison is not None")
    condition = SimpleNamespace(condition_uid="condition", manifest={"score_space": "adc"})
    fiqa = SimpleNamespace(manifest={"fiqa_uid": "fiqa"})
    manifest = {
        "condition_uid": "condition", "fiqa_uid": "fiqa", "score_space": "adc",
        "metric_contract": "genuine-score-topk-v2", "target_fpirs": [0.1],
        "condition_manifest_sha256": canonical_sha256(condition.manifest),
        "fiqa_manifest_sha256": canonical_sha256(fiqa.manifest),
        "quality_condition": {"column": "fiqa_score", "bin_count": 2,
            "cutpoint_source": "calibration_fit_only", "shrinkage_strength": 200.0,
            "minimum_group_non_mated": 100},
        "safety_calibration": {"fraction": 0.3, "partition_seed": 8972,
            "partition_key": "sha256(identity_id)", "partition_unit": "identity_cluster"},
    }
    if stale_field == "partition_seed":
        manifest["safety_calibration"]["partition_seed"] = 1
    elif stale_field == "shrinkage_strength":
        manifest["quality_condition"]["shrinkage_strength"] = 50
    elif stale_field == "condition_hash":
        manifest["condition_manifest_sha256"] = "stale"
    context = dict(comparison=SimpleNamespace(manifest=manifest), condition_tables=condition,
                   fiqa_artifact=fiqa, METRIC_CONTRACT="genuine-score-topk-v2",
                   TARGET_FPIRS=(0.1,), QUALITY_BIN_COUNT=2, SHRINKAGE_STRENGTH=200.0,
                   MINIMUM_GROUP_NON_MATED=100, SAFETY_FRACTION=0.3, PARTITION_SEED=8972,
                   canonical_sha256=canonical_sha256)
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(NOTEBOOK_PATH), "exec")
    if stale_field is None:
        exec(code, context)
    else:
        with pytest.raises(ValueError, match="lineage"):
            exec(code, context)
