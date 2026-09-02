import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "calibration"
    / "00_fiqa_conditioned_threshold_calibration.ipynb"
)


def test_fiqa_calibration_notebook_is_clean_and_restartable():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    identifiers = [str(cell["id"]) for cell in cells]

    assert len(identifiers) == len(set(identifiers))
    assert notebook["nbformat"] == 4
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_fiqa_calibration_notebook_preserves_execution_and_saliency_contracts():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    full_source = "\n".join(str(cell["source"]) for cell in notebook["cells"])

    for flag in (
        "RUN_MODEL_SMOKE = False",
        "RUN_FIQA_INFERENCE = False",
        "WRITE_FIQA_ARTIFACT = False",
        "RUN_SCORE_REPLAY = False",
        "WRITE_SCORE_ARTIFACT = False",
        "RUN_THRESHOLD_CALIBRATION = False",
        "WRITE_CALIBRATION_ARTIFACT = False",
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
    assert "기존 FIQA artifact가 있습니다" in full_source
    assert "Saliency 1차 목적" in full_source
    assert "Saliency 2차 목적" in full_source
    assert "outside_face_attention" in full_source
    assert "saliency_entropy" in full_source
    assert "recalibrated_minus_frozen_rate" in full_source
    assert "recalibrated_minus_frozen_rho" in full_source
    assert "mask &= frame['saliency_feature']" not in full_source
    assert "random" in full_source
    assert "test leakage" in full_source
    assert "formal FPIR guarantee가 아닙니다" in full_source
    assert "not applicable" in full_source
