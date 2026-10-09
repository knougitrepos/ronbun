from __future__ import annotations

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from research.experiments.rfw_continuous_calibration import (
    _parse_pq_m,
    compute_storage_accounting,
    run_rfw_calibration_workflow,
    run_rfw_continuous_calibration,
)


def _build_fixture():
    rng = np.random.default_rng(8972)
    groups = ("African", "Asian")
    rows = []
    image_ids = []
    vectors = []
    total_pairs = 0

    for group in groups:
        for fold in range(4):
            for i in range(250):
                left_id = f"{group}-f{fold}-p{i}-L"
                right_id = f"{group}-f{fold}-p{i}-R"
                image_ids.extend([left_id, right_id])

                v_left = rng.normal(size=512).astype(np.float32)
                v_left /= np.linalg.norm(v_left)
                is_genuine = (i % 2 == 0)
                if is_genuine:
                    v_right = v_left + rng.normal(scale=0.1, size=512).astype(np.float32)
                else:
                    v_right = rng.normal(size=512).astype(np.float32)
                v_right /= np.linalg.norm(v_right)
                vectors.extend([v_left, v_right])

                pair_id = f"rfw:{group.lower()}:fold{fold:02d}:pair{i:03d}"
                rows.append({
                    "pair_id": pair_id,
                    "rfw_group": group,
                    "fold_index": fold,
                    "left_image_id": left_id,
                    "right_image_id": right_id,
                    "left_identity_id": f"id-{left_id}",
                    "right_identity_id": f"id-{right_id}",
                    "is_genuine": is_genuine,
                })
                total_pairs += 1

    pairs = pd.DataFrame(rows)
    embeddings = np.vstack(vectors)
    fiqa_scores = {img: float(rng.uniform(30.0, 70.0)) for img in image_ids}

    dev_data = rng.normal(size=(256, 512)).astype(np.float32)
    dev_norms = np.linalg.norm(dev_data, axis=1, keepdims=True)
    dev_data /= np.where(dev_norms > 0, dev_norms, 1.0)

    return pairs, image_ids, embeddings, fiqa_scores, dev_data


def test_parse_pq_m_correctly_identifies_subquantizers():
    # Verify exact regex matching for m in (128, 64, 32) and rejection of 512
    assert _parse_pq_m("pq_512_m128_b8") == 128
    assert _parse_pq_m("pq_512_m64_b8") == 64
    assert _parse_pq_m("pq_512_m32_b8") == 32

    with pytest.raises(ValueError, match="expected PQ m in"):
        _parse_pq_m("pq_512_m512_b8")

    with pytest.raises(ValueError, match="cannot determine PQ subquantizers"):
        _parse_pq_m("pq_512_b8")


def test_compute_storage_accounting():
    origin_storage = compute_storage_accounting("origin", vector_count=1000)
    assert origin_storage["code_payload_bytes"] == 2048
    assert origin_storage["codebook_bytes"] == 0
    assert origin_storage["total_storage_bytes"] == 2_048_000
    assert origin_storage["compression_ratio"] == 1.0

    pq128_storage = compute_storage_accounting("pq_512_m128_b8", vector_count=1000)
    assert pq128_storage["code_payload_bytes"] == 128
    assert pq128_storage["codebook_bytes"] == 524_288
    assert pq128_storage["total_storage_bytes"] == 1000 * 128 + 524_288
    assert pq128_storage["compression_ratio"] > 1.0


def test_rfw_continuous_calibration_matrix():
    pairs, image_ids, embeddings, fiqa_scores, dev_data = _build_fixture()

    result = run_rfw_continuous_calibration(
        pairs,
        image_ids=image_ids,
        embeddings=embeddings,
        fiqa_scores=fiqa_scores,
        development_embeddings=dev_data,
        pq_profiles=("pq_512_m128_b8", "pq_512_m32_b8"),
        methods=("global_safe", "continuous_fiqa"),
        target_fmrs=(0.1,),
        strict_official=False,
        bootstrap_seed=8972,
        bootstrap_repeats=100,
        safety_fraction=0.3,
    )

    assert not result.fold_metrics.empty
    assert not result.group_summary.empty
    assert not result.comparison_table.empty

    profiles = set(result.comparison_table["compression_profile"])
    assert profiles == {"origin", "pq_512_m128_b8", "pq_512_m32_b8"}

    methods = set(result.comparison_table["method"])
    assert methods == {"global_safe", "continuous_fiqa"}

    groups_res = set(result.comparison_table["rfw_group"])
    assert groups_res == {"African", "Asian"}

    assert "code_payload_bytes" in result.comparison_table.columns
    assert "codebook_bytes" in result.comparison_table.columns
    assert "total_storage_bytes" in result.comparison_table.columns
    assert "compression_ratio" in result.comparison_table.columns

    # Check payload bytes values
    payload_map = dict(
        zip(result.comparison_table["compression_profile"], result.comparison_table["code_payload_bytes"])
    )
    assert payload_map["origin"] == 2048
    assert payload_map["pq_512_m128_b8"] == 128
    assert payload_map["pq_512_m32_b8"] == 32


def test_rfw_continuous_calibration_blocks_test_data_leakage():
    pairs, image_ids, embeddings, fiqa_scores, _ = _build_fixture()

    # When development_embeddings is None, must reject PQ compression
    with pytest.raises(ValueError, match="development_embeddings must be explicitly provided"):
        run_rfw_continuous_calibration(
            pairs,
            image_ids=image_ids,
            embeddings=embeddings,
            fiqa_scores=fiqa_scores,
            development_embeddings=None,
            pq_profiles=("pq_512_m32_b8",),
            strict_official=False,
        )

    # When development_embeddings is identical to embeddings, must reject
    with pytest.raises(ValueError, match="cannot be identical to evaluation embeddings"):
        run_rfw_continuous_calibration(
            pairs,
            image_ids=image_ids,
            embeddings=embeddings,
            fiqa_scores=fiqa_scores,
            development_embeddings=embeddings,
            pq_profiles=("pq_512_m32_b8",),
            strict_official=False,
        )

    # When development_embeddings is a copy with identical array content, must reject
    with pytest.raises(ValueError, match="cannot have identical content to evaluation embeddings"):
        run_rfw_continuous_calibration(
            pairs,
            image_ids=image_ids,
            embeddings=embeddings,
            fiqa_scores=fiqa_scores,
            development_embeddings=embeddings.copy(),
            pq_profiles=("pq_512_m32_b8",),
            strict_official=False,
        )

    # When development_embeddings is a subset slice or flipped rows, byte hash intersection must catch it
    with pytest.raises(ValueError, match="Data leakage detected"):
        run_rfw_continuous_calibration(
            pairs,
            image_ids=image_ids,
            embeddings=embeddings,
            fiqa_scores=fiqa_scores,
            development_embeddings=embeddings[:50],
            pq_profiles=("pq_512_m32_b8",),
            strict_official=False,
        )

    with pytest.raises(ValueError, match="Data leakage detected"):
        run_rfw_continuous_calibration(
            pairs,
            image_ids=image_ids,
            embeddings=embeddings,
            fiqa_scores=fiqa_scores,
            development_embeddings=np.flip(embeddings, axis=0),
            pq_profiles=("pq_512_m32_b8",),
            strict_official=False,
        )


def test_load_lfw_disjoint_development_embeddings():
    from research.experiments.rfw_continuous_calibration import load_lfw_disjoint_development_embeddings

    pairs_file = Path("data/external/lfw/pairs.txt")
    sources_dir = Path("results/lfw_deepfunneled_resize_v1/sources")
    if not (pairs_file.is_file() and sources_dir.is_dir()):
        pytest.skip("LFW pairs or sources directory not available")

    for model in ("arcface", "adaface", "magface"):
        vectors, meta = load_lfw_disjoint_development_embeddings(".", selected_model=model)
        assert vectors.shape == (1549, 512)
        assert meta["development_sample_count"] == 1549
        assert meta["development_dataset_id"] == "lfw-disjoint-non-test"
        assert meta["identity_overlap_verified"] is False
        assert len(meta["population_sha256"]) == 64
        assert len(meta["embeddings_sha256"]) == 64

    # Test that model UID or checkpoint SHA-256 mismatch is strictly blocked
    with pytest.raises(ValueError, match="Development embedding model UID mismatch"):
        load_lfw_disjoint_development_embeddings(
            ".",
            selected_model="arcface",
            expected_model_uid="wrong-model-uid",
        )

    with pytest.raises(ValueError, match="Development embedding checkpoint hash mismatch"):
        load_lfw_disjoint_development_embeddings(
            ".",
            selected_model="arcface",
            expected_checkpoint_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        )


def test_load_lfw_disjoint_development_embeddings_integrity_contract(tmp_path):
    import json
    import shutil
    from research.experiments.rfw_continuous_calibration import load_lfw_disjoint_development_embeddings

    pairs_file = Path("data/external/lfw/pairs.txt")
    candidates = list(Path("results/lfw_deepfunneled_resize_v1/sources").glob("arcface-*"))
    if not (pairs_file.is_file() and candidates and candidates[0].is_dir()):
        pytest.skip("LFW real source not available")
    real_source = candidates[0]

    # Case 1: _SUCCESS marker missing
    mock_src_1 = tmp_path / "src_no_success"
    shutil.copytree(real_source, mock_src_1)
    (mock_src_1 / "_SUCCESS").unlink()
    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        load_lfw_disjoint_development_embeddings(
            ".",
            selected_model="arcface",
            source_dir=mock_src_1,
        )

    # Case 2: manifest status not completed
    mock_src_2 = tmp_path / "src_status_incomplete"
    shutil.copytree(real_source, mock_src_2)
    mf_path = mock_src_2 / "manifest.json"
    with mf_path.open("r", encoding="utf-8") as f:
        mf = json.load(f)
    mf["status"] = "in_progress"
    with mf_path.open("w", encoding="utf-8") as f:
        json.dump(mf, f)
    with pytest.raises(ValueError, match="must be 'completed' before reuse"):
        load_lfw_disjoint_development_embeddings(
            ".",
            selected_model="arcface",
            source_dir=mock_src_2,
        )

    # Case 3: manifest missing files section
    mock_src_3 = tmp_path / "src_no_files_section"
    shutil.copytree(real_source, mock_src_3)
    mf_path = mock_src_3 / "manifest.json"
    with mf_path.open("r", encoding="utf-8") as f:
        mf = json.load(f)
    del mf["files"]
    with mf_path.open("w", encoding="utf-8") as f:
        json.dump(mf, f)
    with pytest.raises(ValueError, match="Mandatory 'files' section missing"):
        load_lfw_disjoint_development_embeddings(
            ".",
            selected_model="arcface",
            source_dir=mock_src_3,
        )

    # Case 4: manifest missing sha256 entry
    mock_src_4 = tmp_path / "src_no_sha"
    shutil.copytree(real_source, mock_src_4)
    mf_path = mock_src_4 / "manifest.json"
    with mf_path.open("r", encoding="utf-8") as f:
        mf = json.load(f)
    del mf["files"]["population.csv"]["sha256"]
    with mf_path.open("w", encoding="utf-8") as f:
        json.dump(mf, f)
    with pytest.raises(ValueError, match="Mandatory files.*sha256.*entry missing"):
        load_lfw_disjoint_development_embeddings(
            ".",
            selected_model="arcface",
            source_dir=mock_src_4,
        )

    # Case 5: corrupted file content (hash mismatch)
    mock_src_5 = tmp_path / "src_corrupted_file"
    shutil.copytree(real_source, mock_src_5)
    with (mock_src_5 / "population.csv").open("a", encoding="utf-8") as f:
        f.write("\ncorrupted_line,dummy,0\n")
    with pytest.raises(ValueError, match="Corrupted population.csv"):
        load_lfw_disjoint_development_embeddings(
            ".",
            selected_model="arcface",
            source_dir=mock_src_5,
        )



def test_rfw_calibration_workflow_execution(tmp_path):
    # Test full runner with temporary output directory and synthetic test fixture
    cfg_path = Path("configs/experiments/rfw_continuous_calibration.yaml")

    # 1. Non-execute check (plan only)
    plan = run_rfw_calibration_workflow(
        project_root=".",
        config_path=cfg_path,
        output_root=tmp_path / "results_plan",
        execute=False,
        synthetic=True,
    )
    assert plan["status"] == "planned"
    assert plan["execute"] is False
    assert plan["synthetic"] is True

    # 2. Execute with keep_raw_results=False in synthetic mode
    out_no_raw = tmp_path / "out_no_raw"
    res_no_raw = run_rfw_calibration_workflow(
        project_root=".",
        config_path=cfg_path,
        output_root=out_no_raw,
        execute=True,
        keep_raw_results=False,
        synthetic=True,
        seed=8972,
    )
    assert res_no_raw["status"] == "completed"
    assert res_no_raw["manifest"]["status"] == "completed"
    assert res_no_raw["manifest"]["formal_fmr_guarantee"] is False
    assert res_no_raw["manifest"]["identity_overlap_verified"] is False
    assert res_no_raw["manifest"]["raw_results_preserved"] is False
    assert res_no_raw["raw_pair_evaluations_path"] is None
    assert res_no_raw["fitted_models_path"] is None
    assert not (Path(res_no_raw["output_dir"]) / "raw_pair_evaluations.csv.gz").exists()
    assert not (Path(res_no_raw["output_dir"]) / "fitted_calibration_models.pkl").exists()

    assert Path(res_no_raw["group_summary_path"]).exists()
    assert Path(res_no_raw["comparison_table_path"]).exists()
    assert Path(res_no_raw["chatgpt_summary_path"]).exists()
    assert Path(res_no_raw["interpretation_guide_path"]).exists()
    # fold_metrics.csv is ALWAYS preserved as the audit table
    assert (Path(res_no_raw["output_dir"]) / "fold_metrics.csv").exists()

    df_comp = pd.read_csv(res_no_raw["comparison_table_path"])
    assert "tar_gain_vs_global" in df_comp.columns
    assert "pooled_false_accepts" in df_comp.columns
    assert "pooled_true_accepts" in df_comp.columns

    # 3. Execute with keep_raw_results=True in synthetic mode
    out_with_raw = tmp_path / "out_with_raw"
    res_with_raw = run_rfw_calibration_workflow(
        project_root=".",
        config_path=cfg_path,
        output_root=out_with_raw,
        execute=True,
        keep_raw_results=True,
        synthetic=True,
        seed=8972,
    )
    assert res_with_raw["status"] == "completed"
    assert res_with_raw["manifest"]["raw_results_preserved"] is True
    assert (Path(res_with_raw["output_dir"]) / "fold_metrics.csv").exists()
    # Raw artifacts MUST exist when keep_raw_results=True
    raw_csv = Path(res_with_raw["output_dir"]) / "raw_pair_evaluations.csv.gz"
    models_pkl = Path(res_with_raw["output_dir"]) / "fitted_calibration_models.pkl"
    assert raw_csv.exists()
    assert models_pkl.exists()
    assert "raw_pair_evaluations.csv.gz" in res_with_raw["manifest"]["raw_artifacts"]
    assert "fitted_calibration_models.pkl" in res_with_raw["manifest"]["raw_artifacts"]

    df_raw = pd.read_csv(raw_csv)
    assert "pair_id" in df_raw.columns
    assert "threshold" in df_raw.columns
    assert "accepted" in df_raw.columns
    assert len(df_raw) > 0


def test_rfw_calibration_workflow_cr_fiqa_hash_validation(tmp_path):
    # Verify that an incorrect CR-FIQA hash in YAML is strictly detected and raises ValueError
    import yaml

    cfg_path = Path("configs/experiments/rfw_continuous_calibration.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Corrupt expected FIQA hash
    cfg["datasets"]["fiqa"]["expected_sha256"] = "111122223333444455556666777788889999aaaabbbbccccddddeeeeffff0000"
    corrupt_cfg_path = tmp_path / "corrupt_fiqa.yaml"
    with corrupt_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    archive = Path("data/raw/RFW/bin_for_mxnet/RFW_test.tar.gz")
    protocol = Path("data/interim/rfw/pair_protocol.csv")
    cr_fiqa_ckpt = Path("models/fiqa/CR-FIQA(L).pth")
    if not (archive.is_file() and protocol.is_file() and cr_fiqa_ckpt.is_file()):
        pytest.skip("Prerequisites for real extraction hash test not present")

    with pytest.raises(ValueError, match="CR-FIQA checkpoint hash mismatch"):
        run_rfw_calibration_workflow(
            project_root=".",
            config_path=corrupt_cfg_path,
            output_root=tmp_path / "corrupt_out",
            execute=True,
            quick_smoke=True,
            device="cpu",
        )


def test_rfw_calibration_workflow_real_data_quick_smoke(tmp_path):
    # Tests real data and model extraction on a fast smoke slice if archives exist
    archive = Path("data/raw/RFW/bin_for_mxnet/RFW_test.tar.gz")
    protocol = Path("data/interim/rfw/pair_protocol.csv")
    model_ckpt = Path("models/arcface/ms1mv3_r100_backbone.pth")
    cr_fiqa_ckpt = Path("models/fiqa/CR-FIQA(L).pth")

    if not (archive.is_file() and protocol.is_file() and model_ckpt.is_file() and cr_fiqa_ckpt.is_file()):
        pytest.skip("RFW raw archive, protocol, or model/FIQA checkpoint not present for smoke test")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = Path("configs/experiments/rfw_continuous_calibration.yaml")
    out_smoke = tmp_path / "out_real_smoke"

    res = run_rfw_calibration_workflow(
        project_root=".",
        config_path=cfg_path,
        output_root=out_smoke,
        execute=True,
        keep_raw_results=True,  # Test raw results saving in real smoke mode
        quick_smoke=True,
        selected_model="arcface",
        device=device,
    )

    assert res["status"] == "completed"
    assert res["real_data"] is True
    assert res["quick_smoke"] is True
    assert res["manifest"]["status"] == "completed"
    assert res["manifest"]["model_alias"] == "arcface"
    assert res["manifest"]["quality_source"] == "cr_fiqa"
    assert res["manifest"]["development_meta"]["development_dataset_id"] == "lfw-disjoint-non-test"
    assert res["manifest"]["identity_overlap_verified"] is False
    assert res["manifest"]["raw_results_preserved"] is True
    assert Path(res["group_summary_path"]).exists()
    assert Path(res["comparison_table_path"]).exists()
    assert Path(res["chatgpt_summary_path"]).exists()
    assert Path(res["interpretation_guide_path"]).exists()
    assert (Path(res["output_dir"]) / "fold_metrics.csv").exists()
    assert (Path(res["output_dir"]) / "raw_pair_evaluations.csv.gz").exists()
    assert (Path(res["output_dir"]) / "fitted_calibration_models.pkl").exists()

