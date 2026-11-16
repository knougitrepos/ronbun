import hashlib
import copy
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from research.experiments.facex_inputs import ImageReader, load_affines


def test_alignment_pixel_parity_and_raw_source_binding(tmp_path):
    bgr = np.arange(32 * 17 * 3, dtype=np.uint8).reshape(32, 17, 3)
    payload = cv2.imencode(".png", bgr)[1].tobytes()
    (tmp_path / "raw.png").write_bytes(payload)
    matrix = np.array([[0.9876543210987654, -.012, 3.4], [.012, .9876543210987654, 5.6]])
    csv = tmp_path / "affine.csv"
    pd.DataFrame([["q", hashlib.sha256(payload).hexdigest(), *matrix.ravel()]],
                 columns=["image_id", "source_sha256", "a", "b", "tx", "c", "d", "ty"]).to_csv(csv, index=False)
    manifest = {"rows": 1}
    affine = load_affines(csv, manifest, pd.DataFrame({"image_id": ["q"]}))["q"]
    np.testing.assert_array_equal(matrix, affine[1])
    reader = ImageReader(tmp_path)
    rgb, _ = reader.read("raw.png", affine=affine)
    expected = cv2.warpAffine(bgr, matrix, (112, 112), flags=cv2.INTER_CUBIC)
    np.testing.assert_array_equal(rgb, cv2.cvtColor(expected, cv2.COLOR_BGR2RGB))
    with pytest.raises(ValueError, match="source"):
        reader.read("raw.png", affine=("0" * 64, matrix))
    with pytest.raises(ValueError, match="cohort"):
        load_affines(csv, manifest, pd.DataFrame({"image_id": ["different"]}))


def test_survface_config_requires_single_scientific_change(tmp_path, monkeypatch):
    from research.embeddings import facex_zoo, manifests
    from research.experiments import facex_workflow, facex_comparison

    base = dict(head="arcface", dataset="survface", epochs=40, seed=8972, parent_hash="p",
                protocol_record={"roles_sha256": "r"}, preprocessing={"alignment": "resize_only"},
                source_config={"head": {"margin_arc": .6, "margin_am": 0}, "training": {"lr": .001}},
                execution_settings={"augmentation": {}, "heads": {"arcface": {"margin_am": 0}}})
    control = copy.deepcopy(base)
    margin = 1 - math.cos(.6)
    control["source_config"]["head"]["margin_am"] = round(margin, 4)
    control["execution_settings"]["heads"]["arcface"]["margin_am"] = margin
    monkeypatch.setattr(facex_zoo, "_read_validated_export",
                        lambda p: (p, base if p == "ref.pt" else control, {}))
    monkeypatch.setattr(facex_zoo, "register_facex_export", lambda *a: tmp_path / "new.json")
    monkeypatch.setattr(manifests, "read_model_spec", lambda p: SimpleNamespace(checkpoint=SimpleNamespace(path="ref.pt")))
    cfg = dict(project_root=str(tmp_path), models={"survface_arcface_s8972": {"spec": "base.json", "training_dataset": "survface"}})
    monkeypatch.setattr(facex_workflow, "_config", lambda p: copy.deepcopy(cfg))
    args = ("base.yaml", "control.pt", tmp_path, tmp_path / "plan.yaml")
    path = facex_comparison.write_survface_config(*args)
    assert facex_comparison.write_survface_config(*args) == path
    plan = yaml.safe_load(path.read_bytes())
    assert plan["datasets"] == ["survface"] and len(plan["models"]) == 2
    control["source_config"]["training"]["lr"] *= 2
    with pytest.raises(ValueError, match="Additional model or training"):
        facex_comparison.write_survface_config(*args)
    assert yaml.safe_load(path.read_bytes()) == plan


def test_result_contrasts_keep_origin_compression_and_calibration_separate():
    from research.experiments.facex_comparison import control_tables
    records = []
    for profile, space, global_tpir, continuous_tpir in (
        ("origin_1024", "cosine", .2, .22), ("pq_1024_m64_b8", "negative_squared_l2_adc", .18, .19)
    ):
        for method, tpir, fpir in (("global_safe", global_tpir, .09), ("continuous_fiqa", continuous_tpir, .101)):
            records.append(dict(dataset="survface", model_uid="u", model_alias="arc", target_fpir=.1,
                                mated_count=1000, non_mated_count=1000, compression_profile=profile, score_space=space,
                                method=method, tpir=tpir, realized_fpir=fpir, target_met_on_test=fpir <= .1))
    result = dict(manifest={}, metrics=pd.DataFrame(records))
    table = control_tables(result)["SurvFace · compression / calibration"]
    row = table.loc[table.compression_profile.eq("pq_1024_m64_b8")].iloc[0]
    assert row.compression_change_pp == pytest.approx(-2)
    assert row.calibration_change_pp == pytest.approx(1)
    assert not row.both_target_met
    assert row.realized_fpir_global == .09 and row.realized_fpir_continuous == .101
