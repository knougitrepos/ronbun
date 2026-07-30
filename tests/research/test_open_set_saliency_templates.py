from __future__ import annotations

import numpy as np
import pandas as pd

from research.explainability.gradcam.templates import (
    NOT_OFFICIAL_PROBE_REASON,
    ORIGIN_TOP1_GALLERY_TARGET_NAME,
    TOP1_GALLERY_ELIGIBLE_REASON,
    build_origin_top1_gallery_templates,
)


def test_origin_top1_target_includes_survface_unmated_singletons():
    sample_ids = np.asarray(["g1", "g2", "p1", "u1", "train1"])
    identity_ids = np.asarray(
        ["gallery-a", "gallery-b", "gallery-a", "opaque-singleton", "train-a"]
    )
    embeddings = np.zeros((5, 4), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0
    embeddings[2, 0] = 1.0
    embeddings[3] = np.asarray([0.8, 0.6, 0.0, 0.0], dtype=np.float32)
    embeddings[4, 2] = 1.0
    manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "identity_id": identity_ids,
            "protocol_role": [
                "gallery",
                "gallery",
                "registered_probe",
                "unknown_unknown_probe",
                "training",
            ],
        }
    )

    bundle = build_origin_top1_gallery_templates(
        sample_ids,
        identity_ids,
        embeddings,
        manifest,
        model_uid="model-a",
        scope_ids=["test", "test", "test", "test", "development"],
        query_batch_size=1,
        gallery_batch_size=1,
    )
    metadata = bundle.metadata_frame().set_index("sample_id")

    assert bundle.target_name == ORIGIN_TOP1_GALLERY_TARGET_NAME
    assert metadata.loc["p1", "saliency_target_status"] == (
        TOP1_GALLERY_ELIGIBLE_REASON
    )
    assert metadata.loc["u1", "saliency_target_status"] == (
        TOP1_GALLERY_ELIGIBLE_REASON
    )
    assert bool(metadata.loc["u1", "saliency_target_eligible"])
    assert metadata.loc["u1", "saliency_reference_identity_id"] == "gallery-a"
    assert metadata.loc["train1", "saliency_target_status"] == (
        NOT_OFFICIAL_PROBE_REASON
    )
    assert not bool(metadata.loc["train1", "saliency_target_eligible"])
