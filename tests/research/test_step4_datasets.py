from __future__ import annotations

import numpy as np
import pandas as pd

from research.experiments.step4_datasets import (
    select_step4_saliency_sample_mask,
)


def test_survface_saliency_sampling_is_deterministic_and_role_balanced():
    manifest = pd.DataFrame(
        {
            "sample_id": [f"registered-{index}" for index in range(10)]
            + [f"unknown-{index}" for index in range(10)],
            "split": "test",
            "protocol_role": ["registered_probe"] * 10
            + ["unknown_unknown_probe"] * 10,
        }
    )
    eligible = np.ones(len(manifest), dtype=bool)

    first = select_step4_saliency_sample_mask(
        manifest,
        eligible,
        dataset_id="survface",
        maximum_samples=8,
        seed=42,
    )
    repeated = select_step4_saliency_sample_mask(
        manifest,
        eligible,
        dataset_id="survface",
        maximum_samples=8,
        seed=42,
    )

    assert np.array_equal(first, repeated)
    assert int(first.sum()) == 8
    selected_roles = manifest.loc[first, "protocol_role"].value_counts().to_dict()
    assert selected_roles == {
        "registered_probe": 4,
        "unknown_unknown_probe": 4,
    }
