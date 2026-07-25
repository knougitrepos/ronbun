"""Tests for research.preprocessing.aligned_crops materializer."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from research.preprocessing.aligned_crops import (
    ALIGNMENT_TEMPLATE_ID,
    DETECTOR_NAME,
    MATERIALIZER_VERSION,
    materialize_aligned_crops,
)


def test_aligned_crops_validation_errors(tmp_path: Path):
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="필수 열이 없습니다"):
        materialize_aligned_crops(
            empty_df,
            project_root=tmp_path,
            output_dir=tmp_path / "out",
            dataset_id="lfw",
        )

    invalid_df = pd.DataFrame({"image_id": ["1"], "identity_id": ["id1"]})
    with pytest.raises(ValueError, match="필수 열이 없습니다"):
        materialize_aligned_crops(
            invalid_df,
            project_root=tmp_path,
            output_dir=tmp_path / "out",
            dataset_id="lfw",
        )

    valid_cols_df = pd.DataFrame(
        {
            "image_id": [],
            "identity_id": [],
            "split": [],
            "image_path": [],
        }
    )
    with pytest.raises(ValueError, match="manifest가 비어 있습니다"):
        materialize_aligned_crops(
            valid_cols_df,
            project_root=tmp_path,
            output_dir=tmp_path / "out",
            dataset_id="lfw",
        )
