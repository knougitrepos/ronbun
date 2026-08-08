from __future__ import annotations

from io import BytesIO
from pathlib import Path
import pickle
import tarfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from research.datasets.rfw import RFW_GROUPS
from research.datasets.rfw_aligned_bin import (
    inspect_rfw_aligned_bin_archive,
    iter_rfw_aligned_pair_batches,
)
from research.datasets.sources import DatasetIntegrityError


def _jpeg(value: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(
        np.full((112, 112, 3), value, dtype=np.uint8), mode="RGB"
    ).save(buffer, format="JPEG")
    return buffer.getvalue()


def _write_archive(path: Path, *, malicious: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for group_index, group in enumerate(RFW_GROUPS):
            if malicious and group == RFW_GROUPS[0]:
                payload = pickle.dumps(eval, protocol=2)
            else:
                images = [
                    _jpeg(20 + group_index),
                    _jpeg(20 + group_index),
                    _jpeg(40 + group_index),
                    _jpeg(80 + group_index),
                ]
                payload = pickle.dumps((images, [True, False]), protocol=2)
            info = tarfile.TarInfo(f"RFW_test/{group}_test.bin")
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": "rfw:african:fold00:pair000",
                "rfw_group": "African",
                "fold_index": 0,
                "official_index": 0,
                "is_genuine": True,
            },
            {
                "pair_id": "rfw:african:fold00:pair001",
                "rfw_group": "African",
                "fold_index": 0,
                "official_index": 1,
                "is_genuine": False,
            },
        ]
    )


def test_rfw_aligned_bin_inspection_and_batches_preserve_pair_sides(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "RFW_test.tar.gz"
    _write_archive(archive)

    summary = inspect_rfw_aligned_bin_archive(
        archive, strict_official=False
    )
    batches = list(
        iter_rfw_aligned_pair_batches(
            archive,
            _pairs(),
            batch_size=3,
            strict_official=False,
        )
    )

    assert summary.groups == RFW_GROUPS
    assert summary.pair_count == 8
    assert summary.encoded_image_occurrence_count == 16
    assert [len(batch.faces) for batch in batches] == [3, 1]
    occurrences = pd.concat(
        [batch.occurrences for batch in batches], ignore_index=True
    )
    assert occurrences["side"].tolist() == ["left", "right", "left", "right"]
    assert occurrences["occurrence_id"].is_unique
    assert all(batch.faces.shape[1:] == (112, 112, 3) for batch in batches)


def test_rfw_aligned_bin_rejects_executable_pickle_opcodes(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "malicious.tar.gz"
    _write_archive(archive, malicious=True)

    with pytest.raises(DatasetIntegrityError, match="forbidden pickle"):
        inspect_rfw_aligned_bin_archive(archive, strict_official=False)
