"""Dataset protocols and manifest generators for face recognition baseline training.

Ensures strict zero-leakage identity partitioning across:
- FR fine-tuning training set (1,600 identities)
- FR fine-tuning validation set (219 identities)
- 1:N calibration gallery (3,000 identities)
- 1:N calibration non-mated probes (500 identities)
Total identities: 5,319 (100% of SurvFace training_set, strictly disjoint).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from research.datasets.manifests import (
    IMAGE_SUFFIXES,
    MANIFEST_COLUMNS,
    _image_files,
    _relative_posix,
    _resolved_directory,
    _validate_manifest,
)
from research.protocols.open_set import validate_identity_disjoint_splits


@dataclass(frozen=True)
class SurvFaceFinetuneBundle:
    """Multi-role partitioned manifest and summary for SurvFace fine-tuning and calibration."""

    manifest: pd.DataFrame
    summary: dict[str, Any]

    def for_role(self, role: str) -> pd.DataFrame:
        """Filter manifest rows by finetune_role."""
        return self.manifest.loc[self.manifest["finetune_role"] == role].reset_index(
            drop=True
        )


def build_survface_finetune_manifest(
    survface_root: str | Path,
    project_root: str | Path,
    *,
    seed: int = 8972,
    train_identities_count: int = 1600,
    val_identities_count: int = 219,
    calib_gallery_count: int = 3000,
    calib_probe_count: int = 500,
    output_csv_path: str | Path | None = None,
) -> SurvFaceFinetuneBundle:
    """Build a 4-role leak-free SurvFace training manifest.

    Allocates 5,319 identities deterministically across FR fine-tuning train/val
    and 1:N calibration gallery/probe sets.
    """
    root = _resolved_directory(survface_root, "QMUL-SurvFace root")
    project = _resolved_directory(project_root, "project root")
    training_dir = _resolved_directory(root / "training_set", "SurvFace training set")

    identity_images: dict[str, list[Path]] = {}
    for identity_dir in sorted(path for path in training_dir.iterdir() if path.is_dir()):
        images = _image_files(identity_dir)
        if images:
            identity_images[identity_dir.name] = images

    total_ids = len(identity_images)
    total_requested = (
        train_identities_count
        + val_identities_count
        + calib_gallery_count
        + calib_probe_count
    )
    if total_ids < total_requested:
        raise ValueError(
            f"Not enough identities in training_set ({total_ids}) for requested split ({total_requested})"
        )

    rng = random.Random(seed)
    shuffled_ids = sorted(identity_images)
    rng.shuffle(shuffled_ids)

    idx = 0
    train_ids = set(shuffled_ids[idx : idx + train_identities_count])
    idx += train_identities_count

    val_ids = set(shuffled_ids[idx : idx + val_identities_count])
    idx += val_identities_count

    calib_gal_ids = set(shuffled_ids[idx : idx + calib_gallery_count])
    idx += calib_gallery_count

    calib_prb_ids = set(shuffled_ids[idx : idx + calib_probe_count])

    # Assert zero overlap
    assert len(train_ids.intersection(val_ids)) == 0, "train and val overlap"
    assert len(train_ids.intersection(calib_gal_ids)) == 0, "train and calib_gal overlap"
    assert len(train_ids.intersection(calib_prb_ids)) == 0, "train and calib_prb overlap"
    assert len(val_ids.intersection(calib_gal_ids)) == 0, "val and calib_gal overlap"
    assert len(val_ids.intersection(calib_prb_ids)) == 0, "val and calib_prb overlap"
    assert len(calib_gal_ids.intersection(calib_prb_ids)) == 0, "calib_gal and calib_prb overlap"

    role_by_identity: dict[str, tuple[str, str]] = {}
    for identity in identity_images:
        if identity in train_ids:
            role_by_identity[identity] = ("development", "train")
        elif identity in val_ids:
            role_by_identity[identity] = ("development", "val")
        elif identity in calib_gal_ids:
            role_by_identity[identity] = ("calibration", "calib_gallery")
        elif identity in calib_prb_ids:
            role_by_identity[identity] = ("calibration", "calib_probe")
        else:
            role_by_identity[identity] = ("development", "extra_dev")

    rows: list[dict[str, Any]] = []
    for identity in sorted(identity_images):
        stable_identity = f"survface:train:{identity}"
        split, role = role_by_identity[identity]
        for image in identity_images[identity]:
            rows.append(
                {
                    "image_id": f"{stable_identity}:{image.stem}",
                    "identity_id": stable_identity,
                    "split": split,
                    "finetune_role": role,
                    "image_path": _relative_posix(image, project),
                    "dataset": "qmul-survface-v1",
                    "protocol_role": "training",
                    "probe_type": "not_applicable",
                }
            )

    manifest = pd.DataFrame(rows)
    _validate_manifest(manifest)
    validate_identity_disjoint_splits(manifest)

    summary = {
        "dataset": "qmul-survface-v1",
        "protocol": "survface-finetune-v2-identity-disjoint",
        "seed": int(seed),
        "total_identities": int(total_ids),
        "total_images": int(len(manifest)),
        "train_identities": len(train_ids),
        "train_images": int((manifest["finetune_role"] == "train").sum()),
        "val_identities": len(val_ids),
        "val_images": int((manifest["finetune_role"] == "val").sum()),
        "calib_gallery_identities": len(calib_gal_ids),
        "calib_gallery_images": int((manifest["finetune_role"] == "calib_gallery").sum()),
        "calib_probe_identities": len(calib_prb_ids),
        "calib_probe_images": int((manifest["finetune_role"] == "calib_probe").sum()),
    }

    if output_csv_path is not None:
        out_p = Path(output_csv_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_p, index=False)

    return SurvFaceFinetuneBundle(manifest=manifest, summary=summary)


class FRDataset(Dataset):
    """PyTorch Dataset for face recognition training and validation.

    Loads aligned 112x112 face images, applies horizontal flip augmentation,
    and normalizes pixel range to [-1.0, 1.0].
    """

    def __init__(
        self,
        manifest: pd.DataFrame | None = None,
        *,
        project_root: str | Path | None = None,
        finetune_role: str | None = None,
        is_train: bool = True,
        synthetic: bool = False,
        synthetic_samples: int = 64,
        synthetic_classes: int = 8,
        color_order: str = "bgr",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.synthetic = bool(synthetic)
        self.is_train = bool(is_train)
        self.color_order = color_order.lower()
        if self.color_order not in {"bgr", "rgb"}:
            raise ValueError(f"unsupported color_order: {color_order!r}; expected 'bgr' or 'rgb'")

        if self.synthetic:
            self.num_classes = synthetic_classes
            self.samples = [
                (f"syn_{i}", i % synthetic_classes) for i in range(synthetic_samples)
            ]
            self.identity_to_class = {f"class_{c}": c for c in range(synthetic_classes)}
            self.class_to_identity = {c: f"class_{c}" for c in range(synthetic_classes)}
            self.images_root = Path(".")
            return

        if manifest is None:
            raise ValueError("manifest is required when synthetic=False")

        df = manifest.copy()
        if finetune_role is not None:
            df = df.loc[df["finetune_role"] == finetune_role].reset_index(drop=True)

        if df.empty:
            raise ValueError(f"manifest is empty for role {finetune_role!r}")

        self.project_root = (
            Path(project_root).resolve() if project_root else Path(".").resolve()
        )
        unique_identities = sorted(df["identity_id"].unique())
        self.identity_to_class = {ident: i for i, ident in enumerate(unique_identities)}
        self.class_to_identity = {i: ident for ident, i in self.identity_to_class.items()}
        self.num_classes = len(unique_identities)

        self.samples = [
            (row["image_path"], self.identity_to_class[row["identity_id"]])
            for _, row in df.iterrows()
        ]

        if transform is not None:
            self.transform = transform
        else:
            transforms_list = [
                T.Resize((112, 112)),
            ]
            if self.is_train:
                transforms_list.append(T.RandomHorizontalFlip(p=0.5))
            transforms_list.extend([
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self.transform = T.Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.synthetic:
            # Deterministic synthetic image tensor of shape (3, 112, 112)
            gen = torch.Generator().manual_seed(idx)
            img = torch.randn(3, 112, 112, generator=gen)
            img = torch.clamp(img, -1.0, 1.0)
            label = self.samples[idx][1]
            return img, label

        image_rel_path, label = self.samples[idx]
        full_path = self.project_root / image_rel_path
        with Image.open(full_path) as pil_img:
            img = pil_img.convert("RGB")
            tensor = self.transform(img)
            if self.color_order == "bgr":
                # Convert RGB channels (0, 1, 2) to BGR (2, 1, 0) to match model_color_order="bgr"
                tensor = tensor[[2, 1, 0], :, :]
        return tensor, label
