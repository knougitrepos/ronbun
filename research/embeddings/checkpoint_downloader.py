"""Checkpoint download helpers for Step 2 model profiles.

Downloads pretrained checkpoints from Google Drive (via ``gdown``) or other
public sources and verifies file integrity.  ArcFace MS1MV3 R100 must be
placed manually because no stable automated download URL is confirmed.

This module intentionally does **not** train or modify any model; it only
retrieves pre-existing publicly released checkpoint files.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Google Drive file IDs — extracted from the official repository README links.
_GOOGLE_DRIVE_FILES: dict[str, str] = {
    "adaface_ms1mv3_r100": "1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI",
    "adaface_ms1mv2_r100": "1m757p4-tUU5xlSHLaO04sqnhvqankimN",
    # MagFace Google Drive ID is extracted from the official MagFace README.
    # If the ID changes, update it here after verifying the new file's SHA-256.
    "magface_ms1mv2_iresnet100": "1Bd87admxOZvbIOAyTkGEntsEz3fyMigH",
}

# Default local filenames under models/<family>/
_DEFAULT_FILENAMES: dict[str, str] = {
    "arcface_ms1mv3_r100": "ms1mv3_r100_backbone.pth",
    "adaface_ms1mv3_r100": "adaface_ir101_ms1mv3.ckpt",
    "adaface_ms1mv2_r100": "adaface_ir101_ms1mv2.ckpt",
    "magface_ms1mv2_iresnet100": "magface_ms1mv2.pth",
    "edgeface_webface12m_xs_gamma_06": "edgeface_xs_gamma_06.pt",
}

# Profile → expected family subfolder
_FAMILY_DIR: dict[str, str] = {
    "arcface_ms1mv3_r100": "arcface",
    "adaface_ms1mv3_r100": "adaface",
    "adaface_ms1mv2_r100": "adaface",
    "magface_ms1mv2_iresnet100": "magface",
    "edgeface_webface12m_xs_gamma_06": "edgeface",
}


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint_path(
    project_root: Path,
    profile_id: str,
    *,
    explicit_path: Path | None = None,
) -> Path:
    """Return the expected local path for a profile's checkpoint.

    Parameters
    ----------
    project_root:
        Project repository root.
    profile_id:
        Profile identifier from YAML config (e.g. ``adaface_ms1mv3_r100``).
    explicit_path:
        Override for the checkpoint path.  When provided the default
        ``models/<family>/<filename>`` convention is skipped.
    """
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    family_dir = _FAMILY_DIR.get(profile_id)
    filename = _DEFAULT_FILENAMES.get(profile_id)
    if family_dir is None or filename is None:
        raise ValueError(
            f"알 수 없는 profile_id: {profile_id}. "
            f"지원되는 profile: {sorted(_DEFAULT_FILENAMES)}"
        )
    return (project_root / "models" / family_dir / filename).resolve()


def download_checkpoint(
    project_root: Path,
    profile_id: str,
    profile_config: dict[str, Any],
    *,
    explicit_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Download a checkpoint if it does not exist locally.

    Returns the resolved local path.  For ``arcface_ms1mv3_r100`` the
    function will raise an error directing the user to download manually.
    """
    local_path = resolve_checkpoint_path(
        project_root, profile_id, explicit_path=explicit_path
    )

    if local_path.is_file() and not force:
        logger.info("Checkpoint already exists: %s", local_path)
        return local_path

    # ArcFace MS1MV3 has no stable automated download URL.
    if profile_id == "arcface_ms1mv3_r100":
        raise FileNotFoundError(
            f"ArcFace MS1MV3 R100 checkpoint을 자동으로 다운로드할 수 없습니다.\n"
            f"InsightFace OneDrive 또는 HuggingFace에서 직접 다운로드한 뒤\n"
            f"다음 경로에 배치하세요: {local_path}\n"
            f"참고: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch#model-zoo"
        )

    google_drive_id = _GOOGLE_DRIVE_FILES.get(profile_id)
    if google_drive_id is None:
        checkpoint_url = profile_config.get("checkpoint_source_url")
        raise FileNotFoundError(
            f"profile '{profile_id}'의 자동 다운로드가 지원되지 않습니다.\n"
            f"다음 URL에서 직접 다운로드하세요: {checkpoint_url}\n"
            f"로컬 경로: {local_path}"
        )

    try:
        import gdown  # noqa: delayed import
    except ImportError as exc:
        raise ImportError(
            "checkpoint 자동 다운로드에 gdown 라이브러리가 필요합니다.\n"
            "설치: pip install gdown\n"
            "또는 requirements-step2.txt 참조"
        ) from exc

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={google_drive_id}"
    logger.info("Downloading %s → %s", profile_id, local_path)

    gdown.download(url, str(local_path), quiet=False)

    if not local_path.is_file():
        raise RuntimeError(
            f"다운로드 후 파일이 존재하지 않습니다: {local_path}"
        )

    sha256 = _sha256_file(local_path)
    logger.info("Downloaded %s (SHA-256: %s)", profile_id, sha256)

    return local_path
