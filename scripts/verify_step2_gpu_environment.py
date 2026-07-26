from __future__ import annotations

from importlib import metadata
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "requirements-step2-cu118.lock.txt"
EXPECTED_PYTHON = (3, 11, 9)


def _locked_versions() -> dict[str, str]:
    locked: dict[str, str] = {}
    for raw_line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator:
            raise ValueError(f"lock entry must use ==: {line}")
        locked[name] = version
    return locked


def main() -> None:
    failures: list[str] = []
    if sys.version_info[:3] != EXPECTED_PYTHON:
        failures.append(
            "Python version mismatch: "
            f"expected={'.'.join(map(str, EXPECTED_PYTHON))}, "
            f"actual={sys.version.split()[0]}"
        )

    for package, expected in _locked_versions().items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            failures.append(f"missing distribution: {package}=={expected}")
            continue
        if actual != expected:
            failures.append(
                f"version mismatch: {package} expected={expected}, actual={actual}"
            )

    try:
        metadata.version("onnxruntime")
    except metadata.PackageNotFoundError:
        pass
    else:
        failures.append(
            "onnxruntime CPU distribution is installed; keep only onnxruntime-gpu"
        )

    import onnxruntime as ort
    import torch

    if torch.__version__ != "2.7.1+cu118":
        failures.append(f"unexpected torch build: {torch.__version__}")
    if torch.version.cuda != "11.8":
        failures.append(f"unexpected torch CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        failures.append("torch.cuda.is_available() is False")

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        failures.append(
            f"CUDAExecutionProvider is unavailable: providers={providers}"
        )

    if failures:
        raise RuntimeError(
            "Step 2 GPU environment verification failed:\n- "
            + "\n- ".join(failures)
        )

    device_name = torch.cuda.get_device_name(0)
    print(
        {
            "status": "validated",
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": device_name,
            "onnxruntime_gpu": metadata.version("onnxruntime-gpu"),
            "onnx_providers": providers,
            "lock": str(LOCK_PATH),
        }
    )


if __name__ == "__main__":
    main()
