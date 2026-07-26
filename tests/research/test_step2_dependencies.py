from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_step2_requirements_select_the_validated_cuda_118_stack() -> None:
    base = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    step2 = (PROJECT_ROOT / "requirements-step2.txt").read_text(encoding="utf-8")
    lock = (PROJECT_ROOT / "requirements-step2-cu118.lock.txt").read_text(
        encoding="utf-8"
    )

    assert "\nonnxruntime>=" not in f"\n{base}"
    assert "--extra-index-url https://download.pytorch.org/whl/cu118" in step2
    assert "-c requirements-step2-cu118.lock.txt" in step2
    assert "onnxruntime-gpu" in step2
    assert "\nonnxruntime\n" not in f"\n{step2}\n"
    assert "torch==2.7.1+cu118" in lock
    assert "torchvision==0.22.1+cu118" in lock
    assert "onnxruntime-gpu==1.16.3" in lock
    assert "numpy==1.26.4" in lock
    assert "opencv-python==4.11.0.86" in lock
    assert "pandas==3.0.3" in lock


def test_step2_gpu_verifier_is_fail_closed() -> None:
    source = (
        PROJECT_ROOT / "scripts" / "verify_step2_gpu_environment.py"
    ).read_text(encoding="utf-8")
    assert 'EXPECTED_PYTHON = (3, 11, 9)' in source
    assert 'metadata.version("onnxruntime")' in source
    assert 'torch.version.cuda != "11.8"' in source
    assert '"CUDAExecutionProvider" not in providers' in source
