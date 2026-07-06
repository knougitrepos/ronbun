import subprocess
import sys


def test_face_search_cli_dry_run_accepts_all_phase_config():
    result = subprocess.run(
        [
            sys.executable,
            "experiments/run_face_search_study.py",
            "--config",
            "experiments/configs/face_search.yaml",
            "--phase",
            "all",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "dry_run=True" in result.stdout
    assert "phase=all" in result.stdout
    assert "config_hash=" in result.stdout
