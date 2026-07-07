import subprocess
import sys


def test_setup_postgres_pgvector_script_dry_run_reports_health():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/setup_postgres_pgvector.py",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "dry_run=True" in result.stdout
    assert "vector_extension_version=" in result.stdout
    assert "missing_tables=" in result.stdout
