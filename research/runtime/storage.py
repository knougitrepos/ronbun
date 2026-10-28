"""Explicit external roots; repository code and data have independent locations."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import socket
import subprocess
import uuid

from research.runtime.hashing import sha256_file


def roots(dataset_root=None, model_root=None):
    return {
        "DATASET_ROOT": str(Path(dataset_root or os.environ.get("FACE_DATASET_ROOT", "F:/face_datasets" if os.name == "nt" else "/data/face_datasets")).expanduser().resolve()),
        "MODEL_ROOT": str(Path(model_root or os.environ.get("FACE_MODEL_ROOT", "F:/models" if os.name == "nt" else "/data/models")).expanduser().resolve()),
    }


def expand_paths(value, variables):
    if isinstance(value, dict):
        return {k: expand_paths(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_paths(v, variables) for v in value]
    if isinstance(value, str):
        for key, path in variables.items():
            value = value.replace("${" + key + "}", str(path))
        if "${" in value:
            raise ValueError(f"Unresolved storage variable: {value}")
    return value


@contextmanager
def artifact_lock(directory):
    """One producer per explicit destination. Never steal a possibly live lock."""
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = directory.with_name(directory.name + ".lock")
    token = uuid.uuid4().hex
    with lock.open("x", encoding="utf-8") as f:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname(), token=token), f)
    try:
        yield
    finally:
        if json.loads(lock.read_text(encoding="utf-8")).get("token") == token:
            lock.unlink()


def write_provenance(directory, *, operation, sources, parameters, counts, status="completed"):
    """Same observed execution record drives the machine record and human guide."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    payload = dict(schema="dataset-derivation-v1", status=status, operation=operation,
                   created_at=datetime.now(timezone.utc).isoformat(), repository="ronbun", commit=commit,
                   code_sha256=sha256_file(__file__), sources=sources, parameters=parameters, counts=counts,
                   history="Only this operation is attested; earlier extraction/processing history is not inferred.",
                   retention="Input datasets/checkpoints and completed shared derivatives are never deleted by KEEP_RAW_RESULTS.")
    (directory / "derivation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    guide = (f"# Dataset derivation: {operation}\n\nStatus: {status}\n\n"
             "This guide and derivation.json describe the same observed operation.\n"
             "Earlier processing history is unknown unless explicitly linked in sources.\n\n"
             f"Repository: ronbun; commit: {commit}\n\n"
             f"Parameters and reproduction inputs:\n```json\n{json.dumps(parameters, ensure_ascii=False, indent=2)}\n```\n\n"
             f"Counts and validation:\n```json\n{json.dumps(counts, ensure_ascii=False, indent=2)}\n```\n\n"
             f"Sources and hashes:\n```json\n{json.dumps(sources, ensure_ascii=False, indent=2)}\n```\n\n"
             "See derivation.json for provenance. Do not alter completed contents. "
             "No FR training or inference is implied by data preparation.\n")
    (directory / "README.md").write_text(guide, encoding="utf-8")
    return payload
