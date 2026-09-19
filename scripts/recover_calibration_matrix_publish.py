"""Recover audited publish-only failures without changing fitted result artifacts.

Default is read-only. --apply publishes fully written staging results and creates
new job receipts pointing to verified old results after the exact I/O-only patch.
The pinned source audit refuses any additional scientific or runner changes.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from research.experiments.calibration_matrix import _load_result  # noqa: E402
from research.experiments.fiqa_continuous_calibration import TABLES as FIQA_TABLES  # noqa: E402
from research.experiments.saliency_incremental_calibration import TABLES as SALIENCY_TABLES  # noqa: E402
from research.experiments.saliency_calibration_inputs import _atomic_json  # noqa: E402
from research.explainability.gradcam.artifacts import _publish_atomic_directory  # noqa: E402
from research.runtime.hashing import canonical_sha256, sha256_file  # noqa: E402

BASE_COMMIT = "672b7b799aa58c585047b720f1925e6e27451b1e"
CHANGED = ("calibration_matrix.py", "fiqa_continuous_calibration.py", "saliency_incremental_calibration.py")
IMPORT = "from research.explainability.gradcam.artifacts import _publish_atomic_directory\n"


def audit_sources(project):
    """Allow only the three reviewed import/call substitutions, including EOLs."""
    approved = {}
    for name in CHANGED:
        relative = f"research/experiments/{name}"
        old = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{relative}"], cwd=project).decode("utf8")
        old = old.replace("\r\n", "\n")
        destination = "directory" if name == "calibration_matrix.py" else "destination"
        call = f"os.rename(staging, {destination})"
        if old.count(call) != 1 or old.count("import os\n") != 1:
            raise ValueError("unexpected historical publisher")
        expected = old.replace("import os\n", "").replace(
            "from research.runtime.hashing import canonical_sha256, sha256_file\n",
            "from research.runtime.hashing import canonical_sha256, sha256_file\n" + IMPORT,
        ).replace(call, f"_publish_atomic_directory(staging, {destination}, overwrite=False)")
        current = project / relative
        if current.read_text(encoding="utf8") != expected:
            raise ValueError(f"change is not the audited publish-only patch: {name}")
        approved[f"experiments/{name}"] = {
            "before": {hashlib.sha256(text.encode()).hexdigest() for text in (old, old.replace("\n", "\r\n"))},
            "after": sha256_file(current),
        }
        if name == "fiqa_continuous_calibration.py":
            # The audited production checkout had mixed LF/CRLF. Reversing the
            # three edits byte-for-byte reproduced this receipt hash; normalized
            # content was also checked against the pinned Git blob above.
            approved[f"experiments/{name}"]["before"].add(
                "02e332b730cd1434ad6585d16d217aa6281b9cf4da3163571623373a6e1dfccd")
    research = project / "research"
    modules = (set((research / "calibration").glob("*.py"))
               | set((research / "experiments").glob("*fiqa*.py"))
               | set((research / "experiments").glob("*saliency*.py"))
               | {research / "experiments/calibration_matrix.py", research / "evaluation/cluster_bootstrap.py",
                  research / "evaluation/metrics.py"})
    current = {p.relative_to(research).as_posix(): sha256_file(p) for p in modules}
    return approved, current


def revised_spec(spec, approved, current):
    recorded = {key.replace("\\", "/"): value for key, value in spec["implementation"].items()}
    if set(recorded) != set(current):
        raise ValueError("implementation inventory differs")
    for key, digest in recorded.items():
        allowed = {current[key]} if key not in approved else approved[key]["before"] | {approved[key]["after"]}
        if digest not in allowed:
            raise ValueError(f"unaudited implementation change: {key}")
    updated = copy.deepcopy(spec)
    updated["implementation"] = {key: current[key.replace("\\", "/")] for key in spec["implementation"]}
    return updated


def verify_result(directory, spec):
    result = _load_result(directory)
    manifest = result["manifest"]
    core = {key: value for key, value in manifest.items() if key not in ("result_uid", "status", "files")}
    family = spec["family"]
    tables = FIQA_TABLES if family == "fiqa" else SALIENCY_TABLES
    if set(manifest["files"]) != {f"{name}.csv" for name in tables}:
        raise ValueError("result table inventory mismatch")
    prefix, kind = {"fiqa": ("fiqa-continuous-", "fiqa_continuous_calibration"),
                    "saliency": ("saliency-incremental-", "saliency_incremental_calibration")}[family]
    if manifest["result_uid"] != prefix + canonical_sha256(core)[:24] or manifest["artifact_type"] != kind:
        raise ValueError("result UID/type mismatch")
    hashes = {"condition": "condition_manifest_sha256", "fiqa": "fiqa_manifest_sha256"}
    if family == "saliency":
        hashes.update(saliency="saliency_manifest_sha256", faithfulness="faithfulness_manifest_sha256")
    if any(manifest[field] != spec["input_hashes"][key] for key, field in hashes.items()):
        raise ValueError("result input mismatch")
    settings = manifest["settings"]
    if (settings["partition_seeds"] != [spec["seed"]]
            or set(result["method_summary"].partition_seed) != {spec["seed"]}
            or any(settings[key] != value for key, value in spec["common"].items())):
        raise ValueError("result settings mismatch")
    implementation = {Path(key.replace("\\", "/")).name: digest for key, digest in spec["implementation"].items()}
    # saliency_faithfulness.py is present in result provenance, outside the old
    # runner inventory; require the unchanged on-disk implementation too.
    implementation["saliency_faithfulness.py"] = sha256_file(PROJECT / "research/evaluation/saliency_faithfulness.py")
    if any(implementation.get(name) != digest for name, digest in manifest["implementation_sha256"].items()):
        raise ValueError("result implementation mismatch")
    if manifest["versions"] != spec["versions"]:
        raise ValueError("result runtime mismatch")
    if family == "fiqa" and any(settings[key] != spec[key] for key in ("minimum_group_non_mated", "shrinkage_strength")):
        raise ValueError("FIQA settings mismatch")
    return result


def recover(job_root, *, apply=False):
    root = Path(job_root).resolve()
    approved, current = audit_sources(PROJECT)
    jobs, pending, before = [], [], {}
    for path in sorted(root.glob("job-*/receipt.json")):
        receipt = json.loads(path.read_text(encoding="utf8"))
        spec = receipt["spec"]
        if path.parent.name != "job-" + canonical_sha256(spec)[:24]:
            raise ValueError("receipt job ID mismatch")
        updated = revised_spec(spec, approved, current)
        destination = Path(receipt["result_dir"]).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("result path escapes jobs root")
        result_spec = spec
        if "publish_only_reuse" in receipt:
            provenance = receipt["publish_only_reuse"]
            source_path = Path(provenance["source_receipt"]).resolve()
            if not source_path.is_relative_to(root):
                raise ValueError("reuse source escapes jobs root")
            source = json.loads(source_path.read_text(encoding="utf8"))
            result_spec = source["spec"]
            if (canonical_sha256(result_spec) != provenance["source_spec_sha256"]
                    or revised_spec(result_spec, approved, current) != spec
                    or source["result_dir"] != receipt["result_dir"]
                    or source["result_manifest_sha256"] != receipt["result_manifest_sha256"]):
                raise ValueError("publish-only reuse provenance mismatch")
        result = verify_result(destination, result_spec)
        if canonical_sha256(result["manifest"]) != receipt["result_manifest_sha256"]:
            raise ValueError("receipt result hash mismatch")
        before[str(path)] = sha256_file(path)
        jobs.append((path.parent, spec, updated, destination, result["manifest"]))
    # A staged result without a receipt must reconstruct the EXACT job ID from
    # the opposite family's checked spec, including settings/input/code hashes.
    for job in sorted(root.glob("job-*")):
        if (job / "receipt.json").exists():
            continue
        candidates = list(job.glob(".staging-*/manifest.json"))
        if len(candidates) != 1:
            raise ValueError(f"ambiguous/incomplete staging: {job}")
        staging = candidates[0].parent
        matched = []
        for _, peer, _, _, _ in jobs:
            spec = copy.deepcopy(peer)
            spec["family"] = "saliency" if peer["family"] == "fiqa" else "fiqa"
            if job.name == "job-" + canonical_sha256(spec)[:24]:
                matched.append(spec)
        if len(matched) != 1:
            raise ValueError("staging has no unique matching family spec")
        spec = matched[0]
        result = verify_result(staging, spec)
        destination = job / result["manifest"]["result_uid"]
        if destination.exists():
            raise FileExistsError(destination)
        pending.append((staging, destination, spec, result["manifest"]))
        jobs.append((job, spec, revised_spec(spec, approved, current), destination, result["manifest"]))
    audit = dict(base_commit=BASE_COMMIT, recovery_script_sha256=sha256_file(Path(__file__)),
                 completed_receipts=len(before), recovered_staging=len(pending),
                 aliases=sum(old != new for _, old, new, _, _ in jobs),
                 sources=before, current_implementation=current)
    if apply:
        for staging, destination, spec, manifest in pending:
            # Paths were resolved under the explicitly named jobs root above.
            if not staging.resolve().is_relative_to(root) or not destination.resolve().is_relative_to(root):
                raise ValueError("publish path escapes jobs root")
            _publish_atomic_directory(staging, destination, overwrite=False)
            _atomic_json(destination.parent / "receipt.json", dict(spec=spec, result_dir=str(destination),
                         result_manifest_sha256=canonical_sha256(manifest)))
        for old_job, old, new, destination, manifest in jobs:
            if old == new:
                continue
            receipt = dict(spec=new, result_dir=str(destination), result_manifest_sha256=canonical_sha256(manifest),
                           publish_only_reuse=dict(source_receipt=str(old_job / "receipt.json"),
                                                  source_spec_sha256=canonical_sha256(old), base_commit=BASE_COMMIT,
                                                  recovery_script_sha256=audit["recovery_script_sha256"]))
            path = root / ("job-" + canonical_sha256(new)[:24]) / "receipt.json"
            if path.exists():
                if json.loads(path.read_text(encoding="utf8")) != receipt:
                    raise ValueError("existing migrated receipt differs")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_json(path, receipt)
        if any(sha256_file(Path(path)) != digest for path, digest in before.items()):
            raise ValueError("original receipts changed")
        audit_path = root.parent / "publish_recovery" / ("recovery-" + canonical_sha256(audit)[:24] + ".json")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(audit_path, audit)
        audit["audit_path"] = str(audit_path)
    return {key: value for key, value in audit.items() if key not in ("sources", "current_implementation")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, default=PROJECT / "results/calibration/matrix/jobs")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recover(args.job_root, apply=args.apply), indent=2), flush=True)
