"""Derive common CI from one explicit completed, SHA-verified ledger condition.

No search, threshold fitting, or modification of source runs. Output is a new
content-addressed directory; an existing destination is rejected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.evaluation.cluster_bootstrap import CLUSTER_CI_CONTRACT  # noqa: E402
from research.evaluation.retrieval_ledger import load_retrieval_ledger_manifest  # noqa: E402
from research.runtime.hashing import canonical_sha256, sha256_file  # noqa: E402
from scripts.generate_step4_compact_summaries import summarize_retrieval  # noqa: E402


def postprocess(ledger, *, expected_sha256, condition_id, output_root, chunksize=100_000):
    ledger = Path(ledger).resolve()
    if sha256_file(ledger) != expected_sha256:
        raise ValueError("source ledger manifest SHA-256 mismatch")
    source = load_retrieval_ledger_manifest(ledger)
    selected = [c for c in source["conditions"] if c["condition_id"] == condition_id]
    if len(selected) != 1:
        raise ValueError("select exactly one existing condition_id")
    condition = selected[0]["condition"]
    if condition.get("evaluation_split") != "test":
        raise ValueError("common CI postprocessing requires a test condition")
    if condition.get("threshold_source_split") != "calibration":
        raise ValueError("thresholds must come from calibration, not test")
    implementation = [
        Path(__file__),
        PROJECT_ROOT / "research/evaluation/cluster_bootstrap.py",
        PROJECT_ROOT / "research/evaluation/retrieval_ledger.py",
        PROJECT_ROOT / "research/evaluation/metrics.py",
        PROJECT_ROOT / "scripts/generate_step4_compact_summaries.py",
    ]
    manifest = {
        "artifact_type": "common_tpir_identity_cluster_ci", "schema_version": 1,
        "contract": CLUSTER_CI_CONTRACT, "source_manifest": str(ledger),
        "source_manifest_sha256": expected_sha256,
        "condition_id": condition_id, "condition": condition,
        "implementation_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p)
                                  for p in implementation},
        "resamples": 2000, "seed": 42, "confidence_level": .95,
        "thresholds_refitted": False, "gallery_resampled": False,
        "multiple_comparison_adjustment": "none", "fpir_unit": "query",
        "interpretation": "exploratory_fixed_threshold_fixed_gallery",
    }
    uid = "common-ci-" + canonical_sha256(manifest)[:24]
    destination = Path(output_root).resolve() / uid
    if destination.exists():
        raise FileExistsError(destination)
    summary, row_count = summarize_retrieval(
        ledger, chunksize=chunksize, condition_ids=(condition_id,),
    )
    if sha256_file(ledger) != expected_sha256:
        raise ValueError("source manifest changed during evaluation")
    if not summary["tpir_cluster_ci_status"].eq("ok").all():
        raise ValueError("selected condition lacks sufficient mated identity labels")
    destination.mkdir(parents=True, exist_ok=False)
    summary.to_csv(destination / "retrieval_summary.csv", index=False, float_format="%.12g")
    manifest.update(
        artifact_uid=uid, status="completed", logical_row_count=row_count,
        summary_sha256=sha256_file(destination / "retrieval_summary.csv"),
    )
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    print(postprocess(args.ledger, expected_sha256=args.expected_sha256,
                      condition_id=args.condition_id, output_root=args.output_root))


if __name__ == "__main__":
    main()
