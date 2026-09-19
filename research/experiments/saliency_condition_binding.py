"""Bind verified origin saliency to another PQ budget of the same frozen cohort."""

import json
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from research.explainability.gradcam.artifacts import _publish_atomic_directory
from research.experiments.saliency_incremental_calibration import (
    assess_incremental_gate, load_saliency_incremental_inputs,
)
from research.runtime.hashing import canonical_sha256, sha256_file


def bind_saliency_condition(source, target, saliency_directory, faithfulness_directory, output_root):
    """Copy compact evidence with explicit derivation provenance; never change sources.

    Only the PQ codec/scores may differ. Same immutable run, prepared population,
    protocol seed, query membership and alignment imply the same origin gallery
    and origin-top1 Grad-CAM targets. No compressed score enters these features.
    """
    for key in ("dataset_id", "source_run_id", "source_run_manifest_sha256", "model_uid",
                "source_freeze_manifest_sha256", "selected_manifest_sha256",
                "prepared_population_manifest_sha256", "aligned_bundle_manifest_sha256",
                "origin_embedding_artifact_uid", "extraction_uid", "calibration_seed",
                "protocol_uid", "calibration_protocol", "metric_contract", "search_mode", "top_k"):
        if key not in source.manifest or source.manifest[key] != target.manifest.get(key):
            raise ValueError(f"saliency reuse cohort mismatch: {key}")
    if source.manifest["search_mode"] != "pq_adc_exhaustive":
        raise ValueError("cross-budget binding requires PQ ADC")
    cols = ["sample_id", "identity_id", "evaluation_split", "aligned_content_sha256", "is_mated"]
    for split in ("calibration", "test"):
        left, right = getattr(source, split), getattr(target, split)
        pd.testing.assert_frame_equal(left[cols].reset_index(drop=True), right[cols].reset_index(drop=True))
    inputs = load_saliency_incremental_inputs(source, saliency_directory, faithfulness_directory)
    if not assess_incremental_gate(source, inputs)["comparison_enabled"]:
        raise ValueError("source saliency binding is not calibration-ready")
    spec = dict(artifact_type="saliency_condition_binding", schema_version=1,
                source_condition_sha256=canonical_sha256(source.manifest),
                target_condition_sha256=canonical_sha256(target.manifest),
                source_saliency_sha256=canonical_sha256(inputs["saliency_manifest"]),
                source_faithfulness_sha256=canonical_sha256(inputs["faithfulness_manifest"]),
                implementation_sha256=sha256_file(Path(__file__)),
                reason="same_frozen_origin_gallery_queries_and_targets_different_pq_budget")
    destination = Path(output_root) / ("saliency-binding-" + canonical_sha256(spec)[:24])
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf8"))
        if manifest != dict(spec=spec, status="completed"):
            raise ValueError("saliency binding specification mismatch")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / (".staging-" + uuid4().hex)
        staging.mkdir()
        for name, original, key in (("saliency", saliency_directory, "saliency_manifest"),
                                     ("faithfulness", faithfulness_directory, "faithfulness_manifest")):
            folder = staging / name
            folder.mkdir()
            manifest = dict(inputs[key])
            receipts = [manifest["saliency_features"]] if name == "saliency" else manifest["outputs"]
            for receipt in receipts:
                path = Path(original) / receipt["path"]
                copied = folder / receipt["path"]
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, copied)
                if sha256_file(copied) != receipt["sha256"]:
                    raise ValueError("saliency binding copy hash mismatch")
            manifest["condition_manifest_sha256"] = canonical_sha256(target.manifest)
            manifest["condition_binding"] = spec
            (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        (staging / "manifest.json").write_text(json.dumps(dict(spec=spec, status="completed"), indent=2), encoding="utf8")
        checked = load_saliency_incremental_inputs(target, staging / "saliency", staging / "faithfulness")
        if not assess_incremental_gate(target, checked)["comparison_enabled"]:
            raise ValueError("derived saliency binding failed integrity gate")
        # Windows can briefly deny renaming a just-validated directory. Reuse
        # bounded retries while retaining staging on failure and existing results.
        _publish_atomic_directory(staging, destination, overwrite=False)
    checked = load_saliency_incremental_inputs(target, destination / "saliency", destination / "faithfulness")
    if not assess_incremental_gate(target, checked)["comparison_enabled"]:
        raise ValueError("completed binding failed integrity gate")
    return dict(directory=destination, saliency_directory=destination / "saliency",
                faithfulness_directory=destination / "faithfulness")
