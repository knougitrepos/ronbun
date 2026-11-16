"""Register an explicit completed FaceX selection and create its experiment config."""
import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from research.embeddings.facex_zoo import _read_validated_export, register_facex_export
from research.experiments.facex_workflow import _config, _models
from research.runtime.hashing import sha256_file
from research.runtime.storage import artifact_lock


def write_inventory_config(inventory_path, baseline_yaml, registry, output, *, purpose="local_protocol_baseline",
                           origin_only=True):
    if type(origin_only) is not bool:
        raise ValueError("origin_only must be a boolean")
    inventory_path = Path(inventory_path).resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("schema") not in ("facex-model-inventory-v1", "facex-model-inventory-v2") or inventory.get("status") != "completed":
        raise ValueError("A completed FaceX model inventory is required")
    if purpose not in ("local_protocol_baseline", "diagnostic_control", "legacy_local_control"):
        raise ValueError("This importer cannot certify exact UCFace reproduction")
    legacy = inventory["schema"] == "facex-model-inventory-v1"
    if legacy:
        if purpose != "legacy_local_control":
            raise ValueError("Legacy inventory requires purpose='legacy_local_control'; it is not a verified UCFace baseline")
        scope = dict(purpose=purpose, exact_ucface_reproduction=False, origin_comparison_required=True)
    else:
        selection = inventory["selection"]
        scope = selection.get("reproduction_scope", {})
        if (selection.get("schema") != "facex-selection-v2" or purpose == "legacy_local_control"
                or selection.get("purpose") != purpose or scope.get("purpose") != purpose
                or scope.get("exact_ucface_reproduction") is not False
                or scope.get("origin_comparison_required") is not True):
            raise ValueError("Inventory purpose/claim differs; UCFace reproduction is not certified")
    rows = inventory["models"]
    planned = inventory["selection"]["rows"]
    if not rows or len(rows) != len(planned) or len({r["run_id"] for r in rows}) != len(rows):
        raise ValueError("Empty, partial or duplicate model inventory")
    expected = {r["run_id"]: r for r in planned}
    models, datasets = {}, []
    # Validate every entry before registering any model.
    for row in rows:
        if row["run_id"] not in expected or any(row.get(k) != v for k, v in expected[row["run_id"]].items()):
            raise ValueError("Model inventory differs from the selected plan")
        if row["recipe"] == "arcface_continuous_margin" and purpose == "local_protocol_baseline":
            raise ValueError("continuous_margin must remain a diagnostic control")
        if row.get("completed") is not True or sha256_file(row["export"]) != row["sha256"]:
            raise ValueError("Incomplete or modified model export")
        path, meta, _ = _read_validated_export(row["export"])
        ds = row["training_dataset"]
        if (ds != row["dataset"] or ds != meta["dataset"] or row["head"] != meta["head"]
                or row["seed"] != meta["seed"] or row["source_sha256"] != meta["source_sha256"]
                or row["evaluation_datasets"] != [ds] or ds not in ("scface", "tinyface", "survface")):
            raise ValueError("Inventory lineage or evaluation scope differs")
        source = meta.get("source_config", {})
        if (row["head"] == "arcface" and purpose == "local_protocol_baseline"
                and source.get("head", {}).get("margin_am") != 0):
            raise ValueError("Baseline export must retain ArcFace margin_am=0")
        if not legacy:
            declared = scope.get("populations", {}).get(ds, {}).get("local_training", [])
            if (len(declared) != 2 or declared[0] != source.get("dataset", {}).get("num_classes")
                    or declared[1] != meta.get("protocol_record", {}).get("images")):
                raise ValueError("Inventory training population differs from the checkpoint")
        if meta.get("input_geometry"):
            geometry = meta["input_geometry"]
            local = (path.parent / geometry["file"]).resolve()
            if not local.is_relative_to(path.parent.resolve()) or sha256_file(local) != geometry["sha256"]:
                raise ValueError("Geometry is outside the export or has changed")
        if ds not in datasets:
            datasets.append(ds)
    cfg = copy.deepcopy(_config(baseline_yaml))
    for row in rows:
        spec = register_facex_export(row["export"], registry)
        models[row["run_id"]] = dict(spec=str(Path(spec).resolve()), training_dataset=row["training_dataset"],
                                    evaluation_datasets=row["evaluation_datasets"])
    cfg.update(models=models, datasets=datasets,
               study={"stage": "baseline"},
               model_scope=copy.deepcopy(scope),
               model_inventory=dict(path=str(inventory_path), sha256=sha256_file(inventory_path)))
    # Establish uncompressed FR performance before spending time on PCA/PQ.
    # Existing experiment YAMLs/results are never edited by this generator.
    if origin_only:
        cfg["compression"] = {ds: [{"family": "origin"}] for ds in datasets}
    cfg["model_scope"]["evaluation_phase"] = "origin_reference" if origin_only else "compression_and_calibration"
    # Same dataset/protocol checks as 02, before publishing a usable config.
    from research.protocols.facex_handoff import load_handoff
    _models(cfg, load_handoff(cfg["handoff"]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    with artifact_lock(output):
        if output.exists():
            if output.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"Different experiment config exists: {output}")
        else:
            with TemporaryDirectory(dir=output.parent) as temp:
                pending = Path(temp) / output.name
                pending.write_text(content, encoding="utf-8")
                pending.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--purpose", choices=("local_protocol_baseline", "diagnostic_control", "legacy_local_control"),
                        default="local_protocol_baseline")
    parser.add_argument("--with-compression", action="store_true", help="Keep baseline YAML compression profiles after reviewing Origin results")
    args = parser.parse_args()
    print(write_inventory_config(args.inventory, args.baseline, args.registry, args.output, purpose=args.purpose,
                                 origin_only=not args.with_compression))


if __name__ == "__main__":
    main()
