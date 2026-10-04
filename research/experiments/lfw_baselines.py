"""Explicit checkpoint conditions for paired LFW experiments.

An alias is a comparison label, not an FR family. A fine-tuned checkpoint gets a
new UID, source directory, PQ fit and calibration; its parent is never replaced.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import re

import pandas as pd
import yaml

from research.embeddings.base import CheckpointProvenance, ModelSpec, PreprocessingSpec
from research.experiments.calibration_matrix import MODEL_UIDS
from research.runtime.hashing import sha256_file


def validate_aliases(config):
    models = config["inputs"]["models"]
    if not models or set(models) != set(config["source_runs"]):
        raise ValueError("input models and source_runs must define the same baseline matrix")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name) for name in models):
        raise ValueError("baseline aliases must be safe lowercase identifiers")
    paths = [str(Path(p).resolve()).casefold() for p in config["source_runs"].values()]
    if len(paths) != len(set(paths)):
        raise ValueError("baseline source paths must be distinct")
    for alias, source in models.items():
        kind = source.get("kind", "pretrained")
        if kind not in ("pretrained", "fine_tuned"):
            raise ValueError("baseline kind must be pretrained or fine_tuned")
        if alias not in MODEL_UIDS and not source.get("expected_sha256"):
            raise ValueError("additional baselines require an explicit expected_sha256")
        if "parent_baseline" in source and (source["parent_baseline"] not in models or source["parent_baseline"] == alias):
            raise ValueError("parent baseline must be a distinct retained condition")
        if kind == "fine_tuned":
            for field in ("source_url", "training_dataset", "parent_baseline", "fine_tuning_data",
                          "lfw_identity_overlap", "overlap_evidence", "selection_evidence"):
                if not isinstance(source.get(field), str) or not source[field].strip():
                    raise ValueError(f"fine-tuned baseline requires {field}")
            if source["parent_baseline"] not in models or source["parent_baseline"] == alias:
                raise ValueError("fine-tuned baseline must retain a distinct parent condition")
            if source["lfw_identity_overlap"] not in ("unknown", "disjoint", "overlap"):
                raise ValueError("lfw_identity_overlap must be unknown, disjoint or overlap")
            if source["lfw_identity_overlap"] == "overlap":
                raise ValueError("fine-tuning on LFW evaluation identities invalidates this experiment")
            if source.get("evaluation_used_for_training_or_selection") is not False:
                raise ValueError("declare evaluation_used_for_training_or_selection=false with evidence")


def model_specs(root, config, models):
    """Resolve static metadata and verify each condition's pinned checkpoint."""
    root = Path(root).resolve()
    validate_aliases(config)
    profiles = yaml.safe_load((root / config["inputs"]["model_profiles_config"]).read_text(encoding="utf8"))
    result = {}
    for alias in models:
        source = config["inputs"]["models"][alias]
        profile = profiles["models"]["profiles"][source["profile"]]
        prep = source.get("preprocessing", profile["preprocessing"])
        checkpoint = CheckpointProvenance.from_file(root / source["checkpoint"], source_url=(
            source.get("source_url") or profile.get("checkpoint_source_url")
            or profile.get("checkpoint_source_page") or profile["implementation_repository"]))
        if source.get("expected_sha256") and checkpoint.sha256 != source["expected_sha256"]:
            raise ValueError(f"checkpoint SHA-256 mismatch: {alias}")
        spec = ModelSpec(family=profile["family"], architecture=profile["architecture"],
            training_dataset=source.get("training_dataset", profile["training_dataset"]),
            implementation_repository=profile["implementation_repository"], checkpoint=checkpoint,
            preprocessing=PreprocessingSpec(input_height=prep["input_size"][0], input_width=prep["input_size"][1],
                source_color_order=profiles["aligned_crops"]["source_color_order"], model_color_order=prep["model_color_order"],
                channel_mean=tuple(prep["mean"]), channel_std=tuple(prep["std"])),
            target_layer=profile["target_layer"], embedding_dim=profile["embedding_dim"], module_factory=profile["loader_factory"])
        if alias in MODEL_UIDS and spec.model_uid != MODEL_UIDS[alias]:
            raise ValueError(f"original baseline is pinned; register a new alias for changed weights: {alias}")
        result[alias] = spec
    # Metadata relabeling of the same file is not a new recognition baseline.
    digests = [s.checkpoint.sha256 for s in result.values()]
    if len(digests) != len(set(digests)):
        raise ValueError("duplicate checkpoint bytes across baseline conditions")
    for alias, spec in result.items():
        entry = config["inputs"]["models"][alias]
        if entry.get("kind") == "fine_tuned" and entry["parent_baseline"] in result:
            parent = result[entry["parent_baseline"]]
            if (spec.family, spec.architecture) != (parent.family, parent.architecture):
                raise ValueError("fine-tuned condition must retain its parent backbone architecture")
    return result


def baseline_catalog(config, specs):
    rows = []
    for alias, spec in specs.items():
        source = config["inputs"]["models"][alias]
        rows.append(dict(model=alias, model_uid=spec.model_uid, family=spec.family,
            architecture=spec.architecture, kind=source.get("kind", "pretrained"),
            checkpoint_sha256=spec.checkpoint.sha256, source_url=spec.checkpoint.source_url,
            preprocess_hash=spec.preprocessing.preprocess_hash, training_dataset=spec.training_dataset,
            parent_baseline=source.get("parent_baseline", "none"), fine_tuning_data=source.get("fine_tuning_data", "not_applicable"),
            lfw_identity_overlap=source.get("lfw_identity_overlap", "unknown"),
            overlap_evidence=source.get("overlap_evidence", "unverified"),
            selection_evidence=source.get("selection_evidence", "pretrained checkpoint; training overlap unverified"),
            overlap_status_is_declaration=True, unseen_identity_claim_supported=False))
    return pd.DataFrame(rows)


def register_baseline(project_root, *, config_path, output_config, alias, profile, checkpoint,
                      source_url, training_dataset, kind, parent_baseline=None, fine_tuning_data=None,
                      lfw_identity_overlap="unknown", overlap_evidence="unverified",
                      evaluation_used_for_training_or_selection=None, selection_evidence=None,
                      preprocessing=None):
    """Write a NEW config retaining all original conditions; never tune or download."""
    root = Path(project_root).resolve()
    config = deepcopy(yaml.safe_load((root / config_path).read_text(encoding="utf8")))
    if alias in config["inputs"]["models"]:
        raise ValueError("new baseline alias required; existing conditions are immutable")
    path = (root / checkpoint).resolve()
    entry = dict(profile=profile, checkpoint=path.as_posix(), expected_sha256=sha256_file(path),
                 kind=kind, source_url=source_url, training_dataset=training_dataset,
                 lfw_identity_overlap=lfw_identity_overlap, overlap_evidence=overlap_evidence)
    if parent_baseline is not None:
        entry["parent_baseline"] = parent_baseline
    if kind == "fine_tuned":
        entry.update(fine_tuning_data=fine_tuning_data,
            evaluation_used_for_training_or_selection=evaluation_used_for_training_or_selection,
            selection_evidence=selection_evidence)
    if preprocessing is not None:
        entry["preprocessing"] = preprocessing
    config["inputs"]["models"][alias] = entry
    source_root = Path(config["inputs"]["registry_root"]).parent / "sources"
    config["source_runs"][alias] = (source_root / alias).as_posix()
    specs = model_specs(root, config, tuple(config["inputs"]["models"]))
    config["source_runs"][alias] = (source_root / f"{alias}-{specs[alias].model_uid}").as_posix()
    # A new condition has its own shard store; adding a baseline cannot prune old shards.
    config["resize_inputs"]["checkpoint_file"] = (source_root.parent / f"extraction-{alias}-{specs[alias].checkpoint.sha256[:16]}.sqlite3").as_posix()
    config["output_root"] = (Path(config["output_root"]) / f"with-{alias}").as_posix()
    destination = root / output_config
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    return destination


def export_step4_profile(project_root, *, config_path, alias, output_config):
    """Export the same checkpoint metadata for the independent SurvFace 1:N run.

    Only recognizer metadata is shared. LFW pairs, PQ fits and thresholds are not
    transplanted into the 1:N protocol.
    """
    root = Path(project_root).resolve()
    config = yaml.safe_load((root / config_path).read_text(encoding="utf8"))
    spec = model_specs(root, config, (alias,))[alias]
    entry = config["inputs"]["models"][alias]
    base = yaml.safe_load((root / config["inputs"]["model_profiles_config"]).read_text(encoding="utf8"))
    profile = deepcopy(base["models"]["profiles"][entry["profile"]])
    profile.update(training_dataset=spec.training_dataset, checkpoint_path=spec.checkpoint.path,
        expected_checkpoint_sha256=spec.checkpoint.sha256, checkpoint_source_url=spec.checkpoint.source_url,
        baseline_provenance=deepcopy(entry))
    if "preprocessing" in entry:
        profile["preprocessing"] = entry["preprocessing"]
    base["models"]["profiles"][alias] = profile
    base["execution"]["model_profile"] = alias
    destination = root / output_config
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf8") as stream:
        yaml.safe_dump(base, stream, allow_unicode=True, sort_keys=False)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/experiments/lfw_pair_verification.yaml")
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--step4-output-config", help="optional independent 1:N recognizer profile export")
    for name in ("alias", "profile", "checkpoint", "source-url", "training-dataset"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--kind", choices=("pretrained", "fine_tuned"), required=True)
    parser.add_argument("--parent-baseline")
    parser.add_argument("--fine-tuning-data")
    parser.add_argument("--lfw-identity-overlap", choices=("unknown", "disjoint", "overlap"), default="unknown")
    parser.add_argument("--overlap-evidence", default="unverified")
    parser.add_argument("--selection-evidence")
    parser.add_argument("--evaluation-used-for-training-or-selection", action=argparse.BooleanOptionalAction, default=None)
    args = vars(parser.parse_args())
    step4 = args.pop("step4_output_config")
    args["config_path"] = args.pop("config")
    output = register_baseline(**args)
    print(output)
    if step4:
        print(export_step4_profile(args["project_root"], config_path=output,
                                  alias=args["alias"], output_config=step4))


if __name__ == "__main__":
    main()
