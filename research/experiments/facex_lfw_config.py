"""Generate pinned LFW input/evaluation configs from registered FaceX-Zoo exports.

This writes configuration only. It neither trains a recognizer nor extracts images.
"""

import argparse
from pathlib import Path
import re

import yaml

from research.embeddings.manifests import read_model_spec
from research.runtime.hashing import canonical_sha256


def generate_configs(project_root, model_specs, output_dir, *, artifact_root=None):
    project = Path(project_root).resolve()
    if not model_specs or any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", k) for k in model_specs):
        raise ValueError("provide unique lowercase model aliases and registered spec paths")
    specs = {alias: read_model_spec(project / path) for alias, path in model_specs.items()}
    for spec in specs.values():
        if spec.module_factory != "research.embeddings.pytorch.facex_zoo:load_facex_mobilefacenet_checkpoint":
            raise ValueError("FaceX-Zoo export registration is required")
        if spec.preprocessing.source_color_order != "rgb":
            raise ValueError("LFW resize supplies RGB; register source_color_order=rgb")
    dimensions = {spec.embedding_dim for spec in specs.values()}
    if len(dimensions) != 1:
        raise ValueError("one embedding dimension per experiment config is required")
    dimension = dimensions.pop()
    bindings = {alias: spec.model_uid for alias, spec in specs.items()}
    if len(set(bindings.values())) != len(bindings):
        raise ValueError("duplicate checkpoint UIDs under different aliases")
    namespace = "lfw_facex_" + canonical_sha256(bindings)[:20]
    artifacts = Path(artifact_root) if artifact_root else Path("results") / namespace
    source_paths = {alias: (artifacts / "sources" / uid).as_posix() for alias, uid in bindings.items()}
    output = (project / output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"select a new config directory: {output}")
    input_path, pair_path = output / "inputs.yaml", output / "pairs.yaml"
    profiles = [f"pq_{dimension}_m{m}_b8" for m in (128, 64, 32)]
    pair = yaml.safe_load((project / "configs/experiments/lfw_pair_verification.yaml").read_text(encoding="utf8"))
    base_inputs = yaml.safe_load((project / "configs/experiments/lfw_blufr_calibration_resize.yaml").read_text(encoding="utf8"))
    inputs = {key: base_inputs[key] for key in (
        "protocol_file", "protocol_sha256", "protocol_source_url", "image_manifest", "source_kind", "fiqa_variant")}
    shared = dict(embedding_dim=dimension, model_uids=bindings, source_runs=source_paths,
                  pq_profiles=profiles, fiqa_root=(artifacts / "fiqa_scores").as_posix())
    inputs.update(shared)
    inputs["resize_inputs"] = dict(base_inputs["resize_inputs"],
        checkpoint_file=(artifacts / "embedding_work.sqlite3").as_posix(),
        model_specs={alias: (project / path).resolve().as_posix() for alias, path in model_specs.items()})
    pair.update(shared, input_config_path=input_path.as_posix(),
        output_root=(artifacts / "calibration").as_posix(),
        pq_training_policy="outer_fold_unused_identities", people_file="data/raw/LFW/people.csv")
    output.mkdir(parents=True, exist_ok=False)
    for path, config in ((input_path, inputs), (pair_path, pair)):
        path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf8")
    # Detailed inputs remain local even if the caller chooses a new output root.
    artifact_path = (project / artifacts).resolve()
    artifact_path.mkdir(parents=True, exist_ok=True)
    ignore = artifact_path / ".gitignore"
    patterns = "sources/**/*.npy\nsources/**/population.csv\nfiqa_scores/**/*.csv\nembedding_work.sqlite3*\n"
    existing = ignore.read_text(encoding="utf8") if ignore.exists() else ""
    additions = [line for line in patterns.splitlines() if line not in existing.splitlines()]
    if additions:
        ignore.write_text(existing.rstrip() + "\n" + "\n".join(additions) + "\n", encoding="utf8")
    return {"input_config": input_path, "pair_config": pair_path, "model_uids": bindings,
            "embedding_dim": dimension, "artifact_root": artifact_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model", action="append", required=True, help="alias=registered_model_spec.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    entries = [value.split("=", 1) for value in args.model]
    if any(len(entry) != 2 for entry in entries) or len({e[0] for e in entries}) != len(entries):
        parser.error("each --model must be a unique alias=path")
    print(generate_configs(args.project_root, dict(entries), args.output_dir, artifact_root=args.artifact_root))


if __name__ == "__main__":
    main()
