"""Within-run contrasts for the control study. Rates are fractions; deltas are percentage points."""

import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml


def verify_alignment_ranks(input_index):
    """Check native-cohort Rank-1 for the two released single-seed checkpoints."""
    import json
    from threadpoolctl import threadpool_limits
    from research.experiments.facex_inputs import read_inputs
    from research.experiments.facex_workflow import evaluate_rank
    from research.runtime.hashing import sha256_file

    expected = {"arcface-17aec48d7fb653e5ee0b": ("scface", {False: (1020, 1200)}),
                "arcface-d9e71f85b07b1e7540f4": ("tinyface", {False: (2297, 3728), True: (2071, 3728)})}
    index = json.loads(Path(input_index).read_text(encoding="utf-8"))
    verified = {}
    for job, entry in index["inputs"].items():
        alias, dataset = job.split("/")
        uid = index["identity"]["models"][alias]
        if uid not in expected:
            continue
        native, counts = expected[uid]
        if dataset != native:
            raise ValueError(f"Alignment control {alias} cannot be evaluated on {dataset}")
        folder = Path(entry["path"])
        manifest = folder / "manifest.json"
        if sha256_file(manifest) != entry["manifest_sha256"]:
            raise ValueError("Alignment control input no longer matches the index")
        _, population, vectors, _, _ = read_inputs(folder, expected_uid=uid,
            expected_population_hash=index["identity"]["populations"][dataset])
        with threadpool_limits(limits=2):
            scores, _, _ = evaluate_rank(population, vectors, {"origin_1024": ("origin", None, {"payload_bytes": 4096})}, dataset)
        rows = scores.loc[scores["rank"].eq(1) & scores.distance.eq("all")]
        actual = {r.distractors: (r.success_count, r.probe_count) for r in rows.itertuples()}
        if actual != counts:
            raise ValueError(f"{dataset} control differs from the original diagnostic: {actual} vs {counts}")
        verified[alias] = actual
    return verified


def write_survface_config(baseline_yaml, control_export, registry, output):
    """Register a completed control and select the matching baseline under identical evaluation settings."""
    from research.embeddings.facex_zoo import _read_validated_export, register_facex_export
    from research.embeddings.manifests import read_model_spec
    from research.experiments.facex_workflow import _config
    from research.runtime.storage import artifact_lock

    cfg = _config(baseline_yaml)
    alias = "survface_arcface_s8972"
    baseline = cfg["models"][alias]
    spec = read_model_spec(Path(cfg["project_root"]) / baseline["spec"])
    _, reference, _ = _read_validated_export(spec.checkpoint.path)
    _, control, _ = _read_validated_export(control_export)
    if (control["head"] != "arcface" or control["dataset"] != "survface" or control["seed"] != 8972
            or any(control[k] != reference[k] for k in ("head", "dataset", "seed", "parent_hash", "epochs",
                                                         "preprocessing", "protocol_record"))
            or reference["source_config"]["head"]["margin_am"] != 0):
        raise ValueError("Control model or reference lineage differs")
    trained = control["source_config"]["head"]
    expected_margin = 1 - math.cos(0.6)
    if (trained["margin_arc"] != 0.6 or trained["margin_am"] != round(expected_margin, 4)
            or control["execution_settings"]["heads"]["arcface"]["margin_am"] != expected_margin):
        raise ValueError("Control has a different ArcFace margin")
    # Check the canonical scientific configuration independently of the output paths.
    def invariant(meta):
        cfg = meta["source_config"].copy()
        cfg.pop("config_hash", None)
        cfg["head"] = {k: v for k, v in cfg["head"].items() if k != "margin_am"}
        return cfg
    if invariant(control) != invariant(reference):
        raise ValueError("Additional model or training settings differ between controls")
    if control["execution_settings"]["augmentation"] != reference["execution_settings"]["augmentation"]:
        raise ValueError("Training augmentation differs between controls")
    path = register_facex_export(control_export, registry)
    cfg["datasets"] = ["survface"]
    cfg["models"] = {alias: {**baseline, "evaluation_datasets": ["survface"]},
                     "survface_arcface_margin_s8972": dict(training_dataset="survface", evaluation_datasets=["survface"],
                                                           spec=str(path))}
    content = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with artifact_lock(output):
        if output.exists():
            if output.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"Different config already exists: {output}")
            return output
        with TemporaryDirectory(dir=output.parent) as staging:
            pending = Path(staging) / output.name
            pending.write_text(content, encoding="utf-8")
            pending.replace(output)
    return output


def control_tables(result):
    metrics = result["metrics"]
    if result["manifest"].get("study_stage") == "calibration_size":
        return {}
    tables = {}
    rank = metrics.loc[metrics.dataset.isin(["scface", "tinyface"])].copy()
    if not rank.empty:
        rank = rank.loc[rank["rank"].eq(1) & rank.distance.eq("all")]
        keys = ["dataset", "model_uid", "distractors", "probe_count"]
        origin = rank.loc[rank.compression_profile.eq("origin_1024"), keys + ["recognition_rate"]]
        rank = rank.merge(origin.rename(columns={"recognition_rate": "origin_rank1"}), on=keys, validate="many_to_one")
        rank["rank1_change_pp"] = 100 * (rank.recognition_rate - rank.origin_rank1)
        tables["Rank-1 · Origin / compression"] = rank[["dataset", "model_alias", "compression_profile", "distractors",
                                                         "success_count", "probe_count", "origin_rank1", "recognition_rate", "rank1_change_pp"]]
    surv = metrics.loc[metrics.dataset.eq("survface")]
    if not surv.empty:
        keys = ["model_uid", "model_alias", "target_fpir", "mated_count", "non_mated_count"]
        g = surv.loc[surv.method.eq("global_safe")].copy()
        origin = g.loc[g.compression_profile.eq("origin_1024")]
        diagnostic = ["origin_test_roc_tpir_at_20", "origin_test_roc_fpir", "unknown_cosine_ge_095_count"]
        if set(diagnostic) <= set(origin):
            tables["SurvFace · Origin test ROC"] = origin[keys + diagnostic]
        g = g.merge(origin[keys + ["tpir"]].rename(columns={"tpir": "origin_global_tpir"}), on=keys, validate="many_to_one")
        g["compression_change_pp"] = 100 * (g.tpir - g.origin_global_tpir)
        pair_keys = keys + ["compression_profile", "score_space"]
        c = surv.loc[surv.method.eq("continuous_fiqa"), pair_keys + ["tpir", "realized_fpir", "target_met_on_test"]]
        both = g.merge(c, on=pair_keys, suffixes=("_global", "_continuous"), validate="one_to_one")
        both["calibration_change_pp"] = 100 * (both.tpir_continuous - both.tpir_global)
        both["both_target_met"] = both.target_met_on_test_global.eq(True) & both.target_met_on_test_continuous.eq(True)
        tables["SurvFace · compression / calibration"] = both[pair_keys + ["origin_global_tpir", "tpir_global",
            "compression_change_pp", "tpir_continuous", "calibration_change_pp", "realized_fpir_global",
            "realized_fpir_continuous", "both_target_met"]]
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=write_survface_config.__doc__)
    parser.add_argument("control_export", type=Path)
    parser.add_argument("--baseline-yaml", default="configs/experiments/facex_workflow.yaml")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(write_survface_config(args.baseline_yaml, args.control_export, args.registry, args.output))

